# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Run a DC-Ops student on one RGB888 frame on the PC: float and int8 (pt2e) simulation.

No ExecuTorch runtime: the float model is ``dcops/student_model.py``'s StudentNHWC loaded from
``<run>/student.pt``; the int8 model is the same EthosUQuantizer pt2e flow as
``dcops/export/export_student.py`` (symmetric per-channel, calibrated on ``<run>/calib.npz``).
The input is what the firmware feeds the NPU: NHWC float = uint8 / 255.

Run with the ExecuTorch 1.1 env (torch + executorch Python only):
    ~/projects/arm-executorch-hack/experiments/blazeface/.venv-et110/bin/python \
        dcops/board/pc_infer.py --run dcops/models/hub/vision/r160_kd --frame frame.bin [--res 160]
Prints one JSON object: {"labels", "res", "float": {"probs", "top3"}, "int8": {...}}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DCOPS = os.path.dirname(HERE)
REPO = os.path.dirname(DCOPS)


def load_frame(path: str, res: int) -> np.ndarray:
    """Load a frame as HxWx3 uint8 (.bin raw RGB888, .npy, or an image file).

    :param path:    Frame file.
    :param res:     Model input side (for raw .bin).
    :returns:       ``(res, res, 3)`` uint8 array.
    :raises ValueError: On a size mismatch.
    """
    if path.endswith(".npy"):
        a = np.load(path)
    elif path.endswith(".bin"):
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size != res * res * 3:
            raise ValueError(f"{path}: {raw.size} bytes, expected {res}x{res}x3")
        a = raw.reshape(res, res, 3)
    else:
        from PIL import Image
        a = np.asarray(Image.open(path).convert("RGB").resize((res, res)))
    if a.shape != (res, res, 3):
        raise ValueError(f"frame shape {a.shape}, expected {(res, res, 3)}")
    return a.astype(np.uint8)


def topk(probs: np.ndarray, labels: list[str], k: int = 3) -> list[dict]:
    """:returns: Top-k ``{idx, label, prob}``."""
    order = np.argsort(-probs)[:k]
    return [{"idx": int(i), "label": labels[i] if i < len(labels) else f"#{i}",
             "prob": float(probs[i])} for i in order]


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder: student.pt, metrics.json, calib.npz, labels.txt")
    ap.add_argument("--frame", required=True, action="append", help="frame file(s)")
    ap.add_argument("--res", type=int, help="input side (default: metrics.json args.res)")
    ap.add_argument("--no-int8", action="store_true", help="float only")
    ap.add_argument("--calib-limit", type=int, default=64, help="calibration images used")
    a = ap.parse_args()

    warnings.filterwarnings("ignore")
    import torch
    sys.path.insert(0, DCOPS)
    from student_model import StudentNHWC

    run = a.run
    metrics = json.load(open(os.path.join(run, "metrics.json")))
    res = a.res or int(metrics["args"]["res"])
    labels = [l.strip() for l in open(os.path.join(run, "labels.txt")) if l.strip()]
    model = StudentNHWC(len(labels), res, pretrained=False).eval()
    model.load_state_dict(torch.load(os.path.join(run, "student.pt"), map_location="cpu"))
    frames = [load_frame(f, res) for f in a.frame]
    xs = [torch.from_numpy(f[None]).float() / 255 for f in frames]

    def run_model(m) -> list[np.ndarray]:
        with torch.no_grad():
            return [torch.softmax(m(x).float(), dim=1)[0].numpy() for x in xs]

    result = {"run": os.path.abspath(run), "labels": labels, "res": res, "frames": a.frame,
              "float": [{"probs": p.tolist(), "top3": topk(p, labels)} for p in run_model(model)]}
    if not a.no_int8:
        from executorch.backends.arm.ethosu import EthosUCompileSpec
        from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        c = np.load(os.path.join(run, "calib.npz"))
        calib = [torch.from_numpy(x[None]).float() / 255 for x in c["images"][:a.calib_limit]]
        spec = EthosUCompileSpec(target="ethos-u85-256",
                                 config_ini=os.path.join(REPO, "scripts/vela/ensemble_vela.ini"),
                                 system_config="Ethos_U85_SRAM_MRAM", memory_mode="Shared_Sram",
                                 extra_flags=["--output-format=raw", "--debug-force-regor"])
        g = torch.export.export(model, (calib[0],)).module()
        q = EthosUQuantizer(spec)
        q.set_global(get_symmetric_quantization_config(is_per_channel=True))
        p = prepare_pt2e(g, q)
        with torch.no_grad():
            for x in calib:
                p(x)
        qm = convert_pt2e(p)
        result["int8"] = [{"probs": pr.tolist(), "top3": topk(pr, labels)} for pr in run_model(qm)]
        result["int8_calib_images"] = len(calib)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
