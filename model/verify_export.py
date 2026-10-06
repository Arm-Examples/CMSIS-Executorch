#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Host-side checks for the pico-faces export in model.py.

    python model/verify_export.py --stage float     --seed 3 --class 1 --w 4 --k 4 --out out/pf_float.png
    python model/verify_export.py --stage fakequant --seed 3 --class 1 --w 4 --k 4 --out out/pf_fakequant.png
    python model/verify_export.py --stage tosa
    python model/verify_export.py --check-upstream /path/to/pico-faces
    python model/verify_export.py --compare out/fvp_image.bin out/pf_fakequant.png

--stage float      samples with the float modules of model.py: the reference.
--stage fakequant  samples with the quantize/dequantize graphs create_ai_layer.py
                   produces (convert_pt2e), i.e. what the NPU is expected to compute.
--stage tosa       lowers both methods with the TOSA backend and executes one
                   input each through the TOSA reference model: the integer
                   semantics the Ethos-U implements (tables, rescales, int48
                   accumulators), which the fake-quant graphs do not model. A
                   large deviation from fakequant here means the NPU output will
                   be wrong too; run it before flashing a changed 16-bit graph.
--check-upstream   compares model.py's re-implementation (folded scales, conv
                   patchify, BatchNorm and latent folds) against the original
                   pico-faces modules from a clone of the upstream repository.
--compare          PSNR between two images: PNGs, or the raw 128x128 RGB frame
                   the FVP run writes to out/fvp_image.bin.

The host stages use the firmware's noise generator, so the same seed, class,
guidance and step count give the image the board and the FVP produce (up to
the rounding differences between the fake-quant graphs and the NPU).

Runs itself in the solution's .venv, like create_ai_layer.py.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run_in_venv(required: bool) -> None:
    """Re-run under the project's .venv unless this interpreter already is it.

    Only --compare works without it (numpy and pillow are enough)."""
    venv = ROOT / ".venv"
    if Path(sys.prefix).resolve() == venv.resolve():
        return
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        if not required:
            return
        sys.exit(f"{venv} does not exist. Create it first: ./setup_venv.sh (Linux/macOS) or setup_venv.bat (Windows)")
    sys.exit(subprocess.run([str(python), __file__, *sys.argv[1:]]).returncode)


def save_png(img_u8, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8, "RGB" if img_u8.shape[-1] == 3 else "L").save(path)
    print(f"wrote {path}")


def stage_modules(stage: str, mlops_file: str | None):
    """(dit_step callable, decode callable) for the requested stage."""
    import model as M

    if stage == "float":
        return M.DiTStep.from_checkpoint(), M.Decoder.from_checkpoint()

    sys.path.insert(0, str(ROOT))
    import create_ai_layer as C

    spec = C.compile_spec_from_file(mlops_file) if mlops_file else C.default_compile_spec()
    mods = {m.name: C.quantize_method(m, spec) for m in M.get_methods()}
    return mods["dit_step"], mods["decode"]


def sample(args) -> None:
    import model as M

    dit, dec = stage_modules(args.stage, args.mlops)
    cond = M.Cond.from_checkpoint().cond_table()
    z = M.noise(args.seed, *M.example_inputs()[0].shape[1:3])
    t0 = time.time()
    z0 = M.euler_sample(dit, cond, z, args.cls, args.w, args.k)
    img = M.to_image(dec(z0))
    print(f"{args.stage}: seed {args.seed} class {args.cls} w {args.w} k {args.k}: {time.time() - t0:.1f} s")
    save_png(img, Path(args.out))


def check_tosa(args) -> None:
    """Run both methods through the TOSA reference model and compare with fake-quant and float."""
    import torch
    from executorch.backends.arm.quantizer import TOSAQuantizer
    from executorch.backends.arm.test.runner_utils import TosaReferenceModelDispatch
    from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
    from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    import model as M

    sys.path.insert(0, str(ROOT))
    import create_ai_layer as C

    z = M.noise(args.seed, *M.example_inputs()[0].shape[1:3])
    cond = M.Cond.from_checkpoint().cond_table()
    floats = {"dit_step": M.DiTStep.from_checkpoint(), "decode": M.Decoder.from_checkpoint()}
    worst = 0.0
    for m in M.get_methods():
        spec = TosaCompileSpec("TOSA-1.0+INT+int16" if m.quantization == "a16w8" else "TOSA-1.0+INT")
        cfg = C.quant_config(m.quantization)
        graph = torch.export.export(m.module, m.example_inputs).module()
        C.strip_guards_fn(graph)
        quantizer = TOSAQuantizer(spec)
        quantizer.set_global(cfg)
        prepared = prepare_pt2e(graph, quantizer)
        C.calibrate(prepared, m)
        converted = convert_pt2e(prepared)
        edge = to_edge_transform_and_lower(
            torch.export.export(converted, m.example_inputs),
            partitioner=[TOSAPartitioner(spec)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        program = edge.to_executorch()
        inputs = (z, cond[args.cls, 0][None]) if m.name == "dit_step" else (z,)
        with torch.no_grad():
            fq = converted(*inputs)
            fl = floats[m.name](*inputs)
            with TosaReferenceModelDispatch():
                ref = program.exported_program().module()(*inputs)
        ref = ref[0] if isinstance(ref, (list, tuple)) else ref
        d_fq, d_fl = (ref - fq).abs().max().item(), (ref - fl).abs().max().item()
        worst = max(worst, d_fq / max(fl.abs().max().item(), 1e-6))
        print(f"  {m.name}: refmodel-vs-fakequant max {d_fq:.4f}, refmodel-vs-float max {d_fl:.4f}, "
              f"fakequant-vs-float max {(fq - fl).abs().max().item():.4f}, |float| max {fl.abs().max().item():.2f}")
    ok = worst <= 0.05
    print(f"tosa: worst relative deviation from fakequant {worst:.3f} -> {'OK' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


def check_upstream(clone: str) -> None:
    import torch

    import model as M

    sys.path.insert(0, clone)
    from train.common.sincos import timestep_embedding
    from train.dit.model import build_model
    from train.vae.model import build_vae

    ck = M.load_checkpoint()
    up = build_model(ck.dit_cfg).eval()
    up.load_state_dict(ck.dit)
    vae_ckpt = torch.load(M.checkpoint_dir() / "vae_final.pt", map_location="cpu", weights_only=False)
    up_vae = build_vae(vae_ckpt["cfg"]).eval()
    up_vae.load_state_dict(vae_ckpt["model"])

    ours, cond, dec = M.DiTStep.from_checkpoint(), M.Cond.from_checkpoint(), M.Decoder.from_checkpoint()
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    with torch.no_grad():
        for i in range(8):
            z = torch.randn(1, ours.z_ch, ours.z_hw, ours.z_hw, generator=g) * (1.0 + i / 4)
            t = torch.rand(1, generator=g)
            y = torch.tensor([i % (M.N_CLASSES + 1)])
            c_up = up.t_mlp(timestep_embedding(t, up.t_dim)) + up.y_emb(y)
            c_ours = cond(t, y)
            d_c = (c_up - c_ours).abs().max().item()
            v_up = up(z, t, y)
            v_ours = ours(z, c_ours)
            d_v = (v_up - v_ours).abs().max().item() / max(v_up.abs().max().item(), 1e-6)
            worst = max(worst, d_c, d_v)
            print(f"  sample {i}: |dc| {d_c:.2e}  rel |dv| {d_v:.2e}")
        z = torch.randn(1, ours.z_ch, ours.z_hw, ours.z_hw, generator=g)
        img_up = up_vae.decoder(z * ck.latent_std[None, :, None, None] + ck.latent_mean[None, :, None, None]).clamp(-1, 1)
        img_ours = dec(z)
        d_img = (img_up - img_ours).abs().max().item()
        worst = max(worst, d_img)
        print(f"  decoder: |dimg| {d_img:.2e} (max pixel step 7.8e-3)")
    # The DiT and conditioning must match to float precision; the decoder folds the
    # latent mean into the zero-padded first conv (as upstream does), which shifts
    # border pixels by about one 8-bit step.
    ok = worst <= 2e-2
    print(f"check-upstream: worst deviation {worst:.2e} -> {'OK' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


def load_image(path: str):
    """An image as float HWC RGB: a PNG, or a raw 128x128x3 frame (.bin)."""
    import numpy as np
    from PIL import Image

    if path.endswith(".bin"):
        return np.fromfile(path, dtype=np.uint8).reshape(128, 128, 3).astype(np.float64)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def compare(a: str, b: str, min_psnr: float | None) -> None:
    import numpy as np

    x, y = load_image(a), load_image(b)
    if x.shape != y.shape:
        sys.exit(f"shape mismatch: {x.shape} vs {y.shape}")
    mse = ((x - y) ** 2).mean()
    psnr = math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)
    print(f"PSNR {psnr:.2f} dB, max |diff| {int(np.abs(x - y).max())}, mean |diff| {np.abs(x - y).mean():.2f}")
    if min_psnr is not None and psnr < min_psnr:
        sys.exit(f"PSNR below {min_psnr} dB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["float", "fakequant", "tosa"], default="float")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--class", dest="cls", type=int, default=1, help="0..3, or 4 = unconditional")
    ap.add_argument("--w", type=float, default=4.0, help="guidance strength, 0 = plain")
    ap.add_argument("--k", type=int, default=4, help="Euler steps: 8, 4, 2 or 1")
    ap.add_argument("--out", default="out/pf.png")
    ap.add_argument("--mlops", help="*.cbuild-mlops.yml to take the Vela options from (fakequant)")
    ap.add_argument("--check-upstream", metavar="CLONE", help="path to a pico-faces clone")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="two images: .png, or a raw 128x128 RGB .bin")
    ap.add_argument("--min-psnr", type=float, help="with --compare: fail below this PSNR (dB)")
    args = ap.parse_args()
    run_in_venv(required=not args.compare)

    sys.path.insert(0, str(HERE))
    if args.check_upstream:
        check_upstream(args.check_upstream)
    elif args.compare:
        compare(*args.compare, args.min_psnr)
    elif args.stage == "tosa":
        check_tosa(args)
    else:
        sample(args)


if __name__ == "__main__":
    main()
