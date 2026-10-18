#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Export a separate U85 MobileNet candidate without changing deployment files."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    """Hash an existing file.

    :param path:    File to hash.
    :returns:       SHA256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """Calibrate with local images, evaluate PT2E, and compile a distinct PTE.

    :returns:   Zero after completing export.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory only")
    parser.add_argument("--calibration", type=Path,
                        default=Path.home() / "alif/ai-workflow/CalImages")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path.home() / ".cache/torch/hub/checkpoints/"
                        "mobilenet_v2-7ebf99e0.pth")
    args = parser.parse_args()
    images = sorted(path for path in args.calibration.iterdir()
                    if path.is_file() and not path.name.startswith("."))
    if len(images) != 60:
        parser.error(f"Expected the verified 60-image corpus, found {len(images)}")
    if sha256(args.checkpoint) != "7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f":
        parser.error("Unexpected FP32 checkpoint: refusing to change original weights")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    torch.manual_seed(0)
    torch.set_num_threads(1)

    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer, get_symmetric_quantization_config,
    )
    from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import ReplaceQuantNodesPass
    from executorch.exir import (
        EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower,
    )

    original = ROOT / "resources_downloaded/img_class/mv2_arm_delegate_ethos-u85-256.pte"
    protected = [original, args.checkpoint,
                 ROOT / "build_appkit_img_class/bin/mlek_alif_img_class.axf",
                 *sorted((ROOT / "build_appkit_img_class/bin/sectors/alif_img_class").glob("*.bin"))]
    before = {str(path): sha256(path) for path in protected if path.is_file()}
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    model = mobilenet_v2(weights=None).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    compile_spec = EthosUCompileSpec(
        target="ethos-u85-256", system_config="Ethos_U85_SRAM_MRAM",
        memory_mode="Shared_Sram", config_ini=str(ROOT / "scripts/vela/ensemble_vela.ini"),
        extra_flags=["--output-format=raw", "--debug-force-regor", "--verbose-allocation"],
    )
    compile_spec.dump_intermediate_artifacts_to(str(args.output / "intermediates"))
    example = (torch.zeros(1, 3, 224, 224),)
    exported = torch.export.export(model, example, strict=True)
    prepared = prepare_pt2e(exported.module(check_guards=False), EthosUQuantizer(compile_spec).set_global(
        get_symmetric_quantization_config(is_per_channel=True)))
    calibration_hashes = set()
    manifest = []
    with torch.inference_mode():
        for index, path in enumerate(images):
            with Image.open(path) as image:
                tensor = preprocess(ImageOps.exif_transpose(image).convert("RGB"))[None].contiguous()
            calibration_hashes.add(hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
            prepared(tensor)
            manifest.append(dict(path=str(path), sha256=sha256(path)))
            print(f"Calibration {index + 1}/60: {path.name}", flush=True)
    quantized = convert_pt2e(prepared)
    quantized_export = torch.export.export(quantized, example, strict=True)
    torch.export.save(quantized_export, args.output / "mobilenet_v2_cal60_quantized.pt2")
    quant_nodes = [dict(operator=str(node.target), parameters=list(node.args[1:]))
                   for node in quantized.graph.nodes
                   if "quantize_per_tensor" in str(node.target)]
    # Filename/source-caption based smoke-test labels; not an ImageNet validation benchmark.
    targets = {"cat.bmp": [281, 282, 285], "dog.bmp": [254],
               "kimono.bmp": [614], "tiger.bmp": [292]}
    results = []
    labels = weights.meta["categories"]
    with torch.inference_mode():
        for name, acceptable in targets.items():
            path = ROOT / "resources/img_class/samples" / name
            with Image.open(path) as image:
                tensor = preprocess(image.convert("RGB"))[None].contiguous()
            input_hash = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
            if input_hash in calibration_hashes:
                raise ValueError(f"Evaluation image overlaps calibration after preprocessing: {name}")
            fp32 = model(tensor)
            int8 = quantized(tensor)
            record = dict(image=str(path), sha256=sha256(path), input_sha256=input_hash,
                          acceptable_indices=acceptable, max_abs_logit_error=float(
                              (fp32 - int8).abs().max()),
                          mean_abs_logit_error=float((fp32 - int8).abs().mean()))
            for key, values in (("fp32", fp32), ("pt2e_quantized", int8)):
                indices = values[0].topk(5).indices.tolist()
                record[key] = dict(minimum=float(values.min()), maximum=float(values.max()),
                                   top5=[dict(index=i, label=labels[i], logit=float(values[0, i]),
                                              probability=float(values.softmax(1)[0, i]))
                                         for i in indices],
                                   top1_correct=indices[0] in acceptable,
                                   top5_correct=bool(set(indices) & set(acceptable)))
                np.save(args.output / f"{name}.{key}.npy", values.numpy())
            np.save(args.output / f"{name}.input.npy", tensor.numpy())
            results.append(record)
            print("EVALUATED", name, "FP32:", record["fp32"]["top5"][0]["label"],
                  "INT8:", record["pt2e_quantized"]["top5"][0]["label"], flush=True)
    report = dict(checkpoint_sha256=sha256(args.checkpoint), seed=0,
                  preprocessing=str(preprocess), calibration=manifest,
                  quantization="EthosUQuantizer, per-channel INT8 weights, INT8 activations",
                  quantization_nodes=quant_nodes, results=results,
                  versions={name: importlib.metadata.version(name) for name in
                            ("torch", "torchvision", "torchao", "executorch", "ethos-u-vela")},
                  protected_artifacts=before,
                  compiled_u85_execution="Not yet verified; PT2E results precede backend lowering")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("Lowering to Ethos-U85-256", flush=True)
    edge = to_edge_transform_and_lower(
        quantized_export, partitioner=[EthosUPartitioner(compile_spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False))
    edge = edge.transform([ReplaceQuantNodesPass()])
    program = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False))
    output = args.output / "mv2_cal60_imagenet_v2_ethos-u85-256_shared_sram.pte"
    with output.open("xb") as stream:
        program.write_to_file(stream)
    report["candidate"] = dict(path=str(output), bytes=output.stat().st_size,
                               sha256=sha256(output))
    report["protected_artifacts_unchanged"] = all(
        sha256(Path(path)) == value for path, value in before.items())
    if not report["protected_artifacts_unchanged"]:
        raise AssertionError("Existing artifact changed")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("EXPORTED", report["candidate"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
