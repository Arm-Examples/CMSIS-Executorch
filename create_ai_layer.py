#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Create the AI layer of the CMSIS solution from its MLOps information.

    python create_ai_layer.py <solution>.cbuild-mlops.yml

Step 2 of the three-step flow:

    cbuild setup <solution>.csolution.yml --active <target>   # writes *.cbuild-mlops.yml
    python create_ai_layer.py <solution>.cbuild-mlops.yml     # this script
    cbuild <solution>.csolution.yml --active <target>         # compile and link

The *.cbuild-mlops.yml is what CMSIS-Toolbox generates from the `mlops:` node
of the csolution. This script reads the NPU and Vela settings from it, exports
the PyTorch model in model/model.py for that NPU (quantize, delegate to
Ethos-U, compile with Vela) and writes the complete AI layer into the
directory of the clayer named under `model.clayer`:

    ai_layer.clayer.yml   runtime, Ethos-U backend and the operator components
                          the exported program actually uses
    model_pte.c / .h      the ExecuTorch program as a C array
    model.pte             the program itself, for inspection

The script runs itself in the solution's .venv (see setup_venv.py) when it is
started with an interpreter that has no torch.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = "PyTorch::ExecuTorch"
SYMBOL = "model_pte"


def run_in_venv() -> None:
    """Re-run under .venv when torch is not importable from this interpreter."""
    try:
        import torch  # noqa: F401
    except ImportError:
        venv = HERE / ".venv"
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # sys.prefix is the venv directory when running inside it (comparing
        # interpreter paths does not work: venv symlinks resolve to the base).
        if not python.is_file() or Path(sys.prefix).resolve() == venv.resolve():
            sys.exit(
                f"torch is not installed for {sys.executable}.\n"
                "Create the venv first: ./setup_venv.sh (Linux/macOS) or setup_venv.bat (Windows)"
            )
        sys.exit(subprocess.run([str(python), __file__, *sys.argv[1:]]).returncode)


def pack_root() -> Path:
    """The CMSIS pack root: $CMSIS_PACK_ROOT, else cpackget's default."""
    if env := os.environ.get("CMSIS_PACK_ROOT"):
        return Path(env)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Arm/Packs"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "arm/packs"


def executorch_pack(mlops_file: Path) -> Path:
    """Directory of the ExecuTorch pack version that cbuild setup resolved."""
    import yaml

    pack_file = mlops_file.with_name(mlops_file.name.replace(".cbuild-mlops.yml", ".cbuild-pack.yml"))
    for entry in yaml.safe_load(pack_file.read_text())["cbuild-pack"]["resolved-packs"]:
        name, _, version = entry["resolved-pack"].partition("@")
        if name == PACK and "selected-by-pack" in entry:
            vendor, _, pack = PACK.partition("::")
            return pack_root() / vendor / pack / version
    sys.exit(f"{pack_file}: {PACK} is not among the resolved packs")


def compile_spec(mlops: dict, mlops_dir: Path):
    """EthosUCompileSpec from the npu: and vela: nodes of the cbuild-mlops.yml."""
    from executorch.backends.arm.ethosu import EthosUCompileSpec

    npu = mlops.get("npu")
    if not npu:
        sys.exit("the solution's mlops: node names no NPU; this example needs an Ethos-U")
    vela = mlops.get("vela", {})
    options = vela.get("options", "")

    def option(name: str) -> str | None:
        found = re.search(rf"--{name}[= ](\S+)", options)
        return found.group(1) if found else None

    target = option("accelerator-config") or f"{npu['type'].lower()}-{npu.get('macs', 256)}"
    kwargs = {
        "target": target,
        "system_config": option("system-config"),
        "memory_mode": option("memory-mode"),
    }
    if vela.get("ini"):
        kwargs["config_ini"] = str(mlops_dir / vela["ini"])
    print(f"[ai_layer] Vela: {kwargs}")
    return EthosUCompileSpec(**kwargs)


def export_model(spec) -> bytes:
    """Quantize model/model.py, delegate it to the Ethos-U and return the .pte."""
    import torch
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    sys.path.insert(0, str(HERE / "model"))
    from model import get_calibration_inputs, get_model

    model, samples = get_model(), get_calibration_inputs()
    example = (samples[0],)
    graph = torch.export.export(model, example).module()

    # Quantize the whole graph so the partitioner can move every node into the
    # Ethos-U delegate; only a float<->int8 boundary stays on the CPU.
    quantizer = EthosUQuantizer(spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for sample in samples:
            prepared(sample)  # calibrate
    quantized = convert_pt2e(prepared)

    edge = to_edge_transform_and_lower(
        torch.export.export(quantized, example),
        partitioner=[EthosUPartitioner(spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    return bytes(program.buffer)


def components(pte: bytes, pack: Path) -> tuple[list[str], list[str]]:
    """Runtime + backend + one operator component per operator the .pte uses."""
    pdsc = next(pack.glob("*.pdsc"))
    available = set(re.findall(r'Csub="([^"]+)"', pdsc.read_text()))
    family = {"aten": "Portable", "quantized_decomposed": "Quantized", "cortex_m": "Cortex-M"}

    selected, unknown = set(), []
    for ns, op in sorted(set(re.findall(rb"(aten|quantized_decomposed|cortex_m)::(\w+)", pte))):
        ns, op = ns.decode(), op.decode()
        candidates = [
            f"{family[ns]} {op}",
            f"{family[ns]} {re.sub(r'_(per_tensor|per_channel|byte|copy)$', '', op)}",
        ]
        if match := next((c for c in candidates if c in available), None):
            selected.add(match)
        else:
            unknown.append(f"{ns}::{op}")
    if unknown:
        print(f"[ai_layer] warning: no component in {pack.name} for {unknown}", file=sys.stderr)

    return ["Runtime", "Kernel Utils", "Kernel Registration", "Backend EthosU"], sorted(selected)


def c_array(pte: bytes) -> str:
    rows = [", ".join(f"0x{b:02x}" for b in pte[i : i + 16]) for i in range(0, len(pte), 16)]
    return (
        "// Generated by create_ai_layer.py -- do not edit.\n"
        f"__attribute__((aligned(16))) const unsigned char {SYMBOL}[] = {{\n  "
        + ",\n  ".join(rows)
        + f"\n}};\nconst unsigned long {SYMBOL}_size = sizeof({SYMBOL});\n"
    )


HEADER = f"""// Generated by create_ai_layer.py -- do not edit.
#pragma once
#ifdef __cplusplus
extern "C" {{
#endif
extern const unsigned char {SYMBOL}[];
extern const unsigned long {SYMBOL}_size;
#ifdef __cplusplus
}}
#endif
"""


def clayer(mlops: dict, runtime: list[str], operators: list[str], mlops_file: Path) -> str:
    lines = [
        f"# Generated by create_ai_layer.py from {mlops_file.name} -- do not edit.",
        "# Re-run `python create_ai_layer.py` after changing model/model.py or the",
        "# csolution's mlops: node.",
        "layer:",
        "  type: AI",
        f"  description: {mlops.get('description', mlops['model'].get('name', 'AI layer'))}",
        "",
        "  packs:",
        f"    - pack: {PACK}",
        "",
        "  define:",
        "    - ET_LOG_ENABLED: 0",
        "",
        "  add-path:",
        "    - .",
        "",
        "  components:",
        *[f"    - component: Machine Learning:ExecuTorch:{c}" for c in runtime],
        *[f"    - component: Machine Learning:ExecuTorch Operators:{c}" for c in operators],
        "",
        "  groups:",
        f"    - group: {mlops['model'].get('name', 'Model')}",
        "      files:",
        f"        - file: ./{SYMBOL}.c",
        f"        - file: ./{SYMBOL}.h",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".cbuild-mlops.yml"):
        sys.exit(f"usage: {Path(__file__).name} <solution>.cbuild-mlops.yml")
    run_in_venv()

    import yaml

    mlops_file = Path(sys.argv[1]).resolve()
    mlops = yaml.safe_load(mlops_file.read_text())["cbuild-mlops"]
    layer_file = mlops_file.parent / mlops["model"]["clayer"]
    layer_dir = layer_file.parent

    pte = export_model(compile_spec(mlops, mlops_file.parent))
    runtime, operators = components(pte, executorch_pack(mlops_file))

    layer_dir.mkdir(parents=True, exist_ok=True)
    (layer_dir / "model.pte").write_bytes(pte)
    (layer_dir / f"{SYMBOL}.c").write_text(c_array(pte), newline="\n")
    (layer_dir / f"{SYMBOL}.h").write_text(HEADER, newline="\n")
    layer_file.write_text(clayer(mlops, runtime, operators, mlops_file), newline="\n")

    print(f"[ai_layer] {len(pte)} byte program, operators: {operators}")
    print(f"[ai_layer] wrote {layer_file}")


if __name__ == "__main__":
    main()
