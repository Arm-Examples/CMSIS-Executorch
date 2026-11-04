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
of the csolution. This script reads the processor, NPU and Vela settings from
it, exports the PyTorch model in model/model.py for that target and writes the
complete AI layer into the directory of the clayer named under `model.clayer`:

    with an `npu:` node     quantize, delegate the graph to the Ethos-U and
                            compile it with Vela
    without one             quantize for the Cortex-M alone: the ExecuTorch
                            Cortex-M backend lowers the operators to CMSIS-NN

The extra keys of the `model:` node (input-shape, calibration-samples) are
model parameters that CMSIS-Toolbox passes through to this script:

    ai_layer.clayer.yml   runtime, kernel utilities and registration, the
                          Ethos-U backend on an NPU target (CMSIS-NN on a
                          CPU-only one) and the operator components the
                          exported program actually uses
    model_pte.c / .h      the ExecuTorch program as a C array
    model.pte             the program itself, for inspection

The script runs itself in the solution's .venv (see setup_venv.py) when it is
started with an interpreter that has no torch.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = "PyTorch::ExecuTorch"
SYMBOL = "model_pte"


def run_in_venv() -> None:
    """Re-run under the project's .venv unless this interpreter already is it.

    Deciding by "does torch import" is not enough: a torch installed for the
    host interpreter would keep the export outside the environment with the
    pinned executorch. sys.prefix is the venv directory when running inside it
    (comparing interpreter paths does not work: venv symlinks resolve to the
    base interpreter).
    """
    venv = HERE / ".venv"
    if Path(sys.prefix).resolve() == venv.resolve():
        return
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        sys.exit(
            f"{venv} does not exist.\n"
            "Create it first: ./setup_venv.sh (Linux/macOS) or setup_venv.bat (Windows)"
        )
    print(f"[ai_layer] running in {python}", flush=True)
    sys.exit(subprocess.run([str(python), __file__, *sys.argv[1:]]).returncode)


def pack_root() -> Path:
    """The CMSIS pack root: $CMSIS_PACK_ROOT, else cpackget's default."""
    if env := os.environ.get("CMSIS_PACK_ROOT"):
        return Path(env)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Arm/Packs"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "arm/packs"


def executorch_version(mlops_file: Path) -> str:
    """The ExecuTorch pack version cbuild setup resolved, from <solution>.cbuild-pack.yml.

    The file is a lock file that keeps earlier resolutions, so an unversioned
    selector can still point at an older pack; the entry selected by the
    csolution's exact pin (PyTorch::ExecuTorch@<version>) is the one in use.
    """
    import yaml

    pack_file = mlops_file.with_name(mlops_file.name.replace(".cbuild-mlops.yml", ".cbuild-pack.yml"))
    fallback = None
    for entry in yaml.safe_load(pack_file.read_text())["cbuild-pack"]["resolved-packs"]:
        name, _, version = entry["resolved-pack"].partition("@")
        selectors = entry.get("selected-by-pack", [])
        if name != PACK or not selectors:
            continue
        if f"{PACK}@{version}" in selectors:
            return version
        fallback = fallback or version
    if fallback:
        return fallback
    sys.exit(f"{pack_file}: {PACK} is not among the resolved packs")


def executorch_pack(version: str) -> Path:
    """Directory of the installed ExecuTorch pack of that version."""
    vendor, _, pack = PACK.partition("::")
    return pack_root() / vendor / pack / version


# Vela options that are arguments of EthosUCompileSpec, and the ones the export
# sets itself: the configuration file comes from vela.ini of the cbuild-mlops.yml.
SPEC_OPTIONS = ("accelerator-config", "system-config", "memory-mode")
EXPORT_OPTIONS = ("config", "output-format", "output-dir")


def vela_options(options: str) -> tuple[dict[str, str], list[str]]:
    """Split vela.options of the cbuild-mlops.yml: the EthosUCompileSpec arguments, and every other option.

    The other options are the `misc:` of the csolution's mlops: node (for
    example `--optimise Size`); they reach Vela as extra flags, written as
    `--name=value`.
    """
    tokens = shlex.split(options)
    spec, extra = {}, []
    while tokens:
        token = tokens.pop(0)
        if not token.startswith("--"):
            sys.exit(f"vela options: unexpected '{token}' in '{options}'")
        name, has_value, value = token[2:].partition("=")
        if not has_value and tokens and not tokens[0].startswith("-"):
            value = tokens.pop(0)
        if name in EXPORT_OPTIONS:
            sys.exit(
                f"vela options: --{name} is set by the export; "
                "the configuration file is vela: ini: of the mlops: node"
            )
        if name in SPEC_OPTIONS:
            if name in spec:
                sys.exit(f"vela options: --{name} is given twice in '{options}'")
            spec[name] = value
        else:
            extra.append(f"--{name}={value}" if value else f"--{name}")
    return spec, extra


def compile_spec(mlops: dict, mlops_dir: Path):
    """EthosUCompileSpec from the npu: and vela: nodes of the cbuild-mlops.yml.

    No npu: node means a CPU-only target: the model is then compiled for the
    processor with the Cortex-M backend (see cpu_target) and None is returned.
    """
    npu = mlops.get("npu")
    if not npu:
        return None
    from executorch.backends.arm.ethosu import EthosUCompileSpec

    vela = mlops.get("vela", {})
    spec, extra = vela_options(vela.get("options", ""))

    target = spec.get("accelerator-config") or f"{npu['type'].lower()}-{npu.get('macs', 256)}"
    kwargs = {
        "target": target,
        "system_config": spec.get("system-config"),
        "memory_mode": spec.get("memory-mode"),
        "extra_flags": extra,
    }
    if vela.get("ini"):
        # ExecuTorch stores the path in the compile spec, and the spec ends up
        # in the .pte. A path relative to the working directory keeps the
        # program identical between checkouts; Vela resolves it from there.
        kwargs["config_ini"] = os.path.relpath(mlops_dir / vela["ini"])
    print(f"[ai_layer] Vela: {kwargs}")
    return EthosUCompileSpec(**kwargs)


def cpu_target(mlops: dict):
    """CortexMTargetConfig for the processor: node of the cbuild-mlops.yml.

    The toolbox writes the core as the DFP names it (Cortex-M55, Cortex-M0+,
    Cortex-M35P); the Cortex-M backend's target strings are cortex-m55,
    cortex-m0plus, cortex-m35p. The config picks the CMSIS-NN kernel variant
    (scalar, DSP or MVE) the core can run.
    """
    from executorch.backends.cortex_m.target_config import CortexMTargetConfig

    core = mlops.get("processor", {}).get("type")
    if not core:
        sys.exit("the cbuild-mlops.yml names neither an NPU nor a processor")
    target = core.lower().replace("+", "plus")
    try:
        config = CortexMTargetConfig.from_target_string(target)
    except ValueError as error:
        sys.exit(f"no Cortex-M backend target for processor {core!r}: {error}")
    print(f"[ai_layer] Cortex-M: {config.target_string}, CMSIS-NN {config.backend.name} kernels")
    return config


def model_params(mlops: dict) -> dict:
    """The extra keys of the model: node, as keyword arguments for model/model.py."""
    params = {}
    for key, value in mlops["model"].items():
        if key == "input-shape":
            params["input_shape"] = tuple(int(d) for d in re.split(r"[x,]", str(value)))
        elif key == "calibration-samples":
            params["calibration_samples"] = int(value)
        elif key not in ("clayer", "name"):
            print(f"[ai_layer] warning: ignoring unknown model key {key!r}", file=sys.stderr)
    if params:
        print(f"[ai_layer] model parameters: {params}")
    return params


def load_model(params: dict):
    """model/model.py's model and calibration samples; the first sample is the export example."""
    import torch

    sys.path.insert(0, str(HERE / "model"))
    from model import get_calibration_inputs, get_model

    model = get_model(**{k: v for k, v in params.items() if k == "input_shape"})
    samples = get_calibration_inputs(**params)
    example = (samples[0],)
    return torch.export.export(model, example).module(), samples, example


def quantize(graph, quantizer, samples):
    """PT2E quantization: annotate, calibrate with every sample, convert."""
    import torch
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for sample in samples:
            prepared(sample)  # calibrate
    return convert_pt2e(prepared)


def export_model(spec, params: dict) -> bytes:
    """Quantize model/model.py, delegate it to the Ethos-U and return the .pte."""
    import torch
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower

    graph, samples, example = load_model(params)

    # Quantize the whole graph so the partitioner can move every node into the
    # Ethos-U delegate; only a float<->int8 boundary stays on the CPU.
    quantizer = EthosUQuantizer(spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    quantized = quantize(graph, quantizer, samples)

    edge = to_edge_transform_and_lower(
        torch.export.export(quantized, example),
        partitioner=[EthosUPartitioner(spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    return bytes(program.buffer)


def export_model_cpu(target, params: dict) -> bytes:
    """Quantize model/model.py for the Cortex-M alone and return the .pte.

    The Cortex-M backend delegates nothing: its passes rewrite the quantized
    operators in place into cortex_m:: operators, which the pack implements
    with CMSIS-NN. The sequence is the one of ExecuTorch's aot_arm_compiler
    for a cortex-m<n> target, in its explicit-layout mode: the convolution
    and pooling kernels then take NHWC tensors and the passes cancel the
    NHWC-to-NCHW permute a model like model/model.py applies to its camera
    frame input, so the program consumes the frame as is. (The backend's
    default dim-order mode turns that permute into a data-moving transpose
    whose output the kernels reject at run time.)
    """
    import torch
    from executorch.backends.cortex_m.edge_compile_config import cortex_m_edge_compile_config
    from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower

    graph, samples, example = load_model(params)
    quantized = quantize(graph, CortexMQuantizer(use_explicit_layout=True), samples)

    edge = to_edge_transform_and_lower(
        torch.export.export(quantized, example),
        compile_config=cortex_m_edge_compile_config(),
    )
    edge._edge_programs["forward"] = CortexMPassManager(
        edge.exported_program(), target_config=target, use_explicit_layout=True
    ).transform()
    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    return bytes(program.buffer)


# Operator namespaces in the .pte and the pack's component families; a few
# component names abbreviate the operator.
FAMILY = {"aten": "Portable", "dim_order_ops": "Portable", "quantized_decomposed": "Quantized", "cortex_m": "Cortex-M"}
ALIASES = {"quantized_depthwise_conv2d": "quantized_dw_conv2d", "quantized_transpose_conv2d": "quantized_tr_conv2d"}


def pack_components(pack: Path) -> tuple[set[str], set[str]]:
    """The pack's operator components, and those of them that need CMSIS-NN.

    A component needs CMSIS-NN when its condition requires the pack's CMSIS-NN
    condition, directly or through another condition.
    """
    root = ET.parse(next(pack.glob("*.pdsc"))).getroot()
    requires = {
        c.get("id"): [r.get("condition") for r in c if r.tag == "require" and r.get("condition")]
        for c in root.iter("condition")
    }

    def needs_cmsis_nn(condition: str | None) -> bool:
        return condition == "CMSIS-NN" or any(needs_cmsis_nn(r) for r in requires.get(condition, []))

    operators = {
        c.get("Csub"): c.get("condition") for c in root.iter("component") if c.get("Cgroup") == "ExecuTorch Operators"
    }
    return set(operators), {name for name, condition in operators.items() if needs_cmsis_nn(condition)}


def components(pte: bytes, pack: Path, delegated: bool) -> tuple[list[str], list[str], bool]:
    """Runtime components and one operator component per operator the .pte uses, and whether CMSIS-NN is needed.

    The runtime list is Runtime, Kernel Utils and Kernel Registration, plus
    the Ethos-U backend for a delegated program. The pack's "Extension Tensor"
    is not selected: its tensor_ptr_maker.cpp needs std::random_device, which
    the LLVM embedded toolchain lacks; the runner wraps its input in a
    TensorImpl instead.
    """
    available, cmsis_nn = pack_components(pack)
    namespaces = "|".join(FAMILY)

    selected, unknown = set(), []
    for ns, op in sorted(set(re.findall(rf"({namespaces})::(\w+)".encode(), pte))):
        ns, op = ns.decode(), op.decode()
        op = ALIASES.get(op, op)
        candidates = [
            f"{FAMILY[ns]} {op}",
            f"{FAMILY[ns]} {op.lstrip('_')}",
            f"{FAMILY[ns]} {re.sub(r'_(per_tensor|per_channel|byte|copy|nhwc)$', '', op)}",
        ]
        if match := next((c for c in candidates if c in available), None):
            selected.add(match)
        else:
            unknown.append(f"{ns}::{op}")
    if unknown:
        print(f"[ai_layer] warning: no component in {pack.name} for {unknown}", file=sys.stderr)

    runtime = ["Runtime", "Kernel Utils", "Kernel Registration"] + (["Backend EthosU"] if delegated else [])
    return runtime, sorted(selected), bool(selected & cmsis_nn)


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


def clayer(
    mlops: dict, runtime: list[str], operators: list[str], cmsis_nn: bool, mlops_file: Path, version: str
) -> str:
    # The Cortex-M operator components are CMSIS-NN kernels: they need the
    # library, and on a core with Helium floating point (Cortex-M55, M85) the
    # CMSIS-NN 8.0.0 pack only compiles with its float support enabled (see
    # the ExecuTorch pack's README).
    lines = [
        f"# Generated by create_ai_layer.py from {mlops_file.name} -- do not edit.",
        f"# Re-run `python create_ai_layer.py {mlops_file.name}` after changing",
        "# model/model.py or the csolution's mlops: node.",
        "layer:",
        "  type: AI",
        f"  description: {mlops.get('description', mlops['model'].get('name', 'AI layer'))}",
        "",
        "  packs:",
        f"    - pack: {PACK}@{version}",
        *(["    - pack: ARM::CMSIS-NN"] if cmsis_nn else []),
        "",
        "  define:",
        "    - ET_LOG_ENABLED: 0",
        *(["    - ARM_NN_ENABLE_F32: 1", "    - ARM_NN_ENABLE_F16: 1"] if cmsis_nn else []),
        "",
        "  add-path:",
        "    - .",
        "",
        "  components:",
        *[f"    - component: Machine Learning:ExecuTorch:{c}" for c in runtime],
        *[f"    - component: Machine Learning:ExecuTorch Operators:{c}" for c in operators],
        *(["    - component: CMSIS:NN Lib"] if cmsis_nn else []),
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

    params = model_params(mlops)
    spec = compile_spec(mlops, mlops_file.parent)
    pte = export_model(spec, params) if spec else export_model_cpu(cpu_target(mlops), params)
    version = executorch_version(mlops_file)
    runtime, operators, cmsis_nn = components(pte, executorch_pack(version), delegated=spec is not None)

    layer_dir.mkdir(parents=True, exist_ok=True)
    (layer_dir / "model.pte").write_bytes(pte)
    (layer_dir / f"{SYMBOL}.c").write_text(c_array(pte), newline="\n")
    (layer_dir / f"{SYMBOL}.h").write_text(HEADER, newline="\n")
    layer_file.write_text(clayer(mlops, runtime, operators, cmsis_nn, mlops_file, version), newline="\n")

    print(f"[ai_layer] {len(pte)} byte program, operators: {operators}")
    print(f"[ai_layer] wrote {layer_file}")


if __name__ == "__main__":
    main()
