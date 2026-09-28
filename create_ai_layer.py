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

    ai_layer.clayer.yml   runtime, kernel utilities and registration, Ethos-U
                          backend and the operator components the exported
                          program actually uses
    model_pte.c / .h      the ExecuTorch program as a C array
    model_params.h        constants of the model for the application (optional)
    model.pte             the program itself, for inspection

model/model.py describes the model through a small contract:

    get_model(), get_calibration_inputs()   one method, "forward" (the simple case)
    get_methods()                           several methods, each with its module,
                                            example inputs, quantization and
                                            calibration data (see Method below)
    get_params()                            constants written to model_params.h

Every method is quantized, calibrated on its data and delegated to the Ethos-U.
The script prints, per method, how many Ethos-U delegates the graph has and
which operators remain on the CPU. Set AI_LAYER_STRICT=1 to fail when a method
is not a single delegate, AI_LAYER_VERBOSE=1 for the backend's partitioning
diagnostics, AI_LAYER_DUMP=<dir> to keep the TOSA and Vela artefacts, and
AI_LAYER_VELA_FLAGS for extra Vela options (e.g. --verbose-performance).

The script runs itself in the solution's .venv (see setup_venv.py) when it is
started with another interpreter.
"""

from __future__ import annotations

import logging
import operator
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
PACK = "PyTorch::ExecuTorch"
SYMBOL = "model_pte"
PARAMS_HEADER = "model_params.h"
# The float <-> integer boundary of a fully delegated method stays on the CPU.
BOUNDARY_OPS = {
    "quantized_decomposed::quantize_per_tensor",
    "quantized_decomposed::dequantize_per_tensor",
}


@dataclass(frozen=True)
class Method:
    """One method of the exported program (model.get_methods() returns these).

    calibration yields input tuples for the quantizer's observers; without it
    the example inputs are used. quantization is "a8w8" (int8 activations and
    weights) or "a16w8" (int16 activations, int8 weights).
    """

    name: str
    module: Any
    example_inputs: tuple
    calibration: Callable[[], Iterable[tuple]] | None = None
    quantization: str = "a8w8"


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
    """The ExecuTorch pack version cbuild setup resolved for the csolution.

    <solution>.cbuild-pack.yml is a lock file that keeps earlier resolutions:
    after a pack update, an AI layer generated for the previous version still
    selects that one. The entry selected by the csolution's own pack entry
    (PyTorch::ExecuTorch@<version>) is the one in use.
    """
    import yaml

    solution = mlops_file.with_name(mlops_file.name.replace(".cbuild-mlops.yml", ".csolution.yml"))
    wanted = [
        entry["pack"]
        for entry in yaml.safe_load(solution.read_text())["solution"].get("packs", [])
        if entry["pack"].partition("@")[0] == PACK
    ]
    pack_file = mlops_file.with_name(mlops_file.name.replace(".cbuild-mlops.yml", ".cbuild-pack.yml"))
    for entry in yaml.safe_load(pack_file.read_text())["cbuild-pack"]["resolved-packs"]:
        name, _, version = entry["resolved-pack"].partition("@")
        if name == PACK and set(wanted) & set(entry.get("selected-by-pack", [])):
            return version
    sys.exit(f"{pack_file}: no {PACK} resolved for {wanted or 'the csolution'}; run cbuild setup first")


def executorch_pack(version: str) -> Path:
    """Directory of the installed ExecuTorch pack of that version."""
    vendor, _, pack = PACK.partition("::")
    return pack_root() / vendor / pack / version


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
        # ExecuTorch stores the path in the compile spec, and the spec ends up
        # in the .pte. A path relative to the working directory keeps the
        # program identical between checkouts; Vela resolves it from there.
        kwargs["config_ini"] = os.path.relpath(mlops_dir / vela["ini"])
    if flags := os.environ.get("AI_LAYER_VELA_FLAGS", "").split():
        kwargs["extra_flags"] = flags
    print(f"[ai_layer] Vela: {kwargs}")
    spec = EthosUCompileSpec(**kwargs)
    if dump := os.environ.get("AI_LAYER_DUMP"):
        spec.dump_intermediate_artifacts_to(dump)
    return spec


def compile_spec_from_file(mlops_file: str | Path):
    """compile_spec() for a *.cbuild-mlops.yml (used by model/verify_export.py)."""
    import yaml

    path = Path(mlops_file).resolve()
    return compile_spec(yaml.safe_load(path.read_text())["cbuild-mlops"], path.parent)


def default_compile_spec():
    """An Ethos-U85-256 spec with Vela's built-in system config, for host checks
    that run without a *.cbuild-mlops.yml (the quantizer only needs the NPU)."""
    return compile_spec(
        {"npu": {"type": "Ethos-U85", "macs": 256},
         "vela": {"options": "--system-config Ethos_U85_SYS_DRAM_Mid --memory-mode Shared_Sram"}},
        HERE,
    )


# ----------------------------------------------------------------------------
# The model contract


def load_model_module():
    sys.path.insert(0, str(HERE / "model"))
    import model

    return model


def methods(model) -> list[Method]:
    """model.get_methods(), or "forward" from get_model() and get_calibration_inputs()."""
    if hasattr(model, "get_methods"):
        return list(model.get_methods())
    samples = model.get_calibration_inputs()
    return [Method("forward", model.get_model(), (samples[0],), lambda: [(s,) for s in samples])]


def quant_config(kind: str):
    from executorch.backends.arm.quantizer import (
        get_symmetric_a16w8_quantization_config,
        get_symmetric_quantization_config,
    )

    if kind == "a8w8":
        return get_symmetric_quantization_config(is_per_channel=True)
    if kind == "a16w8":
        return get_symmetric_a16w8_quantization_config(is_per_channel=True)
    sys.exit(f"unknown quantization {kind!r}: expected a8w8 or a16w8")


def strip_guards_fn(gm) -> None:
    """Drop the dead `_guards_fn` call_module that torch.export emits for some
    models; the Arm annotation passes iterate over every module unconditionally."""
    changed = False
    for node in list(gm.graph.nodes):
        if node.op == "call_module" and str(node.target) == "_guards_fn":
            gm.graph.erase_node(node)
            changed = True
    if changed:
        gm.graph.eliminate_dead_code()
        gm.recompile()


def calibrate(prepared, method: Method) -> None:
    """Run the observers over the method's calibration data (or its example inputs)."""
    import torch

    batches = method.calibration() if method.calibration else [method.example_inputs]
    count, t0 = 0, time.time()
    with torch.no_grad():
        for inputs in batches:
            prepared(*inputs)
            count += 1
    print(f"[ai_layer] {method.name}: calibrated on {count} input set(s) in {time.time() - t0:.1f} s")


def quantize_method(method: Method, spec):
    """The method's module with quantize/dequantize nodes (the fake-quant graph)."""
    import torch
    from executorch.backends.arm.quantizer import EthosUQuantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    graph = torch.export.export(method.module, method.example_inputs).module()
    strip_guards_fn(graph)
    # Quantize the whole graph so the partitioner can move every node into the
    # Ethos-U delegate; only the float <-> integer boundary stays on the CPU.
    quantizer = EthosUQuantizer(spec)
    quantizer.set_global(quant_config(method.quantization))
    prepared = prepare_pt2e(graph, quantizer)
    calibrate(prepared, method)
    return convert_pt2e(prepared)


# ----------------------------------------------------------------------------
# Export


def op_name(target) -> str | None:
    """`aten::mul.Tensor` for an edge or aten op object; None for graph plumbing."""
    if target is operator.getitem:
        return None
    name = getattr(target, "name", None)
    if callable(name):
        try:
            return name()
        except Exception:
            pass
    text = str(target)
    if found := re.search(r"schema = ([a-z_0-9]+::[\w.]+)", text):
        return found.group(1)
    if found := re.match(r"^([a-z_0-9]+::[\w.]+)$", text):
        return found.group(1)
    return None


def count_ops(graph_module) -> tuple[int, Counter]:
    delegates, cpu_ops = 0, Counter()
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if "executorch_call_delegate" in str(node.target):
            delegates += 1
        elif name := op_name(node.target):
            cpu_ops[name] += 1
    return delegates, cpu_ops


def report_partitioning(edge, name: str) -> set[str]:
    """Print the delegate count and CPU operators of one method; return the CPU ops."""
    delegates, cpu_ops = count_ops(edge.exported_program(name).graph_module)
    ops = ", ".join(f"{op} x{n}" if n > 1 else op for op, n in cpu_ops.most_common()) or "none"
    print(f"[ai_layer] {name}: {delegates} Ethos-U delegate(s); CPU operators: {ops}")
    stray = {re.sub(r"\.\w+$", "", op) for op in cpu_ops} - BOUNDARY_OPS
    if delegates != 1 or stray:
        message = (
            f"[ai_layer] warning: {name} is not a single Ethos-U delegate "
            f"({delegates} delegate(s), CPU operators beyond the quantize/dequantize "
            f"boundary: {sorted(stray) or 'none'}); set AI_LAYER_VERBOSE=1 for the reasons"
        )
        if os.environ.get("AI_LAYER_STRICT"):
            sys.exit(message)
        print(message, file=sys.stderr)
    return set(cpu_ops)


def export_model(spec, model) -> tuple[bytes, set[str]]:
    """Quantize, delegate and serialize every method; return the .pte and the CPU operators."""
    import torch
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower

    programs = {}
    for method in methods(model):
        print(f"[ai_layer] {method.name}: quantization {method.quantization}")
        programs[method.name] = torch.export.export(quantize_method(method, spec), method.example_inputs)

    edge = to_edge_transform_and_lower(
        programs,
        partitioner={name: [EthosUPartitioner(spec)] for name in programs},
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    cpu_ops: set[str] = set()
    for name in programs:
        cpu_ops |= report_partitioning(edge, name)

    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    for name in programs:  # operators to_executorch adds (copies, for example) need components too
        cpu_ops |= set(count_ops(program.exported_program(name).graph_module)[1])
    return bytes(program.buffer), cpu_ops


# ----------------------------------------------------------------------------
# The generated layer


def components(pte: bytes, pack: Path, cpu_ops: set[str]) -> tuple[list[str], list[str]]:
    """Runtime, kernel utils and registration, backend, plus one operator component per operator the .pte uses.

    The pack's "Extension Tensor" is not selected: its tensor_ptr_maker.cpp
    needs std::random_device, which the LLVM embedded toolchain lacks; the
    runner wraps its input in a TensorImpl instead.
    """
    pdsc = next(pack.glob("*.pdsc"))
    available = set(re.findall(r'Csub="([^"]+)"', pdsc.read_text()))
    family = {"aten": "Portable", "quantized_decomposed": "Quantized", "cortex_m": "Cortex-M"}

    found = {(ns.decode(), op.decode()) for ns, op in re.findall(rb"(aten|quantized_decomposed|cortex_m)::(\w+)", pte)}
    for name in cpu_ops:
        ns, _, op = name.partition("::")
        if ns in family:
            found.add((ns, op.split(".")[0]))

    selected, unknown = set(), []
    for ns, op in sorted(found):
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
    """The program as a C array in the section .rodata.model, so a board layer's
    linker script can give it a memory of its own (the Corstone-320 layer puts
    it in DDR: the 2 MB FPGA SRAM that holds the code is too small); scripts
    that only know .rodata* or +RO still pick it up."""
    rows = [", ".join(f"0x{b:02x}" for b in pte[i : i + 16]) for i in range(0, len(pte), 16)]
    return (
        "// Generated by create_ai_layer.py -- do not edit.\n"
        f'__attribute__((aligned(16), section(".rodata.model"))) const unsigned char {SYMBOL}[] = {{\n  '
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


def c_number(value: float) -> str:
    text = f"{float(value):.9g}"
    if not any(ch in text for ch in ".en"):  # e, inf, nan
        text += ".0"
    return text + "f"


def c_initializer(values, indent: str = "    ") -> list[str]:
    """Lines of a nested brace initializer for a float array of any rank."""
    if values.ndim == 1:
        items = [c_number(v) for v in values.tolist()]
        return [indent + ", ".join(items[i : i + 8]) + "," for i in range(0, len(items), 8)]
    lines = []
    for row in values:
        lines += [indent + "{", *c_initializer(row, indent + "  "), indent + "},"]
    return lines


def params_header(params: dict[str, Any]) -> str:
    """model_params.h from model.get_params(), names as given: strings, integers
    and floats become #defines, sequences and tensors of numbers `static const
    float` arrays of the same shape."""
    import numpy as np

    lines = [
        "// Generated by create_ai_layer.py from get_params() in model/model.py -- do not edit.",
        "#pragma once",
        "",
    ]
    for name, value in params.items():
        if isinstance(value, str):
            lines.append(f'#define {name} "{value}"')
        elif isinstance(value, (bool, int)):
            lines.append(f"#define {name} {int(value)}")
        elif isinstance(value, float):
            lines.append(f"#define {name} {c_number(value)}")
        else:
            array = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float32)
            dims = "".join(f"[{d}]" for d in array.shape)
            lines += ["", f"static const float {name}{dims} = {{", *c_initializer(array), "};"]
    return "\n".join(lines) + "\n"


def clayer(mlops: dict, runtime: list[str], operators: list[str], mlops_file: Path, version: str, files: list[str]) -> str:
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
        "",
        "  define:",
        "    - ET_LOG_ENABLED: 1",  # the runner prints the runtime's error messages,
        "    - ET_MIN_LOG_LEVEL: Error",  # but not its progress notes
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
        *[f"        - file: ./{name}" for name in files],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".cbuild-mlops.yml"):
        sys.exit(f"usage: {Path(__file__).name} <solution>.cbuild-mlops.yml")
    run_in_venv()

    import yaml

    if os.environ.get("AI_LAYER_VERBOSE"):
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("executorch.backends.arm").setLevel(logging.INFO)

    mlops_file = Path(sys.argv[1]).resolve()
    mlops = yaml.safe_load(mlops_file.read_text())["cbuild-mlops"]
    layer_file = mlops_file.parent / mlops["model"]["clayer"]
    layer_dir = layer_file.parent

    model = load_model_module()
    pte, cpu_ops = export_model(compile_spec(mlops, mlops_file.parent), model)
    version = executorch_version(mlops_file)
    runtime, operators = components(pte, executorch_pack(version), cpu_ops)

    layer_dir.mkdir(parents=True, exist_ok=True)
    files = [f"{SYMBOL}.c", f"{SYMBOL}.h"]
    (layer_dir / "model.pte").write_bytes(pte)
    (layer_dir / f"{SYMBOL}.c").write_text(c_array(pte), newline="\n")
    (layer_dir / f"{SYMBOL}.h").write_text(HEADER, newline="\n")
    if hasattr(model, "get_params"):
        (layer_dir / PARAMS_HEADER).write_text(params_header(model.get_params()), newline="\n")
        files.append(PARAMS_HEADER)
    layer_file.write_text(clayer(mlops, runtime, operators, mlops_file, version, files), newline="\n")

    print(f"[ai_layer] {len(pte)} byte program, operators: {operators}")
    print(f"[ai_layer] wrote {layer_file}")


if __name__ == "__main__":
    main()
