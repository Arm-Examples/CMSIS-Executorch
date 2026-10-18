# SPDX-License-Identifier: Apache-2.0
"""Export a trained TinyEEGNet checkpoint to an int8 Ethos-U85 ``.pte`` plus its contract files.

    python -m src.export --checkpoint runs/loso_s1_seed0/model.pt --subject 1 \\
        --out artifacts/tinyeegnet_int8.pte

Mirrors ``CMSIS-Executorch/create_ai_layer.py:export_model`` (PTQ with ``EthosUQuantizer``,
``EthosUPartitioner``, ``ethos-u85-256``). Writes next to ``--out``:

- ``tinyeegnet_int8.pte``     the delegated program
- ``tinyeegnet_float.pt``     copy of the float state_dict
- ``calibration.npz``         ``x`` (200, 1, 22, 500) float32 aligned+scaled training windows
- ``test_vectors.npz``        ``x`` int8 (20, 1, 22, 500) = subject N session 2 trials 25-44,
                              ``logits`` float32 (20, 2) of the quantized model, ``y`` int64 (20,)
- ``preprocess_params.json``  filter SOS, window, EA matrix of subject N, input scaling, int8
                              input qparams, channel order, class map

and ``results/op_placement.md`` / ``results/deployment.md``. The ``.pte`` is not executed on
the host (the Ethos-U delegate needs the NPU); host numbers come from the quantized graph.

v2 student runs (``student.json`` beside the checkpoint, or a 256-sample ``params.json``):

    python -m src.export --checkpoint runs/v2/student_wide32_seed0/model.pt --subject 3 \\
        --out runs/v2/student_wide32_seed0/export/tinyeegnet_int8.pte

builds ``TinyEEGNet(n_samples=256, **student.json)``, input ``(1, 1, 22, 256)``, calibrates with
the run's ``calibration.npz`` and takes the 2a data from ``src.data.splits125.v2_data`` (seed,
config and ``input_scale`` from the run's ``params.json``). It writes the same files plus
``student.json``; ``test_vectors.npz`` holds ``x`` int8 (20, 1, 22, 256) = subject N session 2
trials 25-44 (the canonical 125 Hz window) and ``preprocess_params.json`` the v2 board pipeline
(250 Hz UART, CAR, SOS at 250 Hz, decimate by 2, 256-sample window, EA, scaling). The
op-placement and deployment reports go beside ``--out`` unless ``--results-dir`` is given.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import torch

from src.models.tinyeegnet import INPUT_SHAPE
from src.quantize import (DEFAULT_TARGET, calibration_windows, check_v2_data, compile_spec, float_logits,
                          input_qparams, int8_logits, is_v2_run, load_float_model, load_v2_data, quantize,
                          quantize_input, student_widths, v2_run_config)

N_VECTORS = 20
CLASS_MAP = {"0": "left_hand", "1": "right_hand"}


def lower(quantized: torch.fx.GraphModule, spec, input_shape: tuple[int, ...] = INPUT_SHAPE):
    """Quantized GraphModule -> (EdgeProgramManager after Ethos-U partitioning, ExecuTorch program)."""
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower

    example = (torch.zeros(input_shape),)
    edge = to_edge_transform_and_lower(
        torch.export.export(quantized, example),
        partitioner=[EthosUPartitioner(spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    # Keep the lowered edge program for the op-placement report before to_executorch mutates it.
    from executorch.devtools.backend_debug import get_delegation_info

    info = get_delegation_info(edge.exported_program().graph_module)
    top_level = [_node_name(n) for n in edge.exported_program().graph.nodes if n.op == "call_function"]
    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    return info, top_level, program


def _node_name(node) -> str:
    target = node.target
    return getattr(target, "__name__", None) or str(target)


def pte_operators(pte: bytes) -> list[str]:
    """Operators the CPU runtime must provide, as ``create_ai_layer.py:components`` finds them."""
    return sorted({f"{ns.decode()}::{op.decode()}"
                   for ns, op in re.findall(rb"(aten|quantized_decomposed|cortex_m)::(\w+)", pte)})


def count_macs(model: torch.nn.Module, input_shape: tuple[int, ...] = INPUT_SHAPE) -> tuple[int, dict[str, int]]:
    """Multiply-accumulates of one ``input_shape`` forward pass (conv + linear), per layer."""
    per_layer: dict[str, int] = {}

    def hook(name):
        def fn(mod, inp, out):
            if isinstance(mod, torch.nn.Conv2d):
                k = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
                per_layer[name] = int(out.numel() * k)
            else:
                per_layer[name] = int(mod.in_features * mod.out_features)
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
               if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for h in handles:
        h.remove()
    return sum(per_layer.values()), per_layer


def _vela_summary(log: str) -> list[str]:
    """The "Network summary" block Vela prints while compiling the delegate."""
    keep = ("Accelerator configuration", "System configuration", "Memory mode", "Total SRAM used",
            "Total On-chip Flash used", "Total Off-chip Flash used", "Total DRAM used", "CPU operators",
            "NPU operators", "Neural network macs", "Batch Inference time")
    return [" ".join(line.split()) for line in log.splitlines() if line.strip().startswith(keep)]


def vela_breakdown(quantized: torch.fx.GraphModule, args: argparse.Namespace,
                   input_shape: tuple[int, ...] = INPUT_SHAPE) -> list[tuple]:
    """Per-operator Vela estimates ``(original op, NPU op, count, MACs, cycles)``, from a second,
    report-only lowering with ``--verbose-performance`` (the shipped ``.pte`` keeps the plain spec)."""
    spec = compile_spec(args.target, args.system_config, args.memory_mode, args.config_ini,
                        extra_flags=["--verbose-performance"])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        lower(quantized, spec, input_shape)
    agg: dict[tuple[str, str], list[int]] = {}
    for line in log.getvalue().splitlines():
        f = line.split()
        # Original Operator, NNG Operator, Target, Staging, Peak%, Op Cycles, ..., MAC Count (13th), ...
        if len(f) >= 15 and f[2] == "NPU" and f[5].isdigit() and f[12].isdigit():
            a = agg.setdefault((f[0], f[1]), [0, 0, 0])
            a[0] += 1; a[1] += int(f[12]); a[2] += int(f[5])
    return sorted(((o, n, *v) for (o, n), v in agg.items()), key=lambda r: -r[4])


def mac_reconciliation(per_layer: dict[str, int], vela_rows: list[tuple], kern_t: int = 64) -> str:
    """Explain the analytic vs Vela MAC totals from the per-operator breakdown (``kern_t`` = the
    temporal conv's kernel length, for the text)."""
    if not vela_rows:
        return "Vela per-operator breakdown not captured."
    analytic = sum(per_layer.values())
    conv = sum(r[3] for r in vela_rows if r[1] in ("Conv2D", "DepthwiseConv2D", "FullyConnected", "MatMul"))
    other = sum(r[3] for r in vela_rows) - conv
    text = (f"Analytic MACs (conv + linear, every output position x kernel taps x input channels per group, "
            f"padding taps included) = {analytic:,}. Vela total = {conv + other:,} = {conv:,} in "
            f"Conv2D/DepthwiseConv2D rows + {other:,} that Vela also counts for transposes, slices and "
            f"avg-pools. ")
    gap = analytic - conv
    half_t = per_layer.get("conv_temporal", 0) // 2
    if gap == half_t and gap:
        text += (f"The conv gap {gap:,} is exactly half the temporal conv (1x{kern_t}, {per_layer['conv_temporal']:,} "
                 "analytic): Vela reports that layer at 50 %, and every other conv/linear layer matches exactly. "
                 "Padding does not explain it (padding taps would not give an exact 1/2); Ethos-U85 costing "
                 "lives in Vela's compiled `regor` backend, so the cause is not identified. Treat Vela's MAC "
                 "count as Vela's bookkeeping; the analytic number is the model's arithmetic.")
    else:
        text += f"The conv gap of {gap:,} MACs is unexplained."
    return text


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preprocess_params(data: dict, cfg: dict, subject: int, scale: float, zero_point: int) -> dict:
    from src.data.download import CUE_OFFSET_S
    from src.data.preprocess import design_filter
    from src.data.splits import CALIB_TRIALS
    from src.train import CHANNELS

    fs = float(cfg["fs"])
    tmin, tmax = float(cfg["window"]["tmin"]), float(cfg["window"]["tmax"])
    sos = design_filter(fs, cfg["filter"]["low"], cfg["filter"]["high"], cfg["filter"]["order"])
    return {
        "subject": subject,
        "fs": fs,
        "n_channels": len(CHANNELS),
        "n_samples": int(round((tmax - tmin) * fs)),
        "channel_order": CHANNELS,
        "units": "uV",
        "sos": sos.tolist(),
        "filter": {"type": "butterworth_bandpass", "low_hz": cfg["filter"]["low"], "high_hz": cfg["filter"]["high"],
                   "order": cfg["filter"]["order"], "causal": True, "applied": "scipy.signal.sosfilt on the "
                   "continuous signal, state carried across samples (direct form II transposed)"},
        "tmin": tmin,
        "tmax": tmax,
        "cue_offset_s": CUE_OFFSET_S,
        "ea_matrix": np.asarray(data["A_target"]).tolist(),
        "ea_calibration": {"subject": subject, "session": 2, "trials": f"1-{CALIB_TRIALS}",
                           "labels_used": False},
        "input_scale": float(data["input_scale"]),
        "input_quant_scale": float(scale),
        "input_zero_point": int(zero_point),
        "input_formula": "x_model = (ea_matrix @ x_uV) / input_scale; "
                         "x_int8 = clamp(round(x_model / input_quant_scale) + input_zero_point, -128, 127)",
        "class_map": CLASS_MAP,
        "output": "logits (1, 2); argmax = class, no softmax in the model",
    }


def preprocess_params_v2(run: dict, data: dict, subject: int, scale: float, zero_point: int) -> dict:
    """``preprocess_params.json`` for a v2 student: the board pipeline from 250 Hz UART samples to
    the int8 ``(1, 1, 22, 256)`` model input (mirrors ``src.data.cache125.windows_2a``)."""
    from src.data import cache125
    from src.data.download import CUE_OFFSET_S
    from src.data.preprocess import design_filter
    from src.train import CHANNELS

    cfg, params = run["cfg"], run["params"]
    filt = cfg.get("filter") or params.get("filter") or cache125.FILTER
    fs, fs_model = cache125.FS_2A, float(cfg.get("fs_model", cache125.FS_MODEL))
    factor = int(round(fs / fs_model))
    window, start = int(run["window"]), int(cfg.get("window_start", cache125.WINDOW_START))
    n_calib = int(cfg.get("test", {}).get("calib_trials", 24))
    sos = design_filter(fs, filt["low"], filt["high"], filt["order"])
    student = params.get("student") if isinstance(params.get("student"), dict) else {}
    return {
        "version": "v2",
        "subject": subject,
        "fs": fs,
        "fs_model": fs_model,
        "n_channels": len(CHANNELS),
        "n_samples": window,
        "input_shape": list(run["input_shape"]),
        "channel_order": CHANNELS,
        "units": "uV",
        "reference": "common_average_22",
        "reference_applied": "subtract the per-sample mean of the 22 channels at 250 Hz, before the filter",
        "sos": sos.tolist(),
        "filter": {"type": "butterworth_bandpass", "low_hz": filt["low"], "high_hz": filt["high"],
                   "order": filt["order"], "fs_hz": fs, "causal": True, "applied": "scipy.signal.sosfilt at "
                   "250 Hz on the continuous common-average-referenced signal, state carried across samples "
                   "(direct form II transposed)"},
        "decimation": {"factor": factor, "phase": 0, "applied_after": "sos_filter_250hz",
                       "note": "keep filtered 250 Hz samples 0, 2, 4, ... of the continuous stream; the cue "
                               "sample at 125 Hz is cue_250 // 2"},
        "window": {"start_samples_after_cue_125hz": start, "length": window,
                   "start_s_after_cue": start / fs_model, "end_s_after_cue": (start + window) / fs_model},
        "cue_offset_s": CUE_OFFSET_S,
        "ea_matrix": np.asarray(data["ea"][subject]).tolist(),
        "ea_matrices": {str(s): np.asarray(A).tolist() for s, A in sorted(data["ea"].items())},
        "ea_calibration": {"subject": subject, "session": 2, "trials": f"1-{n_calib}", "labels_used": False,
                           "window": "the same 256-sample 125 Hz window as the model input",
                           "eps": cfg.get("ea_eps", params.get("ea_eps"))},
        "input_scale": float(run["input_scale"]),
        "input_quant_scale": float(scale),
        "input_zero_point": int(zero_point),
        "input_formula": "x_model = (ea_matrix @ x_uV_125hz_window) / input_scale; "
                         "x_int8 = clamp(round(x_model / input_quant_scale) + input_zero_point, -128, 127)",
        "class_map": CLASS_MAP,
        "output": "logits (1, 2); argmax = class, no softmax in the model",
        "student": {"name": student.get("name"), **run["widths"]},
    }


def write_op_placement(path: Path, info, top_level: list[str], pte_ops: list[str], meta: dict) -> dict:
    rows = sorted(info.delegation_by_operator.values(), key=lambda b: b.op_type)
    conv_linear_cpu = [b.op_type for b in rows
                       if re.search(r"conv|linear|addmm|mm\b", b.op_type) and b.non_delegated > 0]
    lines = [
        "# Operator placement — TinyEEGNet int8 on Ethos-U85",
        "",
        f"Generated by `python -m src.export` on {date.today().isoformat()} from `{meta['checkpoint']}` "
        f"(subject {meta['subject']}), target `{meta['target']}`. Counts come from "
        "`executorch.devtools.backend_debug.get_delegation_info` on the lowered edge program: "
        "**Ethos-U** = inside the `EthosUBackend` delegate (compiled by Vela), **CPU** = left in the "
        "ExecuTorch graph and run by the Cortex-M55 portable/quantized kernels.",
        "",
        f"- Delegated subgraphs: **{info.num_delegated_subgraphs}**",
        f"- Ops on Ethos-U: **{info.num_delegated_nodes}**; ops on CPU: **{info.num_non_delegated_nodes}**",
        f"- All conv/linear on Ethos-U: **{'yes' if not conv_linear_cpu else 'NO: ' + ', '.join(conv_linear_cpu)}**",
        "",
        "| Operator | Ethos-U | CPU |",
        "|---|---:|---:|",
        *[f"| `{b.op_type}` | {b.delegated} | {b.non_delegated} |" for b in rows],
        "",
        "## Top-level program (execution order)",
        "",
        "```",
        *top_level,
        "```",
        "",
        "## Operators the CPU runtime needs (`.pte` operator table)",
        "",
        *([f"- `{op}`" for op in pte_ops] or ["- none"]),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return {"delegated": info.num_delegated_nodes, "cpu": info.num_non_delegated_nodes,
            "subgraphs": info.num_delegated_subgraphs, "conv_linear_on_cpu": conv_linear_cpu}


def write_deployment(path: Path, d: dict) -> None:
    lines = [
        "# Deployment — export subject",
        "",
        f"Generated by `python -m src.export` on {date.today().isoformat()}. Host numbers come from the "
        "quantized (fake-quant) graph module; the delegated `.pte` runs only on the Ethos-U "
        "(FVP latency and arena size are added from Emulator_Deploy).",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Checkpoint | `{d['checkpoint']}` |",
        f"| Export subject | A{d['subject']:02d} ({d['mode']}, seed {d['seed']}) |",
        f"| Target / Vela | `{d['target']}`, system config `{d['system_config']}`, memory mode `{d['memory_mode']}`, "
        f"ini `{d['config_ini']}` |",
        f"| Parameters | {d['params']:,} |",
        f"| MACs per window | {d['macs']:,} |",
        f"| `.pte` size | {d['pte_bytes']:,} B ({d['pte_bytes'] / 1024:.1f} KiB) |",
        f"| Ops on Ethos-U / CPU | {d['ops']['delegated']} / {d['ops']['cpu']} "
        f"({d['ops']['subgraphs']} delegated subgraph(s)); see [op_placement.md](op_placement.md) |",
        f"| CPU operators in `.pte` | {', '.join(f'`{o}`' for o in d['pte_ops']) or 'none'} |",
        f"| Input qparams | scale {d['q_scale']:.6g}, zero point {d['q_zp']} (data scale `input_scale` "
        f"{d['input_scale']:.6g}) |",
        f"| int8 accuracy, {d['n_vectors']} test vectors | {d['acc_int8_vec']:.1%} "
        f"(float {d['acc_float_vec']:.1%}) |",
        f"| Max \\|logit\\| diff int8 vs float, {d['n_vectors']} vectors | {d['maxdiff_vec']:.4f} |",
        f"| int8 / float accuracy, full test block ({d['n_test']} trials) | {d['acc_int8_test']:.1%} / "
        f"{d['acc_float_test']:.1%} |",
        f"| Max \\|logit\\| diff, full test block | {d['maxdiff_test']:.4f} |",
        f"| Argmax agreement int8 vs float, full test block | {d['agree_test']:.1%} |",
        *d.get("extra_rows", []),
        "",
        "Vela summary (estimates, not FVP measurements):",
        "",
        "```",
        *(d["vela_summary"] or ["(not captured)"]),
        "```",
        "",
        "MACs per layer: " + ", ".join(f"{k} {v:,}" for k, v in d["macs_per_layer"].items()),
        "",
        "## Vela per-operator estimate",
        "",
        "From a report-only Vela pass with `--verbose-performance`, grouped by operator. Cycles are "
        "Vela's static estimate for the configuration above, not FVP measurements.",
        "",
        "| Original op | NPU op | Count | MACs | Cycles | Cycles % |",
        "|---|---|---:|---:|---:|---:|",
        *[f"| {o} | {n} | {c} | {m:,} | {cy:,} | {cy / max(1, d['vela_cycles']):.1%} |"
          for o, n, c, m, cy in d["vela_rows"]],
        f"| **Total** | | {sum(r[2] for r in d['vela_rows'])} | {sum(r[3] for r in d['vela_rows']):,} | "
        f"{d['vela_cycles']:,} | |",
        "",
        "**MAC reconciliation.** " + d["mac_note"],
        "",
        "## Files",
        "",
        "| File | Bytes | SHA-256 |",
        "|---|---:|---|",
        *[f"| `{name}` | {size:,} | `{digest}` |" for name, size, digest in d["files"]],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _prepare_v01(args: argparse.Namespace) -> dict:
    """v0.1 run: BCI IV 2a split from ``src.train.get_data``, (1, 1, 22, 500) input."""
    from src.train import get_data, load_config, to_model_input

    cfg = load_config()
    data = get_data(args.mode, args.subject, args.seed, "session2", cfg["training"]["val_fraction"])
    model = load_float_model(args.checkpoint)
    return {
        "model": model, "calib": calibration_windows(data, args.seed), "input_shape": INPUT_SHAPE,
        "X_test": to_model_input(data["test"][0], data["input_scale"]).numpy(), "y_test": data["test"][1],
        "input_scale": float(data["input_scale"]), "extra_files": {}, "extra_rows": [],
        "params": lambda scale, zp: preprocess_params(data, cfg, args.subject, scale, zp),
    }


def _prepare_v2(args: argparse.Namespace) -> dict:
    """v2 student run: widths from ``student.json``, calibration from the run, 2a test block of
    ``args.subject`` from ``splits125.v2_data`` scaled with the run's ``input_scale``."""
    from src.train import to_model_input

    run_dir = Path(args.run_dir)
    run = v2_run_config(run_dir)
    run["widths"] = student_widths(Path(args.checkpoint).parent) or run["widths"]
    calib = np.load(run_dir / "calibration.npz")["x"].astype(np.float32)
    if calib.shape[1:] != run["input_shape"][1:]:
        raise ValueError(f"{run_dir}/calibration.npz windows {calib.shape[1:]} != model input "
                         f"{run['input_shape'][1:]}")
    data = load_v2_data(run["seed"], run["cfg"])
    check = check_v2_data(data, run, str(run_dir))
    if args.subject not in data["test"]:
        raise ValueError(f"subject {args.subject} not in the v2 test subjects {sorted(data['test'])}")
    X, y = data["test"][args.subject]
    student = run["params"].get("student") if isinstance(run["params"].get("student"), dict) else {}
    widths_txt = ", ".join(f"{k} {v}" for k, v in run["widths"].items())
    return {
        "model": load_float_model(args.checkpoint, run["window"], run["widths"]), "calib": calib,
        "input_shape": run["input_shape"], "X_test": to_model_input(X, run["input_scale"]).numpy(),
        "y_test": np.asarray(y, dtype=np.int64), "input_scale": run["input_scale"],
        "extra_files": {"student.json": json.dumps(run["widths"], indent=1) + "\n"},
        "extra_rows": [
            f"| Input shape | {tuple(run['input_shape'])} (125 Hz, 256 samples) |",
            f"| Student | {student.get('name', 'v2')} ({widths_txt}) |",
            f"| Calibration | `{run_dir / 'calibration.npz'}` ({len(calib)} windows) |",
            f"| `input_scale` | {run['input_scale']:.6g} from params.json (v2_data now gives "
            f"{check['input_scale_v2_data']:.6g}) |",
        ],
        "params": lambda scale, zp: preprocess_params_v2(run, data, args.subject, scale, zp),
    }


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    art = out.parent
    art.mkdir(parents=True, exist_ok=True)
    results = Path(args.results_dir)

    prep = _prepare_v2(args) if args.v2 else _prepare_v01(args)
    model, calib, shape = prep["model"], prep["calib"], prep["input_shape"]
    np.savez(art / "calibration.npz", x=calib)

    spec = compile_spec(args.target, args.system_config, args.memory_mode, args.config_ini)
    quantized = quantize(model, calib, spec, input_shape=shape)
    scale, zp = input_qparams(quantized)

    vela_log = io.StringIO()
    with contextlib.redirect_stdout(vela_log):
        info, top_level, program = lower(quantized, spec, shape)
    print(vela_log.getvalue(), end="")
    vela_summary = _vela_summary(vela_log.getvalue())
    pte = bytes(program.buffer)
    out.write_bytes(pte)
    shutil.copyfile(args.checkpoint, art / "tinyeegnet_float.pt")
    for name, text in prep["extra_files"].items():
        (art / name).write_text(text)

    # Test vectors: the first 20 scored trials (session 2 trials 25-44), EA from trials 1-24.
    X_test, y_test = prep["X_test"], prep["y_test"]
    x_q = quantize_input(X_test[:N_VECTORS], scale, zp)
    x_dq = (x_q.astype(np.float32) - zp) * np.float32(scale)
    logits_vec = int8_logits(quantized, x_dq)
    np.savez(art / "test_vectors.npz", x=x_q, logits=logits_vec, y=y_test[:N_VECTORS].astype(np.int64))

    params = prep["params"](scale, zp)
    (art / "preprocess_params.json").write_text(json.dumps(params, indent=1) + "\n")

    float_vec = float_logits(model, X_test[:N_VECTORS])
    int8_test = int8_logits(quantized, X_test)
    float_test = float_logits(model, X_test)
    ops = write_op_placement(results / "op_placement.md", info, top_level, pte_operators(pte),
                             {"checkpoint": args.checkpoint, "subject": args.subject, "target": args.target})
    macs, macs_per_layer = count_macs(model, shape)
    vela_rows = vela_breakdown(quantized, args, shape)
    names = [out.name, "tinyeegnet_float.pt", "calibration.npz", *prep["extra_files"], "test_vectors.npz",
             "preprocess_params.json"]
    d = {
        "checkpoint": args.checkpoint, "subject": args.subject, "mode": args.mode, "seed": args.seed,
        "target": args.target,
        "system_config": args.system_config or "Vela default", "memory_mode": args.memory_mode or "Vela default",
        "config_ini": args.config_ini or "Arm/vela.ini (bundled)", "vela_summary": vela_summary,
        "vela_rows": vela_rows, "vela_cycles": sum(r[4] for r in vela_rows),
        "mac_note": mac_reconciliation(macs_per_layer, vela_rows, int(model.conv_temporal.kernel_size[1])),
        "params": sum(p.numel() for p in model.parameters()), "macs": macs, "macs_per_layer": macs_per_layer,
        "pte_bytes": len(pte), "ops": ops, "pte_ops": pte_operators(pte),
        "q_scale": scale, "q_zp": zp, "input_scale": float(prep["input_scale"]),
        "n_vectors": N_VECTORS, "n_test": len(y_test),
        "acc_int8_vec": float(np.mean(logits_vec.argmax(1) == y_test[:N_VECTORS])),
        "acc_float_vec": float(np.mean(float_vec.argmax(1) == y_test[:N_VECTORS])),
        "maxdiff_vec": float(np.abs(logits_vec - float_vec).max()),
        "acc_int8_test": float(np.mean(int8_test.argmax(1) == y_test)),
        "acc_float_test": float(np.mean(float_test.argmax(1) == y_test)),
        "maxdiff_test": float(np.abs(int8_test - float_test).max()),
        "agree_test": float(np.mean(int8_test.argmax(1) == float_test.argmax(1))),
        "files": [(n, (art / n).stat().st_size, sha256(art / n)) for n in names],
        "input_shape": list(shape), "extra_rows": prep["extra_rows"],
    }
    write_deployment(results / "deployment.md", d)
    print(json.dumps({k: v for k, v in d.items() if k not in ("files", "macs_per_layer", "vela_summary", "vela_rows",
                                                             "extra_rows")}, indent=1, default=str))
    return d


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="TinyEEGNet state_dict (model.pt)")
    p.add_argument("--subject", type=int, required=True, help="export subject (1-9)")
    p.add_argument("--out", default="artifacts/tinyeegnet_int8.pte")
    p.add_argument("--mode", choices=["loso", "per-subject"], default=None,
                   help="split the checkpoint was trained with (default: metrics.json beside it, else loso)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed the checkpoint was trained with (default: metrics.json beside it, else 0)")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--system-config", default=None, help="Vela system config (default: Vela's)")
    p.add_argument("--memory-mode", default=None, help="Vela memory mode (default: Vela's)")
    p.add_argument("--config-ini", default=None, help="Vela ini (default: bundled Arm/vela.ini)")
    p.add_argument("--run-dir", default=None,
                   help="v2: student run directory with params.json and calibration.npz (default: the checkpoint's)")
    p.add_argument("--results-dir", default=None,
                   help="op_placement.md / deployment.md (default: results for v0.1, beside --out for v2)")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)
    ckpt_dir = Path(args.checkpoint).parent
    args.run_dir = args.run_dir or str(ckpt_dir)
    args.v2 = is_v2_run(ckpt_dir) or is_v2_run(args.run_dir)
    if args.v2:
        run_cfg = v2_run_config(args.run_dir)
        args.mode = args.mode or f"v2 student {(run_cfg['params'].get('student') or {}).get('name', '')}".rstrip()
        if args.seed is not None and args.seed != run_cfg["seed"]:
            p.error(f"{args.run_dir}/params.json is seed {run_cfg['seed']}, not --seed {args.seed}")
        args.seed = run_cfg["seed"]
        args.results_dir = args.results_dir or str(Path(args.out).parent)
        return args
    args.results_dir = args.results_dir or "results"
    # The seed selects the train/validation split, hence the training windows the PTQ calibration is
    # drawn from; take it (and the mode) from the checkpoint's own run unless given explicitly.
    meta_file = Path(args.checkpoint).with_name("metrics.json")
    meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
    if meta and int(meta["subject"]) != args.subject:
        p.error(f"{meta_file} is subject {meta['subject']}, not --subject {args.subject}")
    args.mode = args.mode or meta.get("mode", "loso")
    args.seed = int(meta.get("seed", 0)) if args.seed is None else args.seed
    return args


if __name__ == "__main__":
    run(parse_args())
