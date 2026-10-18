# SPDX-License-Identifier: Apache-2.0
"""Post-training int8 quantization of TinyEEGNet for the Ethos-U85, and the host-side int8 check.

Mirrors ``CMSIS-Executorch/create_ai_layer.py:export_model``: ``EthosUQuantizer`` with the
per-channel symmetric config, ``prepare_pt2e`` -> calibrate -> ``convert_pt2e``. Calibration
uses 200 real aligned+scaled training windows (the same draw ``src.train`` saves as
``calibration.npz``), never target-subject data.

The delegated ``.pte`` does not run on the x86 runtime (the Ethos-U delegate needs the NPU),
so the host-side int8 numbers come from the quantized (fake-quant) graph module returned by
:func:`quantize`; the target check is the FVP run in Emulator_Deploy.

    python -m src.quantize --run runs/loso_s1_seed0

writes ``runs/loso_s1_seed0/int8_predictions.npz`` (``y_true``, ``logits``, ``test_trials``)
for the same test block as ``predictions.npz``.

v2 student runs (``student.json`` in the run directory, or ``params.json`` with a 256-sample
window) are detected automatically:

    python -m src.quantize --run runs/v2/student_wide32_seed0

builds ``TinyEEGNet(n_samples=256, **student.json)``, calibrates with the run's own
``calibration.npz`` (200 scaled canonical training windows), scores every BCI IV 2a test subject
from ``src.data.splits125.v2_data(seed, config)`` (both from ``params.json``, the input scale
too) and writes ``int8_predictions.npz`` (``subjects``, ``y_true_{s}``, ``logits_{s}`` as in
``predictions.npz``, plus ``float_logits_{s}``) and ``int8_metrics.json`` (per-subject and mean
int8 / float KPIs, argmax agreement).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from src.models.tinyeegnet import INPUT_SHAPE, STUDENT_KEYS, TinyEEGNet

N_CALIB = 200
DEFAULT_TARGET = "ethos-u85-256"
V2_WINDOW = 256


def compile_spec(target: str = DEFAULT_TARGET, system_config: str | None = None,
                 memory_mode: str | None = None, config_ini: str | None = None, extra_flags=None):
    """``EthosUCompileSpec`` as ``create_ai_layer.py:compile_spec`` builds it from the mlops node.

    Without ``config_ini`` the spec uses Vela's bundled ``Arm/vela.ini``; the Alif board
    configuration (``ensemble_vela.ini``, ``Ethos_U85_SRAM_MRAM`` / ``Shared_Sram``) only
    exists after ``cbuild setup`` in the starter, which re-exports the model itself.
    """
    from executorch.backends.arm.ethosu import EthosUCompileSpec

    kwargs = {"target": target, "system_config": system_config, "memory_mode": memory_mode,
              "extra_flags": list(extra_flags or [])}
    if config_ini:
        kwargs["config_ini"] = config_ini
    return EthosUCompileSpec(**kwargs)


def load_float_model(checkpoint: str | Path, n_samples: int = INPUT_SHAPE[-1],
                     widths: dict[str, int] | None = None) -> torch.nn.Module:
    """TinyEEGNet in eval mode with the weights of a ``model.pt`` state_dict; ``n_samples`` and
    ``widths`` (``student.json`` keys) default to the v0.1 model."""
    model = TinyEEGNet(n_samples=n_samples, **(widths or {}))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


# ----------------------------------------------------------------------------- v2 runs
def student_widths(directory: str | Path) -> dict[str, int]:
    """Layer widths from ``student.json`` in ``directory`` (only ``STUDENT_KEYS``), or ``{}``."""
    path = Path(directory) / "student.json"
    if not path.is_file():
        return {}
    cfg = json.loads(path.read_text())
    return {k: int(cfg[k]) for k in STUDENT_KEYS if k in cfg}


def run_params(directory: str | Path) -> dict:
    """``params.json`` of a run directory, or ``{}``."""
    path = Path(directory) / "params.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def is_v2_run(directory: str | Path) -> bool:
    """A v2 student run: ``student.json`` present, or ``params.json`` with a 256-sample window."""
    directory = Path(directory)
    if (directory / "student.json").is_file():
        return True
    params = run_params(directory)
    return int(params.get("window_samples", params.get("window", 0)) or 0) == V2_WINDOW


def v2_run_config(directory: str | Path) -> dict:
    """What a v2 export or int8 check needs from a run directory: ``window``, ``input_shape``,
    ``widths``, ``seed``, ``cfg`` (the effective config the run was trained with), ``input_scale``
    (the training value), ``ea_matrices`` and ``params``."""
    directory = Path(directory)
    params = run_params(directory)
    if not params:
        raise FileNotFoundError(f"{directory}: no params.json")
    window = int(params.get("window_samples", V2_WINDOW))
    widths = student_widths(directory)
    if not widths and isinstance(params.get("student"), dict):
        widths = {k: int(params["student"][k]) for k in STUDENT_KEYS if k in params["student"]}
    cfg = params.get("config")
    if not cfg:
        from src.data.splits125 import load_config as load_v2_config

        cfg = load_v2_config()
    return {
        "window": window,
        "input_shape": (1, 1, int(params.get("n_channels", 22)), window),
        "widths": widths,
        "seed": int(params.get("seed", 0)),
        "cfg": cfg,
        "input_scale": float(params["input_scale"]),
        "ea_matrices": {int(s): np.asarray(A, dtype=np.float64) for s, A in params.get("ea_matrices", {}).items()},
        "params": params,
    }


def load_v2_data(seed: int, cfg: dict) -> dict:
    """``src.data.splits125.v2_data`` (looked up at call time so tests can monkeypatch it)."""
    from src.data import splits125

    return splits125.v2_data(seed, cfg)


def check_v2_data(data: dict, run: dict, where: str = "") -> dict:
    """Compare the regenerated data with the run's training record; warns on a mismatch.
    The model is always fed with the run's own ``input_scale`` (training value)."""
    scale_now = float(data["input_scale"])
    ea_diff = {s: float(np.abs(np.asarray(data["ea"][s]) - run["ea_matrices"][s]).max())
               for s in data.get("ea", {}) if s in run["ea_matrices"]}
    if not np.isclose(scale_now, run["input_scale"], rtol=1e-5):
        warnings.warn(f"{where}: v2_data input_scale {scale_now:.6f} != params.json {run['input_scale']:.6f} "
                      "(training data changed since the run); using the params.json value", stacklevel=2)
    if ea_diff and max(ea_diff.values()) > 1e-6:
        warnings.warn(f"{where}: 2a EA matrices differ from params.json (max {max(ea_diff.values()):.3g})",
                      stacklevel=2)
    return {"input_scale_used": run["input_scale"], "input_scale_v2_data": scale_now,
            "ea_max_abs_diff_vs_params": max(ea_diff.values()) if ea_diff else None}


def calibration_windows(data: dict, seed: int = 0, n: int = N_CALIB) -> np.ndarray:
    """``(n, 1, 22, 500)`` float32 aligned+scaled training windows from a ``loso_data`` /
    ``per_subject_data`` dict; the same seeded draw ``src.train`` writes to ``calibration.npz``."""
    from src.train import to_model_input

    X_train = to_model_input(data["train"][0], data["input_scale"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=min(n, len(X_train)), replace=False)
    return X_train[np.sort(idx)].numpy().astype(np.float32)


def quantize(model: torch.nn.Module, calib_windows, spec=None,
             input_shape: tuple[int, ...] | None = None) -> torch.fx.GraphModule:
    """PTQ: export, annotate with ``EthosUQuantizer``, calibrate, convert. Returns the
    quantized (fake-quant, batch 1) GraphModule that ``src.export`` lowers to Ethos-U.
    ``input_shape`` defaults to ``(1, 1, 22, T)`` with ``T`` the calibration windows' length."""
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    spec = spec or compile_spec()
    if input_shape is None:
        first = np.asarray(calib_windows[0])
        input_shape = (*INPUT_SHAPE[:2], *first.shape[-2:])
    samples = [torch.as_tensor(np.asarray(w, dtype=np.float32)).reshape(input_shape) for w in calib_windows]
    # check_guards=False: the default inserts a `_guards_fn` call_module, which the Arm
    # quantizer's DecomposeAvgPool2dPass cannot retrace ("call_module is not supported").
    graph = torch.export.export(model.eval(), (samples[0],)).module(check_guards=False)
    quantizer = EthosUQuantizer(spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for sample in samples:
            prepared(sample)
    return convert_pt2e(prepared)


def input_qparams(quantized: torch.fx.GraphModule) -> tuple[float, int]:
    """``(scale, zero_point)`` of the quantize op applied to the model input."""
    placeholder = next(n for n in quantized.graph.nodes if n.op == "placeholder")
    for user in placeholder.users:
        if "quantize_per_tensor" in str(user.target):
            scale, zero_point = user.args[1], user.args[2]
            return float(scale), int(zero_point)
    raise RuntimeError("model input is not followed by a quantize_per_tensor op")


def quantize_input(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """float model input -> int8, as the quantize op does it (round half to even, clamp)."""
    q = np.round(np.asarray(x, dtype=np.float32) / np.float32(scale)) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


@torch.no_grad()
def int8_logits(quantized: torch.fx.GraphModule, X: np.ndarray | torch.Tensor) -> np.ndarray:
    """Dequantized int8 logits ``(n, 2)`` float32 of the quantized model for ``(n, 1, 22, T)``
    float inputs (the exported graph has a static batch of 1, so samples run one by one)."""
    X = torch.as_tensor(np.asarray(X, dtype=np.float32))
    return np.concatenate([quantized(X[i : i + 1]).numpy() for i in range(len(X))]).astype(np.float32)


@torch.no_grad()
def float_logits(model: torch.nn.Module, X: np.ndarray | torch.Tensor) -> np.ndarray:
    return model.eval()(torch.as_tensor(np.asarray(X, dtype=np.float32))).numpy().astype(np.float32)


def run_int8_predictions(run_dir: str | Path, checkpoint: str | Path | None = None) -> dict:
    """Quantize the run's checkpoint with its own training-data calibration windows and write
    ``int8_predictions.npz`` (dequantized int8 logits on the run's test block). v2 student runs
    go to :func:`run_v2_int8_predictions` (returns its mean int8 KPIs)."""
    from src.eval import compute_kpis
    from src.train import get_data, load_config, to_model_input

    run_dir = Path(run_dir)
    if is_v2_run(run_dir):
        return run_v2_int8_predictions(run_dir, checkpoint)["int8_mean"]
    meta = json.loads((run_dir / "metrics.json").read_text())
    if meta.get("model", "tinyeegnet") != "tinyeegnet":
        raise ValueError(f"{run_dir}: only TinyEEGNet runs are quantized, got {meta['model']}")
    cfg = load_config()
    data = get_data(meta["mode"], int(meta["subject"]), int(meta["seed"]), "session2",
                    cfg["training"]["val_fraction"])
    model = load_float_model(checkpoint or run_dir / "model.pt")
    quantized = quantize(model, calibration_windows(data, int(meta["seed"])))
    logits = int8_logits(quantized, to_model_input(data["test"][0], data["input_scale"]))
    y = data["test"][1]
    np.savez(run_dir / "int8_predictions.npz", y_true=y, logits=logits, test_trials=data["test_trials"])
    return compute_kpis(y, logits)


def run_v2_int8_predictions(run_dir: str | Path, checkpoint: str | Path | None = None,
                            data: dict | None = None) -> dict:
    """v2 student run: PTQ with the run's ``calibration.npz``, int8 and float logits for every 2a
    test subject; writes ``int8_predictions.npz`` and ``int8_metrics.json`` and returns the latter."""
    from src.eval import KPI_KEYS, compute_kpis
    from src.train import to_model_input

    run_dir = Path(run_dir)
    run = v2_run_config(run_dir)
    checkpoint = Path(checkpoint or run_dir / "model.pt")
    model = load_float_model(checkpoint, run["window"], run["widths"])
    calib = np.load(run_dir / "calibration.npz")["x"].astype(np.float32)
    if calib.shape[1:] != run["input_shape"][1:]:
        raise ValueError(f"{run_dir}/calibration.npz windows {calib.shape[1:]} != model input "
                         f"{run['input_shape'][1:]}")
    quantized = quantize(model, calib, input_shape=run["input_shape"])
    q_scale, q_zp = input_qparams(quantized)
    data = data if data is not None else load_v2_data(run["seed"], run["cfg"])
    check = check_v2_data(data, run, str(run_dir))
    scale = run["input_scale"]
    subjects = sorted(int(s) for s in data["test"])
    arrays: dict[str, np.ndarray] = {"subjects": np.array(subjects)}
    per: dict[str, dict] = {}
    for s in subjects:
        X, y = data["test"][s]
        X = to_model_input(X, scale).numpy()
        q, f = int8_logits(quantized, X), float_logits(model, X)
        arrays[f"y_true_{s}"] = np.asarray(y, dtype=np.int64)
        arrays[f"logits_{s}"] = q
        arrays[f"float_logits_{s}"] = f
        per[str(s)] = {"int8": compute_kpis(y, q), "float": compute_kpis(y, f), "n": int(len(y)),
                       "agreement": float(np.mean(q.argmax(1) == f.argmax(1))),
                       "max_abs_logit_diff": float(np.abs(q - f).max())}
    np.savez(run_dir / "int8_predictions.npz", **arrays)

    # Float logits here should reproduce the run's own predictions.npz (same data, same scale).
    pred_diff = None
    if (run_dir / "predictions.npz").is_file():
        z = np.load(run_dir / "predictions.npz")
        diffs = [float(np.abs(z[f"logits_{s}"] - arrays[f"float_logits_{s}"]).max())
                 for s in subjects if f"logits_{s}" in z.files]
        pred_diff = max(diffs) if diffs else None
    mean = lambda prec: {k: float(np.mean([per[str(s)][prec][k] for s in subjects])) for k in KPI_KEYS}  # noqa: E731
    all_q = np.concatenate([arrays[f"logits_{s}"] for s in subjects])
    all_f = np.concatenate([arrays[f"float_logits_{s}"] for s in subjects])
    metrics = {
        "run": str(run_dir), "checkpoint": str(checkpoint), "seed": run["seed"], "widths": run["widths"],
        "input_shape": list(run["input_shape"]), "n_calib": int(len(calib)),
        "input_quant_scale": q_scale, "input_zero_point": q_zp, **check,
        "subjects": subjects, "test": per,
        "int8_mean": mean("int8"), "float_mean": mean("float"),
        "int8_minus_float_accuracy": mean("int8")["accuracy"] - mean("float")["accuracy"],
        "agreement_mean": float(np.mean([per[str(s)]["agreement"] for s in subjects])),
        "agreement_all": float(np.mean(all_q.argmax(1) == all_f.argmax(1))),
        "max_abs_logit_diff": float(np.abs(all_q - all_f).max()),
        "float_vs_predictions_npz_max_abs_diff": pred_diff,
    }
    (run_dir / "int8_metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    return metrics


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory with model.pt and metrics.json "
                   "(v2 student runs: student.json, params.json, calibration.npz)")
    p.add_argument("--checkpoint", default=None, help="override the run's model.pt")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    if is_v2_run(args.run):
        m = run_v2_int8_predictions(args.run, args.checkpoint)
        for prec in ("int8", "float"):
            print(f"{args.run} {prec} mean: " + " ".join(f"{k} {v:.4f}" for k, v in m[f"{prec}_mean"].items()))
        print(f"{args.run} int8-float accuracy {m['int8_minus_float_accuracy']:+.4f}, argmax agreement "
              f"{m['agreement_all']:.4f}, max |logit diff| {m['max_abs_logit_diff']:.4f}")
        return
    kpis = run_int8_predictions(args.run, args.checkpoint)
    print(f"{args.run} int8: " + " ".join(f"{k} {v:.4f}" for k, v in kpis.items()))


if __name__ == "__main__":
    main()
