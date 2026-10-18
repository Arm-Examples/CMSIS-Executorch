"""v2 student quantization and export: run detection, MAC counting, int8 predictions, Ethos-U export."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import REPO_ROOT

from src.models.tinyeegnet import INPUT_SHAPE, TinyEEGNet

WIDE32 = {"F1": 16, "D": 2, "F2": 32, "kern_t": 32, "kern_s": 16}
V2_SHAPE = (1, 1, 22, 256)


def _executorch_available() -> bool:
    return importlib.util.find_spec("executorch") is not None


def _fake_run(root: Path, subjects=(1, 2), n_calib: int = 8, seed: int = 0) -> Path:
    """A v2 student run directory with random weights and random calibration windows."""
    run = root / "student_wide32_fake"
    run.mkdir(parents=True)
    torch.manual_seed(0)
    torch.save(TinyEEGNet(n_samples=256, **WIDE32).eval().state_dict(), run / "model.pt")
    (run / "student.json").write_text(json.dumps(WIDE32))
    rng = np.random.default_rng(20261016)
    np.savez(run / "calibration.npz", x=rng.standard_normal((n_calib, *V2_SHAPE[1:])).astype(np.float32))
    params = {
        "fs": 125.0, "n_channels": 22, "window_samples": 256, "train_window_samples": 306, "crop_offset": 25,
        "input_shape": list(V2_SHAPE), "input_scale": 1.0, "seed": seed,
        "ea_matrices": {str(s): np.eye(22).tolist() for s in subjects},
        "student": {"name": "wide32", **WIDE32},
        "config": {"fs_model": 125, "window": 256, "window_start": 62, "train_window": 306, "crop_offset": 25,
                   "filter": {"low": 7.0, "high": 30.0, "order": 4}, "ea_eps": 0.001,
                   "test": {"dataset": "BNCI2014_001", "subjects": list(subjects), "session": 2,
                            "calib_trials": 24}},
    }
    (run / "params.json").write_text(json.dumps(params))
    return run


def _fake_v2_data(subjects=(1, 2), n_trials: int = 4):
    rng = np.random.default_rng(7)
    test = {s: (rng.standard_normal((n_trials, 22, 256)).astype(np.float32),
                np.arange(n_trials, dtype=np.int64) % 2) for s in subjects}
    calls = []

    def v2_data(seed, cfg=None, loader=None):
        calls.append((seed, cfg))
        return {"test": test, "ea": {s: np.eye(22) for s in subjects}, "input_scale": 1.0,
                "calib": {}, "train": (None, None), "val": (None, None), "meta": {}}

    return v2_data, test, calls


# ----------------------------------------------------------------------------- no executorch needed
def test_v2_run_detection(tmp_path: Path) -> None:
    from src.quantize import is_v2_run

    empty = tmp_path / "empty"
    empty.mkdir()
    assert not is_v2_run(empty)

    student = tmp_path / "student"
    student.mkdir()
    (student / "student.json").write_text(json.dumps(WIDE32))
    assert is_v2_run(student)

    p256 = tmp_path / "p256"
    p256.mkdir()
    (p256 / "params.json").write_text(json.dumps({"window_samples": 256}))
    assert is_v2_run(p256)

    p500 = tmp_path / "p500"
    p500.mkdir()
    (p500 / "params.json").write_text(json.dumps({"window_samples": 500, "fs": 250.0}))
    assert not is_v2_run(p500)

    v01 = REPO_ROOT / "runs" / "loso_s3_seed2"
    if v01.is_dir():
        assert not is_v2_run(v01)


def test_count_macs_wide32_at_256_samples() -> None:
    from src.export import count_macs

    total, per_layer = count_macs(TinyEEGNet(n_samples=256, **WIDE32).eval(), input_shape=V2_SHAPE)
    assert total == 3_176_128
    assert sum(per_layer.values()) == total


def test_count_macs_v01_default_unchanged() -> None:
    from src.export import count_macs

    assert inspect.signature(count_macs).parameters["input_shape"].default == (1, 1, 22, 500) == INPUT_SHAPE
    total, _ = count_macs(TinyEEGNet().eval())
    assert total == 5_884_608  # results/deployment.md, v0.1 release model


def test_v2_parse_args_defaults(tmp_path: Path) -> None:
    from src.export import parse_args

    run = _fake_run(tmp_path)
    out = run / "export" / "tinyeegnet_int8.pte"
    args = parse_args(["--checkpoint", str(run / "model.pt"), "--subject", "1", "--out", str(out)])
    assert args.v2 and args.seed == 0 and args.run_dir == str(run)
    assert Path(args.results_dir) == out.parent, "v2 reports must default beside the .pte, not results/"
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", str(run / "model.pt"), "--subject", "1", "--out", str(out), "--seed", "2"])


# ----------------------------------------------------------------------------- executorch
def test_v2_int8_predictions_on_fake_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not _executorch_available():
        pytest.skip("executorch is not importable")
    from src.data import splits125
    from src.quantize import run_int8_predictions, run_v2_int8_predictions

    run = _fake_run(tmp_path)
    fake, test, calls = _fake_v2_data()
    monkeypatch.setattr(splits125, "v2_data", fake)

    metrics = run_v2_int8_predictions(run)
    assert calls and calls[0][0] == 0 and calls[0][1]["window"] == 256, "seed/config must come from params.json"
    z = np.load(run / "int8_predictions.npz")
    expected = {"subjects"} | {f"{k}_{s}" for s in (1, 2) for k in ("y_true", "logits", "float_logits")}
    assert set(z.files) == expected
    assert z["subjects"].tolist() == [1, 2]
    for s in (1, 2):
        assert z[f"logits_{s}"].shape == (4, 2) and z[f"logits_{s}"].dtype == np.float32
        assert z[f"float_logits_{s}"].shape == (4, 2)
        assert np.isfinite(z[f"logits_{s}"]).all()
        np.testing.assert_array_equal(z[f"y_true_{s}"], test[s][1])

    saved = json.loads((run / "int8_metrics.json").read_text())
    assert saved["input_shape"] == list(V2_SHAPE) and saved["n_calib"] == 8 and saved["widths"] == WIDE32
    assert set(saved["test"]) == {"1", "2"}
    for key in ("int8_mean", "float_mean", "agreement_all", "agreement_mean", "int8_minus_float_accuracy"):
        assert key in saved
    assert 0.0 <= saved["agreement_all"] <= 1.0
    assert metrics["int8_mean"] == saved["int8_mean"]
    # The generic entry point dispatches v2 runs and returns the mean int8 KPIs.
    assert run_int8_predictions(run) == saved["int8_mean"]

    # src.run_v2 reads the file with the same keys as predictions.npz.
    from src.run_v2 import per_subject_npz

    parsed = per_subject_npz(run / "int8_predictions.npz")
    assert set(parsed) == {1, 2} and parsed[1][1].shape == (4, 2)


def test_v2_export_on_fake_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not _executorch_available():
        pytest.skip("executorch is not importable")
    from src.data import splits125
    from src.export import parse_args, run

    run_dir = _fake_run(tmp_path, subjects=(1, 3))
    fake, test, _ = _fake_v2_data(subjects=(1, 3), n_trials=24)
    monkeypatch.setattr(splits125, "v2_data", fake)
    out = tmp_path / "export" / "tinyeegnet_int8.pte"
    d = run(parse_args(["--checkpoint", str(run_dir / "model.pt"), "--subject", "3", "--out", str(out)]))

    art = out.parent
    for name in ("tinyeegnet_int8.pte", "tinyeegnet_float.pt", "calibration.npz", "student.json",
                 "test_vectors.npz", "preprocess_params.json", "op_placement.md", "deployment.md"):
        assert (art / name).is_file(), name
    assert 0 < out.stat().st_size < 100 * 1024
    assert d["macs"] == 3_176_128 and d["input_shape"] == list(V2_SHAPE)
    assert not d["ops"]["conv_linear_on_cpu"]
    assert json.loads((art / "student.json").read_text()) == WIDE32

    with np.load(art / "test_vectors.npz") as v:
        assert v["x"].shape == (20, *V2_SHAPE[1:]) and v["x"].dtype == np.int8
        assert v["logits"].shape == (20, 2) and v["logits"].dtype == np.float32
        np.testing.assert_array_equal(v["y"], test[3][1][:20])
    with np.load(art / "calibration.npz") as c:
        assert c["x"].shape == (8, *V2_SHAPE[1:])

    p = json.loads((art / "preprocess_params.json").read_text())
    assert p["subject"] == 3 and p["fs"] == 250.0 and p["fs_model"] == 125.0 and p["n_samples"] == 256
    assert p["decimation"] == {**p["decimation"], "factor": 2, "phase": 0, "applied_after": "sos_filter_250hz"}
    assert p["reference"] == "common_average_22"
    assert p["window"]["start_samples_after_cue_125hz"] == 62 and p["window"]["length"] == 256
    assert np.asarray(p["sos"]).shape == (4, 6)
    assert np.asarray(p["ea_matrix"]).shape == (22, 22) and set(p["ea_matrices"]) == {"1", "3"}
    assert p["input_scale"] == 1.0 and {"input_quant_scale", "input_zero_point", "channel_order",
                                        "class_map"} <= set(p)
    assert {k: p["student"][k] for k in WIDE32} == WIDE32
