"""Release artifact schema, type, shape, and size contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from conftest import REPO_ROOT


ARTIFACTS = REPO_ROOT / "artifacts"


def _required(path: Path) -> Path:
    if not path.is_file():
        pytest.skip(f"artifact is absent: {path.relative_to(REPO_ROOT)}")
    return path


def test_test_vectors_contract() -> None:
    path = _required(ARTIFACTS / "test_vectors.npz")
    with np.load(path, allow_pickle=False) as vectors:
        assert set(vectors.files) == {"x", "logits", "y"}
        assert vectors["x"].shape == (20, 1, 22, 500)
        assert vectors["x"].dtype == np.int8
        assert vectors["logits"].shape == (20, 2)
        assert vectors["logits"].dtype == np.float32
        assert np.isfinite(vectors["logits"]).all()
        assert vectors["y"].shape == (20,)
        assert vectors["y"].dtype == np.int64
        assert set(np.unique(vectors["y"])).issubset({0, 1})


def _output_quantization_step(quantized) -> float:
    """Return the scale of the final dequantize op (one output int8 step)."""
    candidates = [
        node
        for node in quantized.graph.nodes
        if node.op == "call_function" and "dequantize_per_tensor" in str(node.target)
    ]
    assert candidates, "quantized graph has no output dequantization"
    return float(candidates[-1].args[1])


@pytest.mark.data
def test_test_vectors_match_rebuilt_quantized_model_and_subject_3_labels() -> None:
    if importlib.util.find_spec("executorch") is None:
        pytest.skip("executorch is not importable")
    vectors_path = _required(ARTIFACTS / "test_vectors.npz")
    checkpoint_path = _required(ARTIFACTS / "tinyeegnet_float.pt")
    calibration_path = _required(ARTIFACTS / "calibration.npz")
    params_path = _required(ARTIFACTS / "preprocess_params.json")

    from src.data.splits import CALIB_TRIALS, cached_subject
    from src.quantize import int8_logits, load_float_model, quantize

    params = json.loads(params_path.read_text(encoding="utf-8"))
    assert params.get("subject") == 3, "release test vectors must be for export subject 3"
    quant_scale = float(params["input_quant_scale"])
    zero_point = int(params["input_zero_point"])

    with np.load(calibration_path, allow_pickle=False) as calibration:
        quantized = quantize(load_float_model(checkpoint_path), calibration["x"])
    output_step = _output_quantization_step(quantized)

    with np.load(vectors_path, allow_pickle=False) as vectors:
        dequantized_x = (
            vectors["x"].astype(np.float32) - np.float32(zero_point)
        ) * np.float32(quant_scale)
        rebuilt_logits = int8_logits(quantized, dequantized_x)
        assert_allclose(
            vectors["logits"],
            rebuilt_logits,
            rtol=0,
            atol=output_step + np.finfo(np.float32).eps,
        )

        _, session_2_labels = cached_subject(3)[2]
        expected_y = session_2_labels[CALIB_TRIALS : CALIB_TRIALS + len(vectors["y"])]
        assert_array_equal(vectors["y"], expected_y)


def test_preprocess_params_contract() -> None:
    path = _required(ARTIFACTS / "preprocess_params.json")
    params: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "sos",
        "tmin",
        "tmax",
        "fs",
        "n_samples",
        "ea_matrix",
        "input_scale",
        "input_quant_scale",
        "input_zero_point",
        "channel_order",
        "class_map",
    }
    assert required <= set(params), f"missing keys: {sorted(required - set(params))}"

    sos = np.asarray(params["sos"], dtype=float)
    ea = np.asarray(params["ea_matrix"], dtype=float)
    assert sos.ndim == 2 and sos.shape[1] == 6 and sos.shape[0] > 0
    assert np.isfinite(sos).all()
    assert params["tmin"] == pytest.approx(0.5)
    assert params["tmax"] == pytest.approx(2.5)
    assert params["fs"] == pytest.approx(250.0)
    assert params["n_samples"] == 500
    assert ea.shape == (22, 22)
    assert np.isfinite(ea).all()
    assert np.allclose(ea, ea.T, rtol=0, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(ea) > 0)
    assert isinstance(params["input_scale"], (int, float)) and params["input_scale"] > 0
    assert isinstance(params["input_quant_scale"], (int, float)) and params["input_quant_scale"] > 0
    assert type(params["input_zero_point"]) is int
    assert -128 <= params["input_zero_point"] <= 127
    assert isinstance(params["channel_order"], list) and len(params["channel_order"]) == 22
    assert len(set(params["channel_order"])) == 22
    assert isinstance(params["class_map"], dict) and len(params["class_map"]) == 2
    class_names = " ".join(
        str(value).lower() for item in params["class_map"].items() for value in item
    )
    assert "left" in class_names and "right" in class_names


def test_pte_is_under_100_kib() -> None:
    path = _required(ARTIFACTS / "tinyeegnet_int8.pte")
    assert 0 < path.stat().st_size < 100 * 1024

