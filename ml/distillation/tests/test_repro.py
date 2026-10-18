"""Slow fixed-seed reproducibility acceptance test."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from conftest import REPO_ROOT


def _accuracy(payload: Any) -> float | None:
    if isinstance(payload, dict):
        for key in ("test_accuracy", "test_acc"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        test_metrics = payload.get("test")
        if isinstance(test_metrics, dict):
            found = _accuracy(test_metrics)
            if found is not None:
                return found
        for key, value in payload.items():
            if key.lower() == "accuracy" and isinstance(value, (int, float)):
                return float(value)
        for value in payload.values():
            found = _accuracy(value)
            if found is not None:
                return found
    return None


def _recorded_subject_one_seed_zero() -> tuple[Path, float] | None:
    for path in sorted((REPO_ROOT / "runs").glob("**/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        searchable = (str(path) + " " + json.dumps(payload)).lower().replace("-", "_")
        is_loso = "loso" in searchable
        is_tinyeegnet = payload.get("model", "tinyeegnet") == "tinyeegnet"
        is_subject_one = any(token in searchable for token in ('"subject": 1', "subject_1", "subject1", "a01"))
        is_seed_zero = any(token in searchable for token in ('"seed": 0', "seed_0", "seed0"))
        value = _accuracy(payload)
        if is_loso and is_tinyeegnet and is_subject_one and is_seed_zero and value is not None:
            return path, value / 100.0 if value > 1.0 else value
    return None


@pytest.mark.slow
@pytest.mark.data
def test_loso_subject_one_seed_zero_reproduces_recorded_accuracy(tmp_path: Path) -> None:
    recorded = _recorded_subject_one_seed_zero()
    if recorded is None:
        pytest.skip("no recorded LOSO subject-1 seed-0 runs/**/metrics.json")
    recorded_path, expected_accuracy = recorded
    output = tmp_path / "loso_subject1_seed0"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--mode",
            "loso",
            "--subject",
            "1",
            "--seed",
            "0",
            "--out",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=14_400,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    actual_path = output / "metrics.json"
    assert actual_path.is_file(), "training did not write <out>/metrics.json"
    actual_payload = json.loads(actual_path.read_text(encoding="utf-8"))
    actual_accuracy = _accuracy(actual_payload)
    assert actual_accuracy is not None, "rerun metrics.json has no accuracy"
    actual_accuracy = actual_accuracy / 100.0 if actual_accuracy > 1.0 else actual_accuracy
    assert actual_accuracy == pytest.approx(expected_accuracy, abs=0.01), (
        f"rerun accuracy {actual_accuracy:.4f} differs by more than one point from "
        f"{expected_accuracy:.4f} in {recorded_path.relative_to(REPO_ROOT)}"
    )
