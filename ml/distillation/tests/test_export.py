"""ExecuTorch export and Ethos-U delegation acceptance checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT


def _executorch_available() -> bool:
    return importlib.util.find_spec("executorch") is not None


def _checkpoint() -> Path | None:
    candidates = sorted((REPO_ROOT / "runs").glob("**/model.pt"))
    fallback = REPO_ROOT / "artifacts" / "tinyeegnet_float.pt"
    return candidates[0] if candidates else fallback if fallback.is_file() else None


def test_export_cli_succeeds(tmp_path: Path) -> None:
    if not _executorch_available():
        pytest.skip("executorch is not importable")
    checkpoint = _checkpoint()
    if checkpoint is None:
        pytest.skip("no trained checkpoint artifact is available")

    output = tmp_path / "tinyeegnet_int8.pte"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.export",
            "--checkpoint",
            str(checkpoint),
            "--subject",
            "1",
            "--out",
            str(output),
            "--results-dir",
            str(tmp_path / "results"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert output.is_file()
    assert 0 < output.stat().st_size < 100 * 1024


def test_all_convolution_and_linear_ops_are_delegated_to_ethos_u() -> None:
    if not _executorch_available():
        pytest.skip("executorch is not importable")

    # Exercise quantization and lowering independently. This deliberately does not
    # call the exporter CLI or trust results/op_placement.md, which the exporter itself
    # writes from the same metadata.
    from src.export import lower
    from src.models.tinyeegnet import TinyEEGNet
    from src.quantize import compile_spec, quantize

    rng = np.random.default_rng(20261016)
    calibration = rng.standard_normal((4, 1, 22, 500), dtype=np.float32)
    spec = compile_spec()
    quantized = quantize(TinyEEGNet().eval(), calibration, spec)
    info, top_level, _ = lower(quantized, spec)

    assert info.num_delegated_subgraphs >= 1
    assert any("delegate" in op.lower() for op in top_level), top_level
    relevant = [
        breakdown
        for breakdown in info.delegation_by_operator.values()
        if re.search(r"conv|linear|addmm|mm(?:\.|$)", breakdown.op_type, re.I)
    ]
    assert relevant, "lowered program exposes no convolution/linear operator metadata"
    assert sum(item.delegated for item in relevant) > 0
    failures = [item.op_type for item in relevant if item.non_delegated != 0]
    assert not failures, "conv/linear ops left outside Ethos-U delegate: " + ", ".join(failures)

