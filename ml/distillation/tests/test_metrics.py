"""KPI definitions checked directly against scikit-learn."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


def test_compute_kpis_matches_sklearn_with_class_one_positive() -> None:
    from src.eval import compute_kpis

    y_true = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    # Deliberately asymmetric: using class 0 or hard predictions for AUPRC fails.
    logits = np.array(
        [
            [3.0, -1.0],
            [0.1, 0.2],
            [1.2, 0.8],
            [0.9, 1.0],
            [-0.5, 2.4],
            [2.0, 0.0],
            [0.4, 1.7],
        ],
        dtype=np.float64,
    )
    scores = softmax(logits, axis=1)[:, 1]
    predictions = np.argmax(logits, axis=1)
    expected = {
        "accuracy": accuracy_score(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, scores),
        "f1": f1_score(y_true, predictions, average="macro"),
        "auprc": average_precision_score(y_true, scores),
        "kappa": cohen_kappa_score(y_true, predictions),
    }

    actual = compute_kpis(y_true, logits)
    assert set(actual) == set(expected)
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value, abs=1e-12), name

