"""KPIs from logits with scikit-learn, plus a small CLI over saved runs.

``compute_kpis(y_true, logits)`` -> ``accuracy, roc_auc, f1 (macro), auprc, kappa``.
Scores are ``softmax(logits)[:, 1]`` (positive class 1 = right hand); predictions are
``argmax``. Chance level: 50 % accuracy, 0.5 ROC-AUC/AUPRC, 0 kappa.

CLI: ``python -m src.eval --run runs/loso_s1_seed0`` (reads ``predictions.npz``) or
``python -m src.eval --checkpoint runs/.../model.pt --mode loso --subject 1 --seed 0``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

KPI_KEYS = ("accuracy", "roc_auc", "f1", "auprc", "kappa")


def compute_kpis(y_true, logits) -> dict:
    y_true = np.asarray(y_true).astype(np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    scores = softmax(logits, axis=1)[:, 1]
    y_pred = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "auprc": float(average_precision_score(y_true, scores)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def confusion(y_true, logits) -> np.ndarray:
    """2x2 confusion matrix, rows = true (left, right), columns = predicted."""
    return confusion_matrix(np.asarray(y_true), np.argmax(np.asarray(logits), axis=1), labels=[0, 1])


def evaluate_predictions(path: Path) -> dict:
    z = np.load(path)
    return compute_kpis(z["y_true"], z["logits"])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="KPIs of a saved run")
    p.add_argument("--run", type=Path, help="run directory holding predictions.npz")
    p.add_argument("--predictions", type=Path, help="explicit predictions .npz (y_true, logits)")
    p.add_argument("--checkpoint", type=Path, help="re-evaluate a state_dict on the test block")
    p.add_argument("--mode", choices=["loso", "per-subject"], default="loso")
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=["tinyeegnet", "eegnet"], default="tinyeegnet")
    a = p.parse_args(argv)
    if a.checkpoint:
        from src.train import evaluate_checkpoint

        kpis = evaluate_checkpoint(a.checkpoint, a.mode, a.subject, a.seed, a.model)
    else:
        path = a.predictions or (a.run / "predictions.npz")
        kpis = evaluate_predictions(path)
    print(json.dumps(kpis, indent=2))


if __name__ == "__main__":
    main()
