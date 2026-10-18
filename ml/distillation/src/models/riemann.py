# SPDX-License-Identifier: Apache-2.0
"""Riemannian teacher member (v4): OAS covariance -> tangent space -> logistic regression.

``RiemannMember`` is an ``nn.Module`` wrapper (no trainable torch parameters) so it can sit in
the teacher ensemble next to the CNN members: ``forward(x)`` takes the model input
``(B, 1, 22, T)`` (aligned, divided by ``input_scale``) and returns the **log probabilities**
``(B, 2)``. ``softmax`` of them is the classifier's probability, so ``ensemble_logits`` (mean of
softmax) and ``distill.teacher_probs`` (mean of ``softmax(z / T)``) use it unchanged; at
temperature ``T`` the member contributes ``p^(1/T)`` renormalised.

Pipeline (scikit-learn, fitted on the CPU in seconds, pickled with ``joblib``):

1. ``OASCovariances``: per-trial OAS shrunk covariance (vectorised, same formula as
   ``sklearn.covariance.oas``), plus a ridge ``ridge * trace / C * I`` (common average reference
   leaves every trial rank-deficient by one; OAS shrinkage alone keeps it SPD, the ridge keeps
   it SPD under channel dropout too);
2. ``pyriemann.tangentspace.TangentSpace(metric="riemann")`` at the Riemannian mean of the
   training covariances;
3. ``sklearn.linear_model.LogisticRegression(C=1)``.

Offline only: nothing of this runs on the board.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from torch import nn


def oas_covariances(X: np.ndarray, ridge: float = 0.0, chunk: int = 1024) -> np.ndarray:
    """OAS shrunk covariance of every trial of ``X`` ``(n, C, T)`` -> ``(n, C, C)`` float64.

    Matches ``sklearn.covariance.oas(x.T)`` per trial (centred, divided by ``T``), then adds
    ``ridge * trace(S) / C * I``."""
    X = np.asarray(X)
    if len(X) > chunk:
        return np.concatenate([oas_covariances(X[i : i + chunk], ridge, chunk) for i in range(0, len(X), chunk)])
    X = X.astype(np.float64)
    n, C, T = X.shape
    Xc = X - X.mean(axis=-1, keepdims=True)
    S = np.einsum("nct,ndt->ncd", Xc, Xc) / T
    alpha = np.mean(S ** 2, axis=(1, 2))
    mu = np.trace(S, axis1=1, axis2=2) / C
    num = alpha + mu ** 2
    den = (T + 1) * (alpha - mu ** 2 / C)
    with np.errstate(divide="ignore", invalid="ignore"):
        shrink = np.where(den == 0, 1.0, np.minimum(num / np.where(den == 0, 1.0, den), 1.0))
    eye = np.eye(C)
    out = (1.0 - shrink)[:, None, None] * S + (shrink * mu)[:, None, None] * eye
    if ridge:
        out = out + (ridge * np.trace(out, axis1=1, axis2=2) / C)[:, None, None] * eye
    return out


class OASCovariances(BaseEstimator, TransformerMixin):
    """scikit-learn transformer ``(n, C, T)`` -> ``(n, C, C)`` (see :func:`oas_covariances`)."""

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return oas_covariances(X, self.ridge)


def build_pipeline(C: float = 1.0, ridge: float = 1e-3, max_iter: int = 1000) -> Pipeline:
    from pyriemann.tangentspace import TangentSpace

    return Pipeline([("cov", OASCovariances(ridge=ridge)), ("ts", TangentSpace(metric="riemann")),
                     ("lr", LogisticRegression(C=C, max_iter=max_iter))])


class RiemannMember(nn.Module):
    """Teacher member wrapping a fitted (or unfitted) :func:`build_pipeline`; ``forward``
    returns log probabilities on the input's device."""

    def __init__(self, pipeline: Pipeline | None = None, **kwargs):
        super().__init__()
        self.pipeline = pipeline if pipeline is not None else build_pipeline(**kwargs)

    @staticmethod
    def _numpy(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().to("cpu").numpy()
        x = np.asarray(x)
        return x[:, 0] if x.ndim == 4 else x

    def fit(self, X, y, sample_weight=None) -> "RiemannMember":
        """``sample_weight`` (optional, per window) goes to the logistic regression only."""
        kw = {} if sample_weight is None else {"lr__sample_weight": np.asarray(sample_weight, dtype=np.float64)}
        self.pipeline.fit(self._numpy(X), np.asarray(y, dtype=np.int64), **kw)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(self._numpy(X))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logp = np.log(np.clip(self.predict_proba(x), 1e-12, None))
        return torch.as_tensor(logp, dtype=torch.float32, device=x.device)

    @property
    def n_coefficients(self) -> int:
        """Logistic-regression weights + intercept (``C (C + 1) / 2 + 1`` for ``C`` channels)."""
        lr = self.pipeline.named_steps["lr"]
        return int(lr.coef_.size + lr.intercept_.size) if hasattr(lr, "coef_") else 0

    def save(self, path: str | Path) -> None:
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "RiemannMember":
        return cls(joblib.load(path))


def n_coefficients(n_channels: int = 22) -> int:
    return n_channels * (n_channels + 1) // 2 + 1


__all__ = ["RiemannMember", "OASCovariances", "oas_covariances", "build_pipeline", "n_coefficients"]
