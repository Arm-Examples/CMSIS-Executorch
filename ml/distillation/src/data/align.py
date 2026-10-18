"""Euclidean Alignment (He & Wu, 2020).

``R = mean_i X_i X_iᵀ / T`` over trials, ``A = R^(-1/2)`` (symmetric, via ``eigh``),
``X̃_i = A X_i``. After alignment the mean trial covariance is the identity, which
removes most of the subject/session shift before a shared model sees the data.
"""

from __future__ import annotations

import numpy as np


def mean_covariance(X: np.ndarray) -> np.ndarray:
    """Mean of per-trial covariances ``X Xᵀ / T`` for ``X`` of shape ``(n, C, T)``."""
    X = np.asarray(X, dtype=np.float64)
    n, _, T = X.shape
    return np.einsum("nct,ndt->cd", X, X) / (n * T)


def ea_matrix(X: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Inverse symmetric square root of the mean trial covariance, shape ``(C, C)``."""
    R = mean_covariance(X)
    if eps:
        R = R + eps * np.trace(R) / R.shape[0] * np.eye(R.shape[0])
    w, V = np.linalg.eigh(R)
    if np.any(w <= 0):
        raise np.linalg.LinAlgError("mean covariance is not positive definite")
    A = (V * (1.0 / np.sqrt(w))) @ V.T
    return 0.5 * (A + A.T)


def apply_ea(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """``A @ X_i`` for every trial; same shape and dtype as ``X``."""
    X = np.asarray(X)
    out = np.einsum("cd,ndt->nct", np.asarray(A, dtype=np.float64), X)
    return out.astype(X.dtype, copy=False)
