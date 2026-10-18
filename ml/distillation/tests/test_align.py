"""Independent linear-algebra checks for Euclidean Alignment."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose


def _mean_covariance(X: np.ndarray) -> np.ndarray:
    return np.mean(X @ np.swapaxes(X, -1, -2) / X.shape[-1], axis=0)


def test_ea_whitens_the_empirical_mean_covariance(rng: np.random.Generator) -> None:
    from src.data.align import apply_ea, ea_matrix

    mixing = rng.standard_normal((22, 22))
    mixing = mixing @ mixing.T + 0.5 * np.eye(22)
    root = np.linalg.cholesky(mixing)
    X = np.einsum("ij,njt->nit", root, rng.standard_normal((48, 22, 700)))

    A = ea_matrix(X)
    aligned = apply_ea(X, A)

    assert A.shape == (22, 22)
    assert_allclose(A, A.T, rtol=0, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(A) > 0), "EA transform must be positive definite"
    assert_allclose(aligned, np.einsum("ij,njt->nit", A, X), rtol=1e-12, atol=1e-12)
    assert_allclose(_mean_covariance(aligned), np.eye(22), rtol=0, atol=1e-6)

