# SPDX-License-Identifier: Apache-2.0
"""Signal-quality gate for the v3 training data (Cho2017, Lee2019_MI).

Three per-trial measures on the filtered, common-average-referenced windows in µV
(``(n, 22, T)``, before Euclidean Alignment), chosen so the same formulas can later run on
the Cortex-M55 next to the motion gate (no eigensolver at run time):

1. **Peak-to-peak** (``ptp``): any channel with ``|x - mean_t(x)| > ptp_uv`` (default 100 µV).
2. **Whitened-covariance distance** (``potato``, a Riemannian-potato surrogate in EA space):
   with the session's EA matrix ``A``, ``d = ||(A X)(A X)^T / T - I||_F``; ``d`` is z-scored
   within the subject-session and a trial is flagged when ``z > potato_z`` (default 2.5).
   Two iterations: the second recomputes ``A`` from the trials the first one kept and
   z-scores against those trials' mean and std.
3. **Flat channel** (``flat``): any channel with standard deviation ``< flat_uv`` (0.1 µV).

``session_decision`` drops a whole subject-session when more than ``max_flagged_fraction``
(0.2) of its trials are flagged by any measure, or when any channel is flat over the whole
session. Kept sessions keep all their trials (no trial-level dropping). BCI IV 2a test
trials are never gated; ``src.data.splits125`` only logs the flags of its 24 calibration
trials.

``sqi(window, A)`` returns the three raw measures for a single window (no session z-score),
the per-window form a firmware port would compute.
"""

from __future__ import annotations

import numpy as np

from src.data.align import ea_matrix

DEFAULTS = {"ptp_uv": 100.0, "potato_z": 2.5, "flat_uv": 0.1, "max_flagged_fraction": 0.2,
            "potato_iterations": 2}
FLAG_KEYS = ("ptp", "potato", "flat", "any")


def _cfg(cfg: dict | None) -> dict:
    return {**DEFAULTS, **(cfg or {})}


def ptp_measure(X: np.ndarray) -> np.ndarray:
    """Largest ``|x - mean_t(x)|`` over channels and samples, per trial: ``(n,)`` µV."""
    X = np.asarray(X, dtype=np.float64)
    return np.abs(X - X.mean(axis=-1, keepdims=True)).max(axis=(-2, -1))


def min_channel_std(X: np.ndarray) -> np.ndarray:
    """Smallest per-channel standard deviation, per trial: ``(n,)`` µV."""
    return np.asarray(X, dtype=np.float64).std(axis=-1).min(axis=-1)


def whitened_distance(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """``||(A X)(A X)^T / T - I||_F`` per trial: ``(n,)``."""
    Xa = np.einsum("cd,ndt->nct", np.asarray(A, dtype=np.float64), np.asarray(X, dtype=np.float64))
    T = Xa.shape[-1]
    C = np.einsum("nct,ndt->ncd", Xa, Xa) / T
    return np.linalg.norm(C - np.eye(C.shape[-1]), axis=(-2, -1))


def _zscore(d: np.ndarray, ref: np.ndarray) -> np.ndarray:
    sd = float(d[ref].std()) if ref.sum() > 1 else 0.0
    if sd <= 0:
        return np.zeros_like(d)
    return (d - float(d[ref].mean())) / sd


def trial_flags(X_uv: np.ndarray, A: np.ndarray | None = None, cfg: dict | None = None,
                eps: float = 1e-3) -> dict:
    """Per-trial quality flags of one subject-session ``X_uv`` ``(n, C, T)`` (µV, unaligned).

    ``A`` is the session's EA matrix for the first potato iteration (computed from all trials
    with ``eps`` when None). Returns boolean ``(n,)`` arrays ``ptp``, ``potato``, ``flat``,
    ``any`` plus the raw measures ``ptp_uv``, ``min_std_uv``, ``distance``, ``z``.
    """
    c = _cfg(cfg)
    X = np.asarray(X_uv, dtype=np.float64)
    n = len(X)
    ptp_v = ptp_measure(X)
    std_v = min_channel_std(X)
    f_ptp = ptp_v > float(c["ptp_uv"])
    f_flat = std_v < float(c["flat_uv"])
    A_cur = ea_matrix(X, eps) if A is None else np.asarray(A, dtype=np.float64)
    keep = np.ones(n, dtype=bool)
    d = z = np.zeros(n)
    for it in range(int(c["potato_iterations"])):
        if it > 0:
            if keep.sum() < 2:
                break
            A_cur = ea_matrix(X[keep], eps)
        d = whitened_distance(X, A_cur)
        z = _zscore(d, keep)
        keep = z <= float(c["potato_z"])
    f_pot = z > float(c["potato_z"])
    return {"ptp": f_ptp, "potato": f_pot, "flat": f_flat, "any": f_ptp | f_pot | f_flat,
            "ptp_uv": ptp_v, "min_std_uv": std_v, "distance": d, "z": z}


def session_flat(X_uv: np.ndarray, flat_uv: float) -> bool:
    """True when any channel's std over the whole session (all trials pooled) is below ``flat_uv``."""
    X = np.asarray(X_uv, dtype=np.float64)
    pooled = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
    return bool(np.any(pooled.std(axis=-1) < float(flat_uv)))


def session_decision(flags: dict, cfg: dict | None = None, X_uv: np.ndarray | None = None) -> dict:
    """Keep or drop one subject-session from its ``trial_flags``.

    Drop when the fraction of flagged trials exceeds ``max_flagged_fraction`` or a channel is
    flat for the whole session (``X_uv`` given: pooled std below ``flat_uv``; otherwise every
    trial flagged ``flat``). Returns ``{"keep", "reason", "n", "flagged", "fraction", "rates"}``.
    """
    c = _cfg(cfg)
    n = int(len(flags["any"]))
    flagged = int(np.sum(flags["any"]))
    frac = flagged / max(n, 1)
    flat_session = session_flat(X_uv, c["flat_uv"]) if X_uv is not None else bool(n and np.all(flags["flat"]))
    reasons = []
    if frac > float(c["max_flagged_fraction"]):
        reasons.append(f"flagged {flagged}/{n} = {frac:.3f} > {float(c['max_flagged_fraction'])}")
    if flat_session:
        reasons.append("flat channel over the whole session")
    return {"keep": not reasons, "reason": "; ".join(reasons), "n": n, "flagged": flagged, "fraction": frac,
            "rates": {k: float(np.mean(flags[k])) if n else 0.0 for k in FLAG_KEYS},
            "counts": {k: int(np.sum(flags[k])) for k in FLAG_KEYS}}


def sqi(window: np.ndarray, A: np.ndarray) -> dict:
    """Raw quality measures of one window ``(C, T)`` µV with EA matrix ``A``:
    ``ptp_uv`` (max ``|x - mean|``), ``min_std_uv`` and ``distance`` (whitened-covariance
    distance to the identity). No session statistics, so no z-score."""
    w = np.asarray(window, dtype=np.float64)[None]
    return {"ptp_uv": float(ptp_measure(w)[0]), "min_std_uv": float(min_channel_std(w)[0]),
            "distance": float(whitened_distance(w, A)[0])}


__all__ = ["trial_flags", "session_decision", "sqi", "session_flat", "whitened_distance", "ptp_measure",
           "min_channel_std", "DEFAULTS", "FLAG_KEYS"]
