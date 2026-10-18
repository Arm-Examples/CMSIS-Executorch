"""Leave-one-subject-out (row B) and per-subject (row A) data splits with EA.

Test protocol (identical for every method): the target subject's session 2, trials
1-24 are an unlabeled calibration block that only provides the EA matrix ``A``;
trials 25-144 (120 trials) are scored. The test session is never used for early
stopping, model selection, normalisation statistics or quantization calibration.

``loso_data`` / ``per_subject_data`` return a dict::

    {"train": (X, y), "val": (X, y), "calib": (X, y), "test": (X, y),
     "A_target": (22, 22), "input_scale": float, "val_subject": int,
     "test_trials": 0-based indices into session 2, "calib_source": str}

where every ``X`` is ``(n, 22, 500)`` float32 aligned µV (EA applied, not scaled).
LOSO validation = stratified 10 % of every source subject-session (``val_subject`` None);
per-subject validation = stratified 20 % of the target's session 1 (``val_subject`` = target).
``input_scale`` is one global float from the training data; the model consumes
``X[:, None] / input_scale`` (see ``src.train.to_model_input``). Calibration labels are
returned for bookkeeping only and must never be used for training.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.data.align import apply_ea, ea_matrix
from src.data.preprocess import load_subject

SUBJECTS = list(range(1, 10))
CALIB_TRIALS = 24
CACHE_DIR = Path(os.environ.get("RB_CACHE_DIR", "runs/cache"))


def cached_subject(subject: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``load_subject`` with an on-disk cache of the filtered windows (µV, unaligned)."""
    path = CACHE_DIR / f"subject_{subject:02d}.npz"
    if path.exists():
        z = np.load(path)
        return {1: (z["X1"], z["y1"]), 2: (z["X2"], z["y2"])}
    data = load_subject(subject)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, X1=data[1][0], y1=data[1][1], X2=data[2][0], y2=data[2][1])
    os.replace(tmp, path)
    return data


def align_session(X: np.ndarray) -> np.ndarray:
    """EA with ``A`` computed from the given trials themselves (one A per subject-session)."""
    return apply_ea(X, ea_matrix(X))


def target_calibration(data, calib_source: str = "session2") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(A_target, X_calib, y_calib)`` for the target subject.

    ``calib_source``: ``"session2"`` (default: first 24 trials of session 2, labels unused)
    or ``"session1"`` (variant: all of session 1, labels unused).
    """
    if calib_source == "session2":
        X_calib, y_calib = data[2][0][:CALIB_TRIALS], data[2][1][:CALIB_TRIALS]
    elif calib_source == "session1":
        X_calib, y_calib = data[1]
    else:
        raise ValueError(calib_source)
    return ea_matrix(X_calib), X_calib, y_calib


def _finish(X_train, y_train, X_val, y_val, data_target, calib_source, val_subject) -> dict:
    A_target, X_calib, y_calib = target_calibration(data_target, calib_source)
    X2, y2 = data_target[2]
    test_idx = np.arange(CALIB_TRIALS, len(y2))
    X_train = np.asarray(X_train, dtype=np.float32)
    return {
        "train": (X_train, np.asarray(y_train, dtype=np.int64)),
        "val": (np.asarray(X_val, dtype=np.float32), np.asarray(y_val, dtype=np.int64)),
        "calib": (apply_ea(X_calib, A_target), np.asarray(y_calib, dtype=np.int64)),
        "test": (apply_ea(X2[test_idx], A_target), np.asarray(y2[test_idx], dtype=np.int64)),
        "A_target": np.asarray(A_target, dtype=np.float64),
        # One global scale from the aligned training data only.
        "input_scale": float(np.std(X_train)),
        "val_subject": None if val_subject is None else int(val_subject),
        "test_trials": test_idx,
        "calib_source": calib_source,
    }


def stratified_holdout(y: np.ndarray, fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Class-balanced (train_idx, val_idx) holding out ``round(fraction * n_c)`` trials per class."""
    val_idx = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        n_val = int(round(fraction * len(idx)))
        val_idx.append(rng.choice(idx, size=n_val, replace=False))
    val_idx = np.sort(np.concatenate(val_idx))
    return np.setdiff1d(np.arange(len(y)), val_idx), val_idx


def loso_data(target: int, seed: int = 0, calib_source: str = "session2", val_fraction: float = 0.1,
              loader=None) -> dict:
    """Row B: train on the other 8 subjects (both sessions, EA per subject-session).

    EA for each source subject-session is computed on all of that session's trials
    (unlabeled). Validation for early stopping = a stratified, seed-dependent
    ``val_fraction`` (10 %) of the trials of EVERY source subject-session; the remaining
    90 % train. No target-subject trial reaches training or validation. ``loader``
    defaults to :func:`load_subject`; pass ``cached_subject`` to reuse filtered windows.
    ``val_subject`` in the result is ``None`` (all source subjects contribute).
    """
    loader = loader or load_subject
    others = [s for s in SUBJECTS if s != target]
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xva, yva = [], [], [], []
    for s in others:
        data = loader(s)
        for session in (1, 2):
            X, y = data[session]
            Xa = align_session(X)
            tr_idx, va_idx = stratified_holdout(y, val_fraction, rng)
            Xtr.append(Xa[tr_idx]); ytr.append(y[tr_idx])
            Xva.append(Xa[va_idx]); yva.append(y[va_idx])
    return _finish(np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(Xva), np.concatenate(yva),
                   loader(target), calib_source, None)


def per_subject_data(target: int, seed: int = 0, val_fraction: float = 0.2,
                     calib_source: str = "session2", loader=None) -> dict:
    """Row A: train on the target's session 1 (stratified 80/20 train/val split drawn by
    ``seed``), EA from session 1; test on session 2 trials 25-144 aligned with the
    calibration block exactly as in :func:`loso_data`."""
    loader = loader or load_subject
    data = loader(target)
    X1, y1 = data[1]
    Xa = align_session(X1)
    train_idx, val_idx = stratified_holdout(y1, val_fraction, np.random.default_rng(seed))
    return _finish(Xa[train_idx], y1[train_idx], Xa[val_idx], y1[val_idx], data, calib_source, target)


def raw_loso_data(target: int, aligned: bool = True, loader=None) -> dict:
    """``(n, 22, 500)`` arrays for the CSP+LDA baselines: all 8 other subjects for
    training (no validation split), target test block as above. ``aligned=False`` skips
    EA everywhere (µV)."""
    loader = loader or load_subject
    Xtr, ytr = [], []
    for s in SUBJECTS:
        if s == target:
            continue
        data = loader(s)
        for session in (1, 2):
            X, y = data[session]
            Xtr.append(align_session(X) if aligned else np.asarray(X, dtype=np.float32)); ytr.append(y)
    data = loader(target)
    X2, y2 = data[2]
    test_idx = np.arange(CALIB_TRIALS, len(y2))
    if aligned:
        A, _, _ = target_calibration(data, "session2")
        X_test = apply_ea(X2[test_idx], A)
    else:
        X_test = np.asarray(X2[test_idx], dtype=np.float32)
    return {"train": (np.concatenate(Xtr), np.concatenate(ytr)), "test": (X_test, y2[test_idx]),
            "test_trials": test_idx}
