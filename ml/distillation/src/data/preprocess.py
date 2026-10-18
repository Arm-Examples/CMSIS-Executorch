"""Causal band-pass filtering and epoching for BCI IV 2a.

The filter is applied with ``scipy.signal.sosfilt`` on the continuous signal, i.e.
causally and without zero-phase correction, so the same second-order sections can be
run sample by sample on the board (``preprocess_params.json`` carries the SOS
coefficients). Windows are cue + 0.5 s to cue + 2.5 s = 500 samples at 250 Hz.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .download import CLASS_MAP, FS, N_EEG, load_raw

UV_PER_V = 1e6


def design_filter(fs: float = FS, low: float = 7.0, high: float = 30.0, order: int = 4) -> np.ndarray:
    """Butterworth band-pass as second-order sections, shape ``(n_sections, 6)``."""
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def causal_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Causal IIR filtering along the last axis of continuous data ``(22, n_samples)``."""
    return signal.sosfilt(sos, x, axis=-1)


GDF_TO_CLASS = {769: 0, 770: 1}  # left hand -> 0, right hand -> 1


def epoch(
    x_filtered: np.ndarray,
    events: np.ndarray,
    fs: float = FS,
    tmin: float = 0.5,
    tmax: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut ``(n, 22, 500)`` float32 windows at ``cue + tmin .. cue + tmax`` (end exclusive).

    ``events`` is an MNE-style ``(n, 3)`` int array ``[cue_sample, _, code]``. Only GDF
    codes 769 (left hand -> 0) and 770 (right hand -> 1) are kept; every other code
    (768 trial start, 771 feet, 772 tongue, 1023 rejected, ...) is dropped. Trials are
    returned in recording order.
    """
    events = np.asarray(events)
    start = int(round(tmin * fs))
    length = int(round((tmax - tmin) * fs))
    n_samples = x_filtered.shape[-1]
    X, y = [], []
    for row in events:
        code = int(row[-1])
        if code not in GDF_TO_CLASS:
            continue
        a = int(row[0]) + start
        b = a + length
        if a < 0 or b > n_samples:
            raise ValueError(f"trial at sample {row[0]} does not fit the recording")
        X.append(x_filtered[:, a:b])
        y.append(GDF_TO_CLASS[code])
    X = np.asarray(X, dtype=np.float32).reshape(-1, x_filtered.shape[0], length)
    return X, np.asarray(y, dtype=np.int64)


def preprocess_raw(raw, sos: np.ndarray | None = None, **epoch_kw) -> tuple[np.ndarray, np.ndarray]:
    """Raw (22 EEG ch, volts) -> filtered µV windows ``(n, 22, 500)`` and labels."""
    sos = design_filter() if sos is None else sos
    x = raw.get_data() * UV_PER_V  # (22, n_samples) in µV
    if x.shape[0] != N_EEG:
        raise ValueError(f"expected {N_EEG} channels, got {x.shape[0]}")
    xf = causal_filter(x, sos)
    return epoch(xf, raw.info["temp"]["events"], fs=raw.info["sfreq"], **epoch_kw)


def load_subject(subject: int, **epoch_kw) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{1: (X, y), 2: (X, y)}``: 144 left/right trials per session, filtered, in µV."""
    sos = design_filter()
    out = {}
    for session, raw in load_raw(subject).items():
        X, y = preprocess_raw(raw, sos, **epoch_kw)
        out[session] = (X, y)
    return out


__all__ = ["design_filter", "causal_filter", "epoch", "preprocess_raw", "load_subject", "CLASS_MAP"]
