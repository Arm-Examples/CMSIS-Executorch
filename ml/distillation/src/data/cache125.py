"""125 Hz filtered windows for v2 (BCI IV 2a test set, Cho2017 + Lee2019_MI training sets).

One pipeline per recording session, all in microvolts, 22 channels in ``CHANNELS`` order:

1. ``external.standardize_channels`` (montage, interpolate missing names, pick + order).
2. Common average reference over the 22: subtract the per-sample channel mean.
3. To 125 Hz and band-pass 7-30 Hz, causal (``sosfilt`` on the continuous signal):
   - BCI IV 2a (250 Hz) = the board path: ``causal_filter`` with the v0.1 SOS designed at
     250 Hz, then keep every second sample (phase 0, ``x[:, ::2]``); cue -> ``cue // 2``.
     The band-pass is the anti-alias filter.
   - Cho2017 (512 Hz), Lee2019_MI (1000 Hz): MNE polyphase resample to 125 Hz, cue ->
     ``round(cue * 125 / fs)``, then ``causal_filter`` with the SOS designed at 125 Hz.
4. Epoch with ``preprocess.epoch`` at integer sample offsets after the cue:
   2a ``[62, 62 + 256)`` (cue + 0.496 s .. cue + 2.544 s), training sets
   ``[37, 37 + 306)`` so that ``X[..., 25:281]`` of a training trial is exactly the 2a
   window. 0.5 s is 62.5 samples at 125 Hz; 62 is what ``preprocess.epoch`` gives for
   ``tmin=0.5`` (Python rounds half to even) and the training start follows from it.

``cached_windows(dataset, subject)`` -> ``{session: (X (n, 22, W) float32 µV, y int64)}``
with sessions numbered from 1 in MOABB order (2a: 1 = train, 2 = test). Cached at
``runs/cache125/{dataset}_s{NN}.npz`` (``X``, ``y``, ``session``, ``version``); EA is applied later
(``src.data.splits125``), the cache is unaligned.
"""

from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np

from src.data.external import DATASETS, DataNotDownloaded, load_external, standardize_channels
from src.data.preprocess import UV_PER_V, causal_filter, design_filter, epoch

TEST_DATASET = "BNCI2014_001"
FS_MODEL = 125.0
FS_2A = 250.0
DECIMATION = 2
WINDOW = 256          # model input length at 125 Hz
TRAIN_WINDOW = 306    # training trial length, random crops of WINDOW inside it
CROP_OFFSET = 25      # canonical crop inside a training trial: [25, 25 + WINDOW)
WINDOW_START = 62     # samples after the cue (125 Hz) where the canonical window starts
TRAIN_START = WINDOW_START - CROP_OFFSET
FILTER = {"low": 7.0, "high": 30.0, "order": 4}
# Bump when the pipeline changes; caches with another version are rebuilt.
CACHE_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("RB_CACHE125_DIR", REPO_ROOT / "runs" / "cache125"))


def common_average_reference(x: np.ndarray) -> np.ndarray:
    """Subtract the per-sample mean over channels (axis -2) of ``(..., C, T)`` data."""
    x = np.asarray(x)
    return x - x.mean(axis=-2, keepdims=True)


def _car_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.apply_function(lambda d: d - d.mean(axis=0, keepdims=True), channel_wise=False, verbose=False)
    return raw


def board_decimate(x_uv_250: np.ndarray, sos_250: np.ndarray) -> np.ndarray:
    """Board path: causal band-pass at 250 Hz on the continuous signal, keep every
    second sample starting with sample 0. ``(C, T)`` -> ``(C, ceil(T / 2))``."""
    return causal_filter(x_uv_250, sos_250)[:, ::DECIMATION]


def _epoch_samples(xf: np.ndarray, events: np.ndarray, start: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = epoch(xf, events, fs=FS_MODEL, tmin=start / FS_MODEL, tmax=(start + length) / FS_MODEL)
    if X.shape[-1] != length:
        raise AssertionError(f"window length {X.shape[-1]} != {length}")
    return X, y


def windows_2a(raw: mne.io.BaseRaw, filt: dict = FILTER) -> tuple[np.ndarray, np.ndarray]:
    """One 2a session (250 Hz Raw from ``download.load_raw``) -> ``(n, 22, 256)`` µV, labels."""
    if abs(raw.info["sfreq"] - FS_2A) > 1e-6:
        raise ValueError(f"expected {FS_2A} Hz, got {raw.info['sfreq']}")
    raw = standardize_channels(raw)
    x = common_average_reference(raw.get_data() * UV_PER_V)
    xd = board_decimate(x, design_filter(fs=FS_2A, **filt))
    ev = np.asarray(raw.info["temp"]["events"], dtype=np.int64).copy()
    ev[:, 0] = ev[:, 0] // DECIMATION
    return _epoch_samples(xd, ev, WINDOW_START, WINDOW)


def windows_external(raw: mne.io.BaseRaw, filt: dict = FILTER) -> tuple[np.ndarray, np.ndarray]:
    """One Cho2017 / Lee2019_MI session -> ``(n, 22, 306)`` µV training trials, labels."""
    fs = float(raw.info["sfreq"])
    raw = _car_raw(standardize_channels(raw))
    ev = np.asarray(raw.info["temp"]["events"], dtype=np.int64).copy()
    if abs(fs - FS_MODEL) > 1e-6:
        raw.resample(FS_MODEL, method="polyphase", verbose=False)
        ev[:, 0] = np.round(ev[:, 0] * FS_MODEL / fs).astype(np.int64)
    x = raw.get_data() * UV_PER_V
    xf = causal_filter(x, design_filter(fs=FS_MODEL, **filt))
    return _epoch_samples(xf, ev, TRAIN_START, TRAIN_WINDOW)


def build_windows(dataset: str, subject: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Run the pipeline for every session of one subject (no cache)."""
    if dataset == TEST_DATASET:
        from src.data.download import load_raw
        from src.data.external import use_upstream_provider

        use_upstream_provider()
        raws = load_raw(subject)
        return {int(s): windows_2a(r) for s, r in sorted(raws.items())}
    if dataset in DATASETS:
        raws = load_external(dataset, subject)
        return {i: windows_external(raws[k]) for i, k in enumerate(raws, start=1)}
    raise ValueError(f"unknown dataset {dataset!r}")


def cache_path(dataset: str, subject: int, cache_dir: Path | None = None) -> Path:
    return Path(cache_dir or CACHE_DIR) / f"{dataset}_s{subject:02d}.npz"


def _unpack(z) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    X, y, sess = z["X"], z["y"], z["session"]
    return {int(s): (X[sess == s], y[sess == s]) for s in np.unique(sess)}


def cached_windows(dataset: str, subject: int, cache_dir: Path | None = None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``build_windows`` with an on-disk cache (µV, unaligned). Raises
    :class:`~src.data.external.DataNotDownloaded` when the raw files are not on disk."""
    path = cache_path(dataset, subject, cache_dir)
    if path.exists():
        with np.load(path) as z:
            if "version" in z.files and int(z["version"]) == CACHE_VERSION:
                return _unpack(z)
    data = build_windows(dataset, subject)
    sessions = sorted(data)
    X = np.concatenate([data[s][0] for s in sessions]).astype(np.float32)
    y = np.concatenate([data[s][1] for s in sessions]).astype(np.int64)
    sess = np.concatenate([np.full(len(data[s][1]), s, dtype=np.int64) for s in sessions])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npz")
    np.savez(tmp, X=X, y=y, session=sess, version=np.int64(CACHE_VERSION))
    os.replace(tmp, path)
    return {s: (data[s][0].astype(np.float32), data[s][1].astype(np.int64)) for s in sessions}


__all__ = ["cached_windows", "build_windows", "windows_2a", "windows_external", "board_decimate",
           "common_average_reference", "DataNotDownloaded", "FS_MODEL", "WINDOW", "TRAIN_WINDOW",
           "CROP_OFFSET", "WINDOW_START", "TRAIN_START", "TEST_DATASET"]
