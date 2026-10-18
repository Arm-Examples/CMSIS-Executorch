"""v2 data split: train on Cho2017 + Lee2019_MI, test on all 9 BCI IV 2a subjects.

``v2_data(seed, cfg)`` returns::

    {"train": (X (n, 22, 306) float32, y int64),   # EA per training subject-session, not scaled
     "val":   (X (m, 22, 306) float32, y int64),   # stratified 10 % of every training subject-session
     "calib": {subject: X (24, 22, 256) float32},  # 2a session 2 trials 1-24, EA applied, no labels
     "test":  {subject: (X (120, 22, 256), y)},    # 2a session 2 trials 25-144, EA applied
     "ea":    {subject: A (22, 22) float64},       # 2a EA matrix from the 24 calibration trials
     "input_scale": float,                         # std of the aligned training X
     "meta": {...},
     # added for v3 (v2 keys above unchanged):
     "train_group": int64 (n,),                    # index into list(ea_train) of every training trial
     "ea_train": {"<dataset>_s<NN>_<session>": A}, # EA matrix of every training subject-session
     "val_group": int64 (m,), "ea_val": {...}}     # same for the validation trials

Optional v3 config sections (absent in ``configs/v2.yaml``, so v2 behaviour is unchanged):

- ``quality``: the signal-quality gate of :mod:`src.data.quality` per training-source
  subject-session (Cho2017, Lee2019_MI, including validation subjects). A dropped session
  contributes no trial; kept sessions keep all trials. ``meta["quality"]`` lists the dropped
  sessions and per-dataset flag rates; the 24 2a calibration trials of every test subject are
  flagged and logged only, the 2a test trials are never gated.
- ``validation.val_subjects``: ``{dataset: [subjects]}`` held out as whole subjects (all
  sessions) for validation instead of the per-trial stratified holdout.

Model input = ``X[:, None] / input_scale``. The canonical window of a training trial is
``X[..., 25:281]``; test and calib arrays are already that window. No 2a trial is used for
training or validation; 2a session 1 is unused by ``v2_data`` (v4 reads its windows, never
its labels, through ``target_unlabeled``). EA uses ``align.ea_matrix`` with
``cfg["ea_eps"]`` (default 1e-3): common average reference leaves the 22x22 covariance with
one (numerically) zero eigenvalue, which the ``eps * trace / 22`` ridge lifts.

Validation draws are seeded per ``(seed, dataset, subject, session)`` so a subject's split
does not change when other subjects are added or skipped. Subjects whose raw files are not
downloaded yet are skipped with a warning and listed in ``meta["skipped"]``.
"""

from __future__ import annotations

import warnings
import zlib
from pathlib import Path

import numpy as np
import yaml

from src.data import cache125
from src.data.align import apply_ea, ea_matrix
from src.data.external import DataNotDownloaded
from src.data.quality import session_decision, trial_flags
from src.data.splits import stratified_holdout

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "v2.yaml"
DEFAULT_EA_EPS = 1e-3


def load_config(path: Path = CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_subjects(spec) -> list[int]:
    """``[1, 2, 3]``, ``5`` or ``"1-52"`` -> sorted list of ints."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        out: list[int] = []
        for part in spec.split(","):
            a, _, b = part.strip().partition("-")
            out.extend(range(int(a), int(b or a) + 1))
        return sorted(set(out))
    return sorted({int(s) for s in spec})


def training_subjects(cfg: dict) -> list[tuple[str, int]]:
    """``[(dataset, subject), ...]`` from ``cfg["datasets"]`` minus each ``exclude`` list."""
    pairs = []
    for dataset, spec in cfg["datasets"].items():
        excl = set(expand_subjects(spec.get("exclude") or []))
        pairs += [(dataset, s) for s in expand_subjects(spec["subjects"]) if s not in excl]
    return pairs


def check_config(cfg: dict) -> None:
    """The cached windows are built with the constants of ``cache125``; refuse a config
    that silently disagrees with them."""
    want = {"fs_model": cache125.FS_MODEL, "window": cache125.WINDOW, "train_window": cache125.TRAIN_WINDOW,
            "crop_offset": cache125.CROP_OFFSET, "window_start": cache125.WINDOW_START}
    for key, value in want.items():
        if key in cfg and float(cfg[key]) != float(value):
            raise ValueError(f"cfg[{key!r}] = {cfg[key]} but cache125 builds {value}")
    for key, value in cache125.FILTER.items():
        if float(cfg.get("filter", {}).get(key, value)) != float(value):
            raise ValueError(f"cfg['filter'][{key!r}] = {cfg['filter'][key]} but cache125 uses {value}")


def _rng(seed: int, dataset: str, subject: int, session: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(dataset.encode()), int(subject), int(session)])


def v2_data(seed: int, cfg: dict | None = None, loader=None) -> dict:
    """Training/validation from the external datasets, calibration/test from 2a (see module doc).

    ``loader(dataset, subject) -> {session: (X, y)}`` defaults to
    :func:`src.data.cache125.cached_windows`; it may raise ``DataNotDownloaded``.
    """
    cfg = load_config() if cfg is None else cfg
    check_config(cfg)
    loader = loader or cache125.cached_windows
    eps = float(cfg.get("ea_eps", DEFAULT_EA_EPS))
    val_fraction = float(cfg.get("training", {}).get("val_fraction", 0.1))
    test_cfg = cfg.get("test", {})
    test_dataset = test_cfg.get("dataset", cache125.TEST_DATASET)
    test_session = int(test_cfg.get("session", 2))
    n_calib = int(test_cfg.get("calib_trials", 24))

    qcfg = cfg.get("quality")
    val_subjects = {ds: set(expand_subjects(v)) for ds, v in ((cfg.get("validation") or {}).get("val_subjects") or {}).items()}
    cross_subject_val = bool(val_subjects)

    Xtr, ytr, Xva, yva, gtr, gva = [], [], [], [], [], []
    ea_train: dict[str, np.ndarray] = {}
    ea_val: dict[str, np.ndarray] = {}
    sources: dict[str, dict] = {}
    skipped: list[str] = []
    quality: dict = {"config": dict(qcfg), "dropped": [], "datasets": {}} if qcfg is not None else {}
    n_sessions = 0
    for dataset, subject in training_subjects(cfg):
        if dataset == test_dataset:
            raise ValueError("the test dataset must not be a training source")
        src = sources.setdefault(dataset, {"subjects": 0, "subject_sessions": 0, "train": 0, "val": 0,
                                           "skipped": 0})
        try:
            data = loader(dataset, subject)
        except DataNotDownloaded as exc:
            warnings.warn(f"skipping {dataset} subject {subject}: not downloaded ({exc})", stacklevel=2)
            src["skipped"] += 1
            skipped.append(f"{dataset}_s{subject:02d}")
            continue
        src["subjects"] += 1
        is_val_subject = subject in val_subjects.get(dataset, set())
        for session in sorted(data):
            X, y = data[session]
            key = f"{dataset}_s{subject:02d}_{session}"
            A = ea_matrix(X, eps)
            if qcfg is not None:
                flags = trial_flags(X, A, qcfg, eps)
                dec = session_decision(flags, qcfg, X)
                q = quality["datasets"].setdefault(dataset, {"sessions": 0, "dropped": 0, "trials": 0,
                                                             "counts": {k: 0 for k in dec["counts"]}})
                q["sessions"] += 1
                q["trials"] += dec["n"]
                for k, v in dec["counts"].items():
                    q["counts"][k] += v
                if not dec["keep"]:
                    q["dropped"] += 1
                    quality["dropped"].append({"session": key, "reason": dec["reason"], "counts": dec["counts"]})
                    src.setdefault("dropped_sessions", 0)
                    src["dropped_sessions"] += 1
                    continue
            Xa = apply_ea(np.asarray(X, dtype=np.float32), A)
            if cross_subject_val:
                tr = np.arange(0) if is_val_subject else np.arange(len(y))
                va = np.arange(len(y)) if is_val_subject else np.arange(0)
            else:
                tr, va = stratified_holdout(y, val_fraction, _rng(seed, dataset, subject, session))
            if len(tr):
                gtr.append(np.full(len(tr), len(ea_train), dtype=np.int64))
                ea_train[key] = np.asarray(A, dtype=np.float64)
            if len(va):
                gva.append(np.full(len(va), len(ea_val), dtype=np.int64))
                ea_val[key] = np.asarray(A, dtype=np.float64)
            Xtr.append(Xa[tr]); ytr.append(y[tr])
            Xva.append(Xa[va]); yva.append(y[va])
            src["subject_sessions"] += 1
            src["train"] += len(tr)
            src["val"] += len(va)
            n_sessions += 1
    if not Xtr or not sum(len(v) for v in ytr):
        raise RuntimeError("no training subject could be loaded")
    if not sum(len(v) for v in yva):
        raise RuntimeError("the validation set is empty (check validation.val_subjects)")
    for q in quality.get("datasets", {}).values():
        q["flag_rates"] = {k: v / max(q["trials"], 1) for k, v in q["counts"].items()}
    X_train = np.concatenate(Xtr).astype(np.float32, copy=False)
    y_train = np.concatenate(ytr).astype(np.int64)
    X_val = np.concatenate(Xva).astype(np.float32, copy=False)
    y_val = np.concatenate(yva).astype(np.int64)

    calib, test, ea = {}, {}, {}
    for subject in expand_subjects(test_cfg.get("subjects", list(range(1, 10)))):
        X2, y2 = loader(test_dataset, subject)[test_session]
        if len(y2) <= n_calib:
            raise ValueError(f"{test_dataset} s{subject}: only {len(y2)} trials")
        A = ea_matrix(X2[:n_calib], eps)
        calib[subject] = apply_ea(np.asarray(X2[:n_calib], dtype=np.float32), A)
        test[subject] = (apply_ea(np.asarray(X2[n_calib:], dtype=np.float32), A),
                         np.asarray(y2[n_calib:], dtype=np.int64))
        ea[subject] = np.asarray(A, dtype=np.float64)
        if qcfg is not None:  # logged only: 2a trials are never gated
            f = trial_flags(X2[:n_calib], A, qcfg, eps)
            quality.setdefault("calib_2a", {})[str(subject)] = {
                k: int(np.sum(f[k])) for k in ("ptp", "potato", "flat", "any")}

    meta_extra = {}
    if qcfg is not None:
        meta_extra["quality"] = quality
    if cross_subject_val:
        meta_extra["validation"] = {"mode": "cross_subject",
                                    "val_subjects": {d: sorted(v) for d, v in val_subjects.items()},
                                    "val_sessions": list(ea_val)}

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "calib": calib,
        "test": test,
        "ea": ea,
        "input_scale": float(np.std(X_train)),
        "meta": {
            "n_train_subject_sessions": n_sessions,
            "sources": sources,
            "skipped": skipped,
            "fs": int(cache125.FS_MODEL),
            "window": cache125.WINDOW,
            "train_window": cache125.TRAIN_WINDOW,
            "crop_offset": cache125.CROP_OFFSET,
            "window_start": cache125.WINDOW_START,
            "ea_eps": eps,
            "seed": int(seed),
            **meta_extra,
        },
        "train_group": np.concatenate(gtr) if gtr else np.zeros(0, dtype=np.int64),
        "ea_train": ea_train,
        "val_group": np.concatenate(gva) if gva else np.zeros(0, dtype=np.int64),
        "ea_val": ea_val,
    }


# ----------------------------------------------------------------------------- v4: unlabelled target data
def _session_X_only(dataset: str, subject: int, session: int) -> np.ndarray:
    """Windows of one cached session, reading only the ``X`` and ``session`` arrays of the cache
    file (the labels in it are never loaded). Falls back to building the cache."""
    path = cache125.cache_path(dataset, subject)
    if path.exists():
        with np.load(path) as z:
            if "version" in z.files and int(z["version"]) == cache125.CACHE_VERSION:
                sess = z["session"]
                return np.asarray(z["X"][sess == session], dtype=np.float32)
    return np.asarray(cache125.cached_windows(dataset, subject)[session][0], dtype=np.float32)


def target_unlabeled(cfg: dict, loader=None, input_scale: float | None = None) -> dict:
    """v4: the unlabelled target session (``cfg["self_training"]``, default BCI IV 2a session 1)
    of every test subject, for self-training and as extra KD-only distillation windows.

    Returns ``{"X": {subject: (n, 22, 256) float32}, "y": {subject: int64 (n,) all -1},
    "ea": {subject: A (22, 22)}, "dataset", "session"}``. EA per subject from all of that
    session's trials (unlabelled); divided by ``input_scale`` when it is given (else µV, like
    ``v2_data``'s calib/test). **Only X is read**: with ``loader(dataset, subject) -> {session:
    (X, y)}`` the ``y`` element is never touched; the default reads the cache's ``X`` alone.
    The session must differ from the test session (``cfg["test"]["session"]``).
    """
    st = cfg.get("self_training") or {}
    test_cfg = cfg.get("test", {})
    dataset = st.get("dataset", test_cfg.get("dataset", cache125.TEST_DATASET))
    session = int(st.get("session", 1))
    if dataset == test_cfg.get("dataset", cache125.TEST_DATASET) and session == int(test_cfg.get("session", 2)):
        raise ValueError("the unlabelled target session must not be the test session")
    eps = float(cfg.get("ea_eps", DEFAULT_EA_EPS))
    X_out, y_out, ea = {}, {}, {}
    for subject in expand_subjects(test_cfg.get("subjects", list(range(1, 10)))):
        X = loader(dataset, subject)[session][0] if loader is not None else _session_X_only(dataset, subject, session)
        X = np.asarray(X, dtype=np.float32)
        A = ea_matrix(X, eps)
        Xa = apply_ea(X, A)
        if input_scale is not None:
            Xa = (Xa / np.float32(input_scale)).astype(np.float32)
        X_out[subject], y_out[subject], ea[subject] = Xa, np.full(len(Xa), -1, dtype=np.int64), A
    return {"X": X_out, "y": y_out, "ea": ea, "dataset": dataset, "session": session}


def session_labels_for_reporting_only(cfg: dict, loader=None) -> dict[int, np.ndarray]:
    """REPORTING ONLY, never called by a training path: the true labels of the unlabelled target
    session, used solely to report pseudo-label agreement after the pseudo-labels are fixed."""
    st = cfg.get("self_training") or {}
    dataset = st.get("dataset", cfg.get("test", {}).get("dataset", cache125.TEST_DATASET))
    session = int(st.get("session", 1))
    loader = loader or cache125.cached_windows
    return {s: np.asarray(loader(dataset, s)[session][1], dtype=np.int64)
            for s in expand_subjects(cfg.get("test", {}).get("subjects", list(range(1, 10))))}


__all__ = ["v2_data", "load_config", "expand_subjects", "training_subjects", "check_config", "target_unlabeled",
           "session_labels_for_reporting_only"]
