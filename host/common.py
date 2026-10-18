"""Shared paths and parameter loading for the host tools.

Defaults resolve in this order so the tools work both in a full workspace checkout and in
a standalone clone of Edge_Deploy:

* params: ``--params`` > ``firmware/model/params.json`` (fetched with the model) >
  ``../Reproduce_Baselines/artifacts/v4/preprocess_params.json``.
* offline predictions: ``host/reference/offline_int8_predictions.json`` (committed, written by
  ``make_vectors.py`` from ``Reproduce_Baselines/runs/v4soft/student_wide32_ft_seed0/int8_predictions.npz``,
  override with ``make_vectors.py --predictions``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HOST_DIR = Path(__file__).resolve().parent
REPO_DIR = HOST_DIR.parent
WORKSPACE_DIR = REPO_DIR.parent
RB_DIR = WORKSPACE_DIR / "Reproduce_Baselines"

DATA_DIR = HOST_DIR / "data"
REFERENCE_DIR = HOST_DIR / "reference"
OFFLINE_PRED_JSON = REFERENCE_DIR / "offline_int8_predictions.json"
# Shipped model: Reproduce_Baselines v0.5-distill (local bundle artifacts/v4; v0.3-distill = artifacts/v2
# is the previous model, same contract).
MODEL_TAG = "v0.5-distill"
MODEL_DESC = "wide32_ft seed 0"
RB_ARTIFACTS = RB_DIR / "artifacts" / "v4"
RB_PARAMS = RB_ARTIFACTS / "preprocess_params.json"
RB_TEST_VECTORS = RB_ARTIFACTS / "test_vectors.npz"
RB_INT8_PREDICTIONS = RB_DIR / "runs" / "v4soft" / "student_wide32_ft_seed0" / "int8_predictions.npz"
PARAM_CANDIDATES = (REPO_DIR / "firmware" / "model" / "params.json", RB_PARAMS)

# Session 2 of every 2a subject: trials 0..23 (0-based) calibrate EA, 24..143 are test trials.
FIRST_TEST_TRIAL = 24
N_TRIALS = 144
FS = 250
# UART contract: int16 count = round(µV * UV_SCALE). BCI IV 2a is stored in steps of 100/2048 µV,
# so 20.48 counts per µV is lossless (µV = count * 0.048828125, exact in float32); range ±1600 µV.
UV_SCALE = 20.48
UV_PER_COUNT = 100.0 / 2048.0
UDP_DEFAULT = "127.0.0.1:5555"


def params_path(path: str | Path | None = None) -> Path:
    if path:
        return Path(path)
    for p in PARAM_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("no params.json: run tools/fetch_model.sh or pass --params "
                            f"(looked in {', '.join(str(p) for p in PARAM_CANDIDATES)})")


def load_params(path: str | Path | None = None) -> dict:
    p = params_path(path)
    params = json.loads(Path(p).read_text())
    if params.get("version") != "v2":
        raise ValueError(f"{p}: params version {params.get('version')!r}, expected 'v2'")
    params["_path"] = str(p)
    return params


def replay_path(subject: int, session: int = 2) -> Path:
    return DATA_DIR / f"A0{subject}_session{session}.npz"


def load_offline_predictions(path: str | Path | None = None) -> dict[int, dict[str, np.ndarray]]:
    """``{subject: {"trial", "y_true", "logits"}}`` of the offline int8 model (float stream),
    session 2 test trials 24..143 (0-based). Reads the committed JSON, else the RB npz."""
    p = Path(path) if path else OFFLINE_PRED_JSON
    if p.suffix == ".json" and p.exists():
        d = json.loads(p.read_text())
        return {int(s): {"trial": np.asarray(v["trial"], dtype=np.int64),
                         "y_true": np.asarray(v["y_true"], dtype=np.int64),
                         "logits": np.asarray(v["logits"], dtype=np.float32)} for s, v in d["subjects"].items()}
    src = p if p.suffix == ".npz" and p.exists() else RB_INT8_PREDICTIONS
    with np.load(src) as z:
        out = {}
        for s in z["subjects"]:
            s = int(s)
            lg = z[f"logits_{s}"].astype(np.float32)
            out[s] = {"trial": np.arange(FIRST_TEST_TRIAL, FIRST_TEST_TRIAL + len(lg), dtype=np.int64),
                      "y_true": z[f"y_true_{s}"].astype(np.int64), "logits": lg}
    return out


def parse_trials(spec: str) -> list[int]:
    """``"24-143"`` / ``"33,34,40-42"`` -> sorted unique 0-based trial indices."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = [t for t in out if not 0 <= t < N_TRIALS]
    if bad:
        raise ValueError(f"trial indices out of range 0..{N_TRIALS - 1}: {bad}")
    return sorted(out)


def parse_hostport(s: str) -> tuple[str, int]:
    host, _, port = s.rpartition(":")
    return (host or "127.0.0.1"), int(port)
