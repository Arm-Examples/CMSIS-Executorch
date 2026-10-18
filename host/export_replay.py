"""Export one BCI IV 2a session as the int16 replay stream the bridge sends to the board.

    python host/export_replay.py --subject 3 [--session 2] [--out host/data/A03_session2.npz]

Output npz (git-ignored, ~3 MB compressed per session):

* ``x_int16`` int16 (22, n): µV × 20.48 (the 2a native 100/2048 µV step, lossless; int16 = ±1600 µV), channels in
  ``params.json`` ``channel_order``; this is exactly the UART payload.
* ``cue_sample`` int64 (144,): 0-based sample of each left/right cue in recording order
  (the 2a marker + 2.0 s, as ``Reproduce_Baselines.src.data.download.load_raw`` gives it).
* ``label`` int64 (144,): 0 left hand, 1 right hand. ``fs`` 250, ``subject``, ``session``.

Needs the Reproduce_Baselines environment (mne, moabb, scipy) and its MOABB cache; set
``MOABB_DOWNLOAD_PROVIDER=upstream MNE_DATA=~/mne_data``. Nothing is downloaded if the files
are already cached.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, N_TRIALS, RB_DIR, UV_SCALE, load_params, replay_path  # noqa: E402

GDF_TO_CLASS = {769: 0, 770: 1}
UV_PER_V = 1e6


def load_session_uv(subject: int, session: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Float µV (22, n) float64 as Reproduce_Baselines sees it, cue samples, labels, channel names."""
    if str(RB_DIR) not in sys.path:
        sys.path.insert(0, str(RB_DIR))
    os.environ.setdefault("MOABB_DOWNLOAD_PROVIDER", "upstream")
    from src.data.download import load_raw  # Reproduce_Baselines

    raw = load_raw(subject)[session]
    if abs(raw.info["sfreq"] - FS) > 1e-6:
        raise ValueError(f"expected {FS} Hz, got {raw.info['sfreq']}")
    ev = np.asarray(raw.info["temp"]["events"], dtype=np.int64)
    keep = np.isin(ev[:, 2], list(GDF_TO_CLASS))
    cue = ev[keep, 0].astype(np.int64)
    label = np.array([GDF_TO_CLASS[int(c)] for c in ev[keep, 2]], dtype=np.int64)
    x_uv = raw.get_data() * UV_PER_V
    return x_uv, cue, label, list(raw.ch_names)


def to_int16(x_uv: np.ndarray, uv_scale: float = UV_SCALE) -> tuple[np.ndarray, int]:
    q = np.rint(np.asarray(x_uv, dtype=np.float64) * uv_scale)
    n_clip = int(np.count_nonzero((q < -32768) | (q > 32767)))
    return np.clip(q, -32768, 32767).astype(np.int16), n_clip


def export(subject: int, session: int, out: Path, params: dict, uv_scale: float = UV_SCALE) -> dict:
    x_uv, cue, label, names = load_session_uv(subject, session)
    order = list(params["channel_order"])
    if names != order:
        raise AssertionError(f"raw channel order {names} != params channel_order {order}")
    if len(cue) != N_TRIALS:
        raise AssertionError(f"subject {subject} session {session}: {len(cue)} left/right trials, expected {N_TRIALS}")
    if np.any(np.diff(cue) <= 0):
        raise AssertionError("cue samples not in recording order")
    x16, n_clip = to_int16(x_uv, uv_scale)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x_int16=x16, cue_sample=cue, label=label, fs=np.int64(FS),
                        subject=np.int64(subject), session=np.int64(session),
                        channel_order=np.array(order), uv_scale=np.float64(uv_scale))
    return {"subject": subject, "n": x16.shape[1], "clipped": n_clip, "out": str(out),
            "left": int((label == 0).sum()), "right": int((label == 1).sum()),
            "max_abs_uv": float(np.abs(x_uv).max()), "int_dev": float(np.abs(x_uv * uv_scale - np.rint(x_uv * uv_scale)).max())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", default="3", help="1..9, comma list, or 'all'")
    ap.add_argument("--session", type=int, default=2)
    ap.add_argument("--out", default=None, help="output npz (single subject only)")
    ap.add_argument("--params", default=None)
    ap.add_argument("--uv-scale", type=float, default=UV_SCALE,
                    help="int16 counts per µV (UART contract: 20.48 = the 2a native step 100/2048 µV, lossless)")
    a = ap.parse_args(argv)
    subjects = list(range(1, 10)) if a.subject == "all" else [int(s) for s in a.subject.split(",")]
    if a.out and len(subjects) > 1:
        ap.error("--out needs a single --subject")
    params = load_params(a.params)
    for s in subjects:
        out = Path(a.out) if a.out else replay_path(s, a.session)
        r = export(s, a.session, out, params, a.uv_scale)
        print(f"A0{s} session {a.session}: {r['n']} samples ({r['n'] / FS / 60:.1f} min), "
              f"{r['left']} left + {r['right']} right, max |uV| {r['max_abs_uv']:.2f}, {r['clipped']} clipped, "
              f"max dev from integer counts {r['int_dev']:.1e} -> {r['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
