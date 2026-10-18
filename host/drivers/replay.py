"""Replay data and the UART framing shared by bridge.py, the sim board and the tests.

Host -> board (little endian, contract in README):
* data frame    ``AA 55 | u8 n_ch | u16 seq | i16[n_ch]``  one 250 Hz sample, µV x 20.48
* control frame ``AA 56 | u8 kind | u8 arg | u16 seq``      1 SET_SUBJECT, 2 CUE, 3 RESET, 4 FREE_RUN
"""

from __future__ import annotations

import bisect
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIRST_TEST_TRIAL, FS, N_TRIALS, UV_SCALE, replay_path  # noqa: E402

SYNC = 0xAA
DATA = 0x55
CTRL = 0x56
# Control kinds = firmware/src/uart_rx.h enum ControlKind (checked by tests/test_contract.py).
CTRL_SET_SUBJECT, CTRL_CUE, CTRL_RESET, CTRL_FREE_RUN = 1, 2, 3, 4
CTRL_NAMES = {CTRL_RESET: "RESET", CTRL_SET_SUBJECT: "SET_SUBJECT", CTRL_CUE: "CUE", CTRL_FREE_RUN: "FREE_RUN"}
LABEL_UNKNOWN = 255
CTRL_LEN = 6
DEFAULT_UV_SCALE = UV_SCALE


def data_frame_len(n_ch: int = 22) -> int:
    return 5 + 2 * n_ch


def pack_data_frame(seq: int, values) -> bytes:
    v = np.asarray(values, dtype="<i2").ravel()
    return struct.pack("<BBBH", SYNC, DATA, len(v), seq & 0xFFFF) + v.tobytes()


def pack_data_frames(seq0: int, block: np.ndarray) -> bytes:
    """``(k, n_ch)`` int16 samples -> k consecutive data frames starting at ``seq0`` (fast path)."""
    block = np.ascontiguousarray(block, dtype="<i2")
    k, n_ch = block.shape
    out = np.empty((k, data_frame_len(n_ch)), dtype=np.uint8)
    out[:, 0], out[:, 1], out[:, 2] = SYNC, DATA, n_ch
    seq = (np.arange(seq0, seq0 + k) & 0xFFFF).astype("<u2")
    out[:, 3:5] = seq.view(np.uint8).reshape(k, 2)
    out[:, 5:] = block.view(np.uint8).reshape(k, 2 * n_ch)
    return out.tobytes()


def pack_control_frame(kind: int, arg: int = 0, seq: int = 0) -> bytes:
    return struct.pack("<BBBBH", SYNC, CTRL, kind, arg & 0xFF, seq & 0xFFFF)


@dataclass
class DataBatch:
    seq: np.ndarray        # (k,) uint16
    x: np.ndarray          # (k, n_ch) int16


@dataclass
class Control:
    kind: int
    arg: int
    seq: int


class FrameParser:
    """Incremental parser; ``feed(bytes)`` returns a list of ``DataBatch`` / ``Control``.
    Consecutive data frames are returned as one batch. Garbage is skipped byte by byte
    (counted in ``errors``) until the next valid header."""

    def __init__(self, n_ch: int = 22):
        self.n_ch = n_ch
        self.flen = data_frame_len(n_ch)
        self.buf = bytearray()
        self.errors = 0

    def feed(self, data: bytes) -> list:
        self.buf += data
        out: list = []
        buf = self.buf
        i = 0
        while len(buf) - i >= 2:
            if buf[i] != SYNC or buf[i + 1] not in (DATA, CTRL):
                i += 1
                self.errors += 1
                continue
            if buf[i + 1] == CTRL:
                if len(buf) - i < CTRL_LEN:
                    break
                _, _, kind, arg, seq = struct.unpack_from("<BBBBH", buf, i)
                if kind not in CTRL_NAMES:
                    i += 1
                    self.errors += 1
                    continue
                out.append(Control(kind, arg, seq))
                i += CTRL_LEN
                continue
            n = (len(buf) - i) // self.flen
            if n == 0:
                break
            arr = np.frombuffer(bytes(buf[i:i + n * self.flen]), dtype=np.uint8).reshape(n, self.flen)
            ok = (arr[:, 0] == SYNC) & (arr[:, 1] == DATA) & (arr[:, 2] == self.n_ch)
            m = n if ok.all() else int(np.argmin(ok))
            if m == 0:  # header claims data but n_ch is wrong
                i += 1
                self.errors += 1
                continue
            rows = arr[:m]
            seq = rows[:, 3:5].copy().view("<u2").ravel()
            x = rows[:, 5:].copy().view("<i2").reshape(m, self.n_ch)
            out.append(DataBatch(seq.astype(np.uint16), x.astype(np.int16)))
            i += m * self.flen
        del buf[:i]
        return out


# ---- replay file ------------------------------------------------------------------------------
@dataclass
class Replay:
    subject: int
    session: int
    x_int16: np.ndarray       # (22, n)
    cue_sample: np.ndarray    # (144,)
    label: np.ndarray         # (144,)
    fs: int
    uv_scale: float           # int16 counts per µV
    channel_order: list[str]


def load_replay(subject: int, session: int = 2, path: str | Path | None = None) -> Replay:
    p = Path(path) if path else replay_path(subject, session)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run python host/export_replay.py --subject {subject}")
    with np.load(p) as z:
        r = Replay(int(z["subject"]), int(z["session"]), z["x_int16"], z["cue_sample"].astype(np.int64),
                   z["label"].astype(np.int64), int(z["fs"]),
                   float(z["uv_scale"]) if "uv_scale" in z.files else DEFAULT_UV_SCALE,
                   [str(c) for c in z["channel_order"]] if "channel_order" in z.files else [])
    if r.fs != FS or len(r.cue_sample) != N_TRIALS:
        raise ValueError(f"{p}: fs {r.fs}, {len(r.cue_sample)} trials")
    return r


@dataclass
class Plan:
    """What one replay session streams: recording samples ``[start, stop)`` and the cues."""
    subject: int
    start: int
    stop: int
    trials: list[int]          # session trial indices (0-based), cue index k -> trials[k]
    cue_rel: list[int]         # stream index of each cue
    labels: list[int]

    @property
    def n(self) -> int:
        return self.stop - self.start


def window_last_rel(cue_rel: int) -> int:
    """Stream index of the 250 Hz sample that completes the trial's 125 Hz window."""
    return 2 * (cue_rel // 2 + 62 + 256 - 1)


def plan_session(rep: Replay, trials: list[int], lead_in_s: float = 10.0, tail_s: float = 1.0) -> Plan:
    """Continuous stream from ``lead_in_s`` before the first cue (moved back to an even recording
    sample, so the board's decimation phase equals the training pipeline's) to ``tail_s`` after
    the last trial's window."""
    trials = sorted(trials)
    cue = rep.cue_sample
    start = max(0, int(cue[trials[0]]) - int(round(lead_in_s * rep.fs)))
    start -= start % 2
    last = int(cue[trials[-1]]) - start
    stop = min(rep.x_int16.shape[1], start + window_last_rel(last) + 1 + int(round(tail_s * rep.fs)))
    return Plan(rep.subject, start, stop, trials, [int(cue[t]) - start for t in trials],
                [int(rep.label[t]) for t in trials])


def iter_frames(rep: Replay, plan: Plan, free_run: bool = False, chunk: int = 0) -> Iterator[tuple[int, bytes, dict | None]]:
    """Yields ``(k, frame_bytes, event)`` in send order. ``k`` is the stream index of the next
    data frame (for pacing: data frame k is due at ``t0 + k / fs``). Control frames carry the
    seq of the data frame they precede; the session opens with RESET, SET_SUBJECT [, FREE_RUN].
    ``event`` is a dict for CUE frames (``{"event": "cue", ...}``), else ``None``.
    ``chunk > 0`` packs up to ``chunk`` consecutive data frames per item (no cue inside)."""
    yield 0, pack_control_frame(CTRL_RESET, 0, 0), None
    yield 0, pack_control_frame(CTRL_SET_SUBJECT, plan.subject, 0), None
    if free_run:
        yield 0, pack_control_frame(CTRL_FREE_RUN, 1, 0), None
    cues = {c: i for i, c in enumerate(plan.cue_rel)}
    xs = rep.x_int16[:, plan.start:plan.stop].T
    step = max(1, chunk)
    k = 0
    while k < plan.n:
        if k in cues:
            i = cues[k]
            ev = {"event": "cue", "trial": i, "session_trial": plan.trials[i], "label": plan.labels[i],
                  "seq": k & 0xFFFF, "subject": plan.subject}
            yield k, pack_control_frame(CTRL_CUE, plan.labels[i], k), ev
        end = min(plan.n, k + step)
        j = bisect.bisect_right(plan.cue_rel, k)
        if j < len(plan.cue_rel) and plan.cue_rel[j] < end:
            end = plan.cue_rel[j]
        yield k, pack_data_frames(k, xs[k:end]), None
        k = end


def default_trials() -> str:
    return f"{FIRST_TEST_TRIAL}-{N_TRIALS - 1}"
