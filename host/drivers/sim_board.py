"""In-process stand-in for the board (``bridge.py --board sim``).

Same interface as ``serial.Serial`` for what the bridge uses (``write``, ``readline``,
``close``, ``in_waiting``). It parses the UART frames, runs ``pipeline_ref.StreamingPipeline``
(firmware semantics: RESET, SET_SUBJECT, CUE before the cue frame, FREE_RUN) and answers each
completed window with the contract JSON line. Logits come from

* ``model="offline"`` (default): the offline int8 logits of that session trial
  (``host/reference/offline_int8_predictions.json``; float stream, Reproduce_Baselines), or
* ``model="int8"``: ``int8_oracle.Int8Oracle`` on the window itself (needs torch + executorch;
  this is exactly what the board's .pte computes on the int16 stream).

``lat_us`` is the host time spent on the window (preprocessing + lookup/model), labelled by
``"board":"sim"``; ``motion`` is 0 unless ``set_motion(True)``.
"""

from __future__ import annotations

import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import UV_SCALE, load_offline_predictions, load_params  # noqa: E402
from drivers.replay import (CTRL_CUE, CTRL_FREE_RUN, CTRL_RESET, CTRL_SET_SUBJECT,  # noqa: E402
                            Control, DataBatch, FrameParser)
from pipeline_ref import PipelineRef, StreamingPipeline  # noqa: E402


def softmax2(l: np.ndarray) -> np.ndarray:
    e = np.exp(l - l.max())
    return e / e.sum()


class SimBoard:
    def __init__(self, params: dict | None = None, uv_scale: float = UV_SCALE, model: str = "offline",
                 session_trials: list[int] | None = None, offline: dict | None = None):
        self.params = params if params is not None else load_params()
        self.ref = PipelineRef(self.params, uv_scale)
        self.pipe = StreamingPipeline(self.ref)
        self.parser = FrameParser(int(self.params["n_channels"]))
        self.lines: "queue.Queue[bytes]" = queue.Queue()
        self.t0 = time.monotonic()
        self.session_trials = list(session_trials or [])
        self.offline = offline if offline is not None else load_offline_predictions()
        self.oracle = None
        if model == "int8":
            from int8_oracle import Int8Oracle

            self.oracle = Int8Oracle(params=self.params)
        elif model != "offline":
            raise ValueError(f"unknown sim model {model!r}")
        self.model = model
        self.motion = False
        self.seq0 = None
        self.n_inf = 0
        self._emit(f"READY n_ch={self.params['n_channels']} model=tinyeegnet_int8 pte_bytes=11456 fs=250 "
                   f"fs_model=125 win=256 baud=921600 board=sim sim_model={model}")

    # ---- serial.Serial subset ------------------------------------------------------------
    @property
    def in_waiting(self) -> int:
        return self.lines.qsize()

    def write(self, data: bytes) -> int:
        for item in self.parser.feed(bytes(data)):
            if isinstance(item, Control):
                self._control(item)
            elif isinstance(item, DataBatch):
                self._data(item)
        return len(data)

    def readline(self, timeout: float | None = 0.1) -> bytes:
        try:
            return self.lines.get(timeout=timeout) if timeout else self.lines.get_nowait()
        except queue.Empty:
            return b""

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def set_motion(self, on: bool) -> None:
        self.motion = bool(on)

    # ---- board behaviour -------------------------------------------------------------------
    def _emit(self, line: str) -> None:
        self.lines.put((line + "\n").encode())

    def _ms(self) -> int:
        return int((time.monotonic() - self.t0) * 1000)

    def _control(self, c: Control) -> None:
        if c.kind == CTRL_RESET:
            self.pipe.reset()
            self.seq0 = None
            self._emit("# reset")
        elif c.kind == CTRL_SET_SUBJECT:
            self.pipe.set_subject(c.arg)
            self._emit(f"# subject {c.arg}")
        elif c.kind == CTRL_CUE:
            if not self.pipe.subject:
                self._emit("# cue_without_subject")
                return
            self.pipe.cue(c.arg)
        elif c.kind == CTRL_FREE_RUN:
            self.pipe.free_run = bool(c.arg)

    def _data(self, b: DataBatch) -> None:
        if self.seq0 is None:
            self.seq0 = int(b.seq[0])
        t = time.perf_counter()
        for inf in self.pipe.push(b.x):
            self._answer(inf, t)
            t = time.perf_counter()

    def _answer(self, inf, t_start: float) -> None:
        subject = self.pipe.subject
        if self.oracle is not None:
            logits = self.oracle.logits(inf.x_int8[None])[0]
        else:
            logits = np.zeros(2, np.float32)
            st = self.session_trials[inf.trial] if 0 <= inf.trial < len(self.session_trials) else None
            off = self.offline.get(subject)
            if st is not None and off is not None:
                hit = np.nonzero(off["trial"] == st)[0]
                if len(hit):
                    logits = off["logits"][hit[0]]
        raw_cls = int(np.argmax(logits))
        conf = float(softmax2(np.asarray(logits, np.float64))[raw_cls])
        motion = int(self.motion)
        rec = {"t": self._ms(), "seq": (self.seq0 + inf.n250_last) & 0xFFFF, "trial": inf.trial,
               "subject": subject, "cls": -1 if motion else raw_cls, "raw_cls": raw_cls,
               "conf": round(conf, 2), "logits": [round(float(v), 3) for v in logits],
               "lat_us": int((time.perf_counter() - t_start) * 1e6), "motion": motion,
               "label": inf.label, "board": "sim"}
        self.n_inf += 1
        self._emit(json.dumps(rec, separators=(",", ":")))
