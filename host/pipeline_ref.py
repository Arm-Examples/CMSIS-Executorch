"""Python reference of the board's per-sample pipeline (``Reproduce_Baselines/docs/v2_handoff.md``).

Input is the stream the board receives: int16 counts = µV × 20.48 per channel at 250 Hz, channels in
``params["channel_order"]``, sample 0 = the first data frame after RESET. Per sample:

1. µV = x * (1 / 20.48) = x * 0.048828125 (exact), CAR (subtract the mean of the 22 channels),
2. causal band-pass ``sosfilt(params["sos"])`` (DF2-T, 4 sections, state never reset),
3. keep samples whose stream index n_250 is even (decimation phase 0 from the first streamed
   sample) -> 125 Hz,
then per trial with cue at stream sample ``cue_rel``: ``cue_125 = cue_rel // 2``, window =
125 Hz samples ``[cue_125 + 62, cue_125 + 318)``, ``ea_matrices[subject] @ w``,
``/ input_scale``, ``round(x / input_quant_scale) + input_zero_point`` clamped to int8.

Numerics: this reference filters in float64 (``scipy.signal.sosfilt``; filtering the whole
stream at once or chunk by chunk with carried ``zi`` gives identical bits), casts the window to
float32, applies EA with a float64 einsum cast back to float32 and scales/quantizes in float32,
exactly like ``Reproduce_Baselines`` (``cache125.windows_2a`` -> ``align.apply_ea`` ->
``train.to_model_input`` -> ``quantize.quantize_input``). On the float µV stream from the
recording start it reproduces ``test_vectors.npz`` bit for bit (``host/tests``). The firmware
(``firmware/src/preprocess.cpp``) follows the same precision plan: CAR and the SOS filter in
float64, the 125 Hz ring buffer in float32, EA accumulated in float64 and cast to float32, then
``/ input_scale``, ``/ input_quant_scale``, round half to even and clamp in float32. What must
match is the **int8 window**. ``emulate_float32`` is a fully float32 variant, kept only to show
how many int8 elements a float32-only implementation would flip (one step, a handful).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi  # noqa: F401  (sosfilt_zi kept for experiments)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import UV_SCALE, load_params  # noqa: E402


class PipelineRef:
    """Batch and streaming reference of the firmware preprocessing for one params.json."""

    def __init__(self, params: dict | None = None, uv_scale: float = UV_SCALE):
        params = params if params is not None else load_params()
        self.params = params
        self.uv_scale = float(uv_scale)
        self.n_ch = int(params["n_channels"])
        self.sos = np.asarray(params["sos"], dtype=np.float64)
        self.ea = {int(k): np.asarray(v, dtype=np.float64) for k, v in params["ea_matrices"].items()}
        self.input_scale = np.float32(params["input_scale"])
        self.q_scale = np.float32(params["input_quant_scale"])
        self.zp = int(params["input_zero_point"])
        self.factor = int(params["decimation"]["factor"])
        self.win_start = int(params["window"]["start_samples_after_cue_125hz"])
        self.win_len = int(params["window"]["length"])
        if self.factor != 2 or int(params["decimation"]["phase"]) != 0:
            raise ValueError("reference implements decimation factor 2, phase 0 only")

    # ---- steps 1-3 -------------------------------------------------------------------------
    def to_uv(self, x_int16: np.ndarray) -> np.ndarray:
        # multiply by the reciprocal: 1 / 20.48 = 0.048828125 is exact, as on the board (float32)
        return np.asarray(x_int16, dtype=np.float64) * (1.0 / self.uv_scale)

    def car_filter(self, x_uv: np.ndarray, zi: np.ndarray | None = None):
        """CAR + band-pass of ``(22, n)`` float µV. Returns ``y`` (and the new ``zi`` if given)."""
        x = np.ascontiguousarray(x_uv, dtype=np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        if zi is None:
            return sosfilt(self.sos, x, axis=-1)
        return sosfilt(self.sos, x, axis=-1, zi=zi)

    def filter_decimate(self, x_uv: np.ndarray) -> np.ndarray:
        """Whole stream ``(22, n)`` float µV -> 125 Hz ``(22, ceil(n / 2))`` float64."""
        return self.car_filter(x_uv)[:, :: self.factor]

    # ---- steps 5-8 -------------------------------------------------------------------------
    def window_bounds(self, cue_rel: int) -> tuple[int, int]:
        """125 Hz ``[a, b)`` of the trial whose cue is stream sample ``cue_rel``."""
        c125 = int(cue_rel) // self.factor
        return c125 + self.win_start, c125 + self.win_start + self.win_len

    def last_sample_250(self, cue_rel: int) -> int:
        """Stream index (250 Hz) of the sample that completes the window."""
        return (self.window_bounds(cue_rel)[1] - 1) * self.factor

    def window(self, y125: np.ndarray, cue_rel: int) -> np.ndarray:
        a, b = self.window_bounds(cue_rel)
        if a < 0 or b > y125.shape[-1]:
            raise ValueError(f"window [{a}, {b}) outside the 125 Hz stream of {y125.shape[-1]} samples")
        return np.asarray(y125[:, a:b], dtype=np.float32)

    def model_float(self, w: np.ndarray, subject: int) -> np.ndarray:
        """``(n, 22, 256)`` float32 µV windows -> float32 model input ``(n, 1, 22, 256)``."""
        w = np.asarray(w, dtype=np.float32).reshape(-1, self.n_ch, self.win_len)
        xa = np.einsum("cd,ndt->nct", self.ea[int(subject)], w).astype(np.float32)
        return xa[:, None] / self.input_scale

    def quantize(self, w: np.ndarray, subject: int) -> np.ndarray:
        """``(n, 22, 256)`` float32 µV windows -> int8 ``(n, 1, 22, 256)``."""
        x = self.model_float(w, subject)
        q = np.round(x / self.q_scale) + self.zp
        return np.clip(q, -128, 127).astype(np.int8)

    def windows(self, x_uv: np.ndarray, cues_rel, subject: int) -> np.ndarray:
        y125 = self.filter_decimate(x_uv)
        w = np.stack([self.window(y125, c) for c in cues_rel]) if len(cues_rel) else \
            np.zeros((0, self.n_ch, self.win_len), np.float32)
        return self.quantize(w, subject)

    def dequantize(self, x_int8: np.ndarray) -> np.ndarray:
        """What the board feeds the float-I/O .pte: ``(x_int8 - zp) * q_scale``."""
        return (np.asarray(x_int8, dtype=np.float32) - self.zp) * self.q_scale


def windows_for_trials(x_int16_stream: np.ndarray, cue_samples_rel, subject: int,
                       params: dict | None = None, units: str = "int16",
                       uv_scale: float = UV_SCALE) -> np.ndarray:
    """int8 ``(n, 1, 22, 256)`` windows for cues given as stream-relative 250 Hz indices.

    ``x_int16_stream`` is ``(22, n)``; ``units="int16"`` (µV × 20.48, the UART payload) or
    ``"uV"`` (float µV, used to check against Reproduce_Baselines)."""
    ref = PipelineRef(params, uv_scale)
    x = ref.to_uv(x_int16_stream) if units == "int16" else np.asarray(x_int16_stream, dtype=np.float64)
    return ref.windows(x, list(cue_samples_rel), subject)


# ---- streaming (what bridge --board sim runs) -----------------------------------------------
@dataclass
class Inference:
    trial: int                 # cue index since RESET, -1 = free run
    label: int                 # CUE arg (255 unknown -> -1)
    n250_last: int             # stream index of the window's last sample
    x_int8: np.ndarray         # (1, 22, 256)


@dataclass
class StreamingPipeline:
    """Sample-by-sample (chunked) twin of ``PipelineRef.windows`` with firmware semantics:
    RESET, SET_SUBJECT, CUE (applies to the next data frame), FREE_RUN."""

    ref: PipelineRef
    ring_len: int = 1024
    subject: int = 0
    free_run: bool = False
    free_stride: int = 32
    n250: int = 0
    n125: int = 0
    cue_count: int = 0
    zi: np.ndarray = field(default=None)  # type: ignore[assignment]
    ring: np.ndarray = field(default=None)  # type: ignore[assignment]
    pending: list = field(default_factory=list)  # (end_125, cue_rel, trial, label)
    next_cue: tuple | None = None

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        self.zi = np.zeros((self.ref.sos.shape[0], self.ref.n_ch, 2), dtype=np.float64)
        self.ring = np.zeros((self.ref.n_ch, self.ring_len), dtype=np.float64)
        self.n250 = self.n125 = self.cue_count = 0
        self.pending = []
        self.next_cue = None

    def set_subject(self, s: int) -> None:
        if int(s) not in self.ref.ea:
            raise ValueError(f"no EA matrix for subject {s}")
        self.subject = int(s)

    def cue(self, label: int) -> None:
        """CUE control frame: the next data frame is the cue sample."""
        self.next_cue = (self.cue_count, -1 if label == 255 else int(label))
        self.cue_count += 1

    def push(self, x_int16: np.ndarray) -> list[Inference]:
        """Data frames ``(k, 22)`` int16 (k >= 1). Returns completed inferences in order."""
        x = np.asarray(x_int16, dtype=np.int16).reshape(-1, self.ref.n_ch)
        if self.next_cue is not None:
            trial, label = self.next_cue
            a, b = self.ref.window_bounds(self.n250)
            self.pending.append((b, self.n250, trial, label))
            self.next_cue = None
        y, self.zi = self.ref.car_filter(self.ref.to_uv(x.T), zi=self.zi)
        out: list[Inference] = []
        for j in range(y.shape[1]):
            n = self.n250 + j
            if n % self.ref.factor:
                continue
            self.ring[:, self.n125 % self.ring_len] = y[:, j]
            self.n125 += 1
            for item in [p for p in self.pending if p[0] == self.n125]:
                self.pending.remove(item)
                end, cue_rel, trial, label = item
                out.append(Inference(trial, label, n, self._infer(end)))
            if self.free_run and self.n125 >= self.ref.win_len and self.n125 % self.free_stride == 0:
                out.append(Inference(-1, -1, n, self._infer(self.n125)))
        self.n250 += y.shape[1]
        return out

    def _infer(self, end_125: int) -> np.ndarray:
        idx = np.arange(end_125 - self.ref.win_len, end_125) % self.ring_len
        w = self.ring[:, idx].astype(np.float32)
        return self.ref.quantize(w[None], self.subject)[0]


# ---- float32 firmware emulation (estimate only) ---------------------------------------------
def emulate_float32(x_int16_stream: np.ndarray, cues_rel, subject: int, params: dict | None = None,
                    uv_scale: float = UV_SCALE) -> np.ndarray:
    """Per-sample fully-float32 variant of the pipeline (not what the firmware does) (sequential channel sum for CAR, DF2-T in
    float32, float32 EA accumulate, round half to even). Only used to estimate how far a
    float32 board can drift from the float64 reference in int8 steps."""
    ref = PipelineRef(params, uv_scale)
    sos = ref.sos.astype(np.float32)
    x = np.asarray(x_int16_stream, dtype=np.int16)
    n_ch, n = x.shape
    z = np.zeros((sos.shape[0], 2, n_ch), np.float32)
    y125 = np.zeros((n_ch, (n + 1) // 2), np.float32)
    inv = np.float32(1.0 / uv_scale)
    inv_n = np.float32(n_ch)
    for i in range(n):
        u = x[:, i].astype(np.float32) * inv
        acc = np.float32(0.0)
        for c in range(n_ch):
            acc = np.float32(acc + u[c])
        v = u - acc / inv_n
        for s in range(sos.shape[0]):
            b0, b1, b2, _, a1, a2 = sos[s]
            yv = b0 * v + z[s, 0]
            z[s, 0] = b1 * v - a1 * yv + z[s, 1]
            z[s, 1] = b2 * v - a2 * yv
            v = yv
        if i % 2 == 0:
            y125[:, i // 2] = v
    A = ref.ea[int(subject)].astype(np.float32)
    out = []
    for c in cues_rel:
        w = ref.window(y125, c)
        xa = np.zeros_like(w)
        for d in range(n_ch):  # sequential accumulate over input channels
            xa = xa + A[:, d : d + 1] * w[d : d + 1, :]
        xm = xa / ref.input_scale
        q = np.clip(np.round(xm / ref.q_scale) + ref.zp, -128, 127).astype(np.int8)
        out.append(q[None])
    return np.stack(out) if out else np.zeros((0, 1, n_ch, ref.win_len), np.int8)
