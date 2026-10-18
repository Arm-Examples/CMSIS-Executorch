"""Keyword-spotting features, shared by training and checked against the C
version in board/DevKit-E8/voice.c (keep the two identical).

1 s of 8 kHz audio -> 49 frames x 40 log-mel bands:
  remove DC, scale to [-1, 1); gain-normalise so the loudest frame has RMS 0.1
  (gain capped at MAX_GAIN so silence is not blown up); 30 ms Hamming frames
  every 20 ms, 256-point FFT, power, 40 triangular mel filters (20 Hz - 4 kHz),
  log(e + 1e-6), minus each band's mean over the second, divided by FEAT_STD.
"""

import numpy as np
import torch

SR = 8000
N_SAMPLES = 8000
FRAME = 240
HOP = 160
NFFT = 256
NBIN = NFFT // 2 + 1
NMEL = 40
NFR = (N_SAMPLES - FRAME) // HOP + 1  # 49
TARGET_RMS = 0.1
MAX_GAIN = 400.0
FEAT_STD = 3.0

WORDS = ["up", "down", "left", "right", "go", "stop", "yes", "no", "happy"]
CLASSES = WORDS + ["_silence_", "_unknown_"]


def hamming():
    return (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FRAME) / (FRAME - 1))).astype(np.float32)


def mel_matrix():
    h2m = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    m2h = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    m0, m1 = h2m(20.0), h2m(4000.0)
    edge = [m2h(m0 + (m1 - m0) * k / (NMEL + 1)) * NFFT / SR for k in range(NMEL + 2)]
    w = np.zeros((NMEL, NBIN), np.float32)
    for m in range(NMEL):
        for b in range(NBIN):
            f = float(b)
            if edge[m] < f <= edge[m + 1]:
                w[m, b] = (f - edge[m]) / (edge[m + 1] - edge[m])
            elif edge[m + 1] < f < edge[m + 2]:
                w[m, b] = (edge[m + 2] - f) / (edge[m + 2] - edge[m + 1])
    return w


_WIN = torch.from_numpy(hamming())
_MEL = torch.from_numpy(mel_matrix())


def features_torch(x):
    """x: (B, 8000) float in [-1, 1). Returns (B, 1, 49, 40)."""
    x = x - x.mean(1, keepdim=True)
    fr = x.unfold(1, FRAME, HOP)                       # (B, 49, 240)
    rms = fr.pow(2).mean(2).sqrt().amax(1)             # loudest frame
    g = (TARGET_RMS / rms.clamp_min(1e-9)).clamp(max=MAX_GAIN)
    fr = fr * g[:, None, None] * _WIN
    spec = torch.fft.rfft(fr, NFFT).abs().pow(2)       # (B, 49, 129)
    lm = torch.log(spec @ _MEL.T + 1e-6)               # (B, 49, 40)
    lm = lm - lm.mean(1, keepdim=True)
    return (lm / FEAT_STD).unsqueeze(1)


def features_np(pcm16):
    """Reference for one clip of int16 samples, written the way the C does it."""
    x = pcm16.astype(np.float64) / 32768.0
    x = x - x.mean()
    win, mel = hamming().astype(np.float64), mel_matrix().astype(np.float64)
    rms = max(np.sqrt(np.mean(x[f * HOP:f * HOP + FRAME] ** 2)) for f in range(NFR))
    g = min(TARGET_RMS / max(rms, 1e-9), MAX_GAIN)
    out = np.zeros((NFR, NMEL))
    for f in range(NFR):
        fr = x[f * HOP:f * HOP + FRAME] * g * win
        p = np.abs(np.fft.rfft(fr, NFFT)) ** 2
        out[f] = np.log(mel @ p + 1e-6)
    out -= out.mean(0)
    return (out / FEAT_STD).astype(np.float32)
