# SPDX-License-Identifier: Apache-2.0
"""TinyEEGNet: an EEGNet-8,2 shaped classifier for 2 s / 22-channel motor-imagery windows.

Deployable variant of EEGNet (Lawhern et al., 2018): ReLU instead of ELU, no max-norm,
BatchNorm folded at export, no Softmax (logits out). Input ``(B, 1, 22, 500)`` float
(aligned µV / ``input_scale``), output ``(B, 2)`` logits, ~1.9k parameters.

Exposes the CMSIS-Executorch starter interface ``get_model`` / ``get_calibration_inputs``
so ``create_ai_layer.py`` can quantize and delegate it unchanged: copy this file,
``tinyeegnet_float.pt`` and ``calibration.npz`` next to each other into the starter's
``model/`` directory. An optional ``student.json`` beside this file (keys ``F1``, ``D``,
``F2``, ``kern_t``, ``kern_s``) sets the layer widths ``get_model`` builds; without it the
defaults below are used.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

INPUT_SHAPE = (1, 1, 22, 500)  # (batch, 1, channels, samples) NCHW
N_CLASSES = 2
_HERE = Path(__file__).resolve().parent
FLOAT_CHECKPOINT = _HERE / "tinyeegnet_float.pt"
CALIBRATION_FILE = _HERE / "calibration.npz"
STUDENT_CONFIG = _HERE / "student.json"
STUDENT_KEYS = ("F1", "D", "F2", "kern_t", "kern_s")


def _feature_length(n_samples: int, kern_t: int, kern_s: int, pool1: int, pool2: int) -> int:
    n = n_samples + 2 * (kern_t // 2) - kern_t + 1  # temporal conv, padding kern/2
    n = n // pool1
    n = n + 2 * (kern_s // 2) - kern_s + 1  # separable conv, padding kern/2
    return n // pool2


class TinyEEGNet(nn.Module):
    """Conv(1->8, 1x64) -> BN -> depthwise Conv(8->16, 22x1) -> BN -> ReLU -> AvgPool 1x4
    -> Dropout -> separable Conv(16, 1x16 depthwise + 1x1) -> BN -> ReLU -> AvgPool 1x8
    -> Dropout -> Flatten(240) -> Linear(2)."""

    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 500,
        n_classes: int = N_CLASSES,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_t: int = 64,
        kern_s: int = 16,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.conv_temporal = nn.Conv2d(1, F1, (1, kern_t), padding=(0, kern_t // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.conv_sep_dw = nn.Conv2d(F1 * D, F1 * D, (1, kern_s), groups=F1 * D, padding=(0, kern_s // 2), bias=False)
        self.conv_sep_pw = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.n_features = F2 * _feature_length(n_samples, kern_t, kern_s, 4, 8)
        self.fc = nn.Linear(self.n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv_temporal(x))
        x = self.drop1(self.pool1(self.act1(self.bn2(self.conv_spatial(x)))))
        x = self.drop2(self.pool2(self.act2(self.bn3(self.conv_sep_pw(self.conv_sep_dw(x))))))
        return self.fc(torch.flatten(x, 1))


def student_widths() -> dict[str, int]:
    """Layer widths from ``student.json`` beside this file (only ``STUDENT_KEYS``), or ``{}``."""
    if not STUDENT_CONFIG.is_file():
        return {}
    import json

    cfg = json.loads(STUDENT_CONFIG.read_text())
    return {k: int(cfg[k]) for k in STUDENT_KEYS if k in cfg}


def get_model(input_shape: tuple[int, ...] = INPUT_SHAPE) -> nn.Module:
    """Starter entry point: TinyEEGNet in eval mode, with the trained weights from
    ``tinyeegnet_float.pt`` beside this file when present (random init with a fixed
    seed otherwise, so an export without weights is still deterministic). Layer widths
    come from ``student.json`` beside this file when present, else the defaults."""
    _, _, n_channels, n_samples = input_shape
    torch.manual_seed(0)
    model = TinyEEGNet(n_channels=n_channels, n_samples=n_samples, **student_widths())
    if FLOAT_CHECKPOINT.is_file():
        state = torch.load(FLOAT_CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model.eval()


def get_calibration_inputs(
    input_shape: tuple[int, ...] = INPUT_SHAPE, calibration_samples: int = 200
) -> list[torch.Tensor]:
    """Float samples for quantization calibration; the first one is also the export example.

    Reads ``calibration.npz`` (key ``x``, shape ``(N, 1, 22, 500)`` or ``(N, 22, 500)``,
    already aligned and scaled training windows) beside this file and returns up to
    ``calibration_samples`` tensors of ``input_shape`` (fewer if the file holds fewer; a
    ``ValueError`` if it holds none). Only without the file does it fall back to seeded
    random normal samples (unit scale, like the scaled real data).
    """
    samples: list[torch.Tensor] = []
    if CALIBRATION_FILE.is_file():
        import numpy as np

        x = np.load(CALIBRATION_FILE)["x"]
        if x.shape[0] == 0:
            raise ValueError(f"{CALIBRATION_FILE} holds no calibration windows")
        x = x.reshape(x.shape[0], *input_shape[1:])
        # Real windows only: never pad with random data; fewer than requested is fine.
        for i in range(min(calibration_samples, x.shape[0])):
            samples.append(torch.from_numpy(np.ascontiguousarray(x[i : i + 1])).to(torch.float32))
        return samples
    for i in range(calibration_samples):
        samples.append(torch.randn(input_shape, generator=torch.Generator().manual_seed(i)))
    return samples
