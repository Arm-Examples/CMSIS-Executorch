# SPDX-License-Identifier: Apache-2.0
"""ShallowConvNet and DeepConvNet (Schirrmeister et al., 2017) for 125 Hz / 22-channel windows.

Float-only teacher members for the v2 distillation (``src.train_teacher``); never exported to
the NPU. Same I/O as TinyEEGNet: ``(B, 1, 22, T)`` aligned µV / ``input_scale`` in, ``(B, 2)``
logits out. Kernel lengths are the published 250 Hz values roughly halved for 125 Hz.

With the defaults and ``T = 256``: ShallowConvNet 38,242 parameters, DeepConvNet 146,302
(BatchNorm included).
"""

from __future__ import annotations

import torch
from torch import nn


class ShallowConvNet(nn.Module):
    """Conv(1->F, 1xkern_t) -> Conv(F->F, Cx1, no bias) -> BN -> square -> AvgPool(1xpool, stride)
    -> log(clamp 1e-6) -> Dropout -> Flatten -> Linear(2)."""

    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 256,
        n_classes: int = 2,
        F: int = 40,
        kern_t: int = 13,
        pool: int = 37,
        stride: int = 7,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv_t = nn.Conv2d(1, F, (1, kern_t))
        self.conv_s = nn.Conv2d(F, F, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(F)
        self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
        self.drop = nn.Dropout(dropout)
        n = (n_samples - kern_t + 1 - pool) // stride + 1
        self.n_features = F * n
        self.fc = nn.Linear(self.n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv_s(self.conv_t(x)))
        x = self.pool(x * x)
        x = self.drop(torch.log(torch.clamp(x, min=1e-6)))
        return self.fc(torch.flatten(x, 1))


class DeepConvNet(nn.Module):
    """Block 1: Conv(1->25, 1xkern_t) -> Conv(25->25, Cx1, no bias) -> BN -> ELU -> MaxPool -> Dropout;
    blocks 2-4: Conv(1xkern_t, 50 / 100 / 200, no bias) -> BN -> ELU -> MaxPool -> Dropout;
    Flatten -> Linear(2)."""

    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 256,
        n_classes: int = 2,
        kern_t: int = 5,
        pool: int = 3,
        dropout: float = 0.5,
        filters: tuple[int, ...] = (25, 50, 100, 200),
    ) -> None:
        super().__init__()
        f0 = filters[0]
        layers: list[nn.Module] = [
            nn.Conv2d(1, f0, (1, kern_t)),
            nn.Conv2d(f0, f0, (n_channels, 1), bias=False),
            nn.BatchNorm2d(f0), nn.ELU(), nn.MaxPool2d((1, pool)), nn.Dropout(dropout),
        ]
        n = (n_samples - kern_t + 1) // pool
        for f_in, f_out in zip(filters[:-1], filters[1:]):
            layers += [nn.Conv2d(f_in, f_out, (1, kern_t), bias=False), nn.BatchNorm2d(f_out), nn.ELU(),
                       nn.MaxPool2d((1, pool)), nn.Dropout(dropout)]
            n = (n - kern_t + 1) // pool
        if n < 1:
            raise ValueError(f"n_samples {n_samples} too short for {len(filters)} blocks of kernel {kern_t}, pool {pool}")
        self.features = nn.Sequential(*layers)
        self.n_features = filters[-1] * n
        self.fc = nn.Linear(self.n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.features(x), 1))
