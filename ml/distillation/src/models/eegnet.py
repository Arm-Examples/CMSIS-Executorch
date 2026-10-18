"""EEGNet-8,2 reference (Lawhern et al., 2018): ELU activations and max-norm constraints.

Float-only comparison row; same I/O as TinyEEGNet (``(B, 1, 22, 500)`` -> ``(B, 2)``).
Call ``apply_max_norm()`` after each optimizer step (``src.train`` does when present).
"""

from __future__ import annotations

import torch
from torch import nn

from .tinyeegnet import _feature_length


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 500,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_t: int = 64,
        kern_s: int = 16,
        dropout: float = 0.25,
        depthwise_max_norm: float = 1.0,
        fc_max_norm: float = 0.25,
    ) -> None:
        super().__init__()
        self.depthwise_max_norm = depthwise_max_norm
        self.fc_max_norm = fc_max_norm
        self.conv_temporal = nn.Conv2d(1, F1, (1, kern_t), padding=(0, kern_t // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.conv_sep_dw = nn.Conv2d(F1 * D, F1 * D, (1, kern_s), groups=F1 * D, padding=(0, kern_s // 2), bias=False)
        self.conv_sep_pw = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.n_features = F2 * _feature_length(n_samples, kern_t, kern_s, 4, 8)
        self.fc = nn.Linear(self.n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv_temporal(x))
        x = self.drop1(self.pool1(self.act1(self.bn2(self.conv_spatial(x)))))
        x = self.drop2(self.pool2(self.act2(self.bn3(self.conv_sep_pw(self.conv_sep_dw(x))))))
        return self.fc(torch.flatten(x, 1))

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        """Keras-style max-norm: depthwise filters (per output channel) and the dense layer."""
        w = self.conv_spatial.weight  # (16, 1, 22, 1)
        norm = w.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12).view(-1, 1, 1, 1)
        w.mul_(torch.clamp(norm, max=self.depthwise_max_norm) / norm)
        w = self.fc.weight  # (2, 240); Keras constrains per output unit
        norm = w.norm(dim=1, keepdim=True).clamp_min(1e-12)
        w.mul_(torch.clamp(norm, max=self.fc_max_norm) / norm)
