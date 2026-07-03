# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""The example model: a tiny int8 CNN classifier for Ethos-U85.

Small enough to export in seconds and fully partition onto the NPU, but real
enough to exercise conv / relu / pool / linear through Vela. Input is a single
16x16 RGB image; output is a 10-class logit vector.
"""

import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        self.classifier = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_model() -> nn.Module:
    return TinyCNN().eval()


def get_example_inputs() -> tuple[torch.Tensor, ...]:
    return (torch.randn(1, 3, 16, 16),)
