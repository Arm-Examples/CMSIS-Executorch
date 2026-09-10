# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""The example model: a tiny CNN classifier, quantized to int8 for Ethos-U85 by create_ai_layer.py.

Small enough to export in seconds and fully partition onto the NPU, but real
enough to exercise conv / relu / pool / linear through Vela. Input is a single
16x16 RGB image (NCHW); output is a 10-class logit vector.
"""

import torch
from torch import nn

INPUT_SHAPE = (1, 3, 16, 16)


class TinyCNN(nn.Module):
    def __init__(self, input_shape: tuple[int, ...] = INPUT_SHAPE, num_classes: int = 10) -> None:
        super().__init__()
        _, channels, height, width = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        self.classifier = nn.Linear(16 * (height // 4) * (width // 4), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_model(input_shape: tuple[int, ...] = INPUT_SHAPE) -> nn.Module:
    # Fixed seed: the example uses untrained (random) weights, and a fixed seed
    # keeps the generated AI layer identical from one export to the next.
    torch.manual_seed(0)
    return TinyCNN(input_shape).eval()


def get_calibration_inputs(
    input_shape: tuple[int, ...] = INPUT_SHAPE, calibration_samples: int = 2
) -> list[torch.Tensor]:
    """Float samples for quantization calibration; the first one is also the export example.

    Fixed values keep the quantization parameters, and with them the output
    logits in the README, the same from one export to the next. Replace them
    with representative, preprocessed data when calibrating a trained model.
    """
    samples = [torch.full(input_shape, 0.5), torch.full(input_shape, -0.5)][:calibration_samples]
    for i in range(len(samples), calibration_samples):
        samples.append(torch.randn(input_shape, generator=torch.Generator().manual_seed(i)))
    return samples
