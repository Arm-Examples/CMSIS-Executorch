# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Student model shared by the export scripts. Copied verbatim from jobs/student.py
(StudentNHWC); keep both identical or the trained weights will not load."""
import math

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class StudentNHWC(nn.Module):
    """MobileNetV2, NHWC [0,1] input, ImageNet normalisation folded in, NPU-friendly head."""

    def __init__(self, num_classes: int, res: int, pretrained: bool = True):
        super().__init__()
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = base.features
        self.pool = nn.AvgPool2d(math.ceil(res / 32))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):
        x = (x.permute(0, 3, 1, 2) - self.mean) / self.std
        x = self.pool(self.features(x))
        return self.fc(self.dropout(x.reshape(x.shape[0], 1280)))
