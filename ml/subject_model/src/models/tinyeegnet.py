"""TinyEEGNet: fixed 22-channel, 500-sample windows and two logits.

Explicit asymmetric zero padding preserves time length for even kernels.
ReLU follows the README's Arm-friendly alternative to ELU. BatchNorm remains
unfolded during training; export/backends must fold it for eventual deployment.
"""
import torch
from torch import nn


class TinyEEGNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, 8, (1, 64), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (22, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(16, 16, (1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(16 * 15, 2)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))
