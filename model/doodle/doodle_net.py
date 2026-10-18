"""Doodle classifier: a 28x28 finger drawing in, one score per class out (Ethos-U85).

White strokes on black, values 0..1, like Google's Quick, Draw! bitmaps. Only
layers the NPU runs natively (3x3 conv, ReLU, 2x2 max pool, linear).
"""

import torch
from torch import nn

CLASSES = ["star", "sun", "house", "cat", "flower", "fish", "smiley face", "lightning"]
INPUT_SHAPE = (1, 1, 28, 28)


class DoodleNet(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 3x3
        )
        self.head = nn.Sequential(nn.Linear(64 * 3 * 3, 96), nn.ReLU(), nn.Linear(96, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.features(x), 1))
