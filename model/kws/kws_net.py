"""Small DS-CNN keyword spotter (after Arm's "Hello Edge" DS-CNN), 49x40 log-mel in."""

import torch
import torch.nn as nn

from kws_features import CLASSES, NFR, NMEL

INPUT_SHAPE = (1, 1, NFR, NMEL)


def ds_block(c):
    return nn.Sequential(
        nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
        nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
    )


class KwsNet(nn.Module):
    def __init__(self, c=64, n_classes=len(CLASSES)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, c, (10, 4), stride=2, padding=(5, 1), bias=False),
                                  nn.BatchNorm2d(c), nn.ReLU())
        self.blocks = nn.Sequential(*[ds_block(c) for _ in range(4)])
        self.fc = nn.Linear(c, n_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.fc(x.mean(dim=(2, 3)))  # global average (25 x 20)
