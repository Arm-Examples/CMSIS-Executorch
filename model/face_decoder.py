"""The face decoder: a latent vector in, a SIZE x SIZE grey face out.

Built from layers the Ethos-U85 runs natively (linear, nearest-neighbour
upsample, 3x3 convolution, ReLU, sigmoid) so the whole network is one NPU
command stream.
"""

import torch
from torch import nn

LATENT = 16
SIZE = 32


class FaceDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(LATENT, 64 * 4 * 4)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(64, 48, 3, padding=1), nn.ReLU())   # 8x8
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(48, 24, 3, padding=1), nn.ReLU())   # 16x16
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(24, 1, 3, padding=1))               # 32x32

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc(z)).reshape(-1, 64, 4, 4)
        return torch.sigmoid(self.up3(self.up2(self.up1(h))))
