"""The bunny decoder: 30 pose dials in, an RGB bunny on white out (Ethos-U85).

Linear(30->256)+ReLU -> Linear(->C0*s*s)+ReLU, reshape to C0 x s x s ->
4 x [nearest upsample x2 -> conv3x3 -> ReLU] -> conv3x3 -> 3 channels -> sigmoid.
size = 16 * s (96 -> s=6, 64 -> s=4). width 1.0 = channels 128,96,64,32;
width 0.5 = 64,48,32,16. Only layers the Ethos-U85 runs natively (no
transposed conv, no Group/InstanceNorm).

Dials go in normalised to [0,1]: eye (5) and mouth (6) weights as they are,
each of the 19 scalars with its (lo, hi) range from SCALARS (normalize_dials).
"""

import torch
from torch import nn

NDIAL = 30
LOGIT_CLAMP = 6.0  # see forward()
BASE_CHANNELS = (128, 96, 64, 32)


class BunnyDecoder(nn.Module):
    def __init__(self, size: int = 64, width: float = 0.5) -> None:
        super().__init__()
        if size % 16:
            raise ValueError("size must be a multiple of 16")
        self.size = size
        self.start = size // 16
        ch = [max(8, int(c * width)) for c in BASE_CHANNELS]
        self.c0 = ch[0]
        self.fc = nn.Sequential(nn.Linear(NDIAL, 256), nn.ReLU(),
                                nn.Linear(256, self.c0 * self.start * self.start), nn.ReLU())
        blocks, cin = [], self.c0
        for cout in ch:
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU()]
            cin = cout
        self.up = nn.Sequential(*blocks)
        self.out = nn.Conv2d(cin, 3, 3, padding=1)
        self.int8_head = False  # see prepare_for_int8()
        self.register_buffer("logit_shift", torch.full((1, 3, 1, 1), LOGIT_CLAMP), persistent=False)

    def forward(self, dials: torch.Tensor) -> torch.Tensor:
        h = self.fc(dials).reshape(-1, self.c0, self.start, self.start)
        # Clamp the logits before the sigmoid: training drives the white background
        # to huge logits (up to ~1800), and int8 then has to span that whole range,
        # which rounds every colour to grey. Past +-6 the sigmoid is white/dark anyway.
        z = self.out(self.up(h))
        if self.int8_head:
            # same maths, in the form the Arm quantizer fuses into the conv (a clamp
            # starting at 0), so int8 spans only 0..12: bias was shifted by +6
            return torch.sigmoid(nn.functional.hardtanh(z, 0.0, 2 * LOGIT_CLAMP) - self.logit_shift)
        return torch.sigmoid(torch.clamp(z, -LOGIT_CLAMP, LOGIT_CLAMP))

    def prepare_for_int8(self) -> "BunnyDecoder":
        """Switch the head to the quantizer-friendly form (call once, after loading weights)."""
        if not self.int8_head:
            with torch.no_grad():
                self.out.bias += LOGIT_CLAMP
            self.int8_head = True
        return self


def scalar_ranges():
    """(lo, hi) of the 19 scalar dials, in NN-input order (from bunny_rig3.SCALARS)."""
    # Copied from model/bunny/bunny_rig3.py so the model loads without cairosvg.
    return [(-75, 15), (-15, 75), (-10, 90), (108, 148), (-7, 7), (-6, 6), (-30, 160), (-160, 30),
            (-20, 6), (-20, 6), (0.82, 1.14), (-14, 14), (0, 1), (-16, 16), (0.4, 1.0), (0, 1),
            (-0.07, 0.07), (0.8, 1.15), (0.92, 1.08)]


def normalize_dials(raw: torch.Tensor) -> torch.Tensor:
    """Raw dial vectors [N,30] -> NN input in [0,1]."""
    out = raw.clone()
    for i, (lo, hi) in enumerate(scalar_ranges()):
        out[:, 11 + i] = (raw[:, 11 + i] - lo) / (hi - lo)
    return out.clamp(0.0, 1.0)
