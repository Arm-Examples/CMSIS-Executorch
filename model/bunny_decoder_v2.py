"""Bunny decoder v2 (the cloud-trained model): 32 pose dials in, an RGB bunny out.

Same layers and parameter names as model/bunny/kit_v2/train_bunny.py's Decoder,
so its decoder_final.pt loads as-is (model/bunny/decoder_v2_<size>.pt). Dials
are normalised with model/bunny/dial_ranges_v2.json.

Adds the int8-friendly head used for the v1 decoder (see bunny_decoder.py):
sigmoid(clamp(logits, -6, 6)), exported as hardtanh(0, 12) - 6 with the last
bias shifted by +6, so the quantizer does not round every colour to grey.
"""

import torch
import torch.nn.functional as F
from torch import nn

LOGIT_CLAMP = 6.0


class DecoderV2(nn.Module):
    def __init__(self, ndial: int = 32, size: int = 128, width: int = 128) -> None:
        super().__init__()
        assert size % 16 == 0
        self.g = size // 16
        self.w = width
        self.fc1 = nn.Linear(ndial, 256)
        self.fc2 = nn.Linear(256, width * self.g * self.g)
        ch = [width, width, width // 2, width // 2, width // 4]
        self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, padding=1) for i in range(4)])
        self.out = nn.Conv2d(ch[-1], 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.int8_head = False
        self.register_buffer("logit_shift", torch.full((1, 3, 1, 1), LOGIT_CLAMP), persistent=False)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(d))
        x = F.relu(self.fc2(x)).reshape(-1, self.w, self.g, self.g)
        for c in self.convs:
            x = F.relu(c(self.up(x)))
        z = self.out(x)
        if self.int8_head:
            return torch.sigmoid(F.hardtanh(z, 0.0, 2 * LOGIT_CLAMP) - self.logit_shift)
        return torch.sigmoid(torch.clamp(z, -LOGIT_CLAMP, LOGIT_CLAMP))

    def prepare_for_int8(self) -> "DecoderV2":
        if not self.int8_head:
            with torch.no_grad():
                self.out.bias += LOGIT_CLAMP
            self.int8_head = True
        return self


def load_v2(path, size: int) -> DecoderV2:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "m" in sd:
        sd = sd["m"]
    width = sd["convs.0.weight"].shape[1]
    m = DecoderV2(sd["fc1.weight"].shape[1], size, width)
    m.load_state_dict(sd)
    return m.eval()
