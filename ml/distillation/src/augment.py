# SPDX-License-Identifier: Apache-2.0
"""On-the-fly training augmentation shared by the v3 teachers and the v3 student.

``Augmenter(cfg_augment, ...)(x, groups, gen)`` takes a batch of scaled model inputs
``(B, 1, C, T)`` (``T`` = ``train_window`` for training trials, or already ``window`` for the
unlabelled 2a calibration trials) and returns ``(B, 1, C, window)``. Steps, in order:

1. random crop of ``window`` samples at offset ``crop_offset ± crop_jitter`` (trials that are
   already ``window`` long are passed through);
2. **EA-matrix mixing** (``ea_mix_p``): a trial of training subject-session ``s`` is re-aligned
   with the matrix of another training subject-session ``o``: ``A_o A_s^-1 (A_s X) = A_o X``.
   The pair matrices are precomputed from ``ea_train`` (``mixing_matrices``), so the
   pre-EA windows are not needed. Trials with ``group < 0`` (2a calibration) are not mixed;
3. amplitude jitter: every trial times ``U(amp_low, amp_high)``;
4. additive Gaussian noise ``noise_std`` (inputs are unit scale after ``input_scale``);
5. channel dropout: whole channels zeroed with probability ``channel_dropout`` (no rescaling).

``mixup(x, y, gen, alpha, p)`` mixes trials with a random partner from the same batch
(``lam ~ Beta(alpha, alpha)``, probability ``p`` per trial; only labelled trials, ``y >= 0``,
are mixed or used as partners). ``mixup_ce`` is the matching cross-entropy.

Every random draw comes from the CPU ``torch.Generator`` passed in, so a run is repeatable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

DEFAULTS = {"crop_jitter": 25, "noise_std": 0.05, "channel_dropout": 0.2, "amp_low": 0.8, "amp_high": 1.2,
            "ea_mix_p": 0.1, "mixup_alpha": 0.4, "mixup_p": 0.5}


def mixing_matrices(ea_train: dict) -> torch.Tensor:
    """``M[o, s] = A_o @ inv(A_s)`` for every ordered pair of training subject-sessions,
    float32 ``(G, G, C, C)``; ``list(ea_train)`` gives the group order."""
    A = np.stack([np.asarray(a, dtype=np.float64) for a in ea_train.values()])
    Ainv = np.linalg.inv(A)
    M = np.einsum("ocd,sde->osce", A, Ainv)
    return torch.from_numpy(M.astype(np.float32))


def crop(x: torch.Tensor, offsets: torch.Tensor, window: int) -> torch.Tensor:
    """Per-trial crop ``x[i, ..., offsets[i] : offsets[i] + window]`` of ``(B, 1, C, T)``."""
    B, one, C, _ = x.shape
    idx = offsets.view(B, 1, 1, 1).to(torch.long) + torch.arange(window).view(1, 1, 1, window)
    return torch.gather(x, 3, idx.expand(B, one, C, window))


class Augmenter:
    def __init__(self, cfg: dict | None, *, window: int, crop_offset: int, train_window: int,
                 mix: torch.Tensor | None = None) -> None:
        c = {**DEFAULTS, **(cfg or {})}
        self.window, self.crop_offset = int(window), int(crop_offset)
        max_off = int(train_window) - self.window
        if max_off < 0:
            raise ValueError(f"train_window {train_window} shorter than window {window}")
        j = c.get("crop_jitter")
        self.lo, self.hi = (0, max_off) if j is None else (max(0, self.crop_offset - int(j)),
                                                            min(max_off, self.crop_offset + int(j)))
        self.noise_std = float(c["noise_std"])
        self.channel_dropout = float(c["channel_dropout"])
        self.amp = (float(c["amp_low"]), float(c["amp_high"]))
        self.ea_mix_p = float(c["ea_mix_p"]) if mix is not None and len(mix) > 1 else 0.0
        self.mix = mix
        self.cfg = c

    def __call__(self, x: torch.Tensor, groups: torch.Tensor | None, gen: torch.Generator) -> torch.Tensor:
        B = x.shape[0]
        if x.shape[-1] != self.window:
            off = torch.randint(self.lo, self.hi + 1, (B,), generator=gen)
            x = crop(x, off, self.window)
        else:
            x = x.clone()
        if self.ea_mix_p > 0 and groups is not None:
            g = torch.as_tensor(groups, dtype=torch.long)
            pick = (torch.rand(B, generator=gen) < self.ea_mix_p) & (g >= 0)
            if pick.any():
                own = g[pick]
                G = self.mix.shape[0]
                other = torch.randint(0, G - 1, (len(own),), generator=gen)
                other = other + (other >= own).long()  # uniform over the other G - 1 groups
                M = self.mix[other, own]  # (k, C, C)
                x[pick] = torch.einsum("kcd,kodt->koct", M, x[pick])
        lo, hi = self.amp
        if hi > lo or lo != 1.0:
            x = x * (lo + (hi - lo) * torch.rand((B, 1, 1, 1), generator=gen, dtype=x.dtype))
        if self.noise_std > 0:
            x = x + self.noise_std * torch.randn(x.shape, generator=gen, dtype=x.dtype)
        if self.channel_dropout > 0:
            keep = torch.rand((B, 1, x.shape[2], 1), generator=gen) >= self.channel_dropout
            x = x * keep.to(x.dtype)
        return x


def mixup(x: torch.Tensor, y: torch.Tensor, gen: torch.Generator, alpha: float, p: float):
    """``(x_mix, y_a, y_b, lam)``: each labelled trial is mixed with probability ``p`` with a
    random labelled partner of the batch, ``lam ~ Beta(alpha, alpha)`` per trial;
    ``x_mix = lam x + (1 - lam) x_partner``. Unmixed trials have ``lam = 1`` and ``y_b = y_a``."""
    B = x.shape[0]
    lam = torch.ones(B, dtype=x.dtype)
    y_b = y.clone()
    lab = torch.nonzero(y >= 0).flatten()
    if p <= 0 or alpha <= 0 or len(lab) < 2:
        return x, y, y_b, lam
    partner = lab[torch.randint(0, len(lab), (len(lab),), generator=gen)]
    do = torch.rand(len(lab), generator=gen) < p
    # torch's Beta sampler takes no generator: draw lam from numpy, seeded from ``gen``.
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
    l = torch.from_numpy(np.random.default_rng(seed).beta(alpha, alpha, len(lab))).to(x.dtype)
    l = torch.where(do, l, torch.ones_like(l))
    lam[lab] = l
    y_b[lab] = y[partner]
    x_mix = x.clone()
    lv = l.view(-1, *([1] * (x.dim() - 1)))
    x_mix[lab] = lv * x[lab] + (1 - lv) * x[partner]
    return x_mix, y, y_b, lam


def mixup_ce(logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: torch.Tensor,
             label_smoothing: float = 0.0) -> torch.Tensor:
    """Per-trial ``lam CE(z, y_a) + (1 - lam) CE(z, y_b)``; 0 for unlabelled trials (``y_a < 0``)."""
    labelled = (y_a >= 0).to(logits.dtype)
    ca = F.cross_entropy(logits, y_a.clamp_min(0), reduction="none", label_smoothing=label_smoothing)
    cb = F.cross_entropy(logits, y_b.clamp_min(0), reduction="none", label_smoothing=label_smoothing)
    lam = lam.to(logits.device, logits.dtype)
    return (lam * ca + (1 - lam) * cb) * labelled


__all__ = ["Augmenter", "mixing_matrices", "crop", "mixup", "mixup_ce", "DEFAULTS"]
