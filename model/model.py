# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# The network modules re-implement pico-faces by Tim B. (cpldcpu), MIT license,
# see model/LICENSE-pico-faces.
"""The example model: pico-faces, a latent rectified-flow face generator, for Ethos-U85.

pico-faces (https://github.com/cpldcpu/pico-faces) generates 128x128 RGB faces
with a small diffusion transformer (DiT) on an 8x16x16 latent and a
convolutional VAE decoder. Upstream runs it through a hand-written int8 engine
on an RP2350; here the float checkpoints are re-expressed as ExecuTorch methods
and delegated to the Ethos-U85:

    dit_step(z (1,8,16,16), c (1,128)) -> v (1,8,16,16)   one velocity evaluation
    decode(z (1,8,16,16))              -> img (1,3,128,128) in [-1, 1]

The conditioning vector c = t_mlp(timestep_embedding(t)) + y_emb(y) depends only
on the schedule step and the class, so it is evaluated at export into a table
(pf_cond in model_params.h) and the firmware feeds it as an input; the Euler
loop and the classifier-free guidance blend run in C on the Cortex-M.

Contract with create_ai_layer.py:
    get_methods()   the methods of the exported program
    get_params()    constants for ai_layer/model_params.h

Environment knobs: PICO_FACES_VARIANT (m3_long_cfg | m3_decD_deep_full),
PICO_FACES_QUANT (a16w8 | a8w8 for dit_step; a8w8 shows visible artefacts),
PICO_FACES_DECODE_QUANT (a8w8 | a16w8 for decode), PICO_FACES_CALIB_SEEDS
(calibration trajectories, default 16), PICO_FACES_DIR (a local pico-faces
clone instead of the checkpoints setup_venv.py downloads into model/pico_faces/).
"""

from __future__ import annotations

import functools
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent

VARIANT = os.environ.get("PICO_FACES_VARIANT", "m3_long_cfg")
# The DiT's residual stream needs more than 8 bits; the decoder's convolutions
# do not, and run about twice as fast with 8-bit activations.
QUANTIZATION = os.environ.get("PICO_FACES_QUANT", "a16w8")
DECODE_QUANTIZATION = os.environ.get("PICO_FACES_DECODE_QUANT", "a8w8")
CALIB_SEEDS = int(os.environ.get("PICO_FACES_CALIB_SEEDS", "16"))

# The sampling schedule of the exported models (models/<variant>/export.yaml
# upstream): K_MAX Euler steps from t=1 (noise) towards t=0; k_steps in
# {8, 4, 2, 1} strides through it.
SCHEDULE = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
CFG_W = [4.0, 6.0, 8.0]  # guidance strengths upstream bakes tables for
N_CLASSES = 4  # 0 f/neutral, 1 f/smile, 2 m/neutral, 3 m/smile
NULL_CLASS = N_CLASSES  # unconditional (also the negative of the CFG blend)
EPS = torch.finfo(torch.float32).eps  # nn.RMSNorm(eps=None) upstream


# ----------------------------------------------------------------------------
# Checkpoints


@dataclass(frozen=True)
class Checkpoint:
    dit: dict[str, torch.Tensor]
    dit_cfg: dict
    dec: dict[str, torch.Tensor]
    dec_plan: list[tuple[int, int]]
    img_ch: int
    latent_mean: torch.Tensor
    latent_std: torch.Tensor


def checkpoint_dir() -> Path:
    if clone := os.environ.get("PICO_FACES_DIR"):
        return Path(clone) / "checkpoints" / VARIANT
    return HERE / "pico_faces" / VARIANT


@functools.lru_cache(maxsize=None)
def load_checkpoint() -> Checkpoint:
    sys.path.insert(0, str(HERE.parent))
    from setup_venv import PICO_FACES_FILES, sha256  # stdlib only

    folder = checkpoint_dir()
    expected = PICO_FACES_FILES[VARIANT]
    for name in expected:
        if not (folder / name).is_file():
            sys.exit(
                f"{folder / name} is missing. Run ./setup_venv.sh (or `python3 "
                "setup_venv.py --download-only`) to download the pico-faces "
                "checkpoints, or point PICO_FACES_DIR at a pico-faces clone."
            )

    dit_ckpt = torch.load(folder / "dit_qat.pt", map_location="cpu", weights_only=True)
    dit = {k.replace("_orig_mod.", ""): v.float() for k, v in dit_ckpt["ema"].items()}
    dit_cfg = dit_ckpt["cfg"]
    if dit_cfg.get("arch") != "dit" or dit_cfg.get("act") != "relu2":
        sys.exit(f"unsupported pico-faces DiT config: arch={dit_cfg.get('arch')} act={dit_cfg.get('act')}")

    # vae_final.pt pickles a numpy scalar and needs weights_only=False: only
    # load it after the hash check, so an unexpected file is never unpickled.
    vae_file = folder / "vae_final.pt"
    if (actual := sha256(vae_file)) != expected["vae_final.pt"]:
        sys.exit(f"{vae_file}: SHA-256 {actual} does not match the pinned checkpoint")
    vae_ckpt = torch.load(vae_file, map_location="cpu", weights_only=False)
    dec = {k[len("decoder."):]: v.float() for k, v in vae_ckpt["model"].items() if k.startswith("decoder.")}
    vae_cfg = vae_ckpt["cfg"]

    stats = np.load(folder / "latent_stats.npz")
    return Checkpoint(
        dit=dit,
        dit_cfg=dit_cfg,
        dec=dec,
        dec_plan=[tuple(int(x) for x in p) for p in vae_cfg["dec_plan"]],
        img_ch=int(vae_cfg.get("img_ch", 1)),
        latent_mean=torch.from_numpy(stats["mean"].astype(np.float32)),
        latent_std=torch.from_numpy(stats["std"].astype(np.float32)),
    )


# ----------------------------------------------------------------------------
# Building blocks, written with the operators the Ethos-U delegate supports


class RMSNorm(nn.Module):
    """RMS normalisation over the last dimension (no mean subtraction).

    aten.rms_norm has no Ethos-U lowering, so it is spelled out in primitives:
    mul, a mean, add, rsqrt (a table on the NPU) and mul. The mean is a matrix
    product with a constant 1/dim vector rather than aten.mean: aten.mean lowers
    to a TOSA REDUCE_SUM, for which the ExecuTorch Arm backend documents a
    16-bit Ethos-U85 issue (silent zeros in its softmax tests); the product is
    a MATMUL with an int48 accumulator that is exact at both precisions.
    """

    def __init__(self, dim: int, weight: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) if weight else None
        self.register_buffer("mean_weight", torch.full((1, dim), 1.0 / dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = F.linear(x * x, self.mean_weight)  # (..., 1): mean of the squares
        y = x * torch.rsqrt(ms + EPS)
        return y * self.weight if self.weight is not None else y


class Attention(nn.Module):
    """Multi-head self-attention over the 64 tokens on rank-3 tensors.

    Vela accepts non-elementwise operators up to rank 4 with a batch of 1, so
    the heads become the batch dimension of rank-3 bmm operands. The softmax
    scale 1/sqrt(hd) is folded into q_norm.weight at load time. The softmax is
    written out with its row sums as a matrix product with a ones vector (see
    RMSNorm for why aten.mean / REDUCE_SUM is avoided). The max subtraction (a
    REDUCE_MAX and a SUB, about 10% of the NPU cycles) is not needed in float,
    because the qk-RMSNorm bounds the scores, but without it rows whose maximum
    lies far below the calibrated range of the exp table lose all precision and
    the images fall apart. `model/verify_export.py --stage tosa` checks the
    lowered graphs against the TOSA reference model.
    """

    def __init__(self, dim: int, heads: int, tokens: int) -> None:
        super().__init__()
        self.dim, self.heads, self.hd, self.tokens = dim, heads, dim // heads, tokens
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RMSNorm(self.hd, weight=True)
        self.k_norm = RMSNorm(self.hd, weight=True)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("ones", torch.ones(heads, tokens, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (1, N, dim)
        n, h, hd = self.tokens, self.heads, self.hd
        q, k, v = self.qkv(x).split(self.dim, dim=-1)
        q = self.q_norm(q.reshape(n, h, hd)).permute(1, 0, 2)  # (H, N, hd)
        k = self.k_norm(k.reshape(n, h, hd)).permute(1, 2, 0)  # (H, hd, N)
        v = v.reshape(n, h, hd).permute(1, 0, 2)  # (H, N, hd)
        s = torch.bmm(q, k)  # (H, N, N)
        e = torch.exp(s - s.amax(-1, keepdim=True))
        att = e * torch.reciprocal(torch.bmm(e, self.ones))  # softmax rows
        out = torch.bmm(att, v).permute(1, 0, 2).reshape(1, n, self.dim)
        return self.proj(out)


class DiTBlock(nn.Module):
    """One adaLN-zero transformer block.

    The upstream `mod` (SiLU + Linear(dim, 6*dim), then chunk) is split into
    six Linear(dim, dim) so every modulation vector gets its own activation
    scale when quantized; the "1 + s" of the scale vectors is folded into the
    biases of the two scale Linears. The relu2 activation is relu followed by
    a multiplication (not pow, which would become a table).
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: int, tokens: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, weight=False)
        self.attn = Attention(dim, heads, tokens)
        self.norm2 = RMSNorm(dim, weight=False)
        self.fc1 = nn.Linear(dim, mlp_ratio * dim)
        self.fc2 = nn.Linear(mlp_ratio * dim, dim)
        self.mods = nn.ModuleList([nn.Linear(dim, dim) for _ in range(6)])  # s1 b1 g1 s2 b2 g2

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:  # a = silu(c), (1, dim)
        s1, b1, g1, s2, b2, g2 = (m(a).reshape(1, 1, -1) for m in self.mods)
        x = x + g1 * self.attn(self.norm1(x) * s1 + b1)
        h = F.relu(self.fc1(self.norm2(x) * s2 + b2))
        h = h * h
        return x + g2 * self.fc2(h)


class DiTStep(nn.Module):
    """dit_step(z, c) -> v: the DiT velocity for one Euler step.

    z: normalised latent (1, z_ch, z_hw, z_hw); c: conditioning vector (1, dim).
    Patchify is a strided Conv2d with the upstream Linear weights re-laid (the
    (c, py, px) feature order of upstream's patchify is exactly the flattened
    conv kernel layout). Unpatchify keeps the upstream output Linear (with its
    bias) and follows it with a strided ConvTranspose2d whose fixed 0/1 weights
    are a depth-to-space shuffle; an extra bias add after the transposed conv
    would fuse into a RESCALE whose shift is too small for the 16-bit path.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d = cfg["dit"]
        self.dim, self.depth, self.heads = int(d["dim"]), int(d["depth"]), int(d["heads"])
        self.patch, self.z_ch, self.z_hw = int(d["patch"]), int(cfg.get("latent_ch", 4)), int(cfg.get("latent_hw", 16))
        self.grid = self.z_hw // self.patch
        self.tokens = self.grid * self.grid
        pdim = self.z_ch * self.patch * self.patch

        self.embed = nn.Conv2d(self.z_ch, self.dim, self.patch, stride=self.patch)
        self.register_buffer("pos", torch.zeros(1, self.tokens, self.dim))
        self.blocks = nn.ModuleList(
            [DiTBlock(self.dim, self.heads, int(d["mlp_ratio"]), self.tokens) for _ in range(self.depth)]
        )
        self.final_norm = RMSNorm(self.dim, weight=False)
        self.final_mods = nn.ModuleList([nn.Linear(self.dim, self.dim) for _ in range(2)])  # s b
        self.final = nn.Linear(self.dim, pdim)
        self.unshuffle = nn.ConvTranspose2d(pdim, self.z_ch, self.patch, stride=self.patch, bias=False)
        shuffle = torch.zeros(pdim, self.z_ch, self.patch, self.patch)
        for c in range(self.z_ch):
            for py in range(self.patch):
                for px in range(self.patch):
                    shuffle[c * self.patch * self.patch + py * self.patch + px, c, py, px] = 1.0
        with torch.no_grad():
            self.unshuffle.weight.copy_(shuffle)
        self.unshuffle.weight.requires_grad_(False)
        self._pdim = pdim

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.embed(z).reshape(1, self.dim, self.tokens).permute(0, 2, 1) + self.pos
        a = F.silu(c)
        for blk in self.blocks:
            x = blk(x, a)
        s, b = (m(a).reshape(1, 1, -1) for m in self.final_mods)
        x = self.final(self.final_norm(x) * s + b)  # (1, tokens, pdim)
        x = x.permute(0, 2, 1).reshape(1, self._pdim, self.grid, self.grid)
        return self.unshuffle(x)

    @torch.no_grad()
    def load_state(self, sd: dict[str, torch.Tensor]) -> "DiTStep":
        dim, p, C = self.dim, self.patch, self.z_ch
        self.embed.weight.copy_(sd["embed.weight"].reshape(dim, C, p, p))
        self.embed.bias.copy_(sd["embed.bias"])
        self.pos.copy_(sd["pos"])
        for i, blk in enumerate(self.blocks):
            pre = f"blocks.{i}."
            blk.attn.qkv.weight.copy_(sd[pre + "attn.qkv.weight"])
            blk.attn.qkv.bias.copy_(sd[pre + "attn.qkv.bias"])
            blk.attn.q_norm.weight.copy_(sd[pre + "attn.q_norm.weight"] * self.blocks[0].attn.hd ** -0.5)
            blk.attn.k_norm.weight.copy_(sd[pre + "attn.k_norm.weight"])
            blk.attn.proj.weight.copy_(sd[pre + "attn.proj.weight"])
            blk.attn.proj.bias.copy_(sd[pre + "attn.proj.bias"])
            blk.fc1.weight.copy_(sd[pre + "mlp.0.weight"])
            blk.fc1.bias.copy_(sd[pre + "mlp.0.bias"])
            blk.fc2.weight.copy_(sd[pre + "mlp.2.weight"])
            blk.fc2.bias.copy_(sd[pre + "mlp.2.bias"])
            w, b = sd[pre + "mod.1.weight"], sd[pre + "mod.1.bias"]
            for j, m in enumerate(blk.mods):
                m.weight.copy_(w[j * dim:(j + 1) * dim])
                m.bias.copy_(b[j * dim:(j + 1) * dim])
            blk.mods[0].bias.add_(1.0)  # 1 + s1
            blk.mods[3].bias.add_(1.0)  # 1 + s2
        w, b = sd["final_mod.1.weight"], sd["final_mod.1.bias"]
        for j, m in enumerate(self.final_mods):
            m.weight.copy_(w[j * dim:(j + 1) * dim])
            m.bias.copy_(b[j * dim:(j + 1) * dim])
        self.final_mods[0].bias.add_(1.0)  # 1 + s
        self.final.weight.copy_(sd["final.weight"])
        self.final.bias.copy_(sd["final.bias"])
        return self

    @classmethod
    def from_checkpoint(cls) -> "DiTStep":
        ck = load_checkpoint()
        return cls(ck.dit_cfg).load_state(ck.dit).eval()


class Cond(nn.Module):
    """c = t_mlp(timestep_embedding(t)) + y_emb(y): float, evaluated on the host."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        dim, t_dim = int(cfg["dit"]["dim"]), int(cfg["t_embed_dim"])
        half = t_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs1000", freqs * 1000.0)  # t in [0, 1] scaled like DDPM
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.y_emb = nn.Embedding(int(cfg.get("n_classes", 0)) + 1, dim)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # t (B,) float, y (B,) int64
        args = t.float()[:, None] * self.freqs1000[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.t_mlp(emb) + self.y_emb(y)

    @torch.no_grad()
    def cond_table(self) -> torch.Tensor:
        """(n_classes + 1, K_MAX, dim): c for every class at every schedule point."""
        n = self.y_emb.num_embeddings
        t = torch.tensor(SCHEDULE, dtype=torch.float32)
        return torch.stack([self(t, torch.full((len(SCHEDULE),), y, dtype=torch.int64)) for y in range(n)])

    @torch.no_grad()
    def load_state(self, sd: dict[str, torch.Tensor]) -> "Cond":
        self.t_mlp[0].weight.copy_(sd["t_mlp.0.weight"])
        self.t_mlp[0].bias.copy_(sd["t_mlp.0.bias"])
        self.t_mlp[2].weight.copy_(sd["t_mlp.2.weight"])
        self.t_mlp[2].bias.copy_(sd["t_mlp.2.bias"])
        self.y_emb.weight.copy_(sd["y_emb.weight"])
        return self

    @classmethod
    def from_checkpoint(cls) -> "Cond":
        ck = load_checkpoint()
        return cls(ck.dit_cfg).load_state(ck.dit).eval()


class Decoder(nn.Module):
    """decode(z_norm) -> img in [-1, 1]: the VAE decoder.

    A chain of Conv3x3 (+ folded BatchNorm) + ReLU with nearest 2x upsampling
    before the layers the plan flags, and a linear output conv. The latent
    de-normalisation z = z_norm * std + mean is folded into the first conv.
    """

    def __init__(self, plan: list[tuple[int, int]], z_ch: int, img_ch: int) -> None:
        super().__init__()
        self.up_before: set[int] = set()
        body, c_in = [], z_ch
        for i, (c_out, up) in enumerate(plan):
            if up:
                self.up_before.add(i)
            body.append(nn.Conv2d(c_in, c_out, 3, padding=1))
            c_in = c_out
        self.body = nn.ModuleList(body)
        self.out = nn.Conv2d(c_in, img_ch, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for i, conv in enumerate(self.body):
            if i in self.up_before:
                h = F.interpolate(h, scale_factor=2.0, mode="nearest")
            h = F.relu(conv(h))
        return torch.clamp(self.out(h), -1.0, 1.0)

    @torch.no_grad()
    def load_state(self, sd: dict[str, torch.Tensor], mean: torch.Tensor, std: torch.Tensor) -> "Decoder":
        for i, conv in enumerate(self.body):
            w, b = sd[f"body.{i}.0.weight"], sd[f"body.{i}.0.bias"]
            gamma, beta = sd[f"body.{i}.1.weight"], sd[f"body.{i}.1.bias"]
            mu, var = sd[f"body.{i}.1.running_mean"], sd[f"body.{i}.1.running_var"]
            f = gamma / torch.sqrt(var + 1e-5)
            w = w * f[:, None, None, None]
            b = (b - mu) * f + beta
            if i == 0:  # z_real = z_norm * std + mean, folded into the first conv
                b = b + (w * mean[None, :, None, None]).sum(dim=(1, 2, 3))
                w = w * std[None, :, None, None]
            conv.weight.copy_(w)
            conv.bias.copy_(b)
        self.out.weight.copy_(sd["out.weight"])
        self.out.bias.copy_(sd["out.bias"])
        return self

    @classmethod
    def from_checkpoint(cls) -> "Decoder":
        ck = load_checkpoint()
        z_ch = int(ck.dit_cfg.get("latent_ch", 4))
        return cls(ck.dec_plan, z_ch, ck.img_ch).load_state(ck.dec, ck.latent_mean, ck.latent_std).eval()


# ----------------------------------------------------------------------------
# Sampling (host reference) and calibration data


def schedule_steps(k_steps: int) -> list[tuple[int, float]]:
    """[(k, dt)] for k_steps Euler steps striding the K_MAX schedule."""
    stride = len(SCHEDULE) // k_steps
    if stride * k_steps != len(SCHEDULE):
        raise ValueError(f"k_steps must divide {len(SCHEDULE)}")
    ks = list(range(0, len(SCHEDULE), stride))
    ts = [SCHEDULE[k] for k in ks] + [0.0]
    return [(k, t - t_next) for k, t, t_next in zip(ks, ts[:-1], ts[1:])]


@torch.no_grad()
def euler_sample(
    dit: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cond: torch.Tensor,
    z: torch.Tensor,
    y: int,
    w: float | None = None,
    k_steps: int = 4,
    tap: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Rectified-flow Euler sampling from noise z (1, C, H, W) to the final latent.

    cond: the (n_cond, K_MAX, dim) table of Cond.cond_table(). w: classifier-free
    guidance strength (None or 0 = plain). tap(z, c) sees every dit input pair.
    """
    guided = w is not None and w > 0 and y != NULL_CLASS
    for k, dt in schedule_steps(k_steps):
        c = cond[y, k][None]
        if tap:
            tap(z, c)
        v = dit(z, c)
        if guided:
            c_null = cond[NULL_CLASS, k][None]
            if tap:
                tap(z, c_null)
            v_null = dit(z, c_null)
            v = v_null + w * (v - v_null)
        z = z - dt * v
    return z


def noise(seed: int, z_ch: int, z_hw: int) -> torch.Tensor:
    """The firmware's starting noise for a seed: PCG32 (XSH-RR) and a sum of twelve
    12-bit uniforms per value (CLT-12), in CHW order, as src/app_main.cpp computes
    it, so the host reference and the board sample from the same latent. The
    generator is pico-faces' engine/src/prng.c; upstream fills in token order."""
    mask, mult, inc = (1 << 64) - 1, 6364136223846793005, 1442695040888963407
    state = inc
    state = ((state + seed) * mult + inc) & mask
    values = np.empty(z_ch * z_hw * z_hw, dtype=np.float32)
    for i in range(values.size):
        acc = 0
        for _ in range(12):
            x, count = state, state >> 59
            state = (x * mult + inc) & mask
            x ^= x >> 18
            out = (x >> 27) & 0xFFFFFFFF
            acc += (((out >> count) | (out << ((32 - count) & 31))) & 0xFFFFFFFF) >> 20
        values[i] = (acc - 24576) / 4096.0
    return torch.from_numpy(values).reshape(1, z_ch, z_hw, z_hw)


@torch.no_grad()
def to_image(img: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) in [-1, 1] -> uint8 HWC, the same mapping the firmware uses."""
    x = ((img.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return x[0].permute(1, 2, 0).contiguous().numpy()


@functools.lru_cache(maxsize=None)
def _trajectories() -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor]]]:
    """Calibration inputs from float sampling runs (as upstream quant/calibrate.py).

    CALIB_SEEDS trajectories at the full K_MAX schedule; classes cycle 0..3,
    every second run is guided with w=8 so the hotter guided activations are
    covered too (both passes of each guided step are tapped).
    """
    dit, cond = DiTStep.from_checkpoint(), Cond.from_checkpoint().cond_table()
    dit_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    dec_inputs: list[tuple[torch.Tensor]] = []
    for seed in range(CALIB_SEEDS):
        z = noise(777 + seed, dit.z_ch, dit.z_hw)
        y, w = seed % N_CLASSES, (max(CFG_W) if seed % 2 else None)
        z0 = euler_sample(dit, cond, z, y, w, k_steps=len(SCHEDULE), tap=lambda z, c: dit_inputs.append((z.clone(), c.clone())))
        dec_inputs.append((z0,))
    return dit_inputs, dec_inputs


def calib_dit_step() -> Iterable[tuple[torch.Tensor, ...]]:
    return _trajectories()[0]


def calib_decode() -> Iterable[tuple[torch.Tensor, ...]]:
    return _trajectories()[1]


# ----------------------------------------------------------------------------
# The contract with create_ai_layer.py


@dataclass(frozen=True)
class Method:
    """One method of the program; the fields create_ai_layer.Method has."""

    name: str
    module: nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    calibration: Callable[[], Iterable[tuple[torch.Tensor, ...]]] | None = None
    quantization: str = "a8w8"


def quantization(kind: str, variable: str) -> str:
    if kind not in ("a8w8", "a16w8"):
        sys.exit(f"{variable}={kind!r}: expected a8w8 or a16w8")
    return kind


def example_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """(z, c) for dit_step; z alone is decode's input."""
    ck = load_checkpoint()
    z_ch, z_hw = int(ck.dit_cfg.get("latent_ch", 4)), int(ck.dit_cfg.get("latent_hw", 16))
    return noise(0, z_ch, z_hw), Cond.from_checkpoint().cond_table()[NULL_CLASS, 0][None]


def get_methods() -> list[Method]:
    z, c = example_inputs()
    return [
        Method("dit_step", DiTStep.from_checkpoint(), (z, c), calib_dit_step,
               quantization(QUANTIZATION, "PICO_FACES_QUANT")),
        Method("decode", Decoder.from_checkpoint(), (z,), calib_decode,
               quantization(DECODE_QUANTIZATION, "PICO_FACES_DECODE_QUANT")),
    ]


def macs(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> int:
    """Multiply-accumulates of the convolutions and matrix products of one call."""
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter, torch.no_grad():
        module(*inputs)
    return counter.get_total_flops() // 2


def get_params() -> dict:
    """Constants the firmware needs; create_ai_layer.py writes them to model_params.h."""
    ck = load_checkpoint()
    dit, dec = DiTStep.from_checkpoint(), Decoder.from_checkpoint()
    cond = Cond.from_checkpoint().cond_table()  # (N_CLASSES + 1, K_MAX, dim)
    z, c = example_inputs()
    return {
        "PF_VARIANT": VARIANT,
        "PF_QUANT": f"{QUANTIZATION}/{DECODE_QUANTIZATION}",  # dit_step / decode
        "PF_LATENT_CH": dit.z_ch,
        "PF_LATENT_HW": dit.z_hw,
        "PF_IMG_CH": ck.img_ch,
        "PF_IMG_HW": dit.z_hw << len(dec.up_before),
        "PF_COND_DIM": cond.shape[2],
        "PF_N_CLASSES": N_CLASSES,
        "PF_NULL_CLASS": NULL_CLASS,  # unconditional; also the negative of the guidance blend
        "PF_N_COND": cond.shape[0],
        "PF_K_MAX": len(SCHEDULE),
        "PF_N_CFG_W": len(CFG_W),
        "PF_MACS_DIT_STEP": macs(dit, (z, c)),
        "PF_MACS_DECODE": macs(dec, (z,)),
        # t of every schedule step; a run of k steps uses every (PF_K_MAX / k)-th
        # entry and integrates from t to the next used entry (0 after the last).
        "pf_schedule": SCHEDULE,
        # Guidance strengths pico-faces' firmware cycles through (any w > 0 works here).
        "pf_cfg_w": CFG_W,
        # c(t_k, y) = t_mlp(timestep_embedding(t_k)) + y_emb(y), indexed [class][step].
        "pf_cond": cond,
    }
