"""Train the BunnyDecoder (30 dials -> RGB bunny on white) on model/bunny/bunny_train_96.npz.

Run with the project venv:
    .venv/Scripts/python.exe model/train_bunny.py
Settings (environment): BUNNY_SIZE (64), BUNNY_WIDTH (0.5), TRAIN_MINUTES (17), BATCH (64).
Training runs on a clock: the learning rate follows a cosine over TRAIN_MINUTES and
training stops when the time is up.
Writes model/bunny/bunny_decoder_<size>_w<width>.pt (after every epoch) and grids in
model/bunny/runs/: val_grid.png (target row over prediction row, 24 val frames) and
blend_grid.png (in-between dials: each row blends two val dial vectors).
Keeps Windows awake while it runs.
"""

import ctypes
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bunny_decoder import BunnyDecoder, normalize_dials  # noqa: E402

SIZE = int(os.environ.get("BUNNY_SIZE", "64"))
WIDTH = float(os.environ.get("BUNNY_WIDTH", "0.5"))
MINUTES = float(os.environ.get("TRAIN_MINUTES", "17"))
BATCH = int(os.environ.get("BATCH", "64"))
OUT = HERE / "bunny" / f"bunny_decoder_{SIZE}_w{WIDTH:g}.pt"
RUNS = HERE / "bunny" / "runs"


def keep_awake() -> None:
    if os.name == "nt":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no sleep while training
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def load():
    with np.load(HERE / "bunny" / "bunny_train_96.npz", allow_pickle=False) as z:
        images, dials, val = z["images"], z["dials"], z["val_mask"]
    imgs = np.empty((len(images), 3, SIZE, SIZE), np.uint8)
    for i in range(0, len(images), 2000):  # downsample in chunks (area filter)
        x = torch.from_numpy(images[i : i + 2000]).permute(0, 3, 1, 2).float()
        if SIZE != x.shape[-1]:
            x = F.interpolate(x, size=(SIZE, SIZE), mode="area")
        imgs[i : i + 2000] = x.round().clamp(0, 255).byte().numpy()
    return torch.from_numpy(imgs), normalize_dials(torch.from_numpy(dials)), torch.from_numpy(val)


def loss_fn(pred, target):
    # Extra weight on dark (outline) pixels so the line stays crisp
    dark = (target.min(dim=1, keepdim=True).values < 0.45).float()
    w = 1.0 + 4.0 * dark
    return (w * (pred - target).abs()).mean() + 0.2 * (w * (pred - target) ** 2).mean()


def grid(rows, path):
    from PIL import Image

    r = torch.stack(rows)  # R,N,3,H,W
    R, N = r.shape[:2]
    g = r.permute(0, 3, 1, 4, 2).reshape(R * SIZE, N * SIZE, 3)
    Image.fromarray((g.clamp(0, 1).numpy() * 255).astype("uint8")).resize((N * SIZE * 2, R * SIZE * 2), Image.NEAREST).save(path)


def main() -> None:
    keep_awake()
    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    RUNS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    imgs, dials, val = load()
    tr_idx, va_idx = torch.nonzero(~val).squeeze(1), torch.nonzero(val).squeeze(1)
    print(f"data {SIZE}x{SIZE}: {len(tr_idx)} train, {len(va_idx)} val, loaded in {time.time()-t0:.0f}s", flush=True)

    model = BunnyDecoder(SIZE, WIDTH)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    t_start, limit = time.time(), MINUTES * 60.0

    def set_lr():  # short warm-up, then cosine over the time budget
        f = min(1.0, (time.time() - t_start) / limit)
        lr = 2e-3 * min(1.0, f / 0.03) * 0.5 * (1 + math.cos(math.pi * f)) + 2e-5
        for g in opt.param_groups:
            g["lr"] = lr
        return f

    def evaluate():
        model.eval()
        with torch.no_grad():
            err = 0.0
            for i in range(0, len(va_idx), 256):
                j = va_idx[i : i + 256]
                err += (model(dials[j]) - imgs[j].float() / 255).abs().sum().item()
        model.train()
        return err / (len(va_idx) * 3 * SIZE * SIZE)

    epoch, done = 0, False
    while not done:
        epoch += 1
        te = time.time()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        total, seen = 0.0, 0
        for i in range(0, len(perm), BATCH):
            if set_lr() >= 1.0:
                done = True
                break
            j = perm[i : i + BATCH]
            loss = loss_fn(model(dials[j]), imgs[j].float() / 255)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
            seen += len(j)
        torch.save(model.state_dict(), OUT)
        print(f"epoch {epoch} ({seen} samples) train {total/max(1, seen):.4f} val L1 {evaluate():.4f} "
              f"({time.time()-te:.0f}s, total {(time.time()-t0)/60:.1f} min)", flush=True)

    model.eval()
    with torch.no_grad():
        j = va_idx[:24]
        grid([imgs[j].float() / 255, model(dials[j])], RUNS / "val_grid.png")
        a, b = dials[va_idx[24:32]], dials[va_idx[32:40]]
        rows = [model(a + (b - a) * t) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        grid([torch.stack([r[k] for r in rows]) for k in range(8)], RUNS / "blend_grid.png")
    print(f"done in {(time.time()-t0)/60:.1f} min, wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
