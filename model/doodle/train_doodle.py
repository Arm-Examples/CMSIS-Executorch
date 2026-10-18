"""Train the doodle classifier on Quick, Draw! bitmaps (model/doodle/data, see get_data.py).

Run:  .venv/Scripts/python.exe model/doodle/train_doodle.py
Writes model/doodle/doodle_net.pt and prints validation accuracy + a confusion matrix.
Augmentation (shift, scale, rotation, stroke thickness) makes it cope with finger
drawings rasterised on the board.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from doodle_net import CLASSES, DoodleNet  # noqa: E402

EPOCHS = int(os.environ.get("EPOCHS", "6"))


def load():
    xs, ys = [], []
    for i, c in enumerate(CLASSES):
        a = np.load(HERE / "data" / f"{c.replace(' ', '_')}.npy")
        xs.append(a)
        ys.append(np.full(len(a), i))
    x = torch.from_numpy(np.concatenate(xs)).float().unsqueeze(1) / 255.0
    y = torch.from_numpy(np.concatenate(ys)).long()
    return x, y


def augment(x):
    n = x.shape[0]
    ang = torch.empty(n).uniform_(-0.2, 0.2)
    sc = torch.empty(n).uniform_(0.85, 1.15)
    tx, ty = torch.empty(n).uniform_(-0.12, 0.12), torch.empty(n).uniform_(-0.12, 0.12)
    theta = torch.stack([torch.stack([torch.cos(ang) / sc, -torch.sin(ang) / sc, tx], 1),
                         torch.stack([torch.sin(ang) / sc, torch.cos(ang) / sc, ty], 1)], 1)
    out = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    r = torch.rand(1).item()
    if r < 0.25:
        out = F.max_pool2d(out, 3, stride=1, padding=1)  # thicker strokes
    elif r < 0.4:
        out = -F.max_pool2d(-out, 2, stride=1, padding=0)  # thinner
        out = F.pad(out, (0, 1, 0, 1))
    return out.clamp(0, 1)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    x, y = load()
    perm = torch.randperm(len(x))
    n_val = len(x) // 20
    va, tr = perm[:n_val], perm[n_val:]
    model = DoodleNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    steps = EPOCHS * ((len(tr) + 127) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), 128):
            j = p[i:i + 128]
            loss = F.cross_entropy(model(augment(x[j])), y[j], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(x[va[i:i + 1024]]).argmax(1) for i in range(0, len(va), 1024)])
        acc = (pred == y[va]).float().mean().item()
        print(f"epoch {ep + 1}/{EPOCHS} val acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), HERE / "doodle_net.pt")
    cm = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    for t, p_ in zip(y[va], pred):
        cm[t, p_] += 1
    print("confusion (rows = true, cols = predicted):")
    print("            " + " ".join(f"{c[:5]:>5}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c[:11]:>11} " + " ".join(f"{v:5d}" for v in cm[i].tolist()))
    print("wrote", HERE / "doodle_net.pt")


if __name__ == "__main__":
    main()
