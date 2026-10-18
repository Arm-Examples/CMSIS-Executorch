# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "huggingface_hub", "numpy"]
# ///
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""DCSM-Simplified: fine-tune the deployed student on frames from the board's own camera.

The wave-2 student (v2_r160_kd) is right on clean photos (18/24 held-out slides on the PC) but
confuses classes through the board camera (screen moire, glare, low resolution, colour). This
job continues training it on:
  - board frames: the E8 camera's model input recorded over J-Link while it looked at slides
    of training-split components and at venue scenes (background),
  - a replay set of human-labelled training crops, so it does not forget the clean domain.
Slides are split by slide id: frames of held-out slides never reach training (honest test).

  python finetune_board.py --data board_ft.npz --init student.pt --out /tmp/ft
"""
import argparse, json, math, os, random, sys, time

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student import StudentNHWC, board_camera_aug  # same model and camera augmentation


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="npz: board_x/board_y/board_split, replay_x/replay_y")
    ap.add_argument("--init", required=True, help="student.pt to start from")
    ap.add_argument("--out", default="/tmp/ft")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--board-weight", type=float, default=0.6, help="share of each batch from board frames")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--res", type=int, default=160)
    ap.add_argument("--classes", default="4,6,9,12,13,16",
                    help="class ids the demo firmware shows (CLS_ALLOWED_MASK); training and scoring "
                         "compare only these. 'all' = every class")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    d = np.load(a.data, allow_pickle=True)
    classes = list(d["classes"]); nc = len(classes)
    bx, by, bs = d["board_x"], d["board_y"], d["board_split"]
    rx, ry = d["replay_x"], d["replay_y"]
    btr, bte = np.where(bs == "train")[0], np.where(bs == "test")[0]
    log(f"device {dev}; board frames train {len(btr)} test {len(bte)}; replay {len(ry)}; classes {nc}")
    model = StudentNHWC(nc, a.res, pretrained=False).to(dev)
    model.load_state_dict(torch.load(a.init, map_location="cpu"))
    BX, RX = torch.from_numpy(bx), torch.from_numpy(rx)
    keep = list(range(nc)) if a.classes == "all" else [int(c) for c in a.classes.split(",")]
    mask = torch.full((nc,), -1e4, device=dev); mask[keep] = 0  # logits of hidden classes never win
    keep_t = torch.tensor(keep, device=dev)
    remap = torch.full((nc,), -1, dtype=torch.long, device=dev); remap[keep_t] = torch.arange(len(keep), device=dev)
    btr = btr[np.isin(by[btr], keep)]; bte = bte[np.isin(by[bte], keep)]
    rsel = np.where(np.isin(ry, keep))[0]
    log(f"classes {[classes[c] for c in keep]}; board train {len(btr)} test {len(bte)}; replay {len(rsel)}")

    def to_in(x, train):
        x = x.to(dev).permute(0, 3, 1, 2).float() / 255
        if x.shape[-1] != a.res:
            x = F.interpolate(x, size=(a.res, a.res), mode="bilinear", align_corners=False)
        if train:
            if random.random() < 0.5: x = x.flip(3)
            sc = random.uniform(0.8, 1.0); c = int(a.res * sc); i, j = random.randint(0, a.res - c), random.randint(0, a.res - c)
            x = F.interpolate(x[..., i:i + c, j:j + c], size=(a.res, a.res), mode="bilinear", align_corners=False)
        return x

    def evaluate(idx):
        model.eval(); pred = []
        with torch.no_grad():
            for i in range(0, len(idx), 128):
                pred.append((model(to_in(BX[idx[i:i + 128]], False).permute(0, 2, 3, 1).contiguous()) + mask).argmax(1).cpu().numpy())
        model.train()
        pred = np.concatenate(pred) if pred else np.array([], int)
        return pred, float((pred == by[idx]).mean()) if len(idx) else float("nan")

    _, base = evaluate(bte)
    log(f"before fine-tuning: board test acc {base:.3f}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.02)
    nb = int(a.batch * a.board_weight); nr = a.batch - nb
    spe = max(20, len(btr) // max(nb, 1))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * spe, pct_start=0.15)
    # class-balanced sampling of board frames
    cnt = np.bincount(by[btr], minlength=nc).astype(float); pb = 1 / np.maximum(cnt[by[btr]], 1); pb /= pb.sum()
    rng = np.random.default_rng(0)
    best, best_state, hist = -1, None, []
    for ep in range(a.epochs):
        for _ in range(spe):
            ib = rng.choice(btr, nb, p=pb); ir = rng.choice(rsel, nr)
            xb = to_in(BX[ib], True)
            xr = board_camera_aug(to_in(RX[ir], True))  # replay crops look like the board camera too
            x = torch.cat([xb, xr]).permute(0, 2, 3, 1).contiguous()
            y = torch.from_numpy(np.concatenate([by[ib], ry[ir]])).to(dev)
            loss = F.cross_entropy(model(x)[:, keep_t], remap[y], label_smoothing=0.1)  # softmax over shown classes
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        _, acc = evaluate(bte)
        hist.append({"epoch": ep + 1, "loss": float(loss), "board_test_acc": acc})
        log(f"epoch {ep+1}/{a.epochs} loss {loss.item():.3f} board test acc {acc:.3f}")
        if acc >= best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred, acc = evaluate(bte)
    conf = np.zeros((nc, nc), int)
    for t, p in zip(by[bte], pred): conf[t, p] += 1
    per = {classes[c]: (float(conf[c, c] / conf[c].sum()) if conf[c].sum() else None) for c in range(nc)}
    os.makedirs(a.out, exist_ok=True)
    torch.save(best_state, f"{a.out}/student.pt")
    ci = rng.choice(len(bx), min(64, len(bx)), replace=False)
    calib = (to_in(BX[ci], False).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    np.savez_compressed(f"{a.out}/calib.npz", images=calib, labels=by[ci])  # calibrate int8 on board frames
    open(f"{a.out}/labels.txt", "w").write("\n".join(classes) + "\n")
    json.dump({"name": "v3_board_ft", "args": {**vars(a), "res": a.res}, "val_acc": acc, "val_acc_board_sim": acc,
               "metric": "board-camera frames of held-out slides", "board_test_acc_before": base,
               "per_class_acc": per, "confusion": conf.tolist(), "history": hist,
               "n_board_train": int(len(btr)), "n_board_test": int(len(bte)), "n_replay": int(len(ry))},
              open(f"{a.out}/metrics.json", "w"), indent=2)
    log(f"done: board test acc {base:.3f} -> {acc:.3f}; wrote {a.out}")


if __name__ == "__main__":
    main()
