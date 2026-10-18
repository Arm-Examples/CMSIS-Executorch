# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "huggingface_hub", "numpy"]
# ///
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""DCSM-Simplified, stage 2 (student): distil the teacher into a MobileNetV2 for the Ethos-U85.

Waits for the teacher's crops (dataset repo, meta.json last), then fine-tunes an ImageNet
MobileNetV2 on them with knowledge distillation (soft teacher logits + hard labels) and an
augmentation that imitates the E8 board camera (low resolution, blur, colour cast, washed-out
contrast, noise, screen moire and glare). The model takes the camera's layout directly
(NHWC float RGB in [0,1]) and ends in a fixed average pool + reshape so the whole graph lowers
to the Ethos-U85 (see dcops/student_model.py, which must stay identical to StudentNHWC here).

  hf jobs uv run --flavor a10g-small --timeout 4h --secrets HF_TOKEN dcops/jobs/student.py \
      --name r160_kd --res 160 --kd-alpha 0.7
"""
import argparse, json, math, os, random, time

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class StudentNHWC(nn.Module):
    """MobileNetV2, NHWC [0,1] input, ImageNet normalisation folded in, NPU-friendly head."""

    def __init__(self, num_classes: int, res: int, pretrained: bool = True):
        super().__init__()
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = base.features
        self.pool = nn.AvgPool2d(math.ceil(res / 32))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):
        x = (x.permute(0, 3, 1, 2) - self.mean) / self.std
        x = self.pool(self.features(x))
        return self.fc(self.dropout(x.reshape(x.shape[0], 1280)))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def board_camera_aug(x):
    """x: float NCHW [0,1] on GPU. Imitates the DevKit-E8 MT9M114 frames seen at the venue."""
    b, _, h, w = x.shape
    dev = x.device
    # resolution loss: downscale then upscale
    if random.random() < 0.7:
        s = random.uniform(0.35, 0.9)
        x = F.interpolate(F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False),
                          size=(h, w), mode="bilinear", align_corners=False)
    # blur
    if random.random() < 0.4:
        k = random.choice([3, 5])
        x = F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="replicate"), k, 1)
    # colour cast (the board image is bluish/washed out), gamma, contrast
    gain = 1 + 0.35 * (torch.rand(b, 3, 1, 1, device=dev) - 0.5)
    gain[:, 2] += 0.15 * torch.rand(b, 1, 1, device=dev)
    x = x * gain
    gamma = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.7, 1.4)
    x = x.clamp(1e-4, 1) ** gamma
    c = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.6, 1.1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * c + m + torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.1, 0.15)
    # screen moire (demo images are often shown on a laptop or phone screen)
    if random.random() < 0.3:
        yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
        f, th = random.uniform(0.2, 1.2), random.uniform(0, math.pi)
        pat = torch.sin(f * (xx * math.cos(th) + yy * math.sin(th)))
        x = x + 0.06 * pat
    # glare
    if random.random() < 0.3:
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=dev), torch.linspace(0, 1, w, device=dev), indexing="ij")
        cx, cy = random.random(), random.random()
        x = x + 0.5 * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / random.uniform(0.01, 0.08))
    # sensor noise
    x = x + 0.03 * torch.rand(1, device=dev) * torch.randn_like(x)
    return x.clamp(0, 1)


def geo_aug(x, res):
    """Random crop/zoom (objects at different distances), flip, small rotation; out: res x res."""
    b, _, h, w = x.shape
    sc = random.uniform(0.55, 1.0)
    c = int(h * sc); i, j = random.randint(0, h - c), random.randint(0, w - c)
    x = x[..., i:i + c, j:j + c]
    if random.random() < 0.5: x = x.flip(3)
    ang = random.uniform(-25, 25) * math.pi / 180
    theta = torch.tensor([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0]],
                         device=x.device, dtype=x.dtype).unsqueeze(0).repeat(b, 1, 1)
    grid = F.affine_grid(theta, (b, 3, res, res), align_corners=False)
    return F.grid_sample(F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False), grid,
                         padding_mode="reflection", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--res", type=int, default=160)
    ap.add_argument("--kd-alpha", type=float, default=0.7, help="0 = no distillation")
    ap.add_argument("--temp", type=float, default=4.0)
    ap.add_argument("--domain-aug", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--repo-in", default="abhijitbetigeri/dcops-e8-crops")
    ap.add_argument("--repo-out", default="abhijitbetigeri/dcops-e8-student")
    ap.add_argument("--crops", default=None, help="local crops.npz instead of the Hub (smoke tests)")
    ap.add_argument("--wait-min", type=int, default=120)
    ap.add_argument("--meta", default=None, help="meta json next to --crops (default meta.json)")
    ap.add_argument("--mixup", type=float, default=0.0, help="probability of mixup per batch (wave 2)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights, e.g. 0.999 (wave 2)")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log("run", a.name, vars(a), "device", dev)

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    if a.crops:
        crops_path = a.crops
        meta = json.load(open(a.meta or os.path.join(os.path.dirname(a.crops), "meta.json")))
    else:
        t0 = time.time()
        while True:  # the teacher job writes meta.json last
            try:
                meta_path = hf_hub_download(a.repo_in, "meta.json", repo_type="dataset", force_download=True)
                break
            except Exception:  # noqa: BLE001
                if time.time() - t0 > a.wait_min * 60: raise SystemExit("teacher crops never appeared")
                log("waiting for teacher crops..."); time.sleep(60)
        meta = json.load(open(meta_path))
        crops_path = hf_hub_download(a.repo_in, "crops.npz", repo_type="dataset")
    d = np.load(crops_path)
    X, Y, S, T = d["images"], d["labels"], d["split"], d["teacher_logits"]
    KIND = d["kind"] if "kind" in d.files else np.array(["auto"] * len(Y))
    classes = meta["classes"]; nc = len(classes)
    tr, va = np.where(S == "train")[0], np.where(S == "val")[0]
    hv = KIND[va] == "human"  # human-labelled validation crops (wave 2): the trustworthy metric
    log(f"{len(Y)} crops ({len(tr)} train / {len(va)} val, {int(hv.sum())} human-labelled), {nc} classes; "
        f"teacher val acc {meta['teacher_val_acc']:.3f}")

    counts = np.bincount(Y[tr], minlength=nc).astype(float)
    p = 1.0 / np.maximum(counts[Y[tr]], 1); p /= p.sum()  # class-balanced sampling
    Xt, Tt = torch.from_numpy(X), torch.from_numpy(T)
    model = StudentNHWC(nc, a.res).to(dev)
    ema = None
    if a.ema > 0:
        import copy
        ema = copy.deepcopy(model).eval()
        for q in ema.parameters(): q.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.02)
    spe = max(1, len(tr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * spe, pct_start=0.1)

    def prep(idx, train):
        x = Xt[idx].to(dev).permute(0, 3, 1, 2).float() / 255
        if train:
            x = geo_aug(x, a.res)
            if a.domain_aug: x = board_camera_aug(x)
        else:
            x = F.interpolate(x, size=(a.res, a.res), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous()  # NHWC, as on the board

    def evaluate(aug=False):
        net = ema if ema is not None else model
        model.eval(); pred = []
        with torch.no_grad():
            for i in range(0, len(va), 128):
                idx = va[i:i + 128]
                x = prep(idx, False)
                if aug:  # the same frames as the board camera would deliver them
                    x = board_camera_aug(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
                pred.append(net(x).argmax(1).cpu().numpy())
        model.train()
        pred = np.concatenate(pred) if pred else np.array([])
        if len(va) == 0: return pred, 0.0
        if hv.any():  # wave 2: score on the human-labelled split only
            return pred, float((pred[hv] == Y[va][hv]).mean())
        return pred, float((pred == Y[va]).mean())

    rng = np.random.default_rng(0)
    best, best_state, hist = -1.0, None, []
    for ep in range(a.epochs):
        for _ in range(spe):
            idx = rng.choice(tr, a.batch, p=p)
            x = prep(idx, True)
            y = torch.from_numpy(Y[idx]).to(dev)
            t = Tt[idx].to(dev)
            lam, perm = 1.0, None
            if a.mixup > 0 and random.random() < a.mixup:  # mix images and both targets the same way
                lam = float(np.random.beta(0.4, 0.4)); perm = torch.randperm(x.shape[0], device=dev)
                x = lam * x + (1 - lam) * x[perm]
            logits = model(x)
            ce = F.cross_entropy(logits, y, label_smoothing=0.1)
            if perm is not None:
                ce = lam * ce + (1 - lam) * F.cross_entropy(logits, y[perm], label_smoothing=0.1)
            loss = ce
            if a.kd_alpha > 0:
                pt = F.softmax(t / a.temp, 1)
                if perm is not None: pt = lam * pt + (1 - lam) * pt[perm]
                kd = F.kl_div(F.log_softmax(logits / a.temp, 1), pt, reduction="batchmean")
                loss = (1 - a.kd_alpha) * ce + a.kd_alpha * a.temp ** 2 * kd
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                        if pe.dtype.is_floating_point: pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
                        else: pe.copy_(pm)
        _, acc = evaluate()
        _, acc_cam = evaluate(aug=True)
        hist.append({"epoch": ep + 1, "loss": float(loss.item()), "val_acc": acc, "val_acc_board_sim": acc_cam})
        log(f"epoch {ep+1}/{a.epochs} loss {loss.item():.3f} val {acc:.3f} val(board-sim) {acc_cam:.3f}")
        score = 0.5 * acc + 0.5 * acc_cam
        if score > best:
            src = ema if ema is not None else model
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}

    model.load_state_dict(best_state)
    if ema is not None: ema.load_state_dict(best_state)
    pred, acc = evaluate(); _, acc_cam = evaluate(aug=True)
    conf = np.zeros((nc, nc), int)
    sel = hv if hv.any() else np.ones(len(va), bool)
    for t_, p_ in zip(Y[va][sel], pred[sel]): conf[t_, p_] += 1
    per_class = {classes[c]: (float(conf[c, c] / conf[c].sum()) if conf[c].sum() else None) for c in range(nc)}
    log(f"best: val acc {acc:.3f}, board-sim {acc_cam:.3f}")

    out = f"/tmp/run_{a.name}"; os.makedirs(out, exist_ok=True)
    torch.save(best_state, f"{out}/student.pt")
    calib_idx = np.random.default_rng(1).choice(va if len(va) >= 64 else np.arange(len(Y)), 64, replace=False)
    calib = (prep(calib_idx, False).cpu().numpy() * 255).round().astype(np.uint8)
    np.savez_compressed(f"{out}/calib.npz", images=calib, labels=Y[calib_idx])
    open(f"{out}/labels.txt", "w").write("\n".join(classes) + "\n")
    metrics = {"name": a.name, "args": vars(a), "val_acc": acc, "val_acc_board_sim": acc_cam,
               "per_class_acc": per_class, "confusion": conf.tolist(), "history": hist,
               "teacher_val_acc": meta.get("teacher_human_val_acc", meta["teacher_val_acc"]),
               "metric": "human-labelled val" if hv.any() else "val (auto labels)", "n_train": int(len(tr)), "n_val": int(len(va))}
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=2)
    if a.no_push:
        log("no push; outputs in", out); return
    api.create_repo(a.repo_out, private=True, exist_ok=True)
    api.upload_folder(folder_path=out, path_in_repo=f"vision/{a.name}", repo_id=a.repo_out)
    log("pushed", f"{a.repo_out}/vision/{a.name}")


if __name__ == "__main__":
    main()
