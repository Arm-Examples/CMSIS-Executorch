# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "timm", "huggingface_hub", "numpy", "pillow"]
# ///
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""DCSM-Simplified, wave 2 teacher: human labels + out-of-fold soft labels.

Wave 1 learned from YOLO pseudo-labels only (teacher 69 %). Wave 2 adds the DC-Ops merged
dataset: 2,056 human-labelled images (32.5k boxes, Roboflow CC BY 4.0, mapped to the 16 DC-Ops
classes), keeps the wave-1 crops (319 auto-labelled images) for the classes the human set lacks,
and adds background crops.

Teacher: two ConvNeXt-Base (ImageNet-22k) models trained on disjoint halves of the training
images; each labels the half it did not see, so every training crop gets an *out-of-fold* soft
label (no memorised targets). Validation crops get the average of both. The primary metric is
accuracy on the human-labelled validation split, which the wave-1 run did not have.

Writes crops_v2.npz + meta_v2.json to --out (local disk / Drive); pushes meta + teacher weights
(not the multi-GB crops) to the Hub.
"""
import argparse, hashlib, json, os, random, time, zipfile

import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

CLASSES = ["server rack", "compute tray", "NVLink switch tray", "network switch", "power shelf",
           "cable", "network port", "LED indicator", "label", "fan", "cooling manifold",
           "cable cartridge", "power connector", "drive bay", "management port", "DPU", "background"]
BG = 16
CROP = 192
CAP = 3000          # per-class cap from the human set (network port alone has 14k boxes)
BG_PER_IMAGE = 1


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def square_crop(img, box, context, size=CROP):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    s = max(max(box[2] - box[0], box[3] - box[1]) * context, 16)
    return img.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((size, size), Image.BILINEAR)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def human_crops(root, rng):
    """YOLO polygon/box labels of the merged dataset -> balanced crops."""
    boxes = []  # (split, img_path, box_px, cls)
    for split, sub in (("train", "train"), ("val", "val")):
        for lab in sorted(os.listdir(f"{root}/{sub}/labels")):
            stem = os.path.splitext(lab)[0]
            img_path = next((f"{root}/{sub}/images/{stem}{ext}" for ext in (".jpg", ".jpeg", ".png", ".webp")
                             if os.path.exists(f"{root}/{sub}/images/{stem}{ext}")), None)
            if img_path is None: continue
            for line in open(f"{root}/{sub}/labels/{lab}"):
                p = line.split()
                if len(p) < 5: continue
                xy = np.array(p[1:], dtype=float)
                if len(xy) == 4:  # cx cy w h
                    cx, cy, w, h = xy; xy = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                    boxes.append((split, img_path, xy, int(p[0])))
                else:
                    pts = xy.reshape(-1, 2)
                    boxes.append((split, img_path, np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]), int(p[0])))
    # cap per class on the training split only (validation is kept whole)
    by_cls = {}
    for b in boxes:
        if b[0] == "train": by_cls.setdefault(b[3], []).append(b)
    keep = [b for b in boxes if b[0] == "val"]
    for c, lst in by_cls.items():
        rng.shuffle(lst); keep += lst[:CAP]
    log("human boxes", len(boxes), "-> kept", len(keep))
    keep.sort(key=lambda b: b[1])
    X, Y, S, SRC = [], [], [], []
    cur, img, per_img = None, None, {}
    for k, (split, path, b, c) in enumerate(keep):
        if path != cur:
            cur = path
            try: img = Image.open(path).convert("RGB")
            except Exception: img = None
        if img is None: continue
        w, h = img.size
        bp = b * [w, h, w, h]
        if (bp[2] - bp[0]) < 12 or (bp[3] - bp[1]) < 12: continue
        X.append(np.asarray(square_crop(img, bp, rng.uniform(1.15, 1.5)))); Y.append(c); S.append(split)
        SRC.append(os.path.basename(path)); per_img.setdefault(path, []).append(bp)
        if k % 5000 == 0: log(f"human crops {k}/{len(keep)}")
    # background crops from the same images
    for path, bl in per_img.items():
        try: img = Image.open(path).convert("RGB")
        except Exception: continue
        w, h = img.size
        for _ in range(30):
            s = rng.uniform(0.12, 0.4) * min(w, h); x0, y0 = rng.uniform(0, w - s), rng.uniform(0, h - s)
            cand = (x0, y0, x0 + s, y0 + s)
            if all(iou(cand, q) < 0.03 for q in bl):
                X.append(np.asarray(img.crop(tuple(map(int, cand))).resize((CROP, CROP), Image.BILINEAR)))
                Y.append(BG); S.append("val" if "/val/" in path else "train"); SRC.append(os.path.basename(path)); break
    return X, Y, S, SRC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/out_v2")
    ap.add_argument("--human-repo", default="abhijitbetigeri/dcops-merged")
    ap.add_argument("--v1-repo", default="abhijitbetigeri/dcops-e8-crops")
    ap.add_argument("--arch", default="convnext_base.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: cap crops")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    rng = random.Random(0)
    from huggingface_hub import hf_hub_download, HfApi
    z = hf_hub_download(a.human_repo, "merged_dataset.zip", repo_type="dataset")
    root = "/tmp/merged"
    if not os.path.isdir(root):
        with zipfile.ZipFile(z) as f: f.extractall(root)
    inner = next(os.path.join(dp, d) for dp, dn, _ in os.walk(root) for d in dn if d == "merged_dataset")
    X, Y, S, SRC = human_crops(inner, rng)
    KIND = ["human"] * len(Y)
    # wave-1 crops (auto-labelled DC-Ops HF set): keep the classes the human set lacks + their background
    v1 = np.load(hf_hub_download(a.v1_repo, "crops.npz", repo_type="dataset"))
    human_classes = set(Y)
    for x, y, s, src in zip(v1["images"], v1["labels"], v1["split"], v1["source"]):
        if y in human_classes and y != BG: continue
        X.append(np.asarray(Image.fromarray(x).resize((CROP, CROP), Image.BILINEAR))); Y.append(int(y)); S.append(str(s)); SRC.append(str(src)); KIND.append("auto")
    X, Y, S, SRC, KIND = np.stack(X), np.array(Y), np.array(S), np.array(SRC), np.array(KIND)
    if a.limit:
        idx = np.random.default_rng(0).choice(len(Y), min(a.limit, len(Y)), replace=False)
        X, Y, S, SRC, KIND = X[idx], Y[idx], S[idx], SRC[idx], KIND[idx]
    counts = {CLASSES[c]: int((Y == c).sum()) for c in range(len(CLASSES))}
    log("crops", len(Y), "train", int((S == "train").sum()), "val", int((S == "val").sum()),
        "human val", int(((S == "val") & (KIND == "human")).sum()))
    log("per class", counts)

    import timm
    tr = np.where(S == "train")[0]; va = np.where(S == "val")[0]
    fold = np.array([int(hashlib.md5(s.encode()).hexdigest(), 16) % 2 for s in SRC])  # by source image
    Xt = torch.from_numpy(X)
    soft = np.zeros((len(Y), len(CLASSES)), np.float32); val_sum = np.zeros((len(va), len(CLASSES)), np.float32)

    def batch(model, idx, aug, mean, std):
        x = Xt[idx].to(dev).permute(0, 3, 1, 2).float() / 255
        if aug:
            if random.random() < 0.5: x = x.flip(3)
            sc = random.uniform(0.7, 1.0); c = int(CROP * sc); i, j = random.randint(0, CROP - c), random.randint(0, CROP - c)
            x = F.interpolate(x[..., i:i + c, j:j + c], size=(CROP, CROP), mode="bilinear", align_corners=False)
            x = (x * (1 + 0.3 * (torch.rand(x.shape[0], 3, 1, 1, device=dev) - 0.5))).clamp(0, 1)
        return (x - mean) / std

    states = []
    for f in (0, 1):
        trf = tr[fold[tr] == f]; held = tr[fold[tr] != f]
        model = timm.create_model(a.arch, pretrained=True, num_classes=len(CLASSES)).to(dev)
        cfg = model.pretrained_cfg
        mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
        cnt = np.bincount(Y[trf], minlength=len(CLASSES)).astype(float)
        p = 1.0 / np.sqrt(np.maximum(cnt[Y[trf]], 1)); p /= p.sum()  # sqrt-balanced sampling
        spe = max(1, len(trf) // a.batch)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-4, total_steps=a.epochs * spe, pct_start=0.1)
        scaler = torch.amp.GradScaler(enabled=dev == "cuda")
        r = np.random.default_rng(f)
        for ep in range(a.epochs):
            model.train()
            for _ in range(spe):
                idx = r.choice(trf, a.batch, p=p)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    loss = F.cross_entropy(model(batch(model, idx, True, mean, std)), torch.from_numpy(Y[idx]).to(dev), label_smoothing=0.1)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            model.eval(); pred = []
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                for i in range(0, len(va), 128):
                    pred.append(model(batch(model, va[i:i + 128], False, mean, std)).float().argmax(1).cpu().numpy())
            pred = np.concatenate(pred); hv = KIND[va] == "human"
            log(f"fold {f} epoch {ep+1}/{a.epochs} loss {loss.item():.3f} val {(pred == Y[va]).mean():.3f} human-val {(pred[hv] == Y[va][hv]).mean():.3f}")

        def logits_for(idx):
            out = np.zeros((len(idx), len(CLASSES)), np.float32)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                for i in range(0, len(idx), 128):
                    x = batch(model, idx[i:i + 128], False, mean, std)
                    out[i:i + 128] = ((model(x) + model(x.flip(3))) / 2).float().cpu().numpy()
            return out
        soft[held] = logits_for(held)
        val_sum += logits_for(va)
        states.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
        del model; torch.cuda.empty_cache()
    soft[va] = val_sum / 2
    hv = KIND[va] == "human"
    acc = float((soft[va].argmax(1) == Y[va]).mean()); acc_h = float((soft[va][hv].argmax(1) == Y[va][hv]).mean())
    oof = float((soft[tr].argmax(1) == Y[tr]).mean())
    log(f"teacher ensemble: val {acc:.3f}, human-val {acc_h:.3f}, out-of-fold train {oof:.3f}")

    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/crops_v2.npz", images=X, labels=Y, split=S, source=SRC, kind=KIND, teacher_logits=soft)
    meta = {"classes": CLASSES, "crop": CROP, "n": int(len(Y)), "counts": counts, "teacher_arch": a.arch,
            "teacher_val_acc": acc, "teacher_human_val_acc": acc_h, "teacher_oof_train_acc": oof,
            "sources": {"human": "DC-Ops merged Roboflow set (CC BY 4.0)", "auto": "dc-ops-dataset auto labels (wave 1)"},
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(f"{a.out}/meta_v2.json", "w"), indent=2)
    torch.save(states, f"{a.out}/teacher_v2_folds.pt")
    log("wrote", a.out)
    if not a.no_push:
        api = HfApi()
        for fn in ("meta_v2.json", "teacher_v2_folds.pt"):
            api.upload_file(path_or_fileobj=f"{a.out}/{fn}", path_in_repo=fn, repo_id=a.v1_repo, repo_type="dataset")
        log("pushed meta + teacher to", a.v1_repo)


if __name__ == "__main__":
    main()
