# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch", "torchvision", "timm", "ultralytics", "huggingface_hub", "numpy", "pillow", "opencv-python-headless",
# ]
# ///
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""DCSM-Simplified, stage 1 (teacher): build component crops and fine-tune a large teacher.

1. Download the DC-Ops dataset (319 images, auto-labelled polygons) and the fine-tuned
   DC-Ops YOLOv8n-seg detector (16 classes) from the Hub.
2. Run the YOLO detector over every image (GPU) and merge its boxes with the dataset's own
   labels: these are the component crops. Add background crops that avoid every box.
3. Fine-tune a large image classifier (ConvNeXt, ImageNet-22k) on the crops: the teacher.
4. Write the teacher's soft labels (logits) for every crop and push crops + logits to a private
   dataset repo; the student jobs distil from it.

Run on HF Jobs (GPU):
  hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN dcops/jobs/teacher.py
"""
import argparse, hashlib, io, json, os, random, time

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

CLASSES = ["server rack", "compute tray", "NVLink switch tray", "network switch", "power shelf",
           "cable", "network port", "LED indicator", "label", "fan", "cooling manifold",
           "cable cartridge", "power connector", "drive bay", "management port", "DPU", "background"]
BG = CLASSES.index("background")
# Dataset's own 8-class auto-labels -> DC-Ops classes (only the unambiguous ones).
HF_MAP = {"server rack": 0, "server": 1, "cable": 5, "LED light": 7, "label": 8,
          "network port": 6, "power supply": 4}
CROP = 224


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def square_crop(img, box, context=1.3, size=CROP):
    w, h = img.size
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    s = max(box[2] - box[0], box[3] - box[1]) * context
    s = max(s, 16)
    x0, y0 = cx - s / 2, cy - s / 2
    return img.crop((int(x0), int(y0), int(x0 + s), int(y0 + s))).resize((size, size), Image.BILINEAR)


def split_of(name):  # deterministic 85/15 split by source image
    return "val" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 100 < 15 else "train"


def build_crops(root, yolo_path, device, limit=None, conf=0.30):
    from ultralytics import YOLO
    names = open(f"{root}/labels/classes.txt").read().split("\n")
    teacher = YOLO(yolo_path)
    imgs = sorted(os.listdir(f"{root}/images"))[:limit]
    X, Y, S, SRC, KIND = [], [], [], [], []
    rng = random.Random(0)
    for k, fn in enumerate(imgs):
        try:
            img = Image.open(f"{root}/images/{fn}").convert("RGB")
        except Exception as e:  # noqa: BLE001
            log("skip", fn, e); continue
        w, h = img.size
        stem = os.path.splitext(fn)[0]
        boxes = []  # (x0,y0,x1,y1, cls, source)
        r = teacher.predict(img, conf=conf, imgsz=640, device=device, verbose=False, retina_masks=False)[0]
        for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            boxes.append((*b, int(c), "yolo"))
        lab = f"{root}/labels/{stem}.txt"
        if os.path.exists(lab):
            for line in open(lab):
                p = line.split()
                if len(p) < 5: continue
                nm = names[int(p[0])] if int(p[0]) < len(names) else None
                if nm not in HF_MAP: continue
                xy = np.array(p[1:], dtype=float).reshape(-1, 2) * [w, h]
                b = (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
                if all(iou(b, q[:4]) < 0.6 for q in boxes):  # teacher box wins on overlap
                    boxes.append((*b, HF_MAP[nm], "label"))
        sp = split_of(stem)
        for b in boxes:
            if (b[2] - b[0]) < 12 or (b[3] - b[1]) < 12: continue
            X.append(np.asarray(square_crop(img, b[:4]))); Y.append(b[4]); S.append(sp); SRC.append(stem); KIND.append(b[5])
        # background crops that avoid every box
        n_bg = 0
        for _ in range(40):
            if n_bg >= 3: break
            s = rng.uniform(0.15, 0.5) * min(w, h)
            x0, y0 = rng.uniform(0, w - s), rng.uniform(0, h - s)
            cand = (x0, y0, x0 + s, y0 + s)
            if all(iou(cand, q[:4]) < 0.05 for q in boxes):
                X.append(np.asarray(img.crop(tuple(map(int, cand))).resize((CROP, CROP), Image.BILINEAR)))
                Y.append(BG); S.append(sp); SRC.append(stem); KIND.append("bg"); n_bg += 1
        if k % 25 == 0:
            log(f"crops: {k+1}/{len(imgs)} images -> {len(X)} crops")
    return np.stack(X), np.array(Y), np.array(S), np.array(SRC), np.array(KIND)


def augment(x):  # x: float [B,3,H,W] in [0,1]
    if random.random() < 0.5: x = x.flip(3)
    b = x.shape[0]
    gain = 1 + 0.3 * (torch.rand(b, 3, 1, 1, device=x.device) - 0.5)
    x = (x * gain).clamp(0, 1)
    sc = random.uniform(0.75, 1.0); hw = x.shape[-1]; c = int(hw * sc)
    i, j = random.randint(0, hw - c), random.randint(0, hw - c)
    return F.interpolate(x[..., i:i + c, j:j + c], size=(hw, hw), mode="bilinear", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-in", default="abhijitbetigeri/dc-ops-dataset")
    ap.add_argument("--repo-out", default="abhijitbetigeri/dcops-e8-crops")
    ap.add_argument("--arch", default="convnext_small.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    from huggingface_hub import HfApi, snapshot_download
    root = snapshot_download(a.repo_in, repo_type="dataset", local_dir="/tmp/dcops",
                             allow_patterns=["images/*", "labels/*", "models/dc_ops_yolov8n_seg_v3.pt"])
    X, Y, S, SRC, KIND = build_crops(root, f"{root}/models/dc_ops_yolov8n_seg_v3.pt",
                                     0 if dev == "cuda" else "cpu", a.limit)
    counts = {CLASSES[c]: int((Y == c).sum()) for c in range(len(CLASSES))}
    log("crops", len(Y), "train", int((S == "train").sum()), "val", int((S == "val").sum()))
    log("per class", counts)

    import timm
    model = timm.create_model(a.arch, pretrained=True, num_classes=len(CLASSES)).to(dev)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
    tr, va = np.where(S == "train")[0], np.where(S == "val")[0]
    w = np.array([1.0 / max(1, counts[CLASSES[c]]) for c in Y[tr]]); w /= w.sum()  # class-balanced sampling
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    steps = a.epochs * max(1, len(tr) // 32); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-4, total_steps=steps)
    Xt = torch.from_numpy(X)

    def batch(idx, aug):
        x = Xt[idx].to(dev).permute(0, 3, 1, 2).float() / 255
        if aug: x = augment(x)
        return (x - mean) / std

    def evaluate():
        model.eval(); correct = 0
        with torch.no_grad():
            for i in range(0, len(va), 64):
                idx = va[i:i + 64]
                correct += (model(batch(idx, False)).argmax(1).cpu().numpy() == Y[idx]).sum()
        model.train(); return correct / max(1, len(va))

    best, best_state = -1, None
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        for _ in range(max(1, len(tr) // 32)):
            idx = rng.choice(tr, 32, p=w)
            loss = F.cross_entropy(model(batch(idx, True)), torch.from_numpy(Y[idx]).to(dev), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = evaluate(); log(f"teacher epoch {ep+1}/{a.epochs} loss {loss.item():.3f} val acc {acc:.3f}")
        if acc > best: best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()

    logits = np.zeros((len(Y), len(CLASSES)), np.float32)
    with torch.no_grad():
        for i in range(0, len(Y), 64):
            idx = np.arange(i, min(i + 64, len(Y)))
            x = batch(idx, False)
            logits[idx] = ((model(x) + model(x.flip(3))) / 2).cpu().numpy()
    val_acc_tta = float((logits[va].argmax(1) == Y[va]).mean()) if len(va) else 0.0
    log(f"teacher best val acc {best:.3f}, with flip TTA {val_acc_tta:.3f}")

    os.makedirs("/tmp/out", exist_ok=True)
    np.savez_compressed("/tmp/out/crops.npz", images=X, labels=Y, split=S, source=SRC, kind=KIND, teacher_logits=logits)
    meta = {"classes": CLASSES, "crop": CROP, "n": int(len(Y)), "counts": counts, "teacher_arch": a.arch,
            "teacher_val_acc": float(best), "teacher_val_acc_tta": val_acc_tta,
            "yolo_teacher": "dc_ops_yolov8n_seg_v3.pt", "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open("/tmp/out/meta.json", "w"), indent=2)
    torch.save(best_state, "/tmp/out/teacher.pt")
    if a.no_push:
        log("no push; outputs in /tmp/out"); return
    api = HfApi()
    api.create_repo(a.repo_out, repo_type="dataset", private=True, exist_ok=True)
    for f in ("crops.npz", "teacher.pt", "meta.json"):  # meta.json last: students wait for it
        api.upload_file(path_or_fileobj=f"/tmp/out/{f}", path_in_repo=f, repo_id=a.repo_out, repo_type="dataset")
    log("pushed to", a.repo_out)


if __name__ == "__main__":
    main()
