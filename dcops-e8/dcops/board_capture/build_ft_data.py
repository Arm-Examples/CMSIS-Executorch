"""Assemble the board fine-tuning set: cleaned board frames + a replay set of human crops."""
import glob, os, random
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["server rack", "compute tray", "NVLink switch tray", "network switch", "power shelf", "cable",
           "network port", "LED indicator", "label", "fan", "cooling manifold", "cable cartridge",
           "power connector", "drive bay", "management port", "DPU", "background"]

X, Y, G = [], [], []
gid = 0
# new captures land in frames/; the captures used for the deployed model are in dcops/data/board_captures/
FILES = sorted(glob.glob(f"{HERE}/frames/*.npz")) or sorted(glob.glob(f"{HERE}/../data/board_captures/*.npz"))
for f in FILES:
    d = np.load(f); prev = None; prev_y = None
    sid = d["slide_ids"] if "slide_ids" in d.files else np.full(len(d["labels"]), -1)
    keep = d["keep"] if "keep" in d.files else np.ones(len(d["labels"]), bool)  # hand-marked: phone in view
    for x, y, s_, k_ in zip(d["images"], d["labels"], sid, keep):
        if not k_:
            continue
        if x.std() < 8:                                  # blank / black frame
            continue
        if prev is not None and np.abs(x.astype(int) - prev.astype(int)).mean() < 1.0:
            continue                                      # stale: camera did not deliver a new frame
        if "background" in f:
            gid += 1 if len(Y) % 6 == 0 else 0            # background: blocks of ~6 consecutive frames
        elif s_ >= 0:
            gid = (200000 if "deck2" in f else 100000) + int(s_)  # recorded slide id: split by slide
        elif y == 16:
            gid = 300000 + len(Y) // 8                    # unlabelled scenes: blocks of 8 frames
        elif y != prev_y:
            gid += 1                                      # a new slide starts when the label changes
        X.append(x); Y.append(int(y)); G.append(gid); prev, prev_y = x, y
X, Y, G = np.stack(X), np.array(Y), np.array(G)
groups = sorted(set(G.tolist()))
test_groups = set(groups[3::6])                           # every 6th slide/block held out
S = np.array(["test" if g in test_groups else "train" for g in G])
print("board frames", len(Y), "train", int((S == "train").sum()), "test", int((S == "test").sum()))
print("per class", {CLASSES[c]: int((Y == c).sum()) for c in sorted(set(Y.tolist()))})

# replay: human-labelled training crops (and background) at the model resolution
root = os.path.join(HERE, "../data/dc_components/train")
random.seed(0); per_cls = {}
labs = sorted(glob.glob(f"{root}/labels/*.txt")); random.shuffle(labs)
RX, RY = [], []
for lab in labs:
    stem = os.path.splitext(os.path.basename(lab))[0]
    img = next((f"{root}/images/{stem}{e}" for e in (".jpg", ".jpeg", ".png", ".webp") if os.path.exists(f"{root}/images/{stem}{e}")), None)
    if img is None: continue
    rows = [l.split() for l in open(lab) if len(l.split()) >= 5]
    if not rows: continue
    im = Image.open(img).convert("RGB"); w, h = im.size
    for p in rows[:3]:
        c = int(p[0])
        if per_cls.get(c, 0) >= 180: continue
        xy = [float(v) for v in p[1:]]; xs, ys = xy[0::2], xy[1::2]
        b = (min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h)
        if b[2] - b[0] < 16 or b[3] - b[1] < 16: continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = max(b[2] - b[0], b[3] - b[1]) * random.uniform(1.15, 1.5)
        RX.append(np.asarray(im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((160, 160))))
        RY.append(c); per_cls[c] = per_cls.get(c, 0) + 1
    if per_cls.get(16, 0) < 180:                          # one background crop per image
        s = random.uniform(0.15, 0.4) * min(w, h); x0, y0 = random.uniform(0, w - s), random.uniform(0, h - s)
        RX.append(np.asarray(im.crop((int(x0), int(y0), int(x0 + s), int(y0 + s))).resize((160, 160)))); RY.append(16)
        per_cls[16] = per_cls.get(16, 0) + 1
    if len(RY) >= 180 * 9: break
print("replay", len(RY), {CLASSES[c]: n for c, n in sorted(per_cls.items())})
out = f"{HERE}/board_ft.npz"
np.savez_compressed(out, board_x=X, board_y=Y, board_split=S, replay_x=np.stack(RX), replay_y=np.array(RY),
                    classes=np.array(CLASSES))
print("wrote", out, round(os.path.getsize(out) / 1e6, 1), "MB")
