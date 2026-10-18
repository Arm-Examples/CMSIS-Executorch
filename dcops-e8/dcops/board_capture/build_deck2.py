"""Deck 2: more and more varied slides for the 5 demo classes (judges show their own pictures).

Picks photos not used by deck 1 from the DC-Ops training split, crops each around one labelled
component at a random zoom (tight close-up up to the whole picture), and writes
  <out>/s2/NNN.jpg  and  slides2.json  ([file, name, class id] in show order).
"""
import glob, json, os, random, sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["server rack", "compute tray", "NVLink switch tray", "network switch", "power shelf", "cable",
           "network port", "LED indicator", "label", "fan", "cooling manifold", "cable cartridge",
           "power connector", "drive bay", "management port", "DPU", "background"]
DEMO = [4, 6, 9, 12, 13]          # power shelf, network port, fan, power connector, drive bay
PER = 30
out = sys.argv[1]
os.makedirs(f"{out}/s2", exist_ok=True)
root = os.path.join(HERE, "../data/dc_components/train")
random.seed(2)
labs = sorted(glob.glob(f"{root}/labels/*.txt")); random.shuffle(labs)
picked = {c: [] for c in DEMO}
for lab in labs:
    stem = os.path.splitext(os.path.basename(lab))[0]
    img = next((f"{root}/images/{stem}{e}" for e in (".jpg", ".jpeg", ".png", ".webp")
                if os.path.exists(f"{root}/images/{stem}{e}")), None)
    if img is None:
        continue
    rows = [l.split() for l in open(lab) if len(l.split()) >= 5]
    im = None
    for r in sorted(rows, key=lambda r: -len(r)):           # every demo-class box, scarce classes too
        c = int(r[0])
        if c not in picked or len(picked[c]) >= PER:
            continue
        if im is None:
            im = Image.open(img).convert("RGB"); w, h = im.size
        xy = [float(v) for v in r[1:]]; xs, ys = xy[0::2], xy[1::2]
        b = (min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h)
        if b[2] - b[0] < 32 or b[3] - b[1] < 32:
            continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        s = min(max(b[2] - b[0], b[3] - b[1]) * random.uniform(1.1, 2.2), max(w, h))
        picked[c].append(im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((560, 560)))
        break                                                # one slide per photo
    if all(len(v) >= PER for v in picked.values()):
        break
order = [(c, im) for c in DEMO for im in picked[c]]
random.shuffle(order)
slides = []
for k, (c, im) in enumerate(order):
    f = f"{k:03d}.jpg"; im.save(f"{out}/s2/{f}", quality=85)
    slides.append([f, CLASSES[c], c])
json.dump(slides, open(f"{HERE}/slides2.json", "w"))
print(len(slides), "slides", {CLASSES[c]: len(v) for c, v in picked.items()})
