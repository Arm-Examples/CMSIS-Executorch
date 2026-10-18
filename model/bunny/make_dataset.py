"""Render the bunny training set: white background, labelled by dial vector.
usage: python make_dataset.py OUT.npz N SIZE [SEED] [WORKERS]
Mix: 55% clip frames (coherent motion), 25% transitions between clips, 20% random dials with extremes oversampled.
Every sample also gets random identity dials (hue, sat, ear length, head scale) so each bunny can be unique."""
import sys, time
import numpy as np
from multiprocessing import Pool
from bunny_rig3 import (render, default_dials, to_vec, from_vec, SCALARS, EYES, MOUTHS, DIAL_NAMES, NDIAL)
from clips import CLIPS, DURATION, clip

SC = {n: (lo, hi) for n, _, lo, hi in SCALARS}
IDENT = ["hue", "sat", "ear_len", "head_scale"]
CLIP_NAMES = list(CLIPS)


def clean(d, rng):
    """Add slight noise, clip to range, renormalise eye/mouth weights."""
    d = dict(d)
    for n, (lo, hi) in SC.items():
        d[n] = float(np.clip(d[n] + rng.normal(0, 0.012 * (hi - lo)), lo, hi))
    for grp, pre in ((EYES, "eye_"), (MOUTHS, "mouth_")):
        w = np.array([max(0.0, d[pre + k] + rng.normal(0, 0.015)) for k in grp])
        w = w / (w.sum() or 1.0)
        for k, v in zip(grp, w): d[pre + k] = float(v)
    return d


def identity(d, rng):
    for n in IDENT:
        lo, hi = SC[n]; d[n] = float(rng.uniform(lo, hi))
    return d


def clip_dials(rng):
    name = CLIP_NAMES[rng.integers(len(CLIP_NAMES))]
    dur = DURATION.get(name, 2.4)
    d, _, _, _ = clip(name, float(rng.uniform(0, dur)), float(rng.uniform(0.75, 1.15)))
    return d, name


def random_dials(rng):
    d = default_dials()
    for n, _, lo, hi in SCALARS:
        d[n] = float(lo + (hi - lo) * rng.beta(0.6, 0.6))      # U-shaped: extremes get lots of samples
    r = rng.random()
    d["lie"] = 0.0 if r < 0.7 else (1.0 if r < 0.85 else float(rng.uniform()))
    if rng.random() < 0.7:                                      # ears usually roughly symmetric
        d["ear_r"] = -d["ear_l"] * float(rng.uniform(0.7, 1.3))
        d["ear_r"] = float(np.clip(d["ear_r"], *SC["ear_r"]))
    if rng.random() < 0.5:                                      # arms often mirrored too
        d["arm_r"] = float(np.clip(-d["arm_l"] * rng.uniform(0.7, 1.3), *SC["arm_r"]))
    if d["lie"] > 0.5:
        d["tilt"] = 0.0; d["squash"] = float(np.clip(d["squash"], 0.88, 1.0))
    for grp, pre in ((EYES, "eye_"), (MOUTHS, "mouth_")):
        k = grp[rng.integers(len(grp))]
        w = {g: 0.0 for g in grp}; w[k] = 1.0
        if rng.random() < 0.3:
            k2 = grp[rng.integers(len(grp))]; u = float(rng.uniform())
            w = {g: 0.0 for g in grp}; w[k] += 0; w[k] = 1 - u; w[k2] += u
        for g, v in w.items(): d[pre + g] = v
    return d, "random"


def sample(rng):
    r = rng.random()
    if r < 0.55:
        d, tag = clip_dials(rng)
    elif r < 0.80:
        a, na = clip_dials(rng); b, nb = clip_dials(rng); u = float(rng.uniform())
        d = {k: a[k] + (b[k] - a[k]) * u for k in a}; tag = f"{na}>{nb}"
    else:
        d, tag = random_dials(rng)
    return identity(clean(d, rng), rng), tag


def work(args):
    seed, n, size = args
    rng = np.random.default_rng(seed)
    imgs = np.empty((n, size, size, 3), np.uint8); dials = np.empty((n, NDIAL), np.float32); tags = []
    for i in range(n):
        d, tag = sample(rng)
        imgs[i] = np.asarray(render(d, size, ss=2))
        dials[i] = to_vec(d); tags.append(tag)
    return imgs, dials, tags


if __name__ == "__main__":
    out, N, SIZE = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    chunk = 250
    jobs = [(seed * 100000 + j, min(chunk, N - j * chunk), SIZE) for j in range((N + chunk - 1) // chunk)]
    t0 = time.time(); I, D, T = [], [], []
    with Pool(workers) as p:
        for k, (im, di, tg) in enumerate(p.imap(work, jobs)):
            I.append(im); D.append(di); T += tg
            if k % 10 == 0: print(f"{(k+1)*chunk}/{N}  {time.time()-t0:.0f}s", flush=True)
    I = np.concatenate(I); D = np.concatenate(D)
    rng = np.random.default_rng(seed + 7)
    val = rng.random(len(I)) < 0.05
    np.savez_compressed(out, images=I, dials=D, tags=np.array(T), dial_names=np.array(DIAL_NAMES), val_mask=val)
    print("saved", out, I.shape, D.shape, f"{time.time()-t0:.0f}s")
