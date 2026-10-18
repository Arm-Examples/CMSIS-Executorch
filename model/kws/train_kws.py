"""Train the keyword spotter on Google Speech Commands v2 (resampled to 8 kHz, the
board's mic rate). Background noise, time shifts and spectrogram masking are
mixed in on the fly so it copes with a noisy room.

Run from the repo root:
  .venv/Scripts/python.exe model/kws/train_kws.py --data ../kws_data/sc
"""

import argparse
import random
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kws_features import CLASSES, N_SAMPLES, WORDS, features_torch  # noqa: E402
from kws_net import KwsNet  # noqa: E402

SIL, UNK = CLASSES.index("_silence_"), CLASSES.index("_unknown_")


def load_wav8k(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        sr = w.getframerate()
    if sr == 16000:
        x = resample_poly(x, 1, 2).astype(np.float32)
    return x


def fit(x):
    out = np.zeros(N_SAMPLES, np.float32)
    n = min(len(x), N_SAMPLES)
    out[:n] = x[:n]
    return out


def build(root, cache):
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    root = Path(root)
    val = set((root / "validation_list.txt").read_text().split())
    test = set((root / "testing_list.txt").read_text().split())
    rng = random.Random(0)
    split = {"train": ([], []), "val": ([], []), "test": ([], [])}
    others = [d.name for d in root.iterdir() if d.is_dir() and d.name not in WORDS and not d.name.startswith("_")]
    t0 = time.time()
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        files = sorted(d.glob("*.wav"))
        if d.name in WORDS:
            label = CLASSES.index(d.name)
        else:  # unknown words: a sample of each
            rng.shuffle(files)
            files = files[: 4500 // len(others) * 2]
            label = UNK
        for f in files:
            rel = f"{d.name}/{f.name}"
            s = "val" if rel in val else "test" if rel in test else "train"
            split[s][0].append(fit(load_wav8k(f)))
            split[s][1].append(label)
        print(f"  {d.name}: {len(files)} clips ({time.time() - t0:.0f} s)", flush=True)
    noise = [load_wav8k(f) for f in sorted((root / "_background_noise_").glob("*.wav"))]
    out = {}
    for s, (xs, ys) in split.items():
        n_sil = len(ys) // 10  # silence: zeros here, noise is added at training time
        out[f"x_{s}"] = np.concatenate([np.stack(xs), np.zeros((n_sil, N_SAMPLES), np.float32)]).astype(np.float16)
        out[f"y_{s}"] = np.array(ys + [SIL] * n_sil, np.int64)
    out["noise"] = np.concatenate(noise).astype(np.float32)
    np.savez(cache, **out)
    return out


def augment(x, y, noise, train, gen):
    """x: (B, 8000) float32 clean clips -> noisy, shifted clips"""
    B = x.shape[0]
    if train:
        sh = torch.randint(-1200, 1201, (B,), generator=gen)
        idx = (torch.arange(N_SAMPLES)[None] - sh[:, None])
        valid = (idx >= 0) & (idx < N_SAMPLES)
        x = torch.gather(x, 1, idx.clamp(0, N_SAMPLES - 1)) * valid
    st = torch.randint(0, noise.shape[0] - N_SAMPLES, (B,), generator=gen)
    nz = torch.stack([noise[s:s + N_SAMPLES] for s in st.tolist()])
    nz = nz / nz.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)  # unit RMS
    sp_rms = x.pow(2).mean(1, keepdim=True).sqrt()
    snr = torch.empty(B, 1).uniform_(0.0, 25.0, generator=gen) if train else torch.full((B, 1), 15.0)
    amp = sp_rms * 10 ** (-snr / 20)
    use = (torch.rand(B, 1, generator=gen) < (0.8 if train else 1.0)).float()
    sil = (y == SIL)[:, None].float()
    sil_amp = torch.empty(B, 1).uniform_(0.0005, 0.1, generator=gen)  # silence: just room noise
    x = x + nz * (amp * use * (1 - sil) + sil_amp * sil)
    x = x + torch.randn(B, N_SAMPLES, generator=gen) * 1e-4  # mic hiss
    return x


def spec_mask(f, gen):
    """half the clips get one time mask (<= 6 frames) and one band mask (<= 5 bands)"""
    B, _, T, M = f.shape
    on = torch.rand(B, generator=gen) < 0.5
    tw = torch.randint(0, 7, (B,), generator=gen); t0 = (torch.rand(B, generator=gen) * (T - tw + 1)).long()
    mw = torch.randint(0, 6, (B,), generator=gen); m0 = (torch.rand(B, generator=gen) * (M - mw + 1)).long()
    t = torch.arange(T)[None]; m = torch.arange(M)[None]
    tmask = (t >= t0[:, None]) & (t < (t0 + tw)[:, None]) & on[:, None]
    mmask = (m >= m0[:, None]) & (m < (m0 + mw)[:, None]) & on[:, None]
    keep = ~(tmask[:, :, None] | mmask[:, None, :])
    return f * keep[:, None].float()


def evaluate(model, x, y, noise, gen, bs=512):
    model.eval()
    pred = []
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb = torch.from_numpy(x[i:i + bs].astype(np.float32))
            yb = torch.from_numpy(y[i:i + bs])
            pred.append(model(features_torch(augment(xb, yb, noise, False, gen))).argmax(1))
    pred = torch.cat(pred).numpy()
    return (pred == y).mean(), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE.parents[2] / "kws_data" / "sc"))
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()
    torch.set_num_threads(14)
    cache = Path(args.data).parent / "sc8k.npz"
    print("loading data (first run resamples every clip, a few minutes)...", flush=True)
    d = build(args.data, cache)
    noise = torch.from_numpy(d["noise"])
    xtr, ytr = d["x_train"], d["y_train"]
    print("train", len(ytr), "val", len(d["y_val"]), "test", len(d["y_test"]),
          "per class", np.bincount(ytr, minlength=len(CLASSES)).tolist(), flush=True)

    model = KwsNet().to(memory_format=torch.channels_last)  # faster depthwise convs on the CPU
    print("params", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    steps = args.epochs * (len(ytr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    gen = torch.Generator().manual_seed(0)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(ytr))
        t0, tl = time.time(), 0.0
        for k in range(len(ytr) // args.bs):
            b = perm[k * args.bs:(k + 1) * args.bs]
            xb = torch.from_numpy(xtr[b].astype(np.float32))
            yb = torch.from_numpy(ytr[b])
            f = spec_mask(features_torch(augment(xb, yb, noise, True, gen)), gen)
            loss = loss_fn(model(f.contiguous(memory_format=torch.channels_last)), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item()
        acc, _ = evaluate(model, d["x_val"], d["y_val"], noise, torch.Generator().manual_seed(1))
        print(f"epoch {ep + 1}/{args.epochs} loss {tl / (k + 1):.3f} val {acc * 100:.1f}% ({time.time() - t0:.0f} s)", flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), HERE / "kws_net.pt")
    model.load_state_dict(torch.load(HERE / "kws_net.pt"))
    acc, pred = evaluate(model, d["x_test"], d["y_test"], noise, torch.Generator().manual_seed(2))
    print(f"best val {best * 100:.1f}%, test {acc * 100:.1f}%")
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(d["y_test"], pred):
        cm[t, p] += 1
    print("confusion (rows = truth):", " ".join(c[:5] for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c[:9]:9s}", " ".join(f"{v:5d}" for v in cm[i]))
    # calibration set for the int8 export: clean-ish val clips with features
    idx = np.random.RandomState(0).permutation(len(d["y_val"]))[:300]
    xc = torch.from_numpy(d["x_val"][idx].astype(np.float32))
    feats = features_torch(augment(xc, torch.from_numpy(d["y_val"][idx]), noise, False, torch.Generator().manual_seed(3)))
    np.save(HERE / "calib.npy", feats.numpy())


if __name__ == "__main__":
    main()
