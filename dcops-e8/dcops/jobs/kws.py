# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "huggingface_hub", "numpy", "soundfile"]
# ///
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""DCSM-Simplified, voice: distil a keyword spotter for the E8 from a large speech model.

Teacher: MIT/ast-finetuned-speech-commands-v2 (Audio Spectrogram Transformer, ~87M params).
Student: DS-CNN (depthwise-separable CNN, tens of thousands of params) on the exact MFCC the
ML Embedded Evaluation Kit computes on the board (source/lib/mlek/common/Mfcc.cc with
MicroNetKwsMFCC: 16 kHz, 640-sample Hann frames, stride 320, 1024-pt FFT, 40 HTK mel bands
20-4000 Hz on the magnitude spectrum, natural log, DCT to 10 coefficients -> 49 x 10 per second).
Labels and their order match resources/kws/labels so the firmware can reuse them.

  hf jobs uv run --flavor a10g-small --timeout 5h --secrets HF_TOKEN dcops/jobs/kws.py --name dscnn_s_kd
"""
import argparse, glob, json, math, os, random, tarfile, time, urllib.request

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

LABELS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "_silence_", "_unknown_"]
KEYWORDS = LABELS[:10]
SIL, UNK = 10, 11
SR, FRAME, STRIDE, NFFT, NMEL, NMFCC, NFRAMES = 16000, 640, 320, 1024, 40, 10, 49
URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- MLEK MFCC, exactly
def mel_htk(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


def mlek_mel_filterbank():
    """Mfcc.cc CreateMelFilterBank (HTK, no normalisation), as a dense [NMEL, NFFT//2] matrix."""
    n_bins = NFFT // 2
    bin_w = SR / NFFT
    lo, hi = mel_htk(20.0), mel_htk(4000.0)
    delta = (hi - lo) / (NMEL + 1)
    fb = np.zeros((NMEL, n_bins), np.float32)
    mels = mel_htk(bin_w * np.arange(n_bins, dtype=np.float32)).astype(np.float32)
    for b in range(NMEL):
        l, c, r = lo + b * delta, lo + (b + 1) * delta, lo + (b + 2) * delta
        for i in range(n_bins):
            m = mels[i]
            if l < m < r:
                fb[b, i] = (m - l) / (c - l) if m <= c else (r - m) / (r - c)
    return fb


def mlek_dct():
    n = np.arange(NMEL, dtype=np.float32)
    k = np.arange(NMFCC, dtype=np.float32)[:, None]
    return (np.sqrt(2.0 / NMEL) * np.cos((n + 0.5) * k * np.pi / NMEL)).astype(np.float32)


class MlekMfcc(nn.Module):
    """Batched torch version of MicroNetKwsMFCC: [B, 16000] float in [-1,1] -> [B, 1, 49, 10]."""

    def __init__(self):
        super().__init__()
        i = torch.arange(FRAME, dtype=torch.float32)
        self.register_buffer("win", 0.5 - 0.5 * torch.cos(2 * math.pi * i / FRAME))
        self.register_buffer("fb", torch.from_numpy(mlek_mel_filterbank()))
        self.register_buffer("dct", torch.from_numpy(mlek_dct()))

    def forward(self, audio):
        frames = audio.unfold(1, FRAME, STRIDE)[:, :NFRAMES] * self.win          # [B,49,640]
        mag = torch.fft.rfft(frames, n=NFFT).abs()[..., : NFFT // 2]            # sqrt(power), bins 0..511
        mel = mag @ self.fb.T + 1.1754944e-38                                   # + FLT_MIN
        return (torch.log(mel) @ self.dct.T).unsqueeze(1)                       # [B,1,49,10]


# ---------------------------------------------------------------- student
class DSCNN(nn.Module):
    """DS-CNN (Hello Edge). Input [B,1,49,10]. Fixed pool + reshape so it lowers fully to Ethos-U."""

    def __init__(self, ch=64, blocks=4, n_classes=len(LABELS)):
        super().__init__()
        layers = [nn.Conv2d(1, ch, (10, 4), (2, 2), padding=(5, 1), bias=False), nn.BatchNorm2d(ch), nn.ReLU()]
        for _ in range(blocks):
            layers += [nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False), nn.BatchNorm2d(ch), nn.ReLU(),
                       nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.ch = ch
        self.pool = nn.AvgPool2d((25, 5))
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(ch, n_classes)

    def forward(self, x):
        x = self.pool(self.body(x))
        return self.fc(self.drop(x.reshape(x.shape[0], self.ch)))


# ---------------------------------------------------------------- data
def load_wav(p):
    import soundfile as sf
    a, sr = sf.read(p, dtype="float32")
    assert sr == SR
    out = np.zeros(SR, np.float32); out[: min(SR, len(a))] = a[:SR]
    return out


def prepare(root, limit=None):
    if not os.path.isdir(f"{root}/yes"):
        os.makedirs(root, exist_ok=True)
        log("downloading speech commands v0.02 ...")
        tgz = root.rstrip("/") + ".tgz"  # per-job file: parallel jobs must not share it
        urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as t: t.extractall(root)
    val = set(open(f"{root}/validation_list.txt").read().split())
    test = set(open(f"{root}/testing_list.txt").read().split())
    items = {"train": [], "val": [], "test": []}
    for d in sorted(os.listdir(root)):
        if d.startswith("_") or not os.path.isdir(f"{root}/{d}"): continue
        for f in sorted(os.listdir(f"{root}/{d}")):
            rel = f"{d}/{f}"
            sp = "val" if rel in val else "test" if rel in test else "train"
            items[sp].append((rel, d))
    noise = [load_wav_full(p) for p in sorted(glob.glob(f"{root}/_background_noise_/*.wav"))]
    rng = random.Random(0)
    out = {}
    for sp, lst in items.items():
        kw = [(r, LABELS.index(w)) for r, w in lst if w in KEYWORDS]
        unk = [(r, UNK) for r, w in lst if w not in KEYWORDS]
        rng.shuffle(unk)
        n_unk = len(kw) // len(KEYWORDS)  # about as many unknowns as an average keyword
        chosen = kw + unk[:n_unk]
        if limit: rng.shuffle(chosen); chosen = chosen[:limit]
        audio = np.stack([load_wav(f"{root}/{r}") for r, _ in chosen])
        y = np.array([c for _, c in chosen])
        words = [r.split("/")[0] for r, _ in chosen]
        n_sil = max(1, len(chosen) // 11)
        sil = []
        for _ in range(n_sil):
            n = rng.choice(noise); s = rng.randint(0, len(n) - SR)
            sil.append(n[s:s + SR] * rng.uniform(0.0, 0.5))
        out[sp] = (np.concatenate([audio, np.stack(sil)]), np.concatenate([y, np.full(n_sil, SIL)]),
                   words + ["_silence_"] * n_sil)
        log(f"{sp}: {len(out[sp][1])} clips")
    return out, noise


def load_wav_full(p):
    import soundfile as sf
    a, _ = sf.read(p, dtype="float32")
    return a


def teacher_probs(audio, words, dev, batch=128):
    """AST (35 speech-commands words) -> our 12 labels; unknown = mass on non-keyword words."""
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    name = "MIT/ast-finetuned-speech-commands-v2"
    fe = ASTFeatureExtractor.from_pretrained(name)
    m = ASTForAudioClassification.from_pretrained(name).to(dev).eval()
    id2l = m.config.id2label
    kw_idx = {w: int(i) for i, w in id2l.items() if w in KEYWORDS}
    other = [int(i) for i, w in id2l.items() if w not in KEYWORDS]
    out = np.zeros((len(audio), len(LABELS)), np.float32)
    with torch.no_grad():
        for i in range(0, len(audio), batch):
            feats = fe(list(audio[i:i + batch]), sampling_rate=SR, return_tensors="pt")["input_values"].to(dev)
            p = m(feats).logits.softmax(-1).float().cpu().numpy()
            for w, j in kw_idx.items(): out[i:i + batch, LABELS.index(w)] = p[:, j]
            out[i:i + batch, UNK] = p[:, other].sum(1)
            if i % (batch * 50) == 0: log(f"teacher {i}/{len(audio)}")
    sil = np.array([w == "_silence_" for w in words])
    out[sil] = 0; out[sil, SIL] = 1.0  # AST has no silence class: keep the hard label there
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--kd-alpha", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data", default="/tmp/speech_commands")
    ap.add_argument("--repo-out", default="abhijitbetigeri/dcops-e8-student")
    ap.add_argument("--agc", type=float, default=0.0,
                    help="probability of emulating the board's auto-gain (peak-normalise to 0.25-1.0) per clip (wave 2)")
    ap.add_argument("--calib-n", type=int, default=128, help="int8 calibration clips saved for export")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("run", a.name, vars(a), "device", dev)
    data, noise = prepare(a.data, a.limit)
    mfcc = MlekMfcc().to(dev)

    # cross-check the torch MFCC against a straight numpy port on one clip
    x0 = data["train"][0][0]
    fr = np.stack([x0[i * STRIDE: i * STRIDE + FRAME] for i in range(NFRAMES)]) * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / FRAME))
    ref = (np.log(np.abs(np.fft.rfft(fr, n=NFFT))[:, :512] @ mlek_mel_filterbank().T + 1.1754944e-38) @ mlek_dct().T)
    got = mfcc(torch.from_numpy(x0[None]).to(dev))[0, 0].cpu().numpy()
    log(f"MFCC torch vs numpy max abs diff {np.abs(ref - got).max():.2e}")

    Atr, Ytr, Wtr = data["train"]
    T = teacher_probs(Atr, Wtr, dev) if a.kd_alpha > 0 else None
    if T is not None:
        log(f"teacher train acc vs labels {(T.argmax(1) == Ytr).mean():.3f}")
    noise_t = [torch.from_numpy(n).to(dev) for n in noise]

    def augment(x):  # raw audio [B,16000]
        b = x.shape[0]
        shift = torch.randint(-1600, 1601, (b,))
        x = torch.stack([torch.roll(x[i], int(shift[i])) for i in range(b)])
        mix = torch.rand(b, device=dev) < 0.8
        nz = []
        for _ in range(b):
            n = random.choice(noise_t); s = random.randint(0, len(n) - SR); nz.append(n[s:s + SR])
        nz = torch.stack(nz) * torch.empty(b, 1, device=dev).uniform_(0, 0.1)
        x = x + nz * mix[:, None]
        return agc(x, a.agc)

    def agc(x, prob):
        """The board's audio HAL scales each window toward full scale (up to +80 dB): quiet speech
        and even silence arrive loud. Emulate that by peak-normalising a share of the clips."""
        if prob <= 0: return x
        peak = x.abs().amax(1, keepdim=True).clamp_min(1e-4)
        target = torch.empty_like(peak).uniform_(0.25, 1.0)
        gain = (target / peak).clamp(max=10 ** (80 / 20))
        sel = (torch.rand_like(peak) < prob).float()
        return x * (sel * gain + (1 - sel))

    model = DSCNN(a.ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    spe = len(Ytr) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.epochs * spe)
    Atr_t = torch.from_numpy(Atr)

    def accuracy(split, board=False):
        A, Y, _ = data[split]; model.eval(); pred = []
        with torch.no_grad():
            for i in range(0, len(Y), 1024):
                x = torch.from_numpy(A[i:i + 1024]).to(dev)
                if board: x = agc(x, 1.0)  # every window peak-normalised, as on the board
                pred.append(model(mfcc(x)).argmax(1).cpu().numpy())
        model.train(); pred = np.concatenate(pred)
        return float((pred == Y).mean()), pred

    best, best_state, hist = -1, None, []
    for ep in range(a.epochs):
        perm = np.random.permutation(len(Ytr))
        for s in range(spe):
            idx = perm[s * a.batch:(s + 1) * a.batch]
            x = mfcc(augment(Atr_t[idx].to(dev)))
            logits = model(x); y = torch.from_numpy(Ytr[idx]).to(dev)
            loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            if T is not None:
                t = torch.from_numpy(T[idx]).to(dev).clamp_min(1e-6)
                t = F.softmax(torch.log(t) / a.temp, 1)
                loss = (1 - a.kd_alpha) * loss + a.kd_alpha * a.temp ** 2 * F.kl_div(F.log_softmax(logits / a.temp, 1), t, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc, _ = accuracy("val"); hist.append({"epoch": ep + 1, "loss": float(loss), "val_acc": acc})
        log(f"epoch {ep+1}/{a.epochs} loss {loss.item():.3f} val {acc:.4f}")
        if acc > best: best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_acc, pred = accuracy("test")
    test_acc_board, _ = accuracy("test", board=True)
    log(f"test {test_acc:.4f}, test with board auto-gain {test_acc_board:.4f}")
    Yte = data["test"][1]
    per_class = {LABELS[c]: float((pred[Yte == c] == c).mean()) for c in range(len(LABELS)) if (Yte == c).any()}
    log(f"best val {best:.4f}, test {test_acc:.4f}")

    out = f"/tmp/kws_{a.name}"; os.makedirs(out, exist_ok=True)
    torch.save(best_state, f"{out}/student.pt")
    ci = np.random.default_rng(1).choice(len(Yte), min(a.calib_n, len(Yte)), replace=False)
    with torch.no_grad():
        xa = torch.from_numpy(data["test"][0][ci]).to(dev)
        if a.agc > 0: xa = agc(xa, 1.0)  # calibrate on board-like levels
        feats = mfcc(xa).cpu().numpy()
    np.savez_compressed(f"{out}/calib.npz", mfcc=feats, labels=Yte[ci])
    open(f"{out}/labels.txt", "w").write("\n".join(LABELS) + "\n")
    json.dump({"name": a.name, "args": vars(a), "arch": {"ch": a.ch, "blocks": 4}, "val_acc": best,
               "test_acc": test_acc, "test_acc_board_agc": test_acc_board, "per_class_test_acc": per_class, "history": hist,
               "params": sum(p.numel() for p in model.parameters()),
               "mfcc": {"sr": SR, "frame": FRAME, "stride": STRIDE, "nfft": NFFT, "nmel": NMEL,
                        "nmfcc": NMFCC, "frames": NFRAMES, "mel_hz": [20, 4000], "htk": True}},
              open(f"{out}/metrics.json", "w"), indent=2)
    if a.no_push:
        log("no push; outputs in", out); return
    from huggingface_hub import HfApi
    api = HfApi(); api.create_repo(a.repo_out, private=True, exist_ok=True)
    api.upload_folder(folder_path=out, path_in_repo=f"kws/{a.name}", repo_id=a.repo_out)
    log("pushed", f"{a.repo_out}/kws/{a.name}")


if __name__ == "__main__":
    main()
