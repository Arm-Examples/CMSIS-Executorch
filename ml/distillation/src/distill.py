# SPDX-License-Identifier: Apache-2.0
"""Distil the v2 teacher ensemble into a TinyEEGNet student at 125 Hz / 256 samples.

    python -m src.distill --seed 0 --student wide32 --teacher runs/v2/teacher_seed0 \\
        --out runs/v2/student_wide32_seed0

Student = ``TinyEEGNet(n_samples=window, dropout=training.dropout, **cfg["student"][name])``.
Per batch (own loop, modelled on ``src.train.fit``):

1. training windows (``train_window`` = 306 samples) get a random crop of ``window`` = 256
   samples, offset uniform per trial in ``crop_offset ± distill.crop_jitter`` clipped to
   ``[0, train_window - window]`` (the whole range when ``crop_jitter`` is absent);
   the 2a calibration trials (session 2 trials 1-24 of every test subject, already the canonical
   window, **no labels**) are shuffled into the same stream;
2. Gaussian noise ``noise_std`` (inputs are unit scale after ``input_scale``) and channel dropout
   (whole channels zeroed with probability ``channel_dropout``, no rescaling);
3. the teacher ensemble scores the **same augmented batch** without gradients:
   ``p_T = mean_m softmax(z_m / T)``;
4. loss per trial = ``(1 - alpha) CE(z_s, y) + alpha T^2 KL(p_T || softmax(z_s / T))``; calibration
   trials (label ``-1``) contribute the KD term only. The batch loss is the mean over trials.

Validation (canonical crop, no augmentation, teacher soft labels precomputed): early stopping
and the restored best weights follow the validation KL ``KL(p_T || softmax(z_s / T))`` alone
(``best_val_kl`` in ``metrics.json``, no ``T^2`` factor); the combined loss on the validation set
is logged as ``val_loss`` (``best_val_loss`` = its value at the selected epoch). The 2a test block
is only scored after training.

Writes to ``--out``: ``model.pt``, ``student.json`` (the five width keys for
``src.models.tinyeegnet.get_model``), ``predictions.npz`` (``subjects``, and per 2a subject ``s``
``y_true_{s}`` / ``logits_{s}`` on session 2 trials 25-144), ``calibration.npz`` (``x``
(200, 1, 22, 256) float32 scaled canonical training windows for PTQ), ``params.json`` (as the
teacher's plus ``student``), ``metrics.json``, ``history.json``, ``train.log``.

v3 (``training.loop: v3``): ``src.train_v3.distill_fit_v3`` replaces the loop above: the shared
``src.augment`` pipeline (``augment`` section: crop, EA-matrix mixing, amplitude jitter, noise,
channel dropout) plus mixup under KD (``distill.mixup_alpha``, ``distill.mixup_p``), AdamW
(``distill.weight_decay``), cosine learning rate with ``distill.warmup`` epochs of warm-up and
an EMA of the weights (``distill.ema``). The validation set is whatever ``v2_data`` returns
(cross-subject in v3). Without the key the v2 loop runs unchanged.

v4 (``distill.target_unlabeled: true``): the unlabelled target session of every test subject
(``splits125.target_unlabeled``, BCI IV 2a session 1, 144 x 9 = 1,296 windows of 256 samples, EA
from that session's own trials, X only) is concatenated with the calibration trials: KD term only
(label -1), no random crop. The teacher is whatever ``members`` the config lists (``riemann``
included, loaded from ``members/riemann.pkl``).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.eval import KPI_KEYS, compute_kpis, confusion
from src.models.tinyeegnet import STUDENT_KEYS, TinyEEGNet
from src.train import predict, set_seed, to_model_input
from src.train_teacher import (CONFIG_V2, MEMBERS, canonical, jsonable, load_members, load_target_unlabeled,
                               load_v2_config, load_v2_data, member_names, subset_cho, write_params)

N_CALIB = 200


# ----------------------------------------------------------------------------- pieces
def build_student(name: str, cfg: dict) -> TinyEEGNet:
    widths = {k: int(cfg["student"][name][k]) for k in STUDENT_KEYS}
    return TinyEEGNet(n_samples=int(cfg["window"]), dropout=cfg["training"]["dropout"], **widths)


def crop(x: torch.Tensor, offsets: torch.Tensor, window: int) -> torch.Tensor:
    """Per-trial crop ``x[i, ..., offsets[i] : offsets[i] + window]`` of ``(B, 1, C, T)``."""
    B, one, C, _ = x.shape
    idx = offsets.view(B, 1, 1, 1).to(torch.long) + torch.arange(window).view(1, 1, 1, window)
    return torch.gather(x, 3, idx.expand(B, one, C, window))


def augment(x: torch.Tensor, gen: torch.Generator, noise_std: float, channel_dropout: float) -> torch.Tensor:
    """Additive Gaussian noise and whole-channel dropout; shape unchanged."""
    if noise_std > 0:
        x = x + noise_std * torch.randn(x.shape, generator=gen, dtype=x.dtype)
    if channel_dropout > 0:
        keep = torch.rand((x.shape[0], 1, x.shape[2], 1), generator=gen) >= channel_dropout
        x = x * keep.to(x.dtype)
    return x


@torch.no_grad()
def teacher_probs(members, x: torch.Tensor, T: float) -> torch.Tensor:
    """Ensemble soft labels ``mean_m softmax(z_m / T)`` for ``x``."""
    return torch.stack([F.softmax(m(x) / T, dim=1) for m in members]).mean(0)


def kd_loss(student_logits: torch.Tensor, teacher_p: torch.Tensor, y: torch.Tensor, alpha: float,
            T: float) -> torch.Tensor:
    """Mean over trials of ``(1 - alpha) CE(z, y) + alpha T^2 KL(p_T || softmax(z / T))``;
    trials with ``y < 0`` (unlabelled calibration trials) get the KD term only."""
    log_ps = F.log_softmax(student_logits / T, dim=1)
    kl = (torch.xlogy(teacher_p, teacher_p) - teacher_p * log_ps).sum(1)
    labelled = (y >= 0).to(student_logits.dtype)
    ce = F.cross_entropy(student_logits, y.clamp_min(0), reduction="none") * labelled
    return ((1.0 - alpha) * ce + alpha * T * T * kl).mean()


@torch.no_grad()
def teacher_probs_batched(members, X: torch.Tensor, T: float, device: str = "cpu", batch_size: int = 256):
    return torch.cat([teacher_probs(members, X[i : i + batch_size].to(device), T).cpu()
                      for i in range(0, len(X), batch_size)])


def offset_range(train_window: int, window: int, crop_offset: int, crop_jitter: int | None) -> tuple[int, int]:
    """Inclusive ``(lo, hi)`` of the random crop offset: ``crop_offset ± crop_jitter`` clipped to
    ``[0, train_window - window]``, or that whole range when ``crop_jitter`` is None."""
    max_offset = train_window - window
    if max_offset < 0:
        raise ValueError(f"training windows ({train_window}) shorter than the model window ({window})")
    if crop_jitter is None:
        return 0, max_offset
    return max(0, crop_offset - crop_jitter), min(max_offset, crop_offset + crop_jitter)


def validate(student: nn.Module, X_val: torch.Tensor, p_val: torch.Tensor, y_val: torch.Tensor, alpha: float,
             T: float, device: str = "cpu") -> dict:
    """Validation metrics on the canonical, unaugmented windows. ``val_kl`` = mean
    ``KL(p_T || softmax(z / T))`` (no ``T^2``) is the early-stopping and selection criterion;
    ``val_loss`` is the combined training objective, logged only."""
    val_logits = torch.from_numpy(predict(student, X_val, device=device))
    return {
        "val_loss": kd_loss(val_logits, p_val, y_val, alpha, T).item(),
        "val_kl": kd_loss(val_logits, p_val, y_val, 1.0, T).item() / (T * T),
        "val_ce": F.cross_entropy(val_logits, y_val).item(),
        "val_acc": (val_logits.argmax(1) == y_val).float().mean().item(),
        "val_teacher_agreement": (val_logits.argmax(1) == p_val.argmax(1)).float().mean().item(),
    }


def distill_fit(student: nn.Module, members, X_train: torch.Tensor, y_train, X_calib: torch.Tensor,
                X_val: torch.Tensor, y_val, *, window: int, crop_offset: int, alpha: float, T: float,
                noise_std: float, channel_dropout: float, lr: float, batch_size: int, max_epochs: int,
                patience: int, seed: int, crop_jitter: int | None = None, device: str = "cpu", log=print) -> dict:
    """KD training loop; ``X_train`` ``(n, 1, C, train_window)``, ``X_calib``/``X_val``
    ``(m, 1, C, window)``, all scaled. Early stopping and the restored weights follow the
    validation KL (``validate``); returns fit info like ``src.train.fit``."""
    members = list(members)
    student.to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    lo, hi = offset_range(X_train.shape[-1], window, crop_offset, crop_jitter)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    yv = torch.as_tensor(y_val, dtype=torch.long)
    n_tr, n_cal = len(X_train), len(X_calib)
    X_val_c = X_val if X_val.shape[-1] == window else crop(X_val, torch.full((len(X_val),), crop_offset), window)
    p_val = teacher_probs_batched(members, X_val_c, T, device)
    best = {"kl": float("inf"), "epoch": -1, "state": None, "acc": 0.0, "loss": float("nan")}
    history = []
    t0 = time.time()
    for ep in range(max_epochs):
        student.train()
        perm = torch.randperm(n_tr + n_cal, generator=gen)
        tot, correct, n_lab = 0.0, 0, 0
        for i in range(0, n_tr + n_cal, batch_size):
            idx = perm[i : i + batch_size]
            tr_idx, cal_idx = idx[idx < n_tr], idx[idx >= n_tr] - n_tr
            parts, ys = [], []
            if len(tr_idx):
                off = torch.randint(lo, hi + 1, (len(tr_idx),), generator=gen)
                parts.append(crop(X_train[tr_idx], off, window)); ys.append(yt[tr_idx])
            if len(cal_idx):
                parts.append(X_calib[cal_idx]); ys.append(torch.full((len(cal_idx),), -1, dtype=torch.long))
            xb = augment(torch.cat(parts), gen, noise_std, channel_dropout).to(device)
            yb = torch.cat(ys).to(device)
            p_t = teacher_probs(members, xb, T)
            opt.zero_grad(set_to_none=True)
            logits = student(xb)
            loss = kd_loss(logits, p_t, yb, alpha, T)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            lab = yb >= 0
            correct += (logits.argmax(1)[lab] == yb[lab]).sum().item()
            n_lab += int(lab.sum().item())
        v = validate(student, X_val_c, p_val, yv, alpha, T, device)
        n_seen = n_tr + n_cal
        history.append({"epoch": ep, "train_loss": tot / n_seen, "train_acc": correct / max(n_lab, 1), **v})
        if v["val_kl"] < best["kl"]:
            best = {"kl": v["val_kl"], "epoch": ep, "acc": v["val_acc"], "loss": v["val_loss"],
                    "state": {k: t.detach().cpu().clone() for k, t in student.state_dict().items()}}
        if ep % 10 == 0 or ep == max_epochs - 1:
            log(f"epoch {ep:3d} train loss {tot / n_seen:.4f} acc {correct / max(n_lab, 1):.3f} | val kl "
                f"{v['val_kl']:.4f} loss {v['val_loss']:.4f} acc {v['val_acc']:.3f} agree "
                f"{v['val_teacher_agreement']:.3f} | best ep {best['epoch']} ({time.time() - t0:.0f}s)")
        if ep - best["epoch"] >= patience:
            log(f"early stop at epoch {ep}, best epoch {best['epoch']} val kl {best['kl']:.4f}")
            break
    student.load_state_dict(best["state"])
    student.cpu().eval()
    return {"best_epoch": best["epoch"], "best_val_kl": best["kl"], "best_val_loss": best["loss"],
            "best_val_acc": best["acc"], "epochs_run": len(history), "fit_time_s": time.time() - t0,
            "history": history}


# ----------------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> dict:
    cfg = load_v2_config(args.config)
    if args.subset_cho:
        cfg = subset_cho(cfg, args.subset_cho)
    tr, kd = cfg["training"], cfg["distill"]
    lr = args.lr if args.lr is not None else tr["lr"]
    batch_size = args.batch_size or tr["batch_size"]
    max_epochs = args.epochs or tr["max_epochs"]
    patience = args.patience or tr["patience"]
    window, offset = int(cfg["window"]), int(cfg["crop_offset"])
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train.log", "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    t_all = time.time()
    set_seed(args.seed)
    data = load_v2_data(args.seed, cfg)
    scale = float(data["input_scale"])
    teacher_params = json.loads((Path(args.teacher) / "params.json").read_text())
    if not np.isclose(teacher_params["input_scale"], scale, rtol=1e-5):
        log(f"WARNING: teacher input_scale {teacher_params['input_scale']:.6f} != data input_scale {scale:.6f} "
            "(teacher trained on different data or seed)")
    subjects = sorted(data["test"])
    X_train = to_model_input(data["train"][0], scale)
    X_val = to_model_input(canonical(data["val"][0], cfg), scale)
    X_calib = torch.cat([to_model_input(data["calib"][s], scale) for s in subjects])
    n_target = 0
    if kd.get("target_unlabeled"):
        # v4: the unlabelled target session (2a session 1, X only) joins the calibration trials as
        # extra KD-only windows (canonical 256 samples, no random crop, label -1).
        target = load_target_unlabeled(cfg, scale)
        X_tgt = torch.cat([torch.from_numpy(np.ascontiguousarray(target["X"][s][:, None], dtype=np.float32))
                           for s in subjects])
        n_target = len(X_tgt)
        X_calib = torch.cat([X_calib, X_tgt])
    X_test = {s: to_model_input(data["test"][s][0], scale) for s in subjects}
    teacher_round = getattr(args, "teacher_round", None)
    members = load_members(args.teacher, cfg, args.device,
                           subdir="members_round0" if teacher_round == 0 else "members")
    if teacher_round == 0:
        log(f"teacher = round-0 members ({Path(args.teacher) / 'members_round0'}), before self-training")
    set_seed(args.seed)
    student = build_student(args.student, cfg)
    n_params = sum(p.numel() for p in student.parameters())
    log(f"distill {args.student} seed {args.seed} ({n_params} params) from {args.teacher}: train "
        f"{tuple(X_train.shape)} calib {tuple(X_calib.shape)} (of which {n_target} unlabelled target) val {tuple(X_val.shape)} test subjects {subjects} "
        f"input_scale {scale:.4f} alpha {kd['alpha']} T {kd['temperature']} noise {kd.get('noise_std')} "
        f"channel dropout {kd.get('channel_dropout')} threads {args.threads} device {args.device}")

    if tr.get("loop", "v2") == "v3":
        from src.augment import Augmenter, mixing_matrices
        from src.train_v3 import distill_fit_v3

        augmenter = Augmenter(cfg.get("augment"), window=window, crop_offset=offset, train_window=X_train.shape[-1],
                              mix=mixing_matrices(data["ea_train"]))
        log(f"v3 loop: weight decay {kd.get('weight_decay', 0.0)} ema {kd.get('ema', 0.0)} warmup "
            f"{kd.get('warmup', 0)} mixup alpha {kd.get('mixup_alpha', 0.0)} p {kd.get('mixup_p', 0.0)} augment "
            f"{augmenter.cfg} (ea_mix_p used {augmenter.ea_mix_p})")
        info = distill_fit_v3(student, members.values(), X_train, data["train"][1], data["train_group"], X_calib,
                              X_val, data["val"][1], augmenter=augmenter, alpha=float(kd["alpha"]),
                              T=float(kd["temperature"]), lr=lr, weight_decay=float(kd.get("weight_decay", 0.0)),
                              ema_decay=float(kd.get("ema", 0.0)), warmup_epochs=int(kd.get("warmup", 0)),
                              mixup_alpha=float(kd.get("mixup_alpha", 0.0)), mixup_p=float(kd.get("mixup_p", 0.0)),
                              batch_size=batch_size, max_epochs=max_epochs, patience=patience, seed=args.seed,
                              device=args.device, log=log)
    else:
        info = distill_fit(student, members.values(), X_train, data["train"][1], X_calib, X_val, data["val"][1],
                           window=window, crop_offset=offset, alpha=float(kd["alpha"]), T=float(kd["temperature"]),
                           noise_std=float(kd["noise_std"]), channel_dropout=float(kd["channel_dropout"]), lr=lr,
                           batch_size=batch_size, max_epochs=max_epochs, patience=patience, seed=args.seed,
                           crop_jitter=None if kd.get("crop_jitter") is None else int(kd["crop_jitter"]),
                           device=args.device, log=log)

    logits = {s: predict(student, X_test[s]) for s in subjects}
    arrays = {"subjects": np.array(subjects)}
    for s in subjects:
        arrays[f"y_true_{s}"] = data["test"][s][1]
        arrays[f"logits_{s}"] = logits[s]
    np.savez(out / "predictions.npz", **arrays)
    torch.save(student.state_dict(), out / "model.pt")
    widths = {k: int(cfg["student"][args.student][k]) for k in STUDENT_KEYS}
    (out / "student.json").write_text(json.dumps(widths, indent=1) + "\n")

    # 200 real scaled canonical training windows for post-training quantization calibration.
    X_train_c = canonical(X_train.numpy(), cfg)
    rng = np.random.default_rng(args.seed)
    calib_idx = np.sort(rng.choice(len(X_train_c), size=min(N_CALIB, len(X_train_c)), replace=False))
    np.savez(out / "calibration.npz", x=np.ascontiguousarray(X_train_c[calib_idx], dtype=np.float32))
    write_params(out, data, cfg, {"seed": args.seed, "student": {"name": args.student, **widths},
                                  "teacher": str(args.teacher), "teacher_members": list(member_names(cfg)),
                                  **({"teacher_round": 0} if teacher_round == 0 else {})})

    per = {s: compute_kpis(data["test"][s][1], logits[s]) for s in subjects}
    metrics = {
        "model": "tinyeegnet", "student": args.student, "widths": widths, "seed": args.seed,
        "teacher": str(args.teacher), "params": n_params, "input_scale": scale,
        "n_train": int(len(X_train)), "n_calib_kd": int(len(X_calib)), "n_val": int(len(X_val)),
        **({"n_target_kd": n_target} if kd.get("target_unlabeled") else {}),
        "n_test": int(sum(len(X_test[s]) for s in subjects)),
        "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
        "distill": kd, "threads": args.threads, "device": args.device,
        **{k: v for k, v in info.items() if k != "history"},
        "test": {str(s): k for s, k in per.items()},
        "test_mean": {k: float(np.mean([per[s][k] for s in subjects])) for k in KPI_KEYS},
        "confusion": {str(s): confusion(data["test"][s][1], logits[s]).tolist() for s in subjects},
        "meta": jsonable(data.get("meta", {})),
        "versions": {"torch": torch.__version__, "python": platform.python_version()},
    }
    metrics["wall_time_s"] = time.time() - t_all
    with open(out / "metrics.json", "w") as f:
        json.dump(jsonable(metrics), f, indent=1)
    with open(out / "history.json", "w") as f:
        json.dump(info["history"], f)
    log("test mean: " + " ".join(f"{k} {v:.4f}" for k, v in metrics["test_mean"].items())
        + f" | best epoch {info['best_epoch']} val kl {info['best_val_kl']:.4f} | wall {metrics['wall_time_s']:.0f}s")
    log_f.close()
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--student", required=True, help="key of cfg['student'] (wide32, control, wide64)")
    p.add_argument("--teacher", required=True, help="teacher run directory (members/*.pt, params.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--config", type=Path, default=CONFIG_V2)
    p.add_argument("--device", default="cpu", help="cpu or mps")
    p.add_argument("--threads", type=int, default=int(os.environ.get("RB_THREADS", "1")))
    p.add_argument("--subset-cho", type=int, default=None, help="keep only the first N Cho2017 subjects (smoke)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--teacher-round", type=int, choices=(0,), default=None,
                   help="v4: distil from the teacher's round-0 members (members_round0/, before self-training)")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
