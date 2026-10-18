# SPDX-License-Identifier: Apache-2.0
"""v3 training loops (teachers and student), selected by ``training.loop: v3`` in the config.

Shared by ``src.train_teacher`` and ``src.distill``; the v2 loops (``src.train.fit``,
``src.distill.distill_fit``) are untouched and stay the default.

- AdamW (``weight_decay``), per-step cosine learning rate with linear warm-up
  (``warmup`` epochs), EMA of the weights and BatchNorm buffers (``ema`` decay, with the usual
  ``min(decay, (1 + k) / (10 + k))`` ramp so the first steps are not dominated by the
  initialisation). The EMA copy is validated every epoch, selected for early stopping and
  returned as the trained model.
- Augmentation on the fly with :class:`src.augment.Augmenter` (crop, EA-matrix mixing,
  amplitude jitter, noise, channel dropout).
- Teachers: cross-entropy with ``label_smoothing``; validation loss = plain cross-entropy of
  the EMA model on the canonical, unaugmented validation windows.
- Student: mixup (``augment.mixup_*``) under KD: the teacher ensemble scores the mixed,
  augmented batch; the CE term uses the mixed labels; early stopping on the validation KL.
"""

from __future__ import annotations

import copy
import math
import time

import torch
import torch.nn.functional as F
from torch import nn

from src.augment import Augmenter, mixup, mixup_ce
from src.train import predict


class EMA:
    """Exponential moving average of a model's parameters and float buffers."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.steps = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        src = model.state_dict()
        for k, v in self.model.state_dict().items():
            s = src[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(d).add_(s.to(v.dtype), alpha=1 - d)
            else:
                v.copy_(s)


def cosine_warmup(opt: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """Per-step LR multiplier: linear warm-up to 1, then cosine to 0 at ``total_steps``."""

    def f(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def _state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit_v3(model: nn.Module, X_train: torch.Tensor, y_train, groups, X_val: torch.Tensor, y_val, *,
           augmenter: Augmenter, lr: float, weight_decay: float, label_smoothing: float, ema_decay: float,
           warmup_epochs: int, batch_size: int, max_epochs: int, patience: int, seed: int,
           device: str = "cpu", log=print) -> dict:
    """Teacher loop. ``X_train`` ``(n, 1, C, train_window)`` scaled, ``groups`` ``(n,)`` training
    subject-session index (EA mixing), ``X_val`` canonical ``(m, 1, C, window)``. Returns fit info
    like ``src.train.fit``; ``model`` holds the best EMA weights afterwards."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(X_train)
    steps_per_epoch = math.ceil(n / batch_size)
    sched = cosine_warmup(opt, warmup_epochs * steps_per_epoch, max_epochs * steps_per_epoch)
    ema = EMA(model, ema_decay) if ema_decay else None
    gen = torch.Generator().manual_seed(seed)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    yv = torch.as_tensor(y_val, dtype=torch.long)
    g = torch.as_tensor(groups, dtype=torch.long)
    best = {"loss": float("inf"), "epoch": -1, "state": None, "acc": 0.0}
    history = []
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = augmenter(X_train[idx], g[idx], gen).to(device)
            yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)
            loss.backward()
            opt.step()
            sched.step()
            if hasattr(model, "apply_max_norm"):
                model.apply_max_norm()
            if ema is not None:
                ema.update(model)
            tot += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        evm = ema.model if ema is not None else model
        val_logits = torch.from_numpy(predict(evm, X_val, device=device))
        val_loss = F.cross_entropy(val_logits, yv).item()
        val_acc = (val_logits.argmax(1) == yv).float().mean().item()
        history.append({"epoch": ep, "train_loss": tot / n, "train_acc": correct / n, "val_loss": val_loss,
                        "val_acc": val_acc, "lr": sched.get_last_lr()[0]})
        if val_loss < best["loss"]:
            best = {"loss": val_loss, "epoch": ep, "acc": val_acc, "state": _state(evm)}
        if ep % 10 == 0 or ep == max_epochs - 1:
            log(f"epoch {ep:3d} train loss {tot / n:.4f} acc {correct / n:.3f} | val loss {val_loss:.4f} "
                f"acc {val_acc:.3f} | lr {sched.get_last_lr()[0]:.2e} | best ep {best['epoch']} "
                f"({time.time() - t0:.0f}s)")
        if ep - best["epoch"] >= patience:
            log(f"early stop at epoch {ep}, best epoch {best['epoch']} val loss {best['loss']:.4f}")
            break
    model.load_state_dict(best["state"])
    model.eval()
    return {"best_epoch": best["epoch"], "best_val_loss": best["loss"], "best_val_acc": best["acc"],
            "epochs_run": len(history), "fit_time_s": time.time() - t0, "ema": ema_decay,
            "history": history}


def distill_fit_v3(student: nn.Module, members, X_train: torch.Tensor, y_train, groups, X_calib: torch.Tensor,
                   X_val: torch.Tensor, y_val, *, augmenter: Augmenter, alpha: float, T: float, lr: float,
                   weight_decay: float, ema_decay: float, warmup_epochs: int, mixup_alpha: float, mixup_p: float,
                   batch_size: int, max_epochs: int, patience: int, seed: int, device: str = "cpu",
                   log=print) -> dict:
    """Student KD loop (v3). Same inputs as ``src.distill.distill_fit`` plus ``groups``;
    returns the same fit info keys. ``student`` holds the best EMA weights afterwards."""
    from src.distill import kd_loss, teacher_probs, teacher_probs_batched, validate

    members = list(members)
    student.to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    yv = torch.as_tensor(y_val, dtype=torch.long)
    g = torch.as_tensor(groups, dtype=torch.long)
    n_tr, n_cal = len(X_train), len(X_calib)
    steps_per_epoch = math.ceil((n_tr + n_cal) / batch_size)
    sched = cosine_warmup(opt, warmup_epochs * steps_per_epoch, max_epochs * steps_per_epoch)
    ema = EMA(student, ema_decay) if ema_decay else None
    gen = torch.Generator().manual_seed(seed)
    p_val = teacher_probs_batched(members, X_val, T, device)
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
                parts.append(augmenter(X_train[tr_idx], g[tr_idx], gen)); ys.append(yt[tr_idx])
            if len(cal_idx):
                parts.append(augmenter(X_calib[cal_idx], None, gen))
                ys.append(torch.full((len(cal_idx),), -1, dtype=torch.long))
            xb, yb = torch.cat(parts), torch.cat(ys)
            xb, ya, yb2, lam = mixup(xb, yb, gen, mixup_alpha, mixup_p)
            xb, ya, yb2 = xb.to(device), ya.to(device), yb2.to(device)
            p_t = teacher_probs(members, xb, T)
            opt.zero_grad(set_to_none=True)
            logits = student(xb)
            kd = kd_loss(logits, p_t, ya, 1.0, T)  # alpha 1: KD term only (mean over all trials)
            ce = mixup_ce(logits, ya, yb2, lam).mean()
            loss = (1.0 - alpha) * ce + alpha * kd
            loss.backward()
            opt.step()
            sched.step()
            if ema is not None:
                ema.update(student)
            tot += loss.item() * len(idx)
            lab = ya >= 0
            correct += (logits.argmax(1)[lab] == ya[lab]).sum().item()
            n_lab += int(lab.sum().item())
        evm = ema.model if ema is not None else student
        v = validate(evm, X_val, p_val, yv, alpha, T, device)
        n_seen = n_tr + n_cal
        history.append({"epoch": ep, "train_loss": tot / n_seen, "train_acc": correct / max(n_lab, 1),
                        "lr": sched.get_last_lr()[0], **v})
        if v["val_kl"] < best["kl"]:
            best = {"kl": v["val_kl"], "epoch": ep, "acc": v["val_acc"], "loss": v["val_loss"], "state": _state(evm)}
        if ep % 10 == 0 or ep == max_epochs - 1:
            log(f"epoch {ep:3d} train loss {tot / n_seen:.4f} acc {correct / max(n_lab, 1):.3f} | val kl "
                f"{v['val_kl']:.4f} loss {v['val_loss']:.4f} acc {v['val_acc']:.3f} agree "
                f"{v['val_teacher_agreement']:.3f} | lr {sched.get_last_lr()[0]:.2e} | best ep {best['epoch']} "
                f"({time.time() - t0:.0f}s)")
        if ep - best["epoch"] >= patience:
            log(f"early stop at epoch {ep}, best epoch {best['epoch']} val kl {best['kl']:.4f}")
            break
    student.load_state_dict(best["state"])
    student.cpu().eval()
    return {"best_epoch": best["epoch"], "best_val_kl": best["kl"], "best_val_loss": best["loss"],
            "best_val_acc": best["acc"], "epochs_run": len(history), "fit_time_s": time.time() - t0,
            "history": history}


__all__ = ["EMA", "cosine_warmup", "fit_v3", "distill_fit_v3"]
