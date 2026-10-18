# SPDX-License-Identifier: Apache-2.0
"""v4soft: soft-target, per-window weighted variant of ``src.train.fit`` for the self-training round.

Every training window ``i`` has a target distribution ``p_i`` (one-hot for the labelled
Cho2017 / Lee2019_MI windows, the round-0 ensemble probability for the unlabelled 2a session-1
windows) and a weight ``w_i`` (1 for labelled windows, ``2 |p_i(right) - 0.5|`` for session 1).
Per batch the loss is ``sum_i w_i KL(p_i || softmax(z_i)) / B``; for a one-hot ``p_i`` the KL
term equals the cross-entropy, so labelled windows train exactly as in ``src.train.fit``.
Everything else is ``src.train.fit``: Adam, the same shuffling generator, max-norm hook, early
stopping on the (hard-label) validation cross-entropy, best weights restored, same history keys.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.train import predict


def soft_kl_loss(logits: torch.Tensor, target_p: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``sum_i w_i KL(p_i || softmax(z_i)) / B`` (``xlogy`` so one-hot targets give the CE)."""
    log_q = F.log_softmax(logits, dim=1)
    kl = (torch.xlogy(target_p, target_p) - target_p * log_q).sum(1)
    return (weight * kl).sum() / len(logits)


def target_weights(p: np.ndarray) -> np.ndarray:
    """``w = 2 |p(right) - 0.5|`` per window of ``(n, 2)`` probabilities (0 = undecided, 1 = certain)."""
    p = np.asarray(p, dtype=np.float64)
    return 2.0 * np.abs(p[:, 1] - 0.5)


def fit_soft(model: nn.Module, X_train, p_train, w_train, X_val, y_val, *, lr: float, batch_size: int,
             max_epochs: int, patience: int, seed: int, device: str = "cpu", log=print) -> dict:
    """``src.train.fit`` with soft targets ``p_train`` ``(n, 2)`` and weights ``w_train`` ``(n,)``."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    pt = torch.as_tensor(np.asarray(p_train, dtype=np.float32))
    wt = torch.as_tensor(np.asarray(w_train, dtype=np.float32))
    yt = pt.argmax(1)
    yv = torch.as_tensor(y_val, dtype=torch.long)
    best = {"loss": float("inf"), "epoch": -1, "state": None, "acc": 0.0}
    history = []
    t0 = time.time()
    n = len(X_train)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, pb, wb = X_train[idx].to(device), pt[idx].to(device), wt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = soft_kl_loss(logits, pb, wb)
            loss.backward()
            opt.step()
            if hasattr(model, "apply_max_norm"):
                model.apply_max_norm()
            tot += loss.item() * len(idx)
            correct += (logits.argmax(1).cpu() == yt[idx]).sum().item()
        val_logits = torch.from_numpy(predict(model, X_val, device=device))
        val_loss = loss_fn(val_logits, yv).item()
        val_acc = (val_logits.argmax(1) == yv).float().mean().item()
        history.append({"epoch": ep, "train_loss": tot / n, "train_acc": correct / n,
                        "val_loss": val_loss, "val_acc": val_acc})
        if val_loss < best["loss"]:
            best = {"loss": val_loss, "epoch": ep, "acc": val_acc,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if ep % 10 == 0 or ep == max_epochs - 1:
            log(f"epoch {ep:3d} train loss {tot / n:.4f} acc {correct / n:.3f} | val loss {val_loss:.4f} "
                f"acc {val_acc:.3f} | best ep {best['epoch']} ({time.time() - t0:.0f}s)")
        if ep - best["epoch"] >= patience:
            log(f"early stop at epoch {ep}, best epoch {best['epoch']} val loss {best['loss']:.4f}")
            break
    model.load_state_dict(best["state"])
    model.eval()
    return {"best_epoch": best["epoch"], "best_val_loss": best["loss"], "best_val_acc": best["acc"],
            "epochs_run": len(history), "fit_time_s": time.time() - t0, "history": history}


__all__ = ["fit_soft", "soft_kl_loss", "target_weights"]
