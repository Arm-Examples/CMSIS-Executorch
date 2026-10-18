"""Train TinyEEGNet (or the EEGNet-ELU reference) in LOSO or per-subject mode.

    python -m src.train --mode loso --subject 1 --seed 0 --out runs/loso_s1_seed0
    python -m src.train --mode per-subject --subject 1 --seed 0 --out runs/per_subject_s1_seed0

Writes to ``--out``: ``model.pt`` (state_dict), ``metrics.json``, ``predictions.npz``
(``y_true``, ``logits`` on the target's session 2 trials 25-144), ``calibration.npz``
(``x``: 200 aligned+scaled training windows for PTQ), ``params.json`` (input scale,
target EA matrix, filter/epoch settings) and, in LOSO mode, ``predictions_calib_s1.npz``
(same model, EA matrix from the target's session 1 instead of the calibration block).

One torch thread per process is ~8x faster than four for this tiny model on Apple
silicon (thread-pool overhead dominates), so ``--threads`` defaults to 1 and
``src.run_all`` parallelises across processes instead.

Early stopping monitors the validation loss only (LOSO: stratified 10 % of every source
subject-session; per-subject: 20 % of the target's session 1). The test block is never
looked at during training.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from src.data.download import CUE_OFFSET_S, FS
from src.data.preprocess import design_filter
from src.data.splits import CALIB_TRIALS, cached_subject, loso_data, per_subject_data, target_calibration
from src.eval import compute_kpis, confusion
from src.models.eegnet import EEGNet
from src.models.tinyeegnet import TinyEEGNet

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
CHANNELS = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]


def load_config(path: Path = CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_model_input(X: np.ndarray, input_scale: float) -> torch.Tensor:
    """``(n, 22, 500)`` aligned µV -> ``(n, 1, 22, 500)`` float32 model input."""
    return torch.from_numpy(np.asarray(X, dtype=np.float32)[:, None, :, :] / np.float32(input_scale))


def build_model(name: str, dropout: float) -> nn.Module:
    if name == "tinyeegnet":
        return TinyEEGNet(dropout=dropout)
    if name == "eegnet":
        return EEGNet(dropout=dropout)
    raise ValueError(name)


def get_data(mode: str, subject: int, seed: int, calib_source: str = "session2", val_fraction: float = 0.2) -> dict:
    if mode == "loso":
        return loso_data(subject, seed=seed, calib_source=calib_source, loader=cached_subject)
    if mode == "per-subject":
        return per_subject_data(subject, seed=seed, val_fraction=val_fraction, calib_source=calib_source,
                                loader=cached_subject)
    raise ValueError(mode)


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    model.eval()
    out = [model(X[i : i + batch_size].to(device)).cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy().astype(np.float32)


def fit(model: nn.Module, X_train, y_train, X_val, y_val, *, lr: float, batch_size: int, max_epochs: int,
        patience: int, seed: int, device: str = "cpu", log=print) -> dict:
    """Adam + cross-entropy, early stopping on validation loss, best weights restored."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    yt = torch.as_tensor(y_train, dtype=torch.long)
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
            xb, yb = X_train[idx].to(device), yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            if hasattr(model, "apply_max_norm"):
                model.apply_max_norm()
            tot += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
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


def write_params(out: Path, data: dict, cfg: dict) -> dict:
    sos = design_filter(cfg["fs"], cfg["filter"]["low"], cfg["filter"]["high"], cfg["filter"]["order"])
    params = {
        "fs": cfg["fs"],
        "n_channels": cfg["n_channels"],
        "channels": CHANNELS,
        "window_samples": int(round((cfg["window"]["tmax"] - cfg["window"]["tmin"]) * cfg["fs"])),
        "cue_offset_s": CUE_OFFSET_S,
        "tmin_s": cfg["window"]["tmin"],
        "tmax_s": cfg["window"]["tmax"],
        "filter": {"type": "butterworth_bandpass_sos", "low_hz": cfg["filter"]["low"],
                   "high_hz": cfg["filter"]["high"], "order": cfg["filter"]["order"],
                   "sos": sos.tolist(), "causal": True},
        "units": "uV",
        "ea_matrix": np.asarray(data["A_target"]).tolist(),
        "ea_calibration": {"source": data["calib_source"], "trials": CALIB_TRIALS},
        "input_scale": float(data["input_scale"]),
        "class_map": {"0": "left_hand (769)", "1": "right_hand (770)"},
        "val_subject": data["val_subject"],
    }
    with open(out / "params.json", "w") as f:
        json.dump(params, f, indent=1)
    return params


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    tr = cfg["training"]
    lr = args.lr if args.lr is not None else tr["lr"]
    batch_size = args.batch_size or tr["batch_size"]
    max_epochs = args.epochs or tr["max_epochs"]
    patience = args.patience or tr["patience"]
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train.log", "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    t_all = time.time()
    data = get_data(args.mode, args.subject, args.seed, "session2", tr["val_fraction"])
    scale = data["input_scale"]
    X_train = to_model_input(data["train"][0], scale)
    X_val = to_model_input(data["val"][0], scale)
    X_test = to_model_input(data["test"][0], scale)
    log(f"{args.mode} subject {args.subject} seed {args.seed} model {args.model}: train {tuple(X_train.shape)} "
        f"val {tuple(X_val.shape)} (val {'10 % of each source subject-session' if data['val_subject'] is None else 'subject %d' % data['val_subject']}) test {tuple(X_test.shape)} "
        f"input_scale {scale:.4f} threads {args.threads}")

    model = build_model(args.model, tr["dropout"])
    fit_info = fit(model, X_train, data["train"][1], X_val, data["val"][1], lr=lr, batch_size=batch_size,
                   max_epochs=max_epochs, patience=patience, seed=args.seed, device=args.device, log=log)

    logits = predict(model, X_test, device=args.device)
    y_test = data["test"][1]
    kpis = compute_kpis(y_test, logits)
    cm = confusion(y_test, logits)
    np.savez(out / "predictions.npz", y_true=y_test, logits=logits, test_trials=data["test_trials"])
    torch.save(model.state_dict(), out / "model.pt")

    # 200 real aligned+scaled training windows for post-training quantization calibration.
    rng = np.random.default_rng(args.seed)
    calib_idx = rng.choice(len(X_train), size=min(200, len(X_train)), replace=False)
    np.savez(out / "calibration.npz", x=X_train[np.sort(calib_idx)].numpy())
    write_params(out, data, cfg)

    metrics = {
        "mode": args.mode, "subject": args.subject, "seed": args.seed, "model": args.model,
        "val_subject": data["val_subject"], "input_scale": scale,
        "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
        "threads": args.threads, "device": args.device,
        "best_epoch": fit_info["best_epoch"], "epochs_run": fit_info["epochs_run"],
        "best_val_loss": fit_info["best_val_loss"], "best_val_acc": fit_info["best_val_acc"],
        "fit_time_s": fit_info["fit_time_s"],
        "test": kpis, "confusion": cm.tolist(),
        "versions": {"torch": torch.__version__, "python": platform.python_version()},
    }
    if args.mode == "loso":
        # EA calibration variant: A from the target's session 1 (labels unused), same model.
        A1, _, _ = target_calibration(cached_subject(args.subject), "session1")
        from src.data.align import apply_ea

        X2 = cached_subject(args.subject)[2][0][data["test_trials"]]
        logits_s1 = predict(model, to_model_input(apply_ea(X2, A1), scale), device=args.device)
        np.savez(out / "predictions_calib_s1.npz", y_true=y_test, logits=logits_s1, test_trials=data["test_trials"])
        metrics["test_calib_session1"] = compute_kpis(y_test, logits_s1)
    metrics["wall_time_s"] = time.time() - t_all
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    with open(out / "history.json", "w") as f:
        json.dump(fit_info["history"], f)
    log(f"test: " + " ".join(f"{k} {v:.4f}" for k, v in kpis.items())
        + f" | confusion {cm.tolist()} | wall {metrics['wall_time_s']:.0f}s")
    log_f.close()
    return metrics


def evaluate_checkpoint(checkpoint: Path, mode: str, subject: int, seed: int, model_name: str = "tinyeegnet") -> dict:
    cfg = load_config()
    data = get_data(mode, subject, seed, "session2", cfg["training"]["val_fraction"])
    model = build_model(model_name, cfg["training"]["dropout"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    logits = predict(model, to_model_input(data["test"][0], data["input_scale"]))
    return compute_kpis(data["test"][1], logits)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["loso", "per-subject"], required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=["tinyeegnet", "eegnet"], default="tinyeegnet")
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--threads", type=int, default=int(os.environ.get("RB_THREADS", "1")))
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
