# SPDX-License-Identifier: Apache-2.0
"""v4soft ablation: KD-only fine-tune of a trained student on the unlabelled 2a windows.

    python -m src.finetune_student --config configs/v4soft.yaml --init runs/v2/student_wide32_seed0 \\
        --teacher runs/v4soft/teacher_seed0 --out runs/v4soft/student_wide32_ft_seed0 --device mps

Starts from ``--init``'s ``model.pt`` (its ``student.json`` widths). The stream is the unlabelled
2a session-1 windows (``splits125.target_unlabeled``, X only, 1,296) plus the session-2
calibration trials 1-24 (216), all 256-sample canonical windows, no labels. Per batch: the v2
augmentation (``distill.noise_std``, ``distill.channel_dropout``), the teacher ensemble scores the
same augmented batch, loss = ``T^2 KL(p_T || softmax(z / T))`` (``distill.kd_loss`` with alpha 1).
Adam ``--lr`` (1e-4) for ``--epochs`` (30) epochs, **no early stopping: the last epoch is kept**;
the validation KL (training-source validation split) is logged only. The 2a test block (session 2
trials 25-144) is only scored at the end.

Writes the same files as ``src.distill`` (``model.pt``, ``student.json``, ``predictions.npz``,
``calibration.npz`` = 200 canonical training windows for PTQ, ``params.json``, ``metrics.json``,
``history.json``, ``train.log``) so ``src.quantize`` and ``src.run_v2`` treat it as a student run.
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

from src.distill import N_CALIB, augment, kd_loss, teacher_probs, teacher_probs_batched
from src.eval import KPI_KEYS, compute_kpis, confusion
from src.models.tinyeegnet import STUDENT_KEYS, TinyEEGNet
from src.train import predict, set_seed, to_model_input
from src.train_teacher import (CONFIG_V2, canonical, jsonable, load_members, load_target_unlabeled, load_v2_config,
                               load_v2_data, member_names, write_params)


def finetune(student, members, X_unl: torch.Tensor, X_val: torch.Tensor, *, T: float, noise_std: float,
             channel_dropout: float, lr: float, batch_size: int, epochs: int, seed: int, device: str = "cpu",
             log=print) -> dict:
    """KD-only fine-tune on ``X_unl`` ``(n, 1, C, W)``; keeps the last epoch."""
    members = list(members)
    student.to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    p_val = teacher_probs_batched(members, X_val, T, device)
    history = []
    t0 = time.time()
    n = len(X_unl)
    for ep in range(epochs):
        student.train()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = augment(X_unl[idx], gen, noise_std, channel_dropout).to(device)
            p_t = teacher_probs(members, xb, T)
            opt.zero_grad(set_to_none=True)
            loss = kd_loss(student(xb), p_t, torch.full((len(idx),), -1, device=device), 1.0, T)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        val_logits = torch.from_numpy(predict(student, X_val, device=device))
        val_kl = kd_loss(val_logits, p_val, torch.full((len(X_val),), -1), 1.0, T).item() / (T * T)
        agree = (val_logits.argmax(1) == p_val.argmax(1)).float().mean().item()
        history.append({"epoch": ep, "train_loss": tot / n, "val_kl": val_kl, "val_teacher_agreement": agree})
        log(f"epoch {ep:3d} train kd {tot / n:.4f} | val kl {val_kl:.4f} agree {agree:.3f} ({time.time() - t0:.0f}s)")
    student.cpu().eval()
    return {"epochs_run": len(history), "best_epoch": len(history) - 1, "best_val_kl": history[-1]["val_kl"],
            "best_val_loss": history[-1]["val_kl"] * T * T, "best_val_acc": float("nan"),
            "selection": "last epoch (no early stopping)", "fit_time_s": time.time() - t0, "history": history}


def run(args: argparse.Namespace) -> dict:
    cfg = load_v2_config(args.config)
    kd = cfg["distill"]
    T = float(kd["temperature"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    log_f = open(out / "train.log", "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    t_all = time.time()
    set_seed(args.seed)
    data = load_v2_data(args.seed, cfg)
    scale = float(data["input_scale"])
    init = Path(args.init)
    init_params = json.loads((init / "params.json").read_text())
    if not np.isclose(init_params["input_scale"], scale, rtol=1e-5):
        raise ValueError(f"{init}: input_scale {init_params['input_scale']} != data {scale}")
    subjects = sorted(data["test"])
    target = load_target_unlabeled(cfg, scale)
    X_tgt = torch.cat([torch.from_numpy(np.ascontiguousarray(target["X"][s][:, None], dtype=np.float32))
                       for s in subjects])
    X_cal = torch.cat([to_model_input(data["calib"][s], scale) for s in subjects])
    X_unl = torch.cat([X_tgt, X_cal])
    X_val = to_model_input(canonical(data["val"][0], cfg), scale)
    X_test = {s: to_model_input(data["test"][s][0], scale) for s in subjects}
    widths = {k: int(v) for k, v in json.loads((init / "student.json").read_text()).items() if k in STUDENT_KEYS}
    name = args.student
    student = TinyEEGNet(n_samples=int(cfg["window"]), dropout=cfg["training"]["dropout"], **widths)
    student.load_state_dict(torch.load(init / "model.pt", map_location="cpu", weights_only=True))
    n_params = sum(p.numel() for p in student.parameters())
    members = load_members(args.teacher, cfg, args.device)
    log(f"fine-tune {name} ({n_params} params) from {init / 'model.pt'} with teacher {args.teacher} "
        f"({list(members)}): {len(X_tgt)} target + {len(X_cal)} calibration windows, KD only, lr {args.lr}, "
        f"{args.epochs} epochs, keep last, T {T}, noise {kd['noise_std']}, channel dropout {kd['channel_dropout']}")
    info = finetune(student, members.values(), X_unl, X_val, T=T, noise_std=float(kd["noise_std"]),
                    channel_dropout=float(kd["channel_dropout"]), lr=args.lr, batch_size=args.batch_size,
                    epochs=args.epochs, seed=args.seed, device=args.device, log=log)

    logits = {s: predict(student, X_test[s]) for s in subjects}
    arrays = {"subjects": np.array(subjects)}
    for s in subjects:
        arrays[f"y_true_{s}"], arrays[f"logits_{s}"] = data["test"][s][1], logits[s]
    np.savez(out / "predictions.npz", **arrays)
    torch.save(student.state_dict(), out / "model.pt")
    (out / "student.json").write_text(json.dumps(widths, indent=1) + "\n")
    X_train_c = canonical(to_model_input(data["train"][0], scale).numpy(), cfg)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(X_train_c), size=min(N_CALIB, len(X_train_c)), replace=False))
    np.savez(out / "calibration.npz", x=np.ascontiguousarray(X_train_c[idx], dtype=np.float32))
    write_params(out, data, cfg, {"seed": args.seed, "student": {"name": name, **widths},
                                  "teacher": str(args.teacher), "teacher_members": list(member_names(cfg)),
                                  "init": str(init / "model.pt")})
    per = {s: compute_kpis(data["test"][s][1], logits[s]) for s in subjects}
    metrics = {"model": "tinyeegnet", "student": name, "widths": widths, "seed": args.seed, "params": n_params,
               "teacher": str(args.teacher), "init": str(init / "model.pt"), "input_scale": scale,
               "n_unlabeled": int(len(X_unl)), "n_target_kd": int(len(X_tgt)), "n_calib_kd": int(len(X_cal)),
               "n_val": int(len(X_val)), "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
               "device": args.device, **{k: v for k, v in info.items() if k != "history"},
               "test": {str(s): k for s, k in per.items()},
               "test_mean": {k: float(np.mean([per[s][k] for s in subjects])) for k in KPI_KEYS},
               "confusion": {str(s): confusion(data["test"][s][1], logits[s]).tolist() for s in subjects},
               "versions": {"torch": torch.__version__, "python": platform.python_version()},
               "wall_time_s": time.time() - t_all}
    (out / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=1))
    (out / "history.json").write_text(json.dumps(info["history"]))
    log("test mean: " + " ".join(f"{k} {v:.4f}" for k, v in metrics["test_mean"].items())
        + f" | wall {metrics['wall_time_s']:.0f}s")
    log_f.close()
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=CONFIG_V2)
    p.add_argument("--init", required=True, help="student run directory to start from (model.pt, student.json)")
    p.add_argument("--teacher", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--student", default="wide32_ft", help="name recorded in metrics/params")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=int(os.environ.get("RB_THREADS", "1")))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
