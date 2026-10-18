# SPDX-License-Identifier: Apache-2.0
"""Train the v2 teacher ensemble (EEGNet-ELU, ShallowConvNet, DeepConvNet) at 125 Hz.

    python -m src.train_teacher --seed 0 --out runs/v2/teacher_seed0
    python -m src.train_teacher --seed 0 --out runs/v2/teacher_smoke --subset-cho 10 --epochs 2

Data: ``src.data.splits125.v2_data(seed, cfg)`` (Cho2017 + Lee2019 train/val, BCI IV 2a
session 2 calibration and test blocks; see that module). Every member is trained with the
v0.1 ``src.train.fit`` (Adam, cross-entropy, early stopping on the validation loss, best
weights restored) on the canonical crop ``X[..., crop_offset:crop_offset + window]`` of the
training windows, divided by ``input_scale``. No 2a trial is trained on, and the 2a test
block is only scored after training.

Writes to ``--out``:

- ``members/{eegnet,shallow,deep}.pt``  float state dicts
- ``teacher_logits.npz``  per member ``{m}_train``, ``{m}_val``, ``{m}_calib_{s}``, ``{m}_test_{s}``
  (logits, canonical window, no augmentation), plus ``y_train``, ``y_val``, ``y_test_{s}`` and
  ``subjects``
- ``metrics.json``  per member and for the ensemble (mean of the members' softmax): KPIs per
  2a subject (``src.eval.compute_kpis``), their mean over subjects, confusion matrices, fit info
- ``params.json``   input scale, per-subject EA matrices (``ea_matrices``), fs, window, crop
  offset, data meta and the full config echo
- ``history.json``  per-member epoch history; ``train.log``

v3 (``training.loop: v3``, ``configs/v3.yaml``): members train with ``src.train_v3.fit_v3``
instead (AdamW, label smoothing, cosine + warm-up, EMA, on-the-fly ``src.augment`` on the
full ``train_window`` trials) with the settings of ``training.teacher`` and ``augment``;
``training.teacher.lr_<member>`` overrides the learning rate per member. Without the key the
v2 loop above runs unchanged.

v4 (``configs/v4.yaml``; every key optional, so v2 / v3 configs behave as before):

- ``members: [eegnet, shallow, deep, riemann]``: ``riemann`` = ``src.models.riemann``
  (OAS covariance -> tangent space -> logistic regression ``C = riemann.C``), fitted in
  seconds on the same canonical training windows, saved as ``members/riemann.pkl``. The
  ensemble is the mean of the members' softmax (the Riemannian member's probabilities).
- ``training.lr_member: {shallow: 3e-4}`` per-member learning rate of the v2 loop. With
  ``training.lr_reference: runs/v2/teacher_seed0`` the member's best validation loss is
  compared with that run's (same split only): the lower wins; otherwise it is retrained at
  ``training.lr``. The choice is in ``members*/{name}.json`` (``lr``, ``lr_choice``).
- ``self_training: {dataset, session, min_conf, rounds: 1}``: round 0 trains every member on
  the training split (``members_round0/``); its ensemble scores the unlabelled target session
  (``splits125.target_unlabeled``: X only, EA from that session's own trials) and the windows
  with max probability ``>= min_conf`` get hard pseudo-labels (``pseudo_labels.npz``); round 1
  retrains every member from scratch on train + pseudo-labelled windows (``members/``, same
  recipe, same validation split). ``teacher_logits.npz`` and ``metrics.json`` describe the final
  ensemble; ``metrics.json["round0"]`` and ``round0_metrics.json`` / ``teacher_logits_round0.npz``
  the ensemble before self-training. The agreement of the pseudo-labels with the true session
  labels is computed afterwards by ``pseudo_label_agreement_report`` (report only; the labels
  are never read by the training path). ``--round 0|1`` runs one round (parallel members).
- ``self_training.mode: soft`` (v4soft, ``configs/v4soft.yaml``): round 1 trains on **all** target
  windows with the round-0 ensemble probability as target, ``KL(p_ens || softmax(z))`` weighted by
  ``w = 2 |p_ens(right) - 0.5|`` (``src.train_soft.fit_soft``; labelled windows keep CE, weight 1);
  the Riemannian member gets the argmax label with ``sample_weight = w``. With
  ``self_training.finetune: {lr, max_epochs, patience}`` the CNN members start from their round-0
  weights (fine-tune) instead of from scratch; the Riemannian member is refitted.
  ``self_training.reuse_round0: <teacher run dir>`` copies that run's ``members_round0/``
  (same data, seed and split, checked) instead of retraining round 0.
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
import yaml
from scipy.special import softmax
from torch import nn

from src.eval import KPI_KEYS, compute_kpis, confusion
from src.models.convnets import DeepConvNet, ShallowConvNet
from src.models.eegnet import EEGNet
from src.train import CHANNELS, fit, predict, set_seed, to_model_input

ROOT = Path(__file__).resolve().parent.parent
CONFIG_V2 = ROOT / "configs" / "v2.yaml"
MEMBERS = ("eegnet", "shallow", "deep")              # v2 / v3 default (``cfg["members"]`` overrides)
ALL_MEMBERS = ("eegnet", "shallow", "deep", "riemann")
NON_TORCH = ("riemann",)                              # sklearn pipeline in members/{name}.pkl
MEMBER_LABELS = {"eegnet": "EEGNet-ELU", "shallow": "ShallowConvNet", "deep": "DeepConvNet",
                 "riemann": "Riemann TS + LR"}


# ----------------------------------------------------------------------------- config and data
def load_v2_config(path: str | Path = CONFIG_V2) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def subject_list(spec) -> list[int]:
    """Subjects from a config value: list of ints, one int, or an ``"a-b"`` range string."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        if "-" in spec:
            a, b = (int(v) for v in spec.split("-"))
            return list(range(a, b + 1))
        return [int(v) for v in spec.replace(",", " ").split()]
    return [int(v) for v in spec]


def subset_cho(cfg: dict, n: int) -> dict:
    """Copy of ``cfg`` keeping only the first ``n`` Cho2017 subjects (smoke runs)."""
    cfg = json.loads(json.dumps(cfg))
    cho = cfg["datasets"]["Cho2017"]
    exclude = set(subject_list(cho.get("exclude") or []))
    cho["subjects"] = [s for s in subject_list(cho["subjects"]) if s not in exclude][:n]
    return cfg


def load_v2_data(seed: int, cfg: dict) -> dict:
    """``src.data.splits125.v2_data`` (looked up at call time so tests can monkeypatch it)."""
    from src.data import splits125

    return splits125.v2_data(seed, cfg)


def canonical(X: np.ndarray, cfg: dict) -> np.ndarray:
    """Canonical ``window``-sample crop at ``crop_offset`` of longer training windows;
    arrays that already have ``window`` samples are returned unchanged."""
    W, off = int(cfg["window"]), int(cfg["crop_offset"])
    if X.shape[-1] == W:
        return X
    return X[..., off : off + W]


def jsonable(obj):
    """Numpy scalars/arrays and int dict keys -> plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_params(out: Path, data: dict, cfg: dict, extra: dict | None = None) -> dict:
    """``params.json`` shared by teacher and student runs (and read by the export step)."""
    params = {
        "fs": float(cfg.get("fs_model", 125)),
        "n_channels": len(CHANNELS),
        "channels": CHANNELS,
        "window_samples": int(cfg["window"]),
        "train_window_samples": int(cfg.get("train_window", cfg["window"])),
        "crop_offset": int(cfg["crop_offset"]),
        "input_shape": [1, 1, len(CHANNELS), int(cfg["window"])],
        "units": "uV",
        "reference": "common_average_22",
        "filter": cfg.get("filter"),
        "ea_eps": cfg.get("ea_eps"),
        "ea_matrices": {str(s): np.asarray(A).tolist() for s, A in data["ea"].items()},
        "input_scale": float(data["input_scale"]),
        "class_map": {"0": "left_hand (769)", "1": "right_hand (770)"},
        "meta": jsonable(data.get("meta", {})),
        "config": jsonable(cfg),
    }
    params.update(jsonable(extra or {}))
    with open(out / "params.json", "w") as f:
        json.dump(params, f, indent=1)
    return params


# ----------------------------------------------------------------------------- models
def member_names(cfg: dict) -> tuple[str, ...]:
    """Teacher members of a config: ``cfg["members"]`` (v4) or the v2 three."""
    names = tuple(cfg.get("members") or MEMBERS)
    bad = [n for n in names if n not in ALL_MEMBERS]
    if bad:
        raise ValueError(f"unknown teacher members {bad}")
    return names


def build_member(name: str, cfg: dict) -> nn.Module:
    n_samples = int(cfg["window"])
    if name == "eegnet":
        return EEGNet(n_samples=n_samples, kern_t=64, dropout=cfg["training"]["dropout"])
    if name == "shallow":
        return ShallowConvNet(n_samples=n_samples)
    if name == "deep":
        return DeepConvNet(n_samples=n_samples)
    if name == "riemann":
        from src.models.riemann import RiemannMember

        rc = cfg.get("riemann") or {}
        return RiemannMember(C=float(rc.get("C", 1.0)), ridge=float(rc.get("ridge", 1e-3)))
    raise ValueError(name)


def member_file(mdir: Path, name: str) -> Path:
    return Path(mdir) / (f"{name}.pkl" if name in NON_TORCH else f"{name}.pt")


def load_member(name: str, mdir: str | Path, cfg: dict, device: str = "cpu") -> nn.Module:
    if name in NON_TORCH:
        from src.models.riemann import RiemannMember

        model = RiemannMember.load(member_file(Path(mdir), name))
    else:
        model = build_member(name, cfg)
        model.load_state_dict(torch.load(member_file(Path(mdir), name), map_location="cpu", weights_only=True))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_members(teacher_dir: str | Path, cfg: dict, device: str = "cpu", subdir: str = "members") -> dict[str, nn.Module]:
    """Trained teacher members in eval mode, gradients off (``member_names(cfg)``; the
    Riemannian member from ``members/riemann.pkl``)."""
    return {name: load_member(name, Path(teacher_dir) / subdir, cfg, device) for name in member_names(cfg)}


def ensemble_logits(logits: list[np.ndarray]) -> np.ndarray:
    """log of the mean member softmax; ``softmax`` of it is the ensemble probability."""
    p = np.mean([softmax(np.asarray(z, dtype=np.float64), axis=1) for z in logits], axis=0)
    return np.log(np.clip(p, 1e-12, None)).astype(np.float32)


def subject_report(test: dict, logits: dict) -> dict:
    """KPIs and confusion per 2a subject plus the mean of each KPI over subjects."""
    per = {s: compute_kpis(test[s][1], logits[s]) for s in test}
    return {"test": {str(s): k for s, k in per.items()},
            "test_mean": {k: float(np.mean([per[s][k] for s in per])) for k in KPI_KEYS},
            "confusion": {str(s): confusion(test[s][1], logits[s]).tolist() for s in test}}


def fit_riemann(model: nn.Module, X_train: torch.Tensor, y_train, X_val: torch.Tensor, y_val, log=print,
                sample_weight=None) -> dict:
    """Fit the Riemannian member (no epochs, no early stopping); fit info like ``src.train.fit``.
    ``sample_weight`` (v4soft round 1) weights the logistic regression per window."""
    t0 = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    logp = torch.from_numpy(predict(model, X_val))
    yv = torch.as_tensor(y_val, dtype=torch.long)
    info = {"best_epoch": 0, "epochs_run": 1, "best_val_loss": float(nn.functional.nll_loss(logp, yv)),
            "best_val_acc": float((logp.argmax(1) == yv).float().mean()), "fit_time_s": time.time() - t0,
            "n_coefficients": int(model.n_coefficients), "history": []}
    log(f"riemann: val loss {info['best_val_loss']:.4f} acc {info['best_val_acc']:.3f} ({info['fit_time_s']:.1f}s)")
    return info


# ----------------------------------------------------------------------------- v4 self-training
def load_target_unlabeled(cfg: dict, input_scale: float) -> dict:
    """``src.data.splits125.target_unlabeled`` (looked up at call time so tests can monkeypatch it)."""
    from src.data import splits125

    return splits125.target_unlabeled(cfg, input_scale=input_scale)


def select_pseudo_labels(probs: dict, min_conf: float) -> dict:
    """``{subject: (n, 2) ensemble probability}`` -> per subject kept indices (max prob
    ``>= min_conf``), hard pseudo-labels (argmax) and confidences of the kept trials."""
    out = {}
    for s, p in probs.items():
        p = np.asarray(p, dtype=np.float64)
        conf = p.max(1)
        keep = np.flatnonzero(conf >= min_conf)
        out[s] = {"idx": keep.astype(np.int64), "y": p.argmax(1)[keep].astype(np.int64), "conf": conf[keep],
                  "n": int(len(p))}
    return out


def save_pseudo_labels(path: Path, pseudo: dict, min_conf: float, probs: dict | None = None) -> None:
    """Hard pseudo-labels per subject; with ``probs`` also the round-0 ensemble probability of
    every target window (``p_{s}``, the v4soft targets)."""
    arrays = {"subjects": np.array(sorted(pseudo)), "min_conf": np.float64(min_conf)}
    for s, d in pseudo.items():
        arrays[f"idx_{s}"], arrays[f"y_{s}"], arrays[f"conf_{s}"] = d["idx"], d["y"], d["conf"]
        arrays[f"n_{s}"] = np.int64(d["n"])
        if probs is not None:
            arrays[f"p_{s}"] = np.asarray(probs[s], dtype=np.float64)
    np.savez(path, **arrays)


def load_pseudo_labels(path: Path) -> dict:
    z = np.load(path)
    out = {}
    for s in z["subjects"]:
        s = int(s)
        out[s] = {"idx": z[f"idx_{s}"], "y": z[f"y_{s}"], "conf": z[f"conf_{s}"], "n": int(z[f"n_{s}"])}
        if f"p_{s}" in z.files:
            out[s]["p"] = z[f"p_{s}"]
    return out


def soft_targets(pseudo: dict, subjects) -> tuple[np.ndarray, np.ndarray]:
    """v4soft: every target window with its round-0 ensemble probability and weight
    ``w = 2 |p(right) - 0.5|`` (subjects concatenated in order)."""
    from src.train_soft import target_weights

    p = np.concatenate([pseudo[s]["p"] for s in subjects]).astype(np.float64)
    return p, target_weights(p)


def pseudo_label_counts(pseudo: dict) -> dict:
    """Training-side summary (no true labels): kept trials and pseudo-class counts per subject."""
    per = {str(s): {"n": d["n"], "kept": int(len(d["idx"])), "kept_frac": len(d["idx"]) / max(d["n"], 1),
                    "pseudo_left": int(np.sum(d["y"] == 0)), "pseudo_right": int(np.sum(d["y"] == 1)),
                    "mean_conf": float(np.mean(d["conf"])) if len(d["conf"]) else float("nan")}
           for s, d in sorted(pseudo.items())}
    n, kept = sum(v["n"] for v in per.values()), sum(v["kept"] for v in per.values())
    return {"per_subject": per, "n": n, "kept": kept, "kept_frac": kept / max(n, 1)}


def pseudo_label_agreement_report(pseudo: dict, all_pred: dict, cfg: dict, label_loader=None,
                                  weights: dict | None = None) -> dict:
    """REPORTING ONLY (transparency, after the pseudo-labels are fixed and saved): agreement of
    the pseudo-labels, and of the round-0 ensemble on every target trial, with the true labels of
    the unlabelled session. Reads them via ``splits125.session_labels_for_reporting_only``; its
    result never flows back into training."""
    from src.data import splits125

    truth = splits125.session_labels_for_reporting_only(cfg, loader=label_loader)
    per = {}
    for s, d in sorted(pseudo.items()):
        y = truth[s]
        per[str(s)] = {"kept": int(len(d["idx"])),
                       "agreement_kept": float(np.mean(d["y"] == y[d["idx"]])) if len(d["idx"]) else float("nan"),
                       "accuracy_all": float(np.mean(np.asarray(all_pred[s]) == y)), "n": int(len(y))}
        if weights is not None:
            w = np.asarray(weights[s], dtype=np.float64)
            per[str(s)]["weighted_agreement_all"] = float(np.sum(w * (np.asarray(all_pred[s]) == y)) / max(w.sum(), 1e-12))
    kept = sum(v["kept"] for v in per.values())
    agree = sum(v["agreement_kept"] * v["kept"] for v in per.values() if v["kept"])
    rep = {"per_subject": per, "agreement_kept": agree / max(kept, 1),
           "accuracy_all": float(np.mean([v["accuracy_all"] for v in per.values()])),
           "note": "true session labels read for this report only, never for training"}
    if weights is not None:
        wsum = sum(float(np.sum(weights[s])) for s in pseudo)
        rep["weighted_agreement_all"] = sum(per[str(s)]["weighted_agreement_all"] * float(np.sum(weights[s]))
                                            for s in pseudo) / max(wsum, 1e-12)
    return rep


# ----------------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> dict:
    cfg = load_v2_config(args.config)
    if args.subset_cho:
        cfg = subset_cho(cfg, args.subset_cho)
    tr = cfg["training"]
    v3 = tr.get("loop", "v2") == "v3"
    tt = {**tr, **(tr.get("teacher") or {})} if v3 else tr
    lr = args.lr if args.lr is not None else tt["lr"]
    batch_size = args.batch_size or tt["batch_size"]
    max_epochs = args.epochs or tt["max_epochs"]
    patience = args.patience or tt["patience"]
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    (out / "members").mkdir(parents=True, exist_ok=True)
    only = list(args.members) if getattr(args, "members", None) else None
    reuse = bool(getattr(args, "reuse_members", False))
    names = member_names(cfg)
    if only is not None and set(only) - set(names):
        raise ValueError(f"--members {only} not in the config's members {list(names)}")
    st = cfg.get("self_training")
    if st and v3:
        raise ValueError("self_training is implemented for the v2 loop only")
    if st and int(st.get("rounds", 1)) != 1:
        raise ValueError("self_training.rounds: only one round is implemented")
    mode = str((st or {}).get("mode", "hard"))
    if mode not in ("hard", "soft"):
        raise ValueError(f"self_training.mode {mode!r}: hard or soft")
    rounds = [None] if not st else ([int(args.round)] if getattr(args, "round", None) is not None else [0, 1])
    stem = "train" + ("" if getattr(args, "round", None) is None else f"_round{args.round}")
    log_f = open(out / (f"{stem}.log" if only is None else f"{stem}_{'_'.join(only)}.log"), "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    t_all = time.time()
    set_seed(args.seed)
    data = load_v2_data(args.seed, cfg)
    scale = float(data["input_scale"])
    subjects = sorted(data["test"])
    X_train = to_model_input(canonical(data["train"][0], cfg), scale)
    X_val = to_model_input(canonical(data["val"][0], cfg), scale)
    X_calib = {s: to_model_input(data["calib"][s], scale) for s in subjects}
    X_test = {s: to_model_input(data["test"][s][0], scale) for s in subjects}
    y_train, y_val = data["train"][1], data["val"][1]
    augmenter = X_train_full = None
    if v3:
        from src.augment import Augmenter, mixing_matrices

        X_train_full = to_model_input(data["train"][0], scale)
        augmenter = Augmenter(cfg.get("augment"), window=int(cfg["window"]), crop_offset=int(cfg["crop_offset"]),
                              train_window=X_train_full.shape[-1], mix=mixing_matrices(data["ea_train"]))
    X_target = None
    if st:
        target = load_target_unlabeled(cfg, scale)  # X only, already divided by input_scale
        X_target = {s: torch.from_numpy(np.ascontiguousarray(target["X"][s][:, None], dtype=np.float32))
                    for s in subjects}
    log(f"teacher seed {args.seed}: train {tuple(X_train.shape)} val {tuple(X_val.shape)} test subjects {subjects} "
        f"({sum(len(X_test[s]) for s in subjects)} trials) input_scale {scale:.4f} threads {args.threads} "
        f"device {args.device} members {list(names)} meta {json.dumps(jsonable(data.get('meta', {})))}"
        + (f" | self-training {st}, unlabelled target {sum(len(v) for v in X_target.values())} windows" if st else ""))

    lr_member = dict(tr.get("lr_member") or {}) if not v3 else {}

    def member_lr(name: str, rnd) -> float:
        """Learning rate of a v2-loop member; round 1 reuses round 0's choice."""
        if args.lr is not None:
            return float(args.lr)
        if rnd == 1:
            info0 = out / "members_round0" / f"{name}.json"
            if info0.exists():
                return float(json.loads(info0.read_text()).get("lr", lr))
        return float(lr_member.get(name, lr))

    def lr_reference_loss(name: str) -> float | None:
        ref = tr.get("lr_reference")
        if not ref or name not in lr_member:
            return None
        path = (ROOT / ref if not Path(ref).is_absolute() else Path(ref)) / "metrics.json"
        if not path.exists():
            log(f"lr reference {path} missing; keeping lr {lr_member[name]} for {name}")
            return None
        m = json.loads(path.read_text())
        if int(m.get("n_val", -1)) != len(X_val) or not np.isclose(float(m.get("input_scale", 0)), scale, rtol=1e-5):
            log(f"lr reference {path} has another validation split (n_val {m.get('n_val')} vs {len(X_val)}); "
                f"keeping lr {lr_member[name]} for {name}")
            return None
        return float(m["members"][name]["best_val_loss"])

    def copy_round0(mdir: Path) -> None:
        """v4soft: ``self_training.reuse_round0`` = a teacher run directory whose round-0 members
        (same data, seed and split) are copied instead of retrained."""
        import shutil

        src_dir = Path(st["reuse_round0"])
        src_dir = src_dir if src_dir.is_absolute() else ROOT / src_dir
        m = json.loads((src_dir / "metrics.json").read_text())
        if int(m["n_val"]) != len(X_val) or int(m["n_train"]) != len(X_train) or \
                not np.isclose(float(m["input_scale"]), scale, rtol=1e-6) or int(m["seed"]) != int(args.seed):
            raise ValueError(f"{src_dir}: round-0 members were trained on another split")
        for name in names:
            for f in (member_file(src_dir / "members_round0", name), src_dir / "members_round0" / f"{name}.json"):
                if not (mdir / f.name).exists():
                    shutil.copy2(f, mdir / f.name)
        log(f"round 0: members copied from {src_dir / 'members_round0'} (not retrained)")

    def train_round(rnd, Xtr: torch.Tensor, ytr: np.ndarray, soft=None) -> dict:
        """Train (or reuse) every member of one round; returns per-member results or ``partial``.
        ``soft = (p (n, 2), w (n,))``: v4soft round 1 (``src.train_soft.fit_soft``; the Riemannian
        member gets ``ytr`` = argmax with ``sample_weight = w``)."""
        mdir = out / ("members_round0" if rnd == 0 else "members")
        mdir.mkdir(parents=True, exist_ok=True)
        reuse_r = reuse
        if rnd == 0 and st and st.get("reuse_round0"):
            copy_round0(mdir)
            reuse_r = True
        res = {"members": {}, "histories": {}, "models": {}, "missing": []}
        for name in names:
            set_seed(args.seed)
            model = build_member(name, cfg)
            n_params = sum(p.numel() for p in model.parameters())
            member_pt, member_json = member_file(mdir, name), mdir / f"{name}.json"
            have = member_pt.exists() and member_json.exists()
            tag = "" if rnd is None else f" round {rnd}"
            if (only is not None and name not in only) or (reuse_r and have):
                if not (reuse_r and have):
                    res["missing"].append(name)
                    continue
                model = load_member(name, mdir, cfg)
                n_show = model.n_coefficients if name in NON_TORCH else n_params
                log(f"--- {MEMBER_LABELS[name]}{tag} ({n_show} params): reusing {member_pt}")
                info = json.loads(member_json.read_text())
            elif name in NON_TORCH:
                log(f"--- {MEMBER_LABELS[name]}{tag}: OAS covariance -> tangent space (riemann) -> logistic regression")
                info = fit_riemann(model, Xtr, ytr, X_val, y_val, log=log,
                                   sample_weight=None if soft is None else soft[1])
            elif v3:
                log(f"--- {MEMBER_LABELS[name]} ({n_params} params)")
                from src.train_v3 import fit_v3

                lr_m = args.lr if args.lr is not None else float(tt.get(f"lr_{name}", lr))
                log(f"v3 loop: lr {lr_m} weight decay {tt.get('weight_decay', 0.0)} label smoothing "
                    f"{tt.get('label_smoothing', 0.0)} ema {tt.get('ema', 0.0)} warmup {tt.get('warmup', 0)} "
                    f"augment {augmenter.cfg} (ea_mix_p used {augmenter.ea_mix_p})")
                info = fit_v3(model, X_train_full, y_train, data["train_group"], X_val, y_val, augmenter=augmenter,
                              lr=lr_m, weight_decay=float(tt.get("weight_decay", 0.0)),
                              label_smoothing=float(tt.get("label_smoothing", 0.0)), ema_decay=float(tt.get("ema", 0.0)),
                              warmup_epochs=int(tt.get("warmup", 0)), batch_size=batch_size, max_epochs=max_epochs,
                              patience=patience, seed=args.seed, device=args.device, log=log)
                info["lr"] = lr_m
            else:
                lr_m = member_lr(name, rnd)
                log(f"--- {MEMBER_LABELS[name]}{tag} ({n_params} params)" + (f" lr {lr_m}" if lr_m != lr else ""))
                if soft is None:
                    info = fit(model, Xtr, ytr, X_val, y_val, lr=lr_m, batch_size=batch_size, max_epochs=max_epochs,
                               patience=patience, seed=args.seed, device=args.device, log=log)
                else:
                    from src.train_soft import fit_soft

                    ft = (st or {}).get("finetune")
                    ep_m, pat_m = max_epochs, patience
                    if ft:  # fine-tune the round-0 member instead of training from scratch
                        model.load_state_dict(torch.load(member_file(out / "members_round0", name), map_location="cpu",
                                                         weights_only=True))
                        lr_m = float(args.lr if args.lr is not None else ft.get("lr", lr_m))
                        ep_m = int(args.epochs or ft.get("max_epochs", max_epochs))
                        pat_m = int(args.patience or ft.get("patience", patience))
                        log(f"fine-tuning round-0 {name}: lr {lr_m} max epochs {ep_m} patience {pat_m}")
                    info = fit_soft(model, Xtr, soft[0], soft[1], X_val, y_val, lr=lr_m, batch_size=batch_size,
                                    max_epochs=ep_m, patience=pat_m, seed=args.seed, device=args.device, log=log)
                    info.update({"soft_targets": True, "lr": lr_m, "finetune_from_round0": bool(ft),
                                 "max_epochs": ep_m, "patience": pat_m})
                if lr_member:
                    info["lr"] = lr_m
                ref = lr_reference_loss(name) if rnd in (None, 0) and lr_m != lr and args.lr is None else None
                if ref is not None:
                    choice = {"tried_lr": lr_m, "tried_val_loss": info["best_val_loss"], "reference_val_loss": ref,
                              "reference": str(tr.get("lr_reference")), "fallback_lr": lr}
                    if info["best_val_loss"] < ref:
                        log(f"{name}: lr {lr_m} val loss {info['best_val_loss']:.4f} < reference {ref:.4f}: kept")
                        choice["kept_lr"] = lr_m
                    else:
                        log(f"{name}: lr {lr_m} val loss {info['best_val_loss']:.4f} >= reference {ref:.4f}: "
                            f"retraining at lr {lr}")
                        set_seed(args.seed)
                        model = build_member(name, cfg)
                        info = fit(model, Xtr, ytr, X_val, y_val, lr=lr, batch_size=batch_size, max_epochs=max_epochs,
                                   patience=patience, seed=args.seed, device=args.device, log=log)
                        info["lr"] = lr
                        choice["kept_lr"] = lr
                    info["lr_choice"] = choice
            model.cpu().eval()
            if not (reuse_r and have):
                if name in NON_TORCH:
                    model.save(member_pt)
                else:
                    torch.save(model.state_dict(), member_pt)
                member_json.write_text(json.dumps(jsonable(info)))
            res["models"][name] = model
            res["histories"][name] = info.pop("history", [])
            res["members"][name] = {"label": MEMBER_LABELS[name],
                                    "params": int(info.get("n_coefficients", n_params)) if name in NON_TORCH else n_params,
                                    **info}
        return res

    def score_round(res: dict, arrays: dict | None) -> dict:
        """Member and ensemble logits / KPIs of a finished round (``arrays`` collects npz keys)."""
        test_logits, val_logits, target_logits = {}, {}, {}
        members_m = {}
        for name in names:
            model = res["models"][name]
            if arrays is not None:
                arrays[f"{name}_train"] = predict(model, X_train)
            val_logits[name] = predict(model, X_val)
            test_logits[name] = {s: predict(model, X_test[s]) for s in subjects}
            if arrays is not None:
                arrays[f"{name}_val"] = val_logits[name]
                for s in subjects:
                    arrays[f"{name}_calib_{s}"] = predict(model, X_calib[s])
                    arrays[f"{name}_test_{s}"] = test_logits[name][s]
            if X_target is not None:
                target_logits[name] = {s: predict(model, X_target[s]) for s in subjects}
                if arrays is not None:
                    for s in subjects:
                        arrays[f"{name}_target_{s}"] = target_logits[name][s]
            rep = subject_report(data["test"], test_logits[name])
            members_m[name] = {**res["members"][name], **rep}
            log(f"{MEMBER_LABELS[name]} test mean: " + " ".join(f"{k} {v:.4f}" for k, v in rep["test_mean"].items()))
        ens_test = {s: ensemble_logits([test_logits[m][s] for m in names]) for s in subjects}
        ens_val = ensemble_logits([val_logits[m] for m in names])
        yv = torch.as_tensor(y_val, dtype=torch.long)
        ens = {"label": "Teacher ensemble", "members": list(names),
               "params": sum(members_m[m]["params"] for m in names),
               "val_loss": float(nn.functional.nll_loss(torch.from_numpy(ens_val), yv)),
               "val_acc": float(np.mean(ens_val.argmax(1) == y_val)),
               **subject_report(data["test"], ens_test)}
        ens_target = ({s: ensemble_logits([target_logits[m][s] for m in names]) for s in subjects}
                      if X_target is not None else None)
        return {"members": members_m, "ensemble": ens, "ens_target": ens_target, "ens_test": ens_test}

    arrays: dict[str, np.ndarray] = {"y_train": y_train, "y_val": y_val, "subjects": np.array(subjects)}
    for s in subjects:
        arrays[f"y_test_{s}"] = data["test"][s][1]
    metrics: dict = {"seed": args.seed, "n_train": int(len(X_train)), "n_val": int(len(X_val)),
                     "n_calib": int(sum(len(X_calib[s]) for s in subjects)),
                     "n_test": int(sum(len(X_test[s]) for s in subjects)), "input_scale": scale,
                     "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
                     "threads": args.threads, "device": args.device, "meta": jsonable(data.get("meta", {})),
                     "members": {}}
    if lr_member:
        metrics["lr_member"] = lr_member
    final = None
    histories: dict = {}
    for rnd in rounds:
        soft = None
        if rnd in (None, 0):
            Xtr, ytr = X_train, y_train
        else:
            pl_path = out / "pseudo_labels.npz"
            if not pl_path.exists():
                raise FileNotFoundError(f"{pl_path} missing: assemble round 0 first (--reuse-members --round 0)")
            pseudo = load_pseudo_labels(pl_path)
            if mode == "soft":  # every target window, round-0 probability as target, weight 2 |p_right - 0.5|
                X_pl = torch.cat([X_target[s] for s in subjects])
                p_pl, w_pl = soft_targets(pseudo, subjects)
                y_pl = p_pl.argmax(1).astype(np.int64)
                p_tr = np.eye(2)[y_train]
                soft = (np.concatenate([p_tr, p_pl]), np.concatenate([np.ones(len(y_train)), w_pl]))
                log(f"round 1 (soft): train {len(X_train)} + {len(X_pl)} target windows with soft targets, "
                    f"weight sum {w_pl.sum():.1f} (mean {w_pl.mean():.3f})")
            else:
                X_pl = torch.cat([X_target[s][pseudo[s]["idx"]] for s in subjects])
                y_pl = np.concatenate([pseudo[s]["y"] for s in subjects]).astype(np.int64)
                log(f"round 1: train {len(X_train)} + {len(X_pl)} pseudo-labelled target windows "
                    f"(left {int(np.sum(y_pl == 0))}, right {int(np.sum(y_pl == 1))})")
            Xtr, ytr = torch.cat([X_train, X_pl]), np.concatenate([y_train, y_pl])
        res = train_round(rnd, Xtr, ytr, soft)
        if res["missing"]:
            log(f"partial run: trained {only}; {res['missing']} not present, ensemble outputs not written "
                "(assemble with --reuse-members once every member exists)")
            log_f.close()
            return {"partial": True, "trained": only, "missing": res["missing"], "round": rnd}
        if rnd == 0:
            log("round 0 (before self-training):")
            arrays0 = {k: v for k, v in arrays.items()}
            sc0 = score_round(res, arrays0)
            for s in subjects:
                arrays0[f"ensemble_target_{s}"] = sc0["ens_target"][s]
            np.savez(out / "teacher_logits_round0.npz", **arrays0)
            min_conf = float(st.get("min_conf", 0.8))
            probs = {s: softmax(sc0["ens_target"][s].astype(np.float64), axis=1) for s in subjects}
            pseudo = select_pseudo_labels(probs, min_conf)
            save_pseudo_labels(out / "pseudo_labels.npz", pseudo, min_conf, probs)
            counts = pseudo_label_counts(pseudo)
            weights = None
            if mode == "soft":
                from src.train_soft import target_weights

                weights = {s: target_weights(probs[s]) for s in subjects}
                for s in subjects:
                    counts["per_subject"][str(s)].update({"w_sum": float(weights[s].sum()),
                                                          "w_mean": float(weights[s].mean())})
                counts["w_sum"] = float(sum(weights[s].sum() for s in subjects))
            counts["mode"] = mode
            counts["n_round1_target"] = counts["n"] if mode == "soft" else counts["kept"]
            log(f"pseudo-labels (max prob >= {min_conf}): kept {counts['kept']} of {counts['n']} "
                f"({100 * counts['kept_frac']:.1f} %): " + ", ".join(
                    f"A0{s} {v['kept']}" for s, v in counts["per_subject"].items()))
            # Reporting only, after the pseudo-labels are saved; never read back by training.
            agreement = pseudo_label_agreement_report(pseudo, {s: probs[s].argmax(1) for s in subjects}, cfg,
                                                      weights=weights)
            log(f"pseudo-label agreement with the true session labels (report only): "
                f"{100 * agreement['agreement_kept']:.1f} % of kept; round-0 ensemble on all target trials "
                f"{100 * agreement['accuracy_all']:.1f} %")
            r0 = {"members": sc0["members"], "ensemble": sc0["ensemble"],
                  "self_training": {"config": st, "counts": counts, "agreement_report_only": agreement}}
            (out / "round0_metrics.json").write_text(json.dumps(jsonable(r0), indent=1))
            (out / "round0_history.json").write_text(json.dumps(res["histories"]))
            if rounds == [0]:
                log_f.close()
                return {"partial": True, "round": 0, **r0}
            continue
        histories = res["histories"]
        final = score_round(res, arrays)

    metrics["members"] = final["members"]
    metrics["ensemble"] = final["ensemble"]
    if st:
        r0 = json.loads((out / "round0_metrics.json").read_text())
        metrics["round0"] = {"members": r0["members"], "ensemble": r0["ensemble"]}
        c0 = r0["self_training"]["counts"]
        metrics["self_training"] = {**r0["self_training"], "n_train_round1": int(len(X_train)) + int(
            c0.get("n_round1_target", c0["kept"]))}
        for s in subjects:
            arrays[f"ensemble_target_{s}"] = final["ens_target"][s]
    np.savez(out / "teacher_logits.npz", **arrays)
    write_params(out, data, cfg, {"teacher_members": list(names), "seed": args.seed,
                                  **({"self_training": st} if st else {})})
    metrics["wall_time_s"] = time.time() - t_all
    metrics["versions"] = {"torch": torch.__version__, "python": platform.python_version()}
    with open(out / "metrics.json", "w") as f:
        json.dump(jsonable(metrics), f, indent=1)
    with open(out / "history.json", "w") as f:
        json.dump(histories, f)
    log("ensemble test mean: " + " ".join(f"{k} {v:.4f}" for k, v in metrics["ensemble"]["test_mean"].items())
        + f" | wall {metrics['wall_time_s']:.0f}s")
    log_f.close()
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--config", type=Path, default=CONFIG_V2)
    p.add_argument("--device", default="cpu", help="cpu or mps")
    p.add_argument("--threads", type=int, default=int(os.environ.get("RB_THREADS", "1")))
    p.add_argument("--subset-cho", type=int, default=None, help="keep only the first N Cho2017 subjects (smoke)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--members", nargs="+", choices=ALL_MEMBERS, default=None,
                   help="train only these members (parallel runs); writes members/{name}.pt + .json, no ensemble")
    p.add_argument("--reuse-members", action="store_true",
                   help="load members whose members/{name}.pt + .json exist instead of training them, "
                        "then write the ensemble outputs")
    p.add_argument("--round", type=int, choices=(0, 1), default=None,
                   help="self_training configs only: run just this round (0 = before, 1 = with the pseudo-labels); "
                        "default runs both in sequence")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
