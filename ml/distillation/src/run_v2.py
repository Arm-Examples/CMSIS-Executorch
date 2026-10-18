# SPDX-License-Identifier: Apache-2.0
"""Run the v2 teacher/student experiments and write ``results/results_v2.md``.

    python -m src.run_v2 --jobs 3 --seeds 0 1 2 --students wide32 control
    python -m src.run_v2 --skip-train                      # only rebuild the report from runs/v2
    python -m src.run_v2 --config configs/v3.yaml --seeds 0 --out runs/v3 --results results/results_v3.md \
        --teacher-parallel                                 # v3; members of a teacher in parallel
    python -m src.run_v2 --config configs/v4.yaml --seeds 0 --out runs/v4 --results results/results_v4.md \
        --teacher-parallel                                 # v4; per self-training round, members in parallel

Runs (one directory each under ``runs/v2/``, finished runs are skipped):
  teacher_seed{K}             src.train_teacher (EEGNet-ELU, ShallowConvNet, DeepConvNet)
  student_{name}_seed{K}      src.distill from teacher_seed{K}

Every training process gets ``--threads`` torch threads (1 by default, as ``src.run_all``);
teachers run first, then the students. Int8 rows are read from
``runs/v2/student_{name}_seed{K}/int8_predictions.npz`` (same keys as ``predictions.npz``:
``subjects``, ``y_true_{s}``, ``logits_{s}``) when present, otherwise marked pending.
Test set for every row: BCI IV 2a session 2 trials 25-144 of subjects 1-9, never trained on.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import torch

from src.eval import KPI_KEYS, compute_kpis, confusion
from src.run_all import across_subjects, fmt, fmt_pm, git_commit, subject_mean, subject_std
from src.train_teacher import (CONFIG_V2, MEMBER_LABELS, MEMBERS, NON_TORCH, build_member, load_v2_config,
                               member_names, subject_list)

ROOT = Path(__file__).resolve().parent.parent
RUNS_V2 = ROOT / "runs" / "v2"
RESULTS = ROOT / "results"


def teacher_dir(runs: Path, seed: int) -> Path:
    return runs / f"teacher_seed{seed}"


def student_dir(runs: Path, name: str, seed: int) -> Path:
    return runs / f"student_{name}_seed{seed}"


# ----------------------------------------------------------------------------- training
def run_job(cmd: list[str], out: Path, log_name: str = "stdout.log") -> tuple[int, float]:
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(out / log_name, "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - t


def common_flags(a: argparse.Namespace) -> list[str]:
    flags = ["--config", str(a.config), "--threads", str(a.threads), "--device", a.device]
    if a.epochs:
        flags += ["--epochs", str(a.epochs)]
    if a.subset_cho:
        flags += ["--subset-cho", str(a.subset_cho)]
    return flags


def train_all(a: argparse.Namespace) -> None:
    runs = Path(a.runs_dir)
    py = sys.executable
    teachers = [k for k in a.seeds if not (teacher_dir(runs, k) / "metrics.json").exists()]
    students = [(n, k) for n in a.students for k in a.seeds if not (student_dir(runs, n, k) / "metrics.json").exists()]
    print(f"{len(teachers)} teacher and {len(students)} student runs pending, {a.jobs} in parallel x {a.threads} "
          f"threads", flush=True)
    if teachers or students:
        # Build every per-subject cache once before parallel workers start.
        from src.train_teacher import load_v2_data, subset_cho

        cfg = load_v2_config(a.config)
        load_v2_data(a.seeds[0], subset_cho(cfg, a.subset_cho) if a.subset_cho else cfg)
    t0 = time.time()

    cfg_run = load_v2_config(a.config)
    names = member_names(cfg_run)
    # v4: self-training = round 0 (members_round0/, pseudo-labels), then round 1 (members/).
    rounds = [0, 1] if cfg_run.get("self_training") else [None]

    def teacher(k: int):
        out = teacher_dir(runs, k)
        base = [py, "-m", "src.train_teacher", "--seed", str(k), "--out", str(out), *common_flags(a)]
        if not a.teacher_parallel:
            return ("teacher", k), *run_job(base, out)
        # Per round: one process per member in parallel, then one process that assembles the ensemble.
        t = time.time()
        for rnd in rounds:
            rflag = [] if rnd is None else ["--round", str(rnd)]
            sfx = "" if rnd is None else f"_round{rnd}"
            if rnd == 0 and (cfg_run.get("self_training") or {}).get("reuse_round0"):
                # v4soft: round-0 members are copied from another run by the assembly step
                rc, _ = run_job([*base, *rflag, "--reuse-members"], out, f"stdout{sfx}.log")
                print(f"  teacher seed{k}{sfx} (reused members) ensemble: rc {rc} ({time.time() - t:.0f}s)", flush=True)
                if rc:
                    return ("teacher", k), rc, time.time() - t
                continue
            with ThreadPoolExecutor(max_workers=len(names)) as mex:
                res = list(mex.map(lambda m: (m, *run_job([*base, *rflag, "--members", m], out,
                                                          f"stdout{sfx}_{m}.log")), names))
            for m, rc, dt in res:
                print(f"  teacher seed{k}{sfx} member {m}: rc {rc} ({dt:.0f}s)", flush=True)
            if any(rc for _, rc, _ in res):
                return ("teacher", k), max(rc for _, rc, _ in res), time.time() - t
            rc, _ = run_job([*base, *rflag, "--reuse-members"], out, f"stdout{sfx}.log")
            print(f"  teacher seed{k}{sfx} ensemble: rc {rc} ({time.time() - t:.0f}s)", flush=True)
            if rc:
                return ("teacher", k), rc, time.time() - t
        return ("teacher", k), 0, time.time() - t

    def student(job):
        n, k = job
        out = student_dir(runs, n, k)
        if not (teacher_dir(runs, k) / "metrics.json").exists():
            return (n, k), -1, 0.0
        return (n, k), *run_job([py, "-m", "src.distill", "--seed", str(k), "--student", n, "--teacher",
                                 str(teacher_dir(runs, k)), "--out", str(out), *common_flags(a)], out)

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for job, rc, dt in ex.map(teacher, teachers):
            print(f"teacher seed{job[1]}: rc {rc} ({dt:.0f}s, elapsed {time.time() - t0:.0f}s)", flush=True)
        for job, rc, dt in ex.map(student, students):
            msg = "teacher missing" if rc == -1 else f"rc {rc}"
            print(f"student {job[0]} seed{job[1]}: {msg} ({dt:.0f}s, elapsed {time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- collection
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def per_subject_npz(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
    """``{subject: (y_true, logits)}`` from a ``predictions.npz`` / ``int8_predictions.npz``."""
    if not path.exists():
        return None
    z = np.load(path)
    return {int(s): (z[f"y_true_{int(s)}"], z[f"logits_{int(s)}"]) for s in z["subjects"]}


def _cell() -> dict:
    return {"float": [], "int8": [], "confusion": [], "confusion_int8": [], "seeds": []}


def collect(runs: Path, seeds, students, subjects, members=MEMBERS) -> tuple[dict, dict]:
    """``table[tag][subject] = {"float": [kpis per seed], "int8": [...], "confusion": [...]}`` and run facts.
    v4 teachers with a self-training round also fill ``round0_{member}`` / ``round0_ensemble``."""
    members = tuple(members)
    teacher_tags = (*members, "ensemble")
    tags = [*teacher_tags, *(f"student_{n}" for n in students), *(f"round0_{t}" for t in teacher_tags)]
    table = {t: {s: _cell() for s in subjects} for t in tags}
    facts: dict = {"teacher": {}, "student": {}}
    for k in seeds:
        m = load_json(teacher_dir(runs, k) / "metrics.json")
        if m is None:
            continue
        facts["teacher"][k] = m
        for prefix, block in (("", m), ("round0_", m.get("round0"))):
            if not block:
                continue
            for tag in teacher_tags:
                src = block["ensemble"] if tag == "ensemble" else block["members"].get(tag)
                if src is None:
                    continue
                for s in subjects:
                    if str(s) in src["test"]:
                        c = table[prefix + tag][s]
                        c["float"].append(src["test"][str(s)])
                        c["confusion"].append(np.array(src["confusion"][str(s)]))
                        c["seeds"].append(k)
    for n in students:
        tag = f"student_{n}"
        for k in seeds:
            d = student_dir(runs, n, k)
            m = load_json(d / "metrics.json")
            if m is None:
                continue
            facts["student"][(n, k)] = m
            q = per_subject_npz(d / "int8_predictions.npz")
            for s in subjects:
                c = table[tag][s]
                if str(s) in m["test"]:
                    c["float"].append(m["test"][str(s)]); c["confusion"].append(np.array(m["confusion"][str(s)]))
                    c["seeds"].append(k)
                if q is not None and s in q:
                    c["int8"].append(compute_kpis(*q[s])); c["confusion_int8"].append(confusion(*q[s]))
    return table, facts


# ----------------------------------------------------------------------------- params and MACs
def count_macs(model: torch.nn.Module, input_shape: tuple[int, ...]) -> int:
    """Multiply-accumulates of conv + linear layers for one window (as ``src.export.count_macs``)."""
    try:
        from src.export import count_macs as export_count_macs

        if "input_shape" in inspect.signature(export_count_macs).parameters:
            return int(export_count_macs(model, input_shape=input_shape)[0])
    except Exception:
        pass
    total = 0

    def hook(mod, inp, out):
        nonlocal total
        if isinstance(mod, torch.nn.Conv2d):
            total += out.numel() * mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
        else:
            total += mod.in_features * mod.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        model.eval()(torch.zeros(input_shape))
    for h in handles:
        h.remove()
    return int(total)


def model_costs(cfg: dict, students) -> dict[str, tuple[int, int]]:
    """``tag -> (params, MACs)`` for one ``(1, 1, 22, window)`` window."""
    from src.distill import build_student

    from src.models.riemann import n_coefficients

    shape = (1, 1, 22, int(cfg["window"]))
    costs = {}
    names = member_names(cfg)
    for name in names:
        if name in NON_TORCH:  # logistic-regression coefficients; covariance + matrix log, not conv MACs
            costs[name] = (n_coefficients(22), 0)
            continue
        m = build_member(name, cfg)
        costs[name] = (sum(p.numel() for p in m.parameters()), count_macs(m, shape))
    costs["ensemble"] = tuple(sum(costs[n][i] for n in names) for i in (0, 1))
    for n in students:
        # variants such as wide32_r0 / wide32_ft use their base student's widths
        m = build_student(n if n in cfg["student"] else n.rsplit("_", 1)[0], cfg)
        costs[f"student_{n}"] = (sum(p.numel() for p in m.parameters()), count_macs(m, shape))
    return costs


V01_LABEL = "*v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)*"


def v01_row() -> dict | None:
    """v0.1 LOSO TinyEEGNet (250 Hz, 500 samples, trained on 8 of the 2a subjects) from results/summary.json."""
    s = load_json(RESULTS / "summary.json")
    if not s or "loso" not in s.get("rows", {}):
        return None
    rows = s["rows"]["loso"]
    out = {}
    for prec in ("float", "int8"):
        per = {int(k): v[prec] for k, v in rows.items() if v.get(prec)}
        out[prec] = per
    return out


# ----------------------------------------------------------------------------- report
def summary_rows(students, table, members=MEMBERS) -> list[tuple[str, str, str]]:
    """``(label, tag, precision)`` rows of the summary table."""
    rows = [(MEMBER_LABELS[m], m, "float") for m in members]
    rows.append(("Teacher ensemble (mean softmax)", "ensemble", "float"))
    for n in students:
        tag = f"student_{n}"
        rows.append((f"Student {n}", tag, "float"))
        if any(table[tag][s]["int8"] for s in table[tag]):
            rows.append((f"Student {n}", tag, "int8"))
    return rows


def select_student(table, students, subjects, costs) -> tuple[str | None, str]:
    """Highest mean int8 accuracy over subjects on the **test block**, tie-break on MACs; float when no
    int8 exists. This ranks variants for the report (exploratory): a variant picked from this ranking
    has an optimistic test estimate (selection among the candidates). Only the seed within a variant
    is selected without the test set (``export_seed``, validation KL)."""
    for prec in ("int8", "float"):
        cand = []
        for n in students:
            m, _, cnt = across_subjects(table, f"student_{n}", prec, subjects, "accuracy")
            if cnt:
                cand.append((-round(m, 6), costs[f"student_{n}"][1], n))
        if cand:
            return sorted(cand)[0][2], prec
    return None, ""


def export_seed(facts, name: str, seeds) -> int | None:
    """Seed of the student run with the lowest best validation KL to the teacher (``best_val_kl``,
    the distillation selection criterion; never the test set)."""
    kls = {k: facts["student"][(name, k)]["best_val_kl"] for k in seeds if (name, k) in facts["student"]}
    return min(kls, key=kls.get) if kls else None


def write_report(table, facts, cfg, seeds, students, subjects, costs, export_subject: int = 3,
                 runs: Path = RUNS_V2, compare: Path | None = None, compare_seeds=(0, 1, 2)) -> str:
    L = []
    runs = Path(runs)
    runs_shown = runs.relative_to(ROOT) if runs.is_relative_to(ROOT) else runs
    W, off = int(cfg["window"]), int(cfg["crop_offset"])
    fs = cfg.get("fs_model", 125)
    start = cfg.get("window_start", 0.5 * fs) / fs  # canonical window start after the cue, s
    version = str(cfg.get("version", "v2"))
    val_cfg = (cfg.get("validation") or {}).get("val_subjects")
    L.append(f"# Reproduce_Baselines {version} results: teacher ensemble distilled into TinyEEGNet at 125 Hz\n")
    L.append("## 1. Setup\n")
    meta = next(iter(facts["teacher"].values()), {}).get("meta", {}) if facts["teacher"] else {}
    L.append(f"Training data: Cho2017 + Lee2019_MI (left vs right hand), 22 channels in the BCI IV 2a order, "
             f"common average reference over the 22, {fs} Hz, causal Butterworth 7–30 Hz, Euclidean Alignment per "
             f"subject-session, one global `input_scale`. Training windows {cfg.get('train_window', W)} samples with "
             f"random {W}-sample crops for the students; teachers train on the canonical crop at offset {off} "
             f"(cue + {start:.3f} s to cue + {start + W / fs:.3f} s). "
             + (f"Validation = whole held-out training subjects {json.dumps(val_cfg)} (cross-subject; early "
                "stopping and checkpoint selection only). " if val_cfg else
                "Validation = stratified 10 % of every training subject-session (early stopping only). ")
             + "**Test: BCI IV 2a (MOABB `BNCI2014_001`) subjects "
             f"{subjects[0]}–{subjects[-1]}, session 2 trials 25–144 (120 per subject), never trained on.** "
             "Session 2 trials 1–24 of every test subject give its EA matrix and enter the distillation stream "
             "with teacher soft labels only (their labels are never used). "
             f"Seeds {list(seeds)}; each subject's value is the mean over seeds, ± is the std across subjects. "
             "KPIs: accuracy, ROC-AUC, macro F1, AUPRC (positive class = right hand), Cohen's kappa.\n")
    kd = cfg.get("distill", {})
    L.append(f"Distillation: loss = {1 - kd.get('alpha', 0):.2g} CE + {kd.get('alpha', 0)} T² KL, T = "
             f"{kd.get('temperature')}, noise std {kd.get('noise_std')}, channel dropout {kd.get('channel_dropout')}; "
             "the teacher scores each augmented batch on the fly.\n")
    try:
        vers = {p: pkg_version(p) for p in ("torch", "moabb", "mne", "scikit-learn", "scipy", "numpy")}
    except Exception:
        vers = {"torch": torch.__version__}
    L.append("Environment: Python " + platform.python_version() + ", " + ", ".join(f"{k} {v}" for k, v in vers.items())
             + f"; git commit `{git_commit()}`; generated {time.strftime('%Y-%m-%d %H:%M')}.\n")

    # summary
    names = member_names(cfg)
    rows = summary_rows(students, table, names)
    L.append(f"## 2. Summary (mean ± std across the {len(subjects)} 2a subjects)\n")
    L.append("| Model | Precision | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | Params | MACs | n subj | n seeds |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, tag, prec in rows:
        cells = [fmt_pm(*across_subjects(table, tag, prec, subjects, k)[:2], pct=(k == "accuracy")) for k in KPI_KEYS]
        n = across_subjects(table, tag, prec, subjects, "accuracy")[2]
        n_seeds = max((len(table[tag][s][prec]) for s in subjects), default=0)
        p, macs = costs[tag]
        L.append(f"| {label} | {prec} | " + " | ".join(cells) + f" | {p:,} | {macs / 1e6:.2f} M | {n} | {n_seeds} |")
    v01 = v01_row()
    if v01:
        from src.models.tinyeegnet import TinyEEGNet

        ref = TinyEEGNet()
        p, macs = sum(q.numel() for q in ref.parameters()), count_macs(ref, (1, 1, 22, 500))
        for prec in ("float", "int8"):
            per = v01[prec]
            if not per:
                continue
            cells = []
            for k in KPI_KEYS:
                vals = [per[s][k] for s in per]
                cells.append(fmt_pm(float(np.mean(vals)), float(np.std(vals)), pct=(k == "accuracy")))
            L.append(f"| {V01_LABEL} | {prec} | " + " | ".join(cells)
                     + f" | {p:,} | {macs / 1e6:.2f} M | {len(per)} | 3 |")
        L.append("\nThe v0.1 row trains on the other 8 BCI IV 2a subjects (leave-one-subject-out) at 250 Hz with "
                 "500-sample windows; it is shown for context only (`results/results.md`).\n")
    else:
        L.append("")
    L.append("MACs = multiply-accumulates of the conv and linear layers for one window; params include BatchNorm. "
             f"Ensemble params/MACs are the sum of its {len(names)} members (offline only)"
             + ("; the Riemannian member's params are its logistic-regression coefficients and its MACs are "
                "not counted (covariance + matrix logarithm)" if any(n in NON_TORCH for n in names) else "")
             + ".\n")

    # per-subject accuracy
    L.append("## 3. Per-subject accuracy (%)\n")
    L.append("| Model | Precision | " + " | ".join(f"A0{s}" for s in subjects) + " | Mean |")
    L.append("|---|---|" + "---|" * (len(subjects) + 1))
    for label, tag, prec in rows:
        cells = [fmt(subject_mean(table[tag][s][prec], "accuracy"), True) for s in subjects]
        L.append(f"| {label} | {prec} | " + " | ".join(cells)
                 + f" | {fmt(across_subjects(table, tag, prec, subjects, 'accuracy')[0], True)} |")
    for prec in ("float", "int8"):
        per = v01[prec] if v01 else {}
        if not per:
            continue
        cells = [fmt(per[s]["accuracy"], True) if s in per else "n/a" for s in subjects]
        mean = fmt(float(np.mean([per[s]["accuracy"] for s in subjects if s in per])), True)
        L.append(f"| {V01_LABEL} | {prec} | " + " | ".join(cells) + f" | {mean} |")
    if v01:
        L.append("\nv0.1 rows: different protocol (LOSO on BCI IV 2a, 250 Hz, 500 samples), context only.")
    L.append("")

    # shipped student
    chosen, chosen_prec = select_student(table, students, subjects, costs)
    n_cand = len(((cfg.get("report") or {}).get("selection_candidates")) or students)
    L.append("## 4. Shipped student\n" if version == "v2" else
             "## 4. Variant comparison on the test block (exploratory)\n")
    if chosen is None:
        L.append("pending (no finished student runs).\n")
    elif version == "v2":
        tag = f"student_{chosen}"
        k_exp = export_seed(facts, chosen, seeds)
        L.append(f"Selected: **student {chosen}** (highest mean {chosen_prec} accuracy across subjects, tie-break on "
                 f"MACs{'; int8 results pending, selected on float' if chosen_prec == 'float' else ''}). Export subject "
                 f"A0{export_subject}, seed {k_exp} (lowest best validation KL to the teacher: "
                 + (f"{facts['student'][(chosen, k_exp)]['best_val_kl']:.4f}" if k_exp is not None else "n/a")
                 + f"), checkpoint `{runs_shown}/student_{chosen}_seed{k_exp}/model.pt`.\n")
    else:
        tag = f"student_{chosen}"
        k_exp = export_seed(facts, chosen, seeds)
        L.append(f"Variant comparison on the test block (exploratory; the shipped variant was chosen by the team from "
                 f"this table, which makes its test estimate optimistic by selection among {n_cand} candidates); the "
                 f"seed within the variant is chosen by validation KL. Highest mean {chosen_prec} test accuracy in this "
                 f"run: **student {chosen}**; its lowest-validation-KL seed is {k_exp} ("
                 + (f"{facts['student'][(chosen, k_exp)]['best_val_kl']:.4f}" if k_exp is not None else "n/a")
                 + f"), checkpoint `{runs_shown}/student_{chosen}_seed{k_exp}/model.pt`, export subject "
                 f"A0{export_subject}. See the selection note.\n")
        L.append("Per-subject KPIs, float (mean ± std over seeds):\n")
        L.append("| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | best epochs |")
        L.append("|---|---|---|---|---|---|---|")
        epochs = ", ".join(str(facts["student"][(chosen, k)]["best_epoch"]) for k in seeds if (chosen, k) in facts["student"])
        for s in subjects:
            c = table[tag][s]["float"]
            cells = [fmt_pm(subject_mean(c, k), subject_std(c, k), pct=(k == "accuracy")) for k in KPI_KEYS]
            L.append(f"| A0{s}{' (export)' if s == export_subject else ''} | " + " | ".join(cells) + f" | {epochs} |")
        cells = [fmt_pm(*across_subjects(table, tag, "float", subjects, k)[:2], pct=(k == "accuracy")) for k in KPI_KEYS]
        L.append("| **mean ± std** | " + " | ".join(cells) + " | |\n")

    # int8 minus float
    L.append("## 5. Int8 minus float accuracy (points)\n")
    have_int8 = [n for n in students if any(table[f"student_{n}"][s]["int8"] for s in subjects)]
    if not have_int8:
        L.append("pending (no `int8_predictions.npz` in the student run directories yet).\n")
    else:
        L.append("| Student | " + " | ".join(f"A0{s}" for s in subjects) + " | Mean |")
        L.append("|---|" + "---|" * (len(subjects) + 1))
        for n in have_int8:
            c = table[f"student_{n}"]
            deltas = [subject_mean(c[s]["int8"], "accuracy") - subject_mean(c[s]["float"], "accuracy") for s in subjects]
            cells = ["pending" if d != d else f"{100 * d:+.1f}" for d in deltas]
            valid = [d for d in deltas if d == d]
            mean = f"{100 * np.mean(valid):+.1f}" if valid else "pending"
            L.append(f"| {n} | " + " | ".join(cells) + f" | {mean} |")
        L.append("")

    # training facts
    L.append("## 6. Training facts\n")
    if facts["teacher"]:
        m0 = next(iter(facts["teacher"].values()))
        L.append(f"Trials: train {m0['n_train']:,}, validation {m0['n_val']:,}, 2a calibration {m0['n_calib']:,} "
                 f"(unlabelled), 2a test {m0['n_test']:,}. Data meta: `{json.dumps(meta)}`.\n")
        L.append("Teachers stop and select on the validation cross-entropy, students on the validation KL to the "
                 "teacher ensemble (the combined distillation loss at that epoch in brackets).\n")
        L.append("| Run | Seed | Best epochs (per model) | Best val CE (teachers) / val KL (students) | Wall (s) |")
        L.append("|---|---|---|---|---|")
        for k, m in sorted(facts["teacher"].items()):
            be = ", ".join(f"{MEMBER_LABELS[n]} {m['members'][n]['best_epoch']}" for n in names if n in m["members"])
            vl = ", ".join(f"{m['members'][n]['best_val_loss']:.4f}" for n in names if n in m["members"])
            L.append(f"| teacher | {k} | {be} | {vl} | {m['wall_time_s']:.0f} |")
        for (n, k), m in sorted(facts["student"].items()):
            L.append(f"| student {n} | {k} | {m['best_epoch']} | {m['best_val_kl']:.4f} ({m['best_val_loss']:.4f}) "
                     f"| {m['wall_time_s']:.0f} |")
        L.append("")
    else:
        L.append("pending (no finished teacher runs).\n")
    L.append(f"Confusion matrices per subject (rows = true left/right, columns = predicted, summed over seeds): "
             f"`results/confusion_{version}/A01.md` … `A0{subjects[-1]}.md`.\n")
    if version.startswith("v4"):
        L += v4_sections(table, facts, cfg, students, subjects, compare, compare_seeds)
    elif version != "v2":
        L += v3_sections(table, facts, cfg, students, subjects, compare, compare_seeds)
    return "\n".join(L)


def v3_sections(table, facts, cfg, students, subjects, compare: Path | None, compare_seeds) -> list[str]:
    """Quality-gate counts, cross-subject validation accuracy per model and the per-subject delta
    against a comparison run directory (v2)."""
    L = []
    m0 = next(iter(facts["teacher"].values()), None)
    meta = (m0 or {}).get("meta", {})
    L.append("## 7. Signal-quality gate (training sources)\n")
    q = meta.get("quality")
    if not q:
        L.append("not configured.\n")
    else:
        c = q.get("config", {})
        L.append(f"Per trial: peak |x − mean| > {c.get('ptp_uv')} µV (`ptp`), whitened-covariance distance "
                 f"z > {c.get('potato_z')} within the subject-session (`potato`, 2 iterations), channel std < "
                 f"{c.get('flat_uv')} µV (`flat`). A subject-session is dropped when more than "
                 f"{c.get('max_flagged_fraction')} of its trials are flagged or a channel is flat over the whole "
                 "session; kept sessions keep all trials.\n")
        L.append("| Dataset | Sessions | Dropped | Trials | ptp | potato | flat | any (rate) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for ds, d in q.get("datasets", {}).items():
            k = d["counts"]
            L.append(f"| {ds} | {d['sessions']} | {d['dropped']} | {d['trials']:,} | {k['ptp']} | {k['potato']} | "
                     f"{k['flat']} | {k['any']} ({100 * d['flag_rates']['any']:.1f} %) |")
        L.append("")
        if q.get("dropped"):
            L.append("Dropped: " + "; ".join(f"`{d['session']}` ({d['reason']})" for d in q["dropped"]) + "\n")
        else:
            L.append("Dropped sessions: none.\n")
        if q.get("calib_2a"):
            L.append("BCI IV 2a calibration trials (session 2 trials 1–24, logged only, never gated), flagged per "
                     "subject (any): " + ", ".join(f"A0{s} {v['any']}" for s, v in q["calib_2a"].items()) + ".\n")
    L.append("## 8. Cross-subject validation accuracy\n")
    v = meta.get("validation")
    if v:
        L.append(f"Validation subjects `{json.dumps(v.get('val_subjects'))}` ({len(v.get('val_sessions', []))} "
                 "subject-sessions); value at the selected epoch (EMA weights).\n")
    L.append("| Model | Seed | Val accuracy (%) | Val loss / KL | Best epoch | Epochs run |")
    L.append("|---|---|---|---|---|---|")
    for k, m in sorted(facts["teacher"].items()):
        for n in MEMBERS:
            mm = m["members"][n]
            L.append(f"| {MEMBER_LABELS[n]} | {k} | {100 * mm['best_val_acc']:.1f} | {mm['best_val_loss']:.4f} | "
                     f"{mm['best_epoch']} | {mm['epochs_run']} |")
        e = m["ensemble"]
        L.append(f"| Teacher ensemble | {k} | {100 * e['val_acc']:.1f} | {e['val_loss']:.4f} | | |")
    for (n, k), m in sorted(facts["student"].items()):
        L.append(f"| Student {n} | {k} | {100 * m['best_val_acc']:.1f} | KL {m['best_val_kl']:.4f} | "
                 f"{m['best_epoch']} | {m['epochs_run']} |")
    L.append("")
    L += compare_section(table, students, subjects, compare, compare_seeds, 9)
    return L


def v4_sections(table, facts, cfg, students, subjects, compare: Path | None, compare_seeds) -> list[str]:
    """Pseudo-label table, ensemble before/after self-training, per-member KPIs per subject, the
    per-member learning-rate choice and the per-subject delta against the v2 runs."""
    L = []
    names = member_names(cfg)
    st = cfg.get("self_training") or {}
    L.append("## 7. Self-training on the unlabelled target session\n")
    L.append(f"Round 0: the {len(names)}-member ensemble ({', '.join(MEMBER_LABELS[n] for n in names)}) is trained on "
             f"the training split only and scores {st.get('dataset', 'BNCI2014_001')} session {st.get('session', 1)} "
             "(left/right trials of every test subject, EA from that session's own trials; its labels are never read "
             f"for training). Windows with ensemble max probability ≥ {st.get('min_conf', 0.8)} get the argmax as a "
             "hard label (weight 1) and round 1 retrains every member from scratch on train + pseudo-labelled windows "
             "(same recipe and validation split). Agreement columns compare with the true session-1 labels and are "
             "computed afterwards for this report only. The session-2 test block (trials 25–144) is not involved.\n")
    for k, m in sorted(facts["teacher"].items()):
        sti = m.get("self_training")
        if not sti:
            L.append(f"Seed {k}: no self-training record.\n")
            continue
        cnt, agr = sti["counts"], sti.get("agreement_report_only", {})
        L.append(f"Seed {k}:\n")
        if cnt.get("mode") == "soft":
            L.append(f"Soft mode: round 1 uses **all** {cnt['n']:,} target windows with the round-0 ensemble probability "
                     "as target, KL weighted by w = 2·|p(right) − 0.5| (Riemannian member: argmax label, sample weight "
                     f"w); total weight {cnt.get('w_sum', float('nan')):.1f}. The 'Kept' columns show how many windows "
                     f"pass max prob ≥ {st.get('min_conf', 0.8)} (the hard variant's set), for comparison.\n")
            L.append("| Subject | Trials | Mean w | Weight sum | ≥ threshold | Argmax left / right (kept) | "
                     "Weighted agreement, all (%) | Agreement of ≥ threshold (%) | Round-0 ensemble on all trials (%) |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for s in subjects:
                c = cnt["per_subject"].get(str(s))
                a = (agr.get("per_subject") or {}).get(str(s), {})
                if c is None:
                    continue
                L.append(f"| A0{s} | {c['n']} | {c.get('w_mean', float('nan')):.3f} | {c.get('w_sum', float('nan')):.1f} | "
                         f"{c['kept']} | {c['pseudo_left']} / {c['pseudo_right']} | "
                         f"{fmt(a.get('weighted_agreement_all', float('nan')), True)} | "
                         f"{fmt(a.get('agreement_kept', float('nan')), True)} | {fmt(a.get('accuracy_all', float('nan')), True)} |")
            L.append(f"| **all** | {cnt['n']} | | {cnt.get('w_sum', float('nan')):.1f} | {cnt['kept']} | | "
                     f"{fmt(agr.get('weighted_agreement_all', float('nan')), True)} | "
                     f"{fmt(agr.get('agreement_kept', float('nan')), True)} | {fmt(agr.get('accuracy_all', float('nan')), True)} |\n")
            L.append(f"Round-1 training set: {sti.get('n_train_round1', 'n/a'):,} windows "
                     f"({m['n_train']:,} labelled + {cnt['n']:,} soft-labelled).\n")
            continue
        L.append("| Subject | Trials | Kept | Kept (%) | Pseudo left / right | Mean conf | Agreement of kept (%) "
                 "| Round-0 ensemble on all trials (%) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for s in subjects:
            c = cnt["per_subject"].get(str(s))
            a = (agr.get("per_subject") or {}).get(str(s), {})
            if c is None:
                continue
            L.append(f"| A0{s} | {c['n']} | {c['kept']} | {100 * c['kept_frac']:.1f} | {c['pseudo_left']} / "
                     f"{c['pseudo_right']} | {c['mean_conf']:.3f} | {fmt(a.get('agreement_kept', float('nan')), True)} "
                     f"| {fmt(a.get('accuracy_all', float('nan')), True)} |")
        L.append(f"| **all** | {cnt['n']} | {cnt['kept']} | {100 * cnt['kept_frac']:.1f} | | | "
                 f"{fmt(agr.get('agreement_kept', float('nan')), True)} | {fmt(agr.get('accuracy_all', float('nan')), True)} |\n")
        L.append(f"Round-1 training set: {sti.get('n_train_round1', 'n/a'):,} windows "
                 f"({m['n_train']:,} labelled + {cnt['kept']:,} pseudo-labelled).\n")

    L.append("## 8. Teachers before and after self-training (2a test block, accuracy %)\n")
    L.append("| Model | Round | " + " | ".join(f"A0{s}" for s in subjects) + " | Mean | ROC-AUC | Kappa |")
    L.append("|---|---|" + "---|" * (len(subjects) + 3))
    for tag in (*names, "ensemble"):
        label = "Teacher ensemble" if tag == "ensemble" else MEMBER_LABELS[tag]
        for rlabel, t in (("0 (before)", f"round0_{tag}"), ("1 (after)", tag)):
            if t not in table or not any(table[t][s]["float"] for s in subjects):
                continue
            cells = [fmt(subject_mean(table[t][s]["float"], "accuracy"), True) for s in subjects]
            mean = across_subjects(table, t, "float", subjects, "accuracy")[0]
            auc = across_subjects(table, t, "float", subjects, "roc_auc")[0]
            kap = across_subjects(table, t, "float", subjects, "kappa")[0]
            L.append(f"| {label} | {rlabel} | " + " | ".join(cells) + f" | {fmt(mean, True)} | {fmt(auc)} | {fmt(kap)} |")
    L.append("")

    L.append("## 9. Per-member KPIs per subject (final teachers, float)\n")
    for tag in (*names, "ensemble"):
        label = "Teacher ensemble" if tag == "ensemble" else MEMBER_LABELS[tag]
        if not any(table[tag][s]["float"] for s in subjects):
            continue
        L.append(f"**{label}**\n")
        L.append("| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |")
        L.append("|---|---|---|---|---|---|")
        for s in subjects:
            c = table[tag][s]["float"]
            L.append(f"| A0{s} | " + " | ".join(fmt(subject_mean(c, k), k == "accuracy") for k in KPI_KEYS) + " |")
        L.append("| **mean** | " + " | ".join(
            fmt(across_subjects(table, tag, "float", subjects, k)[0], k == "accuracy") for k in KPI_KEYS) + " |\n")

    L.append("## 10. Per-member learning rate and validation\n")
    L.append("| Seed | Member | Round | lr | Best epoch | Epochs run | Best val loss | Val acc (%) | lr choice |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, m in sorted(facts["teacher"].items()):
        for rlabel, block in (("0", m.get("round0")), ("1", m)):
            if not block:
                continue
            for n in names:
                mm = block["members"].get(n)
                if mm is None:
                    continue
                ch = mm.get("lr_choice")
                chs = (f"tried {ch['tried_lr']} (val {ch['tried_val_loss']:.4f}) vs reference {ch['reference_val_loss']:.4f}"
                       f" → {ch['kept_lr']}" if ch else "")
                L.append(f"| {k} | {MEMBER_LABELS[n]} | {rlabel} | {mm.get('lr', m.get('lr', '')) if n not in NON_TORCH else '—'} "
                         f"| {mm.get('best_epoch')} | {mm.get('epochs_run')} | {mm['best_val_loss']:.4f} | "
                         f"{100 * mm.get('best_val_acc', float('nan')):.1f} | {chs} |")
    L.append("")
    L += compare_section(table, students, subjects, compare, compare_seeds, 11, names)
    if compare is not None and Path(compare).is_dir() and list(compare_seeds) != [0]:
        L += compare_section(table, students, subjects, compare, [0], 12, names)
    L += three_way_section(table, facts, cfg, students, subjects, 13)
    L += selection_note(cfg, subjects, 14)
    return L


def _int8_mean(run: Path) -> float:
    q = per_subject_npz(Path(run) / "int8_predictions.npz")
    return float(np.mean([compute_kpis(*q[s])["accuracy"] for s in sorted(q)])) if q else float("nan")


def selection_note(cfg, subjects, number: int) -> list[str]:
    """``report.selection_candidates`` (label list) and ``report.paired`` ({new, ref, seeds, label}:
    run-directory prefixes ``<prefix>_seed{K}``): disclosure of how the shipped variant was chosen and
    the per-seed paired int8 comparison of the shipped variant against its reference."""
    rc = cfg.get("report") or {}
    cands = rc.get("selection_candidates")
    if not cands:
        return []
    L = [f"## {number}. Selection note\n",
         "Variant comparison on the test block (exploratory; the shipped variant was chosen by the team from this "
         f"table, which makes its test estimate optimistic by selection among {len(cands)} candidates); the seed "
         "within the variant is chosen by validation KL.\n",
         "Candidates: " + "; ".join(f"({i}) {c}" for i, c in enumerate(cands, 1)) + ".\n"]
    pr = rc.get("paired")
    if pr:
        new = [_int8_mean(ROOT / f"{pr['new']}_seed{k}") for k in pr["seeds"]]
        ref = [_int8_mean(ROOT / f"{pr['ref']}_seed{k}") for k in pr["seeds"]]
        L.append(f"Paired per seed ({pr.get('label', 'shipped vs reference')}), int8 accuracy (%) over the 9 subjects:\n")
        L.append("| Seed | " + f"`{Path(pr['new']).parent.name}/{Path(pr['new']).name}` | "
                 + f"`{Path(pr['ref']).parent.name}/{Path(pr['ref']).name}` | Δ |")
        L.append("|---|---|---|---|")
        for k, a, b in zip(pr["seeds"], new, ref):
            L.append(f"| {k} | {fmt(a, True)} | {fmt(b, True)} | {100 * (a - b):+.1f} |")
        ok = [(a, b) for a, b in zip(new, ref) if a == a and b == b]
        if ok:
            na, nb = np.array(ok).T
            L.append(f"| **mean ± std (seeds)** | {100 * na.mean():.1f} ± {100 * na.std():.1f} | "
                     f"{100 * nb.mean():.1f} ± {100 * nb.std():.1f} | {100 * (na - nb).mean():+.1f} "
                     f"({int(np.sum(na > nb))}/{len(ok)} seeds better) |")
        L.append("")
    return L


def three_way_section(table, facts, cfg, students, subjects, number: int) -> list[str]:
    """v4soft: ensemble and wide32 int8 per subject for round 0, the hard-label run
    (``report.three_way.hard``) and this (soft) run, plus any fine-tuned student of this run."""
    tw = (cfg.get("report") or {}).get("three_way")
    if not tw:
        return []
    hard = Path(tw["hard"])
    hard = hard if hard.is_absolute() else ROOT / hard
    seeds = sorted(facts["teacher"]) or [0]
    names = member_names(cfg)
    ref_students = sorted({m.group(1) for d in hard.glob(f"student_*_seed{seeds[0]}")
                           if (m := re.fullmatch(rf"student_(.+)_seed{seeds[0]}", d.name))})
    ref, _ = collect(hard, seeds, ref_students, subjects, names)
    rows = [("Teacher ensemble, round 0 (before self-training)", table, "round0_ensemble", "float"),
            (f"Teacher ensemble, hard pseudo-labels (`{hard.name}`)", ref, "ensemble", "float"),
            ("Teacher ensemble, soft targets (this run)", table, "ensemble", "float")]
    for n in ref_students:
        for prec in ("float", "int8"):
            rows.append((f"Student {n}, " + ("round-0 teacher" if n.endswith("_r0") else "hard") + f" (`{hard.name}`)",
                         ref, f"student_{n}", prec))
    for n in students:
        for prec in ("float", "int8"):
            rows.append((f"Student {n}, " + ("fine-tuned v2 student (KD only)" if n.endswith("_ft") else "soft (this run)"),
                         table, f"student_{n}", prec))
    L = [f"## {number}. Round 0 vs hard pseudo-labels vs soft targets (2a test block, accuracy %)\n",
         "| Model | Precision | " + " | ".join(f"A0{s}" for s in subjects) + " | Mean |",
         "|---|---|" + "---|" * (len(subjects) + 1)]
    for label, tb, tag, prec in rows:
        if tag not in tb or not any(tb[tag][s][prec] for s in subjects):
            continue
        cells = [fmt(subject_mean(tb[tag][s][prec], "accuracy"), True) for s in subjects]
        L.append(f"| {label} | {prec} | " + " | ".join(cells)
                 + f" | {fmt(across_subjects(tb, tag, prec, subjects, 'accuracy')[0], True)} |")
    L.append("")
    return L


def compare_section(table, students, subjects, compare: Path | None, compare_seeds, number: int,
                    members=MEMBERS) -> list[str]:
    """Per-subject accuracy change against a comparison run directory (v2), for the rows both have."""
    L = []
    if compare is not None and Path(compare).is_dir():
        ref_table, _ = collect(Path(compare), compare_seeds, students, subjects)
        L.append(f"## {number}. Change against `{Path(compare).name}` (seeds {list(compare_seeds)}), accuracy points\n")
        L.append("Positive = this run is better. Reference = the comparison run's per-subject mean over its seeds.\n")
        L.append("| Model | Precision | " + " | ".join(f"A0{s}" for s in subjects) + " | Mean | Ref mean |")
        L.append("|---|---|" + "---|" * (len(subjects) + 2))
        for label, tag, prec in summary_rows(students, table, members):
            if tag not in ref_table:
                continue
            new_v = [subject_mean(table[tag][s][prec], "accuracy") for s in subjects]
            ref_v = [subject_mean(ref_table[tag][s][prec], "accuracy") for s in subjects]
            if all(r != r for r in ref_v):
                continue
            cells = [("n/a" if (a != a or b != b) else f"{100 * (a - b):+.1f}") for a, b in zip(new_v, ref_v)]
            ok = [(a, b) for a, b in zip(new_v, ref_v) if a == a and b == b]
            mean = f"{100 * (np.mean([a for a, _ in ok]) - np.mean([b for _, b in ok])):+.1f}" if ok else "n/a"
            ref_mean = f"{100 * np.mean([b for _, b in ok]):.1f}" if ok else "n/a"
            L.append(f"| {label} | {prec} | " + " | ".join(cells) + f" | {mean} | {ref_mean} |")
        L.append("")
    return L


def write_confusions(table, students, subjects, results: Path, version: str = "v2", members=MEMBERS) -> None:
    d = results / f"confusion_{version}"
    d.mkdir(parents=True, exist_ok=True)
    tags = [(MEMBER_LABELS[m], m) for m in members] + [("Teacher ensemble", "ensemble")] + \
           [(f"Student {n}", f"student_{n}") for n in students]
    for s in subjects:
        L = [f"# Confusion matrices {version}, subject A0{s}\n",
             "Rows = true (left, right), columns = predicted (left, right); summed over seeds.\n"]
        for label, tag in tags:
            for key, prec in (("confusion", "float"), ("confusion_int8", "int8")):
                cms = table[tag][s][key]
                if not cms:
                    continue
                cm = np.sum(cms, axis=0)
                L.append(f"## {label}, {prec}\n\n| | pred left | pred right |\n|---|---|---|\n"
                         f"| true left | {cm[0, 0]} | {cm[0, 1]} |\n| true right | {cm[1, 0]} | {cm[1, 1]} |\n")
        (d / f"A0{s}.md").write_text("\n".join(L))


def report(a: argparse.Namespace) -> Path:
    cfg = load_v2_config(a.config)
    runs = Path(a.runs_dir)
    try:
        subjects = subject_list(cfg["test"]["subjects"])
    except (KeyError, TypeError, ValueError):
        subjects = list(range(1, 10))
    names = member_names(cfg)
    table, facts = collect(runs, a.seeds, a.students, subjects, names)
    costs = model_costs(cfg, a.students)
    results = Path(a.results).parent
    results.mkdir(parents=True, exist_ok=True)
    Path(a.results).write_text(write_report(table, facts, cfg, a.seeds, a.students, subjects, costs, runs=runs,
                                            compare=a.compare, compare_seeds=a.compare_seeds))
    write_confusions(table, a.students, subjects, results, str(cfg.get("version", "v2")), names)
    print(f"wrote {a.results}")
    return Path(a.results)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--students", nargs="+", default=["wide32", "control"])
    p.add_argument("--skip-train", action="store_true", help="only rebuild the report from existing runs")
    p.add_argument("--config", type=Path, default=CONFIG_V2)
    p.add_argument("--threads", type=int, default=int(os.environ.get("RB_THREADS", "1")))
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=None, help="passed to every run (smoke)")
    p.add_argument("--subset-cho", type=int, default=None, help="passed to every run (smoke)")
    p.add_argument("--runs-dir", "--out", dest="runs_dir", type=Path, default=RUNS_V2,
                   help="run directory (default runs/v2)")
    p.add_argument("--results", type=Path, default=RESULTS / "results_v2.md")
    p.add_argument("--teacher-parallel", action="store_true",
                   help="train the three teacher members of a seed as three parallel processes")
    p.add_argument("--compare", type=Path, default=None,
                   help="run directory to compare against in the report (v3 sections; default runs/v2 for a "
                        "non-v2 config)")
    p.add_argument("--compare-seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args(argv)
    if a.compare is None:
        try:
            if str(load_v2_config(a.config).get("version", "v2")) != "v2":
                a.compare = RUNS_V2
        except OSError:
            pass
    return a


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    a.runs_dir, a.config, a.results = a.runs_dir.resolve(), a.config.resolve(), a.results.resolve()
    if not a.skip_train:
        train_all(a)
    report(a)


if __name__ == "__main__":
    main()
