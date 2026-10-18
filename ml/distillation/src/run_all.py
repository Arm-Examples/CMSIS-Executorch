"""Run every experiment and write ``results/results.md``.

    python -m src.run_all --jobs 6            # all 9 subjects x 3 seeds, resumes finished runs
    python -m src.run_all --report-only       # only rebuild results/ from runs/

Runs (one directory each under ``runs/``):
  loso_s{S}_seed{K}          TinyEEGNet+EA, LOSO (row B, headline)
  loso_eegnet_s{S}_seed{K}   EEGNet-ELU+EA, LOSO (reference, float only)
  per_subject_s{S}_seed{K}   TinyEEGNet, per-subject (row A)
  csp_lda_s{S}, csp_lda_ea_s{S}   CSP+LDA LOSO without / with EA (deterministic)

Int8 columns are read from ``runs/<run>/int8_predictions.npz`` (``y_true``, ``logits``
of the fake-quant model on the same test block; written by ``src.quantize``/``src.export``)
when present, otherwise marked pending.
"""

from __future__ import annotations

import argparse
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

from src.data.splits import CALIB_TRIALS, SUBJECTS, cached_subject, raw_loso_data
from src.eval import KPI_KEYS, compute_kpis, confusion
from src.models import csp_lda

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
SEEDS = [0, 1, 2]

NN_RUNS = {  # tag -> (mode, model)
    "loso": ("loso", "tinyeegnet"),
    "loso_eegnet": ("loso", "eegnet"),
    "per_subject": ("per-subject", "tinyeegnet"),
}
# (label, training, precision, source tag, int8?)
SUMMARY_ROWS = [
    ("CSP+LDA", "LOSO", "float", "csp_lda", False),
    ("CSP+LDA+EA", "LOSO", "float", "csp_lda_ea", False),
    ("EEGNet-ELU+EA", "LOSO", "float", "loso_eegnet", False),
    ("**TinyEEGNet+EA, row B**", "LOSO", "float", "loso", False),
    ("**TinyEEGNet+EA, row B**", "LOSO", "int8", "loso", True),
    ("TinyEEGNet, row A", "per-subject", "float", "per_subject", False),
    ("TinyEEGNet, row A", "per-subject", "int8", "per_subject", True),
]


def run_dir(tag: str, subject: int, seed: int | None = None) -> Path:
    return RUNS / (f"{tag}_s{subject}" if seed is None else f"{tag}_s{subject}_seed{seed}")


# ----------------------------------------------------------------------------- training
def nn_jobs(subjects, seeds, tags) -> list[tuple[str, int, int]]:
    jobs = []
    for tag in tags:
        for s in subjects:
            for k in seeds:
                if not (run_dir(tag, s, k) / "metrics.json").exists():
                    jobs.append((tag, s, k))
    return jobs


def run_nn(job: tuple[str, int, int], threads: int) -> tuple[tuple[str, int, int], int, float]:
    tag, s, k = job
    mode, model = NN_RUNS[tag]
    out = run_dir(tag, s, k)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.train", "--mode", mode, "--model", model, "--subject", str(s),
           "--seed", str(k), "--out", str(out), "--threads", str(threads)]
    t = time.time()
    with open(out / "stdout.log", "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    return job, rc, time.time() - t


def run_csp(subject: int) -> None:
    for tag, aligned in (("csp_lda", False), ("csp_lda_ea", True)):
        out = run_dir(tag, subject)
        if (out / "metrics.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        d = raw_loso_data(subject, aligned=aligned, loader=cached_subject)
        logits = csp_lda.fit_predict(d["train"][0], d["train"][1], d["test"][0])
        y = d["test"][1]
        np.savez(out / "predictions.npz", y_true=y, logits=logits, test_trials=d["test_trials"])
        metrics = {"mode": "loso", "subject": subject, "model": tag, "ea": aligned, "n_train": int(len(d["train"][1])),
                   "n_test": int(len(y)), "test": compute_kpis(y, logits), "confusion": confusion(y, logits).tolist(),
                   "wall_time_s": time.time() - t}
        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=1)
        print(f"{tag} subject {subject}: acc {metrics['test']['accuracy']:.3f} ({metrics['wall_time_s']:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- collection
def load_metrics(path: Path) -> dict | None:
    f = path / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def int8_kpis(path: Path) -> dict | None:
    f = path / "int8_predictions.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return compute_kpis(z["y_true"], z["logits"])


def collect(subjects, seeds) -> dict:
    """``table[tag][subject] = {"float": [kpis per seed], "int8": [...], "calib_s1": [...]}``."""
    table: dict[str, dict[int, dict]] = {}
    for tag in NN_RUNS:
        table[tag] = {}
        for s in subjects:
            cell = {"float": [], "int8": [], "calib_s1": [], "confusion": [], "wall": [], "epochs": []}
            for k in seeds:
                m = load_metrics(run_dir(tag, s, k))
                if m is None:
                    continue
                cell["float"].append(m["test"])
                cell["confusion"].append(np.array(m["confusion"]))
                cell["wall"].append(m.get("wall_time_s", float("nan")))
                cell["epochs"].append(m.get("best_epoch", -1))
                if "test_calib_session1" in m:
                    cell["calib_s1"].append(m["test_calib_session1"])
                q = int8_kpis(run_dir(tag, s, k))
                if q is not None:
                    cell["int8"].append(q)
            table[tag][s] = cell
    for tag in ("csp_lda", "csp_lda_ea"):
        table[tag] = {}
        for s in subjects:
            m = load_metrics(run_dir(tag, s))
            table[tag][s] = {"float": [m["test"]] if m else [], "int8": [], "calib_s1": [],
                             "confusion": [np.array(m["confusion"])] if m else [], "wall": [], "epochs": []}
    return table


def subject_mean(cell_list: list[dict], key: str) -> float:
    return float(np.mean([c[key] for c in cell_list])) if cell_list else float("nan")


def subject_std(cell_list: list[dict], key: str) -> float:
    return float(np.std([c[key] for c in cell_list])) if len(cell_list) > 1 else float("nan")


def across_subjects(table, tag, precision, subjects, key) -> tuple[float, float, int]:
    vals = [subject_mean(table[tag][s][precision], key) for s in subjects if table[tag][s][precision]]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def fmt(x: float, pct: bool = False) -> str:
    if x != x:
        return "pending"
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def fmt_pm(m: float, s: float, pct: bool = False) -> str:
    if m != m:
        return "pending"
    if s != s:
        return fmt(m, pct)
    return f"{fmt(m, pct)} ± {fmt(s, pct)}"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def write_report(table: dict, subjects, seeds, export_subject: int | None, export_seed_k: int | None = None) -> str:
    L = []
    L.append("# Reproduce_Baselines results\n")
    L.append("## 1. Setup\n")
    L.append(f"BCI Competition IV 2a (MOABB `BNCI2014_001`), left vs right hand, 22 EEG channels, 250 Hz, "
             f"causal Butterworth 7–30 Hz (order 4, `sosfilt`), window cue+0.5 s to cue+2.5 s (500 samples). "
             f"Test set for every method: target subject's session 2 trials {CALIB_TRIALS + 1}–144 "
             f"({144 - CALIB_TRIALS} trials/subject); trials 1–{CALIB_TRIALS} of session 2 are the unlabeled "
             f"Euclidean-Alignment calibration block. LOSO rows train on the other 8 subjects (both sessions, EA per "
             f"subject-session), early stopping on a stratified, seed-dependent 10 % of every source subject-session's trials; per-subject rows "
             f"train on the target's session 1 (80/20 stratified early-stop split). Seeds {seeds} per subject; "
             f"each subject's value is the mean over seeds, the ± in the summary is the std across the 9 subjects. "
             f"KPIs: accuracy, ROC-AUC, macro F1, AUPRC (positive class = right hand), Cohen's kappa; chance = "
             f"50 % / 0.5 / 0.5 / 0.5 / 0.\n")
    vers = {p: pkg_version(p) for p in ("torch", "executorch", "torchao", "moabb", "mne", "scikit-learn", "scipy", "numpy")}
    L.append("Environment: Python " + platform.python_version() + ", " + ", ".join(f"{k} {v}" for k, v in vers.items())
             + f"; git commit `{git_commit()}`; generated {time.strftime('%Y-%m-%d %H:%M')}.\n")

    # 1. summary
    L.append("## 2. Summary (mean ± std across 9 subjects)\n")
    L.append("| Method | Training | Precision | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | n subj |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for label, training, precision, tag, is_int8 in SUMMARY_ROWS:
        prec = "int8" if is_int8 else "float"
        cells = [fmt_pm(*across_subjects(table, tag, prec, subjects, k)[:2], pct=(k == "accuracy")) for k in KPI_KEYS]
        n = across_subjects(table, tag, prec, subjects, "accuracy")[2]
        L.append(f"| {label} | {training} | {precision} | " + " | ".join(cells) + f" | {n} |")
    L.append("\nPublished references (not comparable: different class count or protocol): BCI Competition IV 2a "
             "winner kappa 0.57 (4-class, within-subject; Tangermann et al. 2012, Review of the BCI Competition IV, "
             "Front. Neurosci., https://doi.org/10.3389/fnins.2012.00055); EEGNet 73.13 % on 2a (4-class, "
             "within-session; Yang et al. 2025, Sci. Data, https://doi.org/10.1038/s41597-025-04826-y, Table 4).\n")
    L.append("**Headline rows are the two bold TinyEEGNet+EA row-B rows (float and int8).**\n")

    # 2. per subject row B
    L.append("## 3. Per subject, row B (TinyEEGNet+EA, LOSO)\n")
    L.append("Values are mean ± std over seeds; Δ int8 = int8 − float accuracy in points.\n")
    L.append("| Subject | Acc float (%) | Acc int8 (%) | Δ int8 | ROC-AUC | F1 | AUPRC | Kappa | best epoch | wall/fold (s) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    accs_f, accs_q = [], []
    for s in subjects:
        c = table["loso"][s]
        af, aq = subject_mean(c["float"], "accuracy"), subject_mean(c["int8"], "accuracy")
        accs_f.append(af); accs_q.append(aq)
        delta = "pending" if aq != aq else f"{100 * (aq - af):+.1f}"
        mark = " (export)" if s == export_subject else ""
        row = [f"A0{s}{mark}", fmt_pm(af, subject_std(c["float"], "accuracy"), True),
               fmt_pm(aq, subject_std(c["int8"], "accuracy"), True), delta]
        row += [fmt_pm(subject_mean(c["float"], k), subject_std(c["float"], k)) for k in ("roc_auc", "f1", "auprc", "kappa")]
        row += [", ".join(str(e) for e in c["epochs"]) or "pending",
                f"{np.mean(c['wall']):.0f}" if c["wall"] else "pending"]
        L.append("| " + " | ".join(row) + " |")
    valid_f = [a for a in accs_f if a == a]
    valid_q = [a for a in accs_q if a == a]
    mean_row = ["**mean ± std**", fmt_pm(np.mean(valid_f), np.std(valid_f), True) if valid_f else "pending",
                fmt_pm(np.mean(valid_q), np.std(valid_q), True) if valid_q else "pending",
                f"{100 * (np.mean(valid_q) - np.mean([a for a, q in zip(accs_f, accs_q) if q == q])):+.1f}" if valid_q else "pending"]
    mean_row += [fmt_pm(*across_subjects(table, "loso", "float", subjects, k)[:2]) for k in ("roc_auc", "f1", "auprc", "kappa")]
    mean_row += ["", ""]
    L.append("| " + " | ".join(mean_row) + " |\n")

    # 3. per-subject accuracy across methods
    L.append("## 4. Per-subject accuracy (%) across methods\n")
    heads = [f"{lab.strip('*')} ({tr}, {pr})" for lab, tr, pr, _, _ in SUMMARY_ROWS]
    L.append("| Subject | " + " | ".join(heads) + " |")
    L.append("|---|" + "---|" * len(heads))
    for s in subjects:
        cells = []
        for _, _, _, tag, is_int8 in SUMMARY_ROWS:
            cells.append(fmt(subject_mean(table[tag][s]["int8" if is_int8 else "float"], "accuracy"), True))
        L.append(f"| A0{s} | " + " | ".join(cells) + " |")
    cells = [fmt(across_subjects(table, tag, "int8" if q else "float", subjects, "accuracy")[0], True)
             for _, _, _, tag, q in SUMMARY_ROWS]
    L.append("| **mean** | " + " | ".join(cells) + " |\n")

    # 4. EA calibration variant
    L.append("## 5. EA calibration variant (row B, float)\n")
    L.append("Same trained row-B models re-scored on the same 120 test trials; only the target's EA matrix changes: "
             f"default `A` from session 2 trials 1–{CALIB_TRIALS} vs `A` from all 144 trials of session 1 (labels unused "
             "in both; `predictions_calib_s1.npz` in each run directory).\n")
    L.append("| Subject | Acc (%) A from session 2 calib | Acc (%) A from session 1 | Kappa (s2 calib) | Kappa (s1) |")
    L.append("|---|---|---|---|---|")
    for s in subjects:
        c = table["loso"][s]
        L.append(f"| A0{s} | {fmt(subject_mean(c['float'], 'accuracy'), True)} | {fmt(subject_mean(c['calib_s1'], 'accuracy'), True)} "
                 f"| {fmt(subject_mean(c['float'], 'kappa'))} | {fmt(subject_mean(c['calib_s1'], 'kappa'))} |")
    m2 = across_subjects(table, "loso", "float", subjects, "accuracy"); m1 = across_subjects(table, "loso", "calib_s1", subjects, "accuracy")
    k2 = across_subjects(table, "loso", "float", subjects, "kappa"); k1 = across_subjects(table, "loso", "calib_s1", subjects, "kappa")
    L.append(f"| **mean ± std** | {fmt_pm(m2[0], m2[1], True)} | {fmt_pm(m1[0], m1[1], True)} | {fmt_pm(k2[0], k2[1])} | {fmt_pm(k1[0], k1[1])} |\n")

    # 5. deployment
    L.append("## 6. Deployment (export subject)\n")
    if export_subject is None:
        L.append("pending (no finished row-B runs).\n")
    else:
        c = table["loso"][export_subject]
        seed_m = load_metrics(run_dir("loso", export_subject, export_seed_k)) if export_seed_k is not None else None
        L.append(f"Export subject: **A0{export_subject}**, checkpoint `runs/loso_s{export_subject}_seed{export_seed_k}/model.pt` "
                 "(seed with the lowest best validation loss"
                 + (f": {seed_m['best_val_loss']:.4f}, float test accuracy {100 * seed_m['test']['accuracy']:.1f} %" if seed_m else "")
                 + f"). Mean over seeds: float {fmt(subject_mean(c['float'], 'accuracy'), True)} %, "
                 f"int8 {fmt(subject_mean(c['int8'], 'accuracy'), True)} %. TinyEEGNet: 1938 parameters.\n")
        dep = deployment_rows()
        if dep:
            L.append("Key deployment numbers (host fake-quant checks; from [deployment.md](deployment.md), operator detail "
                     "in [op_placement.md](op_placement.md), both written by `src.export`):\n")
            L.append("| Item | Value |\n|---|---|")
            for item, value in dep:
                L.append(f"| {item} | {value} |")
            L.append("\nFVP latency and arena size are added from Emulator_Deploy.\n")
        else:
            L.append("`results/deployment.md` / `results/op_placement.md` pending (written by `src.export`).\n")
    L.append("## 7. Confusion matrices and gates\n")
    gates = []
    if export_subject is not None:
        af = subject_mean(table["loso"][export_subject]["float"], "accuracy")
        aq = subject_mean(table["loso"][export_subject]["int8"], "accuracy")
        gates.append(("Export subject float accuracy ≥ 75 %", f"{100 * af:.1f} %", "pass" if af >= 0.75 else "FAIL"))
        gates.append(("Int8 accuracy drop ≤ 2 points (export subject)",
                      "pending" if aq != aq else f"{100 * (aq - af):+.1f}",
                      "pending" if aq != aq else ("pass" if aq >= af - 0.02 else "FAIL")))
    pte = ROOT / "artifacts" / "tinyeegnet_int8.pte"
    if pte.exists():
        size = pte.stat().st_size
        gates.append(("`.pte` < 100 KB", f"{size / 1024:.1f} KiB", "pass" if size < 100 * 1024 else "FAIL"))
    else:
        gates.append(("`.pte` < 100 KB", "`artifacts/tinyeegnet_int8.pte` missing", "pending"))
    opp = RESULTS / "op_placement.md"
    if opp.exists():
        txt = opp.read_text()
        m = re.search(r"All conv/linear on Ethos-U: \*\*(\w+)\*\*", txt)
        gates.append(("All conv/linear ops on Ethos-U", f"[op_placement.md](op_placement.md): {m.group(1) if m else 'unparsed'}",
                      "pass" if m and m.group(1).lower() == "yes" else ("FAIL" if m else "pending")))
    else:
        gates.append(("All conv/linear ops on Ethos-U", "`results/op_placement.md` missing", "pending"))
    L.append("| Gate | Value | Status |\n|---|---|---|")
    for g in gates:
        L.append(f"| {g[0]} | {g[1]} | {g[2]} |")
    L.append("\nConfusion matrices per subject and method (rows = true left/right, columns = predicted, summed over "
             "seeds): `results/confusion/A01.md` … `A09.md`. Row B, export subject, summed over seeds:\n")
    if export_subject is not None and table["loso"][export_subject]["confusion"]:
        cm = np.sum(table["loso"][export_subject]["confusion"], axis=0)
        L.append(f"| A0{export_subject} row B | pred left | pred right |\n|---|---|---|\n| true left | {cm[0, 0]} | {cm[0, 1]} |\n"
                 f"| true right | {cm[1, 0]} | {cm[1, 1]} |\n")
    return "\n".join(L)


DEPLOYMENT_ITEMS = [  # (label in results.md, row label in deployment.md, or None for the Vela summary SRAM line)
    ("Checkpoint", "Checkpoint"),
    ("Parameters", "Parameters"),
    ("MACs per window", "MACs per window"),
    ("`.pte` size", "`.pte` size"),
    ("Ops on Ethos-U / CPU", "Ops on Ethos-U / CPU"),
    ("Vela SRAM estimate", None),
    ("int8 accuracy, 20 test vectors", "int8 accuracy, 20 test vectors"),
    ("Max \\|logit\\| diff int8 vs float, 20 vectors", "Max \\|logit\\| diff int8 vs float, 20 vectors"),
]


def deployment_rows() -> list[tuple[str, str]]:
    """Pick the key rows out of ``results/deployment.md`` (table ``| Item | Value |`` + Vela summary)."""
    path = RESULTS / "deployment.md"
    if not path.exists():
        return []
    text = path.read_text()
    table = {}
    for line in text.splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]  # keep escaped \| in cells
        if len(cells) == 2:
            table[cells[0]] = cells[1]
    sram = re.search(r"Total SRAM used\s+([\d.]+\s*\w+)", text)
    rows = []
    for label, key in DEPLOYMENT_ITEMS:
        if key is None:
            value = sram.group(1) + " (Vela estimate, not an FVP measurement)" if sram else "pending"
        else:
            value = table.get(key, "pending")
        rows.append((label, value))
    return rows


def write_confusions(table: dict, subjects) -> None:
    d = RESULTS / "confusion"
    d.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        L = [f"# Confusion matrices, subject A0{s}\n", "Rows = true (left, right), columns = predicted (left, right); "
             "summed over seeds for the neural rows.\n"]
        for tag in ("loso", "loso_eegnet", "per_subject", "csp_lda", "csp_lda_ea"):
            cms = table[tag][s]["confusion"]
            if not cms:
                continue
            cm = np.sum(cms, axis=0)
            L.append(f"## {tag}\n\n| | pred left | pred right |\n|---|---|---|\n| true left | {cm[0,0]} | {cm[0,1]} |\n"
                     f"| true right | {cm[1,0]} | {cm[1,1]} |\n")
        (d / f"A0{s}.md").write_text("\n".join(L))


def write_summary_json(table: dict, subjects, seeds, export_subject) -> None:
    out = {"subjects": subjects, "seeds": seeds, "export_subject": export_subject, "rows": {}}
    for tag in table:
        out["rows"][tag] = {}
        for s in subjects:
            c = table[tag][s]
            out["rows"][tag][str(s)] = {
                "float": {k: subject_mean(c["float"], k) for k in KPI_KEYS} if c["float"] else None,
                "float_seed_std": {k: subject_std(c["float"], k) for k in KPI_KEYS} if c["float"] else None,
                "int8": {k: subject_mean(c["int8"], k) for k in KPI_KEYS} if c["int8"] else None,
                "calib_s1": {k: subject_mean(c["calib_s1"], k) for k in KPI_KEYS} if c["calib_s1"] else None,
                "n_seeds": len(c["float"]),
            }
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))


def export_seed(subject: int, seeds) -> int | None:
    """Seed of the row-B run with the lowest best validation loss (never the test set)."""
    losses = {}
    for k in seeds:
        m = load_metrics(run_dir("loso", subject, k))
        if m is not None:
            losses[k] = m["best_val_loss"]
    return min(losses, key=losses.get) if losses else None


def report(subjects, seeds, export_subject: int | None = None, export_seed_override: int | None = None) -> None:
    table = collect(subjects, seeds)
    means = {s: subject_mean(table["loso"][s]["float"], "accuracy") for s in subjects if table["loso"][s]["float"]}
    if export_subject is None:
        export_subject = max(means, key=means.get) if means else None
    seed = export_seed_override if export_seed_override is not None else (
        export_seed(export_subject, seeds) if export_subject is not None else None)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.md").write_text(write_report(table, subjects, seeds, export_subject, seed))
    write_confusions(table, subjects)
    write_summary_json(table, subjects, seeds, export_subject)
    if export_subject is not None and seed is not None:
        (RESULTS / "export_subject.json").write_text(json.dumps(
            {"subject": export_subject, "seed": seed, "run": str(run_dir("loso", export_subject, seed).relative_to(ROOT)),
             "selection": "subject: manager decision / highest mean row-B float accuracy; seed: lowest best validation loss",
             "mean_float_accuracy": means.get(export_subject)}, indent=1))
    print(f"wrote {RESULTS / 'results.md'} (export subject {export_subject}, seed {seed})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", default="all")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--tags", default=",".join(NN_RUNS), help="subset of " + ",".join(NN_RUNS))
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 3))
    p.add_argument("--threads", type=int, default=1, help="torch threads per training process (1 is fastest for this model)")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--skip-csp", action="store_true")
    p.add_argument("--export-subject", type=int, default=None, help="override the export subject (default: best row-B mean)")
    p.add_argument("--export-seed", type=int, default=None, help="override the export seed (default: lowest val loss)")
    a = p.parse_args(argv)
    subjects = SUBJECTS if a.subjects == "all" else [int(s) for s in a.subjects.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    tags = a.tags.split(",")
    if not a.report_only:
        for s in SUBJECTS:
            cached_subject(s)  # fill the cache once before parallel workers start
        if not a.skip_csp:
            for s in subjects:
                run_csp(s)
        jobs = nn_jobs(subjects, seeds, tags)
        print(f"{len(jobs)} training runs pending, {a.jobs} in parallel x {a.threads} threads", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            for job, rc, dt in ex.map(lambda j: run_nn(j, a.threads), jobs):
                m = load_metrics(run_dir(*job))
                acc = f"acc {m['test']['accuracy']:.3f} best ep {m['best_epoch']}" if m else "NO METRICS"
                print(f"{job[0]} s{job[1]} seed{job[2]}: rc {rc} {acc} ({dt:.0f}s, elapsed {time.time() - t0:.0f}s)", flush=True)
    report(subjects, seeds, a.export_subject, a.export_seed)


if __name__ == "__main__":
    main()
