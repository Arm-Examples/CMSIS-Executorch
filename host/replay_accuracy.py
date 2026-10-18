"""Host-side replay ground truth for all 9 subjects -> ``results/replay_accuracy.md``.

    python host/replay_accuracy.py [--subjects all] [--no-raw] [--no-oracle]

Per subject (session 2 test trials 24..143, default replay plan: 10 s lead-in, int16 stream):
* offline int8 accuracy (Reproduce_Baselines, float stream; ``int8_predictions.npz``),
* sim-board replay accuracy: ``bridge.py --board sim --fast`` through the real framing and the
  streaming pipeline, logits looked up from the offline predictions (UI/measure path check),
* int8 model on the int16-stream windows (``--sim-model int8``, i.e. what the board must
  output; needs torch + executorch) and its argmax agreement with the offline predictions,
* int8 window element mismatch between the int16 stream and the float stream (needs the raw
  GDF via Reproduce_Baselines; skipped with ``--no-raw``).
Also writes ``host/reference/expected_board_predictions.json`` (bridge.py ``match expected``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge  # noqa: E402
from common import MODEL_DESC, MODEL_TAG, REPO_DIR, load_offline_predictions, load_params  # noqa: E402
from drivers.replay import default_trials, load_replay, plan_session  # noqa: E402
from common import parse_trials  # noqa: E402
from pipeline_ref import PipelineRef  # noqa: E402

OUT_MD = REPO_DIR / "results" / "replay_accuracy.md"


def sim_run(subject: int, model: str, log: Path) -> list[dict]:
    if log.exists():
        log.unlink()
    bridge.run(["--subject", str(subject), "--board", "sim", "--fast", "--quiet", "--publish", "none",
                "--sim-model", model, "--log", str(log)])
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    return [r for r in recs if r.get("event") == "pred" and r.get("trial", -1) >= 0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--no-raw", action="store_true")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--out", default=str(OUT_MD))
    ap.add_argument("--tmp", default=None, help="directory for the sim JSONL logs")
    a = ap.parse_args(argv)
    subjects = list(range(1, 10)) if a.subjects == "all" else [int(s) for s in a.subjects.split(",")]
    params = load_params()
    off = load_offline_predictions()
    tmp_ctx = None
    if a.tmp:
        tmp = Path(a.tmp)
        tmp.mkdir(parents=True, exist_ok=True)
    else:  # sim JSONL logs never land in results/
        tmp_ctx = tempfile.TemporaryDirectory(prefix="replay_accuracy_")
        tmp = Path(tmp_ctx.name)
    oracle_ok = not a.no_oracle
    if oracle_ok:
        try:
            import int8_oracle  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            oracle_ok = False

    rows, expected = [], {}
    conf_board = np.zeros((2, 2), int)
    uv_scale = None
    for s in subjects:
        rep = load_replay(s)
        uv_scale = rep.uv_scale
        plan = plan_session(rep, parse_trials(default_trials()), 10.0)
        o = off[s]
        y = o["y_true"]
        off_cls = o["logits"].argmax(1)
        row = {"s": s, "n": len(y), "offline": float((off_cls == y).mean())}
        recs = sim_run(s, "offline", tmp / f".sim_A0{s}.jsonl")
        row["sim_n"] = len(recs)
        row["sim"] = float(np.mean([r["correct"] for r in recs])) if recs else float("nan")
        if oracle_ok:
            recs8 = sim_run(s, "int8", tmp / f".sim8_A0{s}.jsonl")
            recs8.sort(key=lambda r: r["session_trial"])
            cls8 = np.array([r["raw_cls"] for r in recs8])
            row["int16"] = float((cls8 == y).mean())
            row["agree"] = int((cls8 == off_cls).sum())
            for t, p in zip(y, cls8):
                conf_board[t, p] += 1
            expected[str(s)] = {"trial": [r["session_trial"] for r in recs8], "logits": [r["logits"] for r in recs8]}
        if not a.no_raw:
            from export_replay import load_session_uv

            ref = PipelineRef(params, rep.uv_scale)
            x_uv, cue, _, _ = load_session_uv(s)
            w_float = ref.windows(x_uv, cue[plan.trials], s)              # = Reproduce_Baselines windows
            seg = rep.x_int16[:, plan.start:plan.stop]
            w_int = ref.windows(ref.to_uv(seg), plan.cue_rel, s)
            d = np.abs(w_int.astype(int) - w_float.astype(int))
            row["elem"] = float((d > 0).mean())
            row["maxd"] = int(d.max())
        rows.append(row)
        print(f"A0{s}: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        for f in (tmp / f".sim_A0{s}.jsonl", tmp / f".sim8_A0{s}.jsonl"):
            if f.exists():
                f.unlink()

    if expected:
        ej = bridge.EXPECTED_JSON
        ej.parent.mkdir(parents=True, exist_ok=True)
        ej.write_text(json.dumps({"source": "host/replay_accuracy.py: int8 model (Int8Oracle) on pipeline_ref windows "
                                  "of the int16 replay stream, default plan (10 s lead-in, trials 24-143)",
                                  "uv_scale": uv_scale, "subjects": expected}, separators=(",", ":")) + "\n")

    def pct(v):
        return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.1f}"
    mean = lambda k: float(np.mean([r[k] for r in rows])) if all(k in r for r in rows) else None  # noqa: E731
    lines = [
        "# Replay accuracy (host side)",
        "",
        f"Generated {dt.date.today().isoformat()} by `host/replay_accuracy.py`. Model `tinyeegnet_int8.pte` "
        f"({MODEL_TAG}, {MODEL_DESC}), one `.pte` for all subjects, EA matrix per subject. "
        "BCI IV 2a session 2, test trials 25–144 (0-based 24–143, 120 per subject), streamed as the "
        f"bridge streams them: int16 µV × {uv_scale:g}, 250 Hz, 10 s lead-in, continuous.",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| Offline int8 | Reproduce_Baselines int8 model on the float-stream windows (`int8_predictions.npz`) |",
        "| Sim replay | `bridge.py --board sim --fast`: real framing + streaming pipeline, logits looked up from the offline predictions (checks the bridge/UI/measure path; equals Offline by construction) |",
        "| Board target | int8 model (host rebuild, bit-exact on `test_vectors.npz`) on the `pipeline_ref` windows of the int16 stream = what the board must output (`bridge.py --sim-model int8`) |",
        "| Agree | Board-target argmax == offline argmax, of 120 |",
        "| Window diff | int8 input elements that differ between int16-stream and float-stream windows (max abs diff in steps) |",
        "",
        "| Subject | Offline int8 % | Sim replay % | Board target % | Agree | Window diff % (max) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| A0{r['s']} | {pct(r['offline'])} | {pct(r['sim'])} ({r['sim_n']}) | {pct(r.get('int16'))} | "
                     f"{r.get('agree', '–')} | {pct(r.get('elem'))} ({r.get('maxd', '–')}) |")
    agree_tot = sum(r.get("agree", 0) for r in rows) if all("agree" in r for r in rows) else None
    lines.append(f"| **Mean** | **{pct(mean('offline'))}** | **{pct(mean('sim'))}** | **{pct(mean('int16'))}** | "
                 f"{'' if agree_tot is None else f'{agree_tot}/{120 * len(rows)}'} | {pct(mean('elem'))} |")
    lines += ["", "Board-target confusion matrix, all subjects summed (rows true, columns predicted):", "",
              "| | pred left | pred right |", "|---|---:|---:|",
              f"| true left | {conf_board[0, 0]} | {conf_board[0, 1]} |",
              f"| true right | {conf_board[1, 0]} | {conf_board[1, 1]} |", "",
              "Notes:", "",
              "- The int16 stream reproduces the float-stream windows exactly (0 elements differ); on the float stream the reference "
              "reproduces `test_vectors.npz` bit for bit (`host/tests/test_pipeline_ref.py`), with or without a 10 s lead-in.",
              f"- UART encoding µV × {uv_scale:g}: BCI IV 2a is stored in steps of 100/2048 µV, so 20.48 counts per µV "
              "carries the recording losslessly (int16 range ±1600 µV, 0 samples clipped); any int8 window difference "
              "left here comes from float rounding only. The earlier µV × 10 encoding flipped 28.6 % of window elements "
              "(max 4 steps) and 13/1080 argmaxes, because the EA whitening amplifies the 0.05 µV rounding error.",
              "- Board results go to `results/board_metrics.md` (`host/measure.py`)."]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}")
    if tmp_ctx is not None:
        tmp_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
