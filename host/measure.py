"""Collect board inference records from bridge.py and write a metrics table.

    python host/measure.py --listen 5555 --trials 120 --out results/board_metrics.md
    python host/measure.py --jsonl session.jsonl --out results/board_metrics.md   # offline, from bridge --log

Reports accuracy vs the true label, argmax agreement with the offline int8 predictions
(float stream) and with the expected board output (int8 model on the int16-stream windows,
``host/reference/expected_board_predictions.json``), latency (``lat_us``: last UART byte of the
window -> JSON line formatted; excludes formatting/queueing) mean / p50 / p95 / max, and gated (motion) count. The board's int8
window cannot be read back, so agreement is on the argmax.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from listen import Listener, parse_listen  # noqa: E402


def collect_udp(listen: str, n: int, idle_timeout: float) -> tuple[list[dict], list[str]]:
    host, port = parse_listen(listen)
    lst = Listener(port, host)
    preds, logs = [], []
    last = time.monotonic()
    print(f"[measure] listening on {host}:{port} for {n} trials (idle timeout {idle_timeout:g} s)", flush=True)
    try:
        while len(preds) < n and time.monotonic() - last < idle_timeout:
            for rec in lst.poll(0.2):
                last = time.monotonic()
                if rec.get("event") == "pred" and rec.get("trial", -1) >= 0:
                    preds.append(rec)
                    if len(preds) % 10 == 0:
                        print(f"[measure] {len(preds)}/{n}", flush=True)
                elif rec.get("event") == "log":
                    logs.append(rec.get("line", ""))
    except KeyboardInterrupt:
        pass
    finally:
        lst.close()
    return preds[:n], logs


def collect_jsonl(path: str, n: int | None) -> tuple[list[dict], list[str]]:
    preds, logs = [], []
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        if rec.get("event") == "pred" and rec.get("trial", -1) >= 0:
            preds.append(rec)
        elif rec.get("event") == "log":
            logs.append(rec.get("line", ""))
    return (preds[:n] if n else preds), logs


def metrics(preds: list[dict]) -> dict:
    n = len(preds)
    gated = [p for p in preds if p.get("cls") == -1 or p.get("motion")]
    raw = np.array([int(p.get("raw_cls", p.get("cls", -1))) for p in preds])
    lab = np.array([int(p.get("true_label", p.get("label", -1))) for p in preds])
    ungated = np.array([not (p.get("cls") == -1 or p.get("motion")) for p in preds], dtype=bool)
    lat = np.array([float(p.get("lat_us", np.nan)) for p in preds])

    def agree(key):
        v = [int(p[key]) for p in preds if p.get(key) is not None]
        return (sum(v), len(v))

    return {
        "n": n, "gated": len(gated),
        "acc_raw": (int((raw == lab).sum()), n),
        "acc_ungated": (int((raw[ungated] == lab[ungated]).sum()), int(ungated.sum())),
        "match_offline": agree("match_offline"), "match_expected": agree("match_expected"),
        "lat": {k: float(f(lat)) for k, f in (("mean", np.nanmean), ("p50", np.nanmedian),
                                               ("p95", lambda x: np.nanpercentile(x, 95)), ("max", np.nanmax))} if n else {},
        "subjects": sorted({int(p.get("subject", 0)) for p in preds}),
        "boards": sorted({str(p.get("board", "e8")) for p in preds}),
        "conf_mean": float(np.mean([float(p.get("conf", 0)) for p in preds])) if n else None,
        "span_s": (preds[-1].get("t", 0) - preds[0].get("t", 0)) / 1000 if n > 1 else 0.0,
    }


def frac(t) -> str:
    return f"{t[0]}/{t[1]} ({100 * t[0] / t[1]:.1f} %)" if t[1] else "–"


def render(m: dict, logs: list[str], source: str) -> str:
    ready = next((l for l in logs if l.startswith("READY")), "–")
    lat = m["lat"]
    rows = [
        "# Board metrics",
        "",
        f"Generated {dt.datetime.now().isoformat(timespec='seconds')} by `host/measure.py` from {source}. "
        f"Board: {', '.join(m['boards'])}; subjects {', '.join(f'A0{s}' for s in m['subjects'])}.",
        "",
        f"Boot line: `{ready}`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Inferences (cued trials) | {m['n']} over {m['span_s']:.0f} s of board time |",
        f"| Accuracy vs true label (raw argmax, all) | {frac(m['acc_raw'])} |",
        f"| Accuracy vs true label (not gated) | {frac(m['acc_ungated'])} |",
        f"| Argmax = offline int8 (float stream) | {frac(m['match_offline'])} |",
        f"| Argmax = expected board output (int8 model on int16-stream windows) | {frac(m['match_expected'])} |",
        f"| Motion-gated windows | {m['gated']} |",
    ]
    if lat:
        rows += [f"| Latency `lat_us` mean / p50 / p95 / max | {lat['mean']:.0f} / {lat['p50']:.0f} / {lat['p95']:.0f} / "
                 f"{lat['max']:.0f} µs |"]
    rows += [f"| Mean confidence | {m['conf_mean']:.2f} |" if m["conf_mean"] is not None else "| Mean confidence | – |", "",
             "`lat_us` = last UART byte of the window's last sample (UART ISR stamp) -> just before the JSON line is formatted (preprocessing tail + EA + "
             "quantize + Ethos-U85 inference + output), measured on the board. NPU/CPU cycle counts are not in "
             "the UART JSON; see the FVP numbers in Emulator_Deploy `results/fvp_v2_metrics.md`.", ""]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listen", default="5555")
    ap.add_argument("--jsonl", default=None, help="read a bridge --log file instead of UDP")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--idle-timeout", type=float, default=60.0)
    ap.add_argument("--out", default=None, help="markdown file (default: stdout)")
    a = ap.parse_args(argv)
    if a.jsonl:
        preds, logs = collect_jsonl(a.jsonl, a.trials)
        source = f"`{a.jsonl}`"
    else:
        preds, logs = collect_udp(a.listen, a.trials, a.idle_timeout)
        source = f"UDP {a.listen} (bridge.py)"
    if not preds:
        print("[measure] no inference records received", file=sys.stderr)
        return 2
    md = render(metrics(preds), logs, source)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
        print(f"[measure] {len(preds)} records -> {a.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
