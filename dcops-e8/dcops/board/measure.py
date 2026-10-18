# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Measure the running CLASSIFY firmware over J-Link, without halting the core.

Reads the debug globals every ``--interval`` seconds for ``--seconds`` (one J-Link session):
``g_clsFrames`` and the 1 ms SysTick counter ``tick_count`` give the on-chip FPS; ``g_clsPerfUs``
(firmware's ~1 s averages) and the raw cycle totals (``preCycles``/``infCycles``/``postCycles``/
``frameCycles`` over ``timingFrames``) give the stage times; raw and smoothed top-1 come with
labels from the build's labels file. Any ``g_kws*`` globals are read and reported too.

Writes ``dcops/board/measurements/<timestamp>.json`` and ``.md`` and prints the markdown.

Usage:
    python3 dcops/board/measure.py <axf> [--seconds 10] [--interval 1]
    python3 dcops/board/measure.py <axf> --dry-run              # print the J-Link script only
    python3 dcops/board/measure.py <axf> --dry-run --simulate   # also run the parser on fake output
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import axfinfo  # noqa: E402
import jlink  # noqa: E402

CLS_GLOBALS = ["g_clsFrames", "g_clsTopIdx", "g_clsTopProb", "g_clsSmoothedTopIdx",
               "g_clsSmoothedTopProb", "g_clsShownIdx", "g_clsPerfUs"]
CYCLE_GLOBALS = ["timingFrames", "frameIntervals", "preCycles", "infCycles", "postCycles",
                 "frameCycles"]
CLOCK_GLOBALS = ["tick_count", "SystemCoreClock"]
STAGES = ["pre", "npu", "post", "frame"]


def plan_reads(syms: jlink.Symbols) -> tuple[list[tuple[str, int, int]], list[str]]:
    """Choose the globals to read.

    :param syms:    Image symbols.
    :returns:       ``([(name, addr, words)], missing_names)``.
    """
    reads, missing = [], []
    kws = [s.name for s in syms.matching(r"^g_kws")]
    for name in CLS_GLOBALS + CYCLE_GLOBALS + CLOCK_GLOBALS + kws:
        s = syms.find(name)
        if s is None:
            missing.append(name)
            continue
        reads.append((name, s.addr, max(1, ((s.size or 4) + 3) // 4)))
    return reads, missing


def build_script(reads: list[tuple[str, int, int]], samples: int, interval_ms: int) -> list[str]:
    """Commander lines: ``samples`` rounds of ``mem32`` reads separated by ``Sleep``.

    The core is never halted.

    :param reads:       From :func:`plan_reads`.
    :param samples:     Number of rounds.
    :param interval_ms: Sleep between rounds.
    :returns:           Commander lines.
    """
    cmds = ["connect"]
    for k in range(samples):
        cmds += [jlink.mem32_cmd(addr, n) for _, addr, n in reads]
        if k + 1 < samples:
            cmds.append(f"Sleep {interval_ms}")
    return cmds + ["exit"]


def decode_samples(out: str, reads: list[tuple[str, int, int]], samples: int) -> list[dict]:
    """Turn Commander output into one dict of raw word lists per sample.

    :param out:     Commander output.
    :param reads:   From :func:`plan_reads`.
    :param samples: Rounds in the script.
    :returns:       ``[{name: [words] | None}]``.
    """
    requests = [(addr, n) for _ in range(samples) for _, addr, n in reads]
    vals = jlink.collect_reads(jlink.parse_events(out), requests)
    res = []
    for k in range(samples):
        chunk = vals[k * len(reads):(k + 1) * len(reads)]
        res.append({name: v for (name, _, _), v in zip(reads, chunk)})
    return res


def interpret(sample: dict) -> dict:
    """Typed view of one sample.

    :param sample:  ``{name: [words]}``.
    :returns:       Decoded values (signed indices, 64-bit cycle totals).
    """
    g = lambda n: sample.get(n)  # noqa: E731
    d: dict = {}
    if g("g_clsFrames"):
        d["frames"] = g("g_clsFrames")[0]
    if g("g_clsTopIdx") and g("g_clsTopProb"):
        d["top3"] = [(jlink.as_i32(i), jlink.as_i32(p)) for i, p in
                     zip(g("g_clsTopIdx"), g("g_clsTopProb"))]
    for n, key in (("g_clsSmoothedTopIdx", "smoothed_idx"), ("g_clsSmoothedTopProb", "smoothed_prob"),
                   ("g_clsShownIdx", "shown_idx")):
        if g(n):
            d[key] = jlink.as_i32(g(n)[0])
    if g("g_clsPerfUs"):
        d["perf_us"] = list(g("g_clsPerfUs")[:4])
    for n in ("timingFrames", "frameIntervals", "tick_count", "SystemCoreClock"):
        if g(n):
            d[n] = g(n)[0]
    for n in ("preCycles", "infCycles", "postCycles", "frameCycles"):
        if g(n) and len(g(n)) >= 2:
            d[n] = jlink.as_u64(g(n))
    d["kws"] = {n: v for n, v in sample.items() if n.startswith("g_kws") and v is not None}
    return d


KWS_I32 = ("g_kwsInitState", "g_kwsLastIdx", "g_kwsLastConf", "g_kwsTopIdx", "g_kwsTopProb")


def decode_kws(raw: dict, labels: list[str]) -> dict:
    """Typed view of the VoiceControl.cc globals (unknown g_kws* names are kept as raw words).

    :param raw:     ``{g_kwsName: [words]}``.
    :param labels:  KWS labels.
    :returns:       Decoded dict (``last``/``top`` carry labels, ``perf_us`` = MFCC, NPU, post).
    """
    lab = lambda i: labels[i] if 0 <= i < len(labels) else ("-" if i < 0 else f"#{i}")  # noqa: E731
    d: dict = {}
    for n, w in raw.items():
        if not w:
            continue
        key = n[len("g_kws"):]
        key = key[0].lower() + key[1:] if key else n
        if n in KWS_I32:
            d[key] = jlink.as_i32(w[0])
        elif n == "g_kwsLogits":
            d[key] = [round(jlink.as_f32(x), 4) for x in w]
        elif n == "g_kwsPerfUs":
            d[key] = list(w)
        else:
            d[key] = w[0] if len(w) == 1 else list(w)
    if "lastIdx" in d:
        d["last"] = {"idx": d["lastIdx"], "label": lab(d["lastIdx"]), "prob": d.get("lastConf", 0) / 1000}
    if "topIdx" in d:
        d["top"] = {"idx": d["topIdx"], "label": lab(d["topIdx"]), "prob": d.get("topProb", 0) / 1000}
    if "logits" in d:
        n = min(len(d["logits"]), len(labels)) or len(d["logits"])
        i = max(range(n), key=lambda k: d["logits"][k])
        d["logits_argmax"] = {"idx": i, "label": lab(i)}
    return d


def summarize_kws(samples: list[dict], labels: list[str], elapsed: float) -> dict:
    """KWS rates and times over the run.

    :param samples:     Interpreted samples (their ``kws`` raw words).
    :param labels:      KWS labels.
    :param elapsed:     Seconds between first and last sample.
    :returns:           Summary dict (empty when the image has no g_kws*).
    """
    ks = [decode_kws(x.get("kws", {}), labels) for x in samples]
    if not ks or not ks[-1]:
        return {}
    first, last = ks[0], ks[-1]
    s: dict = {"now": last}
    state = last.get("initState")
    if state is not None:
        s["state"] = "running" if state == 1 else "not started" if state == 0 else f"FAILED at step {state}"
    for key in ("strides", "captured", "count", "gaps", "audioErrors"):
        if key in first and key in last:
            s[key + "_delta"] = (last[key] - first[key]) & 0xFFFFFFFF
    if elapsed > 0 and "strides_delta" in s:
        s["strides_per_s"] = round(s["strides_delta"] / elapsed, 2)
    if "captured" in last and "strides" in last:
        s["backlog_strides"] = (last["captured"] - last["strides"]) & 0xFFFFFFFF
    wins = []
    for k in ks:
        p = k.get("perfUs")
        if p and any(p) and (not wins or p != wins[-1]):
            wins.append(p)
    if wins:
        s["perf_ms"] = {st: round(statistics.mean(w[i] for w in wins) / 1000.0, 3)
                        for i, st in enumerate(("mfcc", "npu", "post")) if i < len(wins[0])}
        s["perf_windows"] = len(wins)
    return s


def summarize(samples: list[dict], labels: list[str], nominal_s: float, host_s: float,
              kws_labels: list[str] | None = None) -> dict:
    """Compute FPS, stage times and the current prediction.

    :param samples:     Interpreted samples (:func:`interpret`).
    :param labels:      Class labels.
    :param nominal_s:   Script sleep time between first and last sample (fallback clock).
    :param host_s:      Wall time of the J-Link run (upper bound, includes connect).
    :returns:           Summary dict.
    """
    lab = lambda i: labels[i] if 0 <= i < len(labels) else ("-" if i < 0 else f"#{i}")  # noqa: E731
    first, last = samples[0], samples[-1]
    clock = last.get("SystemCoreClock") or 400_000_000
    s: dict = {"samples": len(samples), "core_clock_hz": clock}
    # FPS from on-chip time (SysTick at 1 kHz), falling back to the script's Sleep time.
    if "tick_count" in first and "tick_count" in last and last["tick_count"] > first["tick_count"]:
        elapsed = (last["tick_count"] - first["tick_count"]) / 1000.0
        s["elapsed_source"] = "tick_count (on-chip 1 ms SysTick)"
    else:
        elapsed = nominal_s
        s["elapsed_source"] = "J-Link Sleep total (approximate)"
    s["elapsed_s"] = round(elapsed, 3)
    s["host_wall_s"] = round(host_s, 3)
    if "frames" in first and "frames" in last:
        df = (last["frames"] - first["frames"]) & 0xFFFFFFFF
        s["frames_delta"] = df
        s["fps"] = round(df / elapsed, 2) if elapsed > 0 else None
        s["running"] = df > 0
    # g_clsPerfUs: firmware ~1 s window averages; average the distinct windows we saw.
    windows = []
    for x in samples:
        p = x.get("perf_us")
        if p and any(p) and (not windows or p != windows[-1]):
            windows.append(p)
    if windows:
        s["perf_windows"] = len(windows)
        s["perf_ms"] = {st: round(statistics.mean(w[i] for w in windows) / 1000.0, 3)
                        for i, st in enumerate(STAGES)}
        s["perf_ms_sd"] = {st: round(statistics.pstdev(w[i] for w in windows) / 1000.0, 3)
                           for i, st in enumerate(STAGES)}
    # Cycle totals: partial window at the moment of each read.
    cyc = {st: [] for st in STAGES}
    for x in samples:
        tf, fi = x.get("timingFrames", 0), x.get("frameIntervals", 0)
        if tf:
            for st, n in (("pre", "preCycles"), ("npu", "infCycles"), ("post", "postCycles")):
                if n in x:
                    cyc[st].append(x[n] / tf / clock * 1000.0)
        if fi and "frameCycles" in x:
            cyc["frame"].append(x["frameCycles"] / fi / clock * 1000.0)
    if any(cyc.values()):
        s["cycles_ms"] = {st: round(statistics.mean(v), 3) if v else None for st, v in cyc.items()}
    ms = s.get("perf_ms") or s.get("cycles_ms") or {}
    if ms.get("frame"):
        s["fps_from_frame_time"] = round(1000.0 / ms["frame"], 2)
    # Prediction now, and the smoothed top-1 over the run.
    if "top3" in last:
        s["raw_top3"] = [{"idx": i, "label": lab(i), "prob": p / 1000.0} for i, p in last["top3"]]
    if "smoothed_idx" in last:
        s["smoothed_top1"] = {"idx": last["smoothed_idx"], "label": lab(last["smoothed_idx"]),
                              "prob": last.get("smoothed_prob", 0) / 1000.0}
        s["shown"] = lab(last.get("shown_idx", -1)) if last.get("shown_idx", -1) >= 0 else "unsure"
        hist: dict[str, int] = {}
        for x in samples:
            if "smoothed_idx" in x:
                hist[lab(x["smoothed_idx"])] = hist.get(lab(x["smoothed_idx"]), 0) + 1
        s["smoothed_top1_histogram"] = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
    if last.get("kws"):
        s["kws"] = summarize_kws(samples, kws_labels or [], elapsed)
    return s


def perf_markdown(s: dict, facts: dict) -> str:
    """Markdown tables: perf, prediction, DEVPOST-ready rows and build facts.

    :param s:       :func:`summarize` output.
    :param facts:   :func:`axfinfo.build_facts` output.
    :returns:       Markdown text.
    """
    f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    pm, cm = s.get("perf_ms", {}), s.get("cycles_ms", {})
    lines = [f"Measured {s.get('timestamp', '')} on the DevKit-E8, M55 HP at "
             f"{s['core_clock_hz'] / 1e6:.0f} MHz, {s['samples']} reads over {s['elapsed_s']} s "
             f"({s['elapsed_source']}).", "",
             "| Stage | g_clsPerfUs mean (ms) | sd | cycle counters (ms) |", "|---|---|---|---|"]
    names = {"pre": "Pre-process (RGB888 -> float)", "npu": "NPU inference (Ethos-U85, incl. int8 boundary)",
             "post": "Post (softmax, EMA, UI update)", "frame": "Frame-to-frame"}
    for st in STAGES:
        lines.append(f"| {names[st]} | {f(pm.get(st))} | {f(s.get('perf_ms_sd', {}).get(st))} | "
                     f"{f(cm.get(st))} |")
    lines += ["", "| Metric | Value |", "|---|---|",
              f"| FPS (g_clsFrames delta / time) | {f(s.get('fps'))} ({s.get('frames_delta', '-')} frames) |",
              f"| FPS (1 / frame time) | {f(s.get('fps_from_frame_time'))} |",
              f"| g_clsPerfUs windows averaged | {s.get('perf_windows', 0)} |"]
    if "smoothed_top1" in s:
        t = s["smoothed_top1"]
        lines.append(f"| Smoothed top-1 now | {t['label']} ({t['prob']:.3f}), card: {s['shown']} |")
    if "raw_top3" in s:
        lines.append("| Raw top-3 now | " + ", ".join(f"{r['label']} {r['prob']:.3f}" for r in s["raw_top3"]) + " |")
    if s.get("smoothed_top1_histogram"):
        lines.append("| Smoothed top-1 over run | " + ", ".join(
            f"{k} x{v}" for k, v in s["smoothed_top1_histogram"].items()) + " |")
    k = s.get("kws") or {}
    if k:
        now = k["now"]
        kp = k.get("perf_ms", {})
        lines += ["", "| Voice (KWS) | Value |", "|---|---|",
                  f"| State | {k.get('state', '-')} (heap used {now.get('heapUsed', '-')} B) |",
                  f"| Strides/s (expect 2.0 for 0.5 s stride) | {f(k.get('strides_per_s'))} "
                  f"({k.get('strides_delta', '-')} strides, backlog {k.get('backlog_strides', '-')}) |",
                  f"| MFCC / NPU / post (ms) | {f(kp.get('mfcc'))} / {f(kp.get('npu'))} / {f(kp.get('post'))} "
                  f"({k.get('perf_windows', 0)} windows) |",
                  f"| Keywords fired during run / since boot | {k.get('count_delta', '-')} / {now.get('count', '-')} |",
                  f"| Gaps / audio errors during run | {k.get('gaps_delta', '-')} / {k.get('audioErrors_delta', '-')} |"]
        if "last" in now:
            lines.append(f"| Last keyword | {now['last']['label']} ({now['last']['prob']:.3f}) |")
        if "top" in now:
            lines.append(f"| Top-1 last stride | {now['top']['label']} ({now['top']['prob']:.3f}) |")
        if "logits_argmax" in now:
            lines.append(f"| Logits argmax | {now['logits_argmax']['label']} |")
        if kp.get("npu") is not None:
            kws_row = (f"| KWS latency (MFCC + inference) | {f(kp.get('mfcc'))} ms + {f(kp.get('npu'))} ms "
                       f"(post {f(kp.get('post'))} ms) | measured on board |")
            s.setdefault("_devpost_extra", []).append(kws_row)
    if s.get("running") is False:
        lines += ["", "**WARNING: g_clsFrames did not advance -- the app is not running "
                      "(camera hung? power-cycle the board).**"]
    # DEVPOST rows (dcops/docs/DEVPOST.md, ## Performance)
    m = facts["mram"]
    reg = facts["regions"]
    npu = pm.get("npu") or cm.get("npu")
    pre, post = pm.get("pre") or cm.get("pre"), pm.get("post") or cm.get("post")
    frame = pm.get("frame") or cm.get("frame")
    model = os.path.basename(facts["pte"]["file"] or "?")
    lines += ["", "DEVPOST rows (paste into `dcops/docs/DEVPOST.md` ## Performance):", "",
              "| Metric | Value | Status |", "|---|---|---|",
              f"| Classifier (`{model}`) NPU inference | {f(npu)} ms | measured "
              f"({s.get('perf_windows', 0)} x ~1 s firmware averages) |",
              f"| Classifier pre / post | {f(pre)} ms / {f(post)} ms | measured |",
              f"| Frame-to-frame (camera + preview + model + LVGL) | {f(frame)} ms, "
              f"{f(s.get('fps'))} FPS | measured (classifier build) |",
              f"| MRAM used (HP image) | {axfinfo.mib(m['used_bytes'])} of {axfinfo.mib(m['usable_bytes'])} "
              f"usable, {(m['headroom_bytes'] or 0) // 1024} KB headroom | from ELF |",
              f"| .pte size (embedded) | {facts['pte']['symbol_bytes'] / 1024:.0f} KB | from ELF |"
              if facts["pte"]["symbol_bytes"] else "",
              f"| SRAM used | {axfinfo.mib(reg.get('SRAM', {}).get('used'))} bulk SRAM + "
              f"{axfinfo.mib(reg.get('UNSTRIPED', {}).get('used'))} framebuffer region, "
              f"{axfinfo.mib(reg.get('DTCM', {}).get('used'))} DTCM | from ELF |"]
    lines += s.pop("_devpost_extra", [])
    lines += ["", axfinfo.facts_markdown(facts)]
    return "\n".join(l for l in lines if l is not None)


def simulated_output(reads: list[tuple[str, int, int]], samples: int, interval_ms: int) -> str:
    """Fake Commander output with plausible values (for ``--simulate``).

    :param reads:       From :func:`plan_reads`.
    :param samples:     Rounds.
    :param interval_ms: Sleep between rounds.
    :returns:           Text shaped like JLinkExe output.
    """
    lines = ["SEGGER J-Link Commander V8.10 (simulated)", "Connecting to target via SWD",
             "Cortex-M55 identified.", "J-Link>connect"]
    clock = 400_000_000
    for k in range(samples):
        t_ms = 5000 + k * interval_ms
        frames = 300 + int(k * interval_ms / 166.0)
        tf = 1 + (k * 3) % 6
        vals = {"g_clsFrames": [frames], "g_clsTopIdx": [3, 7, 1], "g_clsTopProb": [612, 201, 55],
                "g_clsSmoothedTopIdx": [3], "g_clsSmoothedTopProb": [580], "g_clsShownIdx": [3],
                "g_clsPerfUs": [1080 + k % 2, 9850 + 10 * (k // 2), 180, 166000],
                "timingFrames": [tf], "frameIntervals": [tf],
                "preCycles": [tf * 432000, 0], "infCycles": [tf * 3940000, 0],
                "postCycles": [tf * 72000, 0], "frameCycles": [tf * 66400000 & 0xFFFFFFFF, 0],
                "tick_count": [t_ms], "SystemCoreClock": [clock],
                "g_kwsInitState": [1], "g_kwsLastIdx": [2], "g_kwsLastConf": [812], "g_kwsCount": [3 + k // 4],
                "g_kwsTopIdx": [0], "g_kwsTopProb": [904], "g_kwsStrides": [100 + 2 * k],
                "g_kwsCaptured": [100 + 2 * k], "g_kwsGaps": [0], "g_kwsAudioErrors": [0],
                "g_kwsPerfUs": [3100, 450, 20], "g_kwsHeapUsed": [40960],
                "g_kwsLogits": [0x40000000] + [0xBF800000] * 15}
        for name, addr, n in reads:
            words = vals.get(name, [k] * n)
            words = (words + [0] * n)[:n]
            lines.append(f"J-Link>{jlink.mem32_cmd(addr, n)}")
            for j in range(0, n, 4):
                lines.append(f"{addr + 4 * j:08X} = " + " ".join(f"{w & 0xFFFFFFFF:08X}" for w in words[j:j + 4]) + " ")
        if k + 1 < samples:
            lines.append(f"J-Link>Sleep {interval_ms}")
    return "\n".join(lines + ["J-Link>exit"]) + "\n"


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("axf")
    ap.add_argument("--seconds", type=float, default=10.0, help="span between first and last read")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between reads")
    ap.add_argument("--labels", help="labels file (default: CMakeCache CLASS_LABELS of the build)")
    ap.add_argument("--kws-labels", help="KWS labels file (default: CMakeCache KWS_LABELS)")
    ap.add_argument("--dry-run", action="store_true", help="print the J-Link script, do not run it")
    ap.add_argument("--simulate", action="store_true", help="with --dry-run: parse fake output")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "measurements"),
                    help="where <timestamp>.json/.md go (not written for --dry-run unless given)")
    ap.add_argument("--note", default="", help="free text stored with the measurement")
    a = ap.parse_args()

    syms = jlink.Symbols(a.axf)
    facts = axfinfo.build_facts(a.axf, syms)
    labels = axfinfo.read_labels(a.labels or facts["build"]["labels_path"])
    kws_labels = axfinfo.read_labels(a.kws_labels or facts["build"].get("kws_labels_path"))
    reads, missing = plan_reads(syms)
    if "g_clsFrames" in missing:
        print("note: no g_clsFrames in this image (not a CLASSIFY build): only build facts, cycle "
              "and g_kws* globals are read.", file=sys.stderr)
    if missing:
        print(f"note: not in image: {', '.join(missing)}", file=sys.stderr)
    for name, addr, n in reads:
        print(f"  {name:22s} @ 0x{addr:08X} x{n}", file=sys.stderr)

    interval_ms = max(10, int(a.interval * 1000))
    samples = max(2, int(round(a.seconds / a.interval)) + 1)
    cmds = build_script(reads, samples, interval_ms)
    nominal = (samples - 1) * interval_ms / 1000.0
    j = jlink.JLink(dry_run=a.dry_run, timeout=60 + nominal * 3)
    if a.dry_run and a.simulate:
        j.fake = simulated_output(reads, samples, interval_ms)
    t0 = time.time()
    out = j.run(cmds, label=f"measure, {samples} reads")
    host = time.time() - t0
    if a.dry_run and not a.simulate:
        return
    raw = decode_samples(out, reads, samples)
    good = [interpret(x) for x in raw if any(v is not None for v in x.values())]
    if len(good) < 2:
        sys.exit("error: fewer than two samples parsed from the J-Link output:\n" + out[-3000:])
    s = summarize(good, labels, nominal, host, kws_labels)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    s["timestamp"] = stamp + (" (SIMULATED)" if a.simulate else "")
    md = perf_markdown(s, facts)
    print(md)
    write = not a.dry_run or "--out-dir" in sys.argv
    if write:
        os.makedirs(a.out_dir, exist_ok=True)
        base = os.path.join(a.out_dir, stamp + ("-sim" if a.simulate else ""))
        with open(base + ".json", "w") as fh:
            json.dump({"summary": s, "facts": facts, "samples": good, "note": a.note,
                       "simulated": bool(a.simulate), "jlink_output_tail": out[-4000:]}, fh, indent=2)
        with open(base + ".md", "w") as fh:
            fh.write(md + "\n")
        print(f"\nwrote {base}.json and {base}.md", file=sys.stderr)


if __name__ == "__main__":
    main()
