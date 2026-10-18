# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Capture the frame the model saw plus the board's results, and rerun the model on the PC.

Method (validated on 2026-10-17): halt the core; if the PC is inside ``wait_for_capture`` (inlined
in ``hal_camera_get_captured_frame``; or in its callee ``camera_image_ready`` with LR there),
the previous frame is fully processed, so ``rgb_image`` (176-byte TIFF header + model-input
RGB888 HxWx3) and the ``g_cls*`` results belong together. Each attempt in the J-Link session
halts, reads registers and globals, dumps the frame and resumes; the first attempt halted in
``wait_for_capture`` wins. The core is always resumed (``g``) before the script exits.

Then the same student runs on the PC on that frame (``pc_infer.py`` in the ExecuTorch 1.1 env:
float + int8 pt2e simulation) and board vs PC top-3 are printed side by side.

Output: ``dcops/board/snapshots/<timestamp>/`` with frame.bin/png, lcd.png (``--lcd``),
board.json, pc.json, compare.md.

Usage:
    python3 dcops/board/snapshot.py <axf> [--run r160_kd] [--lcd] [--attempts 8]
    python3 dcops/board/snapshot.py <axf> --dry-run                    # print the J-Link script
    python3 dcops/board/snapshot.py <axf> --dry-run --simulate --sim-frame img.png --run r160_kd
    python3 dcops/board/snapshot.py --pc-only dcops/board/snapshots/<ts> [--run ...]
Needs numpy + PIL (use dcops/.venv-train/bin/python, or any python3 with them).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import axfinfo  # noqa: E402
import jlink  # noqa: E402
import measure  # noqa: E402

TIFF_HEADER = 176
ET_PY = os.path.expanduser("~/projects/arm-executorch-hack/experiments/blazeface/.venv-et110/bin/python")
VISION = os.path.join(jlink.REPO, "dcops", "models", "hub", "vision")
WAIT_FUNC = "wait_for_capture"
WAIT_LEAVES = ("camera_image_ready", "hal_camera_get_status", "camera_process_frame_end")
RESULT_GLOBALS = ["g_clsFrames", "g_clsTopIdx", "g_clsTopProb", "g_clsSmoothedTopIdx",
                  "g_clsSmoothedTopProb", "g_clsShownIdx", "g_clsPerfUs"]
LCD_W, LCD_H = 480, 800
SLEEPS_MS = [37, 53, 71, 29, 97, 43, 61, 83, 31, 109, 47, 67]  # de-phase attempts from the frame period


def resolve_run(arg: str | None, facts: dict) -> str | None:
    """Find the student run folder for the PC comparison.

    :param arg:     ``--run`` value: a path or a name under dcops/models/hub/vision.
    :param facts:   Build facts (the CMakeCache MODEL_PATH folder is used when it has student.pt).
    :returns:       Run folder or ``None``.
    """
    cands = []
    if arg:
        cands += [arg, os.path.join(VISION, arg)]
    mp = facts.get("build", {}).get("model_path")
    if mp:
        cands.append(os.path.dirname(mp))
    for c in cands:
        if c and os.path.exists(os.path.join(c, "student.pt")):
            return os.path.abspath(c)
    return None


def model_res(args_res: int | None, run: str | None) -> int:
    """:returns: Model input side: ``--res``, else the run's metrics.json, else 160."""
    if args_res:
        return args_res
    if run and os.path.exists(os.path.join(run, "metrics.json")):
        return int(json.load(open(os.path.join(run, "metrics.json")))["args"]["res"])
    print("warning: input size unknown (no --res / run); assuming 160x160", file=sys.stderr)
    return 160


def build_script(reads: list[tuple[str, int, int]], frame_addr: int, frame_bytes: int,
                 out_dir: str, attempts: int, lcd: tuple[int, int] | None) -> tuple[list[str], list[str]]:
    """Commander lines for the halt/read/dump/resume attempts.

    :param reads:       ``(name, addr, words)`` globals to read while halted.
    :param frame_addr:  ``rgb_image`` address.
    :param frame_bytes: Header + HxWx3.
    :param out_dir:     Where the dumps go.
    :param attempts:    Number of halt attempts.
    :param lcd:         ``(addr, bytes)`` of ``lcd_image`` to dump once at the end, or ``None``.
    :returns:           ``(commands, frame_paths)``.
    """
    cmds, paths = ["connect"], []
    for k in range(attempts):
        p = os.path.join(out_dir, f"rgb_image_{k}.bin")
        paths.append(p)
        cmds += ["h", "regs"]
        cmds += [jlink.mem32_cmd(addr, n) for _, addr, n in reads]
        cmds += [jlink.savebin_cmd(p, frame_addr, frame_bytes), "g"]
        if k + 1 < attempts:
            cmds.append(f"Sleep {SLEEPS_MS[k % len(SLEEPS_MS)]}")
    if lcd:
        cmds += ["h", jlink.savebin_cmd(os.path.join(out_dir, "lcd_image.bin"), *lcd), "g"]
    cmds += ["g", "exit"]
    return cmds, paths


def parse_attempts(out: str, reads: list[tuple[str, int, int]], attempts: int) -> list[dict]:
    """Split Commander output into per-attempt PC/LR and global values.

    :param out:         Commander output.
    :param reads:       Globals read per attempt.
    :param attempts:    Attempts in the script.
    :returns:           ``[{"pc", "lr", "values": {name: words}}]``.
    """
    events = jlink.parse_events(out)
    requests = [(addr, n) for _ in range(attempts) for _, addr, n in reads]
    got = jlink.collect_reads_indexed(events, requests)
    res = []
    for k in range(attempts):
        chunk = got[k * len(reads):(k + 1) * len(reads)]
        first = next((idx for _, idx in chunk if idx >= 0), -1)
        res.append({"pc": jlink.last_reg_before(events, first, "pc") if first >= 0 else None,
                    "lr": jlink.last_reg_before(events, first, "lr") if first >= 0 else None,
                    "values": {name: w for (name, _, _), (w, _) in zip(reads, chunk)}})
    return res


def decode_results(values: dict, labels: list[str], kws_labels: list[str] | None = None) -> dict:
    """Board results in readable form.

    :param values:      ``{name: words}``.
    :param labels:      Firmware labels.
    :param kws_labels:  KWS labels (for the g_kws* globals of a voice build).
    :returns:       Dict with frames, top3, smoothed, shown, perf and any g_kws* words.
    """
    lab = lambda i: labels[i] if 0 <= i < len(labels) else ("-" if i < 0 else f"#{i}")  # noqa: E731
    v = lambda n: values.get(n)  # noqa: E731
    d: dict = {}
    if v("g_clsFrames"):
        d["frames"] = v("g_clsFrames")[0]
    if v("g_clsTopIdx") and v("g_clsTopProb"):
        d["top3"] = [{"idx": jlink.as_i32(i), "label": lab(jlink.as_i32(i)), "prob": jlink.as_i32(p) / 1000}
                     for i, p in zip(v("g_clsTopIdx"), v("g_clsTopProb"))]
    if v("g_clsSmoothedTopIdx"):
        i = jlink.as_i32(v("g_clsSmoothedTopIdx")[0])
        d["smoothed"] = {"idx": i, "label": lab(i),
                         "prob": jlink.as_i32((v("g_clsSmoothedTopProb") or [0])[0]) / 1000}
    if v("g_clsShownIdx"):
        i = jlink.as_i32(v("g_clsShownIdx")[0])
        d["shown"] = lab(i) if i >= 0 else "unsure"
    if v("g_clsPerfUs"):
        d["perf_us"] = v("g_clsPerfUs")
    kws = {n: w for n, w in values.items() if n.startswith("g_kws") and w is not None}
    if kws:
        d["kws"] = measure.decode_kws(kws, kws_labels or [])
    return d


def save_png(rgb: np.ndarray, path: str, scale: int = 1):
    """Write an HxWx3 uint8 array as PNG (nearest-neighbour upscaled by ``scale``)."""
    from PIL import Image
    im = Image.fromarray(rgb)
    if scale > 1:
        im = im.resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)
    im.save(path)


def rgb565_to_rgb(raw: bytes, w: int = LCD_W, h: int = LCD_H) -> np.ndarray:
    """Decode a little-endian RGB565 framebuffer.

    :param raw:     ``w*h*2`` bytes.
    :param w:       Width.
    :param h:       Height.
    :returns:       HxWx3 uint8.
    """
    px = np.frombuffer(raw[:w * h * 2], dtype="<u2").reshape(h, w).astype(np.uint32)
    r, g, b = (px >> 11) & 0x1F, (px >> 5) & 0x3F, px & 0x1F
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.uint8)


def tiff_dims(header: bytes) -> tuple[int, int] | None:
    """:returns: ``(width, height)`` from the 176-byte TIFF header, if it is valid (often it is
    not: get_image_data() receives the header by value and fills a copy)."""
    if len(header) >= 36 and header[:4] == b"II*\x00":
        w = int.from_bytes(header[20:24], "little")
        h = int.from_bytes(header[32:36], "little")
        if 0 < w <= 1024 and 0 < h <= 1024:
            return w, h
    return None


def run_pc(run: str, frame_bin: str, res: int, out_json: str) -> dict | None:
    """Run ``pc_infer.py`` in the ExecuTorch 1.1 env.

    :param run:         Student run folder.
    :param frame_bin:   Raw RGB888 frame.
    :param res:         Input side.
    :param out_json:    Where to store the result.
    :returns:           Parsed result, or ``None`` on failure.
    """
    py = os.environ.get("ET_PY", ET_PY)
    if not os.path.exists(py):
        print(f"warning: {py} missing; skipping the PC run", file=sys.stderr)
        return None
    print(f"running the student on the PC (float + int8 pt2e sim, ~1 min): {run}", file=sys.stderr)
    p = subprocess.run([py, os.path.join(HERE, "pc_infer.py"), "--run", run, "--frame", frame_bin,
                        "--res", str(res)], capture_output=True, text=True, check=False)
    if p.returncode != 0:
        print(f"warning: pc_infer failed:\n{p.stderr[-3000:]}", file=sys.stderr)
        return None
    res_json = json.loads(p.stdout.strip().splitlines()[-1])
    with open(out_json, "w") as f:
        json.dump(res_json, f, indent=1)
    return res_json


def compare_markdown(board: dict, pc: dict | None, fw_labels: list[str], note: str = "") -> str:
    """Board vs PC top-3 table.

    :param board:       :func:`decode_results` output.
    :param pc:          ``pc_infer.py`` output (or ``None``).
    :param fw_labels:   Firmware labels.
    :param note:        Extra line under the table.
    :returns:           Markdown.
    """
    lines = []
    if pc and fw_labels and pc["labels"] != fw_labels:
        lines.append(f"**WARNING: firmware labels ({len(fw_labels)}) differ from the run's labels.txt "
                     f"({len(pc['labels'])}) -- the image was not built from this run; indices are "
                     "not comparable.**\n")
    cols = ["Rank", "Board (NPU, int8)"]
    rows = [[str(r + 1)] for r in range(3)]
    for r, t in enumerate(board.get("top3", [])[:3]):
        rows[r].append(f"{t['label']} {100 * t['prob']:.1f}%")
    for key, title in (("int8", "PC int8 sim"), ("float", "PC float")):
        if pc and key in pc:
            cols.append(title)
            for r, t in enumerate(pc[key][0]["top3"][:3]):
                rows[r].append(f"{t['label']} {100 * t['prob']:.1f}%")
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(r + ["-"] * (len(cols) - len(r))) + " |" for r in rows]
    if pc and "top3" in board:
        for key in ("int8", "float"):
            if key not in pc:
                continue
            probs = pc[key][0]["probs"]
            same = board["top3"][0]["idx"] == pc[key][0]["top3"][0]["idx"]
            diffs = [abs(t["prob"] - probs[t["idx"]]) for t in board["top3"] if 0 <= t["idx"] < len(probs)]
            lines.append(f"\n{key}: top-1 {'MATCH' if same else 'DIFFERS'}; max |p_board - p_pc| over the "
                         f"board's top-3 = {100 * max(diffs, default=0):.1f} pp")
    if "smoothed" in board:
        s = board["smoothed"]
        lines.append(f"\nBoard smoothed top-1: {s['label']} {100 * s['prob']:.1f}%, card: {board.get('shown')}"
                     f" (frame #{board.get('frames')})")
    k = board.get("kws")
    if k:
        parts = [f"init {k.get('initState')}"]
        if "last" in k:
            parts.append(f"last keyword {k['last']['label']} {100 * k['last']['prob']:.0f}%")
        if "top" in k:
            parts.append(f"top-1 {k['top']['label']} {100 * k['top']['prob']:.0f}%")
        if "strides" in k:
            parts.append(f"{k['strides']} strides")
        lines.append("\nVoice: " + ", ".join(parts))
    if note:
        lines.append("\n" + note)
    return "\n".join(lines)


def find_wait_pc(syms: jlink.Symbols) -> tuple[int, int]:
    """A (PC, LR) pair inside ``wait_for_capture`` for ``--simulate``.

    :param syms:    Image symbols.
    :returns:       ``(pc, lr)``.
    """
    f = syms.require("hal_camera_get_captured_frame")
    base = f.addr & ~1
    for a in range(base, base + f.size, 4):
        if syms.in_function(a, WAIT_FUNC):
            return a, (base + 0x40) | 1
    raise RuntimeError("no wait_for_capture address found")


def simulate(out_dir: str, paths: list[str], reads: list[tuple[str, int, int]], syms: jlink.Symbols,
             frame_src: str | None, res: int, lcd: bool) -> str:
    """Fake a session: attempt 0 halts elsewhere, attempt 1 in wait_for_capture.

    :param out_dir:     Snapshot folder.
    :param paths:       Frame dump paths of the attempts.
    :param reads:       Globals.
    :param syms:        Image symbols.
    :param frame_src:   Image used as the "camera frame" (else a gradient).
    :param res:         Input side.
    :param lcd:         Also fake an LCD dump.
    :returns:           Fake Commander output.
    """
    if frame_src:
        if frame_src.endswith(".bin"):
            img = np.fromfile(frame_src, dtype=np.uint8).reshape(res, res, 3)
        else:
            from PIL import Image
            img = np.asarray(Image.open(frame_src).convert("RGB").resize((res, res)))
    else:
        y, x = np.mgrid[0:res, 0:res]
        img = np.stack([x * 255 // res, y * 255 // res, np.full_like(x, 128)], -1).astype(np.uint8)
    for p in paths:
        with open(p, "wb") as f:
            f.write(bytes(TIFF_HEADER) + img.tobytes())
    if lcd:
        with open(os.path.join(out_dir, "lcd_image.bin"), "wb") as f:
            f.write(np.full((LCD_H, LCD_W), 0x07E0, dtype="<u2").tobytes())
    wpc, wlr = find_wait_pc(syms)
    other = syms.require("alif::app::ObjectDetectionHandler(arm::app::ApplicationContext&)").addr & ~1
    vals = {"g_clsFrames": [1234], "g_clsTopIdx": [4, 0, 1], "g_clsTopProb": [690, 150, 30],
            "g_clsSmoothedTopIdx": [4], "g_clsSmoothedTopProb": [640], "g_clsShownIdx": [4],
            "g_clsPerfUs": [1080, 9850, 180, 166000]}
    lines = ["SEGGER J-Link Commander (simulated)"]
    for k in range(len(paths)):
        pc, lr = (other + 0x100, other | 1) if k == 0 else (wpc, wlr)
        lines += ["J-Link>h", f"PC = {pc:08X}, CycleCnt = 00000000", "J-Link>regs",
                  f"PC = {pc:08X}, CycleCnt = 00000000", "R0 = 00000000, R1 = 00000001",
                  f"SP(R13)= 200FFF00, MSP= 200FFF00, PSP= 00000000, R14(LR) = {lr:08X}",
                  "XPSR = 61000000: APSR = nZCvq, EPSR = 01000000, IPSR = 000 (NoException)"]
        for name, addr, n in reads:
            w = (vals.get(name, [k]) + [0] * n)[:n]
            for j in range(0, n, 4):
                lines.append(f"{addr + 4 * j:08X} = " + " ".join(f"{x:08X}" for x in w[j:j + 4]))
        lines.append("J-Link>g")
    return "\n".join(lines) + "\n"


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("axf", nargs="?")
    ap.add_argument("--run", help="student run (path or name under dcops/models/hub/vision); "
                                  "default: the folder of the build's MODEL_PATH if it has student.pt")
    ap.add_argument("--res", type=int, help="model input side (default: run metrics.json)")
    ap.add_argument("--labels", help="firmware labels (default: CMakeCache CLASS_LABELS)")
    ap.add_argument("--attempts", type=int, default=8, help="halt attempts in one J-Link session")
    ap.add_argument("--lcd", action="store_true", help="also dump the LCD framebuffer (768 KB) to PNG")
    ap.add_argument("--no-pc", action="store_true", help="skip the PC model run")
    ap.add_argument("--out-dir", help="snapshot folder (default dcops/board/snapshots/<timestamp>)")
    ap.add_argument("--dry-run", action="store_true", help="print the J-Link script only")
    ap.add_argument("--simulate", action="store_true", help="with --dry-run: fake the board side")
    ap.add_argument("--sim-frame", help="image/.bin used as the fake frame in --simulate")
    ap.add_argument("--pc-only", metavar="SNAPSHOT_DIR", help="rerun the PC side on a saved snapshot")
    a = ap.parse_args()

    if a.pc_only:
        board = json.load(open(os.path.join(a.pc_only, "board.json")))
        run = resolve_run(a.run or board.get("run"), board.get("facts", {}))
        if not run:
            sys.exit("no student run: pass --run")
        pc = run_pc(run, os.path.join(a.pc_only, "frame.bin"), board["res"], os.path.join(a.pc_only, "pc.json"))
        md = compare_markdown(board["results"], pc, board.get("fw_labels", []))
        print(md)
        open(os.path.join(a.pc_only, "compare.md"), "w").write(md + "\n")
        return
    if not a.axf:
        ap.error("axf is required (or --pc-only)")

    syms = jlink.Symbols(a.axf)
    facts = axfinfo.build_facts(a.axf, syms)
    fw_labels = axfinfo.read_labels(a.labels or facts["build"]["labels_path"])
    run = resolve_run(a.run, facts)
    res = model_res(a.res, run)
    rgb = syms.require("rgb_image")
    frame_bytes = TIFF_HEADER + res * res * 3
    if frame_bytes > rgb.size:
        sys.exit(f"{res}x{res}x3 does not fit rgb_image ({rgb.size} bytes)")
    lcd_sym = syms.find("lcd_image")
    lcd = (lcd_sym.addr, min(lcd_sym.size, LCD_W * LCD_H * 2)) if (a.lcd and lcd_sym) else None
    reads, missing = [], []
    for name in RESULT_GLOBALS + [s.name for s in syms.matching(r"^g_kws")]:
        s = syms.find(name)
        if s is None:
            missing.append(name)
        else:
            reads.append((name, s.addr, max(1, (s.size + 3) // 4)))
    if missing:
        print(f"note: not in image: {', '.join(missing)}", file=sys.stderr)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = a.out_dir or os.path.join(HERE, "snapshots", stamp + ("-sim" if a.simulate else ""))
    if a.dry_run and not a.simulate:
        out_dir = a.out_dir or "/tmp/snapshot-dryrun"
    else:
        os.makedirs(out_dir, exist_ok=True)
    cmds, paths = build_script(reads, rgb.addr, frame_bytes, out_dir, a.attempts, lcd)
    j = jlink.JLink(dry_run=a.dry_run, timeout=120)
    if a.dry_run and a.simulate:
        j.fake = simulate(out_dir, paths, reads, syms, a.sim_frame, res, bool(lcd))
    out = j.run(cmds, label=f"snapshot, {a.attempts} halt attempts, core resumed at the end")
    if a.dry_run and not a.simulate:
        print(f"(model input {res}x{res}; rgb_image @ 0x{rgb.addr:08X}; run: {run or '-'})")
        return

    attempts = parse_attempts(out, reads, a.attempts)
    chosen = None
    for k, at in enumerate(attempts):
        pc, lr = at["pc"], at["lr"]
        where = syms.describe(pc) if pc is not None else "no PC parsed"
        ok = pc is not None and syms.in_function(pc, WAIT_FUNC, lr, WAIT_LEAVES)
        print(f"attempt {k}: PC={pc if pc is None else f'0x{pc:08X}'} {where} -> "
              f"{'IN wait_for_capture' if ok else 'skip'}", file=sys.stderr)
        if ok and chosen is None and os.path.exists(paths[k]):
            chosen = k
    if chosen is None:
        sys.exit("no attempt halted in wait_for_capture (camera hung? app not running?); "
                 "rerun with more --attempts. J-Link output tail:\n" + out[-2000:])

    raw = open(paths[chosen], "rb").read()
    if len(raw) < frame_bytes:
        sys.exit(f"frame dump is {len(raw)} bytes, expected {frame_bytes}")
    dims = tiff_dims(raw[:TIFF_HEADER])
    if dims and dims != (res, res):
        print(f"warning: TIFF header says {dims}, using {res}x{res}", file=sys.stderr)
    frame = np.frombuffer(raw[TIFF_HEADER:frame_bytes], dtype=np.uint8).reshape(res, res, 3)
    frame.tofile(os.path.join(out_dir, "frame.bin"))
    save_png(frame, os.path.join(out_dir, "frame.png"))
    save_png(frame, os.path.join(out_dir, "frame_x4.png"), 4)
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
    if lcd and os.path.exists(os.path.join(out_dir, "lcd_image.bin")):
        lraw = open(os.path.join(out_dir, "lcd_image.bin"), "rb").read()
        save_png(rgb565_to_rgb(lraw), os.path.join(out_dir, "lcd.png"))

    board = decode_results(attempts[chosen]["values"], fw_labels,
                           axfinfo.read_labels(facts["build"].get("kws_labels_path")))
    pcv = attempts[chosen]["pc"]
    meta = {"timestamp": stamp, "simulated": bool(a.simulate), "attempt": chosen,
            "pc": f"0x{pcv:08X}", "pc_where": syms.describe(pcv), "res": res, "run": run,
            "fw_labels": fw_labels, "results": board, "facts": facts,
            "frame_stats": {"min": int(frame.min()), "max": int(frame.max()),
                            "mean_rgb": [round(float(m), 1) for m in frame.reshape(-1, 3).mean(0)]}}
    with open(os.path.join(out_dir, "board.json"), "w") as f:
        json.dump(meta, f, indent=1)
    pc = None
    if not a.no_pc:
        if run:
            pc = run_pc(run, os.path.join(out_dir, "frame.bin"), res, os.path.join(out_dir, "pc.json"))
        else:
            print("note: no student run for this image (pass --run <run>); PC comparison skipped",
                  file=sys.stderr)
    note = "(SIMULATED board values)" if a.simulate else ""
    md = compare_markdown(board, pc, fw_labels, note)
    print(md)
    with open(os.path.join(out_dir, "compare.md"), "w") as f:
        f.write(md + "\n")
    print(f"\nsnapshot in {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
