#!/usr/bin/env python3
# Copyright (c) 2026 Sentinel contributors.
# SPDX-License-Identifier: Apache-2.0
"""Summarise a Sentinel UART log (the `SN t=...` lines, 115200 8N1 on UART4).

usage: python3 tools/sentinel_log_report.py LOG [LOG ...]

For each log it reports:
- the window count, the board-time span and any restart (uptime going back);
- learn starts and saved/failed baselines;
- state counts and every state change. For a change to CHECK or ALERT it
  gives the time since the start of the unbroken run of windows whose mic or
  vib score exceeded 4.0 and that led to the change (the detection latency
  from the change in sound or vibration);
- per-100-ms-block and per-window processing time, the share of new
  accelerometer samples, and the final FIFO error and ring-full counts.

It reads the console format of board/DevKit-E8/sentinel.c and needs only the
Python standard library.
"""
import re
import statistics
import sys

LINE = re.compile(
    rb"SN t=(\d+) (IDLE|VERIFYING|LEARNING|OK|CHECK|ALERT)(?: \d+/\d+)? "
    rb"(?:z=\S+ mic=([\d.]+)(!?) vib=([\d.]+)(!?) )?f=\S+ imu=(\d+)/(\d+) "
    rb"blk_us=(\d+) win_us=(\d+) err=(\d+) full=(\d+)")
EVENT = re.compile(rb"SN: learning \w+ normal|SN PROFILE \w+:? [^\r\n]*|SN: selected [^\r\n]*")


def report(path):
    data = open(path, "rb").read()
    rows = []
    for m in LINE.finditer(data):
        g = m.groups()
        rows.append(dict(t=int(g[0]) / 1000.0, state=g[1].decode(),
                         mic=float(g[2] or 0), vib=float(g[4] or 0),
                         imu_new=int(g[6]), imu_read=int(g[7]),
                         blk_us=int(g[8]), win_us=int(g[9]),
                         err=int(g[10]), full=int(g[11])))
    print(f"== {path}")
    if not rows:
        print("   no Sentinel lines")
        return
    restarts = [(a["t"], b["t"]) for a, b in zip(rows, rows[1:]) if b["t"] < a["t"]]
    print(f"   {len(rows)} windows, board time {rows[0]['t']:.1f}-{rows[-1]['t']:.1f} s"
          + (f", restarts at {', '.join(f'{a:.1f}->{b:.1f} s' for a, b in restarts)}" if restarts else ", no restart"))
    events = [e.decode() for e in EVENT.findall(data)]
    learns = sum(1 for e in events if e.startswith("SN: learning"))
    saved = [e for e in events if "saved in MRAM" in e]
    failed = [e for e in events if "SAVE FAILED" in e]
    print(f"   learn starts {learns}, saved {len(saved)}, save failures {len(failed)}")
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    print("   states: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("   state changes:")
    for prev, cur in zip(rows, rows[1:]):
        if cur["state"] == prev["state"] or cur["t"] < prev["t"]:
            continue
        note = ""
        if cur["state"] in ("CHECK", "ALERT"):
            # Start of the unbroken run of windows with a score > 4 that
            # ends at this change (the 3-window rule needs such a run)
            i = rows.index(cur)
            j = i
            while j > 0 and max(rows[j - 1]["mic"], rows[j - 1]["vib"]) > 4.0 \
                    and rows[j - 1]["t"] < rows[j]["t"]:
                j -= 1
            if max(rows[j]["mic"], rows[j]["vib"]) > 4.0:
                note = f" ({cur['t'] - rows[j]['t']:.0f} s after the run of scores > 4 began at {rows[j]['t']:.1f} s)"
        print(f"     {cur['t']:9.1f} s  {prev['state']} -> {cur['state']}{note}")
    blk = [r["blk_us"] for r in rows if r["blk_us"]]
    win = [r["win_us"] for r in rows if r["win_us"]]
    if blk:
        print(f"   block time median {statistics.median(blk):.0f} us, max {max(blk)} us "
              f"(per 100 ms block = {100 * max(blk) / 100000:.2f} % CPU at worst)")
        if win:
            print(f"   window time median {statistics.median(win):.0f} us, max {max(win)} us "
                  f"(the max includes saving a learned baseline to MRAM, if one was saved)")
    else:
        print("   block/window time not measured (DWT counter not running)")
    new = sum(r["imu_new"] for r in rows)
    read = sum(r["imu_read"] for r in rows)
    if read:
        print(f"   accelerometer: {100.0 * new / read:.1f} % new samples ({new}/{read})")
    print(f"   final FIFO errors {rows[-1]['err']}, ring-full stops {rows[-1]['full']}")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    for path in argv[1:]:
        report(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
