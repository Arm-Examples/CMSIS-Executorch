#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Stand-in for JLinkExe used by the offline tests (JLINK=<this file>). Never touches hardware.

Reads the -CommanderScript file, logs it to $FAKE_JLINK_LOG, and answers:
- ``savebin path addr n`` -> writes n bytes from $FAKE_JLINK_BACKUP (partition image at
  0x80200000), optionally with byte $FAKE_JLINK_CORRUPT (hex address) flipped;
- ``mem32 addr n`` -> prints n zero words in Commander's format;
- ``loadfile`` -> prints a success line.
"""
import os
import sys

args = sys.argv[1:]
script = args[args.index("-CommanderScript") + 1]
with open(script) as fh:
    cmds = [l.strip() for l in fh if l.strip()]
log = os.environ.get("FAKE_JLINK_LOG")
if log:
    with open(log, "a") as f:
        f.write(" ".join(args[:-1]) + "\n" + "\n".join(cmds) + "\n---\n")
print("SEGGER J-Link Commander (FAKE)")
for c in cmds:
    print(f"J-Link>{c}")
    op = c.split()[0].lower()
    parts = c[len(op):].replace(",", " ").split()  # accepts space- (real J-Link) or comma-separated args
    if op == "savebin":
        path, addr, n = parts[0], int(parts[1], 16), int(parts[2], 16)
        with open(os.environ["FAKE_JLINK_BACKUP"], "rb") as f:
            f.seek(addr - 0x80200000)
            data = bytearray(f.read(n))
        bad = os.environ.get("FAKE_JLINK_CORRUPT")
        if bad:
            data[int(bad, 16) - addr] ^= 0xFF
        with open(path, "wb") as f:
            f.write(data)
        print(f"Reading {n} bytes from addr 0x{addr:08X} into file...O.K.")
    elif op == "mem32":
        addr, n = int(parts[0], 16), int(parts[1], 16)
        for j in range(0, n, 4):
            print(f"{addr + 4 * j:08X} = " + " ".join("00000000" for _ in range(min(4, n - j))))
    elif op == "loadfile":
        print("Downloading file... O.K.")
