# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Build facts of an M55-HP .axf, without the board (stdlib only).

MRAM end and headroom (same rule as scripts/alif/flash_e8_hp.sh), embedded .pte size, memory
region usage and the CMake options the image was built with.

Usage: python3 dcops/board/axfinfo.py <axf> [--json] [--check-limit]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jlink  # noqa: E402

MRAM_BASE = 0x80200000
MRAM_LIMIT = 0x80578000    # flash_e8_hp.sh refuses images ending past this
ATOC_BASE = 0x8057BC00     # Alif boot table (DEVICE, A32_STUB, HP_APP, HE_STUB)
PARTITION_END = 0x80580000
BOOT_CHECK = (0x8057B000, 0x80580000)  # range flash.sh compares with the backup

# Fallback when no .map is next to the axf (RTSS-HP.ld for DevKit-E8).
DEFAULT_REGIONS = {"ITCM": (0x01000000, 0x00040000), "DTCM": (0x20000000, 0x00100000),
                   "SRAM": (0x02000000, 0x00700000), "UNSTRIPED": (0x02700000, 0x00100000),
                   "MRAM": (MRAM_BASE, 0x00477000)}
SHF_ALLOC = 0x2


def read_elf(path: str) -> tuple[list[dict], list[dict]]:
    """Read the program and section headers of a little-endian ELF32.

    :param path:    ELF file.
    :returns:       ``(segments, sections)`` as dicts.
    :raises ValueError: If the file is not a little-endian ELF32.
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 1:
        raise ValueError(f"{path}: not a little-endian ELF32")
    (e_phoff, e_shoff, _flags, _ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum,
     e_shstrndx) = struct.unpack_from("<IIIHHHHHH", data, 28)
    segs = []
    for i in range(e_phnum):
        p_type, p_off, vaddr, paddr, filesz, memsz, flags, _align = struct.unpack_from(
            "<IIIIIIII", data, e_phoff + i * e_phentsize)
        segs.append(dict(type=p_type, offset=p_off, vaddr=vaddr, paddr=paddr, filesz=filesz,
                         memsz=memsz, flags=flags))
    raw = [struct.unpack_from("<IIIIIIIIII", data, e_shoff + i * e_shentsize)
           for i in range(e_shnum)]
    strtab = raw[e_shstrndx][4] if raw else 0

    def name(off: int) -> str:
        end = data.index(b"\0", strtab + off)
        return data[strtab + off:end].decode()

    secs = [dict(name=name(r[0]), type=r[1], flags=r[2], addr=r[3], offset=r[4], size=r[5])
            for r in raw]
    return segs, secs


def mram_end(segs: list[dict]) -> int:
    """Highest MRAM load address, as in ``flash_e8_hp.sh`` (LOAD, paddr >= 0x80000000, filesz>0).

    :param segs:    Program headers from :func:`read_elf`.
    :returns:       End address (exclusive), or 0 when nothing loads to MRAM.
    """
    ends = [s["paddr"] + s["filesz"] for s in segs
            if s["type"] == 1 and s["paddr"] >= 0x80000000 and s["filesz"] > 0]
    return max(ends, default=0)


def map_regions(axf: str) -> dict[str, tuple[int, int]]:
    """Memory regions from the linker map next to the axf (``Memory Configuration``).

    :param axf:     ELF path; ``<axf without .axf>.map`` is read when present.
    :returns:       ``{name: (origin, length)}``.
    """
    path = os.path.splitext(axf)[0] + ".map"
    regions: dict[str, tuple[int, int]] = {}
    if os.path.exists(path):
        with open(path, errors="replace") as f:
            inside = False
            for line in f:
                if line.startswith("Memory Configuration"):
                    inside = True
                    continue
                if inside and line.startswith("Linker script and memory map"):
                    break
                m = re.match(r"^(\w+)\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)", line) if inside else None
                if m and m.group(1) != "default":
                    regions[m.group(1)] = (int(m.group(2), 16), int(m.group(3), 16))
    return regions or dict(DEFAULT_REGIONS)


def cmake_cache(axf: str) -> dict[str, str]:
    """Options from ``<build>/CMakeCache.txt`` for ``<build>/bin/<app>.axf``.

    :param axf:     ELF path.
    :returns:       ``{key: value}`` (empty when there is no cache).
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(axf))), "CMakeCache.txt")
    out: dict[str, str] = {}
    if os.path.exists(path):
        with open(path, errors="replace") as f:
            for line in f:
                m = re.match(r"^([A-Za-z0-9_]+):[A-Z]+=(.*)$", line.rstrip("\n"))
                if m:
                    out[m.group(1)] = m.group(2)
    return out


def use_case_key(cache: dict[str, str], suffix: str) -> str | None:
    """:returns: The value of ``<use_case>_<suffix>`` (e.g. MODEL_PATH), or ``None``."""
    uc = cache.get("USE_CASE_BUILD", "")
    for key in (f"{uc}_{suffix}",):
        if key in cache:
            return cache[key]
    hits = [v for k, v in cache.items() if k.endswith("_" + suffix) and not k.startswith("CMAKE")]
    return hits[0] if hits else None


def read_labels(path: str | None) -> list[str]:
    """:returns: Non-empty lines of a labels file (empty list when missing)."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def build_facts(axf: str, syms: "jlink.Symbols | None" = None) -> dict:
    """Collect board-independent facts about an image.

    :param axf:     ELF path.
    :param syms:    Pre-loaded symbols (loaded here when ``None``).
    :returns:       JSON-ready dict.
    """
    segs, secs = read_elf(axf)
    syms = syms or jlink.Symbols(axf)
    end = mram_end(segs)
    regions = map_regions(axf)
    used: dict[str, int] = {k: 0 for k in regions}
    for s in secs:
        if not s["flags"] & SHF_ALLOC or s["size"] == 0:
            continue
        for rname, (org, ln) in regions.items():
            if rname != "MRAM" and org <= s["addr"] < org + ln:
                used[rname] += s["size"]
    if "MRAM" in regions:
        used["MRAM"] = max(0, end - regions["MRAM"][0])
    big = sorted(((s["name"], s["addr"], s["size"]) for s in secs
                  if s["flags"] & SHF_ALLOC and s["size"] >= 0x10000), key=lambda t: -t[2])

    cache = cmake_cache(axf)
    model_path = use_case_key(cache, "MODEL_PATH")
    labels_path = use_case_key(cache, "CLASS_LABELS") or use_case_key(cache, "LABELS_TXT_FILE")
    nn = [s for s in syms.syms if s.name.endswith("::nn_model") or s.name == "nn_model"]
    nn.sort(key=lambda s: (s.name.endswith("kws::nn_model"), -s.size))  # vision model first
    big_syms = sorted((s for s in syms.syms if s.kind.lower() in "bd" and s.size >= 0x8000),
                      key=lambda s: -s.size)[:8]
    facts = {
        "axf": os.path.abspath(axf),
        "axf_mtime": os.path.getmtime(axf),
        "mram": {
            "end": f"0x{end:08X}", "end_int": end, "limit": f"0x{MRAM_LIMIT:08X}",
            "used_bytes": end - MRAM_BASE if end else 0,
            "usable_bytes": MRAM_LIMIT - MRAM_BASE,
            "headroom_bytes": MRAM_LIMIT - end if end else None,
            "within_limit": bool(end) and end <= MRAM_LIMIT,
        },
        "pte": {
            "symbol": nn[0].name if nn else None,
            "symbol_bytes": nn[0].size if nn else None,
            "symbol_addr": f"0x{nn[0].addr:08X}" if nn else None,
            "file": model_path,
            "file_bytes": os.path.getsize(model_path) if model_path and os.path.exists(model_path) else None,
            "all_models": [{"symbol": s.name, "bytes": s.size, "addr": f"0x{s.addr:08X}"} for s in nn],
        },
        "regions": {k: {"origin": f"0x{o:08X}", "length": ln, "used": used.get(k, 0),
                        "pct": round(100.0 * used.get(k, 0) / ln, 1) if ln else None}
                    for k, (o, ln) in regions.items() if k not in ("OSPI_FLASH", "OSPI_RAM")
                    or used.get(k)},
        "big_sections": [{"name": n, "addr": f"0x{a:08X}", "size": sz} for n, a, sz in big],
        "big_buffers": [{"name": s.name, "addr": f"0x{s.addr:08X}", "size": s.size}
                        for s in big_syms],
        "build": {
            "use_case": cache.get("USE_CASE_BUILD"),
            "build_type": cache.get("CMAKE_BUILD_TYPE"),
            "et_task": use_case_key(cache, "ET_TASK"),
            "activation_buf_sz": use_case_key(cache, "ACTIVATION_BUF_SZ"),
            "model_path": model_path,
            "labels_path": labels_path,
            "num_labels": len(read_labels(labels_path)),
            "kws_labels_path": use_case_key(cache, "KWS_LABELS"),
            "kws_model_path": use_case_key(cache, "KWS_MODEL_PATH"),
            "voice": bool(use_case_key(cache, "KWS_MODEL_PATH")),
            "npu": f"{cache.get('ETHOS_U_NPU_ID', '?')}-{cache.get('ETHOS_U_NPU_CONFIG_ID', '?')}",
        },
    }
    if facts["pte"]["file_bytes"] and facts["pte"]["symbol_bytes"]:
        # gen-model-cpp pads the array a little; a real model change moves the size by far more.
        facts["pte"]["matches_image"] = abs(facts["pte"]["file_bytes"] - facts["pte"]["symbol_bytes"]) <= 64
        if facts["pte"]["matches_image"] and model_path and os.path.getmtime(model_path) > facts["axf_mtime"]:
            facts["pte"]["note"] = "model file is newer than the image (same size)"
    return facts


def compare_boot(readback: str, backup: str, addr: int = BOOT_CHECK[0],
                 length: int = BOOT_CHECK[1] - BOOT_CHECK[0], base: int = MRAM_BASE) -> tuple[bool, str]:
    """Compare a boot-area readback with the same range of a full HP-partition backup.

    :param readback:    File read from the board at ``addr`` (``length`` bytes).
    :param backup:      Backup of the partition starting at ``base``.
    :param addr:        Start address of the readback.
    :param length:      Bytes compared.
    :param base:        Address of the first byte of ``backup``.
    :returns:           ``(match, message)``.
    """
    with open(readback, "rb") as f:
        got = f.read()
    with open(backup, "rb") as f:
        f.seek(addr - base)
        want = f.read(length)
    if len(got) != length or len(want) != length:
        return False, (f"size mismatch: read {len(got)} bytes, backup slice {len(want)} bytes "
                       f"(expected {length})")
    diff = [i for i in range(length) if got[i] != want[i]]
    if diff:
        return False, (f"MISMATCH: {len(diff)} bytes differ, first at 0x{addr + diff[0]:08X} "
                       f"(read {got[diff[0]]:02X}, backup {want[diff[0]]:02X}), "
                       f"last at 0x{addr + diff[-1]:08X}")
    return True, f"boot area 0x{addr:08X}..0x{addr + length:08X} matches the backup ({length} bytes)"


def mib(n: int | None) -> str:
    """:returns: ``n`` bytes as MiB text."""
    return "?" if n is None else f"{n / 1048576:.2f} MiB"


def facts_markdown(f: dict) -> str:
    """Render :func:`build_facts` as a markdown table.

    :param f:       Facts dict.
    :returns:       Markdown text.
    """
    m, p, b = f["mram"], f["pte"], f["build"]
    rows = [
        ("Image", f"`{os.path.relpath(f['axf'], jlink.REPO)}`"),
        ("Use case / task", f"{b['use_case']} / {b['et_task'] or '-'} ({b['build_type']}, {b['npu']})"),
        ("MRAM end (limit)", f"{m['end']} ({m['limit']}) -> "
                             f"{'OK' if m['within_limit'] else '**OVER LIMIT, do not flash**'}"),
        ("MRAM used", f"{mib(m['used_bytes'])} of {mib(m['usable_bytes'])} usable, "
                      f"{(m['headroom_bytes'] or 0) // 1024} KB headroom"),
        (".pte embedded", f"{p['symbol_bytes'] / 1024:.0f} KB ({p['symbol_bytes'] / 1048576:.2f} MiB, "
                          f"`{p['symbol']}`)" if p["symbol_bytes"] else "-"),
        (".pte file", (f"{p['file_bytes'] / 1024:.0f} KB `{os.path.basename(p['file'] or '')}`"
                       + ("" if p.get("matches_image", True) else
                          " -- **differs from the embedded model: image is stale vs MODEL_PATH, rebuild**"))
                      if p["file_bytes"] else (p["file"] or "-")),
        ("Labels", f"{b['num_labels']} from `{os.path.basename(b['labels_path'] or '-')}`"),
    ]
    for mdl in p.get("all_models", [])[1:]:
        rows.append((f".pte embedded ({mdl['symbol'].split('::')[-2]})", f"{mdl['bytes'] / 1024:.0f} KB"))
    if b.get("kws_labels_path"):
        rows.append(("KWS labels", f"{len(read_labels(b['kws_labels_path']))} from "
                                   f"`{os.path.basename(b['kws_labels_path'])}`"))
    rows += [
        ("Activation arena", b["activation_buf_sz"] or "-"),
    ]
    for k, r in f["regions"].items():
        if k == "MRAM":
            continue
        rows.append((f"{k} used", f"{mib(r['used'])} of {mib(r['length'])} ({r['pct']}%)"))
    lines = ["| Build fact | Value |", "|---|---|"] + [f"| {a} | {v} |" for a, v in rows]
    return "\n".join(lines)


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("axf", nargs="?")
    ap.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    ap.add_argument("--check-limit", action="store_true",
                    help="exit 1 if the image ends past 0x80578000 (prints the end address)")
    ap.add_argument("--compare-boot", nargs=2, metavar=("READBACK", "BACKUP"),
                    help="compare a 0x8057B000..0x80580000 readback with the backup; exit 1 if not equal")
    a = ap.parse_args()
    if a.compare_boot:
        ok, msg = compare_boot(*a.compare_boot)
        print(msg)
        sys.exit(0 if ok else 1)
    if not a.axf:
        ap.error("axf is required")
    if a.check_limit:
        end = mram_end(read_elf(a.axf)[0])
        print(f"image ends at 0x{end:08X} (limit 0x{MRAM_LIMIT:08X})")
        sys.exit(0 if 0 < end <= MRAM_LIMIT else 1)
    f = build_facts(a.axf)
    print(json.dumps(f, indent=2) if a.json else facts_markdown(f))


if __name__ == "__main__":
    main()
