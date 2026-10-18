# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""J-Link Commander and ELF symbol helpers for the DevKit-E8 board tools (stdlib only).

The serial console is not on the PRG USB, so on-board state is read from global variables over
J-Link: ``mem32`` works while the core runs, ``h``/``g`` halt and resume, ``savebin`` dumps RAM.

- :class:`JLink` runs a Commander script (or prints it with ``dry_run``).
- :func:`parse_events` / :func:`collect_reads` turn Commander output into register and memory values.
- :class:`Symbols` resolves addresses and sizes from ``arm-none-eabi-nm -C -S`` (anonymous
  namespaces included) and maps a PC to its inline call chain with ``addr2line -f -i``.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field

JLINK_EXE = os.environ.get("JLINK", "/Applications/SEGGER/JLink/JLinkExe")
DEVICE_HP = "AE822FA0E5597LS0_M55_HP"
DEVICE_HE = "AE822FA0E5597LS0_M55_HE"
SWD_KHZ = 4000

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


# ----------------------------------------------------------------------------- toolchain


def tool(name: str) -> str:
    """Find an Arm GNU toolchain binary.

    :param name:    Tool name, e.g. ``arm-none-eabi-nm``.
    :returns:       Absolute path; PATH first, then the vcpkg artifacts that ``env.sh`` uses.
    :raises FileNotFoundError: If the tool is nowhere to be found.
    """
    found = shutil.which(name)
    if found:
        return found
    pattern = os.path.expanduser(
        f"~/.vcpkg/downloads/artifacts/*/compilers.arm.arm.none.eabi.gcc/*/bin/{name}")
    hits = sorted(glob.glob(pattern))
    if hits:
        return hits[-1]
    raise FileNotFoundError(f"{name} not found: source ~/projects/arm-executorch-hack/env.sh")


# ----------------------------------------------------------------------------- J-Link


def mem32_cmd(addr: int, words: int) -> str:
    """:returns: A Commander ``mem32`` line (Commander parses numbers as hex)."""
    return f"mem32 0x{addr:08X} 0x{max(words, 1):X}"  # space-separated: this J-Link rejects the comma form


def savebin_cmd(path: str, addr: int, nbytes: int) -> str:
    """:returns: A Commander ``savebin`` line."""
    return f"savebin {path} 0x{addr:08X} 0x{nbytes:X}"  # space-separated: this J-Link rejects the comma form


@dataclass
class JLink:
    """Runs J-Link Commander scripts against one core.

    :param device:  J-Link device name.
    :param dry_run: Print the script instead of running it; :meth:`run` then returns ``fake``.
    :param fake:    Output returned in dry-run mode (lets the parsers be exercised offline).
    """

    device: str = DEVICE_HP
    speed: int = SWD_KHZ
    exe: str = JLINK_EXE
    dry_run: bool = False
    fake: str = ""
    timeout: float = 120.0
    log: list = field(default_factory=list)

    def script(self, cmds: list[str]) -> str:
        """Build a full Commander script: ``connect``, the commands, ``exit``.

        :param cmds:    Commander lines.
        :returns:       Script text.
        """
        body = [c for c in cmds if c.strip()]
        if not body or body[0].strip().lower() != "connect":
            body.insert(0, "connect")
        if body[-1].strip().lower() not in ("exit", "q", "qc"):
            body.append("exit")
        return "\n".join(body) + "\n"

    def argv(self, script_path: str) -> list[str]:
        """:returns: The JLinkExe command line for ``script_path``."""
        return [self.exe, "-device", self.device, "-if", "SWD", "-speed", str(self.speed),
                "-autoconnect", "1", "-CommanderScript", script_path]

    def run(self, cmds: list[str], label: str = "") -> str:
        """Run (or print) a Commander script.

        :param cmds:    Commander lines; ``connect``/``exit`` are added when missing.
        :param label:   Shown in the dry-run banner.
        :returns:       Commander stdout+stderr (``fake`` in dry-run mode).
        :raises RuntimeError: If JLinkExe is missing, times out, or reports a connect failure.
        """
        text = self.script(cmds)
        self.log.append(text)
        if self.dry_run:
            print(f"----- J-Link script{(' (' + label + ')') if label else ''}: "
                  f"{' '.join(self.argv('<script>'))}")
            print(text, end="")
            print("----- end script")
            return self.fake
        if not os.path.exists(self.exe):
            raise RuntimeError(f"JLinkExe not found at {self.exe} (set $JLINK)")
        with tempfile.NamedTemporaryFile("w", suffix=".jlink", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            proc = subprocess.run(self.argv(path), capture_output=True, text=True,
                                  timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"JLinkExe timed out after {self.timeout} s") from e
        finally:
            os.unlink(path)
        out = proc.stdout + proc.stderr
        check_connected(out)
        return out


CONNECT_ERRORS = ("Cannot connect to target", "Could not find core", "Failed to power up DAP",
                  "Could not read AHB ROM", "No J-Link found", "Connecting to J-Link failed",
                  "Could not connect to target")


def check_connected(out: str):
    """Raise if Commander output shows a failed connection.

    :param out:     Commander output.
    :raises RuntimeError: With a recovery hint when the HP core is unreachable.
    """
    for err in CONNECT_ERRORS:
        if err.lower() in out.lower():
            hint = ""
            if "core" in err.lower() or "AHB" in err:
                hint = (" -- if the HP core no longer connects but the HE core does, the boot table"
                        " was overwritten: see dcops/board/README.md 'Recovery'.")
            raise RuntimeError(f"J-Link: '{err}'{hint}\n--- output tail ---\n{out[-2000:]}")


# ----------------------------------------------------------------------------- output parsing

MEM_RE = re.compile(r"^\s*([0-9A-Fa-f]{8})\s*=\s*((?:[0-9A-Fa-f]{8}(?:\s+|$))+)")
PC_RE = re.compile(r"(?<![\w(])(?:PC|R15)\s*(?:\(PC\))?\s*=\s*([0-9A-Fa-f]{8})")
LR_RE = re.compile(r"(?<![\w(])(?:R14\s*\(LR\)|R14|LR)\s*=\s*([0-9A-Fa-f]{8})")


def parse_mem32(text: str) -> list[tuple[int, list[int]]]:
    """Parse ``mem32`` output lines such as ``200242D4 = 000000A9 0000007E``.

    :param text:    Commander output.
    :returns:       ``(address, [words])`` per output line, in order.
    """
    rows = []
    for line in text.splitlines():
        m = MEM_RE.match(line)
        if m:
            rows.append((int(m.group(1), 16), [int(w, 16) for w in m.group(2).split()]))
    return rows


def parse_events(text: str) -> list[tuple]:
    """Tokenise Commander output into ordered events.

    :param text:    Commander output.
    :returns:       ``("pc", v)``, ``("lr", v)`` and ``("mem", addr, [words])`` tuples in order.
    """
    events: list[tuple] = []
    for line in text.splitlines():
        m = MEM_RE.match(line)
        if m:
            events.append(("mem", int(m.group(1), 16), [int(w, 16) for w in m.group(2).split()]))
            continue
        for m in PC_RE.finditer(line):
            events.append(("pc", int(m.group(1), 16)))
        for m in LR_RE.finditer(line):
            events.append(("lr", int(m.group(1), 16)))
    return events


def collect_reads(events: list[tuple], requests: list[tuple[int, int]]) -> list[list[int] | None]:
    """Match ``mem32`` requests (in script order) to output lines.

    A request for N words may span several output lines (Commander prints 4 words per line).
    Lines whose address does not continue the current request are skipped, so stray output is
    tolerated; a request with no output yields ``None``.

    :param events:      Output of :func:`parse_events` (or ``("mem", ...)`` tuples).
    :param requests:    ``(address, words)`` in the order the script issued them.
    :returns:           Word lists aligned with ``requests``.
    """
    return [w for w, _ in collect_reads_indexed(events, requests)]


def collect_reads_indexed(events: list[tuple],
                          requests: list[tuple[int, int]]) -> list[tuple[list[int] | None, int]]:
    """Like :func:`collect_reads`, also giving the index in ``events`` where each read starts
    (-1 when missing), so register events can be attributed to the right read.

    :param events:      Output of :func:`parse_events`.
    :param requests:    ``(address, words)`` in script order.
    :returns:           ``(words | None, event_index)`` aligned with ``requests``.
    """
    mems = [(k, e[1], e[2]) for k, e in enumerate(events) if e[0] == "mem"]
    out: list[tuple[list[int] | None, int]] = []
    i = 0
    for addr, n in requests:
        words: list[int] = []
        j = i
        while j < len(mems) and mems[j][1] != addr:
            j += 1
        if j == len(mems):
            out.append((None, -1))
            continue
        start = mems[j][0]
        while j < len(mems) and len(words) < n and mems[j][1] == addr + 4 * len(words):
            words.extend(mems[j][2])
            j += 1
        i = j
        out.append((words[:n] if len(words) >= n else None, start))
    return out


def last_reg_before(events: list[tuple], index: int, reg: str) -> int | None:
    """:returns: The value of the last ``reg`` (``"pc"``/``"lr"``) event before ``index``."""
    for e in reversed(events[:index] if index >= 0 else events):
        if e[0] == reg:
            return e[1]
    return None


def events_with_index(events: list[tuple]) -> list[tuple[int, tuple]]:
    """:returns: ``(index, event)`` pairs, handy to find the last PC before a given read."""
    return list(enumerate(events))


def as_i32(w: int) -> int:
    """:returns: ``w`` reinterpreted as a signed 32-bit value."""
    return w - (1 << 32) if w & 0x80000000 else w


def as_u64(words: list[int]) -> int:
    """:returns: Little-endian 64-bit value from two words."""
    return words[0] | (words[1] << 32)


def as_f32(w: int) -> float:
    """:returns: ``w`` reinterpreted as an IEEE float."""
    return struct.unpack("<f", struct.pack("<I", w))[0]


# ----------------------------------------------------------------------------- symbols


@dataclass
class Sym:
    """One ELF symbol."""

    name: str
    addr: int
    size: int
    kind: str


class Symbols:
    """Symbol table of an .axf from ``nm -C -S``, plus ``addr2line`` lookups.

    :param axf: Path to the ELF image.
    """

    ANON = "(anonymous namespace)::"

    def __init__(self, axf: str):
        self.axf = axf
        out = subprocess.run([tool("arm-none-eabi-nm"), "-C", "-S", axf], capture_output=True,
                             text=True, check=True).stdout
        self.syms = self.parse_nm(out)
        self.by_name: dict[str, Sym] = {}
        for s in self.syms:
            self.by_name.setdefault(s.name, s)
        self._a2l: dict[int, list[tuple[str, str]]] = {}

    @staticmethod
    def parse_nm(out: str) -> list[Sym]:
        """Parse ``nm -C -S`` output (lines without a size are kept with size 0).

        :param out: nm output.
        :returns:   Symbols in nm order.
        """
        syms = []
        for line in out.splitlines():
            m = re.match(r"^([0-9a-fA-F]{8})\s+(?:([0-9a-fA-F]{8})\s+)?([A-Za-z])\s+(.+)$", line)
            if m:
                syms.append(Sym(m.group(4).strip(), int(m.group(1), 16),
                                int(m.group(2), 16) if m.group(2) else 0, m.group(3)))
        return syms

    def find(self, name: str) -> Sym | None:
        """Look a symbol up by exact name, then by ``(anonymous namespace)::name``, then by a
        unique ``::name`` suffix (data symbols preferred).

        :param name:    Plain or qualified name.
        :returns:       The symbol, or ``None``.
        """
        if name in self.by_name:
            return self.by_name[name]
        if self.ANON + name in self.by_name:
            return self.by_name[self.ANON + name]
        hits = [s for s in self.syms if s.name.endswith("::" + name)]
        data = [s for s in hits if s.kind.lower() in "bdgrs"]
        hits = data or hits
        return hits[0] if len({(s.addr, s.size) for s in hits}) == 1 else None

    def require(self, name: str) -> Sym:
        """:returns: The symbol. :raises KeyError: When it is missing."""
        s = self.find(name)
        if s is None:
            raise KeyError(f"symbol {name} not in {self.axf}")
        return s

    def matching(self, regex: str, kinds: str = "bd") -> list[Sym]:
        """Data symbols whose unqualified name matches ``regex``.

        :param regex:   Pattern searched in the name after the last ``::``.
        :param kinds:   nm type letters accepted (lowercased); default writable data/bss only, so
                        const tables such as ``g_kwsLabels`` are left out.
        :returns:       Symbols sorted by address.
        """
        rx = re.compile(regex)
        seen, out = set(), []
        for s in self.syms:
            base = s.name.split("::")[-1]
            if s.kind.lower() in kinds and rx.search(base) and s.name not in seen:
                seen.add(s.name)
                out.append(s)
        return sorted(out, key=lambda s: s.addr)

    def inline_chain(self, addr: int) -> list[tuple[str, str]]:
        """Inline call chain at ``addr``, innermost first.

        :param addr:    Code address.
        :returns:       ``(function, file:line)`` pairs from ``addr2line -f -i -C``.
        """
        if addr in self._a2l:
            return self._a2l[addr]
        out = subprocess.run([tool("arm-none-eabi-addr2line"), "-e", self.axf, "-f", "-i", "-C",
                              "-s", f"0x{addr:x}"], capture_output=True, text=True,
                             check=False).stdout.splitlines()
        chain = [(out[k].strip(), out[k + 1].strip() if k + 1 < len(out) else "?")
                 for k in range(0, len(out), 2)]
        self._a2l[addr] = chain
        return chain

    def function_at(self, addr: int) -> str:
        """:returns: The symbol-table function containing ``addr`` (outermost, not inlined)."""
        best = None
        for s in self.syms:
            if s.kind.lower() in "tw" and s.size and s.addr <= (addr | 1) < s.addr + s.size + 1:
                if best is None or s.size < best.size:
                    best = s
        return best.name if best else "?"

    def in_function(self, pc: int, func: str, lr: int | None = None,
                    leaf_ok: tuple[str, ...] = ()) -> bool:
        """Whether ``pc`` executes ``func`` (inlined or not).

        :param pc:      Halted PC.
        :param func:    Function name to look for in the inline chain.
        :param lr:      Halted LR; used when the PC is in a leaf callee listed in ``leaf_ok``.
        :param leaf_ok: Out-of-line callees of ``func`` (e.g. ``camera_image_ready``).
        :returns:       True if ``func`` is on the inline chain of ``pc`` (or of ``lr`` when the PC
                        is in one of ``leaf_ok``).
        """
        names = [f for f, _ in self.inline_chain(pc)]
        if any(n == func or n.endswith("::" + func) for n in names):
            return True
        if lr is not None and (set(names) & set(leaf_ok) or self.function_at(pc) in leaf_ok):
            lr_names = [f for f, _ in self.inline_chain((lr & ~1) - 2)]
            return any(n == func for n in lr_names)
        return False

    def describe(self, addr: int) -> str:
        """:returns: ``inner <- outer`` inline chain text for ``addr``."""
        chain = self.inline_chain(addr)
        return " <- ".join(f"{short_name(f)} ({loc.split(' ')[0]})" for f, loc in chain) \
            or self.function_at(addr)


def short_name(func: str, limit: int = 60) -> str:
    """Drop C++ template/parameter noise: ``a::b<..>::c(int, ...) const`` -> ``a::b::c``.

    :param func:    Demangled name.
    :param limit:   Maximum length kept.
    :returns:       Shortened name.
    """
    out, depth = [], 0
    for ch in func:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    s = "".join(out).replace(" const", "").strip()
    return s if len(s) <= limit else "..." + s[-(limit - 3):]
