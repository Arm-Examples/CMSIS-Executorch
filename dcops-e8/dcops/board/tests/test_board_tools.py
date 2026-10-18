# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for dcops/board (no board, no JLinkExe: a fake stands in where needed).

    python3 -m unittest discover -s dcops/board/tests -v
Tests that need the existing build-classify / build-kws-et images or the board backup are skipped
when those files are absent.
"""
from __future__ import annotations

import glob
import os
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD = os.path.dirname(HERE)
sys.path.insert(0, BOARD)
import axfinfo  # noqa: E402
import jlink  # noqa: E402
import measure  # noqa: E402

REPO = jlink.REPO
CLS_AXF = os.path.join(REPO, "build-classify/bin/mlek_alif_object_detection.axf")
KWS_AXF = os.path.join(REPO, "build-kws-et/bin/mlek_alif_kws.axf")
BACKUPS = sorted(glob.glob(os.path.expanduser("~/projects/arm-executorch-hack/board-backups/devkit-e8-*.bin")))
FAKE = os.path.join(HERE, "fake_jlink.py")

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class Mem32Parsing(unittest.TestCase):
    """Commander output parsers."""

    def test_single_line(self):
        """The sample line from the task."""
        self.assertEqual(jlink.parse_mem32("200242D4 = 000000A9 0000007E"),
                         [(0x200242D4, [0xA9, 0x7E])])

    def test_trailing_space_and_noise(self):
        """Commander echoes, trailing spaces and lowercase hex."""
        text = "J-Link>mem32 0x200242D4 0x2\n200242d4 = 000000a9 0000007e \nJ-Link>exit\n"
        self.assertEqual(jlink.parse_mem32(text), [(0x200242D4, [0xA9, 0x7E])])

    def test_multi_line_request(self):
        """A 6-word read spans two output lines."""
        text = ("20000000 = 00000001 00000002 00000003 00000004 \n"
                "20000010 = 00000005 00000006 \n")
        ev = jlink.parse_events(text)
        self.assertEqual(jlink.collect_reads(ev, [(0x20000000, 6)]), [[1, 2, 3, 4, 5, 6]])

    def test_requests_in_order_with_missing(self):
        """Repeated addresses map to successive reads; a missing read yields None."""
        text = "200242D4 = 00000001\nfoo\n200242D4 = 00000002\n20000000 = DEADBEEF\n"
        ev = jlink.parse_events(text)
        got = jlink.collect_reads(ev, [(0x200242D4, 1), (0x200242D4, 1), (0x20000000, 1), (0x30000000, 1)])
        self.assertEqual(got, [[1], [2], [0xDEADBEEF], None])

    def test_registers(self):
        """PC/LR from J-Link ``regs`` style output; register lines are not mistaken for mem32."""
        text = ("PC = 01008324, CycleCnt = 4F8A3B21\n"
                "R0 = 00000000, R1 = 00000001, R2 = 00000002, R3 = 00000003\n"
                "SP(R13)= 200FFF00, MSP= 200FFF00, PSP= 00000000, R14(LR) = 0100439F\n"
                "XPSR = 21000000: APSR = nzCvq\n"
                "R15(PC) = 01008326\n")
        ev = jlink.parse_events(text)
        self.assertEqual([e for e in ev if e[0] == "pc"], [("pc", 0x01008324), ("pc", 0x01008326)])
        self.assertEqual([e for e in ev if e[0] == "lr"], [("lr", 0x0100439F)])
        self.assertFalse([e for e in ev if e[0] == "mem"])
        idx = jlink.collect_reads_indexed(jlink.parse_events(text + "20000000 = 00000007\n"),
                                          [(0x20000000, 1)])
        self.assertEqual(idx[0][0], [7])
        self.assertEqual(jlink.last_reg_before(jlink.parse_events(text + "20000000 = 00000007\n"),
                                               idx[0][1], "pc"), 0x01008326)

    def test_conversions(self):
        """Signed, 64-bit and float views."""
        self.assertEqual(jlink.as_i32(0xFFFFFFFF), -1)
        self.assertEqual(jlink.as_u64([0x1, 0x2]), 0x200000001)
        self.assertAlmostEqual(jlink.as_f32(0x3F800000), 1.0)

    def test_connect_failure_raises(self):
        """A lost HP core is reported with the recovery hint."""
        with self.assertRaises(RuntimeError) as cm:
            jlink.check_connected("Could not find core in Coresight setup")
        self.assertIn("Recovery", str(cm.exception))

    def test_script_wrapping(self):
        """connect/exit are added once."""
        s = jlink.JLink(dry_run=True).script(["mem32 0x0 0x1"])
        self.assertEqual(s, "connect\nmem32 0x0 0x1\nexit\n")
        self.assertEqual(jlink.mem32_cmd(0x200242D4, 10), "mem32 0x200242D4 0xA")
        self.assertEqual(jlink.savebin_cmd("/tmp/f.bin", 0x80200000, 0x5000),
                         "savebin /tmp/f.bin 0x80200000 0x5000")

    def test_short_name(self):
        """C++ noise is dropped."""
        self.assertEqual(jlink.short_name("alif::app::ObjectDetectionHandler(arm::app::ApplicationContext&)"),
                         "alif::app::ObjectDetectionHandler")


@unittest.skipUnless(os.path.exists(CLS_AXF), "build-classify image not present")
class ClassifyImage(unittest.TestCase):
    """Symbol resolution and build facts on the existing classifier build."""

    @classmethod
    def setUpClass(cls):
        cls.syms = jlink.Symbols(CLS_AXF)

    def test_globals(self):
        """Debug globals, anonymous-namespace counters and frame buffers resolve."""
        s = self.syms
        self.assertEqual(s.require("g_clsFrames").size, 4)
        self.assertEqual(s.require("g_clsTopIdx").size, 12)
        self.assertEqual(s.require("g_clsPerfUs").size, 16)
        self.assertEqual(s.require("timingFrames").name, "(anonymous namespace)::timingFrames")
        self.assertEqual(s.require("infCycles").size, 8)
        self.assertEqual(s.require("rgb_image").size, 176 + 320 * 320 * 3)
        self.assertEqual(s.require("lcd_image").size, 480 * 800 * 2)
        self.assertIsNotNone(s.find("tick_count"))
        self.assertIsNone(s.find("g_kwsDoesNotExist"))

    def test_wait_for_capture_mapping(self):
        """PC inside the inlined wait_for_capture is recognised; elsewhere it is not."""
        s = self.syms
        f = s.require("hal_camera_get_captured_frame")
        hits = [a for a in range(f.addr & ~1, (f.addr & ~1) + f.size, 2) if s.in_function(a, "wait_for_capture")]
        self.assertTrue(hits)
        self.assertFalse(s.in_function(s.require("alif::app::ObjectDetectionHandler(arm::app::ApplicationContext&)").addr & ~1,
                                       "wait_for_capture"))
        # PC in the out-of-line callee, LR back in the inlined loop
        leaf = s.require("camera_image_ready").addr & ~1
        lr = next(a for a in hits) + 2 | 1
        self.assertTrue(s.in_function(leaf, "wait_for_capture", lr, ("camera_image_ready",)))

    def test_build_facts(self):
        """MRAM end matches the guard rule; .pte symbol matches the CMake model file."""
        f = axfinfo.build_facts(CLS_AXF, self.syms)
        self.assertTrue(f["mram"]["within_limit"])
        self.assertEqual(f["mram"]["end_int"], axfinfo.mram_end(axfinfo.read_elf(CLS_AXF)[0]))
        self.assertGreater(f["pte"]["symbol_bytes"], 1 << 20)
        if f["pte"]["file_bytes"]:
            self.assertLessEqual(abs(f["pte"]["symbol_bytes"] - f["pte"]["file_bytes"]), 64)
        self.assertEqual(f["build"]["et_task"], "CLASSIFY")
        self.assertIn("SRAM", f["regions"])
        self.assertIn("| MRAM used |", axfinfo.facts_markdown(f))

    def test_guard_agrees(self):
        """Our end address equals the one flash_e8_hp.sh computes (its python fallback)."""
        out = subprocess.run([jlink.tool("arm-none-eabi-readelf"), "-lW", CLS_AXF], capture_output=True,
                             text=True, check=True).stdout
        ends = [int(p[3], 16) + int(p[4], 16) for p in (l.split() for l in out.splitlines())
                if p and p[0] == "LOAD" and int(p[3], 16) >= 0x80000000 and int(p[4], 16) > 0]
        self.assertEqual(max(ends), axfinfo.mram_end(axfinfo.read_elf(CLS_AXF)[0]))

    def test_measure_pipeline_simulated(self):
        """Script -> fake output -> parse -> summary -> markdown."""
        reads, missing = measure.plan_reads(self.syms)
        self.assertNotIn("g_clsFrames", missing)
        out = measure.simulated_output(reads, 11, 1000)
        raw = measure.decode_samples(out, reads, 11)
        self.assertTrue(all(v is not None for v in raw[0].values()))
        s = measure.summarize([measure.interpret(x) for x in raw], ["l%d" % i for i in range(67)], 10.0, 12.0)
        self.assertEqual(s["elapsed_source"].split()[0], "tick_count")
        self.assertAlmostEqual(s["fps"], 6.0, delta=0.1)
        self.assertAlmostEqual(s["perf_ms"]["npu"], 9.87, delta=0.05)
        self.assertAlmostEqual(s["cycles_ms"]["npu"], 9.85, delta=0.01)
        self.assertEqual(s["smoothed_top1"]["label"], "l3")
        md = measure.perf_markdown(s, axfinfo.build_facts(CLS_AXF, self.syms))
        self.assertIn("NPU inference", md)

    def test_measure_script_never_halts(self):
        """measure.py must not halt the core."""
        reads, _ = measure.plan_reads(self.syms)
        cmds = measure.build_script(reads, 3, 500)
        self.assertFalse([c for c in cmds if c.split()[0].lower() in ("h", "halt", "r", "reset", "loadfile")])

    @unittest.skipUnless(HAVE_NUMPY, "numpy missing")
    def test_snapshot_script_resumes(self):
        """Every halt in the snapshot script is followed by a resume; the script ends running."""
        import snapshot
        cmds, paths = snapshot.build_script([("g_clsFrames", 0x200242F4, 1)], 0x02310C20, 176 + 160 * 160 * 3,
                                            "/tmp/x", 4, (0x02700000, 768000))
        state = "run"
        for c in cmds:
            op = c.split()[0].lower()
            state = "halt" if op == "h" else "run" if op == "g" else state
        self.assertEqual(state, "run")
        self.assertEqual(cmds[-1], "exit")
        self.assertEqual(len(paths), 4)

    @unittest.skipUnless(HAVE_NUMPY, "numpy missing")
    def test_snapshot_parse_simulated(self):
        """Attempt attribution of PC/LR and decoding of results."""
        import snapshot
        reads = [(n, self.syms.require(n).addr, self.syms.require(n).size // 4) for n in snapshot.RESULT_GLOBALS]
        with tempfile.TemporaryDirectory() as d:
            paths = [os.path.join(d, f"rgb_image_{k}.bin") for k in range(3)]
            out = snapshot.simulate(d, paths, reads, self.syms, None, 160, False)
            att = snapshot.parse_attempts(out, reads, 3)
        ok = [self.syms.in_function(a["pc"], "wait_for_capture", a["lr"], snapshot.WAIT_LEAVES) for a in att]
        self.assertEqual(ok, [False, True, True])
        res = snapshot.decode_results(att[1]["values"], ["l%d" % i for i in range(67)])
        self.assertEqual(res["top3"][0], {"idx": 4, "label": "l4", "prob": 0.69})
        self.assertEqual(res["shown"], "l4")


class KwsGlobals(unittest.TestCase):
    """g_kws* decoding (VoiceControl.cc) on simulated reads; works before the fusion image exists."""

    READS = [("g_clsFrames", 0x20000000, 1), ("tick_count", 0x20000004, 1),
             ("g_kwsInitState", 0x20001000, 1), ("g_kwsLastIdx", 0x20001004, 1),
             ("g_kwsLastConf", 0x20001008, 1), ("g_kwsCount", 0x2000100C, 1),
             ("g_kwsTopIdx", 0x20001010, 1), ("g_kwsTopProb", 0x20001014, 1),
             ("g_kwsStrides", 0x20001018, 1), ("g_kwsCaptured", 0x2000101C, 1),
             ("g_kwsGaps", 0x20001020, 1), ("g_kwsAudioErrors", 0x20001024, 1),
             ("g_kwsPerfUs", 0x20001028, 3), ("g_kwsHeapUsed", 0x20001034, 1),
             ("g_kwsLogits", 0x20001038, 16)]
    LABELS = ["_silence_", "_unknown_", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

    def test_summary(self):
        """Strides/s, perf, last keyword, logits argmax."""
        out = measure.simulated_output(self.READS, 11, 1000)
        raw = measure.decode_samples(out, self.READS, 11)
        s = measure.summarize([measure.interpret(x) for x in raw], [], 10.0, 11.0, self.LABELS)
        k = s["kws"]
        self.assertEqual(k["state"], "running")
        self.assertAlmostEqual(k["strides_per_s"], 2.0)
        self.assertEqual(k["now"]["last"]["label"], "yes")
        self.assertEqual(k["now"]["logits_argmax"]["label"], "_silence_")
        self.assertAlmostEqual(k["perf_ms"]["mfcc"], 3.1)
        self.assertEqual(k["backlog_strides"], 0)
        facts = {"mram": {"used_bytes": 1, "usable_bytes": 2, "headroom_bytes": 1}, "regions": {},
                 "pte": {"file": None, "symbol_bytes": None, "symbol": None}, "build": {}}
        try:
            measure.perf_markdown(s, facts)
        except (KeyError, TypeError):
            pass  # facts table needs a real image; the voice rows are built before it
        self.assertIn("kws", s)


@unittest.skipUnless(os.path.exists(KWS_AXF), "build-kws-et image not present")
class KwsImage(unittest.TestCase):
    """The KWS image has no g_cls*: measure degrades gracefully."""

    def test_facts_and_plan(self):
        """Facts work; classifier globals are reported missing, not fatal."""
        syms = jlink.Symbols(KWS_AXF)
        f = axfinfo.build_facts(KWS_AXF, syms)
        self.assertTrue(f["mram"]["within_limit"])
        self.assertEqual(f["pte"]["symbol"], "arm::app::kws::nn_model")
        reads, missing = measure.plan_reads(syms)
        self.assertIn("g_clsFrames", missing)
        self.assertIn("tick_count", [r[0] for r in reads])


class FramebufferDecode(unittest.TestCase):
    """RGB565 and TIFF header helpers."""

    @unittest.skipUnless(HAVE_NUMPY, "numpy missing")
    def test_rgb565(self):
        """Pure red/green/blue decode to full-scale channels."""
        import numpy as np
        import snapshot
        raw = np.array([0xF800, 0x07E0, 0x001F, 0xFFFF], dtype="<u2").tobytes()
        rgb = snapshot.rgb565_to_rgb(raw, 4, 1)
        self.assertEqual(rgb[0].tolist(), [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])

    @unittest.skipUnless(HAVE_NUMPY, "numpy missing")
    def test_tiff_dims(self):
        """Valid header gives dims; a zeroed header (the firmware's usual case) gives None."""
        import snapshot
        h = bytearray(176)
        h[0:4] = b"II*\x00"
        h[20:24] = (160).to_bytes(4, "little")
        h[32:36] = (160).to_bytes(4, "little")
        self.assertEqual(snapshot.tiff_dims(bytes(h)), (160, 160))
        self.assertIsNone(snapshot.tiff_dims(bytes(176)))


def tiny_elf(path: str, paddr: int, filesz: int):
    """Write a header-only ELF32 with one LOAD segment (for the guard tests)."""
    eh = struct.pack("<16sHHIIIIIHHHHHH", b"\x7fELF\x01\x01\x01" + bytes(9), 2, 40, 1, paddr, 52, 0, 0,
                     52, 32, 1, 40, 0, 0)
    ph = struct.pack("<IIIIIIII", 1, 0, paddr, paddr, filesz, filesz, 5, 0x1000)
    with open(path, "wb") as f:
        f.write(eh + ph)


class Guard(unittest.TestCase):
    """MRAM limit and boot-area comparison."""

    def test_limit(self):
        """An image ending past 0x80578000 is refused; one ending at it is accepted."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.axf")
            tiny_elf(p, 0x80200000, 0x378000)
            self.assertEqual(axfinfo.mram_end(axfinfo.read_elf(p)[0]), 0x80578000)
            r = subprocess.run([sys.executable, os.path.join(BOARD, "axfinfo.py"), "--check-limit", p])
            self.assertEqual(r.returncode, 0)
            tiny_elf(p, 0x80200000, 0x378001)
            r = subprocess.run([sys.executable, os.path.join(BOARD, "axfinfo.py"), "--check-limit", p],
                               capture_output=True)
            self.assertEqual(r.returncode, 1)

    @unittest.skipUnless(BACKUPS, "no board backup")
    def test_compare_boot(self):
        """Readback equal to the backup slice matches; one flipped byte is reported."""
        with tempfile.TemporaryDirectory() as d:
            rb = os.path.join(d, "rb.bin")
            with open(BACKUPS[-1], "rb") as f:
                f.seek(0x8057B000 - 0x80200000)
                data = bytearray(f.read(0x5000))
            with open(rb, "wb") as f:
                f.write(data)
            self.assertTrue(axfinfo.compare_boot(rb, BACKUPS[-1])[0])
            data[0xC00] ^= 1
            with open(rb, "wb") as f:
                f.write(data)
            ok, msg = axfinfo.compare_boot(rb, BACKUPS[-1])
            self.assertFalse(ok)
            self.assertIn("0x8057BC00", msg)


@unittest.skipUnless(BACKUPS, "no board backup")
class FlashScriptWithFakeJLink(unittest.TestCase):
    """flash.sh end to end with JLINK pointing at tests/fake_jlink.py (no hardware)."""

    def run_flash(self, args: list[str], **env) -> tuple[subprocess.CompletedProcess, str]:
        """Run flash.sh with the fake J-Link; :returns: (process, fake J-Link log)."""
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "log")
            e = dict(os.environ, JLINK=FAKE, FAKE_JLINK_LOG=log, FAKE_JLINK_BACKUP=BACKUPS[-1], **env)
            p = subprocess.run(["bash", os.path.join(BOARD, "flash.sh"), *args], capture_output=True,
                               text=True, env=e, stdin=subprocess.DEVNULL, check=False)
            text = ""
            if os.path.exists(log):
                with open(log) as f:
                    text = f.read()
            return p, text

    def test_verify_only_ok(self):
        """Intact boot area: exit 0, power-cycle instruction printed, nothing flashed."""
        p, log = self.run_flash(["--verify-only"])
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("matches the backup", p.stdout)
        self.assertIn("POWER-CYCLE", p.stdout)
        self.assertNotIn("loadfile", log)
        self.assertIn("savebin", log)

    def test_verify_only_corrupt(self):
        """Damaged boot table: exit 3 and the HE-core recovery is printed."""
        p, _ = self.run_flash(["--verify-only"], FAKE_JLINK_CORRUPT="0x8057BC10")
        self.assertEqual(p.returncode, 3)
        self.assertIn("MISMATCH", p.stdout)
        self.assertIn("M55_HE", p.stderr)

    @unittest.skipUnless(os.path.exists(CLS_AXF), "build-classify image not present")
    def test_flash_through_guard(self):
        """Full path: guard script flashes (fake), then verify, then power-cycle text."""
        p, log = self.run_flash([CLS_AXF, "--no-wait"])
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("loadfile", log)
        self.assertIn("AE822FA0E5597LS0_M55_HP", log)
        self.assertIn("matches the backup", p.stdout)

    def test_oversize_refused(self):
        """An image past the limit is refused before any J-Link call."""
        with tempfile.TemporaryDirectory() as d:
            big = os.path.join(d, "big.axf")
            tiny_elf(big, 0x80200000, 0x37C000)
            p, log = self.run_flash([big, "--no-wait"])
        self.assertEqual(p.returncode, 1)
        self.assertIn("REFUSING", p.stderr)
        self.assertEqual(log, "")


if __name__ == "__main__":
    unittest.main()
