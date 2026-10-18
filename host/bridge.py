"""Replay held-out BCI IV 2a trials to the board over UART at 250 Hz and republish its output.

    python host/bridge.py --source replay --subject 3 --session 2 --board /dev/tty.usbmodemXXXX
    python host/bridge.py --source replay --subject 3 --board sim            # no hardware

Session: RESET, SET_SUBJECT [, FREE_RUN 1], then a continuous stream of data frames from
``--lead-in`` s before the first trial's cue through the last trial's window, with a CUE
control frame (arg = true label) immediately before each cue sample. Data frame k is sent no
earlier than t0 + k/250 s (monotonic clock, ~10 ms batches, no drift).

Only this process opens the serial port. Every board JSON line is enriched with the host's
expectation and republished as a UDP datagram to ``--publish`` (default 127.0.0.1:5555),
together with ``{"event":"cue"}``, ``{"event":"sample"}`` (every 10th sample, 4 channels),
``{"event":"log"}`` and ``{"event":"session"}`` records; ``demo_ui.py`` and ``measure.py``
listen there. One summary line per trial goes to stdout.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FS, REFERENCE_DIR, UDP_DEFAULT, UV_SCALE, load_offline_predictions, load_params,  # noqa: E402
                    parse_hostport, parse_trials)
from drivers.replay import Plan, Replay, default_trials, iter_frames, load_replay, plan_session  # noqa: E402

FEED_CHANNELS = ("C3", "Cz", "C4", "FCz")
FEED_EVERY = 10
EXPECTED_JSON = REFERENCE_DIR / "expected_board_predictions.json"


def load_expected(subject: int, uv_scale: float) -> dict[int, dict]:
    """``{session_trial: {"cls", "logits"}}`` of the int8 model on the int16-stream windows
    (written by ``replay_accuracy.py``), or ``{}``."""
    if not EXPECTED_JSON.exists():
        return {}
    d = json.loads(EXPECTED_JSON.read_text())
    if abs(float(d.get("uv_scale", UV_SCALE)) - uv_scale) > 1e-9:
        return {}
    s = d["subjects"].get(str(subject))
    if not s:
        return {}
    return {int(t): {"cls": int(np.argmax(l)), "logits": l} for t, l in zip(s["trial"], s["logits"])}


class Bridge:
    def __init__(self, board, rep: Replay, plan: Plan, publish: tuple[str, int] | None, log_path: str | None,
                 offline: dict, expected: dict, quiet: bool = False):
        self.board, self.rep, self.plan = board, rep, plan
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if publish else None
        self.addr = publish
        self.log = open(log_path, "a") if log_path else None
        self.off = offline.get(plan.subject)
        self.expected = expected
        self.quiet = quiet
        self.results: list[dict] = []
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.ready_line = None
        self.feed_idx = [rep.channel_order.index(c) for c in FEED_CHANNELS] if rep.channel_order else [7, 9, 11, 3]

    # ---- output side ---------------------------------------------------------------------
    def publish(self, rec: dict) -> None:
        msg = json.dumps(rec, separators=(",", ":"))
        if self.sock:
            try:
                self.sock.sendto(msg.encode(), self.addr)
            except OSError:
                pass
        if self.log:
            with self.lock:
                self.log.write(msg + "\n")

    def offline_cls(self, session_trial: int) -> int | None:
        if self.off is None:
            return None
        hit = np.nonzero(self.off["trial"] == session_trial)[0]
        return int(np.argmax(self.off["logits"][hit[0]])) if len(hit) else None

    def handle_line(self, raw: bytes) -> None:
        line = raw.decode(errors="replace").strip()
        if not line:
            return
        if not line.startswith("{"):
            if line.startswith("READY"):
                self.ready_line = line
            self.publish({"event": "log", "line": line, "host_t": time.time()})
            if not self.quiet:
                print(f"[board] {line}", flush=True)
            return
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            self.publish({"event": "log", "line": f"bad json: {line}", "host_t": time.time()})
            return
        rec["event"] = "pred"
        rec["host_t"] = time.time()
        k = int(rec.get("trial", -1))
        if 0 <= k < len(self.plan.trials):
            st = self.plan.trials[k]
            rec["session_trial"] = st
            rec["true_label"] = self.plan.labels[k]
            oc = self.offline_cls(st)
            rec["offline_cls"] = oc
            raw_cls = int(rec.get("raw_cls", rec.get("cls", -1)))
            rec["match_offline"] = None if oc is None else int(raw_cls == oc)
            ex = self.expected.get(st)
            rec["expected_cls"] = None if ex is None else ex["cls"]
            rec["match_expected"] = None if ex is None else int(raw_cls == ex["cls"])
            rec["correct"] = int(raw_cls == self.plan.labels[k])
            with self.lock:
                self.results.append(rec)
            if not self.quiet:
                print(f"trial {st} label {rec['true_label']} cls {rec.get('cls')} conf {float(rec.get('conf', 0)):.2f} "
                      f"lat {rec.get('lat_us')} us motion {rec.get('motion')} match offline {rec['match_offline']}"
                      + ("" if ex is None else f" expected {rec['match_expected']}"), flush=True)
        self.publish(rec)

    def reader(self) -> None:
        while not self.stop.is_set():
            try:
                raw = self.board.readline()
            except Exception as exc:  # noqa: BLE001  (serial unplugged etc.)
                print(f"[bridge] read error: {exc}", file=sys.stderr, flush=True)
                time.sleep(0.2)
                continue
            if raw:
                self.handle_line(raw)

    # ---- input side ----------------------------------------------------------------------
    def feed(self, k: int) -> None:
        v = self.rep.x_int16[self.feed_idx, self.plan.start + k] / self.rep.uv_scale
        self.publish({"event": "sample", "seq": k & 0xFFFF, "k": k, "ch": list(FEED_CHANNELS),
                      "uv": [round(float(a), 1) for a in v]})

    def stream(self, realtime: bool = True, free_run: bool = False, batch_s: float = 0.010) -> float:
        self.publish({"event": "session", "subject": self.plan.subject, "trials": [self.plan.trials[0], self.plan.trials[-1]],
                      "n_trials": len(self.plan.trials), "start": self.plan.start, "n": self.plan.n,
                      "uv_scale": self.rep.uv_scale, "host_t": time.time()})
        pending = bytearray()
        t0 = time.monotonic()
        chunk = 0 if realtime else 4096
        for k, frame, ev in iter_frames(self.rep, self.plan, free_run=free_run, chunk=chunk):
            if realtime:
                due = t0 + k / FS
                if due > time.monotonic():
                    if pending:
                        self.board.write(bytes(pending))
                        pending.clear()
                    time.sleep(max(0.0, due + batch_s - 1.0 / FS - time.monotonic()))
                pending += frame
                if ev is None and frame[1] == 0x55 and k % FEED_EVERY == 0:
                    self.feed(k)
            else:
                self.board.write(frame)
            if ev is not None:
                ev["host_t"] = time.time()
                self.publish(ev)
        if pending:
            self.board.write(bytes(pending))
        return time.monotonic() - t0

    def wait_results(self, n: int, timeout: float) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                if len(self.results) >= n:
                    return
            time.sleep(0.02)

    def summary(self) -> dict:
        r = self.results
        def frac(key):
            v = [x[key] for x in r if x.get(key) is not None]
            return (sum(v), len(v))
        lat = np.array([x.get("lat_us", 0) for x in r], dtype=float)
        return {"n": len(r), "correct": frac("correct"), "match_offline": frac("match_offline"),
                "match_expected": frac("match_expected"), "gated": sum(1 for x in r if x.get("motion")),
                "lat_mean_us": float(lat.mean()) if len(lat) else None}


def open_board(port: str, baud: int, sim_kwargs: dict):
    if port == "sim":
        from drivers.sim_board import SimBoard

        return SimBoard(**sim_kwargs)
    try:
        import serial  # pyserial
    except ImportError as exc:
        raise SystemExit("pyserial missing: pip install -r host/requirements.txt (or use --board sim)") from exc
    try:
        return serial.Serial(port, baud, timeout=0.1, write_timeout=2.0)
    except serial.SerialException as exc:
        raise SystemExit(f"cannot open {port}: {exc}\n(close other programs using the port: screen, "
                         "VS Code serial monitor, a second bridge; only bridge.py may own it)") from exc


def run(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=["replay"], default="replay")
    ap.add_argument("--subject", type=int, default=3)
    ap.add_argument("--session", type=int, default=2)
    ap.add_argument("--board", required=True, help="serial port, or 'sim' for the in-process board")
    ap.add_argument("--trials", default=default_trials(), help="0-based session trials, e.g. 24-143 or 33-37")
    ap.add_argument("--lead-in", type=float, default=10.0)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--publish", default=UDP_DEFAULT, help="UDP host:port, or 'none'")
    ap.add_argument("--free-run", action="store_true")
    ap.add_argument("--loop", action="store_true", help="repeat the session (RESET each time) until Ctrl-C")
    ap.add_argument("--fast", action="store_true", help="no 250 Hz pacing (sim board only)")
    ap.add_argument("--sim-model", choices=["offline", "int8"], default="offline")
    ap.add_argument("--params", default=None)
    ap.add_argument("--replay-file", default=None)
    ap.add_argument("--log", default=None, help="append every published record to this JSONL file")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.fast and a.board != "sim":
        ap.error("--fast only with --board sim (the board needs real time)")

    params = load_params(a.params)
    rep = load_replay(a.subject, a.session, a.replay_file)
    if rep.channel_order and rep.channel_order != list(params["channel_order"]):
        raise SystemExit("replay channel order != params.json channel_order")
    plan = plan_session(rep, parse_trials(a.trials), a.lead_in)
    offline = load_offline_predictions()
    expected = load_expected(a.subject, rep.uv_scale)
    board = open_board(a.board, a.baud, {"params": params, "uv_scale": rep.uv_scale, "model": a.sim_model,
                                         "session_trials": plan.trials, "offline": offline})
    br = Bridge(board, rep, plan, None if a.publish == "none" else parse_hostport(a.publish), a.log,
                offline, expected, a.quiet)
    th = threading.Thread(target=br.reader, daemon=True)
    th.start()
    print(f"[bridge] A0{a.subject} session {a.session}: {len(plan.trials)} trials {plan.trials[0]}..{plan.trials[-1]}, "
          f"{plan.n} samples ({plan.n / FS:.1f} s), lead-in {a.lead_in:g} s, board {a.board}"
          f"{' (fast)' if a.fast else ''}, publish {a.publish}", flush=True)
    try:
        while True:
            with br.lock:
                br.results.clear()
            dt = br.stream(realtime=not a.fast, free_run=a.free_run)
            br.wait_results(len(plan.trials), timeout=3.0 if a.board != "sim" else 1.0)
            s = br.summary()
            c, mo, me = s["correct"], s["match_offline"], s["match_expected"]
            print(f"[bridge] done in {dt:.1f} s: {s['n']}/{len(plan.trials)} inferences, accuracy {c[0]}/{c[1]}, "
                  f"match offline {mo[0]}/{mo[1]}, match expected {me[0]}/{me[1]}, gated {s['gated']}", flush=True)
            if not a.loop:
                break
    except KeyboardInterrupt:
        print("[bridge] interrupted", flush=True)
        s = br.summary()
    finally:
        br.stop.set()
        th.join(timeout=1.0)
        board.close()
        if br.log:
            br.log.close()
    return s


def main(argv: list[str] | None = None) -> int:
    s = run(argv)
    return 0 if s["n"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
