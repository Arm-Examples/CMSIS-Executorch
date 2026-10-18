"""Live demo window: EEG trace, predicted hand, confidence, latency, motion gate, accuracy.

    python host/demo_ui.py --listen 5555            # matplotlib window
    python host/demo_ui.py --listen 5555 --headless # same information as text lines

Reads only the UDP records ``bridge.py`` publishes (never the serial port).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from listen import Listener, parse_listen  # noqa: E402

FS_FEED = 25.0          # bridge publishes every 10th 250 Hz sample
TRACE_S = 2.0
NAMES = {0: "LEFT", 1: "RIGHT", -1: "GATED"}


class State:
    def __init__(self):
        n = int(TRACE_S * FS_FEED)
        self.ch = ["C3", "Cz", "C4", "FCz"]
        self.trace = deque(maxlen=n)
        self.last: dict | None = None
        self.n = self.correct = self.gated = self.match_off = self.n_off = 0
        self.subject = None
        self.n_trials = None
        self.cue: dict | None = None
        self.log: deque = deque(maxlen=3)

    def update(self, rec: dict) -> str | None:
        ev = rec.get("event")
        if ev == "sample":
            self.ch = rec.get("ch", self.ch)
            self.trace.append(rec["uv"])
        elif ev == "session":
            self.subject, self.n_trials = rec.get("subject"), rec.get("n_trials")
            self.n = self.correct = self.gated = self.match_off = self.n_off = 0
            return f"session A0{self.subject}: {self.n_trials} trials"
        elif ev == "cue":
            self.cue = rec
            return f"cue trial {rec.get('session_trial')} (label {NAMES.get(rec.get('label'), '?')})"
        elif ev == "log":
            self.log.append(rec.get("line", ""))
            return f"board: {rec.get('line', '')}"
        elif ev == "pred":
            self.last = rec
            if rec.get("trial", -1) >= 0:
                self.n += 1
                lab = rec.get("true_label", rec.get("label"))
                if rec.get("cls") == -1:
                    self.gated += 1
                elif lab is not None and lab >= 0:
                    self.correct += int(rec.get("cls") == lab)
                if rec.get("match_offline") is not None:
                    self.n_off += 1
                    self.match_off += int(rec["match_offline"])
            return self.line(rec)
        return None

    def acc(self) -> str:
        scored = self.n - self.gated
        return f"{self.correct}/{scored} ({100 * self.correct / scored:.0f} %)" if scored else "–"

    def line(self, r: dict) -> str:
        return (f"trial {r.get('session_trial', r.get('trial'))}: {NAMES.get(r.get('cls'), '?'):5s} "
                f"conf {float(r.get('conf', 0)):.2f} label {NAMES.get(r.get('true_label', r.get('label', -2)), '?'):5s} "
                f"lat {r.get('lat_us')} us motion {r.get('motion')} | acc {self.acc()} gated {self.gated} "
                f"| n {self.n}{'' if self.n_trials is None else '/' + str(self.n_trials)}")


def headless(lst: Listener, seconds: float) -> int:
    st = State()
    end = time.monotonic() + seconds if seconds else None
    print("[demo_ui] headless, waiting for bridge records ...", flush=True)
    while end is None or time.monotonic() < end:
        for rec in lst.poll(0.2):
            msg = st.update(rec)
            if msg and rec.get("event") != "sample":
                print(msg, flush=True)
    return 0


def window(lst: Listener) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    st = State()
    fig = plt.figure("ArmExecuTorchEEG demo", figsize=(11, 6))
    gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 1, 1], height_ratios=[1, 1, 0.4])
    ax_eeg = fig.add_subplot(gs[0:2, 0])
    ax_cls = fig.add_subplot(gs[0, 1:])
    ax_conf = fig.add_subplot(gs[1, 1:])
    ax_txt = fig.add_subplot(gs[2, :])
    n = int(TRACE_S * FS_FEED)
    t = np.linspace(-TRACE_S, 0, n)
    offsets = np.arange(4)[::-1] * 40.0
    lines = [ax_eeg.plot(t, np.full(n, o), lw=1)[0] for o in offsets]
    ax_eeg.set_yticks(offsets, st.ch)
    ax_eeg.set_ylim(-40, 160)
    ax_eeg.set_xlim(-TRACE_S, 0)
    ax_eeg.set_xlabel("s (raw µV, 25 Hz preview; the board filters at 250 Hz)")
    ax_eeg.set_title("EEG replay (BCI IV 2a, session 2)")
    for ax in (ax_cls, ax_txt):
        ax.set_axis_off()
    big = ax_cls.text(0.5, 0.55, "–", ha="center", va="center", fontsize=54, weight="bold")
    sub = ax_cls.text(0.5, 0.1, "waiting for bridge", ha="center", va="center", fontsize=12)
    bar = ax_conf.barh([0], [0], color="0.5")[0]
    ax_conf.set_xlim(0, 1)
    ax_conf.set_yticks([])
    ax_conf.set_title("confidence")
    info = ax_txt.text(0.0, 0.5, "", ha="left", va="center", family="monospace", fontsize=11)
    colors = {0: "tab:green", 1: "tab:blue", -1: "tab:red"}

    def tick(_):
        for rec in lst.poll(0.0):
            st.update(rec)
        if st.trace:
            arr = np.asarray(st.trace, dtype=float)
            arr = arr - arr.mean(axis=0)
            for i, ln in enumerate(lines):
                y = np.full(n, np.nan)
                y[-len(arr):] = arr[:, i] + offsets[i]
                ln.set_ydata(y)
        r = st.last
        if r:
            c = r.get("cls")
            big.set_text(NAMES.get(c, "?"))
            big.set_color(colors.get(c, "black"))
            lab = r.get("true_label", r.get("label"))
            sub.set_text(f"trial {r.get('session_trial', r.get('trial'))}   true {NAMES.get(lab, '?')}   "
                         f"motion {'YES' if r.get('motion') else 'no'}")
            bar.set_width(float(r.get("conf", 0)))
            bar.set_color(colors.get(c, "0.5"))
            info.set_text(f"subject A0{st.subject}   latency {r.get('lat_us')} us   accuracy {st.acc()}   "
                          f"gated {st.gated}   trials {st.n}{'' if st.n_trials is None else '/' + str(st.n_trials)}   "
                          f"match offline {st.match_off}/{st.n_off}")
        return [*lines, big, sub, bar, info]

    _anim = FuncAnimation(fig, tick, interval=50, blit=False, cache_frame_data=False)  # noqa: F841
    fig.tight_layout()
    plt.show()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listen", default="5555", help="UDP port (or host:port) bridge.py publishes to")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=0, help="headless: stop after N s (0 = forever)")
    a = ap.parse_args(argv)
    host, port = parse_listen(a.listen)
    lst = Listener(port, host)
    try:
        if a.headless:
            return headless(lst, a.seconds)
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("[demo_ui] matplotlib missing, falling back to --headless", file=sys.stderr)
            return headless(lst, a.seconds)
        return window(lst)
    except KeyboardInterrupt:
        return 0
    finally:
        lst.close()


if __name__ == "__main__":
    raise SystemExit(main())
