"""End to end without hardware: bridge -> UART frames -> sim board -> JSON -> UDP -> measure."""

import json
import socket

import numpy as np
import pytest

import bridge
import measure
from common import replay_path
from listen import Listener

CONTRACT_KEYS = {"t", "seq", "trial", "subject", "cls", "raw_cls", "conf", "logits", "lat_us", "motion", "label"}


@pytest.fixture(scope="module")
def have_a03():
    if not replay_path(3).exists():
        pytest.skip("host/data/A03_session2.npz missing (python host/export_replay.py --subject 3)")


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_sim_five_trials_end_to_end(have_a03, tmp_path):
    port = _free_port()
    lst = Listener(port)
    log = tmp_path / "sim.jsonl"
    s = bridge.run(["--subject", "3", "--board", "sim", "--trials", "33-37", "--fast", "--quiet",
                    "--publish", f"127.0.0.1:{port}", "--log", str(log)])
    assert s["n"] == 5 and s["correct"] == (5, 5) and s["match_offline"] == (5, 5)
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    preds = [r for r in recs if r["event"] == "pred"]
    cues = [r for r in recs if r["event"] == "cue"]
    assert [r["session_trial"] for r in preds] == [33, 34, 35, 36, 37]
    assert [c["session_trial"] for c in cues] == [33, 34, 35, 36, 37]
    for p, c in zip(preds, cues):
        assert CONTRACT_KEYS <= set(p)
        assert p["label"] == c["label"] == p["true_label"]
        assert p["seq"] == (c["seq"] // 2 + 317) * 2  # last sample of the cue's window
        assert p["cls"] == p["raw_cls"] == int(np.argmax(p["logits"]))
        assert 0.5 <= p["conf"] <= 1.0 and p["motion"] == 0
    got = list(lst.poll(1.0))
    lst.close()
    assert sum(1 for r in got if r.get("event") == "pred") == 5
    out = tmp_path / "metrics.md"
    assert measure.main(["--jsonl", str(log), "--trials", "5", "--out", str(out)]) == 0
    assert "5/5 (100.0 %)" in out.read_text()


def test_sim_motion_gate_and_free_run(have_a03, tmp_path):
    from drivers.replay import iter_frames, load_replay, plan_session
    from drivers.sim_board import SimBoard

    rep = load_replay(3)
    plan = plan_session(rep, [33], 2.0)
    sb = SimBoard(uv_scale=rep.uv_scale, session_trials=plan.trials)
    sb.set_motion(True)
    for _, frame, _ in iter_frames(rep, plan, free_run=True, chunk=64):
        sb.write(frame)
    lines = []
    while sb.in_waiting:
        lines.append(sb.readline().decode().strip())
    preds = [json.loads(l) for l in lines if l.startswith("{")]
    cued = [p for p in preds if p["trial"] == 0]
    free = [p for p in preds if p["trial"] == -1]
    assert lines[0].startswith("READY n_ch=22")
    assert len(cued) == 1 and cued[0]["cls"] == -1 and cued[0]["motion"] == 1
    assert len(free) == (plan.n // 2 - 256) // 32 + 1
