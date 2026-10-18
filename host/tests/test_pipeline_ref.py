"""pipeline_ref against Reproduce_Baselines' test vectors and the generated firmware headers."""

import re

import numpy as np
import pytest

from common import FIRST_TEST_TRIAL, REPO_DIR, RB_TEST_VECTORS, load_params, replay_path
from drivers.replay import load_replay, plan_session
from pipeline_ref import PipelineRef, StreamingPipeline

GEN = REPO_DIR / "firmware" / "tests" / "generated"
TV_TRIALS = slice(FIRST_TEST_TRIAL, FIRST_TEST_TRIAL + 20)   # test_vectors.npz = trials 24..43


@pytest.fixture(scope="module")
def params():
    return load_params()


@pytest.fixture(scope="module")
def tv():
    if not RB_TEST_VECTORS.exists():
        pytest.skip("Reproduce_Baselines/artifacts/v4/test_vectors.npz not present")
    return np.load(RB_TEST_VECTORS)


@pytest.fixture(scope="module")
def raw_a03():
    pytest.importorskip("mne")
    pytest.importorskip("moabb")
    from export_replay import load_session_uv

    try:
        return load_session_uv(3)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"raw BCI IV 2a not available: {exc}")


@pytest.fixture(scope="module")
def a03():
    if not replay_path(3).exists():
        pytest.skip("host/data/A03_session2.npz missing (python host/export_replay.py --subject 3)")
    return load_replay(3)


def test_a_float_stream_reproduces_test_vectors_exactly(params, tv, raw_a03):
    x_uv, cue, label, _ = raw_a03
    w = PipelineRef(params).windows(x_uv, cue[TV_TRIALS], 3)
    assert np.array_equal(label[TV_TRIALS], tv["y"])
    assert int(np.count_nonzero(w != tv["x"])) == 0


def test_a2_float_stream_with_lead_in_reproduces_test_vectors(params, tv, raw_a03):
    x_uv, cue, _, _ = raw_a03
    start = cue[FIRST_TEST_TRIAL] - 2500
    start -= start % 2
    w = PipelineRef(params).windows(x_uv[:, start:], cue[TV_TRIALS] - start, 3)
    assert int(np.count_nonzero(w != tv["x"])) == 0


def test_b_int16_stream_vs_test_vectors(params, tv, a03, capsys):
    ref = PipelineRef(params, a03.uv_scale)
    plan = plan_session(a03, list(range(24, 44)), 10.0)
    w = ref.windows(ref.to_uv(a03.x_int16[:, plan.start:plan.stop]), plan.cue_rel, 3)
    d = np.abs(w.astype(int) - tv["x"].astype(int))
    frac = float((d > 0).mean())
    with capsys.disabled():
        print(f"\n[int16 stream x{a03.uv_scale:g}, 10 s lead-in] differing int8 elements vs test_vectors: "
              f"{int((d > 0).sum())}/{d.size} ({100 * frac:.2f} %), max |diff| {int(d.max())}")
    # at 20.48 counts/uV the stream is lossless (2a native step 100/2048 uV)
    assert abs(a03.uv_scale - 20.48) < 1e-9
    assert int(d.max()) == 0  # bit-exact with test_vectors.npz


def test_streaming_equals_batch(params, a03):
    ref = PipelineRef(params, a03.uv_scale)
    plan = plan_session(a03, [33, 34, 35], 10.0)
    seg = a03.x_int16[:, plan.start:plan.stop]
    batch = ref.windows(ref.to_uv(seg), plan.cue_rel, 3)
    sp = StreamingPipeline(ref)
    sp.set_subject(3)
    rng = np.random.default_rng(0)
    out, k, cues = [], 0, set(plan.cue_rel)
    while k < seg.shape[1]:
        if k in cues:
            sp.cue(plan.labels[plan.cue_rel.index(k)])
        n = int(rng.integers(1, 6))
        nxt = [c for c in plan.cue_rel if k < c < k + n]
        end = min(seg.shape[1], nxt[0] if nxt else k + n)
        out += sp.push(seg[:, k:end].T)
        k = end
    assert [o.trial for o in out] == [0, 1, 2]
    assert [o.n250_last for o in out] == [ref.last_sample_250(c) for c in plan.cue_rel]
    for o in out:
        assert np.array_equal(o.x_int8, batch[o.trial])


def _c_array(text: str, name: str) -> np.ndarray:
    m = re.search(rf"\b{name}\[[^=]*=\s*\{{(.*?)\}};", text, re.S)
    assert m, name
    return np.array([int(v) for v in re.findall(r"-?\d+", m.group(1))])


def test_generated_headers_match_reference(params):
    pv = GEN / "preprocess_vectors.h"
    if not pv.exists():
        pytest.skip("run host/make_vectors.py first")
    text = pv.read_text()
    n = int(re.search(r"#define PV_N (\d+)u", text).group(1))
    subject = int(re.search(r"#define PV_SUBJECT (\d+)u", text).group(1))
    scale = float(re.search(r"#define PV_COUNTS_PER_UV ([\d.]+)f", text).group(1))
    stream = _c_array(text, "STREAM").reshape(n, 22).T.astype(np.int16)
    cue = _c_array(text, "CUE_N250")
    expected = _c_array(text, "EXPECTED").reshape(len(cue), 1, 22, 256)
    last = _c_array(text, "WINDOW_LAST_N250")
    ref = PipelineRef(params, scale)
    assert np.array_equal(ref.windows(ref.to_uv(stream), cue, subject), expected)
    assert last.tolist() == [ref.last_sample_250(int(c)) for c in cue]
    assert last[-1] == n - 1
    st = (GEN / "selftest_vectors.h").read_text()
    assert '#include "preprocess_vectors.h"' in st
    assert len(_c_array(st, "EXPECTED_ARGMAX")) == len(cue)
