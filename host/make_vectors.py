"""Generate the firmware test-vector headers and the committed offline-prediction reference.

    python host/make_vectors.py [--subject 3] [--trials auto|33-37] [--lead-in 10]

Writes
* ``firmware/tests/generated/preprocess_vectors.h``: an int16 replay stream (the exact UART
  payload) of one subject, 5 cues, and the int8 windows ``pipeline_ref`` computes on exactly
  that stream. The board's preprocessing must reproduce ``EXPECTED`` element for element.
* ``firmware/tests/generated/selftest_vectors.h``: includes the above and adds
  ``EXPECTED_ARGMAX`` (offline int8 model on the float stream) and ``OFFLINE_LOGITS``.
* ``host/reference/offline_int8_predictions.json``: offline int8 logits of all 9 subjects,
  test trials 24..143, from ``Reproduce_Baselines/runs/v4soft/student_wide32_ft_seed0/int8_predictions.npz``
  (``--predictions``).

Trial choice (``--trials auto``): the 5 consecutive test trials with the shortest stream
whose offline logit margin |l0 - l1| > 0.5 for all five, preferring trials inside
``test_vectors.npz`` (24..43), so the board's argmax is robust to the int16 quantization.
The stream starts ``lead-in`` seconds before the first cue, moved back to an even recording
sample so the board's decimation phase equals the training pipeline's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIRST_TEST_TRIAL, FS, MODEL_DESC, MODEL_TAG, OFFLINE_PRED_JSON, UV_SCALE,  # noqa: E402
                    RB_DIR, RB_INT8_PREDICTIONS, REPO_DIR, load_offline_predictions, load_params, parse_trials, replay_path)
from pipeline_ref import PipelineRef, emulate_float32  # noqa: E402

GEN_DIR = REPO_DIR / "firmware" / "tests" / "generated"
MARGIN = 0.5
N_VEC = 5


def stream_start(cue0: int, lead_in_s: float) -> int:
    s = int(cue0) - int(round(lead_in_s * FS))
    return max(0, s - (s % 2))


def choose_trials(cue: np.ndarray, logits: np.ndarray, lead_in_s: float, ref: PipelineRef) -> list[int]:
    margin = np.abs(logits[:, 0] - logits[:, 1])
    best = None
    for first in range(FIRST_TEST_TRIAL, FIRST_TEST_TRIAL + len(logits) - N_VEC + 1):
        trials = list(range(first, first + N_VEC))
        if not all(margin[t - FIRST_TEST_TRIAL] > MARGIN for t in trials):
            continue
        start = stream_start(cue[first], lead_in_s)
        n = ref.last_sample_250(int(cue[trials[-1]] - start)) + 1
        key = (trials[-1] >= 44, n)  # prefer trials covered by test_vectors.npz
        if best is None or key < best[0]:
            best = (key, trials)
    if best is None:
        raise RuntimeError("no 5 consecutive trials with a robust margin")
    return best[1]


def c_array(values: np.ndarray, per_line: int) -> str:
    flat = [str(int(v)) for v in np.asarray(values).ravel()]
    return "\n".join("  " + ",".join(flat[i:i + per_line]) + "," for i in range(0, len(flat), per_line))


def c_rows(rows: np.ndarray, per_line: int) -> str:
    """2-D initializer with one brace pair per row (no -Wmissing-braces)."""
    rows = np.asarray(rows)
    if rows.shape[1] <= per_line:
        return "\n".join("  {" + ",".join(str(int(v)) for v in r) + "}," for r in rows)
    return "\n".join("  {\n" + c_array(r, per_line) + "\n  }," for r in rows)


def write_offline_json(path: Path, src: Path = RB_INT8_PREDICTIONS) -> dict:
    preds = load_offline_predictions(src)
    try:
        src_txt = "Reproduce_Baselines/" + Path(src).resolve().relative_to(RB_DIR.resolve()).as_posix()
    except ValueError:
        src_txt = Path(src).name
    doc = {
        "source": src_txt,
        "model": f"tinyeegnet_int8.pte {MODEL_TAG} ({MODEL_DESC}), int8 fake-quant on the float stream",
        "trials": "session 2, 0-based 24..143 (1-based 25..144)",
        "subjects": {str(s): {"trial": v["trial"].tolist(), "y_true": v["y_true"].tolist(),
                              "logits": [[float(a), float(b)] for a, b in v["logits"]]}
                     for s, v in sorted(preds.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return preds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", type=int, default=3)
    ap.add_argument("--trials", default="auto")
    ap.add_argument("--lead-in", type=float, default=10.0)
    ap.add_argument("--params", default=None)
    ap.add_argument("--out-dir", default=str(GEN_DIR))
    ap.add_argument("--predictions", type=Path, default=RB_INT8_PREDICTIONS,
                    help="Reproduce_Baselines int8_predictions.npz of the shipped model")
    a = ap.parse_args(argv)

    params = load_params(a.params)
    ref = PipelineRef(params)
    preds = write_offline_json(OFFLINE_PRED_JSON, a.predictions) if a.predictions.exists() else load_offline_predictions()
    z = np.load(replay_path(a.subject))
    x16, cue, label = z["x_int16"], z["cue_sample"], z["label"]
    uv_scale = float(z["uv_scale"]) if "uv_scale" in z.files else UV_SCALE
    ref = PipelineRef(params, uv_scale)
    pr = preds[a.subject]
    if not np.array_equal(pr["y_true"], label[FIRST_TEST_TRIAL:]):
        raise AssertionError("offline predictions labels != replay labels")

    trials = choose_trials(cue, pr["logits"], a.lead_in, ref) if a.trials == "auto" else parse_trials(a.trials)
    if len(trials) != N_VEC:
        raise ValueError(f"need {N_VEC} trials, got {trials}")
    start = stream_start(cue[trials[0]], a.lead_in)
    cue_rel = (cue[trials] - start).astype(np.int64)
    n = ref.last_sample_250(int(cue_rel[-1])) + 1
    stream = x16[:, start:start + n]
    expected = ref.windows(ref.to_uv(stream), cue_rel, a.subject)          # (5, 1, 22, 256)
    emu = emulate_float32(stream, cue_rel, a.subject, params, uv_scale)
    emu_diff = int(np.count_nonzero(emu != expected))
    emu_max = int(np.abs(emu.astype(int) - expected.astype(int)).max())
    lg = pr["logits"][np.asarray(trials) - FIRST_TEST_TRIAL]
    argmax = lg.argmax(1).astype(np.int8)
    last = [ref.last_sample_250(int(c)) for c in cue_rel]
    psha = hashlib.sha256(Path(params["_path"]).read_bytes()).hexdigest()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_txt = ", ".join(str(t) for t in trials)
    hdr = f"""/* GENERATED by host/make_vectors.py -- do not edit.
 *
 * Reference stream for the board preprocessing (Reproduce_Baselines/docs/v2_handoff.md steps 1-8).
 * Subject A0{a.subject}, session 2, recording samples [{start}, {start + n}) = {n} samples
 * ({n / FS:.2f} s at 250 Hz), {a.lead_in:g} s lead-in (start moved back to an even recording sample).
 * Trials (0-based session-2 L/R trial index): {trials_txt}  (1-based: {", ".join(str(t + 1) for t in trials)}).
 * params.json sha256 {psha}
 *
 * STREAM[i][c]: int16 microvolts x PV_COUNTS_PER_UV ({uv_scale:g}), channel order = params.json channel_order; STREAM[0] is the
 *   first data frame after RESET (n_250 = 0). This is exactly what bridge.py sends.
 * CUE_N250[k]: index into STREAM of trial k's cue sample (send CUE just before that frame).
 * WINDOW_LAST_N250[k]: index of the 250 Hz sample that completes trial k's window
 *   (125 Hz window [CUE_N250/2 + 62, CUE_N250/2 + 318) -> last 250 Hz index 2 * (CUE_N250/2 + 317)).
 * EXPECTED[k]: int8 model input (22 x 256, row-major channel x time, i.e. NCHW (1,1,22,256))
 *   from host/pipeline_ref.py on exactly this stream: CAR + filter in float64, window cast to float32,
 *   EA in float64 cast to float32, scale/quantize in float32 (the precision plan of preprocess.cpp).
 *   A fully float32 variant (host/pipeline_ref.py emulate_float32, not what the firmware does)
 *   differs in {emu_diff} of {expected.size} elements, max |diff| {emu_max}; PV_TOLERANCE covers such ties.
 */
#ifndef EDGE_PREPROCESS_VECTORS_H
#define EDGE_PREPROCESS_VECTORS_H
#include <stdint.h>

#define PV_N {n}u
#define PV_N_CH 22u
#define PV_WIN 256u
#define PV_N_TRIALS {N_VEC}u
#define PV_SUBJECT {a.subject}u
#define PV_COUNTS_PER_UV {uv_scale!r}f  /* int16 count = round(uV * PV_COUNTS_PER_UV) */
#define PV_UV_PER_COUNT {1.0 / uv_scale!r}f  /* uV = count * PV_UV_PER_COUNT (exact in float32 for 20.48) */
#define PV_TOLERANCE 1   /* max |board - EXPECTED| per element accepted for float32 rounding ties */
#define PV_F32_EMU_DIFF {emu_diff}u

#ifdef __cplusplus
static const uint32_t N = PV_N;
static const uint8_t SUBJECT = PV_SUBJECT;
#endif

static const uint16_t TRIAL_INDEX[PV_N_TRIALS] = {{{trials_txt}}};
static const uint32_t CUE_N250[PV_N_TRIALS] = {{{", ".join(str(int(c)) for c in cue_rel)}}};
static const uint32_t WINDOW_LAST_N250[PV_N_TRIALS] = {{{", ".join(str(int(v)) for v in last)}}};
static const uint8_t LABEL[PV_N_TRIALS] = {{{", ".join(str(int(v)) for v in label[trials])}}};

static const int8_t EXPECTED[PV_N_TRIALS][PV_N_CH * PV_WIN] = {{
{c_rows(expected.reshape(N_VEC, -1), 32)}
}};

static const int16_t STREAM[PV_N][PV_N_CH] = {{
{c_rows(stream.T, 22)}
}};

#endif /* EDGE_PREPROCESS_VECTORS_H */
"""
    (out_dir / "preprocess_vectors.h").write_text(hdr)

    margins = ", ".join(f"{abs(l0 - l1):.2f}" for l0, l1 in lg)
    board_txt = ""
    try:  # exact expectation for the board: the int8 model on EXPECTED (needs torch + executorch)
        from int8_oracle import Int8Oracle

        bl = Int8Oracle(params=params).logits(expected)
        board_txt = (
            "\n/* BOARD_ARGMAX / BOARD_LOGITS: the same int8 model (host fake-quant rebuild, bit-exact on\n"
            " * test_vectors.npz) run on EXPECTED, i.e. what the .pte must output for these windows. */\n"
            f"static const int8_t BOARD_ARGMAX[PV_N_TRIALS] = {{{', '.join(str(int(v)) for v in bl.argmax(1))}}};\n"
            "static const float BOARD_LOGITS[PV_N_TRIALS][2] = {"
            + ", ".join(f"{{{l0:.6f}f, {l1:.6f}f}}" for l0, l1 in bl) + "};\n")
        print(f"board (oracle) argmax {bl.argmax(1).tolist()} logits {np.round(bl, 3).tolist()}")
    except Exception as exc:  # noqa: BLE001
        print(f"int8 oracle unavailable ({exc.__class__.__name__}: {exc}); BOARD_* not written")
    st = f"""/* GENERATED by host/make_vectors.py -- do not edit.
 *
 * Self-test vectors: preprocess_vectors.h plus the offline int8 model's prediction for each trial.
 * EXPECTED_ARGMAX / OFFLINE_LOGITS: tinyeegnet_int8 ({MODEL_TAG}) on the float-stream windows
 *   (Reproduce_Baselines int8_predictions.npz, logits_{a.subject}), trials {trials_txt}.
 *   Chosen so every |l0 - l1| > {MARGIN} (margins: {margins}), so the argmax survives the int16 stream.
 * PASS = EXPECTED window match (within PV_TOLERANCE) and argmax == EXPECTED_ARGMAX for all trials.
 */
#ifndef EDGE_SELFTEST_VECTORS_H
#define EDGE_SELFTEST_VECTORS_H
#include "preprocess_vectors.h"

static const int8_t EXPECTED_ARGMAX[PV_N_TRIALS] = {{{", ".join(str(int(v)) for v in argmax)}}};
static const float OFFLINE_LOGITS[PV_N_TRIALS][2] = {{{", ".join(f"{{{l0:.6f}f, {l1:.6f}f}}" for l0, l1 in lg)}}};
{board_txt}
#endif /* EDGE_SELFTEST_VECTORS_H */
"""
    (out_dir / "selftest_vectors.h").write_text(st)
    print(f"A0{a.subject} trials {trials} start {start} N {n} cues {cue_rel.tolist()} labels {label[trials].tolist()} "
          f"argmax {argmax.tolist()} margins [{margins}] f32-emu diff {emu_diff} (max {emu_max})")
    for f in ("preprocess_vectors.h", "selftest_vectors.h"):
        p = out_dir / f
        print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
