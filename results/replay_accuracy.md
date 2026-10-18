# Replay accuracy (host side)

Generated 2026-10-17 by `host/replay_accuracy.py`. Model `tinyeegnet_int8.pte` (v0.5-distill, wide32_ft seed 0), one `.pte` for all subjects, EA matrix per subject. BCI IV 2a session 2, test trials 25–144 (0-based 24–143, 120 per subject), streamed as the bridge streams them: int16 µV × 20.48, 250 Hz, 10 s lead-in, continuous.

| Column | Meaning |
|---|---|
| Offline int8 | Reproduce_Baselines int8 model on the float-stream windows (`int8_predictions.npz`) |
| Sim replay | `bridge.py --board sim --fast`: real framing + streaming pipeline, logits looked up from the offline predictions (checks the bridge/UI/measure path; equals Offline by construction) |
| Board target | int8 model (host rebuild, bit-exact on `test_vectors.npz`) on the `pipeline_ref` windows of the int16 stream = what the board must output (`bridge.py --sim-model int8`) |
| Agree | Board-target argmax == offline argmax, of 120 |
| Window diff | int8 input elements that differ between int16-stream and float-stream windows (max abs diff in steps) |

| Subject | Offline int8 % | Sim replay % | Board target % | Agree | Window diff % (max) |
|---|---:|---:|---:|---:|---:|
| A01 | 87.5 | 87.5 (120) | 87.5 | 120 | 0.0 (0) |
| A02 | 62.5 | 62.5 (120) | 62.5 | 120 | 0.0 (0) |
| A03 | 99.2 | 99.2 (120) | 99.2 | 120 | 0.0 (0) |
| A04 | 74.2 | 74.2 (120) | 74.2 | 120 | 0.0 (0) |
| A05 | 70.8 | 70.8 (120) | 70.8 | 120 | 0.0 (0) |
| A06 | 56.7 | 56.7 (120) | 56.7 | 120 | 0.0 (0) |
| A07 | 65.8 | 65.8 (120) | 65.8 | 120 | 0.0 (0) |
| A08 | 94.2 | 94.2 (120) | 94.2 | 120 | 0.0 (0) |
| A09 | 59.2 | 59.2 (120) | 59.2 | 120 | 0.0 (0) |
| **Mean** | **74.4** | **74.4** | **74.4** | 1080/1080 | 0.0 |

Board-target confusion matrix, all subjects summed (rows true, columns predicted):

| | pred left | pred right |
|---|---:|---:|
| true left | 431 | 109 |
| true right | 167 | 373 |

Notes:

- The int16 stream reproduces the float-stream windows exactly (0 elements differ); on the float stream the reference reproduces `test_vectors.npz` bit for bit (`host/tests/test_pipeline_ref.py`), with or without a 10 s lead-in.
- UART encoding µV × 20.48: BCI IV 2a is stored in steps of 100/2048 µV, so 20.48 counts per µV carries the recording losslessly (int16 range ±1600 µV, 0 samples clipped); any int8 window difference left here comes from float rounding only. The earlier µV × 10 encoding flipped 28.6 % of window elements (max 4 steps) and 13/1080 argmaxes, because the EA whitening amplifies the 0.05 µV rounding error.
- Board results go to `results/board_metrics.md` (`host/measure.py`).
