# DCSM-Simplified: results

PC numbers come from `dcops/jobs/` on a Colab GPU (2026-10-18). Board numbers come from the DevKit-E8, M55 HP @ 400 MHz.

## Teachers

| | crops | teacher | accuracy |
|---|---|---|---|
| wave 1 | 3,494 (YOLO pseudo-labels) | ConvNeXt-Small | 69.4 % (val, auto labels) |
| wave 2 | 19,203 (2,056 human-labelled images + wave-1 crops for missing classes) | 2x ConvNeXt-Base, out-of-fold | **97.2 %** human-labelled val; 94.0 % out-of-fold train |

## Vision students (MobileNetV2, 160 px unless named, all ops on the Ethos-U85)

| run | metric | val | board-camera sim | int8 = float | .pte KB |
|---|---|---|---|---|---|
| r128_kd | val (auto labels) | 0.581 | 0.560 | 1.0 | 2302 |
| r160_kd | val (auto labels) | 0.653 | 0.593 | 0.98 | 2303 |
| r160_kd_e120 | val (auto labels) | 0.644 | 0.636 | 0.95 | 2306 |
| r160_noaug | val (auto labels) | 0.584 | 0.498 | 0.95 | 2301 |
| r160_nokd | val (auto labels) | 0.667 | 0.536 | 0.95 | 2304 |
| r192_kd | val (auto labels) | 0.632 | 0.531 | 0.97 | 2301 |
| v2_r160_kd | human-labelled val | 0.912 | 0.913 | 1.0 | 2302 |
| v2_r160_kd_mix | human-labelled val | 0.914 | 0.814 | 0.98 | 2301 |
| v2_r160_kd_mix_e80 | human-labelled val | 0.927 | 0.839 | 1.0 | 2304 |
| v2_r160_noaug | human-labelled val | 0.923 | 0.758 | 1.0 | 2295 |
| v2_r160_nokd | human-labelled val | 0.912 | 0.804 | 1.0 | 2304 |

Wave 1 (rows without `v2_`) is scored on noisy auto labels; wave 2 on the human-labelled split, so the two are not directly comparable.
Ablations (wave 2): board-camera augmentation 0.913 vs 0.758 on board-like images (`v2_r160_kd` vs `v2_r160_noaug`); distillation 0.913 vs 0.804 (`v2_r160_kd` vs `v2_r160_nokd`). Board-sim scores are a single random draw per evaluation, so differences of a few points are noise.
Resolution sweep (wave 1): 128 px 0.581, 160 px 0.653, 192 px 0.632. We kept 160 px.

## Board-camera fine-tune (`v3_board_ft`, then `v4_demo_ft`, deployed)

Data: the E8 camera's own model-input frames, recorded over J-Link while a phone showed a clock-synced slideshow (labels from the clock, no network link).

| capture | content |
|---|---|
| deck 1, x2 | 96 training-split slides, 8 classes |
| deck 2 | 150 new photos, 5 demo classes, varied zoom |
| background | ~190 room/ceiling frames |
| total | 1,330 board frames: 1,119 train / 211 test, split by slide (test slides never trained on) |
| replay | 1,503 crops of human-labelled photos, to prevent forgetting |

Training: 25 epochs, lr 3e-4, 60 % board frames per batch.

| run | held-out board frames, all classes |
|---|---|
| ceiling capture (discarded: camera framed wrong) | 0 % -> 2.4 % |
| first good deck-1 run | 30.7 % -> 64.6 % |
| **v3_board_ft (deck 1 x2 + deck 2 + background)** | **35.5 % -> 73.0 %** |
| **v4_demo_ft** (same data, loss and scoring over the 6 shown classes) | 6-way top-1 on held-out demo-class frames: 51.3 % -> **84.8 %** (v3: 83.5 %) |

### Demo mask (5 classes + background, held-out board frames)

`CLS_ALLOWED_MASK=0x13250`: power shelf, network port, fan, power connector, drive bay, background.

| cutoff | component frames correct | wrong | blank | background frames given a label |
|---|---|---|---|---|
| 0.5 (`v3_board_ft`) | 73 % | 6 % | rest | 0 % |
| **0.6 (`v4_demo_ft`, demo build)** | **79 %** | **7 %** | rest | **0 %** |
| 0.35 | 78 % | 11 % | rest | |

Clean held-out photos (dataset val split, 5 classes, PC): 25/28 correct.
Why 5: through the board camera, compute tray, cooling manifold and DPU were confused with each other, and 8 classes had no board-camera data.

## Keyword spotters (DS-CNN, MLEK MFCC 49x10, 12 classes)

| run | test | test with board auto-gain | params | int8 = float | .pte KB |
|---|---|---|---|---|---|
| dscnn_m_agc | 0.9466 | 0.9470 | 79116 | 0.88 | 93 |
| dscnn_m_kd | 0.9444 | - | 79116 | 0.89 | 93 |
| dscnn_s_agc | 0.9394 | 0.9409 | 23180 | 0.89 | 34 |
| dscnn_s_kd | 0.9366 | - | 23180 | 0.88 | 33 |

**Deployed:** `dscnn_m_agc` (DS-CNN 128 ch x 4 blocks, trained with the board's auto-gain emulated).

## Export

| model | .pte | CPU ops | int8 vs float |
|---|---|---|---|
| vision `v3_board_ft` | 2301 KB | quantize / dequantize only | 98.4 % top-1 agreement (calibration frames, calibrated on board frames) |
| KWS `dscnn_m_agc` | 93 KB | quantize / dequantize only | |

## On the board (build-fusion: camera classifier + voice, one HP image)

Measured with the earlier `v2` vision model (same architecture as deployed).

| metric | value |
|---|---|
| vision NPU inference | 7.9–8.0 ms |
| vision preprocessing (M55) | 1.75 ms |
| full frame | 255–287 ms (3.5–3.9 FPS); LCD footer 274 ms / 3.6 FPS |
| NPU share of frame | ~3 % |
| KWS MFCC (M55) | 3.3–4.0 ms |
| KWS DS-CNN (NPU) | 0.34 ms, 2 strides/s |
| After removing per-frame UART logging (judged build) | **124 ms / 8.0 FPS** (was 274 ms / 3.6 FPS, 2.2x); NPU 7.9 ms; KWS MFCC 3.3 ms + NPU 0.3 ms |
| power | `TODO(measure)`: not measured; needs a USB power meter or the power use case |
| MRAM image end | 0x805473C4 (limit 0x80578000) -> ~195 KB headroom |
| KWS activation buffer | 0x20000 (128 KB) |

Correctness: on the same captured frame, the board's top-3 matched the PC's top-3 (e.g. background 66.4 % vs 66.4 %).
