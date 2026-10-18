# Results summary

Numbers shown in the README before the dashboard. Sources: [`ml/README.md`](../ml/README.md), [`fvp_selftest.md`](fvp_selftest.md), [`replay_accuracy.md`](replay_accuracy.md), [`docs/architecture.md`](../docs/architecture.md).

| Metric | Value | Where measured |
|---|---|---|
| Accuracy, generic model (9 BCI IV 2a subjects, 1,080 unseen trials) | 74.6 % int8 (mean of 3 seeds); shipped seed 74.4 %, best subject 99.2 % | laptop, [`ml/README.md`](../ml/README.md) |
| Accuracy, subject-specific model (subject 1, session 2) | 79.9 % float; 79.2 % int8 on the board | laptop; physical E8 |
| Accuracy, generic model, physical E8 | 77.1 % (54/70 trials), [`board_metrics.md`](board_metrics.md) | physical E8 |
| Board output = FVP, subject-specific model | 160/160 windows | physical E8 |
| End-to-end latency (last sample in → prediction out) | not measured on the board; FVP estimate under 1 ms | FVP |
| Inference on the Ethos-U85 | 128,056 NPU cycles ≈ 0.32 ms at 400 MHz | FVP |
| Preprocessing on the M55 | 243 cycles per sample (0.6 µs); 99.7 k cycles per window (0.25 ms) | FVP |
| Model size | 11,456 B `.pte`, 3,426 parameters | |
| Memory | NPU pools peak 124.5 KiB (SRAM0); code + `.pte` + tables 227 KiB MRAM; app data 110 KiB DTCM | FVP, linker map |
| Preprocessing correctness | 0 of 28,160 int8 values differ from the laptop reference | FVP self-test |

**Optimisation** (same pipeline, first release vs shipped):

| | v0.1 baseline | v0.5 shipped |
|---|---|---|
| Input | 22 × 500 at 250 Hz | 22 × 256 at 125 Hz |
| Accuracy (int8, 9 subjects) | 71.5 % | 74.6 % |
| NPU cycles per inference | 226,056 | 128,056 |
| NPU memory pools | 237 KiB | 124.5 KiB |
| `.pte` | 10,624 B | 11,456 B |
