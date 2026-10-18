# Firmware self-test on the Corstone-320 FVP

Date 2026-10-17, re-run for the v0.5-distill model (v0.3-distill, the previous model, has the same contract and passed the same test). Build `edge_eeg.SelfTest+SSE-320-U85` (Arm Compiler 6.24, CMSIS-Toolbox 2.15, ExecuTorch 1.5.1), model `tinyeegnet_int8.pte` v0.5-distill (wide32_ft seed 0, 11,456 B, sha256 `6caa8a98…`), params.json sha256 `3d34f632…`. Full console: [`fvp_selftest.log`](fvp_selftest.log).

The SelfTest build replays an embedded stream through the real firmware path: RESET, SET_SUBJECT 3, then 13,381 int16 data frames (A03 session 2, 10 s lead-in plus trials 33–37 0-based, 53.5 s at 250 Hz, µV × 20.48), with a CUE frame before each cue sample. All of it goes through the UART frame parser, the per-sample preprocessing (CAR, SOS band-pass, decimation, ring buffer), EA and quantization, the Ethos-U85 inference and the JSON formatter. No UART input is needed. Vectors come from `firmware/tests/generated/selftest_vectors.h` (`host/make_vectors.py`, reference `host/pipeline_ref.py`).

```
docker run --rm -i --init --user $(id -u) --mount "type=bind,src=$HOME/,dst=$HOME/" --workdir "$PWD" \
  --env "ARMLM_CACHED_LICENSES_LOCATION=$HOME/.armlm" cmsis-fvp:11.32.23 stdbuf -oL -eL \
  FVP_Corstone_SSE-320 -f board/Corstone-320/fvp_config.txt --simlimit 600 --stat \
  -a out/edge_eeg/SSE-320-U85/SelfTest/edge_eeg.axf          # from firmware/
```

## Result: `Test_result: PASS`

| Trial (0-based) | Label | int8 window vs reference | Board argmax | Expected (BOARD_ARGMAX) | Logits | Max logit err vs host model |
|---|---|---|---|---|---|---|
| 33 | 1 | 0 / 5632 differ | 1 | 1 | −0.669, 0.966 | 0.000 |
| 34 | 0 | 0 / 5632 | 0 | 0 | 1.437, −1.412 | 0.025 |
| 35 | 0 | 0 / 5632 | 0 | 0 | 1.981, −2.031 | 0.025 |
| 36 | 0 | 0 / 5632 | 0 | 0 | 1.139, −1.189 | 0.025 |
| 37 | 1 | 0 / 5632 | 1 | 1 | −1.709, 2.081 | 0.025 |

- PASS requires every trial's int8 window to equal the reference exactly and the argmax to equal `BOARD_ARGMAX`. The header's `PV_TOLERANCE` (±1) is printed for information only.
- The int8 windows match `host/pipeline_ref.py` bit for bit: 0 of 28,160 elements differ, max |diff| 0. The board filters in float64 like the reference. A float32 filter gives 6–9 elements off by one (measured on the host and in `pipeline_ref.emulate_float32`).
- Every window completes on the documented sample (`seq` = `WINDOW_LAST_N250`). There were 0 frame errors and 0 filled gaps.
- The logits differ from the host fake-quant model by at most one output quantization step (0.025). Emulator_Deploy v0.4-fvp shows the same step.

## Cycles (FVP)

The FVP is not cycle accurate, and its SSE-320 CPU is a Cortex-M85 with `SystemCoreClock` = 25 MHz. The `lat_us` and `t` values in this log therefore do not represent board timing. Cycle counts are what to carry over.

| Item | Value |
|---|---|
| NPU cycles per inference (Ethos-U PMU) | 128,056 |
| CPU cycles around `forward()` | ≈33,300 |
| Preprocessing per 250 Hz sample (CAR + 4 SOS, float64) | 243 cycles |
| Window: EA 22×22×256 (float64 accumulate) + scale + quantize | 99,720 cycles |
| ExecuTorch pools (peak) | method 28,812 B, temp 98,688 B |

At 400 MHz this comes to about 0.32 ms NPU, 0.25 ms window preparation and 0.6 µs per sample. The board has not measured it yet.

The FVP prints a benign `Warning: load application failed … [0x12200000..]` at start. Emulator_Deploy's runs show the same warning, and the image runs normally.
