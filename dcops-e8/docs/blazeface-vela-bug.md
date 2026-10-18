# BlazeFace scrambled outputs on the Ethos-U85: root cause and fix

Found 2026-10-17/18 with a Corstone-320 FVP harness (experiments/fvp, outside this repo).

## Summary
- **Symptom (board):** BlazeFace runs fully on the Ethos-U85 but its outputs are scrambled: board
  scores[0:16] equal the PC simulation's scores[0,2,4,...,30], and scores[16:32] repeat them.
- **Root cause: a Vela graph-optimiser miscompile** (Vela 4.5.0, still present in 5.1.0).
  ExecuTorch 1.1.0 lowers each head `conv -> permute(0,2,3,1) -> reshape` to TOSA
  `CONV2D -> RESCALE -> TRANSPOSE[0,2,3,1] -> TRANSPOSE[0,3,1,2] -> RESHAPE`. The two transposes cancel;
  Vela removes the pair and then gives the convolution the *reshape's* output shape
  (e.g. `regressor_8`: OFM [1,1,512,16] instead of [1,16,16,32]), so the NPU computes each head over a
  one-pixel strip.
- **Not at fault:** the TOSA itself (TOSA reference model on the exact flatbuffer Vela compiled is
  bit-exact with the int8 simulation), ExecuTorch's EthosUBackend output copy, output order, dim_order.
- **Why classifiers work:** `AvgPool -> reshape` has a 1x1 spatial size, so no transposes are emitted.
- **ExecuTorch 1.5.1** emits `CONV2D -> RESHAPE` with no transposes; even Vela 4.5 compiles that correctly.

Rule for ExecuTorch 1.1 + Vela 4.5/5.1: never delegate `spatial op -> permute(0,2,3,1) -> reshape/flatten`.

## Fix applied (patch A)
`resources/object_detection/blazeface/export.py` keeps the heads' reshape/cat/squeeze off the NPU (on
the Cortex-M55). The program's outputs are unchanged (scores 1x896, boxes 1x896x16), so the firmware
needs no C++ change, only a rebuild. On the FVP the patched program is bit-exact with the TOSA
reference. `blazeface_front_128_ethos-u85-256.pte` is the patched export.

Alternative (not applied): export the four heads as NHWC 4D tensors and concatenate them in
`BlazeFaceProcessing` (single NPU delegate, ~5-line handler change).

## Evidence (FVP vs TOSA reference)
| Model | Tail | Result |
|---|---|---|
| MobileNetV2 (AvgPool -> reshape -> Linear) | classifier | bit-exact, 1000/1000 |
| conv -> reshape | no cancelling transposes | bit-exact |
| conv -> permute -> reshape | cancelling pair | corr -0.05 |
| BlazeFace (board .pte) | same | scores corr 0.03 |
| BlazeFace, raw NCHW or NHWC 4D heads | no reshape in graph | bit-exact |
| BlazeFace, patch A (tail on CPU) | tail on CPU | bit-exact |

## Not verified
- On the E8 board (one run with the patched .pte should confirm: same signature on FVP and board).
- CPU cost of the tail on the M55 (the FVP is not cycle-accurate).
