# Isolated Moamen v4 streaming firmware

This variant embeds the unchanged unfused `wide32_ft` seed-0 program, SHA-256
`6caa8a9876d3cd0c36b39b3d4474a0fe7dbdbfb32a4d6f80ed4776623a19e02e`.
It does not alter `firmware/streaming`, the working Model 2 generated project,
the four-trial fallback, or the physical board.

The wire protocol retains the validated EEG2 v1 header, COBS framing, CRC32,
session/request/trial identities, cached duplicate response, and finite receive
timeouts. Each request contains exactly 5,632 signed INT8 values in C order.
The 32-bit trial field identifies an entry in the verified 1,080-row manifest;
it is not an arrival-order label or a request counter. Classes remain 0=left,
1=right. READY identifies this model by its hash, dimensions `(1,1,22,256)`,
wire dtype INT8, scale `0.06955837458372116`, zero point `-12` and build label
`eeg-multi-v4.0`. A Model 2 host must reject this READY profile.

Moamen's existing PTE takes **float32**, despite its INT8 internal execution.
`app_main.cpp` reconstructs each wire value as `(q - (-12)) * scale` before
calling the persistent ExecuTorch module. The PTE's existing CPU quantizer
recovers the original INT8 values. All 256 possible input values are checked
by the native regression test. No weights or saved quantization are changed.
The input float buffer adds 22,528 bytes. Both CPU quantize and dequantize
operators must be present in the AI layer.

`tools/multisubject_trials/prepare_firmware.py` creates a new ignored project at
`artifacts/multisubject-v4/firmware/project`. It verifies the bundle PTE and
preprocessing hashes and copies the proven `firmware/streaming/uart_stream.c`
byte-for-byte into the new board layer. It copies the working startup, memory,
clock and pin configuration without invoking any deployment command. The
initial active build context is **SSE-320-U85**, not physical hardware. The
DevKit-E8 target remains available for later compilation and separately
authorized deployment. The csolution MLOps profile documents the existing
precompiled model; never run an exporter over this AI layer.

Default fixtures are the 20 genuine A03 vectors already validated by Moamen.
After preparing all recorded inputs, generate FVP batches with:

```sh
python tools/multisubject_trials/prepare_firmware.py \
  --data artifacts/multisubject-v4/data --start 0 --count 240
```

Use starts 0, 240, 480, 720 and 960 (last count 120) for full coverage; use
`--indices 240,0,1079,240,0` to check order and deterministic repeats. This
240-case fixture limit only keeps the FVP build inside the existing 2 MiB code
region. The production streaming loop has no fixed request count or duration.
Generate the next fixture only after preserving the current run's log and
`artifacts/multisubject-v4/firmware/expected.json`. Build/run through CMSIS
Developer Assistant with the explicit SSE-320-U85 target.

FVP goes through the same parser, checksum checks, input adapter and inference
callback as production. It deliberately rejects two invalid frames before
valid inference. `STREAM_FRAME` records exact binary responses, while
`STREAM_RESULT` reports request/trial IDs, input CRC, 9-significant-digit logits
and their exact float32 bit patterns. `STREAM_TEST PASS` verifies counts and
rejections; external comparison must separately verify numerical outputs and
predictions against the saved references. These outputs are FVP results,
not physical board results. Production emits binary framed replies only.

Offline protocol regression (no board or serial access):

```sh
clang++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined \
  tests/multisubject/native_tests.cpp -o /tmp/edgebci-multisubject-native-test
/tmp/edgebci-multisubject-native-test
```

It covers integrity, lengths, stale sessions, duplicated/conflicting IDs,
truncated-frame timeout, garbage/oversize bounds, READY metadata, exact input
reconstruction and 1,200 sustained requests. Host simulation can launch the
same executable with any argument for binary stdin/stdout transport. Its toy
inference callback is test-only; the production/FVP app always invokes the
embedded PTE.
