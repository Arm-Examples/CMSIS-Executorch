# Model 2 streaming variant

Physically validated on `feature/model2-random-eeg-trials`: 160 real E8 requests
plus 21 Chrome-captured results match the saved FVP outputs exactly. See
`tools/model2_random_trials/PHYSICAL_STREAMING_VALIDATION.md`.
The four-trial fallback is unchanged on `demo/edgebci-model2-4trial` at
`c85ab70f6030243283431f3a12b1a7ea87aa8b72`.

- `protocol.hpp`: portable bounded COBS/CRC32 framing, explicit little-endian
  fields, session/request/trial matching, receive deadlines, input validation,
  duplicate-response cache and inference dispatch.
- `app_main.cpp`: unchanged embedded Model 2 and ExecuTorch forward contract;
  one aligned 11,000-byte INT8 input and one persistent module instance. FVP-only
  fixture branch feeds encoded bytes to the SAME production receiver/server.
- `uart_stream.c`: dedicated UART4 owner, 115200/8N1, nonblocking one-byte RX
  rearmed from the driver callback, bounded 1KiB ring, SysTick deadlines, bounded
  TX, UART fault invalidation/rearming. RX ISR never runs the model. Runtime
  diagnostics are retained in a bounded RAM buffer rather than corrupting frames.
- `fvp-capture.sh`: simulation-only output capture wrapper, invoked through CMSIS.

## Protocol contract

See `tools/model2_random_trials/PHASE2_DESIGN.md` for the header layout. Enum values:
HELLO1, READY2, INFER3, ACCEPTED4, RESULT5, ERROR6, STATUS7, LOG8 (reserved).
CRC32/ISO-HDLC over decoded header+payload, COBS encoding, trailing zero delimiter.
Protocol header32bytes + payload + CRC4bytes; maximum input11000bytes. Receiver
wire buffer11200bytes, inter-byte timeout1s, total frame timeout5s, drop-until-zero
resynchronization after timeout/overrun/malformed input. Invalid/untrusted headers
and CRCs are discarded rather than used to address a response. Host timeout then
stops the request; no malformed frame invokes inference.

HELLO has empty payload, request0, trial0xFFFFFFFF, nonzero64-bit session nonce.
READY payload69bytes: PTE SHA25632bytes, input scale float32LE, zero-point int32LE,
input byte count uint32LE, four uint16LE dimensions, dtype byte1(signed INT8),
16-byte build descriptor. Build descriptor is `eeg-stream-v1.0`, a protocol build
label, not a binary checksum or security attestation. Deployment manifest below
records actual binary/source SHA256 values. Host validates model/shape/quantization.

INFER payload is exactly11000unchanged two's-complement tensor bytes in C order.
ACCEPTED payload is inputCRC32(uint32LE). RESULT payload16bytes: inputCRC32,
logit0float32LE, logit1float32LE, predicted class byte, three reserved zero bytes.
Tie rule remains logit1>logit0 ?1:0. ERROR payload is numeric uint32 code; host
stops. STATUS can retrieve the last cached result/error. The first UI version
intentionally does not automatically retransmit or resolve timed-out requests;
it retains the unresolved ID and requires review/reconnect. A manual diagnostic
STATUS client can be added later without resetting hardware.

LOG is reserved: ordinary stdout/stderr is currently a 256-byte diagnostic RAM
ring, not wire LOG packets. This is a documented reduction from the optional LOG
proposal. No model references/predictions are compiled into streaming production
firmware. Only the separate FVP test fixture contains prerecorded input requests.

## Reproduce from local preserved sources

From the frontend development worktree, using a NEW/absent output location:

```sh
BASELINE_REPO='/path/to/original/Reproduce_Baselines'
"$BASELINE_REPO/.venv/bin/python" -B tools/model2_random_trials/create_streaming_project.py \
  --baseline-repo "$BASELINE_REPO"
"$BASELINE_REPO/.venv/bin/python" -B tools/model2_random_trials/prepare_streaming_assets.py \
  --baseline-repo "$BASELINE_REPO"
```

The project creator refuses an existing rollback directory; it never overwrites
original firmware. It creates the archive and isolated project under ignored
`artifacts/`, verifies every archive member, and imports the original proven tool
PATH into the development workspace only. If creating this from another machine,
restore the ignored original training/export/dataset assets first; these scripts
are not a download/install or model-export workflow.

Use CMSIS Developer Assistant in the **frontend window only**:

1. `open_solution`: `artifacts/streaming-firmware/project/cmsis-executorch.csolution.yml`.
2. `build`, target `SSE-320-U85`; confirm status success.
3. `run`, target `SSE-320-U85`; this starts only FVP.
4. Run `node tests/streaming/compare-fvp.cjs` after `STREAM_TEST PASS` appears in
   `artifacts/streaming-firmware/project/fvp-stream.log`.
5. After smoke success, rerun the asset preparer with `--full`: original4 + all144
   seed42 held-out trials + first3 repeated =151. Build/run FVP again. No trial
   selection is based on correctness; all three flagged recordings are included.
6. `build`, target `DevKit-E8`, is a compile-only test. Never invoke load/run on
   that target before separate physical-deployment approval.

All CMSIS builds, simulator runs and eventual hardware loads must use CMSIS tools,
not shell cbuild/debugger commands. No exporter/retraining/quantization is used.
The generated AI-layer model C/header and EmbeddedModule sources are verified
byte-identical to originals during preparation.

Local tests (no serial port):

```sh
clang++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined \
  tests/streaming/native_tests.cpp -o artifacts/streaming-firmware/parser-tests
artifacts/streaming-firmware/parser-tests
EDGE_STREAM_NATIVE="$PWD/artifacts/streaming-firmware/parser-tests" node tests/streaming/host-tests.cjs
node tests/streaming/failure-tests.cjs
node frontend/tests/integration.cjs
node frontend/tests/recorded-eeg.cjs
CINDY_UI_DIR=frontend/design-reference/cindy-new-ui node frontend/tests/design.cjs
```

Browser tests use an isolated QA profile and fake serial only:
`node tests/streaming/browser-stream.mjs`; existing browser/visual tests still apply.
Never run QA against the user's normal Chrome profile or real serial permissions.

## Frontend modes and local assets

Legacy four-trial demo: `http://127.0.0.1:8000/?v=eeg-1`.
Opt-in streaming preview: `http://127.0.0.1:8000/?stream=1`.
Do not connect the streaming preview to hardware until deployment is authorized.
Opening either URL never opens a serial port or starts inference.

Jihyung's stylesheet and SVG design references remain unchanged. Minimal guarded
GUI hooks permit continuous playback and genuine Session2 EEG buffers. The public
ArmEEGDemo/ArmEEGHardware APIs remain available. The default playback completion
at four remains intact; opt-in streaming uses no fixed trial-count limit.

`frontend/streaming/data/` is ignored except its .gitignore. Generation creates
inputs.bin1,584,000bytes, eeg.f326,336,000bytes, and identity/hash metadata.
No host expected logits or model evaluation code is included in browser assets.
Inputs and EEG have asset checksums; each selected input is verified again against
its original Phase1 SHA256. Channels are cached per recording across repeat cycles.
Licensing review is still required before publishing any of these larger assets.

Connect validates READY before enabling Play. Play sends one trial, waits for real
RESULT and finishes its1.8s visualization before advancing. Pause stops new sends
and freezes visualization while retaining any in-flight result. Reset waits for
in-flight completion, archives old results and starts a new seed/sequence; unresolved
requests block reset. Disconnect/timeouts stop scheduling; no automatic reset,
reflash, retransmission, mock fallback or fabricated prediction exists.

The existing accuracy figure is labeled Visualized accuracy in streaming mode.
UART status separately reports completed attempts, unique source recordings,
attempt correctness, unique-first correctness and repeated-prediction disagreements.
Capture export includes original response frames, outbound requests, model handshake,
IDs/checksums, current results and prior session archives. Captures remain in page
memory; export before reload/closing the tab. Long-running raw-frame capture grows
memory; indefinite unattended operation is not yet qualified.

## Physical deployment procedure — verified; future loads require authorization

Before asking for board approval, verify the final DevKit build and deployment
manifest, the rollback archive, all original hashes, FVP comparison and local tests.
The original J-Link session may still be open: do not detach/reconfigure/reset it
without separate authorization. Loading can restart execution and is a board
modification even if the command is called only `load`.

After explicit authorization for a single controlled deployment:

1. Preserve Chrome capture/export and stop the old demo reader as needed through
   its UI. No agent serial tools claim the port. Keep SW4 on UART4 and115200/8N1.
2. CMSIS selects the **isolated** DevKit-E8/M55_HP project and confirms its image
   and target. Inspect `get_section_layout`/build artefacts for the exact load ranges;
   use the generated application-only cbuild-run recipe, not provisioning tasks.
3. Use one authorized CMSIS application load of the final isolated image. Do not
   erase the chip, update SE/debug stubs, run SETOOLS or re-export the AI layer.
   Any required start/reset must be explicitly included in that approval; do not
   assume load implies permission for repeated resets.
4. Chrome opens the opt-in streaming URL, Connect UART selects SEGGER, and HELLO/
   READY verifies protocol, model SHA and tensor contract. Keep Play paused.
5. Start a small seeded hardware sequence and inspect ACCEPTED/RESULT IDs, CRCs,
   actual logits and matching prerecorded EEG/labels. Export capture; compare
   against independent TOSA/FVP at the approved 0.01 tolerance; retain the original Q/DQ 0.01 diagnostic separately. This is
   the first real UART RX/IRQ/FIFO throughput test, not already proven by FVP.
6. Stop on error/no response; retain evidence. Do not automatically reset, retry
   deployment or initiate recovery. Request review before any further board action.

Rollback, if separately authorized, loads the preserved original M55_HP image
from `artifacts/streaming-rollback/validated-four-trial.tar.gz` after checksum
verification and restores the demo URL. Confirm the original four predictions and
logits rather than treating a successful load as validation. Archive includes
original source/config, model, AXF/HEX and prior physical UART evidence; no new
firmware has overwritten that archive or the original worktree.

## Corrected startup and preservation

Use CMSIS `load_and_debug` on the isolated DevKit-E8/M55_HP context, wait for
`main`, then explicitly `continue_execution`. Verify actual programmed bytes against
the selected HEX using CMSIS memory tools before treating the deployment as done;
a successful load task or a running server alone is insufficient. Confirm the
receive loop and valid HELLO/READY model/tensor contract. Never rebuild/export the
model or run provisioning tasks as part of this procedure. The successful physical
image is fingerprinted in `tools/model2_random_trials/validation/physical-streaming-build.json`.

The original four-trial rollback procedure and archive are unchanged. Compact
physical results are tracked in `tools/model2_random_trials/validation/`; raw EEG,
full Chrome captures and HEX/AXF stay ignored. See `tools/model2_random_trials/LOCAL_BACKUP_INVENTORY.md`
for the local assets and tool dependencies required beyond a fresh Git clone.
