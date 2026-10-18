# Multi-subject streaming mode

Open `http://127.0.0.1:8000/?stream=multisubject` when the server serves `frontend/`.
The existing `?stream=1` Model 2 streaming mode and `?v=eeg-1` four-trial replay
remain unchanged. No page opens a serial port automatically. The new mode requires
its separately built multi-subject firmware; the Model 2 handshake is rejected.
Do not deploy or reset hardware as part of frontend tests.

`profile.js` describes the unfused v4 wide32-ft seed-0 PTE by SHA256, its 5,632-byte
INT8 wire tensor and Float32 PTE input contract. The shared binary protocol and
one-request controller are reused, while the original Model 2 scripts remain
byte-identical. UART is 115200/8N1. No prediction reference or model evaluator is
loaded by this page. Every completed result comes from an accepted CRC-checked,
session/request/trial-matched response; desktop and FVP references are offline only.

`data/` contains local-only genuine EEG assets prepared by the dataset utility.
The loader checks asset hashes, exact model/quantization/shape, channel order,
1,080 unique identifiers, 120 Session-2 trials 25–144 per participant, and all94
artifact flags. Trial labels are verified metadata and are never derived from
predictions. Participant Session1 and Session2 trials1–24 are excluded. Inputs are
aligned/scaled/quantized model data. EEG displays use CAR-filtered-decimated µV
before EA/scaling/quantization, at125Hz for256 samples (2.048s), preserving physical
electrode identity. RMS is relative channel activity, not source localization.

The existing controls run the same continuous seeded selection and1.8s delayed
visualization. There is no inference count limit. One complete cycle visits every
eligible recording exactly once, including flagged recordings, then reshuffles.
Pause blocks further sends and retains any in-flight result; Resume continues.
Reset archives completed results and creates a new sequence without resetting the
board. Disconnect/errors retain capture and stop scheduling without retry.

## Session-summary interface

Jihyung's `feature/session-summary` is not merged. Presentation CSS/SVG, cursor,
person, accuracy and timing remain unchanged. `window.EdgeStreamingController`
continues to expose the controller. Its existing `records` are preserved; new
`completedRecords` contains immutable standardized records with these extra fields:

- `schema_version:1`, `model_id`, `model_sha256`, `experiment_id`
- `subject_id` (1–9), `session_id:'1test'`, `session_trial` (25–144)
- Existing `trial_id`, wire `trial`, `session`, `request`, `ground_truth`,
  actual `class_id`/`logits`, `artifact`, input SHA/CRC, seed/cycle and receive time
- `unique`/`repeated` within the current run (Reset starts a new run)
- `timing.hardware_inference_ms:null`: firmware does not report isolated inference time
- `timing.host_round_trip_ms`: request-write start to matched RESULT receipt
- `timing.host_ack_to_result_ms`: matched ACK receipt to RESULT receipt; includes
  host buffering/scheduling, so this is not hardware-only inference latency
- `timing.visualization_duration_ms:1800`: intentional display delay, not inference time

The page emits `edgebci:trial-completed` with the standardized immutable record
as `CustomEvent.detail` only after a valid actual response. This is completion of
inference, before visualization may finish; summary code must not count it again
when animation finishes. `EdgeMultisubjectSession.getCompletedTrials()` returns a
snapshot. `getCapture()` / `exportCapture()` include records, prior reset archives,
prior connections, handshake and raw UART frames, with no synthetic predictions.

`await EdgeMultisubjectSession.endRun()` stops sending, safely waits for an
in-flight request (or its bounded timeout), cancels delayed visualization, retains
all completed results, and emits `edgebci:run-ended`. It does not close/reset the
board. Play thereafter requires Reset. End Run is an integration API, not a new
visual control, so Jihyung's future session-summary control can call it directly.
A pending failed request remains unresolved in the exported state; no result is
fabricated. Export before closing/reloading the page; capture is held in RAM.
