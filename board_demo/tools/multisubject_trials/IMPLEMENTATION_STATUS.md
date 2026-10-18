# Multi-subject integration status

Branch: `feature/multisubject-eeg-trials`, based on validated Model 2 commit
`f9c40dd0fcfcec633f18786e63c1f6c0ea74a2a2`. This implementation is preserved on this branch; Git HEAD identifies its revision.

Implemented in separate directories:

- Genuine 1,080-window pool: nine participants, 120 eligible Session-2 recordings
  each, 540 left/540 right, all 94 artifact flags retained. Original trial IDs and
  ground truth remain independent of predictions. Display uses real pre-EA µV.
- Unchanged v4 student PTE, INT8 wire-to-float32 input adapter, bounded bidirectional
  UART framing/CRC/session/request/trial IDs and model-specific READY contract.
- Continuous seeded selection without replacement, reshuffled cycles, one actual
  response per request, safe Pause/Resume/Reset, retained captures and distinct
  unique/repeated metrics. No fixed inference limit.
- Separate frontend profile/controller/bridge. Cindy's GUI/CSS, animations and
  public hardware APIs are preserved. Immutable completed-trial and End Run APIs
  are available for her session-summary work; that branch is not merged.

## Completed validation

- Seven dataset integrity/provenance tests passed.
- Native C++ parser under ASan/UBSan: corruption, bounds, timeout, duplicate/session
  behavior, exact input reconstruction and 1,200-request reuse soak passed.
- JS controller/native parser interoperability: 1,081 simulated requests, full pool
  plus reshuffle, passed. This endpoint uses a clearly identified toy test callback.
- Production Cindy Ku GUI synchronization with genuine prerecorded EEG and simulated
  UART passed: exact scalp RMS, independent ground truth/prediction, cursor,
  accuracy, controls, safe End Run, reset archives and disconnect retention.
- Existing connection/error regression tests passed; no fabricated predictions.
- FVP: all 1,080 unique inputs, 20 released vectors and 14 reordered repeats passed
  exact integer TOSA comparison. See NUMERICAL_VALIDATION_REPORT.md for the retained
  desktop diagnostic failures and two changed decisions on tied desktop logits.
- DevKit-E8 M55_HP application compiled successfully; no load or board operation.
  Image hashes and memory figures: `validation/streaming-image.json`.
- Local page, all 16 scripts, dynamic bridge and metadata returned HTTP 200.
  Actual Web Serial/Chrome visual operation with the new model is not yet tested.

The macOS tools blocker was corrected only in the isolated project: native FVP
has no Darwin executable, so its unsupported package requirement was removed.
The existing Docker image/shim remains pinned to 11.32.23. The extension regenerated
its tools environment; schema, paths, CMake, Ninja and compilers now pass.
The original Model 2 project/tool environment was not edited.

Original four-trial fallback: archive and all 136 original files checksum-verified.
Working Model 2 streaming HEX/AXF, UART adapter and physical capture also match
their saved hashes. Both preservation branches remain at their prior commits.
See DEPLOYMENT_REVIEW.md before any separately authorized physical test.

## Local-only reproduction dependencies

Back up `artifacts/multisubject-v4/` (release bundle, pinned source snapshot, prepared
inputs/display EEG, numerical/FVP evidence and generated project), original raw
A01E.mat–A09E.mat recordings, and the working Model 2 template project. Keep model
binaries, EEG datasets and generated images out of Git. Installed licensed Arm
compiler/packs, existing UART driver dependencies and Docker/FVP are also required.
Use the source, release and dataset manifests to verify a restored installation.

Preview: `http://127.0.0.1:8000/?stream=multisubject`. The existing Model 2 mode
`?stream=1` and four-trial replay `?v=eeg-1` remain available. The new mode rejects
Model 2 firmware; preview does not authorize a physical connection or deployment.

The final target-selection-only incremental SSE build did not rewrite its unchanged
AXF and CMSIS therefore marked that no-op build unconfirmed. No image was loaded
after it. The actual FVP executions and the DevKit-E8 compile above had separately
confirmed successful builds; their evidence and image hashes are retained.
