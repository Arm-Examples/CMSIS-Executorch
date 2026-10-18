# EdgeBCI · Model 2 UART visualization

Jihyung's newer dark interface is now the presentation layer. Her `styles.css` is
byte-identical, and her index structure, human/scalp SVGs, 22 electrodes, three
waveform columns, target design, type, borders and desktop panel geometry are
preserved. Public `ArmEEGDemo` and `ArmEEGHardware` methods remain available.
No hardware/model changes are required.

## Open the frontend

From this repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory frontend
```

Open **http://127.0.0.1:8000/?v=eeg-1** in desktop Chrome/Edge. Hard-refresh an
older tab with Cmd+Shift+R. There are no npm or Python serial dependencies.

## Check animation without the board

Open **UART & replay** in the header and click **Replay saved UART**. The four actual saved Model 2 result strings and
`Execution_result: PASS (compare host reference separately)` are parsed as one
burst, just like physical firmware. The source is labeled **SAVED UART REPLAY**;
this never opens a serial port or runs inference. Visualization starts automatically.

Each prediction receives **1.8 seconds** of visual time, moving from the center
toward left for class 0 or right for class 1. Recentering each trial makes three
consecutive right predictions visibly distinct. This is **delayed visualization
of hardware results, not live inference timing**. The cursor is an element inside
the page, not the operating-system pointer. The existing EEG panels now show the verified prerecorded EEG window for each
known Model 2 trial, labeled **Prerecorded EEG · filtered µV**. This is not live
acquisition. No synthetic EEG is substituted in hardware or saved UART.

- **Pause** freezes the current visual trial; UART capture continues unchanged.
- **Play** resumes. After completion, **Replay results** replays the retained
  capture without re-running firmware or requesting the serial port again.
- **Reset** pauses and recenters, rewinding all retained results to trial 0. It
  does not reset the board, erase capture, or change predictions/logits.
- **Captured results** lists the original UART classes/logits even after EOT,
  serial disconnection, pause, reset or completion. Results are retained in this
  page's memory; reloading the page or starting a new capture clears that session.
- **Use mock preview** explicitly switches to the original synthetic fixture.
  Errors never silently select mock data.

## Verified task cues (independent ground truth)

UART results have no ground-truth or model-identity fields. For a live run of the
**unchanged verified Model 2 deployment**, open **UART & replay** and choose **Verified Model 2 · four staged
trials** under Task cue metadata **before Connect UART**. The selection is
snapshotted for that capture. Unknown deployment is the default and keeps the
person neutral. Saved replay explicitly selects its known Model 2 profile.

`model2-trials.json` and `trial-metadata.js` record the source and trial mapping.
Verification on 2026-10-17 checked original export `development_y`, selected source
trial manifest entries, reconstructed quantized export windows, staged NPZ and
`eeg_vectors.h`. The 44,000 input bytes have SHA-256
`038c1f93ea0818fb2c2b21da6521506e38d222280a1178b09ec5ce3e28d9c8fb`,
matching the physical-board memory read recorded in the baseline worktree's
`results/e8-model2-uart-validation-20261017-1404/README.md`. Model PTE SHA-256 is
`644bf8865ed515ed41c5418070a0b147008c4326b9da34d3232884459d9b3298`.
The source preprocessing contract maps left_hand=0, right_hand=1.

| UART trial | Development index | Subject/session/run | Cue sample | Dataset task | Model prediction |
|---|---|---|---|---|---|
| 0 | 0 | S1 / 0train / 0 | 20544 | Left (0) | Left (0) |
| 1 | 2 | S1 / 0train / 5 | 750 | Left (0) | Right (1) |
| 2 | 1 | S1 / 0train / 0 | 4671 | Right (1) | Right (1) |
| 3 | 8 | S1 / 0train / 4 | 81163 | Right (1) | Right (1) |

The accuracy counter scores only the ground-truth label paired with each
visualized result; it never scores against a separately displayed cue. These four
trials end at **75% (3 / 4)**. Unknown metadata remains **— (0 / 0)**. Reset clears
the visual tally before replaying retained results, so replay does not double-count.

In particular, trial 1 intentionally shows a **left-hand imagery cue and a
right-moving cursor**. The task labels are never inferred from model predictions.
Arm movement illustrates the recorded motor-imagery cue, not measured physical
arm motion. Without the verified profile or a supplied ground-truth API message,
the person stays neutral. A matching prediction sequence alone is not evidence
that another firmware deployment contains these same windows.

Authoritative local sources (baseline worktree):
`artifacts/fvp-subject1-20261016-ready/reference.json`,
`artifacts/subject1-ethosu85-seed42-20261016/export_data.pt`,
`artifacts/subject1-ethosu85-seed42-20261016/data_manifest.json`,
`artifacts/fvp-subject1-20261016-ready/validation_vectors.npz`, and
`artifacts/fvp-subject1-20261016-ready/project/src/eeg_vectors.h`.

## Physical UART capture

Close other serial readers, choose the appropriate task metadata profile, then
click **Connect UART** and select SEGGER UART4 (previously mapped to
`/dev/cu.usbmodem0012191658361`). Web Serial opens 115200 baud, 8N1, no flow control.
The transport never writes commands, explicitly toggles reset signals, flashes,
erases, provisions, or starts firmware. A port-open result alone proves no inference.
Connecting and saved replay arm automatic visual playback; Pause remains available.

Connect **before** firmware executes: the current image emits four windows and
exits. This frontend cannot restart it. Any new physical restart remains a separate,
user-authorized CMSIS action. No hardware run is part of this animation change.

Firmware execution status, link status, and visualization status are separate:
**Complete** requires four unique trials 0–3 plus the exact PASS token (with the
firmware's optional parenthesized explanation). EOT marks execution ended; it does
not assert UART disconnection or discard the queue. Disconnect without PASS is an
incomplete capture, not success. EEG_ERROR or a failing execution line reports
failure while retaining already captured results. Latest capture and the current
visualized trial/logits are shown separately.

## Checks

```sh
node frontend/tests/integration.cjs
node frontend/tests/recorded-eeg.cjs
node frontend/tests/design.cjs
node --check frontend/gui.js
node --check frontend/playback.js
node --check frontend/serial-transport.js
node --check frontend/uart-parser.js
node --check frontend/hardware-adapter.js
```

The deterministic integration check executes the actual GUI, adapter and transport
using the saved 340-byte physical capture (`tests/model2-uart.txt`), a fake serial
port, and an animation clock. It verifies burst/EOT/disconnect retention, direction
and 1.8-second trial playback, pause/resume/reset, immutable logits, independent
verified task cues, unknown-profile neutrality, PASS/failure handling, and errors.
No real serial port is opened by these tests.

## Jihyung Ku design merge and preservation

Original reference folder: `/Users/philippinechauvet/Desktop/cindy-new-ui/`.
None of its six supplied files, including `Newer UI.pdf`, were changed. Their
SHA-256 values are in `tests/cindy-source-manifest.json`.

The previous working frontend was archived and all 11 files hash-verified at:
`Reproduce_Baselines/artifacts/frontend-preserved-before-cindy-ui-20261018T005618Z/`.
The frontend worktree was on `codex/edgebci-hardware-playback` with an untracked
`frontend/` directory. This merge is on `codex/cindy-ui-uart-merge`; no commit or
push was made. The archive remains the complete restore source for the old UI.

Implementation:

- `index.html`, `styles.css`, `gui.js`: Jihyung's presentation, with focused queue,
  return-state and retained-result hooks in the GUI. The stylesheet is unchanged.
- `integration.css`, `hardware-controls.js`: only the collapsible UART controls.
  It occupies the header on desktop; on mobile the one integration widget moves
  outside the filtered shell to keep its popup in the viewport. Original panels
  and breakpoints are not overridden.
- `hardware-adapter.js`: the working adapter retained, forwarding full trial,
  logits and matching ground-truth fields, with Jihyung's full 22-channel list.
- `playback.js`, `uart-parser.js`, `serial-transport.js`: retained transport and
  delayed visualization. Saved replay is visibly labeled separately from hardware.
- `model2-trials.json`, `trial-metadata.js`: existing verified deployment profile.

## Browser verification

The in-app browser tool could not initialize due to its sandbox metadata error.
Verification instead used a separate headless Chrome profile, with no existing
user profile or real serial permissions. No physical-board run was initiated.

Passed:

- Visual inspection against the supplied PDF and original source screenshots.
- Exact original-versus-merged desktop panel geometry, fonts, borders and
  backgrounds at **1512 × 950**. The PDF contains the same interface with an
  additional top screenshot margin; artwork and panel layout agree.
- All 22 electrodes and waveform columns **8 / 7 / 7**.
- Actual browser saved-burst playback, class directions **0/1/1/1**, independent
  cues **0/0/1/1**, final **75%, 3/4**, pause, reset and retained results.
- Connect/Disconnect buttons and 115200/8N1 settings with an injected **fake**
  Web Serial port; EOT/completion/disconnection separation and retained capture.
- Blank hardware EEG when samples are absent, including missing channels.
- Original responsive panel geometry at widths **390, 720, 900**; no horizontal
  overflow; all added UART controls fit the viewport.
- No JavaScript exceptions or resource-load errors.

Dependency-free browser checks use Node with global WebSocket support (Node 22+)
and an explicitly isolated Chrome debugging endpoint on port 9228. Start Chrome
with a fresh temporary profile, never the user's normal profile:

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --remote-debugging-port=9228 --user-data-dir=/tmp/edgebci-ui-qa-chrome --no-first-run --disable-background-networking --disable-sync --disable-extensions about:blank
# In another terminal, with the frontend server on port 8000:
CINDY_UI_DIR=/path/to/cindy-new-ui node frontend/tests/visual.mjs
CINDY_UI_DIR=/path/to/cindy-new-ui node frontend/tests/browser.mjs
```

`EDGEBCI_URL` and `EDGEBCI_QA_DIR` optionally select the URL and screenshot folder.
The browser test injects a fake serial implementation before exercising Connect.
The parser/playback tests never access serial. A new **physical UART smoke test
with this merged UI remains unperformed** by design and requires separate approval.


## Prerecorded EEG panel update

The only new runtime behavior is feeding the verified recorded window to the EEG
renderers at the start of its corresponding visualized trial, and clearing EEG
when rewinding or encountering unknown trial metadata. [EEG provenance and units](recorded-eeg/README.md)
include all four original source identities, the 22-channel order, source hashes,
filtering and exact export/staged-input comparisons. Buffers are filtered float32
microvolts before training normalization and int8 quantization. The scalp field is
relative per-window channel RMS, not a cortical/clinical measurement.

`recorded-eeg/model2-eeg-data.js`, `recorded-eeg/provenance.json`,
`recorded-eeg.js`, and `tools/export-recorded-eeg.py` supply the display-only data.
Jihyung's EEG renderer, montage and styling are retained. The existing UART transport,
parser, hardware adapter, playback queue, ground-truth metadata and all styling
files are byte-identical to the working version before this EEG change. Cursor,
arm, prediction, scoring and timing logic are unchanged; regression tests still
assert directions 0/1/1/1, task cues 0/0/1/1 and 75% (3/4).

The pre-change frontend is preserved in the baseline repository at
`artifacts/frontend-before-recorded-eeg-20261018T033829Z/`. Work is on
`codex/model2-recorded-eeg`, with no commit or push. Saved replay and a fake-serial
hardware-result path are tested; no physical-board run, serial claim, reset,
firmware or model modification was performed for this change.

## Preserved demonstration snapshot

The user confirmed successful physical Alif E8 Model 2 operation with this UI,
including UART, EEG panels, cursor, task arms and accuracy. Automated tests use
saved physical UART output and simulated serial transport; they do not rerun the
board. Earlier verification notes above describe checks at their original time.

Jihyung’s six untouched originals are preserved in `design-reference/cindy-new-ui/`,
including the PDF. For portable reference checks:

```sh
CINDY_UI_DIR=frontend/design-reference/cindy-new-ui node frontend/tests/design.cjs
```

The four processed EEG excerpts in `recorded-eeg/model2-eeg-data.js` are necessary
for offline waveform/scalp playback. They are display-only, filtered microvolt
windows with verified provenance, not the full dataset or model inputs. Raw MAT
recordings, training artifacts, models, virtual environments, temporary browser
profiles and local archives are not part of this frontend snapshot.
