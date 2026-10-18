# Verified prerecorded EEG for the four Model 2 display trials

These are genuine Participant 1 BCI Competition IV Dataset 2a training-session
samples. They are **prerecorded EEG**, not live acquisition from Alif E8. The
firmware continues sending only predictions/logits; no UART or firmware change
is involved. Original cache/model/run files remain authoritative and unchanged.

## Where the recordings came from

Original baseline repository:
`/Users/philippinechauvet/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Reproduce_Baselines`.

- Original recording: `mne_data/MNE-bnci-data/~bci/database/001-2014/A01T.mat`.
- Selected window identities/labels: `artifacts/fvp-subject1-20261016-ready/reference.json`.
- Pre-quantization normalized float32 tensors:
  `artifacts/subject1-ethosu85-seed42-20261016/export_data.pt` (`development_x`).
- Original trial metadata and normalization/filter parameters:
  `artifacts/subject1-ethosu85-seed42-20261016/data_manifest.json` and
  `float_preprocess_params.json` in the same directory.
- Staged int8 windows: `artifacts/fvp-subject1-20261016-ready/validation_vectors.npz`
  and `project/src/eeg_vectors.h` below that run.
- Preprocessing code: `src/data/preprocess.py` (`extract_session`, `filter_trial`)
  and `src/experiments/dropout_cv.py` (`load_allowed`). No dataset download,
  training, model execution, physical UART or debug operation is performed here.

| UART trial | Verified source identity | Retained sample bounds [start, stop) | Export development index | Original MAT cell / trial (1-based) | Task label |
|---|---|---|---|---|---|
| 0 | S1/0train/run0/cue20544 | [20794, 21294) | 0 | 4 / 11 | Left (0) |
| 1 | S1/0train/run5/cue750 | [1000, 1500) | 2 | 9 / 1 | Left (0) |
| 2 | S1/0train/run0/cue4671 | [4921, 5421) | 1 | 4 / 3 | Right (1) |
| 3 | S1/0train/run4/cue81163 | [81413, 81913) | 8 | 8 / 41 | Right (1) |

The original MAT file contains three calibration cells before its six task runs;
these are excluded from the converted task-run numbering 0–5. Cue and window
sample indices are zero-based within the task run. Each artifact flag is false.
Task labels were checked against the original MAT `y`, export `development_y`,
staged labels, and existing verified deployment profile. They are never inferred
from the model predictions.

## Processing and units

Each display buffer is **22 × 500 float32 samples in microvolts (µV)**, sampled at
**250 Hz**: the two-second cue-relative window **[1, 3) seconds**.

The MAT samples are first converted from µV to volts, as in the original pipeline.
Filtering is a causal 7–30 Hz Butterworth bandpass, fourth-order prototype
(eighth-order bandpass), using the original second-order sections. Filter state
starts at zero at cue minus two seconds, giving three seconds of warm-up before
the retained window. No filter state carries across trials. Filtered volts are
converted back to µV and stored as float32 for the browser. **The display buffers
are filtered, but are not normalized, quantized, dequantized int8, or synthetic.**

Verification before export:

1. Filtered original recording samples, normalized with the original training-only
   mean/population std and cast to float32, match the four saved pre-quantization
   `development_x` windows **exactly**.
2. Requantization of those windows matches every byte of both staged NPZ inputs
   and `eeg_vectors.h`. SHA-256 of the 44,000 input bytes:
   `038c1f93ea0818fb2c2b21da6521506e38d222280a1178b09ec5ce3e28d9c8fb`.
   This is the same input hash previously read from the physical board.
3. The selected PTE hash matches the verified Model 2 profile:
   `644bf8865ed515ed41c5418070a0b147008c4326b9da34d3232884459d9b3298`.

`provenance.json` records source-file hashes, sample hashes, full source identities,
processing parameters and expected per-channel RMS. Each sample hash covers the
22 channels in the order below, each with 500 float32 little-endian values.
`model2-eeg-data.js` contains only these four windows plus provenance identifiers.

## Channel order and rendering

Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz,
CP2, CP4, P1, Pz, P2, POz.

Samples are passed by channel **name** to Jihyung's existing montage coordinates.
Her 22 electrodes and three waveform columns (8/7/7), colors, spacing, SVGs, and
stylesheet are unchanged. Her waveform renderer retains the full 500-sample
buffer and projects it to 160 SVG points, with independent channel peak scaling;
waveform height is not a calibrated voltage axis.

Her scalp renderer uses `sqrt(mean(samples²))` over the same 500 samples for each
channel, then divides by that trial's maximum channel RMS to drive the original
field opacity/radius. The underlying RMS units are µV; the displayed intensity is
**dimensionless relative RMS within one window**, not an absolute scale shared
across trials. This is illustrative scalp amplitude, **not cortical source
localization, ERD, or a clinical measurement**.

The complete recorded two-second window is held during the existing 1.8-second
prediction visualization. It does not pretend to stream new live samples. Both
panels update together when the visualized trial changes. The existing map note
says `Prerecorded EEG · trial N · filtered µV`; its tooltip gives full source
identity and processing context. Pause holds the displayed window; Reset clears
it until playback selects a trial again; disconnection/EOT does not erase it.

## Identity and unavailable-data behavior

The existing verified metadata profile must be selected for hardware capture;
saved replay already selects it. `recorded-eeg.js` checks profile ID, model/input
hashes, UART trial number, complete source trial metadata, labels, and channel/sample
shape. It resolves by **source trial ID**, not arrival sequence or predicted class.
Unknown profiles, mismatched metadata or missing data clear the recorded EEG
panels rather than substitute another window or generate samples. Ground truth,
logits, classifications, scoring, transport and cursor/arm timing are not changed.

## Regeneration and checks

From the frontend workspace, using the original baseline Python environment:

```sh
/path/to/Reproduce_Baselines/.venv/bin/python frontend/tools/export-recorded-eeg.py \
  --baseline-repo /path/to/Reproduce_Baselines \
  --out-dir /tmp/new-verified-eeg-export
node frontend/tests/recorded-eeg.cjs
node frontend/tests/integration.cjs
node frontend/tests/design.cjs
```

The exporter requires local original data and refuses to overwrite an existing
output directory. Normal browser use requires only the packaged four-window asset.
Browser replay tests additionally verify all 22 waveform SVG paths and scalp RMS
values against the corresponding prerecorded window for each of the four trials.
