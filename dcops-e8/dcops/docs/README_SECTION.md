## DCSM-Simplified: build, export, flash, run

A handheld, air-gapped data-center component identifier on the Alif Ensemble E8 DevKit. The camera
feeds a MobileNetV2 on the Ethos-U85, and the LCD shows the component and its info card. A DS-CNN
keyword spotter on the same NPU drives an inspection checklist by voice (`go`, `yes`, `no`, `up`,
`down`, `stop`). It ports our earlier phone project [DC-Ops](https://github.com/abhijitbetigeri/DC-Ops),
whose dataset, YOLOv8n-seg detector and info-card text we reuse. This repo is a fork of
[npow/alif-mlek-yolo](https://github.com/npow/alif-mlek-yolo) (camera pipeline by Nissan Pow). The
distillation pipeline, board-camera fine-tune, ExecuTorch/Ethos-U85 export and two-model firmware
are new. Details: [`dcops/docs/ARCHITECTURE.md`](dcops/docs/ARCHITECTURE.md).

### Prerequisites

- Arm GNU toolchain 14.3.1, CMake ≥ 3.21, `git submodule update --init --recursive`
- ExecuTorch 1.1.0 Python environment at `resources_downloaded/env` (pinned to match MLEK). See
  [docs/blazeface-executorch.md](docs/blazeface-executorch.md#build) for the venv and
  `resources_downloaded_metadata.json` setup.
- SEGGER J-Link (on-board J-Link on the PRG USB port)

### 1. Train (optional, GPU)

All training ran on Colab GPUs. Scripts are in `dcops/jobs/`, notebooks in `dcops/colab/`.

| Step | Script | Notebook |
|---|---|---|
| Teachers (wave 1, wave 2) | `dcops/jobs/teacher.py`, `teacher_v2.py` | `dcops_overnight.ipynb`, `dcops_wave2.ipynb` |
| Vision student | `dcops/jobs/student.py` | `dcops_overnight.ipynb`, `dcops_wave2.ipynb` |
| Keyword spotter | `dcops/jobs/kws.py` | `dcops_overnight.ipynb` |
| Board-camera fine-tune | `dcops/jobs/finetune_board.py` | `dcops_board_ft.ipynb` |

Board-camera data: `dcops/board_capture/` (`capture.py` in clock mode records the camera's
model-input frames while a phone shows a clock-synced slideshow; `build_ft_data.py` and
`build_deck2.py` turn captures into a labelled, slide-split dataset).

Dataset: [`dcops/data/`](../data/README.md) in this repo (component photos and board-camera captures).
Landing page and slides: https://abhijitbetigeri.github.io/executorch-alife8/
Trained models (deployed, ablations, teachers): [abhijitbetigeri/dcops-e8](https://huggingface.co/abhijitbetigeri/dcops-e8) on Hugging Face.

### 2. Export to ExecuTorch for the Ethos-U85-256

```sh
# run folder contains student.pt, metrics.json, calib.npz, labels.txt
resources_downloaded/env/bin/python dcops/export/export_student.py <run-dir>   # -> <run-dir>/student_ethos-u85-256.pte + export.json
resources_downloaded/env/bin/python dcops/export/export_kws.py     <kws-run-dir>
```

Pipeline: torch.export -> PT2E with `EthosUQuantizer` (symmetric per-channel int8) ->
`to_edge_transform_and_lower` with `EthosUPartitioner` -> TOSA -> Vela 4.5.0 (`ethos-u85-256`,
`Ethos_U85_SRAM_MRAM`, `Shared_Sram`) -> `cortex_m` passes -> `.pte`.

Check `export.json`: `cpu_ops` should list only the quantize/dequantize boundary, and
`float_int8_agreement` should be close to 1 (deployed vision model: 98.4%).

### 3. Build the demo firmware (M55 HP, `build-fusion`)

```sh
scripts/alif/build_dcops_demo.sh
```

It configures `build-fusion` with the deployed vision (`v4_demo_ft`) and KWS (`dscnn_m_agc`)
models, `CLS_ALLOWED_MASK=0x13250`, `CLS_CONF_THRESHOLD=0.6`, `KWS_THRESHOLD=0.5`, builds, and prints the image end
address (deployed build: 0x805473C4, limit 0x80578000).

Manual build: the classify build plus these options (see `source/app/use_case/alif_object_detection/usecase.cmake`):

| Option | Demo value | Meaning |
|---|---|---|
| `-Dalif_object_detection_ET_TASK` | `CLASSIFY` | classifier task |
| `-Dalif_object_detection_MODEL_PATH` | `resources/object_detection/dcops/dcops_v4_demo_ft_ethos-u85-256.pte` | MobileNetV2 student, board-camera fine-tuned on the demo classes |
| `-Dalif_object_detection_CLASS_LABELS` | `labels.txt` | 16 classes + background |
| `-Dalif_object_detection_CLS_ALLOWED_MASK` | `0x13250` | power shelf, network port, fan, power connector, drive bay (+ background); default all |
| `-Dalif_object_detection_CLS_CONF_THRESHOLD` | `0.5` | smoothed top-1 needed to show a label |
| `-Dalif_object_detection_CLS_EMA_ALPHA` | `0.3` (default) | EMA weight of the newest frame |
| `-Dalif_object_detection_ENABLE_VOICE` | `ON` | second `.pte`: keyword spotter |
| `-Dalif_object_detection_KWS_MODEL_PATH` | `resources/kws/dcops/dscnn_m_agc_ethos-u85-256.pte` | DS-CNN |
| `-Dalif_object_detection_KWS_THRESHOLD` | `0.5` (demo build; default 0.7) | averaged softmax needed to fire |
| `-Dalif_object_detection_KWS_ACTIVATION_BUF_SZ` | `0x00020000` (default) | KWS arena, 128 KB |

Info cards come from `dcops/dc_components.json` by default (`CLS_KNOWLEDGE_BASE`).

### 4. Flash, using the guard script

```sh
scripts/alif/flash_e8_hp.sh build-fusion/bin/mlek_alif_object_detection.axf
```

The HP app partition is **0x80200000–0x80580000**, and Alif's ATOC boot table sits at its top
(0x8057BC00). The linker accepts larger images, but flashing one stops the HP core from booting.
The script refuses any image ending past **0x80578000**. Do not flash with a raw `loadfile` script.
If boot is lost anyway, see "Recovering a board" below.

### 5. Power-cycle, then run

**After every flash, unplug the PRG USB cable for about 10 seconds and plug it back in.** The MT9M114
camera hangs after a J-Link reset. If camera init fails, the firmware blinks LED1: power-cycle again.

Point the camera at server hardware, or at a photo on a phone, from 20–30 cm away, with the
component in the centre of the frame. A label and info card appear only when the smoothed
confidence is at least 0.5; otherwise the card stays blank.

Voice: `go` latches the identified part and opens its checklist (only when a label is shown),
`yes` / `no` mark the selected item pass / fail, `up` / `down` move the cursor, `stop` returns to
scanning. `no` on item 1 ("is this the right part?") rejects the identification.

### Performance

Measured on build-fusion: vision NPU 7.9–8.0 ms, preprocessing 1.75 ms, frame 255–287 ms
(3.5–3.9 FPS), KWS MFCC 3.3–4.0 ms + DS-CNN 0.34 ms. Full tables:
[`dcops/docs/RESULTS.md`](dcops/docs/RESULTS.md) and
[`dcops/docs/DEVPOST.md`](dcops/docs/DEVPOST.md#performance--efficiency-20). The LCD footer shows
frame time and FPS.

### Recovering a board that no longer boots the HP core

Symptom: after flashing, the screen stays dark; J-Link to `AE822FA0E5597LS0_M55_HP` fails with
"Could not read AHB ROM register / Could not find core", while `AE822FA0E5597LS0_M55_HE` still connects.
Cause (seen on 2026-10-17): an image that extended past 0x8057BC00 overwrote Alif's ATOC boot table
(entries DEVICE, A32_STUB, HP_APP, HE_STUB at the top of the HP partition), so the Secure Enclave stops
starting the HP core. `scripts/alif/flash_e8_hp.sh` now refuses such images.

Recovery that worked, using a full backup of the HP partition taken before flashing:

```bash
# backup (do this once, before experimenting): reads 0x80200000..0x80580000 (3.5 MB)
printf 'connect\nh\nsavebin backup.bin 0x80200000 0x380000\ng\nexit\n' > /tmp/b.jlink
JLinkExe -device AE822FA0E5597LS0_M55_HP -if SWD -speed 4000 -autoconnect 1 -CommanderScript /tmp/b.jlink

# restore THROUGH THE HE CORE (the HP core is unreachable), then power-cycle the board
printf 'connect\nh\nloadbin backup.bin 0x80200000\nexit\n' > /tmp/r.jlink
JLinkExe -device AE822FA0E5597LS0_M55_HE -if SWD -speed 4000 -autoconnect 1 -CommanderScript /tmp/r.jlink
```

Verify by reading 0x8057B000..0x80580000 back and comparing it with the same range of the backup.
Without a backup, rewrite the boot table with Alif SETOOLS (`app-gen-toc` + `app-write-mram`).
