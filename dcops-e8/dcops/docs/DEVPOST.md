# DCSM-Simplified: an air-gapped data-center component identifier

*DataCenter Service Management, Simplified.*

> Devpost text. Every number below is measured; `TODO(...)` marks the few items not measured or not done yet.
> Repo: https://github.com/abhijitbetigeri/executorch-alife8 (fork of
> [npow/alif-mlek-yolo](https://github.com/npow/alif-mlek-yolo)) · Judged commit/tag: `TODO(tag)` ·
> Demo video: `TODO(record)`

## Inspiration

Secure data centers often ban phones, cameras and cloud services on the floor. Technicians still
have to identify components they rarely touch: a power shelf, a drive bay, a network port, a fan.
They do it from memory or a printed binder. We wanted a small device with one fixed function that
runs fully offline: point it at the hardware and it tells you what the part is, then walks you
through an inspection checklist by voice. Nothing to upload, no general-purpose camera app.

**Prior work, disclosed:** this ports our earlier phone project
[DC-Ops](https://github.com/abhijitbetigeri/DC-Ops) (Qualcomm x Meta ExecuTorch hackathon:
YOLOv8n-seg on a Snapdragon NPU, 16 component classes, component info cards). We reused its
dataset, its fine-tuned YOLOv8n-seg detector (as a pseudo-labeller), and its info-card text
(`dcops/dc_components.json`). The E8 camera pipeline comes from Nissan Pow's
[alif-mlek-yolo](https://github.com/npow/alif-mlek-yolo). Everything else was built at this event:
the teacher/student distillation, the board-camera capture and fine-tune, the ExecuTorch/Ethos-U85
export, the two-model E8 firmware and the voice workflow.

## What it does

- The MT9M114 camera streams a live preview to the LCD.
- Each frame, a MobileNetV2 classifier on the **Ethos-U85** labels the centre of the view.
- Labels are smoothed over frames and only shown above 50% confidence. When it is unsure it shows
  nothing instead of a flickering guess.
- An info card (title, summary, specs from `dc_components.json`) appears under the label.
- A keyword spotter on the same NPU listens all the time. Voice drives the workflow hands-free:

| Say | Action |
|---|---|
| `go` (or `yes` / `on`) | Latch the identified part and open its inspection checklist. The live label freezes. Only works when vision has a confident identification. |
| `go` / `on` / `right` while holding | Next knowledge page: specs, troubleshooting, LED status, maintenance. `left` goes back. |
| `yes` | Mark the selected item pass and advance. |
| `no` | Mark it fail and advance. On item 1 ("is this the right part?"), `no` rejects the identification and returns to scanning. |
| `up` / `down` | Move the cursor. |
| `stop` | Back to live scanning. |

- Vision decides *what* the part is. Voice confirms it and drives the checklist. A `no` on item 1 is
  a human-in-the-loop correction of the vision model.
- No network, no cloud, no images stored.

## How it meets the judging criteria

### Functionality & reliability (20)

- Full pipeline runs on the board in one HP firmware image (`build-fusion`): camera -> NPU
  classifier -> info card, plus mic -> NPU keyword spotter -> checklist.
- Board output matches the PC on the same captured frame: top-3 identical, e.g. background 66.4%
  on the board vs 66.4% on PC.
- Demo build shows 5 classes the board camera separates reliably, measured on held-out slides:
  79% of component frames labelled correctly, 7% wrong, the rest blank; 0% of background frames
  given a label.
- Confidence gate + EMA: a blank beats a wrong label.
- Known failure and fix: the camera hangs after a J-Link reset, so we power-cycle after flashing.
  If camera init fails, the firmware blinks LED1 instead of hanging silently.
- A flash guard refuses images that would overwrite the boot table (we bricked HP boot once and
  recovered it).

### Performance & efficiency (20)

| Metric (build-fusion, M55 HP @ 400 MHz) | Value |
|---|---|
| Vision NPU inference (MobileNetV2 160 px) | 7.9–8.0 ms |
| Vision preprocessing (M55) | 1.75 ms |
| Full frame | 255–287 ms (3.5–3.9 FPS); LCD footer 274 ms / 3.6 FPS |
| KWS MFCC (M55) / DS-CNN (NPU) | 3.3–4.0 ms / 0.34 ms, 2 strides/s |
| After removing per-frame UART logging (judged build) | **124 ms / 8.0 FPS** (was 274 ms / 3.6 FPS, 2.2x); NPU 7.9 ms; KWS MFCC 3.3 ms + NPU 0.3 ms |
| Power | `TODO(measure)`: not measured (needs a USB power meter or the power use case) |

- Frame time dropped from 274 ms to 124 ms (3.6 to 8.0 FPS) by removing per-frame UART logging; it stays the same whether or not a part is on screen. The NPU is about 6% of the frame; camera capture and LCD/LVGL dominate.
- First optimisation done: per-frame UART result/profiler printing removed from the classify path.
- Models are small: vision .pte 2,301 KB, KWS .pte 93 KB (79,116 params).
- int8 costs almost nothing: int8/float top-1 agreement 98.4% on calibration frames.

### Effective use of hardware (15)

- Both models run on the **Ethos-U85-256**: 100% of the graph ops; only quantize/dequantize at the
  graph edges run on the M55.
- The M55 HP does what does not map to the NPU: crop/resize, MFCC, softmax + EMA, UI.
- Camera over MIPI/CPI + DMA, PDM mic, LCD via LVGL, models and info cards in MRAM.
- MRAM budget managed by hand: image ends at 0x805473C4, guard at 0x80578000, ~195 KB headroom.
- Not used: the HE core (M55 + Ethos-U55). Moving KWS there for always-on low power is the next
  step. Not built.

### Multimodal & sensor integration (15)

- Two sensors, two models, one image: camera + vision classifier and PDM mic + keyword spotter run
  concurrently.
- Fusion is a gate, not a gimmick: `go` only works when vision is confident; `no` on item 1
  overrides vision.
- The KWS is trained on the firmware's exact MFCC and with the board's auto-gain emulated.

### Innovation & use-case value (10)

- Air-gapped data centers ban phones and cloud. A fixed-function, offline device fits that rule.
- Info cards plus a hands-free inspection checklist: the technician's hands stay on the hardware.
- Board-camera data collected with no network link: labels come from a clock-synced slideshow.

### Technical ambition & integration (10)

- Distillation chain: YOLOv8n-seg pseudo-labels -> ConvNeXt teachers -> MobileNetV2 student;
  AST -> DS-CNN for voice.
- On-device data loop: recorded the E8 camera's own model-input frames, fine-tuned, held-out board
  accuracy 35.5% -> 73.0%.
- Two ExecuTorch 1.1.0 programs in one image: we changed MLEK's `EtModel` from a static to a
  per-instance method.
- Found a Vela miscompile (BlazeFace output scrambled) and shipped a workaround.

### Engineering quality & demo (10)

- Reproducible: training scripts (`dcops/jobs/`), Colab notebooks (`dcops/colab/`), export
  scripts (`dcops/export/`), one-command demo build (`scripts/alif/build_dcops_demo.sh`).
- Ablations reported, honest limits stated (5 demo classes, no power number yet).
- Guarded flash script, recovery procedure written down.
- Rehearsed 2-minute live demo with fallbacks (`dcops/docs/DEMO_SCRIPT.md`).

## How we built it

**Training (all on Colab GPU; scripts in `dcops/jobs/`, notebooks in `dcops/colab/`)**

1. **Teacher, wave 1.** The DC-Ops YOLOv8n-seg detector pseudo-labels the DC-Ops images into 3,494
   component and background crops. A ConvNeXt-Small teacher reaches 69.4% (on auto labels).
2. **Teacher, wave 2.** 19,203 crops from 2,056 human-labelled images. Two ConvNeXt-Base teachers,
   out-of-fold, reach 97.2% on human-labelled val.
3. **Student `v2_r160_kd`.** MobileNetV2, 160 px, knowledge distillation (alpha 0.7, T 4),
   board-camera augmentation (low-res, blur, colour cast, wash-out, noise, glare, screen moire),
   EMA weights. 16,998 train / 2,205 val. 0.912 on human-labelled val, 0.913 on board-camera sim.
   Input is NHWC float [0,1]; normalisation is folded into the model.
4. **Board-camera fine-tune `v3_board_ft` (deployed).** We recorded the E8 camera's own model-input
   frames over J-Link while a phone showed a clock-synced slideshow. Labels come from the clock, so
   no network link was needed. 1,330 board frames (1,119 train / 211 test, split by slide) plus
   1,503 replay crops of human-labelled photos to prevent forgetting. 25 epochs, lr 3e-4, 60% board
   frames per batch. Held-out board accuracy (all classes): **35.5% -> 73.0%**.
5. **Keyword spotter `dscnn_m_agc`.** DS-CNN (128 ch x 4 blocks, 79,116 params) distilled from an
   Audio Spectrogram Transformer on Speech Commands v0.02. Trained on a torch copy of the
   firmware's exact MFCC (MLEK `Mfcc.cc`, 49x10) with the board's auto-gain emulated. 94.66% test
   (94.70% with board AGC), 12 classes incl. `_silence_` / `_unknown_`.

**Export (`dcops/export/export_student.py`, `export_kws.py`)**

torch.export -> PT2E with `EthosUQuantizer` (symmetric per-channel int8, vision calibrated on board
frames) -> `to_edge_transform_and_lower` with `EthosUPartitioner` -> TOSA -> Vela 4.5.0
(`ethos-u85-256`, `Ethos_U85_SRAM_MRAM`, `Shared_Sram`) -> `cortex_m` passes -> `.pte`.
ExecuTorch 1.1.0 is pinned to match MLEK.

**Firmware (`source/app/use_case/alif_object_detection`, build-fusion)**

We started from the Alif ML Embedded Evaluation Kit and Nissan Pow's DevKit-E8 camera pipeline.
We added an ExecuTorch path with the Ethos-U85 delegate, a classifier task with EMA and a
confidence gate, info cards, the PDM audio + MFCC + KWS path, and the voice state machine.
Running two models needed one change in MLEK: `EtModel` went from a static method to a
per-instance method (`source/lib/mlek/fwk/executorch/EtModel.cc/.hpp`).

## Architecture and workload placement

```
MT9M114 --MIPI/CPI+DMA--> SRAM frame --M55 HP: crop, resize--> 160x160 RGB
   --M55 HP: float NHWC [0,1]--> [int8 quantize, M55] --Ethos-U85-256: MobileNetV2 (all ops)--> [dequantize, M55]
   --M55 HP: softmax + EMA (alpha 0.3), gate 0.5--> info card (dc_components.json) --> LVGL --> LCD
PDM mic --background--> M55 HP: MFCC 49x10 --Ethos-U85: DS-CNN, every 0.5 s--> threshold 0.5 --> voice UI
```

| Workload | Runs on | Cost |
|---|---|---|
| Camera capture | MIPI CSI + CPI + DMA into SRAM | dominates the frame with LCD/LVGL |
| Crop / resize to 160x160 RGB, float NHWC | Cortex-M55 HP | 1.75 ms |
| int8 quantize / dequantize | Cortex-M55 HP | only CPU ops in the graph |
| Vision classifier (MobileNetV2) | **Ethos-U85-256** | 7.9–8.0 ms |
| Softmax, EMA, confidence gate | Cortex-M55 HP | trivial |
| UI: preview, label, info card, checklist | M55 HP + display controller | LVGL |
| Audio capture | PDM mic, background | |
| MFCC 49x10 | Cortex-M55 HP | 3.3–4.0 ms |
| Keyword spotter (DS-CNN) | **Ethos-U85-256** | 0.34 ms, 2 per second |
| HE core (M55 + Ethos-U55) | unused | next step: always-on KWS |

Why this split: the NPU takes the dense convolutions. The M55 keeps camera geometry, MFCC,
softmax and UI, plus the int8 boundary. The classifier is small enough to fit the HP MRAM
partition with ~195 KB to spare and keeps NPU time at about 6% of the frame.

## Results

| Model | Metric | Value |
|---|---|---|
| Teacher wave 1 (ConvNeXt-Small) | val, auto labels | 69.4% |
| Teacher wave 2 (2x ConvNeXt-Base) | human-labelled val | 97.2% |
| Student `v2_r160_kd` | human-labelled val / board-camera sim | 0.912 / 0.913 |
| Ablation: no board-camera aug | board-camera sim | 0.758 |
| Ablation: no KD | board-camera sim | 0.804 |
| Resolution sweep (wave 1) | 128 / 160 / 192 px | 0.581 / 0.653 / 0.632 |
| `v3_board_ft` (deployed) | held-out board frames, all classes | 35.5% -> 73.0% |
| `v4_demo_ft`: same data, trained over the 6 shown classes | held-out demo-class frames, 6-way top-1 | 84.8% (v3: 83.5%) |
| Demo build, 5 classes + bg, cutoff 0.6 | held-out board frames | 79% correct, 7% wrong, rest blank; bg labelled 0% |
| Demo build, cutoff 0.35 | held-out board frames | 78% correct, 11% wrong |
| Demo build, clean held-out photos | dataset val split, 5 classes, PC | 25/28 |
| KWS `dscnn_m_agc` | test / with board AGC | 94.66% / 94.70% |
| KWS `dscnn_s_agc` (smaller) | test, 23,180 params, 34 KB | 93.94% |
| Vision int8 vs float | top-1 agreement, calibration frames | 98.4% |

We use cutoff 0.6 in the demo: a blank beats a wrong label.

**Why 5 demo classes.** The model has 16 component classes + background. Through the board camera,
compute tray, cooling manifold and DPU were confused with each other, and 8 classes had no
board-camera data at all. The firmware option `CLS_ALLOWED_MASK` limits the shown classes; the demo
build uses `0x13250` = power shelf, network port, fan, power connector, drive bay (+ background).
These are the 5 classes the board camera separates reliably, measured on held-out slides.

## Challenges we ran into

- **We bricked HP boot, then recovered it.** Alif's ATOC boot table sits at 0x8057BC00, at the top
  of the HP partition. An oversized image links fine, flashes fine, and overwrites it. We restored
  an MRAM backup through the HE core (J-Link device `AE822FA0E5597LS0_M55_HE`).
  `scripts/alif/flash_e8_hp.sh` now refuses any image ending above 0x80578000.
- **Vela miscompile.** ExecuTorch 1.1 lowers conv -> permute -> reshape into cancelling transposes
  that Vela miscompiles; BlazeFace output came out scrambled. Workaround: keep that tail on the CPU,
  or use an NPU-friendly head (AvgPool + reshape + Linear on NHWC). Written up in
  `docs/blazeface-vela-bug.md`.
- **Domain gap.** The board camera is low-res, bluish and washed out. Before the board fine-tune,
  held-out board accuracy was 35.5%. We built a no-network capture rig (clock-synced slideshow on a
  phone) and fine-tuned on the board's own frames.
- **Verify the framing.** Our first capture had the camera pointed at the ceiling: 0% -> 2.4%. We
  discarded it. The first good run went 30.7% -> 64.6%; adding a second deck reached 73.0%.
- **The camera hangs after a J-Link reset.** Every flash ends with a power-cycle (unplug PRG USB
  about 10 s).
- **Two models, one runtime.** MLEK's `EtModel` assumed one program per image; we made it
  per-instance.

## Accomplishments we're proud of

- Camera and voice, two ExecuTorch models, 100% of graph ops on the Ethos-U85, one firmware image.
- Board output identical to the PC on the same frame (66.4% vs 66.4%).
- A labelled board-camera dataset collected with no network link, and a fine-tune that doubled
  held-out board accuracy (35.5% -> 73.0%).
- Ablations that show both distillation and board-camera augmentation matter.
- A UI that prefers a blank to a wrong label, and a voice `no` that corrects the vision model.
- A flash guard that turns a brick into a refused command.

## What we learned

- On this board the NPU is not the bottleneck: about 8 ms of a ~274 ms frame. Camera capture,
  LCD/LVGL and logging take the rest.
- Simulated camera augmentation helps (0.758 -> 0.913 on board-sim), but real board frames help
  more. Collect them early, and check the framing first.
- MRAM partitions and boot tables matter more than total MRAM size.
- When train and deploy features match exactly (MFCC, NHWC [0,1] input), there are fewer silent
  accuracy losses.

## What's next

- Move KWS to the HE core (M55 + Ethos-U55) for always-on low power; wake the HP core on a keyword.
- Measure power, then lift the frame rate (camera and LVGL path, not the model).
- More board-camera data to bring back the classes the demo mask hides.
- Component-level identification (resistor, capacitor, diode). We evaluated the Roboflow
  "Components" dataset (CC BY 4.0, 672 PCB images, 23 classes). Resistors are ~1.3% of image width
  (~8 px), so ~2 px at the 160 px model input. That needs a detector at higher resolution or tiled
  inference. Larger parts (IC, connector, electrolytic capacitor) would fit the current classifier.

## Datasets and licenses

- Landing page and slides: https://abhijitbetigeri.github.io/executorch-alife8/
- Trained models (deployed .pte + PyTorch weights, ablations, teachers): https://huggingface.co/abhijitbetigeri/dcops-e8
- Dataset: dcops/data/ in the GitHub repo (CC BY 4.0 sources listed there)
- Speech Commands v0.02 (KWS training).
- Roboflow "Components" (CC BY 4.0): evaluated for future work only, not used in the model.

## Built with

ExecuTorch 1.1.0 · Arm Ethos-U85 / Vela 4.5.0 · Arm ML Embedded Evaluation Kit · Alif Ensemble E8
DevKit · PyTorch · timm (ConvNeXt) · torchvision (MobileNetV2) · Ultralytics YOLOv8 · Hugging Face
Transformers (AST) · LVGL · GCC 14.3.1 · SEGGER J-Link · Google Colab

## Team

Abhijit Betigeri and Nissan Pow. Camera pipeline: Nissan Pow.
