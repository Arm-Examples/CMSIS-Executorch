# Board demo (physical Alif E8 run)

Source of the physical-board results in the main README (77.1 % on 70 trials) and of the `cmsis-executorch.axf` attached to release `v1.0-submission`.

| Folder | What it is |
|---|---|
| [`firmware/multisubject/`](firmware/multisubject/) | Board application (`app_main.cpp`, `protocol.hpp`) for the generic model (`.pte` sha256 `6caa8a98…`). Receives one preprocessed int8 window (22 × 256 = 5,632 values) per request over UART (COBS framing, CRC32), runs ExecuTorch on the Ethos-U85, replies with logits. |
| [`firmware/streaming/`](firmware/streaming/) | The earlier subject-specific streaming firmware and the UART driver (`uart_stream.c`) that the multi-subject build reuses. |
| [`tools/multisubject_trials/`](tools/multisubject_trials/) | Builds the 1,080-trial manifest and int8 windows, generates the CMSIS project (`prepare_firmware.py`), compares FVP output with the TOSA reference (`compare_fvp.py`), validation reports. |
| [`frontend/`](frontend/) | The demo UI shown in the video: Chrome Web Serial to the board, cue, scalp map, 22 EEG channels, running accuracy, session summary. |

**How this differs from the main firmware in [`../firmware/`](../firmware/):** here the laptop does the preprocessing (filtering, alignment, int8 quantization) and sends finished windows; the board runs inference only. The main firmware does the preprocessing on the Cortex-M55 sample by sample (Figure 2) and is verified on the Corstone-320 FVP; it has not run on the board yet.

Build: `python tools/multisubject_trials/prepare_firmware.py` creates the CMSIS project; build it for target `DevKit-E8` with CMSIS-Toolbox and flash with SEGGER J-Link (details in [`firmware/multisubject/README.md`](firmware/multisubject/README.md)). Paths inside these files refer to the original Reproduce_Baselines checkout.

## Source

Copied from `ArmExecuTorchEEG/Reproduce_Baselines`, branch `demo/edgebci-multisubject-working`, commit `19dc889` (2026-10-18), plus the `frontend/` changes of branch `feature/final-ui-design` commit `08c93bc` (final demo design and session summaries), by Philippine Chauvet, with the UI by Cindy Ku. Left out: the processed EEG excerpts `frontend/recorded-eeg/model2-eeg-data.js` (dataset content; regenerate with `frontend/tools/export-recorded-eeg.py`) and the design PDF in `frontend/design-reference/`.
