# Deploy runbook: raw dataset signals → Alif E8 → prediction

Short, complete, in order. For a person or an agent with a fresh clone. Model: TinyEEGNet wide32, release `v0.5-distill` (`.pte` 11,456 B, sha256 `6caa8a98…`). Nothing on the laptop filters or normalises; the board does every step (see the README diagram).

## 0. Get the model

Pick one. All three give the same bytes.

- Already in the clone: `firmware/ai_layer/model_pte.c` embeds the `.pte`; `firmware/model/params.json` is committed. Build works offline.
- GitHub release (needs `gh` logged in to the org): `./tools/fetch_model.sh` → `firmware/model/{tinyeegnet_int8.pte, test_vectors.npz, preprocess_params.json, tinyeegnet_float.pt, calibration.npz, student.json}`, sha256 checked, `params.json` written.
- Google Drive (no GitHub access): folder `ArmExecuTorchEEG/Reproduce_Baselines/v0.5-distill/bundle/` at https://drive.google.com/drive/folders/1Q9bduq4tFBu0q5QRQ6d16NgSJGH6ea-w → copy the six files into `firmware/model/`, then `cp firmware/model/preprocess_params.json firmware/model/params.json`.

Then `python3 tools/gen_params_h.py --check` (regenerates `firmware/src/params_gen.h` if `params.json` changed).

## 1. Laptop tools

- Python: `Reproduce_Baselines/.venv/bin/python` (sibling clone) has everything; or `pip install -r host/requirements.txt` (numpy, pyserial, matplotlib; plus mne, moabb, scipy for the one-time export).
- Firmware: VS Code + Keil Studio Pack (CMSIS-Toolbox 2.15, Arm Compiler 6), Alif SETOOLS ≥ 1.110, SEGGER J-Link ≥ 8.42. Command line: `source ~/.vcpkg/vcpkg-init && (cd firmware && vcpkg-shell activate)`.
- Dataset: BCI IV 2a (`BNCI2014_001`) in `~/mne_data` (MOABB download; 747 MB).

## 2. Prepare the raw replay file (once per subject)

```bash
export MOABB_DOWNLOAD_PROVIDER=upstream MNE_DATA=~/mne_data MNE_DONTWRITE_HOME=true
python host/export_replay.py --subject 3        # or --subject all
```
- Writes `host/data/A03_session2.npz`: raw digital samples (22 EEG channels, 250 Hz, int16 counts = µV × 20.48, lossless), cue sample indices, labels. No filtering.
- Session 2, trials 25–144 are the demo/test trials; trials 1–24 were used offline for the subject's alignment matrix, never with labels.

## 3. Build the firmware

```bash
cbuild firmware/edge_eeg.csolution.yml --active DevKit-E8 --packs     # board image
```
- Output: `firmware/out/edge_eeg/DevKit-E8/Debug/edge_eeg.axf`. Prebuilt copy: release `v0.9-edge-fvp` asset `edge_eeg_Debug_DevKit-E8.axf`.
- Expected: `1 succeeded, 0 failed`; only two external `-Wnan-infinity-disabled` warnings from ExecuTorch.

## 4. Flash and boot the board

1. USB-C on **PRG USB**; jumpers JP5 1-2, JP7 3-4; SW4 → **SEUART**.
2. First time only: VS Code Terminal > Run Task > *Alif: Install M55_HP debug stubs* (needs `alif.setools.root`; stub in `firmware/.alif/`).
3. SW4 → **UART4**. VS Code CMSIS view > target-type `DevKit-E8` > Run (J-Link, SWD 4 MHz, loads the `.axf` into MRAM).
4. Serial console (`screen /dev/tty.usbmodemXXXX 921600` or VS Code) must show:
   `READY n_ch=22 model=tinyeegnet_int8 pte_bytes=11456 fs=250 fs_model=125 win=256 baud=921600`
   then `# imu ok who_am_i=0x67` (or `# imu init failed: …`, then the gate is off, predictions still run).
5. Close the console before step 5 (only `bridge.py` may own the port).

## 5. Stream the raw signals and read predictions

```bash
python host/bridge.py --source replay --subject 3 --session 2 --board /dev/tty.usbmodemXXXX
python host/demo_ui.py --listen 5555            # second terminal (or --headless)
python host/measure.py --listen 5555 --trials 120 --out results/board_metrics.md   # third, optional
```
- `bridge.py` sends RESET, SET_SUBJECT 3, then one raw 250 Hz sample per UART frame in real time (10 s lead-in, then trials 25–144), a CUE control frame before each trial. It prints one line per trial and republishes the board's JSON over UDP 5555.
- Board output per trial: `{"t":…,"seq":…,"trial":i,"subject":3,"cls":0|1|-1,"raw_cls":…,"conf":0.xx,"logits":[…],"lat_us":…,"motion":0|1,"label":…}`. `cls` −1 = gated by the IMU (shake the board to show it; red LED).
- Expected for A03: 119 of 120 correct; every trial's `raw_cls` must equal `host/reference/expected_board_predictions.json` (the board pipeline is bit-exact with `host/pipeline_ref.py`). Other subjects: `results/replay_accuracy.md` (mean 74.4 %).
- Switch subject: restart `bridge.py --subject N`; same firmware, the board picks `ea_matrices[N]` from `params.json`.

## 6. No board? Same pipeline on the emulator or in software

- FVP (Corstone-320, same Cortex-M55 + Ethos-U85): `cbuild firmware/edge_eeg.csolution.yml --context edge_eeg.SelfTest+SSE-320-U85 --packs`, then the Docker command in `results/fvp_selftest.md`; expect `Test_result: PASS` (embedded raw stream, 5 trials, 0 int8 mismatches).
- Software board: `python host/bridge.py --source replay --subject 3 --board sim` runs the identical pipeline in Python and answers with the offline model's predictions; UI and measure work unchanged.

## 7. Record and tag

- `results/board_metrics.md` from `measure.py` (accuracy, agreement, `lat_us` mean/p95/max, gated count).
- Flip the repo to public, attach `edge_eeg_Debug_DevKit-E8.axf` and `tinyeegnet_int8.pte`, tag `v1.0-submission`.

## If something fails

| Symptom | Check |
|---|---|
| No `READY` | SW4 on UART4 (not SEUART); console at 921600; debug stubs installed once |
| `READY` but no JSON after a trial | bridge sent SET_SUBJECT (1–9) before the first CUE; `# frame_err` counter in the periodic `#` line (baud, cable) |
| Predictions differ from `expected_board_predictions.json` | stream started mid-recording with a different lead-in (filter state); `params.json` sha256 must be `3d34f632…` |
| `# imu init failed` | demo continues without the gate; I3C wiring/pack config, see `firmware/board/DevKit-E8/README.md` |
| Serial port busy | close the console or the UI; only `bridge.py` opens the port |
