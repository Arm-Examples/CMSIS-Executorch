# Demo script (3 minutes, judged)

Real-time left/right motor-imagery decoding on the Alif Ensemble E8: held-out BCI IV 2a trials stream from the laptop at 250 Hz over UART, the Cortex-M55 filters and aligns them, the Ethos-U85 runs TinyEEGNet int8, the on-board IMU gates motion. No EEG headset (team decision 2026-10-16): the replay is the EEG source.

## T−15 min: setup checklist

| Check | Command / action | Expect |
|---|---|---|
| Firmware flashed (DevKit-E8), SW4 → **UART4**, USB-C on PRG USB | VS Code: CMSIS → Run | — |
| Port name | `ls /dev/tty.usbmodem*` | one or two ports; the UART4 one prints `READY` |
| Nothing else holds the port | close VS Code serial monitor, `screen`; `lsof /dev/tty.usbmodem*` is empty | — |
| Replay data | `ls host/data/A0{3,8}_session2.npz` (else `python host/export_replay.py --subject all`) | files present |
| Host tests | `python -m pytest host/tests -q` | all pass |
| Dry run without board | `python host/bridge.py --subject 3 --board sim --trials 33-37` | `accuracy 5/5, match offline 5/5` |

Two terminals side by side, the UI window on the projector, the board in reach on the table.

## Run of show

| Time | Do | Say |
|---|---|---|
| 0:00 | Terminal 1: `python host/demo_ui.py --listen 5555` | "Stroke-rehab BCI: the user imagines moving the left or right hand; the board decides which, on the NPU." |
| 0:15 | Terminal 2: `python host/bridge.py --source replay --subject 3 --session 2 --board /dev/tty.usbmodemXXXX --trials 24-143`. Press the board's reset button once so the boot line shows. | Point at `[board] READY n_ch=22 model=tinyeegnet_int8 pte_bytes=11456 … baud=921600`: "11 KB model, all 35 compute ops on the Ethos-U85." |
| 0:30 | Let 3–4 trials run (one every ~8 s after the 10 s lead-in). LED green = left, blue = right. | "These are recorded trials of subject A03, session 2, never seen in training, sent sample by sample at 250 Hz exactly like a headset would. The M55 does common-average reference, the 7–30 Hz band-pass and decimation per sample, then per-subject Euclidean alignment and int8 quantization; the NPU does the whole network." Read `lat` from the UI: "end to end from the last sample to the answer". |
| 1:15 | **Motion gate.** When the UI shows a new cue, pick the board up and shake it for ~3 s (the window closes 2.5 s after the cue; the gate looks at the last 500 ms). | UI shows **GATED** in red, LED red, `motion 1` in the bridge line. "The accelerometer sees movement, so we refuse to trust a motion-corrupted window instead of guessing." Put the board down: the next trial classifies normally. |
| 1:50 | Ctrl-C the bridge. `python host/bridge.py --subject 8 --board /dev/tty.usbmodemXXXX --trials 24-143` (UI keeps running and resets its counters). | "One model serves all nine subjects; switching subject only switches the 22×22 alignment matrix sent by `SET_SUBJECT`." |
| 2:20 | Point at the UI footer (accuracy, `match offline`). | "The board reproduces the offline int8 pipeline: `results/replay_accuracy.md` has all nine subjects; A03 99 %, A08 94 %, mean 73 % for a 3.4 k-parameter model trained on other datasets." |
| 2:45 | Close: `results/board_metrics.md` on screen if recorded. | "Laptop only replays and displays; filtering, alignment, quantization, inference and gating all run on the E8." |

Recording metrics during a rehearsal (third terminal, same UDP feed): `python host/measure.py --listen 5555 --trials 40 --out results/board_metrics.md`.

## Fallbacks

| Symptom | Action |
|---|---|
| No board / board dead / flashing fails | Same commands with `--board sim`. The sim board runs the same frame parser and Python reference pipeline in-process and answers with the offline int8 logits (`--sim-model int8` runs the int8 model itself, needs the Reproduce_Baselines env). Say so: "this is the host reference; the board run is in the backup video `docs/video/`." |
| `cannot open /dev/tty…: Resource busy` | Another program holds the port: quit the VS Code serial monitor / `screen` (`lsof /dev/tty.usbmodem*`), unplug-replug, rerun. Only `bridge.py` may open the port; UI and measure read UDP. |
| Bridge runs but no `trial …` lines | Check SW4 is on UART4, baud 921600 (`--baud`), press reset and look for `READY` / `#` lines in the bridge output. `# frame_err n` growing = wrong baud or noisy cable. |
| UI window does not open (no display, matplotlib backend error) | `python host/demo_ui.py --listen 5555 --headless` prints the same per-trial line; or just show the bridge terminal. |
| UDP port 5555 taken | `--publish 127.0.0.1:5556` on the bridge and `--listen 5556` on the UI. |
| Replay file missing | `python host/export_replay.py --subject 3` (needs the Reproduce_Baselines env and `~/mne_data`); `--board sim` needs the same file. |
| A trial classifies wrong | Expected: A03 is 119/120 offline; harder subjects (A02, A06, A09) are near 60 %. Point at `match offline`: the board agrees with the offline pipeline, the error is the model's. |
| Gate does not trigger | Shake harder and longer (threshold 0.25 g deviation, last 500 ms before the inference). If the boot log shows `# imu` failure, say the IMU init failed and move on; `motion` stays 0. |
