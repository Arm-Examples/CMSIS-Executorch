# Sentinel: details, evidence and test records

The short overview is in [../README.md](../README.md). This page keeps the
detailed architecture, measured performance, build modes, demo procedure,
full test records and limitations.

Sentinel scores the DevKit-E8's PDM microphones and BMI323 accelerometer
against a baseline learned from a machine running normally. In the default
status-only build the LCD shows text only, landscape and left to right:
SENTINEL at the top, the state (IDLE, LEARNING, OK, CHECK or ALERT) large
below it, the selected profile, the learning progress, and one line each for
Sound and Vibration.
The status-only build loads no model and generates no faces; the pico-faces
example stays in the repository and runs when `APP_STATUS_ONLY` is not
defined.

Sentinel detects a change in a machine's operating state against its own
learned normal. It is not fault diagnosis, and its scoring is statistical
(z-scores on the CPU), not an ExecuTorch model or NPU inference.

## Evidence at a glance

| Claim | Status | Evidence |
|---|---|---|
| Learn a normal state, store it in MRAM | Demonstrated | `SN PROFILE ... saved in MRAM` on every learn; LCD shows no save warning |
| Baseline survives power-off | Demonstrated | Several full power cycles on 2026-10-17/18: boot report `saved baseline available`, then VERIFYING -> OK |
| Fan medium -> high gives CHECK, back to medium gives OK | Demonstrated repeatedly on the current fan (and in earlier runs with a fan whose identity was not recorded) | CHECK 2 s after the sound changes in every logged cycle (at the final location the first cycle went on to ALERT because the board had moved); operator confirmed the LCD |
| Audio capture keeps up | Demonstrated | 0 FIFO errors, 0 ring-full stops in every status-only log |
| Fan vibration detection | **Not demonstrated** | BMI323 driver gives about 50 Hz samples; fan speeds stay at the noise floor |
| Phone vibration detection | **Inconclusive** | Sound changed during both trials; vibration flagged in one of two |
| Fault diagnosis (blade damage, imbalance, bearings) | **Not tested** | No fault was ever introduced |
| Power consumption | **Not measured** | No instrument used |

## Performance and resources (measured, status-only build)

| Quantity | Value | Source |
|---|---|---|
| Detection latency | 2 s from the first 1 s window above the threshold to CHECK (3-window rule); the first window itself can take up to 1 s | UART logs, `tools/sentinel_log_report.py` |
| Recovery latency | about 2-3 s after the sound returns (3 windows below 2.5) | 2026-10-17 logs; not logged at the final location (serial link dropped) |
| CPU per 100 ms audio block | median 88-130 us, worst 192 us (about 0.19 % of the 100 ms interval; not total CPU utilization) | `blk_us` in the UART log |
| CPU per 1 s window | median 2 us; about 21 ms once, when a learned baseline is written to MRAM and verified | `win_us` |
| Audio | 2 x 16 kHz PDM, 4 x 100 ms ring (25.6 kB), interrupt driven | `mic_probe.c` |
| Accelerometer | 50-52 new samples/s, about 99 % of reads new | `imu=` in the UART log |
| Memory | DTCM 64 kB of 864 kB (`.data`/`.bss`), 96 kB heap, 32 kB stack; SRAM0 1.15 MB (LCD frame buffer only); MRAM 175 kB code + 256 B profiles; ITCM 24 B (MRAM write routine) | linker map |
| Power | not measured | |

## Why the CPU and not the NPU

The workload is 32 k audio samples/s (2 x 16 kHz) of feature arithmetic,
50 accelerometer reads/s and a few comparisons per second. The
worst measured audio-block processing was 192 microseconds per 100 ms
block (about 0.19 % of that interval); this is not total CPU utilization,
and power was not measured. A model on the
Ethos-U85 would add latency and memory without improving this scoring, and
there is no labelled fault data to train one. The NPU (and the pico-faces
ExecuTorch path) stays in the repository for workloads that need it, such as
a learned audio or image model.

## Architecture

| Part | File | Work and timing |
|------|------|-----------------|
| Microphones | `board/DevKit-E8/mic_probe.c` | PDM channels 4/5, 16 kHz, 16-bit, `ARM_PDM_MODE_AUDIOFREQ_16K_DECM_64`, channel settings from the pack's `demo_pdm.c`. The PDM callback restarts the capture into a ring of 4 x 100 ms static blocks (25.6 kB), as the pack's `vstream_audio_in.c` does; the main loop consumes the blocks. 0 FIFO errors and 0 ring-full stops in the status-only build, in idle runs and (in the face-demo mode) in 65 s of continuous face generation. A 49 kB viewer frame on the console still costs one ~4.5 s ring-full stop, so do not run `view_serial.py` with Sentinel. |
| Accelerometer | `board/DevKit-E8/imu_probe.c` | BMI323 on I3C through the pack driver, read every 20 ms from the main loop (~120 us per read). The installed driver writes a fixed configuration and offers no rate or filter control; measured rate 50-52 new samples/s (`APP_IMU_VIBRATION_PROBE`, `imu_vib_probe.c`). |
| Display | `board/DevKit-E8/display.c`, `sentinel.c` | `board_display_line()` and `board_display_rect()` draw 5 x 7 text (A-Z, 0-9, `: / - .`) scaled up, in an 800 x 480 landscape view rotated into the panel's 480 x 800 RGB888 frame buffer (`APP_DISPLAY_LANDSCAPE`). Layout: title band, state band in the state colour, info line with a 30-step progress bar while learning, `SOUND:` and `VIBRATION:` lines reading MATCHES BASELINE (green), CHANGED (amber), LEARNING or NOT LEARNED. Each part is redrawn only when its text changes, from the main loop. |
| Scoring | `board/DevKit-E8/sentinel.c` | 1 s windows (10 microphone blocks). Features: f0/f1 log power of channel 4/5, f2 log high-frequency ratio of channel 4 (first difference / signal; no CMSIS-DSP or FFT is installed, so no band energies), f3 log variance of the acceleration magnitude. Baseline = mean/std over 30 windows with std floors. A sensor is anomalous after 3 windows with \|z\| > 4.0 and normal after 3 windows with \|z\| < 2.5; 0/1/2 anomalous sensors = OK/CHECK/ALERT. About 68 us (worst 192 us) per 100 ms block, 1-6 us per window, all in the main loop. |
| Profiles | `board/DevKit-E8/sentinel_profile_store.c` | Two separately learned baselines, stored as checksummed alternating records in the final 256 bytes of reserved user MRAM. The 16-byte write routine runs from ITCM because the pack warns against executing from MRAM while writing it. A partial or invalid record is rejected; a new profile is verified after saving. The first slot is FAN; the second is PHONE (labels only; same IDs and layout as before). |

Compile-time modes in `board/DevKit-E8/Board-U85.clayer.yml`: the default is
`APP_SENTINEL` + `APP_STATUS_ONLY` + `APP_DISPLAY_LANDSCAPE` (status-only text
demo, no model, no faces). Without `APP_STATUS_ONLY` the pico-faces demo runs
as well, and its faces would be drawn over part of the status text (not
tested with the text layout). Instead of
`APP_SENTINEL` (and `APP_STATUS_ONLY`): `APP_IMU_PROBE` + `APP_MIC_PROBE` for
raw sensor prints, or `APP_IMU_VIBRATION_PROBE` for the accelerometer rate
diagnostic (one at a time). `DISPLAY_LANDSCAPE_FLIP: 1` turns the landscape
view the other way round. After switching modes build with `--rebuild`: an
incremental build can keep objects compiled with the previous define and
fail to link.

## Build, flash, run

```
cbuild cmsis-executorch.csolution.yml --active DevKit-E8
# flash (the "CMSIS Load" task does the same):
printf 'Connect\nLoadFile "out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.hex"\nExit\n' > .cmd.jlink
JLinkExe -device AE822FA0E5597LS0_M55_HP -if swd -speed 4000 -commandfile .cmd.jlink
```

The load leaves the core halted: press reset (or J-Link `r`, `g`). Console:
SW4 = UART4, 115200 8N1. After boot Sentinel prints one `SN t=...` line per
second (state, z-scores, `!` = sensor anomalous, features, IMU new/read,
block and window time in us, FIFO errors, ring-full stops). SW2 **left**
selects FAN; **right** selects PHONE; **up** learns/relearns the selected
normal state (30 s). Push the joystick firmly; a light press may not register.
On boot the board waits for a selection. If that slot has a saved baseline,
selection checks three fresh windows (blue VERIFYING) before showing OK or
CHECK; if empty, it stays
IDLE until learned. Changing slots clears the partial sensor window and alarm
history. Do not interpret OK before the appliance is running at the same
normal condition and in the same board position as it was during learning.

After a successful learn the UART must say `SN PROFILE ... saved in MRAM`.
`WARNING: PROFILE NOT SAVED` on the LCD means the current RAM baseline works
only until power-off; inspect the UART storage error before relying on it.
Power-cycle persistence has been verified on the board several times
(2026-10-17/18). Flashing the firmware HEX with J-Link `LoadFile` should leave the profiles
intact: the image ends at 0x8022AE1F, and J-Link reported rewriting only the
16 kB units the image covers. This was not checked separately by reading the
profiles back after a firmware-only flash. **Never
write the profile area with J-Link:** J-Link programs this MRAM in 16 kB
units (0x80574000-0x80577FFF) and fills everything outside the written file
with 0xFF. A 128-byte write erased the FAN records this way on 2026-10-18.
Learn on the board instead; the firmware writes only its own 16-byte rows.
The gyroscope and chip temperature are not yet included in the score.

The LCD after boot (hold the board landscape): SENTINEL, a large grey IDLE,
`SW2 LEFT FAN RIGHT PHONE`, and `SOUND: NOT LEARNED`,
`VIBRATION: NOT LEARNED`.

## Safe demo: fan speed change (microphone)

1. Fix the board and fan in marked positions, guard on, nothing touching the blades.
2. Press SW2 left to select FAN. If it is empty or the setup changed, let the
   fan run steadily at medium for about 10 s, room quiet; press SW2 up. The LCD
   shows a large blue LEARNING, `BASELINE n/30 S` and a filling bar, then a
   large green OK, `NO CHANGE FROM BASELINE` and both sensor lines
   `MATCHES BASELINE` in green. Learn again if the fan or the room was not
   steady: a baseline learned during changing sound makes later scores
   meaningless (see the status-only result below).
3. Leave medium for 45 s (expect OK), switch to high (expect a large amber
   CHECK, `ONE SENSOR CHANGED`, `SOUND: CHANGED` in amber), return to the
   same medium setting (expect green OK).

### Result: final demo location (2026-10-18, operator present)

Fan oscillation off, board beside the fan out of its airflow, positions
marked, USB cable taped down. Raw logs and a summary are kept with the
project backup.

- First cycle medium -> high: ALERT (sound and vibration), because the board
  had moved; it is not counted as a sound-only result.
- Two cycles after securing the board: high gave sound-only amber CHECK
  (`SOUND: CHANGED`) about 2 s after the first abnormal sound window.
- Recovery to green OK on medium was confirmed on screen, but its exact time
  was not logged: the laptop's USB serial link dropped at every
  high -> medium switch, without the board resetting.
- After a power cycle, FAN reloaded (`checking saved baseline`, VERIFYING,
  OK).
- False alarms at medium, all cleared by themselves:
  - talking near the board: short CHECKs on sound only;
  - a bump near the board: CHECK on sound and vibration together.
- One SW2 press sometimes registered two or three times (contact bounce);
  each extra press only restarts the 30-window learn.
- The board was moved once after learning. Afterwards the baseline did not
  match even with the board put back by hand, so FAN was learned again.
- 0 FIFO errors, 0 ring-full stops.

### Result: status-only text demo, current fan (2026-10-17, operator present)

Same current fan as the result below, but moved farther from the board since
then, so the baseline differs. The operator confirmed every LCD screen
described above: IDLE layout, LEARNING with counter and bar, OK, CHECK with
`SOUND: CHANGED` and `VIBRATION: MATCHES BASELINE`, and OK again.

- First learning attempt: the sound during learning was louder and less
  steady than the medium that followed (f0 1.35 +/- 0.29 against 0.63), so
  high (f0 1.08) stayed inside the baseline and gave no CHECK. Re-learned
  with the fan settled at medium: f0 0.519 +/- 0.072, f1 -0.282 +/- 0.103,
  f2 -0.376 +/- 0.060, f3 0.037 +/- 0.100 (floor).
- Medium 45 s: OK throughout, mic score at most 3.3.
- High: the mic score jumped to 7.1 and the sound was flagged 2 s later;
  CHECK held on high (mic scores about 5). The vibration score rose to 17
  while the speed control was handled, which gave CHECK just before and
  2 ALERT windows just after the sound change; that is handling, not fan
  vibration detection.
- Back to medium: OK 2 s after the sound dropped; 26 windows OK after.
- 0 FIFO errors, 0 ring-full stops, about 99.6 % new accelerometer samples,
  worst 192 us per microphone block, no unexpected restart in 10 minutes.

### Result: current fan (validated 2026-10-17, operator present)

Current fan, fixed with the board in marked positions, guard on, only the
speed control touched (operator report). The operator confirmed on the LCD:
grey IDLE at boot, blue LEARNING then green OK after SW2 up, amber CHECK on
high (both times), green OK back on medium (both times).

- Baseline (30 windows, about 30 s): f0 1.991 +/- 0.148, f1 0.690 +/- 0.093,
  f2 -1.038 +/- 0.052, f3 0.882 +/- 0.189.
- Medium before the first change (86 windows, including the 45 s
  observation): OK throughout, mic score at most 1.8.
- High, twice: the mic score jumps in one window (9.4 and 10.9) and CHECK
  follows 2 s later (the 3-window rule); mic scores about 9-11 while on high,
  f1 up about 1 decade. The operator's confirmations arrived 7 s and 7 s
  after the jump, so the state had already changed when the change was
  confirmed; latency is measured from the jump in the sound.
- Back to medium, twice: the mic score drops in one window and OK follows
  2 s later (operator confirmations 6 s and 56 s after the drop).
- Medium between and after the cycles: OK, mic score at most 2.5; no
  microphone false alarm.
- Vibration: never flagged during the speed changes (vib score at most 5.5,
  never more than 2 windows in a row above 4). In the last 15 s of the run f3 rose to about 2.0
  and vibration alone gave 5 CHECK windows while the microphone stayed
  normal; the operator reported no handling, so the cause is unknown. This is
  not fan vibration detection.
- 0 FIFO errors, 0 ring-full stops, about 99 % new accelerometer samples,
  worst 183 us per microphone block.

### Historical runs (earlier setup, fan identity unknown)

Which fan was used in these runs was not recorded; they must not be carried
over to the current fan. Times are from the operator prompt, so they include
the operator and the fan changing speed.

- Fixed-mount run (baseline f0 = 2.50): CHECK 17 s and 22 s after the prompt
  to go to high (about 4 s after the first window above the threshold); OK
  10 s and 12 s after the prompt to return to medium; no false alarm in the
  medium phases.
- Final demo (baseline f0 = 1.61): CHECK 13 s and 17 s after the prompt to go
  to high (2 s and 4 s after the first window above the threshold); no false
  alarm in the medium phases. After the second return to medium, OK after
  10 s. After the first return, the microphone features stayed at their
  high-speed level and the state stayed CHECK for about 77 s before it went
  to OK; the cause was not established (the fan's control position was not
  recorded). In that period 15 windows were ALERT because the vibration score
  rose while the setup was being handled; that is not a two-sensor fan
  detection.

This is a speed-change demonstration, not blade-fault detection.

## Analysing a UART log

    python3 tools/sentinel_log_report.py capture.log

This prints the restarts, learns and saves, every state change with its
detection latency, the CPU time per block and per window, the accelerometer
sample rate, and the FIFO and ring-full counts. It parses the `SN t=` lines;
capture them with any serial terminal at 115200 8N1 (SW4 = UART4).

## Limitations

- The installed BMI323 driver delivers only ~50 Hz samples; fan vibration at
  medium and high speed stays at the noise floor (~0.45 mg^2) and is not
  resolved, so the vibration score is unsupported for this fan. ALERT
  (two sensors) has not been demonstrated on a fan fault; vibration flags
  seen so far came from handling the setup or from disturbances of unknown
  cause, never from the fan.
- A phone vibrating against the setup produced a strong accelerometer signal
  (~10^4 x the noise floor): this tests the sensor chain only, not a fan fault.
- Microphone features are absolute levels: moving the fan or board shifts
  them like a fault would. Re-learn after moving anything.
- OK means no change from the learned baseline in these two simple
  features; it is not a statement that the machine is healthy.
- Voices and bumps near the board can cause short CHECKs, and moving the
  board after learning needs a relearn. The sound baseline can be very
  tight (standard deviation at its 0.05 floor), so a small steady offset
  can keep CHECK latched until the next relearn.
