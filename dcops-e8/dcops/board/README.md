# Board-day tools (DevKit-E8, M55 HP + Ethos-U85-256)

The serial console is not on the PRG USB, so everything is read over J-Link from debug globals
(`g_cls*`, `g_kws*`, the cycle counters and the 1 ms `tick_count`). Addresses come from the .axf
on every run (`arm-none-eabi-nm -C`), so the tools follow each rebuild.

| File | What it does | Touches the board |
|---|---|---|
| `axfinfo.py <axf>` | Build facts: MRAM end/headroom vs 0x80578000, embedded .pte size(s), RAM per region, labels; flags an image that is stale vs its CMake `MODEL_PATH` | no |
| `flash.sh <axf>` | Size check -> `scripts/alif/flash_e8_hp.sh` -> reads 0x8057B000..0x80580000 and compares it with the backup -> power-cycle prompt | flashes |
| `measure.py <axf>` | Reads globals every 1 s for N s **without halting**: FPS, pre/NPU/post/frame ms, raw + smoothed top-1, KWS stats -> `measurements/<ts>.json/.md` | reads only |
| `snapshot.py <axf>` | Halts in `wait_for_capture`, dumps the model-input frame + results (+ LCD), resumes, reruns the student on the PC (float + int8 sim), board vs PC top-3 -> `snapshots/<ts>/` | halts for a few ms per attempt |
| `pc_infer.py` | PC side of the snapshot (ET 1.1 env, no ExecuTorch runtime) | no |
| `jlink.py` | Commander runner, `mem32`/register parsers, symbol + addr2line helpers | - |
| `tests/` | Offline tests; `fake_jlink.py` stands in for JLinkExe | no |

All scripts take `--dry-run` (print the J-Link Commander script, run nothing). `measure.py` and
`snapshot.py` also take `--dry-run --simulate` to push fake output through the whole pipeline.

## Morning sequence

```sh
cd ~/projects/arm-executorch-hack/executorch-alife8
source ~/projects/arm-executorch-hack/env.sh
AXF=build-fusion/bin/mlek_alif_object_detection.axf      # or build-dcops/..., whichever is judged

# 0. offline sanity (5 s): all tests must pass
python3 -m unittest discover -s dcops/board/tests

# 1. build (see dcops/docs/README_SECTION.md step 3 for a fresh configure)
cmake --build build-fusion --target mlek_alif_object_detection -j8

# 2. check the image BEFORE flashing: "OK" on the MRAM line, no "stale" warning on the .pte line
python3 dcops/board/axfinfo.py $AXF

# 3. flash (guarded) + boot-area check; it then tells you to power-cycle and waits
dcops/board/flash.sh $AXF
#    -> unplug the PRG USB, wait ~5 s, plug back in, wait for the camera preview on the LCD, press Enter.
#    Exit code 3 = boot area differs from the backup: STOP, see Recovery below before power-cycling again.

# 4. performance (point the camera at a component; 20 s, core keeps running)
python3 dcops/board/measure.py $AXF --seconds 20 --note "server rack photo, 25 cm"
#    -> prints the tables, writes dcops/board/measurements/<timestamp>.json + .md

# 5. end-to-end check on one frame (board NPU vs PC int8 sim vs PC float), with the LCD screenshot
dcops/.venv-train/bin/python dcops/board/snapshot.py $AXF --lcd
#    run is taken from the folder of the build's MODEL_PATH when it has student.pt; else --run r160_kd
#    -> dcops/board/snapshots/<timestamp>/{frame.png, frame_x4.png, lcd.png, board.json, pc.json, compare.md}
```

### 6. Paste into `dcops/docs/DEVPOST.md` (## Performance)

`measure.py` prints a block headed **"DEVPOST rows"**. It maps onto the table like this:

| DEVPOST row (placeholder) | Source |
|---|---|
| Classifier (MobileNetV2 160 px) NPU inference `TODO(measure)` ms | DEVPOST rows: "NPU inference" (mean of the firmware's ~1 s `g_clsPerfUs` windows; the cycle-counter column is a cross-check) |
| Classifier pre / post `TODO(measure)` ms | DEVPOST rows: "pre / post" |
| Classifier build frame rate ~6 FPS / Frame-to-frame | DEVPOST rows: "Frame-to-frame ... FPS" (FPS = `g_clsFrames` delta over on-chip `tick_count` time) |
| DC-Ops student .pte (17 classes) `TODO(measure)` KB | ".pte size (embedded)" (CPU ops: `export.json` `cpu_ops` of the run) |
| KWS latency (MFCC + inference) `TODO(measure)` | "KWS latency" row (only on a voice build with `g_kwsPerfUs`) |
| MRAM used / SRAM used | "MRAM used" / "SRAM used" rows (from the ELF; update the "validation build" wording) |
| **End-to-end check** paragraph | `snapshots/<ts>/compare.md`: top-3 board vs PC int8 sim |

Keep the measurement `.md`/`.json` (and the snapshot folder) as evidence; the main session commits them.

## Notes and gotchas

- **Power-cycle after every J-Link flash or reset.** The MT9M114 hangs otherwise: the preview
  freezes and `g_clsFrames` stops. `measure.py` warns when frames do not advance.
  `measure.py` never halts or resets; `snapshot.py` halts for a moment per attempt and always
  ends with `g` (a test checks this), which does not upset the camera (validated on 2026-10-17).
- `snapshot.py` takes `--attempts 8` halts, 30-110 ms apart, and keeps the first whose PC is inside
  `wait_for_capture` (inlined into `hal_camera_get_captured_frame`; PC in `camera_image_ready` with LR
  there also counts). At that point the last frame is fully processed, so `rgb_image` and `g_cls*`
  match. If none qualifies: is the app running (run `measure.py`)? Then try `--attempts 16`.
- `rgb_image` = 176-byte TIFF header (usually all zeros: the firmware fills a copy) + model-input
  RGB888. The input side comes from the run's `metrics.json` (`--res` overrides it).
- The PC rerun takes about a minute (it calibrates the int8 model on the run's `calib.npz` the
  same way `dcops/export/export_student.py` does). `--no-pc` skips it;
  `snapshot.py --pc-only snapshots/<ts> --run <run>` reruns it later.
- If the firmware's labels differ from the run's `labels.txt`, `compare.md` says so: the image was
  not built from that run, and indices cannot be compared.
- `axfinfo.py` flags a `.pte file ... differs from the embedded model` when the CMake `MODEL_PATH`
  changed after the build: rebuild before flashing.
- Optional, before flashing anything new: take a fresh full backup
  (`printf 'connect\nh\nsavebin backup.bin 0x80200000 0x380000\ng\nexit\n'`, see README_SECTION.md).
  `flash.sh --backup FILE` compares against a different backup.

## Recovery

**Symptom:** after a flash the screen stays dark and J-Link to `AE822FA0E5597LS0_M55_HP` fails
("Could not find core" / "Could not read AHB ROM"), while `AE822FA0E5597LS0_M55_HE` connects; or
`flash.sh` exits 3 with "MISMATCH". **Cause:** the Alif boot table (ATOC at 0x8057BC00) was overwritten.

```sh
BK=$(ls -t ~/projects/arm-executorch-hack/board-backups/devkit-e8-*.bin | head -1)
# 1. write the full HP-partition backup back THROUGH THE HE CORE
printf 'connect\nh\nloadbin %s 0x80200000\nexit\n' "$BK" > /tmp/r.jlink
/Applications/SEGGER/JLink/JLinkExe -device AE822FA0E5597LS0_M55_HE -if SWD -speed 4000 \
    -autoconnect 1 -CommanderScript /tmp/r.jlink
# 2. power-cycle (unplug/replug PRG USB). The old YOLO preview app should boot.
# 3. confirm the boot area, then re-flash the (fixed) image
dcops/board/flash.sh --verify-only --no-wait
dcops/board/flash.sh $AXF
```

`board-backups/restore.sh` does the same restore through the **HP** core: it only works while the
HP core still connects (e.g. to undo a bad app, not a lost boot table). Without any backup, rewrite
the boot table with Alif SETOOLS (`app-gen-toc` + `app-write-mram`).

## Testing without the board

```sh
python3 -m unittest discover -s dcops/board/tests -v       # 27 tests, ~5 s
python3 dcops/board/measure.py $AXF --dry-run --simulate   # full pipeline on fake J-Link output
dcops/.venv-train/bin/python dcops/board/snapshot.py $AXF --dry-run --simulate \
    --sim-frame some.png --out-dir /tmp/snap-sim             # includes the real PC rerun
JLINK=dcops/board/tests/fake_jlink.py FAKE_JLINK_BACKUP=$BK dcops/board/flash.sh --verify-only --no-wait
```

The tests cover: `mem32`/register parsing (incl. multi-line reads and missing output), symbol lookup
with anonymous namespaces on the existing `build-classify` and `build-kws-et` images, the
`wait_for_capture` PC mapping via addr2line, MRAM-limit agreement with the guard script, boot-area
compare (match and one flipped byte), `flash.sh` end to end with a fake JLinkExe (flash path,
verify-only, corrupted boot table -> exit 3, oversize image refused before any J-Link call), the
measure/snapshot pipelines on simulated output, KWS decoding, RGB565 and TIFF helpers.
