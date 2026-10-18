# Camera + voice in one image (alif_object_detection, ENABLE_VOICE)

Status (2026-10-18, overnight): **builds and links** with placeholder and real models. MRAM, SRAM
and DTCM all fit. **Not flashed and not run on the board.**

The camera classifier (MobileNetV2, CLASSIFY mode) and a DS-CNN keyword spotter run as two
ExecuTorch programs in one bare-metal image on the M55-HP. Both use the HP core's Ethos-U85-256.
Spoken keywords drive the DC-Ops scanner card.

## What the user sees

| Say | Effect |
|---|---|
| **go** | Freezes ("holds") the class on the card. The card border turns blue, the confidence slot shows `HOLD`, and the summary becomes an inspection checklist: `Identification correct?`, each `LED: ...` line from the knowledge base, then `Cables and latches seated?`. |
| **down / up** | Moves the cursor (`>`) through the checklist. |
| **yes / no** | Marks the item under the cursor pass (check mark) or fail (X), then moves to the next item. A fail turns the border red. `no` on `Identification correct?` rejects the ID and goes back to live scanning. When every item is marked, the card shows `Inspection done: N pass, M fail`. |
| **stop** | Goes back to live scanning. |
| left / right / on / off | Shown in the mic indicator only. |

- **Mic indicator** (bottom left, above the perf footer):
  - `listening: say "go" to hold` until the first keyword.
  - After a keyword: `"go"  93%  -  holding`, green for about 1.5 s (red for fail or reject), then grey.
  - If voice init failed: `voice off (init failed)`.
- **Perf footer** gets a second line: `KWS / 0.5 s:  MFCC x.x ms  |  NPU y.y ms`.
- While a class is held, the classifier keeps running (its globals stay live), but the card does not change.

## Architecture

```
            HP core (M55 @400 MHz, bare metal, one loop)            Ethos-U85-256       background
frame n:    hal_camera_start() ------------------------------------------------------> CPI DMA fills frame
            VoiceTick(): new 0.5 s stride ready?
              no  -> return (~1 us)
              yes -> gain (audio_alif) + MFCC of the newest 0.5 s (25 frames, cached 24)
                     DS-CNN q/dq on the CPU ---------------> DS-CNN (12 ops)
                     softmax, 2-window average, fire? -> card/mic UI
            hal_camera_get_captured_frame()  (waits for the rest of the frame)
            crop/resize -> float -> MobileNetV2 ------------> MobileNetV2
            softmax, EMA, card (unless held), perf footer
            LVGL renders in PendSV
audio:      PDM FIFO IRQ (prio 0) -> audio_alif voice_data_cb -> float16 into slot k;
            once 8000 samples are in, the callback (interrupt context) re-arms the capture
            into slot k+1 of a 3-slot ring. The capture never waits for the loop.
```

- **Overlap with the camera.** KWS runs once per 0.5 s stride. It runs between starting a camera
  capture and waiting for it, so its cost is partly hidden behind the camera's exposure and DMA.
  On other frames, KWS costs one load and compare.
- **Missed strides.** Audio is captured continuously into a 3-slot ring of 0.5 s strides.
  - If the loop falls 2 or more strides behind, the 1 s window is rebuilt from the two newest strides.
  - The MFCC cache is dropped, and `g_kwsGaps` counts these events.
  - Normally only the newest 0.5 s is transformed: `KwsPreProcess` reuses the cached frames of the overlap.
- **Decision rule:**
  - softmax per 1 s window, averaged over the 2 newest windows;
  - top-1 must not be `_silence_` or `_unknown_`, and must reach `KWS_THRESHOLD` (default 0.7);
  - after a keyword fires, nothing fires for the next 2 strides, so one utterance cannot fire twice.
- **No MHU, no RTOS, no SE messages from the voice path.**
  - `alif_kws`'s `send_msg_if_needed` is not used.
  - `SE_SERVICES_SUPPORT` stays at the platform default because the camera app had it ON; voice adds no SE calls.
- **Two ExecuTorch programs:**
  - `EtModel` now keeps its method per instance, so each model has its own `Method`, method allocator and planned buffers:
    - vision: `tensorArena`, 1.5 MiB in SRAM;
    - KWS: `kwsArena`, 128 KiB in DTCM.
  - Both share the one `sTmpAllocationPool` (2 MiB SRAM) that holds the Ethos-U scratch. This is safe because inferences run one after another and ExecuTorch resets the temp allocator per instruction. Vela needs 755 KiB (vision) and 51 KiB (KWS).
  - The Ethos-U driver is used by one model at a time. The DTCM arena is CPU-only: the NPU only touches scratch (SRAM), the command stream and weights (MRAM).

## Build

These options exist only with `-DML_FRAMEWORK=ExecuTorch -Dalif_object_detection_ET_TASK=CLASSIFY`.
The default `ENABLE_VOICE=OFF` leaves every existing build unchanged; `build-classify` was re-checked (see below).

| Option | Default | |
|---|---|---|
| `alif_object_detection_ENABLE_VOICE` | OFF | Turns on KWS + voice UI. A FATAL_ERROR unless CLASSIFY. |
| `alif_object_detection_KWS_MODEL_PATH` | `dcops/models/placeholder/kws_random/kws_ethos-u85-256.pte` | DS-CNN .pte from `export_kws.py` |
| `alif_object_detection_KWS_LABELS` | `resources/kws/labels/micronet_kws_labels.txt` | Must match the model's output order. The hub runs' `labels.txt` are identical. |
| `alif_object_detection_KWS_ACTIVATION_BUF_SZ` | `0x00020000` | KWS method arena, in DTCM |
| `alif_object_detection_KWS_THRESHOLD` | `0.7` | Averaged softmax needed to fire |
| `TARGET_MICS` | I2S (board default) | Use **PDM** on the DevKit-e8, as in `voice-firmware.md` |

With voice ON, the use case also:
- links the `kws` API (MFCC);
- generates the KWS model into `generated/alif_object_detection/src/kws/` under namespace `arm::app::kws`;
- runs `generate_pte_ops_lib` for the KWS program (no aten ops, so no extra library);
- compiles `audio_alif` with `AUDIO_ALIF_QUIET`, which removes 2 UART lines per stride;
- compiles `arm_math` with `MLEK_FFT_FIXED_LEN=1024` and CMSIS-DSP with `-ffunction-sections`. Without this, the generic `arm_rfft_fast_init_f32` / `arm_cfft_init_f32` pull in the twiddle tables of every FFT size: 92 KiB of extra DTCM data and MRAM.

### Placeholder build (what `build-fusion` is configured with)

```sh
source ~/projects/arm-executorch-hack/env.sh
cmake -S . -B build-fusion -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=scripts/cmake/toolchains/bare-metal-gcc.cmake \
  -DTARGET_PLATFORM=alif -DTARGET_SUBSYSTEM=RTSS-HP -DTARGET_BOARD=DevKit-e8 \
  -DUSE_CASE_BUILD=alif_object_detection \
  -DML_FRAMEWORK=ExecuTorch -DETHOS_U_NPU_ID=U85 -DETHOS_U_NPU_CONFIG_ID=Z256 \
  -DALIF_CAMERA_ENABLED=ON -DALIF_CAMERA_MODULE=MT9M114 -DALIF_ISP_ENABLED=OFF \
  -Dalif_object_detection_ET_TASK=CLASSIFY \
  -Dalif_object_detection_MODEL_PATH=$PWD/dcops/models/placeholder/run17/student_ethos-u85-256.pte \
  -Dalif_object_detection_CLASS_LABELS=$PWD/dcops/models/placeholder/run17/labels.txt \
  -Dalif_object_detection_ACTIVATION_BUF_SZ=0x00180000 \
  -Dalif_object_detection_ENABLE_VOICE=ON -DTARGET_MICS=PDM -DCONSOLE_UART=2 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-fusion -j4 --target mlek_alif_object_detection
arm-none-eabi-readelf -lW build-fusion/bin/mlek_alif_object_detection.axf | grep "LOAD.*0x80"   # highest PhysAddr+FileSiz must be < 0x80578000
# or: python3 <scratchpad>/endaddr.py build-fusion/bin/mlek_alif_object_detection.axf   # must say OK
```

### Real models

Reconfigure: the model C arrays are generated at configure time. Pick the runs you want; this
pair was built and checked:

```sh
cmake build-fusion \
  -Dalif_object_detection_MODEL_PATH=$PWD/dcops/models/hub/vision/r160_kd/student_ethos-u85-256.pte \
  -Dalif_object_detection_CLASS_LABELS=$PWD/dcops/models/hub/vision/r160_kd/labels.txt \
  -Dalif_object_detection_KWS_MODEL_PATH=$PWD/dcops/models/hub/kws/dscnn_m_agc/kws_ethos-u85-256.pte
cmake --build build-fusion -j4 --target mlek_alif_object_detection
```

- **Choosing the KWS run:** prefer an `_agc` run (`dscnn_m_agc` or `dscnn_s_agc`). These were trained with gain augmentation, which matters because the HAL's auto-gain is in the loop (see the "Known risk" section of `voice-firmware.md`).
- **Flashing:** only with `scripts/alif/flash_e8_hp.sh`, then power-cycle.

## Memory

| | voice OFF (same 17-class placeholder) | voice ON (placeholders) | delta |
|---|---|---|---|
| MRAM image end | 0x8052E1D4 | **0x80548794** | +105.4 KiB |
| headroom below 0x80578000 | 302,636 B | **194,668 B** | |
| ITCM text (256 KiB) | 170,592 B | 192,688 B | +21.6 KiB (free: 69,456 B) |
| DTCM .data | 7,400 B | 28,952 B | +21.0 KiB (FFT-1024 tables) |
| DTCM .bss | 38,312 B | 256,120 B | +212.7 KiB (`kwsArena` 128 KiB, ring 46.9 KiB, window 31.3 KiB, ...) |
| DTCM free (heap end to stack) | | 501,640 B | |
| SRAM .bss.sram (7 MiB) | 7,191,760 B | 7,191,760 B | **0** (148,272 B free) |
| heap (newlib, 92 KiB) | | MFCC vectors + KwsPreProcess; measured on board via `g_kwsHeapUsed` | |

- **Real models** (r160_kd + dscnn_m_agc, before the 272-byte self-test was added): image end **0x805478E4**, 198,428 B headroom. The real vision .pte files are about 65 KB smaller than the run17 placeholder; dscnn_m is about 61 KB bigger than the KWS placeholder.
- **`build-classify`** (voice OFF, 67-class default): rebuilt with these changes. Image end is still **0x8052AD44**, and every allocatable section has the same size as before.

## Debug globals (J-Link, no halt needed)

Look up addresses per build with:
`arm-none-eabi-nm build-fusion/bin/mlek_alif_object_detection.axf | grep -E " g_(kws|voice|cls)"`.

| Global | Meaning |
|---|---|
| `g_kwsInitState` | 1 = running. Failures: -1 model init, -2 tensor types, -3 MFCC geometry, -4 audio init, -5 capture start, -6 self-test inference |
| `g_kwsSelfTestA[12]` (float) | Logits for the deterministic input. Placeholder host int8 reference: `0.7370 -0.5484 -1.0626 -0.9769 -1.3711 -0.5656 -0.7712 0.9083 -0.3256 -1.3711 1.0797 -1.5596` (`dcops/models/placeholder/kws_random/host_check.txt`). Also printed as `KWS_SELFTEST_A` on the console UART. |
| `g_kwsCaptured` / `g_kwsStrides` | Strides captured by the interrupt / processed by the loop. Both should climb by 2 per second. |
| `g_kwsGaps`, `g_kwsAudioErrors` | Should stay at 0 |
| `g_kwsLastIdx`, `g_kwsLastConf`, `g_kwsCount` | Last keyword fired (label index), its confidence x1000, and the count since boot |
| `g_kwsTopIdx`, `g_kwsTopProb` | Averaged top-1 on the last stride, including silence/unknown |
| `g_kwsLogits[12]` (float) | Raw logits of the last stride |
| `g_kwsPerfUs[3]` | About 2 s averages: MFCC (incl. gain), NPU, post, in us |
| `g_kwsHeapUsed` | malloc'd bytes after voice init |
| `g_voiceLatchedIdx`, `g_voiceCursor`, `g_voiceItemState[6]`, `g_voiceCommands` | Scanner state |
| `g_clsPerfUs[4]` | Unchanged meaning. Frame time now includes KWS on stride frames. |

Label indices: 0 down, 1 go, 2 left, 3 no, 4 off, 5 on, 6 right, 7 stop, 8 up, 9 yes, 10 _silence_, 11 _unknown_.

## Verified vs not verified

**Verified (host and build only):**
- Configures and links with ENABLE_VOICE=ON for two model pairs: the placeholders and r160_kd + dscnn_m_agc.
- MRAM end is below 0x80578000 with at least 190 KB to spare. SRAM is unchanged.
- Neither KWS .pte needs a portable ops library.
- With ENABLE_VOICE=OFF, `build-classify` produces sections of the same size and the same end address.

**Not verified (needs the board):**
- Two ET programs actually executing side by side on the U85.
- PDM capture on the DevKit-e8 while the camera runs.
- MFCC and KWS timings, and the camera FPS drop.
- Heap headroom.
- Real keyword accuracy through the board mic and auto-gain.
- The LVGL layout: card plus checklist height, mic label position against the two-line footer.

## Risks

- **PDM without DMA.** On HP the PDM driver is FIFO-interrupt driven (`RTE_PDM_DMA_ENABLE 0`, IRQ priority 0). Overflows show as one `PDM_fifo_callback: ARM_PDM_EVENT_ERROR` print. The LVGL lock only masks PendSV, so camera and NPU work should not block it. If PDM is silent, try `-DTARGET_MICS=I2S`.
- **Heap.** The heap is 92 KiB and shared with the rest of the app. The MFCC uses `std::vector`. Check `g_kwsHeapUsed`; there are no exceptions, so an allocation failure aborts.
- **ITCM is getting full** (69 KB left). More code in this image may need moving out of ITCM.
- **Auto-gain over-firing.** Auto-gain can make silence look like loud noise. If keywords fire on their own, raise `KWS_THRESHOLD` (for example 0.85), or use an `_agc` model.
- **Card height.** The held card with 3–4 checklist items is taller than the live card. It overlaps more of the preview; that is acceptable while held.

## On-board test plan (after flashing build-fusion with `flash_e8_hp.sh` and a power-cycle)

1. **Boot.** On UART2, look for `Voice: KWS ready` and a `KWS_SELFTEST_A` line. Over J-Link, `g_kwsInitState == 1`, and `g_kwsSelfTestA` should be close to the host int8 line above (placeholder) or to `kws_host_check.py` for the real run.
2. **Idle, 10 s.** `g_kwsCaptured` and `g_kwsStrides` both go up by about 20. `g_kwsGaps` and `g_kwsAudioErrors` stay at 0. The mic label shows `listening: say "go" to hold`.
3. **Point at a component until the card shows it.** Say **"go"**. The card turns blue with `HOLD` and a checklist, and the mic shows `"go" NN% - holding`. `g_voiceLatchedIdx` equals `g_clsShownIdx`, and `g_kwsLastIdx == 1`.
4. Say **"down"** twice, then **"up"**. The cursor moves, and `g_voiceCursor` follows.
5. Say **"yes"**, then **"no"**. Items get a check and an X, the border turns red, and `g_voiceItemState` shows 1 and 2.
6. Say **"stop"**. Live scanning resumes and `g_voiceLatchedIdx == -1`.
7. Say **"go"** with nothing on the card. The mic shows `nothing identified`.
8. **Performance.** Compare the footer FPS against a `build-classify` run of the same vision model. Read `g_kwsPerfUs`; the expected cost is a few ms per 0.5 s.
9. **False fires.** Stay quiet for 60 s. `g_kwsCount` should not change.

With the placeholder KWS (random weights), keywords are meaningless. Use it only for steps 1, 2 and 8. Steps 3–7 need a real KWS model.
