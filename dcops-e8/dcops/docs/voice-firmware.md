# Voice on the E8: keyword spotting with ExecuTorch (alif_kws)

Status as of 2026-10-17 night: **builds and links**, with a **random-weight placeholder** model. Not flashed and not run on the board yet.

## How it works on the board

```
PDM (or I2S) mics -> audio_alif HAL (DMA, 0.5 s strides, auto-gain) -> 1 s int16 window (16000 samples)
  -> KwsPreProcess / MicroNetKwsMFCC (float, CPU): 49 frames x 10 MFCC, 640-sample Hann, stride 320,
     1024-pt FFT, 40 HTK mel 20-4000 Hz on the magnitude, ln, DCT-II
  -> DsCnnKwsModel (.pte): cortex_m::quantize_per_tensor (CPU) -> Ethos-U85 delegate (all 12 ops) ->
     cortex_m::dequantize_per_tensor (CPU) -> 12 float logits
  -> KwsPostProcess / KwsClassifier: softmax, average over the last N windows, top-1 vs threshold 0.5
  -> UART log + LCD text, plus an MHU message to the other core on "go" / "stop" (SE services)
```

- The use case is `source/app/use_case/alif_kws/`. It now supports `ML_FRAMEWORK=TensorFlowLiteMicro;ExecuTorch`. The TFLM path is unchanged: same defaults, and the MicroNet model is picked through a `KwsModel` alias.
- The new model wrapper is `source/lib/mlek/fwk/executorch/DsCnnKwsModel.hpp`, input `[1,1,49,10]` float. It is NCHW, so frames is index 2 and coeffs is index 3. For MicroNet TFLM they are 1 and 2.
- With ExecuTorch, `EtTensor::GetQuantParams()` returns scale 0, so `KwsPreProcess` takes its existing float-MFCC path automatically. No quantisation is done in the app, because the .pte quantises its own input.
- The audio loop is the stock `alif_kws` loop: a 1 s window that moves on every 0.5 s, `RESULTS_MEMORY 8`.

### What the ExecuTorch build prints

- **At start-up, once (`KWS_ET_SELFTEST`, default 1):**
  - `KWS_SELFTEST_A logits: ...`: model only, deterministic input `x[i] = ((i*37)%41) - 30`.
  - `KWS_SELFTEST_B mfcc[0]`, `mfcc[24]`, `mfcc sum`, `logits`: a synthetic 1 s tone (440 Hz + 1250 Hz, int16) run through the firmware MFCC and then the model.
- **Every window:** `logits: <12 floats>`, followed by the usual "For timestamp ..." results.

To build without the self-test, add `-DKWS_ET_SELFTEST=0`: append it to `alif_kws_COMPILE_DEFS` in usecase.cmake, or put it in CXX flags.

## Host reference (compare against the UART log)

```sh
PY=~/projects/arm-executorch-hack/experiments/blazeface/.venv-et110/bin/python
PATH=$(dirname $PY):$PATH $PY dcops/export/kws_host_check.py <run_dir> resources/kws/samples/yes.wav [--peak-normalise]
```

- It prints the same `KWS_SELFTEST_A` / `KWS_SELFTEST_B` lines. Each logit line comes twice: from the float model and from the int8 PT2E model, which uses the same recipe as `export_kws.py`. For each wav it also prints MFCC and logits.
- The .pte itself cannot run on the host, because its body is an Ethos-U command stream.
- **What should match:**
  - The `mfcc` lines should match the board to about 1e-3, since it is the same algorithm in float32 and in numpy.
  - The board logits should be close to the **int8** host logits. They will not be bit-exact: Vela's integer rounding differs from torchao's fake-quant.
- The placeholder's expected output is saved in `dcops/models/placeholder/kws_random/host_check.txt`.

## Dropping in the real model

When `dcops/models/hub/kws/<run>/` appears (the overnight watcher runs `export_kws.py` on it), or to export by hand:

```sh
PY=~/projects/arm-executorch-hack/experiments/blazeface/.venv-et110/bin/python
PATH=$(dirname $PY):$PATH $PY dcops/export/export_kws.py dcops/models/hub/kws/<run>
#   -> <run>/kws_ethos-u85-256.pte + export.json (check cpu_ops == only cortex_m q/dq, int8 agreement)
PATH=$(dirname $PY):$PATH $PY dcops/export/kws_host_check.py dcops/models/hub/kws/<run> resources/kws/samples/*.wav
```

Then rebuild. You must reconfigure, because the model C array is generated at configure time:

```sh
source ~/projects/arm-executorch-hack/env.sh
cmake -B build-kws-et -Dalif_kws_MODEL_PATH=$PWD/dcops/models/hub/kws/<run>/kws_ethos-u85-256.pte
cmake --build build-kws-et -j4 --target mlek_alif_kws
```

- **Labels:** no change is needed. `kws.py` uses the MLEK order, and the build uses `resources/kws/labels/micronet_kws_labels.txt`.
- **Bug fixed tonight in `export_kws.py`:** `torch.from_numpy(x[None]).float()` kept numpy's stride-0 batch dim, and `to_executorch` failed with "0 in strides is not supported". The calib tensors are now fresh copies. Before the fix, every KWS export would have failed. The watcher retries any run without `export.json`.

## Build (ExecuTorch, HP core, DevKit-e8)

```sh
source ~/projects/arm-executorch-hack/env.sh
cmake -B build-kws-et -DCMAKE_TOOLCHAIN_FILE=scripts/cmake/toolchains/bare-metal-gcc.cmake \
  -DTARGET_PLATFORM=alif -DTARGET_SUBSYSTEM=RTSS-HP -DTARGET_BOARD=DevKit-e8 -DUSE_CASE_BUILD=alif_kws \
  -DML_FRAMEWORK=ExecuTorch -DETHOS_U_NPU_ID=U85 -DETHOS_U_NPU_CONFIG_ID=Z256 -DCONSOLE_UART=4 \
  -DCMAKE_BUILD_TYPE=Release -DMLEK_LOG_LEVEL=MLEK_LOG_LEVEL_INFO -DTARGET_MICS=PDM \
  -DML_FWK_TMP_MEM_SIZE=0x00080000
cmake --build build-kws-et -j4 --target mlek_alif_kws
```

- **`TARGET_MICS=PDM`:** this follows the E8 ASR example in `build_commands.txt`. The DevKit-e8 default is I2S. If PDM is silent on the board, rebuild with `-DTARGET_MICS=I2S`.
- **`ML_FWK_TMP_MEM_SIZE`:** the platform default is 2 MiB. Vela reports 21.6 KiB SRAM for this model, so 512 KiB is plenty and could go down to 64 KiB.
- **Not built:** the TFLM variant, because the MicroNet .tflite files are not downloaded.

### Build result (placeholder model)

Output: `build-kws-et/bin/mlek_alif_kws.axf` plus `bin/sectors/alif_kws/mram.bin` (1,002,964 B).

| | bytes |
|---|---|
| text | 880,180 |
| data | 122,772 |
| bss | 2,123,984 |
| model (.pte in MRAM) | 34,576 |

Largest SRAM users:

| buffer | size |
|---|---|
| LCD image buffer | 768 KB |
| ExecuTorch temp pool | 512 KB |
| `tensorArena` (method allocator) | 256 KB |
| display buffer | 150 KB |
| `audio_inf` | 48 KB |

**MRAM:** the LOAD segments run from 0x80200000 up to **0x802F4DD4**, about 979 KB. That is well below the 0x80578000 limit, with about 2.5 MB of headroom. The boot table at 0x8057BC00 is untouched.

To check this yourself: `arm-none-eabi-readelf -lW build-kws-et/bin/mlek_alif_kws.axf | grep 'LOAD.*0x80'`. The end address is the highest PhysAddr+FileSiz.

### Flashing (not done)

- An HP ATOC entry like `alif_img_class_hp.json` (`"cpu_id": "M55_HP"`) should work. The build writes `build-kws-et/bin/sectors/alif_kws/mram.bin` and the .axf, but no standalone .bin. If the ATOC flow needs one, make it with `arm-none-eabi-objcopy -O binary`. Check that `mramAddress` matches the 0x80200000 link base that the other E8 HP builds use.
- `SE_SERVICES_SUPPORT` is ON (the alif_kws default). On "go" / "stop" the app sends an MHU message to the HE core. If the HE core is not listening, `SERVICES_send_msg` gives up after its ack timeout. That is a bounded stall, not a hang. Turn it off with `-DSE_SERVICES_SUPPORT=OFF` if it gets in the way.

## Verified vs not verified

**Verified (host / build only):**
- `export_kws.py` produces a 34 KB .pte from the DSCNN(64) placeholder. All 12 ops are on the NPU, and the only CPU ops are `cortex_m::quantize/dequantize_per_tensor`.
- `generate_pte_ops_lib` finds no aten ops, so no portable ops library is needed. `cortex_m_ops_lib` is already linked.
- The firmware compiles and links for HP + U85-256 + ExecuTorch. The embedded model bytes equal the .pte.
- MRAM end is 0x802F4DD4, below 0x80578000.
- The host reference runs, and float vs int8 agree on top-1 for the wavs tested.

**Not verified:**
- Anything on the board: ET method load, Ethos-U85 execution, the mic path on the DevKit-e8 (PDM vs I2S), and the self-test numbers matching the host.
- Accuracy of the real model. That depends on the Colab run.
- The display layout of the KWS text on the E8 LCD.
- The DSCNN tail (AvgPool, then reshape, then Linear) is the "proven classifier" pattern, not the permute+cat pattern that scrambled detector outputs. The self-test will confirm this on the board.

**Known risk: train/deploy gain mismatch.**
- `audio_alif` applies auto-gain before the MFCC: up to +80 dB, scaling the peak toward full scale and adapting at 2 dB/s.
- Speech Commands clips in `kws.py` are used at their raw level, with no gain augmentation.
- A gain g adds about ln(g)·√80 to MFCC c0 and leaves c1..c9 unchanged. So on the board, quiet or silent input looks like loud, amplified noise.
- **If the real model over-fires on silence or noise on the board, try these:**
  - (a) add random gain or peak-normalisation augmentation in `kws.py`;
  - (b) fix the gain with `set_audio_gain()` in `audio_alif.c`. Careful: despite its `gain_db` name, it stores the argument directly as the *linear* gain, and NaN turns auto-gain back on;
  - (c) use `kws_host_check.py --peak-normalise` to see how much the logits move.

## Running KWS alongside the camera classifier (notes, not implemented)

### Option 1: same app on HP

Add the KWS model and MFCC to the object-detection / classifier app, and run KWS on the 0.5 s audio stride between camera frames.

**Memory cost:**
- MRAM: about 35 KB model plus about 20 KB code. The ExecuTorch runtime and delegate are already in that image.
- SRAM: `audio_inf` 48 KB, MFCC tables (about 32 KB twiddles, already present for 4096 FFT) and the feature cache, about 64 KB arena, and a share of the temp pool.

**MRAM budget:**
- The current camera build (`build-classify`) ends at **0x8052AD84**. That leaves only about 308 KB below 0x80578000.
- KWS fits, but only just, so check readelf after every change.

**Blockers:**
- `EtModel::Init` holds the loaded `Method` in a function-local `static auto method` (`source/lib/mlek/fwk/executorch/EtModel.cc:217`). A second `EtModel` instance in the same image would silently reuse the first model's method. To run two ExecuTorch models in one app, that static has to become per-instance storage (a member or placement buffer). This is a small change in `source/lib`, but it touches every ET use case.
- Both models also share the single static `sTmpAllocationPool`. That is fine while inferences run one after another on one core.
- SRAM is already tight in camera builds: the camera build's bss is about 8.3 MB. Shrink the KWS arena and temp pool (64 KB each is enough).
- The audio DMA and the camera/ISP can coexist on HP, but nobody has tried it in this tree.

### Option 2: KWS on the HE core with its own NPU

This is the path MLEK intended: `alif_kws` on `RTSS-HE` with `LINKER_SCRIPT_NAME=RTSS-HE-TCM`, `GLCD_UI=NO` (see `build_commands.txt` and `alif_kws_he.json`).

**How it splits:**
- HE does MFCC + KWS on its own Ethos-U55.
- HE sends the keyword to HP over MHU. This is already in `send_msg_if_needed()`, but only for "go" / "stop". Extend it to all keywords.
- HP keeps the camera app and gets a `services_init(cb)` callback for the keyword.

**What it needs:**
- A separate export with `target="ethos-u55-128"` in `export_kws.py`. Check the E8 HE NPU config. The build uses `-DETHOS_U_NPU_ID=U55`.
- ExecuTorch on HE: about 250 KB runtime code from MRAM.
- Data in 256 KB DTCM: audio 48 KB + arena 64 KB + temp 64 KB + MFCC. It fits, but set `-Dalif_kws_ACTIVATION_BUF_SZ=0x10000 -DML_FWK_TMP_MEM_SIZE=0x10000`.

**MRAM conflict:**
- `alif_kws_he.json` places the HE image at **0x80480000**, which is *inside* the current camera HP image (that image ends at 0x8052AD84).
- The HE image would have to move above the HP end and below 0x80578000. That leaves only about 300 KB, which is probably too small for an ExecuTorch HE image.
- The alternative is to move the HP image or shrink it, for example by moving models to OSPI flash (`*_MODEL_IN_EXT_FLASH`).

**Recommendation:** for the demo, use option 1 (same HP app) once the `static method` issue is fixed. It reuses the ExecuTorch runtime already in that image, and there is MRAM headroom for a 35 KB model. Option 2 is the cleaner low-power design, but it needs an MRAM re-layout and a U55 export, and neither has been tried.

## Files

- `source/app/use_case/alif_kws/usecase.cmake`: ExecuTorch framework branch, model path default (placeholder), arena size, `generate_pte_ops_lib`.
- `source/app/use_case/alif_kws/src/MainLoop.cc`: `KwsModel` alias (`DsCnnKwsModel` / `MicroNetKwsModel`).
- `source/app/use_case/alif_kws/src/UseCaseHandler.cc`: `KwsModel` alias, ET self-test, per-window logit print.
- `source/lib/mlek/fwk/executorch/DsCnnKwsModel.hpp`: new.
- `dcops/export/make_kws_placeholder.py`: new; writes `dcops/models/placeholder/kws_random/`.
- `dcops/export/kws_host_check.py`: new; host reference.
- `dcops/export/export_kws.py`: stride-0 calib fix (one line).

## Update: two ExecuTorch models in one image (2026-10-17, late)
The blocker above is fixed: `EtModel::Init` (source/lib/mlek/fwk/executorch/EtModel.cc) now keeps the
loaded method in a per-instance member (`m_method`) instead of a function-level `static`, so a keyword
spotter and the camera classifier can each own their own ExecuTorch method in the same HP image.
Verified: `build-classify` and `build-kws-et` both rebuild and link (MRAM end 0x8052AD44 and
0x802F4E64). Not yet verified: two models actually running side by side on the board.
