# Build and Deployment — Alif Industrial AOI

This guide documents how to reproduce the **Alif Industrial Automated Optical Inspection (AOI)** application from its source overlay. It complements the [project overview](README.md).

**Status:** A clean-checkout build was attempted but dependency downloads failed before compilation; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md). A fully reproducible clean-checkout build has **not** been demonstrated. The exact calibrated PTE is included in `model/`; its 60-image calibration corpus is not bundled; the instructions below document the required inputs and workflow, not a verified one-command build. The root repository's Pico Faces CMSIS build is separate from this Alif MLEK CMake build.

## Reproduction status and missing inputs

**The source overlay and original calibrated PTE are included; a clean-checkout
firmware build has not yet been demonstrated.** No firmware was configured, rebuilt, flashed,
or hardware-tested as part of this submission preparation. The steps below are
instructions for a later reproduction run, not a claim they passed here.

| Required input | Status / recovery |
| --- | --- |
| Base MLEK and dependencies | Download separately; exact revisions in `dependencies.lock.json`. Seven observed submodule revisions differ from their parent pins. |
| Calibrated `.pte` used by the working build | Included in `model/mv2_cal60_imagenet_v2_ethos-u85-256_shared_sram.pte`, 3,470,320 bytes; SHA-256 `f468ea2feae405abd7cfb9c0e95a7e9a357fd42698646846dba5ec081411ec7a`. No separate model download or re-export is needed. |
| Exact 60 calibration images | Optional workflow B only; not included; local corpus `~/alif/ai-workflow/CalImages`. Filenames and hashes recorded in `model/manifest.json`; publication rights and distribution remain unresolved. |
| Original pretrained checkpoint | Optional workflow B only; `mobilenet_v2-7ebf99e0.pth`, torchvision MobileNetV2 `IMAGENET1K_V2`; SHA-256 in manifest. Obtain through torchvision's weight download mechanism. |
| Python export environment | Recorded: torch 2.10.0, torchvision 0.25.0, torchao 0.15.0, ExecuTorch 1.1.0, Vela 5.0.0. Upstream requirements and Cortex-M backend requirements supply other packages; a complete transitive environment lock is not available. |
| Board provisioning | Matching Alif SETOOLS, device-specific `app-device-config.json`, board firmware/provisioning, and serial access must be supplied by the board owner. |

Do not silently substitute the upstream default `.pte`: it is a different model
artifact. A different calibration corpus or package resolver outcome is an
alternative experiment and requires separate model and on-board validation.
The working model's export report itself says compiled U85 execution was not
yet verified in that report; that report is not a hardware acceptance test.

## Build from a clean checkout

Use **Python 3.10–3.12**, Git, **CMake ≥3.21** (upstream requirements pin 3.30.5),
Make, and **Arm GNU Toolchain 13.3.Rel1 / GCC 13.3.1** as observed in the working
build. Put `arm-none-eabi-gcc`, `arm-none-eabi-g++`, and `arm-none-eabi-objcopy` on
PATH. Network access and substantial disk space are needed for the external
dependencies. This is the Alif MLEK CMake flow, separate from the root CMSIS flow.

From this repository's root, choose a **new** checkout directory:

```sh
export AOI_SUBMISSION="$PWD/alif-industrial-aoi"
git clone https://github.com/alifsemi/alif_ml-embedded-evaluation-kit.git "$AOI_SUBMISSION/work/mlek"
export AOI_MLEK="$AOI_SUBMISSION/work/mlek"
git -C "$AOI_MLEK" checkout --detach d7c39929fe482c1ce17b7be03edfab1db03acf06
git -C "$AOI_MLEK" submodule update --init --recursive
```

Restore the observed submodule revisions, including nested overrides, in this
new checkout only. The following uses Git fetch when an object is absent:

```sh
python3 - <<'PY'
import json, os, subprocess
from pathlib import Path
root = Path(os.environ["AOI_MLEK"])
lock = json.loads((Path(os.environ["AOI_SUBMISSION"]) / "dependencies.lock.json").read_text())
for item in lock["submodules"]:
    directory = root / item["path"]
    prefix = ["git", "-C", str(directory)]
    revision = item["revision"]
    if subprocess.run(prefix + ["cat-file", "-e", revision + "^{commit}"], capture_output=True).returncode:
        subprocess.run(prefix + ["fetch", "origin", revision], check=True)
    subprocess.run(prefix + ["checkout", "--detach", revision], check=True)
PY
python3 "$AOI_SUBMISSION/scripts/prepare.py" "$AOI_MLEK"
python3 "$AOI_SUBMISSION/scripts/prepare.py" "$AOI_MLEK" --apply
```

The helper checks the base revision, all recorded submodule revisions, checkout
cleanliness, and patch applicability before applying files. Run it before resource
setup; it intentionally rejects an already modified working tree. It does not
build, install packages, or access hardware. Missing/inaccessible remote commits
are a reproduction blocker; do not replace them with branch tips.

Set up the upstream ExecuTorch tools/resources using its installer.
`--skip-vela` skips default-model optimization/export; workflow A uses the included
PTE unchanged. Use a supported Python 3.10–3.12 interpreter for `python3`:

```sh
cd "$AOI_MLEK"
python3 set_up_default_resources.py --ml-frameworks executorch --use-case img_class --skip-vela
. resources_downloaded/env/bin/activate
python -m pip install ethos-u-vela==5.0.0
python -m pip show torch torchvision torchao executorch ethos-u-vela
```

Check these versions against the manifest. The upstream installer builds/installs
ExecuTorch and its Cortex-M dependencies; installation may modify dependency
checkouts, so retain a record of `git submodule status --recursive` and
`python -m pip freeze` afterward. Resolve incompatible packages before exporting.

### A. Build with the exact validated model (recommended)

Use the **original calibrated PTE included in `model/`**. It was copied unchanged
from the working project's `cal60_u85_candidate_20261018_v3` directory, without
re-exporting. Its recorded size is
**3,470,320 bytes**, and its SHA-256 is:

```text
f468ea2feae405abd7cfb9c0e95a7e9a357fd42698646846dba5ec081411ec7a
```

This is the reproducible firmware-build path: use the pinned source and
dependencies above with this exact model artifact. Calibration images and model
re-export are not required for this workflow. “Validated model” identifies the
original artifact and its existing validation evidence; it does not mean that a
clean-checkout build or additional hardware validation was performed during
submission preparation. The validation limits described above still apply.

Set the path to the included file, then verify its SHA-256 against the recorded
value before configuring. Stop if it differs. The existing CMake configuration
also enforces this same hash; leave that check unchanged.

```sh
export AOI_MODEL="$AOI_SUBMISSION/model/mv2_cal60_imagenet_v2_ethos-u85-256_shared_sram.pte"
shasum -a 256 "$AOI_MODEL"
```

After confirming the hash matches, use the existing build configuration. Place
`-D` for the model **before** `-C`:

```sh
cmake -S "$AOI_MLEK" -B "$AOI_SUBMISSION/build-appkit-e8" \
  -Dalif_img_class_MODEL_PATH="$AOI_MODEL" \
  -C "$AOI_SUBMISSION/config/appkit-e8.cmake" \
  -DCMAKE_TOOLCHAIN_FILE="$AOI_MLEK/scripts/cmake/toolchains/bare-metal-gcc.cmake"
cmake --build "$AOI_SUBMISSION/build-appkit-e8" --parallel 4
```

Expected outputs are `build-appkit-e8/bin/mlek_alif_img_class.axf` and
`build-appkit-e8/bin/sectors/alif_img_class/mram.bin`. Model arrays, labels,
sample sources, and the adapted touch driver are generated by the upstream
build and are deliberately not submitted.

### B. Regenerate a new model candidate (optional)

Use the recorded checkpoint, exact 60-image calibration corpus, copied export
script, and recorded export toolchain. Verify the checkpoint and image hashes
against `model/manifest.json`, and check tool versions against that manifest and
`dependencies.lock.json`. Keep the candidate separate from the original model
and working firmware. The export output directory must not already exist:

```sh
cd "$AOI_MLEK"
python scripts/py/export_calibrated_mobilenet_u85.py \
  --calibration /absolute/path/to/CalImages \
  --checkpoint /absolute/path/to/mobilenet_v2-7ebf99e0.pth \
  --output "$AOI_MLEK/resources_downloaded/img_class/aoi-cal60-candidate"
export AOI_CANDIDATE="$AOI_MLEK/resources_downloaded/img_class/aoi-cal60-candidate/mv2_cal60_imagenet_v2_ethos-u85-256_shared_sram.pte"
shasum -a 256 "$AOI_CANDIDATE"
```

The exporter requires exactly 60 images, uses seed 0, per-channel INT8 weights,
per-tensor INT8 activations, `ethos-u85-256`, `Shared_Sram`, and
`scripts/vela/ensemble_vela.ini` with `Ethos_U85_SRAM_MRAM`.

**Re-export is not guaranteed to be byte-for-byte reproducible.** The PTE can
contain configuration paths and intermediate-output paths that differ between
checkouts. Export tooling and environment differences can also change the
artifact. Even with the recorded checkpoint and calibration images, a new PTE
may therefore have a different SHA-256. A different hash alone neither proves
nor disproves equivalent inference behavior.

Record the candidate's SHA-256, file size, input hashes, source revisions,
resolved package versions, export command, and generated evaluation report.
Before proposing it for acceptance, validate and retain evidence for:

- **Input/output contracts:** tensor shapes, layouts and dtypes, 224 × 224 RGB
  preprocessing and normalization, 1,000-class output, and ImageNet label order.
- **Quantization:** per-channel weight and per-tensor activation settings,
  boundary scales and zero points, numerical ranges, and observed saturation.
- **TOSA behavior:** compare the candidate's pre-Vela TOSA reference outputs
  with FP32 and PT2E outputs on the same recorded inputs. The exporter's four
  sample-image checks alone do not validate TOSA or compiled-NPU behavior.
- **Memory requirements:** activation arena, runtime buffers, model storage,
  link-map placement, and SRAM/MRAM fit for the AppKit E8 HP target.
- **Hardware inference:** execute the compiled candidate on the E8/Ethos-U85
  and check outputs, latency, camera preprocessing, mode changes, and LED behavior.
  TOSA reference agreement is not a substitute for this hardware validation.

The unchanged `config/appkit-e8.cmake` **rejects a candidate whose hash differs
from the original artifact**. Do not silently disable, bypass, or replace the
fixed hash check. Accepting a separately validated candidate requires an
**explicit, reviewed update** to the expected model hash or a separate build
configuration, together with its provenance and validation evidence. Preserve
the original hash and artifact record when documenting that update. Any build
needed for candidate hardware validation likewise requires an explicit candidate
configuration change; the commands in workflow A accept only the original hash.

This optional workflow creates a new candidate for evaluation; it does not
reproduce the original PTE or automatically qualify it for the firmware build.

## Deployment after building

Use the E8 board's Alif SETOOLS instructions and the pinned upstream
`ML_Embedded_Evaluation_Kit.md`, section **Running the Applications Standalone
without Debuggers**. The supplied `deployment/alif_img_class_hp.json` is the
upstream HP image template: it references `mlek_alif_img_class.bin`, M55_HP,
MRAM address `0x80008000`, and a separately supplied `app-device-config.json`.
It is not a complete board provisioning package.

1. Copy the generated sector `mram.bin` into the SETOOLS `build/images/` area as
   `mlek_alif_img_class.bin`; place the matching device configuration alongside
   it as required by that SETOOLS release. Copy the template into `build/config/`.
2. Check the device SKU, image address, board configuration, and boot selection
   against the generated link map and your board's setup. Use the board manual's
   programming UART/USB selection for SETOOLS.
3. In the SETOOLS directory, generate the package with
   `./app-gen-toc -f build/config/alif_img_class_hp.json`, then program with
   `./app-write-mram -p` following that release's serial-port selection procedure.
4. Select the application UART4 connection if monitoring logs, reset the board,
   and perform the demo above. A debugger may instead load the `.axf` with the
   appropriate E8 HP target setup.

These deployment commands are documentation only; they were not executed during
submission preparation. The original calibrated PTE is included; firmware
binaries and external-flash images are not included.

## Model licensing status

The included PTE derives from torchvision MobileNetV2 `IMAGENET1K_V2` weights.
TorchVision, ExecuTorch and torchao software carry BSD-style licenses; the
installed Vela tooling and this repository's root license are Apache-2.0.
The Alif application sources also carry their preserved Alif-specific terms
in `licenses/License.txt`. Those software licenses alone do not establish
permission to redistribute the pretrained weights or this derived PTE.
[TorchVision's model documentation](https://docs.pytorch.org/vision/0.25/models.html#general-information-on-pre-trained-weights)
notes that pretrained models may have separate terms associated with their
training data. Weight-specific redistribution permission has not been confirmed
by this packaging review. Local inclusion does not resolve that question;
confirm applicable terms before publishing the binary. Calibration photographs
remain excluded.
