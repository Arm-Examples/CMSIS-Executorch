# ExecuTorch on Ethos-U: pico-faces

This example shows how to deploy and run an
[ExecuTorch](https://github.com/pytorch/executorch) model on an Arm Ethos-U NPU.
The model is [pico-faces](https://github.com/cpldcpu/pico-faces) by cpldcpu: a
latent rectified-flow diffusion transformer (DiT, 128 x 8 layers, 2.5M
parameters) plus a small convolutional decoder that generate 128x128 RGB faces
in five classes (gender x smile, plus unconditional). Upstream runs it through a
hand-written int8 engine on a Raspberry Pi Pico 2; here its float checkpoints
are exported through the ExecuTorch Arm backend and every layer runs on the
Ethos-U85, while the Cortex-M drives the sampling loop.
The pack [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/)
provides the source code components to build the ExecuTorch runtime, required operators, and Ethos-U backend.
The build process uses the [CMSIS-Toolbox 2.15.0](https://open-cmsis-pack.github.io/cmsis-toolbox/) or higher.

This example application targets the Arm Corstone-320 reference platform with
an Ethos-U85 NPU, simulated on the Arm FVP, and the
[Alif Ensemble E8 DevKit](https://alifsemi.com/support/kits/ensemble-e8devkit/),
real hardware with the same NPU. It demonstrates the same overall workflow used
for other Ethos-U systems: export and quantize a PyTorch model, delegate it to
Ethos-U, select only the required runtime components, and build it into an
embedded application.

| Target-type | Hardware | Run/debug through |
|-------------|----------|-------------------|
| `SSE-320-U85` | Corstone-320 (Cortex-M85 + Ethos-U85), FVP | `FVP_Corstone_SSE-320` |
| `DevKit-E8` | Alif Ensemble E8 DevKit (Cortex-M55 HP core + Ethos-U85) | On-board J-Link, UART console |

The zero-to-running setup for the DevKit-E8, from installing Keil Studio to
the first debug session, is the [getting-started README](../README.md).

## What the example demonstrates

- Exporting a generative model with two methods (`dit_step` and `decode`) into
  one ExecuTorch program, quantized to 16-bit activations and 8-bit weights
  (the decoder to 8-bit activations) and calibrated on real sampling
  trajectories, with both methods fully delegated to the Ethos-U85.
- A runner (`src/app_main.cpp`) that loads both methods once and runs the
  rectified-flow Euler sampler with classifier-free guidance around them; on
  the DevKit it shows every face on the board's LCD, generates new ones on the
  joystick, and serves pico-faces' serial protocol so the upstream viewer
  displays them on the host too.
- Exporting an ExecuTorch model for Ethos-U from a Python virtual environment.
- The pack [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/) links only the required and operator components that the ML model needs.
- Managing the NPU and Vela configuration in the CMSIS solution project rather than duplicating it in the Python exporter.
- A three-step flow with a clean hand-over from the CMSIS-Toolbox to an MLOps system: the toolbox describes the target in a `*.cbuild-mlops.yml` file, a script turns that into the AI layer, and the toolbox builds the application.
- Running the finished application on a Corstone-320 FVP simulation model or on
  the Alif Ensemble E8 DevKit, switching between them by target-type only.

## Prerequisites

- Python `>=3.10,<3.14` (tosa-tools 2026.5.0 has no 3.14 wheels).
- [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack) from the VS Code marketplace.
- Tools listed in [`vcpkg-configuration.json`](../vcpkg-configuration.json).
- Keil Studio manages the required license; the free Keil MDK Community edition can be used for evaluation.
- [Python extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python).

The pack [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/) can be optionally installed manually with:

```bash
cpackget add PyTorch::ExecuTorch@1.5.1
```

> [!Note]
> The pack and Python exporter versions must match, as the generated `.pte`
> format is consumed by the runtime supplied in `PyTorch::ExecuTorch@1.5.1`.
> The matching wheel is `executorch==1.5.1` from PyPI, pinned in
> `requirements.txt`.

## Quick start

The example can be built and run entirely in Keil Studio for VS Code.

1. Install [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack) and [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) from the VS Code marketplace.
2. Clone or download this repository, then open its folder in VS Code.
3. Before using the example for the first time, select **Terminal > Run Task >
   Setup Python virtual environment**. Wait for the task to create the `.venv`
   environment, install the packages required to export the model and
   download the pico-faces checkpoints (about 60 MB) into `model/pico_faces/`.
   (The **(uv)** variant of the task uses [uv](https://docs.astral.sh/uv/)
   instead of pip and can download the Python version it asks for.)
4. Select **Terminal > Run Task > Create AI layer**. This exports the model for
   the NPU of the active target and writes the `ai_layer/` directory. It reads
   `cmsis-executorch.cbuild-mlops.yml`, which the extension writes when the
   solution is loaded or built (`cbuild setup` on the command line). The
   model data (`ai_layer/model_pte.c`, 2.8 MB of program as a 14 MB C source)
   is generated, not committed, so this step is needed once per checkout and
   again after changing the model.
5. Use the CMSIS action buttons to build the application, then select **Run** or
   **Debug**. Keil Studio starts the Corstone-320 FVP automatically. On macOS,
   where Arm ships no FVP build, `.vscode/fvp.sh` runs the model in Docker:
   Docker Desktop must be running, and the first Run or Debug builds the
   container image (about 100 MB download). On Windows, set `model:` in the
   csolution's target-set back to `FVP_Corstone_SSE-320` (the shim is a bash
   script).

For the Alif Ensemble E8 DevKit, choose the `DevKit-E8` target-type in
**Manage Solution** and follow the [getting-started README](../README.md) for the
one-time board preparation (SETOOLS, switches, J-Link).

A successful run prints the Ethos-U configuration, the loaded methods, the
timings of the boot demo (seed 3, 4 steps, class 1, guidance 4), the CRC-32
of the image, a coarse ASCII preview and a pass result. On the Corstone-320
FVP:

```text
Ethos-U version info:
    Arch:       v2.0.0
    MACs/cc:    256
    Cmd stream: v1
ExecuTorch pico-faces (m3_long_cfg, a16w8/a8w8): 2796016 byte program
Methods:
  dit_step  2 input(s), 1 output(s), 16384 planned byte(s)
  decode    1 input(s), 1 output(s), 245760 planned byte(s)
Generating: seed 3, 4 steps, class 1, w 4.0
  dit_step: 8 call(s), 835 ms total (104 ms each)
  decode:   41 ms
  total:    953 ms at 25 MHz (wall clock; not meaningful on the FVP)
  dit_step: NPU 20040 kcycles, active 99%, MAC active 52%, 44 MAC/cycle, read 21279 kB on AXI0 + 0 kB on AXI1
  decode:   NPU 617 kcycles, active 99%, MAC active 78%, 175 MAC/cycle, read 434 kB on AXI0 + 0 kB on AXI1
Image: 128x128x3, CRC32 6b938c66
  |::::::::.......  .    ..........|
  |::::::::...........     .....   |
  |:.:::::::...:---:::..     ....  |
  |:.::...:..:-==+++=-:............|
  ...
Result files in out: written
Test_result: PASS
```

The CRC is deterministic for a given program and toolchain: the same image
comes out of every run, and the DevKit-E8 produces the identical image. It
matches the host's fake-quant rendering of the same seed at 48.4 dB PSNR and
the float model at 44.0 dB (`--compare`, see step 5 below). On the DevKit-E8
the face takes 78 ms at 400 MHz: 8.5 ms per `dit_step` (27.1 M NPU cycles for
the eight calls, NPU active 95%, MAC array active 39%) and 3 ms for `decode`
(1.3 M cycles).

The two `NPU` lines come from the Ethos-U performance monitoring unit, read
by the runner around every method call: NPU clock cycles, the share of them
the NPU and its MAC array were busy, the average MACs per cycle derived from
the model's MAC count (`PF_MACS_*` in `model_params.h`), and the data read on
the NPU's two AXI ports. The DiT keeps the NPU busy but its MAC array only
40 to 50 percent of the time: most of the rest is the elementwise work of the
16-bit graph. Vela's per-operator estimate (`AI_LAYER_VELA_FLAGS=--verbose-performance`)
attributes the largest share to the RESCALE operators around every 16-bit
add and multiply, followed by the multiplies themselves, the softmax
max-reduction and transposes; the convolutions and matrix products are about
a quarter. The weights, 2.6 MB per `dit_step` pass, are streamed from DDR
on the FVP and from MRAM on the DevKit concurrently with that work. The
FVP's NPU cycle counts are its performance model's estimates; the DevKit's
come from the silicon.

### Command-line build

The same workflow from the VS Code Terminal (or any shell with the tools from
`vcpkg-configuration.json` on the path) is three commands plus the one-time
venv setup.

#### 0. Create the Python environment (once)

On Linux or macOS:

```bash
./setup_venv.sh
```

On Windows:

```powershell
.\setup_venv.bat
```

The setup script creates `.venv/`, installs the packages required to
quantize and export the model, and downloads the three pico-faces checkpoint
files from a pinned upstream commit into `model/pico_faces/m3_long_cfg/`,
verifying their SHA-256. It is safe to run again; use `--recreate` when you
want a completely new environment, `--skip-download` to leave the checkpoints
alone, or `--download-only` to fetch just the checkpoints. With
`PICO_FACES_DIR` pointing at a local pico-faces clone, its `checkpoints/`
directory is used instead of the download. The wrappers use `python3` (`python` on
Windows); point them at another interpreter with `PYTHON=python3.12 ./setup_venv.sh`.

With [uv](https://docs.astral.sh/uv/getting-started/installation/) on `PATH`,
`./setup_venv.sh --uv --python 3.12` (or `.\setup_venv.bat ...`) creates the
environment with `uv venv` for that Python version, downloading the interpreter
if needed, and installs with `uv pip`. Add `--recreate` to change the Python
version of an existing environment.

> [!Note]
> On Windows, enable long-path support or keep the repository close to the drive
> root. PyTorch packages can otherwise exceed the legacy 260-character path limit.

#### 1. Generate the MLOps information

```bash
cbuild setup cmsis-executorch.csolution.yml --active SSE-320-U85 --packs
```

This resolves the packs and the active target and writes
`cmsis-executorch.cbuild-mlops.yml`: the processor, NPU and Vela
settings of the target, and the location of the AI layer. (`--packs`
installs missing packs and is only needed on a fresh checkout; the layers'
RTE configuration is committed, so no `--update-rte` is required.) Use
`--active DevKit-E8` in this and the following commands to build for the
Alif Ensemble E8 DevKit.

#### 2. Create the AI layer

```bash
python3 create_ai_layer.py cmsis-executorch.cbuild-mlops.yml
```

This is the MLOps step. The script reads the NPU and Vela settings from the
file, quantizes and exports the two methods of `model/model.py` for that NPU
(calibrating each on data the model provides), and writes the complete AI
layer into `ai_layer/`: the component selection, the model as a C array and
`model_params.h` with the sampling schedule and the conditioning table. It
prints, per method, the number of Ethos-U delegates and the operators left on
the CPU; for this model both methods are one delegate each, with only the
quantize/dequantize boundary on the CPU. It runs itself in `.venv` when
started with another interpreter (use `python` on Windows) and takes a few
minutes (calibration plus Vela).

#### 3. Build the application

```bash
cbuild cmsis-executorch.csolution.yml --active SSE-320-U85
```

A plain CMSIS build; no Python is involved. The resulting image is:

```text
out/cmsis-executorch/SSE-320-U85/Debug/cmsis-executorch.axf
```

#### 4. Run on the FVP

```bash
.vscode/fvp.sh \
    -f board/Corstone-320/fvp_config.txt --simlimit 60 \
    -a out/cmsis-executorch/SSE-320-U85/Debug/cmsis-executorch.axf
```

`.vscode/fvp.sh` is the model command the Run and Debug buttons use too; on
Linux and Windows `FVP_Corstone_SSE-320` can be called directly with the same
arguments. The application generates one face (seed 3, 4 steps, class 1,
guidance 4), prints the timings, the CRC-32 of the image and a coarse ASCII
preview, writes the image and the measurements to `out/fvp_image.bin` and
`out/fvp_result.txt` through semihosting, and ends the simulation. A run
takes a few minutes.

#### 5. Check the result on the host

```bash
python3 model/verify_export.py --stage fakequant --seed 3 --class 1 --w 4 --k 4 --out out/pf_fakequant.png
python3 model/verify_export.py --compare out/fvp_image.bin out/pf_fakequant.png
```

The host stages use the firmware's noise generator, so the same seed, class,
guidance and step count give the same face as on the FVP and the board.
`--stage float` samples with the float model, `--stage fakequant` with the
quantized graphs the export produces (what the NPU is expected to compute);
`--compare` prints the PSNR between two images (PNGs, or the raw frame the
FVP writes), and `--min-psnr` turns it into a check, as in CI. `--stage tosa`
executes both methods through the TOSA reference model, the integer
semantics the Ethos-U implements, and reports the deviation from the
fake-quant graphs: the check that catches 16-bit lowering problems before a
build. `--check-upstream <clone>` verifies the re-implementation in
`model/model.py` against the original pico-faces modules.

#### 6. Generate faces on the DevKit-E8

On the board the runner keeps going after the boot demo. Every face appears
on the LCD; the SW2 joystick generates new ones (left: one image; right:
start or stop generating back to back). The runner also serves image
requests on the UART console in the serial protocol of pico-faces' firmware,
so pico-faces' viewer works unchanged (in a clone of the upstream repository,
`pip install pyserial pillow`):

```bash
python viewer/view_serial.py --port /dev/tty.usbmodemXXXX --seed 3 --steps 4 --class 1 --cfg 4 --show
```

Close the Serial Monitor first: the viewer needs the port. `--class` is 0
(female, neutral), 1 (female, smiling), 2 (male, neutral), 3 (male, smiling)
or 4 (unconditional); `--cfg` is the guidance strength (0 = plain);
`--steps` is 8, 4, 2 or 1. A frame takes about 4 s to transfer at 115200
baud, much longer than its generation.

## How model generation works

The target is described by the `mlops:` node in
`cmsis-executorch.csolution.yml`:

```yaml
mlops:
  npu:
    type: Ethos-U85
  vela:
    system: Ethos_U85_SRAM_MRAM
    memory: Shared_Sram
  model:
    clayer: $AI-Layer$
    name: PicoFaces
```

`cbuild setup --active DevKit-E8` resolves it into
`cmsis-executorch.cbuild-mlops.yml`, which contains the processor, NPU
and Vela options. With the Ensemble pack in the solution, the toolbox also
copies the pack's Vela configuration to `.cmsis/ensemble_vela.ini` and adds
`--accelerator-config ethos-u85-256`, for both targets; the system
configuration named above comes from that file. `create_ai_layer.py` reads
those options and passes them to ExecuTorch's `EthosUCompileSpec`, so the
target configuration is never duplicated in Python. The script then writes:

- `ai_layer/ai_layer.clayer.yml`: the CMSIS components required by the model.
- `ai_layer/model_pte.c` and `model_pte.h`: the ExecuTorch program embedded as a C array.
- `ai_layer/model_params.h`: the constants the runner needs (latent and image
  shapes, the schedule, the conditioning vectors per class and step).
- `ai_layer/model.pte`: the program itself, for inspection.

`model/model.py` re-implements the pico-faces modules with the operators the
Ethos-U delegate supports (RMSNorm from primitives, attention on rank-3
tensors, patchify as a strided convolution) and loads the upstream weights by
name. Two details of the graph matter for the 16-bit path and were found with
the TOSA reference model (the host fake-quant reference does not model them):
reductions (the mean in RMSNorm, the softmax row sums) are matrix products
with a constant ones vector instead of `aten.mean`/`aten.sum`, whose
`REDUCE_SUM` has a documented 16-bit issue on the Ethos-U85; and the output
projection stays a Linear with its bias, followed by a transposed convolution
with fixed 0/1 weights for the unpatchify, because a bias add after the
transposed convolution fuses into a RESCALE whose shift is too small for
16-bit accumulators (the reference model reports "value should stay within
[-262144, 262143]", and the NPU output saturates). The conditioning tower
(timestep embedding, class embedding) is evaluated at export time for the 8
schedule points and 5 classes, so the program takes the conditioning vector as
an input and the firmware needs no float operators on the CPU.

| Method | Runs on | Work |
|--------|---------|------|
| `dit_step(z, c) -> v` | Ethos-U85 | 111 MMAC per call; 2 calls per step with guidance |
| `decode(z) -> img` | Ethos-U85 | 109 MMAC |
| Euler update, guidance blend, noise, RGB conversion | Cortex-M | a few thousand float operations |

The DiT runs at `a16w8` (16-bit activations, 8-bit weights): it reproduces
the float model closely, whereas int8 activations (`PICO_FACES_QUANT=a8w8`)
give visibly degraded faces because the DiT's residual stream needs more than
8 bits. The decoder runs at `a8w8` (`PICO_FACES_DECODE_QUANT`): its
convolutions do not need the 16-bit path and run about three times as fast.

See [the MLOps flow](mlops-flow.md) for a detailed walkthrough.

## Component selection

The [ExecuTorch CMSIS Pack](https://www.keil.arm.com/packs/executorch-pytorch/)
provides the runtime, backends, and individual operators as selectable CMSIS
components. `create_ai_layer.py` examines the exported program and writes
`ai_layer/ai_layer.clayer.yml` so only the required components are linked.
Because the layer is complete before the build starts, a model change that
changes the operator set needs nothing more than re-running steps 2 and 3.

## Adapting the example

To use a different model, replace or modify `model/model.py`. A model with
one method needs only `get_model()` and `get_calibration_inputs()` (the
samples the quantizer is calibrated with; give it representative data for a
trained model). Several methods, their calibration data and quantization,
and constants for the firmware come through `get_methods()` and
`get_params()` (see [the MLOps flow](mlops-flow.md)). The runner in
`src/app_main.cpp` is written for pico-faces' two methods, so another model
needs a runner of its own; `EmbeddedModule` (`src/arm_embedded_module.hpp`)
loads and executes methods by name. Then re-run `create_ai_layer.py` and
build. `PICO_FACES_VARIANT=m3_decD_deep_full` exports pico-faces' larger
model (12 layers, 3.7 MB of weights), which does not fit the DevKit's MRAM
region without further memory work.

To target another Ethos-U configuration, update the target and `mlops:`
settings in the CMSIS solution and re-run all three steps. The generated Vela
options then follow that configuration automatically. Moving to a different
board or reference platform also requires the corresponding device pack, board
support, memory layout, and FVP configuration.

When updating ExecuTorch, update the CMSIS pack and Python package versions
together. More information is available in
[pack provenance](pack-provenance.md).

## Project layout

| Path | Purpose |
|------|---------|
| `cmsis-executorch.csolution.yml` | Solution, target, and MLOps configuration |
| `cmsis-executorch.cproject.yml` | Application project: sources plus the Board and AI layers |
| `model/model.py` | pico-faces re-expressed as the ExecuTorch methods `dit_step` and `decode` |
| `model/verify_export.py` | Host checks: float and fake-quant sampling to PNG, TOSA reference model, upstream comparison, PSNR |
| `model/pico_faces/` | Downloaded checkpoints (not committed) |
| `create_ai_layer.py` | Exports the model for the target and writes the AI layer |
| `ai_layer/` | Generated: component selection, `model_params.h` and the embedded model data (`model_pte.c` not committed) |
| `setup_venv.py` (`.sh` / `.bat`) | Creates the Python environment for the export and downloads the checkpoints |
| `.vscode.d/tasks.json` | The VS Code tasks (venv setup, Create AI layer, Alif debug stubs) merged by the CMSIS Solution extension |
| `.vscode/fvp.sh`, `.vscode/fvp.Dockerfile` | The FVP model command used by Run and Debug; runs the model in Docker on macOS |
| `board/Corstone-320/` | Corstone-320 platform support and FVP configuration; the linker scripts in its `RTE/` put the model in DDR |
| `board/DevKit-E8/` | Alif Ensemble E8 DevKit board layer (M55_HP core, Ethos-U85, UART console, LCD, joystick) |
| `.alif/` | SETOOLS configuration and debug stubs for the DevKit-E8 (from the Ensemble pack) |
| `src/app_main.cpp` | Loads the two methods and runs the sampler; board features through `APP_DISPLAY`, `APP_INTERACTIVE`, `APP_BUTTONS`; pool sizes overridable with `APP_METHOD_POOL_SIZE`, `APP_TEMP_POOL_SIZE`, `APP_POOL_SECTION` |
| `src/arm_embedded_module.*` | `EmbeddedModule`: ExecuTorch's `Module` class without the POSIX file loading (BSD-3-Clause, `src/LICENSE-ExecuTorch`) |
| `board/DevKit-E8/README.md` | The DevKit-E8 layer, its memory map and RTE configuration |
| `documentation/mlops-flow.md` | The MLOps flow in detail |
| `documentation/pack-provenance.md` | Where the ExecuTorch pack comes from, how to update it |

## Known limitations

- The supplied platform configurations target Corstone-320 (FVP) and the Alif
  Ensemble E8 DevKit, both with Ethos-U85; another target needs its
  corresponding platform integration.
- The `mlops:` node is solution-wide, so both targets share one exported
  model. That is correct here because both have an Ethos-U85 with 256 MACs;
  the Vela system configuration is the Ensemble pack's, which the FVP runs
  just as well.

## License

The example code is licensed under Apache-2.0; see `LICENSE`. ExecuTorch and
`src/arm_embedded_module.*`, which is derived from it, use a BSD-3-Clause
license; see `src/LICENSE-ExecuTorch`. pico-faces, whose modules
`model/model.py` re-implements and whose checkpoints `setup_venv.py`
downloads, is MIT-licensed; see `model/LICENSE-pico-faces`.

## References

- [pico-faces](https://github.com/cpldcpu/pico-faces)
- [PyTorch ExecuTorch CMSIS Pack](https://www.keil.arm.com/packs/executorch-pytorch/)
- [ExecuTorch](https://github.com/pytorch/executorch)
- [ExecuTorch Arm Ethos-U backend](https://docs.pytorch.org/executorch/main/backends-arm-ethos-u.html)
- [CMSIS-Toolbox MLOps information](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
- [Arm CMSIS documentation](https://arm-software.github.io/CMSIS_6/latest/index.html)
