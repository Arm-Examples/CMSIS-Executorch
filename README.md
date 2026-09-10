# ExecuTorch on Ethos-U

This example shows how to deploy and run an
[ExecuTorch](https://github.com/pytorch/executorch) model on an Arm Ethos-U NPU.
The pack [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/)
provides the source code components to build the ExecuTorch runtime, required operators, and Ethos-U backend.
The build process uses the [CMSIS-Toolbox](https://open-cmsis-pack.github.io/cmsis-toolbox/)
2.14.1+p88 or higher, as bundled with the Keil Studio csolution extension 1.70.1
or higher; see [How model generation works](#how-model-generation-works).

This example application targets the Arm Corstone-320 reference platform with
an Ethos-U85 NPU. It demonstrates the same overall workflow used for other
Ethos-U systems: export and quantize a PyTorch model, delegate it to Ethos-U,
select only the required runtime components, and build it into an embedded
application.

## What the example demonstrates

- Exporting an ExecuTorch model for Ethos-U from a Python virtual environment.
- The pack [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/) links only the required and operator components that the ML model needs.
- Managing the NPU and Vela configuration in the CMSIS solution project rather than duplicating it in the Python exporter.
- A three-step flow with a clean hand-over from the CMSIS-Toolbox to an MLOps system: the toolbox describes the target in a `*.cbuild-mlops.yml` file, a script turns that into the AI layer, and the toolbox builds the application.
- Running the finished application on a Corstone-320 FVP simulation model.

## Prerequisites

- Python `>=3.10,<3.15`.
- [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack) from the VS Code marketplace.
- Tools listed in [`vcpkg-configuration.json`](./vcpkg-configuration.json).
- Keil Studio manages the required license; the free Keil MDK Community edition can be used for evaluation.
- [Python extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python).

The pack [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/) can be optionally installed manually with:

```bash
cpackget add PyTorch::ExecuTorch@1.4.1
```

> [!Note]
> The pack and Python exporter versions must match, as the generated `.pte`
> format is consumed by the runtime supplied in `PyTorch::ExecuTorch@1.4.1`.
> The matching wheel is `executorch==1.4.1` from PyPI, pinned in
> `requirements.txt`.

## Quick start

The example can be built and run entirely in Keil Studio for VS Code.

1. Install [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack) and [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) from the VS Code marketplace.
2. Clone or download this repository, then open its folder in VS Code.
3. Before using the example for the first time, select **Terminal > Run Task >
   Setup Python virtual environment**. Wait for the task to create the `.venv`
   environment and install the packages required to export the model. (The
   **(uv)** variant of the task uses [uv](https://docs.astral.sh/uv/) instead
   of pip and can download the Python version it asks for.)
4. Select **Terminal > Run Task > Create AI layer**. This exports the model for
   the NPU of the active target and writes the `ai_layer/` directory. (The
   repository ships a generated layer, so this step is only needed after
   changing the model or the target.)
5. Use the CMSIS action buttons to build the application, then select **Run** or
   **Debug**. Keil Studio starts the Corstone-320 FVP automatically. On macOS,
   where Arm ships no FVP build, `.vscode/fvp.sh` runs the model in Docker:
   Docker Desktop must be running, and the first Run or Debug builds the
   container image (about 100 MB download). On Windows, set `model:` in the
   csolution's target-set back to `FVP_Corstone_SSE-320` (the shim is a bash
   script).

A successful run prints the Ethos-U configuration, output logits, and a pass
result:

```text
Ethos-U version info:
    Arch:       v2.0.0
    MACs/cc:    256
    Cmd stream: v1
ExecuTorch Ethos-U85 example: 8832 byte model
Output: 10 element(s): 0.0079 0.0459 0.0475 -0.0475 0.0791 0.0411 -0.0285 -0.0744 -0.2246 -0.0016
Test_result: PASS
```

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

The setup script creates `.venv/` and installs the packages required to
quantize and export the model. It is safe to run again; use `--recreate` when
you want a completely new environment. The wrappers use `python3` (`python` on
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
RTE configuration is committed, so no `--update-rte` is required.)

#### 2. Create the AI layer

```bash
python3 create_ai_layer.py cmsis-executorch.cbuild-mlops.yml
```

This is the MLOps step. The script reads the NPU and Vela settings from the
file, quantizes and exports `model/model.py` for that NPU, and writes the
complete AI layer into `ai_layer/`: the component selection and the model as a
C array. It runs itself in `.venv` when started with another interpreter (use
`python` on Windows).

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
arguments.

## How model generation works

The target is described by the `mlops:` node in
`cmsis-executorch.csolution.yml`:

```yaml
mlops:
  npu:
    type: Ethos-U85
  vela:
    system: Ethos_U85_SYS_DRAM_Mid
    memory: Shared_Sram
  model:
    clayer: $AI-Layer$
    name: TinyCNN
    input-shape: 1x3x16x16
    calibration-samples: 2
```

`cbuild setup --active SSE-320-U85` resolves it into
`cmsis-executorch.cbuild-mlops.yml`, which contains the processor, NPU
and Vela options, and the FVP target-set as the `simulator:` to test on.
`create_ai_layer.py` reads those options and passes them to ExecuTorch's
`EthosUCompileSpec`, so the target configuration is never duplicated in
Python. The two extra keys under `model:` are parameters of the model itself;
the toolbox passes them through and the script hands them to `model/model.py`.
This needs CMSIS-Toolbox 2.14.1+p88 or newer; an older toolbox stops at
`error csolution: schema check failed, verify syntax` on those two lines (the
`main` branch of this repository works with the released 2.14.1). The script
then writes:

- `ai_layer/ai_layer.clayer.yml`: the CMSIS components required by the model.
- `ai_layer/model_pte.c` and `model_pte.h`: the ExecuTorch program embedded as a C array.
- `ai_layer/model.pte`: the program itself, for inspection.

See [the MLOps flow](documentation/mlops-flow.md) for a detailed walkthrough.

## Component selection

The [ExecuTorch CMSIS Pack](https://www.keil.arm.com/packs/executorch-pytorch/)
provides the runtime, backends, and individual operators as selectable CMSIS
components. `create_ai_layer.py` examines the exported program and writes
`ai_layer/ai_layer.clayer.yml` so only the required components are linked.
Because the layer is complete before the build starts, a model change that
changes the operator set needs nothing more than re-running steps 2 and 3.

## Adapting the example

To use a different model, replace or modify `model/model.py` and update the
model name or input handling as required: `INPUT_SHAPE` and
`get_model(input_shape)` define the input, and
`get_calibration_inputs(input_shape, calibration_samples)` returns the samples
the quantizer is calibrated with; give it representative data for a trained
model. Then re-run `create_ai_layer.py` and build.

To target another Ethos-U configuration, update the target and `mlops:`
settings in the CMSIS solution and re-run all three steps. The generated Vela
options then follow that configuration automatically. Moving to a different
board or reference platform also requires the corresponding device pack, board
support, memory layout, and FVP configuration.

When updating ExecuTorch, update the CMSIS pack and Python package versions
together. More information is available in
[pack provenance](documentation/pack-provenance.md).

## Project layout

| Path | Purpose |
|------|---------|
| `cmsis-executorch.csolution.yml` | Solution, target, and MLOps configuration |
| `cmsis-executorch.cproject.yml` | Application project: sources plus the Board and AI layers |
| `model/model.py` | Example TinyCNN model |
| `create_ai_layer.py` | Exports the model for the target and writes the AI layer |
| `ai_layer/` | Generated: component selection and the embedded model data |
| `setup_venv.py` (`.sh` / `.bat`) | Creates the Python environment for the export |
| `.vscode.d/tasks.json` | The VS Code tasks (venv setup, Create AI layer) merged by the CMSIS Solution extension |
| `.vscode/fvp.sh`, `.vscode/fvp.Dockerfile` | The FVP model command used by Run and Debug; runs the model in Docker on macOS |
| `board/Corstone-320/` | Corstone-320 platform support and FVP configuration |
| `src/app_main.cpp` | Loads the model, runs inference, and prints the result; pool sizes overridable with `APP_METHOD_POOL_SIZE`, `APP_TEMP_POOL_SIZE`, `APP_POOL_SECTION` |
| `src/arm_embedded_module.*` | `EmbeddedModule`: ExecuTorch's `Module` class without the POSIX file loading (BSD-3-Clause, `src/LICENSE-ExecuTorch`) |
| `documentation/mlops-flow.md` | The MLOps flow in detail |
| `documentation/pack-provenance.md` | Where the ExecuTorch pack comes from, how to update it |

## Known limitations

- The supplied platform configuration targets Corstone-320 with Ethos-U85;
  another target needs its corresponding platform integration.

## License

The example code is licensed under Apache-2.0; see `LICENSE`. ExecuTorch and
`src/arm_embedded_module.*`, which is derived from it, use a BSD-3-Clause
license; see `src/LICENSE-ExecuTorch`.

## References

- [PyTorch ExecuTorch CMSIS Pack](https://www.keil.arm.com/packs/executorch-pytorch/)
- [ExecuTorch](https://github.com/pytorch/executorch)
- [ExecuTorch Arm Ethos-U backend](https://docs.pytorch.org/executorch/main/backends-arm-ethos-u.html)
- [CMSIS-Toolbox MLOps information](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
- [Arm CMSIS documentation](https://arm-software.github.io/CMSIS_6/latest/index.html)
