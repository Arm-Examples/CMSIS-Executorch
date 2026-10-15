# ExecuTorch on Ethos-U

[![Build and test Ethos-U85-256](https://github.com/Arm-Examples/CMSIS-Executorch/actions/workflows/test-cmsis-executorch.yml/badge.svg)](https://github.com/Arm-Examples/CMSIS-Executorch/actions/workflows/test-cmsis-executorch.yml)
[![Build and test Ethos-U55-128](https://github.com/Arm-Examples/CMSIS-Executorch/actions/workflows/test-cmsis-executorch-u55.yml/badge.svg)](https://github.com/Arm-Examples/CMSIS-Executorch/actions/workflows/test-cmsis-executorch-u55.yml)
[![Test Python environment](https://github.com/Arm-Examples/CMSIS-Executorch/actions/workflows/test-python-environment.yml/badge.svg)](https://github.com/Arm-Examples/CMSIS-Executorch/actions/workflows/test-python-environment.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![CMSIS-Toolbox: 2.15](https://img.shields.io/badge/CMSIS--Toolbox-2.15-blue.svg)](https://open-cmsis-pack.github.io/cmsis-toolbox/)
[![ExecuTorch: 1.5.1](https://img.shields.io/badge/ExecuTorch-1.5.1-blue.svg)](https://www.keil.arm.com/packs/executorch-pytorch/)

This repository is a template for creating embedded applications that deploy
and run an [ExecuTorch](https://github.com/pytorch/executorch) ML model on an Arm
Ethos-U NPU.

- The [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/)
  pack provides the ExecuTorch runtime, required operators, and Ethos-U backend.
- The [`ARM::CMSIS-Ethos-U`](https://www.keil.arm.com/packs/cmsis-ethos-u-arm/overview/)
  pack provides Ethos-U drivers, configuration support, Corstone reference
  integrations and examples, and documentation for deploying ML models on
  Ethos-U.
- The build process uses [CMSIS-Toolbox 2.15.0](https://open-cmsis-pack.github.io/cmsis-toolbox/)
  or higher.

The template can be used for applications targeting any Ethos-U configuration;
the default solution targets the Arm Corstone-320 reference platform with an
Ethos-U85 NPU. The same workflow applies to other ML models, targets, and
applications: export and quantize an ML model developed in PyTorch, delegate it
to Ethos-U, select only the required runtime components, and build it into an
embedded application.

## What the template provides

- Exporting an ExecuTorch ML model for Ethos-U from a Python virtual environment.
- The pack [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/) links only the required and operator components that the ML model needs.
- Managing the NPU and Vela configuration in the CMSIS solution project rather than duplicating it in the Python exporter.
- A three-step flow with a clean hand-over from the CMSIS-Toolbox to an MLOps system: the toolbox describes the target in a `*.cbuild-mlops.yml` file, a script turns that into the AI layer, and the toolbox builds the application.
- Running the finished application on a Corstone-320 FVP simulation model.

## Prerequisites

- [Git](https://git-scm.com/) to clone the repository created from this template.
- [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack) from the VS Code marketplace.
- [Python extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python).
- [Astral's uv](https://docs.astral.sh/uv/getting-started/installation/) on `PATH` for the
  recommended environment task, or an installed Python `>=3.10,<3.14` for pip.
- On macOS, follow [FVPs on macOS](https://github.com/Arm-Examples/FVPs-on-Mac)
  to install Docker Desktop and configure the FVP environment.

Keil Studio manages the required license; the free Keil MDK Community edition
can be used for evaluation.

> [!NOTE]
> On Windows, enable [long-path support](https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation#enable-long-paths-in-windows-10-version-1607-and-later).

## Quick start

The steps below build the default Corstone-320 / Ethos-U85-256 application
using `cmsis-executorch.csolution.yml`. The alternative
`cmsis-executorch-u55.csolution.yml` targets Corstone-300 / Ethos-U55-128.
Both can be built and run entirely in Keil Studio for VS Code.

### 1. Create the application

On [GitHub](https://github.com/Arm-Examples/CMSIS-Executorch), select **Use this
template** to create your application repository, then clone it and open its
folder in VS Code. In the CMSIS view, select **... > Select Active Solution
from Workspace**, then choose `cmsis-executorch.csolution.yml` for this Quick
start.

### 2. Set up the project tools

When the folder opens, Arm Environment Manager installs the tools listed in
`vcpkg-configuration.json`.

To set up ML model export, select **Terminal > Run Task > Create Python environment
(uv)**. This task uses [Astral's uv](https://docs.astral.sh/uv/) from `PATH` and can
download a supported Python (`>=3.10,<3.14`) before installing the dependencies.
Alternatively, **Create Python environment (pip)** uses a compatible Python
already installed on the host. Python 3.14 is not supported because
tosa-tools 2026.5.0 has no 3.14 wheels.

The [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/)
pack is resolved through the Arm pack service. To install it manually instead:

```bash
cpackget add PyTorch::ExecuTorch@1.5.1
```

> [!IMPORTANT]
> The pack and [ExecuTorch Python exporter](https://docs.pytorch.org/executorch/stable/getting-started.html)
> versions must match, as the generated `.pte` is consumed by the runtime in
> `PyTorch::ExecuTorch@1.5.1`. The matching `executorch==1.5.1` wheel is pinned
> in `requirements.txt`.

### 3. Build and run

The repository contains the generated U85 AI layer, so the default solution
builds immediately. After changing the ML model or target, select **Terminal > Run
Task > Create AI layer**. The task exports the ML model for the active solution's
NPU from its `*.cbuild-mlops.yml` file and rewrites `ai_layer/`.

To use Ethos-U55-128 instead, select
`cmsis-executorch-u55.csolution.yml` as the active solution and run **Create AI
layer** before building.

Use the CMSIS action buttons to build the application, then select **Run** or
**Debug**. Keil Studio starts the selected FVP simulation model automatically.
On macOS, where Arm ships no native FVP, set `model:` in the active csolution's
target-set to `${workspaceFolder}/.vscode/fvp.sh` and copy
`.vscode/launch.json.mac` to
`.vscode/launch.json`. Docker Desktop must be running; the first Run or Debug
builds the container image (about 100 MB download).

A successful run prints the Ethos-U configuration, output logits, and a pass
result:

```text
Ethos-U version info:
    Arch:       v2.0.0
    MACs/cc:    256
    Cmd stream: v1
ExecuTorch Ethos-U85-256 application: 8720 byte model
Output: 10 element(s): -0.0142 0.0206 0.0601 -0.0886 0.0791 0.0522 -0.0301 -0.0775 -0.2800 -0.0095
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
quantize and export the ML model. It is safe to run again; use `--recreate` when
you want a completely new environment. The wrappers use `python3` (`python` on
Windows); point them at another interpreter with `PYTHON=python3.12 ./setup_venv.sh`.

With [Astral's uv](https://docs.astral.sh/uv/getting-started/installation/) on `PATH`,
`./setup_venv.sh --uv --python 3.12` (or `.\setup_venv.bat ...`) creates the
environment with `uv venv` for that Python version, downloading the interpreter
if needed, and installs with `uv pip`. Add `--recreate` to change the Python
version of an existing environment.

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
complete AI layer into `ai_layer/`: the component selection and the ML model as a
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
FVP_Corstone_SSE-320 \
    -f board/Corstone-320/fvp_config.txt --simlimit 60 \
    -a out/cmsis-executorch/SSE-320-U85/Debug/cmsis-executorch.axf
```

This is the FVP command the Run and Debug buttons use too. On macOS, call
`.vscode/fvp.sh` with the same arguments; it runs the FVP simulation model in
Docker.

## How ML model generation works

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
    input-shape: 1x16x16x3
    calibration-samples: 2
```

`cbuild setup --active SSE-320-U85` resolves it into
`cmsis-executorch.cbuild-mlops.yml`, which contains the processor, NPU
and Vela options, and the FVP simulation target-set as the `simulator:` to test
on.
`create_ai_layer.py` reads those options and passes all of them to ExecuTorch's
`EthosUCompileSpec`, so the target configuration is never duplicated in
Python. The two extra keys under `model:` are parameters of the ML model itself;
the toolbox passes them through and the script hands them to `model/model.py`.
The script then writes:

- `ai_layer/ai_layer.clayer.yml`: the CMSIS components required by the ML model.
- `ai_layer/model_pte.c` and `model_pte.h`: the ExecuTorch program embedded as a C array.
- `ai_layer/model.pte`: the program itself, for inspection.

See [the MLOps flow](documentation/mlops-flow.md) for a detailed walkthrough.

## Component selection

The [ExecuTorch pack](https://www.keil.arm.com/packs/executorch-pytorch/)
provides the runtime, backends, and individual operators as selectable CMSIS
components. `create_ai_layer.py` examines the exported program and writes
`ai_layer/ai_layer.clayer.yml` so only the required components are linked.
Because the layer is complete before the build starts, an ML model change that
changes the operator set needs nothing more than re-running steps 2 and 3.

## Adapting the template

To use a different ML model, replace or modify `model/model.py` and update the
ML model name or input handling as required: `get_model(input_shape)` builds the
ML model for the `input-shape` of the `mlops:` node, and
`get_calibration_inputs(input_shape, calibration_samples)` returns the samples
the quantizer is calibrated with; give it representative data for a trained
ML model. The runner in `src/app_main.cpp` builds its input tensor with the same
fixed shape and prints the output as ten floats, so an ML model with another
input shape or output needs matching changes there. Then re-run
`create_ai_layer.py` and build.

Give a vision ML model its input as NHWC, the layout of a camera frame, as
`model/model.py` does: the ML model permutes it to the NCHW its convolutions take,
and Vela folds that into the first convolution. With an NCHW input the NPU
first reorders the image in an operation of its own.

To target another Ethos-U configuration, update the target and `mlops:`
settings in the CMSIS solution and re-run all three steps. The generated Vela
options then follow that configuration automatically; `vela: misc:` takes any
further Vela option, for example `--optimise Size`. Moving to a different
board or reference platform also requires the corresponding device pack, board
support, memory layout, and FVP configuration.

When updating ExecuTorch, update the pack and Python package versions
together. More information is available in
[ExecuTorch pack maintenance and versioning](documentation/executorch-pack-maintenance.md).

## Continuous integration

Two GitHub Actions workflows validate the supplied Ethos-U configurations on
every push and pull request to `main`. They create the Python environment,
generate the target-specific AI layer, build the Corstone-320/Ethos-U85-256 and
Corstone-300/Ethos-U55-128 solutions with AC6, GCC, and Clang, run each
application on its FVP simulation model, verify the reported test result, and
upload the build artifacts. A separate workflow verifies the Python environment
setup on Linux, macOS, and Windows. The badges at the top of this README show
the current status of all three workflows.

## Project layout

| Path | Purpose |
| ------ | --------- |
| `cmsis-executorch.csolution.yml` | Corstone-320 / Ethos-U85 solution and MLOps configuration |
| `cmsis-executorch-u55.csolution.yml` | Corstone-300 / Ethos-U55-128 solution and MLOps configuration |
| `cmsis-executorch.cproject.yml` | Application project: sources plus the Board and AI layers |
| `model/model.py` | Starter TinyCNN ML model |
| `create_ai_layer.py` | Exports the ML model for the target and writes the AI layer |
| `ai_layer/` | Generated: component selection and the embedded ML model data |
| `setup_venv.py` (`.sh` / `.bat`) | Creates the Python environment for the export |
| `.vscode.d/tasks.json` | The VS Code tasks (venv setup, Create AI layer) merged by the CMSIS Solution extension |
| `.vscode/fvp.sh`, `.vscode/fvp.Dockerfile`, `.vscode/launch.json.mac` | macOS only: runs the FVP in Docker, and the debug configuration that waits for it |
| `board/Corstone-300/`, `board/Corstone-320/` | Platform support and FVP configurations |
| `src/app_main.cpp` | Loads the ML model, runs inference, and prints the result; pool sizes overridable with `APP_METHOD_POOL_SIZE`, `APP_TEMP_POOL_SIZE`, `APP_POOL_SECTION` |
| `src/arm_embedded_module.*` | `EmbeddedModule`: ExecuTorch's `Module` class without the POSIX file loading (BSD-3-Clause, `src/LICENSE-ExecuTorch`) |
| `documentation/mlops-flow.md` | The MLOps flow in detail |
| `documentation/executorch-pack-maintenance.md` | Where the ExecuTorch pack comes from, how to update it |

## Usage notes

The generated `ai_layer/` is target-specific. After switching targets, run
`create_ai_layer.py` with the selected solution's `*.cbuild-mlops.yml`.

## License

The template code is licensed under Apache-2.0; see `LICENSE`. ExecuTorch and
`src/arm_embedded_module.*`, which is derived from it, use a BSD-3-Clause
license; see `src/LICENSE-ExecuTorch`.

## References

- [PyTorch ExecuTorch pack](https://www.keil.arm.com/packs/executorch-pytorch/)
- [ExecuTorch](https://github.com/pytorch/executorch)
- [ExecuTorch Arm Ethos-U backend](https://docs.pytorch.org/executorch/main/backends-arm-ethos-u.html)
- [CMSIS-Ethos-U documentation](https://arm-software.github.io/CMSIS-Ethos-U/)
- [CMSIS-Ethos-U pack](https://www.keil.arm.com/packs/cmsis-ethos-u-arm/overview/)
- [CMSIS-Toolbox MLOps information](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
- [Arm CMSIS documentation](https://arm-software.github.io/CMSIS_6/latest/index.html)
