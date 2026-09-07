# ExecuTorch on Ethos-U

This example shows how to deploy and run an
[ExecuTorch](https://github.com/pytorch/executorch) model on an Arm Ethos-U NPU.
The pack [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/)
provides the source code components to build the ExecuTorch runtime, required operators, and Ethos-U backend.
The build process uses the [CMSIS-Toolbox 2.14.1](https://open-cmsis-pack.github.io/cmsis-toolbox/) or higher.

This example application targets the Arm Corstone-320 reference platform with
an Ethos-U85 NPU. It demonstrates the same overall workflow used for other
Ethos-U systems: export and quantize a PyTorch model, delegate it to Ethos-U,
select only the required runtime components, and build it into an embedded
application.

## What the example demonstrates

- Exporting an ExecuTorch model for Ethos-U using a Python virtual environment (without requiring Docker).
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
cpackget add PyTorch::ExecuTorch@1.4.0
```

> [!Note]
> The pack and Python exporter versions must match, as the generated `.pte`
> format is consumed by the runtime supplied in `PyTorch::ExecuTorch@1.4.0`.

## Quick start

The example can be built and run entirely in Keil Studio for VS Code.

1. Install [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack) and [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) from the VS Code marketplace.
2. Clone or download this repository, then open its folder in VS Code.
3. Before using the example for the first time, select **Terminal > Run Task >
   Setup Python virtual environment**. Wait for the task to create the `.venv`
   environment and install the packages required to export the model.
4. Select **Terminal > Run Task > Create AI layer**. This exports the model for
   the NPU of the active target and writes the `ai_layer/` directory. (The
   repository ships a generated layer, so this step is only needed after
   changing the model or the target.)
5. Use the CMSIS action buttons to build the application, then select **Run** or
   **Debug**. Keil Studio starts the Corstone-320 FVP automatically.

A successful run prints the Ethos-U configuration, output logits, and a pass
result:

```text
Ethos-U version info:
    Arch:       v2.0.0
    MACs/cc:    256
    Cmd stream: v1
ExecuTorch Ethos-U85 example: 8864 byte model
Output: 10 element(s): 0.0082 0.0489 0.0489 -0.0489 0.0857 ...
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
you want a completely new environment.

> [!Note]
> On Windows, enable long-path support or keep the repository close to the drive
> root. PyTorch packages can otherwise exceed the legacy 260-character path limit.

#### 1. Generate the MLOps information

```bash
cbuild setup cmsis-executorch-simple.csolution.yml --active SSE-320-U85 --packs --update-rte
```

This resolves the packs and the active target and writes
`cmsis-executorch-simple.cbuild-mlops.yml`: the processor, NPU and Vela
settings of the target, and the location of the AI layer. (`--packs` and
`--update-rte` are only needed on a fresh checkout.)

#### 2. Create the AI layer

```bash
python3 create_ai_layer.py cmsis-executorch-simple.cbuild-mlops.yml
```

This is the MLOps step. The script reads the NPU and Vela settings from the
file, quantizes and exports `model/model.py` for that NPU, and writes the
complete AI layer into `ai_layer/`: the component selection and the model as a
C array. It runs itself in `.venv` when started with another interpreter (use
`python` on Windows).

#### 3. Build the application

```bash
cbuild cmsis-executorch-simple.csolution.yml --active SSE-320-U85
```

A plain CMSIS build; no Python is involved. The resulting image is:

```text
out/cmsis-executorch-simple/SSE-320-U85/Debug/cmsis-executorch-simple.axf
```

#### 4. Run on the FVP

```bash
FVP_Corstone_SSE-320 \
    -f board/Corstone-320/fvp_config.txt \
    -a out/cmsis-executorch-simple/SSE-320-U85/Debug/cmsis-executorch-simple.axf
```

## How model generation works

The target is described by the `mlops:` node in
`cmsis-executorch-simple.csolution.yml`:

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
```

`cbuild setup --active SSE-320-U85` resolves it into
`cmsis-executorch-simple.cbuild-mlops.yml`, which contains the processor, NPU
and Vela options. `create_ai_layer.py` reads those options and passes them to
ExecuTorch's `EthosUCompileSpec`, so the target configuration is never
duplicated in Python. The script then writes:

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
model name or input handling as required. Then re-run `create_ai_layer.py` and
build.

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
| `cmsis-executorch-simple.csolution.yml` | Solution, target, and MLOps configuration |
| `cmsis-executorch-simple.cproject.yml` | Application project: sources plus the Board and AI layers |
| `model/model.py` | Example TinyCNN model |
| `create_ai_layer.py` | Exports the model for the target and writes the AI layer |
| `ai_layer/` | Generated: component selection and the embedded model data |
| `setup_venv.py` (`.sh` / `.bat`) | Creates the Python environment for the export |
| `board/Corstone-320/` | Corstone-320 platform support and FVP configuration |
| `src/app_main.cpp` | Loads the model, runs inference, and prints the result |
| `documentation/` | Detailed MLOps, pack, and cross-platform notes |

## Known limitations

- The supplied platform configuration targets Corstone-320 with Ethos-U85;
  another target needs its corresponding platform integration.

## License

The example code is licensed under Apache-2.0; see `LICENSE`. ExecuTorch uses a BSD-3-Clause license.

## References

- [PyTorch ExecuTorch CMSIS Pack](https://www.keil.arm.com/packs/executorch-pytorch/)
- [ExecuTorch](https://github.com/pytorch/executorch)
- [ExecuTorch Arm Ethos-U backend](https://docs.pytorch.org/executorch/main/backends-arm-ethos-u.html)
- [CMSIS-Toolbox MLOps information](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
- [Arm CMSIS documentation](https://arm-software.github.io/CMSIS_6/latest/index.html)
