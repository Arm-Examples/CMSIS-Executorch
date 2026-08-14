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
- Manage NPU and Vela configuration using CMSIS solution project rather duplicating the Python exporter.
- Generating the model as part of the normal CMSIS-Toolbox build process.
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

The example can be built and run entirely in Keil Studio for VS Code; no
command-line commands are required.

1. Install [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack) and [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) from the VS Code marketplace.
2. Clone or download this repository, then open its folder in VS Code.
3. Before using the example for the first time, select **Terminal > Run Task >
   Setup Python virtual environment**. Wait for the task to create the `.venv`
   environment and install the packages required to export the model.
4. Use the CMSIS action buttons to build the application, then select **Run** or
   **Debug**. Keil Studio starts the Corstone-320 FVP automatically.

A successful run prints the Ethos-U configuration, output logits, and a pass
result:

```text
Ethos-U version info:
    Arch:       v2.0.0
    MACs/cc:    256
    Cmd stream: v1
ExecuTorch Ethos-U85 example: 8864 byte model
Output: 10 element(s): 0.0187 -0.0204 -0.0645 0.0034 0.0187 ...
Test_result: PASS
```

### Command-line build

The same workflow can be performed from the VS Code Terminal as described below.

#### 1. Create the Python environment

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

#### 2. Build the application

```bash
cbuild cmsis-executorch-simple.csolution.yml --active SSE-320-U85 --packs --update-rte
```

This command:

1. Resolves and installs the required CMSIS packs.
2. Generates the MLOps build information for the selected target.
3. Quantizes and exports the model for Ethos-U85.
4. Generates the model's CMSIS component selection.
5. Compiles and links the embedded application.

The resulting image is:

```text
out/cmsis-executorch-simple/SSE-320-U85/Debug/cmsis-executorch-simple.hex
```

#### 3. Run on the FVP

```bash
FVP_Corstone_SSE-320 \
    -f board/Corstone-320/fvp_config.txt \
    -a out/cmsis-executorch-simple/SSE-320-U85/Debug/cmsis-executorch-simple.hex
```

## How model generation works

The selected target is described by the `mlops:` node in
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

Building with `--active SSE-320-U85` generates
`cmsis-executorch-simple.cbuild-mlops.yml`. This file contains the resolved
processor, NPU, and Vela options. `model/export_model.py` reads those options
and passes them to ExecuTorch's `EthosUCompileSpec`, so the target configuration
does not need to be duplicated in Python.

The export step produces:

- `model/model.pte`: the ExecuTorch program.
- `ai_layer/model/model_pte.h`: the same program embedded as a C array.
- `ai_layer/ai_layer.clayer.yml`: the CMSIS components required by the model.

See [the MLOps flow](documentation/mlops-flow.md) for a detailed walkthrough.

## Component selection

The [ExecuTorch CMSIS Pack](https://www.keil.arm.com/packs/executorch-pytorch/)
provides the runtime, backends, and individual operators as selectable CMSIS
components. `scripts/gen_components.py` examines the exported `.pte` and
updates `ai_layer/ai_layer.clayer.yml` so only the required components are
linked.

CMSIS-Toolbox resolves components before it executes the model-export step. If
a model change also changes its operator set, the first build updates the
component list and asks you to build again:

```text
[run_export] The model's operator set changed: ai_layer.clayer.yml was regenerated.
[run_export] Re-run the build to compile and link the updated component selection.
```

Run the same build command a second time to use the new selection.

## Adapting the example

To use a different model, replace or modify `model/model.py` and update the
model name or input handling as required. The next build regenerates the `.pte`
and embedded model data.

To target another Ethos-U configuration, update the target and `mlops:`
settings in the CMSIS solution. The generated Vela options then follow that
configuration automatically. Moving to a different board or reference platform
also requires the corresponding device pack, board support, memory layout, and
FVP configuration.

When updating ExecuTorch, update the CMSIS pack and Python package versions
together. More information is available in
[pack provenance](documentation/pack-provenance.md).

## Project layout

| Path | Purpose |
|------|---------|
| `cmsis-executorch-simple.csolution.yml` | Solution, target, and MLOps configuration |
| `cmsis-executorch-simple.cproject.yml` | Application project and model-conversion build step |
| `model/model.py` | Example TinyCNN model |
| `model/export_model.py` | Quantizes and delegates the model to Ethos-U |
| `scripts/run_export.py` | Runs model export and component generation |
| `scripts/gen_components.py` | Maps model operators to CMSIS components |
| `ai_layer/ai_layer.clayer.yml` | Generated model-specific component selection |
| `board/Corstone-320/` | Corstone-320 platform support and FVP configuration |
| `src/app_main.cpp` | Loads the model, runs inference, and prints the result |
| `documentation/` | Detailed MLOps, pack, and cross-platform notes |

## Known limitations

- A model change that changes the operator set requires two builds.
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
