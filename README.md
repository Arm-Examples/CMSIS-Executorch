# ExecuTorch on Ethos-U85 via a source CMSIS pack

**Branch `pack-based-mlops` of [Arm-Examples/CMSIS-Executorch](https://github.com/Arm-Examples/CMSIS-Executorch).**

A minimal [ExecuTorch](https://github.com/pytorch/executorch) example that runs a
small int8 CNN on the **Ethos-U85** NPU of an Arm **Corstone-320 (SSE-320)**,
built with the CMSIS-Toolbox and the `PyTorch::ExecuTorch` CMSIS Pack.

This branch is an alternative to the [`main`](https://github.com/Arm-Examples/CMSIS-Executorch/tree/main)
template, not a replacement for it. It is deliberately small and has two goals:

1. **No Docker.** The model is exported from PyTorch in a plain `venv`
   (`pip install executorch`), not in a container.
2. **MLOps-driven model conversion.** The NPU/Vela configuration lives once in
   the csolution's `mlops:` node. CMSIS-Toolbox turns that into a
   `*.cbuild-mlops.yml`, and the model-conversion **build step** reads it to
   drive the Ethos-U compiler — so retargeting the NPU is a one-line edit.

## Relation to `main`

| | [`main`](https://github.com/Arm-Examples/CMSIS-Executorch/tree/main) | this branch |
|---|---|---|
| Model build environment | Docker container, multi-stage | plain `venv`, any host OS |
| ExecuTorch delivery | prebuilt static libraries committed into `ai_layer/` | **source** CMSIS pack; every operator is a selectable component |
| Who builds the AI layer | a GitHub Action, which commits artifacts back | the `executes:` build step, locally, on every build |
| NPU / Vela configuration | hard-coded in `model/aot_model.py` | csolution `mlops:` node → `cbuild-mlops.yml` → Vela |
| Target | Corstone-300 / Ethos-U55 | Corstone-320 / Ethos-U85 |
| Run / debug | FVP via the CMSIS action buttons | FVP from the command line (the [`Codespaces`](../../tree/Codespaces) branch adds Run/Debug buttons and a devcontainer) |

Pick `main` if you want a hands-off, CI-driven template with a reproducible
container. Pick this branch if you want a local, inspectable flow with a
minimal linked footprint and a single place to change the NPU target.

Project files are named `cmsis-executorch-simple.*` here rather than `main`'s
`executorch_project.*`. That is intentional: renaming would touch `build.sh`,
every `out/` path and the debug wiring on the `Codespaces` branch, for no
benefit on a parallel branch.

## Layout

| Path | Purpose |
|------|---------|
| `cmsis-executorch-simple.csolution.yml` | Solution; target `SSE-320-U85`; the `mlops:` node |
| `cmsis-executorch-simple.cproject.yml` | App project; the `executes:` model-conversion build step |
| `model/model.py` | The TinyCNN (conv → relu → pool ×2 → linear) |
| `model/export_model.py` | Quantize + fully delegate to Ethos-U85; **consumes `cbuild-mlops.yml`**; emits `model_pte.h` |
| `scripts/run_export.cmake` | Host-OS dispatcher for the build step: picks the venv interpreter |
| `scripts/run_export.py` | Runs the export, then refreshes the AI layer component list |
| `scripts/gen_components.py` | Reads the `.pte` and writes the AI layer's **component list** |
| `ai_layer/ai_layer.clayer.yml` | The model's CMSIS component selection (generated) |
| `board/Corstone-320/` | Board bring-up (semihosting stdout + Ethos-U driver), trimmed |
| `src/app_main.cpp` | Headless runner: load `.pte`, run one inference, print logits |
| `documentation/` | [MLOps flow](documentation/mlops-flow.md), [pack provenance](documentation/pack-provenance.md), [cross-platform notes](documentation/cross-platform.md) |

## Prerequisites

**Python `>=3.10,<3.15`** for the export venv — that is ExecuTorch's supported
range, and `setup_venv.py` checks it up front rather than letting pip fail
later with an unrelated-looking resolver error.

```bash
./setup_venv.sh     # Linux, macOS
setup_venv.bat      # Windows
```

Either wrapper just runs `setup_venv.py`, which creates `.venv/` and installs
the pinned packages in three passes — see [Version pinning](#version-pinning)
for what lands and why the passes are separate. The script is idempotent: run
it again any time, and it rebuilds the venv by itself if the interpreter it
points at has gone stale (which happens after a container rebuild).

**CMSIS-Toolbox with MLOps support** — the `mlops:` node was added in
`csolution 2.14.1+p9`. Older builds (including some registry artifacts that
report `2.14.0`) reject it with `schema check failed`. The CMSIS Solution
VS Code extension bundles a working build; `vcpkg` pulls a compatible one via
`vcpkg-configuration.json`. `arm-none-eabi-gcc` comes from the same place.
`build.sh` runs `vcpkg activate` for you.

**A CMSIS pack root** with `PyTorch::ExecuTorch@1.4.0` and the public
dependency packs (`ARM::CMSIS`, `ARM::CMSIS-NN`, `ARM::CMSIS-Compiler`,
`ARM::Cortex_DFP`, `ARM::SSE_320_BSP`, `ARM::ethos-u-core-driver`). `cbuild
--packs` — which `build.sh` passes — installs any that are missing, so a fresh
clone needs no pack step of its own. To install the ExecuTorch pack by hand:

```bash
cpackget add PyTorch::ExecuTorch@1.4.0
```

The version is pinned exactly, because the pack's C++ runtime and the Python
exporter must be the same ExecuTorch version. See
[pack provenance](documentation/pack-provenance.md) for where the pack comes
from and how to build one yourself.

### Host OS support

Building and exporting the model works on Linux, macOS and Windows. Two caveats:

- **Windows long paths.** `torch` unpacks paths long enough to hit the legacy
  260-character `MAX_PATH` limit. `setup_venv.py` warns if long-path support is
  off; enable it, or clone nearer the drive root.
- **Windows FVP invocation.** Building, exporting and running the FVP all work
  from PowerShell; only the path separators differ. See
  [cross-platform notes](documentation/cross-platform.md).

## Build

```bash
./setup_venv.sh   # create .venv and pip install executorch + ethos-u-vela
./build.sh        # vcpkg activate, then cbuild for SSE-320-U85 / Ethos-U85
```

On Windows use `setup_venv.bat` and `build.ps1`. Inside Keil Studio the Arm
Environment Manager has already activated the toolchain, so plain `cbuild`
works from its terminal.

`build.sh` runs:

```bash
cbuild cmsis-executorch-simple.csolution.yml --active SSE-320-U85 \
       --packs --update-rte
```

`--active SSE-320-U85` makes CMSIS-Toolbox generate
`cmsis-executorch-simple.cbuild-mlops.yml`. cbuild then runs the `executes:`
step, which calls `model/export_model.py` to (re)produce `model.pte` and
`ai_layer/model/model_pte.h`, and finally compiles and links the ELF.
(`--context` cannot be combined with `--active`; the csolution's `target-set`
already selects the Debug context.)

The generated `model_pte.h` is listed as a project source in the cproject `App`
group. That marks it a build *input*, so CMSIS-Toolbox schedules the
`convert-model` step **before** compilation (rather than the default post-build);
the app is always built against a freshly converted model.

## How the MLOps flow works

The csolution declares the target once:

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

CMSIS-Toolbox emits `cmsis-executorch-simple.cbuild-mlops.yml` with a
`vela.options` string (`--accelerator-config ethos-u85-256 --system-config …
--memory-mode …`). `export_model.py` parses those and passes them straight to
`EthosUCompileSpec`. Change the NPU in the csolution and the next build retargets
the model — no Python edits.

The full hop-by-hop walkthrough is in [documentation/mlops-flow.md](documentation/mlops-flow.md).

## Regenerating the component list

The AI layer selects exactly the ExecuTorch components the model needs. The
`convert-model` build step regenerates `ai_layer/ai_layer.clayer.yml` from the
exported `.pte` automatically (via `scripts/gen_components.py`). Because
CMSIS-Toolbox resolves components *before* the build runs, a changed operator
set cannot take effect in the same build: the step then stops with

```
[run_export] The model's operator set changed: ai_layer.clayer.yml was regenerated.
[run_export] Re-run the build to compile and link the updated component selection.
```

Simply re-run the build — the second run uses the updated selection and goes
through. The script can still be run manually:

```bash
.venv/bin/python scripts/gen_components.py \
    --pte model/model.pte \
    --output ai_layer/ai_layer.clayer.yml
```

It reads the component list from the pack installed in your pack root. Add
`--pack-version` to select between several installed versions, or `--pack-path`
to read an unpacked pack directory somewhere else.

## Version pinning

The Python side must match the runtime shipped by the pack. The pins
below come from ExecuTorch's own `release/1.4` branch
(`install_requirements.py`, `pyproject.toml`):

| Package | Pin | Index | Why |
|---|---|---|---|
| `executorch` | `1.4.0.dev20260728` | PyTorch nightly | 1.4.0 is not on PyPI yet; this wheel is cut from `release/1.4` |
| `torch` | `2.13.0` | PyPI | ExecuTorch declares **no** torch dependency on purpose, so the pin lives here. 1.4 needs 2.13 |
| `torchao` | `0.18.0.dev20260715` | PyTorch nightly | what 1.4 pins; PyPI is still on 0.17.0 |
| `ethos-u-vela` | `5.1.0` | PyPI | the Ethos-U compiler |
| `tosa-tools` | `2026.5.0` | PyPI, `--no-deps` | provides `tosa_serializer` |
| `flatbuffers` | `24.3.25` | PyPI, `--no-deps` | `tosa-tools` wants `25.2.10`; ExecuTorch needs `24.3.25` |

That last row is why the requirements are split across three files and
installed in three ordered passes:

- `requirements.txt` — PyPI only, no index directives, so pip cannot prefer a
  nightly `torch` build over the pinned release.
- `requirements-executorch.txt` — the nightly index, for `executorch` and
  `torchao` only.
- `requirements-arm-tosa.txt` — installed with `--no-dependencies`, so
  `tosa-tools` cannot drag in a `flatbuffers` that breaks serialization. This
  is also why `pip install executorch[ethos_u]` is *not* used, despite that
  extra declaring exactly the right pins.

**When ExecuTorch 1.4.0 ships on PyPI**, change the `executorch` line in
`requirements-executorch.txt` to `executorch==1.4.0` and move it to
`requirements.txt`. Nothing else needs to change.

**To build from the `release/1.4` branch head** instead of the nightly wheel:

```bash
./setup_venv.sh --executorch-ref release/1.4
```

This is a *source* build: it needs CMake and a C++ toolchain and takes tens of
minutes, so the prebuilt nightly wheel is the default.

## Run on the FVP

Run the built ELF on `FVP_Corstone_SSE-320` directly. `vcpkg activate` (or the
Arm Environment Manager inside VS Code) puts it on `PATH` from the
`arm:models/arm/avh-fvp` pin:

```bash
FVP_Corstone_SSE-320 \
    -f board/Corstone-320/fvp_config.txt \
    -a out/cmsis-executorch-simple/SSE-320-U85/Debug/cmsis-executorch-simple.elf
```

`board/Corstone-320/fvp_config.txt` pins `INITSVTOR` to the application's
vector table in BRAM so debugger resets survive, and enables semihosting —
stdio is retargeted to semihosting (`board/Corstone-320/retarget_stdio.c`), so
printf output appears directly on the FVP's stdout with no UART model in
between. The application does not implement the semihosting exit call, so the
model keeps running after the last line; add `--simlimit 60` (as CI does) or
stop it yourself.

> **Run/Debug from the CMSIS Solution panel** — the
> [`Codespaces`](../../tree/Codespaces) branch wires the **Load / Run / Debug**
> buttons onto the model (which is its own GDB server from MDK FVP 11.32.23 on)
> and adds a devcontainer for GitHub Codespaces.

A successful run prints the output logits followed by `Test_result: PASS` on
the FVP console:

```
Ethos-U version info:
	Arch:       v2.0.0
	MACs/cc:    256
	Cmd stream: v1
ExecuTorch Ethos-U85 example: 8864 byte model
Output: 10 element(s): 0.0187 -0.0204 -0.0645 0.0034 0.0187 ...
Test_result: PASS
```

## Known limitations

- **The Python side is still pre-release.** ExecuTorch 1.4.0 is not on PyPI
  yet, so the pins are nightlies cut from `release/1.4`. See
  [Version pinning](#version-pinning) for the one-line switch when it lands.
- **`csolution 2.14.1+p9` or newer is required** for the `mlops:` node.
- **An operator-set change needs two builds** — components are resolved before
  the build runs. See [Regenerating the component list](#regenerating-the-component-list).
- **Running the FVP needs an Arm user-based license.**
- **No editor run/debug wiring.** Running the FVP is a command line away; the
  [`Codespaces`](../../tree/Codespaces) branch is the one that maps it onto the
  CMSIS Solution panel's buttons.

## Notes

- The pack is a **source** pack: every operator is a selectable component, so the
  image links only the kernels the model uses. A fully-delegated model needs just
  the runtime, the Ethos-U backend, and the int8 boundary quant/dequant kernels.

## License

The example code is licensed under **Apache-2.0** (see `LICENSE`). The
ExecuTorch pack it builds against is a separate download under its own
upstream **BSD** license (`LICENSE` inside the pack).

## References

- [Arm CMSIS documentation](https://arm-software.github.io/CMSIS_6/latest/index.html)
- [CMSIS-Toolbox: MLOps information](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
- [ExecuTorch](https://github.com/pytorch/executorch) · [Arm Ethos-U backend tutorial](https://docs.pytorch.org/executorch/main/backends-arm-ethos-u.html)
- [CMSIS Solution extension: action buttons](https://github.com/Open-CMSIS-Pack/vscode-cmsis-solution#action-buttons)
