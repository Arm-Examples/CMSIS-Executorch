# cmsis-executorch-simple

A minimal [ExecuTorch](https://github.com/pytorch/executorch) example that runs a
small int8 CNN on the **Ethos-U85** NPU of an Arm **Corstone-320 (SSE-320)**,
built with the CMSIS-Toolbox and the `PyTorch::ExecuTorch` CMSIS Pack.

It is deliberately small and has two goals:

1. **No Docker.** The model is exported from PyTorch in a plain `venv`
   (`pip install executorch`), not in a container.
2. **MLOps-driven model conversion.** The NPU/Vela configuration lives once in
   the csolution's `mlops:` node. CMSIS-Toolbox turns that into a
   `*.cbuild-mlops.yml`, and the model-conversion **build step** reads it to
   drive the Ethos-U compiler — so retargeting the NPU is a one-line edit.

## Layout

| Path | Purpose |
|------|---------|
| `cmsis-executorch-simple.csolution.yml` | Solution; target `SSE-320-U85`; the `mlops:` node |
| `cmsis-executorch-simple.cproject.yml` | App project; the `executes:` model-conversion build step |
| `model/model.py` | The TinyCNN (conv → relu → pool ×2 → linear) |
| `model/export_model.py` | Quantize + fully delegate to Ethos-U85; **consumes `cbuild-mlops.yml`**; emits `model_pte.h` |
| `scripts/gen_components.py` | Reads the `.pte` and writes the AI layer's **component list** |
| `ai_layer/ai_layer.clayer.yml` | The model's CMSIS component selection (generated) |
| `board/Corstone-320/` | Board bring-up (UART stdout + Ethos-U driver), trimmed |
| `src/app_main.cpp` | Headless runner: load `.pte`, run one inference, print logits |
| `packs/PyTorch.ExecuTorch.1.3.1-rc9/` | The vendored ExecuTorch source pack |

## Prerequisites

- Python 3.10+ (for the export venv). `setup_venv.sh` installs `executorch` and
  `ethos-u-vela` from `requirements.txt`, then the TOSA serializer
  (`tosa-tools`, which provides `tosa_serializer`) from `requirements-arm-tosa.txt`
  with `--no-dependencies` so it does not pull a conflicting torch.
- **CMSIS-Toolbox with MLOps support** — the `mlops:` node was added in
  `csolution 2.14.1+p9`. Older builds (including some registry artifacts that
  report `2.14.0`) reject it with `schema check failed`. The CMSIS Solution
  VS Code extension bundles a working build; `vcpkg` pulls a compatible one via
  `vcpkg-configuration.json`. `arm-none-eabi-gcc` comes from the same place.
  `build.sh` runs `vcpkg activate` for you.
- A CMSIS pack root with the public dependency packs (`ARM::CMSIS`,
  `ARM::CMSIS-NN`, `ARM::CMSIS-Compiler`, `ARM::Cortex_DFP`, `ARM::SSE_320_BSP`,
  `ARM::ethos-u-core-driver`). `cbuild --packs` installs any that are missing.

## Build

```bash
./setup_venv.sh   # create .venv and pip install executorch + ethos-u-vela
./build.sh        # vcpkg activate, then cbuild for SSE-320-U85 / Ethos-U85
```

`build.sh` runs:

```bash
cbuild cmsis-executorch-simple.csolution.yml --active SSE-320-U85 --packs --update-rte \
       --context cmsis-executorch-simple.Debug+SSE-320-U85
```

`--active SSE-320-U85` makes CMSIS-Toolbox generate
`cmsis-executorch-simple.cbuild-mlops.yml`. cbuild then runs the `executes:`
step, which calls `model/export_model.py` to (re)produce `model.pte` and
`ai_layer/model/model_pte.h`, and finally compiles and links the ELF.

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
    --pack-path packs/PyTorch.ExecuTorch.1.3.1-rc9 \
    --output ai_layer/ai_layer.clayer.yml
```

## Notes

- The pack is a **source** pack: every operator is a selectable component, so the
  image links only the kernels the model uses. A fully-delegated model needs just
  the runtime, the Ethos-U backend, and the int8 boundary quant/dequant kernels.
- This example is **build-only**. The `board/Corstone-320/fvp_config.txt` and the
  `mlops.simulator` target are included so you can extend it to run on
  `FVP_Corstone_SSE-320`, but no FVP execution is wired up here.

## License

The example code is licensed under **Apache-2.0** (see `LICENSE`). The vendored
ExecuTorch pack keeps its upstream **BSD** license
(`packs/PyTorch.ExecuTorch.1.3.1-rc9/LICENSE`).
