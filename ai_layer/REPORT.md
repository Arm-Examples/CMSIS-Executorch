# ExecuTorch AI Layer Build Report

**Generated:** 2026-01-27T15:55:53Z
**Git Commit:** f58e85dc74eb on main
**Repository Status:** 🔄 Modified
**Last Commit:** 2026-01-27 16:47:30 +0100

## 📊 Build Summary

- **Libraries:** 9 static libraries
- **Models:** 1 model assets
- **Operators:** 2 selected operators
- **Build Type:** Release

## 📚 Library Assets

**Total Size:** 9.9 MiB

| Library | Size | Percentage | Modified | Hash |
|---------|------|------------|----------|------|
| libcortex_m_kernels.a | 20.0 KiB | 0.2% | 2026-01-27 15:55:25 | `86ef9627fa72a103` |
| libcortex_m_ops_lib.a | 12.6 KiB | 0.1% | 2026-01-27 15:55:26 | `f4a47ddf1b21f081` |
| libexecutorch.a | 51.9 KiB | 0.5% | 2026-01-27 15:55:25 | `4f930898f0b42cb2` |
| libexecutorch_core.a | 216.5 KiB | 2.1% | 2026-01-27 15:55:25 | `236f25130fc02703` |
| libexecutorch_delegate_ethos_u.a | 18.6 KiB | 0.2% | 2026-01-27 15:55:26 | `7018288463b78ba6` |
| libportable_kernels.a | 9.1 MiB | 92.3% | 2026-01-27 15:55:25 | `7e65742bade39aa2` |
| libportable_ops_lib.a | 199.1 KiB | 2.0% | 2026-01-27 15:55:25 | `c4073b752a26f3fe` |
| libquantized_kernels.a | 238.6 KiB | 2.4% | 2026-01-27 15:55:26 | `95ab0f70d52a7c81` |
| libquantized_ops_lib.a | 28.7 KiB | 0.3% | 2026-01-27 15:55:26 | `6a439a0e67d94665` |

## 🤖 Model Assets

| Asset | Type | Size | Modified | Hash |
|-------|------|------|----------|------|
| ethos_u_minimal_example.pte | Model | 3.8 KiB | 2026-01-27 15:55:53 | `bd7a211160a18572` |

## ⚙️ Build Configuration

### CMake Configuration
- **Build Type:** `Release`
- **Toolchain File:** `/workspace2/model/arm-none-eabi-gcc.cmake`
- **ARM Baremetal:** `ON`
- **Cortex-M Support:** `ON`
- **Portable Ops:** `ON`
- **Quantized Kernels:** `ON`

### Selected Operators

**Source:** Model file: ethos_u_minimal_example.pte (inferred)

**Count:** 2 operators

```
quantized_decomposed::dequantize_per_tensor.out
quantized_decomposed::quantize_per_tensor.out
```

## 🔄 Model Conversion Details

**Ethos-U Compile Specification:**
  - target: ethos-u55-128
  - system_config: Ethos_U55_High_End_Embedded
  - memory_mode: Shared_Sram
  - extra_flags: --output-format=raw, --debug-force-regor, --verbose-all

**Quantization Configuration:**
  - Using EthosUQuantizer with symmetric quantization
  - Post-training quantization enabled

**Model Architecture:**
  - Model class: Add

**Vela Compilation Summary:**
  - Accelerator configuration               Ethos_U55_128
  - System configuration             Ethos_U55_High_End_Embedded
  - Memory mode                               Shared_Sram
  - Accelerator clock                                 500 MHz
  - Design peak SRAM bandwidth                       3.73 GB/s
  - Design peak Off-chip Flash bandwidth             0.47 GB/s
  - Total SRAM used                                  0.14 KiB
  - Total Off-chip Flash used                        0.03 KiB
  - CPU operators = 0 (0.0%)
  - NPU operators = 12 (100.0%)
  - Average SRAM bandwidth                           0.27 GB/s
  - Input   SRAM bandwidth                           0.00 MB/batch
  - Weight  SRAM bandwidth                           0.00 MB/batch
  - Output  SRAM bandwidth                           0.00 MB/batch
  - Total   SRAM bandwidth                           0.00 MB/batch
  - Total   SRAM bandwidth            per input      0.00 MB/inference (batch size 1)
  - Average Off-chip Flash bandwidth                 0.04 GB/s
  - Input   Off-chip Flash bandwidth                 0.00 MB/batch
  - Weight  Off-chip Flash bandwidth                 0.00 MB/batch
  - Output  Off-chip Flash bandwidth                 0.00 MB/batch
  - Total   Off-chip Flash bandwidth                 0.00 MB/batch
  - Total   Off-chip Flash bandwidth  per input      0.00 MB/inference (batch size 1)
  - Original Weights Size                            0.00 KiB
  - NPU Encoded Weights Size                         0.00 KiB
  - Neural network macs                                 0 MACs/batch
  - Info: The numbers below are internal compiler estimates.
  - For performance numbers the compiled network should be run on an FVP Model or FPGA.
  - Network Tops/s                                   0.00 Tops/s
  - NPU cycles                                        349 cycles/batch
  - SRAM Access cycles                                 24 cycles/batch
  - DRAM Access cycles                                  0 cycles/batch
  - On-chip Flash Access cycles                         0 cycles/batch
  - Off-chip Flash Access cycles                       32 cycles/batch
  - Total cycles                                      349 cycles/batch
  - Batch Inference time                 0.00 ms, 1432664.76 inferences/s (batch size 1)

## 📦 Source Layer Export

**Total Source Files:** 226 files across 2 layers

| Layer | Description | Groups | Files |
|-------|-------------|--------|-------|
| stage1 | Generated from compile_commands.json (327 files) | 4 | 37 |
| stage2 | Generated from compile_commands.json (240 files) | 1 | 189 |

### Group Details

**STAGE1:**

- `Runtime`: 19 files
- `Schema`: 1 files
- `Kernels/quantized`: 11 files
- `Backends`: 6 files

**STAGE2:**

- `Kernels`: 189 files


## 🛠️ Build Environment

- **Platform:** `Linux f16f207ed567 6.12.54-linuxkit #1 SMP Fri Nov 21 10:33:45 UTC 2025 aarch64 aarch64 aarch64 GNU/Linux`
- **Python:** `Python 3.12.3`
- **CMake:** `cmake version 4.2.0`
- **ARM GCC:** `arm-none-eabi-gcc (Arm GNU Toolchain 13.3.Rel1 (Build arm-13.24)) 13.3.1 20240614`

## 📁 Asset Locations

```
ai_layer/
├── engine/
│   ├── lib/           # Static libraries
│   ├── include/       # Header files
│   └── model/         # Model assets
└── REPORT.md          # This report
```

---
*Report generated by ExecuTorch AI Layer build system at 2026-01-27T15:55:53Z*