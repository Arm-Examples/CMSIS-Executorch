# Requirements — PyTorch::ExecuTorch CMSIS Pack

**Target project:** PyTorch::ExecuTorch CMSIS Pack (build infrastructure under
`backends/arm/scripts/cmsis_pack/` in `pytorch/executorch`).

**Pack versions exercised:**
- `1.3.0-nightly-20260430` — initial Ethos-U integration target.
- `1.3.0-nightly-20260504` — current: R1 + R2 landed; new follow-ups R3, R4.

**Status summary (all four landed in pack 20260504 build of 2026-05-04 18:44):**
- R1 (cortex_m kernel registrations in `RegisterAllKernels.cpp`) — ✅ done
- R2 (per-op cortex_m pack components with `RTE_ML_EXECUTORCH_OP_CORTEX_M_*` defines) — ✅ done
- R3 (cortex_m fwd-decls now have inline `using` aliases) — ✅ done
- R4 (`minimal.cpp` definitions now use `ET_WEAK` directly) — ✅ done

**No project-side workarounds remain.** All four `__attribute__((weak))` markers
on the project's `et_pal_*` overrides, the `-Wl,-z,muldefs` GCC link flag, the
`-include cortex_m_ops_common.h` force-include, the `Backend CortexM` umbrella
component, the `Platform Bare-Metal` component, the `Stubs RandomOps`
component, and the `CMSIS:NN Lib` pin (the q/dq pair is pure-CPU, no longer
needs CMSIS-NN headers transitively) have all been removed.

**Symptom on the consumer side:** any model that has a CPU-side
`cortex_m::quantize_per_tensor.out` / `cortex_m::dequantize_per_tensor.out`
boundary node (the standard output of `ReplaceQuantNodesPass` from
`backends/arm/scripts/aot_arm_compiler.py` for an Ethos-U-delegated graph
with float in/out) traps at `Method::execute` time with `Error::NotFound`,
because the operator-registry lookup for the `cortex_m::*` kernel returns
nullptr.

## Background — what is and isn't there today

The pack already ships everything except the registration:

| What | Where | Status |
|---|---|---|
| Kernel implementations (`cortex_m::native::quantize_per_tensor_out`, `dequantize_per_tensor_out`, and the `quantized_*` family) | `src/backends/cortex_m/ops/op_*.cpp` | ✅ shipped |
| Kernel schema declarations | `src/backends/cortex_m/ops/operators.yaml` (16 ops) | ✅ shipped, 7-arg quantize / 7-arg dequantize / etc. matching the upstream `kernels/quantized/quantized.yaml` shape |
| Pack component that compiles those `.cpp` files | `<component … Csub="Backend CortexM" condition="CMSIS-NN">` in `PyTorch.ExecuTorch.pdsc` | ✅ shipped |
| `Pre_Include_Global_h` define triggered by selecting the component | `#define EXECUTORCH_BUILD_CORTEX_M 1` and `#define RTE_ML_EXECUTORCH_BACKEND_CORTEX_M` | ✅ shipped |
| **Runtime kernel-registry registration** for the cortex_m ops | — | ❌ **missing** |
| Per-op components (one CMSIS pack component per cortex_m op, with `#define RTE_ML_EXECUTORCH_OP_CORTEX_M_<NAME>`), so a project can selectively pull in only the ops its model needs | — | ❌ **missing** (mirrors the existing portable / quantized_decomposed component pattern) |

`RegisterAllKernels.cpp` today contains zero `cortex_m` references. The
codegen that produces it (a) walks `kernels/portable/cpu/operators.yaml` and
`kernels/quantized/quantized.yaml`, (b) emits a `Kernel(...)` registration for
each `.out` overload guarded by `#ifdef RTE_ML_EXECUTORCH_OP_PORTABLE_<NAME>`
or `#ifdef RTE_ML_EXECUTORCH_OP_QUANTIZED_<NAME>`. The cortex_m yaml is
simply not in the codegen's input set.

## Requirement R1 — register `cortex_m::*.out` kernels in `RegisterAllKernels.cpp` ✅ done in 20260504

The codegen MUST also walk `src/backends/cortex_m/ops/operators.yaml` and emit
a `Kernel(...)` registration for every entry, in the same generated file.
Each registration MUST be guarded by an `#ifdef` so the registration
(and its symbol-pulling effect on the static initializer) only enters the
final binary when the consumer project selects the corresponding pack
component(s). The existing convention is:

```cpp
#ifdef RTE_ML_EXECUTORCH_OP_CORTEX_M_QUANTIZE_PER_TENSOR
    Kernel(
        "cortex_m::quantize_per_tensor.out",
        [](KernelRuntimeContext& context, Span<EValue*> stack) {
            ET_KERNEL_CHECK_MSG(context, stack.size() == 7, InvalidProgram,
                                /*void*/, "Expected 7 args");
            EValue& input       = *stack[0];
            EValue& scale       = *stack[1];
            EValue& zero_point  = *stack[2];
            EValue& quant_min   = *stack[3];
            EValue& quant_max   = *stack[4];
            EValue& dtype       = *stack[5];
            EValue& out         = *stack[6];
            cortex_m::native::quantize_per_tensor_out(
                context,
                input.to<Tensor>(), scale.to<double>(),
                zero_point.to<int64_t>(), quant_min.to<int64_t>(),
                quant_max.to<int64_t>(), dtype.to<ScalarType>(),
                out.to<Tensor>());
        }),
#endif
```

Forward declarations for the `cortex_m::native::*_out` functions go in the
same forward-decl block where the existing `RegisterAllKernels.cpp`
forward-declares the `torch::executor::native::*_out` functions.

This MUST cover every entry currently in `cortex_m/ops/operators.yaml`:
`quantize_per_tensor.out`, `dequantize_per_tensor.out`, `quantized_add.out`,
`quantized_mul.out`, `minimum.out`, `maximum.out`, `quantized_linear.out`,
`softmax.out`, `transpose.out`, `pad.out`, `quantized_conv2d.out`,
`quantized_depthwise_conv2d.out`, `quantized_transpose_conv2d.out`,
`quantized_avg_pool2d.out`, `quantized_max_pool2d.out`,
`quantized_batch_matmul.out`.

## Requirement R2 — per-op CMSIS pack components for cortex_m ops ✅ done in 20260504

The pdsc MUST add one `<component … Csub="Operators Cortex-M <NAME>"
condition="op_cortex_m_<name>">` per op (mirroring the existing per-op
components under `Cgroup="ExecuTorch Operators"` for portable and
quantized_decomposed ops). Each per-op component MUST:

- Set `<RTE_Components_h>#define RTE_ML_EXECUTORCH_OP_CORTEX_M_<NAME></RTE_Components_h>` so the codegen guard in R1 fires.
- Include the matching `op_<name>.cpp` file from `src/backends/cortex_m/ops/`.

Rationale: image-size sensitivity. A consumer that only needs the boundary
q/dq pair (the common case for fully-delegated Ethos-U models) MUST be able
to pull in only `Operators Cortex-M Quantize Per Tensor` and
`Operators Cortex-M Dequantize Per Tensor`, not the entire CMSIS-NN-backed
operator family with its CMSIS-NN dependency.

The current `Backend CortexM` component compiles **all** cortex_m op
sources unconditionally. R2 lets that umbrella component be retired (or
made `condition`-gated to "any cortex_m op selected"), aligning the cortex_m
operator tree with how portable and quantized_decomposed ops are already
structured.

## Requirement R3 — match the AOT-emitted argument shape

The runtime kernel signatures and codegen-emitted trampolines for
`cortex_m::quantize_per_tensor.out` and `cortex_m::dequantize_per_tensor.out`
MUST agree with what `executorch.exir.passes.ToOutVarPass` (or its
torchao-side equivalent) actually emits in current nightly toolchains.

Today there is a drift: the AOT path emits an extra optional `out_dtype`
EValue (so `stack.size() == 8`) for the `.out` variant of both quantize
and dequantize, while the pack's `cortex_m/ops/operators.yaml` declares
7-argument schemas. This is the same drift that already affects
`quantized_decomposed::quantize_per_tensor.out` (model 8 vs runtime 7) and
`quantized_decomposed::dequantize_per_tensor.out` (model 7 vs runtime 8 —
opposite direction).

The pack project MUST pick one of:

a) Update `cortex_m/ops/operators.yaml` and the kernel C++ signatures to
   include `out_dtype: optional<ScalarType>` for both quantize and
   dequantize, matching the AOT side. Codegen R1 follows.
b) Have the codegen R1 emit trampolines that accept either 7 or 8 args
   and route the trailing `out` correctly (i.e. `stack[stack.size()-1]`),
   ignoring an extra `out_dtype` if present. The native function
   signature stays at 7 args.

Option (b) is the lower-risk choice for a pack that pins to a specific
nightly: it tolerates AOT-side schema drift in either direction without
recompiling kernels. This is the pattern the workaround in
`src/cortex_m_kernel_shim.cpp` of the consumer project uses today.

<!-- The CLANG/picolibc / __bothinit_array_* linker-script issue is not
pack-fixable: it lives in the DFP/board linker script (here Alif Ensemble),
not in the executorch pack. The CMSIS-Executorch project drops CLANG from
its tested-toolchain matrix (GCC 13.x/14.x/15.x and AC6 only) until the
relevant DFP linker scripts are updated. -->

## Requirement R3 — `RegisterAllKernels.cpp` cortex_m fwd-decls don't compile ✅ done in 20260504 (rebuild)

In pack `1.3.0-nightly-20260504`, the new cortex_m forward-declaration
block at `src/registration/RegisterAllKernels.cpp:814-885` looks like:

```cpp
namespace cortex_m {
namespace native {
#ifdef RTE_ML_EXECUTORCH_OP_CORTEX_M_QUANTIZE_PER_TENSOR
Tensor& quantize_per_tensor_out(KernelRuntimeContext& context,
                                const Tensor& input, double scale, …,
                                ScalarType dtype, Tensor& out);
#endif
…
} // namespace native
} // namespace cortex_m
```

Inside `cortex_m::native` the names `Tensor`, `KernelRuntimeContext`,
`ScalarType`, `Int64ArrayRef` are unqualified and unresolved. GCC 13/14/15
and AC6 6.24 both reject the file:

```
error: 'Tensor' does not name a type
error: unknown type name 'KernelRuntimeContext'
error: unknown type name 'ScalarType'
error: 'dequantize_per_tensor_out' is not a member of 'cortex_m::native'
```

The pack already ships a header that declares the matching `using` aliases
inside `cortex_m::native` —
`include/executorch/backends/cortex_m/ops/cortex_m_ops_common.h`:

```cpp
namespace cortex_m { namespace native {
using Tensor = torch::executor::Tensor;
using ScalarType = executorch::aten::ScalarType;
using Error = executorch::runtime::Error;
using Int64ArrayRef = executorch::aten::ArrayRef<int64_t>;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;
…
}}
```

**Required fix:** the codegen that produces `RegisterAllKernels.cpp` MUST
`#include <executorch/backends/cortex_m/ops/cortex_m_ops_common.h>` before
the cortex_m forward-declaration block (or alternatively emit the matching
`using` aliases inline at the top of `namespace cortex_m { namespace native`).

**Project-side workaround (active in CMSIS-Executorch):** force-include
`cortex_m_ops_common.h` for every translation unit via cproject misc.CPP
`-include` flag. This drags in `arm_nn_types.h` (CMSIS-NN) so the project
also pins `ARM::CMSIS-NN` and selects `CMSIS:NN Lib` even when the model
only uses the boundary q/dq pair, which is heavier than necessary.

## Requirement R4 — `minimal.cpp` defaults must emit as weak symbols ✅ done in 20260504 (rebuild)

`src/runtime/platform/default/minimal.cpp` shipped by the Runtime
component is intended to provide overridable defaults for `et_pal_init`,
`et_pal_abort`, `et_pal_current_ticks`, `et_pal_ticks_to_ns_multiplier`,
`et_pal_emit_log_message`, `et_pal_allocate`, `et_pal_free`. The file
defines:

```cpp
#define ET_INTERNAL_PLATFORM_WEAKNESS ET_WEAK
#include <executorch/runtime/platform/platform.h>   // declarations gain ET_WEAK

void et_pal_init(void) {}                            // *** definition not weak ***
ET_NORETURN void et_pal_abort(void) { __builtin_trap(); }
…
```

Both GCC `arm-none-eabi-gcc` (13.3.1, 14.3.1, 15.2.1) and `armclang` 6.24.0
emit these definitions as **strong** symbols — the `__attribute__((weak))`
on the previous declaration does not propagate to a separate definition in
either toolchain in practice. Result: any application that supplies its
own `et_pal_*` (which is the documented intent of the override pattern)
gets a hard multiple-definition link error:

```
GCC: multiple definition of 'et_pal_init'; minimal.cpp.obj … first defined here
AC6: Error: L6200E: Symbol et_pal_emit_log_message multiply defined
     (by minimal.o and arm_executor_runner.o)
```

**Required fix:** apply the weak attribute on the *definitions* in
`minimal.cpp`, not just on the prior declarations. The easiest pattern that
works for both GCC and armclang is:

```cpp
ET_WEAK void et_pal_init(void) {}
ET_WEAK ET_NORETURN void et_pal_abort(void) { __builtin_trap(); }
…
```

Equivalently, use `__attribute__((weak))` directly on each definition.
The `ET_INTERNAL_PLATFORM_WEAKNESS` macro currently only weakens the
declarations and is ineffective.

**Project-side workaround (active in CMSIS-Executorch):**
`src/executor_runner/arm_executor_runner.cpp` marks its strong overrides
of `et_pal_*` with `__attribute__((weak))` so that both project and pack
definitions are weak; the linker picks one without erroring. This means
the project loses its DWT-cycle-counter `et_pal_current_ticks` and
printf-routed `et_pal_emit_log_message` non-deterministically (linker
order can pick the pack's no-op defaults instead). Acceptable for a
self-test, not for production observability.

## Acceptance criteria

A consumer project that:

- Selects `Machine Learning:ExecuTorch:Backend EthosU`
- Selects `Machine Learning:ExecuTorch Operators:Cortex-M quantize_per_tensor` and `… dequantize_per_tensor`
- Runs an Ethos-U-delegated PTE produced by the upstream
  `backends/arm/scripts/aot_arm_compiler.py` (or any AOT pipeline that
  applies `ReplaceQuantNodesPass`)

…must execute end-to-end on hardware (`Method::execute` returns `Error::Ok`,
the application reaches its idle loop) without any project-side kernel
registration shim.

## Validation evidence

- **Pack `1.3.0-nightly-20260430`** (R1 missing): on AppKit-E8 (Alif
  AE822FA0E5597BS0, M55_HP) the fully-delegated Ethos-U85 add and mul
  models from `model/aot_model_a.py` and `model/aot_model_b.py` trap with
  `Error::InvalidProgram` at `Method::execute_instruction` because the
  cortex_m boundary q/dq kernels are not in the operator registry. A
  100-line project-side `src/cortex_m_kernel_shim.cpp` (since deleted)
  registered them and unblocked execution.
- **Pack `1.3.0-nightly-20260504` (R1 + R2 only)**: per-op components
  selectable, but the cortex_m fwd-decls in RegisterAllKernels.cpp don't
  resolve `Tensor` / `KernelRuntimeContext` etc., and `minimal.cpp` ships
  strong et_pal_* definitions colliding with project overrides. Required
  project-side workarounds (force-include + weak overrides) to build.
- **Pack `1.3.0-nightly-20260504` rebuild (R1 + R2 + R3 + R4)**: the
  cortex_m fwd-decls now embed inline `using` aliases for the four kernel-
  API types, and every `et_pal_*` definition in `minimal.cpp` is marked
  `ET_WEAK` directly. The Alif AppKit-E8 (M55_HP + Ethos-U85) board runs
  fully-delegated Ethos-U85 add and mul models with **zero project-side
  workarounds**: `ra = 0`, `rb = 0`, idle reached. Same models on the
  Corstone-320 FVP (AVH-SSE-320 with `num_macs=256`) print
  `Model A: 2.000000 ×4` and `Model B: 1.000000 ×4` and exit cleanly.

### Build matrix observed against pack 1.3.0-nightly-20260504 (R1+R2+R3+R4 build, no project workarounds)

| Toolchain | AVH-SSE-320 (FVP) | AppKit-E8-HP (HW) |
|---|---|---|
| GCC 13.3.1 | ✅ Debug + Release | ✅ Debug + Release |
| GCC 14.3.1 | ✅ Debug + Release | ✅ Debug + Release |
| GCC 15.2.1 | ✅ Debug + Release | ✅ Debug + Release |
| AC6 6.24.0 | ✅ Debug + Release | ✅ Debug + Release |

**16 of 16 contexts pass.** The previous AC6 + AppKit-E8-HP `armlink L6629E:
Unmatched parentheses` failure was caused by a Python-style `#` comment in
`board/AppKit-E8_M55_HP/RTE/Device/AE822FA0E5597BS0_M55_HP/app_mem_regions.h`
(line 193: `#define APP_HP_DTCM_SIZE 0x00030000  # Optimized from 1MB…`). C
preprocessor leaves a mid-line `#` in the preprocessed scatter expression,
which armlink rejects as an unbalanced parenthesis. Fixed by replacing the
`#` with a `//` comment. GCC tolerated the `#` because GNU `ld` doesn't
re-preprocess the scatter file, but armlink does. This is a project-side
fix on a header we already maintain (the file's a copy of the Alif DFP
template with project-specific size adjustments).

Runtime validation: AVH-SSE-320 GCC Debug runs both models on the
Corstone-320 FVP with NPU MACs/cc = 256 (matching the AOT compile
target `ethos-u85-256`); AppKit-E8-HP GCC Debug runs both models on
hardware (Alif AE822FA0E5597BS0, M55_HP) with `ra = rb = 0`.
