#!/usr/bin/env bash
set -euo pipefail

# Stage 2: Build model-selective portable operator registration library (portable_ops_lib)
# while reusing the generic runtime from Stage 1 conceptually.
#
# Strategy: Re-configure ExecuTorch with selective operator registration using either:
#   1. EXECUTORCH_SELECT_OPS_MODEL for automatic model analysis, or
#   2. EXECUTORCH_SELECT_OPS_LIST for manual operator specification from file
#
# Usage:
#   scripts/build_stage2_selective.sh /path/to/executorch [/path/to/model.pte] [/abs/build/dir] [/path/to/toolchain.cmake] [/path/to/ops_list.txt]
# Examples:
#   # Model-based selection:
#   scripts/build_stage2_selective.sh "$HOME/src/executorch" dist/models/my_model.pte build/stage2
#   # Manual selection from file:
#   scripts/build_stage2_selective.sh "$HOME/src/executorch" "" build/stage2 "" ops_list.txt
#   # Manual selection inline:
#   EXECUTORCH_SELECT_OPS="aten::add.out,aten::relu.out" scripts/build_stage2_selective.sh "$HOME/src/executorch"
#
# Environment variables:
#   EXECUTORCH_EXTRA_CMAKE_ARGS  Space-separated extra cmake args to append.
#   EXECUTORCH_SELECT_OPS        Comma-separated operator list (overrides file/model)

EXECUTORCH_SRC=${1:-}
MODEL_PTE=${2:-}
BUILD_DIR=${3:-build/stage2}
TOOLCHAIN_FILE=${4:-}
OPS_LIST_FILE=${5:-}

if [[ -z "${EXECUTORCH_SRC}" ]]; then
  echo "[ERROR] Usage: $0 <executorch_src> [model.pte] [build_dir] [toolchain.cmake] [ops_list.txt]" >&2
  exit 1
fi
if [[ ! -d "${EXECUTORCH_SRC}" ]]; then
  echo "[ERROR] ExecuTorch source directory not found: ${EXECUTORCH_SRC}" >&2
  exit 1
fi

# Validate inputs - either model PTE or ops list should be provided
if [[ -n "${MODEL_PTE}" && ! -f "${MODEL_PTE}" ]]; then
  echo "[ERROR] Model PTE file not found: ${MODEL_PTE}" >&2
  exit 1
fi
if [[ -n "${OPS_LIST_FILE}" && ! -f "${OPS_LIST_FILE}" ]]; then
  echo "[ERROR] Operators list file not found: ${OPS_LIST_FILE}" >&2
  exit 1
fi

# Determine selection strategy
SELECTION_STRATEGY=""
SELECTION_VALUE=""

if [[ -n "${EXECUTORCH_SELECT_OPS:-}" ]]; then
  # Environment variable takes precedence
  SELECTION_STRATEGY="list"
  SELECTION_VALUE="${EXECUTORCH_SELECT_OPS}"
  echo "[Stage2] Using operator list from environment: ${SELECTION_VALUE}"
elif [[ -n "${OPS_LIST_FILE}" ]]; then
  # Read from file, filter out comments and empty lines
  SELECTION_STRATEGY="list"
  SELECTION_VALUE=$(grep -v '^\s*#' "${OPS_LIST_FILE}" | grep -v '^\s*$' | tr '\n' ',' | sed 's/,$//')
  echo "[Stage2] Using operator list from file: ${OPS_LIST_FILE}"
  echo "[Stage2] Operators: ${SELECTION_VALUE}"
elif [[ -n "${MODEL_PTE}" ]]; then
  # Model-based analysis
  SELECTION_STRATEGY="model"
  SELECTION_VALUE="${MODEL_PTE}"
  echo "[Stage2] Using model analysis from: ${MODEL_PTE}"
else
  echo "[ERROR] Must provide either MODEL_PTE, OPS_LIST_FILE, or EXECUTORCH_SELECT_OPS" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "[Stage2] Configuring selective portable ops build..."

# CMake Configuration Arguments:
# ┌──────────────────────────────────────┬────────┬────────────────────────────────────────────────────────────────────────┐
# │ CMake Argument                       │ Value  │ Explanation                                                            │
# ├──────────────────────────────────────┼────────┼────────────────────────────────────────────────────────────────────────┤
# │ CMAKE_BUILD_TYPE                     │ Release│ Sets build to Release mode (optimized, no debug symbols)               │
# │ EXECUTORCH_BUILD_ARM_BAREMETAL       │ ON     │ Enables Ethos-U backend and ARM baremetal target for embedded systems │
# │ EXECUTORCH_BUILD_PORTABLE_OPS        │ ON     │ Builds portable ops with selective operator registration               │
# │ EXECUTORCH_BUILD_KERNELS_QUANTIZED   │ OFF    │ Disables quantized ops (use Stage1 libs instead)                       │
# │ EXECUTORCH_BUILD_CORTEX_M            │ OFF    │ Disables Cortex-M ops (use Stage1 libs instead)                        │
# │ EXECUTORCH_BUILD_EXECUTOR_RUNNER     │ OFF    │ Disables executor runner utility (not needed for embedded deployment)  │
# │ EXECUTORCH_BUILD_DEVTOOLS            │ OFF    │ Disables development tools (not needed for production builds)          │
# │ EXECUTORCH_ENABLE_EVENT_TRACER       │ OFF    │ Disables event tracing to reduce binary size                           │
# │ EXECUTORCH_ENABLE_LOGGING            │ ON     │ Enables logging for debugging and monitoring                           │
# │ EXECUTORCH_OPTIMIZE_SIZE            │ ON     │ Optimize for binary size (critical for embedded targets)               │
# │ EXECUTORCH_ENABLE_PROGRAM_VERIFICATION│ OFF  │ Disables program verification to reduce overhead                       │
# │ EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL│ OFF  │ Disables runner utility extension (not needed for embedded)            │
# │ BUILD_TESTING                       │ OFF    │ Disables building test executables                                     │
# │ FETCH_ETHOS_U_CONTENT                │ OFF    │ Skips fetching Ethos-U content (assumes already available)             │
# │ EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD│ ON   │ Only include data types used by model (reduces binary size)            │
# │ EXECUTORCH_SELECT_OPS_MODEL          │ (model)│ Path to .pte model for automatic operator analysis (if using model)    │
# │ EXECUTORCH_SELECT_OPS_LIST           │ (list) │ Comma-separated operator list for manual selection (if using list)     │
# │ CMAKE_TOOLCHAIN_FILE                 │ (arg4) │ Specifies cross-compilation toolchain (e.g., arm-none-eabi-gcc.cmake)  │
# └──────────────────────────────────────┴────────┴────────────────────────────────────────────────────────────────────────┘
#
# Note: Stage2 builds ONLY selective portable_ops_lib. Quantized and Cortex-M kernels from Stage1 are reused.

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON           # Build portable ops with selection
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF     # Disable quantized ops  
  -DEXECUTORCH_BUILD_CORTEX_M=OFF              # Disable cortex-m ops
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF
  -DEXECUTORCH_BUILD_DEVTOOLS=OFF
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF
  -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=OFF
  -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF
  -DEXECUTORCH_ENABLE_LOGGING=ON
  -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=OFF
  -DEXECUTORCH_OPTIMIZE_SIZE=ON
  -DBUILD_TESTING=OFF
  -DFETCH_ETHOS_U_CONTENT=OFF
  -DEXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD=ON # Only build support for dtypes in model
)

# Add selection strategy to CMake args
if [[ "${SELECTION_STRATEGY}" == "model" ]]; then
  CMAKE_ARGS+=("-DEXECUTORCH_SELECT_OPS_MODEL=${SELECTION_VALUE}")
  echo "[Stage2] Strategy: Model analysis"
elif [[ "${SELECTION_STRATEGY}" == "list" ]]; then
  CMAKE_ARGS+=("-DEXECUTORCH_SELECT_OPS_LIST=${SELECTION_VALUE}")
  echo "[Stage2] Strategy: Manual operator list"
fi

if [[ -n "${TOOLCHAIN_FILE}" ]]; then
  CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}")
fi

if [[ -n "${EXECUTORCH_EXTRA_CMAKE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SPLIT=(${EXECUTORCH_EXTRA_CMAKE_ARGS})
  CMAKE_ARGS+=("${EXTRA_SPLIT[@]}")
fi

cmake "${EXECUTORCH_SRC}" "${CMAKE_ARGS[@]}" -B .

echo "[Stage2] Building selective operator library..."
# Build portable_ops_lib with selective operator list
cmake --build . -j"$(nproc)" --target portable_ops_lib

echo "[Stage2] Copying artifacts..."
ASSETS_DIR="${BUILD_DIR}/assets"
mkdir -p "${ASSETS_DIR}/lib" "${ASSETS_DIR}/meta" "${ASSETS_DIR}/include"

# Look for the selective portable_ops_lib.a
FOUND_LIB=$(find . -maxdepth 4 -name "libportable_ops_lib.a" | head -n1 || true)
if [[ -n "$FOUND_LIB" ]]; then
  cp "$FOUND_LIB" "${ASSETS_DIR}/lib/"
  echo "[Stage2] Copied libportable_ops_lib.a from $FOUND_LIB -> ${ASSETS_DIR}/lib/"
else
  echo "[Stage2][ERROR] No selective library libportable_ops_lib.a found." >&2
  exit 2
fi

SELECTED_YAML=$(find . -name selected_operators.yaml | head -n1 || true)
if [[ -f "${SELECTED_YAML}" ]]; then
  cp "${SELECTED_YAML}" "${ASSETS_DIR}/meta/selected_operators.yaml"
fi

# Collect generated selective registration headers (they may live under arm_portable_ops_lib or executorch_selected_kernels dirs)
echo "[Stage2] Collecting generated selective headers."
SELECTIVE_HDR_DIR="$ASSETS_DIR/include/executorch/generated"
mkdir -p "$SELECTIVE_HDR_DIR"
find . -type f \( -name 'Functions.h' -o -name 'NativeFunctions.h' -o -name 'CustomOpsNativeFunctions.h' \) -print0 2>/dev/null | while IFS= read -r -d '' H; do
  SUBDIR=$(dirname "$H")
  TARGET="$SELECTIVE_HDR_DIR/${SUBDIR#./}"
  mkdir -p "$TARGET"
  cp "$H" "$TARGET/"
done

if [[ "${EXECUTORCH_HEADER_MODE:-}" == "all" ]]; then
  echo "[Stage2] EXECUTORCH_HEADER_MODE=all -> syncing source headers (runtime,kernels,backends,extension)."
  for DIR in runtime kernels backends extension; do
    if [[ -d "${EXECUTORCH_SRC}/${DIR}" ]]; then
      rsync -a --ignore-existing "${EXECUTORCH_SRC}/${DIR}" "$ASSETS_DIR/include/executorch/"
    fi
  done
fi

HDR_TOTAL=$(find "$ASSETS_DIR/include" -type f -name '*.h' | wc -l | tr -d ' ' || true)
echo "[Stage2] Header total: $HDR_TOTAL"

echo "[Stage2] Done. Selective library & YAML under: ${ASSETS_DIR}"
echo "         Library name may vary by ExecuTorch version (executorch_selected_kernels vs portable_ops_lib)."
