#!/usr/bin/env bash
set -euo pipefail

# Package ExecuTorch SDK artifacts (headers + libs + optional model) for external application projects.
#
# Usage:
#   scripts/package_sdk.sh <stage1_assets_dir> [stage2_assets_dir] [model.pte] [output_dir]
# Example:
#   scripts/package_sdk.sh build/stage1/install build/stage2/install dist/models/my_model.pte dist/sdk
#
# Resulting layout:
#   <output_dir>/include/            ExecuTorch headers
#   <output_dir>/lib/                Core & kernel libs (+ optional selective portable ops lib)
#   <output_dir>/model/              Model .pte (if provided)
#   <output_dir>/meta/selected_operators.yaml (if available)

STAGE1_INSTALL=${1:-}
STAGE2_INSTALL=${2:-}
MODEL_PTE=${3:-}
OUT_DIR=${4:-dist/sdk}
EXECUTORCH_SRC=${EXECUTORCH_SRC:-}

# Derive stage1 build root to locate third-party outputs (flatbuffers from flatc_ep).
STAGE1_ROOT=$(cd "${STAGE1_INSTALL}"/.. && pwd)

if [[ -z "${STAGE1_INSTALL}" || ! -d "${STAGE1_INSTALL}" ]]; then
  echo "[ERROR] Stage1 install dir invalid: ${STAGE1_INSTALL}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/include" "${OUT_DIR}/lib" "${OUT_DIR}/model" "${OUT_DIR}/meta"

echo "[SDK] Copying headers..."
# Avoid recursive include explosion from generated/assets inside stage1 include tree.
if [[ -d "${STAGE1_INSTALL}/include/executorch/generated/assets" ]]; then
  rm -rf "${STAGE1_INSTALL}/include/executorch/generated/assets"
fi
rsync -a --delete "${STAGE1_INSTALL}/include/" "${OUT_DIR}/include/"

# Add FlatBuffers headers produced by stage1 (flatc_ep) so downstream CMSIS builds resolve flatbuffers/flatbuffers.h.
FLATBUF_SRC="${STAGE1_ROOT}/third-party/flatc_ep/include/flatbuffers"
# Place headers under .../include/third-party/flatbuffers/include/flatbuffers so
# -I<path> lets "flatbuffers/flatbuffers.h" resolve correctly.
FLATBUF_DST="${OUT_DIR}/include/third-party/flatbuffers/include/flatbuffers"
if [[ -d "${FLATBUF_SRC}" ]]; then
  mkdir -p "${FLATBUF_DST}"
  rsync -a "${FLATBUF_SRC}/" "${FLATBUF_DST}/"
  echo "[SDK] Added FlatBuffers headers from ${FLATBUF_SRC}"
else
  echo "[SDK] WARN: FlatBuffers headers not found at ${FLATBUF_SRC}" >&2
fi

# Add extended_header.h from ExecuTorch source (not in the stage1 assets install).
if [[ -n "${EXECUTORCH_SRC}" ]]; then
  EXT_HDR_SRC="${EXECUTORCH_SRC}/schema/extended_header.h"
  EXT_HDR_DST_DIR="${OUT_DIR}/include/executorch/schema"
  if [[ -f "${EXT_HDR_SRC}" ]]; then
    mkdir -p "${EXT_HDR_DST_DIR}"
    cp "${EXT_HDR_SRC}" "${EXT_HDR_DST_DIR}/"
    echo "[SDK] Added extended_header.h from ${EXT_HDR_SRC}"
  else
    echo "[SDK] WARN: extended_header.h not found at ${EXT_HDR_SRC}" >&2
  fi

  # Copy flatcc headers from ExecuTorch source (used by flatcc runtime C sources).
  FLATCC_SRC="${EXECUTORCH_SRC}/third-party/flatcc/include/flatcc"
  FLATCC_DST="${OUT_DIR}/include/third-party/flatcc/include/flatcc"
  if [[ -d "${FLATCC_SRC}" ]]; then
    mkdir -p "${FLATCC_DST}"
    rsync -a "${FLATCC_SRC}/" "${FLATCC_DST}/"
    echo "[SDK] Added flatcc headers from ${FLATCC_SRC}"
  else
    echo "[SDK] WARN: flatcc headers not found at ${FLATCC_SRC}" >&2
  fi

  # Copy Cortex-M backend headers (e.g., cortex_m_ops_common.h) needed by CMSIS build.
  CORTEXM_SRC="${EXECUTORCH_SRC}/backends/cortex_m"
  CORTEXM_DST="${OUT_DIR}/include/executorch/backends/cortex_m"
  if [[ -d "${CORTEXM_SRC}" ]]; then
    mkdir -p "${CORTEXM_DST}"
    rsync -a --include='*/' --include='*.h' --exclude='*' \
      "${CORTEXM_SRC}/" "${CORTEXM_DST}/"
    echo "[SDK] Added Cortex-M backend headers from ${CORTEXM_SRC}"
  else
    echo "[SDK] WARN: Cortex-M backend headers not found at ${CORTEXM_SRC}" >&2
  fi
else
  echo "[SDK] NOTE: EXECUTORCH_SRC not set; skipping extended_header.h copy" >&2
fi

echo "[SDK] Copying core libraries from Stage1..."
if [[ -d "${STAGE1_INSTALL}/lib" ]]; then
  rsync -a "${STAGE1_INSTALL}/lib/" "${OUT_DIR}/lib/"
else
  echo "[SDK] WARN: ${STAGE1_INSTALL}/lib not found"
fi

if [[ -n "${STAGE2_INSTALL}" && -d "${STAGE2_INSTALL}" ]]; then
  echo "[SDK] Overlaying selective ops lib from Stage2..."
  SELECTIVE_COPIED=false
  if [[ -d "${STAGE2_INSTALL}/lib" ]]; then
    for CAND in libexecutorch_selected_kernels.a libportable_ops_lib.a; do
      if [[ -f "${STAGE2_INSTALL}/lib/${CAND}" ]]; then
        cp "${STAGE2_INSTALL}/lib/${CAND}" "${OUT_DIR}/lib/"
        echo "[SDK] Added ${CAND}"; SELECTIVE_COPIED=true; break
      fi
    done
  else
    echo "[SDK] WARN: ${STAGE2_INSTALL}/lib not found"
  fi
  if [[ "${SELECTIVE_COPIED}" = false ]]; then
  echo "[SDK] WARN: No selective ops library found in Stage2." >&2
  fi
  if [[ -f "${STAGE2_INSTALL}/meta/selected_operators.yaml" ]]; then
    cp "${STAGE2_INSTALL}/meta/selected_operators.yaml" "${OUT_DIR}/meta/"
  fi
fi

if [[ -n "${MODEL_PTE}" && -f "${MODEL_PTE}" ]]; then
  echo "[SDK] Copying model PTE..."
  cp "${MODEL_PTE}" "${OUT_DIR}/model/"
fi

echo "[SDK] Package complete at: ${OUT_DIR}"
