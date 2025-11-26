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

if [[ -z "${STAGE1_INSTALL}" || ! -d "${STAGE1_INSTALL}" ]]; then
  echo "[ERROR] Stage1 install dir invalid: ${STAGE1_INSTALL}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/include" "${OUT_DIR}/lib" "${OUT_DIR}/model" "${OUT_DIR}/meta"

echo "[SDK] Copying headers..."
rsync -a --delete "${STAGE1_INSTALL}/include/" "${OUT_DIR}/include/"

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
