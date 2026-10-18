#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Builds the judged DCSM-Simplified firmware: camera classifier + voice keywords in one HP image.
#   vision: resources/object_detection/dcops/dcops_v4_demo_ft_ethos-u85-256.pte (board-camera fine-tune; VISION_PTE overrides)
#   voice:  resources/kws/dcops/dscnn_m_agc_ethos-u85-256.pte
#   shown classes: power shelf, network port, fan, power connector, drive bay (+ background)
#
# Usage: scripts/alif/build_dcops_demo.sh [build-dir]     (default: build-fusion)
# Then:  scripts/alif/flash_e8_hp.sh <build-dir>/bin/mlek_alif_object_detection.axf
#        and power-cycle the board (unplug PRG USB ~10 s) so the camera restarts.
# Needs: arm-none-eabi-gcc on PATH (or ARM_GCC_BIN pointing at its bin directory), cmake, ninja,
#        the dependencies/ submodules (ExecuTorch v1.1.0).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${1:-$ROOT/build-fusion}"
[[ -n "${ARM_GCC_BIN:-}" ]] && export PATH="$ARM_GCC_BIN:$PATH"
command -v arm-none-eabi-gcc >/dev/null || { echo "arm-none-eabi-gcc not on PATH (set ARM_GCC_BIN)" >&2; exit 1; }

VISION="$ROOT/resources/object_detection/dcops"
KWS="$ROOT/resources/kws"

# Class ids (labels.txt order): 4 power shelf, 6 network port, 9 fan, 12 power connector,
# 13 drive bay, 16 background -> mask 0x13250. Use 0xFFFFFFFF to show all 16 classes.
MASK="${CLS_ALLOWED_MASK:-0x13250}"

cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/scripts/cmake/toolchains/bare-metal-gcc.cmake" \
    -DTARGET_PLATFORM=alif -DTARGET_SUBSYSTEM=RTSS-HP -DTARGET_BOARD=DevKit-e8 \
    -DALIF_DEVICE_SKU=AE822FA0E5597 -DALIF_CAMERA_MODULE=MT9M114 \
    -DML_FRAMEWORK=ExecuTorch -DETHOS_U_NPU_ID=U85 -DETHOS_U_NPU_CONFIG_ID=Z256 \
    -DETHOS_U_NPU_MEMORY_MODE=Shared_Sram \
    -DALIF_CAMERA_ENABLED=ON -DALIF_ISP_ENABLED=OFF \
    -DUSE_CASE_BUILD=alif_object_detection \
    -Dalif_object_detection_ET_TASK=CLASSIFY \
    -Dalif_object_detection_MODEL_PATH="$VISION/${VISION_PTE:-dcops_v4_demo_ft_ethos-u85-256.pte}" \
    -Dalif_object_detection_CLASS_LABELS="$VISION/labels.txt" \
    -Dalif_object_detection_ACTIVATION_BUF_SZ=0x00180000 \
    -Dalif_object_detection_CLS_EMA_ALPHA=0.3 \
    -Dalif_object_detection_CLS_CONF_THRESHOLD="${CLS_CONF_THRESHOLD:-0.6}" \
    -Dalif_object_detection_CLS_ALLOWED_MASK="$MASK" \
    -Dalif_object_detection_ENABLE_VOICE=ON \
    -Dalif_object_detection_KWS_MODEL_PATH="$KWS/dcops/dscnn_m_agc_ethos-u85-256.pte" \
    -Dalif_object_detection_KWS_LABELS="$KWS/labels/micronet_kws_labels.txt" \
    -Dalif_object_detection_KWS_ACTIVATION_BUF_SZ=0x00020000 \
    -Dalif_object_detection_KWS_THRESHOLD="${KWS_THRESHOLD:-0.5}"
cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

AXF="$BUILD/bin/mlek_alif_object_detection.axf"
END=$(arm-none-eabi-readelf -lW "$AXF" | python3 -c '
import sys
ends = [int(p[3], 16) + int(p[4], 16) for p in (l.split() for l in sys.stdin)
        if p and p[0] == "LOAD" and int(p[3], 16) >= 0x80000000 and int(p[4], 16) > 0]
print(hex(max(ends)))')
echo "built $AXF; MRAM image ends at $END (flash guard limit 0x80578000)"
