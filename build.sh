#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Build the example for the Corstone-320 / Ethos-U85 target. Acquires the
# CMSIS-Toolbox + arm-none-eabi-gcc via vcpkg, then runs cbuild with --active so
# the MLOps metadata (cbuild-mlops.yml) is generated and the model-conversion
# build step can consume it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

# Acquire the toolchain set declared in vcpkg-configuration.json.
export Z_VCPKG_POSTSCRIPT="$(mktemp /tmp/vcpkg.XXXXXX.sh)"
vcpkg activate
# shellcheck disable=SC1090
source "${Z_VCPKG_POSTSCRIPT}"

cbuild cmsis-executorch-simple.csolution.yml \
    --active SSE-320-U85 \
    --packs --update-rte \
    --context cmsis-executorch-simple.Debug+SSE-320-U85

echo
echo "ELF: $(find out -name '*.elf' | head -1)"
