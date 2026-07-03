#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Create the Python venv used by the model-conversion build step. No Docker.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

"${PY}" -m venv "${HERE}/.venv"
"${HERE}/.venv/bin/pip" install --upgrade pip
"${HERE}/.venv/bin/pip" install -r "${HERE}/requirements.txt"
# TOSA serializer (and its pinned peers) without pulling a conflicting torch,
# mirroring executorch's examples/arm/setup.sh.
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    "${HERE}/.venv/bin/pip" install --no-dependencies -r "${HERE}/requirements-arm-tosa.txt"

echo
echo "venv ready: ${HERE}/.venv"
echo "The build step invokes ${HERE}/.venv/bin/python automatically."
