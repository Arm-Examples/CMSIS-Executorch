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

# Sanity check: executorch 1.3.1's Arm quantizer references named-tensor op
# overloads (torch.ops.aten.transpose.Dimname) that PyTorch 2.13 removed. A
# stray torch upgrade in this venv would only fail later, at export time, with
# a confusing AttributeError - catch it here instead.
"${HERE}/.venv/bin/python" - <<'EOF'
import sys
import torch

try:
    torch.ops.aten.transpose.Dimname
except AttributeError:
    sys.exit(
        f"error: torch {torch.__version__} in this venv has no named-tensor op "
        "overloads (removed in torch 2.13), which executorch 1.3.1 still needs.\n"
        "Recreate the venv so pip resolves the torch version pinned by "
        "executorch: rm -rf .venv && ./setup_venv.sh"
    )
EOF

echo
echo "venv ready: ${HERE}/.venv"
echo "The build step invokes ${HERE}/.venv/bin/python automatically."
