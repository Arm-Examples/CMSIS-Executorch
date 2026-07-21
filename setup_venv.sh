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

# Sanity check: import the module the export flow actually needs. A torch /
# executorch version mismatch in this venv (e.g. a stray torch upgrade) would
# otherwise only fail later, at export time, with a confusing error - e.g.
# executorch <= 1.3.1 referenced named-tensor op overloads that torch 2.13
# removed, which surfaced as an AttributeError deep in this import.
"${HERE}/.venv/bin/python" - <<'EOF'
import sys

try:
    import executorch.backends.arm.quantizer.quantization_annotator  # noqa: F401
except Exception as exc:
    import torch, executorch  # noqa: E401
    from importlib.metadata import version
    sys.exit(
        "error: the executorch Arm quantizer failed to import "
        f"(torch {torch.__version__}, executorch {version('executorch')}):\n"
        f"  {type(exc).__name__}: {exc}\n"
        "The torch and executorch versions in this venv are likely mismatched. "
        "Recreate the venv with the pinned versions: rm -rf .venv && ./setup_venv.sh"
    )
EOF

echo
echo "venv ready: ${HERE}/.venv"
echo "The build step invokes ${HERE}/.venv/bin/python automatically."
