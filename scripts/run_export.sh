#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Build-step launcher for the model conversion. Invoked by the cproject
# `executes:` node as `bash $input(0)$`; self-locates the solution directory so
# it does not depend on the caller's working directory.
set -euo pipefail
SOL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${SOL}/.venv/bin/python3" "${SOL}/model/export_model.py" \
    --mlops      "${SOL}/cmsis-executorch-simple.cbuild-mlops.yml" \
    --output-dir "${SOL}/model" \
    --header     "${SOL}/ai_layer/model/model_pte.h"

# Regenerate the AI layer's component selection from the fresh .pte. The clayer
# is resolved at *configure* time: touching it mid-build makes CMSIS-Toolbox
# reconfigure underneath the running ninja, so only move the regenerated file
# into place when its content actually changed -- and then fail with a re-run
# hint, since the new selection cannot take effect in this build anyway.
packs=( "${SOL}"/packs/PyTorch.ExecuTorch.* )
CLAYER="${SOL}/ai_layer/ai_layer.clayer.yml"
NEW="${CLAYER}.new"
"${SOL}/.venv/bin/python3" "${SOL}/scripts/gen_components.py" \
    --pte       "${SOL}/model/model.pte" \
    --pack-path "${packs[0]}" \
    --output    "${NEW}"
if cmp -s "${NEW}" "${CLAYER}"; then
    rm -f "${NEW}"
else
    mv "${NEW}" "${CLAYER}"
    echo "[run_export] The model's operator set changed: ai_layer.clayer.yml was regenerated." >&2
    echo "[run_export] Re-run the build to compile and link the updated component selection." >&2
    exit 1
fi
