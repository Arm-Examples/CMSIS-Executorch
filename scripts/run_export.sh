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
# is resolved at *configure* time, so a changed selection cannot take effect in
# this build -- fail with a re-run hint instead of failing later at link.
packs=( "${SOL}"/packs/PyTorch.ExecuTorch.* )
CLAYER="${SOL}/ai_layer/ai_layer.clayer.yml"
before="$(cat "${CLAYER}" 2>/dev/null || true)"
"${SOL}/.venv/bin/python3" "${SOL}/scripts/gen_components.py" \
    --pte       "${SOL}/model/model.pte" \
    --pack-path "${packs[0]}" \
    --output    "${CLAYER}"
if [[ "$(cat "${CLAYER}")" != "${before}" ]]; then
    echo "[run_export] The model's operator set changed: ai_layer.clayer.yml was regenerated." >&2
    echo "[run_export] Re-run the build to compile and link the updated component selection." >&2
    exit 1
fi
