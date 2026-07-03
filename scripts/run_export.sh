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
