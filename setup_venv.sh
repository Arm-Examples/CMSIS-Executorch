#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Linux/macOS wrapper. All the logic lives in setup_venv.py so the same setup
# runs on Windows too; this only picks an interpreter. Override with e.g.
#   PYTHON=python3.12 ./setup_venv.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "${HERE}/setup_venv.py" "$@"
