#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Linux/macOS wrapper. Environment setup lives in setup_venv.py so the same setup
# runs on Windows too; this only picks an interpreter. Override with e.g.
#   PYTHON=python3.12 ./setup_venv.sh
# All options are forwarded, e.g. ./setup_venv.sh --uv --python 3.12
# Add --recreate to change the Python version of an existing environment.
# With --uv, uv supplies the launcher too; no system Python is required.
# Otherwise PYTHON selects the launcher.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
use_uv=false
uv_python='>=3.10,<3.15'
previous=''
for arg in "$@"; do
    if [[ "$previous" == --python ]]; then
        uv_python="$arg"
    fi
    case "$arg" in
        --uv) use_uv=true ;;
        --python=*) uv_python="${arg#--python=}" ;;
    esac
    previous="$arg"
done
if "$use_uv"; then
    if ! command -v uv >/dev/null 2>&1; then
        echo 'error: --uv requires uv on PATH; install it from https://docs.astral.sh/uv/getting-started/installation/' >&2
        exit 2
    fi
    # Isolation lets --recreate remove .venv without removing the running launcher.
    exec uv run --no-project --isolated --python "$uv_python" "${HERE}/setup_venv.py" "$@"
fi
exec "${PYTHON:-python3}" "${HERE}/setup_venv.py" "$@"
