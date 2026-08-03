#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Devcontainer/Codespaces bootstrap. Deliberately minimal: the Arm Environment
# Manager (part of the Keil Studio extension pack) reads
# vcpkg-configuration.json itself and installs/activates CMSIS-Toolbox,
# arm-none-eabi-gcc, cmake, ninja and the AVH FVPs (incl.
# FVP_Corstone_SSE-320), and it bundles armlm for the license activation.
# This script only covers what the extension does not:
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

# FVP runtime dependencies (libpython3.9 via deadsnakes on Ubuntu 22.04) and
# lsof for .vscode/start-fvp-gdb.sh. software-properties-common first: the
# devcontainer base image does not ship add-apt-repository.
sudo apt-get update
sudo apt-get install -y --no-install-recommends software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y --no-install-recommends \
    libatomic1 libpython3.9 libstdc++6 lsof python3-venv python3-pip
sudo rm -rf /var/lib/apt/lists/*

# fvp-gdb: GDB-RSP front-end for the FVP's Iris interface; drives the CMSIS
# Solution panel's Load/Run/Debug buttons (see .vscode/start-fvp-gdb.sh).
python3 -m pip install --user --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple fvp-gdb
sudo ln -sf "${HOME}/.local/bin/fvp-gdb" /usr/local/bin/fvp-gdb

# Model-export venv (executorch + ethos-u-vela). setup_venv.py is idempotent
# and rebuilds the venv by itself if its interpreter no longer runs -- the case
# after a container rebuild, since /workspaces persists but the image's Python
# does not.
./setup_venv.sh

echo
echo "Devcontainer ready. The Arm Environment Manager installs the vcpkg"
echo "artifacts (toolbox, gcc, FVPs) on first activation and prompts for the"
echo "license. Then: ./build.sh, and use the CMSIS Solution panel's buttons."
