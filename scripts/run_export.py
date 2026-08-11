#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Build-step launcher for the model conversion. Invoked by the cproject
# `executes:` node through scripts/run_export.cmake, which picks the venv
# interpreter for the host OS and runs this file with it.
#
# Self-locates the solution directory: the working directory of an `executes:`
# step is not controlled by CMSIS-Toolbox (it is typically tmp/), so nothing
# here may assume the caller's CWD.
"""Export the model, then refresh the AI layer's component selection."""

from __future__ import annotations

import filecmp
import re
import shutil
import subprocess
import sys
from pathlib import Path

SOL = Path(__file__).resolve().parent.parent
PACK = "PyTorch::ExecuTorch"


def run(*args: str | Path) -> None:
    cmd = [str(a) for a in args]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def resolved_pack_version(sol: Path) -> str | None:
    """The pack version CMSIS-Toolbox settled on, from its own record of the
    resolution. Without it gen_components.py falls back to the newest version
    installed, which is only the same thing when one is installed."""
    # A resolved-pack entry is the selected one only when a selected-by-pack
    # list follows it; the file also records versions that are merely installed.
    selected = rf"resolved-pack: {re.escape(PACK)}@(\S+)\n\s+selected-by-pack:"
    for record in sol.glob("*.cbuild-pack.yml"):
        found = re.search(selected, record.read_text())
        if found:
            return found.group(1)
    return None


def main() -> int:
    # sys.executable is the venv interpreter: run_export.cmake invoked us with it.
    python = sys.executable

    run(
        python,
        SOL / "model" / "export_model.py",
        "--mlops",
        SOL / "cmsis-executorch-simple.cbuild-mlops.yml",
        "--output-dir",
        SOL / "model",
        "--header",
        SOL / "ai_layer" / "model" / "model_pte.h",
    )

    # Regenerate the AI layer's component selection from the fresh .pte. The
    # clayer is resolved at *configure* time: touching it mid-build makes
    # CMSIS-Toolbox reconfigure underneath the running ninja, so only move the
    # regenerated file into place when its content actually changed -- and then
    # fail with a re-run hint, since the new selection cannot take effect in
    # this build anyway.
    clayer = SOL / "ai_layer" / "ai_layer.clayer.yml"
    new = clayer.with_suffix(clayer.suffix + ".new")

    version = resolved_pack_version(SOL)
    run(
        python,
        SOL / "scripts" / "gen_components.py",
        "--pte",
        SOL / "model" / "model.pte",
        *(("--pack-version", version) if version else ()),
        "--output",
        new,
    )

    if clayer.is_file() and filecmp.cmp(new, clayer, shallow=False):
        new.unlink()
        return 0

    shutil.move(str(new), str(clayer))
    print(
        "[run_export] The model's operator set changed: "
        "ai_layer.clayer.yml was regenerated.",
        file=sys.stderr,
    )
    print(
        "[run_export] Re-run the build to compile and link the updated "
        "component selection.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
