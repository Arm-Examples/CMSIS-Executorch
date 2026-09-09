#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Create the Python venv used by the model-conversion build step. No Docker.
#
# Runs on Linux, macOS and Windows. The thin wrappers setup_venv.sh and
# setup_venv.bat just delegate here; everything OS-specific lives in this file.
"""Create (or repair) the .venv used to export the model."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_DIR = HERE / ".venv"

# ExecuTorch 1.4 declares requires-python = ">=3.10,<3.15" in its pyproject.toml.
# Check it up front: without this, an unsupported interpreter fails much later
# with a resolver error that says nothing about the Python version.
MIN_PYTHON = (3, 10)
MAX_PYTHON_EXCLUSIVE = (3, 15)

EXECUTORCH_REPO = "https://github.com/pytorch/executorch.git"


def venv_python(venv_dir: Path) -> Path:
    """Path to the interpreter inside a venv, on any host OS.

    Windows puts it in Scripts/python.exe, everyone else in bin/python. This is
    the one place that difference is encoded; scripts/run_export.cmake makes the
    same choice for the build step.
    """
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def check_host_python() -> None:
    if not (MIN_PYTHON <= sys.version_info[:2] < MAX_PYTHON_EXCLUSIVE):
        have = ".".join(str(n) for n in sys.version_info[:3])
        lo = ".".join(str(n) for n in MIN_PYTHON)
        hi = ".".join(str(n) for n in MAX_PYTHON_EXCLUSIVE)
        sys.exit(
            f"error: ExecuTorch needs Python >={lo},<{hi}; this is {have}\n"
            f"  ({sys.executable})\n"
            "Re-run with a supported interpreter, e.g.\n"
            "  PYTHON=python3.12 ./setup_venv.sh        (Linux/macOS)\n"
            "  py -3.12 setup_venv.py                   (Windows)"
        )


def warn_windows_long_paths() -> None:
    """Warn before pip fails halfway through a multi-GB torch install.

    torch unpacks paths long enough to exceed the legacy 260-character MAX_PATH,
    which surfaces as an opaque failure deep inside pip rather than as a path
    error. Nothing here is fatal: the install often succeeds anyway if the
    workspace sits near the drive root.
    """
    if os.name != "nt":
        return
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            enabled, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
    except OSError:
        return  # Key missing or unreadable; not worth failing over.

    if not enabled:
        print(
            "warning: Windows long paths are disabled. Installing torch into a\n"
            "         deeply nested workspace may fail with a confusing pip error.\n"
            "         Enable them (elevated PowerShell) with:\n"
            '           New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" \\\n'
            '             -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force\n'
            "         ...or clone this repository closer to the drive root.\n",
            file=sys.stderr,
        )


def venv_is_usable(venv_dir: Path) -> bool:
    """True if the venv exists and its interpreter still runs.

    /workspaces persists across devcontainer rebuilds, so an existing .venv can
    reference the previous image's interpreter: the directory is there but the
    symlinks and lib/pythonX.Y paths are stale.
    """
    python = venv_python(venv_dir)
    if not python.is_file():
        return False
    try:
        subprocess.run(
            [str(python), "-c", "import sys"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def python_version(value: str) -> tuple[int, ...]:
    """Accept a supported major.minor version, optionally with a patch version."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", value):
        raise argparse.ArgumentTypeError("expected a Python version such as 3.12 or 3.12.10")
    version = tuple(int(n) for n in value.split("."))
    if not (MIN_PYTHON <= version[:2] < MAX_PYTHON_EXCLUSIVE):
        raise argparse.ArgumentTypeError("ExecuTorch needs Python >=3.10,<3.15")
    return version


def check_venv_python(python: Path, requested: tuple[int, ...] | None) -> None:
    result = subprocess.run(
        [str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        check=True,
        capture_output=True,
        text=True,
    )
    have = result.stdout.strip()
    version = tuple(int(n) for n in have.split("."))
    if not (MIN_PYTHON <= version[:2] < MAX_PYTHON_EXCLUSIVE):
        sys.exit(
            f"error: {python} uses Python {have}; ExecuTorch needs >=3.10,<3.15. "
            "Re-run with --recreate and a supported Python version."
        )
    if requested and version[:len(requested)] != requested:
        want = ".".join(map(str, requested))
        sys.exit(
            f"error: .venv uses Python {have}, but --python {want} was requested. "
            "Re-run with --recreate to change the environment's Python version."
        )


def pip(
    python: Path, *args: str, uv: str | None = None, env: dict[str, str] | None = None
) -> None:
    if uv:
        # Match pip's selection across PyPI and the PyTorch nightly index:
        # uv's default first-index strategy can hide the pinned nightly wheels.
        cmd = [
            uv, "pip", *args, "--python", str(python),
            "--index-strategy", "unsafe-best-match",
        ]
    else:
        cmd = [str(python), "-m", "pip", *args]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def smoke_test(python: Path) -> None:
    """Import the module the export flow actually needs.

    A torch/executorch version mismatch in this venv (a stray torch upgrade, a
    nightly that moved on) would otherwise only surface later, at export time,
    as an AttributeError deep inside this import.
    """
    script = """
import sys

try:
    import executorch.backends.arm.quantizer.quantization_annotator  # noqa: F401
except Exception as exc:
    from importlib.metadata import version

    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        torch_version = "not installed"
    try:
        et_version = version("executorch")
    except Exception:
        et_version = "not installed"
    sys.exit(
        "error: the executorch Arm quantizer failed to import "
        f"(torch {torch_version}, executorch {et_version}):\\n"
        f"  {type(exc).__name__}: {exc}\\n"
        "The torch and executorch versions in this venv are likely mismatched. "
        "Recreate it with: python setup_venv.py --recreate"
    )
"""
    subprocess.run([str(python), "-c", script], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uv",
        action="store_true",
        help=(
            "create the environment with uv venv and install packages with uv pip "
            "(requires uv on PATH)"
        ),
    )
    parser.add_argument(
        "--python",
        metavar="VERSION",
        type=python_version,
        help=(
            "Python version for uv, e.g. 3.12 or 3.12.10 "
            "(requires --uv; defaults to the setup interpreter)"
        ),
    )
    parser.add_argument(
        "--executorch-ref",
        metavar="REF",
        help=(
            "install executorch from this git ref of pytorch/executorch "
            "(e.g. release/1.4) instead of the pinned wheel. Builds from "
            "source: needs CMake and a C++ toolchain, and takes tens of minutes."
        ),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="delete and rebuild .venv even if it looks usable",
    )
    args = parser.parse_args()

    if args.python and not args.uv:
        parser.error("--python requires --uv")
    uv = shutil.which("uv") if args.uv else None
    if args.uv and not uv:
        parser.error(
            "--uv requires uv on PATH; install it from "
            "https://docs.astral.sh/uv/getting-started/installation/"
        )
    if not args.python:
        check_host_python()
    warn_windows_long_paths()

    if args.recreate and VENV_DIR.exists():
        print(f"Removing {VENV_DIR}")
        shutil.rmtree(VENV_DIR)

    if VENV_DIR.exists() and not venv_is_usable(VENV_DIR):
        print(f"{VENV_DIR} exists but its interpreter does not run; recreating.")
        shutil.rmtree(VENV_DIR)

    if not VENV_DIR.exists():
        print(f"Creating venv at {VENV_DIR}")
        if uv:
            # The wrapper may run us in uv's temporary isolated environment.
            # Request its Python version, not a path inside that environment.
            requested = ".".join(map(str, args.python or sys.version_info[:3]))
            cmd = [uv, "venv", "--python", requested, str(VENV_DIR)]
            print(f"+ {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)
        else:
            venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(VENV_DIR)

    python = venv_python(VENV_DIR)
    check_venv_python(python, args.python)
    if not uv:
        pip(python, "install", "--upgrade", "pip")

    # Pass 1: everything that resolves from PyPI. Kept free of any index
    # directive so pip cannot prefer a nightly torch over the pinned release.
    pip(python, "install", "-r", str(HERE / "requirements.txt"), uv=uv)

    # Pass 2: executorch + torchao. From the PyTorch nightly index (see the
    # file header), or from a git ref when the caller asked for one.
    if args.executorch_ref:
        print(
            f"\nBuilding executorch from {EXECUTORCH_REPO}@{args.executorch_ref}.\n"
            "This is a source build: it needs CMake and a C++ toolchain and\n"
            "takes tens of minutes. Ctrl-C now to use the pinned wheel instead.\n",
            file=sys.stderr,
        )
        pip(python, "install", f"git+{EXECUTORCH_REPO}@{args.executorch_ref}", uv=uv)
        # The git install brings no torchao pin; take the one 1.4 expects.
        pip(
            python,
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/nightly/cpu",
            "--extra-index-url",
            "https://pypi.org/simple",
            "torchao==0.18.0.dev20260715",
            uv=uv,
        )
    else:
        pip(python, "install", "-r", str(HERE / "requirements-executorch.txt"), uv=uv)

    # Pass 3: the TOSA serializer, without dependencies. See the header of
    # requirements-arm-tosa.txt for why --no-dependencies is load-bearing.
    # CMAKE_POLICY_VERSION_MINIMUM keeps an sdist fallback building under CMake 4.
    env = dict(os.environ, CMAKE_POLICY_VERSION_MINIMUM="3.5")
    pip(
        python,
        "install",
        "--no-deps",
        "-r",
        str(HERE / "requirements-arm-tosa.txt"),
        uv=uv,
        env=env,
    )

    smoke_test(python)

    print()
    print(f"venv ready: {VENV_DIR}")
    print("The build step invokes this interpreter automatically:")
    print(f"  {python}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
