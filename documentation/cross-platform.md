# Running this example on Linux, macOS and Windows

The model export runs in a plain `venv` rather than a container, so the scripts
around it have to work on all three host OSes themselves. This page records how
that is arranged.

## The rule

**Everything the model-export flow needs is Python.** The `.sh` / `.bat` files
in the repository are thin wrappers that pick an interpreter and delegate; they
contain no logic.

Python is the one interpreter guaranteed to be present, because you already
need it to create the venv. Encoding the real behaviour once, in Python, is
what keeps the three platforms from drifting apart.

| Entry point | Role |
|---|---|
| `setup_venv.sh` (Linux/macOS), `setup_venv.bat` (Windows) | wrappers around `setup_venv.py` |
| `setup_venv.py` | creates `.venv` with torch, executorch, Vela |
| `create_ai_layer.py` | exports the model and writes the AI layer |

The two `setup_venv` wrappers invoke Python under *different* names,
deliberately: `python3` is the reliable name on Linux and macOS (many
distributions ship no bare `python`), while `python` is the reliable name on
Windows (`python3.exe` is not always installed). Both honour a `PYTHON`
environment variable if you need a specific interpreter:

```bash
PYTHON=python3.12 ./setup_venv.sh
```

## `bin/` vs `Scripts/`

A venv puts its interpreter in `bin/python` on POSIX and
`Scripts/python.exe` on Windows. That difference is encoded in exactly two
places, and both must stay in agreement:

- `venv_python()` in `setup_venv.py`
- `run_in_venv()` in `create_ai_layer.py`

Nothing else in the repository may hard-code either path.

## `create_ai_layer.py` finds the venv itself

`create_ai_layer.py` can be started with any interpreter -- `python3` on
POSIX, `python` on Windows, or the venv's own. If `torch` is not importable
from the interpreter it was started with, it re-runs itself with
`.venv/bin/python` (or `.venv/Scripts/python.exe`) and reports a clear error
when the venv has not been created yet. This is what lets the README, the CI
workflow and the VS Code task all use the same plain command:

```bash
python3 create_ai_layer.py cmsis-executorch.cbuild-mlops.yml
```

The build itself (`cbuild setup`, `cbuild`) is CMSIS-Toolbox only and needs no
Python, which is why there is no build-time hook to make host-OS aware.

## Windows specifics

### Long paths

`torch` unpacks paths long enough to exceed the legacy 260-character
`MAX_PATH`, which surfaces as an opaque pip failure partway through a
multi-gigabyte install rather than as a path error. `setup_venv.py` reads
`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled` and warns
before starting if it is off. To fix, in an elevated PowerShell:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Cloning nearer the drive root also works.

### Running the FVP on Windows

Nothing here is POSIX-only: building, exporting, and running the FVP from a
command line all work as they do elsewhere.

```powershell
FVP_Corstone_SSE-320 -f board/Corstone-320/fvp_config.txt `
    -a out/cmsis-executorch/SSE-320-U85/Debug/cmsis-executorch.axf
```

## CI coverage

`.github/workflows/build_pack_based.yml` has two jobs:

- **`build-and-run`** -- Linux only. The three build steps plus an FVP run
  asserting `Test_result: PASS`. Needs an Arm license, so it cannot be
  matrixed cheaply.
- **`venv-cross-platform`** -- `ubuntu-latest`, `macos-latest`,
  `windows-latest`. Runs `setup_venv` and starts `create_ai_layer.py`: no
  FVP, no license.

The second job is what actually protects the claims on this page. Without it,
Windows support regresses on the first refactor and nobody finds out.
