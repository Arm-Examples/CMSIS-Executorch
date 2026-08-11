# Running this example on Linux, macOS and Windows

The model export runs in a plain `venv` rather than a container, so the scripts
around it have to work on all three host OSes themselves. This page records how
that is arranged and what is still missing.

## The rule

**Everything the model-export flow needs is Python.** The `.sh` / `.bat` /
`.ps1` files in the repository are thin wrappers that pick an interpreter and
delegate; none of them contain logic.

Python is the one interpreter guaranteed to be present, because you already
need it to create the venv. Encoding the real behaviour once, in Python, is
what keeps the three platforms from drifting apart.

| Wrapper | Delegates to |
|---|---|
| `setup_venv.sh` (Linux/macOS) | `setup_venv.py` |
| `setup_venv.bat` (Windows) | `setup_venv.py` |
| `build.sh` (Linux/macOS) | `cbuild` |
| `build.ps1` (Windows) | `cbuild` |
| `scripts/run_export.cmake` | `scripts/run_export.py` |

Note that the two `setup_venv` wrappers invoke Python under *different* names,
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
- the `CMAKE_HOST_WIN32` branch in `scripts/run_export.cmake`

Nothing else in the repository may hard-code either path.

## Why the build step goes through CMake

The model conversion is a cproject `executes:` node. It has to run one command
string on all three hosts, and `bash` is not present by default on Windows.

The obvious fix — an OS conditional on the step — does not exist. The csolution
schema (`common.schema.json` → `ExecuteType` in CMSIS-Toolbox 2.14.1) allows
only `for-context:` and `not-for-context:`, and those select *build and target
types*, not the host OS.

What is available is CMake. `cbuild` drives CMake, so `${CMAKE_COMMAND}` always
resolves, and the CMSIS-Toolbox documentation explicitly recommends routing
around OS-specific commands this way. Hence:

```yaml
executes:
  - execute: convert-model
    run: ${CMAKE_COMMAND} -P $input(0)$
    input:
      - $SolutionDir()$/scripts/run_export.cmake
      - $SolutionDir()$/scripts/run_export.py
      # ...
```

`run_export.cmake` does nothing but choose `.venv/Scripts/python.exe` or
`.venv/bin/python` and hand over to `run_export.py`, with a readable error if
the venv is missing.

Two constraints from the toolbox docs shape this:

- **The working directory of an `executes:` step is undefined** (typically
  `tmp/`). Both scripts self-locate — `CMAKE_CURRENT_LIST_DIR` and
  `Path(__file__)` respectively — and never rely on the caller's CWD.
- **CMake rejects Windows `\` separators** in the `run:` string, so every path
  in these files stays forward-slashed. CMake accepts that on Windows too.

`run_export.cmake` also uses `CMAKE_HOST_WIN32` rather than `WIN32`: in script
mode (`cmake -P`) there is no `project()` call, so the target-platform
variables are never set.

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

(The `Codespaces` branch, which wires the FVP onto the CMSIS Solution panel's
**Load / Run / Debug** buttons, does add a POSIX shell script in that path and
so needs Git Bash on Windows. This branch has no such wiring.)

```powershell
FVP_Corstone_SSE-320 -f board/Corstone-320/fvp_config.txt `
    -a out/cmsis-executorch-simple/SSE-320-U85/Debug/cmsis-executorch-simple.elf
```

## CI coverage

`.github/workflows/build_pack_based.yml` has two jobs:

- **`build-and-run`** — Linux only. Full build plus an FVP run asserting
  `Test_result: PASS`. Needs an Arm license, so it cannot be matrixed cheaply.
- **`venv-cross-platform`** — `ubuntu-latest`, `macos-latest`,
  `windows-latest`. Runs `setup_venv` and the export only: no FVP, no license.

The second job is what actually protects the claims on this page. Without it,
Windows support regresses on the first refactor and nobody finds out.
