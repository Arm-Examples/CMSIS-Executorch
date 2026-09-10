@echo off
REM Copyright 2026 Arm Limited and/or its affiliates.
REM SPDX-License-Identifier: Apache-2.0
REM
REM Windows wrapper. All the logic lives in setup_venv.py. "python" rather
REM than "python3": python3.exe is not reliably present on Windows, while
REM python3 is the reliable name on Linux/macOS -- which is why this wrapper
REM and setup_venv.sh differ. With --uv, uv supplies the interpreter, e.g.
REM   setup_venv.bat --uv --python 3.12
setlocal
set "SETUP_USE_UV="
set "SETUP_UV_PYTHON=>=3.10,<3.15"
REM SHIFT only changes numbered arguments; %* still forwards the original list.
:scan_args
if "%~1"=="" goto launch
if "%~1"=="--uv" set "SETUP_USE_UV=1"
if "%~1"=="--python" if not "%~2"=="" set "SETUP_UV_PYTHON=%~2"
set "SETUP_ARG=%~1"
if "%SETUP_ARG:~0,9%"=="--python=" set "SETUP_UV_PYTHON=%SETUP_ARG:~9%"
shift /1
goto scan_args

:launch
if defined SETUP_USE_UV goto launch_uv
if defined PYTHON (
    "%PYTHON%" "%~dp0setup_venv.py" %*
) else (
    python "%~dp0setup_venv.py" %*
)
exit /b %ERRORLEVEL%

:launch_uv
where uv >nul 2>nul
if errorlevel 1 (
    echo error: --uv requires uv on PATH; install it from https://docs.astral.sh/uv/getting-started/installation/ 1>&2
    exit /b 2
)
REM Isolation lets --recreate remove .venv without removing the running launcher.
uv run --no-project --isolated --python "%SETUP_UV_PYTHON%" "%~dp0setup_venv.py" %*
exit /b %ERRORLEVEL%
