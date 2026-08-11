@echo off
REM Copyright 2026 Arm Limited and/or its affiliates.
REM SPDX-License-Identifier: Apache-2.0
REM
REM Windows wrapper. All the logic lives in setup_venv.py.
REM "python" rather than "python3": python3.exe is not reliably present on
REM Windows, while python3 is the reliable name on Linux/macOS -- which is why
REM this wrapper and setup_venv.sh differ.
setlocal
if defined PYTHON (
    "%PYTHON%" "%~dp0setup_venv.py" %*
) else (
    python "%~dp0setup_venv.py" %*
)
exit /b %ERRORLEVEL%
