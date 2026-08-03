# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Host-OS dispatcher for the model-conversion build step. Run as:
#   cmake -P scripts/run_export.cmake
#
# Why CMake rather than a shell script: the cproject `executes:` node has no
# host-OS conditional (the csolution schema allows only for-context /
# not-for-context, which select build and target types), so a single `run:`
# string has to work on Linux, macOS and Windows. `bash` is not available by
# default on Windows, but ${CMAKE_COMMAND} always is -- cbuild drives CMake.
#
# All this does is pick the venv interpreter for the host and hand over to
# scripts/run_export.py.

cmake_minimum_required(VERSION 3.20)

set(SOLUTION_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(VENV_DIR "${SOLUTION_DIR}/.venv")

# CMAKE_HOST_WIN32 rather than WIN32: in script mode (-P) there is no project()
# call, so the target-platform variables are not set.
if(CMAKE_HOST_WIN32)
    set(VENV_PYTHON "${VENV_DIR}/Scripts/python.exe")
    set(SETUP_HINT "setup_venv.bat")
else()
    set(VENV_PYTHON "${VENV_DIR}/bin/python")
    set(SETUP_HINT "./setup_venv.sh")
endif()

if(NOT EXISTS "${VENV_PYTHON}")
    message(FATAL_ERROR
        "Model-export venv not found at ${VENV_DIR}\n"
        "Create it first:\n"
        "  ${SETUP_HINT}\n")
endif()

execute_process(
    COMMAND "${VENV_PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/run_export.py"
    RESULT_VARIABLE result
)

if(NOT result EQUAL 0)
    # run_export.py has already explained itself on stderr (either an export
    # failure or the "operator set changed, re-run the build" notice).
    message(FATAL_ERROR "convert-model step failed (exit ${result})")
endif()
