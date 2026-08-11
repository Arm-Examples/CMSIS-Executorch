# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Windows counterpart of build.sh. Builds the example for the Corstone-320 /
# Ethos-U85 target.
#
# Inside Keil Studio the Arm Environment Manager has already activated the
# toolchain, so `cbuild` is on PATH and the vcpkg activation below is skipped.
# From a plain PowerShell prompt it runs `vcpkg activate` first.
$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

if (-not (Get-Command cbuild -ErrorAction SilentlyContinue)) {
    if (-not (Get-Command vcpkg -ErrorAction SilentlyContinue)) {
        throw "Neither cbuild nor vcpkg found on PATH. Open a Keil Studio terminal, or install vcpkg (see vcpkg-configuration.json)."
    }
    vcpkg activate
}

# --context must not be combined with --active (cbuild 2.14.1 rejects it);
# the csolution's target-set already selects the Debug context.
cbuild cmsis-executorch-simple.csolution.yml --active SSE-320-U85 --packs --update-rte
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$elf = Get-ChildItem -Path out -Filter *.elf -Recurse | Select-Object -First 1
if ($elf) {
    Write-Host ""
    Write-Host "ELF: $($elf.FullName)"
}
