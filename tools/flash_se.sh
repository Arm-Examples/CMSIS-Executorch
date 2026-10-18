#!/bin/bash
# Flash the DevKit-E8 through the Secure Enclave (Alif SETOOLS over the SE UART),
# for when J-Link cannot connect. Writes the built firmware as the M55_HP app at
# 0x80200000 (where the SE boots it), with a fresh application TOC.
#
#   1. Move SW4 to SE.
#   2. tools/flash_se.sh            (after cbuild; COM port default COM4)
#   3. Move SW4 back to U4, unplug/replug the USB: the new firmware starts.
#
# Uses SETOOLS in C:\alif\setools\app-release-exec (build/config/bunny_mram_cfg.json
# = M55_HP_mram_cfg.json with the HP_APP binary replaced by fw_app.bin).
set -e
COM=${1:-COM4}
REPO=$(cd "$(dirname "$0")/.." && pwd)
SET=/c/alif/setools/app-release-exec
A=/c/Users/rdjma/.vcpkg/artifacts/2139c4c6
FROMELF=$A/compilers.arm.armclang/6.24.0/bin/fromelf.exe
AXF=$REPO/out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.axf

"$FROMELF" --bincombined --output "$(cygpath -w "$SET/build/images/fw_app.bin")" "$AXF"
ls -la "$SET/build/images/fw_app.bin"
cd "$SET"
sed 's/"M55_HP_mram_stub.bin"/"fw_app.bin"/' build/config/M55_HP_mram_cfg.json > build/config/fw_mram_cfg.json
./app-gen-toc.exe -f build/config/fw_mram_cfg.json | tr '\r' '\n' | grep -E "Binary File|Done|rror"
OUT=$(yes y | ./app-write-mram.exe -p -c "$COM" 2>&1 | tr '\r' '\n')
if echo "$OUT" | grep -q "Target did not respond"; then
    echo "FAILED: the Secure Enclave did not respond. Keep SW4 on SE, press RESET, run again."
    exit 1
fi
echo "$OUT" | grep -E "Done|seconds" | tail -4
echo "Flashed. Move SW4 to U4 and replug the USB."
