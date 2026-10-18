#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Flash an M55-HP image to the DevKit-E8 with J-Link, refusing images that would overwrite
# the boot area. The HP app partition is 0x80200000-0x80580000, and Alif's ATOC boot table
# and stubs (DEVICE, A32_STUB, HP_APP, HE_STUB) sit in its top ~17 KB (0x8057BC00 up).
# The linker script allows a larger image, which links fine but stops the HP core from
# booting. Recovery: write an MRAM backup through the HE core, see docs/blazeface-executorch.md.
#
# Usage: scripts/alif/flash_e8_hp.sh build-dir/bin/app.axf
# After flashing, power-cycle the board: the MT9M114 camera hangs after a J-Link reset.
set -euo pipefail
AXF="${1:?usage: $0 <image.axf>}"
LIMIT=0x80578000
JLINK="${JLINK:-/Applications/SEGGER/JLink/JLinkExe}"
READELF="${READELF:-arm-none-eabi-readelf}"

END=$("$READELF" -lW "$AXF" | awk '$1=="LOAD" { p=strtonum($4); f=strtonum($5); if (p>=0x80000000 && f>0 && p+f>m) m=p+f } END { printf "0x%x", m }' 2>/dev/null || true)
if [[ -z "$END" || "$END" == "0x0" ]]; then
    # macOS awk has no strtonum: fall back to python
    END=$(python3 - "$AXF" "$READELF" <<'PY'
import subprocess, sys
out = subprocess.run([sys.argv[2], "-lW", sys.argv[1]], capture_output=True, text=True).stdout
ends = [int(p[3], 16) + int(p[4], 16) for p in (l.split() for l in out.splitlines())
        if p and p[0] == "LOAD" and int(p[3], 16) >= 0x80000000 and int(p[4], 16) > 0]
print(hex(max(ends)))
PY
)
fi
echo "image ends at $END (limit $LIMIT)"
if (( END > LIMIT )); then
    echo "REFUSING: the image would reach the ATOC boot area at 0x8057BC00." >&2
    exit 1
fi
SCRIPT=$(mktemp)
printf 'connect\nh\nloadfile %s\nr\ng\nexit\n' "$(cd "$(dirname "$AXF")" && pwd)/$(basename "$AXF")" > "$SCRIPT"
"$JLINK" -device AE822FA0E5597LS0_M55_HP -if SWD -speed 4000 -autoconnect 1 -CommanderScript "$SCRIPT"
rm -f "$SCRIPT"
echo "Done. Power-cycle the board (unplug PRG USB for a few seconds) to restart the camera."
