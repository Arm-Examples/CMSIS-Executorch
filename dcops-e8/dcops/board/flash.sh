#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Flash an M55-HP image through the guard script, check that Alif's boot area is intact, then
# walk the user through the power-cycle the MT9M114 camera needs after any J-Link reset.
#
#   dcops/board/flash.sh <axf> [--dry-run] [--no-wait] [--verify-only] [--backup FILE]
#
#   --dry-run      print the size check and the J-Link scripts, touch nothing
#   --no-wait      do not wait for the power-cycle confirmation
#   --verify-only  skip flashing, only compare the boot area with the backup
#   --backup FILE  full HP-partition backup (0x80200000, 3.5 MB); default: newest
#                  ~/projects/arm-executorch-hack/board-backups/devkit-e8-*.bin
#
# The boot area check reads 0x8057B000..0x80580000 (Alif ATOC at 0x8057BC00 plus the stubs) and
# compares it byte for byte with the same range of the backup. Images never reach it (the guard
# refuses anything ending past 0x80578000), so any difference means the boot table was damaged.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
JLINK="${JLINK:-/Applications/SEGGER/JLink/JLinkExe}"
DEVICE="AE822FA0E5597LS0_M55_HP"
GUARD="$REPO/scripts/alif/flash_e8_hp.sh"
BOOT_ADDR=0x8057B000
BOOT_LEN=0x5000
PART_BASE=0x80200000

AXF="" DRY=0 WAIT=1 VERIFY_ONLY=0 BACKUP=""
while (($#)); do
    case "$1" in
        --dry-run) DRY=1 ;;
        --no-wait) WAIT=0 ;;
        --verify-only) VERIFY_ONLY=1 ;;
        --backup) BACKUP="${2:?--backup needs a file}"; shift ;;
        -h|--help) sed -n '5,20p' "$0"; exit 0 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) AXF="$1" ;;
    esac
    shift
done
if [[ -z "$AXF" && $VERIFY_ONLY -eq 0 ]]; then
    echo "usage: $0 <image.axf> [--dry-run] [--no-wait] [--verify-only] [--backup FILE]" >&2
    exit 2
fi
if [[ -z "$BACKUP" ]]; then
    BACKUP=$(ls -t "$HOME"/projects/arm-executorch-hack/board-backups/devkit-e8-*.bin 2>/dev/null | head -1 || true)
fi

# Toolchain for readelf (the guard script needs it).
if ! command -v arm-none-eabi-readelf >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$HOME/projects/arm-executorch-hack/env.sh"
fi

run_jlink() {  # $1 = script text, $2 = label
    if ((DRY)); then
        echo "----- J-Link script ($2): $JLINK -device $DEVICE -if SWD -speed 4000 -autoconnect 1 -CommanderScript <script>"
        printf '%s\n' "$1"
        echo "----- end script"
        return 0
    fi
    local f rc=0
    f=$(mktemp "${TMPDIR:-/tmp}/jlink-board.XXXXXX")
    printf '%s\n' "$1" > "$f"
    "$JLINK" -device "$DEVICE" -if SWD -speed 4000 -autoconnect 1 -CommanderScript "$f" || rc=$?
    rm -f "$f"
    return $rc
}

# ---------------------------------------------------------------- 1. size check + flash
if ((! VERIFY_ONLY)); then
    [[ -f "$AXF" ]] || { echo "no such file: $AXF" >&2; exit 2; }
    AXF_ABS="$(cd "$(dirname "$AXF")" && pwd)/$(basename "$AXF")"
    echo "== build facts"
    { python3 "$HERE/axfinfo.py" "$AXF_ABS" 2>/dev/null | sed -n '3,6p'; } || true
    if ! python3 "$HERE/axfinfo.py" --check-limit "$AXF_ABS"; then
        echo "REFUSING: image ends past 0x80578000 and would overwrite the boot table." >&2
        exit 1
    fi
    if ((DRY)); then
        echo "== would run: $GUARD $AXF_ABS   (which flashes with:)"
        run_jlink "$(printf 'connect\nh\nloadfile %s\nr\ng\nexit\n' "$AXF_ABS")" "flash_e8_hp.sh"
    else
        echo "== flashing via $GUARD"
        "$GUARD" "$AXF_ABS"
    fi
fi

# ---------------------------------------------------------------- 2. boot-area check
echo "== boot area check: $BOOT_ADDR..+$BOOT_LEN vs backup ${BACKUP:-<none>}"
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/board-flash.XXXXXX")
trap 'rm -rf "$TMPD"' EXIT
READBACK="$TMPD/boot_readback.bin"
run_jlink "$(printf 'connect\nsavebin %s %s %s\nexit\n' "$READBACK" "$BOOT_ADDR" "$BOOT_LEN")" "boot-area read (core keeps running)" || true
BOOT_OK=1
if ((DRY)); then
    echo "(dry run: comparison skipped)"
elif [[ -z "$BACKUP" || ! -f "$BACKUP" ]]; then
    echo "WARNING: no backup found, cannot verify the boot area." >&2
    BOOT_OK=0
elif [[ ! -s "$READBACK" ]]; then
    echo "WARNING: boot-area read failed (no $READBACK). Is the HP core reachable?" >&2
    BOOT_OK=0
else
    if ! python3 "$HERE/axfinfo.py" --compare-boot "$READBACK" "$BACKUP"; then
        BOOT_OK=0
    fi
fi
if ((! BOOT_OK)) && ((! DRY)); then
    cat >&2 <<EOF

!!! The boot area does not match the backup (or could not be checked).
!!! If it really changed, the HP core will NOT boot after the power cycle.
!!! Recovery (dcops/board/README.md "Recovery"): write the backup back THROUGH THE HE CORE:
!!!   printf 'connect\nh\nloadbin $BACKUP 0x80200000\nexit\n' > /tmp/r.jlink
!!!   $JLINK -device AE822FA0E5597LS0_M55_HE -if SWD -speed 4000 -autoconnect 1 -CommanderScript /tmp/r.jlink
!!! then power-cycle, then re-flash the image with this script.
EOF
fi

# ---------------------------------------------------------------- 3. power cycle
cat <<'EOF'

==> POWER-CYCLE THE BOARD NOW: unplug the PRG USB cable, wait ~5 s, plug it back in.
    (The MT9M114 camera hangs after any J-Link flash/reset until the board loses power;
     without this the preview stays frozen/black and g_clsFrames does not advance.)
EOF
if ((WAIT)) && ((! DRY)) && [[ -t 0 ]]; then
    read -r -p "Press Enter once the board is back up (LCD shows the preview)... " _
    echo "OK. Next: python3 dcops/board/measure.py ${AXF:-<axf>}"
fi
((BOOT_OK)) || exit 3
