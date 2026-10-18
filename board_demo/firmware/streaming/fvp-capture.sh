#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${HERE}/fvp-base.sh" "$@" 2>&1 | tee "${HERE}/../fvp-stream.log"
exit "${PIPESTATUS[0]}"
