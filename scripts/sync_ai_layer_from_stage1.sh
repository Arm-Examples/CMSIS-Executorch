#!/usr/bin/env bash
set -euo pipefail

# Sync headers from the ExecuTorch stage1 build into ai_layer for CMSIS builds.
# - FlatBuffers headers are taken from the stage1 build output (flatc_ep).
# - extended_header.h is copied from the ExecuTorch source tree (supply EXECUTORCH_SRC).

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAGE1_ROOT=${STAGE1_ROOT:-"$ROOT_DIR/out/stage1"}
EXECUTORCH_SRC=${EXECUTORCH_SRC:-""}

SRC_FLATBUFFERS="$STAGE1_ROOT/third-party/flatc_ep/include/flatbuffers"
DEST_FLATBUFFERS="$ROOT_DIR/ai_layer/engine/include/third-party/flatbuffers/include"

if [[ -d "$SRC_FLATBUFFERS" ]]; then
  mkdir -p "$DEST_FLATBUFFERS"
  rsync -a "$SRC_FLATBUFFERS/" "$DEST_FLATBUFFERS/"
  echo "Synced FlatBuffers headers from $SRC_FLATBUFFERS to $DEST_FLATBUFFERS"
else
  echo "ERROR: FlatBuffers headers not found at $SRC_FLATBUFFERS (set STAGE1_ROOT)" >&2
  exit 1
fi

if [[ -n "$EXECUTORCH_SRC" ]]; then
  SRC_EXT_HDR="$EXECUTORCH_SRC/schema/extended_header.h"
  DEST_EXT_HDR_DIR="$ROOT_DIR/ai_layer/engine/include/executorch/schema"
  if [[ -f "$SRC_EXT_HDR" ]]; then
    mkdir -p "$DEST_EXT_HDR_DIR"
    cp "$SRC_EXT_HDR" "$DEST_EXT_HDR_DIR/"
    echo "Copied extended_header.h from $SRC_EXT_HDR to $DEST_EXT_HDR_DIR"
  else
    echo "WARN: extended_header.h not found at $SRC_EXT_HDR" >&2
  fi
else
  echo "NOTE: EXECUTORCH_SRC not set; skipping extended_header.h copy" >&2
fi

echo "Sync complete. Rebuild the CMSIS project to pick up headers."
