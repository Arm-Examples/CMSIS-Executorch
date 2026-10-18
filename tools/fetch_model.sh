#!/usr/bin/env bash
# Download the model bundle of a Reproduce_Baselines release into firmware/model/, check every
# file against the sha256 digest GitHub records for the release asset, and copy
# preprocess_params.json to params.json (the firmware's copy, which must equal it).
#
#   ./tools/fetch_model.sh                 # v0.5-distill (the v2-contract model the firmware is built for)
#   ./tools/fetch_model.sh v0.3-distill    # the previous model (same contract)
#   ./tools/fetch_model.sh --local <dir>   # copy a local bundle (e.g. Reproduce_Baselines/artifacts/v4)
#                                          # instead of downloading; no release digest to check, so the
#                                          # sha256 check is skipped and the local hashes are printed
#
# Then regenerate the parameter header if params.json changed:
#   python tools/gen_params_h.py && git diff --stat firmware/src/params_gen.h
#
# Needs the GitHub CLI (gh), logged in with read access to the org. Mirrors
# Emulator_Deploy/tools/fetch_model.sh.
set -euo pipefail

REPO="ArmExecuTorchEEG/Reproduce_Baselines"
ASSETS="tinyeegnet_int8.pte test_vectors.npz preprocess_params.json student.json tinyeegnet_float.pt"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/firmware/model"
mkdir -p "$DEST"

if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$1" | cut -d' ' -f1; }
else
  sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi

if [ "${1:-}" = "--local" ]; then
  SRC="${2:?usage: fetch_model.sh --local <bundle dir>}"
  fail=0
  for name in $ASSETS; do
    if [ -f "$SRC/$name" ]; then
      cp "$SRC/$name" "$DEST/$name"
      echo "COPIED  $name  $(sha256 "$DEST/$name")"
    else
      echo "MISSING  $SRC/$name"
      fail=1
    fi
  done
  echo "NOTE: local bundle, no release digest: sha256 check skipped"
  if [ "$fail" -eq 0 ]; then
    cp "$DEST/preprocess_params.json" "$DEST/params.json"
    echo "params.json = preprocess_params.json ($(sha256 "$DEST/params.json"))"
  fi
  exit "$fail"
fi

TAG="${1:-v0.5-distill}"

patterns=()
for name in $ASSETS; do
  patterns+=(--pattern "$name")
done
gh release download "$TAG" -R "$REPO" --dir "$DEST" --clobber "${patterns[@]}"

digests="$(gh release view "$TAG" -R "$REPO" --json assets \
  -q '.assets[] | "\(.name) \(.digest)"')"

fail=0
for name in $ASSETS; do
  want="$(printf '%s\n' "$digests" | awk -v n="$name" '$1 == n { sub(/^sha256:/, "", $2); print $2 }')"
  if [ ! -f "$DEST/$name" ]; then
    echo "MISSING  $name"
    fail=1
    continue
  fi
  got="$(sha256 "$DEST/$name")"
  if [ -z "$want" ]; then
    echo "MISSING DIGEST  $name"
    fail=1
  elif [ "$want" = "$got" ]; then
    echo "OK    $name  $got"
  else
    echo "BAD   $name  got $got, release says $want"
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  cp "$DEST/preprocess_params.json" "$DEST/params.json"
  echo "params.json = preprocess_params.json ($(sha256 "$DEST/params.json"))"
  # The committed C array (firmware/ai_layer/model_pte.c) must embed this .pte; regenerate with
  #   python firmware/create_ai_layer.py firmware/edge_eeg.cbuild-mlops.yml --pte firmware/model/tinyeegnet_int8.pte
fi
exit "$fail"
