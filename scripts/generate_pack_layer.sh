#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# generate_pack_layer.sh — Generate a pack-based AI layer from model metadata
# ──────────────────────────────────────────────────────────────────────────────
#
# This script replaces the Docker-based stage1/stage2 build workflow with a
# simple operator-to-pack-component translation. Instead of compiling ExecuTorch
# from source, it references pre-built components from the PyTorch::ExecuTorch
# CMSIS-Pack.
#
# Prerequisites:
#   - Python 3.8+
#   - The PyTorch::ExecuTorch pack (either local path or registered in cpackget)
#   - A .pte model file OR an operators list file
#
# Usage:
#   # From a .pte model (auto-extracts operators):
#   ./scripts/generate_pack_layer.sh model/ethos_u_minimal_example.pte
#
#   # From an operators file:
#   ./scripts/generate_pack_layer.sh --ops-file model/operators_minimal.txt
#
#   # With explicit pack path:
#   ./scripts/generate_pack_layer.sh --pack-path ../packs/PyTorch.ExecuTorch.1.1.0-rc1-build.12 model/my_model.pte
#
#   # Full options:
#   ./scripts/generate_pack_layer.sh \
#     --pack-path ../packs/PyTorch.ExecuTorch.1.1.0-rc1-build.12 \
#     --ops-file model/operators_cnn.txt \
#     --model-header ./model/model_pte.h \
#     --cortex-m \
#     --output ai_layer/ai_layer.clayer.yml
#
# ──────────────────────────────────────────────────────────────────────────────

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT_DIR="$ROOT_DIR/scripts"

# ── Defaults ──────────────────────────────────────────────────────────────────
OUTPUT="${OUTPUT:-$ROOT_DIR/ai_layer/ai_layer.clayer.yml}"
PACK_PATH="${PACK_PATH:-}"
PACK_NAME="${PACK_NAME:-PyTorch::ExecuTorch}"
MODEL_PTE=""
OPS_FILE=""
MODEL_HEADER=""
REPORT_FILE=""
DESCRIPTION="AI layer using PyTorch ExecuTorch pack"
CORTEX_M=false
NO_ETHOS_U=false
CONVERT_MODEL=false
VERBOSE=false

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [model.pte]

Generate a pack-based ai_layer.clayer.yml from model operators.

Options:
  --ops-file FILE       Operators list file (one per line, # comments)
  --selected-yaml FILE  selected_operators.yaml from ExecuTorch build
  --pack-path PATH      Relative path to the ExecuTorch pack directory
  --pack-name NAME      Pack name (default: PyTorch::ExecuTorch)
  --output FILE         Output clayer.yml path (default: ai_layer/ai_layer.clayer.yml)
  --model-header PATH   Relative path to model_pte.h in the layer
  --report PATH         Relative path to REPORT.md in the layer
  --cortex-m            Include CMSIS-NN Cortex-M backend
  --no-ethos-u          Exclude Ethos-U NPU backend
  --convert-model       Convert .pte to model_pte.h header
  --description TEXT     Layer description
  --verbose, -v         Enable verbose output
  --help, -h            Show this help

Positional:
  model.pte             Path to .pte model file (extracts operators automatically)

Examples:
  # Simple: from a model file
  $(basename "$0") model/my_model.pte

  # From operators list with pack path
  $(basename "$0") --ops-file model/operators_cnn.txt --pack-path ../packs/PyTorch.ExecuTorch.1.1.0-rc1-build.12

  # Full pipeline: extract ops, convert model, generate layer
  $(basename "$0") --convert-model --model-header ./model/model_pte.h model/my_model.pte
EOF
}

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ops-file)       OPS_FILE="$2"; shift 2 ;;
        --selected-yaml)  SELECTED_YAML="$2"; shift 2 ;;
        --pack-path)      PACK_PATH="$2"; shift 2 ;;
        --pack-name)      PACK_NAME="$2"; shift 2 ;;
        --output)         OUTPUT="$2"; shift 2 ;;
        --model-header)   MODEL_HEADER="$2"; shift 2 ;;
        --report)         REPORT_FILE="$2"; shift 2 ;;
        --cortex-m)       CORTEX_M=true; shift ;;
        --no-ethos-u)     NO_ETHOS_U=true; shift ;;
        --convert-model)  CONVERT_MODEL=true; shift ;;
        --description)    DESCRIPTION="$2"; shift 2 ;;
        --verbose|-v)     VERBOSE=true; shift ;;
        --help|-h)        usage; exit 0 ;;
        -*)               echo "[ERROR] Unknown option: $1" >&2; usage; exit 1 ;;
        *)                MODEL_PTE="$1"; shift ;;
    esac
done

# ── Validate inputs ──────────────────────────────────────────────────────────
if [[ -z "$MODEL_PTE" && -z "$OPS_FILE" && -z "${SELECTED_YAML:-}" ]]; then
    echo "[ERROR] Must provide a .pte model, --ops-file, or --selected-yaml" >&2
    usage
    exit 1
fi

if [[ -n "$MODEL_PTE" && ! -f "$MODEL_PTE" ]]; then
    echo "[ERROR] Model file not found: $MODEL_PTE" >&2
    exit 1
fi

if [[ -n "$OPS_FILE" && ! -f "$OPS_FILE" ]]; then
    echo "[ERROR] Operators file not found: $OPS_FILE" >&2
    exit 1
fi

if [[ -n "${SELECTED_YAML:-}" && ! -f "${SELECTED_YAML}" ]]; then
    echo "[ERROR] Selected operators YAML not found: ${SELECTED_YAML}" >&2
    exit 1
fi

# ── Step 1: Convert model to header if requested ─────────────────────────────
if [[ "$CONVERT_MODEL" == "true" && -n "$MODEL_PTE" ]]; then
    echo "=== Step 1: Convert model to header ==="
    MODEL_DIR="$ROOT_DIR/ai_layer/model"
    mkdir -p "$MODEL_DIR"
    python3 "$SCRIPT_DIR/pte_to_header.py" \
        -p "$MODEL_PTE" \
        -d "$MODEL_DIR" \
        -o model_pte.h
    
    # Auto-set model header if not explicitly provided
    if [[ -z "$MODEL_HEADER" ]]; then
        MODEL_HEADER="./model/model_pte.h"
    fi
    echo ""
fi

# ── Step 2: Generate pack-based clayer.yml ────────────────────────────────────
echo "=== Step 2: Generate pack-based ai_layer.clayer.yml ==="

GENERATE_ARGS=(
    "--output" "$OUTPUT"
    "--pack-name" "$PACK_NAME"
    "--description" "$DESCRIPTION"
)

# Add pack path if provided
if [[ -n "$PACK_PATH" ]]; then
    GENERATE_ARGS+=("--pack-path" "$PACK_PATH")
fi

# Add model file for operator extraction
if [[ -n "$MODEL_PTE" ]]; then
    GENERATE_ARGS+=("--model" "$MODEL_PTE")
fi

# Add operators file
if [[ -n "$OPS_FILE" ]]; then
    GENERATE_ARGS+=("--ops-file" "$OPS_FILE")
fi

# Add selected operators YAML
if [[ -n "${SELECTED_YAML:-}" ]]; then
    GENERATE_ARGS+=("--selected-yaml" "${SELECTED_YAML}")
fi

# Add model files
if [[ -n "$MODEL_HEADER" ]]; then
    GENERATE_ARGS+=("--model-header" "$MODEL_HEADER")
fi
if [[ -n "$REPORT_FILE" ]]; then
    GENERATE_ARGS+=("--report" "$REPORT_FILE")
fi

# Backend options
if [[ "$CORTEX_M" == "true" ]]; then
    GENERATE_ARGS+=("--cortex-m")
fi
if [[ "$NO_ETHOS_U" == "true" ]]; then
    GENERATE_ARGS+=("--no-ethos-u")
fi

# Verbose
if [[ "$VERBOSE" == "true" ]]; then
    GENERATE_ARGS+=("--verbose")
fi

python3 "$SCRIPT_DIR/generate_pack_clayer.py" "${GENERATE_ARGS[@]}"

echo ""
echo "=== Done ==="
echo "Generated: $OUTPUT"
echo ""
echo "Next steps:"
echo "  1. Ensure the PyTorch::ExecuTorch pack is available"
echo "     (either via --pack-path or registered with cpackget)"
echo "  2. Build the project:"
echo "     cbuild executorch_project.csolution.yml --packs"
echo ""
