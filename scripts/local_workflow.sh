#!/bin/bash

# Pack-based build script: converts model and generates a pack-based AI layer.
#
# This replaces the source-compilation workflow (stage1/stage2) with a simple
# operator-to-pack-component translation using the PyTorch::ExecuTorch CMSIS-Pack.
#
# Steps:
#   1. Convert PyTorch model to .pte (requires ExecuTorch Python environment)
#   2. Convert .pte model to C header (model_pte.h)
#   3. Generate pack-based ai_layer.clayer.yml from model operators
#
# The generated layer references pre-built components from the pack instead of
# compiling ExecuTorch from source, eliminating steps 2-4 of the old workflow.

set -e

# Setup logging
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="/workspace2/ai_layer/logs"
MAIN_LOG="$LOG_DIR/build_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# Function to log with timestamp and tee to both console and log file
log_and_tee() {
    local step_name="$1"
    local log_file="$LOG_DIR/${step_name}_${TIMESTAMP}.log"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting: $step_name" | tee -a "$MAIN_LOG"
    
    # Execute the command and capture both stdout and stderr
    # Use PIPESTATUS to capture the exit code of the command before the pipe
    eval "$2" 2>&1 | tee -a "$log_file" "$MAIN_LOG"
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] Completed: $step_name" | tee -a "$MAIN_LOG"
        return 0
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Failed: $step_name (exit code: $exit_code)" | tee -a "$MAIN_LOG"
        return 1
    fi
}

# Log build start
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ==================================" | tee "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ExecuTorch Pack-Based AI Layer Build" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Timestamp: $TIMESTAMP" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Log Directory: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ==================================" | tee -a "$MAIN_LOG"

# Step 1: Convert and build model (.py → .pte)
echo -e "\033[1;36m=== Step 1: Convert and build model ===\033[0m"

# Install model-specific requirements if requirements.txt exists
if [ -f /workspace2/model/requirements.txt ]; then
    log_and_tee "model_requirements" "pip install --no-cache-dir -r /workspace2/model/requirements.txt"
fi

log_and_tee "model_conversion" "cd /workspace2/model && python3 aot_model.py"

# Step 2: Convert model to header file (.pte → model_pte.h)
echo -e "\033[1;31m=== Step 2: Convert model to header file ===\033[0m"
log_and_tee "model_to_header" "cd /workspace2 && python3 scripts/pte_to_header.py -p model/ethos_u_minimal_example.pte -d ai_layer/model -o model_pte.h"

# Step 3: Generate pack-based ai_layer.clayer.yml
# Extracts operators from the .pte model and translates them to
# PyTorch::ExecuTorch CMSIS-Pack component references.
echo -e "\033[1;34m=== Step 3: Generate pack-based AI layer ===\033[0m"
log_and_tee "generate_pack_layer" "cd /workspace2 && python3 scripts/generate_pack_clayer.py \
    --model model/ethos_u_minimal_example.pte \
    --model-header ./model/model_pte.h \
    --report ./model/REPORT.md \
    --verbose \
    -o ai_layer/ai_layer.clayer.yml"

# Step 4: Generate AI Layer Report (REPORT.md)
echo -e "\033[1;92m=== Step 4: Generate AI Layer Report ===\033[0m"
log_and_tee "generate_report" "cd /workspace2 && python3 scripts/generate_report.py \
    --conversion-log $LOG_DIR/model_conversion_${TIMESTAMP}.log \
    --pack-log $LOG_DIR/generate_pack_layer_${TIMESTAMP}.log \
    --verbose \
    -o ai_layer/model/REPORT.md"

# Step 5: Build Report Summary
echo -e "\033[1;92m=== Step 5: Build Report Summary ===\033[0m" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Generated ai_layer.clayer.yml:" | tee -a "$MAIN_LOG"
cat /workspace2/ai_layer/ai_layer.clayer.yml 2>/dev/null | tee -a "$MAIN_LOG" || echo "Layer file could not be displayed" | tee -a "$MAIN_LOG"


# Log build completion
echo "" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] ==================================" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] Pack-Based AI Layer Build Completed" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] ==================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "📋 Build logs available at: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "📋 Main build log: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "📋 Generated layer: ai_layer/ai_layer.clayer.yml" | tee -a "$MAIN_LOG"

# Create a build summary file
cat > "$LOG_DIR/build_summary_${TIMESTAMP}.txt" << EOF
ExecuTorch Pack-Based AI Layer Build Summary
=============================================
Build Timestamp: $TIMESTAMP
Build Status: SUCCESS
Build Mode: Pack-based (PyTorch::ExecuTorch CMSIS-Pack)

Log Files Generated:
- Main build log: $MAIN_LOG
- Model conversion: $LOG_DIR/model_conversion_${TIMESTAMP}.log
- Model to header: $LOG_DIR/model_to_header_${TIMESTAMP}.log
- Pack layer generation: $LOG_DIR/generate_pack_layer_${TIMESTAMP}.log

Outputs Generated:
- AI Layer definition: ai_layer/ai_layer.clayer.yml
- Model header: ai_layer/model/model_pte.h

Next steps:
  Build the project with: cbuild executorch_project.csolution.yml --packs
EOF

echo ""
echo "✅ Build completed successfully!"
echo "📊 Summary file: $LOG_DIR/build_summary_${TIMESTAMP}.txt"
