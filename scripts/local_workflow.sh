#!/bin/bash

# Simple build script that executes the same commands as GitHub Actions workflow

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
    if eval "$2" 2>&1 | tee -a "$log_file" "$MAIN_LOG"; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] Completed: $step_name" | tee -a "$MAIN_LOG"
        return 0
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Failed: $step_name" | tee -a "$MAIN_LOG"
        return 1
    fi
}

# Log build start
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ==================================" | tee "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ExecuTorch AI Layer Build Started" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Timestamp: $TIMESTAMP" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Log Directory: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ==================================" | tee -a "$MAIN_LOG"

# Step 1: Convert and build model
echo "=== Step 1: Convert and build model ==="
log_and_tee "model_conversion" "cd /workspace2/model && python3 aot_model.py"

# Step 2: Build ExecuTorch Core Libraries  
echo "=== Step 2: Build ExecuTorch Core Libraries ==="
log_and_tee "stage1_build" "/workspace2/scripts/build_stage1.sh /workspace/executorch /workspace2/out/stage1 /workspace/executorch/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake"

# Step 3: Build ExecuTorch Selective Kernel Libraries
echo "=== Step 3: Build ExecuTorch Selective Kernel Libraries ==="
log_and_tee "stage2_build" "/workspace2/scripts/build_stage2_selective.sh /workspace/executorch \"\" /workspace2/out/stage2 /workspace/executorch/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake /workspace2/model/operators_minimal.txt"

# Step 4: Collect artifacts
echo "=== Step 4: Collect artifacts ==="
log_and_tee "package_artifacts" "cd /workspace2 && ./scripts/package_sdk.sh \"/workspace2/out/stage1/assets\" \"/workspace2/out/stage2/assets\" \"/workspace2/model/ethos_u_minimal_example.pte\" \"/workspace2/ai_layer/engine\""

# Step 5: Convert model to header file
echo "=== Step 5: Convert model to header file ==="
log_and_tee "model_to_header" "cd /workspace2 && python3 scripts/pte_to_header.py -p model/ethos_u_minimal_example.pte -d ai_layer/model -o model_pte.h"

# Step 6: Generate comprehensive AI layer report
echo "=== Step 6: Generate AI layer report ==="
log_and_tee "generate_report" "cd /workspace2 && python3 scripts/generate_ai_layer_report.py"

# Step 7: Build Report Summary
echo "=== Step 7: Build Report Summary ===" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "📊 Build Summary:" | tee -a "$MAIN_LOG"
head -n 20 /workspace2/ai_layer/REPORT.md 2>/dev/null | tee -a "$MAIN_LOG" || echo "Report could not be displayed" | tee -a "$MAIN_LOG"

# Log build completion
echo "" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] ==================================" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] ExecuTorch AI Layer Build Completed" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] ==================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "📋 Build logs available at: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "📋 Main build log: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "📋 Full report available at: ai_layer/REPORT.md" | tee -a "$MAIN_LOG"

# Create a build summary file
cat > "$LOG_DIR/build_summary_${TIMESTAMP}.txt" << EOF
ExecuTorch AI Layer Build Summary
=================================
Build Timestamp: $TIMESTAMP
Build Status: SUCCESS
Build Duration: Started at $(head -n 1 "$MAIN_LOG" | cut -d' ' -f1-2)

Log Files Generated:
- Main build log: $MAIN_LOG
- Model conversion: $LOG_DIR/model_conversion_${TIMESTAMP}.log
- Stage1 build: $LOG_DIR/stage1_build_${TIMESTAMP}.log  
- Stage2 build: $LOG_DIR/stage2_build_${TIMESTAMP}.log
- Package artifacts: $LOG_DIR/package_artifacts_${TIMESTAMP}.log
- Model to header: $LOG_DIR/model_to_header_${TIMESTAMP}.log
- Generate report: $LOG_DIR/generate_report_${TIMESTAMP}.log

Outputs Generated:
- AI Layer libraries: ai_layer/engine/lib/
- AI Layer headers: ai_layer/engine/include/
- Model header: ai_layer/model/model_pte.h
- Build report: ai_layer/REPORT.md

For detailed information, see the individual log files above.
EOF

echo ""
echo "✅ Build completed successfully!"
echo "📊 Summary file: $LOG_DIR/build_summary_${TIMESTAMP}.txt"
