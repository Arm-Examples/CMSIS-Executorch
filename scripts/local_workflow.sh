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
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ExecuTorch AI Layer Build Started" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Timestamp: $TIMESTAMP" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Log Directory: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ==================================" | tee -a "$MAIN_LOG"

# Step 1: Convert and build model
echo -e "\033[1;36m=== Step 1: Convert and build model ===\033[0m"

# Install model-specific requirements if requirements.txt exists
if [ -f /workspace2/model/requirements.txt ]; then
    log_and_tee "model_requirements" "pip install --no-cache-dir -r /workspace2/model/requirements.txt"
fi

log_and_tee "model_conversion" "cd /workspace2/model && python3 aot_model.py"

# Step 2: Build ExecuTorch Core Libraries  
echo -e "\033[1;33m=== Step 2: Build ExecuTorch Core Libraries ===\033[0m"
log_and_tee "stage1_build" "/workspace2/scripts/build_stage1.sh /workspace/executorch /workspace2/out/stage1 /workspace2/model/arm-none-eabi-gcc.cmake"

# Step 2a: Generate source layer from stage1 compile_commands.json
# Note: posix.cpp is excluded because it uses std::chrono::steady_clock which is not available in ARM Clang bare-metal libc++
# Two strip paths are needed: one for ExecuTorch source, one for generated files in build dir
# Registration files (RegisterCodegenUnboxedKernelsEverything.cpp) are NOT included here - they come from stage2 (selective build)
# Flags -fPIC and -s are removed for AC6 compatibility (causes compiler crash in AC6 6.24.0)
echo -e "\033[1;35m=== Step 2a: Generate source layer ===\033[0m"
log_and_tee "generate_source_layer" "cd /workspace2/out/stage1 && python3 /workspace2/scripts/ccdb2clayer.py -c compile_commands.json -v --copy-sources -A /workspace/executorch -A /workspace2/out/stage1 -L . -o /workspace2/ai_layer/engine/stage1_source.clayer.yml -n 'ExecuTorch Stage1 Sources' -g runtime -g extension -g schema -g 'kernels/quantized' -g backends --exclude '*/posix.cpp' --remove-flags=-fPIC --remove-flags=-s"

# Step 2b: Copy NativeFunctions.h to stage1 source directories
# Registration files use #include "NativeFunctions.h" expecting the header in the same directory
echo -e "\033[1;35m=== Step 2b: Copy NativeFunctions.h to stage1 directories ===\033[0m"
log_and_tee "copy_native_functions_stage1" "
# Copy NativeFunctions.h for backends/cortex_m/cortex_m_ops_lib
ops_dir=/workspace2/out/stage1/backends/cortex_m/cortex_m_ops_lib
dst_dir=\"/workspace2/ai_layer/engine/backends/cortex_m/cortex_m_ops_lib\"
if [ -f \"\${ops_dir}/NativeFunctions.h\" ]; then
    if [ -d \"\${dst_dir}\" ]; then
        cp \"\${ops_dir}/NativeFunctions.h\" \"\${dst_dir}/\"
        cp \"\${ops_dir}/Functions.h\" \"\${dst_dir}\" 2>/dev/null || true
        cp \"\${ops_dir}/CustomOpsNativeFunctions.h\" \"\${dst_dir}\" 2>/dev/null || true
        echo \"[Copy] NativeFunctions.h -> \${dst_dir}\"
    fi
fi
# Copy NativeFunctions.h for kernels/quantized/quantized_ops_lib
ops_dir=/workspace2/out/stage1/kernels/quantized/quantized_ops_lib
dst_dir=\"/workspace2/ai_layer/engine/kernels/quantized/quantized_ops_lib\"
if [ -f \"\${ops_dir}/NativeFunctions.h\" ]; then
    if [ -d \"\${dst_dir}\" ]; then
        cp \"\${ops_dir}/NativeFunctions.h\" \"\${dst_dir}/\"
        cp \"\${ops_dir}/Functions.h\" \"\${dst_dir}\" 2>/dev/null || true
        cp \"\${ops_dir}/CustomOpsNativeFunctions.h\" \"\${dst_dir}\" 2>/dev/null || true
        echo \"[Copy] NativeFunctions.h -> \${dst_dir}\"
    fi
fi
"

# Step 3: Build ExecuTorch Selective Kernel Libraries
# Using model PTE for automatic operator analysis (instead of manual operators list)
echo -e "\033[1;34m=== Step 3: Build ExecuTorch Selective Kernel Libraries ===\033[0m"
log_and_tee "stage2_build" "/workspace2/scripts/build_stage2_selective.sh /workspace/executorch /workspace2/model/ethos_u_minimal_example.pte /workspace2/out/stage2 /workspace2/model/arm-none-eabi-gcc.cmake"

# Step 3a: Generate source layer from stage2 compile_commands.json
# Note: Files using std::random_device are excluded (not available in ARM Clang bare-metal libc++)
# Note: Quantized kernels are excluded because they're already in stage1_source.clayer.yml
# Two strip paths are needed: one for ExecuTorch source, one for generated files in build dir
# Flags -fPIC and -s are removed for AC6 compatibility (causes compiler crash in AC6 6.24.0)
echo -e "\033[1;35m=== Step 3a: Generate stage2 source layer ===\033[0m"
log_and_tee "generate_source_layer_stage2" "cd /workspace2/out/stage2 && python3 /workspace2/scripts/ccdb2clayer.py -c compile_commands.json -v --copy-sources -A /workspace/executorch -A /workspace2/out/stage2 -L . -o /workspace2/ai_layer/engine/stage2_source.clayer.yml -n 'ExecuTorch Stage2 Selective Ops' -g kernels --exclude '*/op_rand.cpp' --exclude '*/op_randn.cpp' --exclude '*/op_native_dropout.cpp' --exclude 'kernels/quantized/*' --remove-flags=-fPIC --remove-flags=-s"

# Step 3a.1: Fix stage2 layer to use selective kernels (executorch_selected_kernels instead of portable_ops_lib)
echo -e "\033[1;35m=== Step 3a.1: Fix stage2 layer for selective kernels ===\033[0m"
log_and_tee "fix_stage2_layer" "sed -i 's|kernels/portable/portable_ops_lib/RegisterCodegenUnboxedKernelsEverything.cpp|executorch_selected_kernels/RegisterCodegenUnboxedKernelsEverything.cpp|g' /workspace2/ai_layer/engine/stage2_source.clayer.yml && echo '[Fix] Updated stage2_source.clayer.yml to use executorch_selected_kernels'"

# Step 3b: Copy NativeFunctions.h to registration source directories (stage2 only)
# Registration files use #include "NativeFunctions.h" expecting the header in the same directory
echo -e "\033[1;35m=== Step 3b: Copy NativeFunctions.h to registration directories ===\033[0m"
log_and_tee "copy_native_functions" "
# First try executorch_selected_kernels (selective build target)
select_dir=/workspace2/out/stage2/executorch_selected_kernels
if [ -f \"\${select_dir}/NativeFunctions.h\" ]; then
    dst_dir=\"/workspace2/ai_layer/engine/executorch_selected_kernels\"
    mkdir -p \"\${dst_dir}\"
    cp \"\${select_dir}/NativeFunctions.h\" \"\${dst_dir}/\"
    if [ -f \"\${select_dir}/RegisterCodegenUnboxedKernelsEverything.cpp\" ]; then
        cp \"\${select_dir}/RegisterCodegenUnboxedKernelsEverything.cpp\" \"\${dst_dir}/\"
    fi
    echo \"[Copy] NativeFunctions.h + registration -> \${dst_dir}\"
fi
# Fallback to portable_ops_lib if executorch_selected_kernels doesn't exist
ops_dir=/workspace2/out/stage2/kernels/portable/portable_ops_lib
if [ -f \"\${ops_dir}/NativeFunctions.h\" ]; then
    dst_dir=\"/workspace2/ai_layer/engine/kernels/portable/portable_ops_lib\"
    if [ -d \"\${dst_dir}\" ]; then
        cp \"\${ops_dir}/NativeFunctions.h\" \"\${dst_dir}/\"
        echo \"[Copy] NativeFunctions.h -> \${dst_dir}\"
    fi
fi
"

# Step 3c: Copy Clang platform stubs to ai_layer for self-contained layer
# These stubs are needed for ARM Clang builds (std::chrono and std::random_device not available)
echo -e "\033[1;35m=== Step 3c: Copy Clang platform stubs to ai_layer ===\033[0m"
log_and_tee "copy_clang_stubs" "
mkdir -p /workspace2/ai_layer/stubs
cp /workspace2/src/posix_stub.cpp /workspace2/ai_layer/stubs/
cp /workspace2/src/random_ops_stubs.cpp /workspace2/ai_layer/stubs/
echo '[Copy] Clang stubs -> ai_layer/stubs/'
"

# Step 3d: Fix empty #include "" in RegisterCodegenUnboxedKernelsEverything.cpp
# ExecuTorch generates this file with #include "" which needs to be NativeFunctions.h
echo -e "\033[1;35m=== Step 3d: Fix empty include in registration files ===\033[0m"
log_and_tee "fix_empty_include" "
find /workspace2/ai_layer/engine -name 'RegisterCodegenUnboxedKernelsEverything.cpp' | while read f; do
    if grep -q '#include \"\"' \"\$f\"; then
        sed -i 's|#include \"\"|#include \"NativeFunctions.h\"|g' \"\$f\"
        echo \"[Fix] #include \\\"\\\" -> #include \\\"NativeFunctions.h\\\" in \$(basename \$f)\"
    fi
done
"

# Step 3e: Fix assert(False) -> assert(false) in synced source files
# ExecuTorch uses Python-style assert(False) which doesn't work in C++
echo -e "\033[1;35m=== Step 3e: Fix assert(False) bugs in synced sources ===\033[0m"
log_and_tee "fix_assert_false" "
find /workspace2/ai_layer/engine -name '*.cpp' -o -name '*.h' | xargs grep -l 'assert(False)' 2>/dev/null | while read f; do
    sed -i 's/assert(False)/assert(false)/g' \"\$f\"
    echo \"[Fix] assert(False) -> assert(false) in \$(basename \$f)\"
done
"

# Step 4: Collect artifacts
echo -e "\033[1;32m=== Step 4: Collect artifacts ===\033[0m"
# Pass the ExecuTorch source path so package_sdk can copy extended_header.h.
log_and_tee "package_artifacts" "cd /workspace2 && EXECUTORCH_SRC=/workspace/executorch ./scripts/package_sdk.sh \"/workspace2/out/stage1/assets\" \"/workspace2/out/stage2/assets\" \"/workspace2/model/ethos_u_minimal_example.pte\" \"/workspace2/ai_layer/engine\""

# Step 4a: Fix compiler.h for AC6 (ARM Compiler 6) compatibility
# Must run AFTER package_artifacts which copies fresh headers from ExecuTorch
# AC6 uses armclang which doesn't have <sys/types.h> in bare-metal mode
echo -e "\033[1;35m=== Step 4a: Fix compiler.h for AC6 compatibility ===\033[0m"
log_and_tee "fix_ac6_compiler_h" "
COMPILER_H=/workspace2/ai_layer/engine/include/executorch/runtime/platform/compiler.h
if [ -f \"\$COMPILER_H\" ]; then
    # Replace the sys/types.h include block to also handle AC6 (__ARMCC_VERSION)
    # Original: #ifndef _MSC_VER / #include <sys/types.h> / #else / #include <stddef.h>
    # New: Also check for __ARMCC_VERSION (AC6)
    if grep -q '#include <sys/types.h>' \"\$COMPILER_H\"; then
        sed -i 's|#ifndef _MSC_VER|#if !defined(_MSC_VER) \\&\\& !defined(__ARMCC_VERSION)|g' \"\$COMPILER_H\"
        echo \"[Fix] compiler.h: Added AC6 (__ARMCC_VERSION) detection for sys/types.h workaround\"
    else
        echo \"[Skip] compiler.h: sys/types.h include not found (already fixed or different version)\"
    fi
else
    echo \"[Skip] compiler.h not found\"
fi
"

# Step 5: Convert model to header file
echo -e "\033[1;31m=== Step 5: Convert model to header file ===\033[0m"
log_and_tee "model_to_header" "cd /workspace2 && python3 scripts/pte_to_header.py -p model/ethos_u_minimal_example.pte -d ai_layer/model -o model_pte.h"

# Step 6: Generate comprehensive AI layer report
echo -e "\033[1;93m=== Step 6: Generate AI layer report ===\033[0m"
log_and_tee "generate_report" "cd /workspace2 && python3 scripts/generate_ai_layer_report.py"

# Step 7: Build Report Summary
echo -e "\033[1;92m=== Step 7: Build Report Summary ===\033[0m" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "📊 Build Summary:" | tee -a "$MAIN_LOG"
head -n 20 /workspace2/ai_layer/REPORT.md 2>/dev/null | tee -a "$MAIN_LOG" || echo "Report could not be displayed" | tee -a "$MAIN_LOG"/9


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
- Generate source layer: $LOG_DIR/generate_source_layer_${TIMESTAMP}.log
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
