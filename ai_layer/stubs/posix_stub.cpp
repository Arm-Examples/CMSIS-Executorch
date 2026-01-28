/**
 * @file posix_stub.cpp
 * @brief Stub implementation of posix.cpp functions for ARM Clang
 * 
 * This file provides weak symbol implementations that will be overridden
 * by the strong symbols in arm_executor_runner.cpp.
 * 
 * This is needed because the original posix.cpp uses std::chrono::steady_clock
 * which is not available in ARM Clang's bare-metal libc++.
 */

#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <cstdint>
#include <cstdio>

namespace torch {
namespace executor {

// Weak symbol - will be overridden by arm_executor_runner.cpp
__attribute__((weak)) void et_pal_init(void) {
    // Empty stub - actual implementation in arm_executor_runner.cpp
}

// Weak symbol - will be overridden by arm_executor_runner.cpp  
__attribute__((weak)) et_timestamp_t et_pal_current_ticks(void) {
    // Return dummy value - actual implementation in arm_executor_runner.cpp
    return 0;
}

// Weak symbol - will be overridden by arm_executor_runner.cpp
__attribute__((weak)) et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
    return {1, 1};
}

// Weak symbol - will be overridden by arm_executor_runner.cpp
__attribute__((weak)) void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t length) {
    // Empty stub - actual implementation in arm_executor_runner.cpp
    (void)timestamp;
    (void)level;
    (void)filename;
    (void)function;
    (void)line;
    (void)message;
    (void)length;
}

} // namespace executor
} // namespace torch
