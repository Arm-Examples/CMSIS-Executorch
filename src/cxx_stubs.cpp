/*
 * ARM C++ runtime stubs for bare-metal targets
 * 
 * Provides minimal stub implementations for C++ runtime functions
 * required by ARM Clang that are not needed in bare-metal contexts.
 * 
 * Note: std::chrono::steady_clock and std::random_device are not available
 * in ARM Clang bare-metal libc++ (_LIBCPP_HAS_MONOTONIC_CLOCK=0, 
 * _LIBCPP_HAS_RANDOM_DEVICE=0). Files using these features must be excluded
 * or use alternative implementations.
 */

#include <cstdint>
#include <cstring>

extern "C" {

/**
 * Stub for C++ atexit handler registration
 * In bare-metal systems without dynamic library loading,
 * we don't need to track exit handlers
 */
int __aeabi_atexit(void *object, void (*destructor)(void *), void *dso_handle) {
    (void)object;
    (void)destructor;
    (void)dso_handle;
    return 0;  // Success
}

/**
 * GCC libstdc++ std::random_device stubs 
 * These are needed when linking against GCC-built libraries that use random_device.
 * The mangled names are for GCC's libstdc++ ABI.
 */

// Mangled: _ZNSt13random_device9_M_getvalEv
unsigned int _ZNSt13random_device9_M_getvalEv() {
    static unsigned int seed = 0x12345678;
    // Simple LCG for deterministic "random" values
    seed = (seed * 1103515245U + 12345U) & 0x7FFFFFFF;
    return seed;
}

// Mangled: _ZNSt13random_device7_M_finiEv
void _ZNSt13random_device7_M_finiEv() {
    // No cleanup needed in bare-metal
}

// Mangled: _ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
void _ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(const void* token) {
    (void)token;  // Ignore token in bare-metal
}

/**
 * GCC libstdc++ std::__cxx11::basic_string stubs
 * These are needed when linking against GCC-built libraries.
 */

// Mangled: _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_S_copy_charsEPcPKcS7_
void _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_S_copy_charsEPcPKcS7_(
    char* dst, const char* src_begin, const char* src_end)
{
    size_t len = src_end - src_begin;
    if (len > 0) {
        memcpy(dst, src_begin, len);
    }
}

// Mangled: _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
void _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv() {
    // No-op for bare-metal - no dynamic memory to free
}

} // extern "C"