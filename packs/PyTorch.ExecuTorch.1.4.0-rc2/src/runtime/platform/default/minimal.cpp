/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Fallback PAL implementations that do not depend on any assumptions about
 * capabililties of the system.
 *
 * Each et_pal_* fallback is marked ET_WEAK directly on its definition,
 * not just on the declaration in platform.h. The weak attribute is a
 * property of a specific symbol *definition* in the ELF symbol table —
 * it is not transitively inherited from a forward declaration. GCC
 * 13/14/15 and armclang 6.24 follow that rule: a definition without an
 * explicit weak attribute emits a STB_GLOBAL (strong) symbol, even if
 * the matching declaration was weak. (Older GCCs were laxer about this;
 * the current behaviour is the standards-aligned one.) Without the
 * direct ET_WEAK on each definition, a downstream consumer that
 * supplies its own et_pal_* override hits a multi-def link error
 * against this fallback.
 */

#define ET_INTERNAL_PLATFORM_WEAKNESS ET_WEAK
#include <executorch/runtime/platform/platform.h>

#include <executorch/runtime/platform/compiler.h>

ET_WEAK void et_pal_init(void) {}

ET_WEAK ET_NORETURN void et_pal_abort(void) {
  __builtin_trap();
}

ET_WEAK et_timestamp_t et_pal_current_ticks(void) {
  // This file cannot make any assumptions about the presence of functions that
  // return the current time, so all users should provide a strong override for
  // it. To help make it more obvious when this weak version is being used,
  // return a number that should be easier to search for than 0.
  return 11223344;
}

ET_WEAK et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  // Since we don't define a tick rate, return a conversion ratio of 1.
  return {1, 1};
}

ET_WEAK void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    ET_UNUSED et_pal_log_level_t level,
    ET_UNUSED const char* filename,
    ET_UNUSED const char* function,
    ET_UNUSED size_t line,
    ET_UNUSED const char* message,
    ET_UNUSED size_t length) {}

ET_WEAK void* et_pal_allocate(ET_UNUSED size_t size) {
  return nullptr;
}

ET_WEAK void et_pal_free(ET_UNUSED void* ptr) {}
