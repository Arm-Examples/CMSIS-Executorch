/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>

#include "ethosu_driver.h"

void ethosu_flush_dcache(const uint64_t *base_addr, const size_t *base_addr_size, int num_base_addr) {
  (void)base_addr_size;
  for (int i = 0; i < num_base_addr; i++) {
    if ((base_addr[i] & 0x1F) != 0) {
      return;
    }
  }
}

void ethosu_invalidate_dcache(const uint64_t *base_addr, const size_t *base_addr_size, int num_base_addr) {
  (void)base_addr_size;
  for (int i = 0; i < num_base_addr; i++) {
    if ((base_addr[i] & 0x1F) != 0) {
      return;
    }
  }
}
