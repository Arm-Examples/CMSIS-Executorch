/*---------------------------------------------------------------------------
 * Copyright (c) 2026 Sentinel contributors.
 * SPDX-License-Identifier: Apache-2.0
 *---------------------------------------------------------------------------*/
#ifndef SENTINEL_PROFILE_STORE_H
#define SENTINEL_PROFILE_STORE_H

#define SENTINEL_PROFILE_COUNT 2u
#define SENTINEL_PROFILE_FEATURES 4u

typedef struct {
    float mean[SENTINEL_PROFILE_FEATURES];
    float std[SENTINEL_PROFILE_FEATURES];
} sentinel_baseline_t;

/* 1 = saved baseline, 0 = empty, -1 = storage error. */
int sentinel_profile_load(unsigned id, sentinel_baseline_t *out);
/* 0 = written and read back, -1 = invalid data or storage error. */
int sentinel_profile_save(unsigned id, const sentinel_baseline_t *value);

#endif
