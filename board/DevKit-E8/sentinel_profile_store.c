/*---------------------------------------------------------------------------
 * Copyright (c) 2026 Sentinel contributors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Two alternating records per profile in the FINAL 256 bytes of user MRAM.
 * Each 64-byte record has a header, eight float bit patterns, three reserved
 * words, and a checksum. Its header is written as IN_PROGRESS first and as
 * VALID last, so an interrupted save cannot replace the last good record.
 *---------------------------------------------------------------------------*/
#include "sentinel_profile_store.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define STORE_BYTES 256u
#define RECORD_BYTES 64u
#define SLOT_BYTES 16u
#define MAGIC_VALID 0x31504e53u /* 'SNP1' on little-endian E8 */
#define MAGIC_BUSY  0x3f504e53u
#define STORE_VERSION 1u

#ifdef SENTINEL_PROFILE_HOST_TEST
/* The host test includes this source so it runs the real record logic. */
static uint8_t test_mram[STORE_BYTES];
static int test_fail_after = -1;

static int store_read(unsigned offset, void *out, unsigned bytes)
{
    if (offset > STORE_BYTES || bytes > STORE_BYTES - offset) return -1;
    memcpy(out, test_mram + offset, bytes);
    return 0;
}

static int store_write16(unsigned offset, const void *src)
{
    if (offset > STORE_BYTES - SLOT_BYTES || (offset & (SLOT_BYTES - 1u))) return -1;
    if (test_fail_after == 0) return -1;
    if (test_fail_after > 0) --test_fail_after;
    memcpy(test_mram + offset, src, 8u);
    if (test_fail_after == 0) return -1;
    if (test_fail_after > 0) --test_fail_after;
    memcpy(test_mram + offset + 8u, (const uint8_t *) src + 8u, 8u);
    return 0;
}
#else
#include "RTE_Components.h"
#include CMSIS_device_header
#include "app_mem_regions.h"

_Static_assert(APP_MRAM_USER_SIZE >= STORE_BYTES, "user MRAM too small");
_Static_assert((APP_MRAM_USER_BASE + APP_MRAM_USER_SIZE) % SLOT_BYTES == 0,
               "profile MRAM not 16-byte aligned");

#define STORE_BASE (APP_MRAM_USER_BASE + APP_MRAM_USER_SIZE - STORE_BYTES)

static int store_read(unsigned offset, void *out, unsigned bytes)
{
    if (offset > STORE_BYTES || bytes > STORE_BYTES - offset) return -1;
    memcpy(out, (const void *) (uintptr_t) (STORE_BASE + offset), bytes);
    return 0;
}

/* Rev A cannot execute out of MRAM while programming MRAM. This leaf
 * routine is in ITCM and has no calls, literal-pool reads or unwind table.
 * Source is aligned RAM; destination is 16-byte-aligned Device MRAM. */
extern void sentinel_mram_write16_itcm(volatile uint64_t *dst, const uint64_t *src);
__asm(".section .sentinel_itcm,\"ax\"\n"
      ".align 2\n"
      ".global sentinel_mram_write16_itcm\n"
      ".type sentinel_mram_write16_itcm,%function\n"
      "sentinel_mram_write16_itcm:\n"
      "ldrd r2, r3, [r1]\n"
      "strd r2, r3, [r0]\n"
      "ldrd r2, r3, [r1, #8]\n"
      "strd r2, r3, [r0, #8]\n"
      "dsb sy\n"
      "bx lr\n"
      ".size sentinel_mram_write16_itcm, .-sentinel_mram_write16_itcm\n"
      ".previous\n");

static int store_write16(unsigned offset, const void *src)
{
    if (offset > STORE_BYTES - SLOT_BYTES || (offset & (SLOT_BYTES - 1u))) return -1;
    uint64_t aligned[2];
    memcpy(aligned, src, SLOT_BYTES);
    const uint32_t mask = __get_PRIMASK();
    __disable_irq();
    sentinel_mram_write16_itcm((volatile uint64_t *) (uintptr_t) (STORE_BASE + offset), aligned);
    __set_PRIMASK(mask);
    return 0;
}
#endif

static unsigned slot_offset(unsigned profile, unsigned slot)
{
    return (profile * 2u + slot) * RECORD_BYTES;
}

static uint32_t checksum(const uint32_t *w)
{
    uint32_t h = 2166136261u;
    for (unsigned i = 0; i < 15u; ++i) {
        h = (h ^ w[i]) * 16777619u;
    }
    return h;
}

static int baseline_valid(const sentinel_baseline_t *b)
{
    if (!b) return 0;
    for (unsigned i = 0; i < SENTINEL_PROFILE_FEATURES; ++i) {
        if (!isfinite(b->mean[i]) || !isfinite(b->std[i]) || b->std[i] <= 0.0f) return 0;
    }
    return 1;
}

static void record_baseline(const uint32_t *w, sentinel_baseline_t *b)
{
    for (unsigned i = 0; i < SENTINEL_PROFILE_FEATURES; ++i) {
        memcpy(&b->mean[i], &w[4u + i], sizeof(float));
        memcpy(&b->std[i], &w[8u + i], sizeof(float));
    }
}

static int record_valid(const uint32_t *w, unsigned profile)
{
    sentinel_baseline_t b;
    if (w[0] != MAGIC_VALID || w[1] != profile || w[2] != STORE_VERSION ||
        w[15] != checksum(w)) return 0;
    record_baseline(w, &b);
    return baseline_valid(&b);
}

static int record_owned_or_blank(const uint32_t *w, unsigned profile)
{
    if (w[0] == MAGIC_VALID) {
        return w[1] == profile && w[2] == STORE_VERSION;
    }
    if (w[0] == MAGIC_BUSY) {
        /* The first 8-byte MRAM store writes magic + owner together.
         * Version may still be blank if power fails before the second. */
        return w[1] == profile &&
               (w[2] == STORE_VERSION || w[2] == UINT32_MAX || w[2] == 0u);
    }
    int all_ff = 1, all_zero = 1;
    for (unsigned i = 0; i < 16u; ++i) {
        all_ff &= w[i] == UINT32_MAX;
        all_zero &= w[i] == 0u;
    }
    return all_ff || all_zero;
}

static int read_pair(unsigned id, uint32_t w[2][16])
{
    for (unsigned slot = 0; slot < 2u; ++slot) {
        if (store_read(slot_offset(id, slot), w[slot], RECORD_BYTES) != 0) return -1;
    }
    return 0;
}

static int newest_valid_slot(const uint32_t w[2][16], unsigned id)
{
    const int a = record_valid(w[0], id), b = record_valid(w[1], id);
    if (!a) return b ? 1 : -1;
    if (!b) return 0;
    return (int32_t) (w[1][3] - w[0][3]) > 0 ? 1 : 0;
}

int sentinel_profile_load(unsigned id, sentinel_baseline_t *out)
{
    uint32_t w[2][16];
    if (id >= SENTINEL_PROFILE_COUNT || !out || read_pair(id, w) != 0) return -1;
    const int newest = newest_valid_slot(w, id);
    if (newest >= 0) {
        record_baseline(w[newest], out);
        return 1;
    }
    return record_owned_or_blank(w[0], id) && record_owned_or_blank(w[1], id) ? 0 : -1;
}

int sentinel_profile_save(unsigned id, const sentinel_baseline_t *value)
{
    uint32_t pair[2][16], w[16] = {0}, verify[16];
    if (id >= SENTINEL_PROFILE_COUNT || !baseline_valid(value) ||
        read_pair(id, pair) != 0) return -1;
    if (!record_owned_or_blank(pair[0], id) || !record_owned_or_blank(pair[1], id)) return -1;

    const int newest = newest_valid_slot(pair, id);
    const unsigned target = newest == 0 ? 1u : 0u;
    const unsigned offset = slot_offset(id, target);
    w[0] = MAGIC_VALID;
    w[1] = id;
    w[2] = STORE_VERSION;
    w[3] = newest < 0 ? 1u : pair[newest][3] + 1u;
    for (unsigned i = 0; i < SENTINEL_PROFILE_FEATURES; ++i) {
        memcpy(&w[4u + i], &value->mean[i], sizeof(float));
        memcpy(&w[8u + i], &value->std[i], sizeof(float));
    }
    w[15] = checksum(w);

    uint32_t busy_header[4];
    memcpy(busy_header, w, sizeof busy_header);
    busy_header[0] = MAGIC_BUSY;
    if (store_write16(offset, busy_header) != 0) return -1;
    for (unsigned i = 1u; i < 4u; ++i) {
        if (store_write16(offset + i * SLOT_BYTES, w + 4u * i) != 0) return -1;
    }
    if (store_write16(offset, w) != 0) return -1;
    if (store_read(offset, verify, sizeof verify) != 0 || !record_valid(verify, id)) return -1;
    sentinel_baseline_t got;
    record_baseline(verify, &got);
    return memcmp(&got, value, sizeof got) == 0 ? 0 : -1;
}
