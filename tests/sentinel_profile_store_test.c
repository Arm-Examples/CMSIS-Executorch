/* Host tests for the exact profile record code used on the E8. */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SENTINEL_PROFILE_HOST_TEST 1
#include "../board/DevKit-E8/sentinel_profile_store.c"

static sentinel_baseline_t sample(float shift)
{
    sentinel_baseline_t b = {{1.1f + shift, 0.7f, -1.0f, 0.9f},
                             {0.15f, 0.09f, 0.06f, 0.20f}};
    return b;
}

static void empty_then_both_profiles(void)
{
    sentinel_baseline_t got;
    sentinel_baseline_t fan = sample(0.0f), second = sample(3.0f);
    memset(test_mram, 0xff, sizeof test_mram);
    test_fail_after = -1;
    assert(sentinel_profile_load(0, &got) == 0);
    assert(sentinel_profile_load(1, &got) == 0);
    assert(sentinel_profile_save(0, &fan) == 0);
    assert(sentinel_profile_save(1, &second) == 0);
    assert(sentinel_profile_load(0, &got) == 1);
    assert(memcmp(&got, &fan, sizeof got) == 0);
    assert(sentinel_profile_load(1, &got) == 1);
    assert(memcmp(&got, &second, sizeof got) == 0);
}

static void overwrite_keeps_previous_valid_record(void)
{
    sentinel_baseline_t got;
    sentinel_baseline_t first = sample(0.0f), next = sample(1.0f);
    memset(test_mram, 0xff, sizeof test_mram);
    test_fail_after = -1;
    assert(sentinel_profile_save(0, &first) == 0);
    assert(sentinel_profile_save(0, &next) == 0);
    assert(sentinel_profile_load(0, &got) == 1);
    assert(memcmp(&got, &next, sizeof got) == 0);
    /* Flip one payload bit in the newest slot. The older baseline survives. */
    test_mram[64 + 24] ^= 1;
    assert(sentinel_profile_load(0, &got) == 1);
    assert(memcmp(&got, &first, sizeof got) == 0);
}

static void interrupted_overwrite_keeps_old_profile(void)
{
    for (int fail = 0; fail < 8; ++fail) {
        sentinel_baseline_t got;
        sentinel_baseline_t first = sample(0.0f), next = sample(2.0f);
        memset(test_mram, 0xff, sizeof test_mram);
        test_fail_after = -1;
        assert(sentinel_profile_save(0, &first) == 0);
        test_fail_after = fail;
        assert(sentinel_profile_save(0, &next) == -1);
        test_fail_after = -1;
        assert(sentinel_profile_load(0, &got) == 1);
        assert(memcmp(&got, &first, sizeof got) == 0);
    }
}

static void torn_halfword_can_be_retried(void)
{
    for (int fail = 0; fail < 10; ++fail) {
        sentinel_baseline_t got, first = sample(0.0f), next = sample(2.0f);
        memset(test_mram, 0xff, sizeof test_mram);
        test_fail_after = fail;
        assert(sentinel_profile_save(0, &first) == -1);
        test_fail_after = -1;
        assert(sentinel_profile_save(0, &next) == 0);
        assert(sentinel_profile_load(0, &got) == 1);
        assert(memcmp(&got, &next, sizeof got) == 0);
    }
}

static void reject_foreign_content_and_invalid_baseline(void)
{
    sentinel_baseline_t b = sample(0.0f), got;
    memset(test_mram, 0xff, sizeof test_mram);
    test_fail_after = -1;
    test_mram[0] = 0x31;
    assert(sentinel_profile_save(0, &b) == -1);
    assert(test_mram[0] == 0x31);
    memset(test_mram, 0xff, sizeof test_mram);
    b.std[2] = 0.0f;
    assert(sentinel_profile_save(0, &b) == -1);
    assert(sentinel_profile_load(0, &got) == 0);
    b = sample(0.0f);
    b.mean[1] = NAN;
    assert(sentinel_profile_save(0, &b) == -1);
}

static void reject_record_for_the_wrong_profile(void)
{
    sentinel_baseline_t b = sample(0.0f), got;
    memset(test_mram, 0xff, sizeof test_mram);
    test_fail_after = -1;
    assert(sentinel_profile_save(0, &b) == 0);
    memcpy(test_mram + 128, test_mram, 64);
    assert(sentinel_profile_load(1, &got) == -1);
    assert(sentinel_profile_save(1, &b) == -1);
    assert(memcmp(test_mram + 128, test_mram, 64) == 0);
}

int main(void)
{
    empty_then_both_profiles();
    overwrite_keeps_previous_valid_record();
    interrupted_overwrite_keeps_old_profile();
    torn_halfword_can_be_retried();
    reject_foreign_content_and_invalid_baseline();
    reject_record_for_the_wrong_profile();
    puts("sentinel profile store: PASS");
    return 0;
}
