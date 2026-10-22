/* Host control test: exercises the actual Sentinel selection and learn path. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../board/DevKit-E8/sentinel_profile_store.h"
#include "fakes/fake_device.h"

fake_dwt_t fake_dwt;
uint32_t SystemCoreClock = 400000000u;

static sentinel_baseline_t saved[2];
static int saved_valid[2], save_fail, saves;
int sentinel_profile_load(unsigned id, sentinel_baseline_t *out)
{
    if (id >= 2) return -1;
    if (!saved_valid[id]) return 0;
    *out = saved[id];
    return 1;
}
int sentinel_profile_save(unsigned id, const sentinel_baseline_t *value)
{
    if (save_fail || id >= 2) return -1;
    saved[id] = *value;
    saved_valid[id] = 1;
    ++saves;
    return 0;
}
int board_imu_init(void) { return 0; }
int board_imu_accel_mg(int32_t mg[3]) { mg[0] = mg[1] = 0; mg[2] = 1000; return 0; }
int board_mic_init(void) { return 0; }
void board_mic_poll(uint32_t now_ms) { (void) now_ms; }
void board_mic_counters(uint32_t *done, uint32_t *err, uint32_t *full)
{ *done = *err = *full = 0; }
int board_display_line(int y, int h, int scale, const char *text, uint32_t fg, uint32_t bg)
{ (void)y; (void)h; (void)scale; (void)text; (void)fg; (void)bg; return 0; }
int board_display_rect(int x, int y, int w, int h, uint32_t rgb)
{ (void)x; (void)y; (void)w; (void)h; (void)rgb; return 0; }

#include "../board/DevKit-E8/sentinel.c"

static void saved_profile_selects_and_switch_resets(void)
{
    saved_valid[0] = 1;
    for (unsigned i = 0; i < 4; ++i) {
        saved[0].mean[i] = (float) i;
        saved[0].std[i] = 0.25f;
    }
    assert(board_sentinel_init() == 0);
    assert(sn_state == SN_IDLE);
    board_sentinel_select(0);
    assert(sn_state == SN_VERIFYING);
    assert(bl_mean[0] == 0.0 && bl_mean[3] == 3.0);
    for (unsigned i = 0; i < 3; ++i) {
        win_n = 16000;
        win_p4 = win_p5 = win_d4 = 16000.0;
        win_imu_n = 0;
        sn_window(1000u * (i + 1u));
        assert(sn_state == (i < 2 ? SN_VERIFYING : SN_CHECK));
    }
    win_blocks = 7;
    sens_anom[0] = 1;
    board_sentinel_select(1);
    assert(sn_state == SN_IDLE);
    assert(win_blocks == 0 && sens_anom[0] == 0);
}

static void learning_selected_profile_saves_it(void)
{
    board_sentinel_learn();
    assert(sn_state == SN_LEARNING);
    for (unsigned i = 0; i < SN_NF; ++i) {
        bl_n[i] = SN_LEARN_WINDOWS - 1u;
        bl_mean[i] = 1.0 + i;
        bl_m2[i] = 0.2;
    }
    learn_windows = SN_LEARN_WINDOWS - 1u;
    win_n = 16000;
    win_p4 = win_p5 = 16000.0;
    win_d4 = 16000.0;
    win_imu_n = 20;
    win_a_sum = 20000.0;
    win_a_sq = 20000010.0;
    sn_window(30000u);
    assert(sn_state == SN_OK);
    assert(saved_valid[1] && saves == 1);
    assert(sn_save_failed == 0);
    board_sentinel_select(0);
    assert(sn_state == SN_VERIFYING && bl_mean[3] == 3.0);
    board_sentinel_select(1);
    assert(sn_state == SN_VERIFYING && bl_mean[0] == saved[1].mean[0]);
}

static void failed_save_is_visible(void)
{
    save_fail = 1;
    board_sentinel_learn();
    for (unsigned i = 0; i < SN_NF; ++i) {
        bl_n[i] = SN_LEARN_WINDOWS - 1u;
        bl_mean[i] = 1.0 + i;
        bl_m2[i] = 0.2;
    }
    learn_windows = SN_LEARN_WINDOWS - 1u;
    win_n = 16000;
    win_p4 = win_p5 = win_d4 = 16000.0;
    win_imu_n = 20;
    win_a_sum = 20000.0;
    win_a_sq = 20000010.0;
    sn_window(60000u);
    assert(sn_state == SN_OK);
    assert(sn_save_failed == 1);
    board_sentinel_select(0);
    board_sentinel_select(1);
    assert(sn_state == SN_VERIFYING);
    assert(sn_save_failed == 1);
}

int main(void)
{
    saved_profile_selects_and_switch_resets();
    learning_selected_profile_saves_it();
    failed_save_is_visible();
    puts("sentinel profile control: PASS");
    return 0;
}
