// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

// Selection values shared by the touchscreen, display, and generator.
enum {
    TOUCH_MODE_NONE = 0,
    TOUCH_MODE_CONDITIONAL,
    TOUCH_MODE_UNCONDITIONAL,
};

enum {
    TOUCH_GENDER_MALE = 0,
    TOUCH_GENDER_FEMALE,
    TOUCH_EXPRESSION_NEUTRAL = 0,
    TOUCH_EXPRESSION_SMILING,
    TOUCH_CHOICE_NONE = -1,
};

enum {
    TOUCH_EVENT_NONE = 0,
    TOUCH_EVENT_CONDITIONAL,
    TOUCH_EVENT_UNCONDITIONAL,
    TOUCH_EVENT_MALE,
    TOUCH_EVENT_FEMALE,
    TOUCH_EVENT_NEUTRAL,
    TOUCH_EVENT_SMILING,
    TOUCH_EVENT_GENERATE,
};

#ifdef __cplusplus
extern "C" {
#endif

int board_touch_init(void);
int board_touch_poll(int mode, int gender, int expression);
int board_display_hit_test(int x, int y, int mode, int gender, int expression);
int board_display_ui(int mode, int gender, int expression, const uint8_t *image,
                     int width, int height, int channels);

#ifdef __cplusplus
}
#endif
