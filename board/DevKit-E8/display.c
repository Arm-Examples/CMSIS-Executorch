/*---------------------------------------------------------------------------
 * Copyright (c) 2026 Arm Limited (or its affiliates). All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * DevKit-E8 display: shows an RGB image on the 4.3" MIPI-DSI panel (ILI9806E,
 * 480 x 800 portrait) through the CDC200 display controller. The application
 * calls board_display_image() with an 8-bit RGB image (HWC); the image is
 * scaled up on the panel and written into the RGB888 frame buffer in SRAM1.
 * The touchscreen UI draws mode and conditioning controls above the image.
 * The display is brought up on the first call (CDC200 + DSI + panel), as in
 * the pack's Boards/Templates/Baremetal/demo_cdc200.c.
 *---------------------------------------------------------------------------*/

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "RTE_Components.h"
#include CMSIS_device_header
#include "RTE_Device.h"

#include "Driver_CDC200.h"
#include "touch_ui.h"
#include "ethos_u85_logo.h"

#define LCD_WIDTH   RTE_PANEL_HACTIVE_TIME   /* 480 */
#define LCD_HEIGHT  RTE_PANEL_VACTIVE_LINE   /* 800 */
#define OK_WIDTH    264
#define OK_HEIGHT   108
#define LOGO_GAP    12
#define OK_X        ((LCD_WIDTH - OK_WIDTH - LOGO_GAP - ETHOS_U85_LOGO_WIDTH) / 2)
#define OK_Y        (LCD_HEIGHT - OK_HEIGHT - 32)
#define LOGO_X      (OK_X + OK_WIDTH + LOGO_GAP)
#define LOGO_Y      (OK_Y + (OK_HEIGHT - ETHOS_U85_LOGO_HEIGHT) / 2)

#define MODE_Y      66
#define MODE_H      80
#define CHOICE_Y1   222
#define CHOICE_Y2   312
#define CHOICE_H    58
#define LEFT_X      24
#define RIGHT_X     246
#define CHOICE_W    210

#if RTE_CDC200_PIXEL_FORMAT != 1
#error "display.c expects RTE_CDC200_PIXEL_FORMAT 1 (RGB888) in RTE_Device.h"
#endif
#define LCD_BPP     3

extern ARM_DRIVER_CDC200 Driver_CDC200;
static ARM_DRIVER_CDC200 *CDCdrv = &Driver_CDC200;

/* RGB888 frame buffer, 480 x 800 x 3 = 1152000 bytes, in SRAM1 so SRAM0 stays
   available to the NPU; the layer's linker scripts route the section. */
static uint8_t lcd_frame[LCD_HEIGHT][LCD_WIDTH][LCD_BPP]
    __attribute__((section(".bss.lcd_frame_buf"), aligned(32)));

static int display_ready = 0;

/* CDC200 stores RGB888 pixels in B, G, R byte order. */
static void set_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t *pixel = &lcd_frame[y][x][0];
    pixel[0] = b;
    pixel[1] = g;
    pixel[2] = r;
}

static void fill_rect(int x, int y, int width, int height, uint8_t r, uint8_t g, uint8_t b)
{
    for (int py = y; py < y + height; py++) {
        for (int px = x; px < x + width; px++) {
            set_pixel(px, py, r, g, b);
        }
    }
}

/* Five-column uppercase bitmap font; each bit in a column is a row. */
static void glyph(char ch, uint8_t out[5])
{
    static const struct { char ch; uint8_t cols[5]; } letters[] = {
        {'A', {0x7e, 0x11, 0x11, 0x11, 0x7e}},
        {'C', {0x3e, 0x41, 0x41, 0x41, 0x22}},
        {'D', {0x7f, 0x41, 0x41, 0x22, 0x1c}},
        {'E', {0x7f, 0x49, 0x49, 0x49, 0x41}},
        {'F', {0x7f, 0x09, 0x09, 0x09, 0x01}},
        {'G', {0x3e, 0x41, 0x49, 0x49, 0x7a}},
        {'I', {0x41, 0x41, 0x7f, 0x41, 0x41}},
        {'L', {0x7f, 0x40, 0x40, 0x40, 0x40}},
        {'M', {0x7f, 0x02, 0x0c, 0x02, 0x7f}},
        {'N', {0x7f, 0x04, 0x08, 0x10, 0x7f}},
        {'O', {0x3e, 0x41, 0x41, 0x41, 0x3e}},
        {'P', {0x7f, 0x09, 0x09, 0x09, 0x06}},
        {'R', {0x7f, 0x09, 0x19, 0x29, 0x46}},
        {'S', {0x46, 0x49, 0x49, 0x49, 0x31}},
        {'T', {0x01, 0x01, 0x7f, 0x01, 0x01}},
        {'U', {0x3f, 0x40, 0x40, 0x40, 0x3f}},
        {'X', {0x63, 0x14, 0x08, 0x14, 0x63}},
        {'Y', {0x07, 0x08, 0x70, 0x08, 0x07}},
    };
    memset(out, 0, 5);
    for (size_t i = 0; i < sizeof(letters) / sizeof(letters[0]); i++) {
        if (letters[i].ch == ch) {
            memcpy(out, letters[i].cols, 5);
            return;
        }
    }
}

static void draw_text(int x, int y, const char *s, int scale, uint8_t r, uint8_t g, uint8_t b)
{
    while (*s) {
        uint8_t cols[5];
        glyph(*s++, cols);
        for (int col = 0; col < 5; col++) {
            for (int row = 0; row < 7; row++) {
                if (cols[col] & (1u << row)) {
                    fill_rect(x + col * scale, y + row * scale, scale, scale, r, g, b);
                }
            }
        }
        x += 6 * scale;
    }
}

static void draw_button(int x, int y, int width, int height, const char *label, int selected)
{
    const int scale = 2;
    fill_rect(x, y, width, height, selected ? 48 : 65, selected ? 173 : 72,
              selected ? 117 : 88);
    fill_rect(x + 3, y + 3, width - 6, height - 6, selected ? 19 : 27,
              selected ? 110 : 35, selected ? 74 : 48);
    int text_width = (int)strlen(label) * 6 * scale - scale;
    draw_text(x + (width - text_width) / 2, y + (height - 7 * scale) / 2,
              label, scale, 255, 255, 255);
}

static void fill_circle(int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b)
{
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            if (x * x + y * y <= radius * radius) {
                set_pixel(cx + x, cy + y, r, g, b);
            }
        }
    }
}

static void draw_stroke(int x0, int y0, int x1, int y1, int radius,
                        uint8_t r, uint8_t g, uint8_t b)
{
    int dx = x1 - x0, dy = y1 - y0;
    int steps = (dx < 0 ? -dx : dx);
    if ((dy < 0 ? -dy : dy) > steps) {
        steps = dy < 0 ? -dy : dy;
    }
    for (int i = 0; i <= steps; i++) {
        fill_circle(x0 + dx * i / steps, y0 + dy * i / steps, radius, r, g, b);
    }
}

/* The source JPEG is converted to RGB565 so the LCD needs no JPEG decoder. */
static void draw_ethos_u85_logo(void)
{
    for (int y = 0; y < ETHOS_U85_LOGO_HEIGHT; y++) {
        for (int x = 0; x < ETHOS_U85_LOGO_WIDTH; x++) {
            const uint16_t pixel = ethos_u85_logo[y * ETHOS_U85_LOGO_WIDTH + x];
            const uint8_t r = (uint8_t)(pixel >> 11);
            const uint8_t g = (uint8_t)((pixel >> 5) & 0x3f);
            const uint8_t b = (uint8_t)(pixel & 0x1f);
            set_pixel(LOGO_X + x, LOGO_Y + y,
                      (uint8_t)((r << 3) | (r >> 2)),
                      (uint8_t)((g << 2) | (g >> 4)),
                      (uint8_t)((b << 3) | (b >> 2)));
        }
    }
}

/* Rounded green Generate badge with the Ethos-U85 image beside it. */
static int draw_ok_sign(void)
{
    const int badge_width = OK_WIDTH, badge_height = OK_HEIGHT;
    const int badge_x = OK_X, badge_y = OK_Y;
    const int bar_top = badge_y - 24;
    const int radius = badge_height / 2;
    const int icon_x = badge_x + 60, mid_y = badge_y + radius;

    /* Keep the badge readable even if a future image extends into this area. */
    memset(&lcd_frame[bar_top][0][0], 0,
           (size_t)(LCD_HEIGHT - bar_top) * LCD_WIDTH * LCD_BPP);
    for (int y = badge_y; y < badge_y + badge_height; y++) {
        for (int x = badge_x; x < badge_x + badge_width; x++) {
            int cap_x = x < badge_x + radius ? badge_x + radius : x;
            if (cap_x > badge_x + badge_width - radius - 1) {
                cap_x = badge_x + badge_width - radius - 1;
            }
            const int dx = x - cap_x, dy = y - mid_y;
            const int distance = dx * dx + dy * dy;
            if (distance <= radius * radius) {
                if (distance > (radius - 3) * (radius - 3)) {
                    set_pixel(x, y, 78, 220, 125);
                } else {
                    const int shade = (y - badge_y) * 34 / badge_height;
                    set_pixel(x, y, 35 - shade / 2, 178 - shade, 87 - shade / 2);
                }
            }
        }
    }

    fill_circle(icon_x, mid_y, 31, 242, 255, 245);
    draw_stroke(icon_x - 15, mid_y, icon_x - 4, mid_y + 11, 4, 24, 132, 70);
    draw_stroke(icon_x - 4, mid_y + 11, icon_x + 18, mid_y - 15, 4, 24, 132, 70);

    draw_text(badge_x + 105, mid_y - 10, "GENERATE", 3, 255, 255, 255);
    draw_ethos_u85_logo();
    return bar_top;
}

/* Use the same bounds for drawing and touch hit testing. */
static int board_display_ok_contains(int x, int y)
{
    if (x < OK_X || x >= OK_X + OK_WIDTH ||
        y < OK_Y || y >= OK_Y + OK_HEIGHT) {
        return 0;
    }
    const int radius = OK_HEIGHT / 2;
    int cap_x = x < OK_X + radius ? OK_X + radius : x;
    if (cap_x > OK_X + OK_WIDTH - radius - 1) {
        cap_x = OK_X + OK_WIDTH - radius - 1;
    }
    const int dx = x - cap_x, dy = y - (OK_Y + radius);
    return dx * dx + dy * dy <= radius * radius;
}

static int in_rect(int x, int y, int left, int top, int width, int height)
{
    return x >= left && x < left + width && y >= top && y < top + height;
}

int board_display_hit_test(int x, int y, int mode, int gender, int expression)
{
    if (in_rect(x, y, LEFT_X, MODE_Y, CHOICE_W, MODE_H)) return TOUCH_EVENT_CONDITIONAL;
    if (in_rect(x, y, RIGHT_X, MODE_Y, CHOICE_W, MODE_H)) return TOUCH_EVENT_UNCONDITIONAL;
    if (mode == TOUCH_MODE_CONDITIONAL) {
        if (in_rect(x, y, LEFT_X, CHOICE_Y1, CHOICE_W, CHOICE_H)) return TOUCH_EVENT_MALE;
        if (in_rect(x, y, RIGHT_X, CHOICE_Y1, CHOICE_W, CHOICE_H)) return TOUCH_EVENT_FEMALE;
        if (in_rect(x, y, LEFT_X, CHOICE_Y2, CHOICE_W, CHOICE_H)) return TOUCH_EVENT_NEUTRAL;
        if (in_rect(x, y, RIGHT_X, CHOICE_Y2, CHOICE_W, CHOICE_H)) return TOUCH_EVENT_SMILING;
    }
    if ((mode == TOUCH_MODE_UNCONDITIONAL ||
         (mode == TOUCH_MODE_CONDITIONAL && gender != TOUCH_CHOICE_NONE &&
          expression != TOUCH_CHOICE_NONE)) && board_display_ok_contains(x, y)) {
        return TOUCH_EVENT_GENERATE;
    }
    return TOUCH_EVENT_NONE;
}

static void display_callback(uint32_t event)
{
    if (event & ARM_CDC_DSI_ERROR_EVENT) {
        printf("Display: DSI error event\n");
    }
}

/* Bring up the display controller, DSI link and panel; returns 0 on success. */
static int display_init(void)
{
    int32_t ret;

    memset(lcd_frame, 0, sizeof(lcd_frame));
    SCB_CleanDCache_by_Addr((uint32_t *) lcd_frame, sizeof(lcd_frame));

    ret = CDCdrv->Initialize(display_callback);
    if (ret != ARM_DRIVER_OK) {
        printf("Display: CDC200 Initialize failed (%d)\n", (int) ret);
        return -1;
    }
    ret = CDCdrv->PowerControl(ARM_POWER_FULL);
    if (ret != ARM_DRIVER_OK) {
        printf("Display: CDC200 PowerControl failed (%d)\n", (int) ret);
        return -1;
    }
    ret = CDCdrv->Control(CDC200_CONFIGURE_DISPLAY, (uint32_t) lcd_frame);
    if (ret != ARM_DRIVER_OK) {
        printf("Display: CDC200 configure failed (%d)\n", (int) ret);
        return -1;
    }
    ret = CDCdrv->Start();
    if (ret != ARM_DRIVER_OK) {
        printf("Display: CDC200 Start failed (%d)\n", (int) ret);
        return -1;
    }
    printf("Display: %ux%u RGB888 panel started\n", (unsigned) LCD_WIDTH, (unsigned) LCD_HEIGHT);
    return 0;
}

/*
  Show an 8-bit RGB (HWC, or gray when channels == 1) image on the panel.

  \param[in]   rgb       Pixel data, height x width x channels bytes
  \param[in]   width     Image width in pixels
  \param[in]   height    Image height in pixels
  \param[in]   channels  1 (gray) or 3 (RGB)
  \return      0 on success, -1 when the display is not available
*/
int board_display_image(const uint8_t *rgb, int width, int height, int channels)
{
    if (!display_ready) {
        if (display_init() != 0) {
            return -1;
        }
        display_ready = 1;
    }
    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
        return -1;
    }

    /* Largest integer scale that fits, centred. */
    int scale = LCD_WIDTH / width;
    if (LCD_HEIGHT / height < scale) {
        scale = LCD_HEIGHT / height;
    }
    if (scale < 1) {
        scale = 1;
    }
    const int out_w = width * scale, out_h = height * scale;
    const int x0 = (LCD_WIDTH - out_w) / 2, y0 = (LCD_HEIGHT - out_h) / 2;

    for (int y = 0; y < out_h && y0 + y < LCD_HEIGHT; y++) {
        const uint8_t *row = rgb + (size_t)(y / scale) * width * channels;
        uint8_t *dst = &lcd_frame[y0 + y][x0][0];
        for (int x = 0; x < out_w && x0 + x < LCD_WIDTH; x++) {
            const uint8_t *px = row + (x / scale) * channels;
            /* CDC200 RGB888: little-endian 0x00RRGGBB in memory, i.e. B, G, R */
            dst[0] = channels == 3 ? px[2] : px[0];
            dst[1] = channels == 3 ? px[1] : px[0];
            dst[2] = px[0];
            dst += LCD_BPP;
        }
    }
    /* The frame buffer is read by the display controller's DMA: push the
       cached lines out (bulk SRAM is write-through, but be explicit). */
    SCB_CleanDCache_by_Addr((uint32_t *) &lcd_frame[y0][0][0], (int32_t)(out_h * LCD_WIDTH * LCD_BPP));
    const int bar_top = draw_ok_sign();
    SCB_CleanDCache_by_Addr((uint32_t *) &lcd_frame[bar_top][0][0],
                            (int32_t)((LCD_HEIGHT - bar_top) * LCD_WIDTH * LCD_BPP));
    return 0;
}

/* Draw all touchscreen controls and the latest face in one frame. A null image
   leaves the preview area empty, including on the initial mode-only screen. */
int board_display_ui(int mode, int gender, int expression, const uint8_t *image,
                     int width, int height, int channels)
{
    if (!display_ready) {
        if (display_init() != 0) return -1;
        display_ready = 1;
    }
    memset(lcd_frame, 0, sizeof(lcd_frame));
    draw_text(150, 25, "GENERATION MODE", 2, 232, 240, 246);
    draw_button(LEFT_X, MODE_Y, CHOICE_W, MODE_H, "CONDITIONAL",
                mode == TOUCH_MODE_CONDITIONAL);
    draw_button(RIGHT_X, MODE_Y, CHOICE_W, MODE_H, "UNCONDITIONAL",
                mode == TOUCH_MODE_UNCONDITIONAL);

    if (mode == TOUCH_MODE_CONDITIONAL) {
        draw_text(120, 164, "CONDITIONING OPTIONS", 2, 190, 205, 221);
        draw_text(24, 193, "GENDER", 2, 232, 240, 246);
        draw_button(LEFT_X, CHOICE_Y1, CHOICE_W, CHOICE_H, "MALE",
                    gender == TOUCH_GENDER_MALE);
        draw_button(RIGHT_X, CHOICE_Y1, CHOICE_W, CHOICE_H, "FEMALE",
                    gender == TOUCH_GENDER_FEMALE);
        draw_text(24, 289, "EXPRESSION", 2, 232, 240, 246);
        draw_button(LEFT_X, CHOICE_Y2, CHOICE_W, CHOICE_H, "NEUTRAL",
                    expression == TOUCH_EXPRESSION_NEUTRAL);
        draw_button(RIGHT_X, CHOICE_Y2, CHOICE_W, CHOICE_H, "SMILING",
                    expression == TOUCH_EXPRESSION_SMILING);
    }

    if (image != NULL) {
        const int top = mode == TOUCH_MODE_CONDITIONAL ? 378 : 164;
        const int available_h = 636 - top;
        if (width <= 0 || height <= 0 || (channels != 1 && channels != 3) ||
            width > LCD_WIDTH || height > available_h) return -1;
        int scale = LCD_WIDTH / width;
        if (available_h / height < scale) scale = available_h / height;
        const int out_w = width * scale, out_h = height * scale;
        const int x0 = (LCD_WIDTH - out_w) / 2;
        const int y0 = top + (available_h - out_h) / 2;
        for (int y = 0; y < out_h; y++) {
            const uint8_t *row = image + (size_t)(y / scale) * width * channels;
            uint8_t *dst = &lcd_frame[y0 + y][x0][0];
            for (int x = 0; x < out_w; x++) {
                const uint8_t *px = row + (x / scale) * channels;
                dst[0] = channels == 3 ? px[2] : px[0];
                dst[1] = channels == 3 ? px[1] : px[0];
                dst[2] = px[0];
                dst += LCD_BPP;
            }
        }
    }

    if (mode == TOUCH_MODE_UNCONDITIONAL ||
        (mode == TOUCH_MODE_CONDITIONAL && gender != TOUCH_CHOICE_NONE &&
         expression != TOUCH_CHOICE_NONE)) {
        draw_ok_sign();
    }
    SCB_CleanDCache_by_Addr((uint32_t *)lcd_frame, sizeof(lcd_frame));
    return 0;
}
