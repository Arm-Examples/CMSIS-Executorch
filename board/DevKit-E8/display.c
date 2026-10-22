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
 * scaled up by the largest integer factor that fits the panel, centred on a
 * black background and written into the RGB888 frame buffer in bulk SRAM.
 * The display is brought up on the first call (CDC200 + DSI + panel), as in
 * the pack's Boards/Templates/Baremetal/demo_cdc200.c. board_display_line()
 * and board_display_rect() draw the Sentinel status text.
 *---------------------------------------------------------------------------*/

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "RTE_Components.h"
#include CMSIS_device_header
#include "RTE_Device.h"

#include "Driver_CDC200.h"

#define LCD_WIDTH   RTE_PANEL_HACTIVE_TIME   /* 480 */
#define LCD_HEIGHT  RTE_PANEL_VACTIVE_LINE   /* 800 */

#if RTE_CDC200_PIXEL_FORMAT != 1
#error "display.c expects RTE_CDC200_PIXEL_FORMAT 1 (RGB888) in RTE_Device.h"
#endif
#define LCD_BPP     3

extern ARM_DRIVER_CDC200 Driver_CDC200;
static ARM_DRIVER_CDC200 *CDCdrv = &Driver_CDC200;

/* RGB888 frame buffer, 480 x 800 x 3 = 1152000 bytes, in bulk SRAM (NPU and
   display DMA accessible); the layer's linker scripts route the section. */
static uint8_t lcd_frame[LCD_HEIGHT][LCD_WIDTH][LCD_BPP]
    __attribute__((section(".bss.lcd_frame_buf"), aligned(32)));

static int display_ready = 0;

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
    return 0;
}

/* 5 x 7 glyphs (bit 4 = left column): A-Z, 0-9 and : / - . ; other
   characters are drawn blank. */
static const struct {
    char    c;
    uint8_t rows[7];
} text_font[] = {
    {'A', {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'B', {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}},
    {'C', {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}},
    {'D', {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E}},
    {'E', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}},
    {'F', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}},
    {'G', {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}},
    {'H', {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'I', {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'J', {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C}},
    {'K', {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}},
    {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}},
    {'M', {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}},
    {'N', {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11}},
    {'O', {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
    {'P', {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}},
    {'Q', {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}},
    {'R', {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}},
    {'S', {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}},
    {'T', {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}},
    {'U', {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
    {'V', {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}},
    {'W', {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A}},
    {'X', {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}},
    {'Y', {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}},
    {'Z', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}},
    {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}},
    {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}},
    {'3', {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}},
    {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}},
    {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
    {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}},
    {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
    {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}},
    {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
    {':', {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00}},
    {'/', {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10}},
    {'-', {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}},
    {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}},
};

/* Logical view for the status text. Portrait by default (the panel's
   480 x 800 rows, as board_display_image()); with APP_DISPLAY_LANDSCAPE the
   panel is viewed with its long side horizontal: an 800 x 480 view drawn
   rotated by 90 degrees into the frame buffer, the other way round with
   DISPLAY_LANDSCAPE_FLIP 1, so text reads left to right. */
#ifdef APP_DISPLAY_LANDSCAPE
#ifndef DISPLAY_LANDSCAPE_FLIP
#define DISPLAY_LANDSCAPE_FLIP 0
#endif
#define VIEW_W LCD_HEIGHT
#define VIEW_H LCD_WIDTH
static inline uint8_t *view_px(int lx, int ly)
{
#if DISPLAY_LANDSCAPE_FLIP
    return lcd_frame[VIEW_W - 1 - lx][ly];
#else
    return lcd_frame[lx][VIEW_H - 1 - ly];
#endif
}
#else
#define VIEW_W LCD_WIDTH
#define VIEW_H LCD_HEIGHT
static inline uint8_t *view_px(int lx, int ly)
{
    return lcd_frame[ly][lx];
}
#endif

static int view_ready(void)
{
    if (!display_ready) {
        if (display_init() != 0) {
            return -1;
        }
        display_ready = 1;
    }
    return 0;
}

/* Fill a view rectangle (clipped to the view) with a 0xRRGGBB colour */
static void view_fill(int x0, int y0, int w, int h, uint32_t rgb)
{
    for (int y = y0 < 0 ? 0 : y0; y < y0 + h && y < VIEW_H; y++) {
        for (int x = x0 < 0 ? 0 : x0; x < x0 + w && x < VIEW_W; x++) {
            uint8_t *px = view_px(x, y);
            /* CDC200 RGB888: B, G, R in memory */
            px[0] = (uint8_t) rgb;
            px[1] = (uint8_t) (rgb >> 8);
            px[2] = (uint8_t) (rgb >> 16);
        }
    }
}

/* Push the frame buffer out of the cache for the display controller's DMA
   (in landscape a band of view rows spans every frame row). */
static void view_clean(int y0, int h)
{
#ifdef APP_DISPLAY_LANDSCAPE
    (void) y0;
    (void) h;
    SCB_CleanDCache_by_Addr((uint32_t *) lcd_frame, (int32_t) sizeof(lcd_frame));
#else
    SCB_CleanDCache_by_Addr((uint32_t *) &lcd_frame[y0][0][0], (int32_t)(h * LCD_WIDTH * LCD_BPP));
#endif
}

/*
  Draw one line of text across the view: fill the band of rows y .. y+h-1
  with bg, then draw text centred in it with 5 x 7 glyphs scaled by scale
  (a glyph is 5*scale x 7*scale pixels, one glyph column apart). Text that
  does not fit the view width is cut. Colours are 0xRRGGBB.

  \return      0 on success, -1 when the display is not available
*/
int board_display_line(int y, int h, int scale, const char *text, uint32_t fg, uint32_t bg)
{
    if (view_ready() != 0 || y < 0 || h <= 0 || y + h > VIEW_H || scale <= 0) {
        return -1;
    }
    view_fill(0, y, VIEW_W, h, bg);

    const int adv = 6 * scale;
    int       n   = (int) strlen(text);
    if (n * adv - scale > VIEW_W) {
        n = (VIEW_W + scale) / adv;
    }
    const int x0 = (VIEW_W - (n * adv - scale)) / 2;
    const int y0 = y + (h - 7 * scale) / 2;
    for (int i = 0; i < n; i++) {
        const uint8_t *rows = NULL;
        for (unsigned k = 0; k < sizeof text_font / sizeof text_font[0]; k++) {
            if (text_font[k].c == text[i]) {
                rows = text_font[k].rows;
            }
        }
        if (rows == NULL) {
            continue;
        }
        for (int gy = 0; gy < 7 * scale; gy++) {
            for (int gx = 0; gx < 5 * scale; gx++) {
                if (rows[gy / scale] & (0x10u >> (gx / scale))) {
                    view_fill(x0 + i * adv + gx, y0 + gy, 1, 1, fg);
                }
            }
        }
    }
    view_clean(y, h);
    return 0;
}

/*
  Fill a rectangle of the view with a 0xRRGGBB colour (for bars).

  \return      0 on success, -1 when the display is not available
*/
int board_display_rect(int x, int y, int w, int h, uint32_t rgb)
{
    if (view_ready() != 0) {
        return -1;
    }
    view_fill(x, y, w, h, rgb);
    view_clean(y, h);
    return 0;
}
