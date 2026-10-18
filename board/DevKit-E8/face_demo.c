/*---------------------------------------------------------------------------
 * Live face generator for the Alif DevKit-E8, drawn on the MIPI LCD.
 *
 * app_main (src/app_main.cpp) loads the FaceDecoder on the Ethos-U85 and then
 * calls face_demo_loop(), which never returns. Every frame it runs the decoder
 * on the NPU (face_generate), scales the 32x32 grey face up with bilinear
 * smoothing, and shows the 16 numbers the AI is "thinking" as bars below it.
 *
 *   joystick A/B   morph along one direction of the face space (+ / -)
 *   joystick C/D   morph along another direction (+ / -)
 *   joystick CENTER new random face
 *   no input       the face slowly drifts to random new faces by itself
 *
 * Built only with BOARD_FACE_DEMO (board layer define); BOARD_BUNNY_DEMO builds
 * bunny_demo.c instead.
 *---------------------------------------------------------------------------*/

#ifdef BOARD_FACE_DEMO

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#include "board_config.h"
#include "Driver_IO.h"
#include "Driver_I2C.h"

#include "display.h"
#include "camera.h"
#include "mic.h"
#include "imu.h"

#define IMU_MOTION_TH 25.0f   /* gyro deg/s (+100 x jolt in g); still board ~0-5 */
#define CAM_MOTION_TH 0.12f   /* fraction of camera grid cells that changed */

volatile float g_imu_motion, g_cam_motion;  /* debugger-readable */
#include "Driver_Touch_Screen.h"

extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(15); /* joystick: LPGPIO port 15 */
#define JOY_GPIO (&ARM_Driver_GPIO_(15))
extern int face_generate(const float *latent, uint8_t *pixels);

#define LATENT     16
#define SRC        32
#define PIXELS     (SRC * SRC)
#define STEP       0.05f   /* latent units moved per frame while a direction is held */
#define AUTO_RATE  0.02f   /* how fast the idle drift approaches its target          */
#define IDLE_AUTO  60      /* frames without input before the face drifts by itself  */
#define MAX_SIDE   1024

/* Joystick pins; the pad pull-ups are set by the board pin configuration. */
static const uint8_t joy_pin[5] = {
    BOARD_JOY_SW_A_GPIO_PIN,
    BOARD_JOY_SW_B_GPIO_PIN,
    BOARD_JOY_SW_C_GPIO_PIN,
    BOARD_JOY_SW_D_GPIO_PIN,
    BOARD_JOY_SW_CENTER_GPIO_PIN,
};

/* Small xorshift random numbers; the seed is stirred by frame count and input. */
static uint32_t rng_state = 2463534242u;

static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float rand_uniform(void)
{
    return (float)(rng() >> 8) * (1.0f / 16777216.0f);
}

/* Roughly unit-variance Gaussian: four uniforms summed. */
static float rand_gauss(void)
{
    return (rand_uniform() + rand_uniform() + rand_uniform() + rand_uniform() - 2.0f) * 1.732f;
}

static void random_vector(float *v)
{
    for (int i = 0; i < LATENT; ++i) {
        v[i] = rand_gauss();
    }
}

static void random_unit_vector(float *v)
{
    float n = 0.0f;
    random_vector(v);
    for (int i = 0; i < LATENT; ++i) {
        n += v[i] * v[i];
    }
    n = sqrtf(n);
    for (int i = 0; i < LATENT; ++i) {
        v[i] /= n;
    }
}

/* ---- drawing ----------------------------------------------------------- */

static inline void write_px(uint8_t *p, int bpp, uint8_t r, uint8_t g, uint8_t b)
{
    if (bpp >= 3) {
        /* the panel reads the bytes as blue, green, red */
        p[0] = b;
        p[1] = g;
        p[2] = r;
        if (bpp == 4) {
            p[3] = 255;
        }
    } else {
        uint16_t v = (uint16_t)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
        p[0] = (uint8_t)(v & 0xFF);
        p[1] = (uint8_t)(v >> 8);
    }
}

static void fill_rect(int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t *fb = display_buffer();
    int w = display_width(), h = display_height(), bpp = display_bytes_per_pixel();
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > w) x1 = w;
    if (y1 > h) y1 = h;
    for (int y = y0; y < y1; ++y) {
        uint8_t *p = fb + ((size_t)y * (size_t)w + (size_t)x0) * (size_t)bpp;
        for (int x = x0; x < x1; ++x, p += bpp) {
            write_px(p, bpp, r, g, b);
        }
    }
}

/* Bilinear lookup tables for scaling SRC -> side */
static uint8_t  tab_i0[MAX_SIDE];
static uint16_t tab_w[MAX_SIDE];  /* weight of the second sample, 0..256 */
static int      tab_side;

static void build_tables(int side)
{
    int scale = side / SRC;
    for (int d = 0; d < side; ++d) {
        /* source position of the centre of destination pixel d */
        int pos256 = ((2 * d + 1) * 256) / (2 * scale) - 128;  /* (d+0.5)/scale - 0.5, in 1/256 */
        if (pos256 < 0) pos256 = 0;
        if (pos256 > (SRC - 1) * 256) pos256 = (SRC - 1) * 256;
        int i0 = pos256 >> 8;
        tab_i0[d] = (uint8_t)(i0 > SRC - 2 ? SRC - 2 : i0);
        tab_w[d] = (uint16_t)(pos256 - ((int)tab_i0[d] << 8));
    }
    tab_side = side;
}

static void draw_face(const uint8_t *pix, int ox, int oy, int side)
{
    uint8_t *fb = display_buffer();
    int w = display_width(), bpp = display_bytes_per_pixel();
    static uint16_t row_a[MAX_SIDE], row_b[MAX_SIDE];

    if (tab_side != side) {
        build_tables(side);
    }
    for (int y = 0; y < side; ++y) {
        const uint8_t *s0 = pix + tab_i0[y] * SRC;
        const uint8_t *s1 = s0 + SRC;
        uint32_t wy = tab_w[y];
        uint8_t *p = fb + ((size_t)(oy + y) * (size_t)w + (size_t)ox) * (size_t)bpp;
        for (int x = 0; x < side; ++x, p += bpp) {
            int xi = tab_i0[x];
            uint32_t wx = tab_w[x];
            uint32_t top = (uint32_t)s0[xi] * (256 - wx) + (uint32_t)s0[xi + 1] * wx;
            uint32_t bot = (uint32_t)s1[xi] * (256 - wx) + (uint32_t)s1[xi + 1] * wx;
            uint8_t v = (uint8_t)(((top >> 8) * (256 - wy) + (bot >> 8) * wy) >> 8);
            write_px(p, bpp, v, v, v);
        }
    }
    (void)row_a;
    (void)row_b;
}

static void draw_bars(const float *z, int x0, int y0, int x1, int y1)
{
    fill_rect(x0, y0, x1, y1, 12, 12, 20);
    int w = x1 - x0, h = y1 - y0;
    int bw = w / LATENT;
    int mid = y0 + h / 2;
    int maxh = h / 2 - 8;
    fill_rect(x0, mid, x1, mid + 1, 90, 90, 110);
    for (int i = 0; i < LATENT; ++i) {
        float v = z[i] / 3.0f;
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        int bh = (int)(fabsf(v) * (float)maxh);
        int bx = x0 + i * bw + 3;
        if (v >= 0.0f) {
            fill_rect(bx, mid - bh, bx + bw - 6, mid, 0, 210, 255);
        } else {
            fill_rect(bx, mid, bx + bw - 6, mid + bh, 255, 140, 0);
        }
    }
}

/* Height of the camera view when the 4:3 frame is scaled to the screen width */
static int cam_view_h(int w)
{
    return w * camera_height() / camera_width();
}

/* Camera frame (RGB565) scaled to the full screen width at the top */
static void draw_camera(int w)
{
    uint8_t *fb = display_buffer();
    int bpp = display_bytes_per_pixel();
    const uint8_t *cam = camera_frame();
    int cw = camera_width(), ch = camera_height();
    int vh = cam_view_h(w);
    for (int y = 0; y < vh; ++y) {
        const uint8_t *row = cam + (size_t)(y * ch / vh) * (size_t)cw * 2;
        uint8_t *p = fb + (size_t)y * (size_t)w * (size_t)bpp;
        for (int x = 0; x < w; ++x, p += bpp) {
            const uint8_t *s = row + (size_t)(x * cw / w) * 2;
            uint16_t v = (uint16_t)(s[0] | (s[1] << 8));
            write_px(p, bpp, (uint8_t)((v >> 11) << 3), (uint8_t)(((v >> 5) & 0x3F) << 2),
                     (uint8_t)((v & 0x1F) << 3));
        }
    }
}

/* ---- GT911 touch over the shared I2C1 bus ----------------------------- */

/* The camera and the touch controller share I2C1. The Alif I2C driver keeps
 * only the callback of whoever initializes it first (the camera, which needs
 * it on every frame), so the pack's GT911 driver never sees its transfers
 * finish. The pack driver is still used for the reset/GPIO setup; register
 * reads are done here by polling the I2C driver's busy flag instead. */
extern ARM_DRIVER_TOUCH_SCREEN GT911;
extern ARM_DRIVER_I2C Driver_I2C1;
static uint8_t gt_addr = 0x14;
volatile int g_touch_rc = 99;          /* debugger-readable: setup result */
volatile uint32_t g_touch_count;       /* debugger-readable: touches seen */

static int gt_wait(void)
{
    uint32_t t0 = DWT->CYCCNT, limit = SystemCoreClock / 50U;  /* 20 ms */
    while (Driver_I2C1.GetStatus().busy) {
        if (DWT->CYCCNT - t0 > limit) return -1;
    }
    return 0;
}

static int gt_write(uint16_t reg, const uint8_t *data, int n)
{
    uint8_t buf[8];
    buf[0] = (uint8_t)(reg >> 8);
    buf[1] = (uint8_t)reg;
    for (int i = 0; i < n; ++i) buf[2 + i] = data[i];
    if (Driver_I2C1.MasterTransmit(gt_addr, buf, (uint32_t)(2 + n), false) != ARM_DRIVER_OK) return -1;
    return gt_wait();
}

static int gt_read(uint16_t reg, uint8_t *out, int n)
{
    if (gt_write(reg, NULL, 0) != 0) return -1;
    if (Driver_I2C1.MasterReceive(gt_addr, out, (uint32_t)n, false) != ARM_DRIVER_OK) return -1;
    return gt_wait();
}

/* Reset the GT911 and find it on the bus. 0 = found. */
static int gt_setup(void)
{
    int rc = GT911.Initialize();  /* GPIOs + I2C (already up from the camera) */
    printf("touch: GT911 Initialize -> %d\n", rc);
    rc = GT911.PowerControl(ARM_POWER_FULL);  /* resets the chip; its own I2C check fails, expected */
    printf("touch: GT911 reset -> %d\n", rc);
    static const uint8_t addrs[2] = {0x14, 0x5D};
    for (int i = 0; i < 2; ++i) {
        uint8_t zero = 0, id[4] = {0};
        gt_addr = addrs[i];
        gt_write(0x8040, &zero, 1);      /* command: read coordinates */
        int r = gt_read(0x8140, id, 4);  /* product id, "911" */
        printf("touch: addr 0x%02X -> %d id '%c%c%c'\n", gt_addr, r, id[0], id[1], id[2]);
        if (r == 0 && id[0] == '9') return 0;
    }
    return -1;
}

/* Non-blocking. 1 = finger down at (*x, *y), 0 = no touch, -1 = I2C error. */
static int gt_poll(int *x, int *y)
{
    uint8_t st = 0, p[4], zero = 0;
    if (gt_read(0x814E, &st, 1) != 0) return -1;
    if (!(st & 0x80)) return 0;  /* no new data yet */
    int n = st & 0x0F, got = 0;
    if (n > 0 && gt_read(0x8150, p, 4) == 0) {
        *x = p[0] | (p[1] << 8);
        *y = p[2] | (p[3] << 8);
        got = 1;
    }
    gt_write(0x814E, &zero, 1);  /* clear the ready flag */
    return got;
}

/* ---- LED flash -------------------------------------------------------- */

extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB0_R_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB0_G_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB0_B_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB1_R_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB1_G_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB1_B_GPIO_PORT);

static ARM_DRIVER_GPIO *const led_drv[6] = {
    &ARM_Driver_GPIO_(BOARD_LEDRGB0_R_GPIO_PORT), &ARM_Driver_GPIO_(BOARD_LEDRGB0_G_GPIO_PORT),
    &ARM_Driver_GPIO_(BOARD_LEDRGB0_B_GPIO_PORT), &ARM_Driver_GPIO_(BOARD_LEDRGB1_R_GPIO_PORT),
    &ARM_Driver_GPIO_(BOARD_LEDRGB1_G_GPIO_PORT), &ARM_Driver_GPIO_(BOARD_LEDRGB1_B_GPIO_PORT),
};
static const uint8_t led_pin_no[6] = {
    BOARD_LEDRGB0_R_GPIO_PIN, BOARD_LEDRGB0_G_GPIO_PIN, BOARD_LEDRGB0_B_GPIO_PIN,
    BOARD_LEDRGB1_R_GPIO_PIN, BOARD_LEDRGB1_G_GPIO_PIN, BOARD_LEDRGB1_B_GPIO_PIN,
};

static void leds_init(void)
{
    for (int i = 0; i < 6; ++i) {
        led_drv[i]->Initialize(led_pin_no[i], NULL);
        led_drv[i]->PowerControl(led_pin_no[i], ARM_POWER_FULL);
        led_drv[i]->SetDirection(led_pin_no[i], GPIO_PIN_DIRECTION_OUTPUT);
        led_drv[i]->SetValue(led_pin_no[i], GPIO_PIN_OUTPUT_STATE_LOW);
    }
}

/* Both RGB LEDs set to a colour (each channel on/off); only writes on change */
static void leds_rgb(int r, int g, int b)
{
    static int last = -1;
    int want = (r ? 4 : 0) | (g ? 2 : 0) | (b ? 1 : 0);
    if (want == last) return;
    last = want;
    const int on[6] = {r, g, b, r, g, b};
    for (int i = 0; i < 6; ++i) {
        led_drv[i]->SetValue(led_pin_no[i], on[i] ? GPIO_PIN_OUTPUT_STATE_HIGH : GPIO_PIN_OUTPUT_STATE_LOW);
    }
}

/* ---- camera motion (hand wave in front of the lens) -------------------- */

#define MOT_GX 32  /* sample grid over the camera frame */
#define MOT_GY 24

/* Fraction (0..1) of grid cells whose brightness changed a lot since the
 * previous frame. A hand passing the lens changes many cells at once. */
static float camera_motion(void)
{
    static uint8_t prev[MOT_GX * MOT_GY];
    static int have_prev;
    const uint8_t *cam = camera_frame();
    int cw = camera_width(), ch = camera_height();
    int changed = 0;
    for (int gy = 0; gy < MOT_GY; ++gy) {
        const uint8_t *row = cam + (size_t)((gy * ch / MOT_GY) + ch / (2 * MOT_GY)) * (size_t)cw * 2;
        for (int gx = 0; gx < MOT_GX; ++gx) {
            const uint8_t *s = row + (size_t)((gx * cw / MOT_GX) + cw / (2 * MOT_GX)) * 2;
            uint16_t v = (uint16_t)(s[0] | (s[1] << 8));
            uint8_t lum = (uint8_t)((((v >> 11) << 1) + ((v >> 5) & 0x3F) + ((v & 0x1F) << 1)) >> 1);
            int i = gy * MOT_GX + gx;
            int d = (int)lum - (int)prev[i];
            if (have_prev && (d > 12 || d < -12)) ++changed;
            prev[i] = lum;
        }
    }
    have_prev = 1;
    return (float)changed / (float)(MOT_GX * MOT_GY);
}

/* ---- microphone loudness graph ---------------------------------------- */

#define GRAPH_N 120  /* history columns, newest on the right */

static float graph_hist[GRAPH_N];

/* Scrolling bar graph of mic loudness in dB, green -> yellow -> red */
static void draw_mic_graph(int x0, int y0, int x1, int y1)
{
    fill_rect(x0, y0, x1, y1, 12, 12, 20);
    int w = x1 - x0, h = y1 - y0;
    int bw = w / GRAPH_N;
    if (bw < 1) bw = 1;
    for (int i = 0; i < GRAPH_N; ++i) {
        float rms = graph_hist[i];
        /* The PDM samples are small (quiet room: RMS ~1-10), so show 0..40 dB */
        float db = rms > 1.0f ? 20.0f * log10f(rms) : 0.0f;
        float t = db / 40.0f;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        int bh = (int)(t * (float)(h - 4));
        uint8_t r = (uint8_t)(t < 0.5f ? 510.0f * t : 255.0f);
        uint8_t g = (uint8_t)(t < 0.5f ? 220.0f : 220.0f * (1.0f - t) * 2.0f);
        int bx = x0 + i * bw;
        fill_rect(bx, y1 - 2 - bh, bx + bw - 1, y1 - 2, r, g, 40);
    }
}

static void graph_push(float rms)
{
    memmove(graph_hist, graph_hist + 1, sizeof(graph_hist) - sizeof(graph_hist[0]));
    graph_hist[GRAPH_N - 1] = rms;
}

/* ---- main loop --------------------------------------------------------- */

void face_demo_loop(void)
{
    /* Start the LCD first: its reset pin shares GPIO port 15 with the joystick */
    int display_rc = display_init();
    int have_display = (display_rc == 0);

    for (int j = 0; j < 5; ++j) {
        JOY_GPIO->Initialize(joy_pin[j], NULL);
        JOY_GPIO->PowerControl(joy_pin[j], ARM_POWER_FULL);
        JOY_GPIO->SetDirection(joy_pin[j], GPIO_PIN_DIRECTION_INPUT);
    }

    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    int camera_rc = camera_init();
    int mic_rc = mic_init();

    leds_init();
    int touch_rc = gt_setup();
    g_touch_rc = touch_rc;
    printf("touch: ready -> %d\n", touch_rc);
    int imu_rc = imu_init();
    printf("imu: ready -> %d\n", imu_rc);
    uint32_t flash = 0, imu_flash = 0, cam_flash = 0;
    float imu_motion = 0.0f, cam_motion = 0.0f;
    int have_camera = (camera_rc == 0);

    int w = display_width(), h = display_height();
    int side, face_x, face_y;
    int portrait = (w < h);
    if (have_camera) {
        /* camera on top (scaled to the full width), AI face centred below it */
        side = 256;
        face_x = (w - side) / 2;
        face_y = cam_view_h(w) + 12;
    } else {
        side = (w < h ? w : h) / SRC * SRC;  /* face is a square, a multiple of 32 */
        if (side > MAX_SIDE) side = MAX_SIDE;
        face_x = portrait ? (w - side) / 2 : 0;
        face_y = 0;
    }

    if (have_display) {
        fill_rect(0, 0, w, h, 12, 12, 20);
        display_flush();
    }

    static float z[LATENT], target[LATENT], dir_a[LATENT], dir_b[LATENT];
    static uint8_t pix[PIXELS];
    uint32_t was_centre = 0, idle = 0, frame = 0;
    int cam_rc = 0;

    uint32_t t0 = DWT->CYCCNT;

    random_unit_vector(dir_a);
    random_unit_vector(dir_b);
    random_vector(target);

    printf("face_demo: running (joystick A/B/C/D morph, CENTER new face)\n");

    for (;;) {
        uint32_t pressed[5];
        for (int j = 0; j < 5; ++j) {
            uint32_t level = 1;
            JOY_GPIO->GetValue(joy_pin[j], &level);
            pressed[j] = (level == 0U);
        }

        if (pressed[0] || pressed[1] || pressed[2] || pressed[3]) {
            float s_a = (pressed[0] ? STEP : 0.0f) - (pressed[1] ? STEP : 0.0f);
            float s_b = (pressed[2] ? STEP : 0.0f) - (pressed[3] ? STEP : 0.0f);
            for (int i = 0; i < LATENT; ++i) {
                z[i] += s_a * dir_a[i] + s_b * dir_b[i];
            }
            idle = 0;
        }
        if (pressed[4] && !was_centre) {
            rng_state ^= frame * 2654435761u;  /* stir in the timing of the press */
            random_vector(target);             /* glide to a brand-new face */
            idle = IDLE_AUTO;
        }
        was_centre = pressed[4];

        if (idle >= IDLE_AUTO) {
            float dist2 = 0.0f;
            for (int i = 0; i < LATENT; ++i) {
                float d = target[i] - z[i];
                z[i] += AUTO_RATE * d;
                dist2 += d * d;
            }
            if (dist2 < 0.5f) {
                random_vector(target);
            }
        } else {
            ++idle;
        }

        for (int i = 0; i < LATENT; ++i) {
            if (z[i] > 3.0f) z[i] = 3.0f;
            if (z[i] < -3.0f) z[i] = -3.0f;
        }

        float level = 0.0f;
        int new_blocks = 0;
        if (touch_rc == 0) {
            int tx = 0, ty = 0;
            if (gt_poll(&tx, &ty) == 1) {
                ++g_touch_count;
                if (flash == 0) {
                    printf("touch: x=%d y=%d\n", tx, ty);
                }
                flash = 6;  /* frames the LEDs stay on */
                if (have_display) {
                    fill_rect(tx - 10, ty - 10, tx + 10, ty + 10, 255, 255, 255);
                }
            } else if (flash) {
                --flash;
            }
        }

        /* Board moved / tapped (BMI323 IMU) -> green */
        if (imu_rc == 0 && imu_poll(&imu_motion, NULL, NULL, NULL) == 0) {
            if (imu_motion > IMU_MOTION_TH) {
                if (imu_flash == 0) printf("imu: motion %d\n", (int)imu_motion);
                imu_flash = 6;
            } else if (imu_flash) {
                --imu_flash;
            }
        }

        while (mic_rc == 0 && mic_poll(&level) == 1) {
            graph_push(level);
            ++new_blocks;
        }
        if (have_camera) {
            cam_rc = camera_capture();
            if (cam_rc >= 0 && have_display) {
                draw_camera(w);
            }
            /* Hand waved in front of the camera -> blue */
            if (cam_rc >= 0) {
                cam_motion = camera_motion();
                if (cam_motion > CAM_MOTION_TH && frame > 10) {
                    if (cam_flash == 0) printf("camera: motion %d%%\n", (int)(cam_motion * 100.0f));
                    cam_flash = 6;
                } else if (cam_flash) {
                    --cam_flash;
                }
            }
        }

        /* LEDs: touch = white, board motion = green, hand at camera = blue */
        if (flash) {
            leds_rgb(1, 1, 1);
        } else if (imu_flash) {
            leds_rgb(0, 1, 0);
        } else if (cam_flash) {
            leds_rgb(0, 0, 1);
        } else {
            leds_rgb(0, 0, 0);
        }

        int rc = face_generate(z, pix);
        if (rc == 0 && have_display) {
            draw_face(pix, face_x, face_y, side);
            if (have_camera) {
                draw_mic_graph(0, face_y + side + 8, w, h);
            } else {
                if (portrait) {
                    draw_bars(z, 0, face_y + side, w, h);
                } else {
                    draw_bars(z, face_x + side, 0, w, h);
                }
            }
            display_flush();
        }

        uint32_t every = have_camera ? 30U : (have_display ? 60U : 2000U);
        if (++frame % every == 0) {
            uint32_t t1 = DWT->CYCCNT;
            uint32_t cycles = t1 - t0;
            t0 = t1;
            printf("face_demo: %lu frames/s (rc=%d) display=%d camera=%d cap=%d ev=0x%lx mic=%d blocks=%lu level=%d imu=%d motion=%d cam_motion=%d%%\n",
                   (unsigned long)((uint64_t)every * SystemCoreClock / (cycles ? cycles : 1)), rc,
                   display_rc, camera_rc, cam_rc, (unsigned long)camera_last_events(),
                   mic_rc, (unsigned long)mic_block_count(), (int)graph_hist[GRAPH_N - 1],
                   imu_rc, (int)imu_motion, (int)(cam_motion * 100.0f));
            g_imu_motion = imu_motion;
            g_cam_motion = cam_motion;
        }
    }
}

#endif /* BOARD_FACE_DEMO */
