/*---------------------------------------------------------------------------
 * Bunny companion for the Alif DevKit-E8 (BOARD_BUNNY_DEMO).
 *
 * The CPU runs the behaviour: motion clips (bunny_clips.c, a port of
 * model/bunny/clips.py), a small state machine (idle, wander, sleep, reactions)
 * and blending/smoothing of the 30 pose dials. Every frame the Ethos-U85
 * draws the bunny from those dials (BunnyDecoder, nn_forward in app_main.cpp).
 * The CPU scales the sprite up, places and flips it, and draws the effects
 * (zzz, sweat, hearts), the touch dot and a small mic graph on a white screen.
 *
 *   touch the bunny      pet: love + hearts       touch elsewhere  run there
 *   hold on the bunny    hatch a new-looking bunny (identity dials)
 *   shake the board      startled shake           loud sound       ears perk up
 *   no input ~20 s       lie down, sleep; any input wakes it
 *   joystick A/B/C/D     wave / yawn / hop / clap; CENTER eat
 *---------------------------------------------------------------------------*/

#ifdef BOARD_BUNNY_DEMO

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#include "board_config.h"
#include "Driver_IO.h"
#include "Driver_I2C.h"
#include "Driver_Touch_Screen.h"

#include "display.h"
#include "camera.h"
#include "mic.h"
#include "imu.h"
#include "bunny_clips.h"
#include "voice.h"
#include "se_services_port.h"
#include "Driver_WDT.h"

extern ARM_DRIVER_WDT Driver_WDT0;
extern uint32_t Image$$ARM_LIB_STACK$$ZI$$Base[];
extern uint32_t Image$$ARM_LIB_STACK$$ZI$$Limit[];
#define WATCHDOG_MS     3000    /* loop stuck this long -> the chip resets itself */
#define STACK_PAINT     0xA5A5A5A5u

/* Fill the unused stack with a pattern; stack_used() then finds the deepest use */
static void stack_paint(void)
{
    uint32_t *p = Image$$ARM_LIB_STACK$$ZI$$Base, *sp = (uint32_t *)__get_MSP();
    while (p < sp - 64) *p++ = STACK_PAINT;
}

static uint32_t stack_used(void)
{
    uint32_t *base = Image$$ARM_LIB_STACK$$ZI$$Base, *p = base;
    while (*p == STACK_PAINT) ++p;
    return (uint32_t)((uint8_t *)Image$$ARM_LIB_STACK$$ZI$$Limit - (uint8_t *)p);
}

extern int nn_forward(const float *in, int n_in, const float **out, int *n_out);

/* ---- tuning ------------------------------------------------------------ */

#define SLEEP_AFTER_S   20.0f   /* no input this long -> lie down and sleep   */
#define BLEND_S         0.45f   /* cross-fade between clips                   */
#define DIAL_SMOOTH     0.55f   /* per-frame low-pass on the dials (1 = none) */
#define WALK_PX_S       70.0f
#define RUN_PX_S        170.0f
#define RUN_IF_FARTHER  170.0f  /* px: run instead of walk                    */
#define HOLD_HATCH_S    1.2f    /* touch-and-hold on the bunny -> new bunny   */
#define IMU_WAKE_TH     25.0f   /* IMU motion that wakes him (still ~1-12)    */
#define IMU_SHAKE_TH    150.0f  /* IMU motion that counts as a shake          */
#define LOUD_DB         30.0f   /* mic level (dB of RMS) that makes ears perk */
#define SPRITE_PX       448     /* sprite size on screen (64 -> 448, 48 -> 432) */
#define EDGE_FRAC       0.15f   /* how far the (mostly empty) sprite square may leave the screen */
#define RELEASE_S       0.15f   /* no finger this long = lifted (GT911 sends gaps) */
#define COLD_START_MS   1000    /* let the panel power up before the display init */
#define GROUND_Y        690     /* sprite bottom on screen                    */
#define GRAPH_Y0        700     /* mic graph strip                            */
#define ICON_W          84      /* world-switch button, bottom-right corner   */
#define SPRITE_CAM_PX   224     /* bunny size in the camera world (~half)     */
#define KEY_WHITE       226     /* camera world: sprite pixels brighter than this are see-through */
#define AR_AXIS         1       /* gyro axis (0 x, 1 y, 2 z) for turning the board left/right */
#define AR_SIGN         (-1.0f) /* flip if the bunny moves the wrong way when the board turns */
#define AR_PX_PER_DEG   6.0f    /* screen px per degree of turn (~80 deg camera view) */
#define AR_DEADBAND     3.0f    /* deg/s of gyro noise to ignore              */
#define AR_MAX          170.0f  /* px the anchor may drift                    */
#define AR_RETURN       0.5f    /* 1/s: drift back to the middle when still   */

volatile uint32_t g_bunny_nn_us, g_bunny_fps;  /* debugger-readable */
volatile int32_t g_force_world = -1;
volatile int32_t g_force_clip = -1;  /* debugger-writable: play this clip now (remote test) */
volatile int32_t g_force_voice = -1; /* debugger-writable: act as if this KWS_ word was heard */
volatile int32_t g_force_listen = 0; /* debugger-writable: 1 = as if the joystick centre was pressed */
volatile int32_t g_stats = 0;        /* stats overlay on/off (tap the FOOD bar); debugger-writable */
extern volatile uint32_t g_kws_us;   /* voice.c */

/* push to talk: press the joystick centre, say a word (keyword model on the NPU) */
#define VOICE_TIMEOUT_S 3.0f
#define VOICE_MIN_CONF  0.55f   /* less sure than this: "???"                        */
#define RIDE_BLOCK_S    8.0f    /* after "down" takes him off your shoulder: stay down */
volatile int32_t g_test_doodle = 0;  /* debugger-writable: 1 star, 2 house, 3 smiley, 4 fish (remote test) */  /* debugger-writable: 1 = camera world, 0 = home (acts like a tap on the button) */

#define SLOW_FRAME_MS   150     /* a frame slower than this prints its stage times */
#define IMU_EVERY       3       /* read the IMU every Nth frame                     */
#define IMU_SLOW_US     5000    /* an IMU read slower than this counts as a failure */
#define SENSOR_FAILS    3       /* consecutive failures before backing off          */

static inline uint32_t us_since(uint32_t c0)
{
    return (DWT->CYCCNT - c0) / (SystemCoreClock / 1000000u);
}

/* ---- small helpers ----------------------------------------------------- */

static uint32_t rng_state = 2463534242u;

static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float frand(float lo, float hi)
{
    return lo + (hi - lo) * (float)(rng() >> 8) * (1.0f / 16777216.0f);
}

static float smoothstep(float t)
{
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return t * t * (3.0f - 2.0f * t);
}

static inline void write_px(uint8_t *p, int bpp, uint8_t r, uint8_t g, uint8_t b)
{
    if (bpp >= 3) {
        p[0] = b;  /* the panel reads the bytes as blue, green, red */
        p[1] = g;
        p[2] = r;
        if (bpp == 4) p[3] = 255;
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
        for (int x = x0; x < x1; ++x, p += bpp) write_px(p, bpp, r, g, b);
    }
}

static void fill_circle(int cx, int cy, int rad, uint8_t r, uint8_t g, uint8_t b)
{
    for (int y = -rad; y <= rad; ++y) {
        int half = (int)sqrtf((float)(rad * rad - y * y));
        fill_rect(cx - half, cy + y, cx + half + 1, cy + y + 1, r, g, b);
    }
}

/* Thick line by stamping squares */
static void thick_line(int x0, int y0, int x1, int y1, int t, uint8_t r, uint8_t g, uint8_t b)
{
    int n = (int)(fabsf((float)(x1 - x0)) > fabsf((float)(y1 - y0)) ? fabsf((float)(x1 - x0)) : fabsf((float)(y1 - y0)));
    if (n < 1) n = 1;
    for (int i = 0; i <= n; ++i) {
        int x = x0 + (x1 - x0) * i / n, y = y0 + (y1 - y0) * i / n;
        fill_rect(x - t / 2, y - t / 2, x + (t + 1) / 2, y + (t + 1) / 2, r, g, b);
    }
}

/* Filled heart, size s px (implicit heart curve) */
static void fill_heart(int cx, int cy, int s, uint8_t r, uint8_t g, uint8_t b)
{
    for (int y = -s; y <= s; ++y) {
        for (int x = -s; x <= s; ++x) {
            float fx = (float)x / (float)s * 1.2f, fy = -(float)y / (float)s * 1.2f + 0.25f;
            float a = fx * fx + fy * fy - 1.0f;
            if (a * a * a - fx * fx * fy * fy * fy <= 0.0f) fill_rect(cx + x, cy + y, cx + x + 1, cy + y + 1, r, g, b);
        }
    }
}

/* ---- touch (GT911 on the shared I2C1; polled, see HANDOFF) ------------- */

extern ARM_DRIVER_TOUCH_SCREEN GT911;
extern ARM_DRIVER_I2C Driver_I2C1;
static uint8_t gt_addr = 0x14;

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

static int gt_setup(void)
{
    GT911.Initialize();
    GT911.PowerControl(ARM_POWER_FULL);  /* resets the chip; its own I2C check fails, expected */
    static const uint8_t addrs[2] = {0x14, 0x5D};
    for (int i = 0; i < 2; ++i) {
        uint8_t zero = 0, id[4] = {0};
        gt_addr = addrs[i];
        gt_write(0x8040, &zero, 1);
        if (gt_read(0x8140, id, 4) == 0 && id[0] == '9') return 0;
    }
    return -1;
}

static void busy_ms(uint32_t ms);

/* Touch stopped answering (bus or chip stuck): abort, clear the bus (clocks a
 * stuck data line free), reset the GT911 and find it again. */
static int gt_recover(void)
{
    Driver_I2C1.Control(ARM_I2C_ABORT_TRANSFER, 0);
    busy_ms(2);
    int bc = Driver_I2C1.Control(ARM_I2C_BUS_CLEAR, 0);
    busy_ms(2);
    GT911.PowerControl(ARM_POWER_OFF);
    int r = gt_setup();
    printf("touch: recovery (bus clear %d) -> %s\n", bc, r == 0 ? "ok" : "failed");
    return r;
}

/* 1 = new touch report with a finger down at (*x, *y), 0 = no new report,
 * 2 = new report with no finger (released), -1 = error */
static int gt_poll(int *x, int *y)
{
    uint8_t st = 0, p[4], zero = 0;
    if (gt_read(0x814E, &st, 1) != 0) return -1;
    if (!(st & 0x80)) return 0;
    int n = st & 0x0F, got = 2;
    if (n > 0 && gt_read(0x8150, p, 4) == 0) {
        *x = p[0] | (p[1] << 8);
        *y = p[2] | (p[3] << 8);
        got = 1;
    }
    gt_write(0x814E, &zero, 1);
    return got;
}

/* ---- LEDs and joystick ------------------------------------------------- */

extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB0_R_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB0_G_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB0_B_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB1_R_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB1_G_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB1_B_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(15); /* joystick */

static ARM_DRIVER_GPIO *const led_drv[6] = {
    &ARM_Driver_GPIO_(BOARD_LEDRGB0_R_GPIO_PORT), &ARM_Driver_GPIO_(BOARD_LEDRGB0_G_GPIO_PORT),
    &ARM_Driver_GPIO_(BOARD_LEDRGB0_B_GPIO_PORT), &ARM_Driver_GPIO_(BOARD_LEDRGB1_R_GPIO_PORT),
    &ARM_Driver_GPIO_(BOARD_LEDRGB1_G_GPIO_PORT), &ARM_Driver_GPIO_(BOARD_LEDRGB1_B_GPIO_PORT),
};
static const uint8_t led_pin_no[6] = {
    BOARD_LEDRGB0_R_GPIO_PIN, BOARD_LEDRGB0_G_GPIO_PIN, BOARD_LEDRGB0_B_GPIO_PIN,
    BOARD_LEDRGB1_R_GPIO_PIN, BOARD_LEDRGB1_G_GPIO_PIN, BOARD_LEDRGB1_B_GPIO_PIN,
};
static const uint8_t joy_pin[5] = {
    BOARD_JOY_SW_A_GPIO_PIN, BOARD_JOY_SW_B_GPIO_PIN, BOARD_JOY_SW_C_GPIO_PIN,
    BOARD_JOY_SW_D_GPIO_PIN, BOARD_JOY_SW_CENTER_GPIO_PIN,
};

static int led_last = -1;

static void leds_init_once(void)
{
    static int done;
    if (done) return;
    done = 1;
    for (int i = 0; i < 6; ++i) {
        led_drv[i]->Initialize(led_pin_no[i], NULL);
        led_drv[i]->PowerControl(led_pin_no[i], ARM_POWER_FULL);
        led_drv[i]->SetDirection(led_pin_no[i], GPIO_PIN_DIRECTION_OUTPUT);
        led_drv[i]->SetValue(led_pin_no[i], GPIO_PIN_OUTPUT_STATE_LOW);
    }
}

static void leds_rgb(int r, int g, int b)
{
    int want = (r ? 4 : 0) | (g ? 2 : 0) | (b ? 1 : 0);
    if (want == led_last) return;
    led_last = want;
    const int on[6] = {r, g, b, r, g, b};
    for (int i = 0; i < 6; ++i) {
        led_drv[i]->SetValue(led_pin_no[i], on[i] ? GPIO_PIN_OUTPUT_STATE_HIGH : GPIO_PIN_OUTPUT_STATE_LOW);
    }
}

/* Boot progress (called from main.c too): red = started, yellow = clocks,
 * blue = model loaded / display starting, green = running, blinking red =
 * display failed. */
void bunny_boot_led(int r, int g, int b)
{
    leds_init_once();
    leds_rgb(r, g, b);
}

static void busy_ms(uint32_t ms)
{
    uint32_t t0 = DWT->CYCCNT, n = (SystemCoreClock / 1000U) * ms;
    while (DWT->CYCCNT - t0 < n) {}
}

static void io_init(void)
{
    leds_init_once();
    for (int j = 0; j < 5; ++j) {
        ARM_Driver_GPIO_(15).Initialize(joy_pin[j], NULL);
        ARM_Driver_GPIO_(15).PowerControl(joy_pin[j], ARM_POWER_FULL);
        ARM_Driver_GPIO_(15).SetDirection(joy_pin[j], GPIO_PIN_DIRECTION_INPUT);
    }
}

/* ---- crash reporter ---------------------------------------------------- */

/* A fault prints where it happened (UART) and blinks the LEDs purple instead
 * of freezing silently. The stacked PC/LR map to source lines with addr2line. */
void bunny_fault_report(uint32_t *sp, uint32_t exc_return)
{
    printf("FAULT pc=%08lx lr=%08lx cfsr=%08lx hfsr=%08lx bfar=%08lx mmfar=%08lx exc=%08lx\n",
           (unsigned long)sp[6], (unsigned long)sp[5], (unsigned long)SCB->CFSR, (unsigned long)SCB->HFSR,
           (unsigned long)SCB->BFAR, (unsigned long)SCB->MMFAR, (unsigned long)exc_return);
    for (;;) {
        for (volatile uint32_t i = 0; i < 8000000u; ++i) {}
        for (int k = 0; k < 6; ++k) led_drv[k]->SetValue(led_pin_no[k], (k % 3 != 1) ? GPIO_PIN_OUTPUT_STATE_HIGH : GPIO_PIN_OUTPUT_STATE_LOW);
        for (volatile uint32_t i = 0; i < 8000000u; ++i) {}
        for (int k = 0; k < 6; ++k) led_drv[k]->SetValue(led_pin_no[k], GPIO_PIN_OUTPUT_STATE_LOW);
    }
}

__attribute__((naked)) void HardFault_Handler(void)
{
    __asm volatile(
        "tst lr, #4 \n"
        "ite eq \n"
        "mrseq r0, msp \n"
        "mrsne r0, psp \n"
        "mov r1, lr \n"
        "b bunny_fault_report \n");
}

/* ---- sprite: NN output -> screen --------------------------------------- */

#define MAX_NN  128
#define MAX_SPR 480

static uint8_t  src_rgb[MAX_NN * MAX_NN * 3];
static uint8_t  src_a[MAX_NN * MAX_NN];   /* alpha: 0 background, 255 bunny, in between at the edge */
static uint16_t tab_i0[MAX_SPR];
static uint16_t tab_w[MAX_SPR];  /* weight of the second sample, 0..256 */
static int      nn_side, spr;

static void build_tables(void)
{
    for (int d = 0; d < spr; ++d) {
        int pos256 = ((2 * d + 1) * 256 * nn_side) / (2 * spr) - 128;  /* (d+0.5)*S/spr - 0.5 */
        if (pos256 < 0) pos256 = 0;
        if (pos256 > (nn_side - 1) * 256) pos256 = (nn_side - 1) * 256;
        int i0 = pos256 >> 8;
        if (i0 > nn_side - 2) i0 = nn_side - 2;
        tab_i0[d] = (uint16_t)i0;
        tab_w[d] = (uint16_t)(pos256 - (i0 << 8));
    }
}

/* NN output (3 x S x S floats, 0..1) -> src_rgb */
/* Bold lines: darken the outline / eyes / mouth and thicken them by about a
 * pixel (at NN resolution, so ~4-5 px on screen). Runs on the small NN image. */
#define LINE_LUM_DARK   60.0f   /* luminance treated as fully "line"            */
#define LINE_LUM_LIGHT  150.0f  /* luminance above this is not line at all      */
#define LINE_DARKEN     0.88f   /* 0..1: how far line pixels move to LINE_RGB   */
#define LINE_THICKEN    0.4f   /* 0..1: how much a dark neighbour darkens a pixel */
#define LINE_R 24
#define LINE_G 10
#define LINE_B 36

/* Alpha matte (like a PNG): the background is the near-white area connected to
 * the image border (flood fill), so the white belly and eye highlights stay
 * solid. Pixels touching the background are the anti-aliased edge: they get
 * partial alpha and the white is taken out of their colour, so the outline
 * blends into the camera / drawings without a white halo. */
#define BG_MIN_CHANNEL  222     /* near-white: every channel at least this */
#define EDGE_DARK       60.0f   /* outline darkness used to estimate edge alpha */

static void make_alpha(void)
{
    static uint16_t stack[MAX_NN * MAX_NN];
    int s = nn_side, n = s * s, sp = 0;
    memset(src_a, 255, (size_t)n);
#define NEAR_WHITE(i) (src_rgb[(i) * 3] >= BG_MIN_CHANNEL && src_rgb[(i) * 3 + 1] >= BG_MIN_CHANNEL && src_rgb[(i) * 3 + 2] >= BG_MIN_CHANNEL)
    for (int i = 0; i < s; ++i) {  /* seed with the border */
        int e[4] = {i, (s - 1) * s + i, i * s, i * s + s - 1};
        for (int k = 0; k < 4; ++k) {
            if (src_a[e[k]] && NEAR_WHITE(e[k])) { src_a[e[k]] = 0; stack[sp++] = (uint16_t)e[k]; }
        }
    }
    while (sp > 0) {
        int i = stack[--sp], x = i % s, y = i / s;
        int nb[4] = {x > 0 ? i - 1 : -1, x < s - 1 ? i + 1 : -1, y > 0 ? i - s : -1, y < s - 1 ? i + s : -1};
        for (int k = 0; k < 4; ++k) {
            int j = nb[k];
            if (j >= 0 && src_a[j] && NEAR_WHITE(j)) { src_a[j] = 0; stack[sp++] = (uint16_t)j; }
        }
    }
#undef NEAR_WHITE
    for (int i = 0; i < n; ++i) {
        uint8_t *c = &src_rgb[i * 3];
        if (src_a[i] == 0) {  /* background: dark, so bilinear mixing never brings in white */
            c[0] = LINE_R; c[1] = LINE_G; c[2] = LINE_B;
            continue;
        }
        int x = i % s, y = i / s;
        int edge = (x > 0 && !src_a[i - 1]) || (x < s - 1 && !src_a[i + 1]) || (y > 0 && !src_a[i - s]) || (y < s - 1 && !src_a[i + s]);
        if (!edge) continue;
        int mn = c[0] < c[1] ? (c[0] < c[2] ? c[0] : c[2]) : (c[1] < c[2] ? c[1] : c[2]);
        float a = (255.0f - (float)mn) / (255.0f - EDGE_DARK);
        if (a > 1.0f) a = 1.0f;
        if (a < 0.05f) a = 0.05f;
        for (int k = 0; k < 3; ++k) {  /* un-mix the white: colour = (pixel - (1-a) * 255) / a */
            float v = ((float)c[k] - (1.0f - a) * 255.0f) / a;
            c[k] = (uint8_t)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
        }
        src_a[i] = (uint8_t)(a * 255.0f);
    }
}

static void convert_output(const float *out)
{
    static float dark[MAX_NN * MAX_NN];
    int n = nn_side * nn_side, s = nn_side;
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = out[c * n + i] * 255.0f + 0.5f;
            src_rgb[i * 3 + c] = (uint8_t)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
        }
        float lum = 0.30f * src_rgb[i * 3] + 0.59f * src_rgb[i * 3 + 1] + 0.11f * src_rgb[i * 3 + 2];
        float d = (LINE_LUM_LIGHT - lum) / (LINE_LUM_LIGHT - LINE_LUM_DARK);
        if (src_rgb[i * 3] > src_rgb[i * 3 + 2] + 12) d = 0.0f;  /* reddish (mouth inside, tongue): not a line */
        dark[i] = d < 0.0f ? 0.0f : (d > 1.0f ? 1.0f : d);
    }
    for (int y = 0; y < s; ++y) {
        for (int x = 0; x < s; ++x) {
            int i = y * s + x;
            float d = dark[i], nb = 0.0f;
            if (x > 0 && dark[i - 1] > nb) nb = dark[i - 1];
            if (x < s - 1 && dark[i + 1] > nb) nb = dark[i + 1];
            if (y > 0 && dark[i - s] > nb) nb = dark[i - s];
            if (y < s - 1 && dark[i + s] > nb) nb = dark[i + s];
            nb *= LINE_THICKEN;
            if (nb > d) d = nb;
            if (d <= 0.0f) continue;
            float k = LINE_DARKEN * d;
            uint8_t *p = &src_rgb[i * 3];
            p[0] = (uint8_t)(p[0] + (LINE_R - p[0]) * k);
            p[1] = (uint8_t)(p[1] + (LINE_G - p[1]) * k);
            p[2] = (uint8_t)(p[2] + (LINE_B - p[2]) * k);
        }
    }
    make_alpha();
}

/* Bilinear scale src_rgb to spr x spr at (ox, oy), mirrored if flip */
/* ---- ink layer: a 1-bit mask of the finger drawing ------------------------ */
/* Everything that paints background (the sprite's white, clearing, the camera)
 * puts ink back where the mask is set, so the drawing never flickers. */
#define INK_W   480
#define INK_H   700                 /* = GRAPH_Y0 */
#define INK_R   70                  /* ink colour */
#define INK_G   40
#define INK_B   95
static uint8_t ink_mask[INK_W * INK_H / 8];
static int ink_any;

static inline int ink_at(int x, int y)
{
    if (!ink_any || x < 0 || x >= INK_W || y < 0 || y >= INK_H) return 0;
    int i = y * INK_W + x;
    return (ink_mask[i >> 3] >> (i & 7)) & 1;
}

/* Background (white, or ink where drawn) over a rectangle */
static void fill_bg(int x0, int y0, int x1, int y1)
{
    if (!ink_any) {
        fill_rect(x0, y0, x1, y1, 255, 255, 255);
        return;
    }
    uint8_t *fb = display_buffer();
    int w = display_width(), h = display_height(), bpp = display_bytes_per_pixel();
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > w) x1 = w;
    if (y1 > h) y1 = h;
    for (int y = y0; y < y1; ++y) {
        uint8_t *p = fb + ((size_t)y * (size_t)w + (size_t)x0) * (size_t)bpp;
        for (int x = x0; x < x1; ++x, p += bpp) {
            if (ink_at(x, y)) write_px(p, bpp, INK_R, INK_G, INK_B);
            else write_px(p, bpp, 255, 255, 255);
        }
    }
}

/* One screen row of the sprite: row y (0..spr-1) of the scaled sprite at (ox, oy) */
static void draw_sprite_row(int ox, int oy, int flip, int key, int y)
{
    uint8_t *fb = display_buffer();
    int w = display_width(), h = display_height(), bpp = display_bytes_per_pixel();
    static uint8_t vrow[MAX_NN * 3];
    int sy = oy + y;
    if (y < 0 || y >= spr || sy < 0 || sy >= h) return;
    int x_lo = ox < 0 ? -ox : 0, x_hi = ox + spr > w ? w - ox : spr;
    /* vertical blend once per row, then only the horizontal blend per pixel */
    const uint8_t *r0 = src_rgb + (size_t)tab_i0[y] * nn_side * 3;
    const uint8_t *r1 = r0 + nn_side * 3;
    uint32_t wy = tab_w[y], wy0 = 256 - wy;
    for (int k = 0; k < nn_side * 3; ++k) {
        vrow[k] = (uint8_t)((r0[k] * wy0 + r1[k] * wy) >> 8);
    }
    static uint8_t arow[MAX_NN];
    const uint8_t *a0 = src_a + (size_t)tab_i0[y] * nn_side, *a1 = a0 + nn_side;
    int any = 0;
    for (int k = 0; k < nn_side; ++k) {
        arow[k] = (uint8_t)((a0[k] * wy0 + a1[k] * wy) >> 8);
        any |= arow[k];
    }
    if (key && !any) return;  /* a fully transparent row */
    uint8_t *row = fb + (size_t)sy * (size_t)w * (size_t)bpp;
    for (int x = x_lo; x < x_hi; ++x) {
        int src_x = flip ? spr - 1 - x : x;
        const uint8_t *v = vrow + tab_i0[src_x] * 3;
        uint32_t wx = tab_w[src_x], wx0 = 256 - wx;
        uint8_t r = (uint8_t)((v[0] * wx0 + v[3] * wx) >> 8);
        uint8_t g = (uint8_t)((v[1] * wx0 + v[4] * wx) >> 8);
        uint8_t b = (uint8_t)((v[2] * wx0 + v[5] * wx) >> 8);
        if (key) {  /* alpha blend over what is already there (camera, drawings, white) */
            uint32_t al = (uint32_t)((arow[tab_i0[src_x]] * wx0 + arow[tab_i0[src_x] + 1] * wx) >> 8);
            if (al < 8) continue;
            uint8_t *d = row + (size_t)(ox + x) * (size_t)bpp;
            if (al < 248 && bpp >= 3) {  /* panel bytes are B, G, R */
                r = (uint8_t)(d[2] + (((int)r - d[2]) * (int)al >> 8));
                g = (uint8_t)(d[1] + (((int)g - d[1]) * (int)al >> 8));
                b = (uint8_t)(d[0] + (((int)b - d[0]) * (int)al >> 8));
            }
            write_px(d, bpp, r, g, b);
            continue;
        }
        if (r > KEY_WHITE && g > KEY_WHITE && b > KEY_WHITE && ink_at(ox + x, sy)) {  /* the drawing shows through the white */
            write_px(row + (size_t)(ox + x) * (size_t)bpp, bpp, INK_R, INK_G, INK_B);
            continue;
        }
        write_px(row + (size_t)(ox + x) * (size_t)bpp, bpp, r, g, b);
    }
}

static void draw_sprite(int ox, int oy, int flip, int key)
{
    for (int y = 0; y < spr; ++y) draw_sprite_row(ox, oy, flip, key, y);
}

/* Camera world background: the 640x480 frame scaled to fill the screen above
 * the graph strip (height-fit, centre-cropped left and right). */
/* The MT9M114 picture is dim and washed out indoors: brighten it toward a
 * target average (gain follows the last frame) and push the colours away from
 * grey. Only what is shown changes; BlazeFace still sees the raw frame. */
#define CAM_SAT         460     /* saturation x256 (460 = 1.8x)         */
#define CAM_TARGET_Y    105     /* average brightness to aim for        */
#define CAM_MAX_GAIN    3.0f
#define CAM_GAMMA       0.85f   /* < 1 lifts the shadows                */
#define CAM_WB          0.25f   /* white balance: 0 off, 1 full grey-world (removes the room light's tint) */

static void draw_camera_bg(int w, int y1, int sox, int soy, int sflip)
{
    static uint16_t cam_x[480];
    static int ready;
    static uint8_t lut[3][256];
    static float gain = 1.5f, wb[3] = {1.0f, 1.0f, 1.0f};
    static uint32_t ysum, yn, csum[3];
    if (yn > 0) {  /* new gain and white balance from the last frame */
        float mean = (float)ysum / (float)yn;
        float g = (float)CAM_TARGET_Y / (mean > 1.0f ? mean : 1.0f);  /* mean = raw brightness */
        g = g < 1.0f ? 1.0f : (g > CAM_MAX_GAIN ? CAM_MAX_GAIN : g);
        gain += (g - gain) * 0.3f;
        for (int c = 0; c < 3; ++c) {  /* grey world: pull each channel's average toward the brightness */
            float cm = (float)csum[c] / (float)yn;
            float t = powf((mean + 1.0f) / (cm + 1.0f), CAM_WB);
            t = t < 0.7f ? 0.7f : (t > 1.5f ? 1.5f : t);
            wb[c] += (t - wb[c]) * 0.3f;
            for (int v = 0; v < 256; ++v) {
                float x = (float)v * gain * wb[c] / 255.0f;
                float o = 255.0f * powf(x > 1.0f ? 1.0f : x, CAM_GAMMA);
                lut[c][v] = (uint8_t)(o > 255.0f ? 255.0f : o);
            }
        }
        ysum = yn = 0;
        csum[0] = csum[1] = csum[2] = 0;
    } else if (!ready) {
        for (int v = 0; v < 256; ++v) lut[0][v] = lut[1][v] = lut[2][v] = (uint8_t)v;
    }
    const uint8_t *cam = camera_frame();
    int cw = camera_width(), ch = camera_height();
    if (!ready) {
        int vis_w = (w * ch) / y1;  /* source columns that fit the screen width */
        if (vis_w > cw) vis_w = cw;
        for (int x = 0; x < w && x < 480; ++x) cam_x[x] = (uint16_t)((cw - vis_w) / 2 + x * vis_w / w);
        ready = 1;
    }
    uint8_t *fb = display_buffer();
    int bpp = display_bytes_per_pixel();
    for (int y = 0; y < y1; ++y) {
        const uint8_t *src = cam + (size_t)(y * ch / y1) * (size_t)cw * 2;
        uint8_t *p = fb + (size_t)y * (size_t)w * (size_t)bpp;
        for (int x = 0; x < w; ++x, p += bpp) {
            const uint8_t *q = src + (size_t)cam_x[x] * 2;
            uint16_t v = (uint16_t)(q[0] | (q[1] << 8));
            if (ink_at(x, y)) { write_px(p, bpp, INK_R, INK_G, INK_B); continue; }
            int r = (v >> 11) << 3, g = ((v >> 5) & 0x3F) << 2, b = (v & 0x1F) << 3;
            int yy = (77 * r + 150 * g + 29 * b) >> 8;
            if (((x | y) & 7) == 0) {  /* raw brightness and colour, sampled */
                ysum += (uint32_t)yy; ++yn;
                csum[0] += (uint32_t)r; csum[1] += (uint32_t)g; csum[2] += (uint32_t)b;
            }
            r = yy + (((r - yy) * CAM_SAT) >> 8);
            g = yy + (((g - yy) * CAM_SAT) >> 8);
            b = yy + (((b - yy) * CAM_SAT) >> 8);
            r = r < 0 ? 0 : (r > 255 ? 255 : r);
            g = g < 0 ? 0 : (g > 255 ? 255 : g);
            b = b < 0 ? 0 : (b > 255 ? 255 : b);
            write_px(p, bpp, lut[0][r], lut[1][g], lut[2][b]);
        }
        draw_sprite_row(sox, soy, sflip, 1, y - soy);
    }
}

typedef struct { int x0, y0, x1, y1; } rect_t;

/* Clear (to white) the parts of old not covered by cur */
static void clear_uncovered(rect_t old, rect_t cur)
{
    if (old.x1 <= old.x0) return;
    if (cur.x1 <= old.x0 || cur.x0 >= old.x1 || cur.y1 <= old.y0 || cur.y0 >= old.y1) {
        fill_bg(old.x0, old.y0, old.x1, old.y1);
        return;
    }
    if (old.y0 < cur.y0) fill_bg(old.x0, old.y0, old.x1, cur.y0);
    if (old.y1 > cur.y1) fill_bg(old.x0, cur.y1, old.x1, old.y1);
    int ya = old.y0 > cur.y0 ? old.y0 : cur.y0, yb = old.y1 < cur.y1 ? old.y1 : cur.y1;
    if (old.x0 < cur.x0) fill_bg(old.x0, ya, cur.x0, yb);
    if (old.x1 > cur.x1) fill_bg(cur.x1, ya, old.x1, yb);
}

/* Effects at rig coordinates (512 canvas, see bunny_rig3.bunny_svg) */
static void draw_effects(const bunny_pose_t *p, int ox, int oy, int flip)
{
    float k = (float)spr / 512.0f;
#define FX(X) (ox + (int)((flip ? 512.0f - (X) : (X)) * k))
#define FY(Y) (oy + (int)((Y) * k))
    if (p->zzz > 0.3f) {
        float lie = p->d[D_LIE];
        float sx = -60.0f * lie, sy = 120.0f * lie;
        int t = spr / 60 + 2;
        struct { float x, y, s; } z[3] = {{378, 82, 40}, {434, 38, 28}, {478, 4, 20}};
        for (int i = 0; i < 3; ++i) {
            float y0 = z[i].y + sy, s = z[i].s;
            /* left edge of the letter, mirrored with the bunny; the Z itself is never mirrored */
            float l = flip ? 512.0f - (z[i].x + sx) - s : z[i].x + sx;
            int xa = ox + (int)(l * k), xb = ox + (int)((l + s) * k);
            if (xa < 2) { xb += 2 - xa; xa = 2; }  /* keep the letter on the screen */
            if (xb > 477) { xa -= xb - 477; xb = 477; }
            thick_line(xa, FY(y0), xb, FY(y0), t, 70, 40, 95);
            thick_line(xb, FY(y0), xa, FY(y0 + s), t, 70, 40, 95);
            thick_line(xa, FY(y0 + s), xb, FY(y0 + s), t, 70, 40, 95);
        }
    }
    if (p->sweat > 0.3f) {
        fill_circle(FX(392), FY(200), (int)(14 * k) + 1, 159, 216, 255);
        thick_line(FX(392), FY(172), FX(392), FY(196), (int)(10 * k) + 1, 159, 216, 255);
    }
    if (p->hearts > 0.3f) {
        float s = p->hearts;
        fill_heart(FX(400), FY(120), (int)(26 * k * s) + 2, 240, 100, 143);
        fill_heart(FX(120), FY(90), (int)(20 * k * s) + 2, 240, 100, 143);
        fill_heart(FX(430), FY(190), (int)(16 * k * s) + 2, 240, 100, 143);
    }
#undef FX
#undef FY
}

/* ---- tiny 5x7 font (speech bubbles) ---------------------------------------- */

static const uint8_t font5x7[42][7] = {
    {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}, {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E},  /* A B */
    {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}, {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E},  /* C D */
    {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}, {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10},  /* E F */
    {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}, {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},  /* G H */
    {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}, {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C},  /* I J */
    {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}, {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F},  /* K L */
    {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}, {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11},  /* M N */
    {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10},  /* O P */
    {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}, {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11},  /* Q R */
    {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}, {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},  /* S T */
    {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04},  /* U V */
    {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A}, {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11},  /* W X */
    {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}, {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F},  /* Y Z */
    {0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04}, {0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04},  /* ? ! */
    {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}, {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},  /* 0 1 */
    {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}, {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E},  /* 2 3 */
    {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},  /* 4 5 */
    {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}, {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},  /* 6 7 */
    {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}, {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},  /* 8 9 */
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}, {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00},  /* . : */
    {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10}, {0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03},  /* / % */
};

/* Upper-case text (A-Z, space, ?, !) with s x s pixel dots; returns the width */
static int draw_text(int x, int y, const char *t, int sc, uint8_t r, uint8_t g, uint8_t b)
{
    int x0 = x;
    for (; *t; ++t) {
        char c = *t;
        if (c >= 'a' && c <= 'z') c = (char)(c - 32);
        int gi = (c >= 'A' && c <= 'Z') ? c - 'A' : (c >= '0' && c <= '9') ? 28 + c - '0'
               : c == '?' ? 26 : c == '!' ? 27 : c == '.' ? 38 : c == ':' ? 39 : c == '/' ? 40 : c == '%' ? 41 : -1;
        if (gi >= 0) {
            for (int row = 0; row < 7; ++row) {
                for (int col = 0; col < 5; ++col) {
                    if (font5x7[gi][row] & (0x10 >> col)) {
                        fill_rect(x + col * sc, y + row * sc, x + (col + 1) * sc, y + (row + 1) * sc, r, g, b);
                    }
                }
            }
        }
        x += 6 * sc;
    }
    return x - x0;
}

/* ---- doodles: finger drawing -> doodle classifier on the NPU --------------- */

extern int doodle_classify(const float *in, float *scores, int n);

#define INK_MAX       1500    /* stored stroke points (x = -1 marks a pen lift) */
#define INK_THICK     9       /* px, on screen                                  */
#define DRAG_START_PX 18      /* a touch that moves this far draws instead of tapping */
#define INK_IDLE_S    0.9f    /* finger lifted this long -> recognise the drawing */
#define BUBBLE_S      3.0f    /* speech bubble (and the drawing) stays this long */
#define DOODLE_MIN_P  0.45f   /* below this confidence the bunny is confused     */

static const char *const doodle_names[8] = {"STAR", "SUN", "HOUSE", "CAT", "FLOWER", "FISH", "SMILEY FACE", "LIGHTNING"};
/* how the bunny reacts to each class (same order) */
static const bunny_clip_t doodle_react[8] = {CLIP_CLAP, CLIP_STRETCH, CLIP_HOP, CLIP_SHAKE, CLIP_LOVE, CLIP_EAT, CLIP_WAVE, CLIP_ANGRY};

static int16_t ink_x[INK_MAX], ink_y[INK_MAX];
static int ink_n;
static int ink_x0 = 9999, ink_y0 = 9999, ink_x1 = -1, ink_y1 = -1;  /* bounding box */

/* Thick segment into the ink mask only */
static int g_dbuf;  /* double buffered: every frame is redrawn, no ink mask needed */

static void mask_seg(int x0, int y0, int x1, int y1, int thick)
{
    if (g_dbuf) return;
    int n = abs(x1 - x0) > abs(y1 - y0) ? abs(x1 - x0) : abs(y1 - y0);
    if (n < 1) n = 1;
    int r = thick / 2;
    for (int k = 0; k <= n; ++k) {
        int cx = x0 + (x1 - x0) * k / n, cy = y0 + (y1 - y0) * k / n;
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                int x = cx + dx, y = cy + dy;
                if (dx * dx + dy * dy > r * r + 1 || x < 0 || x >= INK_W || y < 0 || y >= INK_H) continue;
                int i = y * INK_W + x;
                ink_mask[i >> 3] |= (uint8_t)(1u << (i & 7));
            }
        }
    }
    ink_any = 1;
}

/* Thick segment into the ink mask (and onto the screen once) */
static void ink_stamp(int x0, int y0, int x1, int y1)
{
    int n = abs(x1 - x0) > abs(y1 - y0) ? abs(x1 - x0) : abs(y1 - y0);
    if (n < 1) n = 1;
    (void)n;
    mask_seg(x0, y0, x1, y1, INK_THICK);
    thick_line(x0, y0, x1, y1, INK_THICK, INK_R, INK_G, INK_B);
}

static void ink_add(int x, int y)
{
    if (ink_n >= INK_MAX) return;
    if (x >= 0 && ink_n > 0 && ink_x[ink_n - 1] >= 0) ink_stamp(ink_x[ink_n - 1], ink_y[ink_n - 1], x, y);
    ink_x[ink_n] = (int16_t)x;
    ink_y[ink_n] = (int16_t)y;
    ++ink_n;
    if (x >= 0) {
        if (x < ink_x0) ink_x0 = x;
        if (y < ink_y0) ink_y0 = y;
        if (x > ink_x1) ink_x1 = x;
        if (y > ink_y1) ink_y1 = y;
    }
}

static void ink_draw(void)
{
    for (int i = 1; i < ink_n; ++i) {
        if (ink_x[i] < 0 || ink_x[i - 1] < 0) continue;
        thick_line(ink_x[i - 1], ink_y[i - 1], ink_x[i], ink_y[i], INK_THICK, 70, 40, 95);
    }
}

static void world_rebuild(int cam_mode, int ex0, int ey0, int ex1, int ey1);

/* Forget the live drawing (the world objects stay) */
static void ink_clear(int cam_mode)
{
    int x0 = ink_x0 - INK_THICK, y0 = ink_y0 - INK_THICK, x1 = ink_x1 + INK_THICK, y1 = ink_y1 + INK_THICK;
    int had = ink_x1 >= 0;
    ink_n = 0;
    ink_x0 = ink_y0 = 9999;
    ink_x1 = ink_y1 = -1;
    if (had) world_rebuild(cam_mode, x0, y0, x1, y1);
    else world_rebuild(cam_mode, 0, 0, -1, -1);
}

/* Strokes -> 28x28 like Quick, Draw! (white ink on black, fitted to ~24 px and
 * centred, ~1.5 px lines): drawn at 4x resolution, then averaged down. */
static void ink_rasterize(float *out28)
{
    static uint8_t big[112 * 112];
    memset(big, 0, sizeof(big));
    int bw = ink_x1 - ink_x0, bh = ink_y1 - ink_y0;
    int span = bw > bh ? bw : bh;
    if (span < 1) span = 1;
    float sc = 96.0f / (float)span;
    float offx = 8.0f + (96.0f - (float)bw * sc) / 2.0f, offy = 8.0f + (96.0f - (float)bh * sc) / 2.0f;
    for (int i = 1; i < ink_n; ++i) {
        if (ink_x[i] < 0 || ink_x[i - 1] < 0) continue;
        float ax = offx + (float)(ink_x[i - 1] - ink_x0) * sc, ay = offy + (float)(ink_y[i - 1] - ink_y0) * sc;
        float bx2 = offx + (float)(ink_x[i] - ink_x0) * sc, by2 = offy + (float)(ink_y[i] - ink_y0) * sc;
        int n = (int)(fabsf(bx2 - ax) > fabsf(by2 - ay) ? fabsf(bx2 - ax) : fabsf(by2 - ay)) + 1;
        for (int k = 0; k <= n; ++k) {
            int cx = (int)(ax + (bx2 - ax) * (float)k / (float)n), cy = (int)(ay + (by2 - ay) * (float)k / (float)n);
            for (int dy = -3; dy <= 3; ++dy) {
                for (int dx = -3; dx <= 3; ++dx) {
                    int px = cx + dx, py = cy + dy;
                    if (px >= 0 && px < 112 && py >= 0 && py < 112 && dx * dx + dy * dy <= 10) big[py * 112 + px] = 255;
                }
            }
        }
    }
    float mx = 0.0f;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            int sum = 0;
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 4; ++i) sum += big[(y * 4 + j) * 112 + x * 4 + i];
            }
            float v = (float)sum / (16.0f * 255.0f);
            out28[y * 28 + x] = v;
            if (v > mx) mx = v;
        }
    }
    if (mx > 0.0f) {
        for (int i = 0; i < 28 * 28; ++i) out28[i] /= mx;
    }
}

/* Speech bubble at the top of the screen */
/* A speech bubble; '|' starts a new line (lines are centred) */
static void draw_bubble(const char *text, int w)
{
    int sc = 4, lines = 1, longest = 0, n = 0;
    for (const char *c = text;; ++c) {
        if (*c == '|' || *c == 0) {
            if (n > longest) longest = n;
            n = 0;
            if (*c == 0) break;
            ++lines;
        } else {
            ++n;
        }
    }
    int lh = 7 * sc + 10, tw = longest * 6 * sc - sc;
    int x0 = (w - tw) / 2 - 16, y0 = 24, x1 = x0 + tw + 32, y1 = y0 + lines * lh + 18;
    fill_rect(x0, y0, x1, y1, 120, 80, 200);
    fill_rect(x0 + 4, y0 + 4, x1 - 4, y1 - 4, 255, 255, 255);
    const char *t = text;
    for (int l = 0; l < lines; ++l) {
        char buf[32];
        int k = 0;
        while (*t && *t != '|' && k < (int)sizeof(buf) - 1) buf[k++] = *t++;
        buf[k] = 0;
        if (*t == '|') ++t;
        draw_text((w - (k * 6 * sc - sc)) / 2, y0 + 14 + l * lh, buf, sc, 70, 40, 95);
    }
}

/* ---- the bunny's world: drawings that come to life ------------------------ */
/* A recognised drawing becomes an object made of its own strokes: it shrinks
 * and moves to its place (floor or sky) and the bunny reacts to it. Objects
 * live in the same ink mask as the live drawing, so they never flicker. */

#define OBJ_MAX     4
#define OBJ_PTS     320
#define OBJ_THICK   6
#define OBJ_PEN_UP  (-32768)
#define FLOOR_Y     (GROUND_Y - 22)  /* where floor objects stand */
#define HUNGER_S    180.0f           /* full -> empty                       */
#define HUNGRY_AT   0.3f
#define EAT_S       2.8f

enum { C_STAR, C_SUN, C_HOUSE, C_CAT, C_FLOWER, C_FISH, C_SMILEY, C_LIGHTNING };

typedef struct {
    int active, cls, n;
    uint32_t age;
    int16_t px[OBJ_PTS], py[OBJ_PTS];  /* strokes relative to the drawing centre */
    float x, y, s;                     /* current centre and scale */
    float x0, y0, s0, x1, y1, s1;      /* animation from -> to */
    float t, dur;                      /* animation time */
    float life;                        /* seconds left, < 0 = stays */
    float bite;                        /* 0..1 eaten (fish) */
    int bx0, by0, bx1, by1;            /* last stamped box */
} wobj_t;

static wobj_t objs[OBJ_MAX];
static uint32_t obj_age;

static void obj_box(wobj_t *o, int *x0, int *y0, int *x1, int *y1)
{
    int a = 9999, b = 9999, c = -9999, d = -9999;
    for (int i = 0; i < o->n; ++i) {
        if (o->px[i] == OBJ_PEN_UP) continue;
        int x = (int)(o->x + o->px[i] * o->s), y = (int)(o->y + o->py[i] * o->s);
        if (x < a) a = x;
        if (y < b) b = y;
        if (x > c) c = x;
        if (y > d) d = y;
    }
    *x0 = a - OBJ_THICK; *y0 = b - OBJ_THICK; *x1 = c + OBJ_THICK; *y1 = d + OBJ_THICK;
}

static void obj_stamp(wobj_t *o)
{
    int upto = o->n - (int)(o->bite * (float)o->n);  /* a fish loses points as it is eaten */
    for (int i = 1; i < upto; ++i) {
        if (o->px[i] == OBJ_PEN_UP || o->px[i - 1] == OBJ_PEN_UP) continue;
        mask_seg((int)(o->x + o->px[i - 1] * o->s), (int)(o->y + o->py[i - 1] * o->s),
                 (int)(o->x + o->px[i] * o->s), (int)(o->y + o->py[i] * o->s), OBJ_THICK);
    }
    obj_box(o, &o->bx0, &o->by0, &o->bx1, &o->by1);
}

/* Rebuild the mask (live drawing + objects) and repaint what changed */
static void world_rebuild(int cam_mode, int ex0, int ey0, int ex1, int ey1)
{
    int ux0 = ex0, uy0 = ey0, ux1 = ex1, uy1 = ey1;
#define GROW(a, b, c, d) do { if ((c) > (a)) { if (ux1 < ux0) { ux0 = (a); uy0 = (b); ux1 = (c); uy1 = (d); } \
        else { if ((a) < ux0) ux0 = (a); if ((b) < uy0) uy0 = (b); if ((c) > ux1) ux1 = (c); if ((d) > uy1) uy1 = (d); } } } while (0)
    for (int k = 0; k < OBJ_MAX; ++k) if (objs[k].bx1 > objs[k].bx0) GROW(objs[k].bx0, objs[k].by0, objs[k].bx1, objs[k].by1);
    memset(ink_mask, 0, sizeof(ink_mask));
    ink_any = 0;
    for (int i = 1; i < ink_n; ++i) {
        if (ink_x[i] < 0 || ink_x[i - 1] < 0) continue;
        mask_seg(ink_x[i - 1], ink_y[i - 1], ink_x[i], ink_y[i], INK_THICK);
    }
    for (int k = 0; k < OBJ_MAX; ++k) {
        if (objs[k].active) {
            obj_stamp(&objs[k]);
            GROW(objs[k].bx0, objs[k].by0, objs[k].bx1, objs[k].by1);
        } else {
            objs[k].bx0 = objs[k].bx1 = 0;
        }
    }
#undef GROW
    if (!cam_mode && !g_dbuf && ux1 > ux0) fill_bg(ux0, uy0, ux1, uy1);
}

/* Double buffering: the live drawing and the world objects as plain lines
 * (no background), drawn every frame */
static void world_draw(void)
{
    for (int i = 1; i < ink_n; ++i) {
        if (ink_x[i] < 0 || ink_x[i - 1] < 0) continue;
        thick_line(ink_x[i - 1], ink_y[i - 1], ink_x[i], ink_y[i], INK_THICK, INK_R, INK_G, INK_B);
    }
    for (int k = 0; k < OBJ_MAX; ++k) {
        wobj_t *o = &objs[k];
        if (!o->active) continue;
        int upto = o->n - (int)(o->bite * (float)o->n);
        for (int i = 1; i < upto; ++i) {
            if (o->px[i] == OBJ_PEN_UP || o->px[i - 1] == OBJ_PEN_UP) continue;
            thick_line((int)(o->x + o->px[i - 1] * o->s), (int)(o->y + o->py[i - 1] * o->s),
                       (int)(o->x + o->px[i] * o->s), (int)(o->y + o->py[i] * o->s), OBJ_THICK, INK_R, INK_G, INK_B);
        }
    }
}

/* White background above the graph strip, fast (white = 0xFF in every byte) */
static void clear_world(int w)
{
    memset(display_buffer(), 0xFF, (size_t)w * (size_t)GRAPH_Y0 * (size_t)display_bytes_per_pixel());
}

/* Turn the live drawing into an object of class cls that moves to (tx, ty)
 * with size tsize px; life < 0 = stays. Returns its index. */
static int obj_from_ink(int cls, float tx, float ty, float tsize, float life)
{
    int k = -1;
    for (int i = 0; i < OBJ_MAX; ++i) if (objs[i].active && (cls == C_SUN || cls == C_HOUSE) && objs[i].cls == cls) k = i;  /* one sun, one house */
    for (int i = 0; i < OBJ_MAX && k < 0; ++i) if (!objs[i].active) k = i;
    if (k < 0) {  /* full: replace the oldest */
        k = 0;
        for (int i = 1; i < OBJ_MAX; ++i) if (objs[i].age < objs[k].age) k = i;
    }
    wobj_t *o = &objs[k];
    float cx = (ink_x0 + ink_x1) / 2.0f, cy = (ink_y0 + ink_y1) / 2.0f;
    int span = (ink_x1 - ink_x0) > (ink_y1 - ink_y0) ? (ink_x1 - ink_x0) : (ink_y1 - ink_y0);
    if (span < 8) span = 8;
    int step = (ink_n + OBJ_PTS - 1) / OBJ_PTS, n = 0;
    if (step < 1) step = 1;
    for (int i = 0; i < ink_n && n < OBJ_PTS; ++i) {
        if (ink_x[i] < 0) {  /* keep pen lifts */
            if (n > 0 && o->px[n - 1] != OBJ_PEN_UP) { o->px[n] = OBJ_PEN_UP; o->py[n] = 0; ++n; }
            continue;
        }
        if (i % step && i + 1 < ink_n && ink_x[i + 1] >= 0) continue;
        o->px[n] = (int16_t)(ink_x[i] - cx);
        o->py[n] = (int16_t)(ink_y[i] - cy);
        ++n;
    }
    o->n = n;
    o->active = 1;
    o->cls = cls;
    o->age = ++obj_age;
    o->x = o->x0 = cx;
    o->y = o->y0 = cy;
    o->s = o->s0 = 1.0f;
    o->x1 = tx;
    o->s1 = tsize / (float)span;
    o->y1 = ty;
    o->t = 0.0f;
    o->dur = 0.8f;
    o->life = life;
    o->bite = 0.0f;
    return k;
}

/* Object standing on the floor: its bottom on FLOOR_Y */
static float floor_y_for(float tsize, int span_h, int span)
{
    float h = (float)span_h * tsize / (float)(span < 8 ? 8 : span);
    return (float)FLOOR_Y - h / 2.0f;
}

/* Animate objects; 1 if anything moved or vanished */
static int world_update(float dt)
{
    int dirty = 0;
    for (int k = 0; k < OBJ_MAX; ++k) {
        wobj_t *o = &objs[k];
        if (!o->active) continue;
        if (o->t < o->dur) {
            o->t += dt;
            float u = smoothstep(o->t / o->dur);
            o->x = o->x0 + (o->x1 - o->x0) * u;
            o->y = o->y0 + (o->y1 - o->y0) * u;
            o->s = o->s0 + (o->s1 - o->s0) * u;
            dirty = 1;
        }
        if (o->life >= 0.0f) {
            o->life -= dt;
            if (o->life < 0.0f) { o->active = 0; dirty = 1; }
        }
    }
    return dirty;
}

static int world_find(int cls)
{
    for (int k = 0; k < OBJ_MAX; ++k) if (objs[k].active && objs[k].cls == cls) return k;
    return -1;
}

static void world_clear_all(int cam_mode)
{
    for (int k = 0; k < OBJ_MAX; ++k) objs[k].active = 0;
    ink_clear(cam_mode);
}

/* Food meter, top-left */
/* Live performance overlay (tap the FOOD bar to toggle): what runs where and how fast */
static void draw_stats(int cam, uint32_t fps, uint32_t bunny_us, uint32_t face_us, uint32_t doodle_us)
{
    char l[7][32];
    snprintf(l[0], sizeof(l[0]), "%lu FPS  M55 400MHZ", (unsigned long)fps);
    snprintf(l[1], sizeof(l[1]), "U85 BUNNY  %lu.%lu MS", (unsigned long)(bunny_us / 1000), (unsigned long)(bunny_us / 100 % 10));
    if (cam) snprintf(l[2], sizeof(l[2]), "U85 FACE   %lu.%lu MS", (unsigned long)(face_us / 1000), (unsigned long)(face_us / 100 % 10));
    else snprintf(l[2], sizeof(l[2]), "U85 FACE   CAMERA OFF");
    if (g_kws_us) snprintf(l[3], sizeof(l[3]), "U85 VOICE  %lu.%02lu MS", (unsigned long)(g_kws_us / 1000), (unsigned long)(g_kws_us / 10 % 100));
    else snprintf(l[3], sizeof(l[3]), "U85 VOICE  WAITING");
    if (doodle_us) snprintf(l[4], sizeof(l[4]), "U85 DOODLE %lu.%02lu MS", (unsigned long)(doodle_us / 1000), (unsigned long)(doodle_us / 10 % 100));
    else snprintf(l[4], sizeof(l[4]), "U85 DOODLE WAITING");
    snprintf(l[5], sizeof(l[5]), "SRAM 7.9/8 MB  4 MODELS");
    snprintf(l[6], sizeof(l[6]), "ALL ON DEVICE");
    int x0 = 8, y0 = 124, lh = 18, sc = 2;  /* below the speech bubble */
    fill_rect(x0, y0, x0 + 24 * 6 * sc + 12, y0 + 7 * lh + 10, 120, 80, 200);
    fill_rect(x0 + 2, y0 + 2, x0 + 24 * 6 * sc + 10, y0 + 7 * lh + 8, 255, 255, 255);
    for (int i = 0; i < 7; ++i) draw_text(x0 + 8, y0 + 8 + i * lh, l[i], sc, 70, 40, 95);
}

static void draw_hunger(float hunger)
{
    fill_bg(4, 2, 176, 24);
    draw_text(8, 6, "FOOD", 2, 70, 40, 95);
    fill_rect(62, 6, 172, 20, 70, 40, 95);
    fill_rect(64, 8, 170, 18, 255, 255, 255);
    int fw = (int)(hunger * 106.0f);
    uint8_t r = (uint8_t)(hunger < 0.5f ? 230 : 230 - (hunger - 0.5f) * 2.0f * 150.0f);
    uint8_t g = (uint8_t)(hunger < 0.5f ? 60 + hunger * 2.0f * 140.0f : 200);
    if (fw > 0) fill_rect(64, 8, 64 + fw, 18, r, g, 90);
}

/* Eraser button, bottom-left */
static void draw_eraser(int h)
{
    fill_rect(0, GRAPH_Y0, ICON_W, h, 120, 80, 200);
    int cx = ICON_W / 2, cy = GRAPH_Y0 + (h - GRAPH_Y0) / 2;
    for (int i = 0; i < 18; ++i) {  /* slanted eraser block */
        fill_rect(cx - 22 + i, cy + 12 - i, cx + 2 + i, cy + 14 - i, 255, 255, 255);
    }
    for (int i = 0; i < 8; ++i) fill_rect(cx - 22 + i, cy + 12 - i, cx - 12 + i, cy + 14 - i, 240, 150, 180);
    fill_rect(cx - 26, cy + 18, cx + 26, cy + 20, 255, 255, 255);
}

/* ---- face -> shoulder (camera world) --------------------------------------- */
/* BlazeFace on the NPU finds the largest face in a 128x128 crop of the camera;
 * the shoulder is about a face-width to the side and 1.25 face-heights down. */

#include "face_anchors.h"

extern float *face_input(void);
extern int face_detect(const float **out);

#define FACE_MIN_SCORE  0.70f   /* BlazeFace front model uses 0.75          */
#define FACE_EVERY      2       /* run the detector every Nth camera frame  */
#define FACE_STABLE_S   1.0f    /* face seen this long -> jump on the shoulder */
#define FACE_LOST_S     2.0f    /* face gone this long -> hop back down     */
#define FACE_BOX        0       /* draw a thin box around the face (debug)  */
#define CROP_X0         80      /* centre 480x480 square of the 640x480 frame */
#define CROP_SIZE       480

/* BlazeFace keypoints: 0 right eye, 1 left eye, 2 nose, 3 mouth, 4 right ear, 5 left ear
 * ("right/left" of the person; on screen the person's right eye is on the left) */
typedef struct { int ok; float x, y, w, h, score; float kx[6], ky[6]; float roll; } face_t;

#define SHOULDER_DX     0.95f   /* shoulder: face widths out from the face centre  */
#define SHOULDER_DY     1.05f   /*           face heights down                     */
#define FACE_JUMP_MAX   1.5f    /* a jump of more than this (face widths) between
                                 * detections is a glitch unless it repeats        */

/* Camera crop -> the detector's 3x128x128 input (-1..1) */
static void face_fill_input(void)
{
    float *in = face_input();
    const uint8_t *cam = camera_frame();
    int cw = camera_width();
    for (int y = 0; y < 128; ++y) {
        const uint8_t *row = cam + (size_t)(y * CROP_SIZE / 128) * (size_t)cw * 2;
        for (int x = 0; x < 128; ++x) {
            const uint8_t *q = row + (size_t)(CROP_X0 + x * CROP_SIZE / 128) * 2;
            uint16_t v = (uint16_t)(q[0] | (q[1] << 8));
            int i = y * 128 + x;
            in[i] = (float)((v >> 11) << 3) / 127.5f - 1.0f;
            in[16384 + i] = (float)(((v >> 5) & 0x3F) << 2) / 127.5f - 1.0f;
            in[32768 + i] = (float)((v & 0x1F) << 3) / 127.5f - 1.0f;
        }
    }
}

/* Detector output -> the best face, in screen pixels (camera shown height-fit,
 * centre-cropped to the screen width over y 0..y1) */
static face_t face_decode(const float *raw, int w, int y1)
{
    face_t f;
    memset(&f, 0, sizeof(f));
    int best = -1;
    float bs = FACE_MIN_SCORE;
    for (int a = 0; a < FACE_N_ANCHORS; ++a) {
        float lg = raw[a * 17 + 16];
        if (lg < -50.0f) continue;
        float sc = 1.0f / (1.0f + expf(-lg));
        if (sc > bs) { bs = sc; best = a; }
    }
    if (best < 0) return f;
    /* blend the overlapping detections (BlazeFace's weighted suppression) */
    const float *bb = &raw[best * 17];
    float bx = bb[0] / 128.0f * face_anchors[best][2] + face_anchors[best][0];
    float by = bb[1] / 128.0f * face_anchors[best][3] + face_anchors[best][1];
    float wsum = 0.0f, cx = 0.0f, cy = 0.0f, ww = 0.0f, hh = 0.0f, kx[6] = {0}, ky[6] = {0};
    for (int a = 0; a < FACE_N_ANCHORS; ++a) {
        float lg = raw[a * 17 + 16];
        if (lg < 0.85f) continue;  /* score > 0.7 */
        float sc = 1.0f / (1.0f + expf(-lg));
        const float *r = &raw[a * 17];
        float x = r[0] / 128.0f * face_anchors[a][2] + face_anchors[a][0];
        float y = r[1] / 128.0f * face_anchors[a][3] + face_anchors[a][1];
        float fw = r[2] / 128.0f * face_anchors[a][2], fh = r[3] / 128.0f * face_anchors[a][3];
        if (fabsf(x - bx) > fw * 0.5f || fabsf(y - by) > fh * 0.5f) continue;  /* not the same face */
        wsum += sc; cx += sc * x; cy += sc * y; ww += sc * fw; hh += sc * fh;
        for (int k = 0; k < 6; ++k) {
            kx[k] += sc * (r[4 + 2 * k] / 128.0f * face_anchors[a][2] + face_anchors[a][0]);
            ky[k] += sc * (r[5 + 2 * k] / 128.0f * face_anchors[a][3] + face_anchors[a][1]);
        }
    }
    if (wsum <= 0.0f) return f;
    cx /= wsum; cy /= wsum; ww /= wsum; hh /= wsum;
    /* crop fractions -> camera pixels -> screen pixels */
    int cw = camera_width(), ch = camera_height();
    float vis_w = (float)(w * ch) / (float)y1;          /* camera columns shown */
    float cam_x0 = ((float)cw - vis_w) / 2.0f;
    float sx = (float)w / vis_w, sy = (float)y1 / (float)ch;
    f.ok = 1;
    f.score = bs;
    f.x = ((float)CROP_X0 + cx * (float)CROP_SIZE - cam_x0) * sx;
    f.y = (cy * (float)CROP_SIZE) * sy;
    f.w = ww * (float)CROP_SIZE * sx;
    f.h = hh * (float)CROP_SIZE * sy;
    for (int k = 0; k < 6; ++k) {
        f.kx[k] = ((float)CROP_X0 + kx[k] / wsum * (float)CROP_SIZE - cam_x0) * sx;
        f.ky[k] = (ky[k] / wsum * (float)CROP_SIZE) * sy;
    }
    /* head roll from the eyes (degrees, + = tilted toward the screen's right) */
    f.roll = atan2f(f.ky[1] - f.ky[0], f.kx[1] - f.kx[0]) * 57.2958f;
    return f;
}

static void draw_box(int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b)
{
    fill_rect(x0, y0, x1, y0 + 2, r, g, b);
    fill_rect(x0, y1 - 2, x1, y1, r, g, b);
    fill_rect(x0, y0, x0 + 2, y1, r, g, b);
    fill_rect(x1 - 2, y0, x1, y1, r, g, b);
}

/* ---- mic graph --------------------------------------------------------- */

#define GRAPH_N 80
static float graph_db[GRAPH_N];

/* World-switch button: a camera in the bunny's home, a house in the camera world */
static void draw_icon(int cam_mode, int w, int h)
{
    int x0 = w - ICON_W, y0 = GRAPH_Y0;
    fill_rect(x0, y0, w, h, 120, 80, 200);
    int cx = x0 + ICON_W / 2, cy = y0 + (h - y0) / 2;
    if (!cam_mode) {
        fill_rect(cx - 24, cy - 14, cx + 24, cy + 18, 255, 255, 255);  /* body */
        fill_rect(cx - 10, cy - 21, cx + 6, cy - 14, 255, 255, 255);   /* top */
        fill_circle(cx, cy + 2, 12, 120, 80, 200);                     /* lens ring */
        fill_circle(cx, cy + 2, 7, 255, 255, 255);
        fill_circle(cx, cy + 2, 4, 70, 40, 95);
    } else {
        for (int i = 0; i < 20; ++i) fill_rect(cx - i - 2, cy - 22 + i, cx + i + 2, cy - 21 + i, 255, 255, 255);  /* roof */
        fill_rect(cx - 17, cy - 2, cx + 17, cy + 22, 255, 255, 255);  /* walls */
        fill_rect(cx - 5, cy + 8, cx + 5, cy + 22, 120, 80, 200);     /* door */
    }
}

static void draw_graph(int x0, int x1, int h)
{
    fill_rect(x0, GRAPH_Y0, x1, h, 244, 240, 250);
    int bw = (x1 - x0) / GRAPH_N, gh = h - GRAPH_Y0 - 8;
    if (bw < 1) bw = 1;
    for (int i = 0; i < GRAPH_N; ++i) {
        float t = graph_db[i] / 40.0f;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        int bh = (int)(t * (float)gh);
        fill_rect(x0 + i * bw + 1, h - 4 - bh, x0 + i * bw + bw - 1, h - 4, 160, 120, 230);
    }
}

/* ---- behaviour --------------------------------------------------------- */

typedef enum { ST_IDLE, ST_WANDER, ST_ACTION, ST_LIE, ST_SLEEP } state_t;

static const char *const state_names[] = {"idle", "wander", "action", "lie", "sleep"};

static bunny_clip_t clip = CLIP_IDLE;
static float        clip_t;      /* time in the current clip */
static float        blend_t = BLEND_S;
static float        from_d[BUNNY_NDIAL], from_dx, from_dy;
static bunny_pose_t shown;       /* smoothed pose that is drawn */
static float        identity[4] = {0.0f, 1.0f, 128.0f, 1.0f};  /* hue, sat, ear_len, head_scale */

static void set_clip(bunny_clip_t c)
{
    if (c == clip) return;
    printf("clip: %s\n", bunny_clip_name(c));
    memcpy(from_d, shown.d, sizeof(from_d));
    from_dx = shown.dx;
    from_dy = shown.dy;
    clip = c;
    clip_t = 0.0f;
    blend_t = 0.0f;
}

static void hatch(void)
{
    static const int ids[4] = {D_HUE, D_SAT, D_EAR_LEN, D_HEAD_SCALE};
    for (int i = 0; i < 4; ++i) {
        float lo, hi;
        bunny_dial_range(ids[i], &lo, &hi);
        identity[i] = frand(lo, hi);
    }
    printf("bunny: hatched hue=%d sat=%d ear=%d head=%d\n", (int)(identity[0] * 1000), (int)(identity[1] * 100),
           (int)identity[2], (int)(identity[3] * 100));
}

/* ---- personality: needs, mood, choices, learning -------------------------- */
/* Instead of a timer, the bunny weighs what he could do against his needs
 * (energy, fun, affection, food) and what he has learned you like, and picks
 * at random in proportion: you cannot predict his next move, but it fits his
 * mood. Petting him right after something he did on his own makes him do it
 * more; shaking the board right after makes him do it less. */

typedef enum { A_WANDER, A_LOOK, A_SIT, A_HOP, A_CLAP, A_WAVE, A_STRETCH, A_YAWN, A_POUT, A_COUNT } act_t;
static const char *const act_more[A_COUNT] = {"MORE WALKS!", "MORE LOOKS!", "MORE SITS!", "MORE HOPS!", "MORE CLAPS!",
                                              "MORE WAVES!", "MORE STRETCH!", "MORE YAWNS!", "MORE POUTS!"};
static const char *const act_name[A_COUNT] = {"walk", "look", "sit", "hop", "clap", "wave", "stretch", "yawn", "pout"};
static float pref[A_COUNT] = {1, 1, 1, 1, 1, 1, 1, 1, 1};  /* learned: how much you like each */

#define LEARN_WINDOW_S  4.0f    /* pet within this after an action = you liked it */
#define CHOICE_TEMP     0.22f   /* lower = more predictable                       */

static float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

/* how attractive each action is right now (before learning) */
static void act_utility(float *u, float energy, float fun, float affection, float hunger)
{
    float valence = clamp01(0.4f * affection + 0.35f * fun + 0.25f * hunger);
    u[A_WANDER]  = 0.45f + 0.7f * (1.0f - fun) * energy;
    u[A_LOOK]    = 0.35f + 0.3f * (1.0f - energy);
    u[A_SIT]     = 0.20f + 0.8f * (1.0f - energy);
    u[A_HOP]     = 0.05f + 1.0f * valence * energy;
    u[A_CLAP]    = 0.03f + 0.7f * valence * energy;
    u[A_WAVE]    = 0.03f + 1.1f * (1.0f - affection) * energy;   /* attention seeking */
    u[A_STRETCH] = 0.05f + 0.4f * (1.0f - energy);
    u[A_YAWN]    = 0.02f + 1.3f * (1.0f - energy) * (1.0f - energy);
    u[A_POUT]    = 0.01f + 1.2f * clamp01(0.45f - valence) * 2.0f;
}

static act_t choose_action(float energy, float fun, float affection, float hunger)
{
    float u[A_COUNT], w[A_COUNT], sum = 0.0f, mx = 0.0f;
    act_utility(u, energy, fun, affection, hunger);
    for (int i = 0; i < A_COUNT; ++i) { u[i] *= pref[i]; if (u[i] > mx) mx = u[i]; }
    for (int i = 0; i < A_COUNT; ++i) { w[i] = expf((u[i] - mx) / CHOICE_TEMP); sum += w[i]; }
    float r = frand(0.0f, sum);
    for (int i = 0; i < A_COUNT; ++i) { r -= w[i]; if (r <= 0.0f) return (act_t)i; }
    return A_LOOK;
}

/* ---- main loop --------------------------------------------------------- */

void bunny_demo_loop(void)
{
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    bunny_boot_led(0, 0, 1);  /* blue: model loaded, starting the display */
    /* After a USB power-on the display init hung (a RESET always worked):
     * give the panel and its supplies time to come up first. */
    busy_ms(COLD_START_MS);

    /* The LCD's DPHY needs the SE's 20 MHz clock; after some power cycles the
     * first request fails, so re-request it and retry the display a few times. */
    int display_rc = -1;
    for (int attempt = 0; attempt < 5 && display_rc != 0; ++attempt) {
        if (attempt > 0) {
            uint32_t err = 0;
            SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_CLK_20M, true, &err);
            busy_ms(300);
        }
        display_rc = display_init();
        printf("bunny: display_init attempt %d -> %d\n", attempt + 1, display_rc);
    }
    /* The camera opens I2C1 first so its callback owns the bus (touch polls) */
    int camera_rc = camera_init();
    int mic_rc = mic_init();
    io_init();
    int touch_rc = gt_setup();
    if (touch_rc != 0) touch_rc = gt_recover();  /* not found: clear the bus, reset, try once more */
    int imu_rc = imu_init();
    printf("bunny: display=%d camera=%d mic=%d touch=%d imu=%d\n", display_rc, camera_rc, mic_rc, touch_rc, imu_rc);
    if (display_rc != 0) {
        printf("bunny: no display, stopping\n");
        for (;;) {  /* blinking red: display failed */
            bunny_boot_led(1, 0, 0);
            busy_ms(150);
            bunny_boot_led(0, 0, 0);
            busy_ms(150);
        }
    }
    bunny_boot_led(0, 1, 0);  /* green: display up */
    display_enable_double_buffer();
    g_dbuf = 1;

    int w = display_width(), h = display_height();
    fill_rect(0, 0, w, h, 255, 255, 255);

    /* First run tells the output size (3 x S x S) */
    float nn_in[BUNNY_NDIAL];
    const float *out = NULL;
    int n_out = 0;
    bunny_clip(CLIP_IDLE, 0.0f, 1.0f, &shown);
    bunny_normalize(shown.d, nn_in);
    if (nn_forward(nn_in, BUNNY_NDIAL, &out, &n_out) != 0) {
        printf("bunny: model is not the BunnyDecoder (nn_forward failed)\n");
        for (;;) {}
    }
    nn_side = (int)(sqrtf((float)n_out / 3.0f) + 0.5f);
    if (nn_side > MAX_NN || nn_side < 2) {
        printf("bunny: unexpected output size %d\n", n_out);
        for (;;) {}
    }
    int spr_home = SPRITE_PX;  /* bilinear scaling handles any ratio */
    if (spr_home > MAX_SPR) spr_home = MAX_SPR;
    spr = spr_home;
    build_tables();
    int cam_mode = 0;
    float ar_off = 0.0f;  /* camera world: px the bunny is shifted to stay put when the board turns */
    printf("bunny: NN draws %dx%d, shown at %dx%d\n", nn_side, nn_side, spr, spr);

    state_t st = ST_IDLE;
    /* The bunny fills only the middle of its square, so let the square go
     * partly off-screen: the whole width stays usable for wandering. */
    float x_min = -EDGE_FRAC * (float)spr, x_max = (float)(w - spr) + EDGE_FRAC * (float)spr;
    float bx = (float)(w - spr) / 2.0f, target_x = bx;
    int facing_left = 0;
    float state_t_s = 0.0f, next_decision = 3.0f, idle_s = 0.0f, action_len = 0.0f;
    float perk = 0.0f, hold_s = 0.0f;
    int touching = 0, touch_x = 0, touch_y = 0, hatched_this_hold = 0;
    float since_finger = 1.0f;  /* s since the last finger-down report */
    float touch_pause = 0.0f, imu_last = 0.0f;
    /* doodles */
    int tap_candidate = 0, drawing = 0, was_touching = 0, start_x = 0, start_y = 0;
    float ink_idle = 0.0f, bubble_t = 0.0f;
    char bubble_text[24] = "";
    /* the world */
    float hunger = 0.8f, hungry_nag = 0.0f;
    int pending_obj = -1, eat_obj = -1, go_sleep = 0, flee_after = 0;
    bunny_clip_t pending_clip = CLIP_COUNT;
    /* personality */
    float energy = 0.9f, fun = 0.6f, affection = 0.5f, thought_cd = 25.0f, spont_t = 99.0f;
    int last_spont = -1;
    /* camera world: face -> shoulder ride */
    face_t face = {0, 0, 0, 0, 0, 0};
    float face_seen_s = 0.0f, face_lost_s = 0.0f, sh_x = 0.0f, sh_y = 0.0f, ride_t = 0.0f;
    float jump_x0 = 0.0f, jump_y0 = 0.0f;
    int ride = 0;          /* 0 on the floor, 1 jumping up, 2 riding, 3 hopping down */
    int vstate = 0;        /* voice: 0 off, 1 listening */
    float vrec_t = 0.0f, ride_block_s = 0.0f;
    int voice_act = -1;    /* heard: a KWS_ word */
    int voice_jump = 0;    /* "up" / "down" in the camera world: onto / off the shoulder */
    int ride_side = 1;     /* which shoulder (screen side: -1 left, +1 right), locked at the jump */
    int glitches = 0;      /* detections rejected as jumps */
    face_t fs;             /* smoothed face */
    memset(&fs, 0, sizeof(fs));
    float gaze = 0.0f;     /* 0..1: how much he looks at your face */
    uint32_t cam_frames = 0, face_us = 0;
    int touch_fails = 0, imu_fails = 0, touch_bad = 0;
    uint32_t was_pressed = 0, frame = 0, nn_us_acc = 0, bunny_npu_us = 0, doodle_us = 0;
    rect_t prev = {0, 0, 0, 0}, prev_dot = {0, 0, 0, 0};
    uint32_t t_prev = DWT->CYCCNT, t_stat = t_prev;

    stack_paint();
    int wdt_rc = Driver_WDT0.Initialize(WATCHDOG_MS);
    if (wdt_rc == ARM_DRIVER_OK) wdt_rc = Driver_WDT0.PowerControl(ARM_POWER_FULL);
    if (wdt_rc == ARM_DRIVER_OK) wdt_rc = Driver_WDT0.Control(ARM_WATCHDOG_UNLOCK, 0);
    if (wdt_rc == ARM_DRIVER_OK) wdt_rc = Driver_WDT0.Start();
    printf("bunny: watchdog %d ms -> %d\n", WATCHDOG_MS, wdt_rc);
    printf("bunny: running\n");
    snprintf(bubble_text, sizeof(bubble_text), "HEY!|MY NAME IS ALFY!");  /* hello on start-up, with a wave */
    bubble_t = 4.0f;
    g_force_clip = CLIP_WAVE;
    for (;;) {
        /* Keep the cycle counter running: all timing and timeouts use it, and a
         * debugger disconnect can switch the trace unit (and the counter) off. */
        DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
        DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
        uint32_t now = DWT->CYCCNT;
        float dt = (float)(now - t_prev) / (float)SystemCoreClock;
        t_prev = now;
        if (dt > 0.1f) dt = 0.1f;
        rng_state ^= now;

        /* ---- inputs ---- */
        int any_input = 0, poked = 0;
        uint32_t st0 = DWT->CYCCNT, us_touch = 0, us_imu = 0, us_mic = 0, us_nn = 0, us_draw = 0;
        if (touch_pause > 0.0f) touch_pause -= dt;
        if (touch_rc == 0 && touch_pause <= 0.0f) {
            int tx, ty, r = gt_poll(&tx, &ty);
            if (r < 0) {
                if (++touch_fails >= SENSOR_FAILS) {  /* bus trouble: pause touch for 1 s */
                    printf("touch: %d failed reads, pausing 1 s\n", touch_fails);
                    touch_pause = 1.0f;
                    touch_fails = 0;
                    if (++touch_bad >= 2) {  /* still failing after a pause: recover */
                        gt_recover();
                        touch_bad = 0;
                    }
                }
            } else {
                touch_fails = 0;
                touch_bad = 0;
            }
            if (r == 1 && (tx >= w || ty >= h)) {
                r = 0;  /* ignore coordinates off the panel */
            }
            since_finger += dt;
            if (r == 1) {
                since_finger = 0.0f;
                if (!touching) {
                    poked = 1;
                    printf("touch: down x=%d y=%d frame %lu\n", tx, ty, (unsigned long)frame);
                }
                touching = 1;
                touch_x = tx;
                touch_y = ty;
                any_input = 1;
            } else if (touching && since_finger > RELEASE_S) {
                touching = 0;  /* finger lifted */
            }
        }
        if (g_force_world >= 0) {  /* remote test: press the button */
            if (g_force_world != cam_mode) {
                poked = 1;
                touch_x = w - 10;
                touch_y = h - 10;
            }
            g_force_world = -1;
        }
        if (poked && touch_x < 180 && touch_y < 30) {  /* the FOOD bar: stats overlay on / off */
            poked = 0;
            g_stats = !g_stats;
            printf("stats overlay %s\n", g_stats ? "on" : "off");
        }
        /* world-switch button */
        if (poked && touch_x >= w - ICON_W && touch_y >= GRAPH_Y0) {
            poked = 0;
            if (camera_rc == 0 || cam_mode) {
                cam_mode = !cam_mode;
                /* Entering: start a snapshot. Leaving: do NOT stop the camera. A full
                 * Driver_CPI.Stop() puts the sensor in standby and stops the CSI link,
                 * and after that the camera froze on its first frame the next time
                 * (Abdul). A snapshot still in flight just finishes; it is picked up
                 * (and replaced) on the next visit. */
                if (cam_mode) camera_capture_start();
                float centre = bx + (float)spr / 2.0f;
                spr = cam_mode ? SPRITE_CAM_PX : spr_home;
                build_tables();
                bx = centre - (float)spr / 2.0f;
                target_x = bx;
                x_min = -EDGE_FRAC * (float)spr;
                x_max = (float)(w - spr) + EDGE_FRAC * (float)spr;
                ar_off = 0.0f;
                if (!cam_mode) fill_rect(0, 0, w, GRAPH_Y0, 255, 255, 255);
                prev = (rect_t){0, 0, 0, 0};
                prev_dot = (rect_t){0, 0, 0, 0};
                printf("bunny: world -> %s\n", cam_mode ? "camera" : "home");
                st = ST_ACTION;  /* a happy hop to arrive */
                set_clip(CLIP_HOP);
                action_len = bunny_clip_duration(CLIP_HOP);
                state_t_s = 0.0f;
                idle_s = 0.0f;
            }
        }
        if (touching && touch_x >= w - ICON_W && touch_y >= GRAPH_Y0) poked = 0;
        if (poked && touch_x < ICON_W && touch_y >= GRAPH_Y0) {  /* eraser: clear the world */
            poked = 0;
            world_clear_all(cam_mode);
            pending_obj = eat_obj = -1;
            pending_clip = CLIP_COUNT;
            go_sleep = flee_after = 0;
            printf("bunny: world cleared\n");
        }
        if (touching && touch_x < ICON_W && touch_y >= GRAPH_Y0) poked = 0;
        /* hit box: the middle of the sprite square, where the bunny is drawn */
        int bxa = (int)(bx + ar_off);
        int on_bunny = touching && touch_x >= bxa + spr / 5 && touch_x < bxa + spr * 4 / 5 &&
                       touch_y >= GROUND_Y - spr * 17 / 20 && touch_y < GROUND_Y;

        /* empty screen: a short tap = run there (on release), a drag = draw */
        int tap_run = 0;
        if (poked && !on_bunny && touch_y < GRAPH_Y0) {
            tap_candidate = 1;
            start_x = touch_x;
            start_y = touch_y;
            if (bubble_t > 0.0f) {  /* start a new drawing right away */
                bubble_t = 0.0f;
                ink_clear(cam_mode);
            }
        }
        if (touching && tap_candidate && !cam_mode) {  /* no drawing in the camera world */
            int ddx = touch_x - start_x, ddy = touch_y - start_y;
            if (!drawing && ddx * ddx + ddy * ddy > DRAG_START_PX * DRAG_START_PX) {
                drawing = 1;
                if (ink_n > 0) ink_add(-1, -1);  /* pen lift between strokes */
                ink_add(start_x, start_y);
            }
            if (drawing && touch_y < GRAPH_Y0) {
                int lx = ink_x[ink_n - 1], ly = ink_y[ink_n - 1];
                if ((touch_x - lx) * (touch_x - lx) + (touch_y - ly) * (touch_y - ly) > 9) ink_add(touch_x, touch_y);
            }
            ink_idle = 0.0f;
        }
        if (!touching && was_touching && tap_candidate) {
            if (!drawing) tap_run = 1;
            tap_candidate = 0;
            drawing = 0;
        }
        was_touching = touching;
        if (ink_n > 0 && !touching && bubble_t <= 0.0f) ink_idle += dt;

        us_touch = us_since(st0);
        uint32_t st1 = DWT->CYCCNT;
        float motion = 0.0f;
        if (imu_rc == 0 && frame % IMU_EVERY == 0) {
            imu_poll(&imu_last, NULL, NULL, NULL);
            us_imu = us_since(st1);
            if (us_imu > IMU_SLOW_US) {
                if (++imu_fails >= SENSOR_FAILS) {  /* the I3C bus keeps timing out: stop using it */
                    printf("imu: %d slow reads (%lu us), IMU off\n", imu_fails, (unsigned long)us_imu);
                    imu_rc = -9;
                }
            } else {
                imu_fails = 0;
            }
        }
        motion = imu_last;
        uint32_t st2 = DWT->CYCCNT;
        float level = 0.0f;
        int loud = 0;
        int mic_blocks_now = 0;
        while (mic_rc == 0 && mic_blocks_now++ < 8 && mic_poll(&level) == 1) {
            float db = level > 1.0f ? 20.0f * log10f(level) : 0.0f;
            memmove(graph_db, graph_db + 1, sizeof(graph_db) - sizeof(graph_db[0]));
            graph_db[GRAPH_N - 1] = db;
            if (db > LOUD_DB) loud = 1;
        }

        uint32_t pressed = 0;
        for (int j = 0; j < 5; ++j) {
            uint32_t v = 1;
            ARM_Driver_GPIO_(15).GetValue(joy_pin[j], &v);
            if (v == 0U) pressed |= 1u << j;
        }
        uint32_t newly = pressed & ~was_pressed;
        was_pressed = pressed;
        if (newly) any_input = 1;
        if (motion > IMU_WAKE_TH || loud) any_input = 1;
        if (newly) printf("joy: pressed %s\n", (newly & 1u) ? "A" : (newly & 2u) ? "B" : (newly & 4u) ? "C" : (newly & 8u) ? "D" : "centre");

        /* ---- voice: push to talk ---- */
        if (g_force_listen) { newly |= 16u; g_force_listen = 0; }
        if (vstate == 0 && mic_rc == 0 && (newly & 16u) && mic_record_start() == 0) {
            vstate = 1;
            vrec_t = 0.0f;
            snprintf(bubble_text, sizeof(bubble_text), "LISTENING...");
            bubble_t = 2.0f;  /* as long as the recording */
            if (!cam_mode) fill_bg(0, 24, w, 100);
            printf("voice: listening\n");
        }
        if (vstate) {
            const int16_t *pcm;
            int npcm, r = mic_record_poll(&pcm, &npcm);
            vrec_t += dt;
            if (r == 1) {
                uint32_t vt0 = DWT->CYCCNT;
                float conf;
                int k = voice_listen(pcm, npcm, &conf);
                if (k == KWS_SILENCE || k < 0) {
                    snprintf(bubble_text, sizeof(bubble_text), "DIDN'T HEAR");
                } else if (k == KWS_UNKNOWN || conf < VOICE_MIN_CONF) {
                    snprintf(bubble_text, sizeof(bubble_text), "???");
                    voice_act = KWS_UNKNOWN;
                } else {
                    char wd[12];
                    const char *nm = voice_class_name(k);
                    int i = 0;
                    for (; nm[i] && i < (int)sizeof(wd) - 1; ++i) wd[i] = (char)(nm[i] - 32 * (nm[i] >= 'a' && nm[i] <= 'z'));
                    wd[i] = 0;
                    snprintf(bubble_text, sizeof(bubble_text), "%s!", wd);
                    voice_act = k;
                }
                printf("voice: %lu us\n", (unsigned long)((DWT->CYCCNT - vt0) / (SystemCoreClock / 1000000U)));
                bubble_t = 1.6f;
                if (!cam_mode) fill_bg(0, 24, w, 100);
                vstate = 0;
            } else if (r < 0 || vrec_t > VOICE_TIMEOUT_S) {
                printf("voice: recording failed (%d)\n", r);
                mic_record_cancel();
                vstate = 0;
            }
            idle_s = 0.0f;
        }
        if (g_force_voice >= 0) { voice_act = g_force_voice < KWS_N_CLASSES ? g_force_voice : -1; g_force_voice = -1; }
        if (cam_mode && ((voice_act == KWS_UP && fs.ok && ride == 0) || (voice_act == KWS_DOWN && ride == 2))) {
            voice_jump = 1;  /* onto / off your shoulder */
            voice_act = -1;
        }

        /* ---- behaviour ---- */
        state_t_s += dt;
        idle_s = any_input ? 0.0f : idle_s + dt;

        if (st == ST_LIE || st == ST_SLEEP) {
            if (any_input) {  /* bolt awake */
                st = ST_ACTION;
                set_clip(motion > IMU_SHAKE_TH ? CLIP_DIZZY : CLIP_STRETCH);
                action_len = bunny_clip_duration(clip);
                state_t_s = 0.0f;
            } else if (st == ST_LIE && state_t_s > bunny_clip_duration(CLIP_LIE_DOWN)) {
                st = ST_SLEEP;
                set_clip(CLIP_SLEEP);
                blend_t = BLEND_S;  /* lie_down ends where sleep starts */
            } else if (st == ST_SLEEP && energy > 0.97f) {  /* slept enough: wakes up by himself */
                st = ST_ACTION;
                set_clip(CLIP_STRETCH);
                action_len = bunny_clip_duration(CLIP_STRETCH);
                state_t_s = 0.0f;
                idle_s = 0.0f;
                printf("bunny: woke up rested\n");
            }
        } else {
            bunny_clip_t act = CLIP_COUNT;
            if (motion > IMU_SHAKE_TH) act = CLIP_DIZZY;
            else if (newly & 1u) act = CLIP_WAVE;
            else if (newly & 2u) act = CLIP_YAWN;
            else if (newly & 4u) act = CLIP_HOP;
            else if (newly & 8u) act = CLIP_CLAP;
            else if (poked && on_bunny) act = CLIP_LOVE;
            if (voice_act >= 0 && act == CLIP_COUNT) {  /* a spoken command */
                int v = voice_act;
                fun = clamp01(fun + 0.15f);
                if (v == KWS_UP) act = CLIP_HOP;
                else if (v == KWS_YES) act = CLIP_CLAP;
                else if (v == KWS_NO) act = CLIP_SAD;
                else if (v == KWS_HAPPY) act = CLIP_DANCE;
                else if (v == KWS_UNKNOWN) act = CLIP_CONFUSED;
                else if (v == KWS_DOWN && !ride) {  /* lie down (and nap) */
                    st = ST_LIE;
                    set_clip(CLIP_LIE_DOWN);
                    state_t_s = 0.0f;
                } else if (v == KWS_STOP) {  /* freeze */
                    st = ST_IDLE;
                    set_clip(CLIP_IDLE);
                    state_t_s = 0.0f;
                    next_decision = 6.0f;
                    pending_obj = -1;
                    pending_clip = CLIP_COUNT;
                } else if (v == KWS_LEFT || v == KWS_RIGHT || v == KWS_GO) {  /* walk / run over there */
                    target_x = v == KWS_LEFT ? x_min : v == KWS_RIGHT ? x_max
                             : (bx < (x_min + x_max) / 2.0f ? x_max : x_min);
                    st = ST_WANDER;
                    state_t_s = 0.0f;
                }
                printf("voice: -> %s\n", voice_class_name(v));
            }
            voice_act = -1;
            if (g_force_clip >= 0 && g_force_clip < CLIP_COUNT) { act = (bunny_clip_t)g_force_clip; g_force_clip = -1; }

            if (act == CLIP_LOVE) {  /* petted: affection, and maybe a lesson */
                affection = clamp01(affection + 0.35f);
                fun = clamp01(fun + 0.15f);
                if (last_spont >= 0 && spont_t < LEARN_WINDOW_S) {
                    pref[last_spont] = pref[last_spont] * 1.35f > 3.0f ? 3.0f : pref[last_spont] * 1.35f;
                    printf("bunny: learned you like it when I %s (x%d.%d)\n", act_name[last_spont],
                           (int)pref[last_spont], (int)(pref[last_spont] * 10.0f) % 10);
                    snprintf(bubble_text, sizeof(bubble_text), "%s", act_more[last_spont]);
                    if (!cam_mode) fill_bg(0, 24, w, 100);
                    bubble_t = 2.0f;
                    last_spont = -1;
                }
            } else if (act == CLIP_DIZZY && last_spont >= 0 && spont_t < LEARN_WINDOW_S) {
                pref[last_spont] *= 0.75f;  /* shaken right after: do that less */
                if (pref[last_spont] < 0.3f) pref[last_spont] = 0.3f;
                printf("bunny: learned you don't like it when I %s\n", act_name[last_spont]);
                last_spont = -1;
            }

            if (touching && on_bunny) {
                hold_s += dt;
                if (hold_s > HOLD_HATCH_S && !hatched_this_hold) {
                    hatch();
                    hatched_this_hold = 1;
                    act = CLIP_HOP;
                }
            } else {
                hold_s = 0.0f;
                hatched_this_hold = 0;
            }

            if (act != CLIP_COUNT && !(act == clip && st == ST_ACTION)) {
                st = ST_ACTION;
                set_clip(act);
                action_len = bunny_clip_duration(act) * (act == CLIP_HOP || act == CLIP_LOVE || act == CLIP_DANCE ? 1.0f : 1.5f);
                state_t_s = 0.0f;
            } else if (tap_run) {  /* run to where you tapped */
                target_x = (float)touch_x - ar_off - (float)spr / 2.0f;
                st = ST_WANDER;
                state_t_s = 0.0f;
            }

            switch (st) {
            case ST_ACTION:
                if (eat_obj >= 0 && clip == CLIP_EAT && objs[eat_obj].active) {  /* nibble the fish */
                    objs[eat_obj].bite += dt / EAT_S;
                    if (objs[eat_obj].bite >= 1.0f) {
                        objs[eat_obj].active = 0;
                        eat_obj = -1;
                        hunger = hunger + 0.5f > 1.0f ? 1.0f : hunger + 0.5f;
                        printf("bunny: ate the fish, food %d%%\n", (int)(hunger * 100.0f));
                        snprintf(bubble_text, sizeof(bubble_text), "YUM!");
                        if (!cam_mode) fill_bg(0, 24, w, 100);
                        bubble_t = 1.6f;
                        state_t_s = action_len;  /* done */
                    }
                    world_rebuild(cam_mode, 0, 0, -1, -1);
                }
                if (state_t_s > action_len) {
                    eat_obj = -1;
                    if (flee_after) {  /* scared of the cat: run to the far side */
                        int c = world_find(C_CAT);
                        float cxp = c >= 0 ? objs[c].x : (float)w / 2.0f;
                        target_x = cxp < (float)w / 2.0f ? x_max : x_min;
                        flee_after = 0;
                        st = ST_WANDER;
                    } else {
                        st = ST_IDLE;
                        set_clip(CLIP_IDLE);
                        next_decision = frand(2.0f, 5.0f);
                    }
                    state_t_s = 0.0f;
                }
                break;
            case ST_WANDER: {
                float dist = target_x - bx;
                if (target_x < x_min) target_x = x_min;
                if (target_x > x_max) target_x = x_max;
                if (fabsf(dist) < 4.0f) {
                    state_t_s = 0.0f;
                    if (pending_obj >= 0 && objs[pending_obj].active && pending_clip != CLIP_COUNT) {
                        /* arrived at a drawing: eat it / sniff it */
                        facing_left = objs[pending_obj].x < bx + (float)spr / 2.0f;
                        st = ST_ACTION;
                        set_clip(pending_clip);
                        action_len = pending_clip == CLIP_EAT ? EAT_S + 0.5f : 2.6f;
                        if (pending_clip == CLIP_EAT) eat_obj = pending_obj;
                        printf("bunny: reached the %s\n", pending_clip == CLIP_EAT ? "fish" : "flower");
                        pending_obj = -1;
                        pending_clip = CLIP_COUNT;
                    } else if (go_sleep) {  /* home: lie down here */
                        go_sleep = 0;
                        st = ST_LIE;
                        set_clip(CLIP_LIE_DOWN);
                    } else {
                        st = ST_IDLE;
                        set_clip(CLIP_IDLE);
                        next_decision = frand(2.0f, 5.0f);
                    }
                } else {
                    int fast = fabsf(dist) > RUN_IF_FARTHER;
                    set_clip(fast ? CLIP_RUN : CLIP_WALK);
                    float step = (fast ? RUN_PX_S : WALK_PX_S) * dt;
                    facing_left = dist < 0.0f;
                    bx += dist > 0.0f ? (step < dist ? step : dist) : (-step > dist ? -step : dist);
                }
                break;
            }
            case ST_IDLE:
            default:
                if (energy < 0.12f || (idle_s > 45.0f && energy < 0.6f)) {  /* tired: go to sleep */
                    int hk = world_find(C_HOUSE);
                    if (hk >= 0 && !go_sleep && fabsf(objs[hk].x - (bx + (float)spr / 2.0f)) > 50.0f) {
                        go_sleep = 1;  /* walk home first */
                        target_x = objs[hk].x - (float)spr / 2.0f + (objs[hk].x < (float)w / 2.0f ? 110.0f : -110.0f);
                        st = ST_WANDER;
                    } else {
                        go_sleep = 0;
                        st = ST_LIE;
                        set_clip(CLIP_LIE_DOWN);
                    }
                    state_t_s = 0.0f;
                    int sk = world_find(C_SUN);  /* bedtime: the sun sets */
                    if (sk >= 0 && objs[sk].life < 0.0f) {
                        objs[sk].x0 = objs[sk].x; objs[sk].y0 = objs[sk].y; objs[sk].s0 = objs[sk].s;
                        objs[sk].x1 = objs[sk].x; objs[sk].y1 = (float)FLOOR_Y; objs[sk].s1 = objs[sk].s;
                        objs[sk].t = 0.0f; objs[sk].dur = 3.0f; objs[sk].life = 3.0f;
                    }
                } else if (hunger < HUNGRY_AT && hungry_nag <= 0.0f && bubble_t <= 0.0f) {
                    hungry_nag = 25.0f;  /* hungry: complain */
                    snprintf(bubble_text, sizeof(bubble_text), "HUNGRY!");
                    if (!cam_mode) fill_bg(0, 24, w, 100);
                    bubble_t = 2.5f;
                    st = ST_ACTION;
                    set_clip(CLIP_SAD);
                    action_len = 3.0f;
                    state_t_s = 0.0f;
                } else if (thought_cd <= 0.0f && bubble_t <= 0.0f &&
                           (energy < 0.3f || fun < 0.25f || affection < 0.2f)) {
                    /* a thought about what he needs */
                    thought_cd = frand(25.0f, 40.0f);
                    snprintf(bubble_text, sizeof(bubble_text), "%s",
                             energy < 0.3f ? "SLEEPY" : (affection < 0.2f ? "PET ME?" : "BORED!"));
                    if (!cam_mode) fill_bg(0, 24, w, 100);
                    bubble_t = 2.2f;
                } else if (state_t_s > next_decision) {
                    act_t a = choose_action(energy, fun, affection, hunger);
                    static const bunny_clip_t act_clip[A_COUNT] = {CLIP_WALK, CLIP_LOOK, CLIP_SIT, CLIP_HOP, CLIP_CLAP,
                                                                   CLIP_WAVE, CLIP_STRETCH, CLIP_YAWN, CLIP_SAD};
                    state_t_s = 0.0f;
                    next_decision = frand(2.5f, 6.0f);
                    last_spont = a;
                    spont_t = 0.0f;
                    if (a == A_WANDER) {
                        target_x = frand(x_min, x_max);
                        st = ST_WANDER;
                    } else if (a == A_LOOK || a == A_SIT) {
                        set_clip(act_clip[a]);  /* stays idle, just looks around / sits */
                    } else {
                        st = ST_ACTION;
                        set_clip(act_clip[a]);
                        action_len = bunny_clip_duration(act_clip[a]);
                    }
                }
                break;
            }
        }
        perk = loud ? 1.2f : (perk > dt ? perk - dt : 0.0f);
        if (world_update(dt)) world_rebuild(cam_mode, 0, 0, -1, -1);

        /* camera world: jump onto the shoulder of the person in view, ride it */
        if (cam_mode) {
            int fresh = face.ok;
            face.ok = 0;  /* consumed */
            if (fresh && fs.ok) {  /* reject one-off jumps (a glitch), accept if it repeats */
                float jx = fabsf(face.x - fs.x), jy = fabsf(face.y - fs.y);
                if ((jx > FACE_JUMP_MAX * fs.w || jy > FACE_JUMP_MAX * fs.h) && glitches < 2) {
                    ++glitches;
                    fresh = 0;
                } else {
                    glitches = 0;
                }
            }
            if (fresh) {
                float a = fs.ok ? 0.5f : 1.0f;  /* smooth the face itself */
                fs.x += (face.x - fs.x) * a; fs.y += (face.y - fs.y) * a;
                fs.w += (face.w - fs.w) * a; fs.h += (face.h - fs.h) * a;
                fs.roll += (face.roll - fs.roll) * a;
                fs.score = face.score;
                fs.ok = 1;
                face_seen_s += dt * FACE_EVERY; face_lost_s = 0.0f;
            } else {
                face_lost_s += dt;
                if (face_lost_s > 0.3f) face_seen_s = 0.0f;
                if (face_lost_s > FACE_LOST_S) fs.ok = 0;
            }
            if (fs.ok) {
                if (ride == 0 || ride == 3) ride_side = (bx + (float)spr / 2.0f) < fs.x ? -1 : 1;  /* the side he is on */
                float tx = fs.x + (float)ride_side * SHOULDER_DX * fs.w, ty = fs.y + SHOULDER_DY * fs.h;
                /* that shoulder off the screen: use the other one */
                if (tx < 30.0f || tx > (float)w - 30.0f) {
                    ride_side = -ride_side;
                    tx = fs.x + (float)ride_side * SHOULDER_DX * fs.w;
                }
                if (ty > (float)GRAPH_Y0 - 10.0f) ty = (float)GRAPH_Y0 - 10.0f;
                float a = (ride == 2) ? 0.3f : 1.0f;  /* gentle while riding */
                sh_x += (tx - sh_x) * a;
                sh_y += (ty - sh_y) * a;
            }
            ride_t += dt;
            if (ride_block_s > 0.0f) ride_block_s -= dt;
            if (ride == 0 && fs.ok && (voice_jump || (face_seen_s > FACE_STABLE_S && ride_block_s <= 0.0f)) &&
                st != ST_LIE && st != ST_SLEEP) {
                voice_jump = 0;
                ride_block_s = 0.0f;
                ride = 1;  /* jump! */
                ride_t = 0.0f;
                jump_x0 = bx;
                jump_y0 = (float)(GROUND_Y - spr);
                st = ST_ACTION;
                set_clip(CLIP_HOP);
                action_len = 0.8f;
                state_t_s = 0.0f;
                printf("bunny: jumping onto the %s shoulder (face %d%%)\n", ride_side < 0 ? "left" : "right", (int)(fs.score * 100.0f));
            } else if (ride == 1 && ride_t > 0.8f) {
                ride = 2;  /* landed: ride along */
                st = ST_ACTION;
                set_clip(CLIP_SIT);
                action_len = 1e9f;
                state_t_s = 0.0f;
            } else if (ride == 2 && (face_lost_s > FACE_LOST_S || voice_jump)) {
                if (voice_jump) ride_block_s = RIDE_BLOCK_S;
                voice_jump = 0;
                ride = 3;  /* hop down */
                ride_t = 0.0f;
                jump_x0 = sh_x - (float)spr / 2.0f;
                jump_y0 = sh_y - 0.95f * (float)spr;
                bx = jump_x0 < x_min ? x_min : (jump_x0 > x_max ? x_max : jump_x0);
                set_clip(CLIP_HOP);
                printf("bunny: hopping down\n");
            } else if (ride == 3 && ride_t > 0.8f) {
                ride = 0;
                st = ST_IDLE;
                set_clip(CLIP_IDLE);
                state_t_s = 0.0f;
            }
            if (ride == 2) idle_s = 0.0f;  /* no napping on a shoulder */
            if (ride == 2 && st != ST_ACTION) {  /* an action (dance, wave) ended: sit back down */
                st = ST_ACTION;
                set_clip(CLIP_SIT);
                action_len = 1e9f;
                state_t_s = 0.0f;
            }
            if (ride == 1 || ride == 3) voice_jump = 0;
        } else if (ride || fs.ok) {
            ride = 0;
            voice_jump = 0;
            face.ok = fs.ok = 0;
            face_seen_s = face_lost_s = 0.0f;
        }
        gaze += ((cam_mode && fs.ok ? 1.0f : 0.0f) - gaze) * (dt * 4.0f > 1.0f ? 1.0f : dt * 4.0f);
        if (st != ST_SLEEP) hunger -= dt / HUNGER_S;
        /* needs drift */
        if (st == ST_SLEEP || st == ST_LIE) energy = clamp01(energy + dt / 40.0f);
        else energy = clamp01(energy - dt / ((clip == CLIP_RUN || clip == CLIP_HOP) ? 120.0f : 300.0f));
        fun = clamp01(fun - dt / 90.0f + (any_input ? 0.05f : 0.0f) + (cam_mode && fs.ok ? dt / 40.0f : 0.0f));
        affection = clamp01(affection - dt / 150.0f + (cam_mode && fs.ok ? dt / 80.0f : 0.0f));
        if (thought_cd > 0.0f) thought_cd -= dt;
        spont_t += dt;
        if (hunger < 0.0f) hunger = 0.0f;
        if (hungry_nag > 0.0f) hungry_nag -= dt;

        if (g_test_doodle) {  /* remote test: draw a shape as if by finger */
            int t = g_test_doodle;
            g_test_doodle = 0;
            ink_clear(cam_mode);
            int cx = w / 2, cy = 300;
            if (t == 1) {  /* 5-point star, one stroke */
                for (int i = 0; i <= 10; ++i) {
                    float a = -1.5708f + (float)i * 2.5133f;  /* every 144 degrees */
                    ink_add(cx + (int)(130.0f * cosf(a)), cy + (int)(130.0f * sinf(a)));
                }
            } else if (t == 2) {  /* house: square + roof */
                int xs[] = {cx - 100, cx - 100, cx + 100, cx + 100, cx - 100, cx, cx + 100};
                int ys[] = {cy + 120, cy - 20, cy - 20, cy + 120, cy + 120, cy - 20, cy - 20};
                for (int i = 0; i < 5; ++i) ink_add(xs[i], ys[i]);
                ink_add(-1, -1);
                ink_add(cx - 115, cy - 20);
                ink_add(cx, cy - 130);
                ink_add(cx + 115, cy - 20);
            } else if (t == 4) {  /* fish: body ellipse + tail + eye */
                for (int i = 0; i <= 36; ++i) {
                    float an = (float)i * 0.17453f;
                    ink_add(cx + (int)(110.0f * cosf(an)), cy + (int)(55.0f * sinf(an)));
                }
                ink_add(-1, -1);
                ink_add(cx + 105, cy); ink_add(cx + 170, cy - 50); ink_add(cx + 170, cy + 50); ink_add(cx + 105, cy);
                ink_add(-1, -1); ink_add(cx - 60, cy - 15); ink_add(cx - 58, cy - 12);
            } else {  /* smiley: face circle, two eyes, mouth */
                for (int i = 0; i <= 40; ++i) {
                    float a = (float)i * 0.15708f;
                    ink_add(cx + (int)(120.0f * cosf(a)), cy + (int)(120.0f * sinf(a)));
                }
                ink_add(-1, -1); ink_add(cx - 45, cy - 40); ink_add(cx - 44, cy - 25);
                ink_add(-1, -1); ink_add(cx + 45, cy - 40); ink_add(cx + 44, cy - 25);
                ink_add(-1, -1);
                for (int i = 0; i <= 12; ++i) {
                    float a = 0.4f + (float)i * 0.195f;
                    ink_add(cx + (int)(70.0f * cosf(a)), cy + 10 + (int)(60.0f * sinf(a)));
                }
            }
            ink_idle = INK_IDLE_S + 1.0f;
            bubble_t = 0.0f;
        }
        if (ink_n > 0 && ink_idle > INK_IDLE_S && bubble_t <= 0.0f) {
            static float in28[28 * 28];
            float sc8[8];
            ink_rasterize(in28);
            uint32_t dc0 = DWT->CYCCNT;
            int nd = doodle_classify(in28, sc8, 8);
            uint32_t dus = us_since(dc0);
            doodle_us = dus;
            if (nd == 8) {
                int best = 0;
                float mx = sc8[0], sum = 0.0f;
                for (int i = 1; i < 8; ++i) if (sc8[i] > mx) { mx = sc8[i]; best = i; }
                for (int i = 0; i < 8; ++i) sum += expf(sc8[i] - mx);
                float pbest = 1.0f / sum;
                printf("doodle: %s %d%% (%lu us)\n", doodle_names[best], (int)(pbest * 100.0f), (unsigned long)dus);
                fun = clamp01(fun + 0.25f);
                bunny_clip_t react = pbest >= DOODLE_MIN_P ? doodle_react[best] : CLIP_CONFUSED;
                if (pbest >= DOODLE_MIN_P) {
                    snprintf(bubble_text, sizeof(bubble_text), "%s!", doodle_names[best]);
                } else {
                    snprintf(bubble_text, sizeof(bubble_text), "???");
                }
                if (!cam_mode) fill_bg(0, 24, w, 100);
                bubble_t = BUBBLE_S;
                facing_left = (ink_x0 + ink_x1) / 2 < (int)(bx + ar_off) + spr / 2;
                int awake = st != ST_LIE && st != ST_SLEEP;
                if (awake) {
                    st = ST_ACTION;
                    set_clip(react);
                    action_len = BUBBLE_S;
                    state_t_s = 0.0f;
                }
                idle_s = 0.0f;
                if (pbest >= DOODLE_MIN_P) {
                    /* the drawing comes to life */
                    float dcx = (ink_x0 + ink_x1) / 2.0f;
                    if (dcx < 70.0f) dcx = 70.0f;
                    if (dcx > (float)w - 70.0f) dcx = (float)w - 70.0f;
                    int spanw = ink_x1 - ink_x0, spanh = ink_y1 - ink_y0, span = spanw > spanh ? spanw : spanh;
                    int k = -1;
                    switch (best) {
                    case C_FISH:
                        k = obj_from_ink(C_FISH, dcx, floor_y_for(110.0f, spanh, span), 110.0f, -1.0f);
                        if (awake) {  /* go and eat it */
                            pending_obj = k;
                            pending_clip = CLIP_EAT;
                            target_x = dcx - (float)spr / 2.0f + (dcx < bx + (float)spr / 2.0f ? 120.0f : -120.0f);
                            st = ST_WANDER;
                        }
                        break;
                    case C_FLOWER:
                        k = obj_from_ink(C_FLOWER, dcx, floor_y_for(120.0f, spanh, span), 120.0f, -1.0f);
                        if (awake) {  /* go and sniff it */
                            pending_obj = k;
                            pending_clip = CLIP_LOVE;
                            target_x = dcx - (float)spr / 2.0f + (dcx < bx + (float)spr / 2.0f ? 120.0f : -120.0f);
                            st = ST_WANDER;
                        }
                        break;
                    case C_HOUSE:
                        k = obj_from_ink(C_HOUSE, dcx, floor_y_for(160.0f, spanh, span), 160.0f, -1.0f);
                        break;
                    case C_CAT:
                        k = obj_from_ink(C_CAT, dcx, floor_y_for(110.0f, spanh, span), 110.0f, 8.0f);
                        if (awake) {
                            action_len = 1.3f;  /* startled, then runs away */
                            flee_after = 1;
                        }
                        break;
                    case C_SUN:
                        k = obj_from_ink(C_SUN, 80.0f, 150.0f, 90.0f, -1.0f);
                        break;
                    case C_STAR:
                        k = obj_from_ink(C_STAR, frand(200.0f, (float)w - 60.0f), frand(130.0f, 200.0f), 70.0f, -1.0f);
                        break;
                    case C_LIGHTNING:
                        k = obj_from_ink(C_LIGHTNING, (ink_x0 + ink_x1) / 2.0f, (ink_y0 + ink_y1) / 2.0f, (float)span * 0.8f, 1.8f);
                        break;
                    case C_SMILEY:
                    default:
                        k = obj_from_ink(C_SMILEY, (ink_x0 + ink_x1) / 2.0f, (ink_y0 + ink_y1) / 2.0f - 60.0f, (float)span * 0.8f, 3.5f);
                        break;
                    }
                    (void)k;
                    ink_clear(cam_mode);  /* the strokes now live in the object */
                }
            } else {
                printf("doodle: classify failed (%d)\n", nd);
                ink_clear(cam_mode);
            }
            ink_idle = 0.0f;
        }
        if (bubble_t > 0.0f) {
            bubble_t -= dt;
            if (bubble_t <= 0.0f) {  /* bubble (and an unrecognised drawing) go away */
                if (ink_n > 0) ink_clear(cam_mode);
                if (!cam_mode) fill_bg(0, 24, w, 100);
                prev = (rect_t){0, 0, 0, 0};
            }
        }

        /* ---- pose: clip, cross-fade, identity, perk, smoothing ---- */
        clip_t += dt;
        blend_t += dt;
        bunny_pose_t target;
        bunny_clip(clip, clip_t, 1.0f, &target);
        float u = smoothstep(blend_t / BLEND_S);
        for (int i = 0; i < BUNNY_NDIAL; ++i) target.d[i] = from_d[i] + (target.d[i] - from_d[i]) * u;
        target.dx = from_dx + (target.dx - from_dx) * u;
        target.dy = from_dy + (target.dy - from_dy) * u;
        target.d[D_HUE] = identity[0];
        target.d[D_SAT] = identity[1];
        target.d[D_EAR_LEN] = identity[2];
        target.d[D_HEAD_SCALE] = identity[3];
        if (perk > 0.0f && st != ST_SLEEP) {
            float p = perk > 1.0f ? 1.0f : perk;
            target.d[D_EAR_L] += (-2.0f - target.d[D_EAR_L]) * p;
            target.d[D_EAR_R] += (2.0f - target.d[D_EAR_R]) * p;
            target.d[D_CURL] *= 1.0f - p;
            for (int i = D_EYE_OPEN; i <= D_EYE_SAD; ++i) target.d[i] *= 1.0f - 0.8f * p;
            target.d[D_EYE_WIDE] += 0.8f * p;
            target.d[D_LOOK_Y] = -3.0f * p;
        }
        if (clip == CLIP_IDLE || clip == CLIP_LOOK || clip == CLIP_SIT || clip == CLIP_WALK) {
            /* mood shows on his face while he is not busy with something else */
            float valence = clamp01(0.4f * affection + 0.35f * fun + 0.25f * hunger);
            float happy = clamp01((valence - 0.65f) * 3.0f) * 0.35f, sad = clamp01((0.35f - valence) * 3.0f) * 0.4f;
            float sleepy = clamp01((0.3f - energy) * 3.0f) * 0.45f;
            float tot = happy + sad + sleepy;
            if (tot > 0.6f) { happy *= 0.6f / tot; sad *= 0.6f / tot; sleepy *= 0.6f / tot; }  /* never more than 60% */
            for (int i = D_EYE_OPEN; i <= D_EYE_SAD; ++i) target.d[i] *= 1.0f - happy - sad - sleepy;
            target.d[D_EYE_HAPPY] += happy;
            target.d[D_EYE_SAD] += sad;
            target.d[D_EYE_SLEEP] += sleepy;
            if (sad > 0.0f || sleepy > 0.0f) target.d[D_CURL] += 30.0f * (sad + sleepy);  /* droopy ears */
        }
        if (gaze > 0.01f && fs.ok) {
            /* he looks at your face; on your shoulder he copies your head tilt; lean in close and he is surprised */
            float hx = (float)(ride ? sh_x : bx + (float)spr / 2.0f);
            float hy = ride ? sh_y - 0.7f * (float)spr : (float)(GROUND_Y - spr) + 0.35f * (float)spr;
            float lx = (fs.x - hx) / 30.0f, ly = (fs.y - hy) / 30.0f;
            if (facing_left) lx = -lx;  /* the sprite is mirrored */
            lx = lx > 7.0f ? 7.0f : (lx < -7.0f ? -7.0f : lx);
            ly = ly > 6.0f ? 6.0f : (ly < -6.0f ? -6.0f : ly);
            target.d[D_LOOK_X] += (lx - target.d[D_LOOK_X]) * gaze;
            target.d[D_LOOK_Y] += (ly - target.d[D_LOOK_Y]) * gaze;
            float tilt = ride == 2 ? fs.roll * 0.8f : lx * 1.6f;
            if (facing_left && ride == 2) tilt = -tilt;
            tilt = tilt > 16.0f ? 16.0f : (tilt < -16.0f ? -16.0f : tilt);
            target.d[D_HEAD_TILT] += (tilt - target.d[D_HEAD_TILT]) * gaze;
            if (!ride && fs.w > 0.42f * (float)w) {  /* very close: wide eyes */
                for (int i = D_EYE_OPEN; i <= D_EYE_SAD; ++i) target.d[i] *= 0.3f;
                target.d[D_EYE_WIDE] += 0.7f;
            }
        }
        if (frame == 0) shown = target;
        for (int i = 0; i < BUNNY_NDIAL; ++i) shown.d[i] += (target.d[i] - shown.d[i]) * DIAL_SMOOTH;
        shown.dx += (target.dx - shown.dx) * DIAL_SMOOTH;
        shown.dy += (target.dy - shown.dy) * DIAL_SMOOTH;
        shown.zzz = target.zzz;
        shown.sweat = target.sweat;
        shown.hearts = target.hearts;

        /* ---- keep the dials in the space the model was trained on ---- */
        {
            /* a NaN / infinity (one bad formula) would warp the picture and can stop the runtime */
            for (int i = 0; i < BUNNY_NDIAL; ++i) {
                if (!isfinite(shown.d[i])) shown.d[i] = isfinite(target.d[i]) ? target.d[i] : 0.0f;
            }
            if (!isfinite(shown.dx)) shown.dx = 0.0f;
            if (!isfinite(shown.dy)) shown.dy = 0.0f;
            float se = 0.0f, sm = 0.0f;
            for (int i = D_EYE_OPEN; i <= D_EYE_SAD; ++i) { if (shown.d[i] < 0.0f) shown.d[i] = 0.0f; se += shown.d[i]; }
            for (int i = D_MOUTH_SMILE; i <= D_MOUTH_YAWN; ++i) { if (shown.d[i] < 0.0f) shown.d[i] = 0.0f; sm += shown.d[i]; }
            if (se < 1e-3f) { shown.d[D_EYE_OPEN] = 1.0f; se = 1.0f; }
            if (sm < 1e-3f) { shown.d[D_MOUTH_SMILE] = 1.0f; sm = 1.0f; }
            for (int i = D_EYE_OPEN; i <= D_EYE_SAD; ++i) shown.d[i] /= se;
            for (int i = D_MOUTH_SMILE; i <= D_MOUTH_YAWN; ++i) shown.d[i] /= sm;
            for (int i = D_EAR_L; i < BUNNY_NDIAL; ++i) {
                float lo, hi;
                bunny_dial_range(i, &lo, &hi);
                if (shown.d[i] < lo) shown.d[i] = lo;
                if (shown.d[i] > hi) shown.d[i] = hi;
            }
        }

        /* ---- NN: the Ethos-U85 draws the bunny ---- */
        bunny_normalize(shown.d, nn_in);
        us_mic = us_since(st2);
        uint32_t c0 = DWT->CYCCNT;
        int rc = 0;
        if (!cam_mode || (frame & 1u) == 0u) {  /* camera world: every other frame */
            rc = nn_forward(nn_in, BUNNY_NDIAL, &out, &n_out);
            bunny_npu_us = (DWT->CYCCNT - c0) / (SystemCoreClock / 1000000u);
            if (rc == 0) convert_output(out);
        }
        nn_us_acc += (DWT->CYCCNT - c0) / (SystemCoreClock / 1000000u);

        us_nn = us_since(c0);
        uint32_t st3 = DWT->CYCCNT;
        /* ---- draw ---- */
        if (cam_mode && imu_rc == 0) {
            /* AR anchor: turning the board shifts the bunny the other way, so
             * he seems to stay put in the room; drifts back when still */
            float g[3];
            imu_last_gyro(g);
            float rate = g[AR_AXIS];
            if (fabsf(rate) > AR_DEADBAND) ar_off += AR_SIGN * rate * AR_PX_PER_DEG * dt;
            ar_off -= ar_off * AR_RETURN * dt;
            if (ar_off > AR_MAX) ar_off = AR_MAX;
            if (ar_off < -AR_MAX) ar_off = -AR_MAX;
        }
        float k = (float)spr / 512.0f;
        int ox = (int)(bx + ar_off) + (int)(shown.dx * k);
        int oy = GROUND_Y - spr + (int)(shown.dy * k);
        if (cam_mode && ride) {
            /* feet (about 0.95 of the sprite square) on the shoulder point */
            float rx = sh_x - (float)spr / 2.0f, ry = sh_y - 0.95f * (float)spr;
            float fx = bx, fy = (float)(GROUND_Y - spr);
            float u2 = smoothstep(ride_t / 0.8f);
            float px = rx, py = ry;
            if (ride == 1) {  /* arc up from where he stood */
                px = jump_x0 + (rx - jump_x0) * u2;
                py = jump_y0 + (ry - jump_y0) * u2 - 160.0f * sinf(3.14159f * u2);
            } else if (ride == 3) {  /* hop back down */
                px = jump_x0 + (fx - jump_x0) * u2;
                py = jump_y0 + (fy - jump_y0) * u2 - 90.0f * sinf(3.14159f * u2);
            }
            ox = (int)px + (int)(shown.dx * k);
            oy = (int)py + (int)(shown.dy * k);
        }
        rect_t cur = {ox, oy, ox + spr, oy + spr};
        /* Every frame is drawn whole into the hidden buffer, then swapped in:
         * background, drawings (lines only), the bunny with its white see-through */
        if (cam_mode) {
            /* the frame started last loop; wait up to 40 ms for it, then start the next */
            uint32_t cw0 = DWT->CYCCNT;
            while (!camera_capture_poll() && us_since(cw0) < 40000u) {}
            draw_camera_bg(w, GRAPH_Y0, -10000, 0, 0);  /* camera only */
            if (++cam_frames % FACE_EVERY == 0) {  /* face detector on this frame */
                face_fill_input();
                const float *fo = NULL;
                uint32_t fc0 = DWT->CYCCNT;
                if (face_detect(&fo) == 0) face = face_decode(fo, w, GRAPH_Y0);
                face_us = us_since(fc0);
            }
            camera_capture_start();
            if (FACE_BOX && fs.ok) {
                draw_box((int)(fs.x - fs.w / 2), (int)(fs.y - fs.h / 2), (int)(fs.x + fs.w / 2),
                         (int)(fs.y + fs.h / 2), 120, 255, 160);
            }
        } else {
            clear_world(w);
        }
        world_draw();
        draw_sprite(ox, oy, facing_left, 1);
        draw_effects(&shown, ox, oy, facing_left);
        if (bubble_t > 0.0f) draw_bubble(bubble_text, w);
        prev = cur;
        prev_dot = (rect_t){0, 0, 0, 0};
        if (touching && touch_y < GRAPH_Y0 - 8) {
            fill_circle(touch_x, touch_y, 6, 120, 80, 200);
            prev_dot = (rect_t){touch_x - 7, touch_y - 7, touch_x + 8, touch_y + 8};
        }
        draw_graph(ICON_W, w - ICON_W, h);
        draw_icon(cam_mode, w, h);
        draw_eraser(h);
        draw_hunger(hunger);
        if (g_stats) draw_stats(cam_mode, g_bunny_fps, bunny_npu_us, face_us, doodle_us);
        display_flush();
        if (wdt_rc == ARM_DRIVER_OK) Driver_WDT0.Feed();
        us_draw = us_since(st3);
        if (us_since(st0) > SLOW_FRAME_MS * 1000u) {
            printf("slow frame %lu: touch %lu us, imu %lu us, mic %lu us, nn %lu us, draw %lu us\n",
                   (unsigned long)frame, (unsigned long)us_touch, (unsigned long)us_imu, (unsigned long)us_mic,
                   (unsigned long)us_nn, (unsigned long)us_draw);
        }

        /* LEDs follow the mood */
        if (world_find(C_LIGHTNING) >= 0) leds_rgb((frame >> 1) & 1, (frame >> 1) & 1, (frame >> 1) & 1);
        else if (clip == CLIP_LOVE) leds_rgb(1, 0, 1);
        else if (clip == CLIP_SHAKE || clip == CLIP_DIZZY) leds_rgb(1, 1, 0);
        else if (clip == CLIP_ANGRY) leds_rgb(1, 0, 0);
        else if (clip == CLIP_HOP || clip == CLIP_CLAP) leds_rgb(0, 1, 0);
        else if (perk > 0.0f) leds_rgb(0, 0, 1);
        else leds_rgb(0, ((now / (SystemCoreClock / 2U)) & 3U) == 0U, 0);  /* slow green heartbeat */

        if (++frame % 60 == 0) {
            uint32_t t1 = DWT->CYCCNT;
            g_bunny_fps = (uint32_t)(60ull * SystemCoreClock / (t1 - t_stat));
            g_bunny_nn_us = nn_us_acc / 60;
            t_stat = t1;
            nn_us_acc = 0;
            if (cam_mode) printf("face: %s score %d%% at %d,%d size %d roll %d, detect %lu us, ride %d side %d\n", fs.ok ? "yes" : "no",
                                 (int)(fs.score * 100.0f), (int)fs.x, (int)fs.y, (int)fs.w, (int)fs.roll, (unsigned long)face_us, ride, ride_side);
            printf("mood: energy %d fun %d love %d food %d | likes walk %d look %d sit %d hop %d clap %d wave %d\n",
                   (int)(energy * 100), (int)(fun * 100), (int)(affection * 100), (int)(hunger * 100),
                   (int)(pref[A_WANDER] * 10), (int)(pref[A_LOOK] * 10), (int)(pref[A_SIT] * 10), (int)(pref[A_HOP] * 10),
                   (int)(pref[A_CLAP] * 10), (int)(pref[A_WAVE] * 10));
            printf("bunny: [%s] %lu fps, NN %lu us, state %s, clip %s, imu %d, mic %d dB, stack %lu B\n",
                   cam_mode ? "camera" : "home", (unsigned long)g_bunny_fps, (unsigned long)g_bunny_nn_us, state_names[st],
                   bunny_clip_name(clip), (int)motion, (int)graph_db[GRAPH_N - 1], (unsigned long)stack_used());
        }
    }
}

#endif /* BOARD_BUNNY_DEMO */
