/*---------------------------------------------------------------------------
 * MIPI-DSI LCD (ILI9806E panel on the DevKit-E8) bring-up.
 *
 * Follows Alif's bare-metal CDC200 demo: power the MIPI PHY through the Secure
 * Enclave, then start the CDC200 display controller scanning out a frame
 * buffer. The frame buffer lives in the bulk SRAM next to the NPU pools.
 *---------------------------------------------------------------------------*/

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "RTE_Components.h"
#include CMSIS_device_header
#include "RTE_Device.h"

#include "board_config.h"
#include "Driver_CDC200.h"
#include "Driver_MIPI_DSI.h"
#include "DPHY_DSI.h"
#include "Driver_IO.h"
#include "sys_utils.h"
#include "se_services_port.h"

#include "display.h"

#if RTE_CDC200_PIXEL_FORMAT == 0
#define PIXEL_BYTES (4)
#elif RTE_CDC200_PIXEL_FORMAT == 1
#define PIXEL_BYTES (3)
#else
#define PIXEL_BYTES (2)
#endif

/* VBAT PWR_CTRL fields (from the pack's Boards/DevKit-e8/Layers/M55_HP/main.c) */
#define VBAT_PWR_CTRL_TX_DPHY_PWR_MASK        (1U << 0)
#define VBAT_PWR_CTRL_TX_DPHY_ISO             (1U << 1)
#define VBAT_PWR_CTRL_RX_DPHY_PWR_MASK        (1U << 4)
#define VBAT_PWR_CTRL_RX_DPHY_ISO             (1U << 5)
#define VBAT_PWR_CTRL_DPHY_PLL_PWR_MASK       (1U << 8)
#define VBAT_PWR_CTRL_DPHY_PLL_ISO            (1U << 9)
#define VBAT_PWR_CTRL_DPHY_VPH_1P8_PWR_BYP_EN (1U << 12)
#define VBAT_PWR_CTRL_UPHY_PWR_MASK           (1U << 16)
#define VBAT_PWR_CTRL_UPHY_ISO                (1U << 17)

#define LCD_W RTE_PANEL_HACTIVE_TIME
#define LCD_H RTE_PANEL_VACTIVE_LINE

/* .bss.ai_pool is routed to the bulk SRAM by the linker scripts of this layer */
static uint8_t lcd_image[LCD_H][LCD_W][PIXEL_BYTES] __attribute__((section(".bss.ai_pool"), aligned(32)));
/* Second frame buffer for double buffering (display_enable_double_buffer) */
static uint8_t lcd_image2[LCD_H][LCD_W][PIXEL_BYTES] __attribute__((section(".bss.ai_pool"), aligned(32)));
static uint8_t *fb_front = &lcd_image[0][0][0], *fb_back = &lcd_image[0][0][0];
static int fb_double;

extern ARM_DRIVER_CDC200 Driver_CDC200;
extern ARM_DRIVER_MIPI_DSI Driver_MIPI_DSI;

static void dsi_callback(uint32_t event)
{
    (void)event;
}

extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LCD_RESET_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LCD_BL_LED_GPIO_PORT);

/* Diagnostics: the panel driver's reset/backlight GPIO steps, one by one. */
static void pin_steps(const char *name, ARM_DRIVER_GPIO *gpio, uint8_t pin)
{
    int32_t a = gpio->Initialize(pin, NULL);
    int32_t b = gpio->PowerControl(pin, ARM_POWER_FULL);
    int32_t c = gpio->SetDirection(pin, GPIO_PIN_DIRECTION_OUTPUT);
    int32_t d = gpio->SetValue(pin, GPIO_PIN_OUTPUT_STATE_HIGH);
    printf("display: %s pin: init %ld power %ld dir %ld high %ld\n", name, (long)a, (long)b,
           (long)c, (long)d);
}

static volatile uint32_t dsi_error;

/* Result of the 20 MHz clock request made in main() (0xFFFFFFFF = not made) */
volatile uint32_t g_clk20m_rc = 0xFFFFFFFFU, g_clk20m_err = 0U;
/* main(): [0] board_clocks_config result, [1] SE heartbeat result */
volatile uint32_t g_boot_dbg[4] = {0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU};

static void display_callback(uint32_t event)
{
    if (event & ARM_CDC_DSI_ERROR_EVENT) {
        dsi_error = 1;
    }
}

int display_init(void)
{
    /* Power the MIPI DPHY and take it out of isolation, the way the pack's own
     * DevKit-E8 board layer does (vbat_init). Asking the Secure Enclave for a
     * new run profile timed out on this board. */
    VBAT->PWR_CTRL &= ~(VBAT_PWR_CTRL_TX_DPHY_PWR_MASK | VBAT_PWR_CTRL_RX_DPHY_PWR_MASK |
                        VBAT_PWR_CTRL_DPHY_PLL_PWR_MASK | VBAT_PWR_CTRL_DPHY_VPH_1P8_PWR_BYP_EN);
    VBAT->PWR_CTRL &= ~(VBAT_PWR_CTRL_TX_DPHY_ISO | VBAT_PWR_CTRL_RX_DPHY_ISO | VBAT_PWR_CTRL_DPHY_PLL_ISO);
    /* ...and the USB PHY power/isolation, exactly as the pack's vbat_init does */
    VBAT->PWR_CTRL &= ~VBAT_PWR_CTRL_UPHY_PWR_MASK;
    VBAT->PWR_CTRL &= ~VBAT_PWR_CTRL_UPHY_ISO;
    sys_busy_loop_us(1000);

    {
        extern volatile uint32_t g_clk20m_rc, g_clk20m_err;
        printf("display: 20 MHz clock request at boot -> %lu/%lu, VBAT PWR_CTRL=0x%08lx\n",
               (unsigned long)g_clk20m_rc, (unsigned long)g_clk20m_err,
               (unsigned long)VBAT->PWR_CTRL);
    }

    /* Diagnostics: run the steps CDC200 Initialize performs one by one and
     * report each, so a failure names its step. Repeating them inside the
     * driver afterwards is harmless (each returns OK once initialized). */
    {
        uint32_t htotal = RTE_PANEL_HSYNC_TIME + RTE_PANEL_HBP_TIME + RTE_PANEL_HFP_TIME + LCD_W;
        uint32_t vtotal = RTE_PANEL_VSYNC_LINE + RTE_PANEL_VBP_LINE + RTE_PANEL_VFP_LINE + LCD_H;
        uint32_t pixclk = htotal * vtotal * RTE_CDC200_DPI_FPS;
        uint32_t freq = (uint32_t)(pixclk * (24 * 1.333f / (8 * RTE_ILI9806E_PANEL_DSI_N_LANES)) * 4 + 0.5f);
        pin_steps("reset", &ARM_Driver_GPIO_(BOARD_LCD_RESET_GPIO_PORT), BOARD_LCD_RESET_GPIO_PIN);
        pin_steps("backlight", &ARM_Driver_GPIO_(BOARD_LCD_BL_LED_GPIO_PORT), BOARD_LCD_BL_LED_GPIO_PIN);
        int32_t r = DSI_DPHY_Initialize(freq, RTE_ILI9806E_PANEL_DSI_N_LANES);
        printf("display: DPHY init (%lu Hz, %d lanes) -> %ld\n", (unsigned long)freq,
               RTE_ILI9806E_PANEL_DSI_N_LANES, (long)r);
        r = Driver_MIPI_DSI.Initialize(dsi_callback);
        printf("display: DSI + panel init -> %ld\n", (long)r);
    }

    int32_t ri = Driver_CDC200.Initialize(display_callback);
    if (ri != ARM_DRIVER_OK) {
        printf("display: CDC init failed (%ld)\n", (long)ri);
        return -3;
    }
    if (Driver_CDC200.PowerControl(ARM_POWER_FULL) != ARM_DRIVER_OK) {
        printf("display: CDC power failed\n");
        return -4;
    }
    if (Driver_CDC200.Control(CDC200_CONFIGURE_DISPLAY, (uint32_t)lcd_image) != ARM_DRIVER_OK) {
        printf("display: CDC configure failed\n");
        return -5;
    }
    if (Driver_CDC200.Start() != ARM_DRIVER_OK) {
        printf("display: CDC start failed\n");
        return -6;
    }
    printf("display: %dx%d, %d bytes/pixel, buffer at 0x%08lx\n", LCD_W, LCD_H, PIXEL_BYTES,
           (unsigned long)lcd_image);
    return 0;
}

uint8_t *display_buffer(void)
{
    return fb_back;  /* = the shown buffer unless double buffering is on */
}

void display_enable_double_buffer(void)
{
    memcpy(lcd_image2, lcd_image, sizeof(lcd_image2));
    fb_front = &lcd_image[0][0][0];
    fb_back = &lcd_image2[0][0][0];
    fb_double = 1;
}

int display_width(void)
{
    return LCD_W;
}

int display_height(void)
{
    return LCD_H;
}

int display_bytes_per_pixel(void)
{
    return PIXEL_BYTES;
}

void display_flush(void)
{
    SCB_CleanDCache_by_Addr((uint32_t *)fb_back, (int32_t)sizeof(lcd_image));
    if (!fb_double) return;
    /* Show the finished frame from the next vertical blank, and wait until the
     * controller has switched (it clears the reload request) before the old
     * front buffer is drawn into: no torn or half-drawn frames on screen. */
    Driver_CDC200.Control(CDC200_FRAMEBUF_UPDATE_VSYNC, (uint32_t)fb_back);
    uint32_t t0 = DWT->CYCCNT, limit = SystemCoreClock / 20U;  /* 50 ms */
    while ((CDC->CDC_LAYER_CFG[0].CDC_L_REL_CTRL & (1UL << 1)) && (DWT->CYCCNT - t0) < limit) {}
    uint8_t *t = fb_front;
    fb_front = fb_back;
    fb_back = t;
}
