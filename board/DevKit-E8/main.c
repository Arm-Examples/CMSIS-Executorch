/*---------------------------------------------------------------------------
 * Copyright (c) 2025-2026 Arm Limited (or its affiliates). All rights reserved.
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
 * DevKit-E8 (M55_HP) board bring-up for the Ethos-U85 face generator. Derived
 * from the pack's Boards/DevKit-e8/Layers/M55_HP/main.c without the USB,
 * Ethernet and VIO initialisation; the MIPI DPHY is powered for the display.
 *---------------------------------------------------------------------------*/

#include "RTE_Components.h"
#include CMSIS_device_header

#include "board_config.h"
#include "main.h"

#include "se_services_port.h"

/* VBAT PWR_CTRL fields: power and isolation of the MIPI DPHYs and their PLL */
#define VBAT_PWR_CTRL_TX_DPHY_PWR_MASK        (1U <<  0)
#define VBAT_PWR_CTRL_TX_DPHY_ISO             (1U <<  1)
#define VBAT_PWR_CTRL_RX_DPHY_PWR_MASK        (1U <<  4)
#define VBAT_PWR_CTRL_RX_DPHY_ISO             (1U <<  5)
#define VBAT_PWR_CTRL_DPHY_PLL_PWR_MASK       (1U <<  8)
#define VBAT_PWR_CTRL_DPHY_PLL_ISO            (1U <<  9)
#define VBAT_PWR_CTRL_DPHY_VPH_1P8_PWR_BYP_EN (1U << 12)

/* Power up the MIPI DPHYs (the display's DSI link needs the TX DPHY and PLL) */
static void vbat_init(void)
{
    VBAT->PWR_CTRL &= ~(VBAT_PWR_CTRL_TX_DPHY_PWR_MASK | VBAT_PWR_CTRL_RX_DPHY_PWR_MASK |
                        VBAT_PWR_CTRL_DPHY_PLL_PWR_MASK | VBAT_PWR_CTRL_DPHY_VPH_1P8_PWR_BYP_EN);
    VBAT->PWR_CTRL &= ~(VBAT_PWR_CTRL_TX_DPHY_ISO | VBAT_PWR_CTRL_RX_DPHY_ISO |
                        VBAT_PWR_CTRL_DPHY_PLL_ISO);
}

int main(void)
{
    /* The pack's startup enables the caches without invalidating them, but
       after a flash update and a warm reset from the debugger the data cache
       can still hold lines of the previous image, and the runtime then reads
       a corrupted program. Drop them before anything uses the model. */
    SCB_CleanInvalidateDCache();
    SCB_InvalidateICache();

    /* Apply the Conductor pin configuration (includes the UART4 console pins) */
    board_pins_config();

    /* Apply the Conductor GPIO configuration */
    board_gpios_config();

    /* Bring up the Secure Enclave services (MHU link to the SE) */
    se_services_port_init();

    /* Request the clocks the SE has to enable for this core (HFOSC and
       100 MHz also feed the display controller and the DSI link) */
    board_clocks_config(CLKEN_HFOSC_MASK | CLKEN_CLK_100M_MASK);

    /* Power up the MIPI DPHY for the display */
    vbat_init();

    /* Initialize STDIO (UART4 on the PRG USB connector) */
    stdio_init();

    #if defined(ETHOSU_ARCH)
    /* Initialize Ethos NPU */
    ethos_setup();
    #endif

    return app_main();
}
