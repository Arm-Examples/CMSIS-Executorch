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
 * DevKit-E8 (M55_HP) board bring-up for a headless Ethos-U85 runner. Derived
 * from the pack's Boards/DevKit-e8/Layers/M55_HP/main.c without the MIPI,
 * USB, Ethernet and VIO initialisation.
 *---------------------------------------------------------------------------*/

#include "RTE_Components.h"
#include CMSIS_device_header

#include "board_config.h"
#include "main.h"

#include "se_services_port.h"

#ifdef BOARD_BUNNY_DEMO
/* Boot progress on the RGB LEDs (bunny_demo.c), visible without a debugger */
extern void bunny_boot_led(int r, int g, int b);
#define BOOT_LED(r, g, b) bunny_boot_led(r, g, b)
#else
#define BOOT_LED(r, g, b)
#endif

int main(void)
{
    /* Apply the Conductor pin configuration (includes the UART4 console pins) */
    board_pins_config();

    /* Apply the Conductor GPIO configuration */
    board_gpios_config();

    BOOT_LED(1, 0, 0);  /* red: started */

    /* Bring up the Secure Enclave services (MHU link to the SE) */
    se_services_port_init();

    /* Request the clocks the SE has to enable for this core */
#if defined(BOARD_FACE_DEMO) || defined(BOARD_BUNNY_DEMO)
    extern volatile uint32_t g_boot_dbg[4];
    g_boot_dbg[0] = (uint32_t)board_clocks_config(CLKEN_HFOSC_MASK | CLKEN_CLK_100M_MASK);
    g_boot_dbg[1] = SERVICES_heartbeat(se_services_s_handle);
#else
    board_clocks_config(CLKEN_HFOSC_MASK | CLKEN_CLK_100M_MASK);
#endif

#if defined(BOARD_FACE_DEMO) || defined(BOARD_BUNNY_DEMO)
    /* The MIPI DPHY (LCD) also needs the 20 MHz clock; request it here, like
     * the pack's board layer (clock_init), while the SE answers requests. */
    {
        extern volatile uint32_t g_clk20m_rc, g_clk20m_err;
        uint32_t service_error_code = 0U;
        g_clk20m_rc  = SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_CLK_20M, true,
                                                    &service_error_code);
        g_clk20m_err = service_error_code;
        /* The PDM microphones need the 76.8 MHz clock on the E8 (Alif demo_pdm.c) */
        g_boot_dbg[2] = SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_HFOSCx2, true,
                                                     &service_error_code);
    }
#endif

    BOOT_LED(1, 1, 0);  /* yellow: clocks requested */

    /* Initialize STDIO (UART4 on the PRG USB connector) */
    stdio_init();

    #if defined(ETHOSU_ARCH)
    /* Initialize Ethos NPU */
    ethos_setup();
    #endif

    return app_main();
}
