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
 * DevKit-E8 user input for the runner: the SW2 joystick and a non-blocking
 * console read.
 *
 * SW2 is a five-way switch on LPGPIO (port 15) pins 0-4, active low with the
 * pad pull-ups (schematic 220-00319-B sheet 12, "Use Internal GPIO Pullups").
 * The direction names follow the pack's own VIO driver for this board
 * (Boards/DevKit-e8/Drivers/vio_DevKit-E8.c): A = left, B = up, C = down,
 * D = right, CENTER = select. The Conductor GPIO configuration of the layer
 * (RTE/BSP/.../gpios.h, applied by board_gpios_config()) makes the pins
 * inputs with debounce and falling-edge interrupt detection; the interrupt
 * is never enabled in the NVIC, so the LPGPIO interrupt status register works
 * as a press latch: a press is recorded even while the CPU is busy in an
 * inference, and board_buttons() collects and clears it.
 *---------------------------------------------------------------------------*/

#include <stdint.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#include "board_defs.h"

#define BUTTON_LEFT   (1u << 0)
#define BUTTON_RIGHT  (1u << 1)
#define BUTTON_UP     (1u << 2)
#define BUTTON_DOWN   (1u << 3)
#define BUTTON_SELECT (1u << 4)

#define LPGPIO ((volatile GPIO_Type *) LPGPIO_BASE)

static unsigned map_pins(uint32_t pins)
{
    unsigned m = 0u;
    if (pins & (1u << BOARD_JOY_SW_A_GPIO_PIN))      m |= BUTTON_LEFT;
    if (pins & (1u << BOARD_JOY_SW_D_GPIO_PIN))      m |= BUTTON_RIGHT;
    if (pins & (1u << BOARD_JOY_SW_B_GPIO_PIN))      m |= BUTTON_UP;
    if (pins & (1u << BOARD_JOY_SW_C_GPIO_PIN))      m |= BUTTON_DOWN;
    if (pins & (1u << BOARD_JOY_SW_CENTER_GPIO_PIN)) m |= BUTTON_SELECT;
    return m;
}

/*
  Return the SW2 directions pressed since the previous call as a bit mask
  (BUTTON_LEFT, BUTTON_RIGHT, BUTTON_UP, BUTTON_DOWN, BUTTON_SELECT) and
  clear them.
*/
unsigned board_buttons(void)
{
    uint32_t pressed = LPGPIO->GPIO_INTSTATUS & 0x1Fu;
    if (pressed) {
        LPGPIO->GPIO_PORTA_EOI = pressed;
    }
    return map_pins(pressed);
}

/*
  Return the SW2 directions currently held down (same bits), for the debugger
  and for tests of the wiring.
*/
unsigned board_buttons_held(void)
{
    return map_pins(~LPGPIO->GPIO_EXT_PORTA & 0x1Fu);
}

/*
  Return the next console character, or -1 when none is waiting. The pack's
  stdin_getchar() blocks until a character arrives, which would stop the
  runner from polling the joystick; this reads the console UART's receive
  register directly (UART4 on the PRG USB connector, set up by stdio_init()).
*/
int board_console_poll(void)
{
    volatile UART_Type *uart = (volatile UART_Type *) UART4_BASE;
    if (uart->UART_LSR & 0x01u) {               /* LSR[0]: receiver data ready */
        return (int) (uart->UART_RBR & 0xFFu);
    }
    return -1;
}
