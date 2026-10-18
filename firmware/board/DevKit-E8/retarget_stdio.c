/*---------------------------------------------------------------------------
 * Copyright (c) 2025 Arm Limited (or its affiliates).
 * All rights reserved.
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
 *      Name:    retarget_stdio.c
 *      Purpose: stdio and the EEG stream on UART4, interrupt driven
 *
 * Edge_Deploy: UART4 (PRG USB, SW4 -> UART4) carries both directions of the stream at
 * UART_BAUDRATE (921600 8N1): the host's 49-byte frames at 250 Hz in (122.5 kbit/s, more than
 * 115200 can carry) and the JSON/diagnostic lines out. The starter's polled Alif retarget is
 * replaced by the CMSIS USART driver in interrupt mode (RTE_UART4_BLOCKING_MODE_ENABLE 0):
 *  - RX: two 2 KiB buffers received back to back; the completion callback restarts the
 *    receive on the other buffer at once, the main loop drains them with stdio_rx_read().
 *    Each buffer holds 42 ms of stream, far more than one inference takes.
 *  - TX: stdout_putchar collects a line; '\n' (or a full buffer) hands it to Send() and
 *    returns, with a second buffer filling meanwhile. If the previous line has not left after
 *    TX_TIMEOUT_MS (DWT deadline), the send is aborted, the new line dropped and tx_faults
 *    counted (reported by the main loop as "# ... tx_fault n").
 *  - RX timestamp: with Arm Compiler 6 the driver's UART4_IRQHandler is wrapped
 *    ($Sub$$/$Super$$) to record the DWT cycle count of the last interrupt that stored bytes;
 *    stdio_rx_cycles() returns it (latency reference for the JSON lat_us).
 * Untested without a board.
 *---------------------------------------------------------------------------*/

#include <stdint.h>
#include <string.h>

#include "RTE_Components.h"
#include CMSIS_device_header
#include CMSIS_target_header

#include "main.h"

/* Compile-time configuration */
#ifndef UART_BAUDRATE
#define UART_BAUDRATE 921600
#endif

#define RX_BUF_SIZE 2048u
#define TX_BUF_SIZE 256u
#define TX_TIMEOUT_MS 20u

/* Reference to the underlying USART driver */
#define ptrUSART (&ARM_Driver_USART_(RETARGET_STDIO_UART))

static uint8_t           rx_buf[2][RX_BUF_SIZE];
static volatile uint32_t rx_active;     /* buffer the driver is filling             */
static volatile uint32_t rx_done[2];    /* buffer completely received, not consumed */
static volatile uint32_t rx_errors;     /* overflow/framing/parity events, lost buffers */
static volatile uint32_t rx_resync;     /* a buffer was overwritten: consumer restarts */
static uint32_t          rx_cons_buf;   /* consumer position (main loop only)       */
static uint32_t          rx_cons_pos;

static volatile uint32_t rx_last_cycles; /* DWT stamp of the last RX interrupt with data */
static volatile uint32_t tx_faults;

static uint8_t           tx_buf[2][TX_BUF_SIZE];
static uint32_t          tx_fill_buf;
static uint32_t          tx_fill_len;
static int               uart_ready;

static void usart_event(uint32_t event)
{
    if (event & ARM_USART_EVENT_RECEIVE_COMPLETE) {
        uint32_t done = rx_active;
        uint32_t next = done ^ 1u;
        rx_done[done] = 1u;
        if (rx_done[next]) {
            /* The main loop has not drained the other buffer: it is overwritten and the
               consumer restarts at the buffer just completed. */
            rx_errors++;
            rx_done[next] = 0u;
            rx_resync     = 1u;
        }
        rx_active = next;
        rx_last_cycles = DWT->CYCCNT;
        (void)ptrUSART->Receive(rx_buf[next], RX_BUF_SIZE);
    }
    if (event & (ARM_USART_EVENT_RX_OVERFLOW | ARM_USART_EVENT_RX_FRAMING_ERROR |
                 ARM_USART_EVENT_RX_PARITY_ERROR | ARM_USART_EVENT_RX_BREAK)) {
        rx_errors++;
    }
}

#if defined(__ARMCC_VERSION) && (RETARGET_STDIO_UART == 4)
/* Linker patch (armlink $Sub$$/$Super$$): run the driver's handler, then stamp the time if it
   stored received bytes (the receive count moved or a buffer completed). */
extern void $Super$$UART4_IRQHandler(void);
void $Sub$$UART4_IRQHandler(void);
void $Sub$$UART4_IRQHandler(void)
{
    uint32_t count  = ptrUSART->GetRxCount();
    uint32_t active = rx_active;
    $Super$$UART4_IRQHandler();
    if (ptrUSART->GetRxCount() != count || rx_active != active) {
        rx_last_cycles = DWT->CYCCNT;
    }
}
#endif

uint32_t stdio_rx_cycles(void)
{
#if defined(__ARMCC_VERSION) && (RETARGET_STDIO_UART == 4)
    return rx_last_cycles;
#else
    return DWT->CYCCNT;   /* no ISR stamp: the time the main loop reads the bytes */
#endif
}

uint32_t stdio_tx_faults(void)
{
    return tx_faults;
}

/**
  Initialize stdio

  \return          0 on success, or -1 on error.
*/
int stdio_init(void)
{
    /* DWT cycle counter for the TX timeout and the RX stamps (app_main restarts it). */
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;

    if (ptrUSART->Initialize(usart_event) != ARM_DRIVER_OK) {
        return -1;
    }

    if (ptrUSART->PowerControl(ARM_POWER_FULL) != ARM_DRIVER_OK) {
        return -1;
    }

    if (ptrUSART->Control(ARM_USART_MODE_ASYNCHRONOUS | ARM_USART_DATA_BITS_8 |
                              ARM_USART_PARITY_NONE | ARM_USART_STOP_BITS_1 |
                              ARM_USART_FLOW_CONTROL_NONE,
                          UART_BAUDRATE) != ARM_DRIVER_OK) {
        return -1;
    }

    if (ptrUSART->Control(ARM_USART_CONTROL_TX, 1U) != ARM_DRIVER_OK) {
        return -1;
    }

    if (ptrUSART->Control(ARM_USART_CONTROL_RX, 1U) != ARM_DRIVER_OK) {
        return -1;
    }

    rx_active   = 0u;
    rx_done[0]  = 0u;
    rx_done[1]  = 0u;
    rx_resync   = 0u;
    rx_cons_buf = 0u;
    rx_cons_pos = 0u;
    if (ptrUSART->Receive(rx_buf[0], RX_BUF_SIZE) != ARM_DRIVER_OK) {
        return -1;
    }

    uart_ready = 1;
    return 0;
}

/**
  Copy up to max received bytes (non-blocking).

  \return          number of bytes copied
*/
uint32_t stdio_rx_read(uint8_t *buf, uint32_t max)
{
    uint32_t copied = 0u;

    while (copied < max) {
        uint32_t avail;
        uint32_t primask = __get_PRIMASK();
        __disable_irq();
        if (rx_resync) {
            rx_resync   = 0u;
            rx_cons_buf = rx_active ^ 1u;
            rx_cons_pos = 0u;
        }
        if (rx_done[rx_cons_buf]) {
            avail = RX_BUF_SIZE;
        } else {
            avail = ptrUSART->GetRxCount();   /* rx_cons_buf is the buffer being filled */
        }
        __set_PRIMASK(primask);

        if (avail > RX_BUF_SIZE) {
            avail = RX_BUF_SIZE;
        }
        if (avail > rx_cons_pos) {
            uint32_t n = avail - rx_cons_pos;
            if (n > max - copied) {
                n = max - copied;
            }
            memcpy(&buf[copied], &rx_buf[rx_cons_buf][rx_cons_pos], n);
            copied += n;
            rx_cons_pos += n;
        }

        if (rx_cons_pos >= RX_BUF_SIZE) {
            /* Finished this buffer: release it and continue in the other one. */
            primask = __get_PRIMASK();
            __disable_irq();
            rx_done[rx_cons_buf] = 0u;
            __set_PRIMASK(primask);
            rx_cons_buf ^= 1u;
            rx_cons_pos = 0u;
            continue;
        }
        break;
    }
    return copied;
}

uint32_t stdio_rx_overruns(void)
{
    return rx_errors;
}

static void tx_flush(void)
{
    if (tx_fill_len == 0u || !uart_ready) {
        return;
    }
    /* Wait for the previous line (the other buffer) to leave, at most TX_TIMEOUT_MS. */
    uint32_t start = DWT->CYCCNT;
    uint32_t limit = (SystemCoreClock / 1000u) * TX_TIMEOUT_MS;
    while (ptrUSART->GetStatus().tx_busy) {
        if ((uint32_t)(DWT->CYCCNT - start) > limit) {
            (void)ptrUSART->Control(ARM_USART_ABORT_SEND, 0U);
            tx_faults++;
            tx_fill_len = 0u;   /* drop this line */
            return;
        }
    }
    (void)ptrUSART->Send(tx_buf[tx_fill_buf], tx_fill_len);
    tx_fill_buf ^= 1u;
    tx_fill_len = 0u;
}

static int put_char(int ch)
{
    if (!uart_ready) {
        return -1;
    }
    tx_buf[tx_fill_buf][tx_fill_len++] = (uint8_t)ch;
    if (ch == '\n' || ch == 0x04 || tx_fill_len == TX_BUF_SIZE) {
        tx_flush();
    }
    return ch;
}

/**
  Put a character to the stdout

  \param[in]   ch  Character to output
  \return          The character written, or -1 on write error.
*/
int stdout_putchar(int ch)
{
    return put_char(ch);
}

/**
  Put a character to the stderr

  \param[in]   ch  Character to output
  \return          The character written, or -1 on write error.
*/
int stderr_putchar(int ch)
{
    return put_char(ch);
}

/**
  Get a character from stdin (blocking; the firmware reads the stream with stdio_rx_read).

  \return     The next character from the input, or -1 on read error.
*/
int stdin_getchar(void)
{
    uint8_t c;
    if (!uart_ready) {
        return -1;
    }
    while (stdio_rx_read(&c, 1u) == 0u) {
    }
    return c;
}
