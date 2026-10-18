/*---------------------------------------------------------------------------
 * Copyright (c) 2020-2025 Arm Limited (or its affiliates).
 * All rights reserved.
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
 *---------------------------------------------------------------------------*/

#ifndef MAIN_H__
#define MAIN_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Prototypes */
extern int app_main (void);
extern int stdio_init (void);
/* Edge_Deploy: non-blocking read of received UART bytes (retarget_stdio.c; none on the FVP)
   and the count of receive errors. */
extern uint32_t stdio_rx_read (uint8_t *buf, uint32_t max);
extern uint32_t stdio_rx_overruns (void);
/* DWT cycle stamp of the last UART RX interrupt that stored bytes (0 if none / no UART), and
   the count of output lines dropped because the transmitter did not free up within 20 ms. */
extern uint32_t stdio_rx_cycles(void);
extern uint32_t stdio_tx_faults(void);

#if defined(ETHOSU_ARCH)
extern void ethos_setup (void);
#endif

#ifdef __cplusplus
}
#endif

#endif
