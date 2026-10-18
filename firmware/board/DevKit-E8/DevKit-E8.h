/*---------------------------------------------------------------------------
 * Copyright (c) 2025-2026 Arm Limited (or its affiliates).
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
 *
 * CMSIS target header for the headless DevKit-E8 layer: only the USART used
 * for stdio. See the pack's Boards/DevKit-e8/Layers/M55_HP/DevKit-E8.h for
 * the full board pin-out.
 *---------------------------------------------------------------------------*/

#ifndef DEVKIT_E8_H_
#define DEVKIT_E8_H_

#include "Driver_USART.h"

// CMSIS Driver instances of Board peripherals
#define CMSIS_DRIVER_USART   4  // CMSIS Driver USART instance number (PRG USB, J3)

// Retarget stdio to CMSIS UART
#define RETARGET_STDIO_UART  4

// CMSIS Drivers
extern ARM_DRIVER_USART   Driver_USART4;          /* PRG USB          */

#endif /* DEVKIT_E8_H_ */
