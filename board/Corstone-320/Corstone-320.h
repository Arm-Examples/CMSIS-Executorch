/*
 * Copyright 2026 Arm Limited and/or its affiliates.
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
 */

#ifndef CORSTONE_320_H_
#define CORSTONE_320_H_

#include "Driver_USART.h"

#define CMSIS_DRIVER_USART  0
#define RETARGET_STDIO_UART 0

extern ARM_DRIVER_USART Driver_USART0;

#ifdef CMSIS_shield_header
#include CMSIS_shield_header
#endif

#endif /* CORSTONE_320_H_ */
