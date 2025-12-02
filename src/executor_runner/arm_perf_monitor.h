/* Copyright 2024-2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined (__RTE__)
#include "RTE_Components.h"
#include CMSIS_device_header
#else
#include <SSE300MPS3.h>
#endif // defined (__RTE__)

#include <m-profile/armv8m_pmu.h>

void StartMeasurements();
void StopMeasurements(int num_inferences);
