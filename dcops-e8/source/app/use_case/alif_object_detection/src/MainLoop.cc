/* This file was ported to work on Alif Semiconductor devices. */

/* Copyright (C) 2023 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/*
 * Copyright (c) 2022, 2026 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "hal.h"                                /* Brings in platform definitions. */
#include "UseCaseHandler.hpp"                   /* Handlers for different user options. */
#include "UseCaseCommonUtils.hpp"               /* Utils functions. */
#include "mlek/log/log_macros.h"                /* Logging functions */
#include "BufAttributes.hpp"                    /* Buffer attributes to be applied */
#include "VoiceControl.hpp"                     /* Optional keyword spotting (ENABLE_VOICE) */

namespace arm {
namespace app {
    static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
    namespace object_detection {
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
    } /* namespace object_detection */
} /* namespace app */
} /* namespace arm */

void MainLoop()
{
    init_trigger_rx();

    alif::app::FaceModel model;  /* Model wrapper object (BlazeFace or YOLO Fastest). */

    arm::app::fwk::iface::MemoryRegion modelMem{arm::app::object_detection::GetModelPointer(),
                                                arm::app::object_detection::GetModelLen()};
    arm::app::fwk::iface::MemoryRegion computeMem{arm::app::tensorArena,
                                                  sizeof(arm::app::tensorArena)};

    /* Load the model. */
    if (!model.Init(computeMem, modelMem)) {
        printf_err("Failed to initialise model\n");
        return;
    }

    if (!alif::app::ObjectDetectionInit(model)) {
        printf_err("Failed to initialise use case handler\n");
        return;
    }

#if ENABLE_VOICE && defined(MLEK_FWK_EXECUTORCH)
    /* Second ExecuTorch program (keyword spotter) with its own arena; the camera app keeps
     * running if it fails. */
    if (!alif::app::voice::Init()) {
        printf_err("Voice control unavailable, continuing with the camera only\n");
    }
#endif

    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;

    arm::app::Profiler profiler{"object_detection"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<arm::app::fwk::iface::Model&>("model", model);

    /* Loop. */
    do {
        alif::app::ObjectDetectionHandler(caseContext);
    } while (1);
}
