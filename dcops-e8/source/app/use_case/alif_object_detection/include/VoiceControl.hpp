/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#ifndef ALIF_OBJ_DET_VOICE_CONTROL_HPP
#define ALIF_OBJ_DET_VOICE_CONTROL_HPP

/* Keyword spotting next to the camera classifier (alif_object_detection_ENABLE_VOICE).
 *
 * Audio is captured in the background: the audio_alif HAL fills one 0.5 s stride at a time and
 * its completion callback (interrupt context) immediately re-arms the capture into the next slot
 * of a 3-slot ring, so the camera loop never waits for audio. Poll() is called once per camera
 * frame; when a new stride has completed it runs gain + MFCC (M55) and the DS-CNN (Ethos-U85) on
 * the latest 1 s window, otherwise it returns at once. No RTOS and no MHU messages. */

#ifndef ENABLE_VOICE
#define ENABLE_VOICE 0
#endif

#if ENABLE_VOICE
#include <cstdint>

namespace alif {
namespace app {
namespace voice {

    /** Loads the KWS program into its own arena and starts the microphone.
     *  @return false if voice is unavailable (the camera app keeps running without it). */
    bool Init();

    /** @return true once Init() succeeded. */
    bool IsReady();

    /**
     * @brief Non-blocking. If a new 0.5 s stride is ready, runs MFCC + KWS on the last 1 s.
     * @param[out] keyword  index of a keyword that fired on this stride, or -1.
     * @param[out] conf     its averaged softmax probability (0-1).
     * @return true if KWS ran on this call (so the UI may want to refresh), false otherwise.
     */
    bool Poll(int& keyword, float& conf);

    /** Label of keyword index i ("go", "stop", ...), or "?" when out of range. */
    const char* Label(int i);

    /** ~2 s averages of the per-stride stages, in microseconds. */
    uint32_t MfccUs();
    uint32_t NpuUs();

} /* namespace voice */
} /* namespace app */
} /* namespace alif */
#endif /* ENABLE_VOICE */

#endif /* ALIF_OBJ_DET_VOICE_CONTROL_HPP */
