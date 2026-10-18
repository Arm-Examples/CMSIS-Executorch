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
#include "VoiceControl.hpp"

#if ENABLE_VOICE && defined(MLEK_FWK_EXECUTORCH)

#include "hal.h"
#include "timer_alif.h"
#include "mlek/fwk/executorch/DsCnnKwsModel.hpp"
#include "mlek/log/log_macros.h"
#include "mlek/use_case/kws/KwsProcessing.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <malloc.h>

#ifndef KWS_ACTIVATION_BUF_SZ
#define KWS_ACTIVATION_BUF_SZ 0x00020000
#endif
#ifndef KWS_THRESHOLD
#define KWS_THRESHOLD 0.7f
#endif

namespace arm {
namespace app {
namespace kws {
    /* Generated from usecase.cmake (generate_model_code into generated/.../src/kws). */
    extern const uint8_t* GetModelPointer();
    extern size_t GetModelLen();
    extern const char* const g_kwsLabels[];
    extern const int g_numKwsLabels;
} /* namespace kws */
} /* namespace app */
} /* namespace arm */

/* Debug state readable over J-Link without halting the core. */
extern "C" {
volatile int32_t  g_kwsInitState;     /* 0 = not started, 1 = running, <0 = failed step */
volatile int32_t  g_kwsLastIdx = -1;  /* last keyword that fired (label index), -1 = none */
volatile int32_t  g_kwsLastConf;      /* its averaged probability x1000 */
volatile uint32_t g_kwsCount;         /* keywords fired since boot */
volatile int32_t  g_kwsTopIdx = -1;   /* top-1 of the averaged softmax on the last stride */
volatile int32_t  g_kwsTopProb;       /* its probability x1000 (includes _silence_/_unknown_) */
volatile uint32_t g_kwsStrides;       /* strides processed */
volatile uint32_t g_kwsCaptured;      /* strides completed by the audio HAL (interrupt) */
volatile uint32_t g_kwsGaps;          /* times the loop fell >= 2 strides behind (window rebuilt) */
volatile uint32_t g_kwsAudioErrors;   /* audio HAL errors reported to the callback */
volatile uint32_t g_kwsPerfUs[3];     /* ~2 s averages: MFCC (incl. gain), NPU, post (us) */
volatile uint32_t g_kwsHeapUsed;      /* malloc'd bytes after voice init (heap is ~92 KiB) */
volatile float    g_kwsLogits[16];    /* raw logits of the last stride (compare with host) */
volatile float    g_kwsSelfTestA[16]; /* logits for x[i] = ((i*37)%41)-30, see kws_host_check.py */
}

namespace alif {
namespace app {
namespace voice {

namespace {
    using KwsModel = arm::app::fwk::et::DsCnnKwsModel;

    constexpr int kWindow = 16000; /* 1 s at 16 kHz */
    constexpr int kStride = 8000;  /* 0.5 s */
    constexpr int kSlots  = 3;     /* capture ring */
    constexpr int kAvg    = 2;     /* windows averaged: the newest two 1 s windows (1.5 s span) */
    constexpr int kMaxLabels      = 16;
    constexpr int kRefractory     = 2;  /* strides after a fire during which nothing fires */
    constexpr uint32_t kPerfStrides = 4;

    uint8_t kwsArena[KWS_ACTIVATION_BUF_SZ] __attribute__((aligned(16)));

    /* The HAL writes float16 into a slot, audio_preprocessing() turns it into int16 in place. */
    int16_t slots[kSlots][kStride] __attribute__((aligned(32)));
    int16_t window[kWindow] __attribute__((aligned(32)));

    volatile uint32_t capSeq  = 0; /* strides completed; stride n (1-based) is in slot (n-1)%3 */
    volatile int      capFill = 0; /* slot being filled */

    KwsModel model;
    arm::app::KwsPreProcess* preProcess = nullptr;
    bool ready = false;

    uint32_t doneSeq   = 0;  /* last stride folded into the window */
    size_t   mfccIndex = 0;  /* 0 = no MFCC cache (first window or after a gap) */
    float    probHist[kAvg][kMaxLabels];
    int      histCount = 0;
    int      histPos   = 0;
    int      sinceFire = 1000;
    int      numLabels = 0;
    int      silenceIdx = -1, unknownIdx = -1;

    uint64_t mfccCyc = 0, npuCyc = 0, postCyc = 0;
    uint32_t perfN = 0;

    /* Interrupt context (audio_alif voice_data_cb, last thing it does): re-arm into the next slot
     * straight away so no samples are lost while the camera loop is busy. */
    void AudioDone(uint32_t err)
    {
        if (err) {
            ++g_kwsAudioErrors;
        }
        const int next = (capFill + 1) % kSlots;
        capFill = next;
        capSeq  = capSeq + 1;
        g_kwsCaptured = capSeq;
        hal_get_audio_data(slots[next], kStride);
    }

    int16_t* SlotOf(uint32_t seq) /* seq >= 1 */
    {
        return slots[(seq - 1) % kSlots];
    }
} /* namespace */

const char* Label(int i)
{
    return (i >= 0 && i < numLabels) ? arm::app::kws::g_kwsLabels[i] : "?";
}

bool IsReady()
{
    return ready;
}

uint32_t MfccUs()
{
    return g_kwsPerfUs[0];
}

uint32_t NpuUs()
{
    return g_kwsPerfUs[1];
}

bool Init()
{
    arm::app::fwk::iface::MemoryRegion modelMem{
        const_cast<uint8_t*>(arm::app::kws::GetModelPointer()), arm::app::kws::GetModelLen()};
    arm::app::fwk::iface::MemoryRegion computeMem{kwsArena, sizeof(kwsArena)};

    info("Voice: loading KWS program (%zu bytes), arena %zu bytes\n",
         static_cast<size_t>(modelMem.size), sizeof(kwsArena));
    if (!model.Init(computeMem, modelMem)) {
        printf_err("Voice: KWS model init failed\n");
        g_kwsInitState = -1;
        return false;
    }

    auto in  = model.GetInputTensor(0);
    auto out = model.GetOutputTensor(0);
    const auto shape = in->Shape();
    if (in->Type() != arm::app::fwk::iface::TensorType::FP32 ||
        out->Type() != arm::app::fwk::iface::TensorType::FP32 || shape.size() < 4) {
        printf_err("Voice: expected a float [1,1,49,10] -> float logits DS-CNN program\n");
        g_kwsInitState = -2;
        return false;
    }
    numLabels = std::min({static_cast<int>(out->GetNumElements()),
                          arm::app::kws::g_numKwsLabels, kMaxLabels});
    for (int i = 0; i < numLabels; ++i) {
        if (std::strcmp(arm::app::kws::g_kwsLabels[i], "_silence_") == 0) {
            silenceIdx = i;
        } else if (std::strcmp(arm::app::kws::g_kwsLabels[i], "_unknown_") == 0) {
            unknownIdx = i;
        }
    }

    preProcess = new arm::app::KwsPreProcess(in, shape[KwsModel::ms_inputColsIdx],
                                             shape[KwsModel::ms_inputRowsIdx],
                                             arm::app::kws::g_FrameLength,
                                             arm::app::kws::g_FrameStride);
    if (preProcess->m_audioDataWindowSize != static_cast<size_t>(kWindow) ||
        preProcess->m_audioDataStride != static_cast<size_t>(kStride)) {
        printf_err("Voice: MFCC window %zu / stride %zu, expected %d / %d\n",
                   preProcess->m_audioDataWindowSize, preProcess->m_audioDataStride, kWindow,
                   kStride);
        g_kwsInitState = -3;
        return false;
    }

    /* Self-test A (same as alif_kws / kws_host_check.py): model only, deterministic input. */
    {
        float* x = in->GetData<float>();
        for (size_t i = 0; i < in->GetNumElements(); ++i) {
            x[i] = static_cast<float>(static_cast<int>((i * 37) % 41) - 30);
        }
        if (!model.RunInference()) {
            printf_err("Voice: KWS self-test inference failed\n");
            g_kwsInitState = -6;
            return false;
        }
        const float* y = out->GetData<float>();
        printf("KWS_SELFTEST_A logits:");
        for (int i = 0; i < numLabels; ++i) {
            g_kwsSelfTestA[i] = y[i];
            printf(" %.4f", static_cast<double>(y[i]));
        }
        printf("\n");
    }

    std::fill(window, window + kWindow, 0);
    const int err = hal_audio_alif_init(16000);
    if (err) {
        printf_err("Voice: hal_audio_alif_init failed: %d\n", err);
        g_kwsInitState = -4;
        return false;
    }
    hal_set_audio_callback(AudioDone);
    capFill = 0;
    if (hal_get_audio_data(slots[0], kStride) != 0) {
        printf_err("Voice: could not start audio capture\n");
        g_kwsInitState = -5;
        return false;
    }

    g_kwsHeapUsed  = mallinfo().uordblks;
    g_kwsInitState = 1;
    ready = true;
    info("Voice: KWS ready, %d labels, threshold %.2f, heap in use %" PRIu32 " bytes\n",
         numLabels, static_cast<double>(KWS_THRESHOLD), g_kwsHeapUsed);
    return true;
}

bool Poll(int& keyword, float& conf)
{
    keyword = -1;
    conf    = 0.0f;
    if (!ready) {
        return false;
    }
    const uint32_t seq = capSeq; /* snapshot; the ISR only ever increments it */
    if (seq == doneSeq) {
        return false;
    }

    const uint32_t t0 = Get_SysTick_Cycle_Count32();
    if (seq == doneSeq + 1) {
        /* Normal case: slide the window by one stride. */
        int16_t* s = SlotOf(seq);
        hal_audio_alif_preprocessing(s, kStride);
        std::memmove(window, window + kStride, kStride * sizeof(int16_t));
        std::memcpy(window + kStride, s, kStride * sizeof(int16_t));
        mfccIndex = (doneSeq == 0) ? 0 : mfccIndex + 1;
    } else {
        /* Fell >= 2 strides behind: rebuild the window from the two newest strides. They are both
         * intact (the HAL is filling slot seq % 3) and both still float16. */
        ++g_kwsGaps;
        int16_t* a = SlotOf(seq - 1);
        int16_t* b = SlotOf(seq);
        hal_audio_alif_preprocessing(a, kStride);
        hal_audio_alif_preprocessing(b, kStride);
        std::memcpy(window, a, kStride * sizeof(int16_t));
        std::memcpy(window + kStride, b, kStride * sizeof(int16_t));
        mfccIndex = 0;
        histCount = 0;
    }
    doneSeq = seq;

    /* MFCC: with index > 0 the first half of the frames comes from the cache of the previous
     * window (KwsPreProcess), so only the newest 0.5 s is transformed. */
    if (!preProcess->DoPreProcess(window, mfccIndex)) {
        printf_err("Voice: MFCC failed\n");
        return false;
    }
    const uint32_t t1 = Get_SysTick_Cycle_Count32();
    if (!model.RunInference()) {
        printf_err("Voice: KWS inference failed\n");
        return false;
    }
    const uint32_t t2 = Get_SysTick_Cycle_Count32();

    /* Softmax, then the mean over the newest kAvg windows. */
    const float* logits = model.GetOutputTensor(0)->GetData<float>();
    float* p = probHist[histPos];
    float maxLogit = -1e30f;
    for (int i = 0; i < numLabels; ++i) {
        maxLogit = std::max(maxLogit, logits[i]);
        g_kwsLogits[i] = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < numLabels; ++i) {
        p[i] = std::exp(logits[i] - maxLogit);
        sum += p[i];
    }
    for (int i = 0; i < numLabels; ++i) {
        p[i] /= sum;
    }
    histPos   = (histPos + 1) % kAvg;
    histCount = std::min(histCount + 1, kAvg);

    int best = -1;
    float bestProb = 0.0f;
    for (int i = 0; i < numLabels; ++i) {
        float m = 0.0f;
        for (int k = 0; k < histCount; ++k) {
            m += probHist[(histPos - 1 - k + kAvg) % kAvg][i];
        }
        m /= histCount;
        if (best < 0 || m > bestProb) {
            best     = i;
            bestProb = m;
        }
    }
    g_kwsTopIdx  = best;
    g_kwsTopProb = static_cast<int32_t>(1000.0f * bestProb);

    /* Fire on a confident command word, at most once per kRefractory + 1 strides: one utterance
     * shows up in two overlapping windows and must not fire twice. */
    ++sinceFire;
    const bool isCommand = best >= 0 && best != silenceIdx && best != unknownIdx;
    if (isCommand && bestProb >= KWS_THRESHOLD && sinceFire > kRefractory) {
        keyword   = best;
        conf      = bestProb;
        sinceFire = 0;
        g_kwsLastIdx  = best;
        g_kwsLastConf = static_cast<int32_t>(1000.0f * bestProb);
        ++g_kwsCount;
        info("Voice: \"%s\" (%" PRId32 "/1000), keyword #%" PRIu32 "\n", Label(best),
             g_kwsLastConf, g_kwsCount);
    }
    const uint32_t t3 = Get_SysTick_Cycle_Count32();

    ++g_kwsStrides;
    mfccCyc += t1 - t0;
    npuCyc += t2 - t1;
    postCyc += t3 - t2;
    if (++perfN == kPerfStrides) {
        const float usPerCycle = 1e6f / SystemCoreClock;
        g_kwsPerfUs[0] = static_cast<uint32_t>(mfccCyc * usPerCycle / perfN);
        g_kwsPerfUs[1] = static_cast<uint32_t>(npuCyc * usPerCycle / perfN);
        g_kwsPerfUs[2] = static_cast<uint32_t>(postCyc * usPerCycle / perfN);
        mfccCyc = npuCyc = postCyc = 0;
        perfN = 0;
    }
    return true;
}

} /* namespace voice */
} /* namespace app */
} /* namespace alif */

#endif /* ENABLE_VOICE && MLEK_FWK_EXECUTORCH */
