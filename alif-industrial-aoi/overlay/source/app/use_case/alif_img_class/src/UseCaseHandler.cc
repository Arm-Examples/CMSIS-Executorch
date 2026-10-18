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
 * SPDX-FileCopyrightText: Copyright 2021-2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "UseCaseHandler.hpp"

#include "mlek/common/Classifier.hpp"
#include "mlek/common/ImageUtils.hpp"
#include "mlek/use_case/img_class/ImgClassProcessing.hpp"
#include "UseCaseCommonUtils.hpp"
#include "hal.h"
#include "mlek/log/log_macros.h"
#include "AnimalUi.hpp"
#include "AnimalClasses.hpp"

#if defined(MLEK_FWK_TFLM)
#include "mlek/fwk/tflm/MobileNetModel.hpp"
#elif defined(MLEK_FWK_EXECUTORCH)
#include "mlek/fwk/executorch/MobileNetModel.hpp"
#endif

#if defined(GPIO_PROFILING)
#include "board_utils.h"
#endif

#include <cinttypes>

#include "lvgl.h"
#include "lv_port.h"
#include "lv_paint_utils.h"

// Do we get LVGL to zoom the camera image, or do we double it up?
#define USE_LVGL_ZOOM


using ImgClassClassifier = arm::app::Classifier;
using arm::app::fwk::iface::Model;


/** Based on ML framework, set up the model namespace. */
#if defined(MLEK_FWK_TFLM)
    using arm::app::fwk::tflm::MobileNetModel;
#elif defined(MLEK_FWK_EXECUTORCH)
    using arm::app::fwk::et::MobileNetModel;
    namespace arm::app::img_class {
        extern const unsigned int g_numChannels;
        extern const float g_normMean[];
        extern const float g_normStddev[];
    } /* namespace img_class */
#endif /** MLEK_FWK_TFLM or MLEK_FWK_EXECUTORCH */


#define MIMAGE_X 224
#define MIMAGE_Y 224

#ifdef USE_LVGL_ZOOM
#define LIMAGE_X        MIMAGE_X
#define LIMAGE_Y        MIMAGE_Y
#define LV_ZOOM         (2 * 256)
#else
#define LIMAGE_X        (MIMAGE_X * 2)
#define LIMAGE_Y        (MIMAGE_Y * 2)
#define LV_ZOOM         (1 * 256)
#endif


extern "C" {
extern uint32_t tprof1, tprof2, tprof3, tprof4, tprof5;
}

namespace {

lvgl_pixel_t  lvgl_image[LIMAGE_Y][LIMAGE_X] __attribute__((section(".bss.lcd_image_buf")));                      // 448x448x4 = 802,856
};

namespace alif {
namespace app {

    using namespace arm::app;

    bool ClassifyImageInit(MobileNetModel& model)
    {

#if !SKIP_MODEL
        auto inputTensor = model.GetInputTensor(0);
        const auto inputShape = inputTensor->Shape();
        const uint32_t nCols  = inputShape[MobileNetModel::ms_inputColsIdx];
        const uint32_t nRows  = inputShape[MobileNetModel::ms_inputRowsIdx];
#else
        const uint32_t nCols       = MIMAGE_X;
        const uint32_t nRows       = MIMAGE_Y;
#endif

        /* Initialise the camera */
        if (!hal_camera_init()) {
            printf_err("hal_camera_init failed!\n");
            return false;
        }

        auto bCamera = hal_camera_configure(nCols, nRows, HAL_CAMERA_MODE_SINGLE_FRAME, HAL_CAMERA_COLOUR_FORMAT_RGB888);
        if (!bCamera) {
            printf_err("Failed to configure camera.\n");
            return false;
        }

        // Touch shares the camera's initialized I2C callback; initialize it last.
        AnimalUiInit(lvgl_image, sizeof lvgl_image, LIMAGE_X, LIMAGE_Y, LV_ZOOM);
        return true;
    }

    /* Image classification inference handler. */
    bool ClassifyImageHandler(ApplicationContext& ctx)
    {
#if !SKIP_MODEL
        auto& profiler = ctx.Get<Profiler&>("profiler");
        auto& model = ctx.Get<Model&>("model");

        if (!model.IsInited()) {
            printf_err("Model is not initialised! Terminating processing.\n");
            return false;
        }

        auto inputTensor = model.GetInputTensor(0);
        auto outputTensor = model.GetOutputTensor(0);
        const auto inputShape = inputTensor->Shape();
        if (inputShape.empty()) {
            printf_err("Invalid input tensor dims\n");
            return false;
        } else if (inputShape.size() < 4) {
            printf_err("Input tensor dimension should be >= 4\n");
            return false;
        }

#if defined(MLEK_FWK_EXECUTORCH)
        /**
         * For ExecuTorch we typically have input tensors in floating point.
         * In this case, normalisation parameters should be made available
         * with the model. We try to use these here.
         */
        /* Set up pre- and post-processing. */
        std::array<float, ImgClassPreProcess::kNumChannels> normMean{0.f, 0.f, 0.f};
        std::array<float, ImgClassPreProcess::kNumChannels> normStddev{1.f, 1.f, 1.f};

        if (inputTensor->Type() == fwk::iface::TensorType::FP32 ||
            inputTensor->Type() == fwk::iface::TensorType::FP16) {

            /* Check for mismatch in size of channels. */
            if (ImgClassPreProcess::kNumChannels != img_class::g_numChannels) {
                printf_err("Number of channels mismatch in norm parameters\n");
                return false;
            }

            /* Assign values. */
            for (size_t i = 0; i < ImgClassPreProcess::kNumChannels; ++i) {
                normMean[i] = img_class::g_normMean[i];
                normStddev[i] = img_class::g_normStddev[i];

                if (0.f == normStddev[i]) {
                    printf_err("Invalid std value: %f\n", normStddev[i]);
                    return false;
                }
            }
        }
        auto preProcess = ImgClassPreProcess(inputTensor, normMean, normStddev);
#else /* defined(MLEK_FWK_EXECUTORCH) */
        auto preProcess = ImgClassPreProcess(inputTensor);
#endif /* defined(MLEK_FWK_EXECUTORCH) */

        std::vector<ClassificationResult> results;
        ImgClassPostProcess postProcess = ImgClassPostProcess(outputTensor,
                ctx.Get<ImgClassClassifier&>("classifier"), ctx.Get<std::vector<std::string>&>("labels"),
                results);
#endif

        AnimalUiService();
        bool buttonTriggered = false;
        (void)run_requested_with_button(&buttonTriggered);
        if (buttonTriggered) { AnimalUiSingle(); }
        const uint32_t request = AnimalUiRequest();
        hal_camera_start();

        uint32_t capturedFrameSize = 0;
        const uint8_t* image_data = hal_camera_get_captured_frame(&capturedFrameSize);
        if (!image_data || !capturedFrameSize) {
            printf_err("hal_camera_get_captured_frame failed");
            return false;
        }

        AnimalUiService();
        uint32_t lv_lock_state = lv_port_lock();
        tprof5 = Get_SysTick_Cycle_Count32();
        /* Display this image on the LCD. */
#ifdef USE_LVGL_ZOOM
        write_to_lvgl_buf(
#else
        write_to_lvgl_buf_doubled(
#endif
                MIMAGE_X, MIMAGE_Y, image_data, &lvgl_image[0][0]);
        tprof5 = Get_SysTick_Cycle_Count32() - tprof5;

        lv_obj_invalidate(AnimalUiImageObject());

        if (SKIP_MODEL || !AnimalUiCurrent(request) || !AnimalUiClaim(request)) {
#if SHOW_PROFILING
            lv_label_set_text_fmt(AnimalUiLabelObject(0), "tprof1=%.3f ms", (double)tprof1 / SystemCoreClock * 1000);
            lv_label_set_text_fmt(AnimalUiLabelObject(1), "tprof2=%.3f ms", (double)tprof2 / SystemCoreClock * 1000);
            lv_label_set_text_fmt(AnimalUiLabelObject(2), "tprof3=%.3f ms", (double)tprof3 / SystemCoreClock * 1000);
            lv_label_set_text_fmt(AnimalUiLabelObject(3), "tprof4=%.3f ms", (double)tprof4 / SystemCoreClock * 1000);
            lv_label_set_text_fmt(AnimalUiLabelObject(4), "tprof5=%.3f ms", (double)tprof5 / SystemCoreClock * 1000);
#endif
#if SHOW_EXPOSURE
            lv_label_set_text_fmt(AnimalUiLabelObject(1), "low=%" PRIu32, exposure_low_count);
            lv_label_set_text_fmt(AnimalUiLabelObject(2), "high=%" PRIu32, exposure_high_count);
            lv_label_set_text_fmt(AnimalUiLabelObject(3), "gain=%.3f", get_image_gain());
#endif
            lv_led_off(AnimalUiLEDObject());
            lv_port_unlock(lv_lock_state);
            return true;
        }

        lv_led_on(AnimalUiLEDObject());
        lv_port_unlock(lv_lock_state);

#if !SKIP_MODEL
        const size_t imgSz = inputTensor->Bytes();

#if SHOW_INF_TIME
        uint32_t inf_prof = Get_SysTick_Cycle_Count32();
#endif

        /* Run the pre-processing, inference and post-processing. */
#if defined(GPIO_PROFILING)
        lv_lock_state = lv_port_lock();
        BOARD_LED1_BLUE_Control(BOARD_LED_STATE_TOGGLE);
#endif
        if (!preProcess.DoPreProcess(image_data, imgSz)) {
            printf_err("Pre-processing failed.");
            return false;
        }

        if (!RunInference(model, profiler)) {
            printf_err("Inference failed.");
            return false;
        }

        if (!postProcess.DoPostProcess()) {
            printf_err("Post-processing failed.");
            return false;
        }
#if defined(GPIO_PROFILING)
        BOARD_LED1_BLUE_Control(BOARD_LED_STATE_TOGGLE);
        lv_port_unlock(lv_lock_state);
#endif
#if SHOW_INF_TIME
        inf_prof = Get_SysTick_Cycle_Count32() - inf_prof;
#endif

        AnimalUiService();
        if (!AnimalUiCurrent(request)) { return true; }

        /* Add results to context for access outside handler. */
        ctx.Set<std::vector<ClassificationResult>>("results", results);

        bool animal = false;
        // Inspect every original top-three class ID, without filtering or rescaling scores.
        for (size_t i = 0; i < 3 && i < results.size(); ++i) {
            animal = IsAnimalClass(results[i].m_labelIdx) || animal;
#if defined(ANIMAL_DEBUG_TOP3)
            printf("Top %u: class=%u label=%s raw_score=%f\n",
                   static_cast<unsigned>(i + 1), results[i].m_labelIdx,
                   results[i].m_label.c_str(), static_cast<double>(results[i].m_normalisedVal));
#endif
        }
        lv_lock_state = lv_port_lock();
        if (AnimalUiCurrent(request)) { AnimalUiResult(animal); }
#if SHOW_INF_TIME
        lv_label_set_text_fmt(AnimalUiHeaderObject(), "%s - %.2f FPS", "Image Classifier", (double) SystemCoreClock / inf_prof);
        lv_label_set_text_fmt(AnimalUiTimeObject(), "%.3f ms", (double)inf_prof / SystemCoreClock * 1000);
#endif
        lv_port_unlock(lv_lock_state);



#endif


        return true;
    }

} /* namespace app */
} /* namespace alif */
