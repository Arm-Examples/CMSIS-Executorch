# This file was ported to work on Alif Semiconductor devices.

#  Copyright (C) 2023, 2026 Alif Semiconductor - All Rights Reserved.
#  Use, distribution and modification of this code is permitted under the
#  terms stated in the Alif Semiconductor Software License Agreement
#
#  You should have received a copy of the Alif Semiconductor Software
#  License Agreement with this file. If not, please write to:
#  contact@alifsemi.com, or visit: https://alifsemi.com/license

#----------------------------------------------------------------------------
#  Copyright (c) 2022, 2026 Arm Limited. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#----------------------------------------------------------------------------

# Specify the ML frameworks the use case supports:
#   TensorFlowLiteMicro - YOLO Fastest face detector (192x192 grayscale), the original demo
#   ExecuTorch          - MediaPipe BlazeFace (128x128 RGB), see resources/object_detection/blazeface
set(${use_case}_ML_FRAMEWORK "TensorFlowLiteMicro;ExecuTorch")
if (NOT ${ML_FRAMEWORK} IN_LIST ${use_case}_ML_FRAMEWORK)
    message(STATUS "${use_case} does not support ${ML_FRAMEWORK} framework")
    set(${use_case}_supports_${ML_FRAMEWORK} OFF)
    return()
endif ()

set(${use_case}_supports_${ML_FRAMEWORK} ON)

# Append the API to use for this use case
list(APPEND ${use_case}_API_LIST "object_detection" "alif_ui")

if (${ML_FRAMEWORK} STREQUAL "ExecuTorch")
    if (NOT ETHOS_U_NPU_ENABLED)
        message(FATAL_ERROR "${use_case}: the ExecuTorch BlazeFace program is compiled for Ethos-U85")
    endif()

    # FACE: BlazeFace detector. CLASSIFY: an image classifier (one logits vector out) on the
    # same camera pipeline, with the top label drawn over the preview.
    USER_OPTION(${use_case}_ET_TASK "ExecuTorch task: FACE or CLASSIFY"
        FACE
        STRING)

    if (${use_case}_ET_TASK STREQUAL "CLASSIFY")
        set(_DEFAULT_ET_MODEL ${MLEK_ROOT}/resources/object_detection/classifier/mv2nhwc_160_desk67_ethos-u85-256.pte)
    else()
        set(_DEFAULT_ET_MODEL ${MLEK_ROOT}/resources/object_detection/blazeface/blazeface_front_128_ethos-u85-256.pte)
    endif()

    USER_OPTION(${use_case}_MODEL_PATH "ExecuTorch program (.pte) for the Ethos-U85."
        ${_DEFAULT_ET_MODEL}
        FILEPATH)

    USER_OPTION(${use_case}_CLASS_LABELS "Labels file for the CLASSIFY task, one per line"
        ${MLEK_ROOT}/resources/object_detection/classifier/labels_desk67.txt
        FILEPATH)

    # Vela reports 387 KB of SRAM for the delegate; the rest is ExecuTorch's method memory.
    USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "Activation buffer size for the chosen model"
        0x00100000
        STRING)

    USER_OPTION(${use_case}_SCORE_THRESHOLD "BlazeFace minimum face score (0-1)"
        0.75
        STRING)

    USER_OPTION(${use_case}_SHOW_INF_TIME "Show inference time"
        OFF
        BOOL)

    USER_OPTION(${use_case}_FACE_ROTATIONS "Run BlazeFace on 1, 2 (0/180) or 4 (0/90/180/270) rotations of each frame"
        4
        STRING)

    # CLASSIFY UX: decisions are made on an exponential moving average of the softmax
    # (smoothed = alpha * current + (1 - alpha) * smoothed), and a label is only shown when
    # the smoothed top-1 reaches the threshold and is not the background class.
    USER_OPTION(${use_case}_CLS_EMA_ALPHA "CLASSIFY: EMA weight of the newest frame (0-1, 1 = no smoothing)"
        0.3
        STRING)

    USER_OPTION(${use_case}_CLS_CONF_THRESHOLD "CLASSIFY: smoothed top-1 probability needed to show a label (0-1)"
        0.5
        STRING)

    USER_OPTION(${use_case}_CLS_ALLOWED_MASK "CLASSIFY: bitmask of classes that may be shown (bit i = class i)"
        0xFFFFFFFF
        STRING)

    USER_OPTION(${use_case}_CLS_BACKGROUND_LABEL "CLASSIFY: label treated as 'nothing to identify'"
        "background"
        STRING)

    # Optional: info cards are matched to labels by name; a missing file just shows names.
    USER_OPTION(${use_case}_CLS_KNOWLEDGE_BASE "CLASSIFY: component knowledge base JSON for info cards"
        ${MLEK_ROOT}/dcops/dc_components.json
        STRING)

    if (${use_case}_ET_TASK STREQUAL "CLASSIFY")
        set(_ET_TASK_CLASSIFY 1)
    else()
        set(_ET_TASK_CLASSIFY 0)
    endif()

    set(${use_case}_COMPILE_DEFS
        SHOW_INF_TIME=$<BOOL:${${use_case}_SHOW_INF_TIME}>
        FACE_ROTATIONS=${${use_case}_FACE_ROTATIONS}
        ET_TASK_CLASSIFY=${_ET_TASK_CLASSIFY}
        CLS_EMA_ALPHA=${${use_case}_CLS_EMA_ALPHA}f
        CLS_CONF_THRESHOLD=${${use_case}_CLS_CONF_THRESHOLD}f
        CLS_ALLOWED_MASK=${${use_case}_CLS_ALLOWED_MASK}u
    )

    set(EXTRA_MODEL_CODE
        "extern const float blazeFaceScoreThreshold = ${${use_case}_SCORE_THRESHOLD}f;"
    )

    if (_ET_TASK_CLASSIFY)
        # Labels plus info-card titles/summaries (from the knowledge base, keyed by label
        # name) generated into a C++ source; the summary lines are fitted to the card width.
        set(_CARDS_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/gen_class_cards.py)
        set(_CARDS_ARGS
            --labels ${${use_case}_CLASS_LABELS}
            --background-label ${${use_case}_CLS_BACKGROUND_LABEL}
            --width 440
            --output ${SRC_GEN_DIR}/ClassCards.cc)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${_CARDS_SCRIPT} ${${use_case}_CLASS_LABELS})
        if (EXISTS "${${use_case}_CLS_KNOWLEDGE_BASE}")
            list(APPEND _CARDS_ARGS --kb ${${use_case}_CLS_KNOWLEDGE_BASE})
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                ${${use_case}_CLS_KNOWLEDGE_BASE})
        else()
            message(WARNING "${use_case}: no knowledge base at "
                "'${${use_case}_CLS_KNOWLEDGE_BASE}', info cards will show names only")
        endif()
        if (DEFINED LVGL_SRC_PATH AND EXISTS ${LVGL_SRC_PATH}/src/font/lv_font_montserrat_16.c)
            list(APPEND _CARDS_ARGS --font ${LVGL_SRC_PATH}/src/font/lv_font_montserrat_16.c)
        endif()
        execute_process(
            COMMAND ${PYTHON} ${_CARDS_SCRIPT} ${_CARDS_ARGS}
            RESULT_VARIABLE _CARDS_RESULT)
        if (NOT _CARDS_RESULT EQUAL "0")
            message(FATAL_ERROR "${use_case}: failed to generate class labels and cards")
        endif()
    else()
        # Class labels as a C array (unused by FACE, kept so the sources link unchanged).
        file(STRINGS "${${use_case}_CLASS_LABELS}" _CLASS_LABELS)
        list(LENGTH _CLASS_LABELS _NUM_CLASS_LABELS)
        set(_LABELS_C "")
        foreach(_L IN LISTS _CLASS_LABELS)
            string(REPLACE "\\" "\\\\" _L "${_L}")
            string(REPLACE "\"" "\\\"" _L "${_L}")
            string(APPEND _LABELS_C "\"${_L}\", ")
        endforeach()
        list(APPEND EXTRA_MODEL_CODE
            "extern const char* const g_classLabels[] = { ${_LABELS_C} };"
            "extern const int g_numClassLabels = ${_NUM_CLASS_LABELS};"
        )
    endif()

    # Voice (CLASSIFY only): a DS-CNN keyword spotter as a second ExecuTorch program in the same
    # HP image. Audio comes from the audio_alif HAL (TARGET_MICS), the MFCC from the kws API, and
    # the keywords drive the scanner card (go = latch, stop = resume, up/down = checklist,
    # yes/no = pass/fail). See dcops/docs/fusion-firmware.md. OFF leaves the build unchanged.
    USER_OPTION(${use_case}_ENABLE_VOICE "CLASSIFY: keyword spotting (second .pte) drives the scanner UI"
        OFF
        BOOL)

    if (${use_case}_ENABLE_VOICE)
        if (NOT _ET_TASK_CLASSIFY)
            message(FATAL_ERROR "${use_case}: ENABLE_VOICE needs ${use_case}_ET_TASK=CLASSIFY")
        endif()

        USER_OPTION(${use_case}_KWS_MODEL_PATH "Voice: DS-CNN keyword spotter (.pte) for the Ethos-U85"
            ${MLEK_ROOT}/dcops/models/placeholder/kws_random/kws_ethos-u85-256.pte
            FILEPATH)

        USER_OPTION(${use_case}_KWS_LABELS "Voice: keyword labels, one per line, in model output order"
            ${MLEK_ROOT}/resources/kws/labels/micronet_kws_labels.txt
            FILEPATH)

        # Method allocator of the KWS program (planned buffers + method metadata), in DTCM.
        # The Ethos-U scratch comes from the shared ML_FWK_TMP_MEM_SIZE pool (used in turn).
        USER_OPTION(${use_case}_KWS_ACTIVATION_BUF_SZ "Voice: KWS method arena size (DTCM)"
            0x00020000
            STRING)

        USER_OPTION(${use_case}_KWS_THRESHOLD "Voice: averaged softmax needed to fire a keyword (0-1)"
            0.7
            STRING)

        list(APPEND ${use_case}_API_LIST "kws")
        list(APPEND ${use_case}_COMPILE_DEFS
            ENABLE_VOICE=1
            KWS_ACTIVATION_BUF_SZ=${${use_case}_KWS_ACTIVATION_BUF_SZ}
            KWS_THRESHOLD=${${use_case}_KWS_THRESHOLD}f)

        file(STRINGS "${${use_case}_KWS_LABELS}" _KWS_LABELS)
        list(LENGTH _KWS_LABELS _NUM_KWS_LABELS)
        set(_KWS_LABELS_C "")
        foreach(_L IN LISTS _KWS_LABELS)
            string(APPEND _KWS_LABELS_C "\"${_L}\", ")
        endforeach()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${${use_case}_KWS_LABELS})

        # Generated into a subdirectory so it can never clash with the vision model's .cc name.
        generate_model_code(
            MODEL_PATH  ${${use_case}_KWS_MODEL_PATH}
            DESTINATION ${SRC_GEN_DIR}/kws
            EXPRESSIONS
                "extern const int   g_FrameLength    = 640"
                "extern const int   g_FrameStride    = 320"
                "extern const float g_ScoreThreshold = ${${use_case}_KWS_THRESHOLD}f"
                "extern const char* const g_kwsLabels[] = { ${_KWS_LABELS_C} }"
                "extern const int g_numKwsLabels = ${_NUM_KWS_LABELS}"
            NAMESPACE   "arm" "app" "kws")

        # The audio HAL prints two lines per 0.5 s stride; at UART speed that is several ms of
        # blocking in the camera loop, so keep it quiet in this app.
        if (TARGET audio_alif)
            target_compile_definitions(audio_alif PRIVATE AUDIO_ALIF_QUIET)
        endif()

        # The MFCC is the only FFT user in this image and always uses 1024 points: init it with
        # the size-specific CMSIS-DSP call so the tables of the other sizes (~110 KiB, copied to
        # DTCM from MRAM) are not linked. MRAM headroom below 0x80578000 is tight.
        # CMSIS-DSP keeps all sizes in one object per function family, and the toolchain builds
        # with -fno-function-sections, so also give CMSIS-DSP per-function sections for the
        # linker's --gc-sections to drop the unused sizes (its code all lands in ITCM via .text*).
        if (TARGET arm_math)
            target_compile_definitions(arm_math PRIVATE MLEK_FFT_FIXED_LEN=1024)
        endif()
        if (TARGET CMSISDSP)
            target_compile_options(CMSISDSP PRIVATE -ffunction-sections)
        endif()
    endif()

    # Link only the CPU operators the program still calls (the int8 boundary).
    if (COMMAND generate_pte_ops_lib)
        generate_pte_ops_lib(
            MODEL_PATH      "${${use_case}_MODEL_PATH}"
            LIB_NAME        "${use_case}_portable_ops_lib"
            SELECT_OPS_LIST "")
        if (TARGET ${use_case}_portable_ops_lib)
            set(${use_case}_LINK_LIBS ${use_case}_portable_ops_lib)
        endif()
        if (${use_case}_ENABLE_VOICE)
            generate_pte_ops_lib(
                MODEL_PATH      "${${use_case}_KWS_MODEL_PATH}"
                LIB_NAME        "${use_case}_kws_portable_ops_lib"
                SELECT_OPS_LIST "")
            if (TARGET ${use_case}_kws_portable_ops_lib)
                list(APPEND ${use_case}_LINK_LIBS ${use_case}_kws_portable_ops_lib)
            endif()
        endif()
    endif()

    generate_model_code(
        MODEL_PATH ${${use_case}_MODEL_PATH}
        DESTINATION ${SRC_GEN_DIR}
        EXPRESSIONS ${EXTRA_MODEL_CODE}
        NAMESPACE   "arm" "app" "object_detection")

    return()
endif()

# ---- TensorFlowLiteMicro (YOLO Fastest) below ----

USER_OPTION(${use_case}_MODEL_IN_EXT_FLASH "Run model from external flash"
    OFF
    BOOL)

USER_OPTION(${use_case}_IMAGE_SIZE "Square image size in pixels. Images will be resized to this size."
    192
    STRING)

USER_OPTION(${use_case}_ANCHOR_1 "First anchor array estimated during training."
    "{38, 77, 47, 97, 61, 126}"
    STRING)

USER_OPTION(${use_case}_ANCHOR_2 "Second anchor array estimated during training."
    "{14, 26, 19, 37, 28, 55 }"
    STRING)

USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "Activation buffer size for the chosen model"
    0x00082000
    STRING)

USER_OPTION(${use_case}_SHOW_INF_TIME "Show inference time"
    OFF
    BOOL)

set(${use_case}_COMPILE_DEFS
    SHOW_INF_TIME=$<BOOL:${${use_case}_SHOW_INF_TIME}>
)

if (ETHOS_U_NPU_ENABLED)
    set(DEFAULT_MODEL_PATH      ${RESOURCES_PATH}/object_detection/yolo-fastest_192_face_v4_vela_${ETHOS_U_NPU_CONFIG_ID}.tflite)
else()
    set(DEFAULT_MODEL_PATH      ${RESOURCES_PATH}/object_detection/yolo-fastest_192_face_v4.tflite)
endif()

set(${use_case}_ORIGINAL_IMAGE_SIZE
    ${${use_case}_IMAGE_SIZE}
    CACHE STRING
    "Original image size - for the post processing step to upscale the box co-ordinates.")

set(EXTRA_MODEL_CODE
    "extern const int originalImageSize = ${${use_case}_ORIGINAL_IMAGE_SIZE};"
    "/* NOTE: anchors are different for any given input model size, estimated during training phase */"
    "extern const float anchor1[] = ${${use_case}_ANCHOR_1};"
    "extern const float anchor2[] = ${${use_case}_ANCHOR_2};"
    )

USER_OPTION(${use_case}_MODEL_PATH "NN models file to be used in the evaluation application. Model files must be in tflite format."
    ${DEFAULT_MODEL_PATH}
    FILEPATH
    )

# Generate model file
generate_model_code(
    MODEL_PATH ${${use_case}_MODEL_PATH}
    DESTINATION ${SRC_GEN_DIR}
    EXPRESSIONS ${EXTRA_MODEL_CODE}
    NAMESPACE   "arm" "app" "object_detection")

# Rest are for static images use case
set_input_file_path_user_option(".bmp" object_detection)

# Generate input files
generate_images_code("${object_detection_FILE_PATH}"
${SAMPLES_GEN_DIR}
"${${use_case}_IMAGE_SIZE}")
