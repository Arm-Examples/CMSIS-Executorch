# This file was ported to work on Alif Semiconductor devices.

#  Copyright (C) 2023 Alif Semiconductor - All Rights Reserved.
#  Use, distribution and modification of this code is permitted under the
#  terms stated in the Alif Semiconductor Software License Agreement
#
#  You should have received a copy of the Alif Semiconductor Software
#  License Agreement with this file. If not, please write to:
#  contact@alifsemi.com, or visit: https://alifsemi.com/license

#----------------------------------------------------------------------------
#  Copyright (c) 2021 Arm Limited. All rights reserved.
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

# Specify the ML frameworks the use case supports
set(${use_case}_ML_FRAMEWORK "TensorFlowLiteMicro;ExecuTorch")
if (NOT ${ML_FRAMEWORK} IN_LIST ${use_case}_ML_FRAMEWORK)
    set(${use_case}_supports_${ML_FRAMEWORK} OFF)
    return()
endif ()

set(${use_case}_supports_${ML_FRAMEWORK} ON)

# Append the API to use for this use case
list(APPEND ${use_case}_API_LIST "kws")

USER_OPTION(${use_case}_LABELS_TXT_FILE "Labels' txt file for the chosen model."
    ${MLEK_ROOT}/resources/kws/labels/micronet_kws_labels.txt
    FILEPATH)

USER_OPTION(${use_case}_AUDIO_RATE "Specify the target sampling rate. Default is 16000."
    16000
    STRING)

USER_OPTION(${use_case}_MODEL_SCORE_THRESHOLD "Specify the score threshold [0.0, 1.0) that must be applied to the inference results for a label to be deemed valid."
    0.5
    STRING)

set(SE_SERVICES_SUPPORT ON CACHE BOOL "Enables SE Services initialization. Needed for power examples and KWS MHU communication.")

message(STATUS "alif_kws: SE_SERVICES_SUPPORT: ${SE_SERVICES_SUPPORT}")

set(${use_case}_COMPILE_DEFS
    $<$<BOOL:${SE_SERVICES_SUPPORT}>:SE_SERVICES_SUPPORT>
)

# Generate labels file
set(${use_case}_LABELS_CPP_FILE Labels)
generate_labels_code(
    INPUT           "${${use_case}_LABELS_TXT_FILE}"
    DESTINATION_SRC ${SRC_GEN_DIR}
    DESTINATION_HDR ${INC_GEN_DIR}
    OUTPUT_FILENAME "${${use_case}_LABELS_CPP_FILE}"
)

if (${ML_FRAMEWORK} STREQUAL "TensorFlowLiteMicro")
    USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "Activation buffer size for the chosen model"
        0x00020000
        STRING)

    if (ETHOS_U_NPU_ENABLED)
        set(DEFAULT_MODEL_PATH      ${RESOURCES_PATH}/kws/kws_micronet_m_vela_${ETHOS_U_NPU_CONFIG_ID}.tflite)
    else()
        set(DEFAULT_MODEL_PATH      ${RESOURCES_PATH}/kws/kws_micronet_m.tflite)
    endif()
elseif (${ML_FRAMEWORK} STREQUAL "ExecuTorch")
    # DS-CNN keyword spotter exported with dcops/export/export_kws.py (ExecuTorch 1.1, Ethos-U85-256).
    # Input float [1,1,49,10] MicroNetKwsMFCC features, output 12 float logits (labels order above).
    # The default is a RANDOM-WEIGHT placeholder so the firmware builds before the trained model
    # exists; pass -Dalif_kws_MODEL_PATH=<run>/kws_ethos-u85-256.pte for the real one.
    set(DEFAULT_MODEL_PATH ${MLEK_ROOT}/dcops/models/placeholder/kws_random/kws_ethos-u85-256.pte)

    # Method allocator: planned buffers (input/output/intermediates) + method metadata. DS-CNN(64)
    # needs only tens of KiB; the Ethos-U scratch comes from ML_FWK_TMP_MEM_SIZE.
    USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "Activation buffer size for the chosen model"
        0x00040000
        STRING)
endif()

set(EXTRA_MODEL_CODE
    "/* Model parameters for ${use_case} */"
    "extern const int   g_FrameLength    = 640"
    "extern const int   g_FrameStride    = 320"
    "extern const int   g_AudioRate      = ${${use_case}_AUDIO_RATE}"
    "extern const float g_ScoreThreshold = ${${use_case}_MODEL_SCORE_THRESHOLD}"
    )

USER_OPTION(${use_case}_MODEL_PATH "NN models file to be used in the evaluation application. Model files must be in tflite or pte format."
    ${DEFAULT_MODEL_PATH}
    FILEPATH)

if (COMMAND generate_pte_ops_lib)
    generate_pte_ops_lib(
        MODEL_PATH      "${${use_case}_MODEL_PATH}"     # Path to the model PTE
        LIB_NAME        "${use_case}_portable_ops_lib"  # Library target name
        SELECT_OPS_LIST "")                             # Always included ops list

    # If the target is generated, request it to be linked for this use case.
    if (TARGET ${use_case}_portable_ops_lib)
        set(${use_case}_LINK_LIBS ${use_case}_portable_ops_lib)
    endif()
endif()

# Generate model file
generate_model_code(
    MODEL_PATH ${${use_case}_MODEL_PATH}
    DESTINATION ${SRC_GEN_DIR}
    EXPRESSIONS ${EXTRA_MODEL_CODE}
    NAMESPACE   "arm" "app" "kws"
)
