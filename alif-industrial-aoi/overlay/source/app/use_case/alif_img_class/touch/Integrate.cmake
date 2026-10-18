# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
# Generate a marked adaptation, leaving the dependency source unchanged. The stock
# driver would lose I2C callbacks when sharing I2C1 with the camera.
file(READ "${ALIF_CMSIS_PATH}/components/Source/GT911_touch_driver.c" _touch)
foreach(_function Write Read)
    string(FIND "${_touch}" "static int32_t TOUCH_${_function}(" _start)
    string(SUBSTRING "${_touch}" ${_start} -1 _tail)
    string(FIND "${_tail}" "\n/**" _end)
    if (_start LESS 0 OR _end LESS 0)
        message(FATAL_ERROR "GT911 source layout changed; review shared-I2C adaptation")
    endif()
    string(SUBSTRING "${_tail}" 0 ${_end} _old)
    string(TOLOWER "${_function}" _lower)
    set(_new "static int32_t TOUCH_${_function}(uint8_t a, uint16_t r, uint8_t* d, uint8_t n)\n{ return animal_touch_${_lower}(a, r, d, n); }\n")
    string(REPLACE "${_old}" "${_new}" _touch "${_touch}")
endforeach()
string(FIND "${_touch}" "    ret = I2C_Driver->Initialize(TOUCH_I2C_CB);" _start)
string(FIND "${_touch}" "    ret = GPIO_Driver_INT->Initialize" _end)
if (_start LESS 0 OR _end LESS _start)
    message(FATAL_ERROR "GT911 initialization changed; review shared-I2C adaptation")
endif()
math(EXPR _length "${_end}-${_start}")
string(SUBSTRING "${_touch}" ${_start} ${_length} _old)
string(REPLACE "${_old}" "" _touch "${_touch}")
string(REPLACE "state->numtouches = 0;" "state->numtouches = -1;" _touch "${_touch}")
file(WRITE "${CMAKE_BINARY_DIR}/generated/animal_gt911.c"
    "/* Modified vendor GT911: serialized camera I2C transport; -1 means no new sample. */\n#include <stdint.h>\nextern int32_t animal_touch_read(uint8_t, uint16_t, uint8_t*, uint8_t);\nextern int32_t animal_touch_write(uint8_t, uint16_t, uint8_t*, uint8_t);\n${_touch}")
target_sources(cmsis_device PRIVATE "${CMAKE_BINARY_DIR}/generated/animal_gt911.c")
target_compile_definitions(cmsis_device PRIVATE RTE_Drivers_GT911)
set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/../src/AnimalTouch.cc" APPEND PROPERTY
    INCLUDE_DIRECTORIES "${ALIF_CMSIS_PATH}/Alif_CMSIS/Source")
