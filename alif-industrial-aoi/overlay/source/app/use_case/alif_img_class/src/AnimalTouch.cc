// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
#include "AnimalTouch.hpp"
#include "Driver_Touch_Screen.h"
#include "Camera_Sensor_i2c.h"
#include "board_defs.h"
#include "sys_utils.h"
#include <cstdio>
extern "C" ARM_DRIVER_TOUCH_SCREEN GT911;
extern "C" ARM_DRIVER_I2C ARM_Driver_I2C_(BOARD_GT911_TOUCHSCREEN_I2C_INSTANCE);
namespace {
CAMERA_SENSOR_SLAVE_I2C_CONFIG s_bus = {
    &ARM_Driver_I2C_(BOARD_GT911_TOUCHSCREEN_I2C_INSTANCE), ARM_I2C_BUS_SPEED_STANDARD,
    RTE_GT911_TOUCH_I2C_SLAVE_ADDRESS_SEL, CAMERA_SENSOR_I2C_REG_ADDR_TYPE_16BIT};
bool s_ready = false;
uint16_t s_width, s_height;
char s_status[128] = "Touch not initialized";
uint16_t s_failedReg = 0;
}
// Used by the generated, minimally adapted vendor driver. Do not initialize/power-cycle
// I2C1 here: the OV5675 owns its callback. All operations are serialized in the main loop.
extern "C" int32_t animal_touch_read(uint8_t address, uint16_t reg, uint8_t* data, uint8_t size)
{
    if (address != s_bus.cam_sensor_slave_addr || !data || !size || size > 4) {
        return ARM_DRIVER_ERROR_PARAMETER;
    }
    // GT911 coordinates and dimensions are contiguous little-endian byte blocks.
    // The camera helper returns a big-endian integer; unpack it in wire order.
    // A three-byte request reads four bytes and discards the trailing byte.
    const unsigned count = size > 2 ? 4 : size;
    for (unsigned attempt = 0; attempt < 3; ++attempt) {
        uint32_t value = 0;
        int32_t status = camera_sensor_i2c_read(
            &s_bus, reg, &value, static_cast<CAMERA_SENSOR_I2C_REG_SIZE>(count));
        if (status == ARM_DRIVER_OK) {
            for (unsigned i = 0; i < size; ++i) {
                data[i] = static_cast<uint8_t>(value >> (8 * (count - 1 - i)));
            }
            return ARM_DRIVER_OK;
        }
        // Retry completed failed transactions only; do not reset the camera's bus.
        if (s_bus.drv_i2c->GetStatus().busy || attempt == 2) {
            s_failedReg = reg;
            return status;
        }
        sys_busy_loop_us(1000);
    }
    return ARM_DRIVER_ERROR;
}
extern "C" int32_t animal_touch_write(uint8_t address, uint16_t reg, uint8_t* data, uint8_t size)
{
    if (address != s_bus.cam_sensor_slave_addr) { return ARM_DRIVER_ERROR; }
    for (unsigned i = 0; i < size; ++i) {
        int32_t status = camera_sensor_i2c_write(&s_bus, reg + i, data[i],
                                                CAMERA_SENSOR_I2C_REG_SIZE_8BIT);
        if (status != ARM_DRIVER_OK) { return status; }
    }
    return ARM_DRIVER_OK;
}
namespace alif::app {
bool AnimalTouchInit()
{
    const char* stage = "GPIO init";
    int32_t status = GT911.Initialize();
    if (status == ARM_DRIVER_OK) {
        stage = "reset/setup";
        status = GT911.PowerControl(ARM_POWER_FULL);
    }
    uint8_t identity[3] = {}, dimensions[4] = {};
    if (status == ARM_DRIVER_OK) {
        stage = "product read";
        status = animal_touch_read(s_bus.cam_sensor_slave_addr, 0x8140, identity, 3);
    }
    if (status == ARM_DRIVER_OK) {
        stage = "dimensions read";
        status = animal_touch_read(s_bus.cam_sensor_slave_addr, 0x8048, dimensions, 4);
    }
    s_width = dimensions[0] | (dimensions[1] << 8);
    s_height = dimensions[2] | (dimensions[3] << 8);
    s_ready = status == ARM_DRIVER_OK && identity[0] == '9' && identity[1] == '1' &&
              identity[2] == '1' && s_width > 1 && s_height > 1;
    if (status != ARM_DRIVER_OK) {
        snprintf(s_status, sizeof(s_status),
                 "Touch: %s failed (%ld)\nI2C 0x%02x register 0x%04x",
                 stage, static_cast<long>(status), s_bus.cam_sensor_slave_addr, s_failedReg);
    } else if (!s_ready) {
        snprintf(s_status, sizeof(s_status), "Touch: ID %02x %02x %02x\nDimensions %ux%u",
                 identity[0], identity[1], identity[2], s_width, s_height);
    } else {
        snprintf(s_status, sizeof(s_status), "Touch ready: GT911 %ux%u", s_width, s_height);
    }
    printf("%s\n", s_status);
    printf("GT911 status=%ld id=%c%c%c coordinates=%ux%u panel=%ux%u swap=%d\n",
           static_cast<long>(status), identity[0], identity[1], identity[2], s_width, s_height,
           RTE_PANEL_HACTIVE_TIME, RTE_PANEL_VACTIVE_LINE,
           (s_width > s_height) != (RTE_PANEL_HACTIVE_TIME > RTE_PANEL_VACTIVE_LINE));
    return s_ready;
}
const char* AnimalTouchStatus() { return s_status; }
int AnimalTouchPoll(int32_t& x, int32_t& y)
{
    if (!s_ready) { return -1; }
    ARM_TOUCH_STATE state = {};
    int32_t status = GT911.GetState(&state);
    if (status != ARM_DRIVER_OK) {
        s_ready = false;
        printf("GT911 read failed (%ld); touch disabled, joystick remains available\n",
               static_cast<long>(status));
        return 0;
    }
    if (state.numtouches < 0) { return -1; }
    if (!state.numtouches) { return 0; }
    x = state.coordinates[0].x;
    y = state.coordinates[0].y;
    uint32_t width = s_width, height = s_height;
    if ((width > height) != (RTE_PANEL_HACTIVE_TIME > RTE_PANEL_VACTIVE_LINE)) {
        int32_t temp = x; x = y; y = temp;
        uint32_t dimension = width; width = height; height = dimension;
    }
    if (x < 0 || y < 0 || static_cast<uint32_t>(x) >= width ||
        static_cast<uint32_t>(y) >= height) { return 0; }
    x = x * (RTE_PANEL_HACTIVE_TIME - 1) / (width - 1);
    y = y * (RTE_PANEL_VACTIVE_LINE - 1) / (height - 1);
    return 1;
}
}
