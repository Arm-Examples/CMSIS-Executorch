/*---------------------------------------------------------------------------
 * Copyright (c) 2026 Sentinel contributors.
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
 *
 * DevKit-E8 IMU probe: brings up the BMI323 through the Ensemble pack's
 * ARM_DRIVER_IMU driver (components/Source/BMI323_IMU.c) and prints the
 * accelerometer on the console at about 10 Hz.
 *
 * The driver does not return the chip ID. Its PowerControl(ARM_POWER_FULL)
 * initialises the I3C master, assigns the dynamic address to static address
 * 0x69 (SETDASA), reads CHIP_ID (register 0x00) and fails unless it is 0x43,
 * then configures the accelerometer, gyroscope and INT1 and reads them back.
 * A successful power-up therefore identifies a BMI323; a failure is one of
 * those steps (the driver returns ARM_DRIVER_ERROR for all of them).
 *
 * Pins: board_pins_config() (main.c) already muxes I3C_SDA_D/I3C_SCL_D on
 * P7_6/P7_7 and the IMU interrupt pins P8_4/P8_5. The FLEX-IO voltage is left
 * as it is: the pack's demo_bmi323.c switches it to 1.8 V with
 * SERVICES_set_run_cfg(), which this layer does not do (see README.md).
 *
 * Each poll prints the driver's data-ready state before the read: irq is the
 * GetStatus() data_rcvd flag, which the driver's GPIO interrupt callback sets
 * (BOARD_BMI323_IMU_IRQ_GPIO_PORT/PIN, board_defs.h), and drdy the INT1
 * status bits it then reads. The accelerometer is read on every poll whether
 * or not the interrupt fired, so a wrong interrupt pin shows as irq=0 with
 * live values. The driver ignores I3C errors in its data reads, so a bus
 * failure after power-up shows as frozen values, not as an error code.
 *---------------------------------------------------------------------------*/

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#include "Driver_IMU.h"

#define IMU_POLL_MS 100u

extern ARM_DRIVER_IMU BMI323;
static ARM_DRIVER_IMU *imu = &BMI323;

static int      imu_ready;
static uint32_t imu_next_ms;

/* Sensor value (integer part + millionths, same sign) in thousandths */
static int32_t milli(ARM_IMU_SENSOR_VALUE v)
{
    return v.int_val * 1000 + v.fract_val / 1000;
}

/*
  Bring up the BMI323. Prints the driver version, the result of each step and
  the identity. Returns 0 when the IMU is ready.
*/
int board_imu_init(void)
{
    ARM_DRIVER_VERSION   ver = imu->GetVersion();
    ARM_IMU_CAPABILITIES cap = imu->GetCapabilities();
    int32_t              ret;

    printf("IMU: BMI323 driver api 0x%04x drv 0x%04x, accel %u gyro %u temp %u\n",
           ver.api, ver.drv, cap.accel, cap.gyro, cap.temp_sens);

    /* I3C driver (needs I3C driver version 7.0+), I3C power, IRQ GPIO */
    ret = imu->Initialize();
    if (ret != ARM_DRIVER_OK) {
        printf("IMU: Initialize failed (%" PRId32 "): I3C or IRQ GPIO driver init\n", ret);
        return -1;
    }

    /* The driver's dynamic-address waits have no timeout: if the bus is
       stuck, the board stops here, after this line. */
    printf("IMU: powering up (I3C SETDASA 0x69, CHIP_ID check 0x43)...\n");
    fflush(stdout);
    ret = imu->PowerControl(ARM_POWER_FULL);
    if (ret != ARM_DRIVER_OK) {
        printf("IMU: PowerControl failed (%" PRId32 "): no BMI323 at 0x69, CHIP_ID != 0x43, "
               "or config readback mismatch. FLEX-IO voltage not changed.\n", ret);
        return -1;
    }

    printf("IMU: BMI323 detected (driver CHIP_ID check 0x43 passed), polling every %u ms\n",
           IMU_POLL_MS);
    imu_ready = 1;
    return 0;
}

/*
  Read and print the accelerometer every IMU_POLL_MS; call from the main loop.
*/
void board_imu_poll(uint32_t now_ms)
{
    ARM_IMU_STATUS      st;
    ARM_IMU_COORDINATES acc = {0};
    int32_t             ret;

    if (!imu_ready || (int32_t)(now_ms - imu_next_ms) < 0) {
        return;
    }
    imu_next_ms = now_ms + IMU_POLL_MS;

    st  = imu->GetStatus();
    ret = imu->Control(IMU_GET_ACCELEROMETER_DATA, (uint32_t) &acc);
    /* The interrupt callback disables the interrupt; re-arm it */
    (void) imu->Control(IMU_SET_INTERRUPT, 1u);

    printf("IMU t=%" PRIu32 " ax=%+" PRId32 " ay=%+" PRId32 " az=%+" PRId32 " mg irq=%u drdy=0x%02x rc=%" PRId32 "\n",
           now_ms, milli(acc.x), milli(acc.y), milli(acc.z),
           (unsigned) st.data_rcvd, (unsigned) st.drdy_status, ret);
}

/*
  Read the accelerometer in mg (x, y, z) for callers that sample it
  themselves. Returns the driver's result; -1 when the IMU is not ready.
*/
int board_imu_accel_mg(int32_t mg[3])
{
    ARM_IMU_COORDINATES acc = {0};
    int32_t             ret;

    if (!imu_ready) {
        return -1;
    }
    ret = imu->Control(IMU_GET_ACCELEROMETER_DATA, (uint32_t) &acc);
    /* The interrupt callback disables the interrupt; re-arm it */
    (void) imu->Control(IMU_SET_INTERRUPT, 1u);
    mg[0] = milli(acc.x);
    mg[1] = milli(acc.y);
    mg[2] = milli(acc.z);
    return (int) ret;
}

/*
  Data-ready state of the driver: 1 when its INT1 GPIO interrupt has fired
  since the last read (GetStatus() data_rcvd), with the BMI323 data-ready
  bits (IMU_*_DATA_READY) in *drdy. Returns -1 when the IMU is not ready.
*/
int board_imu_data_ready(unsigned *drdy)
{
    if (!imu_ready) {
        return -1;
    }
    ARM_IMU_STATUS st = imu->GetStatus();
    *drdy             = st.drdy_status;
    return st.data_rcvd ? 1 : 0;
}

/*
  Read the accelerometer in micro-g (x, y, z), the driver's full resolution,
  and re-arm the data-ready interrupt. Returns the driver's result; -1 when
  the IMU is not ready.
*/
int board_imu_accel_ug(int32_t ug[3])
{
    ARM_IMU_COORDINATES acc = {0};
    int32_t             ret;

    if (!imu_ready) {
        return -1;
    }
    ret = imu->Control(IMU_GET_ACCELEROMETER_DATA, (uint32_t) &acc);
    (void) imu->Control(IMU_SET_INTERRUPT, 1u);
    ug[0] = acc.x.int_val * 1000000 + acc.x.fract_val;
    ug[1] = acc.y.int_val * 1000000 + acc.y.fract_val;
    ug[2] = acc.z.int_val * 1000000 + acc.z.fract_val;
    return (int) ret;
}
