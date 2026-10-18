/*---------------------------------------------------------------------------
 * BMI323 IMU (accelerometer + gyroscope) of the DevKit-E8, on the I3C bus.
 *
 * Uses the pack's BMI323 driver. Alif's demo (demo_bmi323.c) first switches
 * the flexible I/O pins (port 7, where the I3C bus is) to 1.8 V. Port 7 also
 * carries I2C1 (camera + touch), so we try the IMU at the current voltage
 * first and only switch to 1.8 V if the IMU does not answer.
 *
 * imu_poll() reads the latest sample and returns a "motion" amount:
 * gyro speed (deg/s) plus the change in acceleration since the last read.
 *---------------------------------------------------------------------------*/

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#include "Driver_IMU.h"
#include "se_services_port.h"

#include "imu.h"

extern ARM_DRIVER_IMU BMI323;

volatile int32_t g_imu_dbg[4];  /* debugger-readable: init results */

static float last_ax, last_ay, last_az;
static float last_g[3];  /* latest gyro reading, deg/s */
static int have_last;

static float value_to_float(const ARM_IMU_SENSOR_VALUE *v)
{
    return (float)v->int_val + (float)v->fract_val / 1000000.0f;
}

static int32_t set_ioflex_1v8(void)
{
    run_profile_t runp;
    uint32_t err = 0;
    uint32_t rc = SERVICES_get_run_cfg(se_services_s_handle, &runp, &err);
    if (rc != 0U) return -(int32_t)rc;
    runp.vdd_ioflex_3V3 = IOFLEX_LEVEL_1V8;
    rc = SERVICES_set_run_cfg(se_services_s_handle, &runp, &err);
    return rc != 0U ? -(int32_t)rc : 0;
}

int imu_init(void)
{
    int32_t r = BMI323.Initialize();
    g_imu_dbg[0] = r;
    printf("imu: BMI323 Initialize -> %ld\n", (long)r);
    if (r != ARM_DRIVER_OK) return -1;

    r = BMI323.PowerControl(ARM_POWER_FULL);
    g_imu_dbg[1] = r;
    printf("imu: BMI323 power up (current I/O voltage) -> %ld\n", (long)r);
    if (r != ARM_DRIVER_OK) {
        int32_t v = set_ioflex_1v8();
        g_imu_dbg[2] = v;
        printf("imu: switch flex I/O to 1.8V -> %ld\n", (long)v);
        r = BMI323.PowerControl(ARM_POWER_FULL);
        g_imu_dbg[3] = r;
        printf("imu: BMI323 power up (1.8V) -> %ld\n", (long)r);
        if (r != ARM_DRIVER_OK) return -2;
    }
    return 0;
}

int imu_poll(float *motion, float *ax, float *ay, float *az)
{
    ARM_IMU_COORDINATES a, g;
    if (BMI323.Control(IMU_GET_ACCELEROMETER_DATA, (uint32_t)&a) != ARM_DRIVER_OK) return -1;
    if (BMI323.Control(IMU_GET_GYROSCOPE_DATA, (uint32_t)&g) != ARM_DRIVER_OK) return -1;

    float x = value_to_float(&a.x), y = value_to_float(&a.y), z = value_to_float(&a.z);
    float gx = value_to_float(&g.x), gy = value_to_float(&g.y), gz = value_to_float(&g.z);
    last_g[0] = gx;
    last_g[1] = gy;
    last_g[2] = gz;

    float jolt = 0.0f;  /* change in acceleration, in g */
    if (have_last) {
        float dx = x - last_ax, dy = y - last_ay, dz = z - last_az;
        jolt = sqrtf(dx * dx + dy * dy + dz * dz);
    }
    last_ax = x;
    last_ay = y;
    last_az = z;
    have_last = 1;

    /* 1 g of jolt counts like 100 deg/s of rotation */
    *motion = sqrtf(gx * gx + gy * gy + gz * gz) + 100.0f * jolt;
    if (ax) *ax = x;
    if (ay) *ay = y;
    if (az) *az = z;
    return 0;
}

void imu_last_gyro(float *g)
{
    g[0] = last_g[0];
    g[1] = last_g[1];
    g[2] = last_g[2];
}
