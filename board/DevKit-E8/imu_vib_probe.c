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
 * DevKit-E8 IMU vibration diagnostic (APP_IMU_VIBRATION_PROBE): measures the
 * BMI323 accelerometer's real sample rate and vibration features.
 *
 * The pack's BMI323 driver (components/Source/BMI323_IMU.c) writes a fixed
 * accelerometer configuration (BMI323_ACCEL_CONFIG_VAL 0x4037, "Enable Normal
 * mode") and its ARM_DRIVER_IMU interface (Driver_IMU.h) has no control code
 * for the output data rate, bandwidth or filter, so the rate is measured, not
 * set. The windows alternate between two ways of sampling:
 *   drdy  one read per data-ready interrupt (the driver's INT1 GPIO callback,
 *         GetStatus() data_rcvd; board_imu_data_ready())
 *   poll  timed reads every VIB_POLL_US, whether or not a new sample is there
 * A read that returns the same x, y, z as the previous one is counted as a
 * repeat; the new samples per second are the sensor's effective rate.
 *
 * Per VIB_WINDOW_MS window, over the new samples of the magnitude |a| (mg):
 * the variance less the mean, the first-difference energy (mean squared
 * change between consecutive new samples) and the largest change. Streaming
 * sums only, no sample buffer; everything runs in board_imu_vib_poll() from
 * the main loop.
 *---------------------------------------------------------------------------*/

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#ifdef APP_IMU_VIBRATION_PROBE

#if defined(APP_SENTINEL) || defined(APP_IMU_PROBE)
#error "APP_IMU_VIBRATION_PROBE reads the IMU itself: do not define APP_SENTINEL or APP_IMU_PROBE with it"
#endif

#define VIB_WINDOW_MS 500u
#define VIB_POLL_US   1000u   /* 1 kHz timed polling in poll windows */

extern int board_imu_init(void);
extern int board_imu_data_ready(unsigned *drdy);
extern int board_imu_accel_ug(int32_t ug[3]);

enum { VIB_DRDY, VIB_POLL };

static int      vib_ready;
static int      vib_mode;
static uint32_t win_start_cyc, next_poll_cyc, cyc_per_us;

static uint32_t w_reads, w_new, w_err, w_irq, w_ndiff;
static double   w_sum, w_sq, w_dsq, w_dmax, w_prev;
static uint32_t w_read_cyc_sum, w_read_cyc_max;
static int32_t  last_ug[3];
static int      have_last, have_prev;

static void vib_reset(void)
{
    w_reads = w_new = w_err = w_irq = w_ndiff = 0;
    w_sum = w_sq = w_dsq = w_dmax = 0.0;
    w_read_cyc_sum = w_read_cyc_max = 0;
    have_prev = 0;   /* differences only within a window */
}

static void vib_sample(void)
{
    int32_t        a[3];
    const uint32_t c0  = DWT->CYCCNT;
    const int      ret = board_imu_accel_ug(a);
    const uint32_t c   = DWT->CYCCNT - c0;

    w_reads++;
    w_read_cyc_sum += c;
    if (c > w_read_cyc_max) {
        w_read_cyc_max = c;
    }
    if (ret != 0) {
        w_err++;
        return;
    }
    if (have_last && memcmp(a, last_ug, sizeof last_ug) == 0) {
        return;   /* repeat: no new sample since the last read */
    }
    memcpy(last_ug, a, sizeof last_ug);
    have_last = 1;
    w_new++;

    const double mag = sqrt((double) a[0] * a[0] + (double) a[1] * a[1] + (double) a[2] * a[2]) /
                       1000.0;   /* mg */
    w_sum += mag;
    w_sq += mag * mag;
    if (have_prev) {
        const double d = mag - w_prev;
        w_dsq += d * d;
        w_ndiff++;
        if (fabs(d) > w_dmax) {
            w_dmax = fabs(d);
        }
    }
    w_prev    = mag;
    have_prev = 1;
}

/*
  Bring up the IMU (imu_probe.c) and the DWT cycle counter. Returns 0 when
  ready.
*/
int board_imu_vib_init(void)
{
    if (board_imu_init() != 0) {
        printf("VIB: IMU not available\n");
        return -1;
    }
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    cyc_per_us    = SystemCoreClock / 1000000u;
    win_start_cyc = next_poll_cyc = DWT->CYCCNT;
    vib_mode      = VIB_DRDY;
    vib_reset();
    vib_ready = 1;
    printf("VIB: accelerometer diagnostic, %u ms windows alternating data-ready and %u us polling\n",
           VIB_WINDOW_MS, VIB_POLL_US);
    return 0;
}

/* Sample, and print and switch mode at the end of each window; main loop */
void board_imu_vib_poll(uint32_t now_ms)
{
    if (!vib_ready) {
        return;
    }
    const uint32_t now = DWT->CYCCNT;

    if (vib_mode == VIB_DRDY) {
        unsigned drdy;
        if (board_imu_data_ready(&drdy) == 1) {
            w_irq++;
            vib_sample();
        }
    } else if ((int32_t) (now - next_poll_cyc) >= 0) {
        next_poll_cyc += VIB_POLL_US * cyc_per_us;
        if ((int32_t) (now - next_poll_cyc) >= 0) {
            next_poll_cyc = now + VIB_POLL_US * cyc_per_us;   /* fell behind */
        }
        vib_sample();
    }

    const uint32_t win_cyc = DWT->CYCCNT - win_start_cyc;
    if (win_cyc < VIB_WINDOW_MS * 1000u * cyc_per_us) {
        return;
    }

    const double secs = (double) win_cyc / (cyc_per_us * 1e6);
    const double mean = w_new ? w_sum / w_new : 0.0;
    double       var  = w_new ? w_sq / w_new - mean * mean : 0.0;
    if (var < 0.0) {
        var = 0.0;
    }
    const double de = w_ndiff ? w_dsq / w_ndiff : 0.0;
    printf("VIB t=%" PRIu32 " %s reads=%" PRIu32 " irq=%" PRIu32 " new=%" PRIu32 " (%u%%) rate=%.1fHz"
           " mean=%.1fmg var=%.3fmg2 dE=%.3fmg2 dmax=%.2fmg read_us=%" PRIu32 "/%" PRIu32
           " err=%" PRIu32 "\n",
           now_ms, vib_mode == VIB_DRDY ? "drdy" : "poll", w_reads, w_irq, w_new,
           w_reads ? (unsigned) (100u * w_new / w_reads) : 0u, w_new / secs, mean, var, de, w_dmax,
           w_reads ? w_read_cyc_sum / w_reads / cyc_per_us : 0u, w_read_cyc_max / cyc_per_us, w_err);

    vib_mode      = vib_mode == VIB_DRDY ? VIB_POLL : VIB_DRDY;
    win_start_cyc = next_poll_cyc = DWT->CYCCNT;
    vib_reset();
}

#endif /* APP_IMU_VIBRATION_PROBE */
