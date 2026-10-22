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
 * DevKit-E8 microphone probe: captures the PDM microphones through the
 * Ensemble pack's Driver_PDM (Alif_CMSIS/Source/Driver_PDM.c) and prints
 * sample statistics on the console twice per second.
 *
 * Setup follows the pack's Boards/Templates/Baremetal/demo_pdm.c: the 76.8 MHz
 * HFOSCx2 clock from the Secure Enclave, channels 4 and 5 (clock PDM_C2_A on
 * P6_7, data PDM_D2_B on P5_4, both muxed by board_pins_config() in main.c),
 * the DC-blocking IIR control, and the demo's per-channel phase, gain, peak
 * detection and FIR/IIR coefficients. The mode is 16 kHz (decimation 64)
 * instead of the demo's 8 kHz.
 *
 * Receive() is one-shot: the driver's FIFO interrupt (RTE_PDM_DMA_ENABLE 0)
 * stores the frames as ch4, ch5, ch4, ch5, ... 16-bit samples
 * (drivers/source/pdm.c, pdm_warning_irq_handler()) and signals
 * ARM_PDM_EVENT_CAPTURE_COMPLETE when the buffer is full. Until the next
 * Receive() nobody reads the hardware FIFO, which then overflows (the driver
 * reports ARM_PDM_EVENT_ERROR right after the next Receive()). So the
 * callback re-arms the capture itself into the next block of a ring of
 * MIC_NBLK static blocks, as the pack's vStream audio driver does with SAI
 * (Boards/DevKit-e8/Drivers/vstream_audio_in.c), and board_mic_poll()
 * accumulates and releases the finished blocks. A main loop blocked for
 * longer than the ring (a 49 kB image frame on the console takes ~4.7 s)
 * fills it; the callback then stops, as the vStream driver does, the stop
 * is counted as full, and board_mic_poll() restarts the capture.
 *
 * Printed per channel over the last ~500 ms: the mean (DC level), the RMS and
 * the peak of the samples less their 100 ms block mean.
 *---------------------------------------------------------------------------*/

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#include "Driver_PDM.h"
#include "se_services_port.h"

#ifdef APP_SENTINEL
/* sentinel.c: one finished block (interleaved ch4, ch5), from board_mic_poll() */
extern void sentinel_mic_block(const uint16_t *b, unsigned n);
#endif

#ifndef SOC_FEAT_CLK76P8M_CLK_ENABLE
#error "SOC_FEAT_CLK76P8M_CLK_ENABLE not defined: cannot tell whether PDM needs HFOSCx2"
#endif

#define MIC_RATE_HZ      16000u
#define MIC_CHANNELS     2u                                   /* 4 and 5 */
#define MIC_BLOCK_MS     100u
#define MIC_BLOCK        (MIC_RATE_HZ / 1000u * MIC_BLOCK_MS * MIC_CHANNELS) /* even */
#define MIC_PRINT_MS     500u

extern ARM_DRIVER_PDM  Driver_PDM;
static ARM_DRIVER_PDM *pdm = &Driver_PDM;

/* demo_pdm.c's channel 4 and 5 settings */
static const uint32_t ch4_fir[PDM_MAX_FIR_COEFFICIENT] = {
    0x00000001, 0x00000003, 0x00000003, 0x000007F4, 0x00000004, 0x000007ED,
    0x000007F5, 0x000007F4, 0x000007D3, 0x000007FE, 0x000007BC, 0x000007E5,
    0x000007D9, 0x00000793, 0x00000029, 0x0000072C, 0x00000072, 0x000002FD};
static const uint32_t ch5_fir[PDM_MAX_FIR_COEFFICIENT] = {
    0x00000000, 0x000007FF, 0x00000000, 0x00000004, 0x00000004, 0x000007FC,
    0x00000000, 0x000007FB, 0x000007E4, 0x00000000, 0x0000002B, 0x00000009,
    0x00000016, 0x00000049, 0x00000793, 0x000006F8, 0x00000045, 0x00000178};

static const struct {
    uint8_t         ch;
    uint32_t        phase, gain, peak_th, peak_itv;
    const uint32_t *fir;
} mic_ch[MIC_CHANNELS] = {
    {4, 0x0000001F, 0x0000000D, 0x00060002, 0x0004002D, ch4_fir},
    {5, 0x00000003, 0x00000013, 0x00060002, 0x00020027, ch5_fir},
};

#define MIC_NBLK 4u     /* ring of 4 x 100 ms */

/* Block k of the capture goes to mic_buf[k % MIC_NBLK]. The callback advances
   mic_filled, board_mic_poll() mic_used; blocks mic_used..mic_filled-1 are
   ready, block mic_filled is being captured unless mic_stopped. */
static uint16_t          mic_buf[MIC_NBLK][MIC_BLOCK];
static volatile uint32_t mic_filled;
static volatile uint32_t mic_used;
static volatile int      mic_stopped;
static int               mic_running;

/* Callback state: event counts, and DWT cycle stamps of the last Receive(),
   error and ring-full stop */
static volatile uint32_t mic_done_events;
static volatile uint32_t mic_err_events;
static volatile uint32_t mic_detect_events;
static volatile uint32_t mic_full_events;
static volatile uint32_t mic_rearm_fail;
static volatile int      mic_capturing;    /* set after the first Receive() */
static volatile uint32_t mic_pre_events;   /* callbacks before that, ignored */
static volatile uint32_t mic_arm_cyc;
static volatile uint32_t mic_err_cyc;
static volatile uint32_t mic_stop_cyc;
static uint32_t          mic_err_seen;

/* Per print window: the longest ring-full stop until board_mic_poll()
   restarted the capture, and where in its block the last error came */
static uint32_t win_stop_max;
static int32_t  win_err_at = -1;

/* Start capturing block mic_filled; from the callback or board_mic_poll() */
static void mic_arm(void)
{
    mic_arm_cyc = DWT->CYCCNT;
    if (pdm->Receive(mic_buf[mic_filled % MIC_NBLK], MIC_BLOCK) != ARM_DRIVER_OK) {
        mic_rearm_fail++;
        mic_stopped = 1;
    }
}

/* Totals and the current print window */
static uint64_t mic_total;
static uint32_t mic_next_print;
static uint32_t win_blocks;
static int64_t  win_mean_sum[MIC_CHANNELS];
static uint64_t win_sq[MIC_CHANNELS];
static uint32_t win_n[MIC_CHANNELS];
static int32_t  win_peak[MIC_CHANNELS];

static void mic_callback(uint32_t event)
{
    /* Before the first Receive() the driver's transfer count is 0, and its
       FIFO interrupt reports ARM_PDM_EVENT_CAPTURE_COMPLETE for any full
       FIFO (pdm_warning_irq_handler(): curr_cnt >= total_cnt), even while
       PowerControl() is still running (it enables the NVIC interrupts
       before it marks the driver powered, so a Receive() then fails).
       Count and ignore everything until board_mic_init() has started the
       capture. */
    if (!mic_capturing) {
        mic_pre_events++;
        return;
    }
    if (event & ARM_PDM_EVENT_ERROR) {
        mic_err_cyc = DWT->CYCCNT;
        mic_err_events++;
    }
    if (event & ARM_PDM_EVENT_CAPTURE_COMPLETE) {
        mic_done_events++;
        mic_filled++;
        if (mic_filled - mic_used >= MIC_NBLK) {
            /* Ring full: the next block is not released yet */
            mic_stop_cyc = DWT->CYCCNT;
            mic_stopped  = 1;
            mic_full_events++;
        } else {
            mic_arm();
        }
    }
    if (event & ARM_PDM_EVENT_AUDIO_DETECTION) {
        mic_detect_events++;
    }
}

/* Add one finished block (interleaved ch4, ch5) to the window */
static void mic_accumulate(const uint16_t *b)
{
    for (unsigned c = 0; c < MIC_CHANNELS; c++) {
        int64_t sum = 0;
        for (unsigned i = c; i < MIC_BLOCK; i += MIC_CHANNELS) {
            sum += (int16_t) b[i];
        }
        const int32_t mean = (int32_t) (sum / (MIC_BLOCK / MIC_CHANNELS));
        for (unsigned i = c; i < MIC_BLOCK; i += MIC_CHANNELS) {
            const int32_t d = (int16_t) b[i] - mean;
            win_sq[c] += (uint64_t) ((int64_t) d * d);
            if (d > win_peak[c]) {
                win_peak[c] = d;
            } else if (-d > win_peak[c]) {
                win_peak[c] = -d;
            }
        }
        win_mean_sum[c] += mean;
        win_n[c] += MIC_BLOCK / MIC_CHANNELS;
    }
    win_blocks++;
}

static int32_t mic_setup(void)
{
    int32_t ret;

    if ((ret = pdm->Initialize(mic_callback)) != ARM_DRIVER_OK) {
        printf("MIC: Initialize failed (%" PRId32 ")\n", ret);
        return ret;
    }
    if ((ret = pdm->PowerControl(ARM_POWER_FULL)) != ARM_DRIVER_OK) {
        printf("MIC: PowerControl failed (%" PRId32 ")\n", ret);
        return ret;
    }
    ret = pdm->Control(ARM_PDM_SELECT_CHANNEL, ARM_PDM_MASK_CHANNEL_4 | ARM_PDM_MASK_CHANNEL_5, 0);
    if (ret != ARM_DRIVER_OK) {
        printf("MIC: channel select failed (%" PRId32 ")\n", ret);
        return ret;
    }
    if ((ret = pdm->Control(ARM_PDM_MODE, ARM_PDM_MODE_AUDIOFREQ_16K_DECM_64, 0)) != ARM_DRIVER_OK) {
        printf("MIC: mode 16K_DECM_64 failed (%" PRId32 ")\n", ret);
        return ret;
    }
    /* As demo_pdm.c ("Select the DC blocking IIR filter") */
    if ((ret = pdm->Control(ARM_PDM_BYPASS_IIR_FILTER, 1, 0)) != ARM_DRIVER_OK) {
        printf("MIC: IIR filter control failed (%" PRId32 ")\n", ret);
        return ret;
    }
    for (unsigned c = 0; c < MIC_CHANNELS; c++) {
        static PDM_CH_CONFIG cfg;
        const uint32_t       ch = mic_ch[c].ch;
        if ((ret = pdm->Control(ARM_PDM_CHANNEL_PHASE, ch, mic_ch[c].phase)) != ARM_DRIVER_OK ||
            (ret = pdm->Control(ARM_PDM_CHANNEL_GAIN, ch, mic_ch[c].gain)) != ARM_DRIVER_OK ||
            (ret = pdm->Control(ARM_PDM_CHANNEL_PEAK_DETECT_TH, ch, mic_ch[c].peak_th)) != ARM_DRIVER_OK ||
            (ret = pdm->Control(ARM_PDM_CHANNEL_PEAK_DETECT_ITV, ch, mic_ch[c].peak_itv)) != ARM_DRIVER_OK) {
            printf("MIC: channel %" PRIu32 " control failed (%" PRId32 ")\n", ch, ret);
            return ret;
        }
        cfg.ch_num = (uint8_t) ch;
        memcpy(cfg.ch_fir_coef, mic_ch[c].fir, sizeof cfg.ch_fir_coef);
        cfg.ch_iir_coef = 0x00000004;
        if ((ret = pdm->Config(&cfg)) != ARM_DRIVER_OK) {
            printf("MIC: channel %" PRIu32 " Config failed (%" PRId32 ")\n", ch, ret);
            return ret;
        }
    }
    return ARM_DRIVER_OK;
}

/*
  Start the PDM microphones. Prints each failure. Returns 0 when capturing.
*/
int board_mic_init(void)
{
    ARM_DRIVER_VERSION ver = pdm->GetVersion();
    uint32_t           service_error_code = 0;
    uint32_t           err;
    int32_t            ret;

    printf("MIC: PDM driver api 0x%04x drv 0x%04x, ch 4+5, %u Hz, %u-sample blocks\n",
           ver.api, ver.drv, MIC_RATE_HZ, MIC_BLOCK);

#if SOC_FEAT_CLK76P8M_CLK_ENABLE
    /* main.c has run se_services_port_init() */
    err = SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_HFOSCx2, true,
                                       &service_error_code);
    if (err != SERVICES_REQ_SUCCESS) {
        printf("MIC: HFOSCx2 clock enable failed (SE %" PRIu32 ", service %" PRIu32 ")\n",
               err, service_error_code);
        return -1;
    }
#endif

    if (mic_setup() != ARM_DRIVER_OK) {
        return -1;
    }

    /* DWT cycle counter for the re-arm timing */
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
    if (DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) {
        printf("MIC: no DWT cycle counter, timing not measured\n");
    }
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    mic_arm_cyc = DWT->CYCCNT;
    if ((ret = pdm->Receive(mic_buf[0], MIC_BLOCK)) != ARM_DRIVER_OK) {
        printf("MIC: Receive failed (%" PRId32 ")\n", ret);
        return -1;
    }
    /* No completion can come before the block is full, 100 ms from now */
    mic_capturing = 1;
    mic_running   = 1;
    printf("MIC: capturing into %u x %u ms blocks, printing every %u ms "
           "(%" PRIu32 " PDM events before the capture ignored)\n",
           MIC_NBLK, MIC_BLOCK_MS, MIC_PRINT_MS, mic_pre_events);
    return 0;
}

/*
  Accumulate and release the finished blocks, restart the capture after a
  ring-full stop, and print the statistics every MIC_PRINT_MS; call from the
  main loop.
*/
void board_mic_poll(uint32_t now_ms)
{
    if (!mic_running) {
        return;
    }

    const uint32_t cyc_per_us = SystemCoreClock / 1000000u;

    const uint32_t errs = mic_err_events;
    if (errs != mic_err_seen) {
        mic_err_seen = errs;
        win_err_at   = (int32_t) ((mic_err_cyc - mic_arm_cyc) / cyc_per_us);
    }

    while (mic_used != mic_filled) {
#ifdef APP_SENTINEL
        sentinel_mic_block(mic_buf[mic_used % MIC_NBLK], MIC_BLOCK);
#endif
        mic_accumulate(mic_buf[mic_used % MIC_NBLK]);
        mic_total += MIC_BLOCK;
        mic_used++;
    }

    if (mic_stopped) {
        if (mic_rearm_fail) {
            mic_running = 0;
        } else {
            /* Ring-full stop: the callback did not re-arm, so no PDM
               interrupt can change the ring until mic_arm() */
            const uint32_t stop_us = (DWT->CYCCNT - mic_stop_cyc) / cyc_per_us;
            if (stop_us > win_stop_max) {
                win_stop_max = stop_us;
            }
            mic_stopped = 0;
            mic_arm();
        }
    }

#ifdef APP_SENTINEL
    /* sentinel.c prints its own line; report a stop only */
    if (mic_running) {
        return;
    }
#endif
    if ((int32_t) (now_ms - mic_next_print) < 0 && mic_running) {
        return;
    }
    mic_next_print = now_ms + MIC_PRINT_MS;

    printf("MIC t=%" PRIu32 " n=%" PRIu64 " blk=%" PRIu32, now_ms, mic_total, win_blocks);
    for (unsigned c = 0; c < MIC_CHANNELS; c++) {
        const int32_t  mean = win_blocks ? (int32_t) (win_mean_sum[c] / win_blocks) : 0;
        const uint32_t rms  = win_n[c] ? (uint32_t) (sqrt((double) win_sq[c] / win_n[c]) + 0.5) : 0;
        printf(" ch%u mean=%+" PRId32 " rms=%" PRIu32 " pk=%" PRId32, (unsigned) mic_ch[c].ch,
               mean, rms, win_peak[c]);
        win_mean_sum[c] = 0;
        win_sq[c]       = 0;
        win_n[c]        = 0;
        win_peak[c]     = 0;
    }
    printf(" done=%" PRIu32 " err=%" PRIu32 " det=%" PRIu32 " rearm_fail=%" PRIu32
           " full=%" PRIu32 " stop_max_us=%" PRIu32 " err_at_us=%" PRId32 "%s\n",
           mic_done_events, mic_err_events, mic_detect_events, mic_rearm_fail,
           mic_full_events, win_stop_max, win_err_at, mic_running ? "" : " STOPPED");
    win_blocks   = 0;
    win_stop_max = 0;
    win_err_at   = -1;
}

/*
  Capture counters since board_mic_init(): completed blocks, FIFO overflow
  errors and ring-full stops.
*/
void board_mic_counters(uint32_t *done, uint32_t *err, uint32_t *full)
{
    *done = mic_done_events;
    *err  = mic_err_events;
    *full = mic_full_events;
}
