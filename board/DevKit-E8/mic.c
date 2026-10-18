/*---------------------------------------------------------------------------
 * On-board PDM microphones of the DevKit-E8 (PDM channels 4 and 5).
 *
 * Configuration (channels, mode, FIR coefficients, phase, gain) is taken from
 * Alif's bare-metal demo, Boards/Templates/Baremetal/demo_pdm.c. The 76.8 MHz
 * clock the PDM block needs on the E8 is requested in main().
 *
 * Capture runs in the background: mic_poll() returns a new loudness value each
 * time a block of samples has arrived and immediately starts the next block.
 *---------------------------------------------------------------------------*/

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#include "Driver_PDM.h"

#include "mic.h"

#define CHANNEL_4           4
#define CHANNEL_5           5
#define CH4_PHASE           0x0000001F
#define CH4_GAIN            0x0000000D
#define CH4_PEAK_DETECT_TH  0x00060002
#define CH4_PEAK_DETECT_ITV 0x0004002D
#define CH5_PHASE           0x00000003
#define CH5_GAIN            0x00000013
#define CH5_PEAK_DETECT_TH  0x00060002
#define CH5_PEAK_DETECT_ITV 0x00020027

#define FRAMES  256U              /* per channel: 32 ms at 8 kHz */
#define SAMPLES (FRAMES * 2U)     /* channels 4 and 5 interleaved */

static const uint32_t ch4_fir[18] = {0x00000001, 0x00000003, 0x00000003, 0x000007F4, 0x00000004,
                                     0x000007ED, 0x000007F5, 0x000007F4, 0x000007D3, 0x000007FE,
                                     0x000007BC, 0x000007E5, 0x000007D9, 0x00000793, 0x00000029,
                                     0x0000072C, 0x00000072, 0x000002FD};
static const uint32_t ch5_fir[18] = {0x00000000, 0x000007FF, 0x00000000, 0x00000004, 0x00000004,
                                     0x000007FC, 0x00000000, 0x000007FB, 0x000007E4, 0x00000000,
                                     0x0000002B, 0x00000009, 0x00000016, 0x00000049, 0x00000793,
                                     0x000006F8, 0x00000045, 0x00000178};

extern ARM_DRIVER_PDM Driver_PDM;

static int16_t           mic_buf[SAMPLES];
static int16_t           rec_buf[MIC_REC_SAMPLES * 2U];  /* a recording, both channels */
static volatile int      recording;
static volatile uint32_t mic_event;
static uint32_t          mic_blocks;

static void mic_callback(uint32_t event)
{
    mic_event |= event;
}

static int32_t channel_setup(uint32_t ch, uint32_t phase, uint32_t gain, uint32_t th, uint32_t itv,
                             const uint32_t *fir)
{
    PDM_CH_CONFIG cfg;
    int32_t r = Driver_PDM.Control(ARM_PDM_CHANNEL_PHASE, ch, phase);
    r |= Driver_PDM.Control(ARM_PDM_CHANNEL_GAIN, ch, gain);
    r |= Driver_PDM.Control(ARM_PDM_CHANNEL_PEAK_DETECT_TH, ch, th);
    r |= Driver_PDM.Control(ARM_PDM_CHANNEL_PEAK_DETECT_ITV, ch, itv);
    cfg.ch_num = (uint8_t)ch;
    memcpy(cfg.ch_fir_coef, fir, sizeof(cfg.ch_fir_coef));
    cfg.ch_iir_coef = 0x00000004;
    r |= Driver_PDM.Config(&cfg);
    return r;
}

static int32_t start_block(void)
{
    mic_event = 0;
    return Driver_PDM.Receive(mic_buf, SAMPLES);
}

int mic_init(void)
{
    int32_t r;
    r = Driver_PDM.Initialize(mic_callback);
    printf("mic: Initialize -> %ld\n", (long)r);
    if (r != ARM_DRIVER_OK) return -1;
    r = Driver_PDM.PowerControl(ARM_POWER_FULL);
    printf("mic: PowerControl -> %ld\n", (long)r);
    if (r != ARM_DRIVER_OK) return -2;
    r = Driver_PDM.Control(ARM_PDM_SELECT_CHANNEL, ARM_PDM_MASK_CHANNEL_4 | ARM_PDM_MASK_CHANNEL_5, 0);
    r |= Driver_PDM.Control(ARM_PDM_MODE, ARM_PDM_MODE_AUDIOFREQ_8K_DECM_64, 0);
    r |= Driver_PDM.Control(ARM_PDM_BYPASS_IIR_FILTER, 1U, 0); /* 1 = ENABLE, as in demo_pdm.c */
    printf("mic: channels/mode/filter -> %ld\n", (long)r);
    if (r != ARM_DRIVER_OK) return -3;
    r = channel_setup(CHANNEL_4, CH4_PHASE, CH4_GAIN, CH4_PEAK_DETECT_TH, CH4_PEAK_DETECT_ITV, ch4_fir);
    r |= channel_setup(CHANNEL_5, CH5_PHASE, CH5_GAIN, CH5_PEAK_DETECT_TH, CH5_PEAK_DETECT_ITV, ch5_fir);
    printf("mic: channel 4/5 setup -> %ld\n", (long)r);
    if (r != ARM_DRIVER_OK) return -4;
    r = start_block();
    printf("mic: first Receive -> %ld\n", (long)r);
    return (r == ARM_DRIVER_OK) ? 0 : -5;
}

/* RMS of one channel with its DC offset removed */
static float channel_rms(int offset)
{
    int64_t sum = 0, sq = 0;
    for (uint32_t i = 0; i < FRAMES; ++i) {
        int32_t v = mic_buf[2U * i + (uint32_t)offset];
        sum += v;
        sq += (int64_t)v * v;
    }
    float mean = (float)sum / FRAMES;
    float var = (float)sq / FRAMES - mean * mean;
    return var > 0.0f ? sqrtf(var) : 0.0f;
}

int mic_poll(float *level)
{
    if (recording) return 0;  /* the recording owns the mics */
    if (!(mic_event & (ARM_PDM_EVENT_CAPTURE_COMPLETE | ARM_PDM_EVENT_ERROR))) {
        return 0;
    }
    int ok = (mic_event & ARM_PDM_EVENT_CAPTURE_COMPLETE) != 0;
    if (ok) {
        float a = channel_rms(0), b = channel_rms(1);
        *level = a > b ? a : b;  /* the louder of the two mics */
        ++mic_blocks;
    }
    start_block();
    return ok ? 1 : -1;
}

int mic_record_start(void)
{
    /* the driver takes a new buffer even while a block is running: switch to the long one */
    __disable_irq();
    recording = 1;
    mic_event = 0;
    int32_t r = Driver_PDM.Receive(rec_buf, MIC_REC_SAMPLES * 2U);
    __enable_irq();
    if (r != ARM_DRIVER_OK) {
        recording = 0;
        start_block();
        return -1;
    }
    return 0;
}

int mic_record_poll(const int16_t **pcm, int *n)
{
    if (!recording) return -1;
    if (!(mic_event & ARM_PDM_EVENT_CAPTURE_COMPLETE)) return 0;  /* errors (overflow) do not stop a capture */
    /* keep the louder mic, as mono, in place */
    int64_t s0 = 0, s1 = 0, q0 = 0, q1 = 0;
    for (uint32_t i = 0; i < MIC_REC_SAMPLES; ++i) {
        int32_t a = rec_buf[2U * i], b = rec_buf[2U * i + 1U];
        s0 += a; s1 += b; q0 += (int64_t)a * a; q1 += (int64_t)b * b;
    }
    float v0 = (float)q0 - (float)s0 * (float)s0 / MIC_REC_SAMPLES;
    float v1 = (float)q1 - (float)s1 * (float)s1 / MIC_REC_SAMPLES;
    uint32_t ch = v1 > v0 ? 1U : 0U;
    for (uint32_t i = 0; i < MIC_REC_SAMPLES; ++i) rec_buf[i] = rec_buf[2U * i + ch];
    recording = 0;
    start_block();  /* back to the loudness blocks */
    *pcm = rec_buf;
    *n = (int)MIC_REC_SAMPLES;
    return 1;
}

void mic_record_cancel(void)
{
    if (!recording) return;
    __disable_irq();
    recording = 0;
    start_block();
    __enable_irq();
}

uint32_t mic_block_count(void)
{
    return mic_blocks;
}
