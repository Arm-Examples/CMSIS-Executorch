#ifndef MIC_H_
#define MIC_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Starts background capture from the on-board PDM mics. Returns 0 on success. */
int mic_init(void);

/* Non-blocking. 1 = a new block arrived and *level holds its RMS loudness
 * (int16 sample units), 0 = nothing new yet, -1 = the block had an error. */
int mic_poll(float *level);

uint32_t mic_block_count(void);

/* Recording for speech: 2 s at 8 kHz (was 1.5 s: people who paused got cut off). Starts now; mic_poll() pauses meanwhile. */
#define MIC_REC_SAMPLES 16000U
int mic_record_start(void);

/* Non-blocking. 1 = done: *pcm holds *n mono samples (the louder mic) until the
 * next recording; 0 = still recording; -1 = not recording. */
int mic_record_poll(const int16_t **pcm, int *n);

void mic_record_cancel(void);

#ifdef __cplusplus
}
#endif

#endif /* MIC_H_ */
