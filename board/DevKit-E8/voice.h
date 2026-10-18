#ifndef VOICE_H_
#define VOICE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Keyword spotting on the Ethos-U85 (Google Speech Commands words, any voice). */

enum {
    KWS_UP, KWS_DOWN, KWS_LEFT, KWS_RIGHT, KWS_GO, KWS_STOP, KWS_YES, KWS_NO, KWS_HAPPY,
    KWS_SILENCE, KWS_UNKNOWN, KWS_N_CLASSES
};

/* Finds the word in 8 kHz mono samples (at least 1 s) and classifies it.
 * Returns a KWS_ class (KWS_SILENCE if nothing was said, -1 on error);
 * *conf = its probability. */
int voice_listen(const int16_t *pcm, int n, float *conf);

const char *voice_class_name(int c);

#ifdef __cplusplus
}
#endif

#endif /* VOICE_H_ */
