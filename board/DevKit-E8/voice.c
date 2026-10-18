/*---------------------------------------------------------------------------
 * Keyword spotting: a small DS-CNN trained on Google Speech Commands
 * (model/kws/train_kws.py) runs on the Ethos-U85; works for any voice.
 *
 * The 2 s push-to-talk recording -> find the word (the loud burst around
 * the loudest moment) -> 1 s window centred on it -> 49 x 40 log-mel
 * features (exactly as model/kws/kws_features.py) -> NPU -> softmax.
 *---------------------------------------------------------------------------*/

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#include "voice.h"

extern int kws_classify(const float *in, float *scores, int n);  /* src/app_main.cpp */

/* ---- finding the word: loudness in 25 ms frames every 10 ms ---- */
#define DET_FRAME   200
#define DET_HOP     80
#define SKIP_FR     6       /* the joystick click at the start */
#define MIN_DB      10.0f   /* the word must be this much louder than the room */
#define WORD_DB     20.0f   /* the word = frames within this of its loudest frame, */
#define GAP_FR      12      /* joined across quiet gaps up to 120 ms               */

/* ---- features: must match model/kws/kws_features.py ---- */
#define SR          8000
#define N_SAMPLES   8000
#define FRAME       240
#define HOP         160
#define NFFT        256
#define NBIN        (NFFT / 2 + 1)
#define NMEL        40
#define NFR         ((N_SAMPLES - FRAME) / HOP + 1)  /* 49 */
#define TARGET_RMS  0.1f
#define MAX_GAIN    400.0f
#define FEAT_STD    3.0f

static const char *const class_names[KWS_N_CLASSES] = {
    "up", "down", "left", "right", "go", "stop", "yes", "no", "happy", "silence", "unknown",
};

volatile uint32_t g_kws_us;          /* last keyword model run on the NPU, us */
volatile int32_t g_voice_w0 = -1;  /* debugger-readable: where the last 1 s window started */
static float fr_db[208];             /* loudness frames: 2 s = 198 */
static float xbuf[N_SAMPLES];
float g_voice_feat[NFR * NMEL];  /* debugger-readable: the last features */
#define feat g_voice_feat
static float win[FRAME];
static float cos_t[NFFT / 2], sin_t[NFFT / 2];
static float mel_w[NMEL][NBIN];
static int   ready;

static void init(void)
{
    const float pi = 3.14159265f;
    for (int i = 0; i < FRAME; ++i) win[i] = 0.54f - 0.46f * cosf(2.0f * pi * (float)i / (FRAME - 1));
    for (int i = 0; i < NFFT / 2; ++i) {
        cos_t[i] = cosf(2.0f * pi * (float)i / NFFT);
        sin_t[i] = -sinf(2.0f * pi * (float)i / NFFT);
    }
    float m0 = 2595.0f * log10f(1.0f + 20.0f / 700.0f), m1 = 2595.0f * log10f(1.0f + 4000.0f / 700.0f);
    float edge[NMEL + 2];
    for (int k = 0; k < NMEL + 2; ++k) {
        float m = m0 + (m1 - m0) * (float)k / (NMEL + 1);
        edge[k] = 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f) * NFFT / SR;
    }
    for (int m = 0; m < NMEL; ++m) {
        for (int b = 0; b < NBIN; ++b) {
            float f = (float)b, w = 0.0f;
            if (f > edge[m] && f <= edge[m + 1]) w = (f - edge[m]) / (edge[m + 1] - edge[m]);
            else if (f > edge[m + 1] && f < edge[m + 2]) w = (edge[m + 2] - f) / (edge[m + 2] - edge[m + 1]);
            mel_w[m][b] = w;
        }
    }
    ready = 1;
}

/* in-place radix-2 FFT of 256 complex values */
static void fft(float *re, float *im)
{
    for (int i = 1, j = 0; i < NFFT; ++i) {
        int bit = NFFT >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= NFFT; len <<= 1) {
        int step = NFFT / len;
        for (int i = 0; i < NFFT; i += len) {
            for (int k = 0; k < len / 2; ++k) {
                float wr = cos_t[k * step], wi = sin_t[k * step];
                float *ar = &re[i + k], *ai = &im[i + k], *br = &re[i + k + len / 2], *bi = &im[i + k + len / 2];
                float tr = *br * wr - *bi * wi, ti = *br * wi + *bi * wr;
                *br = *ar - tr; *bi = *ai - ti;
                *ar += tr; *ai += ti;
            }
        }
    }
}

/* where the word is: sample index of its middle, or -1 if nothing was said */
static int find_word(const int16_t *pcm, int n)
{
    int nf = (n - DET_FRAME) / DET_HOP + 1;
    if (nf > (int)(sizeof(fr_db) / sizeof(fr_db[0]))) nf = (int)(sizeof(fr_db) / sizeof(fr_db[0]));
    int64_t sum = 0;
    for (int i = 0; i < n; ++i) sum += pcm[i];
    float mean = (float)sum / (float)n;
    float lo[32], peak = -1e9f;
    int nlo = 0, pk = SKIP_FR;
    for (int f = 0; f < nf; ++f) {
        const int16_t *x = pcm + f * DET_HOP;
        float e = 0.0f;
        for (int i = 0; i < DET_FRAME; ++i) { float v = (float)x[i] - mean; e += v * v; }
        fr_db[f] = 10.0f * log10f(e / DET_FRAME + 1.0f);
        if (f < SKIP_FR) continue;
        if (fr_db[f] > peak) { peak = fr_db[f]; pk = f; }
        if (nlo < 32) lo[nlo++] = fr_db[f];
        else {
            int mx = 0;
            for (int k = 1; k < 32; ++k) if (lo[k] > lo[mx]) mx = k;
            if (fr_db[f] < lo[mx]) lo[mx] = fr_db[f];
        }
    }
    float floor_db = 0.0f;
    for (int k = 0; k < nlo; ++k) floor_db += lo[k];
    floor_db /= (float)(nlo ? nlo : 1);
    if (peak - floor_db < MIN_DB) {
        printf("voice: nothing heard (peak %d dB over the room)\n", (int)(peak - floor_db));
        return -1;
    }
    float thr = floor_db + 0.3f * (peak - floor_db);
    if (thr < peak - WORD_DB) thr = peak - WORD_DB;
    int s = pk, e = pk;
    for (int f = pk - 1, gap = 0; f >= SKIP_FR && gap <= GAP_FR; --f) {
        if (fr_db[f] > thr) { s = f; gap = 0; } else ++gap;
    }
    for (int f = pk + 1, gap = 0; f < nf && gap <= GAP_FR; ++f) {
        if (fr_db[f] > thr) { e = f; gap = 0; } else ++gap;
    }
    printf("voice: word %d ms long, %d dB over the room\n", (e - s + 1) * 10, (int)(peak - floor_db));
    return ((s + e) * DET_HOP + DET_FRAME) / 2;
}

static void features(const int16_t *pcm)
{
    static float re[NFFT], im[NFFT];
    float mean = 0.0f;
    for (int i = 0; i < N_SAMPLES; ++i) { xbuf[i] = (float)pcm[i] / 32768.0f; mean += xbuf[i]; }
    mean /= N_SAMPLES;
    for (int i = 0; i < N_SAMPLES; ++i) xbuf[i] -= mean;
    float rms = 0.0f;
    for (int f = 0; f < NFR; ++f) {
        float e = 0.0f;
        for (int i = 0; i < FRAME; ++i) e += xbuf[f * HOP + i] * xbuf[f * HOP + i];
        e = sqrtf(e / FRAME);
        if (e > rms) rms = e;
    }
    float g = TARGET_RMS / (rms > 1e-9f ? rms : 1e-9f);
    if (g > MAX_GAIN) g = MAX_GAIN;
    for (int f = 0; f < NFR; ++f) {
        for (int i = 0; i < NFFT; ++i) {
            re[i] = i < FRAME ? xbuf[f * HOP + i] * g * win[i] : 0.0f;
            im[i] = 0.0f;
        }
        fft(re, im);
        for (int b = 0; b < NBIN; ++b) re[b] = re[b] * re[b] + im[b] * im[b];
        for (int m = 0; m < NMEL; ++m) {
            float e = 0.0f;
            for (int b = 0; b < NBIN; ++b) if (mel_w[m][b] > 0.0f) e += mel_w[m][b] * re[b];
            feat[f * NMEL + m] = logf(e + 1e-6f);
        }
    }
    for (int m = 0; m < NMEL; ++m) {  /* minus each band's mean, scaled */
        float s = 0.0f;
        for (int f = 0; f < NFR; ++f) s += feat[f * NMEL + m];
        s /= NFR;
        for (int f = 0; f < NFR; ++f) feat[f * NMEL + m] = (feat[f * NMEL + m] - s) / FEAT_STD;
    }
}

int voice_listen(const int16_t *pcm, int n, float *conf)
{
    if (!ready) init();
    *conf = 0.0f;
    int mid = find_word(pcm, n);
    if (mid < 0) return KWS_SILENCE;
    int w0 = mid - N_SAMPLES / 2;
    if (w0 < 0) w0 = 0;
    if (w0 > n - N_SAMPLES) w0 = n - N_SAMPLES;
    g_voice_w0 = w0;
    features(pcm + w0);
    float sc[KWS_N_CLASSES];
    uint32_t c0 = DWT->CYCCNT;
    int k = kws_classify(feat, sc, KWS_N_CLASSES);
    g_kws_us = (DWT->CYCCNT - c0) / (SystemCoreClock / 1000000U);
    if (k != KWS_N_CLASSES) {
        printf("voice: kws_classify failed (%d)\n", k);
        return -1;
    }
    float mx = sc[0], sum = 0.0f;
    int best = 0;
    for (int i = 1; i < KWS_N_CLASSES; ++i) if (sc[i] > mx) { mx = sc[i]; best = i; }
    for (int i = 0; i < KWS_N_CLASSES; ++i) { sc[i] = expf(sc[i] - mx); sum += sc[i]; }
    *conf = sc[best] / sum;
    printf("voice: heard \"%s\" %d%% (", class_names[best], (int)(*conf * 100.0f));
    for (int i = 0; i < KWS_N_CLASSES; ++i) printf("%s%s %d", i ? ", " : "", class_names[i], (int)(sc[i] / sum * 100.0f));
    printf(")\n");
    return best;
}

const char *voice_class_name(int c)
{
    return (c >= 0 && c < KWS_N_CLASSES) ? class_names[c] : "?";
}
