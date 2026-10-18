/*---------------------------------------------------------------------------
 * Bunny motion clips: a C port of model/bunny/clips.py (same formulas), and
 * the dial ranges of model/bunny/bunny_rig3.py (SCALARS).
 *---------------------------------------------------------------------------*/

#include <math.h>
#include <string.h>

#include "bunny_clips.h"

#define PI  3.14159265f
#define TAU 6.28318531f

/* (default, lo, hi) of the 21 scalar dials, in NN order (kit_v2 bunny_rig3.SCALARS) */
static const float scalars[21][3] = {
    {-12, -75, 15}, {12, -15, 75}, {0, -10, 90}, {128, 108, 148},
    {0, -7, 7}, {0, -6, 6},
    {0, -30, 160}, {0, -160, 30}, {0, -20, 6}, {0, -20, 6},
    {1.0f, 0.82f, 1.14f}, {0, -14, 14}, {0, 0, 1}, {0, -16, 16},
    {0.85f, 0.4f, 1.0f}, {0, 0, 1},
    {0, -0.07f, 0.07f}, {1.0f, 0.8f, 1.15f}, {1.0f, 0.92f, 1.08f},
    {0, 0, 1}, {0, 0, 1},  /* brow, dizzy */
};

/* NN input normalisation: lo/hi of each dial from model/bunny/dial_ranges_v2.json
 * (what the v2 decoder was trained with) */
static const float norm_lo[BUNNY_NDIAL] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -75.0f, -15.0f, -10.0f, 108.0007f, -7.0f, -6.0f, -30.0f, -160.0f, -20.0f, -20.0f, 0.82f, -14.0f, 0.0f, -16.0f, 0.4f, 0.0f, -0.06999239f, 0.8000113f, 0.9200003f, 0.0f, 0.0f};
static const float norm_hi[BUNNY_NDIAL] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 15.0f, 75.0f, 90.0f, 147.9995f, 7.0f, 6.0f, 160.0f, 30.0f, 6.0f, 6.0f, 1.14f, 14.0f, 1.0f, 16.0f, 1.0f, 1.0f, 0.06999976f, 1.149991f, 1.08f, 1.0f, 1.0f};


static const char *const names[CLIP_COUNT] = {
    "idle", "look", "hop", "walk", "run", "wave", "shake", "lie_down", "sleep",
    "sit", "yawn", "sad", "eat", "love", "stretch", "clap", "angry", "dizzy", "confused", "dance",
};

static float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static float maxf(float a, float b) { return a > b ? a : b; }
static float minf(float a, float b) { return a < b ? a : b; }

static float smooth(float t)
{
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return t * t * (3.0f - 2.0f * t);
}

/* Python's t % m for t >= 0 */
static float modf_pos(float t, float m)
{
    float r = fmodf(t, m);
    return r < 0.0f ? r + m : r;
}

/* P(eye, mouth): default dials with one eye and one mouth shape at weight 1 */
static void P(bunny_pose_t *o, int eye, int mouth)
{
    memset(o, 0, sizeof(*o));
    for (int i = 0; i < 21; ++i) {
        o->d[11 + i] = scalars[i][0];
    }
    o->d[eye] = 1.0f;
    o->d[mouth] = 1.0f;
}

static void blink(bunny_pose_t *o, float t, float period)
{
    float ph = modf_pos(t, period);
    if (ph < 0.16f) {
        float k = sinf(PI * ph / 0.16f);
        o->d[D_EYE_OPEN] = 1.0f - k;
        o->d[D_EYE_SLEEP] = k;
    }
}

static void idle(float t, float a, bunny_pose_t *o)
{
    P(o, D_EYE_OPEN, D_MOUTH_SMILE);
    o->d[D_SQUASH] = 1.0f + 0.012f * sinf(TAU * t / 2.4f) * a;
    blink(o, t, 3.1f);
}

static void look(float t, float a, bunny_pose_t *o)
{
    idle(t, a, o);
    o->d[D_LOOK_X] = 6.0f * a * sinf(TAU * t / 3.5f);
    o->d[D_LOOK_Y] = 2.0f * a * sinf(TAU * t / 2.1f);
    o->d[D_HEAD_TILT] = 7.0f * a * sinf(TAU * t / 3.5f + 0.4f);
}

static void hop(float t, float a, bunny_pose_t *o)
{
    float ph = modf_pos(t, 0.8f) / 0.8f;
    float h = sinf(PI * ph);
    float land = ph > 0.5f ? maxf(0.0f, -sinf(TAU * ph)) : 0.0f;
    P(o, D_EYE_HAPPY, D_MOUTH_OPEN);
    o->d[D_EAR_L] = -10.0f - 12.0f * h * a;
    o->d[D_EAR_R] = 10.0f + 12.0f * h * a;
    o->d[D_CURL] = 30.0f * h * a;
    o->d[D_SQUASH] = 1.0f + 0.14f * h * a - 0.10f * land * a;
    o->d[D_ARM_L] = 30.0f * h * a;
    o->d[D_ARM_R] = -30.0f * h * a;
    o->d[D_FOOT_L] = -14.0f * h * a;
    o->d[D_FOOT_R] = -14.0f * h * a;
    o->dy = -46.0f * h * a;
}

static void walk(float t, float a, bunny_pose_t *o)
{
    float ph = TAU * t / 0.8f, s = sinf(ph);
    P(o, D_EYE_OPEN, D_MOUTH_SMILE);
    o->d[D_EAR_L] = -18.0f;
    o->d[D_EAR_R] = 18.0f;
    o->d[D_CURL] = 14.0f;
    o->d[D_SQUASH] = 1.0f + 0.02f * cosf(2.0f * ph) * a;
    o->d[D_TILT] = 2.0f * s * a;
    o->d[D_FOOT_L] = -9.0f * maxf(0.0f, s) * a;
    o->d[D_FOOT_R] = -9.0f * maxf(0.0f, -s) * a;
    o->d[D_ARM_L] = 22.0f * s * a;
    o->d[D_ARM_R] = 22.0f * s * a;  /* clips.py: -22*sin*a*-1 */
    o->dy = -6.0f * fabsf(s) * a;
}

static void run(float t, float a, bunny_pose_t *o)
{
    float ph = TAU * t / 0.42f, s = sinf(ph);
    P(o, D_EYE_OPEN, D_MOUTH_OPEN);
    o->d[D_EAR_L] = -40.0f;
    o->d[D_EAR_R] = 40.0f;
    o->d[D_CURL] = 30.0f;
    o->d[D_SQUASH] = 0.95f + 0.04f * cosf(2.0f * ph) * a;
    o->d[D_TILT] = 6.0f + 2.5f * s * a;
    o->d[D_FOOT_L] = -16.0f * maxf(0.0f, s) * a;
    o->d[D_FOOT_R] = -16.0f * maxf(0.0f, -s) * a;
    o->d[D_ARM_L] = 35.0f * s * a + 20.0f;
    o->d[D_ARM_R] = -(35.0f * sinf(ph + PI) * a + 20.0f);
    o->dy = -14.0f * fabsf(s) * a;
}

static void wave(float t, float a, bunny_pose_t *o)
{
    float ph = TAU * 3.2f * t;
    P(o, D_EYE_HAPPY, D_MOUTH_OPEN);
    o->d[D_EAR_L] = -14.0f;
    o->d[D_EAR_R] = 22.0f + 6.0f * sinf(ph) * a;
    o->d[D_CURL] = 10.0f;
    o->d[D_HEAD_TILT] = -6.0f * a;
    o->d[D_ARM_R] = -(116.0f + 17.0f * sinf(ph));
    o->d[D_ARM_L] = 6.0f;
}

static void shake(float t, float a, bunny_pose_t *o)
{
    float s13 = sinf(TAU * 13.0f * t);
    P(o, D_EYE_WIDE, D_MOUTH_O);
    o->d[D_EAR_L] = -4.0f + 6.0f * s13 * a;
    o->d[D_EAR_R] = 4.0f - 6.0f * s13 * a;
    o->d[D_SQUASH] = 1.08f;
    o->d[D_ARM_L] = 38.0f;
    o->d[D_ARM_R] = -38.0f;
    o->d[D_TILT] = 3.0f * sinf(TAU * 11.0f * t + 1.0f) * a;
    o->d[D_HEAD_TILT] = 4.0f * sinf(TAU * 9.0f * t) * a;
    o->sweat = 1.0f;
    o->dx = 9.0f * sinf(TAU * 11.0f * t) * a;
}

static void lie_down(float t, float a, bunny_pose_t *o)
{
    (void)a;
    float u = smooth(t / 1.4f);
    float ph = TAU * t / 3.2f;
    P(o, D_EYE_OPEN, D_MOUTH_SMILE);
    o->d[D_EYE_OPEN] = 1.0f - u;
    o->d[D_EYE_SLEEP] = u;
    o->d[D_MOUTH_SMILE] = 1.0f - u;
    o->d[D_MOUTH_FLAT] = u;
    o->d[D_LIE] = u;
    o->d[D_EAR_L] = lerpf(-12.0f, -62.0f, u);
    o->d[D_EAR_R] = lerpf(12.0f, 62.0f, u);
    o->d[D_CURL] = lerpf(0.0f, 80.0f, u);
    o->d[D_SQUASH] = lerpf(1.0f, 0.9f, u) + 0.014f * u * sinf(ph);
    o->d[D_BLUSH] = lerpf(0.85f, 0.7f, u);
    o->d[D_TILT] = 0.0f;
    o->zzz = smooth((t - 1.0f) / 0.6f);
}

static void sit(float t, float a, bunny_pose_t *o)
{
    (void)a;
    P(o, D_EYE_OPEN, D_MOUTH_SMILE);
    o->d[D_SQUASH] = 0.9f + 0.012f * sinf(TAU * t / 2.6f);
    o->d[D_FOOT_L] = -10.0f;
    o->d[D_FOOT_R] = -10.0f;
    o->d[D_ARM_L] = -8.0f;
    o->d[D_ARM_R] = 8.0f;
    o->d[D_EAR_L] = -16.0f;
    o->d[D_EAR_R] = 16.0f;
    blink(o, t, 3.4f);
}

static void yawn(float t, float a, bunny_pose_t *o)
{
    (void)a;
    /* sinf(PI) is a tiny negative number in float, and powf(negative, 1.5) is NaN */
    float sv = sinf(PI * minf(1.0f, modf_pos(t, 3.0f) / 2.4f));
    float u = powf(sv > 0.0f ? sv : 0.0f, 1.5f);
    P(o, D_EYE_OPEN, D_MOUTH_SMILE);
    o->d[D_EYE_OPEN] = 1.0f - u;
    o->d[D_EYE_SLEEP] = u;
    o->d[D_MOUTH_SMILE] = 1.0f - u;
    o->d[D_MOUTH_YAWN] = u;
    o->d[D_ARM_L] = lerpf(0.0f, 100.0f, u);
    o->d[D_ARM_R] = -lerpf(0.0f, 100.0f, u);
    o->d[D_EAR_L] = lerpf(-12.0f, -40.0f, u);
    o->d[D_EAR_R] = lerpf(12.0f, 40.0f, u);
    o->d[D_CURL] = lerpf(0.0f, 50.0f, u);
    o->d[D_SQUASH] = lerpf(1.0f, 1.08f, u);
    o->d[D_HEAD_TILT] = -4.0f * u;
}

static void sad(float t, float a, bunny_pose_t *o)
{
    (void)a;
    float u = smooth(t / 0.8f);
    P(o, D_EYE_SAD, D_MOUTH_FROWN);
    o->d[D_EAR_L] = lerpf(-12.0f, -50.0f, u);
    o->d[D_EAR_R] = lerpf(12.0f, 50.0f, u);
    o->d[D_CURL] = lerpf(0.0f, 72.0f, u);
    o->d[D_TEAR] = smooth((t - 0.5f) / 0.6f);
    o->d[D_HEAD_TILT] = -8.0f * u;
    o->d[D_SQUASH] = lerpf(1.0f, 0.95f, u) + 0.008f * sinf(TAU * t / 1.6f);
    o->d[D_BLUSH] = 0.9f;
    o->d[D_LOOK_Y] = 3.0f * u;
}

static void eat(float t, float a, bunny_pose_t *o)
{
    (void)a;
    float ph = TAU * 2.6f * t, s = sinf(ph);
    float c = 0.5f + 0.5f * s;
    P(o, D_EYE_HAPPY, D_MOUTH_OPEN);
    o->d[D_MOUTH_OPEN] = 0.25f + 0.6f * c;
    o->d[D_MOUTH_SMILE] = 0.75f - 0.6f * c;
    o->d[D_ARM_L] = -22.0f - 6.0f * s;
    o->d[D_ARM_R] = 22.0f + 6.0f * s;
    o->d[D_SQUASH] = 1.0f + 0.012f * sinf(2.0f * ph);
    o->d[D_HEAD_TILT] = 3.0f * s;
    o->d[D_BLUSH] = 1.0f;
}

static void love(float t, float a, bunny_pose_t *o)
{
    (void)a;
    float ph = TAU * t / 1.2f, s = fabsf(sinf(ph));
    P(o, D_EYE_HAPPY, D_MOUTH_SMILE);
    o->d[D_BLUSH] = 1.0f;
    o->d[D_EAR_L] = -8.0f;
    o->d[D_EAR_R] = 8.0f;
    o->d[D_SQUASH] = 1.0f + 0.05f * s;
    o->d[D_ARM_L] = 10.0f;
    o->d[D_ARM_R] = -10.0f;
    o->hearts = 0.5f + 0.5f * sinf(ph);
    o->dy = -8.0f * s;
}

static void stretch(float t, float a, bunny_pose_t *o)
{
    (void)a;
    float u = sinf(PI * minf(1.0f, modf_pos(t, 3.0f) / 2.2f));
    P(o, D_EYE_OPEN, D_MOUTH_SMILE);
    o->d[D_EYE_OPEN] = 1.0f - 0.8f * u;
    o->d[D_EYE_HAPPY] = 0.8f * u;
    o->d[D_MOUTH_SMILE] = 1.0f - u;
    o->d[D_MOUTH_YAWN] = 0.7f * u;
    o->d[D_MOUTH_OPEN] = 0.3f * u;
    o->d[D_ARM_L] = lerpf(0.0f, 155.0f, u);
    o->d[D_ARM_R] = -lerpf(0.0f, 155.0f, u);
    o->d[D_SQUASH] = lerpf(1.0f, 1.12f, u);
    o->d[D_FOOT_L] = -4.0f * u;
    o->d[D_FOOT_R] = -4.0f * u;
    o->d[D_EAR_L] = lerpf(-12.0f, -24.0f, u);
    o->d[D_EAR_R] = lerpf(12.0f, 24.0f, u);
}

static void clap(float t, float a, bunny_pose_t *o)
{
    (void)a;
    float s = 0.5f + 0.5f * sinf(TAU * 3.0f * t);
    P(o, D_EYE_HAPPY, D_MOUTH_OPEN);
    o->d[D_ARM_L] = -10.0f - 30.0f * s;
    o->d[D_ARM_R] = 10.0f + 30.0f * s;
    o->d[D_SQUASH] = 1.0f + 0.02f * s;
    o->d[D_EAR_L] = -14.0f;
    o->d[D_EAR_R] = 14.0f;
}

static void angry(float t, float a, bunny_pose_t *o)
{
    (void)a;
    float u = smooth(t / 0.35f), ph = TAU * 6.0f * t;
    P(o, D_EYE_OPEN, D_MOUTH_FROWN);
    o->d[D_BROW] = u;
    o->d[D_EYE_OPEN] = 1.0f - 0.35f * u;
    o->d[D_EYE_SAD] = 0.35f * u;
    o->d[D_EAR_L] = lerpf(-12.0f, -46.0f, u);
    o->d[D_EAR_R] = lerpf(12.0f, 46.0f, u);
    o->d[D_CURL] = lerpf(0.0f, 55.0f, u);
    o->d[D_SQUASH] = lerpf(1.0f, 1.04f, u) + 0.01f * sinf(ph);
    o->d[D_BLUSH] = lerpf(0.85f, 0.45f, u);
    o->d[D_TILT] = 1.2f * sinf(ph) * u;
    o->d[D_ARM_L] = lerpf(0.0f, 22.0f, u);
    o->d[D_ARM_R] = -lerpf(0.0f, 22.0f, u);
    o->dx = 2.0f * sinf(ph) * u;
}

static void dizzy(float t, float a, bunny_pose_t *o)
{
    (void)a;
    float ph = TAU * t / 1.1f;
    P(o, D_EYE_OPEN, D_MOUTH_O);
    o->d[D_DIZZY] = 1.0f;
    o->d[D_MOUTH_O] = 0.5f + 0.2f * sinf(2.0f * ph);
    o->d[D_MOUTH_FLAT] = 0.5f - 0.2f * sinf(2.0f * ph);
    o->d[D_TILT] = 8.0f * sinf(ph);
    o->d[D_HEAD_TILT] = 8.0f * sinf(ph + 1.0f);
    o->d[D_EAR_L] = -20.0f + 10.0f * sinf(ph);
    o->d[D_EAR_R] = 20.0f + 10.0f * sinf(ph + 2.0f);
    o->d[D_CURL] = 30.0f;
    o->d[D_SQUASH] = 1.0f + 0.02f * sinf(2.0f * ph);
    o->d[D_ARM_L] = 18.0f;
    o->d[D_ARM_R] = -18.0f;
    o->dx = 8.0f * sinf(ph);
}

static void confused(float t, float a, bunny_pose_t *o)
{
    (void)a;
    float u = smooth(t / 0.5f);
    P(o, D_EYE_OPEN, D_MOUTH_FLAT);
    o->d[D_HEAD_TILT] = lerpf(0.0f, 13.0f, u) + 2.0f * sinf(TAU * t / 1.8f);
    o->d[D_LOOK_Y] = -4.0f * u;
    o->d[D_LOOK_X] = 3.0f * sinf(TAU * t / 2.2f);
    o->d[D_EAR_L] = lerpf(-12.0f, -22.0f, u);
    o->d[D_EAR_R] = lerpf(12.0f, 4.0f, u);
    o->d[D_CURL] = lerpf(0.0f, 25.0f, u);
    o->d[D_SQUASH] = 1.0f + 0.01f * sinf(TAU * t / 2.4f);
}

/* a little 120 bpm dance: sway side to side with a bounce on every beat;
 * the arms alternate between waving up high and pumping, every 2 s */
static void dance(float t, float a, bunny_pose_t *o)
{
    (void)a;
    float beat = t / 0.5f;
    float sw = sinf(PI * beat);          /* one side per beat */
    float b = fabsf(sw);                 /* up on every beat */
    int bar = (int)(t / 2.0f) & 1;
    P(o, D_EYE_HAPPY, D_MOUTH_OPEN);
    o->d[D_TILT] = 9.0f * sw;
    o->d[D_HEAD_TILT] = -10.0f * sw;
    o->d[D_SQUASH] = 0.96f + 0.1f * b;
    o->d[D_EAR_L] = -14.0f - 14.0f * sw;
    o->d[D_EAR_R] = 14.0f - 14.0f * sw;
    o->d[D_CURL] = 12.0f * b;
    if (bar == 0) {  /* hands in the air, waving with the beat */
        o->d[D_ARM_L] = 120.0f + 25.0f * sw;
        o->d[D_ARM_R] = -(120.0f - 25.0f * sw);
    } else {         /* pump one arm, then the other */
        o->d[D_ARM_L] = sw > 0.0f ? 20.0f + 110.0f * sw : 10.0f;
        o->d[D_ARM_R] = sw < 0.0f ? -(20.0f - 110.0f * sw) : -10.0f;
    }
    o->d[D_FOOT_L] = sw > 0.0f ? -12.0f * sw : 0.0f;  /* step on the beat */
    o->d[D_FOOT_R] = sw < 0.0f ? 12.0f * sw : 0.0f;
    o->dx = 28.0f * sw;
    o->dy = -16.0f * b;
}

void bunny_clip(bunny_clip_t clip, float t, float a, bunny_pose_t *o)
{
    switch (clip) {
    case CLIP_LOOK:     look(t, a, o); break;
    case CLIP_HOP:      hop(t, a, o); break;
    case CLIP_WALK:     walk(t, a, o); break;
    case CLIP_RUN:      run(t, a, o); break;
    case CLIP_WAVE:     wave(t, a, o); break;
    case CLIP_SHAKE:    shake(t, a, o); break;
    case CLIP_LIE_DOWN: lie_down(t, a, o); break;
    case CLIP_SLEEP:    lie_down(1.4f + t, a, o); break;
    case CLIP_SIT:      sit(t, a, o); break;
    case CLIP_YAWN:     yawn(t, a, o); break;
    case CLIP_SAD:      sad(t, a, o); break;
    case CLIP_EAT:      eat(t, a, o); break;
    case CLIP_LOVE:     love(t, a, o); break;
    case CLIP_STRETCH:  stretch(t, a, o); break;
    case CLIP_CLAP:     clap(t, a, o); break;
    case CLIP_ANGRY:    angry(t, a, o); break;
    case CLIP_DIZZY:    dizzy(t, a, o); break;
    case CLIP_CONFUSED: confused(t, a, o); break;
    case CLIP_DANCE:    dance(t, a, o); break;
    case CLIP_IDLE:
    default:            idle(t, a, o); break;
    }
}

float bunny_clip_duration(bunny_clip_t clip)
{
    switch (clip) {
    case CLIP_LIE_DOWN:
    case CLIP_SLEEP:   return 4.0f;
    case CLIP_SAD:
    case CLIP_YAWN:
    case CLIP_STRETCH: return 3.0f;
    case CLIP_DANCE:   return 4.0f;
    default:           return 2.4f;
    }
}

const char *bunny_clip_name(bunny_clip_t clip)
{
    return (clip >= 0 && clip < CLIP_COUNT) ? names[clip] : "?";
}

void bunny_dial_range(int i, float *lo, float *hi)
{
    if (i < 11) {
        *lo = 0.0f;
        *hi = 1.0f;
    } else {
        *lo = scalars[i - 11][1];
        *hi = scalars[i - 11][2];
    }
}

void bunny_normalize(const float *raw, float *nn_in)
{
    for (int i = 0; i < BUNNY_NDIAL; ++i) {
        float v = (raw[i] - norm_lo[i]) / (norm_hi[i] - norm_lo[i]);
        nn_in[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }
}
