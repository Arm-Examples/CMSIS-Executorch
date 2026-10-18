#ifndef BUNNY_CLIPS_H_
#define BUNNY_CLIPS_H_

/* Bunny motion clips, ported from model/bunny/kit_v2/clips.py (same formulas).
 * A clip maps time to the 30 pose dials the BunnyDecoder draws, plus CPU-only
 * values: screen offset (dx, dy) and effects (zzz, sweat, hearts). */

#ifdef __cplusplus
extern "C" {
#endif

#define BUNNY_NDIAL 32  /* v2 decoder: + brow, dizzy */

/* Dial order = DIAL_NAMES in model/bunny/bunny_rig3.py (the NN input order) */
enum {
    D_EYE_OPEN, D_EYE_WIDE, D_EYE_HAPPY, D_EYE_SLEEP, D_EYE_SAD,
    D_MOUTH_SMILE, D_MOUTH_OPEN, D_MOUTH_O, D_MOUTH_FLAT, D_MOUTH_FROWN, D_MOUTH_YAWN,
    D_EAR_L, D_EAR_R, D_CURL, D_EAR_LEN, D_LOOK_X, D_LOOK_Y, D_ARM_L, D_ARM_R,
    D_FOOT_L, D_FOOT_R, D_SQUASH, D_TILT, D_LIE, D_HEAD_TILT, D_BLUSH, D_TEAR,
    D_HUE, D_SAT, D_HEAD_SCALE, D_BROW, D_DIZZY,
};

typedef enum {
    CLIP_IDLE, CLIP_LOOK, CLIP_HOP, CLIP_WALK, CLIP_RUN, CLIP_WAVE, CLIP_SHAKE,
    CLIP_LIE_DOWN, CLIP_SLEEP, CLIP_SIT, CLIP_YAWN, CLIP_SAD, CLIP_EAT, CLIP_LOVE,
    CLIP_STRETCH, CLIP_CLAP, CLIP_ANGRY, CLIP_DIZZY, CLIP_CONFUSED, CLIP_DANCE, CLIP_COUNT
} bunny_clip_t;

typedef struct {
    float d[BUNNY_NDIAL];     /* raw dial values (degrees, scales, weights) */
    float dx, dy;             /* screen offset in rig pixels (512-px canvas) */
    float zzz, sweat, hearts; /* effect strengths 0..1 */
} bunny_pose_t;

/* Evaluates clip at time t (seconds since it started), amplitude amp (1 = normal). */
void bunny_clip(bunny_clip_t clip, float t, float amp, bunny_pose_t *out);

/* Length of one play of a clip in seconds (clips.py DURATION, default 2.4). */
float bunny_clip_duration(bunny_clip_t clip);

const char *bunny_clip_name(bunny_clip_t clip);

/* Raw dials -> NN input: eye/mouth weights as they are, each scalar mapped to
 * [0,1] with its (lo, hi) range from SCALARS, clamped. */
void bunny_normalize(const float *raw, float *nn_in);

/* (lo, hi) range of dial i (eye/mouth weights: 0..1) */
void bunny_dial_range(int i, float *lo, float *hi);

#ifdef __cplusplus
}
#endif

#endif /* BUNNY_CLIPS_H_ */
