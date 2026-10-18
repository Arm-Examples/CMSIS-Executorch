"""Motion clips: each is a path through the dials over time. (This is also the runtime motion library.)
clip(name, t, amp) -> (dials, fx, dx, dy). dx/dy/fx are CPU-side (screen motion + effects)."""
import math
from bunny_rig3 import default_dials, lerp, smooth, EYES, MOUTHS

TAU = 2 * math.pi


def P(eye="open", mouth="smile", **kw):
    d = default_dials()
    for e in EYES: d[f"eye_{e}"] = 1.0 if e == eye else 0.0
    for m in MOUTHS: d[f"mouth_{m}"] = 1.0 if m == mouth else 0.0
    d.update(kw)
    return d


def mixd(a, b, u):
    return {k: lerp(a[k], b[k], u) for k in a}


def idle(t, a=1.0):
    d = P(eye="open", mouth="smile", squash=1 + 0.012 * math.sin(TAU * t / 2.4) * a)
    ph = t % 3.1
    if ph < 0.16:
        k = math.sin(math.pi * ph / 0.16); d["eye_open"] = 1 - k; d["eye_sleep"] = k
    return d, {}, 0, 0


def blink_look(t, a=1.0):
    d, fx, dx, dy = idle(t, a)
    d["look_x"] = 6 * a * math.sin(TAU * t / 3.5); d["look_y"] = 2 * a * math.sin(TAU * t / 2.1)
    d["head_tilt"] = 7 * a * math.sin(TAU * t / 3.5 + 0.4)
    return d, fx, dx, dy


def hop(t, a=1.0):
    ph = (t % 0.8) / 0.8
    h = math.sin(math.pi * ph)
    land = max(0, -math.sin(TAU * ph + 0.0)) if ph > 0.5 else 0
    d = P(eye="happy", mouth="open", ear_l=-10 - 12 * h * a, ear_r=10 + 12 * h * a, curl=30 * h * a,
          squash=1 + 0.14 * h * a - 0.10 * land * a, arm_l=30 * h * a, arm_r=-30 * h * a,
          foot_l=-14 * h * a, foot_r=-14 * h * a)
    return d, {}, 0, -46 * h * a


def walk(t, a=1.0):
    ph = TAU * t / 0.8
    d = P(eye="open", mouth="smile", ear_l=-18, ear_r=18, curl=14, squash=1 + 0.02 * math.cos(2 * ph) * a, tilt=2 * math.sin(ph) * a,
          foot_l=-9 * max(0, math.sin(ph)) * a, foot_r=-9 * max(0, -math.sin(ph)) * a,
          arm_l=22 * math.sin(ph) * a, arm_r=-22 * math.sin(ph + math.pi) * a - 0)
    d["arm_r"] = -22 * math.sin(ph) * a * -1  # opposite swing
    return d, {}, 0, -6 * abs(math.sin(ph)) * a


def run(t, a=1.0):
    ph = TAU * t / 0.42
    d = P(eye="open", mouth="open", ear_l=-40, ear_r=40, curl=30, squash=0.95 + 0.04 * math.cos(2 * ph) * a, tilt=6 + 2.5 * math.sin(ph) * a,
          foot_l=-16 * max(0, math.sin(ph)) * a, foot_r=-16 * max(0, -math.sin(ph)) * a,
          arm_l=-45 + 35 * math.sin(ph) * a + 45 + 0, arm_r=40 - 35 * math.sin(ph) * a - 40)
    d["arm_l"] = 35 * math.sin(ph) * a + 20
    d["arm_r"] = -(35 * math.sin(ph + math.pi) * a + 20)
    return d, {}, 0, -14 * abs(math.sin(ph)) * a


def wave(t, a=1.0):
    ph = TAU * 3.2 * t
    d = P(eye="happy", mouth="open", ear_l=-14, ear_r=22 + 6 * math.sin(ph) * a, curl=10, head_tilt=-6 * a,
          arm_r=-(116 + 17 * math.sin(ph)), arm_l=6)
    return d, {}, 0, 0


def shake(t, a=1.0):
    d = P(eye="wide", mouth="o", ear_l=-4 + 6 * math.sin(TAU * 13 * t) * a, ear_r=4 - 6 * math.sin(TAU * 13 * t) * a, squash=1.08,
          arm_l=38, arm_r=-38, tilt=3 * math.sin(TAU * 11 * t + 1) * a, head_tilt=4 * math.sin(TAU * 9 * t) * a)
    return d, {"sweat": 1.0}, 9 * math.sin(TAU * 11 * t) * a, 0


def lie_down(t, a=1.0):
    """0-1.4s: lie down (eyes droop, ears fall); then sleeping breaths."""
    u = smooth(t / 1.4)
    ph = TAU * t / 3.2
    d = P(eye="open", mouth="smile")
    d.update(eye_open=1 - u, eye_sleep=u, mouth_smile=1 - u, mouth_flat=u, lie=u, ear_l=lerp(-12, -62, u), ear_r=lerp(12, 62, u),
             curl=lerp(0, 80, u), squash=lerp(1, 0.9, u) + 0.014 * u * math.sin(ph), blush=lerp(0.85, 0.7, u), tilt=0)
    return d, {"zzz": smooth((t - 1.0) / 0.6)}, 0, 0


def sleep_lying(t, a=1.0):
    return lie_down(1.4 + t, a)


def sit(t, a=1.0):
    d = P(eye="open", mouth="smile", squash=0.9 + 0.012 * math.sin(TAU * t / 2.6), foot_l=-10, foot_r=-10, arm_l=-8, arm_r=8, ear_l=-16, ear_r=16)
    ph = t % 3.4
    if ph < 0.16:
        k = math.sin(math.pi * ph / 0.16); d["eye_open"] = 1 - k; d["eye_sleep"] = k
    return d, {}, 0, 0


def yawn(t, a=1.0):
    u = math.sin(math.pi * min(1, (t % 3.0) / 2.4)) ** 1.5
    d = P(eye="open", mouth="smile")
    d.update(eye_open=1 - u, eye_sleep=u, mouth_smile=1 - u, mouth_yawn=u, arm_l=lerp(0, 100, u), arm_r=-lerp(0, 100, u),
             ear_l=lerp(-12, -40, u), ear_r=lerp(12, 40, u), curl=lerp(0, 50, u), squash=lerp(1, 1.08, u), head_tilt=-4 * u)
    return d, {}, 0, 0


def sad(t, a=1.0):
    u = smooth(t / 0.8)
    d = P(eye="sad", mouth="frown")
    d.update(ear_l=lerp(-12, -50, u), ear_r=lerp(12, 50, u), curl=lerp(0, 72, u), tear=smooth((t - 0.5) / 0.6), head_tilt=-8 * u,
             squash=lerp(1, 0.95, u) + 0.008 * math.sin(TAU * t / 1.6), blush=0.9, look_y=3 * u)
    return d, {}, 0, 0


def eat(t, a=1.0):
    ph = TAU * 2.6 * t
    c = 0.5 + 0.5 * math.sin(ph)
    d = P(eye="happy", mouth="open")
    d.update(mouth_open=0.25 + 0.6 * c, mouth_smile=0.75 - 0.6 * c, arm_l=-22 - 6 * math.sin(ph), arm_r=22 + 6 * math.sin(ph),
             squash=1 + 0.012 * math.sin(2 * ph), head_tilt=3 * math.sin(ph), blush=1.0)
    return d, {}, 0, 0


def love(t, a=1.0):
    ph = TAU * t / 1.2
    d = P(eye="happy", mouth="smile", blush=1.0, ear_l=-8, ear_r=8, squash=1 + 0.05 * abs(math.sin(ph)), arm_l=10, arm_r=-10)
    return d, {"hearts": 0.5 + 0.5 * math.sin(ph)}, 0, -8 * abs(math.sin(ph))


def stretch(t, a=1.0):
    u = math.sin(math.pi * min(1, (t % 3.0) / 2.2))
    d = P(eye="open", mouth="smile")
    d.update(eye_open=1 - 0.8 * u, eye_happy=0.8 * u, mouth_smile=1 - u, mouth_yawn=0.7 * u, mouth_open=0.3 * u,
             arm_l=lerp(0, 155, u), arm_r=-lerp(0, 155, u), squash=lerp(1, 1.12, u), foot_l=-4 * u, foot_r=-4 * u, ear_l=lerp(-12, -24, u), ear_r=lerp(12, 24, u))
    return d, {}, 0, 0


def clap(t, a=1.0):
    ph = TAU * 3 * t
    s = 0.5 + 0.5 * math.sin(ph)
    d = P(eye="happy", mouth="open", arm_l=-10 - 30 * s, arm_r=10 + 30 * s, squash=1 + 0.02 * s, ear_l=-14, ear_r=14)
    return d, {}, 0, 0


CLIPS = {"idle": idle, "look": blink_look, "hop": hop, "walk": walk, "run": run, "wave": wave, "shake": shake,
         "lie_down": lie_down, "sleep": sleep_lying, "sit": sit, "yawn": yawn, "sad": sad, "eat": eat, "love": love,
         "stretch": stretch, "clap": clap}
DURATION = {"lie_down": 4.0, "sleep": 4.0, "sad": 3.0, "yawn": 3.0, "stretch": 3.0}   # default 2.4s


def clip(name, t, amp=1.0):
    return CLIPS[name](t, amp)
