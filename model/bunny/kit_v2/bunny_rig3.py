"""Purple bunny rig v3. Every look is a continuous dial; any in-between value renders cleanly.
The 'dials' are exactly what the NPU decoder will take as input.
CPU-side (not NN dials): screen position, hop height, and effects (zzz, sweat, hearts)."""
import io, math, colorsys
import numpy as np
import cairosvg
from PIL import Image

N = 20            # points per side for eye morph
M = 24            # points per side for mouth morph

EYES = ["open", "wide", "happy", "sleep", "sad"]
MOUTHS = ["smile", "open", "o", "flat", "frown", "yawn"]

# (name, default, low, high) for scalar dials, in NN-input order
SCALARS = [
    ("ear_l", -12, -75, 15), ("ear_r", 12, -15, 75), ("curl", 0, -10, 90), ("ear_len", 128, 108, 148),
    ("look_x", 0, -7, 7), ("look_y", 0, -6, 6),
    ("arm_l", 0, -30, 160), ("arm_r", 0, -160, 30), ("foot_l", 0, -20, 6), ("foot_r", 0, -20, 6),
    ("squash", 1.0, 0.82, 1.14), ("tilt", 0, -14, 14), ("lie", 0, 0, 1), ("head_tilt", 0, -16, 16),
    ("blush", 0.85, 0.4, 1.0), ("tear", 0, 0, 1),
    ("hue", 0, -0.07, 0.07), ("sat", 1.0, 0.8, 1.15), ("head_scale", 1.0, 0.92, 1.08),
    ("brow", 0, 0, 1), ("dizzy", 0, 0, 1),
]
SCALAR_NAMES = [s[0] for s in SCALARS]
DIAL_NAMES = [f"eye_{e}" for e in EYES] + [f"mouth_{m}" for m in MOUTHS] + SCALAR_NAMES
NDIAL = len(DIAL_NAMES)


def default_dials():
    d = {n: dv for n, dv, _, _ in SCALARS}
    d.update({f"eye_{e}": 0.0 for e in EYES}); d["eye_open"] = 1.0
    d.update({f"mouth_{m}": 0.0 for m in MOUTHS}); d["mouth_smile"] = 1.0
    return d


def to_vec(d):
    return np.array([d[n] for n in DIAL_NAMES], dtype=np.float32)


def from_vec(v):
    return {n: float(x) for n, x in zip(DIAL_NAMES, v)}


def lerp(a, b, t): return a + (b - a) * t
def smooth(t): t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)


def hexc(r, g, b): return "#%02X%02X%02X" % (int(r * 255), int(g * 255), int(b * 255))


def palette(hue, sat):
    h0, s0, v0 = colorsys.rgb_to_hsv(0xC7 / 255, 0xA6 / 255, 0xF2 / 255)
    h = (h0 + hue) % 1.0
    s = min(1, s0 * sat)
    body = hexc(*colorsys.hsv_to_rgb(h, s, v0))
    belly = hexc(*colorsys.hsv_to_rgb(h, s * 0.40, 0.99))
    out = hexc(*colorsys.hsv_to_rgb(h, 0.62, 0.32))
    return body, belly, out


def ear_svg(bx, by, angle, curl, length, w0, body, out, inner, SW):
    n = 28
    pts = []; x, y = bx, by; step = length / n
    for i in range(n + 1):
        t = i / n
        th = math.radians(angle + curl * t ** 1.6)
        dx, dy = math.sin(th), -math.cos(th)
        pts.append((x, y, dx, dy, t)); x += dx * step; y += dy * step

    def outline(scale, t0, t1):
        L, R = [], []
        for (px, py, dx, dy, t) in pts:
            if t < t0 or t > t1: continue
            u = (t - t0) / (t1 - t0)
            w = w0 * scale * max(0.0, (1 - u ** 2.4)) ** 0.5 * (0.86 + 0.14 * math.sin(math.pi * min(1, u * 1.2)))
            nx, ny = -dy, dx
            L.append((px + nx * w, py + ny * w)); R.append((px - nx * w, py - ny * w))
        poly = L + R[::-1]
        return "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in poly) + "Z"

    return (f'<path d="{outline(1.0, 0.0, 1.0)}" fill="{body}" stroke="{out}" stroke-width="{SW}" stroke-linejoin="round"/>'
            f'<path d="{outline(0.52, 0.10, 0.80)}" fill="{inner}"/>')


# ---------- eye shapes (polygons, same point count -> true morph) ----------
def _ellipse(rx, ry, dy=0.0):
    up = [(-rx * math.cos(math.pi * i / (N - 1)), -ry * math.sin(math.pi * i / (N - 1)) + dy) for i in range(N)]
    lo = [(rx * math.cos(math.pi * i / (N - 1)), ry * math.sin(math.pi * i / (N - 1)) + dy) for i in range(N)]
    return up + lo


def _arc(w, y_end, bulge, t=4.0):
    xs = [-w + 2 * w * i / (N - 1) for i in range(N)]
    yc = lambda x: y_end + bulge * (1 - (x / w) ** 2)
    return [(x, yc(x) - t / 2) for x in xs] + [(x, yc(x) + t / 2) for x in xs[::-1]]


EYE_SHAPES = {"open": _ellipse(16, 21), "wide": _ellipse(20, 26), "happy": _arc(26, 8, -17),
              "sleep": _arc(24, -2, 9), "sad": _ellipse(15, 18, 4)}


def _spiral(cx, cy, turns=2.6, r0=2.0, r1=16.0, n=40):
    pts = []
    for i in range(n + 1):
        u = i / n; th = u * turns * 2 * math.pi; r = r0 + (r1 - r0) * u
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)


def eyes_svg(w, cx, cy, dark, dizzy=0.0, brow=0.0, side="l"):
    tot = sum(w[k] for k in EYES) or 1.0
    pts = []
    for i in range(2 * N):
        x = sum(w[k] * EYE_SHAPES[k][i][0] for k in EYES) / tot
        y = sum(w[k] * EYE_SHAPES[k][i][1] for k in EYES) / tot
        pts.append((cx + x, cy + y))
    stroke = 6.0 * (w["happy"] + w["sleep"]) / tot
    d = "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in pts) + "Z"
    s = [f'<g opacity="{1 - dizzy:.3f}"><path d="{d}" fill="{dark}" stroke="{dark}" stroke-width="{stroke:.2f}" stroke-linejoin="round"/>']
    hl = (w["open"] + w["wide"] + w["sad"]) / tot
    if hl > 0.05:
        sc = (w["wide"] * 1.25 + w["open"] + w["sad"] * 0.9) / max(1e-6, w["wide"] + w["open"] + w["sad"])
        s.append(f'<g opacity="{min(1, hl * hl * 1.6):.3f}"><circle cx="{cx+5*sc:.1f}" cy="{cy-7*sc:.1f}" r="{6.5*sc:.1f}" fill="#fff"/>'
                 f'<circle cx="{cx-5*sc:.1f}" cy="{cy+7*sc:.1f}" r="{3*sc:.1f}" fill="#fff"/></g>')
    s.append('</g>')
    if dizzy > 0.02:
        s.append(f'<g opacity="{dizzy:.3f}"><circle cx="{cx}" cy="{cy}" r="25" fill="#fff" stroke="{dark}" stroke-width="6"/>'
                 f'<path d="{_spiral(cx, cy)}" fill="none" stroke="{dark}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></g>')
    if brow > 0.02:
        sg = -1 if side == "l" else 1
        s.append(f'<path d="M{cx + sg*30} {cy-44} L{cx - sg*26} {cy-22}" fill="none" stroke="{dark}" stroke-width="{11*brow:.2f}" stroke-linecap="round"/>')
    return "".join(s)


# ---------- mouth shapes ----------
def _line(x0, x1, yfun, t=2.0):
    xs = [x0 + (x1 - x0) * i / (M - 1) for i in range(M)]
    return [(x, yfun(i / (M - 1)) - t / 2) for i, x in enumerate(xs)] + [(x, yfun(i / (M - 1)) + t / 2) for i, x in list(enumerate(xs))[::-1]]


def _mouth_ellipse(cx, cy, rx, ry):
    up = [(cx - rx * math.cos(math.pi * i / (M - 1)), cy - ry * math.sin(math.pi * i / (M - 1))) for i in range(M)]
    lo = [(cx + rx * math.cos(math.pi * i / (M - 1)), cy + ry * math.sin(math.pi * i / (M - 1))) for i in range(M)]
    return up + lo


def _open_mouth():
    xs = [228 + 56 * i / (M - 1) for i in range(M)]
    k = lambda x: 1 - ((x - 256) / 28) ** 2
    return [(x, 283 + 6.5 * k(x)) for x in xs] + [(x, 283 + 29.5 * k(x)) for x in xs[::-1]]


def _w_mouth():
    # cat-style 'w' under the nose: two small lobes
    def y(u):
        # u in [0,1] across x 228..284; lobes dip at u=.25 and .75, meet at nose at u=.5
        lobe = abs(math.sin(2 * math.pi * u))
        return 286 + 11 * lobe * (1 if True else 0) - (0 if u in (0, 1) else 0)
    return _line(228, 284, lambda u: 285 + 11 * abs(math.sin(2 * math.pi * u)) * (0.55 + 0.45 * math.sin(math.pi * u)))


MOUTH_SHAPES = {
    "smile": _w_mouth(), "open": _open_mouth(), "o": _mouth_ellipse(256, 300, 9, 12),
    "flat": _line(240, 272, lambda u: 296 + 4 * math.sin(math.pi * u)),
    "frown": _line(236, 276, lambda u: 304 - 10 * math.sin(math.pi * u)),
    "yawn": _mouth_ellipse(256, 308, 22, 29),
}
MOUTH_FILL = {"smile": 0, "open": 1, "o": 1, "flat": 0, "frown": 0, "yawn": 1}


def mouth_svg(w, dark, uid):
    tot = sum(w[k] for k in MOUTHS) or 1.0
    pts = []
    for i in range(2 * M):
        x = sum(w[k] * MOUTH_SHAPES[k][i][0] for k in MOUTHS) / tot
        y = sum(w[k] * MOUTH_SHAPES[k][i][1] for k in MOUTHS) / tot
        pts.append((x, y))
    f = sum(w[k] * MOUTH_FILL[k] for k in MOUTHS) / tot
    # fill mixes from outline colour (lines) to dark wine (open mouths)
    def mix(c1, c2, t):
        a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]; b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
        return "#%02X%02X%02X" % tuple(int(lerp(a[i], b[i], t)) for i in range(3))
    fill = mix(dark, "#7A2D55", f)
    d = "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in pts) + "Z"
    tg = w["open"] + w["yawn"]
    s = [f'<clipPath id="mc{uid}"><path d="{d}"/></clipPath>',
         f'<path d="{d}" fill="{fill}" stroke="{dark}" stroke-width="7" stroke-linejoin="round"/>']
    if tg > 0.03:
        ty = 318 + 8 * (w["yawn"] / max(1e-6, tg))
        s.append(f'<ellipse cx="256" cy="{ty:.1f}" rx="{16+6*w["yawn"]:.1f}" ry="11" fill="#F0648F" opacity="{min(1, tg*1.4):.3f}" clip-path="url(#mc{uid})"/>')
    return "".join(s)


def arm_svg(side, angle, body, out, SW):
    px, py = (176, 346) if side == "l" else (336, 346)
    return (f'<g transform="translate({px} {py}) rotate({angle:.2f})"><ellipse cx="0" cy="27" rx="27" ry="42" '
            f'fill="{body}" stroke="{out}" stroke-width="{SW}"/></g>')


def bunny_svg(d, fx=None, dx=0.0, dy=0.0, uid=0):
    """d: dial dict. fx: optional CPU-side effects (zzz, sweat, hearts) for previews only."""
    SW = 11
    body, belly, out = palette(d["hue"], d["sat"])
    inner = "#F7A8CB"
    K = 0.84
    sy = d["squash"]; sx = 1.0 / (sy ** 0.5)
    lie = d["lie"]
    rot = d["tilt"] + 90 * lie
    gx = 256 + dx - 170 * lie
    gy = 490 + dy * K - 135 * lie
    g = []
    shw = lerp(118 * sx, 195, lie)
    g.append(f'<ellipse cx="{256+dx:.1f}" cy="496" rx="{shw:.1f}" ry="14" fill="#000" opacity="{0.10*(1-0.01*max(0,-dy)):.3f}"/>')
    g.append(f'<g transform="translate({gx:.2f} {gy:.2f}) scale({sx*K:.4f} {sy*K:.4f}) rotate({rot:.2f}) translate(-256 -490)">')
    # feet, body
    g.append(f'<ellipse cx="204" cy="{474+d["foot_l"]:.1f}" rx="50" ry="26" fill="{body}" stroke="{out}" stroke-width="{SW}"/>')
    g.append(f'<ellipse cx="308" cy="{474+d["foot_r"]:.1f}" rx="50" ry="26" fill="{body}" stroke="{out}" stroke-width="{SW}"/>')
    g.append(f'<path d="M168 340 C150 400 160 470 256 474 C352 470 362 400 344 340 Z" fill="{body}" stroke="{out}" stroke-width="{SW}" stroke-linejoin="round"/>')
    g.append(f'<ellipse cx="256" cy="415" rx="52" ry="48" fill="{belly}"/>')
    # arms behind head unless raised
    front_l = d["arm_l"] > 60; front_r = d["arm_r"] < -60
    if not front_l: g.append(arm_svg("l", d["arm_l"], body, out, SW))
    if not front_r: g.append(arm_svg("r", d["arm_r"], body, out, SW))
    # head group (ears + head + face)
    hr = d["head_tilt"] + 0.5 * d["tilt"]
    g.append(f'<g transform="translate(256 366) rotate({hr:.2f}) scale({d["head_scale"]:.4f}) translate(-256 -366)">')
    g.append(ear_svg(196, 160, d["ear_l"], -d["curl"], d["ear_len"], 42, body, out, inner, SW))
    g.append(ear_svg(316, 160, d["ear_r"], d["curl"], d["ear_len"], 42, body, out, inner, SW))
    g.append(f'<path d="M256 112 C142 112 108 190 112 252 C116 330 170 366 256 366 C342 366 396 330 400 252 C404 190 370 112 256 112 Z" fill="{body}" stroke="{out}" stroke-width="{SW}" stroke-linejoin="round"/>')
    g.append(f'<ellipse cx="160" cy="282" rx="30" ry="21" fill="{inner}" opacity="{d["blush"]:.3f}"/>')
    g.append(f'<ellipse cx="352" cy="282" rx="30" ry="21" fill="{inner}" opacity="{d["blush"]:.3f}"/>')
    ew = {e: d[f"eye_{e}"] for e in EYES}
    lx, ly = d["look_x"], d["look_y"]
    g.append(eyes_svg(ew, 200 + lx, 238 + ly, out, d["dizzy"], d["brow"], "l")); g.append(eyes_svg(ew, 312 + lx, 238 + ly, out, d["dizzy"], d["brow"], "r"))
    if d["tear"] > 0.02:
        g.append(f'<path opacity="{d["tear"]:.3f}" d="M186 262 Q196 282 186 292 Q176 282 186 262Z" fill="#9FD8FF" stroke="{out}" stroke-width="4"/>')
    g.append(f'<ellipse cx="{256+lx*0.6:.1f}" cy="262" rx="13" ry="9" fill="{out}"/>')
    g.append(mouth_svg({m: d[f"mouth_{m}"] for m in MOUTHS}, out, uid))
    g.append('</g>')
    if front_l: g.append(arm_svg("l", d["arm_l"], body, out, SW))
    if front_r: g.append(arm_svg("r", d["arm_r"], body, out, SW))
    g.append('</g>')
    if fx:
        if fx.get("zzz", 0) > 0.01:
            g.append(f'<g transform="translate({-60*lie:.1f} {120*lie:.1f})" opacity="{fx["zzz"]:.3f}" fill="{out}" font-family="DejaVu Sans, sans-serif" font-weight="bold">'
                     f'<text x="378" y="120" font-size="46">Z</text><text x="420" y="78" font-size="34">z</text><text x="448" y="46" font-size="24">z</text></g>')
        if fx.get("sweat", 0) > 0.01:
            g.append(f'<path opacity="{fx["sweat"]:.3f}" d="M392 170 Q408 200 392 214 Q376 200 392 170Z" fill="#9FD8FF" stroke="{out}" stroke-width="6"/>')
        if fx.get("hearts", 0) > 0.01:
            hp = lambda x, y, s: (f'<path transform="translate({x} {y}) scale({s})" d="M0 8 C-18 -10 -8 -24 0 -12 C8 -24 18 -10 0 8Z" fill="#F0648F" opacity="{fx["hearts"]:.3f}"/>')
            g.append(hp(400, 120, 1.6) + hp(120, 90, 1.2) + hp(430, 190, 1.0))
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">'
            f'<rect width="512" height="512" fill="#FFFFFF"/>{"".join(g)}</svg>')


def render(d, px, ss=2, fx=None, dx=0.0, dy=0.0):
    png = cairosvg.svg2png(bytestring=bunny_svg(d, fx, dx, dy).encode(), output_width=px * ss, output_height=px * ss)
    return Image.open(io.BytesIO(png)).convert("RGB").resize((px, px), Image.LANCZOS)
