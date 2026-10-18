"""Generate Figure 2: the preprocessing chain inside the PREPROCESSING card of pipeline.svg.

Same glass style as make_pipeline_svg.py. Numbers come from firmware/model/params.json (v0.5-distill).
Run: python docs/media/make_signal_svg.py docs/media
"""
import sys
from pathlib import Path

from make_pipeline_svg import ICONS, THEMES, esc

W, H = 1120, 650
CW, CH = 235, 160
COLS = [24, 303, 582, 861]
ROWS = [30, 240, 450]

HUES = {"in": "6,182,212", "sample": "244,63,94", "window": "234,120,40", "npu": "245,158,11"}

# (col, row, hue, step, title, formula, note lines); step "wave"/"npu" = icon card
CARDS = [
    (0, 0, "in", "wave", "UART IN", "22 × int16 counts", ["one frame per sample,", "250 Hz, from bridge.py"]),
    (1, 0, "sample", "1", "DECODE", "µV = count × 0.0488", ["exact BCI IV 2a ADC step,", "lossless"]),
    (2, 0, "sample", "2", "CAR", "x − mean of 22 ch", ["common average reference:", "removes shared reference"]),
    (3, 0, "sample", "3", "BAND-PASS", "7–30 Hz · 4 biquads", ["causal, state kept;", "keeps mu and beta rhythms"]),
    (0, 1, "sample", "4", "DECIMATE", "keep every 2nd sample", ["250 Hz → 125 Hz; the", "band-pass is the anti-alias"]),
    (1, 1, "window", "5", "WINDOW", "22 × 256 samples", ["2.048 s at 125 Hz,", "cue + 0.5 s to + 2.5 s"]),
    (2, 1, "window", "6", "ALIGN (EA)", "X ← A_subject · X", ["22 × 22 matrix per person,", "from 24 unlabeled trials"]),
    (3, 1, "window", "7", "SCALE", "X ← X / 0.969", ["unit variance, same as", "the training data"]),
    (0, 2, "window", "8", "QUANTIZE", "round(X/0.0696) − 12", ["int8, clipped −128…127,", "exactly as on the laptop"]),
    (1, 2, "npu", "npu", "ETHOS-U85", "TinyEEGNet-wide", ["int8 22 × 256 → 2 logits:", "left / right"]),
]

# legend card in the free slots of the last row: hue, heading, detail
LEGEND = [
    ("sample", "PER SAMPLE · steps 1–4", "every 4 ms on the M55 · ≈ 250 cycles = 0.6 µs"),
    ("window", "PER WINDOW · steps 5–8", "once per trial on the M55 · ≈ 100 k cycles = 0.25 ms"),
]


def glass(t, hue, x, y, w, h, rx, fill=None):
    c = HUES[hue]
    fa = t["fill_a"] if fill is None else fill
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="rgba({c},{fa})" '
            f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1.5"/>'
            f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{w - 3}" height="{h * 0.45}" rx="{rx - 1.5}" fill="url(#sheen)"/>')


def flow(t, d):
    return (f'<path d="{d}" fill="none" stroke="{t["arrow"]}" stroke-width="2.2" stroke-dasharray="5 5" '
            f'marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="10" to="0" dur="0.6s" '
            f'repeatCount="indefinite"/></path>')


def card(t, col, row, hue, step, title, formula, notes):
    x, y = COLS[col], ROWS[row]
    c = HUES[hue]
    p = [glass(t, hue, x, y, CW, CH, 16)]
    if step.isdigit():
        p.append(f'<circle cx="{x + 32}" cy="{y + 32}" r="16" fill="rgba({c},{t["chip_a"]})" '
                 f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1.2"/>'
                 f'<text x="{x + 32}" y="{y + 38}" font-size="16" font-weight="800" text-anchor="middle" '
                 f'fill="{t["title"]}">{step}</text>')
    else:
        p.append(f'<g transform="translate({x + 16},{y + 16}) scale(1.3)" fill="none" stroke="{t["stroke"]}" '
                 f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
                 f'{ICONS[{"wave": "wave", "npu": "npu"}[step]].replace("{stroke}", t["stroke"])}</g>')
    p.append(f'<text x="{x + 60}" y="{y + 39}" font-size="18" font-weight="700" letter-spacing="0.4" '
             f'fill="{t["title"]}">{esc(title)}</text>')
    p.append(f'<rect x="{x + 14}" y="{y + 60}" width="{CW - 28}" height="34" rx="9" '
             f'fill="rgba({c},{t["chip_a"] * 0.6:.2f})"/>')
    p.append(f'<text x="{x + CW / 2}" y="{y + 82}" font-size="15" font-weight="600" text-anchor="middle" '
             f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" '
             f'fill="{t["tools"]}">{esc(formula)}</text>')
    for i, line in enumerate(notes):
        p.append(f'<text x="{x + 16}" y="{y + 120 + 21 * i}" font-size="14.5" fill="{t["role"]}">{esc(line)}</text>')
    return "".join(p)


def build(theme):
    t = THEMES[theme]
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         "font-family=\"Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif\">",
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t["arrow"]}"/></marker>'
         '<linearGradient id="sheen" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0" stop-color="#ffffff" stop-opacity="{t["sheen_a"]}"/>'
         '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient></defs>']

    for c in CARDS:
        p.append(card(t, *c))

    # legend in the last row, columns 2-3
    lx, ly = COLS[2], ROWS[2]
    lw = COLS[3] + CW - lx
    p.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{CH}" rx="16" fill="none" '
             f'stroke="{t["arrow"]}" stroke-width="1.2" stroke-dasharray="6 5"/>')
    for i, (hue, head, detail) in enumerate(LEGEND):
        yy = ly + 32 + 62 * i
        c = HUES[hue]
        p.append(f'<rect x="{lx + 20}" y="{yy - 4}" width="22" height="22" rx="6" fill="rgba({c},{t["chip_a"]})" '
                 f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1.2"/>')
        p.append(f'<text x="{lx + 54}" y="{yy + 13}" font-size="16" font-weight="700" letter-spacing="0.4" '
                 f'fill="{t["title"]}">{esc(head)}</text>')
        p.append(f'<text x="{lx + 54}" y="{yy + 36}" font-size="14.5" fill="{t["role"]}">{esc(detail)}</text>')

    for r in range(3):
        mid = ROWS[r] + CH / 2
        last = 3 if r < 2 else 1
        for col in range(last):
            p.append(flow(t, f"M{COLS[col] + CW + 4},{mid} H{COLS[col + 1] - 4}"))
    # end of a row back to the start of the next one, through the gap between rows
    for r in range(2):
        gy = ROWS[r] + CH + (ROWS[r + 1] - ROWS[r] - CH) / 2
        p.append(flow(t, f"M{COLS[3] + CW / 2},{ROWS[r] + CH + 4} V{gy} H{COLS[0] + CW / 2} V{ROWS[r + 1] - 4}"))

    p.append("</svg>")
    return "\n".join(p) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "signal.svg").write_text(build("light"))
    (out / "signal-dark.svg").write_text(build("dark"))
    print("wrote", out / "signal.svg", out / "signal-dark.svg")
