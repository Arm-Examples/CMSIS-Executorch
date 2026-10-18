"""Generate the workload-on-the-board figure (light + dark animated SVG).

The board render (alif_e8_board_top.png, from Alif's DevKit-E8 layout image with the labels and background removed)
is embedded as base64 so the SVG works inside a GitHub <img>. Glass callouts as in make_pipeline_svg.py.
Run: python docs/media/make_board_svg.py docs/media

GitHub serves SVGs with CSP default-src 'none', which blocks the embedded board image, so the README uses
board.png / board-dark.png: animated PNGs (16 frames, 1.6 s loop, transparent) made from these SVGs:
  tools/render_board_frames.sh board.svg <dir>/light   # frames on black and white, paused at t = 0 .. 1.5 s
  python tools/apng_delta.py <dir>/light board.png     # alpha from the black/white pair; frames store only changes
"""
import base64
import sys
from pathlib import Path

from make_pipeline_svg import ICONS, THEMES, esc

W, H = 1500, 640
IMG = Path(__file__).with_name("alif_e8_board_top.png")
IMG_W, IMG_H = 1100, 708          # pixel size of the jpg
BX, BY, BW = 330, 44, 860         # where the board is drawn
S = BW / IMG_W


def at(px, py):
    """Board-image pixel to SVG coordinates."""
    return BX + px * S, BY + py * S


SOC, IMU, LED = at(646, 288), at(730, 566), at(838, 579)
SOC_U85, SOC_M55, SOC_MEM = at(668, 266), at(626, 290), at(670, 312)

HUES = {"host": "124,92,255", "m55": "244,63,94", "u85": "245,158,11", "mem": "6,182,212",
        "imu": "34,197,94", "led": "59,130,246"}

# (x, y, w, h, hue, icon, title, tag, lines, target)
CX, CW_, CH_ = 1230, 254, 98
CARDS = [
    (CX, 30, CW_, CH_, "u85", "npu", "ETHOS-U85", "NPU", ["TinyEEGNet-wide, int8", "0.32 ms per window"], SOC_U85),
    (CX, 142, CW_, CH_, "m55", "chip", "CORTEX-M55", "CPU", ["UART, preprocessing,", "motion gate, JSON out"], SOC_M55),
    (CX, 254, CW_, CH_, "mem", "chip", "MRAM + SRAM", "MEM", ["model 11 KB + tables;", "NPU pools 124.5 KiB"], SOC_MEM),
    (CX, 366, CW_, CH_, "imu", "imu", "IMU", "SENSOR", ["ICM-42670-P, 100 Hz", "drives the motion gate"], IMU),
    (CX, 478, CW_, CH_, "led", "led", "RGB LEDs", "OUT", ["green left · blue right", "red = motion gated"], LED),
]
HOST = (20, 210, 280, 150, "host", "laptop", "LAPTOP", "HOST",
        ["bridge.py: dataset replay,", "250 Hz over USB serial", "demo_ui.py: live display"])


def glass(t, hue, x, y, w, h, rx):
    c = HUES[hue]
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="rgba({c},{t["fill_a"] + 0.06})" '
            f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1.6"/>'
            f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{w - 3}" height="{h * 0.45}" rx="{rx - 1.5}" fill="url(#sheen)"/>')


def card(t, x, y, w, h, hue, icon, title, tag, lines):
    c = HUES[hue]
    tw = 12 + 8.4 * len(tag)
    p = [glass(t, hue, x, y, w, h, 14),
         f'<g transform="translate({x + 12},{y + 12}) scale(1.15)" fill="none" stroke="{t["stroke"]}" stroke-width="1.8" '
         f'stroke-linecap="round" stroke-linejoin="round">{ICONS[icon].replace("{stroke}", t["stroke"])}</g>',
         f'<text x="{x + 50}" y="{y + 33}" font-size="17" font-weight="700" letter-spacing="0.4" fill="{t["title"]}">{esc(title)}</text>',
         f'<rect x="{x + w - tw - 10}" y="{y + 15}" width="{tw}" height="24" rx="12" fill="rgba({c},{t["chip_a"]})" '
         f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1"/>',
         f'<text x="{x + w - 10 - tw / 2}" y="{y + 31.5}" font-size="12" font-weight="700" text-anchor="middle" '
         f'fill="{t["title"]}">{esc(tag)}</text>']
    for i, line in enumerate(lines):
        p.append(f'<text x="{x + 14}" y="{y + 62 + 20 * i}" font-size="14.5" fill="{t["tools"]}">{esc(line)}</text>')
    return "".join(p)


def leader(t, hue, x1, y1, x2, y2):
    c = HUES[hue]
    mx = x1 - 40
    return (f'<path d="M{x1},{y1} H{mx} L{x2},{y2}" fill="none" stroke="rgba({c},0.95)" stroke-width="2.2" '
            f'stroke-dasharray="6 5"><animate attributeName="stroke-dashoffset" from="22" to="0" dur="0.8s" '
            f'repeatCount="indefinite"/></path>'
            f'<circle cx="{x2}" cy="{y2}" r="7" fill="rgba({c},1)" stroke="#fff" stroke-width="2"/>'
            f'<circle cx="{x2}" cy="{y2}" r="7" fill="none" stroke="rgba({c},0.9)" stroke-width="2">'
            f'<animate attributeName="r" values="7;20" dur="1.6s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0.9;0" dur="1.6s" repeatCount="indefinite"/></circle>')


def build(theme):
    t = THEMES[theme]
    img = base64.b64encode(IMG.read_bytes()).decode()
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="Inter, -apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">',
         '<defs><linearGradient id="sheen" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0" stop-color="#ffffff" stop-opacity="{t["sheen_a"]}"/>'
         '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient>'
         f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="rgba({HUES["host"]},1)"/></marker></defs>',
         f'<image x="{BX}" y="{BY}" width="{BW}" height="{IMG_H * S}" href="data:image/png;base64,{img}" '
         f'xlink:href="data:image/png;base64,{img}"/>',
         f'<text x="{BX + BW / 2}" y="{BY + IMG_H * S + 34}" font-size="15" text-anchor="middle" fill="{t["role"]}">'
         'Alif Ensemble E8 DevKit · the M55, U85, MRAM and SRAM are all inside the ALIF E8 chip</text>']

    for x, y, w, h, hue, icon, title, tag, lines, target in CARDS:
        p.append(leader(t, hue, x, y + h / 2, *target))
    for x, y, w, h, hue, icon, title, tag, lines, target in CARDS:
        p.append(card(t, x, y, w, h, hue, icon, title, tag, lines))

    hx, hy, hw, hh = HOST[:4]
    p.append(card(t, *HOST))
    c = HUES["host"]
    ey = hy + hh / 2
    p.append(f'<path d="M{hx + hw + 4},{ey} H{BX + 6}" fill="none" stroke="rgba({c},1)" stroke-width="2.6" '
             f'stroke-dasharray="7 5" marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="24" to="0" '
             f'dur="0.8s" repeatCount="indefinite"/></path>')
    p.append(f'<text x="{(hx + hw + BX) / 2}" y="{ey - 12}" font-size="13" text-anchor="middle" fill="{t["label"]}">USB</text>')
    p.append(f'<text x="{(hx + hw + BX) / 2}" y="{ey + 24}" font-size="13" text-anchor="middle" fill="{t["label"]}">UART4</text>')
    p.append("</svg>")
    return "\n".join(p) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "board.svg").write_text(build("light"))
    (out / "board-dark.svg").write_text(build("dark"))
    print("wrote", out / "board.svg", out / "board-dark.svg")
