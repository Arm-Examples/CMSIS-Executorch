"""Generate the overview figure (light + dark animated SVG): EEG cap / public dataset -> EEG signals -> Alif E8 -> PC.

The cap photo (eeg_cap.png, background removed) and the board render (alif_e8_board_top.png) are embedded as
base64, so GitHub blocks them inside an SVG <img>. The README uses overview.png / overview-dark.png: animated PNGs
(32 frames, 3.2 s loop, transparent) made from these SVGs:
  python tools/render_frames.py overview.svg <dir>/light 32    # frames on black and white, paused at t = 0 .. 3.1 s
  python tools/apng_delta.py <dir>/light overview.png          # alpha from the black/white pair; frames store only changes
Run: python docs/media/make_overview_svg.py docs/media
"""
import base64
import io
import math
import sys
from pathlib import Path

from PIL import Image

from make_pipeline_svg import ICONS, THEMES, esc

W, H = 1500, 560
HERE = Path(__file__).parent
HUES = {"cap": "124,92,255", "data": "6,182,212", "eeg": "34,197,94", "board": "245,158,11", "pc": "59,130,246"}
LOOP = 3.2
# one colour per channel, cycling through the figure palette
TRACE = ["244,63,94", "249,115,22", "234,179,8", "34,197,94", "20,184,166", "6,182,212",
         "59,130,246", "99,102,241", "139,92,246", "217,70,239", "236,72,153", "132,204,22"]


def b64(path, width):
    img = Image.open(HERE / path)
    img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), img.height


def glass(t, hue, x, y, w, h, rx=16, dash=False):
    c = HUES[hue]
    d = ' stroke-dasharray="7 6"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="rgba({c},{t["fill_a"]})" '
            f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1.6"{d}/>'
            f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{w - 3}" height="{h * 0.4}" rx="{rx - 1.5}" fill="url(#sheen)"/>')


def head(t, hue, x, y, w, icon, title, tag):
    c = HUES[hue]
    tw = 12 + 8.4 * len(tag)
    return (f'<g transform="translate({x + 14},{y + 14}) scale(1.1)" fill="none" stroke="{t["stroke"]}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[icon].replace("{stroke}", t["stroke"])}</g>'
            f'<text x="{x + 50}" y="{y + 34}" font-size="17" font-weight="700" letter-spacing="0.4" fill="{t["title"]}">{esc(title)}</text>'
            f'<rect x="{x + w - tw - 12}" y="{y + 16}" width="{tw}" height="24" rx="12" fill="rgba({c},{t["chip_a"]})" '
            f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1"/>'
            f'<text x="{x + w - 12 - tw / 2}" y="{y + 32.5}" font-size="12" font-weight="700" text-anchor="middle" '
            f'fill="{t["title"]}">{esc(tag)}</text>')


def lines(t, x, y, rows, size=14.5, gap=20):
    return "".join(f'<text x="{x}" y="{y + gap * i}" font-size="{size}" fill="{t["tools"] if i == 0 else t["role"]}">{esc(r)}</text>'
                   for i, r in enumerate(rows))


def arrow(t, d, label=None, lx=0, ly=0, dash=False, color=None):
    col = color or t["arrow"]
    s = (f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.4" stroke-dasharray="{"4 6" if dash else "7 5"}" '
         f'marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="24" to="0" dur="0.8s" '
         f'repeatCount="indefinite"/></path>')
    if label:
        s += f'<text x="{lx}" y="{ly}" font-size="13" text-anchor="middle" fill="{t["label"]}">{esc(label)}</text>'
    return s


def trace(ch, width, amp):
    """Periodic EEG-like trace: one period = 160 px, drawn over width + 160 so it can scroll by one period."""
    pts = []
    for i in range(0, width + 161, 2):
        u = i / 160 * 2 * math.pi
        v = (0.55 * math.sin(3 * u + ch) + 0.30 * math.sin(7 * u + 2.1 * ch) + 0.22 * math.sin(13 * u + 0.7 * ch)
             + 0.12 * math.sin(23 * u + 1.3 * ch))
        pts.append(f"{i},{v * amp:.1f}")
    return "M" + " L".join(pts)


def build(theme):
    t = THEMES[theme]
    cap, cap_h = b64("eeg_cap.png", 120)
    board, board_h = b64("alif_e8_board_top.png", 300)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="Inter, -apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">',
         '<defs><linearGradient id="sheen" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0" stop-color="#ffffff" stop-opacity="{t["sheen_a"]}"/>'
         '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient>'
         '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{t["arrow"]}"/></marker>'
         '<clipPath id="eegclip"><rect x="0" y="0" width="190" height="270"/></clipPath>'
         '<clipPath id="screen"><rect x="0" y="0" width="300" height="186" rx="6"/></clipPath></defs>']

    # 1a. EEG cap (the target input, not used in the demo)
    cx, cy, cw, ch = 20, 20, 320, 240
    p.append(glass(t, "cap", cx, cy, cw, ch, dash=True))
    p.append(head(t, "cap", cx, cy, cw, "wave", "EEG CAP", "TARGET"))
    p.append(f'<image x="{cx + 24}" y="{cy + 62}" width="120" height="{cap_h}" href="data:image/png;base64,{cap}" '
             f'xlink:href="data:image/png;base64,{cap}"/>')
    p.append(lines(t, cx + 176, cy + 90, ["The real input:", "a wearable cap", "on the patient.", "Not used in this", "demo (no headset)."], 14, 21))

    # replaced-by link between cap and dataset
    p.append(f'<path d="M{cx + cw / 2},{cy + ch + 6} V{cy + ch + 44}" stroke="{t["arrow"]}" stroke-width="2" '
             f'stroke-dasharray="3 5" marker-end="url(#ah)"/>')
    p.append(f'<text x="{cx + cw / 2 + 12}" y="{cy + ch + 31}" font-size="13" font-style="italic" fill="{t["label"]}">replaced by</text>')

    # 1b. Public dataset (what the demo actually streams)
    dx, dy, dw, dh = 20, 310, 320, 230
    p.append(glass(t, "data", dx, dy, dw, dh))
    p.append(head(t, "data", dx, dy, dw, "wave", "PUBLIC DATASET", "USED"))
    p.append(lines(t, dx + 18, dy + 72, ["BCI Competition IV 2a", "9 subjects · 22 EEG channels", "250 Hz, left vs right hand",
                                         "held-out session 2, never trained on,", "replayed sample by sample, as a",
                                         "cap would stream it"], 14.5, 24))

    # 2. EEG signals
    ex, ey, ew, eh = 400, 105, 270, 355
    p.append(glass(t, "eeg", ex, ey, ew, eh))
    p.append(head(t, "eeg", ex, ey, ew, "wave", "EEG SIGNALS", "22 CH"))
    names = ["FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CPz"]
    c = HUES["eeg"]
    p.append(f'<g transform="translate({ex + 62},{ey + 64})"><g clip-path="url(#eegclip)"><g>'
             f'<animateTransform attributeName="transform" type="translate" from="0 0" to="-160 0" dur="{LOOP / 2}s" '
             'repeatCount="indefinite"/>')
    for i, n in enumerate(names):
        p.append(f'<path transform="translate(0,{12 + 21.5 * i})" d="{trace(i, 200, 6.5)}" fill="none" '
                 f'stroke="rgb({TRACE[i % len(TRACE)]})" stroke-width="1.4" stroke-linejoin="round"/>')
    p.append('</g></g></g>')
    for i, n in enumerate(names):
        p.append(f'<text x="{ex + 22}" y="{ey + 80 + 21.5 * i}" font-size="11.5" font-weight="600" fill="{t["role"]}" '
                 f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">{n}</text>')
    p.append(f'<text x="{ex + ew / 2}" y="{ey + eh - 14}" font-size="13" text-anchor="middle" fill="{t["role"]}">'
             '12 of the 22 channels shown</text>')

    # arrows into the EEG block
    p.append(arrow(t, f"M{cx + cw + 4},{cy + ch / 2} C{cx + cw + 40},{cy + ch / 2} {ex - 50},{ey + 70} {ex - 6},{ey + 70}", dash=True))
    p.append(arrow(t, f"M{dx + dw + 4},{dy + dh / 2} C{dx + dw + 40},{dy + dh / 2} {ex - 50},{ey + eh - 70} {ex - 6},{ey + eh - 70}"))

    # 3. Board
    bx, by, bw, bh = 750, 95, 340, 380
    p.append(glass(t, "board", bx, by, bw, bh))
    p.append(head(t, "board", bx, by, bw, "chip", "ALIF E8 BOARD", "EDGE"))
    p.append(f'<image x="{bx + 20}" y="{by + 60}" width="300" height="{board_h}" href="data:image/png;base64,{board}" '
             f'xlink:href="data:image/png;base64,{board}"/>')
    p.append(lines(t, bx + 20, by + 60 + board_h + 30, ["Cortex-M55: filtering, alignment, int8",
                                                       "Ethos-U85 NPU: TinyEEGNet, 0.32 ms",
                                                       "→ LEFT or RIGHT hand"], 14.5, 22))
    p.append(arrow(t, f"M{ex + ew + 4},{ey + eh / 2} H{bx - 6}", "UART", (ex + ew + bx) / 2, ey + eh / 2 - 12))
    p.append(f'<text x="{(ex + ew + bx) / 2}" y="{ey + eh / 2 + 24}" font-size="13" text-anchor="middle" fill="{t["label"]}">250 Hz</text>')

    # 4. PC with moving cursor
    px, py, pw, ph = 1170, 95, 310, 380
    p.append(glass(t, "pc", px, py, pw, ph))
    p.append(head(t, "pc", px, py, pw, "laptop", "PC", "DISPLAY"))
    sx, sy = px + 5, py + 72
    scr = "#0d1220"
    p.append(f'<rect x="{sx - 4}" y="{sy - 4}" width="308" height="194" rx="9" fill="#2a2f3a"/>'
             f'<path d="M{sx + 130},{sy + 190} h40 l10,26 h-60 z" fill="#3a404c"/>'
             f'<rect x="{sx + 100}" y="{sy + 214}" width="100" height="7" rx="3" fill="#3a404c"/>')
    p.append(f'<g transform="translate({sx},{sy})"><g clip-path="url(#screen)"><rect width="300" height="186" fill="{scr}"/>')
    kt = f'keyTimes="0;0.08;0.30;0.42;0.50;0.58;0.80;0.92;1"'
    p.append(f'<rect x="20" y="58" width="64" height="64" rx="12" fill="rgba(94,234,212,.35)" fill-opacity="0" stroke="#5eead4" stroke-width="2">'
             f'<animate attributeName="fill-opacity" values="0;0;1;1;0;0;0;0;0" calcMode="discrete" '
             f'{kt} dur="{LOOP}s" repeatCount="indefinite"/></rect>'
             '<text x="52" y="96" font-size="15" font-weight="700" text-anchor="middle" fill="#5eead4">← L</text>')
    p.append(f'<rect x="216" y="58" width="64" height="64" rx="12" fill="rgba(251,191,36,.35)" fill-opacity="0" stroke="#fbbf24" stroke-width="2">'
             f'<animate attributeName="fill-opacity" values="0;0;0;0;0;0;1;1;0" calcMode="discrete" '
             f'{kt} dur="{LOOP}s" repeatCount="indefinite"/></rect>'
             '<text x="248" y="96" font-size="15" font-weight="700" text-anchor="middle" fill="#fbbf24">R →</text>')
    p.append('<path d="M150,30 V156" stroke="#2c3550" stroke-width="1"/>')
    p.append(f'<text x="150" y="22" font-size="12" text-anchor="middle" fill="#9aa4b8">prediction</text>')
    p.append(f'<g><animateTransform attributeName="transform" type="translate" '
             f'values="150 90;150 90;60 90;60 90;150 90;150 90;240 90;240 90;150 90" {kt} dur="{LOOP}s" repeatCount="indefinite"/>'
             '<path d="M0,0 L0,20 L5,15 L9,24 L12,23 L8,14 L15,14 z" fill="#ffffff" stroke="#0d1220" stroke-width="1.2"/></g>')
    p.append('</g></g>')
    p.append(lines(t, px + 20, py + 72 + 260, ["Demo UI moves the cursor", "to the predicted hand"], 14.5, 22))
    p.append(arrow(t, f"M{bx + bw + 4},{by + bh / 2} H{px - 6}", "prediction", (bx + bw + px) / 2, by + bh / 2 - 12))

    p.append("</svg>")
    return "\n".join(p) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "overview.svg").write_text(build("light"))
    (out / "overview-dark.svg").write_text(build("dark"))
    print("wrote", out / "overview.svg", out / "overview-dark.svg")
