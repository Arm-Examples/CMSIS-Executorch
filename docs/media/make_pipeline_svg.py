"""Generate the Edge_Deploy pipeline block diagram (light + dark animated SVG)."""
import sys
from pathlib import Path

W, H = 1000, 730
CW, CH = 300, 138
COLS = [24, 350, 676]
ROWS = [24, 230, 436]

# Glass look: each card is its hue at low alpha, edged in the same hue, with a white sheen on top.
HUES = {"host": "124,92,255", "m55": "244,63,94", "u85": "245,158,11", "board": "6,182,212", "band": "234,179,8"}

THEMES = {
    "light": dict(
        title="#2b2b2b", tools="#3a3a3a", role="#6b6b6b",
        stroke="#4b4b4b", arrow="#9a9a9a", label="#6b6b6b",
        fill_a=0.13, edge_a=0.75, chip_a=0.28, sheen_a=0.65,
    ),
    "dark": dict(
        title="#e6edf3", tools="#c9d1d9", role="#8b949e",
        stroke="#c9d1d9", arrow="#6e7681", label="#8b949e",
        fill_a=0.18, edge_a=0.85, chip_a=0.40, sheen_a=0.10,
    ),
}


def glass(t, place, x, y, w, h, rx):
    """Translucent card: tinted fill, highlighted edge, top sheen."""
    hue = HUES[place]
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="rgba({hue},{t["fill_a"]})" '
            f'stroke="rgba({hue},{t["edge_a"]})" stroke-width="1.5"/>'
            f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{w - 3}" height="{h * 0.5}" rx="{rx - 1.5}" fill="url(#sheen)"/>')

CHIP_TEXT = {"host": "HOST", "m55": "M55", "u85": "U85", "board": "BOARD"}

# 24x24 line icons; extra animated parts are appended per icon.
ICONS = {
    "wave": '<path d="M1 12h3l2-5 3 10 3-13 3 11 2-3h6" stroke-dasharray="56 56" '
            '><animate attributeName="stroke-dashoffset" '
            'values="0;-112" dur="2.4s" repeatCount="indefinite"/></path>',
    "laptop": '<rect x="3" y="5" width="18" height="11" rx="1.5"/><path d="M1 19h22"/>',
    "plug": '<path d="M8 3v5M16 3v5M5 8h14v3a7 7 0 0 1-14 0zM12 18v3"/>',
    "chip": '<rect x="6" y="6" width="12" height="12" rx="1.5"/>'
            '<path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>'
            '<path d="M9 13l1.5-3 2 4 1.5-2"/>',
    "npu": '<rect x="6" y="6" width="12" height="12" rx="1.5"/>'
           '<path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>'
           '<rect x="9.5" y="9.5" width="5" height="5" rx="1" fill="{stroke}" stroke="none">'
           '<animate attributeName="opacity" values="1;0.25;1" dur="0.8s" repeatCount="indefinite"/></rect>',
    "shield": '<path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6z"/>'
              '<path d="M9 12l2 2 4-4"><animate attributeName="opacity" values="1;1;0;1" '
              'keyTimes="0;0.6;0.8;1" dur="3s" repeatCount="indefinite"/></path>',
    "led": '<circle cx="12" cy="9" r="6" stroke="none">'
           '<animate attributeName="fill" values="#2da44e;#1f6feb;#cf222e" calcMode="discrete" '
           'dur="3s" repeatCount="indefinite"/></circle>'
           '<circle cx="12" cy="9" r="6"/><path d="M9.5 15v6M14.5 15v6"/>',
    "monitor": '<rect x="2" y="4" width="20" height="13" rx="1.5"/><path d="M8 21h8M12 17v4"/>'
               '<path d="M5 13l3-3 3 2 4-5 4 3"/>',
    "imu": '<circle cx="12" cy="13" r="2"/><path d="M12 11V3M10 5l2-2 2 2M13.7 14l6.3 4M17.5 18.6l2.5-.6-.6-2.4'
           'M10.3 14L4 18M6.5 18.6L4 18l.6-2.4"/>'
           '<circle cx="12" cy="13" r="8" stroke-dasharray="2 3">'
           '<animate attributeName="opacity" values="0.9;0.1;0.9" dur="1.5s" repeatCount="indefinite"/></circle>',
}

# (col, row, place, icon, title, tools, role lines)
CARDS = [
    (0, 0, "host", "wave", "EEG REPLAY", "BCI IV 2a · session 2", ["held-out trials, sample by sample"]),
    (1, 0, "host", "laptop", "HOST BRIDGE", "bridge.py", ["raw samples only; no filtering, no ML"]),
    (2, 0, "board", "plug", "UART LINK", "921600 baud · 0xAA55 frames", ["22 ch × int16 per sample, 250 Hz"]),
    (2, 1, "m55", "chip", "PREPROCESSING", "CAR · 7–30 Hz · 125 Hz · EA", ["int8 window 22 × 256 · Figure 2"]),
    (1, 1, "u85", "npu", "ETHOS-U85 NPU", "TinyEEGNet-wide int8 · ExecuTorch", ["22 × 256 → 2 logits in 0.32 ms"]),
    (0, 1, "m55", "shield", "MOTION GATE", "IMU > 0.25 g in last 500 ms", ["suppresses the prediction"]),
    (0, 2, "board", "imu", "IMU", "ICM-42670-P · I3C", ["100 Hz accel, on the board"]),
    (1, 2, "board", "led", "OUTPUT", "LEDs · UART JSON", ["green left · blue right · red motion"]),
    (2, 2, "host", "monitor", "DEMO UI", "demo_ui.py · measure.py", ["class, confidence, latency, motion"]),
]

# Card drawn with an animated outline: it is expanded in Figure 2 (make_signal_svg.py).
HIGHLIGHT = (2, 1)

BANDS = [
    (598, "OFFLINE BUILD", "ml/: 4 teachers → student → int8 → Vela → .pte", "release v0.5-distill"),
    (656, "VERIFY", "FVP self-test bit-exact · replay 1,080/1,080 trials", "latency · NPU cycles · accuracy"),
]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def flow(d, t, label=None, lx=0, ly=0, anchor="start"):
    out = (f'<path d="{d}" fill="none" stroke="{t["arrow"]}" stroke-width="2.2" '
           f'stroke-dasharray="5 5" marker-end="url(#ah)">'
           f'<animate attributeName="stroke-dashoffset" from="10" to="0" dur="0.6s" repeatCount="indefinite"/></path>')
    if label:
        out += (f'<text x="{lx}" y="{ly}" font-size="12.5" fill="{t["label"]}" '
                f'text-anchor="{anchor}">{esc(label)}</text>')
    return out


def build(theme):
    t = THEMES[theme]
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         "font-family=\"Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif\">",
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t["arrow"]}"/></marker>'
         '<linearGradient id="sheen" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0" stop-color="#ffffff" stop-opacity="{t["sheen_a"]}"/>'
         '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient></defs>']

    for col, row, place, icon, title, tools, role in CARDS:
        x, y = COLS[col], ROWS[row]
        p.append(glass(t, place, x, y, CW, CH, 16))
        p.append(f'<g transform="translate({x + 20},{y + 20}) scale(1.25)" fill="none" stroke="{t["stroke"]}" '
                 f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
                 f'{ICONS[icon].replace("{stroke}", t["stroke"])}</g>')
        p.append(f'<text x="{x + 62}" y="{y + 43}" font-size="16" font-weight="700" letter-spacing="0.4" '
                 f'fill="{t["title"]}">{esc(title)}</text>')
        p.append(f'<rect x="{x + CW - 76}" y="{y + 18}" width="58" height="22" rx="11" '
                 f'fill="rgba({HUES[place]},{t["chip_a"]})" stroke="rgba({HUES[place]},{t["edge_a"]})" stroke-width="1"/>')
        p.append(f'<text x="{x + CW - 47}" y="{y + 33.5}" font-size="11" font-weight="700" letter-spacing="0.6" '
                 f'text-anchor="middle" fill="{t["title"]}">{CHIP_TEXT[place]}</text>')
        p.append(f'<text x="{x + 22}" y="{y + 84}" font-size="14.5" font-weight="600" fill="{t["tools"]}">{esc(tools)}</text>')
        for i, line in enumerate(role):
            p.append(f'<text x="{x + 22}" y="{y + 110 + 18 * i}" font-size="13" fill="{t["role"]}">{esc(line)}</text>')

    hx, hy = COLS[HIGHLIGHT[0]], ROWS[HIGHLIGHT[1]]
    p.append(f'<rect x="{hx - 6}" y="{hy - 6}" width="{CW + 12}" height="{CH + 12}" rx="20" fill="none" '
             f'stroke="rgba({HUES["m55"]},{t["edge_a"]})" stroke-width="2" stroke-dasharray="8 6">'
             '<animate attributeName="stroke-dashoffset" from="0" to="-28" dur="1.6s" repeatCount="indefinite"/></rect>')

    r0, r1, r2 = ROWS
    c0, c1, c2 = COLS
    mid = CH / 2
    # top row, left to right
    p.append(flow(f"M{c0 + CW + 4},{r0 + mid} H{c1 - 4}", t))
    p.append(flow(f"M{c1 + CW + 4},{r0 + mid} H{c2 - 4}", t))
    # UART down into the M55
    p.append(flow(f"M{c2 + CW / 2},{r0 + CH + 4} V{r1 - 4}", t, "frames · 250 Hz", c2 + CW / 2 + 10, r0 + CH + 38))
    # middle row, right to left
    p.append(flow(f"M{c2 - 4},{r1 + mid} H{c1 + CW + 4}", t))
    p.append(flow(f"M{c1 - 4},{r1 + mid} H{c0 + CW + 4}", t))
    # IMU up into the gate
    p.append(flow(f"M{c0 + CW / 2},{r2 - 4} V{r1 + CH + 4}", t, "100 Hz accel", c0 + CW / 2 + 10, r1 + CH + 38))
    # gate to output: elbow through the row gap
    gx = c0 + CW - 50
    gy = r1 + CH + (r2 - r1 - CH) / 2
    p.append(flow(f"M{gx},{r1 + CH + 4} V{gy} H{c1 + CW / 2} V{r2 - 4}", t, "cls · conf · motion",
                  c1 + CW / 2 + 10, gy - 8))
    # output back to the laptop UI
    p.append(flow(f"M{c1 + CW + 4},{r2 + mid} H{c2 - 4}", t))

    for y, name, body, right in BANDS:
        p.append(glass(t, "band", 24, y, W - 48, 48, 12))
        p.append(f'<text x="46" y="{y + 30}" font-size="15" font-weight="700" letter-spacing="0.4" '
                 f'fill="{t["title"]}">{esc(name)}</text>')
        p.append(f'<text x="210" y="{y + 30}" font-size="14.5" font-weight="600" fill="{t["tools"]}">{esc(body)}</text>')
        p.append(f'<text x="{W - 46}" y="{y + 30}" font-size="13" text-anchor="end" fill="{t["role"]}">{esc(right)}</text>')

    p.append("</svg>")
    return "\n".join(p) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline.svg").write_text(build("light"))
    (out / "pipeline-dark.svg").write_text(build("dark"))
    print("wrote", out / "pipeline.svg", out / "pipeline-dark.svg")
