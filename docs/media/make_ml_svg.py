"""Generate the ML training figure (model A vs model B), light + dark animated SVG.

Same glass style as make_pipeline_svg.py: tinted card, edge in the same hue, white sheen,
line icons, animated dashed flows. Run: python docs/media/make_ml_svg.py docs/media
"""
import sys
from pathlib import Path

from make_pipeline_svg import THEMES, esc

W, H = 1120, 762
CW = 235
COLS = [24, 303, 582, 861]
ROW_A, CH_A = 82, 170
ROW_B1, CH_B1 = 352, 152
ROW_B2, CH_B2 = 560, 170

HUES = {"data": "6,182,212", "train": "124,92,255", "distill": "244,63,94",
        "deploy": "245,158,11", "test": "34,197,94", "a": "100,116,139", "b": "124,92,255"}


def glass(t, hue, x, y, w, h, rx):
    """Translucent card: tinted fill, highlighted edge, top sheen."""
    c = HUES[hue]
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="rgba({c},{t["fill_a"]})" '
            f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1.5"/>'
            f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{w - 3}" height="{h * 0.45}" rx="{rx - 1.5}" fill="url(#sheen)"/>')


# 24x24 line icons; animated parts inline.
ICONS = {
    "data": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'
            '<path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>'
            '<path d="M7 18.5l1.5-2 1.5 3 1.5-4 1.5 3 1.5-1.5 1.5 1" stroke-width="1.4" stroke-dasharray="14 14">'
            '<animate attributeName="stroke-dashoffset" values="0;-28" dur="2.4s" repeatCount="indefinite"/></path>',
    "train": '<path d="M12 4a4 4 0 0 0-4 4 4 4 0 0 0-3 6.5A3.5 3.5 0 0 0 9 20h3zM12 4a4 4 0 0 1 4 4 4 4 0 0 1 3 6.5'
             'A3.5 3.5 0 0 1 15 20h-3z"/><path d="M12 4v16"/>'
             '<circle cx="9" cy="11" r="1.2" stroke="none" fill="{stroke}"><animate attributeName="opacity" '
             'values="1;0.2;1" dur="1.2s" repeatCount="indefinite"/></circle>'
             '<circle cx="15" cy="14" r="1.2" stroke="none" fill="{stroke}"><animate attributeName="opacity" '
             'values="0.2;1;0.2" dur="1.2s" repeatCount="indefinite"/></circle>',
    "teachers": '<path d="M2 9l10-5 10 5-10 5z"/><path d="M6 11v5c0 1.5 2.7 3 6 3s6-1.5 6-3v-5"/>'
                '<path d="M22 9v6"><animate attributeName="opacity" values="1;0.3;1" dur="1.6s" '
                'repeatCount="indefinite"/></path>',
    "distill": '<path d="M3 4h18l-7 8v6l-4 2v-8z"/>'
               '<circle cx="12" cy="22.5" r="1" stroke="none" fill="{stroke}">'
               '<animate attributeName="cy" values="13;23" dur="1.2s" repeatCount="indefinite"/>'
               '<animate attributeName="opacity" values="1;0" dur="1.2s" repeatCount="indefinite"/></circle>',
    "npu": '<rect x="6" y="6" width="12" height="12" rx="1.5"/>'
           '<path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>'
           '<rect x="9.5" y="9.5" width="5" height="5" rx="1" fill="{stroke}" stroke="none">'
           '<animate attributeName="opacity" values="1;0.25;1" dur="0.8s" repeatCount="indefinite"/></rect>',
    "test": '<circle cx="12" cy="12" r="9"/>'
            '<path d="M8 12.5l3 3 5-6" stroke-dasharray="12 12">'
            '<animate attributeName="stroke-dashoffset" values="12;0;0;12" keyTimes="0;0.3;0.85;1" '
            'dur="3s" repeatCount="indefinite"/></path>',
}

# (col, row_y, height, hue, icon, title, tag, tools, role lines, span)
CARDS = [
    (0, ROW_A, CH_A, "data", "data", "DATASET", "DATA", "BCI IV 2a",
     ["subject 1, session 1", "115 train · 29 val trials"], 1),
    (1, ROW_A, CH_A, "train", "train", "TRAIN", "", "TinyEEGNet",
     ["1,938 params", "cross-entropy, early stop", "on validation loss"], 1),
    (2, ROW_A, CH_A, "deploy", "npu", "DEPLOY", "U85", "int8 · 13,072 B",
     ["ExecuTorch .pte", "on the Ethos-U85"], 1),
    (3, ROW_A, CH_A, "test", "test", "TEST", "79.9 %", "subject 1, session 2",
     ["144 trials, float", "board int8 79.2 %, = FVP"], 1),
    (0, ROW_B1, CH_B1, "data", "data", "DATASETS", "DATA", "Cho2017 + Lee2019",
     ["72 subjects", "no BCI IV 2a labels"], 1),
    (1, ROW_B1, CH_B1, "train", "teachers", "TEACHERS", "STEP 1", "4-model ensemble", [], 3),
    (1, ROW_B2, CH_B2, "distill", "distill", "DISTILL", "STEP 2", "TinyEEGNet-wide",
     ["3,426 params, learns", "the teachers' soft labels"], 1),
    (2, ROW_B2, CH_B2, "deploy", "npu", "DEPLOY", "U85", "int8 · 11,456 B",
     ["ExecuTorch .pte", "on the Ethos-U85"], 1),
    (3, ROW_B2, CH_B2, "test", "test", "TEST", "74.6 %", "all 9 BCI IV 2a",
     ["subjects, session 2", "1,080 trials, int8"], 1),
]

TEACHERS = ["EEGNet", "ShallowConvNet", "DeepConvNet", "Riemann"]

PANELS = [
    (ROW_A, CH_A, "a", "A", "SUBJECT-SPECIFIC MODEL", "subject-specific TinyEEGNet, trained and tested on one person"),
    (ROW_B1, ROW_B2 + CH_B2 - ROW_B1, "b", "B", "GENERIC MODEL", "four teachers distilled into one small student that serves every subject"),
]


def flow(t, x1, x2, y, label=None):
    out = (f'<path d="M{x1 + 4},{y} H{x2 - 4}" fill="none" stroke="{t["arrow"]}" stroke-width="2.2" '
           f'stroke-dasharray="5 5" marker-end="url(#ah)">'
           f'<animate attributeName="stroke-dashoffset" from="10" to="0" dur="0.6s" repeatCount="indefinite"/></path>')
    if label:
        out += (f'<text x="{(x1 + x2) / 2}" y="{y - 9}" font-size="11.5" fill="{t["label"]}" '
                f'text-anchor="middle">{esc(label)}</text>')
    return out


def card(t, col, y, h, hue, icon, title, tag, tools, role, span):
    x = COLS[col]
    w = CW * span + (COLS[1] - COLS[0] - CW) * (span - 1)
    c = HUES[hue]
    tag_w = 12 + 8.6 * len(tag)
    p = [glass(t, hue, x, y, w, h, 16),
         f'<g transform="translate({x + 16},{y + 16}) scale(1.3)" fill="none" stroke="{t["stroke"]}" '
         f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
         f'{ICONS[icon].replace("{stroke}", t["stroke"])}</g>',
         f'<text x="{x + 58}" y="{y + 40}" font-size="18" font-weight="700" letter-spacing="0.4" '
         f'fill="{t["title"]}">{esc(title)}</text>',
         '' if not tag else f'<rect x="{x + w - tag_w - 14}" y="{y + 22}" width="{tag_w}" height="26" rx="13" '
         f'fill="rgba({c},{t["chip_a"]})" stroke="rgba({c},{t["edge_a"]})" stroke-width="1"/>',
         '' if not tag else f'<text x="{x + w - 14 - tag_w / 2}" y="{y + 40}" font-size="13" font-weight="700" letter-spacing="0.5" '
         f'text-anchor="middle" fill="{t["title"]}">{esc(tag)}</text>',
         f'<text x="{x + 18}" y="{y + 82}" font-size="17" font-weight="600" fill="{t["tools"]}">{esc(tools)}</text>']
    for i, line in enumerate(role):
        p.append(f'<text x="{x + 18}" y="{y + 110 + 21 * i}" font-size="15" fill="{t["role"]}">{esc(line)}</text>')
    return "".join(p)


def teacher_chips(t, y):
    """Four teacher chips in one row, lighting up one after another."""
    w = CW * 3 + (COLS[1] - COLS[0] - CW) * 2
    gap = 14
    cw, chh = (w - 36 - 3 * gap) / 4, 40
    x0 = COLS[1] + 18
    c = HUES["train"]
    out = []
    for i, name in enumerate(TEACHERS):
        cx = x0 + i * (cw + gap)
        cy = y + 96
        out.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{chh}" rx="12" fill="rgba({c},{t["chip_a"]})" '
                   f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1.2">'
                   f'<animate attributeName="fill-opacity" values="0.35;1;0.35" dur="2s" begin="{i * 0.5}s" '
                   f'repeatCount="indefinite"/></rect>')
        out.append(f'<text x="{cx + cw / 2}" y="{cy + 26}" font-size="16" font-weight="600" '
                   f'text-anchor="middle" fill="{t["title"]}">{esc(name)}</text>')
    return "".join(out)


def build(theme):
    t = THEMES[theme]
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         "font-family=\"Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif\">",
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t["arrow"]}"/></marker>'
         '<linearGradient id="sheen" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0" stop-color="#ffffff" stop-opacity="{t["sheen_a"]}"/>'
         '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient></defs>']

    for y, h, hue, letter, name, sub in PANELS:
        c = HUES[hue]
        p.append(f'<rect x="10" y="{y - 56}" width="{W - 20}" height="{h + 70}" rx="20" fill="rgba({c},0.04)" '
                 f'stroke="rgba({c},0.45)" stroke-width="1.2" stroke-dasharray="6 5"/>')
        p.append(f'<circle cx="{COLS[0] + 14}" cy="{y - 30}" r="16" fill="rgba({c},{t["chip_a"]})" '
                 f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1.2"/>')
        p.append(f'<text x="{COLS[0] + 14}" y="{y - 24}" font-size="16" font-weight="800" text-anchor="middle" '
                 f'fill="{t["title"]}">{letter}</text>')
        p.append(f'<text x="{COLS[0] + 42}" y="{y - 24}" font-size="18" font-weight="700" letter-spacing="0.5" '
                 f'fill="{t["title"]}">{esc(name)}</text>')
        p.append(f'<text x="{COLS[0] + 42 + 12.2 * len(name) + 16}" y="{y - 24}" font-size="15" '
                 f'fill="{t["role"]}">{esc(sub)}</text>')

    for c in CARDS:
        p.append(card(t, *c))
    p.append(teacher_chips(t, ROW_B1))

    def right(col, y):
        return flow(t, COLS[col] + CW, COLS[col + 1], y)

    ya, yb1, yb2 = ROW_A + CH_A / 2, ROW_B1 + CH_B1 / 2, ROW_B2 + CH_B2 / 2
    p += [right(0, ya), right(1, ya), right(2, ya), right(0, yb1), right(1, yb2), right(2, yb2)]
    # teachers down into the student
    sx = COLS[1] + CW / 2
    p.append(f'<path d="M{sx},{ROW_B1 + CH_B1 + 4} V{ROW_B2 - 4}" fill="none" stroke="{t["arrow"]}" '
             f'stroke-width="2.2" stroke-dasharray="5 5" marker-end="url(#ah)"><animate '
             f'attributeName="stroke-dashoffset" from="10" to="0" dur="0.6s" repeatCount="indefinite"/></path>')
    p.append(f'<text x="{sx + 12}" y="{(ROW_B1 + CH_B1 + ROW_B2) / 2 + 5}" font-size="15" '
             f'fill="{t["label"]}">soft labels</text>')

    p.append("</svg>")
    return "\n".join(p) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "ml.svg").write_text(build("light"))
    (out / "ml-dark.svg").write_text(build("dark"))
    print("wrote", out / "ml.svg", out / "ml-dark.svg")
