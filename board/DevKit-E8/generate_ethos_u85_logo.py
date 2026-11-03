"""Convert the supplied Ethos-U85 JPEG into LCD-ready RGB565 pixels.

Run from the repository root with:
    python3 board/DevKit-E8/generate_ethos_u85_logo.py
"""

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Ethos_U85.jpeg"
OUTPUT = Path(__file__).with_name("ethos_u85_logo.h")
WIDTH, HEIGHT = 120, 106

image = Image.open(SOURCE).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
pixels = [((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3) for r, g, b in image.getdata()]

with OUTPUT.open("w", encoding="ascii") as out:
    out.write("/* Generated from Ethos_U85.jpeg by generate_ethos_u85_logo.py. */\n")
    out.write("#ifndef ETHOS_U85_LOGO_H\n#define ETHOS_U85_LOGO_H\n\n")
    out.write("#include <stdint.h>\n\n")
    out.write(f"#define ETHOS_U85_LOGO_WIDTH {WIDTH}\n")
    out.write(f"#define ETHOS_U85_LOGO_HEIGHT {HEIGHT}\n\n")
    out.write("static const uint16_t ethos_u85_logo[ETHOS_U85_LOGO_WIDTH * ETHOS_U85_LOGO_HEIGHT] = {\n")
    for start in range(0, len(pixels), 12):
        out.write("    " + ", ".join(f"0x{pixel:04x}" for pixel in pixels[start:start + 12]) + ",\n")
    out.write("};\n\n#endif\n")
