"""Render an animated SVG to PNG frames on black and white backgrounds (for apng_delta.py).
Usage: python tools/render_frames.py <svg> <prefix> <n_frames> [step_s=0.1] [scale=1.2]
Writes <prefix>_<i>_000000.png and <prefix>_<i>_ffffff.png, paused at t = i * step."""
import re
import subprocess
import sys
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
svg, prefix, n = Path(sys.argv[1]).read_text(), Path(sys.argv[2]), int(sys.argv[3])
step = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
scale = sys.argv[5] if len(sys.argv) > 5 else "1.2"
w, h = re.search(r'width="(\d+)" height="(\d+)"', svg).groups()
prefix.parent.mkdir(parents=True, exist_ok=True)
html = prefix.parent / "frame.html"
for i in range(n):
    for bg in ("000000", "ffffff"):
        html.write_text(f'<html><body style="margin:0;background:#{bg}">' + svg.replace("<svg ", '<svg id="f" ', 1)
                        + f'<script>var s=document.getElementById("f");s.pauseAnimations();s.setCurrentTime({i * step:.3f});</script></body></html>')
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--force-device-scale-factor={scale}",
                        f"--screenshot={prefix}_{i}_{bg}.png", f"--window-size={w},{h}", html.as_uri()],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)
