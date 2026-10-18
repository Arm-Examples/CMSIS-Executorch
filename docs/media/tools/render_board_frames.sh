#!/bin/bash
# render.sh <svg> <prefix> : 16 frames x (black, white) at 1.2x, t = 0.0 .. 1.5 s
SVG=$1; P=$2; D=$(dirname $P)
for i in $(seq 0 15); do
  t=$(python3 -c "print($i*0.1)")
  for bg in 000000 ffffff; do
    { printf '<html><body style="margin:0;background:#%s">' $bg; sed 's/<svg /<svg id="f" /' "$SVG" | sed 's/width="1500" height="640"/width="1500" height="640"/'; printf '<script>var s=document.getElementById("f");s.pauseAnimations();s.setCurrentTime(%s);</script></body></html>' $t; } > $D/frame.html
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1.2 --screenshot=${P}_${i}_${bg}.png --window-size=1500,640 file://$D/frame.html 2>/dev/null
  done
done
