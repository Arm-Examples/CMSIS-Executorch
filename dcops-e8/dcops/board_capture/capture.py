"""Record the E8 board camera's model-input frames over J-Link, labelled for fine-tuning.

slides mode:  python3 capture.py slides   -> waits for SPACE in index.html (it calls
              http://127.0.0.1:8799/start), then captures ~1 frame/s while the slideshow
              advances every SEC seconds; each frame gets the label of the slide on screen.
background:   python3 capture.py background --seconds 120   -> every frame labelled background.
Output: frames/<mode>_<n>.npz (160x160x3 uint8 + label), appended across runs.
"""
import argparse, http.server, json, os, subprocess, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AXF = os.environ.get("AXF", os.path.join(HERE, "../../build-fusion/bin/mlek_alif_object_detection.axf"))
JLINK = "/Applications/SEGGER/JLink/JLinkExe"
SEC = 6
BG = 16
SPEED = os.environ.get("JLINK_SPEED", "12000")


def symbol(name):
    out = subprocess.run(["bash", "-c", f"source ~/projects/arm-executorch-hack/env.sh; arm-none-eabi-nm -S {AXF}"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        p = line.split()
        if len(p) == 4 and p[3] == name:
            return int(p[0], 16), int(p[1], 16)
    raise SystemExit(f"symbol {name} not found")


def grab(addr, size, path):
    script = f"/tmp/cap_{os.getpid()}.jlink"
    open(script, "w").write(f"connect\nsavebin {path} 0x{addr:08X} 0x{176 + 160 * 160 * 3:X}\nexit\n")
    subprocess.run([JLINK, "-device", "AE822FA0E5597LS0_M55_HP", "-if", "SWD", "-speed", SPEED,
                    "-autoconnect", "1", "-CommanderScript", script], capture_output=True)
    if not os.path.exists(path) or os.path.getsize(path) < 176 + 160 * 160 * 3:
        return None
    raw = np.fromfile(path, np.uint8)
    os.remove(path)
    return raw[176:176 + 160 * 160 * 3].reshape(160, 160, 3).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["slides", "background", "clock"])
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--deck", type=int, default=1, help="1: slides.json, 6 s/slide; 2: slides2.json, 4 s/slide")
    a = ap.parse_args()
    global SEC
    if a.deck == 2:
        SEC = 4
    addr, size = symbol("rgb_image")
    os.makedirs(f"{HERE}/frames", exist_ok=True)
    slides = json.load(open(f"{HERE}/slides{2 if a.deck == 2 else ''}.json"))
    t0 = None
    if a.mode == "slides":
        started = threading.Event(); box = {}

        class H(http.server.SimpleHTTPRequestHandler):
            """Serves the slideshow (so a phone on the same Wi-Fi can show it) and /start."""
            def __init__(self, *args, **kw): super().__init__(*args, directory=HERE, **kw)

            def do_GET(self):
                if self.path.startswith("/start"):
                    if not started.is_set():
                        box["t0"] = time.time(); started.set()   # Mac clock: the phone's may differ
                    self.send_response(200); self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                    return
                super().do_GET()

            def log_message(self, *x): pass
        srv = http.server.ThreadingHTTPServer(("0.0.0.0", 8799), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        import socket
        ip = socket.gethostbyname(socket.gethostname())
        print(f"open http://{ip}:8799/index.html (phone) or index.html (laptop); tap/SPACE to start", flush=True)
        started.wait(); t0 = box["t0"]
        end = t0 + len(slides) * SEC
    elif a.mode == "clock":
        # same schedule as the phone artifact: slide k during [EPOCH + 6k, EPOCH + 6(k+1)) s, cycling
        import datetime
        t0 = datetime.datetime(2026, 10, 18, 19 if a.deck == 2 else 0, tzinfo=datetime.timezone.utc).timestamp()
        end = time.time() + len(slides) * SEC + 2
        print(f"clock mode: recording one full cycle ({len(slides) * SEC / 60:.1f} min)", flush=True)
    else:
        end = time.time() + a.seconds
    n_saved = 0; n_skip = 0; frames, labels, ids = [], [], []
    tag = f"{a.mode}{'_deck2' if a.deck == 2 else ''}_{time.strftime('%H%M%S')}"
    while time.time() < end:
        t_req = time.time()
        img = grab(addr, size, f"/tmp/cap_{os.getpid()}.bin")
        if img is None:
            continue
        if a.mode == "background":
            frames.append(img); labels.append(BG); n_saved += 1
        else:
            t_frame = t_req - 0.4 - t0          # camera + loop latency
            k = int(t_frame // SEC); phase = t_frame - k * SEC
            if a.mode == "clock": k %= len(slides)
            if 0 <= k < len(slides) and 1.2 < phase < SEC - 0.6:
                frames.append(img); labels.append(slides[k][2]); ids.append(k); n_saved += 1
            else:
                n_skip += 1
        if (n_saved + n_skip) % 20 == 0:
            print(f"{time.strftime('%H:%M:%S')} saved {n_saved}, skipped {n_skip}", flush=True)
    if frames:
        np.savez_compressed(f"{HERE}/frames/{tag}.npz", images=np.stack(frames), labels=np.array(labels),
                            slide_ids=np.array(ids if ids else [-1] * len(labels)))
    print(f"done: {n_saved} frames -> frames/{tag}.npz ({n_skip} skipped near slide changes)")


if __name__ == "__main__":
    main()
