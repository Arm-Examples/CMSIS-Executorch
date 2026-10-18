"""Live viewer for the DevKit-E8 face demo.

The board sends one text line per frame: 'F' + 2048 hex digits (32x32 grey
pixels) + newline. This window draws them, enlarged and smoothed.

    .venv/Scripts/python.exe tools/face_viewer.py COM4

Close the VS Code Serial Monitor first: only one program can hold the port.
"""

import sys
import time
import tkinter as tk

import serial
from PIL import Image, ImageTk

SIZE = 32
SCALE = 16  # 32 px -> 512 px window


def main() -> None:
    port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
    ser = serial.Serial(port, 115200, timeout=0.01)

    root = tk.Tk()
    root.title("Edge AI faces - generated live on the Ethos-U85")
    root.configure(bg="black")
    label = tk.Label(root, bg="black")
    label.pack()
    status = tk.Label(root, text="waiting for the board...", fg="white", bg="black", font=("Segoe UI", 12))
    status.pack(fill="x")

    state = {"buf": b"", "frames": 0, "t0": time.time(), "photo": None}

    def poll() -> None:
        state["buf"] += ser.read(4096)
        while b"\n" in state["buf"]:
            line, state["buf"] = state["buf"].split(b"\n", 1)
            line = line.strip()
            if len(line) == 1 + 2 * SIZE * SIZE and line[:1] == b"F":
                try:
                    pixels = bytes.fromhex(line[1:].decode("ascii"))
                except ValueError:
                    continue
                img = Image.frombytes("L", (SIZE, SIZE), pixels)
                img = img.resize((SIZE * SCALE, SIZE * SCALE), Image.BICUBIC)
                state["photo"] = ImageTk.PhotoImage(img)
                label.configure(image=state["photo"])
                state["frames"] += 1
                dt = time.time() - state["t0"]
                if dt > 1.0:
                    status.configure(text=f"{state['frames'] / dt:.1f} faces/s   (joystick: A/B/C/D morph, centre = new face)")
                    state["frames"], state["t0"] = 0, time.time()
        root.after(5, poll)

    poll()
    root.mainloop()


if __name__ == "__main__":
    main()
