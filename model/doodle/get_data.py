"""Download the first N drawings of each class from Google's Quick, Draw! dataset
(28x28 bitmaps, CC BY 4.0, https://github.com/googlecreativelab/quickdraw-dataset)
into model/doodle/data/<class>.npy, using HTTP range requests (no full files)."""
import sys
from pathlib import Path

import numpy as np
import requests

CLASSES = ["star", "sun", "house", "cat", "flower", "fish", "smiley face", "lightning"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
OUT = Path(__file__).resolve().parent / "data"
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"

OUT.mkdir(exist_ok=True)
for c in CLASSES:
    url = URL.format(c.replace(" ", "%20"))
    head = requests.get(url, headers={"Range": "bytes=0-255"}, timeout=60).content
    hlen = int.from_bytes(head[8:10], "little") + 10  # .npy v1 header
    data = requests.get(url, headers={"Range": f"bytes={hlen}-{hlen + N * 784 - 1}"}, timeout=300).content
    arr = np.frombuffer(data, np.uint8).reshape(-1, 28, 28)
    np.save(OUT / f"{c.replace(' ', '_')}.npy", arr)
    print(c, arr.shape, flush=True)
