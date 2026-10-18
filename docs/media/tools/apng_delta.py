"""Write an APNG whose frames after the first only carry the changed regions (OP_SOURCE), so a mostly
static figure stays small. Each time step uses at most 4 sub-frames that split the step delay between them:
browsers treat delays of 10 ms or less as 100 ms, so zero-delay sub-frames would slow the loop down."""
import io, struct, sys, zlib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label, find_objects

def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

def idat_of(img):
    b = io.BytesIO(); img.save(b, "PNG", optimize=True); b = b.getvalue(); out = b""; i = 8
    while i < len(b):
        n = struct.unpack(">I", b[i:i + 4])[0]; t = b[i + 4:i + 8]
        if t == b"IDAT": out += b[i + 8:i + 8 + n]
        i += 12 + n
    return out

def regions(prev, cur, pad=3, max_n=4):
    diff = np.any(prev != cur, axis=-1)
    if not diff.any(): return []
    lab, n = label(binary_dilation(diff, iterations=pad))
    boxes = [(s[0].start, s[0].stop, s[1].start, s[1].stop) for s in find_objects(lab)]
    while len(boxes) > max_n:   # merge the two boxes whose union grows least
        best = None
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                u = (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))
                cost = (u[1] - u[0]) * (u[3] - u[2]) - sum((x[1] - x[0]) * (x[3] - x[2]) for x in (a, b))
                if best is None or cost < best[0]: best = (cost, i, j, u)
        _, i, j, u = best; boxes = [b for k, b in enumerate(boxes) if k not in (i, j)] + [u]
    return boxes

def write(frames, out, delay_ms):
    arrs = [np.asarray(f) for f in frames]; H, W = arrs[0].shape[:2]
    plan = [[(0, H, 0, W)]] + [regions(arrs[k - 1], arrs[k]) or [(0, 1, 0, 1)] for k in range(1, len(arrs))]
    nframes = sum(len(p) for p in plan); seq = 0
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 6, 0, 0, 0))
    data += chunk(b"acTL", struct.pack(">II", nframes, 0))
    for k, boxes in enumerate(plan):
        for bi, (y0, y1, x0, x1) in enumerate(boxes):
            share = delay_ms // len(boxes)
            d = delay_ms - share * (len(boxes) - 1) if bi == len(boxes) - 1 else share
            fc = struct.pack(">IIIIIHHBB", seq, x1 - x0, y1 - y0, x0, y0, d, 1000, 0, 0)
            data += chunk(b"fcTL", fc); seq += 1
            raw = idat_of(Image.fromarray(arrs[k][y0:y1, x0:x1], "RGBA"))
            if k == 0 and bi == 0: data += chunk(b"IDAT", raw)
            else: data += chunk(b"fdAT", struct.pack(">I", seq) + raw); seq += 1
    data += chunk(b"IEND", b"")
    open(out, "wb").write(data)
    return nframes

if __name__ == "__main__":
    P, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 16   # number of 100 ms frames
    frames = []
    for i in range(n):
        k = np.asarray(Image.open(f"{P}_{i}_000000.png").convert("RGB")).astype(float)
        w = np.asarray(Image.open(f"{P}_{i}_ffffff.png").convert("RGB")).astype(float)
        a = np.clip(1 - (w - k).mean(-1) / 255, 0, 1)
        c = np.where(a[..., None] > 1e-3, k / np.maximum(a[..., None], 1e-3), 0)
        frames.append(Image.fromarray(np.dstack([c.clip(0, 255), a * 255]).round().astype(np.uint8), "RGBA"))
    print(out, write(frames, out, 100), "sub-frames")
