"""Train the bunny decoder (dials -> RGB image). Runs on CPU or a GPU (Colab).
usage: python train_bunny.py DATA.npz --size 96 --epochs 30 [--width 128] [--out runs/b96] [--resume]
Start grid = size/16, then 4 blocks of (nearest x2 + conv3x3 + ReLU). No ConvTranspose, no GroupNorm/InstanceNorm,
so it stays friendly to ExecuTorch -> Vela -> Ethos-U85. Saves a checkpoint every epoch and a preview sheet."""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, ndial, size, width=128):
        super().__init__()
        assert size % 16 == 0
        self.g = size // 16; self.w = width
        self.fc1 = nn.Linear(ndial, 256)
        self.fc2 = nn.Linear(256, width * self.g * self.g)
        ch = [width, width, width // 2, width // 2, width // 4]
        self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, padding=1) for i in range(4)])
        self.out = nn.Conv2d(ch[-1], 3, 3, padding=1)

    def forward(self, d):
        x = F.relu(self.fc1(d)); x = F.relu(self.fc2(x)).view(-1, self.w, self.g, self.g)
        for c in self.convs:
            x = F.relu(c(F.interpolate(x, scale_factor=2, mode="nearest")))
        return torch.sigmoid(self.out(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="runs/bunny"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_minutes", type=float, default=0, help="stop cleanly after this many minutes (0 = no limit)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(a.data, allow_pickle=False)
    imgs, dials, val = z["images"], z["dials"].astype(np.float32), z["val_mask"]
    assert imgs.shape[1] == a.size, f"data is {imgs.shape[1]}px, --size is {a.size}"
    lo, hi = dials.min(0), dials.max(0); hi = np.where(hi - lo < 1e-6, lo + 1, hi)
    json.dump({"names": [str(n) for n in z["dial_names"]], "lo": lo.tolist(), "hi": hi.tolist()}, open(f"{a.out}/dial_ranges.json", "w"))
    X = torch.from_numpy((dials - lo) / (hi - lo)).clamp(0, 1)
    Y = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()              # uint8 NCHW, converted per batch
    tr, va = np.where(~val)[0], np.where(val)[0]
    m = Decoder(X.shape[1], a.size, a.width).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-5)
    steps = a.epochs * (len(tr) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    ep0 = 0
    ck = f"{a.out}/last.pt"
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=dev); m.load_state_dict(s["m"]); opt.load_state_dict(s["o"]); sched.load_state_dict(s["s"]); ep0 = s["ep"]
        print("resumed from epoch", ep0)
    print(f"device {dev}  params {sum(p.numel() for p in m.parameters())/1e6:.2f}M  train {len(tr)}  val {len(va)}", flush=True)
    t0 = time.time()

    def batch(idx):
        y = Y[idx].to(dev).float() / 255
        return X[idx].to(dev), y

    def loss_fn(p, y):
        w = 1 + 3 * (y.mean(1, keepdim=True) < 0.35).float()                 # dark outline/eye pixels count 4x
        return ((p - y).abs() * w).mean() + 0.2 * ((p - y) ** 2 * w).mean()

    for ep in range(ep0, a.epochs):
        m.train(); perm = np.random.permutation(tr); tot = 0; n = 0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            x, y = batch(torch.from_numpy(perm[i:i + a.bs]))
            l = loss_fn(m(x), y); opt.zero_grad(set_to_none=True); l.backward(); opt.step(); sched.step()
            tot += l.item(); n += 1
        m.eval()
        with torch.no_grad():
            vl = np.mean([loss_fn(m(batch(torch.from_numpy(va[j:j + 256]))[0]), batch(torch.from_numpy(va[j:j + 256]))[1]).item() for j in range(0, len(va), 256)])
        torch.save({"m": m.state_dict(), "o": opt.state_dict(), "s": sched.state_dict(), "ep": ep + 1}, ck)
        torch.save(m.state_dict(), f"{a.out}/decoder_ep{ep+1:03d}.pt")
        print(f"epoch {ep+1}/{a.epochs}  train {tot/n:.4f}  val {vl:.4f}  {(time.time()-t0)/60:.1f} min", flush=True)
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print("time limit reached, stopping cleanly"); break
    # preview: 8 validation targets on top, predictions below
    with torch.no_grad():
        idx = torch.from_numpy(va[:8]); x, y = batch(idx); p = m(x)
    sheet = torch.cat([torch.cat(list(y.cpu()), 2), torch.cat(list(p.cpu()), 2)], 1).permute(1, 2, 0).numpy()
    from PIL import Image
    Image.fromarray((sheet * 255).astype(np.uint8)).save(f"{a.out}/preview.png")
    torch.save(m.state_dict(), f"{a.out}/decoder_final.pt")
    print("done")


if __name__ == "__main__":
    main()
