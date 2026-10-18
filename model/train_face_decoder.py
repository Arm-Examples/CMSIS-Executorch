"""Train a tiny face VAE on the Olivetti faces (400 photos) and save its decoder.

Run with the project venv:  .venv/Scripts/python.exe model/train_face_decoder.py
Writes model/face_decoder.pt (decoder weights, used by model/model.py) and a
preview image face_preview.png next to it.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_decoder import FaceDecoder, LATENT, SIZE  # noqa: E402

HERE = Path(__file__).resolve().parent


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),   # 16x16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 8x8
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), # 4x4
        )
        self.mu = nn.Linear(128 * 4 * 4, LATENT)
        self.logvar = nn.Linear(128 * 4 * 4, LATENT)

    def forward(self, x):
        h = torch.flatten(self.net(x), 1)
        return self.mu(h), self.logvar(h)


def load_faces() -> torch.Tensor:
    from sklearn.datasets import fetch_olivetti_faces

    data = fetch_olivetti_faces(data_home=os.environ.get("TEMP", "."), download_if_missing=True)
    x = torch.tensor(data.images).unsqueeze(1)  # N,1,64,64 in [0,1]
    return F.interpolate(x, size=(SIZE, SIZE), mode="area")


def main() -> None:
    torch.manual_seed(0)
    x = load_faces()
    enc, dec = Encoder(), FaceDecoder()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-3)
    beta = 0.5 / (SIZE * SIZE)  # weak KL term: sharper faces, still a usable latent space
    epochs = int(os.environ.get("EPOCHS", "800"))

    for epoch in range(epochs):
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), 64):
            b = x[perm[i : i + 64]]
            if torch.rand(1) < 0.5:
                b = torch.flip(b, dims=[3])  # mirror augmentation
            mu, logvar = enc(b)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            recon = dec(z)
            rec = F.mse_loss(recon, b, reduction="sum") / len(b)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(b)
            loss = rec + beta * 1000 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d} loss {total:.3f}", flush=True)

    torch.save(dec.state_dict(), HERE / "face_decoder.pt")

    # Preview: row 1 originals, row 2 reconstructions, rows 3-4 random faces
    from PIL import Image

    dec.eval(); enc.eval()
    with torch.no_grad():
        orig = x[:8]
        mu, _ = enc(orig)
        recon = dec(mu)
        rand = dec(torch.randn(16, LATENT))
    rows = [torch.cat(list(orig), dim=2), torch.cat(list(recon), dim=2),
            torch.cat(list(rand[:8]), dim=2), torch.cat(list(rand[8:]), dim=2)]
    grid = torch.cat(rows, dim=1).squeeze(0).clamp(0, 1).numpy()
    img = Image.fromarray((grid * 255).astype(np.uint8)).resize((grid.shape[1] * 4, grid.shape[0] * 4), Image.NEAREST)
    img.save(HERE / "face_preview.png")
    print("saved", HERE / "face_decoder.pt", "and face_preview.png")


if __name__ == "__main__":
    main()
