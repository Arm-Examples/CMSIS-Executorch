# Model for the Ethos-U85, chosen by the input-shape of the csolution's mlops: node:
#
#   input-shape: 1x16 -> FaceDecoder: a latent vector in, a 32x32 grey face out
#                        (joystick face-morph demo; model/train_face_decoder.py,
#                        weights model/face_decoder.pt)
#   input-shape: 1x30 -> BunnyDecoder: 30 pose dials in, an RGB bunny out
#                        (model/bunny_decoder.py, model/train_bunny.py). Output size
#                        and width from the environment: BUNNY_SIZE (64 or 96,
#                        default 64) and BUNNY_WIDTH (default 0.5). Weights
#                        model/bunny/bunny_decoder_<size>_w<width>.pt; without them
#                        the weights are random (enough to check NPU fit and timing).
#
# Replaces the Arm TinyCNN example (kept in model_sample_tinycnn.py).
# create_ai_layer.py quantizes the model to int8 and compiles it for the NPU.

import os
from pathlib import Path

import torch

from face_decoder import LATENT, FaceDecoder

HERE = Path(__file__).resolve().parent
INPUT_SHAPE = (1, LATENT)


def _is_bunny(input_shape: tuple[int, ...]) -> bool:
    return tuple(input_shape) == (1, 30)


def _is_bunny_v2(input_shape: tuple[int, ...]) -> bool:
    """1x32: the cloud-trained DecoderV2 (model/bunny/decoder_v2_<BUNNY_SIZE>.pt)."""
    return tuple(input_shape) == (1, 32)


def bunny_config() -> tuple[int, float, Path]:
    size = int(os.environ.get("BUNNY_SIZE", "64"))
    width = float(os.environ.get("BUNNY_WIDTH", "0.5"))
    return size, width, HERE / "bunny" / f"bunny_decoder_{size}_w{width:g}.pt"


def get_model(input_shape: tuple[int, ...] = INPUT_SHAPE) -> torch.nn.Module:
    if _is_bunny_v2(input_shape):
        from bunny_decoder_v2 import load_v2

        size = int(os.environ.get("BUNNY_SIZE", "128"))
        path = HERE / "bunny" / f"decoder_v2_{size}.pt"
        print(f"[model] DecoderV2 {size}x{size}: {path.name}")
        return load_v2(path, size).prepare_for_int8().eval()
    if _is_bunny(input_shape):
        from bunny_decoder import BunnyDecoder

        size, width, weights = bunny_config()
        torch.manual_seed(0)
        model = BunnyDecoder(size, width)
        if weights.exists():
            model.load_state_dict(torch.load(weights, map_location="cpu"))
            print(f"[model] BunnyDecoder {size}x{size} width {width:g}: {weights.name}")
        else:
            print(f"[model] BunnyDecoder {size}x{size} width {width:g}: no {weights.name}, RANDOM weights")
        return model.prepare_for_int8().eval()
    model = FaceDecoder()
    model.load_state_dict(torch.load(HERE / "face_decoder.pt", map_location="cpu"))
    return model.eval()


def get_calibration_inputs(
    input_shape: tuple[int, ...] = INPUT_SHAPE, calibration_samples: int = 2
) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    if _is_bunny_v2(input_shape):
        # Real dial vectors from the clips, normalised with dial_ranges_v2.json
        import json
        import sys

        import numpy as np

        import types

        sys.modules.setdefault("cairosvg", types.ModuleType("cairosvg"))  # rendering only; not needed here
        sys.path.insert(0, str(HERE / "bunny" / "kit_v2"))
        rng = json.loads((HERE / "bunny" / "dial_ranges_v2.json").read_text())
        lo, hi = np.array(rng["lo"], np.float32), np.array(rng["hi"], np.float32)
        from clips import CLIPS, DURATION, clip  # kit_v2 motion clips (no cairosvg needed)
        from bunny_rig3 import to_vec

        names = list(CLIPS)
        r = np.random.default_rng(0)
        out = []
        for i in range(calibration_samples):
            name = names[i % len(names)]
            d, _, _, _ = clip(name, float(r.uniform(0, DURATION.get(name, 2.4))), float(r.uniform(0.8, 1.1)))
            for k, (a, b) in (("hue", (-0.07, 0.07)), ("sat", (0.8, 1.15)), ("ear_len", (108, 148)), ("head_scale", (0.92, 1.08))):
                d[k] = float(r.uniform(a, b))
            v = (to_vec(d) - lo) / (hi - lo)
            out.append(torch.from_numpy(np.clip(v, 0, 1)).float().unsqueeze(0))
        return out
    if _is_bunny(input_shape):
        # Real training dial vectors (normalised), like the board will feed in
        import numpy as np

        from bunny_decoder import normalize_dials

        data = HERE / "bunny" / "bunny_train_96.npz"
        if data.exists():
            with np.load(data, allow_pickle=False) as z:
                train = ~z["val_mask"]
                dials = torch.from_numpy(z["dials"][train])
            idx = torch.randperm(len(dials), generator=gen)[:calibration_samples]
            norm = normalize_dials(dials[idx])
        else:
            norm = torch.rand((calibration_samples, 30), generator=gen)
        return [norm[i : i + 1] for i in range(len(norm))]
    # Latents drawn like the ones the board will feed in (unit Gaussian)
    return [torch.randn(input_shape, generator=gen) for _ in range(calibration_samples)]
