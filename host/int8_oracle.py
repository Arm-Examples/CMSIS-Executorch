"""Host-side int8 model oracle (optional, needs the Reproduce_Baselines env: torch, executorch).

Rebuilds the fake-quant int8 graph of the release model exactly as ``Reproduce_Baselines``
did (``tinyeegnet_float.pt`` + ``student.json`` + PTQ on ``calibration.npz``) so host tools
can score *any* int8 window, in particular the windows ``pipeline_ref`` computes from the int16
UART stream. Self-check: ``logits(test_vectors.x)`` == ``test_vectors.logits``.

    oracle = Int8Oracle()                 # ~10 s
    logits = oracle.logits(x_int8)        # (n, 1, 22, 256) int8 -> (n, 2) float32
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RB_ARTIFACTS, RB_DIR, load_params  # noqa: E402

ARTIFACTS = RB_ARTIFACTS


class Int8Oracle:
    def __init__(self, artifacts: Path = ARTIFACTS, params: dict | None = None):
        if str(RB_DIR) not in sys.path:
            sys.path.insert(0, str(RB_DIR))
        from src.quantize import input_qparams, load_float_model, quantize  # Reproduce_Baselines

        params = params if params is not None else load_params()
        widths = {k: int(v) for k, v in json.loads((artifacts / "student.json").read_text()).items()}
        shape = tuple(params["input_shape"])
        model = load_float_model(artifacts / "tinyeegnet_float.pt", shape[-1], widths)
        calib = np.load(artifacts / "calibration.npz")["x"].astype(np.float32)
        self.graph = quantize(model, calib, input_shape=shape)
        self.q_scale, self.zp = input_qparams(self.graph)
        if not (np.isclose(self.q_scale, params["input_quant_scale"], rtol=1e-6) and self.zp == params["input_zero_point"]):
            raise AssertionError(f"rebuilt qparams ({self.q_scale}, {self.zp}) != params.json "
                                 f"({params['input_quant_scale']}, {params['input_zero_point']})")

    def logits(self, x_int8: np.ndarray) -> np.ndarray:
        import torch

        x = (np.asarray(x_int8, dtype=np.float32) - self.zp) * np.float32(self.q_scale)
        x = torch.from_numpy(x.reshape(-1, 1, *x.shape[-2:]))
        with torch.no_grad():
            return np.concatenate([self.graph(x[i:i + 1]).numpy() for i in range(len(x))]).astype(np.float32)
