# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Build a PLACEHOLDER keyword-spotter run directory (random-weight DSCNN(64)) laid out exactly like a
real dcops/jobs/kws.py output, so dcops/export/export_kws.py can turn it into kws_ethos-u85-256.pte
and the firmware can be built/linked before the real model exists. Predictions are meaningless.

  .venv-et110/bin/python dcops/export/make_kws_placeholder.py [out_dir]
  .venv-et110/bin/python dcops/export/export_kws.py dcops/models/placeholder/kws_random
"""
import importlib.util, json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec_ = importlib.util.spec_from_file_location("kws", os.path.join(os.path.dirname(HERE), "jobs", "kws.py"))
kws = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(kws)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "models", "placeholder", "kws_random")
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
model = kws.DSCNN(64).eval()
# Default BatchNorm stats (mean 0, var 1) leave a random net almost input-independent (bias-dominated
# logits), which makes a useless board-vs-host check. Give each BN plausible random statistics so
# activations keep a ~unit scale and the logits actually depend on the input.
with torch.no_grad():
    x = torch.from_numpy(np.random.default_rng(2).uniform(-30, 10, (64, 1, kws.NFRAMES, kws.NMFCC)).astype(np.float32))
    for mod in model.body:
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.copy_(x.mean((0, 2, 3))); mod.running_var.copy_(x.var((0, 2, 3)) + 1e-3)
            mod.weight.uniform_(0.5, 1.5); mod.bias.uniform_(-0.2, 0.2)
        x = mod(x)
    model.fc.weight.mul_(4.0)
torch.save(model.state_dict(), f"{out}/student.pt")

# Calibration: MFCC-shaped floats in a realistic range. Real MLEK MFCC: c0 roughly [-30, 10],
# higher coefficients a few units around 0. Uniform [-30, 10] covers the int8 range generously.
rng = np.random.default_rng(1)
mfcc = rng.uniform(-30.0, 10.0, size=(64, 1, kws.NFRAMES, kws.NMFCC)).astype(np.float32)
np.savez_compressed(f"{out}/calib.npz", mfcc=mfcc, labels=rng.integers(0, len(kws.LABELS), 64))
open(f"{out}/labels.txt", "w").write("\n".join(kws.LABELS) + "\n")
json.dump({"name": "placeholder_random", "placeholder": True, "arch": {"ch": 64, "blocks": 4},
           "params": sum(p.numel() for p in model.parameters()), "val_acc": None, "test_acc": None},
          open(f"{out}/metrics.json", "w"), indent=2)
print("wrote", out)
