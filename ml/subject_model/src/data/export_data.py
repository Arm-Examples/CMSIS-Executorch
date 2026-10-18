"""Prepare train-only calibration and Subject-8 development inputs for export."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.experiments.dropout_cv import allowed_recordings, load_allowed
from src.models.tinyeegnet import TinyEEGNet


def main():
    baseline = Path('artifacts/combined-seed42-20261016')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/ethosu85-export-seed42-20261016'))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    old = [p for root in (Path('artifacts'), Path('results')) for p in root.rglob('*')
           if p.is_file() and not p.is_relative_to(out)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in old}
    (out / 'preserved_hashes.json').write_text(json.dumps(hashes, indent=2))
    params = json.loads((baseline / 'preprocess_params.json').read_text())
    mean = np.array(params['mean']).reshape(1, 1, 22, 1)
    std = np.array(params['std']).reshape(1, 1, 22, 1)
    raw, sources, _ = load_allowed(allowed_recordings(range(1, 8)))
    x, y, trials = raw
    np.testing.assert_allclose(x.mean(axis=(0, 3)), mean.reshape(1, 22), rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(x.std(axis=(0, 3)), std.reshape(1, 22), rtol=1e-10, atol=1e-18)
    rng = np.random.default_rng(42)
    strata = sorted({(t.subject, t.session, int(label)) for t, label in zip(trials, y)})
    indices = []
    for index, key in enumerate(strata):
        choices = [i for i, (t, label) in enumerate(zip(trials, y))
                   if (t.subject, t.session, int(label)) == key]
        indices.extend(rng.choice(choices, size=8 + int(index < 8), replace=False).tolist())
    if len(indices) != 200 or np.bincount(y[indices]).tolist() != [100, 100]:
        raise ValueError('Invalid calibration sample allocation')
    calibration = torch.from_numpy(((x[indices] - mean) / std).astype(np.float32))
    development, dev_sources, _ = load_allowed(allowed_recordings([8]))
    dx, dy, dt = development
    dev_x = torch.from_numpy(((dx - mean) / std).astype(np.float32))
    torch.set_num_threads(4)
    model = TinyEEGNet().eval()
    model.load_state_dict(torch.load(baseline / 'best_checkpoint.pt', weights_only=True)['model_state'])
    with torch.inference_mode():
        logits = torch.cat([model(v[None]) for v in dev_x])
    torch.save({'calibration': calibration, 'calibration_y': torch.tensor(y[indices]),
                'development_x': dev_x, 'development_y': torch.tensor(dy),
                'reference_logits': logits}, out / 'export_data.pt')
    metadata = {'seed': 42, 'calibration_count': 200, 'calibration_class_counts': [100, 100],
                'calibration_trials': [asdict(trials[i]) for i in indices],
                'development_trials': [asdict(t) for t in dt],
                'sources': sources + dev_sources, 'forbidden_sources_used': [],
                'checkpoint': str(baseline / 'best_checkpoint.pt'),
                'checkpoint_sha256': hashlib.sha256((baseline / 'best_checkpoint.pt').read_bytes()).hexdigest()}
    (out / 'data_manifest.json').write_text(json.dumps(metadata, indent=2))
    (out / 'float_preprocess_params.json').write_text(json.dumps(params, indent=2))
    print('Prepared 200 balanced calibration samples from 12 training recordings;')
    print('development=Subject 8 only, 288 trials; float accuracy=',
          (logits.argmax(1) == torch.tensor(dy)).float().mean().item())


if __name__ == '__main__':
    main()
