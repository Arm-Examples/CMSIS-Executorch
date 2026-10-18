"""Train one combined-participant TinyEEGNet, select on subject 8 only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.data.splits import prepare_combined
from src.data.download import recording_provenance
from src.eval import evaluate
from src.models.tinyeegnet import TinyEEGNet


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def export_check(model, path):
    example = torch.zeros(1, 1, 22, 500)
    model.eval()
    try:
        exported = torch.export.export(model, (example,), strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(exported.module()(example), model(example))
        torch.export.save(exported, path)
        loaded = torch.export.load(path)
        with torch.inference_mode():
            torch.testing.assert_close(loaded.module()(example), model(example))
        return {'passed': True, 'shape': [1, 2], 'strict': True,
                'note': 'Float torch.export only; no ExecuTorch/Arm lowering or quantization verified.'}
    except Exception as error:
        return {'passed': False, 'error': f'{type(error).__name__}: {error}'}


def train(model, training, validation, config, out):
    settings = config['training']
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(TensorDataset(training.x, training.y),
                        batch_size=settings['batch_size'], shuffle=True,
                        generator=generator, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    best, stale, history = float('inf'), 0, []
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        loss_sum, correct = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
        metrics = evaluate(model, validation, settings['batch_size'])
        record = {'epoch': epoch, 'train_loss': loss_sum / len(training.y),
                  'train_accuracy': correct / len(training.y), 'validation': metrics}
        history.append(record)
        improved = metrics['loss'] < best
        if improved:
            best, stale = metrics['loss'], 0
            torch.save({'model_state': model.state_dict(), 'epoch': epoch,
                        'validation': metrics, 'config': config}, out / 'best_checkpoint.pt')
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or stale >= settings['early_stop']:
            print(f"Epoch {epoch}: train loss={record['train_loss']:.4f}, "
                  f"val loss={metrics['loss']:.4f}, val accuracy={metrics['accuracy']:.3f}, "
                  f"patience={stale}/{settings['early_stop']}", flush=True)
        if stale >= settings['early_stop']:
            break
    save_json(out / 'history.json', history)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'))
    parser.add_argument('--run-name', default=datetime.now(timezone.utc).strftime('combined-%Y%m%dT%H%M%SZ'))
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('run-name must be one directory name')
    config = yaml.safe_load(args.config.read_text())
    if config['fs'] != 250 or config['window'] != [1.0, 3.0] or config['filter_hz'] != [7, 30]:
        raise ValueError('Configuration does not match the implemented preprocessing')
    fixed = {'protocol': 'combined-participant-v1', 'filter_prototype_order': 4,
             'artifact_policy': 'retain_all', 'normalization': 'training_only_channel_mean_std'}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError('Configuration does not match the implemented data protocol')
    required_training = {'optimizer': 'Adam', 'loss': 'CrossEntropyLoss',
                         'checkpoint_metric': 'validation_loss'}
    if any(config['training'].get(key) != value for key, value in required_training.items()):
        raise ValueError('Unsupported optimizer, loss, or checkpoint-selection setting')
    if config['model']['activation'] != 'relu' or config['training']['device'] != 'cpu':
        raise ValueError('This reproducible baseline requires ReLU and CPU')
    out = Path('artifacts') / args.run_name
    out.mkdir(parents=True, exist_ok=False)  # never overwrite a run
    results = Path('results') / args.run_name
    results.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    seed_everything(config['seed'], config['training']['threads'])
    groups, params, manifest = prepare_combined()
    save_json(out / 'preprocess_params.json', params)
    save_json(out / 'label_mapping.json', params['labels'])
    save_json(out / 'config.json', config)
    save_json(out / 'split_manifest.json', manifest)
    save_json(out / 'source_recordings.json', recording_provenance())
    # Save prepared data locally for exact reload/inference checks, not in Git.
    torch.save({name: {'x': group.x, 'y': group.y} for name, group in groups.items()},
               out / 'prepared_tensors.pt')
    for name, group in groups.items():
        print(f'{name}: {tuple(group.x.shape)}, labels={torch.bincount(group.y).tolist()}, '
              f'flags={sum(t.artifact for t in group.trials)}', flush=True)
    model = TinyEEGNet(dropout=config['model']['dropout'])
    parameter_count = sum(p.numel() for p in model.parameters())
    model.eval()
    with torch.inference_mode():
        for batch in (1, 4):
            assert model(torch.zeros(batch, 1, 22, 500)).shape == (batch, 2)
    print(f'Model parameters: {parameter_count}; forward shapes passed', flush=True)
    # Train receives only train/validation groups, never demonstration or test.
    history = train(model, groups['train'], groups['validation'], config, out)
    checkpoint = torch.load(out / 'best_checkpoint.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    reloaded = TinyEEGNet(dropout=config['model']['dropout']).eval()
    reloaded.load_state_dict(torch.load(out / 'best_checkpoint.pt', weights_only=True)['model_state'])
    with torch.inference_mode():
        torch.testing.assert_close(model(groups['validation'].x[:4]),
                                   reloaded(groups['validation'].x[:4]), rtol=0, atol=0)
    torch.save(model.state_dict(), out / 'tinyeegnet_float.pt')
    checkpoint_hash = hashlib.sha256((out / 'best_checkpoint.pt').read_bytes()).hexdigest()
    # The selected checkpoint is now immutable. First held-out evaluations occur here.
    final = {name: evaluate(model, groups[name])
             for name in ('validation', 'demonstration', 'test')}
    report = {
        'run': args.run_name, 'seed': config['seed'], 'config': config,
        'parameters': parameter_count, 'epochs_run': len(history),
        'best_epoch': checkpoint['epoch'], 'selection': 'subject 8 validation loss only',
        'checkpoint_sha256': checkpoint_hash, 'reload_exact_match': True,
        'metrics': final, 'torch_export': export_check(model, out / 'tinyeegnet_float.pt2'),
        'seconds': time.monotonic() - started,
        'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                     'numpy': np.__version__},
        'splits': {name: {k: v for k, v in split.items() if k != 'trials'}
                   for name, split in manifest['splits'].items()},
    }
    import importlib.metadata
    for name in ('scipy', 'scikit-learn', 'moabb', 'mne'):
        report['versions'][name] = importlib.metadata.version(name)
    save_json(results / 'metrics.json', report)
    save_json(out / 'metrics.json', report)
    print(json.dumps(report, indent=2), flush=True)
    print(f'Artifacts: {out}; results: {results}', flush=True)


if __name__ == '__main__':
    main()
