"""Paired regularization-dropout CV; never loads demonstration or S9 recordings.

Uses MOABB's MAT converter directly to restrict IO to explicitly allowed files.
This is not the proposed sensor-dropout experiment. All preprocessing/model and
training operations reuse the baseline implementation without changing it.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.signal import butter
import torch

from src.data.download import configure_cache
from src.data.preprocess import EEG_CHANNELS, LABELS, Split, Trial, extract_session
from src.eval import evaluate
from src.models.tinyeegnet import TinyEEGNet
from src.train import export_check, save_json, seed_everything, train

FOLDS = ((1, 4), (2, 5), (3, 6, 7))
DROPOUTS = (0.5, 0.25)
BASELINE = Path('artifacts/combined-seed42-20261016')
RULE = {'minimum_mean_gain': 0.02, 'minimum_participants_improved': 4,
        'minimum_folds_improved': 2, 'maximum_participant_decline': 0.05}


def allowed_recordings(subjects):
    return [(s, session) for s in subjects for session in
            (('0train', '1test') if s <= 5 or s == 8 else ('0train',))]


def load_allowed(recordings):
    """No get_data(): it would load both sessions before session selection."""
    from moabb.datasets.bnci.base import _convert_mi
    from moabb.datasets.preprocessing import SetRawAnnotations
    cache = configure_cache()
    pieces, provenance, audit = [], [], {}
    event_id = {'left_hand': 1, 'right_hand': 2, 'feet': 3, 'tongue': 4}
    sos = butter(4, [7, 30], btype='bandpass', fs=250, output='sos')
    for subject, session in recordings:
        if (subject, session) not in allowed_recordings(range(1, 9)):
            raise ValueError('Forbidden subject/session requested')
        suffix = 'T' if session == '0train' else 'E'
        paths = list(cache.rglob(f'A{subject:02d}{suffix}.mat'))
        if len(paths) != 1:
            raise ValueError(f'Expected one cached source recording: {subject}/{session}')
        path = paths[0]
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        provenance.append({'subject': subject, 'session': session,
                           'path': str(path.relative_to(cache)), 'sha256': digest})
        runs, codes = _convert_mi(
            str(path), list(EEG_CHANNELS) + ['EOG1', 'EOG2', 'EOG3'],
            ['eeg'] * 22 + ['eog'] * 3, artifact_handling='annotate',
            artifact_interval=(2, 6),
        )
        if codes != event_id:
            raise ValueError(f'Unexpected source class mapping: {codes}')
        marker = SetRawAnnotations(event_id, interval=[2, 6])
        runs = {str(i): marker.transform(raw) for i, raw in enumerate(runs)}
        x, y, trials, flags = extract_session(runs, session, sos, subject=subject)
        pieces.append((x, y, trials))
        audit[f'{subject}/{session}'] = flags
        print(f'Loaded allowed recording {subject}/{session}: {len(y)} trials', flush=True)
    return (np.concatenate([p[0] for p in pieces]), np.concatenate([p[1] for p in pieces]),
            tuple(t for p in pieces for t in p[2])), provenance, audit


def make_fold(raw, held_out):
    x, y, trials = raw
    subjects = np.array([t.subject for t in trials])
    validation = np.isin(subjects, held_out)
    training = ~validation
    if not training.any() or not validation.any():
        raise ValueError('Empty fold partition')
    mean = x[training].mean(axis=(0, 3), keepdims=True)
    std = x[training].std(axis=(0, 3), keepdims=True)
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid fold training statistics')
    result = {}
    for name, mask in [('train', training), ('validation', validation)]:
        tx = ((x[mask] - mean) / std).astype(np.float32)
        if not np.isfinite(tx).all():
            raise ValueError('Non-finite fold input')
        result[name] = {'x': torch.from_numpy(tx), 'y': torch.tensor(y[mask], dtype=torch.int64),
                        'trials': [asdict(t) for t, keep in zip(trials, mask) if keep]}
    tr = {t['subject'] for t in result['train']['trials']}
    va = {t['subject'] for t in result['validation']['trials']}
    if tr & va or va != set(held_out):
        raise ValueError('Participant leakage or missing held-out participant')
    return result, mean, std


def as_split(item):
    return Split(item['x'], item['y'], tuple(Trial(**t) for t in item['trials']))


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def verify_model(model, directory, sample):
    model.eval()
    other = TinyEEGNet(dropout=model.features[7].p).eval()
    state = torch.load(directory / 'best_checkpoint.pt', weights_only=True)['model_state']
    other.load_state_dict(state)
    with torch.inference_mode():
        for batch in (1, 4):
            output = model(torch.zeros(batch, 1, 22, 500))
            if output.shape != (batch, 2) or not torch.isfinite(output).all():
                raise ValueError('Invalid output contract')
        torch.testing.assert_close(model(sample[:4]), other(sample[:4]), rtol=0, atol=0)
    exported = export_check(model, directory / 'tinyeegnet_float.pt2')
    if exported['passed']:
        program = torch.export.load(directory / 'tinyeegnet_float.pt2').module()
        with torch.inference_mode():
            torch.testing.assert_close(model(sample[:1]), program(sample[:1]))
    return {'reload_exact_match': True, 'batch_shapes': [[1, 2], [4, 2]],
            'parameters': sum(p.numel() for p in model.parameters()), 'torch_export': exported}


def worker(inputs, directory, dropout, config_path):
    config = json.loads(config_path.read_text())
    config['model']['dropout'] = dropout
    seed_everything(config['seed'], config['training']['threads'])
    content = torch.load(inputs, weights_only=True)
    training, validation = as_split(content['train']), as_split(content['validation'])
    model = TinyEEGNet(dropout=dropout)
    initial_digest = state_digest(model)
    save_json(directory / 'config.json', config)
    started = time.monotonic()
    history = train(model, training, validation, config, directory)
    duration = time.monotonic() - started
    checkpoint = torch.load(directory / 'best_checkpoint.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state']); model.eval()
    torch.save(model.state_dict(), directory / 'tinyeegnet_float.pt')
    per_subject = {}
    for subject in sorted({t.subject for t in validation.trials}):
        idx = [i for i, t in enumerate(validation.trials) if t.subject == subject]
        part = Split(validation.x[idx], validation.y[idx], tuple(validation.trials[i] for i in idx))
        per_subject[str(subject)] = evaluate(model, part)
    # Training and validation summary metrics both use eval mode.
    report = {'dropout': dropout, 'best_epoch': checkpoint['epoch'], 'epochs_run': len(history),
              'training_seconds': duration, 'initial_state_sha256': initial_digest,
              'training': evaluate(model, training), 'validation': evaluate(model, validation),
              'participants': per_subject,
              'macro_participant_balanced_accuracy': float(np.mean([v['balanced_accuracy'] for v in per_subject.values()])),
              'verification': verify_model(model, directory, validation.x)}
    save_json(directory / 'metrics.json', report)
    print('COMPLETED', json.dumps(report), flush=True)


def decide(reports):
    by_dropout = {}
    fold_means = {}
    for dropout in DROPOUTS:
        selected = [r for r in reports if r['dropout'] == dropout]
        scores = {int(s): m['balanced_accuracy'] for r in selected for s, m in r['participants'].items()}
        if set(scores) != set(range(1, 8)) or len(selected) != 3:
            raise ValueError('Incomplete cross-validation results')
        values = [scores[s] for s in range(1, 8)]
        by_dropout[str(dropout)] = {'participants': scores, 'mean': float(np.mean(values)),
                                    'population_std': float(np.std(values)),
                                    'range': [float(min(values)), float(max(values))]}
        fold_means[dropout] = [r['macro_participant_balanced_accuracy'] for r in selected]
    gains = [by_dropout['0.25']['participants'][s] - by_dropout['0.5']['participants'][s] for s in range(1, 8)]
    mean_gain = float(np.mean(gains))
    improved = sum(g > 1e-12 for g in gains)
    fold_gains = np.array(fold_means[0.25]) - np.array(fold_means[0.5])
    checks = {'mean_gain': mean_gain >= RULE['minimum_mean_gain'],
              'participants_improved': improved >= RULE['minimum_participants_improved'],
              'folds_improved': int(sum(fold_gains > 1e-12)) >= RULE['minimum_folds_improved'],
              'maximum_decline': min(gains) >= -RULE['maximum_participant_decline']}
    return {'configurations': by_dropout, 'participant_gains': gains,
            'mean_gain': mean_gain, 'participants_improved': improved,
            'fold_gains': fold_gains.tolist(), 'checks': checks,
            'advance_025': all(checks.values()), 'rule': RULE}


def snapshot_baseline():
    paths = list(BASELINE.rglob('*')) + list((Path('results') / BASELINE.name).rglob('*'))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def plot_curves(reports, artifacts, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout='constrained')
    for report in reports:
        fold = report['fold']; dropout = report['dropout']
        history = json.loads((artifacts / f'fold{fold}' / f'dropout{dropout}' / 'history.json').read_text())
        epochs = [r['epoch'] for r in history]
        color = 'tab:blue' if dropout == 0.5 else 'tab:orange'
        for col, key in enumerate(('loss', 'accuracy')):
            ax = axes[fold - 1, col]
            ax.plot(epochs, [r[f'train_{key}'] for r in history], color=color, ls='--', label=f'p={dropout} train (dropout on)')
            ax.plot(epochs, [r['validation'][key] for r in history], color=color, label=f'p={dropout} validation')
            ax.axvline(report['best_epoch'], color=color, alpha=.25)
            ax.set(xlabel='Epoch', ylabel=key, title=f'Fold {fold}: {FOLDS[fold-1]}')
            ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(results / 'loss_accuracy_curves.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='dropout-cv-seed42-20261016')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.out, args.dropout, args.config)
        return
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('run-name must be one directory name')
    artifacts = Path('artifacts') / args.run_name
    results = Path('results') / args.run_name
    artifacts.mkdir(parents=True, exist_ok=False); results.mkdir(parents=True, exist_ok=False)
    baseline_hashes = snapshot_baseline()
    config = json.loads((BASELINE / 'config.json').read_text())
    if config['seed'] != 42:
        raise ValueError('Expected baseline seed 42')
    save_json(artifacts / 'base_config.json', config)
    protocol = {'folds': FOLDS, 'dropouts': DROPOUTS, 'rule': RULE, 'base_config': config,
                'early_stopping': 'pooled held-out-fold cross-entropy, patience 20; same as baseline',
                'comparison': 'unweighted mean balanced accuracy over seven participants',
                'interpretation': 'development CV; each validation fold also selects its checkpoint',
                'allowed_recordings': allowed_recordings(range(1, 8)),
                'excluded': 'S6/7 session 2, S9 entirely; S8 used only if final refit is authorized by rule',
                'baseline_hashes_before': baseline_hashes}
    save_json(results / 'protocol.json', protocol)
    raw, source_records, audit = load_allowed(allowed_recordings(range(1, 8)))
    if raw[0].shape != (1728, 1, 22, 500):
        raise ValueError('Unexpected development size')
    if len({t.id for t in raw[2]}) != 1728:
        raise ValueError('Duplicate development trials')
    save_json(artifacts / 'source_recordings.json', source_records)
    save_json(artifacts / 'artifact_audit.json', audit)
    reports = []
    for fold, held_out in enumerate(FOLDS, 1):
        folder = artifacts / f'fold{fold}'; folder.mkdir()
        inputs, mean, std = make_fold(raw, held_out)
        if inputs['train']['x'].shape != (1152, 1, 22, 500) or inputs['validation']['x'].shape != (576, 1, 22, 500):
            raise ValueError('Unexpected fold sizes')
        torch.save(inputs, folder / 'inputs.pt')
        save_json(folder / 'normalization.json', {'mean': mean.reshape(22).tolist(), 'std': std.reshape(22).tolist(),
                                                 'fit_subjects': sorted(set(range(1, 8)) - set(held_out))})
        save_json(folder / 'split_manifest.json', {name: item['trials'] for name, item in inputs.items()})
        processes = []
        for dropout in DROPOUTS:
            directory = folder / f'dropout{dropout}'; directory.mkdir()
            log = (directory / 'training.log').open('x')
            command = [sys.executable, '-u', '-m', 'src.experiments.dropout_cv',
                       '--worker', str(folder / 'inputs.pt'), '--out', str(directory),
                       '--dropout', str(dropout), '--config', str(artifacts / 'base_config.json')]
            print(f'START fold={fold} held_out={held_out} dropout={dropout}', flush=True)
            processes.append((dropout, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
        for dropout, process, log in processes:
            returncode = process.wait(); log.close()
            if returncode:
                raise RuntimeError(f'Worker failed: fold {fold} dropout {dropout}; inspect its training.log')
            report = json.loads((folder / f'dropout{dropout}' / 'metrics.json').read_text())
            report['fold'] = fold
            reports.append(report)
            print(f"FINISH fold={fold} dropout={dropout} best_epoch={report['best_epoch']} "
                  f"participant_mean={report['macro_participant_balanced_accuracy']:.4f}", flush=True)
        if reports[-1]['initial_state_sha256'] != reports[-2]['initial_state_sha256']:
            raise ValueError('Paired initial weights differ')
        save_json(results / 'completed_runs.json', reports)
    decision = decide(reports)
    save_json(results / 'selection.json', decision)
    plot_curves(reports, artifacts, results)
    print('SELECTION', json.dumps(decision), flush=True)
    if decision['advance_025']:
        # Only now access Subject 8, for the predefined final-fit early stopping.
        val_raw, provenance, flags = load_allowed(allowed_recordings([8]))
        train_x, train_y, train_trials = raw
        val_x, val_y, val_trials = val_raw
        mean = train_x.mean(axis=(0, 3), keepdims=True)
        std = train_x.std(axis=(0, 3), keepdims=True)
        content = {}
        for name, x, y, trials in [('train', train_x, train_y, train_trials),
                                    ('validation', val_x, val_y, val_trials)]:
            content[name] = {'x': torch.from_numpy(((x - mean) / std).astype(np.float32)),
                             'y': torch.tensor(y, dtype=torch.int64), 'trials': [asdict(t) for t in trials]}
        final = artifacts / 'final_candidate'; final.mkdir()
        torch.save(content, final / 'inputs.pt')
        params = json.loads((BASELINE / 'preprocess_params.json').read_text())
        np.testing.assert_allclose(mean.reshape(22), params['mean'], rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(std.reshape(22), params['std'], rtol=1e-12, atol=1e-18)
        params['mean'] = mean.reshape(22).tolist(); params['std'] = std.reshape(22).tolist()
        save_json(final / 'preprocess_params.json', params)
        save_json(final / 'label_mapping.json', LABELS)
        save_json(final / 'source_recordings.json', source_records + provenance)
        save_json(final / 'artifact_audit.json', {**audit, **flags})
        save_json(final / 'split_manifest.json', {name: item['trials'] for name, item in content.items()})
        print('START final candidate dropout=0.25; Subject 8 early stopping only', flush=True)
        worker(final / 'inputs.pt', final, 0.25, artifacts / 'base_config.json')
        final_result = json.loads((final / 'metrics.json').read_text())
        final_result['selection'] = 'dropout fixed by development CV, then Subject 8 selects final epoch'
        save_json(results / 'final_candidate.json', final_result)
    else:
        final_result = {'action': 'retain original baseline; dropout 0.25 failed predeclared advancement rule'}
        # Verify baseline reload/forward without running any held-out evaluation.
        original = TinyEEGNet().eval()
        original.load_state_dict(torch.load(BASELINE / 'best_checkpoint.pt', weights_only=True)['model_state'])
        with torch.inference_mode():
            assert original(torch.zeros(1, 1, 22, 500)).shape == (1, 2)
        save_json(results / 'retained_baseline.json', final_result)
    if snapshot_baseline() != baseline_hashes:
        raise RuntimeError('Original baseline files changed')
    save_json(results / 'verification.json', {'baseline_hashes_unchanged': True,
              'paired_initial_weights_match': True, 'forbidden_recordings_used': [],
              'all_checkpoint_reloads_and_exports': [r['verification'] for r in reports],
              'versions': {name: importlib.metadata.version(name) for name in ('torch','numpy','scipy','moabb','mne')}})
    print('DONE', results, flush=True)


if __name__ == '__main__':
    main()
