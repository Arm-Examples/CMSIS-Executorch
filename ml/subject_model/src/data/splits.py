"""Combined-participant protocol; preserves the original subject-1 pipeline."""
from dataclasses import asdict
from itertools import combinations

import numpy as np
from scipy.signal import butter
import torch

from src.data.download import configure_cache
from src.data.preprocess import EEG_CHANNELS, LABELS, Split, extract_session


EXPECTED = {'train': 1728, 'validation': 288, 'demonstration': 288, 'test': 288}


def split_for(subject, session):
    if subject in range(1, 6) or (subject in (6, 7) and session == '0train'):
        return 'train'
    if subject == 8:
        return 'validation'
    if subject in (6, 7) and session == '1test':
        return 'demonstration'
    if subject == 9:
        return 'test'
    raise ValueError(f'Unassigned recording: {subject}/{session}')


def validate_splits(groups, expected=None):
    identities, recordings = {}, {}
    for name, group in groups.items():
        if group.x.shape != (len(group.y), 1, 22, 500):
            raise ValueError(f'Invalid shape in {name}')
        if group.x.dtype != torch.float32 or group.y.dtype != torch.int64:
            raise ValueError(f'Invalid dtype in {name}')
        if not torch.isfinite(group.x).all() or set(group.y.tolist()) != {0, 1}:
            raise ValueError(f'Invalid data or labels in {name}')
        if len(group.trials) != len(group.y):
            raise ValueError('Missing trial metadata')
        identities[name] = {t.id for t in group.trials}
        recordings[name] = {(t.subject, t.session) for t in group.trials}
        if len(identities[name]) != len(group.trials):
            raise ValueError(f'Duplicate trials in {name}')
        for trial in group.trials:
            if split_for(trial.subject, trial.session) != name:
                raise ValueError(f'Incorrect participant/session allocation in {name}')
        by_run = {}
        for trial in group.trials:
            by_run.setdefault((trial.subject, trial.session, trial.run), []).append(trial)
        for trials in by_run.values():
            trials.sort(key=lambda t: t.cue_sample)
            if any(a.stop > b.cue_sample - 500 for a, b in zip(trials, trials[1:])):
                raise ValueError('Overlapping trial/filter histories')
        if expected is not None:
            if len(group.y) != expected[name]:
                raise ValueError(f'Unexpected trial count in {name}: {len(group.y)}')
            if torch.bincount(group.y, minlength=2).tolist() != [expected[name] // 2] * 2:
                raise ValueError(f'Unexpected class balance in {name}')
    for a, b in combinations(groups, 2):
        if identities[a] & identities[b] or recordings[a] & recordings[b]:
            raise ValueError(f'Leakage between {a} and {b}')


def normalize_groups(extracted):
    """Fit statistics exclusively on training windows; no held-out adaptation."""
    train_x = extracted['train'][0]
    mean = train_x.mean(axis=(0, 3), keepdims=True)
    std = train_x.std(axis=(0, 3), keepdims=True)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid normalization statistics')
    groups = {}
    for name, (x, y, trials) in extracted.items():
        groups[name] = Split(torch.from_numpy(((x - mean) / std).astype(np.float32)),
                             torch.from_numpy(y.astype(np.int64)), tuple(trials))
    return groups, mean, std


def prepare_combined():
    """Load cached subjects one at a time; retain all source-flagged trials."""
    cache = configure_cache()
    from moabb.datasets import BNCI2014_001
    dataset = BNCI2014_001(return_all_modalities=True, artifact_handling='annotate')
    sos = butter(4, [7, 30], btype='bandpass', fs=250, output='sos')
    parts = {name: [] for name in EXPECTED}
    audit = {}
    for subject in range(1, 10):
        for suffix in ('T', 'E'):
            if not list(cache.rglob(f'A{subject:02d}{suffix}.mat')):
                raise FileNotFoundError('Missing recording; run python -m src.data.download --subjects all')
        sessions = dataset.get_data(subjects=[subject], cache_config={'use': False})[subject]
        if set(sessions) != {'0train', '1test'}:
            raise ValueError(f'Unexpected sessions for subject {subject}')
        for session, runs in sessions.items():
            x, y, trials, flags = extract_session(runs, session, sos, subject=subject)
            parts[split_for(subject, session)].append((x, y, trials))
            audit[f'{subject}/{session}'] = flags
        del sessions
        print(f'Preprocessed subject {subject}', flush=True)
    extracted = {
        name: (np.concatenate([p[0] for p in values]),
               np.concatenate([p[1] for p in values]),
               tuple(t for p in values for t in p[2]))
        for name, values in parts.items()
    }
    groups, mean, std = normalize_groups(extracted)
    validate_splits(groups, EXPECTED)
    params = {
        'fs': 250, 'window_cue_seconds': [1, 3], 'window_samples': 500,
        'channel_order': list(EEG_CHANNELS), 'labels': LABELS,
        'input_units': 'volts', 'normalization': 'training-only per-channel population mean/std',
        'mean': mean.reshape(22).tolist(), 'std': std.reshape(22).tolist(),
        'bandpass_hz': [7, 30], 'butterworth_prototype_order': 4, 'sos': sos.tolist(),
        'filter_state': 'zero state at trial start (cue minus 2 seconds); 3 seconds warm-up',
        'artifact_policy': 'retain all trials; track source flags',
        'dtype': 'float32', 'quantization': None,
    }
    manifest = {
        name: {'shape': list(group.x.shape), 'class_counts': torch.bincount(group.y).tolist(),
               'flagged_trials': sum(t.artifact for t in group.trials),
               'recordings': [list(r) for r in sorted({(t.subject, t.session) for t in group.trials})],
               'trials': [asdict(t) for t in group.trials]}
        for name, group in groups.items()
    }
    return groups, params, {'splits': manifest, 'source_artifact_counts': audit}
