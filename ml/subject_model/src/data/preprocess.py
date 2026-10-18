"""Subject-1 preprocessing; run with python -m src.data.preprocess --seed 42.

Uses cue-aligned MOABB annotations, not the trial-start STI pulses. Each trial
is filtered independently with a causal Butterworth SOS filter, zero initial
state, and three seconds of history (trial start to window start). This avoids
filter state carrying samples between train/validation trials. No samples are
rejected based on artifact flags; the flags are returned for policy review.
Training tensors are float32; int8 quantization belongs to the later export.
"""

import argparse
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.model_selection import train_test_split
import torch


EEG_CHANNELS = (
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
    'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz',
)
LABELS = {'left_hand': 0, 'right_hand': 1}


@dataclass(frozen=True)
class Trial:
    subject: int
    session: str
    run: str
    cue_sample: int
    start: int
    stop: int
    artifact: bool

    @property
    def id(self):
        return self.subject, self.session, self.run, self.cue_sample


@dataclass
class Split:
    x: torch.Tensor
    y: torch.Tensor
    trials: tuple[Trial, ...]


@dataclass
class PreparedData:
    train: Split
    validation: Split
    test: Split
    mean: np.ndarray
    std: np.ndarray
    sos: np.ndarray
    artifact_report: dict
    seed: int


def filter_trial(samples, sos):
    """Filter one trial's EEG independently along time, starting from zero state."""
    return sosfilt(sos, samples, axis=-1)


def extract_session(runs, session, sos, subject=1):
    """Return windows, binary labels, identity/bounds, and source artifact counts."""
    windows, labels, trials = [], [], []
    audit = Counter()
    for run, raw in runs.items():
        if raw.info['sfreq'] != 250:
            raise ValueError('Expected 250 Hz; resampling is not enabled')
        actual = tuple(name for name, kind in zip(raw.ch_names, raw.get_channel_types())
                       if kind == 'eeg')
        if actual != EEG_CHANNELS:
            raise ValueError('Unexpected EEG channel set/order')
        annotations = list(raw.annotations)
        flags = [a for a in annotations
                 if a['description'] in ('bnci_artifact', 'BAD_artifact')]
        audit['all_class_flags'] += len(flags)
        # MNE annotations include first_time; array indices do not.
        for annotation in annotations:
            name = annotation['description']
            if name not in LABELS:
                continue
            cue_time = annotation['onset'] - raw.first_time
            cue = int(round(cue_time * 250))
            start, stop = cue + 250, cue + 750  # half-open [cue+1, cue+3)
            history = cue - 500  # trial start, two seconds before cue
            if history < 0 or stop > raw.n_times:
                raise ValueError('Incomplete trial history or window')
            # Source artifact annotations cover the imagery interval; compare
            # overlap with the entire four-second trial imagery, not just crop.
            artifact = any(
                a['onset'] < annotation['onset'] + 4
                and a['onset'] + a['duration'] > annotation['onset']
                for a in flags
            )
            audit[f'{name}_trials'] += 1
            audit[f'{name}_flags'] += int(artifact)
            segment = raw.get_data(picks=list(EEG_CHANNELS), start=history, stop=stop)
            if not np.isfinite(segment).all():
                raise ValueError('Non-finite EEG samples; no silent rejection allowed')
            window = filter_trial(segment, sos)[:, start - history:stop - history]
            if window.shape != (22, 500):
                raise ValueError('Expected exactly 500 samples per window')
            windows.append(window)
            labels.append(LABELS[name])
            trials.append(Trial(subject, session, str(run), cue, start, stop, artifact))
    if not windows:
        raise ValueError('No left/right trials found')
    return np.stack(windows)[:, None], np.asarray(labels), tuple(trials), dict(audit)


def prepare_data(sessions, *, seed):
    """Stratify session 1, then fit channel statistics on training windows only."""
    if set(sessions) != {'0train', '1test'}:
        raise ValueError('Expected separate 0train and 1test sessions')
    sos = butter(4, [7, 30], btype='bandpass', fs=250, output='sos')
    x, y, trials, train_audit = extract_session(sessions['0train'], '0train', sos)
    tx, ty, test_trials, test_audit = extract_session(sessions['1test'], '1test', sos)
    ids = [trial.id for trial in trials + test_trials]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate trial identities')
    # Reject overlapping trial histories, as filtering uses samples before the
    # retained window too. Check within each run; runs have independent clocks.
    for session_trials in (trials, test_trials):
        for previous, current in zip(session_trials, session_trials[1:]):
            if previous.run == current.run and previous.stop > current.cue_sample - 500:
                raise ValueError('Overlapping trial histories')
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed
    )
    mean = x[train_idx].mean(axis=(0, 3), keepdims=True)
    std = x[train_idx].std(axis=(0, 3), keepdims=True)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid training channel standard deviation')

    def make_split(values, targets, metadata):
        normalized = (values - mean) / std
        if not np.isfinite(normalized).all():
            raise ValueError('Non-finite normalized samples')
        return Split(torch.from_numpy(normalized.astype(np.float32)),
                     torch.from_numpy(targets.astype(np.int64)), tuple(metadata))

    return PreparedData(
        make_split(x[train_idx], y[train_idx], [trials[i] for i in train_idx]),
        make_split(x[val_idx], y[val_idx], [trials[i] for i in val_idx]),
        make_split(tx, ty, test_trials), mean, std, sos,
        {'0train': train_audit, '1test': test_audit}, seed,
    )


def load_subject_one():
    """Load the existing local cache; refuse to initiate dataset downloads."""
    root = Path(__file__).resolve().parents[2]
    cache = root / 'mne_data'
    for name in ('A01T.mat', 'A01E.mat'):
        if not list(cache.rglob(name)):
            raise FileNotFoundError(f'{name} missing; run src.data.inspect_dataset first')
    os.environ['MNE_DATA'] = str(cache)
    os.environ['MNE_DATASETS_BNCI_PATH'] = str(cache)
    os.environ['MOABB_DOWNLOAD_PROVIDER'] = 'upstream'
    os.environ.setdefault('MPLCONFIGDIR', str(cache / '.matplotlib'))
    from moabb.datasets import BNCI2014_001
    dataset = BNCI2014_001(return_all_modalities=True, artifact_handling='annotate')
    return dataset.get_data(subjects=[1], cache_config={'use': False})[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # No default seed is silently introduced in the absence of default.yaml.
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    result = prepare_data(load_subject_one(), seed=args.seed)
    print(f'Seed: {result.seed}; cue-relative window: [1, 3) seconds')
    print('Filter: Butterworth N=4 bandpass (8th-order), 7-30 Hz, causal SOS;')
    print('zero state per trial; 3 seconds warm-up before retained window.')
    for name in ('train', 'validation', 'test'):
        split = getattr(result, name)
        print(f'{name}: X={tuple(split.x.shape)} {split.x.dtype}, '
              f'y={tuple(split.y.shape)} {split.y.dtype}, '
              f'left/right={torch.bincount(split.y, minlength=2).tolist()}, '
              f'flagged={sum(t.artifact for t in split.trials)}')
    print('Artifact audit (no exclusions):', result.artifact_report)
    print('Policy recommendation: retain all flagged trials in this initial baseline;')
    print('report flagged/clean performance separately later. Any exclusion policy')
    print('must be fixed before training and applied consistently to both sessions.')
    print('Normalization: per-channel training-window mean/population std only.')
    print('Tensors remain in memory; no files exported and no model trained.')


if __name__ == '__main__':
    main()
