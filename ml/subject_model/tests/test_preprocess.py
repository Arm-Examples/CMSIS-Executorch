"""Offline checks; synthetic recordings require no dataset/network access."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tinyeegnet-matplotlib')

import unittest
import mne
import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz
import torch

from src.data.preprocess import (
    EEG_CHANNELS, extract_session, filter_trial, prepare_data,
)


def recordings():
    sessions = {}
    for session in ('0train', '1test'):
        rng = np.random.default_rng(12 if session == '0train' else 34)
        n_trials, spacing = 40, 2000
        samples = n_trials * spacing
        info = mne.create_info(list(EEG_CHANNELS) + ['EOG1', 'STI'], 250,
                               ['eeg'] * 22 + ['eog', 'stim'])
        data = rng.normal(scale=1e-5, size=(24, samples))
        # Large auxiliary signals must never enter tensors.
        data[22:] = 1e9
        raw = mne.io.RawArray(data, info, verbose=False)
        onsets = (np.arange(n_trials) * spacing + 500) / 250
        names = np.tile(['left_hand', 'right_hand', 'feet', 'tongue'], 10)
        annotations = mne.Annotations(onsets, 4, names)
        annotations += mne.Annotations([onsets[0], onsets[1]], 4, ['bnci_artifact'] * 2)
        raw.set_annotations(annotations)
        sessions[session] = {'0': raw}
    return sessions


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sessions = recordings()
        cls.prepared = prepare_data(cls.sessions, seed=42)

    def test_shapes_labels_balance_and_artifacts(self):
        for name, count in [('train', 16), ('validation', 4), ('test', 20)]:
            split = getattr(self.prepared, name)
            self.assertEqual(tuple(split.x.shape), (count, 1, 22, 500))
            self.assertEqual(split.x.dtype, torch.float32)
            self.assertEqual(split.y.dtype, torch.int64)
            self.assertEqual(torch.bincount(split.y).tolist(), [count // 2] * 2)
            self.assertTrue(torch.isfinite(split.x).all())
        for audit in self.prepared.artifact_report.values():
            self.assertEqual(audit['left_hand_flags'], 1)
            self.assertEqual(audit['right_hand_flags'], 1)
        self.assertEqual(sum(t.artifact for t in self.prepared.test.trials), 2)

    def test_split_identities_and_source_bounds(self):
        groups = [self.prepared.train, self.prepared.validation, self.prepared.test]
        ids = [{trial.id for trial in split.trials} for split in groups]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(len(set.union(*ids)), 40)
        for split in groups[:2]:
            self.assertEqual({t.session for t in split.trials}, {'0train'})
        self.assertEqual({t.session for t in groups[2].trials}, {'1test'})
        for a in groups[0].trials:
            for b in groups[1].trials:
                self.assertTrue(a.stop <= b.cue_sample - 500 or b.stop <= a.cue_sample - 500)

    def test_training_normalization(self):
        x = self.prepared.train.x.numpy()
        np.testing.assert_allclose(x.mean(axis=(0, 3)), 0, atol=1e-6)
        np.testing.assert_allclose(x.std(axis=(0, 3)), 1, atol=1e-6)

    def test_held_out_signals_cannot_change_training_statistics(self):
        changed = {s: {r: raw.copy() for r, raw in runs.items()}
                   for s, runs in self.sessions.items()}
        for trial in self.prepared.validation.trials:
            raw = changed[trial.session][trial.run]
            start = trial.cue_sample - 500
            raw._data[:22, start:trial.stop] *= 30
        changed['1test']['0']._data[:22] *= 100
        other = prepare_data(changed, seed=42)
        np.testing.assert_array_equal(self.prepared.mean, other.mean)
        np.testing.assert_array_equal(self.prepared.std, other.std)
        torch.testing.assert_close(self.prepared.train.x, other.train.x, rtol=0, atol=0)
        self.assertFalse(torch.equal(self.prepared.test.x, other.test.x))

    def test_cue_offset_crop_and_mapping(self):
        sos = self.prepared.sos
        x, y, trials, _ = extract_session(self.sessions['0train'], '0train', sos)
        first = trials[0]
        self.assertEqual((first.cue_sample, first.start, first.stop), (500, 750, 1250))
        signal = self.sessions['0train']['0'].get_data(picks=list(EEG_CHANNELS), start=0, stop=1250)
        np.testing.assert_allclose(x[0, 0], sosfilt(sos, signal, axis=-1)[:, 750:1250])
        self.assertEqual(y[:2].tolist(), [0, 1])

    def test_filter_response_and_scipy_reference(self):
        sos = butter(4, [7, 30], btype='bandpass', fs=250, output='sos')
        impulse = np.zeros((22, 2000))
        impulse[:, 0] = 1
        np.testing.assert_array_equal(filter_trial(impulse, sos), sosfilt(sos, impulse, axis=-1))
        _, response = sosfreqz(sos, worN=[2, 15, 80], fs=250)
        self.assertGreater(abs(response[1]), 0.95)
        self.assertLess(abs(response[0]), 0.02)
        self.assertLess(abs(response[2]), 0.02)

    def test_deterministic_splits(self):
        other = prepare_data(self.sessions, seed=42)
        self.assertEqual(self.prepared.train.trials, other.train.trials)
        torch.testing.assert_close(self.prepared.train.x, other.train.x)

    def test_wrong_sampling_frequency_is_rejected(self):
        bad = recordings()
        bad['0train']['0'] = mne.io.RawArray(
            np.zeros((22, 1000)), mne.create_info(list(EEG_CHANNELS), 200, 'eeg'), verbose=False
        )
        with self.assertRaisesRegex(ValueError, '250 Hz'):
            prepare_data(bad, seed=42)


if __name__ == '__main__':
    unittest.main()
