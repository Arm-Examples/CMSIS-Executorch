"""Regression checks for participant-aware regularization-dropout comparison."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tinyeegnet-matplotlib')
import unittest
import numpy as np
from src.data.preprocess import Trial
from src.experiments.dropout_cv import FOLDS, allowed_recordings, make_fold, decide, state_digest
from src.models.tinyeegnet import TinyEEGNet
from src.train import seed_everything


class DropoutCVTests(unittest.TestCase):
    def test_fold_partition_and_excluded_recordings(self):
        self.assertEqual(sorted(s for fold in FOLDS for s in fold), list(range(1, 8)))
        allowed = allowed_recordings(range(1, 8))
        self.assertEqual(len(allowed), 12)
        self.assertNotIn((6, '1test'), allowed)
        self.assertNotIn((7, '1test'), allowed)
        self.assertFalse(any(s in (8, 9) for s, _ in allowed))
        for fold in FOLDS:
            nval = sum(288 if s <= 5 else 144 for s in fold)
            self.assertEqual(nval, 576)

    def test_fold_normalization_and_participant_separation(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(14, 1, 22, 500))
        y = np.tile([0, 1], 7)
        trials = tuple(Trial(s, '0train', '0', 500 + j*2000,
                             750 + j*2000, 1250 + j*2000, False)
                       for s in range(1, 8) for j in range(2))
        for fold in FOLDS:
            result, mean, std = make_fold((x, y, trials), fold)
            changed = x.copy()
            changed[np.array([t.subject in fold for t in trials])] *= 100
            other, m2, s2 = make_fold((changed, y, trials), fold)
            np.testing.assert_array_equal(mean, m2)
            np.testing.assert_array_equal(std, s2)
            np.testing.assert_array_equal(result['train']['x'], other['train']['x'])
            np.testing.assert_allclose(result['train']['x'].numpy().mean(axis=(0, 3)), 0, atol=1e-6)
            self.assertEqual({t['subject'] for t in result['validation']['trials']}, set(fold))
            self.assertFalse({t['subject'] for t in result['train']['trials']} & set(fold))

    def test_matched_initial_weights(self):
        hashes = []
        for dropout in (0.5, 0.25):
            seed_everything(42, 2)
            model = TinyEEGNet(dropout=dropout)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 1938)
            hashes.append(state_digest(model))
        self.assertEqual(hashes[0], hashes[1])

    def test_selection_is_participant_weighted_and_conservative(self):
        reports = []
        for dropout in (0.5, 0.25):
            for fold in FOLDS:
                values = {str(s): {'balanced_accuracy': 0.4 + s / 20 + (0.03 if dropout == 0.25 else 0)} for s in fold}
                reports.append({'dropout': dropout, 'participants': values,
                                'macro_participant_balanced_accuracy': np.mean([v['balanced_accuracy'] for v in values.values()])})
        selected = decide(reports)
        self.assertTrue(selected['advance_025'])
        # Fold 3 has three participants; averaging fold means would be wrong.
        self.assertAlmostEqual(selected['configurations']['0.5']['mean'], 0.6)
        for r in reports:
            if r['dropout'] == 0.25:
                for v in r['participants'].values():v['balanced_accuracy'] = 0.61
        self.assertFalse(decide(reports)['advance_025'])


if __name__ == '__main__':
    unittest.main()
