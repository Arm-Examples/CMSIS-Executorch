"""Acceptance tests for causal filtering, epoching, and the dataset contract."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.signal import sosfilt


def test_causal_filter_matches_scipy_reference(rng: np.random.Generator) -> None:
    from src.data.preprocess import causal_filter, design_filter

    x = rng.standard_normal((22, 4096))
    sos = design_filter(fs=250.0, low=7.0, high=30.0, order=4)

    assert isinstance(sos, np.ndarray)
    assert sos.shape == (4, 6)
    actual = causal_filter(x.copy(), sos)
    expected = sosfilt(sos, x, axis=-1)

    assert actual.shape == x.shape
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_epoch_uses_only_motor_imagery_events_and_exact_window() -> None:
    from src.data.preprocess import epoch

    # Channel/sample ramps make every boundary and ordering error observable.
    x = np.arange(22 * 2400, dtype=np.float64).reshape(22, 2400)
    events = np.array(
        [
            [100, 0, 768],
            [200, 0, 769],
            [900, 0, 771],
            [1200, 0, 770],
            [1800, 0, 1023],
        ],
        dtype=np.int64,
    )

    X, y = epoch(x, events, fs=250.0, tmin=0.5, tmax=2.5)

    assert X.shape == (2, 22, 500)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert_array_equal(y, np.array([0, 1], dtype=np.int64))
    assert_allclose(X[0], x[:, 325:825].astype(np.float32), rtol=0, atol=0)
    assert_allclose(X[1], x[:, 1325:1825].astype(np.float32), rtol=0, atol=0)


@pytest.mark.data
def test_cached_subject_has_two_balanced_144_trial_sessions() -> None:
    from src.data.preprocess import load_subject

    sessions = load_subject(1)
    assert set(sessions) == {1, 2}
    for session, (X, y) in sessions.items():
        assert X.shape == (144, 22, 500), f"session {session} has the wrong trial tensor"
        assert X.dtype == np.float32
        assert y.shape == (144,)
        assert y.dtype == np.int64
        assert set(np.unique(y)) == {0, 1}, "only event 769/770 labels may remain"
        assert_array_equal(np.bincount(y, minlength=2), np.array([72, 72]))

