"""Provenance-coded split tests that make target leakage directly observable."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest


def _fake_subject(subject: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    sessions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for session in (1, 2):
        trials = np.arange(1, 145, dtype=np.int64)
        markers = subject * 10_000 + session * 1_000 + trials
        X = markers.astype(np.float32).reshape(-1, 1, 1)
        y = (trials % 2).astype(np.int64)
        sessions[session] = (X, y)
    return sessions


def _parts(result: object) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    names = ("train", "val", "calib", "test")
    if isinstance(result, Mapping):
        out = {}
        for name in names:
            if name in result:
                value = result[name]
                if isinstance(value, Mapping):
                    x_value = value["X"] if "X" in value else value["x"]
                    out[name] = (np.asarray(x_value), np.asarray(value["y"]))
                else:
                    out[name] = (np.asarray(value[0]), np.asarray(value[1]))
            else:
                x_key = f"X_{name}" if f"X_{name}" in result else f"x_{name}"
                out[name] = (np.asarray(result[x_key]), np.asarray(result[f"y_{name}"]))
        return out
    if isinstance(result, Sequence) and len(result) == 4:
        return {name: (np.asarray(value[0]), np.asarray(value[1])) for name, value in zip(names, result)}
    if isinstance(result, Sequence) and len(result) == 8:
        return {
            name: (np.asarray(result[2 * index]), np.asarray(result[2 * index + 1]))
            for index, name in enumerate(names)
        }
    raise AssertionError("split result must expose train/val/calib/test (X, y) arrays")


def _decode(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marker = np.rint(np.asarray(X)[:, 0, 0]).astype(np.int64)
    return marker // 10_000, (marker % 10_000) // 1_000, marker % 1_000


@pytest.fixture
def identity_split_dependencies(monkeypatch: pytest.MonkeyPatch):
    import src.data.splits as splits

    monkeypatch.setattr(splits, "load_subject", _fake_subject)
    if hasattr(splits, "ea_matrix"):
        monkeypatch.setattr(splits, "ea_matrix", lambda X: np.eye(X.shape[1]))
    if hasattr(splits, "apply_ea"):
        monkeypatch.setattr(splits, "apply_ea", lambda X, A: np.asarray(X).copy())
    return splits


def _assert_target_calibration_and_test(parts: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    calib_subject, calib_session, calib_trial = _decode(parts["calib"][0])
    test_subject, test_session, test_trial = _decode(parts["test"][0])

    assert np.all(calib_subject == 1)
    assert np.all(calib_session == 2)
    assert np.array_equal(calib_trial, np.arange(1, 25))
    assert np.all(test_subject == 1)
    assert np.all(test_session == 2)
    assert np.array_equal(test_trial, np.arange(25, 145))
    assert len(np.intersect1d(calib_trial, test_trial)) == 0


def test_loso_has_no_target_subject_or_early_stopping_leakage(identity_split_dependencies) -> None:
    splits = identity_split_dependencies
    parts = _parts(splits.loso_data(target=1, seed=0))

    train_subject, train_session, train_trial = _decode(parts["train"][0])
    val_subject, val_session, val_trial = _decode(parts["val"][0])
    assert 1 not in train_subject
    assert 1 not in val_subject
    assert set(train_subject) == set(range(2, 10))
    assert set(val_subject) == set(range(2, 10))

    # Every source subject-session contributes its own stratified ~10% validation
    # subset. Provenance-coded trial numbers make overlap and balance observable.
    for subject in range(2, 10):
        for session in (1, 2):
            train_mask = (train_subject == subject) & (train_session == session)
            val_mask = (val_subject == subject) & (val_session == session)
            group_train_trials = train_trial[train_mask]
            group_val_trials = val_trial[val_mask]
            group_val_y = parts["val"][1][val_mask]

            assert len(group_train_trials) + len(group_val_trials) == 144
            assert len(group_val_trials) == pytest.approx(0.1 * 144, abs=1)
            assert len(np.intersect1d(group_train_trials, group_val_trials)) == 0
            class_counts = np.bincount(group_val_y, minlength=2)
            assert abs(int(class_counts[0]) - int(class_counts[1])) <= 1
    _assert_target_calibration_and_test(parts)


def test_per_subject_early_stopping_uses_session_one_only(identity_split_dependencies) -> None:
    splits = identity_split_dependencies
    parts = _parts(splits.per_subject_data(target=1, seed=0))

    for name in ("train", "val"):
        subject, session, _ = _decode(parts[name][0])
        assert np.all(subject == 1)
        assert np.all(session == 1), f"{name} must not use target session 2"
    train_marker = np.asarray(parts["train"][0])[:, 0, 0]
    val_marker = np.asarray(parts["val"][0])[:, 0, 0]
    assert len(np.intersect1d(train_marker, val_marker)) == 0
    assert len(train_marker) + len(val_marker) == 144
    _assert_target_calibration_and_test(parts)

