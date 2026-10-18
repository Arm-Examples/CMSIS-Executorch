"""Fast, offline tests of the v2 125 Hz data pipeline (synthetic Raw objects, fake caches)."""

from __future__ import annotations

import copy

import mne
import numpy as np
import pytest

from src.data import cache125, splits125
from src.data.align import apply_ea, ea_matrix, mean_covariance
from src.data.external import DataNotDownloaded, map_events_by_name, standardize_channels
from src.data.preprocess import design_filter
from src.train import CHANNELS

EXTRA_61 = ["Fp1", "Fp2", "Fpz", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1", "F2", "F4",
            "F6", "F8", "FT7", "FC5", "FC6", "FT8", "T7", "T8", "TP7", "CP5", "CP6", "TP8", "P7", "P5",
            "P3", "P4", "P6", "P8", "PO7", "PO3", "PO4", "PO8", "O1", "Oz", "O2", "Iz"]


def _raw(ch_names: list[str], sfreq: float, n_times: int, events: np.ndarray, rng, data=None) -> mne.io.RawArray:
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    if data is None:
        data = rng.normal(scale=10.0, size=(len(ch_names), n_times))  # µV
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)
    raw.info["temp"] = {"events": np.asarray(events, dtype=np.int64)}
    return raw


def _events(cues, codes) -> np.ndarray:
    cues = np.asarray(cues, dtype=np.int64)
    return np.stack([cues, np.zeros_like(cues), np.asarray(codes, dtype=np.int64)], axis=1)


def _stream_decimate(sos: np.ndarray, x: np.ndarray, factor: int = 2) -> np.ndarray:
    """Reference board loop: one sample in, cascade of DF2T biquads, keep n % factor == 0."""
    z = np.zeros((len(sos), 2))
    out = []
    for n, v in enumerate(x):
        for k, (b0, b1, b2, _a0, a1, a2) in enumerate(sos):
            y = b0 * v + z[k, 0]
            z[k, 0] = b1 * v - a1 * y + z[k, 1]
            z[k, 1] = b2 * v - a2 * y
            v = y
        if n % factor == 0:
            out.append(v)
    return np.asarray(out)


def test_board_decimation_matches_streaming_filter(rng):
    sos = design_filter(fs=250.0)
    x = rng.normal(size=(3, 801))
    got = cache125.board_decimate(x, sos)
    assert got.shape == (3, 401)
    for c in range(3):
        np.testing.assert_allclose(got[c], _stream_decimate(sos, x[c]), rtol=1e-10, atol=1e-10)


def test_2a_window_is_board_samples(rng):
    n = 250 * 30
    cues = [1001, 1750, 3002, 4503]  # odd and even cue samples at 250 Hz
    raw = _raw(CHANNELS, 250.0, n, _events(cues, [769, 770, 771, 769]), rng)
    X, y = cache125.windows_2a(raw)
    assert X.shape == (3, 22, cache125.WINDOW) and X.dtype == np.float32
    np.testing.assert_array_equal(y, [0, 1, 0])
    x = raw.get_data() * 1e6
    x = x - x.mean(axis=0, keepdims=True)
    sos = design_filter(fs=250.0)
    full = np.stack([_stream_decimate(sos, row, factor=1) for row in x])  # filtered at 250 Hz
    for i, cue in enumerate([1001, 1750, 4503]):
        idx = 2 * (cue // 2 + cache125.WINDOW_START + np.arange(cache125.WINDOW))
        np.testing.assert_allclose(X[i], full[:, idx], rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("sfreq", [512.0, 1000.0])
def test_external_window_length_and_crop(rng, sfreq):
    n = int(sfreq * 40)
    cues = (np.array([3.0, 10.0, 17.0, 24.0, 31.0]) * sfreq).astype(int)
    raw = _raw(CHANNELS, sfreq, n, _events(cues, [769, 770, 769, 770, 772]), rng)
    X, y = cache125.windows_external(raw)
    assert X.shape == (4, 22, cache125.TRAIN_WINDOW) and X.dtype == np.float32
    np.testing.assert_array_equal(y, [0, 1, 0, 1])
    assert cache125.TRAIN_START + cache125.CROP_OFFSET == cache125.WINDOW_START
    assert X[..., cache125.CROP_OFFSET:cache125.CROP_OFFSET + cache125.WINDOW].shape[-1] == 256


def test_config_matches_pipeline_constants():
    cfg = splits125.load_config()
    splits125.check_config(cfg)
    assert cfg["window"] == 256 and cfg["train_window"] == 306 and cfg["crop_offset"] == 25
    assert cfg["fs_model"] == 125
    bad = copy.deepcopy(cfg)
    bad["window"] = 250
    with pytest.raises(ValueError):
        splits125.check_config(bad)


def test_standardize_interpolates_fcz_on_61_channels(rng):
    names = [c for c in CHANNELS if c != "FCz"] + EXTRA_61
    assert len(names) == 61
    montage = mne.channels.make_standard_montage("standard_1005")
    pos = montage.get_positions()["ch_pos"]
    n = 2000
    # Smooth scalp field: a linear function of electrode position plus small noise.
    a = rng.normal(size=(3, n))
    field = lambda chs: np.stack([np.asarray(pos[c]) @ a for c in chs]) * 1e3  # noqa: E731
    data = field(names) + 0.01 * rng.normal(size=(len(names), n))
    raw = _raw(names, 1000.0, n, _events([100], [769]), rng, data=data)
    out = standardize_channels(raw)
    assert out.ch_names == CHANNELS
    assert raw.ch_names == names  # input untouched
    np.testing.assert_array_equal(out.info["temp"]["events"], raw.info["temp"]["events"])
    fcz = out.get_data(picks=["FCz"])[0] * 1e6
    assert np.all(np.isfinite(fcz)) and np.std(fcz) > 0
    assert np.corrcoef(fcz, field(["FCz"])[0])[0, 1] > 0.9
    # Channels that were present keep their data exactly.
    np.testing.assert_allclose(out.get_data(picks=["C3"])[0], raw.get_data(picks=["C3"])[0])


def test_standardize_reorders_2a_style_raw(rng):
    shuffled = list(rng.permutation(CHANNELS))
    raw = _raw(shuffled, 250.0, 500, _events([100], [770]), rng)
    out = standardize_channels(raw)
    assert out.ch_names == CHANNELS
    np.testing.assert_allclose(out.get_data(picks=["Cz"]), raw.get_data(picks=["Cz"]))


def test_car_rows_sum_to_zero(rng):
    x = rng.normal(size=(22, 1000)) + 50.0
    np.testing.assert_allclose(cache125.common_average_reference(x).sum(axis=0), 0.0, atol=1e-9)
    raw = _raw(CHANNELS, 250.0, 250 * 10, _events([600, 1300], [769, 770]), rng)
    X, _ = cache125.windows_2a(raw)
    assert np.abs(X.sum(axis=1)).max() < 1e-4 * np.abs(X).max()
    raw = _raw(CHANNELS, 512.0, 512 * 12, _events([1024, 3072], [769, 770]), rng)
    X, _ = cache125.windows_external(raw)
    assert np.abs(X.sum(axis=1)).max() < 1e-4 * np.abs(X).max()


def test_label_mapping_by_name():
    lee = {"left_hand": 2, "right_hand": 1}
    ev = _events([10, 20, 30, 40], [2, 1, 3, 2])
    out = map_events_by_name(ev, lee)
    np.testing.assert_array_equal(out[:, 0], [10, 20, 40])
    np.testing.assert_array_equal(out[:, 2], [769, 770, 769])
    cho = {"left_hand": 1, "right_hand": 2}
    np.testing.assert_array_equal(map_events_by_name(ev, cho)[:, 2], [770, 769, 770])
    with pytest.raises(ValueError):
        map_events_by_name(ev, {"feet": 3})


def test_ea_eps_handles_car_rank_deficiency(rng):
    X = rng.normal(size=(40, 22, 256))
    X = X - X.mean(axis=1, keepdims=True)  # CAR: covariance rank 21
    R = mean_covariance(X)
    assert np.linalg.eigvalsh(R).min() < 1e-10 * np.trace(R)
    try:
        A0 = ea_matrix(X, 0.0)
        assert np.abs(A0).max() > 1e3  # if it does not raise, it explodes
    except np.linalg.LinAlgError:
        pass
    A = ea_matrix(X, 1e-3)
    assert np.all(np.isfinite(A))
    P = np.eye(22) - np.ones((22, 22)) / 22
    np.testing.assert_allclose(P @ A @ R @ A @ P, P, atol=0.02)


# ---- v2_data leakage with fake caches -------------------------------------------------

def _car(x):
    return x - x.mean(axis=-2, keepdims=True)


class FakeLoader:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.calls: list[tuple[str, int]] = []
        self.data: dict[tuple[str, int], dict] = {}

    def __call__(self, dataset: str, subject: int):
        self.calls.append((dataset, subject))
        key = (dataset, subject)
        if key in self.data:
            return self.data[key]
        if dataset == "Cho2017" and subject == 3:
            raise DataNotDownloaded("s03.mat")
        r = self.rng
        if dataset == "BNCI2014_001":
            X2 = _car(r.normal(size=(144, 22, 256))).astype(np.float32)
            d = {1: (np.full((144, 22, 256), np.nan, np.float32), np.arange(144) % 2),
                 2: (X2, (np.arange(144) // 3) % 2)}
        elif dataset == "Cho2017":
            n = 40 if subject == 1 else 48
            d = {1: (_car(r.normal(size=(n, 22, 306))).astype(np.float32), np.arange(n) % 2)}
        else:
            d = {s: (_car(r.normal(size=(20, 22, 306))).astype(np.float32), np.arange(20) % 2) for s in (1, 2)}
        self.data[key] = d
        return d


def _tiny_cfg() -> dict:
    cfg = splits125.load_config()
    cfg["datasets"] = {"Cho2017": {"subjects": [1, 2, 3, 4], "exclude": [4]},
                       "Lee2019_MI": {"subjects": "1-2", "exclude": []}}
    cfg["test"]["subjects"] = [1, 3]
    return cfg


def _fingerprints(X: np.ndarray) -> np.ndarray:
    return np.sort(np.round(X.astype(np.float64).reshape(len(X), -1) @ np.linspace(1, 2, X[0].size), 3))


def test_v2_data_contract_and_no_2a_leakage():
    cfg = _tiny_cfg()
    loader = FakeLoader()
    with pytest.warns(UserWarning, match="Cho2017 subject 3"):
        out = splits125.v2_data(0, cfg, loader=loader)
    eps = cfg["ea_eps"]

    # 2a only for the test subjects; Cho2017 s04 excluded by config.
    assert {c for c in loader.calls if c[0] == "BNCI2014_001"} == {("BNCI2014_001", 1), ("BNCI2014_001", 3)}
    assert ("Cho2017", 4) not in loader.calls
    assert out["meta"]["skipped"] == ["Cho2017_s03"]
    assert out["meta"]["n_train_subject_sessions"] == 2 + 4
    assert out["meta"]["sources"]["Cho2017"]["subjects"] == 2
    assert out["meta"]["sources"]["Lee2019_MI"]["subject_sessions"] == 4

    Xtr, ytr = out["train"]
    Xva, yva = out["val"]
    assert Xtr.dtype == np.float32 and ytr.dtype == np.int64 and Xtr.shape[1:] == (22, 306)
    assert Xva.shape[1:] == (22, 306)
    assert len(ytr) + len(yva) == 40 + 48 + 4 * 20
    assert len(yva) == 4 + 4 + 4 * 2  # round(10 %) per class per subject-session
    assert np.all(np.isfinite(Xtr)) and np.all(np.isfinite(Xva))  # 2a session 1 (NaN) never used

    # Every train/val row is an EA-aligned external trial, each exactly once.
    ext = []
    for (ds, s), d in loader.data.items():
        if ds == "BNCI2014_001":
            continue
        for X, _ in d.values():
            ext.append(apply_ea(X, ea_matrix(X, eps)))
    np.testing.assert_allclose(_fingerprints(np.concatenate([Xtr, Xva])), _fingerprints(np.concatenate(ext)),
                               rtol=1e-6)

    for s in (1, 3):
        X2, y2 = loader.data[("BNCI2014_001", s)][2]
        A = ea_matrix(X2[:24], eps)
        np.testing.assert_allclose(out["ea"][s], A)
        assert out["calib"][s].shape == (24, 22, 256) and out["calib"][s].dtype == np.float32
        np.testing.assert_allclose(out["calib"][s], apply_ea(X2[:24], A), rtol=1e-5, atol=1e-6)
        Xt, yt = out["test"][s]
        assert Xt.shape == (120, 22, 256)
        np.testing.assert_allclose(Xt, apply_ea(X2[24:], A), rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(yt, y2[24:])
    assert set(out["test"]) == {1, 3}
    assert out["input_scale"] == pytest.approx(float(np.std(Xtr)))
    assert out["meta"]["fs"] == 125 and out["meta"]["window"] == 256 and out["meta"]["crop_offset"] == 25


def test_v2_data_seeded_validation():
    cfg = _tiny_cfg()
    loader = FakeLoader()
    with pytest.warns(UserWarning):
        a = splits125.v2_data(0, cfg, loader=loader)
        b = splits125.v2_data(0, cfg, loader=loader)
        c = splits125.v2_data(1, cfg, loader=loader)
    np.testing.assert_array_equal(a["val"][0], b["val"][0])
    assert not np.array_equal(a["val"][0], c["val"][0])
    # Stratified: equal class counts in validation for these balanced sessions.
    assert np.bincount(a["val"][1]).tolist() == [8, 8]


def test_v2_data_rejects_test_dataset_as_source():
    cfg = _tiny_cfg()
    cfg["datasets"] = {"BNCI2014_001": {"subjects": [1]}, **cfg["datasets"]}
    with pytest.raises(ValueError, match="test dataset"):
        splits125.v2_data(0, cfg, loader=FakeLoader())
