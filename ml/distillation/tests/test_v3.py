"""v3: signal-quality gate, cross-subject validation, shared augmentation, v3 loops (no datasets needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data import splits125
from src.data.align import apply_ea, ea_matrix
from src.data.quality import session_decision, sqi, trial_flags, whitened_distance

ROOT = Path(__file__).resolve().parents[1]
WINDOW, TRAIN_WINDOW, OFFSET = 256, 306, 25
QCFG = {"ptp_uv": 100.0, "potato_z": 2.5, "flat_uv": 0.1, "max_flagged_fraction": 0.2}


def _car(x):
    return x - x.mean(axis=-2, keepdims=True)


def clean_session(rng, n=60, T=TRAIN_WINDOW, sd=3.0):
    mix = np.eye(22) + 0.2 * rng.standard_normal((22, 22))
    return _car(np.einsum("cd,ndt->nct", mix, rng.standard_normal((n, 22, T)) * sd)).astype(np.float32)


# ----------------------------------------------------------------------------- quality gate
def test_quality_gate_flags_artifacts_and_keeps_clean_trials():
    rng = np.random.default_rng(0)
    X = clean_session(rng)
    X[0, 5, 100] += 500.0                                      # 500 uV spike
    X[1, 9] = 0.0                                              # flat channel
    X[2] += (rng.standard_t(1.2, size=X[2].shape) * 8).clip(-400, 400)  # kurtotic, heavy tails
    X[3, :6] *= 3.0                                            # covariance outlier, still < 100 uV ptp
    f = trial_flags(X, None, QCFG)
    assert f["ptp"][0] and f["flat"][1] and f["any"][2]
    assert not f["ptp"][3] and f["potato"][3]
    assert f["any"][:4].all()
    assert f["any"][4:].mean() <= 0.05                         # clean trials kept
    dec = session_decision(f, QCFG, X)
    assert dec["keep"] and dec["flagged"] == int(f["any"].sum())


def test_session_decision_drops_bad_or_flat_sessions():
    rng = np.random.default_rng(1)
    X = clean_session(rng, n=40)
    X[:12, 3, 50] += 300.0                                     # 30 % flagged > 20 %
    dec = session_decision(trial_flags(X, None, QCFG), QCFG, X)
    assert not dec["keep"] and "flagged" in dec["reason"]
    X = clean_session(rng, n=40)
    X[:, 4] = 0.01 * rng.standard_normal(X[:, 4].shape)       # channel flat over the whole session
    dec = session_decision(trial_flags(X, None, QCFG), QCFG, X)
    assert not dec["keep"] and "flat" in dec["reason"]


def test_sqi_matches_trial_measures():
    rng = np.random.default_rng(2)
    X = clean_session(rng, n=20)
    A = ea_matrix(X, 1e-3)
    q = sqi(X[3], A)
    assert q["distance"] == pytest.approx(float(whitened_distance(X[3:4], A)[0]))
    assert q["ptp_uv"] == pytest.approx(float(np.abs(X[3] - X[3].mean(-1, keepdims=True)).max()), rel=1e-6)
    assert set(q) == {"ptp_uv", "min_std_uv", "distance"}


# ----------------------------------------------------------------------------- v2_data: gate + val subjects
class Loader:
    def __init__(self, bad=None):
        self.rng = np.random.default_rng(3)
        self.data = {}
        self.bad = bad

    def __call__(self, dataset, subject):
        key = (dataset, subject)
        if key not in self.data:
            r = self.rng
            if dataset == "BNCI2014_001":
                X2 = _car(r.normal(size=(144, 22, 256))).astype(np.float32)
                X2[30, 2, 10] += 900.0  # 2a test trial with an artifact: never filtered
                self.data[key] = {1: (X2.copy(), np.arange(144) % 2), 2: (X2, (np.arange(144) // 3) % 2)}
            elif dataset == "Cho2017":
                self.data[key] = {1: (clean_session(r, n=40), np.arange(40) % 2)}
            else:
                self.data[key] = {s: (clean_session(r, n=20), np.arange(20) % 2) for s in (1, 2)}
            if key == self.bad:
                X = self.data[key][1][0]
                X[: len(X) // 2, 0, 20] += 400.0
        return self.data[key]


def _cfg(**extra):
    cfg = splits125.load_config()
    cfg["datasets"] = {"Cho2017": {"subjects": [1, 2, 3]}, "Lee2019_MI": {"subjects": [1, 2]}}
    cfg["test"]["subjects"] = [1, 2]
    cfg.update(extra)
    return cfg


def _fp(X):
    return np.sort(np.round(X.astype(np.float64).reshape(len(X), -1) @ np.linspace(1, 2, X[0].size), 3))


def test_cross_subject_validation_isolates_val_subjects():
    loader = Loader()
    out = splits125.v2_data(0, _cfg(validation={"val_subjects": {"Cho2017": [3], "Lee2019_MI": [2]}}), loader=loader)
    eps = 1e-3

    def aligned(ds, s):
        return np.concatenate([apply_ea(X, ea_matrix(X, eps)) for X, _ in loader.data[(ds, s)].values()])

    val = np.concatenate([aligned("Cho2017", 3), aligned("Lee2019_MI", 2)])
    train = np.concatenate([aligned("Cho2017", 1), aligned("Cho2017", 2), aligned("Lee2019_MI", 1)])
    np.testing.assert_allclose(_fp(out["val"][0]), _fp(val), rtol=1e-6)
    np.testing.assert_allclose(_fp(out["train"][0]), _fp(train), rtol=1e-6)
    assert len(out["val"][1]) == 40 + 2 * 20 and len(out["train"][1]) == 2 * 40 + 2 * 20
    assert set(out["ea_train"]) == {"Cho2017_s01_1", "Cho2017_s02_1", "Lee2019_MI_s01_1", "Lee2019_MI_s01_2"}
    assert not set(out["ea_train"]) & set(out["ea_val"])
    assert len(out["train_group"]) == len(out["train"][1]) and out["train_group"].max() == 3
    assert out["meta"]["validation"]["val_subjects"] == {"Cho2017": [3], "Lee2019_MI": [2]}
    # every training trial's group points at its own session's EA matrix
    keys = list(out["ea_train"])
    np.testing.assert_allclose(out["ea_train"][keys[0]], ea_matrix(loader.data[("Cho2017", 1)][1][0], eps))
    assert np.all(out["train_group"][:40] == 0)


def test_quality_gate_in_v2_data_drops_session_and_never_touches_2a():
    base = splits125.v2_data(0, _cfg(), loader=Loader(bad=("Cho2017", 2)))
    out = splits125.v2_data(0, _cfg(quality=dict(QCFG)), loader=Loader(bad=("Cho2017", 2)))
    q = out["meta"]["quality"]
    assert [d["session"] for d in q["dropped"]] == ["Cho2017_s02_1"]
    assert q["datasets"]["Cho2017"]["sessions"] == 3 and q["datasets"]["Cho2017"]["dropped"] == 1
    assert len(out["train"][1]) + len(out["val"][1]) == len(base["train"][1]) + len(base["val"][1]) - 40
    assert "Cho2017_s02_1" not in out["ea_train"]
    for s in (1, 2):  # 2a test and calibration blocks identical with and without the gate
        np.testing.assert_array_equal(out["test"][s][0], base["test"][s][0])
        np.testing.assert_array_equal(out["calib"][s], base["calib"][s])
        assert out["test"][s][0].shape == (120, 22, 256)
    assert set(q["calib_2a"]) == {"1", "2"} and "quality" not in base["meta"]


def test_v2_config_has_no_v3_sections_and_v3_config_matches_plan():
    v2 = yaml.safe_load((ROOT / "configs" / "v2.yaml").read_text())
    assert not {"quality", "validation", "augment"} & set(v2) and "loop" not in v2["training"]
    v3 = yaml.safe_load((ROOT / "configs" / "v3.yaml").read_text())
    for k in ("fs_model", "window", "window_start", "train_window", "crop_offset", "filter", "ea_eps", "student",
              "datasets", "test"):
        assert v3[k] == v2[k], k
    assert v3["validation"]["val_subjects"] == {"Cho2017": [48, 49, 50, 51, 52], "Lee2019_MI": [19, 20]}
    assert v3["quality"]["ptp_uv"] == 100 and v3["quality"]["max_flagged_fraction"] == 0.2
    t = v3["training"]["teacher"]
    assert (t["lr_shallow"], t["weight_decay"], t["label_smoothing"], t["ema"], t["warmup"], t["patience"]) == \
        (3e-4, 1e-3, 0.1, 0.999, 5, 40)
    assert v3["augment"]["channel_dropout"] == 0.2 and v3["augment"]["ea_mix_p"] == 0.1
    assert v3["distill"]["mixup_alpha"] == 0.4 and v3["distill"]["weight_decay"] == 1e-4


# ----------------------------------------------------------------------------- augmentation
def _augmenter(**over):
    from src.augment import Augmenter, mixing_matrices

    rng = np.random.default_rng(4)
    ea = {f"g{i}": ea_matrix(clean_session(rng, n=10), 1e-3) for i in range(3)}
    c = {"crop_jitter": 25, "noise_std": 0.05, "channel_dropout": 0.2, "amp_low": 0.8, "amp_high": 1.2,
         "ea_mix_p": 0.1, **over}
    return Augmenter(c, window=WINDOW, crop_offset=OFFSET, train_window=TRAIN_WINDOW, mix=mixing_matrices(ea)), ea


def test_augmenter_shapes_and_labels():
    aug, _ = _augmenter()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(32, 1, 22, TRAIN_WINDOW)
    out = aug(x, torch.arange(32) % 3, gen)
    assert out.shape == (32, 1, 22, WINDOW) and torch.isfinite(out).all()
    cal = torch.randn(8, 1, 22, WINDOW)
    assert aug(cal, None, gen).shape == (8, 1, 22, WINDOW)
    # amplitude jitter only: per-trial ratio in [0.8, 1.2]
    aug2, _ = _augmenter(crop_jitter=0, noise_std=0.0, channel_dropout=0.0, ea_mix_p=0.0)
    r = aug2(x, None, gen) / x[..., OFFSET : OFFSET + WINDOW]
    ratio = r.flatten(1)
    assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio), atol=1e-4)
    assert ratio.min() >= 0.8 - 1e-5 and ratio.max() <= 1.2 + 1e-5


def test_ea_mixing_realigns_with_another_session_matrix():
    from src.augment import mixing_matrices

    aug, ea = _augmenter(crop_jitter=0, noise_std=0.0, channel_dropout=0.0, amp_low=1.0, amp_high=1.0, ea_mix_p=1.0)
    A = list(ea.values())
    rng = np.random.default_rng(5)
    raw = clean_session(rng, n=6, T=TRAIN_WINDOW)
    M = mixing_matrices(ea).numpy()
    np.testing.assert_allclose(M[2, 0] @ A[0] @ raw[0], A[2] @ raw[0], rtol=1e-3, atol=1e-3)
    x = torch.from_numpy(apply_ea(raw, A[0]).astype(np.float32))[:, None]
    out = aug(x, torch.zeros(6, dtype=torch.long), torch.Generator().manual_seed(1))[:, 0].numpy().astype(np.float64)
    want = [apply_ea(raw[:, :, OFFSET : OFFSET + WINDOW], A[o]) for o in (1, 2)]
    for i in range(6):
        assert any(np.allclose(out[i], w[i], rtol=1e-3, atol=1e-3) for w in want)  # an *other* session
    own = np.einsum("nct,ndt->cd", x[:, 0].numpy(), x[:, 0].numpy())
    mixed = np.einsum("nct,ndt->cd", out, out)
    assert np.linalg.norm(own - mixed) > 0.1 * np.linalg.norm(own)


def test_mixup_lambda_and_labels():
    from src.augment import mixup, mixup_ce

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(256, 1, 22, WINDOW)
    y = torch.arange(256) % 2
    y[:16] = -1  # unlabelled calibration trials
    xm, ya, yb, lam = mixup(x, y, gen, 0.4, 0.5)
    assert xm.shape == x.shape and torch.equal(ya, y)
    assert (lam >= 0).all() and (lam <= 1).all()
    assert torch.equal(lam[:16], torch.ones(16)) and torch.equal(xm[:16], x[:16]) and torch.equal(yb[:16], y[:16])
    mixed = lam[16:] < 1
    assert 0.3 < mixed.float().mean() < 0.7 and (yb[16:] >= 0).all()
    logits = torch.randn(256, 2)
    ce = mixup_ce(logits, ya, yb, lam)
    assert torch.all(ce[:16] == 0) and torch.isfinite(ce).all()


# ----------------------------------------------------------------------------- loops
def test_cosine_warmup_and_ema():
    from src.train_v3 import EMA, cosine_warmup

    m = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    s = cosine_warmup(opt, 5, 105)
    lrs = []
    for _ in range(105):
        lrs.append(s.get_last_lr()[0]); opt.step(); s.step()
    assert lrs[0] == pytest.approx(0.2) and lrs[4] == pytest.approx(1.0) and lrs[-1] < 0.01
    ema = EMA(m, 0.9)
    w0 = ema.model.weight.clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update(m)
    d = min(0.9, 2 / 11)
    assert torch.allclose(ema.model.weight, d * w0 + (1 - d) * m.weight)


def fake_v3_data(seed: int, cfg: dict) -> dict:
    rng = np.random.default_rng(seed)

    def trials(n, T):
        y = np.arange(n) % 2
        X = rng.standard_normal((n, 22, T)).astype(np.float32) * 5.0
        X[y == 1, 7] *= 2.0
        return X, y.astype(np.int64)

    X_train, y_train = trials(48, TRAIN_WINDOW)
    subjects = list(range(1, 10))
    ea = {f"Cho2017_s0{i}_1": ea_matrix(clean_session(rng, n=10), 1e-3) for i in (1, 2)}
    return {"train": (X_train, y_train), "val": trials(16, WINDOW + 50),
            "calib": {s: trials(24, WINDOW)[0] for s in subjects}, "test": {s: trials(12, WINDOW) for s in subjects},
            "ea": {s: np.eye(22) for s in subjects}, "input_scale": float(X_train.std()),
            "meta": {"fs": 125, "window": WINDOW, "crop_offset": OFFSET,
                     "quality": {"config": QCFG, "dropped": [], "datasets": {
                         "Cho2017": {"sessions": 2, "dropped": 0, "trials": 64,
                                     "counts": {"ptp": 1, "potato": 2, "flat": 0, "any": 3},
                                     "flag_rates": {"any": 3 / 64}}}},
                     "validation": {"val_subjects": {"Cho2017": [3]}, "val_sessions": ["Cho2017_s03_1"]}},
            "train_group": np.arange(48) % 2, "ea_train": ea, "val_group": np.zeros(16, dtype=np.int64), "ea_val": {}}


def _v3_cfg_file(tmp_path: Path) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "v3.yaml").read_text())
    p = tmp_path / "v3.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_v3_teacher_parallel_members_distill_and_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from src import distill, run_v2, train_teacher

    monkeypatch.setattr(splits125, "v2_data", fake_v3_data)
    cfg = _v3_cfg_file(tmp_path)
    runs = tmp_path / "runs"
    t_out = runs / "teacher_seed0"
    common = ["--config", str(cfg), "--epochs", "2", "--seed", "0", "--out", str(t_out)]
    calls = []
    real = train_teacher.fit
    monkeypatch.setattr(train_teacher, "fit", lambda *a, **k: calls.append(1) or real(*a, **k))
    for m in ("eegnet", "shallow"):
        r = train_teacher.run(train_teacher.parse_args([*common, "--members", m]))
        assert r["partial"] and not (t_out / "metrics.json").exists()
    tm = train_teacher.run(train_teacher.parse_args([*common, "--members", "deep"]))
    assert tm["partial"]
    tm = train_teacher.run(train_teacher.parse_args([*common, "--reuse-members"]))
    assert not calls  # v3 loop, never src.train.fit
    assert set(tm["members"]) == {"eegnet", "shallow", "deep"} and tm["members"]["shallow"]["lr"] == 3e-4
    assert tm["members"]["deep"]["epochs_run"] == 2 and (t_out / "teacher_logits.npz").is_file()
    s_out = runs / "student_wide32_seed0"
    sm = distill.run(distill.parse_args(["--config", str(cfg), "--epochs", "2", "--seed", "0", "--student", "wide32",
                                         "--teacher", str(t_out), "--out", str(s_out)]))
    assert sm["epochs_run"] == 2 and sm["params"] == 3426 and "lr" in json.loads((s_out / "history.json").read_text())[0]
    results = tmp_path / "results" / "results_v3.md"
    run_v2.main(["--skip-train", "--out", str(runs), "--results", str(results), "--config", str(cfg),
                 "--seeds", "0", "--students", "wide32", "--compare", str(tmp_path / "none")])
    text = results.read_text()
    for label in ("v3 results", "Signal-quality gate", "Cross-subject validation accuracy", "whole held-out"):
        assert label in text, label
    assert (tmp_path / "results" / "confusion_v3" / "A01.md").is_file()


def test_v2_config_keeps_v2_teacher_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """configs/v2.yaml: train_teacher calls src.train.fit with the v2 arguments; its first-epoch
    loss equals a direct src.train.fit call on the same inputs."""
    from src import train_teacher
    from src.train import fit, set_seed, to_model_input

    def fake_v2(seed, cfg):
        d = fake_v3_data(seed, cfg)
        return {k: d[k] for k in ("train", "val", "calib", "test", "ea", "input_scale", "meta")}

    monkeypatch.setattr(splits125, "v2_data", fake_v2)
    import src.train_v3 as tv3

    monkeypatch.setattr(tv3, "fit_v3", lambda *a, **k: pytest.fail("v3 loop used with the v2 config"))
    out = tmp_path / "t"
    train_teacher.run(train_teacher.parse_args(["--config", str(ROOT / "configs" / "v2.yaml"), "--epochs", "1",
                                                "--seed", "0", "--out", str(out)]))
    hist = json.loads((out / "history.json").read_text())["eegnet"][0]
    cfg = yaml.safe_load((ROOT / "configs" / "v2.yaml").read_text())
    d = fake_v2(0, cfg)
    s = d["input_scale"]
    Xtr = to_model_input(train_teacher.canonical(d["train"][0], cfg), s)
    Xva = to_model_input(train_teacher.canonical(d["val"][0], cfg), s)
    set_seed(0)
    model = train_teacher.build_member("eegnet", cfg)
    info = fit(model, Xtr, d["train"][1], Xva, d["val"][1], lr=cfg["training"]["lr"],
               batch_size=cfg["training"]["batch_size"], max_epochs=1, patience=30, seed=0, log=lambda m: None)
    assert info["history"][0]["train_loss"] == pytest.approx(hist["train_loss"], rel=1e-6)
    assert info["history"][0]["val_loss"] == pytest.approx(hist["val_loss"], rel=1e-6)
