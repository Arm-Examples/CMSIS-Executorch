"""v4: Riemannian teacher member, 2a session-1 self-training, extra KD windows (no datasets needed).

The guard tests make the session-1 labels unreadable (a poison object) for everything except the
separately named reporting function, and check that the 2a test block never enters training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data import cache125, splits125
from src.data.align import apply_ea, ea_matrix

ROOT = Path(__file__).resolve().parents[1]
WINDOW, TRAIN_WINDOW, OFFSET = 256, 306, 25
N_TARGET = 20  # session-1 windows per subject in the fake data


class PoisonLabels:
    """Stands in for the session-1 labels: any use raises."""

    def _boom(self, *a, **k):
        raise AssertionError("session-1 labels read by the training path")

    __getitem__ = __len__ = __iter__ = __array__ = __eq__ = __int__ = _boom

    def __getattr__(self, name):
        self._boom()


def _trials(rng, n, T, scale=5.0):
    y = (np.arange(n) % 2).astype(np.int64)
    X = rng.standard_normal((n, 22, T)).astype(np.float32) * scale
    X[y == 1, 7] *= 2.5  # class information in the covariance (channel 7 power)
    X[y == 0, 12] *= 2.5
    return X, y


def target_session(subject: int):
    """Fake 2a session 1 of one subject (X, true y)."""
    return _trials(np.random.default_rng(100 + subject), N_TARGET, WINDOW)


def poison_loader(dataset, subject):
    """``{session: (X, y)}`` with poisoned session-1 labels (session 2 is never needed here)."""
    assert dataset == "BNCI2014_001"
    return {1: (target_session(subject)[0], PoisonLabels())}


def label_loader(dataset, subject):
    """Reporting-only loader with the true session-1 labels."""
    X, y = target_session(subject)
    return {1: (X, y)}


def fake_v2_data(seed: int, cfg: dict) -> dict:
    rng = np.random.default_rng(seed)
    X_train, y_train = _trials(rng, 64, TRAIN_WINDOW)
    subjects = list(range(1, 10))
    return {"train": (X_train, y_train), "val": _trials(rng, 16, TRAIN_WINDOW),
            "calib": {s: _trials(rng, 24, WINDOW)[0] for s in subjects},
            "test": {s: _trials(np.random.default_rng(900 + s), 12, WINDOW) for s in subjects},
            "ea": {s: np.eye(22) for s in subjects}, "input_scale": float(X_train.std()),
            "meta": {"fs": 125, "window": WINDOW, "crop_offset": OFFSET}}


def _v4_cfg(tmp_path: Path, min_conf: float = 0.5, **training) -> Path:
    """configs/v4.yaml with a low confidence threshold (1-epoch fakes are not confident) so the
    round-1 training set really grows."""
    cfg = yaml.safe_load((ROOT / "configs" / "v4.yaml").read_text())
    cfg["self_training"]["min_conf"] = min_conf
    cfg["training"].pop("lr_reference", None)
    cfg["training"].update(training)
    p = tmp_path / "v4.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture
def guarded(monkeypatch: pytest.MonkeyPatch):
    """Fake data; session-1 X through the poison loader; the reporting function is the only
    label reader and records the moment it is called."""
    real_target = splits125.target_unlabeled
    monkeypatch.setattr(splits125, "v2_data", fake_v2_data)
    monkeypatch.setattr(splits125, "target_unlabeled",
                        lambda cfg, loader=None, input_scale=None: real_target(cfg, poison_loader, input_scale))
    monkeypatch.setattr(cache125, "cached_windows", lambda *a, **k: pytest.fail("cache read in a unit test"))
    calls = []
    real_report = splits125.session_labels_for_reporting_only

    def report(cfg, loader=None):
        calls.append(1)
        return real_report(cfg, loader=label_loader)

    monkeypatch.setattr(splits125, "session_labels_for_reporting_only", report)
    return calls


# ----------------------------------------------------------------------------- data guards
def test_target_unlabeled_reads_x_only_and_aligns_per_subject():
    cfg = yaml.safe_load((ROOT / "configs" / "v4.yaml").read_text())
    t = splits125.target_unlabeled(cfg, poison_loader, input_scale=2.0)
    assert sorted(t["X"]) == list(range(1, 10)) and t["session"] == 1
    X1 = target_session(4)[0]
    A = ea_matrix(X1, 1e-3)
    np.testing.assert_allclose(t["X"][4], apply_ea(X1, A) / 2.0, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(t["ea"][4], A)
    assert t["X"][4].shape == (N_TARGET, 22, WINDOW) and np.all(t["y"][4] == -1)
    bad = json.loads(json.dumps(cfg))
    bad["self_training"]["session"] = 2  # the test session can never be the unlabelled target
    with pytest.raises(ValueError):
        splits125.target_unlabeled(bad, poison_loader)


def test_default_target_reader_never_loads_the_label_array(monkeypatch, tmp_path):
    X = np.random.default_rng(0).standard_normal((30, 22, WINDOW)).astype(np.float32)
    sess = np.repeat([1, 2], 15)
    for s in range(1, 10):
        np.savez(tmp_path / f"BNCI2014_001_s{s:02d}.npz", X=X, y=np.arange(30) % 2, session=sess,
                 version=np.int64(cache125.CACHE_VERSION))
    monkeypatch.setattr(cache125, "CACHE_DIR", tmp_path)
    keys = []
    real_load = np.load

    class Spy:
        def __init__(self, z):
            self.z = z
            self.files = z.files

        def __getitem__(self, k):
            keys.append(k)
            return self.z[k]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.z.close()

    monkeypatch.setattr(np, "load", lambda *a, **k: Spy(real_load(*a, **k)))
    cfg = yaml.safe_load((ROOT / "configs" / "v4.yaml").read_text())
    t = splits125.target_unlabeled(cfg)
    assert "y" not in keys and set(keys) <= {"X", "session", "version"}
    np.testing.assert_allclose(t["X"][1], apply_ea(X[:15], ea_matrix(X[:15], 1e-3)), rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------------- riemann member
def test_riemann_member_oas_roundtrip_and_logprob_forward(tmp_path):
    from sklearn.covariance import oas

    from src.models.riemann import RiemannMember, oas_covariances

    rng = np.random.default_rng(1)
    X, y = _trials(rng, 80, WINDOW)
    X = X - X.mean(axis=1, keepdims=True)  # CAR: rank-deficient trials
    C = oas_covariances(X[:3])
    for i in range(3):
        np.testing.assert_allclose(C[i], oas(X[i].T.astype(np.float64))[0], rtol=1e-10, atol=1e-12)
    assert np.linalg.eigvalsh(oas_covariances(X[:5], ridge=1e-3)).min() > 0
    x = torch.from_numpy(X[:, None])
    m = RiemannMember().fit(x, y)
    p = m.predict_proba(x)
    assert p.shape == (80, 2) and np.mean(p.argmax(1) == y) > 0.9
    m.save(tmp_path / "riemann.pkl")
    m2 = RiemannMember.load(tmp_path / "riemann.pkl")
    np.testing.assert_allclose(m2.predict_proba(x), p)
    z = m2(x)
    assert z.dtype == torch.float32 and torch.allclose(torch.softmax(z, 1), torch.from_numpy(p).float(), atol=1e-5)
    assert m.n_coefficients == 22 * 23 // 2 + 1 and not list(m.parameters())


def test_pseudo_label_filter_keeps_confident_windows_only():
    from src.train_teacher import pseudo_label_counts, select_pseudo_labels

    probs = {1: np.array([[0.9, 0.1], [0.55, 0.45], [0.2, 0.8], [0.79, 0.21]]), 2: np.array([[0.5, 0.5]])}
    sel = select_pseudo_labels(probs, 0.8)
    assert sel[1]["idx"].tolist() == [0, 2] and sel[1]["y"].tolist() == [0, 1] and sel[1]["n"] == 4
    assert len(sel[2]["idx"]) == 0
    c = pseudo_label_counts(sel)
    assert c["kept"] == 2 and c["n"] == 5 and c["per_subject"]["1"]["pseudo_right"] == 1


# ----------------------------------------------------------------------------- end to end
def test_self_training_never_reads_session1_labels_or_test_block(guarded, monkeypatch, tmp_path):
    from src import distill, train_teacher

    cfg = _v4_cfg(tmp_path)
    t_out = tmp_path / "runs" / "teacher_seed0"
    fit_inputs = []
    real_fit = train_teacher.fit
    monkeypatch.setattr(train_teacher, "fit", lambda m, X, y, *a, **k: fit_inputs.append((X.clone(), np.array(y)))
                        or real_fit(m, X, y, *a, **k))
    real_report = train_teacher.pseudo_label_agreement_report

    def report(*a, **k):  # the label reader runs only after the pseudo-labels are saved
        assert (t_out / "pseudo_labels.npz").is_file()
        return real_report(*a, **k)

    monkeypatch.setattr(train_teacher, "pseudo_label_agreement_report", report)
    tm = train_teacher.run(train_teacher.parse_args(["--config", str(cfg), "--epochs", "1", "--seed", "0",
                                                     "--out", str(t_out)]))
    assert guarded == [1]  # the reporting function read the labels exactly once
    assert set(tm["members"]) == {"eegnet", "shallow", "deep", "riemann"} and len(tm["ensemble"]["members"]) == 4
    for f in ("eegnet.pt", "shallow.pt", "deep.pt", "riemann.pkl"):
        assert (t_out / "members_round0" / f).is_file() and (t_out / "members" / f).is_file()
    kept = tm["self_training"]["counts"]["kept"]
    assert kept == 9 * N_TARGET  # min_conf 0.5 keeps every window
    assert tm["self_training"]["n_train_round1"] == 64 + kept
    assert 0.0 <= tm["self_training"]["agreement_report_only"]["agreement_kept"] <= 1.0
    assert tm["members"]["shallow"]["lr"] == pytest.approx(3e-4)
    # CNN fits: 3 in round 0 on the training split, 3 in round 1 on train + pseudo-labelled session 1
    assert [len(y) for _, y in fit_inputs] == [64] * 3 + [64 + kept] * 3
    d = fake_v2_data(0, {})
    test_rows = {np.round(r, 4).tobytes() for s in range(1, 10)
                 for r in (d["test"][s][0] / d["input_scale"]).reshape(12, -1).astype(np.float32)}
    for X, _ in fit_inputs:
        assert not test_rows & {np.round(r, 4).tobytes() for r in X.numpy().reshape(len(X), -1)}
    pl = np.load(t_out / "pseudo_labels.npz")
    assert sum(len(pl[f"idx_{s}"]) for s in range(1, 10)) == kept and float(pl["min_conf"]) == 0.5
    z = np.load(t_out / "teacher_logits.npz")
    assert z["riemann_test_3"].shape == (12, 2) and z["ensemble_target_9"].shape == (N_TARGET, 2)
    assert np.load(t_out / "teacher_logits_round0.npz")["riemann_target_1"].shape == (N_TARGET, 2)
    params = json.loads((t_out / "params.json").read_text())
    assert params["teacher_members"] == ["eegnet", "shallow", "deep", "riemann"]

    # 4-member ensemble in the KD loop; session 1 joins the calibration windows as KD-only
    members = train_teacher.load_members(t_out, yaml.safe_load(cfg.read_text()))
    assert list(members) == ["eegnet", "shallow", "deep", "riemann"]
    p = distill.teacher_probs(members.values(), torch.randn(5, 1, 22, WINDOW), 4.0)
    assert p.shape == (5, 2) and torch.allclose(p.sum(1), torch.ones(5))
    s_out = tmp_path / "runs" / "student_wide32_seed0"
    sm = distill.run(distill.parse_args(["--config", str(cfg), "--epochs", "1", "--seed", "0", "--student", "wide32",
                                         "--teacher", str(t_out), "--out", str(s_out)]))
    assert sm["n_target_kd"] == 9 * N_TARGET and sm["n_calib_kd"] == 9 * 24 + 9 * N_TARGET
    assert guarded == [1]
    r0_out = tmp_path / "runs" / "student_wide32_r0_seed0"
    distill.run(distill.parse_args(["--config", str(cfg), "--epochs", "1", "--seed", "0", "--student", "wide32",
                                    "--teacher", str(t_out), "--out", str(r0_out), "--teacher-round", "0"]))
    assert json.loads((r0_out / "params.json").read_text())["teacher_round"] == 0
    assert "round-0 members" in (r0_out / "train.log").read_text()


def test_parallel_rounds_lr_choice_and_report(guarded, monkeypatch, tmp_path):
    from src import run_v2, train_teacher

    runs = tmp_path / "runs"
    t_out = runs / "teacher_seed0"
    ref = tmp_path / "ref"
    ref.mkdir()
    d = fake_v2_data(0, {})
    (ref / "metrics.json").write_text(json.dumps({"n_val": 16, "input_scale": d["input_scale"],
                                                  "members": {"shallow": {"best_val_loss": 0.0}}}))
    cfg = _v4_cfg(tmp_path, lr_reference=str(ref))  # unbeatable reference: shallow falls back to lr 1e-3
    lrs = []
    real_fit = train_teacher.fit
    monkeypatch.setattr(train_teacher, "fit", lambda *a, **k: lrs.append(k["lr"]) or real_fit(*a, **k))
    common = ["--config", str(cfg), "--epochs", "1", "--seed", "0", "--out", str(t_out)]
    for rnd in ("0", "1"):
        for m in ("eegnet", "shallow", "deep", "riemann"):
            r = train_teacher.run(train_teacher.parse_args([*common, "--round", rnd, "--members", m]))
            assert r["partial"]
        r = train_teacher.run(train_teacher.parse_args([*common, "--round", rnd, "--reuse-members"]))
        assert bool(r.get("partial")) == (rnd == "0")
    tm = json.loads((t_out / "metrics.json").read_text())
    r0 = tm["round0"]["members"]["shallow"]
    assert r0["lr_choice"]["tried_lr"] == pytest.approx(3e-4) and r0["lr_choice"]["kept_lr"] == pytest.approx(1e-3)
    assert tm["members"]["shallow"]["lr"] == pytest.approx(1e-3)  # round 1 reuses the choice
    assert lrs == [1e-3, 3e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    assert guarded == [1]
    results = tmp_path / "results" / "results_v4.md"
    run_v2.main(["--skip-train", "--out", str(runs), "--results", str(results), "--config", str(cfg),
                 "--seeds", "0", "--students", "wide32", "--compare", str(tmp_path / "none")])
    text = results.read_text()
    for label in ("v4 results", "Riemann TS + LR", "Self-training on the unlabelled target session",
                  "Teachers before and after self-training", "0 (before)", "1 (after)", "Per-member KPIs",
                  "Per-member learning rate", "→ 0.001"):
        assert label in text, label


def test_v2_v3_configs_keep_three_members_and_v4_matches_v2_pipeline():
    from src.train_teacher import MEMBERS, member_names

    v2 = yaml.safe_load((ROOT / "configs" / "v2.yaml").read_text())
    v3 = yaml.safe_load((ROOT / "configs" / "v3.yaml").read_text())
    v4 = yaml.safe_load((ROOT / "configs" / "v4.yaml").read_text())
    assert member_names(v2) == member_names(v3) == MEMBERS == ("eegnet", "shallow", "deep")
    assert not {"self_training", "members", "riemann"} & (set(v2) | set(v3))
    assert "target_unlabeled" not in v2["distill"] and "lr_member" not in v2["training"]
    for k in ("fs_model", "window", "window_start", "train_window", "crop_offset", "filter", "reference", "ea_eps",
              "datasets", "test", "student"):
        assert v4[k] == v2[k], k
    assert {k: v for k, v in v4["training"].items() if k not in ("lr_member", "lr_reference")} == v2["training"]
    assert {k: v for k, v in v4["distill"].items() if k != "target_unlabeled"} == v2["distill"]
    assert v4["members"] == ["eegnet", "shallow", "deep", "riemann"] and v4["self_training"]["session"] == 1
    assert v4["self_training"]["min_conf"] == 0.8 and v4["training"]["lr_member"] == {"shallow": 0.0003}


# ----------------------------------------------------------------------------- v4soft
def test_soft_kl_loss_and_weights():
    from src.train_soft import soft_kl_loss, target_weights

    z = torch.randn(6, 2)
    y = torch.tensor([0, 1, 1, 0, 1, 0])
    onehot = torch.eye(2)[y]
    assert torch.allclose(soft_kl_loss(z, onehot, torch.ones(6)), torch.nn.functional.cross_entropy(z, y))
    p = torch.softmax(torch.randn(6, 2), 1)
    assert soft_kl_loss(z, p, torch.zeros(6)).item() == 0.0
    assert abs(soft_kl_loss(torch.log(p), p, torch.ones(6)).item()) < 1e-6
    np.testing.assert_allclose(target_weights(np.array([[0.5, 0.5], [0.1, 0.9], [1.0, 0.0]])), [0.0, 0.8, 1.0])


def test_soft_round_reuses_round0_finetunes_and_never_reads_labels(guarded, monkeypatch, tmp_path):
    from src import finetune_student, run_v2, train_soft, train_teacher
    from src.models import riemann

    hard_runs = tmp_path / "hard"
    h_out = hard_runs / "teacher_seed0"
    train_teacher.run(train_teacher.parse_args(["--config", str(_v4_cfg(tmp_path)), "--epochs", "1", "--seed", "0",
                                                "--out", str(h_out)]))
    assert guarded == [1]
    cfg = yaml.safe_load((ROOT / "configs" / "v4soft.yaml").read_text())
    cfg["self_training"]["reuse_round0"] = str(h_out)
    cfg["self_training"]["finetune"] = {"lr": 1e-4, "max_epochs": 2, "patience": 15}
    cfg["report"]["three_way"]["hard"] = str(hard_runs)
    cfg["training"].pop("lr_reference", None)
    cfg_path = tmp_path / "v4soft.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(train_teacher, "fit", lambda *a, **k: pytest.fail("round 0 must be reused, not retrained"))
    soft_calls = []
    real_soft = train_soft.fit_soft

    def spy_soft(model, X, p, w, *a, **k):
        name = type(model).__name__
        ref = {"EEGNet": "eegnet", "ShallowConvNet": "shallow", "DeepConvNet": "deep"}[name]
        start = torch.load(h_out / "members_round0" / f"{ref}.pt", weights_only=True)
        assert all(torch.equal(v, start[kk]) for kk, v in model.state_dict().items())  # fine-tune from round 0
        soft_calls.append((len(X), np.array(p), np.array(w), k["lr"], k["max_epochs"]))
        return real_soft(model, X, p, w, *a, **k)

    monkeypatch.setattr(train_soft, "fit_soft", spy_soft)
    weights_seen = []
    real_rfit = riemann.RiemannMember.fit
    monkeypatch.setattr(riemann.RiemannMember, "fit",
                        lambda self, X, y, sample_weight=None: weights_seen.append(sample_weight) or
                        real_rfit(self, X, y, sample_weight=sample_weight))
    s_out = tmp_path / "soft" / "teacher_seed0"
    common = ["--config", str(cfg_path), "--epochs", "2", "--seed", "0", "--out", str(s_out)]
    train_teacher.run(train_teacher.parse_args([*common, "--round", "0", "--reuse-members"]))
    for f in ("eegnet.pt", "riemann.pkl"):
        assert (s_out / "members_round0" / f).read_bytes() == (h_out / "members_round0" / f).read_bytes()
    tm = train_teacher.run(train_teacher.parse_args([*common, "--round", "1"]))
    assert guarded == [1, 1]  # once per run, by the reporting function only
    n_all = 64 + 9 * N_TARGET
    assert [c[0] for c in soft_calls] == [n_all] * 3 and all(c[3] == 1e-4 and c[4] == 2 for c in soft_calls)
    p, w = soft_calls[0][1], soft_calls[0][2]
    pl = np.load(s_out / "pseudo_labels.npz")
    p_tgt = np.concatenate([pl[f"p_{s}"] for s in range(1, 10)])
    np.testing.assert_allclose(p[64:], p_tgt, rtol=1e-6)
    np.testing.assert_allclose(w[64:], 2 * np.abs(p_tgt[:, 1] - 0.5), rtol=1e-5)
    assert np.all(w[:64] == 1) and set(np.unique(p[:64])) <= {0.0, 1.0}
    assert len(weights_seen) == 1 and len(weights_seen[0]) == n_all
    assert tm["self_training"]["counts"]["mode"] == "soft" and tm["self_training"]["n_train_round1"] == n_all
    assert 0 <= tm["self_training"]["agreement_report_only"]["weighted_agreement_all"] <= 1
    assert tm["members"]["deep"]["finetune_from_round0"] and tm["members"]["deep"]["lr"] == 1e-4

    # KD-only student fine-tune from a (fake) v2 student, last epoch kept
    init = tmp_path / "v2student"
    from src.distill import build_student

    init.mkdir()
    torch.save(build_student("wide32", cfg).state_dict(), init / "model.pt")
    (init / "student.json").write_text(json.dumps(cfg["student"]["wide32"]))
    (init / "params.json").write_text(json.dumps({"input_scale": fake_v2_data(0, {})["input_scale"]}))
    ft_out = tmp_path / "soft" / "student_wide32_ft_seed0"
    fm = finetune_student.run(finetune_student.parse_args(
        ["--config", str(cfg_path), "--init", str(init), "--teacher", str(s_out), "--out", str(ft_out),
         "--epochs", "2"]))
    assert fm["n_unlabeled"] == 9 * N_TARGET + 9 * 24 and fm["epochs_run"] == 2 and fm["best_epoch"] == 1
    assert guarded == [1, 1]
    pred = np.load(ft_out / "predictions.npz")
    np.savez(ft_out / "int8_predictions.npz", **{k: pred[k] for k in pred.files})
    results = tmp_path / "results" / "results_v4soft.md"
    run_v2.main(["--skip-train", "--out", str(tmp_path / "soft"), "--results", str(results), "--config",
                 str(cfg_path), "--seeds", "0", "--students", "wide32_ft", "--compare", str(tmp_path / "none")])
    text = results.read_text()
    for label in ("v4soft results", "Soft mode", "Weighted agreement", "Round 0 vs hard pseudo-labels vs soft",
                  "hard pseudo-labels", "fine-tuned v2 student"):
        assert label in text, label
