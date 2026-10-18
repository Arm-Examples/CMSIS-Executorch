"""v2 models, distillation pieces and an end-to-end smoke on a fake ``v2_data`` (no datasets needed)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

WINDOW, TRAIN_WINDOW, OFFSET = 256, 306, 25
STUDENTS = {
    "wide32": {"F1": 16, "D": 2, "F2": 32, "kern_t": 32, "kern_s": 16},
    "control": {"F1": 8, "D": 2, "F2": 16, "kern_t": 32, "kern_s": 16},
    "wide64": {"F1": 16, "D": 2, "F2": 32, "kern_t": 64, "kern_s": 16},
}
# (params, MACs per (1, 1, 22, 256) window), plan.md section 4
EXPECTED = {
    "eegnet": (1714, 3_018_848),
    "shallow": (38_242, 11_382_560),
    "deep": (146_302, 5_508_400),
    "wide32": (3426, 3_176_128),
    "control": (1458, None),
    "wide64": (3938, None),
}


def v2_cfg() -> dict:
    """Inline copy of the configs/v2.yaml keys (block A owns the real file)."""
    return {
        "fs_model": 125, "window": WINDOW, "window_start": 62, "crop_offset": OFFSET, "train_window": TRAIN_WINDOW,
        "filter": {"low": 7.0, "high": 30.0, "order": 4}, "ea_eps": 1e-3,
        "datasets": {"Cho2017": {"subjects": list(range(1, 53)), "exclude": []},
                     "Lee2019_MI": {"subjects": list(range(1, 21))}},
        "test": {"dataset": "BNCI2014_001", "subjects": list(range(1, 10)), "calib_trials": 24},
        "training": {"lr": 0.001, "batch_size": 64, "max_epochs": 300, "patience": 30, "dropout": 0.25},
        "distill": {"alpha": 0.7, "temperature": 4.0, "noise_std": 0.05, "channel_dropout": 0.1,
                    "crop_jitter": 25},
        "seeds": [0, 1, 2],
        "student": STUDENTS,
    }


def build(name: str) -> torch.nn.Module:
    from src.distill import build_student
    from src.train_teacher import build_member

    return build_member(name, v2_cfg()) if name in ("eegnet", "shallow", "deep") else build_student(name, v2_cfg())


# ----------------------------------------------------------------------------- architectures
@pytest.mark.parametrize("name", list(EXPECTED))
def test_param_counts_macs_and_forward_shape(name: str) -> None:
    from src.run_v2 import count_macs

    model = build(name).eval()
    params, macs = EXPECTED[name]
    assert sum(p.numel() for p in model.parameters()) == params
    if macs is not None:
        assert count_macs(model, (1, 1, 22, WINDOW)) == macs
    with torch.no_grad():
        out = model(torch.randn(5, 1, 22, WINDOW))
    assert out.shape == (5, 2)
    assert torch.isfinite(out).all()


def test_get_model_reads_student_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.models import tinyeegnet

    monkeypatch.setattr(tinyeegnet, "FLOAT_CHECKPOINT", tmp_path / "absent.pt")
    monkeypatch.setattr(tinyeegnet, "STUDENT_CONFIG", tmp_path / "student.json")
    default = tinyeegnet.get_model()
    assert sum(p.numel() for p in default.parameters()) == 1938  # v0.1 widths without student.json
    (tmp_path / "student.json").write_text(json.dumps(STUDENTS["wide32"]))
    wide = tinyeegnet.get_model((1, 1, 22, WINDOW))
    assert sum(p.numel() for p in wide.parameters()) == 3426
    assert wide.conv_temporal.kernel_size == (1, 32)


# ----------------------------------------------------------------------------- distillation pieces
def test_kd_loss_reduces_to_ce_and_scaled_kl() -> None:
    from src.distill import kd_loss

    z = torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.5, -0.5]])
    p = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
    y = torch.tensor([0, 1, 1])
    T = 2.0
    assert torch.isclose(kd_loss(z, p, y, 0.0, T), F.cross_entropy(z, y))
    zt = z.numpy() / T
    ps = np.exp(zt) / np.exp(zt).sum(1, keepdims=True)
    kl = (p.numpy() * np.log(p.numpy() / ps)).sum(1)
    assert np.isclose(kd_loss(z, p, y, 1.0, T).item(), T * T * kl.mean(), rtol=1e-5)
    # unlabelled trials (y = -1) contribute the KD term only
    y_unl = torch.tensor([0, -1, 1])
    expected = (0.5 * F.cross_entropy(z, y.clamp_min(0), reduction="none") * torch.tensor([1.0, 0.0, 1.0])
                + 0.5 * T * T * torch.as_tensor(kl, dtype=torch.float32)).mean()
    assert torch.isclose(kd_loss(z, p, y_unl, 0.5, T), expected, rtol=1e-5)


def test_crop_and_augment_shapes_and_canonical_window() -> None:
    from src.distill import augment, crop
    from src.train_teacher import canonical

    X = torch.randn(4, 1, 22, TRAIN_WINDOW)
    canon = crop(X, torch.full((4,), OFFSET), WINDOW)
    assert torch.equal(canon, X[..., OFFSET : OFFSET + WINDOW])
    assert np.array_equal(canonical(X.numpy()[:, 0], v2_cfg()), X.numpy()[:, 0, :, 25:281])
    offsets = torch.tensor([0, 7, 25, TRAIN_WINDOW - WINDOW])
    out = crop(X, offsets, WINDOW)
    assert out.shape == (4, 1, 22, WINDOW)
    for i, o in enumerate(offsets.tolist()):
        assert torch.equal(out[i], X[i, ..., o : o + WINDOW])
    gen = torch.Generator().manual_seed(0)
    aug = augment(canon, gen, 0.05, 0.1)
    assert aug.shape == canon.shape
    assert torch.equal(augment(canon, gen, 0.0, 0.0), canon)
    assert torch.count_nonzero(augment(canon, gen, 0.0, 1.0)) == 0


# ----------------------------------------------------------------------------- distillation loop
class Recorder(torch.nn.Module):
    """Linear stub that logs ``(tag_train|tag_eval, input)`` for every forward call."""

    def __init__(self, events: list, tag: str) -> None:
        super().__init__()
        self.events, self.tag = events, tag
        self.fc = torch.nn.Linear(22 * WINDOW, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.events is not None:
            self.events.append((f"{self.tag}_{'train' if self.training else 'eval'}", x.detach().clone()))
        return self.fc(x.flatten(1))


def loop_inputs(n_train: int = 40, n_calib: int = 8, n_val: int = 10, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X_train = torch.randn((n_train, 1, 22, TRAIN_WINDOW), generator=g)
    X_calib = torch.randn((n_calib, 1, 22, WINDOW), generator=g)
    X_val = torch.randn((n_val, 1, 22, TRAIN_WINDOW), generator=g)  # long windows: loop must crop at 25
    return X_train, np.arange(n_train) % 2, X_calib, X_val, np.arange(n_val) % 2


def fit_kwargs(**over) -> dict:
    kw = dict(window=WINDOW, crop_offset=OFFSET, alpha=0.7, T=4.0, noise_std=0.05, channel_dropout=0.1, lr=1e-3,
              batch_size=16, max_epochs=1, patience=30, seed=0, crop_jitter=25, log=lambda *_: None)
    kw.update(over)
    return kw


def test_random_crop_offsets_cover_and_stay_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import distill

    assert distill.offset_range(TRAIN_WINDOW, WINDOW, OFFSET, 25) == (0, 50)
    assert distill.offset_range(TRAIN_WINDOW, WINDOW, OFFSET, None) == (0, 50)
    assert distill.offset_range(TRAIN_WINDOW, WINDOW, OFFSET, 100) == (0, 50)
    seen: list[int] = []
    real_crop = distill.crop

    def spy(x, offsets, window):
        seen.extend(offsets.tolist())
        return real_crop(x, offsets, window)

    monkeypatch.setattr(distill, "crop", spy)
    X_train, y_train, X_calib, X_val, y_val = loop_inputs(n_train=200)
    X_val = X_val[..., OFFSET : OFFSET + WINDOW]  # canonical already: spy sees training offsets only
    distill.distill_fit(Recorder(None, "student"), [Recorder(None, "teacher").eval()], X_train, y_train, X_calib,
                        X_val, y_val, **fit_kwargs(max_epochs=6, batch_size=64))
    assert len(seen) == 1200
    assert min(seen) == 0 and max(seen) == 50
    assert all(0 <= o <= 50 for o in seen)


def test_teacher_scores_the_students_augmented_batch_and_validation_is_clean() -> None:
    from src.distill import distill_fit

    events: list = []
    X_train, y_train, X_calib, X_val, y_val = loop_inputs()
    distill_fit(Recorder(events, "student"), [Recorder(events, "teacher").eval()], X_train, y_train, X_calib,
                X_val, y_val, **fit_kwargs(noise_std=1.0, channel_dropout=0.5))
    canon_val = X_val[..., OFFSET : OFFSET + WINDOW]
    # validation soft labels: teacher on the canonical crop, no noise or dropout
    assert events[0][0] == "teacher_eval" and torch.equal(events[0][1], canon_val)
    train_steps = [i for i, (tag, _) in enumerate(events) if tag == "student_train"]
    assert len(train_steps) == 3  # (40 + 8) / 16 batches
    for i in train_steps:
        assert events[i - 1][0] == "teacher_eval"
        assert torch.equal(events[i - 1][1], events[i][1])  # same augmented tensor
    zero_channels = sum(int((x.abs().sum(dim=(1, 3)) == 0).sum()) for tag, x in events if tag == "student_train")
    assert zero_channels > 0  # channel dropout was applied to the training stream
    # student validation pass: canonical crop, no augmentation
    val_inputs = [x for tag, x in events if tag == "student_eval"]
    assert len(val_inputs) == 1 and torch.equal(val_inputs[0], canon_val)


def test_checkpoint_and_patience_follow_val_kl_not_combined_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import distill

    def scripted(kls, losses, snapshots):
        it = iter(zip(kls, losses))

        def validate(student, *args, **kwargs):
            kl, loss = next(it)
            snapshots.append({k: v.detach().clone() for k, v in student.state_dict().items()})
            return {"val_loss": loss, "val_kl": kl, "val_ce": loss, "val_acc": 0.5, "val_teacher_agreement": 1.0}
        return validate

    X_train, y_train, X_calib, X_val, y_val = loop_inputs()
    # combined loss is lowest at epoch 1, KL at epoch 3: the kept checkpoint must be epoch 3
    snaps: list = []
    monkeypatch.setattr(distill, "validate", scripted([0.5, 0.3, 0.4, 0.2, 0.6], [0.9, 0.1, 0.8, 0.7, 0.95], snaps))
    student = Recorder(None, "student")
    info = distill.distill_fit(student, [Recorder(None, "teacher").eval()], X_train, y_train, X_calib, X_val, y_val,
                               **fit_kwargs(max_epochs=5, patience=10))
    assert (info["best_epoch"], info["best_val_kl"], info["best_val_loss"]) == (3, 0.2, 0.7)
    state = student.state_dict()
    assert all(torch.equal(state[k], snaps[3][k]) for k in state)
    assert not all(torch.equal(state[k], snaps[1][k]) for k in state)
    # patience counts on KL: KL stops improving after epoch 1 while the combined loss keeps falling
    snaps = []
    monkeypatch.setattr(distill, "validate", scripted([0.5, 0.4, 0.45, 0.46, 0.47, 0.48],
                                                      [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], snaps))
    info = distill.distill_fit(Recorder(None, "student"), [Recorder(None, "teacher").eval()], X_train, y_train,
                               X_calib, X_val, y_val, **fit_kwargs(max_epochs=6, patience=2))
    assert info["best_epoch"] == 1 and info["epochs_run"] == 4


# ----------------------------------------------------------------------------- end-to-end smoke
def fake_v2_data(seed: int, cfg: dict) -> dict:
    """Random data with the block-A contract shapes (tiny sizes) and a weak class signal."""
    rng = np.random.default_rng(seed)

    def trials(n: int, T: int) -> tuple[np.ndarray, np.ndarray]:
        y = np.arange(n) % 2
        X = rng.standard_normal((n, 22, T)).astype(np.float32) * 5.0
        X[y == 1, 7] *= 2.0  # C3 power differs by class
        return X, y.astype(np.int64)

    X_train, y_train = trials(48, TRAIN_WINDOW)
    subjects = list(range(1, 10))
    return {
        "train": (X_train, y_train),
        "val": trials(16, TRAIN_WINDOW),
        "calib": {s: trials(24, WINDOW)[0] for s in subjects},
        "test": {s: trials(12, WINDOW) for s in subjects},
        "ea": {s: np.eye(22) for s in subjects},
        "input_scale": float(X_train.std()),
        "meta": {"n_train_subject_sessions": 2, "sources": {"Cho2017": 1, "Lee2019_MI": 1}, "fs": 125,
                 "window": WINDOW, "crop_offset": OFFSET},
    }


@pytest.fixture
def fake_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    try:
        import src.data.splits125 as mod
    except Exception:  # block A not landed (or broken): stand-in module
        import src.data

        mod = types.ModuleType("src.data.splits125")
        monkeypatch.setitem(sys.modules, "src.data.splits125", mod)
        monkeypatch.setattr(src.data, "splits125", mod, raising=False)
    monkeypatch.setattr(mod, "v2_data", fake_v2_data, raising=False)
    cfg_path = tmp_path / "v2.yaml"
    cfg_path.write_text(yaml.safe_dump(v2_cfg()))
    return cfg_path


def test_teacher_distill_report_smoke(fake_data: Path, tmp_path: Path) -> None:
    from src import distill, run_v2, train_teacher

    runs = tmp_path / "runs"
    common = ["--config", str(fake_data), "--epochs", "2", "--seed", "0"]
    t_out = runs / "teacher_seed0"
    tm = train_teacher.run(train_teacher.parse_args([*common, "--out", str(t_out)]))
    for name in ("eegnet", "shallow", "deep"):
        assert (t_out / "members" / f"{name}.pt").is_file()
        assert tm["members"][name]["epochs_run"] == 2
    z = np.load(t_out / "teacher_logits.npz")
    assert z["deep_train"].shape == (48, 2) and z["eegnet_calib_3"].shape == (24, 2) and z["shallow_test_9"].shape == (12, 2)
    m = json.loads((t_out / "metrics.json").read_text())
    assert set(m["ensemble"]["test"]) == {str(s) for s in range(1, 10)}
    params = json.loads((t_out / "params.json").read_text())
    assert params["window_samples"] == WINDOW and params["crop_offset"] == OFFSET and len(params["ea_matrices"]) == 9

    s_out = runs / "student_wide32_seed0"
    sm = distill.run(distill.parse_args([*common, "--student", "wide32", "--teacher", str(t_out), "--out", str(s_out)]))
    assert sm["epochs_run"] == 2 and sm["params"] == 3426
    assert sm["best_val_kl"] == min(h["val_kl"] for h in json.loads((s_out / "history.json").read_text()))
    for f in ("model.pt", "student.json", "predictions.npz", "calibration.npz", "params.json", "metrics.json",
              "history.json", "train.log"):
        assert (s_out / f).is_file(), f
    assert json.loads((s_out / "student.json").read_text()) == STUDENTS["wide32"]
    cal = np.load(s_out / "calibration.npz")["x"]
    assert cal.shape == (48, 1, 22, WINDOW) and cal.dtype == np.float32
    pred = np.load(s_out / "predictions.npz")
    assert list(pred["subjects"]) == list(range(1, 10)) and pred["logits_3"].shape == (12, 2)

    # int8 rows appear when int8_predictions.npz exists (stand-in: the float predictions)
    np.savez(s_out / "int8_predictions.npz", **{k: pred[k] for k in pred.files})
    results = tmp_path / "results" / "results_v2.md"
    run_v2.main(["--skip-train", "--runs-dir", str(runs), "--results", str(results), "--config", str(fake_data),
                 "--seeds", "0", "--students", "wide32"])
    text = results.read_text()
    for label in ("EEGNet-ELU", "ShallowConvNet", "DeepConvNet", "Teacher ensemble", "Student wide32 | int8",
                  "Selected: **student wide32**", "lowest best validation KL", "A09"):
        assert label in text, label
    assert (tmp_path / "results" / "confusion_v2" / "A01.md").is_file()
    assert "best_val_kl" in json.loads((s_out / "metrics.json").read_text())
    if run_v2.v01_row():  # v0.1 context rows (float, int8) in both the summary and the per-subject table
        per_subject = text.split("## 3. Per-subject accuracy")[1].split("## 4.")[0]
        assert per_subject.count(run_v2.V01_LABEL) == 2 and "different protocol" in per_subject
