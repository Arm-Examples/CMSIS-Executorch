# ML: how the on-board models were trained

Two models were trained for the Alif E8. Both are small TinyEEGNet-type CNNs, quantized to int8 and run entirely on the Ethos-U85 NPU through ExecuTorch.

<p align="center"><picture>
<source media="(prefers-color-scheme: dark)" srcset="../docs/media/ml-dark.svg">
<img src="../docs/media/ml.svg" alt="Model A: BCI IV 2a subject 1 trains a TinyEEGNet, deployed and tested on subject 1. Model B: Cho2017 and Lee2019 train four teachers, distilled into a TinyEEGNet-wide student, deployed and tested on all 9 BCI IV 2a subjects" width="100%">
</picture></p>

| | **A. Subject-specific model** | **B. Generic model** (teachers → student, the firmware default) |
|---|---|---|
| Idea | One model per person, trained on that person's labeled session | One model for everyone, trained on other datasets; a per-subject alignment matrix adapts it |
| Training data | BCI IV 2a, subject 1, session 1 (115 train, 29 validation trials) | Cho2017 (52 subjects) + Lee2019_MI (subjects 1–20); unlabeled BCI IV 2a session 1 for fine-tuning |
| Test data | BCI IV 2a, subject 1, session 2 (144 trials) | BCI IV 2a, all 9 subjects, session 2 trials 25–144 (1,080 trials) |
| Network | TinyEEGNet, 1,938 parameters, 250 Hz, 22 × 500 input | TinyEEGNet-wide, 3,426 parameters, 125 Hz, 22 × 256 input |
| `.pte` | 13,072 B | 11,456 B (release `v0.5-distill`) |
| Code | [`subject_model/`](subject_model/) | [`distillation/`](distillation/) |

## Datasets

All three are public and download through [MOABB](https://moabb.neurotechx.com). No data is stored in this repository.

| Dataset | MOABB name | Subjects | Channels, rate | Used for | Size |
|---|---|---|---|---|---|
| [BCI Competition IV 2a](https://www.bbci.de/competition/iv/#dataset2a) | `BNCI2014_001` | 9 | 22 EEG, 250 Hz | A: train + test. B: test only (labels never used for training) and the demo replay | ~0.7 GB |
| [Cho et al. 2017](https://doi.org/10.5524/100295) | `Cho2017` | 52 | 64 EEG, 512 Hz (22 used) | B: teacher and student training | ~10.5 GB |
| [Lee et al. 2019](https://doi.org/10.5524/100542) | `Lee2019_MI` | 54 (1–20 used) | 62 EEG, 1000 Hz (22 used, FCz interpolated) | B: teacher and student training | ~24 GB |

Only the left-hand and right-hand imagery classes are used.

## Model A: subject-specific model

1. **Download** BCI IV 2a (`src/data/download.py`).
2. **Preprocess** (`src/data/preprocess.py`): 22 channels, causal 7–30 Hz Butterworth, window cue + 1 s to cue + 3 s (22 × 500). Per-channel z-score fitted on the 115 training trials only.
3. **Split** session 1 into 80 % train / 20 % validation, stratified, seed 42 (`src/train_subject.py`). Session 2 is loaded only after the checkpoint is chosen.
4. **Train** TinyEEGNet with cross-entropy, Adam lr 1e-3, batch 64, dropout 0.5, early stopping on validation loss (best epoch 110).
5. **Export** int8 with ExecuTorch 1.5.1 and Vela for `ethos-u85-256`, Alif `Ethos_U85_SRAM_MRAM` memory profile (`src/export.py`).

```bash
cd ml/subject_model && pip install -r requirements.txt
python -m src.data.download --subjects 1
python -m src.train_subject --config configs/subject1/baseline.yaml --run-name <new-run>
python -m src.data.subject_export_data --run artifacts/<new-run> --out artifacts/<new-export>
python -m src.export --workdir artifacts/<new-export> --checkpoint-dir artifacts/<new-run> \
    --config-ini artifacts/<new-export>/ensemble_vela.ini --system-config Ethos_U85_SRAM_MRAM
python -m pytest -q                                  # 14 tests
```
`src/export.py` needs an ExecuTorch 1.5.1 environment with the Arm backend (`--official-repo`, see the Arm CMSIS-Executorch starter). Copy the Alif `ensemble_vela.ini` into the export folder first.

## Model B: generic model, four teachers distilled into one student

1. **Download** Cho2017 and Lee2019_MI once with MOABB (below); BCI IV 2a through `src/data/download.py`. The loaders never download by themselves.
2. **Preprocess every recording the same way** (`src/data/external.py`, `cache125.py`, `splits125.py`): the 22 BCI IV 2a channels in 2a order (missing ones interpolated), common average reference, causal 7–30 Hz, decimate to 125 Hz, window cue + 0.496 s to cue + 2.544 s (22 × 256). **Euclidean alignment** per subject and session (`src/data/align.py`), then one global scale. A test subject's alignment matrix comes from its first 24 session 2 trials, unlabeled.
3. **Step 1: train 4 teachers** (`src/train_teacher.py`, `src/models/eegnet.py`, `convnets.py`, `riemann.py`): EEGNet, ShallowConvNet, DeepConvNet and a Riemannian classifier (covariance → tangent space → logistic regression). Ensemble = mean of the 4 softmax outputs.
4. **Step 2: distill the student**, in two passes. First (`src/distill.py`), TinyEEGNet-wide learns the soft labels of the three CNN teachers: loss 0.3 × cross-entropy + 0.7 × KL at temperature 4, with noise and channel-dropout augmentation. Then (`src/finetune_student.py`) it is fine-tuned for 30 epochs on the soft labels of all four teachers, computed on unlabeled BCI IV 2a windows (session 1 plus the 24 calibration trials per subject). No BCI IV 2a labels are used anywhere.
5. **Quantize and export** (`src/quantize.py`, `src/export.py`): int8 post-training quantization, ExecuTorch 1.5.1, Vela `ethos-u85-256`. Writes the bundle the firmware uses: `.pte`, `preprocess_params.json` (filter, the 9 alignment matrices, scales), test vectors.

```bash
cd ml/distillation && pip install -r requirements.txt
# 1. Data, once (~35 GB into ~/mne_data)
python -m src.data.download --subjects all
MOABB_DOWNLOAD_PROVIDER=upstream python -c "
from moabb.datasets import Cho2017, Lee2019_MI
for s in range(1, 53): Cho2017().data_path(s)
for s in range(1, 21): Lee2019_MI().data_path(s)"
# 3. Step 1 + first distillation pass: 3 CNN teachers, then students, seeds 0-2 -> runs/v2
python -m src.run_v2 --jobs 1 --seeds 0 1 2 --students wide32 control
# 3. Step 1 with the 4th (Riemannian) teacher and soft targets on unlabeled 2a -> runs/v4, runs/v4soft
python -m src.run_v2 --config configs/v4.yaml --seeds 0 --students wide32 control --out runs/v4 \
    --results results/results_v4.md --device mps --jobs 1 --teacher-parallel
python -m src.run_v2 --config configs/v4soft.yaml --seeds 0 --students wide32 control --out runs/v4soft \
    --results results/results_v4soft.md --device mps --jobs 1 --teacher-parallel
# 4. Second distillation pass: fine-tune the seed-0 student on the 4-teacher soft labels
python -m src.finetune_student --config configs/v4soft.yaml --init runs/v2/student_wide32_seed0 \
    --teacher runs/v4soft/teacher_seed0 --out runs/v4soft/student_wide32_ft_seed0 --seed 0 --device mps
# 5. int8 check and export (subject 3 test vectors)
python -m src.quantize --run runs/v4soft/student_wide32_ft_seed0
python -m src.export --checkpoint runs/v4soft/student_wide32_ft_seed0/model.pt \
    --run-dir runs/v4soft/student_wide32_ft_seed0 --subject 3 --out artifacts/v4/tinyeegnet_int8.pte
python -m pytest -q                                  # 64 passed, 6 skipped without the datasets
```
Use `--device cuda` or `cpu` off Apple Silicon. Results tables land in `results/`.

## Metric checks

| Check | A. Subject-specific | B. Generic |
|---|---|---|
| Accuracy on its test set | 79.9 % (float, 144 trials) | **74.6 ± 14.8 %** int8 (mean over 9 subjects, 3 seeds); 74.7 % float |
| Accuracy on BCI IV 2a subject 1 | 79.9 % (all 144 session 2 trials) | 88.6 % int8 (trials 25–144) |
| Cohen's kappa | 0.597 | 0.491 |
| Macro F1 | 0.795 | 0.739 |
| ROC-AUC / AUPRC | not reported | 0.787 / 0.796 |
| Confusion (rows true L, R; columns predicted L, R) | [[48, 24], [5, 67]] | per subject in the source repo, `results/confusion_v4soft/` |
| Float → int8 | Same labels on the 29 validation trials; max logit difference 0.277 | −0.1 points accuracy |
| Emulator (Corstone-320 FVP) | 144/144 predictions agree with host int8, max logit error 0.053 (one int8 step) | 20/20 test vectors, max logit error 0.025, 128,056 NPU cycles (0.32 ms at 400 MHz) |
| Physical Alif E8 | **160 windows, board output = FVP output on every one**; 114/144 distinct trials correct (79.2 %) | Not run yet |

Kappa and F1 for A are computed from its confusion matrix. A and B use different test sets, so only the subject 1 row compares them on the same person. Sources: `results/subject1-seed42-20261016/` on the source branch (A); `results/results_v4soft.md` and `docs/v4_handoff.md` in Reproduce_Baselines (B).

## Failure cases

**A. Subject-specific model**
- **Right-hand bias.** Left trials are recalled 66.7 % of the time, right trials 93.1 %: 24 of 72 left trials are called right.
- **Overfitting.** 94.8 % on training versus 75.9 % on validation; the validation set is only 29 trials, so model selection is noisy.
- **One person only.** Every new user needs a labeled recording session and a new model.

**B. Generic model**
- **Weak subjects.** A06 56.4 %, A09 58.3 % and A02 63.9 % are close to chance; A03 98.6 % and A08 94.7 % are near perfect. The spread (± 14.8) comes from the EEG itself: the weak subjects show little or two-sided motor rhythm change. Quantization is not the cause (float and int8 differ by 0.1 points).
- **Optimistic estimate.** The shipped variant was chosen among 5 candidates by test accuracy, so 74.6 % is slightly optimistic. The gain over the plain distilled student (+1.1 points) holds on 3 of 3 seeds.
- **Not yet on the board.** It passes the emulator and the bit-exact replay checks, but has not run on a physical E8.
