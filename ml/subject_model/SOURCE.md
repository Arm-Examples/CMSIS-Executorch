# Source

Snapshot of [ArmExecuTorchEEG/Reproduce_Baselines](https://github.com/ArmExecuTorchEEG/Reproduce_Baselines) branch `feature/model2-random-eeg-trials` at commit `f9c40dd` (author: philippinech3025), training part only: `src/data/`, `src/models/tinyeegnet.py`, `src/train.py`, `src/train_subject.py`, `src/eval.py`, `src/export.py`, `src/experiments/dropout_cv.py`, `configs/`, four test files, `requirements.txt`. Files are unchanged.

Not copied: checkpoints and `.pte` files (`models_ready_for_alif_e8/`), FVP and UART streaming tools (`src/fvp/`, `tests/streaming/`, `tools/model2_random_trials/`), `src/package_models.py`.

Test suite in this folder: `python -m pytest -q` gives 14 passed.
