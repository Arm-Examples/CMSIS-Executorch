# Source

Snapshot of [ArmExecuTorchEEG/Reproduce_Baselines](https://github.com/ArmExecuTorchEEG/Reproduce_Baselines) `main` at commit `ad61c79` (release `v0.5-distill`): `src/`, `configs/`, `tests/`, `requirements.txt`.

Changes from the source commit:
- Removed `src/export_fused.py` and `tests/test_fused.py` (per-subject fused export, not used by the firmware).
- Added `pyriemann==0.12` to `requirements.txt` (imported by `src/models/riemann.py`, missing from the source pin list).

Test suite in this folder: `python -m pytest -q` gives 64 passed, 6 skipped (skips need the downloaded datasets).
