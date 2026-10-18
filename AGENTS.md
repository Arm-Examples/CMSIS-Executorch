# AGENTS.md — Edge_Deploy

Rules for people and coding agents (Claude Code, Codex, Cursor, Copilot) working in this repo. `CLAUDE.md` imports this file.
Project overview and repo map: https://github.com/ArmExecuTorchEEG

## Ownership
| Path | Owner | Others may |
|---|---|---|
| `firmware/src/{preprocess,uart_rx,app,clock}*` | TBD | PR |
| `firmware/src/{infer,imu,motion_gate,motion_config,leds,json_out,main,selftest}*`, `arm_embedded_module.*` | TBD | PR |
| `firmware/src/params_gen.h` | generated — nobody edits | `python3 tools/gen_params_h.py` |
| `firmware/board/`, `firmware/*.csolution.yml`, `firmware/*.cproject.yml` | TBD | PR |
| `firmware/ai_layer/`, `firmware/model/` | generated/fetched — nobody edits | `./tools/fetch_model.sh`, `firmware/create_ai_layer.py`. Exception to shared rule 4: `firmware/ai_layer/` (the release `.pte` as a C array) IS committed so a clone builds offline, as in Emulator_Deploy; regenerate only with `create_ai_layer.py --pte` |
| `firmware/tests/` (`generated/` by `host/make_vectors.py`) | TBD | PR |
| `host/bridge.py`, `host/drivers/`, `host/export_replay.py`, `host/common.py`, `host/listen.py` | TBD | PR |
| `host/pipeline_ref.py`, `host/int8_oracle.py`, `host/make_vectors.py`, `host/reference/`, `host/tests/` | TBD | PR |
| `host/demo_ui.py`, `host/measure.py`, `host/replay_accuracy.py` | TBD | PR |
| `tools/` | TBD | PR |
| `docs/`, `results/` | everyone | add only; `docs/demo_script.md` edits via PR |

## Contracts
- UART4, 921600 baud 8N1, little endian. Data frame `0xAA 0x55, uint8 n_ch (22), uint16 seq, int16[n_ch]` at 250 Hz, value = µV × 20.48 (decoded as count × 0.048828125), channels in `channel_order`.
- Control frame `0xAA 0x56, uint8 kind, uint8 arg, uint16 seq`: 1 SET_SUBJECT (1–9), 2 CUE (label 0/1/255, cue = data frame with this `seq`), 3 RESET, 4 FREE_RUN (0/1).
- Boot line `READY n_ch=22 model=tinyeegnet_int8 pte_bytes=11456 fs=250 fs_model=125 win=256 baud=921600`. Output JSON line, in this order: `t, seq, trial, subject, cls, raw_cls, conf, logits, lat_us, motion, label` (+ `imu:0` when the IMU is down). `#` lines are diagnostics.
- Host side: only `bridge.py` opens the serial port; it republishes over UDP `127.0.0.1:5555`.
- `firmware/model/params.json` = `preprocess_params.json` of Reproduce_Baselines `v0.5-distill` (sha256 `3d34f632…`; the previous model `v0.3-distill`, `87af6459…`, has the same contract); `firmware/src/params_gen.h` is generated from it.
- Board int8 windows = `host/pipeline_ref.py`, bit for bit.
- Release tag `v1.0-submission` = what judges see.

## Commands
Python: `Reproduce_Baselines/.venv/bin/python` has every dependency. `host/tests/conftest.py` sets `MOABB_DOWNLOAD_PROVIDER=upstream` and `MNE_DATA=~/mne_data`; set them yourself for `export_replay.py`, `make_vectors.py`, `replay_accuracy.py`.
```
./tools/fetch_model.sh                                  # v0.5-distill -> firmware/model/, sha256 checked
./tools/fetch_model.sh --local <bundle dir>             # local bundle (e.g. Reproduce_Baselines/artifacts/v4), no sha256 check
python3 tools/gen_params_h.py [--check]                 # firmware/src/params_gen.h
source ~/.vcpkg/vcpkg-init && (cd firmware && vcpkg-shell activate)   # toolchain; once per shell
cbuild firmware/edge_eeg.csolution.yml --active DevKit-E8 --packs
cbuild firmware/edge_eeg.csolution.yml --active SSE-320-U85 --packs
cbuild firmware/edge_eeg.csolution.yml --context edge_eeg.SelfTest+SSE-320-U85 --packs
#   FVP run (Docker): command in results/fvp_selftest.md, expect Test_result: PASS
make -C firmware/tests test                             # firmware C++ vs pipeline_ref
python -m pytest host/tests -q
python host/bridge.py --source replay --subject 3 --session 2 --board <port|sim>
python host/demo_ui.py --listen 5555 [--headless]
python host/measure.py --listen 5555 --trials 120 --out results/board_metrics.md
python host/replay_accuracy.py                          # results/replay_accuracy.md
```
Firmware in VS Code: CMSIS view on `firmware/edge_eeg.csolution.yml`, target-type `DevKit-E8`, SW4 on `UART4`, console at 921600.
This repo is the submission. Before tagging: flip to public, confirm every README link opens logged-out.

## Shared rules (identical in every ArmExecuTorchEEG repo)

### 1. Before touching anything
- Read `README.md` and this file. Then `git fetch && git status`.
- Work on a branch named `<yourname>/<topic>`. Never commit on `main`.
- One task per branch. One branch per agent session. Two sessions on one repo → two worktrees.

### 2. Ownership
See the table at the top of this file. Touching a path you don't own: open a PR, tag the owner, don't merge it yourself.
Root config, lockfiles, `requirements*.txt`, `.gitignore`, `README.md`, this file: PR only, never direct.

### 3. Contracts
Files other repos or people depend on are listed at the top of this file (shapes, file names, UART frame, params, release tags).
Changing one: update the README in the same PR and post in the team chat before merging. Never rename or delete a contract file.

### 4. Commits and pushes
- Commit only files you changed for this task. No `git add .` across unrelated files.
- Never commit: `.env`, tokens, datasets, `.pte`, checkpoints, generated `ai_layer/`, worktree dirs. If one slips in, say so; don't rewrite history.
- Message: `<area>: <what>` + one body line of why. Agents append their session link.
- Push your own branch freely. Never force-push a branch someone else has pulled. `--force-with-lease` only on your own branch.

### 5. Merging
- PR into `main`. Tests pass. Self-merge only when the diff touches only your own paths; otherwise one approval.
- Several PRs open: merge contracts → data → model → export → firmware → docs, one at a time.
- Before merge: `git rebase main` on your branch, re-run tests.

### 6. Conflicts
- Never resolve by discarding the other side. Same lines changed by both: stop, post the file and both diffs, let the two humans decide.
- `main` is never rewritten. Fix forward with a new commit.

### 7. Agents specifically
- Don't push, open PRs, merge, or delete branches unless the human in the session asked.
- Don't modify files outside this task's listed paths, even to "clean up". Don't reformat files you didn't change.
- Don't run `rm -rf`, `git reset --hard`, `git clean`, `git checkout -- .` without an explicit instruction naming the target.
- Two failed fix attempts → stop and report; don't widen the change.
- Final report: files changed, commands run, test output, branch name, open doubts.

### 8. Keep this file true
Changed a command, path, or contract? Update this file in the same PR. Stale rules are worse than none.
