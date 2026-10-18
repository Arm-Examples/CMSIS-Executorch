# Board metrics: physical Alif E8

Generic model (distilled student, `v0.5-distill`, `.pte` 11,456 B, sha256 `6caa8a98…`) running on the Alif Ensemble E8 DevKit (Cortex-M55 HP + Ethos-U85). Held-out EEG trials replayed from the laptop; predictions come from the board and are shown in the demo UI.

| Metric | Value |
|---|---|
| Trials | 70 (70 unique recordings, 7 artifact-flagged) |
| Accuracy | 77.1 % (54 correct, 16 incorrect) |
| Left hand | 24/32 correct (75 %) |
| Right hand | 30/38 correct (79 %) |
| Session duration | 3:07 min, including pauses |

Confusion matrix (rows = actual, columns = predicted):

| | Left | Right |
|---|---|---|
| **Left** | 24 | 8 |
| **Right** | 8 | 30 |

Model identity on the board: the `.pte` was read back from MRAM after flashing (SEGGER J-Link) and its sha256 matches the release.

Screenshot: [`docs/media/results_dashboard.png`](../docs/media/results_dashboard.png). Videos: [live demo](../docs/video/live_demo.mp4), [model to board verification](../docs/video/deploy_verification.mp4).

Not measured on the board yet: end-to-end latency, Ethos-U PMU cycle counts, IMU motion gate.
