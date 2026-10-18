# Multi-subject v4 numerical validation

The unchanged released student PTE is `wide32_ft`, seed 0, 3,426 parameters,
SHA-256 `6caa8a9876d3cd0c36b39b3d4474a0fe7dbdbfb32a4d6f80ed4776623a19e02e`.
No weights or saved quantization parameters were changed. No physical board was used.

## Completed FVP evidence

The production streaming parser and inference callback executed all **1,080 genuine
held-out windows**, in five fixture batches (240/240/240/240/120). All 2,160 INT8
output values agree **exactly** with the independently executed TOSA integer graph.
The unchanged 0.01 dequantized TOSA/FVP check has zero failures. Trial IDs,
sessions, requests, full input SHA-256, ACK/result CRCs, finite logits and classes
were verified. Each batch rejected both deliberately invalid frames before inference.
The additional 20 published vectors and 14 reordered/repeated cases also pass.
All reordered results are float32-bit-identical to their original full-pool results.
This is actual FVP execution of the release PTE, not fabricated serial responses.

| Metric over 1,080 unique trials | Result |
|---|---:|
| Exact TOSA/FVP integer agreement | 2,160 / 2,160 values |
| TOSA/FVP logit failures at 0.01 | 0 |
| Saved desktop Q/DQ logit failures at 0.01 | 498 |
| Maximum saved Q/DQ/FVP absolute logit difference | 0.049536943435668945 |
| Q/DQ/FVP trial maximum differences: 0 / 1 / 2 output codes | 582 / 497 / 1 |
| Saved Q/DQ/FVP classification agreement | 1,078 / 1,080 |
| FVP correct / saved desktop correct | 802 / 804 |

Output quantization scale is **0.024768464267253876**, zero point **−4**.
FVP accuracy on this previously evaluated pool is **74.2593%**; saved desktop
accuracy is **74.4444%**. These results are not an untouched final-test estimate.
Classification agreement and logit agreement remain separate metrics; the original
0.01 desktop diagnostic was not relaxed or hidden.

Both changed decisions are desktop ties, resolved by the integer backend as right:

- `S4/1test/run3/cue2753`, index 409: desktop `[0.0743053928, 0.0743053928]`,
  FVP `[0.0743053928, 0.0990738571]`; true class 0.
- `S7/1test/run2/cue50625`, index 756: desktop `[0.1238423213, 0.1238423213]`,
  FVP `[0.0990738571, 0.1238423213]`; true class 0.

## Reference reconstruction and limits

The local reconstructed Q/DQ graph matches every saved desktop class, but has
one-step logit discrepancies on 116 trials. All 6,082,560 input codes match the
PTE CPU quantizer. Original float windows versus reconstructed INT8-wire floats
produce the same discrepancy pattern, and native versus serialized Q/DQ graphs
agree exactly. This rules out those input/serialization paths as the cause.
The 20 published desktop logits reproduce exactly, demonstrating why that small
check alone was insufficient.

The diagnostic compiler reconstruction is not byte-identical to the release PTE:
metadata and delegate encoding differ. It is never installed as the model.
The reconstructed TOSA graph was initially provisional; full released-PTE FVP
execution now independently establishes exact output agreement over the entire
prepared pool and reordered cases. This is empirical coverage of the intended
pool, not proof that both compiler encodings agree on every possible input.
The precise origin of the saved/local desktop Q/DQ difference remains unresolved;
we do not attribute it to a particular rounding pass without a causal trace.

No trained model correction is indicated by these backend tests. Physical UART,
E8 memory/cache behavior, and Chrome-to-E8 operation with this wider model remain
untested until a separately authorized deployment. Do not replace actual returned
predictions with saved desktop labels, including the two ties.

## Reproduction and retained evidence

- `prepare_bundle.py`: checksum-verifies the release assets and pinned sources;
  downloads require explicit `--download` and never download EEG recordings.
- `prepare_dataset.py`: original preprocessing with saved parameters and verified
  MAT trial IDs; excludes Session 1 and adaptation trials 1–24 of Session 2.
- `build_reference.py`: diagnostic-only reconstruction; intentionally fails its
  byte-identity gate. Never use its diagnostic PTE as firmware.
- `evaluate_references.py`: executes Q/DQ and TOSA, retaining the desktop mismatch
  as a failing diagnostic. It is not the backend deployment gate.
- `prepare_firmware.py`, `compare_fvp.py`, `summarize_validation.py`: bounded FVP
  fixtures, actual binary-response comparison and full-pool/repeat consolidation.

Summary: `validation/numerical-status.json`. Earlier local reconstruction evidence:
`validation/desktop-reconstruction-diagnostic.json`. Complete logs, expected
manifests, comparisons and `backend-reference.npz` are local-only under
`artifacts/multisubject-v4/fvp-validation/`. Preserve them with the local bundle.
