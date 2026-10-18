# Local deployment assets beyond Git

Git preserves the multi-subject implementation, generators, tests, numerical
summaries, source/image hashes and reviewed deployment procedure. It excludes
EEG datasets, checkpoints/PTE binaries, HEX/AXF, environments, machine settings,
local firmware snapshots and build caches. A fresh clone alone is not a binary
backup or a self-contained installed Arm toolchain.

## Verified firmware snapshot

`artifacts/deployment-preservation/multisubject-v4-37262cc19c74-20261018/`
contains `firmware-and-rollback.tar.gz` and adjacent `manifest.json`.
Archive SHA-256:
`faa349bb9e18735514a7fee4565f8207a014ec0dc95d603ce754691829af92c2`.
315 preserved member files were checked by reading the completed archive.
`validation/local-preservation.json` records its size and checksum.

The archive contains:

- Exact compiled multi-subject E8 HEX/AXF, project sources, board/AI layers,
  CMSIS deployment recipe and local debugger/tool settings.
- Unchanged unfused v4 release bundle and FVP logs/reference outputs.
- Proven Model 2 streaming HEX/AXF, project source/configuration and deployment files.
- Original four-trial rollback archive, its manifest and supplemental CMSIS recipe.
- The five installed Ensemble 2.2.1 UART/startup driver dependencies, with their
  previously verified fingerprints. Preserve the installed pack and tool versions too.

Copy this directory to independent backup storage. A second directory on the same
laptop does not protect against laptop/storage loss. Do not commit the archive.
Do not run an exporter or rebuild over these preserved images to test a restore.
Verify the adjacent manifest/archive checksum before extracting into a NEW folder.
The embedded FILE_MANIFEST.json records each member's SHA-256 and size. Restoring
on a different machine requires CMSIS debugger/tool-path configuration and licensed
Arm tools; never import machine settings or license credentials into Git.

## Additional assets to back up separately

- `artifacts/multisubject-v4/data/` and `frontend/multisubject/data/`: genuine
  prepared model inputs, display EEG, trial metadata and provenance. They are
  intentionally excluded from the firmware snapshot and Git.
- Original A01E.mat–A09E.mat recordings in the existing ignored dataset cache.
- `artifacts/multisubject-v4/upstream/`: pinned original source snapshot. The
  `prepare_bundle.py` utility reconstructs it from the documented Git revision.
- `artifacts/multisubject-v4/numerical/`: full diagnostic Q/DQ/TOSA artifacts.
- Existing Model 2 datasets/captures and original repository dependencies listed
  in `tools/model2_random_trials/LOCAL_BACKUP_INVENTORY.md`.

The authoritative current candidate hashes are in `validation/streaming-image.json`.
The rollback archive and original Model 2 worktree remain in their existing locations.
No physical board, UART or debugger action was performed to make this snapshot.
