# Clean-build status — 2026-10-18

**A clean firmware build has not been demonstrated.** Dependency setup did not
complete, so this attempt produced no new `mram.bin`, link map, memory usage
measurement, or firmware SHA-256. This does not establish a source compilation
failure; compilation was never reached.

## What was checked

- A separate upstream checkout was cloned from the documented public repository
  and checked out at `d7c39929fe482c1ce17b7be03edfab1db03acf06`.
- The included PTE is 3,470,320 bytes. Its SHA-256 matches the manifest and the
  unchanged CMake hash check:
  `f468ea2feae405abd7cfb9c0e95a7e9a357fd42698646846dba5ec081411ec7a`.
- GCC 13.3.1, CMake 3.31.12 and Python 3.10.22 were located independently of the
  original development project. A fresh Python environment was created.
- Source inspection confirms camera capture, GT911 touch integration,
  SINGLE/LIVE/STOP gating, original top-three class membership, YES/NO display,
  and green/red LED control. Linked inclusion and runtime behavior were not
  tested by this attempt.

## Why the attempt stopped

The initial attempt had approximately 500 MiB free disk space. After more space
was made available, setup was retried with approximately 3.5 GiB free.
Network-enabled recursive submodule downloads then failed with HTTP/2 transfer
errors, early EOFs, and connection resets. Failed clones included Arm-2D, LVGL,
and ExecuTorch. The independent Python requirements download failed with a
`files.pythonhosted.org` read timeout. The dependencies remain incomplete and
were not replaced with files from the original development checkout.

The overlay helper previously rejected the incomplete dependency checkout.
No overlay was applied to the fresh checkout and no firmware CMake configuration
or compiler invocation was performed. No evidence of missing source code or an
incompatible compiler configuration was obtained; those checks remain pending.

## What is included and excluded

The submission includes application sources, a supporting patch, build settings,
pinned dependency revisions, the exact calibrated PTE and recorded numerical
model results. Temporary `work/` and build environments are ignored and excluded
from the submission archive. Large third-party source trees and generated
firmware are excluded.

The project author confirms deployment of the original working firmware using
this calibrated PTE. That is separate from a successful clean reconstruction of
the submission. No new hardware test or flashing was performed during packaging.

Follow workflow A in [BUILD.md](BUILD.md) on a machine with sufficient storage
and reliable dependency download access. Complete dependency setup, verify the
pins, apply the overlay, and build with the included model. Inspect the resulting
ELF/map for features and memory fit before performing hardware validation.

The original development project and Arm example files were not modified. No
commit, push, or remote submission was performed during this preparation.
