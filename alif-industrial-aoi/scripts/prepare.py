#!/usr/bin/env python3
"""Check or apply the AOI overlay to a fresh, pinned upstream checkout."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

SUBMISSION = Path(__file__).resolve().parents[1]


def git(root: Path, *args: str) -> str:
    """Run Git against the selected checkout."""
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> None:
    """Validate prerequisites before any overlay writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply after checking; default is read-only")
    args = parser.parse_args()
    root = args.checkout.resolve()
    lock = json.loads((SUBMISSION / "dependencies.lock.json").read_text())
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        parser.error("Select the root of a separate upstream checkout")
    if git(root, "rev-parse", "HEAD") != lock["revision"]:
        parser.error("Checkout must be at the revision in dependencies.lock.json")
    if git(root, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=all"):
        parser.error("Checkout must have no local changes or untracked files")
    for entry in lock["submodules"]:
        dep = root / entry["path"]
        if not (dep / ".git").exists() or git(dep, "rev-parse", "HEAD") != entry["revision"]:
            parser.error(f"Missing or mismatched dependency: {entry['path']}; see README")
        if git(dep, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=all"):
            parser.error(f"Dependency has local changes: {entry['path']}")
    patch = SUBMISSION / "patches/platform-ui.patch"
    git(root, "apply", "--check", str(patch))
    files = sorted(p for p in (SUBMISSION / "overlay").rglob("*") if p.is_file())
    for path in files:
        target = root / path.relative_to(SUBMISSION / "overlay")
        if not target.resolve().is_relative_to(root):
            parser.error(f"Overlay destination escapes checkout: {target}")
    print(f"Validated pinned checkout, {len(files)} overlay files and platform patch")
    if args.apply:
        git(root, "apply", str(patch))
        for path in files:
            target = root / path.relative_to(SUBMISSION / "overlay")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        print("Applied AOI overlay; no build or device operation performed")


if __name__ == "__main__":
    main()
