"""Shared fixtures and collection policy for the acceptance suite."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


# MNE otherwise opens ~/.mne/mne-python.json in read/write mode at import time,
# which breaks collection in read-only-home CI and is unnecessary for these tests.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")


REPO_ROOT = Path(__file__).resolve().parents[1]
MNE_DATA_ROOT = Path.home() / "mne_data"


def _bnci_2a_is_cached() -> bool:
    """Return true only when the local MNE cache looks like BNCI 2014-001."""
    if not MNE_DATA_ROOT.is_dir():
        return False
    return all(next(MNE_DATA_ROOT.rglob(name), None) is not None for name in ("A01T.mat", "A01E.mat"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "data: requires cached BNCI2014-001 data under ~/mne_data")
    config.addinivalue_line("markers", "slow: long-running reproducibility acceptance test")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _bnci_2a_is_cached():
        return
    reason = "BNCI2014-001 is not cached under ~/mne_data"
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker("data") is not None:
            item.add_marker(skip)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261016)

