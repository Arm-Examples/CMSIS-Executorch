"""Load the v2 training datasets (Cho2017, Lee2019_MI) as continuous MNE Raw objects.

``load_external(dataset, subject)`` returns ``{moabb_session_key: raw}`` with only the
EEG channels kept (EMG and stim dropped from the data), in volts as MNE stores them,
and the left/right cue events in ``raw.info["temp"]["events"]`` as an MNE-style
``(n, 3)`` int array ``[cue_sample, 0, code]`` with code 769 = left hand, 770 = right
hand (the BCI IV 2a GDF codes, so ``preprocess.epoch`` applies unchanged). Event codes
are mapped through the dataset's ``event_id`` *names*: Cho2017 has
``{left_hand: 1, right_hand: 2}``, Lee2019_MI the reverse ``{left_hand: 2, right_hand: 1}``.
In both datasets the marker sits at the cue (no offset, unlike 2a).

``standardize_channels(raw)`` turns any of these Raws (and the 2a Raws from
``src.data.download.load_raw``) into the 22 channels of ``src.train.CHANNELS`` in that
order: montage ``standard_1005`` by name, spherical-spline interpolation of missing
names (Lee2019: FCz), then pick + reorder.

Downloads are never started here: ``is_downloaded`` checks the local MOABB layout under
``~/mne_data`` and ``load_external`` raises :class:`DataNotDownloaded` for a missing file.
MOABB 1.7 prefers NEMAR mirrors; the provider is pinned to ``"upstream"`` for this
process (``MOABB_DOWNLOAD_PROVIDER`` env var, which ``moabb.set_download_provider``
would otherwise persist in the user's MNE config) so the GigaDB files are read.

Units: MOABB loads Cho2017 as if the ``.mat`` values were microvolts, but they behave
like BioSemi ActiveTwo ADC counts (31.25 nV per count): read as µV, the 7-30 Hz CAR
signal is ~40-90 µV rms (2a: 2.7 µV) and every trial of s01 would exceed the paper's
±100 µV 8-30 Hz bad-trial threshold, yet ``bad_trial_indices`` lists none; divided by
32 the rms is ~1.5-3 µV and the threshold reproduces the listed bad trial of s07.
``UNIT_SCALE`` applies that factor. EA (per subject-session) makes the model
insensitive to it; it only matters for amplitude sanity checks.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import mne
import numpy as np

from src.train import CHANNELS

DATASETS = ("Cho2017", "Lee2019_MI")
LEFT_CODE, RIGHT_CODE = 769, 770
NAME_TO_CODE = {"left_hand": LEFT_CODE, "right_hand": RIGHT_CODE}
# Multiplier that brings MOABB's EEG data to volts (see module docstring).
UNIT_SCALE = {"Cho2017": 1.0 / 32.0, "Lee2019_MI": 1.0}


class DataNotDownloaded(FileNotFoundError):
    """The subject's raw files are not (yet) on disk; nothing was downloaded."""


def use_upstream_provider() -> None:
    """Pin MOABB to the original (GigaDB / BNCI) files for this process only."""
    os.environ["MOABB_DOWNLOAD_PROVIDER"] = "upstream"


def _dataset(dataset: str):
    use_upstream_provider()
    if dataset == "Cho2017":
        from moabb.datasets import Cho2017

        return Cho2017()
    if dataset == "Lee2019_MI":
        from moabb.datasets import Lee2019_MI

        return Lee2019_MI()
    raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")


def _subject_urls(dataset: str, subject: int) -> tuple[str, list[str]]:
    """``(moabb_sign, [url, ...])`` of the raw files MOABB reads for one subject."""
    if dataset == "Cho2017":
        from moabb.datasets.gigadb import GIGA_URL

        return "GIGADB", [f"{GIGA_URL}s{subject:02d}.mat"]
    if dataset == "Lee2019_MI":
        from moabb.datasets.Lee2019 import Lee2019_URL

        return "Lee2019-MI", [
            "{0}session{1}/s{2}/sess{1:02d}_subj{2:02d}_EEG_MI.mat".format(Lee2019_URL, sess, subject)
            for sess in (1, 2)
        ]
    raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")


def local_paths(dataset: str, subject: int) -> list[Path]:
    """Where MOABB stores (or would store) the subject's raw files; no network access."""
    from moabb.datasets.download import _normalize_destination, get_dataset_path

    sign, urls = _subject_urls(dataset, subject)
    root = Path(get_dataset_path(sign, None)) / f"MNE-{sign.lower()}-data"
    return [Path(_normalize_destination(u, root)) for u in urls]


def is_downloaded(dataset: str, subject: int) -> bool:
    """True when every raw file of the subject is complete on disk (pooch writes to a
    temporary name and renames on success, so an existing file is a finished one)."""
    return all(p.is_file() for p in local_paths(dataset, subject))


def map_events_by_name(events: np.ndarray, event_id: dict[str, int]) -> np.ndarray:
    """Keep left/right-hand events and recode them to 769/770 through ``event_id`` names.

    ``events`` is ``(n, 3)`` ``[sample, _, dataset_code]``; returns ``(m, 3)`` int64
    ``[sample, 0, 769|770]`` in recording order. Codes of other classes are dropped.
    """
    events = np.asarray(events, dtype=np.int64).reshape(-1, 3)
    code_map = {int(event_id[name]): code for name, code in NAME_TO_CODE.items() if name in event_id}
    if len(code_map) != 2:
        raise ValueError(f"event_id {event_id} lacks left_hand/right_hand")
    keep = np.isin(events[:, 2], list(code_map))
    ev = events[keep]
    codes = np.array([code_map[int(c)] for c in ev[:, 2]], dtype=np.int64)
    return np.stack([ev[:, 0], np.zeros_like(ev[:, 0]), codes], axis=1).astype(np.int64)


def _session_index(key: str) -> int:
    m = re.match(r"\d+", str(key))
    return int(m.group()) if m else 0


def _concat_runs(runs: dict[str, mne.io.BaseRaw]) -> mne.io.BaseRaw:
    ordered = [runs[k] for k in sorted(runs, key=_session_index)]
    if len(ordered) == 1:
        return ordered[0].copy().load_data()
    return mne.concatenate_raws([r.copy() for r in ordered], preload=True, verbose=False)


def load_external(dataset: str, subject: int) -> dict[str, mne.io.BaseRaw]:
    """``{moabb_session_key: raw}`` for one Cho2017 or Lee2019_MI subject (EEG only, volts).

    Sessions are ordered by their numeric MOABB key (Cho2017: ``"0"``; Lee2019_MI: two
    sessions). Runs of a session are concatenated in order; Lee2019_MI uses MOABB's
    default runs (the offline ``train`` run of each session, 100 trials).
    """
    if not is_downloaded(dataset, subject):
        raise DataNotDownloaded(f"{dataset} subject {subject}: {', '.join(map(str, local_paths(dataset, subject)))}")
    ds = _dataset(dataset)
    data = ds.get_data(subjects=[subject])[subject]
    out: dict[str, mne.io.BaseRaw] = {}
    for key in sorted(data, key=_session_index):
        raw = _concat_runs(data[key])
        stim = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "stim"]
        if len(stim) != 1:
            raise ValueError(f"{dataset} s{subject} session {key}: expected one stim channel, got {stim}")
        found = mne.find_events(raw, stim_channel=stim[0], shortest_event=0, verbose=False)
        found[:, 0] -= raw.first_samp
        events = map_events_by_name(found, ds.event_id)
        eeg = raw.pick(picks="eeg", exclude=[])
        if UNIT_SCALE[dataset] != 1.0:
            scale = UNIT_SCALE[dataset]
            eeg.apply_function(lambda d: d * scale, picks="all", channel_wise=False, verbose=False)
        eeg.info["temp"] = {"events": events}
        out[str(key)] = eeg
    return out


def _montage() -> mne.channels.DigMontage:
    """``standard_1005`` (MNE 1.13 warns that it will be renamed ``colin27_1005``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return mne.channels.make_standard_montage("standard_1005")


def _canonical_names(raw: mne.io.BaseRaw, channels: list[str]) -> None:
    """Rename channels that match one of ``channels`` up to case (e.g. ``FCZ`` -> ``FCz``)."""
    lower = {c.lower(): c for c in channels}
    rename = {ch: lower[ch.lower()] for ch in raw.ch_names if ch.lower() in lower and ch != lower[ch.lower()]}
    if rename:
        raw.rename_channels(rename)


def standardize_channels(raw: mne.io.BaseRaw, channels: list[str] = CHANNELS) -> mne.io.BaseRaw:
    """The 22 channels of ``channels`` in that order, missing ones interpolated.

    Sets the ``standard_1005`` montage by name (channels without a position are dropped
    unless they are wanted), adds each missing wanted channel as a zero channel marked
    bad, interpolates it with spherical splines from all remaining EEG channels, then
    picks and reorders. ``raw.info["temp"]`` (the events) is carried over. Operates on a
    copy; ``raw`` is untouched.
    """
    temp = raw.info.get("temp")
    raw = raw.copy().load_data()
    raw.pick(picks="eeg", exclude=[])
    _canonical_names(raw, channels)
    montage = _montage()
    missing = [c for c in channels if c not in raw.ch_names]
    unknown = [c for c in missing if c not in montage.ch_names]
    if unknown:
        raise ValueError(f"no standard_1005 position for {unknown}")
    if missing:
        info = mne.create_info(missing, raw.info["sfreq"], ch_types="eeg")
        zeros = mne.io.RawArray(np.zeros((len(missing), raw.n_times)), info, first_samp=raw.first_samp,
                                verbose=False)
        raw.add_channels([zeros], force_update_info=True)
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    if missing:
        pos = {ch["ch_name"]: ch["loc"][:3] for ch in raw.info["chs"]}
        no_pos = [c for c in raw.ch_names
                  if c not in channels and (not np.all(np.isfinite(pos[c])) or np.allclose(pos[c], 0))]
        if no_pos:
            raw.drop_channels(no_pos)
        raw.info["bads"] = list(missing)
        raw.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)
    raw.pick(list(channels))
    raw.reorder_channels(list(channels))
    if temp is not None:
        raw.info["temp"] = temp
    return raw


__all__ = ["DATASETS", "DataNotDownloaded", "is_downloaded", "local_paths", "load_external",
           "map_events_by_name", "standardize_channels", "use_upstream_provider", "UNIT_SCALE"]
