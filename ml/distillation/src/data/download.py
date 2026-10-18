"""Load BCI Competition IV 2a (MOABB ``BNCI2014_001``) as continuous MNE Raw objects.

``load_raw(subject)`` returns ``{1: raw_session1, 2: raw_session2}`` where session 1 is
MOABB's ``"0train"`` and session 2 is ``"1test"``. The six runs of a session are
concatenated in order, only the 22 EEG channels are kept (EOG and stim dropped from the
data; the trial markers are kept as annotations), sampling rate 250 Hz.

CLI: ``python -m src.data.download --subjects all`` downloads and caches everything
under ``~/mne_data`` (MOABB default, no login).
"""

from __future__ import annotations

import argparse

import mne
import numpy as np

SESSION_KEYS = {1: "0train", 2: "1test"}
FS = 250.0
N_EEG = 22
# Marker to cue delay in the BCI IV 2a paradigm: the marker sits at trial start (fixation
# cross), the cue is shown 2 s later. MOABB's interval for this dataset is [2, 6] s.
CUE_OFFSET_S = 2.0
# MOABB "STI" channel codes (BNCI2014_001.event_id): left 1, right 2, feet 3, tongue 4.
EVENT_ID = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
GDF_CODES = {"left_hand": 769, "right_hand": 770, "feet": 771, "tongue": 772}
CLASS_MAP = {"left_hand": 0, "right_hand": 1}


def _dataset():
    from moabb.datasets import BNCI2014_001

    return BNCI2014_001()


def _concat_runs(runs: dict[str, mne.io.BaseRaw]) -> mne.io.BaseRaw:
    ordered = [runs[k] for k in sorted(runs, key=lambda k: int(k))]
    raw = mne.concatenate_raws([r.copy() for r in ordered], preload=True, verbose=False)
    return raw


def find_cue_events(raw: mne.io.BaseRaw, fs: float = FS) -> np.ndarray:
    """MNE-style ``(n_events, 3)`` int array ``[cue_sample, 0, gdf_code]`` for all four
    motor-imagery classes, in recording order.

    Uses the ``STI`` channel MOABB adds (codes 1..4 at the trial marker), maps the codes
    back to the GDF codes 769..772 and shifts the marker by ``CUE_OFFSET_S`` so that
    sample 0 of a trial is the cue. ``epoch`` keeps only 769 (left) and 770 (right).
    """
    events = mne.find_events(raw, stim_channel="STI", shortest_event=0, verbose=False)
    keep = np.isin(events[:, 2], list(EVENT_ID.values()))
    events = events[keep]
    cue = events[:, 0] - raw.first_samp + int(round(CUE_OFFSET_S * fs))
    moabb_to_gdf = {EVENT_ID[k]: GDF_CODES[k] for k in EVENT_ID}
    gdf = np.array([moabb_to_gdf[int(c)] for c in events[:, 2]], dtype=np.int64)
    return np.stack([cue, np.zeros_like(cue), gdf], axis=1).astype(np.int64)


def load_raw(subject: int) -> dict[int, mne.io.BaseRaw]:
    """Continuous raw data for one subject, ``{1: session 1, 2: session 2}``.

    Each Raw holds the 22 EEG channels only (EOG and stim dropped), in volts as MNE
    stores them, with the cue events of all four classes in ``raw.info["temp"]["events"]`` as an
    MNE-style ``(n, 3)`` int array ``[cue_sample, 0, gdf_code]`` (769 left, 770 right).
    """
    data = _dataset().get_data(subjects=[subject])[subject]
    out: dict[int, mne.io.BaseRaw] = {}
    for session, key in SESSION_KEYS.items():
        raw = _concat_runs(data[key])
        if abs(raw.info["sfreq"] - FS) > 1e-6:
            raise ValueError(f"expected {FS} Hz, got {raw.info['sfreq']}")
        events = find_cue_events(raw)
        eeg = raw.pick(picks="eeg", exclude=[])
        if len(eeg.ch_names) != N_EEG:
            raise ValueError(f"expected {N_EEG} EEG channels, got {eeg.ch_names}")
        eeg.info["temp"] = {"events": events}
        out[session] = eeg
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download BCI IV 2a via MOABB")
    p.add_argument("--subjects", default="all", help="'all' or comma list, e.g. 1,2")
    a = p.parse_args(argv)
    subjects = list(range(1, 10)) if a.subjects == "all" else [int(s) for s in a.subjects.split(",")]
    ds = _dataset()
    for s in subjects:
        ds.download(subject_list=[s], verbose=False)
        raws = load_raw(s)
        for sess, raw in raws.items():
            ev = raw.info["temp"]["events"]
            n_lr = int(np.isin(ev[:, 2], [769, 770]).sum())
            print(f"subject {s} session {sess}: {len(raw.ch_names)} ch, "
                  f"{raw.n_times} samples at {raw.info['sfreq']:.0f} Hz, {len(ev)} trials, {n_lr} L/R")


if __name__ == "__main__":
    main()
