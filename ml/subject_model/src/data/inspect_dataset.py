"""Download and inspect subject 1 without filtering, epoching, or normalization.

Run from the repository root:
    .venv/bin/python -m src.data.inspect_dataset

Source timing protocol: https://www.bbci.de/competition/iv/desc_2a.pdf
Raw samples remain unchanged; MOABB converts source microvolts to MNE volts
and creates cue-aligned class annotations as part of loading.
"""

import os
from collections import Counter
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "mne_data"
    data_dir.mkdir(exist_ok=True)
    # Process-local settings avoid changing the user's global MNE configuration.
    os.environ["MNE_DATA"] = str(data_dir)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(data_dir)
    os.environ["MOABB_DOWNLOAD_PROVIDER"] = "upstream"
    os.environ.setdefault("MPLCONFIGDIR", str(data_dir / ".matplotlib"))

    import mne
    import moabb
    import numpy as np
    from moabb.datasets import BNCI2014_001

    dataset = BNCI2014_001(return_all_modalities=True, artifact_handling="annotate")
    print(f"MOABB {moabb.__version__}; MNE {mne.__version__}", flush=True)
    print(f"Local cache: {data_dir}", flush=True)
    # Explicitly limit both downloading and loading to subject 1.
    files = dataset.data_path(subject=1, path=str(data_dir), update_path=False)
    for filename in files:
        path = Path(filename).resolve()
        if not path.is_relative_to(data_dir):
            raise RuntimeError(f"Unexpected cache location: {path}")
        print(f"Source: {path.name} ({path.stat().st_size:,} bytes)")
    data = dataset.get_data(subjects=[1], cache_config={"use": False})
    print(f"Data hierarchy: subject -> session -> run -> MNE Raw; subjects={list(data)}")
    print(f"MOABB class codes: {dataset.event_id}")
    print(f"Source-marker imagery interval: {dataset.interval} seconds")

    totals = Counter()
    artifact_total = 0
    full_windows = 0
    for session, runs in data[1].items():
        counts = Counter()
        artifacts = 0
        print(f"\nSession {session}: {len(runs)} motor-imagery runs")
        for run, raw in runs.items():
            types = Counter(raw.get_channel_types())
            eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
            if len(eeg) != 22 or raw.info["sfreq"] != 250:
                raise RuntimeError("Recording does not match the 22-channel/250 Hz contract")
            events, _ = mne.events_from_annotations(
                raw, event_id=dataset.event_id, verbose=False
            )
            run_counts = Counter(
                {label: int(np.count_nonzero(events[:, 2] == code))
                 for label, code in dataset.event_id.items()}
            )
            counts.update(run_counts)
            flagged = sum(raw.annotations.description == "bnci_artifact")
            artifacts += int(flagged)
            # Inspect bounds only: no epochs or sample arrays are created.
            hand_events = events[np.isin(events[:, 2], [dataset.event_id["left_hand"],
                                                       dataset.event_id["right_hand"]])]
            full_windows += sum(
                0 <= sample - raw.first_samp
                and sample - raw.first_samp + 1000 <= raw.n_times
                for sample in hand_events[:, 0]
            )
            print(f"  Run {run}: raw shape=({len(raw.ch_names)}, {raw.n_times}), "
                  f"sfreq={raw.info['sfreq']:g} Hz, types={dict(types)}, "
                  f"trials={dict(run_counts)}, artifact flags={flagged}")
        totals.update(counts)
        artifact_total += artifacts
        print(f"  Session totals: {sum(counts.values())} trials; {dict(counts)}")
        print(f"  Left/right trials: {counts['left_hand'] + counts['right_hand']}")

    first = next(iter(next(iter(data[1].values())).values()))
    print("\nEEG channel order:", [name for name, kind in
          zip(first.ch_names, first.get_channel_types()) if kind == "eeg"])
    print(f"Total trials: {sum(totals.values())}; by class: {dict(totals)}")
    print(f"Source artifact-flagged trials: {artifact_total}; none rejected")
    hands = totals["left_hand"] + totals["right_hand"]
    print(f"Hand trials with a complete four-second imagery interval: {full_windows}/{hands}")
    if full_windows != hands:
        raise RuntimeError("Some hand trials have incomplete imagery intervals")
    print("\nTiming: fixation/beep at trial t=0; arrow cue at t=2 s for 1.25 s;")
    print("imagery is requested from t=2 to t=6 s (cue-relative 0 to 4 s).")
    print("MOABB get_data() class annotations are already shifted to cue onset.")
    print("Candidate later windows: cue-relative [0, 2) or [1, 3) seconds.")
    print("Use exactly 500 samples: inclusive epoch endpoints can yield 501.")
    print(f"One future two-second window per hand trial: ({hands}, 1, 22, 500).")
    print("Keep 0train for train/validation and 1test for held-out testing.")
    print("Remap class IDs 1/2 to model targets 0/1 later; no remapping applied.")
    print("No filtering, epoching, normalization, artifact rejection, or training performed.")


if __name__ == "__main__":
    main()
