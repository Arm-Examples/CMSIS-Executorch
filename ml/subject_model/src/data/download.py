"""Download selected BNCI2014_001 subjects into the ignored repository cache."""
import argparse
import os
from pathlib import Path


def configure_cache():
    path = Path(__file__).resolve().parents[2] / 'mne_data'
    path.mkdir(exist_ok=True)
    os.environ['MNE_DATA'] = str(path)
    os.environ['MNE_DATASETS_BNCI_PATH'] = str(path)
    os.environ['MOABB_DOWNLOAD_PROVIDER'] = 'upstream'
    os.environ.setdefault('MPLCONFIGDIR', str(path / '.matplotlib'))
    return path


def recording_provenance():
    """Fingerprint cached source files without changing them."""
    import hashlib
    cache = configure_cache()
    records = []
    for subject in range(1, 10):
        for suffix in ('T', 'E'):
            paths = list(cache.rglob(f'A{subject:02d}{suffix}.mat'))
            if len(paths) != 1:
                raise ValueError(f'Expected one source recording for {subject}/{suffix}')
            path = paths[0]
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            records.append({'subject': subject, 'session_file': suffix,
                            'path': str(path.relative_to(cache)),
                            'bytes': path.stat().st_size, 'sha256': digest})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', nargs='+', default=['all'])
    args = parser.parse_args()
    subjects = list(range(1, 10)) if args.subjects == ['all'] else list(map(int, args.subjects))
    if not subjects or any(s not in range(1, 10) for s in subjects):
        parser.error('subjects must be 1–9 or all')
    path = configure_cache()
    from moabb.datasets import BNCI2014_001
    dataset = BNCI2014_001()
    for subject in subjects:
        files = dataset.data_path(subject, path=str(path), update_path=False)
        print(f'Subject {subject}: {[Path(f).name for f in files]}', flush=True)


if __name__ == '__main__':
    main()
