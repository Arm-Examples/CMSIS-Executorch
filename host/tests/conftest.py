import os
import sys
from pathlib import Path

HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))
os.environ.setdefault("MOABB_DOWNLOAD_PROVIDER", "upstream")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
if (Path.home() / "mne_data").is_dir():
    os.environ.setdefault("MNE_DATA", str(Path.home() / "mne_data"))
