"""Dataset download into ``python_project/datasets/archive``."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kagglehub
import libarchive
import requests
from utils import ARCHIVE_DIR


def download_sam40() -> None:
    """Download and unpack the SAM 40 Figshare archive."""
    destination = ARCHIVE_DIR / "SAM 40 dataset"
    if destination.exists():
        print(f"SAM 40: archive already present at {destination}, skipping download")
        return

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)
        archive_path = temporary_directory / "Data.rar"
        print("SAM 40: downloading archive from Figshare")
        with requests.get("https://ndownloader.figshare.com/files/27956376", stream=True) as response:
            response.raise_for_status()
            with open(archive_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    file.write(chunk)

        print("SAM 40: extracting archive")
        extracted = temporary_directory / "extracted"
        extracted.mkdir()
        with libarchive.file_reader(str(archive_path)) as archive:
            for entry in archive:
                target = extracted / entry.pathname
                if entry.isdir:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "wb") as file:
                    for block in entry.get_blocks():
                        file.write(block)

        destination.mkdir(parents=True)
        for name in ["raw_data", "filtered_data", "artifact_removal", "Coordinates.locs", "scales.xls"]:
            matches = list(extracted.rglob(name))
            if matches:
                shutil.move(str(matches[0]), str(destination / name))
        print(f"SAM 40: archive ready at {destination}")


def download_distinguishing() -> None:
    """Download the canonical Distinguishing EEG files.

    The Kaggle archive also includes a byte-identical duplicate in
    ``eeg data/EEG Data``; it is intentionally not copied.  The retained
    files are placed directly in the local ``Distinguishing`` directory.
    """
    data_subdir = "EEG Data"
    destination = ARCHIVE_DIR / "Distinguishing"
    if destination.exists():
        print(f"Distinguishing: archive already present at {destination}, skipping download")
        return

    print("Distinguishing: downloading archive from Kaggle")
    source = kagglehub.dataset_download(
        "inancigdem/eeg-data-for-mental-attention-state-detection"
    )
    source_data = Path(source) / data_subdir
    if not source_data.is_dir():
        raise FileNotFoundError(
            f"Expected {data_subdir!r} directory in Kaggle download: {source}"
        )
    shutil.copytree(source_data, destination)
    print(f"Distinguishing: archive ready at {destination}")
