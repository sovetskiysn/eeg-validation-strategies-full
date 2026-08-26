"""Dataset download into ``python_project/datasets/archive``."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kagglehub
import libarchive
import openneuro
import requests
from utils import ARCHIVE_DIR


def download_sam40() -> None:
    """Download and unpack the SAM 40 Figshare archive."""
    destination = ARCHIVE_DIR / "SAM 40 dataset"
    if destination.exists():
        return

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)
        archive_path = temporary_directory / "Data.rar"
        with requests.get("https://ndownloader.figshare.com/files/27956376", stream=True) as response:
            response.raise_for_status()
            with open(archive_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    file.write(chunk)

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


def download_distinguishing() -> None:
    """Download the canonical Distinguishing EEG files.

    The Kaggle archive also includes a byte-identical duplicate in
    ``eeg data/EEG Data``; it is intentionally not copied.  The retained
    files are placed directly in the local ``Distinguishing`` directory.
    """
    data_subdir = "EEG Data"
    destination = ARCHIVE_DIR / "Distinguishing"
    if not destination.exists():
        source = kagglehub.dataset_download(
            "inancigdem/eeg-data-for-mental-attention-state-detection"
        )
        source_data = Path(source) / data_subdir
        if not source_data.is_dir():
            raise FileNotFoundError(
                f"Expected {data_subdir!r} directory in Kaggle download: {source}"
            )
        shutil.copytree(source_data, destination)


def download_gradcpt() -> None:
    """Download the gradCPT EEG half of OpenNeuro ``ds006040`` v1.0.0.

    The release is a 185 GB simultaneous EEG-fMRI-DWI study of 16 tasks; only
    the sustained-attention task (gradCPT) is relevant here, so ``anat``,
    ``func``, ``dwi`` and ``derivatives`` are left on the server.  Note that
    ``*task-GRAD*`` deliberately misses ``task-IMGRADON`` -- the imagery
    variant is a different condition, not another gradCPT run.

    The root-level metadata patterns are spelled out because openneuro-py only
    force-includes essential BIDS files when they are filtered by ``exclude``.
    """
    openneuro.download(
        dataset="ds006040",
        tag="1.0.0",
        target_dir=ARCHIVE_DIR / "gradcpt",
        include=[
            "sub-*/eeg/*task-GRAD*",
            "sub-*/beh/*task-GRAD*",
            "/*.tsv",
            "/*.json",
            "/*.txt",
            "/README",
            "/CHANGES",
        ],
    )


def download_selective_attention() -> None:
    """Download OpenNeuro ``ds005089`` v1.0.1 in full.

    No ``include`` filter: the release is EEG only -- one
    ``task-competition`` recording per subject for 36 subjects.
    """
    openneuro.download(
        dataset="ds005089",
        tag="1.0.1",
        target_dir=ARCHIVE_DIR / "selective_attention",
    )


def download_cognitive_tasks_eeg() -> None:
    """Download the BrainVision EEG half of the TU Berlin EEG-NIRS dataset.

    The vendor-specific export is taken over the MATLAB bundle because MNE
    reads BrainVision directly, while the bundle would need the BBCI ``.mat``
    structures unpacked by hand.  The NIRS half is skipped: this project is
    EEG only.

    Each subject archive is flat (``nback1.vhdr``, ``gonogo1.eeg``, ...) with
    no directory of its own, so every one of them is extracted into its own
    ``VP0NN`` directory -- otherwise the 26 archives would overwrite each
    other.  For the same reason the skip check is per archive: an interrupted
    run must not re-download the subjects that already arrived.
    """
    base_url = "https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS"
    destination = ARCHIVE_DIR / "cognitive_tasks_eeg"
    destination.mkdir(parents=True, exist_ok=True)

    description = destination / "Dataset description_BrainVision and NIRx.pdf"
    if not description.exists():
        with requests.get(f"{base_url}/{description.name}", stream=True) as response:
            response.raise_for_status()
            with open(description, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    file.write(chunk)

    archives = [(f"EEG/VP{subject:03d}.zip", f"VP{subject:03d}") for subject in range(1, 27)]
    archives.append(("behavior.zip", "behavior"))
    for archive_name, directory_name in archives:
        target = destination / directory_name
        if target.exists():
            continue

        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "archive.zip"
            with requests.get(f"{base_url}/{archive_name}", stream=True) as response:
                response.raise_for_status()
                with open(archive_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        file.write(chunk)

            # Written into a staging directory and renamed, so an interrupted
            # extraction cannot leave a half-filled VP0NN that the skip check
            # above would then accept as complete.
            staging = destination / f".{directory_name}"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir()
            with libarchive.file_reader(str(archive_path)) as archive:
                for entry in archive:
                    entry_path = staging / entry.pathname
                    if entry.isdir:
                        entry_path.mkdir(parents=True, exist_ok=True)
                        continue
                    entry_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(entry_path, "wb") as file:
                        for block in entry.get_blocks():
                            file.write(block)
            staging.rename(target)
