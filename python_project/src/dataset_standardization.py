"""Dataset standardization: external raw EEG datasets into BIDS files."""

from __future__ import annotations

import inspect
import json
import shutil
import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, make_dataset_description, update_sidecar_json, write_raw_bids
from scipy.io import loadmat, savemat, whosmat
from utils import ARCHIVE_DIR, BIDS_DIR, DISTINGUISHING_SOURCE_DIR, SAM40_SOURCE_DIR


def standardize_sam40(replace: bool = True) -> None:
    """Convert the full SAM 40 release into one BIDS dataset."""
    # =============================================================================
    # Step 0: source archive preflight
    # =============================================================================
    raw_data_dir = SAM40_SOURCE_DIR / "raw_data"
    scales_path = SAM40_SOURCE_DIR / "scales.xls"
    coordinates_path = SAM40_SOURCE_DIR / "Coordinates.locs"
    expected_paths = [scales_path, coordinates_path] + [
        raw_data_dir / f"{source_task}_sub_{subject}_trial{trial}.mat"
        for source_task in ("Relax", "Stroop", "Arithmetic", "Mirror_image")
        for subject in range(1, 41)
        for trial in range(1, 4)
    ]
    missing = [path for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"SAM 40 archive is incomplete under {SAM40_SOURCE_DIR}: {len(missing)} files "
            f"are absent, among them {', '.join(path.name for path in missing[:5])}"
        )

    # Every rating is read and validated here, before the first BIDS byte: a rating
    # that turns out to be unusable halfway through leaves a half-written dataset.
    # Three trials of three rated tasks lie side by side in one row per participant.
    table = pd.read_excel(scales_path, header=None)
    rating_rows = {int(row.iloc[0]): row for _, row in table.iloc[2:].iterrows()}
    if rating_rows.keys() != set(range(1, 41)):
        raise ValueError(
            f"SAM 40 ratings in {scales_path} must cover subjects 1--40, "
            f"found {sorted(rating_rows)}"
        )
    task_columns = {"arithmetic": 1, "mirror": 2, "stroop": 3}
    stress_ratings: dict[tuple[int, int, str], int] = {}
    for subject, row in rating_rows.items():
        for trial in range(1, 4):
            for task, offset in task_columns.items():
                rating = int(row.iloc[(trial - 1) * 3 + offset])
                if not 1 <= rating <= 10:
                    raise ValueError(
                        f"SAM 40 rating outside 1--10 in {scales_path}: "
                        f"sub-{subject} trial{trial} {task} = {rating}"
                    )
                stress_ratings[(subject, trial, task)] = rating

    # The output root goes last, once the archive has been read and found complete:
    # clearing it earlier would let an incomplete archive destroy a conversion it
    # cannot rebuild.
    bids_root = BIDS_DIR / "sam40"
    if bids_root.exists():
        if not replace:
            raise FileExistsError(f"SAM 40 BIDS output already exists: {bids_root}")
        shutil.rmtree(bids_root)

    # =============================================================================
    # Step 1: source metadata and texts
    # =============================================================================
    task_names = {
        "Relax": "relax",
        "Stroop": "stroop",
        "Arithmetic": "arithmetic",
        "Mirror_image": "mirror",
    }
    montage = mne.channels.read_custom_montage(coordinates_path)
    channel_names = [
        "CZ", "FZ", "Fp1", "F7", "F3", "FC1", "C3", "FC5",
        "FT9", "T7", "CP5", "CP1", "P3", "P7", "PO9", "O1",
        "PZ", "OZ", "O2", "PO10", "P8", "P4", "CP2", "CP6",
        "T8", "FT10", "FC6", "C4", "FC2", "F4", "F8", "Fp2",
    ]
    events_sidecar = {
        "trial_type": {
            "Description": "Experimental condition and target label for this time segment.",
            "Levels": {
                "relax": "Resting baseline condition.",
                "stroop": "Stroop colour-word task condition.",
                "arithmetic": "Mental arithmetic task condition.",
                "mirror": "Mirror-image recognition task condition.",
            },
        },
        # mne-bids writes the column and its Description from the annotation
        # extras; only LongName has no place in its API.
        "stress_rating": {
            "LongName": "Self-reported stress rating",
            "Description": (
                "Participant feedback after the trial; 1 is minimal and 10 is maximal "
                "perceived stress."
            ),
        },
    }
    # The BrainVision export cannot carry the actual recording device, so facts
    # established from the source publication are written back afterwards.
    eeg_sidecar = {
        "Manufacturer": "Emotiv",
        "ManufacturersModelName": "EPOC Flex gel kit",
        "EEGReference": "CMS electrode on the left mastoid",
        "EEGGround": "DRL electrode on the right mastoid",
    }
    montage_description = (
        "Static 32-channel montage converted by MNE-Python from the source "
        "Coordinates.locs file; not participant-specific digitization."
    )
    changes = (
        inspect.cleandoc("""
            2.0.0 2026-08-15
              - Stopped calling a source trial a session, reversing 1.1.0. The source
                publication (section 3.1) describes three trials but never states that
                electrodes were reapplied between them, so ses-* claimed independent
                visits the release does not confirm. A trial is written as run-01..03
                within each task again; a matching run number across the four tasks
                identifies one source trial and nothing more.
              - Dropped sub-*_sessions.tsv. Without source sessions there is no
                session_role or session_quality to state, and no confirmed ground to
                exclude any recording by quality.
              - Merged the sam40-full, sam40-stroop, sam40-arithmetic and sam40-mirror
                datasets into this single dataset. Those were analytical views of one
                source release, not independent datasets; the task selection now lives
                in the experiment configuration.

            1.3.0 2026-08-14
              - Replaced the source_trial column of sub-*_sessions.tsv with session_role
                and session_quality, the two columns shared with the distinguishing
                dataset. SAM40 has no habituation phase, so every trial is experimental.
              - Named the original .mat file of every recording in the source column of
                sub-*_ses-*_scans.tsv, which is where provenance of a recording belongs.
              - Declared PowerLineFrequency instead of leaving it n/a.

            1.2.0 2026-08-13
              - Wrote the full SAM40 dataset and three relax-plus-task BIDS subsets:
                sam40-stroop, sam40-arithmetic, and sam40-mirror.

            1.1.0 2026-08-11
              - Recorded each source trial as its own session: trial1/2/3 are now
                ses-01/02/03, each holding the four task acquisitions of that trial.
              - Dropped the run entity, which previously encoded the trial inside a
                single ses-01.
              - Added sub-*_sessions.tsv with the source_trial provenance column.

            1.0.0 2026-08-10
              - Initial conversion of the Figshare SAM 40 release to BIDS.
            """)
        + "\n"
    )
    readme_note = inspect.cleandoc("""
        SAM40 trials and the run entity
        -------------------------------
        The source release ships 12 files per participant: four 25-second task
        acquisitions (relax, stroop, mirror, arithmetic) recorded in each of three
        explicitly named trials. All 12 are kept here. A trial is written as
        run-01, run-02, and run-03 within each task, so the same run number across
        the four tasks identifies the four acquisitions of one source trial. The
        original file behind each recording is named in sub-*_scans.tsv as source.

        The run number is the repetition index of the protocol, and nothing more.
        It does not assert separate visits, and it does not assert that electrodes
        were reapplied between trials: section 3.1 of the source publication
        describes the sequence of trials but never states either. An earlier
        revision of this conversion wrote a trial as ses-01..03 and thereby did
        make that claim; the session entity is no longer used for SAM40.

        This dataset holds every task. Selecting a subset of them -- relax against
        one stressor, for instance -- is an analytical choice and belongs to the
        experiment configuration, not to a separate BIDS dataset.

        Consequence downstream: signal preparation estimates one ICA solution per
        source trial, from the recordings that experiment retained, rather than one
        per participant. That grouping is the finest one the source supports; it is
        not evidence of an independent physical session.
        """)

    # =============================================================================
    # Step 2: write one recording per source .mat
    # =============================================================================
    bids_root.mkdir(parents=True)
    make_dataset_description(
        path=bids_root,
        dataset_type="raw",
        overwrite=True,
        verbose=False,
        name="SAM 40 EEG stress-task dataset",
        keywords=["EEG", "stress", "Stroop", "arithmetic"],
        authors=["Rajdeep Ghosh", "Nabamita Deb", "Kaushik Sengupta"],
        references_and_links=[
            "https://figshare.com/articles/dataset/"
            "SAM_40_Dataset_of_40_Subject_EEG_Recordings_to_Monitor_the_Induced-"
            "Stress_while_performing_Stroop_Color-Word_Test_Arithmetic_Task_and_"
            "Mirror_Image_Recognition_Task/14562090"
        ],
    )
    # stress_rating reaches events.tsv as an annotation extra, and mne-bids wants
    # its description at write time. It is already declared in the events sidecar,
    # so it is taken from there rather than spelled a second time.
    extra_columns_descriptions = {
        column: spec["Description"]
        for column, spec in events_sidecar.items()
        if column != "trial_type"
    }
    written = 0
    # A subject directory owns one scans.tsv covering all 12 of its recordings, so
    # both it and coordsystem.json can only be finalized after the last
    # write_raw_bids call for that subject. No session entity is involved: SAM40
    # has none, and the directory below the root is sub-XX/eeg.
    subject_dirs: dict[Path, tuple[BIDSPath, dict[str, str]]] = {}
    for source_task, bids_task in task_names.items():
        for subject in range(1, 41):
            for trial in range(1, 4):
                mat_path = raw_data_dir / f"{source_task}_sub_{subject}_trial{trial}.mat"
                # 0.51 uV/bit is the sample scale documented by the SAM 40 paper.
                data = loadmat(mat_path)["Data"] * 0.51e-6
                info = mne.create_info(channel_names, sfreq=128, ch_types="eeg")
                info["line_freq"] = 50
                raw = mne.io.RawArray(data, info, verbose=False)
                raw.set_montage(montage, match_case=False)
                raw.set_annotations(
                    mne.Annotations(
                        onset=[0.0],
                        duration=[raw.n_times / raw.info["sfreq"]],
                        description=[bids_task],
                        # mne-bids turns annotation extras into events.tsv
                        # columns; relax has no rating and is written as n/a.
                        extras=[
                            {"stress_rating": stress_ratings.get((subject, trial, bids_task))}
                        ],
                    )
                )
                # run is the repetition index of the protocol within each task,
                # which is all the source states. It is deliberately not a session.
                written_path = write_raw_bids(
                    raw,
                    BIDSPath(
                        subject=f"{subject:02d}",
                        task=bids_task,
                        run=f"{trial:02d}",
                        datatype="eeg",
                        root=bids_root,
                    ),
                    format="BrainVision",
                    allow_preload=True,
                    extra_columns_descriptions=extra_columns_descriptions,
                    verbose=False,
                )
                # The label vocabulary and the device facts a BrainVision export
                # cannot carry.
                update_sidecar_json(
                    written_path.copy().update(suffix="events", extension=".json"),
                    events_sidecar,
                    verbose=False,
                )
                update_sidecar_json(
                    written_path.copy().update(suffix="eeg", extension=".json"),
                    eeg_sidecar,
                    verbose=False,
                )
                # write_raw_bids supplies the final ``*_eeg.vhdr`` path. The input
                # BIDSPath has no suffix or extension yet and cannot be matched to
                # the filename stored in scans.tsv.
                _, sources = subject_dirs.setdefault(written_path.directory, (written_path, {}))
                sources[f"{written_path.datatype}/{written_path.fpath.name}"] = mat_path.name
                written += 1

    # =============================================================================
    # Step 3: provenance sidecars
    # =============================================================================
    # Coordinate provenance and raw-file provenance live at the same level of the
    # tree, so both are finalized in the same pass.
    for bids_path, sources in subject_dirs.values():
        coordinate_json = bids_path.copy().update(
            task=None, run=None, space="CapTrak", suffix="coordsystem", extension=".json",
            check=False,
        ).fpath
        # Not update_sidecar_json: it can add a key but never remove one, and the
        # landmark placeholders mne-bids emits have to go.
        metadata = json.loads(coordinate_json.read_text(encoding="utf-8"))
        metadata["EEGCoordinateSystemDescription"] = (
            f"{montage_description} Coordinates use the CapTrak/RAS convention for BIDS "
            "and MNE interoperability; they were not measured with a CapTrak device."
        )
        for key in (
            "AnatomicalLandmarkCoordinates",
            "AnatomicalLandmarkCoordinateSystem",
            "AnatomicalLandmarkCoordinateUnits",
        ):
            metadata.pop(key, None)
        coordinate_json.write_text(
            json.dumps(metadata, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # The table can only be finalized now: write_raw_bids rewrites it for every
        # recording of the subject, so a column added earlier would not survive.
        scans_path = bids_path.copy().update(
            task=None, run=None, datatype=None, suffix="scans", extension=".tsv", check=False
        ).fpath
        scans = pd.read_csv(scans_path, sep="\t", keep_default_na=False)
        scans["source"] = scans["filename"].map(sources)
        if scans["source"].isna().any():
            missing_sources = scans.loc[scans["source"].isna(), "filename"].tolist()
            raise ValueError(f"No source file recorded for {missing_sources} in {scans_path}")
        scans.to_csv(scans_path, sep="\t", index=False)
        scans_path.with_suffix(".json").write_text(
            json.dumps({"source": {"Description": "Original source filename."}}, indent=4) + "\n",
            encoding="utf-8",
        )

    # SAM40 writes no sessions.tsv: the source has no sessions, so there is no
    # session_role or session_quality to state. Filling them with n/a would claim
    # the notion applies but is unknown, which is a different and false statement.

    # =============================================================================
    # Step 4: dataset-level texts
    # =============================================================================
    # CHANGES is the BIDS changelog; it is the only place the layout history of a
    # conversion is recorded, because the raw files themselves never moved.
    (bids_root / "CHANGES").write_text(changes, encoding="utf-8")
    readme = bids_root / "README"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else ""
    readme.write_text(existing.rstrip() + "\n\n" + readme_note + "\n", encoding="utf-8")
    print(f"{bids_root.name}: wrote {written} recordings")


# --- Distinguishing mental attention states ---


def standardize_distinguishing(replace: bool = True) -> None:
    """Convert every Distinguishing raw ``.mat`` file into BIDS."""
    # =============================================================================
    # Step 0: source archive preflight
    # =============================================================================
    if not DISTINGUISHING_SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Distinguishing source directory does not exist: {DISTINGUISHING_SOURCE_DIR}"
        )
    # ``download_distinguishing`` retains the canonical 34 files directly in this
    # directory.  The Kaggle archive also carries a nested byte-identical copy;
    # recursing here would turn it into 34 false duplicate recordings.
    recording_paths: dict[int, Path] = {}
    for path in DISTINGUISHING_SOURCE_DIR.glob("eeg_record*.mat"):
        index_text = path.stem.removeprefix("eeg_record")
        if not index_text.isdigit():
            raise ValueError(f"Unexpected Distinguishing recording filename: {path}")
        if int(index_text) in recording_paths:
            raise ValueError(f"Duplicate Distinguishing recording index {index_text}: {path}")
        recording_paths[int(index_text)] = path
    if recording_paths.keys() != set(range(1, 35)):
        raise ValueError(
            f"Distinguishing archive under {DISTINGUISHING_SOURCE_DIR} must contain "
            f"eeg_record1.mat through eeg_record34.mat, found {sorted(recording_paths)}"
        )
    # The output root goes last, once the archive has been read and found complete:
    # clearing it earlier would let an incomplete archive destroy a conversion it
    # cannot rebuild.
    bids_root = BIDS_DIR / "distinguishing"
    if bids_root.exists():
        if not replace:
            raise FileExistsError(f"Distinguishing BIDS output already exists: {bids_root}")
        shutil.rmtree(bids_root)

    # =============================================================================
    # Step 1: source metadata and texts
    # =============================================================================
    montage = mne.channels.make_standard_montage("standard_1020")
    channel_names = [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
    ]
    events_sidecar = {
        "trial_type": {
            "Description": "Experimental condition and target label for this time segment.",
            "Levels": {
                "focused": "Focused-attention segment of the recording.",
                "unfocused": "Unfocused-attention segment of the recording.",
                "drowsy": "Drowsiness segment; participants could relax, close their eyes, and doze.",
            },
        },
    }
    eeg_sidecar = {"Manufacturer": "Emotiv", "ManufacturersModelName": "EPOC"}
    sessions_sidecar = {
        "session_role": {
            "Description": "Role of this session in the original study protocol.",
            "Levels": {
                "habituation": "Habituation session; the source protocol does not intend it for analysis.",
                "experimental": "Session intended for study analysis.",
            },
        },
        "session_quality": {
            "Description": "Whether the recording covers the protocol it is labelled with.",
            "Levels": {
                "ok": "Recording covers the full labelled protocol.",
                "incomplete": "Recording is shorter than the labelled protocol; retain only with an explicit decision.",
            },
        },
    }
    montage_description = (
        "Static MNE-Python standard_1020 template montage assigned from the 14 "
        "channel names documented by the Kaggle release; not participant-specific digitization."
    )
    changes = (
        inspect.cleandoc("""
            1.1.0 2026-08-14
              - Split the session_status column of sub-*_sessions.tsv into session_role
                and session_quality. The two used to share one column, where a shortened
                experimental session was recorded as having no role at all. Both columns
                and their Levels are now identical to the SAM40 datasets.
              - Moved the original .mat filename out of sub-*_sessions.tsv (source_file)
                into sub-*_ses-*_scans.tsv (source): it describes a recording, not a
                session.
              - Dropped the run entity. A session holds exactly one recording, so the
                entity distinguished nothing.
              - Declared PowerLineFrequency instead of leaving it n/a.

            1.0.0 2026-08-10
              - Initial conversion of the Kaggle mental-attention release to BIDS.
            """)
        + "\n"
    )

    # =============================================================================
    # Step 2: write one recording per source .mat
    # =============================================================================
    bids_root.mkdir(parents=True)
    make_dataset_description(
        path=bids_root,
        dataset_type="raw",
        overwrite=True,
        verbose=False,
        name="EEG mental attention state detection dataset",
        keywords=["EEG", "mental attention", "focused", "unfocused", "drowsy"],
        authors=["Çiğdem İnan Acı", "Murat Kaya", "Yuriy Mishchenko"],
        references_and_links=[
            "https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection"
        ],
    )
    written = 0
    # Role and quality are established while each recording is built; collecting
    # rows by subject avoids reading the output again later.
    session_rows: dict[str, list[dict[str, str]]] = {}
    for index in sorted(recording_paths):
        mat_path = recording_paths[index]
        source_data = loadmat(mat_path)["o"][0][0]["data"]
        # Kaggle's data card uses one-based indices: EEG is columns 4--17.
        # Python's half-open slice is therefore 3:17, retaining all 14 EEG channels.
        # 0.51 uV/bit is the Emotiv EPOC sample scale, the same as in SAM 40.
        data = source_data[:, 3:17].T * 0.51e-6
        info = mne.create_info(channel_names, sfreq=128, ch_types="eeg")
        info["line_freq"] = 50
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_montage(montage)
        recording_duration = raw.n_times / raw.info["sfreq"]
        phase_events = [
            (onset, min(600.0, recording_duration - onset), label)
            for onset, label in ((0.0, "focused"), (600.0, "unfocused"), (1200.0, "drowsy"))
            if recording_duration > onset
        ]
        onsets, durations, labels = zip(*phase_events)
        raw.set_annotations(
            mne.Annotations(onset=onsets, duration=durations, description=labels)
        )
        # Kaggle's ordered records follow the documented 7+7+7+7+6 session layout.
        subject, session = min((index - 1) // 7 + 1, 5), (index - 1) % 7 + 1
        # No run entity: a session holds exactly one recording, so it would
        # distinguish nothing. The Kaggle index is not recoverable from BIDS
        # entities either -- it follows 7 * (subject - 1) + session, which no
        # name in the tree spells out -- so the .mat name goes to scans.tsv.
        written_path = write_raw_bids(
            raw,
            BIDSPath(
                subject=f"{subject:02d}",
                session=f"{session:02d}",
                task="attention",
                datatype="eeg",
                root=bids_root,
            ),
            format="BrainVision",
            allow_preload=True,
            extra_columns_descriptions={},
            verbose=False,
        )
        # The label vocabulary and the device facts a BrainVision export cannot carry.
        update_sidecar_json(
            written_path.copy().update(suffix="events", extension=".json"),
            events_sidecar,
            verbose=False,
        )
        update_sidecar_json(
            written_path.copy().update(suffix="eeg", extension=".json"),
            eeg_sidecar,
            verbose=False,
        )
        # This session owns exactly one recording, so its BIDS provenance can be
        # finalized now rather than collected for a second pass.
        coordinate_json = written_path.copy().update(
            task=None, space="CapTrak", suffix="coordsystem", extension=".json", check=False,
        ).fpath
        # update_sidecar_json cannot remove the placeholder landmarks emitted by
        # mne-bids, so this sidecar needs one direct read-modify-write.
        coordsystem_metadata = json.loads(coordinate_json.read_text(encoding="utf-8"))
        coordsystem_metadata["EEGCoordinateSystemDescription"] = (
            f"{montage_description} Coordinates use the CapTrak/RAS convention for BIDS "
            "and MNE interoperability; they were not measured with a CapTrak device."
        )
        for key in (
            "AnatomicalLandmarkCoordinates",
            "AnatomicalLandmarkCoordinateSystem",
            "AnatomicalLandmarkCoordinateUnits",
        ):
            coordsystem_metadata.pop(key, None)
        coordinate_json.write_text(
            json.dumps(coordsystem_metadata, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        scans_path = written_path.copy().update(
            task=None, datatype=None, suffix="scans", extension=".tsv", check=False
        ).fpath
        scans = pd.read_csv(scans_path, sep="\t", keep_default_na=False)
        if len(scans) != 1:
            raise ValueError(f"Expected one recording in {scans_path}, found {len(scans)}")
        scans["source"] = mat_path.name
        scans.to_csv(scans_path, sep="\t", index=False)
        scans_path.with_suffix(".json").write_text(
            json.dumps({"source": {"Description": "Original source filename."}}, indent=4) + "\n",
            encoding="utf-8",
        )

        session_rows.setdefault(f"{subject:02d}", []).append(
            {
                "session_id": f"ses-{session:02d}",
                "session_role": "habituation" if session <= 2 else "experimental",
                "session_quality": "ok" if recording_duration >= 1800 else "incomplete",
            }
        )
        written += 1

    # =============================================================================
    # Step 3: session facts
    # =============================================================================
    # Written because this source has sessions and states facts about them.
    for subject_id, rows in session_rows.items():
        subject_root = bids_root / f"sub-{subject_id}"
        pd.DataFrame(rows).to_csv(
            subject_root / f"sub-{subject_id}_sessions.tsv", sep="\t", index=False
        )
        (subject_root / f"sub-{subject_id}_sessions.json").write_text(
            json.dumps(
                sessions_sidecar,
                indent=4,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    # =============================================================================
    # Step 4: dataset-level texts
    # =============================================================================
    # CHANGES is the BIDS changelog; it is the only place the layout history of a
    # conversion is recorded, because the raw files themselves never moved.
    (bids_root / "CHANGES").write_text(changes, encoding="utf-8")
    # No README note is appended: one recording per session, and no entity carrying
    # a claim the release does not make, so mne-bids' own README says enough.
    print(f"{bids_root.name}: wrote {written} recordings")


# --- gradCPT sustained attention (OpenNeuro ds006040) ---


def standardize_gradcpt(replace: bool = True) -> None:
    """Convert the gradCPT EEG half of OpenNeuro ds006040 into BIDS."""
    # =============================================================================
    # Step 0: source archive preflight
    # =============================================================================
    source_dir = ARCHIVE_DIR / "gradcpt"
    demographics_path = source_dir / "Demographic_Information.tsv"
    # sub-003 withdrew during the EEG-fMRI session and is absent from the release,
    # so the subject labels are not a contiguous range and are read off the tree.
    subject_labels = sorted(path.name for path in source_dir.glob("sub-*") if path.is_dir())
    if not subject_labels:
        raise FileNotFoundError(f"No gradCPT subject directories under {source_dir}")
    # Every participant has one scanner-off run and the three scanner-on runs of the
    # task. The order of this list is the order the run entity is assigned in.
    source_runs = [("scanoff", "GRADOFF", 1), ("scanon", "GRADON", 1),
                   ("scanon", "GRADON", 2), ("scanon", "GRADON", 3)]
    expected_paths = [demographics_path]
    for subject_label in subject_labels:
        for _, source_task, source_run in source_runs:
            stem = f"{subject_label}_task-{source_task}_run-{source_run}"
            expected_paths.append(source_dir / subject_label / "eeg" / f"{stem}_eeg.set")
            expected_paths.append(source_dir / subject_label / "beh" / f"{stem}_beh.tsv")
    missing = [path for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"gradCPT archive is incomplete under {source_dir}: {len(missing)} files are "
            f"absent, among them {', '.join(path.name for path in missing[:5])}"
        )

    # Demographics are read and validated before the first BIDS byte, like the SAM 40
    # ratings: a table that turns out unusable halfway through leaves a half-written
    # dataset. The second row of the file repeats the header in short form.
    demographics = pd.read_csv(demographics_path, sep="\t", skiprows=[1])
    demographics = demographics.set_index(demographics.columns[0])
    if not set(subject_labels) <= set(demographics.index):
        raise ValueError(
            f"gradCPT demographics in {demographics_path} do not cover "
            f"{sorted(set(subject_labels) - set(demographics.index))}"
        )
    # The release's own read_me states 1 is female and 0 is male.
    participant_sex = {1: "F", 0: "M"}
    participants = {}
    for subject_label in subject_labels:
        row = demographics.loc[subject_label]
        if int(row["Gender"]) not in participant_sex:
            raise ValueError(f"Unexpected gradCPT gender code for {subject_label}: {row['Gender']}")
        participants[subject_label] = {
            "age": int(row["Age"]),
            "sex": participant_sex[int(row["Gender"])],
        }

    # The output root goes last, once the archive has been read and found complete:
    # clearing it earlier would let an incomplete archive destroy a conversion it
    # cannot rebuild.
    bids_root = BIDS_DIR / "gradcpt"
    if bids_root.exists():
        if not replace:
            raise FileExistsError(f"gradCPT BIDS output already exists: {bids_root}")
        shutil.rmtree(bids_root)

    # =============================================================================
    # Step 1: source metadata and texts
    # =============================================================================
    # The release's README_dataset.txt documents S245 as the start and S155 as the
    # stop of a gradCPT scene; the pair brackets each 800 ms transition. 'T  1' is the
    # fMRI volume trigger and 'Sync On' the EEG-MR clock pulse, both carried over
    # because gradient-artifact correction of the scanner-on runs needs them.
    trial_start_marker, trial_stop_marker = "S245", "S155"
    scanner_markers = {"T  1": "mr_volume", "Sync On": "sync_pulse"}
    # The behavioural table codes the scene category; the release's read_me states
    # 6 is city and 16 is mountain.
    scene_categories = {6: "city", 16: "mountain"}
    events_sidecar = {
        "trial_type": {
            "Description": "Scene category of the gradCPT trial, or a scanner timing pulse.",
            "Levels": {
                "city": "City scene; participants press the button (75% of trials).",
                "mountain": "Mountain scene; participants withhold the response (25% of trials).",
                "mr_volume": "Onset of an fMRI volume acquisition (TR = 2 s).",
                "sync_pulse": "Clock pulse synchronising the EEG amplifier with the MR scanner.",
            },
        },
    }
    eeg_sidecar = {
        "Manufacturer": "Brain Products",
        "ManufacturersModelName": "BrainCap MR with Multitrodes",
        "EEGReference": "FCz",
        "InstitutionName": "Institute for Basic Science (IBS), Sungkyunkwan University (SKKU)",
        "TaskDescription": (
            "Gradual-onset continuous performance task (gradCPT). Grayscale city and "
            "mountain scenes fade into one another every 800 ms; participants press a "
            "button for city scenes (75%) and withhold the response for mountain scenes "
            "(25%). Recorded both outside (acq-scanoff) and during (acq-scanon) "
            "simultaneous fMRI."
        ),
    }
    montage_description = (
        "Static MNE-Python standard_1020 template montage assigned from the 63 channel "
        "names of the source release; not participant-specific digitization."
    )
    changes = (
        inspect.cleandoc("""
            1.0.0 2026-08-19
              - Initial conversion of the gradCPT EEG half of OpenNeuro ds006040 v1.0.0
                to the BIDS layout of this project.
            """)
        + "\n"
    )
    readme_note = inspect.cleandoc("""
        gradCPT scanner state, runs and labels
        --------------------------------------
        The source release records the task four times per participant: once with the
        MR scanner off (task-GRADOFF, 225 trials over ~180 s) and three times during
        simultaneous fMRI (task-GRADON, 450 trials over 360 s). Both are kept. The
        scanner state is written as acq-scanoff and acq-scanon, because it is a
        property of the acquisition and not of the task the participant performed.

        The run entity is a unique recording index within the participant, assigned
        as run-01 for the scanner-off recording and run-02..04 for the three
        scanner-on recordings. It deliberately does not repeat the source numbering,
        under which GRADOFF run-1 and GRADON run-1 are different recordings sharing a
        number. The original file of every recording is named in sub-*_scans.tsv as
        source.

        Events. Each trial carries the scene category the source behavioural table
        states -- city or mountain -- for the 800 ms bracketed by the release's S245
        and S155 markers. The fMRI volume triggers and the EEG-MR clock pulses are
        kept as mr_volume and sync_pulse: without them the gradient artifact of the
        scanner-on runs cannot be corrected. Nothing here is an attention label. The
        sustained-attention state of a trial ("in the zone" against "out of the
        zone") is a derived quantity computed from the variance time course of the
        response times, not a fact the release states, and it is therefore not part
        of this conversion.

        Quality. The release's README_dataset.txt names sub-006, sub-007, sub-008,
        sub-024, sub-025, sub-026 and sub-027 as excluded from the validation of the
        source paper "due to quality of EEG and fMRI data", and reports that the
        first TR of sub-025_task-GRADON_run-1 was reconstructed by hand. Those
        recordings are converted here like every other one: the statement is about
        the source paper's analysis, and this project has no source-session table in
        which to state a per-recording quality verdict.
        """)

    # =============================================================================
    # Step 2: write one recording per source .set
    # =============================================================================
    bids_root.mkdir(parents=True)
    make_dataset_description(
        path=bids_root,
        dataset_type="raw",
        overwrite=True,
        verbose=False,
        name="Sustained attention task (gradCPT) EEG dataset",
        keywords=["EEG", "sustained attention", "gradCPT", "EEG-fMRI"],
        authors=[
            "Younghwa Cha", "Yeji Lee", "Eunhee Ji", "SoHyun Han", "Sunhyun Min",
            "Hyoungkyu Kim", "Minseo Cho", "Haesung Lee", "Youngjai Park", "Joon-Young Moon",
        ],
        references_and_links=[
            "https://doi.org/10.18112/openneuro.ds006040.v1.0.0",
            "https://doi.org/10.1038/s41597-026-06616-6",
        ],
    )
    montage = mne.channels.make_standard_montage("standard_1020")
    written = 0
    # A subject directory owns one scans.tsv covering all four of its recordings, so
    # both it and coordsystem.json can only be finalized after the last write_raw_bids
    # call for that subject. The source has no sessions, so the directory below the
    # root is sub-XXX/eeg.
    subject_dirs: dict[Path, tuple[BIDSPath, dict[str, str]]] = {}
    # mne-bids writes one coordsystem.json per acquisition, so the electrode
    # provenance of a subject lives in two files, not one.
    coordinate_paths: dict[tuple[Path, str], BIDSPath] = {}
    for subject_label in subject_labels:
        for run, (scanner_state, source_task, source_run) in enumerate(source_runs, start=1):
            stem = f"{subject_label}_task-{source_task}_run-{source_run}"
            set_path = source_dir / subject_label / "eeg" / f"{stem}_eeg.set"
            raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
            # Channel 32 of the cap is an ECG electrode on the participant's back.
            raw.set_channel_types({"ECG": "ecg"})
            raw.set_montage(montage)
            raw.info["line_freq"] = 60

            # The scene category lives in the behavioural table, one row per key
            # event, so trials are collapsed before they are paired with the markers.
            behaviour = pd.read_csv(
                source_dir / subject_label / "beh" / f"{stem}_beh.tsv", sep="\t"
            )
            trials = behaviour.groupby("Trial", sort=True).first()
            descriptions = raw.annotations.description
            onsets = raw.annotations.onset
            trial_onsets = onsets[descriptions == trial_start_marker]
            trial_offsets = onsets[descriptions == trial_stop_marker]
            if len(trial_onsets) != len(trial_offsets):
                raise ValueError(
                    f"gradCPT recording {stem} has {len(trial_onsets)} start markers "
                    f"and {len(trial_offsets)} stop markers; the two must agree"
                )
            unknown = set(trials["ImgType"]) - set(scene_categories)
            if unknown:
                raise ValueError(f"Unknown gradCPT scene codes in {stem}: {sorted(unknown)}")
            # Markers are paired with the behavioural table by ordinal position
            # whenever the two agree in number, which they do for all but one
            # recording of the release. That recording lost a trigger pair mid-run,
            # and pairing it by ordinal position would give every trial after the gap
            # the label of the one before it -- a plausible result and a false one.
            # There the gaps between neighbouring markers say how many trials each
            # one spans: a dropped trial leaves a gap of exactly two transitions.
            # The step is measured between neighbours rather than against a grid laid
            # from the first marker, because the presentation clock drifts against
            # the amplifier by a quarter of a trial over a full run. The checks are
            # what make the reconstruction safe: every gap must be a whole number of
            # trials, and the last marker must be the last behavioural trial, so
            # nothing can be shifted at either end.
            if len(trial_onsets) == len(trials):
                positions = np.arange(len(trials))
            else:
                spacing = float(np.median(np.diff(trial_onsets)))
                steps = np.diff(trial_onsets) / spacing
                positions = np.concatenate([[0], np.cumsum(np.round(steps))]).astype(int)
                if (np.abs(steps - np.round(steps)) > 0.25).any() or (np.round(steps) < 1).any():
                    raise ValueError(
                        f"gradCPT recording {stem} has {len(trial_onsets)} markers "
                        f"against {len(trials)} behavioural trials, and its gaps are "
                        f"not whole multiples of {spacing:.4f} s; the missing trials "
                        "cannot be located"
                    )
                if positions[-1] + 1 != len(trials):
                    raise ValueError(
                        f"gradCPT recording {stem} spans {positions[-1] + 1} trial "
                        f"positions against {len(trials)} behavioural trials; the two "
                        "do not describe the same run"
                    )
            scene_codes = trials["ImgType"].to_numpy()
            annotation_onsets = list(trial_onsets)
            annotation_durations = list(trial_offsets - trial_onsets)
            annotation_labels = [scene_categories[scene_codes[position]] for position in positions]
            # The scanner pulses are kept at their own onsets, with no duration: they
            # are instants, not intervals.
            for marker, label in scanner_markers.items():
                marker_onsets = onsets[descriptions == marker]
                annotation_onsets.extend(marker_onsets)
                annotation_durations.extend([0.0] * len(marker_onsets))
                annotation_labels.extend([label] * len(marker_onsets))
            # EEGLAB marks a data discontinuity as 'boundary'; the BAD_ prefix is what
            # makes MNE and this project's preparation stage skip the span.
            boundary_onsets = onsets[descriptions == "boundary"]
            annotation_onsets.extend(boundary_onsets)
            annotation_durations.extend([0.0] * len(boundary_onsets))
            annotation_labels.extend(["BAD_boundary"] * len(boundary_onsets))
            raw.set_annotations(
                mne.Annotations(
                    onset=annotation_onsets,
                    duration=annotation_durations,
                    description=annotation_labels,
                )
            )

            written_path = write_raw_bids(
                raw,
                BIDSPath(
                    subject=subject_label.removeprefix("sub-"),
                    task="gradcpt",
                    acquisition=scanner_state,
                    run=f"{run:02d}",
                    datatype="eeg",
                    root=bids_root,
                ),
                format="BrainVision",
                allow_preload=True,
                extra_columns_descriptions={},
                verbose=False,
            )
            # The label vocabulary and the device facts a BrainVision export cannot
            # carry.
            update_sidecar_json(
                written_path.copy().update(suffix="events", extension=".json"),
                events_sidecar,
                verbose=False,
            )
            update_sidecar_json(
                written_path.copy().update(suffix="eeg", extension=".json"),
                eeg_sidecar,
                verbose=False,
            )
            # write_raw_bids supplies the final ``*_eeg.vhdr`` path. The input BIDSPath
            # has no suffix or extension yet and cannot be matched to the filename
            # stored in scans.tsv.
            _, sources = subject_dirs.setdefault(written_path.directory, (written_path, {}))
            sources[f"{written_path.datatype}/{written_path.fpath.name}"] = set_path.name
            coordinate_paths.setdefault((written_path.directory, scanner_state), written_path)
            written += 1

    # =============================================================================
    # Step 3: provenance sidecars
    # =============================================================================
    for bids_path in coordinate_paths.values():
        coordinate_json = bids_path.copy().update(
            task=None, run=None, space="CapTrak", suffix="coordsystem",
            extension=".json", check=False,
        ).fpath
        # Not update_sidecar_json: it can add a key but never remove one, and the
        # landmark placeholders mne-bids emits have to go.
        metadata = json.loads(coordinate_json.read_text(encoding="utf-8"))
        metadata["EEGCoordinateSystemDescription"] = (
            f"{montage_description} Coordinates use the CapTrak/RAS convention for BIDS "
            "and MNE interoperability; they were not measured with a CapTrak device."
        )
        for key in (
            "AnatomicalLandmarkCoordinates",
            "AnatomicalLandmarkCoordinateSystem",
            "AnatomicalLandmarkCoordinateUnits",
        ):
            metadata.pop(key, None)
        coordinate_json.write_text(
            json.dumps(metadata, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # The scans table is one per subject and can only be finalized now: write_raw_bids
    # rewrites it for every recording of the subject, so a column added earlier would
    # not survive.
    for bids_path, sources in subject_dirs.values():
        scans_path = bids_path.copy().update(
            task=None, acquisition=None, run=None, datatype=None, suffix="scans",
            extension=".tsv", check=False,
        ).fpath
        scans = pd.read_csv(scans_path, sep="\t", keep_default_na=False)
        scans["source"] = scans["filename"].map(sources)
        if scans["source"].isna().any():
            missing_sources = scans.loc[scans["source"].isna(), "filename"].tolist()
            raise ValueError(f"No source file recorded for {missing_sources} in {scans_path}")
        scans.to_csv(scans_path, sep="\t", index=False)
        scans_path.with_suffix(".json").write_text(
            json.dumps({"source": {"Description": "Original source filename."}}, indent=4) + "\n",
            encoding="utf-8",
        )

    # gradCPT writes no sessions.tsv: every participant was recorded in a single
    # visit, so there is no session of which to state a role or a quality.

    # =============================================================================
    # Step 4: dataset-level texts
    # =============================================================================
    # Individual age and sex are published by the release, so they are stated rather
    # than left n/a. write_raw_bids has already created the table with n/a columns.
    participants_path = bids_root / "participants.tsv"
    table = pd.read_csv(participants_path, sep="\t", keep_default_na=False)
    table["age"] = table["participant_id"].map(lambda name: participants[name]["age"])
    table["sex"] = table["participant_id"].map(lambda name: participants[name]["sex"])
    table.to_csv(participants_path, sep="\t", index=False)

    # CHANGES is the BIDS changelog; it is the only place the layout history of a
    # conversion is recorded, because the raw files themselves never moved.
    (bids_root / "CHANGES").write_text(changes, encoding="utf-8")
    readme = bids_root / "README"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else ""
    readme.write_text(existing.rstrip() + "\n\n" + readme_note + "\n", encoding="utf-8")
    print(f"{bids_root.name}: wrote {written} recordings")


# --- Proactive selective attention across competition contexts (OpenNeuro ds005089) ---


def standardize_selective_attention(replace: bool = True) -> None:
    """Convert OpenNeuro ds005089 into BIDS."""
    # =============================================================================
    # Step 0: source archive preflight
    # =============================================================================
    source_dir = ARCHIVE_DIR / "selective_attention"
    participants_source = source_dir / "participants.tsv"
    subject_labels = sorted(path.name for path in source_dir.glob("sub-*") if path.is_dir())
    if not subject_labels:
        raise FileNotFoundError(f"No selective-attention subject directories under {source_dir}")
    expected_paths = [participants_source] + [
        source_dir / label / "eeg" / f"{label}_task-competition_eeg.set"
        for label in subject_labels
    ]
    missing = [path for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Selective-attention archive is incomplete under {source_dir}: {len(missing)} "
            f"files are absent, among them {', '.join(path.name for path in missing[:5])}"
        )

    # Demographics are read and validated before the first BIDS byte: a table that
    # turns out unusable halfway through leaves a half-written dataset.
    demographics = pd.read_csv(participants_source, sep="\t").set_index("participant_id")
    if not set(subject_labels) <= set(demographics.index):
        raise ValueError(
            f"Selective-attention demographics in {participants_source} do not cover "
            f"{sorted(set(subject_labels) - set(demographics.index))}"
        )
    participant_sex = {"Male": "M", "Female": "F"}
    unknown_sex = set(demographics["sex"]) - set(participant_sex)
    if unknown_sex:
        raise ValueError(f"Unexpected sex values in {participants_source}: {sorted(unknown_sex)}")

    # The output root goes last, once the archive has been read and found complete:
    # clearing it earlier would let an incomplete archive destroy a conversion it
    # cannot rebuild.
    bids_root = BIDS_DIR / "selective_attention"
    if bids_root.exists():
        if not replace:
            raise FileExistsError(f"Selective-attention BIDS output already exists: {bids_root}")
        shutil.rmtree(bids_root)

    # =============================================================================
    # Step 1: source metadata and texts
    # =============================================================================
    montage = mne.channels.make_standard_montage("standard_1005")
    events_sidecar = {
        "trial_type": {
            "Description": (
                "Stimulus trigger label as shipped by the source release, with the "
                "padding whitespace of the BrainVision label removed. The release and "
                "its publication do not document what the individual codes stand for, "
                "so no condition is asserted here."
            ),
            "Levels": {
                "BAD_boundary": "Discontinuity inherited from the source EEGLAB file.",
            },
        },
    }
    eeg_sidecar = {
        "Manufacturer": "Brain Products",
        "ManufacturersModelName": "actiCap Slim, 64 active channels",
        "EEGReference": "FCz",
        "InstitutionName": "Mind, Brain and Behavior Research Center (CIMCYC), University of Granada",
        "TaskDescription": (
            "Cue-target sex-judgement task on faces and names. Blocks alternate between "
            "a high competition context, where target and distractor appear "
            "simultaneously, a low competition context, where they appear sequentially, "
            "and a stimulus category localizer."
        ),
    }
    montage_description = (
        "Static MNE-Python standard_1005 template montage assigned from the 63 channel "
        "names of the source release; not participant-specific digitization."
    )
    changes = (
        inspect.cleandoc("""
            1.0.0 2026-08-19
              - Initial conversion of OpenNeuro ds005089 v1.0.1 to the BIDS layout of
                this project.
            """)
        + "\n"
    )
    readme_note = inspect.cleandoc("""
        Selective attention: one recording, undecoded trigger codes
        -----------------------------------------------------------
        Each participant contributes exactly one continuous recording of roughly 6160
        seconds, so neither the session nor the run entity is used: there is nothing
        for either of them to distinguish. The original file is named in
        sub-*_scans.tsv as source.

        Events are the trigger labels of the source release, carried over unchanged
        apart from the padding whitespace of the BrainVision label ('S  1' becomes
        'S1'). They are deliberately not translated into condition names. The release
        ships no trigger documentation, the source publication lists none, and the
        authors' public materials (the OSF repository and the preprocessing code on
        GitHub) contain none either. Naming the codes from their counts would be a
        guess, and a plausible wrong guess is worse here than an undecoded label.

        What the publication does state about the design, for whoever decodes the
        codes later: 72 blocks per participant, 24 of each type -- high competition,
        low competition, and localizer. A main-task block holds 24 trials of 3.8 s
        (91.2 s per block, 576 trials per competition condition); a localizer block
        holds 48 trials of 1.25 s (60 s per block, 1152 localizer trials). A trial
        starts with a 50 ms cue, then a 1500 ms cue-target interval. In high
        competition the overlapping target and distractor are shown for 750 ms; in
        low competition the target is shown alone for 500 ms, overlapped for 250 ms,
        and the distractor alone for 500 ms. Until a code is tied to one of these by
        a source, this dataset carries no condition labels.
        """)

    # =============================================================================
    # Step 2: write one recording per source .set
    # =============================================================================
    bids_root.mkdir(parents=True)
    make_dataset_description(
        path=bids_root,
        dataset_type="raw",
        overwrite=True,
        verbose=False,
        name="Proactive selective attention across competition contexts",
        keywords=["EEG", "selective attention", "biased competition", "preparation"],
        authors=[
            "Blanca Aguado-Lopez", "Ana F. Palenciano", "Jose M. G. Penalver",
            "Paloma Diaz-Gutierrez", "David Lopez-Garcia", "Chiara Avancini",
            "Luis F. Ciria", "Maria Ruz",
        ],
        references_and_links=[
            "https://doi.org/10.18112/openneuro.ds005089.v1.0.1",
            "https://doi.org/10.1016/j.cortex.2024.04.009",
        ],
    )
    written = 0
    # This dataset has one recording per subject, so scans.tsv and coordsystem.json of
    # a subject are complete as soon as that recording is written; there is no second
    # pass and no collecting dictionary.
    for subject_label in subject_labels:
        set_path = source_dir / subject_label / "eeg" / f"{subject_label}_task-competition_eeg.set"
        # A few recordings were saved as a whole EEGLAB workspace rather than as one
        # dataset, so the file carries an ALLEEG array beside EEG and MNE refuses it
        # on sight. The EEG variable is an ordinary merged dataset, so a staged copy
        # holding only that variable is what gets read; the archive stays exactly as
        # the source shipped it. whosmat reads the header alone, and variable_names
        # keeps the discarded copy of the signal out of memory.
        if any(name == "ALLEEG" for name, _, _ in whosmat(set_path)):
            staging_dir = Path(tempfile.mkdtemp(prefix="selective_attention_set_"))
            staged_path = staging_dir / set_path.name
            contents = loadmat(set_path, variable_names=["EEG"])
            savemat(staged_path, {"EEG": contents["EEG"]}, appendmat=False)
            data_path = set_path.with_suffix(".fdt")
            if data_path.is_file():
                (staging_dir / data_path.name).symlink_to(data_path)
            raw = mne.io.read_raw_eeglab(staged_path, preload=True, verbose=False)
            shutil.rmtree(staging_dir)
        else:
            raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
        raw.set_montage(montage)
        raw.info["line_freq"] = 50
        # The trigger label is kept as the source wrote it, minus the padding that
        # right-aligns the number inside the BrainVision field. EEGLAB's 'boundary'
        # marks a data discontinuity; the BAD_ prefix is what makes MNE and this
        # project's preparation stage skip the span.
        labels = [
            "BAD_boundary" if description == "boundary" else description.replace(" ", "")
            for description in raw.annotations.description
        ]
        raw.set_annotations(
            mne.Annotations(
                onset=raw.annotations.onset,
                duration=[0.0] * len(labels),
                description=labels,
            )
        )

        written_path = write_raw_bids(
            raw,
            BIDSPath(
                subject=subject_label.removeprefix("sub-"),
                task="competition",
                datatype="eeg",
                root=bids_root,
            ),
            format="BrainVision",
            allow_preload=True,
            extra_columns_descriptions={},
            verbose=False,
        )
        # The label vocabulary and the device facts a BrainVision export cannot carry.
        update_sidecar_json(
            written_path.copy().update(suffix="events", extension=".json"),
            events_sidecar,
            verbose=False,
        )
        update_sidecar_json(
            written_path.copy().update(suffix="eeg", extension=".json"),
            eeg_sidecar,
            verbose=False,
        )

        # =============================================================================
        # Step 3: provenance sidecars
        # =============================================================================
        coordinate_json = written_path.copy().update(
            task=None, space="CapTrak", suffix="coordsystem", extension=".json", check=False,
        ).fpath
        # update_sidecar_json cannot remove the placeholder landmarks emitted by
        # mne-bids, so this sidecar needs one direct read-modify-write.
        coordsystem_metadata = json.loads(coordinate_json.read_text(encoding="utf-8"))
        coordsystem_metadata["EEGCoordinateSystemDescription"] = (
            f"{montage_description} Coordinates use the CapTrak/RAS convention for BIDS "
            "and MNE interoperability; they were not measured with a CapTrak device."
        )
        for key in (
            "AnatomicalLandmarkCoordinates",
            "AnatomicalLandmarkCoordinateSystem",
            "AnatomicalLandmarkCoordinateUnits",
        ):
            coordsystem_metadata.pop(key, None)
        coordinate_json.write_text(
            json.dumps(coordsystem_metadata, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        scans_path = written_path.copy().update(
            task=None, datatype=None, suffix="scans", extension=".tsv", check=False
        ).fpath
        scans = pd.read_csv(scans_path, sep="\t", keep_default_na=False)
        if len(scans) != 1:
            raise ValueError(f"Expected one recording in {scans_path}, found {len(scans)}")
        scans["source"] = set_path.name
        scans.to_csv(scans_path, sep="\t", index=False)
        scans_path.with_suffix(".json").write_text(
            json.dumps({"source": {"Description": "Original source filename."}}, indent=4) + "\n",
            encoding="utf-8",
        )
        written += 1

    # No sessions.tsv: every participant was recorded in a single visit, so there is
    # no session of which to state a role or a quality.

    # =============================================================================
    # Step 4: dataset-level texts
    # =============================================================================
    # Individual age and sex are published by the release, so they are stated rather
    # than left n/a. write_raw_bids has already created the table with n/a columns.
    participants_path = bids_root / "participants.tsv"
    table = pd.read_csv(participants_path, sep="\t", keep_default_na=False)
    table["age"] = table["participant_id"].map(demographics["age"])
    table["sex"] = table["participant_id"].map(demographics["sex"]).map(participant_sex)
    table.to_csv(participants_path, sep="\t", index=False)

    # CHANGES is the BIDS changelog; it is the only place the layout history of a
    # conversion is recorded, because the raw files themselves never moved.
    (bids_root / "CHANGES").write_text(changes, encoding="utf-8")
    readme = bids_root / "README"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else ""
    readme.write_text(existing.rstrip() + "\n\n" + readme_note + "\n", encoding="utf-8")
    print(f"{bids_root.name}: wrote {written} recordings")


# --- Cognitive tasks of the TU Berlin EEG-NIRS release ---


def standardize_cognitive_tasks(replace: bool = True) -> None:
    """Convert the BrainVision EEG half of the TU Berlin EEG-NIRS release into BIDS."""
    # =============================================================================
    # Step 0: source archive preflight
    # =============================================================================
    source_dir = ARCHIVE_DIR / "cognitive_tasks_eeg"
    # The source file name does not name the task: the release's own description maps
    # nback to dataset A (n-back), gonogo to dataset B (discrimination/selection
    # response) and word to dataset C (word generation).
    task_names = {"nback": "nback", "gonogo": "dsr", "word": "wordgen"}
    subject_labels = sorted(path.name for path in source_dir.glob("VP[0-9][0-9][0-9]") if path.is_dir())
    if not subject_labels:
        raise FileNotFoundError(f"No cognitive-tasks subject directories under {source_dir}")
    source_stems = [f"{stem}{repetition}" for stem in task_names for repetition in (1, 2, 3)]
    expected_paths = [
        source_dir / label / f"{stem}{extension}"
        for label in subject_labels
        for stem in source_stems
        for extension in (".vhdr", ".eeg", ".vmrk")
    ]
    missing = [path for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cognitive-tasks archive is incomplete under {source_dir}: {len(missing)} files "
            f"are absent, among them {', '.join(path.name for path in missing[:5])}"
        )

    # One header of the release names files that do not exist: VP017/word2.vhdr
    # points at wor2.eeg and wor2.vmrk. The archive stays exactly as the source
    # shipped it, so that header and its marker file are repaired in a staging copy
    # that links back to the real data, and every recording is read through
    # header_paths afterwards.
    staging_dir = Path(tempfile.mkdtemp(prefix="cognitive_tasks_headers_"))
    header_paths: dict[tuple[str, str], Path] = {}
    for subject_label in subject_labels:
        for stem in source_stems:
            header_path = source_dir / subject_label / f"{stem}.vhdr"
            marker_path = source_dir / subject_label / f"{stem}.vmrk"
            header = header_path.read_text(encoding="utf-8")
            marker = marker_path.read_text(encoding="utf-8")
            if (
                f"DataFile={stem}.eeg" in header
                and f"MarkerFile={stem}.vmrk" in header
                and f"DataFile={stem}.eeg" in marker
            ):
                header_paths[(subject_label, stem)] = header_path
                continue
            staging_subject = staging_dir / subject_label
            staging_subject.mkdir(parents=True, exist_ok=True)
            for staged_name, source_text in (
                (f"{stem}.vhdr", header), (f"{stem}.vmrk", marker)
            ):
                (staging_subject / staged_name).write_text(
                    "\n".join(
                        f"DataFile={stem}.eeg" if line.startswith("DataFile=")
                        else f"MarkerFile={stem}.vmrk" if line.startswith("MarkerFile=")
                        else line
                        for line in source_text.splitlines()
                    ),
                    encoding="utf-8",
                )
            (staging_subject / f"{stem}.eeg").symlink_to(
                source_dir / subject_label / f"{stem}.eeg"
            )
            header_paths[(subject_label, stem)] = staging_subject / f"{stem}.vhdr"

    # The output root goes last, once the archive has been read and found complete:
    # clearing it earlier would let an incomplete archive destroy a conversion it
    # cannot rebuild.
    bids_root = BIDS_DIR / "cognitive_tasks"
    if bids_root.exists():
        if not replace:
            raise FileExistsError(f"Cognitive-tasks BIDS output already exists: {bids_root}")
        shutil.rmtree(bids_root)

    # =============================================================================
    # Step 1: source metadata and texts
    # =============================================================================
    montage = mne.channels.make_standard_montage("standard_1005")
    # The BrainVision export writes four labels differently from the publication,
    # which the release's description states in its own comparison table.
    channel_renames = {"FP1": "Fp1", "FP2": "Fp2", "AFF5": "AFF5h", "AFF6": "AFF6h"}
    # Marker vocabulary of the release's description, table 2 to table 4. The same
    # code means different things in different tasks -- S48 is a 2-back target in
    # dataset A and a series marker in dataset B -- so each task reads its own table.
    block_markers = {
        "nback": {"S112": "0-back", "S128": "2-back", "S144": "3-back"},
        "dsr": {"S48": "dsr"},
        "wordgen": {"S16": "word_generation", "S32": "baseline"},
    }
    trial_markers = {
        "nback": {"S16", "S48", "S64", "S80", "S96"},
        "dsr": {"S16", "S32"},
        "wordgen": set(),
    }
    # The publication documents a 10 s task period for word generation and baseline;
    # for the other two the block is measured from its own trials, because the
    # recordings run slightly longer than the nominal 40 s task period.
    wordgen_block_duration = 10.0
    events_sidecar = {
        "trial_type": {
            "Description": "Task block the participant performed during this time segment.",
            "Levels": {
                "0-back": "0-back block; the participant reports whether the current digit is the target.",
                "2-back": "2-back block; the participant compares the current digit with the one two steps back.",
                "3-back": "3-back block; the participant compares the current digit with the one three steps back.",
                "dsr": "Discrimination/selection response block; the participant responds to the symbol O and withholds for X.",
                "word_generation": "Word generation block; the participant silently generates words starting with a given letter.",
                "baseline": "Baseline block of the word generation task; the participant gazes at the fixation cross.",
            },
        },
    }
    eeg_sidecar = {
        "Manufacturer": "Brain Products",
        "ManufacturersModelName": "BrainAmp",
        "EEGReference": "TP9",
        "EEGGround": "TP10",
        "InstitutionName": "Berlin Institute of Technology",
        "TaskDescription": (
            "Cognitive tasks of the open access EEG-NIRS release: n-back at three "
            "working-memory loads (0-, 2- and 3-back), a discrimination/selection "
            "response task, and word generation against a fixation baseline."
        ),
    }
    montage_description = (
        "Static MNE-Python standard_1005 template montage assigned from the 28 channel "
        "names published for this release; not participant-specific digitization."
    )
    changes = (
        inspect.cleandoc("""
            1.0.0 2026-08-19
              - Initial conversion of the BrainVision EEG half of the TU Berlin
                EEG-NIRS release to the BIDS layout of this project.
            """)
        + "\n"
    )
    readme_note = inspect.cleandoc("""
        Cognitive tasks: nine recordings per participant, blocks as events
        ------------------------------------------------------------------
        Every participant contributes nine recordings: three repetitions each of the
        n-back (task-nback), discrimination/selection response (task-dsr) and word
        generation (task-wordgen) tasks. The source calls a repetition a session, but
        all nine were recorded in one visit of roughly 3.5 hours without the cap
        being reapplied, so no session entity is used, the same decision this project
        already took for the SAM 40 trials.

        The run entity is a unique recording index within the participant, assigned
        by acquisition time, which the BrainVision header states. Run numbers are
        therefore continuous across the three tasks rather than restarting at 1 for
        each: run-01..09 identify nine separate recordings, and no two of them share
        a number. That matters downstream, where a recording unit is the triple
        (subject, session, run): had the numbering restarted per task, three
        unrelated recordings would have been grouped into one unit. The task order
        differs between participants -- the release notes that VP001 and VP002 were
        recorded in a different order from the rest -- so the mapping is read from
        the files rather than assumed. The original file behind each recording is
        named in sub-*_scans.tsv as source.

        Events are task blocks, in the vocabulary of the release's own marker tables.
        A block starts at the series marker the release defines and lasts as long as
        its trials: for n-back and the discrimination task the end is the last trial
        of the series plus one inter-trial interval, because the recorded series run
        slightly longer than the nominal 40 s task period; for word generation the
        documented 10 s task period is used. Trial-level markers -- target against
        non-target, symbol O against symbol X -- are not carried over: this project
        works on block segments, and the source markers stay in the archive.

        One discrepancy is worth naming. The publication describes three series of
        the discrimination task per recording, but the marker files contain six, with
        the 20 trials per series the publication states. The conversion follows the
        files.

        Individual age and sex stay n/a: the publication reports only group
        demographics (26 participants, 9 male and 17 female, 26.1 +/- 3.5 years).
        """)

    # =============================================================================
    # Step 2: write one recording per source header
    # =============================================================================
    bids_root.mkdir(parents=True)
    make_dataset_description(
        path=bids_root,
        dataset_type="raw",
        overwrite=True,
        verbose=False,
        name="EEG of cognitive tasks from the open access EEG-NIRS dataset",
        keywords=["EEG", "n-back", "working memory", "word generation", "cognitive load"],
        authors=[
            "Jaeyoung Shin", "Alexander von Luehmann", "Do-Won Kim", "Jan Mehnert",
            "Han-Jeong Hwang", "Klaus-Robert Mueller",
        ],
        references_and_links=[
            "https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/",
            "https://doi.org/10.1038/sdata.2018.3",
        ],
    )
    written = 0
    # A subject directory owns one scans.tsv covering all nine of its recordings, so
    # both it and coordsystem.json can only be finalized after the last write_raw_bids
    # call for that subject. The source has no sessions, so the directory below the
    # root is sub-XX/eeg.
    subject_dirs: dict[Path, tuple[BIDSPath, dict[str, str]]] = {}
    for subject_number, subject_label in enumerate(subject_labels, start=1):
        # The run entity follows acquisition time, which the BrainVision header
        # records, so the three tasks interleave the way the participant met them.
        recording_times = {}
        for stem in source_stems:
            header = mne.io.read_raw_brainvision(
                header_paths[(subject_label, stem)], preload=False, verbose=False
            )
            recording_times[stem] = header.info["meas_date"]
        if len(set(recording_times.values())) != len(source_stems):
            raise ValueError(
                f"Cognitive-tasks recordings of {subject_label} do not have distinct "
                f"acquisition times: {recording_times}"
            )

        for run, stem in enumerate(sorted(source_stems, key=recording_times.get), start=1):
            bids_task = task_names[stem[:-1]]
            vhdr_path = header_paths[(subject_label, stem)]
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
            raw.rename_channels(channel_renames)
            raw.set_channel_types({"HEOG": "eog", "VEOG": "eog"})
            raw.set_montage(montage)
            raw.info["line_freq"] = 50
            # The impedances the header carries were measured for this recording
            # alone, while electrodes.tsv belongs to the subject and holds template
            # coordinates for all nine. Writing one recording's measurement into a
            # subject-level file would state it of the other eight as well, and
            # mne-bids refuses the second recording outright when the values differ.
            raw.impedances = {}

            # MNE prefixes a BrainVision marker with its type and keeps the padding
            # that right-aligns the number inside the label field.
            labels = [
                description.split("/")[-1].replace(" ", "")
                for description in raw.annotations.description
            ]
            onsets = list(raw.annotations.onset)
            block_onsets, block_durations, block_labels = [], [], []
            starts = [index for index, label in enumerate(labels) if label in block_markers[bids_task]]
            if not starts:
                raise ValueError(f"No block markers found in {vhdr_path}")
            for position, index in enumerate(starts):
                limit = starts[position + 1] if position + 1 < len(starts) else len(labels)
                if bids_task == "wordgen":
                    duration = wordgen_block_duration
                else:
                    # The block covers its own trials: the nominal 40 s task period of
                    # the publication ends before the last trials actually recorded.
                    trials = [
                        onsets[trial]
                        for trial in range(index, limit)
                        if labels[trial] in trial_markers[bids_task]
                    ]
                    if len(trials) < 2:
                        raise ValueError(
                            f"Block at {onsets[index]:.2f} s in {vhdr_path} has "
                            f"{len(trials)} trials; a block is defined by its trials"
                        )
                    intervals = sorted(
                        second - first for first, second in zip(trials, trials[1:])
                    )
                    duration = trials[-1] - onsets[index] + intervals[len(intervals) // 2]
                block_onsets.append(onsets[index])
                block_durations.append(duration)
                block_labels.append(block_markers[bids_task][labels[index]])
            raw.set_annotations(
                mne.Annotations(
                    onset=block_onsets, duration=block_durations, description=block_labels
                )
            )

            written_path = write_raw_bids(
                raw,
                BIDSPath(
                    subject=f"{subject_number:02d}",
                    task=bids_task,
                    run=f"{run:02d}",
                    datatype="eeg",
                    root=bids_root,
                ),
                format="BrainVision",
                allow_preload=True,
                extra_columns_descriptions={},
                verbose=False,
            )
            # The label vocabulary and the device facts a BrainVision export cannot
            # carry.
            update_sidecar_json(
                written_path.copy().update(suffix="events", extension=".json"),
                events_sidecar,
                verbose=False,
            )
            update_sidecar_json(
                written_path.copy().update(suffix="eeg", extension=".json"),
                eeg_sidecar,
                verbose=False,
            )
            # write_raw_bids supplies the final ``*_eeg.vhdr`` path. The input BIDSPath
            # has no suffix or extension yet and cannot be matched to the filename
            # stored in scans.tsv.
            _, sources = subject_dirs.setdefault(written_path.directory, (written_path, {}))
            sources[f"{written_path.datatype}/{written_path.fpath.name}"] = (
                f"{subject_label}/{stem}.vhdr"
            )
            written += 1

    # =============================================================================
    # Step 3: provenance sidecars
    # =============================================================================
    # Coordinate provenance and raw-file provenance live at the same level of the
    # tree, so both are finalized in the same pass.
    for bids_path, sources in subject_dirs.values():
        coordinate_json = bids_path.copy().update(
            task=None, run=None, space="CapTrak", suffix="coordsystem", extension=".json",
            check=False,
        ).fpath
        # Not update_sidecar_json: it can add a key but never remove one, and the
        # landmark placeholders mne-bids emits have to go.
        metadata = json.loads(coordinate_json.read_text(encoding="utf-8"))
        metadata["EEGCoordinateSystemDescription"] = (
            f"{montage_description} Coordinates use the CapTrak/RAS convention for BIDS "
            "and MNE interoperability; they were not measured with a CapTrak device."
        )
        for key in (
            "AnatomicalLandmarkCoordinates",
            "AnatomicalLandmarkCoordinateSystem",
            "AnatomicalLandmarkCoordinateUnits",
        ):
            metadata.pop(key, None)
        coordinate_json.write_text(
            json.dumps(metadata, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # The table can only be finalized now: write_raw_bids rewrites it for every
        # recording of the subject, so a column added earlier would not survive.
        scans_path = bids_path.copy().update(
            task=None, run=None, datatype=None, suffix="scans", extension=".tsv", check=False
        ).fpath
        scans = pd.read_csv(scans_path, sep="\t", keep_default_na=False)
        scans["source"] = scans["filename"].map(sources)
        if scans["source"].isna().any():
            missing_sources = scans.loc[scans["source"].isna(), "filename"].tolist()
            raise ValueError(f"No source file recorded for {missing_sources} in {scans_path}")
        scans.to_csv(scans_path, sep="\t", index=False)
        scans_path.with_suffix(".json").write_text(
            json.dumps({"source": {"Description": "Original source filename."}}, indent=4) + "\n",
            encoding="utf-8",
        )

    # No sessions.tsv: the nine recordings of a participant belong to one visit, so
    # there is no session of which to state a role or a quality.

    # =============================================================================
    # Step 4: dataset-level texts
    # =============================================================================
    # CHANGES is the BIDS changelog; it is the only place the layout history of a
    # conversion is recorded, because the raw files themselves never moved.
    (bids_root / "CHANGES").write_text(changes, encoding="utf-8")
    readme = bids_root / "README"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else ""
    readme.write_text(existing.rstrip() + "\n\n" + readme_note + "\n", encoding="utf-8")
    shutil.rmtree(staging_dir)
    print(f"{bids_root.name}: wrote {written} recordings")
