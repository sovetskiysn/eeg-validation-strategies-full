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
        print(f"sam40: writing {bids_task} recordings ({written} written so far)")
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
    # The source archive retains the canonical 34 files directly in this directory.
    # It also carries a nested byte-identical copy;
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
    previous_subject: int | None = None
    for index in sorted(recording_paths):
        mat_path = recording_paths[index]
        current_subject = min((index - 1) // 7 + 1, 5)
        if current_subject != previous_subject:
            print(f"distinguishing: writing sub-{current_subject:02d} ({written} written so far)")
            previous_subject = current_subject
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
        subject, session = current_subject, (index - 1) % 7 + 1
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
