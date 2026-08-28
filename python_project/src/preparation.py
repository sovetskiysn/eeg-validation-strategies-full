"""Stage 1: cache routing and continuous EEG preparation.

``get_dataset_dir`` addresses and reuses cache entries; ``_prepare_dataset_artifact``
runs the EEG recipe when that address has not been built yet.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from itertools import groupby
from pathlib import Path

import mne
import mne_bids
import numpy as np
import pandas as pd
from mne.preprocessing import ICA, annotate_amplitude, find_bad_channels_lof
from omegaconf import DictConfig, OmegaConf
from utils import BIDS_DIR, PREPARED_CACHE_ROOT

# The full labelled listing of every release: the class a condition belongs to,
# named, and the conditions that make it up. The class label is the key's
# position, so it is never written down; the key exists to say what the group is,
# which an anonymous sublist could not.
#
# The class names are deliberately the same for every dataset and in the same
# order, because transfer trains on one release and tests on another: class 1 has
# to mean the same thing on both sides. Naming them per dataset ("stress" against
# "focused") would hide that the shared construct is what is being transferred --
# and an inverted class order would then produce a plausible-looking result
# instead of an error. What each dataset actually manipulated stays visible in
# the condition names stored with every epoch.
#
# It lives here and not in `configs/dataset/` because the config may only
# subtract names from it -- it can neither move a condition between classes nor
# invent one, and the analysis it asks for is then a deletion instead of a
# rewritten mapping.
#
# The price is that a listing kept in code can go stale while nobody touches it,
# so Step 3 compares it against the annotations the recordings actually carry, in
# both directions. The other price is that it is invisible to the artifact hash:
# editing it changes the output of Stage 1 without changing the address of the
# cache entry, which is exactly what `preparation.provenance.version` exists to
# fix.
DATASET_MAPPING = {
    "sam40": {
        "low_attention": ["relax"],
        "high_attention": ["stroop", "arithmetic", "mirror"],
    },
    "distinguishing": {
        "low_attention": ["unfocused", "drowsy"],
        "high_attention": ["focused"],
    },
}


def get_dataset_dir(dataset_config: DictConfig, preparation_config: DictConfig) -> Path:
    """Return the cached artifact, preparing it when it is absent."""
    dataset = OmegaConf.to_container(dataset_config, resolve=True)
    preparation = OmegaConf.to_container(preparation_config, resolve=True)
    artifact_id = hashlib.sha256(
        json.dumps(
            {"dataset": dataset, "preparation": preparation},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:8]
    artifact_dir = PREPARED_CACHE_ROOT / artifact_id

    if artifact_dir.exists():
        config_path = artifact_dir / "config.yaml"
        if not config_path.exists():
            raise FileExistsError(
                f"Cache entry has no config.yaml: {artifact_dir}. "
                "Move it aside by hand; an existing artifact is never rebuilt over."
            )
        saved_config = OmegaConf.load(config_path)
        saved_dataset = OmegaConf.to_container(saved_config.dataset, resolve=True)
        saved_preparation = OmegaConf.to_container(saved_config.preparation, resolve=True)
        if saved_dataset != dataset or saved_preparation != preparation:
            raise ValueError(
                f"{artifact_dir} holds a different recipe than the one addressing it. "
                "Move it aside by hand; an existing artifact is never rebuilt over."
            )
        print(f"Reusing {artifact_dir}")
        return artifact_dir

    return _prepare_dataset_artifact(dataset_config, preparation_config, artifact_dir)


def _prepare_dataset_artifact(
    ds_cfg: DictConfig,
    prep_cfg: DictConfig,
    artifact_dir: Path,
) -> Path:
    """Run the EEG recipe and atomically publish a new artifact."""
    dataset_name = ds_cfg.name
    window_size = prep_cfg.epoching.window_size

    if prep_cfg.quality.peak_uv <= 0:
        raise ValueError("quality.peak_uv must be positive.")
    if prep_cfg.quality.flat_uv < 0:
        raise ValueError("quality.flat_uv must be non-negative.")
    if prep_cfg.quality.min_duration_s <= 0:
        raise ValueError("quality.min_duration_s must be positive.")
    if not 0 < prep_cfg.quality.bad_percent <= 100:
        raise ValueError("quality.bad_percent must lie in (0, 100].")
    if prep_cfg.quality.lof_n_neighbors < 1:
        raise ValueError("quality.lof_n_neighbors must be positive.")
    if prep_cfg.quality.lof_threshold <= 0:
        raise ValueError("quality.lof_threshold must be positive.")
    if prep_cfg.quality.max_auto_bad_channels < 0:
        raise ValueError("quality.max_auto_bad_channels must be non-negative.")

    # The analysis is the full listing of the release minus the names the config
    # asks to drop; the class's position is the label. Everything below works
    # from the flat dict built here.
    if dataset_name not in DATASET_MAPPING:
        raise ValueError(
            f"No condition mapping for dataset {dataset_name!r}: DATASET_MAPPING in "
            f"preparation.py knows {sorted(DATASET_MAPPING)}."
        )
    classes = DATASET_MAPPING[dataset_name]
    mapped_conditions = {name for group in classes.values() for name in group}
    listed = sum(len(group) for group in classes.values())
    if len(classes) != 2 or not all(classes.values()) or len(mapped_conditions) != listed:
        raise ValueError(
            f"DATASET_MAPPING[{dataset_name!r}] must be two non-empty classes of condition "
            f"names, with no name in both; got {classes}."
        )
    # The class label is the position of its name, so this list turns a label
    # back into the name -- it is what `event_id` is written with.
    class_names = list(classes)

    excluded_conditions = set(OmegaConf.to_container(ds_cfg.exclude_conditions, resolve=True))
    # A name that matches nothing is a typo in the recipe, and continuing would
    # quietly compute a different experiment than the one that was asked for.
    unknown = sorted(excluded_conditions - mapped_conditions)
    if unknown:
        raise ValueError(
            f"{dataset_name}: exclude_conditions {unknown} name no condition of this dataset, "
            f"which has {sorted(mapped_conditions)}."
        )
    conditions = {
        name: label
        for label, group in enumerate(classes.values())
        for name in group
        if name not in excluded_conditions
    }
    emptied = [name for name, group in classes.items() if not set(group) - excluded_conditions]
    if emptied:
        raise ValueError(
            f"{dataset_name}: exclude_conditions {sorted(excluded_conditions)} leaves class "
            f"{emptied} empty, so the task is no longer binary."
        )

    # =============================================================================
    # Step 1: source recordings, grouped into ICA units
    # =============================================================================
    # One unit holds the recordings that share an ICA fit, and it is identified
    # by `(subject, session, run)` with the entity a dataset does not use left as
    # None: a Distinguishing session is ("01", "03", None), a SAM40 source trial
    # is ("01", None, "01") shared by its four task recordings. Naming both
    # entities rather than picking one keeps this correct for a dataset that
    # someday uses runs inside real sessions.
    #
    # `find_matching_paths` globs, so its order is arbitrary, and `groupby` only
    # groups neighbours -- hence an explicit sort key, with `task` last so a
    # unit's recordings stay adjacent. The BIDS path cannot serve as that key:
    # `run` is a suffix entity, so paths sort task-major and one trial would
    # break into four non-adjacent groups.
    bids_root = BIDS_DIR / dataset_name
    recordings = sorted(
        mne_bids.find_matching_paths(
            bids_root, datatypes="eeg", suffixes="eeg", extensions=".vhdr", check=True
        ),
        # `or ""` only so None sorts against str; it never selects an entity.
        key=lambda r: (r.subject, r.session or "", r.run or "", r.task),
    )
    if not recordings:
        raise FileNotFoundError(f"No EEG BrainVision recordings found in {bids_root}")

    # =============================================================================
    # Step 2: availability and quality selection
    # =============================================================================
    # Which recordings enter the analysis is decided before a single one is
    # loaded: a subject's sessions table is read once instead of once per unit,
    # and broken BIDS stops the run before the expensive work.
    #
    # The config names columns and the values to drop; this code knows neither.
    # It reads the named columns out of the actual sessions.tsv, checks that
    # both the column and every excluded value are really there, and drops the
    # rows that match. `keep_default_na=False` keeps BIDS `n/a` the literal
    # string it is, so a column that says nothing shows up in the summary below
    # instead of turning into a float.
    #
    # The criterion applies only where the dataset has source sessions to state
    # facts about. SAM40 has none, so it has no sessions.tsv and every recording
    # is available -- that is the source having no confirmed ground to exclude a
    # trial, not a missing file to work around. This branch has to come first:
    # under it the loud failures below would reject every SAM40 run for a table
    # that is correctly absent.
    exclusions = OmegaConf.to_container(
        prep_cfg.selection.exclude_session_values, resolve=True
    )
    if any(recording.session is not None for recording in recordings):
        # The one column name this code knows, because it is the key it joins
        # on -- and naming it here would be an enumeration of records by id,
        # which is a stored answer rather than a criterion.
        if "session_id" in exclusions:
            raise ValueError(
                "exclude_session_values cannot select on session_id: listing the records to "
                "drop states no reason and goes stale silently. Exclude by a fact instead."
            )

        session_rows: dict[tuple[str, str], dict[str, str]] = {}
        for subject in sorted({recording.subject for recording in recordings}):
            sessions_path = mne_bids.BIDSPath(
                root=bids_root, subject=subject, suffix="sessions", extension=".tsv", check=False
            ).fpath
            sessions = pd.read_csv(sessions_path, sep="\t", dtype=str, keep_default_na=False)
            missing_columns = {"session_id", *exclusions} - set(sessions.columns)
            if missing_columns:
                raise ValueError(
                    f"{sessions_path} has no column {', '.join(sorted(missing_columns))}, which "
                    "exclude_session_values selects on."
                )
            if sessions["session_id"].duplicated().any():
                raise ValueError(f"Broken sessions.tsv at {sessions_path}: duplicate session_id")
            for row in sessions.to_dict("records"):
                session_rows[(subject, row["session_id"].removeprefix("ses-"))] = row

        rows = []
        for recording in recordings:
            if (recording.subject, recording.session) not in session_rows:
                raise ValueError(
                    f"No ses-{recording.session} row in the sessions.tsv of "
                    f"sub-{recording.subject} under {bids_root}"
                )
            rows.append(session_rows[(recording.subject, recording.session)])
        available = [
            recording
            for recording, row in zip(recordings, rows, strict=True)
            if not any(row[column] in values for column, values in exclusions.items())
        ]

        # An exclusion is silent by construction, so what each column actually
        # holds is printed: that summary is the only thing that makes a value
        # nobody excluded visible. A value that matches nothing is a typo, and
        # a recipe that keeps nothing is not a cohort.
        summary = []
        for column, values in exclusions.items():
            counts = Counter(row[column] for row in rows)
            unseen = sorted(set(values) - set(counts))
            if unseen:
                raise ValueError(
                    f"{dataset_name}: exclude_session_values[{column}] excludes {unseen}, which "
                    f"no recording carries. The column holds {sorted(counts)}."
                )
            summary.append(f"{column} " + " ".join(f"{v}={n}" for v, n in sorted(counts.items())))
        print(f"{dataset_name}: {' | '.join(summary)} -> kept {len(available)}/{len(recordings)}")
        if not available:
            raise ValueError(
                f"{dataset_name}: exclude_session_values leaves no recording at all under "
                f"{bids_root}."
            )
    else:
        available = list(recordings)

    # =============================================================================
    # Step 3: label selection
    # =============================================================================
    # A recording the label mapping does not ask for has to disappear here, before
    # the ICA fit below: its signal must not shape the decomposition of an
    # experiment that did not select it. Reading the header and events.tsv is
    # cheap, `load_data()` further down is not, so the Raw is dropped again.
    #
    # One rule covers both datasets because both answer the same question. For
    # SAM40 it discards whole task recordings, since one recording is one block;
    # for Distinguishing it discards nothing, since every block of interest lies
    # inside the one recording of the session and is cut out at epoching.
    dataset_labels: set[str] = set()
    selected = []
    for recording in available:
        # `str` because MNE hands back numpy strings, which print as np.str_(...)
        # in the error below and compare fine but read terribly.
        descriptions = {
            str(ann["description"])
            for ann in mne_bids.read_raw_bids(recording, verbose=False).annotations
        }
        dataset_labels |= descriptions
        if descriptions & conditions.keys():
            selected.append(recording)

    # What keeps a mapping written in code from going stale: it is compared with
    # the names the recordings actually carry, both ways. A name only in code
    # means ingestion dropped or renamed a condition; a name only in the data
    # means ingestion added one and nobody labelled it. Neither would fail on its
    # own -- they would quietly shrink the cohort or leave a class empty.
    # `BAD_*` spans are MNE's mark of unusable signal, not a condition.
    found = {name for name in dataset_labels if not name.startswith("BAD_")}
    if found != mapped_conditions:
        raise ValueError(
            f"{dataset_name}: DATASET_MAPPING and the annotations under {bids_root} disagree. "
            f"Only in code: {sorted(mapped_conditions - found)}; "
            f"only in the data: {sorted(found - mapped_conditions)}."
        )
    print(
        f"{dataset_name}: selected {len(selected)} of {len(recordings)} recordings "
        f"({len(available)} available, {len(selected)} carrying {sorted(conditions)}; "
        f"excluded {sorted(excluded_conditions)})"
    )

    # =============================================================================
    # Step 4: load, clean, decompose and window each ICA unit
    # =============================================================================
    # Every recording of the unit is filtered on its own; only the ICA fit is
    # shared, because the recordings of a unit were acquired without the montage
    # being disturbed in between. For SAM40 one task alone is 25 s (3200 samples)
    # for 32 channels, so the selected tasks of the trial are concatenated to make
    # the fit usable at all -- the excluded ones are already gone and are never
    # pulled back in for the sake of fit length.

    dataset_epochs: list[mne.Epochs] = []

    for unit_index, ((subject, session, run), unit_recordings) in enumerate(
        groupby(selected, key=lambda r: (r.subject, r.session, r.run)), start=1
    ):
        unit_recordings = list(unit_recordings)
        # The unit is named by the entities the dataset actually uses; one using
        # both would simply be named by both.
        unit_id = "_".join(
            [dataset_name, subject]
            + ([f"ses-{session}"] if session is not None else [])
            + ([f"run-{run}"] if run is not None else [])
        )
        print(f"[{dataset_name} {unit_index}] {unit_id}")

        # Load, filter and place every recording in the common sensor space before
        # fitting the shared ICA.  The two releases arrive with 14 and 32 EEG
        # channels respectively; applying an average reference first would make
        # the retained 12 channels depend on those dataset-specific extras.
        # Selecting the fixed intersection before QC, rereferencing and ICA makes
        # the continuous representation identical on both sides of a cross-dataset
        # transfer.
        unit_raws = []
        for recording in unit_recordings:
            raw = mne_bids.read_raw_bids(recording, verbose=False).load_data().pick("eeg")

            notch_freqs = [
                f
                for f in prep_cfg.filtering.notch_frequencies
                if f < raw.info["sfreq"] / 2
            ]
            if notch_freqs:
                raw.notch_filter(notch_freqs, verbose=False)
            raw.filter(
                prep_cfg.filtering.l_freq, prep_cfg.filtering.h_freq, verbose=False
            )
            raw.pick(prep_cfg.epoching.channels)

            # Gross transient and flatline detection is deliberately condition-blind:
            # it sees the full valid recording (including drowsy when it is not a
            # classification condition) and only annotates signal quality.  These
            # BAD_* spans are retained through ICA fitting and Epochs creation.
            quality_annotations, amplitude_bads = annotate_amplitude(
                raw,
                peak={"eeg": prep_cfg.quality.peak_uv * 1e-6},
                flat={"eeg": prep_cfg.quality.flat_uv * 1e-6},
                bad_percent=prep_cfg.quality.bad_percent,
                min_duration=prep_cfg.quality.min_duration_s,
                verbose=False,
            )
            raw.set_annotations(raw.annotations + quality_annotations)

            if prep_cfg.quality.lof_n_neighbors >= len(raw.ch_names):
                raise ValueError(
                    f"{recording.basename}: quality.lof_n_neighbors="
                    f"{prep_cfg.quality.lof_n_neighbors} needs fewer than the "
                    f"{len(raw.ch_names)} selected channels."
                )
            lof_bads, lof_scores = find_bad_channels_lof(
                raw,
                n_neighbors=prep_cfg.quality.lof_n_neighbors,
                threshold=prep_cfg.quality.lof_threshold,
                return_scores=True,
                verbose=False,
            )
            auto_bads = sorted(set(amplitude_bads) | set(lof_bads))
            if len(auto_bads) > prep_cfg.quality.max_auto_bad_channels:
                raise ValueError(
                    f"{recording.basename}: automatic QC found {len(auto_bads)} bad "
                    f"channels {auto_bads} (amplitude={sorted(amplitude_bads)}, "
                    f"LOF={sorted(lof_bads)}). The recipe permits at most "
                    f"{prep_cfg.quality.max_auto_bad_channels}; inspect this recording "
                    "rather than silently interpolating several channels."
                )

            annotation_counts = Counter(quality_annotations.description)
            annotation_duration_s = sum(quality_annotations.duration)
            lof_score_text = ", ".join(
                f"{name}={score:.2f}"
                for name, score in zip(raw.ch_names, lof_scores, strict=True)
            )
            if auto_bads:
                raw.info["bads"] = sorted(set(raw.info["bads"]) | set(auto_bads))
                # Keep the fixed 12-channel geometry, but prevent the faulty signal
                # from influencing average reference or ICA.
                raw.interpolate_bads(reset_bads=True, verbose=False)
            if quality_annotations or auto_bads:
                print(
                    f"{recording.basename}: QC BAD_peak={annotation_counts['BAD_peak']} "
                    f"BAD_flat={annotation_counts['BAD_flat']} "
                    f"spans={annotation_duration_s:.2f}s amplitude={sorted(amplitude_bads)} "
                    f"LOF={sorted(lof_bads)} interpolated={auto_bads} "
                    f"LOF_scores=[{lof_score_text}]"
                )
            raw.set_eeg_reference(prep_cfg.filtering.reference, projection=False, verbose=False)
            unit_raws.append(raw)

        # Drifts below ica_fit_l_freq destabilise the decomposition, but
        # removing them from the analysis signal itself is not wanted --
        # hence a separate, more aggressively high-passed copy to fit on.
        ica_fit_raw = mne.concatenate_raws([raw.copy() for raw in unit_raws], verbose=False).filter(
            prep_cfg.ica.ica_fit_l_freq, prep_cfg.ica.ica_fit_h_freq, verbose=False
        )
        ica = ICA(
            n_components=prep_cfg.ica.ica_n_components,
            method=prep_cfg.ica.ica_method,
            random_state=prep_cfg.ica.ica_random_state,
            max_iter="auto",
        )
        ica.fit(ica_fit_raw, reject_by_annotation=True, verbose=False)

        # Candidates are found once at the configured threshold: never chase a
        # target component count.
        eog_indices, _ = ica.find_bads_eog(
            ica_fit_raw,
            ch_name=[
                ch
                for ch in prep_cfg.ica.ica_eog_proxy_channels
                if ch in ica_fit_raw.ch_names
            ],
            threshold=prep_cfg.ica.ica_eog_threshold,
            measure="zscore",
            verbose=False,
        )
        ica.exclude = sorted(set(eog_indices))

        # The session ICA is applied to every recording of the unit separately,
        # and each recording is then cut into its own labelled windows.
        for recording, raw in zip(unit_recordings, unit_raws, strict=True):
            ica.apply(raw, verbose=False)
            # A no-op when the recording is already at the target rate.
            raw.resample(prep_cfg.filtering.resample_sfreq, verbose=False)

            sfreq = raw.info["sfreq"]
            # The configured window snapped to this recording's sample grid.
            window_duration = round(window_size * sfreq) / sfreq

            # One labelled block of the recording gives one series of windows;
            # blocks shorter than the window give none. Cut by hand and not with
            # `events_from_annotations(chunk_duration=...)`, which agrees with this
            # code sample for sample but supports no overlap at all -- and
            # `overlap` is part of the recipe.
            blocks = [
                ann
                for ann in raw.annotations
                if ann["description"] in conditions and ann["duration"] >= window_size
            ]
            block_events = [
                mne.make_fixed_length_events(
                    raw,
                    id=conditions[ann["description"]],
                    start=ann["onset"],
                    stop=ann["onset"] + ann["duration"],
                    duration=window_size,
                    overlap=prep_cfg.epoching.overlap,
                )
                for ann in blocks
            ]
            if not block_events:
                continue

            events = np.concatenate(block_events)
            windows_before = Counter(
                np.repeat(
                    [ann["description"] for ann in blocks],
                    [len(block) for block in block_events],
                )
            )
            window_onsets = (events[:, 0] - raw.first_samp) / sfreq
            recording_epochs = mne.Epochs(
                raw,
                events,
                # The internal label written by the name it has in
                # DATASET_MAPPING, so the artifact explains itself: a recording
                # carries only the classes its own blocks provide.
                event_id={class_names[code]: int(code) for code in np.unique(events[:, 2])},
                tmin=0.0,
                tmax=window_duration - 1.0 / sfreq,
                baseline=None,
                preload=True,
                metadata=pd.DataFrame(
                    {
                        "dataset": dataset_name,
                        "subject": recording.subject,
                        "session": recording.session,
                        "task": recording.task,
                        "run": recording.run,
                        # The grouping a dataset-agnostic split has to use, and
                        # qualified by dataset because transfer puts two of them
                        # in one table.
                        "recording_unit": unit_id,
                        "label": events[:, 2],
                        "condition": np.repeat(
                            [ann["description"] for ann in blocks],
                            [len(block) for block in block_events],
                        ),
                        "window_start_s": window_onsets,
                        "window_stop_s": window_onsets + window_duration,
                    }
                ),
                reject=(
                    dict(eeg=prep_cfg.epoching.reject_peak_to_peak_uv * 1e-6)
                    if prep_cfg.epoching.reject_peak_to_peak_uv is not None
                    else None
                ),
                reject_by_annotation=True,
                verbose=False,
            )
            # `reject_by_annotation` has already done its work above, and the
            # artifact itself carries no annotations.
            if len(recording_epochs):
                windows_after = Counter(recording_epochs.metadata["condition"])
                dropped_counts = {
                    condition: windows_before[condition] - windows_after[condition]
                    for condition in windows_before
                }
                if any(dropped_counts.values()):
                    dropped = ", ".join(
                        f"{condition}={dropped_counts[condition]}/{windows_before[condition]}"
                        for condition in sorted(windows_before)
                    )
                    print(f"{recording.basename}: epoch rejection by condition {dropped}")
                dataset_epochs.append(recording_epochs.set_annotations(None))

    # =============================================================================
    # Step 5: concatenate and publish the artifact atomically
    # =============================================================================
    # Staging lives in `PREPARED_CACHE_ROOT` itself, so publishing is a
    # same-filesystem rename: an artifact name appears already complete and a
    # half-built one is never visible. The staging name is dot-prefixed and
    # random, so it can never be mistaken for an artifact entry; `ls -a` finds
    # what an interrupted write leaves behind.

    if not dataset_epochs:
        raise ValueError(
            f"Selection and epoching produced no epochs for {dataset_name}; "
            "check BIDS session metadata, exclude_session_values and exclude_conditions."
        )
    epochs = mne.concatenate_epochs(dataset_epochs, verbose=False)

    PREPARED_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{artifact_dir.name}.", dir=PREPARED_CACHE_ROOT))
    try:
        epochs.save(staging / "epochs-epo.fif", overwrite=False, verbose=False)
        OmegaConf.save(
            {
                "dataset": OmegaConf.to_container(ds_cfg, resolve=True),
                "preparation": OmegaConf.to_container(prep_cfg, resolve=True),
            },
            staging / "config.yaml",
        )
        if artifact_dir.exists():
            raise FileExistsError(
                f"Cache entry appeared while building and is never rebuilt over: {artifact_dir}"
            )
        staging.rename(artifact_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Prepared {artifact_dir}")
    return artifact_dir
