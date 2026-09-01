"""Build the data and splits for each validation protocol."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from preparation import DATASET_MAPPING, prepare_epochs


def build_baseline(
    preparation_config: DictConfig, seed: int
) -> tuple[mne.Epochs, np.ndarray, StratifiedGroupKFold]:
    """Return one dataset with physical-session-disjoint stratified folds."""
    # =============================================================================
    # Step 1: prepare the dataset
    # =============================================================================
    if "source" in preparation_config or "targets" in preparation_config:
        raise ValueError("baseline validation accepts one preparation config, not transfer sides.")
    epochs = prepare_epochs(preparation_config)

    # =============================================================================
    # Step 2: define recording-disjoint folds
    # =============================================================================
    metadata = epochs.metadata
    metadata["subject_id"] = metadata["dataset"] + "_" + metadata["subject"]
    # ICA is fitted once per recording unit, so this split must keep every
    # unit entirely on one side of a fold.
    groups = metadata["recording_unit"].to_numpy()
    return epochs, groups, StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)


def build_cross_subject(
    preparation_config: DictConfig, seed: int
) -> tuple[mne.Epochs, np.ndarray, LeaveOneGroupOut]:
    """Return one dataset with leave-one-subject-out folds."""
    # =============================================================================
    # Step 1: prepare the dataset
    # =============================================================================
    if "source" in preparation_config or "targets" in preparation_config:
        raise ValueError("cross-subject validation accepts one preparation config, not transfer sides.")
    epochs = prepare_epochs(preparation_config)

    # =============================================================================
    # Step 2: define leave-one-subject-out folds
    # =============================================================================
    metadata = epochs.metadata
    metadata["subject_id"] = metadata["dataset"] + "_" + metadata["subject"]
    return epochs, metadata["subject_id"].to_numpy(), LeaveOneGroupOut()


def build_cross_session(
    preparation_config: DictConfig, seed: int
) -> tuple[mne.Epochs, None, list[tuple[np.ndarray, np.ndarray]]]:
    """Return leave-one-source-session-out folds within each subject."""
    # =============================================================================
    # Step 1: prepare and validate the dataset
    # =============================================================================
    if "source" in preparation_config or "targets" in preparation_config:
        raise ValueError("cross-session validation accepts one preparation config, not transfer sides.")
    epochs = prepare_epochs(preparation_config)
    metadata = epochs.metadata
    if metadata["session"].isna().any():
        raise ValueError(
            f"cross-session validation needs source sessions, and "
            f"{sorted(metadata['dataset'].unique())} has none."
        )
    metadata["subject_id"] = metadata["dataset"] + "_" + metadata["subject"]

    # =============================================================================
    # Step 2: hold out each session within each subject
    # =============================================================================
    subjects = metadata["subject_id"].to_numpy()
    sessions = (metadata["subject_id"] + "_ses-" + metadata["session"]).to_numpy()
    cv = []
    for subject in np.unique(subjects):
        rows = np.flatnonzero(subjects == subject)
        subject_sessions = sessions[rows]
        for held_out in np.unique(subject_sessions):
            cv.append((rows[subject_sessions != held_out], rows[subject_sessions == held_out]))
    return epochs, None, cv


def build_cross_dataset(
    preparation_config: DictConfig, seed: int
) -> tuple[mne.Epochs, None, list[tuple[np.ndarray, np.ndarray]]]:
    """Train on every source window and test on the windows of every target."""
    # =============================================================================
    # Step 1: validate the transfer direction
    # =============================================================================
    if not preparation_config.targets:
        raise ValueError("cross-dataset validation needs at least one target preparation.")
    source_dataset = Path(str(preparation_config.source.dataset_dir)).name
    if source_dataset not in DATASET_MAPPING:
        raise ValueError(f"No condition mapping for {source_dataset!r}: {sorted(DATASET_MAPPING)}.")
    source_classes = list(DATASET_MAPPING[source_dataset])
    for index, target in enumerate(preparation_config.targets.values()):
        target_dataset = Path(str(target.dataset_dir)).name
        if source_dataset == target_dataset:
            raise ValueError(
                f"cross-dataset validation requires different source and target datasets; "
                f"target {index} is {target_dataset}, the source dataset."
            )
        if target_dataset not in DATASET_MAPPING:
            raise ValueError(f"No condition mapping for {target_dataset!r}: {sorted(DATASET_MAPPING)}.")
        # Class order defines the numeric labels and must match across datasets.
        target_classes = list(DATASET_MAPPING[target_dataset])
        if source_classes != target_classes:
            raise ValueError(
                "cross-dataset validation needs the same classes in the same order on both "
                f"sides; {source_dataset} has {source_classes}, target {index} "
                f"{target_dataset} has {target_classes}."
            )

    # =============================================================================
    # Step 2: prepare and combine source and targets
    # =============================================================================
    source_epochs = prepare_epochs(preparation_config.source)
    target_epochs = []
    for index, target in enumerate(preparation_config.targets.values()):
        epochs = prepare_epochs(target)
        epochs.metadata["target_index"] = index
        target_epochs.append(epochs)
    source_epochs.metadata["target_index"] = pd.NA

    epochs = mne.concatenate_epochs([source_epochs, *target_epochs], verbose=False)
    metadata = epochs.metadata
    metadata["target_index"] = metadata["target_index"].astype("Int64")
    metadata["subject_id"] = metadata["dataset"] + "_" + metadata["subject"]

    # =============================================================================
    # Step 3: train once on the source and test on every target
    # =============================================================================
    return epochs, None, [
        (np.arange(len(source_epochs)), np.arange(len(source_epochs), len(epochs)))
    ]


def build_cross_task(
    preparation_config: DictConfig, seed: int
) -> tuple[mne.Epochs, None, list[tuple[np.ndarray, np.ndarray]]]:
    """Transfer between tasks while holding out each participant from source training."""
    # =============================================================================
    # Step 1: validate the task compositions
    # =============================================================================
    if not preparation_config.targets:
        raise ValueError("cross-task validation needs at least one target task composition.")
    source_dataset = Path(str(preparation_config.source.dataset_dir)).name
    source_task = preparation_config.source.mne_bids_pipeline_config.task
    source_tasks = {source_task} if isinstance(source_task, str) else set(source_task)
    for index, target in enumerate(preparation_config.targets.values()):
        target_dataset = Path(str(target.dataset_dir)).name
        if source_dataset != target_dataset:
            raise ValueError(
                "cross-task validation requires the same dataset on the source and every "
                f"target; target {index} is {target_dataset}, the source is {source_dataset}."
            )
        target_task = target.mne_bids_pipeline_config.task
        target_tasks = {target_task} if isinstance(target_task, str) else set(target_task)
        if source_tasks == target_tasks:
            raise ValueError(
                "cross-task validation requires different tasks on the source and every "
                f"target; target {index} repeats the source composition."
            )

    # =============================================================================
    # Step 2: prepare source and targets
    # =============================================================================
    source_epochs = prepare_epochs(preparation_config.source)
    source_subjects = source_epochs.metadata["subject"].to_numpy()
    target_epochs = []
    for index, target in enumerate(preparation_config.targets.values()):
        epochs = prepare_epochs(target)
        if set(epochs.metadata["subject"].to_numpy()) != set(source_subjects):
            raise ValueError(
                "cross-task validation requires source and target datasets with the same "
                f"participants; target {index} has a different participant set."
            )
        epochs.metadata["target_index"] = index
        target_epochs.append(epochs)
    source_epochs.metadata["target_index"] = pd.NA

    epochs = mne.concatenate_epochs([source_epochs, *target_epochs], verbose=False)
    metadata = epochs.metadata
    metadata["target_index"] = metadata["target_index"].astype("Int64")
    metadata["subject_id"] = metadata["dataset"] + "_" + metadata["subject"]

    # =============================================================================
    # Step 3: hold out each source participant from every target task
    # =============================================================================
    offsets = np.cumsum([len(source_epochs), *(len(one) for one in target_epochs)])
    cv = []
    for subject in np.unique(source_subjects):
        test_idx = np.concatenate(
            [
                offset + np.flatnonzero(one.metadata["subject"].to_numpy() == subject)
                for offset, one in zip(offsets[:-1], target_epochs, strict=True)
            ]
        )
        cv.append((np.flatnonzero(source_subjects != subject), test_idx))
    return epochs, None, cv
