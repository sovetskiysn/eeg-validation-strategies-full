"""Build the data and splits for each validation protocol."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from preparation import DATASET_MAPPING, get_dataset_dir


def build_baseline(
    dataset: DictConfig, preparation: DictConfig, seed: int
) -> tuple[mne.Epochs, np.ndarray, StratifiedGroupKFold]:
    """Return one dataset with physical-session-disjoint stratified folds."""
    dataset_dir = get_dataset_dir(dataset, preparation)
    epochs = mne.read_epochs(dataset_dir / "epochs-epo.fif", preload=True, verbose=False)
    # The same `sub-01` is a different person in each dataset, and transfer puts
    # both in one table, so identity is prefixed by the dataset it came from.
    # `recording_unit` arrives already qualified that way from Stage 1.
    m = epochs.metadata
    m["subject_id"] = m["dataset"] + "_" + m["subject"]
    # ICA is fitted once per recording unit, so this split must keep every
    # recording of that unit on one side. Grouping by task recording here would
    # share a fitted ICA solution between train and test for SAM40.
    groups = m["recording_unit"].to_numpy()
    return epochs, groups, StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)


def build_cross_subject(
    dataset: DictConfig, preparation: DictConfig, seed: int
) -> tuple[mne.Epochs, np.ndarray, LeaveOneGroupOut]:
    """Return one dataset with leave-one-subject-out folds."""
    dataset_dir = get_dataset_dir(dataset, preparation)
    epochs = mne.read_epochs(dataset_dir / "epochs-epo.fif", preload=True, verbose=False)
    m = epochs.metadata
    m["subject_id"] = m["dataset"] + "_" + m["subject"]
    return epochs, m["subject_id"].to_numpy(), LeaveOneGroupOut()


def build_cross_session(
    dataset: DictConfig, preparation: DictConfig, seed: int
) -> tuple[mne.Epochs, None, list[tuple[np.ndarray, np.ndarray]]]:
    """Return leave-one-source-session-out folds within each subject."""
    dataset_dir = get_dataset_dir(dataset, preparation)
    epochs = mne.read_epochs(dataset_dir / "epochs-epo.fif", preload=True, verbose=False)
    m = epochs.metadata
    # This protocol claims transferability across independent physical sessions,
    # so it runs only on a dataset whose source actually has them. SAM40 repeats
    # its protocol within one setup; that is build_cross_trial, and calling it
    # cross-session would be a claim the data does not support.
    if m["session"].isna().any():
        raise ValueError(
            f"cross-session validation needs source sessions, and "
            f"{sorted(m['dataset'].unique())} has none. Use cross_trial."
        )
    m["subject_id"] = m["dataset"] + "_" + m["subject"]
    subjects = m["subject_id"].to_numpy()
    # Named after what this protocol actually splits on. The generic
    # `recording_unit` would give the same strings, but naming it here is what
    # keeps this apart from cross-trial, which is a different claim.
    units = (m["subject_id"] + "_ses-" + m["session"]).to_numpy()
    cv = []
    for subject in np.unique(subjects):
        rows = np.flatnonzero(subjects == subject)
        subject_units = units[rows]
        for held_out in np.unique(subject_units):
            cv.append((rows[subject_units != held_out], rows[subject_units == held_out]))
    return epochs, None, cv


def build_cross_trial(
    dataset: DictConfig, preparation: DictConfig, seed: int
) -> tuple[mne.Epochs, None, list[tuple[np.ndarray, np.ndarray]]]:
    """Return leave-one-source-trial-out folds within each subject.

    Every window and every selected task recording of one trial stays on one side
    of the split, which is what keeps overlapping windows and repeated
    acquisitions out of both train and test. It is not a test of transfer between
    independent physical sessions and must not be reported as one.
    """
    dataset_dir = get_dataset_dir(dataset, preparation)
    epochs = mne.read_epochs(dataset_dir / "epochs-epo.fif", preload=True, verbose=False)
    m = epochs.metadata
    if m["run"].isna().any():
        raise ValueError(
            f"cross-trial validation needs source trials represented as runs, and "
            f"{sorted(m['dataset'].unique())} has none. Use cross_session."
        )
    m["subject_id"] = m["dataset"] + "_" + m["subject"]
    subjects = m["subject_id"].to_numpy()
    units = (m["subject_id"] + "_run-" + m["run"]).to_numpy()
    cv = []
    for subject in np.unique(subjects):
        rows = np.flatnonzero(subjects == subject)
        subject_units = units[rows]
        for held_out in np.unique(subject_units):
            cv.append((rows[subject_units != held_out], rows[subject_units == held_out]))
    return epochs, None, cv


def build_cross_dataset(
    dataset: DictConfig, preparation: DictConfig, seed: int
) -> tuple[mne.Epochs, None, list[tuple[np.ndarray, np.ndarray]]]:
    """Train on every source window and test on the windows of every target."""
    source_dir = get_dataset_dir(dataset.source, preparation)
    source_epochs = mne.read_epochs(source_dir / "epochs-epo.fif", preload=True, verbose=False)
    # A run holds one source and several targets so the decoder is fitted once
    # per source fold instead of once per direction. Every target is still a
    # complete direction of its own, so each one faces the full check below; a
    # loop that checked only the first target would silently license the rest.
    source_classes = list(DATASET_MAPPING[dataset.source.name])
    target_epochs = []
    for index, target in enumerate(dataset.targets):
        if dataset.source.name == target.name:
            raise ValueError(
                f"cross-dataset validation requires different source and target datasets; "
                f"target {index} is {target.name}, the source dataset."
            )
        # Training on one release and testing on another only means anything if
        # class 1 is the same thing on both sides. The class label is the
        # position of its name in DATASET_MAPPING, so comparing the names in
        # order is what checks it -- and it has to be checked, because a swapped
        # order does not fail. It returns roughly `1 - accuracy`, which reads as
        # a plausible transfer result.
        target_classes = list(DATASET_MAPPING[target.name])
        if source_classes != target_classes:
            raise ValueError(
                "cross-dataset validation needs the same classes in the same order on both "
                f"sides; {dataset.source.name} has {source_classes}, target {index} "
                f"{target.name} has {target_classes}."
            )
        epochs = mne.read_epochs(
            get_dataset_dir(target, preparation) / "epochs-epo.fif", preload=True, verbose=False
        )
        # Which target a window belongs to survives the concatenation only in the
        # metadata, because the protocol contract returns one Epochs object. The
        # runner reads it to split the run back into one result per direction and
        # drops it there; it is a carrier, not a recorded fact.
        epochs.metadata["target_index"] = index
        target_epochs.append(epochs)
    source_epochs.metadata["target_index"] = pd.NA

    epochs = mne.concatenate_epochs([source_epochs, *target_epochs], verbose=False)
    m = epochs.metadata
    m["target_index"] = m["target_index"].astype("Int64")
    m["subject_id"] = m["dataset"] + "_" + m["subject"]
    source_size = len(source_epochs)
    # One fold, and its test set is every target at once: that is what makes this
    # a single fit on the source rather than one fit per direction.
    return epochs, None, [(np.arange(source_size), np.arange(source_size, len(epochs)))]


def build_cross_task(
    dataset: DictConfig, preparation: DictConfig, seed: int
) -> tuple[mne.Epochs, None, list[tuple[np.ndarray, np.ndarray]]]:
    """Transfer between tasks while holding out each participant from source training."""
    source_dir = get_dataset_dir(dataset.source, preparation)
    source_epochs = mne.read_epochs(source_dir / "epochs-epo.fif", preload=True, verbose=False)
    # This participant check spans two artifacts, so it compares the bare BIDS
    # subject: dataset-prefixed ids could never match across a source and a target.
    source_subjects = source_epochs.metadata["subject"].to_numpy()
    target_epochs = []
    for index, target in enumerate(dataset.targets):
        # Both sides name the same BIDS dataset and differ only in the conditions
        # the recipe excluded, so that is what must differ here. Comparing names
        # would reject every legitimate SAM40 task pair. Sets, because the order
        # two exclusions are written in is not a difference between them.
        if set(dataset.source.exclude_conditions) == set(target.exclude_conditions):
            raise ValueError(
                "cross-task validation requires different exclude_conditions on the source "
                f"and every target; target {index} repeats the source composition."
            )
        epochs = mne.read_epochs(
            get_dataset_dir(target, preparation) / "epochs-epo.fif", preload=True, verbose=False
        )
        if set(epochs.metadata["subject"].to_numpy()) != set(source_subjects):
            raise ValueError(
                "cross-task validation requires source and target datasets with the same "
                f"participants; target {index} has a different participant set."
            )
        epochs.metadata["target_index"] = index
        target_epochs.append(epochs)
    source_epochs.metadata["target_index"] = pd.NA

    epochs = mne.concatenate_epochs([source_epochs, *target_epochs], verbose=False)
    m = epochs.metadata
    m["target_index"] = m["target_index"].astype("Int64")
    m["subject_id"] = m["dataset"] + "_" + m["subject"]
    # Each target starts where the previous one ended, and the offsets are read
    # off the actual lengths: two compositions of the same dataset hold different
    # numbers of windows, so a stride assumed to be constant would silently test
    # the wrong participant.
    offsets = np.cumsum([len(source_epochs), *(len(one) for one in target_epochs)])
    cv = []
    for subject in np.unique(source_subjects):
        # One fold per held-out source participant, and its test set is that same
        # unseen participant in every target at once. Neither task nor subject
        # identity can leak: the participant is absent from the source training.
        test_idx = np.concatenate(
            [
                offset + np.flatnonzero(one.metadata["subject"].to_numpy() == subject)
                for offset, one in zip(offsets[:-1], target_epochs, strict=True)
            ]
        )
        cv.append((np.flatnonzero(source_subjects != subject), test_idx))
    return epochs, None, cv
