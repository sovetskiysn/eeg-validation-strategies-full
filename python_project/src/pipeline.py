"""Pipeline components that need train-fold data and labels together."""

import numpy as np
from imblearn.over_sampling import RandomOverSampler


def resample_eeg_train_windows(
    X: np.ndarray, y: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample one train fold without changing its EEG window geometry."""
    sampler = RandomOverSampler(random_state=random_state)
    window_shape = X.shape[1:]
    X, y = sampler.fit_resample(X.reshape(len(X), -1), y)
    return X.reshape(-1, *window_shape), y
