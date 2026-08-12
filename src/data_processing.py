"""Leakage-safe data preparation helpers for benchmark workflows."""

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED


def stratified_split_with_missing_validation_copy(
    features: np.ndarray,
    target: np.ndarray,
    test_size: float = 0.25,
    random_state: int = RANDOM_SEED,
    missing_fraction: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data before injecting missing values into a training-only copy.

    Args:
        features: Two-dimensional feature matrix.
        target: One-dimensional target labels aligned with features.
        test_size: Proportion reserved for the stratified test split.
        random_state: Seed used for both splitting and missing-value selection.
        missing_fraction: Fraction of values replaced with NaN in the validation
            copy of the training features.

    Returns:
        Clean training features, clean test features, training labels, test
        labels, and a training-feature copy containing injected missing values.

    Raises:
        TypeError: If features or target is not a NumPy array.
        ValueError: If array dimensions, sample counts, test_size, or
            missing_fraction are invalid.
    """
    if not isinstance(features, np.ndarray):
        raise TypeError(
            "features must be a NumPy ndarray, "
            f"got {type(features).__name__}."
        )
    if not isinstance(target, np.ndarray):
        raise TypeError(
            f"target must be a NumPy ndarray, got {type(target).__name__}."
        )
    if features.ndim != 2:
        raise ValueError(
            f"features must be two-dimensional, got {features.ndim} dimensions."
        )
    if target.ndim != 1:
        raise ValueError(f"target must be one-dimensional, got {target.ndim} dimensions.")
    if len(features) != len(target):
        raise ValueError(
            "features and target must contain the same number of samples, "
            f"got {len(features)} and {len(target)}."
        )
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")
    if not 0.0 < missing_fraction <= 1.0:
        raise ValueError(
            "missing_fraction must be greater than 0 and no greater than 1, "
            f"got {missing_fraction}."
        )
    if not np.issubdtype(features.dtype, np.number):
        raise ValueError("features must have a numeric dtype to support NaN injection.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )
    training_validation_copy = X_train.astype(float, copy=True)
    missing_count = max(1, int(training_validation_copy.size * missing_fraction))
    generator = np.random.default_rng(random_state)
    missing_positions = generator.choice(
        training_validation_copy.size,
        size=missing_count,
        replace=False,
    )
    rows, columns = np.unravel_index(missing_positions, training_validation_copy.shape)
    training_validation_copy[rows, columns] = np.nan
    return X_train, X_test, y_train, y_test, training_validation_copy
