"""Tests for leakage-safe data preparation helpers."""

import numpy as np
import pytest

from src.data_processing import stratified_split_with_missing_validation_copy


def test_stratified_split_creates_missing_training_copy_without_mutation() -> None:
    """Keep clean splits intact while injecting missing values into one copy."""
    features = np.arange(240, dtype=float).reshape(40, 6)
    target = np.array([0] * 20 + [1] * 20)

    X_train, X_test, y_train, y_test, validation_copy = (
        stratified_split_with_missing_validation_copy(features, target)
    )

    assert X_train.shape == (30, 6)
    assert X_test.shape == (10, 6)
    assert np.bincount(y_train).tolist() == [15, 15]
    assert np.bincount(y_test).tolist() == [5, 5]
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert np.isnan(validation_copy).sum() == 1
    assert np.array_equal(features, np.arange(240, dtype=float).reshape(40, 6))


def test_stratified_split_is_deterministic_for_the_same_seed() -> None:
    """Use the configured seed for repeatable splits and missing positions."""
    features = np.arange(240, dtype=float).reshape(40, 6)
    target = np.array([0] * 20 + [1] * 20)

    first = stratified_split_with_missing_validation_copy(features, target, random_state=7)
    second = stratified_split_with_missing_validation_copy(features, target, random_state=7)

    for first_part, second_part in zip(first[:4], second[:4]):
        assert np.array_equal(first_part, second_part)
    assert np.array_equal(first[4], second[4], equal_nan=True)


def test_stratified_split_rejects_misaligned_samples() -> None:
    """Reject targets that cannot be aligned with the feature rows."""
    with pytest.raises(ValueError, match="same number of samples"):
        stratified_split_with_missing_validation_copy(
            np.zeros((10, 2)),
            np.zeros(9),
        )
