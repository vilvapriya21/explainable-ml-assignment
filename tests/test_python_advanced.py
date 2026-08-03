"""Unit tests for src/python_advanced.py."""

import inspect

import numpy as np
import pandas as pd
import pytest

from src.python_advanced import (
    batch_generator,
    get_high_confidence_misclassifications,
    measure_execution_time,
    validate_dataframe,
)


def test_returns_only_misclassified_above_threshold() -> None:
    predictions = [
        {"id": 1, "actual": 1, "predicted": 0, "probability": 0.82},
        {"id": 2, "actual": 0, "predicted": 0, "probability": 0.91},
        {"id": 3, "actual": 1, "predicted": 0, "probability": 0.40},
    ]

    result = get_high_confidence_misclassifications(predictions, threshold=0.5)

    assert result == [predictions[0]]


def test_does_not_mutate_original_list() -> None:
    predictions = [
        {"id": 1, "actual": 1, "predicted": 0, "probability": 0.82},
    ]
    original_length = len(predictions)

    get_high_confidence_misclassifications(predictions, threshold=0.5)

    assert len(predictions) == original_length
    assert predictions[0]["probability"] == 0.82


def test_threshold_is_inclusive() -> None:
    predictions = [{"id": 1, "actual": 1, "predicted": 0, "probability": 0.5}]

    result = get_high_confidence_misclassifications(predictions, threshold=0.5)

    assert result == predictions


def test_empty_predictions_returns_empty_list() -> None:
    assert get_high_confidence_misclassifications([], threshold=0.5) == []


def test_measure_execution_time_returns_original_result() -> None:
    @measure_execution_time
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


def test_measure_execution_time_logs_duration(caplog) -> None:
    @measure_execution_time
    def noop() -> None:
        return None

    with caplog.at_level("INFO"):
        noop()

    assert "noop executed in" in caplog.text


def test_measure_execution_time_preserves_metadata() -> None:
    @measure_execution_time
    def documented() -> None:
        """Docstring to preserve."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Docstring to preserve."


def test_validate_dataframe_rejects_non_dataframe() -> None:
    @validate_dataframe()
    def process(df: pd.DataFrame) -> pd.DataFrame:
        return df

    try:
        process([1, 2, 3])
        assert False, "Expected TypeError"
    except TypeError as exc:
        assert "process" in str(exc)


def test_validate_dataframe_rejects_empty_dataframe() -> None:
    @validate_dataframe()
    def process(df: pd.DataFrame) -> pd.DataFrame:
        return df

    try:
        process(pd.DataFrame())
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "empty" in str(exc)


def test_validate_dataframe_rejects_missing_required_columns() -> None:
    @validate_dataframe(required_columns=["age", "income"])
    def process(df: pd.DataFrame) -> pd.DataFrame:
        return df

    try:
        process(pd.DataFrame({"age": [30]}))
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "income" in str(exc)


def test_validate_dataframe_accepts_valid_input() -> None:
    @validate_dataframe(required_columns=["age"])
    def process(df: pd.DataFrame) -> pd.DataFrame:
        return df

    result = process(pd.DataFrame({"age": [30]}))

    assert result.equals(pd.DataFrame({"age": [30]}))


def test_validate_dataframe_preserves_metadata() -> None:
    @validate_dataframe()
    def documented(df: pd.DataFrame) -> pd.DataFrame:
        """Docstring to preserve."""
        return df

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Docstring to preserve."


def test_batch_generator_yields_evenly_divided_list_batches() -> None:
    result = list(batch_generator([1, 2, 3, 4], batch_size=2))

    assert result == [[1, 2], [3, 4]]


def test_batch_generator_yields_final_incomplete_list_batch() -> None:
    result = list(batch_generator([1, 2, 3, 4, 5], batch_size=2))

    assert result == [[1, 2], [3, 4], [5]]


def test_batch_generator_yields_numpy_array_slices() -> None:
    result = list(batch_generator(np.array([1, 2, 3]), batch_size=2))

    assert all(isinstance(batch, np.ndarray) for batch in result)
    assert np.array_equal(result[0], np.array([1, 2]))
    assert np.array_equal(result[1], np.array([3]))


def test_batch_generator_yields_dataframe_slices() -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3]})
    result = list(batch_generator(dataframe, batch_size=2))

    assert all(isinstance(batch, pd.DataFrame) for batch in result)
    assert result[0].equals(dataframe.iloc[:2])
    assert result[1].equals(dataframe.iloc[2:])


def test_batch_generator_rejects_zero_batch_size() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        list(batch_generator([1, 2], batch_size=0))


def test_batch_generator_rejects_negative_batch_size() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        list(batch_generator([1, 2], batch_size=-1))


def test_batch_generator_rejects_non_integer_batch_size() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        list(batch_generator([1, 2], batch_size=2.5))


def test_batch_generator_rejects_unsupported_data_type() -> None:
    with pytest.raises(TypeError, match="list, NumPy ndarray, or pandas DataFrame"):
        list(batch_generator({"value": 1}, batch_size=1))


def test_batch_generator_returns_a_generator() -> None:
    result = batch_generator([1, 2], batch_size=1)

    assert inspect.isgenerator(result)
