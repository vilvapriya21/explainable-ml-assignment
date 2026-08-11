"""Tests for shared configuration and utility infrastructure."""

import logging

import numpy as np
import pytest

from src.config import FN_COST, FP_COST, RANDOM_SEED
from src.utils.io import load_json, save_json
from src.utils.logger import configure_logging


def test_configuration_defaults_are_expected() -> None:
    assert RANDOM_SEED == 42
    assert FN_COST == 5.0
    assert FP_COST == 1.0


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    handler_count = len(logging.getLogger().handlers)

    configure_logging()

    assert len(logging.getLogger().handlers) == handler_count


def test_load_json_rejects_missing_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_json(str(tmp_path / "missing.json"))


def test_save_and_load_json_round_trip_numpy_scalar(tmp_path) -> None:
    path = tmp_path / "nested" / "data.json"

    save_json({"score": np.float32(0.75)}, str(path))
    result = load_json(str(path))

    assert result == {"score": 0.75}
    assert isinstance(result["score"], float)
