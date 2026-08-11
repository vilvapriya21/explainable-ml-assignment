"""Reusable JSON artifact input and output helpers."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str) -> dict:
    """Load and parse a JSON file.

    Args:
        path: Source JSON file path.

    Returns:
        Parsed JSON object.

    Raises:
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If file contents are not valid JSON.
        ValueError: If the parsed JSON value is not an object.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path!r}.")
    try:
        contents = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"Failed to parse JSON file {path!r}: {exc.msg}",
            exc.doc,
            exc.pos,
        ) from exc
    if not isinstance(contents, dict):
        raise ValueError(f"JSON file must contain an object: {path!r}.")
    return contents


def _to_native(value: Any) -> Any:
    """Convert NumPy values recursively into JSON-native Python values.

    Args:
        value: Value potentially containing NumPy scalar or array values.

    Returns:
        A JSON-serializable representation of value.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _to_native(value.tolist())
    if isinstance(value, dict):
        return {str(key): _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    return value


def save_json(data: dict, path: str, indent: int = 2) -> None:
    """Save a dictionary as JSON, creating parent directories as needed.

    Args:
        data: Dictionary to serialize.
        path: Destination JSON file path.
        indent: Number of spaces used for JSON indentation.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_to_native(data), indent=indent),
        encoding="utf-8",
    )
