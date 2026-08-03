"""Advanced Python utilities for the explainable ML evaluation platform."""

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def get_high_confidence_misclassifications(
    predictions: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Filter misclassified predictions above a confidence threshold.

    Args:
        predictions: Prediction records, each containing at least
            "actual", "predicted", and "probability" keys.
        threshold: Minimum probability (inclusive) a record must have
            to be included.

    Returns:
        A new list containing only misclassified records whose
        probability is greater than or equal to threshold. The
        original list and its records are left unmodified.
    """
    return [
        record
        for record in predictions
        if record["predicted"] != record["actual"]
        and record["probability"] >= threshold
    ]


def measure_execution_time(func: F) -> F:
    """Log how long the wrapped function takes to execute.

    Args:
        func: The function to time.

    Returns:
        The wrapped function, with its original metadata preserved.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.info("%s executed in %.4f seconds", func.__name__, duration)
        return result

    return wrapper  


def validate_dataframe(
    required_columns: Optional[List[str]] = None,
) -> Callable[[F], F]:
    """Build a decorator that validates a function's DataFrame input.

    The decorated function must receive the DataFrame either as its
    first positional argument or as a ``df`` keyword argument.

    Args:
        required_columns: Column names that must be present. If None,
            only type and emptiness are checked.

    Returns:
        A decorator enforcing the DataFrame contract on the wrapped
        function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            df = kwargs.get("df", args[0] if args else None)

            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    f"{func.__name__} expected a pandas DataFrame as "
                    f"input, got {type(df).__name__}."
                )

            if df.empty:
                raise ValueError(f"{func.__name__} received an empty DataFrame.")

            if required_columns:
                missing = set(required_columns) - set(df.columns)
                if missing:
                    raise ValueError(
                        f"{func.__name__} is missing required columns: "
                        f"{sorted(missing)}."
                    )

            return func(*args, **kwargs)

        return wrapper  

    return decorator
