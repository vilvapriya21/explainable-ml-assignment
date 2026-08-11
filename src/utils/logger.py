"""Application-wide logging configuration."""

import logging
from typing import Optional

from src.config import LOG_LEVEL

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging format and level for the whole application.

    Args:
        level: Optional logging level name. Uses configured LOG_LEVEL when None.
    """
    configured_level = level or LOG_LEVEL
    numeric_level = logging.getLevelName(configured_level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {configured_level!r}.")

    root_logger = logging.getLogger()
    formatter = logging.Formatter(_LOG_FORMAT)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
    root_logger.setLevel(numeric_level)
