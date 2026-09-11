"""Logging helpers for the built-in post-processing engine.

A single module-level logger (:data:`log`) is used across the whole package so
that :func:`setup_logging` can configure handlers once, from the entry point,
and every module benefits.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "postprocess"

# Module-level logger shared by every submodule.
log = logging.getLogger(LOGGER_NAME)

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_VALID_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def setup_logging(level: str = "INFO", quiet: bool = False) -> logging.Logger:
    """Configure and return the package logger.

    Parameters
    ----------
    level:
        One of ``CRITICAL``/``ERROR``/``WARNING``/``INFO``/``DEBUG`` (case
        insensitive). Unknown values fall back to ``INFO``.
    quiet:
        When ``True`` only warnings and errors are emitted, regardless of
        ``level``.
    """
    normalized = str(level or "INFO").strip().upper()
    if normalized not in _VALID_LEVELS:
        normalized = "INFO"
    if quiet:
        normalized = "WARNING"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers when ``main`` is invoked more than once (e.g. in
    # tests or interactive sessions).
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(getattr(logging, normalized))
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)

    return logger
