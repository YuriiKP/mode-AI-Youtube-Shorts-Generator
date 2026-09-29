"""Console logging for the publisher package.

Mirrors the style used elsewhere in the project (a single shared logger that is
configured once from the entry point) but keeps its own logger name so the
publishing output can be filtered independently of the clipping pipeline.
"""

from __future__ import annotations

import logging
import sys

# Module-level logger shared by every publisher submodule.
log = logging.getLogger("publisher")

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_VALID_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def setup_logging(level: str = "INFO", quiet: bool = False) -> logging.Logger:
    """Configure and return the publisher logger.

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

    logger = logging.getLogger("publisher")
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers when the entry point is invoked more than once
    # (e.g. in tests or an interactive session).
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(getattr(logging, normalized))
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)

    return logger
