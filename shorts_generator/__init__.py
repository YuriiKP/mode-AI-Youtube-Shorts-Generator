"""AI YouTube Shorts Generator — public API.

Everything is configured through a single :class:`Settings` object loaded from
one ``.env`` file (:func:`load_settings`). The CLI (``python main.py``) is a thin
wrapper around the functions below.
"""

from .config import ConfigError, Settings, load_settings
from .pipeline import generate_shorts, generate_subtitles, resolve_input_videos

__all__ = [
    "Settings",
    "ConfigError",
    "load_settings",
    "generate_shorts",
    "generate_subtitles",
    "resolve_input_videos",
]
