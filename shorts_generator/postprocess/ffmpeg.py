"""FFmpeg binary resolution for the built-in post-processing engine.

MoviePy renders video through FFmpeg. Different environments expose it in
different ways, so this module centralises the lookup order:

1. ``FFMPEG_PATH`` from the ``.env`` file / CLI;
2. the ``IMAGEIO_FFMPEG_EXE`` environment variable already set by the user;
3. ``ffmpeg`` found on the system ``PATH``;
4. the binary shipped with the ``imageio-ffmpeg`` dependency;
5. the bare string ``"ffmpeg"`` as a last resort.

:func:`configure_ffmpeg` also exports the resolved path through
``IMAGEIO_FFMPEG_EXE`` so that MoviePy and imageio pick up the exact same
binary. It must be called *before* MoviePy starts reading or writing files.
"""

from __future__ import annotations

import os
import shutil

from .log import log

_FALLBACK_BINARY = "ffmpeg"


def _existing_file(path: str) -> str:
    """Return an absolute path if ``path`` points to an existing file."""
    if not path:
        return ""
    expanded = os.path.expanduser(str(path))
    if os.path.isfile(expanded):
        return os.path.abspath(expanded)
    return ""


def resolve_ffmpeg_binary(configured_path: str = "") -> str:
    """Resolve the FFmpeg executable using the documented priority order."""
    explicit = _existing_file(configured_path)
    if explicit:
        log.debug("using FFmpeg from FFMPEG_PATH: %s", explicit)
        return explicit

    if configured_path:
        log.warning(
            "FFMPEG_PATH is set but does not point to a file, ignoring it: %s",
            configured_path,
        )

    env_binary = _existing_file(os.environ.get("IMAGEIO_FFMPEG_EXE", ""))
    if env_binary:
        log.debug("using FFmpeg from IMAGEIO_FFMPEG_EXE: %s", env_binary)
        return env_binary

    system_binary = shutil.which("ffmpeg")
    if system_binary:
        log.debug("using FFmpeg from PATH: %s", system_binary)
        return system_binary

    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled:
            log.debug("using bundled FFmpeg from imageio-ffmpeg: %s", bundled)
            return bundled
    except Exception as exc:  # pragma: no cover - depends on environment
        log.debug("could not resolve bundled FFmpeg: %s", exc)

    log.warning(
        "no FFmpeg executable could be located; falling back to %r. "
        "Install FFmpeg or set FFMPEG_PATH in your .env file.",
        _FALLBACK_BINARY,
    )
    return _FALLBACK_BINARY


def configure_ffmpeg(configured_path: str = "") -> str:
    """Resolve FFmpeg and export it for MoviePy/imageio.

    Returns the resolved binary path (or ``"ffmpeg"``).
    """
    binary = resolve_ffmpeg_binary(configured_path)
    # MoviePy (via imageio-ffmpeg) reads this variable; pinning it guarantees the
    # whole process uses one consistent binary.
    os.environ["IMAGEIO_FFMPEG_EXE"] = binary
    return binary
