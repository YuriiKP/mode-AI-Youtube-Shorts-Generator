"""Font resolution for the built-in post-processing engine.

Burn-in subtitles need a concrete TrueType/OpenType file, so this module turns
the small set of font-related settings into an absolute, existing path that
MoviePy/Pillow can load.

Lookup order (first match wins):

1. ``FONT_PATH`` -- an explicit path to a ``.ttf``/``.ttc``/``.otf`` file;
2. ``FONT_NAME`` -- a file name looked up in ``FONTS_DIR`` and a couple of
   sensible fallback folders (``<tool>/fonts`` and ``./fonts``);
3. a small list of well-known system fonts (DejaVu Sans, Arial, ...).

If nothing is found a :class:`FontNotFoundError` is raised with an actionable
message instead of letting Pillow fail deep inside the render stage.
"""

from __future__ import annotations

import os

from ..config import Settings
from .log import log

# Extensions we accept as a font file.
_FONT_EXTENSIONS = (".ttf", ".ttc", ".otf", ".otc")

# Well-known system fonts, tried in order. The list is intentionally small and
# covers the common platforms without bundling megabytes of font data.
_SYSTEM_FONT_CANDIDATES = (
    # Windows
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    # Linux (Debian/Ubuntu, Fedora, Alpine)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


class FontNotFoundError(FileNotFoundError):
    """Raised when no usable font file could be located."""


def _is_font_file(path: str) -> bool:
    return bool(path) and os.path.isfile(path)


def _normalize(path: str) -> str:
    """Return an absolute font path usable by Pillow on every platform.

    On Windows the path is returned with forward slashes; this avoids the
    backslash-as-escape problems that show up when the path is passed through
    configuration strings or log messages.
    """
    absolute = os.path.abspath(os.path.expanduser(path))
    if os.name == "nt":
        return absolute.replace("\\", "/")
    return absolute


def _font_search_dirs(settings: Settings) -> list[str]:
    """Directories searched when resolving ``FONT_NAME`` (in order)."""
    tool_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        settings.resolve(settings.fonts_dir) if settings.fonts_dir else "",
        os.path.join(tool_dir, "fonts"),
        os.path.join(settings.base_dir, "fonts"),
    ]
    # De-duplicate while preserving order.
    seen: list[str] = []
    for directory in candidates:
        if directory and directory not in seen:
            seen.append(directory)
    return seen


def _has_font_extension(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _FONT_EXTENSIONS


def resolve_font_path(settings: Settings) -> str:
    """Resolve the subtitle font to an absolute, existing file path.

    Raises
    ------
    FontNotFoundError
        When no candidate could be resolved to an existing font file.
    """
    # 1. Explicit FONT_PATH ---------------------------------------------------
    if settings.font_path:
        explicit = settings.resolve(settings.font_path)
        if _is_font_file(explicit):
            log.debug("using subtitle font from FONT_PATH: %s", explicit)
            return _normalize(explicit)
        raise FontNotFoundError(
            f"FONT_PATH does not point to an existing file: {settings.font_path!r}"
        )

    # 2. FONT_NAME inside the configured / fallback font directories ----------
    font_name = settings.font_name
    if font_name:
        # A relative name that points directly at an existing file is honoured
        # even when it is not inside one of the search directories.
        direct = settings.resolve(font_name)
        if _is_font_file(direct) and _has_font_extension(direct):
            log.debug("using subtitle font from FONT_NAME path: %s", direct)
            return _normalize(direct)

        for directory in _font_search_dirs(settings):
            candidate = os.path.join(directory, font_name)
            if _is_font_file(candidate):
                log.debug("resolved subtitle font %r in %s", font_name, directory)
                return _normalize(candidate)

    # 3. Well-known system fonts ---------------------------------------------
    for candidate in _SYSTEM_FONT_CANDIDATES:
        if _is_font_file(candidate):
            log.warning(
                "subtitle font %r was not found; falling back to system font: %s",
                font_name or "(unset)",
                candidate,
            )
            return _normalize(candidate)

    searched = ", ".join(_font_search_dirs(settings)) or "(none)"
    raise FontNotFoundError(
        "could not find a subtitle font. Set FONT_PATH to a .ttf/.ttc/.otf file "
        f"or place your font in one of: {searched}. "
        f"Requested FONT_NAME={font_name!r}."
    )


__all__ = ["FontNotFoundError", "resolve_font_path"]
