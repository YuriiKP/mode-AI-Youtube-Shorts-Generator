"""Font resolution for the built-in post-processing engine.

Burn-in subtitles need a concrete TrueType/OpenType file, so this module turns
the single ``FONT`` setting into an absolute, existing path that MoviePy/Pillow
can load.

``FONT`` mirrors ``MUSIC`` -- it accepts either a file or a folder:

* a **folder** -> the first font file inside it is used;
* a **file** (``.ttf``/``.ttc``/``.otf``/``.otc``) -> that file is used;
* an empty value or a bare font **name** -> looked up in the ``fonts/`` folders,
  then in a small list of well-known system fonts (DejaVu Sans, Arial, ...).

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


def _has_font_extension(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _FONT_EXTENSIONS


def _font_search_dirs(settings: Settings) -> list[str]:
    """Default folders searched when ``FONT`` is a bare font file name."""
    tool_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(tool_dir, "fonts"),
        os.path.join(settings.base_dir, "fonts"),
    ]
    # De-duplicate while preserving order.
    seen: list[str] = []
    for directory in candidates:
        if directory and directory not in seen:
            seen.append(directory)
    return seen


def _list_font_files(directory: str) -> list[str]:
    """Return the absolute paths of the font files directly inside ``directory``."""
    if not directory or not os.path.isdir(directory):
        return []
    files: list[str] = []
    for name in sorted(os.listdir(directory), key=str.lower):
        if name.startswith("."):
            continue
        full_path = os.path.join(directory, name)
        if _is_font_file(full_path) and _has_font_extension(full_path):
            files.append(full_path)
    return files


def resolve_font_path(settings: Settings) -> str:
    """Resolve the subtitle font to an absolute, existing file path.

    ``FONT`` is resolved like ``MUSIC``: a folder picks its first font file, a
    file is used as-is, and a bare name is looked up in the ``fonts/`` folders.
    Anything unresolved falls back to a well-known system font.

    Raises
    ------
    FontNotFoundError
        When no candidate could be resolved to an existing font file.
    """
    raw = (settings.font or "").strip()

    if raw:
        path = settings.resolve(raw)

        # 1. FONT points at a folder -> use the first font file inside it.
        if os.path.isdir(path):
            files = _list_font_files(path)
            if files:
                log.debug("using subtitle font from FONT folder: %s", files[0])
                return _normalize(files[0])
            log.warning("no font files found in FONT folder: %s", path)

        # 2. FONT points at an existing font file.
        elif _is_font_file(path) and _has_font_extension(path):
            log.debug("using subtitle font from FONT: %s", path)
            return _normalize(path)

        # 3. FONT is a bare name -> look it up in the default font folders.
        else:
            for directory in _font_search_dirs(settings):
                candidate = os.path.join(directory, raw)
                if _is_font_file(candidate):
                    log.debug("resolved subtitle font %r in %s", raw, directory)
                    return _normalize(candidate)

    # 4. Well-known system fonts --------------------------------------------
    for candidate in _SYSTEM_FONT_CANDIDATES:
        if _is_font_file(candidate):
            log.warning(
                "subtitle font %r was not found; falling back to system font: %s",
                raw or "(unset)",
                candidate,
            )
            return _normalize(candidate)

    searched = ", ".join(_font_search_dirs(settings)) or "(none)"
    raise FontNotFoundError(
        "could not find a subtitle font. Set FONT to a .ttf/.ttc/.otf file or a "
        f"folder with fonts, or place your font in one of: {searched}. "
        f"Requested FONT={raw!r}."
    )


__all__ = ["FontNotFoundError", "resolve_font_path"]
