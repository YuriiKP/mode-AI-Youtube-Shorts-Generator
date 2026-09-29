"""Read the clips produced by the clipping pipeline.

The publisher does not care *how* a short was rendered — it only needs, per
clip: the video file, a title, a description and (optionally) hashtags. Those
come from the clipping pipeline in one of three shapes, tried in order:

1. ``output/shorts_info.json`` — the machine-readable sidecar written next to
   the clips (preferred: exact, no ambiguity);
2. ``output/shorts_info.txt`` — the human-readable copy-paste sheet, parsed as a
   fallback for runs that predate the JSON sidecar;
3. a plain scan of the output directory for video files, used when neither
   sidecar exists (title defaults to the file's base name).

A single ``--video path`` bypasses all of the above and builds one short, pulling
its title/hashtags from the ``<video>.txt`` sidecar when present.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

from .log import log as _log
from .model import Short, guess_thumbnail

# Video containers the publisher recognises when scanning a directory.
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".m4v")

JSON_SIDECAR = "shorts_info.json"
TXT_SIDECAR = "shorts_info.txt"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_shorts(
    output_dir: str,
    *,
    json_path: Optional[str] = None,
    video: Optional[str] = None,
    only: Optional[Union[str, Sequence[str]]] = None,
    limit: Optional[int] = None,
    base_dir: Optional[str] = None,
    logger=None,
) -> List[Short]:
    """Load the shorts to publish.

    Args:
        output_dir: directory the pipeline wrote its clips into.
        json_path: explicit path to a ``shorts_info.json`` (overrides the default
            ``<output_dir>/shorts_info.json``).
        video: a single video file to publish instead of scanning ``output_dir``.
        only: keep only shorts matching these tokens (see :func:`_matches_only`).
            Accepts a comma-separated string or a sequence.
        limit: keep at most this many shorts (applied after ``only``).
        base_dir: directory relative paths are resolved against (defaults to the
            current working directory).

    Returns:
        The list of :class:`Short` objects, in pipeline order.
    """
    log = logger or _log
    working_dir = os.path.abspath(base_dir or os.getcwd())
    out_dir = _resolve(output_dir, working_dir)

    if video:
        shorts = [_single_from_video(video, working_dir)]
        log.info("publishing a single video: %s", shorts[0].name)
    else:
        shorts = _load_from_output_dir(out_dir, json_path, log)

    shorts = _apply_only(shorts, only)
    if limit is not None and limit >= 0:
        shorts = shorts[:limit]

    return shorts


# ---------------------------------------------------------------------------
# Loading strategies
# ---------------------------------------------------------------------------


def _load_from_output_dir(out_dir: str, json_path: Optional[str], log) -> List[Short]:
    """Load shorts from the JSON sidecar, then the text sheet, then a scan."""
    json_file = (
        _resolve(json_path, os.getcwd())
        if json_path
        else os.path.join(out_dir, JSON_SIDECAR)
    )

    if os.path.isfile(json_file):
        shorts = _parse_json_report(json_file, out_dir)
        if shorts:
            log.info("loaded %d short(s) from %s", len(shorts), json_file)
            return shorts
        log.warning("sidecar %s contained no usable shorts; falling back", json_file)

    txt_file = os.path.join(out_dir, TXT_SIDECAR)
    if os.path.isfile(txt_file):
        shorts = _parse_text_report(txt_file, out_dir)
        if shorts:
            log.info("loaded %d short(s) from %s", len(shorts), txt_file)
            return shorts
        log.warning("text sheet %s contained no usable shorts; falling back", txt_file)

    shorts = _scan_directory(out_dir)
    if shorts:
        log.info(
            "no sidecar found; scanned %s and found %d clip(s)", out_dir, len(shorts)
        )
    else:
        log.warning("no shorts found in %s", out_dir)
    return shorts


def _parse_json_report(json_file: str, out_dir: str) -> List[Short]:
    """Parse a ``shorts_info.json`` sidecar into :class:`Short` objects."""
    import json

    try:
        with open(json_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        _log.warning("could not read %s: %s", json_file, exc)
        return []

    entries = payload.get("shorts") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []

    shorts: List[Short] = []
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            continue
        short = Short.from_mapping(entry, base_dir=out_dir, number=index)
        if short.file:
            shorts.append(short)
    return shorts


def _parse_text_report(txt_file: str, out_dir: str) -> List[Short]:
    """Parse a ``shorts_info.txt`` sheet into :class:`Short` objects.

    The sheet is written as a sequence of blank-line separated blocks, each
    looking like::

        Источник: source.mkv

        #1
        Название:    ...
        Описание:    ...
        Файл:        short_01_source.mp4
    """
    with open(txt_file, "r", encoding="utf-8-sig") as handle:
        text = handle.read()

    field_map = {"Название": "title", "Описание": "description", "Файл": "file"}
    shorts: List[Short] = []
    current: dict = {}
    pending_source = ""

    def flush() -> None:
        nonlocal current
        if current.get("file"):
            current.setdefault("source", pending_source)
            shorts.append(
                Short.from_mapping(current, base_dir=out_dir, number=len(shorts) + 1)
            )
        current = {}

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush()
            continue
        stripped = line.strip()
        if stripped.startswith("#") and stripped[1:].strip().isdigit():
            flush()
            current["number"] = int(stripped[1:].strip())
            continue
        if ":" not in line:
            continue  # header / separator lines carry no ":"
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "Источник":
            pending_source = value
        elif key in field_map:
            current[field_map[key]] = value

    flush()
    return shorts


def _scan_directory(out_dir: str) -> List[Short]:
    """Build shorts from the video files directly inside ``out_dir``."""
    if not os.path.isdir(out_dir):
        return []
    files = [
        name
        for name in sorted(os.listdir(out_dir))
        if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS
    ]
    shorts: List[Short] = []
    for index, name in enumerate(files, 1):
        path = os.path.join(out_dir, name)
        shorts.append(
            Short(
                number=index,
                title=os.path.splitext(name)[0],
                description="",
                file=path,
                thumbnail=guess_thumbnail(path),
            )
        )
    return shorts


def _single_from_video(video: str, base_dir: str) -> Short:
    """Build a one-element short list entry from a single ``--video`` path."""
    path = _resolve(video, base_dir)
    title, tags = _read_title_tags_sidecar(path)
    if not title:
        title = os.path.splitext(os.path.basename(path))[0]
    return Short(
        number=1,
        title=title,
        description="",
        tags=tags,
        file=path,
        thumbnail=guess_thumbnail(path),
    )


def _read_title_tags_sidecar(video_path: str) -> "tuple[str, List[str]]":
    """Read the ``<video>.txt`` sidecar (title line + hashtag line), if any.

    This mirrors the convention used across the project: a ``.txt`` next to the
    video, whose first line is the title and whose second line is a list of
    hashtags.
    """
    stem, _ = os.path.splitext(video_path)
    txt_path = stem + ".txt"
    if not os.path.isfile(txt_path):
        return "", []
    try:
        with open(txt_path, "r", encoding="utf-8-sig") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except OSError:
        return "", []

    title = lines[0] if lines else ""
    tags: List[str] = []
    if len(lines) > 1:
        tags = [token.lstrip("#") for token in lines[1].replace(",", " ").split()]
    return title, [tag for tag in tags if tag]


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------


def _apply_only(shorts: List[Short], only) -> List[Short]:
    tokens = _normalize_only(only)
    if not tokens:
        return shorts
    return [short for short in shorts if _matches_only(short, tokens)]


def _normalize_only(only) -> List[str]:
    if not only:
        return []
    if isinstance(only, str):
        items = only.replace(";", ",").split(",")
    else:
        items = list(only)
    return [str(item).strip().lower() for item in items if str(item).strip()]


def _matches_only(short: Short, tokens: List[str]) -> bool:
    """Whether a short matches any of the ``--only`` tokens.

    Flexible on purpose, so any of these select ``short_01_talk.mp4``::

        --only 1
        --only 01
        --only short_01
        --only talk
    """
    name = short.name.lower()
    title = (short.title or "").lower()
    number = short.number
    for token in tokens:
        if token in (str(number), f"{number:02d}", f"short_{number:02d}"):
            return True
        if token in name or (title and token in title):
            return True
    return False


def _resolve(path: str, base_dir: str) -> str:
    """Resolve a possibly-relative path against ``base_dir``."""
    if not path:
        return ""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(base_dir, expanded))


__all__ = ["load_shorts", "JSON_SIDECAR", "TXT_SIDECAR", "VIDEO_EXTENSIONS"]
