"""The :class:`Short` data model shared by the publisher.

A ``Short`` is one rendered clip together with everything that is needed to
publish it: the video file, its title, description and hashtags. The publisher
builds these objects either from the machine-readable sidecar written by the
clipping pipeline (``output/shorts_info.json``), from its human-readable
``output/shorts_info.txt`` counterpart, or from a plain scan of the output
directory.

The model is intentionally dependency-free (only the standard library) so it can
be imported anywhere — including from tests — without pulling in the browser
stack.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional

# Files that are read in chunks when hashing, so multi-hundred-megabyte clips do
# not have to fit in memory.
_HASH_CHUNK_SIZE = 1 << 20  # 1 MiB


def normalize_tags(value: Any) -> List[str]:
    """Normalize a tag list into clean, unique hashtag words.

    Accepts a list/tuple/set or a single comma/semicolon/space separated
    string, strips a leading ``#`` from each item, drops empties and removes
    duplicates while preserving order. This mirrors
    :func:`publisher.config.parse_tags` but is duplicated here so the model has
    no dependency on the configuration module.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items: Iterable[str] = (str(item) for item in value)
    else:
        text = str(value)
        for separator in (";", ","):
            text = text.replace(separator, " ")
        raw_items = text.split()

    seen: List[str] = []
    for item in raw_items:
        tag = item.strip().lstrip("#").strip()
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def file_sha1(path: str) -> str:
    """Return the hexadecimal SHA-1 of a file's contents.

    Used as the identity of a clip for the "already uploaded?" dedupe table, so
    re-running the publisher never re-uploads the same file to the same
    profile/platform. Raises :class:`FileNotFoundError` when the file is
    missing.
    """
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guess_thumbnail(video_path: str) -> Optional[str]:
    """Return a same-named ``.png``/``.jpg`` next to a video, if one exists.

    This mirrors the convention used by ``social-auto-upload`` (a thumbnail sits
    beside the clip with the same base name). Returns ``None`` when no sidecar
    image is found.
    """
    stem, _ = os.path.splitext(video_path)
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = stem + ext
        if os.path.isfile(candidate):
            return candidate
    return None


@dataclass
class Short:
    """One clip ready to be published."""

    # Position in the run (1-based, matching the ``#N`` numbering and the
    # ``short_NN_...`` file names produced by the pipeline).
    number: int = 0
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    # Absolute path to the video file.
    file: str = ""
    # Optional absolute path to a thumbnail image.
    thumbnail: Optional[str] = None
    # Source video the clip was cut from (for reporting only).
    source: str = ""
    # Extra metadata carried over from the highlight, kept for reporting.
    clip_type: str = ""
    score: int = 0
    hook_sentence: str = ""
    punchline: str = ""

    # -- derived values ----------------------------------------------------

    @property
    def name(self) -> str:
        """Base file name of the clip (e.g. ``short_01_talk.mp4``)."""
        return os.path.basename(self.file)

    @property
    def exists(self) -> bool:
        """Whether the video file is present on disk."""
        return bool(self.file) and os.path.isfile(self.file)

    def effective_tags(self, defaults: Optional[Iterable[str]] = None) -> List[str]:
        """The clip's tags, falling back to ``defaults`` when it has none."""
        if self.tags:
            return list(self.tags)
        return normalize_tags(list(defaults) if defaults else [])

    def caption(self, defaults: Optional[Iterable[str]] = None) -> str:
        """Build a platform caption: the title followed by ``#hashtags``.

        Used by TikTok, where the hashtags live in the description text rather
        than in a dedicated field.
        """
        tags = self.effective_tags(defaults)
        parts: List[str] = []
        if self.title.strip():
            parts.append(self.title.strip())
        if self.description.strip():
            parts.append(self.description.strip())
        if tags:
            parts.append(" ".join("#" + tag for tag in tags))
        return "\n".join(parts)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: str = "",
        number: int = 0,
    ) -> "Short":
        """Build a :class:`Short` from a ``shorts_info.json`` entry.

        ``base_dir`` is used to resolve a relative ``file`` against the output
        directory. A thumbnail is taken from the entry when present, otherwise
        guessed from a same-named ``.png`` sidecar.
        """
        raw_file = str(data.get("file") or data.get("clip_url") or "").strip()
        if raw_file and base_dir and not os.path.isabs(raw_file):
            raw_file = os.path.normpath(os.path.join(base_dir, raw_file))

        thumbnail = data.get("thumbnail")
        thumbnail = str(thumbnail).strip() if thumbnail else None
        if thumbnail and base_dir and not os.path.isabs(thumbnail):
            thumbnail = os.path.normpath(os.path.join(base_dir, thumbnail))
        if not thumbnail and raw_file:
            thumbnail = guess_thumbnail(raw_file)

        try:
            number_value = int(data.get("number")) if data.get("number") else number
        except (TypeError, ValueError):
            number_value = number

        try:
            score_value = int(data.get("score") or 0)
        except (TypeError, ValueError):
            score_value = 0

        return cls(
            number=number_value,
            title=str(data.get("title") or "").strip(),
            description=str(data.get("description") or "").strip(),
            tags=normalize_tags(data.get("tags")),
            file=raw_file,
            thumbnail=thumbnail,
            source=str(data.get("source") or "").strip(),
            clip_type=str(data.get("clip_type") or "").strip(),
            score=score_value,
            hook_sentence=str(data.get("hook_sentence") or "").strip(),
            punchline=str(data.get("punchline") or "").strip(),
        )

    def to_dict(self) -> dict:
        """Serialise back to the ``shorts_info.json`` shape."""
        return {
            "number": self.number,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "file": self.name,
            "thumbnail": os.path.basename(self.thumbnail) if self.thumbnail else None,
            "source": self.source,
            "clip_type": self.clip_type,
            "score": self.score,
            "hook_sentence": self.hook_sentence,
            "punchline": self.punchline,
        }


__all__ = [
    "Short",
    "file_sha1",
    "guess_thumbnail",
    "normalize_tags",
]
