"""Banner overlay for the rendered vertical frame.

A banner is a small strip drawn on top of the video — typically a channel handle,
a title or a logo. It is configured with a **single** ``BANNER`` setting whose
meaning depends on its value:

* if ``BANNER`` resolves to an **existing file**, it is used as an image banner
  (a PNG with alpha keeps its transparency, JPG works too);
* otherwise the value itself is drawn as a **text banner**: a full-width coloured
  band with the text centred inside it.

An empty ``BANNER`` draws nothing. The banner is positioned with
``BANNER_POSITION`` (``top`` / ``bottom`` / ``center``) and ``BANNER_MARGIN``
(distance from the frame edge). Problems (unreadable image, bad font) are logged
and skipped rather than aborting the whole render.

The returned clips are overlayed by the post-processing pipeline; they start at 0
and last as long as the video.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from moviepy import ImageClip, TextClip

from ..config import Settings
from .log import log


def banner_kind(settings: Settings) -> str:
    """Classify the configured ``BANNER`` value.

    Returns ``"image"`` when ``BANNER`` points at an existing file, ``"text"``
    when it holds anything else, and ``""`` when it is empty. The check is the
    whole point of the single-setting design: a path that exists is treated as
    an image, everything else as text.
    """
    value = (settings.banner or "").strip()
    if not value:
        return ""
    path = settings.resolve(value)
    if path and os.path.isfile(path):
        return "image"
    return "text"


def _place(clip: Any, settings: Settings, frame_height: int) -> Any:
    """Position ``clip`` horizontally centred at the configured banner slot."""
    margin = max(0, int(settings.banner_margin))
    position = (settings.banner_position or "top").strip().lower()

    if position == "bottom":
        y = frame_height - clip.h - margin
    elif position == "center":
        y = (frame_height - clip.h) / 2.0
    else:  # top (also the fallback for any unexpected value)
        y = margin

    y = max(0, min(y, max(0, frame_height - clip.h)))
    return clip.with_position(("center", y))


def _finish(clip: Any, settings: Settings, frame_height: int, duration: float) -> Any:
    """Apply the banner's position and lifetime (starts at 0, runs to the end)."""
    clip = _place(clip, settings, frame_height)
    clip = clip.with_start(0)
    if duration and duration > 0:
        clip = clip.with_duration(float(duration))
    return clip


def _build_image_banner(
    settings: Settings, frame_width: int, frame_height: int
) -> Optional[Any]:
    """Load ``BANNER`` as an image and scale it to the configured width fraction."""
    path = settings.resolve(settings.banner)
    try:
        clip: Any = ImageClip(path)
    except Exception as exc:  # noqa: BLE001 - surface a clean message, keep going
        log.warning("could not load banner image %r: %s", settings.banner, exc)
        return None

    ratio = float(settings.banner_width_ratio) or 1.0
    target_width = max(2, round(frame_width * ratio))
    if clip.w > 0 and clip.w != target_width:
        clip = clip.resized(width=target_width)

    # Never let a tall banner cover the whole frame.
    max_height = max(2, round(frame_height * 0.5))
    if clip.h > max_height:
        clip = clip.resized(height=max_height)

    opacity = float(settings.banner_opacity)
    if opacity < 1.0:
        clip = clip.with_opacity(max(0.0, min(1.0, opacity)))

    return clip


def _build_text_banner(
    settings: Settings, frame_width: int, font_path: str
) -> Optional[Any]:
    """Render ``BANNER`` as a full-width coloured band."""
    text = (settings.banner or "").strip()
    if not text:
        return None

    font_size = int(settings.banner_font_size)
    # ``size`` is the text area; a vertical margin pads the band. The horizontal
    # margin is zero so the band spans the full width of the frame.
    padding_y = max(8, font_size // 2)

    try:
        clip: Any = TextClip(
            text=text,
            font=font_path,
            font_size=font_size,
            color=settings.banner_text_color,
            bg_color=settings.banner_background_color,
            size=(frame_width, None),
            margin=(0, padding_y),
            text_align="center",
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean message, keep going
        log.warning("could not render text banner %r: %s", text, exc)
        return None

    return clip


def build_banner_clips(
    settings: Settings,
    frame_width: int,
    frame_height: int,
    duration: float,
    font_path: str,
) -> list[Any]:
    """Return the banner clip(s) to overlay on the frame.

    ``BANNER`` decides the kind (see :func:`banner_kind`). ``frame_width`` /
    ``frame_height`` describe the final (possibly re-framed) video size,
    ``duration`` its length in seconds and ``font_path`` the resolved font used
    by a text banner. Returns an empty list when no banner is configured.
    """
    kind = banner_kind(settings)
    if kind == "image":
        clip = _build_image_banner(settings, frame_width, frame_height)
    elif kind == "text":
        if not font_path:
            log.warning("no font available for the text banner; skipping banner")
            return []
        clip = _build_text_banner(settings, frame_width, font_path)
    else:
        return []

    if clip is None:
        return []

    clip = _finish(clip, settings, frame_height, duration)
    log.info("added %s banner at %s", kind, settings.banner_position)
    return [clip]


__all__ = ["banner_kind", "build_banner_clips"]
