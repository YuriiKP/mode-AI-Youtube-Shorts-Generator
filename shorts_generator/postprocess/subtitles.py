"""Subtitle parsing and burn-in rendering.

The rendering mirrors MoneyPrinterTurbo's ``app/services/video.py`` so the
output looks identical: the same word wrapping, the same rounded/rectangular
semi-transparent backgrounds, the same vertical positioning, and the same
"center by visible pixels" trick for the text inside its background box.

Only the parts needed by the built-in engine are kept here. The SRT parser is
slightly more robust than MoviePy's bundled one: it flushes the final block
even when the file does not end with a blank line, and it tolerates both
comma and dot millisecond separators.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, TextClip
from PIL import Image, ImageDraw, ImageFont

from ..config import Settings
from .log import log

# A subtitle entry: ((start_seconds, end_seconds), text).
SubtitleItem = Tuple[Tuple[float, float], str]

_TIMESTAMP_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})")


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------


def _parse_timestamp(value: str) -> float:
    """Convert an SRT timestamp (``HH:MM:SS,mmm``) into seconds."""
    match = _TIMESTAMP_RE.search(value or "")
    if not match:
        raise ValueError(f"invalid SRT timestamp: {value!r}")
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    milliseconds = int(match.group(4).ljust(3, "0"))
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0


def load_subtitles(path: str) -> List[SubtitleItem]:
    """Parse an SRT file into ``[((start, end), text), ...]``.

    The parser accepts files where the last entry is not followed by a blank
    line (MoviePy's own parser silently drops that last entry), and tolerates
    both ``,`` and ``.`` as the millisecond separator.
    """
    if not path:
        return []
    if not os.path.isfile(path):
        raise FileNotFoundError(f"subtitle file not found: {path}")

    with open(path, "r", encoding="utf-8-sig") as handle:
        content = handle.read()
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    items: List[SubtitleItem] = []
    skipped = 0
    for block in re.split(r"\n\s*\n", content):
        block = block.strip("\n")
        if not block.strip():
            continue

        lines = block.split("\n")
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            skipped += 1
            continue

        before, _, after = lines[timing_index].partition("-->")
        try:
            start = _parse_timestamp(before)
            end = _parse_timestamp(after)
        except ValueError as exc:
            log.warning("skipping malformed subtitle block: %s", exc)
            skipped += 1
            continue

        text = "\n".join(lines[timing_index + 1 :]).strip("\n")
        if not text.strip():
            continue
        if end <= start:
            # Zero/negative duration clips are invisible; nudge them open so the
            # text still shows up instead of silently disappearing.
            end = start + 0.04
        items.append(((start, end), text))

    if skipped:
        log.warning("skipped %d malformed subtitle block(s) in %s", skipped, path)
    log.info(
        "loaded %d subtitle entr%s from %s",
        len(items),
        "y" if len(items) == 1 else "ies",
        os.path.basename(path),
    )
    return items


# ---------------------------------------------------------------------------
# Text wrapping (matches the original project)
# ---------------------------------------------------------------------------


def wrap_text(text, max_width, font="Arial", fontsize=60):
    """Wrap ``text`` so each line fits in ``max_width`` pixels.

    Returns ``(wrapped_text, height_pixels)`` where the height is derived from
    the font's own ascent+descent metrics, not the visible ink box, so multi
    line subtitles are never clipped.
    """
    font = ImageFont.truetype(font, fontsize)
    max_width = int(max_width)

    ascent, descent = font.getmetrics()
    line_height = int(ascent + descent)
    if line_height <= 0:
        log.warning(
            "invalid subtitle font metrics, fallback to font size: ascent=%s, "
            "descent=%s, fontsize=%s",
            ascent,
            descent,
            fontsize,
        )
        line_height = max(1, int(fontsize))

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, line_height
        left, top, right, bottom = font.getbbox(inner_text)
        return right - left, line_height

    width, height = get_text_size(text)
    if width <= max_width:
        # Respect explicit line breaks already present in the SRT entry.
        return text, (text.count("\n") + 1) * line_height

    def split_long_token(token):
        # Fall back to character-level splitting for tokens wider than the
        # available width (common for CJK text or very long words).
        lines = []
        current = ""
        for char in token:
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            lines.extend(split_long_token(word))
            current = ""

    if current:
        lines.append(current)

    line_start_punctuation = "，。！？；：、,.!?;:)]}）】》」』”’"
    for index in range(1, len(lines)):
        # Avoid leaving a closing punctuation mark alone at the start of a line
        # by pulling the previous line's last character down with it.
        if not lines[index] or lines[index][0] not in line_start_punctuation:
            continue
        if len(lines[index - 1]) <= 1:
            continue

        candidate = f"{lines[index - 1][-1]}{lines[index]}"
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            lines[index] = candidate
            lines[index - 1] = lines[index - 1][:-1]

    result = "\n".join(line.strip() for line in lines if line.strip()).strip()
    height = (result.count("\n") + 1) * line_height
    return result, height


# ---------------------------------------------------------------------------
# Background helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    """Convert ``#RRGGBB`` into an RGB tuple, falling back to black."""
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        try:
            return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
        except ValueError:
            pass
    return (0, 0, 0)


def _rounded_subtitle_background_clip(
    width: int,
    height: int,
    color: str,
    alpha: int = 140,
    radius: int = 16,
) -> ImageClip:
    """Build a semi-transparent rounded rectangle as a transparent ImageClip."""
    rgb = _hex_to_rgb(color)
    safe_alpha = max(0, min(255, int(alpha)))
    img = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, max(0, width - 1), max(0, height - 1)],
        radius=max(0, int(radius)),
        fill=(rgb[0], rgb[1], rgb[2], safe_alpha),
    )
    return ImageClip(np.array(img), transparent=True)


def _get_visible_center_position(
    text_clip: TextClip,
    container_width: int,
    container_height: int,
) -> Tuple[int, int]:
    """Center a TextClip inside its background box by visible pixels.

    MoviePy creates a transparent canvas sized by the font line height, so the
    visible glyphs are rarely at the geometric center. Reading the alpha mask
    and centering on the actual ink bounding box makes the text look properly
    centered regardless of the font's ascent/descent balance.
    """
    x = int(round((container_width - text_clip.w) / 2))
    y = int(round((container_height - text_clip.h) / 2))

    try:
        if text_clip.mask is None:
            return x, y

        mask_frame = text_clip.mask.get_frame(0)
        ys, _ = np.where(mask_frame > 0.01)
        if len(ys) == 0:
            return x, y

        visible_top = int(ys.min())
        visible_bottom = int(ys.max())
        visible_height = visible_bottom - visible_top + 1
        y = int(round((container_height - visible_height) / 2 - visible_top))
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("failed to center subtitle text by visible mask: %s", exc)

    return x, y


# ---------------------------------------------------------------------------
# Text clip construction
# ---------------------------------------------------------------------------


def create_text_clip(
    subtitle_item: SubtitleItem,
    settings: Settings,
    video_width: int,
    video_height: int,
    font_path: str,
):
    """Render a single subtitle entry into a positioned MoviePy clip."""
    font_size = int(settings.font_size)
    stroke_width = int(settings.stroke_width)
    phrase = subtitle_item[1]

    max_width = video_width * 0.9
    bg_color = settings.text_background_color
    rounded_bg_enabled = bool(settings.rounded_subtitle_background and bg_color)
    has_subtitle_background = bool(bg_color)

    padding_ratio = 0.4 if rounded_bg_enabled else 0.6
    pad_x = int(font_size * padding_ratio) if has_subtitle_background else 0
    text_max_width = max(1, int(max_width) - 2 * pad_x)

    wrapped_txt, txt_height = wrap_text(
        phrase,
        max_width=text_max_width,
        font=font_path,
        fontsize=font_size,
    )
    interline = int(font_size * 0.25)
    line_count = wrapped_txt.count("\n") + 1
    vertical_padding = int(font_size * 0.35)
    # Account for stroke expansion per line so multi-line bold outlines are not
    # clipped at the bottom.
    stroke_padding = int(stroke_width * 2 * line_count)
    text_clip_margin_y = max(int(font_size * 0.3), int(stroke_width * 2))
    clip_h = int(
        txt_height + vertical_padding + (interline * line_count) + stroke_padding
    )

    if rounded_bg_enabled:
        try:
            font = ImageFont.truetype(font_path, font_size)
            text_w = max(
                int(font.getbbox(line)[2] - font.getbbox(line)[0])
                for line in wrapped_txt.split("\n")
            )
        except Exception as exc:
            log.warning(
                "failed to measure subtitle text width, fallback to max width: %s",
                exc,
            )
            text_w = int(max_width)

        box_w = max(1, min(int(max_width), text_w + 2 * pad_x))
        radius = max(8, int(font_size * 0.4))
        text_clip = TextClip(
            text=wrapped_txt,
            font=font_path,
            font_size=font_size,
            color=settings.text_fore_color,
            bg_color=None,
            stroke_color=settings.stroke_color,
            stroke_width=stroke_width,
            interline=interline,
            size=(box_w, None),
            text_align="center",
            margin=(0, text_clip_margin_y),
        )
        clip_h = max(clip_h, text_clip.h)
        bg_clip = _rounded_subtitle_background_clip(
            width=box_w,
            height=clip_h,
            color=bg_color,
            alpha=140,
            radius=radius,
        )
        text_position = _get_visible_center_position(text_clip, box_w, clip_h)
        clip = CompositeVideoClip(
            [bg_clip, text_clip.with_position(text_position)],
            size=(box_w, clip_h),
        )
    elif bg_color:
        text_clip = TextClip(
            text=wrapped_txt,
            font=font_path,
            font_size=font_size,
            color=settings.text_fore_color,
            bg_color=None,
            stroke_color=settings.stroke_color,
            stroke_width=stroke_width,
            interline=interline,
            size=(int(max_width), None),
            text_align="center",
            margin=(0, text_clip_margin_y),
        )
        size = (int(max_width), max(clip_h, text_clip.h))
        bg_clip = _rounded_subtitle_background_clip(
            width=size[0],
            height=size[1],
            color=bg_color,
            alpha=255,
            radius=0,
        )
        text_position = _get_visible_center_position(text_clip, size[0], size[1])
        clip = CompositeVideoClip(
            [bg_clip, text_clip.with_position(text_position)],
            size=size,
        )
    else:
        clip = TextClip(
            text=wrapped_txt,
            font=font_path,
            font_size=font_size,
            color=settings.text_fore_color,
            bg_color=None,
            stroke_color=settings.stroke_color,
            stroke_width=stroke_width,
            interline=interline,
            size=(int(max_width), clip_h),
            text_align="center",
        )

    start, end = subtitle_item[0]
    clip = clip.with_start(start)
    clip = clip.with_end(end)
    clip = clip.with_duration(end - start)

    if settings.subtitle_position == "bottom":
        clip = clip.with_position(("center", video_height * 0.95 - clip.h))
    elif settings.subtitle_position == "top":
        clip = clip.with_position(("center", video_height * 0.05))
    elif settings.subtitle_position == "custom":
        margin = 10
        max_y = video_height - clip.h - margin
        custom_y = (video_height - clip.h) * (settings.custom_position / 100)
        custom_y = max(margin, min(custom_y, max_y))
        clip = clip.with_position(("center", custom_y))
    else:  # center
        clip = clip.with_position(("center", "center"))

    return clip


def build_subtitle_clips(
    subtitle_path: str,
    settings: Settings,
    video_width: int,
    video_height: int,
    font_path: str,
):
    """Parse ``subtitle_path`` and return a list of positioned text clips."""
    items = load_subtitles(subtitle_path)
    clips = []
    for item in items:
        try:
            clips.append(
                create_text_clip(
                    item,
                    settings,
                    video_width,
                    video_height,
                    font_path,
                )
            )
        except Exception as exc:
            log.error("failed to render subtitle %r: %s", item[1][:40], exc)
    return clips
