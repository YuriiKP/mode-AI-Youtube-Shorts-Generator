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

import math
import os
import re
from typing import List, Optional, Tuple

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, TextClip, VideoClip
from moviepy.video.fx import CrossFadeIn, Resize
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..config import Settings
from ..cues import split_segments_into_cues
from .log import log

# A subtitle entry: ((start_seconds, end_seconds), text).
SubtitleItem = Tuple[Tuple[float, float], str]

_TIMESTAMP_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})")


# ---------------------------------------------------------------------------
# MoviePy 2.x + Pillow 11.x: восстановление высоты текста
# ---------------------------------------------------------------------------
# MoviePy в ``TextClip`` считает высоту текста так: если у объекта ``ImageDraw``
# есть метод ``_multiline_spacing`` (он был в старых версиях Pillow), высота
# выходит равной ``(строк - 1) * межстрочный_шаг + (ascent + descent) + 2 * stroke``.
# В новых Pillow (например, 11.x) этот метод убрали, и MoviePy падает в
# резервную ветку, беря высоту по «чернилам» строки (bottom - top). Для строк
# без высоких букв (например, «щупальца дождя») такая высота меньше, чем нужно
# для базовой линии, и нижние выносные элементы (щ, у, д, я) подрезаются снизу.
#
# Возвращаем метод, повторяя шаг строки Pillow: ``textbbox("A")[3] + stroke + spacing``.
# Это чинит сразу и высоту картинки, и вертикальную центровку текста.
if not hasattr(ImageDraw.ImageDraw, "_multiline_spacing"):

    def _multiline_spacing(self, font, spacing, stroke_width=0):
        return (
            self.textbbox((0, 0), "A", font=font, stroke_width=stroke_width)[3]
            + stroke_width
            + spacing
        )

    setattr(ImageDraw.ImageDraw, "_multiline_spacing", _multiline_spacing)


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


def _blur_shadow_clip(clip: TextClip, radius: float) -> ImageClip:
    """Return a static copy of ``clip`` with its alpha silhouette blurred.

    The shadow text never changes over the life of a cue, so it is flattened
    into a single ``ImageClip`` whose mask is the blurred alpha. The Gaussian
    blur therefore runs once per cue instead of on every rendered frame.
    """
    frame = np.asarray(clip.get_frame(0))
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = Image.fromarray(frame[:, :, :3])

    if clip.mask is not None:
        alpha_frame = np.asarray(clip.mask.get_frame(0), dtype=np.float64)
        alpha = Image.fromarray(
            (np.clip(alpha_frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        )
    else:
        alpha = Image.new("L", rgb.size, 255)

    alpha = alpha.filter(ImageFilter.GaussianBlur(float(radius)))

    return ImageClip(np.array(rgb)).with_mask(
        ImageClip(np.asarray(alpha, dtype="float32") / 255.0, is_mask=True)
    )


def _subtitle_shadow_clip(
    wrapped_txt: str,
    settings: Settings,
    font_path: str,
    font_size: int,
    stroke_width: int,
    interline: int,
    size,
    margin,
) -> Optional[VideoClip]:
    """Build a solid drop-shadow copy of the subtitle text, or ``None``.

    The shadow is the same text rendered in a single flat colour — both fill and
    stroke use ``SUBTITLE_SHADOW_COLOR`` so the glyph reads as a solid silhouette
    — then faded to ``SUBTITLE_SHADOW_OPACITY``. :func:`create_text_clip`
    composites it behind the real text, shifted by ``SUBTITLE_SHADOW_OFFSET_X``
    and ``SUBTITLE_SHADOW_OFFSET_Y`` pixels.
    """
    if not getattr(settings, "subtitle_shadow", False):
        return None

    shadow_color = settings.subtitle_shadow_color
    clip = TextClip(
        text=wrapped_txt,
        font=font_path,
        font_size=font_size,
        color=shadow_color,
        bg_color=None,
        stroke_color=shadow_color,
        stroke_width=stroke_width,
        interline=interline,
        size=size,
        text_align="center",
        margin=margin,
    )

    opacity = float(getattr(settings, "subtitle_shadow_opacity", 1.0) or 0.0)
    opacity = max(0.0, min(1.0, opacity))
    if opacity < 1.0:
        clip = clip.with_opacity(opacity)

    blur = float(getattr(settings, "subtitle_shadow_blur", 0.0) or 0.0)
    if blur > 0:
        clip = _blur_shadow_clip(clip, blur)
    return clip


def _stack_subtitle_layers(
    layers,
    width: int,
    height: int,
    pad: int = 0,
):
    """Composite ``(clip, (x, y))`` layers onto a transparent canvas.

    ``pad`` grows the canvas on every side (and shifts each layer by the same
    amount) so a drop shadow offset outwards is not clipped by the layers below
    it. With a single layer and no padding it returns that clip directly.
    """
    if pad <= 0 and len(layers) == 1:
        clip, position = layers[0]
        return clip.with_position(position)

    positioned = [clip.with_position((x + pad, y + pad)) for clip, (x, y) in layers]
    return CompositeVideoClip(positioned, size=(width + 2 * pad, height + 2 * pad))


# ---------------------------------------------------------------------------
# Text clip construction
# ---------------------------------------------------------------------------


def _apply_appearance_animation(
    clip,
    base_x: float,
    base_y: float,
    settings: Settings,
):
    """Add a subtle entrance animation to an already-positioned subtitle clip.

    The animation is chosen by ``SUBTITLE_ANIMATION`` (empty = disabled) and its
    length by ``SUBTITLE_ANIMATION_DURATION``. It only plays at the very start of
    the cue; afterwards the text stays put until it disappears, so nothing keeps
    moving while the phrase is being read.

    * ``fade``  — the cue simply fades in.
    * ``slide`` — the cue fades in while rising into place from below.
    * ``pop``   — the cue fades in while scaling up from 70% to 100%.

    ``base_x``/``base_y`` are the resolved on-screen anchor of the clip so the
    motion can be applied around it.
    """
    mode = (getattr(settings, "subtitle_animation", "") or "").lower()
    if not mode:
        return clip

    duration = float(getattr(settings, "subtitle_animation_duration", 0.25) or 0.0)
    if duration <= 0 or not clip.duration or clip.duration <= 0:
        return clip
    # Never let the entrance eat more than half of a short cue.
    duration = min(duration, clip.duration / 2.0)

    # Every mode fades the clip in; this works because the cue is composited on
    # top of the video (see ``pipeline.run``). Slide/pop then add motion or
    # scale on top of that fade.
    clip = clip.with_effects([CrossFadeIn(duration)])

    if mode == "slide":
        distance = int(settings.font_size * 0.5)

        def _slide_position(t):
            progress = min(1.0, t / duration)
            eased = 1 - (1 - progress) ** 3  # ease-out cubic
            return (base_x, base_y + distance * (1 - eased))

        clip = clip.with_position(_slide_position)

    elif mode == "pop":
        base_w, base_h = clip.size

        def _scale(t):
            progress = min(1.0, t / duration)
            eased = 1 - (1 - progress) ** 3
            return 0.7 + 0.3 * eased

        clip = clip.with_effects([Resize(_scale)])

        def _pop_position(t):
            factor = _scale(t)
            # Keep the box centred on the same point while it scales up.
            return (
                base_x + base_w / 2 - base_w * factor / 2,
                base_y + base_h / 2 - base_h * factor / 2,
            )

        clip = clip.with_position(_pop_position)

    return clip


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

    # Measure the rendered text width so the background box can hug the text
    # instead of stretching across the full available width.
    text_w = int(max_width)
    if has_subtitle_background:
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

    # Drop-shadow placement. ``shadow_pad`` grows the compositing canvas so an
    # outward-offset, blurred shadow is not clipped by the tight background box.
    shadow_dx = int(getattr(settings, "subtitle_shadow_offset_x", 0) or 0)
    shadow_dy = int(getattr(settings, "subtitle_shadow_offset_y", 0) or 0)
    shadow_blur = max(0.0, float(getattr(settings, "subtitle_shadow_blur", 0.0) or 0.0))
    shadow_pad = (
        max(abs(shadow_dx), abs(shadow_dy)) + int(math.ceil(shadow_blur * 3))
        if getattr(settings, "subtitle_shadow", False)
        else 0
    )

    def _shadow_for(size, margin):
        return _subtitle_shadow_clip(
            wrapped_txt,
            settings,
            font_path,
            font_size,
            stroke_width,
            interline,
            size,
            margin,
        )

    if rounded_bg_enabled:
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
        layers: List[Tuple[VideoClip, Tuple[int, int]]] = [
            (bg_clip, (0, 0)),
            (text_clip, text_position),
        ]
        shadow = _shadow_for((box_w, None), (0, text_clip_margin_y))
        if shadow is not None:
            layers.insert(
                1,
                (shadow, (text_position[0] + shadow_dx, text_position[1] + shadow_dy)),
            )
        clip = _stack_subtitle_layers(layers, box_w, clip_h, shadow_pad)
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
            size=(box_w, None),
            text_align="center",
            margin=(0, text_clip_margin_y),
        )
        size = (box_w, max(clip_h, text_clip.h))
        bg_clip = _rounded_subtitle_background_clip(
            width=size[0],
            height=size[1],
            color=bg_color,
            alpha=255,
            radius=0,
        )
        text_position = _get_visible_center_position(text_clip, size[0], size[1])
        layers: List[Tuple[VideoClip, Tuple[int, int]]] = [
            (bg_clip, (0, 0)),
            (text_clip, text_position),
        ]
        shadow = _shadow_for((box_w, None), (0, text_clip_margin_y))
        if shadow is not None:
            layers.insert(
                1,
                (shadow, (text_position[0] + shadow_dx, text_position[1] + shadow_dy)),
            )
        clip = _stack_subtitle_layers(layers, size[0], size[1], shadow_pad)
    else:
        text_clip = TextClip(
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
        shadow = _shadow_for((int(max_width), clip_h), (None, None))
        if shadow is None:
            clip = text_clip
        else:
            # No background: recompose the text over its shadow on a transparent
            # canvas of the same footprint as before.
            clip = _stack_subtitle_layers(
                [(text_clip, (0, 0)), (shadow, (shadow_dx, shadow_dy))],
                int(max_width),
                clip_h,
                shadow_pad,
            )

    start, end = subtitle_item[0]
    clip = clip.with_start(start)
    clip = clip.with_end(end)
    clip = clip.with_duration(end - start)

    # Resolve the position to concrete pixels so the entrance animation can
    # offset/scale the cue around a fixed anchor point.
    clip_w, clip_h = clip.size
    base_x = int(round((video_width - clip_w) / 2))
    if settings.subtitle_position == "bottom":
        base_y = int(round(video_height * 0.95 - clip_h))
    elif settings.subtitle_position == "top":
        base_y = int(round(video_height * 0.05))
    elif settings.subtitle_position == "custom":
        margin = 10
        max_y = video_height - clip_h - margin
        custom_y = (video_height - clip_h) * (settings.custom_position / 100)
        base_y = int(round(max(margin, min(custom_y, max_y))))
    else:  # center
        base_y = int(round((video_height - clip_h) / 2))

    clip = clip.with_position((base_x, base_y))
    return _apply_appearance_animation(clip, base_x, base_y, settings)


def build_subtitle_clips(
    subtitle_path: str,
    settings: Settings,
    video_width: int,
    video_height: int,
    font_path: str,
):
    """Parse ``subtitle_path`` and return a list of positioned text clips.

    Each subtitle entry is passed through the cue splitter first, so an ``.srt``
    that still holds whole sentences (an old cache or a hand-made file) is broken
    into short on-screen phrases instead of one long block. Entries that are
    already short cues pass through unchanged.
    """
    items = load_subtitles(subtitle_path)
    cues = split_segments_into_cues(
        [{"start": start, "end": end, "text": text} for (start, end), text in items],
        max_chars=settings.subtitle_max_chars,
        max_words=settings.subtitle_max_words,
        max_duration=settings.subtitle_max_duration,
        pause_threshold=settings.subtitle_pause_threshold,
    )

    # Shift the whole cue along the timeline: ``SUBTITLE_OFFSET`` compensates
    # for Whisper word timings that lead the actual speech, so a positive value
    # makes the text appear later instead of a fraction of a second too early.
    # Only the on-screen burn-in moves; the ``.srt`` cache keeps raw timings.
    offset = float(getattr(settings, "subtitle_offset", 0.0) or 0.0)
    if offset:
        shifted = []
        for cue in cues:
            start = max(0.0, cue["start"] + offset)
            end = max(start + 0.04, cue["end"] + offset)
            shifted.append({"start": start, "end": end, "text": cue["text"]})
        cues = shifted

    clips = []
    for cue in cues:
        item = ((cue["start"], cue["end"]), cue["text"])
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
