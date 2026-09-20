"""Fast single-frame preview of the post-processing look.

Rendering a real short takes a while: every frame is encoded, the audio is
mixed and, for the ``all`` command, Whisper runs first. When the goal is only
to tune the *picture* — the vertical frame with its blurred background, the
colour/lens effects, the banner and the subtitle look — that whole pipeline is
overkill.

This module renders **one random frame** through exactly the same stages a real
render uses, so what you see is what the final clip will look like, but it
finishes in a second or two. That makes it practical to iterate on ``.env``
values (``FIT_*``, ``BACKGROUND_*``, ``SATURATION`` / ``SHARPNESS`` /
``CHROMATIC_ABERRATION``, ``BANNER_*``, the whole subtitle appearance...) and
see the result immediately.

No transcription happens here. When subtitles are enabled the fixed placeholder
text :data:`TEST_FRAME_TEXT` is drawn with the very same renderer that burns in
real cues, so the subtitle look is previewed faithfully without waiting for
Whisper.
"""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np
from moviepy import CompositeVideoClip, ImageClip

from .config import Settings
from .pipeline import resolve_input_videos
from .postprocess.banner import banner_kind, build_banner_clips
from .postprocess.effects import build_filter_chain
from .postprocess.ffmpeg import configure_ffmpeg
from .postprocess.fonts import resolve_font_path
from .postprocess.layout import build_vertical_clip, needs_vertical_fit
from .postprocess.log import log
from .postprocess.pipeline import (
    ProcessingError,
    _open_video,
    _remove_file,
    _safe_close,
)
from .postprocess.subtitles import create_text_clip

# Placeholder burned into the test frame when subtitles are enabled. Speech
# recognition is not needed to preview the subtitle look, so this fixed sample
# text is drawn by the same :func:`create_text_clip` used for real cues.
TEST_FRAME_TEXT = "тестовый кадр"

# Length of the synthetic preview clip, in seconds. The overlays are time-based
# (the subtitle entrance animation, the banner lifetime), so the frame is
# sampled near the end, where everything has settled into its final look.
_PREVIEW_DURATION = 2.0

# How far before the end the frame is sampled, in seconds. Kept small so the
# subtitle cue is still on screen but past any entrance animation.
_PREVIEW_SAMPLE_GAP = 0.05

_PREVIEW_FILENAME = "preview_frame.png"


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


def _as_uint8_rgb(frame) -> np.ndarray:
    """Normalise a MoviePy frame to a ``uint8`` RGB ``numpy`` array."""
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def _extract_frame(
    source_path: str,
    ffmpeg_binary: str,
    timestamp: Optional[float],
) -> Tuple[np.ndarray, float, float]:
    """Decode a single frame from ``source_path``.

    When ``timestamp`` is ``None`` a random moment inside the video is chosen.
    Returns ``(frame, used_timestamp, duration)``.
    """
    clip, scrubbed = _open_video(source_path, ffmpeg_binary)
    try:
        duration = float(getattr(clip, "duration", 0.0) or 0.0)
        if duration <= 0:
            raise ProcessingError(f"could not read the duration of {source_path!r}")

        used_t = (
            random.uniform(0.0, duration) if timestamp is None else float(timestamp)
        )
        # Keep a hair away from the very end so a frame always exists.
        used_t = max(0.0, min(used_t, duration - _PREVIEW_SAMPLE_GAP))
        frame = _as_uint8_rgb(clip.get_frame(used_t))
    finally:
        _safe_close(clip)
        _remove_file(scrubbed)
    return frame, used_t, duration


def _apply_effects_to_frame(
    frame: np.ndarray,
    filter_chain: str,
    ffmpeg_binary: str,
) -> np.ndarray:
    """Run the configured colour/lens filtergraph over a single frame.

    The real pipeline bakes these effects into the source with one FFmpeg pass;
    applying the exact same ``-vf`` graph to a still image reproduces the look
    faithfully. Returns the frame unchanged when the graph is empty.
    """
    if not filter_chain:
        return frame

    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="shorts_preview_") as tmp:
        src = os.path.join(tmp, "in.png")
        dst = os.path.join(tmp, "out.png")
        Image.fromarray(frame).save(src)

        cmd = [
            ffmpeg_binary,
            "-y",
            "-loglevel",
            "error",
            "-i",
            src,
            "-vf",
            filter_chain,
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            dst,
        ]
        log.info("applying video filters to the preview frame: %s", filter_chain)
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0 or not os.path.isfile(dst):
            stderr = (result.stderr or b"").decode("utf-8", "replace").strip()
            raise ProcessingError(f"failed to apply video effects: {stderr}")
        return _as_uint8_rgb(np.asarray(Image.open(dst).convert("RGB")))


def _compose_frame(
    frame: np.ndarray,
    settings: Settings,
    subtitles: bool,
) -> np.ndarray:
    """Re-frame ``frame`` and draw the overlays, returning the final picture."""
    duration = _PREVIEW_DURATION
    base = ImageClip(frame).with_duration(duration)
    vertical = base
    try:
        if needs_vertical_fit(base, settings):
            vertical = build_vertical_clip(base, settings)

        width, height = (int(value) for value in vertical.size)

        # The font is only needed when something textual is drawn (subtitles or
        # a text banner), mirroring the real pipeline.
        needs_font = subtitles or banner_kind(settings) == "text"
        font_path = resolve_font_path(settings) if needs_font else ""

        overlays = []
        if subtitles:
            cue = ((0.0, duration), TEST_FRAME_TEXT)
            overlays.append(create_text_clip(cue, settings, width, height, font_path))
        overlays.extend(
            build_banner_clips(settings, width, height, duration, font_path)
        )

        sample_t = duration - _PREVIEW_SAMPLE_GAP
        if not overlays:
            return _as_uint8_rgb(vertical.get_frame(sample_t))

        composite = CompositeVideoClip(
            [vertical, *overlays], size=(width, height), use_bgclip=True
        ).with_duration(duration)
        try:
            return _as_uint8_rgb(composite.get_frame(sample_t))
        finally:
            _safe_close(composite)
    finally:
        if vertical is not base:
            _safe_close(vertical)
        _safe_close(base)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_preview_frames(
    settings: Settings,
    *,
    count: int = 1,
    subtitles: bool = True,
    timestamp: Optional[float] = None,
) -> List[str]:
    """Render one or more test frames and return the written PNG paths.

    Args:
        settings: resolved configuration (input, output dir, vertical frame,
            effects, banner, subtitle look).
        count: how many random frames to render (default: 1).
        subtitles: draw the placeholder subtitle cue (default: on).
        timestamp: use this timestamp instead of a random one, handy to compare
            settings against the exact same frame.

    Returns:
        The absolute paths of the written ``.png`` files.
    """
    from PIL import Image

    ffmpeg_binary = configure_ffmpeg(settings.ffmpeg_path)

    sources = resolve_input_videos(settings)
    if not sources:
        raise ProcessingError("no input video found to preview")

    out_dir = settings.resolve(settings.output_dir) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    filter_chain = build_filter_chain(settings)
    count = max(1, int(count))
    written: List[str] = []

    for index in range(count):
        source = random.choice(sources)
        frame, used_t, duration = _extract_frame(source, ffmpeg_binary, timestamp)
        print(
            f"[preview] {index + 1}/{count}: {os.path.basename(source)} "
            f"@ {used_t:.2f}s of {duration:.2f}s",
            flush=True,
        )

        frame = _apply_effects_to_frame(frame, filter_chain, ffmpeg_binary)
        composed = _compose_frame(frame, settings, subtitles)

        name = _PREVIEW_FILENAME if count == 1 else f"preview_frame_{index + 1}.png"
        path = os.path.join(out_dir, name)
        Image.fromarray(composed).save(path)
        written.append(path)
        print(f"[preview]   -> {path}", flush=True)

    log.info("wrote %d preview frame(s)", len(written))
    return written


__all__ = ["TEST_FRAME_TEXT", "render_preview_frames"]
