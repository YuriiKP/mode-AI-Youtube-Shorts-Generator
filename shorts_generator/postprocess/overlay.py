"""Full-screen effect overlay for the rendered frame.

Some shorts look better with a light "texture" on top of the whole picture: a
soft light leak, drifting bokeh, falling snow, glitter, film grain, and so on.
Those are shipped as ordinary *footage* clips (MP4/WebM/... — usually bright
shapes on a black background), and this module composites one of them over the
entire frame.

Placement in the pipeline is deliberate: the overlay is applied to the whole
(already re-framed) picture, *after* the colour/lens pass and the ``UNIQUE_*``
anti-duplicate edits (both baked into the source by FFmpeg) and *before* the
subtitles and the banner are drawn, so the effect tints the video itself and
never the text — the letters stay crisp.

Footage files are read from a fixed ``effects/`` folder (a sibling of
``music/``), resolved against the working directory like every other asset. The
folder is intentionally **not** configurable: drop the clips you want in there
and the engine picks one at random per clip, mirroring how ``MUSIC`` picks a
track from a folder. The only knob is ``EFFECT_OPACITY`` (``0`` disables the
overlay entirely).

The blend is a *screen* (a.k.a. ``lighten``) composite, which is what these
light/particle overlays are made for: black areas of the footage leave the video
untouched and only the bright shapes are added on top. A plain alpha blend would
instead darken the picture wherever the footage is black, so ``opacity`` here
mixes the screened result back in (``0`` = source, ``1`` = full screen blend)
rather than controlling a straight transparency.

The per-frame maths runs in OpenCV (multi-threaded C), with a NumPy fallback, so
the extra cost stays a few tens of milliseconds per frame — the alternative, an
FFmpeg ``blend`` pass, cannot be used here because the effect has to sit *under*
the MoviePy-composited subtitles.
"""

from __future__ import annotations

import os
import random
from typing import Any, List, Optional, Tuple

from moviepy import VideoFileClip, vfx

from ..config import Settings
from .log import log

# Fixed name of the folder the effect footages are read from. Deliberately not a
# setting: this is a project asset folder, resolved against the working
# directory exactly like ``music/`` and ``fonts/``.
EFFECTS_DIRNAME = "effects"

# Video containers the effect footages are expected to use. Kept broad so a
# downloaded clip works as-is; unsupported extensions are simply ignored.
SUPPORTED_VIDEO_EXTENSIONS = (
    ".mp4",
    ".m4v",
    ".mov",
    ".webm",
    ".mkv",
    ".avi",
    ".gif",
)


# ---------------------------------------------------------------------------
# Footage discovery
# ---------------------------------------------------------------------------


def list_effect_files(directory: str) -> List[str]:
    """Return the absolute paths of supported footages in ``directory``.

    A missing directory is treated as empty, so the caller can degrade to "no
    overlay" instead of crashing.
    """
    directory = os.path.abspath(os.path.expanduser(directory)) if directory else ""
    if not directory or not os.path.isdir(directory):
        return []

    files: List[str] = []
    for name in sorted(os.listdir(directory), key=str.lower):
        # Skip hidden files and editor/packaging leftovers.
        if name.startswith("."):
            continue
        if os.path.splitext(name)[1].lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        full_path = os.path.join(directory, name)
        if os.path.isfile(full_path):
            files.append(full_path)
    return files


def resolve_effect_file(settings: Settings) -> str:
    """Pick a random effect footage from the fixed ``effects/`` folder.

    Returns an empty string (and logs a warning) when the folder is missing or
    holds no usable clip, so an enabled overlay degrades to "no effect" rather
    than failing the render.
    """
    directory = settings.resolve(EFFECTS_DIRNAME)
    files = list_effect_files(directory)
    if not files:
        log.warning(
            "EFFECT_OPACITY is set but no effect footage was found in %s; skipping the overlay",
            directory,
        )
        return ""

    chosen = random.choice(files)
    log.info(
        "picked random effect footage: %s (from %d file(s))",
        os.path.basename(chosen),
        len(files),
    )
    return chosen


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


def _as_uint8_rgb(frame: Any) -> Any:
    """Normalise a MoviePy frame to a ``uint8`` 8-bit RGB ``numpy`` array."""
    import numpy as np

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def _screen_frames(base: Any, over: Any, opacity: float) -> Any:
    """Screen-blend ``over`` onto ``base`` and mix it in by ``opacity``.

    ``screen(a, b) = 255 - (255 - a) * (255 - b) / 255``: black pixels of the
    overlay leave the video unchanged and bright pixels lighten it, which is the
    intended look for light/particle footages. The final frame is
    ``(1 - opacity) * base + opacity * screen``, so ``0`` reproduces the source
    exactly and ``1`` applies the full screen blend.

    Uses OpenCV (multi-threaded C) when available and falls back to NumPy.
    """
    import numpy as np

    base = _as_uint8_rgb(base)
    over = _as_uint8_rgb(over)
    if over.shape != base.shape:
        # Defensive: the overlay is cover-resized to the frame, but a mismatch
        # must never crash the render.
        over = _resize_frame(over, base.shape[1], base.shape[0])

    try:
        import cv2
    except Exception:  # pragma: no cover - OpenCV is a hard dependency
        cv2 = None

    if cv2 is not None:
        # ``screen`` = 255 - (255 - a) * (255 - b) / 255, all in uint8.
        not_base = cv2.bitwise_not(base)
        not_over = cv2.bitwise_not(over)
        product = cv2.multiply(not_base, not_over, scale=1.0 / 255.0)
        screened = cv2.bitwise_not(product)
        if opacity >= 1.0:
            return screened
        return cv2.addWeighted(base, 1.0 - opacity, screened, opacity, 0.0)

    base_f = base.astype(np.float32)
    over_f = over.astype(np.float32)
    screened_f = 255.0 - (255.0 - base_f) * (255.0 - over_f) / 255.0
    out = (1.0 - opacity) * base_f + opacity * screened_f
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _resize_frame(frame: Any, width: int, height: int) -> Any:
    """Resize a single frame to ``width`` x ``height`` (fallback path)."""
    try:
        import cv2

        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception:
        from PIL import Image

        return _as_uint8_rgb(
            Image.fromarray(frame).convert("RGB").resize((width, height))
        )


# ---------------------------------------------------------------------------
# Clip building
# ---------------------------------------------------------------------------


def _cover_resize(clip: Any, width: int, height: int) -> Any:
    """Scale ``clip`` to *cover* ``width`` x ``height`` and centre-crop it.

    Scaling to fill and then cropping (rather than stretching to the exact
    frame) keeps the footage's own aspect ratio, so a 16:9 light leak over a
    9:16 frame is cropped instead of squashed.
    """
    src_w, src_h = (int(value) for value in clip.size)
    if src_w <= 0 or src_h <= 0:
        return clip

    scale = max(width / float(src_w), height / float(src_h))
    resized_w = max(2, int(round(src_w * scale)))
    resized_h = max(2, int(round(src_h * scale)))

    if (resized_w, resized_h) != (src_w, src_h):
        clip = clip.resized((resized_w, resized_h))

    if (resized_w, resized_h) != (width, height):
        clip = clip.cropped(
            x_center=resized_w / 2.0,
            y_center=resized_h / 2.0,
            width=width,
            height=height,
        )
    return clip


def _screen_blend(base_clip: Any, overlay_clip: Any, opacity: float) -> Any:
    """Return ``base_clip`` with ``overlay_clip`` screened over every frame."""
    opacity = max(0.0, min(1.0, float(opacity)))

    def blend(get_frame, t):
        return _screen_frames(get_frame(t), overlay_clip.get_frame(t), opacity)

    return base_clip.transform(blend, keep_duration=True)


def build_effect_overlay(
    base_clip: Any, settings: Settings
) -> Tuple[Any, Optional[Any]]:
    """Composite a random ``effects/`` footage over the whole of ``base_clip``.

    Args:
        base_clip: the (already re-framed) picture the overlay is drawn onto.
        settings: resolved configuration; only ``EFFECT_OPACITY`` is read.

    Returns:
        ``(clip, source)``. ``clip`` is ``base_clip`` unchanged when the overlay
        is disabled, no footage is available or the footage cannot be read;
        otherwise it is a new clip with the effect screened on top. ``source``
        is the opened :class:`~moviepy.VideoFileClip` when an overlay was built
        — the caller **must** close it once done — and ``None`` otherwise.
    """
    opacity = max(0.0, min(1.0, float(settings.effect_opacity or 0.0)))
    if opacity <= 0.0:
        return base_clip, None

    path = resolve_effect_file(settings)
    if not path:
        return base_clip, None

    try:
        source = VideoFileClip(path, audio=False)
    except Exception as exc:  # noqa: BLE001 - surface a clean message, keep going
        log.warning("could not open effect footage %r: %s", path, exc)
        return base_clip, None

    try:
        duration = float(base_clip.duration or 0.0)
        overlay: Any = source
        if duration > 0:
            # Loop (or trim) the footage so it lasts exactly as long as the clip.
            overlay = overlay.with_effects([vfx.Loop(duration=duration)])

        width, height = (int(value) for value in base_clip.size)
        overlay = _cover_resize(overlay, width, height)

        result = _screen_blend(base_clip, overlay, opacity)
        log.info(
            "overlaying effect footage %s (opacity=%.2f, screen blend)",
            os.path.basename(path),
            opacity,
        )
        return result, source
    except Exception as exc:  # noqa: BLE001 - report and keep going
        log.warning("could not apply the effect overlay: %s", exc)
        try:
            source.close()
        except Exception:
            pass
        return base_clip, None


__all__ = [
    "EFFECTS_DIRNAME",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "build_effect_overlay",
    "list_effect_files",
    "resolve_effect_file",
]
