"""Vertical re-framing with a blurred background fill.

Clips produced by other tools are not always in the target aspect ratio (for
example a landscape recording, or a clip cropped to ``9:11``). This module
re-frames such a clip into a clean vertical frame without hard black bars:

* the clip is scaled to *fit* inside the frame (``contain``) and centred;
* a *cover*-scaled copy of the same clip is cropped to the frame and blurred,
  filling the empty area behind the video.

The result is a composited :class:`~moviepy.CompositeVideoClip`, so it drops
straight into the existing subtitle/music pipeline and still costs a single
re-encode.
"""

from __future__ import annotations

from typing import Any

from moviepy import CompositeVideoClip, VideoClip

from ..config import Settings, parse_aspect_ratio
from .log import log

# A clip whose ratio is within this tolerance of the target is left untouched.
_RATIO_TOLERANCE = 5e-3

# The blurred fill is produced by shrinking the frame, blurring the small copy
# and scaling it back up: it looks identical but is orders of magnitude cheaper
# than a large-kernel blur at full resolution.
_BLUR_DOWNSCALE = 8


def _even(value: float) -> int:
    """Round to the nearest even integer (>= 2) for video encoders."""
    number = round(value)
    number -= number % 2
    return max(2, number)


def _target_size(ratio: float, height: int) -> tuple[int, int]:
    """Return the ``(width, height)`` canvas for ``ratio`` at ``height`` px."""
    target_h = _even(height)
    target_w = _even(target_h * ratio)
    return target_w, target_h


def _cover_crop(src_w: int, src_h: int, ratio: float) -> tuple[int, int]:
    """Return the largest ``(width, height)`` with ``ratio`` inside the source.

    The result keeps the source centred and is always within its bounds, so the
    background can be cropped *before* it is scaled up — this keeps memory use
    bounded even for very wide or very tall sources.
    """
    if src_w / float(src_h) > ratio:
        crop_w, crop_h = _even(src_h * ratio), _even(src_h)
    else:
        crop_w, crop_h = _even(src_w), _even(src_w / ratio)
    return min(crop_w, src_w), min(crop_h, src_h)


def _contain_size(
    src_w: int, src_h: int, target_w: int, target_h: int
) -> tuple[int, int]:
    """Return the largest size that fits inside ``target`` keeping the ratio."""
    src_ratio = src_w / float(src_h)
    if src_ratio > target_w / float(target_h):
        return target_w, _even(target_w / src_ratio)
    return _even(target_h * src_ratio), target_h


def _blur_frame(frame: Any, blur: int, darken: float) -> Any:
    """Blur and/or darken a single RGB frame using OpenCV.

    The function is dtype-agnostic (MoviePy frames are ``uint8``, but the cast
    back to the original dtype keeps it safe for other readers too).
    """
    import cv2

    height, width = frame.shape[:2]
    small_w = max(2, width // _BLUR_DOWNSCALE)
    small_h = max(2, height // _BLUR_DOWNSCALE)
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    if blur > 0:
        kernel = max(1, int(blur)) | 1  # Gaussian kernels must be odd
        small = cv2.GaussianBlur(small, (kernel, kernel), 0)
    frame = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if darken > 0:
        frame = (frame * (1.0 - darken)).astype(frame.dtype)
    return frame


def _blurred_copy(clip: VideoClip, blur: int, darken: float) -> Any:
    """Return a blurred/darkened copy of ``clip`` (or ``clip`` unchanged)."""
    blur = max(0, int(blur))
    darken = max(0.0, min(1.0, float(darken)))
    if blur <= 0 and darken <= 0:
        return clip
    try:
        return clip.image_transform(lambda frame: _blur_frame(frame, blur, darken))
    except Exception as exc:  # pragma: no cover - depends on OpenCV presence
        log.warning("background blur unavailable (%s); using a plain fill", exc)
        return clip


def needs_vertical_fit(clip: VideoClip, settings: Settings) -> bool:
    """Whether ``clip`` should be re-framed into the configured vertical frame."""
    if not settings.fit_vertical:
        return False
    ratio = parse_aspect_ratio(settings.fit_aspect_ratio or settings.aspect_ratio)
    src_w, src_h = clip.size
    return abs((src_w / float(src_h)) - ratio) > _RATIO_TOLERANCE


def _fit_vertical_moviepy(
    clip: VideoClip,
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    fg_w: int,
    fg_h: int,
    crop_w: int,
    crop_h: int,
    blur: int,
    darken: float,
) -> Any:
    """MoviePy-based vertical fit (fallback for when OpenCV is unavailable).

    This is the original implementation: a ``CompositeVideoClip`` of a resized
    foreground over a blurred, cover-cropped background. It works everywhere but
    runs two PIL bicubic resizes plus a full-resolution RGBA composite per frame,
    which is an order of magnitude slower than :func:`_make_cv2_reframer`.
    """
    foreground: Any = clip.resized((fg_w, fg_h))

    # Crop the source to the target ratio first, then scale to the canvas: the
    # intermediate frame never exceeds the source size.
    background: Any = clip.cropped(
        x_center=src_w // 2,
        y_center=src_h // 2,
        width=crop_w,
        height=crop_h,
    )
    background = background.resized((target_w, target_h))
    background = _blurred_copy(background, blur, darken)

    return CompositeVideoClip(
        [background, foreground.with_position(("center", "center"))],
        size=(target_w, target_h),
    )


def _make_cv2_reframer(
    clip: VideoClip,
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    fg_w: int,
    fg_h: int,
    crop_w: int,
    crop_h: int,
    blur: int,
    darken: float,
) -> Any:
    """Re-frame ``clip`` per frame with a single OpenCV pass.

    Building the foreground/background as separate MoviePy layers made every
    output frame run two PIL bicubic resizes *and* a full-resolution RGBA
    composite (~250 ms/frame for a 1080x1920 canvas). OpenCV's resize and blur
    cost a few milliseconds at the same sizes, so doing the whole re-frame once
    in OpenCV is ~15-20x cheaper for the exact same picture.
    """
    import cv2

    bg_x = max(0, (src_w - crop_w) // 2)
    bg_y = max(0, (src_h - crop_h) // 2)
    fg_x = max(0, (target_w - fg_w) // 2)
    fg_y = max(0, (target_h - fg_h) // 2)
    # INTER_AREA is the best choice when shrinking; for the (common) upscale to
    # the canvas, a cubic filter keeps the foreground as sharp as MoviePy's
    # default bicubic resizer did.
    fg_interp = cv2.INTER_AREA if (fg_w < src_w and fg_h < src_h) else cv2.INTER_CUBIC

    def reframe(frame: Any) -> Any:
        background = frame[bg_y : bg_y + crop_h, bg_x : bg_x + crop_w]
        background = cv2.resize(
            background, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
        # ``_blur_frame`` always returns a fresh array, so writing the foreground
        # into it in place is safe.
        canvas = _blur_frame(background, blur, darken)
        foreground = cv2.resize(frame, (fg_w, fg_h), interpolation=fg_interp)
        canvas[fg_y : fg_y + fg_h, fg_x : fg_x + fg_w] = foreground
        return canvas

    return clip.image_transform(reframe)


def build_vertical_clip(clip: VideoClip, settings: Settings) -> Any:
    """Re-frame ``clip`` to the target ratio using a blurred background fill.

    When the clip already matches the target ratio (within a small tolerance)
    it is returned unchanged, so an already-vertical video is never needlessly
    re-scaled.
    """
    ratio = parse_aspect_ratio(settings.fit_aspect_ratio or settings.aspect_ratio)
    target_w, target_h = _target_size(ratio, settings.fit_height)
    src_w, src_h = clip.size

    if abs((src_w / float(src_h)) - ratio) <= _RATIO_TOLERANCE:
        return clip

    log.info(
        "fitting %dx%d into %dx%d (%s) with a blurred background",
        src_w,
        src_h,
        target_w,
        target_h,
        settings.fit_aspect_ratio or settings.aspect_ratio,
    )

    fg_w, fg_h = _contain_size(src_w, src_h, target_w, target_h)
    crop_w, crop_h = _cover_crop(src_w, src_h, ratio)
    blur = max(0, int(settings.background_blur))
    darken = max(0.0, min(1.0, float(settings.background_darken)))

    try:
        import cv2  # noqa: F401
    except Exception as exc:  # pragma: no cover - OpenCV is a hard dependency
        log.warning("OpenCV unavailable (%s); using the MoviePy vertical fit", exc)
        return _fit_vertical_moviepy(
            clip,
            src_w,
            src_h,
            target_w,
            target_h,
            fg_w,
            fg_h,
            crop_w,
            crop_h,
            blur,
            darken,
        )

    return _make_cv2_reframer(
        clip, src_w, src_h, target_w, target_h, fg_w, fg_h, crop_w, crop_h, blur, darken
    )


__all__ = ["build_vertical_clip", "needs_vertical_fit"]
