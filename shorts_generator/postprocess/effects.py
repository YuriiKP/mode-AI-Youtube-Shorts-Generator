"""Configurable colour / lens effects applied to the rendered video frame.

Three independent, optional effects are exposed through the ``.env`` and are all
implemented as a single NumPy/OpenCV per-frame pass (the same technique the
vertical re-framer uses, so it slots into the existing pipeline and costs only
one extra pass over the frame):

* ``SATURATION`` — a multiplier on colour saturation. ``1.0`` keeps the source
  colours untouched, ``0`` renders greyscale and values above ``1`` boost
  colour.
* ``SHARPNESS`` — an unsharp-mask amount for edge enhancement. ``0`` disables
  it, ``1.0`` is a mild and ``2.0`` a fairly strong boost.
* ``CHROMATIC_ABERRATION`` — the red and blue channels are scaled in opposite
  directions, so colours fringe towards the corners like a real lens. The value
  is the approximate channel separation, in pixels, at the corner of the frame;
  ``0`` disables it.

The effects are applied to the video *before* subtitles and the banner are drawn,
so the text overlays stay crisp. When all three are at their neutral value the
clip is returned untouched and not a single frame is processed.
"""

from __future__ import annotations

import math
from typing import Any

from ..config import Settings
from .log import log

# Rec. 601 luma weights (the classic greyscale filter uses the same ones). The
# channel order matches MoviePy's frames, which are RGB.
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)

# Standard deviation of the Gaussian used by the unsharp mask. A small sigma
# sharpens fine detail without producing the wide halos a large sigma would.
_SHARPEN_SIGMA = 1.5

# Values closer than this to 1.0 are treated as "no saturation change" so a
# default of ``1.0`` never triggers the effect.
_SATURATION_EPSILON = 1e-6


def _saturation_changes_picture(value: float) -> bool:
    """Whether a saturation multiplier would actually alter the frame."""
    return abs(value - 1.0) > _SATURATION_EPSILON


def effects_enabled(settings: Settings) -> bool:
    """Return ``True`` when at least one effect would change the picture."""
    return (
        _saturation_changes_picture(float(settings.saturation))
        or float(settings.sharpness) > 0.0
        or float(settings.chromatic_aberration) > 0.0
    )


def _to_uint8(frame: Any) -> tuple[Any, Any]:
    """Coerce a frame to ``uint8`` for the OpenCV-backed effects.

    Real MoviePy video frames are already ``uint8`` RGB and are returned
    untouched. Anything else — a tiny integer or float test clip, for instance —
    is scaled into the 0..255 range first; the original dtype is returned
    alongside so :func:`_restore_dtype` can cast the processed frame back.
    """
    import numpy as np

    if frame.dtype == np.uint8:
        return frame, None
    data = frame.astype(np.float32)
    # Floats are often normalised to 0..1; bring them into the uint8 range first.
    if frame.dtype.kind == "f" and (data.size == 0 or float(data.max()) <= 1.0):
        data = data * 255.0
    return np.clip(data, 0.0, 255.0).astype(np.uint8), frame.dtype


def _restore_dtype(frame: Any, original_dtype: Any) -> Any:
    """Cast a processed frame back to ``original_dtype`` (``None`` = unchanged)."""
    if original_dtype is None:
        return frame
    return frame.astype(original_dtype)


def _apply_saturation(frame: Any, factor: float) -> Any:
    """Move every pixel towards (or away from) its luma.

    ``factor`` scales the colour difference from the greyscale value: ``1.0``
    leaves the frame alone, ``0`` collapses it to greyscale and values above
    ``1`` extrapolate the colour outwards so it becomes more vivid. This is a
    single vectorised NumPy pass — no HSV round-trip, so hues never shift.
    """
    import numpy as np

    weights = np.array(_LUMA_WEIGHTS, dtype=np.float32)
    data = frame.astype(np.float32)
    luma = (data @ weights)[..., None]
    return np.clip(luma + (data - luma) * factor, 0.0, 255.0).astype(frame.dtype)


def _apply_sharpness(frame: Any, amount: float) -> Any:
    """Sharpen the frame with an unsharp mask.

    A lightly blurred copy is subtracted from the frame, and ``amount`` times
    that difference is added back — the standard unsharp-mask trick. Larger
    ``amount`` values push the edges further while the clamp keeps the result in
    range.
    """
    import cv2
    import numpy as np

    work, original_dtype = _to_uint8(frame)
    blurred = cv2.GaussianBlur(work, (0, 0), sigmaX=_SHARPEN_SIGMA)
    data = work.astype(np.float32)
    sharpened = data + (data - blurred.astype(np.float32)) * amount
    result = np.clip(sharpened, 0.0, 255.0).astype(np.uint8)
    return _restore_dtype(result, original_dtype)


def _scale_channel(channel: Any, scale: float) -> Any:
    """Scale a single-channel image about its centre by ``scale``.

    ``scale`` slightly above ``1`` pushes pixels outwards from the centre and
    below ``1`` pulls them inwards, which is exactly the radial displacement a
    chromatic aberration needs. Edge pixels are replicated so the corners do not
    smear to black.
    """
    import cv2

    height, width = channel.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 0.0, scale)
    return cv2.warpAffine(
        channel,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _apply_chromatic_aberration(frame: Any, pixels: float) -> Any:
    """Fringe the colours by scaling the red and blue channels apart.

    ``pixels`` is the approximate red/blue separation at the corner of the
    frame: the red channel is scaled outwards and the blue inwards by that many
    pixels there (the shift tapers off to zero at the centre). The green
    channel — which carries most of the perceived detail — is left untouched.
    """
    import numpy as np

    work, original_dtype = _to_uint8(frame)
    height, width = work.shape[:2]
    # Half the frame diagonal, i.e. the distance from the centre to a corner.
    radius = max(1.0, math.hypot(width, height) / 2.0)
    delta = pixels / radius

    red = _scale_channel(work[..., 0], 1.0 + delta)
    green = work[..., 1]
    blue = _scale_channel(work[..., 2], 1.0 - delta)
    result = np.stack((red, green, blue), axis=-1)
    return _restore_dtype(result, original_dtype)


def apply_effects(clip: Any, settings: Settings) -> Any:
    """Return ``clip`` with the configured colour/lens effects applied.

    The effects are chained as a single :meth:`~moviepy.VideoClip.image_transform`
    pass so the encode stays one re-encode. When every effect is at its neutral
    value the clip is returned unchanged and no frames are processed.
    """
    saturation = float(settings.saturation)
    sharpness = max(0.0, float(settings.sharpness))
    aberration = max(0.0, float(settings.chromatic_aberration))

    if not effects_enabled(settings):
        return clip

    # Saturation is pure NumPy, but sharpening and the aberration need OpenCV.
    # If it is missing, keep whatever still works instead of dropping everything.
    if sharpness > 0.0 or aberration > 0.0:
        try:
            import cv2  # noqa: F401
        except Exception as exc:  # pragma: no cover - OpenCV is a hard dependency
            log.warning(
                "OpenCV unavailable (%s); skipping sharpness and chromatic aberration",
                exc,
            )
            sharpness = 0.0
            aberration = 0.0

    if not (
        _saturation_changes_picture(saturation) or sharpness > 0.0 or aberration > 0.0
    ):
        return clip

    log.info(
        "applying video effects (saturation=%.2f, sharpness=%.2f, "
        "chromatic_aberration=%.1fpx)",
        saturation,
        sharpness,
        aberration,
    )

    def transform(frame: Any) -> Any:
        # OpenCV needs an 8-bit buffer; real video frames already are uint8, but
        # cast (and later restore) so odd dtypes from other readers still work.
        work, original_dtype = _to_uint8(frame)
        result = work
        if _saturation_changes_picture(saturation):
            result = _apply_saturation(result, saturation)
        if sharpness > 0.0:
            result = _apply_sharpness(result, sharpness)
        if aberration > 0.0:
            result = _apply_chromatic_aberration(result, aberration)
        return _restore_dtype(result, original_dtype)

    return clip.image_transform(transform)


__all__: list[str] = ["apply_effects", "effects_enabled"]
