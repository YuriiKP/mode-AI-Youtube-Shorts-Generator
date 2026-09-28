"""Configurable colour / lens effects, compiled into an FFmpeg filtergraph.

Several independent, optional effects are exposed through the ``.env``:

* ``SATURATION`` — a multiplier on colour saturation. ``1.0`` keeps the source
  colours untouched, ``0`` renders greyscale and values above ``1`` boost
  colour.
* ``SHARPNESS`` — an unsharp-mask amount for edge enhancement. ``0`` disables
  it, ``1.0`` is a mild and ``2.0`` a fairly strong boost.
* ``CHROMATIC_ABERRATION`` — the red and blue channels are shifted in opposite
  directions, so colours fringe at the edges of the frame. The value is the
  channel separation, in pixels; ``0`` disables it.
* ``SPEED`` — a playback-speed multiplier applied to the video *and* its audio.
  ``1.0`` keeps the original pace, values above ``1`` play the clip faster
  (``1.5`` is 50% faster) and values below ``1`` slow it down (``0.5`` is half
  speed). Unlike the colour effects it re-times the audio too, so it drives an
  ``-af`` chain alongside the video filtergraph.

Where the effects run matters. Processing every frame in Python (NumPy/OpenCV
through ``image_transform``) is **single-threaded**: it holds one core while the
rest of the machine sits idle and starves the encoder, which is by far the
slowest part of a render. So instead of touching frames here, the effects are
compiled into an FFmpeg filtergraph string and applied in a dedicated pre-pass
over the source video (see
:func:`shorts_generator.postprocess.pipeline._apply_effects_pass`). FFmpeg then
runs the very same filters in optimised, multi-threaded C — on every core — so
enabling an effect costs a few milliseconds per frame instead of tens of them,
and the per-frame Python cost is gone entirely.

The graph uses only widely available filters (``eq`` / ``unsharp`` /
``rgbashift``, plus ``setpts`` / ``atempo`` for ``SPEED``), so it works with any
standard FFmpeg build. Because the pre-pass rewrites the source *before*
anything is composited on top, the effects land on the video only — the
subtitles and the banner are drawn afterwards and stay crisp. When the colour
values are neutral and ``SPEED`` is ``1`` the graph is empty and the pre-pass is
skipped entirely.
"""

from __future__ import annotations

from ..config import Settings

# Values closer than this to 1.0 count as "no saturation change", so the default
# of ``1.0`` never contributes a filter.
_SATURATION_EPSILON = 1e-6

# Values closer than this to 1.0 count as "no speed change", so the default of
# ``1.0`` never contributes a filter.
_SPEED_EPSILON = 1e-6

# ``atempo`` only accepts a factor in this range per instance; a speed outside
# it is built by chaining several ``atempo`` filters.
_ATEMPO_MIN = 0.5
_ATEMPO_MAX = 2.0


def _fmt(value: float) -> str:
    """Format a number for an FFmpeg option, without a needless trailing ``.0``.

    ``1.0`` becomes ``"1"`` and ``1.25`` stays ``"1.25"`` — both of which FFmpeg
    parses identically, but the shorter form keeps logged filtergraphs readable.
    """
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def build_filter_chain(settings: Settings) -> str:
    """Compile the configured effects into one FFmpeg filtergraph string.

    Returns ``""`` when every effect is at its neutral value; callers must then
    not add a video filter at all. Otherwise the result is a comma-separated
    chain (e.g. ``"eq=saturation=1.2,unsharp=5:5:1:5:5:0,rgbashift=rh=3:bh=-3"``)
    ready to hand to FFmpeg via ``-vf``.
    """
    filters: list[str] = []

    saturation = float(settings.saturation)
    if abs(saturation - 1.0) > _SATURATION_EPSILON:
        # ``eq``'s saturation option uses the same convention as ours: 1.0 keeps
        # the colours, 0 gives greyscale. Other eq knobs stay at their defaults.
        filters.append(f"eq=saturation={_fmt(saturation)}")

    sharpness = max(0.0, float(settings.sharpness))
    if sharpness > 0.0:
        # A 5x5 unsharp mask applied to luma only (chroma amount 0) so sharpening
        # never introduces colour halos around edges.
        filters.append(f"unsharp=5:5:{_fmt(sharpness)}:5:5:0")

    aberration = max(0.0, float(settings.chromatic_aberration))
    if aberration > 0.0:
        # Shift the red channel one way and the blue the other by roughly this
        # many pixels; at least one pixel, so a tiny config still has an effect.
        shift = max(1, round(aberration))
        filters.append(f"rgbashift=rh={shift}:bh=-{shift}")

    return ",".join(filters)


def effects_enabled(settings: Settings) -> bool:
    """Return ``True`` when at least one effect would change the picture."""
    return bool(build_filter_chain(settings))


def _build_atempo_chain(speed: float) -> str:
    """Return an ``atempo`` chain that scales the audio by ``speed``.

    ``atempo`` only covers ``0.5``–``2.0`` per instance, so a speed outside that
    range (e.g. ``3.0`` or ``0.25``) is split across several chained filters,
    each kept inside the supported range.
    """
    parts: list[str] = []
    remaining = float(speed)
    while remaining > _ATEMPO_MAX + _SPEED_EPSILON:
        parts.append(f"atempo={_fmt(_ATEMPO_MAX)}")
        remaining /= _ATEMPO_MAX
    while remaining < _ATEMPO_MIN - _SPEED_EPSILON:
        parts.append(f"atempo={_fmt(_ATEMPO_MIN)}")
        remaining /= _ATEMPO_MIN
    parts.append(f"atempo={_fmt(remaining)}")
    return ",".join(parts)


def build_speed_video_filter(settings: Settings) -> str:
    """Return the video filter that scales playback speed, or ``""``.

    ``setpts=PTS/SPEED`` compresses the frame timestamps for a speed above
    ``1`` (the clip plays faster) and stretches them below it; at ``1.0`` there
    is nothing to do and an empty string is returned.
    """
    speed = float(settings.speed)
    if abs(speed - 1.0) <= _SPEED_EPSILON:
        return ""
    return f"setpts=PTS/{_fmt(speed)}"


def build_speed_audio_filter(settings: Settings) -> str:
    """Return the audio filter that scales playback speed, or ``""``.

    Uses an ``atempo`` chain so the audio is re-timed by the same factor as the
    video and never slips out of sync.
    """
    speed = float(settings.speed)
    if abs(speed - 1.0) <= _SPEED_EPSILON:
        return ""
    return _build_atempo_chain(speed)


def speed_enabled(settings: Settings) -> bool:
    """Return ``True`` when the playback speed differs from ``1.0``."""
    return bool(build_speed_video_filter(settings))


__all__: list[str] = [
    "build_filter_chain",
    "build_speed_audio_filter",
    "build_speed_video_filter",
    "effects_enabled",
    "speed_enabled",
]
