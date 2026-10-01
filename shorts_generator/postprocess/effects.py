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

A second family of effects exists purely to *uniquify* the clip so the platforms
do not treat it as a copy of an already uploaded video (``UNIQUE_*``). They are
tiny, near-invisible edits that move the file hash, the perceptual (frame) hash
and the audio fingerprint:

* ``UNIQUE_MIRROR`` — a horizontal flip (``hflip``); the strongest frame-hash
  mover, but it mirrors any text baked into the source, so it is opt-in.
* ``UNIQUE_CROP`` — trims a few pixels off every edge and stretches the frame
  back to its original size (a subtle "punch-in"); geometry is the most
  effective lever against perceptual hashing.
* ``UNIQUE_NOISE`` — temporal grain (``noise``): a fresh random pattern on every
  frame, which disrupts the pixel statistics the perceptual hash is built from.
* ``UNIQUE_BRIGHTNESS`` / ``UNIQUE_CONTRAST`` / ``UNIQUE_GAMMA`` / ``UNIQUE_HUE``
  — small tone and hue shifts (folded into the same ``eq`` / ``hue`` filters as
  ``SATURATION``).
* ``UNIQUE_PITCH`` — a micro pitch shift (``asetrate`` + ``atempo``) that moves
  the audio fingerprint without changing the clip's duration; ``UNIQUE_LOUDNESS``
  and ``UNIQUE_GAIN`` reshape the loudness.
* ``UNIQUE_RANDOMIZE`` — jitters the *numeric* parameters (crop, noise,
  brightness, contrast, gamma, hue, pitch) a little on every render, so each
  exported file is a distinct variant of the same edit; the boolean switches
  (``UNIQUE_MIRROR`` / ``UNIQUE_LOUDNESS`` / ``UNIQUE_METADATA``) are left as
  configured.

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
``rgbashift`` / ``hue`` / ``noise`` / ``crop`` / ``scale`` / ``hflip``, plus
``setpts`` / ``atempo`` / ``asetrate`` / ``loudnorm`` / ``volume`` for re-timing
and the audio fingerprint), so it works with any standard FFmpeg build. Because
the pre-pass rewrites the source *before* anything is composited on top, the
effects land on the video only — the subtitles and the banner are drawn
afterwards and stay crisp. When every value is neutral and ``SPEED`` is ``1``
the graph is empty and the pre-pass is skipped entirely.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..config import Settings

# Values closer than this to 1.0 count as "no saturation change", so the default
# of ``1.0`` never contributes a filter.
_SATURATION_EPSILON = 1e-6

# Values closer than this to 1.0 count as "no speed change", so the default of
# ``1.0`` never contributes a filter.
_SPEED_EPSILON = 1e-6

# A number closer than this to its neutral value (0 for brightness/hue/noise,
# 1.0 for contrast/gamma) contributes no filter.
_NEUTRAL_EPSILON = 1e-6

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


@dataclass(frozen=True)
class Uniqueness:
    """The resolved anti-duplicate parameters for a single render.

    ``resolve_uniqueness`` turns the ``UNIQUE_*`` settings into one of these,
    optionally jittering every value so each render becomes a distinct variant.
    """

    mirror: bool = False
    crop: int = 0
    noise: float = 0.0
    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    hue: float = 0.0
    pitch: float = 0.0
    loudness: bool = False
    gain: float = 0.0
    metadata: bool = True


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


def _jitter(base: float, amplitude: float, spread: float) -> float:
    """Return ``base`` moved by up to ``amplitude``, scaled by ``spread``.

    Used by ``UNIQUE_RANDOMIZE`` to give every render a slightly different set
    of values, so copies of the same edit do not share a fingerprint.
    """
    if spread <= 0.0:
        return base
    return base + random.uniform(-amplitude, amplitude) * spread


def resolve_uniqueness(settings: Settings) -> Uniqueness:
    """Turn the ``UNIQUE_*`` settings into a concrete :class:`Uniqueness`.

    When ``UNIQUE_RANDOMIZE`` is on, every numeric parameter is jittered around
    its configured value (or a small sensible default when it is neutral), so
    repeated renders of the same clip come out as different files. The boolean
    switches (``mirror`` / ``loudness`` / ``metadata``) are never randomised:
    they always follow the configuration.
    """
    mirror = bool(settings.unique_mirror)
    crop = max(0, int(settings.unique_crop))
    noise = max(0.0, float(settings.unique_noise))
    brightness = float(settings.unique_brightness)
    contrast = max(0.0, float(settings.unique_contrast))
    gamma = max(0.01, float(settings.unique_gamma))
    hue = float(settings.unique_hue)
    pitch = float(settings.unique_pitch)
    loudness = bool(settings.unique_loudness)
    gain = float(settings.unique_gain)

    if settings.unique_randomize:
        spread = _clamp(float(settings.unique_jitter), 0.0, 1.0)
        crop = int(round(_clamp(_jitter(float(crop or 4), 4.0, spread), 0, 12)))
        noise = _clamp(_jitter(noise or 3.0, 3.0, spread), 0.0, 10.0)
        brightness = _clamp(_jitter(brightness, 0.02, spread), -0.1, 0.1)
        contrast = _clamp(_jitter(contrast, 0.03, spread), 0.8, 1.2)
        gamma = _clamp(_jitter(gamma, 0.03, spread), 0.8, 1.2)
        hue = _clamp(_jitter(hue, 3.0, spread), -8.0, 8.0)
        pitch = _clamp(_jitter(pitch or 0.5, 0.5, spread), -2.0, 2.0)

    return Uniqueness(
        mirror=mirror,
        crop=crop,
        noise=noise,
        brightness=brightness,
        contrast=contrast,
        gamma=gamma,
        hue=hue,
        pitch=pitch,
        loudness=loudness,
        gain=gain,
        metadata=bool(settings.unique_metadata),
    )


def build_filter_chain(
    settings: Settings, uniqueness: Optional[Uniqueness] = None
) -> str:
    """Compile the configured effects into one FFmpeg filtergraph string.

    Returns ``""`` when every effect is at its neutral value; callers must then
    not add a video filter at all. Otherwise the result is a comma-separated
    chain (e.g. ``"crop=iw-8:ih-8:4:4,scale=iw+8:ih+8,eq=saturation=1.2,
    unsharp=5:5:1:5:5:0,noise=alls=3:allf=t"``) ready to hand to FFmpeg via
    ``-vf``.

    ``uniqueness`` lets the caller share one resolved set of anti-duplicate
    parameters between the video and audio graphs; when omitted they are derived
    from ``settings`` here.
    """
    u = uniqueness if uniqueness is not None else resolve_uniqueness(settings)
    filters: list[str] = []

    # --- geometry (the strongest perceptual-hash movers) ------------------
    if u.mirror:
        # A horizontal flip moves every pixel to a new place, so it shifts the
        # perceptual hash the most; it also mirrors text baked into the source,
        # which is why it stays opt-in.
        filters.append("hflip")

    crop_px = max(0, int(u.crop))
    if crop_px > 0:
        # Trim ``crop_px`` pixels off every edge, then stretch the frame back to
        # its original size: a subtle "punch-in" that shifts the whole picture.
        filters.append(f"crop=iw-{2 * crop_px}:ih-{2 * crop_px}:{crop_px}:{crop_px}")
        filters.append(f"scale=iw+{2 * crop_px}:ih+{2 * crop_px}")

    # --- tone / colour ----------------------------------------------------
    # ``eq`` carries contrast, brightness, saturation and gamma in a single
    # filter; only the knobs that differ from their default are emitted.
    eq_parts: list[str] = []
    contrast = float(u.contrast)
    if abs(contrast - 1.0) > _NEUTRAL_EPSILON:
        eq_parts.append(f"contrast={_fmt(contrast)}")
    brightness = float(u.brightness)
    if abs(brightness) > _NEUTRAL_EPSILON:
        eq_parts.append(f"brightness={_fmt(brightness)}")
    saturation = float(settings.saturation)
    if abs(saturation - 1.0) > _SATURATION_EPSILON:
        # ``eq``'s saturation option uses the same convention as ours: 1.0 keeps
        # the colours, 0 gives greyscale.
        eq_parts.append(f"saturation={_fmt(saturation)}")
    gamma = float(u.gamma)
    if abs(gamma - 1.0) > _NEUTRAL_EPSILON:
        eq_parts.append(f"gamma={_fmt(gamma)}")
    if eq_parts:
        filters.append("eq=" + ":".join(eq_parts))

    hue = float(u.hue)
    if abs(hue) > _NEUTRAL_EPSILON:
        filters.append(f"hue=h={_fmt(hue)}")

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

    noise = max(0.0, float(u.noise))
    if noise > _NEUTRAL_EPSILON:
        # Temporal grain: a different random pattern on every frame, which
        # disrupts the pixel statistics a perceptual hash is built from. Applied
        # last so the grain is never smoothed away by an earlier filter.
        filters.append(f"noise=alls={int(round(noise))}:allf=t")

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


def build_audio_filter(
    settings: Settings, uniqueness: Optional[Uniqueness] = None
) -> str:
    """Return the audio filtergraph for speed and uniqueness, or ``""``.

    Combines, in order: a micro pitch shift (which moves the audio fingerprint
    without changing the clip's duration), the ``SPEED`` ``atempo`` chain,
    loudness normalisation and a gain nudge. Returns ``""`` when nothing changes
    the audio, so the caller can copy that stream losslessly.
    """
    u = uniqueness if uniqueness is not None else resolve_uniqueness(settings)
    parts: list[str] = []

    pitch = float(u.pitch)
    if abs(pitch) > _NEUTRAL_EPSILON:
        # ``asetrate`` reinterprets the samples at a new rate (shifting pitch
        # *and* speed); ``atempo`` restores the original duration, leaving only
        # the pitch change. The leading ``aresample`` pins the rate so the
        # ``asetrate`` value is well defined whatever the source rate is.
        factor = 1.0 + pitch / 100.0
        parts += [
            "aresample=44100",
            f"asetrate={_fmt(round(44100 * factor))}",
            "aresample=44100",
            f"atempo={_fmt(1.0 / factor)}",
        ]

    speed = float(settings.speed)
    if abs(speed - 1.0) > _SPEED_EPSILON:
        parts.append(_build_atempo_chain(speed))

    if u.loudness:
        # Rewrites the waveform to a platform-standard loudness and, as a bonus,
        # matches the -14 LUFS target the platforms normalise to.
        parts.append("loudnorm=I=-14:TP=-1.5:LRA=11")

    gain = float(u.gain)
    if abs(gain) > _NEUTRAL_EPSILON:
        parts.append(f"volume={_fmt(gain)}dB")

    return ",".join(parts)


def speed_enabled(settings: Settings) -> bool:
    """Return ``True`` when the playback speed differs from ``1.0``."""
    return bool(build_speed_video_filter(settings))


__all__: list[str] = [
    "Uniqueness",
    "build_audio_filter",
    "build_filter_chain",
    "build_speed_audio_filter",
    "build_speed_video_filter",
    "effects_enabled",
    "resolve_uniqueness",
    "speed_enabled",
]
