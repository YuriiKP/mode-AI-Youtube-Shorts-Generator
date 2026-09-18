"""Unified configuration — one ``.env`` drives the whole project.

Everything is configured through a single ``.env`` file:

* the source (a video file, a folder of videos, or a YouTube URL) — ``INPUT``;
* where results are written — ``OUTPUT_DIR``;
* background music, a file *or* a folder — ``MUSIC``;
* clipping (``NUM_CLIPS``, ``ASPECT_RATIO``, ...), the LLM, Whisper,
  subtitle appearance and video encoding.

Value precedence (highest first):

1. CLI flags (passed to :func:`load_settings` as ``extra=``)
2. OS environment variables
3. the file passed to ``--env``
4. ``.env`` in the current working directory
5. ``.env`` in the project root

Relative paths (input, output dir, music, fonts, ...) are resolved against the
current working directory, which keeps behaviour predictable no matter where the
tool is invoked from.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

VALID_SUBTITLE_POSITIONS = ("bottom", "top", "center", "custom")
VALID_SUBTITLE_SOURCES = ("auto", "file", "whisper", "none")
VALID_BANNER_POSITIONS = ("top", "bottom", "center")
# Appearance animation of a subtitle cue. An empty value (the default) disables
# the animation entirely.
VALID_SUBTITLE_ANIMATIONS = ("fade", "slide", "pop")

DEFAULT_TEXT_FORE_COLOR = "#FFFFFF"
DEFAULT_STROKE_COLOR = "#000000"
DEFAULT_BANNER_TEXT_COLOR = "#FFFFFF"
DEFAULT_BANNER_BACKGROUND_COLOR = "#000000"

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_ASSIGNMENT_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")

_TRUTHY = {"1", "true", "yes", "on", "y"}
_FALSY = {"0", "false", "no", "off", "n", "none", ""}


class ConfigError(ValueError):
    """Raised when the configuration contains an invalid value."""


# ---------------------------------------------------------------------------
# Minimal .env parsing (no python-dotenv dependency)
# ---------------------------------------------------------------------------


def _unquote(value: str) -> str:
    """Strip matching surrounding quotes and unescape common sequences."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        inner = value[1:-1]
        if value[0] == '"':
            inner = (
                inner.replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        return inner
    return value


def _strip_inline_comment(value: str) -> str:
    """Remove a trailing ``# comment`` from an unquoted value."""
    if not value or value[0] in ("'", '"'):
        return value
    # A comment must be preceded by whitespace to avoid cutting ``#RRGGBB``.
    match = re.search(r"\s#", value)
    if match:
        return value[: match.start()].rstrip()
    return value


def parse_env_file(path: str) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` file, ignoring blanks and ``#`` comments."""
    values: dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return values
    with open(path, "r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = _ASSIGNMENT_RE.match(line)
            if not match:
                continue
            key = match.group(1)
            value = _strip_inline_comment(match.group(2).strip())
            values[key] = _unquote(value)
    return values


# ---------------------------------------------------------------------------
# Typed getters
# ---------------------------------------------------------------------------


def _as_bool(value: Any, key: str, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    raise ConfigError(f"{key} must be a boolean value, got: {value!r}")


def _as_int(value: Any, key: str, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got: {value!r}") from exc


def _as_float(value: Any, key: str, default: float) -> float:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got: {value!r}") from exc


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _as_color(value: Any, key: str, default: str) -> str:
    text = _as_str(value, default)
    if _HEX_COLOR_RE.match(text):
        return text
    raise ConfigError(f"{key} must be a #RRGGBB color, got: {value!r}")


def _as_background_color(value: Any, key: str) -> Optional[str]:
    """Resolve the subtitle background.

    ``false``/empty disables the background, ``true`` uses black, and a
    ``#RRGGBB`` value uses that color.
    """
    text = _as_str(value, "")
    if text == "":
        return None
    lowered = text.lower()
    if lowered in _FALSY:
        return None
    if lowered in _TRUTHY:
        return "#000000"
    if _HEX_COLOR_RE.match(text):
        return text
    raise ConfigError(f"{key} must be false, true or a #RRGGBB color, got: {value!r}")


def _as_choice(value: Any, key: str, default: str, choices: tuple[str, ...]) -> str:
    text = _as_str(value, default).lower()
    if text not in choices:
        allowed = ", ".join(choices)
        raise ConfigError(f"{key} must be one of: {allowed}; got: {value!r}")
    return text


def _as_subtitle_animation(value: Any, key: str) -> str:
    """Resolve ``SUBTITLE_ANIMATION``.

    An empty value disables the animation. The common "off" spellings are also
    accepted for convenience; anything else must name a known animation.
    """
    text = _as_str(value, "").lower()
    if text in ("", "none", "off", "false", "no"):
        return ""
    if text not in VALID_SUBTITLE_ANIMATIONS:
        allowed = ", ".join(VALID_SUBTITLE_ANIMATIONS)
        raise ConfigError(
            f"{key} must be empty (no animation) or one of: {allowed}; got: {value!r}"
        )
    return text


def parse_aspect_ratio(value: Any, default: float = 9.0 / 16.0) -> float:
    """Parse ``9:16`` (or ``9x16``) into a ``width / height`` float.

    Returns ``default`` when the value is empty or cannot be parsed, so callers
    can safely fall back to a sensible ratio.
    """
    text = _as_str(value, "")
    for separator in (":", "x", "X"):
        if separator in text:
            width, _, height = text.partition(separator)
            try:
                ratio = float(width) / float(height)
            except (ValueError, ZeroDivisionError):
                return default
            return ratio if ratio > 0 else default
    return default


def _has_valid_ratio(value: Any) -> bool:
    """Return ``True`` when ``value`` is empty or a parseable ``W:H`` ratio."""
    text = _as_str(value, "")
    if not text:
        return True
    for separator in (":", "x", "X"):
        if separator in text:
            width, _, height = text.partition(separator)
            try:
                return float(width) > 0 and float(height) > 0
            except ValueError:
                return False
    return False


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """Fully resolved configuration for a single run."""

    # Source & output ------------------------------------------------------
    input: str = ""  # video file, folder of videos, or YouTube URL
    output_dir: str = "output"

    # Clipping -------------------------------------------------------------
    num_clips: int = 3
    aspect_ratio: str = "9:16"
    download_format: str = "720"
    face_tracking: bool = True
    # Границы хайлайта задаёт LLM и часто попадает в середину фразы. Если
    # включено, границы сначала привязываются к фразам транскрипта
    # (highlights.snap_highlights_to_transcript), затем добавляется запас,
    # чтобы не обрезать крайние слова. Паддинги = 0 отключают запас.
    clip_snap_to_transcript: bool = True
    clip_start_padding: float = 0.15
    clip_end_padding: float = 0.4

    # LLM (highlight ranking) ---------------------------------------------
    llm_provider: str = "openai"  # openai | deepseek | gemini
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Whisper (faster-whisper) ---------------------------------------------
    whisper_model: str = "base"
    whisper_device: str = "auto"  # auto | cpu | cuda
    whisper_language: str = ""
    # Silero VAD trims leading/trailing silence, which tightens the word
    # timings Whisper reports and reduces subtitles that lead or lag the voice.
    # Override with WHISPER_VAD_FILTER=false in .env to disable.
    whisper_vad_filter: bool = True

    # Background music -----------------------------------------------------
    music: str = ""  # a file or a folder
    music_volume: float = 0.2
    music_fade_out: float = 3.0

    # Subtitles ------------------------------------------------------------
    subtitle_source: str = "auto"  # auto | file | whisper | none
    subtitle_file: str = ""
    # A single ``FONT`` value — a font file, a font file name looked up in the
    # ``fonts/`` folder(s), or a folder of fonts to pick from (mirrors MUSIC).
    font: str = "STHeitiMedium.ttc"
    font_size: int = 60
    text_fore_color: str = DEFAULT_TEXT_FORE_COLOR
    text_background_color: Optional[str] = None
    rounded_subtitle_background: bool = False
    stroke_color: str = DEFAULT_STROKE_COLOR
    stroke_width: float = 1.5
    subtitle_position: str = "bottom"
    custom_position: float = 70.0
    # An appearance animation for each cue. Empty = disabled; otherwise
    # ``fade`` | ``slide`` | ``pop``. ``subtitle_animation_duration`` is how
    # long the entrance lasts, in seconds.
    subtitle_animation: str = ""
    subtitle_animation_duration: float = 0.25
    # Cue chunking: a single on-screen phrase is split again when it exceeds any
    # of these limits, so long sentences never hang over the whole screen.
    subtitle_max_chars: int = 40
    subtitle_max_words: int = 9
    subtitle_max_duration: float = 3.5
    # Silence (in seconds) between two words that forces a new cue when
    # word-level timings are available, so a subtitle ends at a natural pause in
    # speech — and the text clears from screen during the silence.
    subtitle_pause_threshold: float = 0.6
    # Shift every burned-in cue along the timeline, in seconds. Positive values
    # make the text appear later, which compensates for Whisper word timings
    # that tend to lead the actual speech by a fraction of a second; negative
    # values make it appear earlier. Only the on-screen burn-in is shifted —
    # the ``.srt`` cache keeps the raw transcript timings.
    subtitle_offset: float = 0.0

    # Vertical fit / blurred background ------------------------------------
    # Force the rendered clip into a vertical frame. Videos that do not already
    # match ``FIT_ASPECT_RATIO`` are centred and the empty area is filled with a
    # blurred, cover-scaled copy of the same video.
    fit_vertical: bool = True
    fit_aspect_ratio: str = "9:16"  # empty = use ASPECT_RATIO
    fit_height: int = 1920  # canvas height in pixels (width follows the ratio)
    background_blur: int = 30  # blurred-fill strength (0 = no blur)
    background_darken: float = 0.0  # 0..1 darkening applied to the blur

    # Banner overlay -------------------------------------------------------
    # A single ``BANNER`` value: when it points at an existing image file it is
    # drawn as an image, otherwise the value itself is drawn as a text band.
    # Empty = no banner.
    banner: str = ""
    banner_position: str = "top"  # top | bottom | center
    banner_width_ratio: float = 0.9  # image width as a fraction of the frame
    banner_opacity: float = 1.0  # 0..1 alpha for the image banner
    banner_margin: int = 16  # gap from the frame edge, in pixels
    banner_font_size: int = 48
    banner_text_color: str = DEFAULT_BANNER_TEXT_COLOR
    banner_background_color: str = DEFAULT_BANNER_BACKGROUND_COLOR

    # Colour / lens effects ------------------------------------------------
    # A saturation multiplier applied to the video before subtitles and the
    # banner are drawn: ``1.0`` keeps the source colours untouched, ``0``
    # renders greyscale and values above ``1`` boost colour.
    saturation: float = 1.0
    # Unsharp-mask amount for edge sharpening: ``0`` disables it, ``1.0`` is a
    # mild and ``2.0`` a fairly strong boost.
    sharpness: float = 0.0
    # Chromatic aberration strength, expressed as the approximate red/blue
    # channel separation in pixels at the corner of the frame; ``0`` disables it.
    chromatic_aberration: float = 0.0

    # Encoding -------------------------------------------------------------
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    fps: float = 0.0  # 0 = keep the source frame rate
    # x264/x265 speed/size trade-off. "medium" (the MoviePy default) is slow;
    # "fast"/"veryfast" encode several times quicker with little visual loss.
    preset: str = "fast"
    threads: int = 0  # 0 = let ffmpeg pick every available core
    ffmpeg_path: str = ""

    # Misc -----------------------------------------------------------------
    base_dir: str = field(default_factory=os.getcwd)

    def resolve(self, path: str) -> str:
        """Resolve a possibly-relative path against the working directory."""
        if not path:
            return ""
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded):
            return os.path.normpath(expanded)
        return os.path.normpath(os.path.join(self.base_dir, expanded))


_FIELD_NAMES = [f.name for f in fields(Settings)]

_STR_FIELDS = {
    "input",
    "output_dir",
    "aspect_ratio",
    "download_format",
    "llm_provider",
    "openai_api_key",
    "openai_model",
    "deepseek_api_key",
    "deepseek_model",
    "deepseek_base_url",
    "gemini_api_key",
    "gemini_model",
    "whisper_model",
    "whisper_device",
    "whisper_language",
    "music",
    "subtitle_file",
    "font",
    "fit_aspect_ratio",
    "banner",
    "video_codec",
    "audio_codec",
    "audio_bitrate",
    "preset",
    "ffmpeg_path",
}
_BOOL_FIELDS = {
    "face_tracking",
    "rounded_subtitle_background",
    "whisper_vad_filter",
    "fit_vertical",
    "clip_snap_to_transcript",
}
_INT_FIELDS = {
    "num_clips",
    "font_size",
    "threads",
    "fit_height",
    "background_blur",
    "banner_margin",
    "banner_font_size",
    "subtitle_max_chars",
    "subtitle_max_words",
}
_FLOAT_FIELDS = {
    "music_volume",
    "music_fade_out",
    "stroke_width",
    "custom_position",
    "fps",
    "background_darken",
    "banner_width_ratio",
    "banner_opacity",
    "subtitle_max_duration",
    "subtitle_pause_threshold",
    "subtitle_animation_duration",
    "subtitle_offset",
    "clip_start_padding",
    "clip_end_padding",
    "saturation",
    "sharpness",
    "chromatic_aberration",
}


def settings_field_names() -> tuple[str, ...]:
    """Public accessor for the configurable :class:`Settings` field names."""
    return tuple(_FIELD_NAMES)


def _coerce(field_name: str, raw: Any, current: Any) -> Any:
    """Convert a raw string into the correct type for ``field_name``."""
    key = field_name.upper()
    if field_name in _STR_FIELDS:
        return _as_str(raw, current)
    if field_name in _BOOL_FIELDS:
        return _as_bool(raw, key, bool(current))
    if field_name in _INT_FIELDS:
        return _as_int(raw, key, int(current))
    if field_name in _FLOAT_FIELDS:
        return _as_float(raw, key, float(current))
    if field_name == "text_fore_color":
        return _as_color(raw, key, DEFAULT_TEXT_FORE_COLOR)
    if field_name == "stroke_color":
        return _as_color(raw, key, DEFAULT_STROKE_COLOR)
    if field_name == "text_background_color":
        return _as_background_color(raw, key)
    if field_name == "banner_text_color":
        return _as_color(raw, key, DEFAULT_BANNER_TEXT_COLOR)
    if field_name == "banner_background_color":
        return _as_color(raw, key, DEFAULT_BANNER_BACKGROUND_COLOR)
    if field_name == "subtitle_position":
        return _as_choice(raw, key, str(current), VALID_SUBTITLE_POSITIONS)
    if field_name == "subtitle_animation":
        return _as_subtitle_animation(raw, key)
    if field_name == "subtitle_source":
        return _as_choice(raw, key, str(current), VALID_SUBTITLE_SOURCES)
    if field_name == "banner_position":
        return _as_choice(raw, key, str(current), VALID_BANNER_POSITIONS)
    return _as_str(raw, current)


def _project_root() -> str:
    """Absolute path to the project root (the directory that holds ``main.py``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_env_files(base_dir: str, env_file: Optional[str]) -> list[str]:
    """Return the ``.env`` files to merge, in increasing priority order."""
    candidates = [
        os.path.join(_project_root(), ".env"),
        os.path.join(base_dir, ".env"),
    ]
    if env_file:
        candidates.append(env_file)
    seen: list[str] = []
    for path in candidates:
        absolute = os.path.abspath(os.path.expanduser(path))
        if os.path.isfile(absolute) and absolute not in seen:
            seen.append(absolute)
    return seen


def load_settings(
    env_file: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    base_dir: Optional[str] = None,
) -> Settings:
    """Build a :class:`Settings` from ``.env`` files plus optional overrides.

    ``extra`` maps field names (or their upper-case ``ENV`` names) to raw values;
    typically the values collected from CLI arguments. Those values run through
    the same coercion and validation as ``.env`` values and win over everything
    else.
    """
    working_dir = os.path.abspath(base_dir or os.getcwd())
    settings = Settings(base_dir=working_dir)

    mapping: dict[str, str] = {}
    for path in _default_env_files(working_dir, env_file):
        # Later files win, so process in order and update.
        mapping.update(parse_env_file(path))

    # OS environment variables take precedence over file values.
    for name in _FIELD_NAMES:
        env_key = name.upper()
        if env_key in os.environ:
            mapping[env_key] = os.environ[env_key]

    # CLI-provided values win over everything else.
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            mapping[str(key).upper()] = value if isinstance(value, str) else str(value)

    for name in _FIELD_NAMES:
        if name == "base_dir":
            continue
        env_key = name.upper()
        if env_key in mapping:
            current = getattr(settings, name)
            setattr(settings, name, _coerce(name, mapping[env_key], current))

    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    """Sanity-check values that cannot be validated by type coercion alone."""
    if settings.num_clips <= 0:
        raise ConfigError("NUM_CLIPS must be a positive integer")
    if settings.clip_start_padding < 0 or settings.clip_end_padding < 0:
        raise ConfigError(
            "CLIP_START_PADDING and CLIP_END_PADDING must be zero or greater"
        )
    if settings.font_size <= 0:
        raise ConfigError("FONT_SIZE must be a positive integer")
    if settings.threads < 0:
        raise ConfigError("THREADS must be zero (auto) or a positive integer")
    if settings.music_volume < 0:
        raise ConfigError("MUSIC_VOLUME must be zero or greater")
    if settings.stroke_width < 0:
        raise ConfigError("STROKE_WIDTH must be zero or greater")
    if settings.subtitle_position == "custom" and not (
        0 <= settings.custom_position <= 100
    ):
        raise ConfigError("CUSTOM_POSITION must be between 0 and 100")
    if settings.subtitle_max_chars < 4:
        raise ConfigError("SUBTITLE_MAX_CHARS must be at least 4")
    if settings.subtitle_max_words < 1:
        raise ConfigError("SUBTITLE_MAX_WORDS must be at least 1")
    if settings.subtitle_max_duration <= 0:
        raise ConfigError("SUBTITLE_MAX_DURATION must be greater than 0")
    if settings.subtitle_animation_duration < 0:
        raise ConfigError("SUBTITLE_ANIMATION_DURATION must be zero or greater")
    if abs(settings.subtitle_offset) > 10:
        raise ConfigError(
            "SUBTITLE_OFFSET must be between -10 and 10 seconds "
            f"(got {settings.subtitle_offset})"
        )
    if settings.fps < 0:
        raise ConfigError("FPS must be zero (keep source) or a positive number")
    if settings.fit_height <= 0:
        raise ConfigError("FIT_HEIGHT must be a positive integer")
    if settings.background_blur < 0:
        raise ConfigError("BACKGROUND_BLUR must be zero or greater")
    if not 0.0 <= settings.background_darken <= 1.0:
        raise ConfigError("BACKGROUND_DARKEN must be between 0 and 1")
    if not _has_valid_ratio(settings.fit_aspect_ratio):
        raise ConfigError(
            "FIT_ASPECT_RATIO must look like 9:16 (width:height), "
            f"got: {settings.fit_aspect_ratio!r}"
        )
    if settings.banner_font_size <= 0:
        raise ConfigError("BANNER_FONT_SIZE must be a positive integer")
    if settings.banner_margin < 0:
        raise ConfigError("BANNER_MARGIN must be zero or greater")
    if not 0.0 < settings.banner_width_ratio <= 1.0:
        raise ConfigError("BANNER_WIDTH_RATIO must be between 0 and 1")
    if not 0.0 <= settings.banner_opacity <= 1.0:
        raise ConfigError("BANNER_OPACITY must be between 0 and 1")
    if settings.saturation < 0:
        raise ConfigError("SATURATION must be zero or greater")
    if settings.sharpness < 0:
        raise ConfigError("SHARPNESS must be zero or greater")
    if settings.chromatic_aberration < 0:
        raise ConfigError("CHROMATIC_ABERRATION must be zero or greater")


# ---------------------------------------------------------------------------
# LLM key helpers
# ---------------------------------------------------------------------------


def require_openai_key(settings: Settings) -> str:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. It is required for highlight ranking "
            "when LLM_PROVIDER=openai. Add it to your .env."
        )
    return settings.openai_api_key


def require_deepseek_key(settings: Settings) -> str:
    if not settings.deepseek_api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. It is required when "
            "LLM_PROVIDER=deepseek. Add it to your .env or switch LLM_PROVIDER."
        )
    return settings.deepseek_api_key


def require_gemini_key(settings: Settings) -> str:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. It is required when "
            "LLM_PROVIDER=gemini. Add it to your .env or switch LLM_PROVIDER."
        )
    return settings.gemini_api_key
