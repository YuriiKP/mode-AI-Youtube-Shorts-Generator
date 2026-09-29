"""Publisher configuration — read from the project's ``.env`` / environment.

This module deliberately reuses the project's own minimal ``.env`` parser
(:func:`shorts_generator.config.parse_env_file`) so that there is exactly one
place in the codebase that knows how to read configuration files. Only the
``PUBLISH_*`` keys are consumed here; the rest of the file (pipeline settings
such as ``INPUT`` or ``OUTPUT_DIR``) is ignored.

Value precedence (highest first), matching the rest of the project:

1. values passed explicitly to :func:`load_publish_config` via ``extra``
2. OS environment variables
3. the file passed to ``env_file``
4. ``.env`` in the current working directory
5. ``.env`` in the project root

Recognised keys::

    PUBLISH_PROFILE_DIR=browser_profiles
    PUBLISH_DB=publish_state.sqlite3
    PUBLISH_OUTPUT_DIR=output
    PUBLISH_HEADLESS=false              # keep the window visible for uploads
    PUBLISH_PROXY=                      # http://127.0.0.1:7890 (needed where YT/TikTok are blocked)
    PUBLISH_TAGS=аниме,anime,short      # default hashtags when a clip has none
    PUBLISH_VISIBILITY=public
    PUBLISH_YT_PLAYLIST=                # optional playlist for YouTube uploads
    PUBLISH_DELAY=30                    # seconds to wait between uploads

How clips are spread over profiles::

    PUBLISH_MODE=distribute             # distribute | schedule (default: distribute)
    PUBLISH_SCHEDULE=                   # ISO datetimes, comma separated (schedule mode)

``PUBLISH_MODE``:

* ``distribute`` — clips are handed out round-robin, one per profile per round,
  and published immediately;
* ``schedule`` — same round-robin handout, but every clip is *scheduled* for
  publication; the number of ``PUBLISH_SCHEDULE`` entries is how many clips each
  profile publishes (entry *r* is the time of that profile's *r*-th clip).

Browser automation runs on the ShardX anti-detect engine (through its Python SDK),
which adds these keys::

    PUBLISH_SHARDX_TEMPLATE=            # library template id (e.g. win-rtx4060); "" -> random
    PUBLISH_SHARDX_PLATFORM=Windows     # Windows | macOS | Linux (used when the template is empty)
    PUBLISH_SHARDX_CACHE_DIR=           # SDK cache root; "" -> the SDK default
    PUBLISH_SHARDX_SCREEN_MODE=         # profile | cap_to_host | use_host; "" -> auto
    PUBLISH_SHARDX_RANDOMIZE=false      # re-randomize CPU/RAM/platform_version before launch
    PUBLISH_SHARDX_NOISE=               # canvas,webgl,audio,client_rects,sensors,fonts
    PUBLISH_SHARDX_WEBRTC=auto          # auto | tcp_only | block
    PUBLISH_SHARDX_LANGUAGE=en-US       # en-US | ru-RU | "" (auto: the SDK derives it from geo)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Mapping, Optional, Sequence

from shorts_generator.config import ConfigError, parse_env_file

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

VALID_VISIBILITIES = ("public", "unlisted", "private")

# How clips are handed out to profiles (see the module docstring).
VALID_PUBLISH_MODES = ("distribute", "schedule")

# ShardX-specific choices (see ``publisher.browser.shardx_backend``). The
# platform names are case-sensitive on purpose — they are passed straight to the
# SDK, which expects ``Windows`` / ``macOS`` / ``Linux``.
VALID_SHARDX_PLATFORMS = ("Windows", "macOS", "Linux")
VALID_SHARDX_SCREEN_MODES = ("profile", "cap_to_host", "use_host")
VALID_SHARDX_WEBRTC = ("auto", "tcp_only", "block")
VALID_SHARDX_NOISE = ("canvas", "webgl", "audio", "client_rects", "sensors", "fonts")

_TRUTHY = {"1", "true", "yes", "on", "y"}
_FALSY = {"0", "false", "no", "off", "n", "none", ""}

# field name -> environment key (the keys are intentionally explicit because a
# couple of them do not follow the plain ``PUBLISH_<FIELD>`` rule).
_FIELD_ENV = {
    "profile_dir": "PUBLISH_PROFILE_DIR",
    "db_path": "PUBLISH_DB",
    "output_dir": "PUBLISH_OUTPUT_DIR",
    "headless": "PUBLISH_HEADLESS",
    "proxy": "PUBLISH_PROXY",
    "tags": "PUBLISH_TAGS",
    "visibility": "PUBLISH_VISIBILITY",
    "playlist": "PUBLISH_YT_PLAYLIST",
    "delay": "PUBLISH_DELAY",
    "mode": "PUBLISH_MODE",
    "schedule": "PUBLISH_SCHEDULE",
    # ShardX backend.
    "shardx_template": "PUBLISH_SHARDX_TEMPLATE",
    "shardx_platform": "PUBLISH_SHARDX_PLATFORM",
    "shardx_cache_dir": "PUBLISH_SHARDX_CACHE_DIR",
    "shardx_screen_mode": "PUBLISH_SHARDX_SCREEN_MODE",
    "shardx_randomize": "PUBLISH_SHARDX_RANDOMIZE",
    "shardx_noise": "PUBLISH_SHARDX_NOISE",
    "shardx_webrtc": "PUBLISH_SHARDX_WEBRTC",
    "shardx_language": "PUBLISH_SHARDX_LANGUAGE",
}

_BOOL_FIELDS = {"headless", "shardx_randomize"}
_FLOAT_FIELDS = {"delay"}
_LIST_FIELDS = {"tags", "shardx_noise"}
_STR_FIELDS = {
    "profile_dir",
    "db_path",
    "output_dir",
    "proxy",
    "shardx_template",
    "shardx_cache_dir",
    "shardx_language",
}

# field name -> (default, allowed values) for the plain choice fields.
_CHOICE_FIELDS = {
    "visibility": ("public", VALID_VISIBILITIES),
    "shardx_webrtc": ("auto", VALID_SHARDX_WEBRTC),
    "mode": ("distribute", VALID_PUBLISH_MODES),
}

# field name -> allowed values, where an empty string means "auto / unset".
_OPTIONAL_CHOICE_FIELDS = {
    "shardx_screen_mode": VALID_SHARDX_SCREEN_MODES,
}

# Accepted spellings for PUBLISH_SHARDX_PLATFORM -> the canonical SDK value.
_SHARDX_PLATFORM_ALIASES = {
    "windows": "Windows",
    "win": "Windows",
    "win32": "Windows",
    "win64": "Windows",
    "macos": "macOS",
    "mac": "macOS",
    "osx": "macOS",
    "darwin": "macOS",
    "linux": "Linux",
}


# ---------------------------------------------------------------------------
# Typed getters (kept local so this module has no private imports elsewhere)
# ---------------------------------------------------------------------------


def _as_bool(value: Any, key: str, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    raise ConfigError(f"{key} must be a boolean value, got: {value!r}")


def _as_float(value: Any, key: str, default: float = 0.0) -> float:
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


def _as_choice(value: Any, key: str, default: str, choices: Sequence[str]) -> str:
    text = _as_str(value, default).lower()
    if text not in choices:
        allowed = ", ".join(choices)
        raise ConfigError(f"{key} must be one of: {allowed}; got: {value!r}")
    return text


def _as_optional_choice(value: Any, key: str, choices: Sequence[str]) -> str:
    """Like :func:`_as_choice`, but an empty value means "leave it to the SDK"."""
    text = _as_str(value, "").lower()
    if text == "":
        return ""
    if text not in choices:
        allowed = ", ".join(choices)
        raise ConfigError(f"{key} must be empty or one of: {allowed}; got: {value!r}")
    return text


def _as_platform(value: Any, key: str) -> str:
    """Normalise a ShardX platform name, accepting a few common aliases."""
    text = _as_str(value, "")
    if text == "":
        return ""
    canonical = _SHARDX_PLATFORM_ALIASES.get(text.lower())
    if canonical is None:
        allowed = ", ".join(VALID_SHARDX_PLATFORMS)
        raise ConfigError(f"{key} must be one of: {allowed}; got: {value!r}")
    return canonical


def parse_tags(value: Any) -> List[str]:
    """Parse a tag list from a comma/semicolon/space separated string or list.

    Leading ``#`` markers are stripped, duplicates removed and the original
    order preserved, so both ``PUBLISH_TAGS=аниме, anime`` and
    ``["#аниме", "anime"]`` produce ``["аниме", "anime"]``.

    The same helper is reused for ``PUBLISH_SHARDX_NOISE``.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items: List[str] = [str(item) for item in value]
    else:
        text = str(value)
        for separator in (";", ","):
            text = text.replace(separator, " ")
        raw_items = text.split()

    seen: List[str] = []
    for item in raw_items:
        tag = item.strip().lstrip("#").strip()
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def parse_schedule(value: Any) -> List[datetime]:
    """Parse a publish schedule into a list of :class:`datetime` objects.

    One entry per upload slot. Both ``T`` and a single space are accepted as the
    date/time separator, so all of these work::

        PUBLISH_SCHEDULE=2026-04-01T10:00, 2026-04-01 18:00, 2026-04-02T09:30
        ["2026-04-01T10:00", "2026-04-01T18:00"]

    The list length decides how many clips each profile publishes in
    ``schedule`` mode. Raises :class:`ConfigError` on an unparseable entry (and,
    being a naive wall-clock time, is filled into the platform as-is — YouTube
    Studio interprets it in the channel's timezone).
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_items: List[str] = [str(item) for item in value]
    else:
        text = str(value).replace(";", ",")
        raw_items = text.split(",")

    parsed: List[datetime] = []
    for item in raw_items:
        token = item.strip()
        if not token:
            continue
        normalized = token
        if "T" not in normalized and " " in normalized:
            # ``2026-04-01 18:00`` -> ``2026-04-01T18:00`` (only the first space).
            normalized = normalized.replace(" ", "T", 1)
        try:
            parsed.append(datetime.fromisoformat(normalized))
        except ValueError as exc:
            raise ConfigError(
                f"PUBLISH_SCHEDULE: cannot parse {token!r}; use ISO datetimes "
                "like '2026-04-01T10:00', separated by commas"
            ) from exc
    return parsed


# ---------------------------------------------------------------------------
# Configuration object
# ---------------------------------------------------------------------------


@dataclass
class PublishConfig:
    """Fully resolved configuration for the publisher."""

    # Where browser profiles and the state database live.
    profile_dir: str = "browser_profiles"
    db_path: str = "publish_state.sqlite3"

    # Where the clipping pipeline wrote its clips (shorts_info.json + .mp4).
    output_dir: str = "output"

    # Browser ------------------------------------------------------------------
    # Uploads must keep the window open until the transfer really finishes, so
    # headed (headless=False) is the safe default.
    headless: bool = False
    # Optional proxy applied to every launch, e.g. "http://127.0.0.1:7890".
    # For the shardx backend it is bound to the profile instead.
    proxy: str = ""

    # ShardX backend -------------------------------------------------------------
    # Library template id (e.g. "win-rtx4060"); empty -> a random template for
    # ``shardx_platform`` is frozen under a new id on first use.
    shardx_template: str = ""
    # Platform for the random template: Windows | macOS | Linux; empty -> host.
    shardx_platform: str = ""
    # SDK cache root (engine + saved profiles); empty -> the SDK's own default.
    shardx_cache_dir: str = ""
    # Screen strategy: profile | cap_to_host | use_host; empty -> SDK auto.
    shardx_screen_mode: str = ""
    # Re-randomize CPU/RAM/platform_version before each launch.
    shardx_randomize: bool = False
    # Anti-fingerprint noise vectors (canvas, webgl, audio, ...).
    shardx_noise: List[str] = field(default_factory=list)
    # WebRTC policy: auto | tcp_only | block.
    shardx_webrtc: str = "auto"
    # Browser language (BCP-47) forced on the profile: "en-US", "ru-RU", or "" /
    # "auto" to let the SDK derive it from geo. Only the language is pinned;
    # timezone/geolocation still come from the geo lookup.
    shardx_language: str = "en-US"

    # Distribution / scheduling ------------------------------------------------
    # How clips are spread over profiles: "distribute" (round-robin, publish now)
    # or "schedule" (round-robin, scheduled at ``schedule``).
    mode: str = "distribute"
    # Upload slots for "schedule" mode. Its length is how many clips each profile
    # publishes; a naive local/wall-clock datetime filled into the platform.
    schedule: List[datetime] = field(default_factory=list)

    # Publishing defaults ------------------------------------------------------
    # Hashtags applied to clips that do not carry their own tags.
    tags: List[str] = field(default_factory=list)
    visibility: str = "public"  # public | unlisted | private (YouTube)
    playlist: str = ""  # optional YouTube playlist name
    delay: float = 30.0  # seconds between two uploads

    # Misc ---------------------------------------------------------------------
    base_dir: str = field(default_factory=os.getcwd)

    def resolve(self, path: str) -> str:
        """Resolve a possibly-relative path against the working directory."""
        if not path:
            return ""
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded):
            return os.path.normpath(expanded)
        return os.path.normpath(os.path.join(self.base_dir, expanded))

    @property
    def profiles_path(self) -> str:
        return self.resolve(self.profile_dir)

    @property
    def db_full_path(self) -> str:
        return self.resolve(self.db_path)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _project_root() -> str:
    """Absolute path to the project root (the directory that holds ``main.py``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_env_files(base_dir: str, env_file: Optional[str]) -> List[str]:
    """Return the ``.env`` files to merge, in increasing priority order."""
    candidates = [
        os.path.join(_project_root(), ".env"),
        os.path.join(base_dir, ".env"),
    ]
    if env_file:
        candidates.append(env_file)
    seen: List[str] = []
    for path in candidates:
        absolute = os.path.abspath(os.path.expanduser(path))
        if os.path.isfile(absolute) and absolute not in seen:
            seen.append(absolute)
    return seen


def _env_key_for(name: str) -> str:
    """Map a field name or an already-prefixed env key to its ``PUBLISH_*`` key."""
    upper = name.upper()
    if upper in _FIELD_ENV.values():
        return upper
    if name in _FIELD_ENV:
        return _FIELD_ENV[name]
    return upper


def _coerce(field_name: str, raw: Any) -> Any:
    """Convert a raw string into the correct type for ``field_name``."""
    env_key = _FIELD_ENV[field_name]

    if field_name in _BOOL_FIELDS:
        return _as_bool(raw, env_key, False)
    if field_name in _FLOAT_FIELDS:
        return _as_float(raw, env_key, 0.0)
    if field_name == "schedule":
        return parse_schedule(raw)
    if field_name in _LIST_FIELDS:
        return parse_tags(raw)
    if field_name in _CHOICE_FIELDS:
        default, choices = _CHOICE_FIELDS[field_name]
        return _as_choice(raw, env_key, default, choices)
    if field_name in _OPTIONAL_CHOICE_FIELDS:
        return _as_optional_choice(raw, env_key, _OPTIONAL_CHOICE_FIELDS[field_name])
    if field_name == "shardx_platform":
        return _as_platform(raw, env_key)
    return _as_str(raw, "")


def _validate(cfg: PublishConfig) -> None:
    """Sanity-check values that cannot be validated by type coercion alone."""
    if not cfg.profile_dir:
        raise ConfigError("PUBLISH_PROFILE_DIR must not be empty")
    if not cfg.db_path:
        raise ConfigError("PUBLISH_DB must not be empty")
    if cfg.delay < 0:
        raise ConfigError("PUBLISH_DELAY must be zero or greater")
    if cfg.visibility not in VALID_VISIBILITIES:
        allowed = ", ".join(VALID_VISIBILITIES)
        raise ConfigError(f"PUBLISH_VISIBILITY must be one of: {allowed}")
    if cfg.mode not in VALID_PUBLISH_MODES:
        allowed = ", ".join(VALID_PUBLISH_MODES)
        raise ConfigError(f"PUBLISH_MODE must be one of: {allowed}")
    if cfg.mode == "schedule" and not cfg.schedule:
        raise ConfigError(
            "PUBLISH_MODE=schedule requires PUBLISH_SCHEDULE (one or more ISO "
            "datetimes, e.g. 2026-04-01T10:00,2026-04-01T18:00)"
        )

    unknown = [
        vector
        for vector in cfg.shardx_noise
        if vector.lower() not in VALID_SHARDX_NOISE
    ]
    if unknown:
        allowed = ", ".join(VALID_SHARDX_NOISE)
        raise ConfigError(
            f"PUBLISH_SHARDX_NOISE has unknown vectors: {', '.join(unknown)} "
            f"(allowed: {allowed})"
        )


def load_publish_config(
    env_file: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    base_dir: Optional[str] = None,
) -> PublishConfig:
    """Build a :class:`PublishConfig` from ``.env`` files plus overrides.

    ``extra`` maps field names (or their ``PUBLISH_*`` keys) to raw values;
    typically values collected from CLI arguments. Those values run through the
    same coercion and validation as ``.env`` values and win over everything
    else.
    """
    working_dir = os.path.abspath(base_dir or os.getcwd())
    cfg = PublishConfig(base_dir=working_dir)

    mapping: dict[str, Any] = {}
    for path in _default_env_files(working_dir, env_file):
        # Later files win, so process in order and update.
        mapping.update(parse_env_file(path))

    # OS environment variables take precedence over file values.
    for env_key in _FIELD_ENV.values():
        if env_key in os.environ:
            mapping[env_key] = os.environ[env_key]

    # Explicit overrides (CLI) win over everything else.
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            mapping[_env_key_for(str(key))] = value

    for field_name, env_key in _FIELD_ENV.items():
        if env_key in mapping:
            setattr(cfg, field_name, _coerce(field_name, mapping[env_key]))

    _validate(cfg)
    return cfg


__all__ = [
    "PublishConfig",
    "VALID_PUBLISH_MODES",
    "VALID_SHARDX_NOISE",
    "VALID_SHARDX_PLATFORMS",
    "VALID_SHARDX_SCREEN_MODES",
    "VALID_SHARDX_WEBRTC",
    "VALID_VISIBILITIES",
    "load_publish_config",
    "parse_schedule",
    "parse_tags",
]
