"""Browser profile management for the publisher.

A *profile* is a named, persistent Chromium user-data directory. Everything a
site needs to stay logged in — cookies, local storage, the whole session — lives
inside that one directory, which is exactly why a single profile can hold the
logins for several platforms at once (YouTube *and* TikTok). Nothing is copied
out of it; the directory **is** the credential store.

On disk a profile looks like::

    browser_profiles/
      profile_1/
        Default/            # Chromium user-data-dir (cookies live here)
        meta.json           # our bookkeeping: note, platforms, timestamps
        storage_state.json  # optional, portable backup of the cookies
        .publisher.lock     # advisory lock while a browser is open

This module is deliberately browser-free: it only knows about paths, metadata and
an advisory lock. Launching the browser against a profile lives in
:mod:`publisher.session`.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .config import PublishConfig
from .log import log

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

META_FILE = "meta.json"
STORAGE_STATE_FILE = "storage_state.json"
LOCK_FILE = ".publisher.lock"

# A profile name must be a single, filesystem-safe path segment.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

# ``all`` / ``*`` are selectors, never real profile names.
_RESERVED = {"all", "*"}

# A directory counts as a browser profile when it carries any of these markers.
# ``meta.json`` is ours; ``Default`` and ``Local State`` are created by Chromium.
_PROFILE_MARKERS = (META_FILE, "Default", "Local State")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProfileError(RuntimeError):
    """Raised for an invalid profile name or a missing profile."""


class ProfileInUseError(ProfileError):
    """Raised when a profile is already open in another process."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    """Current local time as an ISO-8601 string (second resolution)."""
    return datetime.now().isoformat(timespec="seconds")


def _pid_alive(pid: int) -> bool:
    """Best-effort check whether a process id is still running.

    On Windows ``os.kill`` cannot be used to probe a process (a signal of 0 is
    interpreted as a real termination request), so the Win32 ``OpenProcess`` API
    is used in a query-only fashion instead.
    """
    if not pid or pid <= 0:
        return False

    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:  # noqa: BLE001 - probing must never raise
            return False

    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but belongs to another user.
        return True
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass
class Profile:
    """A named persistent browser profile and its bookkeeping metadata."""

    name: str
    path: str
    note: str = ""
    created_at: str = ""
    updated_at: str = ""
    # Platforms whose login has been recorded in this profile.
    platforms: List[str] = field(default_factory=list)
    # Id of the matching ShardX SDK profile (its fingerprint and cookies live
    # there); empty until the profile is first opened.
    shardx_id: str = ""

    # -- paths -------------------------------------------------------------

    @property
    def meta_path(self) -> str:
        return os.path.join(self.path, META_FILE)

    @property
    def storage_state_path(self) -> str:
        """Portable backup of the profile's cookies (Playwright storage_state)."""
        return os.path.join(self.path, STORAGE_STATE_FILE)

    @property
    def lock_path(self) -> str:
        return os.path.join(self.path, LOCK_FILE)

    # -- state -------------------------------------------------------------

    def exists(self) -> bool:
        """Whether the profile directory is present on disk."""
        return os.path.isdir(self.path)

    def add_platform(self, platform: str, *, save: bool = True) -> None:
        """Remember that ``platform`` was set up in this profile."""
        platform = (platform or "").strip().lower()
        if platform and platform not in self.platforms:
            self.platforms.append(platform)
            if save:
                self.save_meta()

    def describe(self) -> str:
        """One-line human summary used by the ``profiles`` command."""
        if self.platforms:
            status = ", ".join(self.platforms)
        else:
            status = "no logins recorded"
        suffix = f" — {self.note}" if self.note else ""
        return f"{self.name}: {status}{suffix}"

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "note": self.note,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "platforms": list(self.platforms),
            "shardx_id": self.shardx_id,
        }

    def save_meta(self) -> None:
        """Write ``meta.json`` atomically (write-then-rename)."""
        os.makedirs(self.path, exist_ok=True)
        self.updated_at = _now()
        if not self.created_at:
            self.created_at = self.updated_at
        tmp_path = self.meta_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.meta_path)

    @classmethod
    def load(cls, path: str) -> "Profile":
        """Load a profile from a directory, reading ``meta.json`` when present."""
        path = os.path.normpath(path)
        profile = cls(name=os.path.basename(path), path=path)

        meta_path = os.path.join(path, META_FILE)
        if not os.path.isfile(meta_path):
            return profile

        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            log.warning("could not read %s: %s", meta_path, exc)
            return profile

        if not isinstance(data, dict):
            return profile

        profile.note = str(data.get("note") or "")
        profile.created_at = str(data.get("created_at") or "")
        profile.updated_at = str(data.get("updated_at") or "")
        profile.shardx_id = str(data.get("shardx_id") or "")
        platforms = data.get("platforms") or []
        if isinstance(platforms, list):
            profile.platforms = [
                str(item).strip().lower() for item in platforms if str(item).strip()
            ]
        return profile


# ---------------------------------------------------------------------------
# Name / path helpers
# ---------------------------------------------------------------------------


def validate_name(name: str) -> str:
    """Validate a user-supplied profile name and return it normalised."""
    name = (name or "").strip()
    if name.lower() in _RESERVED:
        raise ProfileError(f"'{name}' is a reserved profile selector, not a name")
    if not _NAME_RE.match(name):
        raise ProfileError(
            "Profile names must start with a letter or digit and may contain "
            f"only letters, digits, '.', '_' or '-' (got: {name!r})"
        )
    return name


def profiles_root(cfg: PublishConfig, *, create: bool = True) -> str:
    """Absolute path to the directory that holds every profile."""
    root = cfg.profiles_path
    if create:
        os.makedirs(root, exist_ok=True)
    return root


def profile_path(cfg: PublishConfig, name: str) -> str:
    """Absolute path of a single profile directory."""
    return os.path.join(profiles_root(cfg), validate_name(name))


def _looks_like_profile(path: str) -> bool:
    """Whether a directory carries the markers of a browser profile."""
    return any(
        os.path.exists(os.path.join(path, marker)) for marker in _PROFILE_MARKERS
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def list_profiles(cfg: PublishConfig) -> List[Profile]:
    """Return every profile found under the profiles root (sorted by name)."""
    root = profiles_root(cfg, create=False)
    if not os.path.isdir(root):
        return []

    profiles: List[Profile] = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if not os.path.isdir(path) or not _looks_like_profile(path):
            continue
        profiles.append(Profile.load(path))
    return profiles


def create_profile(cfg: PublishConfig, name: str, note: str = "") -> Profile:
    """Create (or open) a profile directory and write its metadata."""
    name = validate_name(name)
    path = os.path.join(profiles_root(cfg), name)
    os.makedirs(path, exist_ok=True)
    profile = Profile(name=name, path=path, note=note, created_at=_now())
    profile.save_meta()
    log.info("created browser profile '%s' at %s", name, path)
    return profile


def get_profile(
    cfg: PublishConfig,
    name: str,
    *,
    create: bool = False,
    note: str = "",
) -> Profile:
    """Fetch one profile by name, optionally creating it when missing."""
    name = validate_name(name)
    path = os.path.join(profiles_root(cfg), name)
    if not os.path.isdir(path):
        if not create:
            raise ProfileError(
                f"Profile '{name}' does not exist under {profiles_root(cfg)}. "
                f"Create it with: python main.py publish manual --profile {name}"
            )
        return create_profile(cfg, name, note=note)
    return Profile.load(path)


def resolve_profiles(
    cfg: PublishConfig,
    spec: str = "all",
    *,
    create: bool = False,
) -> List[Profile]:
    """Resolve a profile selector into a concrete list of profiles.

    ``spec`` is either ``all``/``*`` (every existing profile) or a
    comma-separated list of names. Duplicates are removed while the requested
    order is preserved.
    """
    if not spec or spec.strip().lower() in _RESERVED:
        profiles = list_profiles(cfg)
        if not profiles:
            raise ProfileError(
                f"No profiles found under {profiles_root(cfg)}. Create one with: "
                "python main.py publish manual --profile <name>"
            )
        return profiles

    names = [name.strip() for name in spec.split(",") if name.strip()]
    seen: set = set()
    profiles: List[Profile] = []
    for name in names:
        profile = get_profile(cfg, name, create=create)
        if profile.name not in seen:
            seen.add(profile.name)
            profiles.append(profile)
    return profiles


# ---------------------------------------------------------------------------
# Advisory lock
# ---------------------------------------------------------------------------


class ProfileLock:
    """Advisory lock that stops two processes from opening one profile at once.

    Chromium refuses to open a user-data-dir that is already in use, and the
    failure mode is a cryptic crash. This lock turns that into a clear message
    *before* the browser is launched. A lock left behind by a crashed process is
    reclaimed automatically by checking whether the recorded PID is still alive.

    Use it as a context manager::

        with ProfileLock(profile):
            await session.open_profile_context(cfg, profile)
    """

    def __init__(self, profile: Profile, *, wait: float = 0.0, poll: float = 0.5):
        self.profile = profile
        self.wait = max(0.0, wait)
        self.poll = max(0.05, poll)
        self._owned = False

    # -- internals ---------------------------------------------------------

    def _read_owner(self) -> Optional[int]:
        try:
            with open(self.profile.lock_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return None
        try:
            return int(data.get("pid"))
        except (TypeError, ValueError):
            return None

    def _remove_lock(self) -> None:
        try:
            os.remove(self.profile.lock_path)
        except OSError:
            pass

    def _try_acquire(self) -> object:
        """Return ``True`` on success, ``"retry"`` after reclaiming, else ``False``."""
        os.makedirs(self.profile.path, exist_ok=True)
        try:
            fd = os.open(
                self.profile.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError:
            owner = self._read_owner()
            # A lock whose owner is gone is stale: drop it and retry at once.
            if owner is None or not _pid_alive(owner):
                self._remove_lock()
                return "retry"
            return False
        except OSError:
            return False

        try:
            payload = json.dumps({"pid": os.getpid(), "started_at": _now()})
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)

        self._owned = True
        return True

    # -- public API --------------------------------------------------------

    def acquire(self) -> "ProfileLock":
        """Acquire the lock, waiting up to ``wait`` seconds for it to free up."""
        deadline = time.monotonic() + self.wait
        while True:
            result = self._try_acquire()
            if result is True:
                return self
            if result == "retry":
                continue
            if time.monotonic() >= deadline:
                owner = self._read_owner()
                who = f" (held by pid {owner})" if owner else ""
                raise ProfileInUseError(
                    f"Profile '{self.profile.name}' is already open{who}. "
                    "Close the other browser window or command and try again."
                )
            time.sleep(self.poll)

    def release(self) -> None:
        if self._owned:
            self._remove_lock()
            self._owned = False

    def __enter__(self) -> "ProfileLock":
        return self.acquire()

    def __exit__(self, *exc_info) -> None:
        self.release()


__all__ = [
    "Profile",
    "ProfileError",
    "ProfileInUseError",
    "ProfileLock",
    "META_FILE",
    "STORAGE_STATE_FILE",
    "LOCK_FILE",
    "create_profile",
    "get_profile",
    "list_profiles",
    "profile_path",
    "profiles_root",
    "resolve_profiles",
    "validate_name",
]
