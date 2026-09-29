"""Launch a browser bound to a persistent profile — the single place the
publisher starts a browser.

:func:`open_profile_context` delegates the whole launch *and* teardown to the
ShardX engine (:mod:`publisher.browser.shardx_backend`). Everything downstream —
the platform uploaders in :mod:`publisher.platforms` — only ever sees the
``BrowserContext`` the backend yields, so it stays completely engine-agnostic.

The profile is opened as a **persistent context**: the cookies live in the
profile's user-data dir (owned by the SDK), so nothing has to be exported or
re-imported between runs, and the interactive ``manual`` mode saves the user's
logins automatically.

The engine is imported lazily (inside the backend module), so importing this
module — or simply running the clipping pipeline — never fails on a machine
where the ShardX SDK has not been installed yet.

This module also holds the small, engine-independent helpers the commands use:
the per-platform entry URLs, opening a set of tabs, and exporting a portable
cookie backup.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterable, List, Optional

from .browser import shardx_backend
from .browser.base import (  # re-exported so callers keep importing from here
    DEFAULT_TIMEOUT_MS,
    BrowserUnavailableError,
)
from .config import PublishConfig
from .log import log
from .profile import Profile, ProfileInUseError, ProfileLock

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The login / studio entry pages used by the interactive manual mode.
SITE_URLS = {
    "youtube": "https://studio.youtube.com",
    "tiktok": "https://www.tiktok.com/tiktokstudio/upload?lang=en",
}


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def open_profile_context(
    cfg: PublishConfig,
    profile: Profile,
    *,
    headless: Optional[bool] = None,
) -> AsyncIterator["object"]:
    """Open ``profile`` in the ShardX browser and yield its persistent context.

    An advisory :class:`~publisher.profile.ProfileLock` is taken for the
    duration, so two commands can never fight over the same profile. The browser
    is always closed on exit (by the backend).

    Args:
        cfg: resolved publisher configuration.
        profile: the profile to open (its ShardX identity holds the cookies).
        headless: override the configured headless mode for this launch. Uploads
            should stay headed (``False``) so a transfer is not cut short;
            defaults to ``cfg.headless``.
    """
    use_headless = cfg.headless if headless is None else bool(headless)

    try:
        lock = ProfileLock(profile)
    except ProfileInUseError:
        raise

    with lock:
        async with shardx_backend.open(cfg, profile, headless=use_headless) as context:
            yield context


# ---------------------------------------------------------------------------
# Small helpers used by the commands
# ---------------------------------------------------------------------------


def platform_urls(platforms: Iterable[str]) -> List[str]:
    """Return the entry URLs for a set of platform names (order preserved)."""
    urls: List[str] = []
    for platform in platforms:
        url = SITE_URLS.get(platform.strip().lower())
        if url and url not in urls:
            urls.append(url)
    return urls


async def open_pages(context, urls: Iterable[str]) -> None:
    """Open one tab per URL in ``context``, ignoring blank entries."""
    for url in urls:
        if not url:
            continue
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001 - keep going, open the rest
            log.warning("could not open %s: %s", url, exc)


async def export_storage_state(context, profile: Profile) -> str:
    """Write a portable ``storage_state.json`` backup of the profile cookies."""
    path = profile.storage_state_path
    try:
        await context.storage_state(path=path)
        log.info("saved cookie backup to %s", path)
        return path
    except Exception as exc:  # noqa: BLE001 - a backup must not break a run
        log.warning("could not export cookies for %s: %s", profile.name, exc)
        return ""


__all__ = [
    "BrowserUnavailableError",
    "DEFAULT_TIMEOUT_MS",
    "SITE_URLS",
    "export_storage_state",
    "open_pages",
    "open_profile_context",
    "platform_urls",
]
