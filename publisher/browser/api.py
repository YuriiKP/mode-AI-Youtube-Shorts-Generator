"""Low-level access to the ShardX Launcher's local automation HTTP API.

Kept separate from :mod:`publisher.browser.shardx_backend` so that
:mod:`publisher.profile` can discover launcher profiles without importing the
browser-driving backend — importing that from ``profile`` would be a cycle
(``shardx_backend`` type-checks against ``profile``).

``httpx`` is imported lazily, so importing this module never fails on a machine
without the browser stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from ..log import log

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import PublishConfig


#: Where the launcher's automation API listens by default.
DEFAULT_API_URL = "http://127.0.0.1:40325"


def api_endpoint(cfg: "PublishConfig") -> Tuple[str, Dict[str, str]]:
    """Return the launcher API ``(base_url, headers)`` described by ``cfg``."""
    base = (getattr(cfg, "shardx_api_url", "") or "").strip() or DEFAULT_API_URL
    headers = {"Accept": "application/json"}
    token = (getattr(cfg, "shardx_api_token", "") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return base.rstrip("/"), headers


def launcher_profiles(cfg: "PublishConfig") -> List[Dict[str, Any]]:
    """List the launcher's persistent profiles (blocking, best-effort).

    Used by :mod:`publisher.profile` to discover profiles that already exist in
    the launcher but were not registered locally yet. Returns ``[]`` when the
    launcher (or ``httpx``) is unavailable, so callers fall back to the local
    registry instead of failing.
    """
    try:
        import httpx
    except ImportError:  # pragma: no cover - depends on the environment
        return []

    base_url, headers = api_endpoint(cfg)
    try:
        with httpx.Client(base_url=base_url, headers=headers, timeout=15.0) as client:
            response = client.get("/profiles")
    except httpx.HTTPError as exc:
        log.debug("could not list ShardX Launcher profiles: %s", exc)
        return []

    if response.status_code != 200:
        log.debug(
            "could not list ShardX Launcher profiles: HTTP %s", response.status_code
        )
        return []
    try:
        data = response.json()
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


__all__ = ["DEFAULT_API_URL", "api_endpoint", "launcher_profiles"]
