"""The publisher's browser backend.

Browser automation runs on the **ShardX anti-detect engine**
(:mod:`publisher.browser.shardx_backend`): the SDK launches its own hardened
Chromium build and spoofs the fingerprint at the engine level, so no separate
Chrome/Chromium install is needed.

The backend is imported lazily (inside :mod:`publisher.session`), so importing
this package never fails on a machine where the ShardX SDK is not installed yet.
"""

from __future__ import annotations

from .base import (
    DEFAULT_TIMEOUT_MS,
    BrowserUnavailableError,
)

__all__ = [
    "BrowserUnavailableError",
    "DEFAULT_TIMEOUT_MS",
]
