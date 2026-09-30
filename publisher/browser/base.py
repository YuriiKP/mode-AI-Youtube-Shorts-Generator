"""Shared primitives for the publisher's browser backend.

Everything that touches a browser is launched through the ShardX Launcher's
automation API (see :mod:`publisher.browser.shardx_backend`). This module only
holds the small engine-independent pieces the rest of the package imports: the
error raised when a launch cannot happen, and the shared page-operation timeout.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BrowserUnavailableError(RuntimeError):
    """Raised when the browser engine is missing or a launch failed."""


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Default timeout for page operations (waits, clicks). Uploads can sit on slow
# selectors for a while, so this is deliberately generous.
DEFAULT_TIMEOUT_MS = 60_000


__all__ = [
    "BrowserUnavailableError",
    "DEFAULT_TIMEOUT_MS",
]
