"""The platform registry.

Maps the command-line platform names to the modules that implement them, so the
rest of the publisher never has to import a concrete platform directly:

    from publisher import platforms

    for module in platforms.resolve_platforms("all"):
        if await module.check(context, cfg, logger):
            result = await module.upload(context, short, cfg, logger)

Only the two supported platforms are registered here (``youtube`` and
``tiktok``). Adding a new one is a two-line change: drop its module into this
package and add it to :data:`PLATFORMS`.

The platform modules are imported eagerly, but that is safe: they never import
the browser engine themselves — they only ever receive an open
``BrowserContext`` — so importing this package never fails on a machine without
the browser stack.
"""

from __future__ import annotations

from types import ModuleType
from typing import Dict, List

from . import tiktok, youtube

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Ordered mapping of platform name -> implementing module. The order defines
#: the default processing order (YouTube first, then TikTok).
PLATFORMS: Dict[str, ModuleType] = {
    youtube.NAME: youtube,
    tiktok.NAME: tiktok,
}

#: Names of every known platform, in registry order.
ALL_PLATFORMS = tuple(PLATFORMS)

#: Selectors that mean "every platform" on the command line.
_RESERVED = {"all", "*"}


class PlatformError(RuntimeError):
    """Raised for an unknown platform name or an empty platform selection."""


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def all_platforms() -> List[ModuleType]:
    """Return every registered platform module, in registry order."""
    return [PLATFORMS[name] for name in ALL_PLATFORMS]


def names() -> List[str]:
    """Return every known platform name, in registry order."""
    return list(ALL_PLATFORMS)


def labels() -> Dict[str, str]:
    """Return a ``{name: LABEL}`` mapping for human-friendly output."""
    return {name: module.LABEL for name, module in PLATFORMS.items()}


def is_known(name: str) -> bool:
    """Whether ``name`` is a registered platform."""
    return (name or "").strip().lower() in PLATFORMS


def get_platform(name: str) -> ModuleType:
    """Return the module for ``name`` or raise :class:`PlatformError`."""
    key = (name or "").strip().lower()
    module = PLATFORMS.get(key)
    if module is None:
        allowed = ", ".join(ALL_PLATFORMS)
        raise PlatformError(f"unknown platform: {name!r} (available: {allowed})")
    return module


def resolve_platforms(spec: str = "all") -> List[ModuleType]:
    """Resolve a platform selector into a list of modules.

    ``spec`` is either ``all``/``*`` (every platform) or a comma-separated list
    of names such as ``"youtube"`` or ``"youtube,tiktok"``. Blank items are
    ignored and duplicates removed while the requested order is preserved.
    """
    text = (spec or "all").strip()

    if not text or text.lower() in _RESERVED:
        return all_platforms()

    modules: List[ModuleType] = []
    seen: set = set()
    for part in text.split(","):
        name = part.strip().lower()
        if not name:
            continue
        module = get_platform(name)
        if module.NAME not in seen:
            seen.add(module.NAME)
            modules.append(module)

    if not modules:
        raise PlatformError("no platforms selected")

    return modules


def describe(module: ModuleType) -> str:
    """Return a one-line description of a platform module."""
    return f"{module.LABEL} ({module.NAME})"


__all__ = [
    "PLATFORMS",
    "ALL_PLATFORMS",
    "PlatformError",
    "all_platforms",
    "names",
    "labels",
    "is_known",
    "get_platform",
    "resolve_platforms",
    "describe",
]
