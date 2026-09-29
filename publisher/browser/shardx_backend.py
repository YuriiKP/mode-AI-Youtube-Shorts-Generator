"""The browser backend: the ShardX anti-detect engine via its Python SDK.

This is the only browser backend of the publisher. It does not launch Chromium
itself: the ``shardx`` SDK spawns a patched Chromium build with an engine-level
fingerprint and hands back a ready-to-use Playwright-compatible ``Browser`` over
CDP. We take that browser's default context (``browser.contexts[0]``) — the one
that carries the profile's cookies — and yield it, so the platform uploaders
stay completely engine-agnostic.

Two fingerprints of this backend are deliberate:

* **No ``stealth.min.js``.** The engine spoofs at the C++ level (Blink, V8, the
  network stack), so a JavaScript shim would be redundant and could even read as
  tampering.
* **We never close anything.** The whole lifecycle (launch *and* teardown) is
  owned by ``sdk.session(...)``; closing the context or browser by hand would
  fight the SDK.

A profile is *persistent*: cookies and the frozen fingerprint live in the SDK's
own user-data dir (``<cache>/profiles/<id>/``). We remember that id on our
:class:`~publisher.profile.Profile` (``shardx_id``) so the next run reopens the
same identity instead of minting a new one.

The SDK is imported lazily, so importing the publisher never fails on a machine
where ``shardx`` is not installed.
"""

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator, Dict, Optional

from ..log import log
from .base import DEFAULT_TIMEOUT_MS, BrowserUnavailableError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import PublishConfig
    from ..profile import Profile


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def _import_shardx():
    """Import the ShardX SDK lazily and turn a missing dependency into a hint."""
    try:
        from shardx import ShardX
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise BrowserUnavailableError(
            "the ShardX SDK is not installed. Install it with 'pip install shardx' "
            "(the engine is downloaded from the CDN on first use)."
        ) from exc
    return ShardX


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------


def _resolve_sdk_profile(sdk, cfg: "PublishConfig", profile: "Profile"):
    """Return the SDK profile behind ``profile``, creating it on first use.

    A saved ShardX profile is reopened by the id we stored in ``meta.json`` (same
    fingerprint *and* cookies). When there is no id yet — or the SDK no longer
    knows it — a fresh profile is frozen from the configured library template
    (or a random one for the configured platform) and its id is persisted.
    """
    if profile.shardx_id:
        try:
            return sdk.open_profile(profile.shardx_id)
        except Exception as exc:  # noqa: BLE001 - fall back to creating one
            log.warning(
                "could not reopen ShardX profile %s (%s); creating a new one",
                profile.shardx_id,
                exc,
            )

    try:
        if cfg.shardx_template:
            sdk_profile = sdk.create_profile(cfg.shardx_template)
        else:
            sdk_profile = sdk.create_profile(platform=cfg.shardx_platform or None)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise BrowserUnavailableError(
            _creation_error_message(cfg, profile, exc)
        ) from exc

    profile.shardx_id = getattr(sdk_profile, "id", "") or ""
    if profile.shardx_id:
        profile.save_meta()
    log.info(
        "created ShardX profile %s for '%s'", profile.shardx_id or "?", profile.name
    )
    return sdk_profile


def _creation_error_message(
    cfg: "PublishConfig", profile: "Profile", exc: Exception
) -> str:
    """Turn a raw profile-creation failure into an actionable message."""
    text = str(exc).strip() or exc.__class__.__name__
    parts = [f"could not create a ShardX profile for '{profile.name}': {text}"]
    if cfg.shardx_template:
        parts.append(
            f"Check PUBLISH_SHARDX_TEMPLATE={cfg.shardx_template!r} — list valid "
            "ids with 'python -c \"from shardx import ShardX; print(ShardX().list_profiles())\"'."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def _apply_noise(sdk, sdk_profile, cfg: "PublishConfig") -> None:
    """Apply the configured anti-fingerprint noise vectors to the profile."""
    vectors = [v for v in (cfg.shardx_noise or []) if v]
    if not vectors:
        return
    try:
        sdk_profile.set_noise(*vectors)
        sdk.save_profile(sdk_profile)
        log.info("applied ShardX noise vectors: %s", ", ".join(vectors))
    except Exception as exc:  # noqa: BLE001 - noise is best-effort
        log.warning("could not apply ShardX noise vectors %s: %s", vectors, exc)


def _apply_language(sdk, sdk_profile, cfg: "PublishConfig") -> None:
    """Pin the browser language (and the matching Accept-Language/Intl locale).

    Left alone, the SDK derives ``navigator.language`` from the geo (the bound
    proxy, or the host), so a Russian IP yields a Russian UI — which the
    uploaders' English-text selectors do not expect. Writing a concrete locale
    here stops that: the SDK only rewrites ``navigator.language`` when it is the
    ``"auto"`` sentinel, so a fixed value is left untouched. Timezone and
    geolocation are still resolved from the geo lookup.

    Set ``PUBLISH_SHARDX_LANGUAGE=`` (empty) or ``auto`` to let the SDK decide.
    """
    locale = (cfg.shardx_language or "").strip()
    if not locale or locale.lower() == "auto":
        return

    base = locale.split("-", 1)[0]
    if locale == "en-US":
        accept_language = "en-US,en;q=0.9"
        languages = ["en-US", "en"]
    else:
        accept_language = f"{locale},{base};q=0.9,en-US;q=0.8,en;q=0.7"
        languages = [locale, base, "en-US", "en"]

    navigator = dict(sdk_profile.config.get("navigator") or {})
    navigator["language"] = locale
    navigator["accept_language"] = accept_language
    navigator["languages"] = languages
    sdk_profile.config["navigator"] = navigator
    sdk_profile.config["icu_locale"] = locale

    try:
        sdk.save_profile(sdk_profile)
    except Exception as exc:  # noqa: BLE001 - language is best-effort
        log.warning("could not persist language %s: %s", locale, exc)
        return
    log.info("forced ShardX browser language: %s", locale)


def _filter_supported_kwargs(func, candidates: Dict[str, object]) -> Dict[str, object]:
    """Keep only the ``candidates`` the callee actually accepts.

    The SDK's ``session()`` signature has changed across releases; rather than
    hard-coding a call that may ``TypeError`` on a different version, we inspect
    it and drop anything unsupported (logging what we skipped).
    """
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return dict(candidates)

    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        return dict(candidates)

    supported = {k: v for k, v in candidates.items() if k in parameters}
    dropped = sorted(set(candidates) - set(supported))
    if dropped:
        log.debug("shardx session() does not accept: %s", ", ".join(dropped))
    return supported


def _session_kwargs(
    cfg: "PublishConfig", session_callable, headless: bool
) -> Dict[str, object]:
    """Build the keyword arguments for ``sdk.session(...)``."""
    candidates: Dict[str, object] = {}
    if cfg.proxy:
        candidates["proxy"] = cfg.proxy
    if cfg.shardx_screen_mode:
        candidates["screen_mode"] = cfg.shardx_screen_mode
    if cfg.shardx_randomize:
        candidates["randomize"] = True
    if cfg.shardx_webrtc and cfg.shardx_webrtc != "auto":
        candidates["webrtc"] = cfg.shardx_webrtc
    if headless:
        candidates["headless"] = True
    return _filter_supported_kwargs(session_callable, candidates)


def _launch_error_message(
    cfg: "PublishConfig", profile: "Profile", exc: Exception
) -> str:
    """Turn a raw launch exception into an actionable message."""
    text = str(exc).strip() or exc.__class__.__name__
    parts = [f"shardx: could not launch profile '{profile.name}': {text}"]
    if cfg.proxy:
        parts.append(
            "The bound proxy is probed before launch (UDP + geo); a broken or "
            "unreachable proxy fails the launch."
        )
    parts.append(
        "Run 'python -c \"import shardx; shardx.ShardX()\"' to force the engine "
        "download."
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def open(
    cfg: "PublishConfig", profile: "Profile", *, headless: bool
) -> AsyncIterator[object]:
    """Open ``profile`` in the ShardX engine and yield its browser context.

    The engine is launched by the SDK; on exit the SDK's context manager shuts
    the browser down. We deliberately do not close anything ourselves.
    """
    ShardX = _import_shardx()

    cache_dir: Optional[str] = cfg.shardx_cache_dir or None
    try:
        sdk = ShardX(cache_dir=cache_dir)
    except TypeError:  # pragma: no cover - older SDK without cache_dir
        sdk = ShardX()

    sdk_profile = _resolve_sdk_profile(sdk, cfg, profile)
    _apply_language(sdk, sdk_profile, cfg)
    _apply_noise(sdk, sdk_profile, cfg)

    kwargs = _session_kwargs(cfg, sdk.session, headless)
    log.debug("shardx session kwargs: %s", sorted(kwargs))

    try:
        async with sdk.session(sdk_profile, **kwargs) as browser:
            contexts = list(getattr(browser, "contexts", []) or [])
            if not contexts:
                raise BrowserUnavailableError(
                    f"shardx: profile '{profile.name}' launched without a browser "
                    "context; the profile's cookies would not be reachable."
                )
            context = contexts[0]
            try:
                context.set_default_timeout(DEFAULT_TIMEOUT_MS)
            except Exception:  # noqa: BLE001 - not all clients expose it
                pass
            yield context
    except BrowserUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise BrowserUnavailableError(_launch_error_message(cfg, profile, exc)) from exc


__all__ = ["open"]
