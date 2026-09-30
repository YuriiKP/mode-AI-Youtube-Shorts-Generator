"""The browser backend: the ShardX Launcher's local automation HTTP API.

This is the only browser backend of the publisher. Instead of importing the
standalone ``shardx`` Python SDK, it drives the **ShardX Launcher desktop app**
through its local HTTP API (``http://127.0.0.1:40325`` by default, Bearer JWT):

* ``GET /fingerprint/new/{platform}`` — a freshly uniquified fingerprint;
* ``POST /profiles`` — freeze it into a persistent launcher profile;
* ``POST /profiles/{id}/start`` — launch it and return a CDP endpoint;
* ``POST /profiles/{id}/stop`` — shut it down.

We then attach **patchright** to that CDP endpoint and hand the profile's
default browser context (``browser.contexts[0]``) to the caller, so the
platform uploaders stay completely engine-agnostic.

Two fingerprints of this backend are deliberate:

* **No ``stealth.min.js``.** The engine spoofs at the C++ level (Blink, V8, the
  network stack), so a JavaScript shim would be redundant and could even read as
  tampering.
* **We own the lifecycle over the API.** A profile we started here is stopped by
  ``POST /profiles/{id}/stop`` on exit; a profile that was already running (e.g.
  started from the launcher UI) is attached to but left running.

A profile is *persistent*: cookies and the frozen fingerprint live in the
launcher's own user-data dir. We remember the launcher profile id (a UUID) on
our :class:`~publisher.profile.Profile` (``shardx_id``) so the next run reopens
the same identity instead of minting a new one.

``httpx`` and ``patchright`` are imported lazily, so importing the publisher
never fails on a machine where the browser stack is not installed.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple

from ..log import log
from .api import api_endpoint
from .base import DEFAULT_TIMEOUT_MS, BrowserUnavailableError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import PublishConfig
    from ..profile import Profile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Every noise vector the pipeline can switch on, in a stable order.
NOISE_VECTORS = ("canvas", "webgl", "audio", "client_rects", "sensors", "fonts")

#: vector -> (soft knob, value written when the vector is enabled at 0 strength).
_NOISE_KNOB = {"webgl": ("intensity", 0.0005), "client_rects": ("max_offset", 1)}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class _ApiError(BrowserUnavailableError):
    """A non-2xx answer from the launcher's automation API."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


class _LauncherApi:
    """A small async client for the launcher's automation API."""

    def __init__(self, cfg: "PublishConfig") -> None:
        self.base_url, self._headers = api_endpoint(cfg)
        self._client = None

    async def __aenter__(self) -> "_LauncherApi":
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BrowserUnavailableError(
                "httpx is not installed; it is required to talk to the ShardX "
                "launcher API. Install it with 'pip install httpx'."
            ) from exc

        self._client = httpx.AsyncClient(
            base_url=self.base_url, headers=self._headers, timeout=120.0
        )
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        ok: Tuple[int, ...] = (200,),
        missing_ok: bool = False,
    ) -> Any:
        """Send a request and decode the JSON answer.

        ``missing_ok`` turns a ``404`` into ``None`` (used when probing whether
        a remembered profile still exists on the launcher).
        """
        import httpx

        assert self._client is not None  # __aenter__ guarantees this

        try:
            response = await self._client.request(method, path, json=body)
        except httpx.HTTPError as exc:
            raise BrowserUnavailableError(
                f"could not reach the ShardX launcher at {self.base_url} ({exc}). "
                "Start the ShardX Launcher and enable Settings > Automation API."
            ) from exc

        if missing_ok and response.status_code == 404:
            return None
        if response.status_code not in ok:
            raise self._error(method, path, response)

        if not response.content:
            return None
        try:
            return response.json()
        except json.JSONDecodeError:
            return None

    def _error(self, method: str, path: str, response) -> "_ApiError":
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                detail = str(payload["error"])
        except Exception:  # noqa: BLE001 - the body may not be JSON at all
            detail = ""

        if response.status_code == 401:
            return _ApiError(
                401,
                "the ShardX launcher rejected the API token (HTTP 401). Set "
                "PUBLISH_SHARDX_API_TOKEN to the token shown in the launcher's "
                "Settings > Automation API.",
            )
        text = detail or (response.text or "").strip()[:200] or "no details"
        return _ApiError(
            response.status_code,
            f"{method} {path} -> HTTP {response.status_code}: {text}",
        )


# ---------------------------------------------------------------------------
# Fingerprint shaping
# ---------------------------------------------------------------------------


def _language_values(locale: str) -> Tuple[str, List[str]]:
    base = locale.split("-", 1)[0]
    if locale == "en-US":
        accept = "en-US,en;q=0.9"
        languages = ["en-US", "en"]
    else:
        accept = f"{locale},{base};q=0.9,en-US;q=0.8,en;q=0.7"
        languages = [locale, base, "en-US", "en"]
    return accept, languages


def _apply_language(fingerprint: Dict[str, Any], cfg: "PublishConfig") -> None:
    """Pin ``navigator.language`` / Accept-Language on a fresh fingerprint.

    Left alone, the launcher derives the language from the geo (the bound proxy,
    or the host), so a Russian IP yields a Russian UI — which the uploaders'
    English-text selectors do not expect. Timezone and geolocation still come
    from the launcher's geo lookup.

    Set ``PUBLISH_SHARDX_LANGUAGE=`` (empty) or ``auto`` to let the launcher
    decide.
    """
    locale = (cfg.shardx_language or "").strip()
    if not locale or locale.lower() == "auto":
        return

    accept, languages = _language_values(locale)
    navigator = dict(fingerprint.get("navigator") or {})
    navigator["language"] = locale
    navigator["accept_language"] = accept
    navigator["languages"] = languages
    fingerprint["navigator"] = navigator
    fingerprint["icu_locale"] = locale
    log.info("forced ShardX browser language: %s", locale)


def _apply_noise(fingerprint: Dict[str, Any], cfg: "PublishConfig") -> None:
    """Enable exactly the configured anti-fingerprint noise vectors."""
    vectors = [v for v in (cfg.shardx_noise or []) if v]
    if not vectors:
        return

    on = set(vectors)
    noise = fingerprint.get("noise")
    if not isinstance(noise, dict):
        noise = {}
    for vector in NOISE_VECTORS:
        slot = noise.get(vector)
        if not isinstance(slot, dict):
            slot = {}
        slot["enabled"] = vector in on
        slot.setdefault("seed", 0)
        if vector in on and vector in _NOISE_KNOB:
            knob, soft = _NOISE_KNOB[vector]
            if not slot.get(knob):
                slot[knob] = soft
        noise[vector] = slot
    fingerprint["noise"] = noise
    log.info("applied ShardX noise vectors: %s", ", ".join(vectors))


def _apply_webrtc(fingerprint: Dict[str, Any], cfg: "PublishConfig") -> None:
    """Pin the profile's WebRTC policy when it is not the launcher's ``auto``."""
    policy = (cfg.shardx_webrtc or "auto").strip().lower()
    if policy and policy != "auto":
        fingerprint["webrtc"] = policy


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------


async def _new_fingerprint(api: _LauncherApi, cfg: "PublishConfig") -> Dict[str, Any]:
    """Fetch a fresh fingerprint from the launcher for the configured platform."""
    platform = (cfg.shardx_platform or "").strip()
    path = f"/fingerprint/new/{platform}" if platform else "/fingerprint/new"
    data = await api.request("GET", path)
    fingerprint = (data or {}).get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise BrowserUnavailableError(
            f"the ShardX launcher returned no fingerprint for {path!r}; is the "
            "fingerprint library installed?"
        )
    return fingerprint


async def _create_profile(
    api: _LauncherApi, cfg: "PublishConfig", profile: "Profile"
) -> str:
    """Freeze a fresh fingerprint into a new persistent launcher profile."""
    fingerprint = await _new_fingerprint(api, cfg)
    _apply_language(fingerprint, cfg)
    _apply_noise(fingerprint, cfg)
    _apply_webrtc(fingerprint, cfg)

    body: Dict[str, Any] = {"name": profile.name, "fingerprint": fingerprint}
    if cfg.proxy:
        body["proxy"] = cfg.proxy

    data = await api.request("POST", "/profiles", body=body)
    api_id = str((data or {}).get("id") or "")
    if not api_id:
        raise BrowserUnavailableError(
            f"the ShardX launcher created no profile for '{profile.name}'"
        )
    log.info("created ShardX profile %s for '%s'", api_id, profile.name)
    return api_id


async def _profile_id_by_name(api: _LauncherApi, name: str) -> str:
    """Return the id of the launcher profile named ``name``, or ``""``.

    Lets a profile that already exists in the launcher (created in its UI, or by
    the old SDK) be reused by name instead of being duplicated.
    """
    data = await api.request("GET", "/profiles")
    matches = [
        str(item.get("id") or "")
        for item in (data or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip() == name
    ]
    matches = [match for match in matches if match]
    if len(matches) > 1:
        log.warning(
            "the ShardX Launcher has several profiles named '%s'; using %s",
            name,
            matches[0],
        )
    return matches[0] if matches else ""


async def _resolve_profile(
    api: _LauncherApi, cfg: "PublishConfig", profile: "Profile"
) -> str:
    """Return the launcher profile id behind ``profile``, creating it on first use.

    A saved profile is reopened by the id we stored in ``meta.json`` (same
    fingerprint *and* cookies). When there is no id yet — or the launcher no
    longer knows it — a fresh fingerprint is frozen into a new profile and its
    id is persisted.
    """
    if profile.shardx_id:
        existing = await api.request(
            "GET", f"/profiles/{profile.shardx_id}", missing_ok=True
        )
        if existing is not None:
            log.info(
                "reusing ShardX profile %s for '%s'", profile.shardx_id, profile.name
            )
            return profile.shardx_id
        log.warning(
            "ShardX profile %s is unknown to the launcher; adopting it by name",
            profile.shardx_id,
        )

    adopted = await _profile_id_by_name(api, profile.name)
    if adopted:
        profile.shardx_id = adopted
        profile.save_meta()
        log.info(
            "adopted existing ShardX Launcher profile %s for '%s'",
            adopted,
            profile.name,
        )
        return adopted

    api_id = await _create_profile(api, cfg, profile)
    profile.shardx_id = api_id
    profile.save_meta()
    return api_id


# ---------------------------------------------------------------------------
# Launch / attach
# ---------------------------------------------------------------------------


def _running_cdp_url(stored: Any) -> str:
    """Return the CDP URL of a running profile, or ``""`` when it is stopped."""
    if not isinstance(stored, dict) or not stored.get("running"):
        return ""
    cdp = stored.get("cdp")
    if isinstance(cdp, dict):
        return str(cdp.get("web_socket_debugger_url") or "")
    return ""


async def _start_profile(
    api: _LauncherApi,
    cfg: "PublishConfig",
    profile: "Profile",
    api_id: str,
    headless: bool,
) -> Tuple[str, bool]:
    """Launch ``api_id`` and return ``(cdp_url, started_here)``.

    A profile that is already running is attached to (``started_here`` is
    ``False``) so we do not stop a browser the user opened from the launcher UI.
    """
    stored = await api.request("GET", f"/profiles/{api_id}", missing_ok=True)
    existing = _running_cdp_url(stored)
    if existing:
        log.info("ShardX profile %s is already running; attaching", api_id)
        return existing, False

    data = await api.request(
        "POST", f"/profiles/{api_id}/start", body={"headless": bool(headless)}
    )
    cdp = (data or {}).get("cdp")
    cdp_url = (
        str(cdp.get("web_socket_debugger_url") or "") if isinstance(cdp, dict) else ""
    )
    if not cdp_url:
        reason = (data or {}).get("cdp_error") or "no CDP endpoint was exposed"
        raise BrowserUnavailableError(
            f"shardx: could not launch profile '{profile.name}': {reason}"
        )
    return cdp_url, True


@asynccontextmanager
async def _patchright():
    """Yield a started patchright Playwright, or raise an actionable error."""
    try:
        from patchright.async_api import async_playwright
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise BrowserUnavailableError(
            "patchright is not installed; it is required to drive the launched "
            "ShardX profile over CDP. Install it with 'pip install patchright'."
        ) from exc

    async with async_playwright() as playwright:
        yield playwright


def _default_context(browser, profile: "Profile"):
    """Return the browser's default context (the one carrying the cookies)."""
    contexts = list(getattr(browser, "contexts", []) or [])
    if not contexts:
        raise BrowserUnavailableError(
            f"shardx: profile '{profile.name}' launched without a browser "
            "context; the profile's cookies would not be reachable."
        )
    return contexts[0]


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def open(
    cfg: "PublishConfig", profile: "Profile", *, headless: bool
) -> AsyncIterator[object]:
    """Open ``profile`` in the ShardX launcher and yield its browser context.

    The profile is launched through the launcher API, patchright attaches to the
    returned CDP endpoint, and the profile's default context (which carries its
    cookies) is yielded. A profile we started is stopped again on exit.
    """
    async with _LauncherApi(cfg) as api:
        api_id = await _resolve_profile(api, cfg, profile)
        cdp_url, started_here = await _start_profile(
            api, cfg, profile, api_id, headless
        )
        try:
            async with _patchright() as playwright:
                try:
                    browser = await playwright.chromium.connect_over_cdp(cdp_url)
                except Exception as exc:  # re-raised with context below
                    raise BrowserUnavailableError(
                        f"shardx: could not attach to '{profile.name}' over CDP "
                        f"({cdp_url}): {exc}"
                    ) from exc
                try:
                    context = _default_context(browser, profile)
                    try:
                        context.set_default_timeout(DEFAULT_TIMEOUT_MS)
                    except Exception:  # noqa: BLE001 - not all clients expose it
                        pass
                    yield context
                finally:
                    try:
                        await browser.close()
                    except Exception:  # noqa: BLE001 - teardown must not mask errors
                        pass
        finally:
            if started_here:
                try:
                    await api.request("POST", f"/profiles/{api_id}/stop")
                except Exception as exc:  # noqa: BLE001 - best-effort teardown
                    log.warning("could not stop ShardX profile %s: %s", api_id, exc)


__all__ = ["open"]
