"""The contract every platform uploader implements.

Each platform lives in its own module (``youtube.py``, ``tiktok.py``) and
exposes three things:

* ``NAME`` — the stable identifier used on the command line (``youtube``);
* ``LABEL`` — a human-friendly name for the report (``YouTube``);
* two coroutines: ``check(context, cfg, logger)`` and
  ``upload(context, short, cfg, logger, schedule_at=...)``.

``check`` answers "is this profile still logged into this platform?" without
changing anything. ``upload`` publishes one :class:`~publisher.model.Short` and
returns an :class:`UploadResult`; passing ``schedule_at`` asks the platform to
schedule the clip for that time instead of publishing it now.

The contract is intentionally tiny: it keeps the browser lifecycle and the
profile selection inside :mod:`publisher.publish`, so a platform module only has
to know how to drive the site's own UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from ..model import Short


@dataclass
class UploadResult:
    """The outcome of publishing a single short to a single platform."""

    ok: bool
    #: Public URL of the published video, when the site exposes one.
    url: str = ""
    #: The site's own id for the video (TikTok numeric id, YouTube video id).
    video_id: str = ""
    #: Human-readable failure reason (empty when ``ok`` is true).
    error: str = ""
    #: True when the failure was an expired/missing login, so the caller can
    #: skip this platform's remaining clips instead of retrying each one.
    auth_required: bool = False

    @classmethod
    def success(cls, url: str = "", video_id: str = "") -> "UploadResult":
        return cls(ok=True, url=url, video_id=video_id)

    @classmethod
    def failure(cls, error: str, *, auth_required: bool = False) -> "UploadResult":
        return cls(
            ok=False,
            error=str(error).strip() or "unknown error",
            auth_required=auth_required,
        )


@runtime_checkable
class Platform(Protocol):
    """Structural type that a platform module satisfies."""

    NAME: str
    LABEL: str

    async def check(self, context, cfg, logger=None) -> bool:
        """Return whether the profile behind ``context`` is still logged in."""
        ...

    async def upload(
        self,
        context,
        short: Short,
        cfg,
        logger=None,
        *,
        schedule_at: Optional[datetime] = None,
    ) -> UploadResult:
        """Publish ``short`` and report the outcome.

        When ``schedule_at`` is given the platform schedules the clip for that
        date/time instead of publishing it immediately; ``None`` keeps the
        "publish now" behaviour.
        """
        ...


def page_url(page) -> str:
    """Best-effort current URL of a page (never raises)."""
    try:
        return page.url or ""
    except Exception:  # noqa: BLE001 - reporting must not fail
        return ""


def clip_error(exc: BaseException, limit: int = 400) -> str:
    """Turn an exception into a short, single-line message for the report."""
    text = str(exc).strip() or exc.__class__.__name__
    text = text.replace("\n", " ").strip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def trim(text: Optional[str], limit: int) -> str:
    """Trim ``text`` to ``limit`` characters (never ``None``)."""
    return (text or "").strip()[:limit]


__all__ = [
    "UploadResult",
    "Platform",
    "page_url",
    "clip_error",
    "trim",
]
