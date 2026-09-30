"""TikTok uploader — browser automation via TikTok Studio.

The publish flow is adapted from ``social-auto-upload``
(``uploader/tk_uploader/main_chrome.py``, the Chrome variant — it is the one that
also supports custom covers and reads back the published video id), rewired onto
this project's browser stack:

* the browser is launched by :mod:`publisher.session` (the ShardX engine) as a
  **persistent** context, so the whole profile's cookies are reused
  automatically — this module only ever receives an open ``BrowserContext``;
* login/auth is handled by the interactive ``publish manual`` command; here we
  only *check* that the stored session is still valid, and fail cleanly if not.

TikTok Studio renders its upload form either inline or inside an ``iframe``
depending on rollout; the code picks the right locator for whichever it finds.

This module exposes the two coroutines required by
:mod:`publisher.platforms.base`: :func:`check` and :func:`upload`.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple

from .. import human
from ..log import log as _default_log
from ..model import Short
from .base import UploadResult, clip_error, page_url

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

NAME = "tiktok"
LABEL = "TikTok"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOME_URL = "https://www.tiktok.com"
LOGIN_URL = "https://www.tiktok.com/login"
UPLOAD_URL = "https://www.tiktok.com/tiktokstudio/upload"
CONTENT_URL = "https://www.tiktok.com/tiktokstudio/content"

# The upload form is sometimes wrapped in this iframe.
_UPLOAD_IFRAME = '[data-tt="Upload_index_iframe"]'

# How long (in 2-second polls) to wait for the file to finish uploading:
# 300 * 2s = 10 min.
_MAX_UPLOAD_POLLS = 300
_POLL_INTERVAL_SECONDS = 2

# Matches TikTok's generated widget class names, e.g.
# ``tiktok-xyz-SelectFormContainer``.
_SELECT_FORM_RE = re.compile(r"tiktok-.*-SelectFormContainer.*")
_VIDEO_ID_RE = re.compile(r"video/(\d+)")


# ---------------------------------------------------------------------------
# Low-level DOM helpers (kept close to the original implementation)
# ---------------------------------------------------------------------------


async def _change_language(page) -> None:
    """Force the TikTok UI into English so the text selectors keep matching."""
    try:
        await page.goto(HOME_URL, wait_until="domcontentloaded")
        await page.wait_for_selector('[data-e2e="nav-more-menu"]', timeout=15000)
        await human.pause(page, 600, 1300)
    except Exception:  # noqa: BLE001 - the UI is left as-is when this fails
        return

    menu = page.locator('[data-e2e="nav-more-menu"]')
    try:
        label = (await menu.text_content()) or ""
        if label.strip() == "More":
            return  # already English
    except Exception:  # noqa: BLE001 - treat an unreadable menu as "not set"
        pass

    try:
        await human.click(page, menu)
        await human.click(page, page.locator('[data-e2e="language-select"]'))
        await human.click(
            page,
            page.locator(
                "#creator-tools-selection-menu-header", has_text="English (US)"
            ),
        )
    except Exception:  # noqa: BLE001 - non-fatal; we simply continue
        pass


async def _choose_base_locator(page):
    """Return the locator scope holding the upload form (frame or body)."""
    if await page.locator(_UPLOAD_IFRAME).count():
        return page.frame_locator(_UPLOAD_IFRAME)
    return page.locator("body")


async def _add_title_and_tags(base, page, title: str, tags) -> None:
    """Type the title and hashtags into TikTok's rich-text caption editor."""
    editor = base.locator("div.public-DraftEditor-content")
    await human.click(page, editor)

    # Clear whatever placeholder/prefill is already there.
    await page.keyboard.press("End")
    await page.keyboard.press("Control+A")
    await page.keyboard.press("Delete")
    await page.keyboard.press("End")
    await human.pause(page, 700, 1400)

    if title.strip():
        await page.keyboard.insert_text(title.strip())
        await human.pause(page, 800, 1500)
        await page.keyboard.press("End")
        await page.keyboard.press("Enter")

    for tag in tags:
        await page.keyboard.press("End")
        await human.pause(page, 700, 1400)
        # Type as "#tag " then remove the trailing space so TikTok registers the
        # hashtag chip without leaving a dangling space behind.
        await page.keyboard.insert_text("#" + tag + " ")
        await page.keyboard.press("Space")
        await human.pause(page, 700, 1400)
        await page.keyboard.press("Backspace")
        await page.keyboard.press("End")


async def _detect_upload_status(base, page, file_path: str, logger) -> bool:
    """Wait until the file finished uploading (the "Post" button enables).

    While waiting, recover once from a stalled/errored transfer by re-selecting
    the file if the site shows its "Select file" error button.
    """
    post_button = base.locator('div.button-group > button:has-text("Post")').first
    retried = False

    for _ in range(_MAX_UPLOAD_POLLS):
        try:
            if await post_button.get_attribute("disabled") is None:
                return True
        except Exception:  # noqa: BLE001 - button not present yet, keep waiting
            pass

        if not retried:
            try:
                error_button = base.locator('button[aria-label="Select file"]')
                if await error_button.count() and await error_button.first.is_visible():
                    logger.warning("TikTok: upload stalled, re-selecting the file")
                    async with page.expect_file_chooser() as fc_info:
                        await human.click(page, error_button.first)
                    chooser = await fc_info.value
                    await chooser.set_files(file_path)
                    retried = True
            except Exception:  # noqa: BLE001 - the retry itself must not break
                pass

        await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    return False


async def _upload_thumbnail(base, page, thumbnail: str) -> None:
    """Set a custom cover image (optional; failures are not fatal)."""
    # Every click glides the cursor onto the target instead of teleporting.
    await human.click(page, base.locator(".cover-container"))
    await human.click(
        page, base.locator(".cover-edit-container", has_text="Upload cover")
    )
    async with page.expect_file_chooser() as fc_info:
        await human.click(page, base.locator(".upload-image-upload-area"))
        chooser = await fc_info.value
        await chooser.set_files(thumbnail)
    await human.click(
        page,
        base.locator("div.cover-edit-panel:not(.hide-panel)").get_by_role(
            "button", name="Confirm"
        ),
    )
    await human.pause(page, 2000, 3500)


async def _set_schedule_time(base, page, schedule_at) -> bool:
    """Enable TikTok's "Schedule" toggle and pick the date/time (best-effort).

    Ported from social-auto-upload (uploader/tk_uploader/main_chrome.py,
    set_schedule_time). Returns False when the control could not be driven, so
    the caller can fail the upload instead of publishing immediately.
    """
    try:
        toggle = base.get_by_label("Schedule")
        await toggle.wait_for(state="visible", timeout=8000)
        await toggle.click(force=True)

        # Some rollouts ask for a confirmation first.
        allow = base.locator("div.TUXButton-content >> text=Allow")
        if await allow.count():
            await allow.first.click()

        picker = base.locator("div.scheduled-picker")
        await picker.locator("div.TUXInputBox").nth(1).click()

        month_title = await base.locator(
            "div.calendar-wrapper span.month-title"
        ).inner_text()
        calendar_month = datetime.strptime(month_title.strip(), "%B").month
        if calendar_month != schedule_at.month:
            arrow = base.locator("div.calendar-wrapper span.arrow")
            await arrow.nth(-1 if calendar_month < schedule_at.month else 0).click()

        days = base.locator("div.calendar-wrapper span.day.valid")
        for i in range(await days.count()):
            element = days.nth(i)
            if (await element.inner_text()).strip() == str(schedule_at.day):
                await element.click()
                break

        await picker.locator("div.TUXInputBox").nth(0).click()
        hour_str = schedule_at.strftime("%H")
        minute_str = f"{int(schedule_at.minute / 5):02d}"
        await page.wait_for_timeout(800)
        await base.locator(
            f"span.tiktok-timepicker-left:has-text('{hour_str}')"
        ).click()
        await page.wait_for_timeout(800)
        await base.locator(
            f"span.tiktok-timepicker-right:has-text('{minute_str}')"
        ).click()
        await page.wait_for_timeout(800)
        return True
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        _default_log.warning("TikTok: could not set the schedule (%s)", clip_error(exc))
        return False


async def _click_publish(base, page, logger) -> bool:
    """Click "Post" and wait for the confirmation (redirect to the content list)."""
    publish_button = base.locator("div.button-group button").nth(0)
    for attempt in range(60):
        try:
            if await publish_button.count():
                # The first press is the real one: glide the cursor onto "Post"
                # and click like a person would. Later retries (only reached when
                # the click did not register) stay quick.
                if attempt == 0:
                    await human.click(page, publish_button)
                else:
                    await publish_button.click()
            try:
                await page.wait_for_url(CONTENT_URL, timeout=3000)
                return True
            except Exception:  # noqa: BLE001 - not navigated yet, keep trying
                pass
            # Some flows confirm via a modal instead of a navigation.
            if await page.locator("div.common-modal-confirm-modal").count():
                return True
        except Exception as exc:  # noqa: BLE001 - retry the click
            logger.debug("TikTok: publish retry (%s)", clip_error(exc))
        # Space out the retries with a short, human-like pause rather than a
        # metronomic half-second interval.
        await human.pause(page, 600, 1200)
    return False


async def _get_last_video_id(base, page) -> Tuple[str, str]:
    """Read the newest published video's id and URL from the content list."""
    try:
        await page.wait_for_selector(
            'div[data-tt="components_PostTable_Container"]', timeout=15000
        )
        links = base.locator(
            'div[data-tt="components_PostTable_Container"] '
            'div[data-tt="components_PostInfoCell_Container"] a'
        )
        if await links.count():
            href = await links.nth(0).get_attribute("href") or ""
            match = _VIDEO_ID_RE.search(href)
            video_id = match.group(1) if match else ""
            return video_id, href
    except Exception:  # noqa: BLE001 - the id/url is only informational
        pass
    return "", ""


# ---------------------------------------------------------------------------
# Login check
# ---------------------------------------------------------------------------


async def check(context, cfg, logger=None) -> bool:
    """Return whether the profile behind ``context`` is logged into TikTok.

    Opens the Studio upload page and treats the session as invalid when it is
    bounced to the login page or when the site shows its expired-session widget.
    """
    log = logger or _default_log
    page = await context.new_page()
    try:
        await page.goto(f"{UPLOAD_URL}?lang=en", wait_until="domcontentloaded")
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:  # noqa: BLE001 - networkidle is only a nicety
            pass

        url = page_url(page)
        if "/login" in url.lower():
            log.debug("TikTok: not logged in (redirected to login)")
            return False

        try:
            selects = await page.query_selector_all("select")
            for element in selects:
                class_name = await element.get_attribute("class") or ""
                if _SELECT_FORM_RE.match(class_name):
                    log.debug("TikTok: cookie expired (login form container shown)")
                    return False
        except Exception:  # noqa: BLE001 - absence of the widget means "ok"
            pass

        return True
    except Exception as exc:  # noqa: BLE001 - a failed check is simply "no"
        log.debug("TikTok: login check failed: %s", clip_error(exc))
        return False
    finally:
        try:
            await page.close()
        except Exception:  # noqa: BLE001 - best effort
            pass


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


async def upload(
    context, short: Short, cfg, logger=None, *, schedule_at=None
) -> UploadResult:
    """Publish one clip to TikTok and return the outcome.

    When ``schedule_at`` is given, the clip is scheduled for that date/time
    instead of being posted immediately.
    """
    log = logger or _default_log
    file_path = short.file
    if not file_path or not Path(file_path).exists():
        return UploadResult.failure(f"video file not found: {file_path}")

    tags = short.effective_tags(cfg.tags)
    page = await context.new_page()
    page.set_default_timeout(60_000)
    try:
        # 1) English UI, then the upload form -----------------------------------
        await _change_language(page)
        await page.goto(UPLOAD_URL, wait_until="domcontentloaded")
        # Let the Studio page load, then pause before reacting to it.
        await human.wait_ready(page)

        # 2) the form may be inline or inside an iframe -------------------------
        try:
            await page.wait_for_selector(
                f"{_UPLOAD_IFRAME}, div.upload-container", timeout=30_000
            )
        except Exception:  # noqa: BLE001 - proceed and let the next step fail
            pass

        base = await _choose_base_locator(page)

        # 3) select the video file ----------------------------------------------
        upload_button = base.locator('button:has-text("Select video"):visible')
        await upload_button.wait_for(state="visible", timeout=60_000)
        await human.pause(page, 700, 1500)  # a beat before picking the file
        async with page.expect_file_chooser() as fc_info:
            # Glide the cursor onto "Select video" (multi-step mouse move +
            # jittered click) before the file dialog opens, instead of an
            # instant teleport-and-click.
            await human.click(page, upload_button)
        file_chooser = await fc_info.value
        await file_chooser.set_files(file_path)

        # 4) caption: title + hashtags ------------------------------------------
        await _add_title_and_tags(base, page, short.title, tags)

        # 5) wait for the transfer to finish ------------------------------------
        if not await _detect_upload_status(base, page, file_path, log):
            log.warning(
                "TikTok: upload did not confirm within the timeout; "
                "attempting to publish anyway"
            )

        # 6) custom cover (optional) --------------------------------------------
        if short.thumbnail and Path(short.thumbnail).exists():
            try:
                await _upload_thumbnail(base, page, short.thumbnail)
            except Exception as exc:  # noqa: BLE001 - a cover is optional
                log.warning("TikTok: cover upload skipped (%s)", clip_error(exc))

        # 6.5) schedule (instead of publish-now) ---------------------------------
        if schedule_at is not None:
            if not await _set_schedule_time(base, page, schedule_at):
                return UploadResult.failure(
                    "could not set the TikTok schedule (schedule control not found)"
                )

        # 7) publish -------------------------------------------------------------
        if not await _click_publish(base, page, log):
            return UploadResult.failure(
                "publish did not confirm (no redirect to the content list)"
            )

        # 8) read back the published video id -----------------------------------
        video_id, video_url = await _get_last_video_id(base, page)
        return UploadResult.success(url=video_url, video_id=video_id)

    except Exception as exc:  # noqa: BLE001 - reported to the caller
        log.debug("TikTok upload failed: %s", exc)
        return UploadResult.failure(clip_error(exc))
    finally:
        try:
            await page.close()
        except Exception:  # noqa: BLE001 - best effort
            pass


__all__ = ["NAME", "LABEL", "check", "upload", "UPLOAD_URL", "CONTENT_URL"]
