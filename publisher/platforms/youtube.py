"""YouTube uploader — browser automation via YouTube Studio.

The publish flow is adapted from ``social-auto-upload``
(``uploader/youtube_uploader/main.py``), keeping its DOM interaction step by
step, but rewired onto this project's browser stack:

* the browser is launched by :mod:`publisher.session` (the ShardX engine) as a
  **persistent** context, so the whole profile's cookies are reused
  automatically — this module only ever receives an open ``BrowserContext``;
* login/auth is handled by the interactive ``publish manual`` command; here we
  only *check* that the stored session is still valid, and fail cleanly if not.

An official Data API is deliberately not used: videos uploaded through an
*unaudited* API project are force-locked to private, whereas driving YouTube
Studio publishes public videos directly — same as every other uploader in the
original project.

This module exposes the two coroutines required by
:mod:`publisher.platforms.base`: :func:`check` and :func:`upload`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .. import human
from ..log import log as _default_log
from ..model import Short
from .base import UploadResult, clip_error, page_url, trim

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

NAME = "youtube"
LABEL = "YouTube"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUDIO_URL = "https://studio.youtube.com"
UPLOAD_URL = "https://www.youtube.com/upload"

# Maps the config's visibility value to YouTube's radio-button ``name``.
VISIBILITY = {"public": "PUBLIC", "unlisted": "UNLISTED", "private": "PRIVATE"}

# Field limits enforced by YouTube Studio.
TITLE_LIMIT = 100
DESCRIPTION_LIMIT = 5000
TAGS_LIMIT = 500

# How long (in 5-second polls) to wait for the upload to reach 100% before
# attempting to publish anyway: 360 * 5s = 30 min.
_MAX_UPLOAD_POLLS = 360
_POLL_INTERVAL_MS = 5000

_YOUTUBE_URL_RE = re.compile(r"(?:youtu\.be/|watch\?v=)([A-Za-z0-9_-]{6,})")


# ---------------------------------------------------------------------------
# Low-level DOM helpers (kept close to the original implementation)
# ---------------------------------------------------------------------------


async def _dismiss_autocomplete(page) -> None:
    """Close the ``#``/``@`` autocomplete dropdown that covers the buttons.

    Blur the focused element first; if a dropdown is still visible, press
    Escape — but only when one is actually open, so we never accidentally close
    the whole upload dialog.
    """
    try:
        await page.evaluate(
            "() => { const a = document.activeElement; if (a && a.blur) a.blur(); }"
        )
    except Exception:  # noqa: BLE001 - best effort
        pass
    try:
        dropdown = page.locator("tp-yt-iron-dropdown:visible")
        if await dropdown.count() > 0:
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(200)
    except Exception:  # noqa: BLE001 - best effort
        pass


async def _fill_editable(page, selector: str, text: str) -> None:
    """Fill a YouTube Studio contenteditable box (title / description).

    Uses ``fill()`` to write the whole value at once instead of typing it
    character by character: a ``#`` in the text (e.g. ``#Shorts``) would
    otherwise pop the topic-autocomplete dropdown on every keystroke and cover
    the "Next"/"Publish" buttons. Falls back to slow typing only when the
    element refuses ``fill()``.
    """
    box = page.locator(selector).first
    await box.wait_for(state="visible", timeout=30000)
    # Glide the cursor onto the field (multi-step mouse move + jittered click)
    # rather than teleporting to its centre and clicking instantly.
    await human.click(page, box, timeout=30000)
    await page.keyboard.press("Control+A")
    await page.keyboard.press("Delete")
    try:
        await box.fill(text)
    except Exception:  # noqa: BLE001 - some contenteditable nodes reject fill()
        await human.type_text(box, text)
    await human.pause(page, 500, 1000)
    await _dismiss_autocomplete(page)


async def _click_if_present(page, selector: str, timeout: int = 4000) -> bool:
    """Click the first element matching ``selector`` if it shows up in time.

    The click goes through :func:`human.click`, so the element is waited for,
    hovered and only then clicked — instead of being hit the instant it exists.
    """
    try:
        element = page.locator(selector).first
        await human.click(page, element, timeout=timeout)
        return True
    except Exception:  # noqa: BLE001 - absence is a valid outcome
        return False


async def _wait_upload_complete(page, max_polls: int = _MAX_UPLOAD_POLLS) -> bool:
    """Wait for the browser upload to move from ``X%`` to done before publishing.

    Browser uploads only finish while the window stays open: publishing and
    closing the browser mid-transfer cuts the upload off (e.g. stuck at 76%).
    We consider the transfer finished as soon as the progress label stops
    saying "uploading" (processing/checks/complete take over).
    """
    last = ""
    for _ in range(max_polls):
        text = ""
        for selector in (
            ".progress-label",
            "span.progress-label",
            "ytcp-video-upload-progress",
        ):
            loc = page.locator(selector).first
            try:
                if await loc.count():
                    text = (await loc.inner_text()).strip()
                    if text:
                        break
            except Exception:  # noqa: BLE001 - keep polling
                pass

        if text:
            if any(
                key in text
                for key in (
                    "处理",
                    "检查",
                    "上传完成",
                    "已上传",
                    "Processing",
                    "complete",
                    "Checks",
                    "Finished",
                )
            ):
                return True
            if text != last:
                last = text

        await page.wait_for_timeout(_POLL_INTERVAL_MS)

    return False


# ---------------------------------------------------------------------------
# Login check
# ---------------------------------------------------------------------------


async def check(context, cfg, logger=None) -> bool:
    """Return whether the profile behind ``context`` is logged into YouTube.

    Opens YouTube Studio and treats the session as valid when it lands on a
    ``/channel/...`` page instead of being bounced to a Google sign-in page.
    """
    log = logger or _default_log
    page = await context.new_page()
    try:
        await page.goto(STUDIO_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        url = page_url(page)
        if "accounts.google.com" in url or "/signin" in url.lower():
            log.debug("YouTube: not logged in (redirected to sign-in)")
            return False
        return "/channel/" in url
    except Exception as exc:  # noqa: BLE001 - a failed check is simply "no"
        log.debug("YouTube: login check failed: %s", clip_error(exc))
        return False
    finally:
        try:
            await page.close()
        except Exception:  # noqa: BLE001 - best effort
            pass


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


async def _upload_playlist(page, playlist: str) -> None:
    """Add the video to ``playlist``, creating it when it does not exist."""
    try:
        await _click_if_present(
            page,
            "#basics ytcp-text-dropdown-trigger, "
            "ytcp-video-metadata-playlists ytcp-dropdown-trigger",
            8000,
        )
        await human.pause(page, 900, 1600)
        existing = page.locator(
            f"tp-yt-paper-checkbox:has-text('{playlist}'), "
            f"ytcp-checkbox-group:has-text('{playlist}')"
        ).first
        if await existing.count():
            await human.click(page, existing)
        else:
            if await _click_if_present(
                page,
                "ytcp-button:has-text('New playlist'), "
                "ytcp-button:has-text('创建播放列表')",
                4000,
            ):
                await human.pause(page, 600, 1200)
                await _click_if_present(
                    page,
                    "tp-yt-paper-item:has-text('New playlist'), "
                    "tp-yt-paper-item:has-text('新建播放列表')",
                    3000,
                )
                title_box = page.locator(
                    "ytcp-playlist-metadata-editor #textbox, #create-playlist-form #textbox"
                ).first
                if await title_box.count():
                    await human.click(page, title_box)
                    await title_box.type(playlist, delay=6)
                    await _click_if_present(
                        page,
                        "ytcp-button#create-button, "
                        "tp-yt-paper-dialog ytcp-button:has-text('Create'), "
                        "tp-yt-paper-dialog ytcp-button:has-text('创建')",
                        4000,
                    )
    finally:
        # The dialog must be closed, otherwise it covers the following steps.
        await _click_if_present(
            page,
            "ytcp-playlist-dialog #save-button, "
            "ytcp-button:has-text('Done'), ytcp-button:has-text('完成')",
            3000,
        )
        await page.keyboard.press("Escape")
        await human.pause(page, 500, 900)


async def _upload_thumbnail(page, thumbnail: str) -> None:
    """Upload a custom cover image (skipped, not fatal, when it fails)."""
    try:
        thumb_input = page.locator(
            "#file-loader input[type='file'], ytcp-thumbnail-uploader input[type='file']"
        ).first
        await thumb_input.wait_for(state="attached", timeout=20000)
        await thumb_input.set_input_files(thumbnail)
        await human.pause(page, 1500, 2600)
    except Exception:  # noqa: BLE001 - a cover is optional
        pass


# The "Schedule" visibility option. Structural selectors first (language
# independent), then English/Russian text — the Studio UI may be localized.
_SCHEDULE_OPTION = (
    "tp-yt-paper-radio-button[name='SCHEDULE'], "
    "#schedule-radio-button, "
    "tp-yt-paper-radio-button:has-text('Schedule'), "
    "tp-yt-paper-radio-button:has-text('Запланировать')"
)


async def _set_schedule(page, schedule_at) -> bool:
    """Choose "Schedule" and set the publish date/time (YouTube Studio).

    Structure taken from a captured dump of the visibility step:

    * the option is a ``tp-yt-paper-radio-button`` (named ``SCHEDULE``);
    * selecting it renders a ``ytcp-visibility-scheduler`` holding a
      ``ytcp-datetime-picker``;
    * the date is a dropdown trigger (``#datepicker-trigger``) that opens a
      calendar; the time is a text input (``#time-of-day-container input``).

    The opened calendar was not part of the capture, so it is driven with generic
    cell selectors plus a JS fallback that writes the scheduler's ``date`` model.
    Returns ``False`` when the schedule could not be set, so the caller fails the
    upload instead of publishing immediately.
    """
    option = page.locator(_SCHEDULE_OPTION).first
    try:
        await option.wait_for(state="visible", timeout=8000)
        await human.click(page, option, timeout=8000)
    except Exception:  # noqa: BLE001 - reported to the caller
        return False

    try:
        await page.locator("ytcp-visibility-scheduler").first.wait_for(
            state="visible", timeout=8000
        )
    except Exception:  # noqa: BLE001 - reported to the caller
        return False
    await human.pause(page, 500, 900)

    if not await _set_schedule_date(page, schedule_at):
        return False

    try:
        time_input = page.locator(
            "ytcp-datetime-picker #time-of-day-container input"
        ).first
        await human.click(page, time_input)
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Delete")
        # Studio's date/time picker expects an English-style "10:00 AM".
        await time_input.type(schedule_at.strftime("%I:%M %p"), delay=40)
        await page.keyboard.press("Enter")
        await human.pause(page, 400, 800)
    except Exception:  # noqa: BLE001 - reported to the caller
        return False
    return True


async def _set_schedule_date(page, schedule_at) -> bool:
    """Pick the day in the schedule date dropdown (with a JS model fallback)."""
    try:
        await human.click(page, page.locator("#datepicker-trigger").first)
        await human.pause(page, 500, 900)
        day = schedule_at.day
        for selector in (
            f"ytcp-date-picker .day:not([disabled]):text-is('{day}')",
            f"[role='gridcell']:text-is('{day}')",
            f"ytcp-date-picker span.day:text-is('{day}')",
        ):
            cell = page.locator(selector).first
            if await cell.count():
                await human.click(page, cell)
                await human.pause(page, 300, 700)
                return True
    except Exception:  # noqa: BLE001 - fall through to the JS fallback
        pass

    # Fallback: write the scheduler's ``date`` model (see the captured markup:
    # {"date":{"year":..,"month":..,"day":..}, ...}).
    try:
        await page.keyboard.press("Escape")
        ok = await page.evaluate(
            """([year, month, day]) => {
                const el = document.querySelector('ytcp-visibility-scheduler');
                if (!el) return false;
                if ('date' in el) el.date = {year: year, month: month, day: day};
                const picker = el.querySelector('ytcp-datetime-picker');
                if (picker && 'date' in picker) {
                    picker.date = {year: year, month: month, day: day};
                }
                return true;
            }""",
            [schedule_at.year, schedule_at.month, schedule_at.day],
        )
        await human.pause(page, 300, 700)
        return bool(ok)
    except Exception:  # noqa: BLE001 - reported to the caller
        return False


async def upload(
    context, short: Short, cfg, logger=None, *, schedule_at=None
) -> UploadResult:
    """Publish one clip to YouTube and return the outcome.

    When ``schedule_at`` is given, the clip is scheduled for that date/time
    instead of being published immediately.
    """
    log = logger or _default_log
    file_path = short.file
    if not file_path or not Path(file_path).exists():
        return UploadResult.failure(f"video file not found: {file_path}")

    page = await context.new_page()
    page.set_default_timeout(60_000)
    try:
        await page.goto(UPLOAD_URL, wait_until="domcontentloaded")
        # Let the Studio page finish loading, then take a short human pause
        # before reacting to it, instead of a fixed three-second sleep.
        await human.wait_ready(page)
        current = page_url(page)
        if "accounts.google.com" in current or "signin" in current.lower():
            return UploadResult.failure(
                "YouTube login expired — run: "
                "python main.py publish manual --profile <name>",
                auth_required=True,
            )

        # 1) choose the video file -------------------------------------------------
        file_input = page.locator('input[type="file"]').first
        await file_input.wait_for(state="attached", timeout=60000)
        await human.pause(page, 700, 1500)  # a beat before picking the file
        await file_input.set_input_files(file_path)

        # 2) wait for the details dialog ------------------------------------------
        await page.locator("#title-textarea").wait_for(state="visible", timeout=120000)

        # 3) title ----------------------------------------------------------------
        await _fill_editable(
            page,
            "#title-textarea #textbox",
            trim(short.title or short.name, TITLE_LIMIT),
        )

        # 4) description ----------------------------------------------------------
        description = trim(short.description, DESCRIPTION_LIMIT)
        if description:
            await _fill_editable(page, "#description-textarea #textbox", description)

        # 5) thumbnail (optional) -------------------------------------------------
        if short.thumbnail and Path(short.thumbnail).exists():
            await _upload_thumbnail(page, short.thumbnail)

        # 6) playlist (optional) --------------------------------------------------
        if cfg.playlist:
            await _upload_playlist(page, cfg.playlist)

        # 7) "not made for kids" (required) ---------------------------------------
        if not await _click_if_present(
            page, "tp-yt-paper-radio-button[name='VIDEO_MADE_FOR_KIDS_NOT_MFK']", 10000
        ):
            await _click_if_present(
                page,
                "tp-yt-paper-radio-button:has-text('not made for kids'), "
                "tp-yt-paper-radio-button:has-text('不是面向儿童')",
                6000,
            )

        # 8) tags (inside "show more") --------------------------------------------
        tags: List[str] = short.effective_tags(cfg.tags)
        if tags:
            try:
                await _click_if_present(page, "#toggle-button", 6000)
                await human.pause(page, 600, 1200)
                tag_input = page.locator(
                    "#tags-container #text-input, "
                    "ytcp-form-input-container#tags-container input"
                ).first
                await human.click(page, tag_input)
                await tag_input.type(",".join(tags)[:TAGS_LIMIT] + ",", delay=4)
            except Exception:  # noqa: BLE001 - tags are optional
                pass

        # 9) click Next until the visibility step becomes active -------------------
        for _ in range(5):
            vis = page.locator("tp-yt-paper-radio-button[name='PUBLIC']")
            if await vis.count() and await vis.first.is_visible():
                break
            if not await _click_if_present(page, "#next-button", 6000):
                await human.pause(page, 900, 1600)
            await human.pause(page, 800, 1400)

        # 10) visibility / schedule -----------------------------------------------
        if schedule_at is not None:
            if not await _set_schedule(page, schedule_at):
                return UploadResult.failure(
                    "could not set the YouTube schedule (schedule picker not found)"
                )
        else:
            desired = VISIBILITY.get(str(cfg.visibility).lower(), "PUBLIC")
            await _click_if_present(
                page, f"tp-yt-paper-radio-button[name='{desired}']", 10000
            )

        # 10.5) let the transfer actually finish before publishing -----------------
        await _wait_upload_complete(page)

        # 11) publish -------------------------------------------------------------
        await human.pause(page, 900, 1800)
        if not await _click_if_present(page, "#done-button", 15000):
            return UploadResult.failure(
                "publish button (#done-button) not found — the upload may not "
                "have reached a publishable state"
            )

        await human.settle(page, timeout=6000)
        await human.pause(page, 2500, 4000)
        video_url = ""
        video_id = ""
        try:
            link = page.locator("a[href*='youtu.be'], a[href*='watch?v=']").first
            if await link.count():
                video_url = await link.get_attribute("href") or ""
        except Exception:  # noqa: BLE001 - the url is only informational
            pass

        if video_url:
            match = _YOUTUBE_URL_RE.search(video_url)
            if match:
                video_id = match.group(1)

        await _click_if_present(
            page,
            "ytcp-button:has-text('Close'), ytcp-button:has-text('关闭'), #close-button",
            8000,
        )

        return UploadResult.success(url=video_url, video_id=video_id)

    except Exception as exc:  # noqa: BLE001 - reported to the caller
        log.debug("YouTube upload failed: %s", exc)
        return UploadResult.failure(clip_error(exc))
    finally:
        try:
            await page.close()
        except Exception:  # noqa: BLE001 - best effort
            pass


__all__ = ["NAME", "LABEL", "check", "upload", "STUDIO_URL", "UPLOAD_URL"]
