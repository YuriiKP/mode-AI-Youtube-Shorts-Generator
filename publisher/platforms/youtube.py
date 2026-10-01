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


# When an upload cannot proceed, YouTube Studio raises an *error scrim* over the
# dialog: ``ytcp-uploads-dialog`` gains ``has-error`` / ``show-error-scrim`` and
# shows ``#error-block`` / ``#error-message`` ("Oops, something went wrong.")
# together with a specific reason in ``.error-short`` and ``.error-details``
# (e.g. "Daily upload limit reached" / "Upload more videos daily after a one-time
# verification or wait 24 hours."). That scrim swallows every click, so without
# this the flow merely times out looking for a button it can no longer reach —
# reading the reason makes the failure actionable instead.
_DIALOG_ERROR_STATE_JS = """() => {
    const host = document.querySelector('ytcp-uploads-dialog');
    if (!host) return { text: '', specific: false };
    const clean = (el) => el ? (el.textContent || '').replace(/\\s+/g, ' ').trim() : '';
    // Several of these selectors match more than one node, and the first match
    // can be an empty placeholder (e.g. an ``.error-details`` carrying no text),
    // so take the first *non-empty* match of each.
    const firstText = (selector) => {
        for (const el of host.querySelectorAll(selector)) {
            const text = clean(el);
            if (text) return text;
        }
        return '';
    };
    const short = firstText('.error-short');
    const details = firstText('.error-details');
    const generic = firstText('#error-message');
    const block = host.querySelector('#error-block, .error-area');
    const flagged = host.hasAttribute('has-error')
        || host.hasAttribute('show-error-scrim') || !!block;
    if (!flagged) return { text: '', specific: false };
    const parts = [];
    if (short) parts.push(short);
    if (generic && !parts.includes(generic)) parts.push(generic);
    if (details && !parts.includes(details)) parts.push(details);
    if (!parts.length && block) {
        const text = clean(block);
        if (text) parts.push(text);
    }
    // ``specific`` is true once Studio has filled in the actual reason
    // (``.error-short``); before that only the generic "#error-message" is up.
    return { text: parts.join(' — '), specific: !!short };
}"""


async def _dialog_error_state(page) -> dict:
    """``{"text": str, "specific": bool}`` for the upload dialog's error, if any."""
    try:
        data = await page.evaluate(_DIALOG_ERROR_STATE_JS)
    except Exception:  # noqa: BLE001 - a missing error is not an error
        return {"text": "", "specific": False}
    if not isinstance(data, dict):
        return {"text": "", "specific": False}
    return {
        "text": str(data.get("text") or "").strip(),
        "specific": bool(data.get("specific")),
    }


async def _dialog_error(page) -> str:
    """The upload dialog's error text, or ``""`` while the dialog is healthy.

    Studio shows this when it refuses the upload (daily limit, processing
    failure, …); it is surfaced by the caller so the upload fails with a real
    reason rather than an unexplained missing-button timeout.
    """
    return (await _dialog_error_state(page))["text"]


async def _wait_dialog_error(page, attempts: int = 8, settle_reads: int = 3) -> str:
    """Poll for the dialog's error, preferring the *specific* reason.

    Studio paints the generic ``#error-message`` first and only then fills in
    ``.error-short`` / ``.error-details`` with the actual reason, so a single look
    can catch just "Oops, something went wrong.".

    Once the specific reason appears, a few more reads are taken and the longest
    message seen is kept, so the guidance line (``.error-details``, e.g.
    "…after a one-time verification or wait 24 hours.") is included when it
    renders. A healthy dialog is not held up: two empty reads in a row return
    immediately.
    """
    best = ""
    empty_reads = 0
    for _ in range(max(1, attempts)):
        state = await _dialog_error_state(page)
        text = state["text"]
        if text:
            empty_reads = 0
            if len(text) > len(best):
                best = text
            if state["specific"]:
                # The specific reason is up; give ``.error-details`` (the longer
                # guidance line) a few more reads to render, keeping whichever
                # message is the most complete.
                for _ in range(max(0, settle_reads)):
                    await human.pause(page, 300, 600)
                    enriched = await _dialog_error_state(page)
                    enriched_text = enriched["text"]
                    if len(enriched_text) > len(best):
                        best = enriched_text
                return best
        else:
            empty_reads += 1
            if empty_reads >= 2:
                return best
        await human.pause(page, 300, 600)
    return best


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


# ---------------------------------------------------------------------------
# Scheduling (the "Schedule" card of the visibility step)
# ---------------------------------------------------------------------------
#
# The visibility step of the current Studio UI is a two-card accordion:
#
#   #first-container   "Save or publish"  -> the PRIVATE/UNLISTED/PUBLIC radios
#   #second-container  "Schedule"         -> collapsed behind an expand button
#
# Selecting one card empties the other out of the DOM, so there is no longer a
# ``tp-yt-paper-radio-button[name='SCHEDULE']`` to click — the old selector could
# never match, which is exactly why scheduling used to fail with "schedule
# picker not found". Expanding the Schedule card reveals ``ytcp-visibility-scheduler``
# with an ``ytcp-datetime-picker``: a date dropdown (``#datepicker-trigger``) whose
# calendar renders as ``ytcp-scrollable-calendar`` → ``.calendar-month`` (each with
# a ``.calendar-month-label`` like "Nov 2026") → ``.calendar-week`` →
# ``span.calendar-day``, plus a time field (``#time-of-day-container input``).
_SCHEDULE_CARD = "#second-container"
_SCHEDULE_CARD_EXPAND = "#second-container-expand-button"
_DATE_TRIGGER = "#datepicker-trigger"
_CALENDAR = "ytcp-scrollable-calendar"
_CALENDAR_MONTH = f"{_CALENDAR} .calendar-month"
_CALENDAR_DAY = ".calendar-day"
_TIME_INPUT = "#time-of-day-container input"

#: True when the Schedule card is the selected one (its privacy radios replaced
#: by the scheduler).
_SCHEDULE_SELECTED_JS = (
    "() => { const el = document.querySelector('ytcp-video-visibility-select');"
    " return !!el && el.getAttribute('second-container-style') === 'selected'; }"
)

#: True when the date dropdown is actually open. A rect test is used because the
#: calendar sits in a ``position: fixed`` dialog (so ``offsetParent`` is null) and
#: stays in the DOM, collapsed to zero size, once it is dismissed.
_CALENDAR_OPEN_JS = """() => {
    const dialog = document.querySelector('ytcp-date-picker tp-yt-paper-dialog#dialog');
    if (!dialog) return false;
    const rect = dialog.getBoundingClientRect();
    return rect.width > 2 && rect.height > 2;
}"""


def _expected_date_text(schedule_at) -> str:
    """How the date dropdown renders a date, e.g. ``Nov 5, 2026``."""
    return f"{schedule_at.strftime('%b')} {schedule_at.day}, {schedule_at.year}"


def _month_label(schedule_at) -> str:
    """How the calendar labels a month, e.g. ``Nov 2026``."""
    return schedule_at.strftime("%b %Y")


def _expected_time_text(schedule_at) -> str:
    """How the time field renders a time, e.g. ``06:00 AM``."""
    return schedule_at.strftime("%I:%M %p")


def _normalize_spaces(text: str) -> str:
    """Fold the no-break spaces Studio uses in dates/times into plain spaces."""
    return (text or "").replace("\u202f", " ").replace("\xa0", " ").strip()


async def _page_truthy(page, script: str) -> bool:
    """Best-effort boolean page evaluation (``False`` when the page is gone)."""
    try:
        return bool(await page.evaluate(script))
    except Exception:  # noqa: BLE001 - a dead page is simply "not true"
        return False


async def _open_schedule_card(page) -> bool:
    """Expand the "Schedule" card of the visibility step.

    Clicks the card's expand button (falling back to the card body) and waits for
    the accordion to select the second container, so the scheduler replaces the
    privacy radios.
    """
    if await _page_truthy(page, _SCHEDULE_SELECTED_JS):
        return True
    for selector in (_SCHEDULE_CARD_EXPAND, _SCHEDULE_CARD):
        try:
            button = page.locator(selector).first
            if not await button.count():
                continue
            await human.click(page, button, timeout=8000)
        except Exception:  # noqa: BLE001 - try the card body next
            continue
        for _ in range(10):
            await human.pause(page, 200, 400)
            if await _page_truthy(page, _SCHEDULE_SELECTED_JS):
                return True
    return await _page_truthy(page, _SCHEDULE_SELECTED_JS)


async def _open_date_picker(page) -> bool:
    """Open the schedule date dropdown and wait for the calendar to appear."""
    if await _page_truthy(page, _CALENDAR_OPEN_JS):
        return True
    try:
        await human.click(page, page.locator(_DATE_TRIGGER).first, timeout=8000)
    except Exception:  # noqa: BLE001 - reported to the caller
        return False
    for _ in range(10):
        await human.pause(page, 200, 400)
        if await _page_truthy(page, _CALENDAR_OPEN_JS):
            return True
    return False


async def _month_state(page, label: str) -> dict:
    """Where the month ``label`` (e.g. ``Nov 2026``) sits inside the calendar."""
    script = """(label) => {
        const dialog = document.querySelector('ytcp-date-picker tp-yt-paper-dialog#dialog');
        const list = document.querySelector('ytcp-scrollable-calendar #calendar-main');
        const months = [...document.querySelectorAll('ytcp-scrollable-calendar .calendar-month')];
        const labelOf = (m) => {
            const el = m.querySelector('.calendar-month-label');
            return el ? (el.textContent || '').trim() : '';
        };
        const state = {
            rendered: months.map(labelOf),
            found: false, visible: false,
            monthTop: null, monthBottom: null,
            dialogTop: null, dialogBottom: null,
            listScrollTop: list ? Math.round(list.scrollTop) : null,
            listClientHeight: list ? Math.round(list.clientHeight) : null,
        };
        if (!dialog) return state;
        const dialogRect = dialog.getBoundingClientRect();
        state.dialogTop = Math.round(dialogRect.top);
        state.dialogBottom = Math.round(dialogRect.bottom);
        const month = months.find((m) => labelOf(m) === label);
        if (!month) return state;
        const rect = month.getBoundingClientRect();
        state.found = true;
        state.monthTop = Math.round(rect.top);
        state.monthBottom = Math.round(rect.bottom);
        state.visible = rect.height > 2 && rect.bottom > dialogRect.top + 2
            && rect.top < dialogRect.bottom - 2;
        return state;
    }"""
    try:
        result = await page.evaluate(script, label)
    except Exception:  # noqa: BLE001 - a dead page reports "not found"
        return {}
    return result if isinstance(result, dict) else {}


async def _reveal_month(page, label: str) -> bool:
    """Scroll the virtual month list until ``label`` is inside the dialog.

    The calendar renders a few months into a scrollable, virtualised list, so a
    month further out exists in the DOM but sits below the visible box. The
    ``#next-month`` button alone was not reliable, so the list is scrolled
    directly and the month re-measured until it is on screen.
    """
    for _ in range(10):
        state = await _month_state(page, label)
        if state.get("visible"):
            return True
        try:
            moved = await page.evaluate(
                """(label) => {
                    const list = document.querySelector('ytcp-scrollable-calendar #calendar-main');
                    if (!list) return false;
                    const dialog = document.querySelector('ytcp-date-picker tp-yt-paper-dialog#dialog');
                    const months = [...document.querySelectorAll('ytcp-scrollable-calendar .calendar-month')];
                    const labelOf = (m) => {
                        const el = m.querySelector('.calendar-month-label');
                        return el ? (el.textContent || '').trim() : '';
                    };
                    const month = months.find((m) => labelOf(m) === label);
                    const step = Math.max(list.clientHeight * 0.8, 120);
                    if (month && dialog) {
                        const rect = month.getBoundingClientRect();
                        const dialogRect = dialog.getBoundingClientRect();
                        // Bring the month's top just under the dialog's top.
                        list.scrollTop += (rect.top - dialogRect.top - 4);
                    } else {
                        // Not rendered yet: nudge the list so more months load.
                        list.scrollTop += step;
                    }
                    return true;
                }""",
                label,
            )
        except Exception:  # noqa: BLE001 - fall through to the button fallback
            moved = False
        if not moved:
            await _click_if_present(page, "#next-month", 3000)
        await human.pause(page, 250, 600)
    return bool((await _month_state(page, label)).get("visible"))


async def _date_trigger_text(page) -> str:
    """Text of the date dropdown (e.g. ``Nov 5, 2026``), or ``""``."""
    try:
        return await page.locator(
            f"{_DATE_TRIGGER} .dropdown-trigger-text"
        ).first.inner_text()
    except Exception:  # noqa: BLE001 - reporting only
        return ""


async def _set_schedule_date(page, schedule_at) -> bool:
    """Open the date dropdown and click the target day in the target month."""
    if not await _open_date_picker(page):
        return False

    label = _month_label(schedule_at)
    day = str(schedule_at.day)
    await _reveal_month(page, label)

    scope = f"{_CALENDAR_MONTH}:has(.calendar-month-label:text-is('{label}'))"
    expected = _normalize_spaces(_expected_date_text(schedule_at))
    for selector in (
        f"{scope} {_CALENDAR_DAY}:not(.invisible):text-is('{day}')",
        f"{scope} {_CALENDAR_DAY}:text-is('{day}')",
    ):
        # The day cell is clicked through Playwright's own hit-tested click
        # rather than the glide helper: the glide reads the cell's box, moves the
        # mouse, then clicks at a remembered offset, and if the virtual list
        # scrolls the cell in between, that offset lands beside the tiny cell —
        # on the dialog's backdrop, which closes the whole upload dialog.
        try:
            cell = page.locator(selector).first
            if not await cell.count():
                continue
            await cell.click(timeout=8000)
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
        await human.pause(page, 500, 900)
        shown = _normalize_spaces(await _date_trigger_text(page))
        if expected in shown or (day in shown and str(schedule_at.year) in shown):
            return True
    return False


async def _set_schedule_time(page, schedule_at) -> bool:
    """Type the publish time into the time field and commit it by blurring."""
    wanted = _expected_time_text(schedule_at)
    try:
        field = page.locator(_TIME_INPUT).first
        if not await field.count():
            return False
        await human.click(page, field, timeout=8000)
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Delete")
        await page.keyboard.type(wanted, delay=40)
        # Commit by blurring instead of pressing Enter: an Enter inside the
        # upload dialog can trigger the dialog's default action and close the
        # whole flow without scheduling anything.
        await page.keyboard.press("Tab")
        await human.pause(page, 400, 800)
        value = _normalize_spaces(await page.locator(_TIME_INPUT).first.input_value())
        return _normalize_spaces(wanted) in value
    except Exception:  # noqa: BLE001 - reported to the caller
        return False


async def _set_schedule(page, schedule_at) -> bool:
    """Schedule the clip: expand the Schedule card, pick the date, set the time.

    The visibility step has no "SCHEDULE" radio in the current UI; scheduling
    lives in the second card of a two-card accordion. This drives that card and
    verifies each part took, returning ``False`` (so the caller fails the upload
    instead of publishing immediately) when it cannot.
    """
    if not await _open_schedule_card(page):
        return False
    if not await _set_schedule_date(page, schedule_at):
        return False
    if not await _set_schedule_time(page, schedule_at):
        return False
    return True


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

        # A refused upload raises the error scrim over the dialog, which swallows
        # every click. Wait here so that case fails with YouTube's own reason
        # (e.g. "Daily upload limit reached") instead of a 60s click timeout on
        # the title box. A healthy dialog returns immediately.
        error = await _wait_dialog_error(page)
        if error:
            return UploadResult.failure(f"YouTube refused the upload: {error}")

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
            # A refused upload raises the error scrim, which covers every button:
            # report the reason instead of clicking into a dead dialog.
            error = await _dialog_error(page)
            if error:
                return UploadResult.failure(f"YouTube refused the upload: {error}")
            vis = page.locator("tp-yt-paper-radio-button[name='PUBLIC']")
            if await vis.count() and await vis.first.is_visible():
                break
            if not await _click_if_present(page, "#next-button", 6000):
                await human.pause(page, 900, 1600)
            await human.pause(page, 800, 1400)

        # 10) visibility / schedule -----------------------------------------------
        error = await _dialog_error(page)
        if error:
            return UploadResult.failure(f"YouTube refused the upload: {error}")
        if schedule_at is not None:
            if not await _set_schedule(page, schedule_at):
                return UploadResult.failure(
                    "could not set the YouTube schedule (the Schedule card or its "
                    "date/time picker could not be driven)"
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
        error = await _dialog_error(page)
        if error:
            return UploadResult.failure(f"YouTube refused the upload: {error}")
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
        # An error scrim makes the next click time out; prefer the dialog's own
        # reason over the timeout text when one is showing.
        error = await _dialog_error(page)
        if error:
            return UploadResult.failure(f"YouTube refused the upload: {error}")
        log.debug("YouTube upload failed: %s", exc)
        return UploadResult.failure(clip_error(exc))
    finally:
        try:
            await page.close()
        except Exception:  # noqa: BLE001 - best effort
            pass


__all__ = ["NAME", "LABEL", "check", "upload", "STUDIO_URL", "UPLOAD_URL"]
