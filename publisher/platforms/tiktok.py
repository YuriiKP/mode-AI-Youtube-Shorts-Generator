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
from .base import UploadResult, clip_error, page_url, trim

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

# TikTok's caption box accepts ~2200 characters. The title and the description
# are trimmed to leave room for the hashtags that are typed after them, so the
# whole caption stays within the limit.
_CAPTION_LIMIT = 2200

# Matches TikTok's generated widget class names, e.g.
# ``tiktok-xyz-SelectFormContainer``.
_SELECT_FORM_RE = re.compile(r"tiktok-.*-SelectFormContainer.*")
_VIDEO_ID_RE = re.compile(r"video/(\d+)")

# Publishing: how many "Post" presses to attempt, and how long (ms) to wait for
# the redirect to the content list after each one.
_PUBLISH_ATTEMPTS = 60
_PUBLISH_POLL_MS = 3000

# A modal shown instead of publishing the clip (e.g. TikTok refusing a post as
# an account violation). Its presence means "the post was not created".
_BLOCKING_MODAL = (
    ".account-violation-modal, div.common-modal:has-text('Post not created')"
)

# The upload form's primary action button. It reads "Post" for an immediate
# post, but switches to "Schedule" once the "Schedule" tab is selected, so the
# code has to match whichever is on screen.
_ACTION_BUTTON = (
    'div.button-group > button:has-text("Post"), '
    'div.button-group > button:has-text("Schedule")'
)

# TikTok's onboarding tour renders this overlay; it intercepts pointer events
# and blocks every click, so it is detached before any interaction.
_JOYRIDE_PORTAL = "#react-joyride-portal"


# ---------------------------------------------------------------------------
# Low-level DOM helpers (kept close to the original implementation)
# ---------------------------------------------------------------------------


async def _is_english_ui(page) -> bool:
    """Best-effort: is the current TikTok UI in English?

    TikTok keeps the chosen language in the account, so the answer holds across
    pages. ``<html lang>`` reflects the rendered language; the "More" nav label
    (present on the main site) is used as a fallback.
    """
    try:
        lang = await page.evaluate("document.documentElement.lang || ''") or ""
        if lang.strip():
            return lang.strip().lower().startswith("en")
    except Exception:  # noqa: BLE001 - fall back to the DOM check
        pass
    try:
        menu = page.locator('[data-e2e="nav-more-menu"]')
        if await menu.count():
            return ((await menu.first.text_content()) or "").strip() == "More"
    except Exception:  # noqa: BLE001 - treat an unreadable menu as "not set"
        pass
    return False


async def _pick_english_language(page) -> None:
    """Switch the account's UI language to English via the main-site menu.

    The language menu only exists on the main site (www.tiktok.com), not on the
    Studio pages, so this navigates there, opens the menu and selects
    "English (US)". Best-effort: failures are ignored.
    """
    try:
        await page.goto(HOME_URL, wait_until="domcontentloaded")
        await page.wait_for_selector('[data-e2e="nav-more-menu"]', timeout=15000)
        await human.pause(page, 600, 1300)
    except Exception:  # noqa: BLE001 - the UI is left as-is when this fails
        return

    try:
        await human.click(page, page.locator('[data-e2e="nav-more-menu"]'))
        await human.click(page, page.locator('[data-e2e="language-select"]'))
        await human.click(
            page,
            page.locator(
                "#creator-tools-selection-menu-header", has_text="English (US)"
            ),
        )
    except Exception:  # noqa: BLE001 - non-fatal; we simply continue
        pass


async def _ensure_english_ui(page, url: str) -> None:
    """Open ``url`` and make sure the page is shown in English.

    The uploader matches English labels ("Select video", "Post"). Instead of
    always detouring through the main site to read the language menu, the
    language of the page we are already on is checked first, and the switch only
    happens when the UI is not English — after which ``url`` is reopened.
    """
    await page.goto(url, wait_until="domcontentloaded")
    await human.wait_ready(page)
    if await _is_english_ui(page):
        return
    await _pick_english_language(page)
    await page.goto(url, wait_until="domcontentloaded")
    await human.wait_ready(page)


async def _choose_base_locator(page):
    """Return the locator scope holding the upload form (frame or body)."""
    if await page.locator(_UPLOAD_IFRAME).count():
        return page.frame_locator(_UPLOAD_IFRAME)
    return page.locator("body")


async def _dismiss_onboarding(page) -> None:
    """Remove TikTok's onboarding (react-joyride) overlay if it is present.

    The tour darkens the page with a spotlight and, on this rollout, has no
    tooltip or buttons — there is nothing to click to close it. It appears when
    the upload page loads and again once a file has been chosen, covering the
    caption box and the buttons, so a click would land on the overlay instead of
    the page. Detaching the portal node is the only reliable way to get rid of
    it. Best-effort: any failure is ignored.
    """
    try:
        await page.evaluate(
            "(sel) => document.querySelector(sel)?.remove()", _JOYRIDE_PORTAL
        )
    except Exception:  # noqa: BLE001 - the tour is optional; never fail on it
        pass


async def _add_title_and_tags(base, page, title: str, description: str, tags) -> None:
    """Type the caption (title + description) and hashtags into the editor.

    TikTok keeps the whole caption — title, description and hashtags — in a
    single rich-text box. The clip's description used to be dropped, so only the
    title and the hashtags reached the site; here the title and the description
    go on separate lines, and each hashtag is typed with a trailing space so
    TikTok turns it into a chip.
    """
    editor = base.locator("div.public-DraftEditor-content")
    await editor.wait_for(state="visible", timeout=30_000)
    await human.click(page, editor)

    # Clear whatever placeholder/prefill is already there.
    await page.keyboard.press("End")
    await page.keyboard.press("Control+A")
    await page.keyboard.press("Delete")
    await page.keyboard.press("End")
    await human.pause(page, 700, 1400)

    # The hashtags are appended after the caption, so reserve room for them and
    # keep the whole caption within TikTok's limit.
    budget = max(0, _CAPTION_LIMIT - sum(len(tag) + 2 for tag in tags))
    title_text = trim(title, budget)
    description_text = trim(description, max(0, budget - len(title_text) - 1))

    wrote_line = False
    if title_text:
        await page.keyboard.insert_text(title_text)
        wrote_line = True
        await human.pause(page, 800, 1500)

    if description_text:
        if wrote_line:
            await page.keyboard.press("End")
            await page.keyboard.press("Enter")
        await page.keyboard.insert_text(description_text)
        wrote_line = True
        await human.pause(page, 800, 1500)

    if wrote_line:
        await page.keyboard.press("End")
        await page.keyboard.press("Enter")

    for tag in tags:
        await page.keyboard.press("End")
        await human.pause(page, 700, 1400)
        # Type as "#tag " then drop the extra space so TikTok registers the
        # hashtag chip without leaving a dangling space behind.
        await page.keyboard.insert_text("#" + tag + " ")
        await page.keyboard.press("Space")
        await human.pause(page, 700, 1400)
        await page.keyboard.press("Backspace")
        await page.keyboard.press("End")


async def _detect_upload_status(base, page, file_path: str, logger) -> bool:
    """Wait until the file finished uploading (the "Post"/"Schedule" button enables).

    The form's primary button reads "Post" for an immediate post and "Schedule"
    once the "Schedule" tab is selected; either one becoming enabled means the
    transfer is done. While waiting, recover once from a stalled/errored
    transfer by re-selecting the file if the site shows its "Select file" error
    button.
    """
    post_button = base.locator(_ACTION_BUTTON).first
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


async def _close_blocking_modal(base, page) -> bool:
    """Dismiss a "the post was not created" modal; return whether one was found.

    TikTok sometimes refuses a post and shows a modal (e.g. an account-violation
    dialog) instead of redirecting to the content list. Detecting it lets the
    caller stop retrying and report a real failure instead of waiting forever.
    """
    modal = base.locator(_BLOCKING_MODAL)
    try:
        if not await modal.count():
            return False
    except Exception:  # noqa: BLE001 - absence means "no block"
        return False
    try:
        close = base.locator(".close-modal-btn").first
        if await close.count():
            await human.click(page, close)
    except Exception:  # noqa: BLE001 - closing is best-effort
        pass
    return True


async def _click_publish(base, page, logger) -> Tuple[bool, str]:
    """Press "Post", accept the "Post now" dialog, and wait for the outcome.

    Returns ``(True, "")`` once TikTok redirects to the content list, otherwise
    ``(False, reason)``. TikTok does not publish on the first press: it opens a
    confirmation dialog (``common-modal-confirm-modal``, buttons "Cancel" /
    "Post now") and only posts once the primary button is pressed, which this
    does. A refused post shows a modal instead of the redirect and is reported
    as a failure so the run can move on.
    """
    # Target the buttons by label/role instead of position: the button group also
    # holds "Discard", so an index-based pick could hit the wrong one.
    publish_button = base.locator(_ACTION_BUTTON).first
    confirm_button = base.locator(
        "div.common-modal-confirm-modal button.TUXButton--primary"
    ).first
    for attempt in range(_PUBLISH_ATTEMPTS):
        try:
            if await publish_button.count():
                # The first press is the real one: glide the cursor onto "Post"
                # and click like a person would. Later retries stay quick.
                if attempt == 0:
                    await human.click(page, publish_button)
                else:
                    await publish_button.click()

            # The "Post now" confirmation is what actually publishes the clip.
            if await confirm_button.count():
                if attempt == 0:
                    await human.click(page, confirm_button)
                else:
                    await confirm_button.click()

            try:
                await page.wait_for_url(CONTENT_URL, timeout=_PUBLISH_POLL_MS)
                return True, ""
            except Exception:  # noqa: BLE001 - not navigated yet, keep trying
                pass

            # A refused post shows a modal instead of the redirect.
            if await _close_blocking_modal(base, page):
                return False, "TikTok refused the post (see the dialog on the page)"
        except Exception as exc:  # noqa: BLE001 - retry the click
            logger.debug("TikTok: publish retry (%s)", clip_error(exc))
        # Space out the retries with a short, human-like pause rather than a
        # metronomic half-second interval.
        await human.pause(page, 600, 1200)
    return False, "publish did not confirm (no redirect to the content list)"


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
        # 1) open the upload form, in English -----------------------------------
        await _ensure_english_ui(page, UPLOAD_URL)

        # A stale session bounces the upload page to the login form; fail with a
        # clear message instead of an opaque selector timeout further down.
        if "/login" in page_url(page).lower():
            return UploadResult.failure(
                "TikTok login expired — run: "
                "python main.py publish manual --profile <name>",
                auth_required=True,
            )

        # 2) the form may be inline or inside an iframe -------------------------
        # Clear the onboarding tour before touching the form (it can be raised
        # again as soon as the file is chosen), then wait for a real signal —
        # the "Select video" button — instead of a container that does not exist.
        await _dismiss_onboarding(page)
        try:
            await page.wait_for_selector(
                f'{_UPLOAD_IFRAME}, button:has-text("Select video")', timeout=30_000
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

        # 4) caption: title + description + hashtags ----------------------------
        # Choosing a file can raise the onboarding tour again; clear it before
        # typing so the overlay does not swallow the clicks on the caption box.
        await _dismiss_onboarding(page)
        await _add_title_and_tags(base, page, short.title, short.description, tags)

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
        published, publish_error = await _click_publish(base, page, log)
        if not published:
            return UploadResult.failure(publish_error or "publish did not confirm")

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
