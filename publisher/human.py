"""Pacing helpers that make the upload flow move at a human pace.

Upload bots stand out because they act instantly: a click the very millisecond a
button appears, a form filled in one shot, no time spent "reading" the page.
Real users hover, hesitate, and type with an uneven rhythm. The helpers below add
just enough of that randomness — a single shared RNG plus a couple of seconds
here and there — to keep the automation from looking machine-timed, without
turning it into a full human-behaviour simulator.

They are deliberately tiny and page-agnostic, so every caller shares them. The
platform uploaders import them from the parent package:

    from .. import human

    await human.settle(page)              # let the page finish loading
    await human.click(page, some_locator) # glide cursor, pause, click
    await human.pause(page, 500, 1200)    # random "reading" pause

while :mod:`publisher.publish` (a sibling) uses ``from . import human``.
"""

from __future__ import annotations

import asyncio
import random

# A single shared generator is plenty here: we only need plausible variability,
# not reproducibility, and callers never depend on the exact values.
_rng = random.Random()


async def pause(page=None, lo_ms: int = 400, hi_ms: int = 1200) -> None:
    """Sleep for a random interval between ``lo_ms`` and ``hi_ms``.

    When a ``page`` is given the wait goes through ``page.wait_for_timeout`` so
    the browser keeps rendering and running its scripts while we idle; otherwise
    it falls back to a plain ``asyncio.sleep``. The random spread is what keeps
    two runs from sharing the same tell-tale rhythm.
    """
    if hi_ms < lo_ms:
        lo_ms, hi_ms = hi_ms, lo_ms
    delay = _rng.uniform(lo_ms, hi_ms) / 1000.0
    if page is not None:
        await page.wait_for_timeout(delay * 1000.0)
    else:
        await asyncio.sleep(delay)


async def settle(page, timeout: int = 8000) -> None:
    """Best-effort wait for the page's network activity to calm down.

    Used right after a navigation instead of a fixed sleep, so we continue as
    soon as the page is actually ready. A page that never goes fully idle (heavy
    SPAs like the Studio UIs) simply times out and we move on.
    """
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout)
    except Exception:  # noqa: BLE001 - "calm enough" is good enough
        pass


async def wait_ready(
    page, *, timeout: int = 8000, lo_ms: int = 800, hi_ms: int = 1800
) -> None:
    """Let a freshly opened page load, then take a short human pause.

    Convenience wrapper around :func:`settle` + :func:`pause` for the moment
    right after a navigation, before the first interaction.
    """
    await settle(page, timeout)
    await pause(page, lo_ms, hi_ms)


async def _box(locator):
    """Best-effort bounding box of ``locator`` (``None`` when unavailable)."""
    try:
        return await locator.bounding_box()
    except Exception:  # noqa: BLE001 - a missing box just means "click centre"
        return None


async def click(
    page, locator, *, timeout: int = 20_000, settle_after: bool = True
) -> None:
    """Glide the cursor onto ``locator`` over several steps, pause, then click.

    Instead of firing the click the instant the node exists, this waits for the
    element to load, moves the mouse to a random point *inside* it in a handful
    of animated steps (so the pointer does not teleport straight to the centre),
    waits a beat, and only then clicks — at that same jittered spot.
    """
    await locator.wait_for(state="visible", timeout=timeout)
    await pause(page, 150, 500)  # reaction time before reaching for the element

    box = await _box(locator)
    if box:
        # Aim for a random spot within the element, not its exact centre.
        tx = box["x"] + box["width"] * _rng.uniform(0.25, 0.75)
        ty = box["y"] + box["height"] * _rng.uniform(0.25, 0.75)
        # Travel there in many small steps so the move is a glide, not a jump.
        try:
            await page.mouse.move(tx, ty, steps=_rng.randint(8, 24))
        except Exception:  # noqa: BLE001 - the trajectory is cosmetic
            pass
        await pause(page, 80, 220)
        # The cursor is already sitting on the target, so clicking that point
        # does not produce a fresh jump to the element's middle.
        await locator.click(position={"x": tx - box["x"], "y": ty - box["y"]})
    else:
        # No box to aim at (e.g. an off-screen node): fall back to a hover+click.
        try:
            await locator.hover(timeout=timeout)
        except Exception:  # noqa: BLE001 - some nodes refuse hover; click still works
            pass
        await pause(page, 120, 400)
        await locator.click()

    if settle_after:
        await pause(page, 250, 800)


async def type_text(locator, text: str, *, lo: int = 40, hi: int = 150) -> None:
    """Type ``text`` word by word with an uneven, human-like keystroke rhythm.

    Splitting on spaces (rather than a constant per-character delay) gives the
    typing a slightly irregular cadence, and an occasional longer beat mid-text
    mimics the pauses people take while composing a caption.
    """
    if hi < lo:
        lo, hi = hi, lo
    for index, word in enumerate(text.split(" ")):
        piece = word if index == 0 else " " + word
        await locator.type(piece, delay=_rng.randint(lo, hi))
        if _rng.random() < 0.25:
            await pause(None, 150, 500)


async def delay(seconds: float, jitter: float = 0.35) -> None:
    """Sleep for ``seconds`` with a random ±``jitter`` spread applied.

    Used to space consecutive uploads apart without making them exactly evenly
    spaced, which is itself a giveaway. A non-positive ``seconds`` is a no-op.
    """
    if seconds <= 0:
        return
    spread = max(0.0, min(1.0, jitter))
    await asyncio.sleep(seconds * (1.0 + _rng.uniform(-spread, spread)))


__all__ = ["click", "delay", "pause", "settle", "type_text", "wait_ready"]
