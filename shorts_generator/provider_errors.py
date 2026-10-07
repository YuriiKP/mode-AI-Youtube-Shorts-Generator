"""Provider-side failures: the ones that belong to the whole run, not one video.

A batch run is built to isolate per-video problems: a clip with no detectable
speech or a broken ffmpeg pass is recorded as a failure and the loop moves on to
the next input. A provider failure is the opposite case. An exhausted quota, a
rejected key, a retired model name or an overload that outlived every retry
answers every remaining input in exactly the same way — so walking the rest of
the batch is not isolation, it is a way to pay for a Whisper pass, a
frame-by-frame scene scan and a 100+ MB upload per input before reaching the
same wall. Those errors are raised as :class:`RunFatalError`, and the batch stops
at the first one with a single message instead of one copy per input.

The classification lives here rather than next to either caller because both
``llm.py`` (highlight ranking, every provider) and ``visual_indexer.py`` (the
cloud video engine) have to agree on what is transient, what is permanent and
what means "come back tomorrow".
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# HTTP statuses worth another attempt: request timeout, conflict, and "too many
# requests" when it is a per-minute limit that clears inside the backoff. Any
# other 4xx is permanent; 5xx is transient overload.
RETRYABLE_HTTP_STATUS = frozenset({408, 409, 429})

# Message-level fallback for SDKs that expose no usable status code.
RETRYABLE_ERROR_MARKERS = (
    "resource_exhausted",
    "resource exhausted",
    "unavailable",
    "overloaded",
    "high demand",
    "rate limit",
    "rate_limit",
    "too many requests",
    "timed out",
    "timeout",
    "deadline",
    "temporarily",
    "try again",
)

# A 429 that arrives with a longer wait than this is not a per-minute hiccup but
# a spent daily quota: no 2-4-8-16s backoff outlasts it.
LONG_QUOTA_WAIT_SECONDS = 60.0

# How a provider spells "this quota is used up for the day". Google reports the
# quota id (``GenerateRequestsPerDayPerProjectPerModel-FreeTier``), the rest say
# it in prose.
DAILY_QUOTA_MARKERS = (
    "perday",
    "per day",
    "per-day",
    "daily limit",
    "free_tier",
    "free tier",
)

# How a provider spells "there is no such model here" — a wrong or retired model
# id, which is a configuration problem, not an outage. The wording is checked
# because the status cannot be trusted: one OpenAI-compatible gateway answers an
# unknown model with a 503 ``model_not_found`` body, and a 503 normally means
# "come back later".
#
# ``"temporarily unavailable"`` is deliberately absent: the same gateway prefixes
# a genuine overload message with it, and that one *is* worth retrying.
MODEL_UNAVAILABLE_MARKERS = (
    "model_not_found",
    "model not found",
    "unknown model",
    "no such model",
    "unsupported model",
    "invalid model",
    "model does not exist",
    "does not exist for",
    # google-genai: "models/gemini-3.8-flash is not found for API version v1beta"
    "is not found for",
    "no longer available",
)

# google-genai puts the status in the message prefix ("429 RESOURCE_EXHAUSTED.").
# Some SDK versions expose neither ``.code`` nor ``.status_code``.
_STATUS_IN_MESSAGE_RE = re.compile(r"\b([1-5]\d{2})\b\s+[A-Z][A-Z_]{2,}")

# Both spellings Google uses for a quota reset: the structured
# ``'retryDelay': '12265s'`` of a 429 body, and the sentence
# "Please retry in 3h24m25.566590266s."
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)\s*s")
_RETRY_IN_TEXT_RE = re.compile(
    r"retry in\s+(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(\d+(?:\.\d+)?)\s*s"
)

# The .env variables each engine reads, so a fix-it hint can name the right one.
_ENV_NAMES = {
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL"),
    "gemini_video": ("GEMINI_API_KEY", "GEMINI_MODEL"),
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL"),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"),
}
_DEFAULT_ENV_NAMES = ("GEMINI_API_KEY", "GEMINI_MODEL")


class RunFatalError(RuntimeError):
    """A failure that is not about the current input, so the batch stops.

    Raised for provider-side failures (an exhausted quota, a rejected key, a
    retired model, an overload that outlived every retry) and for deterministic
    setup problems (a missing dependency or key). ``pipeline._run`` reads it as
    "every remaining input would fail identically" and breaks the loop instead of
    finishing the batch out of habit.
    """


class ProviderError(RunFatalError):
    """A provider refused to answer in a way that running again will not fix."""


def env_names(engine: str) -> Tuple[str, str]:
    """``(key_env, model_env)`` — the .env variables the given engine reads."""
    return _ENV_NAMES.get((engine or "").strip().lower(), _DEFAULT_ENV_NAMES)


def http_status(exc: Exception) -> Optional[int]:
    """Best-effort HTTP status behind a provider exception, else ``None``.

    ``google-genai`` (and google-api-core) expose it as ``.code``; the OpenAI and
    DeepSeek SDKs use ``.status_code``. When neither is set, the
    ``"429 RESOURCE_EXHAUSTED. …"`` prefix of the message is the last resort.
    """
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, int):
            if 100 <= value <= 599:
                return value
            continue
        match = re.search(r"\b([1-5]\d{2})\b", str(value))
        if match:
            return int(match.group(1))
    match = _STATUS_IN_MESSAGE_RE.search(str(exc))
    return int(match.group(1)) if match else None


def retry_after_seconds(exc: Exception) -> Optional[float]:
    """How long the provider itself asks us to wait, when it says so."""
    text = str(exc)
    match = _RETRY_DELAY_RE.search(text)
    if match:
        return float(match.group(1))
    match = _RETRY_IN_TEXT_RE.search(text)
    if match:
        hours, minutes, seconds = match.groups()
        return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds)
    return None


def _human_wait(seconds: float) -> str:
    """A duration as the log reads it: ``3 h 24 min`` / ``6 min`` / ``45 s``."""
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        hours, rest = divmod(int(seconds), 3600)
        return f"{hours} h {rest // 60} min"
    if seconds >= 60:
        return f"{int(seconds) // 60} min"
    return f"{int(seconds)} s"


def is_model_unavailable(exc: Exception) -> bool:
    """True when the provider says the *model* is not there, whatever the status.

    A real run ran into the reason this cannot be decided by the status code: an
    OpenAI-compatible gateway answered an unknown model with

        503 {'error': {'code': 'model_not_found', 'message': 'Model
        anthropic/claude-opus-4.5 is temporarily unavailable'}}

    — a status that normally means "try again later" wrapped around an error that
    trying later cannot fix, because the model id is simply not one this endpoint
    serves. Retrying it ten times (the DeepSeek path asks twice, JSON mode and
    plain) bought nothing but a minute of log noise.
    """
    text = str(exc).lower()
    return any(marker in text for marker in MODEL_UNAVAILABLE_MARKERS)


def is_quota_exhausted(exc: Exception) -> bool:
    """True when the provider says the quota is spent, not that we are too fast.

    A spent per-day/per-project quota cannot be waited out with a short backoff:
    asking again only produces the same answer. The provider even says so — a
    real free-tier run answered "Please retry in 3h24m25s." while the caller was
    already sleeping 2, 4, 8 and 16 seconds and calling it a transient blip.
    """
    if http_status(exc) != 429:
        return False
    text = str(exc).lower()
    if any(marker in text for marker in DAILY_QUOTA_MARKERS):
        return True
    wait = retry_after_seconds(exc)
    return wait is not None and wait > LONG_QUOTA_WAIT_SECONDS


def is_retryable_provider_error(exc: Exception) -> bool:
    """True only for errors a second attempt can plausibly fix.

    4xx other than 408/409/429 (a retired model name, a bad key, a malformed
    request) are permanent: they are surfaced immediately instead of being
    retried, so the real cause is not masked as a transient network blip.

    The "no such model" wording is checked *before* the status, because a gateway
    that reports it as a 503 would otherwise be retried five times over for an
    answer that never changes.
    """
    if is_model_unavailable(exc):
        return False
    status = http_status(exc)
    if status is not None:
        return status in RETRYABLE_HTTP_STATUS or status >= 500
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)


def provider_error(
    engine: str,
    model: str,
    exc: Exception,
    *,
    attempts: int = 1,
) -> ProviderError:
    """Build the error raised for a provider failure that re-running will not fix.

    ``attempts`` is how many tries it took to give up — 1 when the failure was
    obviously permanent on first sight (a rejected key, a retired model), more
    when transient-looking answers (503, timeouts) exhausted the backoff and
    turned out not to be transient at all. ``engine`` selects the ``.env`` names
    the fix-it hint points at.
    """
    status = http_status(exc)
    key_env, model_env = env_names(engine)
    tries = f" after {attempts} attempt(s)" if attempts > 1 else ""

    if is_model_unavailable(exc):
        return ProviderError(
            f"{engine} request failed{tries}: the model {model!r} is not available "
            f"on this endpoint, and no retry changes that: {exc}\n"
            "The provider may dress this up as a transient 503 "
            '("temporarily unavailable"), but in a ``model_not_found`` / '
            '"not found" body the complaint is about the name, not about load.\n'
            f"Check {model_env} in .env against the models this endpoint actually "
            "serves: an aggregator gateway spells ids its own way "
            "(``anthropic/claude-opus-4.5`` and ``claude-opus-4-5`` are not "
            "interchangeable), and a model can also be retired or tied to another "
            "plan."
        )

    if is_quota_exhausted(exc):
        message = (
            f"{engine} request failed{tries}: the provider's quota is spent, and no "
            f"retry can fix that: {exc}\n"
        )
        wait = retry_after_seconds(exc)
        if wait is not None:
            message += (
                f"The provider asks to wait {_human_wait(wait)} before the next "
                "request; asking sooner only repeats this answer.\n"
            )
        message += (
            f"Options: wait for the quota to reset, enable billing for the project "
            f"behind {key_env}, or point {model_env} at another model — quota is "
            "counted per model, so a different one starts with its own budget."
        )
        return ProviderError(message)

    what = f"HTTP {status}" if status is not None else "a non-retryable error"
    message = f"{engine} request failed with {what}{tries}: {exc}"
    text = str(exc).lower()
    if status in (401, 403) or "api key" in text or "permission" in text:
        message += (
            f"\nCheck {key_env} in .env — the key looks rejected or lacks access "
            "to this API."
        )
    elif status == 404 or "not found" in text or "no longer available" in text:
        message += (
            f"\nThe model {model!r} ({model_env} in .env) is not available to this "
            f"key — it may be retired or renamed. Update {model_env} and re-run."
        )
    elif status == 400:
        message += (
            f"\nThe request was rejected as malformed; check {model_env} ({model!r})."
        )
    elif attempts > 1:
        # Looked transient on every attempt (5xx / overload / timeout) and never
        # turned into an answer.
        message += (
            "\nThe provider never returned a usable answer — an outage, an overload "
            "or a timeout on its side, not a problem with this video. Running again "
            "later should get past it."
        )
    return ProviderError(message)
