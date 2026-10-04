"""Local LLM backend — OpenAI, DeepSeek, or Gemini, selected by ``LLM_PROVIDER``.

Each backend reads its key, model and base URL from the resolved
:class:`~shorts_generator.config.Settings`, so the whole provider setup lives in
the single ``.env`` file.
"""

from __future__ import annotations

import time
import warnings

from .config import (
    Settings,
    require_deepseek_key,
    require_gemini_key,
    require_openai_key,
)

# Значения по умолчанию, если в Settings не заданы LLM_MAX_ATTEMPTS/LLM_RETRY_BACKOFF.
LLM_MAX_ATTEMPTS = 5
LLM_RETRY_BACKOFF = 2.0

_RETRYABLE_MARKERS = (
    "429", "500", "502", "503", "504",
    "resource_exhausted", "resource exhausted",
    "unavailable", "overloaded", "high demand",
    "rate limit", "rate_limit", "too many requests",
    "timed out", "timeout", "deadline",
    "temporarily", "try again",
)


def _is_retryable(exc: Exception) -> bool:
    """True, если ошибка выглядит как временная и её стоит повторить."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _call_with_retries(call, settings: Settings):
    """Вызвать ``call()``, повторяя временные ошибки с экспоненциальным бэкоффом."""
    attempts = max(1, int(getattr(settings, "llm_max_attempts", 0) or LLM_MAX_ATTEMPTS))
    backoff = abs(float(getattr(settings, "llm_retry_backoff", LLM_RETRY_BACKOFF) or 0.0))
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - classify by message text
            if attempt >= attempts or not _is_retryable(exc):
                raise
            delay = backoff * (2 ** (attempt - 1))
            print(
                f"[llm] transient provider error ({type(exc).__name__}: {exc}); "
                f"retrying in {delay:.0f}s (attempt {attempt}/{attempts})",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("LLM call failed without a result")


def call_openai_llm(prompt: str, settings: Settings) -> str:
    """OpenAI Chat Completions backend."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required. Install it with:\n    pip install -r requirements.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key(settings))
    response = _call_with_retries(
        lambda: client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        ),
        settings,
    )
    return response.choices[0].message.content or ""


def call_deepseek_llm(prompt: str, settings: Settings) -> str:
    """DeepSeek backend used when ``LLM_PROVIDER=deepseek``.

    DeepSeek is OpenAI-compatible, so we reuse the openai client and only point
    ``base_url`` at the DeepSeek API. We ask for JSON output so the highlight
    parser always receives a machine-readable response; if the selected model
    rejects JSON mode (e.g. ``deepseek-reasoner``), we transparently retry
    without it.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required. Install it with:\n    pip install -r requirements.txt"
        ) from e

    client = OpenAI(
        api_key=require_deepseek_key(settings),
        base_url=settings.deepseek_base_url,
    )
    def _create(**extra):
        return _call_with_retries(
            lambda: client.chat.completions.create(
                model=settings.deepseek_model,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
                **extra,
            ),
            settings,
        )

    try:
        response = _create(response_format={"type": "json_object"})
    except Exception:
        # Some DeepSeek-compatible models reject response_format/temperature;
        # fall back to a plain chat completion rather than failing.
        response = _create()
    return response.choices[0].message.content or ""


def call_gemini_llm(prompt: str, settings: Settings) -> str:
    """Gemini backend used when ``LLM_PROVIDER=gemini``."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required. Install it with:\n"
            "    pip install -r requirements.txt"
        ) from e

    client = genai.Client(api_key=require_gemini_key(settings))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*automatic function calling.*"
        )
        response = _call_with_retries(
            lambda: client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                    "max_output_tokens": 8192,
                },
            ),
            settings,
        )
    return response.text or ""


def call_llm(prompt: str, settings: Settings) -> str:
    """Dispatch to the configured LLM provider."""
    provider = (settings.llm_provider or "openai").strip().lower()
    if provider == "openai":
        return call_openai_llm(prompt, settings)
    if provider == "deepseek":
        return call_deepseek_llm(prompt, settings)
    if provider == "gemini":
        return call_gemini_llm(prompt, settings)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'openai', 'deepseek', or 'gemini'."
    )
