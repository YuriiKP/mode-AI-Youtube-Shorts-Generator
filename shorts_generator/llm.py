"""Local LLM backend — OpenAI, DeepSeek, or Gemini, selected by ``LLM_PROVIDER``.

Each backend reads its key, model and base URL from the resolved
:class:`~shorts_generator.config.Settings`, so the whole provider setup lives in
the single ``.env`` file.

Retries are for answers that might come on the next try. An error that no retry
can fix — a spent daily quota, a rejected key, a retired model — is raised as a
:class:`~shorts_generator.provider_errors.ProviderError` instead, which stops the
whole batch: the remaining videos would walk into the same wall, each after
paying for its own Whisper pass, scene scan and upload.
"""

from __future__ import annotations

import time

from .config import (
    Settings,
    require_deepseek_key,
    require_gemini_key,
    require_openai_key,
)
from .provider_errors import (
    RunFatalError,
    http_status,
    is_quota_exhausted,
    is_retryable_provider_error,
    provider_error,
)

# Значения по умолчанию, если в Settings не заданы LLM_MAX_ATTEMPTS/LLM_RETRY_BACKOFF.
LLM_MAX_ATTEMPTS = 5
LLM_RETRY_BACKOFF = 2.0


def _call_with_retries(call, settings: Settings, *, engine: str, model: str):
    """Вызвать ``call()``, повторяя временные ошибки с экспоненциальным бэкоффом.

    Повтор имеет смысл только для ответа, который может прийти со следующей
    попытки. Исчерпанная квота — нет: провайдер сам пишет «повторите через
    3ч24м», и наши 2-4-8-16 секунд её не пересидят, поэтому такой 429 улетает
    сразу, как и любая другая ошибка, которую повтор не исправит. И то и другое
    поднимается как :class:`~shorts_generator.provider_errors.ProviderError`, из-за
    которого ``pipeline._run`` останавливает весь батч: следующие видео упрутся
    в ту же стену, каждое — после своей транскрибации, сканирования сцен и
    заливки.
    """
    attempts = max(1, int(getattr(settings, "llm_max_attempts", 0) or LLM_MAX_ATTEMPTS))
    backoff = abs(
        float(getattr(settings, "llm_retry_backoff", LLM_RETRY_BACKOFF) or 0.0)
    )
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - classify by status/message
            # Спенчённая квота и всё, что повтор не исправит, — сразу наружу:
            # ждать её бессмысленно, а лишние попытки лишь маскируют причину.
            if is_quota_exhausted(exc) or not is_retryable_provider_error(exc):
                raise provider_error(engine, model, exc) from exc
            # Временным это выглядело на каждой попытке (5xx, таймаут, отдача
            # «попробуйте позже») — но ответа так и не пришло.
            if attempt >= attempts:
                raise provider_error(engine, model, exc, attempts=attempt) from exc
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
        engine="openai",
        model=settings.openai_model,
    )
    return response.choices[0].message.content or ""


def _json_mode_rejected(exc: Exception) -> bool:
    """True when the failure looks like "this model has no JSON mode".

    Exists to tell a model that merely dislikes ``response_format`` — worth one
    plain retry without it — apart from a provider that cannot answer at all (an
    unknown model, a spent quota, a rejected key), where the same call without
    ``response_format`` would only spend a second retry budget on the very same
    wall.
    """
    if http_status(exc) == 400:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "response_format",
            "response format",
            "json_object",
            "json mode",
        )
    )


def call_deepseek_llm(prompt: str, settings: Settings) -> str:
    """DeepSeek backend used when ``LLM_PROVIDER=deepseek``.

    DeepSeek is OpenAI-compatible, so we reuse the openai client and only point
    ``base_url`` at the DeepSeek API. We ask for JSON output so the highlight
    parser always receives a machine-readable response; if the selected model
    rejects JSON mode (e.g. ``deepseek-reasoner``), we transparently retry
    without it. That fallback covers a rejection of JSON mode only: a
    provider-side failure (an unknown model, a spent quota) is not retried a
    second time, because the plain call would fail exactly like the first one.
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
            engine="deepseek",
            model=settings.deepseek_model,
        )

    try:
        response = _create(response_format={"type": "json_object"})
    except RunFatalError as exc:
        # A model that merely dislikes ``response_format`` is worth one plain
        # retry; a provider that cannot answer at all — an unknown model, a spent
        # quota, a rejected key — is not. Asking again without ``response_format``
        # there would spend a second full retry budget to reach the same wall,
        # which is what a real run did: ten 503s instead of five, and twice the
        # log lines for one cause.
        if not _json_mode_rejected(exc):
            raise
        response = _create()
    except Exception:  # noqa: BLE001 - the JSON-mode fallback is deliberately broad
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
    response = _call_with_retries(
        lambda: client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
                "max_output_tokens": 8192,
                # We pass no tools/callables, so automatic function calling
                # (AFC) has nothing to run. Disabling it explicitly makes the
                # SDK take the plain generate_content path — no AFC loop and no
                # "direct use of AFC is not recommended" log line.
                "automatic_function_calling": {"disable": True},
            },
        ),
        settings,
        engine="gemini",
        model=settings.gemini_model,
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
