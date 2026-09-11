"""Local LLM backend — OpenAI, DeepSeek, or Gemini, selected by ``LLM_PROVIDER``.

Each backend reads its key, model and base URL from the resolved
:class:`~shorts_generator.config.Settings`, so the whole provider setup lives in
the single ``.env`` file.
"""

from __future__ import annotations

from .config import (
    Settings,
    require_deepseek_key,
    require_gemini_key,
    require_openai_key,
)


def call_openai_llm(prompt: str, settings: Settings) -> str:
    """OpenAI Chat Completions backend."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required. Install it with:\n    pip install -r requirements.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key(settings))
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
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
    try:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
    except Exception:
        # Some DeepSeek-compatible models reject response_format/temperature;
        # fall back to a plain chat completion rather than failing.
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
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
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 8192,
        },
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
