"""Local LLM backend — OpenAI, DeepSeek, or Gemini, selected by LLM_PROVIDER."""

from ..config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_MODEL,
    require_deepseek_key,
    require_gemini_key,
    require_openai_key,
)


def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend used by --mode local."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key())
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_deepseek_llm(prompt: str) -> str:
    """DeepSeek backend used by --mode local when LLM_PROVIDER=deepseek.

    DeepSeek is OpenAI-compatible, so we reuse the openai client and only
    point base_url at the DeepSeek API. We ask for JSON output so the highlight
    parser always receives a machine-readable response; if the selected model
    rejects JSON mode (e.g. deepseek-reasoner), we transparently retry without
    it.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for LLM_PROVIDER=deepseek. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_deepseek_key(), base_url=DEEPSEEK_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
    except Exception:
        # Some DeepSeek-compatible models reject response_format/temperature;
        # fall back to a plain chat completion rather than failing.
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
    return response.choices[0].message.content or ""


def call_gemini_llm(prompt: str) -> str:
    """Gemini backend used by --mode local when LLM_PROVIDER=gemini."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required for LLM_PROVIDER=gemini. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = genai.Client(api_key=require_gemini_key())
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 8192,
        },
    )
    return response.text or ""


def call_local_llm(prompt: str) -> str:
    """Dispatch to the configured local LLM provider."""
    provider = (LLM_PROVIDER or "openai").strip().lower()
    if provider == "openai":
        return call_openai_llm(prompt)
    if provider == "deepseek":
        return call_deepseek_llm(prompt)
    if provider == "gemini":
        return call_gemini_llm(prompt)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'openai', 'deepseek', or 'gemini'."
    )
