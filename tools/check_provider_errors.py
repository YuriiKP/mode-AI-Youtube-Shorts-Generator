"""Standalone check for the provider-error classification and the batch abort.

Two things are verified here:

* the classification in ``shorts_generator/provider_errors.py`` — what counts as
  transient, what is permanent, and what means "the quota is spent, come back
  tomorrow";
* the reaction in ``shorts_generator/pipeline.py`` — a provider-side failure must
  stop the batch at the first input instead of walking the rest into the same
  wall, while still being recorded in ``failures`` with ``aborted`` set.

No pytest needed: every check prints its own PASS/FAIL line and the script exits
non-zero if any of them failed. The exception bodies are synthetic, copied from a
real failing run (a spent free-tier quota ``429``, a per-minute ``429``, a ``503``
overload, a ``404`` retired model) and attached to fake exception classes, so no
provider is ever contacted and nothing is downloaded or uploaded.

Run it from the project root:

    python tools/check_provider_errors.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shorts_generator import pipeline  # noqa: E402
from shorts_generator.config import Settings  # noqa: E402
from shorts_generator.provider_errors import (  # noqa: E402
    ProviderError,
    RunFatalError,
    http_status,
    is_model_unavailable,
    is_quota_exhausted,
    is_retryable_provider_error,
    provider_error,
    retry_after_seconds,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
        return
    FAILURES.append(name)
    print(f"FAIL  {name}{(': ' + detail) if detail else ''}")


class FakeClientError(Exception):
    """Stands in for google.genai.errors.ClientError (4xx)."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


class FakeServerError(Exception):
    """Stands in for google.genai.errors.ServerError (5xx)."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


class FakeOpenAIError(Exception):
    """Stands in for openai.InternalServerError / APIStatusError.

    The OpenAI (and DeepSeek) SDKs expose the status as ``.status_code``, not as
    ``.code``, so this class is what makes that reading path reachable in the
    checks below.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


# --- the exact shapes a real run produced ------------------------------------
QUOTA_TEXT = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
    "your current quota, please check your plan and billing details.', 'status': "
    "'RESOURCE_EXHAUSTED', 'details': [{'@type': 'google.rpc.QuotaFailure', "
    "'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/"
    "generate_content_free_tier_requests', 'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'}]}, "
    "{'@type': 'google.rpc.RetryInfo', 'retryDelay': '12265s'}]}}"
)
QUOTA_SENTENCE_ONLY = "Please retry in 3h24m25.566590266s."
PER_MINUTE_TEXT = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Rate limit "
    "reached', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'google.rpc.RetryInfo', 'retryDelay': '30s'}]}}"
)
UNUSED_503_TEXT = (
    "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently "
    "experiencing high demand. Spikes in demand are usually temporary. Please try "
    "again later.', 'status': 'UNAVAILABLE'}}"
)
NOT_FOUND_TEXT = (
    "404 NOT_FOUND. {'error': {'code': 404, 'message': 'models/gemini-3.8-flash is "
    "not found for API version v1beta'}}"
)
# An OpenAI-compatible gateway answering an *unknown model*: a 503 body, because
# 503 is what the gateway uses, wrapped around a code that means "wrong name".
GATEWAY_TEXT = (
    "Error code: 503 - {'error': {'code': 'model_not_found', 'message': 'Model "
    "anthropic/claude-opus-4.5 is temporarily unavailable (request id: "
    "elysiumai-20261007215520865639078d4bb237bASnDZmkM)', 'type': "
    "'elyziumai_error'}}"
)
# And the opposite case: a model that merely has no JSON mode, which the DeepSeek
# backend is supposed to survive by asking again without ``response_format``.
JSON_MODE_TEXT = (
    "Error code: 400 - {'error': {'message': 'response_format type is "
    "unavailable', 'type': 'invalid_request_error'}}"
)


def check_classification() -> None:
    quota_exc = FakeClientError(QUOTA_TEXT, code=429)
    sentence_exc = FakeClientError(QUOTA_SENTENCE_ONLY, code=429)
    per_minute_exc = FakeClientError(PER_MINUTE_TEXT, code=429)
    overloaded_exc = FakeServerError(UNUSED_503_TEXT, code=503)
    not_found_exc = FakeClientError(NOT_FOUND_TEXT, code=404)
    gateway_exc = FakeOpenAIError(GATEWAY_TEXT, status_code=503)

    check("http_status(quota) == 429", http_status(quota_exc) == 429)
    check("http_status(503) == 503", http_status(overloaded_exc) == 503)

    check(
        "retry_after_seconds reads retryDelay",
        retry_after_seconds(quota_exc) == 12265.0,
        repr(retry_after_seconds(quota_exc)),
    )
    sentence_wait = retry_after_seconds(sentence_exc)
    check(
        "retry_after_seconds reads '3h24m25.566590266s'",
        sentence_wait is not None and abs(sentence_wait - 12265.57) < 0.01,
        repr(sentence_wait),
    )

    check(
        "is_quota_exhausted(daily 429)",
        is_quota_exhausted(quota_exc) is True,
    )
    check(
        "is_quota_exhausted(per-minute 429) is False",
        is_quota_exhausted(per_minute_exc) is False,
    )
    check(
        "is_retryable_provider_error(503)",
        is_retryable_provider_error(overloaded_exc) is True,
    )
    check(
        "is_retryable_provider_error(per-minute 429)",
        is_retryable_provider_error(per_minute_exc) is True,
    )
    check(
        "is_retryable_provider_error(404) is False",
        is_retryable_provider_error(not_found_exc) is False,
    )
    check(
        "is_model_unavailable(gateway 503 model_not_found) is True",
        is_model_unavailable(gateway_exc) is True,
    )
    check(
        "is_model_unavailable(high-demand 503) is False",
        is_model_unavailable(overloaded_exc) is False,
    )

    quota_error = provider_error("gemini_video", "gemini-3.8-flash", quota_exc)
    text = str(quota_error)
    check(
        "quota error is ProviderError and RunFatalError",
        isinstance(quota_error, ProviderError)
        and isinstance(quota_error, RunFatalError),
    )
    check("quota error names the wait", "3 h 24 min" in text, text)
    check("quota error hints GEMINI_MODEL", "GEMINI_MODEL" in text)
    check("quota error hints GEMINI_API_KEY", "GEMINI_API_KEY" in text)
    check(
        "quota error is not pre-chained",
        quota_error.__cause__ is None,
    )

    missing_error = provider_error("gemini_video", "gemini-3.8-flash", not_found_exc)
    missing_text = str(missing_error)
    check(
        "404 error names the model and GEMINI_MODEL",
        "gemini-3.8-flash" in missing_text and "GEMINI_MODEL" in missing_text,
        missing_text,
    )


def check_llm_retries() -> None:
    from shorts_generator import llm

    settings = Settings(
        llm_max_attempts=3,
        llm_retry_backoff=0.0,
        gemini_api_key="smoke",
        gemini_model="gemini-3.8-flash",
    )

    # (1) A spent daily quota must not be slept through: the provider itself says
    # "retry in 3h24m", so the very first 429 is the answer.
    quota_calls = {"n": 0}

    def spent_quota():
        quota_calls["n"] += 1
        raise FakeClientError(QUOTA_TEXT, code=429)

    try:
        llm._call_with_retries(
            spent_quota, settings, engine="gemini", model=settings.gemini_model
        )
    except ProviderError as exc:
        check(
            "llm stops at once on a spent quota",
            quota_calls["n"] == 1,
            f"calls={quota_calls['n']}",
        )
        check("llm quota error names the wait", "3 h 24 min" in str(exc), str(exc))
    except Exception as exc:  # noqa: BLE001 - any other type is a failure here
        check(
            "llm stops at once on a spent quota", False, f"{type(exc).__name__}: {exc}"
        )
    else:
        check("llm stops at once on a spent quota", False, "no error raised")

    # (2) A transient 503 is still retried, and the call that finally answers wins.
    flaky_calls = {"n": 0}

    def flaky():
        flaky_calls["n"] += 1
        if flaky_calls["n"] < 3:
            raise FakeServerError(UNUSED_503_TEXT, code=503)
        return "ok"

    try:
        result = llm._call_with_retries(
            flaky, settings, engine="gemini", model=settings.gemini_model
        )
    except Exception as exc:  # noqa: BLE001 - reported through ``check`` below
        check("llm retries a 503 and succeeds", False, f"{type(exc).__name__}: {exc}")
    else:
        check("llm retries a 503 and succeeds", result == "ok", repr(result))
        check(
            "llm retried once per transient failure",
            flaky_calls["n"] == 3,
            f"calls={flaky_calls['n']}",
        )

    # (3) Transient on every attempt: the backoff runs out and the cause is
    # surfaced instead of being reported as a per-video failure.
    dead_calls = {"n": 0}

    def always_503():
        dead_calls["n"] += 1
        raise FakeServerError(UNUSED_503_TEXT, code=503)

    try:
        llm._call_with_retries(
            always_503, settings, engine="gemini", model=settings.gemini_model
        )
    except ProviderError as exc:
        text = str(exc)
        check(
            "llm gives up after the last attempt",
            dead_calls["n"] == settings.llm_max_attempts,
            f"calls={dead_calls['n']}",
        )
        check("llm failure counts the attempts", "after 3 attempt(s)" in text, text)
        check(
            "llm failure explains the outage", "never returned a usable answer" in text
        )
    except Exception as exc:  # noqa: BLE001 - any other type is a failure here
        check(
            "llm gives up after the last attempt", False, f"{type(exc).__name__}: {exc}"
        )
    else:
        check("llm gives up after the last attempt", False, "no error raised")


def check_model_unavailable() -> None:
    """An unknown model is a configuration problem, not an outage.

    A real run answered every ranking request with ``503 {'code':
    'model_not_found', 'message': 'Model anthropic/claude-opus-4.5 is temporarily
    unavailable'}`` and spent ten attempts on it (five with ``response_format``,
    five without) plus a minute of log noise, before the batch stopped. The status
    says "later"; the body says "wrong name".
    """
    from shorts_generator import llm

    exc = FakeOpenAIError(GATEWAY_TEXT, status_code=503)

    check("http_status reads .status_code", http_status(exc) == 503)
    check("is_model_unavailable(gateway) is True", is_model_unavailable(exc) is True)
    check(
        "is_retryable_provider_error(gateway) is False",
        is_retryable_provider_error(exc) is False,
    )
    check("is_quota_exhausted(gateway) is False", is_quota_exhausted(exc) is False)

    error = provider_error("deepseek", "anthropic/claude-opus-4.5", exc)
    text = str(error)
    check("unknown model is a ProviderError", isinstance(error, ProviderError))
    check("unknown model error names the model", "anthropic/claude-opus-4.5" in text)
    check("unknown model error hints DEEPSEEK_MODEL", "DEEPSEEK_MODEL" in text)
    check(
        "unknown model error blames the endpoint",
        "not available on this endpoint" in text,
        text,
    )

    # The JSON-mode fallback in ``call_deepseek_llm`` must not fire here: a plain
    # call without ``response_format`` would meet the very same unknown model.
    check(
        "gateway error is not a JSON-mode rejection",
        llm._json_mode_rejected(exc) is False,
    )
    json_exc = FakeOpenAIError(JSON_MODE_TEXT, status_code=400)
    check(
        "response_format rejection is recognised",
        llm._json_mode_rejected(json_exc) is True,
    )

    settings = Settings(
        llm_max_attempts=3,
        llm_retry_backoff=0.0,
        deepseek_model="anthropic/claude-opus-4.5",
    )
    calls = {"n": 0}

    def unknown_model():
        calls["n"] += 1
        raise FakeOpenAIError(GATEWAY_TEXT, status_code=503)

    try:
        llm._call_with_retries(
            unknown_model, settings, engine="deepseek", model=settings.deepseek_model
        )
    except ProviderError as caught:
        check(
            "unknown model is not retried at all",
            calls["n"] == 1,
            f"calls={calls['n']}",
        )
        check(
            "unknown model failure names the model",
            "anthropic/claude-opus-4.5" in str(caught),
            str(caught),
        )
    except Exception as caught:  # noqa: BLE001 - any other type is a failure here
        check(
            "unknown model is not retried at all",
            False,
            f"{type(caught).__name__}: {caught}",
        )
    else:
        check("unknown model is not retried at all", False, "no error raised")


def check_batch_abort() -> None:
    work_dir = tempfile.mkdtemp(prefix="shorts_abort_check_")
    try:
        for name in ("first.mp4", "second.mp4"):
            path = os.path.join(work_dir, name)
            with open(path, "wb"):
                pass

        settings = Settings(input=work_dir, output_dir=work_dir)
        original = pipeline._process_one
        calls: list[str] = []

        def stub(source_path, *args, **kwargs):
            calls.append(source_path)
            raise ProviderError("provider is out of quota")

        pipeline._process_one = stub
        try:
            result = pipeline._run(settings)
        finally:
            pipeline._process_one = original

        check("only the first input was attempted", len(calls) == 1, repr(calls))
        check("result['aborted'] is True", result.get("aborted") is True)
        check(
            "one failure recorded",
            len(result.get("failures") or []) == 1,
            repr(result.get("failures")),
        )
        check(
            "the failure is marked fatal",
            (result.get("failures") or [{}])[0].get("fatal") is True,
        )
        check("abort_reason is set", result.get("abort_reason") is not None)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> int:
    check_classification()
    check_llm_retries()
    check_model_unavailable()
    check_batch_abort()
    if FAILURES:
        print(f"\n{len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
