"""Distinguished LLM failure exceptions used across host + agent.

``LLMAuthError`` → credential broken (401/403). ``HealthMonitor`` counts
these per-agent; threshold crossing triggers quarantine.

``LLMConfigError`` → model not supported by current credential
(400 model_not_found). Task fails with actionable detail; agent is
NOT quarantined (this is operator misconfig, not a broken credential).

``LLMTransientError`` → upstream call hit a transient/retryable
condition the mesh recognizes at the source (Claude subscription
throttle empty-responses, ``RemoteProtocolError`` stream interruptions
wrapped by ``friendly_streaming_error``). The mesh re-emits via
``APIProxyResponse.error_type = "transient"``; the agent consumer
routes that to ``LLMRetryableError`` so ``_llm_call_with_retry`` backs
off. Replaces the substring-tuple-only path that left a new wrapper
message unclassified each time a new transient signal appeared.

``LLMRetryableError`` already exists in :mod:`src.agent.llm` — leave it
there for back-compat (transient failures, rate limits).

All classes inherit ``RuntimeError`` so existing ``except Exception``
handlers keep working without modification.
"""
from __future__ import annotations


class LLMAuthError(RuntimeError):
    """Raised when LLM credentials are broken (401/403) — actionable by rotating creds."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str | None = None,
        http_status: int | None = None,
        raw_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.http_status = http_status
        # Cap the body to avoid blowing up logs with HTML error pages.
        self.raw_body = (raw_body or "")[:500]


class LLMConfigError(RuntimeError):
    """Raised when the model is not supported by the current credential — actionable by switching model."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        allowed_models: set[str] | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.allowed_models = allowed_models or set()
        self.http_status = http_status


class LLMTransientError(RuntimeError):
    """Raised when the LLM call hit a transient/retryable condition — the
    caller should back off and retry rather than fail the task.

    Examples: Claude subscription throttle producing an empty stream,
    ``httpx.RemoteProtocolError`` mid-stream (wrapped by
    ``friendly_streaming_error``), LiteLLM returning no choices.

    Falls through ``_call_llm_with_failover`` the same way bare
    ``RuntimeError`` does today — failover to the next model in the
    chain is attempted first, then the error propagates to
    ``execute_api_call`` which serializes it as
    ``APIProxyResponse(error_type="transient")``.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "unknown",
        model: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        # Seconds — populated when the upstream signal carries explicit
        # backoff guidance. ``_llm_call_with_retry`` honours ``Retry-After``
        # via the existing ``httpx.HTTPStatusError`` path; surfacing this
        # field positions the agent loop to extend that to typed transient
        # errors in a follow-up without changing the exception shape.
        self.retry_after = retry_after
