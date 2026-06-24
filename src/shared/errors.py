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

import re


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


# Substrings (matched case-insensitively) that identify a context-length /
# "prompt is too long" 400 across providers (Anthropic, OpenAI, OpenRouter
# passthrough). Lives in shared/ because BOTH sides of the trust boundary need
# it: the mesh proxy (which has the raw provider detail) tags
# ``error_type="context_overflow"`` on the masked envelope, and the agent's LLM
# classifier uses it as a backstop. One list keeps the two copies from drifting.
CONTEXT_OVERFLOW_MARKERS = (
    "prompt is too long",
    "context length exceeded",
    "context_length_exceeded",
    "input length and max_tokens exceed",
    "maximum context length",
)


def is_context_overflow(text: str) -> bool:
    """True when an error message indicates the request exceeded the model's
    context window (a context-length 400).

    Used by the mesh proxy to tag ``error_type="context_overflow"`` (the proxy
    masks the raw message across the trust boundary, so the agent cannot
    substring-match it itself) and by ``LLMClient._raise_classified_error`` as a
    backstop for any path that does forward the raw text. Routing to
    ``LLMContextOverflowError`` lets the chat loop self-heal (prune + retry)
    instead of aborting the turn and re-wedging.
    """
    # Strip backticks first: Anthropic also emits "input length and
    # `max_tokens` exceed context limit...", which would defeat the marker.
    t = (text or "").lower().replace("`", "")
    return any(marker in t for marker in CONTEXT_OVERFLOW_MARKERS)


# Anthropic phrases the overflow as "prompt is too long: 1000961 tokens >
# 1000000 maximum". When that detail survives to the agent (the streaming SSE
# path forwards it un-masked), the actual token count is ground truth we can
# use to CALIBRATE our local estimate — which is content-dependent and can
# undershoot the real tokenizer by ~2x on dense CSV/JSON/code, leaving the
# emergency prune a no-op. See ContextManager.calibrate_from_overflow.
_OVERFLOW_COUNT_RE = re.compile(r"(\d[\d,]*)\s*tokens?\s*>\s*(\d[\d,]*)")


def parse_overflow_tokens(text: str) -> tuple[int, int] | None:
    """Parse ``(actual_tokens, limit_tokens)`` from a context-overflow message.

    Returns ``None`` when the message carries no counts (e.g. the non-streaming
    proxy path masks the detail to "Upstream service call failed."). Callers
    must handle ``None`` by forcing progress some other way.
    """
    if not text:
        return None
    m = _OVERFLOW_COUNT_RE.search(text)
    if not m:
        return None
    try:
        actual = int(m.group(1).replace(",", ""))
        limit = int(m.group(2).replace(",", ""))
    except (ValueError, AttributeError):
        return None
    if actual <= 0 or limit <= 0:
        return None
    return actual, limit
