"""Distinguished LLM failure exceptions used across host + agent.

``LLMAuthError`` → credential broken (401/403). ``HealthMonitor`` counts
these per-agent; threshold crossing triggers quarantine.

``LLMConfigError`` → model not supported by current credential
(400 model_not_found). Task fails with actionable detail; agent is
NOT quarantined (this is operator misconfig, not a broken credential).

``LLMRetryableError`` already exists in :mod:`src.agent.llm` — leave it
there for back-compat (transient failures, rate limits).

Both classes inherit ``RuntimeError`` so existing ``except Exception``
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
