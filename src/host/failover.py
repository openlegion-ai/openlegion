"""Model failover chain for LLM API proxy.

When a model is down, rate-limited, or its API key is exhausted, the vault
cascades to the next model in a configured failover chain. Transparent to
agents — they just see a response from whichever model succeeded.

State is in-memory (not SQLite). A clean slate on restart is correct: stale
cooldowns from a previous run should not block models that may now be healthy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.shared.utils import setup_logging

logger = setup_logging("host.failover")

# Cooldown durations in seconds, by error category
_BILLING_COOLDOWN = 3600  # 402, 429 — 1 hour
_AUTH_COOLDOWN = 3600  # 401, 403 — 1 hour
_DEFAULT_COOLDOWN = 60  # Catch-all
_TRANSIENT_MAX_COOLDOWN = 1500  # Cap for exponential backoff

# Status codes by category
_BILLING_CODES = {402, 429}
_AUTH_CODES = {401, 403}
_TRANSIENT_CODES = {500, 502, 503}
_TRANSIENT_KEYWORDS = {"connection", "timeout", "connecterror", "readtimeout"}


@dataclass
class ModelHealth:
    model: str
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_error: str = ""
    last_error_time: float = 0.0
    last_success_time: float = 0.0


class ModelHealthTracker:
    """Tracks per-model health and applies cooldown windows."""

    def __init__(self) -> None:
        self._models: dict[str, ModelHealth] = {}

    def _get(self, model: str) -> ModelHealth:
        if model not in self._models:
            self._models[model] = ModelHealth(model=model)
        return self._models[model]

    def is_available(self, model: str) -> bool:
        """Check if a model is past its cooldown window."""
        h = self._get(model)
        return time.time() >= h.cooldown_until

    def record_success(self, model: str) -> None:
        h = self._get(model)
        h.success_count += 1
        h.consecutive_failures = 0
        h.cooldown_until = 0.0
        h.last_success_time = time.time()

    def record_failure(
        self, model: str, error_type: str = "", status_code: int = 0,
    ) -> None:
        h = self._get(model)
        h.failure_count += 1
        h.consecutive_failures += 1
        h.last_error = error_type
        h.last_error_time = time.time()

        cooldown = self._compute_cooldown(h, error_type, status_code)
        h.cooldown_until = time.time() + cooldown
        logger.warning(
            f"Model '{model}' failed (#{h.consecutive_failures}), "
            f"cooldown {cooldown:.0f}s",
            extra={"model": model, "status_code": status_code, "cooldown": cooldown},
        )

    def _compute_cooldown(
        self, h: ModelHealth, error_type: str, status_code: int,
    ) -> float:
        if status_code in _BILLING_CODES:
            return _BILLING_COOLDOWN
        if status_code in _AUTH_CODES:
            return _AUTH_COOLDOWN
        if status_code in _TRANSIENT_CODES or self._is_transient_keyword(error_type):
            # Exponential: 60, 300, 1500, 1500, ...
            raw = 60 * (5 ** (h.consecutive_failures - 1))
            return min(raw, _TRANSIENT_MAX_COOLDOWN)
        return _DEFAULT_COOLDOWN

    @staticmethod
    def _is_transient_keyword(error_type: str) -> bool:
        lower = error_type.lower()
        return any(kw in lower for kw in _TRANSIENT_KEYWORDS)

    def get_status(self) -> list[dict]:
        now = time.time()
        return [
            {
                "model": h.model,
                "available": now >= h.cooldown_until,
                "success_count": h.success_count,
                "failure_count": h.failure_count,
                "consecutive_failures": h.consecutive_failures,
                "cooldown_remaining": max(0.0, h.cooldown_until - now),
                "last_error": h.last_error,
            }
            for h in self._models.values()
        ]


class FailoverChain:
    """Resolves an ordered list of models to try, respecting cooldowns."""

    def __init__(
        self,
        chains: dict[str, list[str]],
        health: ModelHealthTracker,
    ) -> None:
        self._chains = chains
        self._health = health

    def get_models_to_try(self, requested_model: str) -> list[str]:
        """Return models in priority order, filtering out cooled-down ones.

        If ALL models (primary + fallbacks) are in cooldown, return all of
        them sorted by least remaining cooldown time (best-effort).
        """
        candidates = [requested_model]
        candidates.extend(self._chains.get(requested_model, []))
        # De-duplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for m in candidates:
            if m not in seen:
                seen.add(m)
                unique.append(m)

        available = [m for m in unique if self._health.is_available(m)]
        if available:
            return available

        # All in cooldown — sort by least remaining time
        now = time.time()
        unique.sort(key=lambda m: self._health._get(m).cooldown_until - now)
        return unique
