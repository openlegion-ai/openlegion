"""Unit tests for model failover chain."""

import time
from unittest.mock import patch

import pytest

from src.host.failover import FailoverChain, ModelHealthTracker


# ── ModelHealthTracker ────────────────────────────────────────


class TestModelHealthTracker:
    def test_unknown_model_is_available(self):
        tracker = ModelHealthTracker()
        assert tracker.is_available("anthropic/claude-haiku-4-5-20251001") is True

    def test_success_resets_consecutive(self):
        tracker = ModelHealthTracker()
        tracker.record_failure("m1", "timeout", 503)
        tracker.record_failure("m1", "timeout", 503)
        assert tracker._get("m1").consecutive_failures == 2
        tracker.record_success("m1")
        assert tracker._get("m1").consecutive_failures == 0
        assert tracker._get("m1").success_count == 1

    def test_transient_cooldown_exponential(self):
        tracker = ModelHealthTracker()
        # First failure: 60s
        tracker.record_failure("m1", "ServerError", 500)
        h = tracker._get("m1")
        cooldown_1 = h.cooldown_until - time.time()
        assert 55 < cooldown_1 < 65

        # Second failure: 300s
        tracker.record_failure("m1", "ServerError", 500)
        cooldown_2 = h.cooldown_until - time.time()
        assert 295 < cooldown_2 < 305

        # Third failure: 1500s (capped)
        tracker.record_failure("m1", "ServerError", 500)
        cooldown_3 = h.cooldown_until - time.time()
        assert 1495 < cooldown_3 < 1505

        # Fourth failure: still 1500s (cap)
        tracker.record_failure("m1", "ServerError", 500)
        cooldown_4 = h.cooldown_until - time.time()
        assert 1495 < cooldown_4 < 1505

    def test_billing_cooldown_1h(self):
        tracker = ModelHealthTracker()
        tracker.record_failure("m1", "RateLimitError", 429)
        cooldown = tracker._get("m1").cooldown_until - time.time()
        assert 3595 < cooldown < 3605

        tracker2 = ModelHealthTracker()
        tracker2.record_failure("m2", "PaymentRequired", 402)
        cooldown2 = tracker2._get("m2").cooldown_until - time.time()
        assert 3595 < cooldown2 < 3605

    def test_auth_cooldown_1h(self):
        tracker = ModelHealthTracker()
        tracker.record_failure("m1", "AuthenticationError", 401)
        cooldown = tracker._get("m1").cooldown_until - time.time()
        assert 3595 < cooldown < 3605

        tracker.record_failure("m2", "PermissionDenied", 403)
        cooldown2 = tracker._get("m2").cooldown_until - time.time()
        assert 3595 < cooldown2 < 3605

    def test_cooldown_expires(self):
        tracker = ModelHealthTracker()
        tracker.record_failure("m1", "timeout", 500)
        assert tracker.is_available("m1") is False

        # Fast-forward past cooldown
        tracker._get("m1").cooldown_until = time.time() - 1
        assert tracker.is_available("m1") is True

    def test_transient_keyword_matching(self):
        tracker = ModelHealthTracker()
        tracker.record_failure("m1", "ConnectionError", 0)
        cooldown = tracker._get("m1").cooldown_until - time.time()
        # Should use transient exponential (60s for first failure)
        assert 55 < cooldown < 65

    def test_default_cooldown(self):
        tracker = ModelHealthTracker()
        tracker.record_failure("m1", "SomeWeirdError", 418)
        cooldown = tracker._get("m1").cooldown_until - time.time()
        # Default: 60s flat
        assert 55 < cooldown < 65

    def test_get_status(self):
        tracker = ModelHealthTracker()
        tracker.record_success("m1")
        tracker.record_failure("m2", "timeout", 503)
        status = tracker.get_status()
        assert len(status) == 2
        m1 = next(s for s in status if s["model"] == "m1")
        m2 = next(s for s in status if s["model"] == "m2")
        assert m1["available"] is True
        assert m1["success_count"] == 1
        assert m1["failure_count"] == 0
        assert m2["available"] is False
        assert m2["failure_count"] == 1
        assert m2["consecutive_failures"] == 1
        assert m2["cooldown_remaining"] > 0
        assert m2["last_error"] == "timeout"


# ── FailoverChain ─────────────────────────────────────────────


class TestFailoverChain:
    def test_healthy_primary_first(self):
        tracker = ModelHealthTracker()
        chain = FailoverChain(
            chains={"m1": ["m2", "m3"]}, health=tracker,
        )
        result = chain.get_models_to_try("m1")
        assert result == ["m1", "m2", "m3"]

    def test_unhealthy_primary_skipped(self):
        tracker = ModelHealthTracker()
        tracker.record_failure("m1", "timeout", 500)
        chain = FailoverChain(
            chains={"m1": ["m2", "m3"]}, health=tracker,
        )
        result = chain.get_models_to_try("m1")
        assert result == ["m2", "m3"]

    def test_all_cooldown_sorted(self):
        tracker = ModelHealthTracker()
        now = time.time()
        # m1 cools down in 100s, m2 in 50s, m3 in 200s
        tracker._get("m1").cooldown_until = now + 100
        tracker._get("m2").cooldown_until = now + 50
        tracker._get("m3").cooldown_until = now + 200
        chain = FailoverChain(
            chains={"m1": ["m2", "m3"]}, health=tracker,
        )
        result = chain.get_models_to_try("m1")
        assert result == ["m2", "m1", "m3"]

    def test_no_chain_returns_requested(self):
        tracker = ModelHealthTracker()
        chain = FailoverChain(chains={}, health=tracker)
        result = chain.get_models_to_try("some/model")
        assert result == ["some/model"]

    def test_deduplication(self):
        tracker = ModelHealthTracker()
        chain = FailoverChain(
            chains={"m1": ["m1", "m2"]}, health=tracker,
        )
        result = chain.get_models_to_try("m1")
        assert result == ["m1", "m2"]

    def test_partial_cooldown(self):
        tracker = ModelHealthTracker()
        tracker.record_failure("m2", "timeout", 500)
        chain = FailoverChain(
            chains={"m1": ["m2", "m3"]}, health=tracker,
        )
        result = chain.get_models_to_try("m1")
        # m1 and m3 are healthy, m2 is in cooldown
        assert result == ["m1", "m3"]
