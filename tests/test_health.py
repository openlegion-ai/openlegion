"""Tests for the health monitor, including ephemeral agent cleanup."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.host.health import AgentHealth, HealthMonitor


def _make_monitor(agents_info: dict | None = None):
    """Create a HealthMonitor with mocked dependencies."""
    runtime = MagicMock()
    runtime.agents = agents_info or {}
    transport = MagicMock()
    router = MagicMock()
    event_bus = MagicMock()
    monitor = HealthMonitor(
        runtime=runtime, transport=transport, router=router, event_bus=event_bus,
    )
    return monitor


class TestEphemeralCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_ephemeral_expired(self):
        """Agent past TTL is stopped and unregistered."""
        monitor = _make_monitor({
            "spawn-abc": {"ephemeral": True, "ttl": 60, "spawned_at": time.time() - 120},
        })
        monitor.agents["spawn-abc"] = AgentHealth(agent_id="spawn-abc")
        await monitor._cleanup_ephemeral_agents()
        monitor.runtime.stop_agent.assert_called_once_with("spawn-abc")
        monitor.router.unregister_agent.assert_called_once_with("spawn-abc")
        assert "spawn-abc" not in monitor.agents

    @pytest.mark.asyncio
    async def test_cleanup_ephemeral_not_expired(self):
        """Agent within TTL is kept."""
        monitor = _make_monitor({
            "spawn-def": {"ephemeral": True, "ttl": 3600, "spawned_at": time.time()},
        })
        monitor.agents["spawn-def"] = AgentHealth(agent_id="spawn-def")
        await monitor._cleanup_ephemeral_agents()
        monitor.runtime.stop_agent.assert_not_called()
        assert "spawn-def" in monitor.agents

    @pytest.mark.asyncio
    async def test_cleanup_non_ephemeral_untouched(self):
        """Regular (non-ephemeral) agents are not affected by cleanup."""
        monitor = _make_monitor({
            "regular": {"role": "assistant"},
        })
        monitor.agents["regular"] = AgentHealth(agent_id="regular")
        await monitor._cleanup_ephemeral_agents()
        monitor.runtime.stop_agent.assert_not_called()
        assert "regular" in monitor.agents

    @pytest.mark.asyncio
    async def test_cleanup_emits_event(self):
        """Removing an expired ephemeral agent emits an agent_state event."""
        monitor = _make_monitor({
            "spawn-xyz": {"ephemeral": True, "ttl": 10, "spawned_at": time.time() - 100},
        })
        monitor.agents["spawn-xyz"] = AgentHealth(agent_id="spawn-xyz")
        await monitor._cleanup_ephemeral_agents()
        monitor._event_bus.emit.assert_called_once_with(
            "agent_state", agent="spawn-xyz",
            data={"state": "removed", "reason": "ttl_expired"},
        )


class TestHealthRestartMissingConfig:
    @pytest.mark.asyncio
    async def test_restart_skipped_when_truly_unknown(self):
        """Restart for an agent absent from BOTH the runtime registry AND
        agents.yaml sets status to 'failed' and does NOT call start_agent."""
        monitor = _make_monitor({})  # empty registry — no in-memory config
        monitor.register("ghost-agent")
        health = monitor.agents["ghost-agent"]
        health.consecutive_failures = 3
        health.status = "unhealthy"

        # YAML also has no entry for ghost-agent → truly unrecoverable.
        fake_cfg = {"agents": {"other-agent": {"role": "x"}}}
        with patch("src.host.health._load_config", return_value=fake_cfg):
            await monitor._try_restart("ghost-agent")

        assert health.status == "failed"
        monitor.runtime.start_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_marks_failed_when_yaml_load_raises(self):
        """New guard: when the runtime registry has NO entry for the agent
        and the yaml fallback's ``_load_config`` RAISES (corrupt/unreadable
        agents.yaml), ``_info_from_yaml`` swallows the exception and returns
        ``None`` so ``_try_restart`` takes its unrecoverable path cleanly —
        marking the agent 'failed' instead of letting the exception
        propagate out of the health-monitor loop and stalling all other
        agents' checks."""
        monitor = _make_monitor({})  # registry has NO entry for the agent
        monitor.register("ghost-agent")
        health = monitor.agents["ghost-agent"]
        health.consecutive_failures = 3
        health.status = "unhealthy"

        # The yaml fallback read blows up (e.g. corrupt YAML on disk).
        with patch(
            "src.host.health._load_config",
            side_effect=RuntimeError("corrupt yaml"),
        ):
            # Must NOT propagate — no exception escapes _try_restart.
            await monitor._try_restart("ghost-agent")

        assert health.status == "failed"
        monitor.runtime.start_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_falls_back_to_yaml_when_registry_missing(self):
        """Production incident: a container died and was deregistered from the
        runtime's in-memory ``agents`` map during a redeploy, but its config
        still lives in agents.yaml. The restart path must rebuild the
        container from the yaml config (with the agent's configured model)
        instead of giving up and marking it permanently 'failed', which turned
        a transient blip into a permanent outage."""
        monitor = _make_monitor({})  # registry has NO entry for the agent
        monitor.register("content-creator")
        health = monitor.agents["content-creator"]
        health.consecutive_failures = 3
        health.status = "unhealthy"
        monitor.runtime.start_agent.return_value = "http://localhost:8401"
        monitor.runtime.wait_for_agent = AsyncMock(return_value=True)

        fake_cfg = {
            "agents": {
                "content-creator": {
                    "role": "writer",
                    "tools_dir": "tools/content",
                    "model": "anthropic/claude-3-5-sonnet",
                },
            },
            "llm": {"default_model": "openai/gpt-4o-mini"},
        }
        with patch("src.host.health._load_config", return_value=fake_cfg):
            await monitor._try_restart("content-creator")

        # Rebuilt from yaml rather than marked failed.
        monitor.runtime.start_agent.assert_called_once()
        _, kwargs = monitor.runtime.start_agent.call_args
        assert kwargs["agent_id"] == "content-creator"
        assert kwargs["role"] == "writer"
        assert kwargs["model"] == "anthropic/claude-3-5-sonnet"
        assert health.status == "healthy"
        assert health.restart_count == 1

    @pytest.mark.asyncio
    async def test_yaml_fallback_uses_default_model_when_unset(self):
        """When the yaml agent entry omits ``model``, the fallback uses the
        configured llm.default_model (mirrors cli/repl.py:_restart_agent)."""
        monitor = _make_monitor({})
        monitor.register("content-creator")
        health = monitor.agents["content-creator"]
        health.consecutive_failures = 3
        health.status = "unhealthy"
        monitor.runtime.start_agent.return_value = "http://localhost:8401"
        monitor.runtime.wait_for_agent = AsyncMock(return_value=True)

        fake_cfg = {
            "agents": {"content-creator": {"role": "writer"}},
            "llm": {"default_model": "anthropic/claude-3-5-haiku"},
        }
        with patch("src.host.health._load_config", return_value=fake_cfg):
            await monitor._try_restart("content-creator")

        _, kwargs = monitor.runtime.start_agent.call_args
        assert kwargs["model"] == "anthropic/claude-3-5-haiku"

    @pytest.mark.asyncio
    async def test_restart_succeeds_with_config(self):
        """Restart with valid agent metadata proceeds normally."""
        monitor = _make_monitor({
            "good-agent": {"role": "coder", "tools_dir": "/tools"},
        })
        monitor.register("good-agent")
        health = monitor.agents["good-agent"]
        health.consecutive_failures = 3
        health.status = "unhealthy"
        monitor.runtime.start_agent.return_value = "http://localhost:8401"
        monitor.runtime.wait_for_agent = AsyncMock(return_value=True)

        await monitor._try_restart("good-agent")

        monitor.runtime.start_agent.assert_called_once()
        assert health.status == "healthy"
        assert health.restart_count == 1

    @pytest.mark.asyncio
    async def test_restart_propagates_max_output_tokens(self):
        """A crash-recovery restart must carry the operator's per-agent
        output cap into the new container's env (LLM_MAX_TOKENS), or the
        agent silently reverts to the 8192 default on every crash."""
        monitor = _make_monitor({
            "good-agent": {"role": "coder", "tools_dir": "/tools"},
        })
        monitor.register("good-agent")
        health = monitor.agents["good-agent"]
        health.consecutive_failures = 3
        health.status = "unhealthy"
        monitor.runtime.start_agent.return_value = "http://localhost:8401"
        monitor.runtime.wait_for_agent = AsyncMock(return_value=True)

        # The cap lives in YAML, not the registry info dict — patch the
        # fresh-config load the restart path performs.
        fake_cfg = {"agents": {"good-agent": {"max_output_tokens": 32000}}}
        with patch("src.host.health._load_config", return_value=fake_cfg):
            await monitor._try_restart("good-agent")

        _, kwargs = monitor.runtime.start_agent.call_args
        assert kwargs["env_overrides"].get("LLM_MAX_TOKENS") == "32000"


class TestParallelHealthChecks:
    @pytest.mark.asyncio
    async def test_check_all_runs_concurrently(self):
        """_check_all dispatches health checks concurrently via asyncio.gather."""
        import asyncio

        call_times = []

        async def slow_reachable(agent_id, timeout=5):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.1)
            return True

        monitor = _make_monitor({
            "a": {"role": "assistant"},
            "b": {"role": "assistant"},
            "c": {"role": "assistant"},
        })
        monitor.register("a")
        monitor.register("b")
        monitor.register("c")
        monitor.transport.is_reachable = slow_reachable

        t0 = time.monotonic()
        await monitor._check_all()
        elapsed = time.monotonic() - t0

        assert len(call_times) == 3
        # Parallel: total ~0.1s, not ~0.3s
        assert elapsed < 0.25, f"Expected parallel but took {elapsed:.2f}s"
        # All calls started at roughly the same time
        assert max(call_times) - min(call_times) < 0.05


class TestHealthRecoveryEvent:
    @pytest.mark.asyncio
    async def test_recovery_emits_health_change(self):
        """When an unhealthy agent becomes reachable, a health_change event fires."""
        monitor = _make_monitor({"agent-a": {"role": "assistant"}})
        monitor.register("agent-a")
        # Simulate prior unhealthy state
        monitor.agents["agent-a"].status = "unhealthy"
        monitor.agents["agent-a"].consecutive_failures = 2
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        await monitor._check_agent("agent-a")
        assert monitor.agents["agent-a"].status == "healthy"
        monitor._event_bus.emit.assert_called_once_with(
            "health_change", agent="agent-a",
            data={
                "previous": "unhealthy", "current": "healthy",
                "failures": 0, "restart_count": 0,
            },
        )


# ── Seam follow-up Fix 4: quarantine on consecutive auth failures ──


class TestQuarantine:
    """Tests for HealthMonitor.record_auth_failure / clear_quarantine
    / is_quarantined / auto-expiry — the credential-failure quarantine path.

    The mesh quarantines an agent after AUTH_FAILURE_THRESHOLD consecutive
    auth failures so the lane stops dispatching work that will obviously
    fail. Clear is implicit on edit_agent(model) or after the auto-expiry
    TTL — no separate operator tool needed (operator UX principle).
    """

    def test_record_auth_failure_increments_counter(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        quarantined = monitor.record_auth_failure(
            "agent-a", provider="openai", model="openai/gpt-5", http_status=401,
        )
        assert quarantined is False
        assert monitor.agents["agent-a"].consecutive_auth_failures == 1
        assert monitor.agents["agent-a"].quarantined is False

    def test_quarantine_triggers_at_threshold(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        # Default threshold is 3 — first two stay below.
        monitor.record_auth_failure("agent-a", provider="openai", model="x", http_status=401)
        monitor.record_auth_failure("agent-a", provider="openai", model="x", http_status=401)
        assert monitor.agents["agent-a"].quarantined is False
        just_quarantined = monitor.record_auth_failure(
            "agent-a", provider="openai", model="x", http_status=401,
        )
        assert just_quarantined is True
        assert monitor.agents["agent-a"].quarantined is True
        assert monitor.agents["agent-a"].status == "quarantined"
        assert monitor.agents["agent-a"].quarantine_reason is not None
        assert "openai" in monitor.agents["agent-a"].quarantine_reason

    def test_quarantine_emits_event(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        # The event bus is called with the quarantine transition.
        emit_calls = monitor._event_bus.emit.call_args_list
        # Find the health_change → quarantined emit
        quarantine_emit = [
            c for c in emit_calls
            if c.args and c.args[0] == "health_change"
            and c.kwargs.get("data", {}).get("current") == "quarantined"
        ]
        assert len(quarantine_emit) == 1

    def test_quarantine_emit_carries_reason_for_reroute(self):
        """The bell store is gone — the runtime's system-signal reroute
        consumes the health_change emit, so the quarantined transition
        must carry the remediation reason in its payload."""
        monitor = _make_monitor({})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="openai/gpt-5", http_status=401,
            )
        quarantine_emit = [
            c for c in monitor._event_bus.emit.call_args_list
            if c.args and c.args[0] == "health_change"
            and c.kwargs.get("data", {}).get("current") == "quarantined"
        ]
        assert len(quarantine_emit) == 1
        assert quarantine_emit[0].kwargs["data"].get("reason")

    def test_is_quarantined_query(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        monitor.register("agent-b")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        assert monitor.is_quarantined("agent-a") is True
        assert monitor.is_quarantined("agent-b") is False
        assert monitor.is_quarantined("unknown") is False

    def test_clear_quarantine_resets_state(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        cleared = monitor.clear_quarantine("agent-a", reason="model changed")
        assert cleared is True
        assert monitor.agents["agent-a"].quarantined is False
        assert monitor.agents["agent-a"].quarantine_reason is None
        assert monitor.agents["agent-a"].consecutive_auth_failures == 0
        assert monitor.agents["agent-a"].status == "healthy"
        # Emits a clear event.
        emit_calls = monitor._event_bus.emit.call_args_list
        clear_emits = [
            c for c in emit_calls
            if c.args and c.args[0] == "health_change"
            and c.kwargs.get("data", {}).get("current") == "healthy"
            and c.kwargs.get("data", {}).get("previous") == "quarantined"
        ]
        assert len(clear_emits) == 1

    def test_clear_quarantine_noop_when_not_quarantined(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        assert monitor.clear_quarantine("agent-a", reason="test") is False

    def test_clear_quarantine_resets_counter_when_not_quarantined(self):
        """Codex P2 r3: clear_quarantine must reset the pre-threshold
        auth-failure counter even when the agent isn't quarantined yet.

        Without this, an agent with 2 partial failures (below threshold)
        whose model is then changed would quarantine on the very next
        failure on the new model — using stale counts from credentials
        that no longer apply. The boolean return value still reflects
        the quarantine-flag transition (False here), NOT the counter.
        """
        monitor = _make_monitor({})
        monitor.register("agent-a")
        # Two partial auth failures — below the threshold of 3.
        monitor.record_auth_failure(
            "agent-a", provider="openai", model="x", http_status=401,
        )
        monitor.record_auth_failure(
            "agent-a", provider="openai", model="x", http_status=401,
        )
        assert monitor.agents["agent-a"].consecutive_auth_failures == 2
        assert monitor.agents["agent-a"].quarantined is False
        # Operator changes the model → clear_quarantine called from the
        # edit-soft hook. Returns False (no quarantine flag to flip) but
        # MUST reset the counter so the next failure doesn't trip the
        # threshold using stale history.
        result = monitor.clear_quarantine("agent-a", reason="model changed")
        assert result is False
        assert monitor.agents["agent-a"].consecutive_auth_failures == 0

    def test_clear_quarantine_unknown_agent_returns_false(self):
        """Codex P2 r3 guardrail: clearing an unregistered agent is a
        clean no-op — return False, no exception."""
        monitor = _make_monitor({})
        assert monitor.clear_quarantine("ghost", reason="test") is False

    def test_clear_quarantine_preserves_failed_status(self):
        """Codex r4 (principal-eng): a credential-side clear (TTL expiry or
        operator edit_agent) must NOT *itself* revive a ``failed`` agent to
        ``healthy``. ``failed`` recovery is owned by ``_check_agent``'s
        reachability probe (the agent self-heals once it answers /status
        again) — a quarantine clear is not a liveness signal, so it leaves
        ``failed`` in place and the operator keeps seeing ``failed`` until a
        real probe says otherwise."""
        monitor = _make_monitor({})
        monitor.register("agent-a")
        # Force quarantine.
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        # Independently drive the runtime to ``failed`` (e.g. restart
        # budget exhausted while quarantined).
        monitor.agents["agent-a"].status = "failed"
        monitor.agents["agent-a"].consecutive_failures = 10

        cleared = monitor.clear_quarantine("agent-a", reason="auto-expiry")
        assert cleared is True
        assert monitor.agents["agent-a"].quarantined is False
        # clear_quarantine itself must not revive failed — recovery is owned
        # by _check_agent's reachability probe, not a credential-side clear.
        assert monitor.agents["agent-a"].status == "failed"

    def test_clear_quarantine_preserves_unhealthy_status(self):
        """Same principle for ``unhealthy``: reachability is genuinely
        broken (the failure counter is non-zero), so clearing the
        credential signal alone must not paint the agent green. The next
        successful ``_check_agent`` poll is what reconciles to healthy."""
        monitor = _make_monitor({})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        # Simulate concurrent unreachable polls accumulating failures.
        monitor.agents["agent-a"].consecutive_failures = 2

        cleared = monitor.clear_quarantine("agent-a", reason="model changed")
        assert cleared is True
        assert monitor.agents["agent-a"].quarantined is False
        # Reachability counter > 0 → keep ``unhealthy``; next poll
        # reconciles to ``healthy`` if the agent actually responds.
        assert monitor.agents["agent-a"].status == "unhealthy"

    def test_clear_quarantine_preserves_restarting_status(self):
        """``restarting`` is an in-flight state the restart path
        reconciles on its own — clearing the credential mid-restart must
        not pre-empt that to ``healthy``."""
        monitor = _make_monitor({})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        # An in-flight restart resets consecutive_failures to 0 (see
        # health.py:531) and sets status="restarting".
        monitor.agents["agent-a"].status = "restarting"
        monitor.agents["agent-a"].consecutive_failures = 0

        cleared = monitor.clear_quarantine("agent-a", reason="model changed")
        assert cleared is True
        # The restart path will set the post-restart status — we just
        # don't preempt it with a stale "healthy".
        assert monitor.agents["agent-a"].status == "restarting"

    def test_auto_expiry_clears_old_quarantine(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        # Backdate the quarantine timestamp past the TTL.
        monitor.agents["agent-a"].quarantined_at = time.time() - (
            monitor.QUARANTINE_AUTO_CLEAR_SECONDS + 60
        )
        monitor._maybe_expire_quarantines(time.time())
        assert monitor.agents["agent-a"].quarantined is False

    def test_get_status_surfaces_quarantine_fields(self):
        monitor = _make_monitor({})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        statuses = monitor.get_status()
        agent_status = next(s for s in statuses if s["agent"] == "agent-a")
        assert agent_status["quarantined"] is True
        assert agent_status["quarantine_reason"] is not None
        assert agent_status["consecutive_auth_failures"] == 3

    @pytest.mark.asyncio
    async def test_reachability_poll_preserves_quarantined_status(self):
        """Codex P2 follow-up: a successful reachability poll must NOT
        flip a quarantined agent back to healthy — the agent is
        reachable but its credentials are broken, which is what lane/cron
        are skipping on. Only clear_quarantine should flip status."""
        monitor = _make_monitor({"agent-a": {"role": "x"}})
        monitor.register("agent-a")
        # Quarantine the agent.
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        assert monitor.agents["agent-a"].status == "quarantined"
        # Successful reachability poll.
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        await monitor._check_agent("agent-a")
        # Status must remain quarantined.
        assert monitor.agents["agent-a"].status == "quarantined"
        assert monitor.agents["agent-a"].quarantined is True

    @pytest.mark.asyncio
    async def test_unreachable_poll_preserves_quarantined_status(self):
        """Same for unreachable polls — the reachability counter ticks
        but the status string stays quarantined."""
        monitor = _make_monitor({"agent-a": {"role": "x"}})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        assert monitor.agents["agent-a"].status == "quarantined"
        monitor.transport.is_reachable = AsyncMock(return_value=False)
        await monitor._check_agent("agent-a")
        # Reachability counter still ticks.
        assert monitor.agents["agent-a"].consecutive_failures >= 1
        # Status string stays quarantined.
        assert monitor.agents["agent-a"].status == "quarantined"

    @pytest.mark.asyncio
    async def test_try_restart_preserves_quarantined_status_when_ready(self):
        """Principal-eng follow-up: _try_restart was the one path that
        clobbered status to "healthy" on a successful restart even when
        the agent was still quarantined. The lane gate is the bool flag
        and still rejects work — but the dashboard renders ``status`` and
        would lie about availability. Restart should restore the runtime
        without touching the credential-broken signal."""
        monitor = _make_monitor({"agent-a": {"role": "x"}})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        assert monitor.agents["agent-a"].quarantined is True
        monitor.runtime.start_agent.return_value = "http://localhost:8401"
        monitor.runtime.wait_for_agent = AsyncMock(return_value=True)
        monitor.agents["agent-a"].consecutive_failures = 3

        await monitor._try_restart("agent-a")

        assert monitor.agents["agent-a"].status == "quarantined"
        assert monitor.agents["agent-a"].quarantined is True
        assert monitor.agents["agent-a"].restart_count == 1

    @pytest.mark.asyncio
    async def test_try_restart_falls_to_unhealthy_when_not_ready_even_if_quarantined(self):
        """When the runtime fails to come back up, ``unhealthy`` wins —
        more urgent operator signal than ``quarantined``. The lane gate
        (bool flag) is unchanged either way; this is purely about which
        status string surfaces."""
        monitor = _make_monitor({"agent-a": {"role": "x"}})
        monitor.register("agent-a")
        for _ in range(3):
            monitor.record_auth_failure(
                "agent-a", provider="openai", model="x", http_status=401,
            )
        monitor.runtime.start_agent.return_value = "http://localhost:8401"
        monitor.runtime.wait_for_agent = AsyncMock(return_value=False)
        monitor.agents["agent-a"].consecutive_failures = 3

        await monitor._try_restart("agent-a")

        assert monitor.agents["agent-a"].status == "unhealthy"
        # The bool flag is the real lane gate; restart doesn't fix creds.
        assert monitor.agents["agent-a"].quarantined is True


class TestFailedSelfHeal:
    """Regression: ``failed`` was a terminal sink — a restart storm (transient
    upstream-LLM overload timing out /status probes) marked agents ``failed``
    and they were never re-probed, so they showed ``Offline`` forever even
    after becoming reachable. ``_check_agent`` must now re-probe ``failed``
    agents and self-heal them to ``healthy`` once reachable, while leaving the
    windowed restart budget intact so a chronic flapper can't earn unbounded
    restarts.
    """

    @pytest.mark.asyncio
    async def test_failed_agent_recovers_when_reachable(self):
        """A ``failed`` agent flips to ``healthy`` after one reachable probe
        and emits a failed->healthy event. The restart ledger is preserved
        (NOT reset) so the budget stays window-bounded."""
        monitor = _make_monitor({"agent-a": {"role": "x"}})
        monitor.register("agent-a")
        h = monitor.agents["agent-a"]
        h.status = "failed"
        h.restart_timestamps = [1.0, 2.0, 3.0]
        h.consecutive_failures = 3
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        monitor._event_bus.emit.reset_mock()

        await monitor._check_agent("agent-a")

        assert h.status == "healthy"
        # Restart ledger preserved on self-heal (NOT reset): the budget is
        # bounded by the RESTART_WINDOW prune, so a flapper can't loop restarts.
        assert h.restart_timestamps == [1.0, 2.0, 3.0]
        assert h.consecutive_failures == 0
        monitor._event_bus.emit.assert_called_once()
        evt_name, kwargs = (
            monitor._event_bus.emit.call_args.args,
            monitor._event_bus.emit.call_args.kwargs,
        )
        assert evt_name == ("health_change",)
        assert kwargs["data"]["previous"] == "failed"
        assert kwargs["data"]["current"] == "healthy"

    @pytest.mark.asyncio
    async def test_failed_agent_stays_failed_when_unreachable_no_churn(self):
        """A still-unreachable ``failed`` agent stays ``failed`` with no churn:
        no ``consecutive_failures`` bump, no new restart attempt, no flap."""
        monitor = _make_monitor({"agent-a": {"role": "x"}})
        monitor.register("agent-a")
        h = monitor.agents["agent-a"]
        h.status = "failed"
        h.restart_timestamps = [1.0, 2.0, 3.0]
        h.consecutive_failures = 3
        monitor.transport.is_reachable = AsyncMock(return_value=False)
        monitor._try_restart = AsyncMock()
        monitor._event_bus.emit.reset_mock()

        await monitor._check_agent("agent-a")

        assert h.status == "failed"
        assert h.consecutive_failures == 3
        assert h.restart_timestamps == [1.0, 2.0, 3.0]
        monitor._try_restart.assert_not_called()
        monitor._event_bus.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_reachable_but_stale_agent_falls_through_to_restart_path(self):
        """A ``failed`` agent whose /status answers but whose loop is wedged
        (stale) must NOT be parked as failed by the no-churn guard — it falls
        through to the unhealthy/restart path so the wedged loop can eventually
        be restarted once the window frees."""
        monitor = _make_monitor({"agent-a": {"role": "x"}})
        monitor.register("agent-a")
        h = monitor.agents["agent-a"]
        h.status = "failed"
        h.consecutive_failures = 0
        monitor.transport.is_reachable = AsyncMock(return_value=True)
        monitor._is_loop_stale = AsyncMock(return_value=True)
        monitor._try_restart = AsyncMock()

        await monitor._check_agent("agent-a")

        # Fell through (not silently parked): the failure counter advanced and
        # status reflects the still-broken wedged loop rather than staying failed.
        assert h.consecutive_failures == 1
        assert h.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_failed_agent_probe_exception_stays_failed_no_churn(self):
        """If the reachability probe raises for a ``failed`` agent, it stays
        ``failed`` quietly — same no-churn guard as the unreachable path."""
        monitor = _make_monitor({"agent-a": {"role": "x"}})
        monitor.register("agent-a")
        h = monitor.agents["agent-a"]
        h.status = "failed"
        h.consecutive_failures = 3
        h.restart_timestamps = [1.0, 2.0, 3.0]
        monitor.transport.is_reachable = AsyncMock(side_effect=RuntimeError("boom"))
        monitor._try_restart = AsyncMock()
        monitor._event_bus.emit.reset_mock()

        await monitor._check_agent("agent-a")

        assert h.status == "failed"
        assert h.consecutive_failures == 3
        assert h.restart_timestamps == [1.0, 2.0, 3.0]
        monitor._try_restart.assert_not_called()
        monitor._event_bus.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_quarantined_failed_agent_stays_quarantined_when_reachable(self):
        """Quarantine invariant: a reachable poll must NOT flip a quarantined
        (and failed) agent to ``healthy`` — credentials are still broken."""
        monitor = _make_monitor({"agent-a": {"role": "x"}})
        monitor.register("agent-a")
        h = monitor.agents["agent-a"]
        h.status = "failed"
        h.quarantined = True
        monitor.transport.is_reachable = AsyncMock(return_value=True)

        await monitor._check_agent("agent-a")

        # Reachable clears the failure counters but the quarantined+failed
        # status is preserved — only clear_quarantine flips it back to healthy.
        assert h.quarantined is True
        assert h.status == "failed"
