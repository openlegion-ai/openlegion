"""Tests for the post-refactor captcha accounting + metering architecture.

Covers the C1/C2/H1/H2/H3/H4 findings from the §11.13–§11.16 post-merge
review:

  * Auto-detect via ``_check_captcha`` consults the rate-limit + cost-cap
    gates BEFORE the solver HTTP call. Pre-refactor, the gates fired
    only inside ``solve_captcha`` and ran AFTER ``_check_captcha`` had
    already invoked the solver — defeating their purpose.
  * Concurrent solves on different agents do NOT clobber each other's
    ``used_proxy_aware`` / ``compat_rejected`` flags. The old
    per-instance scratch attrs raced; the new :class:`SolveResult`
    dataclass is per-call so this is structurally impossible.
  * Token-retrieved + injection-failed → ``solver_outcome="injection_failed"``,
    cost IS counted. The provider was paid the moment the token came back.
  * CF-Turnstile (``cf-interstitial-turnstile``) cost is counted at the
    Turnstile rate (no spurious "no published rate" warning).
  * CF-auto cleared → no "no published rate" warning logged
    (``solver_attempted=False`` paths skip the warning).
  * Eager / lazy health-check: pre-marked unreachable → no_solver envelope,
    solver mock NOT contacted.
  * Audit-log: cap / rate-limit / behavioral events emit on the
    ``metrics_sink`` (EventBus production wiring) once per minute aggregation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser import captcha_cost_counter as cost
from src.browser import service as svc
from src.browser.captcha import SolveResult
from src.browser.service import BrowserManager, CamoufoxInstance

# ── helpers ────────────────────────────────────────────────────────────────


def _solved(*, used_proxy_aware: bool = False, compat_rejected: bool = False) -> SolveResult:
    return SolveResult(
        token="tok",
        injection_succeeded=True,
        used_proxy_aware=used_proxy_aware,
        compat_rejected=compat_rejected,
    )


def _injection_failed() -> SolveResult:
    """Token retrieved but DOM injection failed — provider charged us."""
    return SolveResult(
        token="tok",
        injection_succeeded=False,
        used_proxy_aware=False,
        compat_rejected=False,
    )


def _mk_inst(*, captcha_present: bool = True, agent_id: str = "agent-1",
             page_url: str = "https://example.com") -> CamoufoxInstance:
    """Build a CamoufoxInstance with a page that reports captcha presence."""
    page = MagicMock()
    page.url = page_url
    locator = MagicMock()
    locator.count = AsyncMock(return_value=1 if captcha_present else 0)
    page.locator = MagicMock(return_value=locator)
    return CamoufoxInstance(agent_id, MagicMock(), MagicMock(), page)


def _mk_solver(*, return_value: SolveResult, provider: str = "2captcha",
               unreachable: bool = False, breaker_open: bool = False) -> MagicMock:
    s = MagicMock()
    s.provider = provider
    s.solve = AsyncMock(return_value=return_value)
    s.is_solver_unreachable = AsyncMock(return_value=unreachable)
    s.is_breaker_open = MagicMock(return_value=breaker_open)
    return s


@pytest.fixture(autouse=True)
async def _isolate_state(tmp_path, monkeypatch):
    """Reset the in-memory cost / rate / audit state between tests."""
    monkeypatch.setenv(
        "CAPTCHA_COST_COUNTER_PATH", str(tmp_path / "captcha_costs.json"),
    )
    await cost.reset()
    svc._solve_rate_window.clear()
    async with svc._get_captcha_audit_lock():
        svc._captcha_audit_buckets.clear()
    yield
    await cost.reset()
    svc._solve_rate_window.clear()
    async with svc._get_captcha_audit_lock():
        svc._captcha_audit_buckets.clear()


@pytest.fixture()
def mgr(tmp_path):
    return BrowserManager(profiles_dir=str(tmp_path / "profiles"))


# ── 1. Auto-detect path: cost cap fires BEFORE solver HTTP call ───────────


class TestAutoDetectGatesEnforced:
    """Pre-refactor, ``_check_captcha`` (called by navigate / click /
    detect_captcha) bypassed the rate-limit + cost-cap entirely; only
    ``solve_captcha`` enforced them, and even there the gates ran AFTER
    the solver call. The metered_solve refactor moves the gates inside
    ``_check_captcha`` so EVERY entry point (navigate auto-detect, click
    auto-detect, explicit solve_captcha) hits them.
    """

    @pytest.mark.asyncio
    async def test_navigate_path_cost_cap_blocks_solver(self, mgr, monkeypatch):
        """``_check_captcha`` direct call (the navigate auto-detect path)
        respects the cost cap."""
        # $0.50 cap → 50_000 millicents. Pre-fill 50_000+1 to clear it.
        # The cost counter stores spend in MILLICENTS (1/1000 of a cent
        # = 1/100_000 of a dollar); see ``captcha_cost_counter`` docstring.
        monkeypatch.setenv("CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", "0.50")
        await cost.add_cost("agent-1", 50_000)  # already over cap

        solver = _mk_solver(return_value=_solved())
        mgr._captcha_solver = solver
        inst = _mk_inst()

        envelope = await mgr._check_captcha(inst)
        assert envelope["captcha_found"] is True
        assert envelope["solver_outcome"] == "cost_cap"
        # The solver mock MUST NOT have been called — gate fires before HTTP.
        solver.solve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_navigate_path_rate_limit_blocks_solver(self, mgr, monkeypatch):
        """Navigate auto-detect respects the rate-limit gate."""
        monkeypatch.setenv("CAPTCHA_RATE_LIMIT_PER_HOUR", "2")
        now = time.time()
        svc._solve_rate_window["agent-1"] = deque([now, now])

        solver = _mk_solver(return_value=_solved())
        mgr._captcha_solver = solver
        inst = _mk_inst()

        envelope = await mgr._check_captcha(inst)
        assert envelope["captcha_found"] is True
        assert envelope["solver_outcome"] == "rate_limited"
        solver.solve.assert_not_awaited()


# ── 2. Concurrent solves on different agents don't clobber state ──────────


class TestConcurrentSolvesNoCrossAgentClobber:
    """Pre-refactor the shared solver instance had per-call mutable state
    (``last_used_proxy_aware`` / ``last_compat_rejected``) — concurrent
    agents could overwrite each other's flags between stamp and read.
    The :class:`SolveResult` dataclass is per-call so the race is
    structurally impossible.
    """

    @pytest.mark.asyncio
    async def test_two_agents_get_independent_solveresult(self, mgr, tmp_path):
        # Two real instances on the same manager.
        inst_a = _mk_inst(agent_id="agent-a")
        inst_b = _mk_inst(agent_id="agent-b")
        mgr._instances["agent-a"] = inst_a
        mgr._instances["agent-b"] = inst_b

        # Solver returns DIFFERENT SolveResults for each call. The
        # AsyncMock side-effect cycles through these, so a serial
        # gather of two concurrent solves yields one of each.
        result_a = SolveResult(
            token="tok-a", injection_succeeded=True,
            used_proxy_aware=True, compat_rejected=False,
        )
        result_b = SolveResult(
            token="tok-b", injection_succeeded=True,
            used_proxy_aware=False, compat_rejected=True,
        )
        solver = MagicMock()
        solver.provider = "2captcha"
        solver.is_solver_unreachable = AsyncMock(return_value=False)
        solver.is_breaker_open = MagicMock(return_value=False)
        solver.solve = AsyncMock(side_effect=[result_a, result_b])
        mgr._captcha_solver = solver

        env_a, env_b = await asyncio.gather(
            mgr._check_captcha(inst_a),
            mgr._check_captcha(inst_b),
        )

        # Both agents see "solved"; both envelopes were built from
        # their OWN per-call SolveResult — no cross-agent flag clobber.
        assert env_a["solver_outcome"] == "solved"
        assert env_b["solver_outcome"] == "solved"
        # Confidence reflects each call's compat_rejected flag separately:
        # agent_a's call had compat_rejected=False (high confidence);
        # agent_b's call had compat_rejected=True (low confidence).
        assert env_a["solver_confidence"] == "high"
        assert env_b["solver_confidence"] == "low"


# ── 3. Token retrieved + injection failed → injection_failed + count cost ─


class TestInjectionFailedCountsCost:
    """When the provider returns a valid token but our DOM injection
    fails, the §11.13 envelope reports ``solver_outcome="injection_failed"``
    AND the cost counter increments — because the provider was paid.
    Pre-refactor, cost increment was gated on ``injection_succeeded`` so
    the failed-injection case silently dropped a real billing event.
    """

    @pytest.mark.asyncio
    async def test_injection_failed_envelope_and_cost(self, mgr):
        solver = _mk_solver(return_value=_injection_failed())
        mgr._captcha_solver = solver
        inst = _mk_inst()

        envelope = await mgr._check_captcha(inst)
        assert envelope["solver_outcome"] == "injection_failed"
        assert envelope["solver_attempted"] is True
        assert envelope["next_action"] == "notify_user"
        assert envelope["injection_failure_reason"] == "injection_failed_unspecified"
        # Cost IS counted — provider was paid.
        # First selector match is recaptcha → kind=recaptcha-v2-checkbox →
        # 100 millicents at 2captcha proxyless.
        assert await cost.get_cents("agent-1") == 100


# ── 4. CF-Turnstile pricing is in the table ───────────────────────────────


class TestCfTurnstilePricing:
    @pytest.mark.asyncio
    async def test_cf_turnstile_solved_increments_cost(self, mgr, monkeypatch):
        """CF-bound Turnstile (``cf-interstitial-turnstile``) takes the same
        rate as standalone Turnstile. Without the alias entries added in
        the refactor, the cost counter would emit a "no published rate"
        warning and skip the increment for every CF-Turnstile solve.
        """
        # Force the CF tri-state classifier to land on "turnstile".
        async def fake_cf_state(page):
            return "turnstile"
        async def fake_behavioral(page):
            return None
        async def fake_recaptcha(page):
            return {"variant": "unknown", "sitekey": None, "action": None,
                    "min_score": None}

        monkeypatch.setattr(svc, "_classify_cf_state", fake_cf_state)
        monkeypatch.setattr(svc, "_classify_behavioral", fake_behavioral)
        monkeypatch.setattr(svc, "_classify_recaptcha", fake_recaptcha)

        solver = _mk_solver(return_value=_solved(), provider="2captcha")
        mgr._captcha_solver = solver

        # Custom inst whose locator only matches the CF challenge selector.
        page = MagicMock()
        page.url = "https://example.com"
        def loc_for(sel):
            loc = MagicMock()
            loc.count = AsyncMock(return_value=(
                1 if sel == 'iframe[src*="challenges.cloudflare.com"]' else 0
            ))
            return loc
        page.locator = MagicMock(side_effect=loc_for)
        inst = CamoufoxInstance("agent-1", MagicMock(), MagicMock(), page)

        envelope = await mgr._check_captcha(inst)
        assert envelope["kind"] == "cf-interstitial-turnstile"
        assert envelope["solver_outcome"] == "solved"
        # 2captcha turnstile = 200 millicents. CF-bound alias picks up the same rate.
        assert await cost.get_cents("agent-1") == 200


# ── 5. CF-auto cleared → no "no published rate" warning logged ────────────


class TestCfAutoNoPriceWarning:
    @pytest.mark.asyncio
    async def test_cf_auto_cleared_emits_no_warning(self, mgr, caplog, monkeypatch):
        """``cf-interstitial-auto`` resolves without a solver call —
        ``solver_attempted=False`` paths must NOT trigger the cost
        counter's "no published rate" warning. Pre-refactor a CF-auto
        clear path could trip the warning if any code path tried to
        estimate cost on a non-priced kind."""
        # Drive the CF classifier to "auto" + still_present=False so the
        # path returns the "solved" envelope without any solver call.
        async def fake_cf_state(page):
            return "auto"
        async def fake_behavioral(page):
            return None
        async def fake_recaptcha(page):
            return {"variant": "unknown"}

        monkeypatch.setattr(svc, "_classify_cf_state", fake_cf_state)
        monkeypatch.setattr(svc, "_classify_behavioral", fake_behavioral)
        monkeypatch.setattr(svc, "_classify_recaptcha", fake_recaptcha)

        # Page locator: first match present, post-recheck absent.
        page = MagicMock()
        page.url = "https://example.com"
        call_counts: dict[str, int] = {}

        def loc_for(sel):
            loc = MagicMock()

            async def _count():
                n = call_counts.get(sel, 0)
                call_counts[sel] = n + 1
                if sel != 'iframe[src*="challenges.cloudflare.com"]':
                    return 0
                return 1 if n == 0 else 0  # cleared on recheck
            loc.count = _count
            return loc

        page.locator = MagicMock(side_effect=loc_for)
        inst = CamoufoxInstance("agent-1", MagicMock(), MagicMock(), page)

        # No solver mock at all on the manager — but it shouldn't matter,
        # the auto-path returns before any solver branch.
        mgr._captcha_solver = None

        # Patch asyncio.sleep inside _check_captcha so the 8s wait collapses.
        from unittest.mock import patch
        with caplog.at_level(logging.WARNING, logger="browser.captcha_cost"), \
             patch("src.browser.service.asyncio.sleep", new=AsyncMock()):
            envelope = await mgr._check_captcha(inst)

        assert envelope["solver_outcome"] == "solved"
        assert envelope["kind"] == "cf-interstitial-auto"
        # No "no published rate" warning emitted.
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "no published rate" not in joined


# ── 6. Eager/lazy health-check — first captcha sees the gate ──────────────


class TestLazyHealthCheckGatesFirstCall:
    @pytest.mark.asyncio
    async def test_unreachable_solver_not_contacted_first_call(self, mgr):
        """Solver pre-marked unreachable → ``no_solver`` envelope; the
        ``solve()`` mock is NOT called. Pre-refactor, the gate read was
        sync but the probe ran inside ``solve()`` — so the FIRST captcha
        of a session always slipped past the unreachable gate. Lazy +
        async ``is_solver_unreachable()`` closes that hole.
        """
        solver = _mk_solver(return_value=_solved(), unreachable=True)
        mgr._captcha_solver = solver
        inst = _mk_inst()

        envelope = await mgr._check_captcha(inst)
        assert envelope["solver_outcome"] == "no_solver"
        assert envelope["solver_attempted"] is False
        assert envelope["next_action"] == "request_captcha_help"
        # Critical: the solver mock was NEVER awaited. Pre-refactor the
        # FIRST call of a session would have invoked the mock before the
        # gate fired.
        solver.solve.assert_not_awaited()


# ── 7. Audit log: cap / rate-limit / behavioral events drain via metrics ──


class TestAuditLogAggregation:
    """The metered-solve gates and the behavioral classifier both record
    events to a per-minute aggregation bucket. ``_emit_metrics`` drains
    the buckets and forwards each as a ``captcha_gate`` event through
    the ``metrics_sink`` (EventBus in production)."""

    @pytest.mark.asyncio
    async def test_cost_cap_event_drains_via_sink(self, tmp_path, monkeypatch):
        # $0.50 cap → 50_000 millicents.
        monkeypatch.setenv("CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", "0.50")
        await cost.add_cost("agent-1", 50_000)

        events: list[dict] = []
        m = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=events.append,
        )
        m._captcha_solver = _mk_solver(return_value=_solved())
        inst = _mk_inst()
        m._instances["agent-1"] = inst

        # Trigger TWO cap-blocked solves so we exercise aggregation.
        await m._check_captcha(inst)
        await m._check_captcha(inst)

        # Drain.
        await m._emit_metrics()

        captcha_events = [e for e in events if e.get("type") == "captcha_gate"]
        assert len(captcha_events) == 1, f"expected one aggregated event, got {captcha_events}"
        ev = captcha_events[0]
        # F4 — payload key MUST be ``agent_id`` (not ``agent``) so the
        # dashboard metrics poller in host/server.py picks it up. The
        # poller's per-payload filter reads ``payload.get("agent_id")``;
        # emitting the legacy ``agent`` key silently dropped events.
        assert ev["agent_id"] == "agent-1"
        assert "agent" not in ev or ev.get("agent") == ev["agent_id"]
        assert ev["outcome"] == "cost_cap"
        # Aggregated count, not per-call.
        assert ev["count"] == 2

    @pytest.mark.asyncio
    async def test_rate_limit_event_drains_via_sink(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CAPTCHA_RATE_LIMIT_PER_HOUR", "1")
        now = time.time()
        svc._solve_rate_window["agent-1"] = deque([now])

        events: list[dict] = []
        m = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=events.append,
        )
        m._captcha_solver = _mk_solver(return_value=_solved())
        inst = _mk_inst()
        m._instances["agent-1"] = inst

        await m._check_captcha(inst)
        await m._emit_metrics()

        rate_events = [e for e in events
                       if e.get("type") == "captcha_gate"
                       and e.get("outcome") == "rate_limited"]
        assert len(rate_events) == 1
        assert rate_events[0]["count"] == 1

    @pytest.mark.asyncio
    async def test_behavioral_skip_event_drains_via_sink(self, tmp_path, monkeypatch):
        events: list[dict] = []
        m = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=events.append,
        )
        m._captcha_solver = _mk_solver(return_value=_solved())

        # Force the behavioral classifier to find a Press & Hold.
        async def fake_behavioral(page):
            return "px-press-hold"
        async def fake_cf_state(page):
            return "none"
        async def fake_recaptcha(page):
            return {"variant": "unknown"}

        monkeypatch.setattr(svc, "_classify_behavioral", fake_behavioral)
        monkeypatch.setattr(svc, "_classify_cf_state", fake_cf_state)
        monkeypatch.setattr(svc, "_classify_recaptcha", fake_recaptcha)

        inst = _mk_inst()
        m._instances["agent-1"] = inst

        envelope = await m._check_captcha(inst)
        assert envelope["solver_outcome"] == "skipped_behavioral"

        await m._emit_metrics()
        beh = [e for e in events
               if e.get("type") == "captcha_gate"
               and e.get("outcome") == "skipped_behavioral"]
        assert len(beh) == 1
        assert beh[0]["count"] == 1
        assert beh[0]["kind"] == "px-press-hold"

    @pytest.mark.asyncio
    async def test_aggregation_keys_distinct_outcomes_separately(
        self, tmp_path, monkeypatch,
    ):
        """Different ``(agent, outcome, kind)`` tuples should NOT merge —
        cap and rate-limit events on the same agent emit two separate
        aggregated payloads.
        """
        # $0.50 cap → 50_000 millicents.
        monkeypatch.setenv("CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", "0.50")
        await cost.add_cost("agent-1", 50_000)

        events: list[dict] = []
        m = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=events.append,
        )
        m._captcha_solver = _mk_solver(return_value=_solved())
        inst = _mk_inst()
        m._instances["agent-1"] = inst

        # First call: cost-cap fires.
        await m._check_captcha(inst)

        # Drain so we know cap-event is in the queue.
        # Then trigger rate-limit on a fresh window.
        monkeypatch.delenv("CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", raising=False)
        await cost.reset()
        monkeypatch.setenv("CAPTCHA_RATE_LIMIT_PER_HOUR", "1")
        now = time.time()
        svc._solve_rate_window["agent-1"] = deque([now])
        await m._check_captcha(inst)

        await m._emit_metrics()

        cap_events = [e for e in events if e.get("outcome") == "cost_cap"]
        rate_events = [e for e in events if e.get("outcome") == "rate_limited"]
        assert len(cap_events) == 1
        assert len(rate_events) == 1


# ── 8. Codex F2 — kill-switch on auto-detect path ─────────────────────────


class TestKillSwitchOnAutoDetect:
    """Codex F2 — pre-fix the ``CAPTCHA_DISABLED`` flag was only checked
    inside ``solve_captcha``. Auto-detect via navigate / click / etc.
    bypassed the gate and still hit the provider. The fix moves the
    check inside ``_metered_solve`` so every path sees the same
    short-circuit, and emits a ``kill_switch_active`` audit event."""

    @pytest.mark.asyncio
    async def test_navigate_path_with_disabled_flag_skips_solver(
        self, mgr, monkeypatch,
    ):
        monkeypatch.setenv("CAPTCHA_DISABLED", "true")
        solver = _mk_solver(return_value=_solved())
        mgr._captcha_solver = solver
        inst = _mk_inst()

        envelope = await mgr._check_captcha(inst)
        assert envelope["captcha_found"] is True
        # Disabled → no_solver envelope (the §11.13 enum doesn't carry a
        # "kill_switch" outcome; reusing no_solver matches the pre-fix
        # solve_captcha early-return shape that callers already handle).
        assert envelope["solver_outcome"] == "no_solver"
        assert envelope["solver_attempted"] is False
        assert envelope["next_action"] == "request_captcha_help"
        # Critical: the solver mock was NEVER awaited.
        solver.solve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_kill_switch_emits_audit_event(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("CAPTCHA_DISABLED", "true")
        events: list[dict] = []
        m = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=events.append,
        )
        m._captcha_solver = _mk_solver(return_value=_solved())
        inst = _mk_inst()
        m._instances["agent-1"] = inst

        await m._check_captcha(inst)
        await m._emit_metrics()

        kill = [e for e in events
                if e.get("type") == "captcha_gate"
                and e.get("outcome") == "kill_switch_active"]
        assert len(kill) == 1
        # F4: payload uses ``agent_id`` (not the legacy ``agent`` key).
        assert kill[0]["agent_id"] == "agent-1"


# ── 9. Codex F5 — rate-limit slot consumed AFTER cost-cap check ───────────


class TestGateOrderingCostBeforeRate:
    """Codex F5 — the original implementation ran rate-limit gate first,
    burning a per-hour slot for an agent that was already over the cost
    cap (the cost gate would then short-circuit the solve). The fix
    swaps the order: cost-cap (read-only) fires first, then rate-limit
    (consumes a slot)."""

    @pytest.mark.asyncio
    async def test_cost_capped_solve_does_not_burn_rate_slot(
        self, mgr, monkeypatch,
    ):
        # $0.50 cap → 50_000 mc. Pre-fill enough to clear it.
        monkeypatch.setenv("CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", "0.50")
        await cost.add_cost("agent-1", 50_000)
        # Tight rate window so we can easily detect a slot getting burned.
        monkeypatch.setenv("CAPTCHA_RATE_LIMIT_PER_HOUR", "5")
        # Start with an empty rate-window for this agent.
        svc._solve_rate_window["agent-1"] = deque()

        solver = _mk_solver(return_value=_solved())
        mgr._captcha_solver = solver

        inst = _mk_inst()
        envelope = await mgr._check_captcha(inst)

        # Cost cap fired (not rate-limited).
        assert envelope["solver_outcome"] == "cost_cap"
        # Rate-limit window UNCHANGED — no slot consumed.
        assert len(svc._solve_rate_window["agent-1"]) == 0


# ── 10. Cost-cap reservation closes concurrent solve race ─────────────────


class TestCostCapReservation:
    @pytest.mark.asyncio
    async def test_concurrent_solves_reserve_cost_before_provider_call(
        self, mgr, monkeypatch,
    ):
        """Two concurrent solves at the cap boundary should not both call
        the provider. The first reserves the published price; the second
        sees the in-flight reservation and returns ``cost_cap``."""
        monkeypatch.setenv("CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", "0.001")

        async def _slow_solve(*args, **kwargs):
            await asyncio.sleep(0.05)
            return _solved()

        solver = _mk_solver(return_value=_solved(), provider="2captcha")
        solver.solve = AsyncMock(side_effect=_slow_solve)
        mgr._captcha_solver = solver
        inst = _mk_inst()

        results = await asyncio.gather(
            mgr._metered_solve(inst, 'iframe[src*="recaptcha"]',
                               "recaptcha-v2-checkbox"),
            mgr._metered_solve(inst, 'iframe[src*="recaptcha"]',
                               "recaptcha-v2-checkbox"),
        )

        assert sum(1 for r in results if r.token == "tok") == 1
        assert sum(1 for r in results if r.skipped == "cost_cap") == 1
        assert solver.solve.await_count == 1
        assert await cost.get_millicents("agent-1") == 100

    @pytest.mark.asyncio
    async def test_reservation_refunded_when_solver_returns_no_token(
        self, mgr, monkeypatch,
    ):
        monkeypatch.setenv("CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", "0.001")
        solver = _mk_solver(
            return_value=SolveResult(
                token=None, injection_succeeded=False,
                used_proxy_aware=False, compat_rejected=False,
            ),
            provider="2captcha",
        )
        mgr._captcha_solver = solver
        inst = _mk_inst()

        result = await mgr._metered_solve(
            inst, 'iframe[src*="recaptcha"]', "recaptcha-v2-checkbox",
        )

        assert result.token is None
        assert result.skipped is None
        assert await cost.get_millicents("agent-1") == 0

    @pytest.mark.asyncio
    async def test_proxy_aware_reservation_refunds_to_actual_cost(
        self, mgr, monkeypatch,
    ):
        monkeypatch.setenv("CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", "1.00")
        solver = _mk_solver(
            return_value=_solved(used_proxy_aware=False),
            provider="2captcha",
        )
        mgr._captcha_solver = solver
        inst = _mk_inst()

        result = await mgr._metered_solve(
            inst, 'iframe[src*="hcaptcha"]', "hcaptcha",
        )

        assert result.token == "tok"
        # 2captcha hcaptcha proxy-aware reservation is 300 mc; actual
        # proxyless result should settle back to 100 mc.
        assert await cost.get_millicents("agent-1") == 100

    @pytest.mark.asyncio
    async def test_unpriced_kind_with_cap_fails_closed(
        self, mgr, monkeypatch,
    ):
        monkeypatch.setenv("CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", "1.00")
        solver = _mk_solver(return_value=_solved(), provider="2captcha")
        mgr._captcha_solver = solver
        inst = _mk_inst()

        result = await mgr._metered_solve(inst, "#captcha", "unknown")

        assert result.skipped == "price_missing"
        solver.solve.assert_not_awaited()
        assert await cost.get_millicents("agent-1") == 0


# ── 12. _max_published_solve_cost_millicents helper ──────────────────────


class TestMaxPublishedSolveCostHelper:
    """``_max_published_solve_cost_millicents`` returns the worst-case
    price across the proxyless / proxy-aware tiers so the cap reservation
    holds even when the solver flips paths mid-solve. End-to-end coverage
    lives in the ``_metered_solve`` tests above; pin the contract here."""

    def test_known_provider_returns_max_of_tiers(self):
        # 2captcha hcaptcha: proxyless=100, proxy_aware=300 → max=300
        from src.browser.service import _max_published_solve_cost_millicents
        assert _max_published_solve_cost_millicents("2captcha", "hcaptcha") == 300

    def test_falls_back_when_only_proxyless_tier_published(self):
        # 2captcha recaptcha-v3 has no proxy-aware row; proxyless=100.
        from src.browser.service import _max_published_solve_cost_millicents
        assert _max_published_solve_cost_millicents("2captcha", "recaptcha-v3") == 100

    def test_unknown_kind_returns_none(self):
        from src.browser.service import _max_published_solve_cost_millicents
        assert _max_published_solve_cost_millicents("2captcha", "made-up") is None

    def test_unknown_provider_returns_none(self):
        from src.browser.service import _max_published_solve_cost_millicents
        assert _max_published_solve_cost_millicents("nopal", "hcaptcha") is None

    def test_empty_provider_returns_none(self):
        from src.browser.service import _max_published_solve_cost_millicents
        assert _max_published_solve_cost_millicents("", "hcaptcha") is None


# ── 11. Codex F7 — token-without-provider warns + (cap-on) fails closed ──


class TestProviderMissingFailsClosedWhenCapOn:
    @pytest.mark.asyncio
    async def test_no_provider_with_cap_blocks_solve(
        self, mgr, monkeypatch, caplog,
    ):
        import logging as _logging
        monkeypatch.setenv("CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", "1.00")
        # Solver mock with NO provider attribute (or empty string).
        solver = _mk_solver(return_value=_solved(), provider="")
        mgr._captcha_solver = solver
        inst = _mk_inst()

        with caplog.at_level(_logging.WARNING, logger="browser.service"):
            envelope = await mgr._check_captcha(inst)

        # Solve was blocked — no provider HTTP call.
        solver.solve.assert_not_awaited()
        assert envelope["solver_outcome"] == "no_solver"
        # A warning was logged about the failing-closed decision.
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "failing closed" in joined.lower() or "fail" in joined.lower()

    @pytest.mark.asyncio
    async def test_no_provider_without_cap_warns_and_proceeds(
        self, mgr, monkeypatch, caplog,
    ):
        """No cap configured → we still warn but don't block. Custom
        solver integrations / tests with ``provider=""`` keep working
        (the cost increment is silently skipped — same as today)."""
        import logging as _logging
        monkeypatch.delenv(
            "CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", raising=False,
        )
        solver = _mk_solver(return_value=_solved(), provider="")
        mgr._captcha_solver = solver
        inst = _mk_inst()

        with caplog.at_level(_logging.WARNING, logger="browser.service"):
            envelope = await mgr._check_captcha(inst)

        assert envelope["solver_outcome"] == "solved"
        # Cost was NOT incremented (provider name missing).
        assert await cost.get_millicents("agent-1") == 0
        # Warning was logged so operators see the misconfig.
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "provider" in joined.lower()
