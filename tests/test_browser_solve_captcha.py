"""Tests for ``BrowserManager.solve_captcha`` (Phase 8 §11.14).

Covers:
  * No-captcha early return.
  * Successful solve increments the cost counter.
  * Cost-cap exceeded → ``cost_cap`` outcome.
  * Rate-limit exceeded → ``rate_limited`` outcome.
  * Recursive-solve guard → ``captcha_during_solve``.
  * ``retry_previous=True`` waits then re-checks once when the initial
    detection found nothing (covers the "page just navigated, captcha
    still rendering" race). A second-detection hit still solves.
  * ``retry_previous=True`` does NOT block when the first detection
    already found a captcha (no extra wait).
  * Bad ``hint`` returns invalid_input.
  * Behavioral-only ``hint`` values are rejected with a routing message.
  * Health-unreachable + breaker-open paths still return §11.13 envelopes.
"""

from __future__ import annotations

import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser import captcha_cost_counter as cost
from src.browser import service as svc
from src.browser.captcha import SolveResult
from src.browser.service import BrowserManager, CamoufoxInstance


def _solved_result() -> SolveResult:
    """Standard "solved + injected" SolveResult for solver mocks."""
    return SolveResult(
        token="tok",
        injection_succeeded=True,
        used_proxy_aware=False,
        compat_rejected=False,
    )


def _mk_inst_with_locator(*, captcha_present: bool, page_url: str = "https://example.com"):
    """Build a CamoufoxInstance with a mocked Page that reports captcha presence."""
    mock_page = MagicMock()
    mock_page.url = page_url

    locator = MagicMock()
    count_value = 1 if captcha_present else 0
    locator.count = AsyncMock(return_value=count_value)
    mock_page.locator = MagicMock(return_value=locator)
    return CamoufoxInstance("agent-1", MagicMock(), MagicMock(), mock_page)


@pytest.fixture(autouse=True)
async def _isolate_cost(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CAPTCHA_COST_COUNTER_PATH", str(tmp_path / "captcha_costs.json"),
    )
    await cost.reset()
    # Reset the in-process rate-limit window between tests.
    svc._solve_rate_window.clear()
    yield
    await cost.reset()
    svc._solve_rate_window.clear()


@pytest.fixture()
def mgr(tmp_path):
    m = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
    return m


class TestSolveCaptchaNoCaptcha:
    @pytest.mark.asyncio
    async def test_returns_no_captcha(self, mgr):
        inst = _mk_inst_with_locator(captcha_present=False)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        result = await mgr.solve_captcha("agent-1")
        assert result["success"] is True
        assert result["data"]["captcha_found"] is False
        assert "No captcha" in result["data"]["message"]


class TestSolveCaptchaSuccess:
    @pytest.mark.asyncio
    async def test_solver_success_increments_cost(self, mgr):
        inst = _mk_inst_with_locator(captcha_present=True)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        # Configure a fake solver that succeeds, with provider=2captcha.
        solver = MagicMock()
        solver.provider = "2captcha"
        solver.is_solver_unreachable = AsyncMock(return_value=False)
        solver.is_breaker_open = MagicMock(return_value=False)
        solver.solve = AsyncMock(return_value=_solved_result())
        mgr._captcha_solver = solver

        # The captcha selector that matches (iframe[src*=hcaptcha]) will
        # classify as kind="hcaptcha", which is priced at 100¢ for 2captcha.
        # We need to inject a locator that returns >0 only for the
        # hcaptcha selector. Default mock returns >0 for any selector;
        # the FIRST iter of _check_captcha matches "iframe[src*=recaptcha]"
        # → kind="recaptcha-v2-checkbox", priced at 100¢ for 2captcha.
        result = await mgr.solve_captcha("agent-1")

        assert result["success"] is True
        assert result["data"]["captcha_found"] is True
        assert result["data"]["solver_outcome"] == "solved"

        # Cost incremented (recaptcha-v2-checkbox @ 2captcha = 100¢)
        assert await cost.get_cents("agent-1") == 100


class TestSolveCaptchaCostCap:
    @pytest.mark.asyncio
    async def test_cost_cap_exceeded_short_circuits(self, mgr, monkeypatch):
        # Configure the cap (USD).
        monkeypatch.setenv("CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", "0.50")
        # Pre-fill spend to exceed the cap (50 cents).
        await cost.add_cost("agent-1", 100)

        inst = _mk_inst_with_locator(captcha_present=True)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        solver = MagicMock()
        solver.provider = "2captcha"
        solver.is_solver_unreachable = AsyncMock(return_value=False)
        solver.is_breaker_open = MagicMock(return_value=False)
        solver.solve = AsyncMock(return_value=_solved_result())
        mgr._captcha_solver = solver

        result = await mgr.solve_captcha("agent-1")
        assert result["success"] is True
        assert result["data"]["solver_outcome"] == "cost_cap"
        assert result["data"]["next_action"] == "request_captcha_help"


class TestSolveCaptchaRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limited_after_threshold(self, mgr, monkeypatch):
        monkeypatch.setenv("CAPTCHA_RATE_LIMIT_PER_HOUR", "2")
        # Pre-fill the rate window for this agent.
        now = time.time()
        svc._solve_rate_window["agent-1"] = deque([now, now])

        inst = _mk_inst_with_locator(captcha_present=True)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        solver = MagicMock()
        solver.provider = "2captcha"
        solver.is_solver_unreachable = AsyncMock(return_value=False)
        solver.is_breaker_open = MagicMock(return_value=False)
        solver.solve = AsyncMock(return_value=_solved_result())
        mgr._captcha_solver = solver

        result = await mgr.solve_captcha("agent-1")
        assert result["success"] is True
        assert result["data"]["solver_outcome"] == "rate_limited"
        assert result["data"]["next_action"] == "request_captcha_help"


class TestSolveCaptchaRecursiveGuard:
    @pytest.mark.asyncio
    async def test_recursive_solve_returns_captcha_during_solve(self, mgr):
        inst = _mk_inst_with_locator(captcha_present=True)
        # Pretend a solve is already in flight.
        inst._captcha_solving = True
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        result = await mgr.solve_captcha("agent-1")
        assert result["success"] is True
        assert result["data"]["solver_outcome"] == "captcha_during_solve"
        assert result["data"]["next_action"] == "request_captcha_help"


class TestSolveCaptchaRetryPrevious:
    """``retry_previous=True`` is a "be patient" hint — when the initial
    detection finds no captcha, wait briefly and re-check ONCE. Useful
    when the page just navigated and the widget is still rendering.

    Does NOT replay any prior solve attempt — there is no per-instance
    state about previous solves. The solver runs freshly against
    whatever the second detection finds.
    """

    @pytest.mark.asyncio
    async def test_retry_does_not_wait_when_captcha_already_present(self, mgr):
        """Captcha visible on first detection → no patience window, normal solve."""
        inst = _mk_inst_with_locator(captcha_present=True)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        solver = MagicMock()
        solver.provider = "2captcha"
        solver.is_solver_unreachable = AsyncMock(return_value=False)
        solver.is_breaker_open = MagicMock(return_value=False)
        solver.solve = AsyncMock(return_value=_solved_result())
        mgr._captcha_solver = solver

        # Patch the patience-window sleep so we can assert it WASN'T called.
        with pytest.MonkeyPatch.context() as mp:
            sleep_mock = AsyncMock()
            mp.setattr("src.browser.service.asyncio.sleep", sleep_mock)
            result = await mgr.solve_captcha("agent-1", retry_previous=True)

        assert result["success"] is True
        assert result["data"]["solver_outcome"] == "solved"
        sleep_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retry_waits_then_finds_captcha_on_second_detection(self, mgr):
        """First detection: nothing. Wait ~500ms, re-detect: captcha present → solve."""
        # Build an instance whose first locator().count() returns 0,
        # second returns 1. We need a locator that toggles based on call
        # count across selectors.
        mock_page = MagicMock()
        mock_page.url = "https://example.com"

        # The detection loop iterates over multiple selectors. We want
        # the FIRST full pass (all selectors) to return 0, and the
        # SECOND full pass (after the patience window) to return 1 on
        # the first selector.
        pass_counter = {"n": 0}
        selector_counter = {"n": 0}

        def locator_factory(sel):
            loc = MagicMock()

            async def _count():
                # Track per-call "which pass are we in?" — every 7
                # calls (we have 7 selectors) we move to a new pass.
                idx = selector_counter["n"]
                selector_counter["n"] += 1
                pass_idx = idx // 7
                if pass_idx == 0:
                    return 0  # first detection: empty
                return 1 if (idx % 7) == 0 else 0

            loc.count = _count
            return loc

        mock_page.locator = MagicMock(side_effect=locator_factory)
        inst = CamoufoxInstance(
            "agent-1", MagicMock(), MagicMock(), mock_page,
        )
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        solver = MagicMock()
        solver.provider = "2captcha"
        solver.is_solver_unreachable = AsyncMock(return_value=False)
        solver.is_breaker_open = MagicMock(return_value=False)
        solver.solve = AsyncMock(return_value=_solved_result())
        mgr._captcha_solver = solver

        with pytest.MonkeyPatch.context() as mp:
            sleep_mock = AsyncMock()
            mp.setattr("src.browser.service.asyncio.sleep", sleep_mock)
            result = await mgr.solve_captcha("agent-1", retry_previous=True)
            del pass_counter  # silence unused-var lint
            # Patience window MUST have been awaited exactly once with
            # the documented bound (500ms → 0.5s).
            sleep_calls = [c.args[0] for c in sleep_mock.await_args_list]
            assert 0.5 in sleep_calls, (
                f"expected 500ms patience sleep; got {sleep_calls}"
            )

        assert result["success"] is True
        assert result["data"]["solver_outcome"] == "solved"

    @pytest.mark.asyncio
    async def test_retry_returns_no_captcha_when_second_detection_also_empty(self, mgr):
        """Both detections empty → no captcha message, no solver call."""
        inst = _mk_inst_with_locator(captcha_present=False)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        with pytest.MonkeyPatch.context() as mp:
            sleep_mock = AsyncMock()
            mp.setattr("src.browser.service.asyncio.sleep", sleep_mock)
            result = await mgr.solve_captcha("agent-1", retry_previous=True)

        assert result["success"] is True
        assert result["data"]["captcha_found"] is False
        assert "No captcha" in result["data"]["message"]
        # Patience window fired exactly once (500ms).
        sleep_args = [c.args[0] for c in sleep_mock.await_args_list]
        assert sleep_args.count(0.5) == 1


class TestSolveCaptchaHintValidation:
    @pytest.mark.asyncio
    async def test_invalid_hint_rejected(self, mgr):
        inst = _mk_inst_with_locator(captcha_present=True)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        result = await mgr.solve_captcha("agent-1", hint="not-a-real-kind")
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert "hint" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_valid_hint_overrides_classification(self, mgr):
        inst = _mk_inst_with_locator(captcha_present=True)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        # No solver — we just want to read the kind back.
        mgr._captcha_solver = None
        result = await mgr.solve_captcha("agent-1", hint="hcaptcha")
        assert result["success"] is True
        assert result["data"]["kind"] == "hcaptcha"

    @pytest.mark.parametrize(
        "behavioral_hint",
        [
            "px-press-hold",
            "datadome-behavioral",
            "cf-interstitial-auto",
            "cf-interstitial-behavioral",
        ],
    )
    @pytest.mark.asyncio
    async def test_behavioral_only_hint_rejected_with_routing_message(
        self, mgr, behavioral_hint,
    ):
        """Behavioral kinds have no solver task entry; passing them as a
        ``hint`` would produce a silent no-op. Validator must reject loudly
        and point at ``request_captcha_help``.
        """
        inst = _mk_inst_with_locator(captcha_present=True)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        result = await mgr.solve_captcha("agent-1", hint=behavioral_hint)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert behavioral_hint in result["error"]["message"]
        assert "request_captcha_help" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_legacy_recaptcha_enterprise_alias_rejected(self, mgr):
        """The legacy coarse ``recaptcha-enterprise`` alias was kept as
        a back-compat hint but had no operational effect (no task-table
        entry, no pricing entry). Removed in favor of the precise
        v2/v3 variants — agents using the alias get a clean error
        pointing at the valid set.
        """
        inst = _mk_inst_with_locator(captcha_present=True)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        result = await mgr.solve_captcha("agent-1", hint="recaptcha-enterprise")
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        # Suggest the precise variants.
        assert "recaptcha-enterprise-v2" in result["error"]["message"]


class TestSolveCaptchaHealthAndBreaker:
    @pytest.mark.asyncio
    async def test_solver_unreachable_returns_no_solver(self, mgr):
        inst = _mk_inst_with_locator(captcha_present=True)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        solver = MagicMock()
        solver.provider = "2captcha"
        solver.is_solver_unreachable = AsyncMock(return_value=True)
        solver.is_breaker_open = MagicMock(return_value=False)
        solver.solve = AsyncMock(return_value=_solved_result())
        mgr._captcha_solver = solver

        result = await mgr.solve_captcha("agent-1")
        assert result["success"] is True
        assert result["data"]["solver_outcome"] == "no_solver"
        assert result["data"]["next_action"] == "request_captcha_help"

    @pytest.mark.asyncio
    async def test_breaker_open_returns_timeout_with_flag(self, mgr):
        inst = _mk_inst_with_locator(captcha_present=True)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        solver = MagicMock()
        solver.provider = "2captcha"
        solver.is_solver_unreachable = AsyncMock(return_value=False)
        solver.is_breaker_open = MagicMock(return_value=True)
        solver.solve = AsyncMock(return_value=_solved_result())
        mgr._captcha_solver = solver

        result = await mgr.solve_captcha("agent-1")
        assert result["success"] is True
        assert result["data"]["solver_outcome"] == "timeout"
        # _check_captcha sets the additive ``breaker_open`` flag on the envelope
        # for the breaker-open path. Because solve_captcha runs the legacy-
        # field shim, the flag is preserved on the returned dict.
        assert result["data"].get("breaker_open") is True


class TestSolveCaptchaDisabledFlag:
    @pytest.mark.asyncio
    async def test_captcha_disabled_short_circuits(self, mgr, monkeypatch):
        monkeypatch.setenv("CAPTCHA_DISABLED", "true")
        inst = _mk_inst_with_locator(captcha_present=True)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        result = await mgr.solve_captcha("agent-1")
        assert result["success"] is True
        assert result["data"]["solver_outcome"] == "no_solver"
