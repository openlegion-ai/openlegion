"""Tests for ``BrowserManager.solve_captcha`` (Phase 8 §11.14).

Covers:
  * No-captcha early return.
  * Successful solve increments the cost counter.
  * Cost-cap exceeded → ``cost_cap`` outcome.
  * Rate-limit exceeded → ``rate_limited`` outcome.
  * Recursive-solve guard → ``captcha_during_solve``.
  * ``retry_previous=True`` within 60s replays last attempt.
  * ``retry_previous=True`` after 60s returns invalid_input.
  * Bad ``hint`` returns invalid_input.
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
    @pytest.mark.asyncio
    async def test_retry_within_ttl_works(self, mgr):
        inst = _mk_inst_with_locator(captcha_present=True)
        # Stamp a prior attempt 1 second ago.
        inst._last_solve_attempt = ("sk", "https://example.com", time.time() - 1)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        solver = MagicMock()
        solver.provider = "2captcha"
        solver.is_solver_unreachable = AsyncMock(return_value=False)
        solver.is_breaker_open = MagicMock(return_value=False)
        solver.solve = AsyncMock(return_value=_solved_result())
        mgr._captcha_solver = solver

        result = await mgr.solve_captcha("agent-1", retry_previous=True)
        assert result["success"] is True
        assert result["data"]["solver_outcome"] == "solved"

    @pytest.mark.asyncio
    async def test_retry_after_ttl_returns_invalid_input(self, mgr):
        inst = _mk_inst_with_locator(captcha_present=True)
        # Stamp a prior attempt 90 seconds ago — past the 60s TTL.
        inst._last_solve_attempt = ("sk", "https://example.com", time.time() - 90)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        result = await mgr.solve_captcha("agent-1", retry_previous=True)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert result["error"]["message"] == "no_recent_attempt_to_retry"

    @pytest.mark.asyncio
    async def test_retry_with_no_prior_attempt_returns_invalid_input(self, mgr):
        inst = _mk_inst_with_locator(captcha_present=True)
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        result = await mgr.solve_captcha("agent-1", retry_previous=True)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"


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
