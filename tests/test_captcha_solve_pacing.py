"""Tests for §11.11: solve-pacing Gaussian delay.

Adds a clamped Gaussian pause between solver token retrieval and DOM
injection. Real users take 5-15s between captcha-appears and form-submit;
instant token injection is a low-but-real anti-bot signal. μ=6000ms,
σ=2500ms, clamped to [3000, 12000], all overridable via env vars.

Covers:

* Helper produces values within the [3000, 12000] clamp across many
  trials (statistical sanity — the delay never escapes the clamp).
* Env-var overrides for μ, σ, and the min/max clamp work.
* The helper is awaited exactly once per SUCCESSFUL solve (token
  retrieved + injection succeeds).
* The helper is NOT awaited on failed solves (no token, timeout,
  exception, unreachable solver, breaker open) — there's nothing to
  inject so pacing would just waste time.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.browser import timing
from src.browser.captcha import CaptchaSolver

# ── helpers ───────────────────────────────────────────────────────────


def _make_solver(provider: str = "2captcha", key: str = "k") -> CaptchaSolver:
    return CaptchaSolver(provider, key)


def _solve_page() -> MagicMock:
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value="site-key-abc")
    page.url = "https://example.com"
    return page


# ── 1. Helper clamp behavior ──────────────────────────────────────────


class TestHelperClamp:
    @pytest.mark.asyncio
    async def test_default_clamp_3000_to_12000(self):
        """Sample many delays; all must fall within [3000ms, 12000ms]."""
        observed: list[float] = []

        async def fake_sleep(s):
            observed.append(s)

        with patch("asyncio.sleep", new=fake_sleep):
            for _ in range(500):
                await timing.captcha_solve_delay()

        assert observed, "captcha_solve_delay should call asyncio.sleep"
        # Convert ms-clamped delay back to seconds: [3.0, 12.0].
        for s in observed:
            assert 3.0 <= s <= 12.0, f"out-of-clamp delay: {s}s"

    @pytest.mark.asyncio
    async def test_clamp_respects_min_override(self, monkeypatch):
        # Push the floor up to 8s. With μ=6000 + σ=2500, plenty of
        # samples want to land below 8s; the clamp must hold.
        monkeypatch.setenv("CAPTCHA_PACING_MS_MIN", "8000")

        observed: list[float] = []

        async def fake_sleep(s):
            observed.append(s)

        with patch("asyncio.sleep", new=fake_sleep):
            for _ in range(200):
                await timing.captcha_solve_delay()

        for s in observed:
            assert 8.0 <= s <= 12.0

    @pytest.mark.asyncio
    async def test_clamp_respects_max_override(self, monkeypatch):
        # Pin the ceiling at 4s. With μ=6000ms (default), most samples
        # want to land above 4s; the clamp must hold.
        monkeypatch.setenv("CAPTCHA_PACING_MS_MAX", "4000")
        # Lower the floor too so the clamp range is meaningful.
        monkeypatch.setenv("CAPTCHA_PACING_MS_MIN", "1000")

        observed: list[float] = []

        async def fake_sleep(s):
            observed.append(s)

        with patch("asyncio.sleep", new=fake_sleep):
            for _ in range(200):
                await timing.captcha_solve_delay()

        for s in observed:
            assert 1.0 <= s <= 4.0


# ── 2. μ / σ env-var overrides ────────────────────────────────────────


class TestMuSigmaOverrides:
    @pytest.mark.asyncio
    async def test_mu_override_shifts_distribution(self, monkeypatch):
        # μ=10s, σ=10ms → essentially deterministic output near 10s.
        monkeypatch.setenv("CAPTCHA_SOLVE_PACING_MU_MS", "10000")
        monkeypatch.setenv("CAPTCHA_SOLVE_PACING_SIGMA_MS", "10")
        # Open the clamp so μ doesn't get truncated.
        monkeypatch.setenv("CAPTCHA_PACING_MS_MIN", "1000")
        monkeypatch.setenv("CAPTCHA_PACING_MS_MAX", "20000")

        observed: list[float] = []

        async def fake_sleep(s):
            observed.append(s)

        with patch("asyncio.sleep", new=fake_sleep):
            for _ in range(50):
                await timing.captcha_solve_delay()

        avg = sum(observed) / len(observed)
        # Within 0.5s of the 10s mean (σ=10ms makes it tight).
        assert 9.5 <= avg <= 10.5, f"avg drifted: {avg}s"

    @pytest.mark.asyncio
    async def test_sigma_override_widens_distribution(self, monkeypatch):
        # μ=6000ms, σ=10ms → very tight cluster.
        monkeypatch.setenv("CAPTCHA_SOLVE_PACING_MU_MS", "6000")
        monkeypatch.setenv("CAPTCHA_SOLVE_PACING_SIGMA_MS", "10")

        observed: list[float] = []

        async def fake_sleep(s):
            observed.append(s)

        with patch("asyncio.sleep", new=fake_sleep):
            for _ in range(100):
                await timing.captcha_solve_delay()

        # All samples should land within ~0.1s of 6s.
        for s in observed:
            assert 5.9 <= s <= 6.1, f"sigma=10 produced wide sample: {s}s"


# ── 3. Helper called exactly once per successful solve ────────────────


def _stub_create_then_ready(token: str = "tok-X") -> list[MagicMock]:
    create = MagicMock()
    create.json = MagicMock(return_value={"errorId": 0, "taskId": "T-1"})
    create.raise_for_status = MagicMock()
    poll = MagicMock()
    poll.json = MagicMock(return_value={
        "errorId": 0, "status": "ready",
        "solution": {"gRecaptchaResponse": token},
    })
    poll.raise_for_status = MagicMock()
    return [create, poll]


def _ok_balance() -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.json = MagicMock(return_value={"errorId": 0, "balance": 12.34})
    return r


class TestPacingFiresOnSuccess:
    @pytest.mark.asyncio
    async def test_pacing_awaited_once_on_successful_solve(self):
        solver = _make_solver()
        solver._solver_health_checked = True  # skip §11.16 probe
        page = _solve_page()

        client = AsyncMock(spec=httpx.AsyncClient)
        client.is_closed = False
        client.post = AsyncMock(side_effect=_stub_create_then_ready())
        solver._client = client

        pace = AsyncMock(return_value=None)
        with patch("src.browser.timing.captcha_solve_delay", new=pace), \
             patch("src.browser.captcha._POLL_INTERVAL", 0.001):
            ok = await solver.solve(
                page, 'iframe[src*="recaptcha"]', "https://example.com",
                kind="recaptcha-v3",
            )

        assert bool(ok) is True
        assert pace.await_count == 1


# ── 4. Helper NOT called on failure paths ─────────────────────────────


class TestPacingSkippedOnFailure:
    @pytest.mark.asyncio
    async def test_pacing_skipped_when_no_token(self):
        """Provider responds with errorId — no token retrieved → no pacing."""
        solver = _make_solver()
        solver._solver_health_checked = True
        page = _solve_page()

        client = AsyncMock(spec=httpx.AsyncClient)
        client.is_closed = False

        create_err = MagicMock()
        create_err.json = MagicMock(return_value={
            "errorId": 1, "errorDescription": "ERROR_NO_SLOT_AVAILABLE",
        })
        create_err.raise_for_status = MagicMock()
        client.post = AsyncMock(return_value=create_err)
        solver._client = client

        pace = AsyncMock(return_value=None)
        with patch("src.browser.timing.captcha_solve_delay", new=pace):
            ok = await solver.solve(
                page, 'iframe[src*="recaptcha"]', "https://example.com",
                kind="recaptcha-v3",
            )

        assert bool(ok) is False
        assert pace.await_count == 0

    @pytest.mark.asyncio
    async def test_pacing_skipped_on_solver_unreachable(self):
        solver = _make_solver()
        solver._solver_unreachable = True
        solver._solver_health_checked = True
        page = _solve_page()

        pace = AsyncMock(return_value=None)
        with patch("src.browser.timing.captcha_solve_delay", new=pace):
            ok = await solver.solve(
                page, 'iframe[src*="recaptcha"]', "https://example.com",
                kind="recaptcha-v3",
            )

        assert bool(ok) is False
        assert pace.await_count == 0

    @pytest.mark.asyncio
    async def test_pacing_skipped_on_breaker_open(self):
        solver = _make_solver()
        solver._solver_health_checked = True
        # Trip the breaker far into the future.
        solver._solver_breaker_until = 1e18
        page = _solve_page()

        pace = AsyncMock(return_value=None)
        with patch("src.browser.timing.captcha_solve_delay", new=pace):
            ok = await solver.solve(
                page, 'iframe[src*="recaptcha"]', "https://example.com",
                kind="recaptcha-v3",
            )

        assert bool(ok) is False
        assert pace.await_count == 0

    @pytest.mark.asyncio
    async def test_pacing_skipped_on_outer_timeout(self):
        solver = _make_solver()
        solver._solver_health_checked = True
        solver._solve_timeouts_ms["recaptcha-v3"] = 50  # 50ms
        page = _solve_page()

        async def slow_submit(self, *a, **kw):
            await asyncio.sleep(2.0)
            return ("tok", False, False)

        pace = AsyncMock(return_value=None)
        with patch.object(CaptchaSolver, "_submit_and_poll", new=slow_submit), \
             patch("src.browser.timing.captcha_solve_delay", new=pace):
            ok = await solver.solve(
                page, 'iframe[src*="recaptcha"]', "https://example.com",
                kind="recaptcha-v3",
            )

        assert bool(ok) is False
        # Outer wait_for fired BEFORE the token came back — pacing skipped.
        assert pace.await_count == 0

    @pytest.mark.asyncio
    async def test_pacing_skipped_on_submit_exception(self):
        solver = _make_solver()
        solver._solver_health_checked = True
        page = _solve_page()

        async def boom(self, *a, **kw):
            raise RuntimeError("provider blew up")

        pace = AsyncMock(return_value=None)
        with patch.object(CaptchaSolver, "_submit_and_poll", new=boom), \
             patch("src.browser.timing.captcha_solve_delay", new=pace):
            ok = await solver.solve(
                page, 'iframe[src*="recaptcha"]', "https://example.com",
                kind="recaptcha-v3",
            )

        assert bool(ok) is False
        assert pace.await_count == 0

    @pytest.mark.asyncio
    async def test_pacing_skipped_when_sitekey_missing(self):
        solver = _make_solver()
        solver._solver_health_checked = True
        page = AsyncMock()
        page.evaluate = AsyncMock(return_value=None)  # no sitekey
        page.url = "https://example.com"

        pace = AsyncMock(return_value=None)
        with patch("src.browser.timing.captcha_solve_delay", new=pace):
            ok = await solver.solve(
                page, 'iframe[src*="recaptcha"]', "https://example.com",
                kind="recaptcha-v3",
            )

        assert bool(ok) is False
        assert pace.await_count == 0
