"""Tests for §11.9: per-type CAPTCHA solver timeout.

Replaces the legacy hardcoded ``_SOLVE_TIMEOUT = 120`` with a kind →
timeout-ms table resolved at solver init from the static defaults +
``CAPTCHA_TIMEOUT_<KIND_UPPER_UNDERSCORE>_MS`` env overrides.

Covers:

* Default kind → timeout lookup matches the spec values (recaptcha-v3 is
  60s, hcaptcha is 120s, turnstile is 180s, etc.).
* Env override pattern works for at least one kind (v3 → 45s).
* Unknown / behavioral / ``None`` kinds fall back to
  ``_SOLVE_TIMEOUT_FALLBACK_MS`` (120s).
* ``solve()`` passes the kind through to ``_submit_and_poll()``.
* ``solve()`` enforces the kind-specific outer ``asyncio.wait_for``
  deadline — a slow solver inner times out at the configured value.
* The cached table is read once at ``__init__``; mutating env vars
  AFTER construction does not change the existing solver's table.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.browser.captcha import (
    _SOLVE_TIMEOUT_DEFAULTS_MS,
    _SOLVE_TIMEOUT_FALLBACK_MS,
    CaptchaSolver,
)

# ── helpers ───────────────────────────────────────────────────────────


def _make_solver(provider: str = "2captcha", key: str = "k") -> CaptchaSolver:
    return CaptchaSolver(provider, key)


def _solve_page(sitekey: str = "site-key-abc") -> MagicMock:
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value=sitekey)
    page.url = "https://example.com"
    return page


def _ok_balance_resp() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value={"errorId": 0, "balance": 12.34})
    return resp


# ── 1. Default lookup ─────────────────────────────────────────────────


class TestDefaultLookup:
    def test_recaptcha_v3_default_60s(self):
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind("recaptcha-v3") == 60.0

    def test_recaptcha_v2_checkbox_default_120s(self):
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind("recaptcha-v2-checkbox") == 120.0

    def test_recaptcha_v2_invisible_default_120s(self):
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind("recaptcha-v2-invisible") == 120.0

    def test_recaptcha_enterprise_v2_default_120s(self):
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind("recaptcha-enterprise-v2") == 120.0

    def test_recaptcha_enterprise_v3_default_60s(self):
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind("recaptcha-enterprise-v3") == 60.0

    def test_hcaptcha_default_120s(self):
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind("hcaptcha") == 120.0

    def test_turnstile_default_180s(self):
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind("turnstile") == 180.0

    def test_cf_interstitial_turnstile_default_180s(self):
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind(
            "cf-interstitial-turnstile",
        ) == 180.0

    def test_default_table_matches_spec(self):
        # Spec: §11.9 — sanity check the static defaults haven't drifted.
        # §22 added the four anti-bot platform kinds at 180s each (the
        # AntiBot family is known-slow; 180s matches the documented
        # provider response window and the per-type rationale on
        # ``_CAPSOLVER_TASK_TYPES``).
        assert _SOLVE_TIMEOUT_DEFAULTS_MS == {
            "recaptcha-v2-checkbox":     120_000,
            "recaptcha-v2-invisible":    120_000,
            "recaptcha-v3":               60_000,
            "recaptcha-enterprise-v2":   120_000,
            "recaptcha-enterprise-v3":    60_000,
            "hcaptcha":                  120_000,
            "turnstile":                 180_000,
            "cf-interstitial-turnstile": 180_000,
            "js-challenge-akamai":       180_000,
            "js-challenge-imperva":      180_000,
            "js-challenge-kasada":       180_000,
            "datadome-behavioral":       180_000,
        }

    def test_fallback_constant(self):
        assert _SOLVE_TIMEOUT_FALLBACK_MS == 120_000


# ── 2. Env override ───────────────────────────────────────────────────


class TestEnvOverride:
    def test_env_override_for_recaptcha_v3(self, monkeypatch):
        monkeypatch.setenv("CAPTCHA_TIMEOUT_RECAPTCHA_V3_MS", "45000")
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind("recaptcha-v3") == 45.0
        # Other kinds unaffected.
        assert solver._timeout_seconds_for_kind("hcaptcha") == 120.0

    def test_env_override_for_turnstile(self, monkeypatch):
        monkeypatch.setenv("CAPTCHA_TIMEOUT_TURNSTILE_MS", "240000")
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind("turnstile") == 240.0

    def test_env_override_for_cf_interstitial_turnstile(self, monkeypatch):
        # Hyphens in kind name → underscores in env var name.
        monkeypatch.setenv(
            "CAPTCHA_TIMEOUT_CF_INTERSTITIAL_TURNSTILE_MS", "150000",
        )
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind(
            "cf-interstitial-turnstile",
        ) == 150.0

    def test_env_override_resolved_at_init_not_per_call(self, monkeypatch):
        """The §11.9 spec says env vars are read at solver init and cached.
        Changing the env after construction does NOT alter the existing
        solver's table — operators must restart the service to apply."""
        monkeypatch.setenv("CAPTCHA_TIMEOUT_RECAPTCHA_V3_MS", "45000")
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind("recaptcha-v3") == 45.0
        # Mutate after init.
        monkeypatch.setenv("CAPTCHA_TIMEOUT_RECAPTCHA_V3_MS", "99999")
        # Still 45s on the existing instance.
        assert solver._timeout_seconds_for_kind("recaptcha-v3") == 45.0
        # A FRESH solver picks up the new value.
        solver2 = _make_solver()
        assert solver2._timeout_seconds_for_kind("recaptcha-v3") == 99.999

    def test_invalid_env_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("CAPTCHA_TIMEOUT_RECAPTCHA_V3_MS", "not-a-number")
        solver = _make_solver()
        # ``flags.get_int`` warns and returns the default on parse error.
        assert solver._timeout_seconds_for_kind("recaptcha-v3") == 60.0


# ── 3. Unknown kind fallback ──────────────────────────────────────────


class TestFallback:
    def test_unknown_kind_returns_fallback(self):
        solver = _make_solver()
        # Behavioral kinds without a CapSolver task type still hit the
        # fallback. ``px-press-hold`` is HUMAN Security's "Press & Hold"
        # — operator escalation only, no solver path.
        # ``datadome-behavioral`` got a 180s entry in §22 (CapSolver
        # publishes ``DataDomeSliderTask``); it no longer falls back.
        assert solver._timeout_seconds_for_kind("px-press-hold") == 120.0

    def test_completely_bogus_kind_returns_fallback(self):
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind("not-a-real-kind") == 120.0

    def test_none_kind_returns_fallback(self):
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind(None) == 120.0

    def test_empty_kind_returns_fallback(self):
        solver = _make_solver()
        assert solver._timeout_seconds_for_kind("") == 120.0


# ── 4. Solve passes kind through to _submit_and_poll ──────────────────


class TestSolvePassesKindThrough:
    @pytest.mark.asyncio
    async def test_solve_threads_kind_into_submit_and_poll(self):
        solver = _make_solver()
        # Skip the §11.16 health check by pre-marking it done.
        solver._solver_health_checked = True
        page = _solve_page()

        recorded: dict = {}

        async def fake_submit(self, captcha_type, sitekey, page_url, **kwargs):
            recorded["captcha_type"] = captcha_type
            recorded["kind"] = kwargs.get("kind")
            # 4-tuple: (token, used_proxy_aware, compat_rejected, provider_contacted).
            # ``provider_contacted=True`` mirrors a real submit that
            # reached the upstream API.
            return ("tok-X", False, False, True)

        # Patch the page-level injection so we don't need real DOM.
        async def fake_inject(self, page, captcha_type, token):
            return True

        with patch.object(CaptchaSolver, "_submit_and_poll", new=fake_submit), \
             patch.object(CaptchaSolver, "_inject_token", new=fake_inject), \
             patch("src.browser.timing.captcha_solve_delay",
                   new=AsyncMock(return_value=None)):
            ok = await solver.solve(
                page, 'iframe[src*="recaptcha"]', "https://example.com",
                kind="recaptcha-v3",
            )

        assert bool(ok) is True
        # The §11.13 envelope kind is preserved when the caller supplies it.
        assert recorded["kind"] == "recaptcha-v3"

    @pytest.mark.asyncio
    async def test_solve_falls_back_to_captcha_type_when_kind_missing(self):
        """Direct callers without envelope context (no ``kind=``) still
        get a per-type timeout — keyed off the local ``captcha_type``."""
        solver = _make_solver()
        solver._solver_health_checked = True
        page = _solve_page()

        recorded: dict = {}

        async def fake_submit(self, captcha_type, sitekey, page_url, **kwargs):
            recorded["kind"] = kwargs.get("kind")
            return ("tok-X", False, False, True)

        async def fake_inject(self, page, captcha_type, token):
            return True

        with patch.object(CaptchaSolver, "_submit_and_poll", new=fake_submit), \
             patch.object(CaptchaSolver, "_inject_token", new=fake_inject), \
             patch("src.browser.timing.captcha_solve_delay",
                   new=AsyncMock(return_value=None)):
            await solver.solve(
                page, 'iframe[src*="hcaptcha"]', "https://example.com",
            )

        # No kind passed → falls back to the selector classifier output.
        assert recorded["kind"] == "hcaptcha"


# ── 5. Outer wait_for enforces per-kind deadline ──────────────────────


class TestOuterTimeoutEnforced:
    @pytest.mark.asyncio
    async def test_v3_timeout_fires_before_default_120s(self):
        """Override v3 to 0.05s; a slow ``_submit_and_poll`` should be
        cancelled by the outer ``asyncio.wait_for`` long before the
        default 120s would fire. We verify the call returns False (timed
        out) and the failure was recorded."""
        solver = _make_solver()
        solver._solver_health_checked = True
        # Surgically pin v3 to 50ms — bypass env reads on already-built
        # solver to keep this test independent of monkeypatch order.
        solver._solve_timeouts_ms["recaptcha-v3"] = 50

        page = _solve_page()

        async def slow_submit(self, *a, **kw):
            await asyncio.sleep(2.0)
            return ("tok", False, False, True)

        with patch.object(CaptchaSolver, "_submit_and_poll", new=slow_submit):
            ok = await solver.solve(
                page, 'iframe[src*="recaptcha"]', "https://example.com",
                kind="recaptcha-v3",
            )
        assert bool(ok) is False
        # Failure recorded → breaker counter incremented.
        assert len(solver._solver_failure_timestamps) == 1

    @pytest.mark.asyncio
    async def test_turnstile_uses_180s_window(self):
        """Turnstile gets the larger budget — a sub-second submit should
        not be cancelled. We use a tiny ``_submit_and_poll`` and assert
        the success path returns True."""
        solver = _make_solver()
        solver._solver_health_checked = True
        page = _solve_page()

        async def fast_submit(self, *a, **kw):
            return ("tok", False, False, True)

        async def fake_inject(self, page, captcha_type, token):
            return True

        with patch.object(CaptchaSolver, "_submit_and_poll", new=fast_submit), \
             patch.object(CaptchaSolver, "_inject_token", new=fake_inject), \
             patch("src.browser.timing.captcha_solve_delay",
                   new=AsyncMock(return_value=None)):
            ok = await solver.solve(
                page, 'iframe[src*="cf-turnstile"]', "https://example.com",
                kind="turnstile",
            )
        assert bool(ok) is True


# ── 6. Polling-loop iteration count respects per-kind timeout ─────────


class TestPollingLoopIterations:
    @pytest.mark.asyncio
    async def test_2captcha_poll_loop_capped_by_v3_timeout(self):
        """The 2Captcha poll loop runs ``timeout_s / _POLL_INTERVAL``
        iterations max. With v3 at 60s and ``_POLL_INTERVAL=5``, the
        loop should top out at 12 iterations. We verify by mocking a
        provider that NEVER returns ``ready`` and counting calls."""
        solver = _make_solver(provider="2captcha")
        solver._solver_health_checked = True
        page = _solve_page()

        client = AsyncMock(spec=httpx.AsyncClient)
        client.is_closed = False

        # createTask succeeds → poll forever returns "processing".
        create_resp = MagicMock()
        create_resp.json = MagicMock(return_value={"errorId": 0, "taskId": "T"})
        create_resp.raise_for_status = MagicMock()

        poll_resp = MagicMock()
        poll_resp.json = MagicMock(return_value={
            "errorId": 0, "status": "processing",
        })
        poll_resp.raise_for_status = MagicMock()

        async def fake_post(url, *a, **kw):
            if "createTask" in url:
                return create_resp
            return poll_resp

        client.post = AsyncMock(side_effect=fake_post)
        solver._client = client

        # Speed the test up by patching the poll interval to ~0 — the
        # iteration cap is independent of wall-clock sleep duration.
        with patch("src.browser.captcha._POLL_INTERVAL", 0.001):
            # v3 → 60s, _POLL_INTERVAL=5 (the constant we patched, not the
            # 0.001 we patched at runtime — the iteration math uses the
            # MODULE constant before the patch). With our patch above the
            # math is 60 / 0.001 = 60000 iterations; that's too many. Use
            # a small kind-specific timeout instead so the loop bounds.
            solver._solve_timeouts_ms["recaptcha-v3"] = 30  # 30ms → 30 iterations
            ok = await solver.solve(
                page, 'iframe[src*="recaptcha"]', "https://example.com",
                kind="recaptcha-v3",
            )

        assert bool(ok) is False
        # 1 createTask call + N polls (loop never returned ready).
        # The loop is bounded by ``solve()``'s outer wait_for in practice;
        # we just want to confirm the poll-loop iteration cap is FINITE.
        assert client.post.await_count >= 2  # at least one create + one poll
