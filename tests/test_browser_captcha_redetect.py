"""Tests for §11.4 / §18.2 — re-detect captcha after non-navigate actions.

Covers the post-action MutationObserver-based redetect path that wraps
``click`` / ``type_text`` / ``press_key`` / ``fill_form``:

  * Happy path — action adds a captcha iframe → redetect fires →
    ``_check_captcha`` invoked → action response carries §11.13 envelope.
  * Empty-mutation path — action with no captcha-shaped DOM additions →
    no ``_check_captcha`` call → no ``captcha`` field on the response.
  * Pre-existing captcha — captcha was already on the page (no DOM
    mutation during the action) → redetect read-back returns empty →
    no ``_check_captcha`` call (correct: navigate-time
    ``_check_captcha`` is the right defense for pre-existing captchas).
  * Navigation during action — page swap or URL change between install
    and read-back → probe is gone → redetect skips, action still
    succeeds (the navigate-time ``_check_captcha`` covers the new page).
  * Install failure — ``page.evaluate`` raises on install → action
    still runs, no read-back attempted, no ``_check_captcha`` call.
  * Read-back failure — ``page.evaluate`` raises on read-back → no
    ``_check_captcha`` call, action still returns its result.
  * Rate-limit — back-to-back actions inside
    :data:`_REDETECT_MIN_INTERVAL_S` only fire ``_check_captcha`` once.
  * Flag-disabled — ``BROWSER_CAPTCHA_REDETECT_ENABLED=false`` →
    ``page.evaluate`` never invoked at all.
  * Metered-solve cost-cap — auto-triggered redetect inherits the
    full :meth:`_metered_solve` gate stack; envelope reports
    ``solver_outcome="cost_cap"`` when over cap.
  * Behavioral classifier — auto-triggered redetect on a CF-behavioral
    page → envelope reports ``solver_outcome="skipped_behavioral"``.
  * Snapshot integration — snapshot following a redetect-triggering
    action carries the pending envelope and clears it on first read.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.browser import captcha_cost_counter as cost
from src.browser import service as svc
from src.browser.captcha import SolveResult
from src.browser.service import (
    _CAPTCHA_REDETECT_SELECTORS,
    BrowserManager,
    CamoufoxInstance,
)

# ── helpers ────────────────────────────────────────────────────────────────


def _solved() -> SolveResult:
    return SolveResult(
        token="tok",
        injection_succeeded=True,
        used_proxy_aware=False,
        compat_rejected=False,
    )


def _mk_solver(
    *,
    return_value: SolveResult | None = None,
    provider: str = "2captcha",
    unreachable: bool = False,
    breaker_open: bool = False,
) -> MagicMock:
    s = MagicMock()
    s.provider = provider
    s.solve = AsyncMock(return_value=return_value or _solved())
    s.is_solver_unreachable = AsyncMock(return_value=unreachable)
    s.is_breaker_open = MagicMock(return_value=breaker_open)
    return s


def _mk_page(
    *,
    redetect_hits: list[str] | None = None,
    captcha_locator_for: str | None = None,
    install_raises: bool = False,
    readback_raises: bool = False,
    url: str = "https://example.com",
):
    """Build a Playwright-shaped page mock with controllable
    evaluate / locator behaviour.

    ``redetect_hits`` — selectors to return from the read-back call.
    ``captcha_locator_for`` — selector that ``_check_captcha`` should
    see as present (its locator(...).count() returns 1); all others
    return 0. ``None`` means no captcha visible to ``_check_captcha``.
    """
    page = MagicMock()
    page.url = url

    async def _evaluate(js, *args):
        if install_raises and "MutationObserver" in js:
            raise RuntimeError("install fail")
        if readback_raises and "p.adds" in js:
            raise RuntimeError("readback fail")
        if "MutationObserver" in js:
            return None
        if "p.adds" in js:
            return list(redetect_hits or [])
        return None

    page.evaluate = AsyncMock(side_effect=_evaluate)

    def _locator(sel):
        loc = MagicMock()
        if captcha_locator_for is not None and sel == captcha_locator_for:
            loc.count = AsyncMock(return_value=1)
        else:
            loc.count = AsyncMock(return_value=0)
        return loc

    page.locator = MagicMock(side_effect=_locator)
    page.keyboard = MagicMock()
    page.keyboard.press = AsyncMock()
    page.viewport_size = {"width": 1280, "height": 720}
    page.query_selector_all = AsyncMock(return_value=[])
    page.title = AsyncMock(return_value="title")
    return page


def _mk_inst(page=None, agent_id: str = "agent-1") -> CamoufoxInstance:
    page = page if page is not None else _mk_page()
    return CamoufoxInstance(agent_id, MagicMock(), MagicMock(), page)


@pytest.fixture(autouse=True)
async def _isolate_state(tmp_path, monkeypatch):
    """Reset shared in-memory state so tests don't leak into each other.

    ``CAPTCHA_COST_COUNTER_PATH`` is redirected to a per-test file so
    cost-cap tests don't pollute peer cases. The solve-rate window and
    audit buckets are module-globals; clearing them here keeps each
    test starting from a clean slate.
    """
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
    m = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
    m._captcha_solver = _mk_solver()
    return m


# ── 1. Happy path — captcha appears mid-click via DOM mutation ─────────────


class TestRedetectHappyPath:
    @pytest.mark.asyncio
    async def test_click_with_added_captcha_iframe_surfaces_envelope(self, mgr):
        """A click that adds a captcha iframe to the DOM fires the
        redetect path; an unsolved-captcha envelope is surfaced inline
        AND stashed for the next snapshot. The "auto-solved" case is
        deliberately NOT surfaced (matching navigate-path semantics) —
        nothing for the agent to do."""
        # Build a page where the read-back captures a recaptcha iframe
        # AND the locator probe in ``_check_captcha`` sees it as present.
        sel = 'iframe[src*="recaptcha"]'
        page = _mk_page(
            redetect_hits=[sel],
            captcha_locator_for=sel,
        )
        # Force the no-solver path so the envelope is surfaced (the
        # "solved" path is a no-op for the agent and is correctly
        # filtered by the wrapper).
        mgr._captcha_solver = _mk_solver(unreachable=True)

        async def _no_recap(p):
            return {"variant": "unknown"}

        async def _no_behavioral(p):
            return None

        with patch.object(svc, "_classify_recaptcha", new=_no_recap), \
             patch.object(svc, "_classify_behavioral", new=_no_behavioral):
            inst = _mk_inst(page=page)
            inst.touch()
            async with inst.lock:
                async def _do():
                    return {"success": True, "data": {"clicked": "btn"}}

                result, envelope = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        assert result == {"success": True, "data": {"clicked": "btn"}}
        assert envelope is not None
        assert envelope.get("captcha_found") is True
        assert envelope.get("solver_outcome") == "no_solver"
        # Inst.pending stash carries the same envelope so a follow-up
        # snapshot can surface it.
        assert inst._pending_captcha_envelope is not None

    @pytest.mark.asyncio
    async def test_solved_envelope_is_filtered_inline(self, mgr):
        """When the solver auto-clears the captcha, there's nothing for
        the agent to do — the wrapper deliberately returns ``None`` for
        the envelope so the action response stays clean. Mirrors the
        ``navigate`` path which also drops solved envelopes."""
        sel = 'iframe[src*="recaptcha"]'
        page = _mk_page(redetect_hits=[sel], captcha_locator_for=sel)

        async def _no_recap(p):
            return {"variant": "unknown"}

        async def _no_behavioral(p):
            return None

        with patch.object(svc, "_classify_recaptcha", new=_no_recap), \
             patch.object(svc, "_classify_behavioral", new=_no_behavioral):
            inst = _mk_inst(page=page)
            async with inst.lock:
                async def _do():
                    return {"success": True}

                _, envelope = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        assert envelope is None
        assert inst._pending_captcha_envelope is None


# ── 2. No captcha — added nodes don't match selectors → no _check_captcha ──


class TestRedetectNoMatch:
    @pytest.mark.asyncio
    async def test_no_captcha_means_no_check_captcha_call(self, mgr):
        # Read-back returns empty list — no captcha-shaped additions.
        page = _mk_page(redetect_hits=[])
        inst = _mk_inst(page=page)

        with patch.object(mgr, "_check_captcha", new=AsyncMock()) as cc:
            async with inst.lock:
                async def _do():
                    return {"success": True, "data": {"x": 1}}

                result, envelope = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        assert result == {"success": True, "data": {"x": 1}}
        assert envelope is None
        cc.assert_not_awaited()


# ── 3. Pre-existing captcha (no DOM mutation during action) ────────────────


class TestRedetectPreExisting:
    @pytest.mark.asyncio
    async def test_pre_existing_captcha_with_no_mutation_skipped(self, mgr):
        """Captcha was already on the page; the action did not add new
        captcha-shaped nodes. The MutationObserver only sees additions —
        so no hits → no ``_check_captcha`` call. Pre-existing captchas
        are the navigate-time ``_check_captcha``'s job.
        """
        sel = 'iframe[src*="hcaptcha"]'
        # Read-back empty (no NEW additions). But a locator probe WOULD
        # find the pre-existing captcha if we asked — we explicitly
        # don't, so verify _check_captcha is never called.
        page = _mk_page(
            redetect_hits=[],
            captcha_locator_for=sel,
        )
        inst = _mk_inst(page=page)
        with patch.object(mgr, "_check_captcha", new=AsyncMock()) as cc:
            async with inst.lock:
                async def _do():
                    return {"success": True}

                result, envelope = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        assert result == {"success": True}
        assert envelope is None
        cc.assert_not_awaited()


# ── 4. Navigation during action — probe gone, fall through ────────────────


class TestRedetectNavigationDuringAction:
    @pytest.mark.asyncio
    async def test_url_change_between_install_and_readback(self, mgr):
        page = _mk_page(redetect_hits=['iframe[src*="recaptcha"]'])
        inst = _mk_inst(page=page)

        # Simulate URL change DURING the action by mutating page.url
        # before the read-back runs.
        async def _do():
            page.url = "https://example.com/after-nav"
            return {"success": True}

        with patch.object(mgr, "_check_captcha", new=AsyncMock()) as cc:
            async with inst.lock:
                result, envelope = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        assert result == {"success": True}
        # Navigation was detected — read-back skipped, no _check_captcha.
        assert envelope is None
        cc.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_page_swap_during_action(self, mgr):
        page = _mk_page(redetect_hits=['iframe[src*="captcha"]'])
        inst = _mk_inst(page=page)

        async def _do():
            # Simulate tab swap: inst.page now points at a brand-new mock.
            inst.page = _mk_page(redetect_hits=[])
            return {"success": True}

        with patch.object(mgr, "_check_captcha", new=AsyncMock()) as cc:
            async with inst.lock:
                result, envelope = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        assert result == {"success": True}
        assert envelope is None
        cc.assert_not_awaited()


# ── 5. Install failure — action still runs, no read-back attempt ──────────


class TestRedetectInstallFailure:
    @pytest.mark.asyncio
    async def test_install_evaluate_raises(self, mgr):
        # Even if read-back would report stale hits, install failure means
        # the probe state is not trustworthy and must not trigger solving.
        page = _mk_page(
            install_raises=True,
            redetect_hits=['iframe[src*="recaptcha"]'],
        )
        inst = _mk_inst(page=page)
        with patch.object(mgr, "_check_captcha", new=AsyncMock()) as cc:
            async with inst.lock:
                async def _do():
                    return {"success": True, "data": "ok"}

                result, envelope = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        assert result == {"success": True, "data": "ok"}
        assert envelope is None
        cc.assert_not_awaited()


# ── 6. Read-back failure — action still succeeds, no auto-trigger ─────────


class TestRedetectReadbackFailure:
    @pytest.mark.asyncio
    async def test_readback_evaluate_raises_skips_check_captcha(self, mgr):
        """Read-back failure (e.g. CSP blocked the eval, page race) must
        not break the action OR fire ``_check_captcha`` against
        unverified probe state. Behavioural choice documented in the
        wrapper docstring: empty-hits semantics on failure.
        """
        page = _mk_page(readback_raises=True, redetect_hits=[])
        inst = _mk_inst(page=page)
        with patch.object(mgr, "_check_captcha", new=AsyncMock()) as cc:
            async with inst.lock:
                async def _do():
                    return {"success": True}

                result, envelope = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        assert result == {"success": True}
        assert envelope is None
        cc.assert_not_awaited()


# ── 7. Rate-limit — only first of two back-to-back actions fires ──────────


class TestRedetectRateLimit:
    @pytest.mark.asyncio
    async def test_back_to_back_actions_invoke_check_captcha_once(self, mgr):
        sel = 'iframe[src*="recaptcha"]'
        page = _mk_page(redetect_hits=[sel], captcha_locator_for=sel)

        # Stub _check_captcha so we count calls and don't depend on
        # solver / classifier behavior here.
        envelope = {
            "captcha_found": True,
            "kind": "recaptcha-v2-checkbox",
            "solver_attempted": False,
            "solver_outcome": "no_solver",
            "next_action": "request_captcha_help",
        }
        check_spy = AsyncMock(return_value=envelope)
        inst = _mk_inst(page=page)
        with patch.object(mgr, "_check_captcha", new=check_spy):
            async with inst.lock:
                async def _do():
                    return {"success": True}

                # First call — fires (within the rate-limit window
                # since ``_last_redetect_ts`` is 0).
                _, env1 = await mgr._with_captcha_redetect(
                    inst, _do(),
                )
                # Second call immediately after — within the
                # ``_REDETECT_MIN_INTERVAL_S`` window → suppressed.
                _, env2 = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        assert env1 is not None
        assert env2 is None
        assert check_spy.await_count == 1

    @pytest.mark.asyncio
    async def test_first_redetect_not_suppressed_when_monotonic_near_zero(
        self, mgr,
    ):
        """``_last_redetect_ts=0`` is the sentinel for "never fired".
        Some Python/runtime combinations expose small monotonic values at
        process start, so the first call must not compare ``now - 0`` and
        accidentally suppress itself."""
        sel = 'iframe[src*="recaptcha"]'
        page = _mk_page(redetect_hits=[sel], captcha_locator_for=sel)
        envelope = {
            "captcha_found": True,
            "kind": "recaptcha-v2-checkbox",
            "solver_attempted": False,
            "solver_outcome": "no_solver",
            "next_action": "request_captcha_help",
        }
        check_spy = AsyncMock(return_value=envelope)
        inst = _mk_inst(page=page)
        with patch.object(mgr, "_check_captcha", new=check_spy), \
             patch.object(svc.time, "monotonic", return_value=1.0):
            async with inst.lock:
                async def _do():
                    return {"success": True}

                _, env = await mgr._with_captcha_redetect(inst, _do())

        assert env is not None
        assert inst._last_redetect_ts == 1.0
        check_spy.assert_awaited_once()


# ── 8. Flag-disabled — wrapper is a passthrough, no JS install/readback ───


class TestRedetectFlagDisabled:
    @pytest.mark.asyncio
    async def test_flag_off_skips_all_observer_work(self, mgr, monkeypatch):
        monkeypatch.setenv("BROWSER_CAPTCHA_REDETECT_ENABLED", "false")
        page = _mk_page(redetect_hits=['iframe[src*="recaptcha"]'])
        inst = _mk_inst(page=page)
        with patch.object(mgr, "_check_captcha", new=AsyncMock()) as cc:
            async with inst.lock:
                async def _do():
                    return {"success": True, "data": "x"}

                result, envelope = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        assert result == {"success": True, "data": "x"}
        assert envelope is None
        cc.assert_not_awaited()
        # Probe install/read-back never invoked.
        assert page.evaluate.await_count == 0


# ── 9. Metered-solve gates — cost cap fires on auto-triggered path ────────


class TestRedetectMeteredSolveCostCap:
    @pytest.mark.asyncio
    async def test_cost_cap_envelope_on_redetect_path(self, mgr, monkeypatch):
        """Auto-triggered ``_check_captcha`` from the redetect wrapper
        flows through ``_metered_solve``, so the cost-cap gate fires
        identically to the navigate auto-detect path."""
        monkeypatch.setenv("CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", "0.50")
        # 50_000 millicents = $0.50 cap; pre-fill to exceed it.
        await cost.add_cost("agent-1", 60_000)

        sel = 'iframe[src*="recaptcha"]'
        page = _mk_page(redetect_hits=[sel], captcha_locator_for=sel)

        async def _no_recap(p):
            return {"variant": "unknown"}

        async def _no_behavioral(p):
            return None

        with patch.object(svc, "_classify_recaptcha", new=_no_recap), \
             patch.object(svc, "_classify_behavioral", new=_no_behavioral):
            inst = _mk_inst(page=page)
            async with inst.lock:
                async def _do():
                    return {"success": True}

                _, envelope = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        assert envelope is not None
        assert envelope.get("captcha_found") is True
        assert envelope.get("solver_outcome") == "cost_cap"
        # Solver mock was NEVER awaited — gate fired before the HTTP call.
        mgr._captcha_solver.solve.assert_not_awaited()


# ── 10. Behavioral classifier — CF-behavioral envelope on redetect path ───


class TestRedetectBehavioralClassification:
    @pytest.mark.asyncio
    async def test_cf_behavioral_envelope_on_redetect_path(self, mgr):
        """When the redetect-driven ``_check_captcha`` lands on a
        CF-behavioral page, the §11.3 classifier short-circuits to
        ``skipped_behavioral`` BEFORE any solver gate runs. Auto-trigger
        path inherits the classification identically."""
        sel = 'iframe[src*="challenges.cloudflare.com"]'
        page = _mk_page(redetect_hits=[sel], captcha_locator_for=sel)

        async def _no_recap(p):
            return {"variant": "unknown"}

        async def _no_behavioral(p):
            return None

        async def _cf_behavioral(p):
            return "behavioral"

        with patch.object(svc, "_classify_recaptcha", new=_no_recap), \
             patch.object(svc, "_classify_behavioral", new=_no_behavioral), \
             patch.object(svc, "_classify_cf_state", new=_cf_behavioral):
            inst = _mk_inst(page=page)
            async with inst.lock:
                async def _do():
                    return {"success": True}

                _, envelope = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        assert envelope is not None
        assert envelope.get("captcha_found") is True
        assert envelope.get("solver_outcome") == "skipped_behavioral"
        assert envelope.get("kind") == "cf-interstitial-behavioral"
        # Solver was never contacted.
        mgr._captcha_solver.solve.assert_not_awaited()


# ── 11. Selector list parity — JS read-back uses the same list as Python ──


class TestRedetectSelectorListParity:
    """The read-back JS must intersect added nodes against the SAME
    selector list ``_check_captcha`` walks; otherwise the auto-trigger
    can fire on additions that ``_check_captcha`` won't recognise (or
    vice-versa). The shared :data:`_CAPTCHA_REDETECT_SELECTORS` tuple
    enforces the invariant statically — this test asserts it covers
    every selector ``_check_captcha`` actually probes for, so a future
    edit that touches one list and not the other shows up here."""

    def test_redetect_selectors_match_check_captcha_selectors(self):
        # The selector list inside ``_check_captcha`` (string-literal
        # near top of the method body). We can't introspect it
        # programmatically without executing the method, so we
        # hard-code the expected set and rely on a single source of
        # truth: when the inline list in ``_check_captcha`` changes,
        # whoever changes it must also update ``_CAPTCHA_REDETECT_SELECTORS``
        # AND this test.
        expected = {
            'iframe[src*="recaptcha"]',
            'iframe[src*="hcaptcha"]',
            'iframe[src*="challenges.cloudflare.com"]',
            'iframe[src*="captcha"]',
            '[class*="cf-turnstile"]',
            '[class*="captcha"]',
            '#captcha',
        }
        assert set(_CAPTCHA_REDETECT_SELECTORS) == expected


# ── 12. Snapshot integration — pending envelope surfaces and clears ───────


class TestSnapshotPendingEnvelope:
    @pytest.mark.asyncio
    async def test_snapshot_surfaces_then_clears_pending_envelope(self, mgr):
        """An envelope stashed by a prior action's redetect must surface
        on the next snapshot AND get cleared so a third snapshot
        doesn't repeat the same envelope.
        """
        # Pre-stash a pending envelope to simulate a prior action's
        # redetect having fired.
        pending = {
            "captcha_found": True,
            "kind": "recaptcha-v2-checkbox",
            "solver_attempted": False,
            "solver_outcome": "no_solver",
            "next_action": "request_captcha_help",
        }
        page = MagicMock()
        page.url = "https://example.com"
        inst = _mk_inst(page=page)
        inst._pending_captcha_envelope = pending
        mgr._instances["agent-1"] = inst

        async def _fresh_snapshot(*args, **kwargs):
            return {
                "success": True,
                "data": {"snapshot": "(text)", "refs": {}},
            }

        with patch.object(
            mgr,
            "_snapshot_impl",
            new=AsyncMock(side_effect=_fresh_snapshot),
        ):
            r1 = await mgr.snapshot("agent-1")
            r2 = await mgr.snapshot("agent-1")

        # First snapshot carries the envelope.
        assert r1["data"].get("captcha") == pending
        # Second snapshot does not — pending was cleared on first read.
        assert "captcha" not in r2["data"]
        assert inst._pending_captcha_envelope is None


# ── 13. Action-response shape — captcha is ADDITIVE, not breaking ─────────


class TestActionResponseShape:
    """The wrapper exposes the §11.13 envelope under a ``"captcha"`` key
    on the action response. Existing callers that don't read this key
    must continue to work unchanged. The wrapper uses
    :py:meth:`dict.setdefault` so a body that already populated the key
    (e.g. ``fill_form``'s mid-flow envelope path) is not overwritten.
    """

    @pytest.mark.asyncio
    async def test_existing_response_keys_unchanged(self, mgr):
        sel = 'iframe[src*="recaptcha"]'
        page = _mk_page(redetect_hits=[sel], captcha_locator_for=sel)
        # Force the no-solver envelope so the wrapper actually returns
        # the captcha envelope (the solved/auto-clear path is filtered).
        mgr._captcha_solver = _mk_solver(unreachable=True)

        async def _no_recap(p):
            return {"variant": "unknown"}

        async def _no_behavioral(p):
            return None

        with patch.object(svc, "_classify_recaptcha", new=_no_recap), \
             patch.object(svc, "_classify_behavioral", new=_no_behavioral):
            inst = _mk_inst(page=page)
            async with inst.lock:
                original = {
                    "success": True,
                    "data": {"clicked": "submit"},
                    "extra": {"k": "v"},
                }

                async def _do():
                    return original

                result, envelope = await mgr._with_captcha_redetect(
                    inst, _do(),
                )

        # All original keys preserved verbatim — wrapper does not
        # rewrite the action result, only stashes the envelope for the
        # caller to attach.
        assert result is original
        assert result["data"]["clicked"] == "submit"
        assert result["extra"] == {"k": "v"}
        assert envelope is not None


# ── 14. Disabled-flag rate-limit — flag short-circuit doesn't poison TS ───


class TestRedetectDisabledDoesNotConsumeRateBudget:
    """When the flag disables redetect, the wrapper is a passthrough.
    It must NOT update ``inst._last_redetect_ts`` — otherwise toggling
    the flag back on later would silently rate-limit the first real
    re-detection in surprising ways.
    """

    @pytest.mark.asyncio
    async def test_disabled_flag_keeps_redetect_ts_zero(self, mgr, monkeypatch):
        monkeypatch.setenv("BROWSER_CAPTCHA_REDETECT_ENABLED", "false")
        page = _mk_page()
        inst = _mk_inst(page=page)
        async with inst.lock:
            async def _do():
                return {"success": True}

            await mgr._with_captcha_redetect(inst, _do())
        assert inst._last_redetect_ts == 0.0


# ── 15. Rate-limit honors clock — once outside the window, fires again ────


class TestRedetectRateLimitClockExpiry:
    @pytest.mark.asyncio
    async def test_after_window_expires_redetect_fires_again(self, mgr):
        sel = 'iframe[src*="recaptcha"]'
        page = _mk_page(redetect_hits=[sel], captcha_locator_for=sel)

        async def _no_recap(p):
            return {"variant": "unknown"}

        async def _no_behavioral(p):
            return None

        with patch.object(svc, "_classify_recaptcha", new=_no_recap), \
             patch.object(svc, "_classify_behavioral", new=_no_behavioral):
            inst = _mk_inst(page=page)
            check_spy = AsyncMock(wraps=mgr._check_captcha)
            with patch.object(mgr, "_check_captcha", new=check_spy):
                async with inst.lock:
                    async def _do():
                        return {"success": True}

                    # First call — fires.
                    await mgr._with_captcha_redetect(inst, _do())
                    # Push the timestamp far enough into the past that
                    # the next call is outside the rate-limit window.
                    inst._last_redetect_ts = (
                        time.monotonic() - svc._REDETECT_MIN_INTERVAL_S - 0.5
                    )
                    # Second call — fires again.
                    await mgr._with_captcha_redetect(inst, _do())

        assert check_spy.await_count == 2


# ── 16. B1 STEALTH — probe-var randomisation + non-enumerable property ────


class TestProbeVarStealth:
    """The captcha re-detection MutationObserver state is stored on a
    per-instance random window property name AND defined as
    ``enumerable: false`` so anti-bot scripts walking
    ``Object.keys(window)`` cannot fingerprint our automation. The JS
    install template uses ``Object.defineProperty``; the probe-var name
    matches ``__telemetry_<8-hex>`` and is generated independently per
    :class:`CamoufoxInstance`.
    """

    def test_install_js_uses_define_property(self):
        """Install JS uses ``Object.defineProperty`` (not ``window.X = …``)
        so the probe state is non-enumerable and stays out of
        ``Object.keys`` walks."""
        from src.browser.service import _JS_CAPTCHA_REDETECT_INSTALL
        assert "Object.defineProperty" in _JS_CAPTCHA_REDETECT_INSTALL
        assert "enumerable: false" in _JS_CAPTCHA_REDETECT_INSTALL
        # The hard-coded ``window.__ol_captcha_probe = …`` enumerable
        # global must NOT appear — that was the pre-fix shape.
        assert "window.__ol_captcha_probe" not in _JS_CAPTCHA_REDETECT_INSTALL

    def test_probe_var_name_matches_telemetry_pattern(self):
        """Per-instance probe-var name follows ``__telemetry_<8-hex>`` —
        mimics ad-tech / RUM globals that real sites carry, so a passive
        observer cannot trivially distinguish ours from a third-party
        SDK."""
        import re as _re
        inst = _mk_inst()
        assert _re.match(
            r"^__telemetry_[0-9a-f]{8}$", inst._captcha_probe_var,
        ), inst._captcha_probe_var

    def test_two_instances_have_independent_probe_vars(self):
        """Probe-var names are generated independently per
        :class:`CamoufoxInstance` so two browsers can't be linked by a
        shared global. ``secrets.token_hex(4)`` has 32 bits of entropy
        — collision probability across two draws is ~2e-10."""
        inst_a = _mk_inst(agent_id="agent-a")
        inst_b = _mk_inst(agent_id="agent-b")
        assert inst_a._captcha_probe_var != inst_b._captcha_probe_var

    @pytest.mark.asyncio
    async def test_install_call_passes_probe_var_to_evaluate(self, mgr):
        """The install JS receives the per-instance probe-var name as
        its first argument so the JS knows which window property to
        attach the state under."""
        page = _mk_page()
        inst = _mk_inst(page=page)
        async with inst.lock:
            async def _do():
                return {"success": True}

            await mgr._with_captcha_redetect(inst, _do())

        # Find the install evaluate call — install JS contains
        # MutationObserver in body.
        install_calls = [
            c for c in page.evaluate.await_args_list
            if "MutationObserver" in c.args[0]
        ]
        assert install_calls, "install JS was never evaluated"
        # Second positional arg is the probe-var name.
        probe_var_arg = install_calls[0].args[1]
        assert probe_var_arg == inst._captcha_probe_var

    @pytest.mark.asyncio
    async def test_readback_call_passes_probe_var_in_args(self, mgr):
        """The read-back JS now expects ``[probeVar, selectors]`` so
        delete works against the same property that install set."""
        sel = 'iframe[src*="recaptcha"]'
        page = _mk_page(redetect_hits=[sel])
        inst = _mk_inst(page=page)
        async with inst.lock:
            async def _do():
                return {"success": True}

            await mgr._with_captcha_redetect(inst, _do())

        readback_calls = [
            c for c in page.evaluate.await_args_list
            if "p.adds" in c.args[0]
        ]
        assert readback_calls, "read-back JS was never evaluated"
        # Second positional arg is ``[probe_var, selectors]``.
        args_arg = readback_calls[0].args[1]
        assert isinstance(args_arg, list) and len(args_arg) == 2
        assert args_arg[0] == inst._captcha_probe_var
        assert isinstance(args_arg[1], list)


# ── 17. B3 — preserve readback hits on action failure ────────────────────


class TestActionFailurePreservesCaptchaHits:
    """When the wrapped action raises after a successful probe install,
    the wrapper now runs READBACK with the real selector list (not an
    empty list) so the captured DOM additions can be used to attach
    diagnostic context to the propagated exception. This lets callers
    distinguish "action failed because a captcha appeared mid-flight"
    from a generic action failure.
    """

    @pytest.mark.asyncio
    async def test_failure_after_captcha_attaches_hits_attribute(self, mgr):
        """Action raises after probe install; read-back returned hits
        → propagated exception carries ``captcha_redetect_hits``."""
        sel = 'iframe[src*="recaptcha"]'
        page = _mk_page(redetect_hits=[sel])
        inst = _mk_inst(page=page)

        class _BoomError(RuntimeError):
            pass

        async def _failing_action():
            raise _BoomError("simulated action failure")

        async with inst.lock:
            with pytest.raises(_BoomError) as excinfo:
                await mgr._with_captcha_redetect(inst, _failing_action())

        # The exception carries the redetect hits so observability
        # downstream can flag this failure as captcha-induced.
        assert hasattr(excinfo.value, "captcha_redetect_hits")
        assert excinfo.value.captcha_redetect_hits == [sel]

    @pytest.mark.asyncio
    async def test_failure_without_captcha_no_hits_attribute(self, mgr):
        """Action raises after probe install but no captcha-shaped DOM
        additions occurred → no ``captcha_redetect_hits`` attribute is
        attached (we only attach when there's signal to attach)."""
        # Empty redetect_hits so read-back returns []
        page = _mk_page(redetect_hits=[])
        inst = _mk_inst(page=page)

        class _BoomError(RuntimeError):
            pass

        async def _failing_action():
            raise _BoomError("simulated action failure")

        async with inst.lock:
            with pytest.raises(_BoomError) as excinfo:
                await mgr._with_captcha_redetect(inst, _failing_action())

        assert not hasattr(excinfo.value, "captcha_redetect_hits")
