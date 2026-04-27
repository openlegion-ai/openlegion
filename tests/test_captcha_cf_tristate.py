"""Tests for §11.3 Cloudflare interstitial tri-state + behavioral-only.

Covers the integration of ``_classify_cf_state`` and ``_classify_behavioral``
inside :meth:`BrowserManager._check_captcha`:

* CF auto-resolving JS challenge: wait + recheck. If page navigates →
  ``solver_outcome="solved"``, ``solver_confidence="medium"``. If still on
  challenge → behavioral envelope.
* CF Under Attack mode (1020) → behavioral envelope, solver never invoked.
* CF interstitial with embedded Turnstile widget → existing solver path,
  but ``solver_confidence`` overridden to ``"low"`` regardless of verdict.
* PerimeterX legacy / modern selectors → ``"px-press-hold"`` envelope.
* DataDome ``/blocker`` path → ``"datadome-behavioral"`` envelope.
* Behavioral classification runs **before** the solver health/breaker
  gates (so CF Under Attack doesn't consume health-check or breaker quota).
* Wait-and-recheck happens at most once per ``_check_captcha`` call.

The ``_classify_*`` helpers each issue a single ``page.evaluate`` JS probe;
tests stub ``page.evaluate`` to return whatever shape we need to drive
each branch and ``asyncio.sleep`` to a no-op so the 8-second wait doesn't
slow the suite.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.browser.captcha import (
    _BEHAVIORAL_PROBE_JS,
    _CF_STATE_PROBE_JS,
    _CLASSIFY_RECAPTCHA_JS,
)
from src.browser.service import _CF_AUTO_WAIT_SECONDS, BrowserManager


def _make_manager(*, solver=None) -> BrowserManager:
    """Build a bare ``BrowserManager`` shell with optional solver mock.

    The §11.16 solver-health side-channels (``is_solver_unreachable`` /
    ``is_breaker_open``) are defaulted to ``False`` so the §11.3
    short-circuits are exercised separately from the §11.16 ones.
    """
    mgr = BrowserManager.__new__(BrowserManager)
    if solver is not None:
        if not isinstance(getattr(solver, "is_solver_unreachable", None), MagicMock):
            solver.is_solver_unreachable = MagicMock(return_value=False)
        if not isinstance(getattr(solver, "is_breaker_open", None), MagicMock):
            solver.is_breaker_open = MagicMock(return_value=False)
    mgr._captcha_solver = solver
    return mgr


def _make_inst(
    *,
    matching_selector: str | None,
    behavioral_probe=None,
    cf_probe=None,
    recaptcha_probe=None,
    title: str = "",
    rechecked_selectors_present: bool | None = None,
) -> MagicMock:
    """Build a mocked CamoufoxInstance for §11.3 tests.

    ``page.evaluate`` is dispatched on the JS string passed in:
      * ``_BEHAVIORAL_PROBE_JS`` → ``behavioral_probe``
      * ``_CF_STATE_PROBE_JS``   → ``cf_probe``
      * ``_CLASSIFY_RECAPTCHA_JS`` → ``recaptcha_probe`` (rare; included
        for completeness — recaptcha selectors aren't covered here)

    ``rechecked_selectors_present`` controls what ``locator(sel).count()``
    returns on the second pass through ``captcha_selectors`` (i.e. after
    the wait+recheck). When ``None``, the locator returns the same
    ``matching_selector`` count both times. When set, the second pass
    returns ``rechecked_selectors_present`` for every selector — used to
    simulate "page navigated away" vs "still on challenge" after wait.
    """
    inst = MagicMock()
    inst.page = MagicMock()
    inst.page.url = "https://example.com"
    inst.page.title = AsyncMock(return_value=title)

    # Track recheck pass: first call to locator(sel).count() per selector
    # is the "initial match"; subsequent calls are the "recheck pass".
    locator_call_counts: dict[str, int] = {}

    def locator(sel: str):
        loc = MagicMock()

        async def _count():
            n = locator_call_counts.get(sel, 0)
            locator_call_counts[sel] = n + 1
            if n == 0:
                # Initial match — only the matching selector returns 1.
                return 1 if sel == matching_selector else 0
            # Recheck pass — driven by ``rechecked_selectors_present``.
            if rechecked_selectors_present is None:
                return 1 if sel == matching_selector else 0
            return 1 if rechecked_selectors_present else 0

        loc.count = _count
        return loc

    inst.page.locator = MagicMock(side_effect=locator)

    # Dispatch ``page.evaluate`` based on which JS body is passed.
    async def evaluate(js, *args, **kwargs):
        if js == _BEHAVIORAL_PROBE_JS:
            return behavioral_probe if behavioral_probe is not None else {
                "px": False,
                "datadome": False,
            }
        if js == _CF_STATE_PROBE_JS:
            return cf_probe if cf_probe is not None else {
                "has_challenge_running": False,
                "has_turnstile": False,
                "has_cf_error_1020": False,
                "has_challenge_error_text": False,
            }
        if js == _CLASSIFY_RECAPTCHA_JS:
            return recaptcha_probe if recaptcha_probe is not None else {
                "enterprise": False, "v3": False, "sitekeys": [],
                "actions_by_key": {}, "invisible_by_key": {},
                "enterprise_script": False, "v3_render_param": None,
            }
        return None

    inst.page.evaluate = evaluate
    inst.lock = asyncio.Lock()
    inst.touch = MagicMock()
    return inst


# ── 1. CF auto-resolving challenge — page clears within wait ─────────────


class TestCfAutoResolves:
    @pytest.mark.asyncio
    async def test_cf_auto_clears_after_wait(self):
        """Auto-resolving CF challenge that clears within the 8s wait.

        ``rechecked_selectors_present=False`` simulates the page navigating
        away from the challenge during the wait. Envelope must report
        ``solver_outcome="solved"`` / ``solver_confidence="medium"``.
        """
        mgr = _make_manager()
        inst = _make_inst(
            matching_selector='iframe[src*="challenges.cloudflare.com"]',
            cf_probe={
                "has_challenge_running": True,
                "has_turnstile": False,
                "has_cf_error_1020": False,
                "has_challenge_error_text": False,
            },
            title="Just a moment...",
            rechecked_selectors_present=False,
        )
        with patch("src.browser.service.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["kind"] == "cf-interstitial-auto"
        assert result["solver_attempted"] is False
        assert result["solver_outcome"] == "solved"
        assert result["next_action"] == "solved"
        assert result["solver_confidence"] == "medium"
        # Exactly one wait+recheck cycle.
        sleep_mock.assert_awaited_once_with(_CF_AUTO_WAIT_SECONDS)


# ── 2. CF auto-resolving — still present after wait → behavioral ─────────


class TestCfAutoStuck:
    @pytest.mark.asyncio
    async def test_cf_auto_still_present_falls_through_to_behavioral(self):
        """Auto-resolving CF challenge that doesn't clear within the wait
        falls through to the behavioral envelope. Solver must NOT be invoked.
        """
        solver = AsyncMock()
        solver.solve = AsyncMock(return_value=True)
        mgr = _make_manager(solver=solver)
        inst = _make_inst(
            matching_selector='iframe[src*="challenges.cloudflare.com"]',
            cf_probe={
                "has_challenge_running": True,
                "has_turnstile": False,
                "has_cf_error_1020": False,
                "has_challenge_error_text": False,
            },
            title="Just a moment...",
            rechecked_selectors_present=True,
        )
        with patch("src.browser.service.asyncio.sleep", new=AsyncMock()):
            result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["kind"] == "cf-interstitial-behavioral"
        assert result["solver_attempted"] is False
        assert result["solver_outcome"] == "skipped_behavioral"
        assert result["next_action"] == "request_captcha_help"
        assert result["solver_confidence"] == "behavioral-only"
        # Solver MUST NOT have been invoked.
        solver.solve.assert_not_awaited()


# ── 3. CF Under Attack (error 1020) → behavioral ─────────────────────────


class TestCfUnderAttack:
    @pytest.mark.asyncio
    async def test_cf_error_1020_is_behavioral(self):
        """Under Attack Mode shows ``cf-error-details`` containing 1020.
        Solver call would be useless — expect behavioral envelope.
        """
        solver = AsyncMock()
        solver.solve = AsyncMock(return_value=True)
        mgr = _make_manager(solver=solver)
        inst = _make_inst(
            matching_selector='iframe[src*="challenges.cloudflare.com"]',
            cf_probe={
                "has_challenge_running": False,
                "has_turnstile": False,
                "has_cf_error_1020": True,
                "has_challenge_error_text": False,
            },
            title="Attention Required! | Cloudflare",
        )
        with patch("src.browser.service.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            result = await mgr._check_captcha(inst)
        assert result["kind"] == "cf-interstitial-behavioral"
        assert result["solver_outcome"] == "skipped_behavioral"
        assert result["next_action"] == "request_captcha_help"
        assert result["solver_confidence"] == "behavioral-only"
        solver.solve.assert_not_awaited()
        # No wait/recheck for the under-attack path.
        sleep_mock.assert_not_awaited()


# ── 4. CF interstitial with Turnstile widget → solver path, low conf ─────


class TestCfTurnstileEmbedded:
    @pytest.mark.asyncio
    async def test_cf_turnstile_embedded_solves_with_low_confidence(self):
        """CF interstitial with a Turnstile widget — solver IS invoked.
        Envelope kind is the new ``cf-interstitial-turnstile`` and
        ``solver_confidence`` is forced to ``"low"`` regardless of verdict.
        """
        solver = AsyncMock()
        solver.solve = AsyncMock(return_value=True)
        mgr = _make_manager(solver=solver)
        inst = _make_inst(
            matching_selector='iframe[src*="challenges.cloudflare.com"]',
            cf_probe={
                "has_challenge_running": False,
                "has_turnstile": True,
                "has_cf_error_1020": False,
                "has_challenge_error_text": False,
            },
            title="Just a moment...",
        )
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "cf-interstitial-turnstile"
        assert result["solver_attempted"] is True
        assert result["solver_outcome"] == "solved"
        # Override applies even on success.
        assert result["solver_confidence"] == "low"
        assert result["next_action"] == "solved"
        solver.solve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cf_turnstile_embedded_low_confidence_on_solver_failure(self):
        """Same override applies when solver returns False (rejected)."""
        solver = AsyncMock()
        solver.solve = AsyncMock(return_value=False)
        mgr = _make_manager(solver=solver)
        inst = _make_inst(
            matching_selector='iframe[src*="challenges.cloudflare.com"]',
            cf_probe={
                "has_challenge_running": False,
                "has_turnstile": True,
                "has_cf_error_1020": False,
                "has_challenge_error_text": False,
            },
            title="Just a moment...",
        )
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "cf-interstitial-turnstile"
        assert result["solver_outcome"] == "rejected"
        # Override fires regardless of solver verdict.
        assert result["solver_confidence"] == "low"


# ── 5. Standalone Turnstile (no "Just a moment") preserves existing flow ──


class TestStandaloneTurnstile:
    @pytest.mark.asyncio
    async def test_standalone_turnstile_unchanged(self):
        """Standalone Turnstile widget (no CF challenge frame, no
        "Just a moment" title) goes through the existing flow with
        ``kind="turnstile"`` and §11.16 firm-kind confidence rules.
        """
        mgr = _make_manager(solver=None)
        inst = _make_inst(
            matching_selector='[class*="cf-turnstile"]',
            cf_probe={
                "has_challenge_running": False,
                "has_turnstile": True,
                "has_cf_error_1020": False,
                "has_challenge_error_text": False,
            },
            # No title → CF state classifier returns "none" because
            # ``title.startswith("Just a moment")`` is False.
            title="Login Page",
        )
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "turnstile"
        # Firm kind without solver → "high".
        assert result["solver_confidence"] == "high"
        assert result["solver_outcome"] == "no_solver"


# ── 6. PerimeterX behavioral selectors ────────────────────────────────────


class TestPerimeterXBehavioral:
    @pytest.mark.asyncio
    async def test_px_legacy_selector_is_behavioral(self):
        """PerimeterX ``#px-captcha`` legacy selector → ``px-press-hold``."""
        solver = AsyncMock()
        solver.solve = AsyncMock(return_value=True)
        mgr = _make_manager(solver=solver)
        inst = _make_inst(
            matching_selector='[class*="captcha"]',
            behavioral_probe={"px": True, "datadome": False},
        )
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "px-press-hold"
        assert result["solver_attempted"] is False
        assert result["solver_outcome"] == "skipped_behavioral"
        assert result["solver_confidence"] == "behavioral-only"
        assert result["next_action"] == "request_captcha_help"
        solver.solve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_px_modern_human_selector_is_behavioral(self):
        """HUMAN-rebrand selector ``[data-human-security]`` → ``px-press-hold``.

        Same outcome envelope as legacy — they're the same challenge
        family, just different DOM markers.
        """
        solver = AsyncMock()
        solver.solve = AsyncMock(return_value=True)
        mgr = _make_manager(solver=solver)
        inst = _make_inst(
            matching_selector='[class*="captcha"]',
            behavioral_probe={"px": True, "datadome": False},
        )
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "px-press-hold"
        solver.solve.assert_not_awaited()


# ── 7. DataDome behavioral vs solvable slider ─────────────────────────────


class TestDataDome:
    @pytest.mark.asyncio
    async def test_datadome_blocker_is_behavioral(self):
        """``/blocker`` path on captcha-delivery.com → behavioral envelope."""
        solver = AsyncMock()
        solver.solve = AsyncMock(return_value=True)
        mgr = _make_manager(solver=solver)
        inst = _make_inst(
            matching_selector='iframe[src*="captcha"]',
            behavioral_probe={"px": False, "datadome": True},
        )
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "datadome-behavioral"
        assert result["solver_outcome"] == "skipped_behavioral"
        assert result["solver_confidence"] == "behavioral-only"
        assert result["next_action"] == "request_captcha_help"
        solver.solve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_datadome_non_blocker_path_falls_through(self):
        """Generic ``captcha-delivery.com`` (no ``/blocker``) MUST NOT
        trip the behavioral classifier — that's the solvable slider
        which routes through §11.5 once it lands. Envelope falls through
        to the existing flow. The matched selector (``[class*="captcha"]``)
        classifies to ``unknown`` in ``_classify_kind``.
        """
        mgr = _make_manager(solver=None)
        inst = _make_inst(
            matching_selector='[class*="captcha"]',
            behavioral_probe={"px": False, "datadome": False},
        )
        result = await mgr._check_captcha(inst)
        # Behavioral classifier returned None → existing flow kicks in.
        assert result["kind"] == "unknown"
        assert result["solver_outcome"] == "no_solver"
        # No behavioral-only here.
        assert result["solver_confidence"] == "low"


# ── 8. Wait-and-recheck only happens once per call ────────────────────────


class TestSingleRecheckPerCall:
    @pytest.mark.asyncio
    async def test_wait_called_at_most_once(self):
        """Even when the page is still on the challenge after the wait,
        we MUST NOT loop. The single wait+recheck cycle is the spec.
        """
        mgr = _make_manager()
        inst = _make_inst(
            matching_selector='iframe[src*="challenges.cloudflare.com"]',
            cf_probe={
                "has_challenge_running": True,
                "has_turnstile": False,
                "has_cf_error_1020": False,
                "has_challenge_error_text": False,
            },
            title="Just a moment...",
            rechecked_selectors_present=True,
        )
        with patch("src.browser.service.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            await mgr._check_captcha(inst)
        # Exactly one wait at the configured duration. No retries.
        assert sleep_mock.await_count == 1
        sleep_mock.assert_awaited_with(_CF_AUTO_WAIT_SECONDS)


# ── 9. Behavioral classifier runs BEFORE solver gates (§11.16 regression) ─


class TestBehavioralBeforeSolverGate:
    """When solver health is DOWN (unreachable / breaker open), a
    behavioral-only page MUST still produce a behavioral envelope —
    NOT ``solver_outcome="no_solver"`` or ``"timeout"``. The classifier
    is invoked before the §11.16 short-circuits so behavioral-only
    flows don't burn health-check / breaker quota.
    """

    @pytest.mark.asyncio
    async def test_behavioral_with_unreachable_solver(self):
        solver = AsyncMock()
        solver.is_solver_unreachable = MagicMock(return_value=True)
        solver.is_breaker_open = MagicMock(return_value=False)
        solver.solve = AsyncMock(return_value=True)
        mgr = _make_manager(solver=solver)
        inst = _make_inst(
            matching_selector='[class*="captcha"]',
            behavioral_probe={"px": True, "datadome": False},
        )
        result = await mgr._check_captcha(inst)
        # Behavioral wins over §11.16 short-circuit.
        assert result["kind"] == "px-press-hold"
        assert result["solver_outcome"] == "skipped_behavioral"
        assert result["solver_confidence"] == "behavioral-only"
        # Solver health was never consulted because behavioral runs first.
        solver.is_solver_unreachable.assert_not_called()
        solver.solve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_behavioral_with_breaker_open(self):
        solver = AsyncMock()
        solver.is_solver_unreachable = MagicMock(return_value=False)
        solver.is_breaker_open = MagicMock(return_value=True)
        solver.solve = AsyncMock(return_value=True)
        mgr = _make_manager(solver=solver)
        inst = _make_inst(
            matching_selector='iframe[src*="captcha"]',
            behavioral_probe={"px": False, "datadome": True},
        )
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "datadome-behavioral"
        assert result["solver_outcome"] == "skipped_behavioral"
        assert result["solver_confidence"] == "behavioral-only"
        solver.is_breaker_open.assert_not_called()
        solver.solve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cf_under_attack_with_unreachable_solver(self):
        """CF Under Attack also runs through ``_classify_cf_state``,
        which is invoked before §11.16 short-circuits. The envelope
        is the CF behavioral kind, not ``no_solver``.
        """
        solver = AsyncMock()
        solver.is_solver_unreachable = MagicMock(return_value=True)
        solver.is_breaker_open = MagicMock(return_value=False)
        solver.solve = AsyncMock(return_value=True)
        mgr = _make_manager(solver=solver)
        inst = _make_inst(
            matching_selector='iframe[src*="challenges.cloudflare.com"]',
            cf_probe={
                "has_challenge_running": False,
                "has_turnstile": False,
                "has_cf_error_1020": True,
                "has_challenge_error_text": False,
            },
        )
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "cf-interstitial-behavioral"
        assert result["solver_outcome"] == "skipped_behavioral"
        # §11.16 health check bypassed.
        solver.is_solver_unreachable.assert_not_called()
        solver.solve.assert_not_awaited()


# ── 10. CF challenge-error-text → behavioral (alternate marker) ───────────


class TestCfChallengeErrorText:
    @pytest.mark.asyncio
    async def test_challenge_error_text_is_behavioral(self):
        """``#challenge-error-text`` on the page is the persistent-challenge
        marker (no widget, just an error block). Maps to behavioral.
        """
        mgr = _make_manager()
        inst = _make_inst(
            matching_selector='iframe[src*="challenges.cloudflare.com"]',
            cf_probe={
                "has_challenge_running": False,
                "has_turnstile": False,
                "has_cf_error_1020": False,
                "has_challenge_error_text": True,
            },
        )
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "cf-interstitial-behavioral"
        assert result["solver_outcome"] == "skipped_behavioral"
        assert result["next_action"] == "request_captcha_help"


# ── 11. CF state="none" preserves existing flow (no behavioral, no solve) ─


class TestCfStateNone:
    @pytest.mark.asyncio
    async def test_cf_state_none_with_cloudflare_iframe(self):
        """CF iframe selector matches but the JS probe finds none of the
        discriminating anchors (no challenge-running, no turnstile, no
        error). Falls through to the existing ``cf-interstitial-auto``
        placeholder kind.
        """
        mgr = _make_manager(solver=None)
        inst = _make_inst(
            matching_selector='iframe[src*="challenges.cloudflare.com"]',
            cf_probe={
                "has_challenge_running": False,
                "has_turnstile": False,
                "has_cf_error_1020": False,
                "has_challenge_error_text": False,
            },
            title="Just a moment...",
        )
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "cf-interstitial-auto"
        assert result["solver_outcome"] == "no_solver"
        # Placeholder kind → "low" confidence per ``_kind_confidence``.
        assert result["solver_confidence"] == "low"
