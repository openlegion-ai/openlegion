"""Tests for the §11.13 structured CAPTCHA detection envelope.

Covers each path through `BrowserManager._check_captcha` and
`BrowserManager.detect_captcha`, asserting that the response data block
matches the schema described in the plan:

  {captcha_found: false}                       # no captcha
  {captcha_found: true, kind: <enum>,          # captcha found
   solver_attempted: <bool>,
   solver_outcome: <enum>,
   injection_failure_reason: null | <enum>,
   solver_confidence: <enum>,
   next_action: <enum>}

Plus the soft-deprecated legacy `type` and `message` fields.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.browser.service import BrowserManager

# Selectors checked in order by `_check_captcha`. We use this to drive
# the mock such that exactly one selector "matches" per scenario.
_SELECTOR_ORDER = [
    'iframe[src*="recaptcha"]',
    'iframe[src*="hcaptcha"]',
    'iframe[src*="challenges.cloudflare.com"]',
    'iframe[src*="captcha"]',
    '[class*="cf-turnstile"]',
    '[class*="captcha"]',
    "#captcha",
]


def _make_manager(*, solver=None) -> BrowserManager:
    mgr = BrowserManager.__new__(BrowserManager)
    mgr._captcha_solver = solver
    return mgr


def _make_inst(matching_selector: str | None, *, title: str = "") -> MagicMock:
    """Build a mocked CamoufoxInstance whose `.page.locator(sel).count()`
    returns 1 only for `matching_selector`. `None` means no captcha.
    """
    inst = MagicMock()
    inst.page = MagicMock()
    inst.page.url = "https://example.com"
    inst.page.title = AsyncMock(return_value=title)

    def locator(sel: str):
        loc = MagicMock()
        loc.count = AsyncMock(return_value=1 if sel == matching_selector else 0)
        return loc

    inst.page.locator = MagicMock(side_effect=locator)
    inst.lock = asyncio.Lock()
    inst.touch = MagicMock()
    return inst


# ── 1. No captcha ─────────────────────────────────────────────────────────


class TestNoCaptcha:
    @pytest.mark.asyncio
    async def test_no_captcha_returns_minimal_envelope(self):
        mgr = _make_manager()
        inst = _make_inst(None)
        result = await mgr._check_captcha(inst)
        assert result == {"captcha_found": False}

    @pytest.mark.asyncio
    async def test_detect_captcha_no_captcha_path(self):
        mgr = _make_manager()
        inst = _make_inst(None)
        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.detect_captcha("a1")
        assert result["success"] is True
        assert result["data"]["captcha_found"] is False
        # Legacy back-compat
        assert result["data"]["message"] == "No CAPTCHA detected"


# ── 2. reCAPTCHA detect, no solver ────────────────────────────────────────


class TestRecaptchaNoSolver:
    @pytest.mark.asyncio
    async def test_recaptcha_no_solver(self):
        mgr = _make_manager(solver=None)
        inst = _make_inst('iframe[src*="recaptcha"]')
        result = await mgr._check_captcha(inst)
        assert result == {
            "captcha_found": True,
            "kind": "recaptcha-v2-checkbox",
            "solver_attempted": False,
            "solver_outcome": "no_solver",
            "injection_failure_reason": None,
            "solver_confidence": "high",
            "next_action": "notify_user",
        }


# ── 3. reCAPTCHA detect, solver succeeds ──────────────────────────────────


class TestRecaptchaSolverSuccess:
    @pytest.mark.asyncio
    async def test_recaptcha_solved(self):
        solver = AsyncMock()
        solver.solve = AsyncMock(return_value=True)
        mgr = _make_manager(solver=solver)
        inst = _make_inst('iframe[src*="recaptcha"]')
        result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["kind"] == "recaptcha-v2-checkbox"
        assert result["solver_attempted"] is True
        assert result["solver_outcome"] == "solved"
        assert result["solver_confidence"] == "high"
        assert result["next_action"] == "solved"
        assert result["injection_failure_reason"] is None
        solver.solve.assert_awaited_once()


# ── 4. reCAPTCHA detect, solver timeout ───────────────────────────────────


class TestRecaptchaSolverTimeout:
    @pytest.mark.asyncio
    async def test_recaptcha_timeout(self):
        solver = AsyncMock()
        solver.solve = AsyncMock(side_effect=asyncio.TimeoutError())
        mgr = _make_manager(solver=solver)
        inst = _make_inst('iframe[src*="recaptcha"]')
        result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["kind"] == "recaptcha-v2-checkbox"
        assert result["solver_attempted"] is True
        assert result["solver_outcome"] == "timeout"
        assert result["next_action"] == "notify_user"
        assert result["solver_confidence"] == "low"


# ── 5. CF interstitial — placeholder for §11.3 ────────────────────────────


class TestCfInterstitial:
    @pytest.mark.asyncio
    async def test_cf_interstitial_kind(self):
        mgr = _make_manager()
        # Title prefix matches "Just a moment" — current §11.13 placeholder
        # still routes to cf-interstitial-auto. §11.3 will refine.
        inst = _make_inst(
            'iframe[src*="challenges.cloudflare.com"]',
            title="Just a moment...",
        )
        result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["kind"] == "cf-interstitial-auto"
        assert result["solver_outcome"] == "no_solver"
        assert result["next_action"] == "notify_user"

    @pytest.mark.asyncio
    async def test_cf_interstitial_no_title_match(self):
        mgr = _make_manager()
        inst = _make_inst(
            'iframe[src*="challenges.cloudflare.com"]',
            title="Some unrelated page",
        )
        result = await mgr._check_captcha(inst)
        # Until §11.3 lands tri-state, we still classify as the auto kind.
        assert result["kind"] == "cf-interstitial-auto"


# ── 6. Turnstile ──────────────────────────────────────────────────────────


class TestTurnstile:
    @pytest.mark.asyncio
    async def test_turnstile_kind(self):
        mgr = _make_manager()
        inst = _make_inst('[class*="cf-turnstile"]')
        result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["kind"] == "turnstile"
        assert result["solver_outcome"] == "no_solver"


# ── 7. hCaptcha ───────────────────────────────────────────────────────────


class TestHCaptcha:
    @pytest.mark.asyncio
    async def test_hcaptcha_kind(self):
        mgr = _make_manager()
        inst = _make_inst('iframe[src*="hcaptcha"]')
        result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["kind"] == "hcaptcha"


# ── 8. Generic / unknown fallback ─────────────────────────────────────────


class TestUnknownFallback:
    @pytest.mark.asyncio
    async def test_generic_iframe_captcha(self):
        mgr = _make_manager()
        inst = _make_inst('iframe[src*="captcha"]')
        result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["kind"] == "unknown"
        # Confidence drops to "low" for unknown when there is no solver,
        # since we cannot vouch for the classification.
        assert result["solver_confidence"] == "low"

    @pytest.mark.asyncio
    async def test_class_captcha(self):
        mgr = _make_manager()
        inst = _make_inst('[class*="captcha"]')
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "unknown"

    @pytest.mark.asyncio
    async def test_id_captcha(self):
        mgr = _make_manager()
        inst = _make_inst("#captcha")
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "unknown"


# ── 9. Backward-compat: old `message` and `type` fields preserved ────────


class TestLegacyFields:
    @pytest.mark.asyncio
    async def test_detect_captcha_carries_legacy_fields(self):
        mgr = _make_manager()
        inst = _make_inst('iframe[src*="hcaptcha"]')
        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.detect_captcha("a1")
        data = result["data"]
        # New shape
        assert data["captcha_found"] is True
        assert data["kind"] == "hcaptcha"
        # Legacy shape (deprecated but present)
        assert data["type"] == "hcaptcha"
        assert "hcaptcha" in data["message"]
        assert "next_action" in data["message"] or data["next_action"] in data["message"]

    @pytest.mark.asyncio
    async def test_detect_captcha_legacy_no_captcha_message(self):
        mgr = _make_manager()
        inst = _make_inst(None)
        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.detect_captcha("a1")
        # Legacy callers still see the old "No CAPTCHA detected" string.
        assert result["data"]["message"] == "No CAPTCHA detected"


# ── 10. JSON serialization round-trip (no enum repr leakage) ─────────────


class TestJsonSerialization:
    @pytest.mark.asyncio
    async def test_envelope_serializes_to_clean_json(self):
        mgr = _make_manager()
        inst = _make_inst('iframe[src*="recaptcha"]')
        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.detect_captcha("a1")
        # Must round-trip through JSON without raising.
        encoded = json.dumps(result)
        decoded = json.loads(encoded)
        assert decoded == result
        # Every enum-style value should be a plain string, not a Python repr.
        data = decoded["data"]
        for field in (
            "kind", "solver_outcome", "solver_confidence", "next_action",
        ):
            assert isinstance(data[field], str)
            assert "<" not in data[field]
            assert ":" not in data[field]
        # Booleans not strings.
        assert isinstance(data["captcha_found"], bool)
        assert isinstance(data["solver_attempted"], bool)
        # Optional null preserved (JSON null → Python None).
        assert data["injection_failure_reason"] is None

    @pytest.mark.asyncio
    async def test_solver_exception_is_rejected_outcome(self):
        """A solver raising arbitrary exception is reported as 'rejected'
        rather than crashing detect_captcha or leaking the exception."""
        solver = AsyncMock()
        solver.solve = AsyncMock(side_effect=RuntimeError("boom"))
        mgr = _make_manager(solver=solver)
        inst = _make_inst('iframe[src*="hcaptcha"]')
        result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["solver_outcome"] == "rejected"
        assert result["next_action"] == "notify_user"
