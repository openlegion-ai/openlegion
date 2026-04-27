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

from src.browser.captcha import SolveResult
from src.browser.service import BrowserManager


def _solved_result(*, used_proxy_aware: bool = False, compat_rejected: bool = False) -> SolveResult:
    return SolveResult(
        token="tok",
        injection_succeeded=True,
        used_proxy_aware=used_proxy_aware,
        compat_rejected=compat_rejected,
    )


def _failed_result() -> SolveResult:
    """Solver returned no token (sitekey-fail / verdict-reject path)."""
    return SolveResult(
        token=None,
        injection_succeeded=False,
        used_proxy_aware=False,
        compat_rejected=False,
    )

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
    if solver is not None:
        # §11.16 short-circuit getters. ``is_solver_unreachable`` is async
        # post-refactor (lazy probe); ``is_breaker_open`` stays sync.
        # Default both to non-tripping (False) UNLESS the test set them
        # with an explicit bool return_value before calling us.
        existing = getattr(solver, "is_solver_unreachable", None)
        if not (isinstance(existing, (AsyncMock, MagicMock))
                and isinstance(getattr(existing, "return_value", None), bool)):
            solver.is_solver_unreachable = AsyncMock(return_value=False)
        existing_b = getattr(solver, "is_breaker_open", None)
        if not (isinstance(existing_b, (AsyncMock, MagicMock))
                and isinstance(getattr(existing_b, "return_value", None), bool)):
            solver.is_breaker_open = MagicMock(return_value=False)
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
        # ``recaptcha-v2-checkbox`` is a §11.1 placeholder until reCAPTCHA
        # variant disambiguation lands; we cannot vouch for the exact
        # subtype yet, so confidence is "low".
        assert result == {
            "captcha_found": True,
            "kind": "recaptcha-v2-checkbox",
            "solver_attempted": False,
            "solver_outcome": "no_solver",
            "injection_failure_reason": None,
            "solver_confidence": "low",
            "next_action": "notify_user",
        }


# ── 3. reCAPTCHA detect, solver succeeds ──────────────────────────────────


class TestRecaptchaSolverSuccess:
    @pytest.mark.asyncio
    async def test_recaptcha_solved(self):
        solver = AsyncMock()
        solver.solve = AsyncMock(return_value=_solved_result())
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


# ── 11. Httpx-specific exceptions map to 'timeout' not 'rejected' ────────


class TestHttpxTimeoutMapping:
    """Concern #1 (third-pass): a solver that lets an httpx.TimeoutException
    bubble out should be reported as ``timeout``, not ``rejected``. The
    bundled CaptchaSolver catches its own httpx errors and returns False,
    but third-party subclasses may not — verify the dispatch is correct.
    """

    @pytest.mark.asyncio
    async def test_httpx_timeout_exception_reports_timeout(self):
        import httpx

        solver = AsyncMock()
        solver.solve = AsyncMock(
            side_effect=httpx.ReadTimeout("upstream slow"),
        )
        mgr = _make_manager(solver=solver)
        inst = _make_inst('iframe[src*="recaptcha"]')
        result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["solver_attempted"] is True
        assert result["solver_outcome"] == "timeout"
        assert result["solver_confidence"] == "low"
        assert result["next_action"] == "notify_user"

    @pytest.mark.asyncio
    async def test_httpx_connect_error_reports_rejected(self):
        """Non-timeout httpx errors (connect refused, DNS fail) currently
        map to 'rejected' — closest fit in the §11.13 enum, until a richer
        SolverResult lands. Pin the behavior so a future change is
        deliberate.
        """
        import httpx

        solver = AsyncMock()
        solver.solve = AsyncMock(
            side_effect=httpx.ConnectError("refused"),
        )
        mgr = _make_manager(solver=solver)
        inst = _make_inst('iframe[src*="hcaptcha"]')
        result = await mgr._check_captcha(inst)
        assert result["solver_outcome"] == "rejected"

    @pytest.mark.asyncio
    async def test_asyncio_timeout_still_reports_timeout(self):
        """asyncio.TimeoutError takes the explicit branch — verify it
        still works alongside the new httpx branch.
        """
        solver = AsyncMock()
        solver.solve = AsyncMock(side_effect=asyncio.TimeoutError())
        mgr = _make_manager(solver=solver)
        inst = _make_inst('iframe[src*="hcaptcha"]')
        result = await mgr._check_captcha(inst)
        assert result["solver_outcome"] == "timeout"


# ── 12. solver_confidence reflects kind-classification firmness ───────────


class TestKindConfidence:
    """Concern #2 (third-pass): no-solver + placeholder kind should not
    return 'high' confidence. Only firmly-classified kinds (hcaptcha,
    turnstile) earn 'high'; reCAPTCHA and CF interstitial placeholders
    are 'low' until §11.1 / §11.3 land.
    """

    @pytest.mark.asyncio
    async def test_hcaptcha_no_solver_high_confidence(self):
        mgr = _make_manager(solver=None)
        inst = _make_inst('iframe[src*="hcaptcha"]')
        result = await mgr._check_captcha(inst)
        assert result["solver_confidence"] == "high"

    @pytest.mark.asyncio
    async def test_turnstile_no_solver_high_confidence(self):
        mgr = _make_manager(solver=None)
        inst = _make_inst('[class*="cf-turnstile"]')
        result = await mgr._check_captcha(inst)
        assert result["solver_confidence"] == "high"

    @pytest.mark.asyncio
    async def test_recaptcha_placeholder_low_confidence(self):
        mgr = _make_manager(solver=None)
        inst = _make_inst('iframe[src*="recaptcha"]')
        result = await mgr._check_captcha(inst)
        # Placeholder kind — §11.1 will refine.
        assert result["solver_confidence"] == "low"

    @pytest.mark.asyncio
    async def test_cf_interstitial_placeholder_low_confidence(self):
        mgr = _make_manager(solver=None)
        inst = _make_inst('iframe[src*="challenges.cloudflare.com"]')
        result = await mgr._check_captcha(inst)
        # Placeholder kind — §11.3 will refine.
        assert result["solver_confidence"] == "low"


# ── 13. Envelope shape is open for additive top-level fields ──────────────


class TestEnvelopeOpenShape:
    """Concern #8 (third-pass): §11.16 will add a top-level ``breaker_open``
    flag alongside ``solver_outcome="timeout"``. Verify the envelope shape
    is open — additional top-level keys survive the legacy-shim and JSON
    round-trip.
    """

    @pytest.mark.asyncio
    async def test_additive_top_level_field_survives_shim(self):
        # Direct shim invocation — simulates §11.16 augmenting the envelope.
        from src.browser.service import _captcha_envelope, _with_legacy_fields

        env = _captcha_envelope(
            kind="hcaptcha", solver_attempted=True,
            solver_outcome="timeout", solver_confidence="low",
            next_action="notify_user",
        )
        env["breaker_open"] = True  # §11.16 future field
        out = _with_legacy_fields(env)
        assert out["breaker_open"] is True
        # Original structured fields preserved.
        assert out["solver_outcome"] == "timeout"
        # Legacy fields populated.
        assert out["type"] == "hcaptcha"
        assert "hcaptcha" in out["message"]
        # JSON round-trip must preserve the additive field.
        decoded = json.loads(json.dumps(out))
        assert decoded["breaker_open"] is True


# ── 14. injection_failure_reason is always present (None when N/A) ────────


class TestInjectionFailureReasonShape:
    """Concern #9 (third-pass): ``injection_failure_reason`` should be
    consistently present (set to None) rather than absent when
    solver_outcome != 'injection_failed'. Stable shape lets downstream
    consumers do ``data['injection_failure_reason']`` without KeyError.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "selector,solver_factory,expected_outcome",
        [
            ('iframe[src*="hcaptcha"]', lambda: None, "no_solver"),
            (
                'iframe[src*="hcaptcha"]',
                lambda: AsyncMock(solve=AsyncMock(return_value=_solved_result())),
                "solved",
            ),
            (
                'iframe[src*="hcaptcha"]',
                lambda: AsyncMock(solve=AsyncMock(return_value=_failed_result())),
                "rejected",
            ),
            (
                'iframe[src*="hcaptcha"]',
                lambda: AsyncMock(
                    solve=AsyncMock(side_effect=asyncio.TimeoutError()),
                ),
                "timeout",
            ),
        ],
    )
    async def test_injection_failure_reason_present_as_none(
        self, selector, solver_factory, expected_outcome,
    ):
        mgr = _make_manager(solver=solver_factory())
        inst = _make_inst(selector)
        result = await mgr._check_captcha(inst)
        assert result["solver_outcome"] == expected_outcome
        # Field present in dict, set to None.
        assert "injection_failure_reason" in result
        assert result["injection_failure_reason"] is None


# ── 15. Solver returning False on injection-fail is conflated to rejected ─


class TestSolverFalseConflation:
    """Concern #10 (third-pass): when the bundled CaptchaSolver returns
    False for an injection failure (token fetched but ``_inject_token``
    returned False), this layer cannot distinguish from a verdict reject.
    Both currently report ``solver_outcome='rejected'``. Pin this so a
    future PR that adds a richer SolverResult does so deliberately.
    """

    @pytest.mark.asyncio
    async def test_solver_no_token_maps_to_rejected(self):
        """Solver returned no token (sitekey-fail / verdict-reject path).

        The post-refactor :class:`SolveResult` distinguishes:
          * ``token=None`` → ``rejected`` (no provider charge).
          * ``token=non-None, injection_succeeded=False`` →
            ``injection_failed`` (provider charged; see
            :class:`TestInjectionFailedDistinct` below).
        """
        solver = AsyncMock()
        solver.solve = AsyncMock(return_value=_failed_result())
        mgr = _make_manager(solver=solver)
        inst = _make_inst('iframe[src*="hcaptcha"]')
        result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["solver_attempted"] is True
        assert result["solver_outcome"] == "rejected"
        assert result["injection_failure_reason"] is None


# ── 16. Concurrency: same-instance calls are serialized via inst.lock ─────


class TestConcurrency:
    """Concern #11 (third-pass): ``detect_captcha`` acquires ``inst.lock``
    before calling ``_check_captcha``. The post-navigate auto-detect path
    runs inside the same lock acquired at the top of ``navigate()``. Two
    parallel ``detect_captcha`` calls on the same agent should serialize.
    """

    @pytest.mark.asyncio
    async def test_two_parallel_detect_captcha_serialize(self):
        # Build a single instance that records lock acquisitions.
        mgr = _make_manager()
        acquisitions: list[int] = []
        max_concurrent = [0]
        active = [0]

        inst = _make_inst('iframe[src*="hcaptcha"]')

        # Replace lock with one that tracks contention.
        real_lock = asyncio.Lock()

        class TrackingLock:
            async def __aenter__(self_inner):
                await real_lock.acquire()
                active[0] += 1
                acquisitions.append(active[0])
                max_concurrent[0] = max(max_concurrent[0], active[0])
                # Hold briefly so a parallel acquirer would race here.
                await asyncio.sleep(0.005)
                return self_inner

            async def __aexit__(self_inner, *exc):
                active[0] -= 1
                real_lock.release()
                return False

        inst.lock = TrackingLock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            results = await asyncio.gather(
                mgr.detect_captcha("a1"),
                mgr.detect_captcha("a1"),
            )
        # Both succeed.
        assert all(r["success"] is True for r in results)
        assert all(
            r["data"]["captcha_found"] is True for r in results
        )
        # Crucial: at most one detect_captcha holds the lock at a time.
        assert max_concurrent[0] == 1, (
            f"detect_captcha did not serialize on inst.lock "
            f"(max concurrent = {max_concurrent[0]})"
        )
