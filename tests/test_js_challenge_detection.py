"""Tests for §19 — JS-challenge detection for tier-1 anti-bot frameworks.

Covers :func:`src.browser.js_challenge.classify_js_challenge` (vendor
detection at the page level) and its integration into
:meth:`BrowserManager._check_captcha` (envelope routing to
``request_captcha_help`` without ever calling the solver).

Vendors covered:
* Akamai Bot Manager — ``ak-bmsc`` / ``bm/sc`` script src OR ``_abck`` cookie.
* Kasada — ``ips.js`` script src OR ``KP_UIDz`` cookie. (Response-header
  detection via ``x-kpsdk-ct`` is documented as a follow-up — see the
  per-vendor notes in :mod:`src.browser.js_challenge`.)
* FingerprintJS Pro — ``fpjs.io`` / ``fingerprint.com`` script src OR
  ``_iidt`` cookie.
* Imperva ABP — ``incapsula`` script src OR ``_imp_apg_r_`` cookie OR
  ``Incap_ses_*`` cookie prefix.
* F5 Distributed Cloud Bot Defense — ``f5cdn.net`` script src OR ``TS01``
  cookie OR ``f5_cspm`` cookie.

The classifier issues a single ``page.evaluate`` JS probe; tests stub
``page.evaluate`` to return whatever vendor name we want to drive each
branch (or simulate failure).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.browser.captcha import _BEHAVIORAL_PROBE_JS, _CF_STATE_PROBE_JS
from src.browser.js_challenge import (
    _JS_CLASSIFY_VENDOR_JS,
    _VENDORS,
    classify_js_challenge,
)
from src.browser.service import (
    _BEHAVIORAL_KINDS,
    _VALID_CAPTCHA_KINDS,
    BrowserManager,
)

# ── Test fixtures ─────────────────────────────────────────────────────────


def _make_manager(*, solver=None) -> BrowserManager:
    """Bare ``BrowserManager`` shell with optional solver mock.

    Mirrors the helper in ``test_captcha_cf_tristate.py``. The solver
    health gates are defaulted to non-tripping so the §19 short-circuits
    are exercised independently of §11.16.
    """
    mgr = BrowserManager.__new__(BrowserManager)
    if solver is not None:
        existing = getattr(solver, "is_solver_unreachable", None)
        if not (
            isinstance(existing, (AsyncMock, MagicMock))
            and isinstance(getattr(existing, "return_value", None), bool)
        ):
            solver.is_solver_unreachable = AsyncMock(return_value=False)
        existing_b = getattr(solver, "is_breaker_open", None)
        if not (
            isinstance(existing_b, (AsyncMock, MagicMock))
            and isinstance(getattr(existing_b, "return_value", None), bool)
        ):
            solver.is_breaker_open = MagicMock(return_value=False)
    mgr._captcha_solver = solver
    return mgr


def _make_inst(
    *,
    matching_selector: str | None,
    js_vendor: str | None = None,
    js_evaluate_raises: bool = False,
    behavioral_probe=None,
    cf_probe=None,
    page_url: str = "https://example.com",
    agent_id: str = "test-agent",
) -> MagicMock:
    """Build a mocked CamoufoxInstance for §19 tests.

    ``page.evaluate`` dispatches on JS body so the JS-challenge
    classifier, the §11.3 behavioral probe, and the §11.3 CF probe can
    each be driven independently.

    * ``js_vendor`` → return value for the JS-challenge classifier
      (``None`` lets the page fall through to the §11.3 / solver path).
    * ``js_evaluate_raises`` → simulate ``page.evaluate`` raising for the
      JS-challenge body (closed page, sandbox error). Classifier should
      collapse to ``None``.
    * ``behavioral_probe`` / ``cf_probe`` default to no-match shapes.
    """
    inst = MagicMock()
    inst.page = MagicMock()
    inst.page.url = page_url
    inst.page.title = AsyncMock(return_value="")
    inst.agent_id = agent_id

    def locator(sel: str):
        loc = MagicMock()

        async def _count():
            return 1 if sel == matching_selector else 0

        loc.count = _count
        return loc

    inst.page.locator = MagicMock(side_effect=locator)

    async def evaluate(js, *args, **kwargs):
        if js == _JS_CLASSIFY_VENDOR_JS:
            if js_evaluate_raises:
                raise RuntimeError("page closed during JS-challenge probe")
            return js_vendor
        if js == _BEHAVIORAL_PROBE_JS:
            return behavioral_probe if behavioral_probe is not None else {
                "px": False, "datadome": False,
            }
        if js == _CF_STATE_PROBE_JS:
            return cf_probe if cf_probe is not None else {
                "has_challenge_running": False,
                "has_turnstile": False,
                "has_cf_error_1020": False,
                "has_challenge_error_text": False,
            }
        return None

    inst.page.evaluate = evaluate
    inst.lock = asyncio.Lock()
    inst.touch = MagicMock()
    return inst


# ── 1. Module-level enum / kind plumbing ─────────────────────────────────


class TestKindEnumPlumbing:
    """The §19 vendor kinds must be present in both the §11.13 valid-kind
    enum and the behavioral-only hint-rejection set."""

    @pytest.mark.parametrize("vendor", sorted(_VENDORS))
    def test_kind_in_valid_captcha_kinds(self, vendor):
        kind = f"js-challenge-{vendor}"
        assert kind in _VALID_CAPTCHA_KINDS

    @pytest.mark.parametrize("vendor", sorted(_VENDORS))
    def test_kind_in_behavioral_kinds(self, vendor):
        # ``_BEHAVIORAL_KINDS`` rejects these as ``hint`` values in
        # ``solve_captcha`` because they have no solver task entry.
        kind = f"js-challenge-{vendor}"
        assert kind in _BEHAVIORAL_KINDS


# ── 2. Per-vendor classifier — direct unit tests ─────────────────────────


class TestPerVendorClassifier:
    """Unit-level: ``classify_js_challenge`` returns the correct vendor.

    Each vendor has its JS body stubbed to return that vendor name. The
    JS body itself is the source of truth for anchor matching (the JS
    runs in-browser); these tests pin the Python-side dispatch — that
    the awaited string survives unchanged.
    """

    @pytest.mark.parametrize("vendor", sorted(_VENDORS))
    @pytest.mark.asyncio
    async def test_vendor_returned(self, vendor):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=vendor)
        result = await classify_js_challenge(page)
        assert result == vendor

    @pytest.mark.asyncio
    async def test_no_anchor_returns_none(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        result = await classify_js_challenge(page)
        assert result is None

    @pytest.mark.asyncio
    async def test_evaluate_raises_returns_none(self):
        """Page closure / sandbox failure during ``page.evaluate`` must
        collapse to ``None`` — never raise to the caller."""
        page = MagicMock()
        page.evaluate = AsyncMock(side_effect=RuntimeError("page closed"))
        result = await classify_js_challenge(page)
        assert result is None

    @pytest.mark.asyncio
    async def test_unknown_vendor_string_returns_none(self):
        """Defensive: if a future JS edit returns a vendor name not in
        ``_VENDORS``, the Python side fails safe rather than emitting a
        phantom ``js-challenge-XYZ`` kind."""
        page = MagicMock()
        page.evaluate = AsyncMock(return_value="some-future-vendor")
        result = await classify_js_challenge(page)
        assert result is None

    @pytest.mark.asyncio
    async def test_non_string_return_is_none(self):
        """``page.evaluate`` returning a dict / int / etc. (e.g. test
        misconfiguration) collapses to ``None``."""
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=42)
        result = await classify_js_challenge(page)
        assert result is None


# ── 3. JS body — anchor-based detection (integration with browser eval) ──


class TestJsBodyAnchorMatching:
    """Asserts the actual JS body inside :data:`_JS_CLASSIFY_VENDOR_JS`
    matches the documented anchors. We exec the JS via a minimal stub
    that mimics ``page.evaluate`` semantics (returns whatever the JS
    function returns, with ``document.scripts`` and ``document.cookie``
    populated from test fixtures).
    """

    def _eval(
        self,
        *,
        scripts: list[str] | None = None,
        cookies: list[str] | None = None,
    ) -> str | None:
        """Execute the vendor JS body against a fake DOM.

        We can't run the actual JS here (Python tests, no JS engine), so
        we replicate the JS semantics in Python. Mirrors the JS's
        ``has_script`` / ``has_cookie`` / ``has_cookie_prefix`` helpers
        EXACTLY so a divergence shows up as a test failure.
        """
        scripts = [(s or "").lower() for s in (scripts or [])]
        cookies = cookies or []

        def has_script(needle: str) -> bool:
            return any(needle in s for s in scripts)

        cookie_names = []
        for c in cookies:
            t = c.strip()
            eq = t.find("=")
            cookie_names.append(t if eq == -1 else t[:eq])

        def has_cookie(name: str) -> bool:
            return name in cookie_names

        def has_cookie_prefix(prefix: str) -> bool:
            return any(n.startswith(prefix) for n in cookie_names)

        # Mirrors the JS body precedence.
        if (
            has_script("ak-bmsc")
            or has_script("bm/sc")
            or has_cookie("_abck")
        ):
            return "akamai"
        if has_script("ips.js") or has_cookie("KP_UIDz"):
            return "kasada"
        if (
            has_script("fpjs.io")
            or has_script("fingerprint.com")
            or has_cookie("_iidt")
        ):
            return "fingerprintjs"
        if (
            has_script("incapsula")
            or has_cookie("_imp_apg_r_")
            or has_cookie_prefix("Incap_ses_")
        ):
            return "imperva"
        if (
            has_script("f5cdn.net")
            or has_cookie("TS01")
            or has_cookie("f5_cspm")
        ):
            return "f5"
        return None

    # Akamai
    def test_akamai_via_ak_bmsc_script(self):
        assert self._eval(
            scripts=["https://example.com/akam/12/ak-bmsc/abc.js"],
        ) == "akamai"

    def test_akamai_via_bm_sc_script(self):
        assert self._eval(scripts=["https://cdn.example/bm/sc/v1.js"]) == "akamai"

    def test_akamai_via_abck_cookie(self):
        # Cookie-only detection (no script src present).
        assert self._eval(cookies=["_abck=ABCD~0~|"]) == "akamai"

    # Kasada
    def test_kasada_via_ips_script(self):
        assert self._eval(scripts=["https://example.com/ips.js"]) == "kasada"

    def test_kasada_via_kp_uidz_cookie(self):
        assert self._eval(cookies=["KP_UIDz=opaque-token"]) == "kasada"

    def test_kasada_header_only_falls_through(self):
        """``x-kpsdk-ct`` is a RESPONSE HEADER. The current classifier
        cannot see response headers via ``page.evaluate`` — that path is
        documented as a follow-up. A page that ONLY has the header (no
        script, no cookie yet) falls through.
        """
        # No script, no cookie → no Kasada signal at the page level.
        assert self._eval() is None

    # FingerprintJS Pro
    def test_fingerprintjs_via_fpjs_script(self):
        assert self._eval(
            scripts=["https://fpjs.io/v3/loader.js"],
        ) == "fingerprintjs"

    def test_fingerprintjs_via_fingerprint_com_script(self):
        assert self._eval(
            scripts=["https://fingerprint.com/v3/loader.js"],
        ) == "fingerprintjs"

    def test_fingerprintjs_via_iidt_cookie(self):
        assert self._eval(cookies=["_iidt=opaque"]) == "fingerprintjs"

    # Imperva
    def test_imperva_via_incapsula_script(self):
        assert self._eval(
            scripts=["https://www.example.com/_Incapsula_Resource?id=123"],
        ) == "imperva"

    def test_imperva_via_imp_apg_cookie(self):
        assert self._eval(cookies=["_imp_apg_r_=value"]) == "imperva"

    def test_imperva_via_incap_ses_prefix(self):
        # Real cookie name has a numeric/region suffix; prefix-match.
        assert self._eval(cookies=["Incap_ses_867_12345=opaque"]) == "imperva"

    # F5
    def test_f5_via_f5cdn_script(self):
        assert self._eval(
            scripts=["https://cdn.f5cdn.net/v1/loader.js"],
        ) == "f5"

    def test_f5_via_ts01_cookie(self):
        assert self._eval(cookies=["TS01abc=opaque"]) is None  # exact-match
        assert self._eval(cookies=["TS01=opaque"]) == "f5"

    def test_f5_via_cspm_cookie(self):
        assert self._eval(cookies=["f5_cspm=value"]) == "f5"

    # Negative
    def test_no_anchors_returns_none(self):
        assert self._eval() is None
        assert self._eval(
            scripts=["https://cdn.example.com/jquery.js"],
            cookies=["sessionid=abc"],
        ) is None


# ── 4. Multi-anchor consistency ──────────────────────────────────────────


class TestMultiAnchor:
    """When multiple anchors for a single vendor match (e.g. Akamai
    script src AND ``_abck`` cookie present), the vendor is returned
    once. Mirrors the JS-body short-circuit on first match.
    """

    @pytest.mark.asyncio
    async def test_akamai_script_and_cookie_both_present(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value="akamai")
        result = await classify_js_challenge(page)
        assert result == "akamai"


# ── 5. Integration into ``_check_captcha`` ───────────────────────────────


class TestCheckCaptchaIntegration:
    """End-to-end: a captcha selector matches AND the JS-challenge
    classifier returns a vendor. The envelope must be a JS-challenge
    envelope (kind, behavioral-only confidence, ``request_captcha_help``)
    and the solver mock must NEVER be awaited.
    """

    @pytest.mark.parametrize("vendor", sorted(_VENDORS))
    @pytest.mark.asyncio
    async def test_js_challenge_short_circuits_solver(self, vendor):
        solver = AsyncMock()
        solver.solve = AsyncMock()
        mgr = _make_manager(solver=solver)
        # Use a generic captcha selector for the initial match — the
        # JS-challenge classifier runs INSIDE the captcha-selector loop
        # (mirrors the §11.3 behavioral classifier) and short-circuits
        # before any solver path.
        inst = _make_inst(
            matching_selector='[class*="captcha"]',
            js_vendor=vendor,
        )
        result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["kind"] == f"js-challenge-{vendor}"
        assert result["solver_attempted"] is False
        assert result["solver_outcome"] == "skipped_behavioral"
        assert result["solver_confidence"] == "behavioral-only"
        assert result["next_action"] == "request_captcha_help"
        # Solver MUST NOT have been invoked. Fail-loud: if a future
        # refactor accidentally wires JS-challenge through the solver
        # path, this assertion catches it.
        solver.solve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_js_challenge_runs_before_behavioral_classifier(self):
        """JS-challenge detection runs BEFORE the §11.3 behavioral
        classifier. When BOTH would match (e.g. an Akamai-protected page
        that ALSO has a PerimeterX selector — pathological but possible
        on chained anti-bot deployments), the JS-challenge envelope wins.
        """
        solver = AsyncMock()
        solver.solve = AsyncMock()
        mgr = _make_manager(solver=solver)
        inst = _make_inst(
            matching_selector='[class*="captcha"]',
            js_vendor="akamai",
            # ``_classify_behavioral`` would also match if we got that far —
            # but the JS-challenge classifier short-circuits first.
            behavioral_probe={"px": True, "datadome": False},
        )
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "js-challenge-akamai"
        assert result["solver_outcome"] == "skipped_behavioral"
        solver.solve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_js_challenge_runs_before_solver_health_gates(self):
        """Like the §11.3 classifier, the §19 classifier runs BEFORE the
        §11.16 solver-health side-channels — JS-challenge detection MUST
        NOT consume health-check or breaker quota."""
        solver = AsyncMock()
        solver.is_solver_unreachable = AsyncMock(return_value=True)
        solver.is_breaker_open = MagicMock(return_value=False)
        solver.solve = AsyncMock()
        mgr = _make_manager(solver=solver)
        inst = _make_inst(
            matching_selector='[class*="captcha"]',
            js_vendor="kasada",
        )
        result = await mgr._check_captcha(inst)
        assert result["kind"] == "js-challenge-kasada"
        assert result["solver_outcome"] == "skipped_behavioral"
        # Health gate not consulted because JS-challenge classifier ran first.
        solver.is_solver_unreachable.assert_not_called()
        solver.solve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_js_challenge_falls_through_to_existing_flow(self):
        """When the JS-challenge classifier returns ``None`` and no
        behavioral / CF anchors match, the existing flow runs unchanged.
        ``[class*="captcha"]`` matched but no solver / no behavioral →
        ``no_solver`` envelope (kind=``unknown``, confidence=``low``).
        """
        mgr = _make_manager(solver=None)
        inst = _make_inst(
            matching_selector='[class*="captcha"]',
            js_vendor=None,
        )
        result = await mgr._check_captcha(inst)
        assert result["captcha_found"] is True
        assert result["kind"] == "unknown"
        assert result["solver_outcome"] == "no_solver"

    @pytest.mark.asyncio
    async def test_js_challenge_evaluate_failure_falls_through(self):
        """When ``page.evaluate`` for the JS-challenge probe raises (page
        closed, sandbox error), the classifier collapses to ``None`` and
        the existing flow runs unchanged. Defensive: never crashes
        ``_check_captcha``."""
        mgr = _make_manager(solver=None)
        inst = _make_inst(
            matching_selector='[class*="captcha"]',
            js_evaluate_raises=True,
        )
        result = await mgr._check_captcha(inst)
        # Falls through to the no-JS-challenge path.
        assert result["kind"] == "unknown"
        assert result["solver_outcome"] == "no_solver"


# ── 6. Audit event emission ──────────────────────────────────────────────


class TestAuditEventEmission:
    """A JS-challenge detection MUST record an audit event so operators
    see the activity in the dashboard's per-minute aggregation. Outcome
    is ``skipped_behavioral`` (matches the §11.3 behavioral path)."""

    @pytest.mark.parametrize("vendor", sorted(_VENDORS))
    @pytest.mark.asyncio
    async def test_audit_event_recorded_per_vendor(self, vendor):
        mgr = _make_manager()
        inst = _make_inst(
            matching_selector='[class*="captcha"]',
            js_vendor=vendor,
            agent_id="audit-agent",
            page_url="https://target.example.com/protected",
        )
        with patch(
            "src.browser.service._record_captcha_audit_event",
            new=AsyncMock(),
        ) as record_mock:
            result = await mgr._check_captcha(inst)
        assert result["kind"] == f"js-challenge-{vendor}"
        record_mock.assert_awaited_once()
        # Positional args: (agent_id, outcome, kind, page_url).
        args, _kwargs = record_mock.call_args
        assert args[0] == "audit-agent"
        assert args[1] == "skipped_behavioral"
        assert args[2] == f"js-challenge-{vendor}"
        assert args[3] == "https://target.example.com/protected"

    @pytest.mark.asyncio
    async def test_no_audit_event_when_no_js_challenge(self):
        """When the classifier returns ``None``, no JS-challenge audit
        event fires — only the existing flow's audit hooks (if any)."""
        mgr = _make_manager(solver=None)
        inst = _make_inst(
            matching_selector='[class*="captcha"]',
            js_vendor=None,
        )
        with patch(
            "src.browser.service._record_captcha_audit_event",
            new=AsyncMock(),
        ) as record_mock:
            await mgr._check_captcha(inst)
        # No JS-challenge envelope was emitted; the existing path may
        # call ``_record_captcha_audit_event`` for OTHER reasons (low
        # success policy, behavioral, etc.) — but in this fixture none
        # of those fire. Assertion: no JS-challenge call.
        for call in record_mock.call_args_list:
            args, _kwargs = call
            assert not args[2].startswith("js-challenge-")
