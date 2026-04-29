"""§22 — anti-bot platform task types (CapSolver-only).

Covers:
* Per-provider task-type table membership: ``_CAPSOLVER_TASK_TYPES``
  carries the four anti-bot kinds; ``_2CAPTCHA_TASK_TYPES`` does NOT.
* Per-type timeout defaults (180s for anti-bot kinds).
* ``CaptchaSolver.supports_kind`` semantics.
* ``MultiProviderSolver.supports_kind`` folds across both providers.
* ``MultiProviderSolver._pick_solver`` kind-aware routing: prefers the
  secondary when primary doesn't support an anti-bot kind.
* ``solve()`` for an anti-bot kind:
  - skips DOM sitekey extraction
  - body builder omits ``websiteKey``
  - failures DON'T tick the §11.16 breaker (anti-bot is low-confidence
    by design)
* ``_extract_solution_token`` accepts the anti-bot solution shape
  (cookies / userAgent / sensorData) and returns a JSON-encoded string.
* Cost-counter pricing for anti-bot kinds is registered (proxyless +
  proxy-aware tables).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.browser.captcha import (
    _2CAPTCHA_TASK_TYPES,
    _ANTIBOT_KINDS,
    _CAPSOLVER_TASK_TYPES,
    _SOLVE_TIMEOUT_DEFAULTS_MS,
    CaptchaSolver,
    MultiProviderSolver,
    _extract_solution_token,
)
from src.browser.captcha_cost_counter import (
    PRICING_MILLICENTS,
    PRICING_MILLICENTS_PROXY_AWARE,
)

# ── 1: per-provider table membership ──────────────────────────────────


class TestTaskTypeTables:
    def test_capsolver_has_all_antibot_kinds(self):
        for kind in _ANTIBOT_KINDS:
            assert kind in _CAPSOLVER_TASK_TYPES, (
                f"{kind} missing from _CAPSOLVER_TASK_TYPES"
            )

    def test_2captcha_has_none_of_the_antibot_kinds(self):
        """Regression guard: copy-pasting these into _2CAPTCHA_TASK_TYPES
        would produce ERROR_INVALID_TASK_TYPE on every solve and trip
        the §11.16 breaker for the whole BrowserManager."""
        for kind in _ANTIBOT_KINDS:
            assert kind not in _2CAPTCHA_TASK_TYPES, (
                f"{kind} must NOT be in _2CAPTCHA_TASK_TYPES — "
                f"2Captcha has no equivalent task type"
            )

    def test_antibot_kinds_have_no_proxyless_variant(self):
        """CapSolver does not publish proxyless variants for the
        anti-bot family; ``proxyless: None`` is what the body builder
        keys on to require operator-configured solver-proxy creds."""
        for kind in _ANTIBOT_KINDS:
            entry = _CAPSOLVER_TASK_TYPES[kind]
            assert entry["proxyless"] is None, (
                f"{kind} proxyless should be None"
            )
            assert isinstance(entry["proxy_aware"], str), (
                f"{kind} proxy_aware must be a task-type string"
            )

    def test_antibot_canonical_task_names(self):
        """Snapshot the names we send over the wire so a future drift
        in the table is surfaced explicitly (the actual provider may
        rename — re-verify against CapSolver docs when tests fail)."""
        expected = {
            "js-challenge-akamai":   "AntiAkamaiBMPTask",
            "js-challenge-imperva":  "AntiImpervaTask",
            "js-challenge-kasada":   "AntiKasadaTask",
            "datadome-behavioral":   "DataDomeSliderTask",
        }
        for kind, task_name in expected.items():
            assert _CAPSOLVER_TASK_TYPES[kind]["proxy_aware"] == task_name


# ── 2: per-type timeouts ──────────────────────────────────────────────


class TestPerTypeTimeouts:
    def test_each_antibot_kind_has_180s_default(self):
        for kind in _ANTIBOT_KINDS:
            assert _SOLVE_TIMEOUT_DEFAULTS_MS[kind] == 180_000, (
                f"{kind} should default to 180s — anti-bot solves are "
                f"known-slow, a tighter default surfaces as a flood "
                f"of timeout envelopes on legitimate solves"
            )


# ── 3: supports_kind semantics ────────────────────────────────────────


class TestSupportsKind:
    def test_capsolver_supports_antibot_kinds(self):
        s = CaptchaSolver("capsolver", "fake-key")
        for kind in _ANTIBOT_KINDS:
            assert s.supports_kind(kind) is True

    def test_2captcha_does_not_support_antibot_kinds(self):
        s = CaptchaSolver("2captcha", "fake-key")
        for kind in _ANTIBOT_KINDS:
            assert s.supports_kind(kind) is False

    def test_unknown_provider_returns_false(self):
        s = CaptchaSolver("nopecha", "fake-key")  # not in supported list
        assert s.supports_kind("js-challenge-akamai") is False
        assert s.supports_kind("recaptcha") is False

    def test_none_kind_returns_false(self):
        s = CaptchaSolver("capsolver", "fake-key")
        assert s.supports_kind(None) is False
        assert s.supports_kind("") is False

    def test_standard_kinds_still_supported_on_both(self):
        cap = CaptchaSolver("capsolver", "fake-key")
        two = CaptchaSolver("2captcha", "fake-key")
        for kind in ("recaptcha", "hcaptcha", "turnstile"):
            assert cap.supports_kind(kind) is True
            assert two.supports_kind(kind) is True


# ── 4: MultiProviderSolver fold ───────────────────────────────────────


class TestMultiProviderSupportsKind:
    def test_only_secondary_supports(self):
        primary = CaptchaSolver("2captcha", "k1")
        secondary = CaptchaSolver("capsolver", "k2")
        wrapper = MultiProviderSolver(primary, secondary)
        for kind in _ANTIBOT_KINDS:
            assert wrapper.supports_kind(kind) is True

    def test_only_primary_supports(self):
        primary = CaptchaSolver("capsolver", "k1")
        secondary = CaptchaSolver("2captcha", "k2")
        wrapper = MultiProviderSolver(primary, secondary)
        for kind in _ANTIBOT_KINDS:
            assert wrapper.supports_kind(kind) is True

    def test_neither_supports(self):
        primary = CaptchaSolver("2captcha", "k1")
        secondary = CaptchaSolver("2captcha", "k2")
        wrapper = MultiProviderSolver(primary, secondary)
        for kind in _ANTIBOT_KINDS:
            assert wrapper.supports_kind(kind) is False

    def test_no_secondary(self):
        primary = CaptchaSolver("2captcha", "k1")
        wrapper = MultiProviderSolver(primary, None)
        # 2Captcha doesn't support anti-bot.
        assert wrapper.supports_kind("js-challenge-akamai") is False
        # Standard kinds still supported.
        assert wrapper.supports_kind("recaptcha") is True


# ── 5: kind-aware routing in _pick_solver ─────────────────────────────


class TestKindAwareRouting:
    """When primary is healthy but doesn't support the kind AND the
    secondary does support it, the wrapper must prefer the secondary —
    overrides the existing primary-first preference."""

    @pytest.mark.asyncio
    async def test_routes_to_secondary_for_antibot_when_primary_lacks_support(self):
        primary = CaptchaSolver("2captcha", "k1")
        primary._solver_health_checked = True  # skip probe
        secondary = CaptchaSolver("capsolver", "k2")
        secondary._solver_health_checked = True

        wrapper = MultiProviderSolver(primary, secondary)
        chosen = await wrapper._pick_solver(kind="js-challenge-akamai")
        assert chosen is secondary
        assert wrapper.provider == "capsolver"

    @pytest.mark.asyncio
    async def test_routes_to_primary_for_standard_kind(self):
        primary = CaptchaSolver("capsolver", "k1")
        primary._solver_health_checked = True
        secondary = CaptchaSolver("2captcha", "k2")
        secondary._solver_health_checked = True

        wrapper = MultiProviderSolver(primary, secondary)
        # Both support recaptcha — primary wins.
        chosen = await wrapper._pick_solver(kind="recaptcha")
        assert chosen is primary

    @pytest.mark.asyncio
    async def test_returns_none_when_only_supporter_is_unhealthy(self):
        primary = CaptchaSolver("2captcha", "k1")
        primary._solver_health_checked = True
        secondary = CaptchaSolver("capsolver", "k2")
        secondary._solver_health_checked = True
        secondary._solver_unreachable = True  # only supporter is down

        wrapper = MultiProviderSolver(primary, secondary)
        chosen = await wrapper._pick_solver(kind="js-challenge-akamai")
        assert chosen is None  # surface no-solver shape

    @pytest.mark.asyncio
    async def test_no_kind_argument_uses_primary_first(self):
        """Backward compat: callers that don't pass ``kind`` still get
        the existing primary-first behavior."""
        primary = CaptchaSolver("capsolver", "k1")
        primary._solver_health_checked = True
        secondary = CaptchaSolver("2captcha", "k2")
        secondary._solver_health_checked = True

        wrapper = MultiProviderSolver(primary, secondary)
        chosen = await wrapper._pick_solver()
        assert chosen is primary


# ── 6: anti-bot solve path skips sitekey + skips breaker ──────────────


def _solve_page_with_url(url: str = "https://example.com/"):
    page = MagicMock()
    page.evaluate = AsyncMock()
    page.url = url
    return page


class TestAntibotSolvePath:
    """End-to-end ``CaptchaSolver.solve()`` for an anti-bot kind."""

    @pytest.mark.asyncio
    async def test_skips_sitekey_extraction(self):
        """For anti-bot kinds, ``_extract_sitekey`` should NOT be called —
        these task types don't use a DOM sitekey marker."""
        solver = CaptchaSolver("capsolver", "fake-key")
        solver._solver_health_checked = True

        # Stub the provider HTTP path so the solve completes without a
        # network round-trip; we only care that sitekey extraction was
        # skipped on the way in.
        async def _no_provider_call(*a, **kw):
            return (None, False, False, False)

        with (
            patch.object(solver, "_extract_sitekey", AsyncMock()) as ext_mock,
            patch.object(solver, "_submit_and_poll", _no_provider_call),
        ):
            page = _solve_page_with_url()
            await solver.solve(
                page, "any-selector", "https://protected.site/",
                kind="js-challenge-akamai",
            )

        ext_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_breaker_not_ticked_on_antibot_failure(self):
        """Anti-bot kinds are low-confidence by design: failures must
        NOT tick the §11.16 breaker. Three rejected anti-bot solves in
        5 minutes should NOT lock the whole fleet's solver out of the
        standard CAPTCHA path."""
        solver = CaptchaSolver("capsolver", "fake-key")
        solver._solver_health_checked = True

        # Provider contacted, returned errorId (transient/non-fatal).
        async def _provider_failure(*a, **kw):
            # (token, used_proxy_aware, compat_rejected, provider_contacted)
            return (None, True, False, True)

        with patch.object(solver, "_submit_and_poll", _provider_failure):
            page = _solve_page_with_url()
            for _ in range(5):  # well above the 3-failure trip threshold
                await solver.solve(
                    page, "any-selector", "https://protected.site/",
                    kind="js-challenge-akamai",
                )

        assert solver.is_breaker_open() is False
        assert len(solver._solver_failure_timestamps) == 0

    @pytest.mark.asyncio
    async def test_breaker_still_ticks_on_standard_kind_failure(self):
        """Regression guard: anti-bot skip must not bleed into standard
        kinds. Three transient errors on a recaptcha solve still trip
        the breaker."""
        solver = CaptchaSolver("capsolver", "fake-key")
        solver._solver_health_checked = True

        # Provider contacted, transient error (non-fatal description).
        err_resp = MagicMock()
        err_resp.json = MagicMock(return_value={
            "errorId": 1,
            "errorDescription": "ERROR_NO_SLOT_AVAILABLE",
        })
        err_resp.raise_for_status = MagicMock()

        client = AsyncMock(spec=httpx.AsyncClient)
        client.is_closed = False
        client.post = AsyncMock(return_value=err_resp)
        solver._client = client

        with (
            patch.object(solver, "_extract_sitekey", AsyncMock(return_value="site-key-abc")),
            patch("src.browser.captcha._POLL_INTERVAL", 0.001),
        ):
            page = _solve_page_with_url()
            for _ in range(3):
                await solver.solve(
                    page, 'iframe[src*="recaptcha"]',
                    "https://example.com/",
                    kind="recaptcha-v2-checkbox",
                )

        # Breaker SHOULD be open — standard-kind path is unchanged.
        assert solver.is_breaker_open() is True


# ── 7: body builder omits websiteKey for anti-bot kinds ───────────────


class TestBodyBuilderAntibot:
    def test_websitekey_omitted_for_antibot(self):
        solver = CaptchaSolver("capsolver", "fake-key")
        body, used_proxy_aware, compat_rejected = solver._build_task_body(
            _CAPSOLVER_TASK_TYPES,
            "js-challenge-akamai",
            "",  # sitekey — empty for anti-bot
            "https://protected.site/",
            page_action=None,
            proxy_config=None,
            provider_name="capsolver",
        )
        if body is not None:
            # When solver-proxy is unconfigured AND the kind has no
            # proxyless variant, the body builder may return None (the
            # fallback path). Either is acceptable — but if a body is
            # returned, websiteKey must NOT be present.
            assert "websiteKey" not in body, (
                "Anti-bot tasks reject ERROR_KEY_MUST_NOT_BE_EMPTY when "
                "websiteKey is included as an empty string"
            )

    def test_websitekey_present_for_standard_kind(self):
        solver = CaptchaSolver("capsolver", "fake-key")
        body, _, _ = solver._build_task_body(
            _CAPSOLVER_TASK_TYPES,
            "recaptcha-v2-checkbox",
            "site-key-xyz",
            "https://example.com/",
            page_action=None,
            proxy_config=None,
            provider_name="capsolver",
        )
        assert body is not None
        assert body.get("websiteKey") == "site-key-xyz"


# ── 8: _extract_solution_token ────────────────────────────────────────


class TestExtractSolutionToken:
    def test_standard_grecaptcha_response_path(self):
        token = _extract_solution_token(
            {"gRecaptchaResponse": "abc"}, "recaptcha",
        )
        assert token == "abc"

    def test_standard_token_field(self):
        token = _extract_solution_token(
            {"token": "xyz"}, "turnstile",
        )
        assert token == "xyz"

    def test_antibot_solution_with_cookies(self):
        sol = {"cookies": [{"name": "_abck", "value": "..."}]}
        token = _extract_solution_token(sol, "js-challenge-akamai")
        assert token is not None
        # Should be JSON-encoded so the caller's "token retrieved"
        # accounting fires; the injection step will gracefully fail
        # downstream because we don't yet thread cookies into the page.
        decoded = json.loads(token)
        assert decoded == sol

    def test_antibot_solution_with_useragent_and_sensor(self):
        sol = {
            "userAgent": "Mozilla/5.0 ...",
            "sensorData": "..."
        }
        token = _extract_solution_token(sol, "js-challenge-imperva")
        assert token is not None
        decoded = json.loads(token)
        assert decoded == sol

    def test_antibot_kind_without_solution_keys_returns_none(self):
        """Empty / unrecognized solution shape → None, even for
        anti-bot kinds. The provider didn't actually return a usable
        solution, so the caller should treat as failure."""
        token = _extract_solution_token({}, "js-challenge-akamai")
        assert token is None

    def test_non_antibot_kind_with_antibot_shape_returns_none(self):
        """A solution that LOOKS like an anti-bot response (cookies)
        but the kind is recaptcha → don't synthesize a fake token.
        Standard fields take priority."""
        sol = {"cookies": [...]}
        token = _extract_solution_token(sol, "recaptcha")
        assert token is None

    def test_non_dict_solution(self):
        assert _extract_solution_token(None, "js-challenge-akamai") is None
        assert _extract_solution_token("string", "js-challenge-akamai") is None


# ── 9: cost-counter pricing ───────────────────────────────────────────


class TestPricing:
    def test_proxyless_pricing_registered(self):
        """Cost-cap reservation reads the proxyless table BEFORE the
        actual tier is known. Anti-bot kinds must have an entry so the
        pre-solve reservation gate doesn't fail closed."""
        for kind in _ANTIBOT_KINDS:
            key = f"capsolver-{kind}"
            assert key in PRICING_MILLICENTS, (
                f"{key} missing from PRICING_MILLICENTS — cost-cap "
                f"reservation will fail closed"
            )
            assert PRICING_MILLICENTS[key] > 0

    def test_proxy_aware_pricing_registered(self):
        for kind in _ANTIBOT_KINDS:
            key = ("capsolver", kind)
            assert key in PRICING_MILLICENTS_PROXY_AWARE, (
                f"{key} missing from PRICING_MILLICENTS_PROXY_AWARE"
            )
            assert PRICING_MILLICENTS_PROXY_AWARE[key] > 0

    def test_no_2captcha_antibot_pricing(self):
        """Regression guard: 2Captcha doesn't support anti-bot kinds,
        and adding entries to the pricing table would suggest the
        solver path is wired (it isn't)."""
        for kind in _ANTIBOT_KINDS:
            assert f"2captcha-{kind}" not in PRICING_MILLICENTS
            assert ("2captcha", kind) not in PRICING_MILLICENTS_PROXY_AWARE
