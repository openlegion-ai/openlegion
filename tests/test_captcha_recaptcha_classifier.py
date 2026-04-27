"""Tests for §11.1 reCAPTCHA variant classifier and provider task tables.

Covers :func:`src.browser.captcha._classify_recaptcha` and the structured
``_2CAPTCHA_TASK_TYPES`` / ``_CAPSOLVER_TASK_TYPES`` lookups added in §11.1.

The classifier is a single ``page.evaluate`` JS probe that returns a
structured dict; tests stub the eval to return whatever shape we need
to drive each branch (enterprise namespace, v3 markers, invisible v2,
etc.). No real browser involved.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.browser.captcha import (
    _2CAPTCHA_TASK_TYPES,
    _CAPSOLVER_TASK_TYPES,
    _DEFAULT_V3_ACTION,
    _DEFAULT_V3_MIN_SCORE,
    CaptchaSolver,
    _classify_recaptcha,
)


def _make_page(probe_result):
    """Return a MagicMock page whose ``evaluate`` returns ``probe_result``.

    ``probe_result`` is whatever the JS classifier would emit (a dict),
    or an ``Exception`` instance to simulate a thrown ``page.evaluate``.
    """
    page = AsyncMock()
    if isinstance(probe_result, BaseException):
        page.evaluate = AsyncMock(side_effect=probe_result)
    else:
        page.evaluate = AsyncMock(return_value=probe_result)
    return page


# ── 1. Each variant — happy path ──────────────────────────────────────────


class TestVariantHappyPath:
    @pytest.mark.asyncio
    async def test_v2_checkbox(self):
        page = _make_page({
            "enterprise": False,
            "v3": False,
            "sitekeys": ["ABCDEFGHIJK_v2_checkbox"],
            "actions_by_key": {},
            "invisible_by_key": {"ABCDEFGHIJK_v2_checkbox": False},
            "enterprise_script": False,
            "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        assert result["variant"] == "recaptcha-v2-checkbox"
        assert result["sitekey"] == "ABCDEFGHIJK_v2_checkbox"
        assert result["action"] is None
        assert result["min_score"] is None  # operator config; not page-extractable

    @pytest.mark.asyncio
    async def test_v2_invisible(self):
        page = _make_page({
            "enterprise": False,
            "v3": False,
            "sitekeys": ["SITEKEY_invisible_widget"],
            "actions_by_key": {},
            "invisible_by_key": {"SITEKEY_invisible_widget": True},
            "enterprise_script": False,
            "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        assert result["variant"] == "recaptcha-v2-invisible"
        assert result["sitekey"] == "SITEKEY_invisible_widget"

    @pytest.mark.asyncio
    async def test_v3(self):
        page = _make_page({
            "enterprise": False,
            "v3": True,
            "sitekeys": ["V3_SITEKEY_AAAAAAAAAA"],
            "actions_by_key": {"V3_SITEKEY_AAAAAAAAAA": "homepage"},
            "invisible_by_key": {},
            "enterprise_script": False,
            "v3_render_param": "V3_SITEKEY_AAAAAAAAAA",
        })
        result = await _classify_recaptcha(page)
        assert result["variant"] == "recaptcha-v3"
        assert result["sitekey"] == "V3_SITEKEY_AAAAAAAAAA"
        assert result["action"] == "homepage"
        assert result["min_score"] is None

    @pytest.mark.asyncio
    async def test_enterprise_v2(self):
        page = _make_page({
            "enterprise": True,
            "v3": False,
            "sitekeys": ["ENT_V2_SITEKEY_BBBBBBB"],
            "actions_by_key": {},
            "invisible_by_key": {},
            "enterprise_script": True,
            "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        assert result["variant"] == "recaptcha-enterprise-v2"
        assert result["sitekey"] == "ENT_V2_SITEKEY_BBBBBBB"

    @pytest.mark.asyncio
    async def test_enterprise_v3(self):
        page = _make_page({
            "enterprise": True,
            "v3": True,
            "sitekeys": ["ENT_V3_SITEKEY_CCCCCCC"],
            "actions_by_key": {"ENT_V3_SITEKEY_CCCCCCC": "submit"},
            "invisible_by_key": {},
            "enterprise_script": True,
            "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        assert result["variant"] == "recaptcha-enterprise-v3"
        assert result["sitekey"] == "ENT_V3_SITEKEY_CCCCCCC"
        assert result["action"] == "submit"


# ── 2. Enterprise detection paths ─────────────────────────────────────────


class TestEnterpriseDetection:
    @pytest.mark.asyncio
    async def test_enterprise_via_script_src(self):
        """``<script src*="enterprise.recaptcha.net">`` flips enterprise on."""
        page = _make_page({
            "enterprise": True,           # set by JS branch when script tag matches
            "enterprise_script": True,
            "v3": False,
            "sitekeys": ["SK_via_script"],
            "actions_by_key": {},
            "invisible_by_key": {},
            "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        assert result["variant"] == "recaptcha-enterprise-v2"

    @pytest.mark.asyncio
    async def test_enterprise_via_grecaptcha_global(self):
        """``window.grecaptcha.enterprise`` global flips enterprise on."""
        page = _make_page({
            "enterprise": True,
            "enterprise_script": False,   # no script tag — only the global signal
            "v3": False,
            "sitekeys": ["SK_via_global"],
            "actions_by_key": {},
            "invisible_by_key": {},
            "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        assert result["variant"] == "recaptcha-enterprise-v2"

    @pytest.mark.asyncio
    async def test_classifier_consumer_plus_enterprise_picks_enterprise(self):
        """Multi-tenant page with BOTH consumer ``api.js`` and enterprise
        ``enterprise.js`` scripts loaded simultaneously, and BOTH a
        consumer sitekey and an enterprise sitekey present in the registry.

        Per security finding F11: the JS probe sets ``enterprise=True``
        when ANY signal points at enterprise (script tag OR global), and
        the first sitekey from the ``sitekeys`` list wins. This documents
        the current behavior — when the agent is intending to solve the
        consumer widget on a page that ALSO embeds an enterprise widget
        for an unrelated flow, the classifier may misalign with intent.

        Fix is deferred (architectural — requires per-widget targeting,
        §11.6 territory). When the deferred-trigger fires in production
        we'll need either snapshot-ref-based selection or a per-sitekey
        enterprise/consumer disambiguation pass.
        """
        page = _make_page({
            # Both scripts loaded → enterprise wins.
            "enterprise": True,
            "enterprise_script": True,
            "v3": False,
            # Consumer key first, enterprise key second. Current
            # classifier picks ``sitekeys[0]`` — but classifies as
            # enterprise based on the script signal. Misalignment risk.
            "sitekeys": ["CONSUMER_SITEKEY", "ENTERPRISE_SITEKEY"],
            "actions_by_key": {},
            "invisible_by_key": {
                "CONSUMER_SITEKEY": False,
                "ENTERPRISE_SITEKEY": False,
            },
            "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        # Documented current behavior — pin so a future refactor of the
        # classifier surfaces here, not in production.
        assert result["variant"] == "recaptcha-enterprise-v2"
        assert result["sitekey"] == "CONSUMER_SITEKEY"


# ── 3. v3 detection paths ─────────────────────────────────────────────────


class TestV3Detection:
    @pytest.mark.asyncio
    async def test_v3_via_render_param(self):
        """``<script src="...api.js?render=<sitekey>">`` is v3-only."""
        page = _make_page({
            "enterprise": False,
            "v3": True,                    # JS sets this when render param present
            "sitekeys": [],
            "actions_by_key": {},
            "invisible_by_key": {},
            "enterprise_script": False,
            "v3_render_param": "RENDER_PARAM_SITEKEY",
        })
        result = await _classify_recaptcha(page)
        assert result["variant"] == "recaptcha-v3"
        # Sitekey falls back to the render param when registry walk yielded nothing.
        assert result["sitekey"] == "RENDER_PARAM_SITEKEY"

    @pytest.mark.asyncio
    async def test_v3_via_clients_action_field(self):
        """A widget config with ``action: "..."`` field → v3."""
        page = _make_page({
            "enterprise": False,
            "v3": True,
            "sitekeys": ["SK_with_action"],
            "actions_by_key": {"SK_with_action": "checkout"},
            "invisible_by_key": {},
            "enterprise_script": False,
            "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        assert result["variant"] == "recaptcha-v3"
        assert result["action"] == "checkout"


# ── 4. v2-invisible detection paths ───────────────────────────────────────


class TestV2InvisibleDetection:
    @pytest.mark.asyncio
    async def test_invisible_via_data_size_attr(self):
        """``data-size="invisible"`` on the widget div → v2-invisible."""
        page = _make_page({
            "enterprise": False,
            "v3": False,
            "sitekeys": ["SK_data_size"],
            "actions_by_key": {},
            "invisible_by_key": {"SK_data_size": True},
            "enterprise_script": False,
            "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        assert result["variant"] == "recaptcha-v2-invisible"

    @pytest.mark.asyncio
    async def test_invisible_via_clients_size_field(self):
        """``size: "invisible"`` inside the registry config → v2-invisible."""
        # Same shape as the data-size test — both feed the same map.
        page = _make_page({
            "enterprise": False,
            "v3": False,
            "sitekeys": ["SK_registry_size"],
            "actions_by_key": {},
            "invisible_by_key": {"SK_registry_size": True},
            "enterprise_script": False,
            "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        assert result["variant"] == "recaptcha-v2-invisible"


# ── 5. pageAction extraction ──────────────────────────────────────────────


class TestPageActionExtraction:
    @pytest.mark.asyncio
    async def test_action_extracted_from_registry(self):
        page = _make_page({
            "enterprise": False,
            "v3": True,
            "sitekeys": ["SK_action"],
            "actions_by_key": {"SK_action": "login_form"},
            "invisible_by_key": {},
            "enterprise_script": False,
            "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        assert result["action"] == "login_form"

    @pytest.mark.asyncio
    async def test_action_extraction_failure_returns_none(self):
        """v3 detected via render param but no action in registry → None."""
        page = _make_page({
            "enterprise": False,
            "v3": True,
            "sitekeys": [],
            "actions_by_key": {},
            "invisible_by_key": {},
            "enterprise_script": False,
            "v3_render_param": "JUST_THE_RENDER_KEY",
        })
        result = await _classify_recaptcha(page)
        assert result["variant"] == "recaptcha-v3"
        assert result["action"] is None  # solver caller must default to "verify"


# ── 6. Sitekey fallback chain ─────────────────────────────────────────────


class TestSitekeyFallbackChain:
    @pytest.mark.asyncio
    async def test_sitekey_from_registry(self):
        page = _make_page({
            "enterprise": False, "v3": False,
            "sitekeys": ["FROM_REGISTRY"],
            "actions_by_key": {}, "invisible_by_key": {},
            "enterprise_script": False, "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        assert result["sitekey"] == "FROM_REGISTRY"

    @pytest.mark.asyncio
    async def test_sitekey_from_render_param_when_registry_empty(self):
        page = _make_page({
            "enterprise": False, "v3": True,
            "sitekeys": [],
            "actions_by_key": {}, "invisible_by_key": {},
            "enterprise_script": False,
            "v3_render_param": "FROM_RENDER_PARAM",
        })
        result = await _classify_recaptcha(page)
        assert result["sitekey"] == "FROM_RENDER_PARAM"

    @pytest.mark.asyncio
    async def test_sitekey_none_when_no_signal(self):
        page = _make_page({
            "enterprise": False, "v3": False,
            "sitekeys": [],
            "actions_by_key": {}, "invisible_by_key": {},
            "enterprise_script": False, "v3_render_param": None,
        })
        result = await _classify_recaptcha(page)
        assert result["sitekey"] is None
        assert result["variant"] == "unknown"


# ── 7. Defensive: malformed / missing probe output ────────────────────────


class TestDefensive:
    @pytest.mark.asyncio
    async def test_evaluate_throws_returns_unknown(self):
        page = _make_page(RuntimeError("page closed"))
        result = await _classify_recaptcha(page)
        assert result == {
            "variant": "unknown",
            "sitekey": None,
            "action": None,
            "min_score": None,
        }

    @pytest.mark.asyncio
    async def test_evaluate_returns_non_dict_returns_unknown(self):
        page = _make_page(None)  # JS returned null/undefined
        result = await _classify_recaptcha(page)
        assert result["variant"] == "unknown"

    @pytest.mark.asyncio
    async def test_evaluate_returns_empty_dict(self):
        page = _make_page({})
        result = await _classify_recaptcha(page)
        assert result["variant"] == "unknown"
        assert result["sitekey"] is None


# ── 8. Provider task-type tables ──────────────────────────────────────────


class TestTaskTypeTables:
    """Table lookup must return the right provider-side ``type`` plus any
    variant-specific extras (``isInvisible``, ``isEnterprise``).

    Task names verified against the public 2captcha + CapSolver docs in
    April 2026; see the comment block in src/browser/captcha.py for the
    URLs and the drift note (2captcha v3-Enterprise reuses the v3 task
    name with ``isEnterprise: true`` rather than offering a standalone
    ``RecaptchaV3EnterpriseTaskProxyless`` type).
    """

    # §11.2 reshaped each entry to ``{"proxyless": ..., "proxy_aware": ...,
    # "extra": {...}}`` so the body builder can pick a proxy-aware task name
    # when a dedicated solver proxy is configured. The proxyless name (the
    # one that goes to the provider when no proxy config is set) is what
    # these tests assert.

    def test_2captcha_v2_checkbox(self):
        entry = _2CAPTCHA_TASK_TYPES["recaptcha-v2-checkbox"]
        assert entry["proxyless"] == "RecaptchaV2TaskProxyless"
        assert entry["proxy_aware"] == "RecaptchaV2Task"
        assert entry["extra"] == {}

    def test_2captcha_v2_invisible(self):
        entry = _2CAPTCHA_TASK_TYPES["recaptcha-v2-invisible"]
        assert entry["proxyless"] == "RecaptchaV2TaskProxyless"
        assert entry["proxy_aware"] == "RecaptchaV2Task"
        assert entry["extra"] == {"isInvisible": True}

    def test_2captcha_v3(self):
        entry = _2CAPTCHA_TASK_TYPES["recaptcha-v3"]
        assert entry["proxyless"] == "RecaptchaV3TaskProxyless"
        # 2captcha has no documented proxy-aware v3 task as of April 2026.
        assert entry["proxy_aware"] is None

    def test_2captcha_enterprise_v2(self):
        entry = _2CAPTCHA_TASK_TYPES["recaptcha-enterprise-v2"]
        assert entry["proxyless"] == "RecaptchaV2EnterpriseTaskProxyless"
        assert entry["proxy_aware"] == "RecaptchaV2EnterpriseTask"

    def test_2captcha_enterprise_v3_uses_v3_with_flag(self):
        """2captcha drift: no standalone v3-Enterprise task; flag instead."""
        entry = _2CAPTCHA_TASK_TYPES["recaptcha-enterprise-v3"]
        assert entry["proxyless"] == "RecaptchaV3TaskProxyless"
        assert entry["extra"] == {"isEnterprise": True}

    def test_capsolver_v2_checkbox(self):
        entry = _CAPSOLVER_TASK_TYPES["recaptcha-v2-checkbox"]
        assert entry["proxyless"] == "ReCaptchaV2TaskProxyLess"
        assert entry["proxy_aware"] == "ReCaptchaV2Task"
        assert entry["extra"] == {}

    def test_capsolver_v2_invisible(self):
        entry = _CAPSOLVER_TASK_TYPES["recaptcha-v2-invisible"]
        assert entry["proxyless"] == "ReCaptchaV2TaskProxyLess"
        assert entry["proxy_aware"] == "ReCaptchaV2Task"
        assert entry["extra"] == {"isInvisible": True}

    def test_capsolver_v3(self):
        entry = _CAPSOLVER_TASK_TYPES["recaptcha-v3"]
        assert entry["proxyless"] == "ReCaptchaV3TaskProxyLess"
        assert entry["proxy_aware"] == "ReCaptchaV3Task"

    def test_capsolver_enterprise_v2(self):
        entry = _CAPSOLVER_TASK_TYPES["recaptcha-enterprise-v2"]
        assert entry["proxyless"] == "ReCaptchaV2EnterpriseTaskProxyLess"
        assert entry["proxy_aware"] == "ReCaptchaV2EnterpriseTask"

    def test_capsolver_enterprise_v3(self):
        entry = _CAPSOLVER_TASK_TYPES["recaptcha-enterprise-v3"]
        assert entry["proxyless"] == "ReCaptchaV3EnterpriseTaskProxyLess"
        assert entry["proxy_aware"] == "ReCaptchaV3EnterpriseTask"


# ── 9. Task-body merger ───────────────────────────────────────────────────


class TestTaskBodyMerger:
    """``CaptchaSolver._build_task_body`` produces the JSON the provider
    receives. v3 / Enterprise-v3 must include ``minScore`` + ``pageAction``;
    v2 variants must NOT include them; Enterprise-v2 does not gain v3 fields.
    """

    def _solver(self):
        # CaptchaSolver's __init__ does no I/O, so we can build directly.
        return CaptchaSolver(provider="2captcha", api_key="test")

    # §11.2 changed ``_build_task_body`` to return
    # ``(body, used_proxy_aware, compat_rejected)``. These tests cover the
    # no-proxy-config path so the second/third tuple slots are always
    # ``(False, False)``; the proxy-aware path is covered separately in
    # ``tests/test_captcha_solver_proxy.py``.

    def test_v2_checkbox_body_has_no_v3_fields(self):
        s = self._solver()
        body, used_proxy_aware, compat_rejected = s._build_task_body(
            _2CAPTCHA_TASK_TYPES, "recaptcha-v2-checkbox",
            "SITEKEY", "https://example.com",
            page_action=None,
        )
        assert body["type"] == "RecaptchaV2TaskProxyless"
        assert body["websiteURL"] == "https://example.com"
        assert body["websiteKey"] == "SITEKEY"
        assert "minScore" not in body
        assert "pageAction" not in body
        assert used_proxy_aware is False
        assert compat_rejected is False

    def test_v2_invisible_body_has_invisible_flag(self):
        s = self._solver()
        body, _, _ = s._build_task_body(
            _CAPSOLVER_TASK_TYPES, "recaptcha-v2-invisible",
            "SITEKEY", "https://example.com",
            page_action=None,
        )
        assert body["type"] == "ReCaptchaV2TaskProxyLess"
        assert body["isInvisible"] is True
        assert "minScore" not in body

    def test_v3_body_has_min_score_and_action(self):
        s = self._solver()
        body, _, _ = s._build_task_body(
            _2CAPTCHA_TASK_TYPES, "recaptcha-v3",
            "SITEKEY", "https://example.com",
            page_action="checkout",
        )
        assert body["type"] == "RecaptchaV3TaskProxyless"
        assert body["minScore"] == _DEFAULT_V3_MIN_SCORE
        assert body["pageAction"] == "checkout"

    def test_v3_body_action_falls_back_to_verify(self):
        s = self._solver()
        body, _, _ = s._build_task_body(
            _2CAPTCHA_TASK_TYPES, "recaptcha-v3",
            "SITEKEY", "https://example.com",
            page_action=None,
        )
        assert body["pageAction"] == _DEFAULT_V3_ACTION

    def test_enterprise_v3_body_has_isEnterprise_and_v3_extras(self):
        s = self._solver()
        body, _, _ = s._build_task_body(
            _2CAPTCHA_TASK_TYPES, "recaptcha-enterprise-v3",
            "SITEKEY", "https://example.com",
            page_action="submit",
        )
        assert body["type"] == "RecaptchaV3TaskProxyless"
        assert body["isEnterprise"] is True
        assert body["minScore"] == _DEFAULT_V3_MIN_SCORE
        assert body["pageAction"] == "submit"

    def test_enterprise_v2_body_no_v3_fields(self):
        s = self._solver()
        body, _, _ = s._build_task_body(
            _2CAPTCHA_TASK_TYPES, "recaptcha-enterprise-v2",
            "SITEKEY", "https://example.com",
            page_action=None,
        )
        assert body["type"] == "RecaptchaV2EnterpriseTaskProxyless"
        assert "minScore" not in body
        assert "pageAction" not in body

    def test_unknown_variant_returns_none(self):
        s = self._solver()
        body, _, _ = s._build_task_body(
            _2CAPTCHA_TASK_TYPES, "no-such-variant",
            "SITEKEY", "https://example.com",
            page_action=None,
        )
        assert body is None

    def test_min_score_honors_env_var(self, monkeypatch):
        s = self._solver()
        monkeypatch.setenv("CAPTCHA_RECAPTCHA_V3_MIN_SCORE", "0.3")
        # flags loader caches; force re-read.
        from src.browser import flags as _flags
        _flags.reload_operator_settings()
        body, _, _ = s._build_task_body(
            _CAPSOLVER_TASK_TYPES, "recaptcha-v3",
            "SITEKEY", "https://example.com",
            page_action="x",
        )
        assert body["minScore"] == 0.3

    def test_min_score_clamped(self, monkeypatch):
        """Out-of-range values clamp to [0.1, 0.9]."""
        s = self._solver()
        monkeypatch.setenv("CAPTCHA_RECAPTCHA_V3_MIN_SCORE", "5.0")
        from src.browser import flags as _flags
        _flags.reload_operator_settings()
        body, _, _ = s._build_task_body(
            _2CAPTCHA_TASK_TYPES, "recaptcha-v3",
            "SITEKEY", "https://example.com",
            page_action="x",
        )
        assert body["minScore"] == 0.9
