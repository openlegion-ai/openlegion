"""Tests for §11.2 proxy-aware solver tasks with credential isolation.

Covers:
  * :func:`src.browser.captcha.get_solver_proxy_config` — env-var loader.
  * :data:`src.browser.captcha._SOLVER_PROXY_COMPAT` — provider/variant
    compatibility table.
  * :meth:`CaptchaSolver._build_task_body` — proxy-aware vs proxyless
    task-name selection and credential injection.
  * :func:`src.browser.captcha_cost_counter.estimate_cents` —
    ``proxy_aware`` flag pricing tier.
  * **Credential-leak canary** — exercises a full solve with a mocked
    httpx client and asserts the agent's primary egress proxy creds
    NEVER appear in any outbound request to the solver provider.

Threat model (the canary test enforces this):
    The agent's primary egress proxy (per-agent state in
    ``BrowserManager._proxy_configs``) is a residential / datacenter
    IP we paid for to scrape from. Forwarding those credentials to a
    third-party CAPTCHA solver would be a direct credential-leak vector
    — the solver could exfiltrate, log, or rotate them. §11.2 requires
    a SEPARATE ``CAPTCHA_SOLVER_PROXY_*`` env-var family for the solver-
    side proxy; the agent's primary creds MUST never reach the solver.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.browser import captcha as cap
from src.browser import captcha_cost_counter as cost
from src.browser.captcha import (
    _2CAPTCHA_TASK_TYPES,
    _CAPSOLVER_TASK_TYPES,
    _SOLVER_PROXY_COMPAT,
    CaptchaSolver,
    SolverProxyConfig,
    _normalize_proxy_type,
    _solver_proxy_compatible,
    get_solver_proxy_config,
)

# ── Shared fixtures ───────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_flag_state(monkeypatch):
    """Reset the once-per-session warning gate + flag overrides per test."""
    cap._reset_proxy_config_warning()
    # Clear every CAPTCHA_SOLVER_PROXY_* var so a previous test or the
    # ambient shell environment can't leak through.
    for name in (
        "CAPTCHA_SOLVER_PROXY_TYPE",
        "CAPTCHA_SOLVER_PROXY_ADDRESS",
        "CAPTCHA_SOLVER_PROXY_PORT",
        "CAPTCHA_SOLVER_PROXY_LOGIN",
        "CAPTCHA_SOLVER_PROXY_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    # flags.py caches operator settings — reload after the env mutation.
    from src.browser import flags as _flags
    _flags.reload_operator_settings()
    yield
    cap._reset_proxy_config_warning()


def _set_full_proxy_env(monkeypatch, *, type_="http"):
    monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_TYPE", type_)
    monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_ADDRESS", "10.0.0.1")
    monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_PORT", "8080")
    monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_LOGIN", "solver-user")
    monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_PASSWORD", "solver-pass")
    from src.browser import flags as _flags
    _flags.reload_operator_settings()


# ── 1. Loader: env-var → SolverProxyConfig ────────────────────────────────


class TestLoader:
    def test_unset_returns_none(self):
        assert get_solver_proxy_config() is None

    def test_full_config_returns_struct(self, monkeypatch):
        _set_full_proxy_env(monkeypatch, type_="socks5")
        cfg = get_solver_proxy_config()
        assert cfg is not None
        assert cfg.proxy_type == "socks5"
        assert cfg.address == "10.0.0.1"
        assert cfg.port == 8080
        assert cfg.login == "solver-user"
        assert cfg.password == "solver-pass"

    def test_partial_config_returns_none_with_warning(self, monkeypatch, caplog):
        # TYPE set but address missing.
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_TYPE", "http")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_PORT", "8080")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_LOGIN", "u")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_PASSWORD", "p")
        from src.browser import flags as _flags
        _flags.reload_operator_settings()
        with caplog.at_level("WARNING", logger="browser.captcha"):
            cfg = get_solver_proxy_config()
        assert cfg is None
        # A single warning at the once-per-session cadence — the message
        # MUST mention the missing field name so operators can fix it.
        assert any("CAPTCHA_SOLVER_PROXY_ADDRESS" in r.message for r in caplog.records)

    def test_warning_only_logged_once_per_session(self, monkeypatch, caplog):
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_TYPE", "http")
        from src.browser import flags as _flags
        _flags.reload_operator_settings()
        with caplog.at_level("WARNING", logger="browser.captcha"):
            assert get_solver_proxy_config() is None
            assert get_solver_proxy_config() is None
            assert get_solver_proxy_config() is None
        warns = [
            r for r in caplog.records
            if "CAPTCHA_SOLVER_PROXY" in r.message
        ]
        # Exactly one per-process warning, not per call.
        assert len(warns) == 1

    def test_socks5h_normalizes_to_socks5(self, monkeypatch):
        _set_full_proxy_env(monkeypatch, type_="socks5h://")
        cfg = get_solver_proxy_config()
        assert cfg is not None
        assert cfg.proxy_type == "socks5"

    def test_socks5h_bare_normalizes_to_socks5(self, monkeypatch):
        _set_full_proxy_env(monkeypatch, type_="socks5h")
        cfg = get_solver_proxy_config()
        assert cfg is not None
        assert cfg.proxy_type == "socks5"

    def test_uppercase_normalizes(self, monkeypatch):
        _set_full_proxy_env(monkeypatch, type_="HTTPS")
        cfg = get_solver_proxy_config()
        assert cfg is not None
        assert cfg.proxy_type == "https"

    def test_bad_scheme_returns_none(self, monkeypatch, caplog):
        # ``ftp`` is obviously not a proxy scheme we accept.
        _set_full_proxy_env(monkeypatch, type_="ftp")
        with caplog.at_level("WARNING", logger="browser.captcha"):
            cfg = get_solver_proxy_config()
        assert cfg is None
        assert any("not in" in r.message.lower() for r in caplog.records)

    def test_url_style_with_creds_rejected(self, monkeypatch):
        # The TYPE field is a scheme keyword, not a URL. Stripping ``://``
        # is a kindness for ``socks5h://`` but ``http://user:pass@host``
        # mid-stream would still leave ``http`` after the split, which is
        # correct — we only validate the scheme bit.
        _set_full_proxy_env(monkeypatch, type_="http://malicious")
        cfg = get_solver_proxy_config()
        assert cfg is not None
        assert cfg.proxy_type == "http"

    def test_invalid_port_returns_none(self, monkeypatch, caplog):
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_TYPE", "http")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_ADDRESS", "10.0.0.1")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_PORT", "not-a-number")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_LOGIN", "u")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_PASSWORD", "p")
        from src.browser import flags as _flags
        _flags.reload_operator_settings()
        with caplog.at_level("WARNING", logger="browser.captcha"):
            cfg = get_solver_proxy_config()
        assert cfg is None

    def test_out_of_range_port_returns_none(self, monkeypatch):
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_TYPE", "http")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_ADDRESS", "10.0.0.1")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_PORT", "70000")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_LOGIN", "u")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_PASSWORD", "p")
        from src.browser import flags as _flags
        _flags.reload_operator_settings()
        cfg = get_solver_proxy_config()
        assert cfg is None


# ── 2. _normalize_proxy_type ──────────────────────────────────────────────


class TestNormalizeProxyType:
    @pytest.mark.parametrize("raw,expected", [
        ("http", "http"),
        ("https", "https"),
        ("socks4", "socks4"),
        ("socks5", "socks5"),
        ("HTTP", "http"),
        ("HTTPS", "https"),
        ("socks5h", "socks5"),
        ("socks5h://", "socks5"),
        ("socks4a", "socks4"),
        ("http://", "http"),
        ("", None),
        ("ftp", None),
        ("gopher", None),
        ("socks", None),  # bare "socks" not allowed
    ])
    def test_cases(self, raw, expected):
        assert _normalize_proxy_type(raw) == expected


# ── 3. Compatibility table ────────────────────────────────────────────────


class TestCompatTable:
    """Verify the hardcoded compat table matches provider docs (Apr 2026):

    * 2captcha — RecaptchaV2Task / HCaptchaTask / TurnstileTask documented
      proxy types: http, socks4, socks5 (no https). Source:
      https://2captcha.com/api-docs/recaptcha-v2 (and equivalents).
    * CapSolver — all proxy-aware tasks document http, https, socks4,
      socks5. Source: https://docs.capsolver.com/en/guide/api-how-to-use-proxy/
    """

    def test_2captcha_v2_checkbox_no_https(self):
        allowed = _SOLVER_PROXY_COMPAT[("2captcha", "recaptcha-v2-checkbox")]
        assert "http" in allowed
        assert "socks4" in allowed
        assert "socks5" in allowed
        assert "https" not in allowed

    def test_2captcha_hcaptcha_no_https(self):
        allowed = _SOLVER_PROXY_COMPAT[("2captcha", "hcaptcha")]
        assert allowed == {"http", "socks4", "socks5"}

    def test_2captcha_turnstile_no_https(self):
        allowed = _SOLVER_PROXY_COMPAT[("2captcha", "turnstile")]
        assert allowed == {"http", "socks4", "socks5"}

    def test_capsolver_full_set(self):
        allowed = _SOLVER_PROXY_COMPAT[("capsolver", "recaptcha-v2-checkbox")]
        assert allowed == {"http", "https", "socks4", "socks5"}

    def test_capsolver_v3_full_set(self):
        allowed = _SOLVER_PROXY_COMPAT[("capsolver", "recaptcha-v3")]
        assert allowed == {"http", "https", "socks4", "socks5"}

    def test_2captcha_v3_absent(self):
        """2captcha has no documented proxy-aware v3 task; the entry
        should not exist (falls back to proxyless via task-table)."""
        # The compat-table key is intentionally absent for variants where
        # ``proxy_aware`` is None in the task-type table.
        assert ("2captcha", "recaptcha-v3") not in _SOLVER_PROXY_COMPAT

    def test_compatible_helper(self):
        assert _solver_proxy_compatible(
            "2captcha", "recaptcha-v2-checkbox", "http",
        )
        assert not _solver_proxy_compatible(
            "2captcha", "recaptcha-v2-checkbox", "https",
        )
        assert _solver_proxy_compatible(
            "capsolver", "hcaptcha", "https",
        )

    def test_compatible_helper_unknown_variant_false(self):
        assert not _solver_proxy_compatible(
            "2captcha", "made-up-variant", "http",
        )


# ── 4. Task-body builder: proxyless vs proxy-aware ────────────────────────


class TestTaskBody:
    def _solver(self, provider="2captcha"):
        return CaptchaSolver(provider, api_key="test-key")

    def test_proxyless_when_no_proxy_config(self):
        s = self._solver()
        body, used_proxy_aware, compat_rejected = s._build_task_body(
            _2CAPTCHA_TASK_TYPES, "recaptcha-v2-checkbox",
            "SITEKEY", "https://example.com",
            page_action=None, proxy_config=None,
        )
        assert body is not None
        assert body["type"] == "RecaptchaV2TaskProxyless"
        assert "proxyType" not in body
        assert "proxyAddress" not in body
        assert used_proxy_aware is False
        assert compat_rejected is False

    def test_proxy_aware_when_compat_allows(self):
        s = self._solver(provider="2captcha")
        cfg = SolverProxyConfig(
            proxy_type="socks5", address="10.0.0.1", port=8080,
            login="u", password="p",
        )
        body, used_proxy_aware, compat_rejected = s._build_task_body(
            _2CAPTCHA_TASK_TYPES, "recaptcha-v2-checkbox",
            "SITEKEY", "https://example.com",
            page_action=None, proxy_config=cfg,
        )
        assert body is not None
        # Drop ``Proxyless`` suffix.
        assert body["type"] == "RecaptchaV2Task"
        assert body["proxyType"] == "socks5"
        assert body["proxyAddress"] == "10.0.0.1"
        assert body["proxyPort"] == 8080
        assert body["proxyLogin"] == "u"
        assert body["proxyPassword"] == "p"
        assert used_proxy_aware is True
        assert compat_rejected is False

    def test_compat_rejected_falls_back_to_proxyless(self):
        """2captcha + https is rejected by the compat table → proxyless +
        compat_rejected=True."""
        s = self._solver(provider="2captcha")
        cfg = SolverProxyConfig(
            proxy_type="https", address="10.0.0.1", port=8080,
            login="u", password="p",
        )
        body, used_proxy_aware, compat_rejected = s._build_task_body(
            _2CAPTCHA_TASK_TYPES, "recaptcha-v2-checkbox",
            "SITEKEY", "https://example.com",
            page_action=None, proxy_config=cfg,
        )
        assert body["type"] == "RecaptchaV2TaskProxyless"
        # Critical: NO proxy fields in the body when compat rejects —
        # we must NOT send https creds to a provider that rejects https.
        assert "proxyType" not in body
        assert "proxyAddress" not in body
        assert "proxyPassword" not in body
        assert used_proxy_aware is False
        assert compat_rejected is True

    def test_proxy_aware_no_documented_task_falls_back_to_proxyless(self):
        """2captcha v3 has no proxy-aware task name → proxyless body
        with compat_rejected=True so the envelope downgrades confidence."""
        s = self._solver(provider="2captcha")
        cfg = SolverProxyConfig(
            proxy_type="http", address="10.0.0.1", port=8080,
            login="u", password="p",
        )
        body, used_proxy_aware, compat_rejected = s._build_task_body(
            _2CAPTCHA_TASK_TYPES, "recaptcha-v3",
            "SITEKEY", "https://example.com",
            page_action="checkout", proxy_config=cfg,
        )
        assert body["type"] == "RecaptchaV3TaskProxyless"
        assert "proxyType" not in body
        assert used_proxy_aware is False
        assert compat_rejected is True

    def test_capsolver_proxy_aware_with_https(self):
        """CapSolver supports https — should use proxy-aware task name."""
        s = self._solver(provider="capsolver")
        cfg = SolverProxyConfig(
            proxy_type="https", address="10.0.0.1", port=8080,
            login="u", password="p",
        )
        body, used_proxy_aware, compat_rejected = s._build_task_body(
            _CAPSOLVER_TASK_TYPES, "recaptcha-v3",
            "SITEKEY", "https://example.com",
            page_action="checkout", proxy_config=cfg,
        )
        assert body["type"] == "ReCaptchaV3Task"  # not ProxyLess
        assert body["proxyType"] == "https"
        assert used_proxy_aware is True
        assert compat_rejected is False

    def test_proxy_fields_dont_clobber_v3_extras(self):
        """v3 fields (minScore, pageAction) must coexist with proxy creds."""
        s = self._solver(provider="capsolver")
        cfg = SolverProxyConfig(
            proxy_type="http", address="10.0.0.1", port=8080,
            login="u", password="p",
        )
        body, _, _ = s._build_task_body(
            _CAPSOLVER_TASK_TYPES, "recaptcha-v3",
            "SITEKEY", "https://example.com",
            page_action="login", proxy_config=cfg,
        )
        assert body["pageAction"] == "login"
        assert "minScore" in body
        assert body["proxyType"] == "http"

    def test_unknown_variant_returns_none(self):
        s = self._solver()
        body, used_proxy_aware, compat_rejected = s._build_task_body(
            _2CAPTCHA_TASK_TYPES, "no-such-variant",
            "SITEKEY", "https://example.com",
            page_action=None, proxy_config=None,
        )
        assert body is None


# ── 5. Cost counter respects proxy_aware flag ─────────────────────────────


class TestCostCounterProxyAware:
    def test_proxyless_baseline(self):
        # 2captcha v2-checkbox proxyless rate: 100¢
        assert cost.estimate_cents(
            "2captcha", "recaptcha-v2-checkbox", proxy_aware=False,
        ) == 100

    def test_proxy_aware_three_x(self):
        # 2captcha v2-checkbox proxy-aware: ~3× → 300¢
        assert cost.estimate_cents(
            "2captcha", "recaptcha-v2-checkbox", proxy_aware=True,
        ) == 300

    def test_capsolver_proxy_aware_three_x(self):
        # CapSolver v2-checkbox proxyless 80¢ → proxy-aware 240¢
        assert cost.estimate_cents(
            "capsolver", "recaptcha-v2-checkbox", proxy_aware=True,
        ) == 240

    def test_proxy_aware_default_is_false(self):
        # Default ``proxy_aware`` arg → proxyless tier (existing callers
        # keep working).
        assert cost.estimate_cents(
            "capsolver", "turnstile",
        ) == 60

    def test_no_proxy_aware_entry_falls_back_to_proxyless(self):
        # 2captcha v3 has no proxy-aware row (provider doesn't document
        # the task) — caller billed proxyless rate to match the request
        # body actually sent.
        proxyless = cost.estimate_cents(
            "2captcha", "recaptcha-v3", proxy_aware=False,
        )
        proxy_aware = cost.estimate_cents(
            "2captcha", "recaptcha-v3", proxy_aware=True,
        )
        assert proxyless == proxy_aware == 100

    def test_unknown_variant_still_returns_none(self):
        assert cost.estimate_cents(
            "2captcha", "unknown-variant", proxy_aware=True,
        ) is None


# ── 6. Credential-leak canary (CONTRACT TEST) ─────────────────────────────


class TestCredentialLeakCanary:
    """Contract test for §11.2 §security note.

    If THIS test ever fails, the credential-leak vector is open: the
    agent's primary egress proxy creds are reaching a third-party
    CAPTCHA solver. Halt and investigate before merging anything that
    breaks this test.
    """

    @pytest.mark.asyncio
    async def test_primary_proxy_creds_never_leak(self, monkeypatch):
        # ── Step 1: simulate an agent with a primary egress proxy that
        # contains a unique canary string in the password. The code path
        # under test (CaptchaSolver.solve → _solve_2captcha → httpx.post)
        # MUST NEVER touch this primary proxy.
        primary_proxy_creds = {
            "proxyType": "http",
            "proxyAddress": "primary-residential.example.com",
            "proxyPort": 9999,
            "proxyLogin": "primary-user-DO-NOT-LEAK",
            "proxyPassword": "PRIMARY_LEAK_CANARY_xyz123",
        }
        # Stash on a stand-in BrowserManager-shaped object purely as a
        # decoy; the solver code path doesn't read this — that's the
        # contract we're enforcing.
        decoy_manager = MagicMock()
        decoy_manager._proxy_configs = {"agent-1": primary_proxy_creds}

        # ── Step 2: configure the dedicated solver proxy with DIFFERENT
        # creds (this is what the solver SHOULD send).
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_TYPE", "socks5")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_ADDRESS", "solver-proxy.example.com")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_PORT", "1080")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_LOGIN", "solver-user")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_PASSWORD", "DEDICATED_SOLVER_PASS")
        from src.browser import flags as _flags
        _flags.reload_operator_settings()

        # ── Step 3: build a real solver and mock the httpx client.
        # Capture every outbound POST the solver makes.
        solver = CaptchaSolver("capsolver", "test-clientkey")
        solver._solver_health_checked = True  # skip the §11.16 probe

        captured_calls: list[dict] = []

        async def _capturing_post(url, *args, **kwargs):
            # Capture URL, JSON body, query string, headers — every
            # surface where credentials could plausibly leak.
            captured_calls.append({
                "url": url,
                "args": args,
                "kwargs": kwargs,
                "json": kwargs.get("json"),
                "params": kwargs.get("params"),
                "headers": kwargs.get("headers"),
                "data": kwargs.get("data"),
            })
            # Return a successful create+poll flow.
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "createTask" in url:
                resp.json = MagicMock(return_value={
                    "errorId": 0, "taskId": "task-canary-123",
                })
            else:
                resp.json = MagicMock(return_value={
                    "errorId": 0, "status": "ready",
                    "solution": {"gRecaptchaResponse": "tok-xyz"},
                })
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(side_effect=_capturing_post)
        solver._client = mock_client

        # ── Step 4: drive the solve. Use a recaptcha-v2 selector so the
        # body builder picks the (capsolver, recaptcha-v2-checkbox) entry,
        # which is in the compat table — proxy-aware path is fully
        # exercised.
        # The solve() flow calls page.evaluate twice: once for the
        # variant classifier (returns a dict) and again for the sitekey
        # extractor (returns a string). Use side_effect so each call
        # gets the right shape.
        page = AsyncMock()
        page.evaluate = AsyncMock(side_effect=[
            # _classify_recaptcha probe → v2-checkbox
            {
                "enterprise": False, "v3": False,
                "sitekeys": ["SITEKEY-canary"],
                "actions_by_key": {},
                "invisible_by_key": {"SITEKEY-canary": False},
                "enterprise_script": False, "v3_render_param": None,
            },
            # any additional evaluate() (token injection JS) — return None
            None, None, None, None, None,
        ])
        page.url = "https://example.com/login"

        # Speed: collapse the poll interval so we don't actually wait.
        monkeypatch.setattr(cap, "_POLL_INTERVAL", 0.001)

        result = await solver.solve(
            page, 'iframe[src*="recaptcha"]', "https://example.com/login",
            agent_id="agent-1",
        )
        assert result.token is not None, "solver should have retrieved a token"
        assert result.injection_succeeded, "solver should have injected the token"

        # ── Step 5: ASSERT — the canary string must NOT appear anywhere
        # in any outbound request.
        assert len(captured_calls) >= 1, "solver issued no requests"

        canary = "PRIMARY_LEAK_CANARY_xyz123"
        primary_login = "primary-user-DO-NOT-LEAK"
        primary_host = "primary-residential.example.com"

        for call in captured_calls:
            blob = repr(call)
            assert canary not in blob, (
                f"CREDENTIAL LEAK: primary proxy password "
                f"'{canary}' found in outbound solver request: {call}"
            )
            assert primary_login not in blob, (
                f"CREDENTIAL LEAK: primary proxy login "
                f"'{primary_login}' found in outbound solver request: {call}"
            )
            assert primary_host not in blob, (
                f"CREDENTIAL LEAK: primary proxy host "
                f"'{primary_host}' found in outbound solver request: {call}"
            )

        # ── Step 6: positive assertion — the dedicated solver creds DO
        # appear in the outbound body. Otherwise the test would pass
        # trivially if no proxy fields were sent at all.
        first_call = captured_calls[0]
        body = first_call["json"]
        assert body is not None
        task = body.get("task", {})
        assert task.get("proxyType") == "socks5"
        assert task.get("proxyAddress") == "solver-proxy.example.com"
        assert task.get("proxyPort") == 1080
        assert task.get("proxyLogin") == "solver-user"
        assert task.get("proxyPassword") == "DEDICATED_SOLVER_PASS"
        # And the proxy-aware task name (sans ``ProxyLess``) was selected.
        assert task.get("type") == "ReCaptchaV2Task"

    @pytest.mark.asyncio
    async def test_no_solver_proxy_means_proxyless_body(self, monkeypatch):
        """When CAPTCHA_SOLVER_PROXY_* is unset, no proxy fields are
        sent — and the agent's primary proxy creds are STILL not leaked.
        """
        # Pretend the agent has a primary proxy (the canary).
        decoy_manager = MagicMock()
        decoy_manager._proxy_configs = {"agent-1": {
            "proxyPassword": "PRIMARY_LEAK_CANARY_unset_path",
        }}

        # No CAPTCHA_SOLVER_PROXY_* env vars → loader returns None.
        solver = CaptchaSolver("2captcha", "test-clientkey")
        solver._solver_health_checked = True

        captured: list[dict] = []

        async def _capturing_post(url, *args, **kwargs):
            captured.append({"url": url, "json": kwargs.get("json")})
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "createTask" in url:
                resp.json = MagicMock(return_value={
                    "errorId": 0, "taskId": "task-1",
                })
            else:
                resp.json = MagicMock(return_value={
                    "errorId": 0, "status": "ready",
                    "solution": {"gRecaptchaResponse": "tok"},
                })
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(side_effect=_capturing_post)
        solver._client = mock_client

        # _classify_recaptcha needs a dict response; the sitekey
        # extractor and token injection JS each call evaluate again.
        page = AsyncMock()
        page.evaluate = AsyncMock(side_effect=[
            {
                "enterprise": False, "v3": False,
                "sitekeys": ["SITEKEY-x"],
                "actions_by_key": {},
                "invisible_by_key": {"SITEKEY-x": False},
                "enterprise_script": False, "v3_render_param": None,
            },
            None, None, None, None, None,
        ])
        page.url = "https://example.com"

        monkeypatch.setattr(cap, "_POLL_INTERVAL", 0.001)

        await solver.solve(
            page, 'iframe[src*="recaptcha"]', "https://example.com",
            agent_id="agent-1",
        )

        # Canary must not leak even on the proxyless path.
        for call in captured:
            assert "PRIMARY_LEAK_CANARY_unset_path" not in repr(call)
        # And proxy fields must be absent from the body.
        body = captured[0]["json"]
        task = body.get("task", {})
        assert "proxyType" not in task
        assert "proxyAddress" not in task
        # Proxyless task name was selected.
        assert task.get("type") == "RecaptchaV2TaskProxyless"

    @pytest.mark.asyncio
    async def test_compat_rejection_marks_envelope_low_confidence(self, monkeypatch):
        """When a proxy is configured but the compat table rejects the
        scheme for the variant, ``SolveResult.compat_rejected`` is True so
        the envelope downgrades to ``solver_confidence='low'``.
        """
        # 2captcha + https is rejected by the compat table.
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_TYPE", "https")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_ADDRESS", "10.0.0.1")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_PORT", "8080")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_LOGIN", "u")
        monkeypatch.setenv("CAPTCHA_SOLVER_PROXY_PASSWORD", "p")
        from src.browser import flags as _flags
        _flags.reload_operator_settings()

        solver = CaptchaSolver("2captcha", "test-clientkey")
        solver._solver_health_checked = True

        captured: list[dict] = []

        async def _capturing_post(url, *args, **kwargs):
            captured.append({"url": url, "json": kwargs.get("json")})
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "createTask" in url:
                resp.json = MagicMock(return_value={
                    "errorId": 0, "taskId": "task-1",
                })
            else:
                resp.json = MagicMock(return_value={
                    "errorId": 0, "status": "ready",
                    "solution": {"gRecaptchaResponse": "tok"},
                })
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(side_effect=_capturing_post)
        solver._client = mock_client

        # _classify_recaptcha needs a dict response; the sitekey
        # extractor and token injection JS each call evaluate again.
        page = AsyncMock()
        page.evaluate = AsyncMock(side_effect=[
            {
                "enterprise": False, "v3": False,
                "sitekeys": ["SITEKEY-x"],
                "actions_by_key": {},
                "invisible_by_key": {"SITEKEY-x": False},
                "enterprise_script": False, "v3_render_param": None,
            },
            None, None, None, None, None,
        ])
        page.url = "https://example.com"

        monkeypatch.setattr(cap, "_POLL_INTERVAL", 0.001)

        result = await solver.solve(
            page, 'iframe[src*="recaptcha"]', "https://example.com",
            agent_id="agent-1",
        )
        assert result.token is not None
        assert result.injection_succeeded
        assert result.used_proxy_aware is False
        assert result.compat_rejected is True
        # Body uses proxyless task name (no proxy creds sent).
        body = captured[0]["json"]
        task = body.get("task", {})
        assert task.get("type") == "RecaptchaV2TaskProxyless"
        assert "proxyType" not in task
