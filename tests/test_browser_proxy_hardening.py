"""Tests for the browser proxy-hardening trio.

Covers three independent knobs added on top of the always-on binding
signature (see ``test_binding_cookie_coherence.py``):

  1. ``BROWSER_REQUIRE_PROXY`` (default off) — fail-closed egress control.
     ``BrowserManager._enforce_require_proxy`` raises ``ProxyRequiredError``
     when the flag is on and no proxy resolved, so the browser never
     egresses from the datacenter IP. Proxy present OR flag off → proceeds.

  2. ``BROWSER_PROXY_EGRESS_IP_CHECK`` (default off) — fold the REAL egress
     IP (probed through the proxy) into the binding signature so a rotating
     residential pool (one URL, changing IP) invalidates bound anti-bot
     cookies. Probe failure / timeout / flag-off falls back to the UA+URL
     signature and never raises.

  3. No-proxy locale/TZ coherence WARNING — a one-time-per-agent diagnostic
     (no behavior change) when GeoIP is off and a coarse non-US-locale vs
     ``America/*`` container-TZ mismatch is detected.

``_start_browser`` itself imports camoufox (absent in the test env), so the
gate + probe + warning are exercised through their extracted helpers,
mirroring the helper-level style of ``test_binding_cookie_coherence.py``.
"""

from __future__ import annotations

import logging
import tempfile

import httpx
import pytest

from src.browser import flags
from src.browser import session_persistence as sp
from src.browser.service import BrowserManager, ProxyRequiredError, _binding_signatures


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Clean flag layers + binding-signature state + the new flags' env."""
    # Operator-settings layer: point at a nonexistent file so a real
    # config/settings.json can't leak BROWSER_* flags into these tests.
    monkeypatch.setenv("OPENLEGION_SETTINGS_PATH", str(tmp_path / "settings.json"))
    saved_agents = dict(flags._agent_overrides)
    flags._agent_overrides.clear()
    flags.reload_operator_settings()
    # Durable binding-signature sidecar → tmp; clear the in-memory global.
    monkeypatch.setenv("FINGERPRINT_STATE_PATH", str(tmp_path / "fp_state.json"))
    _binding_signatures.clear()
    # Ensure the three new flags start unset (default posture).
    for name in (
        "BROWSER_REQUIRE_PROXY",
        "BROWSER_PROXY_EGRESS_IP_CHECK",
        "BROWSER_PROXY_EGRESS_IP_ECHO_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    flags._agent_overrides.clear()
    flags._agent_overrides.update(saved_agents)
    flags.reload_operator_settings()
    _binding_signatures.clear()


def _make_manager() -> BrowserManager:
    root = tempfile.mkdtemp(prefix="ol_proxyhard_")
    return BrowserManager(profiles_dir=root)


# ── httpx fakes ─────────────────────────────────────────────────────────────


class _FakeResp:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeAsyncClient:
    """Records constructor kwargs + the GET url; returns a canned IP."""

    last_proxy: str | None = None
    last_timeout: object = None
    last_url: str | None = None
    ctor_calls: int = 0
    get_calls: int = 0
    reply_text: str = "203.0.113.7"

    def __init__(self, *, proxy=None, timeout=None):
        type(self).last_proxy = proxy
        type(self).last_timeout = timeout
        type(self).ctor_calls += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        type(self).last_url = url
        type(self).get_calls += 1
        return _FakeResp(type(self).reply_text)


def _install_client(monkeypatch, cls=_FakeAsyncClient):
    cls.ctor_calls = 0
    cls.get_calls = 0
    monkeypatch.setattr("httpx.AsyncClient", cls)
    return cls


def _mismatch_warned(caplog) -> bool:
    """True if the locale/timezone-mismatch WARNING was emitted.

    Asserting on the specific message (not ``caplog.records`` emptiness)
    keeps the no-warning tests robust to unrelated log noise — e.g. a
    Playwright-less environment where ``BrowserManager`` construction logs
    a CRITICAL about the artifact-stream API.
    """
    return any("locale/timezone mismatch" in r.getMessage() for r in caplog.records)


# ── 1. BROWSER_REQUIRE_PROXY (fail-closed) ──────────────────────────────────


class TestRequireProxy:
    def test_flag_on_no_proxy_refuses(self, monkeypatch):
        mgr = _make_manager()
        monkeypatch.setenv("BROWSER_REQUIRE_PROXY", "true")
        with pytest.raises(ProxyRequiredError) as exc:
            mgr._enforce_require_proxy("agent-refuse", None)
        assert exc.value.agent_id == "agent-refuse"
        assert exc.value.retry_after_s > 0
        assert "agent-refuse" in str(exc.value)

    def test_flag_on_proxy_present_allows(self, monkeypatch):
        mgr = _make_manager()
        monkeypatch.setenv("BROWSER_REQUIRE_PROXY", "true")
        # A resolved proxy → never refused, even with the flag on.
        mgr._enforce_require_proxy("agent-ok", {"server": "http://proxy:8080"})

    def test_flag_off_no_proxy_allows(self, monkeypatch):
        mgr = _make_manager()
        # Default off → historical fail-open behavior preserved.
        mgr._enforce_require_proxy("agent-open", None)

    def test_per_agent_override_gates_independently(self, monkeypatch):
        mgr = _make_manager()
        # Operator-wide off, but one agent opted in via the override layer.
        flags.set_agent_override("agent-strict", "BROWSER_REQUIRE_PROXY", "true")
        mgr._enforce_require_proxy("agent-loose", None)  # unaffected → no raise
        with pytest.raises(ProxyRequiredError):
            mgr._enforce_require_proxy("agent-strict", None)


# ── 2. BROWSER_PROXY_EGRESS_IP_CHECK (fold egress IP into signature) ─────────


class TestEgressIpProbe:
    @pytest.mark.asyncio
    async def test_probe_folds_egress_ip_into_signature(self, monkeypatch):
        mgr = _make_manager()
        agent = "agent-egress"
        monkeypatch.setenv("BROWSER_PROXY_EGRESS_IP_CHECK", "true")
        mgr.set_proxy_config(agent, {"server": "http://proxy:8080"})
        client = _install_client(monkeypatch)

        url_only = sp._hash_signature(mgr._build_session_binding_signature(agent))
        await mgr._probe_egress_ip(agent, {"server": "http://proxy:8080"})

        assert mgr._egress_ips[agent] == "203.0.113.7"
        assert client.get_calls == 1
        assert client.last_timeout == 5.0
        assert client.last_url == "https://api.ipify.org"  # default echo url

        sig = mgr._build_session_binding_signature(agent)
        assert sig.get("egress_ip") == "203.0.113.7"
        with_ip = sp._hash_signature(sig)
        assert with_ip != url_only  # egress IP changes the hash

    @pytest.mark.asyncio
    async def test_probe_injects_credentials_into_proxy_url(self, monkeypatch):
        mgr = _make_manager()
        agent = "agent-creds"
        monkeypatch.setenv("BROWSER_PROXY_EGRESS_IP_CHECK", "true")
        mgr.set_proxy_config(agent, {"server": "http://proxy:8080"})
        client = _install_client(monkeypatch)

        await mgr._probe_egress_ip(
            agent,
            {"server": "http://proxy:8080", "username": "u", "password": "p@ss"},
        )
        # Creds live in separate keys of the playwright-style dict; the probe
        # re-assembles a percent-encoded httpx proxy URL.
        assert client.last_proxy == "http://u:p%40ss@proxy:8080"

    @pytest.mark.asyncio
    async def test_custom_echo_url_used(self, monkeypatch):
        mgr = _make_manager()
        agent = "agent-echo"
        monkeypatch.setenv("BROWSER_PROXY_EGRESS_IP_CHECK", "true")
        monkeypatch.setenv("BROWSER_PROXY_EGRESS_IP_ECHO_URL", "https://echo.example/ip")
        mgr.set_proxy_config(agent, {"server": "http://proxy:8080"})
        client = _install_client(monkeypatch)

        await mgr._probe_egress_ip(agent, {"server": "http://proxy:8080"})
        assert client.last_url == "https://echo.example/ip"

    @pytest.mark.asyncio
    async def test_probe_failure_falls_back_no_exception(self, monkeypatch):
        mgr = _make_manager()
        agent = "agent-boom"
        monkeypatch.setenv("BROWSER_PROXY_EGRESS_IP_CHECK", "true")
        mgr.set_proxy_config(agent, {"server": "http://proxy:8080"})

        class _BoomClient(_FakeAsyncClient):
            async def get(self, url):
                raise RuntimeError("proxy dead")

        _install_client(monkeypatch, _BoomClient)
        url_only = sp._hash_signature(mgr._build_session_binding_signature(agent))

        # Must NOT raise.
        await mgr._probe_egress_ip(agent, {"server": "http://proxy:8080"})

        assert agent not in mgr._egress_ips
        sig = mgr._build_session_binding_signature(agent)
        assert "egress_ip" not in sig
        assert sp._hash_signature(sig) == url_only  # UA+URL fallback

    @pytest.mark.asyncio
    async def test_probe_timeout_falls_back(self, monkeypatch):
        mgr = _make_manager()
        agent = "agent-timeout"
        monkeypatch.setenv("BROWSER_PROXY_EGRESS_IP_CHECK", "true")
        mgr.set_proxy_config(agent, {"server": "http://proxy:8080"})

        class _SlowClient(_FakeAsyncClient):
            async def get(self, url):
                raise httpx.ConnectTimeout("slow")

        _install_client(monkeypatch, _SlowClient)
        await mgr._probe_egress_ip(agent, {"server": "http://proxy:8080"})
        assert agent not in mgr._egress_ips

    @pytest.mark.asyncio
    async def test_junk_response_not_cached(self, monkeypatch):
        mgr = _make_manager()
        agent = "agent-junk"
        monkeypatch.setenv("BROWSER_PROXY_EGRESS_IP_CHECK", "true")
        mgr.set_proxy_config(agent, {"server": "http://proxy:8080"})

        class _JunkClient(_FakeAsyncClient):
            reply_text = "<html>not an ip</html>"

        _install_client(monkeypatch, _JunkClient)
        await mgr._probe_egress_ip(agent, {"server": "http://proxy:8080"})
        # ipaddress.ip_address() rejects the body → nothing cached.
        assert agent not in mgr._egress_ips

    @pytest.mark.asyncio
    async def test_flag_off_no_probe(self, monkeypatch):
        mgr = _make_manager()
        agent = "agent-off"
        # Flag unset (default off).
        mgr.set_proxy_config(agent, {"server": "http://proxy:8080"})
        client = _install_client(monkeypatch)

        url_only = sp._hash_signature(mgr._build_session_binding_signature(agent))
        await mgr._probe_egress_ip(agent, {"server": "http://proxy:8080"})

        assert client.ctor_calls == 0  # no httpx client constructed at all
        assert agent not in mgr._egress_ips
        sig = mgr._build_session_binding_signature(agent)
        assert "egress_ip" not in sig
        assert sp._hash_signature(sig) == url_only

    @pytest.mark.asyncio
    async def test_probe_cached_not_reprobed(self, monkeypatch):
        mgr = _make_manager()
        agent = "agent-cache"
        monkeypatch.setenv("BROWSER_PROXY_EGRESS_IP_CHECK", "true")
        mgr.set_proxy_config(agent, {"server": "http://proxy:8080"})
        client = _install_client(monkeypatch)

        await mgr._probe_egress_ip(agent, {"server": "http://proxy:8080"})
        await mgr._probe_egress_ip(agent, {"server": "http://proxy:8080"})
        assert client.get_calls == 1  # second call short-circuits on cache

    @pytest.mark.asyncio
    async def test_socks_scheme_skipped(self, monkeypatch):
        mgr = _make_manager()
        agent = "agent-socks"
        monkeypatch.setenv("BROWSER_PROXY_EGRESS_IP_CHECK", "true")
        client = _install_client(monkeypatch)
        # SOCKS was removed deliberately; the probe skips non-HTTP schemes.
        await mgr._probe_egress_ip(agent, {"server": "socks5://proxy:1080"})
        assert client.ctor_calls == 0
        assert agent not in mgr._egress_ips

    def test_set_proxy_config_clears_cached_egress(self):
        mgr = _make_manager()
        agent = "agent-rotate"
        mgr._egress_ips[agent] = "198.51.100.1"
        mgr.set_proxy_config(agent, {"server": "http://new:8080"})
        # A re-push means the old egress IP is stale → dropped for re-probe.
        assert agent not in mgr._egress_ips

    @pytest.mark.asyncio
    async def test_rotated_egress_ip_invalidates_bound_cookies(self, monkeypatch):
        """Compose: a rotated egress IP flips the signature, so the always-on
        coherence hook drops bound anti-bot cookies while keeping generics."""
        mgr = _make_manager()
        agent = "agent-compose"
        monkeypatch.setenv("BROWSER_PROXY_EGRESS_IP_CHECK", "true")
        mgr.set_proxy_config(agent, {"server": "http://rot:8080"})

        # Baseline persisted from a prior launch under an OLD egress IP.
        mgr._egress_ips[agent] = "198.51.100.1"
        old_sig = sp._hash_signature(mgr._build_session_binding_signature(agent))
        _binding_signatures[agent] = old_sig
        # Simulate a stop clearing the cache, then a fresh launch probing a
        # ROTATED egress IP behind the same proxy URL.
        del mgr._egress_ips[agent]

        class _RotClient(_FakeAsyncClient):
            reply_text = "198.51.100.9"

        _install_client(monkeypatch, _RotClient)
        await mgr._probe_egress_ip(agent, {"server": "http://rot:8080"})
        assert mgr._egress_ips[agent] == "198.51.100.9"
        assert sp._hash_signature(mgr._build_session_binding_signature(agent)) != old_sig

        ctx = _FakeCookieContext([_cookie("cf_clearance"), _cookie("datadome"), _cookie("session")])
        await mgr._enforce_binding_cookie_coherence(_inst(agent, ctx))
        assert set(ctx.cleared_names) == {"cf_clearance", "datadome"}
        assert "session" not in ctx.cleared_names


# ── 3. No-proxy locale/TZ coherence warning ─────────────────────────────────


class TestLocaleTzWarning:
    def test_mismatch_warns_once(self, monkeypatch, caplog):
        mgr = _make_manager()
        monkeypatch.setenv("BROWSER_LOCALE", "en-GB")
        monkeypatch.setenv("TZ", "America/New_York")

        with caplog.at_level(logging.WARNING, logger="browser.service"):
            mgr._warn_locale_tz_mismatch("agent-tz")
        assert any("locale/timezone mismatch" in r.getMessage() for r in caplog.records)

        caplog.clear()
        mgr._warn_locale_tz_mismatch("agent-tz")  # second call
        assert not caplog.records  # one-time-per-agent

    def test_us_locale_no_warning(self, monkeypatch, caplog):
        mgr = _make_manager()
        monkeypatch.setenv("BROWSER_LOCALE", "en-US")
        monkeypatch.setenv("TZ", "America/New_York")
        with caplog.at_level(logging.WARNING, logger="browser.service"):
            mgr._warn_locale_tz_mismatch("agent-us")
        # Assert the specific mismatch warning is absent (robust to unrelated
        # log noise, e.g. a Playwright-less env's construction warning) and
        # that no per-agent warned-marker was set.
        assert not _mismatch_warned(caplog)
        assert "agent-us" not in mgr._tz_mismatch_warned

    def test_matching_region_tz_no_warning(self, monkeypatch, caplog):
        mgr = _make_manager()
        monkeypatch.setenv("BROWSER_LOCALE", "en-GB")
        monkeypatch.setenv("TZ", "Europe/London")  # region-consistent → no warn
        with caplog.at_level(logging.WARNING, logger="browser.service"):
            mgr._warn_locale_tz_mismatch("agent-uk")
        assert not _mismatch_warned(caplog)
        assert "agent-uk" not in mgr._tz_mismatch_warned

    def test_localeless_tag_no_warning(self, monkeypatch, caplog):
        mgr = _make_manager()
        monkeypatch.setenv("BROWSER_LOCALE", "en")  # no region subtag
        monkeypatch.setenv("TZ", "America/New_York")
        with caplog.at_level(logging.WARNING, logger="browser.service"):
            mgr._warn_locale_tz_mismatch("agent-noregion")
        assert not _mismatch_warned(caplog)
        assert "agent-noregion" not in mgr._tz_mismatch_warned


# ── shared cookie-context fake (mirrors test_binding_cookie_coherence) ───────


def _cookie(name: str) -> dict:
    return {"name": name, "value": "v-" + name, "domain": ".example.com", "path": "/"}


def _inst(agent_id: str, context):
    import types

    return types.SimpleNamespace(agent_id=agent_id, context=context)


class _FakeCookieContext:
    def __init__(self, cookies: list[dict]):
        self._cookies = cookies
        self.cleared_names: list[str] = []
        self.cleared_all = 0
        self.readded: list[dict] | None = None

    async def cookies(self) -> list[dict]:
        return list(self._cookies)

    async def clear_cookies(self, name: str | None = None) -> None:
        if name is None:
            self.cleared_all += 1
        else:
            self.cleared_names.append(name)

    async def add_cookies(self, cookies: list[dict]) -> None:
        self.readded = list(cookies)
