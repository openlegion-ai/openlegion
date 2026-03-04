"""Tests for the shared browser service (BrowserManager, server, redaction)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCredentialRedactor:
    """Tests for browser.redaction.CredentialRedactor."""

    def test_pattern_redaction(self):
        from src.browser.redaction import CredentialRedactor
        r = CredentialRedactor()
        text = "key is sk-abcdefghijklmnopqrstuvwxyz1234567890"
        result = r.redact("agent1", text)
        assert "[REDACTED]" in result
        assert "sk-abcdefgh" not in result

    def test_exact_value_redaction(self):
        from src.browser.redaction import CredentialRedactor
        r = CredentialRedactor()
        r.track_resolved_value("agent1", "supersecretpassword")
        result = r.redact("agent1", "the password is supersecretpassword here")
        assert "supersecretpassword" not in result
        assert "[REDACTED]" in result

    def test_short_values_not_tracked(self):
        from src.browser.redaction import CredentialRedactor
        r = CredentialRedactor()
        r.track_resolved_value("agent1", "ab")  # < 4 chars
        assert "ab" not in r._resolved_values.get("agent1", set())

    def test_per_agent_isolation(self):
        from src.browser.redaction import CredentialRedactor
        r = CredentialRedactor()
        r.track_resolved_value("agent1", "secret1234")
        # Agent2 should not redact agent1's values
        result = r.redact("agent2", "the value is secret1234")
        assert "secret1234" in result

    def test_deep_redact_nested(self):
        from src.browser.redaction import CredentialRedactor
        r = CredentialRedactor()
        r.track_resolved_value("a1", "mysecretvalue")
        obj = {"key": "has mysecretvalue", "nested": [{"v": "also mysecretvalue"}]}
        result = r.deep_redact("a1", obj)
        assert "mysecretvalue" not in str(result)
        assert "[REDACTED]" in result["key"]

    def test_clear_agent(self):
        from src.browser.redaction import CredentialRedactor
        r = CredentialRedactor()
        r.track_resolved_value("a1", "secret1234")
        r.clear_agent("a1")
        result = r.redact("a1", "secret1234")
        assert "secret1234" in result  # no longer redacted


class TestBrowserManagerLifecycle:
    """Tests for BrowserManager start/stop/idle logic."""

    @pytest.mark.asyncio
    async def test_get_status_no_browser(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles", max_concurrent=3)
        status = await mgr.get_status("nonexistent")
        assert status["running"] is False

    @pytest.mark.asyncio
    async def test_service_status(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles", max_concurrent=5)
        status = await mgr.get_service_status()
        assert status["healthy"] is True
        assert status["active_browsers"] == 0
        assert status["max_concurrent"] == 5

    @pytest.mark.asyncio
    async def test_stop_nonexistent_is_noop(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        # Should not raise
        await mgr.stop("nonexistent")

    @pytest.mark.asyncio
    async def test_reset_stops_instance(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        # Manually inject a mock instance
        mock_context = AsyncMock()
        inst = CamoufoxInstance("agent1", MagicMock(), mock_context, MagicMock())
        mgr._instances["agent1"] = inst

        await mgr.reset("agent1")
        assert "agent1" not in mgr._instances
        mock_context.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_all(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        for aid in ("a1", "a2"):
            mock_ctx = AsyncMock()
            inst = CamoufoxInstance(aid, MagicMock(), mock_ctx, MagicMock())
            mgr._instances[aid] = inst

        await mgr.stop_all()
        assert len(mgr._instances) == 0

    @pytest.mark.asyncio
    async def test_focus_auto_starts_browser(self):
        """Focus auto-starts a browser if one isn't running."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.bring_to_front = AsyncMock()
        inst = CamoufoxInstance("agent1", MagicMock(), AsyncMock(), mock_page)
        mgr.get_or_start = AsyncMock(return_value=inst)
        result = await mgr.focus("agent1")
        assert result is True
        mgr.get_or_start.assert_awaited_once_with("agent1")
        mock_page.bring_to_front.assert_awaited_once()


class TestBrowserServer:
    """Tests for the browser service FastAPI endpoints."""

    @pytest.mark.asyncio
    async def test_service_status_endpoint(self):
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/browser/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["healthy"] is True

    @pytest.mark.asyncio
    async def test_agent_status_not_running(self):
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/browser/test_agent/status")
        assert resp.status_code == 200
        assert resp.json()["running"] is False

    @pytest.mark.asyncio
    async def test_navigate_requires_url(self):
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.post("/browser/test_agent/navigate", json={})
        assert resp.status_code == 400


class TestBrowserManagerRefResolution:
    """Tests for ref-based element resolution using role+name."""

    @pytest.mark.asyncio
    async def test_locator_from_ref_with_name(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_page.get_by_role.return_value = mock_locator
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "button", "name": "Submit"}}

        locator = mgr._locator_from_ref(inst, "e0")
        mock_page.get_by_role.assert_called_once_with("button", name="Submit")
        assert locator is mock_locator

    @pytest.mark.asyncio
    async def test_locator_from_ref_no_name(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_page.get_by_role.return_value = mock_locator
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "textbox", "name": ""}}

        mgr._locator_from_ref(inst, "e0")
        mock_page.get_by_role.assert_called_once_with("textbox")

    @pytest.mark.asyncio
    async def test_locator_from_ref_missing_returns_none(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), MagicMock())
        inst.refs = {}

        assert mgr._locator_from_ref(inst, "e99") is None


class TestBrowserManagerCredentialTracking:
    """Tests for is_credential flag in type_text."""

    @pytest.mark.asyncio
    async def test_non_credential_text_not_tracked(self):
        """Plain text typing should NOT add to redactor."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_page.fill = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        await mgr.type_text("a1", selector="input", text="hello world", is_credential=False)
        assert "hello world" not in mgr.redactor._resolved_values.get("a1", set())

    @pytest.mark.asyncio
    async def test_credential_text_is_tracked(self):
        """Credential text typing SHOULD add to redactor."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_page.fill = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        await mgr.type_text("a1", selector="input", text="secret-password", is_credential=True)
        assert "secret-password" in mgr.redactor._resolved_values.get("a1", set())


class TestTypeTextClearBehavior:
    """Tests for type_text clear parameter."""

    @pytest.mark.asyncio
    async def test_clear_true_uses_fill(self):
        """clear=True (default) should use fill() which replaces content."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_page.fill = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        await mgr.type_text("a1", selector="input", text="hello", clear=True)
        mock_page.fill.assert_called_once_with("input", "hello")

    @pytest.mark.asyncio
    async def test_clear_false_uses_press_sequentially(self):
        """clear=False should use press_sequentially() to append text."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_locator.press_sequentially = AsyncMock()
        mock_page.locator.return_value = mock_locator
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        await mgr.type_text("a1", selector="input", text="appended", clear=False)
        mock_page.locator.assert_called_once_with("input")
        mock_locator.press_sequentially.assert_called_once_with("appended")


class TestCamoufoxInstanceLock:
    """Tests that CamoufoxInstance has a per-instance lock."""

    def test_instance_has_lock(self):
        from src.browser.service import CamoufoxInstance
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), MagicMock())
        assert hasattr(inst, "lock")
        assert isinstance(inst.lock, asyncio.Lock)


class TestStealthConfig:
    """Tests for stealth.py launch options."""

    def test_build_launch_options_no_proxy(self):
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert opts["headless"] is False
        assert opts["humanize"] is True
        assert opts["persistent_context"] is True
        assert opts["user_data_dir"] == "/tmp/profile"
        assert "proxy" not in opts

    def test_build_launch_options_with_proxy(self):
        from src.browser.stealth import build_launch_options  # noqa: F401
        env = {
            "BROWSER_PROXY_URL": "http://proxy.example.com:8080",
            "BROWSER_PROXY_USER": "user",
            "BROWSER_PROXY_PASS": "pass",
        }
        with patch.dict("os.environ", env):
            from src.browser.stealth import get_proxy_config
            config = get_proxy_config()
        assert config is not None
        assert config["server"] == "http://proxy.example.com:8080"
        assert config["username"] == "user"
