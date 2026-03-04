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

    def test_proxy_url_logging_strips_credentials(self):
        """Proxy URL with embedded credentials should not log the password."""
        from src.browser.stealth import get_proxy_config
        env = {"BROWSER_PROXY_URL": "http://user:s3cret@proxy.example.com:8080"}
        with patch.dict("os.environ", env):
            with patch("src.browser.stealth.logger") as mock_logger:
                get_proxy_config()
                log_msg = mock_logger.info.call_args[0][1]
                assert "s3cret" not in log_msg
                assert "user:" not in log_msg
                assert "proxy.example.com" in log_msg


class TestNavigate:
    """Tests for BrowserManager.navigate()."""

    @pytest.mark.asyncio
    async def test_navigate_success(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="Example Domain")
        mock_page.url = "https://example.com"
        mock_page.evaluate = AsyncMock(return_value="Example Domain\nThis domain is for use in examples.")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.navigate("a1", "https://example.com", wait_ms=0)
        assert result["success"] is True
        assert result["data"]["title"] == "Example Domain"
        assert result["data"]["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_navigate_blocked_file_url(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        result = await mgr.navigate("a1", "file:///etc/passwd")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_navigate_blocked_javascript_url(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        result = await mgr.navigate("a1", "javascript:alert(1)")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_navigate_blocked_data_url(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        result = await mgr.navigate("a1", "data:text/html,<h1>hi</h1>")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_navigate_caps_wait_ms(self):
        """wait_ms over 10000 should be capped to 10000."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.url = "https://example.com"
        mock_page.evaluate = AsyncMock(return_value="")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        # Should not hang — wait_ms is capped
        result = await mgr.navigate("a1", "https://example.com", wait_ms=999999)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_navigate_error(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("net::ERR_CONNECTION_REFUSED"))
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.navigate("a1", "https://example.com")
        assert result["success"] is False
        assert "ERR_CONNECTION_REFUSED" in result["error"]


class TestClick:
    """Tests for BrowserManager.click()."""

    @pytest.mark.asyncio
    async def test_click_by_ref(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()
        mock_page.get_by_role.return_value = mock_locator
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "button", "name": "Submit"}}
        mgr._instances["a1"] = inst

        result = await mgr.click("a1", ref="e0")
        assert result["success"] is True
        assert result["data"]["clicked"] == "e0"

    @pytest.mark.asyncio
    async def test_click_by_selector(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.click = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.click("a1", selector="#submit-btn")
        assert result["success"] is True
        assert result["data"]["clicked"] == "#submit-btn"

    @pytest.mark.asyncio
    async def test_click_no_ref_or_selector(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), AsyncMock())
        mgr._instances["a1"] = inst

        result = await mgr.click("a1")
        assert result["success"] is False
        assert "Must provide" in result["error"]

    @pytest.mark.asyncio
    async def test_click_missing_ref(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), MagicMock())
        inst.refs = {}
        mgr._instances["a1"] = inst

        result = await mgr.click("a1", ref="e99")
        assert result["success"] is False


class TestScreenshot:
    """Tests for BrowserManager.screenshot()."""

    @pytest.mark.asyncio
    async def test_screenshot_success(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.screenshot("a1")
        assert result["success"] is True
        assert result["data"]["format"] == "png"
        assert len(result["data"]["image_base64"]) > 0


class TestEvaluate:
    """Tests for BrowserManager.evaluate()."""

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=42)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.evaluate("a1", "1 + 41")
        assert result["success"] is True
        assert result["data"]["result"] == 42

    @pytest.mark.asyncio
    async def test_evaluate_error(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(side_effect=Exception("Evaluation failed"))
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.evaluate("a1", "invalid()")
        assert result["success"] is False
        assert "failed" in result["error"].lower()


class TestLRUEviction:
    """Tests for max_concurrent browser eviction."""

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """When max concurrent is reached, least recently used should be stopped."""
        import time

        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles", max_concurrent=2)

        # Add two instances — a1 is older
        ctx_a1 = AsyncMock()
        inst_a1 = CamoufoxInstance("a1", MagicMock(), ctx_a1, MagicMock())
        inst_a1.last_activity = time.time() - 100  # older
        mgr._instances["a1"] = inst_a1

        ctx_a2 = AsyncMock()
        inst_a2 = CamoufoxInstance("a2", MagicMock(), ctx_a2, MagicMock())
        inst_a2.last_activity = time.time()  # newer
        mgr._instances["a2"] = inst_a2

        # Mock _start_browser so get_or_start can create a3
        mock_page = AsyncMock()
        mock_page.bring_to_front = AsyncMock()
        new_inst = CamoufoxInstance("a3", MagicMock(), AsyncMock(), mock_page)
        mgr._start_browser = AsyncMock(return_value=new_inst)

        await mgr.get_or_start("a3")

        # a1 (oldest) should have been evicted
        assert "a1" not in mgr._instances
        assert "a2" in mgr._instances
        assert "a3" in mgr._instances
        ctx_a1.close.assert_called_once()


class TestIdleCleanup:
    """Tests for idle browser cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_idle_stops_timed_out(self):
        import time

        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles", idle_timeout_minutes=1)

        # One idle, one active
        ctx_idle = AsyncMock()
        inst_idle = CamoufoxInstance("idle", MagicMock(), ctx_idle, MagicMock())
        inst_idle.last_activity = time.time() - 120  # 2 min ago, past 1 min timeout
        mgr._instances["idle"] = inst_idle

        ctx_active = AsyncMock()
        inst_active = CamoufoxInstance("active", MagicMock(), ctx_active, MagicMock())
        inst_active.last_activity = time.time()
        mgr._instances["active"] = inst_active

        await mgr._cleanup_idle()

        assert "idle" not in mgr._instances
        assert "active" in mgr._instances
        ctx_idle.close.assert_called_once()
        ctx_active.close.assert_not_called()


class TestServerAuth:
    """Tests for browser server auth token verification."""

    def test_auth_required_rejects_no_token(self):
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager

        with patch.dict("os.environ", {"BROWSER_AUTH_TOKEN": "secret-token"}):
            mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
            app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/browser/status")
        assert resp.status_code == 401

    def test_auth_required_accepts_valid_token(self):
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager

        with patch.dict("os.environ", {"BROWSER_AUTH_TOKEN": "secret-token"}):
            mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
            app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/browser/status", headers={"Authorization": "Bearer secret-token"})
        assert resp.status_code == 200

    def test_auth_required_rejects_wrong_token(self):
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager

        with patch.dict("os.environ", {"BROWSER_AUTH_TOKEN": "secret-token"}):
            mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
            app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/browser/status", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 401


class TestAgentIdValidation:
    """Tests for agent_id validation."""

    @pytest.mark.asyncio
    async def test_valid_agent_id(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        inst = CamoufoxInstance("agent-1", MagicMock(), AsyncMock(), AsyncMock())
        mgr._start_browser = AsyncMock(return_value=inst)

        result = await mgr.get_or_start("agent-1")
        assert result is inst

    @pytest.mark.asyncio
    async def test_invalid_agent_id_with_slashes(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        with pytest.raises(ValueError, match="Invalid agent_id"):
            await mgr.get_or_start("../../etc/passwd")

    @pytest.mark.asyncio
    async def test_invalid_agent_id_with_spaces(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        with pytest.raises(ValueError, match="Invalid agent_id"):
            await mgr.get_or_start("agent with spaces")


class TestSnapshot:
    """Tests for BrowserManager.snapshot()."""

    @pytest.mark.asyncio
    async def test_snapshot_empty_page(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=None)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        assert result["data"]["snapshot"] == "(empty page)"
        assert result["data"]["refs"] == {}

    @pytest.mark.asyncio
    async def test_snapshot_with_actionable_elements(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        tree = {
            "role": "WebArea",
            "name": "Test Page",
            "children": [
                {"role": "button", "name": "Submit"},
                {"role": "textbox", "name": "Email", "value": "test@example.com"},
                {"role": "heading", "name": "Welcome"},
            ],
        }
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        assert "e0" in result["data"]["refs"]
        assert "e1" in result["data"]["refs"]
        assert "e2" in result["data"]["refs"]
        assert result["data"]["refs"]["e0"]["role"] == "button"
        assert "Submit" in result["data"]["snapshot"]
        # Refs stored on instance for click/type by ref
        assert inst.refs == result["data"]["refs"]

    @pytest.mark.asyncio
    async def test_snapshot_credential_value_masked(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        tree = {
            "role": "WebArea",
            "name": "",
            "children": [
                {"role": "textbox", "name": "Password", "value": "s3cretPass!"},
            ],
        }
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.credential_filled_refs.add("e0")
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        assert "****" in result["data"]["snapshot"]
        assert "s3cretPass!" not in result["data"]["snapshot"]

    @pytest.mark.asyncio
    async def test_snapshot_no_interactive_elements(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        tree = {
            "role": "WebArea",
            "name": "Plain page",
            "children": [
                {"role": "generic", "name": "just text"},
            ],
        }
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        assert result["data"]["snapshot"] == "(no interactive elements)"

    @pytest.mark.asyncio
    async def test_snapshot_element_limit(self):
        """Snapshot should stop adding refs after _MAX_SNAPSHOT_ELEMENTS."""
        from src.browser.service import _MAX_SNAPSHOT_ELEMENTS, BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        # Create tree with more elements than the limit
        children = [{"role": "button", "name": f"btn{i}"} for i in range(_MAX_SNAPSHOT_ELEMENTS + 10)]
        tree = {"role": "WebArea", "name": "", "children": children}
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        assert len(result["data"]["refs"]) == _MAX_SNAPSHOT_ELEMENTS


class TestTypeTextWithRef:
    """Tests for type_text using ref-based element resolution."""

    @pytest.mark.asyncio
    async def test_type_by_ref_clear(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_locator.fill = AsyncMock()
        mock_page.get_by_role.return_value = mock_locator
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "textbox", "name": "Email"}}
        mgr._instances["a1"] = inst

        result = await mgr.type_text("a1", ref="e0", text="test@example.com", clear=True)
        assert result["success"] is True
        mock_locator.fill.assert_called_once_with("test@example.com")

    @pytest.mark.asyncio
    async def test_type_by_ref_no_clear(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_locator.press_sequentially = AsyncMock()
        mock_page.get_by_role.return_value = mock_locator
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "textbox", "name": "Email"}}
        mgr._instances["a1"] = inst

        result = await mgr.type_text("a1", ref="e0", text="appended", clear=False)
        assert result["success"] is True
        mock_locator.press_sequentially.assert_called_once_with("appended")

    @pytest.mark.asyncio
    async def test_type_by_ref_credential_tracks_ref(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_locator.fill = AsyncMock()
        mock_page.get_by_role.return_value = mock_locator
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "textbox", "name": "Password"}}
        mgr._instances["a1"] = inst

        await mgr.type_text("a1", ref="e0", text="secret123", is_credential=True)
        assert "e0" in inst.credential_filled_refs
        assert "secret123" in mgr.redactor._resolved_values.get("a1", set())

    @pytest.mark.asyncio
    async def test_type_no_ref_or_selector(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), AsyncMock())
        mgr._instances["a1"] = inst

        result = await mgr.type_text("a1", text="hello")
        assert result["success"] is False
        assert "Must provide" in result["error"]


class TestEvaluateRedaction:
    """Tests that evaluate results are redacted."""

    @pytest.mark.asyncio
    async def test_evaluate_redacts_credentials(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mgr.redactor.track_resolved_value("a1", "mysecrettoken")
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="token is mysecrettoken")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.evaluate("a1", "document.cookie")
        assert result["success"] is True
        assert "mysecrettoken" not in str(result["data"]["result"])
        assert "[REDACTED]" in str(result["data"]["result"])


class TestSolveCaptcha:
    """Tests for captcha detection."""

    @pytest.mark.asyncio
    async def test_no_captcha_found(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_locator.count = AsyncMock(return_value=0)
        mock_page.locator.return_value = mock_locator
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.solve_captcha("a1")
        assert result["success"] is True
        assert result["data"]["captcha_found"] is False

    @pytest.mark.asyncio
    async def test_captcha_found(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        call_count = [0]
        async def mock_count():
            call_count[0] += 1
            # First selector (recaptcha iframe) returns 1
            return 1 if call_count[0] == 1 else 0

        mock_locator = MagicMock()
        mock_locator.count = mock_count
        mock_page.locator.return_value = mock_locator
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.solve_captcha("a1")
        assert result["success"] is True
        assert result["data"]["captcha_found"] is True
