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
        mock_locator.nth = MagicMock(return_value=mock_locator)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "button", "name": "Submit", "index": 0}}

        locator = mgr._locator_from_ref(inst, "e0")
        mock_page.get_by_role.assert_called_once_with("button", name="Submit")
        # .nth(0) is called to target the specific occurrence
        mock_page.get_by_role.return_value.nth.assert_called_once_with(0)
        assert locator is mock_locator.nth.return_value

    @pytest.mark.asyncio
    async def test_locator_from_ref_no_name(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_page.get_by_role.return_value = mock_locator
        mock_locator.nth = MagicMock(return_value=mock_locator)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "textbox", "name": "", "index": 0}}

        mgr._locator_from_ref(inst, "e0")
        mock_page.get_by_role.assert_called_once_with("textbox")
        mock_locator.nth.assert_called_once_with(0)

    @pytest.mark.asyncio
    async def test_locator_from_ref_uses_nth_for_duplicate(self):
        """Second occurrence of same role+name must use .nth(1) to skip the first."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_page.get_by_role.return_value = mock_locator
        mock_locator.nth = MagicMock(return_value=mock_locator)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        # e1 is the second occurrence (index=1) of the same textbox
        inst.refs = {
            "e0": {"role": "textbox", "name": "Post text", "index": 0},
            "e1": {"role": "textbox", "name": "Post text", "index": 1},
        }

        mgr._locator_from_ref(inst, "e1")
        mock_locator.nth.assert_called_once_with(1)

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

        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.random.random", return_value=1.0):
            await mgr.type_text("a1", selector="input", text="hello world", is_credential=False)
        assert "hello world" not in mgr.redactor._resolved_values.get("a1", set())

    @pytest.mark.asyncio
    async def test_credential_text_is_tracked(self):
        """Credential text typing SHOULD add to redactor."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.random.random", return_value=1.0):
            await mgr.type_text("a1", selector="input", text="secret-password", is_credential=True)
        assert "secret-password" in mgr.redactor._resolved_values.get("a1", set())


class TestTypeTextClearBehavior:
    """Tests for type_text clear parameter."""

    @pytest.mark.asyncio
    async def test_clear_true_clicks_selects_all_then_types(self):
        """clear=True should click, Ctrl+A to select all, then type char-by-char.

        Must NOT use fill() — fill() is atomic and bypasses the keyboard event
        chain that React/Vue controlled components rely on to update state.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.random.random", return_value=1.0):
            await mgr.type_text("a1", selector="input", text="hello", clear=True)
        mock_page.click.assert_called_once_with("input", timeout=10000)
        mock_page.keyboard.press.assert_any_call("Control+a")
        # Printable chars now go through keyboard.press, not evaluate
        press_calls = [c[0][0] for c in mock_page.keyboard.press.call_args_list]
        assert all(c in press_calls for c in list("hello"))
        assert mock_page.evaluate.await_count == 0

    @pytest.mark.asyncio
    async def test_clear_false_types_without_select_all(self):
        """clear=False should click then type char-by-char without Ctrl+A."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.random.random", return_value=1.0):
            await mgr.type_text("a1", selector="input", text="ab", clear=False)
        mock_page.click.assert_called_once_with("input", timeout=10000)
        press_calls = [c[0][0] for c in mock_page.keyboard.press.call_args_list]
        assert "Control+a" not in press_calls
        # keyboard.press used for each char, not evaluate
        assert "a" in press_calls and "b" in press_calls
        assert mock_page.evaluate.await_count == 0


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

    def test_default_os_is_windows(self):
        """Default OS fingerprint must be windows, not linux."""
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert opts["os"] == "windows"

    def test_os_override_via_env(self):
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {"BROWSER_OS": "macos"}):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert opts["os"] == "macos"

    def test_invalid_os_falls_back_to_windows(self):
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {"BROWSER_OS": "solaris"}):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert opts["os"] == "windows"

    def test_webrtc_blocked_via_camoufox_toggle(self):
        """WebRTC must be blocked via Camoufox's block_webrtc toggle, not manual prefs.

        block_webrtc=True is Camoufox's canonical way to block WebRTC — it covers
        all relevant prefs and is more reliable than setting them manually.
        Manual WebRTC prefs must NOT be in firefox_user_prefs (they're redundant
        and could drift out of sync with the Camoufox implementation).
        """
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert opts["block_webrtc"] is True
        # Manual WebRTC prefs are redundant — Camoufox's toggle handles them
        prefs = opts["firefox_user_prefs"]
        assert "media.peerconnection.enabled" not in prefs

    def test_rfp_is_off(self):
        """privacy.resistFingerprinting must be False — RFP values are detectable."""
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert opts["firefox_user_prefs"]["privacy.resistFingerprinting"] is False

    def test_disk_cache_enabled(self):
        """Real browsers have disk cache — missing cache is a bot signal."""
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        prefs = opts["firefox_user_prefs"]
        assert prefs["browser.cache.disk.enable"] is True
        assert prefs["browser.cache.memory.enable"] is True

    def test_geoip_only_with_proxy(self):
        """GeoIP must only be enabled when a proxy is configured."""
        from src.browser.stealth import build_launch_options
        # No proxy — geoip should not be set
        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert "geoip" not in opts
        # With proxy — geoip should be True
        env = {"BROWSER_PROXY_URL": "http://proxy.example.com:8080"}
        with patch.dict("os.environ", env):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert opts.get("geoip") is True

    # Resolution tests removed — _pick_resolution was dead code (VNC is always
    # 1920×1080, so per-agent resolution variation was unused).

    def test_window_fills_vnc_display(self):
        """window= must be (1920, 1080) to fill the KasmVNC display.

        The VNC container runs at 1920×1080.  window= must match screen= so that
        window.innerWidth and window.screen.width are consistent — a mismatch
        is itself a bot detection signal.
        """
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert opts["window"] == (1920, 1080)

    def test_locale_set_and_timezone_absent(self):
        """locale= is a valid Camoufox param; timezone= is NOT (would cause TypeError).

        Playwright's launch_persistent_context uses timezone_id, not timezone.
        Camoufox does not expose timezone as a top-level parameter — BrowserForge
        infers it from locale/geoip.  We must never pass timezone= directly.
        """
        from src.browser.stealth import build_launch_options
        env = {"BROWSER_LOCALE": "de-DE"}
        with patch.dict("os.environ", env):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert opts["locale"] == "de-DE"
        assert "timezone" not in opts

    def test_build_launch_options_with_proxy(self):
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
        mock_locator.nth = MagicMock(return_value=mock_locator)
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
    async def test_snapshot_duplicate_elements_disambiguated(self):
        """Duplicate role+name elements (e.g. X's two composer nodes) must get
        distinct index values so _locator_from_ref can use .nth() to hit the right one.
        The snapshot text must flag the second occurrence with [dup:2] so the agent
        can tell which ref is the active element.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        tree = {
            "role": "WebArea",
            "name": "",
            "children": [
                {"role": "textbox", "name": "Post text"},   # active composer
                {"role": "textbox", "name": "Post text"},   # stale/hidden composer
                {"role": "button", "name": "Post"},         # active Post button
                {"role": "button", "name": "Post"},         # stale Post button
            ],
        }
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        refs = result["data"]["refs"]

        # First occurrences get index=0
        assert refs["e0"]["index"] == 0
        assert refs["e2"]["index"] == 0
        # Second occurrences get index=1
        assert refs["e1"]["index"] == 1
        assert refs["e3"]["index"] == 1

        # Snapshot text should flag duplicates
        snap = result["data"]["snapshot"]
        assert "dup:2" in snap
        # First occurrences should NOT be flagged
        lines = snap.splitlines()
        assert not any("dup" in ln for ln in lines if "e0" in ln)
        assert not any("dup" in ln for ln in lines if "e2" in ln)

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
        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()
        mock_page.get_by_role.return_value = mock_locator
        mock_locator.nth = MagicMock(return_value=mock_locator)
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "textbox", "name": "Email"}}
        mgr._instances["a1"] = inst

        with patch("src.browser.service.random.random", return_value=1.0):
            result = await mgr.type_text("a1", ref="e0", text="test@example.com", clear=True)
        assert result["success"] is True
        mock_locator.click.assert_called_once_with(timeout=10000)
        mock_page.keyboard.press.assert_any_call("Control+a")
        # Each printable char uses keyboard.press, not evaluate
        press_calls = [c[0][0] for c in mock_page.keyboard.press.call_args_list]
        for ch in "test@example.com":
            assert ch in press_calls
        assert mock_page.evaluate.await_count == 0

    @pytest.mark.asyncio
    async def test_type_by_ref_no_clear(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()
        mock_page.get_by_role.return_value = mock_locator
        mock_locator.nth = MagicMock(return_value=mock_locator)
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "textbox", "name": "Email"}}
        mgr._instances["a1"] = inst

        with patch("src.browser.service.random.random", return_value=1.0):
            result = await mgr.type_text("a1", ref="e0", text="ab", clear=False)
        assert result["success"] is True
        mock_locator.click.assert_called_once_with(timeout=10000)
        press_calls = [c[0][0] for c in mock_page.keyboard.press.call_args_list]
        assert "Control+a" not in press_calls
        assert "a" in press_calls and "b" in press_calls
        assert mock_page.evaluate.await_count == 0

    @pytest.mark.asyncio
    async def test_type_by_ref_credential_tracks_ref(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()
        mock_page.get_by_role.return_value = mock_locator
        mock_locator.nth = MagicMock(return_value=mock_locator)
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "textbox", "name": "Password"}}
        mgr._instances["a1"] = inst

        with patch("src.browser.service.random.random", return_value=1.0):
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


class TestHumanTiming:
    """Validate timing helper distributions stay within expected ranges."""

    def test_action_delay_range(self):
        from src.browser.timing import action_delay
        samples = [action_delay() for _ in range(1000)]
        assert all(0.15 <= s <= 0.50 for s in samples)
        mean = sum(samples) / len(samples)
        assert 0.25 <= mean <= 0.35

    def test_navigation_jitter_range(self):
        from src.browser.timing import navigation_jitter
        samples = [navigation_jitter() for _ in range(1000)]
        assert all(0.0 <= s <= 0.50 for s in samples)
        mean = sum(samples) / len(samples)
        assert 0.12 <= mean <= 0.28

    def test_keystroke_delay_alpha(self):
        from src.browser.timing import keystroke_delay
        samples = [keystroke_delay("a") for _ in range(1000)]
        assert all(0.04 <= s <= 0.20 for s in samples)
        mean = sum(samples) / len(samples)
        assert 0.06 <= mean <= 0.10

    def test_keystroke_delay_symbol_slower(self):
        from src.browser.timing import keystroke_delay
        alpha = [keystroke_delay("a") for _ in range(1000)]
        symbol = [keystroke_delay("@") for _ in range(1000)]
        assert sum(symbol) / len(symbol) > sum(alpha) / len(alpha)

    def test_keystroke_delay_space_faster_than_alpha(self):
        """Space should be faster than alpha (word-boundary rhythm)."""
        from src.browser.timing import keystroke_delay
        alpha = [keystroke_delay("a") for _ in range(1000)]
        space = [keystroke_delay(" ") for _ in range(1000)]
        assert sum(space) / len(space) < sum(alpha) / len(alpha)

    def test_think_pause_range(self):
        from src.browser.timing import think_pause
        samples = [think_pause() for _ in range(1000)]
        assert all(0.30 <= s <= 1.50 for s in samples)
        mean = sum(samples) / len(samples)
        assert 0.50 <= mean <= 0.80

    def test_word_boundary_chars_constant(self):
        """_WORD_BOUNDARY_CHARS must contain expected characters."""
        from src.browser.service import _WORD_BOUNDARY_CHARS
        for ch in (" ", ".", ",", "!", "?", ";", ":", "\n", "\t"):
            assert ch in _WORD_BOUNDARY_CHARS, f"'{ch}' missing from _WORD_BOUNDARY_CHARS"
        for ch in ("a", "b", "z", "0"):
            assert ch not in _WORD_BOUNDARY_CHARS

    def test_scroll_pause_range(self):
        from src.browser.timing import scroll_pause
        samples = [scroll_pause() for _ in range(1000)]
        assert all(0.03 <= s <= 0.15 for s in samples)

    def test_scroll_increment_range(self):
        from src.browser.timing import scroll_increment
        samples = [scroll_increment() for _ in range(1000)]
        assert all(80 <= s <= 200 for s in samples)
        mean = sum(samples) / len(samples)
        assert 120 <= mean <= 160


class TestScroll:
    """Tests for BrowserManager.scroll()."""

    @pytest.mark.asyncio
    async def test_scroll_down_default(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.evaluate = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.scroll("a1", direction="down")
        assert result["success"] is True
        assert result["data"]["direction"] == "down"
        assert result["data"]["pixels"] >= 720
        assert mock_page.evaluate.await_count >= 1

    @pytest.mark.asyncio
    async def test_scroll_up(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.evaluate = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.scroll("a1", direction="up", amount=200)
        assert result["success"] is True
        assert result["data"]["direction"] == "up"
        # With parameterized evaluate, the second positional arg is the delta.
        # All scroll deltas must be negative for "up" direction.
        calls = mock_page.evaluate.call_args_list
        for call in calls:
            delta = call[0][1]  # second positional arg is the numeric delta
            assert delta < 0, f"Expected negative delta for 'up', got {delta}"

    @pytest.mark.asyncio
    async def test_scroll_to_ref(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = AsyncMock()
        mock_locator.scroll_into_view_if_needed = AsyncMock()
        mock_page.get_by_role.return_value = mock_locator
        mock_locator.nth = MagicMock(return_value=mock_locator)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e3": {"role": "button", "name": "Submit"}}
        mgr._instances["a1"] = inst

        result = await mgr.scroll("a1", ref="e3")
        assert result["success"] is True
        assert result["data"]["scrolled_to_ref"] == "e3"
        mock_locator.scroll_into_view_if_needed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_scroll_missing_ref(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.evaluate = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "button", "name": "OK"}}
        mgr._instances["a1"] = inst

        result = await mgr.scroll("a1", ref="e99")
        assert result["success"] is False
        assert "not found" in result["error"]
        # Must NOT fall through to pixel scrolling
        mock_page.evaluate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_scroll_invalid_direction(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        result = await mgr.scroll("a1", direction="left")
        assert result["success"] is False
        assert "Invalid direction" in result["error"]

    @pytest.mark.asyncio
    async def test_scroll_amount_capped(self):
        from src.browser.service import _MAX_SCROLL_PX, BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.evaluate = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.scroll("a1", amount=999999)
        assert result["success"] is True
        assert result["data"]["pixels"] <= _MAX_SCROLL_PX

    @pytest.mark.asyncio
    async def test_scroll_no_viewport(self):
        """When viewport_size is None, should fallback to 800px."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = None
        mock_page.evaluate = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.scroll("a1")
        assert result["success"] is True
        assert result["data"]["pixels"] >= 800

    @pytest.mark.asyncio
    async def test_scroll_error_handling(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.evaluate = AsyncMock(side_effect=Exception("page closed"))
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.scroll("a1")
        assert result["success"] is False
        assert "page closed" in result["error"]


class TestClickRandomDelay:
    """Verify click delay is not a fixed 0.3s."""

    @pytest.mark.asyncio
    async def test_click_delay_varies(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        delays = []

        async def capture_sleep(t):
            delays.append(t)

        mock_page = AsyncMock()
        mock_page.click = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep", side_effect=capture_sleep):
            for _ in range(10):
                await mgr.click("a1", selector="#btn")

        # All delays should be in action_delay range, not exactly 0.3
        assert all(0.15 <= d <= 0.50 for d in delays)
        # At least some variance (not all identical)
        assert len(set(f"{d:.4f}" for d in delays)) > 1


class TestTypeWithVariance:
    """Verify per-char execCommand calls with keyboard.type fallback and varying delays."""

    @pytest.mark.asyncio
    async def test_type_with_variance_uses_keyboard_press(self):
        """Printable chars use keyboard.press(char) so browser fires trusted beforeinput.

        execCommand('insertText') in Firefox sets isTrusted=false on the resulting
        beforeinput event, which Lexical/React controlled components ignore.
        keyboard.press() sends a real CDP keyDown event; the browser generates
        the beforeinput with isTrusted=true, enabling submit buttons on SPAs.
        """
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.type = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.evaluate = AsyncMock()

        delays = []

        async def capture_sleep(t):
            delays.append(t)

        with patch("src.browser.service.random.random", return_value=1.0):
            with patch("src.browser.service.asyncio.sleep", side_effect=capture_sleep):
                await mgr._type_with_variance(mock_page, "Hi!")

        # All 3 printable chars go through keyboard.press, not evaluate/type
        assert mock_page.keyboard.press.await_count == 3
        assert mock_page.evaluate.await_count == 0
        assert mock_page.keyboard.type.await_count == 0
        press_chars = [c[0][0] for c in mock_page.keyboard.press.call_args_list]
        assert press_chars == ["H", "i", "!"]
        assert len(delays) == 3
        assert all(0.04 <= d <= 0.20 for d in delays)

    @pytest.mark.asyncio
    async def test_type_with_variance_falls_back_to_keyboard_type(self):
        """If keyboard.press() raises (char outside key map), fall back to keyboard.type."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.type = AsyncMock()
        # Simulate keyboard.press failing for unmapped characters
        mock_page.keyboard.press = AsyncMock(side_effect=Exception("Unknown key"))
        mock_page.evaluate = AsyncMock()

        with patch("src.browser.service.random.random", return_value=1.0):
            with patch("src.browser.service.asyncio.sleep"):
                await mgr._type_with_variance(mock_page, "ab")

        # Both chars fell back to keyboard.type after press raised
        assert mock_page.keyboard.type.await_count == 2
        chars_typed = [c[0][0] for c in mock_page.keyboard.type.call_args_list]
        assert chars_typed == ["a", "b"]
        assert mock_page.evaluate.await_count == 0

    @pytest.mark.asyncio
    async def test_type_with_variance_handles_special_keys(self):
        """\\n uses keyboard.press('Enter'), \\t uses keyboard.press('Tab')."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.type = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.evaluate = AsyncMock()

        with patch("src.browser.service.random.random", return_value=1.0):
            with patch("src.browser.service.asyncio.sleep"):
                await mgr._type_with_variance(mock_page, "a\nb\tc")

        # a, b, c → keyboard.press(char); \n → press('Enter'); \t → press('Tab')
        assert mock_page.keyboard.press.await_count == 5
        assert mock_page.evaluate.await_count == 0
        press_calls = [c[0][0] for c in mock_page.keyboard.press.call_args_list]
        assert press_calls == ["a", "Enter", "b", "Tab", "c"]


class TestNavigateWaitUntil:
    """Tests for wait_until parameter on navigate()."""

    @pytest.mark.asyncio
    async def test_navigate_default_wait_until_is_domcontentloaded(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="X")
        mock_page.url = "https://x.com"
        mock_page.evaluate = AsyncMock(return_value="")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        await mgr.navigate("a1", "https://x.com", wait_ms=0)
        mock_page.goto.assert_awaited_once_with(
            "https://x.com", wait_until="domcontentloaded", timeout=30000
        )

    @pytest.mark.asyncio
    async def test_navigate_networkidle_passed_through(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="X")
        mock_page.url = "https://x.com"
        mock_page.evaluate = AsyncMock(return_value="timeline content")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.navigate("a1", "https://x.com", wait_ms=0, wait_until="networkidle")
        assert result["success"] is True
        mock_page.goto.assert_awaited_once_with(
            "https://x.com", wait_until="networkidle", timeout=30000
        )

    @pytest.mark.asyncio
    async def test_navigate_invalid_wait_until_rejected(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        result = await mgr.navigate("a1", "https://example.com", wait_until="invalid")
        assert result["success"] is False
        assert "Invalid wait_until" in result["error"]


class TestForceClick:
    """Tests for force parameter on click()."""

    @pytest.mark.asyncio
    async def test_click_force_false_by_default(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.click = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        await mgr.click("a1", selector="#btn")
        mock_page.click.assert_awaited_once_with("#btn", timeout=10000, force=False)

    @pytest.mark.asyncio
    async def test_click_force_true_passed_through(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.click = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.click("a1", selector="#btn", force=True)
        assert result["success"] is True
        mock_page.click.assert_awaited_once_with("#btn", timeout=10000, force=True)

    @pytest.mark.asyncio
    async def test_click_force_with_ref(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()
        mock_page.get_by_role.return_value = mock_locator
        mock_locator.nth = MagicMock(return_value=mock_locator)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e0": {"role": "button", "name": "Post"}}
        mgr._instances["a1"] = inst

        result = await mgr.click("a1", ref="e0", force=True)
        assert result["success"] is True
        mock_locator.click.assert_awaited_once_with(timeout=10000, force=True)


class TestWaitForElement:
    """Tests for BrowserManager.wait_for_element()."""

    @pytest.mark.asyncio
    async def test_wait_for_visible_success(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.wait_for_element("a1", selector='[data-testid="tweetTextarea_0"]')
        assert result["success"] is True
        assert result["data"]["state"] == "visible"
        mock_page.wait_for_selector.assert_awaited_once_with(
            '[data-testid="tweetTextarea_0"]', state="visible", timeout=10000
        )

    @pytest.mark.asyncio
    async def test_wait_for_timeout_returns_error(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock(side_effect=Exception("Timeout 10000ms exceeded"))
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.wait_for_element("a1", selector="#missing")
        assert result["success"] is False
        assert "Timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_wait_for_invalid_state_rejected(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        result = await mgr.wait_for_element("a1", selector="#btn", state="hovering")
        assert result["success"] is False
        assert "Invalid state" in result["error"]

    @pytest.mark.asyncio
    async def test_wait_for_timeout_capped(self):
        from src.browser.service import _WAIT_FOR_TIMEOUT_MS, BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        await mgr.wait_for_element("a1", selector="#btn", timeout_ms=999999)
        _, kwargs = mock_page.wait_for_selector.call_args
        assert kwargs["timeout"] <= _WAIT_FOR_TIMEOUT_MS


class TestHover:
    """Tests for BrowserManager.hover()."""

    @pytest.mark.asyncio
    async def test_hover_by_selector(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.hover = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            result = await mgr.hover("a1", selector="nav.menu > li")
        assert result["success"] is True
        assert result["data"]["hovered"] == "nav.menu > li"
        mock_page.hover.assert_awaited_once_with("nav.menu > li", timeout=10000)

    @pytest.mark.asyncio
    async def test_hover_by_ref(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_locator = AsyncMock()
        mock_locator.hover = AsyncMock()
        mock_page.get_by_role.return_value = mock_locator
        mock_locator.nth = MagicMock(return_value=mock_locator)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.refs = {"e3": {"role": "link", "name": "Products"}}
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            result = await mgr.hover("a1", ref="e3")
        assert result["success"] is True
        mock_locator.hover.assert_awaited_once_with(timeout=10000)

    @pytest.mark.asyncio
    async def test_hover_no_ref_or_selector(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), AsyncMock())
        mgr._instances["a1"] = inst

        result = await mgr.hover("a1")
        assert result["success"] is False
        assert "Must provide" in result["error"]

    @pytest.mark.asyncio
    async def test_hover_error_propagated(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.hover = AsyncMock(side_effect=Exception("element not found"))
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            result = await mgr.hover("a1", selector="#missing")
        assert result["success"] is False
        assert "element not found" in result["error"]


class TestAllowedBrowserActions:
    """Regression tests for the mesh-proxy allowed-action allowlist.

    Any browser action added to browser_tool.py must also be in the
    _ALLOWED_BROWSER_ACTIONS set in host/server.py, otherwise the skill silently
    fails with a 400 from the proxy.
    """

    def _get_allowed_actions(self) -> frozenset[str]:
        """Extract _ALLOWED_BROWSER_ACTIONS from host/server.py without running the server."""
        import ast
        from pathlib import Path
        source = (Path(__file__).parent.parent / "src/host/server.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_ALLOWED_BROWSER_ACTIONS"
                and isinstance(node.value, ast.Call)
            ):
                # frozenset({...}) call — extract string elements
                set_literal = node.value.args[0]
                if isinstance(set_literal, ast.Set):
                    return frozenset(
                        elt.value for elt in set_literal.elts
                        if isinstance(elt, ast.Constant)
                    )
        return frozenset()

    def test_wait_for_in_allowed_actions(self):
        """wait_for must be allowed — browser_wait_for skill depends on it."""
        actions = self._get_allowed_actions()
        assert "wait_for" in actions, (
            "wait_for missing from _ALLOWED_BROWSER_ACTIONS in host/server.py — "
            "browser_wait_for skill will silently fail"
        )

    def test_hover_in_allowed_actions(self):
        """hover must be allowed — browser_hover skill depends on it."""
        actions = self._get_allowed_actions()
        assert "hover" in actions, (
            "hover missing from _ALLOWED_BROWSER_ACTIONS in host/server.py — "
            "browser_hover skill will silently fail"
        )

    def test_core_actions_present(self):
        """Core browser actions must remain in the allowlist."""
        actions = self._get_allowed_actions()
        required = {"navigate", "snapshot", "click", "type", "screenshot",
                    "reset", "focus", "scroll", "solve_captcha", "wait_for", "hover"}
        missing = required - actions
        assert not missing, f"Missing browser actions: {missing}"


class TestScrollParameterized:
    """Verify scroll uses parameterized evaluate (not f-string injection)."""

    @pytest.mark.asyncio
    async def test_scroll_uses_parameterized_evaluate(self):
        """evaluate must be called with a function string + separate delta arg."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.evaluate = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            await mgr.scroll("a1", direction="down", amount=200)

        for call in mock_page.evaluate.call_args_list:
            fn_str, delta = call[0][0], call[0][1]
            # Function string must not bake in the delta value
            assert str(delta) not in fn_str, (
                "Delta must be a separate argument, not interpolated into the JS string"
            )
            # Delta must be a plain number, not a string
            assert isinstance(delta, (int, float))
            assert delta > 0  # down = positive

    @pytest.mark.asyncio
    async def test_scroll_up_delta_is_negative(self):
        """Scroll up must pass a negative delta as second evaluate argument."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.evaluate = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            await mgr.scroll("a1", direction="up", amount=200)

        for call in mock_page.evaluate.call_args_list:
            delta = call[0][1]
            assert delta < 0


class TestWordBoundaryPause:
    """Verify think_pause fires at word boundaries with higher probability."""

    @pytest.mark.asyncio
    async def test_think_pause_fires_at_word_boundary_not_midword(self):
        """With random.random()=0.05, pause fires after space (12%) but not mid-word (2.5%)."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)

        sleep_calls: list[float] = []

        async def capture_sleep(t: float):
            sleep_calls.append(t)

        # 0.05 is between 0.025 (non-boundary threshold) and 0.12 (boundary threshold)
        # So: fires only when prev_char was a boundary char
        with patch("src.browser.service.random.random", return_value=0.05):
            with patch("src.browser.service.asyncio.sleep", side_effect=capture_sleep):
                # "h w": h=no-pause, ' '=no-pause, w=PAUSE (prev ' ')
                await mgr._type_with_variance(mock_page, "h w")

        # think_pause values are in [0.30, 1.50]; keystroke_delay values are in [0.03, 0.20]
        think_pauses = [t for t in sleep_calls if t >= 0.30]
        assert len(think_pauses) == 1, (
            f"Expected exactly 1 think_pause for 'h w' with random=0.05, got {think_pauses}"
        )

    @pytest.mark.asyncio
    async def test_no_think_pause_when_random_above_boundary_threshold(self):
        """With random.random()=0.15, no pauses fire (0.15 > 0.12)."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)

        sleep_calls: list[float] = []

        async def capture_sleep(t: float):
            sleep_calls.append(t)

        with patch("src.browser.service.random.random", return_value=0.15):
            with patch("src.browser.service.asyncio.sleep", side_effect=capture_sleep):
                await mgr._type_with_variance(mock_page, "hello world")

        think_pauses = [t for t in sleep_calls if t >= 0.30]
        assert len(think_pauses) == 0


# ── Auto-force click tests ─────────────────────────────────────────────────


class TestAutoForceClick:
    """Tests for auto-force click on disabled button/link refs."""

    def _make_manager(self):
        from src.browser.service import BrowserManager
        return BrowserManager.__new__(BrowserManager)

    def _make_instance(self, refs: dict):
        """Create a mock CamoufoxInstance with given refs and a mock page."""
        inst = MagicMock()
        inst.refs = refs
        inst.page = AsyncMock()
        inst.page.click = AsyncMock()
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()
        return inst

    @pytest.mark.asyncio
    async def test_disabled_button_ref_auto_forces(self):
        """A disabled button ref should auto-force the click."""
        from src.browser.service import BrowserManager

        mgr = self._make_manager()
        refs = {"e1": {"role": "button", "name": "Post", "index": 0, "disabled": True}}
        inst = self._make_instance(refs)

        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch.object(BrowserManager, "_locator_from_ref", return_value=mock_locator):
                with patch("src.browser.service.action_delay", return_value=0.01):
                    with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                        result = await mgr.click("agent1", ref="e1")

        assert result["success"] is True
        mock_locator.click.assert_called_once()
        call_kwargs = mock_locator.click.call_args
        assert call_kwargs[1].get("force") is True or call_kwargs.kwargs.get("force") is True

    @pytest.mark.asyncio
    async def test_disabled_link_ref_auto_forces(self):
        """A disabled link ref should also auto-force."""
        from src.browser.service import BrowserManager

        mgr = self._make_manager()
        refs = {"e2": {"role": "link", "name": "Sign in", "index": 0, "disabled": True}}
        inst = self._make_instance(refs)

        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch.object(BrowserManager, "_locator_from_ref", return_value=mock_locator):
                with patch("src.browser.service.action_delay", return_value=0.01):
                    with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                        result = await mgr.click("agent1", ref="e2")

        assert result["success"] is True
        mock_locator.click.assert_called_once()
        call_kwargs = mock_locator.click.call_args
        assert call_kwargs[1].get("force") is True or call_kwargs.kwargs.get("force") is True

    @pytest.mark.asyncio
    async def test_disabled_textbox_no_auto_force(self):
        """A disabled textbox should NOT auto-force (not in _ARIA_FORCE_ROLES)."""
        from src.browser.service import BrowserManager

        mgr = self._make_manager()
        refs = {"e3": {"role": "textbox", "name": "Search", "index": 0, "disabled": True}}
        inst = self._make_instance(refs)

        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch.object(BrowserManager, "_locator_from_ref", return_value=mock_locator):
                with patch("src.browser.service.action_delay", return_value=0.01):
                    with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                        result = await mgr.click("agent1", ref="e3")

        assert result["success"] is True
        call_kwargs = mock_locator.click.call_args
        assert call_kwargs[1].get("force") is False or call_kwargs.kwargs.get("force") is False

    @pytest.mark.asyncio
    async def test_disabled_menuitem_no_auto_force(self):
        """A disabled menuitem should NOT auto-force."""
        from src.browser.service import BrowserManager

        mgr = self._make_manager()
        refs = {"e4": {"role": "menuitem", "name": "Delete", "index": 0, "disabled": True}}
        inst = self._make_instance(refs)

        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch.object(BrowserManager, "_locator_from_ref", return_value=mock_locator):
                with patch("src.browser.service.action_delay", return_value=0.01):
                    with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                        result = await mgr.click("agent1", ref="e4")

        assert result["success"] is True
        call_kwargs = mock_locator.click.call_args
        assert call_kwargs[1].get("force") is False or call_kwargs.kwargs.get("force") is False

    @pytest.mark.asyncio
    async def test_non_disabled_button_no_auto_force(self):
        """A non-disabled button should not trigger auto-force."""
        from src.browser.service import BrowserManager

        mgr = self._make_manager()
        refs = {"e5": {"role": "button", "name": "Submit", "index": 0, "disabled": False}}
        inst = self._make_instance(refs)

        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch.object(BrowserManager, "_locator_from_ref", return_value=mock_locator):
                with patch("src.browser.service.action_delay", return_value=0.01):
                    with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                        result = await mgr.click("agent1", ref="e5")

        assert result["success"] is True
        call_kwargs = mock_locator.click.call_args
        assert call_kwargs[1].get("force") is False or call_kwargs.kwargs.get("force") is False

    @pytest.mark.asyncio
    async def test_explicit_force_with_disabled_button(self):
        """Explicit force=True should still work with disabled button."""
        from src.browser.service import BrowserManager

        mgr = self._make_manager()
        refs = {"e6": {"role": "button", "name": "Post", "index": 0, "disabled": True}}
        inst = self._make_instance(refs)

        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch.object(BrowserManager, "_locator_from_ref", return_value=mock_locator):
                with patch("src.browser.service.action_delay", return_value=0.01):
                    with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                        result = await mgr.click("agent1", ref="e6", force=True)

        assert result["success"] is True
        call_kwargs = mock_locator.click.call_args
        assert call_kwargs[1].get("force") is True or call_kwargs.kwargs.get("force") is True

    @pytest.mark.asyncio
    async def test_old_ref_format_no_auto_force(self):
        """Refs without 'disabled' key should not trigger auto-force (backward compat)."""
        from src.browser.service import BrowserManager

        mgr = self._make_manager()
        refs = {"e7": {"role": "button", "name": "OK", "index": 0}}
        inst = self._make_instance(refs)

        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch.object(BrowserManager, "_locator_from_ref", return_value=mock_locator):
                with patch("src.browser.service.action_delay", return_value=0.01):
                    with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                        result = await mgr.click("agent1", ref="e7")

        assert result["success"] is True
        call_kwargs = mock_locator.click.call_args
        assert call_kwargs[1].get("force") is False or call_kwargs.kwargs.get("force") is False


class TestSnapshotDisabledField:
    """Tests that snapshot stores disabled state in refs."""

    @pytest.mark.asyncio
    async def test_disabled_element_has_disabled_true(self):
        """Snapshot should store disabled=True for disabled elements."""
        from src.browser.redaction import CredentialRedactor
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        mgr.redactor = CredentialRedactor()
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.page.accessibility = MagicMock()
        inst.page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea",
            "children": [
                {"role": "button", "name": "Post", "disabled": True},
            ],
        })
        inst.page.url = "https://x.com"
        inst.page.title = AsyncMock(return_value="X")
        inst.refs = {}
        inst.credential_filled_refs = set()
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.snapshot("agent1")

        assert result["success"] is True
        assert inst.refs["e0"]["disabled"] is True

    @pytest.mark.asyncio
    async def test_non_disabled_element_has_disabled_false(self):
        """Snapshot should store disabled=False for non-disabled elements."""
        from src.browser.redaction import CredentialRedactor
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        mgr.redactor = CredentialRedactor()
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.page.accessibility = MagicMock()
        inst.page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea",
            "children": [
                {"role": "button", "name": "Submit"},
            ],
        })
        inst.page.url = "https://example.com"
        inst.page.title = AsyncMock(return_value="Example")
        inst.refs = {}
        inst.credential_filled_refs = set()
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.snapshot("agent1")

        assert result["success"] is True
        assert inst.refs["e0"]["disabled"] is False


class TestTypeTextSettleDelays:
    """Tests that type_text includes settle delays for SPA reconciliation."""

    @pytest.mark.asyncio
    async def test_settle_delays_present(self):
        """type_text should include settle delays after focus and after typing."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.page.click = AsyncMock()
        inst.page.keyboard = AsyncMock()
        inst.page.keyboard.press = AsyncMock()
        inst.refs = {"e1": {"role": "textbox", "name": "Tweet", "index": 0, "disabled": False}}
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        sleep_calls: list[float] = []

        async def capture_sleep(t):
            sleep_calls.append(t)

        mock_locator = AsyncMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch.object(BrowserManager, "_locator_from_ref", return_value=mock_locator):
                with patch.object(BrowserManager, "_type_with_variance", new_callable=AsyncMock):
                    with patch("src.browser.service.action_delay", return_value=0.15):
                        with patch("src.browser.service.asyncio.sleep", side_effect=capture_sleep):
                            result = await mgr.type_text(
                                "agent1", text="hello", ref="e1", clear=False,
                            )

        assert result["success"] is True
        # Should have at least 2 settle delays (after focus, after typing)
        settle_delays = [t for t in sleep_calls if 0.10 <= t <= 0.50]
        assert len(settle_delays) >= 2

    @pytest.mark.asyncio
    async def test_settle_delays_with_clear(self):
        """type_text with clear=True should have settle + clear + settle delays."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.page.click = AsyncMock()
        inst.page.keyboard = AsyncMock()
        inst.page.keyboard.press = AsyncMock()
        inst.refs = {"e1": {"role": "textbox", "name": "Tweet", "index": 0, "disabled": False}}
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        sleep_calls: list[float] = []

        async def capture_sleep(t):
            sleep_calls.append(t)

        mock_locator = AsyncMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch.object(BrowserManager, "_locator_from_ref", return_value=mock_locator):
                with patch.object(BrowserManager, "_type_with_variance", new_callable=AsyncMock):
                    with patch("src.browser.service.action_delay", return_value=0.15):
                        with patch("src.browser.service.asyncio.sleep", side_effect=capture_sleep):
                            result = await mgr.type_text(
                                "agent1", text="hello", ref="e1", clear=True,
                            )

        assert result["success"] is True
        # Should have settle delays + the 0.05 clear delay
        settle_delays = [t for t in sleep_calls if 0.10 <= t <= 0.50]
        assert len(settle_delays) >= 2
        # The clear operation adds a small delay
        clear_delays = [t for t in sleep_calls if t < 0.10]
        assert len(clear_delays) >= 1


# ── Press key tests ────────────────────────────────────────────────────────


class TestPressKey:
    """Tests for the press_key browser method."""

    @pytest.mark.asyncio
    async def test_press_escape(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.page.keyboard = AsyncMock()
        inst.page.keyboard.press = AsyncMock()
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch("src.browser.service.action_delay", return_value=0.01):
                with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                    result = await mgr.press_key("agent1", "Escape")

        assert result["success"] is True
        assert result["data"]["pressed"] == "Escape"
        inst.page.keyboard.press.assert_called_once_with("Escape")

    @pytest.mark.asyncio
    async def test_press_modifier_combo(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.page.keyboard = AsyncMock()
        inst.page.keyboard.press = AsyncMock()
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch("src.browser.service.action_delay", return_value=0.01):
                with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                    result = await mgr.press_key("agent1", "Control+a")

        assert result["success"] is True
        inst.page.keyboard.press.assert_called_once_with("Control+a")

    @pytest.mark.asyncio
    async def test_press_empty_key_rejected(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        result = await mgr.press_key("agent1", "")
        assert result["success"] is False
        assert "Invalid key" in result["error"]

    @pytest.mark.asyncio
    async def test_press_oversized_key_rejected(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        result = await mgr.press_key("agent1", "x" * 51)
        assert result["success"] is False
        assert "Invalid key" in result["error"]


# ── Go back / forward tests ───────────────────────────────────────────────


class TestGoBackForward:
    """Tests for browser history navigation."""

    @pytest.mark.asyncio
    async def test_go_back(self):
        from src.browser.redaction import CredentialRedactor
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        mgr.redactor = CredentialRedactor()
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.page.go_back = AsyncMock(return_value=MagicMock())  # non-None = navigated
        inst.page.title = AsyncMock(return_value="Previous Page")
        inst.page.url = "https://example.com/prev"
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch("src.browser.service.action_delay", return_value=0.01):
                with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                    result = await mgr.go_back("agent1")

        assert result["success"] is True
        assert result["data"]["title"] == "Previous Page"
        assert result["data"]["navigated"] is True
        inst.page.go_back.assert_called_once()

    @pytest.mark.asyncio
    async def test_go_forward(self):
        from src.browser.redaction import CredentialRedactor
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        mgr.redactor = CredentialRedactor()
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.page.go_forward = AsyncMock(return_value=MagicMock())  # non-None = navigated
        inst.page.title = AsyncMock(return_value="Next Page")
        inst.page.url = "https://example.com/next"
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch("src.browser.service.action_delay", return_value=0.01):
                with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                    result = await mgr.go_forward("agent1")

        assert result["success"] is True
        assert result["data"]["title"] == "Next Page"
        assert result["data"]["navigated"] is True
        inst.page.go_forward.assert_called_once()

    @pytest.mark.asyncio
    async def test_go_forward_no_history(self):
        from src.browser.redaction import CredentialRedactor
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        mgr.redactor = CredentialRedactor()
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.page.go_forward = AsyncMock(return_value=None)
        inst.page.title = AsyncMock(return_value="Current Page")
        inst.page.url = "https://example.com/current"
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch("src.browser.service.action_delay", return_value=0.01):
                with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                    result = await mgr.go_forward("agent1")

        assert result["success"] is True
        assert result["data"]["navigated"] is False

    @pytest.mark.asyncio
    async def test_go_back_no_history(self):
        from src.browser.redaction import CredentialRedactor
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        mgr.redactor = CredentialRedactor()
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.page.go_back = AsyncMock(return_value=None)
        inst.page.title = AsyncMock(return_value="Current Page")
        inst.page.url = "https://example.com/current"
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch("src.browser.service.action_delay", return_value=0.01):
                with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                    result = await mgr.go_back("agent1")

        assert result["success"] is True
        assert result["data"]["navigated"] is False
        assert result["data"]["url"] == "https://example.com/current"


# ── Switch tab tests ───────────────────────────────────────────────────────


class TestSwitchTab:
    """Tests for tab listing and switching."""

    @pytest.mark.asyncio
    async def test_list_tabs(self):
        from src.browser.redaction import CredentialRedactor
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        mgr.redactor = CredentialRedactor()

        page1 = AsyncMock()
        page1.url = "https://example.com"
        page1.title = AsyncMock(return_value="Example")
        page2 = AsyncMock()
        page2.url = "https://accounts.google.com"
        page2.title = AsyncMock(return_value="Sign in")

        inst = MagicMock()
        inst.page = page1
        inst.context = MagicMock()
        inst.context.pages = [page1, page2]
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.switch_tab("agent1")

        assert result["success"] is True
        tabs = result["data"]["tabs"]
        assert len(tabs) == 2
        assert tabs[0]["active"] is True
        assert tabs[1]["active"] is False
        assert result["data"]["active_tab"] == 0

    @pytest.mark.asyncio
    async def test_switch_to_tab(self):
        from src.browser.redaction import CredentialRedactor
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        mgr.redactor = CredentialRedactor()

        page1 = AsyncMock()
        page1.url = "https://example.com"
        page1.title = AsyncMock(return_value="Example")
        page2 = AsyncMock()
        page2.url = "https://accounts.google.com"
        page2.title = AsyncMock(return_value="Sign in")
        page2.bring_to_front = AsyncMock()

        inst = MagicMock()
        inst.page = page1
        inst.context = MagicMock()
        inst.context.pages = [page1, page2]
        inst.refs = {"e0": {"role": "button", "name": "Old", "index": 0, "disabled": False}}
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.switch_tab("agent1", tab_index=1)

        assert result["success"] is True
        assert result["data"]["active_tab"] == 1
        assert result["data"]["tabs"][1]["active"] is True
        assert result["data"]["tabs"][0]["active"] is False
        assert inst.page == page2
        assert inst.refs == {}  # Refs cleared on switch
        assert inst.dialog_active is False  # Dialog state cleared on switch
        page2.bring_to_front.assert_called_once()

    @pytest.mark.asyncio
    async def test_switch_tab_out_of_range(self):
        from src.browser.redaction import CredentialRedactor
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        mgr.redactor = CredentialRedactor()

        page1 = AsyncMock()
        page1.url = "https://example.com"
        page1.title = AsyncMock(return_value="Example")

        inst = MagicMock()
        inst.page = page1
        inst.context = MagicMock()
        inst.context.pages = [page1]
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.switch_tab("agent1", tab_index=5)

        assert result["success"] is False
        assert "out of range" in result["error"]


# ── CAPTCHA detection tests ────────────────────────────────────────────────


class TestCaptchaDetection:
    """Tests for CAPTCHA detection including Cloudflare Turnstile."""

    @pytest.mark.asyncio
    async def test_turnstile_detected(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        # All selectors return 0 except Turnstile
        async def mock_count(sel):
            mock_loc = MagicMock()
            if "challenges.cloudflare.com" in sel:
                mock_loc.count = AsyncMock(return_value=1)
            else:
                mock_loc.count = AsyncMock(return_value=0)
            return mock_loc

        # Build locator mock that returns different counts per selector
        locator_results = {}
        for sel in [
            'iframe[src*="recaptcha"]',
            'iframe[src*="hcaptcha"]',
            'iframe[src*="challenges.cloudflare.com"]',
            'iframe[src*="captcha"]',
            '[class*="cf-turnstile"]',
            '[class*="captcha"]',
            '#captcha',
        ]:
            loc = MagicMock()
            if "challenges.cloudflare.com" in sel:
                loc.count = AsyncMock(return_value=1)
            else:
                loc.count = AsyncMock(return_value=0)
            locator_results[sel] = loc

        default_loc = MagicMock(count=AsyncMock(return_value=0))
        inst.page.locator = MagicMock(
            side_effect=lambda s: locator_results.get(s, default_loc),
        )

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.solve_captcha("agent1")

        assert result["success"] is True
        assert result["data"]["captcha_found"] is True
        assert "challenges.cloudflare.com" in result["data"]["captcha_type"]

    @pytest.mark.asyncio
    async def test_no_captcha(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        # All selectors return 0
        mock_loc = MagicMock()
        mock_loc.count = AsyncMock(return_value=0)
        inst.page.locator = MagicMock(return_value=mock_loc)

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.solve_captcha("agent1")

        assert result["success"] is True
        assert result["data"]["captcha_found"] is False


# ── Dialog scoping tests ──────────────────────────────────────────────────


class TestDialogScoping:
    """When a modal dialog is open, snapshot and locators scope to dialog only.

    This prevents agents from seeing/clicking elements behind the modal overlay
    (e.g. X's sidebar "Post" button behind the compose modal).
    """

    @pytest.mark.asyncio
    async def test_snapshot_scopes_to_dialog(self):
        """Elements outside the dialog should not appear in snapshot."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        tree = {
            "role": "WebArea",
            "name": "",
            "children": [
                {"role": "button", "name": "Post", "disabled": True},  # sidebar
                {"role": "link", "name": "Home"},
                {
                    "role": "dialog",
                    "name": "Create post",
                    "children": [
                        {"role": "textbox", "name": "What is happening?!"},
                        {"role": "button", "name": "Post"},  # modal button
                    ],
                },
            ],
        }
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        snap = result["data"]["snapshot"]
        refs = result["data"]["refs"]

        # Dialog header should be present
        assert "Modal dialog is open" in snap
        # Dialog context element should appear
        assert "Create post" in snap
        # Elements inside the dialog should appear
        assert "What is happening?!" in snap
        # The modal Post button should appear with index 0 (first in dialog scope)
        post_refs = [r for r in refs.values() if r["name"] == "Post"]
        assert len(post_refs) == 1
        assert post_refs[0]["index"] == 0
        # Elements outside the dialog should NOT appear
        assert "Home" not in snap
        # The sidebar Post button should NOT appear
        sidebar_post = [r for r in refs.values()
                        if r["name"] == "Post" and r.get("disabled")]
        assert len(sidebar_post) == 0
        # dialog_active flag should be set
        assert inst.dialog_active is True

    @pytest.mark.asyncio
    async def test_snapshot_no_dialog_walks_all(self):
        """Without a dialog, all elements should appear as before."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        tree = {
            "role": "WebArea",
            "name": "",
            "children": [
                {"role": "button", "name": "Post"},
                {"role": "link", "name": "Home"},
            ],
        }
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        refs = result["data"]["refs"]
        assert len(refs) == 2
        assert inst.dialog_active is False
        assert "Modal dialog" not in result["data"]["snapshot"]

    @pytest.mark.asyncio
    async def test_snapshot_alertdialog_treated_as_modal(self):
        """alertdialog role should also trigger dialog scoping."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        tree = {
            "role": "WebArea",
            "name": "",
            "children": [
                {"role": "button", "name": "Background"},
                {
                    "role": "alertdialog",
                    "name": "Confirm",
                    "children": [
                        {"role": "button", "name": "OK"},
                        {"role": "button", "name": "Cancel"},
                    ],
                },
            ],
        }
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        refs = result["data"]["refs"]
        assert inst.dialog_active is True
        # Only alertdialog elements: alertdialog + OK + Cancel = 3
        assert len(refs) == 3
        assert "Background" not in result["data"]["snapshot"]

    @pytest.mark.asyncio
    async def test_snapshot_dialog_clears_flag_when_dismissed(self):
        """After dialog is dismissed, next snapshot should clear dialog_active."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()

        # First snapshot: dialog open
        tree_with_dialog = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "dialog", "name": "Modal", "children": [
                    {"role": "button", "name": "Close"},
                ]},
            ],
        }
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree_with_dialog)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        await mgr.snapshot("a1")
        assert inst.dialog_active is True

        # Second snapshot: dialog dismissed
        tree_no_dialog = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Post"},
            ],
        }
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree_no_dialog)
        await mgr.snapshot("a1")
        assert inst.dialog_active is False
        assert len(inst.refs) == 1  # Just the Post button

    @pytest.mark.asyncio
    async def test_snapshot_dialog_duplicate_indexing(self):
        """Occurrence indices should be counted within dialog scope only.

        On X/Twitter, the sidebar "Post" button (index 0 in full page scope)
        must not affect the modal "Post" button's index within the dialog.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        tree = {
            "role": "WebArea", "name": "",
            "children": [
                # These are behind the modal — should be excluded
                {"role": "button", "name": "Post"},
                {"role": "textbox", "name": "Search"},
                {
                    "role": "dialog", "name": "Compose",
                    "children": [
                        {"role": "textbox", "name": "Tweet text"},
                        {"role": "button", "name": "Post"},
                    ],
                },
            ],
        }
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        refs = result["data"]["refs"]

        # Only dialog elements: dialog + textbox + button = 3
        assert len(refs) == 3
        # The Post button inside the dialog should have index 0 (not 1)
        post_ref = [r for r in refs.values()
                    if r["role"] == "button" and r["name"] == "Post"]
        assert len(post_ref) == 1
        assert post_ref[0]["index"] == 0

    @pytest.mark.asyncio
    async def test_locator_from_ref_scopes_to_dialog(self):
        """When dialog_active is True, locator should search within dialog elements."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_dialog_locator = MagicMock()
        mock_role_locator = MagicMock()
        mock_page.locator.return_value = mock_dialog_locator
        mock_dialog_locator.get_by_role.return_value = mock_role_locator
        mock_role_locator.nth.return_value = mock_role_locator

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.dialog_active = True
        inst.refs = {"e0": {"role": "button", "name": "Post", "index": 0}}

        locator = mgr._locator_from_ref(inst, "e0")

        # Should scope to dialog elements
        mock_page.locator.assert_called_once_with(
            '[role="dialog"], [role="alertdialog"], dialog[open]'
        )
        # Should search for the button within the dialog scope
        mock_dialog_locator.get_by_role.assert_called_once_with("button", name="Post")
        mock_role_locator.nth.assert_called_once_with(0)
        assert locator is mock_role_locator.nth.return_value

    @pytest.mark.asyncio
    async def test_locator_from_ref_page_scope_when_no_dialog(self):
        """When dialog_active is False, locator should search entire page."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = MagicMock()
        mock_role_locator = MagicMock()
        mock_page.get_by_role.return_value = mock_role_locator
        mock_role_locator.nth.return_value = mock_role_locator

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.dialog_active = False
        inst.refs = {"e0": {"role": "button", "name": "Post", "index": 0}}

        mgr._locator_from_ref(inst, "e0")

        # Should NOT call page.locator() — searches page directly
        mock_page.locator.assert_not_called()
        mock_page.get_by_role.assert_called_once_with("button", name="Post")

    @pytest.mark.asyncio
    async def test_snapshot_nested_dialog(self):
        """Dialog nested inside non-dialog containers should be found."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Outside"},
                # Dialog wrapped in a generic container (React portals do this)
                {"role": "generic", "name": "", "children": [
                    {"role": "dialog", "name": "Nested", "children": [
                        {"role": "button", "name": "Inside"},
                    ]},
                ]},
            ],
        }
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert inst.dialog_active is True
        assert "Outside" not in result["data"]["snapshot"]
        assert "Inside" in result["data"]["snapshot"]


# ── Speed factor tests ────────────────────────────────────────────────────


class TestSpeed:
    """Tests for the configurable browser speed in timing.py.

    Higher speed = faster (shorter delays). Speed 2.0 means half the delays.
    """

    def setup_method(self):
        """Reset speed to default before each test."""
        from src.browser.timing import set_speed
        set_speed(1.0)

    def teardown_method(self):
        """Reset speed after each test to avoid leaking state."""
        from src.browser.timing import set_speed
        set_speed(1.0)

    def test_default_speed(self):
        from src.browser.timing import get_speed
        assert get_speed() == 1.0

    def test_set_speed(self):
        from src.browser.timing import get_speed, set_speed
        set_speed(2.0)
        assert get_speed() == 2.0

    def test_speed_clamped_low(self):
        from src.browser.timing import get_speed, set_speed
        set_speed(0.01)
        assert get_speed() == 0.25  # _SPEED_MIN

    def test_speed_clamped_high(self):
        from src.browser.timing import get_speed, set_speed
        set_speed(99.0)
        assert get_speed() == 4.0  # _SPEED_MAX

    def test_higher_speed_reduces_action_delay(self):
        """Doubling speed should roughly halve the mean action delay."""
        from src.browser.timing import action_delay, set_speed
        set_speed(1.0)
        baseline = [action_delay() for _ in range(2000)]
        set_speed(2.0)
        fast = [action_delay() for _ in range(2000)]
        baseline_mean = sum(baseline) / len(baseline)
        fast_mean = sum(fast) / len(fast)
        ratio = fast_mean / baseline_mean
        assert 0.35 <= ratio <= 0.65, f"Expected ~0.5x, got {ratio:.2f}x"

    def test_lower_speed_increases_keystroke_delay(self):
        """Halving speed should roughly double keystroke delays."""
        from src.browser.timing import keystroke_delay, set_speed
        set_speed(1.0)
        baseline = [keystroke_delay("a") for _ in range(2000)]
        set_speed(0.5)
        slow = [keystroke_delay("a") for _ in range(2000)]
        baseline_mean = sum(baseline) / len(baseline)
        slow_mean = sum(slow) / len(slow)
        ratio = slow_mean / baseline_mean
        assert 1.6 <= ratio <= 2.4, f"Expected ~2.0x, got {ratio:.2f}x"

    def test_speed_does_not_scale_scroll_increment(self):
        """Scroll increment (pixels) should not change with speed."""
        from src.browser.timing import scroll_increment, set_speed
        set_speed(1.0)
        baseline = [scroll_increment() for _ in range(2000)]
        set_speed(0.25)
        slow = [scroll_increment() for _ in range(2000)]
        baseline_mean = sum(baseline) / len(baseline)
        slow_mean = sum(slow) / len(slow)
        ratio = slow_mean / baseline_mean
        assert 0.85 <= ratio <= 1.15, f"Expected ~1.0x, got {ratio:.2f}x"

    def test_lightning_speed_action_delay_range(self):
        """At 4.0x speed, delays should be much smaller."""
        from src.browser.timing import action_delay, set_speed
        set_speed(4.0)
        samples = [action_delay() for _ in range(500)]
        mean = sum(samples) / len(samples)
        assert mean < 0.15, f"Expected mean < 0.15 at 4.0x speed, got {mean:.3f}"

    def test_stealth_speed_action_delay_range(self):
        """At 0.25x speed, delays should be much larger."""
        from src.browser.timing import action_delay, set_speed
        set_speed(0.25)
        samples = [action_delay() for _ in range(500)]
        mean = sum(samples) / len(samples)
        assert mean > 0.6, f"Expected mean > 0.6 at 0.25x speed, got {mean:.3f}"


# ── Browser settings endpoint tests ──────────────────────────────────────


class TestBrowserSettingsEndpoint:
    """Tests for GET/POST /browser/settings on the browser service."""

    def setup_method(self):
        from src.browser.timing import set_speed
        set_speed(1.0)

    def teardown_method(self):
        from src.browser.timing import set_speed
        set_speed(1.0)

    def test_get_settings_default(self):
        """GET /browser/settings should return default speed=1.0."""
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/browser/settings")
        assert resp.status_code == 200
        assert resp.json()["speed"] == 1.0

    def test_set_settings(self):
        """POST /browser/settings should update the speed."""
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager
        from src.browser.timing import get_speed
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.post("/browser/settings", json={"speed": 2.5})
        assert resp.status_code == 200
        assert resp.json()["speed"] == 2.5
        assert get_speed() == 2.5

    def test_set_settings_clamped(self):
        """POST /browser/settings should clamp out-of-range values."""
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager
        from src.browser.timing import get_speed
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.post("/browser/settings", json={"speed": 100.0})
        assert resp.status_code == 200
        assert resp.json()["speed"] == 4.0  # clamped to max
        assert get_speed() == 4.0


# ── Dead code removal verification ────────────────────────────────────────


class TestStealthDeadCodeRemoved:
    """Verify resolution dead code was removed from stealth.py."""

    def test_pick_resolution_removed(self):
        import src.browser.stealth as stealth
        assert not hasattr(stealth, "_pick_resolution")
        assert not hasattr(stealth, "_WINDOWS_RESOLUTIONS")
        assert not hasattr(stealth, "_MACOS_RESOLUTIONS")
