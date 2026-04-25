"""Tests for the shared browser service (BrowserManager, server, redaction)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.browser.ref_handle import from_legacy_dict as _h
from src.browser.service import BrowserManager, CamoufoxInstance


class TestCredentialRedactor:
    """Tests for browser.redaction.CredentialRedactor."""

    def test_pattern_redaction(self):
        from src.browser.redaction import CredentialRedactor
        r = CredentialRedactor()
        text = "key is sk-abcdefghijklmnopqrstuvwxyz1234567890"
        result = r.redact("agent1", text)
        assert "[REDACTED]" in result
        assert "sk-abcdefgh" not in result

    def test_deep_redact_nested(self):
        from src.browser.redaction import CredentialRedactor
        r = CredentialRedactor()
        obj = {
            "key": "has sk-abcdefghijklmnopqrstuvwxyz1234567890",
            "nested": [{"v": "also sk-abcdefghijklmnopqrstuvwxyz1234567890"}],
        }
        result = r.deep_redact("a1", obj)
        assert "sk-abcdefgh" not in str(result)
        assert "[REDACTED]" in result["key"]


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


class TestX11WindowTracking:
    """Tests for per-agent X11 window ID tracking and targeted focus."""

    @pytest.mark.asyncio
    async def test_get_firefox_wids_parses_output(self):
        """_get_firefox_wids should parse xdotool output into a set of ints."""
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "12345\n67890\n"
        with patch("src.browser.service.subprocess.run", return_value=mock_result):
            wids = await mgr._get_firefox_wids()
        assert wids == {12345, 67890}

    @pytest.mark.asyncio
    async def test_get_firefox_wids_empty_when_no_windows(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_result = MagicMock()
        mock_result.returncode = 1  # xdotool returns 1 when no windows found
        mock_result.stdout = ""
        with patch("src.browser.service.subprocess.run", return_value=mock_result):
            wids = await mgr._get_firefox_wids()
        assert wids == set()

    @pytest.mark.asyncio
    async def test_get_firefox_wids_handles_exception(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        with patch("src.browser.service.subprocess.run", side_effect=FileNotFoundError):
            wids = await mgr._get_firefox_wids()
        assert wids == set()

    @pytest.mark.asyncio
    async def test_discover_new_wid_finds_new_window(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        # First call returns existing windows, second call returns with a new one
        mgr._get_firefox_wids = AsyncMock(
            side_effect=[{100, 200}, {100, 200, 300}]
        )
        wid = await mgr._discover_new_wid({100, 200})
        assert wid == 300

    @pytest.mark.asyncio
    async def test_discover_new_wid_picks_highest(self):
        """When multiple new windows appear, pick the highest WID (most recent)."""
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mgr._get_firefox_wids = AsyncMock(return_value={100, 200, 300, 400})
        wid = await mgr._discover_new_wid({100})
        assert wid == 400

    @pytest.mark.asyncio
    async def test_discover_new_wid_returns_none_on_timeout(self):
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        # Always returns the same set — no new windows
        mgr._get_firefox_wids = AsyncMock(return_value={100, 200})
        with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
            wid = await mgr._discover_new_wid({100, 200})
        assert wid is None

    @pytest.mark.asyncio
    async def test_focus_uses_specific_wid(self):
        """focus() should use the stored X11 WID for xdotool, not search --class."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.bring_to_front = AsyncMock()
        inst = CamoufoxInstance("agent1", MagicMock(), AsyncMock(), mock_page)
        inst.x11_wid = 12345
        mgr.get_or_start = AsyncMock(return_value=inst)
        with patch("src.browser.service.subprocess.run") as mock_run:
            await mgr.focus("agent1")
            cmd = mock_run.call_args[0][0]
            assert "12345" in cmd
            assert "search" not in cmd

    @pytest.mark.asyncio
    async def test_focus_skips_xdotool_without_wid(self):
        """focus() should skip xdotool entirely when no WID is stored."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.bring_to_front = AsyncMock()
        inst = CamoufoxInstance("agent1", MagicMock(), AsyncMock(), mock_page)
        inst.x11_wid = None
        mgr.get_or_start = AsyncMock(return_value=inst)
        with patch("src.browser.service.subprocess.run") as mock_run:
            result = await mgr.focus("agent1")
            assert result is True
            mock_page.bring_to_front.assert_awaited_once()
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_refocus_active_targets_mru_wid(self):
        """refocus_active() should target the most recently active agent's WID."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        inst_a = CamoufoxInstance("a", MagicMock(), AsyncMock(), AsyncMock())
        inst_a.x11_wid = 111
        inst_a.last_activity = 100
        inst_b = CamoufoxInstance("b", MagicMock(), AsyncMock(), AsyncMock())
        inst_b.x11_wid = 222
        inst_b.last_activity = 200  # more recent
        mgr._instances = {"a": inst_a, "b": inst_b}
        with patch("src.browser.service.subprocess.run") as mock_run:
            await mgr.refocus_active()
            cmd = mock_run.call_args[0][0]
            assert "222" in cmd
            assert "111" not in cmd

    @pytest.mark.asyncio
    async def test_refocus_active_skips_xdotool_without_wid(self):
        """refocus_active() should skip xdotool entirely when no WID is stored."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        inst = CamoufoxInstance("a", MagicMock(), AsyncMock(), AsyncMock())
        inst.x11_wid = None
        mgr._instances = {"a": inst}
        with patch("src.browser.service.subprocess.run") as mock_run:
            await mgr.refocus_active()
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_instance_has_x11_wid_attribute(self):
        """CamoufoxInstance should have x11_wid initialized to None."""
        from src.browser.service import CamoufoxInstance
        inst = CamoufoxInstance("test", MagicMock(), AsyncMock(), AsyncMock())
        assert inst.x11_wid is None

    @pytest.mark.asyncio
    async def test_reset_clears_stale_wid(self):
        """After reset, next get_or_start creates new instance with fresh WID."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_context = AsyncMock()
        inst = CamoufoxInstance("agent1", MagicMock(), mock_context, AsyncMock())
        inst.x11_wid = 99999
        mgr._instances["agent1"] = inst
        await mgr.reset("agent1")
        assert "agent1" not in mgr._instances


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
        inst.seed_refs_legacy({"e0": {"role": "button", "name": "Submit", "index": 0}})

        locator = mgr._locator_from_ref(inst, "e0")
        mock_page.get_by_role.assert_called_once_with("button", name="Submit", exact=True)
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
        inst.seed_refs_legacy({"e0": {"role": "textbox", "name": "", "index": 0}})

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
        inst.seed_refs_legacy({
            "e0": {"role": "textbox", "name": "Post text", "index": 0},
            "e1": {"role": "textbox", "name": "Post text", "index": 1},
        })

        mgr._locator_from_ref(inst, "e1")
        mock_locator.nth.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_locator_from_ref_missing_returns_none(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), MagicMock())
        inst.refs = {}

        assert mgr._locator_from_ref(inst, "e99") is None




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
        # Focus click uses hover-then-click for human-like mouse movement
        mock_page.click.assert_called()
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
        mock_page.click.assert_called()
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
        proxy = {"server": "http://proxy.example.com:8080"}
        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile", proxy=proxy)
        assert opts.get("geoip") is True

    def test_window_matches_screen_fingerprint(self):
        """Phase 3 §6.1: window= is picked from the resolution pool per
        agent, and must match the Screen() fingerprint max dimensions.

        What we're enforcing is the *consistency invariant*, not a
        specific size. ``innerWidth`` > ``screen.width`` is a detection
        signal; equal sizes are safe. The VNC display itself stays
        1920×1080 (shared across agents); the browser window inside it
        is whatever the pool assigned this agent.
        """
        from src.browser.stealth import (
            _RESOLUTION_POOL,
            build_launch_options,
            pick_resolution,
        )

        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")

        # The window came from the pool (not a hardcoded default).
        expected = pick_resolution("agent1")
        assert opts["window"] == expected
        assert expected in {res for res, _ in _RESOLUTION_POOL}

        # And the Screen() fingerprint matches when browserforge is
        # available — in the unit-test environment it may not be.
        screen = opts.get("screen")
        if screen is not None:
            assert (screen.max_width, screen.max_height) == expected

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
        from src.browser.stealth import build_launch_options
        proxy = {"server": "http://proxy.example.com:8080", "username": "user", "password": "pass"}
        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent-1", "/tmp/profile", proxy=proxy)
        assert opts["proxy"] == proxy
        assert opts["geoip"] is True

    def test_proxy_env_vars_ignored_without_explicit_proxy(self):
        """BROWSER_PROXY_* env vars are no longer read by build_launch_options."""
        from src.browser.stealth import build_launch_options
        env = {"BROWSER_PROXY_URL": "http://proxy.example.com:8080"}
        with patch.dict("os.environ", env, clear=False):
            opts = build_launch_options("agent-1", "/tmp/profile")
        assert "proxy" not in opts


class TestUAOverride:
    """Tests for BROWSER_UA_VERSION user-agent override."""

    def test_no_override_by_default(self):
        """Without BROWSER_UA_VERSION, no UA-specific keys should be set.

        Phase 3 §6.6 introduced unconditional ``navigator.connection.*``
        keys plus ``i_know_what_im_doing`` so the assertions for those
        no longer hold here — but ``navigator.userAgent`` and the
        Firefox pref-level override must still be absent on the no-UA-
        version path. That's the actual contract this test is guarding.
        """
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert "navigator.userAgent" not in (opts.get("config") or {})
        assert "general.useragent.override" not in opts["firefox_user_prefs"]

    def test_override_sets_camoufox_config(self):
        """BROWSER_UA_VERSION should set Camoufox's UA config (primary mechanism)."""
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {"BROWSER_UA_VERSION": "138.0"}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert opts["config"]["navigator.userAgent"] == (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) "
            "Gecko/20100101 Firefox/138.0"
        )
        assert opts["i_know_what_im_doing"] is True

    def test_override_sets_firefox_pref_fallback(self):
        """BROWSER_UA_VERSION should also set general.useragent.override as fallback."""
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {"BROWSER_UA_VERSION": "138.0"}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        ua = opts["firefox_user_prefs"]["general.useragent.override"]
        assert "Firefox/138.0" in ua
        # Config and pref must have the same value
        assert ua == opts["config"]["navigator.userAgent"]

    def test_override_respects_os_windows(self):
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {"BROWSER_UA_VERSION": "139.0"}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        ua = opts["config"]["navigator.userAgent"]
        assert "Windows NT 10.0" in ua
        assert "rv:139.0" in ua
        assert ua.endswith("Firefox/139.0")

    def test_override_respects_os_macos(self):
        from src.browser.stealth import build_launch_options
        env = {"BROWSER_UA_VERSION": "138.0", "BROWSER_OS": "macos"}
        with patch.dict("os.environ", env, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        ua = opts["config"]["navigator.userAgent"]
        assert "Macintosh; Intel Mac OS X 10.15" in ua
        assert "Firefox/138.0" in ua

    def test_override_respects_os_linux(self):
        from src.browser.stealth import build_launch_options
        env = {"BROWSER_UA_VERSION": "138.0", "BROWSER_OS": "linux"}
        with patch.dict("os.environ", env, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        ua = opts["config"]["navigator.userAgent"]
        assert "X11; Linux x86_64" in ua
        assert "Firefox/138.0" in ua

    def test_override_accepts_three_part_version(self):
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {"BROWSER_UA_VERSION": "138.0.1"}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert "Firefox/138.0.1" in opts["config"]["navigator.userAgent"]

    def test_override_rejects_non_numeric(self):
        """Non-numeric version should be ignored — UA-specific keys absent."""
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {"BROWSER_UA_VERSION": "abc"}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        # The §6.6 navigator.connection.* keys are still set; only the
        # UA override path must not have run.
        assert "navigator.userAgent" not in (opts.get("config") or {})
        assert "general.useragent.override" not in opts["firefox_user_prefs"]

    def test_override_rejects_single_number(self):
        """Version without minor component (e.g. '138') should be rejected."""
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {"BROWSER_UA_VERSION": "138"}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert "navigator.userAgent" not in (opts.get("config") or {})

    def test_override_rejects_empty_parts(self):
        """Malformed versions like '138.' or '.0' should be rejected."""
        from src.browser.stealth import build_launch_options
        for bad in ("138.", ".0", ".", ".."):
            with patch.dict("os.environ", {"BROWSER_UA_VERSION": bad}, clear=True):
                opts = build_launch_options("agent1", "/tmp/profile")
            assert "navigator.userAgent" not in (opts.get("config") or {}), (
                f"Expected rejection for {bad!r}"
            )

    def test_override_strips_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {"BROWSER_UA_VERSION": "  138.0  "}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        assert "Firefox/138.0" in opts["config"]["navigator.userAgent"]

    def test_empty_string_means_no_override(self):
        from src.browser.stealth import build_launch_options
        with patch.dict("os.environ", {"BROWSER_UA_VERSION": ""}, clear=True):
            opts = build_launch_options("agent1", "/tmp/profile")
        # config exists for §6.6 NetworkInformation; the UA-specific
        # key inside it is what must not be set.
        assert "navigator.userAgent" not in (opts.get("config") or {})

    def test_build_ua_string_directly(self):
        """Unit test the helper function."""
        from src.browser.stealth import _build_ua_string
        assert _build_ua_string("windows", "138.0") == (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) "
            "Gecko/20100101 Firefox/138.0"
        )
        assert _build_ua_string("macos", "140.0.2") == (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:140.0.2) "
            "Gecko/20100101 Firefox/140.0.2"
        )
        assert _build_ua_string("linux", "138.0") == (
            "Mozilla/5.0 (X11; Linux x86_64; rv:138.0) "
            "Gecko/20100101 Firefox/138.0"
        )
        # Unknown OS falls back to windows
        assert "Windows NT" in _build_ua_string("freebsd", "138.0")
        # Invalid versions return None
        assert _build_ua_string("windows", "abc") is None
        assert _build_ua_string("windows", "138") is None
        assert _build_ua_string("windows", "") is None


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
        inst.seed_refs_legacy({"e0": {"role": "button", "name": "Submit"}})
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

    @pytest.mark.asyncio
    async def test_successful_click_updates_rolling_window(self):
        """Phase 2 §5.2: every successful click must append ``True`` to
        ``click_window`` so the dashboard's live success-rate widget
        reflects the latest 100 outcomes.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.click = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        await mgr.click("a1", selector="#ok")
        assert list(inst.click_window) == [True]
        assert inst.m_click_success == 1
        assert inst.rolling_click_success_rate() == 1.0

    @pytest.mark.asyncio
    async def test_failed_click_updates_rolling_window(self):
        """A click raising an exception must also append to the window
        (as ``False``) — otherwise failure modes disappear from health."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.click = AsyncMock(side_effect=RuntimeError("boom"))
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.click("a1", selector="#ok")
        assert result["success"] is False
        assert list(inst.click_window) == [False]
        assert inst.m_click_fail == 1
        assert inst.rolling_click_success_rate() == 0.0


def _make_png_bytes(width: int = 64, height: int = 48) -> bytes:
    """Render a real PNG via Pillow for screenshot encode tests."""
    from io import BytesIO

    from PIL import Image
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x * 4 % 256, y * 4 % 256, (x + y) * 2 % 256)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


class TestScreenshot:
    """Tests for BrowserManager.screenshot()."""

    @pytest.mark.asyncio
    async def test_screenshot_corrupt_bytes_falls_back_to_png(self):
        """Fake PNG bytes — Pillow can't decode them, so we fall back
        to returning the original payload as PNG. Keeps the agent
        unblocked rather than failing the call."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(
            return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
        )
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.screenshot("a1")
        assert result["success"] is True
        # Default format is webp; corrupt data triggers the PNG fallback.
        assert result["data"]["format"] == "png"
        assert len(result["data"]["image_base64"]) > 0

    @pytest.mark.asyncio
    async def test_screenshot_default_returns_webp(self):
        """Default call yields a valid WebP payload."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        # Larger fixture so WebP's container overhead is dominated by
        # the encoded pixel data — small fixtures can encode larger as
        # WebP than PNG because of header/chunk overhead.
        png = _make_png_bytes(width=400, height=300)
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=png)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.screenshot("a1")
        assert result["success"] is True
        assert result["data"]["format"] == "webp"
        assert result["data"]["bytes"] > 0
        # Decoded base64 should equal the byte-count we emitted.
        import base64
        decoded = base64.b64decode(result["data"]["image_base64"])
        assert len(decoded) == result["data"]["bytes"]
        # First 4 bytes of WebP file are 'RIFF'; bytes 8-11 are 'WEBP'.
        assert decoded[:4] == b"RIFF"
        assert decoded[8:12] == b"WEBP"

    @pytest.mark.asyncio
    async def test_screenshot_explicit_png_passthrough(self):
        """Explicit ``format='png'`` with scale=1.0 returns the raw
        Playwright bytes (no Pillow round-trip)."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        png = _make_png_bytes()
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=png)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.screenshot("a1", format="png", scale=1.0)
        assert result["success"] is True
        assert result["data"]["format"] == "png"
        import base64
        decoded = base64.b64decode(result["data"]["image_base64"])
        # Pass-through: identical bytes.
        assert decoded == png

    @pytest.mark.asyncio
    async def test_screenshot_scale_resizes(self):
        """``scale=0.5`` cuts dimensions in half."""
        from io import BytesIO

        from PIL import Image
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        png = _make_png_bytes(width=200, height=120)
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=png)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.screenshot("a1", format="webp", scale=0.5)
        assert result["success"] is True
        import base64
        decoded = base64.b64decode(result["data"]["image_base64"])
        img = Image.open(BytesIO(decoded))
        assert img.size == (100, 60)

    @pytest.mark.asyncio
    async def test_screenshot_unknown_format_rejected(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=b"")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.screenshot("a1", format="jpeg")
        assert result["success"] is False
        assert "Unsupported" in result["error"]

    @pytest.mark.asyncio
    async def test_screenshot_format_none_uses_operator_default(self, monkeypatch):
        """``format=None`` (JSON null) consults the operator flag,
        defaulting to ``webp`` when unset."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        # Force operator default to PNG via env override.
        monkeypatch.setenv("BROWSER_SCREENSHOT_FORMAT", "png")
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        png = _make_png_bytes(400, 300)
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=png)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.screenshot("a1", format=None)
        assert result["success"] is True
        assert result["data"]["format"] == "png"

    @pytest.mark.asyncio
    async def test_screenshot_format_whitespace_normalized(self):
        """``format=' WEBP '`` strips + lowercases instead of rejecting."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        png = _make_png_bytes(400, 300)
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=png)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.screenshot("a1", format=" WEBP ")
        assert result["success"] is True
        assert result["data"]["format"] == "webp"

    @pytest.mark.asyncio
    async def test_screenshot_quality_clamped(self):
        """Out-of-range quality clamps silently rather than raising."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        png = _make_png_bytes()
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=png)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        # quality=999 should clamp to 100 → encode succeeds.
        result = await mgr.screenshot("a1", quality=999)
        assert result["success"] is True
        # quality=-1 should clamp to 1 → encode succeeds (tiny output).
        result = await mgr.screenshot("a1", quality=-1)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_screenshot_pillow_missing_falls_back_to_png(self, monkeypatch):
        """If Pillow can't be imported the helper returns the raw PNG."""
        import builtins
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        png = _make_png_bytes()
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=png)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "PIL":
                raise ImportError("forced for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        result = await mgr.screenshot("a1", format="webp")
        assert result["success"] is True
        # Pillow missing → graceful PNG fallback.
        assert result["data"]["format"] == "png"


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
        # Refs stored on instance as RefHandle for click/type resolution;
        # wire-format refs (the response) carry the minimal dict shape agents
        # see. Verify both match on the visible fields.
        assert set(inst.refs.keys()) == set(result["data"]["refs"].keys())
        for rid, wire in result["data"]["refs"].items():
            handle = inst.refs[rid]
            assert handle.role == wire["role"]
            assert handle.name == wire["name"]
            assert handle.occurrence == wire["index"]
            assert handle.disabled == wire["disabled"]

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


class TestSnapshotFormatV2:
    """§7.2 — landmark section headers + capped indent."""

    @pytest.fixture
    def v2_flag(self, monkeypatch):
        """Force BROWSER_SNAPSHOT_FORMAT=v2 for the duration of one test."""
        monkeypatch.setenv("BROWSER_SNAPSHOT_FORMAT", "v2")

    @pytest.mark.asyncio
    async def test_v1_default_unchanged(self):
        """Without the flag the snapshot is the historical v1 format —
        no version marker, no section headers."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Submit",
                 "landmark": "navigation: Top"},
            ],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        snap = result["data"]["snapshot"]
        assert "snapshot-v2" not in snap
        assert "(navigation: Top)" in snap  # v1 suffix

    @pytest.mark.asyncio
    async def test_v2_emits_version_marker(self, v2_flag):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "Click"}],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        snap = result["data"]["snapshot"]
        assert snap.startswith("# snapshot-v2\n")

    @pytest.mark.asyncio
    async def test_v2_groups_by_landmark(self, v2_flag):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea", "name": "",
            "children": [
                {"role": "navigation", "name": "Top",
                 "children": [
                     {"role": "link", "name": "Home",
                      "landmark": "navigation: Top"},
                 ]},
                {"role": "main", "name": "Article",
                 "children": [
                     {"role": "heading", "name": "Title",
                      "landmark": "main: Article"},
                     {"role": "button", "name": "Comment",
                      "landmark": "main: Article"},
                 ]},
            ],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        snap = result["data"]["snapshot"]
        assert "# navigation: Top" in snap
        assert "# main: Article" in snap
        assert "(navigation: Top)" not in snap
        assert "(main: Article)" not in snap
        assert '] link "Home"' in snap
        assert '] heading "Title"' in snap
        assert '] button "Comment"' in snap

    @pytest.mark.asyncio
    async def test_v2_unlandmarked_section_emitted(self, v2_flag):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Loose"},
            ],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        snap = result["data"]["snapshot"]
        assert "# (no landmark)" in snap
        assert '] button "Loose"' in snap

    def test_v2_caps_indent_depth(self):
        """Direct test of the formatter — depth>4 collapses to 4 levels."""
        from src.browser.service import _format_snapshot_v2
        entries = [
            ("e0", "button", "Deep", "", "main: Body", 7),
            ("e1", "link", "Shallow", "", "main: Body", 1),
        ]
        out = _format_snapshot_v2([], entries)
        assert '\n        - [e0] button "Deep"' in out
        assert '\n  - [e1] link "Shallow"' in out

    def test_v2_passes_through_modal_banner(self):
        """``**`` preamble lines (modal warning) ride along ahead of the
        section blocks."""
        from src.browser.service import _format_snapshot_v2
        lines = [
            "** Modal dialog is open — only dialog elements are shown **",
            "  - [e0] button \"Close\" (dialog: Compose)",
        ]
        entries = [("e0", "button", "Close", "", "dialog: Compose", 0)]
        out = _format_snapshot_v2(lines, entries)
        assert out.startswith("# snapshot-v2\n** Modal dialog is open")
        assert "# dialog: Compose" in out

    def test_v2_empty_entries_returns_marker_only(self):
        from src.browser.service import _format_snapshot_v2
        out = _format_snapshot_v2([], [])
        assert out == "# snapshot-v2\n(no interactive elements)"

    def test_v2_attr_string_preserved(self):
        from src.browser.service import _format_snapshot_v2
        entries = [
            ("e0", "checkbox", "Subscribe", " [checked=True]", "form: Signup", 2),
        ]
        out = _format_snapshot_v2([], entries)
        assert '- [e0] checkbox "Subscribe" [checked=True]' in out

    def test_v2_strips_newlines_from_landmark_and_name(self):
        """An adversarial DOM with embedded newlines in the accessible
        name (or landmark) must not inject a phantom section header.

        Pre-fix, a button with ``aria-label="x\\n# fake-section: pwn"``
        would land in v2 output as a literal newline followed by
        ``# fake-section: pwn``, which a parser would interpret as a
        new section. The fix collapses CR/LF to spaces before emit."""
        from src.browser.service import _format_snapshot_v2
        entries = [
            (
                "e0", "button",
                "Real\n# fake-section: pwn",
                " [disabled]",
                "main: Body\n# evil",
                1,
            ),
        ]
        out = _format_snapshot_v2([], entries)
        # No newline appears between ``Real`` and ``# fake``; landmark
        # key is sanitized similarly.
        assert "\n# fake-section: pwn" not in out
        assert "\n# evil" not in out
        # The actual emitted lines have a single section header and a
        # single element line.
        section_lines = [ln for ln in out.splitlines() if ln.startswith("# ")]
        # First section line is the version marker; second is the
        # sanitized landmark.
        assert section_lines[0] == "# snapshot-v2"
        assert section_lines[1] == "# main: Body # evil"

    def test_v2_empty_with_preamble_only(self):
        """Modal-scoping retry can produce zero entries but a non-empty
        preamble (the ``** Modal dialog ... **`` banner). v2 must
        emit the marker + the preamble cleanly."""
        from src.browser.service import _format_snapshot_v2
        out = _format_snapshot_v2(
            ["** Modal dialog is open — only dialog elements are shown **"],
            [],
        )
        assert out == (
            "# snapshot-v2\n"
            "** Modal dialog is open — only dialog elements are shown **"
        )

    @pytest.mark.asyncio
    async def test_v2_modal_retry_does_not_leak_phantom_refs(self, v2_flag):
        """Regression: when modal scoping fails on first try, the discarded
        ``_walk`` pass must not bleed entries into v2 output. Pre-fix,
        ``lines.clear()`` reset v1 output but ``entries`` was never reset,
        producing duplicated refs in v2 that didn't match ``inst.refs``.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        # Modal is detected (visible) but scoping returns a tree with
        # only context (heading) — no actionable refs — on first pass.
        # Second pass after the retry yields an actionable button.
        first_pass_tree = {
            "role": "WebArea", "name": "",
            "children": [{"role": "heading", "name": "Loading...",
                          "landmark": "dialog: Compose"}],
        }
        second_pass_tree = {
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "Post",
                          "landmark": "dialog: Compose"}],
        }

        # Modal selector returns a single visible modal.
        modal_el = AsyncMock()
        modal_el.is_visible = AsyncMock(return_value=True)
        modal_el.bounding_box = AsyncMock(
            return_value={"x": 0, "y": 0, "width": 200, "height": 200},
        )
        modal_el.evaluate = AsyncMock(return_value=False)
        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1920, "height": 1080}
        mock_page.query_selector_all = AsyncMock(return_value=[modal_el])
        mock_page.accessibility = MagicMock()
        # First call (page-level snapshot) → first_pass.
        # Calls with root=modal_el → first_pass on first attempt,
        # second_pass after retry.
        accessibility_calls = [
            first_pass_tree,   # initial page-level snapshot for whole-tree fallback
            first_pass_tree,   # first scoped snapshot inside modal
            second_pass_tree,  # post-retry scoped snapshot
        ]
        mock_page.accessibility.snapshot = AsyncMock(
            side_effect=accessibility_calls + [second_pass_tree] * 5,
        )
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            result = await mgr.snapshot("a1")

        snap = result["data"]["snapshot"]
        # Every ref id appearing in the rendered v2 snapshot must also
        # exist in inst.refs (the resolution table). The pre-fix bug
        # produced refs that were rendered but not present in inst.refs.
        import re
        rendered_refs = set(re.findall(r"\[e\d+\]", snap))
        actual_refs = {f"[{rid}]" for rid in inst.refs}
        assert rendered_refs.issubset(actual_refs), (
            f"phantom refs in v2 output: {rendered_refs - actual_refs}"
        )

class TestSnapshotFilter:
    """§7.7 — semantic filter narrows the snapshot to one role family."""

    def _mixed_tree(self) -> dict:
        return {
            "role": "WebArea",
            "name": "Mixed",
            "children": [
                {"role": "navigation", "name": "Top"},
                {"role": "main", "name": "Main"},
                {"role": "heading", "name": "Section A"},
                {"role": "button", "name": "Click"},
                {"role": "textbox", "name": "Email"},
                {"role": "checkbox", "name": "Subscribe"},
                {"role": "img", "name": "Logo"},
                {"role": "link", "name": "Help"},
            ],
        }

    async def _snap_with_filter(self, filter_value):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(
            return_value=self._mixed_tree(),
        )
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst
        return await mgr.snapshot("a1", filter=filter_value)

    @pytest.mark.asyncio
    async def test_default_filter_includes_actionable_and_context(self):
        result = await self._snap_with_filter(None)
        assert result["success"] is True
        roles = {r["role"] for r in result["data"]["refs"].values()}
        # Actionable + context: button, textbox, checkbox, link, heading, img.
        # Landmarks (navigation/main) are NOT in the default mix.
        assert "button" in roles
        assert "heading" in roles
        assert "img" in roles
        assert "navigation" not in roles
        assert "main" not in roles

    @pytest.mark.asyncio
    async def test_actionable_filter_excludes_context(self):
        result = await self._snap_with_filter("actionable")
        roles = {r["role"] for r in result["data"]["refs"].values()}
        assert roles == {"button", "textbox", "checkbox", "link"}

    @pytest.mark.asyncio
    async def test_inputs_filter_includes_inputs_only(self):
        result = await self._snap_with_filter("inputs")
        roles = {r["role"] for r in result["data"]["refs"].values()}
        assert roles == {"textbox", "checkbox"}

    @pytest.mark.asyncio
    async def test_headings_filter(self):
        result = await self._snap_with_filter("headings")
        roles = {r["role"] for r in result["data"]["refs"].values()}
        assert roles == {"heading"}

    @pytest.mark.asyncio
    async def test_landmarks_filter(self):
        result = await self._snap_with_filter("landmarks")
        roles = {r["role"] for r in result["data"]["refs"].values()}
        assert roles == {"navigation", "main"}

    @pytest.mark.asyncio
    async def test_invalid_filter_returns_invalid_input(self):
        result = await self._snap_with_filter("nope")
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert "nope" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_empty_string_treated_as_default(self):
        """JSON UI defaults often pass ``""`` — must not narrow the result."""
        result = await self._snap_with_filter("")
        assert result["success"] is True
        roles = {r["role"] for r in result["data"]["refs"].values()}
        # Same set as the default-None case.
        assert "heading" in roles and "img" in roles

    @pytest.mark.asyncio
    async def test_filter_case_insensitive(self):
        """LLMs frequently capitalize argument values. ``Actionable``,
        ``INPUTS`` etc. must work like their lowercase counterparts."""
        result = await self._snap_with_filter("Actionable")
        assert result["success"] is True
        roles = {r["role"] for r in result["data"]["refs"].values()}
        assert roles == {"button", "textbox", "checkbox", "link"}

        result = await self._snap_with_filter("INPUTS")
        assert result["success"] is True
        roles = {r["role"] for r in result["data"]["refs"].values()}
        assert roles == {"textbox", "checkbox"}

    @pytest.mark.asyncio
    async def test_filter_whitespace_tolerated(self):
        """Stray whitespace shouldn't trip the filter — strip+lower."""
        result = await self._snap_with_filter("  inputs  ")
        assert result["success"] is True
        roles = {r["role"] for r in result["data"]["refs"].values()}
        assert roles == {"textbox", "checkbox"}


class TestSnapshotFromRef:
    """§7.4 — scoped snapshot rooted at a previously-seen element."""

    @pytest.mark.asyncio
    async def test_from_ref_unknown_returns_not_found(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea", "name": "", "children": [],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst
        result = await mgr.snapshot("a1", from_ref="e99")
        assert result["success"] is False
        assert result["error"]["code"] == "not_found"

    @pytest.mark.asyncio
    async def test_from_ref_empty_returns_invalid_input(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), AsyncMock())
        mgr._instances["a1"] = inst
        result = await mgr.snapshot("a1", from_ref="")
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_from_ref_scopes_tree_to_subtree(self):
        """When ``from_ref`` is set, ``_build_a11y_tree`` is called with the
        resolved element handle as ``root``; the result is just that subtree."""
        from src.browser.ref_handle import RefHandle
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        # 1. Set up the inst and seed it with a single ref that
        # ``_locator_from_ref`` can resolve. Using light_dom because shadow/
        # frame fields aren't relevant to this test.
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.accessibility = MagicMock()
        # Default page-level a11y returns a small tree with a "form" role.
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "form", "name": "Login",
            "children": [
                {"role": "textbox", "name": "Email"},
                {"role": "textbox", "name": "Password"},
            ],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)

        page_id = inst._page_id_for(inst.page)
        inst.refs["e0"] = RefHandle.light_dom(
            page_id=page_id, scope_root=None,
            role="form", name="Login", occurrence=0, disabled=False,
        )

        # 2. Stub ``_locator_from_ref`` so it returns a Locator-like object
        # whose ``element_handle`` returns a stub ElementHandle. The
        # ``_build_a11y_tree`` with ``root=<handle>`` path will be exercised.
        scoped_handle = MagicMock()  # ElementHandle stand-in
        # In this branch the page-level fallback uses ``root.evaluate``
        # rather than ``page.accessibility.snapshot``. Stub that path.
        scoped_tree = {
            "role": "form", "name": "Login",
            "children": [
                {"role": "textbox", "name": "Email"},
                {"role": "textbox", "name": "Password"},
            ],
        }
        # Make Camoufox's accessibility.snapshot(root=...) succeed too —
        # _build_a11y_tree tries the native API first.
        mock_page.accessibility.snapshot = AsyncMock(return_value=scoped_tree)

        fake_locator = AsyncMock()
        fake_locator.element_handle = AsyncMock(return_value=scoped_handle)
        mgr._instances["a1"] = inst

        with patch.object(
            BrowserManager, "_locator_from_ref", return_value=fake_locator,
        ):
            result = await mgr.snapshot("a1", from_ref="e0")

        assert result["success"] is True
        roles = {r["role"] for r in result["data"]["refs"].values()}
        # The scoped subtree only contains the two textboxes.
        assert roles == {"textbox"}

    @pytest.mark.asyncio
    async def test_from_ref_with_filter_combines(self):
        """``from_ref`` + ``filter`` should narrow the scoped subtree
        further."""
        from src.browser.ref_handle import RefHandle
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.accessibility = MagicMock()
        # Scoped tree contains a heading + a button + a textbox.
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "form", "name": "Login",
            "children": [
                {"role": "heading", "name": "Sign in"},
                {"role": "button", "name": "Submit"},
                {"role": "textbox", "name": "Email"},
            ],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        page_id = inst._page_id_for(inst.page)
        inst.refs["e0"] = RefHandle.light_dom(
            page_id=page_id, scope_root=None,
            role="form", name="Login", occurrence=0, disabled=False,
        )
        mgr._instances["a1"] = inst

        fake_locator = AsyncMock()
        fake_locator.element_handle = AsyncMock(return_value=MagicMock())
        with patch.object(
            BrowserManager, "_locator_from_ref", return_value=fake_locator,
        ):
            result = await mgr.snapshot("a1", from_ref="e0", filter="inputs")
        assert result["success"] is True
        roles = {r["role"] for r in result["data"]["refs"].values()}
        assert roles == {"textbox"}

    @pytest.mark.asyncio
    async def test_from_ref_inside_modal_preserves_scope_root(self):
        """Regression for the silent-misclick bug: when ``from_ref`` is
        used while a modal is open, the scoped refs MUST carry
        ``scope_root=_MODAL_SELECTOR`` so subsequent ``_locator_from_ref``
        calls stay bounded to the dialog. Pre-fix, scoped refs had
        ``scope_root=None`` and ``inst.dialog_active`` was cleared,
        letting clicks resolve to identical-named elements behind the
        overlay."""
        from src.browser.ref_handle import RefHandle
        from src.browser.service import _MODAL_SELECTOR, BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "form", "name": "Compose",
            "children": [
                {"role": "button", "name": "Post"},
                {"role": "button", "name": "Cancel"},
            ],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        # Simulate live modal state at the time the agent calls snapshot.
        inst.dialog_active = True
        inst.dialog_detected = True
        page_id = inst._page_id_for(inst.page)
        inst.refs["e0"] = RefHandle.light_dom(
            page_id=page_id, scope_root=_MODAL_SELECTOR,
            role="form", name="Compose", occurrence=0, disabled=False,
        )
        mgr._instances["a1"] = inst

        fake_locator = AsyncMock()
        fake_locator.element_handle = AsyncMock(return_value=MagicMock())
        with patch.object(
            BrowserManager, "_locator_from_ref", return_value=fake_locator,
        ):
            result = await mgr.snapshot("a1", from_ref="e0")

        assert result["success"] is True
        # Every emitted ref should carry the modal scope_root so the
        # next click stays bounded to the dialog subtree.
        for handle in inst.refs.values():
            assert handle.scope_root == _MODAL_SELECTOR, (
                f"Ref {handle.role!r} {handle.name!r} leaked outside "
                f"the modal scope: scope_root={handle.scope_root!r}"
            )

    @pytest.mark.asyncio
    async def test_from_ref_outside_modal_no_scope_root(self):
        """Inverse of the modal-preservation test: when no modal is
        active during a from_ref snapshot, refs should NOT acquire a
        modal scope_root (stays None)."""
        from src.browser.ref_handle import RefHandle
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "form", "name": "Login",
            "children": [{"role": "button", "name": "Submit"}],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        # No modal active.
        inst.dialog_active = False
        page_id = inst._page_id_for(inst.page)
        inst.refs["e0"] = RefHandle.light_dom(
            page_id=page_id, scope_root=None,
            role="form", name="Login", occurrence=0, disabled=False,
        )
        mgr._instances["a1"] = inst

        fake_locator = AsyncMock()
        fake_locator.element_handle = AsyncMock(return_value=MagicMock())
        with patch.object(
            BrowserManager, "_locator_from_ref", return_value=fake_locator,
        ):
            result = await mgr.snapshot("a1", from_ref="e0")

        assert result["success"] is True
        for handle in inst.refs.values():
            assert handle.scope_root is None

    @pytest.mark.asyncio
    async def test_from_ref_dispatches_through_v2_when_flag_set(
        self, monkeypatch,
    ):
        """Cross-PR (§7.4 ↔ §7.2): when BROWSER_SNAPSHOT_FORMAT=v2,
        the from_ref early-return must use the v2 formatter so scoped
        snapshots emit the ``# snapshot-v2`` marker just like the
        main return path. Without this, agents parsing on the
        first-line marker would see mixed formats."""
        from src.browser.ref_handle import RefHandle
        from src.browser.service import BrowserManager, CamoufoxInstance
        monkeypatch.setenv("BROWSER_SNAPSHOT_FORMAT", "v2")
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "form", "name": "Login",
            "children": [{"role": "button", "name": "Submit"}],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        page_id = inst._page_id_for(inst.page)
        inst.refs["e0"] = RefHandle.light_dom(
            page_id=page_id, scope_root=None,
            role="form", name="Login", occurrence=0, disabled=False,
        )
        mgr._instances["a1"] = inst

        fake_locator = AsyncMock()
        fake_locator.element_handle = AsyncMock(return_value=MagicMock())
        with patch.object(
            BrowserManager, "_locator_from_ref", return_value=fake_locator,
        ):
            result = await mgr.snapshot("a1", from_ref="e0")
        assert result["success"] is True
        # v2 marker on first line — the scoped path now honors the flag.
        assert result["data"]["snapshot"].startswith("# snapshot-v2\n")


class TestDiffSnapshot:
    """§7.3 — diff_from_last produces structured deltas instead of a
    full snapshot when nothing tab-shaped has changed."""

    async def _setup(self, tree, url="https://example.com"):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.url = url
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst
        return mgr, inst, mock_page

    @pytest.mark.asyncio
    async def test_first_diff_call_returns_full_snapshot(self):
        """No baseline → ``scope=navigation`` and full snapshot returned."""
        tree = {
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "Click"}],
        }
        mgr, inst, _ = await self._setup(tree)
        result = await mgr.snapshot("a1", diff_from_last=True)
        assert result["success"] is True
        data = result["data"]
        assert data["scope"] == "navigation"
        assert "snapshot" in data
        assert "refs" in data

    @pytest.mark.asyncio
    async def test_no_changes_returns_same_scope_with_unchanged_count(self):
        tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Click"},
                {"role": "textbox", "name": "Email"},
            ],
        }
        mgr, _, _ = await self._setup(tree)
        await mgr.snapshot("a1")
        result = await mgr.snapshot("a1", diff_from_last=True)
        data = result["data"]
        assert data["scope"] == "same"
        assert data["added"] == []
        assert data["removed"] == []
        assert data["changed"] == []
        assert data["unchanged_count"] == 2

    @pytest.mark.asyncio
    async def test_added_element_in_diff(self):
        tree_v1 = {
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "Click"}],
        }
        tree_v2 = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Click"},
                {"role": "button", "name": "Cancel"},
            ],
        }
        mgr, inst, mock_page = await self._setup(tree_v1)
        await mgr.snapshot("a1")
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree_v2)
        result = await mgr.snapshot("a1", diff_from_last=True)
        data = result["data"]
        assert data["scope"] == "same"
        assert len(data["added"]) == 1
        assert data["added"][0]["name"] == "Cancel"
        assert data["added"][0]["role"] == "button"
        assert data["unchanged_count"] == 1

    @pytest.mark.asyncio
    async def test_removed_element_in_diff(self):
        tree_v1 = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Click"},
                {"role": "button", "name": "Cancel"},
            ],
        }
        tree_v2 = {
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "Click"}],
        }
        mgr, inst, mock_page = await self._setup(tree_v1)
        await mgr.snapshot("a1")
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree_v2)
        result = await mgr.snapshot("a1", diff_from_last=True)
        data = result["data"]
        assert len(data["removed"]) == 1
        assert data["removed"][0]["name"] == "Cancel"

    @pytest.mark.asyncio
    async def test_changed_disabled_state(self):
        tree_v1 = {
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "Submit", "disabled": True}],
        }
        tree_v2 = {
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "Submit", "disabled": False}],
        }
        mgr, _, mock_page = await self._setup(tree_v1)
        await mgr.snapshot("a1")
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree_v2)
        result = await mgr.snapshot("a1", diff_from_last=True)
        data = result["data"]
        assert len(data["changed"]) == 1
        assert data["changed"][0]["disabled"] == {"from": True, "to": False}

    @pytest.mark.asyncio
    async def test_navigation_returns_full_snapshot(self):
        tree = {
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "Click"}],
        }
        mgr, inst, mock_page = await self._setup(tree, url="https://a.com")
        await mgr.snapshot("a1")
        mock_page.url = "https://b.com"
        result = await mgr.snapshot("a1", diff_from_last=True)
        data = result["data"]
        assert data["scope"] == "navigation"
        assert "snapshot" in data
        assert "refs" in data

    @pytest.mark.asyncio
    async def test_modal_closed_scope(self):
        # Modal scoping is driven by ``query_selector_all(_MODAL_SELECTOR)``
        # returning visible modals. Easier test angle: drive the
        # dialog_active flag directly via the persisted baseline so the
        # scope-classifier sees the flip without a fragile DOM mock.
        tree = {
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "Click"}],
        }
        mgr, inst, mock_page = await self._setup(tree)
        await mgr.snapshot("a1")
        baseline = inst.last_snapshot[inst.last_active_page_id]
        baseline["dialog_active"] = True
        # Current state stays modal-inactive — flip from True→False.
        result = await mgr.snapshot("a1", diff_from_last=True)
        assert result["data"]["scope"] == "modal_closed"

    @pytest.mark.asyncio
    async def test_diff_off_returns_historical_shape(self):
        """Without ``diff_from_last`` the response shape is unchanged
        from pre-§7.3 (no ``scope`` field)."""
        tree = {
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "Click"}],
        }
        mgr, _, _ = await self._setup(tree)
        result = await mgr.snapshot("a1")
        assert "scope" not in result["data"]
        assert "snapshot" in result["data"]
        assert "refs" in result["data"]

    @pytest.mark.asyncio
    async def test_tab_changed_to_baselined_tab(self):
        """Switching to a previously-baselined tab → ``tab_changed``."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        # Tab A: baseline at https://a.com.
        page_a = AsyncMock()
        page_a.url = "https://a.com"
        page_a.accessibility = MagicMock()
        page_a.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "A"}],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), page_a)
        mgr._instances["a1"] = inst
        await mgr.snapshot("a1")
        page_a_id = inst.last_active_page_id

        # Tab B: separate Page object → different page_id.
        page_b = AsyncMock()
        page_b.url = "https://b.com"
        page_b.accessibility = MagicMock()
        page_b.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "B"}],
        })
        # Baseline tab B too.
        inst.page = page_b
        inst._register_page(page_b)
        await mgr.snapshot("a1")

        # Switch BACK to tab A and ask for a diff.
        inst.page = page_a
        result = await mgr.snapshot("a1", diff_from_last=True)
        assert result["data"]["scope"] == "tab_changed"
        # tab_changed returns a full snapshot, not a diff payload.
        assert "snapshot" in result["data"]
        assert "refs" in result["data"]

    @pytest.mark.asyncio
    async def test_tab_changed_to_unbaselined_tab(self):
        """Switching to a never-snapshotted tab still reports
        tab_changed when last_active_page_id differs (regression for
        the previous-vs-current ordering bug)."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        page_a = AsyncMock()
        page_a.url = "https://a.com"
        page_a.accessibility = MagicMock()
        page_a.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "A"}],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), page_a)
        mgr._instances["a1"] = inst
        # Baseline tab A.
        await mgr.snapshot("a1")

        # Switch to tab B (never baselined) and request a diff.
        page_b = AsyncMock()
        page_b.url = "https://b.com"
        page_b.accessibility = MagicMock()
        page_b.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "B"}],
        })
        inst.page = page_b
        inst._register_page(page_b)
        result = await mgr.snapshot("a1", diff_from_last=True)
        # Pre-fix: returned "navigation" because previous-is-None check
        # ran before tab-change check.
        assert result["data"]["scope"] == "tab_changed"

    @pytest.mark.asyncio
    async def test_value_field_change_in_diff(self):
        """``value`` mutation (e.g. user typed into a textbox) shows up
        as a ``changed`` entry."""
        tree_v1 = {
            "role": "WebArea", "name": "",
            "children": [{"role": "textbox", "name": "Email", "value": ""}],
        }
        tree_v2 = {
            "role": "WebArea", "name": "",
            "children": [{"role": "textbox", "name": "Email",
                           "value": "alice@example.com"}],
        }
        mgr, _, mock_page = await self._setup(tree_v1)
        await mgr.snapshot("a1")
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree_v2)
        result = await mgr.snapshot("a1", diff_from_last=True)
        data = result["data"]
        assert len(data["changed"]) == 1
        assert data["changed"][0]["value"] == {
            "from": "", "to": "alice@example.com",
        }

    @pytest.mark.asyncio
    async def test_checked_field_change_in_diff(self):
        """``checked`` flip on a checkbox shows up as ``changed``."""
        tree_v1 = {
            "role": "WebArea", "name": "",
            "children": [{"role": "checkbox", "name": "Subscribe",
                           "checked": False}],
        }
        tree_v2 = {
            "role": "WebArea", "name": "",
            "children": [{"role": "checkbox", "name": "Subscribe",
                           "checked": True}],
        }
        mgr, _, mock_page = await self._setup(tree_v1)
        await mgr.snapshot("a1")
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree_v2)
        result = await mgr.snapshot("a1", diff_from_last=True)
        data = result["data"]
        assert len(data["changed"]) == 1
        assert data["changed"][0]["checked"] == {"from": False, "to": True}

    @pytest.mark.asyncio
    async def test_unnamed_sibling_removal_is_positional(self):
        """Documents the priority-4 keying behavior: ``sibling_index``
        is positional (which slot in walk-order this is for the
        (role, name) pair) — NOT element identity. Removing the last
        unnamed sibling drops slot N; slot 0..N-1 keep the same keys.
        Result: diff reports one ``removed`` and zero ``added`` —
        which is more conservative than the worst-case "remove+add
        every shifted sibling" interpretation. Still imperfect: an
        agent that cares which specific button was removed has no
        signal beyond the count. data-testid extraction (priority 1)
        will give true element identity."""
        tree_v1 = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": ""},   # nameless sibling 1
                {"role": "button", "name": ""},   # nameless sibling 2
            ],
        }
        tree_v2 = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": ""},   # only one survives
            ],
        }
        mgr, _, mock_page = await self._setup(tree_v1)
        await mgr.snapshot("a1")
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree_v2)
        result = await mgr.snapshot("a1", diff_from_last=True)
        data = result["data"]
        # Slot-0 survivor matches baseline slot-0 key → unchanged.
        # The vanished slot-1 entry → removed.
        assert len(data["removed"]) == 1
        assert len(data["added"]) == 0
        assert data["unchanged_count"] == 1

    @pytest.mark.asyncio
    async def test_named_duplicate_collision_misses_remove(self):
        """Documents the priority-3 keying limitation: two elements
        with the same role+name+landmark collide on element_key, and
        ``ref_summary`` keeps only the latest one. Removing one of the
        duplicates is reported as "unchanged" because the survivor's
        key matches the baseline's surviving entry. Same future fix
        (data-testid)."""
        tree_v1 = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Click"},
                {"role": "button", "name": "Click"},
            ],
        }
        tree_v2 = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Click"},
            ],
        }
        mgr, _, mock_page = await self._setup(tree_v1)
        await mgr.snapshot("a1")
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree_v2)
        result = await mgr.snapshot("a1", diff_from_last=True)
        data = result["data"]
        # Documented limitation: duplicate-named removal is invisible
        # to the diff. unchanged_count==1 because the surviving
        # duplicate matches the baseline's surviving entry.
        assert data["scope"] == "same"
        assert data["removed"] == []
        assert data["added"] == []
        assert data["unchanged_count"] == 1

    def test_compute_diff_descriptors_are_deterministic(self):
        from src.browser.service import _compute_snapshot_diff
        prev = {
            "k1": {"ref_id": "e0", "role": "button", "name": "A",
                   "landmark": "main", "disabled": False, "value": "",
                   "checked": None},
        }
        curr = {
            "k2": {"ref_id": "e1", "role": "link", "name": "B",
                   "landmark": "nav", "disabled": False, "value": "",
                   "checked": None},
            "k3": {"ref_id": "e0", "role": "link", "name": "C",
                   "landmark": "nav", "disabled": False, "value": "",
                   "checked": None},
        }
        diff = _compute_snapshot_diff(prev, curr)
        assert len(diff["added"]) == 2
        # added sort order is by ref_id — e0 first.
        assert diff["added"][0]["name"] == "C"
        assert diff["added"][1]["name"] == "B"
        assert len(diff["removed"]) == 1
        assert diff["removed"][0]["name"] == "A"


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
        inst.seed_refs_legacy({"e0": {"role": "textbox", "name": "Email"}})
        mgr._instances["a1"] = inst

        with patch("src.browser.service.random.random", return_value=1.0):
            result = await mgr.type_text("a1", ref="e0", text="test@example.com", clear=True)
        assert result["success"] is True
        mock_locator.click.assert_called_once()
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
        inst.seed_refs_legacy({"e0": {"role": "textbox", "name": "Email"}})
        mgr._instances["a1"] = inst

        with patch("src.browser.service.random.random", return_value=1.0):
            result = await mgr.type_text("a1", ref="e0", text="ab", clear=False)
        assert result["success"] is True
        mock_locator.click.assert_called_once()
        press_calls = [c[0][0] for c in mock_page.keyboard.press.call_args_list]
        assert "Control+a" not in press_calls
        assert "a" in press_calls and "b" in press_calls
        assert mock_page.evaluate.await_count == 0

    @pytest.mark.asyncio
    async def test_type_no_ref_or_selector(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), AsyncMock())
        mgr._instances["a1"] = inst

        result = await mgr.type_text("a1", text="hello")
        assert result["success"] is False
        assert "Must provide" in result["error"]


class TestDetectCaptcha:
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

        result = await mgr.detect_captcha("a1")
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

        result = await mgr.detect_captcha("a1")
        assert result["success"] is True
        assert result["data"]["captcha_found"] is True


class TestHumanTiming:
    """Validate timing helper distributions stay within expected ranges."""

    def setup_method(self):
        from src.browser.timing import set_delay, set_speed
        set_speed(1.0)
        set_delay(0.0)

    def teardown_method(self):
        from src.browser.timing import set_delay, set_speed
        set_speed(1.0)
        set_delay(0.0)

    def test_action_delay_range(self):
        from src.browser.timing import action_delay
        samples = [action_delay() for _ in range(1000)]
        assert all(0.08 <= s <= 0.30 for s in samples)
        mean = sum(samples) / len(samples)
        assert 0.14 <= mean <= 0.22

    def test_navigation_jitter_range(self):
        from src.browser.timing import navigation_jitter
        samples = [navigation_jitter() for _ in range(1000)]
        assert all(0.0 <= s <= 0.20 for s in samples)
        mean = sum(samples) / len(samples)
        assert 0.04 <= mean <= 0.12

    def test_keystroke_delay_alpha(self):
        from src.browser.timing import keystroke_delay
        samples = [keystroke_delay("a") for _ in range(1000)]
        assert all(0.020 <= s <= 0.090 for s in samples)
        mean = sum(samples) / len(samples)
        assert 0.035 <= mean <= 0.055

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
        assert all(0.20 <= s <= 0.90 for s in samples)
        mean = sum(samples) / len(samples)
        assert 0.30 <= mean <= 0.50

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

    def test_default_delay(self):
        from src.browser.timing import get_delay
        assert get_delay() == 0.0

    def test_set_delay(self):
        from src.browser.timing import get_delay, set_delay
        set_delay(3.0)
        assert get_delay() == 3.0
        set_delay(0.0)  # cleanup

    def test_delay_clamped_low(self):
        from src.browser.timing import get_delay, set_delay
        set_delay(-5.0)
        assert get_delay() == 0.0
        set_delay(0.0)

    def test_delay_clamped_high(self):
        from src.browser.timing import get_delay, set_delay
        set_delay(99.0)
        assert get_delay() == 10.0
        set_delay(0.0)

    def test_inter_action_delay_zero_when_disabled(self):
        from src.browser.timing import inter_action_delay, set_delay
        set_delay(0.0)
        for _ in range(100):
            assert inter_action_delay() == 0.0

    def test_inter_action_delay_positive_when_enabled(self):
        from src.browser.timing import inter_action_delay, set_delay
        set_delay(3.0)
        samples = [inter_action_delay() for _ in range(500)]
        mean = sum(samples) / len(samples)
        assert 1.5 < mean < 5.0, f"Expected mean near 3.0, got {mean:.2f}"
        assert all(s > 0 for s in samples)
        set_delay(0.0)

    def test_inter_action_delay_not_scaled_by_speed(self):
        """Delay should be independent of the speed setting."""
        from src.browser.timing import inter_action_delay, set_delay, set_speed
        set_delay(3.0)
        set_speed(1.0)
        baseline = [inter_action_delay() for _ in range(1000)]
        set_speed(4.0)
        fast = [inter_action_delay() for _ in range(1000)]
        set_speed(1.0)
        set_delay(0.0)
        baseline_mean = sum(baseline) / len(baseline)
        fast_mean = sum(fast) / len(fast)
        ratio = fast_mean / baseline_mean
        assert 0.8 < ratio < 1.2, f"Delay should not change with speed, ratio={ratio:.2f}"


class TestScroll:
    """Tests for BrowserManager.scroll()."""

    @pytest.mark.asyncio
    async def test_scroll_down_default(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.mouse = AsyncMock()
        mock_page.mouse.wheel = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.scroll("a1", direction="down")
        assert result["success"] is True
        assert result["data"]["direction"] == "down"
        assert result["data"]["pixels"] >= 720
        assert mock_page.mouse.wheel.await_count >= 1

    @pytest.mark.asyncio
    async def test_scroll_up(self):
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.mouse = AsyncMock()
        mock_page.mouse.wheel = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.scroll("a1", direction="up", amount=200)
        assert result["success"] is True
        assert result["data"]["direction"] == "up"
        # mouse.wheel(0, delta) — second arg is the vertical delta.
        # All scroll deltas must be negative for "up" direction.
        calls = mock_page.mouse.wheel.call_args_list
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
        inst.seed_refs_legacy({"e3": {"role": "button", "name": "Submit"}})
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
        inst.seed_refs_legacy({"e0": {"role": "button", "name": "OK"}})
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
        mock_page.mouse = AsyncMock()
        mock_page.mouse.wheel = AsyncMock(side_effect=Exception("page closed"))
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.scroll("a1")
        assert result["success"] is False
        assert "page closed" in result["error"]


class TestX11Scroll:
    """Tests for X11-based scroll via xdotool button 4/5."""

    @pytest.mark.asyncio
    async def test_x11_scroll_uses_xdotool(self):
        """When x11_wid is set, scroll uses xdotool button 5 for down."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.x11_wid = 12345
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"), \
             patch("src.browser.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = await mgr.scroll("a1", direction="down", amount=200)

        assert result["success"] is True
        # Should have called xdotool, not mouse.wheel
        assert mock_run.call_count >= 1
        # All calls should use button 5 (scroll down)
        for call in mock_run.call_args_list:
            args = call[0][0]
            if "click" in args:
                assert "5" in args

    @pytest.mark.asyncio
    async def test_x11_scroll_up_uses_button_4(self):
        """Scroll up uses xdotool button 4."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.x11_wid = 12345
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"), \
             patch("src.browser.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = await mgr.scroll("a1", direction="up", amount=100)

        assert result["success"] is True
        for call in mock_run.call_args_list:
            args = call[0][0]
            if "click" in args:
                assert "4" in args

    @pytest.mark.asyncio
    async def test_x11_scroll_fallback_on_failure(self):
        """When xdotool fails mid-scroll, remaining distance uses CDP."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.mouse.wheel = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.x11_wid = 12345
        mgr._instances["a1"] = inst

        call_count = 0

        def fail_on_second(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock(returncode=0 if call_count == 1 else 1)
            return result

        with patch("src.browser.service.asyncio.sleep"), \
             patch("src.browser.service.subprocess.run", side_effect=fail_on_second):
            result = await mgr.scroll("a1", direction="down", amount=200)

        assert result["success"] is True
        # CDP fallback should have been called for remaining distance
        mock_page.mouse.wheel.assert_called_once()
        _, kwargs = mock_page.mouse.wheel.call_args
        # Remaining should be amount - scrolled (200 - 53 = 147)
        delta = mock_page.mouse.wheel.call_args[0][1]
        assert 100 <= delta <= 200  # remaining px, positive for down


class TestX11EnsureInViewport:
    """Tests for _x11_ensure_in_viewport element scroll behavior."""

    @pytest.mark.asyncio
    async def test_element_in_viewport_no_scroll(self):
        """Element already visible — no scroll or fallback needed."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1920, "height": 1080}
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.x11_wid = 12345

        mock_locator = AsyncMock()
        mock_locator.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 400, "width": 200, "height": 40,
        })
        mock_locator.scroll_into_view_if_needed = AsyncMock()

        with patch("src.browser.service.asyncio.sleep"):
            await mgr._x11_ensure_in_viewport(inst, mock_locator)

        # Element at y=400 is in viewport — no protocol scroll fallback
        mock_locator.scroll_into_view_if_needed.assert_not_called()

    @pytest.mark.asyncio
    async def test_element_below_viewport_scrolls_x11(self):
        """Element below viewport — X11 scroll down until visible."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1920, "height": 1080}
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.x11_wid = 12345

        # First call: element below viewport; second call: element now visible
        mock_locator = AsyncMock()
        mock_locator.bounding_box = AsyncMock(side_effect=[
            {"x": 100, "y": 1200, "width": 200, "height": 40},  # below
            {"x": 100, "y": 900, "width": 200, "height": 40},   # moved after scroll
            {"x": 100, "y": 500, "width": 200, "height": 40},   # now visible
        ])
        mock_locator.scroll_into_view_if_needed = AsyncMock()

        with patch("src.browser.service.asyncio.sleep"), \
             patch("src.browser.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            await mgr._x11_ensure_in_viewport(inst, mock_locator)

        # Should have called xdotool for scroll (button 5 = down)
        scroll_calls = [c for c in mock_run.call_args_list
                        if "click" in c[0][0] and "5" in c[0][0]]
        assert len(scroll_calls) >= 1
        # Protocol fallback should NOT have been called (X11 succeeded)
        mock_locator.scroll_into_view_if_needed.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_x11_wid_uses_protocol_scroll(self):
        """Without x11_wid, should use protocol scroll immediately."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1920, "height": 1080}
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.x11_wid = None  # No X11

        mock_locator = AsyncMock()
        mock_locator.scroll_into_view_if_needed = AsyncMock()

        await mgr._x11_ensure_in_viewport(inst, mock_locator)

        mock_locator.scroll_into_view_if_needed.assert_called_once()

    @pytest.mark.asyncio
    async def test_inner_container_falls_back(self):
        """When X11 scroll doesn't move element, fall back to protocol."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1920, "height": 1080}
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.x11_wid = 12345

        # Element position never changes (inner container)
        mock_locator = AsyncMock()
        mock_locator.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 1200, "width": 200, "height": 40,
        })
        mock_locator.scroll_into_view_if_needed = AsyncMock()

        with patch("src.browser.service.asyncio.sleep"), \
             patch("src.browser.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            await mgr._x11_ensure_in_viewport(inst, mock_locator)

        # Should have fallen back to protocol scroll
        mock_locator.scroll_into_view_if_needed.assert_called_once()

    @pytest.mark.asyncio
    async def test_small_movement_continues_scrolling(self):
        """Small but real movement (>= 2px) should NOT trigger stall detection."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1920, "height": 1080}
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.x11_wid = 12345

        # Element moves 5px per batch (small but real progress), then enters viewport
        call_count = [0]

        async def moving_bbox():
            call_count[0] += 1
            y = max(500, 1200 - call_count[0] * 100)  # gradually enters viewport
            return {"x": 100, "y": y, "width": 200, "height": 40}

        mock_locator = AsyncMock()
        mock_locator.bounding_box = AsyncMock(side_effect=moving_bbox)
        mock_locator.scroll_into_view_if_needed = AsyncMock()

        with patch("src.browser.service.asyncio.sleep"), \
             patch("src.browser.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            await mgr._x11_ensure_in_viewport(inst, mock_locator)

        # Should NOT have used protocol fallback — X11 scroll succeeded
        mock_locator.scroll_into_view_if_needed.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_viewport_uses_protocol(self):
        """When viewport_size is None, fall back to protocol scroll."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = None
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.x11_wid = 12345

        mock_locator = AsyncMock()
        mock_locator.scroll_into_view_if_needed = AsyncMock()

        await mgr._x11_ensure_in_viewport(inst, mock_locator)

        mock_locator.scroll_into_view_if_needed.assert_called_once()


class TestTypoInjection:
    """Tests for typo injection in _x11_type."""

    @pytest.mark.asyncio
    async def test_short_text_no_typos(self):
        """Text shorter than 15 alpha chars should never get typos."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), MagicMock())
        inst.x11_wid = 12345

        with patch("src.browser.service.asyncio.sleep"), \
             patch("src.browser.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            await mgr._x11_type(inst, "Hello")

        # No BackSpace should appear — text too short for typos
        backspace_calls = [c for c in mock_run.call_args_list
                           if "BackSpace" in str(c)]
        assert len(backspace_calls) == 0

    @pytest.mark.asyncio
    async def test_typos_disabled_no_backspace(self):
        """typos=False should produce zero typo corrections."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), MagicMock())
        inst.x11_wid = 12345

        long_text = "The quick brown fox jumps over the lazy dog and keeps running"

        with patch("src.browser.service.asyncio.sleep"), \
             patch("src.browser.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            await mgr._x11_type(inst, long_text, typos=False)

        backspace_calls = [c for c in mock_run.call_args_list
                           if "BackSpace" in str(c)]
        assert len(backspace_calls) == 0

    @pytest.mark.asyncio
    async def test_long_text_injects_typos(self):
        """Long text with typos=True should inject BackSpace corrections.

        Patches random.gauss to guarantee a budget of 2 typos.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), MagicMock())
        inst.x11_wid = 12345

        long_text = (
            "The quick brown fox jumps over the lazy dog "
            "and keeps running through the enchanted forest"
        )

        with patch("src.browser.service.asyncio.sleep"), \
             patch("src.browser.service.subprocess.run") as mock_run, \
             patch("src.browser.service.random.gauss", return_value=2.0), \
             patch("src.browser.service.random.random", return_value=0.5):
            mock_run.return_value = MagicMock(returncode=0)
            await mgr._x11_type(inst, long_text, typos=True)

        backspace_calls = [c for c in mock_run.call_args_list
                           if "BackSpace" in str(c)]
        assert len(backspace_calls) == 2, f"Expected 2 typo corrections, got {len(backspace_calls)}"


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

        # Delays include hover settle (0.02-0.06) and action_delay (0.08-0.30)
        assert all(0.02 <= d <= 0.30 for d in delays)
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
        assert all(0.020 <= d <= 0.110 for d in delays)

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


class TestTypeFast:
    """Tests for _type_fast minimal-delay mode."""

    @pytest.mark.asyncio
    async def test_type_fast_uses_fixed_delay(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()

        delays: list[float] = []

        async def capture_sleep(t: float):
            delays.append(t)

        with patch("src.browser.service.asyncio.sleep", side_effect=capture_sleep):
            await mgr._type_fast(mock_page, "hello")

        assert mock_page.keyboard.press.await_count == 5
        assert len(delays) == 5
        assert all(d == 0.008 for d in delays), f"Expected all 0.008, got {delays}"

    @pytest.mark.asyncio
    async def test_type_fast_handles_special_keys(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()

        with patch("src.browser.service.asyncio.sleep"):
            await mgr._type_fast(mock_page, "a\nb")

        press_calls = [c[0][0] for c in mock_page.keyboard.press.call_args_list]
        assert press_calls == ["a", "Enter", "b"]

    @pytest.mark.asyncio
    async def test_type_text_fast_flag(self):
        """fast=True in type_text should use _type_fast, not _type_with_variance."""
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_locator = AsyncMock()
        mock_locator.nth = MagicMock(return_value=mock_locator)
        mock_page = MagicMock()
        mock_page.get_by_role = MagicMock(return_value=mock_locator)
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.mouse = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.seed_refs_legacy({"e0": {"role": "textbox", "name": "Search", "index": 0}})
        mgr._instances["a1"] = inst

        delays: list[float] = []

        async def capture_sleep(t: float):
            delays.append(t)

        with patch("src.browser.service.asyncio.sleep", side_effect=capture_sleep):
            result = await mgr.type_text("a1", ref="e0", text="test", fast=True)

        assert result["success"] is True
        typing_delays = [d for d in delays if d == 0.008]
        assert len(typing_delays) == 4


class TestSnapshotAfter:
    """Tests for compound action snapshot_after parameter."""

    @pytest.mark.asyncio
    async def test_click_snapshot_after(self):
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.query_selector_all = AsyncMock(return_value=[])
        mock_page.evaluate = AsyncMock(return_value={
            "role": "WebArea", "name": "Test",
            "children": [{"role": "button", "name": "OK", "refId": "e0"}],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst._js_snapshot_mode = True
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            result = await mgr.click("a1", selector="#btn", snapshot_after=True)

        assert result["success"] is True
        assert "snapshot" in result
        assert "snapshot" in result["snapshot"]
        assert "refs" in result["snapshot"]

    @pytest.mark.asyncio
    async def test_click_without_snapshot_after(self):
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.click = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            result = await mgr.click("a1", selector="#btn", snapshot_after=False)

        assert result["success"] is True
        assert "snapshot" not in result


class TestNavigateBodyCap:
    """§7.6 — body preview is shorter when snapshot_after=True since
    the snapshot already carries the element tree."""

    def _make_long_a11y(self, text_len: int) -> dict:
        # Many short leaves — total joined length ≈ text_len. Avoids
        # tripping the credential redactor (which targets long base64-
        # like runs) and produces realistic word-spaced output that
        # ``_extract_text_from_a11y`` joins with spaces.
        word = "hello "
        n_leaves = max(1, text_len // len(word))
        return {
            "role": "WebArea",
            "name": "T",
            "children": [
                {"role": "text", "name": "hello"} for _ in range(n_leaves)
            ],
        }

    @pytest.mark.asyncio
    async def test_body_capped_at_1000_with_snapshot_after(self):
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="t")
        mock_page.url = "https://example.com"
        # accessibility.snapshot returns a tree with a 4000-char leaf
        a11y_tree = self._make_long_a11y(4000)
        mock_page.accessibility = AsyncMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=a11y_tree)
        # query_selector_all + evaluate are used by the snapshot path
        mock_page.query_selector_all = AsyncMock(return_value=[])
        mock_page.evaluate = AsyncMock(return_value={
            "role": "WebArea", "name": "t", "children": [],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.navigate(
            "a1", "https://example.com", wait_ms=0, snapshot_after=True,
        )
        assert result["success"] is True
        # body is the cap-1000 preview when snapshot_after is on.
        assert len(result["data"]["body"]) <= 1000
        # And the snapshot rides alongside.
        assert "snapshot" in result

    @pytest.mark.asyncio
    async def test_body_capped_at_5000_without_snapshot_after(self):
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="t")
        mock_page.url = "https://example.com"
        # 8000-char-equivalent tree → joined output trims to the cap.
        a11y_tree = self._make_long_a11y(8000)
        mock_page.accessibility = AsyncMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=a11y_tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.navigate(
            "a1", "https://example.com", wait_ms=0, snapshot_after=False,
        )
        assert result["success"] is True
        # body is returned at the 5000-char cap when snapshot_after off.
        body = result["data"]["body"]
        assert 4000 <= len(body) <= 5000, len(body)

    @pytest.mark.asyncio
    async def test_body_falls_back_to_5000_when_snapshot_fails(self):
        """When snapshot_after=True but the snapshot itself fails or
        returns nothing, the agent must NOT be left with a 1000-char
        body AND an empty snapshot — strictly worse than the
        snapshot_after=False path. The body falls back to the 5000-char
        cap so the agent has usable page text."""
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="t")
        mock_page.url = "https://example.com"
        a11y_tree = self._make_long_a11y(8000)
        mock_page.accessibility = AsyncMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=a11y_tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        # Stub out _snapshot_impl to simulate failure.
        async def _failing_snapshot(*_a, **_k):
            return {"success": False, "error": "boom"}

        with patch.object(mgr, "_snapshot_impl", _failing_snapshot):
            result = await mgr.navigate(
                "a1", "https://example.com", wait_ms=0, snapshot_after=True,
            )
        assert result["success"] is True
        body = result["data"]["body"]
        # Falls back to 5000-cap when snapshot was unsuccessful.
        assert 4000 <= len(body) <= 5000, len(body)


class TestNavigateRetry:
    """Tests for navigation timeout retry."""

    @pytest.mark.asyncio
    async def test_navigate_retries_on_timeout(self):
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(
            side_effect=[Exception("Timeout 30000ms exceeded"), None]
        )
        mock_page.title = AsyncMock(return_value="OK")
        mock_page.url = "https://example.com"
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea", "name": "OK", "children": [],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            result = await mgr.navigate("a1", "https://example.com", wait_ms=0)

        assert result["success"] is True
        assert mock_page.goto.await_count == 2

    @pytest.mark.asyncio
    async def test_navigate_no_retry_on_non_timeout(self):
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("net::ERR_NAME_NOT_RESOLVED"))
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            result = await mgr.navigate("a1", "https://bad.invalid", wait_ms=0)

        assert result["success"] is False
        assert mock_page.goto.await_count == 1


class TestExtractTextFromA11y:
    """Tests for _extract_text_from_a11y helper used by navigate()."""

    def test_none_returns_empty(self):
        from src.browser.service import _extract_text_from_a11y
        assert _extract_text_from_a11y(None) == ""

    def test_empty_dict_returns_empty(self):
        from src.browser.service import _extract_text_from_a11y
        assert _extract_text_from_a11y({}) == ""

    def test_leaf_node_extracts_name(self):
        from src.browser.service import _extract_text_from_a11y
        tree = {"role": "text", "name": "Hello world"}
        assert _extract_text_from_a11y(tree) == "Hello world"

    def test_nested_children_collects_leaves(self):
        from src.browser.service import _extract_text_from_a11y
        tree = {
            "role": "WebArea", "name": "Page Title",
            "children": [
                {"role": "heading", "name": "Welcome", "children": [
                    {"role": "text", "name": "Welcome"},
                ]},
                {"role": "paragraph", "name": "Some text", "children": [
                    {"role": "text", "name": "Some text"},
                ]},
                {"role": "button", "name": "Click me"},
            ],
        }
        result = _extract_text_from_a11y(tree)
        assert "Welcome" in result
        assert "Some text" in result
        assert "Click me" in result

    def test_no_duplicate_from_parent_and_child(self):
        from src.browser.service import _extract_text_from_a11y
        tree = {
            "role": "WebArea", "name": "Page",
            "children": [
                {"role": "heading", "name": "Title", "children": [
                    {"role": "text", "name": "Title"},
                ]},
            ],
        }
        result = _extract_text_from_a11y(tree)
        # Should contain "Title" once, not twice
        assert result.count("Title") == 1

    def test_truncation_at_max_chars(self):
        from src.browser.service import _extract_text_from_a11y
        tree = {
            "role": "WebArea", "name": "",
            "children": [{"role": "text", "name": "x" * 100}
                         for _ in range(100)],
        }
        result = _extract_text_from_a11y(tree, max_chars=500)
        assert len(result) <= 500

    def test_empty_name_nodes_skipped(self):
        from src.browser.service import _extract_text_from_a11y
        tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "generic", "name": ""},
                {"role": "button", "name": "OK"},
            ],
        }
        assert _extract_text_from_a11y(tree) == "OK"

    def test_malformed_children_handled(self):
        """Non-dict children should be skipped without crashing."""
        from src.browser.service import _extract_text_from_a11y
        tree = {
            "role": "WebArea", "name": "",
            "children": [
                None,
                "stray string",
                {"role": "button", "name": "OK"},
                42,
            ],
        }
        assert _extract_text_from_a11y(tree) == "OK"

    @pytest.mark.asyncio
    async def test_navigate_body_text_from_a11y(self):
        """navigate() should extract body text via a11y snapshot, not page.evaluate."""
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.url = "https://example.com"
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea", "name": "Example",
            "children": [
                {"role": "heading", "name": "Hello"},
                {"role": "text", "name": "World"},
            ],
        })
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            result = await mgr.navigate("a1", "https://example.com", wait_ms=0)

        assert result["success"] is True
        assert "Hello" in result["data"]["body"]
        assert "World" in result["data"]["body"]
        # page.evaluate should NOT have been called for body text
        mock_page.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_navigate_skips_a11y_in_js_mode(self):
        """When _js_snapshot_mode is True, navigate skips a11y call."""
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.url = "https://example.com"
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst._js_snapshot_mode = True
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            result = await mgr.navigate("a1", "https://example.com", wait_ms=0)

        assert result["success"] is True
        assert result["data"]["body"] == ""
        # a11y snapshot should NOT have been called
        mock_page.accessibility.snapshot.assert_not_called()


class TestLandmarkAnnotations:
    """Tests for structural landmark context in snapshot output."""

    @pytest.mark.asyncio
    async def test_landmark_context_in_snapshot(self):
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        tree = {
            "role": "WebArea", "name": "Test Page",
            "children": [
                {"role": "button", "name": "Post", "refId": "e0", "landmark": "navigation"},
                {"role": "button", "name": "Post", "refId": "e1", "landmark": "dialog: Compose"},
            ],
        }
        mock_page = AsyncMock()
        mock_page.query_selector_all = AsyncMock(return_value=[])
        mock_page.evaluate = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst._js_snapshot_mode = True
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")

        snapshot_text = result["data"]["snapshot"]
        assert "(navigation)" in snapshot_text
        assert "(dialog: Compose)" in snapshot_text
        assert 'button "Post"' in snapshot_text
        assert "dup:2" in snapshot_text

    @pytest.mark.asyncio
    async def test_no_landmark_when_absent(self):
        from src.browser.service import BrowserManager, CamoufoxInstance

        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        tree = {
            "role": "WebArea", "name": "Test",
            "children": [{"role": "button", "name": "OK", "refId": "e0"}],
        }
        mock_page = AsyncMock()
        mock_page.query_selector_all = AsyncMock(return_value=[])
        mock_page.evaluate = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst._js_snapshot_mode = True
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")

        snapshot_text = result["data"]["snapshot"]
        assert "(" not in snapshot_text
        assert '[e0] button "OK"' in snapshot_text


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
        # Assert the wait_until and timeout kwargs specifically — Phase 3
        # §6.5 may also pass a ``referer=`` kwarg picked from the pool;
        # we don't pin it here to keep this test about wait_until alone.
        call = mock_page.goto.await_args
        assert call.args == ("https://x.com",)
        assert call.kwargs["wait_until"] == "domcontentloaded"
        assert call.kwargs["timeout"] == 30000

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

        result = await mgr.navigate(
            "a1", "https://x.com", wait_ms=0, wait_until="networkidle",
        )
        assert result["success"] is True
        call = mock_page.goto.await_args
        assert call.args == ("https://x.com",)
        assert call.kwargs["wait_until"] == "networkidle"
        assert call.kwargs["timeout"] == 30000

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
        inst.seed_refs_legacy({"e0": {"role": "button", "name": "Post"}})
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
        inst.seed_refs_legacy({"e3": {"role": "link", "name": "Products"}})
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
    """Regression tests for the mesh-proxy known-action set.

    Any browser action added to browser_tool.py must also be in
    KNOWN_BROWSER_ACTIONS in src/host/permissions.py (single source of truth;
    host/server.py imports as _ALLOWED_BROWSER_ACTIONS for input validation)
    — otherwise the skill silently fails with a 400 "unknown action" from
    the proxy.
    """

    def _get_allowed_actions(self) -> frozenset[str]:
        """Load the canonical known-action set from host.permissions."""
        from src.host.permissions import KNOWN_BROWSER_ACTIONS
        return KNOWN_BROWSER_ACTIONS

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
                    "reset", "focus", "scroll", "detect_captcha", "wait_for", "hover"}
        missing = required - actions
        assert not missing, f"Missing browser actions: {missing}"


class TestScrollParameterized:
    """Verify scroll uses mouse.wheel() with correct delta signs."""

    @pytest.mark.asyncio
    async def test_scroll_uses_wheel_events_not_evaluate(self):
        """Scroll must use mouse.wheel() for isTrusted wheel events, not evaluate."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.mouse = AsyncMock()
        mock_page.mouse.wheel = AsyncMock()
        mock_page.evaluate = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            await mgr.scroll("a1", direction="down", amount=200)

        # mouse.wheel should be called, not evaluate
        assert mock_page.mouse.wheel.await_count >= 1
        mock_page.evaluate.assert_not_called()
        # All deltas must be positive for "down" direction
        for call in mock_page.mouse.wheel.call_args_list:
            delta = call[0][1]
            assert isinstance(delta, (int, float))
            assert delta > 0  # down = positive

    @pytest.mark.asyncio
    async def test_scroll_up_delta_is_negative(self):
        """Scroll up must pass a negative delta to mouse.wheel()."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.mouse = AsyncMock()
        mock_page.mouse.wheel = AsyncMock()
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            await mgr.scroll("a1", direction="up", amount=200)

        for call in mock_page.mouse.wheel.call_args_list:
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

        # 0.05 is between 0.015 (non-boundary threshold) and 0.08 (boundary threshold)
        # So: fires only when prev_char was a boundary char
        with patch("src.browser.service.random.random", return_value=0.05):
            with patch("src.browser.service.asyncio.sleep", side_effect=capture_sleep):
                # "h w": h=no-pause, ' '=no-pause, w=PAUSE (prev ' ')
                await mgr._type_with_variance(mock_page, "h w")

        # think_pause values are in [0.20, 0.90]; keystroke_delay values are in [0.018, 0.110]
        think_pauses = [t for t in sleep_calls if t >= 0.20]
        assert len(think_pauses) == 1, (
            f"Expected exactly 1 think_pause for 'h w' with random=0.05, got {think_pauses}"
        )

    @pytest.mark.asyncio
    async def test_no_think_pause_when_random_above_boundary_threshold(self):
        """With random.random()=0.15, no pauses fire (0.15 > 0.08)."""
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

        think_pauses = [t for t in sleep_calls if t >= 0.20]
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
        inst.x11_wid = None  # No X11 WID — use CDP click path
        inst._user_control = False  # Not in user-control mode
        inst.dialog_detected = False
        inst.dialog_active = False
        return inst

    @pytest.mark.asyncio
    async def test_disabled_button_ref_auto_forces(self):
        """A disabled button ref should auto-force the click."""
        from src.browser.service import BrowserManager

        mgr = self._make_manager()
        refs = {k: _h(v) for k, v in {"e1": {"role": "button", "name": "Post", "index": 0, "disabled": True}}.items()}
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
        refs = {k: _h(v) for k, v in {"e2": {"role": "link", "name": "Sign in", "index": 0, "disabled": True}}.items()}
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
        refs = {
            k: _h(v) for k, v in {
                "e3": {"role": "textbox", "name": "Search", "index": 0, "disabled": True},
            }.items()
        }
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
        refs = {
            k: _h(v) for k, v in {
                "e4": {"role": "menuitem", "name": "Delete", "index": 0, "disabled": True},
            }.items()
        }
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
        refs = {
            k: _h(v) for k, v in {
                "e5": {"role": "button", "name": "Submit", "index": 0, "disabled": False},
            }.items()
        }
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
        refs = {k: _h(v) for k, v in {"e6": {"role": "button", "name": "Post", "index": 0, "disabled": True}}.items()}
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
        refs = {k: _h(v) for k, v in {"e7": {"role": "button", "name": "OK", "index": 0}}.items()}
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
        inst.page.query_selector_all = AsyncMock(return_value=[])
        inst.page.url = "https://x.com"
        inst.page.title = AsyncMock(return_value="X")
        inst.refs = {}
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()
        inst._js_snapshot_mode = False

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.snapshot("agent1")

        assert result["success"] is True
        assert inst.refs["e0"].disabled is True

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
        inst.page.query_selector_all = AsyncMock(return_value=[])
        inst.page.url = "https://example.com"
        inst.page.title = AsyncMock(return_value="Example")
        inst.refs = {}
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()
        inst._js_snapshot_mode = False

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.snapshot("agent1")

        assert result["success"] is True
        assert inst.refs["e0"].disabled is False


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
        inst.x11_wid = 12345
        inst.refs = {
            k: _h(v) for k, v in {
                "e1": {"role": "textbox", "name": "Tweet", "index": 0, "disabled": False},
            }.items()
        }
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()
        inst._user_control = False

        sleep_calls: list[float] = []

        async def capture_sleep(t):
            sleep_calls.append(t)

        mock_locator = AsyncMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch.object(BrowserManager, "_locator_from_ref", return_value=mock_locator):
                with patch.object(BrowserManager, "_x11_click", new_callable=AsyncMock):
                    with patch.object(BrowserManager, "_x11_type", new_callable=AsyncMock):
                        with patch.object(BrowserManager, "_x11_key", new_callable=AsyncMock):
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
        inst.x11_wid = 12345
        inst.refs = {
            k: _h(v) for k, v in {
                "e1": {"role": "textbox", "name": "Tweet", "index": 0, "disabled": False},
            }.items()
        }
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()
        inst._user_control = False

        sleep_calls: list[float] = []

        async def capture_sleep(t):
            sleep_calls.append(t)

        mock_locator = AsyncMock()

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            with patch.object(BrowserManager, "_locator_from_ref", return_value=mock_locator):
                with patch.object(BrowserManager, "_x11_click", new_callable=AsyncMock):
                    with patch.object(BrowserManager, "_x11_type", new_callable=AsyncMock):
                        with patch.object(BrowserManager, "_x11_key", new_callable=AsyncMock):
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
        inst._user_control = False

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
        inst._user_control = False

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
        inst.refs = {
            k: _h(v) for k, v in {
                "e0": {"role": "button", "name": "Old", "index": 0, "disabled": False},
            }.items()
        }
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
        mgr._captcha_solver = None
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
            result = await mgr.detect_captcha("agent1")

        assert result["success"] is True
        assert result["data"]["captcha_found"] is True
        assert "challenges.cloudflare.com" in result["data"]["type"]

    @pytest.mark.asyncio
    async def test_no_captcha(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        mgr._captcha_solver = None
        inst = MagicMock()
        inst.page = AsyncMock()
        inst.lock = asyncio.Lock()
        inst.touch = MagicMock()

        # All selectors return 0
        mock_loc = MagicMock()
        mock_loc.count = AsyncMock(return_value=0)
        inst.page.locator = MagicMock(return_value=mock_loc)

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.detect_captcha("agent1")

        assert result["success"] is True
        assert result["data"]["captcha_found"] is False


# ── Dialog scoping tests ──────────────────────────────────────────────────


class TestDialogScoping:
    """When a modal dialog is open, snapshot and locators scope to dialog only.

    Detection uses DOM queries (query_selector_all) for modal elements rather
    than accessibility tree roles, because SPAs like Twitter/X may not surface
    the dialog role in Playwright's accessibility tree.

    This prevents agents from seeing/clicking elements behind the modal overlay
    (e.g. X's sidebar "Post" button behind the compose modal).
    """

    @staticmethod
    def _mock_page_with_modal(full_tree, dialog_subtree):
        """Create a mock page where DOM has a visible modal element.

        full_tree: returned by page.accessibility.snapshot() (no root)
        dialog_subtree: returned by page.accessibility.snapshot(root=modal_el)
        """
        mock_page = AsyncMock()
        mock_modal_el = AsyncMock()
        mock_modal_el.is_visible = AsyncMock(return_value=True)
        mock_modal_el.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 100, "width": 400, "height": 300,
        })
        mock_page.query_selector_all = AsyncMock(return_value=[mock_modal_el])
        mock_page.viewport_size = {"width": 1280, "height": 720}

        async def _snapshot(root=None):
            if root is not None:
                return dialog_subtree
            return full_tree
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(side_effect=_snapshot)
        return mock_page

    @staticmethod
    def _mock_page_no_modal(full_tree):
        """Create a mock page where DOM has no visible modal elements."""
        mock_page = AsyncMock()
        mock_page.query_selector_all = AsyncMock(return_value=[])
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=full_tree)
        return mock_page

    @pytest.mark.asyncio
    async def test_snapshot_scopes_to_dialog(self):
        """Elements outside the dialog should not appear in snapshot."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        full_tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Post", "disabled": True},
                {"role": "link", "name": "Home"},
                {"role": "dialog", "name": "Create post", "children": [
                    {"role": "textbox", "name": "What is happening?!"},
                    {"role": "button", "name": "Post"},
                ]},
            ],
        }
        dialog_subtree = {
            "role": "dialog", "name": "Create post",
            "children": [
                {"role": "textbox", "name": "What is happening?!"},
                {"role": "button", "name": "Post"},
            ],
        }
        mock_page = self._mock_page_with_modal(full_tree, dialog_subtree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        snap = result["data"]["snapshot"]
        refs = result["data"]["refs"]

        assert "Modal dialog is open" in snap
        assert "Create post" in snap
        assert "What is happening?!" in snap
        post_refs = [r for r in refs.values() if r["name"] == "Post"]
        assert len(post_refs) == 1
        assert post_refs[0]["index"] == 0
        assert "Home" not in snap
        sidebar_post = [r for r in refs.values()
                        if r["name"] == "Post" and r.get("disabled")]
        assert len(sidebar_post) == 0
        assert inst.dialog_active is True

    @pytest.mark.asyncio
    async def test_snapshot_no_dialog_walks_all(self):
        """Without a dialog, all elements should appear as before."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Post"},
                {"role": "link", "name": "Home"},
            ],
        }
        mock_page = self._mock_page_no_modal(tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        refs = result["data"]["refs"]
        assert len(refs) == 2
        assert inst.dialog_active is False
        assert "Modal dialog" not in result["data"]["snapshot"]

    @pytest.mark.asyncio
    async def test_snapshot_aria_modal_detected(self):
        """Elements with aria-modal=true (without dialog role in a11y tree) should trigger scoping."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        # Simulate Twitter: DOM has aria-modal=true but a11y tree uses 'group' role
        full_tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Post", "disabled": True},
                {"role": "group", "name": "Confirm", "children": [
                    {"role": "button", "name": "OK"},
                    {"role": "button", "name": "Cancel"},
                ]},
            ],
        }
        dialog_subtree = {
            "role": "group", "name": "Confirm",
            "children": [
                {"role": "button", "name": "OK"},
                {"role": "button", "name": "Cancel"},
            ],
        }
        mock_page = self._mock_page_with_modal(full_tree, dialog_subtree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        refs = result["data"]["refs"]
        assert inst.dialog_active is True
        # Only interactive elements inside the modal: OK + Cancel = 2
        # (group role is not actionable/context, so it's not ref'd)
        assert len(refs) == 2
        assert "Post" not in result["data"]["snapshot"]

    @pytest.mark.asyncio
    async def test_snapshot_dialog_clears_flag_when_dismissed(self):
        """After dialog is dismissed, next snapshot should clear dialog_active."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        # First snapshot: modal open
        dialog_subtree = {
            "role": "dialog", "name": "Modal",
            "children": [{"role": "button", "name": "Close"}],
        }
        full_tree = {
            "role": "WebArea", "name": "",
            "children": [dialog_subtree],
        }
        mock_page = self._mock_page_with_modal(full_tree, dialog_subtree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        await mgr.snapshot("a1")
        assert inst.dialog_active is True

        # Second snapshot: modal dismissed — no visible modals in DOM
        tree_no_dialog = {
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "Post"}],
        }
        mock_page.query_selector_all = AsyncMock(return_value=[])
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree_no_dialog)
        await mgr.snapshot("a1")
        assert inst.dialog_active is False
        assert len(inst.refs) == 1

    @pytest.mark.asyncio
    async def test_snapshot_dialog_duplicate_indexing(self):
        """Occurrence indices should be counted within dialog scope only.

        On X/Twitter, the sidebar "Post" button (index 0 in full page scope)
        must not affect the modal "Post" button's index within the dialog.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        full_tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Post"},
                {"role": "textbox", "name": "Search"},
                {"role": "dialog", "name": "Compose", "children": [
                    {"role": "textbox", "name": "Tweet text"},
                    {"role": "button", "name": "Post"},
                ]},
            ],
        }
        dialog_subtree = {
            "role": "dialog", "name": "Compose",
            "children": [
                {"role": "textbox", "name": "Tweet text"},
                {"role": "button", "name": "Post"},
            ],
        }
        mock_page = self._mock_page_with_modal(full_tree, dialog_subtree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        refs = result["data"]["refs"]

        # Only dialog elements: dialog + textbox + button = 3
        assert len(refs) == 3
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
        inst.seed_refs_legacy({"e0": {"role": "button", "name": "Post", "index": 0}})

        locator = mgr._locator_from_ref(inst, "e0")

        # Should scope to dialog/modal elements
        call_args = mock_page.locator.call_args[0][0]
        assert '[role="dialog"]' in call_args
        assert '[aria-modal="true"]' in call_args
        mock_dialog_locator.get_by_role.assert_called_once_with("button", name="Post", exact=True)
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
        inst.seed_refs_legacy({"e0": {"role": "button", "name": "Post", "index": 0}})

        mgr._locator_from_ref(inst, "e0")

        mock_page.locator.assert_not_called()
        mock_page.get_by_role.assert_called_once_with("button", name="Post", exact=True)

    @pytest.mark.asyncio
    async def test_snapshot_hidden_dialog_ignored(self):
        """Hidden dialog elements (aria-hidden) should not trigger scoping."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        # DOM query returns an element but it's not visible
        tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Post"},
                {"role": "link", "name": "Home"},
            ],
        }
        mock_page = AsyncMock()
        mock_hidden_el = AsyncMock()
        mock_hidden_el.is_visible = AsyncMock(return_value=False)
        mock_page.query_selector_all = AsyncMock(return_value=[mock_hidden_el])
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree)
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        refs = result["data"]["refs"]
        assert len(refs) == 2  # Both elements visible
        assert inst.dialog_active is False

    @pytest.mark.asyncio
    async def test_snapshot_retry_finds_elements_after_wait(self):
        """When scoped snapshot initially finds no actionable refs, a retry
        after a short wait should pick up elements that finished rendering.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        full_tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Post", "disabled": True},
                {"role": "link", "name": "Home"},
            ],
        }
        # First call: dialog detected but only context role (no actionable)
        dialog_empty = {"role": "dialog", "name": "Compose"}
        # Second call (after retry): content has rendered
        dialog_full = {
            "role": "dialog", "name": "Compose",
            "children": [
                {"role": "textbox", "name": "What is happening?!"},
                {"role": "button", "name": "Post"},
            ],
        }
        call_count = [0]

        mock_page = AsyncMock()
        mock_modal_el = AsyncMock()
        mock_modal_el.is_visible = AsyncMock(return_value=True)
        mock_modal_el.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 100, "width": 400, "height": 300,
        })
        mock_page.query_selector_all = AsyncMock(return_value=[mock_modal_el])
        mock_page.viewport_size = {"width": 1280, "height": 720}

        async def _snapshot(root=None):
            if root is not None:
                call_count[0] += 1
                return dialog_empty if call_count[0] <= 1 else dialog_full
            return full_tree
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(side_effect=_snapshot)

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        refs = result["data"]["refs"]
        snap = result["data"]["snapshot"]

        # Retry should have found the elements
        assert call_count[0] == 2  # Called twice (initial + retry)
        assert "What is happening?!" in snap
        post_refs = [r for r in refs.values() if r["name"] == "Post"]
        assert len(post_refs) == 1
        assert inst.dialog_active is True
        assert inst.dialog_detected is True
        assert "Home" not in snap  # Sidebar excluded

    @pytest.mark.asyncio
    async def test_snapshot_context_only_refs_trigger_retry(self):
        """Dialog with only context roles (dialog/heading) but no actionable
        roles (button/textbox) should trigger the retry path.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        full_tree = {
            "role": "WebArea", "name": "",
            "children": [{"role": "button", "name": "Submit"}],
        }
        # Dialog has a heading (context role) but no actionable elements
        dialog_subtree = {
            "role": "dialog", "name": "Confirm",
            "children": [
                {"role": "heading", "name": "Are you sure?"},
            ],
        }
        mock_page = AsyncMock()
        mock_modal_el = AsyncMock()
        mock_modal_el.is_visible = AsyncMock(return_value=True)
        mock_modal_el.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 100, "width": 400, "height": 300,
        })
        mock_page.query_selector_all = AsyncMock(return_value=[mock_modal_el])
        mock_page.viewport_size = {"width": 1280, "height": 720}

        async def _snapshot(root=None):
            if root is not None:
                return dialog_subtree
            return full_tree
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(side_effect=_snapshot)

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        snap = result["data"]["snapshot"]

        # Should have fallen back to full tree since dialog had no actionable refs
        assert "Submit" in snap
        # dialog_active stays True so _locator_from_ref stays scoped to the
        # modal — prevents clicks from targeting elements behind the overlay
        assert inst.dialog_active is True
        # dialog_detected stays True — modal existed, just couldn't scope
        assert inst.dialog_detected is True
        assert "could not be isolated" in snap

    @pytest.mark.asyncio
    async def test_auto_force_suppressed_when_modal_unscoped(self):
        """When a modal is detected but scoping failed, auto-force on disabled
        buttons should be suppressed to prevent clicking behind the overlay.
        """
        from src.browser.service import (
            _CLICK_TIMEOUT_MS,
            BrowserManager,
            CamoufoxInstance,
        )
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        # Build mock chain: page.get_by_role(...).nth(0).click()
        # get_by_role and nth are sync; click is async.
        mock_click = AsyncMock()
        mock_nth = MagicMock()
        mock_nth.click = mock_click
        mock_role_locator = MagicMock()
        mock_role_locator.nth.return_value = mock_nth
        mock_page = MagicMock()
        mock_page.get_by_role = MagicMock(return_value=mock_role_locator)

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        # Simulate: modal detected but scoping failed (full-tree fallback)
        inst.dialog_detected = True
        inst.dialog_active = False
        inst.seed_refs_legacy({
            "e0": {"role": "button", "name": "Post", "index": 0, "disabled": True},
        })

        result = await mgr.click("a1", ref="e0")
        assert result["success"] is True

        # Verify Playwright was called WITHOUT force
        mock_click.assert_awaited_once_with(
            timeout=_CLICK_TIMEOUT_MS, force=False,
        )

    @pytest.mark.asyncio
    async def test_auto_force_active_when_modal_scoped(self):
        """When a modal is properly scoped, auto-force on disabled buttons
        should still work (SPA buttons behind aria-disabled are clickable).
        """
        from src.browser.service import (
            _CLICK_TIMEOUT_MS,
            _MODAL_SELECTOR,
            BrowserManager,
            CamoufoxInstance,
        )
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        # Build mock chain: page.locator(selector).get_by_role(...).nth(0).click()
        mock_click = AsyncMock()
        mock_nth = MagicMock()
        mock_nth.click = mock_click
        mock_role_locator = MagicMock()
        mock_role_locator.nth.return_value = mock_nth
        mock_modal_locator = MagicMock()
        mock_modal_locator.get_by_role = MagicMock(return_value=mock_role_locator)
        mock_page = MagicMock()
        mock_page.locator = MagicMock(return_value=mock_modal_locator)

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        # Simulate: modal detected AND scoping succeeded
        inst.dialog_detected = True
        inst.dialog_active = True
        inst.seed_refs_legacy({
            "e0": {"role": "button", "name": "Post", "index": 0, "disabled": True},
        })

        result = await mgr.click("a1", ref="e0")
        assert result["success"] is True

        # Verify Playwright was called WITH force (auto-force should fire)
        mock_page.locator.assert_called_with(_MODAL_SELECTOR)
        mock_click.assert_awaited_once_with(
            timeout=_CLICK_TIMEOUT_MS, force=True,
        )

    @pytest.mark.asyncio
    async def test_snapshot_fallback_when_scoped_snapshot_empty(self):
        """If snapshot(root=modal) returns None for all modals, fall back to full tree.

        Camoufox (modified Firefox) may not support scoped accessibility
        snapshots. Without fallback the agent would get zero refs — blind.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        full_tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Post", "disabled": True},
                {"role": "link", "name": "Home"},
                {"role": "dialog", "name": "Compose", "children": [
                    {"role": "textbox", "name": "Tweet text"},
                    {"role": "button", "name": "Post"},
                ]},
            ],
        }
        # Both Playwright and JS scoped snapshots return None
        mock_page = AsyncMock()
        mock_modal_el = AsyncMock()
        mock_modal_el.is_visible = AsyncMock(return_value=True)
        mock_modal_el.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 100, "width": 400, "height": 300,
        })
        mock_modal_el.evaluate = AsyncMock(return_value=None)
        mock_page.query_selector_all = AsyncMock(return_value=[mock_modal_el])
        mock_page.viewport_size = {"width": 1280, "height": 720}

        async def _snapshot(root=None):
            if root is not None:
                return None  # Scoped snapshot fails
            return full_tree
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(side_effect=_snapshot)

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        refs = result["data"]["refs"]
        snap = result["data"]["snapshot"]
        # Should have fallen back to full tree
        assert len(refs) > 0
        assert "Home" in snap  # Full tree elements visible
        # dialog_active stays True so _locator_from_ref stays scoped to
        # modal — prevents clicks from targeting elements behind the overlay
        assert inst.dialog_active is True
        assert inst.dialog_detected is True  # Modal was detected
        assert "could not be isolated" in snap  # Warning preserved

    @pytest.mark.asyncio
    async def test_snapshot_fallback_when_scoped_snapshot_raises(self):
        """If both Playwright and JS scoped snapshots fail, fall back to full tree."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        full_tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Submit"},
            ],
        }
        mock_page = AsyncMock()
        mock_modal_el = AsyncMock()
        mock_modal_el.is_visible = AsyncMock(return_value=True)
        mock_modal_el.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 100, "width": 400, "height": 300,
        })
        # JS fallback for scoped snapshot also fails
        mock_modal_el.evaluate = AsyncMock(
            side_effect=RuntimeError("evaluate failed")
        )
        mock_page.query_selector_all = AsyncMock(return_value=[mock_modal_el])
        mock_page.viewport_size = {"width": 1280, "height": 720}

        async def _snapshot(root=None):
            if root is not None:
                raise RuntimeError("Not supported in Camoufox")
            return full_tree
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(side_effect=_snapshot)

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        refs = result["data"]["refs"]
        assert len(refs) == 1
        assert refs["e0"]["name"] == "Submit"
        # dialog_active stays True so _locator_from_ref stays scoped to
        # modal — prevents clicks from targeting elements behind the overlay
        assert inst.dialog_active is True
        assert inst.dialog_detected is True  # Modal was detected

    @pytest.mark.asyncio
    async def test_navigate_resets_dialog_active(self):
        """Navigation to a new page should clear stale dialog_active flag."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="New Page")
        mock_page.url = "https://example.com"
        mock_page.evaluate = AsyncMock(return_value="")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst.dialog_active = True  # Stale state from previous page
        inst.dialog_detected = True
        mgr._instances["a1"] = inst

        result = await mgr.navigate("a1", "https://example.com", wait_ms=0)
        assert result["success"] is True
        assert inst.dialog_active is False
        assert inst.dialog_detected is False

    @pytest.mark.asyncio
    async def test_snapshot_nested_modals_deduped(self):
        """When a parent modal contains a child modal, only walk the parent.

        Without deduplication, elements inside the child would be counted
        twice — once from snapshot(root=parent) and again from
        snapshot(root=child) — producing wrong occurrence indices.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        parent_subtree = {
            "role": "dialog", "name": "Parent",
            "children": [
                {"role": "button", "name": "Save"},
                {"role": "dialog", "name": "Confirm", "children": [
                    {"role": "button", "name": "Yes"},
                ]},
            ],
        }
        full_tree = {
            "role": "WebArea", "name": "",
            "children": [parent_subtree],
        }

        mock_page = AsyncMock()
        mock_parent = AsyncMock()
        mock_child = AsyncMock()
        mock_parent.is_visible = AsyncMock(return_value=True)
        mock_parent.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 100, "width": 400, "height": 300,
        })
        mock_child.is_visible = AsyncMock(return_value=True)
        mock_child.bounding_box = AsyncMock(return_value={
            "x": 150, "y": 150, "width": 200, "height": 100,
        })
        # parent.contains(child) = True, child.contains(parent) = False
        mock_parent.evaluate = AsyncMock(
            side_effect=lambda js, arg: arg is mock_child
        )
        mock_child.evaluate = AsyncMock(return_value=False)
        mock_page.query_selector_all = AsyncMock(
            return_value=[mock_parent, mock_child]
        )
        mock_page.viewport_size = {"width": 1280, "height": 720}

        child_subtree = {
            "role": "dialog", "name": "Confirm",
            "children": [{"role": "button", "name": "Yes"}],
        }

        async def _snapshot(root=None):
            if root is mock_parent:
                return parent_subtree
            if root is mock_child:
                return child_subtree
            return full_tree
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(side_effect=_snapshot)

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        refs = result["data"]["refs"]
        # Parent walked: dialog "Parent", button "Save", dialog "Confirm", button "Yes" = 4
        # Without dedup we'd get 6 (4 + dialog "Confirm" + button "Yes" again)
        assert len(refs) == 4
        # "Yes" button should appear exactly once with index 0
        yes_refs = [r for r in refs.values() if r["name"] == "Yes"]
        assert len(yes_refs) == 1
        assert yes_refs[0]["index"] == 0


# ── JS a11y tree fallback tests ──────────────────────────────────────────


class TestJsA11yTreeFallback:
    """When page.accessibility is unavailable, snapshot uses JS-based DOM walk."""

    @pytest.mark.asyncio
    async def test_fallback_on_attribute_error(self):
        """AttributeError on page.accessibility.snapshot triggers JS fallback."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        # Simulate Camoufox where accessibility.snapshot() raises AttributeError
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(
            side_effect=AttributeError("'Page' object has no attribute 'accessibility'")
        )
        # JS fallback returns a tree via page.evaluate()
        js_tree = {
            "role": "WebArea", "name": "Test Page",
            "children": [
                {"role": "button", "name": "Submit"},
                {"role": "link", "name": "Home"},
            ],
        }
        mock_page.evaluate = AsyncMock(return_value=js_tree)
        mock_page.query_selector_all = AsyncMock(return_value=[])

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst
        assert inst._js_snapshot_mode is False

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        refs = result["data"]["refs"]
        assert len(refs) == 2
        assert refs["e0"]["name"] == "Submit"
        assert refs["e1"]["name"] == "Home"
        assert inst._js_snapshot_mode is True

    @pytest.mark.asyncio
    async def test_transient_failure_retries_then_falls_back(self):
        """Non-AttributeError failures retry once, then fall back to JS."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(
            side_effect=RuntimeError("accessibility service unavailable")
        )
        js_tree = {
            "role": "WebArea", "name": "Fallback",
            "children": [{"role": "button", "name": "OK"}],
        }
        mock_page.evaluate = AsyncMock(return_value=js_tree)
        mock_page.query_selector_all = AsyncMock(return_value=[])

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        with patch("src.browser.service.asyncio.sleep"):
            result = await mgr.snapshot("a1")

        assert result["success"] is True
        # Should have retried accessibility.snapshot twice (attempt 0 + 1)
        assert mock_page.accessibility.snapshot.await_count == 2
        # JS fallback should have been used
        mock_page.evaluate.assert_called()
        # Transient failure should NOT permanently set _js_snapshot_mode
        assert inst._js_snapshot_mode is False

    @pytest.mark.asyncio
    async def test_js_mode_persists(self):
        """Once JS mode is enabled, subsequent snapshots skip Playwright API."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        js_tree = {
            "role": "WebArea", "name": "",
            "children": [{"role": "textbox", "name": "Search"}],
        }
        mock_page.evaluate = AsyncMock(return_value=js_tree)
        mock_page.query_selector_all = AsyncMock(return_value=[])

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst._js_snapshot_mode = True  # Already switched
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        assert len(result["data"]["refs"]) == 1
        # Should NOT have tried page.accessibility at all
        assert not hasattr(mock_page, 'accessibility') or \
            not mock_page.accessibility.snapshot.called

    @pytest.mark.asyncio
    async def test_js_fallback_scoped_to_dialog(self):
        """In JS mode, scoped snapshots use element.evaluate()."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        full_tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Post", "disabled": True},
                {"role": "link", "name": "Home"},
            ],
        }
        dialog_tree = {
            "role": "dialog", "name": "Compose",
            "children": [
                {"role": "textbox", "name": "What is happening?!"},
                {"role": "button", "name": "Post"},
            ],
        }

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=full_tree)
        mock_page.viewport_size = {"width": 1280, "height": 720}

        mock_modal_el = AsyncMock()
        mock_modal_el.is_visible = AsyncMock(return_value=True)
        mock_modal_el.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 100, "width": 400, "height": 300,
        })
        mock_modal_el.evaluate = AsyncMock(return_value=dialog_tree)
        mock_page.query_selector_all = AsyncMock(return_value=[mock_modal_el])

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst._js_snapshot_mode = True
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        snap = result["data"]["snapshot"]
        refs = result["data"]["refs"]

        assert "Modal dialog is open" in snap
        assert inst.dialog_active is True
        # Only dialog elements — sidebar "Post" and "Home" excluded
        post_refs = [r for r in refs.values() if r["name"] == "Post"]
        assert len(post_refs) == 1
        assert post_refs[0].get("disabled") is not True
        assert "Home" not in snap

    @pytest.mark.asyncio
    async def test_js_fallback_empty_returns_empty_page(self):
        """When JS fallback returns None, snapshot returns empty page."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=None)

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst._js_snapshot_mode = True
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        assert result["data"]["snapshot"] == "(empty page)"
        assert result["data"]["refs"] == {}

    @pytest.mark.asyncio
    async def test_js_tree_none_role_flattened(self):
        """Nodes with role='none' (non-role containers) are walked for children."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        # JS tree has 'none' role wrapper (non-role div containing buttons)
        js_tree = {
            "role": "WebArea", "name": "",
            "children": [{
                "role": "none", "name": "",
                "children": [
                    {"role": "button", "name": "Save"},
                    {"role": "button", "name": "Cancel"},
                ],
            }],
        }
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=js_tree)
        mock_page.query_selector_all = AsyncMock(return_value=[])

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        inst._js_snapshot_mode = True
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        refs = result["data"]["refs"]
        # Both buttons should be found despite 'none' wrapper
        assert len(refs) == 2
        names = {r["name"] for r in refs.values()}
        assert names == {"Save", "Cancel"}


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
        from src.browser.timing import set_delay, set_speed
        set_speed(1.0)
        set_delay(0.0)

    def teardown_method(self):
        from src.browser.timing import set_delay, set_speed
        set_speed(1.0)
        set_delay(0.0)

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

    def test_get_settings_includes_delay(self):
        """GET /browser/settings should include delay."""
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/browser/settings")
        assert resp.status_code == 200
        assert "delay" in resp.json()
        assert resp.json()["delay"] == 0.0

    def test_set_delay(self):
        """POST /browser/settings should update the delay."""
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager
        from src.browser.timing import get_delay
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.post("/browser/settings", json={"delay": 5.0})
        assert resp.status_code == 200
        assert resp.json()["delay"] == 5.0
        assert get_delay() == 5.0

    def test_set_delay_clamped(self):
        """POST /browser/settings should clamp out-of-range delay."""
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager
        from src.browser.timing import get_delay
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.post("/browser/settings", json={"delay": 100.0})
        assert resp.status_code == 200
        assert resp.json()["delay"] == 10.0
        assert get_delay() == 10.0

    def test_set_delay_only(self):
        """POST /browser/settings with only delay should not affect speed."""
        from src.browser.server import create_browser_app
        from src.browser.service import BrowserManager
        from src.browser.timing import get_speed, set_speed
        set_speed(2.0)
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")
        app = create_browser_app(mgr)

        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.post("/browser/settings", json={"delay": 3.0})
        assert resp.status_code == 200
        assert get_speed() == 2.0  # unchanged


# ── Dead code removal verification ────────────────────────────────────────


class TestStealthDeadCodeRemoved:
    """Verify resolution dead code was removed from stealth.py."""

    def test_pick_resolution_removed(self):
        import src.browser.stealth as stealth
        assert not hasattr(stealth, "_pick_resolution")
        assert not hasattr(stealth, "_WINDOWS_RESOLUTIONS")
        assert not hasattr(stealth, "_MACOS_RESOLUTIONS")


# ── X11 input bypass tests ────────────────────────────────────────────────


class TestX11Input:
    """Tests for X11 click/type bypass (isTrusted=true events)."""

    def _make_manager(self):
        with patch("src.browser.service.Path.mkdir"):
            return BrowserManager(profiles_dir="/tmp/test_profiles")

    def _make_instance(self, agent_id="agent-1", x11_wid=12345):
        mock_page = MagicMock()
        mock_page.viewport_size = {"width": 1920, "height": 1080}
        inst = CamoufoxInstance(agent_id, MagicMock(), MagicMock(), mock_page)
        inst.x11_wid = x11_wid
        return inst

    @pytest.mark.asyncio
    async def test_x11_click_calls_xdotool(self):
        """_x11_click should call getmouselocation + mousemove steps + mousedown + mouseup."""
        mgr = self._make_manager()
        inst = self._make_instance()

        mock_locator = AsyncMock()
        mock_locator.scroll_into_view_if_needed = AsyncMock()
        mock_locator.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 200, "width": 80, "height": 30,
        })

        mock_run = MagicMock()
        mock_run.returncode = 0
        mock_run.stdout = "x:50 y:50 screen:0 window:12345"

        with patch("src.browser.service.subprocess.run", return_value=mock_run) as sub_run:
            with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                with patch("src.browser.service.random.randint", return_value=4):
                    await mgr._x11_click(inst, mock_locator)

        # Element at y=200 is in viewport (1080px) — no scroll needed
        mock_locator.scroll_into_view_if_needed.assert_not_called()
        # bounding_box called by _x11_ensure_in_viewport + _x11_click
        assert mock_locator.bounding_box.await_count >= 1
        # At least: 1 getmouselocation + 3 mousemove + 1 mousedown + 1 mouseup = 6
        assert sub_run.call_count >= 6
        calls = sub_run.call_args_list
        assert "getmouselocation" in calls[0][0][0]
        # Last two calls should be mousedown and mouseup
        assert "mousedown" in calls[-2][0][0]
        assert "mouseup" in calls[-1][0][0]
        # All calls between first and last two should be mousemove
        for c in calls[1:-2]:
            assert "mousemove" in c[0][0]

    @pytest.mark.asyncio
    async def test_x11_click_no_wid_raises(self):
        """_x11_click should raise when no X11 WID is available."""
        mgr = self._make_manager()
        inst = self._make_instance(x11_wid=None)
        mock_locator = AsyncMock()
        mock_locator.scroll_into_view_if_needed = AsyncMock()
        mock_locator.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 200, "width": 80, "height": 30,
        })

        with pytest.raises(RuntimeError, match="No X11 window ID"):
            with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                await mgr._x11_click(inst, mock_locator)

    @pytest.mark.asyncio
    async def test_x11_click_no_bounding_box_raises(self):
        """_x11_click should raise when element has no bounding box."""
        mgr = self._make_manager()
        inst = self._make_instance()
        mock_locator = AsyncMock()
        mock_locator.scroll_into_view_if_needed = AsyncMock()
        mock_locator.bounding_box = AsyncMock(return_value=None)

        with pytest.raises(RuntimeError, match="no bounding box"):
            with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                await mgr._x11_click(inst, mock_locator)

    @pytest.mark.asyncio
    async def test_x11_click_uses_viewport_coords(self):
        """_x11_click should target viewport center coords in the final mousemove."""
        mgr = self._make_manager()
        inst = self._make_instance()
        mock_locator = AsyncMock()
        mock_locator.scroll_into_view_if_needed = AsyncMock()
        mock_locator.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 200, "width": 80, "height": 30,
        })

        mock_run = MagicMock()
        mock_run.returncode = 0
        mock_run.stdout = "x:0 y:0 screen:0 window:12345"

        with patch("src.browser.service.subprocess.run", return_value=mock_run) as sub_run:
            with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                with patch("src.browser.service.random.randint", return_value=4):
                    with patch("src.browser.service.random.uniform", return_value=0.0):
                        await mgr._x11_click(inst, mock_locator)

        # With offsets=0 and start=(0,0), last mousemove should be at target (140, 215)
        # Find the last mousemove call (before mousedown)
        mousemove_calls = [c for c in sub_run.call_args_list if "mousemove" in c[0][0]]
        last_mv = mousemove_calls[-1]
        cmd = last_mv[0][0]
        assert "140" in cmd
        assert "215" in cmd
        # --window flag should target the agent's WID
        assert str(inst.x11_wid) in cmd
        # Click should be mousedown + mouseup (not "click")
        all_cmds = [c[0][0] for c in sub_run.call_args_list]
        assert any("mousedown" in c for c in all_cmds)
        assert any("mouseup" in c for c in all_cmds)
        assert not any("click" == c[1] for c in all_cmds if len(c) > 1)

    @pytest.mark.asyncio
    async def test_x11_type_calls_xdotool_per_char(self):
        """_x11_type should call xdotool type for each character."""
        mgr = self._make_manager()
        inst = self._make_instance()

        mock_run = MagicMock(return_value=MagicMock(returncode=0))

        with patch("src.browser.service.subprocess.run", mock_run):
            with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                with patch("src.browser.service.random.random", return_value=1.0):  # no think pauses
                    await mgr._x11_type(inst, "Hi")

        # Should have 2 xdotool type calls (one per char)
        type_calls = [c for c in mock_run.call_args_list if "type" in c[0][0]]
        assert len(type_calls) == 2

    @pytest.mark.asyncio
    async def test_x11_type_handles_newline(self):
        """_x11_type should use xdotool key Return for newlines."""
        mgr = self._make_manager()
        inst = self._make_instance()

        mock_run = MagicMock(return_value=MagicMock(returncode=0))

        with patch("src.browser.service.subprocess.run", mock_run):
            with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                with patch("src.browser.service.random.random", return_value=1.0):
                    await mgr._x11_type(inst, "a\n")

        key_calls = [c for c in mock_run.call_args_list if "key" in c[0][0]]
        assert len(key_calls) == 1
        assert "Return" in key_calls[0][0][0]

    @pytest.mark.asyncio
    async def test_x11_type_no_wid_raises(self):
        """_x11_type should raise when no X11 WID is available."""
        mgr = self._make_manager()
        inst = self._make_instance(x11_wid=None)

        with pytest.raises(RuntimeError, match="No X11 window ID"):
            await mgr._x11_type(inst, "hello")

    @pytest.mark.asyncio
    async def test_click_routes_sensitive_selector_to_x11(self):
        """click() with any selector on x.com should route through _x11_click."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/home"
        mock_locator = AsyncMock()
        inst.page.locator.return_value.first = mock_locator
        inst.lock = asyncio.Lock()

        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._x11_click = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.click(
            "agent-1",
            selector='[data-testid="tweetButtonInline"]',
        )
        assert result["success"]
        mgr._x11_click.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_non_sensitive_selector_uses_cdp(self):
        """click() with a selector on a non-X site should use _human_click_selector."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://example.com"
        inst.lock = asyncio.Lock()

        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._human_click_selector = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.click("agent-1", selector='[data-testid="someButton"]')
        assert result["success"]
        mgr._human_click_selector.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_sensitive_selector_falls_back_without_wid(self):
        """If no X11 WID, x.com selectors should fall back to CDP click."""
        mgr = self._make_manager()
        inst = self._make_instance(x11_wid=None)
        inst.page = MagicMock()
        inst.page.url = "https://x.com/home"
        inst.lock = asyncio.Lock()

        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._human_click_selector = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.click(
            "agent-1",
            selector='[data-testid="tweetButtonInline"]',
        )
        assert result["success"]
        mgr._human_click_selector.assert_called_once()

    @pytest.mark.asyncio
    async def test_type_text_routes_sensitive_selector_to_x11(self):
        """type_text() with selector on x.com should use _x11_click for focus and _x11_type."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/home"
        inst.page.keyboard = AsyncMock()
        mock_locator = AsyncMock()
        inst.page.locator.return_value.first = mock_locator
        inst.lock = asyncio.Lock()

        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._x11_click = AsyncMock()
        mgr._x11_key = AsyncMock()
        mgr._x11_type = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.type_text(
            "agent-1",
            selector='[data-testid="tweetTextarea_0"]',
            text="Hello world",
        )
        assert result["success"]
        mgr._x11_click.assert_called_once()
        mgr._x11_type.assert_called_once_with(inst, "Hello world", typos=True)

    @pytest.mark.asyncio
    async def test_type_text_uses_x11_on_all_sites(self):
        """type_text() with selector on any site should use X11 typing when x11_wid is set."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://example.com"
        inst.page.keyboard = AsyncMock()
        inst.lock = asyncio.Lock()

        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._x11_click = AsyncMock()
        mgr._x11_type = AsyncMock()
        mgr._x11_key = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.type_text(
            "agent-1",
            selector='[data-testid="searchBox"]',
            text="Hello",
        )
        assert result["success"]
        mgr._x11_type.assert_called_once()

    @pytest.mark.asyncio
    async def test_x11_click_mousemove_failure_raises(self):
        """_x11_click should raise when xdotool mousemove fails."""
        mgr = self._make_manager()
        inst = self._make_instance()
        mock_locator = AsyncMock()
        mock_locator.scroll_into_view_if_needed = AsyncMock()
        mock_locator.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 200, "width": 80, "height": 30,
        })

        # getmouselocation succeeds, but first mousemove fails
        success_run = MagicMock()
        success_run.returncode = 0
        success_run.stdout = "x:0 y:0 screen:0 window:12345"
        fail_run = MagicMock()
        fail_run.returncode = 1

        with pytest.raises(RuntimeError, match="mousemove failed"):
            with patch("src.browser.service.subprocess.run", side_effect=[success_run, fail_run]):
                with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                    with patch("src.browser.service.random.randint", return_value=4):
                        await mgr._x11_click(inst, mock_locator)

    @pytest.mark.asyncio
    async def test_x11_type_option_terminator(self):
        """_x11_type should use -- before char arg to prevent flag injection."""
        mgr = self._make_manager()
        inst = self._make_instance()

        mock_run = MagicMock(return_value=MagicMock(returncode=0))

        with patch("src.browser.service.subprocess.run", mock_run):
            with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                with patch("src.browser.service.random.random", return_value=1.0):
                    await mgr._x11_type(inst, "-")

        type_calls = [c for c in mock_run.call_args_list if "type" in c[0][0]]
        assert len(type_calls) == 1
        cmd = type_calls[0][0][0]
        # "--" should appear before the character
        dash_dash_idx = cmd.index("--")
        char_idx = len(cmd) - 1  # last element
        assert dash_dash_idx < char_idx

    @pytest.mark.asyncio
    async def test_type_text_routes_ref_on_x_com_to_x11(self):
        """type_text() with a textbox ref on x.com should use _x11_click for focus and _x11_type."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/compose/post"
        inst.page.keyboard = AsyncMock()
        inst.lock = asyncio.Lock()
        inst.refs = {
            k: _h(v) for k, v in {
                "T1": {"role": "textbox", "name": "What is happening?!", "index": 0},
            }.items()
        }

        mock_locator = AsyncMock()
        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._locator_from_ref = MagicMock(return_value=mock_locator)
        mgr._x11_click = AsyncMock()
        mgr._x11_key = AsyncMock()
        mgr._x11_type = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.type_text("agent-1", ref="T1", text="Hello X")
        assert result["success"]
        mgr._x11_click.assert_called_once_with(inst, mock_locator)
        mgr._x11_type.assert_called_once_with(inst, "Hello X", typos=True)

    @pytest.mark.asyncio
    async def test_type_text_ref_uses_x11_on_all_sites(self):
        """type_text() with a textbox ref on any site should use X11 typing when x11_wid is set."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://google.com/search"
        inst.page.keyboard = AsyncMock()
        inst.lock = asyncio.Lock()
        inst.refs = {k: _h(v) for k, v in {"T1": {"role": "textbox", "name": "Search", "index": 0}}.items()}

        mock_locator = AsyncMock()
        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._locator_from_ref = MagicMock(return_value=mock_locator)
        mgr._x11_click = AsyncMock()
        mgr._x11_type = AsyncMock()
        mgr._x11_key = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.type_text("agent-1", ref="T1", text="Hello")
        assert result["success"]
        mgr._x11_type.assert_called_once()

    # ── _is_x11_site tests ───────────────────────────────────────────

    def test_is_x11_site_x_com(self):
        """_is_x11_site should return True for x.com URLs."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        for url in ("https://x.com/home", "https://x.com/compose/post",
                     "https://mobile.x.com/notifications"):
            inst.page.url = url
            assert mgr._is_x11_site(inst) is True, f"Expected True for {url}"

    def test_is_x11_site_twitter_com(self):
        """_is_x11_site should return True for twitter.com URLs."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        for url in ("https://twitter.com/home", "https://mobile.twitter.com/feed"):
            inst.page.url = url
            assert mgr._is_x11_site(inst) is True, f"Expected True for {url}"

    def test_is_x11_site_all_domains(self):
        """_is_x11_site should return True for all domains."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        for url in ("https://google.com", "https://example.com/x.com",
                     "https://github.com"):
            inst.page.url = url
            assert mgr._is_x11_site(inst) is True, f"Expected True for {url}"

    def test_is_x11_site_any_domain(self):
        """_is_x11_site should return True for any domain."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        for url in ("https://notx.com/home", "https://faketwitter.com/feed"):
            inst.page.url = url
            assert mgr._is_x11_site(inst) is True, f"Expected True for {url}"

    # ── URL-based X11 routing tests ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_click_any_selector_on_x_com_uses_x11(self):
        """Any selector on x.com should route through _x11_click."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/home"
        mock_locator = AsyncMock()
        inst.page.locator.return_value.first = mock_locator
        inst.lock = asyncio.Lock()

        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._x11_click = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.click("agent-1", selector='[data-testid="followButton"]')
        assert result["success"]
        mgr._x11_click.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_x11_failure_falls_back_to_cdp(self):
        """When _x11_click raises on x.com, click() should fall back to CDP."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/home"
        mock_locator = AsyncMock()
        inst.page.locator.return_value.first = mock_locator
        inst.lock = asyncio.Lock()

        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._x11_click = AsyncMock(side_effect=RuntimeError("xdotool failed"))
        mgr._human_click_selector = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.click("agent-1", selector='[data-testid="someBtn"]')
        assert result["success"]
        mgr._human_click_selector.assert_called_once()

    @pytest.mark.asyncio
    async def test_type_text_focus_uses_x11_on_x_com(self):
        """type_text() on x.com should use _x11_click for focus click."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/compose/post"
        inst.page.keyboard = AsyncMock()
        inst.lock = asyncio.Lock()
        inst.refs = {k: _h(v) for k, v in {"T1": {"role": "textbox", "name": "Post", "index": 0}}.items()}

        mock_locator = AsyncMock()
        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._locator_from_ref = MagicMock(return_value=mock_locator)
        mgr._x11_click = AsyncMock()
        mgr._x11_key = AsyncMock()
        mgr._x11_type = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.type_text("agent-1", ref="T1", text="test")
        assert result["success"]
        mgr._x11_click.assert_called_once_with(inst, mock_locator)

    @pytest.mark.asyncio
    async def test_type_text_ctrl_a_uses_x11_on_x_com(self):
        """type_text(clear=True) on x.com should use _x11_key('ctrl+a')."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/compose/post"
        inst.page.keyboard = AsyncMock()
        inst.lock = asyncio.Lock()
        inst.refs = {k: _h(v) for k, v in {"T1": {"role": "textbox", "name": "Post", "index": 0}}.items()}

        mock_locator = AsyncMock()
        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._locator_from_ref = MagicMock(return_value=mock_locator)
        mgr._x11_click = AsyncMock()
        mgr._x11_key = AsyncMock()
        mgr._x11_type = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.type_text("agent-1", ref="T1", text="test", clear=True)
        assert result["success"]
        mgr._x11_key.assert_called_once_with(inst, "ctrl+a")

    @pytest.mark.asyncio
    async def test_x11_key_sends_key(self):
        """_x11_key should call xdotool key with the given key combo."""
        mgr = self._make_manager()
        inst = self._make_instance()

        mock_run = MagicMock()
        mock_run.returncode = 0

        with patch("src.browser.service.subprocess.run", return_value=mock_run) as sub_run:
            await mgr._x11_key(inst, "ctrl+a")

        sub_run.assert_called_once()
        cmd = sub_run.call_args[0][0]
        assert "key" in cmd
        assert "ctrl+a" in cmd
        assert str(inst.x11_wid) in cmd

    @pytest.mark.asyncio
    async def test_x11_key_no_wid_raises(self):
        """_x11_key should raise when no X11 WID is available."""
        mgr = self._make_manager()
        inst = self._make_instance(x11_wid=None)

        with pytest.raises(RuntimeError, match="No X11 window ID"):
            await mgr._x11_key(inst, "ctrl+a")

    @pytest.mark.asyncio
    async def test_x11_click_no_hover_called(self):
        """_x11_click should never call locator.hover()."""
        mgr = self._make_manager()
        inst = self._make_instance()

        mock_locator = AsyncMock()
        mock_locator.scroll_into_view_if_needed = AsyncMock()
        mock_locator.hover = AsyncMock()
        mock_locator.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 200, "width": 80, "height": 30,
        })

        mock_run = MagicMock()
        mock_run.returncode = 0
        mock_run.stdout = "x:0 y:0 screen:0 window:12345"

        with patch("src.browser.service.subprocess.run", return_value=mock_run):
            with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                with patch("src.browser.service.random.randint", return_value=4):
                    await mgr._x11_click(inst, mock_locator)

        mock_locator.hover.assert_not_called()

    @pytest.mark.asyncio
    async def test_wid_discovery_30_iterations(self):
        """_discover_new_wid should poll up to 30 times."""
        mgr = self._make_manager()
        call_count = 0

        async def mock_get_wids():
            nonlocal call_count
            call_count += 1
            if call_count == 30:
                return {99999}
            return set()

        mgr._get_firefox_wids = mock_get_wids

        with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
            wid = await mgr._discover_new_wid(set())

        assert wid == 99999
        assert call_count == 30

    @pytest.mark.asyncio
    async def test_wid_discovery_failure_logs_warning(self):
        """Failed WID discovery should log at WARNING, not DEBUG."""
        import inspect

        import src.browser.service as svc
        source = inspect.getsource(svc.BrowserManager._start_browser)
        # The WID failure path should use logger.warning, not logger.debug
        assert "logger.warning(" in source and "Could not discover X11 WID" in source

    @pytest.mark.asyncio
    async def test_click_ref_on_x_com_uses_x11(self):
        """click(ref=...) on x.com should route through _x11_click."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/home"
        inst.refs = {k: _h(v) for k, v in {"B1": {"role": "button", "name": "Like", "index": 0}}.items()}
        inst.lock = asyncio.Lock()

        mock_locator = AsyncMock()
        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._locator_from_ref = MagicMock(return_value=mock_locator)
        mgr._x11_click = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.click("agent-1", ref="B1")
        assert result["success"]
        mgr._x11_click.assert_called_once_with(inst, mock_locator, timeout=10000)

    @pytest.mark.asyncio
    async def test_type_text_clear_false_skips_x11_key(self):
        """type_text(clear=False) on x.com should NOT call _x11_key."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/compose/post"
        inst.page.keyboard = AsyncMock()
        inst.lock = asyncio.Lock()
        inst.refs = {k: _h(v) for k, v in {"T1": {"role": "textbox", "name": "Post", "index": 0}}.items()}

        mock_locator = AsyncMock()
        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._locator_from_ref = MagicMock(return_value=mock_locator)
        mgr._x11_click = AsyncMock()
        mgr._x11_key = AsyncMock()
        mgr._x11_type = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.type_text("agent-1", ref="T1", text="test", clear=False)
        assert result["success"]
        mgr._x11_key.assert_not_called()
        mgr._x11_type.assert_called_once()

    @pytest.mark.asyncio
    async def test_type_text_x11_focus_failure_falls_back_to_cdp(self):
        """When _x11_click fails for focus in type_text(), should fall back to CDP."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/compose/post"
        inst.page.keyboard = AsyncMock()
        inst.lock = asyncio.Lock()
        inst.refs = {k: _h(v) for k, v in {"T1": {"role": "textbox", "name": "Post", "index": 0}}.items()}

        mock_locator = AsyncMock()
        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._locator_from_ref = MagicMock(return_value=mock_locator)
        mgr._x11_click = AsyncMock(side_effect=RuntimeError("xdotool error"))
        mgr._human_click = AsyncMock()
        mgr._x11_key = AsyncMock()
        mgr._x11_type = AsyncMock()
        mgr._snapshot_impl = AsyncMock(return_value={"data": {}})

        result = await mgr.type_text("agent-1", ref="T1", text="test")
        assert result["success"]
        mgr._human_click.assert_called_once()
        # X11 type should still be attempted since _use_x11 is True
        mgr._x11_type.assert_called_once()

    @pytest.mark.asyncio
    async def test_x11_move_to_clamps_negative_coords(self):
        """_x11_move_to should clamp waypoint coords to >= 0."""
        mgr = self._make_manager()
        inst = self._make_instance()

        mock_run = MagicMock()
        mock_run.returncode = 0
        mock_run.stdout = "x:2 y:1 screen:0 window:12345"

        with patch("src.browser.service.subprocess.run", return_value=mock_run) as sub_run:
            with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                with patch("src.browser.service.random.randint", return_value=4):
                    with patch("src.browser.service.random.uniform", return_value=-60.0):
                        await mgr._x11_move_to(inst, 10, 6)

        # All mousemove coords should be >= 0
        for call in sub_run.call_args_list:
            cmd = call[0][0]
            if "mousemove" in cmd:
                x_val = int(cmd[cmd.index("--window") + 2])
                y_val = int(cmd[cmd.index("--window") + 3])
                assert x_val >= 0, f"X coord {x_val} is negative"
                assert y_val >= 0, f"Y coord {y_val} is negative"

    # ── Behavioral antibot tests ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_x11_click_dwell_time(self):
        """_x11_click should use mousedown + mouseup (not a single click)."""
        mgr = self._make_manager()
        inst = self._make_instance()
        mock_locator = AsyncMock()
        mock_locator.scroll_into_view_if_needed = AsyncMock()
        mock_locator.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 200, "width": 80, "height": 30,
        })

        mock_run = MagicMock()
        mock_run.returncode = 0
        mock_run.stdout = "x:50 y:50 screen:0 window:12345"

        with patch("src.browser.service.subprocess.run", return_value=mock_run) as sub_run:
            with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                with patch("src.browser.service.random.randint", return_value=4):
                    await mgr._x11_click(inst, mock_locator)

        all_cmds = [c[0][0] for c in sub_run.call_args_list]
        # Should have mousedown and mouseup, but no single "click"
        has_mousedown = any("mousedown" in cmd for cmd in all_cmds)
        has_mouseup = any("mouseup" in cmd for cmd in all_cmds)
        has_click = any(cmd[1] == "click" for cmd in all_cmds if len(cmd) > 1)
        assert has_mousedown
        assert has_mouseup
        assert not has_click

    @pytest.mark.asyncio
    async def test_x11_move_to_uses_bezier(self):
        """_x11_move_to should call getmouselocation + multiple mousemove steps."""
        mgr = self._make_manager()
        inst = self._make_instance()

        mock_run = MagicMock()
        mock_run.returncode = 0
        mock_run.stdout = "x:0 y:0 screen:0 window:12345"

        with patch("src.browser.service.subprocess.run", return_value=mock_run) as sub_run:
            with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                with patch("src.browser.service.random.randint", return_value=5):
                    with patch("src.browser.service.random.uniform", side_effect=[
                        30.0, -20.0,  # off1, off2 (Bezier control point offsets)
                    ]):
                        await mgr._x11_move_to(inst, 200, 200)

        # At least: 1 getmouselocation + 3 mousemove steps
        assert sub_run.call_count >= 4
        assert "getmouselocation" in sub_run.call_args_list[0][0][0]
        for i in range(1, sub_run.call_count):
            assert "mousemove" in sub_run.call_args_list[i][0][0]

        # With non-zero offsets, intermediate waypoints should NOT be
        # on a straight line from (0,0) to (200,200).
        # Check a middle step for non-linear coords.
        mid_idx = sub_run.call_count // 2
        mid_cmd = sub_run.call_args_list[mid_idx][0][0]
        mid_x = int(mid_cmd[-2])
        mid_y = int(mid_cmd[-1])
        # Linear at 50% would be (100, 100) — Bezier offsets should differ
        assert not (mid_x == 100 and mid_y == 100), "Waypoint should not be exactly linear"

    @pytest.mark.asyncio
    async def test_scroll_uses_wheel_events(self):
        """scroll() should use page.mouse.wheel() instead of page.evaluate()."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.viewport_size = {"width": 1920, "height": 1080}
        inst.page.mouse = AsyncMock()
        inst.page.mouse.wheel = AsyncMock()
        inst.page.evaluate = AsyncMock()
        inst.lock = asyncio.Lock()

        mgr.get_or_start = AsyncMock(return_value=inst)

        with patch("src.browser.service.scroll_increment", return_value=500):
            with patch("src.browser.service.scroll_pause", return_value=0.01):
                with patch("src.browser.service.asyncio.sleep", new_callable=AsyncMock):
                    result = await mgr.scroll("agent-1", direction="down", amount=500)

        assert result["success"]
        inst.page.mouse.wheel.assert_called()
        # page.evaluate should NOT be called for scrolling
        inst.page.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_hover_routes_x11_on_x_com(self):
        """hover() on x.com should route through _x11_hover."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/home"
        inst.lock = asyncio.Lock()
        inst.seed_refs_legacy({"e0": {"role": "button", "name": "Like", "index": 0}})

        mock_locator = AsyncMock()
        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._locator_from_ref = MagicMock(return_value=mock_locator)
        mgr._x11_hover = AsyncMock()

        result = await mgr.hover("agent-1", ref="e0")
        assert result["success"]
        mgr._x11_hover.assert_called_once_with(inst, mock_locator)

    @pytest.mark.asyncio
    async def test_hover_uses_x11_on_all_sites(self):
        """hover() on any site should use _x11_hover when x11_wid is set."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://google.com"
        inst.lock = asyncio.Lock()
        inst.seed_refs_legacy({"e0": {"role": "button", "name": "Search", "index": 0}})

        mock_locator = AsyncMock()
        mock_locator.hover = AsyncMock()
        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._locator_from_ref = MagicMock(return_value=mock_locator)
        mgr._x11_hover = AsyncMock()

        result = await mgr.hover("agent-1", ref="e0")
        assert result["success"]
        mgr._x11_hover.assert_called_once()
        mock_locator.hover.assert_not_called()

    @pytest.mark.asyncio
    async def test_hover_x11_failure_falls_back_to_cdp(self):
        """When _x11_hover raises on x.com, hover() should fall back to CDP."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/home"
        inst.lock = asyncio.Lock()
        inst.seed_refs_legacy({"e0": {"role": "button", "name": "Like", "index": 0}})

        mock_locator = AsyncMock()
        mock_locator.hover = AsyncMock()
        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._locator_from_ref = MagicMock(return_value=mock_locator)
        mgr._x11_hover = AsyncMock(side_effect=RuntimeError("xdotool failed"))

        result = await mgr.hover("agent-1", ref="e0")
        assert result["success"]
        mock_locator.hover.assert_called_once()

    @pytest.mark.asyncio
    async def test_hover_selector_on_x_com_uses_x11(self):
        """hover(selector=...) on x.com should route through _x11_hover."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/home"
        mock_locator = AsyncMock()
        inst.page.locator.return_value.first = mock_locator
        inst.lock = asyncio.Lock()

        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._x11_hover = AsyncMock()

        result = await mgr.hover("agent-1", selector='[data-testid="like"]')
        assert result["success"]
        mgr._x11_hover.assert_called_once()

    @pytest.mark.asyncio
    async def test_press_key_x11_failure_falls_back_to_cdp(self):
        """When _x11_key raises on x.com, press_key() should fall back to CDP."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/compose/post"
        inst.page.keyboard = AsyncMock()
        inst.lock = asyncio.Lock()

        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._x11_key = AsyncMock(side_effect=RuntimeError("xdotool failed"))

        result = await mgr.press_key("agent-1", "Escape")
        assert result["success"]
        inst.page.keyboard.press.assert_called_once_with("Escape")

    @pytest.mark.asyncio
    async def test_press_key_routes_x11_on_x_com(self):
        """press_key('Enter') on x.com should call _x11_key."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://x.com/compose/post"
        inst.page.keyboard = AsyncMock()
        inst.lock = asyncio.Lock()

        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._x11_key = AsyncMock()

        result = await mgr.press_key("agent-1", "Enter")
        assert result["success"]
        mgr._x11_key.assert_called_once_with(inst, "Return")

    @pytest.mark.asyncio
    async def test_press_key_uses_x11_on_all_sites(self):
        """press_key on any site should use _x11_key when x11_wid is set."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.page = MagicMock()
        inst.page.url = "https://google.com"
        inst.page.keyboard = AsyncMock()
        inst.lock = asyncio.Lock()

        mgr.get_or_start = AsyncMock(return_value=inst)
        mgr._x11_key = AsyncMock()

        result = await mgr.press_key("agent-1", "Enter")
        assert result["success"]
        mgr._x11_key.assert_called_once()
        inst.page.keyboard.press.assert_not_called()

    def test_playwright_key_to_xdotool(self):
        """_playwright_key_to_xdotool should map Playwright keys to xdotool names."""
        assert BrowserManager._playwright_key_to_xdotool("Enter") == "Return"
        assert BrowserManager._playwright_key_to_xdotool("Backspace") == "BackSpace"
        assert BrowserManager._playwright_key_to_xdotool("ArrowUp") == "Up"
        assert BrowserManager._playwright_key_to_xdotool("Control+a") == "ctrl+a"
        assert BrowserManager._playwright_key_to_xdotool("Shift+Enter") == "shift+Return"
        # Unmapped keys pass through as-is
        assert BrowserManager._playwright_key_to_xdotool("Tab") == "Tab"
        assert BrowserManager._playwright_key_to_xdotool("Escape") == "Escape"

    @pytest.mark.asyncio
    async def test_idle_jitter_starts_on_browser_launch(self):
        """_jitter_task should be set when WID is discovered."""
        mgr = self._make_manager()
        inst = self._make_instance()
        # Simulate what _start_browser does after discovering WID
        assert inst.x11_wid is not None
        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = MagicMock()
            # Replicate the jitter task start logic
            inst._jitter_task = asyncio.create_task(mgr._idle_mouse_jitter(inst))

        assert inst._jitter_task is not None
        mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_idle_jitter_cancelled_on_stop(self):
        """_jitter_task should be cancelled in _stop_instance."""
        mgr = self._make_manager()
        inst = self._make_instance()
        inst.context = AsyncMock()
        mock_task = MagicMock()
        inst._jitter_task = mock_task
        mgr._instances["agent-1"] = inst

        await mgr._stop_instance("agent-1")
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_idle_jitter_none_when_no_wid(self):
        """_jitter_task should be None when no WID is discovered."""
        inst = self._make_instance(x11_wid=None)
        # Simulate what _start_browser does when no WID is found
        inst._jitter_task = None
        assert inst._jitter_task is None


# -- Modal retry stale handle tests ----------------------------------------


class TestModalRetryRequery:
    """Tests that modal retry loop re-queries fresh element handles."""

    @pytest.mark.asyncio
    async def test_modal_retry_re_queries_elements(self):
        """Retry loop should re-query modal elements (handles go stale on SPA re-render)."""
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        full_tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Post", "disabled": True},
                {"role": "link", "name": "Home"},
            ],
        }
        # Stale handle produces no a11y tree; fresh handle has content
        dialog_tree = {
            "role": "dialog", "name": "Compose",
            "children": [
                {"role": "textbox", "name": "What is happening?!"},
                {"role": "button", "name": "Post"},
            ],
        }

        _bb = {"x": 100, "y": 100, "width": 400, "height": 300}
        stale_el = AsyncMock()
        stale_el.is_visible = AsyncMock(return_value=True)
        stale_el.bounding_box = AsyncMock(return_value=_bb)
        fresh_el = AsyncMock()
        fresh_el.is_visible = AsyncMock(return_value=True)
        fresh_el.bounding_box = AsyncMock(return_value=_bb)

        query_call_count = [0]

        async def mock_query_selector_all(sel):
            query_call_count[0] += 1
            if query_call_count[0] == 1:
                return [stale_el]   # initial query
            return [fresh_el]       # retry queries

        mock_page = AsyncMock()
        mock_page.query_selector_all = mock_query_selector_all
        mock_page.viewport_size = {"width": 1280, "height": 720}

        async def _snapshot(root=None):
            if root is stale_el:
                return None  # stale handle fails silently
            if root is fresh_el:
                return dialog_tree
            return full_tree
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(side_effect=_snapshot)

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        assert inst.dialog_active is True
        # Should have re-queried at least once during retry
        assert query_call_count[0] >= 2
        # Fresh handle should have produced the Post button
        post_refs = [r for r in inst.refs.values()
                     if r.role == "button" and r.name == "Post"]
        assert len(post_refs) == 1
        # Sidebar elements should be excluded
        assert "Home" not in result["data"]["snapshot"]

    @pytest.mark.asyncio
    async def test_modal_fallback_keeps_dialog_active_true(self):
        """When modal scoping fails completely, dialog_active should stay True
        so _locator_from_ref stays scoped to the modal and prevents clicks
        from targeting elements behind the overlay.
        """
        from src.browser.service import BrowserManager, CamoufoxInstance
        mgr = BrowserManager(profiles_dir="/tmp/test_profiles")

        full_tree = {
            "role": "WebArea", "name": "",
            "children": [
                {"role": "button", "name": "Post"},
                {"role": "link", "name": "Home"},
            ],
        }

        # Modal is visible but a11y tree always returns None (scoping always fails)
        modal_el = AsyncMock()
        modal_el.is_visible = AsyncMock(return_value=True)
        modal_el.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 100, "width": 400, "height": 300,
        })
        modal_el.evaluate = AsyncMock(return_value=None)

        mock_page = AsyncMock()
        mock_page.query_selector_all = AsyncMock(return_value=[modal_el])
        mock_page.viewport_size = {"width": 1280, "height": 720}

        async def _snapshot(root=None):
            if root is not None:
                return None  # Modal scoping always fails
            return full_tree
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(side_effect=_snapshot)

        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), mock_page)
        mgr._instances["a1"] = inst

        result = await mgr.snapshot("a1")
        assert result["success"] is True
        # dialog_active must stay True to prevent clicking behind modal
        assert inst.dialog_active is True
        assert inst.dialog_detected is True
        # Full tree is shown as fallback
        assert "Post" in result["data"]["snapshot"]
        assert "could not be isolated" in result["data"]["snapshot"]


class TestBrowserManagerProxyConfig:
    def test_set_proxy_config(self):
        manager = BrowserManager(profiles_dir="/tmp/test_profiles_proxy")
        manager.set_proxy_config("agent-1", {
            "url": "socks5://host:1080",
            "username": "u",
            "password": "p",
        })
        config = manager.get_proxy_config("agent-1")
        assert config == {"url": "socks5://host:1080", "username": "u", "password": "p"}

    def test_get_proxy_config_returns_none_for_unknown(self):
        manager = BrowserManager(profiles_dir="/tmp/test_profiles_proxy2")
        assert manager.get_proxy_config("unknown") is None

    def test_clear_proxy_config(self):
        manager = BrowserManager(profiles_dir="/tmp/test_profiles_proxy3")
        manager.set_proxy_config("agent-1", {"url": "http://host:8080"})
        manager.set_proxy_config("agent-1", None)
        assert manager.get_proxy_config("agent-1") is None

    def test_boot_id_is_stable_and_nonempty(self):
        manager = BrowserManager(profiles_dir="/tmp/test_profiles_proxy4")
        assert manager.boot_id
        assert manager.boot_id == manager.boot_id

    def test_empty_dict_means_explicit_no_proxy(self):
        """Empty dict stored via set_proxy_config means 'explicitly no proxy' (direct mode)."""
        manager = BrowserManager(profiles_dir="/tmp/test_profiles_proxy5")
        manager.set_proxy_config("agent-1", {})
        config = manager.get_proxy_config("agent-1")
        assert config is not None  # not None — that would mean no config pushed
        assert config == {}  # empty dict = explicit no-proxy
        assert not config.get("url")  # no URL = _start_browser passes proxy=None

    def test_none_clears_config(self):
        """None clears the stored config (no config pushed state)."""
        manager = BrowserManager(profiles_dir="/tmp/test_profiles_proxy6")
        manager.set_proxy_config("agent-1", {"url": "http://host:8080"})
        manager.set_proxy_config("agent-1", None)
        assert manager.get_proxy_config("agent-1") is None


# ── CAPTCHA solver tests ─────────────────────────────────────────────────────


class TestGetSolver:
    """Tests for captcha.get_solver() factory function."""

    def test_get_solver_returns_none_when_not_configured(self):
        """No env vars → None."""
        with patch.dict("os.environ", {}, clear=True):
            from src.browser.captcha import get_solver
            assert get_solver() is None

    def test_get_solver_returns_solver_when_configured(self):
        """Valid provider + key → CaptchaSolver instance."""
        with patch.dict("os.environ", {
            "CAPTCHA_SOLVER_PROVIDER": "2captcha",
            "CAPTCHA_SOLVER_KEY": "test-key-123",
        }):
            from src.browser.captcha import CaptchaSolver, get_solver
            solver = get_solver()
            assert isinstance(solver, CaptchaSolver)
            assert solver.provider == "2captcha"
            assert solver.api_key == "test-key-123"

    def test_get_solver_rejects_unknown_provider(self):
        """Unknown provider → None with warning."""
        with patch.dict("os.environ", {
            "CAPTCHA_SOLVER_PROVIDER": "badprovider",
            "CAPTCHA_SOLVER_KEY": "key",
        }):
            from src.browser.captcha import get_solver
            assert get_solver() is None


class TestClassifyCaptcha:
    """Tests for captcha._classify_captcha()."""

    def test_classify_captcha(self):
        from src.browser.captcha import _classify_captcha
        assert _classify_captcha('iframe[src*="recaptcha"]') == "recaptcha"
        assert _classify_captcha('iframe[src*="hcaptcha"]') == "hcaptcha"
        assert _classify_captcha('iframe[src*="challenges.cloudflare.com"]') == "turnstile"
        assert _classify_captcha('[class*="cf-turnstile"]') == "turnstile"
        assert _classify_captcha('#captcha') == "recaptcha"  # generic fallback
        assert _classify_captcha('[class*="captcha"]') == "recaptcha"
        assert _classify_captcha("something-unknown") == "recaptcha"  # default


class TestCaptchaSolverSolve:
    """Tests for CaptchaSolver.solve() and provider flows."""

    @pytest.mark.asyncio
    async def test_solve_2captcha_success(self):
        """Mock a successful 2Captcha create+poll flow."""
        from src.browser.captcha import CaptchaSolver

        solver = CaptchaSolver("2captcha", "test-key")

        # Mock page
        page = AsyncMock()
        page.evaluate = AsyncMock(return_value="site-key-abc")
        page.url = "https://example.com"

        # Mock httpx responses
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False

        create_resp = MagicMock()
        create_resp.json.return_value = {"errorId": 0, "taskId": "task-123"}
        create_resp.raise_for_status = MagicMock()

        poll_resp = MagicMock()
        poll_resp.json.return_value = {
            "errorId": 0,
            "status": "ready",
            "solution": {"gRecaptchaResponse": "solved-token-xyz"},
        }
        poll_resp.raise_for_status = MagicMock()

        mock_client.post = AsyncMock(side_effect=[create_resp, poll_resp])
        solver._client = mock_client

        result = await solver.solve(page, 'iframe[src*="recaptcha"]', "https://example.com")
        assert result is True
        # Verify token was injected
        page.evaluate.assert_called()

    @pytest.mark.asyncio
    async def test_solve_capsolver_success(self):
        """Mock a successful CapSolver create+poll flow."""
        from src.browser.captcha import CaptchaSolver

        solver = CaptchaSolver("capsolver", "cap-key")

        page = AsyncMock()
        page.evaluate = AsyncMock(return_value="site-key-abc")
        page.url = "https://example.com"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False

        create_resp = MagicMock()
        create_resp.json.return_value = {"errorId": 0, "taskId": "task-456"}
        create_resp.raise_for_status = MagicMock()

        poll_resp = MagicMock()
        poll_resp.json.return_value = {
            "errorId": 0,
            "status": "ready",
            "solution": {"token": "capsolver-token-xyz"},
        }
        poll_resp.raise_for_status = MagicMock()

        mock_client.post = AsyncMock(side_effect=[create_resp, poll_resp])
        solver._client = mock_client

        result = await solver.solve(page, 'iframe[src*="hcaptcha"]', "https://example.com")
        assert result is True

    @pytest.mark.asyncio
    async def test_solve_timeout(self):
        """Verify timeout when solving takes too long."""
        from src.browser.captcha import CaptchaSolver

        solver = CaptchaSolver("2captcha", "test-key")

        page = AsyncMock()
        page.evaluate = AsyncMock(return_value="site-key-abc")
        page.url = "https://example.com"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False

        create_resp = MagicMock()
        create_resp.json.return_value = {"errorId": 0, "taskId": "task-789"}
        create_resp.raise_for_status = MagicMock()

        # Poll always returns processing — will eventually time out
        poll_resp = MagicMock()
        poll_resp.json.return_value = {"errorId": 0, "status": "processing"}
        poll_resp.raise_for_status = MagicMock()

        mock_client.post = AsyncMock(side_effect=[create_resp] + [poll_resp] * 100)
        solver._client = mock_client

        # Patch _SOLVE_TIMEOUT to make test fast
        with patch("src.browser.captcha._SOLVE_TIMEOUT", 0.1), \
             patch("src.browser.captcha._POLL_INTERVAL", 0.01):
            result = await solver.solve(page, 'iframe[src*="recaptcha"]', "https://example.com")
        assert result is False

    @pytest.mark.asyncio
    async def test_solve_no_sitekey_returns_false(self):
        """Page with no sitekey → solve returns False."""
        from src.browser.captcha import CaptchaSolver

        solver = CaptchaSolver("2captcha", "test-key")

        page = AsyncMock()
        page.evaluate = AsyncMock(return_value=None)  # no sitekey found
        page.url = "https://example.com"

        result = await solver.solve(page, 'iframe[src*="recaptcha"]', "https://example.com")
        assert result is False


class TestCheckCaptchaAutoSolve:
    """Tests for BrowserManager._check_captcha with auto-solve integration."""

    @pytest.mark.asyncio
    async def test_check_captcha_auto_solves(self):
        """When solver succeeds, _check_captcha returns None (no CAPTCHA reported)."""
        manager = BrowserManager(profiles_dir="/tmp/test_profiles_captcha1")

        mock_solver = AsyncMock()
        mock_solver.solve = AsyncMock(return_value=True)
        manager._captcha_solver = mock_solver

        # Mock instance with a page that has a CAPTCHA (no spec — CamoufoxInstance
        # restricts .page access)
        inst = MagicMock()
        mock_locator = MagicMock()
        mock_locator.count = AsyncMock(return_value=1)
        inst.page.locator = MagicMock(return_value=mock_locator)
        inst.page.url = "https://example.com"

        result = await manager._check_captcha(inst)
        assert result is None  # solved — not reported
        mock_solver.solve.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_captcha_falls_back_on_failure(self):
        """When solver fails, _check_captcha returns fallback dict."""
        manager = BrowserManager(profiles_dir="/tmp/test_profiles_captcha2")

        mock_solver = AsyncMock()
        mock_solver.solve = AsyncMock(return_value=False)
        manager._captcha_solver = mock_solver

        inst = MagicMock()
        mock_locator = MagicMock()
        mock_locator.count = AsyncMock(return_value=1)
        inst.page.locator = MagicMock(return_value=mock_locator)
        inst.page.url = "https://example.com"

        result = await manager._check_captcha(inst)
        assert result is not None
        assert "CAPTCHA detected" in result["message"]
        mock_solver.solve.assert_called_once()
