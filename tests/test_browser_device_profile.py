"""Tests for §19.3 device-profile emulation (Phase 10 §21).

Covers:
  * ``get_device_profile`` selector — known names, default, fallback on unknown.
  * ``build_launch_options`` integration — UA + viewport propagated for
    mobile profiles, default profile preserves the existing behavior.
  * Per-agent override precedence over operator-wide setting.
  * ``build_mobile_init_script`` returns content for mobile profiles only.
"""

from __future__ import annotations

import logging
import os
from unittest import mock

import pytest

from src.browser import flags
from src.browser.stealth import (
    _DESKTOP_WINDOWS_PROFILE,
    _DEVICE_PROFILES,
    _MOBILE_ANDROID_PROFILE,
    _MOBILE_IOS_PROFILE,
    DEFAULT_DEVICE_PROFILE,
    build_launch_options,
    build_mobile_init_script,
    get_device_profile,
)

# ── get_device_profile ──────────────────────────────────────────────────────


class TestGetDeviceProfile:
    def test_default_returns_desktop_windows(self):
        assert get_device_profile() is _DESKTOP_WINDOWS_PROFILE

    def test_none_returns_default(self):
        assert get_device_profile(None) is _DESKTOP_WINDOWS_PROFILE

    def test_empty_string_returns_default(self):
        assert get_device_profile("") is _DESKTOP_WINDOWS_PROFILE

    def test_desktop_windows(self):
        prof = get_device_profile("desktop-windows")
        assert prof is _DESKTOP_WINDOWS_PROFILE
        assert prof["is_mobile"] is False
        assert prof["has_touch"] is False

    def test_desktop_macos(self):
        prof = get_device_profile("desktop-macos")
        assert prof["is_mobile"] is False
        assert prof["platform_navigator"] == "MacIntel"
        assert prof["camoufox_os"] == "macos"

    def test_mobile_ios_documented_shape(self):
        prof = get_device_profile("mobile-ios")
        # UA matches the published Mobile Safari 17.5 / iOS 17.5 string.
        assert prof["user_agent"] == (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 "
            "Mobile/15E148 Safari/604.1"
        )
        # iPhone 14 Pro logical viewport at 3.0 DPR.
        assert prof["viewport"] == {"width": 393, "height": 852}
        assert prof["device_scale_factor"] == 3.0
        assert prof["is_mobile"] is True
        assert prof["has_touch"] is True
        assert prof["max_touch_points"] == 5
        assert prof["platform_navigator"] == "iPhone"
        assert prof["user_agent_data_mobile"] is True

    def test_mobile_android_documented_shape(self):
        prof = get_device_profile("mobile-android")
        # Chrome 124 / Android 14 / Pixel 8 UA.
        assert prof["user_agent"] == (
            "Mozilla/5.0 (Linux; Android 14; Pixel 8) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 "
            "Mobile Safari/537.36"
        )
        # Pixel 8 logical viewport at 2.625 DPR.
        assert prof["viewport"] == {"width": 412, "height": 915}
        assert prof["device_scale_factor"] == 2.625
        assert prof["is_mobile"] is True
        assert prof["has_touch"] is True
        assert prof["platform_navigator"] == "Linux armv8l"
        assert prof["user_agent_data_mobile"] is True

    def test_unknown_falls_back_to_default_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="browser.stealth"):
            prof = get_device_profile("unknown-profile")
        assert prof is _DESKTOP_WINDOWS_PROFILE
        # Warning was emitted naming the bad value
        assert any(
            "Unknown BROWSER_DEVICE_PROFILE" in rec.message
            and "unknown-profile" in rec.message
            for rec in caplog.records
        )

    def test_dispatch_table_has_expected_keys(self):
        # Lock the public surface so a typo in a future rename triggers
        # a test failure rather than silently shipping a renamed profile.
        assert set(_DEVICE_PROFILES.keys()) == {
            "desktop-windows", "desktop-macos", "mobile-ios", "mobile-android",
        }
        assert DEFAULT_DEVICE_PROFILE == "desktop-windows"


# ── build_launch_options propagation ───────────────────────────────────────


class TestBuildLaunchOptionsProfile:
    def test_default_profile_preserves_existing_shape(self):
        # No device_profile passed → behaves exactly as before
        # (Camoufox-driven UA, per-agent resolution pool).
        with mock.patch.dict(os.environ, {}, clear=True):
            opts = build_launch_options("agent-default", "/tmp/p")
        # No pinned UA in config — Camoufox supplies its own.
        assert "navigator.userAgent" not in opts.get("config", {})
        # Window comes from the resolution pool (one of the 6 slots).
        from src.browser.stealth import _RESOLUTION_POOL
        valid = {res for res, _ in _RESOLUTION_POOL}
        assert opts["window"] in valid
        # OS hint is "windows" (default).
        assert opts["os"] == "windows"

    def test_explicit_desktop_windows_matches_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            opts_default = build_launch_options("agent-x", "/tmp/p")
            opts_explicit = build_launch_options(
                "agent-x", "/tmp/p", device_profile="desktop-windows",
            )
        # Same agent_id → same resolution; same OS hint.
        assert opts_default["window"] == opts_explicit["window"]
        assert opts_default["os"] == opts_explicit["os"]

    def test_mobile_ios_sets_ua_viewport_os(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            opts = build_launch_options(
                "agent-iphone", "/tmp/p",
                device_profile="mobile-ios",
            )
        # UA is the iOS profile UA.
        cfg = opts.get("config", {})
        assert cfg.get("navigator.userAgent", "").startswith(
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X)"
        )
        # Viewport pinned to iPhone 14 Pro.
        assert opts["window"] == (393, 852)
        # camoufox_os bound to "macos" for the iOS-Safari plumbing path.
        assert opts["os"] == "macos"
        # i_know_what_im_doing flag is set whenever a UA is pinned.
        assert opts.get("i_know_what_im_doing") is True

    def test_mobile_android_sets_ua_viewport_os(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            opts = build_launch_options(
                "agent-pixel", "/tmp/p",
                device_profile="mobile-android",
            )
        cfg = opts.get("config", {})
        assert cfg.get("navigator.userAgent", "").startswith(
            "Mozilla/5.0 (Linux; Android 14; Pixel 8)"
        )
        assert opts["window"] == (412, 915)
        # mobile-android profile pins Linux for the OS plumbing.
        assert opts["os"] == "linux"

    def test_mobile_profile_overrides_browser_os_env(self):
        # Operator set BROWSER_OS=windows; mobile profile must win and use
        # its own ``camoufox_os`` instead.
        with mock.patch.dict(os.environ, {"BROWSER_OS": "windows"}):
            opts = build_launch_options(
                "agent-x", "/tmp/p", device_profile="mobile-ios",
            )
        assert opts["os"] == "macos"

    def test_unknown_profile_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            opts_unknown = build_launch_options(
                "agent-x", "/tmp/p", device_profile="bogus-profile",
            )
            opts_default = build_launch_options("agent-x", "/tmp/p")
        # Unknown name → fallback to desktop-windows shape.
        assert opts_unknown["os"] == opts_default["os"]
        assert opts_unknown["window"] == opts_default["window"]

    def test_browser_ua_version_ignored_when_mobile_profile_pins_ua(self):
        # On the default profile, BROWSER_UA_VERSION normally rewrites
        # the UA. With a mobile profile that pins its own UA, the env
        # var must NOT override the profile's UA.
        with mock.patch.dict(os.environ, {"BROWSER_UA_VERSION": "200.0"}):
            opts = build_launch_options(
                "agent-x", "/tmp/p", device_profile="mobile-ios",
            )
        ua = opts["config"]["navigator.userAgent"]
        assert "iPhone" in ua
        assert "Firefox/200.0" not in ua


# ── Init script ────────────────────────────────────────────────────────────


class TestMobileInitScript:
    def test_desktop_profiles_return_none(self):
        for name in ("desktop-windows", "desktop-macos"):
            prof = get_device_profile(name)
            assert build_mobile_init_script(prof) is None, name

    def test_mobile_ios_emits_script_without_user_agent_data(self):
        # iOS Safari does NOT expose userAgentData — script must not
        # define it.
        prof = get_device_profile("mobile-ios")
        script = build_mobile_init_script(prof)
        assert script is not None
        assert "maxTouchPoints" in script
        # Touch points pinned to 5
        assert "5" in script
        # Platform string injected
        assert "iPhone" in script
        # The userAgentData branch is gated on a flag — check the literal
        # gating value reflects "false" for iOS.
        assert "__EMIT_USER_AGENT_DATA__" not in script  # placeholders replaced
        # Search for the literal "if (false)" pattern that disables the
        # userAgentData branch (templated value collapsed to "false").
        assert "if (false)" in script

    def test_mobile_android_emits_script_with_user_agent_data(self):
        prof = get_device_profile("mobile-android")
        script = build_mobile_init_script(prof)
        assert script is not None
        assert "Linux armv8l" in script
        assert "if (true)" in script  # userAgentData branch enabled
        # mobile flag is True
        assert "mobile: true" in script


# ── Per-agent override precedence ──────────────────────────────────────────


class TestPerAgentOverride:
    @pytest.fixture(autouse=True)
    def _isolate(self):
        flags._agent_overrides.clear()
        flags.reload_operator_settings()
        yield
        flags._agent_overrides.clear()
        flags.reload_operator_settings()

    def test_agent_override_beats_env(self):
        # Operator sets desktop-windows via env; one specific agent
        # gets mobile-ios via per-agent override.
        with mock.patch.dict(os.environ, {
            "BROWSER_DEVICE_PROFILE": "desktop-windows",
        }):
            flags.set_agent_override(
                "agent-mobile-target", "BROWSER_DEVICE_PROFILE", "mobile-ios",
            )
            # Targeted agent sees the override
            assert flags.get_str(
                "BROWSER_DEVICE_PROFILE",
                DEFAULT_DEVICE_PROFILE,
                agent_id="agent-mobile-target",
            ) == "mobile-ios"
            # Other agents see the operator-wide value
            assert flags.get_str(
                "BROWSER_DEVICE_PROFILE",
                DEFAULT_DEVICE_PROFILE,
                agent_id="agent-other",
            ) == "desktop-windows"

    def test_no_override_no_env_returns_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert flags.get_str(
                "BROWSER_DEVICE_PROFILE",
                DEFAULT_DEVICE_PROFILE,
                agent_id="agent-any",
            ) == DEFAULT_DEVICE_PROFILE

    def test_clear_override_falls_through_to_env(self):
        with mock.patch.dict(os.environ, {
            "BROWSER_DEVICE_PROFILE": "mobile-android",
        }):
            flags.set_agent_override(
                "agent-x", "BROWSER_DEVICE_PROFILE", "mobile-ios",
            )
            assert flags.get_str(
                "BROWSER_DEVICE_PROFILE", DEFAULT_DEVICE_PROFILE,
                agent_id="agent-x",
            ) == "mobile-ios"
            # Clear → back to env layer
            flags.set_agent_override(
                "agent-x", "BROWSER_DEVICE_PROFILE", None,
            )
            assert flags.get_str(
                "BROWSER_DEVICE_PROFILE", DEFAULT_DEVICE_PROFILE,
                agent_id="agent-x",
            ) == "mobile-android"


# ── Integration: profiles + flag → build_launch_options end-to-end ─────────


class TestEndToEndProfileWiring:
    @pytest.fixture(autouse=True)
    def _isolate(self):
        flags._agent_overrides.clear()
        flags.reload_operator_settings()
        yield
        flags._agent_overrides.clear()
        flags.reload_operator_settings()

    def test_flag_value_drives_build_launch_options_shape(self):
        # Simulate: operator sets BROWSER_DEVICE_PROFILE=mobile-android,
        # build_launch_options is called with that name → mobile shape.
        with mock.patch.dict(os.environ, {
            "BROWSER_DEVICE_PROFILE": "mobile-android",
        }):
            chosen = flags.get_str(
                "BROWSER_DEVICE_PROFILE", DEFAULT_DEVICE_PROFILE,
                agent_id="agent-x",
            )
            opts = build_launch_options(
                "agent-x", "/tmp/p", device_profile=chosen,
            )
        assert opts["window"] == (412, 915)
        assert "Pixel 8" in opts["config"]["navigator.userAgent"]

    def test_per_agent_override_drives_build_launch_options(self):
        # Operator runs desktop-windows; one agent overridden to mobile-ios.
        with mock.patch.dict(os.environ, {
            "BROWSER_DEVICE_PROFILE": "desktop-windows",
        }):
            flags.set_agent_override(
                "agent-mobile", "BROWSER_DEVICE_PROFILE", "mobile-ios",
            )
            chosen_mobile = flags.get_str(
                "BROWSER_DEVICE_PROFILE", DEFAULT_DEVICE_PROFILE,
                agent_id="agent-mobile",
            )
            chosen_other = flags.get_str(
                "BROWSER_DEVICE_PROFILE", DEFAULT_DEVICE_PROFILE,
                agent_id="agent-other",
            )
        assert chosen_mobile == "mobile-ios"
        assert chosen_other == "desktop-windows"

        opts_mobile = build_launch_options(
            "agent-mobile", "/tmp/p", device_profile=chosen_mobile,
        )
        opts_other = build_launch_options(
            "agent-other", "/tmp/p", device_profile=chosen_other,
        )
        # Targeted agent gets iOS UA + viewport
        assert opts_mobile["window"] == (393, 852)
        assert "iPhone" in opts_mobile["config"]["navigator.userAgent"]
        # Other agent stays on the desktop default shape
        assert "config" not in opts_other or "navigator.userAgent" not in opts_other.get("config", {})
        # Other agent's window is from the resolution pool (not iPhone)
        assert opts_other["window"] != (393, 852)


# ── Reference data sanity ──────────────────────────────────────────────────


class TestProfileConsistency:
    def test_mobile_profiles_have_required_fields(self):
        for name in ("mobile-ios", "mobile-android"):
            prof = _DEVICE_PROFILES[name]
            for field in (
                "user_agent", "viewport", "device_scale_factor",
                "is_mobile", "has_touch", "platform_navigator",
                "max_touch_points", "user_agent_data_mobile", "camoufox_os",
            ):
                assert field in prof, f"{name} missing {field}"
            assert prof["is_mobile"] is True
            assert prof["has_touch"] is True
            assert prof["max_touch_points"] >= 1

    def test_desktop_profiles_share_invariants(self):
        for name in ("desktop-windows", "desktop-macos"):
            prof = _DEVICE_PROFILES[name]
            assert prof["is_mobile"] is False
            assert prof["has_touch"] is False
            assert prof["max_touch_points"] == 0
            assert prof["user_agent_data_mobile"] is False

    def test_ios_and_android_profile_constants_are_exported(self):
        # The module-level constants are part of the public surface
        # for tests + import paths in service.py.
        assert _MOBILE_IOS_PROFILE is _DEVICE_PROFILES["mobile-ios"]
        assert _MOBILE_ANDROID_PROFILE is _DEVICE_PROFILES["mobile-android"]
