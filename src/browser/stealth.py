"""Stealth/fingerprint configuration for Camoufox instances.

Handles Camoufox launch options, proxy configuration, and
BrowserForge fingerprint generation.

Bot-detection threat model
--------------------------
Detection systems (Cloudflare, Akamai Bot Manager, PerimeterX, DataDome,
Kasada, X/Twitter's own stack) fingerprint browsers at multiple layers:

  1. Network / IP layer  — datacenter IPs vs residential; TLS JA3/JA4
  2. Passive JS fingerprint — navigator.*, screen, canvas, WebGL, fonts
  3. WebRTC                — internal IP leak via ICE candidates
  4. Behavioural          — mouse curves, keystroke rhythm, scroll pattern
  5. Protocol             — CDP/puppeteer artefacts in event ordering

Camoufox (patched Firefox + BrowserForge) handles layers 2, 4, 5.
This module ensures the fingerprint we feed Camoufox is internally
consistent and realistic:
  - OS defaults to Windows (≈70 % market share); Linux is a datacenter signal
  - Resolution sampled from the empirical distribution for that OS
  - Locale and timezone are explicit so they match the fingerprint OS
  - WebRTC fully disabled — Docker internal IPs leak via ICE otherwise
  - privacy.resistFingerprinting OFF — RFP produces detectable sentinel values
  - All background noise (telemetry, prefetch, crash reports) suppressed
"""

from __future__ import annotations

import os
import random
from urllib.parse import urlparse

from src.shared.utils import setup_logging

logger = setup_logging("browser.stealth")

# ── Screen resolution tables ──────────────────────────────────────────────────
# Sampled from StatCounter GlobalStats (desktop, 2024).
# Each tuple is (width, height).  Weights ≈ market-share percentage.
_WINDOWS_RESOLUTIONS = [
    (1920, 1080),  # 22 %
    (1366, 768),   # 11 %
    (1536, 864),   #  8 %
    (1440, 900),   #  7 %
    (1280, 720),   #  5 %
    (2560, 1440),  #  5 %
    (1600, 900),   #  4 %
    (1280, 1024),  #  3 %
    (1680, 1050),  #  3 %
]
_WINDOWS_WEIGHTS = [22, 11, 8, 7, 5, 5, 4, 3, 3]

_MACOS_RESOLUTIONS = [
    (1920, 1080),  # 28 %
    (2560, 1600),  # 18 %  MacBook retina (logical)
    (1440, 900),   # 14 %
    (2880, 1800),  # 10 %  MacBook Pro 15" retina
    (1280, 800),   #  8 %
    (2560, 1440),  #  7 %  iMac / Pro Display XDR scaled
]
_MACOS_WEIGHTS = [28, 18, 14, 10, 8, 7]


def _pick_resolution(os_hint: str) -> tuple[int, int]:
    """Pick a resolution weighted by empirical market share for the OS."""
    if os_hint == "macos":
        return random.choices(_MACOS_RESOLUTIONS, weights=_MACOS_WEIGHTS, k=1)[0]
    return random.choices(_WINDOWS_RESOLUTIONS, weights=_WINDOWS_WEIGHTS, k=1)[0]


# ── Proxy ──────────────────────────────────────────────────────────────────────


def get_proxy_config() -> dict | None:
    """Build proxy config from environment variables.

    Reads BROWSER_PROXY_URL, BROWSER_PROXY_USER, BROWSER_PROXY_PASS.
    Returns dict suitable for Camoufox proxy parameter, or None.

    NOTE: Residential proxies are the single most impactful defence
    against IP-reputation-based blocking.  Datacenter IPs are trivially
    flagged by Cloudflare, Akamai, and X regardless of fingerprint quality.
    """
    proxy_url = os.environ.get("BROWSER_PROXY_URL", "")
    if not proxy_url:
        return None
    config: dict = {"server": proxy_url}
    proxy_user = os.environ.get("BROWSER_PROXY_USER", "")
    proxy_pass = os.environ.get("BROWSER_PROXY_PASS", "")
    if proxy_user:
        config["username"] = proxy_user
    if proxy_pass:
        config["password"] = proxy_pass
    parsed = urlparse(proxy_url)
    safe_url = (
        f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
        if parsed.port
        else f"{parsed.scheme}://{parsed.hostname}"
    )
    logger.info("Proxy configured: %s", safe_url)
    return config


# ── Launch options ─────────────────────────────────────────────────────────────


def build_launch_options(agent_id: str, profile_dir: str) -> dict:
    """Build Camoufox AsyncNewBrowser kwargs for an agent.

    Environment variables (all optional):
      BROWSER_OS       — "windows" (default) | "macos" | "linux"
      BROWSER_LOCALE   — BCP-47 locale tag, e.g. "en-US" (default)
      BROWSER_TIMEZONE — IANA timezone, e.g. "America/New_York" (default)
    """
    proxy = get_proxy_config()

    # ── OS fingerprint ────────────────────────────────────────────────────────
    # Default to Windows: it has ≈70 % desktop market share and produces the
    # least suspicious fingerprint for server-hosted automation.  Linux is an
    # immediate datacenter/bot signal for sites that check navigator.platform.
    os_hint = os.environ.get("BROWSER_OS", "windows").lower()
    if os_hint not in ("windows", "macos", "linux"):
        logger.warning("Invalid BROWSER_OS %r, defaulting to 'windows'", os_hint)
        os_hint = "windows"

    locale = os.environ.get("BROWSER_LOCALE", "en-US")
    timezone = os.environ.get("BROWSER_TIMEZONE", "America/New_York")

    # Pick a resolution representative of the OS's user population.
    resolution = _pick_resolution(os_hint)
    width, height = resolution

    options: dict = {
        "headless": False,
        "humanize": True,        # Camoufox mouse-curves + micro-delays
        "os": os_hint,
        "locale": locale,        # navigator.language / Accept-Language header
        "window": resolution,
    }

    # BrowserForge screen constraints — keep consistent with our resolution.
    try:
        from browserforge.fingerprints import Screen
        options["screen"] = Screen(max_width=width, max_height=height)
    except ImportError:
        pass  # browserforge only available in browser container

    # GeoIP: map the egress IP → timezone/locale for the fingerprint.
    # Only enable when a proxy is configured; without a proxy the egress IP is
    # a datacenter address and GeoIP would produce a mismatched locale.
    if proxy:
        options["proxy"] = proxy
        options["geoip"] = True

    # Persistent profile preserves cookies, localStorage, and session tokens.
    options["persistent_context"] = True
    options["user_data_dir"] = profile_dir

    options["firefox_user_prefs"] = _stealth_prefs()

    return options


def _stealth_prefs() -> dict:
    """Return Firefox user preferences that minimise bot-detection surface.

    Design rationale for each group is inline.  Correctness over completeness:
    only prefs with a clear detection impact are included.
    """
    return {
        # ── UI cosmetics ──────────────────────────────────────────────────────
        "extensions.activeThemeID": "firefox-compact-dark@mozilla.org",
        "browser.uidensity": 1,
        "browser.tabs.inTitlebar": 1,

        # ── WebRTC: fully disabled ────────────────────────────────────────────
        # RTCPeerConnection leaks the Docker container's internal IP (172.x.x.x
        # or 10.x.x.x) via ICE candidates even when behind a proxy.  This is
        # one of the most reliable automated-browser signals in use today.
        "media.peerconnection.enabled": False,
        "media.peerconnection.turn.disable": True,
        "media.peerconnection.use_document_iceservers": False,
        "media.peerconnection.video.enabled": False,

        # ── Geolocation ───────────────────────────────────────────────────────
        # The geo API would expose incorrect coordinates for a server IP, and
        # the browser permission prompt would block any page that auto-requests it.
        "geo.enabled": False,
        "geo.provider.use_corelocation": False,
        "geo.provider.ms-windows-location": False,

        # ── Fingerprint-resistance: leave RFP OFF ─────────────────────────────
        # privacy.resistFingerprinting (RFP) rounds canvas, screen, and timers
        # to fixed sentinel values.  Detection scripts test for these exact
        # values (e.g. screen.width === 1366 regardless of real screen size)
        # and flag RFP users as bots.  Camoufox injects natural noise instead.
        "privacy.resistFingerprinting": False,
        "privacy.trackingprotection.fingerprinting.enabled": False,

        # ── Cache: keep ON ────────────────────────────────────────────────────
        # Real browsers accumulate HTTP cache.  An empty disk cache on every
        # visit is a headless/automation signal.  Persistent profiles already
        # carry cache across sessions; these prefs ensure the cache is used.
        "browser.cache.disk.enable": True,
        "browser.cache.memory.enable": True,
        "browser.cache.offline.enable": True,

        # ── Notifications and push: silently deny ─────────────────────────────
        # Permission prompts for desktop notifications and Web Push block
        # automation.  Deny them silently so the page never gets a prompt.
        "dom.webnotifications.enabled": False,
        "dom.push.enabled": False,
        "permissions.default.desktop-notification": 2,  # 2 = block

        # ── Speculative / prefetch requests ───────────────────────────────────
        # Prefetch requests fire before any user interaction; their absence or
        # presence at unexpected times can reveal scripted navigation patterns.
        # Disable to make request timing purely interaction-driven.
        "network.prefetch-next": False,
        "network.dns.disablePrefetch": True,
        "network.dns.disablePrefetchFromHTTPS": True,
        "network.predictor.enabled": False,

        # ── Popups: allow ─────────────────────────────────────────────────────
        # OAuth flows and login dialogs frequently open popup windows.
        # Blocking them causes silent auth failures.
        "dom.disable_open_during_load": False,

        # ── Telemetry and update checks ───────────────────────────────────────
        # These fire background XHR/DNS requests that look like bot traffic
        # patterns (non-page-triggered network activity).
        "datareporting.healthreport.uploadEnabled": False,
        "datareporting.policy.dataSubmissionEnabled": False,
        "toolkit.telemetry.enabled": False,
        "toolkit.telemetry.unified": False,
        "app.update.enabled": False,
        "app.update.auto": False,
        "browser.ping-centre.telemetry": False,
        "browser.newtabpage.activity-stream.feeds.telemetry": False,

        # ── Crash reporting ───────────────────────────────────────────────────
        "browser.crashReports.unsubmittedCheck.enabled": False,
        "breakpad.reportURL": "",

        # ── Session restore ───────────────────────────────────────────────────
        # Prevents the "restore session?" prompt on startup after a crash/kill.
        "browser.sessionstore.resume_from_crash": False,

        # ── Extension recommendations ─────────────────────────────────────────
        # Disables recommendation popups that interrupt automation.
        "browser.newtabpage.activity-stream.asrouter.userprefs.cfr.addons": False,
        "browser.newtabpage.activity-stream.asrouter.userprefs.cfr.features": False,
        "extensions.htmlaboutaddons.recommendations.enabled": False,
    }
