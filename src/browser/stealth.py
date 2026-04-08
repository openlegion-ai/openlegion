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
  - Resolution is locked to 1920×1080 to match the KasmVNC display
  - Locale and timezone are explicit so they match the fingerprint OS
  - WebRTC fully disabled — Docker internal IPs leak via ICE otherwise
  - privacy.resistFingerprinting OFF — RFP produces detectable sentinel values
  - All background noise (telemetry, prefetch, crash reports) suppressed
"""

from __future__ import annotations

import os

from src.shared.utils import setup_logging

logger = setup_logging("browser.stealth")

# ── Launch options ─────────────────────────────────────────────────────────────


def build_launch_options(agent_id: str, profile_dir: str, proxy: dict | None = None) -> dict:
    """Build Camoufox AsyncNewBrowser kwargs for an agent.

    Args:
        proxy: Camoufox proxy dict (server, username, password) or None.
               Proxy is always resolved per-agent by the mesh host and pushed
               to the browser service — callers must pass it explicitly.

    Environment variables (all optional):
      BROWSER_OS     — "windows" (default) | "macos" | "linux"
      BROWSER_LOCALE — BCP-47 locale tag, e.g. "en-US" (default)
      BROWSER_UA_VERSION — Firefox version to report in User-Agent, e.g. "138.0".
                           If set, overrides the UA string to spoof a newer Firefox.
                           Useful when Camoufox's bundled Firefox is too old for
                           sites that enforce minimum browser versions (e.g. Shopify).
    """

    # ── OS fingerprint ────────────────────────────────────────────────────────
    # Default to Windows: it has ≈70 % desktop market share and produces the
    # least suspicious fingerprint for server-hosted automation.  Linux is an
    # immediate datacenter/bot signal for sites that check navigator.platform.
    os_hint = os.environ.get("BROWSER_OS", "windows").lower()
    if os_hint not in ("windows", "macos", "linux"):
        logger.warning("Invalid BROWSER_OS %r, defaulting to 'windows'", os_hint)
        os_hint = "windows"

    locale = os.environ.get("BROWSER_LOCALE", "en-US")

    # VNC display is always 1920×1080.  We set window= and Screen to match so
    # that window.innerWidth and window.screen.width are consistent — a mismatch
    # (innerWidth > screen.width) is itself a detectable bot signal.  Per-agent
    # resolution variation is not worth breaking the KasmVNC UX.

    options: dict = {
        "headless": False,
        "humanize": True,        # Camoufox mouse-curves + micro-delays
        "os": os_hint,
        "locale": locale,        # navigator.language / Accept-Language header
        "window": (1920, 1080),  # fill the KasmVNC display
        # block_webrtc: Camoufox native toggle — prevents Docker container IP
        # from leaking via ICE candidates.  More reliable than manual prefs.
        "block_webrtc": True,
    }

    # NOTE: "timezone" is NOT a valid Camoufox parameter (Playwright uses
    # timezone_id, not timezone).  Passing it causes a TypeError on browser
    # startup.  Locale implicitly determines timezone via GeoIP or BrowserForge.

    # Lock the BrowserForge screen fingerprint to 1920×1080 so it stays
    # consistent with the actual window size.
    try:
        from browserforge.fingerprints import Screen
        options["screen"] = Screen(max_width=1920, max_height=1080)
    except ImportError:
        pass  # browserforge only available in browser container

    # GeoIP: maps egress IP → timezone/locale for the fingerprint.
    # Only enable when a proxy is configured so the resolved location matches
    # the proxy's egress IP, not the Docker container's internal NAT address.
    if proxy:
        options["proxy"] = proxy
        options["geoip"] = True

    # Persistent profile preserves cookies, localStorage, and session tokens.
    options["persistent_context"] = True
    options["user_data_dir"] = profile_dir

    options["firefox_user_prefs"] = _stealth_prefs()

    # ── User-Agent version override ──────────────────────────────────────────
    # Camoufox bundles a specific Firefox build (e.g. 135.0).  Some sites
    # (Shopify, etc.) enforce minimum browser versions and block old Firefox.
    # BROWSER_UA_VERSION overrides the reported version without upgrading the
    # Camoufox binary.
    #
    # Primary mechanism: Camoufox's `config` dict with `navigator.userAgent`.
    # This feeds into Camoufox's fingerprint injection system, which controls
    # both navigator.userAgent (JS) and the User-Agent HTTP header.
    # Fallback: `general.useragent.override` Firefox pref, in case an older
    # Camoufox version doesn't honour the config dict.
    ua_version = os.environ.get("BROWSER_UA_VERSION", "")
    if ua_version:
        ua = _build_ua_string(os_hint, ua_version)
        if ua:
            options["config"] = {"navigator.userAgent": ua}
            options["i_know_what_im_doing"] = True
            options["firefox_user_prefs"]["general.useragent.override"] = ua
            logger.info("UA override: Firefox/%s", ua_version)

    return options


def _build_ua_string(os_hint: str, version: str) -> str | None:
    """Build a Firefox User-Agent string for the given OS and version.

    Returns None and logs a warning if the version format is invalid.
    Accepts versions like "138.0" or "138.0.1".
    """
    version = version.strip()
    parts = version.split(".")
    if len(parts) < 2 or not all(p.isdigit() for p in parts):
        logger.warning(
            "Invalid BROWSER_UA_VERSION %r (expected e.g. '138.0'), ignoring",
            version,
        )
        return None

    _OS_UA_TEMPLATES = {
        "macos": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:{v}) Gecko/20100101 Firefox/{v}",
        "linux": "Mozilla/5.0 (X11; Linux x86_64; rv:{v}) Gecko/20100101 Firefox/{v}",
        "windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{v}) Gecko/20100101 Firefox/{v}",
    }
    template = _OS_UA_TEMPLATES.get(os_hint, _OS_UA_TEMPLATES["windows"])
    return template.format(v=version)


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

        # ── WebRTC: handled by Camoufox block_webrtc=True ────────────────────
        # RTCPeerConnection leaks the Docker container's internal IP via ICE
        # candidates even when behind a proxy.  Camoufox's block_webrtc option
        # is the canonical way to disable WebRTC — it covers all relevant prefs.

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
