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

import hashlib
import os

from src.shared.utils import setup_logging

logger = setup_logging("browser.stealth")

# ── Per-agent resolution pool (§6.1) ──────────────────────────────────────────


# Distribution approximates Windows-desktop market share on the most common
# aspect ratios (16:9 + 16:10). Numbers sourced from StatCounter worldwide
# desktop 2025 Q1 aggregated into the relevant bins. Each agent picks one
# deterministically from its ``agent_id`` so the same agent reports the same
# resolution across browser restarts, profile wipes, and container rebuilds.
#
# Why a pool at all: a fleet where every agent reports 1920×1080 is itself a
# cross-agent correlation signal at the detection layer — three "different"
# accounts with identical screen / DPR / viewport shape will correlate in
# fingerprint storage. Spreading across realistic sizes masks the cluster
# without straying into rare/fingerprintable outliers (4K, unusual ratios).
#
# Weights sum to 1.0. Keep ordered by descending weight so the common path
# bisects fewer buckets.
_RESOLUTION_POOL: tuple[tuple[tuple[int, int], float], ...] = (
    ((1920, 1080), 0.34),
    ((1366, 768), 0.22),
    ((1536, 864), 0.14),
    ((1440, 900), 0.12),
    ((1280, 720), 0.10),
    ((1680, 1050), 0.08),
)


def pick_resolution(agent_id: str) -> tuple[int, int]:
    """Return the resolution this agent is assigned to report.

    Deterministic from ``agent_id`` alone: SHA-256 of the id → first 8 bytes
    as an unsigned int → normalized to ``[0, 1)`` → cumulative-weight
    bucket from :data:`_RESOLUTION_POOL`. SHA-256 produces a near-uniform
    distribution, which means at fleet scale the pool weights are honored
    to within sampling error.

    Stable per agent by construction — the plan specifies "survives
    profile wipe", which rules out using profile-local state or the
    browser service's ``boot_id``. Using ``agent_id`` alone also means
    operators can predict the resolution an agent will report when
    auditing fleet diversity.
    """
    digest = hashlib.sha256(agent_id.encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / (1 << 64)
    cumulative = 0.0
    for resolution, weight in _RESOLUTION_POOL:
        cumulative += weight
        if u < cumulative:
            return resolution
    # Floating-point slack: sum-of-weights can be microscopically < 1.0;
    # fall back to the highest-weight option so nothing silently picks a
    # default outside the pool.
    return _RESOLUTION_POOL[0][0]


# ── §6.6 NetworkInformation per-agent fingerprint ─────────────────────────────


def pick_network_info(agent_id: str) -> dict:
    """Stable per-agent ``navigator.connection`` values.

    Real desktop users on broadband / 4G / 5G report
    ``effectiveType="4g"`` overwhelmingly; the variability is in
    ``downlink`` (Mbps) and ``rtt`` (ms). Datacenter Firefox often
    leaves all three undefined, which is itself a signal.

    Picks per-agent from plausible bands:
      - downlink: 5–20 Mbps (covers home broadband, mobile 4G good signal)
      - rtt: 20–120 ms (covers wired through mobile)
      - saveData: always False (rare on desktop; True would itself be a flag)

    Deterministic from ``agent_id`` so the same agent reports the same
    network shape across browser restarts. SHA-256 splits into two
    independent 4-byte words for downlink and rtt — using one byte each
    would give a coarse 256-bucket distribution and visible quantisation
    on fleet-scale analysis.
    """
    digest = hashlib.sha256(f"netinfo:{agent_id}".encode("utf-8")).digest()
    dl_unit = int.from_bytes(digest[:4], "big") / (1 << 32)   # [0, 1)
    rtt_unit = int.from_bytes(digest[4:8], "big") / (1 << 32)  # [0, 1)
    return {
        "effectiveType": "4g",
        "downlink": round(5.0 + dl_unit * 15.0, 1),
        "rtt": int(20 + rtt_unit * 100),
        "saveData": False,
    }


# ── §6.4 Client-Hints / Firefox-UA guard ──────────────────────────────────────


def _assert_firefox_ua(ua: str) -> None:
    """Refuse a non-Firefox UA string.

    Detection at the Client-Hints layer (Sec-CH-UA-* headers) is
    Chromium-only — real Firefox doesn't send those headers. If this
    project ever switches to a Chromium base (or someone hand-overrides
    ``BROWSER_UA_VERSION`` with a Chrome string), the resulting browser
    would advertise itself as Chrome via UA but NOT send the matching
    Sec-CH-UA-* hints, which is itself a strong inconsistency signal.

    This assertion is a tripwire: if it fires, whoever changed the UA
    has to also wire up Sec-CH-UA / Sec-CH-UA-Mobile / Sec-CH-UA-Platform
    overrides before removing the guard. Camoufox doesn't currently
    expose Client-Hints injection, so flipping this off without that
    work means shipping a detectable agent.
    """
    if not ua:
        return
    if "Firefox/" not in ua:
        raise ValueError(
            "Refusing non-Firefox UA — Client-Hints would leak the "
            "inconsistency. Wire up Sec-CH-UA-* overrides before "
            f"shipping this UA: {ua!r}",
        )


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

    # §6.1: pick a per-agent resolution from the pool. The KasmVNC display
    # itself stays 1920×1080 (shared by all agents on this container); the
    # chosen resolution determines Firefox's window size + the reported
    # ``window.screen`` dimensions. Undersized windows show dead space
    # around them on VNC, filled by the dark wallpaper set in __main__.py.
    #
    # This also requires the Openbox config NOT to force-maximize the main
    # browser window — the maximize rule was removed from Dockerfile.browser
    # alongside this feature. Otherwise the window would end up at full
    # display size while the fingerprint reported 1280×720, a detectable
    # mismatch.
    width, height = pick_resolution(agent_id)
    logger.debug("Agent '%s' resolution: %dx%d", agent_id, width, height)

    options: dict = {
        "headless": False,
        "humanize": True,        # Camoufox mouse-curves + micro-delays
        "os": os_hint,
        "locale": locale,        # navigator.language / Accept-Language header
        "window": (width, height),
        # block_webrtc: Camoufox native toggle — prevents Docker container IP
        # from leaking via ICE candidates.  More reliable than manual prefs.
        "block_webrtc": True,
    }

    # NOTE: "timezone" is NOT a valid Camoufox parameter (Playwright uses
    # timezone_id, not timezone).  Passing it causes a TypeError on browser
    # startup.  Locale implicitly determines timezone via GeoIP or BrowserForge.

    # Lock the BrowserForge screen fingerprint to the chosen resolution so
    # window.innerWidth ≤ window.screen.width holds (a mismatch is itself a
    # detection signal).
    try:
        from browserforge.fingerprints import Screen
        options["screen"] = Screen(max_width=width, max_height=height)
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

    # ── §6.6 NetworkInformation spoof ────────────────────────────────────────
    # Camoufox's ``config`` dict supports dotted keys for navigator.* override.
    # Set per-agent stable values so ``navigator.connection.{effectiveType,
    # downlink, rtt}`` look like a desktop on broadband. Without this Firefox
    # leaves these undefined on Linux containers, which detection scripts use
    # as a desktop-vs-bot tell.
    netinfo = pick_network_info(agent_id)
    options["config"] = {
        "navigator.connection.effectiveType": netinfo["effectiveType"],
        "navigator.connection.downlink": netinfo["downlink"],
        "navigator.connection.rtt": netinfo["rtt"],
        "navigator.connection.saveData": netinfo["saveData"],
    }
    # Camoufox requires this acknowledgement before applying ``config``
    # overrides. We're past the early-return path so it's safe to set
    # unconditionally now.
    options["i_know_what_im_doing"] = True

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
            # §6.4 tripwire — the UA we're about to ship MUST be Firefox.
            # If a future change introduces a Chromium UA, this raises so
            # the developer has to wire Sec-CH-UA-* overrides first.
            _assert_firefox_ua(ua)
            options["config"]["navigator.userAgent"] = ua
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

        # ── First-run / welcome / default-browser prompts ────────────────────
        # Belt-and-suspenders against the about:welcome tab, default-browser
        # nag, and profile-import wizard. Phase 3 profile schema v2 takes
        # care to NOT delete ``compatibility.ini`` (which would itself
        # trigger first-run UI), but a fresh profile or a Firefox version
        # bump can also cross those code paths. None of these prompts
        # appear on a properly-warmed profile, but they all block
        # automation when they do — so suppress them at the pref layer.
        "browser.shell.checkDefaultBrowser": False,
        "browser.aboutwelcome.enabled": False,
        "browser.startup.homepage_override.mstone": "ignore",
        "startup.homepage_welcome_url": "",
        "startup.homepage_welcome_url.additional": "",
        "browser.disableResetPrompt": True,
        "browser.tabs.warnOnClose": False,
        "browser.tabs.warnOnCloseOtherTabs": False,
    }
