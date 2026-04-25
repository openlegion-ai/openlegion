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
import random
from urllib.parse import urlparse

from src.shared.utils import setup_logging

logger = setup_logging("browser.stealth")


# ── §6.5 Referrer realism on navigate ─────────────────────────────────────────


# Why a referer pool at all: real users overwhelmingly arrive at a page with
# a non-empty ``document.referrer`` and a ``Referer`` request header. An
# agent that issues every navigate with no referer at all is, statistically,
# the easiest fingerprint to flag at scale — a single property that holds
# across every page load.
#
# The pool models the common arrival shapes:
#
#   * **same-origin** — landed via internal link from same host; populated
#                       at runtime per-nav by the picker when the previous
#                       URL's host matches; not a static pool entry
#   * **direct**   — empty referer; typed URL, bookmark, or in-app deep
#                    link; the right shape for first-party logged-in
#                    surfaces (Gmail, GitHub dashboards) where a
#                    search-engine referer would itself be suspicious
#   * **social**   — Twitter/Facebook click-through; targeted at host-
#                    specific destinations where a social referer is
#                    plausible (don't fabricate ``t.co`` for arbitrary
#                    sites — would itself be a tell)
#   * **search**   — "user found us via Google/Bing/DDG"; safe default
#                    for any site that tolerates organic-search arrivals
#
# Per the plan: picked per-nav with small randomness, stored per-agent
# rolling-5 to avoid immediate repeats. The rolling history lives on
# ``CamoufoxInstance`` so it survives across navigate calls but resets on
# browser restart (matches a real session boundary).

# Search-engine referrers — by far the most common organic-arrival shape.
# Google dominates global desktop traffic; Bing + DDG are the next plausible
# non-Google shapes that wouldn't themselves look like a spoof.
_SEARCH_REFERERS: tuple[str, ...] = (
    "https://www.google.com/",
    "https://www.bing.com/",
    "https://duckduckgo.com/",
)

# Social referers, keyed by destination hostname. ``t.co`` is Twitter's
# own click-tracking shim; every real Twitter outbound link travels
# through it.
_SOCIAL_REFERERS: dict[str, tuple[str, ...]] = {
    # Pool of >1 entry per destination so the rolling-5 history can
    # rotate without creating an obvious "every 6th visit is t.co"
    # pattern. Path-bearing entries (``twitter.com/home``) are
    # intentional — they mimic "user clicked a tweet from another
    # tweet/explore tab" arrivals. Real Referer headers carry the
    # full path of the source page, so a path-bearing referer is the
    # closer-to-real shape; the bare-origin entries (``t.co/``) cover
    # the click-tracking shim case. Picker emits these unchanged via
    # ``rng.choice`` — no path-stripping needed.
    "twitter.com": (
        "https://t.co/",
        "https://twitter.com/home",
        "https://twitter.com/explore",
    ),
    "x.com": (
        "https://t.co/",
        "https://x.com/home",
        "https://x.com/explore",
    ),
}

# Hosts where real users typically arrive via direct navigation (typed
# URL, bookmark, app deep-link). A search-engine referer would itself
# be suspicious here — nobody Googles "gmail.com" to check their email.
#
# Includes OAuth identity providers: an OAuth flow lands on
# ``accounts.google.com`` mid-redirect carrying the consumer-app's
# Referer; a fabricated Google referer there breaks detection at
# the IDP layer. Better to send no referer (matches the bookmark /
# typed-URL shape) than a fabricated one.
_DIRECT_NAV_HOSTS: frozenset[str] = frozenset({
    # Webmail / messaging
    "mail.google.com",
    "gmail.com",
    "outlook.office.com",
    "outlook.live.com",
    # Source / collab
    "github.com",
    "gist.github.com",
    "linear.app",
    "notion.so",
    "www.notion.so",
    # Real-time chat
    "app.slack.com",
    # Google productivity
    "calendar.google.com",
    "drive.google.com",
    "docs.google.com",
    "sheets.google.com",
    # OAuth identity providers — preserve the consumer-app referer or
    # send empty; never fabricate. (GitHub OAuth uses ``github.com``,
    # already covered above.)
    "accounts.google.com",
    "login.microsoftonline.com",
    "login.live.com",
    "appleid.apple.com",
    "id.atlassian.com",
})

# Probability of using a social referer when one is registered for the
# target host. Real users mix social inbound with direct/search arrivals;
# always-social would itself be a pattern break.
_SOCIAL_REFERER_PROB: float = 0.30


def validate_referer(referer: str) -> str:
    """Sanitize a caller-supplied ``referer`` string.

    Returns the cleaned value (empty string ⇒ "no referer"). Raises
    ``ValueError`` on a value that isn't safe to forward to Playwright.

    Allowed shapes:
      * ``""`` — explicit no-referer, equivalent to a typed-URL arrival
      * Whitespace-only strings — normalized to ``""``
      * ``http://...`` or ``https://...`` with a hostname

    Rejected (raises ``ValueError``):
      * Pseudo-schemes (``javascript:``, ``data:``, ``file:``, ``about:``)
      * URLs without a hostname (``http:///path``)
      * Non-string types

    The agent skill is LLM-callable, so a malformed value can reach
    here from untrusted-by-default agent output. Playwright doesn't
    validate ``Page.goto(referer=...)`` strictly enough for our threat
    model — defense in depth.
    """
    if not isinstance(referer, str):
        raise ValueError(f"referer must be str, got {type(referer).__name__}")
    cleaned = referer.strip()
    if not cleaned:
        return ""
    try:
        parsed = urlparse(cleaned)
    except Exception as e:
        raise ValueError(f"referer is not a parseable URL: {e}") from e
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(
            f"referer must be http:// or https://, got scheme {parsed.scheme!r}",
        )
    if not parsed.hostname:
        raise ValueError("referer URL has no hostname")
    return cleaned


def pick_referer(
    target_url: str,
    *,
    previous_url: str = "",
    recent_referers: tuple[str, ...] = (),
    rng: random.Random | None = None,
) -> str:
    """Pick a plausible referer for navigating to ``target_url``.

    Returns an empty string for the "direct navigation" case. Caller passes
    the result straight to ``Page.goto(referer=...)`` which sets BOTH the
    network ``Referer`` header and the JS-visible ``document.referrer`` —
    keeping them consistent (a mismatch would itself be a detection signal).

    Decision order:

    1. If ``previous_url`` is a real page on the target host, return its
       origin as a same-origin referer. Real users follow internal links;
       same-origin referer is the commonest case once you're inside a site.
    2. If the target host is in :data:`_DIRECT_NAV_HOSTS`, return empty.
       These are surfaces where a search-engine referer would itself be
       suspicious.
    3. If the target host has a registered social pool, do a weighted
       coin-flip (:data:`_SOCIAL_REFERER_PROB`) between social and search.
    4. Default: pick a search-engine referer.

    ``recent_referers`` is the per-agent rolling history (most-recent
    last). The picker avoids returning a value already in that list,
    falling back to a different slot in the same category. This breaks
    the obvious "every nav has the same Google referer" pattern at
    fleet scale.
    """
    rng = rng or random
    try:
        parsed = urlparse(target_url)
    except Exception:
        return ""
    host = (parsed.hostname or "").lower()
    if not host:
        return ""

    # 1. Same-origin: previous nav was on the same host
    if previous_url:
        try:
            prev = urlparse(previous_url)
            if (prev.hostname and prev.scheme
                    and prev.hostname.lower() == host):
                # Real-user-shape: landed via internal link from "/".
                # Rebuild the origin from ``hostname`` + ``port`` rather
                # than using ``netloc`` directly: ``netloc`` includes
                # userinfo (``user:pass@host``), and emitting that as
                # a Referer would both leak credentials AND look like
                # a bot. Including the non-default port is intentional
                # — browsers treat ``host:8443`` and ``host`` as
                # different origins, so a mismatch on emit is itself
                # a fingerprint bug.
                authority = prev.hostname.lower()
                if prev.port is not None:
                    authority = f"{authority}:{prev.port}"
                return f"{prev.scheme}://{authority}/"
        except Exception:
            pass

    # 2. Direct-nav surface
    if host in _DIRECT_NAV_HOSTS:
        return ""

    # 3. Social — only when destination has one registered AND only some
    # of the time so we don't make every X visit look like it came
    # from t.co.
    social = _SOCIAL_REFERERS.get(host)
    if social and rng.random() < _SOCIAL_REFERER_PROB:
        unseen = [r for r in social if r not in recent_referers]
        if unseen:
            return rng.choice(unseen)

    # 4. Default — search referer, avoid immediate repeats from history
    candidates = [r for r in _SEARCH_REFERERS if r not in recent_referers]
    if not candidates:
        # Rolling history covered every option — fall back to the full
        # set rather than return empty (which would itself be a
        # notable pattern break).
        candidates = list(_SEARCH_REFERERS)
    return rng.choice(candidates)


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
