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

Device profiles (§19.3 — Phase 10 §21)
--------------------------------------
The default profile is ``desktop-windows`` (the historical behavior). Two
mobile profiles + a desktop-macOS variant are available via the
``BROWSER_DEVICE_PROFILE`` flag:

  * ``desktop-windows`` — Firefox on Windows 10/11 (≈70% market share)
  * ``desktop-macos``   — Firefox on macOS (≈10% market share)
  * ``mobile-ios``      — Mobile Safari on iPhone 14 Pro
  * ``mobile-android``  — Chrome on Pixel 8

Mobile profiles are useful when the target site serves a mobile-friendly
version (or when geographies dominated by mobile traffic make a mobile
fingerprint less suspicious). Tradeoff: some desktop-only forms reject
mobile UAs; some sites serve different content / fewer features to mobile.

Camoufox compatibility caveats: Camoufox is built on Firefox, so the iOS
profile (Mobile Safari UA) and Android profile (Chrome UA) ship a UA-engine
mismatch — the underlying engine is still Gecko/Firefox even when the UA
string says WebKit/Blink. We mitigate at the surface layer (UA string,
viewport, DPR, ``has_touch``, ``is_mobile``, ``navigator.userAgentData``)
but cannot spoof every nested API (e.g. WebGL renderer strings, codec
support quirks, deep CSS feature queries) to fully match the claimed
device. For high-accuracy mobile spoofing, a Chromium-based stack would
be required; this implementation gives operators a usable mobile
fingerprint for sites that gate on the easy-to-check signals.
"""

from __future__ import annotations

import hashlib
import os
import random
from urllib.parse import urlparse

from src.shared.utils import setup_logging

logger = setup_logging("browser.stealth")

# ── §19.3 Device profiles (Phase 10 §21) ─────────────────────────────────────


# Each profile bundles the surface-layer fingerprint signals needed to
# present as a particular device class. Consumers (``build_launch_options``,
# the BrowserContext factory in ``service.py``) read these dicts to populate
# UA, viewport, device-scale-factor, ``is_mobile`` / ``has_touch`` flags,
# and the init-script that overrides ``navigator.userAgentData`` /
# ``navigator.maxTouchPoints``.
#
# Profile values are chosen against the current real-world UA matrix
# (April 2026 snapshot):
#
#   * ``desktop-windows`` — Firefox 138 on Windows 10/11. Matches the
#     existing default; UA is built dynamically from ``BROWSER_UA_VERSION``
#     when set.
#   * ``desktop-macos``   — Firefox 138 on macOS 14 (Sonoma). Mac users
#     skew higher-trust on some risk models (consumer rather than
#     datacenter); useful when targeting US/Western-Europe sites.
#   * ``mobile-ios``      — iOS 17.5 / Safari 17.5 on iPhone 14 Pro.
#     iOS 17.x is the dominant iOS major as of early 2026; iPhone 14 Pro
#     is a high-share device with a 393×852 logical viewport at 3.0 DPR
#     (1179×2556 physical). UA matches Apple's WebKit User-Agent for
#     Mobile Safari on iOS 17.5 (build 15E148, Version/17.5).
#   * ``mobile-android``  — Android 14 / Chrome 124 on Pixel 8. Pixel 8
#     uses a 412×915 logical viewport at 2.625 DPR (1080×2400 physical).
#     Chrome 124 corresponds to Chrome's April 2026 stable channel; Mobile
#     Safari/537.36 build literal matches Chrome's standard mobile UA shape.
#
# When picking new profiles, keep three invariants:
#   1. UA, viewport, DPR, ``platform_navigator``, and ``user_agent_data_mobile``
#      must internally agree (mismatch is itself a fingerprint signal).
#   2. Touch capability + max touch points: real iOS / Android ship 5; real
#      desktops ship 0 (or 10 on touch-screen Windows laptops).
#   3. WebRTC stays disabled across all profiles — the ICE-leak risk is
#      device-independent.
_DESKTOP_WINDOWS_PROFILE: dict = {
    # UA, viewport, and DPR for desktop-windows are NOT pinned in the
    # profile: ``build_launch_options`` lets Camoufox + the existing
    # per-agent ``pick_resolution`` pool drive these for the default
    # profile. Profile metadata still recorded so consumers can branch
    # on shape (e.g. has_touch=False → suppress mobile init script).
    "user_agent": None,  # built from BROWSER_UA_VERSION or Camoufox default
    "viewport": None,    # picked per-agent via pick_resolution
    "device_scale_factor": 1.0,
    "is_mobile": False,
    "has_touch": False,
    "platform_navigator": "Win32",
    "max_touch_points": 0,
    "user_agent_data_mobile": False,
    "camoufox_os": "windows",
}

_DESKTOP_MACOS_PROFILE: dict = {
    # Same shape as Windows but with macOS UA + platform — still allows
    # the per-agent resolution pool to drive viewport/DPR. macOS desktop
    # users overwhelmingly run native (non-Retina-doubled) DPR=2 on the
    # browser side; Camoufox handles this when ``os=macos`` is set.
    "user_agent": None,
    "viewport": None,
    "device_scale_factor": 2.0,
    "is_mobile": False,
    "has_touch": False,
    "platform_navigator": "MacIntel",
    "max_touch_points": 0,
    "user_agent_data_mobile": False,
    "camoufox_os": "macos",
}

_MOBILE_IOS_PROFILE: dict = {
    # iOS 17.5 / Safari 17.5 / iPhone 14 Pro. UA string sourced from
    # Apple's published WebKit User-Agents for iOS 17.5 (released
    # 2024-05-13; still common on the in-field installed base in 2026).
    # Build literal "15E148" + "Version/17.5" + "Mobile/15E148" matches
    # Safari's standard format.
    "user_agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 "
        "Mobile/15E148 Safari/604.1"
    ),
    "viewport": {"width": 393, "height": 852},  # iPhone 14 Pro logical
    "device_scale_factor": 3.0,                  # 1179×2556 physical
    "is_mobile": True,
    "has_touch": True,
    "platform_navigator": "iPhone",
    "max_touch_points": 5,
    "user_agent_data_mobile": True,
    # Camoufox is Firefox-based; the closest OS analogue for fingerprint
    # plumbing (font stack, etc.) is macOS — iOS Safari shares the
    # Apple/Cocoa side but has a distinct mobile-engine path.
    "camoufox_os": "macos",
}

_MOBILE_ANDROID_PROFILE: dict = {
    # Android 14 / Chrome 124 / Pixel 8. UA matches Chrome's April 2026
    # stable channel mobile UA shape ("AppleWebKit/537.36 ... Chrome/124
    # ... Mobile Safari/537.36" — the WebKit literal is historical, kept
    # for compatibility with sites that pattern-match on it).
    "user_agent": (
        "Mozilla/5.0 (Linux; Android 14; Pixel 8) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 "
        "Mobile Safari/537.36"
    ),
    "viewport": {"width": 412, "height": 915},   # Pixel 8 logical
    "device_scale_factor": 2.625,                # 1080×2400 physical
    "is_mobile": True,
    "has_touch": True,
    "platform_navigator": "Linux armv8l",
    "max_touch_points": 5,
    "user_agent_data_mobile": True,
    "camoufox_os": "linux",
}

# Dispatch table consumed by :func:`get_device_profile`. Keep
# ``desktop-windows`` first so iteration / tooling sees the default at
# the head of the list.
_DEVICE_PROFILES: dict[str, dict] = {
    "desktop-windows": _DESKTOP_WINDOWS_PROFILE,
    "desktop-macos": _DESKTOP_MACOS_PROFILE,
    "mobile-ios": _MOBILE_IOS_PROFILE,
    "mobile-android": _MOBILE_ANDROID_PROFILE,
}

DEFAULT_DEVICE_PROFILE: str = "desktop-windows"


def get_device_profile(name: str | None = None) -> dict:
    """Return the device profile dict for ``name``.

    Resolution rules:
      * ``None`` or empty string → default (``desktop-windows``).
      * Known name → that profile.
      * Unknown name → log a warning and fall back to default.
      * Engine-mismatch profiles (``mobile-android``) → require explicit
        ``OPENLEGION_BROWSER_ALLOW_ENGINE_MISMATCH=1`` opt-in. The Android
        profile ships a Chrome UA + ``platform_navigator: "Linux armv8l"``
        on top of a Gecko/Firefox engine (Camoufox). Detection products
        catch this via ``WebGL2RenderingContext.prototype.getParameter``,
        ``RTCRtpSender.getCapabilities`` codec ordering, and
        ``navigator.userActivation`` shape (Chromium-only). Without the
        flag this profile is more harmful than helpful — fall back to
        the default and surface a loud warning.

    The returned dict is the live module-level constant — do not mutate.
    Callers that want to layer per-agent overrides on top should
    ``copy()`` the result first.
    """
    if not name:
        return _DEVICE_PROFILES[DEFAULT_DEVICE_PROFILE]
    profile = _DEVICE_PROFILES.get(name)
    if profile is None:
        logger.warning(
            "Unknown BROWSER_DEVICE_PROFILE %r; allowed values are %s. "
            "Falling back to %r.",
            name, sorted(_DEVICE_PROFILES.keys()), DEFAULT_DEVICE_PROFILE,
        )
        return _DEVICE_PROFILES[DEFAULT_DEVICE_PROFILE]
    if name in _ENGINE_MISMATCH_PROFILES and not _engine_mismatch_allowed():
        logger.warning(
            "BROWSER_DEVICE_PROFILE=%r ships a Chrome UA on a Firefox "
            "engine — strong bot signal on any FP-aware site. Set "
            "OPENLEGION_BROWSER_ALLOW_ENGINE_MISMATCH=1 to opt in. "
            "Falling back to %r.",
            name, DEFAULT_DEVICE_PROFILE,
        )
        return _DEVICE_PROFILES[DEFAULT_DEVICE_PROFILE]
    return profile


# Profiles whose UA + JS surface declare a different engine than the
# underlying Camoufox (Gecko/Firefox). Each one is a strong tell on any
# FP-aware site and is opt-in only.
_ENGINE_MISMATCH_PROFILES = frozenset({"mobile-android"})


def _engine_mismatch_allowed() -> bool:
    val = os.environ.get("OPENLEGION_BROWSER_ALLOW_ENGINE_MISMATCH", "")
    return val.strip().lower() in {"1", "true", "yes", "on"}


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


# ── Per-platform pre-navigate timing posture ─────────────────────────────────


# Why this exists: LinkedIn, X/Twitter, and Meta all stack heavy in-house
# behavioral risk-scoring on top of standard CAPTCHA/anti-bot. A signal
# they all key on is "request fired ~0ms after the previous action" —
# the agent typing a URL, hitting Enter, and pulling a page in 50ms is
# itself a fingerprint, separate from JS / TLS / IP layers. Real humans
# pause to focus the address bar, glance at the URL, and only then hit
# Enter. The sub-second-arrival cluster is a reliable bot signal at
# fleet scale.
#
# This module adds a Gaussian pre-nav dwell that fires ONLY when the
# target host matches a known high-protection platform. The delay is
# applied just before ``page.goto(...)`` so it shifts the request-time
# distribution toward the human population without slowing down nav
# to less-defended sites.
#
# Tuning rationale per platform:
# * LinkedIn / X / Twitter — moderate (2-3s μ). PerimeterX/HUMAN on
#   LinkedIn and X's in-house "Bird" risk model both score arrival
#   timing; 2-3s of dwell falls inside the human bell curve without
#   feeling noticeable to operators watching VNC.
# * Meta (facebook.com / instagram.com) — heavier (4s μ). Meta's
#   behavioral checkpoint logic is the most aggressive of the three;
#   real users on Meta also page-dwell longer (feed-scroll culture),
#   so a higher μ is plausible AND less risky.
#
# Operators override globally by unsetting ``BROWSER_PLATFORM_TIMING_ENABLED``
# (default true) or per-agent via the same flag. Per-platform tuning is
# operator-driven via env: ``BROWSER_PLATFORM_TIMING_<HOSTKEY>_MU_S`` etc.
# (deferred work — current values are fine for the launch profile and
# adjustable in code).
_PLATFORM_TIMING_PROFILES: dict[str, dict[str, float | str]] = {
    # Maps the bare-domain canonical hostname (matching the same
    # subdomain semantics as ``captcha_policy._matches`` — bare entry
    # matches apex AND any subdomain). Keep alphabetical for diff
    # readability.
    "facebook.com": {
        "label": "facebook",
        "mu_s": 4.0, "sigma_s": 1.2, "min_s": 1.5, "max_s": 8.0,
    },
    "instagram.com": {
        "label": "instagram",
        "mu_s": 4.0, "sigma_s": 1.2, "min_s": 1.5, "max_s": 8.0,
    },
    "linkedin.com": {
        "label": "linkedin",
        "mu_s": 3.0, "sigma_s": 1.0, "min_s": 1.0, "max_s": 6.0,
    },
    "twitter.com": {
        "label": "twitter",
        "mu_s": 2.5, "sigma_s": 0.8, "min_s": 0.8, "max_s": 5.0,
    },
    "x.com": {
        "label": "x",
        "mu_s": 2.5, "sigma_s": 0.8, "min_s": 0.8, "max_s": 5.0,
    },
}


def _canonical_host(url: str) -> str | None:
    """Lower-cased hostname with leading ``www.`` stripped, or None.

    Mirrors ``captcha_policy._hostname`` so the bare-domain matching
    rules below behave consistently across the two modules.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    return host


def get_platform_timing_profile(url: str) -> dict | None:
    """Return the timing profile dict for ``url``'s host, or ``None``.

    Bare-domain match: an entry ``"linkedin.com"`` matches both
    ``linkedin.com`` and any subdomain (``mobile.linkedin.com``,
    ``api.linkedin.com``). Returns ``None`` when the host isn't on
    the protected-platform list.

    The returned dict is the live module-level entry — do not mutate.
    """
    host = _canonical_host(url)
    if host is None:
        return None
    if host in _PLATFORM_TIMING_PROFILES:
        return _PLATFORM_TIMING_PROFILES[host]
    # Subdomain match: walk parents (``mobile.linkedin.com`` →
    # ``linkedin.com``). Uses the same right-anchored suffix logic as
    # captcha_policy so behavior is consistent across modules.
    for entry, profile in _PLATFORM_TIMING_PROFILES.items():
        if host.endswith("." + entry):
            return profile
    return None


def pick_platform_pre_nav_delay(
    url: str,
    *,
    rng: random.Random | None = None,
) -> tuple[float, str | None]:
    """Sample a pre-nav dwell for ``url`` based on its platform profile.

    Returns ``(delay_seconds, label)``:
      * ``(0.0, None)`` — host is not on the protected-platform list;
        caller skips the delay entirely.
      * ``(delay, label)`` — sample a clamped Gaussian per the
        profile's μ/σ/min/max. ``label`` is the platform short-name
        (``"linkedin"``, ``"x"``, ...) for logging.

    Sampling: ``rng.gauss(mu, sigma)`` clamped into ``[min_s, max_s]``.
    Real-user arrival timing on these platforms is well-modeled by a
    truncated normal — power-law / log-normal alternatives don't match
    the underlying behavior any better at this scale and are harder
    to tune.
    """
    profile = get_platform_timing_profile(url)
    if profile is None:
        return 0.0, None
    rng = rng or random
    mu = float(profile["mu_s"])
    sigma = float(profile["sigma_s"])
    lo = float(profile["min_s"])
    hi = float(profile["max_s"])
    sample = rng.gauss(mu, sigma)
    if sample < lo:
        sample = lo
    elif sample > hi:
        sample = hi
    return sample, str(profile["label"])





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


# ── §6.6 NetworkInformation helper for non-Firefox future paths ───────────────


def pick_network_info(agent_id: str) -> dict:
    """Stable per-agent ``navigator.connection`` values.

    The current browser path is Firefox-shaped, and real Firefox does not
    expose NetworkInformation, so ``build_launch_options`` does not apply
    these values today. Keep the helper deterministic for a future
    Chromium-shaped path where exposing the API would be coherent with the UA.

    Real Chromium-family desktop users on broadband / 4G / 5G report
    ``effectiveType="4g"`` overwhelmingly; the variability is in
    ``downlink`` (Mbps) and ``rtt`` (ms).

    Picks per-agent from plausible bands:
      - downlink: 5–20 Mbps (covers home broadband, mobile 4G good signal)
      - rtt: 20–120 ms (covers wired through mobile)
      - saveData: rare True (~3%), mostly False

    Deterministic from ``agent_id`` so the same agent reports the same
    network shape across browser restarts. SHA-256 splits into independent
    4-byte words for each field — using one byte each would give a coarse
    256-bucket distribution and visible quantisation on fleet-scale analysis.
    """
    digest = hashlib.sha256(f"netinfo:{agent_id}".encode("utf-8")).digest()
    dl_unit = int.from_bytes(digest[:4], "big") / (1 << 32)   # [0, 1)
    rtt_unit = int.from_bytes(digest[4:8], "big") / (1 << 32)  # [0, 1)
    et_unit = int.from_bytes(digest[8:12], "big") / (1 << 32)  # [0, 1)
    sd_unit = int.from_bytes(digest[12:16], "big") / (1 << 32)  # [0, 1)
    # Add light entropy on ``effectiveType`` and ``saveData`` so the fleet
    # doesn't uniformly report ``("4g", False)`` — that's a single
    # fingerprint cluster, opposite of §6.1's resolution-pool intent.
    # Weights match real-world desktop populations: ``4g`` dominates
    # but ``3g`` (mobile-tether or WAN-degraded) is non-trivial; ``2g``
    # is plausibly rare; ``slow-2g`` is essentially never on desktop.
    # ``saveData=True`` is rare (~3% of desktop sessions per public
    # CrUX-style data) but not zero.
    if et_unit < 0.92:
        effective_type = "4g"
    elif et_unit < 0.99:
        effective_type = "3g"
    else:
        effective_type = "2g"
    save_data = sd_unit < 0.03
    return {
        "effectiveType": effective_type,
        "downlink": round(5.0 + dl_unit * 15.0, 1),
        "rtt": int(20 + rtt_unit * 100),
        "saveData": save_data,
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


def build_launch_options(
    agent_id: str,
    profile_dir: str,
    proxy: dict | None = None,
    *,
    device_profile: str | None = None,
) -> dict:
    """Build Camoufox AsyncNewBrowser kwargs for an agent.

    Args:
        proxy: Camoufox proxy dict (server, username, password) or None.
               Proxy is always resolved per-agent by the mesh host and pushed
               to the browser service — callers must pass it explicitly.
        device_profile: Name of a profile from :data:`_DEVICE_PROFILES`
               (``desktop-windows`` | ``desktop-macos`` | ``mobile-ios`` |
               ``mobile-android``) or ``None`` for the default
               (``desktop-windows``). Mobile profiles pin UA + viewport +
               DPR + ``is_mobile``/``has_touch`` from the profile dict;
               desktop profiles still use the per-agent resolution pool.

    Environment variables (all optional):
      BROWSER_OS     — "windows" (default) | "macos" | "linux".
                       Ignored when ``device_profile`` is non-default — the
                       profile's ``camoufox_os`` field takes precedence so
                       e.g. ``mobile-ios`` uses macOS plumbing regardless
                       of operator BROWSER_OS.
      BROWSER_LOCALE — BCP-47 locale tag, e.g. "en-US" (default)
      BROWSER_UA_VERSION — Firefox version to report in User-Agent, e.g. "138.0".
                           If set, overrides the UA string to spoof a newer Firefox.
                           Useful when Camoufox's bundled Firefox is too old for
                           sites that enforce minimum browser versions (e.g. Shopify).
                           Ignored when a mobile profile pins its own UA.
    """
    profile = get_device_profile(device_profile)
    profile_name = device_profile or DEFAULT_DEVICE_PROFILE
    if profile_name not in _DEVICE_PROFILES:
        profile_name = DEFAULT_DEVICE_PROFILE

    # ── OS fingerprint ────────────────────────────────────────────────────────
    # Default profile: read BROWSER_OS env (Windows = ≈70% desktop market
    # share, the least-suspicious server-hosted fingerprint). Non-default
    # profiles pin ``camoufox_os`` directly so the OS plumbing (font stack,
    # platform string) matches the device class regardless of operator
    # BROWSER_OS.  Linux is an immediate datacenter/bot signal for sites
    # that check navigator.platform — only allowed when explicitly chosen
    # (mobile-android profile uses it deliberately).
    if profile_name == DEFAULT_DEVICE_PROFILE:
        os_hint = os.environ.get("BROWSER_OS", "windows").lower()
        if os_hint not in ("windows", "macos", "linux"):
            logger.warning("Invalid BROWSER_OS %r, defaulting to 'windows'", os_hint)
            os_hint = "windows"
    else:
        os_hint = profile["camoufox_os"]

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
    #
    # Mobile profiles override the resolution pool: a mobile fingerprint
    # claiming 1920×1080 would itself be a dead giveaway, so we use the
    # profile's pinned viewport instead. Desktop profiles still use the
    # per-agent pool for fleet-scale diversity.
    profile_viewport = profile.get("viewport")
    if profile_viewport:
        width = int(profile_viewport["width"])
        height = int(profile_viewport["height"])
        logger.debug(
            "Agent '%s' device-profile %r viewport: %dx%d",
            agent_id, profile_name, width, height,
        )
    else:
        width, height = pick_resolution(agent_id)
        logger.debug("Agent '%s' resolution: %dx%d", agent_id, width, height)

    options: dict = {
        "headless": False,
        "humanize": True,        # Camoufox mouse-curves + micro-delays
        "os": os_hint,
        "locale": locale,        # navigator.language / Accept-Language header
        "window": (width, height),
        # block_webrtc: Camoufox native toggle — primary defense against
        # Docker container IP leaks via ICE candidates. Firefox prefs below
        # add defense in depth.
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

    options["firefox_user_prefs"] = _stealth_prefs(locale=locale)

    # ── §6.6 NetworkInformation ──────────────────────────────────────────────
    # Real Firefox does not expose ``navigator.connection``. Earlier phases
    # injected per-agent NetworkInformation values through Camoufox config,
    # but that made a Firefox-shaped UA expose a Chromium-only API — a stronger
    # anti-bot signal than the API being absent. Keep ``pick_network_info`` as
    # a deterministic helper for a future Chromium-shaped path, but do not
    # pass any ``navigator.connection.*`` config while `_assert_firefox_ua`
    # requires a Firefox UA.

    # ── User-Agent override ──────────────────────────────────────────────────
    # Two paths:
    #
    # 1. Mobile / non-Firefox device profile pins a UA on the profile dict.
    #    This is the §19.3 mobile-emulation case — the UA is Mobile Safari
    #    (iOS) or Chrome Mobile (Android). The §6.4 ``_assert_firefox_ua``
    #    tripwire is intentionally bypassed here: the operator has chosen a
    #    non-default device profile knowing the UA-engine mismatch. Sites
    #    that key on Sec-CH-UA-* will see no Client Hints (Firefox-engine
    #    behavior) — that's the documented trade-off.
    #
    # 2. Desktop profile + ``BROWSER_UA_VERSION`` env. Camoufox bundles a
    #    specific Firefox build (e.g. 135.0).  Some sites (Shopify, etc.)
    #    enforce minimum browser versions and block old Firefox.
    #    BROWSER_UA_VERSION overrides the reported version without upgrading
    #    the Camoufox binary.
    #
    # Primary mechanism for both: Camoufox's `config` dict with
    # ``navigator.userAgent``. This feeds into Camoufox's fingerprint
    # injection system, which controls both navigator.userAgent (JS) and
    # the User-Agent HTTP header. Fallback: ``general.useragent.override``
    # Firefox pref, in case an older Camoufox version doesn't honour the
    # config dict.
    profile_ua = profile.get("user_agent")
    if profile_ua:
        # §19.3: mobile / non-Firefox profile UA. Skip the Firefox tripwire.
        options.setdefault("config", {})["navigator.userAgent"] = profile_ua
        options["i_know_what_im_doing"] = True
        options["firefox_user_prefs"]["general.useragent.override"] = profile_ua
        logger.info(
            "Device profile %r UA pinned: %s",
            profile_name, _short_ua(profile_ua),
        )
    else:
        ua_version = os.environ.get("BROWSER_UA_VERSION", "")
        if ua_version:
            ua = _build_ua_string(os_hint, ua_version)
            if ua:
                # §6.4 tripwire — the UA we're about to ship MUST be Firefox.
                # If a future change introduces a Chromium UA, this raises so
                # the developer has to wire Sec-CH-UA-* overrides first.
                _assert_firefox_ua(ua)
                options.setdefault("config", {})["navigator.userAgent"] = ua
                options["i_know_what_im_doing"] = True
                options["firefox_user_prefs"]["general.useragent.override"] = ua
                logger.info("UA override: Firefox/%s", ua_version)

    return options


def _short_ua(ua: str, *, limit: int = 80) -> str:
    """Return a truncated UA for human-readable log lines."""
    return ua if len(ua) <= limit else ua[: limit - 1] + "…"


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

    # Sanity-bound the major version. A typo like "99999999.0" would
    # produce a UA string that's a strong outlier signal — exactly the
    # detection axis §6.4 tries to harden against. Firefox major
    # versions are currently in the 130s; cap at 200 to leave generous
    # headroom while rejecting nonsense.
    try:
        major = int(parts[0])
    except ValueError:
        return None
    if major < 1 or major > 200:
        logger.warning(
            "BROWSER_UA_VERSION major %d is out of sane bound [1, 200]; "
            "ignoring",
            major,
        )
        return None

    _OS_UA_TEMPLATES = {
        "macos": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:{v}) Gecko/20100101 Firefox/{v}",
        "linux": "Mozilla/5.0 (X11; Linux x86_64; rv:{v}) Gecko/20100101 Firefox/{v}",
        "windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{v}) Gecko/20100101 Firefox/{v}",
    }
    template = _OS_UA_TEMPLATES.get(os_hint, _OS_UA_TEMPLATES["windows"])
    return template.format(v=version)


def _build_accept_languages(locale: str) -> str:
    """Build a coherent ``intl.accept_languages`` string from a BCP-47 tag.

    Detection products (Cloudflare, FingerprintJS Pro, DataDome) compare
    ``navigator.languages[0]`` and the ``Accept-Language`` header to the
    primary script-determined locale. Real browsers ship a multi-entry
    quality list — single-entry lists are themselves a soft cluster
    signal. Build:

        ``<locale>,<base>;q=0.9,en;q=0.5``

    where ``<base>`` is the language part (``en-US`` → ``en``). When
    the language already is ``en``, suppress the duplicate fallback.
    """
    locale = (locale or "en-US").strip()
    if "-" in locale:
        base = locale.split("-", 1)[0].lower()
    else:
        base = locale.lower()
    if base == "en":
        # ``en-US,en;q=0.5`` — what a real Firefox ships by default.
        return f"{locale},en;q=0.5"
    return f"{locale},{base};q=0.9,en;q=0.5"


def _stealth_prefs(locale: str = "en-US") -> dict:
    """Return Firefox user preferences that minimise bot-detection surface.

    Design rationale for each group is inline.  Correctness over completeness:
    only prefs with a clear detection impact are included.

    ``locale`` is the BCP-47 tag from ``BROWSER_LOCALE``; it drives the
    ``intl.accept_languages`` pref so the ``Accept-Language`` header,
    ``navigator.language`` and ``navigator.languages`` cohere.
    """
    return {
        # ── Accept-Language coherence (§ antibot-audit F1) ────────────────────
        # Without an explicit ``intl.accept_languages`` Firefox falls back
        # to the build-locale default (en-US,en;q=0.5) — which can mismatch
        # the BROWSER_LOCALE chosen for navigator.language. Detection
        # products free-harvest ``navigator.languages[0] !== Accept-Language[0]``
        # as a bot signal. Pin the pref to a coherent quality list derived
        # from the same locale string Camoufox uses for navigator.language.
        "intl.accept_languages": _build_accept_languages(locale),
        # ── UI cosmetics ──────────────────────────────────────────────────────
        "extensions.activeThemeID": "firefox-compact-dark@mozilla.org",
        "browser.uidensity": 1,
        "browser.tabs.inTitlebar": 1,

        # ── WebRTC: defense in depth ─────────────────────────────────────────
        # RTCPeerConnection leaks the Docker container's internal IP via ICE
        # candidates even when behind a proxy. Camoufox's ``block_webrtc=True``
        # is the primary toggle, BUT a single point of failure: if a future
        # Camoufox release silently drops or renames the option, ICE candidates
        # leak with no other guard. Set the underlying Firefox prefs explicitly
        # too — they're idempotent with the Camoufox toggle and survive any
        # upstream rename.
        #
        # Trade-off: ``media.peerconnection.enabled=false`` makes
        # ``RTCPeerConnection`` itself ``undefined`` — and the API's absence
        # is itself a fingerprint cluster vs the real Firefox population
        # (where it's overwhelmingly present). The two ``ice.*`` prefs are
        # the better long-term shape because they keep the constructor present
        # but prevent host candidates; flipping the master switch is the
        # belt-and-suspenders fallback for the primary IP-leak threat. Operators
        # who care more about cluster shape than IP leakage can override
        # ``media.peerconnection.enabled`` to True via settings.json.
        "media.peerconnection.enabled": False,
        "media.peerconnection.ice.default_address_only": True,
        "media.peerconnection.ice.no_host": True,
        # Disable mDNS rewrite of host candidates (Firefox feature that can
        # itself be a fingerprint-able shape if mismatched with our above
        # settings).
        "media.peerconnection.ice.obfuscate_host_addresses": False,

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

        # ── Bundled extensions (Phase 4 §7.1) ─────────────────────────────────
        # The profile-schema v3 migration drops uBlock Origin's signed XPI
        # into ``{profile}/extensions/{addon-id}.xpi``. Firefox detects
        # extensions there at launch but only enables them automatically
        # when these prefs cooperate:
        #   * ``autoDisableScopes=0`` — bit-mask of scopes whose extensions
        #     start *disabled* on first install. 0 means none of the
        #     auto-discovered scopes are disabled, so our profile-bundled
        #     uBlock loads on first run.
        #   * ``startupScanScopes=15`` — bit-mask of scopes scanned for
        #     extension changes at every startup. 15 = all (1=app, 2=system,
        #     4=user, 8=profile). Without this, Firefox only re-scans the
        #     profile dir when the addon DB tracks a known XPI; a freshly-
        #     dropped XPI from our migration would otherwise wait until the
        #     next manifest change.
        #   * ``xpinstall.signatures.required=False`` — Camoufox's patched
        #     Firefox already loosens this, but setting it explicitly keeps
        #     us robust against upstream Camoufox changes that re-tighten it.
        "extensions.autoDisableScopes": 0,
        "extensions.startupScanScopes": 15,
        "xpinstall.signatures.required": False,

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


# ── §19.3 Navigator override init script (Phase 10 §21) ──────────────────────


# Wired by ``BrowserManager._start_browser`` at context creation via
# ``BrowserContext.add_init_script``.  Runs at ``document_start`` on every
# page and frame BEFORE any site script has a chance to read these
# properties — that's the only point at which we can shape the values
# the page sees without leaving a tell-tale "page-script came back with a
# different value than the runtime exposes" inconsistency.
#
# Only emitted for profiles where ``user_agent_data_mobile`` is True
# (the two mobile profiles). Desktop profiles let Camoufox + the BrowserForge
# fingerprint drive these values; injecting an override would itself be
# a fingerprint anomaly versus the real desktop population.
_MOBILE_NAVIGATOR_INIT_SCRIPT = """
(() => {
  try {
    // navigator.userAgentData: Chromium-only API today, but Mobile Safari
    // doesn't expose it either, so emitting it for mobile-android is the
    // right shape and emitting an absence for mobile-ios matches real
    // iOS Safari. We define both behaviors below — service.py picks which
    // to inject based on the profile.
    if (typeof navigator !== 'undefined') {
      try {
        Object.defineProperty(navigator, 'maxTouchPoints', {
          get: () => __MAX_TOUCH_POINTS__,
          configurable: true,
        });
      } catch (_e) { /* ignore */ }
      try {
        Object.defineProperty(navigator, 'platform', {
          get: () => '__PLATFORM__',
          configurable: true,
        });
      } catch (_e) { /* ignore */ }
      // userAgentData: only emit when the profile claims a Chromium-shaped
      // engine (mobile-android). For Mobile Safari we leave it undefined,
      // matching the real-Safari population.
      if (__EMIT_USER_AGENT_DATA__) {
        try {
          const data = Object.freeze({
            mobile: __MOBILE__,
            platform: '__UA_DATA_PLATFORM__',
            brands: [],
            getHighEntropyValues: () => Promise.resolve({
              mobile: __MOBILE__,
              platform: '__UA_DATA_PLATFORM__',
            }),
            toJSON: () => ({
              mobile: __MOBILE__,
              platform: '__UA_DATA_PLATFORM__',
            }),
          });
          Object.defineProperty(navigator, 'userAgentData', {
            get: () => data,
            configurable: true,
            enumerable: true,
          });
        } catch (_e) { /* ignore */ }
      }
    }
  } catch (_e) {
    // Defensive: any failure here is operator-debuggable via the §6.3
    // navigator self-test, which will flag a platform mismatch.
  }
})();
"""


def build_mobile_init_script(profile: dict) -> str | None:
    """Return the JS init-script body to inject for a mobile profile.

    Returns ``None`` when the profile does not need mobile overrides
    (any desktop profile). The returned string is a complete
    ``add_init_script``-ready snippet — the caller passes it directly to
    ``BrowserContext.add_init_script(script=...)``.

    Two mobile shapes diverge on ``navigator.userAgentData``:
      * Android profile (Chromium-shaped UA): emits a frozen
        ``userAgentData`` object with ``mobile=true``.
      * iOS profile (Safari-shaped UA): leaves ``userAgentData``
        absent (matches real iOS Safari, which doesn't expose Client
        Hints).

    Both shapes inject ``maxTouchPoints`` and ``platform``.
    """
    if not profile.get("is_mobile"):
        return None

    platform = profile.get("platform_navigator") or "iPhone"
    max_touch_points = int(profile.get("max_touch_points") or 5)

    # iOS profile: real Safari hides userAgentData. Android: emit it.
    # Heuristic: emit userAgentData iff the platform string looks
    # Linux-/Android-shaped. iPhone / iPad / Mac platforms get nothing.
    is_chromium_shaped = "Linux" in platform or "Android" in platform
    ua_data_platform = "Android" if "Linux" in platform or "Android" in platform else "iOS"

    return (
        _MOBILE_NAVIGATOR_INIT_SCRIPT
        .replace("__MAX_TOUCH_POINTS__", str(max_touch_points))
        .replace("__PLATFORM__", platform)
        .replace("__EMIT_USER_AGENT_DATA__", "true" if is_chromium_shaped else "false")
        .replace("__UA_DATA_PLATFORM__", ua_data_platform)
        .replace("__MOBILE__", "true")
    )
