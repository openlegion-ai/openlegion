"""Stealth/fingerprint configuration for Camoufox instances.

Handles Camoufox launch options, proxy configuration, and
BrowserForge fingerprint generation.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from src.shared.utils import setup_logging

logger = setup_logging("browser.stealth")


def get_proxy_config() -> dict | None:
    """Build proxy config from environment variables.

    Reads BROWSER_PROXY_URL, BROWSER_PROXY_USER, BROWSER_PROXY_PASS.
    Returns dict suitable for Camoufox proxy parameter, or None.
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
    safe_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}" if parsed.port else f"{parsed.scheme}://{parsed.hostname}"
    logger.info("Proxy configured: %s", safe_url)
    return config


def build_launch_options(
    agent_id: str,
    profile_dir: str,
) -> dict:
    """Build Camoufox AsyncNewBrowser kwargs for an agent.

    Returns a dict of keyword arguments for camoufox.AsyncNewBrowser.
    """
    proxy = get_proxy_config()
    options: dict = {
        "headless": False,
        "humanize": True,
        "os": "linux",
        # Pin window to match Xvnc display so the browser fills the viewport
        "window": (1920, 1080),
    }
    # Constrain BrowserForge fingerprint screen size to match Xvnc display
    try:
        from browserforge.fingerprints import Screen
        options["screen"] = Screen(max_width=1920, max_height=1080)
    except ImportError:
        pass  # browserforge only available in browser container
    if proxy:
        options["proxy"] = proxy
        options["geoip"] = True

    # Persistent profile for cookies/sessions
    options["persistent_context"] = True
    options["user_data_dir"] = profile_dir

    # Cosmetic Firefox prefs — dark theme to blend with VNC background,
    # compact UI density, and tabs-in-titlebar to save vertical space.
    options["firefox_user_prefs"] = {
        "extensions.activeThemeID": "firefox-compact-dark@mozilla.org",
        "browser.uidensity": 1,  # compact
        "browser.tabs.inTitlebar": 1,
    }

    return options
