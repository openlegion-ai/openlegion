"""Stealth/fingerprint configuration for Camoufox instances.

Handles Camoufox launch options, proxy configuration, and
BrowserForge fingerprint generation.
"""

from __future__ import annotations

import os

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
    logger.info("Proxy configured: %s", proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url)
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
    }
    if proxy:
        options["proxy"] = proxy
        options["geoip"] = True

    # Persistent profile for cookies/sessions
    options["persistent_context"] = True
    options["user_data_dir"] = profile_dir

    return options
