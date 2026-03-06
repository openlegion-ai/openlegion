"""Dashboard session cookie verification.

Shared between the dashboard API router (HTTP) and the WebSocket event
stream so both enforce the same auth policy.

No-op when the access token file doesn't exist (self-hosted / dev mode).
In hosted mode, verifies the ``ol_session`` HMAC cookie set by the Caddy
forward_auth gate.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac_mod
import time
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("dashboard.auth")

_ACCESS_TOKEN_PATH = "/opt/openlegion/.access_token"
_HOSTED_INDICATOR = "/opt/openlegion/.subdomain"
_cached_cookie_key: bytes | None = None
_cookie_key_loaded = False
_is_hosted: bool | None = None

# Maximum cookie lifetime (defense-in-depth).  Even if the auth gate
# issues a longer-lived cookie, the engine rejects cookies older than this.
COOKIE_MAX_AGE = 24 * 3600  # 24 hours


def _dashboard_cookie_key() -> bytes | None:
    """Derive cookie signing key from the access token on disk.

    Cached after first read — the access token does not change during
    the lifetime of the engine process (provisioner restarts the service
    when reconfiguring).
    """
    global _cached_cookie_key, _cookie_key_loaded
    if _cookie_key_loaded:
        return _cached_cookie_key
    try:
        token = Path(_ACCESS_TOKEN_PATH).read_text().strip()
        if not token:
            _cached_cookie_key = None
        else:
            _cached_cookie_key = _hmac_mod.new(
                token.encode(), b"ol-cookie-signing", hashlib.sha256
            ).digest()
    except (FileNotFoundError, PermissionError):
        _cached_cookie_key = None
    _cookie_key_loaded = True
    return _cached_cookie_key


def verify_session_cookie(cookie_value: str) -> str | None:
    """Verify an ol_session cookie value.

    Returns None on success, or an error message string on failure.
    Returns None (success) when no access token is configured (dev mode).
    """
    key = _dashboard_cookie_key()
    if key is None:
        # No access token — allow in dev/self-hosted mode, but deny in
        # hosted mode where the auth gate should always be running.
        global _is_hosted
        if _is_hosted is None:
            _is_hosted = Path(_HOSTED_INDICATOR).exists()
        if _is_hosted:
            return "Dashboard authentication required (access token not configured)"
        return None  # Dev / self-hosted mode — skip check
    if not cookie_value:
        return "Dashboard authentication required"
    try:
        expiry_str, sig = cookie_value.split(".", 1)
        expiry = int(expiry_str)
    except (ValueError, AttributeError):
        return "Invalid session cookie"
    now = time.time()
    if expiry < now:
        return "Session expired"
    # Defense-in-depth: reject cookies with expiry unreasonably far in future
    if expiry > now + COOKIE_MAX_AGE + 300:  # 5 min clock skew tolerance
        return "Invalid session cookie"
    expected = _hmac_mod.new(key, expiry_str.encode(), hashlib.sha256).hexdigest()
    if not _hmac_mod.compare_digest(sig, expected):
        return "Invalid session cookie"
    return None


def reset_cache() -> None:
    """Reset the cached cookie key (for testing)."""
    global _cached_cookie_key, _cookie_key_loaded
    _cached_cookie_key = None
    _cookie_key_loaded = False
