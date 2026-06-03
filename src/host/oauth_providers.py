"""OAuth provider registry for one-click third-party "Connect" integrations.

Declarative, provider-agnostic configuration for the authorization-code flow
that backs in-engine integrations (Google Drive / Gmail / Calendar first).
Provider endpoints are fixed here — never user-supplied — so the connect /
callback flow adds no SSRF surface.

Bring-your-own-app model (Option B): the operator registers their own OAuth
app with the provider and supplies the client_id / client_secret as
system-tier credentials (``OPENLEGION_SYSTEM_<PROVIDER>_CLIENT_ID`` /
``_CLIENT_SECRET``). The engine performs the code exchange and token refresh;
agents only ever see a short-lived access token via ``$CRED{<connection>}``.

The registry is deliberately broker-ready (Option A): first-party connectors
differ only in where the client credentials come from and who runs the consent
dance — the provider table below is unchanged.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScopeBundle:
    """A named, least-privilege bundle of provider scopes the operator can pick."""

    key: str
    label: str
    scopes: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class OAuthProvider:
    """Fixed configuration for a single OAuth provider."""

    key: str
    label: str
    authorize_url: str
    token_url: str
    scope_bundles: tuple[ScopeBundle, ...]
    # Scopes always requested (e.g. for account/email display). Merged with
    # whatever bundles the operator selects.
    base_scopes: tuple[str, ...] = ()
    # Extra query params appended to the authorize URL. For Google these force
    # a refresh token to be issued (``access_type=offline`` + ``prompt=consent``).
    extra_authorize_params: dict[str, str] = field(default_factory=dict)
    # Best-effort account display (which account did the user connect).
    userinfo_url: str | None = None
    userinfo_email_field: str = "email"
    # Best-effort token revocation on disconnect.
    revoke_url: str | None = None
    uses_pkce: bool = True
    # True when this provider's access tokens expire AND a refresh token is
    # required to keep the connection alive (Google). The callback rejects a
    # connection that comes back without a refresh token rather than storing one
    # that silently dies at first expiry. Set False for providers whose tokens
    # don't expire / need no refresh (e.g. Notion).
    refresh_required: bool = True

    @property
    def client_id_key(self) -> str:
        """System-tier credential name holding the operator's OAuth client id."""
        return f"{self.key}_client_id"

    @property
    def client_secret_key(self) -> str:
        """System-tier credential name holding the operator's OAuth client secret."""
        return f"{self.key}_client_secret"

    def bundle(self, key: str) -> ScopeBundle | None:
        for b in self.scope_bundles:
            if b.key == key:
                return b
        return None

    def resolve_scopes(self, bundle_keys: list[str]) -> list[str]:
        """Merge base scopes with the requested bundles, de-duplicated, ordered.

        Unknown bundle keys are ignored (callers validate separately if they
        want to reject them). Always includes ``base_scopes`` so account
        display keeps working regardless of which data bundles were chosen.
        """
        out: list[str] = list(self.base_scopes)
        for key in bundle_keys:
            b = self.bundle(key)
            if b is None:
                continue
            for s in b.scopes:
                if s not in out:
                    out.append(s)
        # Drop any accidental dupes from base_scopes overlap, preserve order.
        seen: set[str] = set()
        ordered: list[str] = []
        for s in out:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        return ordered


GOOGLE = OAuthProvider(
    key="google",
    label="Google",
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    base_scopes=("openid", "email"),
    extra_authorize_params={
        # ``offline`` + ``consent`` are both required for Google to return a
        # refresh token (and to re-issue one on re-consent). Without them the
        # connection silently can't be refreshed and dies at first expiry.
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    },
    userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
    revoke_url="https://oauth2.googleapis.com/revoke",
    uses_pkce=True,
    scope_bundles=(
        ScopeBundle(
            key="drive_readonly",
            label="Drive (read-only)",
            scopes=("https://www.googleapis.com/auth/drive.readonly",),
            description="Read files and folders in Google Drive.",
        ),
        ScopeBundle(
            key="drive_file",
            label="Drive (files created by app)",
            scopes=("https://www.googleapis.com/auth/drive.file",),
            description="Read/write only the files this app creates or opens.",
        ),
        ScopeBundle(
            key="gmail_readonly",
            label="Gmail (read-only)",
            scopes=("https://www.googleapis.com/auth/gmail.readonly",),
            description="Read Gmail messages and metadata.",
        ),
        ScopeBundle(
            key="gmail_send",
            label="Gmail (send)",
            scopes=("https://www.googleapis.com/auth/gmail.send",),
            description="Send mail as the connected account.",
        ),
        ScopeBundle(
            key="calendar_readonly",
            label="Calendar (read-only)",
            scopes=("https://www.googleapis.com/auth/calendar.readonly",),
            description="Read calendars and events.",
        ),
        ScopeBundle(
            key="calendar",
            label="Calendar (read/write)",
            scopes=("https://www.googleapis.com/auth/calendar",),
            description="Read and write calendars and events.",
        ),
    ),
)


OAUTH_PROVIDERS: dict[str, OAuthProvider] = {
    GOOGLE.key: GOOGLE,
}


def get_provider(key: str) -> OAuthProvider | None:
    return OAUTH_PROVIDERS.get((key or "").lower())


def generate_pkce() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for PKCE S256.

    ``code_verifier`` is a high-entropy URL-safe string (43–128 chars per
    RFC 7636); ``code_challenge`` is the base64url-encoded SHA-256 of it with
    padding stripped.
    """
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge
