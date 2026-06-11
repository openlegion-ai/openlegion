"""OAuth 2.1 discovery + Dynamic Client Registration for remote MCP
connectors (Phase 3 of the connectors plan).

The trust inversion this module must respect (plan D16): the *operator*
pasted the MCP URL, but everything discovery returns — the protected-
resource metadata, the authorization-server choice, the token /
authorization / registration endpoints — is controlled by the **remote
server**. ``oauth_providers.py``'s "endpoints are fixed here, never
user-supplied" invariant does not hold on this path, so every fetched
AND discovered URL gets the same posture as the gateway: https-only,
resolved-IP private-range blocklist (``mcp_gateway._assert_public_host``),
redirects disabled, 64 KB response caps, content-type checks.

Errors carry the failed step name (``MCPOAuthError.step``) so the
dashboard can say exactly which part of the chain broke — per plan
§11-Q4 there is deliberately NO bring-your-own-client fallback in v1;
a server without discovery/DCR support gets a diagnosable error, not a
second secret-entry path.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from src.host.mcp_gateway import _assert_public_host
from src.shared.utils import setup_logging

logger = setup_logging("host.mcp_oauth")

_FETCH_TIMEOUT = 20
_MAX_METADATA_BYTES = 64 * 1024
_CLIENT_NAME = "OpenLegion"


class MCPOAuthError(RuntimeError):
    """A discovery/registration step failed. ``step`` names it."""

    def __init__(self, step: str, message: str) -> None:
        super().__init__(message)
        self.step = step


@dataclass(frozen=True)
class Discovery:
    """Validated OAuth endpoints for one remote MCP server."""

    authorization_server: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None


async def _validate_endpoint_url(step: str, url: str) -> str:
    """https + public-IP check for a DISCOVERED URL — server-controlled
    input, full D16 posture."""
    p = urlparse(url)
    if p.scheme != "https" or not p.hostname:
        raise MCPOAuthError(
            step, f"{step}: discovered URL must be https:// (got {url[:120]!r})",
        )
    try:
        await _assert_public_host(url)
    except Exception as exc:
        raise MCPOAuthError(step, f"{step}: {exc}") from exc
    return url


async def _fetch_json(
    client: httpx.AsyncClient, step: str, url: str,
) -> dict | None:
    """GET a metadata document under the D16 fetch discipline. Returns
    None on 404 (callers try the next well-known location); raises
    ``MCPOAuthError`` on anything else unexpected."""
    await _validate_endpoint_url(step, url)
    try:
        resp = await client.get(url, headers={"Accept": "application/json"})
    except httpx.HTTPError as exc:
        raise MCPOAuthError(step, f"{step}: fetch failed ({exc})") from exc
    if resp.status_code == 404:
        return None
    if resp.is_redirect:
        # follow_redirects is off by construction; a redirect here is a
        # server trying to bounce us to an unchecked origin.
        raise MCPOAuthError(step, f"{step}: redirect responses are not followed")
    if not resp.is_success:
        raise MCPOAuthError(
            step, f"{step}: HTTP {resp.status_code} from {url[:120]}",
        )
    if len(resp.content) > _MAX_METADATA_BYTES:
        raise MCPOAuthError(step, f"{step}: metadata document exceeds 64KB")
    ctype = resp.headers.get("content-type", "")
    if "json" not in ctype:
        raise MCPOAuthError(
            step, f"{step}: expected JSON, got content-type {ctype[:60]!r}",
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise MCPOAuthError(step, f"{step}: invalid JSON") from exc
    if not isinstance(data, dict):
        raise MCPOAuthError(step, f"{step}: metadata must be a JSON object")
    return data


def _well_known_urls(base: str, suffix: str, path: str) -> list[str]:
    """RFC 8414/9728 well-known locations: path-aware first (the MCP
    spec's preference for path-scoped servers), then origin root."""
    urls = []
    clean_path = path.rstrip("/")
    if clean_path:
        urls.append(f"{base}/.well-known/{suffix}{clean_path}")
    urls.append(f"{base}/.well-known/{suffix}")
    return urls


def _new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        follow_redirects=False, timeout=_FETCH_TIMEOUT,
    )


async def discover(mcp_url: str) -> Discovery:
    """Resolve the OAuth endpoints for a remote MCP server.

    Chain: protected-resource metadata (RFC 9728) on the MCP origin →
    ``authorization_servers[0]`` → AS metadata (RFC 8414, with the OIDC
    ``openid-configuration`` location as fallback) → validated
    authorization/token(/registration) endpoints.
    """
    p = urlparse(mcp_url)
    origin = f"{p.scheme}://{p.netloc}"
    async with _new_client() as client:
        # ── step 1: protected-resource metadata ──────────────────
        prm = None
        for url in _well_known_urls(
            origin, "oauth-protected-resource", p.path,
        ):
            prm = await _fetch_json(client, "protected-resource metadata", url)
            if prm is not None:
                break
        if prm is None:
            raise MCPOAuthError(
                "protected-resource metadata",
                "the server publishes no /.well-known/oauth-protected-resource "
                "document — it does not support the MCP OAuth flow",
            )
        servers = prm.get("authorization_servers") or []
        if not servers or not isinstance(servers, list):
            raise MCPOAuthError(
                "protected-resource metadata",
                "metadata lists no authorization_servers",
            )
        as_issuer = str(servers[0]).rstrip("/")
        await _validate_endpoint_url("authorization server", as_issuer)

        # ── step 2: AS metadata ──────────────────────────────────
        as_url = urlparse(as_issuer)
        as_origin = f"{as_url.scheme}://{as_url.netloc}"
        meta = None
        candidates = _well_known_urls(
            as_origin, "oauth-authorization-server", as_url.path,
        ) + _well_known_urls(as_origin, "openid-configuration", as_url.path)
        for url in candidates:
            meta = await _fetch_json(client, "authorization-server metadata", url)
            if meta is not None:
                break
        if meta is None:
            raise MCPOAuthError(
                "authorization-server metadata",
                f"no RFC 8414 / OIDC metadata found for {as_issuer}",
            )

        authz = meta.get("authorization_endpoint", "")
        token = meta.get("token_endpoint", "")
        if not authz or not token:
            raise MCPOAuthError(
                "authorization-server metadata",
                "metadata is missing authorization_endpoint or token_endpoint",
            )
        await _validate_endpoint_url("authorization endpoint", str(authz))
        await _validate_endpoint_url("token endpoint", str(token))
        registration = meta.get("registration_endpoint") or None
        if registration:
            await _validate_endpoint_url(
                "registration endpoint", str(registration),
            )
        return Discovery(
            authorization_server=as_issuer,
            authorization_endpoint=str(authz),
            token_endpoint=str(token),
            registration_endpoint=str(registration) if registration else None,
        )


async def register_client(
    registration_endpoint: str, redirect_uri: str,
) -> tuple[str, str | None]:
    """RFC 7591 Dynamic Client Registration. Returns
    ``(client_id, client_secret-or-None)``.

    Requests ``client_secret_post`` (our exchange/refresh send the
    secret in the form body); if the AS rejects the metadata, retries
    once as a public client (``none`` — PKCE is the proof). The secret,
    when minted, goes into the connection blob via the callback — never
    the connector record on disk, never the logs.
    """
    base = {
        "client_name": _CLIENT_NAME,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    async with _new_client() as client:
        for auth_method in ("client_secret_post", "none"):
            try:
                resp = await client.post(
                    registration_endpoint,
                    json={**base, "token_endpoint_auth_method": auth_method},
                    headers={"Accept": "application/json"},
                )
            except httpx.HTTPError as exc:
                raise MCPOAuthError(
                    "client registration",
                    f"client registration: request failed ({exc})",
                ) from exc
            if resp.status_code == 400 and auth_method == "client_secret_post":
                continue  # retry as a public client
            if resp.status_code not in (200, 201):
                raise MCPOAuthError(
                    "client registration",
                    f"client registration: HTTP {resp.status_code} "
                    f"from {registration_endpoint[:120]}",
                )
            if len(resp.content) > _MAX_METADATA_BYTES:
                raise MCPOAuthError(
                    "client registration", "registration response exceeds 64KB",
                )
            try:
                data = resp.json()
            except ValueError as exc:
                raise MCPOAuthError(
                    "client registration", "registration returned invalid JSON",
                ) from exc
            client_id = str(data.get("client_id", ""))
            if not client_id:
                raise MCPOAuthError(
                    "client registration", "registration returned no client_id",
                )
            secret = data.get("client_secret") or None
            logger.info(
                "DCR registered client for %s (auth_method=%s, secret=%s)",
                registration_endpoint, auth_method,
                "yes" if secret else "no",
            )
            return client_id, secret
    raise MCPOAuthError(
        "client registration",
        "the authorization server rejected both confidential and "
        "public client registration",
    )
