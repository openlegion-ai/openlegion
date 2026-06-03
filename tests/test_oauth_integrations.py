"""Tests for the OAuth "Connect" integrations flow.

Covers the vault connection store (load/store/remove/list), refresh-on-resolve
(the load-bearing seam that lets ``$CRED{google_drive}`` auto-refresh), the
authorize-URL builder, code exchange, and the single-use/session-bound OAuth
state store. HTTP is mocked — no network.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.host.credentials as cred_mod
from src.host.credentials import CONNECTION_PREFIX, CredentialVault
from src.host.oauth_providers import GOOGLE, generate_pkce, get_provider
from src.host.oauth_state import OAuthStateStore


@pytest.fixture
def captured_env(monkeypatch):
    """Capture _persist/_remove env writes instead of touching the real .env."""
    store: dict[str, str] = {}
    monkeypatch.setattr(
        cred_mod, "_persist_to_env",
        lambda k, val, **kw: store.__setitem__(k, val),
    )
    monkeypatch.setattr(
        cred_mod, "_remove_from_env",
        lambda k, **kw: store.pop(k, None),
    )
    return store


def _fake_response(*, status=200, payload=None, text=""):
    resp = AsyncMock()
    resp.is_success = 200 <= status < 300
    resp.status_code = status
    resp.json = lambda: (payload or {})
    resp.text = text
    return resp


def _vault_with_fake_http(*, post=None, get=None) -> CredentialVault:
    v = CredentialVault()
    fake = AsyncMock()
    fake.is_closed = False
    if post is not None:
        fake.post = post
    if get is not None:
        fake.get = get
    v._http_client = fake
    return v


# ── Provider registry ────────────────────────────────────────────────────

def test_google_provider_registered():
    p = get_provider("google")
    assert p is GOOGLE
    assert p.client_id_key == "google_client_id"
    assert p.client_secret_key == "google_client_secret"
    # offline+consent are required for a refresh token to be issued
    assert p.extra_authorize_params.get("access_type") == "offline"
    assert p.extra_authorize_params.get("prompt") == "consent"


def test_resolve_scopes_merges_base_and_dedupes():
    p = get_provider("google")
    scopes = p.resolve_scopes(["drive_readonly", "drive_readonly", "gmail_send"])
    assert scopes[:2] == ["openid", "email"]
    assert "https://www.googleapis.com/auth/drive.readonly" in scopes
    assert "https://www.googleapis.com/auth/gmail.send" in scopes
    # no duplicates
    assert len(scopes) == len(set(scopes))


def test_unknown_provider_is_none():
    assert get_provider("dropbox") is None
    assert get_provider("") is None


def test_generate_pkce_shape():
    verifier, challenge = generate_pkce()
    assert 43 <= len(verifier) <= 128
    assert "=" not in challenge  # base64url, padding stripped


# ── Connection store: load / store / resolve / remove ────────────────────

def test_connection_loads_from_env(monkeypatch):
    blob = json.dumps({
        "provider": "google", "access_token": "at-1",
        "refresh_token": "rt-1", "expires_at": 9999999999,
        "scopes": ["s"], "account": "u@e.com",
    })
    monkeypatch.setenv(f"{CONNECTION_PREFIX}GOOGLE_DRIVE", blob)
    v = CredentialVault()
    assert "google_drive" in v.connections
    assert v.has_credential("google_drive")
    # sync resolve returns current access token (existence checks, etc.)
    assert v.resolve_credential("google_drive") == "at-1"
    # connections are agent-resolvable handles
    assert "google_drive" in v.list_agent_credential_names()


def test_connection_invalid_json_skipped(monkeypatch):
    monkeypatch.setenv(f"{CONNECTION_PREFIX}BROKEN", "{not json")
    monkeypatch.setenv(f"{CONNECTION_PREFIX}NOTOKEN", json.dumps({"provider": "google"}))
    v = CredentialVault()
    assert "broken" not in v.connections
    assert "notoken" not in v.connections


def test_store_connection_roundtrip(captured_env):
    v = CredentialVault()
    handle = v.store_connection("google_drive", {
        "provider": "google", "access_token": "at", "refresh_token": "rt",
        "expires_at": 9999999999, "scopes": [], "account": "a@b.com",
    })
    assert handle == "$CRED{google_drive}"
    assert f"{CONNECTION_PREFIX}GOOGLE_DRIVE" in captured_env
    assert v.connections["google_drive"]["access_token"] == "at"


def test_store_connection_rejects_bad_name(captured_env):
    v = CredentialVault()
    with pytest.raises(ValueError):
        v.store_connection("bad name!", {"access_token": "x"})
    with pytest.raises(ValueError):
        v.store_connection("google", {"refresh_token": "x"})  # no access_token


def test_list_connections_excludes_tokens(monkeypatch):
    monkeypatch.setenv(f"{CONNECTION_PREFIX}G", json.dumps({
        "provider": "google", "access_token": "secret-at",
        "refresh_token": "secret-rt", "expires_at": 123,
        "scopes": ["x"], "account": "u@e.com",
    }))
    v = CredentialVault()
    listed = v.list_connections()
    assert listed == [{
        "name": "g", "provider": "google", "account": "u@e.com",
        "scopes": ["x"], "expires_at": 123,
    }]
    flat = json.dumps(listed)
    assert "secret-at" not in flat and "secret-rt" not in flat


def test_remove_connection(captured_env, monkeypatch):
    monkeypatch.setenv(f"{CONNECTION_PREFIX}G", json.dumps({
        "provider": "google", "access_token": "at",
    }))
    v = CredentialVault()
    assert v.remove_credential("g") is True
    assert "g" not in v.connections
    assert v.remove_credential("g") is False  # idempotent


# ── Refresh-on-resolve (the load-bearing seam) ───────────────────────────

@pytest.mark.asyncio
async def test_resolve_async_no_refresh_when_fresh():
    post = AsyncMock()
    v = _vault_with_fake_http(post=post)
    v.connections["g"] = {
        "provider": "google", "access_token": "fresh",
        "refresh_token": "rt", "expires_at": int(time.time()) + 3600,
    }
    assert await v.resolve_credential_async("g") == "fresh"
    post.assert_not_awaited()  # no network when token is valid


@pytest.mark.asyncio
async def test_resolve_async_refreshes_when_expired(captured_env):
    post = AsyncMock(return_value=_fake_response(payload={
        "access_token": "new-at", "expires_in": 3600,
        # Google does NOT re-issue a refresh token on refresh
    }))
    v = _vault_with_fake_http(post=post)
    v.system_credentials["google_client_id"] = "cid"
    v.system_credentials["google_client_secret"] = "csec"
    v.connections["g"] = {
        "provider": "google", "access_token": "old-at",
        "refresh_token": "rt-keep", "expires_at": int(time.time()) - 10,
    }
    token = await v.resolve_credential_async("g")
    assert token == "new-at"
    post.assert_awaited_once()
    # refresh_token preserved (not blanked), expiry advanced, persisted
    assert v.connections["g"]["refresh_token"] == "rt-keep"
    assert v.connections["g"]["expires_at"] > int(time.time()) + 3000
    assert f"{CONNECTION_PREFIX}G" in captured_env


@pytest.mark.asyncio
async def test_resolve_async_no_refresh_token_returns_current():
    post = AsyncMock()
    v = _vault_with_fake_http(post=post)
    v.connections["g"] = {
        "provider": "google", "access_token": "stale",
        "refresh_token": "", "expires_at": int(time.time()) - 10,
    }
    # No refresh token: return current token, don't crash, don't call network
    assert await v.resolve_credential_async("g") == "stale"
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_async_refresh_http_error_raises(captured_env):
    post = AsyncMock(return_value=_fake_response(status=400, text="bad_grant"))
    v = _vault_with_fake_http(post=post)
    v.system_credentials["google_client_id"] = "cid"
    v.system_credentials["google_client_secret"] = "csec"
    v.connections["g"] = {
        "provider": "google", "access_token": "old",
        "refresh_token": "rt", "expires_at": int(time.time()) - 10,
    }
    with pytest.raises(RuntimeError):
        await v.resolve_credential_async("g")


@pytest.mark.asyncio
async def test_refresh_does_not_resurrect_disconnected_connection(captured_env):
    """If the operator disconnects mid-refresh, the refresh must not re-store it."""
    v = CredentialVault()
    v.system_credentials["google_client_id"] = "cid"
    v.system_credentials["google_client_secret"] = "csec"
    v.connections["g"] = {
        "provider": "google", "access_token": "old",
        "refresh_token": "rt", "expires_at": int(time.time()) - 10,
    }

    async def _post(*a, **k):
        v.connections.pop("g", None)  # simulate disconnect during refresh
        return _fake_response(payload={"access_token": "new-at", "expires_in": 3600})

    fake = AsyncMock()
    fake.is_closed = False
    fake.post = AsyncMock(side_effect=_post)
    v._http_client = fake

    token = await v.ensure_connection_token("g")
    assert token == "new-at"            # caller still gets the token
    assert "g" not in v.connections      # but it is NOT resurrected


@pytest.mark.asyncio
async def test_plain_credential_unaffected_by_async_resolve():
    v = _vault_with_fake_http(post=AsyncMock())
    v.credentials["github_token"] = "ghp_x"
    assert await v.resolve_credential_async("github_token") == "ghp_x"
    assert await v.resolve_credential_async("missing") is None


# ── Authorize URL + code exchange ────────────────────────────────────────

def test_has_oauth_client():
    v = CredentialVault()
    p = get_provider("google")
    assert v.has_oauth_client(p) is False
    v.system_credentials["google_client_id"] = "cid"
    v.system_credentials["google_client_secret"] = "csec"
    assert v.has_oauth_client(p) is True


def test_build_authorize_url():
    v = CredentialVault()
    v.system_credentials["google_client_id"] = "cid-123"
    p = get_provider("google")
    url = v.build_authorize_url(
        p, redirect_uri="https://x.test/dashboard/integrations/google/callback",
        state="st-abc", scopes=["openid", "email"], code_challenge="chal",
    )
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "client_id=cid-123" in url
    assert "state=st-abc" in url
    assert "code_challenge=chal" in url
    assert "code_challenge_method=S256" in url
    assert "access_type=offline" in url
    assert "scope=openid+email" in url


@pytest.mark.asyncio
async def test_exchange_oauth_code(captured_env):
    post = AsyncMock(return_value=_fake_response(payload={
        "access_token": "at-x", "refresh_token": "rt-x",
        "expires_in": 1800, "scope": "openid email scope.a",
    }))
    get = AsyncMock(return_value=_fake_response(payload={"email": "user@example.com"}))
    v = _vault_with_fake_http(post=post, get=get)
    v.system_credentials["google_client_id"] = "cid"
    v.system_credentials["google_client_secret"] = "csec"
    p = get_provider("google")
    conn = await v.exchange_oauth_code(
        p, code="auth-code", redirect_uri="https://x/cb", code_verifier="ver",
    )
    assert conn["provider"] == "google"
    assert conn["access_token"] == "at-x"
    assert conn["refresh_token"] == "rt-x"
    assert conn["scopes"] == ["openid", "email", "scope.a"]
    assert conn["account"] == "user@example.com"
    assert conn["expires_at"] > int(time.time()) + 1700
    # PKCE verifier forwarded in the token request body
    _, kwargs = post.call_args
    assert kwargs["data"]["code_verifier"] == "ver"
    assert kwargs["data"]["grant_type"] == "authorization_code"


@pytest.mark.asyncio
async def test_exchange_oauth_code_no_client_raises():
    v = _vault_with_fake_http(post=AsyncMock())
    p = get_provider("google")
    with pytest.raises(RuntimeError):
        await v.exchange_oauth_code(p, code="c", redirect_uri="r")


@pytest.mark.asyncio
async def test_exchange_account_fetch_failure_is_nonfatal(captured_env):
    post = AsyncMock(return_value=_fake_response(payload={
        "access_token": "at", "refresh_token": "rt", "expires_in": 3600,
    }))
    get = AsyncMock(side_effect=Exception("userinfo down"))
    v = _vault_with_fake_http(post=post, get=get)
    v.system_credentials["google_client_id"] = "cid"
    v.system_credentials["google_client_secret"] = "csec"
    conn = await v.exchange_oauth_code(
        get_provider("google"), code="c", redirect_uri="r",
    )
    assert conn["access_token"] == "at"
    assert conn["account"] == ""  # best-effort, swallowed


# ── End-to-end: the agent's http_request resolver path ───────────────────

@pytest.mark.asyncio
async def test_http_tool_resolves_connection_to_refreshed_bearer(captured_env):
    """Full agent path: http_tool._resolve_creds → mesh vault_resolve →
    vault.resolve_credential_async → ensure_connection_token (refresh) → bearer.

    Proves an agent's ``Authorization: Bearer $CRED{google_drive}`` ends up
    carrying a *freshly refreshed* token, never the stale one or the refresh
    token. The outbound HTTP call after this point is unchanged stock code.
    """
    from src.agent.builtins.http_tool import _resolve_creds

    vault = CredentialVault()
    vault.system_credentials["google_client_id"] = "cid"
    vault.system_credentials["google_client_secret"] = "csec"
    vault.connections["google_drive"] = {
        "provider": "google", "access_token": "OLD",
        "refresh_token": "rt", "expires_at": int(time.time()) - 10,  # expired
    }
    fake = AsyncMock()
    fake.is_closed = False
    fake.post = AsyncMock(return_value=_fake_response(
        payload={"access_token": "FRESH", "expires_in": 3600},
    ))
    vault._http_client = fake

    # mesh_client.vault_resolve mirrors POST /mesh/vault/resolve, which the mesh
    # endpoint backs with resolve_credential_async.
    async def _vault_resolve(name):
        return await vault.resolve_credential_async(name)

    mesh = MagicMock()
    mesh.vault_resolve = _vault_resolve

    resolved, secrets = await _resolve_creds("Bearer $CRED{google_drive}", mesh)
    assert resolved == "Bearer FRESH"   # refreshed token reached the header
    assert "FRESH" in secrets            # and is captured for redaction
    assert "rt" not in resolved          # refresh token never surfaces


# ── State store ──────────────────────────────────────────────────────────

def test_state_store_single_use_and_session_bound():
    s = OAuthStateStore(ttl_seconds=600)
    st = s.create(
        provider="google", connection_name="g", scopes=("a",),
        code_verifier="v", redirect_uri="r", session_hash="sess",
    )
    # wrong session rejected (and consumed)
    assert s.consume(st, session_hash="other") is None
    st2 = s.create(
        provider="google", connection_name="g2", scopes=("a",),
        code_verifier="v", redirect_uri="r", session_hash="sess",
    )
    got = s.consume(st2, session_hash="sess")
    assert got is not None and got.connection_name == "g2"
    # single-use: replay fails
    assert s.consume(st2, session_hash="sess") is None


def test_state_store_expiry():
    s = OAuthStateStore(ttl_seconds=600)
    st = s.create(
        provider="google", connection_name="g", scopes=("a",),
        code_verifier="v", redirect_uri="r", session_hash="sess",
        now=time.time() - 1000,
    )
    assert s.consume(st, session_hash="sess") is None
