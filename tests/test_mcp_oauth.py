"""Phase 3 tests: OAuth discovery/DCR, the dynamic (blob-embedded)
refresh path, and the MCP connector connect/callback dance.

Discovery runs against ``httpx.MockTransport`` via the ``_new_client``
seam — no network. All test URLs use public-IP literals so the D16
resolved-IP check exercises for real without DNS.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.host.mcp_oauth as mcp_oauth
from src.host.connectors import ConnectorStore
from src.host.mcp_oauth import Discovery, MCPOAuthError, discover, register_client
from src.shared.types import ConnectorAuth, HttpConnector

MCP_URL = "https://93.184.216.34/mcp"
AS_ISSUER = "https://93.184.216.50"


def _mock_client(handler):
    def _factory():
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=False, timeout=5,
        )
    return _factory


def _json(payload, status=200, ctype="application/json"):
    return httpx.Response(
        status, content=json.dumps(payload).encode(),
        headers={"content-type": ctype},
    )


def _happy_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/.well-known/oauth-protected-resource/mcp":
        return _json({"authorization_servers": [AS_ISSUER]})
    if path == "/.well-known/oauth-authorization-server":
        return _json({
            "authorization_endpoint": f"{AS_ISSUER}/authorize",
            "token_endpoint": f"{AS_ISSUER}/token",
            "registration_endpoint": f"{AS_ISSUER}/register",
        })
    return httpx.Response(404)


# ── discovery ────────────────────────────────────────────────


class TestDiscovery:
    @pytest.mark.asyncio
    async def test_full_chain(self, monkeypatch):
        monkeypatch.setattr(
            mcp_oauth, "_new_client", _mock_client(_happy_handler),
        )
        d = await discover(MCP_URL)
        assert d.authorization_endpoint == f"{AS_ISSUER}/authorize"
        assert d.token_endpoint == f"{AS_ISSUER}/token"
        assert d.registration_endpoint == f"{AS_ISSUER}/register"

    @pytest.mark.asyncio
    async def test_root_well_known_fallback(self, monkeypatch):
        def handler(request):
            path = request.url.path
            if path == "/.well-known/oauth-protected-resource":
                return _json({"authorization_servers": [AS_ISSUER]})
            if path == "/.well-known/oauth-authorization-server":
                return _json({
                    "authorization_endpoint": f"{AS_ISSUER}/a",
                    "token_endpoint": f"{AS_ISSUER}/t",
                })
            return httpx.Response(404)
        monkeypatch.setattr(mcp_oauth, "_new_client", _mock_client(handler))
        d = await discover(MCP_URL)
        assert d.registration_endpoint is None

    @pytest.mark.asyncio
    async def test_no_metadata_names_the_step(self, monkeypatch):
        monkeypatch.setattr(
            mcp_oauth, "_new_client",
            _mock_client(lambda r: httpx.Response(404)),
        )
        with pytest.raises(MCPOAuthError) as ei:
            await discover(MCP_URL)
        assert ei.value.step == "protected-resource metadata"

    @pytest.mark.asyncio
    async def test_private_authorization_server_rejected(self, monkeypatch):
        # D16: the AS choice is SERVER-controlled — a metadata document
        # pointing the mesh at an internal address must die here.
        def handler(request):
            return _json({"authorization_servers": ["https://10.0.0.5"]})
        monkeypatch.setattr(mcp_oauth, "_new_client", _mock_client(handler))
        with pytest.raises(MCPOAuthError) as ei:
            await discover(MCP_URL)
        assert ei.value.step == "authorization server"

    @pytest.mark.asyncio
    async def test_loopback_carve_out_not_honored_for_discovered(
        self, monkeypatch,
    ):
        # B5: the explicit-loopback carve-out exists for OPERATOR-pasted
        # connector URLs; a remote server pointing its AS at 127.0.0.1
        # would aim the mesh's authenticated POSTs at local services.
        def handler(request):
            return _json({"authorization_servers": ["https://127.0.0.1:9443"]})
        monkeypatch.setattr(mcp_oauth, "_new_client", _mock_client(handler))
        with pytest.raises(MCPOAuthError) as ei:
            await discover(MCP_URL)
        assert "loopback" in str(ei.value)

    @pytest.mark.asyncio
    async def test_strict_proxy_status_falls_through_to_root(self, monkeypatch):
        # 403/405 on the path-aware well-known location must not abort
        # the fallback chain (strict proxies reject unknown paths).
        def handler(request):
            path = request.url.path
            if path == "/.well-known/oauth-protected-resource/mcp":
                return httpx.Response(403)
            if path == "/.well-known/oauth-protected-resource":
                return _json({"authorization_servers": [AS_ISSUER]})
            if path == "/.well-known/oauth-authorization-server":
                return _json({
                    "authorization_endpoint": f"{AS_ISSUER}/a",
                    "token_endpoint": f"{AS_ISSUER}/t",
                })
            return httpx.Response(404)
        monkeypatch.setattr(mcp_oauth, "_new_client", _mock_client(handler))
        d = await discover(MCP_URL)
        assert d.token_endpoint == f"{AS_ISSUER}/t"

    @pytest.mark.asyncio
    async def test_non_https_discovered_endpoint_rejected(self, monkeypatch):
        def handler(request):
            path = request.url.path
            if path.startswith("/.well-known/oauth-protected-resource"):
                return _json({"authorization_servers": [AS_ISSUER]})
            return _json({
                "authorization_endpoint": "http://93.184.216.50/authorize",
                "token_endpoint": f"{AS_ISSUER}/token",
            })
        monkeypatch.setattr(mcp_oauth, "_new_client", _mock_client(handler))
        with pytest.raises(MCPOAuthError) as ei:
            await discover(MCP_URL)
        assert "https" in str(ei.value)

    @pytest.mark.asyncio
    async def test_oversize_and_wrong_ctype_rejected(self, monkeypatch):
        def big(request):
            return httpx.Response(
                200, content=b"x" * (64 * 1024 + 1),
                headers={"content-type": "application/json"},
            )
        monkeypatch.setattr(mcp_oauth, "_new_client", _mock_client(big))
        with pytest.raises(MCPOAuthError):
            await discover(MCP_URL)

        def html(request):
            return _json({"authorization_servers": [AS_ISSUER]}, ctype="text/html")
        monkeypatch.setattr(mcp_oauth, "_new_client", _mock_client(html))
        with pytest.raises(MCPOAuthError):
            await discover(MCP_URL)

    @pytest.mark.asyncio
    async def test_redirects_not_followed(self, monkeypatch):
        def handler(request):
            return httpx.Response(
                302, headers={"location": "https://93.184.216.99/elsewhere"},
            )
        monkeypatch.setattr(mcp_oauth, "_new_client", _mock_client(handler))
        with pytest.raises(MCPOAuthError) as ei:
            await discover(MCP_URL)
        assert "redirect" in str(ei.value)


class TestRegisterClient:
    @pytest.mark.asyncio
    async def test_confidential_client(self, monkeypatch):
        def handler(request):
            body = json.loads(request.content)
            assert body["token_endpoint_auth_method"] == "client_secret_post"
            return _json({"client_id": "cid", "client_secret": "sec"}, status=201)
        monkeypatch.setattr(mcp_oauth, "_new_client", _mock_client(handler))
        assert await register_client(f"{AS_ISSUER}/register", "https://x/cb") == (
            "cid", "sec",
        )

    @pytest.mark.asyncio
    async def test_public_client_fallback(self, monkeypatch):
        calls = []

        def handler(request):
            body = json.loads(request.content)
            calls.append(body["token_endpoint_auth_method"])
            if body["token_endpoint_auth_method"] == "client_secret_post":
                return httpx.Response(400)
            return _json({"client_id": "pub"})
        monkeypatch.setattr(mcp_oauth, "_new_client", _mock_client(handler))
        assert await register_client(f"{AS_ISSUER}/register", "https://x/cb") == (
            "pub", None,
        )
        assert calls == ["client_secret_post", "none"]

    @pytest.mark.asyncio
    async def test_no_client_id_is_error(self, monkeypatch):
        monkeypatch.setattr(
            mcp_oauth, "_new_client", _mock_client(lambda r: _json({})),
        )
        with pytest.raises(MCPOAuthError) as ei:
            await register_client(f"{AS_ISSUER}/register", "https://x/cb")
        assert ei.value.step == "client registration"


# ── vault: dynamic exchange + blob refresh ───────────────────


def _bare_vault(monkeypatch, handler):
    """CredentialVault skeleton with just the attrs the dynamic paths
    touch — constructing the real thing drags in env/persistence."""
    from src.host.credentials import CredentialVault
    vault = CredentialVault.__new__(CredentialVault)
    vault.connections = {}
    vault.system_credentials = {}
    vault._connection_refresh_failures = {}
    vault._connection_locks = {}
    vault._connection_locks_guard = asyncio.Lock()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=5,
    )
    vault._get_http_client = AsyncMock(return_value=client)
    monkeypatch.setattr(
        "src.host.credentials._persist_to_env", lambda *a, **k: None,
    )
    return vault


class TestDynamicExchange:
    @pytest.mark.asyncio
    async def test_public_client_exchange_embeds_blob(self, monkeypatch):
        seen = {}

        def handler(request):
            seen.update(dict(httpx.QueryParams(request.content.decode())))
            return _json({
                "access_token": "at", "refresh_token": "rt",
                "expires_in": 600, "scope": "mcp",
            })
        vault = _bare_vault(monkeypatch, handler)
        conn = await vault.exchange_code_dynamic(
            token_endpoint=f"{AS_ISSUER}/token", client_id="pub",
            code="c0de", redirect_uri="https://x/cb", code_verifier="ver",
            resource=MCP_URL, provider_label="mcp:linear",
        )
        # Public client: NO client_secret in the form body; PKCE + resource present.
        assert "client_secret" not in seen
        assert seen["code_verifier"] == "ver"
        assert seen["resource"] == MCP_URL
        # Blob embeds the refresh context (no registry entry exists).
        assert conn["token_endpoint"] == f"{AS_ISSUER}/token"
        assert conn["client_id"] == "pub"
        assert conn["uses_pkce"] is True
        assert conn["refresh_token"] == "rt"

    @pytest.mark.asyncio
    async def test_exchange_revalidates_endpoint(self, monkeypatch):
        # B4: the discovered endpoint is validated at the POINT OF USE,
        # not just at discovery — a private/loopback target must never
        # receive the code-exchange POST.
        vault = _bare_vault(monkeypatch, lambda r: _json({"access_token": "x"}))
        for bad in ("https://10.0.0.5/token", "https://127.0.0.1/token",
                    "http://93.184.216.50/token"):
            with pytest.raises(RuntimeError):
                await vault.exchange_code_dynamic(
                    token_endpoint=bad, client_id="pub", code="c",
                    redirect_uri="https://x/cb", code_verifier="v",
                )

    @pytest.mark.asyncio
    async def test_refresh_revalidates_blob_endpoint(self, monkeypatch):
        # B4: a tampered/repointed blob endpoint must not receive
        # refresh_token POSTs on expiry.
        vault = _bare_vault(monkeypatch, lambda r: _json({"access_token": "x"}))
        vault.connections["mcp_evil"] = {
            "provider": "mcp:evil", "access_token": "stale",
            "refresh_token": "rt", "expires_at": 100,
            "token_endpoint": "https://127.0.0.1:9443/token",
            "client_id": "pub", "client_secret": "",
        }
        with pytest.raises(RuntimeError):
            await vault.ensure_connection_token("mcp_evil")

    @pytest.mark.asyncio
    async def test_failed_exchange_raises_truncated(self, monkeypatch):
        vault = _bare_vault(
            monkeypatch, lambda r: httpx.Response(400, content=b"denied " * 200),
        )
        with pytest.raises(RuntimeError) as ei:
            await vault.exchange_code_dynamic(
                token_endpoint=f"{AS_ISSUER}/token", client_id="pub",
                code="x", redirect_uri="https://x/cb", code_verifier="v",
            )
        assert len(str(ei.value)) < 400


class TestBlobRefresh:
    def _expired_conn(self, **extra):
        return {
            "provider": "mcp:linear", "access_token": "stale",
            "refresh_token": "rt0", "expires_at": 100,  # long expired
            **extra,
        }

    @pytest.mark.asyncio
    async def test_refreshes_against_blob_endpoint(self, monkeypatch):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen.update(dict(httpx.QueryParams(request.content.decode())))
            return _json({
                "access_token": "fresh", "refresh_token": "rt1",
                "expires_in": 3600,
            })
        vault = _bare_vault(monkeypatch, handler)
        vault.connections["mcp_linear"] = self._expired_conn(
            token_endpoint=f"{AS_ISSUER}/token", client_id="pub",
            client_secret="", resource=MCP_URL,
        )
        token = await vault.ensure_connection_token("mcp_linear")
        assert token == "fresh"
        assert seen["url"] == f"{AS_ISSUER}/token"
        assert "client_secret" not in seen  # public client
        assert seen["resource"] == MCP_URL  # RFC 8707 on refresh too
        # Rotated refresh token persisted (OAuth 2.1 rotation).
        assert vault.connections["mcp_linear"]["refresh_token"] == "rt1"

    @pytest.mark.asyncio
    async def test_legacy_registry_miss_still_returns_stale(self, monkeypatch):
        # Pre-Phase-3 behavior preserved for connections with neither a
        # blob endpoint nor a registry provider: stale token, no crash.
        vault = _bare_vault(monkeypatch, lambda r: httpx.Response(500))
        vault.connections["odd"] = self._expired_conn(provider="not-a-provider")
        assert await vault.ensure_connection_token("odd") == "stale"

    @pytest.mark.asyncio
    async def test_hard_rejection_cached_with_dynamic_label(self, monkeypatch):
        from src.host.credentials import ConnectionRefreshError
        vault = _bare_vault(
            monkeypatch, lambda r: httpx.Response(400, content=b"invalid_grant"),
        )
        vault.connections["mcp_linear"] = self._expired_conn(
            token_endpoint=f"{AS_ISSUER}/token", client_id="pub",
        )
        with pytest.raises(ConnectionRefreshError) as ei:
            await vault.ensure_connection_token("mcp_linear")
        assert ei.value.provider == "mcp:linear"
        assert ei.value.first_failure is True


# ── state store extra ────────────────────────────────────────


class TestStateExtra:
    def test_extra_round_trips_and_single_use(self):
        from src.host.oauth_state import OAuthStateStore
        store = OAuthStateStore()
        state = store.create(
            provider="mcp:linear", connection_name="mcp_linear",
            scopes=(), code_verifier="v", redirect_uri="https://x/cb",
            session_hash="h",
            extra={"token_endpoint": "https://as/token", "client_id": "cid"},
        )
        pending = store.consume(state, session_hash="h")
        assert pending.extra["client_id"] == "cid"
        assert store.consume(state, session_hash="h") is None  # replay


# ── connect/callback routes ──────────────────────────────────


class _CSRFClient(TestClient):
    def request(self, method, url, **kwargs):
        if method.upper() not in ("GET", "HEAD", "OPTIONS"):
            headers = kwargs.get("headers") or {}
            headers.setdefault("X-Requested-With", "XMLHttpRequest")
            kwargs["headers"] = headers
        return super().request(method, url, **kwargs)


@pytest.fixture
def oauth_dash_env(tmp_path, monkeypatch):
    from src.dashboard.events import EventBus
    from src.dashboard.server import create_dashboard_router
    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard
    from src.host.traces import TraceStore

    store = ConnectorStore(str(tmp_path / "connectors.json"))
    store.upsert(HttpConnector(
        transport="http", name="linear", url=MCP_URL,
        auth=ConnectorAuth(kind="oauth"), agents=["alpha"],
    ))
    gateway = MagicMock()
    vault = MagicMock()
    vault.resolve_credential.return_value = "x"
    vault.list_agent_credential_names.return_value = []
    vault.exchange_code_dynamic = AsyncMock(return_value={
        "provider": "mcp:linear", "access_token": "at",
        "refresh_token": "rt", "expires_at": 9999999999,
        "scopes": [], "account": "",
        "token_endpoint": f"{AS_ISSUER}/token", "client_id": "cid",
        "client_secret": "", "uses_pkce": True, "resource": MCP_URL,
    })
    monkeypatch.setattr(
        "src.host.mcp_oauth.discover",
        AsyncMock(return_value=Discovery(
            authorization_server=AS_ISSUER,
            authorization_endpoint=f"{AS_ISSUER}/authorize",
            token_endpoint=f"{AS_ISSUER}/token",
            registration_endpoint=f"{AS_ISSUER}/register",
        )),
    )
    monkeypatch.setattr(
        "src.host.mcp_oauth.register_client",
        AsyncMock(return_value=("cid", None)),
    )
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    router = create_dashboard_router(
        blackboard=bb,
        health_monitor=None,
        cost_tracker=CostTracker(db_path=str(tmp_path / "costs.db")),
        trace_store=TraceStore(db_path=str(tmp_path / "traces.db")),
        event_bus=EventBus(),
        agent_registry={"alpha": "http://x:1"},
        permissions=MagicMock(),
        credential_vault=vault,
        connector_store=store,
        mcp_gateway=gateway,
    )
    app = FastAPI()
    app.include_router(router)
    client = _CSRFClient(app)
    yield client, store, gateway, vault
    client.close()
    bb.close()


class TestConnectCallback:
    def _connect(self, client):
        resp = client.get(
            "/dashboard/integrations/mcp/linear/connect",
            follow_redirects=False,
        )
        assert resp.status_code == 302, resp.text
        location = httpx.URL(resp.headers["location"])
        assert str(location).startswith(f"{AS_ISSUER}/authorize")
        return dict(location.params)

    def test_connect_builds_pkce_authorize_url(self, oauth_dash_env):
        client, *_ = oauth_dash_env
        params = self._connect(client)
        assert params["client_id"] == "cid"
        assert params["code_challenge_method"] == "S256"
        assert params["resource"] == MCP_URL
        assert params["state"]

    def test_callback_first_bind_marks_pending_restart(self, oauth_dash_env):
        client, store, gateway, vault = oauth_dash_env
        # Simulate "agent already running current catalog".
        _, gen = store.snapshot_for_agent("alpha")
        store.record_agent_start("alpha", gen)
        params = self._connect(client)
        resp = client.get(
            "/dashboard/integrations/mcp/linear/callback",
            params={"code": "c0de", "state": params["state"]},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "integration_connected=mcp_linear" in resp.headers["location"]
        vault.store_connection.assert_called_once()
        bound = store.get("linear")
        assert bound.auth.kind == "oauth"
        assert bound.auth.connection == "mcp_linear"
        # FIRST bind: the agent registered zero tools at its last boot
        # (the connector 401'd), so it must be marked pending-restart —
        # nothing else would ever prompt the bounce that registers the
        # now-reachable tools.
        assert store.pending_restart() == ["alpha"]
        gateway.invalidate.assert_called_with("linear")
        # The discovered endpoint + DCR identity went to the exchange.
        kwargs = vault.exchange_code_dynamic.await_args.kwargs
        assert kwargs["token_endpoint"] == f"{AS_ISSUER}/token"
        assert kwargs["client_id"] == "cid"
        assert kwargs["client_secret"] is None
        assert kwargs["resource"] == MCP_URL

    def test_callback_reconnect_is_rotation_no_restart(self, oauth_dash_env):
        client, store, gateway, vault = oauth_dash_env
        # First bind.
        params = self._connect(client)
        client.get(
            "/dashboard/integrations/mcp/linear/callback",
            params={"code": "c0de", "state": params["state"]},
            follow_redirects=False,
        )
        # Agent restarts onto the bound catalog.
        _, gen = store.snapshot_for_agent("alpha")
        store.record_agent_start("alpha", gen)
        assert store.pending_restart() == []
        # RE-connect (already bound): rotation semantics — no new
        # pending-restart, the fresh token applies per call.
        params = self._connect(client)
        resp = client.get(
            "/dashboard/integrations/mcp/linear/callback",
            params={"code": "c0de", "state": params["state"]},
            follow_redirects=False,
        )
        assert "integration_connected" in resp.headers["location"]
        assert store.pending_restart() == []

    def test_callback_state_replay_rejected(self, oauth_dash_env):
        client, *_ = oauth_dash_env
        params = self._connect(client)
        first = client.get(
            "/dashboard/integrations/mcp/linear/callback",
            params={"code": "c", "state": params["state"]},
            follow_redirects=False,
        )
        assert "integration_connected" in first.headers["location"]
        replay = client.get(
            "/dashboard/integrations/mcp/linear/callback",
            params={"code": "c", "state": params["state"]},
            follow_redirects=False,
        )
        assert "integration_error=invalid_state" in replay.headers["location"]

    def test_connect_without_dcr_is_502_naming_step(
        self, oauth_dash_env, monkeypatch,
    ):
        client, *_ = oauth_dash_env
        monkeypatch.setattr(
            "src.host.mcp_oauth.discover",
            AsyncMock(return_value=Discovery(
                authorization_server=AS_ISSUER,
                authorization_endpoint=f"{AS_ISSUER}/authorize",
                token_endpoint=f"{AS_ISSUER}/token",
                registration_endpoint=None,
            )),
        )
        resp = client.get(
            "/dashboard/integrations/mcp/linear/connect",
            follow_redirects=False,
        )
        assert resp.status_code == 502
        assert "client registration" in resp.json()["detail"]

    def test_connect_unknown_connector_404(self, oauth_dash_env):
        client, *_ = oauth_dash_env
        resp = client.get(
            "/dashboard/integrations/mcp/ghost/connect",
            follow_redirects=False,
        )
        assert resp.status_code == 404
