"""Mesh→agent bearer auth on the agent server (audit C1 second leg, B7).

The mesh already mints a per-agent token (``runtime.auth_tokens``) that the
container uses for the agent→mesh direction as ``MESH_AUTH_TOKEN``. This
change closes the loop: mesh→agent requests carry ``Authorization: Bearer
<that agent's token>`` and the agent server verifies it against its own
``MESH_AUTH_TOKEN`` env. Enforcement and caller wiring land in the SAME
change (B7): the Transport attaches the token per target agent, the mesh's
direct httpx call sites use ``_agent_bearer_headers``, and the detached CLI
fetches the token from the loopback-internal ``/mesh/agents/{id}/token``
endpoint. ``GET /status`` stays tokenless so reachability probes keep
working.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.agent.server import create_agent_app

TOKEN = "mesh-agent-token-123"


def _make_agent_app():
    """Minimal agent app harness — mirrors tests/test_protocol_version.py."""
    loop = MagicMock()
    loop.agent_id = "test_agent"
    loop.role = "researcher"
    loop.state = "idle"
    loop._excluded_tools = frozenset()
    loop.memory = None
    loop.mesh_client = MagicMock()
    loop.tools = MagicMock()
    loop.tools.list_tools = MagicMock(return_value=[])
    loop.tools.get_tool_definitions = MagicMock(return_value=[])
    loop.tools.get_tool_sources = MagicMock(return_value={})
    loop.tools.execute = AsyncMock(return_value={"ok": True})
    loop.workspace = None
    loop.get_status.return_value.model_dump.return_value = {"state": "idle"}
    # /chat/reset awaits this — needed by the POST-requires-bearer test.
    loop.reset_chat = AsyncMock()
    return create_agent_app(loop)


# ── agent server guard ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_correct_bearer_accepted(monkeypatch):
    monkeypatch.setenv("MESH_AUTH_TOKEN", TOKEN)
    app = _make_agent_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        resp = await client.get(
            "/capabilities", headers={"Authorization": f"Bearer {TOKEN}"},
        )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_missing_bearer_rejected_with_diagnosable_body(monkeypatch):
    monkeypatch.setenv("MESH_AUTH_TOKEN", TOKEN)
    app = _make_agent_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        resp = await client.get("/capabilities")
    assert resp.status_code == 401
    body = resp.json()
    # Operators must be able to diagnose: names the agent + the direction.
    assert body["agent"] == "test_agent"
    assert "bearer token" in body["detail"].lower()
    assert "test_agent" in body["detail"]
    assert resp.headers.get("www-authenticate") == "Bearer"


@pytest.mark.asyncio
async def test_wrong_bearer_rejected(monkeypatch):
    monkeypatch.setenv("MESH_AUTH_TOKEN", TOKEN)
    app = _make_agent_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        resp = await client.get(
            "/capabilities", headers={"Authorization": "Bearer wrong-token"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_non_bearer_scheme_rejected(monkeypatch):
    monkeypatch.setenv("MESH_AUTH_TOKEN", TOKEN)
    app = _make_agent_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        resp = await client.get(
            "/capabilities", headers={"Authorization": f"Basic {TOKEN}"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_token_unset_fails_open(monkeypatch):
    """No MESH_AUTH_TOKEN env → no enforcement (dev harnesses, tests)."""
    monkeypatch.delenv("MESH_AUTH_TOKEN", raising=False)
    app = _make_agent_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        resp = await client.get("/capabilities")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_get_status_exempt_even_with_token(monkeypatch):
    """GET /status is the reachability probe — must stay tokenless (B7)."""
    monkeypatch.setenv("MESH_AUTH_TOKEN", TOKEN)
    app = _make_agent_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        resp = await client.get("/status")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_post_requires_bearer(monkeypatch):
    """The exemption is GET /status only; state-mutating verbs are gated."""
    monkeypatch.setenv("MESH_AUTH_TOKEN", TOKEN)
    app = _make_agent_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        denied = await client.post("/chat/reset")
        allowed = await client.post(
            "/chat/reset", headers={"Authorization": f"Bearer {TOKEN}"},
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200


# ── transport wiring (mesh side) ────────────────────────────────────────────


def test_transport_registered_token_attached():
    from src.host.transport import HttpTransport

    t = HttpTransport()
    t.register_token("alpha", "tok-alpha")
    headers = t._resolve_headers(None, agent_id="alpha")
    assert headers["Authorization"] == "Bearer tok-alpha"
    # Existing behavior unchanged.
    assert headers["x-mesh-internal"] == "1"


def test_transport_unknown_agent_no_auth_header():
    from src.host.transport import HttpTransport

    t = HttpTransport()
    t.register_token("alpha", "tok-alpha")
    headers = t._resolve_headers(None, agent_id="beta")
    assert "Authorization" not in headers
    # No agent_id at all (legacy callers) → also no header.
    assert "Authorization" not in t._resolve_headers(None)


def test_transport_bind_tokens_live_mapping_sees_rotation():
    """bind_tokens holds the mapping by reference — restart-rotated tokens
    (runtime.start_agent re-mints into the same dict) are picked up without
    any re-registration call."""
    from src.host.transport import HttpTransport

    live: dict[str, str] = {}
    t = HttpTransport()
    t.bind_tokens(live)
    assert "Authorization" not in t._resolve_headers(None, agent_id="alpha")
    live["alpha"] = "tok-after-restart"
    headers = t._resolve_headers(None, agent_id="alpha")
    assert headers["Authorization"] == "Bearer tok-after-restart"


def test_transport_caller_supplied_authorization_not_clobbered():
    from src.host.transport import HttpTransport

    t = HttpTransport()
    t.register_token("alpha", "tok-alpha")
    headers = t._resolve_headers({"Authorization": "Bearer custom"}, agent_id="alpha")
    assert headers["Authorization"] == "Bearer custom"


def test_sandbox_transport_has_token_storage():
    """SandboxTransport inherits the base token machinery (curl -H path)."""
    from src.host.transport import SandboxTransport

    t = SandboxTransport()
    t.register_token("alpha", "tok-a")
    assert t._resolve_headers(None, agent_id="alpha")["Authorization"] == "Bearer tok-a"


# ── mesh token-disclosure endpoint (detached CLI wiring) ────────────────────


def _build_mesh_app(tmp_path, auth_tokens):
    """Mirrors the harness in tests/test_agent_registration.py."""
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    perms = PermissionMatrix()
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    router = MessageRouter(perms, {})
    app = create_mesh_app(
        blackboard=bb,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        auth_tokens=auth_tokens,
    )
    return app, router


@pytest.mark.asyncio
async def test_token_endpoint_loopback_internal_gets_token(tmp_path):
    app, _router = _build_mesh_app(tmp_path, {"scout": "tok-scout"})
    transport = ASGITransport(app=app, client=("127.0.0.1", 40000))
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get(
            "/mesh/agents/scout/token", headers={"x-mesh-internal": "1"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"agent_id": "scout", "token": "tok-scout"}


@pytest.mark.asyncio
async def test_token_endpoint_rejects_non_loopback(tmp_path):
    """Header alone is forgeable — a bridge-network agent must get 403."""
    app, _router = _build_mesh_app(tmp_path, {"scout": "tok-scout"})
    transport = ASGITransport(app=app, client=("172.18.0.5", 40000))
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get(
            "/mesh/agents/scout/token", headers={"x-mesh-internal": "1"},
        )
    assert resp.status_code == 403


def test_token_endpoint_rejects_missing_internal_header(tmp_path):
    app, _router = _build_mesh_app(tmp_path, {"scout": "tok-scout"})
    client = TestClient(app)
    resp = client.get("/mesh/agents/scout/token")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_token_endpoint_unknown_agent_404(tmp_path):
    app, _router = _build_mesh_app(tmp_path, {"scout": "tok-scout"})
    transport = ASGITransport(app=app, client=("127.0.0.1", 40000))
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get(
            "/mesh/agents/nobody/token", headers={"x-mesh-internal": "1"},
        )
    assert resp.status_code == 404


# ── mesh direct-httpx caller (history fallback → _agent_bearer_headers) ────


@pytest.mark.asyncio
async def test_history_fallback_sends_bearer(tmp_path, monkeypatch):
    """The transport-less /history fallback attaches the agent's bearer.

    Exercises the ``_agent_bearer_headers`` closure that the INTERFACE.md
    fallback and the artifacts/ingest stream also use.
    """
    import httpx as httpx_mod

    app, router = _build_mesh_app(tmp_path, {"scout": "tok-scout"})
    router.register_agent("scout", "http://127.0.0.1:8401")

    captured: dict = {}

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"history": []}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None, headers=None):
            captured["url"] = url
            captured["headers"] = headers or {}
            return _FakeResp()

    monkeypatch.setattr(httpx_mod, "AsyncClient", _FakeClient)
    transport = ASGITransport(app=app, client=("127.0.0.1", 40000))
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get(
            "/mesh/agents/scout/history", headers={"x-mesh-internal": "1"},
        )
    assert resp.status_code == 200, resp.text
    assert captured["headers"].get("Authorization") == "Bearer tok-scout"


# ── detached CLI wiring ─────────────────────────────────────────────────────


def test_cli_agent_auth_headers_fetches_token(monkeypatch):
    import httpx as httpx_mod

    from src.cli.main import _agent_auth_headers

    calls: dict = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"agent_id": "scout", "token": "tok-scout"}

    def _fake_get(url, headers=None, timeout=None):
        calls["url"] = url
        calls["headers"] = headers
        return _Resp()

    monkeypatch.setattr(httpx_mod, "get", _fake_get)
    headers = _agent_auth_headers("scout", 8420)
    assert headers == {"Authorization": "Bearer tok-scout"}
    assert calls["url"].endswith("/mesh/agents/scout/token")
    assert calls["headers"] == {"x-mesh-internal": "1"}


def test_cli_agent_auth_headers_tokenless_mesh_is_empty(monkeypatch):
    """Older/tokenless mesh (404 or connection error) → empty headers, no crash."""
    import httpx as httpx_mod

    from src.cli.main import _agent_auth_headers

    class _Resp:
        status_code = 404

        def json(self):  # pragma: no cover - not reached on 404
            return {}

    monkeypatch.setattr(httpx_mod, "get", lambda *a, **k: _Resp())
    assert _agent_auth_headers("scout", 8420) == {}

    def _boom(*a, **k):
        raise httpx_mod.ConnectError("down")

    monkeypatch.setattr(httpx_mod, "get", _boom)
    assert _agent_auth_headers("scout", 8420) == {}


def test_cli_sync_detached_chat_sends_bearer(monkeypatch):
    import httpx as httpx_mod

    from src.cli.main import _sync_detached_chat

    captured: dict = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"response": "hi", "tool_calls_made": 0}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _Resp()

    monkeypatch.setattr(httpx_mod, "post", _fake_post)
    _sync_detached_chat(
        "scout", "http://127.0.0.1:8401", "hello",
        headers={"Authorization": "Bearer tok-scout"},
    )
    assert captured["url"] == "http://127.0.0.1:8401/chat"
    assert captured["headers"] == {"Authorization": "Bearer tok-scout"}
