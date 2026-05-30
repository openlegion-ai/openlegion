"""Request body-size limit middleware (finding H4).

The mesh and agent FastAPI apps buffer the full HTTP body into memory before
the JSON parser runs. Without a cap, an authenticated agent could POST a
multi-GB body and OOM the single coordination / agent process — a fleet-wide
DoS. The middleware rejects over-cap requests with HTTP 413, via both a
Content-Length header check and a streaming byte counter (so a chunked /
Content-Length-absent body can't bypass it).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.agent.server import create_agent_app

# ---------------------------------------------------------------------------
# Agent app
# ---------------------------------------------------------------------------

def _make_agent_app():
    loop = MagicMock()
    loop.agent_id = "test_agent"
    loop.role = "researcher"
    loop.state = "idle"
    loop._excluded_tools = frozenset()
    loop.memory = None
    loop.mesh_client = MagicMock()
    loop.skills = MagicMock()
    loop.skills.list_skills = MagicMock(return_value=[])
    loop.skills.get_tool_definitions = MagicMock(return_value=[])
    loop.skills.get_tool_sources = MagicMock(return_value={})
    loop.skills.execute = AsyncMock(return_value={"ok": True})
    loop.workspace = None
    return create_agent_app(loop)


@pytest.mark.asyncio
async def test_agent_rejects_oversized_content_length():
    """A POST declaring a Content-Length over the cap → HTTP 413."""
    app = _make_agent_app()
    over = 9 * 1024 * 1024  # 9 MiB > 8 MiB default cap
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/task",
            content=b"x" * over,
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 413, resp.text
    assert "too large" in resp.text.lower()


@pytest.mark.asyncio
async def test_agent_small_request_succeeds():
    """A normal small request passes the middleware and reaches routing."""
    app = _make_agent_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        # GET /workspace returns {"files": []} when workspace is None — a
        # trivial endpoint that doesn't depend on mock loop internals.
        resp = await client.get("/workspace")
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_agent_rejects_oversized_chunked_body():
    """A chunked body (no Content-Length) over the cap → HTTP 413.

    httpx sends a generator-sourced body using chunked transfer encoding
    (no Content-Length header), so this exercises the streaming guard, not
    the header check.
    """
    app = _make_agent_app()
    over = 9 * 1024 * 1024

    async def _gen():
        sent = 0
        chunk = b"x" * (256 * 1024)
        while sent < over:
            yield chunk
            sent += len(chunk)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/task",
            content=_gen(),
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 413, resp.text
    assert "too large" in resp.text.lower()


# ---------------------------------------------------------------------------
# Mesh app
# ---------------------------------------------------------------------------

def _make_mesh_app(tmp_path):
    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    router.register_agent("operator", "http://operator:8400", [])
    auth_tokens = {"operator": "operator-secret"}
    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces, auth_tokens=auth_tokens,
    )
    return app, (bb, costs, traces)


@pytest.mark.asyncio
async def test_mesh_rejects_oversized_content_length(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "warn")
    app, closers = _make_mesh_app(tmp_path)
    over = 9 * 1024 * 1024
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/agents/register",
                content=b"x" * over,
                headers={
                    "content-type": "application/json",
                    "authorization": "Bearer operator-secret",
                },
            )
        assert resp.status_code == 413, resp.text
        assert "too large" in resp.text.lower()
    finally:
        for c in closers:
            c.close()


@pytest.mark.asyncio
async def test_mesh_small_request_succeeds(tmp_path, monkeypatch):
    """A normal small request passes the middleware (reaches auth/routing)."""
    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "warn")
    app, closers = _make_mesh_app(tmp_path)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            # Health-check endpoint hit by the provisioner — no body, must 200.
            resp = await client.get(
                "/mesh/agents", headers={"x-mesh-internal": "1"},
            )
        # The middleware must not interfere: anything other than 413/400 from
        # the body guard means it passed through to routing.
        assert resp.status_code == 200, resp.text
    finally:
        for c in closers:
            c.close()


@pytest.mark.asyncio
async def test_mesh_rejects_oversized_chunked_body(tmp_path, monkeypatch):
    """A chunked over-cap body to the mesh → HTTP 413 via the streaming guard."""
    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "warn")
    app, closers = _make_mesh_app(tmp_path)
    over = 9 * 1024 * 1024

    async def _gen():
        sent = 0
        chunk = b"x" * (256 * 1024)
        while sent < over:
            yield chunk
            sent += len(chunk)

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/agents/register",
                content=_gen(),
                headers={
                    "content-type": "application/json",
                    "authorization": "Bearer operator-secret",
                },
            )
        assert resp.status_code == 413, resp.text
        assert "too large" in resp.text.lower()
    finally:
        for c in closers:
            c.close()


@pytest.mark.asyncio
async def test_env_override_lowers_cap(tmp_path, monkeypatch):
    """OPENLEGION_MAX_BODY_MB shrinks the cap; a sub-8-MiB body now 413s."""
    monkeypatch.setenv("OPENLEGION_MAX_BODY_MB", "1")
    app = _make_agent_app()
    over = 2 * 1024 * 1024  # 2 MiB > 1 MiB override
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/task",
            content=b"x" * over,
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 413, resp.text
