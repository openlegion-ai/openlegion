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
    loop.tools = MagicMock()
    loop.tools.list_tools = MagicMock(return_value=[])
    loop.tools.get_tool_definitions = MagicMock(return_value=[])
    loop.tools.get_tool_sources = MagicMock(return_value={})
    loop.tools.execute = AsyncMock(return_value={"ok": True})
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
async def test_upload_route_allows_body_over_global_cap(tmp_path, monkeypatch):
    """Upload routes get the higher upload cap, not the 8 MiB global cap.

    A 9 MiB body (over the 8 MiB global cap, under the 64 MiB upload cap) must
    pass the body-size middleware and reach the route — never the global 413.
    Regression for the bodysize-vs-uploads conflict (H4 review).
    """
    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "warn")
    app, closers = _make_mesh_app(tmp_path)
    over_global = 9 * 1024 * 1024  # > 8 MiB global cap, < 64 MiB upload cap
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/upload-stage",
                content=b"x" * over_global,
                headers={
                    "content-type": "application/octet-stream",
                    "authorization": "Bearer operator-secret",
                },
            )
        # The middleware must not short-circuit this route with its global 413.
        # The route may itself reject (auth/validation), but never for body size
        # at 9 MiB — so any status other than the body-size 413 proves the
        # higher upload cap applied.
        assert resp.status_code != 413, (
            "upload route was pre-empted by the global body-size cap: "
            f"{resp.status_code} {resp.text}"
        )
    finally:
        for c in closers:
            c.close()


@pytest.mark.asyncio
async def test_upload_route_keeps_oom_backstop(tmp_path, monkeypatch):
    """Upload routes are NOT fully exempt — the streaming guard still caps them.

    With the upload cap lowered to 1 MiB, a 2 MiB body to an upload route must
    still be rejected with 413, proving the OOM backstop (a full exemption
    would let it through and let the buffering /dashboard upload route OOM).
    """
    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "warn")
    monkeypatch.setenv("OPENLEGION_MAX_UPLOAD_BODY_MB", "1")
    app, closers = _make_mesh_app(tmp_path)
    over_upload = 2 * 1024 * 1024  # > 1 MiB upload cap override
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser/upload-stage",
                content=b"x" * over_upload,
                headers={
                    "content-type": "application/octet-stream",
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


# ---------------------------------------------------------------------------
# _body_cap_for_path — route matching (exact upload route vs prefix subtree)
# ---------------------------------------------------------------------------

_GLOBAL_CAP = 8 * 1024 * 1024  # 8 MiB default


def test_body_cap_exact_match_upload_stage():
    """The fixed staging route gets the higher upload cap (exact match)."""
    from src.host.server import _body_cap_for_path
    assert _body_cap_for_path("/mesh/browser/upload-stage") > _GLOBAL_CAP


def test_body_cap_dashboard_uploads_prefix():
    """The dashboard uploads subtree (prefix) gets the higher upload cap."""
    from src.host.server import _body_cap_for_path
    assert _body_cap_for_path("/dashboard/api/uploads/x.png") > _GLOBAL_CAP


def test_body_cap_upload_stage_no_overmatch():
    """A path that merely starts with the staging route must NOT be treated as
    an upload route — exact match, not startswith (the original over-match bug)."""
    from src.host.server import _body_cap_for_path
    assert _body_cap_for_path("/mesh/browser/upload-stage-evil") == _GLOBAL_CAP


def test_body_cap_regular_route_uses_global_cap():
    """A normal route falls back to the 8 MiB global cap."""
    from src.host.server import _body_cap_for_path
    assert _body_cap_for_path("/mesh/agents") == _GLOBAL_CAP
