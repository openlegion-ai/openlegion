"""Tests for the operator read_peer_artifact / list_peer_artifacts skills.

These two tools close operator bug 6: peer artifacts live on each
agent's private /data volume, so save_artifact mirrors metadata to the
blackboard but never the content. The dashboard already exposes a
peer-read path via ``transport.request``; these tools make the same
affordance available to the operator agent (and only the operator).
Mirrors the read_agent_config test shape from PR 898.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    """The skills require ALLOWED_TOOLS to be set (defence-in-depth)."""
    monkeypatch.setenv(
        "ALLOWED_TOOLS",
        "read_peer_artifact,list_peer_artifacts,read_agent_config",
    )


# ── list_peer_artifacts ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_peer_artifacts_happy_path():
    """Happy path returns {agent_id, artifacts: [...]} unchanged."""
    from src.agent.builtins.operator_tools import list_peer_artifacts

    payload = {
        "agent_id": "alpha",
        "artifacts": [
            {"name": "design.md", "size": 1024, "modified": 1700000000.0},
            {"name": "reports/q3.pdf", "size": 50000, "modified": 1700000100.0},
        ],
    }
    mc = MagicMock()
    mc.list_peer_artifacts = AsyncMock(return_value=payload)

    result = await list_peer_artifacts("alpha", mesh_client=mc)

    assert result == payload
    mc.list_peer_artifacts.assert_awaited_once_with("alpha")


@pytest.mark.asyncio
async def test_list_peer_artifacts_blocked_for_non_operator(monkeypatch):
    """Non-operator agents (no ALLOWED_TOOLS) are denied at the skill layer."""
    from src.agent.builtins.operator_tools import list_peer_artifacts

    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    mc = MagicMock()
    mc.list_peer_artifacts = AsyncMock()

    result = await list_peer_artifacts("alpha", mesh_client=mc)

    assert "error" in result
    assert "operator" in result["error"].lower()
    mc.list_peer_artifacts.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_peer_artifacts_agent_not_found():
    """A 404 from the mesh becomes {error: agent_not_found}."""
    from src.agent.builtins.operator_tools import list_peer_artifacts

    fake_response = MagicMock()
    fake_response.status_code = 404
    fake_response.text = "Agent 'ghost' not found"

    class FakeHTTPError(Exception):
        def __init__(self):
            super().__init__("404 Not Found")
            self.response = fake_response

    mc = MagicMock()
    mc.list_peer_artifacts = AsyncMock(side_effect=FakeHTTPError())

    result = await list_peer_artifacts("ghost", mesh_client=mc)
    assert result == {"error": "agent_not_found", "agent_id": "ghost"}


@pytest.mark.asyncio
async def test_list_peer_artifacts_no_mesh_client():
    """Missing mesh_client returns a clear error."""
    from src.agent.builtins.operator_tools import list_peer_artifacts

    result = await list_peer_artifacts("alpha", mesh_client=None)
    assert "error" in result
    assert "mesh_client" in result["error"]


# ── read_peer_artifact ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_peer_artifact_happy_path():
    """Happy path returns {agent_id, name, content, size, encoding}."""
    from src.agent.builtins.operator_tools import read_peer_artifact

    payload = {
        "agent_id": "alpha",
        "name": "design.md",
        "content": "# Design\n\nBody here.",
        "size": 22,
        "encoding": "utf-8",
    }
    mc = MagicMock()
    mc.read_peer_artifact = AsyncMock(return_value=payload)

    result = await read_peer_artifact("alpha", "design.md", mesh_client=mc)

    assert result == payload
    mc.read_peer_artifact.assert_awaited_once_with("alpha", "design.md")


@pytest.mark.asyncio
async def test_read_peer_artifact_binary_base64():
    """Binary artifacts come back base64 with encoding='base64'."""
    from src.agent.builtins.operator_tools import read_peer_artifact

    payload = {
        "agent_id": "alpha",
        "name": "logo.png",
        "content": "iVBORw0KGgoAAAANSUhEUgAA",
        "size": 18,
        "encoding": "base64",
    }
    mc = MagicMock()
    mc.read_peer_artifact = AsyncMock(return_value=payload)

    result = await read_peer_artifact("alpha", "logo.png", mesh_client=mc)
    assert result["encoding"] == "base64"
    assert result["content"] == "iVBORw0KGgoAAAANSUhEUgAA"


@pytest.mark.asyncio
async def test_read_peer_artifact_blocked_for_non_operator(monkeypatch):
    """Non-operator agents are denied at the skill layer."""
    from src.agent.builtins.operator_tools import read_peer_artifact

    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    mc = MagicMock()
    mc.read_peer_artifact = AsyncMock()

    result = await read_peer_artifact("alpha", "design.md", mesh_client=mc)
    assert "error" in result
    assert "operator" in result["error"].lower()
    mc.read_peer_artifact.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_peer_artifact_artifact_not_found():
    """A 404 from the mesh becomes {error: artifact_not_found}."""
    from src.agent.builtins.operator_tools import read_peer_artifact

    fake_response = MagicMock()
    fake_response.status_code = 404
    fake_response.text = "Artifact 'ghost.md' not found on agent 'alpha'"

    class FakeHTTPError(Exception):
        def __init__(self):
            super().__init__("404 Not Found")
            self.response = fake_response

    mc = MagicMock()
    mc.read_peer_artifact = AsyncMock(side_effect=FakeHTTPError())

    result = await read_peer_artifact("alpha", "ghost.md", mesh_client=mc)
    assert result == {
        "error": "artifact_not_found",
        "agent_id": "alpha",
        "name": "ghost.md",
    }


@pytest.mark.asyncio
async def test_read_peer_artifact_oversize():
    """A 413 from the mesh becomes {error: oversize}."""
    from src.agent.builtins.operator_tools import read_peer_artifact

    fake_response = MagicMock()
    fake_response.status_code = 413
    fake_response.text = "Artifact too large"

    class FakeHTTPError(Exception):
        def __init__(self):
            super().__init__("413 Too Large")
            self.response = fake_response

    mc = MagicMock()
    mc.read_peer_artifact = AsyncMock(side_effect=FakeHTTPError())

    result = await read_peer_artifact("alpha", "huge.bin", mesh_client=mc)
    assert result == {
        "error": "oversize",
        "agent_id": "alpha",
        "name": "huge.bin",
    }


@pytest.mark.asyncio
async def test_read_peer_artifact_invalid_name_400():
    """A 400 from the mesh becomes {error: invalid_name}."""
    from src.agent.builtins.operator_tools import read_peer_artifact

    fake_response = MagicMock()
    fake_response.status_code = 400
    fake_response.text = "Path traversal not allowed"

    class FakeHTTPError(Exception):
        def __init__(self):
            super().__init__("400 Bad Request")
            self.response = fake_response

    mc = MagicMock()
    mc.read_peer_artifact = AsyncMock(side_effect=FakeHTTPError())

    result = await read_peer_artifact("alpha", "../etc/passwd", mesh_client=mc)
    assert result == {
        "error": "invalid_name",
        "agent_id": "alpha",
        "name": "../etc/passwd",
    }


@pytest.mark.asyncio
async def test_read_peer_artifact_no_mesh_client():
    """Missing mesh_client returns a clear error."""
    from src.agent.builtins.operator_tools import read_peer_artifact

    result = await read_peer_artifact("alpha", "design.md", mesh_client=None)
    assert "error" in result
    assert "mesh_client" in result["error"]


@pytest.mark.asyncio
async def test_read_peer_artifact_missing_name():
    """Empty name returns a clear error before round-tripping."""
    from src.agent.builtins.operator_tools import read_peer_artifact

    mc = MagicMock()
    mc.read_peer_artifact = AsyncMock()
    result = await read_peer_artifact("alpha", "", mesh_client=mc)
    assert result == {"error": "name is required"}
    mc.read_peer_artifact.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_peer_artifact_truncates_long_error_body():
    """A noisy mesh body is truncated to 200 chars in the error payload."""
    from src.agent.builtins.operator_tools import read_peer_artifact

    fake_response = MagicMock()
    fake_response.status_code = 500
    fake_response.text = "A" * 5000

    class FakeHTTPError(Exception):
        def __init__(self):
            super().__init__("500")
            self.response = fake_response

    mc = MagicMock()
    mc.read_peer_artifact = AsyncMock(side_effect=FakeHTTPError())
    result = await read_peer_artifact("alpha", "x.md", mesh_client=mc)
    assert result["error"] == "mesh_error"
    assert result["status"] == 500
    assert len(result["body"]) <= 200


# ── Integration: ASGI roundtrip through the real mesh app ──────────


@pytest.mark.asyncio
async def test_mesh_endpoint_rejects_path_traversal(tmp_path, monkeypatch):
    """``GET /mesh/agents/{id}/artifacts/../etc/passwd`` is rejected at
    the mesh layer with HTTP 400, before reaching the transport.

    Mirrors the ``test_get_agent_config_strips_agent_id_from_permissions``
    pattern from PR 898 — spins up create_mesh_app over ASGITransport
    so we exercise the real auth + permission machinery without Docker.
    """
    import importlib

    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.traces import TraceStore

    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)
    create_mesh_app = server_module.create_mesh_app

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    router.register_agent("operator", "http://operator:8400", [])
    router.register_agent("alpha", "http://alpha:8400", [])

    auth_tokens = {"operator": "operator-secret", "alpha": "alpha-secret"}
    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces, auth_tokens=auth_tokens,
    )

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            # Authenticated as operator, so the operator gate passes
            # and the path-traversal validator gets to run.
            resp = await client.get(
                "/mesh/agents/alpha/artifacts/..%2Fetc%2Fpasswd",
                headers={"authorization": "Bearer operator-secret"},
            )
            # FastAPI's path:path converter URL-decodes the segment, so
            # the route handler sees the literal '../etc/passwd' and
            # the validator rejects with HTTP 400.
            assert resp.status_code == 400, resp.text
            assert (
                "traversal" in resp.text.lower()
                or "invalid" in resp.text.lower()
            )
    finally:
        bb.close()
        costs.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_PROJECT_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_mesh_endpoint_rejects_non_operator(tmp_path, monkeypatch):
    """An authenticated non-operator agent gets HTTP 403."""
    import importlib

    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.traces import TraceStore

    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)
    create_mesh_app = server_module.create_mesh_app

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    router.register_agent("operator", "http://operator:8400", [])
    router.register_agent("alpha", "http://alpha:8400", [])
    router.register_agent("scout", "http://scout:8400", [])

    # Auth ON: scout's token authenticates the request but caller
    # identity resolves to "scout" — not "operator" — so the
    # endpoint must 403.
    auth_tokens = {
        "operator": "operator-secret",
        "alpha": "alpha-secret",
        "scout": "scout-secret",
    }
    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces, auth_tokens=auth_tokens,
    )

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get(
                "/mesh/agents/alpha/artifacts",
                headers={"authorization": "Bearer scout-secret"},
            )
            assert resp.status_code == 403, resp.text
    finally:
        bb.close()
        costs.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_PROJECT_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_mesh_endpoint_404_for_unknown_agent(tmp_path, monkeypatch):
    """``GET /mesh/agents/ghost/artifacts`` returns 404 — never touches transport."""
    import importlib

    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.traces import TraceStore

    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)
    create_mesh_app = server_module.create_mesh_app

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

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            hdrs = {"authorization": "Bearer operator-secret"}
            resp = await client.get(
                "/mesh/agents/ghost/artifacts", headers=hdrs,
            )
            assert resp.status_code == 404, resp.text
            resp2 = await client.get(
                "/mesh/agents/ghost/artifacts/file.md", headers=hdrs,
            )
            assert resp2.status_code == 404, resp2.text
    finally:
        bb.close()
        costs.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_PROJECT_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_mesh_endpoint_happy_path_forwards_to_transport(
    tmp_path, monkeypatch,
):
    """Happy path: mesh forwards to transport and shapes the response.

    Stubs the transport to confirm:
      * The mesh hits ``/artifacts`` (list) and ``/artifacts/{name}`` (read).
      * The response is wrapped with the agent_id and content metadata.
    """
    import importlib

    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.traces import TraceStore
    from src.host.transport import Transport

    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)
    create_mesh_app = server_module.create_mesh_app

    class StubTransport(Transport):
        def __init__(self):
            self.calls = []

        async def request(self, agent_id, method, path, json=None, timeout=120, headers=None):
            self.calls.append((agent_id, method, path))
            if path == "/artifacts":
                return {"artifacts": [{"name": "design.md", "size": 5, "modified": 1.0}]}
            if path == "/artifacts/design.md":
                return {
                    "name": "design.md",
                    "content": "hello",
                    "size": 5,
                    "encoding": "utf-8",
                    "mime_type": "text/markdown",
                }
            return {"error": "unexpected"}

        async def is_reachable(self, agent_id, timeout=5):  # pragma: no cover
            return True

        def request_sync(self, agent_id, method, path, json=None, timeout=120, headers=None):  # pragma: no cover
            return {}

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    transport = StubTransport()

    router.register_agent("operator", "http://operator:8400", [])
    router.register_agent("alpha", "http://alpha:8400", [])

    auth_tokens = {"operator": "operator-secret", "alpha": "alpha-secret"}
    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces, transport=transport,
        auth_tokens=auth_tokens,
    )

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            hdrs = {"authorization": "Bearer operator-secret"}
            resp = await client.get(
                "/mesh/agents/alpha/artifacts", headers=hdrs,
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["agent_id"] == "alpha"
            assert data["artifacts"] == [
                {"name": "design.md", "size": 5, "modified": 1.0},
            ]

            resp2 = await client.get(
                "/mesh/agents/alpha/artifacts/design.md", headers=hdrs,
            )
            assert resp2.status_code == 200, resp2.text
            data2 = resp2.json()
            assert data2 == {
                "agent_id": "alpha",
                "name": "design.md",
                "content": "hello",
                "size": 5,
                "encoding": "utf-8",
            }

        # Transport was hit with the exact paths the dashboard uses.
        assert ("alpha", "GET", "/artifacts") in transport.calls
        assert ("alpha", "GET", "/artifacts/design.md") in transport.calls
    finally:
        bb.close()
        costs.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_PROJECT_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_mesh_endpoint_413_when_response_oversize(tmp_path, monkeypatch):
    """Read endpoint enforces a 5 MB cap on the mesh layer."""
    import importlib

    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.traces import TraceStore
    from src.host.transport import Transport

    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)
    create_mesh_app = server_module.create_mesh_app

    class StubTransport(Transport):
        async def request(self, agent_id, method, path, json=None, timeout=120, headers=None):
            # Simulate the agent reporting a 10 MB artifact via metadata
            # (we don't actually allocate that much memory in the test).
            return {
                "name": "huge.bin",
                "content": "x",
                "size": 10 * 1024 * 1024,
                "encoding": "utf-8",
            }

        async def is_reachable(self, agent_id, timeout=5):  # pragma: no cover
            return True

        def request_sync(self, agent_id, method, path, json=None, timeout=120, headers=None):  # pragma: no cover
            return {}

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    router.register_agent("operator", "http://operator:8400", [])
    router.register_agent("alpha", "http://alpha:8400", [])

    auth_tokens = {"operator": "operator-secret"}
    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces, transport=StubTransport(),
        auth_tokens=auth_tokens,
    )

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get(
                "/mesh/agents/alpha/artifacts/huge.bin",
                headers={"authorization": "Bearer operator-secret"},
            )
            assert resp.status_code == 413, resp.text
    finally:
        bb.close()
        costs.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_PROJECT_SCOPE_MODE", raising=False)
        importlib.reload(server_module)
