"""PR-O'.1 — cross-project blackboard access counter (observability slice).

Pure telemetry. No enforcement. Counter is process-lifetime, surfaced on
``GET /mesh/system/metrics`` as ``blackboard_cross_team_total``.

The plan (`docs/plans/2026-05-08-post-board-roadmap.md`, Phase 3 PR-O'.1)
calls out the increment/no-increment matrix exercised below:

  - cross-project read   → ``read`` += 1
  - cross-project write  → ``write`` += 1
  - same-project access  → no change
  - operator caller      → no change
  - x-mesh-internal      → no change
  - new key (no prior)   → no change
  - counter visible on operator-gated metrics endpoint
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest
import yaml
from httpx import ASGITransport, AsyncClient


def _build_mesh(tmp_path, monkeypatch, *, auth_tokens=None):
    """Build a mesh app + isolated blackboard with two projects on disk.

    Project layout:
        proj-a: members=[agent-a]
        proj-b: members=[agent-b]
        agent-c is standalone (no project)

    Returns ``(app, blackboard, projects_dir, server_module)`` so callers
    can ``patch('src.cli.config.TEAMS_DIR', projects_dir)`` for the
    duration of their HTTP calls.
    """
    # Pin the scope flag to warn so the unrelated /mesh/agents enforce
    # path doesn't interfere with anything; reload to pick it up.
    monkeypatch.setenv("OPENLEGION_PROJECT_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)

    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.traces import TraceStore
    from src.shared.types import AgentPermissions

    projects_dir = tmp_path / "projects"
    (projects_dir / "proj-a").mkdir(parents=True)
    (projects_dir / "proj-b").mkdir(parents=True)
    (projects_dir / "proj-a" / "metadata.yaml").write_text(
        yaml.dump({
            "name": "proj-a",
            "members": ["agent-a"],
            "created_at": "2026-05-08T00:00:00+00:00",
        }),
    )
    (projects_dir / "proj-b" / "metadata.yaml").write_text(
        yaml.dump({
            "name": "proj-b",
            "members": ["agent-b"],
            "created_at": "2026-05-08T00:00:00+00:00",
        }),
    )

    perms = PermissionMatrix()
    # Permissive — the enforcement layer is NOT under test here.
    for aid in ("agent-a", "agent-b", "agent-c", "operator"):
        perms.permissions[aid] = AgentPermissions(
            agent_id=aid,
            blackboard_read=["*"],
            blackboard_write=["*"],
        )

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    router = MessageRouter(perms, {})
    router.register_agent("agent-a", "http://agent-a:8400", [])
    router.register_agent("agent-b", "http://agent-b:8400", [])
    router.register_agent("agent-c", "http://agent-c:8400", [])
    router.register_agent("operator", "http://operator:8400", [])

    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    app = server_module.create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces, auth_tokens=auth_tokens,
    )
    # Reset the module-level counter so each test starts at zero.
    server_module._blackboard_xteam_count["read"] = 0
    server_module._blackboard_xteam_count["write"] = 0
    # Stash teardown handles on the app for the test caller.
    app.state._test_bb = bb
    app.state._test_costs = costs
    app.state._test_traces = traces
    return app, bb, projects_dir, server_module


def _teardown(app, monkeypatch, server_module):
    try:
        app.state._test_bb.close()
        app.state._test_costs.close()
        app.state._test_traces.close()
    finally:
        monkeypatch.delenv("OPENLEGION_PROJECT_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_cross_project_read_increments_counter(tmp_path, monkeypatch):
    """agent-a writes a key (proj-a); agent-b (proj-b) reads it → read += 1."""
    app, bb, projects_dir, server_module = _build_mesh(tmp_path, monkeypatch)
    try:
        bb.write("scratch/note", {"hello": "world"}, written_by="agent-a")
        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get(
                    "/mesh/blackboard/scratch/note",
                    params={"agent_id": "agent-b"},
                )
        assert resp.status_code == 200
        assert server_module._blackboard_xteam_count == {"read": 1, "write": 0}
    finally:
        _teardown(app, monkeypatch, server_module)


@pytest.mark.asyncio
async def test_cross_project_write_against_existing_increments(tmp_path, monkeypatch):
    """PUT existing key written by another project → write += 1."""
    app, bb, projects_dir, server_module = _build_mesh(tmp_path, monkeypatch)
    try:
        bb.write("scratch/note", {"v": 1}, written_by="agent-a")
        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.put(
                    "/mesh/blackboard/scratch/note",
                    params={"agent_id": "agent-b"},
                    json={"v": 2},
                )
        assert resp.status_code == 200
        assert server_module._blackboard_xteam_count == {"read": 0, "write": 1}
    finally:
        _teardown(app, monkeypatch, server_module)


@pytest.mark.asyncio
async def test_same_project_does_not_increment(tmp_path, monkeypatch):
    """Two members of the same project touching the same key → no increment."""
    app, bb, projects_dir, server_module = _build_mesh(tmp_path, monkeypatch)
    try:
        # Add a second member to proj-a so two callers share one project.
        (projects_dir / "proj-a" / "metadata.yaml").write_text(
            yaml.dump({
                "name": "proj-a",
                "members": ["agent-a", "agent-c"],
                "created_at": "2026-05-08T00:00:00+00:00",
            }),
        )
        bb.write("scratch/note", {"hello": "world"}, written_by="agent-a")
        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get(
                    "/mesh/blackboard/scratch/note",
                    params={"agent_id": "agent-c"},
                )
                resp2 = await client.put(
                    "/mesh/blackboard/scratch/note",
                    params={"agent_id": "agent-c"},
                    json={"hello": "again"},
                )
        assert resp.status_code == 200
        assert resp2.status_code == 200
        assert server_module._blackboard_xteam_count == {"read": 0, "write": 0}
    finally:
        _teardown(app, monkeypatch, server_module)


@pytest.mark.asyncio
async def test_operator_caller_does_not_increment(tmp_path, monkeypatch):
    """Operator is fleet-global by definition — never counted."""
    app, bb, projects_dir, server_module = _build_mesh(tmp_path, monkeypatch)
    try:
        bb.write("scratch/note", {"v": 1}, written_by="agent-a")
        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get(
                    "/mesh/blackboard/scratch/note",
                    params={"agent_id": "operator"},
                )
                resp2 = await client.put(
                    "/mesh/blackboard/scratch/note",
                    params={"agent_id": "operator"},
                    json={"v": 2},
                )
        assert resp.status_code == 200
        assert resp2.status_code == 200
        assert server_module._blackboard_xteam_count == {"read": 0, "write": 0}
    finally:
        _teardown(app, monkeypatch, server_module)


@pytest.mark.asyncio
async def test_internal_caller_does_not_increment(tmp_path, monkeypatch):
    """``x-mesh-internal: 1`` from loopback bypasses the counter."""
    app, bb, projects_dir, server_module = _build_mesh(tmp_path, monkeypatch)
    try:
        bb.write("scratch/note", {"v": 1}, written_by="agent-a")
        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                # ASGITransport synthesises a 127.0.0.1 client peer, so
                # the loopback half of ``_is_internal_caller`` is satisfied.
                resp = await client.get(
                    "/mesh/blackboard/scratch/note",
                    params={"agent_id": "agent-b"},
                    headers={"x-mesh-internal": "1"},
                )
                resp2 = await client.put(
                    "/mesh/blackboard/scratch/note",
                    params={"agent_id": "agent-b"},
                    json={"v": 2},
                    headers={"x-mesh-internal": "1"},
                )
        assert resp.status_code == 200
        assert resp2.status_code == 200
        assert server_module._blackboard_xteam_count == {"read": 0, "write": 0}
    finally:
        _teardown(app, monkeypatch, server_module)


@pytest.mark.asyncio
async def test_new_key_write_does_not_increment(tmp_path, monkeypatch):
    """First write to a key has no prior entry → cannot be cross-project."""
    app, bb, projects_dir, server_module = _build_mesh(tmp_path, monkeypatch)
    try:
        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.put(
                    "/mesh/blackboard/scratch/fresh",
                    params={"agent_id": "agent-b"},
                    json={"v": 1},
                )
        assert resp.status_code == 200
        assert server_module._blackboard_xteam_count == {"read": 0, "write": 0}
    finally:
        _teardown(app, monkeypatch, server_module)


@pytest.mark.asyncio
async def test_counter_surfaced_on_system_metrics(tmp_path, monkeypatch):
    """Counter is reflected on ``/mesh/system/metrics`` (operator-gated)."""
    auth_tokens = {"operator": "tok-op", "agent-a": "tok-a", "agent-b": "tok-b"}
    app, bb, projects_dir, server_module = _build_mesh(
        tmp_path, monkeypatch, auth_tokens=auth_tokens,
    )
    try:
        bb.write("scratch/note", {"v": 1}, written_by="agent-a")
        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                # Trigger a cross-project read as agent-b.
                read_resp = await client.get(
                    "/mesh/blackboard/scratch/note",
                    params={"agent_id": "agent-b"},
                    headers={"Authorization": "Bearer tok-b"},
                )
                assert read_resp.status_code == 200

                # Non-operator hitting metrics → 403 (gated endpoint).
                worker_metrics = await client.get(
                    "/mesh/system/metrics",
                    headers={"Authorization": "Bearer tok-b"},
                )
                assert worker_metrics.status_code == 403

                # Operator can read the counter.
                op_metrics = await client.get(
                    "/mesh/system/metrics",
                    headers={"Authorization": "Bearer tok-op"},
                )
        assert op_metrics.status_code == 200
        body = op_metrics.json()
        assert "blackboard_cross_team_total" in body
        assert body["blackboard_cross_team_total"] == {"read": 1, "write": 0}
    finally:
        _teardown(app, monkeypatch, server_module)


@pytest.mark.asyncio
async def test_delete_existing_cross_project_increments_write(tmp_path, monkeypatch):
    """DELETE counts as a write — kind=write when prior entry is cross-project."""
    app, bb, projects_dir, server_module = _build_mesh(tmp_path, monkeypatch)
    try:
        bb.write("scratch/note", {"v": 1}, written_by="agent-a")
        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.delete(
                    "/mesh/blackboard/scratch/note",
                    params={"agent_id": "agent-b"},
                )
        assert resp.status_code == 200
        assert server_module._blackboard_xteam_count == {"read": 0, "write": 1}
    finally:
        _teardown(app, monkeypatch, server_module)


@pytest.mark.asyncio
async def test_claim_existing_cross_project_increments_write(tmp_path, monkeypatch):
    """Atomic CAS write counts as kind=write when key was last written by other project."""
    app, bb, projects_dir, server_module = _build_mesh(tmp_path, monkeypatch)
    try:
        existing = bb.write("scratch/note", {"v": 1}, written_by="agent-a")
        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/mesh/blackboard/claim",
                    json={
                        "agent_id": "agent-b",
                        "key": "scratch/note",
                        "value": {"v": 2},
                        "expected_version": existing.version,
                    },
                )
        assert resp.status_code == 200
        assert server_module._blackboard_xteam_count == {"read": 0, "write": 1}
    finally:
        _teardown(app, monkeypatch, server_module)


@pytest.mark.asyncio
async def test_standalone_agent_does_not_increment(tmp_path, monkeypatch):
    """Standalone agent (no project membership) → not "cross-project"."""
    app, bb, projects_dir, server_module = _build_mesh(tmp_path, monkeypatch)
    try:
        # agent-c is standalone; agent-a is in proj-a.
        bb.write("scratch/note", {"v": 1}, written_by="agent-a")
        with patch("src.cli.config.TEAMS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get(
                    "/mesh/blackboard/scratch/note",
                    params={"agent_id": "agent-c"},
                )
        assert resp.status_code == 200
        assert server_module._blackboard_xteam_count == {"read": 0, "write": 0}
    finally:
        _teardown(app, monkeypatch, server_module)
