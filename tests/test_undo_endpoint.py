"""Tests for the PR 1 undo endpoint: POST /mesh/changes/undo/{token}."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions, MessageOrigin


def _human_origin_headers(agent_id: str = "operator") -> dict:
    origin = MessageOrigin(kind="human", channel="cli", user="u1")
    return {
        "X-Agent-ID": agent_id,
        "X-Origin": origin.to_header_value(),
    }


def _agents_yaml(tmp_path, names: list[str]) -> Path:
    (tmp_path / "config").mkdir(exist_ok=True)
    afile = tmp_path / "config" / "agents.yaml"
    cfg = {"agents": {
        name: {
            "role": name,
            "model": "openai/gpt-4o-mini",
            "initial_instructions": f"# original {name}",
            "initial_soul": "# original soul",
        }
        for name in names
    }}
    afile.write_text(yaml.dump(cfg))
    return afile


@pytest.fixture
def mesh_app(tmp_path, monkeypatch):
    afile = _agents_yaml(tmp_path, names=["writer"])
    monkeypatch.chdir(tmp_path)
    import src.cli.config as cli_cfg
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", afile)
    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", tmp_path / "config" / "permissions.json")
    (tmp_path / "config" / "permissions.json").write_text('{"permissions": {}}')

    import src.host.server as server_module
    importlib.reload(server_module)

    permissions = PermissionMatrix()
    permissions.permissions["operator"] = AgentPermissions(
        agent_id="operator", can_route_tasks=True,
    )
    permissions.permissions["writer"] = AgentPermissions(agent_id="writer")
    router = MessageRouter(permissions, {})
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
    )
    yield app, server_module, tmp_path
    blackboard.close()
    importlib.reload(server_module)


async def _do_soft_edit(client: AsyncClient, *, field="instructions", value="# v2") -> str:
    r = await client.post(
        "/mesh/agents/writer/edit-soft",
        json={"field": field, "value": value, "reason": "user_asked"},
        headers=_human_origin_headers(),
    )
    assert r.status_code == 200, r.text
    return r.json()["undo_token"]


@pytest.mark.asyncio
async def test_undo_happy_path_restores_yaml(mesh_app):
    app, _, tmp_path = mesh_app
    afile = tmp_path / "config" / "agents.yaml"
    original = "# original writer"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        token = await _do_soft_edit(c, value="# v2 punchier")
        # Sanity: file changed.
        cfg = yaml.safe_load(afile.read_text())
        assert cfg["agents"]["writer"]["initial_instructions"] == "# v2 punchier"
        # Undo.
        r = await c.post(
            f"/mesh/changes/undo/{token}",
            json={},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["agent_id"] == "writer"
    assert body["field"] == "instructions"
    assert body["restored_value"] == original

    # File reverted.
    cfg2 = yaml.safe_load(afile.read_text())
    assert cfg2["agents"]["writer"]["initial_instructions"] == original


@pytest.mark.asyncio
async def test_double_undo_returns_404(mesh_app):
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        token = await _do_soft_edit(c)
        r1 = await c.post(
            f"/mesh/changes/undo/{token}", json={},
            headers=_human_origin_headers(),
        )
        r2 = await c.post(
            f"/mesh/changes/undo/{token}", json={},
            headers=_human_origin_headers(),
        )
    assert r1.status_code == 200
    assert r2.status_code == 404


@pytest.mark.asyncio
async def test_undo_unknown_token_404(mesh_app):
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/changes/undo/does-not-exist",
            json={},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_undo_expired_token_404(mesh_app):
    """Expired tokens return 404 even if the row is still in the table."""
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        token = await _do_soft_edit(c)
        # Force-expire the row by editing the change_history table directly.
        with app.change_history._conn() as conn:
            conn.execute(
                "UPDATE change_history SET expires_at = 0 WHERE undo_token=?",
                (token,),
            )
        r = await c.post(
            f"/mesh/changes/undo/{token}",
            json={},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_undo_403_for_non_operator(mesh_app):
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        token = await _do_soft_edit(c)
        r = await c.post(
            f"/mesh/changes/undo/{token}",
            json={},
            headers={"X-Agent-ID": "writer"},
        )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_heartbeat_schedule_soft_edit_updates_cron(mesh_app):
    """PR-L' — heartbeat_schedule edits must retarget the agent's cron job."""
    app, server_module, tmp_path = mesh_app

    blackboard = Blackboard(str(tmp_path / "bb_hs.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    permissions.permissions["operator"] = AgentPermissions(
        agent_id="operator", can_route_tasks=True,
    )
    permissions.permissions["writer"] = AgentPermissions(agent_id="writer")
    router = MessageRouter(permissions, {})

    from src.host.cron import CronScheduler
    cron = CronScheduler(config_path=str(tmp_path / "cron.json"))
    cron.ensure_heartbeat("writer", schedule="every 1h")

    app2 = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        cron_scheduler=cron,
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t") as c:
            r = await c.post(
                "/mesh/agents/writer/edit-soft",
                json={
                    "field": "heartbeat_schedule",
                    "value": "*/15 * * * *",
                    "reason": "user_asked",
                },
                headers=_human_origin_headers(),
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["success"] is True
        assert body["field"] == "heartbeat_schedule"
        # Cron job updated.
        hb = cron.find_heartbeat_job("writer")
        assert hb is not None
        assert hb.schedule == "*/15 * * * *"
    finally:
        blackboard.close()


@pytest.mark.asyncio
async def test_heartbeat_schedule_undo_restores_prior_cron(mesh_app):
    """Undoing a heartbeat_schedule edit must restore the prior schedule."""
    app, server_module, tmp_path = mesh_app

    blackboard = Blackboard(str(tmp_path / "bb_hs2.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    permissions.permissions["operator"] = AgentPermissions(
        agent_id="operator", can_route_tasks=True,
    )
    permissions.permissions["writer"] = AgentPermissions(agent_id="writer")
    router = MessageRouter(permissions, {})

    from src.host.cron import CronScheduler
    cron = CronScheduler(config_path=str(tmp_path / "cron2.json"))
    cron.ensure_heartbeat("writer", schedule="every 1h")

    app2 = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        cron_scheduler=cron,
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t") as c:
            edit = await c.post(
                "/mesh/agents/writer/edit-soft",
                json={
                    "field": "heartbeat_schedule",
                    "value": "*/15 * * * *",
                    "reason": "user_asked",
                },
                headers=_human_origin_headers(),
            )
            assert edit.status_code == 200, edit.text
            token = edit.json()["undo_token"]
            r = await c.post(
                f"/mesh/changes/undo/{token}",
                json={},
                headers=_human_origin_headers(),
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["success"] is True
        # Cron schedule reverted to the original.
        hb = cron.find_heartbeat_job("writer")
        assert hb is not None
        assert hb.schedule == "every 1h"
    finally:
        blackboard.close()


@pytest.mark.asyncio
async def test_undo_emits_undone_event(mesh_app):
    """After a successful undo the bus must see operator_action_receipt_undone."""
    app, server_module, tmp_path = mesh_app

    events = []

    class _Bus:
        def emit(self, event_type, agent="", data=None):
            events.append((event_type, agent, data))

    blackboard = Blackboard(str(tmp_path / "bb2.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    permissions.permissions["operator"] = AgentPermissions(
        agent_id="operator", can_route_tasks=True,
    )
    router = MessageRouter(permissions, {})
    bus = _Bus()
    app2 = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        event_bus=bus,
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t") as c:
            token = await _do_soft_edit(c, value="# v2")
            r = await c.post(
                f"/mesh/changes/undo/{token}", json={},
                headers=_human_origin_headers(),
            )
        assert r.status_code == 200
        kinds = [e[0] for e in events]
        assert "operator_action_receipt" in kinds
        assert "operator_action_receipt_undone" in kinds
    finally:
        blackboard.close()


@pytest.mark.asyncio
async def test_undo_hard_field_model_restores_yaml(mesh_app):
    """Hard-field undo: a model edit applied immediately via the
    unified /edit-soft endpoint can be reverted by token. Verifies the
    30-min-TTL receipt path is bidirectional — _apply_pending_change
    runs in reverse when the undo endpoint is called."""
    app, _, tmp_path = mesh_app
    afile = tmp_path / "config" / "agents.yaml"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        token = await _do_soft_edit(
            c, field="model", value="anthropic/claude-opus-4-7",
        )
        cfg = yaml.safe_load(afile.read_text())
        assert cfg["agents"]["writer"]["model"] == "anthropic/claude-opus-4-7"
        r = await c.post(
            f"/mesh/changes/undo/{token}",
            json={},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["agent_id"] == "writer"
    assert body["field"] == "model"
    # Restored to the pre-edit model.
    cfg2 = yaml.safe_load(afile.read_text())
    assert cfg2["agents"]["writer"]["model"] == "openai/gpt-4o-mini"


@pytest.mark.asyncio
async def test_undo_hard_field_permissions_restores_matrix(mesh_app):
    """Hard-field undo for permissions: the grant is reverted in
    permissions.json AND the in-memory matrix is reloaded. Defensive
    fix exercise — even when no prior entry existed for the agent, the
    setdefault path stores both the apply and the undo state."""
    app, _, tmp_path = mesh_app
    perms_file = tmp_path / "config" / "permissions.json"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Grant browser permission (writer had no entry → setdefault).
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={
                "field": "permissions",
                "value": {"can_use_browser": True},
                "reason": "user_asked",
            },
            headers=_human_origin_headers(),
        )
        assert r.status_code == 200, r.text
        token = r.json()["undo_token"]
        # Entry exists with the grant.
        import json
        perms_after = json.loads(perms_file.read_text())
        assert perms_after["permissions"]["writer"]["can_use_browser"] is True
        # Undo restores the empty/prior state.
        r2 = await c.post(
            f"/mesh/changes/undo/{token}",
            json={},
            headers=_human_origin_headers(),
        )
    assert r2.status_code == 200, r2.text
    assert r2.json()["field"] == "permissions"
    # File rewritten with the pre-grant state.
    perms_reverted = json.loads(perms_file.read_text())
    # writer's entry may still exist but can_use_browser is no longer set
    # (or absent, depending on initial state — both shapes mean "not
    # granted"). The grant key, if present, must not be True.
    writer_perms = perms_reverted.get("permissions", {}).get("writer", {})
    assert writer_perms.get("can_use_browser") is not True
