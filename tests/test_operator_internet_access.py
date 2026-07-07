"""Tests for the Operator Settings → Internet access toggle.

Surface under test:

* ``AgentPermissions.can_use_internet`` (field default False for the
  deny-all fallback; create paths grant True so new agents get internet on).
* ``PermissionMatrix.can_use_internet(agent_id)`` (mirrors the field;
  trusted-internal callers always pass).
* ``_OPERATOR_ALLOWED_TOOLS`` includes ``http_request`` and
  ``web_search`` so the operator can call them when the toggle is on.
* Operator's idempotent backfill in ``_ensure_operator_agent`` writes
  ``can_use_internet=True`` for any existing operator that predates the
  field, then leaves the user's explicit choice alone.
* ``POST /mesh/operator/internet-access`` flips the permission, writes
  the audit row, and (when the operator container is registered) pushes
  to ``/config`` so the runtime tool surface updates immediately.
* ``GET /mesh/operator/internet-access`` returns the current state.
* The agent loop's ``_runtime_disabled_tools`` is seeded from
  ``OL_INTERNET_ACCESS_ENABLED`` env var at construction time, so a
  restart while the toggle is OFF doesn't briefly re-expose
  http_request / web_search.
* ``set_runtime_disabled_tools`` flips the filter on the running loop.

These tests stub the mesh/router/permission machinery directly rather
than spinning up a real container, so they exercise the contract
without Docker.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock

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


def _agents_yaml(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir(exist_ok=True)
    afile = tmp_path / "config" / "agents.yaml"
    afile.write_text(yaml.dump({"agents": {"operator": {"role": "operator"}}}))
    return afile


@pytest.fixture
def mesh_app(tmp_path, monkeypatch):
    """Mesh app pinned to tmp_path with an ``operator`` agent on disk."""
    afile = _agents_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)
    import src.cli.config as cli_cfg
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", afile)
    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", tmp_path / "config" / "permissions.json")
    (tmp_path / "config" / "permissions.json").write_text(
        json.dumps({"permissions": {"operator": {"can_use_internet": True}}}),
    )

    import src.host.server as server_module
    importlib.reload(server_module)

    permissions = PermissionMatrix()
    permissions.permissions["operator"] = AgentPermissions(
        agent_id="operator", can_use_internet=True,
    )
    router = MessageRouter(permissions, {})
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
    )
    yield app, permissions, router, blackboard
    blackboard.close()
    importlib.reload(server_module)


# ── Permission model + matrix ─────────────────────────────────────────


def test_agent_permissions_field_default_can_use_internet_is_false():
    """The MODEL field default stays ``False`` so the deny-all fallback for an
    unknown agent (PermissionMatrix.get_permissions) is restrictive — same
    convention as ``can_use_browser``. New agents get internet ON via the
    create-path base defaults (see test_templates.TestCreateAgentParity), not
    via the bare field default."""
    perms = AgentPermissions(agent_id="worker")
    assert perms.can_use_internet is False
    # Mirrors the established can_use_browser convention (field default False,
    # create-path default True).
    assert perms.can_use_browser is False


def test_agent_permissions_can_use_internet_round_trip():
    perms = AgentPermissions(agent_id="op", can_use_internet=True)
    assert perms.can_use_internet is True
    data = perms.model_dump()
    assert data["can_use_internet"] is True
    rebuilt = AgentPermissions(**data)
    assert rebuilt.can_use_internet is True


def test_permission_matrix_can_use_internet_reads_field():
    matrix = PermissionMatrix()
    matrix.permissions["alpha"] = AgentPermissions(
        agent_id="alpha", can_use_internet=True,
    )
    matrix.permissions["beta"] = AgentPermissions(
        agent_id="beta", can_use_internet=False,
    )
    assert matrix.can_use_internet("alpha") is True
    assert matrix.can_use_internet("beta") is False


def test_permission_matrix_can_use_internet_trusted_internal_passes():
    """Mesh-internal callers always pass — matches the pattern used by
    ``can_use_browser`` etc."""
    matrix = PermissionMatrix()
    assert matrix.can_use_internet("mesh") is True


# ── Operator's _OPERATOR_ALLOWED_TOOLS + backfill ─────────────────────


def test_operator_allowed_tools_includes_internet_tools():
    """The operator's static allowlist must include http_request and
    web_search so the LLM can call them when the runtime filter is
    empty (toggle ON, default)."""
    from src.cli.config import _OPERATOR_ALLOWED_TOOLS
    assert "http_request" in _OPERATOR_ALLOWED_TOOLS
    assert "web_search" in _OPERATOR_ALLOWED_TOOLS


def test_operator_backfill_grants_can_use_internet(tmp_path, monkeypatch):
    """Existing operator entries that predate this PR get
    ``can_use_internet=True`` on next startup via the idempotent
    backfill block in ``_ensure_operator_agent``. The user's explicit
    choice (True or False) is preserved across restarts."""
    import src.cli.config as cli_cfg

    (tmp_path / "config").mkdir(exist_ok=True)
    afile = tmp_path / "config" / "agents.yaml"
    afile.write_text(yaml.dump({
        "agents": {
            "operator": {
                "role": "Operator", "model": "openai/gpt-4o-mini",
            },
        },
    }))
    pfile = tmp_path / "config" / "permissions.json"
    # Existing operator entry WITHOUT can_use_internet — simulates a
    # deployment that predates this PR.
    pfile.write_text(json.dumps({
        "permissions": {
            "operator": {
                "can_manage_fleet": True,
                "can_manage_teams": True,
                "can_edit_agent_config": True,
                "can_view_fleet_metrics": True,
                "can_route_tasks": True,
                "can_request_user_credentials": True,
                "can_message": ["*"],
                "can_publish": ["*"],
                "can_subscribe": ["*"],
                "blackboard_read": ["*"],
                "blackboard_write": ["*"],
            },
        },
    }))
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", afile)
    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", pfile)
    monkeypatch.chdir(tmp_path)

    cli_cfg._ensure_operator_agent()

    after = json.loads(pfile.read_text())
    assert after["permissions"]["operator"]["can_use_internet"] is True


def test_operator_backfill_preserves_explicit_false(tmp_path, monkeypatch):
    """User toggled internet OFF; backfill must NOT silently flip it
    back to True on restart."""
    import src.cli.config as cli_cfg

    (tmp_path / "config").mkdir(exist_ok=True)
    afile = tmp_path / "config" / "agents.yaml"
    afile.write_text(yaml.dump({
        "agents": {
            "operator": {"role": "Operator", "model": "openai/gpt-4o-mini"},
        },
    }))
    pfile = tmp_path / "config" / "permissions.json"
    pfile.write_text(json.dumps({
        "permissions": {
            "operator": {
                "can_use_internet": False,
                "can_manage_fleet": True,
                "can_manage_teams": True,
                "can_edit_agent_config": True,
                "can_view_fleet_metrics": True,
                "can_route_tasks": True,
                "can_request_user_credentials": True,
                "can_message": ["*"],
                "can_publish": ["*"],
                "can_subscribe": ["*"],
                "blackboard_read": ["*"],
                "blackboard_write": ["*"],
            },
        },
    }))
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", afile)
    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", pfile)
    monkeypatch.chdir(tmp_path)

    cli_cfg._ensure_operator_agent()
    after = json.loads(pfile.read_text())
    assert after["permissions"]["operator"]["can_use_internet"] is False


# ── Mesh endpoint behavior ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_internet_access_get_returns_current_state(mesh_app):
    app, permissions, router, blackboard = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(
            "/mesh/operator/internet-access",
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text
    assert r.json() == {"enabled": True}


@pytest.mark.asyncio
async def test_internet_access_post_flips_permission(mesh_app, tmp_path):
    app, permissions, router, blackboard = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/operator/internet-access",
            json={"enabled": False},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["enabled"] is False
    assert body["previous"] is True
    # Permissions file rewritten.
    after = json.loads((tmp_path / "config" / "permissions.json").read_text())
    assert after["permissions"]["operator"]["can_use_internet"] is False
    # Matrix reloaded.
    assert permissions.can_use_internet("operator") is False


@pytest.mark.asyncio
async def test_internet_access_post_rejects_non_operator(mesh_app):
    app, *_ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/operator/internet-access",
            json={"enabled": False},
            headers={"X-Agent-ID": "writer"},  # not operator
        )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_internet_access_post_validates_body(mesh_app):
    app, *_ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r1 = await c.post(
            "/mesh/operator/internet-access",
            json={},
            headers=_human_origin_headers(),
        )
        r2 = await c.post(
            "/mesh/operator/internet-access",
            json={"enabled": "yes"},
            headers=_human_origin_headers(),
        )
    assert r1.status_code == 400
    assert r2.status_code == 400


@pytest.mark.asyncio
async def test_internet_access_post_writes_audit_row(mesh_app):
    app, _, _, blackboard = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/operator/internet-access",
            json={"enabled": False},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200
    # Audit log should carry the field change.
    audit = blackboard.get_audit_log(action="edit_agent")
    matching = [
        e for e in audit["entries"]
        if e.get("field") == "can_use_internet"
        and e.get("target") == "operator"
    ]
    assert len(matching) == 1
    assert matching[0]["after_value"] == "false"
    assert matching[0]["before_value"] == "true"


@pytest.mark.asyncio
async def test_internet_access_post_emits_agent_config_updated(mesh_app, tmp_path):
    """The SPA listens for ``agent_config_updated`` to flip the toggle
    UI live without a poll. The endpoint must fire it with
    ``field=can_use_internet`` and ``live=<bool>``."""
    import src.host.server as server_module
    events: list[tuple] = []

    class _Bus:
        def emit(self, event_type, agent="", data=None):
            events.append((event_type, agent, data))

    permissions = PermissionMatrix()
    permissions.permissions["operator"] = AgentPermissions(
        agent_id="operator", can_use_internet=True,
    )
    router = MessageRouter(permissions, {})
    blackboard = Blackboard(str(tmp_path / "bb_emit.db"))
    pubsub = PubSub()
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
            r = await c.post(
                "/mesh/operator/internet-access",
                json={"enabled": False},
                headers=_human_origin_headers(),
            )
        assert r.status_code == 200, r.text
        cfg_events = [e for e in events if e[0] == "agent_config_updated"]
        assert len(cfg_events) == 1
        _, agent, data = cfg_events[0]
        assert agent == "operator"
        assert data["field"] == "can_use_internet"
        assert data["agent_id"] == "operator"
        # Live=True because no operator container is registered → the
        # transport push branch was skipped; the durable write is what
        # the user cares about.
        assert data["live"] is True
    finally:
        blackboard.close()


# ── Agent loop runtime filter ─────────────────────────────────────────


def _build_loop(monkeypatch, *, allowed_tools=None, internet_env=None):
    """Construct an AgentLoop with a stub mesh_client.

    ``allowed_tools`` simulates the static operator allowlist; ``None``
    matches a worker. ``internet_env`` ("true"/"false"/None) seeds the
    boot-time runtime filter.
    """
    if internet_env is None:
        monkeypatch.delenv("OL_INTERNET_ACCESS_ENABLED", raising=False)
    else:
        monkeypatch.setenv("OL_INTERNET_ACCESS_ENABLED", internet_env)
    from src.agent.loop import AgentLoop
    mesh_client = MagicMock()
    mesh_client.agent_id = "operator"
    loop = AgentLoop.__new__(AgentLoop)
    # Manually run only the bits we need from __init__ to keep this a
    # unit test (the full __init__ pulls in workspace/tools/etc.).
    loop._allowed_tools = (
        frozenset(allowed_tools) if allowed_tools is not None else None
    )
    loop._excluded_tools = None
    loop._runtime_disabled_tools = frozenset()
    import os as _os
    if _os.environ.get("OL_INTERNET_ACCESS_ENABLED", "true").lower() == "false":
        loop._runtime_disabled_tools = frozenset({"http_request", "web_search"})
    return loop


def test_tool_filter_kw_no_runtime_disabled_uses_static_allowlist(monkeypatch):
    loop = _build_loop(monkeypatch, allowed_tools={"hand_off", "http_request"})
    kw = type(loop)._tool_filter_kw.fget(loop)
    assert "exclude" not in kw  # operator path has no exclude set
    assert kw["allowed"] == frozenset({"hand_off", "http_request"})


def test_tool_filter_kw_runtime_disabled_subtracts_from_allowed(monkeypatch):
    """Operator with internet off — ``http_request`` and ``web_search``
    are subtracted from the static allowlist."""
    loop = _build_loop(
        monkeypatch,
        allowed_tools={"hand_off", "http_request", "web_search", "edit_agent"},
        internet_env="false",
    )
    kw = type(loop)._tool_filter_kw.fget(loop)
    assert kw["allowed"] == frozenset({"hand_off", "edit_agent"})


def test_set_runtime_disabled_tools_flips_at_runtime(monkeypatch):
    """The /config push path calls ``set_runtime_disabled_tools`` to
    swap the filter without restarting the agent."""
    loop = _build_loop(
        monkeypatch,
        allowed_tools={"hand_off", "http_request", "web_search"},
    )
    # Initially all three exposed.
    assert "http_request" in type(loop)._tool_filter_kw.fget(loop)["allowed"]
    # Disable.
    from src.agent.loop import AgentLoop
    AgentLoop.set_runtime_disabled_tools(
        loop, {"http_request", "web_search"},
    )
    after = type(loop)._tool_filter_kw.fget(loop)["allowed"]
    assert "http_request" not in after
    assert "web_search" not in after
    assert "hand_off" in after
    # Re-enable.
    AgentLoop.set_runtime_disabled_tools(loop, [])
    restored = type(loop)._tool_filter_kw.fget(loop)["allowed"]
    assert "http_request" in restored
    assert "web_search" in restored


def test_tool_filter_kw_worker_path_unions_runtime_disabled(monkeypatch):
    """Non-operator agents have ``_allowed_tools=None``; the runtime
    filter folds into ``exclude`` so the global tool surface narrows."""
    loop = _build_loop(monkeypatch, allowed_tools=None)
    loop._excluded_tools = frozenset({"unused"})
    from src.agent.loop import AgentLoop
    AgentLoop.set_runtime_disabled_tools(loop, {"http_request"})
    kw = type(loop)._tool_filter_kw.fget(loop)
    assert kw["exclude"] == frozenset({"unused", "http_request"})
    assert "allowed" not in kw
