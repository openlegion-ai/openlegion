"""Tests for the operator read_agent_config tool — symmetric inverse of edit_agent."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    """read_agent_config requires ALLOWED_TOOLS to be set (defence-in-depth)."""
    monkeypatch.setenv("ALLOWED_TOOLS", "read_agent_config,edit_agent")


def _canonical_config() -> dict:
    """A complete config payload mirroring edit_agent's field surface."""
    return {
        "model": "anthropic/claude-sonnet-4-20250514",
        "role": "writer",
        "thinking": "off",
        "budget": {"daily_usd": 5.0, "monthly_usd": 100.0},
        "instructions": "Do the things.",
        "soul": "Curious and concise.",
        "heartbeat": "Check inbox.",
        "interface": "Markdown sections.",
        "permissions": {"can_use_browser": True, "can_spawn": False},
        "heartbeat_schedule": "every 15m",
    }


@pytest.mark.asyncio
async def test_read_agent_config_returns_canonical_shape():
    """Full read returns {agent_id, config} with all canonical keys."""
    from src.agent.builtins.operator_tools import read_agent_config

    cfg = _canonical_config()
    mc = MagicMock()
    mc.get_agent_config = AsyncMock(
        return_value={"agent_id": "alpha", "config": cfg}
    )

    result = await read_agent_config("alpha", mesh_client=mc)

    assert result["agent_id"] == "alpha"
    assert result["config"] == cfg
    # All canonical keys present.
    expected_keys = {
        "model", "role", "thinking", "budget", "instructions",
        "soul", "heartbeat", "interface", "permissions", "heartbeat_schedule",
    }
    assert set(result["config"].keys()) == expected_keys
    mc.get_agent_config.assert_awaited_once_with("alpha", fields=None)


@pytest.mark.asyncio
async def test_read_agent_config_partial_fields():
    """fields=['instructions', 'soul'] forwards filter and returns subset."""
    from src.agent.builtins.operator_tools import read_agent_config

    mc = MagicMock()
    mc.get_agent_config = AsyncMock(
        return_value={
            "agent_id": "alpha",
            "config": {
                "instructions": "Do the things.",
                "soul": "Curious and concise.",
            },
        }
    )

    result = await read_agent_config(
        "alpha", fields=["instructions", "soul"], mesh_client=mc,
    )

    assert result["agent_id"] == "alpha"
    assert set(result["config"].keys()) == {"instructions", "soul"}
    mc.get_agent_config.assert_awaited_once_with(
        "alpha", fields=["instructions", "soul"],
    )


@pytest.mark.asyncio
async def test_read_agent_config_unknown_field_rejected():
    """Unknown field names short-circuit before calling the mesh."""
    from src.agent.builtins.operator_tools import read_agent_config

    mc = MagicMock()
    mc.get_agent_config = AsyncMock(return_value={})

    result = await read_agent_config(
        "alpha", fields=["nonsense"], mesh_client=mc,
    )

    assert result["error"] == "unknown_fields"
    assert result["unknown"] == ["nonsense"]
    # Valid set is the union of HARD/SOFT — should include known field names.
    assert "instructions" in result["valid"]
    assert "model" in result["valid"]
    # Mesh was NOT called.
    mc.get_agent_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_agent_config_agent_not_found():
    """A 404 from the mesh becomes {error: agent_not_found}."""
    from src.agent.builtins.operator_tools import read_agent_config

    fake_response = MagicMock()
    fake_response.status_code = 404
    fake_response.text = "Agent 'ghost' not found"

    class FakeHTTPError(Exception):
        def __init__(self):
            super().__init__("404 Not Found")
            self.response = fake_response

    mc = MagicMock()
    mc.get_agent_config = AsyncMock(side_effect=FakeHTTPError())

    result = await read_agent_config("ghost", mesh_client=mc)

    assert result == {"error": "agent_not_found", "agent_id": "ghost"}


@pytest.mark.asyncio
async def test_read_agent_config_blocked_for_non_operator(monkeypatch):
    """Non-operator agents (no ALLOWED_TOOLS) are denied."""
    from src.agent.builtins.operator_tools import read_agent_config

    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    mc = MagicMock()
    mc.get_agent_config = AsyncMock()

    result = await read_agent_config("alpha", mesh_client=mc)

    assert "error" in result
    assert "operator" in result["error"].lower()
    mc.get_agent_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_agent_config_no_mesh_client():
    """Missing mesh_client returns a clear error."""
    from src.agent.builtins.operator_tools import read_agent_config

    result = await read_agent_config("alpha", mesh_client=None)
    assert "error" in result
    assert "mesh_client" in result["error"]


@pytest.mark.asyncio
async def test_read_agent_config_normalizes_field_whitespace():
    """fields=['instructions ', ' soul'] is normalized, not rejected."""
    from src.agent.builtins.operator_tools import read_agent_config

    mc = MagicMock()
    mc.get_agent_config = AsyncMock(
        return_value={
            "agent_id": "alpha",
            "config": {"instructions": "x", "soul": "y"},
        }
    )

    result = await read_agent_config(
        "alpha", fields=["instructions ", " soul"], mesh_client=mc,
    )

    # No error raised; mesh receives the cleaned list.
    assert "error" not in result
    mc.get_agent_config.assert_awaited_once_with(
        "alpha", fields=["instructions", "soul"],
    )


@pytest.mark.asyncio
async def test_get_agent_config_strips_agent_id_from_permissions(tmp_path):
    """End-to-end: /mesh/agents/{id}/config must NOT return permissions.agent_id.

    Regression guard for a round-trip poison: AgentPermissions has an
    ``agent_id`` field; if the read endpoint returns it inside
    ``permissions``, an operator's read→edit cycle would write that key
    back into ``config/permissions.json``. The next
    ``PermissionMatrix.reload()`` then calls
    ``AgentPermissions(agent_id=agent_id, **perms)`` and crashes with
    ``multiple values for keyword argument 'agent_id'``.
    """
    from unittest.mock import patch

    import yaml as yaml_mod
    from httpx import ASGITransport, AsyncClient

    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    agents_path = cfg_dir / "agents.yaml"
    agents_path.write_text(yaml_mod.dump({"agents": {
        "alpha": {"role": "writer", "model": "gpt-4o"},
    }}))
    (cfg_dir / "mesh.yaml").write_text(yaml_mod.dump({"mesh": {"port": 8420}}))
    (cfg_dir / "projects").mkdir()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    router.register_agent("alpha", "http://alpha:8400", [])

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
    )

    try:
        with patch("src.cli.config.AGENTS_FILE", agents_path), \
             patch("src.cli.config.CONFIG_FILE", cfg_dir / "mesh.yaml"), \
             patch("src.cli.config.PROJECTS_DIR", cfg_dir / "projects"):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get(
                    "/mesh/agents/alpha/config",
                    params={"requesting_agent": "operator"},
                    headers={"x-mesh-internal": "1"},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert "permissions" in body["config"]
        # The poison key must not be present.
        assert "agent_id" not in body["config"]["permissions"], (
            "permissions payload still contains agent_id — read→edit "
            "round-trip would corrupt config/permissions.json"
        )
    finally:
        bb.close()
