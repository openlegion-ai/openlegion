"""Tests for operator tools: propose_edit, confirm_edit, observations, history, create, projects."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    """Operator tools require ALLOWED_TOOLS to be set (defence-in-depth guard)."""
    monkeypatch.setenv("ALLOWED_TOOLS", "propose_edit,confirm_edit")


@pytest.mark.asyncio
async def test_propose_edit_blocks_self_modification():
    from src.agent.builtins.operator_tools import propose_edit

    result = await propose_edit(
        "operator", "instructions", "new text", mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_propose_edit_blocks_self_modification_case_insensitive():
    from src.agent.builtins.operator_tools import propose_edit

    result = await propose_edit(
        "Operator", "instructions", "new text", mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_propose_edit_validates_field():
    from src.agent.builtins.operator_tools import propose_edit

    result = await propose_edit(
        "writer", "invalid_field", "val", mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "Invalid field" in result["error"]


@pytest.mark.asyncio
async def test_propose_edit_accepts_interface_field():
    """The 'interface' field should be valid and forwarded to the mesh."""
    from src.agent.builtins.operator_tools import propose_edit

    mc = MagicMock()
    mc.propose_config_change = AsyncMock(
        return_value={"change_id": "iface1", "preview_diff": "..."},
    )
    result = await propose_edit(
        "writer", "interface", "Accepts research, produces notes.",
        mesh_client=mc,
    )
    assert "error" not in result
    assert result["change_id"] == "iface1"
    mc.propose_config_change.assert_awaited_once_with(
        "writer", "interface", "Accepts research, produces notes.",
    )


@pytest.mark.asyncio
async def test_propose_edit_permission_ceiling_can_spawn():
    from src.agent.builtins.operator_tools import propose_edit

    result = await propose_edit(
        "writer", "permissions", {"can_spawn": True}, mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "ceiling" in result["error"].lower()


@pytest.mark.asyncio
async def test_propose_edit_permission_ceiling_can_use_wallet():
    from src.agent.builtins.operator_tools import propose_edit

    result = await propose_edit(
        "writer", "permissions", {"can_use_wallet": True}, mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "ceiling" in result["error"].lower()


@pytest.mark.asyncio
async def test_propose_edit_permission_ceiling_allows_permitted():
    """Permissions within the ceiling should pass through to mesh."""
    from src.agent.builtins.operator_tools import propose_edit

    mc = MagicMock()
    mc.propose_config_change = AsyncMock(
        return_value={"change_id": "abc", "preview_diff": "..."},
    )
    result = await propose_edit(
        "writer", "permissions", {"can_use_browser": True}, mesh_client=mc,
    )
    assert result["change_id"] == "abc"
    mc.propose_config_change.assert_awaited_once_with(
        "writer", "permissions", {"can_use_browser": True},
    )


@pytest.mark.asyncio
async def test_propose_edit_permission_ceiling_allows_artifacts_write():
    """artifacts/* must be inside the ceiling because save_artifact relies on
    that namespace; without it the operator could observe an agent that has
    artifacts/* write but couldn't reproduce that pattern via propose_edit."""
    from src.agent.builtins.operator_tools import propose_edit

    mc = MagicMock()
    mc.propose_config_change = AsyncMock(
        return_value={"change_id": "art1", "preview_diff": "..."},
    )
    result = await propose_edit(
        "writer", "permissions", {"blackboard_write": ["artifacts/*"]},
        mesh_client=mc,
    )
    assert "error" not in result
    assert result["change_id"] == "art1"
    mc.propose_config_change.assert_awaited_once_with(
        "writer", "permissions", {"blackboard_write": ["artifacts/*"]},
    )


@pytest.mark.asyncio
async def test_propose_edit_budget_validation_daily_too_low():
    from src.agent.builtins.operator_tools import propose_edit

    result = await propose_edit(
        "writer", "budget", {"daily_usd": -1, "monthly_usd": 100},
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "daily_usd" in result["error"]


@pytest.mark.asyncio
async def test_propose_edit_budget_validation_daily_too_high():
    from src.agent.builtins.operator_tools import propose_edit

    result = await propose_edit(
        "writer", "budget", {"daily_usd": 5000, "monthly_usd": 100},
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "daily_usd" in result["error"]


@pytest.mark.asyncio
async def test_propose_edit_budget_validation_monthly_too_low():
    from src.agent.builtins.operator_tools import propose_edit

    result = await propose_edit(
        "writer", "budget", {"daily_usd": 1, "monthly_usd": 0.01},
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "monthly_usd" in result["error"]


@pytest.mark.asyncio
async def test_propose_edit_budget_validation_monthly_too_high():
    from src.agent.builtins.operator_tools import propose_edit

    result = await propose_edit(
        "writer", "budget", {"daily_usd": 1, "monthly_usd": 50000},
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "monthly_usd" in result["error"]


@pytest.mark.asyncio
async def test_propose_edit_budget_valid():
    from src.agent.builtins.operator_tools import propose_edit

    mc = MagicMock()
    mc.propose_config_change = AsyncMock(
        return_value={"change_id": "b1", "preview_diff": "budget diff"},
    )
    result = await propose_edit(
        "writer", "budget", {"daily_usd": 5, "monthly_usd": 100},
        mesh_client=mc,
    )
    assert result["change_id"] == "b1"


@pytest.mark.asyncio
async def test_propose_edit_thinking_validation():
    from src.agent.builtins.operator_tools import propose_edit

    result = await propose_edit(
        "writer", "thinking", "ultra", mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "thinking" in result["error"]


@pytest.mark.asyncio
async def test_propose_edit_thinking_valid():
    from src.agent.builtins.operator_tools import propose_edit

    mc = MagicMock()
    mc.propose_config_change = AsyncMock(
        return_value={"change_id": "t1", "preview_diff": "..."},
    )
    result = await propose_edit(
        "writer", "thinking", "high", mesh_client=mc,
    )
    assert result["change_id"] == "t1"


@pytest.mark.asyncio
async def test_propose_edit_success():
    from src.agent.builtins.operator_tools import propose_edit

    mc = MagicMock()
    mc.propose_config_change = AsyncMock(
        return_value={"change_id": "abc", "preview_diff": "..."},
    )
    result = await propose_edit("writer", "instructions", "new text", mesh_client=mc)
    assert result["change_id"] == "abc"


@pytest.mark.asyncio
async def test_propose_edit_no_mesh_client():
    from src.agent.builtins.operator_tools import propose_edit

    result = await propose_edit("writer", "instructions", "new text")
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_propose_edit_mesh_error():
    from src.agent.builtins.operator_tools import propose_edit

    mc = MagicMock()
    mc.propose_config_change = AsyncMock(side_effect=RuntimeError("connection refused"))
    result = await propose_edit("writer", "instructions", "new text", mesh_client=mc)
    assert "error" in result
    assert "connection refused" in result["error"]


@pytest.mark.asyncio
async def test_confirm_edit_provenance_rejection():
    from src.agent.builtins.operator_tools import confirm_edit

    messages = [{"role": "user", "content": "check", "_origin": "system:heartbeat"}]
    result = await confirm_edit("abc", mesh_client=MagicMock(), _messages=messages)
    assert result["error"] == "provenance_check_failed"
    assert "detail" in result


@pytest.mark.asyncio
async def test_confirm_edit_success_with_user_origin():
    from src.agent.builtins.operator_tools import confirm_edit

    mc = MagicMock()
    mc.confirm_config_change = AsyncMock(return_value={"success": True})
    messages = [{"role": "user", "content": "yes do it", "_origin": "user"}]
    result = await confirm_edit("abc", mesh_client=mc, _messages=messages)
    assert result["success"] is True


@pytest.mark.asyncio
async def test_confirm_edit_success_no_origin_legacy():
    """Messages without _origin should be treated as user-originated."""
    from src.agent.builtins.operator_tools import confirm_edit

    mc = MagicMock()
    mc.confirm_config_change = AsyncMock(return_value={"success": True})
    messages = [{"role": "user", "content": "yes"}]
    result = await confirm_edit("abc", mesh_client=mc, _messages=messages)
    assert result["success"] is True


@pytest.mark.asyncio
async def test_confirm_edit_no_mesh_client():
    from src.agent.builtins.operator_tools import confirm_edit

    result = await confirm_edit("abc")
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_confirm_edit_no_messages_fails_closed():
    """When _messages is None, provenance fails closed (security: no bypass via cron/invoke)."""
    from src.agent.builtins.operator_tools import confirm_edit

    mc = MagicMock()
    mc.confirm_config_change = AsyncMock(return_value={"success": True})
    result = await confirm_edit("abc", mesh_client=mc, _messages=None)
    assert result["error"] == "provenance_check_failed"


@pytest.mark.asyncio
async def test_confirm_edit_not_found_error():
    from src.agent.builtins.operator_tools import confirm_edit

    mc = MagicMock()
    mc.confirm_config_change = AsyncMock(
        side_effect=RuntimeError("404 not found"),
    )
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await confirm_edit("abc", mesh_client=mc, _messages=messages)
    assert result["error"] == "change_expired_or_lost"
    assert "propose_edit" in result["detail"]


@pytest.mark.asyncio
async def test_confirm_edit_generic_error():
    from src.agent.builtins.operator_tools import confirm_edit

    mc = MagicMock()
    mc.confirm_config_change = AsyncMock(
        side_effect=RuntimeError("server exploded"),
    )
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await confirm_edit("abc", mesh_client=mc, _messages=messages)
    assert "error" in result
    assert "server exploded" in result["error"]


# ── _last_message_is_user_origin tests ────────────────────


def test_last_message_is_user_origin_true():
    from src.agent.loop import _last_message_is_user_origin

    msgs = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "yes", "_origin": "user"},
    ]
    assert _last_message_is_user_origin(msgs) is True


def test_last_message_is_user_origin_false():
    from src.agent.loop import _last_message_is_user_origin

    msgs = [
        {"role": "user", "content": "heartbeat", "_origin": "system:heartbeat"},
    ]
    assert _last_message_is_user_origin(msgs) is False


def test_last_message_is_user_origin_no_origin_legacy():
    from src.agent.loop import _last_message_is_user_origin

    msgs = [{"role": "user", "content": "hello"}]
    assert _last_message_is_user_origin(msgs) is True


def test_last_message_is_user_origin_empty():
    from src.agent.loop import _last_message_is_user_origin

    assert _last_message_is_user_origin([]) is False


def test_last_message_is_user_origin_no_user_messages():
    from src.agent.loop import _last_message_is_user_origin

    msgs = [{"role": "assistant", "content": "hi"}]
    assert _last_message_is_user_origin(msgs) is False


# ── save_observations tests ─────────────────────────────────


class _FakeWorkspace:
    """Minimal workspace manager stub with a real tmp directory."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path


@pytest.mark.asyncio
async def test_save_observations_no_workspace():
    from src.agent.builtins.operator_tools import save_observations

    result = await save_observations("all good", "stable")
    assert "error" in result
    assert "workspace_manager" in result["error"]


@pytest.mark.asyncio
async def test_save_observations_writes_files(tmp_path):
    from src.agent.builtins.operator_tools import save_observations

    ws = _FakeWorkspace(tmp_path)
    result = await save_observations(
        "5/6 healthy", "stable",
        agents_attention=[{"agent_id": "writer", "issue": "slow", "severity": "low"}],
        notes="all fine",
        workspace_manager=ws,
    )
    assert result["saved"] is True
    assert "timestamp" in result
    assert result["chars"] > 0

    # Check OBSERVATIONS.md was created
    obs_path = tmp_path / "OBSERVATIONS.md"
    assert obs_path.exists()
    content = obs_path.read_text()
    assert "Fleet Observations" in content
    assert "5/6 healthy" in content

    # Parse the JSON inside the markdown
    json_start = content.index("```json\n") + len("```json\n")
    json_end = content.index("\n```", json_start)
    obs_data = json.loads(content[json_start:json_end])
    assert obs_data["fleet_summary"] == "5/6 healthy"
    assert obs_data["cost_trend"] == "stable"
    assert len(obs_data["agents_attention"]) == 1

    # Check OBSERVATIONS_HISTORY.md was created
    history_path = tmp_path / "OBSERVATIONS_HISTORY.md"
    assert history_path.exists()
    history = history_path.read_text()
    assert "5/6 healthy" in history


@pytest.mark.asyncio
async def test_save_observations_appends_history(tmp_path):
    from src.agent.builtins.operator_tools import save_observations

    ws = _FakeWorkspace(tmp_path)
    await save_observations("check 1", "stable", workspace_manager=ws)
    await save_observations("check 2", "up_10pct", workspace_manager=ws)

    history_path = tmp_path / "OBSERVATIONS_HISTORY.md"
    history = history_path.read_text()
    entries = [e for e in history.strip().split("\n---\n") if e.strip()]
    assert len(entries) == 2
    assert "check 1" in entries[0]
    assert "check 2" in entries[1]


@pytest.mark.asyncio
async def test_save_observations_truncates_long_notes(tmp_path):
    from src.agent.builtins.operator_tools import save_observations

    ws = _FakeWorkspace(tmp_path)
    long_notes = "x" * 5000
    result = await save_observations(
        "ok", "stable", notes=long_notes, workspace_manager=ws,
    )
    assert result["saved"] is True
    assert result["chars"] <= 1500


@pytest.mark.asyncio
async def test_save_observations_history_cap(tmp_path):
    from src.agent.builtins.operator_tools import save_observations

    ws = _FakeWorkspace(tmp_path)
    # Write 55 entries, history should cap at 50
    for i in range(55):
        await save_observations(f"check {i}", "stable", workspace_manager=ws)

    history_path = tmp_path / "OBSERVATIONS_HISTORY.md"
    entries = [e for e in history_path.read_text().strip().split("\n---\n") if e.strip()]
    assert len(entries) == 50
    # Should have the last 50 entries (5-54)
    last_entry = json.loads(entries[-1])
    assert last_entry["fleet_summary"] == "check 54"


# ── inspect_agents tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_inspect_agents_no_mesh_client():
    from src.agent.builtins.operator_tools import inspect_agents

    result = await inspect_agents()
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_inspect_agents_summary_returns_roster():
    """No agent_id → roster summary built from list_agents()."""
    from src.agent.builtins.operator_tools import inspect_agents

    mc = MagicMock()
    mc.list_agents = AsyncMock(return_value={
        "writer": {"role": "writer", "capabilities": ["http_request"]},
        "scout":  {"role": "researcher", "capabilities": ["web_search"]},
    })
    result = await inspect_agents(mesh_client=mc)
    assert result["count"] == 2
    names = {a["name"] for a in result["agents"]}
    assert names == {"writer", "scout"}


@pytest.mark.asyncio
async def test_inspect_agents_profile_calls_get_agent_profile():
    from src.agent.builtins.operator_tools import inspect_agents

    mc = MagicMock()
    mc.get_agent_profile = AsyncMock(return_value={
        "agent_id": "writer", "role": "writer",
    })
    result = await inspect_agents("writer", depth="profile", mesh_client=mc)
    assert result["agent_id"] == "writer"
    mc.get_agent_profile.assert_awaited_once_with("writer")


@pytest.mark.asyncio
async def test_inspect_agents_history_calls_get_agent_history():
    from src.agent.builtins.operator_tools import inspect_agents

    mc = MagicMock()
    mc.get_agent_history = AsyncMock(return_value={
        "agent_id": "writer",
        "entries": [{"time": "10:00", "event": "task_completed"}],
    })
    result = await inspect_agents("writer", depth="history", mesh_client=mc)
    assert result["agent_id"] == "writer"
    mc.get_agent_history.assert_awaited_once_with("writer")


@pytest.mark.asyncio
async def test_inspect_agents_error_surfaces():
    from src.agent.builtins.operator_tools import inspect_agents

    mc = MagicMock()
    mc.get_agent_history = AsyncMock(side_effect=RuntimeError("connection refused"))
    result = await inspect_agents("writer", depth="history", mesh_client=mc)
    assert "error" in result
    assert "connection refused" in result["error"]


# PR-J' — stale_threshold_hours parameter on inspect_agents


@pytest.mark.asyncio
async def test_inspect_agents_no_threshold_unchanged_shape():
    """Roster summary without ``stale_threshold_hours`` looks unchanged."""
    from src.agent.builtins.operator_tools import inspect_agents

    mc = MagicMock()
    mc.list_agents = AsyncMock(return_value={
        "writer": {"role": "writer", "capabilities": []},
    })
    # get_agent_stale_tasks should NOT be called when threshold is omitted.
    mc.get_agent_stale_tasks = AsyncMock()

    result = await inspect_agents(mesh_client=mc)
    assert result["count"] == 1
    assert "stale_threshold_hours" not in result
    assert "stale_task_count" not in result["agents"][0]
    mc.get_agent_stale_tasks.assert_not_awaited()


@pytest.mark.asyncio
async def test_inspect_agents_with_threshold_annotates_roster():
    """Setting stale_threshold_hours adds count + ids to each roster entry."""
    from src.agent.builtins.operator_tools import inspect_agents

    mc = MagicMock()
    mc.list_agents = AsyncMock(return_value={
        "writer": {"role": "writer", "capabilities": []},
        "scout":  {"role": "researcher", "capabilities": []},
    })

    async def _fake_stale(agent_id, threshold_hours=24):
        if agent_id == "writer":
            return {
                "agent_id": "writer",
                "threshold_hours": threshold_hours,
                "count": 2,
                "task_ids": ["task_a", "task_b"],
            }
        return {
            "agent_id": agent_id,
            "threshold_hours": threshold_hours,
            "count": 0,
            "task_ids": [],
        }

    mc.get_agent_stale_tasks = AsyncMock(side_effect=_fake_stale)

    result = await inspect_agents(stale_threshold_hours=24, mesh_client=mc)
    assert result["stale_threshold_hours"] == 24
    by_name = {a["name"]: a for a in result["agents"]}
    assert by_name["writer"]["stale_task_count"] == 2
    assert by_name["writer"]["stale_task_ids"] == ["task_a", "task_b"]
    assert by_name["scout"]["stale_task_count"] == 0
    assert by_name["scout"]["stale_task_ids"] == []


@pytest.mark.asyncio
async def test_inspect_agents_threshold_zero_treated_as_omitted():
    """``stale_threshold_hours=0`` (JSON-schema default) means 'not supplied'."""
    from src.agent.builtins.operator_tools import inspect_agents

    mc = MagicMock()
    mc.list_agents = AsyncMock(return_value={
        "writer": {"role": "writer", "capabilities": []},
    })
    mc.get_agent_stale_tasks = AsyncMock()

    result = await inspect_agents(stale_threshold_hours=0, mesh_client=mc)
    assert "stale_threshold_hours" not in result
    mc.get_agent_stale_tasks.assert_not_awaited()


@pytest.mark.asyncio
async def test_inspect_agents_threshold_out_of_range_rejected():
    from src.agent.builtins.operator_tools import inspect_agents

    mc = MagicMock()
    mc.list_agents = AsyncMock(return_value={})

    result = await inspect_agents(stale_threshold_hours=200, mesh_client=mc)
    assert "error" in result
    assert "168" in result["error"]


@pytest.mark.asyncio
async def test_inspect_agents_stale_lookup_failure_degrades_to_zero():
    """Per-agent stale lookup failure should NOT poison the whole roster."""
    from src.agent.builtins.operator_tools import inspect_agents

    mc = MagicMock()
    mc.list_agents = AsyncMock(return_value={
        "writer": {"role": "writer", "capabilities": []},
    })
    mc.get_agent_stale_tasks = AsyncMock(side_effect=RuntimeError("boom"))

    result = await inspect_agents(stale_threshold_hours=24, mesh_client=mc)
    assert result["stale_threshold_hours"] == 24
    entry = result["agents"][0]
    assert entry["stale_task_count"] == 0
    assert entry["stale_task_ids"] == []


# ── create_agent tests ──────────────────────────��───────────


@pytest.mark.asyncio
async def test_create_agent_no_mesh_client():
    from src.agent.builtins.operator_tools import create_agent

    result = await create_agent("mybot", "helper", "do things")
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_create_agent_provenance_rejected():
    from src.agent.builtins.operator_tools import create_agent

    messages = [{"role": "user", "content": "heartbeat", "_origin": "system:heartbeat"}]
    result = await create_agent(
        "mybot", "helper", "do things",
        mesh_client=MagicMock(), _messages=messages,
    )
    assert result["error"] == "provenance_check_failed"


@pytest.mark.asyncio
async def test_create_agent_success():
    from src.agent.builtins.operator_tools import create_agent

    mc = MagicMock()
    mc.create_custom_agent = AsyncMock(
        return_value={"created": True, "agent": "mybot"},
    )
    messages = [{"role": "user", "content": "yes create it", "_origin": "user"}]
    result = await create_agent(
        "mybot", "helper", "do things",
        model="anthropic/claude-sonnet-4-20250514",
        soul="friendly",
        mesh_client=mc, _messages=messages,
    )
    assert result["created"] is True
    mc.create_custom_agent.assert_awaited_once_with(
        "mybot", "helper", "anthropic/claude-sonnet-4-20250514", "do things", "friendly",
    )


@pytest.mark.asyncio
async def test_create_agent_conflict():
    from src.agent.builtins.operator_tools import create_agent

    mc = MagicMock()
    mc.create_custom_agent = AsyncMock(
        side_effect=RuntimeError("409 Conflict"),
    )
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await create_agent(
        "mybot", "helper", "do things",
        mesh_client=mc, _messages=messages,
    )
    assert "already exists" in result["error"]


@pytest.mark.asyncio
async def test_create_agent_no_messages_fails_closed():
    """When _messages is None, provenance fails closed (security: no bypass via cron/invoke)."""
    from src.agent.builtins.operator_tools import create_agent

    mc = MagicMock()
    mc.create_custom_agent = AsyncMock(
        return_value={"created": True, "agent": "mybot"},
    )
    result = await create_agent(
        "mybot", "helper", "do things",
        mesh_client=mc, _messages=None,
    )
    assert result["error"] == "provenance_check_failed"


# ── inspect_projects tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_inspect_projects_no_mesh_client():
    from src.agent.builtins.operator_tools import inspect_projects

    result = await inspect_projects()
    assert "error" in result


@pytest.mark.asyncio
async def test_inspect_projects_names_lists_all():
    from src.agent.builtins.operator_tools import inspect_projects

    mc = MagicMock()
    mc.list_projects = AsyncMock(
        return_value={"projects": [{"name": "proj1", "members": ["a1"]}]},
    )
    result = await inspect_projects(detail="names", mesh_client=mc)
    assert len(result["projects"]) == 1
    assert result["projects"][0]["name"] == "proj1"


@pytest.mark.asyncio
async def test_inspect_projects_status_calls_all_projects_status(monkeypatch):
    """detail='status' calls the v2 endpoint."""
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "1")
    from src.agent.builtins.operator_tools import inspect_projects

    mc = MagicMock()
    mc.all_projects_status = AsyncMock(
        return_value={"projects": [{"project": {"name": "p1"}, "counts": {"active": 2}}]},
    )
    result = await inspect_projects(detail="status", mesh_client=mc)
    assert "projects" in result
    mc.all_projects_status.assert_awaited_once()


@pytest.mark.asyncio
async def test_inspect_projects_status_requires_v2_flag(monkeypatch):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_V2", "0")
    from src.agent.builtins.operator_tools import inspect_projects

    result = await inspect_projects(detail="status", mesh_client=MagicMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_inspect_projects_named_returns_full_detail():
    from src.agent.builtins.operator_tools import inspect_projects

    mc = MagicMock()
    mc.list_projects = AsyncMock(
        return_value={"projects": [
            {"name": "proj1", "members": ["a1"]},
            {"name": "proj2", "members": ["a2"]},
        ]},
    )
    result = await inspect_projects(project_name="proj2", mesh_client=mc)
    assert result["name"] == "proj2"


@pytest.mark.asyncio
async def test_inspect_projects_named_not_found():
    from src.agent.builtins.operator_tools import inspect_projects

    mc = MagicMock()
    mc.list_projects = AsyncMock(
        return_value={"projects": [{"name": "proj1"}]},
    )
    result = await inspect_projects(project_name="nonexistent", mesh_client=mc)
    assert "error" in result
    assert "not found" in result["error"]


# ── create_project tests ─────────────────────────���──────────


@pytest.mark.asyncio
async def test_create_project_provenance_rejected():
    from src.agent.builtins.operator_tools import create_project

    messages = [{"role": "user", "content": "hb", "_origin": "system:heartbeat"}]
    result = await create_project(
        "myproj", "a project",
        mesh_client=MagicMock(), _messages=messages,
    )
    assert result["error"] == "provenance_check_failed"


@pytest.mark.asyncio
async def test_create_project_success():
    from src.agent.builtins.operator_tools import create_project

    mc = MagicMock()
    mc.create_project = AsyncMock(
        return_value={"created": True, "name": "myproj"},
    )
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await create_project(
        "myproj", "a project", agent_ids=["writer"],
        mesh_client=mc, _messages=messages,
    )
    assert result["created"] is True
    mc.create_project.assert_awaited_once_with("myproj", "a project", ["writer"])


# ── add_agents_to_project tests ───────────────────────────���─


@pytest.mark.asyncio
async def test_add_agents_to_project_provenance_rejected():
    from src.agent.builtins.operator_tools import add_agents_to_project

    messages = [{"role": "user", "content": "hb", "_origin": "system:heartbeat"}]
    result = await add_agents_to_project(
        "proj", ["a1"],
        mesh_client=MagicMock(), _messages=messages,
    )
    assert result["error"] == "provenance_check_failed"


@pytest.mark.asyncio
async def test_add_agents_to_project_success():
    from src.agent.builtins.operator_tools import add_agents_to_project

    mc = MagicMock()
    mc.add_agent_to_project = AsyncMock(
        return_value={"added": True, "project": "proj", "agent": "a1"},
    )
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await add_agents_to_project(
        "proj", ["a1", "a2"],
        mesh_client=mc, _messages=messages,
    )
    assert result["project"] == "proj"
    assert len(result["results"]) == 2


@pytest.mark.asyncio
async def test_add_agents_partial_failure():
    from src.agent.builtins.operator_tools import add_agents_to_project

    mc = MagicMock()
    mc.add_agent_to_project = AsyncMock(
        side_effect=[
            {"added": True, "project": "proj", "agent": "a1"},
            RuntimeError("agent not found"),
        ],
    )
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await add_agents_to_project(
        "proj", ["a1", "a2"],
        mesh_client=mc, _messages=messages,
    )
    assert result["results"][0]["added"] is True
    assert "error" in result["results"][1]


# ── remove_agents_from_project tests ────────────────────────


@pytest.mark.asyncio
async def test_remove_agents_from_project_provenance_rejected():
    from src.agent.builtins.operator_tools import remove_agents_from_project

    messages = [{"role": "user", "content": "hb", "_origin": "system:heartbeat"}]
    result = await remove_agents_from_project(
        "proj", ["a1"],
        mesh_client=MagicMock(), _messages=messages,
    )
    assert result["error"] == "provenance_check_failed"


@pytest.mark.asyncio
async def test_remove_agents_from_project_success():
    from src.agent.builtins.operator_tools import remove_agents_from_project

    mc = MagicMock()
    mc.remove_agent_from_project = AsyncMock(
        return_value={"removed": True, "project": "proj", "agent": "a1"},
    )
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await remove_agents_from_project(
        "proj", ["a1"],
        mesh_client=mc, _messages=messages,
    )
    assert result["project"] == "proj"
    assert result["results"][0]["removed"] is True


# ── update_project_context tests ────────────────────────────


@pytest.mark.asyncio
async def test_update_project_context_provenance_rejected():
    from src.agent.builtins.operator_tools import update_project_context

    messages = [{"role": "user", "content": "hb", "_origin": "system:heartbeat"}]
    result = await update_project_context(
        "proj", "new context",
        mesh_client=MagicMock(), _messages=messages,
    )
    assert result["error"] == "provenance_check_failed"


@pytest.mark.asyncio
async def test_update_project_context_success():
    from src.agent.builtins.operator_tools import update_project_context

    mc = MagicMock()
    mc.update_project_context = AsyncMock(
        return_value={"updated": True, "project": "proj"},
    )
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await update_project_context(
        "proj", "new context",
        mesh_client=mc, _messages=messages,
    )
    assert result["updated"] is True


@pytest.mark.asyncio
async def test_update_project_context_no_mesh_client():
    from src.agent.builtins.operator_tools import update_project_context

    result = await update_project_context("proj", "ctx")
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_update_project_context_mesh_error():
    from src.agent.builtins.operator_tools import update_project_context

    mc = MagicMock()
    mc.update_project_context = AsyncMock(
        side_effect=RuntimeError("not found"),
    )
    messages = [{"role": "user", "content": "yes", "_origin": "user"}]
    result = await update_project_context(
        "proj", "ctx",
        mesh_client=mc, _messages=messages,
    )
    assert "error" in result
    assert "not found" in result["error"]


# ── PR 1 — edit_agent and undo_change ───────────────────────


@pytest.mark.asyncio
async def test_edit_agent_soft_field_calls_edit_soft_immediately():
    """instructions is a soft field — must hit edit_soft and skip provenance."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock(return_value={
        "success": True,
        "undo_token": "tok-abc",
        "expires_at": "2026-05-02T00:05:00+00:00",
        "summary": "Updated writer's instructions",
    })
    # No _messages → would normally fail provenance, but soft-edits skip it.
    result = await edit_agent(
        "writer", "instructions", "be punchier",
        reason="user_asked",
        mesh_client=mc,
    )
    assert result["success"] is True
    assert result["applied"] is True
    assert result["undo_token"] == "tok-abc"
    assert "Undo" in result["message"]
    mc.edit_soft.assert_awaited_once_with(
        "writer", "instructions", "be punchier", "user_asked",
    )


@pytest.mark.asyncio
async def test_edit_agent_soft_field_proactive_reason_logs_but_applies():
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock(return_value={
        "success": True, "undo_token": "tok-x", "expires_at": "now",
        "summary": "Updated",
    })
    result = await edit_agent(
        "writer", "soul", "calmer",
        reason="operator_proactive",
        mesh_client=mc,
    )
    assert result["success"] is True
    mc.edit_soft.assert_awaited_once()


@pytest.mark.asyncio
async def test_edit_agent_hard_field_proposes_with_provenance():
    """model is a hard field — must call propose_config_change AFTER provenance."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.propose_config_change = AsyncMock(return_value={
        "change_id": "cid-1", "preview_diff": "...",
    })
    messages = [{"role": "user", "content": "switch to opus", "_origin": "user"}]
    result = await edit_agent(
        "writer", "model", "anthropic/claude-opus-4",
        reason="user_asked",
        mesh_client=mc, _messages=messages,
    )
    assert result["change_id"] == "cid-1"
    assert result["requires_confirmation"] is True
    assert "next_step" in result
    mc.propose_config_change.assert_awaited_once()


@pytest.mark.asyncio
async def test_edit_agent_hard_field_no_provenance_fails():
    """Hard fields keep the provenance gate — no user message → error."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.propose_config_change = AsyncMock(return_value={"change_id": "x"})
    result = await edit_agent(
        "writer", "model", "anthropic/claude-opus",
        mesh_client=mc, _messages=None,
    )
    assert result["error"] == "provenance_check_failed"
    mc.propose_config_change.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_agent_blocks_self_modification():
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "operator", "instructions", "x", mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_edit_agent_invalid_field():
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "writer", "made_up", "x", mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "Invalid field" in result["error"]


@pytest.mark.asyncio
async def test_edit_agent_invalid_reason():
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "writer", "instructions", "x",
        reason="just_because",
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "reason" in result["error"]


@pytest.mark.asyncio
async def test_edit_agent_permission_ceiling_still_enforced():
    """Hard-field permission ceiling still blocks even via edit_agent."""
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "writer", "permissions", {"can_spawn": True},
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "ceiling" in result["error"].lower()


@pytest.mark.asyncio
async def test_edit_agent_no_mesh_client():
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent("writer", "instructions", "x")
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_edit_agent_soft_propagates_mesh_error():
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock(side_effect=RuntimeError("connection refused"))
    result = await edit_agent(
        "writer", "instructions", "x", mesh_client=mc,
    )
    assert "error" in result
    assert "connection refused" in result["error"]


@pytest.mark.asyncio
async def test_undo_change_happy_path():
    from src.agent.builtins.operator_tools import undo_change

    mc = MagicMock()
    mc.undo_change = AsyncMock(return_value={
        "success": True,
        "agent_id": "writer",
        "field": "instructions",
        "restored_value": "# original",
    })
    result = await undo_change("tok-abc", mesh_client=mc)
    assert result["success"] is True
    assert result["restored_value"] == "# original"
    mc.undo_change.assert_awaited_once_with("tok-abc")


@pytest.mark.asyncio
async def test_undo_change_404_maps_to_unavailable():
    from src.agent.builtins.operator_tools import undo_change

    mc = MagicMock()
    mc.undo_change = AsyncMock(side_effect=RuntimeError("404 Not Found"))
    result = await undo_change("tok-x", mesh_client=mc)
    assert result["error"] == "undo_unavailable"
    assert "expired" in result["detail"].lower() or "used" in result["detail"].lower()


@pytest.mark.asyncio
async def test_undo_change_no_token():
    from src.agent.builtins.operator_tools import undo_change

    result = await undo_change("", mesh_client=MagicMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_undo_change_no_mesh_client():
    from src.agent.builtins.operator_tools import undo_change

    result = await undo_change("tok-1")
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


# ── list_pending ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_pending_returns_rows():
    from src.agent.builtins.operator_tools import list_pending

    mc = MagicMock()
    mc.list_pending_actions = AsyncMock(return_value={
        "pending": [
            {"nonce": "n1", "action_kind": "edit", "target_kind": "agent",
             "target_id": "writer", "expires_at": 0, "actor": "operator",
             "summary": "model swap"},
        ],
    })
    result = await list_pending(mesh_client=mc)
    assert result["count"] == 1
    assert result["pending"][0]["nonce"] == "n1"
    mc.list_pending_actions.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_list_pending_empty():
    from src.agent.builtins.operator_tools import list_pending

    mc = MagicMock()
    mc.list_pending_actions = AsyncMock(return_value={"pending": []})
    result = await list_pending(mesh_client=mc)
    assert result["count"] == 0
    assert result["pending"] == []


@pytest.mark.asyncio
async def test_list_pending_no_mesh_client():
    from src.agent.builtins.operator_tools import list_pending

    result = await list_pending()
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_list_pending_blocked_for_non_operator(monkeypatch):
    from src.agent.builtins.operator_tools import list_pending

    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    result = await list_pending(mesh_client=MagicMock())
    assert "error" in result
    assert "operator" in result["error"].lower()


# ── cancel_pending_action ────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_pending_action_success():
    from src.agent.builtins.operator_tools import cancel_pending_action

    mc = MagicMock()
    mc.cancel_pending_action = AsyncMock(return_value={
        "ok": True,
        "nonce": "nonce-1",
        "target_kind": "agent",
        "target_id": "writer",
        "action_kind": "edit",
    })
    result = await cancel_pending_action("nonce-1", mesh_client=mc)
    assert result["success"] is True
    assert result["nonce"] == "nonce-1"
    assert result["target_kind"] == "agent"
    mc.cancel_pending_action.assert_awaited_once_with("nonce-1")


@pytest.mark.asyncio
async def test_cancel_pending_action_404_friendly():
    from src.agent.builtins.operator_tools import cancel_pending_action

    mc = MagicMock()
    mc.cancel_pending_action = AsyncMock(side_effect=RuntimeError("404 Not Found"))
    result = await cancel_pending_action("missing", mesh_client=mc)
    assert result["error"] == "pending_unknown_or_expired"
    assert "expired" in result["detail"].lower() or "not found" in result["detail"].lower()


@pytest.mark.asyncio
async def test_cancel_pending_action_no_nonce():
    from src.agent.builtins.operator_tools import cancel_pending_action

    result = await cancel_pending_action("", mesh_client=MagicMock())
    assert "error" in result
    assert "nonce" in result["error"].lower()


@pytest.mark.asyncio
async def test_cancel_pending_action_no_mesh_client():
    from src.agent.builtins.operator_tools import cancel_pending_action

    result = await cancel_pending_action("nonce-1")
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_cancel_pending_action_blocked_for_non_operator(monkeypatch):
    """Non-operator agents must not be able to cancel pending actions."""
    from src.agent.builtins.operator_tools import cancel_pending_action

    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    result = await cancel_pending_action("nonce-1", mesh_client=MagicMock())
    assert "error" in result
    assert "operator" in result["error"].lower()


# ── archive_audit_before ─────────────────────────────────────


@pytest.mark.asyncio
async def test_archive_audit_before_success():
    from src.agent.builtins.operator_tools import archive_audit_before

    mc = MagicMock()
    mc.archive_audit_before = AsyncMock(return_value={
        "ok": True,
        "archived_count": 12,
        "before_date": "2026-04-01",
    })
    result = await archive_audit_before("2026-04-01", mesh_client=mc)
    assert result["success"] is True
    assert result["archived_count"] == 12
    assert result["before_date"] == "2026-04-01"
    assert "12" in result["message"]
    mc.archive_audit_before.assert_awaited_once_with("2026-04-01")


@pytest.mark.asyncio
async def test_archive_audit_before_zero_count_pluralization():
    from src.agent.builtins.operator_tools import archive_audit_before

    mc = MagicMock()
    mc.archive_audit_before = AsyncMock(return_value={
        "ok": True,
        "archived_count": 0,
        "before_date": "2099-01-01",
    })
    result = await archive_audit_before("2099-01-01", mesh_client=mc)
    assert result["archived_count"] == 0
    assert "0 audit entries" in result["message"]


@pytest.mark.asyncio
async def test_archive_audit_before_singular_pluralization():
    from src.agent.builtins.operator_tools import archive_audit_before

    mc = MagicMock()
    mc.archive_audit_before = AsyncMock(return_value={
        "ok": True,
        "archived_count": 1,
        "before_date": "2026-04-01",
    })
    result = await archive_audit_before("2026-04-01", mesh_client=mc)
    assert "1 audit entry" in result["message"]


@pytest.mark.asyncio
async def test_archive_audit_before_no_date():
    from src.agent.builtins.operator_tools import archive_audit_before

    result = await archive_audit_before("", mesh_client=MagicMock())
    assert "error" in result
    assert "before_date" in result["error"]


@pytest.mark.asyncio
async def test_archive_audit_before_no_mesh_client():
    from src.agent.builtins.operator_tools import archive_audit_before

    result = await archive_audit_before("2026-04-01")
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_archive_audit_before_blocked_for_non_operator(monkeypatch):
    from src.agent.builtins.operator_tools import archive_audit_before

    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    result = await archive_audit_before("2026-04-01", mesh_client=MagicMock())
    assert "error" in result
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_archive_audit_before_propagates_mesh_error():
    from src.agent.builtins.operator_tools import archive_audit_before

    mc = MagicMock()
    mc.archive_audit_before = AsyncMock(side_effect=RuntimeError("boom"))
    result = await archive_audit_before("2026-04-01", mesh_client=mc)
    assert "error" in result
    assert "boom" in result["error"]


@pytest.mark.asyncio
async def test_archive_audit_before_surfaces_truncated_flag():
    """When the mesh hits the per-call cap, the tool surfaces a follow-up hint."""
    from src.agent.builtins.operator_tools import archive_audit_before

    mc = MagicMock()
    mc.archive_audit_before = AsyncMock(return_value={
        "ok": True,
        "archived_count": 100_000,
        "truncated": True,
        "before_date": "2026-04-01",
    })
    result = await archive_audit_before("2026-04-01", mesh_client=mc)
    assert result["truncated"] is True
    assert "rerun" in result["message"].lower()
