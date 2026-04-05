"""Tests for operator tools: propose_edit, confirm_edit, observations, history, create, projects."""
import json
import os
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
async def test_confirm_edit_no_messages_passes():
    """When _messages is None (not injected), provenance check is skipped."""
    from src.agent.builtins.operator_tools import confirm_edit

    mc = MagicMock()
    mc.confirm_config_change = AsyncMock(return_value={"success": True})
    result = await confirm_edit("abc", mesh_client=mc, _messages=None)
    assert result["success"] is True


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


# ── read_agent_history tests ────────────────────────────────


@pytest.mark.asyncio
async def test_read_agent_history_no_mesh_client():
    from src.agent.builtins.operator_tools import read_agent_history

    result = await read_agent_history("writer")
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_read_agent_history_success():
    from src.agent.builtins.operator_tools import read_agent_history

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "agent_id": "writer",
        "entries": [{"time": "10:00", "event": "task_completed"}],
    }

    mc = MagicMock()
    mc.mesh_url = "http://localhost:8420"
    mc._get_with_retry = AsyncMock(return_value=mock_response)

    result = await read_agent_history("writer", period="today", mesh_client=mc)
    assert result["agent_id"] == "writer"
    mc._get_with_retry.assert_awaited_once()
    call_args = mc._get_with_retry.call_args
    assert "/mesh/agents/writer/history" in call_args[0][0]
    assert call_args[1]["params"]["period"] == "today"


@pytest.mark.asyncio
async def test_read_agent_history_error():
    from src.agent.builtins.operator_tools import read_agent_history

    mc = MagicMock()
    mc.mesh_url = "http://localhost:8420"
    mc._get_with_retry = AsyncMock(side_effect=RuntimeError("connection refused"))

    result = await read_agent_history("writer", mesh_client=mc)
    assert "error" in result
    assert "connection refused" in result["error"]


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
async def test_create_agent_no_messages_passes():
    """When _messages is None, provenance check is skipped."""
    from src.agent.builtins.operator_tools import create_agent

    mc = MagicMock()
    mc.create_custom_agent = AsyncMock(
        return_value={"created": True, "agent": "mybot"},
    )
    result = await create_agent(
        "mybot", "helper", "do things",
        mesh_client=mc, _messages=None,
    )
    assert result["created"] is True


# ── list_projects tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_list_projects_no_mesh_client():
    from src.agent.builtins.operator_tools import list_projects

    result = await list_projects()
    assert "error" in result


@pytest.mark.asyncio
async def test_list_projects_success():
    from src.agent.builtins.operator_tools import list_projects

    mc = MagicMock()
    mc.list_projects = AsyncMock(
        return_value={"projects": [{"name": "proj1", "members": ["a1"]}]},
    )
    result = await list_projects(mesh_client=mc)
    assert len(result["projects"]) == 1
    assert result["projects"][0]["name"] == "proj1"


# ── get_project tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_get_project_found():
    from src.agent.builtins.operator_tools import get_project

    mc = MagicMock()
    mc.list_projects = AsyncMock(
        return_value={"projects": [
            {"name": "proj1", "members": ["a1"]},
            {"name": "proj2", "members": ["a2"]},
        ]},
    )
    result = await get_project("proj2", mesh_client=mc)
    assert result["name"] == "proj2"


@pytest.mark.asyncio
async def test_get_project_not_found():
    from src.agent.builtins.operator_tools import get_project

    mc = MagicMock()
    mc.list_projects = AsyncMock(
        return_value={"projects": [{"name": "proj1"}]},
    )
    result = await get_project("nonexistent", mesh_client=mc)
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
