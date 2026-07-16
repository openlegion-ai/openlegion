"""Tests for operator tools: inspect, create, manage."""
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    """Operator tools require ALLOWED_TOOLS to be set (defence-in-depth guard)."""
    monkeypatch.setenv("ALLOWED_TOOLS", "edit_agent")


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
    # PR-Q prefilter: report non-zero counts for both agents so the
    # fanout still happens for each (preserves the original assertion).
    mc.get_system_metrics = AsyncMock(return_value={
        "stale_tasks_24h_count": {"writer": 2, "scout": 1},
    })

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
    # PR-Q prefilter: report a non-zero count so the fanout still
    # happens (and we exercise the per-agent failure-handling path).
    mc.get_system_metrics = AsyncMock(return_value={
        "stale_tasks_24h_count": {"writer": 5},
    })

    result = await inspect_agents(stale_threshold_hours=24, mesh_client=mc)
    assert result["stale_threshold_hours"] == 24
    entry = result["agents"][0]
    assert entry["stale_task_count"] == 0
    assert entry["stale_task_ids"] == []


# PR-Q — stale-task fanout prefilter + operator exclusion


@pytest.mark.asyncio
async def test_inspect_agents_prefilter_skips_zero_count_agents():
    """3-agent fleet with counts {a:1, b:0, c:2} → only 2 fanout calls."""
    from src.agent.builtins.operator_tools import inspect_agents

    mc = MagicMock()
    mc.list_agents = AsyncMock(return_value={
        "a": {"role": "writer", "capabilities": []},
        "b": {"role": "writer", "capabilities": []},
        "c": {"role": "writer", "capabilities": []},
    })

    async def _fake_stale(agent_id, threshold_hours=24):
        return {
            "agent_id": agent_id,
            "threshold_hours": threshold_hours,
            "count": 1 if agent_id == "a" else 2,
            "task_ids": [f"{agent_id}_task"],
        }

    mc.get_agent_stale_tasks = AsyncMock(side_effect=_fake_stale)
    mc.get_system_metrics = AsyncMock(return_value={
        "stale_tasks_24h_count": {"a": 1, "b": 0, "c": 2},
    })

    result = await inspect_agents(stale_threshold_hours=24, mesh_client=mc)
    by_name = {a["name"]: a for a in result["agents"]}
    # b had count 0 → prefilter skipped the fanout, attached zero values.
    assert by_name["b"]["stale_task_count"] == 0
    assert by_name["b"]["stale_task_ids"] == []
    # a and c got real fanout calls.
    assert by_name["a"]["stale_task_count"] == 1
    assert by_name["c"]["stale_task_count"] == 2
    # Exactly two fanout calls — b was skipped.
    assert mc.get_agent_stale_tasks.await_count == 2
    awaited_targets = {
        call.args[0] for call in mc.get_agent_stale_tasks.await_args_list
    }
    assert awaited_targets == {"a", "c"}


@pytest.mark.asyncio
async def test_inspect_agents_skips_operator_in_stale_fanout():
    """Operator must never be the target of a stale-task fanout call."""
    from src.agent.builtins.operator_tools import inspect_agents

    mc = MagicMock()
    mc.list_agents = AsyncMock(return_value={
        "operator": {"role": "operator", "capabilities": []},
        "writer": {"role": "writer", "capabilities": []},
    })

    async def _fake_stale(agent_id, threshold_hours=24):
        return {
            "agent_id": agent_id,
            "threshold_hours": threshold_hours,
            "count": 1,
            "task_ids": ["t1"],
        }

    mc.get_agent_stale_tasks = AsyncMock(side_effect=_fake_stale)
    # Operator sits in the metrics dict with non-zero — verify exclusion
    # is by name rather than just by count.
    mc.get_system_metrics = AsyncMock(return_value={
        "stale_tasks_24h_count": {"operator": 99, "writer": 1},
    })

    result = await inspect_agents(stale_threshold_hours=24, mesh_client=mc)
    by_name = {a["name"]: a for a in result["agents"]}
    # Operator still surfaces in the roster but with empty fields.
    assert by_name["operator"]["stale_task_count"] == 0
    assert by_name["operator"]["stale_task_ids"] == []
    # writer fanout happened.
    assert by_name["writer"]["stale_task_count"] == 1
    # Exactly one fanout call — operator was excluded.
    assert mc.get_agent_stale_tasks.await_count == 1
    awaited_targets = {
        call.args[0] for call in mc.get_agent_stale_tasks.await_args_list
    }
    assert awaited_targets == {"writer"}


@pytest.mark.asyncio
async def test_inspect_agents_metrics_failure_falls_back_to_full_fanout():
    """If get_system_metrics fails, fall through to full fanout (defensive)."""
    from src.agent.builtins.operator_tools import inspect_agents

    mc = MagicMock()
    mc.list_agents = AsyncMock(return_value={
        "writer": {"role": "writer", "capabilities": []},
        "scout":  {"role": "researcher", "capabilities": []},
    })

    async def _fake_stale(agent_id, threshold_hours=24):
        return {
            "agent_id": agent_id,
            "threshold_hours": threshold_hours,
            "count": 0,
            "task_ids": [],
        }

    mc.get_agent_stale_tasks = AsyncMock(side_effect=_fake_stale)
    mc.get_system_metrics = AsyncMock(side_effect=RuntimeError("mesh down"))

    result = await inspect_agents(stale_threshold_hours=24, mesh_client=mc)
    assert result["count"] == 2
    # Both agents got fanned out because we couldn't pre-filter.
    assert mc.get_agent_stale_tasks.await_count == 2


# ── create_agent tests ──────────────────────────��───────────


@pytest.mark.asyncio
async def test_create_agent_no_mesh_client():
    from src.agent.builtins.operator_tools import create_agent

    result = await create_agent("mybot", "helper", "do things")
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_create_agent_no_provenance_is_accepted():
    """Provenance gate dropped — create_agent now goes straight through to
    the mesh whether or not _messages carries a user-origin message."""
    from src.agent.builtins.operator_tools import create_agent

    mc = MagicMock()
    mc.create_custom_agent = AsyncMock(
        return_value={"created": True, "agent": "mybot"},
    )
    # No _messages at all — previously a hard reject; now a happy path.
    result = await create_agent(
        "mybot", "helper", "do things",
        mesh_client=mc, _messages=None,
    )
    assert result["created"] is True
    mc.create_custom_agent.assert_awaited_once()


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
async def test_edit_agent_hard_field_applies_immediately():
    """Hard fields now apply IMMEDIATELY via edit_soft (no propose+confirm,
    no provenance gate). Field class on the response is "hard" and the
    undo TTL is 1800s (30 min)."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock(return_value={
        "success": True,
        "undo_token": "tok-hard",
        "expires_at": "2026-05-13T00:30:00+00:00",
        "ttl_seconds": 1800,
        "field_class": "hard",
        "summary": "Updated writer's model",
    })
    messages = [{"role": "user", "content": "switch to opus", "_origin": "user"}]
    result = await edit_agent(
        "writer", "model", "anthropic/claude-opus-4",
        reason="user_asked",
        mesh_client=mc, _messages=messages,
    )
    assert result["success"] is True
    assert result["applied"] is True
    assert result["field_class"] == "hard"
    assert result["ttl_seconds"] == 1800
    assert result["undo_token"] == "tok-hard"
    mc.edit_soft.assert_awaited_once_with(
        "writer", "model", "anthropic/claude-opus-4", "user_asked",
    )


@pytest.mark.asyncio
async def test_edit_agent_hard_field_no_provenance_still_applies():
    """Hard fields no longer require provenance — _messages=None works."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock(return_value={
        "success": True,
        "undo_token": "tok-no-prov",
        "expires_at": "2026-05-13T00:30:00+00:00",
        "ttl_seconds": 1800,
        "field_class": "hard",
        "summary": "Updated writer's model",
    })
    result = await edit_agent(
        "writer", "model", "anthropic/claude-opus",
        mesh_client=mc, _messages=None,
    )
    assert result["applied"] is True
    assert result["field_class"] == "hard"
    mc.edit_soft.assert_awaited_once()


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
    """Hard-field permission ceiling still blocks even via edit_agent.

    can_use_wallet is the sole remaining operator-ungrantable flag (can_spawn
    is now a default-on capability the operator may manage)."""
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "writer", "permissions", {"can_use_wallet": True},
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


# ── PR-L' — heartbeat_schedule soft field ──────────────────────


@pytest.mark.asyncio
async def test_edit_agent_heartbeat_schedule_cron_value():
    """5-field cron value is accepted and routed through edit_soft."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock(return_value={
        "success": True, "undo_token": "tok-hs",
        "expires_at": "2026-05-08T00:05:00+00:00",
        "summary": "Updated writer's heartbeat schedule",
    })
    result = await edit_agent(
        "writer", "heartbeat_schedule", "*/15 * * * *",
        reason="user_asked", mesh_client=mc,
    )
    assert result["success"] is True
    assert result["applied"] is True
    assert result["undo_token"] == "tok-hs"
    mc.edit_soft.assert_awaited_once_with(
        "writer", "heartbeat_schedule", "*/15 * * * *", "user_asked",
    )


@pytest.mark.asyncio
async def test_edit_agent_heartbeat_schedule_interval_value():
    """``every 15m`` shorthand is accepted and routed through edit_soft."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock(return_value={
        "success": True, "undo_token": "tok-hs2",
        "expires_at": "x", "summary": "ok",
    })
    result = await edit_agent(
        "writer", "heartbeat_schedule", "every 15m",
        reason="user_asked", mesh_client=mc,
    )
    assert result["success"] is True
    mc.edit_soft.assert_awaited_once()


@pytest.mark.asyncio
async def test_edit_agent_heartbeat_schedule_rejects_garbage():
    """Free-form values are rejected with a clear validation error.

    ``hourly`` looks "named" but cron.py's ``_is_due`` doesn't honour
    it — accepting silently here would create a soft edit that no-ops
    against the scheduler. Reject early with a usable error message.
    """
    from src.agent.builtins.operator_tools import edit_agent

    for bad in ("garbage", "hourly", "every 0", "* * * *"):
        result = await edit_agent(
            "writer", "heartbeat_schedule", bad,
            mesh_client=MagicMock(),
        )
        assert "error" in result, f"value {bad!r} should have errored"
        assert (
            "schedule" in result["error"].lower()
            or "string" in result["error"].lower()
            or "field" in result["error"].lower()
        )


@pytest.mark.asyncio
async def test_edit_agent_heartbeat_schedule_rejects_six_field_cron():
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "writer", "heartbeat_schedule", "* * * * * *",
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "6-field" in result["error"] or "seconds" in result["error"].lower()


@pytest.mark.asyncio
async def test_edit_agent_heartbeat_schedule_rejects_non_string():
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "writer", "heartbeat_schedule", 900,
        mesh_client=MagicMock(),
    )
    assert "error" in result


@pytest.mark.asyncio
async def test_edit_agent_heartbeat_schedule_blocks_self_modification():
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "operator", "heartbeat_schedule", "*/15 * * * *",
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "operator" in result["error"].lower()


# ── Post-#927 coverage retargeted from deleted propose_edit tests ─────────
#
# PR #927 retired the propose+confirm chain. The shared ``_validate_edit``
# helper still backs ``edit_agent``, so the validation branches the deleted
# ``test_propose_edit_*`` cases used to cover (self-block case-insensitivity,
# interface forwarding, can_use_wallet ceiling, allowed-permissions and
# artifacts_write pass-through, budget bounds, thinking enum) need a home
# against the current entry point.


@pytest.mark.asyncio
async def test_edit_agent_blocks_self_modification_case_insensitive():
    """Self-block is case-insensitive — ``OPERATOR``/``Operator`` reject too."""
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "OPERATOR", "instructions", "new text", mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_edit_agent_accepts_interface_field():
    """``interface`` is a valid soft field and forwards through edit_soft."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock(return_value={
        "success": True,
        "undo_token": "tok-iface",
        "expires_at": "2026-05-13T00:05:00+00:00",
        "ttl_seconds": 300,
        "field_class": "soft",
        "summary": "Updated writer's interface",
    })
    result = await edit_agent(
        "writer", "interface", "Accepts research, produces notes.",
        reason="user_asked", mesh_client=mc,
    )
    assert "error" not in result
    assert result["applied"] is True
    assert result["undo_token"] == "tok-iface"
    mc.edit_soft.assert_awaited_once_with(
        "writer", "interface", "Accepts research, produces notes.", "user_asked",
    )


def test_clamp_to_operator_ceiling_single_source_of_truth():
    """H1: the shared ceiling function (re-used by the operator tool AND the
    mesh edit-soft endpoint) rejects escalations and passes within-ceiling
    edits. This pins the single source of truth behavior."""
    from src.host.permissions import clamp_to_operator_ceiling

    # Escalations rejected. can_use_wallet is the sole operator-ungrantable
    # flag; an out-of-bounds blackboard pattern is still rejected.
    assert clamp_to_operator_ceiling("permissions", {"can_use_wallet": True}) is not None
    assert (
        clamp_to_operator_ceiling("permissions", {"blackboard_write": ["secrets/*"]})
        is not None
    )

    # Within-ceiling edits pass. can_spawn is now a default-on capability the
    # operator may grant (no longer an escalation).
    assert clamp_to_operator_ceiling("permissions", {"can_use_browser": True}) is None
    assert clamp_to_operator_ceiling("permissions", {"can_spawn": True}) is None
    # Signals-only blackboard (Phase-2 unit 4): output/* + artifacts/* left
    # the write ceiling, so granting them now exceeds it.
    assert clamp_to_operator_ceiling("permissions", {"blackboard_write": ["status/*"]}) is None
    assert (
        clamp_to_operator_ceiling("permissions", {"blackboard_write": ["artifacts/*"]})
        is not None
    )
    # False grants and non-permissions fields are not the ceiling's concern.
    assert clamp_to_operator_ceiling("permissions", {"can_spawn": False}) is None
    assert clamp_to_operator_ceiling("model", "anthropic/claude-haiku") is None


@pytest.mark.asyncio
async def test_edit_agent_permission_ceiling_can_use_wallet():
    """can_use_wallet=True is the sole grant blocked by the operator ceiling."""
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "writer", "permissions", {"can_use_wallet": True},
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "ceiling" in result["error"].lower()


@pytest.mark.asyncio
async def test_edit_agent_permission_ceiling_allows_permitted():
    """Permissions beneath the ceiling pass validation and reach edit_soft."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock(return_value={
        "success": True,
        "undo_token": "tok-perm",
        "expires_at": "2026-05-13T00:30:00+00:00",
        "ttl_seconds": 1800,
        "field_class": "hard",
        "summary": "Updated writer's permissions",
    })
    result = await edit_agent(
        "writer", "permissions", {"can_use_browser": True},
        reason="user_asked", mesh_client=mc,
    )
    assert "error" not in result
    assert result["applied"] is True
    assert result["undo_token"] == "tok-perm"
    mc.edit_soft.assert_awaited_once_with(
        "writer", "permissions", {"can_use_browser": True}, "user_asked",
    )


@pytest.mark.asyncio
async def test_edit_agent_permission_ceiling_rejects_artifacts_write():
    """Phase-2 unit 4: ``artifacts/*`` LEFT the blackboard_write ceiling
    (the blackboard is signals-only; artifacts live on the Team Drive), so
    the operator granting it via edit_agent is now rejected before the mesh."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock()
    result = await edit_agent(
        "writer", "permissions", {"blackboard_write": ["artifacts/*"]},
        reason="user_asked", mesh_client=mc,
    )
    assert "error" in result
    assert "ceiling" in result["error"].lower()
    mc.edit_soft.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_agent_budget_daily_too_low():
    """daily_usd below the 0.01 floor is rejected pre-mesh."""
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "writer", "budget", {"daily_usd": -1, "monthly_usd": 100},
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "daily_usd" in result["error"]


@pytest.mark.asyncio
async def test_edit_agent_budget_daily_too_high():
    """daily_usd above the 1000 cap is rejected pre-mesh."""
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "writer", "budget", {"daily_usd": 5000, "monthly_usd": 100},
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "daily_usd" in result["error"]


@pytest.mark.asyncio
async def test_edit_agent_budget_monthly_too_low():
    """monthly_usd below the 0.10 floor is rejected pre-mesh."""
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "writer", "budget", {"daily_usd": 1, "monthly_usd": 0.01},
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "monthly_usd" in result["error"]


@pytest.mark.asyncio
async def test_edit_agent_budget_monthly_too_high():
    """monthly_usd above the 30000 cap is rejected pre-mesh."""
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "writer", "budget", {"daily_usd": 1, "monthly_usd": 50000},
        mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "monthly_usd" in result["error"]


@pytest.mark.asyncio
async def test_edit_agent_budget_valid():
    """A budget inside both bounds passes validation and reaches edit_soft."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock(return_value={
        "success": True,
        "undo_token": "tok-b1",
        "expires_at": "2026-05-13T00:30:00+00:00",
        "ttl_seconds": 1800,
        "field_class": "hard",
        "summary": "Updated writer's budget",
    })
    result = await edit_agent(
        "writer", "budget", {"daily_usd": 5, "monthly_usd": 100},
        reason="user_asked", mesh_client=mc,
    )
    assert result["applied"] is True
    assert result["undo_token"] == "tok-b1"
    mc.edit_soft.assert_awaited_once_with(
        "writer", "budget", {"daily_usd": 5, "monthly_usd": 100}, "user_asked",
    )


@pytest.mark.asyncio
async def test_edit_agent_thinking_invalid():
    """A value outside the off/low/medium/high enum is rejected pre-mesh."""
    from src.agent.builtins.operator_tools import edit_agent

    result = await edit_agent(
        "writer", "thinking", "ultra", mesh_client=MagicMock(),
    )
    assert "error" in result
    assert "thinking" in result["error"]


@pytest.mark.asyncio
async def test_edit_agent_thinking_valid():
    """A valid thinking level passes validation and reaches edit_soft."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock(return_value={
        "success": True,
        "undo_token": "tok-t1",
        "expires_at": "2026-05-13T00:30:00+00:00",
        "ttl_seconds": 1800,
        "field_class": "hard",
        "summary": "Updated writer's thinking",
    })
    result = await edit_agent(
        "writer", "thinking", "high",
        reason="user_asked", mesh_client=mc,
    )
    assert result["applied"] is True
    assert result["undo_token"] == "tok-t1"
    mc.edit_soft.assert_awaited_once_with(
        "writer", "thinking", "high", "user_asked",
    )


# ── max_output_tokens ────────────────────────────────────────


@pytest.mark.asyncio
async def test_edit_agent_max_output_tokens_valid():
    """An in-range integer cap passes validation and reaches edit_soft."""
    from src.agent.builtins.operator_tools import edit_agent

    mc = MagicMock()
    mc.edit_soft = AsyncMock(return_value={
        "success": True,
        "undo_token": "tok-m1",
        "expires_at": "2026-05-13T00:30:00+00:00",
        "ttl_seconds": 1800,
        "field_class": "hard",
        "summary": "Raised translator's output cap",
    })
    result = await edit_agent(
        "translator", "max_output_tokens", 32000,
        reason="user_asked", mesh_client=mc,
    )
    assert result["applied"] is True
    assert result["undo_token"] == "tok-m1"
    mc.edit_soft.assert_awaited_once_with(
        "translator", "max_output_tokens", 32000, "user_asked",
    )


@pytest.mark.asyncio
async def test_edit_agent_max_output_tokens_rejects_non_int():
    """A non-integer (or bool) value is rejected pre-mesh."""
    from src.agent.builtins.operator_tools import edit_agent

    for bad in ("8192", 8192.5, True):
        result = await edit_agent(
            "translator", "max_output_tokens", bad, mesh_client=MagicMock(),
        )
        assert "error" in result
        assert "max_output_tokens" in result["error"]


@pytest.mark.asyncio
async def test_edit_agent_max_output_tokens_rejects_out_of_range():
    """Values below 256 or above 200000 are rejected pre-mesh."""
    from src.agent.builtins.operator_tools import edit_agent

    for bad in (255, 200_001):
        result = await edit_agent(
            "translator", "max_output_tokens", bad, mesh_client=MagicMock(),
        )
        assert "error" in result
        assert "max_output_tokens" in result["error"]


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


# ── workflow_snapshot operator tool ──────────────────────────────


class TestWorkflowSnapshotTool:
    """The operator-only ``workflow_snapshot`` tool is a thin wrapper
    over ``mesh_client.get_workflow_snapshot``: it 404-maps to
    ``{'error': 'not_found'}`` and proxies the dict otherwise."""

    @pytest.mark.asyncio
    async def test_returns_mesh_result(self):
        from src.agent.builtins.operator_tools import workflow_snapshot

        mc = MagicMock()
        snapshot = {
            "root": "task_root",
            "stages": [
                {
                    "task_id": "task_root", "parent_task_id": None,
                    "assignee": "scout", "status": "done",
                    "age_in_state_seconds": 12, "title": "kickoff",
                },
            ],
            "summary": {
                "done": 1, "working": 0, "pending": 0,
                "failed": 0, "blocked": 0, "cancelled": 0, "total": 1,
            },
        }
        mc.get_workflow_snapshot = AsyncMock(return_value=snapshot)

        result = await workflow_snapshot("task_root", mesh_client=mc)
        assert result == snapshot
        mc.get_workflow_snapshot.assert_awaited_once_with("task_root")

    @pytest.mark.asyncio
    async def test_returns_not_found_when_mesh_returns_none(self):
        from src.agent.builtins.operator_tools import workflow_snapshot

        mc = MagicMock()
        mc.get_workflow_snapshot = AsyncMock(return_value=None)
        result = await workflow_snapshot("task_missing", mesh_client=mc)
        assert result == {
            "error": "not_found", "root_task_id": "task_missing",
        }

    @pytest.mark.asyncio
    async def test_blocked_for_non_operator(self, monkeypatch):
        from src.agent.builtins.operator_tools import workflow_snapshot

        monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
        mc = MagicMock()
        mc.get_workflow_snapshot = AsyncMock(return_value={"root": "x"})
        result = await workflow_snapshot("x", mesh_client=mc)
        assert "error" in result
        assert "operator" in result["error"].lower()
        mc.get_workflow_snapshot.assert_not_called()


# ── await_task_event operator tool ───────────────────────────────


class TestAwaitTaskEventTool:
    """``await_task_event`` polls the task's durable status (the
    authoritative tasks-store record via ``mesh_client.get_task``) for a
    terminal transition with exponential backoff. It no longer reads the
    origin-gated back-edge inbox — that left human/cron-driven operator
    handoffs un-awaitable."""

    @pytest.mark.asyncio
    async def test_returns_event_when_already_terminal(self):
        from src.agent.builtins.operator_tools import await_task_event

        mc = MagicMock()
        mc.agent_id = "operator"
        mc.get_task = AsyncMock(return_value={
            "status": "done",
            "title": "stage 1",
            "blocker_note": None,
            "outcome": "accepted",
            "result_summary": "ok",
            "completed_at": 1000,
            "updated_at": 1000,
        })
        result = await await_task_event(
            "task_abc", timeout_s=60, poll_interval_s=1, mesh_client=mc,
        )
        assert "event" in result
        assert result["event"]["kind"] == "task_completed"
        assert result["event"]["task_id"] == "task_abc"
        assert result["event"]["status"] == "done"
        assert result["event"]["summary"] == "ok"

    @pytest.mark.asyncio
    async def test_returns_result_summary(self):
        """The worker's deliverable (persisted on the task row as
        ``result_summary``) surfaces in the event envelope's ``summary``
        field — previously hollow (``""``)."""
        from src.agent.builtins.operator_tools import await_task_event

        mc = MagicMock()
        mc.agent_id = "operator"
        mc.get_task = AsyncMock(return_value={
            "status": "done",
            "title": "t",
            "result_summary": "the deliverable",
            "completed_at": 1,
        })
        result = await await_task_event(
            "task_abc", timeout_s=60, poll_interval_s=1, mesh_client=mc,
        )
        assert result["event"]["summary"] == "the deliverable"

    @pytest.mark.asyncio
    async def test_result_summary_is_sanitized(self):
        """The worker's ``result_summary`` is untrusted text and must pass
        through ``sanitize_for_prompt`` before landing in the event envelope —
        otherwise injected invisible/line-separator chars reach the LLM."""
        from src.agent.builtins.operator_tools import await_task_event
        from src.shared.utils import sanitize_for_prompt

        # U+2028 (LINE SEPARATOR) is rewritten to "\n" by sanitize_for_prompt,
        # so the sanitized form differs from the raw input.
        raw = "deliver\u2028able"
        assert sanitize_for_prompt(raw) != raw

        mc = MagicMock()
        mc.agent_id = "operator"
        mc.get_task = AsyncMock(return_value={
            "status": "done",
            "title": "t",
            "result_summary": raw,
            "completed_at": 1,
        })
        result = await await_task_event(
            "task_abc", timeout_s=60, poll_interval_s=1, mesh_client=mc,
        )
        assert result["event"]["summary"] == sanitize_for_prompt(raw)
        assert result["event"]["summary"] != raw

    @pytest.mark.asyncio
    async def test_returns_timed_out_when_non_terminal(self, monkeypatch):
        import asyncio

        from src.agent.builtins import operator_tools
        from src.agent.builtins.operator_tools import await_task_event

        # Shrink the per-poll budget so a poll runs before the pre-deadline
        # guard short-circuits — otherwise with timeout_s=1 the guard
        # returns before get_task is ever observed.
        monkeypatch.setattr(
            operator_tools, "_AWAIT_TASK_EVENT_POLL_BUDGET_S", 0.2,
        )

        # Short-circuit the inter-poll sleep so the test doesn't wait.
        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        mc = MagicMock()
        mc.agent_id = "operator"
        # Still working → never terminal, so the loop times out.
        mc.get_task = AsyncMock(return_value={"status": "working"})
        # Use the minimum (1s) timeout so the loop exits immediately.
        result = await await_task_event(
            "task_zzz", timeout_s=1, poll_interval_s=1, mesh_client=mc,
        )
        assert result.get("timed_out") is True
        assert result["task_id"] == "task_zzz"
        assert result["waited_seconds"] == 1
        assert result["last_status_seen"] == "working"

    @pytest.mark.asyncio
    async def test_timeout_capped_under_streaming_idle(self):
        """``timeout_s`` is clamped to the demoted cap (kept under the 120s
        streaming idle timeout — Phase 1d delegate-and-release). We pin the
        module-level cap constant and confirm a terminal record returns
        immediately via the early-return path."""
        from src.agent.builtins.operator_tools import (
            _AWAIT_TASK_EVENT_MAX_TIMEOUT_S,
            await_task_event,
        )

        mc = MagicMock()
        mc.agent_id = "operator"
        # Return a terminal failed record on the first poll so the call
        # returns immediately — we just need to validate the internal cap
        # by inspecting it through the early-return path.
        mc.get_task = AsyncMock(return_value={
            "status": "failed",
            "blocker_note": "boom",
            "updated_at": 0,
        })
        result = await await_task_event(
            "task_y", timeout_s=600, poll_interval_s=1, mesh_client=mc,
        )
        # Immediate event → not timed out, but the cap is documented
        # via the module-level constant — pin the value too.
        assert _AWAIT_TASK_EVENT_MAX_TIMEOUT_S == 90
        assert _AWAIT_TASK_EVENT_MAX_TIMEOUT_S < 120  # under streaming idle
        assert "event" in result
        assert result["event"]["kind"] == "task_failed"

    @pytest.mark.asyncio
    async def test_missing_task_keeps_waiting_then_times_out(self):
        """``get_task`` returning ``None`` (task not yet visible) is not
        terminal — the wait continues and ultimately times out."""
        from src.agent.builtins.operator_tools import await_task_event

        mc = MagicMock()
        mc.agent_id = "operator"
        mc.get_task = AsyncMock(return_value=None)
        # 1s timeout — should return timed_out because the task record is
        # never visible (let alone terminal).
        result = await await_task_event(
            "task_target", timeout_s=1, poll_interval_s=1, mesh_client=mc,
        )
        assert result.get("timed_out") is True
        assert result["task_id"] == "task_target"

    @pytest.mark.asyncio
    async def test_per_poll_timeout_treated_as_transient(self, monkeypatch):
        """A per-poll ``asyncio.TimeoutError`` from ``get_task`` must NOT
        terminate the wait — the watchdog treats it as a transient
        hiccup, sleeps, and polls again. We prove this by raising
        TimeoutError on the first call then returning a terminal record
        on the second."""
        import asyncio as _aio

        from src.agent.builtins import operator_tools
        from src.agent.builtins.operator_tools import await_task_event

        # Short-circuit the inter-poll sleep so the test doesn't wait.
        # ``await_task_event`` does ``import asyncio`` locally; the
        # module is a process singleton, so patching ``asyncio.sleep``
        # directly is the simplest seam.
        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr(_aio, "sleep", _no_sleep)
        # Touch ``operator_tools`` so the import isn't flagged unused
        # (the module ref is still useful for future seams).
        assert operator_tools.await_task_event is await_task_event

        calls = {"n": 0}

        async def flaky_get_task(_task_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _aio.TimeoutError("simulated stuck poll")
            return {"status": "done", "title": "t", "completed_at": 0}

        mc = MagicMock()
        mc.agent_id = "operator"
        mc.get_task = flaky_get_task

        result = await await_task_event(
            "task_t", timeout_s=60, poll_interval_s=1, mesh_client=mc,
        )
        # We polled twice: once timed out (transient), once returned the
        # terminal record. The error envelope path must NOT have fired.
        assert calls["n"] == 2
        assert "error" not in result
        assert "event" in result
        assert result["event"]["kind"] == "task_completed"

    @pytest.mark.asyncio
    async def test_polls_status_not_inbox(self, monkeypatch):
        """Regression: the tool polls the durable task status, NOT the
        origin-gated back-edge inbox. ``list_blackboard`` must never be
        called — the inbox dependency is what left human/cron-driven
        operator handoffs un-awaitable (await returned nothing)."""
        import asyncio as _aio

        from src.agent.builtins.operator_tools import await_task_event

        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr(_aio, "sleep", _no_sleep)

        records = [
            {"status": "working"},
            {"status": "done", "title": "t", "completed_at": 5},
        ]

        async def get_task(_task_id):
            return records.pop(0)

        mc = MagicMock()
        mc.agent_id = "operator"
        mc.get_task = get_task
        mc.list_blackboard = AsyncMock()

        result = await await_task_event(
            "task_pin", timeout_s=60, poll_interval_s=1, mesh_client=mc,
        )
        assert result["event"]["kind"] == "task_completed"
        mc.list_blackboard.assert_not_called()

    @pytest.mark.asyncio
    async def test_deadline_check_skips_poll_when_too_close(self, monkeypatch):
        """Pre-iteration deadline math: if ``deadline - now <=
        _AWAIT_TASK_EVENT_POLL_BUDGET_S``, return ``timed_out`` without
        starting a poll that could overrun the deadline. We monkeypatch
        the constant to be larger than the requested timeout so the
        pre-check fires on the very first iteration."""
        from src.agent.builtins import operator_tools
        from src.agent.builtins.operator_tools import await_task_event

        # Force the per-poll budget higher than the user-supplied
        # timeout so the deadline-pre-check trips on iteration 1.
        monkeypatch.setattr(
            operator_tools, "_AWAIT_TASK_EVENT_POLL_BUDGET_S", 30,
        )

        mc = MagicMock()
        mc.agent_id = "operator"
        # Should NEVER be called — the deadline check returns first.
        mc.get_task = AsyncMock(return_value=None)

        result = await await_task_event(
            "task_short", timeout_s=5, poll_interval_s=1, mesh_client=mc,
        )
        assert result.get("timed_out") is True
        assert result["task_id"] == "task_short"
        mc.get_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_error_envelope_not_empty(
        self, monkeypatch,
    ):
        """Bug 2 fallback: an exception escaping the inner loop body
        (NOT routed through the inner ``except Exception`` that wraps
        ``get_task``) must surface via the outer try/except as a
        non-empty ``{"error": "await_task_event_unexpected: ..."}``
        envelope. An empty body would let the LLM treat the call as
        silent-success and break the awareness loop."""
        from src.agent.builtins import operator_tools
        from src.agent.builtins.operator_tools import await_task_event

        class BoomRecord:
            """Return value from ``get_task``: looks dict-shaped until the
            status read, at which point ``.get`` raises an exception that
            the inner ``except Exception`` block does NOT cover (it wraps
            the ``wait_for(get_task(...))`` call only, not the subsequent
            ``record.get("status")`` access in the loop body)."""

            def __bool__(self):
                return True

            def get(self, *_args, **_kw):
                raise ZeroDivisionError("synthetic boom in loop body")

        # Shrink the per-poll budget so the deadline pre-check
        # (``remaining <= _AWAIT_TASK_EVENT_POLL_BUDGET_S``) does NOT
        # short-circuit to timed_out before we ever call get_task.
        # Default budget is 15s; with timeout_s=5 the pre-check would
        # fire on iteration 1 and never exercise the iteration that
        # raises ZeroDivisionError.
        monkeypatch.setattr(
            operator_tools, "_AWAIT_TASK_EVENT_POLL_BUDGET_S", 1,
        )

        mc = MagicMock()
        mc.agent_id = "operator"
        mc.get_task = AsyncMock(return_value=BoomRecord())

        result = await await_task_event(
            "task_boom", timeout_s=5, poll_interval_s=1, mesh_client=mc,
        )

        assert result is not None
        assert isinstance(result, dict)
        assert result, "outer try/except must produce a non-empty envelope"
        assert "error" in result
        assert "await_task_event_unexpected" in result["error"]
        assert result.get("task_id") == "task_boom"

    @pytest.mark.asyncio
    async def test_cancelled_error_logs_and_re_raises(self):
        """Round-5 strengthening: ``asyncio.CancelledError`` is
        ``BaseException``-derived in 3.8+ and slipped past the broad
        ``except Exception`` arm — the loop's 300s tool-execution
        ceiling could cancel ``await_task_event`` mid-poll and the
        function would return ``None`` (empty body to the LLM).

        Pin the contract: a cancellation during the inner sleep is
        re-raised (the cancelling loop must see the cancel) rather than
        swallowed into an envelope. ``asyncio.wait_for`` will convert
        the propagated ``CancelledError`` into ``TimeoutError`` at the
        outer boundary."""
        import asyncio

        from src.agent.builtins.operator_tools import await_task_event

        async def slow_get_task(_task_id):
            # Sleep long enough that wait_for's timeout fires while we
            # are awaiting get_task — propagates a CancelledError into
            # await_task_event.
            await asyncio.sleep(10)
            return None

        mc = MagicMock()
        mc.agent_id = "operator"
        mc.get_task = slow_get_task

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                await_task_event(
                    "task_cancel",
                    timeout_s=60,
                    poll_interval_s=1,
                    mesh_client=mc,
                ),
                timeout=0.1,
            )


# ── set_agent_goals tests ────────────────────────────────────


def _goals_mesh_client():
    """MeshClient mock for set_agent_goals: Team-store goals endpoints."""
    mc = MagicMock()
    mc.set_agent_goals = AsyncMock(
        return_value={"agent_id": "x", "set": True, "count": 1},
    )
    mc.clear_agent_goals = AsyncMock(
        return_value={"agent_id": "x", "cleared": True, "existed": True},
    )
    return mc


@pytest.mark.asyncio
async def test_set_agent_goals_writes_team_store_record():
    """Valid goals → PUT via mesh_client.set_agent_goals, keyed by agent alone."""
    from src.agent.builtins.operator_tools import set_agent_goals

    mc = _goals_mesh_client()
    result = await set_agent_goals(
        "researcher", ["Find 10 qualified leads per week."], mesh_client=mc,
    )
    assert result == {
        "set": True,
        "agent_id": "researcher",
        "count": 1,
        "note": (
            "Takes effect on the agent's next prompt build (<=5 min cache)."
        ),
    }
    mc.set_agent_goals.assert_awaited_once_with(
        "researcher", ["Find 10 qualified leads per week."],
    )
    mc.clear_agent_goals.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_agent_goals_clear_uses_clear_endpoint():
    """goals=[] clears via clear_agent_goals — same keying, no write."""
    from src.agent.builtins.operator_tools import set_agent_goals

    mc = _goals_mesh_client()
    result = await set_agent_goals("researcher", [], mesh_client=mc)
    assert result == {"cleared": True, "agent_id": "researcher"}
    mc.clear_agent_goals.assert_awaited_once_with("researcher")
    mc.set_agent_goals.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_agent_goals_rejects_operator_target():
    """Operator's own goals live in manage_goals, not the Team store."""
    from src.agent.builtins.operator_tools import set_agent_goals

    mc = _goals_mesh_client()
    result = await set_agent_goals("operator", ["Run the fleet."], mesh_client=mc)
    assert "error" in result
    assert "manage_goals" in result["error"]
    mc.set_agent_goals.assert_not_awaited()
    mc.clear_agent_goals.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_agent_goals_caps_count_and_length():
    """Max 5 goals; each goal must be a non-empty string <=300 chars."""
    from src.agent.builtins.operator_tools import set_agent_goals

    mc = _goals_mesh_client()

    result = await set_agent_goals(
        "researcher", [f"goal {i}" for i in range(6)], mesh_client=mc,
    )
    assert "error" in result
    assert "max" in result["error"].lower()

    result = await set_agent_goals("researcher", ["x" * 301], mesh_client=mc)
    assert "error" in result
    assert "300" in result["error"]

    result = await set_agent_goals("researcher", ["   "], mesh_client=mc)
    assert "error" in result

    result = await set_agent_goals("researcher", [42], mesh_client=mc)
    assert "error" in result

    mc.set_agent_goals.assert_not_awaited()
    mc.clear_agent_goals.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_agent_goals_unknown_agent_surfaces_mesh_404():
    """Unknown target → mesh 404 detail rides the tool's error envelope."""
    from src.agent.builtins.operator_tools import set_agent_goals

    mc = _goals_mesh_client()
    mc.set_agent_goals = AsyncMock(
        side_effect=Exception(
            "Client error '404 Not Found': Agent 'ghost' not found. "
            "Available: scout, writer",
        ),
    )
    result = await set_agent_goals("ghost", ["Do things."], mesh_client=mc)
    assert "error" in result
    assert "not found" in result["error"]
    assert "scout" in result["error"]
    assert "writer" in result["error"]
    mc.clear_agent_goals.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_agent_goals_requires_operator(monkeypatch):
    """Defence-in-depth: non-operator env (no ALLOWED_TOOLS) is refused."""
    from src.agent.builtins.operator_tools import set_agent_goals

    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    mc = _goals_mesh_client()
    result = await set_agent_goals("researcher", ["Goal."], mesh_client=mc)
    assert "error" in result
    mc.set_agent_goals.assert_not_awaited()


# ── set_team_lead (Phase-1 autonomous-team-delivery) ───────────────────


@pytest.mark.asyncio
async def test_set_team_lead_appoints_valid_member():
    from src.agent.builtins.operator_tools import set_team_lead

    mc = MagicMock()
    mc.set_team_lead = AsyncMock(return_value={
        "success": True, "team_name": "squad", "lead_agent_id": "ada",
    })
    result = await set_team_lead("squad", "ada", mesh_client=mc)
    mc.set_team_lead.assert_awaited_once_with("squad", "ada")
    assert result["lead_agent_id"] == "ada"


@pytest.mark.asyncio
async def test_set_team_lead_surfaces_operator_rejection():
    """The store rejects the operator (HTTP 400); the tool surfaces it."""
    from src.agent.builtins.operator_tools import set_team_lead

    mc = MagicMock()
    mc.set_team_lead = AsyncMock(side_effect=Exception(
        "Client error '400 Bad Request': Operator is a system agent and "
        "cannot be a team lead",
    ))
    result = await set_team_lead("squad", "operator", mesh_client=mc)
    assert "error" in result
    assert "system agent" in result["error"]


@pytest.mark.asyncio
async def test_set_team_lead_surfaces_non_member_rejection():
    """A non-member appointment surfaces as an error envelope."""
    from src.agent.builtins.operator_tools import set_team_lead

    mc = MagicMock()
    mc.set_team_lead = AsyncMock(side_effect=Exception(
        "Client error '400 Bad Request': Agent 'ghost' is not a member of "
        "team 'squad' — the lead must be a real team member.",
    ))
    result = await set_team_lead("squad", "ghost", mesh_client=mc)
    assert "error" in result
    assert "not a member" in result["error"]


@pytest.mark.asyncio
async def test_set_team_lead_requires_both_args():
    from src.agent.builtins.operator_tools import set_team_lead

    mc = MagicMock()
    mc.set_team_lead = AsyncMock()
    result = await set_team_lead("squad", "", mesh_client=mc)
    assert "error" in result
    mc.set_team_lead.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_team_lead_no_mesh_client():
    from src.agent.builtins.operator_tools import set_team_lead

    result = await set_team_lead("squad", "ada", mesh_client=None)
    assert "error" in result
    assert "mesh_client" in result["error"].lower()


@pytest.mark.asyncio
async def test_set_team_lead_blocked_for_non_operator(monkeypatch):
    """Defence-in-depth: non-operator env (no ALLOWED_TOOLS) is refused."""
    from src.agent.builtins.operator_tools import set_team_lead

    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    mc = MagicMock()
    mc.set_team_lead = AsyncMock()
    result = await set_team_lead("squad", "ada", mesh_client=mc)
    assert "error" in result
    mc.set_team_lead.assert_not_awaited()
