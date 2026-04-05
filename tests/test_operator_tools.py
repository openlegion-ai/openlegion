"""Tests for operator propose_edit and confirm_edit tools."""
import pytest
from unittest.mock import MagicMock, AsyncMock


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
