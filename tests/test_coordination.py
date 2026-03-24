"""Tests for coordination tools: hand_off, check_inbox, update_status."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_mesh_client(agent_id="scout", standalone=False):
    """Create a mock mesh_client with sensible defaults."""
    mc = MagicMock()
    mc.agent_id = agent_id
    mc.is_standalone = standalone
    mc.list_agents = AsyncMock(return_value={})
    mc.write_blackboard = AsyncMock(return_value={"version": 1})
    mc.list_blackboard = AsyncMock(return_value=[])
    return mc


class TestHandOff:
    @pytest.mark.asyncio
    async def test_hand_off_basic(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}

        result = await hand_off(
            to="analyst",
            summary="research done",
            data='{"sources": [1,2,3]}',
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        assert result["to"] == "analyst"
        assert result["output_key"].startswith("output/scout/")
        # Two writes: one for output data, one for task record
        assert mc.write_blackboard.call_count == 2

    @pytest.mark.asyncio
    async def test_hand_off_no_data(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}

        result = await hand_off(
            to="analyst",
            summary="quick note",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        assert "output_key" not in result
        # Only one write: the task record (no output data)
        assert mc.write_blackboard.call_count == 1

    @pytest.mark.asyncio
    async def test_hand_off_invalid_target(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client()
        mc.list_agents.return_value = {"engineer": {}}

        result = await hand_off(
            to="nonexistent",
            summary="some work",
            mesh_client=mc,
        )

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_hand_off_standalone(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(standalone=True)

        result = await hand_off(
            to="analyst",
            summary="some work",
            mesh_client=mc,
        )

        assert "error" in result
        assert "not assigned" in result["error"]


class TestCheckInbox:
    @pytest.mark.asyncio
    async def test_check_inbox_with_tasks(self):
        from src.agent.builtins.coordination_tool import check_inbox

        mc = _make_mesh_client(agent_id="analyst")
        mc.list_blackboard.return_value = [
            {
                "key": "tasks/analyst/ho_abc123",
                "value": {
                    "from": "scout",
                    "summary": "research done",
                    "status": "pending",
                    "ts": 1700000000.0,
                    "output_key": "output/scout/ho_abc123",
                },
            },
            {
                "key": "tasks/analyst/ho_def456",
                "value": {
                    "from": "writer",
                    "summary": "draft complete",
                    "status": "done",
                    "ts": 1700000001.0,
                },
            },
        ]

        result = await check_inbox(mesh_client=mc)

        # Only the pending task should be returned (done is filtered)
        assert result["count"] == 1
        assert len(result["tasks"]) == 1
        task = result["tasks"][0]
        assert task["from"] == "scout"
        assert task["summary"] == "research done"
        assert task["output_key"] == "output/scout/ho_abc123"

    @pytest.mark.asyncio
    async def test_check_inbox_empty(self):
        from src.agent.builtins.coordination_tool import check_inbox

        mc = _make_mesh_client(agent_id="analyst")
        mc.list_blackboard.return_value = []

        result = await check_inbox(mesh_client=mc)

        assert result["count"] == 0
        assert result["tasks"] == []


class TestUpdateStatus:
    @pytest.mark.asyncio
    async def test_update_status(self):
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="engineer")

        result = await update_status(
            state="working",
            summary="implementing login",
            mesh_client=mc,
        )

        assert result["updated"] is True
        assert result["state"] == "working"

        # Verify the blackboard write
        mc.write_blackboard.assert_called_once()
        call_args = mc.write_blackboard.call_args
        assert call_args[0][0] == "status/engineer"
        written_data = call_args[0][1]
        assert written_data["state"] == "working"
        assert written_data["summary"] == "implementing login"
        assert "ts" in written_data

    @pytest.mark.asyncio
    async def test_update_status_standalone(self):
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(standalone=True)

        result = await update_status(
            state="idle",
            mesh_client=mc,
        )

        assert "error" in result
        assert "not assigned" in result["error"]
