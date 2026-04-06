"""Tests for coordination tools: hand_off, check_inbox, update_status."""

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_mesh_client(agent_id="scout", standalone=False):
    """Create a mock mesh_client with sensible defaults."""
    mc = MagicMock()
    mc.agent_id = agent_id
    mc.is_standalone = standalone
    mc.list_agents = AsyncMock(return_value={})
    mc.write_blackboard = AsyncMock(return_value={"version": 1})
    mc.read_blackboard = AsyncMock(return_value={"value": {"status": "pending"}})
    mc.list_blackboard = AsyncMock(return_value=[])
    mc.delete_blackboard = AsyncMock(return_value={"deleted": True})
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


    @pytest.mark.asyncio
    async def test_hand_off_invalid_json_data_falls_back(self):
        """Invalid JSON in data param is wrapped as {"text": ...}."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}

        result = await hand_off(
            to="analyst",
            summary="research done",
            data="not valid json {{{",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        # Output should have been written with the fallback wrapper
        output_call = mc.write_blackboard.call_args_list[0]
        assert output_call[0][1] == {"text": "not valid json {{{"}

    @pytest.mark.asyncio
    async def test_hand_off_task_write_fails_reports_error(self):
        """When task write fails after output write, error is returned."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        # First write (output) succeeds, second write (task) fails
        mc.write_blackboard.side_effect = [
            {"version": 1},
            Exception("connection lost"),
        ]

        result = await hand_off(
            to="analyst",
            summary="research done",
            data='{"sources": [1]}',
            mesh_client=mc,
        )

        assert "error" in result
        assert "Failed to create task" in result["error"]

    @pytest.mark.asyncio
    async def test_hand_off_invalid_agent_id_format(self):
        """Agent IDs with path traversal characters are rejected."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client()

        result = await hand_off(
            to="../admin",
            summary="malicious",
            mesh_client=mc,
        )

        assert "error" in result
        assert "Invalid agent ID" in result["error"]

    @pytest.mark.asyncio
    async def test_hand_off_sanitizes_summary(self):
        """Summary is sanitized before writing to blackboard."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}

        # Unicode zero-width chars should be stripped by sanitize_for_prompt
        result = await hand_off(
            to="analyst",
            summary="clean\u200btext",  # zero-width space
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        # The task record should have sanitized summary
        task_call = mc.write_blackboard.call_args_list[0]
        written_summary = task_call[0][1]["summary"]
        assert "\u200b" not in written_summary

    @pytest.mark.asyncio
    async def test_hand_off_list_agents_fails_proceeds(self):
        """When list_agents fails but ID format is valid, hand_off proceeds."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.side_effect = Exception("mesh timeout")

        result = await hand_off(
            to="analyst",
            summary="research done",
            mesh_client=mc,
        )

        assert result["handed_off"] is True

    @pytest.mark.asyncio
    async def test_hand_off_sets_ttl(self):
        """Both output and task writes include a 24h TTL."""
        from src.agent.builtins.coordination_tool import _HANDOFF_TTL, hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}

        result = await hand_off(
            to="analyst",
            summary="research done",
            data='{"sources": [1,2,3]}',
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        # Two writes: output data then task record
        assert mc.write_blackboard.call_count == 2
        for call in mc.write_blackboard.call_args_list:
            assert call.kwargs.get("ttl") == _HANDOFF_TTL


class TestHandOffStandalone:
    """Tests for hand_off from standalone agents (Operator → project agent)."""

    @pytest.mark.asyncio
    async def test_standalone_hand_off_to_project_agent(self):
        """Standalone agent can hand off to a project agent with correct scoping."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="operator", standalone=True)
        mc.introspect = AsyncMock(return_value={
            "fleet": [
                {"id": "operator", "role": "operator"},
                {"id": "researcher", "role": "researcher"},
            ],
            "agent_projects": {"researcher": "my-project"},
        })

        result = await hand_off(
            to="researcher",
            summary="research this topic",
            data='{"topic": "AI agents"}',
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        # Both writes should be project-scoped
        assert mc.write_blackboard.call_count == 2
        calls = [c[0][0] for c in mc.write_blackboard.call_args_list]
        assert all(k.startswith("projects/my-project/") for k in calls)

    @pytest.mark.asyncio
    async def test_standalone_hand_off_to_standalone_agent_fails(self):
        """Standalone agent cannot hand off to another standalone agent."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="operator", standalone=True)
        mc.introspect = AsyncMock(return_value={
            "fleet": [
                {"id": "operator", "role": "operator"},
                {"id": "orphan", "role": "helper"},
            ],
            "agent_projects": {},  # orphan has no project
        })

        result = await hand_off(
            to="orphan",
            summary="do something",
            mesh_client=mc,
        )

        assert "error" in result
        assert "not assigned to a project" in result["error"]

    @pytest.mark.asyncio
    async def test_standalone_hand_off_target_not_found(self):
        """Standalone agent gets clear error when target doesn't exist."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="operator", standalone=True)
        mc.introspect = AsyncMock(return_value={
            "fleet": [{"id": "operator", "role": "operator"}],
            "agent_projects": {},
        })

        result = await hand_off(
            to="nonexistent",
            summary="do something",
            mesh_client=mc,
        )

        assert "error" in result
        assert "not found" in result["error"]


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


    @pytest.mark.asyncio
    async def test_check_inbox_list_blackboard_fails(self):
        """check_inbox returns error when list_blackboard fails."""
        from src.agent.builtins.coordination_tool import check_inbox

        mc = _make_mesh_client(agent_id="analyst")
        mc.list_blackboard.side_effect = Exception("mesh down")

        result = await check_inbox(mesh_client=mc)

        assert "error" in result
        assert "Failed to check inbox" in result["error"]

    @pytest.mark.asyncio
    async def test_check_inbox_malformed_string_value(self):
        """check_inbox handles entries with string values that aren't JSON."""
        from src.agent.builtins.coordination_tool import check_inbox

        mc = _make_mesh_client(agent_id="analyst")
        mc.list_blackboard.return_value = [
            {"key": "tasks/analyst/ho_1", "value": "just a string"},
        ]

        result = await check_inbox(mesh_client=mc)

        assert result["count"] == 1
        assert result["tasks"][0]["from"] == "unknown"

    @pytest.mark.asyncio
    async def test_check_inbox_non_dict_value(self):
        """check_inbox handles entries with non-dict values (e.g. list)."""
        from src.agent.builtins.coordination_tool import check_inbox

        mc = _make_mesh_client(agent_id="analyst")
        mc.list_blackboard.return_value = [
            {"key": "tasks/analyst/ho_1", "value": [1, 2, 3]},
        ]

        result = await check_inbox(mesh_client=mc)

        assert result["count"] == 1
        assert result["tasks"][0]["from"] == "unknown"


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
    async def test_update_status_write_fails(self):
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="engineer")
        mc.write_blackboard.side_effect = Exception("mesh down")

        result = await update_status(state="working", summary="test", mesh_client=mc)

        assert "error" in result
        assert "Failed to update status" in result["error"]

    @pytest.mark.asyncio
    async def test_update_status_sanitizes_summary(self):
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="engineer")

        await update_status(
            state="working",
            summary="doing\u200bthings",  # zero-width space
            mesh_client=mc,
        )

        written_data = mc.write_blackboard.call_args[0][1]
        assert "\u200b" not in written_data["summary"]

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


class TestCompleteTask:
    @pytest.mark.asyncio
    async def test_complete_task_deletes_and_cleans_output(self):
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="engineer")
        # Simulate existing task record on blackboard
        mc.read_blackboard = AsyncMock(return_value={
            "value": {
                "from": "pm",
                "summary": "implement login",
                "status": "pending",
                "output_key": "output/pm/ho_abc123",
                "ts": 1700000000.0,
            },
        })

        result = await complete_task(
            task_key="tasks/engineer/ho_abc123",
            mesh_client=mc,
        )

        assert result["completed"] is True
        # Task entry should be deleted
        delete_calls = mc.delete_blackboard.call_args_list
        assert any(c[0][0] == "tasks/engineer/ho_abc123" for c in delete_calls)
        # Associated output should also be cleaned up
        assert any(c[0][0] == "output/pm/ho_abc123" for c in delete_calls)

    @pytest.mark.asyncio
    async def test_complete_task_not_found(self):
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="engineer")
        mc.read_blackboard = AsyncMock(return_value=None)

        result = await complete_task(
            task_key="tasks/engineer/ho_nonexistent",
            mesh_client=mc,
        )

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_complete_task_string_value_on_blackboard(self):
        """complete_task handles string values and extracts output_key."""
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="engineer")
        mc.read_blackboard = AsyncMock(return_value={
            "value": '{"from": "pm", "status": "pending", "output_key": "output/pm/ho_abc123"}',
        })

        result = await complete_task(
            task_key="tasks/engineer/ho_abc123",
            mesh_client=mc,
        )

        assert result["completed"] is True
        # Task deleted
        delete_calls = mc.delete_blackboard.call_args_list
        assert any(c[0][0] == "tasks/engineer/ho_abc123" for c in delete_calls)
        # Output cleaned up from parsed string value
        assert any(c[0][0] == "output/pm/ho_abc123" for c in delete_calls)

    @pytest.mark.asyncio
    async def test_complete_task_delete_fails(self):
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="engineer")
        mc.delete_blackboard.side_effect = Exception("mesh down")

        result = await complete_task(
            task_key="tasks/engineer/ho_abc123",
            mesh_client=mc,
        )

        assert "error" in result
        assert "Failed to complete task" in result["error"]

    @pytest.mark.asyncio
    async def test_complete_task_standalone(self):
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(standalone=True)

        result = await complete_task(
            task_key="tasks/x/ho_1",
            mesh_client=mc,
        )

        assert "error" in result
        assert "not assigned" in result["error"]

    @pytest.mark.asyncio
    async def test_complete_task_wrong_agent(self):
        """Agent can only complete tasks in their own inbox."""
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="engineer")

        result = await complete_task(
            task_key="tasks/other-agent/ho_123",
            mesh_client=mc,
        )

        assert "error" in result
        assert "Can only complete your own tasks" in result["error"]

    @pytest.mark.asyncio
    async def test_complete_task_output_cleanup_failure_still_succeeds(self):
        """Task completes successfully even if output cleanup fails."""
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="engineer")
        mc.read_blackboard = AsyncMock(return_value={
            "value": {
                "from": "pm",
                "summary": "implement login",
                "status": "pending",
                "output_key": "output/pm/ho_abc123",
            },
        })
        # Task delete succeeds, but output delete fails
        mc.delete_blackboard = AsyncMock(side_effect=[
            {"deleted": True},           # task delete succeeds
            Exception("output expired"),  # output delete fails
        ])

        result = await complete_task(
            task_key="tasks/engineer/ho_abc123",
            mesh_client=mc,
        )

        # Should still succeed — output cleanup is best-effort
        assert result["completed"] is True
        assert mc.delete_blackboard.call_count == 2
