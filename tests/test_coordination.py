"""Tests for coordination tools: hand_off, check_inbox, update_status."""

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_mesh_client(agent_id="scout", standalone=False, v2_enabled=False):
    """Create a mock mesh_client with sensible defaults.

    ``v2_enabled`` toggles the Task 6 orchestration-tasks path. Default
    False keeps the legacy blackboard route on so the existing test
    suite continues to assert legacy semantics.
    """
    mc = MagicMock()
    mc.agent_id = agent_id
    mc.is_standalone = standalone
    mc.project_name = None if standalone else "default"
    mc.list_agents = AsyncMock(return_value={})
    mc.write_blackboard = AsyncMock(return_value={"version": 1})
    mc.read_blackboard = AsyncMock(return_value={"value": {"status": "pending"}})
    mc.list_blackboard = AsyncMock(return_value=[])
    mc.delete_blackboard = AsyncMock(return_value={"deleted": True})
    mc.wake_agent = AsyncMock(return_value={"woken": True})
    # Task 6 v2 hooks. Default off so legacy tests run unchanged.
    mc.orchestration_v2_enabled = AsyncMock(return_value=v2_enabled)
    mc.create_task = AsyncMock(return_value={
        "id": "task_abc123def456",
        "creator": agent_id,
        "assignee": "analyst",
        "title": "",
        "status": "pending",
    })
    mc.list_task_inbox = AsyncMock(return_value=[])
    mc.set_task_status = AsyncMock(return_value={
        "id": "task_abc123def456", "status": "done",
    })
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
    async def test_hand_off_standalone_cross_project(self):
        """Standalone agent (e.g. operator) can hand off to project-scoped agent."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(standalone=True)
        mc.list_agents.return_value = {
            "analyst": {"url": "http://analyst:8400", "project": "research"},
        }

        result = await hand_off(
            to="analyst",
            summary="analyze this",
            mesh_client=mc,
        )

        assert result.get("handed_off") is True
        assert result["to"] == "analyst"
        # Should write to the target's project scope
        mc.write_blackboard.assert_called_once()
        call_kwargs = mc.write_blackboard.call_args
        assert call_kwargs.kwargs.get("project") == "research"

    @pytest.mark.asyncio
    async def test_hand_off_standalone_fails_closed_on_roster_error(self):
        """Standalone agent fails handoff when roster lookup errors (can't resolve target project)."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(standalone=True)
        mc.list_agents.side_effect = RuntimeError("connection refused")

        result = await hand_off(
            to="analyst",
            summary="analyze this",
            mesh_client=mc,
        )

        assert "error" in result
        assert "roster" in result["error"].lower()
        mc.write_blackboard.assert_not_called()

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


# ── Fix 4: origin propagation in hand_off ───────────────────────


class TestHandOffOriginPropagation:
    @pytest.mark.asyncio
    async def test_hand_off_reads_and_propagates_current_origin(self):
        """hand_off reads current_origin contextvar and passes it to wake_agent."""
        from src.agent.builtins.coordination_tool import hand_off
        from src.shared.trace import current_origin
        from src.shared.types import MessageOrigin

        mc = _make_mesh_client(agent_id="operator")
        mc.list_agents.return_value = {"chef": {"role": "chef"}}

        origin = MessageOrigin(kind="human", channel="whatsapp", user="+1234")
        token = current_origin.set(origin)
        try:
            result = await hand_off(
                to="chef",
                summary="make dinner",
                mesh_client=mc,
            )
        finally:
            current_origin.reset(token)

        assert result["handed_off"] is True
        # wake_agent should have been called with origin kwarg
        mc.wake_agent.assert_awaited_once()
        call_kwargs = mc.wake_agent.call_args
        assert call_kwargs.kwargs.get("origin") == origin

    @pytest.mark.asyncio
    async def test_hand_off_stores_origin_in_task_record(self):
        """Origin is stored in the task_record written to the blackboard."""
        from src.agent.builtins.coordination_tool import hand_off
        from src.shared.trace import current_origin
        from src.shared.types import MessageOrigin

        mc = _make_mesh_client(agent_id="operator")
        mc.list_agents.return_value = {"chef": {"role": "chef"}}

        origin = MessageOrigin(kind="human", channel="telegram", user="99")
        token = current_origin.set(origin)
        try:
            await hand_off(to="chef", summary="do work", mesh_client=mc)
        finally:
            current_origin.reset(token)

        # The task_record write is the last write_blackboard call
        last_call = mc.write_blackboard.call_args_list[-1]
        task_record = last_call.args[1]
        # Origin is persisted as a plain dict so the JSON write doesn't
        # choke on a Pydantic instance.
        assert task_record.get("origin") == origin.model_dump()

    @pytest.mark.asyncio
    async def test_hand_off_stores_typed_origin_as_plain_dict(self):
        """Typed origins must be JSON-serializable when written to blackboard."""
        from src.agent.builtins.coordination_tool import hand_off
        from src.shared.trace import current_origin
        from src.shared.types import MessageOrigin

        mc = _make_mesh_client(agent_id="operator")
        mc.list_agents.return_value = {"chef": {"role": "chef"}}

        origin = MessageOrigin(kind="human", channel="cli", user="jeff")
        token = current_origin.set(origin)
        try:
            await hand_off(to="chef", summary="do work", mesh_client=mc)
        finally:
            current_origin.reset(token)

        task_record = mc.write_blackboard.call_args_list[-1].args[1]
        assert task_record.get("origin") == {
            "kind": "human",
            "channel": "cli",
            "user": "jeff",
        }

    @pytest.mark.asyncio
    async def test_hand_off_no_origin_no_origin_in_task_record(self):
        """When current_origin is None, task_record has no 'origin' key."""
        from src.agent.builtins.coordination_tool import hand_off
        from src.shared.trace import current_origin

        mc = _make_mesh_client(agent_id="operator")
        mc.list_agents.return_value = {"chef": {"role": "chef"}}

        token = current_origin.set(None)
        try:
            await hand_off(to="chef", summary="do work", mesh_client=mc)
        finally:
            current_origin.reset(token)

        last_call = mc.write_blackboard.call_args_list[-1]
        task_record = last_call.args[1]
        assert "origin" not in task_record


class TestOperatorHandoff:
    """Coverage for the project_agent → operator handoff path (Task 0 hotfix)."""

    @pytest.mark.asyncio
    async def test_worker_to_operator_writes_to_global_namespace(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.project_name = "growth"
        mc.is_standalone = False
        mc.list_agents.return_value = {
            "operator": {"url": "http://operator:8400", "scope": "global"},
        }

        result = await hand_off(
            to="operator",
            summary="follow up with the user",
            data='{"note": "ok"}',
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        assert result["task_key"].startswith("global/tasks/operator/")
        assert result["output_key"].startswith("global/output/scout/")
        # Both writes must use global_scope=True (NOT the sender's project scope)
        assert mc.write_blackboard.call_count == 2
        for call in mc.write_blackboard.call_args_list:
            assert call.kwargs.get("global_scope") is True
            assert call.kwargs.get("project") is None

    @pytest.mark.asyncio
    async def test_operator_check_inbox_lists_global_namespace(self):
        from src.agent.builtins.coordination_tool import check_inbox

        mc = _make_mesh_client(agent_id="operator", standalone=True)
        mc.list_blackboard.return_value = [
            {
                "key": "global/tasks/operator/ho_xyz",
                "value": {
                    "from": "scout",
                    "summary": "user wants a recap",
                    "status": "pending",
                    "ts": 1700000000.0,
                },
            },
        ]

        result = await check_inbox(mesh_client=mc)

        assert "error" not in result
        assert result["count"] == 1
        assert result["tasks"][0]["from"] == "scout"
        # Must be called with global_scope=True
        mc.list_blackboard.assert_called_once()
        call = mc.list_blackboard.call_args
        assert call.args[0] == "global/tasks/operator/"
        assert call.kwargs.get("global_scope") is True

    @pytest.mark.asyncio
    async def test_non_operator_standalone_still_blocked(self):
        """Regression: standalone agents that are not the operator still get the standalone error."""
        from src.agent.builtins.coordination_tool import check_inbox

        mc = _make_mesh_client(agent_id="lone-wolf", standalone=True)

        result = await check_inbox(mesh_client=mc)

        assert "error" in result
        assert "not assigned" in result["error"]
        mc.list_blackboard.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_completes_global_task(self):
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="operator", standalone=True)
        mc.read_blackboard = AsyncMock(return_value={
            "value": {
                "from": "scout",
                "summary": "follow up",
                "status": "pending",
                "output_key": "global/output/scout/ho_xyz",
            },
        })

        result = await complete_task(
            task_key="global/tasks/operator/ho_xyz",
            mesh_client=mc,
        )

        assert result["completed"] is True
        # Read on the task must use global_scope=True so the standalone
        # operator's MeshClient does not auto-prefix with project scope.
        mc.read_blackboard.assert_awaited_once()
        read_call = mc.read_blackboard.call_args
        assert read_call.args[0] == "global/tasks/operator/ho_xyz"
        assert read_call.kwargs.get("global_scope") is True
        # Both task delete and output delete must use global_scope=True
        delete_calls = mc.delete_blackboard.call_args_list
        keys_deleted = [c.args[0] for c in delete_calls]
        assert "global/tasks/operator/ho_xyz" in keys_deleted
        assert "global/output/scout/ho_xyz" in keys_deleted
        for call in delete_calls:
            assert call.kwargs.get("global_scope") is True

    @pytest.mark.asyncio
    async def test_operator_complete_global_task_skips_unexpected_output_key(self):
        """A queued operator task cannot make complete_task delete arbitrary global keys."""
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="operator", standalone=True)
        mc.read_blackboard = AsyncMock(return_value={
            "value": {
                "from": "scout",
                "summary": "follow up",
                "status": "pending",
                "output_key": "global/status/operator",
            },
        })

        result = await complete_task(
            task_key="global/tasks/operator/ho_xyz",
            mesh_client=mc,
        )

        assert result["completed"] is True
        deleted = [c.args[0] for c in mc.delete_blackboard.call_args_list]
        assert deleted == ["global/tasks/operator/ho_xyz"]

    @pytest.mark.asyncio
    async def test_hand_off_uses_registry_scope_hint(self):
        """Defense-in-depth: a registry entry with scope=global triggers
        global writes even if the target name is not the literal 'operator'.
        Future-proofs against additional global agents being introduced."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.project_name = "growth"
        mc.is_standalone = False
        # Hypothetical future global agent — registry advertises scope.
        mc.list_agents.return_value = {
            "fleet-monitor": {
                "url": "http://fleet-monitor:8400",
                "scope": "global",
            },
        }

        result = await hand_off(
            to="fleet-monitor",
            summary="alert: agent X failing",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        assert result["task_key"].startswith("global/tasks/fleet-monitor/")
        write_call = mc.write_blackboard.call_args
        assert write_call.kwargs.get("global_scope") is True
        assert write_call.kwargs.get("project") is None

    @pytest.mark.asyncio
    async def test_hand_off_to_operator_when_roster_lookup_fails(self):
        """Resilience: even if list_agents fails, the literal 'operator' name
        still routes the handoff to the global namespace. The roster-failure
        path must not silently fall back to project scope, which would make
        the operator unreachable."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.project_name = "growth"
        mc.is_standalone = False
        mc.list_agents.side_effect = Exception("mesh down")

        result = await hand_off(
            to="operator",
            summary="user wants a recap",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        assert result["task_key"].startswith("global/tasks/operator/")
        write_call = mc.write_blackboard.call_args
        assert write_call.kwargs.get("global_scope") is True

    @pytest.mark.asyncio
    async def test_operator_update_status_writes_global(self):
        """Operator's status update goes to the fleet-global namespace —
        previously it errored out via the standalone gate, leaving status
        updates broken for the only agent that legitimately needs to write
        them at the fleet level."""
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="operator", standalone=True)

        result = await update_status(
            state="working", summary="reviewing fleet", mesh_client=mc,
        )

        assert "error" not in result
        assert result["updated"] is True
        mc.write_blackboard.assert_awaited_once()
        call = mc.write_blackboard.call_args
        assert call.args[0] == "global/status/operator"
        assert call.kwargs.get("global_scope") is True

    @pytest.mark.asyncio
    async def test_hand_off_to_operator_survives_wake_failure(self):
        """Regression: wake_agent failure must not prevent the task from
        being queued. The operator picks it up on its next heartbeat."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.project_name = "growth"
        mc.is_standalone = False
        mc.list_agents.return_value = {
            "operator": {"url": "http://operator:8400", "scope": "global"},
        }
        mc.wake_agent.side_effect = Exception("wake failed: 403")

        result = await hand_off(
            to="operator", summary="follow up", mesh_client=mc,
        )

        # Handoff still succeeds — task is queued, wake is best-effort.
        assert result["handed_off"] is True
        assert result["task_key"].startswith("global/tasks/operator/")
        # The task write happened before the wake attempt.
        mc.write_blackboard.assert_awaited()

    @pytest.mark.asyncio
    async def test_operator_cannot_complete_other_global_inbox(self):
        """Defense: operator can only complete its own global tasks."""
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="operator", standalone=True)

        result = await complete_task(
            task_key="global/tasks/someone-else/ho_1",
            mesh_client=mc,
        )

        assert "error" in result
        assert "your own tasks" in result["error"]

    @pytest.mark.asyncio
    async def test_worker_to_project_peer_unaffected(self):
        """Regression: cross-project worker → worker handoff still uses project scoping."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.project_name = "growth"
        mc.is_standalone = False
        mc.list_agents.return_value = {
            "analyst": {"url": "http://analyst:8400", "project": "research"},
        }

        result = await hand_off(
            to="analyst",
            summary="check this",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        assert result["task_key"].startswith("tasks/analyst/")
        # write_project should be the target's project, NOT global
        write_call = mc.write_blackboard.call_args
        assert write_call.kwargs.get("project") == "research"
        assert write_call.kwargs.get("global_scope") is False


# ── Task 6: orchestration-tasks v2 integration ──────────────────────


class TestCoordinationV2:
    """When ``orchestration_v2_enabled`` is True, coordination_tool
    routes hand_off / check_inbox / update_status / complete_task
    through the durable tasks endpoints instead of the blackboard.
    """

    @pytest.mark.asyncio
    async def test_hand_off_v2_creates_task_row(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout", v2_enabled=True)
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        mc.create_task.return_value = {
            "id": "task_xyz", "creator": "scout", "assignee": "analyst",
            "title": "research", "status": "pending",
        }

        result = await hand_off(
            to="analyst",
            summary="research handoff",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        assert result["to"] == "analyst"
        assert result["task_id"] == "task_xyz"
        assert result["handoff_id"] == "task_xyz"  # legacy alias preserved
        # The legacy blackboard write should NOT have been called.
        mc.write_blackboard.assert_not_called()
        # The new tasks endpoint should have been called instead.
        mc.create_task.assert_called_once()
        call_kwargs = mc.create_task.call_args.kwargs
        assert call_kwargs["assignee"] == "analyst"
        assert "research handoff" in call_kwargs["title"]

    @pytest.mark.asyncio
    async def test_hand_off_v2_with_data_writes_artifact(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout", v2_enabled=True)
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        mc.create_task.return_value = {
            "id": "task_xyz", "creator": "scout", "assignee": "analyst",
            "title": "x", "status": "pending",
        }

        result = await hand_off(
            to="analyst",
            summary="research done",
            data='{"sources": [1,2,3]}',
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        # output_key in result is the artifact ref the data was written to
        assert "output_key" in result
        # Blackboard.write was called for the artifact (separate from the task row)
        mc.write_blackboard.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_inbox_v2_returns_legacy_dict_shape(self):
        from src.agent.builtins.coordination_tool import check_inbox

        mc = _make_mesh_client(agent_id="analyst", v2_enabled=True)
        mc.list_task_inbox.return_value = [
            {
                "id": "task_a", "creator": "scout", "assignee": "analyst",
                "title": "do thing", "status": "pending",
                "artifact_refs": ["output/scout/ho_1"],
                "created_at": 1700000000.0,
            },
        ]

        result = await check_inbox(mesh_client=mc)

        assert result["count"] == 1
        task = result["tasks"][0]
        # Legacy LLM-facing shape preserved
        assert task["key"] == "task_a"
        assert task["task_id"] == "task_a"
        assert task["from"] == "scout"
        assert task["summary"] == "do thing"
        assert task["status"] == "pending"
        assert task["output_key"] == "output/scout/ho_1"
        assert task["ts"] == 1700000000.0
        # The new endpoint was hit, NOT the blackboard list.
        mc.list_task_inbox.assert_called_once_with("analyst")
        mc.list_blackboard.assert_not_called()

    @pytest.mark.asyncio
    async def test_complete_task_v2_transitions_to_done(self):
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="analyst", v2_enabled=True)
        mc.set_task_status.return_value = {
            "id": "task_xyz", "status": "done",
        }

        result = await complete_task("task_xyz", mesh_client=mc)

        assert result["completed"] is True
        assert result["task_id"] == "task_xyz"
        mc.set_task_status.assert_called_once_with("task_xyz", "done")
        # Legacy delete path was NOT taken.
        mc.delete_blackboard.assert_not_called()

    @pytest.mark.asyncio
    async def test_complete_task_v2_strips_legacy_prefix(self):
        """Legacy ``tasks/x/ho_abc`` keys still resolve to the bare id."""
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="analyst", v2_enabled=True)
        mc.set_task_status.return_value = {"id": "ho_abc", "status": "done"}

        await complete_task("tasks/analyst/ho_abc", mesh_client=mc)

        mc.set_task_status.assert_called_once_with("ho_abc", "done")

    @pytest.mark.asyncio
    async def test_update_status_v2_targets_most_recent_task(self):
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst", v2_enabled=True)
        # Two non-terminal tasks; the most-recent (last) is the target.
        mc.list_task_inbox.return_value = [
            {"id": "task_old", "status": "pending"},
            {"id": "task_new", "status": "pending"},
        ]

        result = await update_status("working", mesh_client=mc)

        assert result["updated"] is True
        assert result["task_id"] == "task_new"
        mc.set_task_status.assert_called_once_with(
            "task_new", "working", blocker_note=None,
        )

    @pytest.mark.asyncio
    async def test_update_status_v2_blocker_note_passes_summary(self):
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst", v2_enabled=True)
        mc.list_task_inbox.return_value = [{"id": "task_x", "status": "working"}]

        await update_status("blocked", "waiting on creds", mesh_client=mc)

        mc.set_task_status.assert_called_once_with(
            "task_x", "blocked", blocker_note="waiting on creds",
        )

    @pytest.mark.asyncio
    async def test_v2_probe_failure_falls_back_to_legacy(self):
        """If the v2 probe raises, hand_off falls through to legacy."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        mc.orchestration_v2_enabled = AsyncMock(side_effect=RuntimeError("boom"))

        result = await hand_off(
            to="analyst", summary="legacy fallback", mesh_client=mc,
        )

        assert result["handed_off"] is True
        # Legacy blackboard write happened (no v2 path taken).
        mc.write_blackboard.assert_called_once()
        mc.create_task.assert_not_called()
