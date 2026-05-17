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
    # Both names are set so coordination_tool (which now reads
    # ``team_name``) and any back-compat callers that still read
    # ``project_name`` see the same value.
    mc.team_name = None if standalone else "default"
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


class TestHandOffV2OriginPropagation:
    """PR-K' fix 1: ``hand_off`` v2 path must propagate origin to ``create_task``."""

    @pytest.mark.asyncio
    async def test_hand_off_v2_passes_origin_to_create_task(self):
        """v2 hand_off reads current_origin and forwards it to create_task."""
        from src.agent.builtins.coordination_tool import hand_off
        from src.shared.trace import current_origin
        from src.shared.types import MessageOrigin

        mc = _make_mesh_client(agent_id="scout", v2_enabled=True)
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        mc.create_task.return_value = {"id": "task_v2_xyz", "status": "pending"}

        origin = MessageOrigin(kind="human", channel="telegram", user="999")
        token = current_origin.set(origin)
        try:
            result = await hand_off(
                to="analyst",
                summary="enrich the lead",
                mesh_client=mc,
            )
        finally:
            current_origin.reset(token)

        assert result["handed_off"] is True
        # create_task must receive the origin kwarg with the same value.
        mc.create_task.assert_awaited_once()
        ct_kwargs = mc.create_task.call_args.kwargs
        assert ct_kwargs.get("origin") == origin
        # wake_agent retains its existing origin propagation.
        wake_kwargs = mc.wake_agent.call_args.kwargs
        assert wake_kwargs.get("origin") == origin

    @pytest.mark.asyncio
    async def test_hand_off_v2_no_origin_passes_none_to_create_task(self):
        """When current_origin is None, create_task receives origin=None."""
        from src.agent.builtins.coordination_tool import hand_off
        from src.shared.trace import current_origin

        mc = _make_mesh_client(agent_id="scout", v2_enabled=True)
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        mc.create_task.return_value = {"id": "task_v2_xyz", "status": "pending"}

        token = current_origin.set(None)
        try:
            await hand_off(to="analyst", summary="x", mesh_client=mc)
        finally:
            current_origin.reset(token)

        ct_kwargs = mc.create_task.call_args.kwargs
        assert ct_kwargs.get("origin") is None


class TestMeshClientCreateTaskOriginHeader:
    """``mesh_client.create_task`` must inject the X-Origin header when given an origin."""

    @pytest.mark.asyncio
    async def test_create_task_with_origin_sets_origin_header(self):
        """origin kwarg → origin_header() merged into request headers."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.agent.mesh_client import MeshClient
        from src.shared.types import MessageOrigin

        client = MeshClient("http://localhost:8420", "scout")

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "task_x", "status": "pending"}
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        async_client = MagicMock()
        async_client.post = mock_post

        origin = MessageOrigin(kind="human", channel="cli", user="jeff")

        with patch.object(
            client, "_get_client", new=AsyncMock(return_value=async_client),
        ):
            await client.create_task(
                assignee="analyst", title="t", origin=origin,
            )

        call_kwargs = mock_post.call_args.kwargs
        headers = call_kwargs.get("headers") or {}
        # ``origin_header`` writes the header under the ``X-Origin`` key
        # (typically). At minimum: some header value derived from origin
        # is present alongside the trace headers.
        assert any("origin" in k.lower() for k in headers)

    @pytest.mark.asyncio
    async def test_create_task_without_origin_no_origin_header(self):
        """origin omitted → no X-Origin header present (back-compat)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.agent.mesh_client import MeshClient

        client = MeshClient("http://localhost:8420", "scout")

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "task_x", "status": "pending"}
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        async_client = MagicMock()
        async_client.post = mock_post

        with patch.object(
            client, "_get_client", new=AsyncMock(return_value=async_client),
        ):
            await client.create_task(assignee="analyst", title="t")

        headers = mock_post.call_args.kwargs.get("headers") or {}
        assert not any("origin" in k.lower() for k in headers)


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
        being queued. The operator picks it up on its next heartbeat.

        Bug 2 (operator-seam): the contract was flipped so wake failure
        surfaces ``handed_off: false`` + ``task_queued: true`` — the
        write still succeeds, just the wake signal didn't land."""
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

        # Wake failed — handed_off MUST be False so the LLM doesn't lie.
        assert result["handed_off"] is False
        # The durable row still landed; the operator's next heartbeat
        # picks it up.
        assert result["task_queued"] is True
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
    async def test_hand_off_v2_wake_failure_surfaces_error_field(self):
        """Bug G (silent peer hand_off failure): when wake_agent raises
        AFTER the task row was successfully created, the result envelope
        must carry an ``error`` key — not just a soft ``wake_failed=true``
        flag that LLMs routinely paper over with a "task is complete"
        summary in their next reply.

        The durable task row is still persisted in SQLite (caller can
        retry or operator can re-wake), but the envelope is now
        unambiguous: ``handed_off=false``, ``error="wake_failed: ..."``,
        directive ``recovery_hint`` ("RETRY", "DO NOT mark complete").
        """
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout", v2_enabled=True)
        mc.list_agents.return_value = {"page-writer": {"role": "writer"}}
        mc.create_task.return_value = {
            "id": "task_orphan_42", "creator": "scout",
            "assignee": "page-writer", "title": "x", "status": "pending",
        }
        # Wake fails AFTER the task row was created.
        mc.wake_agent.side_effect = Exception("wake failed: 503 service unavailable")

        result = await hand_off(
            to="page-writer",
            summary="brief ready for write",
            mesh_client=mc,
        )

        # Caller MUST NOT think this succeeded.
        assert result["handed_off"] is False
        # Durable row exists — recipient discovery + manual re-wake stay viable.
        assert result["task_queued"] is True
        assert result["wake_failed"] is True
        assert result["task_id"] == "task_orphan_42"
        # Bug G fix: ``error`` field surfaces the failure unambiguously.
        # LLMs trained on tool-use conventions react strongly to ``error``
        # keys; the soft ``wake_failed=true`` flag alone was being skimmed.
        assert "error" in result
        assert "wake_failed" in result["error"]
        assert "page-writer" in result["error"]
        assert "MUST NOT report success" in result["error"]
        # Directive recovery_hint — no "wait for heartbeat" softness.
        assert "RETRY" in result["recovery_hint"]
        assert "DO NOT mark" in result["recovery_hint"]

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
        # The new endpoint was hit. (Bug 3 fix: the blackboard list IS
        # now called once per check_inbox to surface task_event
        # back-edges, but the dedicated task lookup is still the v2
        # endpoint, not the legacy blackboard scan.)
        mc.list_task_inbox.assert_called_once_with("analyst")
        mc.list_blackboard.assert_called_once_with(
            "inbox/analyst/task_event/", global_scope=True,
        )

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
    async def test_update_status_v2_single_active_no_task_id(self):
        """One active task + no task_id → that task is updated transparently."""
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst", v2_enabled=True)
        mc.list_task_inbox.return_value = [
            {"id": "task_only", "status": "pending"},
        ]

        result = await update_status("working", mesh_client=mc)

        assert result["updated"] is True
        assert result["task_id"] == "task_only"
        mc.set_task_status.assert_called_once_with(
            "task_only", "working", blocker_note=None,
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
    async def test_update_status_v2_ambiguous_with_multiple_active(self):
        """2+ active tasks + no task_id → ambiguous_task with active list + hint.

        The active list carries ``{id, title, state}`` entries so the LLM
        can pick a ``task_id`` directly without a follow-up
        ``check_inbox`` call (PR-Q richer-shape augment).
        """
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst", v2_enabled=True)
        mc.list_task_inbox.return_value = [
            {"id": "task_a", "title": "Draft report", "status": "pending"},
            {"id": "task_b", "title": "Review pricing", "status": "working"},
        ]

        result = await update_status("working", mesh_client=mc)

        assert result.get("error") == "ambiguous_task"
        active = result["active"]
        # Augmented shape: list of {id, title, state} dicts (not bare ids).
        assert isinstance(active, list)
        assert {a["id"] for a in active} == {"task_a", "task_b"}
        for a in active:
            assert "id" in a and "title" in a and "state" in a
        by_id = {a["id"]: a for a in active}
        assert by_id["task_a"]["title"] == "Draft report"
        assert by_id["task_a"]["state"] == "pending"
        assert by_id["task_b"]["title"] == "Review pricing"
        assert by_id["task_b"]["state"] == "working"
        assert "task_id" in result["hint"]
        # Critical: no silent set_task_status call against the wrong task.
        mc.set_task_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_status_v2_empty_inbox_returns_legacy_noop(self):
        """Empty inbox → legacy ``{updated: False, ...}`` no-op shape.

        Standalone agents and just-joined agents on a fresh fleet hit
        the empty-inbox case constantly. PR-Q removed the regressed
        ``{"error": "no_active_task"}`` shape so the LLM doesn't see an
        error for the common "no work yet" case.
        """
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst", v2_enabled=True)
        mc.list_task_inbox.return_value = []

        result = await update_status("working", mesh_client=mc)

        # Legacy success-shape no-op — no error key, updated=False.
        assert "error" not in result
        assert result.get("updated") is False
        assert result.get("state") == "working"
        assert "no active tasks" in result.get("reason", "")
        mc.set_task_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_status_v2_multiple_active_with_explicit_task_id(self):
        """2+ active tasks + valid task_id → that task is updated."""
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst", v2_enabled=True)
        mc.list_task_inbox.return_value = [
            {"id": "task_a", "status": "pending"},
            {"id": "task_b", "status": "working"},
        ]

        result = await update_status(
            "done", task_id="task_a", mesh_client=mc,
        )

        assert result["updated"] is True
        assert result["task_id"] == "task_a"
        mc.set_task_status.assert_called_once_with(
            "task_a", "done", blocker_note=None,
        )

    @pytest.mark.asyncio
    async def test_update_status_v2_unknown_task_id_returns_task_not_found(self):
        """Explicit task_id not in inbox → ``task_not_found`` error."""
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst", v2_enabled=True)
        mc.list_task_inbox.return_value = [
            {"id": "task_a", "status": "pending"},
        ]

        result = await update_status(
            "done", task_id="task_missing", mesh_client=mc,
        )

        assert result.get("error") == "task_not_found"
        assert result.get("task_id") == "task_missing"
        mc.set_task_status.assert_not_called()

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


class TestHandOffTitleQuality:
    """Title-length policy on hand_off (Task 6 v2 path).

    Wall-of-text titles in the dashboard came from agents stuffing the
    full instruction into ``summary`` — both the title AND description
    of the resulting task ended up as the same long string. The fix
    splits a long ``summary`` into a derived short title + the original
    long string as description.
    """

    @pytest.mark.asyncio
    async def test_hand_off_short_summary_passes_through(self):
        """Summary ≤ 100 chars: title and description both equal it."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout", v2_enabled=True)
        mc.list_agents.return_value = {"writer": {"role": "writer"}}
        mc.create_task.return_value = {
            "id": "task_abc", "creator": "scout", "assignee": "writer",
            "title": "Draft Q3 launch brief", "status": "pending",
        }

        await hand_off(
            to="writer",
            summary="Draft Q3 launch brief",
            mesh_client=mc,
        )

        kwargs = mc.create_task.call_args.kwargs
        assert kwargs["title"] == "Draft Q3 launch brief"
        # For short summaries, description mirrors the summary by design
        # (preserves legacy contract — recipients still see the same
        # context they used to).
        assert kwargs["description"] == "Draft Q3 launch brief"

    @pytest.mark.asyncio
    async def test_hand_off_long_summary_splits_into_title_and_description(self):
        """Summary > 100 chars: title is short, description is full text."""
        from src.agent.builtins.coordination_tool import hand_off
        from src.shared.task_titles import SHORT_TITLE_TARGET

        mc = _make_mesh_client(agent_id="scout", v2_enabled=True)
        mc.list_agents.return_value = {"writer": {"role": "writer"}}
        mc.create_task.return_value = {
            "id": "task_abc", "creator": "scout", "assignee": "writer",
            "title": "ok", "status": "pending",
        }

        long_summary = (
            "TEST RUN — execute now, do NOT wait for the 08:00 cron. "
            "The brief is ready at briefs/crewai. Topic: CrewAI vs "
            "OpenLegion comparison. Slug: crewai. Path: src/content/"
            "comparison/crewai.md. Please draft the introduction first."
        )
        await hand_off(
            to="writer", summary=long_summary, mesh_client=mc,
        )

        kwargs = mc.create_task.call_args.kwargs
        # Title got compressed.
        assert len(kwargs["title"]) <= SHORT_TITLE_TARGET + 1
        # Description preserves the full original text.
        assert kwargs["description"] == long_summary
        # Title isn't empty / placeholder — it carries some leading
        # signal from the summary so the recipient can tell tasks apart.
        assert kwargs["title"]
        assert kwargs["title"] != "(handoff)"


# ── Bug 4: surface wake_agent failures in hand_off response ──────


class TestHandOffWakeFailureSurfacing:
    """Operator-reported Bug 4: wake_agent failures were silently swallowed.

    Updated for Bug 2 (operator seam) — the prior contract claimed
    ``handed_off: true`` even on wake failure, which made LLMs report
    success to the user while the recipient never woke. The task IS
    still queued in SQLite/blackboard, so we add ``task_queued: true``
    and a ``recovery_hint`` but flip ``handed_off: false`` so the
    caller sees a partial outcome and can decide whether to retry.
    """

    @pytest.mark.asyncio
    async def test_hand_off_returns_wake_failed_when_wake_raises(self):
        """Legacy path: wake failure surfaces ``handed_off=False`` +
        ``task_queued=True`` + ``wake_failed`` + ``wake_error``."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        mc.wake_agent.side_effect = RuntimeError("agent container 503")

        result = await hand_off(
            to="analyst",
            summary="research done",
            mesh_client=mc,
        )

        # Wake failed — the LLM must NOT see handed_off:true here.
        assert result["handed_off"] is False
        # Task IS queued in the durable store.
        assert result["task_queued"] is True
        # Wake failure detail is surfaced.
        assert result["wake_failed"] is True
        assert "agent container 503" in result["wake_error"]
        assert "recovery_hint" in result

    @pytest.mark.asyncio
    async def test_hand_off_success_omits_wake_failure_keys(self):
        """Legacy path: successful wake leaves no wake_failed/wake_error keys."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}

        result = await hand_off(
            to="analyst",
            summary="research done",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        assert "wake_failed" not in result
        assert "wake_error" not in result

    @pytest.mark.asyncio
    async def test_hand_off_wake_error_message_is_truncated(self):
        """Legacy path: wake_error is bounded to 200 chars to limit payload size."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        huge_msg = "x" * 5000
        mc.wake_agent.side_effect = RuntimeError(huge_msg)

        result = await hand_off(
            to="analyst",
            summary="research done",
            mesh_client=mc,
        )

        # Bug 2: handed_off flips to False on wake failure.
        assert result["handed_off"] is False
        assert result["task_queued"] is True
        assert result["wake_failed"] is True
        assert len(result["wake_error"]) <= 200

    @pytest.mark.asyncio
    async def test_hand_off_wake_error_redacts_credentials(self):
        """Wake exception messages frequently quote the failing URL with
        an API key in the query string. The wake_error field is read
        back into the LLM's context, so credentials there leak into
        the model trail. Must be redacted BEFORE truncation.

        Uses an ``sk-`` prefixed token long enough to match the OpenAI/
        Anthropic short-form SECRET_PATTERN (``sk-[A-Za-z0-9]{20,}``).
        """
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        leaky_token = "sk-" + "a" * 40  # matches SECRET_PATTERN
        leaky = (
            f"POST https://internal.example.com/wake?api_key={leaky_token} "
            f"failed with 500"
        )
        mc.wake_agent.side_effect = RuntimeError(leaky)

        result = await hand_off(
            to="analyst",
            summary="research done",
            mesh_client=mc,
        )

        assert result["wake_failed"] is True
        assert leaky_token not in result["wake_error"]
        assert "[REDACTED]" in result["wake_error"]

    @pytest.mark.asyncio
    async def test_hand_off_wake_error_redacts_url_query_param_value(self):
        """Codex P1: ``redact_string`` only catches token-shaped secrets
        (sk-…, gho_…, AKIA…) — it leaves arbitrary URL query-param
        values intact even when the key signals a credential (e.g.
        ``?api_key=abc123``). The fix swaps to ``redact_text_with_urls``
        which runs ``redact_url`` on embedded URLs first so query-param
        values get stripped regardless of whether the value matches a
        regex pattern.
        """
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        # Value is short, random — would NOT match SECRET_PATTERNS,
        # but the query-param KEY (``api_key``) is in SENSITIVE_QUERY_PARAMS.
        opaque_value = "abc123random"
        leaky = (
            f"POST https://gateway.example.com/wake?api_key={opaque_value} "
            f"failed with 500"
        )
        mc.wake_agent.side_effect = RuntimeError(leaky)

        result = await hand_off(
            to="analyst",
            summary="research done",
            mesh_client=mc,
        )

        assert result["wake_failed"] is True
        assert opaque_value not in result["wake_error"], (
            f"opaque query-param value leaked: {result['wake_error']!r}"
        )

    @pytest.mark.asyncio
    async def test_hand_off_v2_returns_wake_failed_when_wake_raises(self):
        """V2 path: wake failure flips ``handed_off=False`` and surfaces
        ``task_queued=True`` + ``wake_failed`` + ``wake_error`` +
        ``recovery_hint`` (Bug 2 contract honesty)."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout", v2_enabled=True)
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        mc.create_task.return_value = {"id": "task_v2_abc", "status": "pending"}
        mc.wake_agent.side_effect = RuntimeError("auth token rejected")

        result = await hand_off(
            to="analyst",
            summary="enrich the lead",
            mesh_client=mc,
        )

        assert result["handed_off"] is False
        assert result["task_queued"] is True
        assert result["wake_failed"] is True
        assert "auth token rejected" in result["wake_error"]
        assert "recovery_hint" in result

    @pytest.mark.asyncio
    async def test_hand_off_v2_success_omits_wake_failure_keys(self):
        """V2 path: successful wake leaves no wake_failed/wake_error keys."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout", v2_enabled=True)
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        mc.create_task.return_value = {"id": "task_v2_abc", "status": "pending"}

        result = await hand_off(
            to="analyst",
            summary="enrich the lead",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        assert "wake_failed" not in result
        assert "wake_error" not in result

    @pytest.mark.asyncio
    async def test_hand_off_v2_wake_failure_reports_handed_off_false(self):
        """Pins the new contract: on wake failure ``_hand_off_v2`` MUST
        flip ``handed_off`` to False and emit ``task_queued=True``
        alongside the wake_failed signal. Pre-fix this returned
        ``handed_off:true`` AND ``wake_failed:true`` — the LLM picked up
        ``handed_off`` and reported success while the recipient never
        actually woke (Bug 2)."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout", v2_enabled=True)
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        mc.create_task.return_value = {"id": "task_v2_xyz", "status": "pending"}
        mc.wake_agent.side_effect = RuntimeError("network unreachable")

        result = await hand_off(
            to="analyst",
            summary="run the enrichment",
            mesh_client=mc,
        )

        # The PRIMARY guarantee: handed_off must be False so LLMs
        # don't tell the user the work was delivered.
        assert result["handed_off"] is False, (
            "wake failed but handed_off claims success — LLM will lie to user"
        )
        # task_queued lets the LLM know the work is durable.
        assert result["task_queued"] is True
        # The task_id is still surfaced so an operator can retry/reroute.
        assert result["task_id"] == "task_v2_xyz"
        # Wake-failure details remain available.
        assert result["wake_failed"] is True
        assert "network unreachable" in result["wake_error"]
        # Bug G fix: recovery_hint is now directive ("RETRY", "DO NOT
        # mark complete") rather than soft ("wait for heartbeat") —
        # LLMs were papering over the soft hint and finalizing with
        # "task complete" in their next reply.
        assert "RETRY" in result["recovery_hint"]
        assert "DO NOT mark" in result["recovery_hint"]
        # And the unambiguous ``error`` field LLMs reliably react to.
        assert "wake_failed" in result["error"]
