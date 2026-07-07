"""Tests for coordination tools: hand_off, check_inbox, update_status."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest


def _make_mesh_client(agent_id="scout", team_name="default"):
    """Create a mock mesh_client with sensible defaults.

    All coordination tools now route exclusively through the durable
    tasks store. ``team_name=None`` models the operator (the only
    unscoped identity since solo = team-of-one, ratified #5).
    """
    mc = MagicMock()
    mc.agent_id = agent_id
    mc.team_name = team_name
    mc.list_agents = AsyncMock(return_value={})
    mc.write_blackboard = AsyncMock(return_value={"version": 1})
    mc.read_blackboard = AsyncMock(return_value={"value": {"status": "pending"}})
    mc.list_blackboard = AsyncMock(return_value=[])
    # Back-edge task events now come from the Team Threads store via
    # GET /mesh/agents/{id}/task-events (C.3-a) — check_inbox reads this.
    mc.list_inbox_events = AsyncMock(return_value=[])
    mc.delete_blackboard = AsyncMock(return_value={"deleted": True})
    mc.wake_agent = AsyncMock(return_value={"woken": True})
    mc.create_task = AsyncMock(return_value={
        "id": "task_abc123def456",
        "creator": agent_id,
        "assignee": "analyst",
        "title": "",
        "status": "pending",
    })
    mc.list_task_inbox = AsyncMock(return_value=[])
    # Default to a task already in a completable state so complete_task's
    # pre-close status probe is a no-op unless a test overrides it.
    mc.get_task = AsyncMock(return_value={
        "id": "task_abc123def456", "status": "working",
    })
    mc.set_task_status = AsyncMock(return_value={
        "id": "task_abc123def456", "status": "done",
    })
    return mc


class TestHandOff:
    @pytest.mark.asyncio
    async def test_hand_off_emits_handoff_trace(self):
        """Phase 4: a successful handoff records a ``handoff`` trace edge
        (creator → assignee) carrying the new task_id. Best-effort and awaited
        inline; a non-async record_trace (the default MagicMock) is skipped."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        mc.record_trace = AsyncMock()

        result = await hand_off(
            to="analyst",
            summary="dig into the logs",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        mc.record_trace.assert_awaited_once()
        args, kwargs = mc.record_trace.call_args
        assert args[0] == "handoff"
        assert "scout" in kwargs["detail"] and "analyst" in kwargs["detail"]
        assert kwargs["meta"]["task_id"] == "task_abc123def456"
        assert kwargs["meta"]["to"] == "analyst"

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
    async def test_hand_off_unscoped_fails_closed_on_roster_error(self):
        """An unscoped sender (operator) fails handoff when the roster lookup
        errors — it cannot resolve the target's team scope."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="operator", team_name=None)
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
    async def test_hand_off_to_operator_writes_with_global_scope(self):
        """Operator handoffs route to the global namespace (team_id=None)
        on ``create_task``. Codex r1 finding on the legacy-removal PR:
        the v2 collapse needs explicit coverage that the operator's
        fleet-global scope is preserved end-to-end.

        Codex r2: this pin must NOT also set ``scope: "global"`` on the
        operator's registry entry — that would let the routing union
        succeed via arm B (``target_is_global``) even if arm A (the
        literal-name check) is dropped. The whole point is to pin arm
        A independently. The operator entry here has an explicit
        non-global team value so arm B is FALSE, forcing arm A to
        carry the test.
        """
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="strategist")
        # NB: no ``scope: "global"`` here — arm A independence relies on it.
        mc.list_agents.return_value = {"operator": {"team": "ops"}}

        result = await hand_off(
            to="operator",
            summary="status report",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        # The create_task call must carry team_id=None so the row lands
        # in the fleet-global scope. A non-None team here would land
        # the task in the caller's team and operator wouldn't see it.
        kwargs = mc.create_task.call_args.kwargs
        assert kwargs["team_id"] is None, (
            f"operator handoff must write with team_id=None, got "
            f"{kwargs.get('team_id')!r}"
        )

    @pytest.mark.asyncio
    async def test_hand_off_to_operator_wake_403_is_queued_success(self):
        """A worker→operator wake is denied 403 BY DESIGN (the operator
        polls on heartbeat). The task row is persisted, so a queued operator
        handoff is the intended SUCCESS — it must not be classified as
        ``wake_failed`` (which marked the originating task ``failed`` and
        surfaced a scary "403 Forbidden … host.docker.internal" note).
        """
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="dev-publisher")
        mc.list_agents.return_value = {"operator": {"team": "ops"}}
        mc.wake_agent.side_effect = Exception(
            "Client error '403 Forbidden' for url "
            "'http://host.docker.internal:8420/mesh/wake?target=operator'"
        )

        result = await hand_off(
            to="operator", summary="pipeline stalled", mesh_client=mc,
        )

        assert result["handed_off"] is True
        assert result.get("queued_for_heartbeat") is True
        # The expected denial must NOT leak as a failure envelope.
        assert "wake_failed" not in result
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_hand_off_to_operator_non_403_wake_error_still_fails(self):
        """The operator carve-out is narrowed to the BY-DESIGN 403. A genuine
        infra failure (500 / network) on an operator wake must NOT be masked
        as success — it stays visible via the ``wake_failed`` envelope.
        """
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="dev-publisher")
        mc.list_agents.return_value = {"operator": {"team": "ops"}}
        mc.wake_agent.side_effect = Exception(
            "Server error '500 Internal Server Error' for url "
            "'http://host.docker.internal:8420/mesh/wake'"
        )

        result = await hand_off(
            to="operator", summary="pipeline stalled", mesh_client=mc,
        )

        assert result["handed_off"] is False
        assert result.get("wake_failed") is True
        assert "queued_for_heartbeat" not in result

    @pytest.mark.asyncio
    async def test_hand_off_to_worker_wake_failure_still_fails(self):
        """A wake failure to a NON-operator peer is a genuine problem and
        must keep the existing ``wake_failed`` envelope — the operator carve
        out must not swallow real handoff failures between workers.
        """
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"writer": {"team": "scout"}}
        mc.wake_agent.side_effect = Exception("connection refused")

        result = await hand_off(
            to="writer", summary="here you go", mesh_client=mc,
        )

        assert result["handed_off"] is False
        assert result.get("wake_failed") is True
        assert result.get("task_queued") is True
        assert "error" in result

    @pytest.mark.asyncio
    async def test_hand_off_to_global_scope_agent_writes_with_global_scope(self):
        """Forward-compat: any agent with ``scope: "global"`` on the
        registry (not just the literal name "operator") routes with
        team_id=None. Codex r1 flagged that the v2 collapse dropped
        the broader detection — restored with this test as the pin."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="worker")
        # A hypothetical future global agent — e.g. a fleet-wide
        # monitor that isn't called "operator" but has the same
        # cross-team addressability.
        mc.list_agents.return_value = {
            "fleet-monitor": {"scope": "global", "role": "monitor"},
        }

        result = await hand_off(
            to="fleet-monitor",
            summary="anomaly detected",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        kwargs = mc.create_task.call_args.kwargs
        assert kwargs["team_id"] is None, (
            f"scope=global handoff must write with team_id=None, got "
            f"{kwargs.get('team_id')!r}"
        )

    @pytest.mark.asyncio
    async def test_hand_off_to_operator_writes_output_to_global_namespace(self):
        """Operator-bound handoff OUTPUT lands at ``global/output/{sender}/``
        with ``global_scope=True`` — the namespace the operator's read
        carve-out AND the permission model both target (readable by only
        the operator + the author) — not the sender's team scope, which the
        team-less operator cannot read. The recorded artifact_ref is the
        SAME key, so the operator's check_inbox → read_blackboard resolves."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="social-publisher")
        mc.team_name = "social-media"
        mc.list_agents.return_value = {"operator": {"team": "ops"}}

        result = await hand_off(
            to="operator",
            summary="FB audit complete",
            data='{"fields_updated": 3}',
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        args, kwargs = mc.write_blackboard.call_args
        written_key = args[0]
        assert written_key.startswith("global/output/social-publisher/"), (
            f"operator handoff output must land in the global namespace, "
            f"got {written_key!r}"
        )
        assert kwargs.get("global_scope") is True
        # The task points at exactly the key we wrote.
        assert mc.create_task.call_args.kwargs["artifact_refs"] == [written_key]

    @pytest.mark.asyncio
    async def test_hand_off_to_teammate_keeps_output_team_scoped(self):
        """Non-operator handoff output is unchanged: a bare
        ``output/{sender}/`` key written WITHOUT global_scope (the mesh
        client transparently prefixes it to the team namespace)."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.team_name = "research-team"
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}

        result = await hand_off(
            to="analyst",
            summary="findings",
            data='{"x": 1}',
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        args, kwargs = mc.write_blackboard.call_args
        assert args[0].startswith("output/scout/")
        assert not kwargs.get("global_scope", False)

    @pytest.mark.asyncio
    async def test_hand_off_cross_project_routes_to_target_project(self):
        """Worker → worker in a different team: task lands in the
        TARGET's team namespace, not the caller's. Without this the
        recipient's inbox scan (filtered by their own team) would
        never see the task. Codex r1 coverage pin."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.team_name = "research-team"
        mc.list_agents.return_value = {
            "publisher": {"role": "writer", "team": "publishing-team"},
        }

        result = await hand_off(
            to="publisher",
            summary="ship the draft",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        kwargs = mc.create_task.call_args.kwargs
        assert kwargs["team_id"] == "publishing-team", (
            f"cross-team handoff must write into the TARGET's "
            f"team, got {kwargs.get('team_id')!r}"
        )

    @pytest.mark.asyncio
    async def test_hand_off_same_project_uses_callers_team_name(self):
        """Worker → worker in the same team (no explicit
        ``team`` on the registry entry): task lands in the caller's
        own team. The previous test covers cross-team; this one
        pins the default fall-through."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.team_name = "research-team"
        # No ``team`` key on the target — same-team default.
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}

        result = await hand_off(
            to="analyst",
            summary="review findings",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        kwargs = mc.create_task.call_args.kwargs
        assert kwargs["team_id"] == "research-team"

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


class TestCheckInbox:
    @pytest.mark.asyncio
    async def test_check_inbox_empty(self):
        from src.agent.builtins.coordination_tool import check_inbox

        mc = _make_mesh_client(agent_id="analyst")
        mc.list_task_inbox.return_value = []
        mc.list_inbox_events.return_value = []

        result = await check_inbox(mesh_client=mc)

        assert result["count"] == 0
        assert result["tasks"] == []
        assert result["events"] == []
        assert result["event_count"] == 0
        assert result["events_total"] == 0
        assert result["events_truncated"] is False

    @pytest.mark.asyncio
    async def test_check_inbox_caps_events_at_25(self):
        """A flood of completed events is capped at _MAX_INBOX_EVENTS and the
        truncation metadata reflects the full pre-cap count."""
        from src.agent.builtins.coordination_tool import (
            _MAX_INBOX_EVENTS,
            check_inbox,
        )

        mc = _make_mesh_client(agent_id="operator")
        # 40 completed events — more than the cap. Fixture shape mirrors
        # the /mesh/agents/{id}/task-events envelope rows (thread store).
        entries = [
            {
                "kind": "task_completed",
                "task_id": f"task_{i:03d}",
                "recipient": "worker",
                "title": f"task {i}",
                "status": "done",
                "ts": i,  # ascending — higher i = newer
                "summary": f"summary {i}",
            }
            for i in range(40)
        ]
        mc.list_inbox_events.return_value = entries

        result = await check_inbox(mesh_client=mc)

        assert result["event_count"] == _MAX_INBOX_EVENTS
        assert len(result["events"]) == _MAX_INBOX_EVENTS
        assert result["events_total"] == 40
        assert result["events_truncated"] is True
        # Newest informational events survive (ts 39 down to 15).
        kept_ts = {e["ts"] for e in result["events"]}
        assert max(kept_ts) == 39
        assert min(kept_ts) == 40 - _MAX_INBOX_EVENTS  # 15

    @pytest.mark.asyncio
    async def test_check_inbox_retains_actionable_over_completed(self):
        """task_failed / task_blocked events are NEVER dropped even when there
        are far more than 25 completed events; completed events evict first."""
        from src.agent.builtins.coordination_tool import check_inbox

        mc = _make_mesh_client(agent_id="operator")
        completed = [
            {
                "kind": "task_completed",
                "task_id": f"done_{i:03d}",
                "recipient": "worker",
                "title": f"done {i}",
                "status": "done",
                "ts": 1000 + i,
                "summary": "ok",
            }
            for i in range(40)
        ]
        actionable = [
            {
                "kind": "task_failed",
                "task_id": "fail_1",
                "recipient": "worker",
                "title": "broken pipeline",
                "status": "failed",
                "ts": 5,  # deliberately old — must still survive
                "error": "boom",
            },
            {
                "kind": "task_blocked",
                "task_id": "block_1",
                "recipient": "worker",
                "title": "stuck",
                "status": "blocked",
                "ts": 6,
                "blocker_note": "need creds",
            },
        ]
        mc.list_inbox_events.return_value = completed + actionable

        result = await check_inbox(mesh_client=mc)

        kinds = [e["kind"] for e in result["events"]]
        # Both actionable events retained despite being the oldest by ts.
        assert "task_failed" in kinds
        assert "task_blocked" in kinds
        # Actionable events are ordered first (newest actionable first).
        assert kinds[0] == "task_blocked"  # ts 6 > ts 5
        assert kinds[1] == "task_failed"
        task_ids = {e["task_id"] for e in result["events"]}
        assert "fail_1" in task_ids
        assert "block_1" in task_ids
        # Total still reflects the full set; list is capped at 25.
        assert result["events_total"] == 42
        assert result["event_count"] == 25
        assert result["events_truncated"] is True


class TestHandOffOriginPropagation:
    """PR-K' fix 1: ``hand_off`` v2 path must propagate origin to ``create_task``."""

    @pytest.mark.asyncio
    async def test_hand_off_passes_origin_to_create_task(self):
        """v2 hand_off reads current_origin and forwards it to create_task."""
        from src.agent.builtins.coordination_tool import hand_off
        from src.shared.trace import current_origin
        from src.shared.types import MessageOrigin

        mc = _make_mesh_client(agent_id="scout")
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
    async def test_hand_off_no_origin_passes_none_to_create_task(self):
        """When current_origin is None, create_task receives origin=None."""
        from src.agent.builtins.coordination_tool import hand_off
        from src.shared.trace import current_origin

        mc = _make_mesh_client(agent_id="scout")
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


class TestCoordination:
    """coordination_tool routes hand_off / check_inbox /
    update_status / complete_task through the durable tasks
    endpoints (the v2 fallback to blackboard storage is gone).
    """

    @pytest.mark.asyncio
    async def test_hand_off_creates_task_row(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
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
    async def test_hand_off_wake_failure_surfaces_error_field(self):
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

        mc = _make_mesh_client(agent_id="scout")
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
        assert "task_orphan_42" in result["error"]
        # Directive recovery_hint — no "wait for heartbeat" softness,
        # and explicitly NOT "RETRY hand_off" (codex r4: each call
        # creates a new task row, retry would leave orphan duplicates).
        assert "task_orphan_42" in result["recovery_hint"]
        assert "DO NOT retry hand_off" in result["recovery_hint"]
        assert "DO NOT mark" in result["recovery_hint"]
        assert "operator" in result["recovery_hint"].lower()

    @pytest.mark.asyncio
    async def test_hand_off_with_data_writes_artifact(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
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
    async def test_check_inbox_returns_legacy_dict_shape(self):
        from src.agent.builtins.coordination_tool import check_inbox

        mc = _make_mesh_client(agent_id="analyst")
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
        # The new endpoint was hit. The thread-store event feed IS
        # called once per check_inbox to surface task_event back-edges
        # (C.3-a), but the dedicated task lookup is still the v2
        # endpoint. The blackboard is no longer touched at all.
        mc.list_task_inbox.assert_called_once_with("analyst")
        mc.list_inbox_events.assert_called_once_with()
        mc.list_blackboard.assert_not_called()

    @pytest.mark.asyncio
    async def test_complete_task_transitions_to_done(self):
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="analyst")
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
    async def test_complete_task_strips_legacy_prefix(self):
        """Legacy ``tasks/x/ho_abc`` keys still resolve to the bare id."""
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="analyst")
        mc.set_task_status.return_value = {"id": "ho_abc", "status": "done"}

        await complete_task("tasks/analyst/ho_abc", mesh_client=mc)

        mc.set_task_status.assert_called_once_with("ho_abc", "done")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("start_status", ["pending", "accepted"])
    async def test_complete_task_advances_handoff_through_working(
        self, start_status,
    ):
        """A not-yet-``working`` handoff/report task (the operator clearing
        a completion report it never moved through ``working``) is stepped
        through ``working`` so the terminal close is a valid transition
        instead of a ``pending/accepted → done`` 400 that wedges the
        inbox."""
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="operator")
        mc.get_task.return_value = {"id": "task_da9", "status": start_status}
        mc.set_task_status.return_value = {"id": "task_da9", "status": "done"}

        result = await complete_task("task_da9", mesh_client=mc)

        assert result["completed"] is True
        assert mc.set_task_status.await_args_list == [
            call("task_da9", "working"),
            call("task_da9", "done"),
        ]

    @pytest.mark.asyncio
    async def test_complete_task_blocked_closes_directly(self):
        """``blocked → done`` is already a valid transition, so a blocked
        task is closed directly without an intermediate ``working`` step."""
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="operator")
        mc.get_task.return_value = {"id": "task_blk", "status": "blocked"}
        mc.set_task_status.return_value = {"id": "task_blk", "status": "done"}

        result = await complete_task("task_blk", mesh_client=mc)

        assert result["completed"] is True
        mc.set_task_status.assert_called_once_with("task_blk", "done")

    @pytest.mark.asyncio
    async def test_complete_task_probe_failure_falls_back_to_direct_close(self):
        """If the status probe fails, complete_task still attempts the
        direct close — preserving behaviour for already-completable tasks."""
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="analyst")
        mc.get_task.side_effect = RuntimeError("mesh hiccup")
        mc.set_task_status.return_value = {"id": "task_xyz", "status": "done"}

        result = await complete_task("task_xyz", mesh_client=mc)

        assert result["completed"] is True
        mc.set_task_status.assert_called_once_with("task_xyz", "done")

    @pytest.mark.asyncio
    async def test_update_status_single_active_no_task_id(self):
        """One active task + no task_id → that task is updated transparently."""
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst")
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
    async def test_update_status_blocker_note_passes_summary(self):
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst")
        mc.list_task_inbox.return_value = [{"id": "task_x", "status": "working"}]

        await update_status("blocked", "waiting on creds", mesh_client=mc)

        mc.set_task_status.assert_called_once_with(
            "task_x", "blocked", blocker_note="waiting on creds",
        )

    @pytest.mark.asyncio
    async def test_update_status_ambiguous_with_multiple_active(self):
        """2+ active tasks + no task_id → ambiguous_task with active list + hint.

        The active list carries ``{id, title, state}`` entries so the LLM
        can pick a ``task_id`` directly without a follow-up
        ``check_inbox`` call (PR-Q richer-shape augment).
        """
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst")
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
    async def test_update_status_empty_inbox_returns_legacy_noop(self):
        """Empty inbox → legacy ``{updated: False, ...}`` no-op shape.

        Standalone agents and just-joined agents on a fresh fleet hit
        the empty-inbox case constantly. PR-Q removed the regressed
        ``{"error": "no_active_task"}`` shape so the LLM doesn't see an
        error for the common "no work yet" case.
        """
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst")
        mc.list_task_inbox.return_value = []

        result = await update_status("working", mesh_client=mc)

        # Legacy success-shape no-op — no error key, updated=False.
        assert "error" not in result
        assert result.get("updated") is False
        assert result.get("state") == "working"
        assert "no active tasks" in result.get("reason", "")
        mc.set_task_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_status_multiple_active_with_explicit_task_id(self):
        """2+ active tasks + valid task_id → that task is updated."""
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst")
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
    async def test_update_status_unknown_task_id_returns_task_not_found(self):
        """Explicit task_id not in inbox → ``task_not_found`` error."""
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="analyst")
        mc.list_task_inbox.return_value = [
            {"id": "task_a", "status": "pending"},
        ]

        result = await update_status(
            "done", task_id="task_missing", mesh_client=mc,
        )

        assert result.get("error") == "task_not_found"
        assert result.get("task_id") == "task_missing"
        mc.set_task_status.assert_not_called()


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

        mc = _make_mesh_client(agent_id="scout")
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

        mc = _make_mesh_client(agent_id="scout")
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
        ``task_queued=True`` + ``wake_failed`` + ``wake_error`` + (Bug G
        codex r4) the new ``error`` key + directive recovery hint."""
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
        # Bug G (codex r4): legacy path now also carries the explicit
        # ``error`` field LLMs reliably react to, and a directive
        # recovery hint that doesn't suggest a duplicate-creating retry.
        assert "error" in result
        assert "wake_failed" in result["error"]
        assert "MUST NOT report success" in result["error"]
        assert "DO NOT retry hand_off" in result["recovery_hint"]
        assert "DO NOT mark" in result["recovery_hint"]
        # task_key is the legacy locator the operator uses to look up
        # the queued task — it must appear in both error + hint so the
        # LLM can pass it along when escalating.
        assert result["task_key"] in result["error"]
        assert result["task_key"] in result["recovery_hint"]

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
    async def test_hand_off_wake_failure_reports_handed_off_false(self):
        """Pins the new contract: on wake failure ``hand_off`` MUST
        flip ``handed_off`` to False and emit ``task_queued=True``
        alongside the wake_failed signal. Pre-fix this returned
        ``handed_off:true`` AND ``wake_failed:true`` — the LLM picked up
        ``handed_off`` and reported success while the recipient never
        actually woke (Bug 2)."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
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
        # Bug G fix: recovery_hint is now directive ("DO NOT retry
        # hand_off", "DO NOT mark complete") rather than soft ("wait
        # for heartbeat"). LLMs were papering over the soft hint and
        # finalizing with "task complete" in their next reply. Codex
        # r4: must NOT instruct "retry hand_off" — each call creates a
        # new task row, retry would leave orphan duplicates.
        assert "DO NOT retry hand_off" in result["recovery_hint"]
        assert "DO NOT mark" in result["recovery_hint"]
        # And the unambiguous ``error`` field LLMs reliably react to.
        assert "wake_failed" in result["error"]


# ── Bug H: surface create_task / write_blackboard failures in hand_off ─


class TestHandOffCreateFailureSurfacing:
    """Operator-reported Bug H: when ``create_task`` or
    ``write_blackboard`` raised inside ``hand_off``, the bare
    ``{"error": "Failed to create task: ..."}`` envelope was easy for
    LLMs to gloss past — the in-prod case had seo-strategist call
    hand_off, create_task raise, and the agent reply "Brief completed"
    with no task ever queued for page-writer.

    The fix mirrors the Bug G envelope shape — explicit
    ``handed_off=False``, ``error`` field with "MUST NOT report
    success" language, and a directive ``recovery_hint``. ``task_queued``
    is now ``False`` (vs ``True`` for wake_failed) so the LLM can
    tell the difference between "row exists, wake didn't fire" and
    "row never made it".
    """

    @pytest.mark.asyncio
    async def test_create_task_failure_returns_directive_envelope(self):
        """v2 path: ``mesh_client.create_task`` raising surfaces the
        full Bug-G-style directive envelope, not a bare error."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="seo-strategist")
        mc.list_agents.return_value = {"page-writer": {"role": "writer"}}
        mc.create_task.side_effect = RuntimeError("mesh /tasks 503")

        result = await hand_off(
            to="page-writer",
            summary="Brief: langchain-vs-openlegion",
            mesh_client=mc,
        )

        # LLM-facing contract: handed_off must be False so the agent
        # cannot silently report "done".
        assert result["handed_off"] is False, (
            "create_task raised but handed_off is True — LLM will lie"
        )
        # task_queued=False distinguishes this from wake-failed (where
        # the row IS in SQLite). Here no row was persisted.
        assert result["task_queued"] is False
        assert result["create_failed"] is True
        assert result["to"] == "page-writer"
        # ``error`` is the field LLMs react to most strongly.
        assert "create_failed" in result["error"]
        assert "MUST NOT report success" in result["error"]
        assert "page-writer" in result["error"]
        # ``recovery_hint`` is directive, not advisory.
        assert "DO NOT mark this work as complete" in result["recovery_hint"]
        assert "page-writer" in result["recovery_hint"]
        # create_error carries the redacted exception detail for
        # operators who want the raw message without parsing ``error``.
        assert "mesh /tasks 503" in result["create_error"]

    @pytest.mark.asyncio
    async def test_create_task_failure_redacts_credentials(self):
        """v2 path: the exception from create_task can carry the mesh
        URL with an api_key in the query string. The error envelope
        is read back into the LLM context — credentials must be
        redacted before truncation, same precaution as wake_error.
        """
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="seo-strategist")
        mc.list_agents.return_value = {"page-writer": {"role": "writer"}}
        leaky_token = "sk-" + "a" * 40
        leaky = (
            f"POST https://mesh.internal/tasks?api_key={leaky_token} "
            f"failed with 500"
        )
        mc.create_task.side_effect = RuntimeError(leaky)

        result = await hand_off(
            to="page-writer",
            summary="Brief: x",
            mesh_client=mc,
        )

        assert result["create_failed"] is True
        # Credential must not appear anywhere LLM-readable.
        assert leaky_token not in result["create_error"]
        assert leaky_token not in result["error"]
        assert leaky_token not in result["recovery_hint"]

    @pytest.mark.asyncio
    async def test_output_write_failure_returns_directive_envelope(self):
        """v2 path: ``write_blackboard`` for the handoff artifact
        raising surfaces the directive envelope too. This is the
        earliest failure point — no task row, no artifact persisted."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="trend-scout")
        mc.list_agents.return_value = {
            "seo-strategist": {"role": "strategist"},
        }
        mc.write_blackboard.side_effect = RuntimeError("blackboard EBUSY")

        result = await hand_off(
            to="seo-strategist",
            summary="Top 5 trends for week of 2026-05-18",
            data='{"trends": ["a", "b"]}',
            mesh_client=mc,
        )

        assert result["handed_off"] is False
        assert result["task_queued"] is False
        assert result["output_write_failed"] is True
        assert result["to"] == "seo-strategist"
        assert "output_write_failed" in result["error"]
        assert "MUST NOT report success" in result["error"]
        assert "DO NOT mark this work as complete" in result["recovery_hint"]
        # ``create_task`` must NOT have been reached — the write failed
        # first, so trying to create the task would have written into
        # a half-populated state.
        assert mc.create_task.await_count == 0
        # write_error carries the redacted exception detail.
        assert "blackboard EBUSY" in result["write_error"]


    @pytest.mark.asyncio
    async def test_create_failure_does_not_call_wake_agent(self):
        """Defense-in-depth: a create_task failure must short-circuit
        before ``wake_agent`` is called — waking a peer for a task
        that doesn't exist would spam them with a phantom notification.
        """
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="strategist")
        mc.list_agents.return_value = {"writer": {"role": "writer"}}
        mc.create_task.side_effect = RuntimeError("scope rejected")

        await hand_off(
            to="writer",
            summary="x",
            mesh_client=mc,
        )

        assert mc.wake_agent.await_count == 0

    @pytest.mark.asyncio
    async def test_success_envelope_omits_create_failure_keys(self):
        """Regression guard: the happy path must not leak any of the
        new failure flags into the success envelope."""
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="strategist")
        mc.list_agents.return_value = {"writer": {"role": "writer"}}

        result = await hand_off(
            to="writer",
            summary="ok",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        assert "create_failed" not in result
        assert "output_write_failed" not in result
        assert "create_error" not in result
        assert "write_error" not in result


# ── Bug H codex r5: redaction coverage for the remaining sites ──────


class TestHandOffFailureRedaction:
    """Codex r5 finding: redaction was only asserted on the v2
    create_task path. Cover the three remaining sites (v2 output_write,
    legacy output_write, legacy task_write) so dropping
    ``redact_text_with_urls`` from any of them is caught. v2
    create_task redaction lives in ``TestHandOffCreateFailureSurfacing``
    (``test_create_task_failure_redacts_credentials``)."""

    LEAKY = "sk-" + "a" * 40
    LEAKY_URL = f"POST https://mesh.internal/x?api_key={LEAKY} failed 500"

    @pytest.mark.asyncio
    async def test_output_write_redacts(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="trend-scout")
        mc.list_agents.return_value = {
            "seo-strategist": {"role": "strategist"},
        }
        mc.write_blackboard.side_effect = RuntimeError(self.LEAKY_URL)

        result = await hand_off(
            to="seo-strategist",
            summary="x",
            data='{"a": 1}',
            mesh_client=mc,
        )

        assert result["output_write_failed"] is True
        for f in ("error", "recovery_hint", "write_error"):
            assert self.LEAKY not in result[f]


# ── Bug H codex r5: directive envelopes on terminal-transition tools ─


class TestUpdateStatusFailureEnvelope:
    """update_status with state='done' or 'blocked' is a terminal-
    transition operation — the LLM uses it to signal completion. A
    silent failure here would let the agent claim done with no record."""

    @pytest.mark.asyncio
    async def test_done_transition_failure_returns_directive(self):
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="writer")
        mc.list_task_inbox = AsyncMock(return_value=[
            {"id": "task_xyz", "status": "working", "title": "draft"},
        ])
        mc.set_task_status.side_effect = RuntimeError("db locked")

        result = await update_status(
            "done", "finished draft", mesh_client=mc,
        )

        assert result["update_status_failed"] is True
        assert "MUST NOT report success" in result["error"]
        assert "DO NOT mark this work as complete" in result["recovery_hint"]
        assert "db locked" in result["detail"]
        assert result["state"] == "done"
        assert result["task_id"] == "task_xyz"


    @pytest.mark.asyncio
    async def test_done_failure_redacts_credentials(self):
        from src.agent.builtins.coordination_tool import update_status

        mc = _make_mesh_client(agent_id="writer")
        mc.list_task_inbox = AsyncMock(return_value=[
            {"id": "task_xyz", "status": "working", "title": "draft"},
        ])
        leaky = "sk-" + "z" * 40
        mc.set_task_status.side_effect = RuntimeError(
            f"PUT https://mesh/x?api_key={leaky}",
        )

        result = await update_status("done", "ok", mesh_client=mc)

        assert result["update_status_failed"] is True
        for f in ("error", "recovery_hint", "detail"):
            assert leaky not in result[f]


class TestCompleteTaskFailureEnvelope:
    """complete_task is the explicit "I'm done with this work" signal —
    silent failure here is the highest-risk misreport surface."""

    @pytest.mark.asyncio
    async def test_failure_returns_directive(self):
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="writer")
        mc.set_task_status.side_effect = RuntimeError("set_status 503")

        result = await complete_task(
            "task_abc123def456", mesh_client=mc,
        )

        assert result["complete_task_failed"] is True
        assert "MUST NOT report success" in result["error"]
        assert "DO NOT mark this work as complete" in result["recovery_hint"]
        assert "set_status 503" in result["detail"]
        assert result["task_id"] == "task_abc123def456"


    @pytest.mark.asyncio
    async def test_failure_redacts_credentials(self):
        from src.agent.builtins.coordination_tool import complete_task

        mc = _make_mesh_client(agent_id="writer")
        leaky = "sk-" + "b" * 40
        mc.set_task_status.side_effect = RuntimeError(
            f"PUT https://mesh/x?api_key={leaky}",
        )

        result = await complete_task("task_xyz", mesh_client=mc)

        assert result["complete_task_failed"] is True
        for f in ("error", "recovery_hint", "detail"):
            assert leaky not in result[f]


class TestFailedTransitionEnvelopeHelper:
    """Codex r6: the helper merges ``extras`` BEFORE the sentinel
    keys are written, so a caller that accidentally passes
    ``extras={"error": "..."}`` cannot shadow the directive field.
    Pinned here so a future refactor that flips the merge order is
    caught.
    """

    def test_extras_cannot_shadow_sentinel_keys(self):
        from src.agent.builtins.coordination_tool import (
            _failed_transition_envelope,
        )

        result = _failed_transition_envelope(
            kind="test_failed",
            detail="something broke",
            exc=RuntimeError("oops"),
            extras={
                "error": "MALICIOUS — should be overwritten",
                "recovery_hint": "MALICIOUS — should be overwritten",
                "detail": "MALICIOUS — should be overwritten",
                "task_id": "task_abc",
            },
        )

        # Sentinel fields untouched.
        assert "MUST NOT report success" in result["error"]
        assert "DO NOT mark this work as complete" in result["recovery_hint"]
        assert result["detail"] == "oops"
        assert result["test_failed"] is True
        # Non-sentinel extras flow through.
        assert result["task_id"] == "task_abc"


class TestHandOffParentTaskIdPropagation:
    """Operator workflow awareness: ``hand_off`` reads ``current_task_id``
    so the new task chains under the parent for ``workflow_snapshot`` to
    walk descendants from the kickoff root."""

    @pytest.mark.asyncio
    async def test_hand_off_passes_parent_task_id_when_set(self):
        from src.agent.builtins.coordination_tool import hand_off
        from src.shared.trace import current_task_id

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        mc.create_task.return_value = {"id": "task_child", "status": "pending"}

        token = current_task_id.set("task_root_parent")
        try:
            result = await hand_off(
                to="analyst", summary="next stage", mesh_client=mc,
            )
        finally:
            current_task_id.reset(token)

        assert result["handed_off"] is True
        kwargs = mc.create_task.call_args.kwargs
        assert kwargs.get("parent_task_id") == "task_root_parent"

    @pytest.mark.asyncio
    async def test_hand_off_passes_none_when_unset(self):
        """Outside a task context (heartbeats, free chat) the contextvar
        is None and create_task must receive parent_task_id=None — the
        new task lands as a workflow root."""
        from src.agent.builtins.coordination_tool import hand_off
        from src.shared.trace import current_task_id

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        mc.create_task.return_value = {"id": "task_root", "status": "pending"}

        # Reset to the default (None) explicitly so the test is robust to
        # leaking state from prior tests in this module.
        token = current_task_id.set(None)
        try:
            await hand_off(
                to="analyst", summary="kickoff", mesh_client=mc,
            )
        finally:
            current_task_id.reset(token)

        kwargs = mc.create_task.call_args.kwargs
        assert kwargs.get("parent_task_id") is None


class TestHandOffBrief:
    """B2: the `brief` param becomes the task description so the recipient
    starts with full context instead of a title-sized stub."""

    @pytest.mark.asyncio
    async def test_brief_becomes_description(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="operator")
        mc.list_agents.return_value = {"analyst": {"role": "analyst"}}
        brief = (
            "## Objective\nDeep SEO audit of example.com\n\n"
            "## Context\nUser cares about long-tail keyword gaps.\n\n"
            "## Deliverable\nFull written audit saved as an artifact."
        )

        result = await hand_off(
            to="analyst",
            summary="Deep SEO audit of example.com",
            brief=brief,
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        kwargs = mc.create_task.call_args.kwargs
        assert kwargs["title"] == "Deep SEO audit of example.com"
        assert kwargs["description"] == brief

    @pytest.mark.asyncio
    async def test_brief_truncated_at_cap(self):
        from src.agent.builtins.coordination_tool import (
            _MAX_BRIEF_CHARS,
            hand_off,
        )

        mc = _make_mesh_client(agent_id="operator")
        mc.list_agents.return_value = {"analyst": {}}

        await hand_off(
            to="analyst",
            summary="big brief",
            brief="b" * (_MAX_BRIEF_CHARS + 5_000),
            mesh_client=mc,
        )

        kwargs = mc.create_task.call_args.kwargs
        assert len(kwargs["description"]) <= _MAX_BRIEF_CHARS

    @pytest.mark.asyncio
    async def test_long_summary_with_brief_keeps_caller_split(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="operator")
        mc.list_agents.return_value = {"analyst": {}}
        long_summary = "Audit the SEO of example.com and " + "x" * 150

        await hand_off(
            to="analyst",
            summary=long_summary,
            brief="the real instructions",
            mesh_client=mc,
        )

        kwargs = mc.create_task.call_args.kwargs
        # Caller provided both — title hard-capped, brief preserved verbatim.
        assert len(kwargs["title"]) <= 200
        assert kwargs["description"] == "the real instructions"

    @pytest.mark.asyncio
    async def test_no_brief_preserves_legacy_shape(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {}}

        await hand_off(to="analyst", summary="quick ping", mesh_client=mc)

        kwargs = mc.create_task.call_args.kwargs
        assert kwargs["title"] == "quick ping"
        assert kwargs["description"] == "quick ping"


class TestHandOffThinking:
    """B4: hand_off can pin a per-task reasoning depth for the recipient."""

    @pytest.mark.asyncio
    async def test_thinking_passes_through_to_create_task(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="operator")
        mc.list_agents.return_value = {"analyst": {}}

        result = await hand_off(
            to="analyst", summary="deep audit", thinking="high",
            mesh_client=mc,
        )

        assert result["handed_off"] is True
        assert mc.create_task.call_args.kwargs["thinking"] == "high"

    @pytest.mark.asyncio
    async def test_omitted_thinking_sends_none(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {}}

        await hand_off(to="analyst", summary="quick ping", mesh_client=mc)

        assert mc.create_task.call_args.kwargs["thinking"] is None

    @pytest.mark.asyncio
    async def test_invalid_thinking_rejected_before_any_write(self):
        from src.agent.builtins.coordination_tool import hand_off

        mc = _make_mesh_client(agent_id="scout")
        mc.list_agents.return_value = {"analyst": {}}

        result = await hand_off(
            to="analyst", summary="x", thinking="ultra", mesh_client=mc,
        )

        assert "error" in result
        mc.create_task.assert_not_called()
        mc.write_blackboard.assert_not_called()


class TestToolDescriptions:
    """Contract tests pinning the pushback-protocol prose in the
    LLM-visible tool descriptions. The transport for blocked-task
    back-edges (blocker_note → task-thread event → check_inbox
    events[]) shipped without telling workers it exists;
    these pins keep future description edits from silently dropping
    the protocol documentation.
    """

    @staticmethod
    def _description(name: str) -> str:
        # Reload rather than import: other test modules (test_grouped_tools)
        # clear the _tool_staging registry in their setup, and a cached
        # module import would then leave it empty. Re-running the @tool
        # decorators makes the lookup order-independent.
        import importlib

        import src.agent.builtins.coordination_tool as ct
        importlib.reload(ct)
        from src.agent.tools import _tool_staging

        return _tool_staging[name]["description"]

    def test_update_status_description_mentions_pushback(self):
        desc = self._description("update_status")
        assert "pushback" in desc
        # Phase 2 unit 3: a quick clarifying question goes through
        # ask_teammate FIRST — blocked is for real stoppages.
        assert "ask_teammate" in desc

    def test_check_inbox_description_mentions_events(self):
        desc = self._description("check_inbox")
        assert "events[]" in desc
        # Phase 2 unit 3: blocked questions are answered inline via
        # ask_teammate — the "corrected hand_off" reissue dance is gone.
        assert "ask_teammate" in desc
        assert "corrected hand_off" not in desc

    def test_hand_off_description_mentions_blocked_followup(self):
        desc = self._description("hand_off")
        # Phase 2 unit 3: the blocked→re-hand-off dance copy is retired;
        # creators answer blockers inline via ask_teammate instead.
        assert "re-hand-off" not in desc
        assert "ask_teammate" in desc
        assert "duplicate" in desc

    def test_ask_teammate_description_frames_semi_trusted(self):
        """The ask verb's return is semi-trusted teammate INPUT, not
        instructions (plan §4) — the description must say so, and must
        route work to hand_off."""
        desc = self._description("ask_teammate")
        assert "SEMI-TRUSTED" in desc or "semi-trusted" in desc
        assert "not instructions" in desc
        assert "hand_off" in desc

    def test_answer_ask_description_mentions_single_use(self):
        desc = self._description("answer_ask")
        assert "Single-use" in desc or "single-use" in desc
        assert "semi-trusted" in desc
