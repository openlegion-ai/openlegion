"""Tests for the extracted ``_write_task_event_back_edge`` helper.

The helper records back-edge events on the task's thread (ThreadStore,
``kind='event'``, ``recipient=origin_user``) on terminal status
transitions and — for actionable kinds (``task_failed`` /
``task_blocked``) — wakes the originator via the lane with a 60s
per-task rate limit.

This module builds a minimal mesh app, grabs the closure exposed on
``app._write_task_event_back_edge``, and exercises the eligibility /
wake / rate-limit paths directly. The closure shares state with the
endpoint that calls it, so this is testing the exact in-process function
the request handler uses. Wake/eligibility semantics are storage-
independent — the assertions below are unchanged from the blackboard
era; only the storage queries moved to the thread store.
"""

from __future__ import annotations

import asyncio
import importlib
import time

import pytest

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.threads import ThreadStore
from src.shared.types import AgentPermissions


def _reload_server(monkeypatch, *, tasks_db: str):
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_DB", tasks_db)
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


class _FakeLane:
    """Minimal LaneManager stand-in that just records enqueue calls."""

    def __init__(self):
        self.calls: list[dict] = []

    async def enqueue(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return "ok"


@pytest.fixture
def mesh_app_with_back_edge(tmp_path, monkeypatch):
    """Build a mesh app with a fake lane + dispatch loop wired so the
    back-edge helper can fire wake calls."""
    server_module = _reload_server(
        monkeypatch, tasks_db=str(tmp_path / "tasks.db"),
    )

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid in ("operator", "scout", "analyst", "writer"):
        permissions.permissions[aid] = AgentPermissions(
            agent_id=aid, can_route_tasks=True,
        )
    router = MessageRouter(permissions, {
        "operator": "http://operator:8400",
        "scout":    "http://scout:8400",
        "analyst":  "http://analyst:8400",
        "writer":   "http://writer:8400",
    })

    lane = _FakeLane()
    # The helper expects a real running loop reference for
    # ``run_coroutine_threadsafe``. Build a loop, kick it on a worker
    # thread, and feed the reference in.
    loop = asyncio.new_event_loop()
    import threading

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    thread_store = ThreadStore(":memory:")
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        thread_store=thread_store,
        lane_manager=lane,  # type: ignore[arg-type]
        dispatch_loop=loop,
    )
    yield app, thread_store, lane, loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    blackboard.close()
    thread_store.close()
    loop.close()
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


def _record(task_id: str, **overrides) -> dict:
    """Synthesize the post-transition task record shape the helper expects."""
    base = {
        "id": task_id,
        "assignee": "writer",
        # L9 binding: the back-edge wakes ``origin.user`` only when it
        # equals the task's ``creator``. For a legit single-hop handoff
        # (scout → writer), scout is both the creator and the origin user.
        "creator": "scout",
        "title": "stage work",
        "status": "failed",
        "origin": {"kind": "agent", "channel": "", "user": "scout"},
    }
    base.update(overrides)
    return base


def _settle(seconds: float = 0.05) -> None:
    """Wait briefly so ``run_coroutine_threadsafe`` calls land on the
    background dispatch loop."""
    time.sleep(seconds)


def _events_for(thread_store: ThreadStore, recipient: str) -> list[dict]:
    """The store-backed replacement for the old
    ``blackboard.list_by_prefix("inbox/{recipient}/task_event/")``."""
    return thread_store.list_events_for(recipient)


def _age_event(thread_store: ThreadStore, task_id: str, age_seconds: float) -> None:
    """Backdate the task's event row(s) to test the serving windows."""
    with thread_store._conn() as conn:
        conn.execute(
            "UPDATE thread_messages SET created_at = ? WHERE thread_id = ?",
            (time.time() - age_seconds, f"task:{task_id}"),
        )


class TestBackEdgeHelperEligibility:
    def test_task_completed_does_not_wake(self, mesh_app_with_back_edge):
        """``task_completed`` records the event + does NOT wake the
        originator (operator picks up via heartbeat, not interrupt)."""
        app, thread_store, lane, _loop = mesh_app_with_back_edge
        rec = _record("task_done_1", status="done")

        app._write_task_event_back_edge(
            rec, event_kind="task_completed",
            payload_extras={"summary": "all good"},
        )
        _settle()

        events = _events_for(thread_store, "scout")
        assert any(e.get("task_id") == "task_done_1" for e in events)
        # NO lane wake.
        assert lane.calls == []

    def test_task_failed_writes_event_and_wakes(self, mesh_app_with_back_edge):
        """``task_failed`` records the event AND wakes the originator
        (actionable kind)."""
        app, thread_store, lane, _loop = mesh_app_with_back_edge
        rec = _record("task_fail_1", status="failed")

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
            payload_extras={
                "error": "lane_timeout", "timeout_seconds": 900,
            },
        )
        _settle()

        payload = next(
            e for e in _events_for(thread_store, "scout")
            if e.get("task_id") == "task_fail_1"
        )
        assert payload["kind"] == "task_failed"
        assert payload["error"] == "lane_timeout"
        # The event landed on the task's own thread.
        assert payload["thread_id"] == "task:task_fail_1"
        # Wake was scheduled to the originator.
        assert len(lane.calls) == 1
        wake = lane.calls[0]
        # First positional arg is the target agent id (origin_user).
        assert wake["args"][0] == "scout"
        # task_id is forwarded so the recipient's loop can auto-close.
        assert wake["kwargs"].get("task_id") == "task_fail_1"

    def test_rate_limit_collapses_repeated_wakes(self, mesh_app_with_back_edge):
        """Same task_id failing twice within 60s only wakes the
        originator once — the in-memory rate-limit state coalesces bursts."""
        app, _, lane, _loop = mesh_app_with_back_edge
        rec = _record("task_fail_burst")

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
            payload_extras={"error": "first"},
        )
        # Immediate retry — same task_id, well inside the 60s window.
        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
            payload_extras={"error": "retry"},
        )
        _settle()

        # Only ONE lane wake call — the second was rate-limited away.
        assert len(lane.calls) == 1

    def test_self_handoff_skipped(self, mesh_app_with_back_edge):
        """origin.user == assignee → suppress the back-edge so an
        originating agent's check_inbox stays clean."""
        app, thread_store, lane, _loop = mesh_app_with_back_edge
        rec = _record(
            "task_self_1",
            assignee="scout",
            origin={"kind": "agent", "channel": "", "user": "scout"},
        )

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
        )
        _settle()

        events = _events_for(thread_store, "scout")
        assert not any(
            e.get("task_id") == "task_self_1" for e in events
        )
        assert lane.calls == []

    def test_origin_user_mismatch_writes_event_but_does_not_wake(
        self, mesh_app_with_back_edge,
    ):
        """L9 binding: when ``origin.user`` is not the task ``creator``
        (forged origin, or a multi-hop chain whose origin points at a
        distant root), the back-edge EVENT is still recorded so the
        originator's check_inbox/heartbeat sees it — but the privileged
        wake is skipped so no arbitrary agent is interrupted."""
        app, thread_store, lane, _loop = mesh_app_with_back_edge
        # origin.user="analyst" (claimed) but creator="scout" (real).
        rec = _record(
            "task_mismatch_1",
            creator="scout",
            origin={"kind": "agent", "channel": "", "user": "analyst"},
        )

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
            payload_extras={"error": "boom"},
        )
        _settle()

        # Event IS recorded for the claimed origin (delivery intact).
        events = _events_for(thread_store, "analyst")
        assert any(
            e.get("task_id") == "task_mismatch_1" for e in events
        )
        # But NO wake — origin_user != creator.
        assert lane.calls == []

    def test_origin_user_matches_creator_wakes(self, mesh_app_with_back_edge):
        """L9 binding: the legit case (origin.user == creator) still
        wakes the originator — auto-recovery is unaffected."""
        app, thread_store, lane, _loop = mesh_app_with_back_edge
        rec = _record(
            "task_match_1",
            creator="scout",
            origin={"kind": "agent", "channel": "", "user": "scout"},
        )

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
            payload_extras={"error": "boom"},
        )
        _settle()

        events = _events_for(thread_store, "scout")
        assert any(e.get("task_id") == "task_match_1" for e in events)
        assert len(lane.calls) == 1
        assert lane.calls[0]["args"][0] == "scout"

    def test_human_origin_failure_wakes_operator(self, mesh_app_with_back_edge):
        """A2: ``origin.kind=human`` failures don't back-edge the human
        (ChainWatcher informs the user) — they wake the OPERATOR into a
        recovery turn, with the event recorded for the operator."""
        app, thread_store, lane, _loop = mesh_app_with_back_edge
        rec = _record(
            "task_human_1",
            origin={"kind": "human", "channel": "telegram", "user": "9999"},
        )

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
            payload_extras={"error": "boom"},
        )
        _settle()

        # The human user gets no mesh-side back-edge event...
        events = _events_for(thread_store, "9999")
        assert not any(
            e.get("task_id") == "task_human_1" for e in events
        )
        # ...but the operator gets the event + a recovery wake.
        op_events = _events_for(thread_store, "operator")
        assert any(
            e.get("task_id") == "task_human_1" for e in op_events
        )
        assert len(lane.calls) == 1
        assert lane.calls[0]["args"][0] == "operator"
        wake_msg = lane.calls[0]["args"][1]
        assert "task_human_1" in wake_msg
        assert "already informed" in wake_msg
        # No task_id threading — the operator's turn is ABOUT the task,
        # not an execution of it; auto-close must not touch the row.
        assert lane.calls[0]["kwargs"].get("task_id") is None
        assert lane.calls[0]["kwargs"].get("auto_notify") is False

    def test_human_origin_completion_does_not_wake_operator(
        self, mesh_app_with_back_edge,
    ):
        """Successful human-origin chains stay quiet — recovery wakes are
        for failed/blocked only."""
        app, thread_store, lane, _loop = mesh_app_with_back_edge
        rec = _record(
            "task_human_2", status="done",
            origin={"kind": "human", "channel": "telegram", "user": "9999"},
        )

        app._write_task_event_back_edge(
            rec, event_kind="task_completed",
            payload_extras={"summary": "ok"},
        )
        _settle()

        op_events = _events_for(thread_store, "operator")
        assert not any(
            e.get("task_id") == "task_human_2" for e in op_events
        )
        assert lane.calls == []

    def test_human_origin_operator_assignee_not_self_woken(
        self, mesh_app_with_back_edge,
    ):
        """The operator's own failed tasks don't trigger a self-wake."""
        app, _thread_store, lane, _loop = mesh_app_with_back_edge
        rec = _record(
            "task_human_3", assignee="operator",
            origin={"kind": "human", "channel": "web", "user": "u1"},
        )

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
            payload_extras={"error": "boom"},
        )
        _settle()

        assert lane.calls == []

    def test_human_origin_wake_rate_limited_per_task(
        self, mesh_app_with_back_edge,
    ):
        """Burst coalescing: lane timeout + sweep retry on the same task
        produce ONE operator wake inside the 60s window."""
        app, _thread_store, lane, _loop = mesh_app_with_back_edge
        rec = _record(
            "task_human_4",
            origin={"kind": "human", "channel": "web", "user": "u1"},
        )

        app._write_task_event_back_edge(
            rec, event_kind="task_failed", payload_extras={"error": "boom"},
        )
        app._write_task_event_back_edge(
            rec, event_kind="task_failed", payload_extras={"error": "boom"},
        )
        _settle()

        assert len(lane.calls) == 1


class TestBackEdgeTracePropagation:
    """Session observability: a back-edge / operator-recovery wake must
    continue the failed task's trace so ``openlegion session <trace>`` shows
    the recovery branch. ``update_task_status`` does not seed the trace
    contextvar, so the helper must pass the task's stored ``trace_id``
    explicitly — otherwise ``_direct_dispatch`` mints a fresh, disconnected
    trace and the recovery work falls out of the session."""

    def test_agent_origin_back_edge_wake_carries_task_trace(
        self, mesh_app_with_back_edge,
    ):
        app, _thread_store, lane, _loop = mesh_app_with_back_edge
        rec = _record(
            "task_trace_1",
            creator="scout",
            origin={"kind": "agent", "channel": "", "user": "scout"},
            trace_id="tr_aaaaaaaaaaaa",
        )

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
            payload_extras={"error": "boom"},
        )
        _settle()

        assert len(lane.calls) == 1
        assert lane.calls[0]["args"][0] == "scout"
        assert lane.calls[0]["kwargs"].get("trace_id") == "tr_aaaaaaaaaaaa"

    def test_human_origin_recovery_wake_carries_task_trace(
        self, mesh_app_with_back_edge,
    ):
        app, _thread_store, lane, _loop = mesh_app_with_back_edge
        rec = _record(
            "task_trace_2",
            origin={"kind": "human", "channel": "telegram", "user": "9999"},
            trace_id="tr_bbbbbbbbbbbb",
        )

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
            payload_extras={"error": "boom"},
        )
        _settle()

        op_wakes = [c for c in lane.calls if c["args"][0] == "operator"]
        assert len(op_wakes) == 1
        assert op_wakes[0]["kwargs"].get("trace_id") == "tr_bbbbbbbbbbbb"


class TestBackEdgeServingWindows:
    """The old TTL split is now a pair of query windows in
    ``ThreadStore.list_events_for``: actionable events serve for 7 days,
    informational events for 24h — so completed-task events can't pile
    up for a week and flood the operator's check_inbox / LLM context on
    every heartbeat."""

    def test_completed_event_stops_serving_after_24h(
        self, mesh_app_with_back_edge,
    ):
        app, thread_store, _lane, _loop = mesh_app_with_back_edge
        rec = _record("task_ttl_done", status="done")

        app._write_task_event_back_edge(
            rec, event_kind="task_completed",
            payload_extras={"summary": "ok"},
        )
        _settle()

        assert any(
            e.get("task_id") == "task_ttl_done"
            for e in _events_for(thread_store, "scout")
        )
        _age_event(thread_store, "task_ttl_done", 86_400 + 60)
        assert not any(
            e.get("task_id") == "task_ttl_done"
            for e in _events_for(thread_store, "scout")
        )

    def test_cancelled_event_stops_serving_after_24h(
        self, mesh_app_with_back_edge,
    ):
        app, thread_store, _lane, _loop = mesh_app_with_back_edge
        rec = _record("task_ttl_cancel", status="cancelled")

        app._write_task_event_back_edge(
            rec, event_kind="task_cancelled",
        )
        _settle()

        _age_event(thread_store, "task_ttl_cancel", 86_400 + 60)
        assert not any(
            e.get("task_id") == "task_ttl_cancel"
            for e in _events_for(thread_store, "scout")
        )

    def test_failed_event_keeps_serving_past_24h(self, mesh_app_with_back_edge):
        app, thread_store, _lane, _loop = mesh_app_with_back_edge
        rec = _record("task_ttl_fail", status="failed")

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
            payload_extras={"error": "boom"},
        )
        _settle()

        _age_event(thread_store, "task_ttl_fail", 86_400 + 60)
        assert any(
            e.get("task_id") == "task_ttl_fail"
            for e in _events_for(thread_store, "scout")
        )
        # ...but not past the 7-day actionable window.
        _age_event(thread_store, "task_ttl_fail", 604_800 + 60)
        assert not any(
            e.get("task_id") == "task_ttl_fail"
            for e in _events_for(thread_store, "scout")
        )

    def test_blocked_event_keeps_serving_past_24h(self, mesh_app_with_back_edge):
        app, thread_store, _lane, _loop = mesh_app_with_back_edge
        rec = _record("task_ttl_block", status="blocked")

        app._write_task_event_back_edge(
            rec, event_kind="task_blocked",
            payload_extras={"blocker_note": "need creds"},
        )
        _settle()

        _age_event(thread_store, "task_ttl_block", 86_400 + 60)
        payload = next(
            e for e in _events_for(thread_store, "scout")
            if e.get("task_id") == "task_ttl_block"
        )
        assert payload["blocker_note"] == "need creds"


class TestBackEdgeMultiplicity:
    """Overwrite semantics restored read-side: the old blackboard
    back-edge was an upsert per (recipient, task) — one event per task,
    latest transition wins. The append model must serve the same."""

    def test_blocked_then_done_serves_one_informational(
        self, mesh_app_with_back_edge,
    ):
        app, thread_store, _lane, _loop = mesh_app_with_back_edge

        app._write_task_event_back_edge(
            _record("task_multi_1", status="blocked"),
            event_kind="task_blocked",
            payload_extras={"blocker_note": "stuck"},
        )
        app._write_task_event_back_edge(
            _record("task_multi_1", status="done"),
            event_kind="task_completed",
            payload_extras={"summary": "recovered"},
        )
        _settle()

        events = [
            e for e in _events_for(thread_store, "scout")
            if e.get("task_id") == "task_multi_1"
        ]
        # Exactly ONE event — the later informational transition
        # silences the stale actionable one.
        assert [e["kind"] for e in events] == ["task_completed"]

        # ...and it ages out on the 24h informational window without
        # resurfacing the shadowed task_blocked row.
        _age_event(thread_store, "task_multi_1", 86_400 + 60)
        assert not any(
            e.get("task_id") == "task_multi_1"
            for e in _events_for(thread_store, "scout")
        )

    def test_repeated_failures_serve_one_task_failed(
        self, mesh_app_with_back_edge,
    ):
        app, thread_store, _lane, _loop = mesh_app_with_back_edge

        for err in ("first", "retry"):
            app._write_task_event_back_edge(
                _record("task_multi_2", status="failed"),
                event_kind="task_failed",
                payload_extras={"error": err},
            )
        _settle()

        events = [
            e for e in _events_for(thread_store, "scout")
            if e.get("task_id") == "task_multi_2"
        ]
        assert len(events) == 1
        assert events[0]["kind"] == "task_failed"
        assert events[0]["error"] == "retry"  # newest transition wins


class TestOperatorRecoveryWakeGlobalThrottle:
    def test_mass_failure_storm_caps_wakes_but_writes_all_events(
        self, mesh_app_with_back_edge,
    ):
        """A2 hardening: a provider outage failing many user chains at
        once produces at most _OPERATOR_RECOVERY_WAKE_MAX wakes in the
        window — every event is still recorded so the heartbeat's
        check_inbox catches up on the suppressed remainder."""
        app, thread_store, lane, _loop = mesh_app_with_back_edge
        n = 8
        for i in range(n):
            rec = _record(
                f"task_storm_{i}",
                origin={"kind": "human", "channel": "web", "user": "u1"},
            )
            app._write_task_event_back_edge(
                rec, event_kind="task_failed",
                payload_extras={"error": "provider outage"},
            )
        _settle()

        # All 8 events landed for the operator...
        storm_ids = {
            e.get("task_id") for e in _events_for(thread_store, "operator")
            if str(e.get("task_id", "")).startswith("task_storm_")
        }
        assert len(storm_ids) == n
        # ...but only the capped number of wakes fired.
        operator_wakes = [
            c for c in lane.calls if c["args"][0] == "operator"
        ]
        assert len(operator_wakes) == 5  # _OPERATOR_RECOVERY_WAKE_MAX
