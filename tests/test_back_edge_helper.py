"""Tests for the extracted ``_write_task_event_back_edge`` helper.

The helper writes back-edge events to ``inbox/{origin_user}/task_event/{task_id}``
on terminal status transitions and — for actionable kinds
(``task_failed`` / ``task_blocked``) — wakes the originator via the
lane with a 60s per-task rate limit.

This module builds a minimal mesh app, grabs the closure exposed on
``app._write_task_event_back_edge``, and exercises the eligibility /
wake / rate-limit paths directly. The closure shares state with the
endpoint that calls it, so this is testing the exact in-process function
the request handler uses.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
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

    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        lane_manager=lane,  # type: ignore[arg-type]
        dispatch_loop=loop,
    )
    yield app, blackboard, lane, loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    blackboard.close()
    loop.close()
    monkeypatch.delenv("OPENLEGION_ORCHESTRATION_TASKS_DB", raising=False)
    importlib.reload(server_module)


def _record(task_id: str, **overrides) -> dict:
    """Synthesize the post-transition task record shape the helper expects."""
    base = {
        "id": task_id,
        "assignee": "writer",
        "title": "stage work",
        "status": "failed",
        "origin": {"kind": "agent", "channel": "", "user": "scout"},
    }
    base.update(overrides)
    return base


def _settle(seconds: float = 0.05) -> None:
    """Wait briefly so ``run_coroutine_threadsafe`` calls land on the
    background dispatch loop."""
    import time
    time.sleep(seconds)


class TestBackEdgeHelperEligibility:
    def test_task_completed_does_not_wake(self, mesh_app_with_back_edge):
        """``task_completed`` writes inbox + does NOT wake the originator
        (operator picks up via heartbeat, not interrupt)."""
        app, blackboard, lane, _loop = mesh_app_with_back_edge
        rec = _record("task_done_1", status="done")

        app._write_task_event_back_edge(
            rec, event_kind="task_completed",
            payload_extras={"summary": "all good"},
        )
        _settle()

        rows = blackboard.list_by_prefix("inbox/scout/task_event/")
        assert any(r.value.get("task_id") == "task_done_1" for r in rows)
        # NO lane wake.
        assert lane.calls == []

    def test_task_failed_writes_inbox_and_wakes(self, mesh_app_with_back_edge):
        """``task_failed`` writes inbox AND wakes the originator (actionable kind)."""
        app, blackboard, lane, _loop = mesh_app_with_back_edge
        rec = _record("task_fail_1", status="failed")

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
            payload_extras={
                "error": "lane_timeout", "timeout_seconds": 900,
            },
        )
        _settle()

        rows = blackboard.list_by_prefix("inbox/scout/task_event/")
        payload = next(
            r.value for r in rows
            if r.value.get("task_id") == "task_fail_1"
        )
        assert payload["kind"] == "task_failed"
        assert payload["error"] == "lane_timeout"
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
        app, blackboard, lane, _loop = mesh_app_with_back_edge
        rec = _record(
            "task_self_1",
            assignee="scout",
            origin={"kind": "agent", "channel": "", "user": "scout"},
        )

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
        )
        _settle()

        rows = blackboard.list_by_prefix("inbox/scout/task_event/")
        assert not any(
            r.value.get("task_id") == "task_self_1" for r in rows
        )
        assert lane.calls == []

    def test_human_origin_skipped(self, mesh_app_with_back_edge):
        """``origin.kind=human`` callers (channels, web chat) don't get
        a mesh-side back-edge — humans are served by the lane-worker
        auto-notify path instead."""
        app, blackboard, lane, _loop = mesh_app_with_back_edge
        rec = _record(
            "task_human_1",
            origin={"kind": "human", "channel": "telegram", "user": "9999"},
        )

        app._write_task_event_back_edge(
            rec, event_kind="task_failed",
            payload_extras={"error": "boom"},
        )
        _settle()

        rows = blackboard.list_by_prefix("inbox/9999/task_event/")
        assert not any(
            r.value.get("task_id") == "task_human_1" for r in rows
        )
        assert lane.calls == []
