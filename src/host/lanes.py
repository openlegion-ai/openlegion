"""Per-agent lane queues with message queue modes.

Each agent gets its own FIFO queue. Tasks within a lane execute serially
(one at a time per agent), but lanes run in parallel (different agents
can work simultaneously).

Two queue modes control how incoming messages interact with busy agents:

- **followup** (default): FIFO — queue, process after current task.
- **steer**: Inject into active conversation between tool rounds.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from src.host.orchestration import InvalidStatusTransition
from src.shared.types import SILENT_REPLY_TOKEN, MessageOrigin
from src.shared.utils import generate_id, setup_logging

logger = setup_logging("host.lanes")

_STEER_WAKEUP_MAX = 10  # max wakeups per window
_STEER_WAKEUP_WINDOW = 3600  # 1 hour window
_NOTIFY_FORWARD_TIMEOUT = 30  # seconds — cap on auto-notify send
# Per-task lane watchdog (Bug 4): a hung LLM stream or stuck tool call
# previously blocked the lane indefinitely — every subsequent task for that
# agent would queue forever. The default is generous (15 min) because some
# deep-research workloads legitimately take that long; tune via env var.
_DEFAULT_LANE_TIMEOUT_SECONDS = int(
    os.getenv("OPENLEGION_LANE_TIMEOUT_SECONDS", "900"),
)
# H7 — per-agent lane queue depth cap. Bounds the un-drained followup
# backlog so a flood of wakes/handoffs can't grow a single agent's queue
# without limit (memory + a guarantee that backpressure surfaces as a
# 429 rather than a silent unbounded build-up). 100 is generous: a lane
# drains serially, so a healthy agent rarely carries more than a handful
# queued, and 100 deep is already a strong signal something is looping.
# ``<= 0`` disables the bound (unbounded queue, pre-H7 behaviour).
_DEFAULT_LANE_QUEUE_MAX = int(
    os.getenv("OPENLEGION_LANE_QUEUE_MAX", "100"),
)


class LaneQueueFull(Exception):
    """Raised by ``enqueue`` when an agent's followup queue is at its
    depth cap. The mesh wake/task endpoints map this to HTTP 429 so the
    caller backs off rather than the lane silently dropping work."""


@dataclass
class QueuedTask:
    id: str
    agent: str
    message: str
    mode: str = "followup"
    trace_id: str | None = None
    origin: MessageOrigin | None = None
    auto_notify: bool = False
    task_id: str | None = None
    future: asyncio.Future = field(default_factory=asyncio.Future)


class LaneManager:
    """Manages per-agent FIFO queues with serial execution and queue modes."""

    def __init__(
        self,
        dispatch_fn: Callable[..., Coroutine[Any, Any, str]],
        steer_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None,
        trace_store: Any = None,
        notify_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None,
        tasks_store: Any = None,
        task_timeout_seconds: int | None = None,
        quarantine_check: Callable[[str], bool] | None = None,
        queue_maxsize: int | None = None,
    ):
        self._dispatch_fn = dispatch_fn
        # H7 — per-lane queue depth cap. Resolved once: kwarg wins, else
        # env default. ``<= 0`` means unbounded (asyncio.Queue(maxsize=0)).
        self._queue_maxsize = (
            queue_maxsize
            if queue_maxsize is not None
            else _DEFAULT_LANE_QUEUE_MAX
        )
        self._steer_fn = steer_fn
        self._trace_store = trace_store
        self._notify_fn = notify_fn
        # ``tasks_store`` (Bug 4): when wired, the lane watchdog marks the
        # durable task ``failed`` with ``error="lane_timeout"`` on timeout
        # so originators see the back-edge event instead of waiting forever.
        # Lane keeps working without it — falls back to a loud log line.
        self._tasks_store = tasks_store
        # ``task_timeout_seconds`` (Bug 4): per-task wall-clock cap. Read
        # from env at construction so each LaneManager honours its own
        # override (and tests can pass a tight value).
        if task_timeout_seconds is None:
            task_timeout_seconds = _DEFAULT_LANE_TIMEOUT_SECONDS
        self._task_timeout_seconds = task_timeout_seconds
        # ``quarantine_check`` (Fix 5 in seam follow-up): callable that
        # returns True when the agent is currently quarantined. Wired
        # post-construction from runtime.py to
        # ``HealthMonitor.is_quarantined``. When the lane dequeues work
        # for a quarantined agent, the task is rejected with an actionable
        # error instead of dispatched — the agent will keep failing on a
        # broken credential otherwise.
        self._quarantine_check = quarantine_check
        # ``_back_edge_fn``: callable wired by the mesh app after both
        # this LaneManager and ``create_mesh_app`` have been constructed
        # (the helper is a closure over the app's blackboard / router).
        # Called from the lane-timeout path so the originator's back-edge
        # inbox sees a ``task_failed`` event AND gets a wake-on-event
        # followup — without this, lane-timeout failures land in SQLite
        # but never surface to the originating agent's awareness loop.
        self._back_edge_fn: (
            Callable[..., None] | None
        ) = None
        # Per-agent timeout overlay. ``set_agent_timeout`` writes here;
        # ``_timeout_for`` falls back to the module default when unset.
        # Lets workflow-stage agents (page-writer, etc.) run on a tight
        # 10-min cap while deep-research agents keep the 15-min default.
        self._per_agent_timeouts: dict[str, int] = {}
        self._queues: dict[str, asyncio.Queue[QueuedTask]] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._pending: dict[str, list[QueuedTask]] = {}
        self._busy: dict[str, bool] = {}
        self._state_locks: dict[str, asyncio.Lock] = {}
        self._steer_wakeup_ts: dict[str, list[float]] = {}
        # Strong references to in-flight auto-notify forward tasks.  The
        # asyncio event loop only holds weak references, so a bare
        # ``create_task(_forward())`` can be garbage-collected mid-flight
        # per the Python docs warning.
        self._forward_tasks: set[asyncio.Task] = set()

    def set_tasks_store(self, store: Any) -> None:
        """Wire the durable tasks store after construction.

        ``LaneManager`` is built before the mesh app (which owns the tasks
        store), so the bootstrap path needs to inject the store post-hoc.
        Mirrors the ``HealthMonitor.set_queue_depth_fn`` pattern. Pass
        ``None`` to clear.
        """
        self._tasks_store = store

    def set_back_edge_fn(
        self, fn: Callable[..., None] | None,
    ) -> None:
        """Wire the mesh's back-edge writer.

        ``fn(task_record, *, event_kind, payload_extras=None)`` writes a
        back-edge event to the originator's inbox and (for actionable
        kinds) wakes the originator with a rate-limit. Called from the
        lane-timeout path. Wired by ``create_mesh_app`` after the helper
        closure exists. Pass ``None`` to clear.
        """
        self._back_edge_fn = fn

    def set_agent_timeout(self, agent: str, seconds: int | None) -> None:
        """Set a per-agent override on the per-task wall-clock cap.

        ``seconds`` < 60 is clamped to 60 (a 1-second cap would make the
        watchdog the bottleneck instead of a safety net). Pass ``None``
        to drop back to the module default. Per-agent overrides are
        useful for workflow stages with predictable bounds (tight cap →
        faster recovery from a wedged step) vs deep-research agents that
        need the generous default.
        """
        if seconds is None:
            self._per_agent_timeouts.pop(agent, None)
            return
        self._per_agent_timeouts[agent] = max(60, int(seconds))

    def _timeout_for(self, agent: str) -> int:
        """Resolve the per-task wall-clock cap for ``agent``."""
        return self._per_agent_timeouts.get(
            agent, self._task_timeout_seconds,
        )

    def set_quarantine_check(
        self, check: Callable[[str], bool] | None,
    ) -> None:
        """Wire the quarantine check after construction (Fix 5).

        Both ``LaneManager`` and ``HealthMonitor`` are built in
        runtime.py; if construction order makes constructor injection
        awkward, this setter lets the lane wire ``HealthMonitor.is_quarantined``
        in once the monitor is available. ``None`` disables the check.
        """
        self._quarantine_check = check

    def lane_full(self, agent: str) -> bool:
        """Return True when ``agent``'s lane queue is at its depth cap.

        H7 pre-flight check for callers that enqueue cross-thread (fire-
        and-forget on the dispatch loop) and therefore can't catch
        ``LaneQueueFull`` synchronously. ``qsize()`` is a plain int read.
        Unbounded lanes (``maxsize<=0``) are never full. A not-yet-created
        lane is never full (it'll be created with headroom on first put)."""
        if self._queue_maxsize <= 0:
            return False
        q = self._queues.get(agent)
        if q is None:
            return False
        return q.qsize() >= self._queue_maxsize

    def _ensure_lane(self, agent: str) -> None:
        """Lazily create queue, worker, and tracking structures for an agent."""
        if agent not in self._queues:
            # H7 — bound the followup backlog. ``maxsize<=0`` →
            # unbounded (asyncio semantics), preserving the escape hatch.
            self._queues[agent] = asyncio.Queue(
                maxsize=max(0, self._queue_maxsize),
            )
            self._pending[agent] = []
            self._busy[agent] = False
            self._state_locks[agent] = asyncio.Lock()
            self._workers[agent] = asyncio.create_task(self._worker(agent))

    async def enqueue(
        self, agent: str, message: str, *, mode: str = "followup",
        trace_id: str | None = None,
        origin: MessageOrigin | None = None,
        auto_notify: bool = False,
        task_id: str | None = None,
    ) -> str:
        """Queue a message for an agent with the specified mode.

        Modes:
          followup — default FIFO, process after current task.
          steer    — inject into active conversation between tool rounds.

        When ``origin`` and ``auto_notify=True`` are set, the lane worker will
        forward the completed task result back to the originating channel+user
        via the configured ``notify_fn``.

        ``task_id`` (Bug 2/3 fix) plumbs through to the dispatched ``/chat``
        call so the recipient agent's loop can auto-close the task on
        completion. Only honoured for ``followup`` mode; steer
        intentionally doesn't wire it (it isn't single-task semantics).
        """
        self._ensure_lane(agent)

        if mode == "steer":
            return await self._handle_steer(agent, message)
        return await self._handle_followup(
            agent, message, trace_id=trace_id,
            origin=origin, auto_notify=auto_notify,
            task_id=task_id,
        )

    async def _handle_followup(
        self, agent: str, message: str, *, trace_id: str | None = None,
        origin: MessageOrigin | None = None,
        auto_notify: bool = False,
        task_id: str | None = None,
    ) -> str:
        """Standard FIFO enqueue."""
        task = QueuedTask(
            id=generate_id("lane"),
            agent=agent,
            message=message,
            mode="followup",
            trace_id=trace_id,
            origin=origin,
            auto_notify=auto_notify,
            task_id=task_id,
        )
        # H7 — non-blocking put so a full lane surfaces backpressure
        # (LaneQueueFull → HTTP 429) instead of silently awaiting forever
        # or growing without bound. ``_pending`` is only appended on a
        # successful put so it stays in lockstep with the queue.
        try:
            self._queues[agent].put_nowait(task)
        except asyncio.QueueFull:
            logger.warning(
                "Lane queue full for agent '%s' (depth: %d/%d) — rejecting "
                "with backpressure",
                agent, self._queues[agent].qsize(), self._queue_maxsize,
            )
            raise LaneQueueFull(
                f"agent '{agent}' lane queue is full "
                f"(depth {self._queues[agent].qsize()}/{self._queue_maxsize})"
            )
        self._pending[agent].append(task)
        logger.debug(f"Queued task {task.id} for agent '{agent}' (depth: {self._queues[agent].qsize()})")
        return await task.future

    async def _handle_steer(self, agent: str, message: str) -> str:
        """Inject a steer message into the agent's active conversation.

        If a steer_fn is available, calls it directly (bypasses queue).
        Falls back to followup if no steer_fn is configured.
        """
        if self._steer_fn is None:
            logger.debug(f"No steer_fn configured, falling back to followup for '{agent}'")
            return await self._handle_followup(agent, message)

        try:
            result = await self._steer_fn(agent, message)
            injected = result.get("injected", False) if isinstance(result, dict) else False
            if injected:
                return f"Steered: message injected into {agent}'s active conversation"
            else:
                # Agent is idle — dispatch as followup to wake it up.
                # Rate-limited to prevent event storms from draining budget.
                if self._check_steer_wakeup_rate(agent):
                    logger.debug(f"Waking idle agent '{agent}' via followup (steer not injected)")
                    return await self._handle_followup(agent, message)
                logger.debug(f"Steer wakeup rate-limited for idle agent '{agent}', dropping")
                return SILENT_REPLY_TOKEN
        except Exception as e:
            logger.warning(f"Steer to '{agent}' failed, falling back to followup: {e}")
            return await self._handle_followup(agent, message)

    def _check_steer_wakeup_rate(self, agent: str) -> bool:
        """Return True if the agent hasn't exceeded the steer-wakeup rate limit."""
        now = time.monotonic()
        ts = self._steer_wakeup_ts.get(agent, [])
        # Prune old timestamps
        ts = [t for t in ts if now - t < _STEER_WAKEUP_WINDOW]
        if len(ts) >= _STEER_WAKEUP_MAX:
            return False
        ts.append(now)
        self._steer_wakeup_ts[agent] = ts
        return True

    async def _worker(self, agent: str) -> None:
        """Worker loop: drains the queue for a single agent serially."""
        from src.shared.trace import current_trace_id

        queue = self._queues[agent]
        lock = self._state_locks[agent]
        while True:
            try:
                task = await queue.get()
            except asyncio.CancelledError:
                return
            # Fix 5 (seam follow-up): refuse dispatch to a quarantined
            # agent. The agent will keep failing on a broken credential
            # otherwise; surface the actionable hint and free the lane.
            #
            # Codex P2 follow-up: when the queued wake carries a task_id
            # (handoff path), mark the durable task ``failed`` too — same
            # pattern as the lane timeout branch — so the originating
            # agent's back-edge inbox sees a terminal event instead of
            # the task dangling in ``working`` until the stale-task
            # reaper picks it up.
            if self._quarantine_check and self._quarantine_check(agent):
                err_msg = (
                    f"Agent '{agent}' is quarantined (credential failure). "
                    f"Fix via edit_agent(field='model', value=...) — "
                    f"quarantine clears on a successful edit. "
                    f"Task not dispatched."
                )
                logger.warning(err_msg)
                if task.task_id and self._tasks_store is not None:
                    try:
                        # Bug 3: surface failure reason via blocker_note so
                        # the dashboard/workflow_snapshot show *why* the
                        # task failed, not a bare "failed" pill.
                        _quarantine_note = (
                            f"agent_quarantined: '{agent}' quarantined "
                            f"on credential failure (fix via edit_agent)"
                        )[:500]
                        await asyncio.get_running_loop().run_in_executor(
                            None,
                            lambda: self._tasks_store.update_status(
                                task.task_id,
                                "failed",
                                actor="lane_quarantine",
                                blocker_note=_quarantine_note,
                                extra_payload={
                                    "error": "agent_quarantined",
                                    "agent": agent,
                                },
                            ),
                        )
                    except Exception as close_err:
                        logger.warning(
                            "Lane quarantine failed to close task %s: %s",
                            task.task_id, close_err,
                        )
                if not task.future.done():
                    task.future.set_exception(RuntimeError(err_msg))
                try:
                    self._pending[agent].remove(task)
                except ValueError:
                    pass
                queue.task_done()
                continue
            async with lock:
                self._busy[agent] = True
            current_trace_id.set(task.trace_id)
            t0 = time.time()
            if task.trace_id and self._trace_store:
                self._trace_store.record(
                    trace_id=task.trace_id, source="lane", agent=agent,
                    event_type="lane_start", detail=task.message[:200],
                    meta={"mode": task.mode, "queue_depth": queue.qsize()},
                )
            try:
                # Build kwargs lazily — older test/dispatch signatures
                # accept only ``(agent, message)`` (no **kwargs), so only
                # pass extras when they're actually set. Preserves the
                # pre-existing "origin only when present" contract while
                # threading the new task_id through the same gate.
                dispatch_kwargs: dict[str, Any] = {}
                if task.origin is not None:
                    dispatch_kwargs["origin"] = task.origin
                if task.task_id is not None:
                    dispatch_kwargs["task_id"] = task.task_id
                # Bug 4: per-task wall-clock cap. A hung LLM stream or
                # stuck tool previously blocked the lane forever — every
                # subsequent task for this agent would queue forever.
                # Per-agent override takes precedence over the default.
                resolved_timeout = self._timeout_for(agent)
                if dispatch_kwargs:
                    result = await asyncio.wait_for(
                        self._dispatch_fn(agent, task.message, **dispatch_kwargs),
                        timeout=resolved_timeout,
                    )
                else:
                    result = await asyncio.wait_for(
                        self._dispatch_fn(agent, task.message),
                        timeout=resolved_timeout,
                    )
                task.future.set_result(result)
                # Auto-forward result to origin channel+user when requested.
                # Runs as a background task with a timeout so a hung channel
                # send does not leak a coroutine or stall the worker.
                if (
                    task.auto_notify
                    and task.origin
                    and self._notify_fn
                    and isinstance(result, str)
                    and result.strip()
                    and result != SILENT_REPLY_TOKEN
                ):
                    fn = self._notify_fn
                    forward_origin = task.origin
                    task_agent = task.agent
                    task_result = result

                    async def _forward():
                        try:
                            await asyncio.wait_for(
                                fn(forward_origin, task_result, task_agent),
                                timeout=_NOTIFY_FORWARD_TIMEOUT,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Lane auto-notify to origin %s timed out after %ds",
                                forward_origin, _NOTIFY_FORWARD_TIMEOUT,
                            )
                        except Exception as fwd_e:
                            logger.warning(
                                "Lane auto-notify to origin %s failed: %s",
                                forward_origin, fwd_e,
                            )

                    forward_task = asyncio.create_task(_forward())
                    self._forward_tasks.add(forward_task)
                    forward_task.add_done_callback(self._forward_tasks.discard)
            except asyncio.TimeoutError:
                # Bug 4: dispatch exceeded the per-task wall-clock cap.
                # Free the lane, mark the durable task ``failed`` so the
                # originating agent's back-edge inbox sees the timeout,
                # then continue serving the queue. Do NOT raise — the
                # worker must survive a single stuck task.
                effective_timeout = self._timeout_for(agent)
                timeout_msg = (
                    f"Lane task {task.id} for agent={agent} timed out after "
                    f"{effective_timeout}s — freeing lane"
                )
                logger.error(timeout_msg)
                fresh_record: dict | None = None
                already_terminal = False
                if task.task_id and self._tasks_store is not None:
                    # Pre-flight check: read the current status so we can
                    # tell whether the watchdog actually transitioned the
                    # row OR the assignee already wrote a terminal status
                    # while we were timing out. ``Tasks.update_status``
                    # treats ``current == status`` as a no-op success
                    # (returns the row without raising), so without this
                    # check the watchdog would happily overwrite the
                    # assignee's actual back-edge payload with a generic
                    # ``error="lane_timeout"`` at the same blackboard key.
                    try:
                        pre = self._tasks_store.get(task.task_id)
                        if pre is not None and pre.get("status") in (
                            "done", "failed", "cancelled"
                        ):
                            already_terminal = True
                            logger.info(
                                "Lane watchdog: task %s already terminal "
                                "(status=%s) — skipping update + back-edge",
                                task.task_id, pre.get("status"),
                            )
                    except Exception as pre_err:
                        logger.debug(
                            "Lane watchdog pre-check for %s failed: %s",
                            task.task_id, pre_err,
                        )
                    if not already_terminal:
                        try:
                            # ``Tasks.update_status`` is sync — push it
                            # off the event loop so a slow SQLite write
                            # can't pile back onto the lane worker we
                            # just freed. ``InvalidStatusTransition``
                            # here means a benign race terminated the
                            # task between pre-check and UPDATE.
                            _timeout_note = (
                                f"lane_timeout: task exceeded "
                                f"{effective_timeout}s wall-clock cap"
                            )[:500]
                            await asyncio.get_running_loop().run_in_executor(
                                None,
                                lambda: self._tasks_store.update_status(
                                    task.task_id,
                                    "failed",
                                    actor="lane_watchdog",
                                    blocker_note=_timeout_note,
                                    extra_payload={
                                        "error": "lane_timeout",
                                        "timeout_seconds": effective_timeout,
                                    },
                                ),
                            )
                            fresh_record = self._tasks_store.get(task.task_id)
                        except InvalidStatusTransition as race_err:
                            logger.info(
                                "Lane watchdog race: task %s went "
                                "terminal during update (%s) — skipping "
                                "back-edge",
                                task.task_id, race_err,
                            )
                        except Exception as close_err:
                            logger.warning(
                                "Lane watchdog failed to close task %s: %s",
                                task.task_id, close_err,
                            )
                # Fire the back-edge writer so the originating agent
                # learns about the timeout via inbox event + wake (for
                # the actionable ``task_failed`` kind). Best-effort.
                # SQLite-bound work is pushed to an executor so a
                # blackboard write under contention can't block the
                # shared lane-worker event loop for all agents.
                if (
                    fresh_record is not None
                    and self._back_edge_fn is not None
                ):
                    try:
                        await asyncio.get_running_loop().run_in_executor(
                            None,
                            lambda: self._back_edge_fn(
                                fresh_record,
                                event_kind="task_failed",
                                payload_extras={
                                    "error": "lane_timeout",
                                    "timeout_seconds": effective_timeout,
                                },
                            ),
                        )
                    except Exception as be_err:
                        logger.warning(
                            "Lane watchdog back-edge fire failed for "
                            "task %s: %s", task.task_id, be_err,
                        )
                if not task.future.done():
                    task.future.set_exception(
                        asyncio.TimeoutError(timeout_msg),
                    )
            except Exception as e:
                if not task.future.done():
                    task.future.set_exception(e)
                logger.error(f"Lane task {task.id} for '{agent}' failed: {e}")
            finally:
                duration_ms = int((time.time() - t0) * 1000)
                if task.trace_id and self._trace_store:
                    self._trace_store.record(
                        trace_id=task.trace_id, source="lane", agent=agent,
                        event_type="lane_complete", duration_ms=duration_ms,
                        status="error" if task.future.done() and task.future.exception() else "ok",
                        error=str(task.future.exception()) if task.future.done() and task.future.exception() else "",
                    )
                async with lock:
                    self._busy[agent] = False
                    pending = self._pending.get(agent, [])
                    if task in pending:
                        pending.remove(task)
                queue.task_done()

    def get_status(self) -> dict[str, dict]:
        """Return queue depth, pending task count, and busy flag per agent."""
        result = {}
        for agent, queue in self._queues.items():
            result[agent] = {
                "queued": queue.qsize(),
                "pending": len(self._pending.get(agent, [])),
                "busy": self._busy.get(agent, False),
            }
        return result

    def get_queue_depth(self, agent: str) -> int:
        """Return current queue depth for an agent (0 if no lane exists).

        Used by HealthMonitor's loop-liveness check (Bug 1) — combine with
        the agent's ``last_iteration_ts`` from /status to detect a wedged
        inner loop with a live FastAPI thread. We count both queued
        messages waiting and an in-flight task (``busy``) so the "agent has
        actual work" predicate fires correctly during the gap between
        dequeue and dispatch.
        """
        queue = self._queues.get(agent)
        depth = queue.qsize() if queue is not None else 0
        if self._busy.get(agent, False):
            depth += 1
        return depth

    def remove_lane(self, agent: str) -> None:
        """Remove all lane state for an agent, cancelling its worker task."""
        worker = self._workers.pop(agent, None)
        if worker is not None:
            worker.cancel()
        self._queues.pop(agent, None)
        self._pending.pop(agent, None)
        self._busy.pop(agent, None)
        self._state_locks.pop(agent, None)
        self._steer_wakeup_ts.pop(agent, None)

    async def stop(self) -> None:
        """Cancel all worker tasks and any in-flight auto-notify forwards."""
        for task in self._workers.values():
            task.cancel()
        self._workers.clear()
        for fwd in list(self._forward_tasks):
            fwd.cancel()
        self._forward_tasks.clear()
