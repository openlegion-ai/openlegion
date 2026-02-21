"""Per-agent lane queues with message queue modes.

Each agent gets its own FIFO queue. Tasks within a lane execute serially
(one at a time per agent), but lanes run in parallel (different agents
can work simultaneously).

Three queue modes control how incoming messages interact with busy agents:

- **followup** (default): FIFO — queue, process after current task.
- **steer**: Inject into active conversation between tool rounds.
- **collect**: Batch queued messages into a single dispatch when agent becomes free.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from src.shared.utils import generate_id, setup_logging

logger = setup_logging("host.lanes")

SILENT_REPLY_TOKEN = "__SILENT__"


@dataclass
class QueuedTask:
    id: str
    agent: str
    message: str
    mode: str = "followup"
    trace_id: str | None = None
    future: asyncio.Future = field(default_factory=asyncio.Future)


class LaneManager:
    """Manages per-agent FIFO queues with serial execution and queue modes."""

    def __init__(
        self,
        dispatch_fn: Callable[..., Coroutine[Any, Any, str]],
        steer_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ):
        self._dispatch_fn = dispatch_fn
        self._steer_fn = steer_fn
        self._queues: dict[str, asyncio.Queue[QueuedTask]] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._pending: dict[str, list[QueuedTask]] = {}
        self._collect_buffers: dict[str, list[str]] = {}
        self._busy: dict[str, bool] = {}

    def _ensure_lane(self, agent: str) -> None:
        """Lazily create queue, worker, and tracking structures for an agent."""
        if agent not in self._queues:
            self._queues[agent] = asyncio.Queue()
            self._pending[agent] = []
            self._busy[agent] = False
            self._collect_buffers[agent] = []
            self._workers[agent] = asyncio.create_task(self._worker(agent))

    async def enqueue(
        self, agent: str, message: str, *, mode: str = "followup", trace_id: str | None = None,
    ) -> str:
        """Queue a message for an agent with the specified mode.

        Modes:
          followup — default FIFO, process after current task.
          steer    — inject into active conversation between tool rounds.
          collect  — batch when busy, dispatch combined when agent becomes free.
        """
        self._ensure_lane(agent)

        if mode == "steer":
            return await self._handle_steer(agent, message)
        elif mode == "collect":
            return await self._handle_collect(agent, message)
        else:
            return await self._handle_followup(agent, message, trace_id=trace_id)

    async def _handle_followup(
        self, agent: str, message: str, *, trace_id: str | None = None,
    ) -> str:
        """Standard FIFO enqueue."""
        task = QueuedTask(
            id=generate_id("lane"),
            agent=agent,
            message=message,
            mode="followup",
            trace_id=trace_id,
        )
        self._pending[agent].append(task)
        await self._queues[agent].put(task)
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
                return f"Steered: message queued for {agent} (agent is idle, will see it next turn)"
        except Exception as e:
            logger.warning(f"Steer to '{agent}' failed, falling back to followup: {e}")
            return await self._handle_followup(agent, message)

    async def _handle_collect(self, agent: str, message: str) -> str:
        """Batch messages when agent is busy, dispatch immediately when idle."""
        if not self._busy.get(agent, False):
            # Agent is idle — dispatch immediately
            return await self._handle_followup(agent, message)

        # Agent is busy — buffer the message
        self._collect_buffers[agent].append(message)
        logger.debug(
            f"Collected message for '{agent}' "
            f"(buffer size: {len(self._collect_buffers[agent])})"
        )
        return SILENT_REPLY_TOKEN

    def _flush_collect_buffer(self, agent: str) -> None:
        """Drain the collect buffer and enqueue a combined message as followup."""
        buffer = self._collect_buffers.get(agent, [])
        if not buffer:
            return

        messages = list(buffer)
        buffer.clear()

        if len(messages) == 1:
            combined = messages[0]
        else:
            parts = [f"[Message {i + 1}]: {msg}" for i, msg in enumerate(messages)]
            combined = "\n\n".join(parts)

        task = QueuedTask(
            id=generate_id("lane"),
            agent=agent,
            message=combined,
            mode="followup",
        )
        # Suppress "Future exception was never retrieved" — no caller awaits this.
        task.future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        self._pending[agent].append(task)
        self._queues[agent].put_nowait(task)
        logger.debug(
            f"Flushed {len(messages)} collected message(s) as task {task.id} for '{agent}'"
        )

    async def _worker(self, agent: str) -> None:
        """Worker loop: drains the queue for a single agent serially."""
        from src.shared.trace import current_trace_id

        queue = self._queues[agent]
        while True:
            task = await queue.get()
            self._busy[agent] = True
            current_trace_id.set(task.trace_id)
            try:
                result = await self._dispatch_fn(agent, task.message)
                task.future.set_result(result)
            except Exception as e:
                if not task.future.done():
                    task.future.set_exception(e)
                logger.error(f"Lane task {task.id} for '{agent}' failed: {e}")
            finally:
                self._busy[agent] = False
                if agent in self._pending and task in self._pending[agent]:
                    self._pending[agent].remove(task)
                queue.task_done()
                self._flush_collect_buffer(agent)

    def get_status(self) -> dict[str, dict]:
        """Return queue depth, pending task count, collected count, and busy flag per agent."""
        result = {}
        for agent, queue in self._queues.items():
            result[agent] = {
                "queued": queue.qsize(),
                "pending": len(self._pending.get(agent, [])),
                "collected": len(self._collect_buffers.get(agent, [])),
                "busy": self._busy.get(agent, False),
            }
        return result

    async def stop(self) -> None:
        """Cancel all worker tasks."""
        for task in self._workers.values():
            task.cancel()
        self._workers.clear()
