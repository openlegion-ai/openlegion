"""Per-agent lane queues for serial task execution.

Each agent gets its own FIFO queue. Tasks within a lane execute serially
(one at a time per agent), but lanes run in parallel (different agents
can work simultaneously).

When the orchestrator or cron dispatches to a busy agent, the request
is queued rather than rejected. The lane drains automatically.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from src.shared.utils import generate_id, setup_logging

logger = setup_logging("host.lanes")


@dataclass
class QueuedTask:
    id: str
    agent: str
    message: str
    future: asyncio.Future = field(default_factory=asyncio.Future)


class LaneManager:
    """Manages per-agent FIFO queues with serial execution."""

    def __init__(self, dispatch_fn: Callable[..., Coroutine[Any, Any, str]]):
        self._dispatch_fn = dispatch_fn
        self._queues: dict[str, asyncio.Queue[QueuedTask]] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._pending: dict[str, list[QueuedTask]] = {}

    async def enqueue(self, agent: str, message: str) -> str:
        """Queue a message for an agent. Returns the response when done."""
        if agent not in self._queues:
            self._queues[agent] = asyncio.Queue()
            self._pending[agent] = []
            self._workers[agent] = asyncio.create_task(self._worker(agent))

        task = QueuedTask(
            id=generate_id("lane"),
            agent=agent,
            message=message,
        )
        self._pending[agent].append(task)
        await self._queues[agent].put(task)
        logger.debug(f"Queued task {task.id} for agent '{agent}' (depth: {self._queues[agent].qsize()})")
        return await task.future

    async def _worker(self, agent: str) -> None:
        """Worker loop: drains the queue for a single agent serially."""
        queue = self._queues[agent]
        while True:
            task = await queue.get()
            try:
                result = await self._dispatch_fn(agent, task.message)
                task.future.set_result(result)
            except Exception as e:
                if not task.future.done():
                    task.future.set_exception(e)
                logger.error(f"Lane task {task.id} for '{agent}' failed: {e}")
            finally:
                if agent in self._pending and task in self._pending[agent]:
                    self._pending[agent].remove(task)
                queue.task_done()

    def get_status(self) -> dict[str, dict]:
        """Return queue depth and pending task count per agent."""
        result = {}
        for agent, queue in self._queues.items():
            result[agent] = {
                "queued": queue.qsize(),
                "pending": len(self._pending.get(agent, [])),
            }
        return result

    async def stop(self) -> None:
        """Cancel all worker tasks."""
        for task in self._workers.values():
            task.cancel()
        self._workers.clear()
