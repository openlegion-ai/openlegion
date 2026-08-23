"""A dedicated thread pool for the stores' blocking SQLite work.

``asyncio.to_thread`` is the documented idiom for getting blocking work off
the mesh loop (CLAUDE.md, "Event-loop discipline"), and everywhere else in
the host it is the right one. The LLM proxy is the exception, for two
reasons that only apply here.

**It would couple the ledger to the container lifecycle.** ``to_thread``
runs on the loop's DEFAULT executor, and that pool is already carrying the
work that must not queue: ``RuntimeBackend.stop_agent`` (a Docker call that
takes ~10s), health probes, and the Team Drive's git plumbing. The proxy
fires three or four hops per LLM call, so under fleet load the ledger would
be competing with them — and if ``costs.db`` is locked by another process
the SQLite busy timeout is 30 SECONDS, during which each waiting call holds
a pool worker. A handful of those and an agent restart stalls behind a
budget query. A separate pool makes that structurally impossible.

**The extra workers buy nothing anyway.** ``CostTracker`` and ``TraceStore``
each serialize their single connection on an ``RLock``, so at most one
thread per store is ever inside SQLite. Additional concurrency is pure
occupancy. Bounding it costs no throughput.

The one thing this must not lose relative to ``to_thread`` is the context:
``to_thread`` copies the calling context and ``run_in_executor`` does not,
and ``CostTracker.track`` stamps the usage row from the ``current_trace_id``
contextvar the proxy seeded from ``X-Trace-Id``. ``run_in_store_thread``
copies it explicitly; ``tests/test_proxy_blocking_io.py`` pins that a usage
row keeps its trace id across the hop.
"""

from __future__ import annotations

import asyncio
import atexit
import contextvars
import functools
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

# Generous enough that a trace write never queues behind a cost query, small
# enough to stay a bound. Per-store serialization is the real limit.
_MAX_WORKERS = max(4, min(8, (os.cpu_count() or 2) + 2))

_pool: ThreadPoolExecutor | None = None
_pool_pid: int | None = None
_pool_lock = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    """The store pool for THIS process, built on first use.

    Keyed on the pid: a ``fork()`` copies the executor object but none of its
    threads, so a child inheriting the parent's pool would submit work that
    nothing runs. The child builds its own and abandons the copy — deliberately
    without shutting it down, since joining threads that do not exist in this
    process hangs.
    """
    global _pool, _pool_pid
    pid = os.getpid()
    with _pool_lock:
        if _pool is None or _pool_pid != pid:
            _pool = ThreadPoolExecutor(
                max_workers=_MAX_WORKERS, thread_name_prefix=STORE_THREAD_PREFIX,
            )
            _pool_pid = pid
            atexit.register(_shutdown_at_exit, _pool, pid)
        return _pool


STORE_THREAD_PREFIX = "ol-store"


def _shutdown_at_exit(pool: ThreadPoolExecutor, pid: int) -> None:
    """Drop the pool at interpreter exit, but only in the process that made it."""
    if os.getpid() != pid:
        return
    pool.shutdown(wait=False)


async def run_in_store_thread(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking store call off the loop, preserving the caller's context."""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    call = functools.partial(ctx.run, functools.partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_executor(), call)
