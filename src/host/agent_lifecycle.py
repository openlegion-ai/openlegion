"""Per-agent lifecycle serialisation.

``RuntimeBackend._agent_locked`` makes a single ``start_agent`` /
``stop_agent`` call atomic for one agent. That is the innermost layer only.
A *lifecycle operation* — create, delete, restart, archive, wake — is a
sequence of steps spanning five subsystems (agents.yaml + permissions, the
runtime backend, the router/transport registries, the health monitor, and
cron), and none of that sequence is atomic today. Two of them running at
once for the same agent interleave, and the durable state ends up
disagreeing with the runtime state:

* **delete vs restart.** A restart releases the loop between its stop and
  its start. A delete landing in that window destroys the volume, drops the
  config and unregisters the agent — and then the restart's ``start_agent``
  puts a fresh container back and re-registers it. The agent is resurrected
  with no config row, no permissions row, and a live mesh auth token.
* **delete vs create.** Once the delete has unregistered the id, a create of
  the same name gets past the ``already exists`` check and writes its config
  — which the delete, still running, then removes. The new agent is left
  running and routable with nothing on disk describing it.
* **archive vs wake.** Archive unregisters health and stops the container;
  wake, already past its own ``start_agent``, flips the status back to
  ``active``. The status says running, the container is not, and nothing is
  monitoring it.

So lifecycle operations take a lock of their own, one layer up from the
backend's. Lock order is ALWAYS this lock first, then
``RuntimeBackend._agent_locked`` (which start/stop take internally); nothing
acquires them the other way round.

Two things this lock deliberately does not do:

* **It is not re-entrant.** A lifecycle operation nested inside another for
  the same agent is a bug, not a pattern to support, so a re-acquire from
  the same holder raises rather than deadlocking or silently passing
  through. (The detection only covers a nest on the same thread and task;
  one that hops threads still deadlocks — hence the rule below.)
* **A cancelled operation releases the lock; its container call may not
  have finished.** The routes cap ``start_agent`` / ``stop_agent`` at 60s
  with ``asyncio.wait_for(asyncio.to_thread(...))`` so a hung Docker daemon
  returns control, and cancelling that wait does not cancel the worker
  thread. What the lock still guarantees in that case comes from the layer
  below: ``RuntimeBackend._agent_locked`` brackets the whole container call,
  so the next lifecycle operation's start/stop blocks on it and the two
  registries stay coherent. What is NOT guaranteed is the rest of the
  sequence — a start that completes after its route gave up leaves a
  running container with no router, transport or health registration. That
  predates this lock and is unchanged by it.
* **Nothing that can dispatch to an agent may run inside a held region.**
  The cold-wake seam is wired into the transport and the mesh router, so
  *any* message routed to a hibernated agent calls ``ensure_agent_running``,
  which takes this lock. A handover turn (``_offboard_agent``) or a standup
  post inside a locked region would therefore deadlock against itself. Run
  those before taking the lock — they are best-effort steps against an agent
  that is still fully registered anyway.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator

from src.shared.utils import setup_logging

logger = setup_logging("host.agent_lifecycle")

# How long an async acquire waits before giving up. Every locked region is
# internally bounded (the stop/start calls inside them are wrapped in 60s
# ``asyncio.wait_for`` timeouts), so reaching this means a lifecycle
# operation is genuinely wedged — at which point a 409 tells the operator
# far more than a request that never returns.
DEFAULT_LIFECYCLE_TIMEOUT = 300.0

# Poll interval for the async acquire. The wait is a poll rather than a
# blocking acquire in a worker thread on purpose: a cancelled coroutine
# would abandon that thread mid-acquire and leak the lock forever, whereas
# a cancellation here lands on the sleep with the lock not held.
_POLL_INTERVAL = 0.02

# ``{agent_id: [Lock, refcount]}`` — refcounted so ids that appear once
# (ephemeral spawns, an agent created and deleted) don't leak a lock each.
_locks: dict[str, list] = {}
_guard = threading.Lock()

# ``{agent_id: holder_token}`` for the re-entrancy check, written by the
# holder under ``_guard``.
_holders: dict[str, tuple] = {}

# ``{agent_id: seq}`` — the value of a process-global counter at the moment
# that id was last retired. See :func:`agent_incarnation`. The counter is
# global rather than per-id so that an evicted entry can never be recreated
# with a value some in-flight operation is still holding.
_incarnations: dict[str, int] = {}
_incarnation_seq = 0
# What an id that is NOT in the table reads as. Rises to the highest value
# ever evicted, so a captured value can never be re-issued below it.
# Bounded: ephemeral spawns mint a fresh id per call and retire it at TTL,
# so an unbounded table would grow for the life of the process.
_incarnation_floor = 0
_MAX_INCARNATIONS = 4096

# Identifies THIS process. The sequence above is in-memory and restarts at
# zero, so a value written somewhere durable — a pending-action row that
# outlives a mesh restart — has to carry the process it came from or it
# will compare equal to a fresh process's zero for a different agent.
_BOOT_ID = uuid.uuid4().hex


class AgentLifecycleBusy(Exception):
    """Another lifecycle operation for this agent is still running."""

    def __init__(self, agent_id: str, timeout: float):
        self.agent_id = agent_id
        self.timeout = timeout
        super().__init__(
            f"another lifecycle operation for agent '{agent_id}' is still "
            f"in progress after {timeout:g}s",
        )


def _holder_token() -> tuple:
    """Identify the current holder well enough to catch a self-nest.

    A coroutine and a plain call on the same thread are different holders
    only if a task is running, so both halves are needed: the thread ident
    separates worker threads, the task id separates coroutines interleaved
    on one loop thread.
    """
    task = None
    with contextlib.suppress(RuntimeError):
        task = asyncio.current_task()
    return (threading.get_ident(), id(task) if task is not None else None)


def _checkout(agent_id: str) -> list:
    """Claim a reference on ``agent_id``'s lock entry and return it."""
    with _guard:
        entry = _locks.get(agent_id)
        if entry is None:
            entry = [threading.Lock(), 0]
            _locks[agent_id] = entry
        entry[1] += 1
        return entry


def _checkin(agent_id: str, entry: list) -> None:
    """Drop a reference, reclaiming the entry once nobody holds or waits."""
    with _guard:
        entry[1] -= 1
        if entry[1] <= 0 and _locks.get(agent_id) is entry:
            _locks.pop(agent_id, None)


def _claim_holder(agent_id: str, token: tuple) -> None:
    with _guard:
        _holders[agent_id] = token


def _release_holder(agent_id: str, token: tuple) -> None:
    with _guard:
        if _holders.get(agent_id) == token:
            _holders.pop(agent_id, None)


def _reject_self_nest(agent_id: str, token: tuple) -> None:
    with _guard:
        held_by = _holders.get(agent_id)
    if held_by == token:
        raise RuntimeError(
            f"lifecycle lock for agent '{agent_id}' is already held by this "
            f"caller — lifecycle operations must not nest (see "
            f"src/host/agent_lifecycle.py)",
        )


@contextlib.contextmanager
def agent_lifecycle_locked(agent_id: str) -> Iterator[None]:
    """Serialise one agent's lifecycle operation. Blocking; sync callers only.

    For the REPL, boot reconcile and any other caller that is not on an
    event loop. Async callers must use :func:`agent_lifecycle_locked_async`
    — this one blocks the thread it is called on, which on a loop thread
    would stall every other coroutine including the operation it is waiting
    for.
    """
    token = _holder_token()
    _reject_self_nest(agent_id, token)
    entry = _checkout(agent_id)
    lock: threading.Lock = entry[0]
    acquired = False
    try:
        lock.acquire()
        acquired = True
        _claim_holder(agent_id, token)
        yield
    finally:
        if acquired:
            _release_holder(agent_id, token)
            lock.release()
        _checkin(agent_id, entry)


@contextlib.asynccontextmanager
async def agent_lifecycle_locked_async(
    agent_id: str,
    timeout: float | None = DEFAULT_LIFECYCLE_TIMEOUT,
) -> AsyncIterator[None]:
    """Serialise one agent's lifecycle operation from a coroutine.

    Safe to use from any of this process's event loops — the underlying
    primitive is a ``threading.Lock``, not an ``asyncio.Lock`` whose waiter
    queue is bound to the loop that created it (the cold-wake seam alone is
    driven from three of them).

    Raises :class:`AgentLifecycleBusy` if ``timeout`` elapses first. Pass
    ``timeout=None`` to wait indefinitely.
    """
    token = _holder_token()
    _reject_self_nest(agent_id, token)
    entry = _checkout(agent_id)
    lock: threading.Lock = entry[0]
    acquired = False
    try:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not lock.acquire(blocking=False):
            if deadline is not None and time.monotonic() >= deadline:
                raise AgentLifecycleBusy(agent_id, timeout)
            await asyncio.sleep(_POLL_INTERVAL)
        acquired = True
        _claim_holder(agent_id, token)
        yield
    finally:
        if acquired:
            _release_holder(agent_id, token)
            lock.release()
        _checkin(agent_id, entry)


def agent_incarnation(agent_id: str) -> int:
    """How many times this id has been retired. Capture before you queue.

    Holding the lock proves nothing about WHICH agent you hold it for. Agent
    ids are names the operator reuses, and a delete followed by a create of
    the same name inside the window an operation waits produces a live agent
    with the name the operation checked — a different agent entirely. Every
    name-based re-check then passes, and the queued operation acts on the
    replacement: a delete destroys the fresh container and volume, a restart
    bounces it with the previous agent's role and model.

    So the pattern is: read this before you queue, compare it after you
    acquire, and bail if it moved::

        incarnation = agent_incarnation(agent_id)
        ...                                     # slow, unlocked work
        async with agent_lifecycle_locked_async(agent_id):
            if agent_incarnation(agent_id) != incarnation:
                raise HTTPException(404, "agent was replaced")

    Only :func:`retire_agent` moves it, so an id that was never deleted
    always compares equal and the check costs one dict lookup.
    """
    with _guard:
        return _incarnations.get(agent_id, _incarnation_floor)


def retire_agent(agent_id: str) -> int:
    """Mark this id's current agent gone. Call from every delete path.

    Under the agent's lifecycle lock, so that anything comparing against a
    captured incarnation sees the bump exactly when the delete's other
    effects become visible. Returns the new value.
    """
    global _incarnation_seq, _incarnation_floor
    with _guard:
        _incarnation_seq += 1
        _incarnations[agent_id] = _incarnation_seq
        if len(_incarnations) > _MAX_INCARNATIONS:
            # Drop the oldest half, and raise the floor past everything
            # dropped. Both halves matter: a per-id count would restart an
            # evicted id at 1 and hand a stale holder of 1 a false match,
            # and an unraised floor would let an id evicted at N read back
            # as 0 — the value a never-retired id reads. With a global
            # sequence and a rising floor an evicted entry can only ever
            # read HIGHER than what anyone captured, so eviction fails
            # closed: a spurious refusal, never a wrong agent acted on.
            for stale in list(_incarnations)[: _MAX_INCARNATIONS // 2]:
                if stale == agent_id:
                    continue
                _incarnation_floor = max(_incarnation_floor, _incarnations.pop(stale, 0))
        return _incarnation_seq


def agent_incarnation_token(agent_id: str) -> str:
    """An incarnation stamp safe to store somewhere durable.

    :func:`agent_incarnation` alone is not: it counts within one process
    and restarts at zero, so a stamp of ``0`` written before a restart
    matches a fresh process's ``0`` — for whichever agent holds the name
    afterwards. Pairing it with the process id makes a stamp from any
    earlier process compare unequal, which is the honest answer: that
    process's counter is gone and the claim cannot be verified.

    The agent id is part of the token so a stamp can only ever be matched
    against the agent it was minted for — two never-retired agents would
    otherwise share the same counter value and the same token.
    """
    return f"{_BOOT_ID}:{agent_id}:{agent_incarnation(agent_id)}"


def incarnation_token_matches(agent_id: str, token: str | None) -> bool:
    """True only if ``token`` was minted by THIS process for THIS agent.

    False for a missing token, a malformed one, and one from a previous
    process — all cases where the answer is "cannot be verified", which for
    an irreversible action means refuse.
    """
    if not token:
        return False
    return token == agent_incarnation_token(agent_id)


def lifecycle_refcount(agent_id: str) -> int:
    """Holders plus waiters on ``agent_id``'s lock. Test/diagnostic use."""
    with _guard:
        entry = _locks.get(agent_id)
        return entry[1] if entry else 0
