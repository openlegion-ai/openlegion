"""Loop affinity: who owns which event loop, and what happens when that's wrong.

The host used to run six event loops. Four of them existed only because a
periodic sweep was started on a private loop instead of the loop that owned
the objects it drives, and two coping hacks grew around the resulting
breakage — a lock that silently rebound per loop (destroying the mutual
exclusion it existed to provide) and first-toucher-wins lane creation.

The failure these tests pin down is not a crash. Cross-loop use of an
``asyncio.Queue`` accepts the item, never wakes the worker parked on the
owning loop, and never resolves the caller's future. Nothing raises on
either side. It presents as an agent that silently stops responding.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time

import pytest

from src.host.lanes import LaneManager


def _loop_unavailable_exc() -> type[BaseException]:
    """Look the exception up through the module, not a module-level import.

    ``tests/test_lanes.py`` calls ``importlib.reload`` on ``src.host.lanes``,
    which rebinds every class in the module's globals. The decorator resolves
    ``LaneLoopUnavailable`` by name at RAISE time, so after a reload it raises
    the new class while an ``from ... import`` binding here still refers to
    the old one — and ``pytest.raises`` stops matching, depending purely on
    test order.
    """
    import src.host.lanes as lanes_mod

    return lanes_mod.LaneLoopUnavailable


def _spin_loop() -> asyncio.AbstractEventLoop:
    """Start a loop in a daemon thread and return it, running."""
    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=_run, daemon=True).start()
    for _ in range(200):
        if loop.is_running():
            return loop
        time.sleep(0.01)
    raise RuntimeError("loop did not start")


def _stop_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel everything still pending, then stop and close the loop.

    Some of these tests deliberately leave a caller hung, so the teardown
    has to reap those tasks or pytest reports them as unraisable errors.
    """

    async def _cancel_all() -> None:
        me = asyncio.current_task()
        for task in asyncio.all_tasks(loop):
            if task is not me:
                task.cancel()

    if loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(_cancel_all(), loop).result(timeout=5)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        # Generous: on a loaded machine the loop's thread may not be
        # scheduled for a while, and a test that proceeds against a loop it
        # only ASSUMED was stopped fails confusingly and intermittently.
        deadline = time.time() + 10
        while loop.is_running() and time.time() < deadline:
            time.sleep(0.01)
        assert not loop.is_running(), "loop did not stop within 10s"
    if not loop.is_closed():
        loop.close()


@pytest.fixture
def two_loops():
    """A pair of running loops standing in for the dispatch and mesh loops."""
    dispatch, mesh = _spin_loop(), _spin_loop()
    yield dispatch, mesh
    _stop_loop(dispatch)
    _stop_loop(mesh)


async def _call(fn, *args):
    """Run a sync callable inside a coroutine, so it executes ON that loop."""
    return fn(*args)


def _lane(**kw) -> LaneManager:
    async def _dispatch(agent: str, message: str, **_kw) -> str:
        await asyncio.sleep(0.05)
        return f"reply:{message}"

    return LaneManager(_dispatch, queue_maxsize=10, **kw)


class TestLaneOwnerLoop:
    """A bound lane serves callers on any loop; an unbound one does not."""

    def test_cross_loop_caller_is_served_when_bound(self, two_loops):
        # The regression this whole change exists for. Before binding, this
        # exact sequence hangs forever with no exception on either loop.
        dispatch, mesh = two_loops
        lm = _lane()
        lm.bind_loop(dispatch)

        # Lane is created by a dispatch-loop caller (the CLI/cron path)...
        first = asyncio.run_coroutine_threadsafe(lm.enqueue("bob", "one"), dispatch)
        assert first.result(timeout=10) == "reply:one"

        # ...and then reached by a mesh-loop caller (a route handler), which
        # is what server.py and dashboard/server.py do directly.
        second = asyncio.run_coroutine_threadsafe(lm.enqueue("bob", "two"), mesh)
        assert second.result(timeout=10) == "reply:two"

    def test_lane_is_created_on_the_owner_loop_not_the_caller(self, two_loops):
        # First contact comes from the MESH loop, but the queue/lock/worker
        # must still be built on the dispatch loop that owns them.
        dispatch, mesh = two_loops
        lm = _lane()
        lm.bind_loop(dispatch)

        fut = asyncio.run_coroutine_threadsafe(lm.enqueue("carol", "hi"), mesh)
        assert fut.result(timeout=10) == "reply:hi"
        assert lm._first_lane_loop is dispatch
        assert lm._workers["carol"].get_loop() is dispatch

    def test_owner_loop_caller_does_not_hop(self, two_loops):
        # Same-loop calls must not pay a round trip, and must not deadlock.
        dispatch, _mesh = two_loops
        lm = _lane()
        lm.bind_loop(dispatch)

        async def _both() -> list[str]:
            return [await lm.enqueue("dave", "a"), await lm.enqueue("dave", "b")]

        fut = asyncio.run_coroutine_threadsafe(_both(), dispatch)
        assert fut.result(timeout=10) == ["reply:a", "reply:b"]

    def test_stopped_owner_loop_raises_instead_of_hanging(self, two_loops):
        dispatch, mesh = two_loops
        lm = _lane()
        lm.bind_loop(dispatch)
        asyncio.run_coroutine_threadsafe(lm.enqueue("erin", "one"), dispatch).result(timeout=10)
        _stop_loop(dispatch)

        fut = asyncio.run_coroutine_threadsafe(lm.enqueue("erin", "two"), mesh)
        with pytest.raises(_loop_unavailable_exc()):
            fut.result(timeout=10)

    def test_never_started_owner_loop_raises_immediately(self):
        """The same guard, with no shutdown race in the test at all.

        The sibling above stops a live loop, which on a loaded machine is a
        timing exercise. Here the owner is a loop that was never started, so
        ``is_running()`` is False by construction.
        """
        idle_owner = asyncio.new_event_loop()
        try:
            lm = _lane()
            lm.bind_loop(idle_owner)

            async def _go():
                await lm.enqueue("nina", "hi")

            with pytest.raises(_loop_unavailable_exc()):
                asyncio.run(_go())
        finally:
            idle_owner.close()

    def test_unbound_lane_runs_on_the_calling_loop(self):
        # Backwards compatibility: with no owner bound, behaviour is exactly
        # what it was — every test and embedded construction relies on this.
        lm = _lane()
        assert lm.owner_loop is None

        async def _go() -> str:
            reply = await lm.enqueue("frank", "solo")
            assert lm._first_lane_loop is asyncio.get_running_loop()
            return reply

        assert asyncio.run(_go()) == "reply:solo"

    def test_unbound_cross_loop_use_warns(self, two_loops, caplog):
        # It still hangs (that is the pre-existing behaviour we are not
        # changing out from under embedded callers) — but it now says so.
        dispatch, mesh = two_loops
        lm = _lane()  # deliberately NOT bound

        asyncio.run_coroutine_threadsafe(lm.enqueue("gina", "one"), dispatch).result(timeout=10)
        with caplog.at_level("WARNING", logger="host.lanes"):
            hung = asyncio.run_coroutine_threadsafe(lm.enqueue("gina", "two"), mesh)
            with pytest.raises(TimeoutError):
                hung.result(timeout=2)
            hung.cancel()

        assert any("hang forever" in r.message for r in caplog.records), caplog.text

    def test_backpressure_survives_the_hop(self, two_loops):
        """A full lane must still raise LaneQueueFull at the CALLER.

        The hop bridges a concurrent future back onto the calling loop, so
        exceptions have to cross with it — mesh routes turn LaneQueueFull
        into an HTTP 429, and swallowing it would turn backpressure into
        silently dropped work.
        """
        from src.host.lanes import LaneQueueFull

        dispatch, mesh = two_loops

        released = threading.Event()

        async def _blocking_dispatch(agent: str, message: str, **_kw) -> str:
            await asyncio.get_running_loop().run_in_executor(None, released.wait)
            return "done"

        lm = LaneManager(_blocking_dispatch, queue_maxsize=1)
        lm.bind_loop(dispatch)
        try:
            # One turn in flight (occupies the worker), one queued (fills
            # the depth-1 queue), so the third has nowhere to go.
            asyncio.run_coroutine_threadsafe(lm.enqueue("hank", "a"), mesh)
            time.sleep(0.3)
            asyncio.run_coroutine_threadsafe(lm.enqueue("hank", "b"), mesh)
            time.sleep(0.3)
            overflow = asyncio.run_coroutine_threadsafe(lm.enqueue("hank", "c"), mesh)
            with pytest.raises(LaneQueueFull):
                overflow.result(timeout=10)
        finally:
            released.set()

    def test_caller_context_reaches_the_hopped_coroutine(self, two_loops):
        """The hop must not lose the caller's contextvars.

        ``run_coroutine_threadsafe`` schedules through
        ``call_soon_threadsafe``, which copies the SUBMITTING context — so
        the enqueue body runs with the caller's contextvars, not a fresh
        set. Mesh code depends on this shape (server.py reads
        ``current_trace_id`` at bind time before enqueuing), and a hop
        reimplemented with a plain thread pool would silently drop it.
        """
        import contextvars

        probe: contextvars.ContextVar[str] = contextvars.ContextVar("probe", default="UNSET")
        dispatch, mesh = two_loops
        seen: dict[str, str] = {}

        class _Recording(LaneManager):
            async def _handle_followup(self, agent, message, **kw):
                seen["ctx"] = probe.get()
                return "ok"

        async def _noop(agent, message, **_kw):
            return "ok"

        lm = _Recording(_noop, queue_maxsize=10)
        lm.bind_loop(dispatch)

        async def _caller() -> str:
            probe.set("SET-BY-CALLER")
            return await lm.enqueue("jane", "hi")

        assert asyncio.run_coroutine_threadsafe(_caller(), mesh).result(timeout=10) == "ok"
        assert seen["ctx"] == "SET-BY-CALLER"

    def test_worker_trace_id_comes_from_the_argument(self, two_loops):
        # Trace identity rides the QueuedTask, so it is unaffected by which
        # loop the caller was on or what its ambient trace happened to be.
        from src.shared.trace import current_trace_id

        dispatch, mesh = two_loops
        seen: dict[str, str | None] = {}

        async def _capture(agent: str, message: str, **_kw) -> str:
            seen["trace"] = current_trace_id.get()
            return "ok"

        lm = LaneManager(_capture, queue_maxsize=10)
        lm.bind_loop(dispatch)

        async def _caller() -> str:
            current_trace_id.set("caller-ambient-trace")
            return await lm.enqueue("ivy", "hi", trace_id="explicit-trace")

        assert asyncio.run_coroutine_threadsafe(_caller(), mesh).result(timeout=10) == "ok"
        assert seen["trace"] == "explicit-trace"

    def test_every_public_async_entry_point_hops(self):
        """Coverage guard, derived from the class rather than a hand list.

        A new public coroutine on LaneManager that forgets the decorator is
        a new silent-hang path, so the test fails until it is decorated (or
        deliberately excluded here with a reason).
        """
        undecorated = [
            name
            for name, fn in inspect.getmembers(LaneManager, inspect.iscoroutinefunction)
            if not name.startswith("_") and not hasattr(fn, "__wrapped__")
        ]
        assert undecorated == [], (
            f"public async LaneManager methods missing @_on_owner_loop: {undecorated}"
        )


class TestCallerCancellationDoesNotWedgeTheLane:
    """A caller going away must not take the agent's lane with it.

    Cancelling a caller propagates into the per-item future the worker
    resolves. ``Future.set_result`` then raises ``InvalidStateError``, and
    the worker's trace-record path calls ``Future.exception()``, which
    RAISES ``CancelledError`` on a cancelled future rather than returning
    it. Both escape ``_worker``, killing it while it is still registered in
    ``_workers`` — so every later enqueue for that agent is accepted and
    never drained. That is the same silent hang this module's loop work
    exists to remove, arrived at from the other direction.

    Reachable on one loop (a cancelled HTTP request awaiting an enqueue) and
    across loops. Both are covered.

    Note the trace store: the ``exception()`` read is behind
    ``if task.trace_id and self._trace_store``, so a lane built without one
    never reaches the second half of the chain and the bug looks fixed when
    it is not.
    """

    class _TraceStore:
        def record(self, **kwargs) -> None:
            pass

    def _lane_with_tracing(self, hold: float = 1.0) -> LaneManager:
        async def _slow(agent: str, message: str, **_kw) -> str:
            await asyncio.sleep(hold)
            return f"reply:{message}"

        return LaneManager(_slow, queue_maxsize=10, trace_store=self._TraceStore())

    def test_same_loop_cancellation_leaves_the_lane_usable(self):
        lm = self._lane_with_tracing()

        async def _go() -> None:
            first = asyncio.create_task(lm.enqueue("alice", "first", trace_id="t1"))
            await asyncio.sleep(0.3)      # turn in flight
            first.cancel()
            await asyncio.sleep(1.2)      # let the turn finish under it

            assert not lm._workers["alice"].done(), "cancelling a caller killed the worker"
            second = await asyncio.wait_for(
                lm.enqueue("alice", "second", trace_id="t2"), timeout=5,
            )
            assert second == "reply:second"

        asyncio.run(_go())

    def test_cross_loop_cancellation_leaves_the_lane_usable(self, two_loops):
        dispatch, mesh = two_loops
        lm = self._lane_with_tracing()
        lm.bind_loop(dispatch)

        holder: dict[str, asyncio.Task] = {}

        async def _start() -> None:
            holder["task"] = asyncio.create_task(lm.enqueue("bob", "first", trace_id="t1"))

        asyncio.run_coroutine_threadsafe(_start(), mesh).result(timeout=5)
        time.sleep(0.3)
        mesh.call_soon_threadsafe(holder["task"].cancel)
        time.sleep(1.2)

        assert not lm._workers["bob"].done(), "cancelling a caller killed the worker"
        second = asyncio.run_coroutine_threadsafe(
            lm.enqueue("bob", "second", trace_id="t2"), mesh,
        )
        assert second.result(timeout=10) == "reply:second"


class TestLaneTeardownAcrossThreads:
    """remove_lane() is called from three foreign threads."""

    def test_remove_lane_actually_cancels_the_worker(self, two_loops):
        """A bare Task.cancel() does not wake the owning loop.

        ``remove_lane`` runs on the mesh loop (server.py:1493,
        dashboard/server.py:2829) and on the REPL thread (repl.py:1558),
        while the worker lives on the dispatch loop. On a busy loop the
        cancel lands by luck; on an idle one it sits unprocessed and the
        worker outlives the lane. This asserts it lands on an IDLE loop,
        which is the case that used to fail.
        """
        dispatch, mesh = two_loops
        lm = _lane()
        lm.bind_loop(dispatch)

        asyncio.run_coroutine_threadsafe(lm.enqueue("kim", "hi"), mesh).result(timeout=10)
        worker = lm._workers["kim"]
        assert not worker.done()

        # From the mesh loop — foreign to the worker's dispatch loop, which
        # now has nothing else to do.
        asyncio.run_coroutine_threadsafe(_call(lm.remove_lane, "kim"), mesh).result(timeout=5)

        deadline = time.time() + 5
        while not worker.done() and time.time() < deadline:
            time.sleep(0.01)
        assert worker.done(), "worker survived remove_lane on an idle owner loop"

    def test_remove_lane_cancels_from_a_plain_thread(self, two_loops):
        # The REPL path: no event loop under the caller at all.
        dispatch, mesh = two_loops
        lm = _lane()
        lm.bind_loop(dispatch)
        asyncio.run_coroutine_threadsafe(lm.enqueue("liam", "hi"), mesh).result(timeout=10)
        worker = lm._workers["liam"]

        done = threading.Event()

        def _from_repl_thread() -> None:
            lm.remove_lane("liam")
            done.set()

        threading.Thread(target=_from_repl_thread, daemon=True).start()
        assert done.wait(timeout=5)

        deadline = time.time() + 5
        while not worker.done() and time.time() < deadline:
            time.sleep(0.01)
        assert worker.done(), "worker survived remove_lane from a plain thread"


class TestObservabilityCannotKillTheLane:
    """A failing trace write must not take the agent down with it."""

    class _BrokenTraceStore:
        def __init__(self, fail_on: str) -> None:
            self._fail_on = fail_on

        def record(self, **kwargs) -> None:
            if kwargs.get("event_type") == self._fail_on:
                raise RuntimeError("database is locked")

    def _run_two_turns(self, fail_on: str) -> str:
        async def _dispatch(agent: str, message: str, **_kw) -> str:
            return f"reply:{message}"

        lm = LaneManager(
            _dispatch, queue_maxsize=10, trace_store=self._BrokenTraceStore(fail_on),
        )

        async def _go() -> str:
            await asyncio.wait_for(lm.enqueue("alice", "first", trace_id="t1"), timeout=5)
            assert not lm._workers["alice"].done(), (
                f"a failing {fail_on} trace write killed the worker"
            )
            return await asyncio.wait_for(
                lm.enqueue("alice", "second", trace_id="t2"), timeout=5,
            )

        return asyncio.run(_go())

    def test_lane_start_trace_failure_does_not_wedge_the_lane(self):
        # This one is worse than it looks: the raise happens BEFORE the
        # dispatch, so the dequeued task never completes and its caller
        # hangs too.
        assert self._run_two_turns("lane_start") == "reply:second"

    def test_lane_complete_trace_failure_does_not_wedge_the_lane(self):
        # This one sits in the ``finally``, ahead of the busy/pending/
        # task_done cleanup — a raise skips all of it AND kills the worker.
        assert self._run_two_turns("lane_complete") == "reply:second"


class TestStopAgentIsRaceTolerant:
    """stop_agent runs off-loop now, concurrently with wake and shutdown."""

    def _manager(self):
        from src.host.runtime import DockerBackend

        mgr = DockerBackend.__new__(DockerBackend)
        mgr.agents = {}
        mgr.auth_tokens = {}
        return mgr

    def test_a_concurrent_wake_keeps_its_container(self):
        # The agent is stopped off-loop while a cold wake recreates it. The
        # stop must not delete the NEW registration or its auth token.
        mgr = self._manager()
        new_entry = {"container": object(), "generation": "new"}

        class _OldContainer:
            def stop(self, timeout: int = 10) -> None:
                # The wake lands while we are inside the blocking stop.
                mgr.agents["alice"] = new_entry
                mgr.auth_tokens["alice"] = "new-token"

            def remove(self) -> None:
                pass

        mgr.agents["alice"] = {"container": _OldContainer(), "generation": "old"}
        mgr.auth_tokens["alice"] = "old-token"

        mgr.stop_agent("alice")

        assert mgr.agents.get("alice") is new_entry, "the wake's registration was dropped"
        assert mgr.auth_tokens.get("alice") == "new-token", "the wake's auth token was dropped"

    def test_double_stop_does_not_raise(self):
        # shutdown()'s stop_all can race a hibernation stop still in flight.
        # This used to raise KeyError out of stop_agent — outside its own
        # try — aborting teardown partway through.
        mgr = self._manager()

        class _Container:
            def stop(self, timeout: int = 10) -> None:
                mgr.agents.pop("bob", None)   # the other stop got here first

            def remove(self) -> None:
                pass

        mgr.agents["bob"] = {"container": _Container()}
        mgr.stop_agent("bob")            # must not raise
        mgr.stop_agent("bob")            # nor must a plain repeat
        assert "bob" not in mgr.agents


class TestHealthMonitorAgentLock:
    """The rebinding lock is gone, and the plain one still does its job.

    ``_get_agent_lock`` guards ``_cleanup_ephemeral_agents``, whose only
    production caller is the health sweep's own ``_check_all`` tick — so
    the lock exists to stop two overlapping TICKS from removing the same
    ephemeral agent twice. It never needed to span loops. The version it
    replaces re-created itself whenever the running loop differed, which
    handed each loop its own lock and let both sides into the critical
    section at once.
    """

    def _monitor(self):
        from src.host.health import HealthMonitor

        hm = HealthMonitor.__new__(HealthMonitor)
        hm._agent_lock = asyncio.Lock()
        return hm

    def test_lock_identity_is_stable_across_loops(self, two_loops):
        # The heart of it: one monitor, one lock, whoever asks. The old
        # implementation returned a DIFFERENT object per loop, which is
        # why mutual exclusion quietly stopped happening.
        dispatch, mesh = two_loops
        hm = self._monitor()

        async def _get():
            return hm._get_agent_lock()

        first = asyncio.run_coroutine_threadsafe(_get(), dispatch).result(timeout=5)
        second = asyncio.run_coroutine_threadsafe(_get(), mesh).result(timeout=5)
        assert first is second

    def test_mutual_exclusion_holds_for_overlapping_ticks(self):
        # Two health ticks racing on the health loop — the real scenario.
        hm = self._monitor()
        state = {"inside": 0, "overlaps": 0}

        async def _critical() -> None:
            async with hm._get_agent_lock():
                state["inside"] += 1
                if state["inside"] > 1:
                    state["overlaps"] += 1
                await asyncio.sleep(0.05)
                state["inside"] -= 1

        async def _race() -> None:
            await asyncio.gather(*(_critical() for _ in range(5)))

        asyncio.run(_race())
        assert state["overlaps"] == 0

    def test_lock_survives_successive_loops(self):
        # A long-lived monitor used across several short-lived loops (what
        # tests/test_health.py does) must keep working without the rebind
        # hack — an uncontended acquire never binds the lock to a loop.
        hm = self._monitor()

        async def _use() -> str:
            async with hm._get_agent_lock():
                return "ok"

        assert [asyncio.run(_use()) for _ in range(3)] == ["ok"] * 3

    def test_rebinding_hack_is_gone(self):
        from src.host.health import HealthMonitor

        assert not hasattr(HealthMonitor, "_agent_lock_loop")
        src = inspect.getsource(HealthMonitor._get_agent_lock)
        assert "new" not in src.lower().replace("never", "")
        assert "get_running_loop" not in src, (
            "_get_agent_lock is loop-aware again — it must return one stable lock"
        )


class TestNoPrivateSweepLoops:
    """The four private sweep loops stay deleted."""

    def _runtime_source(self) -> str:
        import src.cli.runtime as runtime

        return inspect.getsource(runtime)

    def test_start_background_creates_no_event_loop(self):
        import src.cli.runtime as runtime

        src = inspect.getsource(runtime.RuntimeContext._start_background)
        assert "new_event_loop" not in src, (
            "a periodic sweep is standing up its own loop again — sweeps run "
            "on the mesh loop via _start_sweep"
        )

    def test_only_the_dispatch_loop_is_created(self):
        # One deliberate loop creation in the whole host runtime: the
        # dispatch loop that owns the lane manager. Uvicorn owns the other.
        # Parsed, not grepped — prose about new_event_loop is not a call.
        import ast

        tree = ast.parse(self._runtime_source())
        calls = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "new_event_loop"
        ]
        assert len(calls) == 1, (
            f"expected exactly one event loop created in the host runtime, "
            f"found {len(calls)} (source lines {calls})"
        )

    def test_every_sweep_goes_through_start_sweep(self):
        import src.cli.runtime as runtime

        src = inspect.getsource(runtime.RuntimeContext)
        for sweep in ("cron", "health", "hibernation", "chain_watcher"):
            assert f'_start_sweep("{sweep}"' in src, f"{sweep} sweep is not on the mesh loop"


class TestLaneBoundAtBootstrap:
    """The production wiring that makes all of the above load-bearing."""

    def test_setup_dispatch_binds_the_lane_manager(self):
        import src.cli.runtime as runtime

        src = inspect.getsource(runtime.RuntimeContext._setup_dispatch)
        assert "bind_loop(self._dispatch_loop)" in src, (
            "the lane manager is no longer bound to the dispatch loop at "
            "bootstrap — mesh-loop routes will corrupt lanes again"
        )

    def test_mesh_loop_is_captured_at_startup(self):
        import src.cli.runtime as runtime

        src = inspect.getsource(runtime.RuntimeContext._start_mesh_server)
        assert "_capture_mesh_loop" in src
        assert "self._mesh_loop = asyncio.get_running_loop()" in src
