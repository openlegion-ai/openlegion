"""Per-agent serialisation of the container registry.

``start_agent`` and ``stop_agent`` each mutate two registries — ``agents``
and ``auth_tokens`` — around a slow container call, and they genuinely run
concurrently: hibernation stops agents on a worker thread
(``asyncio.to_thread``), cold wake starts them in an executor, the health
monitor restarts them from its own sweep, and the mesh's restart / archive /
delete routes run on the mesh loop. Before ``_agent_locked`` the two
registries could drift apart mid-flight.

Each test names the guard it pins down in its own docstring. Most fail with
that guard removed; the ones that do not are labelled NEGATIVE CONTROL — they
exist to catch the fix over-reaching (fleet-wide serialisation, self-deadlock)
or to hold a #1298 behaviour in place, so passing without the lock is correct
for them.
"""

from __future__ import annotations

import subprocess
import threading
import time
from unittest.mock import MagicMock

import pytest

from src.host.runtime import DockerBackend, SandboxBackend
from tests.test_runtime import _make_docker_backend


def _wait_until(pred, timeout: float = 2.0, interval: float = 0.005) -> bool:
    """Poll ``pred`` until true or ``timeout`` elapses. Returns the result."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def _waiters(backend, agent_id: str) -> int:
    """How many callers hold or are queued on ``agent_id``'s lock.

    Reads the refcount ``_agent_locked`` maintains. Zero when the guard is
    absent entirely, which is what a backend built via ``__new__`` looks
    like before its first lock acquisition.
    """
    entry = getattr(backend, "_agent_locks", {}).get(agent_id)
    return entry[1] if entry else 0


def _make_sandbox_backend(tmp_path) -> SandboxBackend:
    """A SandboxBackend with no Docker and no ``__init__`` — same ``__new__``
    construction the rest of tests/test_runtime.py uses."""
    backend = SandboxBackend.__new__(SandboxBackend)
    backend.project_root = tmp_path / "project"
    backend.project_root.mkdir(parents=True, exist_ok=True)
    backend.mesh_host_port = 8420
    backend.agents = {}
    backend.auth_tokens = {}
    backend.extra_env = {}
    backend._workspace_root = tmp_path / ".openlegion" / "agents"
    backend._workspace_root.mkdir(parents=True, exist_ok=True)
    return backend


def _docker_client(container=None) -> MagicMock:
    """A mock docker client whose ``containers.run`` returns ``container``
    and whose ``containers.get`` reports no stale container to reap."""
    import docker as _docker

    client = MagicMock()
    client.containers.run.return_value = container if container is not None else MagicMock()
    client.containers.get.side_effect = _docker.errors.NotFound("no stale container")
    return client


class _Thread(threading.Thread):
    """Thread that remembers whatever exception killed it."""

    def __init__(self, fn):
        super().__init__(daemon=True)
        self._fn = fn
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self._fn()
        except BaseException as e:  # noqa: BLE001 - re-raised by .check()
            self.error = e

    def check(self) -> None:
        if self.error is not None:
            raise AssertionError(f"thread died: {self.error!r}") from self.error


# ── The registries must not drift apart ───────────────────────


class TestRegistriesStayCoherent:
    def test_stop_cannot_revoke_a_concurrent_starts_token(self):
        """The headline race.

        ``start_agent`` publishes the auth token BEFORE the slow
        ``containers.run`` and only registers the entry after it. A stop that
        parks inside ``container.stop()`` and resumes inside that window sees
        the PREVIOUS entry, matches it in its compare-and-delete, and pops the
        token the new start just minted. The start then registers its entry.

        Net result without the lock: ``agents`` has the agent, ``auth_tokens``
        does not — a container that looks perfectly healthy and 401s on every
        mesh call.
        """
        backend = _make_docker_backend()
        old_container = MagicMock()
        backend.agents["alpha"] = {"container": old_container, "url": "http://old", "role": "r"}
        backend.auth_tokens["alpha"] = "OLD-TOKEN"

        stop_parked = threading.Event()
        start_in_run = threading.Event()
        stop_done = threading.Event()
        rendezvous: dict[str, bool] = {}

        def park_in_stop(*_a, **_k):
            stop_parked.set()
            # Resume the stop precisely inside the start's danger window:
            # token published, entry not yet registered. Without the lock the
            # start reaches that window (``start_in_run``); with it the start
            # is queued on the lock instead (refcount 2) and there is nothing
            # to wait for. Waiting on the token alone is NOT enough — the
            # start would race past the registration before this returns and
            # the compare-and-delete would then correctly decline to pop.
            rendezvous["met"] = _wait_until(
                lambda: start_in_run.is_set() or _waiters(backend, "alpha") >= 2,
                timeout=10,
            )

        def park_in_run(*_a, **_k):
            start_in_run.set()
            stop_done.wait(3.0)
            return MagicMock()

        old_container.stop.side_effect = park_in_stop
        backend.client = _docker_client()
        backend.client.containers.run.side_effect = park_in_run

        def do_stop():
            try:
                backend.stop_agent("alpha")
            finally:
                stop_done.set()

        stopper = _Thread(do_stop)
        stopper.start()
        assert stop_parked.wait(5), "stop_agent never reached container.stop()"

        starter = _Thread(lambda: backend.start_agent(agent_id="alpha", role="r", tools_dir=""))
        starter.start()

        stopper.join(10)
        starter.join(10)
        assert not stopper.is_alive() and not starter.is_alive()
        assert rendezvous.get("met"), (
            "the start neither reached its danger window nor queued on the lock — "
            "the interleaving under test never happened"
        )
        stopper.check()
        starter.check()

        # Whichever order the lock imposed, the two registries agree.
        assert ("alpha" in backend.agents) == ("alpha" in backend.auth_tokens), (
            f"registries drifted: agents={'alpha' in backend.agents}, auth_tokens={'alpha' in backend.auth_tokens}"
        )
        # Stop went first, so the start is the survivor and its token is live.
        assert backend.agents["alpha"]["url"] != "http://old"
        assert backend.auth_tokens["alpha"] != "OLD-TOKEN"

    def test_stop_waits_for_an_in_flight_start_of_the_same_agent(self):
        """A start holding the lock excludes a stop for the whole container
        build — including the window where only the token is published."""
        backend = _make_docker_backend()
        in_run = threading.Event()
        release = threading.Event()

        def park_in_run(*_a, **_k):
            in_run.set()
            release.wait(10)
            return MagicMock()

        backend.client = _docker_client()
        backend.client.containers.run.side_effect = park_in_run

        starter = _Thread(lambda: backend.start_agent(agent_id="beta", role="r", tools_dir=""))
        starter.start()
        assert in_run.wait(5), "start_agent never reached containers.run()"
        # The token is already published; the entry is not.
        assert "beta" in backend.auth_tokens
        assert "beta" not in backend.agents

        stopper = _Thread(lambda: backend.stop_agent("beta"))
        stopper.start()
        # Prove the contender REACHED the lock and queued on it. A bare
        # ``join(0.5); assert is_alive()`` would also pass for a thread the
        # scheduler simply never ran — which is exactly what an overloaded
        # xdist worker looks like.
        assert _wait_until(lambda: _waiters(backend, "beta") >= 2, timeout=10), (
            "stop_agent never queued on the agent lock"
        )
        stopper.join(0.5)
        assert stopper.is_alive(), "stop_agent ran while a start held the agent lock"

        release.set()
        starter.join(10)
        stopper.join(10)
        assert not starter.is_alive() and not stopper.is_alive()
        starter.check()
        stopper.check()
        # Serialised as start-then-stop: the stop tore down what the start built.
        assert "beta" not in backend.agents
        assert "beta" not in backend.auth_tokens

    def test_spawn_ttl_stamps_survive_a_concurrent_stop(self):
        """``spawn_agent`` stamps ``ephemeral``/``ttl``/``spawned_at`` onto the
        registration AFTER ``start_agent`` returns. A stop landing in that gap
        deregisters the agent and the stamp raises ``KeyError`` out of spawn —
        leaving a running container nothing owns."""
        backend = _make_docker_backend()
        registered = threading.Event()
        stop_returned = threading.Event()
        rendezvous: dict[str, bool] = {}

        def fake_start(agent_id, role, tools_dir, **_kw):
            backend.agents[agent_id] = {"container": MagicMock(), "url": "u", "role": role}
            backend.auth_tokens[agent_id] = "tok"
            registered.set()
            # Hand the stop its chance to land in the gap. Without the lock it
            # runs to completion (``stop_returned``); with it, it queues on the
            # lock spawn is holding (refcount 2). Either way the outcome is
            # RECORDED, so a rendezvous that merely timed out cannot pass as a
            # successful exclusion.
            rendezvous["met"] = _wait_until(
                lambda: stop_returned.is_set() or _waiters(backend, agent_id) >= 2,
                timeout=10,
            )
            return "u"

        backend.start_agent = fake_start

        spawner = _Thread(lambda: backend.spawn_agent(agent_id="spawn-1", role="r", ttl=60))
        spawner.start()
        assert registered.wait(5)

        def do_stop():
            backend.stop_agent("spawn-1")
            stop_returned.set()

        stopper = _Thread(do_stop)
        stopper.start()

        spawner.join(10)
        stopper.join(10)
        assert not spawner.is_alive() and not stopper.is_alive()
        assert rendezvous.get("met"), "the stop never got its chance to land in the gap"
        stopper.check()
        spawner.check()  # KeyError from the stamps lands here without the lock

    def test_spawn_agent_does_not_self_deadlock(self):
        """NEGATIVE CONTROL. ``spawn_agent`` brackets ``start_agent``, which takes the same lock —
        it must be re-entrant."""
        backend = _make_docker_backend()
        backend.client = _docker_client()

        spawner = _Thread(lambda: backend.spawn_agent(agent_id="spawn-2", role="r", ttl=90))
        spawner.start()
        spawner.join(10)
        assert not spawner.is_alive(), "spawn_agent deadlocked on the per-agent lock"
        spawner.check()
        assert backend.agents["spawn-2"]["ephemeral"] is True
        assert backend.agents["spawn-2"]["ttl"] == 90
        assert backend.agents["spawn-2"]["spawned_at"] > 0
        # The mesh spawn endpoint used to re-stamp all of this (plus ``role``)
        # onto the unlocked registry afterwards. It no longer does, so the
        # backend has to be the single writer.
        assert backend.agents["spawn-2"]["role"] == "r"


# ── The lock is per agent, and it is not a leak ───────────────


class TestLockScopeAndLifetime:
    def test_one_agents_start_does_not_block_another_agent(self):
        """NEGATIVE CONTROL. Serialising the whole fleet behind one lock would put every start
        in line behind the slowest container build."""
        backend = _make_docker_backend()
        in_run = threading.Event()
        release = threading.Event()

        def park_first(*_a, **_k):
            if not in_run.is_set():
                in_run.set()
                release.wait(10)
            return MagicMock()

        backend.client = _docker_client()
        backend.client.containers.run.side_effect = park_first

        slow = _Thread(lambda: backend.start_agent(agent_id="slow", role="r", tools_dir=""))
        slow.start()
        assert in_run.wait(5)

        other = _Thread(lambda: backend.start_agent(agent_id="other", role="r", tools_dir=""))
        other.start()
        other.join(5)
        assert not other.is_alive(), "an unrelated agent's start waited on the slow agent's lock"
        other.check()

        release.set()
        slow.join(10)
        slow.check()

    def test_lock_entry_is_reclaimed(self):
        """``spawn_agent`` mints a fresh id per call, so a lock retained per id
        would leak one object per ephemeral agent for the life of the host."""
        backend = _make_docker_backend()
        backend.client = _docker_client()

        backend.start_agent(agent_id="ephemeral-1", role="r", tools_dir="")
        backend.stop_agent("ephemeral-1")

        assert backend._agent_locks == {}, f"lock registry leaked: {backend._agent_locks}"

    def test_lock_is_held_while_reclamation_is_pending(self):
        """Reclamation is refcounted, so an entry cannot be evicted out from
        under a thread that is queued on it — which would let two callers hold
        two different locks for the same agent."""
        backend = _make_docker_backend()
        entered = threading.Event()
        release = threading.Event()

        def hold():
            with backend._agent_locked("gamma"):
                entered.set()
                release.wait(10)

        holder = _Thread(hold)
        holder.start()
        assert entered.wait(5)

        def queue_up():
            with backend._agent_locked("gamma"):
                pass

        waiter = _Thread(queue_up)
        waiter.start()
        assert _wait_until(lambda: _waiters(backend, "gamma") >= 2, timeout=5), (
            "the second caller never registered on the existing lock entry"
        )
        # Both callers are on the SAME entry — not one lock each.
        assert backend._agent_locks["gamma"][1] == 2

        release.set()
        holder.join(10)
        waiter.join(10)
        holder.check()
        waiter.check()
        # ...and once the contention clears, the entry goes away.
        assert backend._agent_locks == {}

    def test_backend_built_without_init_still_locks(self):
        """Tests (and only tests) build backends via ``__new__``, so the lock
        registry has to install itself on first use rather than assume
        ``__init__`` ran."""
        backend = DockerBackend.__new__(DockerBackend)
        backend.agents = {}
        backend.auth_tokens = {}
        assert not hasattr(backend, "_agent_locks_guard")

        with backend._agent_locked("delta"):
            assert _waiters(backend, "delta") == 1
        assert backend._agent_locks == {}


# ── Reading the registry while it is being mutated ────────────


class TestListAgentsIsSnapshotted:
    def test_list_agents_tolerates_a_concurrent_deregistration(self):
        """``list_agents`` runs on the mesh loop; ``stop_agent`` now pops from
        a worker thread. Comprehending over the live dict raises "dictionary
        changed size during iteration" out of whatever route asked for the
        roster.

        Rather than race for it, this drives the mutation deterministically:
        reading the FIRST entry's ``url`` deregisters the second.
        """
        backend = _make_docker_backend()

        class _DeregisterOnRead(dict):
            def __getitem__(self, key):
                if key == "url":
                    backend.agents.pop("victim", None)
                return super().__getitem__(key)

        backend.agents["first"] = _DeregisterOnRead(url="http://first", role="r")
        backend.agents["victim"] = {"url": "http://victim", "role": "r"}

        listed = backend.list_agents()

        assert set(listed) == {"first", "victim"}
        assert listed["victim"]["url"] == "http://victim"


# ── Sandbox backend parity ────────────────────────────────────


class TestSandboxBackendLocking:
    def test_start_excludes_a_concurrent_stop(self, tmp_path, monkeypatch):
        """``_prepare_workspace`` mints the auth token and sandbox creation can
        take 120s, so the same window exists here as in DockerBackend."""
        backend = _make_sandbox_backend(tmp_path)
        in_create = threading.Event()
        release = threading.Event()

        def fake_run(cmd, *_a, **_k):
            if "create" in cmd:
                in_create.set()
                release.wait(10)
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("src.host.runtime.subprocess.run", fake_run)

        starter = _Thread(lambda: backend.start_agent(agent_id="s1", role="r", tools_dir=""))
        starter.start()
        assert in_create.wait(10), "start_agent never reached `docker sandbox create`"
        assert "s1" in backend.auth_tokens
        assert "s1" not in backend.agents

        stopper = _Thread(lambda: backend.stop_agent("s1"))
        stopper.start()
        assert _wait_until(lambda: _waiters(backend, "s1") >= 2, timeout=10), (
            "stop_agent never queued on the agent lock"
        )
        stopper.join(0.5)
        assert stopper.is_alive(), "stop_agent ran while a start held the agent lock"

        release.set()
        starter.join(15)
        stopper.join(10)
        assert not starter.is_alive() and not stopper.is_alive()
        starter.check()
        stopper.check()
        assert "s1" not in backend.agents
        assert "s1" not in backend.auth_tokens

    def test_stop_excludes_a_concurrent_start(self, tmp_path, monkeypatch):
        backend = _make_sandbox_backend(tmp_path)
        backend.agents["s2"] = {"sandbox_name": "openlegion_s2", "workspace": None, "url": "u", "role": "r"}
        backend.auth_tokens["s2"] = "OLD-TOKEN"
        in_rm = threading.Event()
        release = threading.Event()

        def fake_run(cmd, *_a, **_k):
            if "rm" in cmd:
                in_rm.set()
                release.wait(10)
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("src.host.runtime.subprocess.run", fake_run)

        stopper = _Thread(lambda: backend.stop_agent("s2"))
        stopper.start()
        assert in_rm.wait(10), "stop_agent never reached `docker sandbox rm`"

        starter = _Thread(lambda: backend.start_agent(agent_id="s2", role="r", tools_dir=""))
        starter.start()
        assert _wait_until(lambda: _waiters(backend, "s2") >= 2, timeout=10), (
            "start_agent never queued on the agent lock"
        )
        starter.join(0.5)
        assert starter.is_alive(), "start_agent ran while a stop held the agent lock"

        release.set()
        stopper.join(10)
        starter.join(15)
        assert not stopper.is_alive() and not starter.is_alive()
        stopper.check()
        starter.check()
        assert ("s2" in backend.agents) == ("s2" in backend.auth_tokens)
        assert backend.auth_tokens["s2"] != "OLD-TOKEN"


# ── A failed start must leave the registries as it found them ─


class TestFailedStartRollsBack:
    """The lock serialises concurrent access, but it cannot fix an ordering
    that is incoherent on its own. ``start_agent`` overwrites the auth token
    BEFORE the container call; a restart of a LIVE agent that then fails used
    to drop that token and keep the old entry — ``agents`` populated,
    ``auth_tokens`` empty, with no concurrency involved at all."""

    def test_failure_BEFORE_the_reap_keeps_the_previous_token(self):
        """The start died while still looking the stale container up, so it
        never removed anything. The previous container is still running and
        needs its token back."""
        import docker as _docker

        backend = _make_docker_backend()
        backend.agents["live"] = {"container": MagicMock(), "url": "http://live", "role": "r"}
        backend.auth_tokens["live"] = "LIVE-TOKEN"
        backend.client = _docker_client()
        backend.client.containers.get.side_effect = _docker.errors.APIError("daemon hiccup")

        with pytest.raises(_docker.errors.APIError):
            backend.start_agent(agent_id="live", role="r", tools_dir="")

        # The reap never ran, so nothing was destroyed.
        assert backend.agents["live"]["url"] == "http://live"
        assert backend.auth_tokens["live"] == "LIVE-TOKEN", (
            "the still-running previous container was stranded without a token"
        )

    def test_failure_AFTER_the_reap_deregisters(self):
        """``_start_agent_container`` force-removes the same-named container
        immediately before ``containers.run``, so a failure there has already
        destroyed the previous agent. Restoring its token would re-arm a
        credential for a container that no longer exists, and keeping the
        entry would advertise a URL nothing answers."""
        backend = _make_docker_backend()
        backend.agents["gone"] = {"container": MagicMock(), "url": "http://gone", "role": "r"}
        backend.auth_tokens["gone"] = "REAPED-TOKEN"
        backend.client = _docker_client()
        # A stale container EXISTS, so the reap actually runs — the default
        # mock client raises NotFound here and would skip it entirely.
        stale = MagicMock()
        backend.client.containers.get.side_effect = None
        backend.client.containers.get.return_value = stale
        backend.client.containers.run.side_effect = RuntimeError("image missing")

        with pytest.raises(RuntimeError):
            backend.start_agent(agent_id="gone", role="r", tools_dir="")

        stale.remove.assert_called_once_with(force=True)  # the reap really ran
        assert "gone" not in backend.agents
        assert "gone" not in backend.auth_tokens, "a destroyed container's token was re-armed by the rollback"

    def test_container_that_vanished_during_the_reap_also_deregisters(self):
        """``remove()`` raising NotFound means the container disappeared
        between the lookup and the removal. It is just as gone as if we had
        removed it, so reporting "not destroyed" would restore a token for
        something that no longer exists."""
        import docker as _docker

        backend = _make_docker_backend()
        backend.agents["ghosted"] = {"container": MagicMock(), "url": "http://ghosted", "role": "r"}
        backend.auth_tokens["ghosted"] = "GHOST-TOKEN"
        backend.client = _docker_client()
        stale = MagicMock()
        stale.remove.side_effect = _docker.errors.NotFound("already gone")
        backend.client.containers.get.side_effect = None
        backend.client.containers.get.return_value = stale
        backend.client.containers.run.side_effect = RuntimeError("image missing")

        with pytest.raises(RuntimeError):
            backend.start_agent(agent_id="ghosted", role="r", tools_dir="")

        assert "ghosted" not in backend.agents
        assert "ghosted" not in backend.auth_tokens

    def test_sandbox_failure_after_create_deregisters(self, tmp_path, monkeypatch):
        """A successful ``sandbox create`` binds the name to a NEW microVM, so
        the previous registration is unreachable from that point on. Restoring
        its token would hand it to a sandbox started with a different one."""
        backend = _make_sandbox_backend(tmp_path)
        # The registered name and the one start_agent computes are both
        # ``openlegion_{_docker_safe_name(agent_id)}``, so creating this name
        # really does replace what was registered.
        backend.agents["s4"] = {"sandbox_name": "openlegion_s4", "workspace": None, "url": "u", "role": "r"}
        backend.auth_tokens["s4"] = "OLD-TOKEN"

        def fake_run(cmd, *_a, **_k):
            if "exec" in cmd:
                raise subprocess.TimeoutExpired(cmd, 60)
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("src.host.runtime.subprocess.run", fake_run)

        with pytest.raises(subprocess.TimeoutExpired):
            backend.start_agent(agent_id="s4", role="r", tools_dir="")

        assert "s4" not in backend.agents
        assert "s4" not in backend.auth_tokens

    def test_sandbox_failure_at_create_keeps_the_previous_token(self, tmp_path, monkeypatch):
        backend = _make_sandbox_backend(tmp_path)
        backend.agents["s5"] = {"sandbox_name": "openlegion_s5", "workspace": None, "url": "u", "role": "r"}
        backend.auth_tokens["s5"] = "OLD-TOKEN"

        def fake_run(cmd, *_a, **_k):
            if "create" in cmd:
                return MagicMock(returncode=1, stdout="", stderr="no sandbox support")
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("src.host.runtime.subprocess.run", fake_run)

        with pytest.raises(RuntimeError):
            backend.start_agent(agent_id="s5", role="r", tools_dir="")

        assert backend.agents["s5"]["url"] == "u"
        assert backend.auth_tokens["s5"] == "OLD-TOKEN"

    def test_sandbox_failed_create_leaves_no_orphan_token(self, tmp_path, monkeypatch):
        backend = _make_sandbox_backend(tmp_path)

        def fake_run(cmd, *_a, **_k):
            if "create" in cmd:
                return MagicMock(returncode=1, stdout="", stderr="no sandbox support")
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("src.host.runtime.subprocess.run", fake_run)

        with pytest.raises(RuntimeError):
            backend.start_agent(agent_id="s3", role="r", tools_dir="")

        assert "s3" not in backend.auth_tokens
        assert "s3" not in backend.agents

    def test_lock_is_released_after_a_failed_start(self):
        """A raise must not leave the agent permanently locked."""
        backend = _make_docker_backend()
        backend.client = _docker_client()
        backend.client.containers.run.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            backend.start_agent(agent_id="brief", role="r", tools_dir="")

        assert backend._agent_locks == {}
        # And the agent is still usable.
        backend.client.containers.run.side_effect = None
        backend.start_agent(agent_id="brief", role="r", tools_dir="")
        assert "brief" in backend.agents


# ── Regressions the lock must not undo (from the 1a work) ─────


class TestPreservedFrom1a:
    def test_double_stop_is_idempotent(self):
        """A second stop must not raise ``KeyError`` out of ``stop_all``."""
        backend = _make_docker_backend()
        backend.agents["z"] = {"container": MagicMock()}
        backend.auth_tokens["z"] = "t"
        backend.stop_agent("z")
        backend.stop_agent("z")  # must not raise

    def test_volume_wipe_still_independent_of_registration(self):
        """H12: archive already deregistered the agent, so delete must still
        wipe the volume for an agent that is not in the registry."""
        backend = _make_docker_backend()
        vol = MagicMock()
        backend.client = MagicMock()
        backend.client.volumes.get.return_value = vol

        backend.stop_agent("ghost", remove_data=True)

        vol.remove.assert_called_once_with(force=True)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
