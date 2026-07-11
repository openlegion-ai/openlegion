"""Tests for the B-pre #2 config file-lock + atomic-write posture.

Covers three things the ratified decision calls out explicitly:
  1. Concurrent-writer lost-update prevention — threads racing add-agent /
     update-field must not clobber each other (the real exposure recon
     found: every agents.yaml writer used to be a bare truncate-and-write
     with no lock at all).
  2. Atomic-write mechanics for agents.yaml — tempfile in the same dir +
     os.replace, mirroring ``_save_permissions``; a failed write must never
     leave a torn file at the real path.
  3. Cross-helper composition under ONE shared lock — a multi-step flow
     that calls another locked helper from inside an already-locked
     critical section must not deadlock (the reentrancy the shared
     ``_config_lock()`` is built for) and must not lose updates when two
     such flows run concurrently for different agents.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cli.config import (
    AGENTS_FILE,
    PERMISSIONS_FILE,
    PROJECT_ROOT,
    _add_agent_permissions,
    _add_agent_to_config,
    _add_team_blackboard_permissions,
    _config_lock,
    _load_agents_yaml,
    _load_permissions,
    _remove_agent,
    _save_agents_yaml,
    _save_permissions,
    _update_agent_field,
)


class _TempConfigMixin:
    """Mixin that redirects config files to a temp directory (mirrors the
    equivalent mixin in test_templates.py)."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_agents = AGENTS_FILE
        self._orig_perms = PERMISSIONS_FILE
        self._orig_root = PROJECT_ROOT

        import src.cli.config as cfg_mod

        self._agents_path = Path(self._tmpdir) / "config" / "agents.yaml"
        self._perms_path = Path(self._tmpdir) / "config" / "permissions.json"
        self._agents_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_mod.AGENTS_FILE = self._agents_path
        cfg_mod.PERMISSIONS_FILE = self._perms_path
        cfg_mod.PROJECT_ROOT = Path(self._tmpdir)
        self._perms_path.write_text(json.dumps({"permissions": {}}, indent=2))

    def teardown_method(self):
        import src.cli.config as cfg_mod

        cfg_mod.AGENTS_FILE = self._orig_agents
        cfg_mod.PERMISSIONS_FILE = self._orig_perms
        cfg_mod.PROJECT_ROOT = self._orig_root
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _mock_config(self, *, collab=True, agents=None):
        """Patch ``_load_config`` — needed only by helpers (e.g.
        ``_add_agent_permissions``) that read mesh.yaml/collaboration
        settings; ``CONFIG_FILE`` itself isn't redirected by this mixin."""
        return patch(
            "src.cli.config._load_config",
            return_value={
                "llm": {"default_model": "openai/gpt-4o-mini"},
                "agents": agents or {},
                "collaboration": collab,
            },
        )


class TestConcurrentWriterLostUpdate(_TempConfigMixin):
    """The lost-update race B-pre #2 closes: two threads independently
    doing load->mutate->save against the SAME file used to be able to both
    load the same baseline, and the later save would silently clobber the
    earlier one's write."""

    def test_concurrent_add_agent_all_survive(self):
        """20 threads each add a DIFFERENT agent concurrently — every
        write must land in the final agents.yaml."""
        names = [f"agent-{i}" for i in range(20)]

        def _add(name):
            _add_agent_to_config(name, f"role-{name}", "openai/gpt-4o")

        threads = [threading.Thread(target=_add, args=(n,)) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "thread did not complete — possible deadlock"

        cfg = _load_agents_yaml()
        assert set(cfg["agents"].keys()) == set(names)

    def test_concurrent_update_agent_field_all_survive(self):
        """20 threads each set a distinct field on the SAME pre-existing
        agent concurrently — all 20 field writes must be present after,
        none dropped by an interleaved load->mutate->save."""
        _add_agent_to_config("shared", "worker", "openai/gpt-4o")

        def _update(i):
            _update_agent_field("shared", f"field_{i}", i)

        threads = [threading.Thread(target=_update, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "thread did not complete — possible deadlock"

        cfg = _load_agents_yaml()
        entry = cfg["agents"]["shared"]
        for i in range(20):
            assert entry.get(f"field_{i}") == i

    def test_lock_actually_serializes_across_threads(self):
        """Direct proof the lock blocks a second thread rather than just
        bookkeeping in Python: thread A holds the lock across a sleep and
        records a timestamp just before releasing; thread B can only
        record ITS timestamp after acquiring, which must be no earlier
        than A's — impossible unless the OS-level flock actually blocked
        B's acquisition."""
        order: list[str] = []
        stamps: dict[str, float] = {}

        def _holder():
            with _config_lock():
                order.append("A-enter")
                time.sleep(0.3)
                stamps["A"] = time.monotonic()
                order.append("A-exit")

        def _waiter():
            time.sleep(0.05)  # let A acquire first
            with _config_lock():
                stamps["B"] = time.monotonic()
                order.append("B-enter")

        t1 = threading.Thread(target=_holder)
        t2 = threading.Thread(target=_waiter)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert order[0] == "A-enter"
        assert order[-1] == "B-enter"
        assert stamps["B"] >= stamps["A"]


class TestAtomicWrite(_TempConfigMixin):
    """agents.yaml must be written tempfile-then-os.replace, never a bare
    truncate-in-place — a concurrent reader must always see the old OR the
    new complete file, never a torn one."""

    def test_save_agents_yaml_uses_tempfile_and_replace(self):
        import os as _os

        calls: list[tuple[str, str]] = []
        orig_replace = _os.replace

        def _spy_replace(src, dst):
            calls.append((str(src), str(dst)))
            return orig_replace(src, dst)

        with patch("os.replace", side_effect=_spy_replace):
            _save_agents_yaml({"agents": {"x": {"role": "r", "model": "m"}}})

        assert len(calls) == 1
        src, dst = calls[0]
        assert Path(dst) == self._agents_path
        # The temp file must live in the SAME directory as the target —
        # required for os.replace to be atomic on POSIX (a cross-filesystem
        # rename is not atomic).
        assert Path(src).parent == self._agents_path.parent
        assert Path(src) != self._agents_path

    def test_no_torn_file_on_write_failure(self):
        """A failure while writing the tempfile must leave the REAL
        agents.yaml completely untouched — readers never observe a
        half-written file."""
        _save_agents_yaml({"agents": {"orig": {"role": "r", "model": "m"}}})
        original_bytes = self._agents_path.read_bytes()

        with patch("os.fdopen", side_effect=OSError("disk full (simulated)")):
            with pytest.raises(OSError):
                _save_agents_yaml({"agents": {"new": {"role": "r2", "model": "m2"}}})

        assert self._agents_path.read_bytes() == original_bytes
        cfg = _load_agents_yaml()
        assert cfg["agents"] == {"orig": {"role": "r", "model": "m"}}

    def test_torn_write_leaves_no_orphan_tempfile(self):
        """A failed write cleans up its own tempfile instead of littering
        config/ with a stray .tmp file."""
        with patch("os.fdopen", side_effect=OSError("disk full (simulated)")):
            with pytest.raises(OSError):
                _save_agents_yaml({"agents": {"x": {}}})

        leftovers = list(self._agents_path.parent.glob("*.tmp"))
        assert leftovers == []


class TestConcurrentPermissionsWriterLostUpdate(_TempConfigMixin):
    """FIX 3: the seven permissions.json endpoints (assign_agent_skills,
    set_fleet_skills, operator internet/browser toggles, api_set_fleet_skills,
    api_update_agent_permissions, api_wallet_enable_agent) now wrap their full
    ``_load_permissions -> mutate -> _save_permissions`` in ``_config_lock()``.
    These pin that permissions.json writes survive concurrency the same way
    the agents.yaml writers above do — and prove the race the lock closes.
    """

    def _locked_write(self, agent: str, skill: str) -> None:
        # The exact load->mutate->save the FIX 3 permissions endpoints perform,
        # now under the shared lock.
        with _config_lock():
            perms = _load_permissions()
            perms.setdefault("permissions", {}).setdefault(agent, {})["allowed_skills"] = [skill]
            _save_permissions(perms)

    def test_concurrent_permissions_writes_all_survive(self):
        """30 threads each write a DIFFERENT agent's permissions record under
        the shared lock — every write must land in the final permissions.json
        (2-3 of 30 survived before the lock)."""
        agents = [f"pw-{i}" for i in range(30)]
        threads = [
            threading.Thread(target=self._locked_write, args=(a, f"s-{a}"))
            for a in agents
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "thread did not complete — possible deadlock"

        perms = _load_permissions()
        for a in agents:
            assert perms["permissions"][a]["allowed_skills"] == [f"s-{a}"], (
                f"lost update for {a}"
            )

    def test_unlocked_writer_loses_updates_negative_control(self):
        """Deterministic proof the race exists WITHOUT the lock: a barrier
        forces every thread to read the same baseline before any writes, so
        the shared-file last-writer-wins drops all-but-one record. This is why
        FIX 3 wraps the writers in ``_config_lock()``."""
        n = 8
        barrier = threading.Barrier(n)

        def _unlocked(agent: str) -> None:
            perms = _load_permissions()          # all read the same empty baseline
            barrier.wait()
            perms.setdefault("permissions", {})[agent] = {"allowed_skills": [agent]}
            _save_permissions(perms)             # writes serialize; last one wins

        threads = [threading.Thread(target=_unlocked, args=(f"u-{i}",)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive()

        perms = _load_permissions()
        survivors = [f"u-{i}" for i in range(n) if f"u-{i}" in perms.get("permissions", {})]
        assert 0 < len(survivors) < n, (
            f"expected a lost-update race without the lock, got {len(survivors)}/{n} survivors"
        )


class TestConfigLockFdLeak:
    """FIX 7a: ``_ConfigLock.__enter__`` must close the fd it opened if
    ``fcntl.flock`` raises after ``os.open`` succeeds — a leak would exhaust
    the process fd table under repeated failed acquisitions."""

    def test_no_fd_leak_when_flock_raises(self):
        import os as _os

        import src.cli.config as cfg_mod

        fd_dir = "/dev/fd" if _os.path.isdir("/dev/fd") else f"/proc/{_os.getpid()}/fd"

        def _count() -> int:
            return len(_os.listdir(fd_dir))

        def _raising_flock(fd, op):
            raise OSError("flock failed (simulated)")

        with patch.object(cfg_mod.fcntl, "flock", side_effect=_raising_flock):
            baseline = _count()
            for _ in range(50):
                with pytest.raises(OSError):
                    with _config_lock():
                        pass
            after = _count()

        # Pre-fix each failed acquisition leaked its fd (~+50 here); the fix
        # closes the fd before re-raising so the count stays flat.
        assert after <= baseline + 2, f"fd leak across failed acquisitions: {baseline} -> {after}"


class TestCrossHelperUnderOneLock(_TempConfigMixin):
    """Several helpers call each other synchronously while already holding
    the lock (e.g. ``_remove_agent`` -> ``_remove_team_blackboard_permissions``).
    The shared lock is reentrant within one thread specifically so this
    doesn't deadlock; concurrent multi-step flows for DIFFERENT agents must
    still each land completely."""

    def test_remove_agent_reentrant_no_deadlock(self):
        _add_agent_to_config("teammate", "worker", "openai/gpt-4o")
        with self._mock_config():
            _add_agent_permissions("teammate")

        class _FakeTeamStore:
            def remove_agent(self, name):
                return "eng-team"

        with patch("src.cli.config._open_teams_store", return_value=_FakeTeamStore()):
            done = threading.Event()

            def _run():
                _remove_agent("teammate")
                done.set()

            t = threading.Thread(target=_run)
            t.start()
            t.join(timeout=10)
            assert done.is_set(), "_remove_agent hung — reentrant lock deadlocked"

        cfg = _load_agents_yaml()
        assert "teammate" not in cfg.get("agents", {})
        perms = _load_permissions()
        assert "teammate" not in perms.get("permissions", {})

    def test_concurrent_multi_step_permission_flows_survive(self):
        """Two threads each run a two-step flow (join a team's blackboard
        perms, then tag the agent's yaml entry) for DIFFERENT agents at the
        same time — both flows must fully land, exercising exactly the
        interleaving risk that motivated ONE shared lock across both files
        instead of two independent per-file locks."""
        _add_agent_to_config("alpha", "worker", "openai/gpt-4o")
        _add_agent_to_config("beta", "worker", "openai/gpt-4o")
        with self._mock_config():
            _add_agent_permissions("alpha")
            _add_agent_permissions("beta")

        def _flow(agent, team):
            _add_team_blackboard_permissions(agent, team)
            _update_agent_field(agent, "note", f"joined-{team}")

        t1 = threading.Thread(target=_flow, args=("alpha", "red"))
        t2 = threading.Thread(target=_flow, args=("beta", "blue"))
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        assert not t1.is_alive()
        assert not t2.is_alive()

        perms = _load_permissions()
        assert "teams/red/*" in perms["permissions"]["alpha"]["blackboard_read"]
        assert "teams/blue/*" in perms["permissions"]["beta"]["blackboard_read"]
        cfg = _load_agents_yaml()
        assert cfg["agents"]["alpha"]["note"] == "joined-red"
        assert cfg["agents"]["beta"]["note"] == "joined-blue"
