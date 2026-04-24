"""Tests for the profile-schema migration framework (Phase 1.4).

Exercises the six guarantees the :mod:`src.browser.profile_schema` module
promises:

1. First-time stamping (no marker) → profile ends up at current version.
2. Already-current profile → no-op, cookies/state untouched.
3. Empty migration registry between current and target → marker-only bump.
4. Registered migration runs once in order, output preserved.
5. Failing migration → backup restored, exception re-raised.
6. Orphan ``*.tmp`` from a prior crash → cleaned at entry.

Plus two concurrency / robustness tests:

7. Concurrent migrate_profile calls — non-blocking fcntl.flock acts as
   a mutual-exclusion gate; the late caller gets the on-disk version.
8. Malformed ``.ol_schema`` → treated as version 0 (not a crash).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from src.browser import profile_schema
from src.browser.profile_schema import (
    _MARKER_FILENAME,
    PROFILE_SCHEMA_VERSION,
    migrate_profile,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def profile(tmp_path: Path) -> Path:
    """Empty profile directory. Simulates a fresh agent."""
    p = tmp_path / "profile-a"
    p.mkdir()
    return p


@pytest.fixture
def populated_profile(tmp_path: Path) -> Path:
    """Profile dir with a representative set of files — stand-ins for
    Firefox's cookies / storage / preferences — used to verify migrations
    don't destroy user state."""
    p = tmp_path / "profile-b"
    p.mkdir()
    (p / "cookies.sqlite").write_bytes(b"cookies-fake")
    (p / "prefs.js").write_text('user_pref("browser.foo", true);\n')
    storage = p / "storage" / "default"
    storage.mkdir(parents=True)
    (storage / "idb.sqlite").write_bytes(b"idb-fake")
    return p


@pytest.fixture(autouse=True)
def _reset_migration_registry():
    """Ensure tests don't leak registry entries across each other."""
    before = dict(profile_schema._MIGRATIONS)
    yield
    profile_schema._MIGRATIONS.clear()
    profile_schema._MIGRATIONS.update(before)


# ── Core behavior ──────────────────────────────────────────────────────────


class TestFirstTimeStamp:
    def test_no_marker_creates_marker_at_current_version(self, profile):
        result = migrate_profile(profile)
        assert result == PROFILE_SCHEMA_VERSION
        marker = profile / _MARKER_FILENAME
        assert marker.exists()
        assert int(marker.read_text().strip()) == PROFILE_SCHEMA_VERSION

    def test_first_time_does_not_write_backup(self, profile):
        migrate_profile(profile)
        # Empty registry → marker-only bump, no snapshot/backup step.
        backup = profile.with_name(profile.name + ".bak")
        assert not backup.exists()


class TestIdempotence:
    def test_running_twice_is_noop(self, populated_profile):
        migrate_profile(populated_profile)
        first_marker = (populated_profile / _MARKER_FILENAME).read_text()
        first_cookie = (populated_profile / "cookies.sqlite").read_bytes()

        migrate_profile(populated_profile)
        assert (populated_profile / _MARKER_FILENAME).read_text() == first_marker
        assert (populated_profile / "cookies.sqlite").read_bytes() == first_cookie


class TestMigrationRegistryExecution:
    def test_registered_migration_runs_and_marker_advances(
        self, populated_profile, monkeypatch,
    ):
        """A migration registered for the target version actually runs."""
        ran = {"called": False}

        def _mig_v1(p: Path) -> None:
            ran["called"] = True
            (p / "migration_artifact.txt").write_text("ran")

        # Pretend the module is at version 1 and register our test migration.
        monkeypatch.setattr(profile_schema, "PROFILE_SCHEMA_VERSION", 1)
        profile_schema._MIGRATIONS[1] = _mig_v1

        # Seed as version 0 (no marker) so _mig_v1 gets applied.
        result = migrate_profile(populated_profile)
        assert result == 1
        assert ran["called"]
        assert (populated_profile / "migration_artifact.txt").read_text() == "ran"
        assert (populated_profile / _MARKER_FILENAME).read_text().strip() == "1"

    def test_multiple_migrations_run_in_version_order(
        self, profile, monkeypatch,
    ):
        order: list[int] = []

        def _mig_v2(p: Path) -> None:
            order.append(2)
            (p / "v2").write_text("2")

        def _mig_v3(p: Path) -> None:
            order.append(3)
            (p / "v3").write_text("3")

        monkeypatch.setattr(profile_schema, "PROFILE_SCHEMA_VERSION", 3)
        profile_schema._MIGRATIONS[3] = _mig_v3
        profile_schema._MIGRATIONS[2] = _mig_v2

        migrate_profile(profile)
        assert order == [2, 3]
        assert (profile / _MARKER_FILENAME).read_text().strip() == "3"


class TestFailureRestoresBackup:
    def test_migration_exception_restores_original_state(
        self, populated_profile, monkeypatch,
    ):
        """When a migration raises, the profile state is restored from backup
        and the exception propagates."""

        def _broken(p: Path) -> None:
            # Mutate the profile THEN raise — simulates partial work.
            (p / "migration_artifact.txt").write_text("half-done")
            (p / "cookies.sqlite").write_bytes(b"corrupted")
            raise RuntimeError("boom")

        monkeypatch.setattr(profile_schema, "PROFILE_SCHEMA_VERSION", 1)
        profile_schema._MIGRATIONS[1] = _broken

        with pytest.raises(RuntimeError, match="boom"):
            migrate_profile(populated_profile)

        # Cookies restored
        assert (
            populated_profile / "cookies.sqlite"
        ).read_bytes() == b"cookies-fake"
        # Partial artifact rolled back
        assert not (populated_profile / "migration_artifact.txt").exists()
        # Marker stays at pre-migration version (0)
        assert not (populated_profile / _MARKER_FILENAME).exists()

    def test_backup_removed_after_successful_migration(
        self, profile, monkeypatch,
    ):
        def _mig(p: Path) -> None:
            (p / "art").write_text("x")

        monkeypatch.setattr(profile_schema, "PROFILE_SCHEMA_VERSION", 1)
        profile_schema._MIGRATIONS[1] = _mig

        migrate_profile(profile)
        backup = profile.with_name(profile.name + ".bak")
        assert not backup.exists()


# ── Crash-recovery behaviors ────────────────────────────────────────────────


class TestOrphanTmpCleanup:
    def test_orphan_tmp_file_removed(self, profile):
        orphan = profile / f"{_MARKER_FILENAME}.tmp"
        orphan.write_text("99")
        migrate_profile(profile)
        assert not orphan.exists()

    def test_orphan_tmp_cleanup_does_not_recurse(self, profile):
        """Firefox writes .tmp files inside subdirs as normal operation —
        we must not touch those."""
        nested = profile / "storage" / "default"
        nested.mkdir(parents=True)
        live_tmp = nested / "live.tmp"
        live_tmp.write_text("live firefox state")
        migrate_profile(profile)
        assert live_tmp.exists()
        assert live_tmp.read_text() == "live firefox state"


class TestMalformedMarker:
    def test_non_integer_marker_treated_as_version_0(
        self, populated_profile, monkeypatch,
    ):
        (populated_profile / _MARKER_FILENAME).write_text("not-a-number")

        called = {"ran": False}

        def _mig(p: Path) -> None:
            called["ran"] = True

        monkeypatch.setattr(profile_schema, "PROFILE_SCHEMA_VERSION", 1)
        profile_schema._MIGRATIONS[1] = _mig

        migrate_profile(populated_profile)
        assert called["ran"]  # treated as pre-migration


# ── Concurrency ────────────────────────────────────────────────────────────


class TestConcurrencyLock:
    def test_lock_held_with_pending_migration_raises_busy(
        self, profile, monkeypatch,
    ):
        """A peer process mid-migration holds the flock AND the on-disk
        version is below target. Continuing to launch Camoufox now would
        race the peer's writes into user_data_dir. The caller must get a
        distinct, retryable error — ProfileMigrationBusy — not a silent
        skip (silent skip regresses the whole point of the pre-launch
        migration hook)."""
        import fcntl

        from src.browser.profile_schema import (
            _LOCK_FILENAME,
            ProfileMigrationBusy,
        )

        run_count = {"n": 0}

        def _mig(p: Path) -> None:
            run_count["n"] += 1

        monkeypatch.setattr(profile_schema, "PROFILE_SCHEMA_VERSION", 1)
        profile_schema._MIGRATIONS[1] = _mig

        # Simulate another process holding the lock by opening the file
        # and taking an exclusive flock outside of the function's try.
        lock_path = profile / _LOCK_FILENAME
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(ProfileMigrationBusy):
                migrate_profile(profile)
            assert run_count["n"] == 0
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_lock_held_but_already_current_allows_launch(
        self, profile, monkeypatch,
    ):
        """Conversely: if the on-disk version is ALREADY target, the peer
        is doing a pointless re-check. Don't raise — no migration is
        pending, safe to launch against the profile."""
        import fcntl

        from src.browser.profile_schema import _LOCK_FILENAME

        # Stamp the profile at version 1 first (no peer holding the lock).
        monkeypatch.setattr(profile_schema, "PROFILE_SCHEMA_VERSION", 1)
        migrate_profile(profile)

        # Now simulate a peer holding the lock and call again.
        lock_path = profile / _LOCK_FILENAME
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = migrate_profile(profile)
            assert result == 1  # Already current; no raise, no work.
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


# ── Integration w/ service.py ──────────────────────────────────────────────


class TestServiceIntegration:
    def test_service_imports_migrate_profile(self):
        """Regression guard: service.py binds :func:`migrate_profile` at the
        top of the module so it's called pre-launch in ``get_or_start``.

        Kept as a lightweight symbol-presence check rather than a full
        lifecycle test — Camoufox isn't available in the unit-test env
        and mocking `AsyncNewBrowser` through a dynamic import is fragile.
        """
        import src.browser.service as svc
        assert hasattr(svc, "migrate_profile")
        # Must be the real function, not something shadowed mid-refactor.
        assert svc.migrate_profile is migrate_profile

    def test_service_calls_migrate_before_launch(self):
        """Regression guard via source inspection: the call to
        ``migrate_profile`` appears in service.py above the first
        ``AsyncNewBrowser`` call inside the starter, so any future
        refactor that reorders them fails this test."""
        import inspect

        import src.browser.service as svc

        source = inspect.getsource(svc)
        # Find positions of both anchors.
        migrate_pos = source.find("migrate_profile(Path(profile_dir))")
        launch_pos = source.find("await AsyncNewBrowser(")
        assert migrate_pos != -1, "migrate_profile call disappeared"
        assert launch_pos != -1, "AsyncNewBrowser call disappeared"
        assert migrate_pos < launch_pos, (
            "migrate_profile must be called before AsyncNewBrowser — "
            "Camoufox writes into the profile directory during launch and "
            "migrations must complete first."
        )


# ── Refuses missing profile ────────────────────────────────────────────────


class TestMissingProfile:
    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            migrate_profile(tmp_path / "does-not-exist")


# ── Marker atomicity ───────────────────────────────────────────────────────


class TestMarkerAtomicWrite:
    def test_write_uses_temp_and_replace(self, profile, monkeypatch):
        """Regression guard: marker writes must go through .tmp + os.replace
        so a crash never leaves a half-written marker."""
        observed = {"used_tmp": False, "used_replace": False}
        real_replace = os.replace

        def spy_replace(src, dst):
            if str(src).endswith(".tmp") and str(dst).endswith(_MARKER_FILENAME):
                observed["used_replace"] = True
            return real_replace(src, dst)

        with mock.patch("src.browser.profile_schema.os.replace", spy_replace):
            migrate_profile(profile)

        # Under normal flow there's no persistent .tmp after the replace;
        # the spy above already confirms the tmp→final replace happened.
        assert observed["used_replace"]


class TestV2FontCacheClear:
    """§6.2 migration v2: clear Firefox font caches when new fonts land
    in the container image. Must preserve cookies/storage/prefs."""

    def test_clears_startup_cache_directory(self, profile):
        from src.browser.profile_schema import _v2_clear_font_caches

        cache = profile / "startupCache"
        cache.mkdir()
        (cache / "startupCache.4.little").write_bytes(b"compiled-xul")
        (cache / "scriptCache-child-current.bin").write_bytes(b"compiled-js")

        _v2_clear_font_caches(profile)
        assert not cache.exists()

    def test_clears_top_level_cache_blobs(self, profile):
        from src.browser.profile_schema import _v2_clear_font_caches

        (profile / "fontlist.json").write_text("{}")
        (profile / "font.properties").write_text("k=v")
        (profile / "compatibility.ini").write_text("[Compatibility]\n")

        _v2_clear_font_caches(profile)
        assert not (profile / "fontlist.json").exists()
        assert not (profile / "font.properties").exists()
        assert not (profile / "compatibility.ini").exists()

    def test_preserves_cookies_and_storage(self, populated_profile):
        """Hardest invariant: we MUST NOT touch the user's session."""
        from src.browser.profile_schema import _v2_clear_font_caches

        _v2_clear_font_caches(populated_profile)
        assert (populated_profile / "cookies.sqlite").read_bytes() == b"cookies-fake"
        assert (populated_profile / "prefs.js").exists()
        assert (populated_profile / "storage" / "default" / "idb.sqlite").exists()

    def test_idempotent_on_fresh_profile(self, profile):
        """Running on a profile that has no cache files must be a no-op."""
        from src.browser.profile_schema import _v2_clear_font_caches

        _v2_clear_font_caches(profile)  # no caches to clear
        _v2_clear_font_caches(profile)  # and again — still fine

    def test_end_to_end_migration_at_v2(self, profile):
        """Fresh profile → migrate → marker is at v2, no crash on empty caches."""
        (profile / "startupCache").mkdir()
        (profile / "startupCache" / "x.bin").write_bytes(b"x")
        result = migrate_profile(profile)
        assert result == 2
        assert not (profile / "startupCache").exists()

    def test_rerun_after_v2_is_noop(self, populated_profile):
        """Already-at-v2 profile: migrate is a fast no-op and doesn't
        clear caches a second time (idempotence of the framework)."""
        (populated_profile / _MARKER_FILENAME).write_text("2\n")
        (populated_profile / "startupCache").mkdir()
        (populated_profile / "startupCache" / "survives.bin").write_bytes(b"y")

        migrate_profile(populated_profile)
        # Cache survived the no-op re-migrate.
        assert (populated_profile / "startupCache" / "survives.bin").exists()
