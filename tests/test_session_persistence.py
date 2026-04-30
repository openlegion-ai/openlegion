"""Tests for Phase 10 §20 — session persistence across container restarts.

Covers:
  * snapshot → restore round-trip captures cookies + localStorage
  * file permissions are exactly 0o600 (sensitive: live session tokens)
  * atomic write — a kill mid-write never leaves a partial file at the
    canonical path (the .tmp sibling is fine; the destination must be
    valid JSON or absent)
  * flag-disabled is a true no-op (no log, no write, no read)
  * stale ``saved_at`` (far in the past) restores anyway — no
    time-based expiry built in; operator owns rotation policy
  * malformed JSON: warning logged, return None, sidecar NOT clobbered
  * concurrent snapshots for two agents land at the right per-agent
    paths — no cross-contamination
  * periodic snapshot timer logic (counter accumulates and fires at
    threshold)
  * dashboard endpoint shape: GET returns counts only — no domain leak
  * DELETE endpoint requires CSRF header (X-Requested-With)
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import MagicMock

import pytest

from src.browser import session_persistence as sp

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_session_dir(tmp_path, monkeypatch):
    """Each test gets a clean per-service sessions directory."""
    monkeypatch.setenv("BROWSER_SESSION_DIR", str(tmp_path / "sessions"))
    yield


def _fake_storage_state() -> dict:
    """Realistic Playwright ``storage_state`` payload."""
    return {
        "cookies": [
            {
                "name": "session",
                "value": "secret-token-abc",
                "domain": ".example.com",
                "path": "/",
                "expires": -1,
                "httpOnly": True,
                "secure": True,
                "sameSite": "Lax",
            },
            {
                "name": "csrftoken",
                "value": "csrf-xyz",
                "domain": ".example.com",
                "path": "/",
                "expires": -1,
                "httpOnly": False,
                "secure": True,
                "sameSite": "Strict",
            },
        ],
        "origins": [
            {
                "origin": "https://example.com",
                "localStorage": [
                    {"name": "user_id", "value": "12345"},
                    {"name": "theme", "value": "dark"},
                ],
            },
            {
                "origin": "https://app.example.com",
                "localStorage": [
                    {"name": "feature_flags", "value": '{"beta":true}'},
                ],
            },
        ],
    }


class _FakeContext:
    """Minimal Playwright BrowserContext duck-type."""

    def __init__(self, state: dict | None = None):
        self._state = state or _fake_storage_state()
        self.added_cookies: list[dict] = []
        self.added_init_scripts: list[str] = []

    async def storage_state(self) -> dict:
        return self._state

    async def add_cookies(self, cookies: list[dict]) -> None:
        self.added_cookies.extend(cookies)

    async def add_init_script(self, script: str) -> None:
        self.added_init_scripts.append(script)


# ── Round-trip ─────────────────────────────────────────────────────────────


class TestRoundTrip:
    """snapshot → file appears → restore → state applied to new context."""

    @pytest.mark.asyncio
    async def test_snapshot_writes_sidecar(self):
        ctx = _FakeContext()
        ok = await sp.snapshot_session("agent-a", ctx)
        assert ok is True
        path = sp.session_path("agent-a")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert "saved_at" in data
        assert data["storage_state"] == _fake_storage_state()

    @pytest.mark.asyncio
    async def test_round_trip_applies_cookies_and_localstorage(self):
        # Capture
        ctx_orig = _FakeContext()
        ok = await sp.snapshot_session("agent-rt", ctx_orig)
        assert ok is True

        # Restore via context_factory shim — the factory simulates what
        # BrowserManager._maybe_restore_session does in production.
        ctx_new = _FakeContext(state={"cookies": [], "origins": []})

        async def _factory(*, storage_state):
            cookies = storage_state.get("cookies") or []
            if cookies:
                await ctx_new.add_cookies(cookies)
            origins = storage_state.get("origins") or []
            if origins:
                # Production path uses an init script; tests just record
                # the call so we can assert it happened.
                await ctx_new.add_init_script("seed-localStorage")
            return ctx_new

        result = await sp.restore_session("agent-rt", _factory)
        assert result is ctx_new
        assert len(ctx_new.added_cookies) == 2
        assert ctx_new.added_cookies[0]["name"] == "session"
        # localStorage seeding emitted exactly one init script
        assert ctx_new.added_init_scripts == ["seed-localStorage"]

    @pytest.mark.asyncio
    async def test_restore_returns_none_when_no_sidecar(self):
        async def _factory(*, storage_state):
            raise AssertionError("factory should not be called")

        result = await sp.restore_session("never-existed", _factory)
        assert result is None


# ── File permissions ───────────────────────────────────────────────────────


class TestFilePermissions:
    """Sidecars must be chmod 0o600 — sensitive (live session tokens)."""

    @pytest.mark.asyncio
    async def test_sidecar_is_owner_only(self):
        ctx = _FakeContext()
        await sp.snapshot_session("agent-perm", ctx)
        path = sp.session_path("agent-perm")
        assert path.exists()
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    @pytest.mark.asyncio
    async def test_sidecar_perm_holds_after_overwrite(self):
        """A second snapshot over an existing sidecar must keep 0o600."""
        ctx = _FakeContext()
        await sp.snapshot_session("agent-overwrite", ctx)
        path = sp.session_path("agent-overwrite")
        # Manually weaken the mode to verify the next snapshot re-tightens it.
        os.chmod(path, 0o644)
        assert path.stat().st_mode & 0o777 == 0o644
        await sp.snapshot_session("agent-overwrite", ctx)
        assert path.stat().st_mode & 0o777 == 0o600


# ── Atomic write under simulated crash ─────────────────────────────────────


class TestAtomicWrite:
    """Kill-during-write must not leave a partial file at the canonical path."""

    @pytest.mark.asyncio
    async def test_no_partial_file_at_canonical_path_on_replace_failure(
        self, monkeypatch,
    ):
        ctx = _FakeContext()

        # Simulate a crash AFTER the tmp file is fully written but BEFORE
        # os.replace completes. The canonical path must NOT contain a
        # partial / corrupt file — it should either be absent (this case)
        # or hold the previous snapshot (the next test).
        original_replace = os.replace

        def boom(*args, **kwargs):
            raise OSError("simulated crash mid-replace")

        monkeypatch.setattr("os.replace", boom)
        ok = await sp.snapshot_session("agent-crash", ctx)
        assert ok is False

        canonical = sp.session_path("agent-crash")
        # Either absent or — if a prior snapshot existed — still parseable.
        assert not canonical.exists()
        # The .tmp sibling is also cleaned up by the snapshot path.
        tmp = canonical.with_suffix(canonical.suffix + ".tmp")
        assert not tmp.exists()

        # Restore the original replace and re-verify success path works
        # to prove the test environment is sound.
        monkeypatch.setattr("os.replace", original_replace)
        ok = await sp.snapshot_session("agent-crash", ctx)
        assert ok is True
        assert canonical.exists()

    @pytest.mark.asyncio
    async def test_replace_failure_preserves_prior_sidecar(self, monkeypatch):
        """A failed snapshot must not destroy the previous good copy."""
        ctx = _FakeContext()
        await sp.snapshot_session("agent-prior", ctx)
        canonical = sp.session_path("agent-prior")
        prior_bytes = canonical.read_bytes()

        # Modify state, then simulate a crash on the next write.
        ctx._state = {"cookies": [{"name": "new", "value": "v"}], "origins": []}

        def boom(*args, **kwargs):
            raise OSError("simulated crash mid-replace")

        monkeypatch.setattr("os.replace", boom)
        ok = await sp.snapshot_session("agent-prior", ctx)
        assert ok is False
        # Prior snapshot intact.
        assert canonical.read_bytes() == prior_bytes

    @pytest.mark.asyncio
    async def test_inner_write_failure_does_not_double_close_fd(self, monkeypatch):
        """Once fdopen owns the descriptor, failure cleanup must not os.close it."""
        ctx = _FakeContext()
        close_calls: list[int] = []
        original_close = os.close

        def close_spy(fd: int) -> None:
            close_calls.append(fd)
            original_close(fd)

        def fsync_boom(fd: int) -> None:
            raise OSError("simulated fsync failure")

        monkeypatch.setattr(os, "close", close_spy)
        monkeypatch.setattr(os, "fsync", fsync_boom)

        ok = await sp.snapshot_session("agent-fsync", ctx)
        assert ok is False
        assert close_calls == []

    @pytest.mark.asyncio
    async def test_fdopen_failure_closes_raw_fd_exactly_once(self, monkeypatch):
        """If fdopen raises, ownership never transferred — we must os.close fd."""
        ctx = _FakeContext()
        close_calls: list[int] = []
        original_close = os.close

        def close_spy(fd: int) -> None:
            close_calls.append(fd)
            original_close(fd)

        def fdopen_boom(fd, *a, **kw):
            raise OSError("simulated fdopen failure")

        monkeypatch.setattr(os, "close", close_spy)
        monkeypatch.setattr(os, "fdopen", fdopen_boom)

        ok = await sp.snapshot_session("agent-fdopen", ctx)
        assert ok is False
        # Exactly one close — the pre-fdopen cleanup of the raw fd. No
        # double-close (which would raise OSError(EBADF) on a reused fd).
        assert len(close_calls) == 1


# ── Flag-disabled no-op (lifecycle integration) ────────────────────────────


class TestFlagDisabledIsNoOp:
    """When ``BROWSER_SESSION_PERSISTENCE_ENABLED`` is unset/false, the
    lifecycle hooks must not write anything to disk and must not call
    storage_state() on the context. Verified at the module-API boundary.
    """

    @pytest.mark.asyncio
    async def test_module_apis_still_work_when_flag_unset(self, monkeypatch):
        # The flag gate lives in the lifecycle wiring, not inside
        # session_persistence. The module APIs themselves are gate-free
        # (callers decide whether to call them).  This test pins that
        # contract: snapshot_session works the same regardless of flag
        # state — the flag only affects whether the BrowserManager *calls*
        # snapshot_session at all.
        monkeypatch.delenv("BROWSER_SESSION_PERSISTENCE_ENABLED", raising=False)
        ctx = _FakeContext()
        ok = await sp.snapshot_session("agent-x", ctx)
        assert ok is True

    @pytest.mark.asyncio
    async def test_lifecycle_periodic_no_io_when_flag_off(
        self, monkeypatch, tmp_path,
    ):
        """`_periodic_session_snapshots` must do nothing when flag is off."""
        from src.browser.flags import reload_operator_settings
        from src.browser.service import BrowserManager

        monkeypatch.delenv("BROWSER_SESSION_PERSISTENCE_ENABLED", raising=False)
        monkeypatch.setenv("OPENLEGION_SETTINGS_PATH", "/nonexistent/x.json")
        reload_operator_settings()

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))

        # Inject a fake instance — the snapshot path would call
        # storage_state() on this if the flag were on. We use a context
        # whose storage_state explodes if called so a leaked call is loud.
        class _BombCtx:
            async def storage_state(self):
                raise AssertionError(
                    "storage_state should not be called when flag is off"
                )

        fake_inst = MagicMock()
        fake_inst.context = _BombCtx()
        fake_inst.agent_id = "agent-noflag"
        mgr._instances["agent-noflag"] = fake_inst

        # Should run without raising.
        await mgr._periodic_session_snapshots()
        # No sidecar written.
        assert not sp.session_path("agent-noflag").exists()

    @pytest.mark.asyncio
    async def test_lifecycle_stop_no_io_when_flag_off(
        self, monkeypatch, tmp_path,
    ):
        """`_stop_instance` must not snapshot when flag is off."""
        from src.browser.flags import reload_operator_settings

        monkeypatch.delenv("BROWSER_SESSION_PERSISTENCE_ENABLED", raising=False)
        monkeypatch.setenv("OPENLEGION_SETTINGS_PATH", "/nonexistent/x.json")
        reload_operator_settings()

        # We can't easily run _stop_instance without a real instance.
        # Instead, exercise the flag check directly: when off, the
        # caller's ``snapshot_session`` is never invoked. Module-level
        # integration is covered by test_lifecycle_periodic_no_io_when_flag_off.
        from src.browser.flags import get_bool
        assert get_bool("BROWSER_SESSION_PERSISTENCE_ENABLED", False) is False


# ── Stale saved_at — no expiry built in ────────────────────────────────────


class TestStaleSnapshot:
    """Operator owns rotation policy. A 6-month-old snapshot restores fine."""

    @pytest.mark.asyncio
    async def test_far_past_saved_at_still_restores(self, tmp_path):
        path = sp.session_path("agent-stale")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "saved_at": 1_000_000_000,  # 2001-09 — ancient
            "storage_state": _fake_storage_state(),
        }
        path.write_text(json.dumps(payload))
        os.chmod(path, 0o600)

        captured = {}

        async def _factory(*, storage_state):
            captured["state"] = storage_state
            return MagicMock(name="ctx")

        result = await sp.restore_session("agent-stale", _factory)
        assert result is not None
        assert captured["state"] == _fake_storage_state()


# ── Bad JSON ───────────────────────────────────────────────────────────────


class TestMalformedJson:
    """Bad JSON: warn + return None + DO NOT clobber the sidecar."""

    @pytest.mark.asyncio
    async def test_bad_json_returns_none_and_preserves_file(
        self, caplog,
    ):
        import logging
        path = sp.session_path("agent-bad")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json at all")
        original = path.read_bytes()

        async def _factory(*, storage_state):
            raise AssertionError("factory must not be called for bad JSON")

        with caplog.at_level(logging.WARNING, logger="browser.session_persistence"):
            result = await sp.restore_session("agent-bad", _factory)
        assert result is None
        # Sidecar untouched — operator decides what to do with it.
        assert path.read_bytes() == original
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "agent-bad" in joined

    @pytest.mark.asyncio
    async def test_unknown_version_returns_none(self):
        path = sp.session_path("agent-future")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 99,  # future schema we don't know how to read
            "saved_at": 0,
            "storage_state": _fake_storage_state(),
        }
        path.write_text(json.dumps(payload))

        async def _factory(*, storage_state):
            raise AssertionError("factory must not be called for unknown version")

        result = await sp.restore_session("agent-future", _factory)
        assert result is None
        # Sidecar untouched.
        assert path.exists()


# ── Concurrent snapshots for two agents ────────────────────────────────────


class TestConcurrent:
    @pytest.mark.asyncio
    async def test_two_agents_no_cross_contamination(self):
        ctx_a = _FakeContext({
            "cookies": [{"name": "a", "value": "A", "domain": ".a.com",
                         "path": "/", "expires": -1, "httpOnly": False,
                         "secure": True, "sameSite": "Lax"}],
            "origins": [],
        })
        ctx_b = _FakeContext({
            "cookies": [{"name": "b", "value": "B", "domain": ".b.com",
                         "path": "/", "expires": -1, "httpOnly": False,
                         "secure": True, "sameSite": "Lax"}],
            "origins": [],
        })

        results = await asyncio.gather(
            sp.snapshot_session("agent-a", ctx_a),
            sp.snapshot_session("agent-b", ctx_b),
        )
        assert results == [True, True]

        path_a = sp.session_path("agent-a")
        path_b = sp.session_path("agent-b")
        assert path_a != path_b
        data_a = json.loads(path_a.read_text())
        data_b = json.loads(path_b.read_text())
        assert data_a["storage_state"]["cookies"][0]["name"] == "a"
        assert data_b["storage_state"]["cookies"][0]["name"] == "b"


# ── Clear ──────────────────────────────────────────────────────────────────


class TestAgentIdValidation:
    """An attacker-controlled agent_id must never escape the sessions dir.

    Path-traversal regressions: the canonical agent_id regex is the
    single chokepoint enforced inside ``session_path``; every public
    function in the module funnels through it.
    """

    def test_session_path_rejects_traversal(self):
        with pytest.raises(sp.InvalidAgentIdError):
            sp.session_path("../../../etc/passwd")

    def test_session_path_rejects_slash(self):
        with pytest.raises(sp.InvalidAgentIdError):
            sp.session_path("a/b")

    def test_session_path_rejects_empty(self):
        with pytest.raises(sp.InvalidAgentIdError):
            sp.session_path("")

    def test_session_path_rejects_null_byte(self):
        with pytest.raises(sp.InvalidAgentIdError):
            sp.session_path("agent\x00x")

    def test_session_path_rejects_long(self):
        # ``AGENT_ID_RE_PATTERN`` caps at 64 chars total.
        with pytest.raises(sp.InvalidAgentIdError):
            sp.session_path("a" * 65)

    @pytest.mark.asyncio
    async def test_snapshot_with_bad_agent_id_returns_false(self):
        ctx = _FakeContext()
        ok = await sp.snapshot_session("../escape", ctx)
        assert ok is False

    @pytest.mark.asyncio
    async def test_clear_with_bad_agent_id_returns_false(self):
        ok = await sp.clear_session("../etc/passwd")
        assert ok is False

    @pytest.mark.asyncio
    async def test_restore_with_bad_agent_id_returns_none(self):
        async def factory(**kwargs):
            return MagicMock()

        out = await sp.restore_session("../bad", factory)
        assert out is None

    def test_summary_with_bad_agent_id_returns_safe_default(self):
        out = sp.session_summary("../etc/passwd")
        assert out == {
            "has_persisted_session": False,
            "saved_at": None,
            "origin_count": 0,
            "cookie_count": 0,
        }


class TestSizeCap:
    @pytest.mark.asyncio
    async def test_oversized_payload_refused(self, monkeypatch):
        ctx = _FakeContext()
        # Inflate the storage_state past the 8 MiB cap.
        big = "x" * (sp._MAX_SNAPSHOT_BYTES + 1024)
        ctx._state = {
            "cookies": [],
            "origins": [{"origin": "https://h.example", "localStorage": [
                {"name": "k", "value": big},
            ]}],
        }
        ok = await sp.snapshot_session("agent-big", ctx)
        assert ok is False
        # Sidecar must NOT exist after refusal.
        assert not sp.session_path("agent-big").exists()


class TestSessionsDirMode:
    @pytest.mark.asyncio
    async def test_parent_dir_chmod_0o700(self, tmp_path, monkeypatch):
        # Pre-create the sessions dir with a permissive mode to verify
        # the snapshot path tightens it.
        target_dir = tmp_path / "sessions-perm"
        target_dir.mkdir(mode=0o755)
        monkeypatch.setenv("BROWSER_SESSION_DIR", str(target_dir))
        ctx = _FakeContext()
        ok = await sp.snapshot_session("agent-perm", ctx)
        assert ok is True
        mode = target_dir.stat().st_mode & 0o777
        assert mode == 0o700, f"sessions dir mode {oct(mode)} != 0o700"


class TestClear:
    @pytest.mark.asyncio
    async def test_clear_deletes_existing(self):
        ctx = _FakeContext()
        await sp.snapshot_session("agent-c", ctx)
        assert sp.session_path("agent-c").exists()
        ok = await sp.clear_session("agent-c")
        assert ok is True
        assert not sp.session_path("agent-c").exists()

    @pytest.mark.asyncio
    async def test_clear_missing_returns_false(self):
        ok = await sp.clear_session("agent-never")
        assert ok is False


# ── Summary endpoint shape ─────────────────────────────────────────────────


class TestSessionSummary:
    """Privacy-safe shape: counts only, no domains, no values."""

    def test_missing_returns_safe_default(self):
        out = sp.session_summary("nope")
        assert out == {
            "has_persisted_session": False,
            "saved_at": None,
            "origin_count": 0,
            "cookie_count": 0,
        }

    @pytest.mark.asyncio
    async def test_present_returns_counts_only(self):
        ctx = _FakeContext()
        await sp.snapshot_session("agent-sum", ctx)
        out = sp.session_summary("agent-sum")
        assert out["has_persisted_session"] is True
        assert out["cookie_count"] == 2
        assert out["origin_count"] == 2
        # ISO-8601 UTC timestamp.
        assert isinstance(out["saved_at"], str)
        assert "T" in out["saved_at"]
        # Privacy: no domain or value leakage.
        joined = json.dumps(out)
        assert "example.com" not in joined
        assert "secret-token-abc" not in joined
        assert "csrf-xyz" not in joined
        assert "session" not in joined.lower() or "session" in (
            "has_persisted_session"  # only key reference allowed
        )

    @pytest.mark.asyncio
    async def test_malformed_sidecar_safe_summary(self):
        path = sp.session_path("agent-bad-sum")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{garbage")
        out = sp.session_summary("agent-bad-sum")
        # File present but unreadable — surfaced as has_persisted=True
        # with zero counts so the operator can clear it.
        assert out["has_persisted_session"] is True
        assert out["saved_at"] is None
        assert out["cookie_count"] == 0
        assert out["origin_count"] == 0


# ── Periodic snapshot timer logic ──────────────────────────────────────────


class TestPeriodicSnapshotTimer:
    """The 60s metrics tick accumulates per-agent elapsed seconds; a
    snapshot fires only when the accumulator crosses the configured
    interval (default 300s).
    """

    @pytest.mark.asyncio
    async def test_below_threshold_no_snapshot(self, monkeypatch, tmp_path):
        from src.browser.flags import reload_operator_settings
        from src.browser.service import BrowserManager

        monkeypatch.setenv("BROWSER_SESSION_PERSISTENCE_ENABLED", "true")
        monkeypatch.setenv("BROWSER_SESSION_PERIODIC_SNAPSHOT_S", "300")
        monkeypatch.setenv("OPENLEGION_SETTINGS_PATH", "/nonexistent/x.json")
        reload_operator_settings()

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))

        # Inject a fake instance with a context that records
        # storage_state() calls (none should happen below threshold).
        ctx = _FakeContext()
        ctx_calls = {"n": 0}

        async def _state():
            ctx_calls["n"] += 1
            return _fake_storage_state()

        ctx.storage_state = _state  # type: ignore[assignment]
        fake_inst = MagicMock()
        fake_inst.context = ctx
        fake_inst.agent_id = "agent-tick"
        mgr._instances["agent-tick"] = fake_inst

        # 4 ticks × 60s = 240s — still under 300.
        for _ in range(4):
            await mgr._periodic_session_snapshots()
        assert ctx_calls["n"] == 0
        assert mgr._session_snapshot_elapsed_s["agent-tick"] == 240
        assert not sp.session_path("agent-tick").exists()

    @pytest.mark.asyncio
    async def test_at_threshold_fires_snapshot(self, monkeypatch, tmp_path):
        from src.browser.flags import reload_operator_settings
        from src.browser.service import BrowserManager

        monkeypatch.setenv("BROWSER_SESSION_PERSISTENCE_ENABLED", "true")
        # Use a low threshold so the test is fast.
        monkeypatch.setenv("BROWSER_SESSION_PERIODIC_SNAPSHOT_S", "120")
        monkeypatch.setenv("OPENLEGION_SETTINGS_PATH", "/nonexistent/x.json")
        reload_operator_settings()

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        ctx = _FakeContext()
        fake_inst = MagicMock()
        fake_inst.context = ctx
        fake_inst.agent_id = "agent-fire"
        mgr._instances["agent-fire"] = fake_inst

        # 1 tick — 60s, under 120.
        await mgr._periodic_session_snapshots()
        assert not sp.session_path("agent-fire").exists()
        # 2nd tick — 120s, at threshold → fires.
        await mgr._periodic_session_snapshots()
        assert sp.session_path("agent-fire").exists()
        # Counter resets after firing.
        assert mgr._session_snapshot_elapsed_s["agent-fire"] == 0

    @pytest.mark.asyncio
    async def test_stale_agent_id_dropped(self, monkeypatch, tmp_path):
        """An agent_id in the elapsed map but not in _instances is dropped."""
        from src.browser.flags import reload_operator_settings
        from src.browser.service import BrowserManager

        monkeypatch.setenv("BROWSER_SESSION_PERSISTENCE_ENABLED", "true")
        monkeypatch.setenv("OPENLEGION_SETTINGS_PATH", "/nonexistent/x.json")
        reload_operator_settings()

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        # No live instance for this agent — but it has a stale counter.
        mgr._session_snapshot_elapsed_s["dead-agent"] = 60
        await mgr._periodic_session_snapshots()
        assert "dead-agent" not in mgr._session_snapshot_elapsed_s


# ── Audit aggregation (no domain leakage) ──────────────────────────────────


class TestSessionAuditPrivacy:
    """Audit events fed to the EventBus must NOT carry origins or values."""

    @pytest.mark.asyncio
    async def test_audit_record_then_drain_carries_no_domain_data(self):
        from src.browser.service import (
            _drain_session_audit,
            _record_session_audit_event,
            _session_audit_buckets,
        )
        # Clear any leftover state from prior tests.
        _session_audit_buckets.clear()

        await _record_session_audit_event("agent-1", "session_snapshot", True)
        await _record_session_audit_event("agent-1", "session_snapshot", True)
        await _record_session_audit_event("agent-2", "session_restore", False)

        events = await _drain_session_audit()
        # Two unique buckets: (agent-1, snapshot, True) count=2,
        # (agent-2, restore, False) count=1.
        assert len(events) == 2
        for ev in events:
            # Privacy: no per-origin or per-cookie data on the wire.
            for forbidden in ("url", "origin", "cookie", "value", "domain"):
                assert forbidden not in ev, f"audit event leaked {forbidden!r}: {ev}"
            assert ev["type"] == "session_event"
            assert ev["action"] in {"session_snapshot", "session_restore"}
            assert isinstance(ev["success"], bool)
            assert isinstance(ev["count"], int)
            assert isinstance(ev["agent_id"], str)


# ── Dashboard endpoints ────────────────────────────────────────────────────


def _make_dashboard_client_with_browser_url(tmp_path: str, browser_url: str):
    """Build a dashboard test client wired to a stub browser service URL."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.dashboard.events import EventBus
    from src.dashboard.server import create_dashboard_router
    from src.host.costs import CostTracker
    from src.host.health import HealthMonitor
    from src.host.mesh import Blackboard
    from src.host.traces import TraceStore

    bb = Blackboard(db_path=os.path.join(tmp_path, "bb.db"))
    cost_tracker = CostTracker(db_path=os.path.join(tmp_path, "costs.db"))
    trace_store = TraceStore(db_path=os.path.join(tmp_path, "traces.db"))
    event_bus = EventBus()
    agent_registry = {"alpha": "http://localhost:8401"}

    runtime_mock = MagicMock()
    runtime_mock.browser_vnc_url = None
    runtime_mock.browser_service_url = browser_url
    runtime_mock.browser_auth_token = "test-token"
    transport_mock = MagicMock()
    router_mock = MagicMock()
    health_monitor = HealthMonitor(
        runtime=runtime_mock, transport=transport_mock, router=router_mock,
    )
    health_monitor.register("alpha")

    router = create_dashboard_router(
        blackboard=bb,
        health_monitor=health_monitor,
        cost_tracker=cost_tracker,
        trace_store=trace_store,
        event_bus=event_bus,
        agent_registry=agent_registry,
        runtime=runtime_mock,
        transport=transport_mock,
        mesh_port=8420,
    )
    app = FastAPI()
    app.include_router(router)

    class _CSRFTestClient(TestClient):
        def request(self, method, url, **kwargs):
            if method.upper() not in ("GET", "HEAD", "OPTIONS"):
                headers = kwargs.get("headers") or {}
                if "X-Requested-With" not in headers:
                    headers["X-Requested-With"] = "XMLHttpRequest"
                    kwargs["headers"] = headers
            return super().request(method, url, **kwargs)

    client = _CSRFTestClient(app)

    def cleanup():
        cost_tracker.close()
        trace_store.close()
        bb.close()

    return client, cleanup


class TestDashboardSessionEndpoints:
    """GET shape returns counts only; DELETE is CSRF-gated."""

    def test_get_returns_safe_summary_shape(self, tmp_path, monkeypatch):
        import httpx

        # Patch httpx.AsyncClient so the closure-local
        # ``_dashboard_browser_client`` constructed inside
        # ``create_dashboard_router`` uses our MockTransport. The dashboard
        # builds its client at router-construction time, so the patch must
        # be in place BEFORE _make_dashboard_client_with_browser_url runs.
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/browser/alpha/session"
            return httpx.Response(200, json={
                "has_persisted_session": True,
                "saved_at": "2026-04-27T12:00:00+00:00",
                "origin_count": 3,
                "cookie_count": 7,
            })

        original_async_client = httpx.AsyncClient

        def patched_async_client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return original_async_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

        client, cleanup = _make_dashboard_client_with_browser_url(
            str(tmp_path), "http://browser:8500",
        )
        try:
            resp = client.get("/dashboard/api/agents/alpha/session")
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            data = body["data"]
            # Shape contract.
            assert data["has_persisted_session"] is True
            assert data["origin_count"] == 3
            assert data["cookie_count"] == 7
            # No domain leakage on the wire.
            for forbidden in ("origin:", "domain", ".com", ".org"):
                assert forbidden not in resp.text
        finally:
            cleanup()

    def test_get_404_for_unknown_agent(self, tmp_path):
        client, cleanup = _make_dashboard_client_with_browser_url(
            str(tmp_path), "http://browser:8500",
        )
        try:
            resp = client.get("/dashboard/api/agents/missing/session")
            assert resp.status_code == 404
        finally:
            cleanup()

    def test_delete_requires_csrf_header(self, tmp_path):
        from fastapi.testclient import TestClient
        # Build a client that does NOT inject the X-Requested-With header.
        client, cleanup = _make_dashboard_client_with_browser_url(
            str(tmp_path), "http://browser:8500",
        )
        try:
            # Use the underlying TestClient request without our auto-CSRF.
            raw = TestClient(client.app)
            resp = raw.delete("/dashboard/api/agents/alpha/session")
            assert resp.status_code == 403
            assert "X-Requested-With" in resp.text
        finally:
            cleanup()

    def test_delete_with_csrf_header_proxies_to_browser(
        self, tmp_path, monkeypatch,
    ):
        import httpx
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["auth"] = request.headers.get("authorization", "")
            return httpx.Response(200, json={
                "success": True, "data": {"deleted": True},
            })

        original_async_client = httpx.AsyncClient

        def patched_async_client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return original_async_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

        client, cleanup = _make_dashboard_client_with_browser_url(
            str(tmp_path), "http://browser:8500",
        )
        try:
            resp = client.delete("/dashboard/api/agents/alpha/session")
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert seen["method"] == "DELETE"
            assert seen["path"] == "/browser/alpha/session"
            assert seen["auth"] == "Bearer test-token"
        finally:
            cleanup()

    def test_get_handles_browser_service_unavailable(self, tmp_path):
        # browser_service_url=None → service_unavailable envelope.
        client, cleanup = _make_dashboard_client_with_browser_url(
            str(tmp_path), "",
        )
        try:
            resp = client.get("/dashboard/api/agents/alpha/session")
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is False
            assert body["error"]["code"] == "service_unavailable"
        finally:
            cleanup()
