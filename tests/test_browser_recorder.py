"""Tests for the behavioral entropy recorder (Phase 2 §5.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _recorder(monkeypatch, tmp_path, enabled: bool, agent_id: str = "a1"):
    """Construct a recorder with the flag set and dump dir under tmp."""
    if enabled:
        monkeypatch.setenv("BROWSER_RECORD_BEHAVIOR", "1")
    else:
        monkeypatch.delenv("BROWSER_RECORD_BEHAVIOR", raising=False)
    # Reset the flag module's operator-settings cache so env change
    # takes effect immediately.
    import src.browser.flags as flags
    flags._operator_settings = None
    from src.browser.recorder import BehaviorRecorder
    return BehaviorRecorder(agent_id, dump_dir=tmp_path / "debug")


class TestEnabledFlag:
    def test_disabled_by_default(self, monkeypatch, tmp_path):
        r = _recorder(monkeypatch, tmp_path, enabled=False)
        assert r.enabled is False

    def test_flag_enables(self, monkeypatch, tmp_path):
        r = _recorder(monkeypatch, tmp_path, enabled=True)
        assert r.enabled is True

    def test_disabled_recorder_is_noop(self, monkeypatch, tmp_path):
        r = _recorder(monkeypatch, tmp_path, enabled=False)
        r.record_click(method="cdp", success=True)
        r.record_keystrokes(char_count=5, fast=False, method="cdp")
        r.record_scroll(direction="down", delta=100, method="x11")
        r.record_navigate(host="example.com", wait_until="load")
        assert len(r) == 0
        assert r.dump() is None

    def test_runtime_toggle_takes_effect_without_reset(
        self, monkeypatch, tmp_path,
    ):
        """Operators should be able to flip BROWSER_RECORD_BEHAVIOR via
        env/settings and have the next recorded event honor it — no
        browser restart required."""
        r = _recorder(monkeypatch, tmp_path, enabled=False)
        r.record_click(method="cdp", success=True)
        assert len(r) == 0

        monkeypatch.setenv("BROWSER_RECORD_BEHAVIOR", "1")
        import src.browser.flags as flags
        flags._operator_settings = None

        r.record_click(method="cdp", success=True)
        assert len(r) == 1

    def test_dump_survives_runtime_disable(self, monkeypatch, tmp_path):
        """If operator disables the flag between record and dump, the
        already-captured events must still flush — silently dropping
        them would surprise the operator."""
        r = _recorder(monkeypatch, tmp_path, enabled=True)
        r.record_click(method="cdp", success=True)
        assert len(r) == 1

        monkeypatch.delenv("BROWSER_RECORD_BEHAVIOR", raising=False)
        import src.browser.flags as flags
        flags._operator_settings = None

        out = r.dump()
        assert out is not None and out.exists()


class TestRecording:
    def test_click_appends(self, monkeypatch, tmp_path):
        r = _recorder(monkeypatch, tmp_path, enabled=True)
        r.record_click(method="cdp", success=True)
        assert len(r) == 1
        event = r._events[-1]
        assert event["type"] == "click"
        assert event["method"] == "cdp"
        assert event["success"] is True
        assert event["interval_s"] is None  # first event

    def test_interval_tracked(self, monkeypatch, tmp_path):
        r = _recorder(monkeypatch, tmp_path, enabled=True)
        r.record_click(method="cdp", success=True)
        r.record_click(method="cdp", success=True)
        assert r._events[0]["interval_s"] is None
        assert r._events[1]["interval_s"] is not None
        assert r._events[1]["interval_s"] >= 0

    def test_keystrokes_only_store_count(self, monkeypatch, tmp_path):
        """Privacy invariant: the actual text is never stored."""
        r = _recorder(monkeypatch, tmp_path, enabled=True)
        r.record_keystrokes(char_count=12, fast=False, method="x11")
        event = r._events[-1]
        assert event["char_count"] == 12
        # No "text", "chars", or similar field
        forbidden = {"text", "chars", "content", "value"}
        assert not (set(event.keys()) & forbidden)

    def test_navigate_records_only_host(self, monkeypatch, tmp_path):
        """Privacy invariant: path / query / fragment never persisted."""
        r = _recorder(monkeypatch, tmp_path, enabled=True)
        r.record_navigate(host="example.com", wait_until="load")
        event = r._events[-1]
        assert event["host"] == "example.com"
        assert "url" not in event
        assert "path" not in event
        assert "query" not in event


class TestBufferCap:
    def test_ring_buffer_caps(self, monkeypatch, tmp_path):
        """Buffer maxlen bounds memory under long sessions."""
        from src.browser.recorder import BehaviorRecorder
        monkeypatch.setenv("BROWSER_RECORD_BEHAVIOR", "1")
        import src.browser.flags as flags
        flags._operator_settings = None
        r = BehaviorRecorder("a1", buffer_size=3, dump_dir=tmp_path / "d")
        for _ in range(10):
            r.record_click(method="cdp", success=True)
        assert len(r) == 3


class TestDump:
    def test_dump_writes_jsonl(self, monkeypatch, tmp_path):
        r = _recorder(monkeypatch, tmp_path, enabled=True, agent_id="worker")
        r.record_click(method="cdp", success=True)
        r.record_keystrokes(char_count=3, fast=True, method="cdp-fast")
        r.record_navigate(host="x.com", wait_until="networkidle")

        out = r.dump(reason="stop")
        assert out is not None and out.exists()
        assert out.name.startswith("worker-")
        assert out.suffix == ".jsonl"

        lines = out.read_text(encoding="utf-8").splitlines()
        # Header + 3 events
        assert len(lines) == 4
        header = json.loads(lines[0])
        assert header["schema"].startswith("openlegion.browser.recorder/")
        assert header["agent"] == "worker"
        assert header["reason"] == "stop"
        assert header["event_count"] == 3

        events = [json.loads(line) for line in lines[1:]]
        types = [e["type"] for e in events]
        assert types == ["click", "keystrokes", "navigate"]

    def test_dump_clears_buffer(self, monkeypatch, tmp_path):
        r = _recorder(monkeypatch, tmp_path, enabled=True)
        r.record_click(method="cdp", success=True)
        assert len(r) == 1
        r.dump()
        assert len(r) == 0

    def test_dump_empty_buffer_returns_none(self, monkeypatch, tmp_path):
        r = _recorder(monkeypatch, tmp_path, enabled=True)
        assert r.dump() is None

    def test_dump_unwritable_dir_returns_none(self, monkeypatch, tmp_path):
        """Recorder failure must never crash the caller."""
        from src.browser.recorder import BehaviorRecorder
        monkeypatch.setenv("BROWSER_RECORD_BEHAVIOR", "1")
        import src.browser.flags as flags
        flags._operator_settings = None
        # Point at a path that exists as a file, so mkdir fails.
        file_as_dir = tmp_path / "not_a_dir"
        file_as_dir.write_text("")
        r = BehaviorRecorder("a1", dump_dir=file_as_dir / "child")
        r.record_click(method="cdp", success=True)
        assert r.dump() is None  # silent failure

    def test_prune_keeps_only_most_recent(self, monkeypatch, tmp_path):
        """Regression (Codex #2): the recorder must cap the number of
        dump files it retains on disk. A long-running dev session with
        agent churn would otherwise fill the disk even though the
        in-memory buffer is bounded."""
        from src.browser.recorder import BehaviorRecorder
        monkeypatch.setenv("BROWSER_RECORD_BEHAVIOR", "1")
        import src.browser.flags as flags
        flags._operator_settings = None

        dump_dir = tmp_path / "debug"
        # Lower the cap for a tractable test.
        monkeypatch.setattr("src.browser.recorder._MAX_DUMP_FILES", 3)

        # Create 7 back-to-back dumps; only the 3 most recent should survive.
        import time as _t
        for i in range(7):
            r = BehaviorRecorder(f"a{i}", dump_dir=dump_dir)
            r.record_click(method="cdp", success=True)
            r.dump()
            # Monotonic mtime step so sort-by-mtime is deterministic.
            _t.sleep(0.005)

        remaining = sorted(dump_dir.glob("*.jsonl"))
        assert len(remaining) == 3

    def test_rapid_dumps_do_not_overwrite(self, monkeypatch, tmp_path):
        """Two dumps for the same agent within the same second must
        not overwrite each other — the random suffix keeps both on
        disk even when ``int(time.time())`` matches."""
        r1 = _recorder(monkeypatch, tmp_path, enabled=True, agent_id="dup")
        r1.record_click(method="cdp", success=True)
        out1 = r1.dump()
        r2 = _recorder(monkeypatch, tmp_path, enabled=True, agent_id="dup")
        r2.record_click(method="cdp", success=True)
        out2 = r2.dump()
        assert out1 is not None and out2 is not None
        assert out1 != out2
        assert out1.exists() and out2.exists()

    def test_dump_sanitizes_agent_id_in_filename(self, monkeypatch, tmp_path):
        """Even if an unusual id gets through upstream validation, the
        dump path must not escape the dump directory."""
        from src.browser.recorder import BehaviorRecorder
        monkeypatch.setenv("BROWSER_RECORD_BEHAVIOR", "1")
        import src.browser.flags as flags
        flags._operator_settings = None
        r = BehaviorRecorder("../evil", dump_dir=tmp_path / "d")
        r.record_click(method="cdp", success=True)
        out = r.dump()
        assert out is not None
        # No `..` in the final path
        assert ".." not in out.name
        # Dump lives under the configured dir
        assert Path(tmp_path / "d").resolve() in out.parents


class TestIntegrationWithCamoufoxInstance:
    """CamoufoxInstance must construct a recorder unconditionally."""

    def test_instance_has_recorder_attr(self, monkeypatch):
        """Construction is cheap whether the flag is set or not."""
        from unittest.mock import MagicMock

        from src.browser.service import CamoufoxInstance
        monkeypatch.delenv("BROWSER_RECORD_BEHAVIOR", raising=False)
        import src.browser.flags as flags
        flags._operator_settings = None
        inst = CamoufoxInstance(
            "a1", MagicMock(), MagicMock(), MagicMock(),
        )
        assert hasattr(inst, "recorder")
        assert inst.recorder.enabled is False


@pytest.mark.asyncio
class TestDumpFromStopInstance:
    """_stop_instance must trigger the recorder dump so the operator
    gets a JSONL file after each agent tears down."""

    async def test_stop_instance_dumps_recorder(self, monkeypatch, tmp_path):
        from unittest.mock import AsyncMock, MagicMock

        from src.browser.service import BrowserManager, CamoufoxInstance

        monkeypatch.setenv("BROWSER_RECORD_BEHAVIOR", "1")
        import src.browser.flags as flags
        flags._operator_settings = None

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = CamoufoxInstance(
            "bye", MagicMock(), MagicMock(), MagicMock(),
        )
        inst.recorder._dump_dir = tmp_path / "debug"
        inst.context = MagicMock()
        inst.context.close = AsyncMock()
        inst.record_click_method = "cdp"
        mgr._instances["bye"] = inst

        inst.recorder.record_click(method="cdp", success=True)
        async with mgr._manager_lock():
            await mgr._stop_instance("bye")

        # One JSONL file with "bye-<ts>.jsonl" shape
        dumps = list((tmp_path / "debug").glob("bye-*.jsonl"))
        assert len(dumps) == 1

    async def test_stop_waits_for_in_flight_action(self, monkeypatch, tmp_path):
        """Regression (Codex #1 P1): a hot-path task holding ``inst.lock``
        mid-click must finish — including its ``record_*`` append — before
        the dump clears the buffer. Without acquiring the lock in
        ``_stop_instance``, that trailing append lands in a cleared deque
        and is silently lost at instance GC."""
        from unittest.mock import AsyncMock, MagicMock

        from src.browser.service import BrowserManager, CamoufoxInstance

        monkeypatch.setenv("BROWSER_RECORD_BEHAVIOR", "1")
        import src.browser.flags as flags
        flags._operator_settings = None

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = CamoufoxInstance(
            "wait", MagicMock(), MagicMock(), MagicMock(),
        )
        inst.recorder._dump_dir = tmp_path / "debug"
        inst.context = MagicMock()
        inst.context.close = AsyncMock()
        mgr._instances["wait"] = inst

        # Hold inst.lock and schedule a late append from inside. Stop
        # must wait for us to release before dumping.
        import asyncio as _asyncio
        inst_lock_acquired = _asyncio.Event()
        stop_may_proceed = _asyncio.Event()

        async def hold_lock_then_record():
            async with inst.lock:
                inst_lock_acquired.set()
                await stop_may_proceed.wait()
                # This append must land in the dump file, not be lost.
                inst.recorder.record_click(method="cdp", success=True)

        holder = _asyncio.create_task(hold_lock_then_record())
        await inst_lock_acquired.wait()

        async def do_stop():
            async with mgr._manager_lock():
                await mgr._stop_instance("wait")

        stop_task = _asyncio.create_task(do_stop())
        # Give stop_task a chance to start and block on inst.lock.
        await _asyncio.sleep(0.05)
        assert not stop_task.done(), "stop should be blocked on inst.lock"

        stop_may_proceed.set()
        await holder
        await stop_task

        dumps = list((tmp_path / "debug").glob("wait-*.jsonl"))
        assert len(dumps) == 1
        # The trailing click made it to disk.
        content = dumps[0].read_text()
        clicks = [line for line in content.splitlines() if '"click"' in line]
        assert len(clicks) == 1
