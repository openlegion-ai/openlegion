"""Tests for request tracing: TraceStore, contextvar, lane propagation, transport headers."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.host.traces import TraceStore
from src.shared.trace import TRACE_HEADER, current_trace_id, new_trace_id, trace_headers


# ── TraceStore ───────────────────────────────────────────────

class TestTraceStore:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "traces.db")
        self.store = TraceStore(db_path=self.db_path, max_events=100)

    def teardown_method(self):
        self.store.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_record_and_retrieve(self):
        tid = "tr_abc123"
        self.store.record(tid, source="repl", agent="alpha", event_type="chat", detail="hello")
        self.store.record(tid, source="mesh.api_proxy", agent="alpha", event_type="llm_call", detail="llm/chat")
        events = self.store.get_trace(tid)
        assert len(events) == 2
        assert events[0]["trace_id"] == tid
        assert events[0]["source"] == "repl"
        assert events[0]["event_type"] == "chat"
        assert events[1]["event_type"] == "llm_call"

    def test_list_recent(self):
        self.store.record("tr_001", "repl", "a", "chat", "first")
        self.store.record("tr_002", "repl", "b", "chat", "second")
        recent = self.store.list_recent(limit=10)
        assert len(recent) == 2
        # Newest first
        assert recent[0]["trace_id"] == "tr_002"
        assert recent[1]["trace_id"] == "tr_001"

    def test_list_recent_respects_limit(self):
        for i in range(10):
            self.store.record(f"tr_{i:03d}", "repl", "a", "chat")
        recent = self.store.list_recent(limit=3)
        assert len(recent) == 3

    def test_ring_buffer_eviction(self):
        """Insert more than max_events and verify oldest are evicted."""
        small_store = TraceStore(
            db_path=os.path.join(self._tmpdir, "small.db"), max_events=5,
        )
        try:
            for i in range(10):
                small_store.record(f"tr_{i:03d}", "repl", "a", "chat", f"msg-{i}")
            recent = small_store.list_recent(limit=100)
            assert len(recent) <= 5
            # Newest should still be present
            trace_ids = {e["trace_id"] for e in recent}
            assert "tr_009" in trace_ids
            # Oldest should be evicted
            assert "tr_000" not in trace_ids
        finally:
            small_store.close()

    def test_get_trace_empty(self):
        events = self.store.get_trace("tr_nonexistent")
        assert events == []

    def test_duration_ms_field(self):
        self.store.record("tr_dur", "mesh", "a", "llm_call", duration_ms=150)
        events = self.store.get_trace("tr_dur")
        assert events[0]["duration_ms"] == 150


# ── Contextvar + helpers ─────────────────────────────────────

class TestTraceContextvar:
    def test_default_is_none(self):
        tok = current_trace_id.set(None)
        try:
            assert current_trace_id.get() is None
        finally:
            current_trace_id.reset(tok)

    def test_set_and_get(self):
        tid = new_trace_id()
        tok = current_trace_id.set(tid)
        try:
            assert current_trace_id.get() == tid
        finally:
            current_trace_id.reset(tok)

    def test_new_trace_id_format(self):
        tid = new_trace_id()
        assert tid.startswith("tr_")
        assert len(tid) == 15  # "tr_" + 12 hex chars

    def test_new_trace_id_unique(self):
        ids = {new_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_trace_headers_empty_when_no_trace(self):
        tok = current_trace_id.set(None)
        try:
            assert trace_headers() == {}
        finally:
            current_trace_id.reset(tok)

    def test_trace_headers_returns_header(self):
        tid = "tr_abc123def456"
        tok = current_trace_id.set(tid)
        try:
            hdrs = trace_headers()
            assert hdrs == {TRACE_HEADER: tid}
        finally:
            current_trace_id.reset(tok)


# ── Lane propagates trace_id ────────────────────────────────

class TestLanePropagatesTraceId:
    @pytest.mark.asyncio
    async def test_worker_sets_contextvar(self):
        """Verify the lane worker sets current_trace_id before dispatch."""
        from src.host.lanes import LaneManager

        captured_trace_ids = []

        async def mock_dispatch(agent: str, message: str) -> str:
            captured_trace_ids.append(current_trace_id.get())
            return "ok"

        lm = LaneManager(dispatch_fn=mock_dispatch)
        tid = "tr_lane_test_01"
        result = await lm.enqueue("alpha", "hello", trace_id=tid)
        assert result == "ok"
        assert captured_trace_ids == [tid]
        await lm.stop()

    @pytest.mark.asyncio
    async def test_worker_sets_none_when_no_trace(self):
        """Without trace_id, contextvar should be set to None."""
        from src.host.lanes import LaneManager

        captured = []

        async def mock_dispatch(agent: str, message: str) -> str:
            captured.append(current_trace_id.get())
            return "ok"

        lm = LaneManager(dispatch_fn=mock_dispatch)
        await lm.enqueue("alpha", "hello")
        assert captured == [None]
        await lm.stop()


# ── Transport injects trace header ──────────────────────────

class TestTransportInjectsTraceHeader:
    @pytest.mark.asyncio
    async def test_http_transport_sends_trace_header(self):
        """HttpTransport.request should include X-Trace-Id from contextvar."""
        from src.host.transport import HttpTransport

        t = HttpTransport()
        t.register("alpha", "http://localhost:8401")

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        t._clients[id(asyncio.get_running_loop())] = mock_client

        tid = "tr_transport_01"
        tok = current_trace_id.set(tid)
        try:
            result = await t.request("alpha", "GET", "/status")
            assert result == {"status": "ok"}
            call_kwargs = mock_client.request.call_args
            headers = call_kwargs.kwargs.get("headers", {})
            assert headers.get(TRACE_HEADER) == tid
        finally:
            current_trace_id.reset(tok)

    @pytest.mark.asyncio
    async def test_http_transport_explicit_headers_override(self):
        """Explicit headers should take priority over contextvar."""
        from src.host.transport import HttpTransport

        t = HttpTransport()
        t.register("alpha", "http://localhost:8401")

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        t._clients[id(asyncio.get_running_loop())] = mock_client

        tok = current_trace_id.set("tr_should_not_use")
        try:
            await t.request(
                "alpha", "GET", "/status",
                headers={TRACE_HEADER: "tr_explicit"},
            )
            call_kwargs = mock_client.request.call_args
            headers = call_kwargs.kwargs.get("headers", {})
            assert headers.get(TRACE_HEADER) == "tr_explicit"
        finally:
            current_trace_id.reset(tok)

    def test_http_transport_sync_sends_trace_header(self):
        """request_sync should also include X-Trace-Id."""
        from src.host.transport import HttpTransport

        t = HttpTransport()
        t.register("alpha", "http://localhost:8401")

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()

        tid = "tr_sync_test_01"
        tok = current_trace_id.set(tid)
        try:
            with patch("httpx.request", return_value=mock_response) as mock_req:
                result = t.request_sync("alpha", "GET", "/status")
                assert result == {"status": "ok"}
                call_kwargs = mock_req.call_args
                headers = call_kwargs.kwargs.get("headers", {})
                assert headers.get(TRACE_HEADER) == tid
        finally:
            current_trace_id.reset(tok)
