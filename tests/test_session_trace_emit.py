"""Phase 4 session-observability tests: agent-side trace emission +
structured-log correlation.

Covers the *agent* end of the trace pipeline (the mesh ingest endpoint is
tested in ``test_orchestration_endpoints.py``):

* ``MeshClient.record_trace`` — no-ops without an active trace context, POSTs
  to ``/mesh/traces`` under it, and swallows transport errors (best-effort).
* ``StructuredFormatter`` / ``TextFormatter`` — inject ``trace_id`` /
  ``task_id`` / ``agent_id`` correlation IDs, with ``extra_data`` winning and
  the ``AGENT_ID`` env var as the agent-id fallback.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.mesh_client import MeshClient
from src.shared.trace import current_agent_id, current_task_id, current_trace_id
from src.shared.utils import StructuredFormatter, TextFormatter


def _mk_client_with_fake_post():
    mc = MeshClient("http://mesh:8420", "worker")
    fake = MagicMock()
    fake.post = AsyncMock()
    mc._get_client = AsyncMock(return_value=fake)
    return mc, fake


@pytest.mark.asyncio
async def test_record_trace_noops_without_trace_context():
    """No active trace → nothing to correlate → no POST is made."""
    mc, fake = _mk_client_with_fake_post()
    tok = current_trace_id.set(None)
    try:
        await mc.record_trace("tool_call", detail="x")
    finally:
        current_trace_id.reset(tok)
    fake.post.assert_not_called()


@pytest.mark.asyncio
async def test_record_trace_posts_under_trace_context():
    """With an active trace, POST /mesh/traces carries the event + the
    X-Trace-Id header so the mesh records under the originating trace."""
    mc, fake = _mk_client_with_fake_post()
    tok = current_trace_id.set("tr_emit0000001")
    try:
        await mc.record_trace(
            "tool_call", detail="http_request", duration_ms=12,
            status="ok", meta={"a": 1},
        )
    finally:
        current_trace_id.reset(tok)
    fake.post.assert_awaited_once()
    args, kwargs = fake.post.call_args
    assert args[0].endswith("/mesh/traces")
    body = kwargs["json"]
    assert body["event_type"] == "tool_call"
    assert body["agent_id"] == "worker"
    assert body["detail"] == "http_request"
    assert body["duration_ms"] == 12
    assert body["meta"] == {"a": 1}
    assert kwargs["headers"].get("X-Trace-Id") == "tr_emit0000001"


@pytest.mark.asyncio
async def test_record_trace_swallows_post_errors():
    """Transport failure must never propagate — tracing is best-effort."""
    mc = MeshClient("http://mesh:8420", "worker")
    fake = MagicMock()
    fake.post = AsyncMock(side_effect=RuntimeError("boom"))
    mc._get_client = AsyncMock(return_value=fake)
    tok = current_trace_id.set("tr_emit0000002")
    try:
        await mc.record_trace("tool_call")  # must not raise
    finally:
        current_trace_id.reset(tok)


def _rec(msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord("mod", logging.INFO, "f.py", 1, msg, None, None)


def _reset_ctx():
    """Pin all correlation contextvars to None and return reset tokens."""
    return (
        current_trace_id.set(None),
        current_task_id.set(None),
        current_agent_id.set(None),
    )


def test_structured_formatter_omits_ids_without_context(monkeypatch):
    monkeypatch.delenv("AGENT_ID", raising=False)
    toks = _reset_ctx()
    try:
        out = json.loads(StructuredFormatter().format(_rec()))
    finally:
        current_trace_id.reset(toks[0])
        current_task_id.reset(toks[1])
        current_agent_id.reset(toks[2])
    assert "trace_id" not in out
    assert "task_id" not in out
    assert "agent_id" not in out


def test_structured_formatter_injects_correlation_ids(monkeypatch):
    monkeypatch.delenv("AGENT_ID", raising=False)
    t1 = current_trace_id.set("tr_log00000001")
    t2 = current_task_id.set("task_log_1")
    t3 = current_agent_id.set("worker7")
    try:
        out = json.loads(StructuredFormatter().format(_rec()))
    finally:
        current_trace_id.reset(t1)
        current_task_id.reset(t2)
        current_agent_id.reset(t3)
    assert out["trace_id"] == "tr_log00000001"
    assert out["task_id"] == "task_log_1"
    assert out["agent_id"] == "worker7"


def test_structured_formatter_extra_data_wins(monkeypatch):
    monkeypatch.delenv("AGENT_ID", raising=False)
    t1 = current_trace_id.set("tr_ambient")
    try:
        rec = _rec()
        rec.extra_data = {"trace_id": "tr_explicit", "extra": 1}
        out = json.loads(StructuredFormatter().format(rec))
    finally:
        current_trace_id.reset(t1)
    assert out["trace_id"] == "tr_explicit"
    assert out["extra"] == 1


def test_structured_formatter_agent_id_falls_back_to_env(monkeypatch):
    """Per-request contexts may never inherit the boot ``set()``; the env
    var keeps every agent-container line attributed."""
    monkeypatch.setenv("AGENT_ID", "env_worker")
    toks = _reset_ctx()  # current_agent_id explicitly None
    try:
        out = json.loads(StructuredFormatter().format(_rec()))
    finally:
        current_trace_id.reset(toks[0])
        current_task_id.reset(toks[1])
        current_agent_id.reset(toks[2])
    assert out["agent_id"] == "env_worker"


def test_text_formatter_appends_correlation_suffix(monkeypatch):
    monkeypatch.delenv("AGENT_ID", raising=False)
    t1 = current_trace_id.set("tr_text00000001")
    t3 = current_agent_id.set("worker9")
    try:
        line = TextFormatter().format(_rec("doing work"))
    finally:
        current_trace_id.reset(t1)
        current_agent_id.reset(t3)
    assert "doing work" in line
    assert "trace_id=tr_text00000001" in line
    assert "agent_id=worker9" in line
