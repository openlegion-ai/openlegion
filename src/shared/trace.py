"""Request tracing: trace-ID generation, contextvar propagation, header helpers.

Every user message, cron tick, channel dispatch, and orchestrator step
gets a unique trace ID (``tr_<hex12>``) that propagates through all hops
via the ``X-Trace-Id`` HTTP header and a ``contextvars.ContextVar``.

Separate module (not utils.py) to keep tracing concerns isolated and
avoid import cycles.
"""

from __future__ import annotations

import contextvars
import uuid

TRACE_HEADER = "X-Trace-Id"

current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_trace_id", default=None,
)


def new_trace_id() -> str:
    """Generate a fresh trace ID: ``tr_<12-hex-chars>``."""
    return f"tr_{uuid.uuid4().hex[:12]}"


def trace_headers() -> dict[str, str]:
    """Return headers dict with ``X-Trace-Id`` if one is active."""
    tid = current_trace_id.get()
    return {TRACE_HEADER: tid} if tid else {}
