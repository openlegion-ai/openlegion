"""Request tracing: trace-ID generation, contextvar propagation, header helpers.

Every user message, cron tick, and channel dispatch
gets a unique trace ID (``tr_<hex12>``) that propagates through all hops
via the ``X-Trace-Id`` HTTP header and a ``contextvars.ContextVar``.

``current_origin`` travels alongside ``current_trace_id`` and propagates
via the ``X-Origin`` HTTP header.  It carries ``{"channel": ..., "user": ...}``
so lane workers can auto-forward task results back to the originating user
on their messaging channel after a hand-off completes.

Separate module (not utils.py) to keep tracing concerns isolated and
avoid import cycles.
"""

from __future__ import annotations

import contextvars
import json
import uuid

TRACE_HEADER = "X-Trace-Id"
ORIGIN_HEADER = "X-Origin"

current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_trace_id", default=None,
)

current_origin: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "current_origin", default=None,
)


def new_trace_id() -> str:
    """Generate a fresh trace ID: ``tr_<12-hex-chars>``."""
    return f"tr_{uuid.uuid4().hex[:12]}"


def trace_headers() -> dict[str, str]:
    """Return headers dict with ``X-Trace-Id`` if one is active."""
    tid = current_trace_id.get()
    return {TRACE_HEADER: tid} if tid else {}


def origin_header(origin: dict[str, str] | None) -> dict[str, str]:
    """Serialize an origin dict to an ``X-Origin`` header dict.

    Returns ``{}`` when origin is empty so it can be spread into a headers
    dict unconditionally.
    """
    if not origin:
        return {}
    return {ORIGIN_HEADER: json.dumps(origin, separators=(",", ":"))}


_MAX_ORIGIN_CHANNEL_LEN = 32
_MAX_ORIGIN_USER_LEN = 128


def parse_origin_header(raw: str | None) -> dict[str, str] | None:
    """Parse an X-Origin header. Returns None on any error or invalid shape.

    Only ``channel`` and ``user`` fields propagate — all other keys are dropped.
    Both must be non-empty strings within reasonable length bounds so a
    malicious or malformed header cannot inflate downstream payloads.
    """
    if not raw or len(raw) > 512:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    ch = parsed.get("channel")
    us = parsed.get("user")
    if not isinstance(ch, str) or not isinstance(us, str):
        return None
    if not ch or not us:
        return None
    if len(ch) > _MAX_ORIGIN_CHANNEL_LEN or len(us) > _MAX_ORIGIN_USER_LEN:
        return None
    # Whitelist: only `channel` and `user` fields propagate.
    return {"channel": ch, "user": us}
