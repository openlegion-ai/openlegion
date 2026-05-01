"""Request tracing: trace-ID generation, contextvar propagation, header helpers.

Every user message, cron tick, and channel dispatch
gets a unique trace ID (``tr_<hex12>``) that propagates through all hops
via the ``X-Trace-Id`` HTTP header and a ``contextvars.ContextVar``.

``current_origin`` travels alongside ``current_trace_id`` and propagates
via the ``X-Origin`` HTTP header.  It carries an authorization-bearing
:class:`~src.shared.types.MessageOrigin` (``kind`` + ``channel`` + ``user``)
so downstream gates can distinguish a human-initiated wake from an
agent-initiated one.

Backward-compat (Task 2a → 2b): until stamp sites are migrated, the
contextvar may still hold a raw ``{"channel": ..., "user": ...}`` dict.
The helpers in this module accept either shape; readers can use
attribute access (``origin.channel``) on a typed origin or fall back to
dict-style access (``origin.get("channel")``) which works on both via
:class:`MessageOrigin`'s dict-compat shim.

Separate module (not utils.py) to keep tracing concerns isolated and
avoid import cycles.
"""

from __future__ import annotations

import contextvars
import json
import uuid

from src.shared.types import (
    _MAX_ORIGIN_CHANNEL_LEN,
    _MAX_ORIGIN_USER_LEN,
    MessageOrigin,
)
from src.shared.utils import setup_logging

logger = setup_logging("shared.trace")

TRACE_HEADER = "X-Trace-Id"
ORIGIN_HEADER = "X-Origin"

current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_trace_id", default=None,
)

# Accepts either a typed :class:`MessageOrigin` (preferred) or a raw
# ``{"channel": ..., "user": ...}`` dict (legacy stamp sites being
# migrated in Task 2b). Readers should use attribute access on the
# typed model or ``.get()`` (which works on both shapes).
current_origin: contextvars.ContextVar[
    "MessageOrigin | dict[str, str] | None"
] = contextvars.ContextVar("current_origin", default=None)


def new_trace_id() -> str:
    """Generate a fresh trace ID: ``tr_<12-hex-chars>``."""
    return f"tr_{uuid.uuid4().hex[:12]}"


def trace_headers() -> dict[str, str]:
    """Return headers dict with ``X-Trace-Id`` if one is active."""
    tid = current_trace_id.get()
    return {TRACE_HEADER: tid} if tid else {}


def origin_header(
    origin: "MessageOrigin | dict[str, str] | None",
) -> dict[str, str]:
    """Serialize an origin to an ``X-Origin`` header dict.

    Accepts either a typed :class:`MessageOrigin` or a legacy
    ``{"channel": ..., "user": ...}`` dict. Returns ``{}`` when origin
    is empty so it can be spread into a headers dict unconditionally.

    Dict input without a ``kind`` field is upgraded to ``kind="agent"``
    (least-trusted) before serialization so the wire format always
    carries an explicit kind for the next hop.
    """
    if not origin:
        return {}
    if isinstance(origin, MessageOrigin):
        return {ORIGIN_HEADER: origin.to_header_value()}
    # Legacy dict — emit with ``kind="agent"`` if not provided.
    if not isinstance(origin, dict):
        return {}
    # Task 2a deprecation breadcrumb: stamp sites still emitting raw
    # dicts will be migrated in Task 2b. Logging at debug keeps the
    # signal available for the migration audit without spamming logs.
    logger.debug(
        "origin_header received legacy dict (kind=%s); migrate stamp site to MessageOrigin",
        origin.get("kind", "agent"),
    )
    payload = {
        "kind": origin.get("kind", "agent"),
        "channel": origin.get("channel", ""),
        "user": origin.get("user", ""),
    }
    return {ORIGIN_HEADER: json.dumps(payload, separators=(",", ":"))}


def parse_origin_header(raw: str | None) -> "MessageOrigin | None":
    """Parse an X-Origin header.

    Returns a typed :class:`MessageOrigin` on success, ``None`` on any
    error or invalid shape (never a partial model). Request headers are
    parsed as untrusted by default: any caller-supplied kind is downgraded
    to ``kind="agent"`` so downstream auth gates cannot accept forged
    human/operator/system claims from raw headers.

    Field length bounds: ``channel`` ≤ 32 chars, ``user`` ≤ 128 chars,
    raw blob ≤ 512 chars. Legacy headers (no ``kind``) still require
    non-empty ``channel`` and ``user`` to match the pre-Task-2a parser
    contract.
    """
    return MessageOrigin.from_header_value(raw, trust_kind=False)


# Re-export the field-length caps so existing imports keep working.
__all__ = [
    "TRACE_HEADER",
    "ORIGIN_HEADER",
    "current_trace_id",
    "current_origin",
    "new_trace_id",
    "trace_headers",
    "origin_header",
    "parse_origin_header",
    "_MAX_ORIGIN_CHANNEL_LEN",
    "_MAX_ORIGIN_USER_LEN",
]
