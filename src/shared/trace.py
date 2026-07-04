"""Request tracing: trace-ID generation, contextvar propagation, header helpers.

Every user message, cron tick, and channel dispatch
gets a unique trace ID (``tr_<hex12>``) that propagates through all hops
via the ``X-Trace-Id`` HTTP header and a ``contextvars.ContextVar``.

``current_origin`` travels alongside ``current_trace_id`` and propagates
via the ``X-Origin`` HTTP header.  It carries an authorization-bearing
:class:`~src.shared.types.MessageOrigin` (``kind`` + ``channel`` + ``user``)
so downstream gates can distinguish a human-initiated wake from an
agent-initiated one.

Separate module (not utils.py) to keep tracing concerns isolated and
avoid import cycles.
"""

from __future__ import annotations

import contextvars
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

# Wire-protocol version for the mesh↔agent HTTP contract. Bumped only on a
# BREAKING change to the Pydantic contracts or endpoint semantics (additive
# fields do not bump it). Sent by the mesh on every mesh→agent hop; the agent
# server rejects a request whose major version is incompatible so a rolling
# upgrade that leaves a stale container talking to a new mesh fails loudly
# instead of silently mis-decoding JSON. A request with NO version header is
# always accepted (agent self-calls, reachability probes, first-party callers),
# so introducing the header never locks out an existing caller.
PROTOCOL_VERSION_HEADER = "X-Protocol-Version"
PROTOCOL_VERSION = "1"


def protocol_compatible(raw: str | None) -> bool:
    """True if a peer's advertised protocol version is compatible with ours.

    Compatibility is major-version equality (``"1"`` ≡ ``"1.4"``). A missing,
    empty, or whitespace-only value is treated as compatible (fail-open) so the
    header's introduction is non-breaking and unversioned first-party callers
    are never rejected. A present, non-empty value whose major differs — a real
    version we don't speak, or an unparseable one — is rejected.
    """
    if not raw:
        return True
    ours = PROTOCOL_VERSION.split(".", 1)[0]
    theirs = raw.split(".", 1)[0].strip()
    if not theirs:
        return True
    return theirs == ours

current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_trace_id", default=None,
)

current_origin: contextvars.ContextVar[
    "MessageOrigin | None"
] = contextvars.ContextVar("current_origin", default=None)

# Active durable task id, set whenever the loop is executing inside a
# task context (``execute_task`` or ``chat(task_id=...)``). Tools that
# create new work for downstream peers read this to establish the
# parent_task_id linkage so ``workflow_snapshot`` can walk a chain.
# Defaults to ``None`` outside any task context (heartbeats, free chat).
current_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_task_id", default=None,
)

# Identity of the process/agent emitting work. Set once at agent boot
# (``src/agent/__main__.py``) so every log line in an agent container can
# be attributed without threading the id through. The mesh host process
# leaves it ``None`` (host log lines carry the agent in ``extra_data``
# instead). Phase 4 log correlation reads this alongside the trace/task
# ids; the structured formatter additionally falls back to the ``AGENT_ID``
# env var so per-request contexts that never inherited the boot ``set()``
# are still attributed.
current_agent_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_agent_id", default=None,
)


def new_trace_id() -> str:
    """Generate a fresh trace ID: ``tr_<12-hex-chars>``."""
    return f"tr_{uuid.uuid4().hex[:12]}"


def trace_headers() -> dict[str, str]:
    """Return headers dict with ``X-Trace-Id`` if one is active."""
    tid = current_trace_id.get()
    return {TRACE_HEADER: tid} if tid else {}


def origin_header(origin: "MessageOrigin | None") -> dict[str, str]:
    """Serialize an origin to an ``X-Origin`` header dict.

    Returns ``{}`` when origin is empty so it can be spread into a
    headers dict unconditionally.
    """
    if isinstance(origin, MessageOrigin):
        return {ORIGIN_HEADER: origin.to_header_value()}
    return {}


def parse_origin_header(raw: str | None) -> "MessageOrigin | None":
    """Parse an X-Origin header.

    Returns a typed :class:`MessageOrigin` on success, ``None`` on any
    error or invalid shape (never a partial model). Request headers are
    parsed as untrusted by default: any caller-supplied kind is downgraded
    to ``kind="agent"`` so downstream auth gates cannot accept forged
    human/operator/system claims from raw headers.

    Field length bounds: ``channel`` ≤ 32 chars, ``user`` ≤ 128 chars,
    raw blob ≤ 512 chars. Legacy headers (no ``kind``) still require
    non-empty ``channel`` and ``user`` to match the pre-typed-origin parser
    contract.
    """
    return MessageOrigin.from_header_value(raw, trust_kind=False)


# Re-export the field-length caps so existing imports keep working.
__all__ = [
    "TRACE_HEADER",
    "ORIGIN_HEADER",
    "PROTOCOL_VERSION_HEADER",
    "PROTOCOL_VERSION",
    "protocol_compatible",
    "current_trace_id",
    "current_origin",
    "current_task_id",
    "current_agent_id",
    "new_trace_id",
    "trace_headers",
    "origin_header",
    "parse_origin_header",
    "_MAX_ORIGIN_CHANNEL_LEN",
    "_MAX_ORIGIN_USER_LEN",
]
