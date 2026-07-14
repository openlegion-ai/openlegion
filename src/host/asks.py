"""AskBroker — mesh-held registry for inline teammate Q&A (``ask_teammate``).

Phase 2 unit 3 (plan §6: "mesh-mediated, loads recipient expertise,
bypasses the deep-work lane, returns inline, rate-limited, billed to
asker"). The broker owns the live state of every in-flight ask:

* ``/mesh/ask`` creates a record and awaits its future (answer or
  timeout envelope).
* Delivery is busy/idle aware — a BUSY recipient gets a steer
  interjection (no task row, no task_id → no auto-close, Constraint #6
  preserved; never a second parallel turn, plan B1), an IDLE recipient
  gets a normal followup lane turn whose own response text doubles as
  the answer fallback.
* ``answer_ask`` (``/mesh/ask/{id}/answer``) resolves the future —
  single-use, verified recipient only.
* The billing window bills the recipient's LLM spend to the ASKER
  while active (see ``credentials.set_bill_resolver``). The window is
  keyed EXCLUSIVELY off this mesh-held state — never off anything a
  container sends — so a malicious recipient cannot extend or redirect
  billing and a malicious asker is bounded by rate limit × cap.

In-memory is DELIBERATE: an ask is a live RPC (seconds to minutes).
A mesh restart drops the registry; the asker's pending HTTP call dies
with the process and the tool surfaces a failure envelope — nothing
durable is lost because the Q&A is also posted to the team thread
store when one is wired.

Thread-safety: records are touched from multiple event loops — the
uvicorn mesh loop (create/answer/billing checks via the LLM proxy) and
the lane dispatch loop (delivery callbacks). State mutations therefore
serialize on a ``threading.Lock`` (held only for dict work; the spec's
"asyncio.Lock" would be loop-bound and cannot serialize cross-loop
callers), and future resolution always hops to the future's owning
loop via ``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

from src.shared import limits
from src.shared.utils import generate_id, setup_logging

logger = setup_logging("host.asks")


class AskLimitExceeded(Exception):
    """Raised by ``create`` when a concurrency cap is hit (mapped to 429)."""


class AskDeliveryFailed(Exception):
    """Set on the ask future when delivery to the recipient failed."""


@dataclass
class AskRecord:
    ask_id: str
    asker: str
    recipient: str
    question: str
    timeout_seconds: int
    created_at: float
    deadline: float
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    scope_id: str | None = None
    # "busy" (steer-injected) | "idle" (followup turn) | None (undelivered)
    path: str | None = None
    billing_active: bool = False
    billed_usd: float = 0.0
    finished: bool = False
    answered_by: str | None = None
    thread_id: object = None
    extras: dict = field(default_factory=dict)


class AskBroker:
    """In-memory ask registry. One instance per mesh app (no globals)."""

    MAX_ACTIVE_PER_ASKER = 3
    MAX_ACTIVE_GLOBAL = 100
    # Grace after future resolution during which the recipient's already
    # in-flight LLM call still bills the asker. Class attribute so tests
    # can tighten it.
    BILLING_GRACE_SECONDS = 5.0

    def __init__(self, *, thread_store=None, bill_cap_usd: float | None = None):
        self._lock = threading.Lock()
        self._asks: dict[str, AskRecord] = {}
        # Unit-2 (Team Threads) integration is OPTIONAL wiring: ``None``
        # skips thread posting silently. Wired post-construction via
        # ``set_thread_store`` — never imported statically so this module
        # stays green without threads.py in the tree. Agreed unit-2 API:
        #   ensure_dm_thread(scope_id, agent_a, agent_b) -> {"id": ...}
        #   post_message(thread_id, sender, *, body=None, kind="message",
        #                recipient=None, payload=None)
        self._thread_store = thread_store
        # ``None`` → resolve ``limits.ask_bill_cap_usd()`` per accrual so
        # env changes apply without a restart; a float pins it (tests).
        self._bill_cap_usd = bill_cap_usd

    def set_thread_store(self, store) -> None:
        """Wire the Team Threads store after construction (unit 2)."""
        self._thread_store = store

    # ── lifecycle ────────────────────────────────────────────────────

    def create(
        self,
        asker: str,
        recipient: str,
        question: str,
        timeout_seconds: int,
        *,
        scope_id: str | None = None,
    ) -> AskRecord:
        """Register a new ask. Must be called on a running event loop
        (the future is bound to it). Raises ``AskLimitExceeded`` when the
        per-asker or global concurrency cap is hit."""
        loop = asyncio.get_running_loop()
        now = time.time()
        record = AskRecord(
            ask_id=generate_id("ask"),
            asker=asker,
            recipient=recipient,
            question=question,
            timeout_seconds=timeout_seconds,
            created_at=now,
            deadline=now + timeout_seconds,
            future=loop.create_future(),
            loop=loop,
            scope_id=scope_id,
        )
        with self._lock:
            active = [r for r in self._asks.values() if not r.finished]
            if len(active) >= self.MAX_ACTIVE_GLOBAL:
                raise AskLimitExceeded(
                    f"global ask limit reached ({self.MAX_ACTIVE_GLOBAL} in flight)"
                )
            mine = sum(1 for r in active if r.asker == asker)
            if mine >= self.MAX_ACTIVE_PER_ASKER:
                raise AskLimitExceeded(
                    f"agent '{asker}' already has {mine} asks in flight "
                    f"(max {self.MAX_ACTIVE_PER_ASKER})"
                )
            self._asks[record.ask_id] = record
        # Durable + human-visible even if the RPC times out.
        self._post_thread(
            record, sender=asker, body=question, recipient=recipient,
        )
        return record

    def get(self, ask_id: str) -> AskRecord | None:
        with self._lock:
            return self._asks.get(ask_id)

    def mark_path(self, ask_id: str, path: str) -> None:
        with self._lock:
            record = self._asks.get(ask_id)
            if record is not None:
                record.path = path

    def has_open_asks(self, agent_id: str) -> bool:
        """Whether ``agent_id`` is a party (asker or recipient) to any
        in-flight ask. Used by the hibernation sweep (plan §8 #24) — an
        agent waiting on an answer, or one that owes an answer, is not a
        hibernation candidate even if its lane is otherwise idle."""
        with self._lock:
            return any(
                not r.finished and (r.asker == agent_id or r.recipient == agent_id)
                for r in self._asks.values()
            )

    # ── resolution ───────────────────────────────────────────────────

    def resolve(self, ask_id: str, answer: str, by: str) -> dict:
        """Recipient-verified, single-use answer delivery.

        Returns ``{"ok": True}`` or ``{"ok": False, "reason": ...}`` with
        reason ∈ {unknown, wrong_recipient, already_resolved}.
        """
        with self._lock:
            record = self._asks.get(ask_id)
            if record is None:
                return {"ok": False, "reason": "unknown"}
            if by != record.recipient:
                logger.warning(
                    "answer_ask rejected: %s tried to answer ask %s "
                    "addressed to %s", by, ask_id, record.recipient,
                )
                return {"ok": False, "reason": "wrong_recipient"}
            if record.finished or record.future.done():
                return {"ok": False, "reason": "already_resolved"}
            record.answered_by = by
        if not self._set_future(record, result=answer):
            return {"ok": False, "reason": "already_resolved"}
        self._post_thread(
            record, sender=by, body=answer, recipient=record.asker,
        )
        return {"ok": True}

    def resolve_inline(self, ask_id: str, answer: str) -> bool:
        """Idle-path fallback: resolve with the recipient turn's own
        response text. Mesh-internal (the delivery coroutine) — no
        recipient check. No-op when answer_ask already resolved it."""
        with self._lock:
            record = self._asks.get(ask_id)
            if record is None or record.finished or record.future.done():
                return False
            record.answered_by = record.recipient
        if not self._set_future(record, result=answer):
            return False
        self._post_thread(
            record, sender=record.recipient, body=answer,
            recipient=record.asker,
        )
        return True

    def fail(self, ask_id: str, reason: str) -> None:
        """Fail the ask (delivery error). No-op if already resolved."""
        with self._lock:
            record = self._asks.get(ask_id)
        if record is None:
            return
        self._set_future(record, exception=AskDeliveryFailed(reason))

    def _set_future(
        self, record: AskRecord, *,
        result: str | None = None,
        exception: BaseException | None = None,
    ) -> bool:
        """Resolve the future on ITS OWN loop (safe from any thread).

        Returns False when the future was already done/cancelled or the
        owning loop is gone (best-effort — the endpoint's wait_for owns
        the timeout path either way).
        """

        def _apply() -> None:
            fut = record.future
            if fut.done() or fut.cancelled():
                return
            if exception is not None:
                fut.set_exception(exception)
            else:
                fut.set_result(result)

        if record.future.done() or record.future.cancelled():
            return False
        try:
            record.loop.call_soon_threadsafe(_apply)
            return True
        except RuntimeError:  # owning loop closed (shutdown / tests)
            return False

    # ── billing window (mesh-authoritative) ──────────────────────────

    def activate_billing(self, ask_id: str) -> bool:
        """Open the asker-pays window. Called from the lane dispatch
        wrapper when the IDLE-path turn actually starts (never at
        enqueue — queued unrelated work must not bill the asker).

        Guards the single-window-per-recipient invariant: the lane is
        serial per agent so a second concurrent window for the same
        recipient should be impossible — refuse and log loudly if one
        ever shows up rather than double-billing.
        """
        with self._lock:
            record = self._asks.get(ask_id)
            if record is None or record.finished:
                return False
            for other in self._asks.values():
                if (
                    other.ask_id != ask_id
                    and other.billing_active
                    and other.recipient == record.recipient
                ):
                    logger.error(
                        "ask billing invariant violated: recipient %s "
                        "already has an active window (ask %s) — refusing "
                        "to open a second for ask %s",
                        record.recipient, other.ask_id, ask_id,
                    )
                    return False
            record.billing_active = True
        logger.info(
            "ask %s billing window OPEN: %s's turn bills asker %s",
            ask_id, record.recipient, record.asker,
        )
        return True

    def active_billing_for(self, agent_id: str) -> str | None:
        """The asker to bill for ``agent_id``'s LLM spend right now, or
        ``None``. ``agent_id`` is the MESH-VERIFIED proxy caller — the
        mapping lives entirely in mesh-held state (never headers)."""
        with self._lock:
            for record in self._asks.values():
                if record.billing_active and record.recipient == agent_id:
                    return record.asker
        return None

    def note_billed_cost(self, agent_id: str, cost_usd: float) -> None:
        """Accrue an asker-billed cost against ``agent_id``'s active
        window; close the window once the per-ask cap is crossed."""
        if not cost_usd or cost_usd <= 0:
            return
        cap = (
            self._bill_cap_usd
            if self._bill_cap_usd is not None
            else limits.ask_bill_cap_usd()
        )
        with self._lock:
            for record in self._asks.values():
                if record.billing_active and record.recipient == agent_id:
                    record.billed_usd += cost_usd
                    if record.billed_usd >= cap:
                        record.billing_active = False
                        logger.warning(
                            "ask %s billed cap hit ($%.4f >= $%.2f) — "
                            "window CLOSED; %s's subsequent calls bill "
                            "themselves",
                            record.ask_id, record.billed_usd, cap,
                            record.recipient,
                        )
                    return

    def finish(self, ask_id: str) -> None:
        """Endpoint 'finally' hook: the asker's wait ended (answer,
        timeout, or failure). Closes the billing window after the grace
        period and drops the record. Late ``answer_ask`` calls after
        this see the unknown-ask envelope (non-fatal by design)."""
        with self._lock:
            record = self._asks.get(ask_id)
            if record is None:
                return
            record.finished = True
            had_window = record.billing_active

        def _close() -> None:
            with self._lock:
                rec = self._asks.pop(ask_id, None)
                if rec is not None and rec.billing_active:
                    rec.billing_active = False
                    logger.info("ask %s billing window closed (grace over)", ask_id)

        if not had_window:
            _close()
            return
        try:
            record.loop.call_soon_threadsafe(
                lambda: record.loop.call_later(self.BILLING_GRACE_SECONDS, _close)
            )
        except RuntimeError:
            _close()

    # ── team-thread posting (optional unit-2 wiring) ─────────────────

    def _post_thread(
        self, record: AskRecord, *, sender: str, body: str, recipient: str,
    ) -> None:
        """Best-effort post to the pair's DM thread. Silent no-op when no
        store is wired; a store error never breaks the ask RPC."""
        store = self._thread_store
        if store is None:
            return
        try:
            if record.thread_id is None:
                record.thread_id = store.ensure_dm_thread(
                    record.scope_id, record.asker, record.recipient,
                )["id"]
            store.post_message(
                record.thread_id, sender, body=body,
                kind="message", recipient=recipient,
                payload={"ask_id": record.ask_id},
            )
        except Exception as e:
            logger.warning(
                "ask %s thread post failed (non-fatal): %s",
                record.ask_id, e,
            )
