"""Single-use, TTL-bound OAuth state store for the connect/callback dance.

The ``state`` parameter is the CSRF guard for the OAuth redirect flow. Each
entry is:

* unguessable — ``secrets.token_urlsafe`` (256 bits),
* single-use — consumed on first read,
* TTL-bound — default 10 minutes,
* session-bound — tied to a hash of the caller's dashboard session cookie so a
  state minted in one browser can't be redeemed from another.

In-memory by design. A process restart mid-flow simply means the user re-clicks
"Connect" (the whole dance lasts seconds). This is NOT a module-level global —
one instance is created per dashboard router and closed over, so it carries no
cross-process or cross-test leakage (cf. CLAUDE.md constraint #8).
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field


@dataclass
class PendingAuth:
    provider: str
    connection_name: str
    scopes: tuple[str, ...]
    code_verifier: str
    redirect_uri: str
    session_hash: str
    expires_at: float
    # Flow-specific context that must survive the redirect round-trip.
    # MCP connectors (dynamic providers) stash the DISCOVERED token
    # endpoint + DCR client identity here — there is no registry entry
    # to re-derive them from at callback time. May contain a client
    # secret: never log this field.
    extra: dict = field(default_factory=dict)


class OAuthStateStore:
    def __init__(self, ttl_seconds: int = 600, max_entries: int = 512) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._lock = threading.Lock()
        self._pending: dict[str, PendingAuth] = {}

    def create(
        self,
        *,
        provider: str,
        connection_name: str,
        scopes: tuple[str, ...],
        code_verifier: str,
        redirect_uri: str,
        session_hash: str,
        extra: dict | None = None,
        now: float | None = None,
    ) -> str:
        """Mint a new state token and record the pending auth. Returns the token."""
        ts = time.time() if now is None else now
        state = secrets.token_urlsafe(32)
        entry = PendingAuth(
            provider=provider,
            connection_name=connection_name,
            scopes=tuple(scopes),
            code_verifier=code_verifier,
            redirect_uri=redirect_uri,
            session_hash=session_hash,
            expires_at=ts + self._ttl,
            extra=dict(extra or {}),
        )
        with self._lock:
            self._reap(ts)
            # Bound memory: if somehow over cap, drop the soonest-to-expire.
            if len(self._pending) >= self._max:
                oldest = min(self._pending, key=lambda k: self._pending[k].expires_at)
                self._pending.pop(oldest, None)
            self._pending[state] = entry
        return state

    def consume(
        self, state: str, *, session_hash: str, now: float | None = None,
    ) -> PendingAuth | None:
        """Validate + remove a state token.

        Returns the :class:`PendingAuth` only if the token exists, has not
        expired, and the session hash matches. Always single-use: the token is
        removed whether or not validation succeeds (so a leaked/expired token
        can't be retried). Returns ``None`` on any failure.
        """
        ts = time.time() if now is None else now
        with self._lock:
            entry = self._pending.pop(state, None)
            self._reap(ts)
        if entry is None:
            return None
        if entry.expires_at < ts:
            return None
        if not secrets.compare_digest(entry.session_hash, session_hash):
            return None
        return entry

    def _reap(self, ts: float) -> None:
        """Drop expired entries. Caller holds the lock."""
        expired = [k for k, v in self._pending.items() if v.expires_at < ts]
        for k in expired:
            self._pending.pop(k, None)
