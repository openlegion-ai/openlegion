"""Phase 10 §20 — per-agent BrowserContext storage-state persistence.

Without persistence the agent loses cookies, ``localStorage``,
``sessionStorage``, and IndexedDB on container restart or worker
reschedule. Any workflow that requires staying logged in for more
than one task either re-authenticates (slow, often blocked by 2FA /
captcha) or stops working entirely.

This module captures Playwright's ``BrowserContext.storage_state()``
to a per-agent JSON sidecar at ``data/sessions/<agent_id>.json``,
restores it on next launch, and exposes an operator-controlled clear
path. Lifecycle wiring lives in ``src/browser/service.py``; the flag
gate lives in ``src/browser/flags.py``.

──────────────── Privacy posture ────────────────

The sidecars contain session tokens (cookies + ``localStorage``) that,
if leaked, allow account takeover on whatever site the agent was logged
into. Three controls keep the blast radius small:

1. **Default-off.** ``BROWSER_SESSION_PERSISTENCE_ENABLED`` defaults
   to ``False``. Operators must opt in deliberately.
2. **chmod 0o600.** Files are written with owner-only read/write
   from the moment they exist on disk — there is NO world-readable
   window between create and chmod. This matches the captcha-cost
   counter's posture and the ``.agent.env`` file's chmod.
3. **No domain leakage.** The dashboard summary endpoint returns
   counts only — no cookie values, no origin names. Audit events
   sent to the EventBus carry the same shape (count + bool, no
   origins). Operators have to inspect the JSON file directly to
   see specific origins, and that file is chmod 0600.

──────────────── Why a separate module ────────────────

We could inline this into ``service.py``, but the captcha-cost counter
sets the precedent that persistence concerns get their own module
(easier to test in isolation; clearer surface for the lifecycle
hook). The atomic-write protocol below is adapted from
``captcha_cost_counter.snapshot`` — same approach (open with mode
``0o600``, fsync, ``os.replace``), different payload.

──────────────── On-disk schema ────────────────

::

    {
      "version": 1,
      "saved_at": <epoch seconds>,
      "storage_state": {
          "cookies": [...],
          "origins": [{"origin": "...", "localStorage": [...]}, ...]
      }
    }

``version`` is checked on restore. Future schema bumps fall through
to a fresh state (with a warning log) so an older browser-service
build can't crash by reading a newer file. Operators own rotation /
expiry — there is no time-based expiry built in. A file produced
six months ago restores fine; whether the embedded session tokens
are still valid is a question for the upstream site.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from src.shared.types import AGENT_ID_RE_PATTERN
from src.shared.utils import setup_logging

logger = setup_logging("browser.session_persistence")


# Default storage location. Override-able via env for tests + custom
# deployments. The directory is created lazily on first snapshot;
# missing dir on restore is a non-fatal no-op (returns None).
_DEFAULT_DIR = "data/sessions"
_SCHEMA_VERSION = 1
# Hard cap on serialized snapshot size. localStorage can be ~5 MiB per
# origin under spec; a malicious site can inflate one origin's quota by
# writing junk. Cap the whole sidecar at 8 MiB so a hostile workflow
# can't blow out the disk via persisted state.
_MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
# Reuse the canonical agent-id regex — sidecars are written under
# ``<dir>/<agent_id>.json`` and an unvalidated agent_id is a directory
# traversal vector (``../../etc/passwd``).
_AGENT_ID_RE = re.compile(AGENT_ID_RE_PATTERN)


class InvalidAgentIdError(ValueError):
    """Raised when ``agent_id`` does not match ``AGENT_ID_RE_PATTERN``.

    Distinct from generic ``ValueError`` so the caller can map this
    to HTTP 400 (caller error) instead of 500 (server bug).
    """


def _validate_agent_id(agent_id: str) -> None:
    """Reject any agent_id that does not match the canonical regex.

    The regex (``shared.types.AGENT_ID_RE_PATTERN``) restricts agent
    ids to alnum + ``_-`` so they are safe filename components. This
    closes the path-traversal vector at every persistence entry point
    rather than depending on each caller to validate.
    """
    if not isinstance(agent_id, str) or not _AGENT_ID_RE.fullmatch(agent_id):
        raise InvalidAgentIdError(f"invalid agent_id: {agent_id!r}")


def _sessions_dir() -> Path:
    """Return the per-service sessions directory.

    Resolved lazily so tests can monkeypatch the env var between
    cases. A ``data/`` sibling of the captcha-cost counter sidecar
    keeps all browser-service persistence under one parent.
    """
    return Path(os.environ.get("BROWSER_SESSION_DIR", _DEFAULT_DIR))


def session_path(agent_id: str) -> Path:
    """Return the canonical sidecar path for ``agent_id``.

    Pure path construction — does NOT touch the filesystem. Callers
    that want to know whether a session exists should call
    :func:`session_path(agent_id).exists()`.

    Raises ``InvalidAgentIdError`` if ``agent_id`` would result in a
    path-traversal-vulnerable filename (anything not matching
    ``AGENT_ID_RE_PATTERN``). This is the single chokepoint — every
    snapshot/restore/clear path goes through this function.
    """
    _validate_agent_id(agent_id)
    return _sessions_dir() / f"{agent_id}.json"


# ── Snapshot ───────────────────────────────────────────────────────────────


async def snapshot_session(agent_id: str, context: Any) -> bool:
    """Capture ``context.storage_state()`` to the per-agent sidecar.

    ``context`` is duck-typed as a Playwright ``BrowserContext``; only
    ``storage_state()`` is called. Returns ``True`` on success;
    ``False`` on any failure (logged).

    The shutdown path must NOT crash because session persistence
    failed — same posture as the captcha-cost counter snapshot
    (a noisy log + ``False`` is the contract). A late-arriving
    ``Shutdown`` from systemd or Docker is plenty common; raising
    here would orphan the browser process and leak state.

    Atomic-write protocol (same as captcha_cost_counter):

      1. ``os.open`` the tmp file with ``O_WRONLY|O_CREAT|O_TRUNC``
         and **mode 0o600** so there is no world-readable window
         before the explicit chmod. Sensitive data: cookies +
         ``localStorage`` values are session credentials.
      2. Write the JSON payload, ``flush()``, ``fsync()``.
      3. ``os.replace(tmp, path)`` — atomic rename.
      4. Re-``chmod`` the destination to 0o600. Some filesystems
         preserve the destination's prior mode across ``replace``;
         the explicit chmod after replace is belt-and-suspenders so
         operators can rely on the mode regardless of whether the
         target pre-existed.
    """
    try:
        target = session_path(agent_id)
    except InvalidAgentIdError as e:
        logger.warning("session snapshot rejected: %s", e)
        return False
    try:
        # 0o700 on the sessions dir so session sidecars (containing live
        # session tokens) are not readable by other UIDs in the
        # container. ``mkdir(mode=...)`` only takes effect on dirs
        # actually created — so chmod after handles the pre-existing
        # case (e.g. the dir was created by a prior version with the
        # default umask).
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(target.parent, 0o700)
        except OSError:
            # Best-effort — non-fatal if filesystem rejects chmod
            # (e.g. tmpfs with restrictive options).
            pass
    except OSError as e:
        logger.warning(
            "session snapshot for '%s': cannot create parent dir: %s",
            agent_id, e,
        )
        return False

    try:
        state = await context.storage_state()
    except Exception as e:
        logger.warning(
            "session snapshot for '%s': storage_state() failed: %s",
            agent_id, e,
        )
        return False

    payload = {
        "version": _SCHEMA_VERSION,
        "saved_at": int(time.time()),
        "storage_state": state,
    }

    # Cap serialized payload size before opening any file. A hostile
    # site can inflate one origin's localStorage near the per-origin
    # quota; with N origins this can balloon to many MiB. Refuse to
    # write rather than fill the disk.
    try:
        serialized = json.dumps(payload)
    except (TypeError, ValueError) as e:
        logger.warning("session snapshot for '%s': payload not JSON-serializable: %s", agent_id, e)
        return False
    if len(serialized) > _MAX_SNAPSHOT_BYTES:
        logger.warning(
            "session snapshot for '%s' exceeds %d bytes (got %d) — refusing to write",
            agent_id, _MAX_SNAPSHOT_BYTES, len(serialized),
        )
        return False

    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        # Open with 0o600 from the start (umask-aware) so there is no
        # world-readable window before the explicit chmod below.
        # Storage state contains session tokens — same posture as
        # ``.agent.env`` (CLAUDE.md §Security Boundaries).
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        # We own ``fd`` until ``os.fdopen`` returns. Both ``fchmod`` and
        # ``fdopen`` can raise (MemoryError on buffer alloc, OSError on
        # fchmod), which would leak the descriptor unless we close it
        # in the narrow pre-fdopen window. After fdopen returns the
        # returned file object owns ``fd`` and the ``with`` block
        # handles cleanup on writer errors.
        try:
            os.fchmod(fd, 0o600)
            fh = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        with fh:
            # Re-use the pre-serialized payload from the size-cap
            # check — avoids an unnecessary second JSON encode pass on
            # the hot path.
            fh.write(serialized)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        # ``os.replace`` preserves the destination's mode on most
        # filesystems but Python's docs are not load-bearing on this;
        # explicit chmod after replace ensures 0o600 regardless of
        # whether the target existed (and what its prior mode was).
        os.chmod(target, 0o600)
    except OSError as e:
        logger.warning("session snapshot for '%s' failed: %s", agent_id, e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False

    # Don't log cookie counts or origin counts at INFO — we don't want
    # the log stream to be a side channel for "this agent is logged
    # into N sites". DEBUG is fine for operator troubleshooting.
    logger.info("session snapshot wrote sidecar for '%s'", agent_id)
    return True


# ── Restore ────────────────────────────────────────────────────────────────


async def restore_session(
    agent_id: str,
    context_factory: Callable[..., Awaitable[Any]],
) -> Any | None:
    """Build a BrowserContext seeded with the previously-snapshotted state.

    ``context_factory`` is a callable that creates a BrowserContext
    given a ``storage_state`` keyword argument — passed in so the
    caller's full context-creation parameters (proxy, viewport, init
    scripts) are preserved. We just inject the storage_state.

    Returns the new context on success; ``None`` when:

      * No sidecar exists.
      * Sidecar JSON is malformed (logs a warning; sidecar is **not**
        clobbered — operator decides whether to delete it).
      * ``version`` field is missing or doesn't match
        :data:`_SCHEMA_VERSION`. The future-version case logs a
        warning so operators see the version drift; we don't try to
        migrate forward (we're an older build).
      * ``context_factory`` raises while restoring.

    On every failure path the caller falls back to a fresh-state
    context — that's the design intent of returning ``None`` rather
    than raising. A bad sidecar must not block browser startup.
    """
    try:
        target = session_path(agent_id)
    except InvalidAgentIdError as e:
        logger.warning("session restore rejected: %s", e)
        return None
    if not target.exists():
        return None

    try:
        with open(target, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "session restore for '%s': could not read sidecar (%s); "
            "starting fresh — sidecar left intact for operator review",
            agent_id, e,
        )
        return None

    if not isinstance(payload, dict):
        logger.warning(
            "session restore for '%s': payload not a dict; starting fresh",
            agent_id,
        )
        return None

    version = payload.get("version")
    if version != _SCHEMA_VERSION:
        logger.warning(
            "session restore for '%s': unexpected version %r "
            "(expected %d); starting fresh",
            agent_id, version, _SCHEMA_VERSION,
        )
        return None

    state = payload.get("storage_state")
    if not isinstance(state, dict):
        logger.warning(
            "session restore for '%s': storage_state missing or wrong "
            "type; starting fresh", agent_id,
        )
        return None

    try:
        context = await context_factory(storage_state=state)
    except Exception as e:
        logger.warning(
            "session restore for '%s': context_factory raised (%s); "
            "starting fresh", agent_id, e,
        )
        return None

    logger.info("session restore loaded sidecar for '%s'", agent_id)
    return context


# ── Clear ──────────────────────────────────────────────────────────────────


async def clear_session(agent_id: str) -> bool:
    """Delete the sidecar (operator opt-out / fresh-start path).

    Returns ``True`` if the file existed and was deleted, ``False``
    otherwise (already absent, or ``unlink`` failed). The async
    signature matches the rest of the module — there is no actual
    awaiting because ``unlink`` is a synchronous syscall with
    negligible cost; the consistent shape lets callers ``await`` all
    three public functions uniformly.
    """
    try:
        target = session_path(agent_id)
    except InvalidAgentIdError as e:
        logger.warning("session clear rejected: %s", e)
        return False
    if not target.exists():
        return False
    try:
        target.unlink()
    except OSError as e:
        logger.warning(
            "session clear for '%s' failed: %s", agent_id, e,
        )
        return False
    logger.info("session sidecar cleared for '%s'", agent_id)
    return True


# ── Read-only summary (dashboard surface) ──────────────────────────────────


def session_summary(agent_id: str) -> dict[str, Any]:
    """Return a privacy-safe summary of the agent's session sidecar.

    Output shape::

        {
            "has_persisted_session": bool,
            "saved_at": <iso8601 utc string> | None,
            "origin_count": int,
            "cookie_count": int,
        }

    Origins themselves are NOT returned — operators see the COUNT but
    not which sites the agent is logged into. Cookie values are never
    returned. The dashboard uses this for the read-only operator
    panel; the DELETE endpoint clears the sidecar on demand.

    Sync (no I/O blocking) because it's called from a hot dashboard
    poll path — async would buy nothing here (json.load is CPU-only).
    """
    try:
        target = session_path(agent_id)
    except InvalidAgentIdError:
        # Operators reading dashboard summary for a malformed agent_id
        # see the empty shape rather than a stack trace.
        return {
            "has_persisted_session": False,
            "saved_at": None,
            "origin_count": 0,
            "cookie_count": 0,
        }
    if not target.exists():
        return {
            "has_persisted_session": False,
            "saved_at": None,
            "origin_count": 0,
            "cookie_count": 0,
        }
    try:
        with open(target, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        # Malformed sidecar — surface as "present but unreadable" so
        # the operator can clear it. Counts are 0, has_persisted is
        # True (the file exists). saved_at is None to signal we
        # couldn't read the timestamp.
        return {
            "has_persisted_session": True,
            "saved_at": None,
            "origin_count": 0,
            "cookie_count": 0,
        }

    saved_at_epoch = payload.get("saved_at") if isinstance(payload, dict) else None
    saved_at_iso = None
    if isinstance(saved_at_epoch, (int, float)):
        from datetime import datetime, timezone
        saved_at_iso = datetime.fromtimestamp(
            float(saved_at_epoch), tz=timezone.utc,
        ).isoformat()

    state = payload.get("storage_state") if isinstance(payload, dict) else None
    cookie_count = 0
    origin_count = 0
    if isinstance(state, dict):
        cookies = state.get("cookies")
        if isinstance(cookies, list):
            cookie_count = len(cookies)
        origins = state.get("origins")
        if isinstance(origins, list):
            origin_count = len(origins)

    return {
        "has_persisted_session": True,
        "saved_at": saved_at_iso,
        "origin_count": origin_count,
        "cookie_count": cookie_count,
    }
