"""Durable per-agent fingerprint-burn + binding-signature state.

Two pieces of always-on browser identity state used to live only in
``src/browser/service.py`` module globals, so they evaporated on a
browser-service restart:

  1. **Fingerprint-burn state** — the rolling accept/reject window, the
     hard-burn set, and its reasons. When these were in-memory-only, a
     fingerprint flagged "burned" (poisoned) read *clean* after a restart
     even though the cookies that got it burned still sat on disk in the
     Camoufox persistent profile. The operator would see a healthy agent
     that instantly re-tripped every anti-bot vendor.

  2. **Per-agent binding signatures** — a short opaque hash of the
     ``(UA, proxy)`` identity the agent last launched with. The
     always-on profile channel (``user_data_dir``) restores ALL cookies
     on launch, including anti-bot trust tokens (``cf_clearance``,
     ``datadome``, ``_abck`` …) that vendors bind to the original
     ``(UA, IP)`` tuple. Replaying one of those under a rotated UA or a
     different proxy egress is an instant 403 AND lowers the trust score
     for the session lifetime. ``service._enforce_binding_cookie_coherence``
     compares the persisted signature against the current one on each
     launch and drops bound cookies on a mismatch — which requires the
     signature to survive restarts.

Both are identity state that must be durable, so they share one sidecar.

The in-memory globals in ``service.py`` stay the source of truth
in-process; this module is pure serialization. It deliberately does NOT
import ``service`` (no cycle): callers pass the state in on
:func:`snapshot` and receive a :class:`FingerprintState` back from
:func:`restore`, then repopulate their own globals.

Atomic-write + non-fatal-restore protocol mirrors
:mod:`src.browser.captcha_cost_counter` (open ``0o600`` → ``fchmod`` →
``json.dump`` → ``fsync`` → ``os.replace`` → ``chmod``). A missing or
corrupt file restores empty rather than raising — a bad sidecar must
never block browser-service startup.

On-disk schema::

    {
      "version": 1,
      "saved_at": <epoch seconds>,
      "window": {"<agent_id>": [true, false, ...], ...},
      "last_signal": {"<agent_id>": <epoch float>, ...},
      "hard_burned": ["<agent_id>", ...],
      "hard_burn_reason": {"<agent_id>": ["<vendor>", "<signal>"], ...},
      "binding_signatures": {"<agent_id>": "<16-hex>", ...}
    }

Default path ``data/fingerprint_state.json``; env override
``FINGERPRINT_STATE_PATH`` (consistent with ``CAPTCHA_COST_COUNTER_PATH``).
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("browser.fingerprint_state")


# Persistence path. Lives under ``data/`` alongside the captcha-cost
# counter sidecar so all browser-service persistence stays under one
# parent. Override-able via env for tests + custom deployments.
_DEFAULT_PATH = "data/fingerprint_state.json"
_SCHEMA_VERSION = 1
# Default rolling-window bound. Mirrors ``service._FINGERPRINT_WINDOW_SIZE``
# but kept as a local default so this module has NO import cycle back into
# service.py. The live caller passes its own constant to :func:`restore`
# (``window_maxlen=_FINGERPRINT_WINDOW_SIZE``) so the two can never drift
# silently; the default only matters for standalone / test use.
_DEFAULT_WINDOW_MAXLEN = 10


def _state_path() -> Path:
    """Return the fingerprint-state sidecar path (env-overridable)."""
    return Path(os.environ.get("FINGERPRINT_STATE_PATH", _DEFAULT_PATH))


@dataclass
class FingerprintState:
    """In-memory view of the persisted fingerprint state.

    Field shapes match the ``service.py`` globals they seed so a caller
    can ``.update()`` each straight in. ``window`` values are bounded
    ``deque`` rebuilt with the caller-supplied ``maxlen`` on restore.
    """

    window: dict[str, deque] = field(default_factory=dict)
    last_signal: dict[str, float] = field(default_factory=dict)
    hard_burned: set[str] = field(default_factory=set)
    hard_burn_reason: dict[str, tuple[str, str]] = field(default_factory=dict)
    binding_signatures: dict[str, str] = field(default_factory=dict)


def snapshot(
    *,
    window: Mapping[str, Iterable[bool]],
    last_signal: Mapping[str, float],
    hard_burned: Iterable[str],
    hard_burn_reason: Mapping[str, Iterable[str]],
    binding_signatures: Mapping[str, str],
    path: Path | str | None = None,
) -> bool:
    """Atomically write the supplied state to ``path`` as JSON.

    Synchronous by design: the ``service.py`` caller already holds the
    fingerprint lock across its mutation, so there is no internal state
    to guard here. Returns ``True`` on success; failures log + return
    ``False`` rather than raise — the mutation / shutdown paths must not
    abort because persistence failed (same posture as the captcha-cost
    counter snapshot).

    Atomic-write protocol: ``os.open`` the tmp sibling with mode 0o600
    (no world-readable window), ``fchmod``, write + ``fsync``,
    ``os.replace`` over the destination, then re-``chmod`` 0o600.
    """
    target = Path(path) if path else _state_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("fingerprint_state snapshot: cannot create parent dir: %s", e)
        return False

    payload = {
        "version": _SCHEMA_VERSION,
        "saved_at": int(time.time()),
        "window": {str(a): [bool(v) for v in w] for a, w in window.items()},
        "last_signal": {str(a): float(t) for a, t in last_signal.items()},
        "hard_burned": sorted(str(a) for a in hard_burned),
        "hard_burn_reason": {str(a): [str(x) for x in r] for a, r in hard_burn_reason.items()},
        "binding_signatures": {str(a): str(s) for a, s in binding_signatures.items()},
    }

    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        # Open with 0o600 from the start (umask-aware) so there is no
        # world-readable window before the explicit chmod below.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        # We own ``fd`` until ``os.fdopen`` returns; ``fchmod`` / ``fdopen``
        # can raise (OSError / MemoryError) which would leak the descriptor
        # unless we close it in that narrow window. After fdopen returns the
        # file object owns ``fd`` and the ``with`` block handles cleanup.
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
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        # ``os.replace`` preserves the destination's mode on most
        # filesystems but Python's docs are not load-bearing on this;
        # explicit chmod after replace ensures 0o600 regardless of
        # whether the target pre-existed.
        os.chmod(target, 0o600)
    except OSError as e:
        logger.warning("fingerprint_state snapshot failed: %s", e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True


def restore(
    path: Path | str | None = None,
    *,
    window_maxlen: int = _DEFAULT_WINDOW_MAXLEN,
) -> FingerprintState:
    """Load state from ``path`` if it exists, returning a :class:`FingerprintState`.

    Missing / unreadable / malformed files are non-fatal — log + return an
    empty state. Every field is parsed defensively; a malformed individual
    entry is skipped rather than aborting the whole restore. ``window``
    deques are rebuilt with ``maxlen=window_maxlen`` so the bound survives
    the round-trip (the live caller passes ``_FINGERPRINT_WINDOW_SIZE``).
    """
    target = Path(path) if path else _state_path()
    state = FingerprintState()
    if not target.exists():
        return state
    try:
        with open(target, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("fingerprint_state restore: %s — starting empty", e)
        return state
    if not isinstance(payload, dict):
        logger.warning(
            "fingerprint_state restore: unexpected payload type — starting empty",
        )
        return state
    version = payload.get("version")
    if version != _SCHEMA_VERSION:
        logger.warning(
            "fingerprint_state restore: unexpected version %r (expected %d) — starting empty",
            version,
            _SCHEMA_VERSION,
        )
        return state

    window = payload.get("window")
    if isinstance(window, dict):
        for agent_id, values in window.items():
            if not isinstance(agent_id, str) or not isinstance(values, list):
                continue
            state.window[agent_id] = deque(
                (bool(v) for v in values),
                maxlen=window_maxlen,
            )

    last_signal = payload.get("last_signal")
    if isinstance(last_signal, dict):
        for agent_id, ts in last_signal.items():
            if isinstance(agent_id, str) and isinstance(ts, (int, float)):
                state.last_signal[agent_id] = float(ts)

    hard_burned = payload.get("hard_burned")
    if isinstance(hard_burned, list):
        state.hard_burned = {a for a in hard_burned if isinstance(a, str)}

    reasons = payload.get("hard_burn_reason")
    if isinstance(reasons, dict):
        for agent_id, pair in reasons.items():
            if not isinstance(agent_id, str):
                continue
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                state.hard_burn_reason[agent_id] = (str(pair[0]), str(pair[1]))

    sigs = payload.get("binding_signatures")
    if isinstance(sigs, dict):
        for agent_id, sig in sigs.items():
            if isinstance(agent_id, str) and isinstance(sig, str):
                state.binding_signatures[agent_id] = sig

    return state
