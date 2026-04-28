"""Centralized flag loader for browser-service config (Phase 1.6).

Every phase of the browser roadmap adds environment-variable knobs for
feature flags, rollout gates, and tuning parameters. Scattered os.environ
reads across modules produce inconsistent defaults, case-sensitivity bugs,
and make it impossible to inject per-agent overrides later.

This module is the single read point. Precedence (highest → lowest):

1. **Per-agent override** — registered at runtime via :func:`set_agent_override`.
   Intended for dashboard-driven per-template tuning (Phase 1.1 permission
   editor gains a "browser flags" panel; until then, the interface is here).
2. **Operator settings** — ``config/settings.json`` under ``browser_flags``.
   Optional; absent in fresh installs.
3. **Environment variable** — the flag's canonical name (case-sensitive).
4. **Hardcoded default** passed at the call site.

Values pass through typed accessors (:func:`get_bool`, :func:`get_int`,
:func:`get_str`) which coerce and validate; a malformed value in any layer
logs a warning and falls through to the next layer rather than crashing.

Flag names are canonical strings — no enum required. Known flags are
collected in :data:`KNOWN_FLAGS` purely for documentation / tab-completion;
unknown names work too (readers get a warning in debug mode).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("browser.flags")


# ── Known flags (documentation only — not enforced) ────────────────────────


# Inventory of every browser-related flag declared across the roadmap.
# Update when adding new knobs. The left column is the canonical env var
# name; right column is a one-line description.
KNOWN_FLAGS: dict[str, str] = {
    # ── Snapshot / screenshot formats (§7) ────────────────────────────────
    "BROWSER_SNAPSHOT_FORMAT": "v1 | v2 (default v2 after one release gate)",
    "BROWSER_SCREENSHOT_FORMAT": "webp (default) | png",
    "BROWSER_SCREENSHOT_QUALITY": "WebP quality 1-100 (default 75)",
    # ── Ad-blocker / egress (§7.1) ─────────────────────────────
    "BROWSER_ENABLE_ADBLOCK": "true | false (default true; gates uBO install)",
    # ── Resolution pool (§6.1) ────────────────────────────────────────────
    "BROWSER_RESOLUTION_POOL": "true | false (default true after phase 6.1)",
    # ── Canary (§5.4) ─────────────────────────────────────────────────────
    "BROWSER_CANARY_ENABLED": "true | false (default false)",
    # ── Operator kill switches for high-trust phases ──────────────────────
    "BROWSER_DOWNLOADS_DISABLED": "true | false (default false)",
    "BROWSER_NETWORK_INSPECT_DISABLED": "true | false (default false)",
    "BROWSER_COOKIE_IMPORT_DISABLED": "true | false (default false)",
    # ── Redaction (§4.3, existing) ────────────────────────────────────────
    "OPENLEGION_REDACTION_URL_QUERY_ALLOW": "comma-separated param names",
    # ── Concurrency (§8.2) ────────────────────────────────────────────────
    "OPENLEGION_BROWSER_MAX_CONCURRENT": "int, startup-only (default 5)",
    # ── File transfer (§4.5 / §8.1) ───────────────────────────────────────
    "OPENLEGION_UPLOAD_STAGE_MAX_MB": "int, per-file upload byte cap (default 50)",
    "OPENLEGION_UPLOAD_STAGE_DIR": "mesh staging dir (default /tmp/openlegion-upload-stage)",
    "OPENLEGION_UPLOAD_RECV_DIR": "browser receive dir (default /tmp/upload-recv)",
    "OPENLEGION_UPLOAD_STAGE_TTL_S": "orphan staging TTL in seconds (default 60)",
    # ── CAPTCHA solver (§11) ──────────────────────────────────────────────
    "CAPTCHA_SOLVER_PROVIDER": "2captcha | capsolver | unset",
    "CAPTCHA_SOLVER_KEY": "API key for the primary provider",
    "CAPTCHA_SOLVER_PROVIDER_SECONDARY": "failover provider (§11.8)",
    "CAPTCHA_SOLVER_KEY_SECONDARY": "failover API key",
    "CAPTCHA_PACING_MS_MIN": "solve-pacing lower bound (default 3000)",
    "CAPTCHA_PACING_MS_MAX": "solve-pacing upper bound (default 12000)",
    "CAPTCHA_SOLVE_PACING_MU_MS": "solve-pacing Gaussian mean (default 6000) — §11.11",
    "CAPTCHA_SOLVE_PACING_SIGMA_MS": "solve-pacing Gaussian stddev (default 2500) — §11.11",
    "CAPTCHA_TIMEOUT_RECAPTCHA_V2_CHECKBOX_MS": "per-type solver timeout (default 120000) — §11.9",
    "CAPTCHA_TIMEOUT_RECAPTCHA_V2_INVISIBLE_MS": "per-type solver timeout (default 120000) — §11.9",
    "CAPTCHA_TIMEOUT_RECAPTCHA_V3_MS": "per-type solver timeout (default 60000) — §11.9",
    "CAPTCHA_TIMEOUT_RECAPTCHA_ENTERPRISE_V2_MS": "per-type solver timeout (default 120000) — §11.9",
    "CAPTCHA_TIMEOUT_RECAPTCHA_ENTERPRISE_V3_MS": "per-type solver timeout (default 60000) — §11.9",
    "CAPTCHA_TIMEOUT_HCAPTCHA_MS": "per-type solver timeout (default 120000) — §11.9",
    "CAPTCHA_TIMEOUT_TURNSTILE_MS": "per-type solver timeout (default 180000) — §11.9",
    "CAPTCHA_TIMEOUT_CF_INTERSTITIAL_TURNSTILE_MS": "per-type solver timeout (default 180000) — §11.9",
    "CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH": "per-agent USD cap",
    "CAPTCHA_COST_LIMIT_USD_PER_TENANT_MONTH": "per-tenant USD cap",
    "CAPTCHA_DISABLED": "true | false (default false)",
    "CAPTCHA_SOLVER_PROXY_TYPE": "http | https | socks4 | socks5",
    "CAPTCHA_SOLVER_PROXY_ADDRESS": "dedicated solver proxy host",
    "CAPTCHA_SOLVER_PROXY_PORT": "int",
    "CAPTCHA_SOLVER_PROXY_LOGIN": "dedicated solver proxy user",
    "CAPTCHA_SOLVER_PROXY_PASSWORD": "dedicated solver proxy pass",
    "CAPTCHA_RECAPTCHA_V3_MIN_SCORE": "0.1-0.9 (default 0.7)",
    "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS":
        "comma-separated; force normal solver flow on hardcoded-unsolvable hosts (§11.18)",
    "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS":
        "comma-separated; force escalation-only on hosts we'd otherwise solve (§11.18)",
    # ── Observability ─────────────────────────────────────────────────────
    "BROWSER_RECORD_BEHAVIOR": "1 to enable behavior recorder (§5.3)",
}


# ── Overrides state ────────────────────────────────────────────────────────


# thread-safe because reads may happen from multiple asyncio tasks and
# operator mutations (dashboard UI) arrive on a different thread.
_lock = threading.RLock()
_agent_overrides: dict[str, dict[str, str]] = {}     # agent_id -> name -> value
_operator_settings: dict[str, str] | None = None     # lazy-loaded from disk


def _settings_path() -> Path:
    """Location of the operator settings file. Override-able via env for
    tests + containerized deployments."""
    return Path(os.environ.get("OPENLEGION_SETTINGS_PATH", "config/settings.json"))


def _load_operator_settings() -> dict[str, str]:
    """Return ``config/settings.json``'s ``browser_flags`` dict, or empty.

    Lazy-loaded on first flag read; cached thereafter. Call
    :func:`reload_operator_settings` if the file changes at runtime.
    """
    global _operator_settings
    with _lock:
        if _operator_settings is not None:
            return _operator_settings
        path = _settings_path()
        if not path.exists():
            _operator_settings = {}
            return _operator_settings
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Could not parse operator settings at %s; using env+defaults only",
                path,
            )
            _operator_settings = {}
            return _operator_settings
        flags = raw.get("browser_flags", {})
        if not isinstance(flags, dict):
            logger.warning("browser_flags in %s is not a dict; ignoring", path)
            flags = {}
        # Coerce keys/values to str for uniform lookup.
        _operator_settings = {str(k): str(v) for k, v in flags.items()}
        return _operator_settings


def reload_operator_settings() -> None:
    """Force re-read of ``config/settings.json`` on next flag access."""
    global _operator_settings
    with _lock:
        _operator_settings = None


def set_agent_override(agent_id: str, name: str, value: str | None) -> None:
    """Register (or clear) a per-agent flag override.

    ``value=None`` clears the override for that ``(agent_id, name)`` pair.
    Agent-scoped — other agents are unaffected.
    """
    with _lock:
        if value is None:
            bucket = _agent_overrides.get(agent_id)
            if bucket:
                bucket.pop(name, None)
                if not bucket:
                    _agent_overrides.pop(agent_id, None)
            return
        _agent_overrides.setdefault(agent_id, {})[name] = str(value)


def clear_agent_overrides(agent_id: str) -> None:
    """Remove all per-agent overrides for ``agent_id``.

    Intended for container restart / fresh-profile reset paths.
    """
    with _lock:
        _agent_overrides.pop(agent_id, None)


def get_agent_overrides(agent_id: str) -> dict[str, str]:
    """Return the current override dict for ``agent_id`` (copy)."""
    with _lock:
        return dict(_agent_overrides.get(agent_id, {}))


# ── Raw lookup (string layer resolution) ───────────────────────────────────


def _lookup_raw(name: str, agent_id: str | None) -> str | None:
    """Return the first string value found across the precedence chain, or ``None``."""
    with _lock:
        if agent_id is not None:
            bucket = _agent_overrides.get(agent_id)
            if bucket and name in bucket:
                return bucket[name]
        settings = _load_operator_settings()
        if name in settings:
            return settings[name]
    # Env is outside the lock — os.environ is thread-safe in CPython.
    return os.environ.get(name)


def _lookup_raw_candidates(name: str, agent_id: str | None) -> list[tuple[str, str]]:
    """Return all configured values in precedence order.

    Typed accessors use this to fall through when a higher-precedence
    value is malformed. A bad per-agent override should not mask a valid
    operator setting or environment fallback.
    """
    candidates: list[tuple[str, str]] = []
    with _lock:
        if agent_id is not None:
            bucket = _agent_overrides.get(agent_id)
            if bucket and name in bucket:
                candidates.append(("agent_override", bucket[name]))
        settings = _load_operator_settings()
        if name in settings:
            candidates.append(("operator_settings", settings[name]))
    raw_env = os.environ.get(name)
    if raw_env is not None:
        candidates.append(("environment", raw_env))
    return candidates


# ── Typed accessors ────────────────────────────────────────────────────────


def get_str(name: str, default: str = "", *, agent_id: str | None = None) -> str:
    """Return the string value for ``name``, or ``default`` if unset."""
    raw = _lookup_raw(name, agent_id)
    if raw is None:
        return default
    return raw


_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off", ""})


def get_bool(name: str, default: bool, *, agent_id: str | None = None) -> bool:
    """Return the boolean value for ``name``, coercing ``true|1|yes|on`` to
    ``True`` and ``false|0|no|off|<empty>`` to ``False``. Anything else logs
    a warning and falls through to the next precedence layer."""
    for source, raw in _lookup_raw_candidates(name, agent_id):
        lowered = raw.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        logger.warning(
            "Flag %s from %s has non-boolean value %r; "
            "falling through to next layer",
            name, source, raw,
        )
    return default


def get_int(
    name: str,
    default: int,
    *,
    agent_id: str | None = None,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Return the integer value for ``name``, clamped to
    ``[min_value, max_value]`` when bounds are provided. Invalid ints log
    a warning and fall through to the next precedence layer."""
    for source, raw in _lookup_raw_candidates(name, agent_id):
        try:
            value = int(raw.strip())
        except (ValueError, AttributeError):
            logger.warning(
                "Flag %s from %s has non-integer value %r; "
                "falling through to next layer",
                name, source, raw,
            )
            continue
        if min_value is not None and value < min_value:
            return min_value
        if max_value is not None and value > max_value:
            return max_value
        return value
    return default


def get_float(
    name: str,
    default: float,
    *,
    agent_id: str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Return the float value for ``name``, clamped to bounds when provided."""
    for source, raw in _lookup_raw_candidates(name, agent_id):
        try:
            value = float(raw.strip())
        except (ValueError, AttributeError):
            logger.warning(
                "Flag %s from %s has non-float value %r; "
                "falling through to next layer",
                name, source, raw,
            )
            continue
        if min_value is not None and value < min_value:
            return min_value
        if max_value is not None and value > max_value:
            return max_value
        return value
    return default


def snapshot_all(*, agent_id: str | None = None) -> dict[str, Any]:
    """Return every known flag's effective value for ``agent_id``.

    Used by the dashboard flags panel for read-only display. Values are
    raw strings from their winning layer (or ``None`` if unset at every
    layer).
    """
    result: dict[str, Any] = {}
    for name in KNOWN_FLAGS:
        result[name] = _lookup_raw(name, agent_id)
    return result
