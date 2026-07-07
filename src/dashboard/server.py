"""Dashboard API router: fleet overview, costs, comms (blackboard + pubsub), traces, management.

Serves the SPA template and static files, plus JSON API endpoints
consumed by the Alpine.js frontend.  All data comes from live Python
objects — no HTTP round-trips through mesh endpoints.

Phase 10 §24 — billing-export endpoint.

  ``GET /dashboard/api/billing/captcha-rollup?tenant=<id>&period=<period>``
  returns a CSV of CAPTCHA-solver spend rolled up across every agent
  inside the named tenant (team). Operator-only: gated by the same
  ``ol_session`` cookie + ``X-Requested-With`` posture as every other
  ``/dashboard/*`` endpoint (CSRF unnecessary on GET, but auth is still
  enforced via the ``_verify_dashboard_auth`` dependency on the router).
  ``period`` is one of ``daily`` (today UTC), ``weekly`` (rolling 7d), or
  ``monthly`` (current calendar month). Each CSV row is
  ``period_start, agent_id, millicents, dollars, data_scope`` plus a
  final ``__tenant_total__`` synthetic row carrying the rolled-up total.
  The ``data_scope`` column is ``monthly_actual`` when the period is
  ``monthly`` (live state IS current-month so the number is correct);
  for ``daily`` / ``weekly`` it is ``current_month_aggregate`` because
  the in-memory state is month-granularity only — finance reconciliation
  tooling reads the column to flag period-imprecise numbers.
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError

from src.cli.proxy import build_proxy_env_vars, resolve_agent_proxy
from src.dashboard.auth import verify_session_cookie
from src.shared import limits as _limits
from src.shared.paths import resolve_under_root
from src.shared.sqlite_helpers import open_db
from src.shared.types import CRED_HANDLE_RE
from src.shared.utils import (
    dumps_safe,
    friendly_streaming_error,
    sanitize_for_prompt,
    set_llm_max_tokens_env,
    setup_logging,
)

if TYPE_CHECKING:
    from src.dashboard.events import EventBus
    from src.host.costs import CostTracker
    from src.host.health import HealthMonitor
    from src.host.mesh import Blackboard
    from src.host.traces import TraceStore

logger = setup_logging("dashboard.server")
# Phase 6 §9.2: dedicated logger for cookie-import audit. Records domain
# + cookie name only — NEVER the value (see _validate_cookies docstring).
_cookie_audit_logger = setup_logging("dashboard.cookie_import")

_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Cap on in-flight chains the live pipeline card examines per request. Bounds
# the per-task-event cost (each root runs a recursive workflow_snapshot).
_MAX_PIPELINE_ROOTS = 40


async def _fetch_browser_metrics_upstream(
    client: Any,
    service_url: str,
    auth_token: str,
    since_seq: int,
) -> dict:
    """Call ``GET /browser/metrics?since=<seq>`` on the browser service.

    Pulled out of :func:`create_dashboard_router`'s closure so tests can
    patch it without reaching into a closure cell. Returns the §2.3 error
    envelope on any failure (network, HTTP error, malformed JSON) and the
    upstream payload (with ``success: True``) on success.
    """
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    try:
        resp = await client.get(
            f"{service_url}/browser/metrics",
            params={"since": since_seq},
            headers=headers,
            timeout=10,
        )
    except Exception as e:
        logger.debug("Browser metrics fetch failed: %s", e)
        return {
            "success": False,
            "error": {
                "code": "service_unavailable",
                "message": "Browser service unreachable",
                "retry_after_ms": 60_000,
            },
        }
    if resp.status_code >= 400:
        logger.debug(
            "Browser metrics fetch returned HTTP %d", resp.status_code,
        )
        return {
            "success": False,
            "error": {
                "code": "service_unavailable",
                "message": f"Browser service returned HTTP {resp.status_code}",
                "retry_after_ms": 60_000,
            },
        }
    try:
        data = resp.json()
    except Exception as e:
        logger.debug("Browser metrics JSON parse failed: %s", e)
        return {
            "success": False,
            "error": {
                "code": "service_unavailable",
                "message": "Malformed response from browser service",
                "retry_after_ms": None,
            },
        }
    return {"success": True, "data": data}


def _get_builtin_tool_names() -> frozenset[str]:
    """Return the names of all built-in agent tools by scanning the builtins package.

    Uses a regex over source files rather than executing code, so it is safe to
    call from the dashboard host process without any side effects.  Result is
    cached in a module-level variable after the first call.
    """
    if _get_builtin_tool_names._cache is not None:
        return _get_builtin_tool_names._cache
    builtins_dir = Path(__file__).parent.parent / "agent" / "builtins"
    names: set[str] = set()
    if builtins_dir.exists():
        for py_file in builtins_dir.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                text = py_file.read_text()
                names.update(re.findall(r'@tool\s*\(\s*name\s*=\s*["\']([^"\']+)["\']', text))
            except OSError:
                pass
    _get_builtin_tool_names._cache = frozenset(names)
    return _get_builtin_tool_names._cache


_get_builtin_tool_names._cache = None  # type: ignore[attr-defined]


def _mask_mcp_servers_for_get(servers: list[dict] | None) -> list[dict]:
    """Return MCP server dicts with env values stripped, suitable for
    GET responses. Env values may be plaintext secrets or ``$CRED{name}``
    handles pointing at one; neither is safe to ship over the API.

    Each server entry retains ``name``, ``command``, and ``args``;
    ``env`` is omitted entirely (NOT returned as ``null``) and replaced
    with ``env_keys`` — a sorted list of the env variable names. The
    omission is deliberate: a naive ``GET → edit → PUT`` round-trip
    would otherwise lose env when the PUT handler interprets a present
    ``env=null`` as "replace with no env." The connector PUT contract
    is "env absent = preserve, env present (dict or {}) = replace
    wholesale" (see ``api_connector_upsert``).
    """
    if not servers:
        return []
    result: list[dict] = []
    for s in servers:
        masked = {k: v for k, v in s.items() if k != "env"}
        env = s.get("env") or {}
        masked["env_keys"] = sorted(env.keys()) if env else []
        result.append(masked)
    return result


def _redact_mcp_env_for_audit(value: object) -> object:
    """Strip env VALUES from an ``mcp_servers`` payload for audit-log
    logging. Keys are preserved so an operator reviewing the audit
    trail can see which env vars changed; the values are not
    persisted to the audit table (which would be a covert plaintext
    secret leak otherwise — values may be ``$CRED{name}`` handles or
    raw secrets, and both are sensitive).

    Returns the original value unchanged if it isn't shaped like an
    ``mcp_servers`` list-of-dicts payload — the audit caller handles
    both old-and-new sides identically and may pass an empty string
    on first-write.
    """
    if not isinstance(value, list):
        return value
    redacted: list[dict] = []
    for s in value:
        if not isinstance(s, dict):
            return value  # not the shape we redact
        copy = {k: v for k, v in s.items() if k != "env"}
        env = s.get("env") or {}
        if env:
            copy["env_keys"] = sorted(env.keys())
        redacted.append(copy)
    return redacted


def _compute_asset_version() -> str:
    """Hash all static files + the template to produce a cache-bust version string.

    Changes to ANY dashboard file (JS, CSS, HTML template) produce a new hash,
    which changes the query parameter on all static file URLs, forcing browsers
    to fetch fresh copies.  Computed once at import time.
    """
    h = hashlib.sha256()
    for pattern in ("static/**/*", "templates/**/*"):
        for f in sorted(_HERE.glob(pattern)):
            if f.is_file():
                h.update(f.read_bytes())
    return h.hexdigest()[:12]


ASSET_VERSION = _compute_asset_version()


def _verify_dashboard_auth(request: Request) -> None:
    """Verify the ol_session cookie on dashboard API requests."""
    error = verify_session_cookie(request.cookies.get("ol_session", ""))
    if error is not None:
        raise HTTPException(401, error)


def _parse_positive_float(value: Any, field: str, fallback: float) -> float:
    """Validate *value* as a positive float, returning *fallback* if None."""
    if value is None:
        return fallback
    try:
        result = float(value)
        if result <= 0:
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"{field} must be a positive number")
    return result


def _log_cron_task_exception(task: object) -> None:
    """Log unhandled exceptions from fire-and-forget cron tasks."""
    import asyncio
    t = task if isinstance(task, asyncio.Task) else None
    if t is None or t.cancelled():
        return
    exc = t.exception()
    if exc:
        logger.error("Background cron job failed: %s", exc, exc_info=exc)


def _wallet_chain_label(chain_id: str, cfg: dict) -> str:
    """Human-friendly chain label for the dashboard UI."""
    name = chain_id.split(":", 1)[-1].replace("-", " ").title()
    eco = cfg.get("ecosystem", "").upper()
    symbol = cfg.get("symbol", "")
    if "devnet" in chain_id or "sepolia" in chain_id:
        return f"{name} ({eco} Testnet)"
    return f"{name} ({eco} · {symbol})"


# ── Phase 6 §9.2 — operator cookie/session import helpers ─────────────────
# Pure functions split out for unit-testability. The dashboard endpoint
# validates and shape-coerces operator-supplied cookie payloads here, then
# hands a normalized Playwright-shaped list to the browser service.
#
# SECURITY INVARIANTS:
#   • Cookie VALUES never appear in audit logs, error messages, or any
#     return shape that crosses a process boundary. Only domain + name
#     are logged.
#   • Validation rejects cookies whose ``value`` is larger than 4 KiB
#     (per cookie) and whole payloads larger than 256 KiB.
#   • RFC 6265bis violations (`__Host-` with explicit domain) are
#     dropped server-side; importing them is silently rejected by Firefox
#     anyway and confuses operators.
#   • IP-literal domains are dropped — Firefox stores them but our SSRF
#     egress filter would block the requests, so importing is a footgun.

_COOKIE_VALUE_MAX_BYTES = 4 * 1024              # per-cookie value cap
_COOKIE_PAYLOAD_MAX_BYTES = 256 * 1024          # whole-payload cap (DoS guard)
_COOKIE_LIST_MAX_LEN = 1000                     # max cookies per import

_VALID_SAMESITE = {"Lax": "Lax", "Strict": "Strict", "None": "None"}

def _is_ip_literal_domain(domain: str) -> bool:
    """True when ``domain`` is a literal IPv4 or IPv6 address."""
    if not domain:
        return False
    bare = domain.strip().lstrip(".")
    if bare.startswith("[") and bare.endswith("]"):
        bare = bare[1:-1]
    try:
        ipaddress.ip_address(bare)
    except ValueError:
        return False
    return True


def _cookie_url_host(url: str) -> str | None:
    """Return host for an http(s) cookie URL, or None when invalid."""
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        return None
    return parsed.hostname


def _parse_netscape(
    text: str, *, malformed: list[str] | None = None,
) -> list[dict]:
    """Parse Netscape ``cookies.txt`` format → Playwright-shaped dicts.

    Netscape format: ``domain\tincludeSubdomains\tpath\tsecure\texpires\tname\tvalue``
    Lines starting with ``#`` are comments and skipped, EXCEPT the special
    ``#HttpOnly_<domain>`` prefix which marks the cookie as HttpOnly.

    Malformed lines (wrong field count, non-numeric ``expires``) are
    silently skipped. When ``malformed`` is supplied, a sentinel string
    describing the reason is appended for each skipped line so the caller
    can surface a ``malformed_line`` drop count to the operator. This
    parser never raises and never echoes the bad line content (which may
    contain a cookie value).
    """
    result: list[dict] = []
    if not isinstance(text, str):
        return result
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        http_only = False
        if line.startswith("#HttpOnly_"):
            http_only = True
            line = line[len("#HttpOnly_"):]
        elif line.startswith("#"):
            continue
        # Fields: domain, includeSubdomains, path, secure, expires, name, value
        # value may itself contain tabs in some exporters, so don't split unbounded.
        parts = line.split("\t")
        if len(parts) < 7:
            if malformed is not None:
                malformed.append("field_count")
            continue
        domain, _include_sub, path, secure, expires, name, *value_parts = parts
        value = "\t".join(value_parts) if value_parts else ""
        try:
            expires_int = int(float(expires))
        except (ValueError, TypeError):
            if malformed is not None:
                malformed.append("expires")
            continue
        result.append({
            "name": name,
            "value": value,
            "domain": domain,
            "path": path or "/",
            "secure": secure.strip().upper() == "TRUE",
            "httpOnly": http_only,
            "expires": expires_int,
        })
    return result


def _validate_cookies(
    payload: Any, *, fmt: str | None = None,
) -> tuple[list[dict], list[dict], str | None]:
    """Validate + normalize an operator cookie payload.

    Returns ``(accepted, dropped, detected_format_or_None)``:
      • ``accepted`` — list of Playwright-shaped cookie dicts ready to
        push into ``BrowserContext.add_cookies``.
      • ``dropped`` — list of ``{reason, count}`` aggregated by reason.
      • ``detected_format_or_None`` — ``"playwright"`` | ``"netscape"`` | None.
        ``None`` signals an unrecognized shape — callers should return a
        400 ``invalid_input`` envelope.

    Pure function: no I/O, no mutation of inputs. Suitable for unit tests
    without HTTP plumbing.
    """
    detected: str | None = fmt
    items: list[dict] = []

    if detected is None:
        if isinstance(payload, list) and (
            not payload or isinstance(payload[0], dict)
        ):
            detected = "playwright"
        elif isinstance(payload, str) and "\t" in payload:
            detected = "netscape"

    drop_counts: dict[str, int] = {}

    def _drop(reason: str) -> None:
        drop_counts[reason] = drop_counts.get(reason, 0) + 1

    if detected == "playwright":
        if not isinstance(payload, list):
            return [], [], None
        items = list(payload)
    elif detected == "netscape":
        if not isinstance(payload, str):
            return [], [], None
        # Capture per-line malformed reasons so the operator sees a count
        # of dropped Netscape lines instead of a silent zero result.
        malformed_lines: list[str] = []
        items = _parse_netscape(payload, malformed=malformed_lines)
        for _ in malformed_lines:
            _drop("malformed_line")
    else:
        return [], [], None

    accepted: list[dict] = []

    for raw in items:
        if not isinstance(raw, dict):
            _drop("not_a_dict")
            continue
        name = raw.get("name", "")
        value = raw.get("value", "")
        domain = raw.get("domain", "")
        url = raw.get("url")
        path = raw.get("path", "/") or "/"
        if not isinstance(name, str) or not name:
            _drop("empty_name")
            continue
        # Bound the cookie name to a sane length. The wire spec doesn't
        # mandate a cap but Firefox/Playwright treat oversized names
        # poorly downstream; more importantly, names flow into the
        # audit log so an unbounded payload could pollute log
        # aggregation and disk usage.
        if len(name) > 256:
            _drop("invalid_name_length")
            continue
        if domain is None:
            domain = ""
        if not isinstance(domain, str):
            _drop("empty_domain")
            continue
        if url is None:
            url = ""
        if not isinstance(url, str):
            _drop("invalid_url")
            continue
        if domain and url:
            _drop("domain_url_conflict")
            continue
        if not domain and not url:
            _drop("empty_domain")
            continue
        if not isinstance(value, str):
            _drop("invalid_value_type")
            continue
        if len(value.encode("utf-8")) > _COOKIE_VALUE_MAX_BYTES:
            _drop("value_too_large")
            continue
        url_host: str | None = None
        if url:
            url_host = _cookie_url_host(url)
            if url_host is None:
                _drop("invalid_url")
                continue
            if _is_ip_literal_domain(url_host):
                _drop("ip_domain_unsupported")
                continue
        if domain and _is_ip_literal_domain(domain):
            _drop("ip_domain_unsupported")
            continue
        # RFC 6265bis: __Host- prefix forbids an explicit domain.
        if name.startswith("__Host-") and domain:
            _drop("host_prefix_with_domain")
            continue
        # Accept ``secure`` as a JSON bool, the integers ``0``/``1``
        # (Chrome devtools "Copy as cURL", older Chromium JSON exports,
        # Curl bridge tooling), or absent. When absent we infer secure
        # from the URL scheme — real-browser cookie exports often omit
        # the attribute when present in the source, and silently
        # downgrading every cookie to non-secure breaks cross-site auth
        # imports. NB: ``isinstance(x, bool)`` is checked BEFORE the
        # ``in (0, 1)`` branch because ``True/False`` are themselves
        # ``int`` subclasses.
        secure_raw = raw.get("secure")
        if secure_raw is None:
            secure = bool(url and url.lower().startswith("https://"))
        elif isinstance(secure_raw, bool):
            secure = secure_raw
        elif isinstance(secure_raw, int) and secure_raw in (0, 1):
            secure = bool(secure_raw)
        else:
            _drop("invalid_secure")
            continue
        # ``httpOnly`` accepts the same bool/0/1/absent shape as
        # ``secure``. Unlike ``secure``, we do NOT default-set it from
        # any URL property — httpOnly should be explicit so the operator
        # signals the import faithfully (a default-True would lock down
        # cookies that the source intended client-readable; a default-
        # False would accidentally expose session cookies).
        http_only_raw = raw.get("httpOnly")
        if http_only_raw is None:
            pass
        elif isinstance(http_only_raw, bool):
            pass
        elif isinstance(http_only_raw, int) and http_only_raw in (0, 1):
            http_only_raw = bool(http_only_raw)
        else:
            _drop("invalid_httponly")
            continue
        if name.startswith("__Host-"):
            if path != "/":
                _drop("host_prefix_invalid_path")
                continue
            if not secure:
                _drop("host_prefix_without_secure")
                continue
            # Spec: __Host- cookies are only set over a secure origin.
            # If the operator supplied url=http://..., Firefox will
            # silently drop the cookie at SET time. Reject up front
            # so the operator gets a deterministic drop reason instead
            # of an opaque "your import looked fine but nothing landed".
            if url and not url.lower().startswith("https://"):
                _drop("host_prefix_non_https_url")
                continue
        if name.startswith("__Secure-") and not secure:
            _drop("secure_prefix_without_secure")
            continue
        if name.startswith("__Secure-") and url and not url.lower().startswith("https://"):
            _drop("secure_prefix_non_https_url")
            continue
        # Per RFC 6265 §5.4, ``Secure=true`` requires an HTTPS origin.
        # Firefox silently drops a Secure cookie set with a plain-http
        # url at SET time. Reject up front so the operator gets a
        # deterministic drop reason rather than an opaque "your import
        # said success but nothing landed". Applies to ALL cookies
        # (not just ``__Secure-``/``__Host-`` prefixed names).
        if (
            secure
            and url
            and not url.lower().startswith("https://")
        ):
            _drop("secure_non_https_url")
            continue
        normalized: dict = {"name": name, "value": value}
        if url:
            normalized["url"] = url
            if path:
                normalized["path"] = path
        else:
            normalized["domain"] = domain
            normalized["path"] = path
        # SameSite normalize (case-insensitive) — drop entry on unknown.
        ss_raw = raw.get("sameSite")
        if ss_raw is not None:
            if not isinstance(ss_raw, str):
                _drop("invalid_samesite")
                continue
            ss_norm = _VALID_SAMESITE.get(ss_raw.strip().capitalize())
            if ss_norm is None:
                _drop("invalid_samesite")
                continue
            if ss_norm == "None" and not secure:
                _drop("samesite_none_without_secure")
                continue
            normalized["sameSite"] = ss_norm
        # expires: must be a finite number (int or float). Reject
        # bool (subclass of int), strings, NaN, ±Inf, and absurd far-
        # future values that would overflow int conversion or fail
        # downstream Playwright/Firefox validation. ~100 years out is a
        # generous upper bound; cookies aren't realistically dated past
        # that and bounding here protects the audit log too.
        if "expires" in raw and raw["expires"] is not None:
            exp = raw["expires"]
            if isinstance(exp, bool) or not isinstance(exp, (int, float)):
                _drop("invalid_expires")
                continue
            if not math.isfinite(exp):
                _drop("invalid_expires")
                continue
            _MAX_EXPIRES = 4102444800.0  # 2100-01-01 UTC, ~76yr buffer
            if exp < 0 or exp > _MAX_EXPIRES:
                _drop("invalid_expires")
                continue
            normalized["expires"] = int(exp)
        # httpOnly / secure — pass through only when known. ``secure``
        # was inferred above from the URL scheme when the source omitted
        # the attribute; emit it so the downstream Playwright/Firefox
        # call captures the resolved value rather than defaulting to
        # ``False`` again.
        if http_only_raw is not None:
            normalized["httpOnly"] = http_only_raw
        if secure_raw is not None or url:
            normalized["secure"] = secure
        accepted.append(normalized)

    dropped = [
        {"reason": reason, "count": count}
        for reason, count in sorted(drop_counts.items())
    ]
    return accepted, dropped, detected


def create_dashboard_router(
    blackboard: Blackboard,
    health_monitor: HealthMonitor | None,
    cost_tracker: CostTracker,
    trace_store: TraceStore | None,
    event_bus: EventBus | None,
    agent_registry: dict[str, str],
    mesh_port: int = 8420,
    # Optional subsystem dependencies (not all deployments include all subsystems)
    lane_manager: Any = None,
    cron_scheduler: Any = None,
    pubsub: Any = None,
    permissions: Any = None,
    credential_vault: Any = None,
    transport: Any = None,
    runtime: Any = None,
    router: Any = None,
    webhook_manager: Any = None,
    channel_manager: Any = None,
    wallet_service_ref: list | None = None,
    api_key_manager: Any = None,
    # Task 9 — Workplace tab + pending action review surface. Both are
    # optional so existing dashboard tests that don't construct them
    # keep working; the new endpoints fall back to empty/disabled when
    # the relevant store is absent.
    pending_actions: Any = None,
    tasks_store: Any = None,
    # Open help-requests registry (credential / browser-login / captcha asks)
    # backing the "Needs you" feed. Optional so existing tests keep working;
    # when absent the credential-save / complete paths fall back to their
    # legacy direct steer+emit without an in-process registry pop.
    help_requests_store: Any = None,
    # PR-A — Work summaries backing the new Work-tab default landing.
    # Optional so existing dashboard tests that don't construct one keep
    # working; the endpoints return ``{enabled: False}`` when absent.
    summaries_store: Any = None,
    # Session observability (Phase 2) — verbatim-intent store. The dashboard
    # chat endpoints call the agent DIRECTLY (not via the lane's
    # ``_direct_dispatch``, the only other capture point), so this is the sole
    # intent-capture surface for the primary human UI. Optional so existing
    # tests keep working; capture is skipped when absent.
    intent_store: Any = None,
    # Phase -1 onboarding wizard — tracks first-visit activation. Optional
    # so existing tests that don't pass it keep working; we lazy-init a
    # default :class:`DashboardTelemetry` against ``data/telemetry.db``
    # if the caller didn't supply one.
    telemetry: Any = None,
    # Per-session "opened conversations" store (Phase 1 unified messenger).
    # Replaces the previous module-level ``set[str]`` which leaked between
    # concurrent users in multi-tenant SSO deployments. Auto-instantiated
    # when omitted so existing test setups keep working.
    opened_conversations_store: Any = None,
    # Fleet MCP connector catalog (System → Connectors). The same
    # instance must be wired into the runtime backend via
    # ``set_connector_store`` so catalog edits apply on agent restart.
    connector_store: Any = None,
    # Mesh-side gateway for remote (http) connectors: powers the probe
    # ("Test connection") endpoint and the discovery-cache invalidation
    # on auth-only edits (plan D12).
    mcp_gateway: Any = None,
    # THE team authority (src/host/teams.py). The runtime passes the
    # same disk-backed instance the mesh app holds; standalone test
    # constructions fall back to a pure-DB in-memory store.
    teams_store: Any = None,
) -> APIRouter:
    """Create the dashboard FastAPI router."""
    if teams_store is None:
        from src.host.teams import TeamStore
        teams_store = TeamStore(db_path=":memory:")
    # Lazy-init telemetry sink so callers (mesh CLI, tests) can opt out by
    # passing ``telemetry=None`` after explicitly setting it. We keep the
    # explicit-None contract by only auto-creating when the kwarg is
    # missing entirely. Tests that need an in-memory DB pass their own.
    if telemetry is None:
        from src.dashboard.telemetry import DashboardTelemetry as _DashboardTelemetry
        try:
            telemetry = _DashboardTelemetry()
        except Exception as _e:
            logger.warning("Failed to init dashboard telemetry: %s", _e)
            telemetry = None
    # Auto-instantiate the opened-conversations store when the caller
    # hasn't supplied one. Replaces the prior module-level ``set[str]``
    # so opened workers are scoped to the operator's ``ol_session``
    # cookie hash and survive process restarts.
    if opened_conversations_store is None:
        try:
            from src.dashboard.conversations import OpenedConversationsStore
            opened_conversations_store = OpenedConversationsStore()
        except Exception as e:
            logger.warning("Failed to instantiate OpenedConversationsStore: %s", e)
            opened_conversations_store = None
    # Plan limits — read once at startup; provisioner restarts engine after updating .env
    # 0 = unlimited (self-hosted / open-source) unless env var is explicitly set to 0
    _max_agents = int(os.environ.get("OPENLEGION_MAX_AGENTS", "0"))
    # Plan-tier cap — when OPENLEGION_MAX_TEAMS is set to 0, team
    # creation is disabled entirely.
    _max_teams = int(os.environ.get("OPENLEGION_MAX_TEAMS", "0"))
    _teams_disabled = (
        _max_teams == 0 and "OPENLEGION_MAX_TEAMS" in os.environ
    )

    def _emit_team_event(event_type: str, agent: str, data: dict) -> None:
        """Emit a team lifecycle event (``team_*``) on the bus.

        Payload is augmented with a ``team_name`` key mirroring
        ``team_id`` / ``name`` so subscribers get a uniform shape.
        """
        if event_bus is None:
            return
        payload = dict(data)
        name_value = payload.get("name") or payload.get("team_id")
        if name_value and "team_name" not in payload:
            payload["team_name"] = name_value
        try:
            event_bus.emit(event_type, agent=agent, data=payload)
        except Exception as e:
            logger.debug("%s emit failed: %s", event_type, e)

    def _emit_config_changed(scope: str, **extra: Any) -> None:
        """Emit a generic ``config_changed`` event for a System-tab panel.

        One discriminated event covers every settings/integrations/storage
        mutation: ``data.scope`` tells the SPA which panel to re-fetch (e.g.
        ``browser_settings`` / ``channels`` / ``webhooks`` / ``integrations``).
        Best-effort — never raises into the request handler.
        """
        if event_bus is None:
            return
        try:
            event_bus.emit("config_changed", data={"scope": scope, **extra})
        except Exception as e:
            logger.debug("config_changed emit failed (scope=%s): %s", scope, e)

    async def _csrf_check(request: Request) -> None:
        """Require X-Requested-With header on state-changing requests.

        Browsers block custom headers on cross-origin requests (CORS preflight),
        so this prevents CSRF attacks on cookie-authenticated endpoints.
        GET/HEAD/OPTIONS are exempt (safe methods).
        """
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return
        if not request.headers.get("X-Requested-With"):
            raise HTTPException(403, "Missing X-Requested-With header")

    # ── OAuth integrations (connect/callback) state + helpers ───────────
    from src.host.oauth_state import OAuthStateStore
    _oauth_state_store = OAuthStateStore()

    def _oauth_session_hash(request: Request) -> str:
        """Bind OAuth state to the caller's dashboard session (cookie hash).

        Mirrors :func:`_conversations_session_id`. Falls back to a dev constant
        when the ``ol_session`` cookie is absent (single-operator self-hosted).
        """
        raw = request.cookies.get("ol_session", "")
        if not raw:
            return "dev:operator"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _public_base_url(request: Request) -> str:
        """Externally-reachable origin for building OAuth redirect URIs.

        Prefers ``OPENLEGION_PUBLIC_BASE_URL`` (exact, what the operator
        registered with the provider); otherwise derives from the forwarded
        Host/proto Caddy sets. The redirect URI must match byte-for-byte across
        the authorize call, the token exchange, and the provider registration.
        """
        explicit = os.environ.get("OPENLEGION_PUBLIC_BASE_URL", "").strip()
        if explicit:
            return explicit.rstrip("/")
        proto = request.headers.get("x-forwarded-proto") or request.url.scheme
        host = (
            request.headers.get("x-forwarded-host")
            or request.headers.get("host")
            or request.url.netloc
        )
        return f"{proto}://{host}"

    def _oauth_redirect_uri(request: Request, provider_key: str) -> str:
        return f"{_public_base_url(request)}/dashboard/integrations/{provider_key}/callback"

    def _mask_proxy_url(url: str) -> str:
        """Mask credentials in a proxy URL for display."""
        if not url:
            return ""
        from urllib.parse import urlparse, urlunparse
        try:
            parsed = urlparse(url)
            if parsed.username:
                masked_netloc = f"{parsed.username[:2]}***@{parsed.hostname}:{parsed.port}"
                return urlunparse((parsed.scheme, masked_netloc, "", "", "", ""))
            return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}" if parsed.port else url
        except Exception:
            return "***"

    async def _push_browser_proxy_for_agent(agent_id: str) -> None:
        """Push proxy config to browser service for an agent after restart."""
        if not runtime or not hasattr(runtime, "browser_service_url") or not runtime.browser_service_url:
            return
        try:
            from src.cli.config import _load_config
            from src.cli.proxy import parse_proxy_url, resolve_agent_proxy
            _cfg = _load_config()
            proxy_url = resolve_agent_proxy(agent_id, _cfg.get("agents", {}), _cfg.get("network", {}))
            if proxy_url:
                parsed = parse_proxy_url(proxy_url)
                if parsed:
                    body = {"url": parsed["url"], "username": parsed["username"], "password": parsed["password"]}
                else:
                    body = {}  # explicit no-proxy
            else:
                body = {}  # explicit no-proxy (direct mode or no system proxy)
            headers: dict = {}
            svc_token = getattr(runtime, "browser_auth_token", "")
            if svc_token:
                headers["Authorization"] = f"Bearer {svc_token}"
            resp = await _dashboard_browser_client.put(
                f"{runtime.browser_service_url}/browser/{agent_id}/proxy",
                json=body, headers=headers,
            )
            if resp.status_code >= 400:
                logger.warning("Browser proxy push for %s returned %d", agent_id, resp.status_code)
        except Exception as e:
            logger.warning("Failed to push browser proxy for %s: %s", agent_id, e)

    api_router = APIRouter(
        prefix="/dashboard",
        dependencies=[Depends(_verify_dashboard_auth), Depends(_csrf_check)],
    )

    # Per-platform success aggregator — subscribes to EventBus emits and
    # maintains a 24h rolling window of captcha outcomes / fingerprint
    # burns / pre-nav dwells per host.  Instantiated even when
    # ``event_bus`` is None (some dashboards run without a bus) so the
    # GET endpoint can still return an empty snapshot rather than
    # 500'ing.
    from src.dashboard.platform_success import PlatformSuccessAggregator
    platform_success = PlatformSuccessAggregator()
    if event_bus is not None:
        event_bus.add_listener(platform_success.handle_event)

    jinja_env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )

    # Build valid-models set for validation (dynamic from litellm)
    from src.cli.config import _PROVIDER_MODELS
    from src.shared.models import get_provider_models

    def _is_valid_model(model: str) -> bool:
        """Check if a model is known (from litellm or featured lists).

        Ollama models are always accepted since they're user-installed locally.
        Custom LLM providers registered via settings.json are also accepted.
        """
        provider = model.split("/")[0] if "/" in model else ""
        if provider in ("ollama", "ollama_chat"):
            return True
        settings = _load_settings()
        custom_providers = settings.get("custom_llm_providers", {})
        provider_lower = provider.lower()
        if provider_lower in custom_providers:
            custom_models = custom_providers[provider_lower].get("models", [])
            return model.lower() in [m.lower() for m in custom_models]
        if provider:
            return model in get_provider_models(provider)
        return any(model in models for models in _PROVIDER_MODELS.values())

    # ── SPA entry point ──────────────────────────────────────

    @api_router.get("/", response_class=HTMLResponse)
    async def dashboard_index() -> HTMLResponse:
        from src.shared.models import KEYLESS_PROVIDERS, get_all_providers
        all_providers = get_all_providers()
        template = jinja_env.get_template("index.html")
        html = template.render(
            ws_path="/ws/events",
            api_base="/dashboard/api",
            v=ASSET_VERSION,
            providers=[p for p in all_providers if p["name"] not in KEYLESS_PROVIDERS],
            all_providers=all_providers,
        )
        return HTMLResponse(html, headers={
            "Cache-Control": "no-store",
            # M18: frame-ancestors 'self' + X-Frame-Options SAMEORIGIN guard
            # against clickjacking. MUST be 'self'/SAMEORIGIN, NOT 'none'/DENY:
            # the dashboard embeds the per-agent VNC viewer in a same-origin
            # iframe via /agent-vnc/, so 'none' would break the viewer.
            "X-Frame-Options": "SAMEORIGIN",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline'; "
                "connect-src 'self'; "
                "frame-src 'self'; "
                "frame-ancestors 'self'; "
                "object-src 'none'"
            ),
        })

    def _browser_vnc_url_for_request(
        request: Request, agent_id: str,
    ) -> str | None:
        """Build the agent-scoped browser VNC URL.

        Returns ``/agent-vnc/{agent_id}/index.html?path=agent-vnc/{agent_id}/websockify&...``
        — same-origin so it works through HTTPS via the reverse proxy
        without exposing extra ports. The mesh forwards this to the
        browser service which looks up the agent's allocated KasmVNC
        port (display_allocator) and proxies onward, so each agent's
        iframe streams only that agent's framebuffer.

        Returns ``None`` if the browser service hasn't been initialized.
        """
        # Readiness gate: ``browser_service_url`` is set as soon as the
        # FastAPI ``/browser/status`` health probe passes — the right
        # "is the service up" signal. The legacy ``browser_vnc_url``
        # was an alias for ``KasmVNC :6080`` which no longer exists.
        if not runtime or not getattr(runtime, "browser_service_url", None):
            return None

        forwarded_proto = request.headers.get("x-forwarded-proto")
        host = request.headers.get("host", "127.0.0.1:8420")
        scheme = forwarded_proto or "http"
        query = (
            "autoconnect=true"
            "&reconnect=true"
            "&reconnect_delay=2000"
            f"&path=agent-vnc/{agent_id}/websockify"
            "&resize=scale"
            "&quality=7"
            "&enable_perf_stats=0"
        )
        return (
            f"{scheme}://{host}/agent-vnc/{agent_id}/index.html?{query}"
        )

    async def _fetch_active_browser_agents() -> set[str]:
        """Return the set of agent_ids whose browser is currently up.

        Single fan-out to ``/browser/status``; the response includes
        ``agents: [...]`` listing every running instance. Used by the
        dashboard's poll path to gate iframe rendering — without this,
        an agent whose browser stopped (idle timeout, agent reset)
        keeps a live ``vnc_url`` in the payload, the iframe stays
        bound, and noVNC retries forever against a 503'ing endpoint.

        Soft-fails to an empty set on error; the caller treats that
        as "no agents have browsers", which is the safe default
        (button stays in idle state, no iframe).
        """
        if not runtime or not getattr(runtime, "browser_service_url", None):
            return set()
        token = getattr(runtime, "browser_auth_token", "")
        import httpx
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(
                    f"{runtime.browser_service_url}/browser/status",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code != 200:
                    return set()
                return set(resp.json().get("agents", []))
        except Exception:
            return set()

    # ── Fleet overview ───────────────────────────────────────

    @api_router.get("/api/agents")
    async def api_agents(request: Request) -> dict:
        from src.cli.config import _load_config
        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})
        default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")

        health_list = health_monitor.get_status() if health_monitor else []
        health_map = {h["agent"]: h for h in health_list}
        cost_list = cost_tracker.get_all_agents_spend("today")
        cost_map = {c["agent"]: c for c in cost_list}
        # Accurate "last worked" = timestamp of the agent's most recent LLM
        # call (from the usage ledger), distinct from the health probe's
        # "last seen". One query for the whole fleet.
        worked_map = cost_tracker.get_all_agents_last_worked()

        agent_teams = teams_store.agent_team_map()

        # One fan-out per fleet poll. Drives ``browser_running`` so the
        # frontend can tear down the iframe when an agent's browser
        # stops mid-view (idle timeout, reset) instead of letting noVNC
        # retry forever against a 503'ing endpoint.
        active_browsers = await _fetch_active_browser_agents()

        agents = []
        for agent_id, url in agent_registry.items():
            h = health_map.get(agent_id, {})
            c = cost_map.get(agent_id, {})
            acfg = agents_cfg.get(agent_id, {})
            # PR-L' — surface the most recent task event timestamp so
            # the agent card can render "Last task Nm ago" alongside
            # "Last seen" (health probe). Best-effort: ``None`` when
            # tasks_v2 is disabled or the agent has no events yet.
            last_task_ts: float | None = None
            if tasks_store is not None:
                try:
                    last_task_ts = tasks_store.last_event_ts_for_agent(agent_id)
                except Exception:
                    last_task_ts = None
            entry = {
                "id": agent_id,
                "url": url,
                "running": True,
                "over_limit": False,
                "health_status": h.get("status", "unknown"),
                "failures": h.get("failures", 0),
                "restarts": h.get("restarts", 0),
                "last_check": h.get("last_check", 0),
                "last_healthy": h.get("last_healthy", 0),
                "last_task_event_ts": last_task_ts,
                "last_worked_ts": worked_map.get(agent_id),
                "daily_cost": c.get("cost", 0),
                "daily_tokens": c.get("tokens", 0),
                "role": acfg.get("role", ""),
                "model": acfg.get("model", default_model),
                "avatar": acfg.get("avatar", 1),
                "color": acfg.get("color"),
                "team": agent_teams.get(agent_id),
            }
            if cron_scheduler is not None:
                hb = cron_scheduler.find_heartbeat_job(agent_id)
                if hb:
                    entry["heartbeat_job_id"] = hb.id
                    entry["heartbeat_schedule"] = hb.schedule
                    entry["heartbeat_enabled"] = hb.enabled
                    entry["heartbeat_next_run"] = hb.next_run
            vnc_url = _browser_vnc_url_for_request(request, agent_id)
            if vnc_url:
                entry["vnc_url"] = vnc_url
            entry["browser_running"] = agent_id in active_browsers
            agents.append(entry)

        # Append over-limit agents from config that are not running
        for agent_id, acfg in agents_cfg.items():
            if agent_id not in agent_registry:
                entry = {
                    "id": agent_id,
                    "url": None,
                    "running": False,
                    "over_limit": True,
                    "health_status": "stopped",
                    "failures": 0,
                    "restarts": 0,
                    "last_check": 0,
                    "last_healthy": 0,
                    "last_task_event_ts": None,
                    # Sourced from the persistent usage ledger, so a stopped /
                    # over-limit agent still shows when it last actually worked.
                    "last_worked_ts": worked_map.get(agent_id),
                    "daily_cost": 0,
                    "daily_tokens": 0,
                    "role": acfg.get("role", ""),
                    "model": acfg.get("model", default_model),
                    "avatar": acfg.get("avatar", 1),
                    "color": acfg.get("color"),
                    "team": agent_teams.get(agent_id),
                }
                agents.append(entry)

        return {"agents": agents}

    import httpx as _httpx
    _dashboard_browser_client = _httpx.AsyncClient(timeout=10)

    async def _push_browser_settings() -> None:
        """Push saved browser speed/delay to the browser service.

        Called after browser service (re)start so persisted settings
        survive container restarts.  Failures are silently logged —
        the browser service will simply use its defaults.
        """
        if not runtime or not getattr(runtime, 'browser_service_url', ''):
            return
        settings = _load_settings()
        payload: dict = {}
        speed = settings.get("browser_speed")
        if speed is not None:
            payload["speed"] = speed
        delay = settings.get("browser_delay")
        if delay is not None:
            payload["delay"] = delay
        if not payload:
            return
        try:
            browser_auth = getattr(runtime, 'browser_auth_token', '')
            headers = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            await _dashboard_browser_client.post(
                f"{runtime.browser_service_url}/browser/settings",
                json=payload,
                headers=headers,
            )
        except Exception as e:
            logger.debug("Failed to push browser settings on startup: %s", e)

    @api_router.post("/api/browser/{agent_id}/focus")
    async def api_browser_focus(agent_id: str, request: Request) -> dict:
        """Tell the browser service to bring this agent's browser to foreground."""
        if not runtime or not hasattr(runtime, 'browser_service_url') or not runtime.browser_service_url:
            raise HTTPException(503, "Browser service not available")
        try:
            browser_auth = getattr(runtime, 'browser_auth_token', '')
            headers = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            resp = await _dashboard_browser_client.post(
                f"{runtime.browser_service_url}/browser/{agent_id}/focus",
                json={},
                headers=headers,
                timeout=60,  # Cold-start can take 20-30s (Camoufox + geoip)
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    @api_router.post("/api/browser/{agent_id}/control")
    async def api_browser_control(agent_id: str, request: Request) -> dict:
        """Toggle user browser control — pauses agent X11 input."""
        if not runtime or not hasattr(runtime, 'browser_service_url') or not runtime.browser_service_url:
            raise HTTPException(503, "Browser service not available")
        body = await request.json()
        try:
            browser_auth = getattr(runtime, 'browser_auth_token', '')
            headers: dict[str, str] = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            resp = await _dashboard_browser_client.post(
                f"{runtime.browser_service_url}/browser/{agent_id}/control",
                json=body,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    @api_router.post("/api/browser/{agent_id}/reset")
    async def api_browser_reset(agent_id: str) -> dict:
        """Reset an agent's browser session (close and relaunch with current config)."""
        if agent_id not in agent_registry:
            raise HTTPException(404, "Agent not found")
        if not runtime or not hasattr(runtime, 'browser_service_url') or not runtime.browser_service_url:
            raise HTTPException(503, "Browser service not available")
        try:
            browser_auth = getattr(runtime, 'browser_auth_token', '')
            headers = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            resp = await _dashboard_browser_client.post(
                f"{runtime.browser_service_url}/browser/{agent_id}/reset",
                json={},
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Browser reset failed for '%s': %s", agent_id, e)
            raise HTTPException(500, "Browser reset failed")

    # ── Phase 6 §9.2 — operator cookie/session import ────────────────────
    # In-memory rate limiter keyed by (operator_session_user, agent_id).
    # 60-min sliding window; capacity 10. Limiter state lives on this
    # router instance — restarting the engine resets the window which is
    # the desired behavior (operators rarely depend on the precise number
    # surviving a process boundary).
    _COOKIE_IMPORT_LIMIT = 60
    _COOKIE_IMPORT_WINDOW_S = 60 * 60
    _cookie_import_buckets: dict[tuple[str, str], list[float]] = {}
    _cookie_import_lock = threading.Lock()

    def _cookie_import_check_rate_limit(
        operator: str, agent_id: str,
    ) -> tuple[bool, int]:
        """Sliding-window rate limit. Returns (allowed, retry_after_ms).

        ``retry_after_ms`` is 0 when ``allowed`` is True; otherwise it is
        the time until the oldest event in the window expires.
        """
        import time as _time
        now = _time.time()
        cutoff = now - _COOKIE_IMPORT_WINDOW_S
        key = (operator, agent_id)
        with _cookie_import_lock:
            bucket = _cookie_import_buckets.setdefault(key, [])
            # Drop entries outside the window in-place.
            bucket[:] = [t for t in bucket if t > cutoff]
            if len(bucket) >= _COOKIE_IMPORT_LIMIT:
                oldest = bucket[0]
                retry_after_ms = max(
                    1, int((oldest + _COOKIE_IMPORT_WINDOW_S - now) * 1000),
                )
                return False, retry_after_ms
            bucket.append(now)
            # Periodic sweep of stale buckets so the dict doesn't grow
            # unbounded across deleted/rotated agents. Cheap because we
            # only sweep when the table grows past a threshold; the
            # sliding window is the only state we need to retain.
            if len(_cookie_import_buckets) > 1024:
                stale = [
                    k for k, b in _cookie_import_buckets.items()
                    if not b or b[-1] <= cutoff
                ]
                for k in stale:
                    _cookie_import_buckets.pop(k, None)
            return True, 0

    def _cookie_envelope_err(
        code: str, message: str, retry_after_ms: int | None = None,
    ) -> dict:
        """Build a §2.3 error envelope for cookie-import responses."""
        return {
            "success": False,
            "error": {
                "code": code,
                "message": message,
                "retry_after_ms": retry_after_ms,
            },
        }

    def _operator_session_id(request: Request) -> str:
        """Derive a stable per-session identifier for rate-limiting + audit.

        The ``ol_session`` cookie is an HMAC of an expiry; there is no
        per-user identity (the engine has a single operator). We hash the
        cookie value so the audit log records a stable identifier without
        echoing the cookie itself. Empty cookie (dev mode) → "operator".
        """
        raw = request.cookies.get("ol_session", "")
        if not raw:
            return "operator"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
        return f"operator:{digest}"

    @api_router.post("/api/agents/{agent_id}/browser/import_cookies")
    async def api_browser_import_cookies(
        agent_id: str, request: Request,
    ) -> dict:
        """Phase 6 §9.2 — operator-only cookie/session import.

        Merge semantics: Playwright's ``BrowserContext.add_cookies``
        MERGES with existing cookies. A (name, domain, path) collision
        OVERWRITES the existing cookie; non-colliding cookies are
        appended. There is no "wipe and replace" — to start clean,
        reset the agent profile first.

        At-rest leak warning (see plan §13 risk register row "Imported
        cookies at rest in profile (plaintext)"): Firefox stores cookies
        in ``cookies.sqlite`` UNENCRYPTED inside the agent profile. This
        endpoint does NOT silently mitigate that — operators handling
        high-value sessions are expected to use encrypted volumes or
        ephemeral profiles. Audit log records domain + cookie name ONLY.
        """
        # Choice (per plan §9.2): we hit the browser service directly via
        # `runtime.browser_service_url` rather than going through
        # /mesh/browser/command. Rationale: this is operator-only — adding
        # it to KNOWN_BROWSER_ACTIONS would expose it to agents.
        from src.browser import flags as _flags

        operator = _operator_session_id(request)

        if agent_id not in agent_registry:
            raise HTTPException(404, "Agent not found")

        # Kill-switch ─ honor BROWSER_COOKIE_IMPORT_DISABLED.
        if _flags.get_bool("BROWSER_COOKIE_IMPORT_DISABLED", False):
            raise HTTPException(
                403,
                detail=_cookie_envelope_err(
                    "forbidden",
                    "Cookie import disabled by operator policy",
                ),
            )

        # Rate-limit pre-check (10/hr per (operator, agent_id)).
        allowed, retry_ms = _cookie_import_check_rate_limit(operator, agent_id)
        if not allowed:
            raise HTTPException(
                429,
                detail=_cookie_envelope_err(
                    "conflict",
                    "rate limit exceeded",
                    retry_after_ms=retry_ms,
                ),
            )

        # Read raw body once. Enforce 256 KiB DoS guard before JSON parse
        # so we don't allocate a fat string for an attacker payload.
        raw = await request.body()
        if len(raw) > _COOKIE_PAYLOAD_MAX_BYTES:
            raise HTTPException(
                413,
                detail=_cookie_envelope_err(
                    "size_limit",
                    f"payload exceeds {_COOKIE_PAYLOAD_MAX_BYTES} bytes",
                ),
            )

        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HTTPException(
                400,
                detail=_cookie_envelope_err(
                    "invalid_input", "request body is not valid JSON",
                ),
            )

        if not isinstance(body, dict):
            raise HTTPException(
                400,
                detail=_cookie_envelope_err(
                    "invalid_input",
                    "request body must be a JSON object",
                ),
            )

        # Format detection: explicit ``format`` field overrides auto-detect.
        # Payload may live under ``cookies`` (playwright list or netscape
        # text) — also accept the bare list/string for back-compat with
        # the simpler curl-style input.
        explicit_fmt = body.get("format")
        if explicit_fmt is not None and explicit_fmt not in ("playwright", "netscape"):
            raise HTTPException(
                400,
                detail=_cookie_envelope_err(
                    "invalid_input",
                    "format must be 'playwright' or 'netscape'",
                ),
            )

        payload = body.get("cookies", body)
        if isinstance(payload, dict) and "cookies" not in payload:
            # Caller passed the entire body as the cookies field by accident.
            raise HTTPException(
                400,
                detail=_cookie_envelope_err(
                    "invalid_input", "missing 'cookies' field",
                ),
            )

        accepted, dropped, detected_fmt = _validate_cookies(
            payload, fmt=explicit_fmt,
        )
        if detected_fmt is None:
            raise HTTPException(
                400,
                detail=_cookie_envelope_err(
                    "invalid_input",
                    "unable to detect cookie format — supply 'format' or use "
                    "Playwright JSON list / Netscape TSV string",
                ),
            )

        # List-length cap applied AFTER parsing (Netscape format may have
        # supplied >1000 lines).
        if len(accepted) + sum(d["count"] for d in dropped) > _COOKIE_LIST_MAX_LEN:
            raise HTTPException(
                413,
                detail=_cookie_envelope_err(
                    "size_limit",
                    f"more than {_COOKIE_LIST_MAX_LEN} cookies in payload",
                ),
            )

        # Audit log — domain + name only. NEVER cookie values.
        # _COOKIE_AUDIT_VALUE_EXCLUSION_GUARANTEE: the keys passed to
        # the logger here are an explicit allowlist; cookie ``value`` is
        # NOT included by construction. A unit test (test_audit_log_no_value)
        # asserts this property holds against a corpus including JWT,
        # Bearer, and SigV4-shaped values.
        domains = sorted({c["domain"] for c in accepted if c.get("domain")})
        names = sorted({c["name"] for c in accepted if c.get("name")})
        _cookie_audit_logger.info(
            "cookie_import operator=%s agent_id=%s count=%d domains=%s "
            "names=%s format=%s dropped=%s",
            operator, agent_id, len(accepted),
            json.dumps(domains), json.dumps(names),
            detected_fmt, json.dumps(dropped),
        )
        if blackboard is not None:
            try:
                blackboard.log_audit(
                    action="cookie_import",
                    target=agent_id,
                    actor=operator,
                    field=detected_fmt,
                    after_value=json.dumps({
                        "count": len(accepted),
                        "domains": domains,
                        "names": names,
                        "dropped": dropped,
                    }),
                )
            except Exception as e:
                logger.warning("Failed to write cookie_import audit row: %s", e)

        # Push to browser service.
        if not runtime or not getattr(runtime, "browser_service_url", ""):
            raise HTTPException(
                503,
                detail=_cookie_envelope_err(
                    "service_unavailable", "Browser service not available",
                ),
            )
        try:
            browser_auth = getattr(runtime, "browser_auth_token", "")
            headers: dict[str, str] = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            resp = await _dashboard_browser_client.post(
                f"{runtime.browser_service_url}/browser/{agent_id}/import_cookies",
                json={"cookies": accepted},
                headers=headers,
                timeout=30,
            )
        except Exception as e:
            logger.warning("Cookie import push to browser service failed: %s", e)
            raise HTTPException(
                503,
                detail=_cookie_envelope_err(
                    "service_unavailable",
                    "Browser service unreachable",
                ),
            )
        if resp.status_code >= 500:
            raise HTTPException(
                503,
                detail=_cookie_envelope_err(
                    "service_unavailable",
                    f"Browser service returned {resp.status_code}",
                ),
            )
        try:
            svc_payload = resp.json()
        except Exception:
            raise HTTPException(
                503,
                detail=_cookie_envelope_err(
                    "service_unavailable",
                    "Browser service returned non-JSON response",
                ),
            )
        # Browser service may have its own dropped/imported counts; merge.
        if not svc_payload.get("success"):
            err = svc_payload.get("error") or {}
            return {
                "success": False,
                "error": {
                    "code": err.get("code", "service_unavailable"),
                    "message": err.get("message", "import failed"),
                    "retry_after_ms": err.get("retry_after_ms"),
                },
            }
        svc_data = svc_payload.get("data") or {}
        return {
            "success": True,
            "data": {
                "imported": svc_data.get("imported", len(accepted)),
                "dropped": dropped,
                "format": detected_fmt,
            },
        }

    @api_router.get("/api/agents/{agent_id}/browser/metrics")
    async def api_agent_browser_metrics(
        agent_id: str, since: int = 0,
    ) -> dict:
        """Return per-agent browser-metrics history (Phase 7 §10.1).

        Surfaces the per-minute aggregates already collected by
        :meth:`BrowserManager._emit_metrics` (§4.6) — click_success/fail,
        snapshot p50/p95, nav timeouts, rolling click-success-rate. Read-only,
        agent-scoped slice of ``/browser/metrics?since=<seq>``.

        Pagination via ``since=<seq>``: client passes back the response's
        ``current_seq`` on the next call so only new payloads are returned.
        Buffer is bounded (1024 entries service-wide) so a long-idle dashboard
        may miss intermediate payloads — that is by design (§2.7 forbids
        per-call events; an hour of history per agent is more than the panel
        renders anyway).

        On error returns a §2.3 envelope: ``{success, error: {code, message,
        retry_after_ms}}``. Per-call events are forbidden — this endpoint only
        surfaces aggregates.
        """
        if agent_id not in agent_registry:
            raise HTTPException(404, "Agent not found")
        if (
            not runtime
            or not hasattr(runtime, "browser_service_url")
            or not runtime.browser_service_url
        ):
            # Service unavailable rather than 404 — the agent exists, the
            # browser service is just not configured / reachable. Use the
            # error envelope so the panel can render a "service down" state.
            return {
                "success": False,
                "error": {
                    "code": "service_unavailable",
                    "message": "Browser service not available",
                    "retry_after_ms": None,
                },
            }
        try:
            since_seq = max(0, int(since))
        except (TypeError, ValueError):
            since_seq = 0

        result = await _fetch_browser_metrics_upstream(
            _dashboard_browser_client,
            runtime.browser_service_url,
            getattr(runtime, "browser_auth_token", ""),
            since_seq,
        )
        if not result.get("success"):
            return result
        data = result["data"]

        # Filter to this agent only — the upstream endpoint returns
        # service-wide aggregates with one entry per (agent, minute, kind).
        # ``current_seq`` is preserved as the high-water mark so pagination
        # works even when no new payloads belong to this agent.
        all_metrics = data.get("metrics") or []
        agent_metrics = [
            p for p in all_metrics if p.get("agent_id") == agent_id
        ]
        return {
            "success": True,
            "current_seq": int(data.get("current_seq", since_seq)),
            "boot_id": data.get("boot_id") or "",
            "metrics": agent_metrics,
        }

    @api_router.get("/api/agents/{agent_id}/session")
    async def api_agent_session_info(agent_id: str) -> dict:
        """Phase 10 §20 — return the privacy-safe session-sidecar summary.

        Proxies to the browser service's ``/browser/{agent_id}/session``
        endpoint, which returns counts only — no cookie values, no origin
        domains. Operators see whether a session is persisted and roughly
        how many cookies / origins it holds, without learning which sites
        the agent is logged into through the dashboard event stream.

        Returns the §2.3 success/error envelope so the panel can render a
        "service down" state without conflating it with "no session".
        """
        if agent_id not in agent_registry:
            raise HTTPException(404, "Agent not found")
        if (
            not runtime
            or not hasattr(runtime, "browser_service_url")
            or not runtime.browser_service_url
        ):
            return {
                "success": False,
                "error": {
                    "code": "service_unavailable",
                    "message": "Browser service not available",
                    "retry_after_ms": None,
                },
            }
        try:
            browser_auth = getattr(runtime, "browser_auth_token", "")
            headers = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            resp = await _dashboard_browser_client.get(
                f"{runtime.browser_service_url}/browser/{agent_id}/session",
                headers=headers,
            )
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": {
                        "code": "upstream_error",
                        "message": f"Browser service returned {resp.status_code}",
                        "retry_after_ms": None,
                    },
                }
            return {"success": True, "data": resp.json()}
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "service_unavailable",
                    "message": str(e),
                    "retry_after_ms": None,
                },
            }

    @api_router.delete("/api/agents/{agent_id}/session")
    async def api_agent_session_clear(agent_id: str) -> dict:
        """Phase 10 §20 — clear the persisted session sidecar.

        The router-level CSRF guard (``_csrf_check`` requires
        ``X-Requested-With``) gates this endpoint, so a cookie-only CSRF
        attempt cannot trigger a session wipe. The live BrowserContext
        is intentionally NOT closed here — call ``/browser/{agent_id}/reset``
        as the next step if the operator wants a fully fresh state.
        """
        if agent_id not in agent_registry:
            raise HTTPException(404, "Agent not found")
        if (
            not runtime
            or not hasattr(runtime, "browser_service_url")
            or not runtime.browser_service_url
        ):
            return {
                "success": False,
                "error": {
                    "code": "service_unavailable",
                    "message": "Browser service not available",
                    "retry_after_ms": None,
                },
            }
        try:
            browser_auth = getattr(runtime, "browser_auth_token", "")
            headers = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            resp = await _dashboard_browser_client.delete(
                f"{runtime.browser_service_url}/browser/{agent_id}/session",
                headers=headers,
            )
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": {
                        "code": "upstream_error",
                        "message": f"Browser service returned {resp.status_code}",
                        "retry_after_ms": None,
                    },
                }
            return {"success": True, "data": resp.json().get("data", {})}
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "service_unavailable",
                    "message": str(e),
                    "retry_after_ms": None,
                },
            }

    @api_router.get("/api/agents/{agent_id}/fingerprint-health")
    async def api_agent_fingerprint_health(agent_id: str) -> dict:
        """Phase 10 §22 — return the per-agent fingerprint health summary.

        Proxies to the browser service's
        ``/browser/{agent_id}/fingerprint-health`` endpoint, which returns
        only ``{window_size, rejection_rate, burned, last_signal_ts}`` —
        no URL / origin / cookie data leaks here.

        Returns the §2.3 success/error envelope so the panel can render a
        "service down" state without conflating it with "no signal".
        """
        if agent_id not in agent_registry:
            raise HTTPException(404, "Agent not found")
        if (
            not runtime
            or not hasattr(runtime, "browser_service_url")
            or not runtime.browser_service_url
        ):
            return {
                "success": False,
                "error": {
                    "code": "service_unavailable",
                    "message": "Browser service not available",
                    "retry_after_ms": None,
                },
            }
        try:
            browser_auth = getattr(runtime, "browser_auth_token", "")
            headers = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            resp = await _dashboard_browser_client.get(
                f"{runtime.browser_service_url}/browser/{agent_id}/fingerprint-health",
                headers=headers,
            )
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": {
                        "code": "upstream_error",
                        "message": f"Browser service returned {resp.status_code}",
                        "retry_after_ms": None,
                    },
                }
            return {"success": True, "data": resp.json()}
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "service_unavailable",
                    "message": str(e),
                    "retry_after_ms": None,
                },
            }

    @api_router.post("/api/agents/{agent_id}/fingerprint-health/reset")
    async def api_agent_fingerprint_health_reset(agent_id: str) -> dict:
        """Phase 10 §22 — clear the per-agent fingerprint rejection window.

        Operator-only endpoint, used AFTER manually rotating the
        BrowserForge fingerprint.  The router-level CSRF guard
        (``_csrf_check`` requires ``X-Requested-With``) gates this path,
        so a cookie-only CSRF cannot reset a flagged agent's burn state.
        The live BrowserContext is intentionally NOT touched —
        ``/browser/{agent_id}/reset`` remains the operator action for
        wiping in-process browser state.
        """
        if agent_id not in agent_registry:
            raise HTTPException(404, "Agent not found")
        if (
            not runtime
            or not hasattr(runtime, "browser_service_url")
            or not runtime.browser_service_url
        ):
            return {
                "success": False,
                "error": {
                    "code": "service_unavailable",
                    "message": "Browser service not available",
                    "retry_after_ms": None,
                },
            }
        try:
            browser_auth = getattr(runtime, "browser_auth_token", "")
            headers = {}
            if browser_auth:
                headers["Authorization"] = f"Bearer {browser_auth}"
            resp = await _dashboard_browser_client.post(
                f"{runtime.browser_service_url}/browser/{agent_id}/fingerprint-health/reset",
                headers=headers,
            )
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": {
                        "code": "upstream_error",
                        "message": f"Browser service returned {resp.status_code}",
                        "retry_after_ms": None,
                    },
                }
            return {"success": True, "data": resp.json()}
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "service_unavailable",
                    "message": str(e),
                    "retry_after_ms": None,
                },
            }

    # Phase 1 — unified messenger: SQLite-backed, per-session set of
    # worker conversations the user has explicitly "opened". Operator is
    # always implicit (pinned), so it doesn't live here. Replaces the
    # previous module-level ``set[str]`` which leaked between concurrent
    # users in multi-tenant SSO deployments and was wiped on every
    # process restart. The store is keyed on a hash of the operator's
    # ``ol_session`` cookie (see :func:`_conversations_session_id`).
    def _conversations_session_id(request: Request) -> str:
        """Derive a stable per-session identifier for opened-conversations.

        Mirrors :func:`_operator_session_id` (defined later in this
        module) — hashes the ``ol_session`` cookie value so the SQLite
        rows tie to a session without echoing the raw cookie. Falls
        back to a constant in dev mode where the cookie is empty (the
        single-operator self-hosted case naturally collapses to one
        bucket, which is the behavior we want).
        """
        raw = request.cookies.get("ol_session", "")
        if not raw:
            return "dev:operator"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
        return f"operator:{digest}"

    @api_router.get("/api/conversations")
    async def api_conversations(request: Request) -> dict:
        """Return the messenger conversation list — operator + opened workers.

        Phase 1 messenger contract. Operator is always pinned with a
        distinct ``role: "manager"`` flag. Workers appear only when the
        user has explicitly opened them via ``POST /api/conversations/
        {agent_id}/open`` (Decision 7 — progressive disclosure). The
        ``unread_count`` field is reserved for future cross-device unread
        sync; today the client tracks unread locally in ``chatUnread``.

        The opened-workers list is scoped to the requester's session
        (hash of the ``ol_session`` cookie) so concurrent operators in
        multi-user deployments do not see each other's open chats.
        """
        # Operator is always present; pull last health-check ts as a cheap
        # proxy for "last activity" until the chat store grows a real
        # ``last_message_ts`` column.
        op_last_ts = 0.0
        if health_monitor is not None and "operator" in getattr(health_monitor, "agents", {}):
            try:
                op_last_ts = float(health_monitor.agents["operator"].last_check or 0)
            except Exception:
                op_last_ts = 0.0
        operator_entry = {
            "agent_id": "operator",
            "role": "manager",
            "last_message_ts": op_last_ts,
            "unread_count": 0,
            "pinned": True,
        }

        workers = []
        opened_ids: list[str] = []
        if opened_conversations_store is not None:
            try:
                opened_ids = opened_conversations_store.list_for_session(
                    _conversations_session_id(request),
                )
            except Exception as e:
                logger.warning("OpenedConversationsStore.list_for_session failed: %s", e)
                opened_ids = []
        for agent_id in opened_ids:
            # Skip workers that no longer exist in the registry (e.g.
            # after a delete) so the messenger doesn't show ghosts.
            if agent_id not in agent_registry and agent_id != "operator":
                continue
            if agent_id == "operator":
                continue
            last_ts = 0.0
            if health_monitor is not None and agent_id in getattr(health_monitor, "agents", {}):
                try:
                    last_ts = float(health_monitor.agents[agent_id].last_check or 0)
                except Exception:
                    last_ts = 0.0
            workers.append({
                "agent_id": agent_id,
                "role": "worker",
                "last_message_ts": last_ts,
                "unread_count": 0,
                "pinned": False,
            })

        return {"operator": operator_entry, "workers": workers}

    @api_router.post("/api/conversations/{agent_id}/open")
    async def api_conversations_open(agent_id: str, request: Request) -> dict:
        """Mark a worker conversation as opened (visible in messenger).

        Operator is always pinned and cannot be opened/closed; we accept
        the call and return success for symmetry but don't mutate state.
        Opened-state is scoped to the requester's session.
        """
        if agent_id == "operator":
            return {"ok": True, "agent_id": agent_id, "noop": True}
        if agent_id not in agent_registry:
            raise HTTPException(404, "Agent not found")
        if opened_conversations_store is not None:
            try:
                opened_conversations_store.open(
                    _conversations_session_id(request), agent_id,
                )
            except Exception as e:
                logger.warning("OpenedConversationsStore.open failed: %s", e)
        return {"ok": True, "agent_id": agent_id}

    @api_router.post("/api/conversations/{agent_id}/close")
    async def api_conversations_close(agent_id: str, request: Request) -> dict:
        """Hide a worker conversation from the messenger (history preserved).

        Operator is always pinned and cannot be closed; we accept the
        call and return success for symmetry but don't mutate state.
        Opened-state is scoped to the requester's session.
        """
        if agent_id == "operator":
            return {"ok": True, "agent_id": agent_id, "noop": True}
        if opened_conversations_store is not None:
            try:
                opened_conversations_store.close_conversation(
                    _conversations_session_id(request), agent_id,
                )
            except Exception as e:
                logger.warning("OpenedConversationsStore.close failed: %s", e)
        return {"ok": True, "agent_id": agent_id}

    @api_router.get("/api/agent-templates")
    async def api_agent_templates() -> list:
        """Return available tool templates for creating new agents."""
        from src.cli.config import _load_tool_templates
        return _load_tool_templates()

    @api_router.get("/api/fleet/templates")
    async def api_fleet_templates(request: Request) -> dict:
        """Return available fleet templates."""
        from src.cli.config import _load_templates
        templates = _load_templates()
        result = []
        for name, tpl in templates.items():
            result.append({
                "name": name,
                "description": tpl.get("description", ""),
                "agent_count": len(tpl.get("agents", {})),
                "agents": list(tpl.get("agents", {}).keys()),
            })
        return {"templates": result}

    def _clean_skill_names(value) -> list[str]:
        """Validate + normalise a skill-name list (sorted, deduped, path-safe)."""
        if not isinstance(value, list):
            raise HTTPException(400, "skills must be a list of names")
        cleaned: set[str] = set()
        for s in value:
            if not isinstance(s, str):
                raise HTTPException(400, "skill names must be strings")
            s = s.strip()
            if not s:
                continue
            if not all((c.isascii() and c.isalnum()) or c in "_-" for c in s):
                raise HTTPException(400, f"Invalid skill name: {s!r}")
            cleaned.add(s)
        return sorted(cleaned)

    def _skills_installed_dir():
        root = getattr(runtime, "project_root", None) if runtime else None
        return (root / "skills_installed") if root else None

    @api_router.get("/api/skills")
    async def api_skills(request: Request, agent_id: str = "") -> dict:
        """List SKILL.md packs (bundled + installed) with provenance + assignment.

        ``fleet_assigned`` — assigned fleet-wide (every agent sees it). When
        ``agent_id`` is given, ``agent_assigned`` reports the skill's presence on
        that agent's per-agent allowlist. The Skills tab renders fleet-assigned
        packs as inherited (on for all agents, not per-agent toggleable).
        """
        from src.agent.skills import SkillStore
        root = getattr(runtime, "project_root", None) if runtime else None
        store = (
            SkillStore(bundled_dir=root / "skills", installed_dir=root / "skills_installed")
            if root
            else SkillStore()
        )
        fleet = set(getattr(permissions, "fleet_skills", []) or []) if permissions else set()
        per_agent: set[str] = set()
        if agent_id and permissions is not None:
            per_agent = set(permissions.get_permissions(agent_id).allowed_skills)
        result = []
        for s in store.list():
            entry = {
                "name": s.name,
                "description": s.description,
                "version": s.version,
                "provenance": s.source or "bundled",
                "fleet_assigned": s.name in fleet,
            }
            if agent_id:
                entry["agent_assigned"] = s.name in per_agent
            result.append(entry)
        return {"skills": result}

    @api_router.post("/api/fleet/skills")
    async def api_set_fleet_skills(request: Request) -> dict:
        """Set the fleet-wide skill allowlist. Body: {skills: [...]}."""
        if permissions is None:
            raise HTTPException(503, "Permissions not available")
        body = await request.json()
        skills = _clean_skill_names(body.get("skills", []))
        from src.cli.config import _load_permissions, _save_permissions
        perms = _load_permissions()
        perms["fleet_skills"] = skills
        _save_permissions(perms)
        permissions.reload()
        _emit_config_changed("skills")
        return {"fleet_skills": skills}

    @api_router.post("/api/skills/install")
    async def api_install_skill(request: Request) -> dict:
        """Install a SKILL.md pack from a git repo. Body: {repo_url, ref?}."""
        import asyncio as _asyncio

        from src import marketplace
        body = await request.json()
        repo_url = str(body.get("repo_url", "")).strip()
        if not repo_url:
            raise HTTPException(400, "repo_url is required")
        skills_installed = _skills_installed_dir()
        if skills_installed is None:
            raise HTTPException(503, "Skills directory not available")
        result = await _asyncio.to_thread(
            marketplace.install_skill, repo_url, skills_installed,
            str(body.get("ref", "")).strip(),
        )
        if "error" in result:
            raise HTTPException(400, result["error"])
        _emit_config_changed("skills")
        return result

    @api_router.post("/api/skills/remove")
    async def api_remove_skill(request: Request) -> dict:
        """Remove an installed skill pack. Body: {name}."""
        import asyncio as _asyncio

        from src import marketplace
        body = await request.json()
        name = str(body.get("name", "")).strip()
        if not name:
            raise HTTPException(400, "name is required")
        skills_installed = _skills_installed_dir()
        if skills_installed is None:
            raise HTTPException(503, "Skills directory not available")
        result = await _asyncio.to_thread(marketplace.remove_skill, name, skills_installed)
        if "error" in result:
            status = 404 if "not found" in result["error"].lower() else 400
            raise HTTPException(status, result["error"])
        _emit_config_changed("skills")
        return result

    # ── MCP connectors (fleet catalog) ───────────────────────
    # The connector catalog is the single source of truth for which
    # agents run which MCP servers. Records persist in
    # config/connectors.json; the runtime serializes each agent's
    # assigned set into MCP_SERVERS at container start, so every
    # catalog change applies on the next restart of the affected
    # agents (returned as ``affected_agents`` for the UI's
    # restart-now/later prompt — D7: no automatic mass-restarts).

    def _expand_assignment(agents: list[str]) -> list[str]:
        """Concrete running agent ids for an ``agents`` field."""
        from src.shared.types import CONNECTOR_ALL_AGENTS
        if CONNECTOR_ALL_AGENTS in agents:
            return sorted(agent_registry.keys())
        return sorted(a for a in agents if a in agent_registry)

    def _connector_to_api(c: Any) -> dict:
        """Catalog record → API shape, per transport. stdio: env values
        masked to ``env_keys`` (same contract as the old per-agent
        surface: a GET→edit→PUT round-trip that omits ``env`` preserves
        it). http: ``url`` + auth *names* only — the bearer token /
        connection contents live in the vault and never appear in any
        GET response."""
        from src.shared.types import HttpConnector
        if isinstance(c, HttpConnector):
            shaped: dict = {
                "name": c.name,
                "transport": "http",
                "url": c.url,
                "auth": {
                    "kind": c.auth.kind,
                    "cred": c.auth.cred,
                    "connection": c.auth.connection,
                },
            }
        else:
            shaped = _mask_mcp_servers_for_get([c.server_dict()])[0]
            shaped["transport"] = "stdio"
        shaped["agents"] = list(c.agents)
        shaped["assigned_agents"] = _expand_assignment(c.agents)
        return shaped

    def _connector_audit_dict(c: Any) -> list[dict]:
        """Audit-row serialization, per transport. stdio env redacted
        via the existing helper; http carries names only by shape."""
        from src.shared.types import HttpConnector
        if isinstance(c, HttpConnector):
            return [{
                "name": c.name,
                "transport": "http",
                "url": c.url,
                "auth_kind": c.auth.kind,
                "auth_cred": c.auth.cred,
                "auth_connection": c.auth.connection,
                "agents": list(c.agents),
            }]
        return _redact_mcp_env_for_audit(
            [{**c.server_dict(), "agents": c.agents}],
        )

    @api_router.get("/api/connectors")
    async def api_connectors_list() -> dict:
        """The fleet MCP connector catalog + pending-restart state."""
        if connector_store is None:
            raise HTTPException(503, "Connector catalog not available")
        all_agents = sorted(agent_registry.keys())
        return {
            "connectors": [
                _connector_to_api(c) for c in connector_store.list()
            ],
            # The store may retain stamps for agents that no longer
            # exist; only live agents are actionable here.
            "pending_restart": [
                a for a in connector_store.pending_restart()
                if a in agent_registry
            ],
            # Agent-tier vault credential names for the stdio env-row
            # picker AND the remote bearer picker (names only, never
            # values).
            "available_credentials": sorted(
                credential_vault.list_agent_credential_names()
            ) if credential_vault else [],
            "agents": all_agents,
        }

    @api_router.put("/api/connectors/{name}")
    async def api_connector_upsert(name: str, request: Request) -> dict:
        """Create or replace a connector (``name`` taken from the
        path). Body: stdio (``command``/``args``/``env``) or http
        (``transport: "http"`` + ``url``/``auth``) connector fields.
        Absent ``env`` (stdio) / ``auth`` (http) / ``agents`` =
        preserve the persisted value; present = replace wholesale.
        Returns the affected agents so the UI can offer the
        restart-now/later choice — except http auth-only edits, which
        apply on the next gateway call with no restart (plan D12)."""
        if connector_store is None:
            raise HTTPException(503, "Connector catalog not available")
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "Body must be a JSON object")
        body_name = body.get("name")
        if body_name is not None and str(body_name).lower() != name.lower():
            raise HTTPException(
                400, "Connector name in body does not match URL path",
            )
        from src.shared.types import (
            CONNECTOR_ADAPTER,
            HttpConnector,
            MCPConnector,
        )
        previous = connector_store.get(name)
        raw = dict(body)
        raw["name"] = previous.name if previous is not None else name
        # Display-side artifacts a GET-replay would carry.
        raw.pop("env_keys", None)
        raw.pop("assigned_agents", None)
        # Absent transport inherits the EXISTING record's transport —
        # otherwise the union's stdio default would turn a partial PUT
        # against an http connector into a confusing extra_forbidden
        # 400 (or, with stdio fields, a silent http→stdio morph). A
        # cross-transport replace must say so explicitly.
        if "transport" not in body and previous is not None:
            raw["transport"] = (
                "http" if isinstance(previous, HttpConnector) else "stdio"
            )
        # Absent = preserve: a partial PUT must not silently wipe the
        # persisted env / auth / assignment. Guarded per transport —
        # a cross-transport replace (same name, stdio↔http) preserves
        # nothing transport-specific.
        if (
            "env" not in body
            and isinstance(previous, MCPConnector)
            and raw.get("transport", "stdio") == "stdio"
        ):
            raw["env"] = previous.env
        if (
            "auth" not in body
            and isinstance(previous, HttpConnector)
            and raw.get("transport") == "http"
        ):
            raw["auth"] = previous.auth.model_dump()
        if "agents" not in body and previous is not None:
            raw["agents"] = previous.agents
        try:
            connector = CONNECTOR_ADAPTER.validate_python(raw)
        except ValidationError as ve:
            # Structured per-field errors for inline UI rendering.
            # ``ctx``/``input`` stripped — they can contain raw objects
            # FastAPI's JSON encoder rejects. The union prefixes loc
            # with the discriminator tag (('http', 'url')) — strip it,
            # because the UI keys inline errors off loc[0] and a tag
            # there would orphan every field error.
            safe_errors = []
            for e in ve.errors(include_url=False):
                loc = [str(p) for p in e.get("loc", ())]
                if loc and loc[0] in ("stdio", "http"):
                    loc = loc[1:]
                safe_errors.append({
                    "loc": loc,
                    "msg": e.get("msg", ""),
                    "type": e.get("type", ""),
                })
            raise HTTPException(
                400, detail={"field": "connector", "errors": safe_errors},
            )

        # ── PUT-time $CRED check: existence + per-agent permission ──
        # Surfaces failures synchronously with an actionable message
        # instead of at agent start. Checked against LIVE agents only
        # (both for "*" and explicit assignments — a deleted agent's
        # leftover id must not block edits forever). Agents that appear
        # later are covered at start time by the per-connector
        # degradation in ``_build_mcp_servers_env``: the connector is
        # dropped for that agent with an error log, never blocking the
        # agent from booting.
        if isinstance(connector, MCPConnector):
            check_agents = _expand_assignment(connector.agents)
            handles: set[str] = set()
            for arg in connector.args:
                handles.update(CRED_HANDLE_RE.findall(arg))
            for env_val in (connector.env or {}).values():
                handles.update(CRED_HANDLE_RE.findall(env_val))
            for cred_name in sorted(handles):
                if (
                    credential_vault is not None
                    and credential_vault.resolve_credential(cred_name) is None
                ):
                    raise HTTPException(
                        400,
                        f"Credential {cred_name!r} does not exist in the vault. "
                        "Store it via the credentials surface first.",
                    )
                if permissions is not None:
                    blocked = [
                        a for a in check_agents
                        if not permissions.can_access_credential(a, cred_name)
                    ]
                    if blocked:
                        raise HTTPException(
                            400,
                            f"Agent(s) {', '.join(repr(a) for a in blocked)} lack "
                            f"permission for credential {cred_name!r}. Grant it "
                            "in their permissions or narrow the assignment.",
                        )
        elif connector.auth.kind == "bearer":
            # http bearer: vault-EXISTENCE only, deliberately no
            # per-agent can_access_credential (plan D14) — the token is
            # mesh-held and injected by the gateway; agents never see
            # it, so per-agent grants would be exposure, not safety.
            if (
                credential_vault is not None
                and credential_vault.resolve_credential(connector.auth.cred)
                is None
            ):
                raise HTTPException(
                    400,
                    f"Credential {connector.auth.cred!r} does not exist in "
                    "the vault. Store it via the credentials surface first.",
                )

        # ── no-op detection: a GET→unchanged-PUT must not mark agents
        # dirty, prompt for a restart, or write an audit row. Both
        # sides are validated instances, so direct field comparison
        # suffices (stdio: env {} and None compare equal — both mean
        # "no env").
        if previous is not None and type(previous) is type(connector):
            if isinstance(connector, MCPConnector):
                same_record = (
                    connector.command == previous.command
                    and connector.args == previous.args
                    and (connector.env or None) == (previous.env or None)
                )
            else:
                same_record = (
                    connector.url == previous.url
                    and connector.auth == previous.auth
                )
            if same_record and set(connector.agents) == set(previous.agents):
                return {
                    "connector": _connector_to_api(previous),
                    "affected_agents": [],
                    "restart_required": False,
                }

        # Affected = before ∪ after assignment (an agent REMOVED from a
        # connector needs a bounce to lose its tools, too).
        before = _expand_assignment(previous.agents) if previous else []
        after = _expand_assignment(connector.agents)
        affected = sorted(set(before) | set(after))

        # Disk write off the event loop (same pattern as the skill
        # marketplace calls in this file). The dirty matrix (plan D12),
        # refined: an http auth ROTATION (same kind, new secret) applies
        # on the gateway's next per-call resolve — no restart prompt.
        # An auth-MODE change (none→bearer, →oauth, …) is different:
        # agents register a connector's tools at BOOT, and a server
        # that 401'd at the agent's last start registered zero tools —
        # fixing its auth mode only takes effect for agents after a
        # bounce, so it must mark them pending-restart like any other
        # agent-visible change. The gateway cache invalidates for every
        # auth edit either way.
        import asyncio as _asyncio
        restart_relevant = await _asyncio.to_thread(
            connector_store.upsert, connector,
        )
        auth_mode_changed = (
            not restart_relevant
            and previous is not None
            and isinstance(connector, HttpConnector)
            and isinstance(previous, HttpConnector)
            and previous.auth.kind != connector.auth.kind
        )
        if restart_relevant or auth_mode_changed:
            connector_store.mark_dirty(affected)
        if not restart_relevant and mcp_gateway is not None:
            mcp_gateway.invalidate(connector.name)
        prompt_restart = restart_relevant or auth_mode_changed
        try:
            blackboard.log_audit(
                action="edit_connector",
                target=connector.name,
                field="connector",
                before_value=json.dumps(
                    _connector_audit_dict(previous) if previous else None,
                ),
                after_value=json.dumps(_connector_audit_dict(connector)),
                actor="dashboard",
                provenance="user",
            )
        except Exception as e:
            logger.warning("Audit log failed for connector %s: %s", name, e)
        _emit_config_changed("connectors", name=connector.name)
        return {
            "connector": _connector_to_api(connector),
            "affected_agents": affected if prompt_restart else [],
            "restart_required": bool(affected) and prompt_restart,
        }

    @api_router.delete("/api/connectors/{name}")
    async def api_connector_delete(name: str) -> dict:
        """Remove a connector. Affected agents keep the server until
        their next restart (returned for the restart prompt)."""
        if connector_store is None:
            raise HTTPException(503, "Connector catalog not available")
        previous = connector_store.get(name)
        if previous is None:
            raise HTTPException(404, "Connector not found")
        affected = _expand_assignment(previous.agents)
        import asyncio as _asyncio
        await _asyncio.to_thread(connector_store.remove, name)
        connector_store.mark_dirty(affected)
        if mcp_gateway is not None:
            mcp_gateway.invalidate(name)
        try:
            blackboard.log_audit(
                action="delete_connector",
                target=previous.name,
                field="connector",
                before_value=json.dumps(_connector_audit_dict(previous)),
                after_value="",
                actor="dashboard",
                provenance="user",
            )
        except Exception as e:
            logger.warning("Audit log failed for connector %s: %s", name, e)
        _emit_config_changed("connectors", name=previous.name)
        return {
            "removed": True,
            "affected_agents": affected,
            "restart_required": bool(affected),
        }

    @api_router.post("/api/connectors/{name}/probe")
    async def api_connector_probe(name: str) -> dict:
        """'Test connection' for a remote connector: fresh initialize +
        tool discovery through the mesh gateway. Returns
        ``{ok, tools_count}`` or ``{ok: False, error, needs_auth}`` —
        ``needs_auth`` drives the Connect affordance (Phase 3). Covered
        by the X-Requested-With CSRF middleware like every state-
        adjacent route (it makes a mesh-originated outbound request)."""
        if mcp_gateway is None:
            raise HTTPException(503, "Connector gateway not configured")
        # probe() classifies every failure (incl. GatewayUnavailable)
        # into its {ok: False, error, needs_auth} shape — no exception
        # mapping needed here.
        return await mcp_gateway.probe(name)

    # Agents with a batch-initiated restart currently in flight. Guards
    # against overlapping batches interleaving stop/start on the same
    # container (the second batch's stop can kill the container the
    # first batch's start just created).
    _restart_batch_inflight: set[str] = set()

    async def _run_restart_batch(agents: list[str]) -> None:
        """Background batch driver. Sequential by design (no restart
        herd); each agent reuses the single-agent restart handler so
        the SPA gets the same event choreography (``agent_restarting``
        → ``agent_restarted`` / ``restart_failed``). Ends with a
        ``config_changed`` so the Connectors panel re-derives
        pending-restart state."""
        try:
            for agent_id in agents:
                try:
                    await api_restart_agent(agent_id)
                except Exception as e:
                    detail = e.detail if isinstance(e, HTTPException) else e
                    logger.error(
                        "Batch restart of %r failed: %s", agent_id, detail,
                    )
                finally:
                    _restart_batch_inflight.discard(agent_id)
        finally:
            # Belt-and-braces: never leave ids stuck in the in-flight set.
            _restart_batch_inflight.difference_update(agents)
            _emit_config_changed("connectors")

    @api_router.post("/api/agents/restart-batch")
    async def api_restart_agents_batch(request: Request) -> dict:
        """Kick off restarts for a set of agents (the Connectors page's
        "restart now" action) and return immediately — a fleet-wide
        batch can take minutes per agent, far past any proxy/browser
        timeout, so completion is reported through the per-agent
        restart events (which the SPA already renders) rather than the
        response. Unknown and already-restarting agents are skipped and
        reported."""
        body = await request.json()
        agents = body.get("agents")
        if not isinstance(agents, list) or not agents:
            raise HTTPException(400, "agents must be a non-empty list")
        for agent_id in agents:
            if not isinstance(agent_id, str):
                raise HTTPException(400, "agents must be a list of agent ids")
        started: list[str] = []
        skipped: dict[str, str] = {}
        seen: set[str] = set()
        for agent_id in agents:
            if agent_id in seen:
                continue
            seen.add(agent_id)
            if agent_id not in agent_registry:
                skipped[agent_id] = "unknown agent"
            elif agent_id in _restart_batch_inflight:
                skipped[agent_id] = "restart already in progress"
            else:
                started.append(agent_id)
        _restart_batch_inflight.update(started)
        if started:
            import asyncio as _asyncio
            _asyncio.get_running_loop().create_task(
                _run_restart_batch(started),
            )
        return {"started": started, "skipped": skipped}

    @api_router.post("/api/agents")
    async def api_add_agent(request: Request) -> dict:
        """Add a new agent: create config, start container, register."""
        body = await request.json()
        name = body.get("name", "").strip()
        role = body.get("role", "").strip()
        model = body.get("model", "").strip()
        avatar = body.get("avatar", 1)
        color = body.get("color")
        template = body.get("template", "").strip()
        # back-compat with pre-PR-3 clients. Either keyword maps to the
        # same per-agent membership write under config/teams/.
        team = (body.get("team") or "").strip()

        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not re.match(r"^[a-z][a-z0-9_]{0,29}$", name):
            raise HTTPException(status_code=400, detail="name must match ^[a-z][a-z0-9_]{0,29}$")
        if name in agent_registry:
            raise HTTPException(status_code=409, detail=f"Agent '{name}' already exists")
        # Limit based on running agents (resource usage), not config definitions.
        # A stopped agent frees a slot. Operator is excluded from the count.
        from src.cli.config import _OPERATOR_AGENT_ID
        non_operator_count = sum(1 for a in agent_registry if a != _OPERATOR_AGENT_ID)
        if _max_agents > 0 and non_operator_count >= _max_agents:
            raise HTTPException(
                status_code=403,
                detail=f"Agent limit reached ({_max_agents}). Upgrade your plan for more agents.",
            )

        if model and not _is_valid_model(model):
            raise HTTPException(status_code=400, detail=f"Invalid model: {model}")

        try:
            avatar = int(avatar)
            if avatar < 1 or avatar > 50:
                raise HTTPException(status_code=400, detail="Avatar must be between 1 and 50")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Avatar must be an integer between 1 and 50")

        if color is not None:
            try:
                color = int(color)
                if color < 0 or color > 15:
                    raise HTTPException(status_code=400, detail="Color must be between 0 and 15")
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="Color must be an integer between 0 and 15")

        if template and not re.match(r"^[a-z][a-z0-9_-]*/[a-z][a-z0-9_-]*$", template):
            raise HTTPException(status_code=400, detail="Invalid template id format")

        if not model:
            from src.cli.config import _load_config
            default = _load_config().get("llm", {}).get("default_model", "openai/gpt-4o-mini")
            # Only use the configured default if its provider has credentials
            default_provider = default.split("/")[0] if "/" in default else ""
            active = credential_vault.get_providers_with_credentials() if credential_vault else set()
            if active and default_provider not in active:
                # Pick the first model from the first provider that has a key
                model = ""
                for p, models in _PROVIDER_MODELS.items():
                    if p in active and models:
                        model = models[0]
                        break
                if not model:
                    model = default  # No credentials at all — use config default
            else:
                model = default
        if not role:
            role = "assistant"

        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not available")

        try:
            from src.cli.config import (
                _create_agent,
                _create_agent_from_template,
                _load_config,
                _update_agent_field,
            )
            if template:
                try:
                    _create_agent_from_template(name, template, model)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
            else:
                _create_agent(name, role, model)
            _update_agent_field(name, "avatar", avatar)
            if color is not None:
                _update_agent_field(name, "color", color)
            if permissions is not None:
                permissions.reload()

            cfg = _load_config()
            acfg = cfg.get("agents", {}).get(name, {})
            if template:
                role = acfg.get("role", role)
            _td = acfg.get("tools_dir", "")
            tools_dir = os.path.abspath(_td) if _td else ""
            # Build per-agent env overrides (no shared extra_env mutation)
            agent_env: dict[str, str] = {}
            for env_key, cfg_key in (
                ("INITIAL_INSTRUCTIONS", "initial_instructions"),
                ("INITIAL_SOUL", "initial_soul"),
                ("INITIAL_HEARTBEAT", "initial_heartbeat"),
            ):
                val = acfg.get(cfg_key, "")
                if val:
                    agent_env[env_key] = val
            url = runtime.start_agent(
                agent_id=name,
                role=role,
                tools_dir=tools_dir,
                model=acfg.get("model", model),
                thinking=acfg.get("thinking", ""),
                env_overrides=agent_env,
            )
            if router is not None:
                router.register_agent(name, url, role=role)
            else:
                agent_registry[name] = url
            if transport is not None:
                from src.host.transport import HttpTransport
                if isinstance(transport, HttpTransport):
                    transport.register(name, url)
            if health_monitor is not None:
                health_monitor.register(name)
            if cron_scheduler is not None:
                hb_schedule = cfg.get("mesh", {}).get("heartbeat_schedule")
                cron_scheduler.ensure_heartbeat(name, hb_schedule)
            ready = await runtime.wait_for_agent(name, timeout=60)
            if team:
                if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$", team):
                    logger.warning("Skipping invalid team name '%s' for agent '%s'", team, name)
                    team = ""
                else:
                    from src.cli.config import (
                        _add_team_blackboard_permissions,
                        _remove_team_blackboard_permissions,
                    )
                    from src.host.teams import TeamNotFound
                    try:
                        old = teams_store.add_member(team, name)
                    except (TeamNotFound, ValueError):
                        logger.warning("Team '%s' not found; agent '%s' created standalone", team, name)
                        team = ""
                    else:
                        if old and old != team:
                            _remove_team_blackboard_permissions(name, old)
                        _add_team_blackboard_permissions(name, team)
            if event_bus is not None:
                event_bus.emit("agent_state", agent=name,
                    data={"state": "added", "role": role, "ready": ready})
            return {
                "created": True,
                "agent": name,
                "ready": ready,
                "team": team or None,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to add agent {name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.delete("/api/agents/{agent_id}")
    async def api_remove_agent(agent_id: str) -> dict:
        """Remove an agent: stop container, unregister, remove config."""
        if agent_id == "operator":
            raise HTTPException(status_code=403, detail="The operator is a system agent and cannot be deleted")
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Stop container and remove data volume (best-effort — agent may already be gone)
        if runtime is not None:
            try:
                runtime.stop_agent(agent_id, remove_data=True)
            except Exception as e:
                logger.debug("Runtime cleanup for '%s' failed: %s", agent_id, e)

        # Unregister from router, transport, and health monitor
        if router is not None:
            router.unregister_agent(agent_id)
        else:
            agent_registry.pop(agent_id, None)
        if transport is not None:
            from src.host.transport import HttpTransport
            if isinstance(transport, HttpTransport):
                transport._urls.pop(agent_id, None)
        if health_monitor is not None:
            health_monitor.unregister(agent_id)

        # Clean up PubSub subscriptions, cron jobs, and lane state
        if pubsub is not None:
            pubsub.unsubscribe_agent(agent_id)
        if cron_scheduler is not None:
            removed = cron_scheduler.remove_agent_jobs(agent_id)
            if removed:
                logger.info(f"Removed {removed} cron job(s) for agent {agent_id}")
        if lane_manager is not None:
            lane_manager.remove_lane(agent_id)

        # Strip the id from connector assignments + pending-restart
        # stamps — otherwise a future agent recreated under the same
        # name silently inherits this agent's MCP connectors (and their
        # $CRED-bearing env).
        if connector_store is not None:
            try:
                import asyncio as _asyncio
                await _asyncio.to_thread(connector_store.remove_agent, agent_id)
            except Exception as e:
                logger.warning(
                    "Connector cleanup for '%s' failed: %s", agent_id, e,
                )

        # Clean up per-agent data: blackboard, costs, traces, wallet
        try:
            blackboard.cleanup_agent_data(agent_id)
        except Exception as e:
            logger.warning("Blackboard cleanup for '%s' failed: %s", agent_id, e)
        if cost_tracker is not None:
            try:
                cost_tracker.cleanup_agent(agent_id)
            except Exception as e:
                logger.warning("Cost cleanup for '%s' failed: %s", agent_id, e)
        if trace_store is not None:
            try:
                trace_store.cleanup_agent(agent_id)
            except Exception as e:
                logger.warning("Trace cleanup for '%s' failed: %s", agent_id, e)
        _ws_ref_local = wallet_service_ref or [None]
        _wallet_svc = _ws_ref_local[0]
        if _wallet_svc is not None:
            try:
                _wallet_svc.cleanup_agent(agent_id)
            except Exception as e:
                logger.warning("Wallet cleanup for '%s' failed: %s", agent_id, e)

        # Clean up proxy credential if exists
        try:
            from src.cli.config import _load_config as _load_cfg_for_delete
            _del_cfg = _load_cfg_for_delete()
            _del_agent_cfg = _del_cfg.get("agents", {}).get(agent_id, {})
            _del_proxy = _del_agent_cfg.get("proxy", {})
            if _del_proxy.get("credential"):
                from src.host.credentials import _remove_from_env
                _cred_env_key = f"OPENLEGION_CRED_{_del_proxy['credential']}"
                _remove_from_env(_cred_env_key)
                os.environ.pop(_cred_env_key, None)
        except Exception as e:
            logger.warning("Proxy credential cleanup for '%s' failed: %s", agent_id, e)

        # Remove from config and permissions (best-effort — don't fail if files are missing)
        try:
            import yaml

            from src.cli.config import AGENTS_FILE, _load_permissions, _save_permissions

            if AGENTS_FILE.exists():
                with open(AGENTS_FILE) as f:
                    agents_data = yaml.safe_load(f) or {}
                agents_data.get("agents", {}).pop(agent_id, None)
                with open(AGENTS_FILE, "w") as f:
                    yaml.dump(agents_data, f, default_flow_style=False, sort_keys=False)

            perms = _load_permissions()
            perms.get("permissions", {}).pop(agent_id, None)
            _save_permissions(perms)
            if permissions is not None:
                permissions.reload()
        except Exception as e:
            logger.warning(f"Failed to clean config for {agent_id}: {e}")

        if event_bus is not None:
            event_bus.emit("agent_state", agent=agent_id,
                data={"state": "removed"})

        return {"removed": True, "agent": agent_id}

    # ── Agent detail ─────────────────────────────────────────

    @api_router.get("/api/agents/{agent_id}")
    async def api_agent_detail(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")

        url = agent_registry[agent_id]
        health_list = health_monitor.get_status() if health_monitor else []
        health = next((h for h in health_list if h["agent"] == agent_id), {})
        spend_today = cost_tracker.get_spend(agent_id, "today")
        spend_week = cost_tracker.get_spend(agent_id, "week")
        budget = cost_tracker.check_budget(agent_id)

        result = {
            "id": agent_id,
            "url": url,
            "health": health or {"status": "unknown"},
            "spend_today": spend_today,
            "spend_week": spend_week,
            "budget": budget,
        }
        # Include this agent's browser VNC info + live browser-running
        # state. ``browser_running`` lets the frontend tear down the
        # iframe when the browser stops (idle timeout, reset) instead
        # of letting noVNC retry forever against a 503'ing endpoint.
        vnc_url = _browser_vnc_url_for_request(request, agent_id)
        if vnc_url:
            result["vnc_url"] = vnc_url
        active_browsers = await _fetch_active_browser_agents()
        result["browser_running"] = agent_id in active_browsers
        return result

    # ── Agent config CRUD ────────────────────────────────────

    @api_router.get("/api/agents/{agent_id}/config")
    async def api_agent_config(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        from fnmatch import fnmatch

        from src.cli.config import _load_config
        cfg = _load_config()
        agent_cfg = cfg.get("agents", {}).get(agent_id, {})
        default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        allowed_creds: list[str] = []
        if permissions is not None:
            allowed_creds = permissions.get_allowed_credentials(agent_id)

        # Compute credential visibility for the dashboard UI
        agent_cred_names = credential_vault.list_agent_credential_names() if credential_vault else []
        system_cred_names = sorted(
            credential_vault.list_system_credential_names()
        ) if credential_vault else []
        resolved = sorted(
            c for c in agent_cred_names
            if any(fnmatch(c, p) for p in allowed_creds)
        ) if allowed_creds else []

        # Capability flags from permissions
        agent_perms = permissions.get_permissions(agent_id) if permissions else None
        cfg_result = {
            "id": agent_id,
            "model": agent_cfg.get("model", default_model),
            "role": agent_cfg.get("role", ""),
            "avatar": agent_cfg.get("avatar", 1),
            "color": agent_cfg.get("color"),
            "budget": agent_cfg.get("budget", {}),
            "thinking": agent_cfg.get("thinking", "off") or "off",
            # Execution caps — fall back to the EFFECTIVE resolved value so
            # the edit form shows what the agent actually runs with when no
            # explicit per-agent override is set.
            "max_output_tokens": agent_cfg.get("max_output_tokens", 16384) or 16384,
            "max_tool_rounds": (
                agent_cfg.get("max_tool_rounds")
                or _limits.resolve("task_max_tool_rounds")
            ),
            "llm_timeout_seconds": (
                agent_cfg.get("llm_timeout_seconds")
                or _limits.resolve("llm_timeout_seconds")
            ),
            "allowed_credentials": allowed_creds,
            "available_credentials": sorted(agent_cred_names),
            "system_credentials": system_cred_names,
            "resolved_credentials": resolved,
            "can_use_browser": agent_perms.can_use_browser if agent_perms else False,
            "can_use_internet": agent_perms.can_use_internet if agent_perms else False,
            "can_spawn": agent_perms.can_spawn if agent_perms else False,
            "can_manage_cron": agent_perms.can_manage_cron if agent_perms else False,
            "can_use_wallet": agent_perms.can_use_wallet if agent_perms else False,
            "wallet_allowed_chains": (
                agent_perms.wallet_allowed_chains if agent_perms else []
            ),
        }
        # Wallet: available chains + derived addresses
        _ws_ref_local = wallet_service_ref or [None]
        ws = _ws_ref_local[0]
        cfg_result["wallet_configured"] = ws is not None
        if ws is not None:
            cfg_result["wallet_available_chains"] = [
                {"id": cid, "label": _wallet_chain_label(cid, ccfg), "ecosystem": ccfg["ecosystem"]}
                for cid, ccfg in ws.chains.items()
            ]
        else:
            cfg_result["wallet_available_chains"] = []
        if ws is not None and cfg_result["can_use_wallet"]:
            try:
                evm_addr = await ws.get_address(agent_id, "evm:ethereum")
                sol_addr = await ws.get_address(agent_id, "solana:mainnet")
                cfg_result["wallet_addresses"] = {
                    "evm": evm_addr, "solana": sol_addr,
                }
            except Exception:
                cfg_result["wallet_addresses"] = None
        else:
            cfg_result["wallet_addresses"] = None

        # Proxy configuration
        proxy_cfg = agent_cfg.get("proxy", {})
        proxy_mode = proxy_cfg.get("mode", "inherit")
        proxy_info: dict[str, Any] = {"mode": proxy_mode}
        if proxy_mode == "custom":
            cred = proxy_cfg.get("credential", "")
            raw = os.environ.get(f"OPENLEGION_CRED_{cred}", "")
            proxy_info["url"] = _mask_proxy_url(raw) if raw else ""
            if raw:
                from urllib.parse import urlparse as _urlparse
                _p = _urlparse(raw)
                proxy_info["host"] = f"{_p.hostname}:{_p.port}" if _p.hostname and _p.port else (_p.hostname or "")
                proxy_info["scheme"] = _p.scheme or "http"
                proxy_info["has_credential"] = bool(_p.username)
            else:
                proxy_info["has_credential"] = False
        cfg_result["proxy"] = proxy_info

        vnc_url = _browser_vnc_url_for_request(request, agent_id)
        if vnc_url:
            cfg_result["vnc_url"] = vnc_url
        active_browsers = await _fetch_active_browser_agents()
        cfg_result["browser_running"] = agent_id in active_browsers
        return cfg_result

    async def _hot_reload_runtime_config(agent_id: str, payload: dict) -> bool:
        """Push model/thinking changes to a running agent's /config endpoint.

        Returns True on success, False if the agent is unreachable or the
        update returns an error. Callers use the return value to decide
        whether restart_required should remain True. Transport already
        returns an error dict for unknown agent_ids and network failures,
        so no pre-check against agent_registry is needed.
        """
        if transport is None:
            logger.debug("Hot-reload skipped for '%s': transport unavailable", agent_id)
            return False
        try:
            result = await transport.request(
                agent_id, "POST", "/config", json=payload, timeout=10,
            )
        except Exception as e:
            logger.warning("Hot-reload runtime config for '%s' failed: %s", agent_id, e)
            return False
        if isinstance(result, dict) and "error" in result:
            logger.warning(
                "Hot-reload runtime config for '%s' returned error: %s",
                agent_id, result["error"],
            )
            return False
        return True

    @api_router.put("/api/agents/{agent_id}/config")
    async def api_update_agent_config(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")
        from src.cli.config import _load_config, _update_agent_field
        cfg = _load_config()
        agent_cfg = cfg.get("agents", {}).get(agent_id, {})
        default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")

        # Phase 1: validate everything into locals before any YAML write.
        # Raising after a partial _update_agent_field() call would leave
        # agents.yaml in a half-applied state that disagrees with the 4xx
        # response the caller sees.
        pending_writes: list[tuple[str, object]] = []
        budget_apply: dict[str, float] | None = None
        runtime_payload: dict[str, object] = {}

        if "model" in body:
            new_model = body["model"]
            if not _is_valid_model(new_model):
                raise HTTPException(status_code=400, detail=f"Invalid model: {new_model}")
            old_model = agent_cfg.get("model", default_model)
            if new_model != old_model:
                pending_writes.append(("model", new_model))
                runtime_payload["model"] = new_model

        if "role" in body:
            role_val = body["role"]
            if not isinstance(role_val, str):
                raise HTTPException(status_code=400, detail="role must be a string")
            pending_writes.append(("role", role_val))

        if "avatar" in body:
            try:
                av = int(body["avatar"])
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="Avatar must be an integer between 1 and 50")
            if av < 1 or av > 50:
                raise HTTPException(status_code=400, detail="Avatar must be between 1 and 50")
            pending_writes.append(("avatar", av))

        if "color" in body:
            raw_color = body["color"]
            if raw_color is None:
                pending_writes.append(("color", None))
            else:
                try:
                    cv = int(raw_color)
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail="Color must be an integer between 0 and 15")
                if cv < 0 or cv > 15:
                    raise HTTPException(status_code=400, detail="Color must be between 0 and 15")
                pending_writes.append(("color", cv))

        if "budget" in body:
            budget_val = body["budget"]
            if isinstance(budget_val, dict):
                raw_daily = budget_val.get("daily_usd")
                raw_monthly = budget_val.get("monthly_usd")
                if raw_daily is not None or raw_monthly is not None:
                    from src.host.costs import DEFAULT_DAILY_BUDGET_USD, DEFAULT_MONTHLY_BUDGET_USD

                    current = cost_tracker.check_budget(agent_id)
                    daily = _parse_positive_float(
                        raw_daily, "daily_usd", current.get("daily_limit", DEFAULT_DAILY_BUDGET_USD),
                    )
                    monthly = _parse_positive_float(
                        raw_monthly, "monthly_usd", current.get("monthly_limit", DEFAULT_MONTHLY_BUDGET_USD),
                    )
                    budget_apply = {"daily_usd": daily, "monthly_usd": monthly}

        if "thinking" in body:
            thinking_val = body["thinking"]
            from src.agent.llm import LLMClient
            if thinking_val not in LLMClient.VALID_THINKING_LEVELS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "thinking must be one of: "
                        f"{sorted(LLMClient.VALID_THINKING_LEVELS)}"
                    ),
                )
            pending_writes.append(("thinking", thinking_val))
            runtime_payload["thinking"] = thinking_val

        # ── Execution caps (per-agent overrides) ──────────────────
        # The agent /config endpoint keys the output cap as ``max_tokens``,
        # so runtime_payload uses that name while the YAML field stays
        # ``max_output_tokens`` (matching agents.yaml / set_llm_max_tokens_env).
        if "max_output_tokens" in body:
            raw_mot = body["max_output_tokens"]
            if isinstance(raw_mot, bool):
                raise HTTPException(
                    status_code=400,
                    detail="max_output_tokens must be an integer between 256 and 200000",
                )
            try:
                mot = int(raw_mot)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=400,
                    detail="max_output_tokens must be an integer between 256 and 200000",
                )
            if mot < 256 or mot > 200000:
                raise HTTPException(
                    status_code=400,
                    detail="max_output_tokens must be between 256 and 200000",
                )
            pending_writes.append(("max_output_tokens", mot))
            runtime_payload["max_tokens"] = mot

        if "max_tool_rounds" in body:
            raw_mtr = body["max_tool_rounds"]
            _lo, _hi = _limits.LIMIT_SPECS["task_max_tool_rounds"][1:]
            if isinstance(raw_mtr, bool):
                raise HTTPException(
                    status_code=400,
                    detail=f"max_tool_rounds must be an integer between {_lo} and {_hi}",
                )
            try:
                mtr = int(raw_mtr)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=400,
                    detail=f"max_tool_rounds must be an integer between {_lo} and {_hi}",
                )
            if mtr < _lo or mtr > _hi:
                raise HTTPException(
                    status_code=400,
                    detail=f"max_tool_rounds must be between {_lo} and {_hi}",
                )
            pending_writes.append(("max_tool_rounds", mtr))
            runtime_payload["max_tool_rounds"] = mtr

        if "llm_timeout_seconds" in body:
            raw_lts = body["llm_timeout_seconds"]
            _lo, _hi = _limits.LIMIT_SPECS["llm_timeout_seconds"][1:]
            if isinstance(raw_lts, bool):
                raise HTTPException(
                    status_code=400,
                    detail=f"llm_timeout_seconds must be an integer between {_lo} and {_hi}",
                )
            try:
                lts = int(raw_lts)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=400,
                    detail=f"llm_timeout_seconds must be an integer between {_lo} and {_hi}",
                )
            if lts < _lo or lts > _hi:
                raise HTTPException(
                    status_code=400,
                    detail=f"llm_timeout_seconds must be between {_lo} and {_hi}",
                )
            pending_writes.append(("llm_timeout_seconds", lts))
            runtime_payload["llm_timeout_seconds"] = lts

        # Phase 2: apply writes now that every field validated.
        updated: list[str] = []
        def _audit(field: str, old_value: object, new_value: object) -> None:
            """Log a dashboard-initiated edit. Never raises — a broken audit
            sink must not block the caller's config change, but silent
            failure is worth a warning so it doesn't rot unnoticed.
            """
            try:
                blackboard.log_audit(
                    action="edit_agent",
                    target=agent_id,
                    field=field,
                    before_value=(
                        old_value if isinstance(old_value, str)
                        else json.dumps(old_value)
                    ),
                    after_value=(
                        new_value if isinstance(new_value, str)
                        else json.dumps(new_value)
                    ),
                    actor="dashboard",
                    provenance="user",
                )
            except Exception as e:
                logger.warning("Audit log failed for %s/%s: %s", agent_id, field, e)

        for field, value in pending_writes:
            old = agent_cfg.get(field, "")
            _update_agent_field(agent_id, field, value)
            updated.append(field)
            _audit(field, old, value)
        if budget_apply is not None:
            old_budget = agent_cfg.get("budget", "")
            _update_agent_field(agent_id, "budget", budget_apply)
            cost_tracker.set_budget(
                agent_id,
                daily_usd=budget_apply["daily_usd"],
                monthly_usd=budget_apply["monthly_usd"],
            )
            updated.append("budget")
            _audit("budget", old_budget, budget_apply)

        # Phase 3: hot-reload runtime state.
        restart_required = False
        if runtime_payload:
            hot_reloaded = await _hot_reload_runtime_config(agent_id, runtime_payload)
            if not hot_reloaded:
                restart_required = True

        return {"updated": updated, "restart_required": restart_required}

    @api_router.post("/api/agents/{agent_id}/restart")
    async def api_restart_agent(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not available")
        # Emit BEFORE stop so the SPA can paint the pulsing "Restarting"
        # state immediately. Best-effort — if the bus is offline the
        # pulse just doesn't show; the final ``agent_restarted`` event
        # carries the same agent_id so the UI can resolve regardless.
        if event_bus is not None:
            try:
                event_bus.emit(
                    "agent_restarting", agent=agent_id,
                    data={"agent_id": agent_id},
                )
            except Exception as e:
                logger.debug("agent_restarting emit failed: %s", e)
        import asyncio
        # Sentinel ensures SPA always gets a terminal event — covers the
        # path where ``runtime.stop_agent`` / ``start_agent`` blocks
        # indefinitely (e.g. Docker daemon hang) and the ``finally``
        # below has to fire a generic ``restart_failed``. Without this
        # the spinner would never clear and the user would refresh.
        fired_terminal = False
        try:
            from src.cli.config import _load_config
            cfg = _load_config()
            agent_cfg = cfg.get("agents", {}).get(agent_id, {})
            default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
            # Hard-cap the synchronous Docker stop/start at 120s combined
            # so a hung daemon returns control to the request handler.
            await asyncio.wait_for(
                asyncio.to_thread(runtime.stop_agent, agent_id),
                timeout=60,
            )
            tools_dir = agent_cfg.get("tools_dir", "")
            if tools_dir:
                tools_dir = str(Path(tools_dir).resolve())
            # Preserve operator's ALLOWED_TOOLS on restart
            from src.cli.config import (
                _OPERATOR_AGENT_ID,
                _OPERATOR_ALLOWED_TOOLS,
                _load_permissions,
            )
            restart_env: dict[str, str] = {}
            if agent_id == _OPERATOR_AGENT_ID:
                restart_env["ALLOWED_TOOLS"] = ",".join(_OPERATOR_ALLOWED_TOOLS)
                # Re-seed internet/browser access flags on restart so the
                # operator's toggle state survives the bounce. Default
                # True matches the operator-by-default UX.
                try:
                    _op_perms = _load_permissions().get(
                        "permissions", {},
                    ).get(_OPERATOR_AGENT_ID, {})
                    restart_env["OL_INTERNET_ACCESS_ENABLED"] = (
                        "true" if _op_perms.get("can_use_internet", True) else "false"
                    )
                    restart_env["OL_BROWSER_ACCESS_ENABLED"] = (
                        "true" if _op_perms.get("can_use_browser", True) else "false"
                    )
                except Exception:
                    restart_env["OL_INTERNET_ACCESS_ENABLED"] = "true"
                    restart_env["OL_BROWSER_ACCESS_ENABLED"] = "true"
            # Proxy goes in env_overrides (not runtime.extra_env) so
            # concurrent single-agent restarts don't stomp each other.
            _proxy_url = resolve_agent_proxy(
                agent_id, cfg.get("agents", {}), cfg.get("network", {}),
            )
            _proxy_env = build_proxy_env_vars(
                _proxy_url, cfg.get("network", {}).get("no_proxy", ""),
            )
            restart_env.update(_proxy_env)
            # Per-agent output-token cap → LLM_MAX_TOKENS so an edit_agent
            # change survives a single-agent dashboard restart (not just the
            # live hot-reload). Absent = LLMClient default 16384.
            set_llm_max_tokens_env(restart_env, agent_cfg)
            from src.shared.limits import set_llm_limits_env
            set_llm_limits_env(restart_env, agent_cfg)
            url = await asyncio.wait_for(
                asyncio.to_thread(
                    runtime.start_agent,
                    agent_id=agent_id,
                    role=agent_cfg.get("role", "assistant"),
                    tools_dir=tools_dir,
                    model=agent_cfg.get("model", default_model),
                    thinking=agent_cfg.get("thinking", ""),
                    env_overrides=restart_env,
                ),
                timeout=60,
            )
            if router is not None:
                router.register_agent(agent_id, url, role=agent_cfg.get("role", ""))
            else:
                agent_registry[agent_id] = url
            if transport is not None:
                from src.host.transport import HttpTransport
                if isinstance(transport, HttpTransport):
                    transport.register(agent_id, url)
            # Re-establish health monitoring if this agent had been deregistered
            # (e.g. archived, then unarchived and restarted). A normal restart of
            # an already-monitored agent is left untouched. Mirrors the batch
            # restart and boot-reconcile paths, which register unconditionally.
            if health_monitor is not None and agent_id not in getattr(health_monitor, "agents", {}):
                health_monitor.register(agent_id)
            ready = await runtime.wait_for_agent(agent_id, timeout=60)
            # Push proxy config to browser service
            await _push_browser_proxy_for_agent(agent_id)
            # Emit AFTER successful start so the SPA clears the pulse
            # and re-fetches details.
            if event_bus is not None:
                try:
                    event_bus.emit(
                        "agent_restarted", agent=agent_id,
                        data={"agent_id": agent_id, "ready": ready},
                    )
                    fired_terminal = True
                except Exception as e:
                    logger.debug("agent_restarted emit failed: %s", e)
            else:
                # No bus to emit on, but the success path still
                # logically "fired" — don't double-emit in finally.
                fired_terminal = True
            return {"restarted": True, "ready": ready}
        except asyncio.TimeoutError as e:
            logger.error(f"Restart timed out for agent {agent_id}")
            if event_bus is not None:
                try:
                    event_bus.emit(
                        "agent_state", agent=agent_id,
                        data={"state": "restart_failed", "error": "Restart timed out"},
                    )
                    fired_terminal = True
                except Exception as emit_e:
                    logger.debug(
                        "agent_state restart_failed emit failed: %s", emit_e,
                    )
            raise HTTPException(status_code=504, detail="Restart timed out") from e
        except Exception as e:
            logger.error(f"Failed to restart agent {agent_id}: {e}")
            # Emit a failure signal via the existing ``agent_state``
            # literal so the SPA can clear the pulse and surface the
            # error without needing a bespoke event type.
            if event_bus is not None:
                try:
                    event_bus.emit(
                        "agent_state", agent=agent_id,
                        data={"state": "restart_failed", "error": str(e)},
                    )
                    fired_terminal = True
                except Exception as emit_e:
                    logger.debug(
                        "agent_state restart_failed emit failed: %s", emit_e,
                    )
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Belt-and-suspenders: if neither the success emit nor the
            # explicit failure emit fired (e.g. an emit raised, the task
            # was cancelled, or some unexpected control-flow path), make
            # sure the SPA still gets a terminal event so the spinner
            # clears.
            if not fired_terminal and event_bus is not None:
                try:
                    event_bus.emit(
                        "agent_state", agent=agent_id,
                        data={"state": "restart_failed", "error": "Restart did not complete"},
                    )
                except Exception as emit_e:
                    logger.debug(
                        "agent_state restart_failed (finally) emit failed: %s",
                        emit_e,
                    )

    @api_router.put("/api/agents/{agent_id}/budget")
    async def api_update_budget(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        body = await request.json()
        raw_daily = body.get("daily_usd")
        raw_monthly = body.get("monthly_usd")
        if raw_daily is None and raw_monthly is None:
            raise HTTPException(status_code=400, detail="Provide daily_usd and/or monthly_usd")
        from src.host.costs import DEFAULT_DAILY_BUDGET_USD, DEFAULT_MONTHLY_BUDGET_USD

        current = cost_tracker.check_budget(agent_id)
        daily_usd = _parse_positive_float(
            raw_daily, "daily_usd", current.get("daily_limit", DEFAULT_DAILY_BUDGET_USD),
        )
        monthly_usd = _parse_positive_float(
            raw_monthly, "monthly_usd", current.get("monthly_limit", DEFAULT_MONTHLY_BUDGET_USD),
        )
        cost_tracker.set_budget(agent_id, daily_usd=daily_usd, monthly_usd=monthly_usd)
        from src.cli.config import _update_agent_field
        _update_agent_field(agent_id, "budget", {"daily_usd": daily_usd, "monthly_usd": monthly_usd})
        return {"updated": True, "agent": agent_id, "daily_usd": daily_usd, "monthly_usd": monthly_usd}

    # ── Network / Proxy ─────────────────────────────────────

    @api_router.get("/api/network/proxy")
    async def api_get_network_proxy(request: Request):
        """Return system proxy info (masked), NO_PROXY, and per-agent proxy summary."""
        from src.cli.config import _load_config
        from src.cli.proxy import _assemble_proxy_url

        cfg = _load_config()
        browser_proxy_url = os.environ.get("BROWSER_PROXY_URL", "")
        system_proxy = os.environ.get("OPENLEGION_SYSTEM_PROXY", "")
        is_managed = bool(browser_proxy_url)

        # Managed URL (always compute when managed, for display)
        managed_masked = ""
        if browser_proxy_url:
            full = _assemble_proxy_url(
                browser_proxy_url,
                os.environ.get("BROWSER_PROXY_USER", ""),
                os.environ.get("BROWSER_PROXY_PASS", ""),
            )
            managed_masked = _mask_proxy_url(full)

        # Active URL follows resolution order: user override > managed
        if system_proxy:
            active_masked = _mask_proxy_url(system_proxy)
        elif managed_masked:
            active_masked = managed_masked
        else:
            active_masked = ""

        is_overridden = is_managed and bool(system_proxy)

        network_cfg = cfg.get("network", {})
        no_proxy = network_cfg.get("no_proxy", "")

        agents_cfg = cfg.get("agents", {})
        agent_summary = []
        for aid, acfg in agents_cfg.items():
            proxy = acfg.get("proxy", {})
            mode = proxy.get("mode", "inherit")
            agent_proxy_url = ""
            if mode == "custom":
                cred = proxy.get("credential", "")
                raw = os.environ.get(f"OPENLEGION_CRED_{cred}", "")
                agent_proxy_url = _mask_proxy_url(raw) if raw else "(credential missing)"
            agent_summary.append({"agent_id": aid, "mode": mode, "proxy_url": agent_proxy_url})

        return {
            "system_proxy": {
                "configured": bool(active_masked),
                "managed": is_managed,
                "managed_url": managed_masked,
                "overridden": is_overridden,
                "url": active_masked,
            },
            "no_proxy": no_proxy,
            "agents": agent_summary,
        }

    @api_router.put("/api/network/proxy")
    async def api_put_network_proxy(request: Request):
        """Update system proxy and/or NO_PROXY."""
        body = await request.json()
        updated = []

        if "no_proxy" in body:
            from src.cli.config import _update_network_config
            _update_network_config("no_proxy", body["no_proxy"])
            updated.append("no_proxy")

        if "system_proxy" in body:
            from src.cli.proxy import _assemble_proxy_url, validate_proxy_url
            from src.host.credentials import _persist_to_env, _remove_from_env

            sp = body["system_proxy"]
            if sp is None or sp == "":
                _remove_from_env("OPENLEGION_SYSTEM_PROXY")
                os.environ.pop("OPENLEGION_SYSTEM_PROXY", None)
                updated.append("system_proxy_removed")
            else:
                url = sp.get("url", "")
                if url and url.strip().lower().startswith("socks"):
                    raise HTTPException(400, "SOCKS5 proxies are not supported — please use an HTTP/HTTPS proxy")
                username = sp.get("username", "")
                password = sp.get("password", "")
                full_url = _assemble_proxy_url(url, username, password) if username else url
                if not validate_proxy_url(full_url):
                    raise HTTPException(400, "Invalid proxy URL")
                _persist_to_env("OPENLEGION_SYSTEM_PROXY", full_url)
                os.environ["OPENLEGION_SYSTEM_PROXY"] = full_url
                updated.append("system_proxy")

        if updated:
            _emit_config_changed("network_proxy")
        return {"updated": updated, "restart_required": bool(updated)}

    @api_router.put("/api/agents/{agent_id}/proxy")
    async def api_put_agent_proxy(agent_id: str, request: Request):
        """Set per-agent proxy config. Works for stopped agents (checks config, not registry)."""
        from src.cli.config import _load_config, _update_agent_field
        from src.cli.proxy import _assemble_proxy_url, sanitize_agent_id_for_env, validate_proxy_url
        from src.host.credentials import _persist_to_env, _remove_from_env

        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})
        if agent_id not in agents_cfg:
            raise HTTPException(404, "Agent not found in config")

        body = await request.json()
        mode = body.get("mode", "inherit")
        if mode not in ("inherit", "custom", "direct"):
            raise HTTPException(400, f"Invalid proxy mode: {mode}")

        proxy_yaml: dict[str, str] = {"mode": mode}
        safe_id = sanitize_agent_id_for_env(agent_id)

        if mode == "custom":
            url = body.get("url", "")
            if url and url.strip().lower().startswith("socks"):
                raise HTTPException(400, "SOCKS5 proxies are not supported — please use an HTTP/HTTPS proxy")
            username = body.get("username", "")
            password = body.get("password", "")
            full_url = _assemble_proxy_url(url, username, password) if username else url
            if not validate_proxy_url(full_url):
                raise HTTPException(400, "Invalid proxy URL")

            cred_name = f"agent_{safe_id}_proxy"
            env_key = f"OPENLEGION_CRED_{cred_name}"
            _persist_to_env(env_key, full_url)
            os.environ[env_key] = full_url
            proxy_yaml["credential"] = cred_name

        elif mode in ("inherit", "direct"):
            # Clean up any existing custom credential
            old_proxy = agents_cfg.get(agent_id, {}).get("proxy", {})
            old_cred = old_proxy.get("credential", "")
            if old_cred:
                env_key = f"OPENLEGION_CRED_{old_cred}"
                _remove_from_env(env_key)
                os.environ.pop(env_key, None)

        _update_agent_field(agent_id, "proxy", proxy_yaml)

        # Push new proxy config to browser service immediately so manual
        # browser resets pick up the change without a full agent restart.
        await _push_browser_proxy_for_agent(agent_id)

        _emit_config_changed("network_proxy", agent=agent_id)
        return {"updated": ["proxy"], "restart_required": True}

    @api_router.get("/api/agents/{agent_id}/permissions")
    async def api_agent_permissions(agent_id: str) -> dict:
        """Return agent permissions and available agent-tier credentials."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if permissions is None:
            raise HTTPException(status_code=503, detail="Permissions not available")
        perms = permissions.get_permissions(agent_id)
        available_creds = []
        if credential_vault is not None:
            available_creds = credential_vault.list_agent_credential_names()
        return {
            "agent_id": agent_id,
            "allowed_credentials": perms.allowed_credentials,
            "allowed_apis": perms.allowed_apis,
            "allowed_skills": perms.allowed_skills,
            "available_credentials": available_creds,
            "can_use_browser": perms.can_use_browser,
            "can_use_internet": perms.can_use_internet,
            "can_spawn": perms.can_spawn,
            "can_manage_cron": perms.can_manage_cron,
        }

    @api_router.put("/api/agents/{agent_id}/permissions")
    async def api_update_agent_permissions(agent_id: str, request: Request) -> dict:
        """Update allowed_credentials and/or allowed_apis for an agent."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if permissions is None:
            raise HTTPException(status_code=503, detail="Permissions not available")
        body = await request.json()
        from src.cli.config import _load_permissions, _save_permissions
        perms_data = _load_permissions()
        agents = perms_data.setdefault("permissions", {})
        # Materialize full effective permissions before a partial write, so an
        # agent that had no explicit entry (was using the "default" template)
        # doesn't get dropped to a sparse record that strips its other grants
        # (Codex review). No-op for agents that already have an entry.
        if agent_id not in agents:
            agents[agent_id] = permissions.get_permissions(agent_id).model_dump(exclude={"agent_id"})
        agent_perms = agents[agent_id]

        updated = []
        if "allowed_credentials" in body:
            val = body["allowed_credentials"]
            if not isinstance(val, list) or not all(isinstance(v, str) for v in val):
                raise HTTPException(status_code=400, detail="allowed_credentials must be a list of strings")
            agent_perms["allowed_credentials"] = val
            updated.append("allowed_credentials")
        if "allowed_apis" in body:
            val = body["allowed_apis"]
            if not isinstance(val, list) or not all(isinstance(v, str) for v in val):
                raise HTTPException(status_code=400, detail="allowed_apis must be a list of strings")
            agent_perms["allowed_apis"] = val
            updated.append("allowed_apis")
        if "allowed_skills" in body:
            # Per-agent skill-pack allowlist (path-safe-validated, sorted/deduped).
            agent_perms["allowed_skills"] = _clean_skill_names(body["allowed_skills"])
            updated.append("allowed_skills")
        for flag in ("can_use_browser", "can_use_internet", "can_spawn", "can_manage_cron", "can_use_wallet"):
            if flag in body:
                agent_perms[flag] = bool(body[flag])
                updated.append(flag)
        if "wallet_allowed_chains" in body:
            val = body["wallet_allowed_chains"]
            if not isinstance(val, list) or not all(isinstance(v, str) for v in val):
                raise HTTPException(
                    status_code=400,
                    detail="wallet_allowed_chains must be a list of strings",
                )
            agent_perms["wallet_allowed_chains"] = val
            updated.append("wallet_allowed_chains")

        perms_data.setdefault("permissions", {})[agent_id] = agent_perms
        _save_permissions(perms_data)
        permissions.reload()
        return {"updated": updated, "agent_id": agent_id}

    @api_router.get("/api/agents/{agent_id}/status")
    async def api_agent_live_status(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            return await transport.request(agent_id, "GET", "/status", timeout=10)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/capabilities")
    async def api_agent_capabilities(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            data = await transport.request(agent_id, "GET", "/capabilities", timeout=10)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
        # Defensive: if the agent transport returned a non-dict shape
        # (malformed-but-valid JSON like a list or a string), surface
        # 502 rather than crashing the backfill block with AttributeError.
        if not isinstance(data, dict):
            raise HTTPException(
                status_code=502,
                detail="Agent /capabilities returned a non-object response",
            )
        # Backfill tool_sources for agents running pre-badge code (no container
        # restart required).  The host process has access to the builtins package
        # and can classify tools without executing agent-side code.
        if "tool_sources" not in data:
            builtins = _get_builtin_tool_names()
            sources: dict[str, str] = {}
            # If the agent surfaced an mcp_tool_to_server side-channel
            # (post-T6 agents), every name listed there is an MCP tool.
            # Older agents lack this field, in which case we fall back
            # to builtin/custom classification only — there's no
            # legitimate way to identify MCP tools from the OpenAI
            # tool-definition format alone (the prior heuristic
            # ``tool.get("function") == "mcp"`` was always False because
            # the OpenAI format puts a dict at ``function``).
            mcp_names: set[str] = set(data.get("mcp_tool_to_server") or {})
            for tool in data.get("tool_definitions", []):
                name = (tool.get("function") or {}).get("name") or tool.get("name")
                if not name:
                    continue
                if name in mcp_names:
                    sources[name] = "mcp"
                elif name in builtins:
                    sources[name] = "builtin"
                else:
                    sources[name] = "custom"
            data["tool_sources"] = sources
        return data

    # ── Chat with agent ────────────────────────────────────

    @api_router.post("/api/agents/{agent_id}/chat")
    async def api_chat(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        from src.shared.utils import sanitize_for_prompt
        message = sanitize_for_prompt(message)
        chat_session = request.headers.get("x-chat-session", "")
        if event_bus:
            event_bus.emit("chat_user_message", agent=agent_id,
                data={"message": message, "session": chat_session})
        # Task 2b: stamp dashboard chat as human-origin so downstream
        # authorization gates can distinguish a user-driven chat from
        # an agent-initiated wake.
        # Session observability (Phase 1): mint a per-turn trace_id and
        # seed the contextvar BEFORE trace_headers() — mirrors the CLI
        # (repl.py: current_trace_id.set(new_trace_id())). Without this the
        # dashboard's outbound /chat call carried an empty X-Trace-Id and
        # the resulting LLM/task/transcript rows were uncorrelatable.
        from src.shared.trace import current_trace_id, new_trace_id, origin_header, trace_headers
        from src.shared.types import MessageOrigin
        # Mint + seed, then RESET via token after the awaited outbound call.
        # Without the reset, this minted trace_id stays on the (worker-reused)
        # context and contaminates a later handler that calls trace_headers()
        # without minting (e.g. /api/broadcast), so its rows inherit a trace
        # from an unrelated chat. The finally runs after transport.request,
        # so the outbound trace_headers() still carries X-Trace-Id.
        _trace_tok = current_trace_id.set(new_trace_id())
        try:
            origin = MessageOrigin(
                kind="human",
                channel="dashboard",
                user=_operator_session_id(request),
            )
            hdrs = trace_headers()
            hdrs.update(origin_header(origin))
            # Session observability (Phase 2): capture the verbatim dashboard
            # turn. This path calls the agent directly (not via
            # ``_direct_dispatch``), so it is the ONLY intent-capture point for
            # the primary human surface — without it ``sessions``/``session``
            # never see dashboard turns. Best-effort: a store failure must
            # never break chat. Redaction happens at storage (H16).
            if intent_store is not None:
                try:
                    intent_store.record(
                        trace_id=current_trace_id.get(),
                        origin_kind="human",
                        origin_channel="dashboard",
                        origin_user=origin.user,
                        agent=agent_id,
                        message=message,
                        meta={"surface": "dashboard"},
                    )
                except Exception as _intent_err:
                    logger.debug("dashboard intent capture failed: %s", _intent_err)
            result = await transport.request(
                agent_id, "POST", "/chat", json={"message": message}, timeout=120,
                headers=hdrs,
            )
            response = result.get("response", "(no response)")
            if event_bus:
                event_bus.emit("chat_done", agent=agent_id,
                    data={"response": response, "session": chat_session})
            return {"response": response}
        except Exception as e:
            if event_bus:
                event_bus.emit("chat_done", agent=agent_id,
                    data={"response": "", "session": chat_session})
            raise HTTPException(status_code=502, detail=str(e))
        finally:
            current_trace_id.reset(_trace_tok)

    @api_router.post("/api/agents/{agent_id}/chat/stream")
    async def api_chat_stream(agent_id: str, request: Request):
        """SSE streaming chat with an agent."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        from src.shared.utils import sanitize_for_prompt
        message = sanitize_for_prompt(message)
        chat_session = request.headers.get("x-chat-session", "")

        # Broadcast user message so other tabs/devices see it immediately
        if event_bus:
            event_bus.emit("chat_user_message", agent=agent_id,
                data={"message": message, "session": chat_session})

        # Task 2b: stamp dashboard streaming chat as human-origin.
        # Session observability (Phase 1): mint a per-turn trace_id and
        # seed the contextvar before trace_headers() so the outbound
        # /chat/stream call carries X-Trace-Id (see api_chat above).
        # The X-Trace-Id is baked into ``_hdrs`` synchronously here, so the
        # generator reads ``_hdrs`` (not the contextvar) — set + reset around
        # the header build, no leak onto the worker-reused context that a
        # later non-minting handler (e.g. /api/broadcast) would inherit.
        from src.shared.trace import current_trace_id, new_trace_id, origin_header, trace_headers
        from src.shared.types import MessageOrigin
        _trace_tok = current_trace_id.set(new_trace_id())
        try:
            _origin = MessageOrigin(
                kind="human",
                channel="dashboard",
                user=_operator_session_id(request),
            )
            _hdrs = trace_headers()
            _hdrs.update(origin_header(_origin))
            # Session observability (Phase 2): capture the verbatim dashboard
            # turn here (the streaming path also bypasses ``_direct_dispatch``).
            # Best-effort; redaction at storage (H16). See api_chat above.
            if intent_store is not None:
                try:
                    intent_store.record(
                        trace_id=current_trace_id.get(),
                        origin_kind="human",
                        origin_channel="dashboard",
                        origin_user=_origin.user,
                        agent=agent_id,
                        message=message,
                        meta={"surface": "dashboard"},
                    )
                except Exception as _intent_err:
                    logger.debug("dashboard intent capture failed: %s", _intent_err)
        finally:
            current_trace_id.reset(_trace_tok)

        async def event_generator():
            final_response = ""
            try:
                async for event in transport.stream_request(
                    agent_id, "POST", "/chat/stream",
                    json={"message": message}, timeout=120,
                    headers=_hdrs,
                ):
                    if isinstance(event, dict):
                        if event.get("type") == "keepalive":
                            # Forward upstream liveness as an SSE comment so the
                            # browser's idle-abort timer (app.js, 120s) resets
                            # during long silent tool calls. Not a data event —
                            # carries no UI state and updates no other session.
                            yield ": keepalive\n\n"
                            continue
                        yield f"data: {dumps_safe(event)}\n\n"
                        etype = event.get("type", "")
                        if event_bus:
                            if etype in ("tool_start", "tool_result", "text_delta"):
                                event_bus.emit(etype, agent=agent_id,
                                    data={k: v for k, v in event.items()
                                          if k != "type"} | {"session": chat_session})
                            if etype == "done":
                                final_response = event.get("response", "")
                                break
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': friendly_streaming_error(e)})}\n\n"
            # Notify other sessions that the response is complete
            if event_bus:
                event_bus.emit("chat_done", agent=agent_id,
                    data={"response": final_response, "session": chat_session})

        from starlette.responses import StreamingResponse
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @api_router.post("/api/broadcast")
    async def api_broadcast(request: Request) -> dict:
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        message = sanitize_for_prompt(message)
        import asyncio

        targets = list(agent_registry.keys())
        team = body.get("team") or ""
        if not isinstance(team, str):
            raise HTTPException(status_code=400, detail="team must be a string")
        if team:
            members = set(teams_store.members(team))
            targets = [a for a in targets if a in members]
        elif body.get("standalone"):
            assigned = set(teams_store.agent_team_map())
            targets = [a for a in targets if a not in assigned]
        if not targets:
            return {"responses": {}, "message": "No matching agents"}

        # Task 2b: stamp dashboard broadcast as human-origin.
        from src.shared.trace import origin_header, trace_headers
        from src.shared.types import MessageOrigin
        bc_origin = MessageOrigin(
            kind="human",
            channel="dashboard",
            user=_operator_session_id(request),
        )
        bc_hdrs = trace_headers()
        bc_hdrs.update(origin_header(bc_origin))

        results = {}
        async def _send(aid: str) -> tuple[str, str]:
            try:
                data = await transport.request(
                    aid, "POST", "/chat", json={"message": message}, timeout=120,
                    headers=bc_hdrs,
                )
                return aid, data.get("response", "(no response)")
            except Exception as e:
                return aid, f"Error: {e}"
        tasks = [_send(aid) for aid in targets]
        for coro in asyncio.as_completed(tasks):
            aid, resp = await coro
            results[aid] = resp
        return {"responses": results}

    @api_router.post("/api/broadcast/stream")
    async def api_broadcast_stream(request: Request):
        """SSE streaming broadcast — streams per-agent responses as they arrive."""
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        message = sanitize_for_prompt(message)

        import asyncio

        agents = list(agent_registry.keys())
        team = body.get("team") or ""
        if not isinstance(team, str):
            raise HTTPException(status_code=400, detail="team must be a string")
        if team:
            members = set(teams_store.members(team))
            agents = [a for a in agents if a in members]
        elif body.get("standalone"):
            assigned = set(teams_store.agent_team_map())
            agents = [a for a in agents if a not in assigned]
        if not agents:
            return {"responses": {}, "message": "No agents registered"}

        # Task 2b: stamp dashboard streaming broadcast as human-origin.
        from src.shared.trace import origin_header, trace_headers
        from src.shared.types import MessageOrigin
        bcs_origin = MessageOrigin(
            kind="human",
            channel="dashboard",
            user=_operator_session_id(request),
        )
        bcs_hdrs = trace_headers()
        bcs_hdrs.update(origin_header(bcs_origin))

        queue: asyncio.Queue = asyncio.Queue()

        async def _stream_agent(aid: str) -> None:
            await queue.put({"type": "agent_start", "agent": aid})
            try:
                async for event in transport.stream_request(
                    aid, "POST", "/chat/stream",
                    json={"message": message}, timeout=120,
                    headers=bcs_hdrs,
                ):
                    if isinstance(event, dict):
                        if event.get("type") == "keepalive":
                            # Liveness only — forward untagged so the generator
                            # emits an SSE comment (resets the browser idle
                            # timer) without counting toward agent completion.
                            await queue.put({"type": "keepalive"})
                            continue
                        tagged = {**event, "agent": aid}
                        await queue.put(tagged)
                        if event.get("type") == "done":
                            break
                        if event_bus:
                            etype = event.get("type", "")
                            if etype in ("tool_start", "tool_result"):
                                event_bus.emit(etype, agent=aid,
                                    data={k: v for k, v in tagged.items() if k != "type"})
            except Exception as e:
                await queue.put({"type": "error", "agent": aid, "message": friendly_streaming_error(e)})
            await queue.put({"type": "agent_done", "agent": aid})

        async def event_generator():
            tasks = [asyncio.create_task(_stream_agent(aid)) for aid in agents]
            done_count = 0
            try:
                while done_count < len(agents):
                    event = await queue.get()
                    if event.get("type") == "keepalive":
                        yield ": keepalive\n\n"
                        continue
                    if event.get("type") == "agent_done":
                        done_count += 1
                    yield f"data: {dumps_safe(event)}\n\n"
            except asyncio.CancelledError:
                for t in tasks:
                    t.cancel()
                raise
            yield f"data: {json.dumps({'type': 'all_done'})}\n\n"

        from starlette.responses import StreamingResponse
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @api_router.post("/api/agents/{agent_id}/steer")
    async def api_steer(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if lane_manager is None:
            raise HTTPException(status_code=503, detail="Lane manager not available")
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        from src.shared.utils import sanitize_for_prompt
        message = sanitize_for_prompt(message)
        chat_session = request.headers.get("x-chat-session", "")
        if event_bus:
            event_bus.emit("chat_user_message", agent=agent_id,
                data={"message": f"[steer] {message}", "session": chat_session})
        # Task 2b: stamp human origin on dashboard-initiated steer.
        from src.shared.trace import new_trace_id
        from src.shared.types import MessageOrigin
        origin = MessageOrigin(
            kind="human",
            channel="dashboard",
            user=_operator_session_id(request),
        )
        result = await lane_manager.enqueue(
            agent_id, message, mode="steer",
            trace_id=new_trace_id(), origin=origin,
        )
        return {"result": result}

    @api_router.post("/api/agents/{agent_id}/reset")
    async def api_reset(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            await transport.request(agent_id, "POST", "/chat/reset", timeout=10)
            if event_bus:
                event_bus.emit("chat_reset", agent=agent_id)
            return {"reset": True, "agent": agent_id}
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/chat/history")
    async def api_chat_history(agent_id: str) -> dict:
        """Return the agent's current in-memory chat conversation."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            result = await transport.request(agent_id, "GET", "/chat/history", timeout=10)
            return result
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/artifacts")
    async def api_list_artifacts(agent_id: str) -> dict:
        """List artifact files in an agent's workspace."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            return await transport.request(agent_id, "GET", "/artifacts", timeout=10)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/artifacts/{name:path}")
    async def api_get_artifact(agent_id: str, name: str) -> dict:
        """Fetch artifact content from an agent's workspace."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            result = await transport.request(agent_id, "GET", f"/artifacts/{name}", timeout=30)
            if "error" in result:
                status = result.get("status_code", 502)
                raise HTTPException(status_code=status, detail=result["error"])
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.delete("/api/agents/{agent_id}/artifacts/{name:path}")
    async def api_delete_artifact(agent_id: str, name: str) -> dict:
        """Delete an artifact file from an agent's workspace."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            result = await transport.request(agent_id, "DELETE", f"/artifacts/{name}", timeout=10)
            if "error" in result:
                status = result.get("status_code", 502)
                raise HTTPException(status_code=status, detail=result["error"])
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/files")
    async def api_list_files(
        agent_id: str,
        path: str = ".",
        recursive: bool = False,
        pattern: str = "*",
    ) -> dict:
        """List files under the agent's /data volume."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            qs = f"path={path}&recursive={'true' if recursive else 'false'}&pattern={pattern}"
            return await transport.request(agent_id, "GET", f"/files?{qs}", timeout=10)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    # Deliberately NOT under ``/files/`` — that namespace ends in a greedy
    # ``{path:path}`` read route, and a sibling ``/files/{path}/download`` would
    # permanently shadow any real file whose path ends in ``/download``. A
    # separate prefix keeps both routes unambiguous.
    @api_router.get("/api/agents/{agent_id}/file-download/{path:path}")
    async def api_download_file(agent_id: str, path: str):
        """Download an agent's /data file as a Content-Disposition attachment.

        Pages the agent's ``/files`` endpoint in 5 MB chunks and concatenates
        the raw bytes, so the user can save a worker's deliverable (CSV,
        data.md, a binary export) straight to disk — bypassing the JSON read
        cap and never routing the bytes through any LLM context. Memory stays
        bounded two ways: a fast reject on the agent-reported size, AND a
        running byte counter that does NOT trust that size (the agent is an
        untrusted component — a lying ``size`` must not OOM the host).
        """
        import base64 as _b64
        from urllib.parse import quote, urlencode

        from fastapi.responses import Response
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        # Quote the filename portion so spaces / '?' / '#' / '%' in a real
        # filename survive interpolation into the agent URL (httpx splits on a
        # raw '?'; the sandbox-backend curl rejects a raw space). The query
        # string is appended AFTER, so only the path segment is quoted.
        safe_path = quote(path, safe="/")
        _CHUNK = 5 * 1024 * 1024
        _MAX_TOTAL = 64 * 1024 * 1024
        parts: list[bytes] = []
        total = 0
        offset = 0
        mime = "application/octet-stream"
        while True:
            qs = urlencode({"offset": offset, "max_bytes": _CHUNK})
            try:
                result = await transport.request(
                    agent_id, "GET", f"/files/{safe_path}?{qs}", timeout=60,
                )
            except Exception as e:
                raise HTTPException(status_code=502, detail=str(e))
            if not isinstance(result, dict) or "content" not in result:
                status = result.get("status_code", 404) if isinstance(result, dict) else 502
                detail = result.get("error", "download failed") if isinstance(result, dict) else "download failed"
                raise HTTPException(status_code=status, detail=detail)
            # Fast reject on the agent-reported size (first page carries it).
            if int(result.get("size", 0)) > _MAX_TOTAL:
                raise HTTPException(
                    status_code=413,
                    detail="File exceeds the 64 MB download cap",
                )
            mime = result.get("mime_type") or mime
            if result.get("encoding") == "base64":
                blob = _b64.b64decode(result.get("content", ""))
            else:
                blob = (result.get("content") or "").encode("utf-8")
            # Truth, not trust: enforce the cap on actual bytes received so a
            # malformed/lying agent (size=0, truncated=true, advancing offset)
            # can't loop the host into unbounded memory growth.
            total += len(blob)
            if total > _MAX_TOTAL:
                raise HTTPException(
                    status_code=413,
                    detail="File exceeds the 64 MB download cap",
                )
            parts.append(blob)
            next_offset = result.get("next_offset", offset + len(blob))
            if not result.get("truncated") or next_offset <= offset:
                break
            offset = next_offset
        filename = path.rsplit("/", 1)[-1] or "download"
        safe_name = re.sub(r"[^\w.\- ]", "_", filename) or "download"
        return Response(
            content=b"".join(parts),
            media_type=mime,
            headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
        )

    @api_router.get("/api/agents/{agent_id}/files/{path:path}")
    async def api_read_file(agent_id: str, path: str) -> dict:
        """Read a file from the agent's /data volume."""
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            result = await transport.request(agent_id, "GET", f"/files/{path}", timeout=30)
            if "error" in result:
                status = result.get("status_code", 404)
                raise HTTPException(status_code=status, detail=result["error"])
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.post("/api/credentials/validate")
    async def api_validate_credential(request: Request) -> dict:
        """Validate an API key by making a minimal LLM call."""
        body = await request.json()
        service = body.get("service", "").strip().lower()
        key = "".join(body.get("key", "").split())
        base_url = body.get("base_url", "").strip() or None
        if not service or not key:
            raise HTTPException(status_code=400, detail="service and key are required")

        # Anthropic OAuth setup-token: validate directly against provider API
        from src.host.credentials import is_oauth_token
        if is_oauth_token(key):
            from src.setup_wizard import SetupWizard
            fmt_error = SetupWizard._validate_oauth_token_format(key)
            if fmt_error:
                return {"valid": False, "skipped": False, "reason": fmt_error}
            import asyncio
            valid = await asyncio.get_running_loop().run_in_executor(
                None, SetupWizard._validate_oauth_token_live, key,
            )
            if valid:
                return {"valid": True, "skipped": False, "oauth": True}
            return {"valid": False, "skipped": False, "reason": "Invalid or expired setup-token"}

        # OAuth JSON blob detection (Anthropic or OpenAI)
        import json as _json

        from src.host.credentials import CredentialVault as _CV
        try:
            parsed = _json.loads(key)
            if isinstance(parsed, dict):
                # Anthropic OAuth (access_token starts with sk-ant-oat)
                if parsed.get("access_token", "").startswith("sk-ant-oat"):
                    return {"valid": True, "skipped": False, "oauth": True}
                # OpenAI OAuth (flat or nested Codex CLI format)
                if _CV.normalize_openai_oauth(parsed) is not None:
                    return {"valid": True, "skipped": False, "oauth": True}
        except (_json.JSONDecodeError, ValueError):
            pass

        # Strip _api_key suffix to get provider name
        provider = service.replace("_api_key", "")
        from src.setup_wizard import _VALIDATION_MODELS
        validation_model = _VALIDATION_MODELS.get(provider)
        if not validation_model:
            if base_url:
                # Custom provider with base URL — attempt OpenAI-compatible validation
                try:
                    import litellm
                    custom_kwargs: dict = {
                        "model": "openai/test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "api_key": key,
                        "api_base": base_url,
                    }
                    await litellm.acompletion(**custom_kwargs)
                    return {"valid": True, "skipped": False}
                except ImportError:
                    return {"valid": True, "skipped": True, "reason": "litellm not installed"}
                except Exception as e:
                    if isinstance(e, litellm.AuthenticationError):
                        return {"valid": False, "skipped": False, "reason": "Invalid API key"}
                    if isinstance(e, getattr(litellm, "PermissionDeniedError", type(None))):
                        return {"valid": False, "skipped": False, "reason": "Permission denied — check API key"}
                    emsg = str(e).lower()
                    _auth_kw = ("invalid api key", "invalid key", "unauthorized", "authentication fail")
                    if any(k in emsg for k in _auth_kw):
                        return {"valid": False, "skipped": False, "reason": "Invalid API key"}
                    # Non-auth errors (model not found, etc.) suggest key is probably valid
                    return {"valid": True, "skipped": False}
            return {"valid": True, "skipped": True, "reason": "unknown provider"}
        try:
            import litellm
            kwargs: dict = {
                "model": validation_model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "api_key": key,
            }
            if base_url:
                kwargs["api_base"] = base_url
            await litellm.acompletion(**kwargs)
            return {"valid": True, "skipped": False}
        except ImportError:
            return {"valid": True, "skipped": True, "reason": "litellm not installed"}
        except Exception as e:
            if isinstance(e, litellm.AuthenticationError):
                return {"valid": False, "skipped": False, "reason": "Invalid API key"}
            if isinstance(e, getattr(litellm, "PermissionDeniedError", type(None))):
                return {"valid": False, "skipped": False, "reason": "Permission denied — check API key"}
            # Some providers wrap auth errors as BadRequest/APIConnection —
            # check message before treating as transient
            emsg = str(e).lower()
            _auth_keywords = ("invalid api key", "invalid key", "invalid x-api-key",
                              "authentication fail", "login fail", "unauthorized",
                              "api key", "api_key", "secret key")
            if any(kw in emsg for kw in _auth_keywords):
                return {"valid": False, "skipped": False, "reason": "Invalid API key"}
            if isinstance(e, (litellm.Timeout, litellm.RateLimitError,
                              litellm.ServiceUnavailableError)):
                return {"valid": True, "skipped": True, "reason": str(e)[:200]}
            if isinstance(e, litellm.APIConnectionError):
                return {"valid": True, "skipped": True, "reason": str(e)[:200]}
            return {"valid": False, "skipped": False, "reason": f"Validation failed: {str(e)[:200]}"}

    @api_router.post("/api/credentials")
    async def api_add_credential(request: Request) -> dict:
        if credential_vault is None:
            raise HTTPException(status_code=503, detail="Credential vault not available")
        body = await request.json()
        service = body.get("service", "").strip()
        key = "".join(body.get("key", "").split())
        if not service or not key:
            raise HTTPException(status_code=400, detail="service and key are required")
        if not re.match(r"^[a-zA-Z0-9_.-]{1,128}$", service):
            raise HTTPException(
                status_code=400,
                detail="Invalid service name (alphanumeric, _, ., - only, max 128 chars)",
            )
        if len(key) > 10_000:
            raise HTTPException(status_code=400, detail="Key value too long (max 10000 chars)")
        # Detect OAuth JSON blobs (compact JSON after whitespace strip is still valid)
        import json as _json

        from src.host.credentials import CredentialVault as _CV
        try:
            parsed = _json.loads(key)
            if isinstance(parsed, dict):
                # Check for Anthropic OAuth (access_token starts with sk-ant-oat)
                if parsed.get("access_token", "").startswith("sk-ant-oat"):
                    credential_vault.store_anthropic_oauth(parsed)
                    credential_vault.remove_credential("anthropic_api_key")
                    return {"stored": True, "service": "anthropic_oauth", "tier": "system"}
                # Check for OpenAI OAuth (flat or nested Codex CLI format)
                normalized = _CV.normalize_openai_oauth(parsed)
                if normalized is not None:
                    credential_vault.store_openai_oauth(normalized)
                    return {"stored": True, "service": "openai_oauth", "tier": "system"}
        except (_json.JSONDecodeError, ValueError):
            pass
        # Detect bare Anthropic OAuth setup tokens (sk-ant-oat01-...)
        # Store as structured OAuth so they use the primary OAuth path
        from src.host.credentials import is_oauth_token
        if is_oauth_token(key):
            credential_vault.store_anthropic_oauth({"access_token": key})
            # Clear any stale api_key credential to avoid confusion
            credential_vault.remove_credential("anthropic_api_key")
            return {"stored": True, "service": "anthropic_oauth", "tier": "system"}
        # Normalize bare provider names
        from src.host.credentials import (
            SYSTEM_CREDENTIAL_PROVIDERS,
            is_system_credential,
        )
        if service.lower() in SYSTEM_CREDENTIAL_PROVIDERS and not service.lower().endswith("_api_key"):
            service = f"{service}_api_key"
        # Explicit tier override from request body, then auto-detect
        tier_field = body.get("tier", "").strip().lower()
        is_system = tier_field == "system" or is_system_credential(service)
        # Normalize custom LLM provider names to end with _api_key so
        # credential resolution and provider detection work correctly
        if body.get("custom_llm_models", "").strip() and not service.lower().endswith("_api_key"):
            service = f"{service.lower()}_api_key"
            is_system = True  # LLM provider keys are always system-tier
        credential_vault.add_credential(service, key, system=is_system)
        # If storing a regular Anthropic API key, clear any stale OAuth
        # credential so the API key path is used at runtime
        if service.lower() == "anthropic_api_key" and not is_oauth_token(key):
            if credential_vault._has_anthropic_oauth():
                credential_vault.remove_credential("anthropic_oauth")
        # Store optional custom API base URL alongside the key
        base_url = body.get("base_url", "").strip()
        if base_url:
            provider = service.replace("_api_key", "")
            credential_vault.add_credential(f"{provider}_api_base", base_url, system=is_system)
        # Store custom LLM provider models alongside the credential
        custom_models_raw = body.get("custom_llm_models", "").strip()
        if custom_models_raw and is_system:
            provider_name = service.replace("_api_key", "")
            models = [m.strip() for m in custom_models_raw.split(",") if m.strip()]
            models = [m if "/" in m else f"{provider_name}/{m}" for m in models]
            if models:
                with _settings_lock:
                    settings = _load_settings()
                    custom_providers = settings.setdefault("custom_llm_providers", {})
                    custom_providers[provider_name] = {
                        "label": body.get("custom_llm_label", "").strip() or provider_name.replace("_", " ").title(),
                        "models": models,
                    }
                    _save_settings(settings)
        tier = "system" if is_system else "agent"
        return {"stored": True, "service": service, "tier": tier}

    @api_router.post("/api/credentials/agent")
    async def api_add_agent_credential(request: Request) -> dict:
        """Store an agent-tier credential from a chat credential-request card.

        Unlike POST /api/credentials, this endpoint:
        - Always stores as agent-tier (never promotes to system credentials)
        - Rejects system credential names
        - Preserves the exact submitted value (only trims leading/trailing whitespace)
        """
        if credential_vault is None:
            raise HTTPException(status_code=503, detail="Credential vault not available")
        body = await request.json()
        service = body.get("service", "").strip()
        key = body.get("key", "").strip()
        if not service or not key:
            raise HTTPException(status_code=400, detail="service and key are required")
        if not re.match(r"^[a-zA-Z0-9_.-]{1,128}$", service):
            raise HTTPException(
                status_code=400,
                detail="Invalid credential name (alphanumeric, _, ., - only, max 128 chars)",
            )
        if len(key) > 10_000:
            raise HTTPException(status_code=400, detail="Key value too long (max 10000 chars)")
        from src.host.credentials import is_system_credential
        if is_system_credential(service):
            raise HTTPException(
                status_code=403,
                detail=f"Cannot store system credential via chat card: {service}",
            )
        credential_vault.add_credential(service, key)
        agent_id = body.get("agent_id", "").strip()
        request_id = (body.get("request_id", "") or "").strip()
        # Gate the steer + cross-client emit on an ATOMIC claim of the open
        # request, so a save racing a cancel resolves exactly once and the
        # agent never gets contradictory "use it" + "cancelled" steers. The
        # claim also POPS the registry record, clearing the "Needs you" row.
        # With no request_id (manual add) or no registry wired, there's nothing
        # to race — proceed (legacy direct path). Done in-process (the
        # dashboard is mounted on the mesh app): a loopback hop would add a
        # socket dependency and isn't needed since we hold the store ref.
        claimed = True
        if request_id and help_requests_store is not None:
            record = help_requests_store.resolve(
                request_id, expected_kind="credential_request", status="resolved",
            )
            claimed = record is not None
            # Prefer the registry's agent_id (authoritative) over the body's:
            # some card surfaces post the active chat id ('operator'), not the
            # requesting worker, so the steer would otherwise hit the wrong agent.
            if record and record.get("agent_id"):
                agent_id = record["agent_id"]
        if claimed:
            if agent_id and lane_manager is not None and agent_id in agent_registry:
                from src.shared.trace import new_trace_id
                try:
                    await lane_manager.enqueue(
                        agent_id,
                        f"The user just saved credential '{service}' to the vault. "
                        f"You can now use $CRED{{{service}}} in your requests.",
                        mode="steer",
                        trace_id=new_trace_id(),
                    )
                except Exception:
                    pass  # Best effort — credential is already stored
            if event_bus:
                event_bus.emit(
                    "credential_stored",
                    agent=agent_id or "",
                    data={
                        "name": service,
                        "service": service,
                        "request_id": request_id,
                        "agent_id": agent_id or None,
                    },
                )
        return {"stored": True, "service": service, "tier": "agent"}

    @api_router.post("/api/browser-login/complete")
    async def api_browser_login_complete(request: Request) -> dict:
        """User completed browser login — notify the requesting agent."""
        body = await request.json()
        agent_id = body.get("agent_id", "").strip()
        service = body.get("service", "").strip()[:128]
        request_id = (body.get("request_id", "") or "").strip()
        if not agent_id or not service:
            raise HTTPException(status_code=400, detail="agent_id and service are required")
        # Atomic claim gates the steer/emit (no save/cancel double-fire) and
        # pops the registry record so the "Needs you" row clears.
        claimed = True
        if request_id and help_requests_store is not None:
            claimed = help_requests_store.resolve(
                request_id, expected_kind="browser_login_request", status="resolved",
            ) is not None
        if not claimed:
            return {"completed": True, "agent_id": agent_id, "service": service}
        if agent_id in agent_registry and lane_manager is not None:
            from src.shared.trace import new_trace_id
            from src.shared.utils import sanitize_for_prompt
            try:
                msg = sanitize_for_prompt(
                    f"The user has completed the browser login for {service}. "
                    f"The session (cookies, localStorage) is now saved in your browser profile. "
                    f"You can resume using browser tools to interact with {service}."
                )
                await lane_manager.enqueue(
                    agent_id, msg, mode="steer", trace_id=new_trace_id(),
                )
            except Exception:
                pass
        if event_bus:
            event_bus.emit(
                "browser_login_completed", agent=agent_id,
                data={"service": service, "request_id": request_id},
            )
        return {"completed": True, "agent_id": agent_id, "service": service}

    @api_router.post("/api/browser-captcha-help/complete")
    async def api_browser_captcha_help_complete(request: Request) -> dict:
        """Phase 8 §11.14 — operator finished a CAPTCHA handoff."""
        body = await request.json()
        agent_id = body.get("agent_id", "").strip()
        service = body.get("service", "").strip()[:128]
        request_id = (body.get("request_id", "") or "").strip()
        if not agent_id or not service:
            raise HTTPException(
                status_code=400,
                detail="agent_id and service are required",
            )
        claimed = True
        if request_id and help_requests_store is not None:
            claimed = help_requests_store.resolve(
                request_id, expected_kind="browser_captcha_help_request", status="resolved",
            ) is not None
        if not claimed:
            return {"completed": True, "agent_id": agent_id, "service": service}
        if agent_id in agent_registry and lane_manager is not None:
            from src.shared.trace import new_trace_id
            from src.shared.utils import sanitize_for_prompt
            try:
                msg = sanitize_for_prompt(
                    f"The user has completed the CAPTCHA challenge for "
                    f"{service}. You can resume browser interaction; the "
                    f"page should now be past the captcha."
                )
                await lane_manager.enqueue(
                    agent_id, msg, mode="steer", trace_id=new_trace_id(),
                )
            except Exception:
                pass
        if event_bus:
            event_bus.emit(
                "browser_captcha_help_completed",
                agent=agent_id, data={"service": service, "request_id": request_id},
            )
        return {"completed": True, "agent_id": agent_id, "service": service}

    @api_router.post("/api/browser-captcha-help/cancel")
    async def api_browser_captcha_help_cancel(request: Request) -> dict:
        """Phase 8 §11.14 — operator cancelled a CAPTCHA handoff."""
        body = await request.json()
        agent_id = body.get("agent_id", "").strip()
        service = body.get("service", "").strip()[:128]
        if not agent_id or not service:
            raise HTTPException(
                status_code=400,
                detail="agent_id and service are required",
            )
        if agent_id in agent_registry and lane_manager is not None:
            from src.shared.trace import new_trace_id
            from src.shared.utils import sanitize_for_prompt
            try:
                msg = sanitize_for_prompt(
                    f"The user cancelled the CAPTCHA help request for "
                    f"{service}. Try a different approach (e.g. wait + "
                    f"retry, escalate to user via notify_user)."
                )
                await lane_manager.enqueue(
                    agent_id, msg, mode="steer", trace_id=new_trace_id(),
                )
            except Exception:
                pass
        if event_bus:
            event_bus.emit(
                "browser_captcha_help_cancelled",
                agent=agent_id, data={"service": service},
            )
        return {"cancelled": True, "agent_id": agent_id, "service": service}

    @api_router.post("/api/browser-login/cancel")
    async def api_browser_login_cancel(request: Request) -> dict:
        """User cancelled browser login — notify the requesting agent."""
        body = await request.json()
        agent_id = body.get("agent_id", "").strip()
        service = body.get("service", "").strip()[:128]
        if not agent_id or not service:
            raise HTTPException(status_code=400, detail="agent_id and service are required")
        if agent_id in agent_registry and lane_manager is not None:
            from src.shared.trace import new_trace_id
            from src.shared.utils import sanitize_for_prompt
            try:
                msg = sanitize_for_prompt(
                    f"The user cancelled the browser login for {service}. "
                    f"You may need to find an alternative approach or ask again later."
                )
                await lane_manager.enqueue(
                    agent_id, msg, mode="steer", trace_id=new_trace_id(),
                )
            except Exception:
                pass
        if event_bus:
            event_bus.emit("browser_login_cancelled", agent=agent_id, data={"service": service})
        return {"cancelled": True, "agent_id": agent_id, "service": service}

    @api_router.post("/api/credentials/upload-env")
    async def api_upload_env(request: Request, file: UploadFile = File(...)) -> dict:
        """Bulk-import credentials from an uploaded .env file.

        Parses KEY=VALUE pairs (skips comments and blank lines) and stores each
        as a credential in the vault.  Values are never logged or returned.
        Returns the count of credentials loaded and the list of key names only.
        """
        if credential_vault is None:
            raise HTTPException(status_code=503, detail="Credential vault not available")

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > 64 * 1024:
                    raise HTTPException(status_code=413, detail="File too large (max 64KB)")
            except ValueError:
                pass  # Malformed header; body-length check below is authoritative

        content = await file.read(65537)  # 64*1024 + 1
        if len(content) > 64 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 64KB)")
        if not content.strip():
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

        from src.host.credentials import is_system_credential

        loaded_keys: list[str] = []
        parse_errors: list[str] = []

        for line_num, raw_line in enumerate(text.splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                parse_errors.append(f"Line {line_num}: missing '=' separator")
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if not _ENV_KEY_RE.match(key):
                parse_errors.append(f"Line {line_num}: invalid key name '{key}'")
                continue
            if not value:
                parse_errors.append(f"Line {line_num}: empty value for key '{key}'")
                continue
            is_system = is_system_credential(key)
            try:
                credential_vault.add_credential(key, value, system=is_system)
                loaded_keys.append(key)
            except Exception as exc:
                parse_errors.append(f"Line {line_num}: failed to store '{key}': {exc}")

        if not loaded_keys and parse_errors:
            raise HTTPException(
                status_code=400,
                detail=f"No valid credentials found. Errors: {'; '.join(parse_errors)}",
            )

        return {"count": len(loaded_keys), "keys": loaded_keys, "errors": parse_errors}

    @api_router.delete("/api/credentials/{name}")
    async def api_remove_credential(name: str) -> dict:
        if credential_vault is None:
            raise HTTPException(status_code=503, detail="Credential vault not available")
        existed = credential_vault.remove_credential(name)
        if not existed:
            raise HTTPException(status_code=404, detail=f"Credential '{name}' not found")
        # Clean up custom LLM provider config and paired api_base
        if name.endswith("_api_key"):
            provider_name = name[: -len("_api_key")]
            credential_vault.remove_credential(f"{provider_name}_api_base")
            with _settings_lock:
                settings = _load_settings()
                custom_providers = settings.get("custom_llm_providers", {})
                if provider_name in custom_providers:
                    del custom_providers[provider_name]
                    _save_settings(settings)
        return {"removed": True, "service": name}

    @api_router.get("/api/credentials/{name}/value")
    async def api_credential_value(name: str, request: Request) -> dict:
        """Reveal a masked credential value. Dashboard-auth gated."""
        _verify_dashboard_auth(request)
        if credential_vault is None:
            raise HTTPException(status_code=503, detail="Credential vault not available")
        # Check agent tier first, then system tier
        value = credential_vault.resolve_credential(name)
        if value is None:
            value = credential_vault.system_credentials.get(name.lower())
        if value is None:
            value = credential_vault.api_bases.get(name.lower())
        if value is None:
            raise HTTPException(status_code=404, detail=f"Credential '{name}' not found")
        masked = value[-4:].rjust(len(value), "*") if len(value) > 4 else "****"
        return {"name": name, "value": masked}

    # ── OAuth integrations (one-click Connect) ──────────────
    #
    # Bring-your-own-app flow: the operator registers their own OAuth app and
    # supplies client_id/secret once (POST .../setup, stored system-tier), then
    # connects via the browser redirect dance. The resulting connection resolves
    # as ``$CRED{<name>}`` with transparent token refresh — agents never touch
    # the refresh token. The connect/callback GETs inherit ``_verify_dashboard_auth``
    # (cookie rides Google's redirect) and are CSRF-exempt as safe methods; the
    # single-use, session-bound ``state`` is the real CSRF guard.

    @api_router.get("/api/integrations")
    async def api_list_integrations(request: Request) -> dict:
        """List providers, whether their client is configured, and connections."""
        from src.host.oauth_providers import OAUTH_PROVIDERS
        if credential_vault is None:
            return {"providers": []}
        conns_by_provider: dict[str, list] = {}
        for c in credential_vault.list_connections():
            conns_by_provider.setdefault(c["provider"], []).append(c)
        providers = []
        for key, p in OAUTH_PROVIDERS.items():
            providers.append({
                "key": key,
                "label": p.label,
                "configured": credential_vault.has_oauth_client(p),
                "redirect_uri": _oauth_redirect_uri(request, key),
                "scope_bundles": [
                    {"key": b.key, "label": b.label, "description": b.description}
                    for b in p.scope_bundles
                ],
                "connections": conns_by_provider.get(key, []),
            })
        return {"providers": providers}

    @api_router.post("/api/integrations/{provider}/setup")
    async def api_setup_integration(provider: str, request: Request) -> dict:
        """Store the operator's OAuth client id/secret for a provider (system-tier)."""
        from src.host.oauth_providers import get_provider
        p = get_provider(provider)
        if p is None:
            raise HTTPException(404, f"Unknown provider: {provider}")
        if credential_vault is None:
            raise HTTPException(503, "Credential vault not available")
        body = await request.json()
        client_id = body.get("client_id", "").strip()
        client_secret = body.get("client_secret", "").strip()
        if not client_id or not client_secret:
            raise HTTPException(400, "client_id and client_secret are required")
        credential_vault.add_credential(p.client_id_key, client_id, system=True)
        credential_vault.add_credential(p.client_secret_key, client_secret, system=True)
        _emit_config_changed("integrations", provider=provider)
        return {
            "configured": True,
            "provider": provider,
            "redirect_uri": _oauth_redirect_uri(request, provider),
        }

    @api_router.get("/integrations/{provider}/connect")
    async def integration_connect(
        provider: str, request: Request, name: str = "", scopes: str = "",
    ):
        """Begin the OAuth dance: mint state, redirect to the provider consent."""
        from fastapi.responses import RedirectResponse

        from src.host.oauth_providers import generate_pkce, get_provider
        p = get_provider(provider)
        if p is None:
            raise HTTPException(404, f"Unknown provider: {provider}")
        if credential_vault is None:
            raise HTTPException(503, "Credential vault not available")
        if not credential_vault.has_oauth_client(p):
            raise HTTPException(400, f"{p.label} OAuth client not configured")
        bundle_keys = [s.strip() for s in scopes.split(",") if s.strip()]
        # Reject unknown bundle keys rather than silently under-scoping a
        # connection that would then look "connected" but lack data access.
        unknown = [k for k in bundle_keys if p.bundle(k) is None]
        if unknown:
            raise HTTPException(400, f"Unknown scope bundle(s): {', '.join(unknown)}")
        resolved_scopes = p.resolve_scopes(bundle_keys)
        conn_name = re.sub(r"[^a-z0-9_]", "_", (name or provider).strip().lower())[:64]
        conn_name = conn_name.strip("_") or provider
        verifier, challenge = generate_pkce()
        redirect_uri = _oauth_redirect_uri(request, provider)
        state = _oauth_state_store.create(
            provider=provider,
            connection_name=conn_name,
            scopes=tuple(resolved_scopes),
            code_verifier=verifier,
            redirect_uri=redirect_uri,
            session_hash=_oauth_session_hash(request),
        )
        url = credential_vault.build_authorize_url(
            p, redirect_uri=redirect_uri, state=state,
            scopes=resolved_scopes, code_challenge=challenge,
        )
        return RedirectResponse(url, status_code=302)

    @api_router.get("/integrations/{provider}/callback")
    async def integration_callback(
        provider: str, request: Request,
        code: str = "", state: str = "", error: str = "",
    ):
        """Provider redirect target: validate state, exchange code, store conn."""
        from fastapi.responses import RedirectResponse

        from src.host.oauth_providers import get_provider
        landing = "/dashboard/"

        def _back(params: str):
            return RedirectResponse(f"{landing}?{params}", status_code=302)

        if error:
            return _back(f"integration_error={re.sub(r'[^a-zA-Z0-9_-]', '', error)[:64]}")
        p = get_provider(provider)
        if p is None or credential_vault is None:
            return _back("integration_error=unknown_provider")
        pending = _oauth_state_store.consume(
            state, session_hash=_oauth_session_hash(request),
        )
        if pending is None or pending.provider != provider or not code:
            return _back("integration_error=invalid_state")
        try:
            conn = await credential_vault.exchange_oauth_code(
                p, code=code, redirect_uri=pending.redirect_uri,
                code_verifier=pending.code_verifier,
            )
            # A connection that needs refresh but came back without a refresh
            # token would silently die at first expiry — reject it now with a
            # clear, actionable error instead of storing a dead connection.
            if p.refresh_required and not conn.get("refresh_token"):
                logger.warning(
                    "OAuth callback for %s returned no refresh_token; rejecting",
                    provider,
                )
                return _back("integration_error=no_refresh_token")
            credential_vault.store_connection(pending.connection_name, conn)
        except Exception as exc:  # noqa: BLE001 — surface as a UI banner
            logger.warning("OAuth callback exchange failed for %s: %s", provider, exc)
            return _back("integration_error=exchange_failed")
        logger.info(
            "Integration connected: %s (%s)", pending.connection_name, provider,
        )
        _emit_config_changed("integrations", name=pending.connection_name, provider=provider)
        return _back(f"integration_connected={pending.connection_name}")

    # ── MCP connector OAuth (paste URL → Connect) ─────────────
    #
    # Same dance as the provider flow above, but the endpoints are
    # DISCOVERED from the remote MCP server (RFC 9728 → 8414) and the
    # client identity is minted via Dynamic Client Registration (RFC
    # 7591) — there is no registry entry and no BYO client-id fallback
    # (plan §11-Q4: a server without DCR gets a diagnosable error).
    # Everything discovery returns is server-controlled input and goes
    # through the D16 SSRF posture in src/host/mcp_oauth.py. The
    # discovered token endpoint + client identity ride the single-use
    # state entry (server-side) and end up EMBEDDED in the connection
    # blob so refresh-on-resolve works with no registry (plan §7.1).

    @api_router.get("/integrations/mcp/{name}/connect")
    async def mcp_connector_connect(name: str, request: Request):
        """Begin the OAuth dance for a remote MCP connector."""
        from fastapi.responses import RedirectResponse

        from src.host.mcp_oauth import MCPOAuthError, discover, register_client
        from src.host.oauth_providers import generate_pkce
        from src.shared.types import HttpConnector
        if credential_vault is None or connector_store is None:
            raise HTTPException(503, "Connector catalog or vault not available")
        c = connector_store.get(name)
        if not isinstance(c, HttpConnector):
            raise HTTPException(404, f"Unknown remote connector: {name}")
        redirect_uri = (
            f"{_public_base_url(request)}/dashboard/integrations/mcp/"
            f"{c.name}/callback"
        )
        try:
            disco = await discover(c.url)
            if disco.registration_endpoint is None:
                raise MCPOAuthError(
                    "client registration",
                    "the authorization server does not support Dynamic "
                    "Client Registration — one-click Connect is not "
                    "possible for this server",
                )
            client_id, client_secret = await register_client(
                disco.registration_endpoint, redirect_uri,
            )
        except MCPOAuthError as exc:
            raise HTTPException(
                502, f"OAuth setup failed at {exc.step}: {exc}",
            ) from exc
        verifier, challenge = generate_pkce()
        conn_name = re.sub(r"[^a-z0-9_]", "_", f"mcp_{c.name.lower()}")[:64]
        state = _oauth_state_store.create(
            provider=f"mcp:{c.name.lower()}",
            connection_name=conn_name,
            scopes=(),  # AS defaults; MCP servers scope via `resource`
            code_verifier=verifier,
            redirect_uri=redirect_uri,
            session_hash=_oauth_session_hash(request),
            extra={
                "token_endpoint": disco.token_endpoint,
                "client_id": client_id,
                "client_secret": client_secret or "",
                "resource": c.url,
            },
        )
        from urllib.parse import urlencode
        params = urlencode({
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "resource": c.url,  # RFC 8707
        })
        sep = "&" if "?" in disco.authorization_endpoint else "?"
        return RedirectResponse(
            f"{disco.authorization_endpoint}{sep}{params}", status_code=302,
        )

    @api_router.get("/integrations/mcp/{name}/callback")
    async def mcp_connector_callback(
        name: str, request: Request,
        code: str = "", state: str = "", error: str = "",
    ):
        """AS redirect target for a remote MCP connector."""
        from fastapi.responses import RedirectResponse

        from src.shared.types import ConnectorAuth, HttpConnector

        def _back(params: str):
            return RedirectResponse(f"/dashboard/?{params}", status_code=302)

        if error:
            return _back(
                f"integration_error={re.sub(r'[^a-zA-Z0-9_-]', '', error)[:64]}",
            )
        if credential_vault is None or connector_store is None:
            return _back("integration_error=unavailable")
        pending = _oauth_state_store.consume(
            state, session_hash=_oauth_session_hash(request),
        )
        if (
            pending is None
            or pending.provider != f"mcp:{name.lower()}"
            or not code
        ):
            return _back("integration_error=invalid_state")
        extra = pending.extra
        try:
            conn = await credential_vault.exchange_code_dynamic(
                token_endpoint=extra["token_endpoint"],
                client_id=extra["client_id"],
                client_secret=extra.get("client_secret") or None,
                code=code,
                redirect_uri=pending.redirect_uri,
                code_verifier=pending.code_verifier,
                resource=extra.get("resource") or None,
                provider_label=pending.provider,
            )
            # §11-Q3 (deliberate divergence from the Google flow's
            # refresh_required rejection): many MCP authorization
            # servers issue short-lived tokens with NO refresh token.
            # Accept them — expiry surfaces as needs_auth on probe,
            # which renders the reconnect affordance.
            credential_vault.store_connection(pending.connection_name, conn)
        except Exception as exc:  # noqa: BLE001 — surface as a UI banner
            logger.warning("MCP OAuth exchange failed for %s: %s", name, exc)
            return _back("integration_error=exchange_failed")
        # Bind the connection to the connector. The gateway's discovery
        # cache drops so a previously-401 connector re-discovers with
        # the new auth. The FIRST bind also marks assigned agents
        # pending-restart: they registered this connector's tools (zero,
        # pre-auth) at their last boot, so the connection only reaches
        # them after a bounce — without the dirty mark nothing ever
        # prompts it. A RE-connect of an already-bound connector stays
        # restart-free (rotation semantics, plan D12).
        c = connector_store.get(name)
        if isinstance(c, HttpConnector):
            first_bind = c.auth.kind != "oauth" or not c.auth.connection
            bound = c.model_copy(update={
                "auth": ConnectorAuth(
                    kind="oauth", connection=pending.connection_name,
                ),
            })
            import asyncio as _asyncio
            await _asyncio.to_thread(connector_store.upsert, bound)
            if mcp_gateway is not None:
                mcp_gateway.invalidate(c.name)
            if first_bind:
                connector_store.mark_dirty(_expand_assignment(bound.agents))
            logger.info(
                "MCP connector connected: %s → %s (first_bind=%s)",
                name, pending.connection_name, first_bind,
            )
        else:
            # Connector deleted (or replaced with stdio) mid-flow: the
            # exchanged connection is stored but unbound — visible on
            # the integrations list, removable via disconnect. Say so
            # rather than claiming success silently.
            logger.warning(
                "MCP OAuth callback for %r: connector no longer exists; "
                "connection %r stored unbound",
                name, pending.connection_name,
            )
        _emit_config_changed("connectors", name=name)
        return _back(f"integration_connected={pending.connection_name}")

    @api_router.post("/api/integrations/{name}/disconnect")
    async def api_disconnect_integration(name: str, request: Request) -> dict:
        """Revoke (best-effort) and remove an OAuth connection."""
        if credential_vault is None:
            raise HTTPException(503, "Credential vault not available")
        await credential_vault.revoke_connection(name)
        existed = credential_vault.remove_credential(name)
        if not existed:
            raise HTTPException(404, f"Connection '{name}' not found")
        _emit_config_changed("integrations", name=name)
        return {"removed": True, "name": name}

    # ── External API key management ─────────────────────────

    @api_router.get("/api/external-api-keys")
    async def api_list_external_keys(request: Request) -> dict:
        """List all named API keys (metadata only, never raw keys)."""
        _verify_dashboard_auth(request)
        if api_key_manager is None:
            return {"keys": [], "legacy": False}
        keys = api_key_manager.list_keys()
        legacy = bool(os.environ.get("OPENLEGION_API_KEY", ""))
        return {"keys": keys, "legacy": legacy}

    @api_router.post("/api/external-api-keys")
    async def api_create_external_key(request: Request) -> dict:
        """Create a named API key. Returns the raw key once."""
        _verify_dashboard_auth(request)
        if api_key_manager is None:
            raise HTTPException(status_code=503, detail="API key manager not available")
        body = await request.json()
        name = (body.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if len(name) > 128:
            raise HTTPException(status_code=400, detail="name must be 128 characters or fewer")
        try:
            key_id, raw_key = api_key_manager.create_key(name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _emit_config_changed("api_keys", id=key_id)
        return {"id": key_id, "name": name, "key": raw_key}

    @api_router.delete("/api/external-api-keys/{key_id}")
    async def api_revoke_external_key(key_id: str, request: Request) -> dict:
        """Revoke an API key by ID."""
        _verify_dashboard_auth(request)
        if api_key_manager is None:
            raise HTTPException(status_code=503, detail="API key manager not available")
        if not api_key_manager.revoke_key(key_id):
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}")
        _emit_config_changed("api_keys", id=key_id)
        return {"revoked": True, "id": key_id}

    # ── Dashboard telemetry (Phase -1 onboarding wizard) ────────

    @api_router.post("/api/telemetry")
    async def api_telemetry_record(request: Request) -> dict:
        """Persist one frontend telemetry event.

        Hypothesis-test surface for the Phase -1 empty-fleet onboarding
        wizard. Each session gets a sliding 60-event/min budget; over
        budget returns HTTP 429. Events with unknown names are accepted
        (no allowlist) but capped in length so a malicious / runaway
        client cannot fill the table with arbitrary garbage. ``ts`` from
        the client is honored only as advisory; the server records its
        own wall clock so we cannot be back-dated.
        """
        if telemetry is None:
            # Telemetry sink failed to init at startup. Don't crash the
            # frontend on every event — return 204 so the JS keeps
            # firing without affecting wizard UX.
            return {"recorded": False, "reason": "telemetry_disabled"}
        body: dict[str, Any]
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON body")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be an object")
        event_name = (body.get("event_name") or body.get("event") or "").strip()
        if not event_name:
            raise HTTPException(status_code=400, detail="event_name is required")
        props = body.get("props") or {}
        if not isinstance(props, dict):
            raise HTTPException(status_code=400, detail="props must be an object")

        session_id = _operator_session_id(request)
        allowed, retry_ms = telemetry.check_rate_limit(session_id)
        if not allowed:
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "recorded": False,
                    "error": {
                        "code": "rate_limited",
                        "message": "telemetry rate limit exceeded",
                        "retry_after_ms": retry_ms,
                    },
                },
                headers={"Retry-After": str(max(1, retry_ms // 1000))},
            )

        try:
            row_id = telemetry.record(
                event_name=event_name,
                session_id=session_id,
                props=props,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"recorded": True, "id": row_id}

    # ── Wallet management ────────────────────────────────────

    @api_router.post("/api/wallet/init")
    async def api_wallet_init(request: Request) -> dict:
        """Generate a master wallet seed and store it in .env."""
        _verify_dashboard_auth(request)
        if os.environ.get("OPENLEGION_SYSTEM_WALLET_MASTER_SEED"):
            raise HTTPException(
                status_code=409,
                detail="Master seed already configured. Remove "
                "OPENLEGION_SYSTEM_WALLET_MASTER_SEED from .env to reset.",
            )
        try:
            from mnemonic import Mnemonic
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="mnemonic package not installed. Run: pip install mnemonic",
            )
        from src.host.credentials import _persist_to_env

        mnemo = Mnemonic("english")
        words = mnemo.generate(strength=256)
        _persist_to_env("OPENLEGION_SYSTEM_WALLET_MASTER_SEED", words)
        os.environ["OPENLEGION_SYSTEM_WALLET_MASTER_SEED"] = words

        # Hot-load WalletService into the shared ref so mesh endpoints
        # work immediately — no restart needed.
        addresses = {}
        _ws_ref = wallet_service_ref or [None]
        try:
            from src.host.wallet import WalletService

            ws = WalletService()
            _ws_ref[0] = ws
            addresses["evm"] = ws._derive_evm_account(0).address
            addresses["solana"] = str(ws._derive_solana_keypair(0).pubkey())
        except Exception as e:
            logger.warning("WalletService init failed: %s", e)

        _emit_config_changed("wallet")
        from starlette.responses import JSONResponse
        return JSONResponse(
            content={"initialized": True, "seed": words, "sample_addresses": addresses},
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @api_router.get("/api/wallet/seed")
    async def api_wallet_seed(request: Request) -> dict:
        """Seed reveal removed for security.

        The seed is shown once at /api/wallet/init time. After that,
        revealing it requires re-provisioning the wallet. This prevents
        exfiltration via XSS or CSRF on the dashboard.
        """
        raise HTTPException(
            status_code=410,
            detail="Seed reveal disabled. The seed was shown at wallet init time. "
            "Re-initialize the wallet to generate a new seed.",
        )

    @api_router.get("/api/wallet/addresses")
    async def api_wallet_addresses(request: Request) -> dict:
        """List all agent wallet addresses."""
        _verify_dashboard_auth(request)
        seed = os.environ.get("OPENLEGION_SYSTEM_WALLET_MASTER_SEED")
        if not seed:
            return {"configured": False, "agents": []}

        _ws_ref = wallet_service_ref or [None]
        ws = _ws_ref[0]
        temp_ws = None
        if ws is None:
            try:
                from src.host.wallet import WalletService
                temp_ws = WalletService()
                ws = temp_ws
            except Exception as e:
                return {"configured": True, "agents": [], "error": str(e)}

        try:
            # Build agent list from the live registry (not stale DB entries).
            # Derive addresses for agents that have wallet enabled.
            live_agents = set(agent_registry.keys())
            agents = []
            all_agent_wallet_status = []

            for aid in sorted(live_agents):
                enabled = permissions.can_use_wallet(aid) if permissions else False
                chains = (
                    permissions.get_permissions(aid).wallet_allowed_chains
                    if permissions else []
                )
                status_entry: dict = {
                    "agent_id": aid,
                    "wallet_enabled": enabled,
                    "wallet_chains": chains,
                }
                if enabled:
                    try:
                        evm = await ws.get_address(aid, "evm:ethereum")
                        sol = await ws.get_address(aid, "solana:mainnet")
                        agents.append({
                            "agent_id": aid,
                            "evm_address": evm,
                            "solana_address": sol,
                        })
                        status_entry["has_addresses"] = True
                    except Exception:
                        status_entry["has_addresses"] = False
                else:
                    status_entry["has_addresses"] = False
                all_agent_wallet_status.append(status_entry)

            return {
                "configured": True,
                "agents": agents,
                "all_agents": all_agent_wallet_status,
            }
        except Exception as e:
            return {"configured": True, "agents": [], "error": str(e)}
        finally:
            if temp_ws is not None:
                temp_ws.close()

    @api_router.post("/api/wallet/enable/{agent_id}")
    async def api_wallet_enable_agent(agent_id: str, request: Request) -> dict:
        """Quick-enable wallet for an agent with all chains."""
        _verify_dashboard_auth(request)
        if permissions is None:
            raise HTTPException(status_code=503, detail="Permissions not available")
        from src.cli.config import _load_permissions, _save_permissions

        perms_data = _load_permissions()
        agent_perms = perms_data.get("permissions", {}).get(agent_id, {})
        agent_perms["can_use_wallet"] = True
        if not agent_perms.get("wallet_allowed_chains"):
            # Default to all known chains (not wildcard "*")
            _ws_local = (wallet_service_ref or [None])[0]
            if _ws_local:
                agent_perms["wallet_allowed_chains"] = list(_ws_local.chains.keys())
            else:
                agent_perms["wallet_allowed_chains"] = ["*"]
        perms_data.setdefault("permissions", {})[agent_id] = agent_perms
        _save_permissions(perms_data)
        permissions.reload()
        _emit_config_changed("wallet", agent=agent_id)
        return {"enabled": True, "agent_id": agent_id}

    @api_router.get("/api/wallet/rpc")
    async def api_wallet_rpc(request: Request) -> dict:
        """List RPC URLs for all chains (current + default)."""
        _verify_dashboard_auth(request)
        from src.host.wallet import _CHAINS

        chains = []
        for chain_id, cfg in _CHAINS.items():
            env_key = cfg["rpc_env"]
            custom = os.environ.get(env_key, "")
            chains.append({
                "chain_id": chain_id,
                "label": _wallet_chain_label(chain_id, cfg),
                "rpc_env": env_key,
                "rpc_default": cfg["rpc_default"],
                "rpc_current": custom or cfg["rpc_default"],
                "is_custom": bool(custom),
            })
        return {"chains": chains}

    @api_router.put("/api/wallet/rpc")
    async def api_wallet_rpc_update(request: Request) -> dict:
        """Set or clear a custom RPC URL for a chain."""
        _verify_dashboard_auth(request)
        body = await request.json()
        chain_id = body.get("chain_id", "")
        rpc_url = body.get("rpc_url", "").strip()

        from src.host.wallet import _CHAINS

        if chain_id not in _CHAINS:
            raise HTTPException(status_code=400, detail=f"Unknown chain: {chain_id}")

        env_key = _CHAINS[chain_id]["rpc_env"]
        from src.host.credentials import _persist_to_env, _remove_from_env

        if rpc_url:
            # Validate URL format
            if not rpc_url.startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=400, detail="RPC URL must start with http:// or https://",
                )
            _persist_to_env(env_key, rpc_url)
            os.environ[env_key] = rpc_url
        else:
            # Clear custom → revert to default
            _remove_from_env(env_key)
            os.environ.pop(env_key, None)

        # Hot-reload chains in the wallet service
        _ws_local = (wallet_service_ref or [None])[0]
        if _ws_local is not None:
            _ws_local._chains = _ws_local._load_chains()
            # Clear cached providers so they reconnect with new URLs
            _ws_local._evm_providers.pop(chain_id, None)
            _ws_local._solana_clients.pop(chain_id, None)

        _emit_config_changed("wallet", chain_id=chain_id)
        return {"updated": True, "chain_id": chain_id}

    # ── Cost detail per agent ────────────────────────────────

    @api_router.get("/api/costs/{agent_id}")
    async def api_agent_costs(agent_id: str, period: str = "today") -> dict:
        if period not in {"today", "week", "month"}:
            period = "today"
        return cost_tracker.get_spend(agent_id, period)

    # ── Cost dashboard ───────────────────────────────────────

    _VALID_PERIODS = {"today", "week", "month"}

    @api_router.get("/api/costs")
    async def api_costs(period: str = "today") -> dict:
        if period not in _VALID_PERIODS:
            period = "today"
        agents_spend = cost_tracker.get_all_agents_spend(period)
        budgets = {}
        for item in agents_spend:
            budgets[item["agent"]] = cost_tracker.check_budget(item["agent"])
        # Include budgets for registered agents with zero spend in this period
        for aid in agent_registry:
            if aid not in budgets:
                budgets[aid] = cost_tracker.check_budget(aid)
        by_model = cost_tracker.get_spend_by_model(period)
        # Always include month-to-date totals for the stat card
        if period != "month":
            month = cost_tracker.get_spend(period="month")
            month_total = month["total_cost"]
            month_tokens = month["total_tokens"]
        else:
            month_total = sum(a["cost"] for a in agents_spend)
            month_tokens = sum(a["tokens"] for a in agents_spend)
        return {
            "period": period,
            "agents": agents_spend,
            "budgets": budgets,
            "by_model": by_model,
            "month_total": month_total,
            "month_tokens": month_tokens,
        }

    # ── Phase 10 §24 — per-tenant CAPTCHA cost rollup (CSV export) ──────
    #
    # Operator-facing billing reconciliation. Returns the sum of CAPTCHA
    # solver spend across every agent inside the requested tenant
    # (team), broken down per-agent + a rolled-up total row. CSV is
    # the lingua franca for finance/billing tooling — keep the schema
    # stable and the column names self-explanatory.
    #
    # Auth posture: this endpoint inherits ``_verify_dashboard_auth`` and
    # ``_csrf_check`` from the router-level dependencies. ``_csrf_check``
    # is a no-op on GET (per its own docstring) so this endpoint is
    # callable from a plain browser tab as long as the operator is
    # logged in via ``ol_session``. State-changing endpoints in the same
    # router DO require ``X-Requested-With``.
    _VALID_BILLING_PERIODS = {"daily", "weekly", "monthly"}

    @api_router.get("/api/billing/captcha-rollup")
    async def api_billing_captcha_rollup(
        tenant: str = "", period: str = "monthly",
    ):
        """CSV export of per-tenant CAPTCHA spend rollup."""
        from datetime import datetime, timedelta, timezone

        from starlette.responses import PlainTextResponse

        from src.browser import captcha_cost_counter as _ccc

        if not tenant:
            raise HTTPException(400, "Missing required query param: tenant")
        if period not in _VALID_BILLING_PERIODS:
            raise HTTPException(
                400,
                f"Invalid period {period!r}; "
                f"expected one of {sorted(_VALID_BILLING_PERIODS)}",
            )

        # Resolve the period_start timestamp — used as the first column
        # of every row so the export is self-describing (a row carries
        # its own period without relying on filename context).
        now = datetime.now(timezone.utc)
        if period == "daily":
            period_start = now.replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
        elif period == "weekly":
            period_start = (now - timedelta(days=7)).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
        else:  # monthly
            period_start = now.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0,
            )

        # Walk per-agent buckets via the new tenant helpers. The
        # in-memory state is current-month only (see captcha_cost_counter
        # docstring for the trim rationale), so for ``daily`` and
        # ``weekly`` we still report the same per-agent monthly buckets
        # — operators get a month-to-date number with a ``data_scope``
        # column flagging that the data IS NOT period-correct.  Older
        # buckets would require persisted snapshots, deferred per
        # §11.10's SQLite trim.  The honest column on every row beats
        # a billing footgun where finance reconciles a "daily" CSV
        # against month-to-date numbers.
        breakdown = await _ccc.get_tenant_breakdown(tenant)
        total_millicents = await _ccc.get_tenant_total(
            tenant, since=period_start,
        )
        # ``monthly_actual`` for monthly (the in-memory state IS the
        # current month, so the number is correct for the requested
        # period); ``current_month_aggregate`` for daily/weekly since
        # we surface month-to-date data with the requested period_start.
        data_scope = (
            "monthly_actual" if period == "monthly"
            else "current_month_aggregate"
        )

        # CSV assembly — manual rather than ``csv`` module so we can
        # guarantee the row order (sorted agent_id, then synthetic
        # total) and avoid the implicit dialect quoting around plain
        # IDs. Agent IDs use ``[A-Za-z0-9_-]`` per AGENT_ID_RE_PATTERN
        # so no quoting is needed for the agent_id column.  ``data_scope``
        # values are static literals — also no quoting needed.
        period_start_iso = period_start.isoformat().replace("+00:00", "Z")
        lines: list[str] = [
            "period_start,agent_id,millicents,dollars,data_scope"
        ]
        for agent_id in sorted(breakdown):
            mc = breakdown[agent_id]
            dollars = mc / 100_000.0
            lines.append(
                f"{period_start_iso},{agent_id},{mc},{dollars:.5f},"
                f"{data_scope}"
            )
        # Synthetic tenant-total row — operators reading the CSV in a
        # spreadsheet immediately see the rolled-up number without
        # having to re-sum. Prefix the agent_id column with double
        # underscores so it sorts after real agent IDs and is visually
        # distinct.
        lines.append(
            f"{period_start_iso},__tenant_total__,{total_millicents},"
            f"{total_millicents / 100_000.0:.5f},{data_scope}"
        )
        body = "\n".join(lines) + "\n"
        # Sanitize the tenant for the Content-Disposition filename —
        # AGENT_ID_RE_PATTERN-style chars only; everything else collapses
        # to underscore so we never echo arbitrary path/quote bytes
        # into a header.
        safe_tenant = re.sub(r"[^A-Za-z0-9_.-]", "_", tenant)[:64]
        filename = f"captcha-rollup-{safe_tenant}-{period}.csv"
        return PlainTextResponse(
            content=body,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    # ── Teams ─────────────────────────────────────────────────
    # Team CRUD routes. Storage is the TeamStore (data/teams.db);
    # the on-disk ``config/teams/{id}/`` scaffold (team.md +
    # workflows/) is owned by the store's create/delete lifecycle.

    @api_router.get("/api/teams")
    async def api_teams_list() -> dict:
        """List all teams with members."""
        teams_cfg = teams_store.list_teams()
        result = []
        sorted_teams = sorted(teams_cfg.items(), key=lambda x: x[1].get("created_at") or "")
        for i, (pname, pdata) in enumerate(sorted_teams):
            is_over = _teams_disabled or (_max_teams > 0 and i >= _max_teams)
            result.append({
                "name": pname,
                "team_name": pname,
                "description": pdata.get("description", ""),
                "members": pdata.get("members", []),
                "created_at": pdata.get("created_at", ""),
                "over_limit": is_over,
                # Team goal (set via the operator's set_team_goal tool). Surfaced
                # so the team hub can show it; null until set.
                "north_star": pdata.get("north_star"),
            })
        return {"teams": result}

    @api_router.post("/api/teams")
    async def api_teams_create(request: Request) -> dict:
        """Create a new team."""
        if _teams_disabled:
            raise HTTPException(
                status_code=403,
                detail="Teams are not available on your current plan. Upgrade to enable teams.",
            )
        if _max_teams > 0:
            current_count = teams_store.count_teams()
            if current_count >= _max_teams:
                raise HTTPException(
                    status_code=403,
                    detail=f"Team limit reached ({_max_teams}). Upgrade your plan for more teams.",
                )
        from src.cli.config import (
            _add_team_blackboard_permissions,
            _load_config,
            _remove_team_blackboard_permissions,
        )
        body = await request.json()
        name = body.get("name", "").strip()
        description = sanitize_for_prompt(body.get("description", "")).strip()
        members = body.get("members", [])
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not isinstance(members, list):
            raise HTTPException(status_code=400, detail="members must be a list")
        # Validate that member agents exist in the config
        cfg = _load_config()
        known_agents = set(cfg.get("agents", {}).keys())
        # Cross-namespace collision guard (ratified #5): a solo agent's
        # blackboard scope is ``teams/{agent_id}/*``, so a team named after
        # an existing agent would collide with that agent's private
        # namespace. Mirrors the mesh /mesh/teams create guard.
        if name in known_agents or name in agent_registry:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Team name '{name}' conflicts with an existing agent — "
                    "teams and agents share one namespace. Pick a different name."
                ),
            )
        unknown = [m for m in members if m not in known_agents]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown agents: {', '.join(unknown)}")
        # Validate members upfront before any store writes — prevents
        # partial team creation (mirrors the old ``_create_team``).
        if "operator" in members:
            raise HTTPException(
                status_code=400,
                detail="Operator is a system agent and cannot be assigned to teams",
            )
        try:
            teams_store.create_team(name, description=description)
        except ValueError as e:  # TeamExists / invalid name
            raise HTTPException(status_code=400, detail=str(e))
        for agent in members:
            old = teams_store.add_member(name, agent)
            if old and old != name:
                _remove_team_blackboard_permissions(agent, old)
            _add_team_blackboard_permissions(agent, name)
        # Real-time cron lifecycle: schedule the daily work-summary
        # for the newly-created team. Without this, dashboard-created
        # teams went without a live cron until the next mesh restart
        # caught them in reconcile (codex r4 P2). Mirrors the mesh
        # direct-create path's metadata read pattern so a future
        # initial-settings enhancement Just Works on both surfaces.
        if cron_scheduler is not None:
            try:
                persisted = teams_store.get_team(name) or {}
                _custom_schedule = (
                    (persisted.get("settings") or {}).get("summary_schedule")
                )
                cron_scheduler.ensure_summary_job(
                    scope_kind="team", scope_id=name,
                    schedule=_custom_schedule,
                )
            except Exception as e:
                logger.warning(
                    "ensure_summary_job on dashboard team-create %s failed: %s",
                    name, e,
                )
        _emit_team_event(
            "team_created", agent="operator",
            data={
                "team_id": name,
                "name": name,
                "description": description,
                "members": list(members),
            },
        )
        return {"created": True, "name": name, "team_name": name}

    @api_router.delete("/api/teams/{team_name}")
    async def api_teams_delete(team_name: str) -> dict:
        """Delete a team and release its members."""
        from src.cli.config import _remove_team_blackboard_permissions
        from src.host.teams import TeamNotFound
        try:
            former_members = teams_store.delete_team(team_name)
        except TeamNotFound as e:
            raise HTTPException(status_code=404, detail=str(e))
        for agent in former_members:
            _remove_team_blackboard_permissions(agent, team_name)
        # Real-time cron lifecycle: drop the daily work-summary cron
        # for the deleted team. The mesh propose/confirm delete path
        # already does this; the dashboard direct-delete had to mirror
        # the cleanup or a deleted team would keep firing its summary
        # cron until the next mesh restart (codex r2 P2).
        if cron_scheduler is not None:
            try:
                existing = cron_scheduler.find_summary_job("team", team_name)
                if existing is not None:
                    cron_scheduler.remove_job(existing.id)
            except Exception as e:
                logger.warning(
                    "remove summary cron on dashboard team-delete %s failed: %s",
                    team_name, e,
                )
        _emit_team_event(
            "team_deleted", agent="operator",
            data={"team_id": team_name, "name": team_name},
        )
        return {"deleted": True, "name": team_name, "team_name": team_name}

    @api_router.post("/api/teams/{team_name}/members")
    async def api_teams_add_member(team_name: str, request: Request) -> dict:
        """Add a member agent to a team."""
        from src.cli.config import (
            _add_team_blackboard_permissions,
            _remove_team_blackboard_permissions,
        )
        from src.host.teams import TeamNotFound
        body = await request.json()
        agent = body.get("agent", "").strip()
        if not agent:
            raise HTTPException(status_code=400, detail="agent is required")
        try:
            old = teams_store.add_member(team_name, agent)
        except (TeamNotFound, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        if old and old != team_name:
            _remove_team_blackboard_permissions(agent, old)
        _add_team_blackboard_permissions(agent, team_name)
        # Auto-restart the agent so new scope takes effect
        restarted = False
        if transport is not None and agent in agent_registry:
            try:
                await transport.request(agent, "POST", "/restart", timeout=10)
                restarted = True
            except Exception as e:
                logger.warning("Failed to restart agent %s after team change: %s", agent, e)
        _emit_team_event(
            "team_updated", agent="operator",
            data={
                "team_id": team_name,
                "name": team_name,
                "field": "members",
                "added": agent,
            },
        )
        return {
            "added": True,
            "team_name": team_name,
            "agent": agent,
            "restarted": restarted,
        }

    @api_router.delete("/api/teams/{team_name}/members/{agent}")
    async def api_teams_remove_member(team_name: str, agent: str) -> dict:
        """Remove a member agent from a team."""
        from src.cli.config import _remove_team_blackboard_permissions
        if not teams_store.team_exists(team_name):
            raise HTTPException(status_code=400, detail=f"Team '{team_name}' not found")
        teams_store.remove_member(team_name, agent)
        _remove_team_blackboard_permissions(agent, team_name)
        # Auto-restart the agent so new scope takes effect
        restarted = False
        if transport is not None and agent in agent_registry:
            try:
                await transport.request(agent, "POST", "/restart", timeout=10)
                restarted = True
            except Exception as e:
                logger.warning("Failed to restart agent %s after team change: %s", agent, e)
        _emit_team_event(
            "team_updated", agent="operator",
            data={
                "team_id": team_name,
                "name": team_name,
                "field": "members",
                "removed": agent,
            },
        )
        return {
            "removed": True,
            "team_name": team_name,
            "agent": agent,
            "restarted": restarted,
        }

    # ── Team TEAM brief (team.md) ──────────────────────────

    def _resolve_team_path(team: str) -> Path:
        """Validate a team name and return the path to its team.md."""
        from src.cli.config import TEAMS_DIR
        from src.host.teams import validate_team_id
        try:
            validate_team_id(team)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid team name")
        return TEAMS_DIR / team / "team.md"

    @api_router.get("/api/team")
    async def api_team_read(team: str = "") -> dict:
        """Read a team's shared context markdown. Requires a team name."""
        if not team:
            raise HTTPException(status_code=400, detail="team parameter is required")
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not available")
        team_path = _resolve_team_path(team)
        if not team_path.parent.exists():
            raise HTTPException(status_code=404, detail=f"Team '{team}' not found")
        exists = team_path.exists()
        content = team_path.read_text(errors="replace")[:200_000] if exists else ""
        return {
            "content": content,
            "exists": exists,
            "team": team,
        }

    @api_router.put("/api/team")
    async def api_team_write(request: Request, team: str = "") -> dict:
        """Write the team's shared context markdown and push to running agents."""
        if not team:
            raise HTTPException(status_code=400, detail="team parameter is required")
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not available")
        body = await request.json()
        content = body.get("content", "")
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="content must be a string")
        content = sanitize_for_prompt(content)

        team_path = _resolve_team_path(team)
        if not team_path.parent.exists():
            raise HTTPException(status_code=404, detail=f"Team '{team}' not found")
        team_path.write_text(content)

        # Push to team members only
        members = set(teams_store.members(team))
        push_targets = [a for a in agent_registry.keys() if a in members]

        push_results = {}
        if transport is not None and push_targets:
            import asyncio as _asyncio

            async def _push(aid: str) -> tuple[str, bool]:
                try:
                    await transport.request(
                        aid, "PUT", "/team",
                        json={"content": content}, timeout=10,
                    )
                    return aid, True
                except Exception as e:
                    logger.warning("Failed to push TEAM.md to %s: %s", aid, e)
                    return aid, False

            tasks = [_push(aid) for aid in push_targets]
            for coro in _asyncio.as_completed(tasks):
                aid, ok = await coro
                push_results[aid] = ok

        _emit_team_event(
            "team_updated", agent="operator",
            data={
                "team_id": team,
                "name": team,
                "field": "team_md",
            },
        )

        return {
            "saved": True,
            "size": team_path.stat().st_size,
            "pushed": push_results,
        }

    # ── Blackboard viewer + write/delete ─────────────────────

    @api_router.get("/api/blackboard")
    async def api_blackboard(prefix: str = "") -> dict:
        entries = blackboard.list_by_prefix(prefix)
        return {
            "prefix": prefix,
            "entries": [e.model_dump(mode="json") for e in entries],
        }

    def _parse_event_preview(data: str) -> dict:
        """Parse event data JSON and extract a human-readable preview."""
        result: dict = {}
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                result["agent"] = parsed.get("source", parsed.get("agent", ""))
                for f in ("text", "summary", "status", "message",
                          "description", "name", "result"):
                    if f in parsed and isinstance(parsed[f], str):
                        result["preview"] = parsed[f][:200]
                        break
                if "preview" not in result:
                    result["preview"] = dumps_safe(parsed)[:200]
            else:
                result["preview"] = str(parsed)[:200]
        except (ValueError, TypeError):
            result["preview"] = str(data)[:200]
        return result

    @api_router.get("/api/comms/activity")
    async def api_comms_activity(limit: int = 100, team: str = "") -> dict:
        """Recent inter-agent communication: blackboard writes/deletes + pubsub events."""

        limit = max(1, min(limit, 500))
        team_prefix = f"teams/{team}/" if team else ""
        activity: list[dict] = []

        # 1. Blackboard event_log (persisted in SQLite)
        try:
            if team_prefix:
                rows = blackboard.db.execute(
                    "SELECT event_type, key, agent_id, data, timestamp "
                    "FROM event_log WHERE key LIKE ? "
                    "ORDER BY id DESC LIMIT ?",
                    (team_prefix + "%", limit),
                ).fetchall()
            else:
                rows = blackboard.db.execute(
                    "SELECT event_type, key, agent_id, data, timestamp "
                    "FROM event_log ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            for event_type, key, agent_id, data, ts in rows:
                entry: dict = {
                    "source": "blackboard",
                    "action": event_type,
                    "key": key,
                    "agent": agent_id,
                    "timestamp": ts,
                }
                if data:
                    preview_info = _parse_event_preview(data)
                    # Don't overwrite agent from the database row
                    preview_info.pop("agent", None)
                    entry.update(preview_info)
                activity.append(entry)
        except Exception:
            pass  # event_log may not exist yet

        # 2. PubSub events (persisted in SQLite when db_path is set)
        if pubsub and getattr(pubsub, "_db", None) is not None:
            try:
                if team_prefix:
                    rows = pubsub._db.execute(
                        "SELECT topic, data, created_at "
                        "FROM events WHERE topic LIKE ? "
                        "ORDER BY id DESC LIMIT ?",
                        (team_prefix + "%", limit),
                    ).fetchall()
                else:
                    rows = pubsub._db.execute(
                        "SELECT topic, data, created_at "
                        "FROM events ORDER BY id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                for topic, data, ts in rows:
                    entry = {
                        "source": "pubsub",
                        "action": "publish",
                        "topic": topic,
                        "timestamp": ts,
                    }
                    if data:
                        preview_info = _parse_event_preview(data)
                        entry.update(preview_info)
                    activity.append(entry)
            except Exception:
                pass

        # Sort merged activity by timestamp descending, then trim
        activity.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        activity = activity[:limit]

        # Also return current pubsub subscriptions for context
        subs: dict[str, list[str]] = {}
        if pubsub:
            with pubsub._lock:
                for t, agents in pubsub.subscriptions.items():
                    if team_prefix and not t.startswith(team_prefix):
                        continue
                    subs[t] = list(agents)

        return {"activity": activity, "subscriptions": subs}

    _MAX_BB_KEY_LEN = 512
    _MAX_BB_VALUE_BYTES = 262_144  # 256 KB

    @api_router.put("/api/blackboard/{key:path}")
    async def api_blackboard_write(key: str, request: Request) -> dict:

        if len(key) > _MAX_BB_KEY_LEN:
            raise HTTPException(status_code=400, detail=f"Key too long ({len(key)} chars, max {_MAX_BB_KEY_LEN})")
        body = await request.json()
        value = body.get("value", {})
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail="value must be a JSON object")
        value_size = len(dumps_safe(value))
        if value_size > _MAX_BB_VALUE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Value too large ({value_size} bytes, max {_MAX_BB_VALUE_BYTES})",
            )
        # Always attribute to "dashboard" — never trust client-supplied written_by
        # to prevent impersonation of agents via the dashboard API.
        entry = blackboard.write(key, value, written_by="dashboard")
        return entry.model_dump(mode="json")

    @api_router.delete("/api/blackboard/{key:path}")
    async def api_blackboard_delete(key: str) -> dict:
        if key.startswith("history/"):
            raise HTTPException(status_code=400, detail="Cannot delete from history namespace")
        blackboard.delete(key, deleted_by="dashboard")
        # Mirror the mesh-side DELETE emit so the SPA's blackboard view
        # refreshes live without waiting on the next poll. Best-effort.
        if event_bus is not None:
            try:
                event_bus.emit(
                    "blackboard_delete", agent="operator",
                    data={"key": key, "deleted_by": "operator"},
                )
            except Exception as e:
                logger.debug("blackboard_delete emit failed: %s", e)
        return {"deleted": True, "key": key}

    # ── Trace inspector ──────────────────────────────────────

    @api_router.get("/api/traces")
    async def api_traces(limit: int = 50) -> dict:
        if trace_store is None:
            return {"traces": []}
        limit = max(1, min(limit, 200))
        return {"traces": trace_store.list_trace_summaries(limit)}

    @api_router.get("/api/traces/{trace_id}")
    async def api_trace_detail(trace_id: str) -> dict:
        if trace_store is None:
            raise HTTPException(status_code=404, detail="Trace store not configured")
        events = trace_store.get_trace(trace_id)
        if not events:
            raise HTTPException(status_code=404, detail="Trace not found")
        return {"trace_id": trace_id, "events": events}

    @api_router.get("/api/audit")
    async def api_audit(
        agent: str | None = None,
        event_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> dict:
        """Query audit trail with filters."""
        if trace_store is None:
            return {"events": [], "total": 0}
        limit = max(1, min(limit, 1000))
        events = trace_store.query(
            agent=agent, event_type=event_type,
            since=since, until=until, limit=limit,
        )
        return {"events": events, "total": len(events)}

    @api_router.get("/api/operator-audit")
    async def api_operator_audit(request: Request) -> dict:
        """Operator audit log backed by blackboard.

        ``include_archived=true`` (any truthy value) surfaces archived
        rows alongside active ones — used by the "Show archived" toggle
        in the System > Operator change-log card.
        """
        page = int(request.query_params.get("page", "1"))
        per_page = int(request.query_params.get("per_page", "20"))
        agent_id = request.query_params.get("agent_id", "")
        action_filter = request.query_params.get("action", "")
        since = request.query_params.get("since", "")
        include_archived = request.query_params.get(
            "include_archived", "",
        ).lower() in ("1", "true", "yes", "on")
        if blackboard is None:
            return {"entries": [], "total": 0, "page": page, "per_page": per_page}
        return blackboard.get_audit_log(
            page=page, per_page=per_page, agent_id=agent_id,
            action=action_filter, since=since,
            include_archived=include_archived,
        )

    @api_router.post("/api/operator-audit/archive")
    async def api_operator_audit_archive(request: Request) -> dict:
        """Archive operator audit entries older than ``before_date``.

        Backs the "Archive entries older than ..." control on the
        operator System tab. Body: ``{"before_date": "<ISO 8601>"}``.
        Soft-archive: rows are flipped to ``archived=1`` and dropped
        from the default audit-log view. Returns
        ``{archived_count: N, truncated: bool}``.

        Proxies to the mesh's ``POST /mesh/audit/archive`` endpoint
        over loopback with the ``x-mesh-internal`` + ``X-Agent-ID:
        operator`` headers. The mesh route enforces the
        operator-or-internal permission tier and writes the
        audit-of-audit row recording the archive — calling
        ``blackboard.archive_audit_before`` directly here would
        bypass that gate and let any session with an ``ol_session``
        cookie clear audit history, regardless of operator identity.
        The dashboard's CSRF guard (``X-Requested-With``) is still
        enforced on this route by the global middleware.
        """
        if blackboard is None:
            raise HTTPException(503, "Blackboard not available")
        try:
            body = await request.json()
        except Exception:
            body = {}
        before_date = (body or {}).get("before_date") if isinstance(body, dict) else None
        if not before_date or not isinstance(before_date, str):
            raise HTTPException(400, "before_date is required (ISO 8601 string)")
        import httpx
        url = f"http://127.0.0.1:{mesh_port}/mesh/audit/archive"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    url,
                    json={"before_date": before_date},
                    headers={
                        "X-Requested-With": "XMLHttpRequest",
                        "x-mesh-internal": "1",
                        "X-Agent-ID": "operator",
                        "Content-Type": "application/json",
                    },
                )
        except Exception as e:
            logger.warning("operator-audit archive proxy failed: %s", e)
            raise HTTPException(502, f"Mesh unreachable: {e}")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise HTTPException(resp.status_code, detail)
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "archived_count": 0, "before_date": before_date}

    # ── Operator: internet access toggle ─────────────────────

    @api_router.get("/api/operator/internet-access")
    async def api_operator_internet_access_status() -> dict:
        """Return the operator's current internet-access state.

        Proxies to mesh ``GET /mesh/operator/internet-access``. The
        Operator Settings UI calls this on mount to render the toggle's
        initial position.
        """
        import httpx
        url = f"http://127.0.0.1:{mesh_port}/mesh/operator/internet-access"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    url,
                    headers={
                        "x-mesh-internal": "1",
                        "X-Agent-ID": "operator",
                    },
                )
        except Exception as e:
            logger.warning("operator internet-access status proxy failed: %s", e)
            raise HTTPException(502, f"Mesh unreachable: {e}")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise HTTPException(resp.status_code, detail)
        return resp.json()

    @api_router.post("/api/operator/internet-access")
    async def api_operator_internet_access_set(request: Request) -> dict:
        """Flip the operator's internet access on/off.

        Body: ``{"enabled": bool}``. Proxies to mesh
        ``POST /mesh/operator/internet-access`` which writes the
        permission, reloads the matrix, pushes to the operator container,
        and emits ``agent_config_updated``. Response shape:
        ``{success, enabled, previous, live}``.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict) or "enabled" not in body:
            raise HTTPException(400, "'enabled' is required")
        enabled = body.get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(400, "'enabled' must be a boolean")
        import httpx
        url = f"http://127.0.0.1:{mesh_port}/mesh/operator/internet-access"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    url,
                    json={"enabled": enabled},
                    headers={
                        "X-Requested-With": "XMLHttpRequest",
                        "x-mesh-internal": "1",
                        "X-Agent-ID": "operator",
                        "Content-Type": "application/json",
                    },
                )
        except Exception as e:
            logger.warning("operator internet-access set proxy failed: %s", e)
            raise HTTPException(502, f"Mesh unreachable: {e}")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise HTTPException(resp.status_code, detail)
        return resp.json()

    # ── Operator: browser access toggle ───────────────────────

    @api_router.get("/api/operator/browser-access")
    async def api_operator_browser_access_status() -> dict:
        """Return the operator's current browser-access state.

        Proxies to mesh ``GET /mesh/operator/browser-access``. The
        Operator Settings UI calls this on mount to render the toggle.
        """
        import httpx
        url = f"http://127.0.0.1:{mesh_port}/mesh/operator/browser-access"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    url,
                    headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
                )
        except Exception as e:
            logger.warning("operator browser-access status proxy failed: %s", e)
            raise HTTPException(502, f"Mesh unreachable: {e}")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise HTTPException(resp.status_code, detail)
        return resp.json()

    @api_router.post("/api/operator/browser-access")
    async def api_operator_browser_access_set(request: Request) -> dict:
        """Flip the operator's browser access on/off.

        Body: ``{"enabled": bool}``. Proxies to mesh
        ``POST /mesh/operator/browser-access``. Response shape:
        ``{success, enabled, previous, live}``.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict) or "enabled" not in body:
            raise HTTPException(400, "'enabled' is required")
        enabled = body.get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(400, "'enabled' must be a boolean")
        import httpx
        url = f"http://127.0.0.1:{mesh_port}/mesh/operator/browser-access"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    url,
                    json={"enabled": enabled},
                    headers={
                        "X-Requested-With": "XMLHttpRequest",
                        "x-mesh-internal": "1",
                        "X-Agent-ID": "operator",
                        "Content-Type": "application/json",
                    },
                )
        except Exception as e:
            logger.warning("operator browser-access set proxy failed: %s", e)
            raise HTTPException(502, f"Mesh unreachable: {e}")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise HTTPException(resp.status_code, detail)
        return resp.json()

    # ── Queue status ─────────────────────────────────────────

    @api_router.get("/api/queues")
    async def api_queues() -> dict:
        lane_status = lane_manager.get_status() if lane_manager else {}
        # Merge with agent registry so all agents appear (even idle ones)
        queues = {}
        for agent_id in agent_registry:
            queues[agent_id] = lane_status.get(agent_id, {
                "queued": 0, "pending": 0, "busy": False,
            })
        # Include any lanes for agents not in registry (shouldn't happen, but safe)
        for agent_id, status in lane_status.items():
            if agent_id not in queues:
                queues[agent_id] = status
        return {"queues": queues}

    # ── Model health ──────────────────────────────────────

    @api_router.get("/api/model-health")
    async def api_model_health() -> dict:
        if credential_vault is None:
            return {"models": []}
        return {"models": credential_vault.get_model_health()}

    # ── Cron management ──────────────────────────────────

    @api_router.get("/api/cron")
    async def api_cron() -> dict:
        if cron_scheduler is None:
            return {"jobs": []}
        return {"jobs": cron_scheduler.list_jobs()}

    @api_router.post("/api/cron")
    async def api_cron_create(request: Request) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        body = await request.json()
        agent = body.get("agent", "").strip()
        schedule = body.get("schedule", "").strip()
        message = body.get("message", "").strip()
        if not agent or not schedule or not message:
            raise HTTPException(status_code=400, detail="agent, schedule, and message are required")
        if agent not in agent_registry:
            raise HTTPException(status_code=400, detail=f"Agent '{agent}' not found")
        try:
            job = cron_scheduler.add_job(agent=agent, schedule=schedule, message=message)
            return {"created": True, "job_id": job.id}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @api_router.post("/api/cron/{job_id}/run")
    async def api_cron_run(job_id: str) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        if job_id not in cron_scheduler.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        # Fire-and-forget: dispatch in background so the HTTP response returns
        # immediately.  Agent execution can take minutes; blocking the request
        # made the dashboard Run button appear stuck.
        import asyncio
        task = asyncio.create_task(cron_scheduler.run_job(job_id))
        task.add_done_callback(_log_cron_task_exception)
        return {"triggered": True, "job_id": job_id}

    @api_router.put("/api/cron/{job_id}")
    async def api_cron_update(job_id: str, request: Request) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        body = await request.json()
        if "schedule" in body:
            error = cron_scheduler._validate_schedule(body["schedule"])
            if error:
                raise HTTPException(status_code=400, detail=error)
        job = await cron_scheduler.update_job(job_id, **body)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"status": "updated", "job_id": job_id}

    @api_router.post("/api/cron/{job_id}/pause")
    async def api_cron_pause(job_id: str) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        if not await cron_scheduler.pause_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return {"paused": True, "job_id": job_id}

    @api_router.post("/api/cron/{job_id}/resume")
    async def api_cron_resume(job_id: str) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        if not await cron_scheduler.resume_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return {"resumed": True, "job_id": job_id}

    @api_router.delete("/api/cron/{job_id}")
    async def api_cron_delete(job_id: str) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        job = cron_scheduler.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.heartbeat:
            raise HTTPException(status_code=403, detail="Heartbeat jobs cannot be deleted")
        if not cron_scheduler.remove_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return {"deleted": True, "job_id": job_id}

    # ── Settings / environment ───────────────────────────────

    @api_router.get("/api/settings")
    async def api_settings() -> dict:
        from src.host.credentials import SYSTEM_CREDENTIAL_PROVIDERS
        from src.shared.models import get_all_model_costs

        app_url = os.environ.get("OPENLEGION_APP_URL", "")
        cred_names = credential_vault.list_credential_names() if credential_vault else []
        agent_cred_names = credential_vault.list_agent_credential_names() if credential_vault else []
        _llm_key_names = {f"{p}_api_key" for p in SYSTEM_CREDENTIAL_PROVIDERS}
        has_llm = bool(set(cred_names) & _llm_key_names)
        if not has_llm and "openlegion_api_key" in cred_names:
            has_llm = True
        if not has_llm and credential_vault:
            if credential_vault._has_anthropic_oauth() or credential_vault._has_openai_oauth():
                has_llm = True

        # Credit-awareness: distinguish BYOK keys from credit proxy
        credit_proxy_configured = "openlegion_api_key" in cred_names
        _curated_llm_names = set(cred_names) & _llm_key_names  # curated provider keys present
        has_oauth = (
            (credential_vault._has_anthropic_oauth() or credential_vault._has_openai_oauth())
            if credential_vault else False
        )
        has_byok_keys = bool(_curated_llm_names) or has_oauth

        pubsub_subs = pubsub.subscriptions if pubsub else {}

        # Filtered models: only providers with credentials
        available_provider_models: dict[str, list[str]] = {}
        if credential_vault:
            active_providers = credential_vault.get_providers_with_credentials()
            available_provider_models = {
                p: models for p, models in _PROVIDER_MODELS.items()
                if p in active_providers
            }

            # Merge custom LLM providers from settings
            settings = _load_settings()
            for prov, info in settings.get("custom_llm_providers", {}).items():
                if prov in active_providers:
                    available_provider_models[prov] = info.get("models", [])
                    has_llm = True

            # Discover locally-installed Ollama models and merge them in.
            # Only adds Ollama to the dropdown if it's actually reachable.
            try:
                discovered = await credential_vault.discover_ollama_models()
                if discovered:
                    featured = available_provider_models.get("ollama", [])
                    discovered_set = set(discovered)
                    merged = discovered + [
                        m for m in featured if m not in discovered_set
                    ]
                    available_provider_models["ollama"] = merged
                    has_llm = True
            except Exception:
                pass  # Keep whatever's already there (if any)

            # Discover models available through the OpenLegion credit proxy.
            # Similar to Ollama discovery — merges gateway catalog into dropdown.
            try:
                ol_models, ol_pricing = await credential_vault.discover_openlegion_models()
                if ol_models:
                    from src.shared.models import set_gateway_pricing
                    set_gateway_pricing(ol_pricing)
                    featured = available_provider_models.get("openlegion", [])
                    featured_set = set(featured)
                    merged = featured + [m for m in ol_models if m not in featured_set]
                    available_provider_models["openlegion"] = merged
                    has_llm = True
            except Exception:
                pass

        all_costs = get_all_model_costs()

        # Include gateway pricing for openlegion models.  Gateway pricing
        # overwrites litellm/fallback costs so the dashboard displays the
        # same prices that the credit proxy actually charges.
        if "openlegion" in available_provider_models:
            from src.shared.models import get_gateway_pricing
            for gw_model, cost in get_gateway_pricing().items():
                all_costs[f"openlegion/{gw_model}"] = cost

        return {
            "credentials": {"names": cred_names, "count": len(cred_names)},
            "agent_credentials": agent_cred_names,
            "has_llm_credentials": has_llm,
            "pubsub_subscriptions": pubsub_subs,
            "model_costs": {k: {"input_per_1k": v[0], "output_per_1k": v[1]} for k, v in all_costs.items()},
            "provider_models": dict(_PROVIDER_MODELS.items()),
            "available_provider_models": available_provider_models,
            "credit_proxy_configured": credit_proxy_configured,
            "has_byok_keys": has_byok_keys,
            "app_url": app_url,
            "plan_limits": {
                "max_agents": _max_agents,
                "max_teams": _max_teams,
                "teams_enabled": not _teams_disabled,
            },
        }

    # ── Browser settings ─────────────────────────────────────────

    _SETTINGS_PATH = Path("config/settings.json")
    _settings_lock = threading.Lock()

    def _load_settings() -> dict:
        """Load persisted settings from config/settings.json."""
        if _SETTINGS_PATH.exists():
            try:
                return json.loads(_SETTINGS_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_settings(settings: dict) -> None:
        """Persist settings to config/settings.json (atomic write)."""
        import tempfile
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(settings, indent=2) + "\n"
        fd, tmp_path = tempfile.mkstemp(
            dir=str(_SETTINGS_PATH.parent), suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            Path(tmp_path).unlink(missing_ok=True)
            raise
        try:
            Path(tmp_path).replace(_SETTINGS_PATH)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    @api_router.get("/api/browser-settings")
    async def api_get_browser_settings() -> dict:
        """Return saved browser speed and delay settings."""
        settings = _load_settings()
        return {
            "speed": settings.get("browser_speed", 1.0),
            "delay": settings.get("browser_delay", 0.3),
        }

    @api_router.get("/api/dashboard/platform-success")
    async def api_get_platform_success() -> dict:
        """Per-platform fleet success rollup over the last 24h.

        Read-only aggregation surfaced from the dashboard's in-memory
        EventBus listener — no SQLite, no upstream call.  Operators use
        this to spot "which sites is my fleet succeeding on, and which
        ones are flagging or burning fingerprints" without scraping
        logs.  Resets on mesh restart (process-local).
        """
        return platform_success.snapshot()

    @api_router.post("/api/browser-settings")
    async def api_set_browser_settings(request: Request) -> dict:
        """Save browser speed/delay settings and push to browser service."""
        body = await request.json()
        speed = body.get("speed")
        delay = body.get("delay")

        if speed is None and delay is None:
            raise HTTPException(400, "speed or delay is required")

        payload: dict = {}

        if speed is not None:
            try:
                speed = float(speed)
            except (ValueError, TypeError):
                raise HTTPException(400, "speed must be a number")
            if speed < 0.25 or speed > 4.0:
                raise HTTPException(400, "speed must be between 0.25 and 4.0")
            payload["speed"] = speed

        if delay is not None:
            try:
                delay = float(delay)
            except (ValueError, TypeError):
                raise HTTPException(400, "delay must be a number")
            if delay < 0.0 or delay > 10.0:
                raise HTTPException(400, "delay must be between 0.0 and 10.0")
            payload["delay"] = delay

        # Persist to config file
        with _settings_lock:
            settings = _load_settings()
            if "speed" in payload:
                settings["browser_speed"] = payload["speed"]
            if "delay" in payload:
                settings["browser_delay"] = payload["delay"]
            _save_settings(settings)

        # Push to browser service immediately
        if runtime and hasattr(runtime, 'browser_service_url') and runtime.browser_service_url:
            try:
                browser_auth = getattr(runtime, 'browser_auth_token', '')
                headers = {}
                if browser_auth:
                    headers["Authorization"] = f"Bearer {browser_auth}"
                await _dashboard_browser_client.post(
                    f"{runtime.browser_service_url}/browser/settings",
                    json=payload,
                    headers=headers,
                )
            except Exception as e:
                logger.debug("Failed to push browser settings: %s", e)

        settings = _load_settings()
        _emit_config_changed("browser_settings")
        return {
            "speed": settings.get("browser_speed", 1.0),
            "delay": settings.get("browser_delay", 0.3),
        }

    # ── CAPTCHA solver settings ───────────────────────────────

    @api_router.get("/api/captcha-solver")
    async def api_get_captcha_solver() -> dict:
        """Return CAPTCHA solver configuration (key is masked)."""
        settings = _load_settings()
        provider = settings.get("captcha_solver_provider", "")
        key = settings.get("captcha_solver_key", "")
        return {
            "provider": provider,
            "key_masked": f"...{key[-4:]}" if len(key) >= 4 else "",
        }

    @api_router.post("/api/captcha-solver")
    async def api_set_captcha_solver(request: Request) -> dict:
        """Save CAPTCHA solver provider and API key."""
        await _csrf_check(request)
        body = await request.json()
        provider = body.get("provider", "").strip().lower()
        key = body.get("key", "").strip()

        if provider and provider not in ("2captcha", "capsolver"):
            raise HTTPException(400, "provider must be '2captcha' or 'capsolver'")

        with _settings_lock:
            settings = _load_settings()
            settings["captcha_solver_provider"] = provider
            if key:
                settings["captcha_solver_key"] = key
            elif not provider:
                settings.pop("captcha_solver_key", None)
            _save_settings(settings)

        settings = _load_settings()
        stored_key = settings.get("captcha_solver_key", "")
        _emit_config_changed("captcha_solver")
        return {
            "provider": settings.get("captcha_solver_provider", ""),
            "key_masked": f"...{stored_key[-4:]}" if len(stored_key) >= 4 else "",
        }

    @api_router.delete("/api/captcha-solver")
    async def api_delete_captcha_solver(request: Request) -> dict:
        """Remove CAPTCHA solver configuration."""
        await _csrf_check(request)
        with _settings_lock:
            settings = _load_settings()
            settings.pop("captcha_solver_provider", None)
            settings.pop("captcha_solver_key", None)
            _save_settings(settings)
        _emit_config_changed("captcha_solver")
        return {"removed": True}

    # ── System settings (consolidated) ────────────────────────

    # Execution-limit keys whose ranges/defaults are owned by
    # src/shared/limits.py (the single source of truth). The list of which
    # limits the dashboard surfaces also lives there (DASHBOARD_GLOBAL_KEYS),
    # so the UI can never silently diverge from / drop a LIMIT_SPECS entry.
    _LIMIT_KEYS = _limits.DASHBOARD_GLOBAL_KEYS

    _SYSTEM_SETTINGS_VALIDATORS: dict[str, tuple[type, float, float]] = {
        "default_daily_budget": (float, 0.01, 10000),
        "default_monthly_budget": (float, 0.01, 100000),
        "tool_timeout": (int, 10, 3600),
        "browser_idle_timeout": (int, 5, 120),
        "health_poll_interval": (int, 5, 300),
        "health_max_failures": (int, 1, 20),
        "health_restart_limit": (int, 0, 20),
        "health_restart_window": (int, 60, 86400),
        # Execution limits sourced from limits.py (default, lo, hi).
        **{
            k: (int, _limits.LIMIT_SPECS[k][1], _limits.LIMIT_SPECS[k][2])
            for k in _LIMIT_KEYS
        },
    }

    # Settings seeded into agent containers as env vars at launch — they
    # only take effect after the agent restarts (the restart-agents path
    # re-reads settings.json). The dashboard auto-restarts the fleet when
    # one of these changes so the user never has stale config silently
    # in effect. Budgets are per-new-agent defaults and health_* apply
    # live above, so neither needs a restart.
    _RESTART_REQUIRED_SETTINGS: frozenset[str] = frozenset({
        "max_iterations", "chat_max_tool_rounds",
        "chat_max_total_rounds", "tool_timeout",
        "task_max_tool_rounds", "llm_timeout_seconds",
        "lane_timeout_seconds",
    })

    from src.host.costs import DEFAULT_DAILY_BUDGET_USD, DEFAULT_MONTHLY_BUDGET_USD

    _SYSTEM_SETTINGS_DEFAULTS: dict[str, float | int] = {
        # Single-sourced from costs.py (plan B-pre #3): the advertised
        # default and the enforced no-settings-file fallback must never
        # diverge (they were $50 vs $10 before).
        "default_daily_budget": DEFAULT_DAILY_BUDGET_USD,
        "default_monthly_budget": DEFAULT_MONTHLY_BUDGET_USD,
        "tool_timeout": 900,
        "browser_idle_timeout": 30,
        "health_poll_interval": 30,
        "health_max_failures": 3,
        "health_restart_limit": 3,
        "health_restart_window": 3600,
        # Execution-limit defaults sourced from limits.py.
        **{k: _limits.LIMIT_SPECS[k][0] for k in _LIMIT_KEYS},
    }

    @api_router.get("/api/system-settings")
    async def api_get_system_settings() -> dict:
        """Return all system settings with defaults."""
        from src.cli.config import _load_config
        settings = _load_settings()
        result = {}
        for key, default in _SYSTEM_SETTINGS_DEFAULTS.items():
            result[key] = settings.get(key, default)
        # Include default_model from mesh.yaml
        cfg = _load_config()
        result["default_model"] = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")

        # Embedding / semantic-memory selection + resolved status.
        from src.cli.runtime import (
            _EMBEDDING_PROVIDER_LADDER,
            _embedding_providers_with_keys,
            _resolve_embedding,
        )

        raw_embed = cfg.get("llm", {}).get("embedding_model")
        keyed = _embedding_providers_with_keys()
        eff_model, _eff_dim = _resolve_embedding(raw_embed, keyed)
        if raw_embed is None:
            configured_provider = "auto"
        elif str(raw_embed).lower() == "none":
            configured_provider = "none"
        else:
            configured_provider = next(
                (p for p, m, _d in _EMBEDDING_PROVIDER_LADDER if m == raw_embed),
                "custom",
            )
        # An explicit configured model bypasses the resolver's key check, so
        # verify the provider actually has a key — otherwise the status would
        # claim ON after the key was removed. A "custom" model outside the
        # ladder can't be verified here, so it's trusted.
        on = str(eff_model).lower() != "none"
        if (
            on and raw_embed is not None
            and configured_provider != "custom"
            and configured_provider not in keyed
        ):
            on = False
        result["embedding"] = {
            "configured": raw_embed,
            "configured_provider": configured_provider,
            "effective_model": eff_model,
            "on": on,
            "available_providers": [
                p for p, _m, _d in _EMBEDDING_PROVIDER_LADDER if p in keyed
            ],
        }
        return result

    @api_router.post("/api/system-settings")
    async def api_set_system_settings(request: Request) -> dict:
        """Update system settings. Accepts a partial dict of settings."""
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "Request body must be a JSON object")

        updated = []

        with _settings_lock:
            settings = _load_settings()

            for key, value in body.items():
                if key not in _SYSTEM_SETTINGS_VALIDATORS:
                    continue
                typ, min_val, max_val = _SYSTEM_SETTINGS_VALIDATORS[key]
                try:
                    coerced = typ(value)
                except (ValueError, TypeError):
                    raise HTTPException(400, f"{key} must be a {typ.__name__}")
                if coerced < min_val or coerced > max_val:
                    raise HTTPException(400, f"{key} must be between {min_val} and {max_val}")
                settings[key] = coerced
                updated.append(key)

            if updated:
                _save_settings(settings)

        # Apply health settings at runtime
        if health_monitor:
            _health_keys = {
                "health_poll_interval": "POLL_INTERVAL",
                "health_max_failures": "MAX_FAILURES",
                "health_restart_limit": "RESTART_LIMIT",
                "health_restart_window": "RESTART_WINDOW",
            }
            for cfg_key, attr in _health_keys.items():
                if cfg_key in updated:
                    setattr(health_monitor, attr, settings[cfg_key])

        restart_required = bool(set(updated) & _RESTART_REQUIRED_SETTINGS)
        if updated:
            _emit_config_changed("system_settings")
        return {"updated": updated, "restart_required": restart_required}

    @api_router.post("/api/default-model")
    async def api_set_default_model(request: Request) -> dict:
        """Update the default LLM model in mesh.yaml."""
        import yaml
        body = await request.json()
        model = body.get("model", "").strip()
        if not model:
            raise HTTPException(400, "model is required")
        if not _is_valid_model(model):
            raise HTTPException(400, f"Unknown model: {model}")

        config_path = Path("config/mesh.yaml")
        mesh_cfg: dict = {}
        if config_path.exists():
            with open(config_path) as f:
                mesh_cfg = yaml.safe_load(f) or {}
        mesh_cfg.setdefault("llm", {})["default_model"] = model
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(mesh_cfg, f, default_flow_style=False, sort_keys=False)

        _emit_config_changed("system_settings")
        return {"model": model}

    @api_router.post("/api/embedding-model")
    async def api_set_embedding_model(request: Request) -> dict:
        """Update the embedding model selection in mesh.yaml.

        Body: ``{"value": "<provider>|none|auto"}``.
          - ``"auto"``  → remove ``llm.embedding_model`` so the resolver
            auto-picks the best available embedding-capable key.
          - ``"none"``  → store ``"none"`` (keyword-only memory).
          - a provider in the embedding ladder → store that provider's model.
        """
        import yaml

        from src.cli.runtime import (
            _EMBEDDING_PROVIDER_LADDER,
            _embedding_providers_with_keys,
            _resolve_embedding,
        )

        body = await request.json()
        value = str(body.get("value", "")).strip()
        if not value:
            raise HTTPException(400, "value is required")

        stored: str | None
        if value == "auto":
            stored = None
        elif value.lower() == "none":
            stored = "none"
        else:
            model = next(
                (m for p, m, _d in _EMBEDDING_PROVIDER_LADDER if p == value), None
            )
            if model is None:
                raise HTTPException(400, f"Unknown embedding provider: {value}")
            # Reject a provider with no configured key — persisting it would
            # mint a dead embedding model that agents restart into (the embed
            # proxy authenticates by key only). Validate at config-write time,
            # matching the model-allowlist convention used elsewhere.
            if value not in _embedding_providers_with_keys():
                raise HTTPException(
                    400, f"No API key configured for embedding provider: {value}",
                )
            stored = model

        config_path = Path("config/mesh.yaml")
        mesh_cfg: dict = {}
        if config_path.exists():
            with open(config_path) as f:
                mesh_cfg = yaml.safe_load(f) or {}
        llm_cfg = mesh_cfg.setdefault("llm", {})
        if stored is None:
            llm_cfg.pop("embedding_model", None)
        else:
            llm_cfg["embedding_model"] = stored
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(mesh_cfg, f, default_flow_style=False, sort_keys=False)

        # Agent containers read embedding config from runtime.extra_env at
        # container start (cli/runtime.py populates EMBEDDING_MODEL/DIM only
        # at engine boot). Without refreshing it here, "save + restart
        # agents" restarts containers with the OLD embedding env while the
        # UI reports the new model active.
        if runtime is not None:
            eff_model, eff_dim = _resolve_embedding(
                stored, _embedding_providers_with_keys(),
            )
            runtime.extra_env["EMBEDDING_MODEL"] = eff_model
            runtime.extra_env["EMBEDDING_DIM"] = str(eff_dim)

        _emit_config_changed("system_settings")
        return {"value": value, "embedding_model": stored}

    # ── Restart agents ────────────────────────────────────────

    @api_router.post("/api/restart-agents")
    async def api_restart_agents() -> dict:
        """Restart all agent containers and the browser service.

        Re-reads config/settings.json and mesh.yaml so env-var-based
        settings (execution limits, browser idle timeout) take effect.
        """
        import asyncio as _asyncio

        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not available")
        from src.cli.config import _load_config
        from src.host.runtime import DockerBackend

        # Refresh env vars from settings
        settings_path = Path("config/settings.json")
        if settings_path.exists():
            try:
                sys_settings = json.loads(settings_path.read_text())
                for env_key, cfg_key in {
                    "OPENLEGION_MAX_ITERATIONS": "max_iterations",
                    "OPENLEGION_CHAT_MAX_TOOL_ROUNDS": "chat_max_tool_rounds",
                    "OPENLEGION_CHAT_MAX_TOTAL_ROUNDS": "chat_max_total_rounds",
                    "OPENLEGION_TOOL_TIMEOUT": "tool_timeout",
                    "OPENLEGION_TASK_MAX_TOOL_ROUNDS": "task_max_tool_rounds",
                    "OPENLEGION_LLM_TIMEOUT_SECONDS": "llm_timeout_seconds",
                    "OPENLEGION_LANE_TIMEOUT_SECONDS": "lane_timeout_seconds",
                }.items():
                    if cfg_key in sys_settings:
                        runtime.extra_env[env_key] = str(sys_settings[cfg_key])
                # Push CAPTCHA solver config to env for browser service
                _solver_provider = sys_settings.get("captcha_solver_provider", "")
                _solver_key = sys_settings.get("captcha_solver_key", "")
                if _solver_provider:
                    os.environ["OPENLEGION_CAPTCHA_SOLVER_PROVIDER"] = _solver_provider
                if _solver_key:
                    os.environ["OPENLEGION_CAPTCHA_SOLVER_KEY"] = _solver_key
            except (ValueError, OSError):
                pass

        cfg = _load_config()
        agents_cfg = cfg.get("agents", {})
        default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        loop = _asyncio.get_running_loop()
        results = {}

        # Restart browser service first (picks up idle timeout from settings)
        if isinstance(runtime, DockerBackend) and hasattr(runtime, "stop_browser_service"):
            try:
                await loop.run_in_executor(None, runtime.stop_browser_service)
                await loop.run_in_executor(None, runtime.start_browser_service)
                # Push saved speed/delay to the freshly started browser service
                await _push_browser_settings()
            except Exception as e:
                logger.warning("Browser service restart failed: %s", e)

        # Restart all agents in parallel
        _network_cfg = cfg.get("network", {})
        from src.cli.config import (
            _OPERATOR_AGENT_ID,
            _OPERATOR_ALLOWED_TOOLS,
            _load_permissions,
        )

        async def _restart_one(agent_id: str) -> tuple[str, str]:
            agent_cfg = agents_cfg.get(agent_id, {})
            try:
                await loop.run_in_executor(None, runtime.stop_agent, agent_id)
                tools_dir = agent_cfg.get("tools_dir", "")
                if tools_dir:
                    tools_dir = str(Path(tools_dir).resolve())
                # Per-agent env overrides (proxy + operator tools).
                # Proxy goes in env_overrides instead of runtime.extra_env
                # so parallel restarts don't stomp each other's proxy vars.
                _restart_env: dict[str, str] = {}
                if agent_id == _OPERATOR_AGENT_ID:
                    _restart_env["ALLOWED_TOOLS"] = ",".join(_OPERATOR_ALLOWED_TOOLS)
                    # Re-seed internet/browser access flags so a fleet
                    # restart (incl. the dashboard's auto-restart on
                    # restart-gated setting changes) doesn't silently
                    # re-enable a capability the operator toggled OFF.
                    # Mirrors the single-agent restart path; default True
                    # matches the operator-by-default UX.
                    try:
                        _op_perms = _load_permissions().get(
                            "permissions", {},
                        ).get(_OPERATOR_AGENT_ID, {})
                        _restart_env["OL_INTERNET_ACCESS_ENABLED"] = (
                            "true" if _op_perms.get("can_use_internet", True) else "false"
                        )
                        _restart_env["OL_BROWSER_ACCESS_ENABLED"] = (
                            "true" if _op_perms.get("can_use_browser", True) else "false"
                        )
                    except Exception:
                        _restart_env["OL_INTERNET_ACCESS_ENABLED"] = "true"
                        _restart_env["OL_BROWSER_ACCESS_ENABLED"] = "true"
                _proxy_url = resolve_agent_proxy(agent_id, agents_cfg, _network_cfg)
                _proxy_env = build_proxy_env_vars(
                    _proxy_url, _network_cfg.get("no_proxy", ""),
                )
                _restart_env.update(_proxy_env)
                # Per-agent output-token cap → LLM_MAX_TOKENS (survives the
                # dashboard "restart all agents" flow, not just hot-reload).
                set_llm_max_tokens_env(_restart_env, agent_cfg)
                from src.shared.limits import set_llm_limits_env
                set_llm_limits_env(_restart_env, agent_cfg)
                url = await loop.run_in_executor(
                    None,
                    lambda aid=agent_id, acfg=agent_cfg, sd=tools_dir, re=_restart_env: runtime.start_agent(
                        agent_id=aid,
                        role=acfg.get("role", "assistant"),
                        tools_dir=sd,
                        model=acfg.get("model", default_model),
                        thinking=acfg.get("thinking", ""),
                        env_overrides=re,
                    ),
                )
                if router is not None:
                    router.register_agent(agent_id, url, role=agent_cfg.get("role", ""))
                else:
                    agent_registry[agent_id] = url
                if transport is not None:
                    from src.host.transport import HttpTransport
                    if isinstance(transport, HttpTransport):
                        transport.register(agent_id, url)
                ready = await runtime.wait_for_agent(agent_id, timeout=60)
                await _push_browser_proxy_for_agent(agent_id)
                return (agent_id, "ready" if ready else "started")
            except Exception as e:
                logger.error("Failed to restart agent '%s': %s", agent_id, e)
                return (agent_id, f"error: {e}")

        agent_results = await _asyncio.gather(
            *[_restart_one(aid) for aid in list(agent_registry.keys())]
        )
        results = dict(agent_results)

        return {"restarted": results}

    # ── Storage ────────────────────────────────────────────────

    _STORAGE_SKIP_DIRS = {"src", ".git", ".venv", "venv", "node_modules", ".claude"}
    _STORAGE_DB_SUFFIXES = {".db", ".db-wal", ".db-shm"}

    def _scan_storage(root: Path) -> dict:
        """Scan the repo root for storage breakdown (blocking I/O).

        Uses os.walk with dir pruning to avoid descending into source,
        git, and virtualenv directories.
        """
        # System-wide disk usage
        try:
            disk = shutil.disk_usage(str(root))
            disk_info = {"total": disk.total, "used": disk.used, "free": disk.free}
        except OSError:
            disk_info = {"total": 0, "used": 0, "free": 0}

        db_bytes = 0
        log_bytes = 0
        config_bytes = 0
        agent_bytes = 0
        other_bytes = 0

        root_str = str(root)
        for dirpath, dirnames, filenames in os.walk(root):
            # Compute top-level directory relative to root
            rel = os.path.relpath(dirpath, root_str)
            top = rel.split(os.sep, 1)[0] if rel != "." else ""

            # Prune skipped directories so os.walk doesn't descend
            dirnames[:] = [
                d for d in dirnames
                if (d if rel == "." else top) not in _STORAGE_SKIP_DIRS
            ]

            for name in filenames:
                fpath = os.path.join(dirpath, name)
                try:
                    size = os.lstat(fpath).st_size
                except OSError:
                    continue

                # Categorize: files at root level use their own name/suffix,
                # files in subdirs are categorized by top-level dir
                _, suffix = os.path.splitext(name)
                suffix = suffix.lower()
                if suffix in _STORAGE_DB_SUFFIXES:
                    db_bytes += size
                elif suffix == ".log":
                    log_bytes += size
                elif top == "config":
                    config_bytes += size
                elif top == ".openlegion":
                    agent_bytes += size
                elif top == "data":
                    # data/ contains costs.db, traces.db (caught above by suffix);
                    # any other files in data/ are still engine data
                    other_bytes += size
                elif top:
                    other_bytes += size
                else:
                    # Root-level files that aren't db/log
                    other_bytes += size

        engine_total = db_bytes + log_bytes + config_bytes + agent_bytes + other_bytes
        return {
            "disk": disk_info,
            "engine": {
                "total": engine_total,
                "databases": db_bytes,
                "agent_data": agent_bytes,
                "logs": log_bytes,
                "config": config_bytes,
                "other": other_bytes,
            },
        }

    @api_router.get("/api/storage")
    async def api_storage() -> dict:
        """Return disk usage breakdown for the engine's data directory."""
        import asyncio

        project_root = (
            runtime.project_root if runtime and hasattr(runtime, "project_root")
            else Path(__file__).resolve().parent.parent.parent
        )
        return await asyncio.get_running_loop().run_in_executor(
            None, _scan_storage, project_root,
        )

    # ── Database details ─────────────────────────────────────

    _DB_REGISTRY = [
        {
            "id": "blackboard",
            "label": "Blackboard",
            "description": "Shared agent coordination state",
            "path": "blackboard.db",
            "tables": {
                "entries": {"ts_col": "created_at", "ts_type": "text"},
                "event_log": {"ts_col": "timestamp", "ts_type": "text"},
            },
            "purgeable": True,
        },
        {
            "id": "traces",
            "label": "Traces",
            "description": "Request execution traces and events",
            "path": "data/traces.db",
            "tables": {
                "traces": {"ts_col": "timestamp", "ts_type": "real"},
            },
            "purgeable": True,
        },
        {
            "id": "costs",
            "label": "Cost History",
            "description": "LLM token usage and spend tracking",
            "path": "data/costs.db",
            "tables": {
                "usage": {"ts_col": "timestamp", "ts_type": "text"},
            },
            "purgeable": True,
        },
        {
            "id": "wallet",
            "label": "Wallet",
            "description": "Transaction history and key indexes",
            "path": "data/wallet.db",
            "tables": {
                "transactions": {"ts_col": "timestamp", "ts_type": "text"},
                "agent_index": {"ts_col": "created_at", "ts_type": "text"},
            },
            "purgeable": False,
        },
    ]

    def _scan_database_details(root: Path) -> list[dict]:
        """Scan engine databases for record counts and metadata (blocking I/O)."""
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        results = []
        for entry in _DB_REGISTRY:
            db_path = root / entry["path"]
            info: dict = {
                "id": entry["id"],
                "label": entry["label"],
                "description": entry["description"],
                "purgeable": entry["purgeable"],
                "size_bytes": 0,
                "tables": [],
                "total_records": 0,
                "oldest": None,
            }

            # Sum file sizes (.db + .db-wal + .db-shm)
            for suffix in ("", "-wal", "-shm"):
                p = db_path.parent / (db_path.name + suffix)
                try:
                    info["size_bytes"] += p.stat().st_size
                except OSError:
                    pass

            if not db_path.exists():
                results.append(info)
                continue

            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                try:
                    conn.execute("PRAGMA busy_timeout=2000")
                    oldest_ts = None
                    for table_name, meta in entry["tables"].items():
                        try:
                            row = conn.execute(
                                f"SELECT COUNT(*) FROM [{table_name}]"  # noqa: S608
                            ).fetchone()
                            count = row[0] if row else 0
                        except sqlite3.OperationalError:
                            count = 0
                        info["tables"].append({"name": table_name, "count": count})
                        info["total_records"] += count

                        # Find oldest timestamp
                        if count > 0:
                            ts_col = meta["ts_col"]
                            ts_type = meta["ts_type"]
                            try:
                                row = conn.execute(
                                    f"SELECT MIN([{ts_col}]) FROM [{table_name}]"  # noqa: S608
                                ).fetchone()
                                if row and row[0] is not None:
                                    if ts_type == "real":
                                        val = float(row[0])
                                        if oldest_ts is None or val < oldest_ts:
                                            oldest_ts = val
                                    else:
                                        try:
                                            dt = _dt.fromisoformat(
                                                str(row[0]).replace(" ", "T")
                                            )
                                            val = dt.replace(tzinfo=_tz.utc).timestamp()
                                            if oldest_ts is None or val < oldest_ts:
                                                oldest_ts = val
                                        except (ValueError, TypeError):
                                            pass
                            except sqlite3.OperationalError:
                                pass
                finally:
                    conn.close()

                if oldest_ts is not None:
                    info["oldest"] = oldest_ts
            except (sqlite3.Error, OSError) as exc:
                logger.debug("Failed to scan database %s: %s", db_path, exc)

            results.append(info)
        return results

    @api_router.get("/api/storage/databases")
    async def api_storage_databases() -> dict:
        """Return detailed per-database stats."""
        import asyncio

        project_root = (
            runtime.project_root if runtime and hasattr(runtime, "project_root")
            else Path(__file__).resolve().parent.parent.parent
        )
        databases = await asyncio.get_running_loop().run_in_executor(
            None, _scan_database_details, project_root,
        )
        return {"databases": databases}

    @api_router.post("/api/storage/databases/{db_id}/purge")
    async def api_purge_database(db_id: str, request: Request) -> dict:
        """Purge old records from a database."""
        import asyncio
        import time as _time

        # Find the database entry
        entry = next((e for e in _DB_REGISTRY if e["id"] == db_id), None)
        if entry is None:
            raise HTTPException(404, f"Unknown database: {db_id}")
        if not entry["purgeable"]:
            raise HTTPException(400, f"Database '{db_id}' cannot be purged")

        body = await request.json() if await request.body() else {}
        older_than_days = body.get("older_than_days")  # None means purge all
        if older_than_days is not None:
            if not isinstance(older_than_days, (int, float)) or older_than_days <= 0:
                raise HTTPException(400, "older_than_days must be a positive number")
            older_than_days = int(older_than_days)

        project_root = (
            runtime.project_root if runtime and hasattr(runtime, "project_root")
            else Path(__file__).resolve().parent.parent.parent
        )

        def _do_purge() -> dict:
            db_path = project_root / entry["path"]
            if not db_path.exists():
                return {"purged": True, "deleted_records": 0}

            conn = open_db(db_path, busy_timeout_ms=5000)
            try:
                total_deleted = 0

                for table_name, meta in entry["tables"].items():
                    ts_col = meta["ts_col"]
                    ts_type = meta["ts_type"]

                    try:
                        if older_than_days is None:
                            cur = conn.execute(
                                f"DELETE FROM [{table_name}]"  # noqa: S608
                            )
                        elif ts_type == "real":
                            cutoff = _time.time() - (older_than_days * 86400)
                            cur = conn.execute(
                                f"DELETE FROM [{table_name}] WHERE [{ts_col}] < ?",  # noqa: S608
                                (cutoff,),
                            )
                        else:
                            cur = conn.execute(
                                f"DELETE FROM [{table_name}] WHERE [{ts_col}] < datetime('now', ?)",  # noqa: S608
                                (f"-{older_than_days} days",),
                            )
                        total_deleted += cur.rowcount
                    except sqlite3.OperationalError as exc:
                        logger.warning("Purge %s.%s failed: %s", db_id, table_name, exc)

                conn.commit()

                # Best-effort VACUUM to reclaim disk space
                try:
                    conn.execute("VACUUM")
                except sqlite3.OperationalError:
                    pass
            finally:
                conn.close()

            return {"purged": True, "deleted_records": total_deleted}

        result = await asyncio.get_running_loop().run_in_executor(None, _do_purge)
        _emit_config_changed("storage", db_id=db_id)
        return result

    # ── Messages log ─────────────────────────────────────────

    @api_router.get("/api/messages")
    async def api_messages() -> dict:
        if router is None:
            return {"messages": []}
        return {"messages": router.message_log[-100:]}

    # ── Webhooks ──────────────────────────────────────────────

    @api_router.get("/api/webhooks")
    async def api_webhooks_list(request: Request) -> dict:
        if webhook_manager is None:
            return {"webhooks": []}
        hooks = webhook_manager.list_hooks() if hasattr(webhook_manager, "list_hooks") else []
        base = str(request.base_url).rstrip("/")
        result = []
        for h in hooks:
            entry = {k: v for k, v in h.items() if k != "secret"}
            entry["url"] = f"{base}/webhook/hook/{h['id']}"
            entry["has_secret"] = "secret" in h
            result.append(entry)
        return {"webhooks": result}

    @api_router.post("/api/webhooks")
    async def api_webhooks_create(request: Request) -> dict:
        if webhook_manager is None:
            raise HTTPException(status_code=503, detail="Webhook manager not available")
        body = await request.json()
        name = body.get("name", "")
        agent = body.get("agent", "")
        if not name or not agent:
            raise HTTPException(status_code=400, detail="name and agent are required")
        require_signature = bool(body.get("secret"))
        instructions = body.get("instructions", "")
        hook = webhook_manager.add_hook(
            agent=agent,
            name=name,
            require_signature=require_signature,
            instructions=instructions,
        )
        base = str(request.base_url).rstrip("/")
        # Return a copy so we don't mutate the stored dict; include
        # secret once so the user can copy it at creation time.
        result = dict(hook)
        result["url"] = f"{base}/webhook/hook/{hook['id']}"
        _emit_config_changed("webhooks", hook_id=hook["id"])
        return {"created": True, "hook": result}

    @api_router.delete("/api/webhooks/{hook_id}")
    async def api_webhooks_delete(hook_id: str) -> dict:
        if webhook_manager is None:
            raise HTTPException(status_code=503, detail="Webhook manager not available")
        removed = webhook_manager.remove_hook(hook_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Webhook '{hook_id}' not found")
        _emit_config_changed("webhooks", hook_id=hook_id)
        return {"removed": True, "id": hook_id}

    @api_router.patch("/api/webhooks/{hook_id}")
    async def api_webhooks_update(hook_id: str, request: Request) -> dict:
        if webhook_manager is None:
            raise HTTPException(status_code=503, detail="Webhook manager not available")
        body = await request.json()
        fields: dict = {}
        for key in ("name", "agent", "instructions"):
            if key in body:
                fields[key] = body[key]
        if "require_signature" in body:
            fields["require_signature"] = bool(body["require_signature"])
        if body.get("regenerate_secret"):
            fields["regenerate_secret"] = True
        if not fields:
            raise HTTPException(status_code=400, detail="No valid fields provided")
        try:
            updated = webhook_manager.update_hook(hook_id, **fields)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Webhook '{hook_id}' not found")
        base = str(request.base_url).rstrip("/")
        updated["url"] = f"{base}/webhook/hook/{updated['id']}"
        _emit_config_changed("webhooks", hook_id=hook_id)
        return {"updated": True, "hook": updated}

    @api_router.post("/api/webhooks/{hook_id}/test")
    async def api_webhooks_test(hook_id: str, request: Request) -> dict:
        if webhook_manager is None:
            raise HTTPException(status_code=503, detail="Webhook manager not available")
        body = await request.json()
        result = await webhook_manager.test_hook(hook_id, payload=body)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Webhook '{hook_id}' not found")
        return {"tested": True, "id": hook_id, "result": result}

    # ── Channels ──────────────────────────────────────────────

    _CHANNEL_TOKEN_KEYS: dict[str, list[tuple[str, str]]] = {
        "telegram": [("token", "TELEGRAM_BOT_TOKEN")],
        "discord": [("token", "DISCORD_BOT_TOKEN")],
        "slack": [("bot_token", "SLACK_BOT_TOKEN"), ("app_token", "SLACK_APP_TOKEN")],
        "whatsapp": [("access_token", "WHATSAPP_ACCESS_TOKEN"), ("phone_number_id", "WHATSAPP_PHONE_NUMBER_ID")],
    }

    @api_router.get("/api/channels")
    async def api_channels_list() -> dict:
        if channel_manager is None:
            return {"channels": []}
        return {"channels": channel_manager.get_channel_status()}

    @api_router.post("/api/channels/{channel_type}/connect")
    async def api_channel_connect(channel_type: str, request: Request) -> dict:
        if channel_manager is None:
            raise HTTPException(status_code=503, detail="Channel manager not available")
        if channel_type not in _CHANNEL_TOKEN_KEYS:
            raise HTTPException(status_code=400, detail=f"Unknown channel type: {channel_type}")
        body = await request.json()
        tokens = body.get("tokens", {})
        # Validate required token fields
        required = _CHANNEL_TOKEN_KEYS[channel_type]
        missing = [key for key, _env in required if not tokens.get(key)]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required tokens: {', '.join(missing)}")
        # Persist tokens to credential vault before starting
        persisted_env_names: list[str] = []
        if credential_vault is not None:
            for token_key, env_name in required:
                val = tokens.get(token_key, "")
                if val:
                    credential_vault.add_credential(env_name, val, system=True)
                    persisted_env_names.append(env_name)

        def _rollback_credentials() -> None:
            for env_name in persisted_env_names:
                with contextlib.suppress(Exception):
                    credential_vault.remove_credential(env_name)

        try:
            routers = await channel_manager.start_channel(channel_type, tokens)
            if routers:
                for ch_router in routers:
                    request.app.include_router(ch_router)
            _emit_config_changed("channels", channel=channel_type)
            return {"connected": True, "type": channel_type}
        except ValueError as e:
            _rollback_credentials()
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            _rollback_credentials()
            logger.error("Failed to connect channel %s: %s", channel_type, e)
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.post("/api/channels/{channel_type}/disconnect")
    async def api_channel_disconnect(channel_type: str) -> dict:
        if channel_manager is None:
            raise HTTPException(status_code=503, detail="Channel manager not available")
        if channel_type not in _CHANNEL_TOKEN_KEYS:
            raise HTTPException(status_code=400, detail=f"Unknown channel type: {channel_type}")
        try:
            channel_manager.stop_channel(channel_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        # Remove tokens from credential vault
        if credential_vault is not None:
            for _token_key, env_name in _CHANNEL_TOKEN_KEYS[channel_type]:
                with contextlib.suppress(Exception):
                    credential_vault.remove_credential(env_name)
        _emit_config_changed("channels", channel=channel_type)
        return {"disconnected": True, "type": channel_type}

    # ── Agent Workspace (proxy to agent) ─────────────────────

    _WORKSPACE_ALLOWLIST = frozenset({
        "SOUL.md", "HEARTBEAT.md", "USER.md", "INSTRUCTIONS.md", "AGENTS.md", "MEMORY.md",
        "INTERFACE.md",
        # NOTE: GOALS.md / GOALS.json are intentionally NOT listed here.
        # The dashboard exposes goals via the dedicated
        # ``GET /api/workplace/goals`` read endpoint (which calls
        # transport.request directly, bypassing this allowlist). The
        # workspace editor's PUT proxy would inject the mesh-internal
        # header and let a cookie-authed user write raw JSON to
        # GOALS.json, bypassing the ``manage_goals`` tool's validation.
        # The agent-side allowlist keeps both files so the tool's read
        # path (``/workspace/GOALS.json`` over transport) still works.
    })

    @api_router.get("/api/agents/{agent_id}/workspace")
    async def api_agent_workspace(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            return await transport.request(agent_id, "GET", "/workspace", timeout=10)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/workspace/{filename}")
    async def api_agent_workspace_read(agent_id: str, filename: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if filename not in _WORKSPACE_ALLOWLIST:
            raise HTTPException(status_code=400, detail=f"File not allowed: {filename}")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            return await transport.request(
                agent_id, "GET", f"/workspace/{filename}", timeout=10,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.put("/api/agents/{agent_id}/workspace/{filename}")
    async def api_agent_workspace_write(agent_id: str, filename: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if filename not in _WORKSPACE_ALLOWLIST:
            raise HTTPException(status_code=400, detail=f"File not allowed: {filename}")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        body = await request.json()
        content = body.get("content", "")
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="content must be a string")
        content = sanitize_for_prompt(content)
        # Operator identity edits go through this PUT (the SPA exposes
        # the operator's SOUL.md / INSTRUCTIONS.md after the safety
        # modal is acknowledged). Capture before/after into the audit
        # log so the Archive control surfaces these changes — direct
        # workspace writes otherwise bypass blackboard.log_audit and
        # would leave no trace of who edited the operator's identity.
        # Gated narrowly to operator + identity files to avoid an
        # audit-write on every dashboard workspace save.
        is_operator_identity_edit = (
            agent_id == "operator"
            and filename in ("SOUL.md", "INSTRUCTIONS.md")
            and blackboard is not None
        )
        before_value: str = ""
        if is_operator_identity_edit:
            try:
                prior = await transport.request(
                    agent_id, "GET", f"/workspace/{filename}", timeout=10,
                )
                if isinstance(prior, dict):
                    before_value = str(prior.get("content") or "")
            except Exception as e:
                # Pre-fetch is best-effort; the audit row will record
                # an empty before_value rather than blocking the edit.
                logger.warning(
                    "operator workspace pre-fetch failed for %s: %s",
                    filename, e,
                )
        try:
            result = await transport.request(
                agent_id, "PUT", f"/workspace/{filename}",
                json={"content": content}, timeout=10,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
        if is_operator_identity_edit:
            try:
                blackboard.log_audit(
                    action="edit_workspace",
                    target=agent_id,
                    field=filename,
                    before_value=before_value,
                    after_value=content,
                    actor="user",
                    provenance="dashboard",
                )
            except Exception as e:
                logger.warning(
                    "Failed to write operator workspace audit row: %s", e,
                )
        if event_bus is not None:
            event_bus.emit("workspace_updated", agent=agent_id,
                           data={"message": f"Dashboard updated {filename}"})
        return result

    # ── Agent Workspace Logs + Learnings (proxy to agent) ─────

    @api_router.get("/api/agents/{agent_id}/workspace-logs")
    async def api_agent_workspace_logs(agent_id: str, days: int = 3) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        days = max(1, min(days, 14))
        try:
            return await transport.request(
                agent_id, "GET", f"/workspace-logs?days={days}", timeout=10,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @api_router.get("/api/agents/{agent_id}/workspace-learnings")
    async def api_agent_workspace_learnings(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        try:
            return await transport.request(
                agent_id, "GET", "/workspace-learnings", timeout=10,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    # ── Agent standing goals (Team store ``agent_goals``) ──────

    # Human/dashboard surface for the per-agent standing goals the
    # operator writes via ``set_agent_goals`` — injected into the
    # agent's every prompt as "## Your Current Goals" and pursued
    # during idle heartbeats. Ratified #7 / C.3-b: goals live in the
    # Team store, keyed by agent alone (they follow the agent across
    # team moves). Trust model: the dashboard is the full-trust human
    # surface writing through the trusted store directly — the mesh
    # ``PUT /mesh/agents/{id}/goals`` operator gate is for AGENT
    # callers; no agent identity is involved here.

    _MAX_AGENT_GOALS = 5
    _MAX_AGENT_GOAL_CHARS = 300

    @api_router.get("/api/agents/{agent_id}/goals")
    async def api_agent_goals_get(agent_id: str) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        record = teams_store.get_agent_goals(agent_id)
        if record is None:
            return {"goals": [], "updated_at": None}
        raw = record.get("goals", [])
        goals = [str(g) for g in raw] if isinstance(raw, list) else []
        return {
            "goals": goals,
            "updated_at": record.get("updated_at"),
            "set_by": record.get("set_by"),
        }

    @api_router.put("/api/agents/{agent_id}/goals")
    async def api_agent_goals_put(agent_id: str, request: Request) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if agent_id == "operator":
            # Parity with the ``set_agent_goals`` tool, which refuses
            # the operator — its fleet/business goals live in GOALS.json
            # and are managed from the Work tab.
            raise HTTPException(
                status_code=400,
                detail=(
                    "The operator's fleet/business goals live in "
                    "GOALS.json — manage them from the Work tab's goals "
                    "strip, not here."
                ),
            )
        body = await request.json()
        goals = body.get("goals")
        if not isinstance(goals, list):
            raise HTTPException(status_code=400, detail="goals must be a list of strings")
        if len(goals) > _MAX_AGENT_GOALS:
            raise HTTPException(
                status_code=400,
                detail=f"goals exceeds max length {_MAX_AGENT_GOALS}",
            )
        cleaned: list[str] = []
        for g in goals:
            if not isinstance(g, str):
                raise HTTPException(status_code=400, detail="each goal must be a string")
            s = sanitize_for_prompt(g).strip()
            if not s:
                raise HTTPException(status_code=400, detail="each goal must be a non-empty string")
            if len(s) > _MAX_AGENT_GOAL_CHARS:
                raise HTTPException(
                    status_code=400,
                    detail=f"each goal must be <={_MAX_AGENT_GOAL_CHARS} chars (one sentence)",
                )
            cleaned.append(s)

        prior = teams_store.get_agent_goals(agent_id)
        prior_goals = []
        if prior is not None and isinstance(prior.get("goals"), list):
            prior_goals = prior["goals"]

        if not cleaned:
            # clear_agent_goals is idempotent — clearing unset goals is fine.
            teams_store.clear_agent_goals(agent_id)
            blackboard.log_audit(
                action="clear_goals",
                target=agent_id,
                field="goals",
                before_value="\n".join(str(g) for g in prior_goals),
                actor="user",
                provenance="dashboard",
            )
            return {"cleared": True, "agent_id": agent_id}

        teams_store.set_agent_goals(agent_id, cleaned, set_by="user")
        blackboard.log_audit(
            action="edit_goals",
            target=agent_id,
            field="goals",
            before_value="\n".join(str(g) for g in prior_goals),
            after_value="\n".join(cleaned),
            actor="user",
            provenance="dashboard",
        )
        return {
            "set": True,
            "agent_id": agent_id,
            "count": len(cleaned),
            "note": (
                "Takes effect on the agent's next prompt build (<=5 min cache)."
            ),
        }

    # ── Agent Activity Log ────────────────────────────────────

    @api_router.get("/api/agents/{agent_id}/activity")
    async def api_agent_activity(agent_id: str, limit: int = 100) -> dict:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        if transport is None:
            raise HTTPException(status_code=503, detail="Transport not available")
        limit = max(1, min(limit, 500))
        try:
            return await transport.request(
                agent_id, "GET", f"/activity?limit={limit}", timeout=10,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    # ── Logs ──────────────────────────────────────────────────

    @api_router.get("/api/logs")
    async def api_logs(lines: int = 100, level: str = "") -> dict:
        """Return recent log lines from .openlegion.log."""
        from src.cli.config import PROJECT_ROOT

        log_path = PROJECT_ROOT / ".openlegion.log"
        if not log_path.exists():
            return {"lines": [], "total": 0}

        content = log_path.read_text()
        all_lines = content.splitlines()

        if level:
            level_upper = level.upper()
            level_pat = re.compile(r'\b' + re.escape(level_upper) + r'\b')
            all_lines = [ln for ln in all_lines if level_pat.search(ln.upper())]

        result_lines = all_lines[-lines:]
        return {"lines": result_lines, "total": len(all_lines)}

    # ── Static files ─────────────────────────────────────────

    _MEDIA_TYPES = {
        ".css": "text/css",
        ".js": "application/javascript",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
    }

    # ── User Uploads ─────────────────────────────────────────────────────
    # User-managed files that agents can read (read-only) and the VNC browser
    # can navigate to at http://localhost:8500/uploads/<filename>.
    # All endpoints read/write the host uploads dir directly — no transport.

    def _uploads_dir() -> Path:
        root = (
            runtime.project_root if runtime and hasattr(runtime, "project_root")
            else Path(__file__).resolve().parent.parent.parent
        )
        d = root / ".openlegion" / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    _MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

    def _safe_upload_path(name: str) -> Path:
        r"""Resolve upload path, blocking traversal, absolute paths, and null bytes.

        Two-stage check:
          1. Structural: reject absolute paths and any '..' component
             using Path.parts (platform-aware, handles both / and \).
          2. Symlink-safe: resolve and verify the final path is inside root.
        """
        try:
            p = Path(name)
        except (ValueError, TypeError):
            # ValueError is raised for embedded null bytes on all platforms.
            raise HTTPException(400, "Invalid path")
        if p.is_absolute() or ".." in p.parts:
            raise HTTPException(400, "Invalid path")
        candidate = resolve_under_root(_uploads_dir(), name)
        if candidate is None:
            raise HTTPException(400, "Path traversal not allowed")
        return candidate

    @api_router.get("/api/uploads")
    async def api_list_uploads() -> dict:
        """List all files in the uploads directory."""
        import mimetypes
        root = _uploads_dir()
        entries = []
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            rel = str(f.relative_to(root))
            stat = f.stat()
            mime = mimetypes.guess_type(rel)[0] or "application/octet-stream"
            entries.append({
                "name": rel,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "mime_type": mime,
            })
        return {"uploads": entries}

    @api_router.post("/api/uploads/{name:path}")
    async def api_upload_file(name: str, request: Request) -> dict:
        """Upload a file to the uploads directory.

        Accepts raw bytes in the request body.  The caller sets Content-Type
        so the browser download later uses the right MIME type.
        Maximum upload size: 50 MB.
        """
        dest = _safe_upload_path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = await request.body()
        if not body:
            raise HTTPException(400, "Empty body")
        if len(body) > _MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File too large (max {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
        dest.write_bytes(body)
        _emit_config_changed("uploads", name=name)
        return {"uploaded": True, "name": name, "size": len(body)}

    @api_router.get("/api/uploads/{name:path}/download")
    async def api_download_upload(name: str):
        """Download a file from the uploads directory with correct Content-Type."""
        import mimetypes

        from fastapi.responses import Response
        path = _safe_upload_path(name)
        if not path.exists() or not path.is_file():
            raise HTTPException(404, f"Upload not found: {name}")
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return Response(
            content=path.read_bytes(),
            media_type=mime,
            headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
        )

    @api_router.delete("/api/uploads/{name:path}")
    async def api_delete_upload(name: str) -> dict:
        """Delete an uploaded file."""
        path = _safe_upload_path(name)
        if not path.exists() or not path.is_file():
            raise HTTPException(404, f"Upload not found: {name}")
        path.unlink()
        # Clean up empty parent dirs up to (but not including) root
        root = _uploads_dir().resolve()
        parent = path.parent
        while parent != root and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        _emit_config_changed("uploads", name=name)
        return {"deleted": True, "name": name}

    # ── Task 9 — Workplace tab + pending-action review ──────────────
    #
    # The Workplace tab is a top-level peer of Chat / Agents / System
    # under ``src/dashboard/static/js/app.js``. It surfaces the durable
    # task records and the pending-action queue (Task 2d) so a human
    # can run an agent team end-to-end without inspecting blackboard
    # keys directly.

    @api_router.get("/api/workplace/tasks")
    async def api_workplace_tasks(
        team_id: str | None = None,
        assignee: str | None = None,
        status: str | None = None,
    ) -> dict:
        """List task records for the Workplace board.

        Returns ``{"enabled": False, ...}`` when orchestration v2 is
        disabled so the SPA can render the empty state with a hint to
        flip the flag — matches the contract in tests.
        """
        try:
            if assignee:
                rows = tasks_store.list_inbox(
                    assignee, team_id=team_id, include_terminal=True,
                )
            elif team_id:
                rows = tasks_store.list_team(team_id)
            else:
                # Fleet-wide listing — operator-only data is acceptable
                # because the dashboard is operator-authenticated.
                with tasks_store._conn() as conn:
                    sql = (
                        f"SELECT {tasks_store._SELECT_COLS} FROM tasks "
                        "ORDER BY created_at DESC LIMIT 500"
                    )
                    raw = conn.execute(sql).fetchall()
                rows = [tasks_store._row_to_dict(r) for r in raw]
            if status:
                rows = [r for r in rows if r.get("status") == status]
            return {"enabled": True, "tasks": rows}
        except Exception as e:
            logger.warning("workplace tasks listing failed: %s", e)
            return {"enabled": True, "tasks": [], "error": str(e)}

    @api_router.get("/api/workplace/teams")
    async def api_workplace_teams() -> dict:
        """Team status rollups for the Workplace > team-status tab."""
        teams_cfg = teams_store.list_teams()
        result = []
        for pname, pdata in teams_cfg.items():
            try:
                rows = tasks_store.list_team(pname)
            except Exception:
                rows = []
            counts: dict[str, int] = {}
            blockers: list[dict] = []
            for r in rows:
                counts[r["status"]] = counts.get(r["status"], 0) + 1
                if r["status"] == "blocked":
                    blockers.append({
                        "task_id": r["id"],
                        "title": r["title"],
                        "assignee": r["assignee"],
                        "blocker_note": r.get("blocker_note") or "",
                    })
            result.append({
                "name": pname,
                "team_name": pname,
                "description": pdata.get("description", ""),
                "members": pdata.get("members", []),
                "status": pdata.get("status", "active"),
                "north_star": pdata.get("north_star"),
                "success_criteria": pdata.get("success_criteria"),
                "counts": counts,
                "total": len(rows),
                "blockers": blockers,
            })
        return {"enabled": True, "teams": result}

    @api_router.get("/api/workplace/summaries")
    async def api_workplace_summaries(
        scope_kind: str | None = None,
        scope_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """Work summaries for the Work-tab default landing (PR-B).

        Returns ``{enabled: False, summaries: []}`` when the store
        isn't wired (legacy dashboards / standalone tests). Scope
        filter is server-side; the dashboard already runs as the
        operator persona, so visibility expansion is unnecessary —
        operator sees everything via the mesh endpoint's bypass.
        """
        if summaries_store is None:
            return {"enabled": False, "summaries": []}
        limit = max(1, min(int(limit or 100), 500))
        offset = max(0, int(offset or 0))
        try:
            rows = summaries_store.list_recent(
                scope_kind=scope_kind, scope_id=scope_id,
                limit=limit, offset=offset,
            )
            summaries_store._safe_reap()
        except Exception as e:
            logger.warning("workplace summaries listing failed: %s", e)
            return {"enabled": True, "summaries": [], "error": str(e)}
        return {"enabled": True, "summaries": rows}

    @api_router.get("/api/workplace/summaries/{summary_id}")
    async def api_workplace_summary_detail(summary_id: str) -> dict:
        if summaries_store is None:
            raise HTTPException(404, "Summaries store not configured")
        row = summaries_store.get(summary_id)
        if row is None:
            raise HTTPException(404, f"Summary {summary_id!r} not found")
        return row

    @api_router.post("/api/workplace/summaries/{summary_id}/rating")
    async def api_workplace_summary_rate(
        summary_id: str, request: Request,
    ) -> dict:
        """Operator persona (dashboard) records the user's rating.

        CSRF is enforced by the router-level ``_csrf_check`` dependency.
        Body: ``{rating: "accepted"|"acknowledged"|"rework", feedback: str}``.
        ``rework`` requires non-empty feedback so the next summary's
        composition has something to act on (mirrors the per-task
        outcome rule).
        """
        if summaries_store is None:
            raise HTTPException(404, "Summaries store not configured")
        from src.host.summaries import (
            MAX_FEEDBACK_CHARS,
            VALID_RATINGS,
            RatingLocked,
            SummaryNotFound,
        )
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(400, "Invalid JSON body") from e
        if not isinstance(body, dict):
            raise HTTPException(400, "Body must be a JSON object")
        rating = body.get("rating")
        feedback = body.get("feedback") or ""
        if rating not in VALID_RATINGS:
            raise HTTPException(
                400, f"rating must be one of {sorted(VALID_RATINGS)}",
            )
        if not isinstance(feedback, str):
            raise HTTPException(400, "feedback must be a string")
        feedback = feedback.strip()
        if len(feedback) > MAX_FEEDBACK_CHARS:
            raise HTTPException(
                400, f"feedback exceeds {MAX_FEEDBACK_CHARS} chars",
            )
        if rating == "rework" and not feedback:
            raise HTTPException(
                400, "feedback is required for rating='rework'",
            )
        try:
            return summaries_store.set_rating(
                summary_id, rating, feedback or None, actor="operator",
            )
        except SummaryNotFound:
            raise HTTPException(404, f"Summary {summary_id!r} not found")
        except RatingLocked as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @api_router.get("/api/workplace/goals")
    async def api_workplace_goals() -> dict:
        """Return the operator's tracked goals for the Work-tab strip.

        Reads ``GOALS.json`` from the operator container's workspace via
        the transport proxy. Returns ``{enabled: False, goals: []}`` when
        the dashboard can't reach the operator (legacy / standalone test
        configs); the frontend renders that as "no goals tracked yet"
        without erroring.
        """
        empty = {"enabled": False, "goals": []}
        if transport is None:
            return empty
        if "operator" not in agent_registry:
            return empty
        try:
            payload = await transport.request(
                "operator", "GET", "/workspace/GOALS.json", timeout=10,
            )
        except Exception as e:
            # Log the real error for ops; surface a generic message to
            # the browser so internal hostnames / file paths from httpx
            # don't leak into the dashboard.
            logger.warning("operator goals fetch failed: %s", e)
            return {
                "enabled": True,
                "goals": [],
                "error": "unable to reach operator",
            }
        raw = ""
        if isinstance(payload, dict):
            raw = str(payload.get("content") or "")
        if not raw.strip():
            return {"enabled": True, "goals": []}
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError) as e:
            logger.warning("operator GOALS.json parse failed: %s", e)
            return {"enabled": True, "goals": []}
        if not isinstance(parsed, dict):
            return {"enabled": True, "goals": []}
        entries = parsed.get("goals", [])
        if not isinstance(entries, list):
            return {"enabled": True, "goals": []}
        cleaned: list[dict] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cleaned.append({
                "name": str(entry.get("name", "")),
                "status": str(entry.get("status", "")),
                "progress_note": str(entry.get("progress_note", "")),
                "updated_at": str(entry.get("updated_at", "")),
            })
        return {"enabled": True, "goals": cleaned}

    @api_router.get("/api/workplace/blockers")
    async def api_workplace_blockers() -> dict:
        """Fleet-wide blocked-task list for the Workplace > blockers tab."""
        try:
            with tasks_store._conn() as conn:
                sql = (
                    f"SELECT {tasks_store._SELECT_COLS} FROM tasks "
                    "WHERE status='blocked' ORDER BY updated_at DESC LIMIT 200"
                )
                raw = conn.execute(sql).fetchall()
            rows = [tasks_store._row_to_dict(r) for r in raw]
        except Exception as e:
            logger.warning("workplace blockers listing failed: %s", e)
            rows = []
        return {"enabled": True, "blockers": rows}

    @api_router.get("/api/workplace/pipelines")
    async def api_workplace_pipelines() -> dict:
        """In-flight operator-rooted chains + stage progress (live pipeline card).

        Reuses ``list_watchable_human_roots`` (human-origin chain roots not yet
        terminally delivered, within the window) and ``workflow_snapshot`` per
        root. Only chains with a non-terminal stage are returned (in-flight),
        so completed chains drop off automatically. A chain is flagged
        ``stalled`` when a stage is blocked or has been ``working`` a long time
        — mirroring the stall watchdog's signal for the UI.
        """
        if tasks_store is None:
            return {"enabled": False, "pipelines": []}
        import time as _t
        window_s = 24 * 3600   # show in-flight chains from the last day
        slow_s = 300           # a working stage older than this reads as slow
        _terminal = ("done", "failed", "cancelled")
        pipelines: list[dict] = []
        try:
            roots = tasks_store.list_watchable_human_roots(
                since=_t.time() - window_s,
            )
            # Bound the per-root snapshot work: only the most-recent
            # _MAX_PIPELINE_ROOTS are examined. This endpoint runs on every
            # task lifecycle event (WS-debounced), and each root costs a
            # recursive workflow_snapshot — without a cap a busy fleet could
            # turn each event into O(many roots × CTE). In-flight chains are
            # recent, so the newest N cover the realistic card.
            roots.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
            roots = roots[:_MAX_PIPELINE_ROOTS]
            for root in roots:
                snap = tasks_store.workflow_snapshot(root["id"])
                if not snap:
                    continue
                stages = snap.get("stages", [])
                if not any(s.get("status") not in _terminal for s in stages):
                    continue  # wholly terminal — not in-flight
                stalled = any(
                    s.get("status") == "blocked"
                    or (s.get("status") == "working"
                        and (s.get("age_in_state_seconds") or 0) > slow_s)
                    for s in stages
                )
                pipelines.append({
                    "root_task_id": root["id"],
                    "title": root.get("title") or "",
                    "assignee": root.get("assignee") or "",
                    "created_at": root.get("created_at"),
                    # origin lets the chat tab filter to ITS chains
                    # (channel == "dashboard") for the watch chips.
                    "origin": root.get("origin") or {},
                    "updated_at": root.get("updated_at"),
                    "stages": stages,
                    "summary": snap.get("summary", {}),
                    "stalled": stalled,
                })
        except Exception as e:
            logger.warning("workplace pipelines listing failed: %s", e)
            return {"enabled": True, "pipelines": []}
        pipelines.sort(key=lambda p: p.get("created_at") or 0, reverse=True)
        return {"enabled": True, "pipelines": pipelines}

    @api_router.get("/api/workplace/pending")
    async def api_workplace_pending() -> dict:
        """List open pending actions for inline + System>Operator review.

        Proxies to the mesh's ``GET /mesh/pending`` over loopback with
        the ``x-mesh-internal`` + ``X-Agent-ID: operator`` headers so the
        operator-or-internal permission tier is enforced. Reading
        ``pending_actions`` directly here would bypass that gate and let
        any session with an ``ol_session`` cookie enumerate pending
        actions even when the underlying mesh route is restricted. The
        dashboard's CSRF guard runs ahead of this on state-changing
        verbs; this is a GET so CSRF is not relevant.
        """
        import httpx
        url = f"http://127.0.0.1:{mesh_port}/mesh/pending"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    url,
                    headers={
                        "x-mesh-internal": "1",
                        "X-Agent-ID": "operator",
                    },
                )
        except Exception as e:
            logger.warning("workplace pending listing proxy failed: %s", e)
            return {"pending": []}
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise HTTPException(resp.status_code, detail)
        try:
            return resp.json()
        except Exception:
            return {"pending": []}

    @api_router.post("/api/workplace/pending/{nonce}/confirm")
    async def api_workplace_pending_confirm(
        nonce: str, request: Request,
    ) -> dict:
        """Confirm a pending action — backs every dashboard "Confirm" button.

        The browser must NOT hit ``/mesh/pending/{nonce}/confirm``
        directly. The mesh endpoint requires (a) a bearer token (in
        production with ``OPENLEGION_AUTH_TOKEN`` set) OR
        ``x-mesh-internal`` + loopback, AND (b) an ``X-Origin`` header
        with ``kind="human"`` for the ``_confirm_origin_check`` gate.
        A browser session has neither, so the call returned 401 (prod)
        or 403 ("Confirmation requires human origin", dev) — the exact
        symptom users hit when clicking Confirm on a pending action.

        This proxy:
          * Forwards over loopback with ``x-mesh-internal=1`` +
            ``X-Agent-ID: operator`` so the mesh trusts identity.
          * Mints a ``human`` ``MessageOrigin`` from the dashboard
            session id and stamps it via ``X-Origin`` so
            ``_confirm_origin_check`` passes.
          * Threads ``payload_digest`` through from the request body
            (the inline pending-action chat card sends it to detect
            in-flight schema drift on hard-field edits).

        Returns the mesh response on success; mirrors 404 / non-2xx
        upstream codes so the SPA's existing error handling keeps
        working.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        payload_digest = body.get("payload_digest")
        from src.shared.trace import origin_header
        from src.shared.types import MessageOrigin
        origin = MessageOrigin(
            kind="human",
            channel="dashboard",
            user=_operator_session_id(request),
        )
        import httpx
        url = f"http://127.0.0.1:{mesh_port}/mesh/pending/{nonce}/confirm"
        fwd_body: dict = {}
        if payload_digest is not None:
            fwd_body["payload_digest"] = payload_digest
        headers = {
            "X-Requested-With": "XMLHttpRequest",
            "x-mesh-internal": "1",
            "X-Agent-ID": "operator",
            "Content-Type": "application/json",
        }
        headers.update(origin_header(origin))
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, json=fwd_body, headers=headers)
        except Exception as e:
            logger.warning("workplace pending confirm proxy failed: %s", e)
            raise HTTPException(502, f"Mesh unreachable: {e}")
        if resp.status_code == 404:
            raise HTTPException(
                404, "Pending action not found or already expired",
            )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise HTTPException(resp.status_code, detail)
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "nonce": nonce}

    @api_router.post("/api/workplace/pending/{nonce}/cancel")
    async def api_workplace_pending_cancel(nonce: str) -> dict:
        """Cancel a pending action — backs the dashboard "Cancel" button.

        Proxies to the mesh's ``POST /mesh/pending/{nonce}/cancel``
        endpoint over loopback with the ``x-mesh-internal`` +
        ``X-Agent-ID: operator`` headers. The mesh route enforces the
        operator-or-internal permission tier and emits the
        ``pending_action_resolved`` event with ``status="cancelled"``.
        Calling ``pending_actions.cancel`` directly here would bypass
        that gate. The dashboard's CSRF guard (``X-Requested-With``)
        is still enforced by the global middleware.
        """
        import httpx
        url = f"http://127.0.0.1:{mesh_port}/mesh/pending/{nonce}/cancel"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    url,
                    json={},
                    headers={
                        "X-Requested-With": "XMLHttpRequest",
                        "x-mesh-internal": "1",
                        "X-Agent-ID": "operator",
                        "Content-Type": "application/json",
                    },
                )
        except Exception as e:
            logger.warning("workplace pending cancel proxy failed: %s", e)
            raise HTTPException(502, f"Mesh unreachable: {e}")
        if resp.status_code == 404:
            raise HTTPException(404, "Pending action not found or already expired")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise HTTPException(resp.status_code, detail)
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "nonce": nonce}

    @api_router.post("/api/changes/undo/{undo_token}")
    async def api_changes_undo(undo_token: str, request: Request) -> dict:
        """Reverse a recent soft edit (PR 1).


        Backs the [Undo] button on the operator_action_receipt card.
        Proxies to the mesh's ``/mesh/changes/undo/{token}`` endpoint over
        loopback so the YAML/permissions writes go through the canonical
        mesh helper rather than being duplicated here. Sets the
        ``x-mesh-internal`` + ``X-Agent-ID: operator`` headers so the
        mesh's ``_resolve_agent_id`` recognizes us as the operator
        (the dashboard is the trusted in-process caller).
        """
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"http://127.0.0.1:{mesh_port}/mesh/changes/undo/{undo_token}",
                    json={},
                    headers={
                        "X-Requested-With": "XMLHttpRequest",
                        "x-mesh-internal": "1",
                        "X-Agent-ID": "operator",
                    },
                )
        except Exception as e:
            raise HTTPException(502, f"Mesh unreachable: {e}")
        if resp.status_code == 404:
            raise HTTPException(404, "Undo token unknown, expired, or already used")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise HTTPException(resp.status_code, detail)
        return resp.json()

    # ── Task 9 PR 4 — Workplace task drill-in + outcome capture ────
    #
    # Three endpoints back the per-task modal: a full snapshot
    # (task + events + resolved artifacts), a polling-cheap events-only
    # variant, and the POST that records an operator outcome (and
    # optionally spawns a rework task with the same assignee).

    _ARTIFACT_INLINE_CHAR_CAP = 10_000

    def _resolve_artifact(ref: str) -> dict:
        """Resolve a task ``artifact_ref`` to an inline-renderable dict.

        Today every artifact_ref is a Blackboard key (the
        ``output/{agent}/{handoff_id}`` pattern that
        ``coordination_tool.hand_off`` writes). The dict shape is
        ``{ref, kind, content?, content_truncated?, error?}`` so the
        UI can render text inline (kind=``text``) and surface
        unresolved refs (kind=``missing`` / ``error``) without
        crashing the whole drill-in. Future storage backends (S3,
        signed URLs) plug in here by returning ``kind="url"`` with a
        ``url`` field.
        """
        out: dict = {"ref": ref}
        if not ref:
            out.update({"kind": "missing", "error": "empty ref"})
            return out
        try:
            entry = blackboard.read(ref)
        except Exception as e:
            out.update({"kind": "error", "error": str(e)})
            return out
        if entry is None:
            out.update({"kind": "missing"})
            return out
        value = entry.value
        # Render the parsed JSON value back to text for display so
        # the modal can show whatever the agent wrote without a
        # second round-trip. We treat the dict ``{text: ...}`` shape
        # specially because hand_off wraps free text that way.
        if isinstance(value, dict) and set(value.keys()) == {"text"}:
            text = str(value.get("text") or "")
        elif isinstance(value, str):
            text = value
        else:
            try:
                text = dumps_safe(value, indent=2)
            except (TypeError, ValueError):
                text = str(value)
        truncated = False
        if len(text) > _ARTIFACT_INLINE_CHAR_CAP:
            text = text[:_ARTIFACT_INLINE_CHAR_CAP]
            truncated = True
        out.update({
            "kind": "text",
            "content": text,
            "content_truncated": truncated,
            "written_by": entry.written_by,
            "updated_at": entry.updated_at,
        })
        return out

    @api_router.get("/api/workplace/tasks/{task_id}")
    async def api_workplace_task_detail(task_id: str) -> dict:
        """Return a task plus its event timeline and resolved artifacts.

        Used by the Workplace drill-in modal. ``artifacts`` is a list of
        ``{ref, kind, content, ...}`` dicts — text is inlined up to
        10 k chars, missing/error refs return a ``kind`` marker rather
        than 500'ing so a partial timeline still renders.
        """
        try:
            task = tasks_store.get(task_id)
        except Exception as e:
            logger.warning("workplace task fetch failed: %s", e)
            raise HTTPException(500, "task fetch failed") from e
        if task is None:
            raise HTTPException(404, "Task not found")
        try:
            events = tasks_store.list_events(task_id)
        except Exception as e:
            logger.warning("workplace task events fetch failed: %s", e)
            events = []
        artifacts = [_resolve_artifact(ref) for ref in task.get("artifact_refs") or []]
        return {"task": task, "events": events, "artifacts": artifacts}

    @api_router.get("/api/workplace/tasks/{task_id}/events")
    async def api_workplace_task_events(task_id: str) -> dict:
        """Return just the event timeline for a task (cheap to poll)."""
        if tasks_store.get(task_id) is None:
            raise HTTPException(404, "Task not found")
        try:
            events = tasks_store.list_events(task_id)
        except Exception as e:
            logger.warning("workplace task events fetch failed: %s", e)
            events = []
        return {"task_id": task_id, "events": events}

    @api_router.post("/api/workplace/tasks/{task_id}/cancel")
    async def api_workplace_task_cancel(
        task_id: str, request: Request,
    ) -> dict:
        """Cancel a task — backs the [×] button on every kanban card.

        Proxies to the mesh's ``POST /mesh/tasks/{task_id}/cancel``
        endpoint over loopback with ``x-mesh-internal: 1`` +
        ``X-Agent-ID: operator`` so the mesh treats the dashboard as a
        trusted internal caller (the mesh route allows creator,
        assignee, operator, or internal). The dashboard's CSRF guard
        (``X-Requested-With``) runs ahead of this on the verb, and the
        cancel itself emits ``status_changed`` + ``task_status_changed``
        events so the kanban refreshes via the existing WebSocket
        plumbing without a manual reload.

        Body: ``{reason: str}`` (optional, sanitized by the mesh).
        Returns: the updated task record (``status="cancelled"``) so
        the caller can patch the UI optimistically.
        """
        try:
            body = await request.json() if (await request.body()) else {}
        except (json.JSONDecodeError, ValueError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        reason = body.get("reason") or ""
        import httpx
        url = f"http://127.0.0.1:{mesh_port}/mesh/tasks/{task_id}/cancel"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    url,
                    json={"reason": reason},
                    headers={
                        "X-Requested-With": "XMLHttpRequest",
                        "x-mesh-internal": "1",
                        "X-Agent-ID": "operator",
                        "Content-Type": "application/json",
                    },
                )
        except Exception as e:
            logger.warning("workplace task cancel proxy failed: %s", e)
            raise HTTPException(502, f"Mesh unreachable: {e}")
        if resp.status_code == 404:
            raise HTTPException(404, "Task not found")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise HTTPException(resp.status_code, detail)
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "task_id": task_id}

    @api_router.post("/api/workplace/tasks/{task_id}/outcome")
    async def api_workplace_task_outcome(
        task_id: str, request: Request,
    ) -> dict:
        """Record an operator outcome rating on a completed task.

        Body: ``{outcome: "accepted"|"acknowledged"|"rework"|"rejected",
        feedback: str}``. ``accepted`` and ``acknowledged`` allow empty
        feedback (the rating itself is the signal). ``rework`` and
        ``rejected`` require a non-empty comment so the agent / audit
        trail has something to learn from. For ``rework``, also spawns a
        new linked task with the feedback as its brief and the same
        assignee — the new task id is returned in ``rework_task_id``.
        """
        from src.host.orchestration import (
            MAX_FEEDBACK_CHARS,
            VALID_OUTCOMES,
            InvalidStatusTransition,
            TaskNotFound,
        )
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(400, "Invalid JSON body") from e
        if not isinstance(body, dict):
            raise HTTPException(400, "Body must be a JSON object")
        outcome = body.get("outcome")
        feedback = body.get("feedback") or ""
        if outcome not in VALID_OUTCOMES:
            raise HTTPException(
                400,
                f"outcome must be one of {sorted(VALID_OUTCOMES)}",
            )
        if not isinstance(feedback, str):
            raise HTTPException(400, "feedback must be a string")
        feedback = feedback.strip()
        if len(feedback) > MAX_FEEDBACK_CHARS:
            raise HTTPException(
                400, f"feedback exceeds {MAX_FEEDBACK_CHARS} chars",
            )
        # Rework + reject require a non-empty comment so the agent /
        # audit trail has something to learn from. Accept can be
        # silent (the rating itself is the signal).
        if outcome in ("rework", "rejected") and not feedback:
            raise HTTPException(
                400, f"feedback is required for outcome={outcome!r}",
            )
        try:
            updated = tasks_store.set_outcome(
                task_id, outcome, feedback or None, actor="operator",
            )
        except TaskNotFound as e:
            raise HTTPException(404, "Task not found") from e
        except InvalidStatusTransition as e:
            raise HTTPException(409, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        result: dict = {"ok": True, "task": updated}
        # A1 — push actionable feedback into the rated agent's learnings
        # (best-effort; see src/host/feedback_push.py for the contract).
        from src.host.feedback_push import push_outcome_feedback
        push_status = await push_outcome_feedback(
            transport, updated, outcome, feedback,
        )
        if push_status:
            result["feedback_push"] = push_status
        if outcome == "rework":
            try:
                rework = tasks_store.create_rework_task(
                    task_id, feedback, actor="operator",
                )
            except (TaskNotFound, ValueError) as e:
                # The outcome itself committed; surface the rework
                # failure in the response without rolling back the
                # rating.
                logger.warning("rework spawn failed for %s: %s", task_id, e)
                result["rework_error"] = str(e)
            else:
                result["rework_task_id"] = rework["id"]
                result["rework_assignee"] = rework["assignee"]
        return result

    async def _proxy_help_cancel(
        kind: str, request_id: str, body: dict | None,
    ) -> dict:
        """Proxy a Cancel-card click to the matching mesh endpoint.

        Goes over loopback with ``x-mesh-internal: 1`` so the mesh
        treats the dashboard as a trusted internal caller (same
        contract as ``X-Agent-ID: operator`` would have used). Single
        round-trip — both processes are co-located in the runtime.
        """
        import httpx
        url = f"http://127.0.0.1:{mesh_port}/mesh/{kind}-request/{request_id}/cancel"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    url,
                    json=body or {},
                    headers={
                        "x-mesh-internal": "1",
                        "X-Agent-ID": "operator",
                        "Content-Type": "application/json",
                    },
                )
        except Exception as e:
            raise HTTPException(502, f"mesh cancel proxy failed: {e}")
        if resp.status_code == 404:
            raise HTTPException(404, "Request not found or already resolved")
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, resp.text)
        try:
            return resp.json()
        except Exception:
            return {"ok": True}

    @api_router.get("/api/help-requests")
    async def api_help_requests(request: Request) -> dict:
        """Open help requests for the "Needs you" panel — proxied from the
        authoritative mesh registry.

        Distinguishes a backend error (502) from "nothing open" ({items: []})
        so an outage never reads to the user as "nothing needs you". The
        frontend renders an explicit error state on non-200.
        """
        import httpx
        url = f"http://127.0.0.1:{mesh_port}/mesh/help-requests"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    url, headers={"x-mesh-internal": "1", "X-Agent-ID": "operator"},
                )
        except Exception as e:
            raise HTTPException(502, f"mesh help-requests proxy failed: {e}")
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, resp.text)
        return resp.json()

    @api_router.post("/api/credential-request/{request_id}/cancel")
    async def api_credential_request_cancel(
        request_id: str, request: Request,
    ) -> dict:
        """Cancel an open credential-request card (PR 3)."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        return await _proxy_help_cancel("credential", request_id, body)

    @api_router.post("/api/browser-login-request/{request_id}/cancel")
    async def api_browser_login_request_cancel(
        request_id: str, request: Request,
    ) -> dict:
        """Cancel an open browser-login-request card (PR 3)."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        return await _proxy_help_cancel("browser-login", request_id, body)

    @api_router.post("/api/browser-captcha-help-request/{request_id}/cancel")
    async def api_browser_captcha_help_request_cancel(
        request_id: str, request: Request,
    ) -> dict:
        """Cancel an open browser-captcha-help-request card (PR 3)."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        return await _proxy_help_cancel(
            "browser-captcha-help", request_id, body,
        )

    @api_router.get("/static/{file_path:path}")
    async def static_file(file_path: str, v: str | None = None) -> FileResponse:
        full = resolve_under_root(_STATIC_DIR, file_path)
        if full is None or not full.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        suffix = full.suffix.lower()
        # When served with a versioned URL (?v=<hash>), cache aggressively —
        # the URL changes whenever file content changes.  Without a version
        # param (direct access, bookmarks), prevent caching entirely.
        cache = "public, max-age=86400, immutable" if v else "no-store"
        return FileResponse(
            str(full),
            media_type=_MEDIA_TYPES.get(suffix),
            headers={"Cache-Control": cache},
        )

    return api_router


def create_spa_catchall_router() -> APIRouter:
    """Root-level catch-all for SPA deep linking (no /dashboard/ prefix).

    Must be included LAST on the app so it never shadows mesh/dashboard routes.
    """
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    catchall = APIRouter(dependencies=[Depends(_verify_dashboard_auth)])

    @catchall.get("/{path:path}", response_class=HTMLResponse)
    async def spa_catchall(path: str) -> HTMLResponse:
        if path.startswith(("mesh/", "dashboard/", "ws/", "channels/")):
            raise HTTPException(status_code=404, detail="Not found")
        from src.shared.models import KEYLESS_PROVIDERS, get_all_providers
        all_providers = get_all_providers()
        template = env.get_template("index.html")
        html = template.render(
            ws_path="/ws/events", api_base="/dashboard/api", v=ASSET_VERSION,
            providers=[p for p in all_providers if p["name"] not in KEYLESS_PROVIDERS],
            all_providers=all_providers,
        )
        return HTMLResponse(html, headers={
            "Cache-Control": "no-store",
            # M18: frame-ancestors 'self' + X-Frame-Options SAMEORIGIN guard
            # against clickjacking. MUST be 'self'/SAMEORIGIN, NOT 'none'/DENY:
            # the dashboard embeds the per-agent VNC viewer in a same-origin
            # iframe via /agent-vnc/, so 'none' would break the viewer.
            "X-Frame-Options": "SAMEORIGIN",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline'; "
                "connect-src 'self'; "
                "frame-src 'self'; "
                "frame-ancestors 'self'; "
                "object-src 'none'"
            ),
        })

    return catchall
