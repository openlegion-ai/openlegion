"""Operator agent tools -- agent edits (with undo), observations, agent/team management."""
from __future__ import annotations

import json
import os
import re as _re
from datetime import datetime, timezone

from src.agent.skills import skill
from src.shared.types import (
    HARD_EDIT_FIELDS as _HARD_EDIT_FIELDS,
)
from src.shared.types import (
    SOFT_EDIT_FIELDS as _SOFT_EDIT_FIELDS,
)
from src.shared.utils import setup_logging

logger = setup_logging("agent.builtins.operator_tools")

def _is_operator() -> bool:
    """Defence-in-depth: only the operator agent has ALLOWED_TOOLS set.

    Non-operator agents should never execute these tools even if they
    appear in the tool list via auto-discovery.  Evaluated at call time
    so env changes (and test overrides) are respected.
    """
    return os.environ.get("ALLOWED_TOOLS", "") != ""

# Permission ceiling: operator cannot grant permissions beyond these limits
_OPERATOR_PERMISSION_CEILING = {
    "can_use_browser": True,
    "can_spawn": False,       # Created agents can't spawn others
    "can_manage_cron": True,
    "can_use_wallet": False,  # Requires explicit user setup
    "blackboard_read": ["*"],
    "blackboard_write": ["tasks/*", "context/*", "status/*", "output/*", "artifacts/*"],
}

_VALID_FIELDS = frozenset({
    "instructions", "soul", "model", "role", "heartbeat",
    "heartbeat_schedule",
    "interface", "thinking", "budget", "permissions",
})

# Heartbeat schedule validator. Accepts the same forms cron.py accepts:
#  * 5-field cron expressions ("*/15 * * * *", "0 9 * * 1-5")
#  * "every Ns / Nm / Nh / Nd" interval shorthand
# 6-field (seconds) cron is explicitly rejected here because cron.py
# rejects it too — keep the error consistent at both layers.
_HEARTBEAT_INTERVAL_RE = _re.compile(r"^every\s+\d+[smhd]$", _re.IGNORECASE)
_HEARTBEAT_CRON_FIELD_RE = _re.compile(r"^[\d,\-\*/]+$")


def _validate_heartbeat_schedule(value) -> str | None:
    """Return an error string when ``value`` is not a valid schedule, else ``None``.

    Keeps the surface narrow — only the two forms cron.py honours
    (5-field cron + ``every N[smhd]``). Free-form named intervals
    like ``hourly``/``daily`` are intentionally NOT accepted because
    the cron scheduler's ``_is_due`` path doesn't recognise them.
    """
    if not isinstance(value, str) or not value.strip():
        return "heartbeat_schedule must be a non-empty string"
    sched = value.strip()
    if _HEARTBEAT_INTERVAL_RE.match(sched):
        return None
    parts = sched.split()
    if len(parts) == 6:
        return (
            "6-field (seconds) cron is not supported. Use 5-field cron "
            "(minute resolution) or 'every Ns'. Example: '*/15 * * * *' "
            "or 'every 15m'."
        )
    if len(parts) == 5 and all(_HEARTBEAT_CRON_FIELD_RE.match(p) for p in parts):
        return None
    return (
        f"Invalid schedule: {sched!r}. Use 5-field cron "
        "('*/15 * * * *') or 'every N[smhd]' ('every 15m')."
    )

# All edits apply immediately via ``/edit-soft`` + emit an undo receipt.
# Hard fields (model / permissions / budget / thinking) earn a 30-min
# Undo window; soft fields get 5 min. There is no propose+confirm gate
# — that flow was retired in PR #927. The canonical field sets are
# imported from :mod:`src.shared.types` (single source of truth across
# host + agent modules).

# Audited reasons the operator can declare. ``user_asked`` is the common
# path (the user said "do X"); ``operator_proactive`` is the "I noticed"
# case which still skips the gate for soft edits but logs differently.
_VALID_EDIT_REASONS = frozenset({"user_asked", "operator_proactive"})

_OPERATOR_AGENT_ID = "operator"


def _validate_edit(agent_id: str, field: str, value) -> dict | None:
    """Shared validation for edit_agent. Returns error dict or None.

    Centralises the common gates (self-modification block, valid field,
    permission ceiling, budget bounds, thinking enum) so edit_agent (and
    any future edit-flow callers) produce identical error messages for
    the same misuse. Returns ``None`` when the call is safe to forward
    to the mesh.
    """
    if agent_id.lower() == _OPERATOR_AGENT_ID:
        return {
            "error": (
                "Cannot modify the operator agent. "
                "Use the dashboard to change operator settings."
            ),
        }
    if field not in _VALID_FIELDS:
        return {
            "error": (
                f"Invalid field '{field}'. "
                f"Must be one of: {sorted(_VALID_FIELDS)}"
            ),
        }
    if field == "permissions" and isinstance(value, dict):
        for key, max_val in _OPERATOR_PERMISSION_CEILING.items():
            if key not in value:
                continue
            if isinstance(max_val, bool):
                if value[key] and not max_val:
                    return {
                        "error": (
                            f"Permission ceiling exceeded: '{key}' cannot be set "
                            "to True by the operator. Use the dashboard for "
                            "advanced permissions."
                        ),
                    }
            elif isinstance(max_val, list):
                requested = set(value.get(key, []))
                allowed = set(max_val)
                if "*" not in allowed and not requested.issubset(allowed):
                    excess = requested - allowed
                    return {
                        "error": (
                            f"Permission ceiling exceeded: '{key}' patterns "
                            f"{excess} exceed allowed {allowed}. Use the "
                            "dashboard for advanced permissions."
                        ),
                    }
    if field == "budget" and isinstance(value, dict):
        daily = value.get("daily_usd", 0)
        monthly = value.get("monthly_usd", 0)
        if not isinstance(daily, (int, float)) or not (0.01 <= daily <= 1000):
            return {"error": f"daily_usd must be 0.01-1000, got {daily}"}
        if not isinstance(monthly, (int, float)) or not (0.10 <= monthly <= 30000):
            return {"error": f"monthly_usd must be 0.10-30000, got {monthly}"}
    if field == "thinking" and value not in ("off", "low", "medium", "high"):
        return {
            "error": (
                f"thinking must be 'off', 'low', 'medium', or 'high', "
                f"got '{value}'"
            ),
        }
    if field == "heartbeat_schedule":
        err = _validate_heartbeat_schedule(value)
        if err:
            return {"error": err}
    return None


@skill(
    name="confirm_edit",
    description=(
        "DEPRECATED — no-op. Config edits now apply immediately via "
        "edit_agent and emit an undo receipt, so there is no pending change "
        "to confirm. Tool retained only so in-flight conversations don't "
        "break; do not call from new code."
    ),
    parameters={
        "change_id": {
            "type": "string",
            "description": "Legacy change_id (ignored)",
        },
    },
)
async def confirm_edit(change_id: str, *, mesh_client=None, _messages=None, **_kw) -> dict:
    """Deprecated no-op. Returns a friendly hint to use edit_agent."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    return {
        "success": True,
        "applied": False,
        "deprecation_notice": (
            "confirm_edit is a no-op. Config edits now apply immediately "
            "when you call edit_agent; the user sees an undo receipt with a "
            "5–30 minute revert window. Don't call confirm_edit anymore."
        ),
    }


@skill(
    name="read_agent_config",
    description=(
        "Read an agent's current configuration. Symmetric inverse of "
        "edit_agent — returns the same fields edit_agent can change so you "
        "can review current values before/after a tweak. Use this BEFORE "
        "edit_agent whenever you need to know what's there now (e.g. "
        "before appending to instructions, before adjusting a budget, "
        "before changing a heartbeat schedule).\n\n"
        "Returns ``{agent_id, config: {...}}`` with these fields: "
        "model, instructions, soul, heartbeat, heartbeat_schedule, "
        "interface, role, permissions, budget, thinking. Pass "
        "``fields=['instructions','soul']`` to scope the read to a subset."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Target agent ID (use list_agents to find IDs)",
        },
        "fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional list of field names to return. Valid values: "
                "model, instructions, soul, heartbeat, heartbeat_schedule, "
                "interface, role, permissions, budget, thinking. "
                "Omit or pass empty for the full config."
            ),
            "default": [],
        },
    },
)
async def read_agent_config(
    agent_id: str,
    fields: list[str] | None = None,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Read an agent's config — symmetric inverse of edit_agent.

    Validates ``fields`` against the union of HARD/SOFT_EDIT_FIELDS
    upfront so the LLM gets a clear error before the mesh round-trip.
    On unknown field names, returns ``{"error": "unknown_fields", ...}``
    listing the bad names and the valid set. On 404 from the mesh,
    returns ``{"error": "agent_not_found"}``.
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    valid = _HARD_EDIT_FIELDS | _SOFT_EDIT_FIELDS
    if fields:
        # Normalize before validating — mesh-side parsing strips whitespace,
        # so accepting ["instructions "] (trailing space) keeps the read↔write
        # symmetry consistent (edit_agent doesn't reject on whitespace either).
        fields = [f.strip() for f in fields if isinstance(f, str) and f.strip()]
        unknown = [f for f in fields if f not in valid]
        if unknown:
            return {
                "error": "unknown_fields",
                "unknown": unknown,
                "valid": sorted(valid),
            }

    try:
        result = await mesh_client.get_agent_config(agent_id, fields=fields)
    except Exception as e:
        # Detect httpx HTTPStatusError without importing httpx at module
        # load (operator_tools is imported in test paths that may stub
        # networking). Duck-type on .response.status_code.
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None)
        if status == 404:
            return {"error": "agent_not_found", "agent_id": agent_id}
        if status is not None:
            body = getattr(resp, "text", "") or ""
            return {
                "error": "mesh_error",
                "status": status,
                "body": body[:500],
            }
        return {"error": f"Failed to read agent config: {e}"}
    return result


@skill(
    name="list_peer_artifacts",
    description=(
        "List a teammate's artifact files. Artifacts are files agents "
        "write via save_artifact — design docs, reports, generated "
        "assets, datasets — and live on each agent's private /data "
        "volume. Use this to see WHAT a teammate has produced before "
        "pulling content with read_peer_artifact.\n\n"
        "Returns ``{agent_id, artifacts: [{name, size, modified}, ...]}``. "
        "Operator-only: only the operator agent can read peer "
        "artifacts (peer-to-peer reads are intentionally not exposed)."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Target agent ID (use list_agents to find IDs)",
        },
    },
)
async def list_peer_artifacts(
    agent_id: str,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """List a peer agent's artifacts — operator-only read path.

    Mirrors :func:`read_agent_config` in shape and error handling: 404
    from the mesh becomes ``{"error": "agent_not_found"}``; other HTTP
    failures surface as ``mesh_error``. Exceptions are truncated to
    200 chars so a noisy traceback can't blow up the LLM's context.
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    try:
        result = await mesh_client.list_peer_artifacts(agent_id)
    except Exception as e:
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None)
        if status == 404:
            return {"error": "agent_not_found", "agent_id": agent_id}
        if status is not None:
            body = getattr(resp, "text", "") or ""
            return {
                "error": "mesh_error",
                "status": status,
                "body": body[:200],
            }
        return {"error": f"Failed to list peer artifacts: {str(e)[:200]}"}
    return result


@skill(
    name="read_peer_artifact",
    description=(
        "Read a single artifact file written by a teammate. Use this "
        "after list_peer_artifacts to pull the content of a specific "
        "file (e.g. to review a draft, verify a deliverable, or quote "
        "an agent's output back to the user).\n\n"
        "Returns ``{agent_id, name, content, size, encoding}``. Text "
        "artifacts come back as UTF-8 strings; binary artifacts come "
        "back base64-encoded with ``encoding='base64'``. Capped at "
        "5 MB — larger artifacts return an oversize error. "
        "Operator-only."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Target agent ID (use list_agents to find IDs)",
        },
        "name": {
            "type": "string",
            "description": (
                "Artifact file name as returned by list_peer_artifacts "
                "(e.g. 'design.md', 'reports/q3.pdf'). Path traversal "
                "and absolute paths are rejected."
            ),
        },
    },
)
async def read_peer_artifact(
    agent_id: str,
    name: str,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Read one artifact written by a teammate — operator-only.

    Mirrors :func:`read_agent_config` in shape. 404 from the mesh
    becomes ``{"error": "artifact_not_found"}`` (we can't cleanly
    distinguish agent-missing vs file-missing from the transport
    layer, so the message names both possibilities). 413 surfaces
    as ``oversize``. Exception strings are truncated to 200 chars.
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if not name or not isinstance(name, str):
        return {"error": "name is required"}

    try:
        result = await mesh_client.read_peer_artifact(agent_id, name)
    except Exception as e:
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None)
        if status == 404:
            return {
                "error": "artifact_not_found",
                "agent_id": agent_id,
                "name": name,
            }
        if status == 413:
            return {
                "error": "oversize",
                "agent_id": agent_id,
                "name": name,
            }
        if status == 400:
            return {
                "error": "invalid_name",
                "agent_id": agent_id,
                "name": name,
            }
        if status is not None:
            body = getattr(resp, "text", "") or ""
            return {
                "error": "mesh_error",
                "status": status,
                "body": body[:200],
            }
        return {"error": f"Failed to read peer artifact: {str(e)[:200]}"}
    return result


@skill(
    name="list_available_models",
    description=(
        "List models currently usable given the active credential setup. "
        "Use this BEFORE edit_agent or create_agent — never memorize the "
        "OAuth model subsets, the system tracks them for you. Returns "
        "per-provider lists and the credential kind (api_key / oauth / "
        "both) so you know whether a model switch needs a new credential."
    ),
    parameters={
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "description": (
                    "Optional filter (e.g., 'openai', 'anthropic'). "
                    "Omit to list all configured providers."
                ),
            },
        },
        "required": [],
    },
)
async def list_available_models(
    provider: str | None = None, *, mesh_client=None, **_kw,
) -> dict:
    """Operator-only — return per-provider allowed-model lists.

    Sourced from the mesh ``/mesh/introspect?section=llm`` payload so the
    operator and runtime stay aligned. When ``provider`` is supplied, the
    response narrows to that one provider.
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "mesh_client is required"}
    try:
        result = await mesh_client.introspect(section="llm")
    except Exception as e:
        return {"error": f"Failed to query mesh introspect: {str(e)[:200]}"}
    llm = result.get("llm", result)
    allowed = llm.get("allowed_models", {}) or {}
    kinds = llm.get("credential_kinds", {}) or {}
    if provider:
        return {
            "provider": provider,
            "credential_kind": kinds.get(provider, "none"),
            "allowed_models": allowed.get(provider, []),
        }
    return {
        "providers": {
            p: {
                "credential_kind": kinds.get(p, "none"),
                "allowed_models": allowed.get(p, []),
            }
            for p in allowed
        },
    }


@skill(
    name="edit_agent",
    description=(
        "Change an agent's configuration. All edits apply IMMEDIATELY and "
        "emit a receipt card with [View diff] [Undo]. No confirmation step "
        "— act decisively on what the user asked for. The undo window is "
        "5 minutes for soft fields (instructions/soul/role/heartbeat/"
        "heartbeat_schedule/interface) and 30 minutes for hard fields "
        "(model/permissions/budget/thinking) so the user has more time to "
        "catch a costly edit.\n\n"
        "Always pass `reason` so the audit trail captures intent.\n\n"
        "Fields & value formats:\n"
        "- instructions/soul/heartbeat/interface/role: string\n"
        "- heartbeat_schedule: 5-field cron ('*/15 * * * *') OR "
        "'every N[smhd]' ('every 15m', 'every 2h')\n"
        "- budget: {\"daily_usd\": float, \"monthly_usd\": float}\n"
        "- permissions: {\"can_use_browser\": bool, ...}\n"
        "- thinking: \"off\" | \"low\" | \"medium\" | \"high\"\n"
        "- model: e.g. \"anthropic/claude-sonnet-4-20250514\""
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Target agent ID (use list_agents to find IDs)",
        },
        "field": {
            "type": "string",
            "description": "Config field to change",
            "enum": [
                "instructions", "soul", "model", "role", "heartbeat",
                "heartbeat_schedule",
                "interface", "thinking", "budget", "permissions",
            ],
        },
        "value": {
            "type": ["string", "object"],
            "description": "New value for the field",
        },
        "reason": {
            "type": "string",
            "description": (
                "Why you're making this change. 'user_asked' when the user "
                "directly requested it; 'operator_proactive' when you "
                "noticed an opportunity yourself."
            ),
            "enum": ["user_asked", "operator_proactive"],
            "default": "user_asked",
        },
    },
)
async def edit_agent(
    agent_id: str,
    field: str,
    value,
    reason: str = "user_asked",
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Change agent config — apply immediately, emit an undo receipt.

    All fields go through the same path. The mesh records an undo
    receipt with a TTL of 5 min for soft fields or 30 min for hard
    fields; the dashboard renders [View diff] [Undo] on the receipt
    card. Provenance is intentionally NOT required at this layer — the
    receipt is the safety net, and dropping the gate lets the operator
    self-tune during heartbeat without a human in the loop. The
    permission ceiling in :data:`_OPERATOR_PERMISSION_CEILING` still
    blocks irreversible grants (``can_spawn``, ``can_use_wallet``).
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if reason not in _VALID_EDIT_REASONS:
        return {
            "error": (
                f"reason must be one of {sorted(_VALID_EDIT_REASONS)}, "
                f"got {reason!r}"
            ),
        }

    err = _validate_edit(agent_id, field, value)
    if err is not None:
        return err

    if reason == "operator_proactive":
        logger.info(
            "operator_proactive edit: agent=%s field=%s",
            agent_id, field,
        )
    try:
        result = await mesh_client.edit_soft(agent_id, field, value, reason)
    except Exception as e:
        return {"error": f"Failed to apply edit: {e}"}

    field_class = result.get("field_class") or (
        "hard" if field not in _SOFT_EDIT_FIELDS else "soft"
    )
    ttl_seconds = result.get("ttl_seconds") or (1800 if field_class == "hard" else 300)
    minutes = ttl_seconds // 60
    return {
        "success": True,
        "applied": True,
        "agent_id": agent_id,
        "field": field,
        "field_class": field_class,
        "undo_token": result.get("undo_token"),
        "expires_at": result.get("expires_at"),
        "ttl_seconds": ttl_seconds,
        "summary": result.get("summary"),
        "message": (
            f"Done. The user sees a receipt card and can Undo within "
            f"{minutes} minute{'s' if minutes != 1 else ''}."
        ),
    }


@skill(
    name="undo_change",
    description=(
        "Reverse a recent soft edit by undo_token. Use this if you realize "
        "mid-conversation that an edit you applied was wrong, OR if the user "
        "asks you to undo a change. 5-minute window from when the edit was "
        "made; 404 if expired or already undone."
    ),
    parameters={
        "undo_token": {
            "type": "string",
            "description": "The undo_token from a prior edit_agent call",
        },
    },
)
async def undo_change(
    undo_token: str,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Reverse a recent soft edit. Single-shot; double-undo returns 404."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if not undo_token or not isinstance(undo_token, str):
        return {"error": "undo_token is required"}
    try:
        result = await mesh_client.undo_change(undo_token)
    except Exception as e:
        msg = str(e)
        if "404" in msg or "not found" in msg.lower() or "expired" in msg.lower():
            return {
                "error": "undo_unavailable",
                "detail": (
                    "Undo token unknown, expired (5min window passed), or "
                    "already used."
                ),
            }
        return {"error": f"Failed to undo change: {e}"}
    return {
        "success": True,
        "agent_id": result.get("agent_id"),
        "field": result.get("field"),
        "restored_value": result.get("restored_value"),
    }


# ── Observations ─────────────────────────────────────────────


_MAX_OBSERVATIONS_CHARS = 1500
_MAX_HISTORY_ENTRIES = 50


@skill(
    name="save_observations",
    description=(
        "Save fleet health observations from your monitoring check. "
        "Writes structured data to OBSERVATIONS.md for the Fleet Digest display."
    ),
    parameters={
        "fleet_summary": {
            "type": "string",
            "description": "One-line fleet health summary (e.g. '5/6 healthy, cost stable')",
        },
        "agents_attention": {
            "type": "array",
            "description": "Agents needing attention: [{agent_id, issue, severity}]",
            "items": {"type": "object"},
            "default": [],
        },
        "cost_trend": {
            "type": "string",
            "description": "Cost trend (e.g. 'up_40pct', 'stable', 'down_15pct')",
        },
        "notes": {
            "type": "string",
            "description": "Optional freeform notes",
            "default": "",
        },
    },
)
async def save_observations(
    fleet_summary: str,
    cost_trend: str,
    agents_attention: list | None = None,
    notes: str = "",
    *,
    workspace_manager=None,
    **_kw,
) -> dict:
    """Save fleet observations to workspace for dashboard display."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if workspace_manager is None:
        return {"error": "No workspace_manager available"}

    timestamp = datetime.now(timezone.utc).isoformat()

    obs = {
        "timestamp": timestamp,
        "fleet_summary": fleet_summary,
        "agents_attention": agents_attention or [],
        "cost_trend": cost_trend,
        "notes": notes,
    }

    # Build markdown with JSON block
    content = (
        f"# Fleet Observations\nUpdated: {timestamp}\n\n"
        f"```json\n{json.dumps(obs, indent=2)}\n```\n"
    )

    # Enforce char cap by truncating notes
    while len(content) > _MAX_OBSERVATIONS_CHARS and notes:
        notes = notes[:-50] + "..." if len(notes) > 50 else ""
        obs["notes"] = notes
        content = (
            f"# Fleet Observations\nUpdated: {timestamp}\n\n"
            f"```json\n{json.dumps(obs, indent=2)}\n```\n"
        )

    # Write OBSERVATIONS.md directly to workspace root (not in AGENT_WRITABLE)
    obs_path = workspace_manager.root / "OBSERVATIONS.md"
    obs_path.write_text(content)

    # Append to OBSERVATIONS_HISTORY.md (rolling window)
    history_path = workspace_manager.root / "OBSERVATIONS_HISTORY.md"
    history_content = ""
    if history_path.exists():
        try:
            history_content = history_path.read_text(errors="replace")
        except OSError:
            pass
    history_lines = [e for e in history_content.strip().split("\n---\n") if e.strip()]
    history_lines.append(json.dumps(obs))
    if len(history_lines) > _MAX_HISTORY_ENTRIES:
        history_lines = history_lines[-_MAX_HISTORY_ENTRIES:]
    history_path.write_text("\n---\n".join(history_lines) + "\n")

    return {"saved": True, "timestamp": timestamp, "chars": len(content)}


# ── Create Agent ─────────────────────────────────────────────


@skill(
    name="create_agent",
    description=(
        "Create a new custom agent with role/model/instructions. "
        "Requires user confirmation."
    ),
    parameters={
        "name": {
            "type": "string",
            "description": "Agent ID (lowercase, alphanumeric + hyphens, 1-32 chars)",
        },
        "role": {
            "type": "string",
            "description": "Human-readable role description",
        },
        "model": {
            "type": "string",
            "description": "LLM model (optional, defaults to system default)",
            "default": "",
        },
        "instructions": {
            "type": "string",
            "description": "Initial instructions for the agent",
        },
        "soul": {
            "type": "string",
            "description": "Optional personality/identity",
            "default": "",
        },
    },
)
async def create_agent(
    name: str,
    role: str,
    instructions: str,
    model: str = "",
    soul: str = "",
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Create a new custom agent.

    Provenance gate dropped: operator can spawn agents autonomously
    (e.g. during heartbeat in response to its own analysis). The
    plan-tier budget cap ``OPENLEGION_MAX_AGENTS`` and the permission
    ceiling (``can_spawn=False`` for created agents) are the real walls.
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    try:
        return await mesh_client.create_custom_agent(
            name, role, model, instructions, soul,
        )
    except Exception as e:
        error_str = str(e)
        if "409" in error_str:
            return {"error": f"Agent '{name}' already exists or plan limit reached."}
        return {"error": f"Failed to create agent: {e}"}


# ── Project Management ───────────────────────────────────────


# ── Legacy project_* tools — PR 3 sunset stubs ────────────────────
#
# Each ``*_project`` skill below is now a thin redirector: the
# ``@skill`` registration stays so the LLM still discovers the tool
# name and the JSON-schema-validated call doesn't raise, but the
# body returns ``{"error": "renamed", "new_tool": "...", ...}`` so a
# stale prompt fails fast and recoverably instead of executing
# duplicate logic. Real behavior lives on the canonical ``*_team``
# tools defined later in this module.
#
# The parameter schemas are preserved verbatim so a tool call that
# was JSON-schema-valid pre-rename remains valid here — the LLM gets
# the error in the tool result, not as a schema-validation HTTP 4xx
# upstream.


def _renamed_stub(new_tool: str) -> dict:
    """Standard sunset-stub payload."""
    return {
        "error": "renamed",
        "new_tool": new_tool,
        "note": (
            f"Use {new_tool} instead. This deprecated alias was removed "
            "in PR 3 of the project→team rename."
        ),
    }


@skill(
    name="inspect_projects",
    description="[REMOVED — call inspect_teams]",
    parameters={
        "detail": {
            "type": "string",
            "description": "names | status | full",
            "enum": ["names", "status", "full"],
            "default": "names",
        },
        "project_name": {
            "type": "string",
            "description": "Optional — return full detail for this project only",
            "default": "",
        },
    },
)
async def inspect_projects(*_args, **_kw) -> dict:
    """Sunset stub — redirects to ``inspect_teams``."""
    return _renamed_stub("inspect_teams")


@skill(
    name="create_project",
    description="[REMOVED — call create_team]",
    parameters={
        "name": {
            "type": "string",
            "description": "Project name",
        },
        "description": {
            "type": "string",
            "description": "Project brief / description",
        },
        "agent_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agent IDs to assign to the project",
            "default": [],
        },
    },
)
async def create_project(*_args, **_kw) -> dict:
    """Sunset stub — redirects to ``create_team``."""
    return _renamed_stub("create_team")


@skill(
    name="add_agents_to_project",
    description="[REMOVED — call add_agents_to_team]",
    parameters={
        "project_name": {
            "type": "string",
            "description": "Project name",
        },
        "agent_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agent IDs to add",
        },
    },
)
async def add_agents_to_project(*_args, **_kw) -> dict:
    """Sunset stub — redirects to ``add_agents_to_team``."""
    return _renamed_stub("add_agents_to_team")


@skill(
    name="remove_agents_from_project",
    description="[REMOVED — call remove_agents_from_team]",
    parameters={
        "project_name": {
            "type": "string",
            "description": "Project name",
        },
        "agent_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agent IDs to remove",
        },
    },
)
async def remove_agents_from_project(*_args, **_kw) -> dict:
    """Sunset stub — redirects to ``remove_agents_from_team``."""
    return _renamed_stub("remove_agents_from_team")


@skill(
    name="update_project_context",
    description="[REMOVED — call update_team_context]",
    parameters={
        "project_name": {
            "type": "string",
            "description": "Project name",
        },
        "context": {
            "type": "string",
            "description": "New project description / context text",
        },
    },
)
async def update_project_context(*_args, **_kw) -> dict:
    """Sunset stub — redirects to ``update_team_context``."""
    return _renamed_stub("update_team_context")


@skill(
    name="set_project_goal",
    description="[REMOVED — call set_team_goal]",
    parameters={
        "project_name": {
            "type": "string",
            "description": "Project to set the goal on",
        },
        "north_star": {
            "type": "string",
            "description": "Free-text vision statement, ≤2000 characters.",
        },
        "success_criteria": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Up to 10 measurable outcomes, each ≤200 characters.",
        },
    },
)
async def set_project_goal(*_args, **_kw) -> dict:
    """Sunset stub — redirects to ``set_team_goal``."""
    return _renamed_stub("set_team_goal")


# ── Task 7: Operator product tools ───────────────────────────


_TASKS_V2_DISABLED = (
    "Orchestration tasks not enabled — flip OPENLEGION_ORCHESTRATION_TASKS_V2=1"
)


def _orchestration_v2_on() -> bool:
    """Read the orchestration v2 flag at call time so monkeypatch tests work.

    Default-on (rollout). Setting the env var to ``0`` disables the v2
    path; any other value is treated as on.
    """
    return os.environ.get("OPENLEGION_ORCHESTRATION_TASKS_V2", "1") != "0"


def _parse_over_budget(error: Exception) -> dict | None:
    """If a mesh HTTP error wraps an over_budget JSON payload, surface it.

    The reroute / retry endpoints encode a structured budget error in the
    400 body. ``httpx.HTTPStatusError`` stringifies as something like
    ``"Client error '400 Bad Request' for url ... \\nFor more ..."``;
    the JSON body is on ``error.response`` when the client wraps it.
    Returns the structured dict or None if the error isn't a budget one.
    """
    response = getattr(error, "response", None)
    if response is None:
        return None
    try:
        body = response.json()
    except Exception:
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str):
        try:
            parsed = json.loads(detail)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("error") == "over_budget":
            return parsed
    if isinstance(detail, dict) and detail.get("error") == "over_budget":
        return detail
    return None


# ── Read tools ──────────────────────────────────────────────


@skill(
    name="list_agent_queue",
    description=(
        "Read an agent's task queue: current and recent tasks grouped by "
        "status (active / blocked / done / failed / cancelled), up to "
        "`limit` rows per bucket."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Agent ID to inspect",
        },
        "limit": {
            "type": "integer",
            "description": "Max rows per status bucket (default 10, max 100)",
            "default": 10,
        },
    },
)
async def list_agent_queue(
    agent_id: str,
    limit: int = 10,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Per-agent task queue grouped by status."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if not _orchestration_v2_on():
        return {"error": _TASKS_V2_DISABLED}
    try:
        return await mesh_client.agent_queue(agent_id, limit=limit)
    except Exception as e:
        return {"error": f"Failed to read queue for {agent_id}: {e}"}


@skill(
    name="get_team_outputs",
    description=(
        "Completed task artifacts for a project in a time window. "
        "`since` accepts ISO timestamps or duration strings ('24h', '7d'); "
        "default is the last 7 days."
    ),
    parameters={
        "project_id": {
            "type": "string",
            "description": "Project ID",
        },
        "since": {
            "type": "string",
            "description": "ISO timestamp or duration string (e.g. '24h', '7d')",
            "default": "",
        },
    },
)
async def get_team_outputs(
    project_id: str,
    since: str = "",
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Completed task artifacts for a project."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if not _orchestration_v2_on():
        return {"error": _TASKS_V2_DISABLED}
    try:
        return await mesh_client.project_outputs(project_id, since=since)
    except Exception as e:
        return {"error": f"Failed to read outputs for {project_id}: {e}"}


@skill(
    name="summarize_project_progress",
    description="[REMOVED — call summarize_team_progress]",
    parameters={
        "project_id": {
            "type": "string",
            "description": "Project ID",
        },
    },
)
async def summarize_project_progress(*_args, **_kw) -> dict:
    """Sunset stub — redirects to ``summarize_team_progress``."""
    return _renamed_stub("summarize_team_progress")


@skill(
    name="inspect_agents",
    description=(
        "Read agents. Without agent_id: roster summary. With agent_id: "
        "depth='profile' returns role/capabilities/INTERFACE; "
        "depth='history' adds recent activity log. depth defaults to summary. "
        "Pass stale_threshold_hours=N to also annotate each agent in the "
        "roster with its stale-task count and up-to-5 oldest stale task IDs."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Optional — target agent for profile/history",
            "default": "",
        },
        "depth": {
            "type": "string",
            "description": "summary | profile | history",
            "enum": ["summary", "profile", "history"],
            "default": "summary",
        },
        "stale_threshold_hours": {
            "type": "integer",
            "description": (
                "When set, roster entries gain stale_task_count + "
                "stale_task_ids (top 5 oldest). Counts non-terminal "
                "tasks created more than N hours ago. Range 1-168."
            ),
            "default": 0,
        },
    },
)
async def inspect_agents(
    agent_id: str = "",
    depth: str = "summary",
    stale_threshold_hours: int | None = None,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Consolidated agent read tool (replaces operator's use of list_agents
    / get_agent_profile / read_agent_history).
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    # ``0`` is the JSON-schema default for the optional integer param;
    # treat 0 the same as None (param not supplied) so the LLM can omit
    # it without triggering a stale-task lookup.
    threshold_h = stale_threshold_hours or None
    if threshold_h is not None and not (1 <= threshold_h <= 168):
        return {
            "error": (
                "stale_threshold_hours must be between 1 and 168 "
                "(1 hour to 7 days)"
            ),
        }

    # No agent_id → roster (always summary regardless of depth).
    if not agent_id:
        try:
            registry = await mesh_client.list_agents()
        except Exception as e:
            return {"error": f"Failed to list agents: {e}"}

        # Pre-filter: when ``threshold_h`` is set, fetch ``system_metrics``
        # once and use ``stale_tasks_24h_count`` to identify which agents
        # actually have non-zero stale work. Skip per-agent fanout for
        # agents whose count is known-zero. If the system_metrics fetch
        # fails for any reason, fall through to the full fanout
        # (defensive — preserves prior behavior).
        stale_counts: dict[str, int] = {}
        have_counts = False
        if threshold_h is not None:
            try:
                _metrics = await mesh_client.get_system_metrics()
                stale_counts = _metrics.get("stale_tasks_24h_count", {}) or {}
                have_counts = True
            except Exception as e:
                logger.debug(
                    "inspect_agents stale prefilter failed: %s — "
                    "falling back to full fanout", e,
                )

        agents = []
        for name, info in registry.items():
            entry: dict = {"name": name}
            if isinstance(info, dict):
                entry["role"] = info.get("role", "")
                entry["capabilities"] = info.get("capabilities", [])
            if threshold_h is not None:
                # Operator is excluded from stale-task fanout: it's a
                # system agent with no user-facing inbox, so its stale
                # count is always zero. Still attach the empty fields
                # so the response shape is consistent across agents.
                if name == "operator":
                    entry["stale_task_count"] = 0
                    entry["stale_task_ids"] = []
                    agents.append(entry)
                    continue
                # Skip per-agent fanout when the prefilter says this
                # agent has zero stale tasks. Still attach the empty
                # fields so the response shape is consistent.
                if have_counts and int(stale_counts.get(name, 0) or 0) == 0:
                    entry["stale_task_count"] = 0
                    entry["stale_task_ids"] = []
                    agents.append(entry)
                    continue
                # Per-agent stale lookup. Failures degrade to count=0
                # so a single agent's mesh hiccup doesn't poison the
                # whole roster response.
                try:
                    stale = await mesh_client.get_agent_stale_tasks(
                        name, threshold_hours=threshold_h,
                    )
                    entry["stale_task_count"] = int(stale.get("count", 0) or 0)
                    entry["stale_task_ids"] = list(stale.get("task_ids", []))
                except Exception as e:
                    logger.debug(
                        "inspect_agents stale lookup failed for %s: %s", name, e,
                    )
                    entry["stale_task_count"] = 0
                    entry["stale_task_ids"] = []
            agents.append(entry)
        result: dict = {"agents": agents, "count": len(agents)}
        if threshold_h is not None:
            result["stale_threshold_hours"] = threshold_h
        return result

    if depth == "history":
        try:
            return await mesh_client.get_agent_history(agent_id)
        except Exception as e:
            return {"error": f"Failed to read history for {agent_id}: {e}"}

    # depth == "profile" or "summary" with an agent_id → profile call
    try:
        return await mesh_client.get_agent_profile(agent_id)
    except Exception as e:
        return {"error": f"Failed to read profile for {agent_id}: {e}"}


# ── Action tools ────────────────────────────────────────────


@skill(
    name="manage_task",
    description=(
        "Cancel, reroute, or retry a task. action='cancel' stops the task; "
        "action='reroute' moves it to new_assignee (required); "
        "action='retry' clones a failed task (optionally overriding "
        "assignee/title/description via with_changes). Reroute and retry "
        "refuse if the target agent is over budget."
    ),
    parameters={
        "task_id": {
            "type": "string",
            "description": "Task ID",
        },
        "action": {
            "type": "string",
            "description": "cancel | reroute | retry",
            "enum": ["cancel", "reroute", "retry"],
        },
        "new_assignee": {
            "type": "string",
            "description": "Required for reroute; optional override for retry",
            "default": "",
        },
        "reason": {
            "type": "string",
            "description": "Optional reason recorded on the audit trail",
            "default": "",
        },
        "with_changes": {
            "type": "object",
            "description": (
                "retry only: optional patch with 'title', 'description', "
                "'assignee' overrides"
            ),
            "default": {},
        },
    },
)
async def manage_task(
    task_id: str,
    action: str,
    new_assignee: str = "",
    reason: str = "",
    with_changes: dict | None = None,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Consolidated task action tool (replaces cancel_task / reroute_task /
    retry_failed_task).
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if not _orchestration_v2_on():
        return {"error": _TASKS_V2_DISABLED}

    if action == "cancel":
        try:
            return await mesh_client.cancel_task(task_id, reason=reason)
        except Exception as e:
            return {"error": f"Failed to cancel task: {e}"}

    if action == "reroute":
        if not new_assignee:
            return {"error": "reroute requires new_assignee"}
        try:
            return await mesh_client.reroute_task(
                task_id, new_assignee, reason=reason,
            )
        except Exception as e:
            budget = _parse_over_budget(e)
            if budget is not None:
                return {
                    "error": "over_budget",
                    "detail": budget.get("detail") or (
                        f"Agent "
                        f"{budget.get('budget', {}).get('agent', new_assignee)!r} "
                        "is over budget."
                    ),
                    "budget": budget.get("budget"),
                }
            return {"error": f"Failed to reroute task: {e}"}

    if action == "retry":
        patch = dict(with_changes or {})
        # `new_assignee` is a convenience shortcut for retry overrides.
        if new_assignee and "assignee" not in patch:
            patch["assignee"] = new_assignee
        try:
            return await mesh_client.retry_task(
                task_id,
                title=patch.get("title"),
                description=patch.get("description"),
                assignee=patch.get("assignee"),
            )
        except Exception as e:
            budget = _parse_over_budget(e)
            if budget is not None:
                return {
                    "error": "over_budget",
                    "detail": budget.get("detail") or "Target agent is over budget.",
                    "budget": budget.get("budget"),
                }
            return {"error": f"Failed to retry task: {e}"}

    return {"error": f"Unknown action {action!r}; use cancel|reroute|retry"}


@skill(
    name="manage_project",
    description="[REMOVED — call manage_team]",
    parameters={
        "project_name": {
            "type": "string",
            "description": "Project name",
        },
        "action": {
            "type": "string",
            "description": "archive | delete",
            "enum": ["archive", "delete"],
        },
    },
)
async def manage_project(*_args, **_kw) -> dict:
    """Sunset stub — redirects to ``manage_team``."""
    return _renamed_stub("manage_team")


@skill(
    name="manage_agent",
    description=(
        "Archive or delete an agent. action='archive' is reversible and "
        "stops scheduling. action='delete' returns a confirmation nonce; "
        "the agent must already be archived."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "Agent ID",
        },
        "action": {
            "type": "string",
            "description": "archive | delete",
            "enum": ["archive", "delete"],
        },
    },
)
async def manage_agent(
    agent_id: str,
    action: str,
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Consolidated agent lifecycle tool (archive | delete).

    Archive is reversible and applies immediately. Delete still goes
    through a brief confirmation window (mesh-side propose_delete with
    a short TTL); a follow-up PR will convert delete to immediate-apply
    with a 72h undo. Operator can call either autonomously.
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if agent_id.lower() == _OPERATOR_AGENT_ID:
        return {"error": f"Cannot {action} the operator agent."}

    if action == "archive":
        try:
            return await mesh_client.archive_agent(agent_id)
        except Exception as e:
            return {"error": f"Failed to archive agent: {e}"}

    if action == "delete":
        try:
            result = await mesh_client.propose_delete_agent(agent_id)
            result.setdefault("requires_confirmation", True)
            return result
        except Exception as e:
            msg = str(e)
            if "must be archived" in msg.lower() or "400" in msg:
                return {
                    "error": "archive_required",
                    "detail": (
                        f"Agent {agent_id!r} must be archived first. "
                        "Call manage_agent(action='archive') first."
                    ),
                }
            return {"error": f"Failed to propose delete: {e}"}

    return {"error": f"Unknown action {action!r}; use archive|delete"}


# ── Self-cleanup tools ──────────────────────────────────────


@skill(
    name="list_pending",
    description=(
        "List every non-expired pending action awaiting user "
        "confirmation. Returns the nonce, action_kind, target_kind, "
        "target_id, expires_at, actor and summary for each row. Use "
        "this to find the nonce for cancel_pending_action() or to "
        "show the user what is currently waiting on them."
    ),
    parameters={},
)
async def list_pending(*, mesh_client=None, **_kw) -> dict:
    """Return open pending actions (operator-only)."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        result = await mesh_client.list_pending_actions()
    except Exception as e:
        return {"error": f"Failed to list pending actions: {e}"}
    pending = result.get("pending", []) if isinstance(result, dict) else []
    return {"pending": pending, "count": len(pending)}


@skill(
    name="cancel_pending_action",
    description=(
        "Cancel a pending action by nonce. Use this to clean up stale or "
        "incorrect proposals that the user no longer wants. The action "
        "will no longer be available for confirmation. Pair with "
        "list_pending() to find the nonce, or inspect the pending-actions "
        "card on the Board."
    ),
    parameters={
        "nonce": {
            "type": "string",
            "description": (
                "Nonce of the pending action to cancel (from list_pending() "
                "or the edit_agent return value, the Board pending list, "
                "or the pending_action_card surface)."
            ),
        },
    },
)
async def cancel_pending_action(
    nonce: str,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Cancel a stale pending action so the user no longer sees it."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if not nonce or not isinstance(nonce, str):
        return {"error": "nonce is required"}
    try:
        result = await mesh_client.cancel_pending_action(nonce)
    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if "404" in msg or "not found" in lower or "expired" in lower:
            return {
                "error": "pending_unknown_or_expired",
                "detail": (
                    "Pending action not found — it may have already been "
                    "confirmed, cancelled, or expired."
                ),
            }
        return {"error": f"Failed to cancel pending action: {e}"}
    return {
        "success": True,
        "nonce": result.get("nonce", nonce),
        "target_kind": result.get("target_kind"),
        "target_id": result.get("target_id"),
        "action_kind": result.get("action_kind"),
        "message": "Pending action cancelled — the user no longer sees it.",
    }


@skill(
    name="archive_audit_before",
    description=(
        "Archive operator audit entries older than the given date. Removes "
        "them from the active audit-log view but preserves them in the "
        "archived store (recoverable via include_archived=true). Use to "
        "keep the audit log focused on recent activity. Returns the count "
        "of rows archived."
    ),
    parameters={
        "before_date": {
            "type": "string",
            "description": (
                "ISO 8601 date or timestamp — entries strictly older than "
                "this date will be archived. Examples: '2026-04-01', "
                "'2026-04-01T00:00:00Z', '2026-04-01 00:00:00'."
            ),
        },
    },
)
async def archive_audit_before(
    before_date: str,
    *,
    mesh_client=None,
    **_kw,
) -> dict:
    """Bulk-archive old audit entries (soft-delete; preserves history)."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if not before_date or not isinstance(before_date, str):
        return {"error": "before_date is required (ISO 8601 string)"}
    try:
        result = await mesh_client.archive_audit_before(before_date)
    except Exception as e:
        return {"error": f"Failed to archive audit entries: {e}"}
    archived = int(result.get("archived_count", 0))
    truncated = bool(result.get("truncated", False))
    msg = (
        f"Archived {archived} audit "
        f"{'entry' if archived == 1 else 'entries'} older than "
        f"{before_date}."
    )
    if truncated:
        # Hard cap was hit — operator can re-run to keep sweeping.
        msg += " Hit per-call hard cap; rerun to continue archiving."
    return {
        "success": True,
        "archived_count": archived,
        "truncated": truncated,
        "before_date": result.get("before_date", before_date),
        "message": msg,
    }


# ── Team-named canonical aliases (PR 2 of the project→team rename) ────
#
# These register the new ``*_team`` tool names on top of the existing
# ``*_project`` skills. Each canonical tool accepts BOTH ``team_name``
# and ``project_name`` kwargs and forwards to the legacy implementation
# unchanged. The legacy tool names remain registered with their original
# descriptions so SDK consumers that grep for ``create_project`` still
# match — only the docstring on each legacy tool gets a soft deprecation
# nudge. PR 3 will retire the aliases.


@skill(
    name="inspect_teams",
    description=(
        "Read team info. detail='names' lists name+description; "
        "detail='status' adds task-count rollups (requires v2). "
        "Setting team_name returns full detail for that team."
    ),
    parameters={
        "detail": {
            "type": "string",
            "description": "names | status | full",
            "enum": ["names", "status", "full"],
            "default": "names",
        },
        "team_name": {
            "type": "string",
            "description": "Optional — return full detail for this team only",
            "default": "",
        },
    },
)
async def inspect_teams(
    detail: str = "names",
    team_name: str = "",
    *,
    project_name: str = "",
    mesh_client=None,
    **_kw,
) -> dict:
    """Consolidated team read tool (replaces list_teams / get_team / list_team_status)."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    target = team_name or project_name
    if target:
        try:
            result = await mesh_client.list_teams()
        except Exception as e:
            return {"error": f"Failed to inspect team: {e}"}
        for p in result.get("teams", result.get("projects", [])):
            if p.get("name") == target:
                return p
        return {"error": f"Team '{target}' not found"}

    if detail == "status":
        if not _orchestration_v2_on():
            return {"error": _TASKS_V2_DISABLED}
        try:
            return await mesh_client.all_teams_status()
        except Exception as e:
            return {"error": f"Failed to read team status: {e}"}

    try:
        return await mesh_client.list_teams()
    except Exception as e:
        return {"error": f"Failed to list teams: {e}"}


@skill(
    name="create_team",
    description=(
        "Create a new team and optionally assign agents to it. "
        "Requires user confirmation."
    ),
    parameters={
        "name": {"type": "string", "description": "Team name"},
        "description": {"type": "string", "description": "Team brief / description"},
        "agent_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agent IDs to assign to the team",
            "default": [],
        },
    },
)
async def create_team(
    name: str,
    description: str = "",
    agent_ids: list[str] | None = None,
    *,
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Create a new team. Provenance gate dropped — operator can spawn autonomously."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    try:
        return await mesh_client.create_team(
            name, description, agent_ids or [],
        )
    except Exception as e:
        return {"error": f"Failed to create team: {e}"}


@skill(
    name="add_agents_to_team",
    description="Add one or more agents to a team. Requires user confirmation.",
    parameters={
        "team_name": {"type": "string", "description": "Team name"},
        "agent_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agent IDs to add",
        },
    },
)
async def add_agents_to_team(
    team_name: str = "",
    agent_ids: list[str] | None = None,
    *,
    project_name: str = "",
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Add agents to a team."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    target = team_name or project_name
    results = []
    for aid in (agent_ids or []):
        try:
            r = await mesh_client.add_agent_to_team(target, aid)
            results.append(r)
        except Exception as e:
            results.append({"agent": aid, "error": str(e)})
    return {"team": target, "results": results}


@skill(
    name="remove_agents_from_team",
    description="Remove one or more agents from a team. Requires user confirmation.",
    parameters={
        "team_name": {"type": "string", "description": "Team name"},
        "agent_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agent IDs to remove",
        },
    },
)
async def remove_agents_from_team(
    team_name: str = "",
    agent_ids: list[str] | None = None,
    *,
    project_name: str = "",
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Remove agents from a team."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    target = team_name or project_name
    results = []
    for aid in (agent_ids or []):
        try:
            r = await mesh_client.remove_agent_from_team(target, aid)
            results.append(r)
        except Exception as e:
            results.append({"agent": aid, "error": str(e)})
    return {"team": target, "results": results}


@skill(
    name="update_team_context",
    description="Update a team's description / shared context.",
    parameters={
        "team_name": {"type": "string", "description": "Team name"},
        "context": {"type": "string", "description": "New context text"},
    },
)
async def update_team_context(
    team_name: str = "",
    context: str = "",
    *,
    project_name: str = "",
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Update team description/context."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    target = team_name or project_name
    try:
        return await mesh_client.update_team_context(target, context)
    except Exception as e:
        return {"error": f"Failed to update team context: {e}"}


@skill(
    name="set_team_goal",
    description="Set a team's north star + success criteria.",
    parameters={
        "team_name": {"type": "string", "description": "Team name"},
        "north_star": {
            "type": "string",
            "description": "Single-sentence north-star goal",
            "default": "",
        },
        "success_criteria": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of measurable success criteria",
            "default": [],
        },
    },
)
async def set_team_goal(
    team_name: str = "",
    north_star: str = "",
    success_criteria: list[str] | None = None,
    *,
    project_name: str = "",
    mesh_client=None,
    **_kw,
) -> dict:
    """Set the team's north_star + success_criteria. No gate — meta-config the user asked for."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    target = team_name or project_name
    if not isinstance(target, str) or not target.strip():
        return {"error": "team_name is required"}
    if not isinstance(north_star, str):
        return {"error": "north_star must be a string"}
    if len(north_star) > 2000:
        return {"error": "north_star must be 2000 characters or fewer"}

    cleaned_criteria: list[str] | None
    if success_criteria is None:
        cleaned_criteria = None
    else:
        if not isinstance(success_criteria, list):
            return {"error": "success_criteria must be a list of strings"}
        if len(success_criteria) > 10:
            return {"error": "success_criteria may contain at most 10 items"}
        cleaned_criteria = []
        for item in success_criteria:
            if not isinstance(item, str):
                return {"error": "each success_criteria entry must be a string"}
            if len(item) > 200:
                return {
                    "error": "each success_criteria entry must be 200 characters or fewer",
                }
            stripped = item.strip()
            if stripped:
                cleaned_criteria.append(stripped)
        if not cleaned_criteria:
            cleaned_criteria = None

    try:
        return await mesh_client.set_team_goal(
            target, north_star.strip() or None, cleaned_criteria,
        )
    except Exception as e:
        return {"error": f"Failed to set team goal: {e}"}


@skill(
    name="summarize_team_progress",
    description="Synthesised progress summary for a team.",
    parameters={
        "team_id": {"type": "string", "description": "Team id"},
    },
)
async def summarize_team_progress(
    team_id: str = "",
    *,
    project_id: str = "",
    mesh_client=None,
    **_kw,
) -> dict:
    """Synthesized progress summary for a team."""
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}
    if not _orchestration_v2_on():
        return {"error": _TASKS_V2_DISABLED}
    target = team_id or project_id
    try:
        return await mesh_client.team_summary(target)
    except Exception as e:
        return {"error": f"Failed to summarize {target}: {e}"}


@skill(
    name="manage_team",
    description=(
        "Team lifecycle action (archive / unarchive / propose-delete). "
        "Destructive actions require user confirmation."
    ),
    parameters={
        "action": {
            "type": "string",
            "enum": ["archive", "unarchive", "propose_delete"],
        },
        "team_name": {"type": "string", "description": "Team name"},
    },
)
async def manage_team(
    action: str,
    team_name: str = "",
    *,
    project_name: str = "",
    mesh_client=None,
    _messages=None,
    **_kw,
) -> dict:
    """Team lifecycle dispatcher — archive, unarchive, or propose deletion.

    Dispatches each action to the matching mesh endpoint rather than
    delegating to :func:`manage_project`, whose action surface predates
    the rename and only covers ``archive``/``delete``.
    """
    if not _is_operator():
        return {"error": "This tool is only available to the operator agent."}
    if mesh_client is None:
        return {"error": "No mesh_client available"}

    name = team_name or project_name
    if not name:
        return {"error": "team_name is required"}

    if action == "archive":
        try:
            return await mesh_client.archive_team(name)
        except Exception as e:
            return {"error": f"Failed to archive team: {e}"}

    if action == "unarchive":
        try:
            return await mesh_client.unarchive_team(name)
        except Exception as e:
            return {"error": f"Failed to unarchive team: {e}"}

    if action == "propose_delete":
        try:
            result = await mesh_client.propose_delete_team(name)
            result.setdefault("requires_confirmation", True)
            return result
        except Exception as e:
            msg = str(e)
            if "must be archived" in msg.lower() or "400" in msg:
                return {
                    "error": "archive_required",
                    "detail": (
                        f"Team {name!r} must be archived first. "
                        "Call manage_team(action='archive') first."
                    ),
                }
            return {"error": f"Failed to propose delete: {e}"}

    return {
        "error": f"Unknown action {action!r}; use archive|unarchive|propose_delete"
    }
