"""Single source of truth for the engine's operational limits.

Historically these limits (LLM call timeout, per-task tool-round budget,
interactive round ceilings, lane wall-clock) were scattered hard-coded magic
numbers — set low, mutually inconsistent, and not operator-adjustable. A
client whose agent was configured for an 80k-token output but given only a
120s LLM timeout could never converge: the engine permitted work its own
limits couldn't accommodate.

This module centralizes every operational limit with a HIGH default and a
clamp range, resolved in priority order:

    per-agent override (agents.yaml / edit_agent)
        -> OPENLEGION_* env var (incl. dashboard system-settings)
            -> built-in default

IMPORTANT: these are convergence / UX / robustness bounds, NOT cost controls.
The per-agent daily **budget** (cost vault preflight) remains the hard spend
backstop, and the lane wall-clock remains the hung-task backstop. Raising
these limits widens how much *legitimate* work can complete before a soft
bound trips; it does not remove the cost/liveness guarantees.

Agent-side limits (round caps, LLM timeout) are read inside the agent
container from env; the host injects the resolved values into each container's
env at creation (see ``set_llm_limits_env``) so a settings.json / per-agent
change survives a restart, mirroring the proven ``LLM_MAX_TOKENS`` path.
Host-side limits (lane timeout/queue) are read directly in the host process.
"""

from __future__ import annotations

import os
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("shared.limits")

# key -> (default, lo, hi). ``hi`` is the adjustability ceiling: an operator
# may tune anywhere in [lo, hi]; values outside are clamped (never rejected).
LIMIT_SPECS: dict[str, tuple[int, int, int]] = {
    # LLM call timeout. With streaming task execution this is an *idle*
    # (between-chunk) ceiling, so it can be generous without risking a hang.
    "llm_timeout_seconds": (1800, 10, 7200),
    # Per-task convergence budget — tool rounds while a durable task_id drives
    # the turn. Must stay <= chat_max_tool_rounds (loop.py enforces it).
    "task_max_tool_rounds": (300, 1, 1000),
    # Interactive (no task_id) per-turn round ceiling.
    "chat_max_tool_rounds": (500, 1, 1000),
    # Interactive cross-turn total round ceiling for one session.
    "chat_max_total_rounds": (1000, 1, 5000),
    # Legacy execute_task bounded-loop ceiling.
    "max_iterations": (300, 1, 500),
    # Host-side lane wall-clock watchdog (hung-stream / stuck-tool backstop).
    "lane_timeout_seconds": (14400, 30, 86400),
    # Host-side per-agent followup queue depth cap (0 disables).
    "lane_queue_max": (100, 0, 10000),
    # Team Drive: per-push request-body ceiling (MB). Also written into the
    # bare repo as ``receive.maxInputSize`` at provision time — the config
    # copy re-syncs on every ensure, so a settings change applies on the
    # next boot/provision, not retroactively.
    "drive_push_max_mb": (64, 1, 512),
    # Team Drive: on-disk repo size quota (MB). Checked BEFORE receive-pack
    # runs, under a per-repo lock that serializes the check→push→invalidate
    # window — so a concurrent-push overshoot is bounded to ONE push
    # (drive_push_max_mb), not N simultaneous pushes.
    "drive_quota_mb": (512, 1, 65536),
}

# key -> OPENLEGION_* env var name (the second-lowest precedence source).
ENV_NAMES: dict[str, str] = {
    "llm_timeout_seconds": "OPENLEGION_LLM_TIMEOUT_SECONDS",
    "task_max_tool_rounds": "OPENLEGION_TASK_MAX_TOOL_ROUNDS",
    "chat_max_tool_rounds": "OPENLEGION_CHAT_MAX_TOOL_ROUNDS",
    "chat_max_total_rounds": "OPENLEGION_CHAT_MAX_TOTAL_ROUNDS",
    "max_iterations": "OPENLEGION_MAX_ITERATIONS",
    "lane_timeout_seconds": "OPENLEGION_LANE_TIMEOUT_SECONDS",
    "lane_queue_max": "OPENLEGION_LANE_QUEUE_MAX",
    "drive_push_max_mb": "OPENLEGION_DRIVE_PUSH_MAX_MB",
    "drive_quota_mb": "OPENLEGION_DRIVE_QUOTA_MB",
}

# Per-agent config keys (agents.yaml / edit_agent) -> limits key. Only the
# limits worth tuning per-agent are exposed; the rest stay global.
AGENT_CONFIG_KEYS: dict[str, str] = {
    "max_tool_rounds": "task_max_tool_rounds",
    "llm_timeout_seconds": "llm_timeout_seconds",
}

# Limits surfaced in the dashboard global "Agent Execution" settings panel.
# Declared here (not in the dashboard) so the UI can never silently diverge
# from LIMIT_SPECS and so the deliberate exclusion is explicit:
# ``lane_queue_max`` is intentionally omitted — it's a niche backpressure
# knob that stays env-only (OPENLEGION_LANE_QUEUE_MAX).
DASHBOARD_GLOBAL_KEYS: tuple[str, ...] = (
    "max_iterations",
    "chat_max_tool_rounds",
    "chat_max_total_rounds",
    "task_max_tool_rounds",
    "llm_timeout_seconds",
    "lane_timeout_seconds",
)


# ── Shared validation values ─────────────────────────────────────────
# These rules are deliberately enforced in MULTIPLE trust zones (agent
# container + mesh host + CLI — defense in depth). The enforcement sites
# stay where they are; only the VALUES live here so the copies can't
# drift.

# Per-agent LLM output-token cap bounds. Enforced identically by the
# agent ``/config`` endpoint, the host ``/edit-soft`` route and
# ``operator_tools._validate_edit``, and used as the clamp range for the
# ``LLM_MAX_TOKENS`` env read in ``src/agent/__main__.py``.
MAX_OUTPUT_TOKENS_MIN = 256
MAX_OUTPUT_TOKENS_MAX = 200_000

# Valid agent "thinking" levels, in display order (the CLI interactive
# editor renders them 1..N in this order). Consumers wrap in set()/
# list() as needed; ``LLMClient.VALID_THINKING_LEVELS`` derives from
# this.
THINKING_LEVELS: tuple[str, ...] = ("off", "low", "medium", "high")


def clamp(key: str, value: int) -> int:
    """Clamp ``value`` into the spec range for ``key`` (logs if it moved)."""
    _default, lo, hi = LIMIT_SPECS[key]
    clamped = max(lo, min(value, hi))
    if clamped != value:
        logger.info("limit %s=%d clamped to %d (range %d-%d)", key, value, clamped, lo, hi)
    return clamped


def _coerce_int(raw: Any, key: str) -> int | None:
    """Parse ``raw`` to a clamped int, or None if absent/invalid.

    ``bool`` is rejected explicitly (it is an ``int`` subclass) so a stray
    ``True``/``False`` can't silently become ``1``/``0``.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return clamp(key, int(raw))
    except (TypeError, ValueError):
        logger.warning("Invalid value %r for limit %s — ignoring", raw, key)
        return None


def resolve(key: str, *, agent_cfg: dict | None = None) -> int:
    """Resolve a limit through the precedence chain to a clamped int.

    per-agent override -> env var -> built-in default. The first source that
    yields a usable value wins; everything is clamped to the spec range.
    """
    default, _lo, _hi = LIMIT_SPECS[key]

    # 1. per-agent override (highest precedence)
    if agent_cfg:
        for cfg_key, limit_key in AGENT_CONFIG_KEYS.items():
            if limit_key == key:
                v = _coerce_int(agent_cfg.get(cfg_key), key)
                if v is not None:
                    return v

    # 2. env var (set directly, or by the dashboard system-settings surface)
    v = _coerce_int(os.environ.get(ENV_NAMES[key]), key)
    if v is not None:
        return v

    # 3. built-in default
    return default


def set_llm_limits_env(env: dict[str, str], agent_cfg: dict) -> None:
    """Inject per-agent limit overrides into a container env dict from config.

    Mirrors ``set_llm_max_tokens_env``: called by every restart-from-config
    path so an operator's ``edit_agent`` change to ``max_tool_rounds`` /
    ``llm_timeout_seconds`` survives a container restart. No-op for unset /
    non-int values (the env/settings/default chain then applies), and a no-op
    for a missing/malformed agent config (a null agents.yaml entry yields
    ``None``) rather than raising.
    """
    if not isinstance(agent_cfg, dict):
        return
    for cfg_key, limit_key in AGENT_CONFIG_KEYS.items():
        v = _coerce_int(agent_cfg.get(cfg_key), limit_key)
        if v is not None:
            env[ENV_NAMES[limit_key]] = str(v)
