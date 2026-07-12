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
    # ask_teammate inline Q&A wait (Phase 2 unit 3). How long the asker's
    # /mesh/ask call waits for an answer before returning the timeout
    # envelope. Host-side; the agent tool derives its HTTP timeout from
    # the same resolution + headroom.
    "ask_timeout_seconds": (180, 30, 600),
    # Team Drive: per-file cap for the direct-commit artifact endpoint
    # (POST /drive/artifacts) — hand_off data payloads + save_artifact
    # registration. Larger content degrades gracefully (handoff → inline
    # brief; artifact → saved-but-unregistered with the workspace path).
    "drive_artifact_max_mb": (8, 1, 512),
    # Priority steer lane (Phase 3 unit 3). How long a busy-chat steer
    # (``LaneManager.deliver_chat`` / ``try_steer_and_wait``) waits for the
    # running turn's actual reply before returning a "still processing"
    # placeholder. The message was already folded into the turn by then —
    # a timeout here is degraded UX, not a dropped message.
    "steer_reply_timeout_seconds": (150, 15, 600),
    # Offboarding-with-handover (plan §8 #15). How long the offboard
    # helper waits for the departing agent's own handover turn (a
    # normal followup-lane dispatch on the still-live container) before
    # giving up and committing the snapshot without a handover doc.
    # Mirrors ``ask_timeout_seconds``'s shape — a one-shot bounded LLM
    # turn, not a cost control.
    "offboard_handover_timeout_seconds": (180, 30, 600),
    # Kernel-executed auto-merge (plan §8 #20) — count of TODAY's
    # auto-merges across the whole deployment (UTC-midnight boundary,
    # queried from the track record's system-rated `auto_merged` events)
    # must stay below this cap. 0 DISABLES auto-merge entirely (B4-style
    # 0-valid kill switch) — the low end is 0 on purpose, do not clamp it up.
    "auto_merge_daily_cap": (3, 0, 1000),
    # Kernel-executed auto-merge (plan §8 #20) — HUMAN-executed merges of
    # this (lead, submitter) pair's lead-approved reviews required (zero
    # rejected-after-approve, zero flag/revert decay events) before the
    # kernel may auto-merge for the pair. A floor of 0 would make every
    # pair trivially eligible, so the low end is clamped to 1.
    "auto_merge_trust_floor": (5, 1, 10000),
    # Kernel-executed auto-merge (plan §8 #20) — a pair's first this-many
    # auto-merges sample at `auto_merge_sample_rate_initial`; afterward
    # sampling decays to `auto_merge_sample_rate_floor`.
    "auto_merge_sample_decay_after": (10, 1, 100000),
    # Blocked-task escalation ladder (plan §8 #22) — minutes a task must
    # sit blocked before each rung climb (rung 1 re-drives the assignee,
    # rung 2 the creator, rung 3 lands on the lead's plate). 0 DISABLES
    # the ENTIRE ladder including the rung-4 budget fast path (B4-style
    # 0-valid kill switch) — the low end is 0 on purpose, do not clamp it up.
    "ladder_rung_interval_minutes": (30, 0, 10080),
    # Blocked-task escalation ladder (plan §8 #22) — hours blocked after
    # which rung 4 (ONE durable human Needs-you entry per task) fires for
    # any blocker; budget-exhausted blockers skip straight to rung 4
    # without waiting. Not independently disableable — the interval kill
    # switch above turns the whole ladder off.
    "ladder_human_fallback_hours": (48, 1, 8760),
    # Goal-coverage probe (plan §8 #22) — a lead whose team has goals set
    # (north_star / success_criteria) but fewer than this many open
    # (pending/accepted/working) team tasks gets a plate alert to
    # decompose the goals into tasks. 0 DISABLES the probe (B4-style).
    "goal_coverage_min_open_tasks": (1, 0, 1000),
    # Idle-agent hibernation sweep (plan §8 #24) — minutes an agent must
    # sit idle (no busy/queued lane work, no working task, no open ask)
    # before the mesh-side sweep stops its container (data persisted,
    # cron jobs kept, auto-wakes on demand). 0 DISABLES the sweep
    # entirely (B4-style 0-valid default) — hibernation is opt-in; the
    # wake path itself works regardless (an operator can hibernate
    # manually via the endpoint even with the sweep off).
    "hibernate_idle_minutes": (0, 0, 43200),
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
    "ask_timeout_seconds": "OPENLEGION_ASK_TIMEOUT_SECONDS",
    "drive_artifact_max_mb": "OPENLEGION_DRIVE_ARTIFACT_MAX_MB",
    "steer_reply_timeout_seconds": "OPENLEGION_STEER_REPLY_TIMEOUT_SECONDS",
    "offboard_handover_timeout_seconds": "OPENLEGION_OFFBOARD_HANDOVER_TIMEOUT_SECONDS",
    "auto_merge_daily_cap": "OPENLEGION_AUTO_MERGE_DAILY_CAP",
    "auto_merge_trust_floor": "OPENLEGION_AUTO_MERGE_TRUST_FLOOR",
    "auto_merge_sample_decay_after": "OPENLEGION_AUTO_MERGE_SAMPLE_DECAY_AFTER",
    "ladder_rung_interval_minutes": "OPENLEGION_LADDER_RUNG_INTERVAL_MINUTES",
    "ladder_human_fallback_hours": "OPENLEGION_LADDER_HUMAN_FALLBACK_HOURS",
    "goal_coverage_min_open_tasks": "OPENLEGION_GOAL_COVERAGE_MIN_OPEN_TASKS",
    "hibernate_idle_minutes": "OPENLEGION_HIBERNATE_IDLE_MINUTES",
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

# ask_teammate payload caps. Enforced on BOTH sides of the trust
# boundary: the agent-side tool truncates before sending, the mesh
# endpoints truncate again after sanitize (the container is untrusted,
# so the mesh copy is the one that counts).
ASK_QUESTION_MAX_CHARS = 4_000
ASK_ANSWER_MAX_CHARS = 8_000

# Per-ask billed-spend cap (USD). While an ask's billing window is
# active, the recipient's LLM calls bill the ASKER; once the billed
# total crosses this cap the window closes and subsequent calls bill
# the recipient normally — bounding both an asker-funded runaway turn
# and recipient-side cost-dumping. Float-valued, so it lives outside
# the int-only ``LIMIT_SPECS`` table with the same env-override +
# clamp contract.
ASK_BILL_CAP_USD_DEFAULT = 0.50
_ASK_BILL_CAP_USD_RANGE = (0.01, 100.0)


def ask_bill_cap_usd() -> float:
    """Resolve the per-ask billed-spend cap (env override, clamped)."""
    raw = os.environ.get("OPENLEGION_ASK_BILL_CAP_USD")
    if raw is None:
        return ASK_BILL_CAP_USD_DEFAULT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid OPENLEGION_ASK_BILL_CAP_USD=%r — using default %.2f",
            raw, ASK_BILL_CAP_USD_DEFAULT,
        )
        return ASK_BILL_CAP_USD_DEFAULT
    lo, hi = _ASK_BILL_CAP_USD_RANGE
    clamped = max(lo, min(value, hi))
    if clamped != value:
        logger.info(
            "ask_bill_cap_usd=%s clamped to %s (range %s-%s)",
            value, clamped, lo, hi,
        )
    return clamped


# Per-agent daily coordination-spend cap (USD) — B2 spend split (plan
# §8 #11). Utility-model (coordination) LLM calls skip the per-agent
# work preflight and the team envelope, and are gated by THIS cap
# instead. Unlike ask_bill_cap_usd, 0 is a VALID value: it blocks the
# coordination tier entirely (operator kill-switch restoring
# probe-only ticks) — do not clamp it up to a minimum.
COORDINATION_DAILY_CAP_USD_DEFAULT = 2.0
_COORDINATION_DAILY_CAP_USD_RANGE = (0.0, 100.0)


def coordination_daily_cap_usd() -> float:
    """Resolve the per-agent daily coordination cap (env override, clamped; 0 = tier blocked)."""
    raw = os.environ.get("OPENLEGION_COORDINATION_DAILY_CAP_USD")
    if raw is None:
        return COORDINATION_DAILY_CAP_USD_DEFAULT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid OPENLEGION_COORDINATION_DAILY_CAP_USD=%r — using default %.2f",
            raw, COORDINATION_DAILY_CAP_USD_DEFAULT,
        )
        return COORDINATION_DAILY_CAP_USD_DEFAULT
    lo, hi = _COORDINATION_DAILY_CAP_USD_RANGE
    clamped = max(lo, min(value, hi))
    if clamped != value:
        logger.info(
            "coordination_daily_cap_usd=%s clamped to %s (range %s-%s)",
            value, clamped, lo, hi,
        )
    return clamped


# Kernel-executed auto-merge sampling (plan §8 #20). Fraction of a
# trust-cleared pair's auto-merges flagged for human post-review — high
# at first, decaying once the pair has accumulated
# `auto_merge_sample_decay_after` auto-merges with no flag/revert.
# Float-valued, so these live outside the int-only LIMIT_SPECS table
# with the same env-override + clamp contract as ask_bill_cap_usd /
# coordination_daily_cap_usd.
AUTO_MERGE_SAMPLE_RATE_INITIAL_DEFAULT = 0.20
AUTO_MERGE_SAMPLE_RATE_FLOOR_DEFAULT = 0.05
_AUTO_MERGE_SAMPLE_RATE_RANGE = (0.0, 1.0)


def _resolve_rate_env(env_name: str, default: float) -> float:
    """Shared float-limit resolver: env override, clamped to [0, 1]."""
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r — using default %.2f", env_name, raw, default)
        return default
    lo, hi = _AUTO_MERGE_SAMPLE_RATE_RANGE
    clamped = max(lo, min(value, hi))
    if clamped != value:
        logger.info("%s=%s clamped to %s (range %s-%s)", env_name, value, clamped, lo, hi)
    return clamped


def auto_merge_sample_rate_initial() -> float:
    """Resolve the initial (pre-decay) auto-merge sampling rate."""
    return _resolve_rate_env(
        "OPENLEGION_AUTO_MERGE_SAMPLE_RATE_INITIAL",
        AUTO_MERGE_SAMPLE_RATE_INITIAL_DEFAULT,
    )


def auto_merge_sample_rate_floor() -> float:
    """Resolve the decayed (post-threshold) auto-merge sampling rate."""
    return _resolve_rate_env(
        "OPENLEGION_AUTO_MERGE_SAMPLE_RATE_FLOOR",
        AUTO_MERGE_SAMPLE_RATE_FLOOR_DEFAULT,
    )


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
