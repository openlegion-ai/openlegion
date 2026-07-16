"""Grouped Tool Search — capability index + lazy schema loading (Phase 2 / B2).

Goal: keep the per-turn tool-schema load lean WITHOUT the agent ever losing
awareness of a capability ("it knows exactly what to do"). Three layers:

1. **Always-loaded core** — the daily-driver tools, full schemas, unchanged.
2. **Always-loaded capability INDEX** — names + one-line intent, grouped by
   job-to-be-done (~300-500 tokens, NO full schemas). A tool's *capability*
   is never hidden; only its verbose schema is lazy.
3. **On-demand schema loading** — the ``load_tools(group | tool_name)`` bridge
   pulls full schemas into context for SUBSEQUENT turns.

This is **budget-gated**: the index + deferral only activate when an agent's
deferrable schemas exceed a fraction of the context window. Small-toolset
agents present everything unchanged (no behaviour change) — the budget gate is
the sole activation control.

Groups are defined here as DATA (a registry mapping group → tool names +
one-line intent + role-eligibility). Applies to BOTH operator and worker
scope; a worker simply never sees an operator-only group's tools in its
allowed/exclude surface, so the index naturally renders only what it can call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Budget gate: only activate index+defer when the deferrable tools' estimated
# schema cost exceeds this fraction of the model's context window. Mirrors
# hermes-agent's "~10% of the window" Tool Search activation threshold.
BUDGET_FRACTION = 0.10

# Rough token estimate per deferred tool schema. Tool schemas vary, but a
# typical builtin definition (name + description + a few params) lands around
# 120-180 tokens; we use a conservative midpoint so the gate trips on genuinely
# large surfaces, not marginal ones. Only used for the gate, never for billing.
SCHEMA_TOKENS_PER_TOOL = 150


@dataclass(frozen=True)
class ToolGroup:
    """A job-to-be-done grouping of tools for the capability index.

    *intent* is the one-line "when would I reach for this" string shown in the
    index. *tools* are the tool names whose full schemas are deferred until
    ``load_tools`` pulls the group in. *operator_only* mirrors the registry's
    operator-only marking so the index never advertises a group a worker can't
    use (defensive — the allowed/exclude surface already filters the names).
    """

    key: str
    label: str
    intent: str
    tools: tuple[str, ...]
    operator_only: bool = False


# ── Group registry (DATA) ──────────────────────────────────────────────────
# Intent-named (job-to-be-done), NOT module-named. Order here is the order the
# index renders. Tool names that don't exist in a given agent's surface are
# simply skipped at render/defer time, so this list can stay a superset.
TOOL_GROUPS: tuple[ToolGroup, ...] = (
    ToolGroup(
        key="fleet_setup",
        label="Fleet setup",
        intent=(
            "build, restructure, or check spend on a team (create agents/"
            "teams, apply templates, read team spend)"
        ),
        tools=(
            "create_agent", "create_team", "apply_template", "list_templates",
            "add_agents_to_team", "remove_agents_from_team", "manage_team",
            "manage_agent", "edit_agent", "read_agent_config",
            "update_team_context", "set_team_lead", "inspect_team_spend",
        ),
        operator_only=True,
    ),
    ToolGroup(
        key="scheduling",
        label="Scheduling",
        intent="recurring or timed work (cron jobs)",
        tools=("set_cron", "list_cron", "remove_cron"),
        operator_only=True,
    ),
    ToolGroup(
        key="credentials",
        label="Credentials & access",
        intent="an agent needs to reach a service (vault, credential/login requests)",
        tools=(
            "vault_list", "request_credential", "request_browser_login",
        ),
    ),
    ToolGroup(
        key="goals_review",
        label="Goals & review",
        intent="set or review team goals and rate deliveries",
        tools=(
            "set_team_goal", "manage_goals", "rate_delivery",
            "assess_team_progress",
        ),
        operator_only=True,
    ),
    ToolGroup(
        key="audit_undo",
        label="Audit & undo",
        intent="inspect pending actions, undo a change, or trim the audit log",
        tools=(
            "list_pending", "cancel_pending_action", "archive_audit_before",
            "undo_change",
        ),
        operator_only=True,
    ),
    ToolGroup(
        key="web_browse",
        label="Web & browse",
        intent="research or interact with websites (browser automation)",
        tools=(
            "browser_navigate", "browser_warmup", "browser_get_elements",
            "browser_wait_for", "browser_screenshot",
            "browser_screenshot_marks", "browser_click",
            "browser_click_xy", "browser_type", "browser_hover",
            "browser_scroll", "browser_reset", "browser_press_key",
            "browser_go_back", "browser_go_forward", "browser_switch_tab",
            "browser_find_text", "browser_fill_form", "browser_open_tab",
            "browser_inspect_requests", "browser_detect_captcha",
            "browser_upload_file", "browser_solve_captcha", "browser_download",
            "browser_set_dialog_policy", "browser_drag",
            "browser_grant_permissions", "browser_set_geolocation",
            "browser_right_click", "browser_read_clipboard",
            "browser_write_clipboard", "browser_wait_for_network_idle",
            "browser_select_option",
        ),
    ),
)

# Index of group-by-key for O(1) lookup.
_GROUPS_BY_KEY: dict[str, ToolGroup] = {g.key: g for g in TOOL_GROUPS}

# Every tool name that belongs to *some* group — the deferrable universe.
_GROUPED_TOOL_NAMES: frozenset[str] = frozenset(
    name for g in TOOL_GROUPS for name in g.tools
)


def is_grouped_tool(name: str) -> bool:
    """True iff *name* belongs to a deferrable group (vs. always-loaded core)."""
    return name in _GROUPED_TOOL_NAMES


def groups_for_tool(name: str) -> list[str]:
    """Return the group keys a tool name belongs to (usually 0 or 1)."""
    return [g.key for g in TOOL_GROUPS if name in g.tools]


@dataclass
class GroupedPlan:
    """The resolved grouped-tools plan for one turn.

    *active* — whether the budget gate tripped (index+defer in effect). When
    false, the agent presents its full surface unchanged.
    *defer* — tool names whose full schemas to omit this turn (deferred groups
    minus anything already loaded).
    *index_text* — the always-present capability index string (or "" when
    inactive).
    """

    active: bool = False
    defer: frozenset[str] = field(default_factory=frozenset)
    index_text: str = ""


def _available_for_role(
    available: set[str], *, operator: bool,
) -> list[ToolGroup]:
    """Groups that have at least one tool in *available* and are role-eligible.

    A non-operator agent never sees operator-only groups (their tools won't be
    in *available* anyway, but we also skip the group entirely so the index
    doesn't advertise an empty group).
    """
    out: list[ToolGroup] = []
    for g in TOOL_GROUPS:
        if g.operator_only and not operator:
            continue
        if any(t in available for t in g.tools):
            out.append(g)
    return out


def _render_index(
    groups: list[ToolGroup],
    available: set[str],
    loaded_groups: set[str],
) -> str:
    """Render the always-present capability index (names + intent, no schemas).

    Loaded groups are marked so the model knows their full schemas are already
    in context (no need to ``load_tools`` them again).
    """
    lines = [
        "## Capability Index",
        (
            "These tools are available but their full schemas are loaded "
            "on demand to keep context lean. The capability is never hidden — "
            "only its detailed parameters are lazy. Call "
            "`load_tools(group=\"<group>\")` (or `load_tools(tool=\"<name>\")`) "
            "to pull a group's full schemas into context for your NEXT turn, "
            "then call the tool normally."
        ),
        "",
    ]
    for g in groups:
        names = [t for t in g.tools if t in available]
        if not names:
            continue
        loaded_mark = " (loaded)" if g.key in loaded_groups else ""
        lines.append(
            f"- **{g.label}** [`{g.key}`]{loaded_mark} — {g.intent}: "
            + ", ".join(names)
        )
    return "\n".join(lines)


def plan_grouped_tools(
    *,
    available: set[str],
    loaded_groups: set[str],
    operator: bool,
    context_window: int,
) -> GroupedPlan:
    """Compute the grouped-tools plan for one turn (pure, deterministic).

    Budget gate: only activate when the deferrable schemas' estimated cost
    exceeds ``BUDGET_FRACTION`` of *context_window*. Below that, return an
    inactive plan (full surface, no index) so small-toolset agents are
    unchanged.

    *available* — the tool names this agent can actually call this turn (after
    allowed/exclude filtering). *loaded_groups* — groups the agent has already
    pulled in via ``load_tools`` (their schemas stay present). *operator* —
    role eligibility for operator-only groups.
    """
    groups = _available_for_role(available, operator=operator)
    # Deferrable universe = grouped tools present in this agent's surface.
    deferrable = {t for g in groups for t in g.tools if t in available}
    if not deferrable:
        return GroupedPlan(active=False)

    # Budget gate — only kick in for genuinely large surfaces.
    est_tokens = len(deferrable) * SCHEMA_TOKENS_PER_TOOL
    if context_window <= 0 or est_tokens < context_window * BUDGET_FRACTION:
        return GroupedPlan(active=False)

    # Tools whose schemas stay loaded: any group the agent has pulled in.
    loaded_tool_names = {
        t
        for key in loaded_groups
        for t in (_GROUPS_BY_KEY[key].tools if key in _GROUPS_BY_KEY else ())
    }
    defer = frozenset(deferrable - loaded_tool_names)
    index_text = _render_index(groups, available, loaded_groups)
    return GroupedPlan(active=True, defer=defer, index_text=index_text)


def resolve_load_request(
    *, group: str | None, tool: str | None, available: set[str],
) -> tuple[set[str], str | None]:
    """Resolve a ``load_tools`` request to a set of group keys to load.

    Accepts either a group key/label or a single tool name (resolved to its
    group). Returns ``(group_keys, error)`` — *error* is a human string when
    the request matched nothing, else None.
    """
    keys: set[str] = set()
    if group:
        g = group.strip().lower()
        match = _GROUPS_BY_KEY.get(g)
        if match is None:
            # Allow matching by human label too.
            for grp in TOOL_GROUPS:
                if grp.label.lower() == g:
                    match = grp
                    break
        if match is None:
            valid = ", ".join(grp.key for grp in TOOL_GROUPS)
            return set(), f"Unknown tool group '{group}'. Valid groups: {valid}"
        keys.add(match.key)
    if tool:
        t = tool.strip()
        owning = groups_for_tool(t)
        if not owning:
            if t in available:
                return set(), (
                    f"Tool '{t}' is already in your core surface — call it directly."
                )
            return set(), f"Tool '{t}' is not a known grouped tool."
        keys.update(owning)
    if not keys:
        return set(), "Specify a 'group' or a 'tool' to load."
    return keys, None
