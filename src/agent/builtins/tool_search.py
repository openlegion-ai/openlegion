"""Grouped Tool Search bridge — the ``load_tools`` builtin (Phase 2 / B2).

When grouped tool search is active (the budget gate trips on a large enough
tool surface), most non-core tools are advertised cheaply in the system
prompt's capability index but their full schemas are deferred. ``load_tools``
is the explicit, always-available bridge that pulls a group's (or a single
tool's owning group's) full schemas into context for the agent's NEXT turn —
so a capability is never lost, only its verbose schema is lazy.

The actual schema change is deferred to the next turn boundary by the loop
(``AgentLoop.request_load_tools`` queues the group; the next system-prompt
build promotes it) so the toolset never mutates mid-turn — preserving the
prompt cache. This tool is a thin pass-through to that loop method.
"""

from __future__ import annotations

from src.agent.tools import tool
from src.shared.utils import setup_logging

logger = setup_logging("agent.builtins.tool_search")


@tool(
    name="load_tools",
    description=(
        "Load the full schemas for a capability group (or a single tool's "
        "group) so you can call those tools. Use this when the Capability "
        "Index lists a tool you need but its parameters aren't shown. The "
        "schemas become available on your NEXT turn — call load_tools, then "
        "call the tool in a following turn. Pass either 'group' (e.g. "
        "'fleet_setup', 'scheduling', 'credentials', 'goals_review', "
        "'audit_undo', 'web_browse') or 'tool' (a specific tool name)."
    ),
    parameters={
        "group": {
            "type": "string",
            "description": "Capability group key (or label) to load, e.g. 'fleet_setup'.",
            "default": "",
        },
        "tool": {
            "type": "string",
            "description": "A specific tool name to load (its whole group is pulled in).",
            "default": "",
        },
    },
    loop_exempt=True,
)
async def load_tools(group: str = "", tool: str = "", *, agent_loop=None) -> dict:
    """Queue a capability group's full schemas for the next turn."""
    if agent_loop is None:
        return {
            "loaded": [],
            "error": "load_tools is unavailable outside an agent loop.",
        }
    return agent_loop.request_load_tools(
        group=group or None, tool=tool or None,
    )
