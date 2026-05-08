"""Unit tests for shared Pydantic types."""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

from src.shared.types import (
    AgentConfig,
    AgentMessage,
    AgentStatus,
    DashboardEvent,
    TaskAssignment,
    TaskResult,
    TokenBudget,
)


def test_agent_message_defaults():
    msg = AgentMessage(from_agent="a1", to="a2", type="task_request", payload={"x": 1})
    assert msg.id.startswith("msg_")
    assert msg.ttl == 300
    assert msg.priority == "normal"


def test_token_budget_can_spend():
    budget = TokenBudget(max_tokens=10_000, used_tokens=0)
    assert budget.can_spend(5_000)
    assert budget.can_spend(10_000)
    assert not budget.can_spend(10_001)


def test_token_budget_record_usage():
    budget = TokenBudget(max_tokens=100_000)
    budget.record_usage(1000, "anthropic/claude-sonnet-4-5-20250929")
    assert budget.used_tokens == 1000
    assert budget.estimated_cost_usd > 0


def test_task_assignment_defaults():
    ta = TaskAssignment(workflow_id="wf_1", step_id="s1", task_type="research", input_data={"q": "test"})
    assert ta.task_id.startswith("task_")
    assert ta.timeout == 120
    assert ta.context == {}


def test_task_result_serialization():
    tr = TaskResult(task_id="t1", status="complete", result={"key": "val"}, promote_to_blackboard={"ctx/a": "b"})
    d = tr.model_dump()
    assert d["status"] == "complete"
    assert d["promote_to_blackboard"] == {"ctx/a": "b"}


def test_agent_status_fields():
    status = AgentStatus(agent_id="a1", role="research", state="idle")
    assert status.tasks_completed == 0
    assert status.capabilities == []


def test_token_budget_record_usage_uses_unified_costs():
    """WU2: record_usage delegates to estimate_cost from costs.py (18+ models)."""
    budget = TokenBudget(max_tokens=1_000_000)
    # Test a model that was NOT in the old inline dict
    budget.record_usage(1000, "openai/gpt-4o")
    assert budget.estimated_cost_usd > 0
    # Test unknown model gets a reasonable fallback
    budget2 = TokenBudget(max_tokens=1_000_000)
    budget2.record_usage(1000, "unknown/model")
    assert budget2.estimated_cost_usd > 0


# ── Task 8: AgentConfig structured routing fields ──


def test_agent_config_defaults_empty():
    """All five Task-8 routing fields default cleanly so existing
    agents.yaml entries without them load unchanged."""
    cfg = AgentConfig(role="researcher", model="openai/gpt-4o-mini")
    assert cfg.capabilities == []
    assert cfg.preferred_inputs == []
    assert cfg.expected_outputs == []
    assert cfg.escalation_to is None
    assert cfg.forbidden == []


def test_agent_config_accepts_structured_fields():
    cfg = AgentConfig(
        role="pm",
        model="openai/gpt-4o-mini",
        capabilities=["Break down specs", "Coordinate handoffs"],
        preferred_inputs=["User requests"],
        expected_outputs=["Task specs"],
        escalation_to="operator",
        forbidden=["Writing code directly"],
    )
    assert cfg.capabilities == ["Break down specs", "Coordinate handoffs"]
    assert cfg.preferred_inputs == ["User requests"]
    assert cfg.expected_outputs == ["Task specs"]
    assert cfg.escalation_to == "operator"
    assert cfg.forbidden == ["Writing code directly"]


def test_agent_config_extra_fields_allowed():
    """Extra keys (e.g., legacy ``initial_interface``) round-trip via
    ``extra='allow'`` so loading isn't a strict-validation gate."""
    cfg = AgentConfig(role="pm", model="x", initial_interface="hello")
    assert cfg.initial_interface == "hello"
    cfg2 = AgentConfig(
        role="pm",
        model="x",
        capabilities=["a"],
        legacy_field="ignored-but-kept",  # type: ignore[call-arg]
    )
    dumped = cfg2.model_dump()
    assert dumped["legacy_field"] == "ignored-but-kept"
    assert dumped["capabilities"] == ["a"]


# ── DashboardEvent.type literal coverage ───────────────────────────────


# Strings that look like emit calls but are NOT WebSocket event names —
# they're rate-limit category keys, audit actions, etc. These must be
# excluded from the literal-coverage sweep below.
_NON_WS_EMIT_STRINGS: frozenset[str] = frozenset({
    # Rate-limit category passed to ``_check_rate_limit`` — same call
    # shape as ``event_bus.emit`` but completely separate namespace.
    "blackboard_write",
    # ``cli/runtime.py`` emits ``message_sent`` / ``message_received``
    # which are bona-fide WS event types — keep them out of the
    # exclusion set.
})

# Pattern matches:
#   event_bus.emit("name", ...
#   self._event_bus.emit("name", ...
#   self.event_bus.emit("name", ...
#   self._safe_emit(\n    "name", ...
# It deliberately scans only ``src/`` (not test fixtures or docs).
_EMIT_RE = re.compile(
    r"""(?xs)
    (?:event_bus|_event_bus|_safe_emit)
    \s* (?: \. \s* emit )? \s*
    \( \s*
    "([a-z][a-z0-9_]+)"   # the event-type literal — captured
    """,
)


def _src_root() -> Path:
    # tests/test_types.py → repo/src
    return Path(__file__).resolve().parent.parent / "src"


def test_dashboard_event_literal_count_pinned():
    """The literal count is referenced in CLAUDE.md and the
    ``_dashboard/server.py`` docstring — fail loud if it drifts so the
    docs can be updated in lockstep with the contract change."""
    literals = get_args(DashboardEvent.model_fields["type"].annotation)
    # When this fails, update CLAUDE.md ("DashboardEvent.type Literal
    # enumerates N WebSocket event names") and the comment in the
    # ``Known Constraints & Decisions`` section in lockstep.
    assert len(literals) >= 50, (
        f"DashboardEvent.type has {len(literals)} literals — "
        f"CLAUDE.md last referenced 50."
    )


def test_every_emit_string_in_src_matches_a_dashboard_event_literal():
    """Regex sweep — guards against silent EventBus drops.

    ``EventBus.emit("foo")`` raises ``ValidationError`` when ``"foo"``
    isn't in the ``DashboardEvent.type`` Literal, and the emit-site
    ``try/except`` swallows the error at debug level. This means a
    typo or a forgotten-to-register literal silently drops the event
    on the floor — exactly the regression that prompted this audit.

    This test trips when:
      * a new ``event_bus.emit("new_type", ...)`` lands in src/ but
        ``DashboardEvent.type`` wasn't extended in lockstep, or
      * a literal is removed from ``DashboardEvent.type`` while an
        emit site still references it.
    """
    literals = set(get_args(DashboardEvent.model_fields["type"].annotation))
    found: set[str] = set()
    src = _src_root()
    for path in src.rglob("*.py"):
        # __pycache__ shows up as binary garbage — skip it.
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _EMIT_RE.finditer(text):
            name = match.group(1)
            if name in _NON_WS_EMIT_STRINGS:
                continue
            found.add(name)

    missing = sorted(found - literals)
    assert not missing, (
        f"emit() strings without a matching DashboardEvent.type literal: "
        f"{missing}. Either add the literal to src/shared/types.py or "
        f"strip the emit call. Silent ValidationError swallowing means "
        f"these events never reach the dashboard."
    )


