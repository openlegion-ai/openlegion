"""Unit tests for the central operational-limits resolver."""

from src.shared import limits


def test_resolve_default_when_unset(monkeypatch):
    monkeypatch.delenv("OPENLEGION_TASK_MAX_TOOL_ROUNDS", raising=False)
    assert limits.resolve("task_max_tool_rounds") == limits.LIMIT_SPECS["task_max_tool_rounds"][0]


def test_resolve_env_override(monkeypatch):
    monkeypatch.setenv("OPENLEGION_TASK_MAX_TOOL_ROUNDS", "123")
    assert limits.resolve("task_max_tool_rounds") == 123


def test_resolve_env_clamped_to_ceiling(monkeypatch):
    _default, _lo, hi = limits.LIMIT_SPECS["task_max_tool_rounds"]
    monkeypatch.setenv("OPENLEGION_TASK_MAX_TOOL_ROUNDS", str(hi + 9999))
    assert limits.resolve("task_max_tool_rounds") == hi


def test_resolve_invalid_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OPENLEGION_TASK_MAX_TOOL_ROUNDS", "not-a-number")
    assert limits.resolve("task_max_tool_rounds") == limits.LIMIT_SPECS["task_max_tool_rounds"][0]


def test_agent_override_beats_env(monkeypatch):
    monkeypatch.setenv("OPENLEGION_TASK_MAX_TOOL_ROUNDS", "50")
    # agents.yaml key is ``max_tool_rounds`` -> limits key ``task_max_tool_rounds``
    assert limits.resolve("task_max_tool_rounds", agent_cfg={"max_tool_rounds": 200}) == 200


def test_agent_override_clamped(monkeypatch):
    monkeypatch.delenv("OPENLEGION_TASK_MAX_TOOL_ROUNDS", raising=False)
    _default, _lo, hi = limits.LIMIT_SPECS["task_max_tool_rounds"]
    assert limits.resolve("task_max_tool_rounds", agent_cfg={"max_tool_rounds": hi + 1}) == hi


def test_defaults_are_high():
    # Regression guard: the production incident was caused by low caps. Pin the
    # new floors so a future edit can't silently shrink them back.
    assert limits.LIMIT_SPECS["task_max_tool_rounds"][0] >= 100
    assert limits.LIMIT_SPECS["chat_max_tool_rounds"][0] >= 100
    assert limits.LIMIT_SPECS["llm_timeout_seconds"][0] >= 600
    assert limits.LIMIT_SPECS["lane_timeout_seconds"][0] >= 3600
    # The task budget must stay <= the interactive ceiling (loop.py invariant).
    assert limits.LIMIT_SPECS["task_max_tool_rounds"][0] <= limits.LIMIT_SPECS["chat_max_tool_rounds"][0]


def test_set_llm_limits_env_injects_per_agent():
    env: dict[str, str] = {}
    limits.set_llm_limits_env(env, {"max_tool_rounds": 250, "llm_timeout_seconds": 900})
    assert env["OPENLEGION_TASK_MAX_TOOL_ROUNDS"] == "250"
    assert env["OPENLEGION_LLM_TIMEOUT_SECONDS"] == "900"


def test_set_llm_limits_env_skips_absent_and_bool():
    env: dict[str, str] = {}
    limits.set_llm_limits_env(env, {})
    assert env == {}
    # bool is an int subclass — must be rejected, not coerced to 1/0.
    limits.set_llm_limits_env(env, {"max_tool_rounds": True})
    assert env == {}


def test_set_llm_limits_env_clamps():
    env: dict[str, str] = {}
    _default, _lo, hi = limits.LIMIT_SPECS["task_max_tool_rounds"]
    limits.set_llm_limits_env(env, {"max_tool_rounds": hi + 5000})
    assert env["OPENLEGION_TASK_MAX_TOOL_ROUNDS"] == str(hi)


# --- edit_agent field support for the per-agent limit knobs (PR B) ---

def test_new_limit_fields_are_editable():
    from src.shared.types import HARD_EDIT_FIELDS
    assert "max_tool_rounds" in HARD_EDIT_FIELDS
    assert "llm_timeout_seconds" in HARD_EDIT_FIELDS


def test_validate_edit_accepts_valid_rounds():
    from src.agent.builtins.operator_tools import _validate_edit
    assert _validate_edit("a", "max_tool_rounds", 250) is None
    assert _validate_edit("a", "llm_timeout_seconds", 900) is None


def test_validate_edit_rejects_out_of_range_and_bool():
    from src.agent.builtins.operator_tools import _validate_edit
    _d, _lo, hi = limits.LIMIT_SPECS["task_max_tool_rounds"]
    assert _validate_edit("a", "max_tool_rounds", hi + 1) is not None
    assert _validate_edit("a", "max_tool_rounds", True) is not None
    assert _validate_edit("a", "llm_timeout_seconds", "nope") is not None
