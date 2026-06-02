"""Tests for demoting agent-authored Python tools behind an opt-in (Task 6)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.agent.builtins import tool_authoring
from src.agent.builtins.tool_authoring import create_tool, tool_authoring_enabled
from src.agent.loop import _TOOL_AUTHORING_TOOLS
from tests.test_loop import _make_loop

ENV = tool_authoring.TOOL_AUTHORING_ENABLED_ENV


# ── flag parsing ──────────────────────────────────────────────────────────

def test_enabled_flag_truthy(monkeypatch):
    for v in ("1", "true", "TRUE", "yes", "on", " On "):
        monkeypatch.setenv(ENV, v)
        assert tool_authoring_enabled() is True


def test_enabled_flag_falsy(monkeypatch):
    for v in ("", "0", "false", "no", "off", "nope"):
        monkeypatch.setenv(ENV, v)
        assert tool_authoring_enabled() is False


def test_disabled_when_unset(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert tool_authoring_enabled() is False


# ── create_tool self-gate ─────────────────────────────────────────────────

def test_create_tool_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    result = create_tool("x", "code", workspace_manager=MagicMock())
    assert result["error"] == "tool_authoring_disabled"
    # Points the agent at the Skills path.
    assert "skills_list" in result["detail"]
    assert "install_skill" in result["detail"]


def test_create_tool_gate_opens_with_flag(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    # Past the gate now → falls through to the normal workspace_manager check.
    result = create_tool("x", "code", workspace_manager=None)
    assert result["error"] == "No workspace_manager available"


# ── worker tool-surface exclusion ─────────────────────────────────────────

def test_worker_hides_authoring_tools_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    loop = _make_loop()
    assert loop._excluded_tools is not None
    assert _TOOL_AUTHORING_TOOLS <= loop._excluded_tools


def test_worker_exposes_authoring_tools_when_enabled(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    loop = _make_loop()
    # Non-standalone worker with authoring on → no authoring exclusions.
    assert not loop._excluded_tools or not (_TOOL_AUTHORING_TOOLS & loop._excluded_tools)
