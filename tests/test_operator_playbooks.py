"""Tests for operator playbook detection, state tracking, and content assembly."""
from __future__ import annotations

import pytest

from src.cli.operator_playbooks import (
    PLAYBOOK_STICKY_TURNS,
    _OPERATOR_CORE,
    _PLAYBOOK_CREDENTIALS,
    _PLAYBOOK_EDIT,
    _PLAYBOOK_MONITOR,
    _PLAYBOOK_TEAM_BUILD,
    _TOOL_PLAYBOOK_MAP,
    extract_triggered_playbooks,
    get_playbook_content,
)


class TestExtractTriggeredPlaybooks:
    """Test tool-call scanning for playbook triggers."""

    def test_empty_messages(self):
        assert extract_triggered_playbooks([]) == set()

    def test_no_tool_calls(self):
        msgs = [
            {"role": "user", "content": "Build me a team"},
            {"role": "assistant", "content": "I'll help with that"},
        ]
        assert extract_triggered_playbooks(msgs) == set()

    def test_single_trigger(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "create_agent"}, "id": "tc1", "type": "function"}
            ]},
        ]
        assert extract_triggered_playbooks(msgs) == {"team_build"}

    def test_multiple_triggers_same_playbook(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "create_agent"}, "id": "tc1", "type": "function"},
                {"function": {"name": "create_project"}, "id": "tc2", "type": "function"},
            ]},
        ]
        assert extract_triggered_playbooks(msgs) == {"team_build"}

    def test_multiple_playbooks(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "create_agent"}, "id": "tc1", "type": "function"},
            ]},
            {"role": "tool", "content": "{}", "tool_call_id": "tc1"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "propose_edit"}, "id": "tc2", "type": "function"},
            ]},
        ]
        assert extract_triggered_playbooks(msgs) == {"team_build", "edit"}

    def test_non_mapped_tools_ignored(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "list_agents"}, "id": "tc1", "type": "function"},
                {"function": {"name": "get_system_status"}, "id": "tc2", "type": "function"},
            ]},
        ]
        assert extract_triggered_playbooks(msgs) == set()

    def test_all_playbook_triggers(self):
        """Every tool in _TOOL_PLAYBOOK_MAP triggers the expected playbook."""
        for tool_name, expected_pb in _TOOL_PLAYBOOK_MAP.items():
            msgs = [{"role": "assistant", "tool_calls": [
                {"function": {"name": tool_name}, "id": "tc1", "type": "function"}
            ]}]
            result = extract_triggered_playbooks(msgs)
            assert expected_pb in result, f"{tool_name} should trigger {expected_pb}"


class TestGetPlaybookContent:
    """Test playbook content assembly."""

    def test_empty_list(self):
        assert get_playbook_content([]) == ""

    def test_single_playbook(self):
        content = get_playbook_content(["team_build"])
        assert "Team Setup" in content
        assert "create_agent" in content

    def test_two_playbooks(self):
        content = get_playbook_content(["team_build", "credentials"])
        assert "Team Setup" in content
        assert "Credential Setup" in content

    def test_max_two_playbooks(self):
        content = get_playbook_content(["team_build", "edit", "monitor"])
        # Should only include first two
        assert "Team Setup" in content
        assert "Configuration Edit" in content
        assert "Fleet Monitoring" not in content

    def test_invalid_playbook_key(self):
        content = get_playbook_content(["nonexistent"])
        assert content == ""

    def test_all_playbooks_have_content(self):
        for key in ["team_build", "edit", "monitor", "credentials"]:
            content = get_playbook_content([key])
            assert len(content) > 100, f"Playbook {key} should have substantial content"


class TestPlaybookConstants:
    """Verify playbook content quality."""

    def test_core_has_sentinel(self):
        assert "<!-- playbook_v2 -->" in _OPERATOR_CORE

    def test_core_size_reasonable(self):
        # Core should be significantly smaller than old monolithic instructions (7800 chars)
        assert len(_OPERATOR_CORE) < 5000
        assert len(_OPERATOR_CORE) > 1000

    def test_playbooks_have_numbered_steps(self):
        for name, content in [
            ("team_build", _PLAYBOOK_TEAM_BUILD),
            ("edit", _PLAYBOOK_EDIT),
            ("monitor", _PLAYBOOK_MONITOR),
            ("credentials", _PLAYBOOK_CREDENTIALS),
        ]:
            assert "1." in content, f"{name} should have numbered steps"

    def test_tool_map_covers_all_action_tools(self):
        # All action tools should be mapped
        expected_tools = {
            "create_agent", "apply_template", "create_project",
            "add_agents_to_project", "remove_agents_from_project",
            "update_project_context", "propose_edit", "confirm_edit",
            "read_agent_history", "save_observations",
            "request_credential", "vault_list",
        }
        assert set(_TOOL_PLAYBOOK_MAP.keys()) == expected_tools

    def test_sticky_turns_reasonable(self):
        assert 3 <= PLAYBOOK_STICKY_TURNS <= 10

    def test_playbooks_have_key_tools(self):
        """Each playbook references the tools it guides the operator to use."""
        assert "propose_edit" in _PLAYBOOK_TEAM_BUILD
        assert "confirm_edit" in _PLAYBOOK_TEAM_BUILD
        assert "create_project" in _PLAYBOOK_TEAM_BUILD
        assert "create_agent" in _PLAYBOOK_TEAM_BUILD
        assert "apply_template" in _PLAYBOOK_TEAM_BUILD
        assert "add_agents_to_project" in _PLAYBOOK_TEAM_BUILD
        assert "update_project_context" in _PLAYBOOK_TEAM_BUILD
        assert "vault_list" in _PLAYBOOK_TEAM_BUILD
        assert "request_credential" in _PLAYBOOK_TEAM_BUILD

        assert "propose_edit" in _PLAYBOOK_EDIT
        assert "confirm_edit" in _PLAYBOOK_EDIT

        assert "check_inbox" in _PLAYBOOK_MONITOR
        assert "get_system_status" in _PLAYBOOK_MONITOR
        assert "save_observations" in _PLAYBOOK_MONITOR

        assert "vault_list" in _PLAYBOOK_CREDENTIALS
        assert "request_credential" in _PLAYBOOK_CREDENTIALS

    def test_core_has_key_sections(self):
        """Core instructions contain all expected section headers."""
        for section in [
            "## Core Approach",
            "## Plan Limits",
            "## Assessment",
            "## Routing Work",
            "## Workflow Overview",
            "## Proactive Improvement",
            "## Tool Errors",
        ]:
            assert section in _OPERATOR_CORE, f"Missing section: {section}"
