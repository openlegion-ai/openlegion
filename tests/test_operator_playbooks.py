"""Tests for operator playbook detection, state tracking, and content assembly."""

from __future__ import annotations

import pytest  # noqa: F401

from src.shared.operator_playbooks import (
    _OPERATOR_CORE,
    _OPERATOR_GREETING,
    _PLAYBOOK_CREDENTIALS,
    _PLAYBOOK_EDIT,
    _PLAYBOOK_MONITOR,
    _PLAYBOOK_TEAM_BUILD,
    _TOOL_PLAYBOOK_MAP,
    PLAYBOOK_STICKY_TURNS,
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
                {"function": {"name": "create_team"}, "id": "tc2", "type": "function"},
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
                {"function": {"name": "edit_agent"}, "id": "tc2", "type": "function"},
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

    def test_core_has_delegate_and_release(self):
        # Phase 1d: the operator must hand off and release the turn, not
        # block-watch a pipeline (the chain watcher delivers the outcome).
        assert "Delegate and release" in _OPERATOR_CORE
        assert "await_task_event to babysit" in _OPERATOR_CORE

    def test_core_size_reasonable(self):
        # Core should be significantly smaller than old monolithic instructions (7800 chars).
        # Bumped 5200 → 6000 to accommodate the autonomous-by-default
        # frame copy (immediate-apply edits, ratings-as-feedback guidance);
        # 6000 → 6400 for the Phase-1d "delegate and release" routing rule
        # (hand off → stop; the chain watcher delivers the terminal outcome).
        assert len(_OPERATOR_CORE) < 6400
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
        # The approval-workflow redesign dropped propose_edit/confirm_edit
        # from the playbook map: edit_agent is the single immediate-apply
        # path (5-min undo for soft fields, 30-min for hard). Phase 1 of
        # the back-compat cleanup then deleted both propose_edit and
        # confirm_edit entirely (no more dangling registrations).
        # The legacy ``*_project`` aliases in ``_TOOL_PLAYBOOK_MAP`` were
        # also dropped — only the canonical ``*_team`` names trigger the
        # build playbook now.
        keys = set(_TOOL_PLAYBOOK_MAP.keys())
        # Every canonical *_team tool must be registered.
        for canonical in {
            "create_team", "add_agents_to_team",
            "remove_agents_from_team", "update_team_context",
            "set_team_goal",
        }:
            assert canonical in keys, f"team-named tool {canonical} missing from playbook map"
        # Non-domain tools that are unrelated to the rename must remain.
        for shared in {
            "create_agent", "apply_template", "edit_agent",
            "undo_change",
            "request_credential", "request_browser_login",
        }:
            assert shared in keys, f"core tool {shared} missing from playbook map"
        # Phase 1 deletions — the legacy *_project aliases must NOT be
        # present anymore.
        for legacy in {
            "create_project", "add_agents_to_project",
            "remove_agents_from_project", "update_project_context",
            "set_project_goal", "propose_edit", "confirm_edit",
        }:
            assert legacy not in keys, f"deleted shim {legacy} still in playbook map"

    def test_sticky_turns_reasonable(self):
        assert 3 <= PLAYBOOK_STICKY_TURNS <= 10

    def test_playbooks_have_key_tools(self):
        """Each playbook references the tools it guides the operator to use."""
        # PR 1 — edit_agent replaces the legacy propose_edit/confirm_edit
        # guidance in team_build; the edit playbook references edit_agent
        # (the single immediate-apply tool with field-aware undo TTLs).
        assert "edit_agent" in _PLAYBOOK_TEAM_BUILD
        # Project→team rename PR 2 flipped the playbook prose to the
        # canonical team-named tools. The legacy ``*_project`` names
        # remain callable (and remain in ``_TOOL_PLAYBOOK_MAP``) but the
        # prose itself nudges the operator toward the new names.
        assert "create_team" in _PLAYBOOK_TEAM_BUILD
        assert "create_agent" in _PLAYBOOK_TEAM_BUILD
        assert "apply_template" in _PLAYBOOK_TEAM_BUILD
        assert "add_agents_to_team" in _PLAYBOOK_TEAM_BUILD
        # PR 5: the operator should proactively save the goal as a north star.
        assert "set_team_goal" in _PLAYBOOK_TEAM_BUILD
        assert "north star" in _PLAYBOOK_TEAM_BUILD.lower()
        assert "request_credential" in _PLAYBOOK_TEAM_BUILD
        assert "request_browser_login" in _PLAYBOOK_TEAM_BUILD

        assert "edit_agent" in _PLAYBOOK_EDIT
        # New guidance for the act-and-undo posture must be present.
        # (Phase 1 deleted confirm_edit — edits apply immediately and
        # the playbook no longer references the retired stub.)
        assert "Undo" in _PLAYBOOK_EDIT
        assert "soft" in _PLAYBOOK_EDIT.lower()
        assert "hard" in _PLAYBOOK_EDIT.lower()

        assert "check_inbox" in _PLAYBOOK_MONITOR
        assert "get_system_status" in _PLAYBOOK_MONITOR
        assert "inspect_agents" in _PLAYBOOK_MONITOR

        assert "request_credential" in _PLAYBOOK_CREDENTIALS
        assert "request_browser_login" in _PLAYBOOK_CREDENTIALS

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


class TestActionChipsInstruction:
    """Phase 3 — operator response chips contract.

    The operator's prompt must instruct the LLM to end every response
    with 2-4 ``ACTION: <label>`` lines so the dashboard can render them
    as clickable suggestion chips below the bubble. The greeting also
    seeds an example block so the very first conversation doesn't ship
    without chips.
    """

    def test_core_mentions_action_format(self):
        """``_OPERATOR_CORE`` documents the ACTION line contract."""
        assert "ACTION:" in _OPERATOR_CORE
        # Reference both the format and what the dashboard does with it,
        # so a future edit doesn't accidentally drop the rendering hint.
        assert "Suggested next steps" in _OPERATOR_CORE
        assert "chips" in _OPERATOR_CORE.lower()

    def test_core_specifies_label_constraints(self):
        """Length cap and "user might want to do next" framing present."""
        assert "≤40" in _OPERATOR_CORE or "<=40" in _OPERATOR_CORE or "40 char" in _OPERATOR_CORE
        # Encourage user-facing labels; otherwise the LLM emits engineer
        # phrases like "call inspect_agents".
        assert "user-facing" in _OPERATOR_CORE.lower() or "user might want" in _OPERATOR_CORE.lower()

    def test_core_specifies_at_end(self):
        """ACTION block must appear at the very end of the response."""
        assert "end" in _OPERATOR_CORE.lower()

    def test_operator_greeting_no_action_lines(self):
        """Bootstrap greeting must NOT carry ACTION: chip lines.

        Before fix #3 the greeting ended with four ACTION: lines that
        matched the wizard ``ask`` card chips, which caused TWO chip
        rows to render on cold start (one parsed from the bootstrap
        greeting, one from the wizard). The wizard now owns the chips;
        the greeting is prose-only.
        """
        for line in _OPERATOR_GREETING.splitlines():
            assert not line.strip().upper().startswith("ACTION:"), (
                f"Greeting must not contain ACTION lines (chips owned "
                f"by wizard): {line!r}"
            )


class TestTaskTitleQualityGuidance:
    """The Routing Work playbook section must steer the operator toward
    SHORT hand_off summaries.

    Symptom that drove this: the Operator agent was emitting hand_off
    calls where the ``summary`` carried a full instruction (250+ chars)
    — that string became both the task title and description, so the
    dashboard kanban rendered as wall-of-text. The prompt fix here is
    paired with server-side defensive splitting in ``Tasks.create``.
    """

    def test_core_mentions_short_titles(self):
        """``_OPERATOR_CORE`` instructs the operator to keep summaries short."""
        # The guidance must reference both the cap and the alternative
        # location for long content (the ``data`` field), otherwise the
        # LLM may just truncate aggressively and lose context.
        core_lower = _OPERATOR_CORE.lower()
        assert "short" in core_lower
        assert "summary" in core_lower
        # Either the explicit char cap OR the "title" framing must be
        # there — both means the LLM has redundant signal.
        assert "80" in _OPERATOR_CORE or "title" in core_lower
        assert "data" in core_lower or "description" in core_lower

    def test_core_has_concrete_handoff_example(self):
        """A bad-vs-good example anchors the rule in pattern-matching."""
        assert "Draft" in _OPERATOR_CORE  # the good example
        assert "TEST RUN" in _OPERATOR_CORE  # the bad example
