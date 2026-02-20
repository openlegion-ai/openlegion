"""Tests for provider-specific transcript sanitization."""

import copy
import re

from src.host.transcript import (
    _detect_provider,
    _ensure_tool_call_pairing,
    _remap_tool_ids,
    sanitize_for_provider,
)


# ── _detect_provider ───────────────────────────────────────────


class TestDetectProvider:
    def test_slash_format(self):
        assert _detect_provider("gemini/gemini-1.5-pro") == "gemini"
        assert _detect_provider("anthropic/claude-sonnet-4-5-20250929") == "anthropic"
        assert _detect_provider("openai/gpt-4o") == "openai"
        assert _detect_provider("mistral/mistral-large") == "mistral"

    def test_bare_openai_names(self):
        assert _detect_provider("gpt-4o") == "openai"
        assert _detect_provider("gpt-4o-mini") == "openai"
        assert _detect_provider("o1") == "openai"
        assert _detect_provider("o3-mini") == "openai"
        assert _detect_provider("o4-mini") == "openai"

    def test_bare_anthropic_names(self):
        assert _detect_provider("claude-sonnet-4-5-20250929") == "anthropic"
        assert _detect_provider("claude-haiku-4-5-20251001") == "anthropic"

    def test_bare_embedding(self):
        assert _detect_provider("text-embedding-3-small") == "openai"

    def test_unknown(self):
        assert _detect_provider("some-random-model") == "unknown"

    def test_empty(self):
        assert _detect_provider("") == "unknown"


# ── _remap_tool_ids ───────────────────────────────────────────


class TestRemapToolIds:
    def _make_messages(self):
        return [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_tc_a1b2c3d4e5f6", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call_tc_x9y8z7w6v5u4", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_tc_a1b2c3d4e5f6", "content": "result1"},
            {"role": "tool", "tool_call_id": "call_tc_x9y8z7w6v5u4", "content": "result2"},
            {"role": "assistant", "content": "done"},
        ]

    def test_gemini_ids_are_12_char_alphanumeric(self):
        msgs = self._make_messages()
        result = _remap_tool_ids(msgs, "gemini")

        for msg in result:
            for tc in msg.get("tool_calls") or []:
                assert re.fullmatch(r"[a-f0-9]{12}", tc["id"]), f"Bad Gemini ID: {tc['id']}"
            if msg.get("tool_call_id"):
                assert re.fullmatch(r"[a-f0-9]{12}", msg["tool_call_id"])

    def test_mistral_ids_are_9_char(self):
        msgs = self._make_messages()
        result = _remap_tool_ids(msgs, "mistral")

        for msg in result:
            for tc in msg.get("tool_calls") or []:
                assert re.fullmatch(r"[a-f0-9]{9}", tc["id"]), f"Bad Mistral ID: {tc['id']}"
            if msg.get("tool_call_id"):
                assert re.fullmatch(r"[a-f0-9]{9}", msg["tool_call_id"])

    def test_pairing_preserved(self):
        """tool_call ID and its matching tool_call_id get the same new ID."""
        msgs = self._make_messages()
        result = _remap_tool_ids(msgs, "gemini")

        call_ids = [tc["id"] for tc in result[1]["tool_calls"]]
        result_ids = [result[2]["tool_call_id"], result[3]["tool_call_id"]]
        assert call_ids == result_ids

    def test_no_tool_calls_passthrough(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = _remap_tool_ids(msgs, "gemini")
        assert result == msgs

    def test_no_underscores_in_gemini_ids(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_tc_with_underscores", "type": "function", "function": {"name": "t", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_tc_with_underscores", "content": "ok"},
        ]
        result = _remap_tool_ids(msgs, "gemini")
        new_id = result[0]["tool_calls"][0]["id"]
        assert "_" not in new_id
        assert "-" not in new_id


# ── _ensure_tool_call_pairing ──────────────────────────────────


class TestEnsureToolCallPairing:
    def test_orphaned_result_removed(self):
        """Tool result with no matching tool_call in history is dropped."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "nonexistent_call", "content": "orphan"},
            {"role": "assistant", "content": "ok"},
        ]
        result = _ensure_tool_call_pairing(msgs)
        assert len(result) == 2
        assert all(m.get("role") != "tool" for m in result)

    def test_orphaned_call_gets_placeholder(self):
        """Tool call with no result gets a [no result] placeholder."""
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_001", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            # No tool result for call_001
            {"role": "user", "content": "what happened?"},
        ]
        result = _ensure_tool_call_pairing(msgs)
        # Should have: user, assistant, tool(placeholder), user
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_001"
        assert tool_msgs[0]["content"] == "[no result]"

    def test_properly_paired_unchanged(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_001", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_001", "content": "found it"},
            {"role": "assistant", "content": "done"},
        ]
        result = _ensure_tool_call_pairing(msgs)
        assert len(result) == 4
        assert result[2]["tool_call_id"] == "call_001"
        assert result[2]["content"] == "found it"

    def test_partial_pairing(self):
        """One call has a result, another doesn't — only the missing one gets placeholder."""
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "result_a"},
            # No result for call_b
            {"role": "assistant", "content": "next"},
        ]
        result = _ensure_tool_call_pairing(msgs)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        ids = {m["tool_call_id"] for m in tool_msgs}
        assert ids == {"call_a", "call_b"}
        placeholder = [m for m in tool_msgs if m["tool_call_id"] == "call_b"][0]
        assert placeholder["content"] == "[no result]"

    def test_multiple_assistant_messages_with_orphaned_calls(self):
        """Orphaned calls across separate assistant messages each get placeholders."""
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            # No result for call_1
            {"role": "user", "content": "continue"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_2", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                ],
            },
            # No result for call_2
            {"role": "user", "content": "done?"},
        ]
        result = _ensure_tool_call_pairing(msgs)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert tool_msgs[0]["content"] == "[no result]"
        assert tool_msgs[1]["tool_call_id"] == "call_2"
        assert tool_msgs[1]["content"] == "[no result]"
        # Verify ordering: placeholder inserted after each assistant, before next user
        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "tool", "user", "assistant", "tool", "user"]

    def test_combined_orphaned_result_and_orphaned_call(self):
        """Orphaned result dropped AND orphaned call gets placeholder in same transcript."""
        msgs = [
            {"role": "user", "content": "hi"},
            # Orphaned tool result (no matching assistant tool_call)
            {"role": "tool", "tool_call_id": "ghost_id", "content": "stale result"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_x", "type": "function", "function": {"name": "exec", "arguments": "{}"}},
                ],
            },
            # No tool result for call_x
            {"role": "assistant", "content": "done"},
        ]
        result = _ensure_tool_call_pairing(msgs)
        # ghost_id result should be removed, call_x should get placeholder
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_x"
        assert tool_msgs[0]["content"] == "[no result]"
        # ghost_id completely gone
        all_tc_ids = [m.get("tool_call_id") for m in result if m.get("tool_call_id")]
        assert "ghost_id" not in all_tc_ids


# ── sanitize_for_provider (integration) ────────────────────────


class TestSanitizeForProvider:
    def _make_full_transcript(self):
        return [
            {"role": "user", "content": "search for cats"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_tc_a1b2c3d4e5f6", "type": "function", "function": {"name": "web_search", "arguments": '{"q": "cats"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_tc_a1b2c3d4e5f6", "content": "cats are great"},
            {"role": "assistant", "content": "I found info about cats."},
        ]

    def test_gemini_full_pipeline(self):
        msgs = self._make_full_transcript()
        result = sanitize_for_provider(msgs, "gemini/gemini-1.5-pro")
        # IDs should be 12-char hex
        tc_id = result[1]["tool_calls"][0]["id"]
        assert re.fullmatch(r"[a-f0-9]{12}", tc_id)
        # Pairing preserved
        assert result[2]["tool_call_id"] == tc_id

    def test_mistral_full_pipeline(self):
        msgs = self._make_full_transcript()
        result = sanitize_for_provider(msgs, "mistral/mistral-large")
        tc_id = result[1]["tool_calls"][0]["id"]
        assert re.fullmatch(r"[a-f0-9]{9}", tc_id)
        assert result[2]["tool_call_id"] == tc_id

    def test_openai_minimal_changes(self):
        """OpenAI messages get orphan cleanup but IDs stay unchanged."""
        msgs = self._make_full_transcript()
        result = sanitize_for_provider(msgs, "openai/gpt-4o")
        assert result[1]["tool_calls"][0]["id"] == "call_tc_a1b2c3d4e5f6"
        assert result[2]["tool_call_id"] == "call_tc_a1b2c3d4e5f6"

    def test_anthropic_minimal_changes(self):
        msgs = self._make_full_transcript()
        result = sanitize_for_provider(msgs, "anthropic/claude-sonnet-4-5-20250929")
        assert result[1]["tool_calls"][0]["id"] == "call_tc_a1b2c3d4e5f6"

    def test_empty_messages(self):
        result = sanitize_for_provider([], "gemini/gemini-1.5-pro")
        assert result == []

    def test_idempotency(self):
        """Running sanitize twice produces the same structure."""
        msgs = self._make_full_transcript()
        first = sanitize_for_provider(msgs, "gemini/gemini-1.5-pro")
        second = sanitize_for_provider(first, "gemini/gemini-1.5-pro")
        # Same structure, though IDs differ (re-randomized)
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert a["role"] == b["role"]

    def test_original_not_mutated(self):
        """The input list and its dicts must not be modified."""
        msgs = self._make_full_transcript()
        original = copy.deepcopy(msgs)
        sanitize_for_provider(msgs, "gemini/gemini-1.5-pro")
        assert msgs == original
