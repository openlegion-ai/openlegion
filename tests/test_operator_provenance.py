"""Tests for message provenance tagging."""
from src.agent.loop import _last_message_is_user_origin


def test_user_origin_detected():
    messages = [
        {"role": "user", "content": "hello", "_origin": "user"},
        {"role": "assistant", "content": "hi"},
    ]
    assert _last_message_is_user_origin(messages) is True


def test_heartbeat_origin_not_user():
    messages = [
        {"role": "user", "content": "check health", "_origin": "system:heartbeat"},
    ]
    assert _last_message_is_user_origin(messages) is False


def test_agent_origin_not_user():
    messages = [
        {"role": "user", "content": "result", "_origin": "agent:writer"},
    ]
    assert _last_message_is_user_origin(messages) is False


def test_no_origin_tag_defaults_to_user():
    """Legacy messages without _origin are treated as user-originated (backward compat)."""
    messages = [
        {"role": "user", "content": "hello"},
    ]
    assert _last_message_is_user_origin(messages) is True


def test_empty_messages():
    assert _last_message_is_user_origin([]) is False


def test_most_recent_user_message_checked():
    """Should check the most recent user message, not the first."""
    messages = [
        {"role": "user", "content": "first", "_origin": "user"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "second", "_origin": "system:heartbeat"},
    ]
    assert _last_message_is_user_origin(messages) is False


def test_skips_non_user_roles():
    """Should skip assistant and tool messages when searching for user origin."""
    messages = [
        {"role": "user", "content": "hello", "_origin": "user"},
        {"role": "assistant", "content": "response"},
        {"role": "tool", "content": "result", "tool_call_id": "tc_1"},
        {"role": "assistant", "content": "done"},
    ]
    assert _last_message_is_user_origin(messages) is True


def test_origin_stripped_by_sanitize_for_provider():
    """Verify _origin is stripped before messages reach the LLM API."""
    from src.host.transcript import sanitize_for_provider

    messages = [
        {"role": "user", "content": "hello", "_origin": "user"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "heartbeat", "_origin": "system:heartbeat"},
    ]
    sanitized = sanitize_for_provider(messages, "openai/gpt-4o")
    for msg in sanitized:
        assert "_origin" not in msg, f"_origin not stripped from {msg}"


def test_origin_preserved_in_original_after_sanitize():
    """Verify sanitize_for_provider doesn't mutate the original messages."""
    from src.host.transcript import sanitize_for_provider

    messages = [
        {"role": "user", "content": "hello", "_origin": "user"},
    ]
    sanitize_for_provider(messages, "openai/gpt-4o")
    assert messages[0].get("_origin") == "user"
