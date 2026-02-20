"""Provider-specific transcript sanitization.

Cleans up tool-call IDs and orphaned tool messages before sending
transcripts to LLM providers via litellm.  Operates on copies —
the agent's original transcript is never mutated.

LiteLLM handles message alternation and same-role merging, but does
NOT handle:
  - Tool ID format constraints (Gemini: alphanumeric-only, Mistral: 9-char)
  - Orphaned tool results / tool calls with no matching counterpart
"""

from __future__ import annotations

import json
import uuid


def sanitize_for_provider(messages: list[dict], model: str) -> list[dict]:
    """Public entry point. Return a sanitized copy of *messages* for *model*."""
    provider = _detect_provider(model)
    msgs = _deep_copy_messages(messages)

    # 1. Orphaned tool call/result cleanup (all providers)
    msgs = _ensure_tool_call_pairing(msgs)

    # 2. Tool ID format (Gemini/Mistral only)
    if provider in ("gemini", "mistral"):
        msgs = _remap_tool_ids(msgs, provider)

    return msgs


# ── Internal helpers ───────────────────────────────────────────


def _detect_provider(model: str) -> str:
    """Extract provider from a ``provider/model`` string.

    Falls back to prefix heuristics for bare model names (e.g. ``gpt-4o``).
    """
    if "/" in model:
        return model.split("/", 1)[0]

    # Bare-name heuristics — mirrors credentials.py:165-178
    bare_prefixes: dict[str, str] = {
        "gpt-": "openai",
        "o1": "openai",
        "o3": "openai",
        "o4": "openai",
        "text-embedding-": "openai",
        "claude-": "anthropic",
    }
    for prefix, provider in bare_prefixes.items():
        if model.startswith(prefix):
            return provider

    return "unknown"


def _deep_copy_messages(messages: list[dict]) -> list[dict]:
    """Fast, lossless deep-copy for JSON-serialisable data."""
    return json.loads(json.dumps(messages))


def _remap_tool_ids(messages: list[dict], provider: str) -> list[dict]:
    """Replace tool-call IDs with provider-compliant ones.

    Gemini: 12-char hex (alphanumeric only, no underscores/hyphens).
    Mistral: 9-char hex (alphanumeric, max 9 chars).
    """
    length = 12 if provider == "gemini" else 9
    id_map: dict[str, str] = {}

    def _get_new_id(old_id: str) -> str:
        if old_id not in id_map:
            id_map[old_id] = uuid.uuid4().hex[:length]
        return id_map[old_id]

    for msg in messages:
        # Remap IDs in assistant tool_calls
        for tc in msg.get("tool_calls") or []:
            old_id = tc.get("id")
            if old_id:
                tc["id"] = _get_new_id(old_id)

        # Remap IDs in tool result messages
        old_tc_id = msg.get("tool_call_id")
        if old_tc_id:
            msg["tool_call_id"] = _get_new_id(old_tc_id)

    return messages


def _ensure_tool_call_pairing(messages: list[dict]) -> list[dict]:
    """Fix orphaned tool calls and tool results.

    - Remove tool-role messages whose ``tool_call_id`` has no matching
      ``tool_calls[].id`` in any prior assistant message.
    - Append a placeholder ``"[no result]"`` tool message for any
      assistant tool_call whose ID never appears in a subsequent tool message.
    """
    # Collect all tool_call IDs emitted by assistant messages
    call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                tc_id = tc.get("id")
                if tc_id:
                    call_ids.add(tc_id)

    # Collect all tool_call_ids referenced by tool-role messages
    result_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id:
                result_ids.add(tc_id)

    # 1. Remove orphaned tool results (no matching call)
    cleaned: list[dict] = []
    for msg in messages:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id and tc_id not in call_ids:
                continue  # orphaned result — drop it
        cleaned.append(msg)

    # 2. Add placeholder results for orphaned calls
    orphaned_calls = call_ids - result_ids
    if orphaned_calls:
        result: list[dict] = []
        i = 0
        while i < len(cleaned):
            msg = cleaned[i]
            result.append(msg)

            if msg.get("role") == "assistant":
                msg_orphans = []
                for tc in msg.get("tool_calls") or []:
                    tc_id = tc.get("id")
                    if tc_id and tc_id in orphaned_calls:
                        msg_orphans.append(tc)

                if msg_orphans:
                    # Skip past any existing tool results
                    i += 1
                    while i < len(cleaned) and cleaned[i].get("role") == "tool":
                        result.append(cleaned[i])
                        i += 1
                    # Append placeholders for orphaned calls
                    for tc in msg_orphans:
                        result.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": "[no result]",
                        })
                    continue  # already advanced i

            i += 1
        return result

    return cleaned
