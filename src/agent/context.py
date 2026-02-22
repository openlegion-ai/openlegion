"""Context window manager — virtual memory for cognition.

Monitors token usage across the conversation and auto-compacts when
the context window fills up. Before compacting, important facts are
flushed to MEMORY.md so nothing is lost.

Write-then-compact: always persist before discarding.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from src.agent.llm import LLMClient
    from src.agent.memory import MemoryStore
    from src.agent.workspace import WorkspaceManager

logger = setup_logging("agent.context")

_FLUSH_THRESHOLD = 0.60
_COMPACT_THRESHOLD = 0.70
_WARNING_THRESHOLD = 0.80

_encoding_cache: dict[str, object | None] = {}

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "openai/gpt-4o": 128_000,
    "openai/gpt-4o-mini": 128_000,
    "openai/gpt-4.1": 1_047_576,
    "openai/gpt-4.1-mini": 1_047_576,
    "openai/gpt-4.1-nano": 1_047_576,
    "openai/o3": 200_000,
    "openai/o3-mini": 200_000,
    "openai/o4-mini": 200_000,
    "anthropic/claude-opus-4-6": 200_000,
    "anthropic/claude-sonnet-4-6": 200_000,
    "anthropic/claude-sonnet-4-5-20250929": 200_000,
    "anthropic/claude-haiku-4-5-20251001": 200_000,
}
_DEFAULT_CONTEXT_WINDOW = 128_000


def _get_tiktoken_encoding(model: str):
    """Get cached tiktoken encoding for an OpenAI model. Returns None if unavailable."""
    if model in _encoding_cache:
        return _encoding_cache[model]
    try:
        import tiktoken
        bare = model.removeprefix("openai/")
        enc = tiktoken.encoding_for_model(bare)
        _encoding_cache[model] = enc
        return enc
    except Exception as e:
        logger.debug("tiktoken encoding unavailable for '%s': %s", model, e)
        _encoding_cache[model] = None
        return None


def estimate_tokens(messages: list[dict], model: str = "") -> int:
    """Estimate token count for a message list.

    - OpenAI models: use tiktoken for accurate counting
    - Anthropic models: ~3.5 chars per token
    - Others/fallback: ~4 chars per token
    """
    if model.startswith("openai/"):
        enc = _get_tiktoken_encoding(model)
        if enc is not None:
            total = 0
            for msg in messages:
                total += 4  # per-message overhead
                for val in msg.values():
                    if isinstance(val, str):
                        total += len(enc.encode(val))
                    elif isinstance(val, list):
                        total += len(enc.encode(json.dumps(val)))
            return total

    chars = sum(len(json.dumps(m)) for m in messages)
    if model.startswith("anthropic/"):
        return int(chars / 3.5)
    return chars // 4


class ContextManager:
    """Monitors and manages the LLM context window.

    Modes:
      safeguard (default) — auto-compact at 70%, emergency prune at 90%
    """

    def __init__(
        self,
        max_tokens: int = 0,
        llm: LLMClient | None = None,
        workspace: WorkspaceManager | None = None,
        memory: MemoryStore | None = None,
        model: str = "",
    ):
        self.model = model
        self.max_tokens = max_tokens or MODEL_CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)
        self.llm = llm
        self.workspace = workspace
        self.memory = memory
        self._flush_triggered = False

    def reset(self) -> None:
        """Reset per-session state for a new conversation."""
        self._flush_triggered = False

    def usage(self, messages: list[dict]) -> float:
        """Return context usage as a fraction (0.0 to 1.0)."""
        return estimate_tokens(messages, model=self.model) / self.max_tokens

    def token_count(self, messages: list[dict]) -> int:
        """Return absolute token count for the message list."""
        return estimate_tokens(messages, model=self.model)

    def context_warning(self, messages: list[dict]) -> str | None:
        """Return a warning string if context usage >= 80%, else None."""
        tokens = self.token_count(messages)
        usage = tokens / self.max_tokens
        if usage >= _WARNING_THRESHOLD:
            pct = int(usage * 100)
            return (
                f"CONTEXT WARNING: Your context is {pct}% full "
                f"({tokens:,}/{self.max_tokens:,} tokens). "
                f"Wrap up your current work or save important facts with memory_save. "
                f"Context will be auto-compacted soon."
            )
        return None

    def should_compact(self, messages: list[dict]) -> bool:
        return self.usage(messages) >= _COMPACT_THRESHOLD

    async def maybe_compact(
        self, system_prompt: str, messages: list[dict],
    ) -> list[dict]:
        """Check context usage and compact if needed.

        1. If between 60-70%, proactively flush structured facts (once).
        2. If above 70%, flush to MEMORY.md, then summarize.
        3. If above 90%, hard-prune oldest messages as emergency fallback.
        """
        usage = self.usage(messages)

        # Proactive flush at 60% — save facts before compaction discards them
        if (
            not self._flush_triggered
            and usage >= _FLUSH_THRESHOLD
            and usage < _COMPACT_THRESHOLD
            and self.llm
            and self.workspace
        ):
            self._flush_triggered = True
            await self._proactive_flush(system_prompt, messages)
            return messages  # don't compact yet

        if not self.should_compact(messages):
            return messages

        usage_pct = int(usage * 100)
        logger.info(f"Context at {usage_pct}% — compacting")

        # Reset so proactive flush can fire again after compaction
        self._flush_triggered = False

        # Step 1: Extract important facts and flush to MEMORY.md + DB
        if self.workspace and self.llm:
            await self._flush_to_memory(system_prompt, messages)

        # Step 2: Summarize and compress
        if self.llm:
            return await self._summarize_compact(system_prompt, messages)

        # Fallback: hard prune if no LLM available
        return self._hard_prune(messages)

    async def _extract_and_store_facts(
        self, messages: list[dict], *, label: str,
    ) -> int:
        """Extract structured facts from messages, store to workspace + memory DB.

        Returns the number of facts extracted (0 on failure or empty).
        Shared by proactive flush (60%) and compaction flush (70%).
        """
        conversation_text = self._messages_to_text(messages)
        if len(conversation_text) < 100:
            return 0

        extract_prompt = (
            "Extract the most important facts, decisions, and user preferences "
            "from this conversation as structured JSON.\n\n"
            "Return a JSON array of objects, each with:\n"
            '  - "key": short identifier (e.g. "user_language_preference")\n'
            '  - "value": the fact content\n'
            '  - "category": one of "preference", "decision", "fact", "context"\n\n'
            "Only include information worth remembering long-term.\n"
            "If there's nothing important, respond with an empty array: []\n\n"
            f"Conversation:\n{conversation_text[:20000]}"
        )

        try:
            response = await self.llm.chat(
                system="You extract structured facts from conversations. Return only valid JSON.",
                messages=[{"role": "user", "content": extract_prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            raw = response.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            facts = json.loads(raw)
            if not isinstance(facts, list) or not facts:
                return 0

            # Write human-readable summary to MEMORY.md
            lines = [f"- **{f['key']}**: {f['value']}" for f in facts if f.get("key") and f.get("value")]
            if lines and self.workspace:
                self.workspace.append_memory(
                    f"\n## {label} ({_now_str()})\n\n" + "\n".join(lines),
                )

            # Store structured facts in memory DB for search
            if self.memory:
                stored = await self.memory.store_facts_batch(facts)
                logger.info(f"{label}: {stored} facts stored to memory DB")

            logger.info(f"{label}: {len(facts)} facts extracted")
            return len(facts)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"{label}: failed to parse LLM response as JSON: {e}")
            return 0
        except Exception as e:
            logger.warning(f"{label} failed: {e}")
            return 0

    async def _proactive_flush(
        self, system_prompt: str, messages: list[dict],
    ) -> None:
        """Extract structured facts at 60% and save to workspace + memory store."""
        try:
            await self._extract_and_store_facts(messages, label="Proactive Flush")
        except Exception as e:
            # Reset so flush can be retried on the next call
            self._flush_triggered = False
            logger.warning(f"Proactive flush failed, will retry: {e}")

    async def _flush_to_memory(
        self, system_prompt: str, messages: list[dict],
    ) -> None:
        """Extract structured facts at 70% and save to workspace + memory store."""
        count = await self._extract_and_store_facts(messages, label="Extracted")
        if count and self.workspace:
            self.workspace.append_daily_log(
                f"Context compacted — {count} facts flushed to memory"
            )

    _SUMMARIZE_RETRIES = 2
    _SUMMARIZE_BACKOFF = 2  # seconds

    async def _summarize_compact(
        self, system_prompt: str, messages: list[dict],
    ) -> list[dict]:
        """Summarize the conversation and replace history with summary + recent."""
        conversation_text = self._messages_to_text(messages[:-4] if len(messages) > 4 else messages)

        summary_prompt = (
            "Summarize this conversation concisely, preserving key context "
            "the assistant needs to continue helpfully. Include: what was discussed, "
            "what actions were taken, what's pending, and any user preferences revealed.\n\n"
            f"Conversation:\n{conversation_text[:20000]}"
        )

        import asyncio as _asyncio

        last_err = None
        for attempt in range(self._SUMMARIZE_RETRIES + 1):
            try:
                response = await self.llm.chat(
                    system="You produce concise conversation summaries.",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=1024,
                    temperature=0.3,
                )
                summary = response.content.strip()
                break
            except Exception as e:
                last_err = e
                if attempt < self._SUMMARIZE_RETRIES:
                    wait = self._SUMMARIZE_BACKOFF * (2 ** attempt)
                    logger.warning(
                        f"Summarization failed (attempt {attempt + 1}/{self._SUMMARIZE_RETRIES + 1}), "
                        f"retrying in {wait}s: {e}"
                    )
                    await _asyncio.sleep(wait)
                else:
                    logger.warning(f"Summarization failed after {self._SUMMARIZE_RETRIES + 1} attempts, "
                                   f"falling back to hard prune: {last_err}")
                    return self._hard_prune(messages)

        summary_msg = {
            "role": "user",
            "content": f"## Conversation Summary (auto-compacted)\n\n{summary}",
        }

        # Check if the message right after the summary would be "user" role,
        # which would create two consecutive user messages (breaking LLM APIs).
        # If so, insert a bridge assistant message and keep 3 recent to stay compact.
        tail_start = -4 if len(messages) > 4 else 0
        needs_bridge = messages[tail_start:] and messages[tail_start].get("role") == "user"
        if needs_bridge:
            recent = messages[-3:] if len(messages) > 3 else messages
            bridge = {"role": "assistant", "content": "Understood, continuing from the summary above."}
            result = [summary_msg, bridge] + recent
        else:
            recent = messages[-4:] if len(messages) > 4 else messages
            result = [summary_msg] + recent
        logger.info(
            f"Compacted {len(messages)} messages -> {len(result)} "
            f"(usage: {int(self.usage(result) * 100)}%)"
        )
        return result

    def _hard_prune(self, messages: list[dict]) -> list[dict]:
        """Emergency: keep first message and last N messages (group-aware).

        Groups tool_calls with their tool results so we never orphan a
        tool_call without its matching tool responses.
        """
        if len(messages) <= 8:
            return messages

        # Build groups: assistant(tool_calls) + following tool messages stay together
        groups: list[list[dict]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                group = [msg]
                i += 1
                while i < len(messages) and messages[i].get("role") == "tool":
                    group.append(messages[i])
                    i += 1
                groups.append(group)
            else:
                groups.append([msg])
                i += 1

        if len(groups) <= 5:
            return messages

        # Keep first group + last 4 groups
        kept = groups[:1] + groups[-4:]
        pruned = [msg for group in kept for msg in group]
        logger.warning(f"Hard-pruned {len(messages)} -> {len(pruned)} messages (group-aware)")
        return pruned

    @staticmethod
    def _messages_to_text(messages: list[dict]) -> str:
        """Convert messages to readable text for summarization."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "tool":
                content = content[:200]
            if content:
                parts.append(f"[{role}]: {content}")
        return "\n\n".join(parts)


def _now_str() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
