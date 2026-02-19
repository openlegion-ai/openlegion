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
_PRUNE_THRESHOLD = 0.90


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~4 chars per token."""
    return sum(len(json.dumps(m)) for m in messages) // 4


class ContextManager:
    """Monitors and manages the LLM context window.

    Modes:
      safeguard (default) — auto-compact at 70%, emergency prune at 90%
    """

    def __init__(
        self,
        max_tokens: int = 128_000,
        llm: LLMClient | None = None,
        workspace: WorkspaceManager | None = None,
        memory: MemoryStore | None = None,
    ):
        self.max_tokens = max_tokens
        self.llm = llm
        self.workspace = workspace
        self.memory = memory
        self._flush_triggered = False

    def usage(self, messages: list[dict]) -> float:
        """Return context usage as a fraction (0.0 to 1.0)."""
        return estimate_tokens(messages) / self.max_tokens

    def should_compact(self, messages: list[dict]) -> bool:
        return self.usage(messages) >= _COMPACT_THRESHOLD

    def _should_prune(self, messages: list[dict]) -> bool:
        return self.usage(messages) >= _PRUNE_THRESHOLD

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

        # Step 1: Extract important facts and flush to MEMORY.md
        if self.workspace and self.llm:
            await self._flush_to_memory(system_prompt, messages)

        # Step 2: Summarize and compress
        if self.llm:
            return await self._summarize_compact(system_prompt, messages)

        # Fallback: hard prune if no LLM available
        return self._hard_prune(messages)

    async def _proactive_flush(
        self, system_prompt: str, messages: list[dict],
    ) -> None:
        """Extract structured facts at 60% and save to workspace + memory store.

        Runs once per conversation (guarded by _flush_triggered). Extracts
        facts as JSON so they're searchable in the memory DB, not just
        appended as free text to MEMORY.md.
        """
        conversation_text = self._messages_to_text(messages)
        if len(conversation_text) < 100:
            return

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
                return

            # Write human-readable summary to MEMORY.md
            lines = [f"- **{f['key']}**: {f['value']}" for f in facts if "key" in f and "value" in f]
            if lines:
                self.workspace.append_memory(
                    f"\n## Proactive Flush ({_now_str()})\n\n" + "\n".join(lines),
                )

            # Store structured facts in memory DB for search
            if self.memory:
                stored = await self.memory.store_facts_batch(facts)
                logger.info(f"Proactive flush: {stored} facts stored to memory DB")

            logger.info(f"Proactive flush: {len(facts)} facts extracted at 60% context")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Proactive flush: failed to parse LLM response as JSON: {e}")
        except Exception as e:
            logger.warning(f"Proactive flush failed: {e}")

    async def _flush_to_memory(
        self, system_prompt: str, messages: list[dict],
    ) -> None:
        """Ask the LLM to extract important facts, write them to MEMORY.md."""
        conversation_text = self._messages_to_text(messages)
        if len(conversation_text) < 100:
            return

        extract_prompt = (
            "Extract the most important facts, decisions, and user preferences "
            "from this conversation. Output a concise bullet list in Markdown. "
            "Only include information worth remembering long-term. "
            "If there's nothing important, respond with 'NONE'.\n\n"
            f"Conversation:\n{conversation_text[:20000]}"
        )

        try:
            response = await self.llm.chat(
                system="You extract key facts from conversations. Be concise.",
                messages=[{"role": "user", "content": extract_prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            facts = response.content.strip()
            if facts and facts.upper() != "NONE":
                self.workspace.append_memory(f"\n## Extracted ({_now_str()})\n\n{facts}")
                self.workspace.append_daily_log(f"Context compacted — {len(facts)} chars flushed to MEMORY.md")
                logger.info(f"Flushed {len(facts)} chars to MEMORY.md")
        except Exception as e:
            logger.warning(f"Failed to flush facts to MEMORY.md: {e}")

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

        try:
            response = await self.llm.chat(
                system="You produce concise conversation summaries.",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            summary = response.content.strip()
        except Exception as e:
            logger.warning(f"Summarization failed, falling back to hard prune: {e}")
            return self._hard_prune(messages)

        summary_msg = {
            "role": "user",
            "content": f"## Conversation Summary (auto-compacted)\n\n{summary}",
        }

        recent = messages[-4:] if len(messages) > 4 else messages
        result = [summary_msg] + recent
        logger.info(
            f"Compacted {len(messages)} messages -> {len(result)} "
            f"(usage: {int(self.usage(result) * 100)}%)"
        )
        return result

    def _hard_prune(self, messages: list[dict]) -> list[dict]:
        """Emergency: keep only the first and last few messages."""
        if len(messages) <= 6:
            return messages
        pruned = messages[:1] + messages[-4:]
        logger.warning(f"Hard-pruned {len(messages)} -> {len(pruned)} messages")
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
