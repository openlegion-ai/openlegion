"""Context window manager — virtual memory for cognition.

Monitors token usage across the conversation and auto-compacts when
the context window fills up. Before compacting, important facts are
flushed to MEMORY.md so nothing is lost.

Write-then-compact: always persist before discarding.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from src.agent.workspace import _MEMORY_HEAD_BUDGET
from src.shared.models import get_context_window
from src.shared.utils import sanitize_for_prompt, setup_logging

if TYPE_CHECKING:
    from src.agent.llm import LLMClient
    from src.agent.memory import MemoryStore
    from src.agent.workspace import WorkspaceManager

logger = setup_logging("agent.context")

_FLUSH_THRESHOLD = 0.60  # proactive fact extraction before compaction
_COMPACT_THRESHOLD = 0.70  # summarize-and-replace conversation history
_WARNING_THRESHOLD = 0.80  # warn agent to wrap up or save facts

_encoding_cache: dict[str, object | None] = {}

_SUMMARIZATION_INPUT_LIMIT = 20_000  # max chars per summarization LLM call (chunk size)
_SUMMARIZATION_MAX_CHUNKS = 8        # cap on summarization calls per compaction
_SUMMARY_FOLD_THRESHOLD = 12_000     # combined partial summaries above this get folded once
_TOOL_RESULT_SUMMARY_CHARS = 2_000   # per tool-result slice fed to the compaction summarizer

_CONSOLIDATION_MIN_INTERVAL_S = 6 * 3600   # at most every 6 hours
_CONSOLIDATION_MIN_LOG_CHARS = 1_500       # skip if little new material accrued
_CONSOLIDATION_FAIL_BACKOFF_S = 30 * 60    # after a failed run, wait before retrying
_DECAY_MIN_INTERVAL_S = 6 * 3600           # salience-decay cadence for the maintenance pass


def _tail_on_boundary(text: str, limit: int) -> str:
    """Return the last ``limit`` chars of ``text``, re-aligned forward to the
    next ``\\n## `` section boundary so the slice never starts mid-entry.

    The memory log is append-only (oldest first), so the NEWEST entries live
    at the tail — this is what consolidation must read.
    """
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    idx = tail.find("\n## ")
    if idx != -1:
        tail = tail[idx + 1:]
    return tail


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


def _content_chars(content) -> int:
    """Return character count for message content, handling multimodal blocks.

    Image blocks (``image_url`` with data URIs) are counted as a fixed
    estimate (~1,600 tokens ≈ 6,400 chars) rather than their raw base64
    size, which would overestimate by ~100x and trigger premature
    context compaction.
    """
    _IMAGE_TOKEN_ESTIMATE = 1_600  # conservative; covers high-detail images
    _IMAGE_CHAR_ESTIMATE = _IMAGE_TOKEN_ESTIMATE * 4

    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return len(str(content))
    total = 0
    for block in content:
        if not isinstance(block, dict):
            total += len(str(block))
        elif block.get("type") == "image_url":
            total += _IMAGE_CHAR_ESTIMATE
        else:
            # text blocks, tool_use blocks, etc.
            total += len(json.dumps(block))
    return total


def estimate_tokens(messages: list[dict], model: str = "") -> int:
    """Estimate token count for a message list.

    - OpenAI models: use tiktoken for accurate counting
    - Anthropic models: ~3.5 chars per token
    - Others/fallback: ~4 chars per token

    Image blocks use a fixed token estimate (~1,600 tokens) instead of
    counting base64 characters, which would overestimate by ~100x.
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
                        # Use image-aware char count, then estimate tokens
                        total += _content_chars(val) // 4
            return total

    chars = sum(_content_chars(m.get("content", "")) + len(json.dumps({
        k: v for k, v in m.items() if k != "content"
    })) for m in messages)
    if model.startswith("anthropic/"):
        return int(chars / 3.5)
    return chars // 4


def group_messages_by_tool_call(messages: list[dict]) -> list[list[dict]]:
    """Group consecutive tool-call related messages into atomic units.

    Each group is either:

    * a standalone non-tool message (``[user_msg]``, ``[assistant_text]``,
      ``[system]``)
    * an assistant+tools group: ``[assistant(tool_calls), tool_1, tool_2,
      ...]`` collecting an assistant message that emitted ``tool_calls``
      together with every subsequent ``role=tool`` reply that answers
      them.

    The grouping is the invariant the LLM API expects: a ``tool`` message
    must immediately follow its parent ``assistant`` with the matching
    ``tool_call_id``, and every ``tool_call`` from the assistant must be
    answered by a ``tool`` message before the next assistant turn. Any
    slice/trim operation that respects group boundaries cannot orphan a
    tool result from its parent or vice-versa.

    Three call sites use this — :meth:`ContextManager._summarize_compact`,
    :meth:`ContextManager._hard_prune`, and ``AgentLoop._trim_context``
    in ``loop.py``. They had three identical inline copies of this loop;
    consolidating them here eliminates the drift risk and makes the
    invariant explicit.
    """
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
    return groups


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
        on_memory_update: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens or get_context_window(model)
        # Surface the resolved budget so a silent fallthrough (model missing
        # from litellm's registry → 128K default) is visible in agent logs
        # rather than only showing up as surprisingly-early compaction.
        logger.info(
            "context budget resolved: model=%s max_tokens=%d",
            model or "(unset)", self.max_tokens,
        )
        self.llm = llm
        self.workspace = workspace
        self.memory = memory
        self._flush_triggered = False
        self._flush_lock = asyncio.Lock()
        self._on_memory_update = on_memory_update
        # Backoff clock for the maintenance pass: set to a future time after a
        # failed consolidation so a frequent tick doesn't hammer the LLM.
        self._consolidation_retry_after = 0.0

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

    def estimate_request_tokens(
        self,
        messages: list[dict],
        system_prompt: str = "",
        tools: list[dict] | None = None,
    ) -> int:
        """Estimate the TRUE size of the request the LLM client will send.

        ``usage()`` / ``should_compact()`` count only ``messages`` — they are
        blind to the system prompt, the tool schemas, and the per-turn volatile
        suffix that ``llm.py`` prepends/appends at the call site. On a model
        with a large nominal window (e.g. Opus 4.8's 1M) a fat tool schema +
        system prompt is a large fixed overhead; ignoring it lets the message
        list grow until the assembled request exceeds the hard model limit and
        every call 400s. This returns the conservative full-request estimate so
        the pre-flight guard / emergency prune can act on the real number.

        The tool-schema estimate intentionally errs slightly HIGH
        (``len(json.dumps(tools)) // 3``, a tighter divisor than the 4-chars/tok
        message default) — overshooting the budget is safe; undershooting
        re-wedges.
        """
        total = estimate_tokens(messages, self.model)
        if system_prompt:
            total += estimate_tokens(
                [{"role": "system", "content": system_prompt}], self.model,
            )
        if tools:
            # Tool schemas come from @tool-decorated functions and are
            # JSON-safe, but this runs on every turn's pre-flight path now, so
            # fall back rather than crash the turn if a schema is ever not
            # serializable.
            try:
                total += len(json.dumps(tools)) // 3
            except (TypeError, ValueError):
                total += len(str(tools)) // 4
        return total

    def prune_to_fit(
        self,
        messages: list[dict],
        system_prompt: str = "",
        tools: list[dict] | None = None,
        *,
        ceiling_frac: float = 0.90,
    ) -> list[dict]:
        """Emergency hard safety net: drop oldest tool-call groups until the
        FULL request (messages + system + tools) fits under a fraction of the
        model window.

        This is NOT a threshold change — the 0.60/0.70/0.80 compaction
        thresholds are untouched. This is a separate last-resort guard for the
        case where a turn STARTS already over the model limit (inherited fat
        context, restored fat checkpoint) so the first LLM call would 400 with
        "prompt is too long" before reactive compaction ever runs.

        Group-aware (via :func:`group_messages_by_tool_call`): always keeps the
        FIRST group (initial user/context) and at least the most-recent group,
        and prunes only at group boundaries so a tool result is never orphaned
        from its parent assistant. Reuses the same role-alternation bridge logic
        as :meth:`_hard_prune` / ``AgentLoop._trim_context`` so it never emits
        consecutive same-role messages.

        Returns the (possibly) pruned list; a no-op when already under ceiling.
        """
        if not messages:
            return messages

        ceiling = int(self.max_tokens * ceiling_frac)
        before = self.estimate_request_tokens(messages, system_prompt, tools)
        if before <= ceiling:
            return messages

        groups = group_messages_by_tool_call(messages)
        # Keep first group + at least the most-recent group. We drop the OLDEST
        # non-first group repeatedly: kept = [first] + groups[drop_start:].
        # ``drop_start`` walks forward from index 2 (index 1 would re-test the
        # full unmodified list — guaranteed over ceiling since ``before`` was).
        drop_start = 2
        while drop_start < len(groups):
            # Candidate: first group + everything from drop_start onward.
            kept_groups = groups[:1] + groups[drop_start:]
            candidate = self._merge_groups_alternating(kept_groups)
            if self.estimate_request_tokens(candidate, system_prompt, tools) <= ceiling:
                after = self.estimate_request_tokens(candidate, system_prompt, tools)
                logger.info(
                    "prune_to_fit: dropped %d oldest group(s), %s->%s est tokens "
                    "(ceiling=%s, frac=%.2f)",
                    drop_start - 1, f"{before:,}", f"{after:,}",
                    f"{ceiling:,}", ceiling_frac,
                )
                return candidate
            # Don't shed past [first, last]: leave the most-recent group intact.
            if drop_start >= len(groups) - 1:
                break
            drop_start += 1

        # Couldn't get under the ceiling even at the minimal kept set
        # (system + tools alone, or first+last group, already too big). Return
        # the minimal kept set — the pre-flight guard re-checks and the LLM
        # self-heal retry will still surface a clear failure if even this 400s.
        minimal_groups = groups[:1] + groups[-1:] if len(groups) > 1 else groups
        pruned = self._merge_groups_alternating(minimal_groups)
        after = self.estimate_request_tokens(pruned, system_prompt, tools)
        logger.warning(
            "prune_to_fit: could not fit under ceiling=%s even at minimal kept "
            "set (%s est tokens); system+tools overhead may exceed budget. "
            "before=%s groups=%d kept=%d",
            f"{ceiling:,}", f"{after:,}", f"{before:,}",
            len(groups), len(minimal_groups),
        )
        return pruned

    @staticmethod
    def _merge_groups_alternating(groups: list[list[dict]]) -> list[dict]:
        """Flatten tool-call groups into a message list, inserting bridge
        messages so no two consecutive messages share a role (user/assistant).

        Mirrors the bridge logic in :meth:`_hard_prune` so a prune that drops a
        middle group can't create a ``user → user`` / ``assistant → assistant``
        seam across the gap (which the LLM API rejects).
        """
        merged = [m for g in groups for m in g]
        i = 0
        while i < len(merged) - 1:
            role_a = merged[i].get("role")
            role_b = merged[i + 1].get("role")
            if role_a == role_b == "user":
                merged.insert(i + 1, {
                    "role": "assistant",
                    "content": "Understood, continuing from above.",
                })
                i += 2
            elif role_a == role_b == "assistant":
                merged.insert(i + 1, {"role": "user", "content": "Continue."})
                i += 2
            else:
                i += 1
        return merged

    async def maybe_compact(
        self, system_prompt: str, messages: list[dict],
    ) -> tuple[list[dict], bool]:
        """Check context usage and compact if needed.

        Returns ``(messages, did_compact)`` where ``did_compact`` is True only
        when the conversation was actually summarised and replaced.

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
            async with self._flush_lock:
                if self._flush_triggered:
                    return messages, False  # another coroutine already flushed
                self._flush_triggered = True
            await self._proactive_flush(system_prompt, messages)
            return messages, False  # don't compact yet

        if not self.should_compact(messages):
            return messages, False

        result = await self._do_compact(system_prompt, messages, label="Compaction")
        return result, True

    async def force_compact(
        self, system_prompt: str, messages: list[dict],
    ) -> list[dict]:
        """Force compaction regardless of token usage.

        Used by session auto-continue to reset context at round-count
        checkpoints.  Follows the same flush-then-summarize pattern as
        ``maybe_compact`` but skips the threshold checks.
        """
        if not messages:
            return messages

        return await self._do_compact(system_prompt, messages, label="Force-compaction")

    async def _do_compact(
        self, system_prompt: str, messages: list[dict], *, label: str,
    ) -> list[dict]:
        """Shared compaction logic used by both maybe_compact and force_compact.

        Flushes facts to memory, summarises (or hard-prunes), and logs stats.
        Returns the compacted message list.
        """
        tokens_before = self.token_count(messages)
        msg_count_before = len(messages)
        t0 = time.monotonic()
        usage_pct = int(self.usage(messages) * 100)
        logger.info(
            "%s: context at %d%% (%s tokens, %d msgs)",
            label, usage_pct, f"{tokens_before:,}", msg_count_before,
        )

        # Reset so proactive flush can fire again after compaction
        self._flush_triggered = False

        # Step 1: Extract important facts and flush to MEMORY.md + DB
        if self.workspace and self.llm:
            try:
                await self._flush_to_memory(system_prompt, messages)
            except Exception as e:
                logger.warning(
                    "Flush to memory failed during %s, proceeding: %s",
                    label.lower(), e,
                )

        # Step 1b: opportunistically consolidate the compiled memory head.
        if self.workspace and self.llm:
            try:
                await self._maybe_consolidate_memory()
            except Exception as e:
                logger.warning("Memory consolidation skipped: %s", e)

        # Step 2: Summarize and compress
        if self.llm:
            result = await self._summarize_compact(system_prompt, messages)
        else:
            result = self._hard_prune(messages)

        tokens_after = self.token_count(result)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        ratio = round(tokens_after / tokens_before, 2) if tokens_before else 0
        method = "summarize" if self.llm else "hard_prune"
        logger.info(
            "%s complete: %s, %d->%d msgs, %s->%s tokens (%.0f%% reduction), %dms",
            label, method, msg_count_before, len(result),
            f"{tokens_before:,}", f"{tokens_after:,}",
            (1 - ratio) * 100, elapsed_ms,
        )
        return result

    async def _extract_and_store_facts(
        self, messages: list[dict], *, label: str,
    ) -> int:
        """Extract structured facts from messages, store to workspace + memory DB.

        Returns the number of facts extracted (0 on failure or empty).
        Shared by proactive flush (60%) and compaction flush (70%).
        """
        t0 = time.monotonic()
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
            f"Conversation:\n{conversation_text[:_SUMMARIZATION_INPUT_LIMIT]}"
        )

        try:
            response = await self.llm.chat(
                system="You extract structured facts from conversations. Return only valid JSON.",
                messages=[{"role": "user", "content": extract_prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            raw = response.content.strip()
            # Strip markdown code fences if present (e.g. ```json\n...\n```)
            if raw.startswith("```"):
                lines = raw.split("\n", 1)
                body = lines[1] if len(lines) > 1 else ""
                # Remove only a trailing ``` fence, not arbitrary occurrences
                raw = body.rstrip()[:-3].strip() if body.rstrip().endswith("```") else body.strip()

            facts = json.loads(raw)
            if not isinstance(facts, list) or not facts:
                return 0

            # Write human-readable summary to MEMORY.md. M2: sanitize each
            # LLM-extracted key/value for Unicode hygiene before it lands in
            # the persistent workspace markdown (re-read into context later).
            lines = [
                f"- **{sanitize_for_prompt(str(f['key']))}**: "
                f"{sanitize_for_prompt(str(f['value']))}"
                for f in facts if f.get("key") and f.get("value")
            ]
            if lines and self.workspace:
                self.workspace.append_memory(
                    f"\n## {label} ({_now_str()})\n\n" + "\n".join(lines),
                )
                if self._on_memory_update:
                    try:
                        await self._on_memory_update()
                    except Exception:
                        logger.debug("Non-fatal: memory update notification failed")

            # Store structured facts in memory DB for search
            if self.memory:
                stored = await self.memory.store_facts_batch(facts)
                logger.info(f"{label}: {stored} facts stored to memory DB")

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.info(f"{label}: {len(facts)} facts extracted in {elapsed_ms}ms")
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

    async def run_maintenance(self) -> None:
        """Off-live-path memory maintenance: re-derive the compiled MEMORY.md
        head and decay fact salience.

        Driven by the agent's periodic background pass (gbrain "dream cycle" /
        hermes "Curator"), NOT the live turn — the caller is expected to hold
        the chat lock so this never races a turn's memory writes. Each step is
        internally gated (consolidation/decay >=6h), so a frequent tick is
        cheap when nothing is due. Best-effort: never raises.
        """
        await self._maybe_consolidate_memory()
        await self._maybe_decay_salience()

    async def _maybe_decay_salience(self) -> None:
        """Decay fact salience at most once per window.

        Gated by the shared ``.memory_decayed`` sentinel that the task path
        also stamps on every fresh-task decay (loop.py). This gating only binds
        the background maintenance pass: it won't pile an extra decay onto an
        agent the task path recently decayed. The task path itself stays
        ungated (it decays per fresh task), so this pass exists to cover an
        idle agent (e.g. the operator, which runs no tasks) — the gap it closes.
        """
        ws, mem = self.workspace, self.memory
        if not ws or not mem:
            return
        if not ws.decay_due(_DECAY_MIN_INTERVAL_S):
            return
        try:
            await mem.decay_all()
            ws.mark_decayed()
        except Exception as e:
            logger.debug("salience decay failed: %s", e)

    async def _maybe_consolidate_memory(self) -> None:
        """Re-derive the compiled MEMORY.md head from the append-only log +
        high-salience facts. Time-gated (>=6h) and material-gated (>=1.5k log
        chars, OR a head exceeding its injection budget) so it adds at most
        one LLM call per cycle. Best-effort: never raises into the caller
        (compaction or the maintenance pass).
        """
        ws, llm = self.workspace, self.llm
        if not ws or not llm:
            return
        if time.time() < self._consolidation_retry_after:
            return  # backing off after a recent failure
        if not ws.consolidation_due(_CONSOLIDATION_MIN_INTERVAL_S):
            return
        log = ws.load_memory_log()
        head = ws.load_compiled_memory()
        # An oversized head is either a legacy marker-less MEMORY.md (split as
        # all-head, empty log) or an LLM-overshot compile; both must be
        # re-compiled even with no new log material — otherwise they inject
        # clipped + stale at the head budget forever.
        if len(log) < _CONSOLIDATION_MIN_LOG_CHARS and len(head) <= _MEMORY_HEAD_BUDGET:
            return
        salient = ""
        if self.memory:
            try:
                facts = await self.memory.get_high_salience_facts(top_k=30)
                salient = "\n".join(f"- {f.key}: {f.value}" for f in facts)
            except Exception as e:
                logger.debug("consolidation: salience fetch failed: %s", e)
        prompt = (
            "You maintain an agent's COMPILED long-term memory — a small, durable, "
            "deduplicated brief that is injected into context every turn.\n\n"
            "Rewrite it from the inputs below. Rules: merge duplicates; on "
            "contradiction prefer the most RECENT; drop stale/transient/one-off "
            "items; keep durable facts, user preferences, and decisions. Group "
            "related points under short markdown headings. Be concise (aim < 1200 "
            "words). Output ONLY the compiled memory markdown — no preamble.\n\n"
            f"## Current compiled memory\n{head[:_SUMMARIZATION_INPUT_LIMIT]}\n\n"
            f"## High-salience facts\n{salient[:4000]}\n\n"
            f"## Recent activity log (newest last)\n"
            f"{_tail_on_boundary(log, _SUMMARIZATION_INPUT_LIMIT)}"
        )
        try:
            resp = await llm.chat(
                system="You compile concise, durable agent memory.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500, temperature=0.3,
            )
        except Exception as e:
            logger.warning("Memory consolidation LLM call failed: %s", e)
            self._consolidation_retry_after = time.time() + _CONSOLIDATION_FAIL_BACKOFF_S
            return
        new_head = (resp.content or "").strip()
        if not new_head:
            self._consolidation_retry_after = time.time() + _CONSOLIDATION_FAIL_BACKOFF_S
            return
        ws.write_compiled_memory(sanitize_for_prompt(new_head))
        ws.mark_consolidated()
        self._consolidation_retry_after = 0.0
        if self._on_memory_update:
            try:
                await self._on_memory_update()
            except Exception:
                logger.debug("Non-fatal: memory update notification failed")

    _SUMMARIZE_RETRIES = 2
    _SUMMARIZE_BACKOFF = 2  # seconds

    async def _summarize_compact(
        self, system_prompt: str, messages: list[dict],
    ) -> list[dict]:
        """Summarize the conversation and replace history with summary + recent.

        Group-aware: the recent tail is sliced by tool-call group, never
        by message index, so a multi-tool turn at the boundary stays
        atomic. Slicing by index (the legacy ``messages[-4:]`` shape)
        could land on ``[tool_a, tool_b, tool_c, tool_d]`` and orphan
        the parent ``assistant(tool_calls)`` at index ``-5``; the next
        LLM call would then reject the messages and the post-tool
        continuation would silently fail.
        """
        groups = group_messages_by_tool_call(messages)

        # Need at least 2 groups to compact meaningfully — one to
        # summarize, one to keep as recent. Otherwise return as-is
        # (the caller's threshold check decided we should try, but
        # there is nothing safe to summarize without orphaning).
        if len(groups) <= 1:
            return messages

        # Keep last N groups intact; reserve at least one group for the
        # summary. Caps at 4 to roughly match the legacy 4-message tail
        # for normal alternation, but a multi-tool group preserves the
        # whole group instead of slicing inside it.
        keep_n = min(4, len(groups) - 1)
        older_groups = groups[:-keep_n]
        recent_groups = groups[-keep_n:]

        # P4 tail cap: a few huge tool-result groups in the retained tail can
        # defeat compaction (the summary shrinks the OLD context but the kept
        # recent groups are still over budget). Drop the OLDEST of the kept
        # groups (group-aware, keep >=1) until the retained tail is under
        # ~0.5 * max_tokens. The dropped groups still feed the summary because
        # ``older_groups`` is recomputed to include them.
        _tail_cap = int(self.max_tokens * 0.5)
        while (
            len(recent_groups) > 1
            and estimate_tokens(
                [m for g in recent_groups for m in g], self.model,
            ) > _tail_cap
        ):
            recent_groups = recent_groups[1:]
        older_groups = groups[: len(groups) - len(recent_groups)]

        older_messages = [m for g in older_groups for m in g]

        # Chunked summarization: the legacy shape fed only the FIRST 20k chars
        # of the older history (and 200 chars per tool result) to a single
        # summarization call and silently discarded the rest — on research
        # tasks that orphaned most gathered data right before synthesis.
        # Instead, pack the older history into ≤20k chunks at message
        # boundaries, summarize each, and stitch the partials.
        base_instruction = (
            "Summarize this conversation concisely, preserving key context "
            "the assistant needs to continue helpfully. Include: what was discussed, "
            "what actions were taken, what's pending, and any user preferences revealed. "
            "Preserve concrete data points the assistant gathered (names, numbers, "
            "metrics, URLs, findings) — they cannot be recovered after compaction."
        )
        chunks = self._chunk_message_texts(
            older_messages, _SUMMARIZATION_INPUT_LIMIT, _TOOL_RESULT_SUMMARY_CHARS,
        )
        if not chunks:
            return self._hard_prune(messages)
        if len(chunks) > _SUMMARIZATION_MAX_CHUNKS:
            # Pathological volume: keep the first chunk (original ask / early
            # framing) plus the newest chunks; the omission is logged only.
            omitted = len(chunks) - _SUMMARIZATION_MAX_CHUNKS
            logger.warning(
                f"Compaction input spans {len(chunks)} chunks; "
                f"omitting {omitted} middle chunk(s)"
            )
            chunks = [chunks[0]] + chunks[-(_SUMMARIZATION_MAX_CHUNKS - 1):]

        if len(chunks) == 1:
            summary = await self._summarize_text(
                f"{base_instruction}\n\nConversation:\n{chunks[0]}"
            )
            if summary is None:
                logger.warning("Summarization unavailable, falling back to hard prune")
                return self._hard_prune(messages)
        else:
            partials: list[str] = []
            for i, chunk in enumerate(chunks):
                part = await self._summarize_text(
                    f"{base_instruction}\n\n"
                    f"This is part {i + 1} of {len(chunks)} of the conversation "
                    f"(chronological order).\n\nConversation:\n{chunk}"
                )
                if part is None:
                    logger.warning("Chunk summarization unavailable, falling back to hard prune")
                    return self._hard_prune(messages)
                partials.append(part)
            summary = "\n\n".join(
                f"### Part {i + 1} of {len(partials)}\n{p}"
                for i, p in enumerate(partials)
            )
            if len(summary) > _SUMMARY_FOLD_THRESHOLD:
                folded = await self._summarize_text(
                    "Merge these sequential summaries of one conversation into a "
                    "single coherent summary. Preserve concrete data points "
                    "(names, numbers, metrics, URLs, findings), decisions, and "
                    "pending work.\n\n" + summary,
                    max_tokens=2048,
                )
                # Fold failure is non-fatal: the unfolded partials are valid,
                # just longer.
                if folded:
                    summary = folded

        # M2: sanitize the LLM-produced summary before re-injecting it into the
        # context window, for Unicode-hygiene uniformity with the memory /
        # bootstrap entry paths (which all call ``sanitize_for_prompt``). This
        # strips invisible/control characters; it is lossless for normal text so
        # summary quality is unaffected. NOT an anti-injection control.
        summary_msg = {
            "role": "user",
            "content": "## Conversation Summary (auto-compacted)\n\n"
            + sanitize_for_prompt(summary),
        }

        # The summary is ``role=user``. The recent tail's leading group
        # must compose cleanly with that:
        #
        #   * ``assistant`` (with or without tool_calls) — valid; the
        #     summary (user) → assistant alternation is well-formed.
        #     If the assistant carries tool_calls, the tool messages
        #     that follow are part of the same atomic group so no
        #     orphan can arise.
        #   * ``system`` — accepted (uncommon mid-conversation but
        #     valid alternation).
        #   * ``user`` — back-to-back ``user`` messages break the API
        #     contract. Drop the group if we have another to fall back
        #     on, else insert a bridge assistant.
        #   * ``tool`` — orphan: no preceding assistant carries the
        #     matching tool_call_id. Drop the group; if the orphan is
        #     the last group, drop the whole tail and return summary
        #     alone (losing a stray tool message is strictly better
        #     than emitting an API-invalid sequence).
        #   * anything else (``developer``, malformed/future roles) —
        #     same treatment as orphan tool: drop the tail. Better to
        #     return a summary-only result than ship unknown roles
        #     downstream and surface as cryptic API rejections.
        #
        # Drop leading invalid groups in a loop so consecutive bad
        # groups (e.g. two queued user turns followed by an assistant)
        # are all stripped, not just the first one.
        while (
            len(recent_groups) > 1
            and recent_groups[0][0].get("role") in ("user", "tool")
        ):
            recent_groups = recent_groups[1:]
        recent_messages = [m for g in recent_groups for m in g]

        first_role = recent_messages[0].get("role") if recent_messages else None
        if first_role == "user":
            # Only one group left and it leads with ``user`` — insert a
            # bridging assistant so summary(user) → bridge(asst) → user
            # is valid alternation.
            bridge = {
                "role": "assistant",
                "content": "Understood, continuing from the summary above.",
            }
            result = [summary_msg, bridge] + recent_messages
        elif first_role in ("assistant", "system"):
            # Valid alternation after summary(user). The tool messages
            # inside an ``assistant(tool_calls)`` group are paired with
            # their parent in the same group (helper invariant) so no
            # orphan can surface from this branch.
            result = [summary_msg] + recent_messages
        else:
            # ``tool`` (orphan after the loop's dedup pass) or any
            # unknown role. Pathological input that shouldn't occur in
            # normal flow — compaction runs post-tool so every tool
            # message has its parent assistant in the same group, and
            # the codebase only emits ``user``/``assistant``/``tool``/
            # ``system`` roles into ``_chat_messages``. Drop the tail
            # rather than emit an API-invalid sequence; the summary
            # already captured the older context.
            logger.warning(
                "Recent tail leads with unexpected role %r after "
                "group-aware dedup; returning summary-only. "
                "messages=%d groups=%d",
                first_role, len(messages), len(recent_groups),
            )
            result = [summary_msg]
        logger.info(
            f"Compacted {len(messages)} messages -> {len(result)} "
            f"(usage: {int(self.usage(result) * 100)}%)"
        )
        return result

    def _hard_prune(self, messages: list[dict]) -> list[dict]:
        """Emergency: keep first message and last N messages (group-aware).

        Groups tool_calls with their tool results via the shared
        :func:`group_messages_by_tool_call` helper so we never orphan
        a tool_call from its matching tool responses.
        """
        if len(messages) <= 8:
            return messages

        groups = group_messages_by_tool_call(messages)

        if len(groups) <= 5:
            return messages

        # Keep first group + last 4 groups, bridging role-alternation gaps so
        # the dropped middle can't leave a user->user / assistant->assistant
        # seam. Shared with ``prune_to_fit`` via ``_merge_groups_alternating``.
        pruned = self._merge_groups_alternating(groups[:1] + groups[-4:])
        logger.warning(f"Hard-pruned {len(messages)} -> {len(pruned)} messages (group-aware)")
        return pruned

    @staticmethod
    def _messages_to_text(messages: list[dict], tool_chars: int = 200) -> str:
        """Convert messages to readable text for summarization.

        ``tool_chars`` bounds each tool-result slice. The default keeps the
        fact-extraction path cheap; the compaction summarizer passes a much
        larger bound because tool results are where gathered data (research
        findings, fetched pages, query output) lives — over-truncating them
        here is unrecoverable data loss for the post-compaction conversation.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Multimodal messages have list content — extract text blocks only;
            # binary image data is not useful for summarization.
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if role == "tool":
                content = content[:tool_chars]
            if content:
                parts.append(f"[{role}]: {content}")
        return "\n\n".join(parts)

    @classmethod
    def _chunk_message_texts(
        cls, messages: list[dict], chunk_size: int, tool_chars: int,
    ) -> list[str]:
        """Render messages to text and pack them into ≤``chunk_size`` chunks.

        Chunk boundaries fall between messages so no message is split across
        summarization calls (a single oversized message is truncated to fit).
        """
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for msg in messages:
            piece = cls._messages_to_text([msg], tool_chars=tool_chars)
            if not piece:
                continue
            if len(piece) > chunk_size:
                piece = piece[:chunk_size]
            if current and current_len + len(piece) + 2 > chunk_size:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(piece)
            current_len += len(piece) + 2
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    async def _summarize_text(self, prompt: str, max_tokens: int = 1024) -> str | None:
        """One summarization LLM call with retry/backoff.

        Returns the stripped summary, or None on persistent failure or an
        empty response (callers decide the fallback).
        """
        last_err: Exception | None = None
        for attempt in range(self._SUMMARIZE_RETRIES + 1):
            try:
                response = await self.llm.chat(
                    system="You produce concise conversation summaries.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                return response.content.strip() or None
            except Exception as e:
                last_err = e
                if attempt < self._SUMMARIZE_RETRIES:
                    wait = self._SUMMARIZE_BACKOFF * (2 ** attempt)
                    logger.warning(
                        f"Summarization failed (attempt {attempt + 1}/{self._SUMMARIZE_RETRIES + 1}), "
                        f"retrying in {wait}s: {e}"
                    )
                    await asyncio.sleep(wait)
        logger.warning(
            f"Summarization failed after {self._SUMMARIZE_RETRIES + 1} attempts: {last_err}"
        )
        return None


def _now_str() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
