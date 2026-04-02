"""Bounded agent execution loop.

Each task runs: perceive -> decide (LLM) -> act (tool) -> learn.
Max 20 iterations per task. Proper LLM tool-calling message roles.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import httpx

from src.agent.attachments import enrich_message_with_attachments
from src.agent.loop_detector import ToolLoopDetector
from src.agent.workspace import INTROSPECT_PERM_KEYS
from src.shared.types import SILENT_REPLY_TOKEN, AgentStatus, LLMResponse, TaskAssignment, TaskResult
from src.shared.utils import format_dict, generate_id, sanitize_for_prompt, setup_logging, truncate

if TYPE_CHECKING:
    from src.agent.context import ContextManager
    from src.agent.llm import LLMClient
    from src.agent.memory import MemoryStore
    from src.agent.mesh_client import MeshClient
    from src.agent.skills import SkillRegistry
    from src.agent.workspace import WorkspaceManager

logger = setup_logging("agent.loop")

# Status codes that indicate transient server-side errors worth retrying
_RETRYABLE_STATUS_CODES = {429, 502, 503}
_MAX_RETRIES = 3
_BACKOFF_BASE = 1  # seconds: 1, 2, 4
_TOOL_TIMEOUT = int(os.environ.get("OPENLEGION_TOOL_TIMEOUT", "300"))  # seconds — hard ceiling per tool
_FLEET_ROSTER_TTL = 600  # seconds — cache TTL for fleet roster
_GOALS_TTL = 300  # seconds — cache TTL for goals fetch
_FALLBACK_MAX_TOKENS = 100_000  # context trim fallback when no context manager
_TOOL_HISTORY_LIMIT = 10  # recent tool outcomes in system prompt
HEARTBEAT_MAX_ITERATIONS = 10  # tighter bound for heartbeat (cheaper than task/chat)

# Markdown heading pattern for detecting effectively-empty heartbeat files
_HEADING_OR_EMPTY_RE = re.compile(r"^(#+\s.*|\s*)$")

# Strip leading <think>…</think> blocks emitted by reasoning models
# (Qwen3, DeepSeek-R1 etc.) so chat bubbles and conversation history
# contain only the actual answer.
_THINK_TAG_RE = re.compile(r"^(?:<think>[\s\S]*?</think>\s*)+")


def _is_heartbeat_empty(content: str | None) -> bool:
    """Check if HEARTBEAT.md has no actionable content (only headings/blanks).

    Returns True when the file is missing, empty, or contains only markdown
    headings and whitespace — meaning there are no heartbeat rules to execute.
    """
    if not content:
        return True
    return all(_HEADING_OR_EMPTY_RE.match(line) for line in content.splitlines())


def _strip_think_tags(text: str) -> str:
    """Remove leading ``<think>…</think>`` blocks from model output."""
    if not text.startswith("<think>"):
        return text
    stripped = _THINK_TAG_RE.sub("", text).strip()
    return stripped if stripped else text


def _extract_json_response(text: str) -> str:
    """Extract ``response`` value from JSON chain-of-thought output.

    Some local models (Qwen3) emit their full reasoning as a JSON object::

        {"thought": {...}, "response": "The actual answer"}

    When the entire content is a JSON object with a ``response`` key,
    return just the response value.  Otherwise return the text unchanged.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return text
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return text
    if isinstance(obj, dict) and "response" in obj:
        resp = obj["response"]
        return str(resp) if resp is not None else text
    return text


# Files already injected via bootstrap — skip in first-message auto-search
# to avoid duplicate content.  Matches WorkspaceManager._BOOTSTRAP_FILES.
_BOOTSTRAP_SEARCH_EXCLUDE = frozenset({
    "PROJECT.md", "SYSTEM.md", "INSTRUCTIONS.md",
    "SOUL.md", "USER.md", "MEMORY.md",
})
_MAX_STEER_INTERRUPTS = 3  # max times a steer can interrupt a final answer per turn

# ContextVar so tools (e.g. notify_user) can detect heartbeat mode
_heartbeat_mode: ContextVar[bool] = ContextVar("_heartbeat_mode", default=False)

# Tools that require a project blackboard — excluded for standalone agents.
_BLACKBOARD_TOOLS = frozenset({
    "read_blackboard", "write_blackboard", "list_blackboard",
    "publish_event", "subscribe_event", "watch_blackboard",
    "claim_task", "hand_off", "check_inbox", "update_status", "complete_task",
})


async def _llm_call_with_retry(llm_chat_fn, *, system, messages, tools, **kwargs):
    """Call the LLM with exponential backoff on transient errors.

    Retries on: connection errors, timeouts, 429/502/503 status codes.
    Does NOT retry on: budget exceeded (RuntimeError), permanent errors.
    """
    last_exc: Exception = RuntimeError("LLM call failed after all retries")
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await llm_chat_fn(system=system, messages=messages, tools=tools, **kwargs)
        except RuntimeError:
            # Budget exceeded or permanent LLM errors — don't retry
            raise
        except httpx.HTTPStatusError as e:
            if e.response.status_code in _RETRYABLE_STATUS_CODES:
                last_exc = e
                if attempt < _MAX_RETRIES:
                    backoff = _BACKOFF_BASE * (2 ** attempt)
                    # Honour Retry-After header when present (429 responses)
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait = max(float(retry_after), backoff)
                        except (ValueError, TypeError):
                            wait = backoff
                    else:
                        wait = backoff
                    logger.warning(
                        f"LLM call returned {e.response.status_code}, retrying in {wait}s "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                    )
                    await asyncio.sleep(wait)
                    continue
            raise
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                wait = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    f"LLM call failed ({type(e).__name__}), retrying in {wait}s "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                )
                await asyncio.sleep(wait)
                continue
            raise
    # Should not reach here, but just in case
    raise last_exc  # type: ignore[misc]


class AgentLoop:
    """Bounded agent execution loop with proper LLM tool-calling protocol.

    Key invariants:
    - Max 20 iterations per task (prevents runaway agents)
    - Messages follow: user -> assistant(tool_calls) -> tool(result) -> assistant
    - Context window management (trims old exchanges when too large)
    - Token budget tracking (prevents runaway API spend)
    - Cancellation support via flag checked each iteration
    """

    MAX_ITERATIONS = 20

    def __init__(
        self,
        agent_id: str,
        role: str,
        memory: MemoryStore,
        skills: SkillRegistry,
        llm: LLMClient,
        mesh_client: MeshClient,
        workspace: WorkspaceManager | None = None,
        context_manager: ContextManager | None = None,
        excluded_tools: frozenset[str] | None = None,
    ):
        # Override class defaults from env vars (set by dashboard system settings)
        def _clamp_env(name: str, default: int, lo: int, hi: int) -> int:
            try:
                val = int(os.environ.get(name, str(default)))
            except ValueError:
                logger.warning("Invalid %s value, using default %d", name, default)
                return default
            clamped = max(lo, min(val, hi))
            if clamped != val:
                logger.info("%s=%d clamped to %d (range %d-%d)", name, val, clamped, lo, hi)
            return clamped

        self.MAX_ITERATIONS = _clamp_env("OPENLEGION_MAX_ITERATIONS", 20, 1, 100)
        self.CHAT_MAX_TOOL_ROUNDS = _clamp_env("OPENLEGION_CHAT_MAX_TOOL_ROUNDS", 30, 1, 200)
        self.CHAT_MAX_TOTAL_ROUNDS = _clamp_env("OPENLEGION_CHAT_MAX_TOTAL_ROUNDS", 200, 1, 1000)
        self.agent_id = agent_id
        self.role = role
        self.memory = memory
        self.skills = skills
        self.llm = llm
        self.mesh_client = mesh_client
        self.workspace = workspace
        self.context_manager = context_manager
        self.state: str = "idle"
        self.current_task: str | None = None
        self.tasks_completed: int = 0
        self.tasks_failed: int = 0
        self._start_time = time.time()
        self._cancel_requested: bool = False
        self._current_task_handle: asyncio.Task | None = None
        self._last_result: TaskResult | None = None
        self._chat_messages: list[dict] = []
        self._chat_lock = asyncio.Lock()
        self._chat_total_rounds: int = 0
        self._chat_auto_continues: int = 0
        self._steer_queue: asyncio.Queue[str] = asyncio.Queue()
        self._fleet_roster: list[dict] | None = None  # cached fleet info
        self._fleet_roster_ts: float = 0  # timestamp of last fetch
        self._introspect_cache: dict | None = None
        self._introspect_cache_ts: float = 0
        self._goals_cache: dict | None | object = AgentLoop._GOALS_NOT_FETCHED
        self._goals_cache_ts: float = 0
        self._loop_detector = ToolLoopDetector(
            exempt_tools=skills.get_loop_exempt_tools(),
        )
        # Merge tool exclusions: standalone agents lose blackboard tools,
        # and agents may have config-level exclusions (e.g. concierge).
        _base_excluded = _BLACKBOARD_TOOLS if mesh_client.is_standalone else frozenset()
        _cfg_excluded = excluded_tools or frozenset()
        _merged = _base_excluded | _cfg_excluded
        self._excluded_tools: frozenset[str] | None = _merged if _merged else None
        self._skills_reloaded: bool = False

    async def _fetch_fleet_roster(self) -> list[dict]:
        """Fetch and cache the fleet roster from the mesh (TTL: 10 min)."""
        if self._fleet_roster is not None and (time.time() - self._fleet_roster_ts) < _FLEET_ROSTER_TTL:
            return self._fleet_roster
        try:
            registry = await self.mesh_client.list_agents()
            roster = []
            for name, info in registry.items():
                if name == self.agent_id:
                    continue  # skip self
                entry = {"name": name}
                if isinstance(info, dict):
                    entry["role"] = info.get("role", "")
                roster.append(entry)
            self._fleet_roster = roster
            self._fleet_roster_ts = time.time()
        except Exception as e:
            logger.debug("Fleet roster fetch failed, using empty roster: %s", e)
            self._fleet_roster = []
            self._fleet_roster_ts = time.time()
        return self._fleet_roster

    def _build_fleet_context(self, roster: list[dict]) -> str:
        """Build fleet collaboration context for the system prompt."""
        if not roster:
            return ""
        lines = ["## Your Team\n"]
        lines.append("You are part of a multi-agent fleet. Your teammates:\n")
        for agent in roster:
            role = agent.get("role", "")
            if role:
                lines.append(f"- **{agent['name']}**: {role}")
            else:
                lines.append(f"- **{agent['name']}**")
        lines.append(
            "\nUse `hand_off(to=\"agent_id\", summary=\"...\")` to send work to a teammate.\n"
            "Use `check_inbox()` to see tasks sent to you.\n"
            "Use `update_status(state, summary)` so teammates know what you're doing.\n"
            "Report results to the user via chat or notify_user, not the blackboard."
        )
        return "\n".join(lines)

    _INTROSPECT_TTL = 300  # 5 minutes

    async def _fetch_introspect_cached(self) -> dict | None:
        """Fetch and cache introspect data from the mesh (TTL: 5 min).

        On cache miss (fresh fetch), also regenerates SYSTEM.md so the
        bootstrap context stays reasonably fresh without a restart.
        """
        now = time.time()
        if self._introspect_cache is not None and (now - self._introspect_cache_ts) < self._INTROSPECT_TTL:
            return self._introspect_cache
        try:
            data = await self.mesh_client.introspect("all")
            self._introspect_cache = data
            self._introspect_cache_ts = now
            # Refresh SYSTEM.md on disk so bootstrap picks it up next prompt
            if self.workspace:
                try:
                    from src.agent.workspace import generate_system_md
                    system_md = generate_system_md(
                        data, self.agent_id,
                        is_standalone=self.mesh_client.is_standalone,
                    )
                    (self.workspace.root / "SYSTEM.md").write_text(system_md)
                except Exception as e:
                    logger.debug("Failed to refresh SYSTEM.md: %s", e)
            return data
        except Exception as e:
            logger.debug("Introspect fetch failed, using cached data: %s", e)
            return self._introspect_cache

    @staticmethod
    def _format_runtime_context(data: dict, *, exclude_fleet: bool = False) -> str:
        """Format introspect data into a compact runtime context block.

        This is the authoritative source of live numbers in the system
        prompt — SYSTEM.md contains the static preamble + a startup snapshot
        while this block has fresh data fetched each turn (with a 5-min cache).

        Set *exclude_fleet* when the detailed fleet context block is already
        present (chat mode) to avoid token-wasting duplication.
        """
        lines = ["## Runtime Context\n"]

        perms = data.get("permissions")
        if perms:
            for key in INTROSPECT_PERM_KEYS:
                patterns = perms.get(key, [])
                if isinstance(patterns, list) and patterns:
                    lines.append(f"- {key}: {', '.join(str(p) for p in patterns)}")

        budget = data.get("budget")
        if budget:
            allowed = budget.get("allowed", True)
            lines.append(
                f"- Budget: daily ${budget.get('daily_used', 0):.2f}"
                f"/${budget.get('daily_limit', 0):.2f}, "
                f"monthly ${budget.get('monthly_used', 0):.2f}"
                f"/${budget.get('monthly_limit', 0):.2f}"
                + ("" if allowed else " [EXCEEDED]")
            )

        if not exclude_fleet:
            fleet = data.get("fleet")
            if fleet:
                names = [sanitize_for_prompt(str(a.get("id", "?"))) for a in fleet]
                lines.append(f"- Fleet: [{', '.join(names)}] ({len(fleet)} agents)")

        cron = data.get("cron")
        if cron:
            summaries = []
            for j in cron:
                hb = " (heartbeat)" if j.get("heartbeat") else ""
                schedule = sanitize_for_prompt(str(j.get("schedule", "?")))
                summaries.append(f"{schedule}{hb}")
            lines.append(f"- Cron: {'; '.join(summaries)}")

        return "\n".join(lines) if len(lines) > 1 else ""

    async def inject_steer(self, message: str) -> bool:
        """Inject a steer message. Returns True if agent is working."""
        await self._steer_queue.put(message)
        return self.state == "working"

    def _has_pending_steers(self) -> bool:
        """Check if steer messages are waiting without draining them."""
        return not self._steer_queue.empty()

    def _drain_steer_messages(self) -> list[str]:
        """Non-blocking drain of all pending steer messages."""
        messages = []
        while not self._steer_queue.empty():
            try:
                messages.append(self._steer_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def _check_tool_loop_terminate(self, tool_calls) -> str | None:
        """Pre-scan all tool calls in a batch for the terminate condition.

        Returns an error message if any call triggers terminate, else None.
        Called BEFORE appending the assistant message to context so we don't
        create orphaned tool_calls without matching tool results.

        Uses would_terminate() instead of check_before() to avoid duplicate
        log messages — the per-tool check_before() will log if needed.
        """
        for tc in tool_calls:
            if self._loop_detector.would_terminate(tc.name, tc.arguments):
                return (
                    f"Tool loop detected: {tc.name} called too many times "
                    f"with the same arguments. Aborting to prevent wasted spend."
                )
        return None

    async def execute_task(
        self, assignment: TaskAssignment, *, trace_id: str | None = None,
    ) -> TaskResult:
        """Main execution method. Runs bounded loop for a single task.

        CRITICAL: Maintains proper LLM conversation history with correct roles.
        Messages grow across iterations:
          user -> assistant(tool_calls) -> tool(result) -> assistant(final)
        """
        from src.shared.trace import current_trace_id
        current_trace_id.set(trace_id)
        self._loop_detector.reset()
        # State is already set to "working" by receive_task() before spawning
        # this coroutine. Setting current_task is a no-op but documents intent.
        self.current_task = assignment.task_id
        start = time.time()
        total_tokens = 0

        introspect_data = await self._fetch_introspect_cached()
        system_prompt = self._build_system_prompt(assignment, introspect_data=introspect_data)
        messages = await self._build_initial_context(assignment)

        # Decay salience scores so old facts don't dominate forever
        if self.memory:
            await self.memory.decay_all()

        try:
            for iteration in range(self.MAX_ITERATIONS):
                if self._cancel_requested:
                    self._cancel_requested = False
                    self.state = "idle"
                    self.current_task = None
                    result = TaskResult(
                        task_id=assignment.task_id,
                        status="cancelled",
                        tokens_used=total_tokens,
                        duration_ms=int((time.time() - start) * 1000),
                    )
                    self._last_result = result
                    return result

                if assignment.token_budget and not assignment.token_budget.can_spend(4096):
                    self.state = "idle"
                    self.tasks_failed += 1
                    result = TaskResult(
                        task_id=assignment.task_id,
                        status="failed",
                        error=(
                            f"Token budget exhausted: "
                            f"{assignment.token_budget.used_tokens}/{assignment.token_budget.max_tokens}"
                        ),
                        tokens_used=total_tokens,
                        duration_ms=int((time.time() - start) * 1000),
                    )
                    self._last_result = result
                    return result

                # === DECIDE (LLM call) ===
                # Refresh system prompt with context warning if applicable
                effective_system = system_prompt
                if self.context_manager:
                    warning = self.context_manager.context_warning(messages)
                    if warning:
                        effective_system = system_prompt + f"\n\n## {warning}"

                available_tools = self.skills.get_tool_definitions(exclude=self._excluded_tools) or None
                llm_response = await _llm_call_with_retry(
                    self.llm.chat,
                    system=effective_system,
                    messages=messages,
                    tools=available_tools,
                )
                total_tokens += llm_response.tokens_used
                if assignment.token_budget:
                    assignment.token_budget.record_usage(llm_response.tokens_used, self.llm.default_model)

                # Early cancel check after LLM call — avoids executing
                # tools from a response we're about to discard.
                if self._cancel_requested:
                    self._cancel_requested = False
                    self.state = "idle"
                    self.current_task = None
                    result = TaskResult(
                        task_id=assignment.task_id,
                        status="cancelled",
                        tokens_used=total_tokens,
                        duration_ms=int((time.time() - start) * 1000),
                    )
                    self._last_result = result
                    return result

                # === ACT ===
                if llm_response.tool_calls:
                    # Pre-scan for terminate BEFORE appending assistant message
                    terminate_msg = self._check_tool_loop_terminate(llm_response.tool_calls)
                    if terminate_msg:
                        self.state = "idle"
                        self.current_task = None
                        self.tasks_failed += 1
                        result = TaskResult(
                            task_id=assignment.task_id,
                            status="failed",
                            error=terminate_msg,
                            tokens_used=total_tokens,
                            duration_ms=int((time.time() - start) * 1000),
                        )
                        self._last_result = result
                        return result

                    # Append assistant response with tool calls (correct role)
                    tool_call_entries = [
                        {
                            "id": f"call_{generate_id('tc')}",
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in llm_response.tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": llm_response.content or "",
                        "tool_calls": tool_call_entries,
                    })

                    # Execute tools — parallel-safe tools run concurrently
                    tool_results = await self._run_tools_parallel(
                        llm_response.tool_calls,
                    )
                    for i, (result_str, _result) in enumerate(tool_results):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_entries[i]["id"],
                            "content": result_str,
                        })

                    # Rebuild system prompt after skill hot-reload
                    if self._skills_reloaded:
                        self._skills_reloaded = False
                        system_prompt = self._build_system_prompt(
                            assignment, introspect_data=introspect_data,
                        )

                    if self.context_manager:
                        try:
                            messages, _ = await self.context_manager.maybe_compact(system_prompt, messages)
                        except Exception as compact_err:
                            logger.warning("Task compaction failed, falling back to trim: %s", compact_err)
                            messages = self._trim_context(messages, max_tokens=_FALLBACK_MAX_TOKENS)
                    else:
                        messages = self._trim_context(messages, max_tokens=_FALLBACK_MAX_TOKENS)

                else:
                    # LLM returned text with no tool calls.
                    # If this is iteration 0, the agent hasn't used any tools,
                    # AND tools are actually available, nudge it to take action.
                    if iteration == 0 and available_tools:
                        messages.append({"role": "assistant", "content": llm_response.content or ""})
                        messages.append({
                            "role": "user",
                            "content": (
                                "You responded without using any tools. "
                                "You have tools available — use them to make progress on this task. "
                                "If you've genuinely completed the task or it's impossible, "
                                "respond with your final JSON result."
                            ),
                        })
                        continue

                    # LLM returned final answer -- task is done
                    result_data, promotions = self._parse_final_output(llm_response.content)

                    self.state = "idle"
                    self.current_task = None
                    self.tasks_completed += 1
                    duration_s = round(time.time() - start, 1)

                    logger.info(
                        f"Task {assignment.task_id} complete",
                        extra={"extra_data": {"iterations": iteration + 1, "tokens": total_tokens}},
                    )

                    # Log task completion to daily log
                    if self.workspace:
                        task_tools = self._collect_tool_names(messages)
                        input_summary = truncate(str(assignment.input_data).replace("\n", " "), 120)
                        tools_str = ", ".join(task_tools) if task_tools else "none"
                        self.workspace.append_daily_log(
                            f"Task complete: {assignment.task_type} | "
                            f"{iteration + 1} iterations, {total_tokens} tokens, {duration_s}s | "
                            f"Tools: {tools_str} | Input: {input_summary}"
                        )

                    result = TaskResult(
                        task_id=assignment.task_id,
                        status="complete",
                        result=result_data,
                        promote_to_blackboard=promotions,
                        tokens_used=total_tokens,
                        duration_ms=int(duration_s * 1000),
                    )
                    self._last_result = result
                    return result

            # Max iterations reached
            self.state = "idle"
            self.current_task = None
            self.tasks_failed += 1
            if self.workspace:
                input_summary = truncate(str(assignment.input_data).replace("\n", " "), 120)
                self.workspace.append_daily_log(
                    f"Task FAILED (max iterations): {assignment.task_type} | "
                    f"{total_tokens} tokens | Input: {input_summary}"
                )
            result = TaskResult(
                task_id=assignment.task_id,
                status="failed",
                error=f"Max iterations ({self.MAX_ITERATIONS}) reached",
                tokens_used=total_tokens,
                duration_ms=int((time.time() - start) * 1000),
            )
            self._last_result = result
            return result

        except asyncio.CancelledError:
            self.state = "idle"
            self.current_task = None
            result = TaskResult(
                task_id=assignment.task_id,
                status="cancelled",
                tokens_used=total_tokens,
                duration_ms=int((time.time() - start) * 1000),
            )
            self._last_result = result
            return result
        except Exception as e:
            self.state = "idle"
            self.current_task = None
            self.tasks_failed += 1
            logger.error(f"Task {assignment.task_id} failed: {e}", exc_info=True)
            if self.workspace:
                error_summary = truncate(str(e).replace("\n", " "), 200)
                self.workspace.append_daily_log(
                    f"Task FAILED (error): {assignment.task_type} | {error_summary}"
                )
            result = TaskResult(
                task_id=assignment.task_id,
                status="failed",
                error=str(e),
                tokens_used=total_tokens,
                duration_ms=int((time.time() - start) * 1000),
            )
            self._last_result = result
            return result

    _GOALS_NOT_FETCHED = object()  # sentinel distinct from None

    async def _fetch_goals(self) -> dict | None:
        """Read this agent's current goals from the shared blackboard (TTL: 5 min)."""
        now = time.time()
        if self._goals_cache is not self._GOALS_NOT_FETCHED and (now - self._goals_cache_ts) < _GOALS_TTL:
            return self._goals_cache
        try:
            entry = await self.mesh_client.read_blackboard(f"goals/{self.agent_id}")
            self._goals_cache = entry.get("value", entry) if entry else None
            self._goals_cache_ts = now
        except Exception as e:
            logger.debug("Failed to fetch goals for '%s': %s", self.agent_id, e)
            # Keep stale cache on failure rather than returning None
        return self._goals_cache if self._goals_cache is not self._GOALS_NOT_FETCHED else None

    async def _build_initial_context(self, assignment: TaskAssignment) -> list[dict]:
        """Build initial user message with task, goals, memory, and blackboard context."""
        parts = []

        goals = await self._fetch_goals()
        if goals:
            parts.append(f"## Your Current Goals\n{sanitize_for_prompt(format_dict(goals))}")

        sanitized_input = sanitize_for_prompt(format_dict(assignment.input_data))
        parts.append(
            f"## Task: {assignment.task_type}\n\n## Input\n{sanitized_input}"
        )

        high_salience = await self.memory.get_high_salience_facts(top_k=10)
        if high_salience:
            memory_text = "\n".join(f"- {f.key}: {f.value}" for f in high_salience)
            parts.append(f"## Your Memory (most relevant)\n{sanitize_for_prompt(memory_text)}")

        query = f"{assignment.task_type} {format_dict(assignment.input_data)}"
        relevant = await self.memory.search_hierarchical(query, top_k=10)
        seen_ids = {f.id for f in high_salience}
        novel = [f for f in relevant if f.id not in seen_ids]
        if novel:
            memory_text = "\n".join(f"- {f.key}: {f.value}" for f in novel)
            parts.append(f"## Related Memory\n{sanitize_for_prompt(memory_text)}")

        if assignment.context:
            parts.append(f"## Shared Context from Other Agents\n{sanitize_for_prompt(format_dict(assignment.context))}")

        return [{"role": "user", "content": "\n\n".join(parts)}]

    def _trim_context(self, messages: list[dict], max_tokens: int = 100_000) -> list[dict]:
        """Trim old tool exchanges to manage context window.

        Groups messages into tool-call groups (assistant+tool responses)
        so we never split a tool-call from its results.
        """
        from src.agent.context import _content_chars
        estimated_tokens = sum(
            _content_chars(m.get("content", "")) // 4 + len(json.dumps({
                k: v for k, v in m.items() if k != "content"
            })) // 4
            for m in messages
        )
        if estimated_tokens <= max_tokens:
            return messages

        # Build groups: each group is either a standalone message or
        # an assistant(tool_calls) + its following tool messages
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

        if len(groups) <= 3:
            return messages

        # Keep first group (initial user message) and last 2 groups
        first_group = groups[0]
        recent_groups = groups[-2:]
        middle_groups = groups[1:-2]

        summary_parts = []
        for group in middle_groups:
            for msg in group:
                if msg.get("role") == "tool":
                    content = msg.get("content", "")
                    # Multimodal content — extract text blocks only
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    summary_parts.append(f"Tool result: {truncate(content, 100)}")
                elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                    names = [tc["function"]["name"] for tc in msg["tool_calls"]]
                    summary_parts.append(f"Called: {', '.join(names)}")

        summary_text = "\n\n## Previous Actions (summarized)\n" + "\n".join(summary_parts)
        # Merge summary into the first user message to avoid consecutive
        # same-role messages, which violates the LLM role-alternation invariant.
        if not first_group:
            result = [{"role": "user", "content": summary_text.strip()}]
        elif first_group[0].get("role") == "user" and isinstance(first_group[0].get("content"), str):
            first_msg = {**first_group[0], "content": first_group[0]["content"] + summary_text}
            result = [first_msg] + first_group[1:]
        else:
            result = first_group + [{"role": "user", "content": summary_text.strip()}]
        for group in recent_groups:
            result.extend(group)
        return result

    async def _learn(self, tool_name: str, tool_input: dict, tool_output: Any) -> None:
        """Extract and store useful facts from successful tool execution."""
        await self.memory.log_action(
            action=f"tool:{tool_name}",
            input_summary=truncate(str(tool_input), 200),
            output_summary=truncate(str(tool_output), 200),
        )
        # Record structured tool outcome (hash-deduplicated, searchable
        # via get_tool_history).  Automatic fact extraction was removed in
        # Phase 3 — it generated 3-10 embedding calls per tool execution
        # with minimal retrieval value.  Agents retain memory_save for
        # explicit storage and context compaction captures important facts.
        await self.memory.store_tool_outcome(
            tool_name=tool_name,
            arguments=tool_input,
            outcome=truncate(str(tool_output), 500),
            success=True,
        )

    async def _record_failure(
        self, tool_name: str, error: str, context: str = "", arguments: dict | None = None,
    ) -> None:
        """Record a tool failure so the agent can avoid repeating mistakes."""
        if self.workspace:
            self.workspace.record_error(tool_name, error, context)
        await self.memory.store_tool_outcome(
            tool_name=tool_name,
            arguments=arguments,
            outcome=truncate(error, 500),
            success=False,
        )

    async def _maybe_reload_skills(self, result: Any) -> None:
        """If a tool returned reload_requested, hot-reload the skill registry.

        Sets ``_skills_reloaded`` so callers can rebuild system prompts
        with updated tool descriptions.  Re-registers with the mesh so the
        dashboard receives an ``agent_state: registered`` event and can
        refresh the capabilities view in real time.
        """
        if isinstance(result, dict) and result.get("reload_requested"):
            count = self.skills.reload()
            self._skills_reloaded = True
            logger.info(f"Hot-reloaded skills: {count} available")
            try:
                await self.mesh_client.register(
                    capabilities=self.skills.list_skills(),
                )
            except Exception as e:
                logger.warning("Failed to re-register after skill reload: %s", e)

    @staticmethod
    def _collect_tool_names(messages: list[dict]) -> list[str]:
        """Extract unique tool names from a message list, in order of first appearance."""
        seen: set[str] = set()
        names: list[str] = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc.get("function", {}).get("name", "")
                    if name and name not in seen:
                        seen.add(name)
                        names.append(name)
        return names

    def _build_tool_history_context(self, limit: int = _TOOL_HISTORY_LIMIT) -> str:
        """Build a system prompt section with recent tool outcomes."""
        history = self.memory.get_tool_history(limit=limit)
        if not history:
            return ""
        lines = []
        for h in history:
            status = "OK" if h["success"] else "FAILED"
            lines.append(f"- {h['tool_name']} [{status}]: {truncate(h['outcome'], 100)}")
        return "## Recent Tool History\n\n" + "\n".join(lines)

    def _build_system_prompt(
        self, assignment: TaskAssignment, introspect_data: dict | None = None,
    ) -> str:
        parts = []

        # Load workspace identity + project files into system prompt
        if self.workspace:
            bootstrap = self.workspace.get_bootstrap_content()
            if bootstrap:
                parts.append(bootstrap)  # pre-sanitized by workspace cache

        is_standalone = self.mesh_client.is_standalone
        rules = (
            f"You are the '{self.role}' agent in the OpenLegion multi-agent system.\n"
            f"Your current task: {assignment.task_type}\n\n"
            f"## Operating Rules\n"
            f"- Default: call tools without narration. Only narrate multi-step plans or risky actions.\n"
            f"- Be resourceful — read files, search memory, check context. "
            f"Come back with answers, not questions.\n"
            f"- If your first approach fails, try at least one alternative before reporting a blocker.\n"
            f"- Never respond with just text when a tool could make progress.\n"
            f"- Before acting on past context, run memory_search first.\n"
            f"- When done, respond with JSON: "
        )
        if is_standalone:
            rules += (
                "{\"result\": {...}}\n"
                "- Use notify_user to report results to the user.\n"
            )
        else:
            rules += (
                "{\"result\": {...}, \"promote\": {...}} "
                "('promote' = data other agents need).\n"
                "- Use notify_user for the user; blackboard for other agents only.\n"
            )
        rules += (
            f"- You have max {self.MAX_ITERATIONS} iterations.\n"
            f"- Use update_workspace to evolve your SOUL.md, INSTRUCTIONS.md, "
            f"USER.md, and HEARTBEAT.md over time.\n"
        )
        parts.append(rules)
        if self.workspace:
            learnings = self.workspace.get_learnings_context()
            if learnings:
                parts.append(f"## Learnings\n\n{learnings}")  # pre-sanitized
        tool_history = self._build_tool_history_context()
        if tool_history:
            parts.append(sanitize_for_prompt(tool_history))

        if introspect_data:
            runtime_ctx = self._format_runtime_context(introspect_data)
            if runtime_ctx:
                parts.append(runtime_ctx)

        return "\n\n".join(parts)

    def _parse_final_output(self, content: str) -> tuple[dict, dict]:
        """Parse the LLM's final response into result data and blackboard promotions."""
        try:
            parsed = json.loads(content)
            return parsed.get("result", {"raw": content}), parsed.get("promote", {})
        except (json.JSONDecodeError, AttributeError):
            return {"raw": content}, {}

    # ── Heartbeat mode ────────────────────────────────────────

    async def execute_heartbeat(self, message: str) -> dict:
        """Execute an autonomous heartbeat — stateless, separate from chat.

        Returns a structured dict with response, tools used, duration, etc.
        Does NOT touch _chat_messages.  Uses its own message list and the
        _heartbeat_mode ContextVar so tools can detect heartbeat context.
        Notifications are still persisted to the chat transcript so users
        can find them in chat history.
        """
        # Don't run if the agent is busy with a task, chat, or queued chat
        if self.state != "idle" or self._chat_lock.locked():
            return {"skipped": True, "reason": "agent_busy"}

        # Skip the LLM call entirely when HEARTBEAT.md has no actionable
        # content and no goals are set — saves tokens on empty heartbeats.
        if self.workspace and _is_heartbeat_empty(self.workspace.load_heartbeat_rules()):
            # Still need to check goals before skipping
            goals = await self._fetch_goals()
            if not goals:
                return {"skipped": True, "reason": "no_heartbeat_rules"}

        token = _heartbeat_mode.set(True)
        start = time.time()
        total_tokens = 0
        tools_used: list[str] = []
        notifications: list[str] = []
        self._loop_detector.reset()
        self.state = "working"

        try:
            # Parallel fetch of goals + introspect + fleet roster
            is_standalone = self.mesh_client.is_standalone
            if is_standalone:
                goals, introspect_data = await asyncio.gather(
                    self._fetch_goals(), self._fetch_introspect_cached(),
                )
                roster: list[dict] = []
            else:
                goals, roster, introspect_data = await asyncio.gather(
                    self._fetch_goals(), self._fetch_fleet_roster(),
                    self._fetch_introspect_cached(),
                )

            parts: list[str] = []

            # 1. Goals — the agent's north star
            if goals:
                parts.append(f"## Your Current Goals\n\n{sanitize_for_prompt(format_dict(goals))}")

            # 2. Bootstrap (identity, instructions, project)
            if self.workspace:
                bootstrap = self.workspace.get_bootstrap_content()
                if bootstrap:
                    parts.append(bootstrap)  # pre-sanitized by workspace cache

            # 3. Core rules
            inbox_line = (
                "- Call check_inbox() to see if teammates sent you tasks.\n"
                if not is_standalone else ""
            )
            nothing_clause = "goals, or inbox" if not is_standalone else "goals"
            parts.append(
                f"You are the '{self.role}' agent.\n\n"
                f"## Operating Rules\n"
                f"- This is a HEARTBEAT wakeup. Check your HEARTBEAT.md rules and "
                f"goals, then act on anything that needs attention.\n"
                f"- Follow HEARTBEAT.md strictly. Do not infer tasks from prior sessions.\n"
                f"{inbox_line}"
                f"- If nothing in HEARTBEAT.md, {nothing_clause} needs attention, reply HEARTBEAT_OK immediately.\n"
                f"- You have max {HEARTBEAT_MAX_ITERATIONS} iterations.\n"
                f"- Use notify_user to report results to the user.\n"
            )

            # 4. Learnings — avoid repeating past mistakes (half of chat cap)
            if self.workspace:
                learnings = self.workspace.get_learnings_context(max_chars=1500)
                if learnings:
                    parts.append(f"## Learnings from Past Sessions\n\n{learnings}")

            # 5. Fleet context — know your teammates (multi-agent only)
            has_fleet_ctx = False
            if roster:
                fleet_ctx = self._build_fleet_context(roster)
                if fleet_ctx:
                    parts.append(fleet_ctx)
                    has_fleet_ctx = True

            # 6. Self-evolution nudge
            parts.append(
                "## Self-Evolution\n"
                "You can update INSTRUCTIONS.md, SOUL.md, USER.md, and "
                "HEARTBEAT.md during heartbeats to improve future sessions."
            )

            # 7. Runtime context (budget, permissions, cron)
            if introspect_data:
                runtime_ctx = self._format_runtime_context(
                    introspect_data, exclude_fleet=has_fleet_ctx,
                )
                if runtime_ctx:
                    parts.append(runtime_ctx)

            system_prompt = "\n\n".join(parts)

            # Drain any pending coordination signals into the heartbeat
            steered = self._drain_steer_messages()
            if steered:
                steer_context = "\n".join(f"- {s}" for s in steered)
                message = (
                    f"{message}\n\n"
                    f"## Pending Coordination Signals\n\n{steer_context}"
                )

            # Stateless message list — fresh each heartbeat
            messages: list[dict] = [{"role": "user", "content": message}]

            for _iteration in range(HEARTBEAT_MAX_ITERATIONS):
                if self._cancel_requested:
                    self._cancel_requested = False
                    self.state = "idle"
                    duration_ms = int((time.time() - start) * 1000)
                    if self.workspace:
                        self.workspace.append_activity(
                            trigger="heartbeat",
                            summary="Cancelled",
                            tools_used=tools_used,
                            duration_ms=duration_ms,
                            tokens_used=total_tokens,
                            outcome="cancelled",
                        )
                    return {
                        "response": "",
                        "summary": "Cancelled",
                        "tools_used": tools_used,
                        "duration_ms": duration_ms,
                        "tokens_used": total_tokens,
                        "outcome": "cancelled",
                        "skipped": False,
                    }

                # When approaching the iteration limit, nudge the agent to
                # wrap up so it finishes with a proper summary instead of
                # being cut off with "Max iterations reached".
                _remaining = HEARTBEAT_MAX_ITERATIONS - _iteration
                if _remaining == 2:
                    messages.append({
                        "role": "user",
                        "content": (
                            "[SYSTEM] You have 2 iterations remaining. "
                            "Start wrapping up — use notify_user to report "
                            "your results, then give your final answer."
                        ),
                    })
                elif _remaining == 1:
                    messages.append({
                        "role": "user",
                        "content": (
                            "[SYSTEM] LAST iteration. Give your final answer "
                            "now. Do NOT call any more tools."
                        ),
                    })

                # On the very last iteration, withhold tools so the LLM is
                # forced to produce a text-only response.
                iter_tools = (
                    None if _remaining == 1
                    else self.skills.get_tool_definitions(
                        exclude=self._excluded_tools,
                    ) or None
                )

                llm_response = await _llm_call_with_retry(
                    self.llm.chat,
                    system=system_prompt,
                    messages=messages,
                    tools=iter_tools,
                )
                total_tokens += llm_response.tokens_used

                # Early cancel check after LLM call
                if self._cancel_requested:
                    self._cancel_requested = False
                    self.state = "idle"
                    duration_ms = int((time.time() - start) * 1000)
                    if self.workspace:
                        self.workspace.append_activity(
                            trigger="heartbeat",
                            summary="Cancelled",
                            tools_used=tools_used,
                            duration_ms=duration_ms,
                            tokens_used=total_tokens,
                            outcome="cancelled",
                        )
                    return {
                        "response": "",
                        "summary": "Cancelled",
                        "tools_used": tools_used,
                        "duration_ms": duration_ms,
                        "tokens_used": total_tokens,
                        "outcome": "cancelled",
                        "skipped": False,
                    }

                # On the last iteration, ignore any tool_calls — the LLM
                # shouldn't return them (tools were withheld) but guard
                # against provider edge cases.
                if _remaining == 1 and llm_response.tool_calls:
                    llm_response = LLMResponse(
                        content=llm_response.content or "Heartbeat complete.",
                        tool_calls=[],
                        tokens_used=0,
                    )

                if not llm_response.tool_calls:
                    # Final answer
                    content = llm_response.content or ""
                    duration_ms = int((time.time() - start) * 1000)

                    summary = truncate(content.replace("\n", " ").strip(), 200)
                    if self.workspace:
                        tools_str = ", ".join(tools_used) if tools_used else "none"
                        self.workspace.append_daily_log(
                            f"Heartbeat complete | {total_tokens} tokens, "
                            f"{duration_ms}ms | Tools: {tools_str}"
                        )
                        self.workspace.append_activity(
                            trigger="heartbeat",
                            summary=summary,
                            tools_used=tools_used,
                            duration_ms=duration_ms,
                            tokens_used=total_tokens,
                            outcome="ok",
                            notifications=notifications or None,
                        )

                    self.state = "idle"
                    return {
                        "response": content,
                        "summary": summary,
                        "tools_used": tools_used,
                        "duration_ms": duration_ms,
                        "tokens_used": total_tokens,
                        "outcome": "ok",
                        "skipped": False,
                    }

                # Pre-scan for terminate
                terminate_msg = self._check_tool_loop_terminate(llm_response.tool_calls)
                if terminate_msg:
                    self.state = "idle"
                    duration_ms = int((time.time() - start) * 1000)
                    if self.workspace:
                        self.workspace.append_activity(
                            trigger="heartbeat",
                            summary=f"Tool loop: {terminate_msg}",
                            tools_used=tools_used,
                            duration_ms=duration_ms,
                            tokens_used=total_tokens,
                            outcome="error",
                        )
                    return {
                        "response": terminate_msg,
                        "summary": terminate_msg,
                        "tools_used": tools_used,
                        "duration_ms": duration_ms,
                        "tokens_used": total_tokens,
                        "outcome": "error",
                        "skipped": False,
                    }

                # Execute tool calls
                tool_call_entries = [
                    {
                        "id": f"call_{generate_id('tc')}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in llm_response.tool_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": llm_response.content or "",
                    "tool_calls": tool_call_entries,
                })

                # Capture metadata before execution
                for tool_call in llm_response.tool_calls:
                    if tool_call.name not in tools_used:
                        tools_used.append(tool_call.name)
                    if tool_call.name == "notify_user":
                        msg_arg = tool_call.arguments.get("message", "")
                        if msg_arg:
                            notifications.append(msg_arg)

                # Execute tools — parallel-safe tools run concurrently
                tool_results = await self._run_tools_parallel(
                    llm_response.tool_calls,
                )
                for i, (result_str, _result) in enumerate(tool_results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_entries[i]["id"],
                        "content": result_str,
                    })

                # Clear reload flag if set (heartbeat rarely creates skills,
                # but the flag must be consumed to avoid stale state).
                self._skills_reloaded = False

                # Trim if context grows large
                messages = self._trim_context(messages, max_tokens=_FALLBACK_MAX_TOKENS)

            # Safety net — should not normally be reached because the last
            # iteration withholds tools and forces a text response.
            self.state = "idle"
            duration_ms = int((time.time() - start) * 1000)
            if self.workspace:
                self.workspace.append_activity(
                    trigger="heartbeat",
                    summary=f"Max iterations ({HEARTBEAT_MAX_ITERATIONS}) reached",
                    tools_used=tools_used,
                    duration_ms=duration_ms,
                    tokens_used=total_tokens,
                    outcome="max_iterations",
                )
            return {
                "response": f"Max iterations ({HEARTBEAT_MAX_ITERATIONS}) reached",
                "summary": f"Max iterations ({HEARTBEAT_MAX_ITERATIONS}) reached",
                "tools_used": tools_used,
                "duration_ms": duration_ms,
                "tokens_used": total_tokens,
                "outcome": "max_iterations",
                "skipped": False,
            }

        except asyncio.CancelledError:
            self.state = "idle"
            duration_ms = int((time.time() - start) * 1000)
            if self.workspace:
                self.workspace.append_activity(
                    trigger="heartbeat",
                    summary="Cancelled",
                    tools_used=tools_used,
                    duration_ms=duration_ms,
                    tokens_used=total_tokens,
                    outcome="cancelled",
                )
            raise
        except Exception as e:
            self.state = "idle"
            duration_ms = int((time.time() - start) * 1000)
            logger.error("Heartbeat failed: %s", e, exc_info=True)
            if self.workspace:
                self.workspace.append_activity(
                    trigger="heartbeat",
                    summary=f"Error: {e}",
                    tools_used=tools_used,
                    duration_ms=duration_ms,
                    tokens_used=total_tokens,
                    outcome="error",
                )
            return {
                "response": f"Error: {e}",
                "summary": f"Error: {e}",
                "tools_used": tools_used,
                "duration_ms": duration_ms,
                "tokens_used": total_tokens,
                "outcome": "error",
                "skipped": False,
            }
        finally:
            _heartbeat_mode.reset(token)

    # ── Chat mode ──────────────────────────────────────────────

    CHAT_MAX_TOOL_ROUNDS = 30
    CHAT_MAX_TOTAL_ROUNDS = 200
    _CHAT_ROUND_WARNING = 160
    _MAX_SESSION_CONTINUES = 5

    async def _auto_continue_session(self, system: str) -> None:
        """Force-compact conversation and reset round counter.

        Called when ``_chat_total_rounds`` reaches ``CHAT_MAX_TOTAL_ROUNDS``.
        Instead of killing the session, we flush facts to memory, summarize
        the conversation, and reset the counter so the session continues
        seamlessly — the same pattern used for token-based compaction.

        The round counter and loop detector are always reset, even if
        compaction fails, to prevent the session from getting stuck at the
        limit on every subsequent message.
        """
        self._chat_auto_continues += 1
        logger.info(
            "Auto-continuing chat session (continuation %d/%d, round %d)",
            self._chat_auto_continues, self._MAX_SESSION_CONTINUES,
            self._chat_total_rounds,
        )
        try:
            if self.context_manager:
                self._chat_messages = await self.context_manager.force_compact(
                    system, self._chat_messages,
                )
            else:
                self._chat_messages = self._trim_context(
                    self._chat_messages, max_tokens=_FALLBACK_MAX_TOKENS,
                )
        except Exception as e:
            logger.warning(
                "Auto-continue compaction failed, falling back to trim: %s", e,
            )
            self._chat_messages = self._trim_context(
                self._chat_messages, max_tokens=_FALLBACK_MAX_TOKENS,
            )
        self._chat_total_rounds = 0
        self._loop_detector.reset()
        if self.workspace:
            self.workspace.append_chat_message(
                "system",
                f"Session continued — conversation summarized after {self.CHAT_MAX_TOTAL_ROUNDS} turns.",
            )

    async def _maybe_restore_session(self) -> None:
        """Restore chat state from checkpoint on first call after restart."""
        if self._chat_messages or not self.memory:
            return
        try:
            cp = await self.memory._run_db(self.memory.load_chat_checkpoint)
        except Exception as e:
            logger.warning("Failed to load chat checkpoint: %s", e)
            return
        if cp is None:
            return
        self._chat_messages = cp["messages"]
        self._chat_total_rounds = cp["total_rounds"]
        self._chat_auto_continues = cp["auto_continues"]
        if self.context_manager:
            self.context_manager._flush_triggered = cp["flush_triggered"]
        logger.info(
            "chat-session-restored messages=%d rounds=%d continues=%d",
            len(self._chat_messages),
            self._chat_total_rounds,
            self._chat_auto_continues,
        )

    async def _checkpoint_chat_session(self) -> None:
        """Persist current chat state for crash recovery."""
        if not self.memory:
            return
        if not self._chat_messages:
            try:
                await self.memory._run_db(self.memory.clear_chat_checkpoint)
            except Exception as e:
                logger.debug("Failed to clear chat checkpoint: %s", e)
            return
        try:
            await self.memory._run_db(
                self.memory.save_chat_checkpoint,
                self._chat_messages,
                self._chat_total_rounds,
                self._chat_auto_continues,
                self.context_manager._flush_triggered if self.context_manager else False,
            )
        except Exception as e:
            logger.warning("Failed to save chat checkpoint: %s", e)

    async def chat(self, user_message: str, *, trace_id: str | None = None) -> dict:
        """Handle a single chat turn with persistent conversation history.

        On first message of a session, loads workspace context (INSTRUCTIONS.md,
        SOUL.md, USER.md, MEMORY.md, daily logs) into the system prompt
        and auto-searches memory for relevant context.

        Uses an asyncio.Lock so concurrent callers queue instead of being
        rejected.  The lock serialises chat turns; the /status endpoint
        remains available while the lock is held.

        Returns {"response": str, "tool_outputs": list[dict], "tokens_used": int}.
        """
        from src.shared.trace import current_trace_id
        current_trace_id.set(trace_id)
        async with self._chat_lock:
            await self._maybe_restore_session()
            try:
                return await self._chat_inner(user_message)
            finally:
                await self._checkpoint_chat_session()

    # ── Chat helpers (shared by streaming and non-streaming) ────

    async def _prepare_chat_turn(self, user_message: str) -> tuple[str, str]:
        """Set up chat context: corrections, memory, steer, system prompt.

        Returns (possibly-enriched user_message, system_prompt).
        """
        # Correction check uses only workspace + _chat_messages — no I/O,
        # safe to run before the parallel fetch.
        if (
            self.workspace
            and self._chat_messages
            and self.workspace.looks_like_correction(user_message)
        ):
            prev_assistant = next(
                (m["content"] for m in reversed(self._chat_messages) if m.get("role") == "assistant"),
                None,
            )
            if prev_assistant:
                self.workspace.record_correction(
                    original=truncate(prev_assistant, 200),
                    correction=user_message,
                )

        # Persist clean user message to transcript before enrichment
        if self.workspace:
            self.workspace.append_chat_message("user", user_message)

        if not self._chat_messages and self.workspace:
            memory_hits = self.workspace.search(
                user_message, max_results=3, exclude_files=_BOOTSTRAP_SEARCH_EXCLUDE,
            )
            if memory_hits:
                memory_context = sanitize_for_prompt("\n".join(
                    f"- [{h['file']}] {h['snippet']}" for h in memory_hits
                ))
                user_message = (
                    f"{user_message}\n\n"
                    f"[Relevant memory auto-loaded]\n{memory_context}"
                )

        # Enrich with multimodal blocks for images/PDFs attached via the UI.
        # The plain-text message was already persisted to the transcript above;
        # the enriched form is only used for the LLM call.
        llm_content = enrich_message_with_attachments(user_message)
        self._chat_messages.append({"role": "user", "content": llm_content})
        steered = self._drain_steer_messages()
        if steered:
            combined = "\n\n".join(steered)
            steer_suffix = f"\n\n[Additional context]: {combined}"
            current = self._chat_messages[-1]["content"]
            if isinstance(current, list):
                # Multimodal content — append steer as a new text block
                self._chat_messages[-1]["content"].append(
                    {"type": "text", "text": steer_suffix.strip()}
                )
            else:
                self._chat_messages[-1]["content"] += steer_suffix
            # Persist steers as separate transcript entries
            if self.workspace:
                for s in steered:
                    self.workspace.append_chat_message("user", f"[steer] {s}")

        # Parallel fetch: goals, fleet roster (if multi-agent), introspect.
        # Saves 30-100ms per turn vs sequential requests.
        if self.mesh_client.is_standalone:
            goals, introspect_data = await asyncio.gather(
                self._fetch_goals(), self._fetch_introspect_cached(),
            )
            roster: list[dict] = []
        else:
            goals, roster, introspect_data = await asyncio.gather(
                self._fetch_goals(), self._fetch_fleet_roster(),
                self._fetch_introspect_cached(),
            )
        system = self._build_chat_system_prompt(
            goals=goals, fleet_roster=roster, introspect_data=introspect_data,
        )
        return user_message, system

    def _build_tool_call_entries(self, llm_response) -> list[dict]:
        """Build tool-call entry dicts and append assistant message."""
        entries = [
            {
                "id": f"call_{generate_id('tc')}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in llm_response.tool_calls
        ]
        self._chat_messages.append({
            "role": "assistant",
            "content": llm_response.content or "",
            "tool_calls": entries,
        })
        return entries

    async def _run_tool(self, tool_call) -> tuple[str | list, dict]:
        """Execute a single tool call with loop detection, learning, and error handling.

        Returns (content, result_dict) for the caller to append to messages.
        ``content`` is a plain string for text-only results, or a list of
        content blocks (text + image_url) when the tool returns an ``_image``
        key.  Shared by both task mode and chat mode.
        """
        loop_verdict = self._loop_detector.check_before(tool_call.name, tool_call.arguments)

        if loop_verdict in ("block", "terminate"):
            block_error = (
                f"Tool loop detected: {tool_call.name} has been called "
                "repeatedly with the same arguments and is producing the same "
                "result. Try a different approach or different arguments."
            )
            result_str = json.dumps({"error": block_error})
            result = {"error": block_error}
            self._loop_detector.record(tool_call.name, tool_call.arguments, result_str)
            return result_str, result

        try:
            result = await asyncio.wait_for(
                self.skills.execute(
                    tool_call.name,
                    tool_call.arguments,
                    mesh_client=self.mesh_client,
                    workspace_manager=self.workspace,
                    memory_store=self.memory,
                ),
                timeout=_TOOL_TIMEOUT,
            )

            # Pop _image before JSON serialization — keeps base64 out of
            # result_str so loop detection, learning, and event streaming
            # never see the massive blob.
            image_block = None
            if isinstance(result, dict):
                image_block = result.pop("_image", None)

            try:
                result_str = json.dumps(result, default=str) if isinstance(result, dict) else str(result)
            except (TypeError, ValueError, OverflowError) as ser_err:
                logger.warning("JSON serialization of %s result failed: %s", tool_call.name, ser_err)
                result_str = str(result)[:2000]
            result_str = sanitize_for_prompt(result_str)
            self._loop_detector.record(tool_call.name, tool_call.arguments, result_str)
            if loop_verdict == "warn":
                result_str = (
                    "[WARNING: You have called this tool multiple times with "
                    "identical arguments and received the same result. Consider "
                    "a different approach.]\n" + result_str
                )
            try:
                await self._learn(tool_call.name, tool_call.arguments, result)
                await self._maybe_reload_skills(result)
            except Exception as learn_err:
                logger.warning("Post-tool learning failed for %s: %s", tool_call.name, learn_err)

            # Build multimodal content when an image is present
            if image_block and isinstance(image_block, dict) and image_block.get("data"):
                media_type = image_block.get("media_type", "image/png")
                data_uri = f"data:{media_type};base64,{image_block['data']}"
                content: str | list = [
                    {"type": "text", "text": result_str},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]
                return content, result

            return result_str, result
        except asyncio.TimeoutError:
            result_str = json.dumps({"error": f"Tool {tool_call.name} timed out after {_TOOL_TIMEOUT}s"})
            result_str = sanitize_for_prompt(result_str)
            self._loop_detector.record(tool_call.name, tool_call.arguments, result_str)
            result = {"error": f"Timed out after {_TOOL_TIMEOUT}s"}
            logger.error(f"Tool {tool_call.name} timed out after {_TOOL_TIMEOUT}s")
            await self._record_failure(
                tool_call.name, f"Timed out after {_TOOL_TIMEOUT}s",
                truncate(str(tool_call.arguments), 200),
                arguments=tool_call.arguments,
            )
            return result_str, result
        except Exception as e:
            result_str = json.dumps({"error": str(e)})
            result_str = sanitize_for_prompt(result_str)
            self._loop_detector.record(tool_call.name, tool_call.arguments, result_str)
            result = {"error": str(e)}
            logger.error(f"Tool {tool_call.name} failed: {e}")
            await self._record_failure(
                tool_call.name, str(e),
                truncate(str(tool_call.arguments), 200),
                arguments=tool_call.arguments,
            )
            return result_str, result

    async def _run_tools_parallel(
        self,
        tool_calls: list,
    ) -> list[tuple[str | list, dict]]:
        """Execute tool calls with parallel-safe tools gathered concurrently.

        Partitions tool calls into batches: consecutive parallel-safe tools
        run via ``asyncio.gather``, non-parallel-safe tools run sequentially.
        Returns results in the original order.

        Uses ``return_exceptions=True`` so individual tool failures don't
        abort the whole batch — consistent with Phase 1 error-fill behavior.
        """
        if len(tool_calls) <= 1:
            return [await self._run_tool(tool_calls[0])]

        # Partition into batches: (start_idx, end_idx, is_parallel)
        batches: list[tuple[int, int, bool]] = []
        i = 0
        n = len(tool_calls)
        while i < n:
            safe = self.skills.is_parallel_safe(tool_calls[i].name)
            j = i + 1
            if safe:
                while j < n and self.skills.is_parallel_safe(tool_calls[j].name):
                    j += 1
            batches.append((i, j, safe and (j - i) > 1))
            i = j

        results: list[tuple[str | list, dict] | None] = [None] * n

        for start, end, is_parallel in batches:
            if is_parallel:
                coros = [self._run_tool(tool_calls[k]) for k in range(start, end)]
                batch_results = await asyncio.gather(*coros, return_exceptions=True)
                for k, br in enumerate(batch_results):
                    idx = start + k
                    if isinstance(br, BaseException):
                        err_str = json.dumps({"error": f"Tool execution failed: {br}"})
                        results[idx] = (err_str, {"error": str(br)})
                    else:
                        results[idx] = br
            else:
                for k in range(start, end):
                    results[k] = await self._run_tool(tool_calls[k])

        return results  # type: ignore[return-value]

    async def _execute_chat_tools_parallel(
        self,
        tool_calls: list,
        entries: list[dict],
        tool_outputs: list[dict],
    ) -> None:
        """Execute chat tool calls with parallel-safe tools gathered concurrently.

        Appends results to ``self._chat_messages`` in the original order.
        """
        tool_results = await self._run_tools_parallel(tool_calls)
        for i, (result_str, result) in enumerate(tool_results):
            self._chat_messages.append({
                "role": "tool",
                "tool_call_id": entries[i]["id"],
                "content": result_str,
            })
            tool_outputs.append({
                "tool": tool_calls[i].name,
                "input": tool_calls[i].arguments,
                "output": result,
            })

    async def _compact_chat_context(self, system: str) -> None:
        """Run context compaction and drain any pending steer messages."""
        if self.context_manager:
            try:
                self._chat_messages, compacted = await self.context_manager.maybe_compact(
                    system, self._chat_messages,
                )
                if compacted and self.workspace:
                    self.workspace.append_chat_message(
                        "system",
                        "Context compacted — key facts saved to memory, conversation summarized.",
                    )
            except Exception as e:
                logger.warning("Context compaction failed, falling back to trim: %s", e)
                self._chat_messages = self._trim_context(
                    self._chat_messages, max_tokens=_FALLBACK_MAX_TOKENS,
                )
        else:
            self._chat_messages = self._trim_context(self._chat_messages, max_tokens=_FALLBACK_MAX_TOKENS)

        steered = self._drain_steer_messages()
        if steered:
            combined = "\n\n".join(f"[User interjection]: {s}" for s in steered)
            self._chat_messages.append({"role": "user", "content": combined})
            if self.workspace:
                for s in steered:
                    self.workspace.append_chat_message("user", f"[steer] {s}")

    @staticmethod
    def _resolve_content(llm_response) -> str:
        """Extract text content, suppressing silent acknowledgments.

        Falls back to ``thinking_content`` for models that return only
        reasoning tokens (Qwen3, DeepSeek-R1).  Strips ``<think>`` tags
        and JSON chain-of-thought wrappers so that chat history and
        displayed bubbles contain the answer only.
        """
        content = llm_response.content or ""
        if content and content.strip() == SILENT_REPLY_TOKEN:
            content = ""
        # Fall back to thinking content when the model produced only
        # reasoning tokens (common with Ollama thinking models).
        if not content and llm_response.thinking_content:
            content = llm_response.thinking_content
        # Strip <think>…</think> blocks so conversation history stays
        # lean and the chat bubble shows the answer, not internal reasoning.
        content = _strip_think_tags(content)
        # Some models (Qwen3) wrap their answer in a JSON object with a
        # "response" key.  Extract just the response value.
        content = _extract_json_response(content)
        return content

    # ── Non-streaming chat ────────────────────────────────────

    async def _chat_inner(self, user_message: str) -> dict:
        self.state = "working"
        total_tokens = 0
        tool_outputs: list[dict] = []

        try:
            user_message, system = await self._prepare_chat_turn(user_message)

            if self._chat_total_rounds >= self.CHAT_MAX_TOTAL_ROUNDS:
                if self._chat_auto_continues >= self._MAX_SESSION_CONTINUES:
                    self.state = "idle"
                    msg = (
                        "Chat session has reached its absolute limit "
                        f"({self._MAX_SESSION_CONTINUES} continuations × "
                        f"{self.CHAT_MAX_TOTAL_ROUNDS} rounds). "
                        "Please reset the chat to continue."
                    )
                    if self.workspace:
                        self.workspace.append_chat_message("assistant", msg)
                    return {"response": msg, "tool_outputs": [], "tokens_used": 0}
                await self._auto_continue_session(system)

            steer_interrupts = 0
            for _ in range(self.CHAT_MAX_TOOL_ROUNDS):
                llm_response = await _llm_call_with_retry(
                    self.llm.chat,
                    system=system,
                    messages=self._chat_messages,
                    tools=self.skills.get_tool_definitions(exclude=self._excluded_tools) or None,
                )
                total_tokens += llm_response.tokens_used

                if not llm_response.tool_calls:
                    content = self._resolve_content(llm_response)
                    # Check for steers that arrived during the LLM call.
                    # If present, keep the assistant's response in context,
                    # inject steers as user interjection, and continue the
                    # loop so the LLM can adjust its answer.
                    if (
                        self._has_pending_steers()
                        and steer_interrupts < _MAX_STEER_INTERRUPTS
                    ):
                        steer_interrupts += 1
                        self._chat_messages.append({"role": "assistant", "content": content})
                        steered = self._drain_steer_messages()
                        combined = "\n\n".join(
                            f"[User interjection]: {s}" for s in steered
                        )
                        self._chat_messages.append({"role": "user", "content": combined})
                        if self.workspace:
                            for s in steered:
                                self.workspace.append_chat_message("user", f"[steer] {s}")
                        self._chat_total_rounds += 1
                        await self._compact_chat_context(system)
                        continue
                    self._chat_messages.append({"role": "assistant", "content": content})
                    self._log_chat_turn(user_message, content)
                    self.state = "idle"
                    return {
                        "response": content,
                        "tool_outputs": tool_outputs,
                        "tokens_used": total_tokens,
                    }

                # Pre-scan for terminate before appending assistant message
                terminate_msg = self._check_tool_loop_terminate(llm_response.tool_calls)
                if terminate_msg:
                    self.state = "idle"
                    msg = f"Stopped: {terminate_msg}"
                    if self.workspace:
                        self.workspace.append_chat_message("assistant", msg)
                    return {
                        "response": msg,
                        "tool_outputs": tool_outputs,
                        "tokens_used": total_tokens,
                    }

                entries = self._build_tool_call_entries(llm_response)
                try:
                    await self._execute_chat_tools_parallel(
                        llm_response.tool_calls, entries, tool_outputs,
                    )
                except Exception as tool_err:
                    logger.error("Chat tool batch raised unexpected error: %s", tool_err)
                    # Error-fill any missing tool results to maintain role alternation
                    existing_ids = {
                        m["tool_call_id"]
                        for m in self._chat_messages
                        if m.get("role") == "tool" and "tool_call_id" in m
                    }
                    for entry in entries:
                        if entry["id"] not in existing_ids:
                            self._chat_messages.append({
                                "role": "tool",
                                "tool_call_id": entry["id"],
                                "content": json.dumps({"error": f"Internal error: {tool_err}"}),
                            })
                self._chat_total_rounds += 1

                # If skills were hot-reloaded during tool execution,
                # rebuild the system prompt so tool descriptions stay in sync.
                if self._skills_reloaded:
                    self._skills_reloaded = False
                    system = self._build_chat_system_prompt(
                        goals=self._goals_cache if self._goals_cache is not self._GOALS_NOT_FETCHED else None,
                        fleet_roster=self._fleet_roster,
                        introspect_data=self._introspect_cache,
                    )

                if self._chat_total_rounds >= self.CHAT_MAX_TOTAL_ROUNDS:
                    if self._chat_auto_continues >= self._MAX_SESSION_CONTINUES:
                        logger.warning("Chat session hit absolute limit (%d continues)", self._MAX_SESSION_CONTINUES)
                        break
                    await self._auto_continue_session(system)

                await self._compact_chat_context(system)

            # Max tool rounds exhausted — force final text response.
            # Omit tools so the LLM cannot return more tool calls.
            llm_response = await _llm_call_with_retry(
                self.llm.chat,
                system=system,
                messages=self._chat_messages,
                tools=None,
            )
            total_tokens += llm_response.tokens_used
            content = self._resolve_content(llm_response)
            self._chat_messages.append({"role": "assistant", "content": content})
            self._log_chat_turn(user_message, content)
            self.state = "idle"
            return {
                "response": content,
                "tool_outputs": tool_outputs,
                "tokens_used": total_tokens,
                "tool_limit_reached": True,
            }

        except asyncio.CancelledError:
            self.state = "idle"
            raise
        except Exception as e:
            self.state = "idle"
            logger.error(f"Chat failed: {e}", exc_info=True)
            msg = f"Error: {e}"
            if self.workspace:
                self.workspace.append_chat_message("assistant", msg)
            return {"response": msg, "tool_outputs": tool_outputs, "tokens_used": total_tokens}

    def _log_chat_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Append a rich summary of the chat turn to the daily log."""
        if not self.workspace:
            return
        # Skip logging for suppressed responses (__SILENT__ → empty string)
        if not assistant_msg.strip():
            return
        # Strip auto-loaded memory context from user message before summarizing
        clean_user = user_msg.split("\n[Relevant memory auto-loaded]")[0]
        user_summary = truncate(clean_user.replace("\n", " ").strip(), 120)

        # Collect tool names used in the current turn (chronological order).
        # Find the last user message index, then collect from there forward.
        last_user_idx = -1
        for i in range(len(self._chat_messages) - 1, -1, -1):
            if self._chat_messages[i].get("role") == "user":
                last_user_idx = i
                break
        tool_names: list[str] = []
        for msg in self._chat_messages[last_user_idx + 1:]:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc.get("function", {}).get("name", "")
                    if name and name not in tool_names:
                        tool_names.append(name)

        # Multi-line aware response summary
        response_lines = [line.strip() for line in assistant_msg.splitlines() if line.strip()]
        if len(response_lines) <= 2:
            response_summary = " ".join(response_lines)
        else:
            response_summary = f"{response_lines[0]} (+{len(response_lines)-1} lines)"
        response_summary = truncate(response_summary, 200)

        parts = [f"Chat: {user_summary}"]
        if tool_names:
            parts.append(f"Tools: {', '.join(tool_names)}")
        parts.append(f"Response: {response_summary}")
        self.workspace.append_daily_log(" | ".join(parts))

        # Persist assistant response to transcript
        self.workspace.append_chat_message(
            "assistant", assistant_msg,
            tool_names=tool_names or None,
        )

    def get_chat_messages(self) -> list[dict]:
        """Return chat messages suitable for history restoration.

        Reads from the persistent transcript file so history survives
        context compaction, container restarts, and is accessible from
        any device.  Falls back to filtering in-memory messages when
        the workspace is unavailable (tests, no transcript yet).
        """
        if self.workspace:
            transcript = self.workspace.load_chat_transcript()
            if transcript:
                return transcript
        # Fallback: filter in-memory messages
        result = []
        for m in self._chat_messages:
            role = m.get("role", "unknown")
            if role == "tool":
                continue
            content = m.get("content", "")
            if isinstance(content, str):
                content = sanitize_for_prompt(content)
            result.append({"role": role, "content": content})
        return result

    async def reset_chat(self) -> None:
        """Clear conversation history. Flushes important facts to memory
        before clearing (unless the conversation was mostly errors).
        Acquires the chat lock to avoid corrupting state during an active
        chat turn."""
        async with self._chat_lock:
            if self._chat_messages and self.context_manager:
                # Skip flush if the conversation is dominated by tool errors
                # — extracting "facts" from error messages poisons memory.
                error_count = sum(
                    1 for m in self._chat_messages
                    if m.get("role") == "tool"
                    and isinstance(m.get("content", ""), str)
                    and '"error"' in m.get("content", "")
                )
                tool_count = sum(1 for m in self._chat_messages if m.get("role") == "tool")
                should_flush = tool_count == 0 or error_count < tool_count * 0.5
                if should_flush:
                    try:
                        await self.context_manager._flush_to_memory(
                            "", self._chat_messages,
                        )
                    except Exception as e:
                        logger.warning("Failed to flush memory on chat reset: %s", e)
                else:
                    logger.info(
                        "Skipping memory flush on reset — conversation had "
                        "%d/%d tool errors", error_count, tool_count,
                    )
            # Archive transcript before clearing in-memory state
            if self.workspace:
                self.workspace.archive_chat_transcript()
            self._chat_messages = []
            self._chat_total_rounds = 0
            self._chat_auto_continues = 0
            self._loop_detector.reset()
            if self.context_manager:
                self.context_manager.reset()
            await self._checkpoint_chat_session()

    def _build_chat_system_prompt(
        self,
        goals: dict | None = None,
        fleet_roster: list[dict] | None = None,
        introspect_data: dict | None = None,
    ) -> str:
        parts = []

        if goals:
            parts.append(f"## Your Current Goals\n\n{sanitize_for_prompt(format_dict(goals))}")

        if self.workspace:
            bootstrap = self.workspace.get_bootstrap_content()
            if bootstrap:
                parts.append(bootstrap)  # pre-sanitized by workspace cache

            learnings = self.workspace.get_learnings_context()
            if learnings:
                parts.append(f"## Learnings from Past Sessions\n\n{learnings}")  # pre-sanitized

        has_browser = (
            "browser_navigate" in self.skills.skills
            and (not self._excluded_tools or "browser_navigate" not in self._excluded_tools)
        )

        is_standalone = self.mesh_client.is_standalone
        rules = (
            f"You are the '{self.role}' agent in the OpenLegion fleet.\n\n"
            f"## Operating Rules\n"
            f"- Default: call tools without narration. Only narrate multi-step plans or risky actions.\n"
            f"- Be resourceful — read files, search memory, check context. "
            f"Come back with answers, not questions.\n"
            f"- If your first approach fails, try at least one alternative before reporting a blocker.\n"
            f"- Make decisions with reasonable defaults. Ask only when truly ambiguous.\n"
            f"- Never respond with just text when a tool could make progress.\n"
        )
        if is_standalone:
            rules += "- Use notify_user to report results to the user.\n"
        else:
            rules += "- Use notify_user for the user; blackboard for other agents only.\n"
        rules += (
            "- Before answering from memory, run memory_search first.\n"
            "- Use update_workspace to save lasting knowledge and user preferences.\n"
        )

        if has_browser:
            rules += (
                "\n## Browser\n"
                "browser_navigate → browser_get_elements (read refs) → "
                "browser_click(ref=)/browser_type(ref=). Always re-snapshot "
                "after state-changing actions. Use snapshot_after=true to "
                "combine action + snapshot in one call.\n"
            )

        parts.append(rules)

        # Fleet collaboration context (only for multi-agent setups)
        has_fleet_ctx = False
        if fleet_roster:
            fleet_ctx = self._build_fleet_context(fleet_roster)
            if fleet_ctx:
                parts.append(fleet_ctx)
                has_fleet_ctx = True

        tool_history = self._build_tool_history_context()
        if tool_history:
            parts.append(sanitize_for_prompt(tool_history))

        if introspect_data:
            runtime_ctx = self._format_runtime_context(
                introspect_data, exclude_fleet=has_fleet_ctx,
            )
            if runtime_ctx:
                parts.append(runtime_ctx)

        # Context usage warning at 80%+
        if self.context_manager and self._chat_messages:
            warning = self.context_manager.context_warning(self._chat_messages)
            if warning:
                parts.append(f"## {warning}")

        # Round-count warning at 80% of checkpoint interval
        if self._chat_total_rounds >= self._CHAT_ROUND_WARNING:
            remaining = self.CHAT_MAX_TOTAL_ROUNDS - self._chat_total_rounds
            parts.append(
                f"## Session Note\n"
                f"This session has been running for {self._chat_total_rounds} tool rounds. "
                f"Context will be auto-refreshed in ~{remaining} rounds. "
                f"Consider saving important context to memory if you haven't already."
            )

        return "\n\n".join(parts)

    def get_status(self) -> AgentStatus:
        """Return current agent status."""
        ctx_tokens = 0
        ctx_max = 0
        ctx_pct = 0.0
        if self.context_manager:
            ctx_max = self.context_manager.max_tokens
            if self._chat_messages:
                ctx_tokens = self.context_manager.token_count(self._chat_messages)
                ctx_pct = round(ctx_tokens / ctx_max, 4) if ctx_max else 0.0
        return AgentStatus(
            agent_id=self.agent_id,
            role=self.role,
            state=self.state,
            current_task=self.current_task,
            capabilities=self.skills.list_skills(exclude=self._excluded_tools),
            uptime_seconds=time.time() - self._start_time,
            tasks_completed=self.tasks_completed,
            tasks_failed=self.tasks_failed,
            context_tokens=ctx_tokens,
            context_max=ctx_max,
            context_pct=ctx_pct,
        )

    # ── Streaming chat ────────────────────────────────────────

    async def chat_stream(self, user_message: str, *, trace_id: str | None = None):
        """Streaming chat that yields SSE events as they happen.

        Events yielded (as dicts, caller serialises to SSE):
          {"type": "tool_start", "name": str, "input": dict}
          {"type": "tool_result", "name": str, "output": any}
          {"type": "text_delta", "content": str}
          {"type": "done", "response": str, "tool_outputs": list, "tokens_used": int}
        """
        from src.shared.trace import current_trace_id
        current_trace_id.set(trace_id)
        async with self._chat_lock:
            await self._maybe_restore_session()
            try:
                async for event in self._chat_stream_inner(user_message):
                    yield event
            finally:
                await self._checkpoint_chat_session()

    async def _chat_stream_inner(self, user_message: str):
        self.state = "working"
        total_tokens = 0
        tool_outputs: list[dict] = []

        try:
            user_message, system = await self._prepare_chat_turn(user_message)

            if self._chat_total_rounds >= self.CHAT_MAX_TOTAL_ROUNDS:
                if self._chat_auto_continues >= self._MAX_SESSION_CONTINUES:
                    msg = (
                        "Chat session has reached its absolute limit "
                        f"({self._MAX_SESSION_CONTINUES} continuations × "
                        f"{self.CHAT_MAX_TOTAL_ROUNDS} rounds). "
                        "Please reset the chat to continue."
                    )
                    if self.workspace:
                        self.workspace.append_chat_message("assistant", msg)
                    yield {"type": "text_delta", "content": msg}
                    yield {"type": "done", "response": msg, "tool_outputs": [], "tokens_used": 0}
                    return
                await self._auto_continue_session(system)

            steer_interrupts = 0
            for _ in range(self.CHAT_MAX_TOOL_ROUNDS):
                # Try token-level streaming, fall back to non-streaming on error
                llm_response = None
                used_streaming = False
                any_text_streamed = False
                tools = self.skills.get_tool_definitions(exclude=self._excluded_tools) or None
                try:
                    async for event in self.llm.chat_stream(
                        system=system, messages=self._chat_messages, tools=tools,
                    ):
                        etype = event.get("type", "")
                        if etype == "text_delta":
                            any_text_streamed = True
                            yield event  # Forward token to caller immediately
                        elif etype == "done":
                            llm_response = event["response"]
                    used_streaming = True
                except Exception as e:
                    logger.warning(f"LLM streaming failed ({e}), falling back to non-streaming")

                streamed = llm_response is not None
                if llm_response is None:
                    if used_streaming:
                        logger.warning("LLM stream ended without done event, falling back")
                    llm_response = await _llm_call_with_retry(
                        self.llm.chat, system=system, messages=self._chat_messages, tools=tools,
                    )

                total_tokens += llm_response.tokens_used

                if not llm_response.tool_calls:
                    content = self._resolve_content(llm_response)
                    # Check for steers that arrived during the LLM call.
                    if (
                        self._has_pending_steers()
                        and steer_interrupts < _MAX_STEER_INTERRUPTS
                    ):
                        steer_interrupts += 1
                        if not streamed and not any_text_streamed and content:
                            yield {"type": "text_delta", "content": content}
                        self._chat_messages.append({"role": "assistant", "content": content})
                        steered = self._drain_steer_messages()
                        combined = "\n\n".join(
                            f"[User interjection]: {s}" for s in steered
                        )
                        self._chat_messages.append({"role": "user", "content": combined})
                        if self.workspace:
                            for s in steered:
                                self.workspace.append_chat_message("user", f"[steer] {s}")
                        self._chat_total_rounds += 1
                        await self._compact_chat_context(system)
                        continue
                    # Emit text_delta for non-streaming fallback only if no tokens
                    # were already streamed (avoids doubled content on partial failure)
                    if not streamed and not any_text_streamed and content:
                        yield {"type": "text_delta", "content": content}
                    self._chat_messages.append({"role": "assistant", "content": content})
                    self._log_chat_turn(user_message, content)
                    yield {
                        "type": "done",
                        "response": content,
                        "tool_outputs": tool_outputs,
                        "tokens_used": total_tokens,
                    }
                    return

                # Pre-scan for terminate before appending assistant message
                terminate_msg = self._check_tool_loop_terminate(llm_response.tool_calls)
                if terminate_msg:
                    msg = f"Stopped: {terminate_msg}"
                    if self.workspace:
                        self.workspace.append_chat_message("assistant", msg)
                    yield {
                        "type": "done",
                        "response": msg,
                        "tool_outputs": tool_outputs,
                        "tokens_used": total_tokens,
                    }
                    return

                entries = self._build_tool_call_entries(llm_response)
                # Emit tool_start events for all tools upfront
                for tool_call in llm_response.tool_calls:
                    yield {"type": "tool_start", "name": tool_call.name, "input": tool_call.arguments}
                try:
                    await self._execute_chat_tools_parallel(
                        llm_response.tool_calls, entries, tool_outputs,
                    )
                    # Emit tool_result events for all completed tools
                    for output in tool_outputs[-len(llm_response.tool_calls):]:
                        yield {"type": "tool_result", "name": output["tool"], "output": output["output"]}
                except Exception as tool_err:
                    logger.error("Chat tool batch raised unexpected error: %s", tool_err)
                    existing_ids = {
                        m["tool_call_id"]
                        for m in self._chat_messages
                        if m.get("role") == "tool" and "tool_call_id" in m
                    }
                    for idx, entry in enumerate(entries):
                        if entry["id"] not in existing_ids:
                            self._chat_messages.append({
                                "role": "tool",
                                "tool_call_id": entry["id"],
                                "content": json.dumps({"error": f"Internal error: {tool_err}"}),
                            })
                            yield {
                                "type": "tool_result",
                                "name": llm_response.tool_calls[idx].name,
                                "output": {"error": str(tool_err)},
                            }
                self._chat_total_rounds += 1

                # Rebuild system prompt after skill hot-reload
                if self._skills_reloaded:
                    self._skills_reloaded = False
                    system = self._build_chat_system_prompt(
                        goals=self._goals_cache if self._goals_cache is not self._GOALS_NOT_FETCHED else None,
                        fleet_roster=self._fleet_roster,
                        introspect_data=self._introspect_cache,
                    )

                if self._chat_total_rounds >= self.CHAT_MAX_TOTAL_ROUNDS:
                    if self._chat_auto_continues >= self._MAX_SESSION_CONTINUES:
                        logger.warning("Chat session hit absolute limit (%d continues)", self._MAX_SESSION_CONTINUES)
                        break
                    await self._auto_continue_session(system)

                await self._compact_chat_context(system)

            # Max tool rounds exhausted — force final response (non-streaming ok).
            # Omit tools so the LLM cannot return more tool calls.
            llm_response = await _llm_call_with_retry(
                self.llm.chat,
                system=system,
                messages=self._chat_messages,
                tools=None,
            )
            total_tokens += llm_response.tokens_used
            content = self._resolve_content(llm_response)
            if content:
                yield {"type": "text_delta", "content": content}
            self._chat_messages.append({"role": "assistant", "content": content})
            self._log_chat_turn(user_message, content)
            yield {
                "type": "done",
                "response": content,
                "tool_outputs": tool_outputs,
                "tokens_used": total_tokens,
                "tool_limit_reached": True,
            }

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Streaming chat failed: {e}", exc_info=True)
            msg = f"Error: {e}"
            if self.workspace:
                self.workspace.append_chat_message("assistant", msg)
            yield {
                "type": "done",
                "response": msg,
                "tool_outputs": tool_outputs,
                "tokens_used": total_tokens,
            }
        finally:
            self.state = "idle"
