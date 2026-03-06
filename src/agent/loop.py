"""Bounded agent execution loop.

Each task runs: perceive -> decide (LLM) -> act (tool) -> learn.
Max 20 iterations per task. Proper LLM tool-calling message roles.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Optional

import httpx

from src.agent.loop_detector import ToolLoopDetector
from src.agent.workspace import INTROSPECT_PERM_KEYS
from src.shared.types import SILENT_REPLY_TOKEN, AgentStatus, TaskAssignment, TaskResult
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
_TOOL_TIMEOUT = 300  # seconds — hard ceiling for a single tool execution

# Tools that require a project blackboard — excluded for standalone agents.
_BLACKBOARD_TOOLS = frozenset({
    "read_shared_state", "write_shared_state", "list_shared_state",
    "publish_event", "subscribe_event", "watch_blackboard",
    "claim_task",
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
        except (httpx.ConnectError, httpx.TimeoutException) as e:
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
        workspace: Optional[WorkspaceManager] = None,
        context_manager: Optional[ContextManager] = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.memory = memory
        self.skills = skills
        self.llm = llm
        self.mesh_client = mesh_client
        self.workspace = workspace
        self.context_manager = context_manager
        self.state: str = "idle"
        self.current_task: Optional[str] = None
        self.tasks_completed: int = 0
        self.tasks_failed: int = 0
        self._start_time = time.time()
        self._cancel_requested: bool = False
        self._current_task_handle: Optional[asyncio.Task] = None
        self._last_result: Optional[TaskResult] = None
        self._chat_messages: list[dict] = []
        self._chat_lock = asyncio.Lock()
        self._chat_total_rounds: int = 0
        self._steer_queue: asyncio.Queue[str] = asyncio.Queue()
        self._fleet_roster: list[dict] | None = None  # cached fleet info
        self._fleet_roster_ts: float = 0  # timestamp of last fetch
        self._introspect_cache: dict | None = None
        self._introspect_cache_ts: float = 0
        self._loop_detector = ToolLoopDetector()
        # Standalone agents have no project blackboard — hide those tools
        self._excluded_tools: frozenset[str] | None = (
            _BLACKBOARD_TOOLS if mesh_client.is_standalone else None
        )

    async def _fetch_fleet_roster(self) -> list[dict]:
        """Fetch and cache the fleet roster from the mesh (TTL: 10 min)."""
        if self._fleet_roster is not None and (time.time() - self._fleet_roster_ts) < 600:
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
            "\nCoordinate via blackboard — write only data a teammate needs to act on.\n"
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

                llm_response = await _llm_call_with_retry(
                    self.llm.chat,
                    system=effective_system,
                    messages=messages,
                    tools=self.skills.get_tool_definitions(exclude=self._excluded_tools) or None,
                )
                total_tokens += llm_response.tokens_used
                if assignment.token_budget:
                    assignment.token_budget.record_usage(llm_response.tokens_used, self.llm.default_model)

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

                    # Execute each tool and append results with CORRECT role
                    for i, tool_call in enumerate(llm_response.tool_calls):
                        result_str, _result = await self._run_tool(tool_call)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_entries[i]["id"],
                            "content": result_str,
                        })

                    if self.context_manager:
                        messages = await self.context_manager.maybe_compact(system_prompt, messages)
                    else:
                        messages = self._trim_context(messages, max_tokens=100_000)

                else:
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

    async def _fetch_goals(self) -> dict | None:
        """Read this agent's current goals from the shared blackboard."""
        try:
            entry = await self.mesh_client.read_blackboard(f"goals/{self.agent_id}")
            if entry:
                return entry.get("value", entry)
        except Exception as e:
            logger.debug("Failed to fetch goals for '%s': %s", self.agent_id, e)
        return None

    async def _build_initial_context(self, assignment: TaskAssignment) -> list[dict]:
        """Build initial user message with task, goals, memory, and blackboard context."""
        parts = []

        goals = await self._fetch_goals()
        if goals:
            parts.append(f"## Your Current Goals\n{sanitize_for_prompt(format_dict(goals))}")

        parts.append(f"## Task: {assignment.task_type}\n\n## Input\n{format_dict(assignment.input_data)}")

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
        estimated_tokens = sum(len(json.dumps(m)) // 4 for m in messages)
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
                    summary_parts.append(f"Tool result: {truncate(msg.get('content', ''), 100)}")
                elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                    names = [tc["function"]["name"] for tc in msg["tool_calls"]]
                    summary_parts.append(f"Called: {', '.join(names)}")

        summary_msg = {
            "role": "user",
            "content": "## Previous Actions (summarized)\n" + "\n".join(summary_parts),
        }
        result = first_group + [summary_msg]
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
        # Record structured tool outcome
        await self.memory.store_tool_outcome(
            tool_name=tool_name,
            arguments=tool_input,
            outcome=truncate(str(tool_output), 500),
            success=True,
        )
        if isinstance(tool_output, dict):
            for key, value in tool_output.items():
                if isinstance(value, (str, int, float, bool)):
                    await self.memory.store_fact(
                        key=f"{tool_name}_{key}",
                        value=str(value),
                        category=tool_name,
                        source=f"tool:{tool_name}",
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

    def _maybe_reload_skills(self, result: Any) -> None:
        """If a tool returned reload_requested, hot-reload the skill registry."""
        if isinstance(result, dict) and result.get("reload_requested"):
            count = self.skills.reload()
            logger.info(f"Hot-reloaded skills: {count} available")

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

    def _build_tool_history_context(self, limit: int = 10) -> str:
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
        tools_desc = self.skills.get_descriptions(exclude=self._excluded_tools)
        parts = []

        # Load workspace identity + project files into system prompt
        if self.workspace:
            bootstrap = self.workspace.get_bootstrap_content()
            if bootstrap:
                parts.append(sanitize_for_prompt(bootstrap))

        is_standalone = self.mesh_client.is_standalone
        rules = (
            f"You are the '{self.role}' agent in the OpenLegion multi-agent system.\n"
            f"Your current task: {assignment.task_type}\n\n"
            f"## Available Tools\n\n{tools_desc}\n\n"
            f"## Operating Rules\n"
            f"- Act first — call tools immediately, explain results after.\n"
            f"- Never refuse without trying. Attempt the task, report blockers after.\n"
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
                parts.append(f"## Learnings\n\n{sanitize_for_prompt(learnings)}")
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

    # ── Chat mode ──────────────────────────────────────────────

    CHAT_MAX_TOOL_ROUNDS = 30
    CHAT_MAX_TOTAL_ROUNDS = 200

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
            return await self._chat_inner(user_message)

    # ── Chat helpers (shared by streaming and non-streaming) ────

    async def _prepare_chat_turn(self, user_message: str) -> tuple[str, str]:
        """Set up chat context: corrections, memory, steer, system prompt.

        Returns (possibly-enriched user_message, system_prompt).
        """
        goals = await self._fetch_goals()

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

        if not self._chat_messages and self.workspace:
            memory_hits = self.workspace.search(user_message, max_results=3)
            if memory_hits:
                memory_context = sanitize_for_prompt("\n".join(
                    f"- [{h['file']}] {h['snippet']}" for h in memory_hits
                ))
                user_message = (
                    f"{user_message}\n\n"
                    f"[Relevant memory auto-loaded]\n{memory_context}"
                )

        self._chat_messages.append({"role": "user", "content": user_message})
        steered = self._drain_steer_messages()
        if steered:
            combined = "\n\n".join(steered)
            self._chat_messages[-1]["content"] += f"\n\n[Additional context]: {combined}"

        roster = [] if self.mesh_client.is_standalone else await self._fetch_fleet_roster()
        introspect_data = await self._fetch_introspect_cached()
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

    async def _run_tool(self, tool_call) -> tuple[str, dict]:
        """Execute a single tool call with loop detection, learning, and error handling.

        Returns (result_str, result_dict) for the caller to append to messages.
        Shared by both task mode and chat mode to avoid duplicated logic.
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
            result_str = json.dumps(result, default=str) if isinstance(result, dict) else str(result)
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
                self._maybe_reload_skills(result)
            except Exception as learn_err:
                logger.warning("Post-tool learning failed for %s: %s", tool_call.name, learn_err)
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

    async def _execute_chat_tool_call(
        self, tool_call, tool_call_id: str, tool_outputs: list[dict],
    ) -> dict:
        """Execute a single tool call, append result to chat messages, return output."""
        result_str, result = await self._run_tool(tool_call)
        self._chat_messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_str,
        })
        output = {"tool": tool_call.name, "input": tool_call.arguments, "output": result}
        tool_outputs.append(output)
        return output

    async def _compact_chat_context(self, system: str) -> None:
        """Run context compaction and drain any pending steer messages."""
        if self.context_manager:
            self._chat_messages = await self.context_manager.maybe_compact(
                system, self._chat_messages,
            )
        else:
            self._chat_messages = self._trim_context(self._chat_messages, max_tokens=100_000)

        steered = self._drain_steer_messages()
        if steered:
            combined = "\n\n".join(f"[User interjection]: {s}" for s in steered)
            self._chat_messages.append({"role": "user", "content": combined})

    @staticmethod
    def _resolve_content(llm_response) -> str:
        """Extract text content, suppressing silent acknowledgments."""
        content = llm_response.content or ""
        if content and content.strip() == SILENT_REPLY_TOKEN:
            content = ""
        return content

    # ── Non-streaming chat ────────────────────────────────────

    async def _chat_inner(self, user_message: str) -> dict:
        self.state = "working"
        total_tokens = 0
        tool_outputs: list[dict] = []

        try:
            user_message, system = await self._prepare_chat_turn(user_message)

            if self._chat_total_rounds >= self.CHAT_MAX_TOTAL_ROUNDS:
                self.state = "idle"
                return {
                    "response": (
                        "Chat session has reached its maximum tool round limit "
                        f"({self.CHAT_MAX_TOTAL_ROUNDS}). Please reset the chat "
                        "to continue."
                    ),
                    "tool_outputs": [],
                    "tokens_used": 0,
                }

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
                    return {
                        "response": f"Stopped: {terminate_msg}",
                        "tool_outputs": tool_outputs,
                        "tokens_used": total_tokens,
                    }

                entries = self._build_tool_call_entries(llm_response)
                for i, tool_call in enumerate(llm_response.tool_calls):
                    await self._execute_chat_tool_call(tool_call, entries[i]["id"], tool_outputs)
                self._chat_total_rounds += 1

                if self._chat_total_rounds >= self.CHAT_MAX_TOTAL_ROUNDS:
                    logger.warning("Chat session hit total round limit (%d)", self.CHAT_MAX_TOTAL_ROUNDS)
                    break

                await self._compact_chat_context(system)

            # Max tool rounds exhausted — force final text response.
            llm_response = await _llm_call_with_retry(
                self.llm.chat,
                system=system,
                messages=self._chat_messages,
                tools=self.skills.get_tool_definitions(exclude=self._excluded_tools) or None,
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
            }

        except asyncio.CancelledError:
            self.state = "idle"
            raise
        except Exception as e:
            self.state = "idle"
            logger.error(f"Chat failed: {e}", exc_info=True)
            return {"response": f"Error: {e}", "tool_outputs": tool_outputs, "tokens_used": total_tokens}

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

    async def reset_chat(self) -> None:
        """Clear conversation history. Flushes important facts to memory
        before clearing. Acquires the chat lock to avoid corrupting state
        during an active chat turn."""
        async with self._chat_lock:
            if self._chat_messages and self.context_manager:
                try:
                    await self.context_manager._flush_to_memory(
                        "", self._chat_messages,
                    )
                except Exception as e:
                    logger.warning("Failed to flush memory on chat reset: %s", e)
            self._chat_messages = []
            self._chat_total_rounds = 0
            self._loop_detector.reset()
            if self.context_manager:
                self.context_manager.reset()

    def _build_chat_system_prompt(
        self,
        goals: dict | None = None,
        fleet_roster: list[dict] | None = None,
        introspect_data: dict | None = None,
    ) -> str:
        tools_desc = self.skills.get_descriptions(exclude=self._excluded_tools)
        parts = []

        if goals:
            parts.append(f"## Your Current Goals\n\n{sanitize_for_prompt(format_dict(goals))}")

        if self.workspace:
            bootstrap = self.workspace.get_bootstrap_content()
            if bootstrap:
                parts.append(sanitize_for_prompt(bootstrap))

            learnings = self.workspace.get_learnings_context()
            if learnings:
                parts.append(f"## Learnings from Past Sessions\n\n{sanitize_for_prompt(learnings)}")

        has_browser = "browser_navigate" in self.skills.list_skills(exclude=self._excluded_tools)

        is_standalone = self.mesh_client.is_standalone
        rules = (
            f"You are the '{self.role}' agent in the OpenLegion fleet.\n\n"
            f"## Available Tools\n\n{tools_desc}\n\n"
            f"## Operating Rules\n"
            f"- Act first — call tools immediately. Report results, not intentions.\n"
            f"- Never refuse without trying. Attempt the task, report blockers after.\n"
            f"- Make decisions with reasonable defaults. Ask only when truly ambiguous.\n"
        )
        if is_standalone:
            rules += "- Use notify_user to report results to the user.\n"
        else:
            rules += "- Use notify_user for the user; blackboard for other agents only.\n"
        rules += "- Before answering from memory, run memory_search first.\n"

        if has_browser:
            rules += (
                "\n## Browser\n"
                "browser_navigate → browser_snapshot (get refs) → "
                "browser_click(ref=)/browser_type(ref=). Re-snapshot after changes.\n"
            )

        rules += (
            "\n## Self-Evolution\n"
            "You improve across sessions by updating your workspace files:\n"
            "- **INSTRUCTIONS.md**: add procedures, rules, and domain knowledge "
            "you discover.\n"
            "- **SOUL.md**: refine your identity and communication style "
            "based on feedback.\n"
            "- **USER.md**: record user preferences, corrections, and context "
            "immediately when given.\n"
            "- **HEARTBEAT.md**: tune your autonomous wakeup behavior.\n\n"
            "Read the file first before updating — merge, don't overwrite.\n"
            "When errors repeat, distill the pattern into INSTRUCTIONS.md "
            "and clear resolved entries from errors.md.\n"
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
            async for event in self._chat_stream_inner(user_message):
                yield event

    async def _chat_stream_inner(self, user_message: str):
        self.state = "working"
        total_tokens = 0
        tool_outputs: list[dict] = []

        try:
            user_message, system = await self._prepare_chat_turn(user_message)

            if self._chat_total_rounds >= self.CHAT_MAX_TOTAL_ROUNDS:
                self.state = "idle"
                msg = (
                    "Chat session has reached its maximum tool round limit "
                    f"({self.CHAT_MAX_TOTAL_ROUNDS}). Please reset the chat "
                    "to continue."
                )
                yield {"type": "text_delta", "content": msg}
                yield {"type": "done", "response": msg, "tool_outputs": [], "tokens_used": 0}
                return

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
                    # Emit text_delta for non-streaming fallback only if no tokens
                    # were already streamed (avoids doubled content on partial failure)
                    if not streamed and not any_text_streamed and content:
                        yield {"type": "text_delta", "content": content}
                    self._chat_messages.append({"role": "assistant", "content": content})
                    self._log_chat_turn(user_message, content)
                    self.state = "idle"
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
                    self.state = "idle"
                    yield {
                        "type": "done",
                        "response": f"Stopped: {terminate_msg}",
                        "tool_outputs": tool_outputs,
                        "tokens_used": total_tokens,
                    }
                    return

                entries = self._build_tool_call_entries(llm_response)
                for i, tool_call in enumerate(llm_response.tool_calls):
                    yield {"type": "tool_start", "name": tool_call.name, "input": tool_call.arguments}
                    output = await self._execute_chat_tool_call(tool_call, entries[i]["id"], tool_outputs)
                    yield {"type": "tool_result", "name": tool_call.name, "output": output["output"]}
                self._chat_total_rounds += 1

                if self._chat_total_rounds >= self.CHAT_MAX_TOTAL_ROUNDS:
                    logger.warning("Chat session hit total round limit (%d)", self.CHAT_MAX_TOTAL_ROUNDS)
                    break

                await self._compact_chat_context(system)

            # Max tool rounds exhausted — force final response (non-streaming ok).
            llm_response = await _llm_call_with_retry(
                self.llm.chat,
                system=system,
                messages=self._chat_messages,
                tools=self.skills.get_tool_definitions(exclude=self._excluded_tools) or None,
            )
            total_tokens += llm_response.tokens_used
            content = self._resolve_content(llm_response)
            if content:
                yield {"type": "text_delta", "content": content}
            self._chat_messages.append({"role": "assistant", "content": content})
            self._log_chat_turn(user_message, content)
            self.state = "idle"
            yield {
                "type": "done",
                "response": content,
                "tool_outputs": tool_outputs,
                "tokens_used": total_tokens,
            }

        except asyncio.CancelledError:
            self.state = "idle"
            raise
        except Exception as e:
            self.state = "idle"
            logger.error(f"Streaming chat failed: {e}", exc_info=True)
            yield {
                "type": "done",
                "response": f"Error: {e}",
                "tool_outputs": tool_outputs,
                "tokens_used": total_tokens,
            }
