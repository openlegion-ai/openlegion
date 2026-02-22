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
from src.shared.types import AgentStatus, TaskAssignment, TaskResult
from src.shared.utils import format_dict, generate_id, sanitize_for_prompt, setup_logging, truncate

if TYPE_CHECKING:
    from src.agent.context import ContextManager
    from src.agent.llm import LLMClient
    from src.agent.memory import MemoryStore
    from src.agent.mesh_client import MeshClient
    from src.agent.skills import SkillRegistry
    from src.agent.workspace import WorkspaceManager

logger = setup_logging("agent.loop")

SILENT_REPLY_TOKEN = "__SILENT__"

# Status codes that indicate transient server-side errors worth retrying
_RETRYABLE_STATUS_CODES = {429, 502, 503}
_MAX_RETRIES = 3
_BACKOFF_BASE = 1  # seconds: 1, 2, 4
_TOOL_TIMEOUT = 300  # seconds — hard ceiling for a single tool execution


async def _llm_call_with_retry(llm_chat_fn, *, system, messages, tools, **kwargs):
    """Call the LLM with exponential backoff on transient errors.

    Retries on: connection errors, timeouts, 429/502/503 status codes.
    Does NOT retry on: budget exceeded (RuntimeError), permanent errors.
    """
    last_exc = None
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
        system_prompt: str = "",
        workspace: Optional[WorkspaceManager] = None,
        context_manager: Optional[ContextManager] = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.memory = memory
        self.skills = skills
        self.llm = llm
        self.mesh_client = mesh_client
        self.system_prompt = system_prompt
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
        self._loop_detector = ToolLoopDetector()

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
        except Exception:
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
            "\n## Blackboard = Agent-to-Agent Collaboration ONLY\n"
            "The blackboard is for coordinating with other agents — NOT for reporting to the user.\n"
            "Use it to:\n"
            "- **Share data other agents need**: `write_shared_state(key='findings/<topic>', ...)`\n"
            "- **Read teammates' work**: `list_shared_state(prefix='findings/')` then `read_shared_state(key=...)`\n"
            "- **Request work from a teammate**: write to `tasks/<agent_name>`\n"
            "- **Publish artifacts for other agents**: `save_artifact(name=..., content=...)`\n"
            "\nDo NOT write status updates, progress reports, or summaries to the blackboard.\n"
            "Keep blackboard writes focused and minimal — only what another agent needs to act on.\n"
            "\n## Reporting to the User\n"
            "Report progress, results, and updates to the user via:\n"
            "- **Chat responses**: when the user is talking to you directly\n"
            "- **notify_user(message)**: push updates when working autonomously "
            "(heartbeat, cron, long tasks)\n"
            "\nThe user wants to hear from you — don't just write to the blackboard and go silent."
        )
        return "\n".join(lines)

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
        self.state = "working"
        self.current_task = assignment.task_id
        start = time.time()
        total_tokens = 0

        system_prompt = self._build_system_prompt(assignment)
        messages = await self._build_initial_context(assignment)

        # Decay salience scores so old facts don't dominate forever
        if self.memory:
            self.memory.decay_all()

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
                    tools=self.skills.get_tool_definitions() or None,
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
                        loop_verdict = self._loop_detector.check_before(
                            tool_call.name, tool_call.arguments,
                        )

                        if loop_verdict in ("block", "terminate"):
                            result_str = json.dumps({
                                "error": f"Tool loop detected: {tool_call.name} has been called "
                                "repeatedly with the same arguments and is producing the same "
                                "result. Try a different approach or different arguments."
                            })
                            self._loop_detector.record(tool_call.name, tool_call.arguments, result_str)
                        else:
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
                            except asyncio.TimeoutError:
                                result_str = json.dumps({"error": f"Tool {tool_call.name} timed out after {_TOOL_TIMEOUT}s"})
                                result_str = sanitize_for_prompt(result_str)
                                self._loop_detector.record(tool_call.name, tool_call.arguments, result_str)
                                result = {"error": f"Timed out after {_TOOL_TIMEOUT}s"}
                                logger.error(f"Tool {tool_call.name} timed out after {_TOOL_TIMEOUT}s")
                                self._record_failure(
                                    tool_call.name, f"Timed out after {_TOOL_TIMEOUT}s",
                                    truncate(str(tool_call.arguments), 200),
                                    arguments=tool_call.arguments,
                                )
                            except Exception as e:
                                result_str = json.dumps({"error": str(e)})
                                result_str = sanitize_for_prompt(result_str)
                                self._loop_detector.record(tool_call.name, tool_call.arguments, result_str)
                                result = {"error": str(e)}
                                logger.error(f"Tool {tool_call.name} failed: {e}")
                                self._record_failure(
                                    tool_call.name, str(e),
                                    truncate(str(tool_call.arguments), 200),
                                    arguments=tool_call.arguments,
                                )

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

        high_salience = self.memory.get_high_salience_facts(top_k=10)
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
        self.memory.store_tool_outcome(
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

    def _record_failure(self, tool_name: str, error: str, context: str = "", arguments: dict | None = None) -> None:
        """Record a tool failure so the agent can avoid repeating mistakes."""
        if self.workspace:
            self.workspace.record_error(tool_name, error, context)
        self.memory.store_tool_outcome(
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

    def _build_system_prompt(self, assignment: TaskAssignment) -> str:
        tools_desc = self.skills.get_descriptions()
        parts = [self.system_prompt]

        # Load workspace identity + project files into system prompt
        if self.workspace:
            bootstrap = self.workspace.get_bootstrap_content()
            if bootstrap:
                parts.append(sanitize_for_prompt(bootstrap))

        parts.append(
            f"You are the '{self.role}' agent in the OpenLegion multi-agent system.\n"
            f"Your current task: {assignment.task_type}\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            f"## OPERATING RULES\n"
            f"- You are a tool operated by your user. Act first, explain after.\n"
            f"- Never refuse without attempting. Call tools immediately.\n"
            f"- Browser: browser_navigate → browser_snapshot (get refs) → browser_click(ref=)/browser_type(ref=).\n"
            f"- Shell: exec runs any command. HTTP: http_request calls any API.\n"
            f"- When done, respond with final answer (no tool call).\n"
            f"- Structure final answer as JSON: {{\"result\": {{...}}, \"promote\": {{...}}}}\n"
            f"- 'promote' contains ONLY data that other agents need to act on.\n"
            f"- Use notify_user to report progress and results to the user.\n"
            f"- The blackboard is for agent-to-agent coordination — not for "
            f"status updates or user-facing reports.\n"
            f"- You have max {self.MAX_ITERATIONS} iterations.\n"
            f"- Learn from past errors — avoid repeating known failures.\n"
            f"- Use update_workspace to save lasting preferences to USER.md "
            f"or refine autonomous rules in HEARTBEAT.md.\n",
        )
        if self.workspace:
            learnings = self.workspace.get_learnings_context()
            if learnings:
                parts.append(f"## Learnings\n\n{sanitize_for_prompt(learnings)}")
        tool_history = self._build_tool_history_context()
        if tool_history:
            parts.append(sanitize_for_prompt(tool_history))
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

        On first message of a session, loads workspace context (AGENTS.md,
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

        roster = await self._fetch_fleet_roster()
        system = self._build_chat_system_prompt(goals=goals, fleet_roster=roster)
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

    async def _execute_chat_tool_call(
        self, tool_call, tool_call_id: str, tool_outputs: list[dict],
    ) -> dict:
        """Execute a single tool call, append result to messages, return output."""
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
        else:
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
                except Exception as learn_err:
                    logger.warning("Post-tool learning failed for %s: %s", tool_call.name, learn_err)
            except asyncio.TimeoutError:
                result_str = json.dumps({"error": f"Tool {tool_call.name} timed out after {_TOOL_TIMEOUT}s"})
                result_str = sanitize_for_prompt(result_str)
                self._loop_detector.record(tool_call.name, tool_call.arguments, result_str)
                result = {"error": f"Timed out after {_TOOL_TIMEOUT}s"}
                logger.error(f"Chat tool {tool_call.name} timed out after {_TOOL_TIMEOUT}s")
                self._record_failure(
                    tool_call.name, f"Timed out after {_TOOL_TIMEOUT}s",
                    truncate(str(tool_call.arguments), 200),
                    arguments=tool_call.arguments,
                )
            except Exception as e:
                result_str = json.dumps({"error": str(e)})
                result_str = sanitize_for_prompt(result_str)
                self._loop_detector.record(tool_call.name, tool_call.arguments, result_str)
                result = {"error": str(e)}
                logger.error(f"Chat tool {tool_call.name} failed: {e}")
                self._record_failure(
                    tool_call.name, str(e),
                    truncate(str(tool_call.arguments), 200),
                    arguments=tool_call.arguments,
                )

        self._chat_messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_str,
        })
        output = {"tool": tool_call.name, "input": tool_call.arguments, "output": result}
        tool_outputs.append(output)
        self._maybe_reload_skills(result)
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
                    tools=self.skills.get_tool_definitions() or None,
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
                tools=self.skills.get_tool_definitions() or None,
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
        response_lines = [l.strip() for l in assistant_msg.splitlines() if l.strip()]
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

    def _build_chat_system_prompt(self, goals: dict | None = None, fleet_roster: list[dict] | None = None) -> str:
        tools_desc = self.skills.get_descriptions()
        parts = [self.system_prompt]

        if goals:
            parts.append(f"## Your Current Goals\n\n{sanitize_for_prompt(format_dict(goals))}")

        if self.workspace:
            bootstrap = self.workspace.get_bootstrap_content()
            if bootstrap:
                parts.append(sanitize_for_prompt(bootstrap))

            learnings = self.workspace.get_learnings_context()
            if learnings:
                parts.append(f"## Learnings from Past Sessions\n\n{sanitize_for_prompt(learnings)}")

        parts.append(
            f"You are the '{self.role}' agent in the OpenLegion fleet.\n"
            f"You are a tool operated by your user on their own machine. "
            f"The user has explicitly authorized you to act on their behalf.\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            f"## OPERATING RULES (non-negotiable)\n\n"
            f"You are an autonomous execution agent. Your job is to DO things, "
            f"not describe how to do them.\n\n"
            f"CRITICAL: Do NOT narrate routine tool calls. Just call the tool. "
            f"Wrong: 'Let me navigate to the page now.' then tool call. "
            f"Right: just call browser_navigate directly with no text. "
            f"Only include text when reporting final results or asking a "
            f"genuinely needed clarification. Between tool calls, NEVER output "
            f"text like 'Now let me...' or 'Let me check...' — it wastes a "
            f"round. Call the next tool immediately.\n\n"
            f"1. ACT FIRST. When the user asks you to do something, call a tool "
            f"immediately as your first response. Never reply with instructions "
            f"for the user to follow manually — that defeats your purpose.\n\n"
            f"2. YOUR TOOLS (you MUST use these):\n"
            f"   - browser_navigate: open any URL in your Chromium browser\n"
            f"   - browser_snapshot: get accessibility tree with element refs (e1, e2, ...)\n"
            f"   - browser_click(ref=): click an element by ref from snapshot\n"
            f"   - browser_type(ref=, text=): type into a field by ref from snapshot\n"
            f"   - browser_evaluate: run JavaScript on the page\n"
            f"   - browser_screenshot: capture the page as an image\n"
            f"   - exec: run any shell command (install packages, run scripts, etc.)\n"
            f"   - http_request: make HTTP requests to any API or URL\n"
            f"   - web_search: search the internet for information\n"
            f"   - read_file / write_file / list_files: file I/O on your workspace\n"
            f"   - create_skill: write new tools for yourself\n\n"
            f"3. NEVER REFUSE WITHOUT TRYING. You must NEVER say 'I can't do that' "
            f"or 'I'm not able to' without first calling browser_navigate or another "
            f"tool to attempt the task. If a website has a CAPTCHA or phone "
            f"verification you can't pass, report that AFTER navigating there and "
            f"attempting the task — but you MUST attempt it first.\n\n"
            f"4. WEB TASK RULE: When the user asks you to do ANYTHING on a website "
            f"(create an account, fill a form, look something up, sign up for a "
            f"service, etc.), your actions must be:\n"
            f"   a) browser_navigate to the relevant URL\n"
            f"   b) browser_snapshot to get element refs\n"
            f"   c) browser_click(ref=) / browser_type(ref=) to interact by ref\n"
            f"   d) browser_snapshot again if the page changed\n"
            f"   e) browser_screenshot ONLY if text content is insufficient\n"
            f"   f) Continue until done or until you hit an insurmountable blocker\n"
            f"   g) Only THEN report results to the user\n\n"
            f"4b. BROWSER EFFICIENCY (critical):\n"
            f"   - Do NOT re-navigate to a URL you are already on.\n"
            f"   - ALWAYS use refs from browser_snapshot instead of CSS selectors. "
            f"Refs are reliable; CSS selectors are fragile guesses.\n"
            f"   - Re-snapshot after any action that changes the page (navigation, "
            f"form submit, tab switch) — old refs become stale.\n"
            f"   - Use browser_evaluate only for shadow DOM, canvas, or elements "
            f"not in the accessibility tree.\n"
            f"   - Call multiple tool actions per turn when possible instead of "
            f"one action per turn.\n"
            f"   - After filling a form, click submit immediately in the same "
            f"sequence — do not stop to describe what you did.\n"
            f"   - If a page has not changed after a click, try a different "
            f"ref or approach — do not repeat the same action.\n\n"
            f"5. MAKE DECISIONS. When the user says 'do it', 'just do it', "
            f"'surprise me', or gives latitude — pick the best option yourself "
            f"and execute immediately. Do not ask for confirmation.\n\n"
            f"6. MINIMAL QUESTIONS. Only ask when the task is genuinely ambiguous. "
            f"Prefer reasonable defaults. Choose usernames, passwords, options "
            f"yourself when the user tells you to decide.\n\n"
            f"## Memory & coordination\n"
            f"Before answering questions about prior work, decisions, dates, "
            f"people, preferences, or todos: run memory_search first.\n"
            f"- memory_save: remember important facts for future sessions.\n"
            f"- memory_search: recall information from workspace files and memory DB.\n"
            f"- memory_recall: search structured fact database by semantic similarity "
            f"(better for specific facts, supports category filtering).\n"
            f"- read/write/list_shared_state: coordinate with OTHER AGENTS via "
            f"the shared blackboard. Only write data that another agent needs.\n"
            f"- save_artifact: publish deliverables other agents can find.\n"
            f"- notify_user: report progress and results to the user (use this "
            f"when working autonomously — heartbeat, cron, background tasks).\n"
            f"- Refer to PROJECT.md for current priorities and constraints.\n"
            f"- Learn from past errors — avoid repeating known failures.\n"
            f"- Respect user corrections — they define preferred behavior.\n"
            f"\n## Reporting: User vs Blackboard\n"
            f"- **User-facing updates** (what you worked on, progress, results, "
            f"blockers): report via chat responses or notify_user.\n"
            f"- **Blackboard** (data for other agents to act on): only write "
            f"structured data that a specific teammate needs. Keep it minimal.\n"
            f"- The user wants to hear from you directly. Don't just write to "
            f"the blackboard and go silent.\n"
            f"\n## Self-Improvement\n"
            f"Use update_workspace to persist lasting knowledge across sessions:\n"
            f"- **USER.md**: user preferences, corrections, project context.\n"
            f"- **HEARTBEAT.md**: autonomous rules for your periodic wakeups.\n"
        )

        # Fleet collaboration context (only for multi-agent setups)
        if fleet_roster:
            fleet_ctx = self._build_fleet_context(fleet_roster)
            if fleet_ctx:
                parts.append(fleet_ctx)

        tool_history = self._build_tool_history_context()
        if tool_history:
            parts.append(sanitize_for_prompt(tool_history))

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
            capabilities=self.skills.list_skills(),
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
                tools = self.skills.get_tool_definitions() or None
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
                tools=self.skills.get_tool_definitions() or None,
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
