"""Bounded agent execution loop.

Each task runs: perceive -> decide (LLM) -> act (tool) -> learn.
Max 20 iterations per task. Proper LLM tool-calling message roles.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Optional

from src.shared.types import AgentStatus, TaskAssignment, TaskResult
from src.shared.utils import format_dict, generate_id, setup_logging, truncate

if TYPE_CHECKING:
    from src.agent.context import ContextManager
    from src.agent.llm import LLMClient
    from src.agent.memory import MemoryStore
    from src.agent.mesh_client import MeshClient
    from src.agent.skills import SkillRegistry
    from src.agent.workspace import WorkspaceManager

logger = setup_logging("agent.loop")


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

    async def execute_task(self, assignment: TaskAssignment) -> TaskResult:
        """Main execution method. Runs bounded loop for a single task.

        CRITICAL: Maintains proper LLM conversation history with correct roles.
        Messages grow across iterations:
          user -> assistant(tool_calls) -> tool(result) -> assistant(final)
        """
        self.state = "working"
        self.current_task = assignment.task_id
        start = time.time()
        total_tokens = 0

        system_prompt = self._build_system_prompt(assignment)
        messages = await self._build_initial_context(assignment)

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
                llm_response = await self.llm.chat(
                    system=system_prompt,
                    messages=messages,
                    tools=self.skills.get_tool_definitions() or None,
                )
                total_tokens += llm_response.tokens_used
                if assignment.token_budget:
                    assignment.token_budget.record_usage(llm_response.tokens_used, self.llm.default_model)

                # === ACT ===
                if llm_response.tool_calls:
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
                        try:
                            result = await self.skills.execute(
                                tool_call.name,
                                tool_call.arguments,
                                mesh_client=self.mesh_client,
                                workspace_manager=self.workspace,
                            )
                            result_str = json.dumps(result, default=str) if isinstance(result, dict) else str(result)
                        except Exception as e:
                            result_str = json.dumps({"error": str(e)})
                            result = {"error": str(e)}
                            logger.error(f"Tool {tool_call.name} failed: {e}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_entries[i]["id"],
                            "content": result_str,
                        })

                        await self._learn(tool_call.name, tool_call.arguments, result)

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

                    logger.info(
                        f"Task {assignment.task_id} complete",
                        extra={"extra_data": {"iterations": iteration + 1, "tokens": total_tokens}},
                    )

                    result = TaskResult(
                        task_id=assignment.task_id,
                        status="complete",
                        result=result_data,
                        promote_to_blackboard=promotions,
                        tokens_used=total_tokens,
                        duration_ms=int((time.time() - start) * 1000),
                    )
                    self._last_result = result
                    return result

            # Max iterations reached
            self.state = "idle"
            self.current_task = None
            self.tasks_failed += 1
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
            result = TaskResult(
                task_id=assignment.task_id,
                status="failed",
                error=str(e),
                tokens_used=total_tokens,
                duration_ms=int((time.time() - start) * 1000),
            )
            self._last_result = result
            return result

    async def _build_initial_context(self, assignment: TaskAssignment) -> list[dict]:
        """Build initial user message with task, memory, and blackboard context."""
        parts = []

        parts.append(f"## Task: {assignment.task_type}\n\n## Input\n{format_dict(assignment.input_data)}")

        high_salience = self.memory.get_high_salience_facts(top_k=10)
        if high_salience:
            memory_text = "\n".join(f"- {f.key}: {f.value}" for f in high_salience)
            parts.append(f"## Your Memory (most relevant)\n{memory_text}")

        query = f"{assignment.task_type} {format_dict(assignment.input_data)}"
        relevant = await self.memory.search(query, top_k=10)
        seen_ids = {f.id for f in high_salience}
        novel = [f for f in relevant if f.id not in seen_ids]
        if novel:
            memory_text = "\n".join(f"- {f.key}: {f.value}" for f in novel)
            parts.append(f"## Related Memory\n{memory_text}")

        if assignment.context:
            parts.append(f"## Shared Context from Other Agents\n{format_dict(assignment.context)}")

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
        """Extract and store useful facts from tool execution."""
        await self.memory.log_action(
            action=f"tool:{tool_name}",
            input_summary=truncate(str(tool_input), 200),
            output_summary=truncate(str(tool_output), 200),
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

    def _build_system_prompt(self, assignment: TaskAssignment) -> str:
        tools_desc = self.skills.get_descriptions()
        return (
            f"{self.system_prompt}\n\n"
            f"You are the '{self.role}' agent in the OpenLegion multi-agent system.\n"
            f"Your current task: {assignment.task_type}\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            f"Instructions:\n"
            f"- Use tools to gather information and take actions.\n"
            f"- When done, respond with your final answer (no tool call).\n"
            f"- Structure final answer as JSON: {{\"result\": {{...}}, \"promote\": {{...}}}}\n"
            f"- 'promote' contains data to share with other agents via blackboard.\n"
            f"- You have max {self.MAX_ITERATIONS} iterations.\n"
        )

    def _parse_final_output(self, content: str) -> tuple[dict, dict]:
        """Parse the LLM's final response into result data and blackboard promotions."""
        try:
            parsed = json.loads(content)
            return parsed.get("result", {"raw": content}), parsed.get("promote", {})
        except (json.JSONDecodeError, AttributeError):
            return {"raw": content}, {}

    # ── Chat mode ──────────────────────────────────────────────

    CHAT_MAX_TOOL_ROUNDS = 10

    async def chat(self, user_message: str) -> dict:
        """Handle a single chat turn with persistent conversation history.

        On first message of a session, loads workspace context (AGENTS.md,
        SOUL.md, USER.md, MEMORY.md, daily logs) into the system prompt
        and auto-searches memory for relevant context.

        Returns {"response": str, "tool_outputs": list[dict], "tokens_used": int}.
        """
        if self.state == "working":
            return {"response": "Agent is busy with a task.", "tool_outputs": [], "tokens_used": 0}

        self.state = "working"
        total_tokens = 0
        tool_outputs: list[dict] = []

        try:
            if not hasattr(self, "_chat_messages"):
                self._chat_messages: list[dict] = []

            # On first message: preload relevant memory context
            if not self._chat_messages and self.workspace:
                memory_hits = self.workspace.search(user_message, max_results=3)
                if memory_hits:
                    memory_context = "\n".join(
                        f"- [{h['file']}] {h['snippet']}" for h in memory_hits
                    )
                    user_message = (
                        f"{user_message}\n\n"
                        f"[Relevant memory auto-loaded]\n{memory_context}"
                    )

            self._chat_messages.append({"role": "user", "content": user_message})

            system = self._build_chat_system_prompt()

            for _ in range(self.CHAT_MAX_TOOL_ROUNDS):
                llm_response = await self.llm.chat(
                    system=system,
                    messages=self._chat_messages,
                    tools=self.skills.get_tool_definitions() or None,
                )
                total_tokens += llm_response.tokens_used

                if not llm_response.tool_calls:
                    self._chat_messages.append({"role": "assistant", "content": llm_response.content})
                    self._log_chat_turn(user_message, llm_response.content)
                    self.state = "idle"
                    return {
                        "response": llm_response.content,
                        "tool_outputs": tool_outputs,
                        "tokens_used": total_tokens,
                    }

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
                self._chat_messages.append({
                    "role": "assistant",
                    "content": llm_response.content or "",
                    "tool_calls": tool_call_entries,
                })

                for i, tool_call in enumerate(llm_response.tool_calls):
                    try:
                        result = await self.skills.execute(
                            tool_call.name,
                            tool_call.arguments,
                            mesh_client=self.mesh_client,
                            workspace_manager=self.workspace,
                        )
                        result_str = json.dumps(result, default=str) if isinstance(result, dict) else str(result)
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                        result = {"error": str(e)}
                        logger.error(f"Chat tool {tool_call.name} failed: {e}")

                    self._chat_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_entries[i]["id"],
                        "content": result_str,
                    })
                    tool_outputs.append({
                        "tool": tool_call.name,
                        "input": tool_call.arguments,
                        "output": result,
                    })

                if self.context_manager:
                    self._chat_messages = await self.context_manager.maybe_compact(
                        system, self._chat_messages,
                    )
                else:
                    self._chat_messages = self._trim_context(self._chat_messages, max_tokens=100_000)

            # Exhausted tool rounds -- ask LLM for final answer without tools
            llm_response = await self.llm.chat(
                system=system,
                messages=self._chat_messages,
                tools=None,
            )
            total_tokens += llm_response.tokens_used
            self._chat_messages.append({"role": "assistant", "content": llm_response.content})
            self._log_chat_turn(user_message, llm_response.content)
            self.state = "idle"
            return {
                "response": llm_response.content,
                "tool_outputs": tool_outputs,
                "tokens_used": total_tokens,
            }

        except Exception as e:
            self.state = "idle"
            logger.error(f"Chat failed: {e}", exc_info=True)
            return {"response": f"Error: {e}", "tool_outputs": tool_outputs, "tokens_used": total_tokens}

    def _log_chat_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Append a summary of the chat turn to the daily log."""
        if not self.workspace:
            return
        user_summary = truncate(user_msg.split("\n")[0], 80)
        assistant_summary = truncate(assistant_msg.split("\n")[0], 80)
        self.workspace.append_daily_log(f"User: {user_summary} → Agent: {assistant_summary}")

    def reset_chat(self) -> None:
        """Clear conversation history."""
        self._chat_messages = []

    def _build_chat_system_prompt(self) -> str:
        tools_desc = self.skills.get_descriptions()
        parts = [self.system_prompt]

        if self.workspace:
            workspace_context = self.workspace.load_prompt_context()
            if workspace_context:
                parts.append(workspace_context)

            memory_content = self.workspace.load_memory()
            if memory_content.strip():
                parts.append(f"## Your Long-Term Memory\n\n{truncate(memory_content, 4000)}")

            daily_logs = self.workspace.load_daily_logs(days=2)
            if daily_logs.strip():
                parts.append(f"## Recent Session Logs\n\n{truncate(daily_logs, 2000)}")

        parts.append(
            f"You are the '{self.role}' agent in the OpenLegion multi-agent system.\n"
            f"You are in interactive chat mode.\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            f"Instructions:\n"
            f"- Use tools to take actions when needed.\n"
            f"- Respond conversationally when no tools are needed.\n"
            f"- Use memory_save to remember important facts for future sessions.\n"
            f"- Use memory_search to recall information from past sessions.\n"
        )

        return "\n\n".join(parts)

    def get_status(self) -> AgentStatus:
        """Return current agent status."""
        return AgentStatus(
            agent_id=self.agent_id,
            role=self.role,
            state=self.state,
            current_task=self.current_task,
            capabilities=self.skills.list_skills(),
            uptime_seconds=time.time() - self._start_time,
            tasks_completed=self.tasks_completed,
            tasks_failed=self.tasks_failed,
        )
