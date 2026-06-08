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
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import httpx

from src.agent.attachments import enrich_message_with_attachments
from src.agent.loop_detector import ToolLoopDetector
from src.agent.tool_groups import (
    GroupedPlan,
    grouped_tools_enabled,
    plan_grouped_tools,
    resolve_load_request,
)
from src.agent.workspace import INTROSPECT_PERM_KEYS
from src.shared import limits
from src.shared.errors import LLMAuthError, LLMConfigError
from src.shared.types import SILENT_REPLY_TOKEN, AgentStatus, LLMResponse, TaskAssignment, TaskResult
from src.shared.utils import dumps_safe, format_dict, generate_id, sanitize_for_prompt, setup_logging, truncate

if TYPE_CHECKING:
    from src.agent.context import ContextManager
    from src.agent.llm import LLMClient
    from src.agent.memory import MemoryStore
    from src.agent.mesh_client import MeshClient
    from src.agent.tools import ToolRegistry
    from src.agent.workspace import WorkspaceManager
    from src.shared.types import MessageOrigin

logger = setup_logging("agent.loop")


# Status codes that indicate transient server-side errors worth retrying
_RETRYABLE_STATUS_CODES = {429, 502, 503, 504, 529}  # 529 = Anthropic overloaded
_MAX_RETRIES = 3
_BACKOFF_BASE = 1  # seconds: 1, 2, 4
_TOOL_TIMEOUT = int(os.environ.get("OPENLEGION_TOOL_TIMEOUT", "900"))  # seconds — hard ceiling per tool
_FLEET_ROSTER_TTL = 600  # seconds — cache TTL for fleet roster
_GOALS_TTL = 300  # seconds — cache TTL for goals fetch
_FALLBACK_MAX_TOKENS = 100_000  # context trim fallback when no context manager
_TOOL_HISTORY_LIMIT = 10  # recent tool outcomes in system prompt
HEARTBEAT_MAX_ITERATIONS = 12  # tighter bound for heartbeat (cheaper than task/chat)

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


def _last_message_is_user_origin(messages: list[dict]) -> bool:
    """Check whether the most recent user message was genuinely user-originated.

    Returns True when the last ``role=user`` message in *messages* does not
    carry an ``_origin`` metadata key **or** when ``_origin == "user"``.
    Returns False when it is tagged with a non-user origin (e.g. ``"system"``,
    ``"auto_continue"``, ``"heartbeat"``).

    If there are no user messages at all, returns False as a safe default.
    """
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            origin = msg.get("_origin", "user")
            return origin == "user"
    return False


# Files already injected via bootstrap — skip in first-message auto-search
# to avoid duplicate content.  Matches WorkspaceManager._BOOTSTRAP_FILES.
_BOOTSTRAP_SEARCH_EXCLUDE = frozenset({
    "TEAM.md", "PROJECT.md", "SYSTEM.md", "INSTRUCTIONS.md",
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

# Runtime-toggleable capability gates. Each gate maps to the set of tools
# hidden from the LLM surface when that capability is switched OFF (via the
# Operator Settings toggles → agent ``/config`` push, or the boot-time env
# seed). Folding both gates through a single ``_disabled_gates`` set lets the
# internet and browser toggles coexist without one clobbering the other.
_RUNTIME_GATE_TOOLS: dict[str, frozenset[str]] = {
    "internet": frozenset({"http_request", "web_search"}),
    # Mirrors the @tool names in ``builtins/browser_tool.py``.
    "browser": frozenset({
        "browser_navigate", "browser_warmup", "browser_get_elements",
        "browser_wait_for", "browser_screenshot", "browser_click",
        "browser_click_xy", "browser_type", "browser_hover", "browser_scroll",
        "browser_reset", "browser_press_key", "browser_go_back",
        "browser_go_forward", "browser_switch_tab", "browser_find_text",
        "browser_fill_form", "browser_open_tab", "browser_inspect_requests",
        "browser_detect_captcha", "browser_upload_file",
        "browser_solve_captcha", "browser_download",
    }),
}

# Agent-authored-Python tool surface — hidden from workers unless the advanced
# OPENLEGION_ENABLE_TOOL_AUTHORING opt-in is set (Task 6: Skills are the default
# extension path now). reload_tools is paired with create_tool and vestigial
# without it (marketplace tools load at container start, not via runtime reload).
_TOOL_AUTHORING_TOOLS = frozenset({"create_tool", "reload_tools"})

# Read-only tools allowed during operator heartbeat (unsupervised execution).
# The full operator allowlist is restricted to this subset so heartbeats
# cannot mutate fleet state without user approval. ``check_inbox`` is
# read-only — needed so any agent can pick up back-edge events during
# its heartbeat tick instead of waiting until the next /chat turn.
# ``workflow_snapshot`` and ``await_task_event`` are operator-tier reads
# (the tools self-reject for non-operator callers via ``_is_operator``);
# adding them here lets operator's heartbeat surface workflow state
# without dropping out to a full /chat turn.
_HEARTBEAT_TOOLS = frozenset({
    "list_agents", "get_agent_profile", "get_system_status",
    "notify_user", "check_inbox",
    "workflow_snapshot", "await_task_event",
    # PR 2 of Work tab rewrite. Heartbeat instructions tell operator
    # to rate up to 10 oldest unrated done tasks via ``rate_delivery``
    # and update tracked goals via ``manage_goals`` based on user
    # chat direction; both surfaces would be unreachable from heartbeat
    # ticks if the loop restricts them to the legacy read-only set.
    "rate_delivery", "manage_goals",
    # ``inspect_agents`` is read-only (operator-tier read with
    # ``_is_operator`` self-gate). Heartbeat step 5 explicitly calls
    # ``inspect_agents()`` for the roster summary and
    # ``inspect_agents(agent_id, depth="profile")`` / ``stale_threshold_hours=24``
    # for drill-ins; without it the prompted procedure is denied by
    # this allowlist (caught by Codex pre-merge review of PR 972).
    "inspect_agents",
})

# Tool calls whose ONLY purpose is to read state — they don't produce
# downstream work, no peer is woken, no artifact is written. The lazy-
# completion guards in chat() / execute_task use this set to detect
# ghost-completion ("checked my inbox, said 'Done!', didn't actually
# hand off or write anything"). A turn whose ONLY tool calls are in
# this set is treated as zero-tool for guard purposes — the LLM must
# either return a structured ``{"result": {...}}`` final or call at
# least one non-read-only tool to satisfy the contract.
#
# New tools are treated as OUTBOUND by default — soft-failing open is
# better than soft-passing closed. A future read-only tool that's not
# yet enumerated here is lenient (the guard might let some lazy
# completions through), not harsh (real work blocked). Enrol new
# read-only tools here as they're added.
_READ_ONLY_TOOLS = frozenset({
    # Coordination reads
    "check_inbox", "list_task_inbox",
    # Blackboard reads. ``subscribe_event`` and ``watch_blackboard``
    # are deliberately NOT here — they create persistent subscription
    # records (state change) which can be the entire deliverable of
    # a "set up monitoring for X" worker task.
    "read_blackboard", "list_blackboard",
    # Fleet / agent introspection
    "list_agents", "get_agent_profile", "get_system_status",
    "inspect_agents", "inspect_teams", "list_agent_queue",
    "list_templates", "list_available_models", "read_agent_config",
    # Operator workflow diagnostics
    "workflow_snapshot", "await_task_event",
    "summarize_team_progress",
    # Operator observation-log read (agent→user notifications). Purely
    # observational — classify read-only so a turn that only reads
    # notifications isn't miscounted as outbound work by the guard.
    "read_user_notifications",
    # Cron + history reads (codex round-5 audit) — writes go through
    # ``manage_cron`` / append-only paths, not these.
    "list_cron", "read_agent_history",
    # Pending-action / artifact reads (writes go through manage_task etc.)
    "list_pending", "list_peer_artifacts", "read_peer_artifact",
    # Peer file reads (full /data volume, not just artifacts/) — read-only.
    "list_peer_files", "read_peer_file",
    "get_team_outputs",
    # Memory / file reads (writes go through memory_save / write_file)
    "memory_search", "memory_think", "read_file",
    # Credential discovery — names only, no values
    "vault_list",
    # Self-introspection
    "introspect",
    # Skill-pack discovery/read (SKILL.md packs). Pure reads — consulting
    # a procedure is not itself outbound work.
    "skills_list", "skill_view",
    # NOTE: ``web_search`` is intentionally NOT here — it makes an
    # external network call which counts as outbound work. Same logic
    # applies to any external HTTP / image-gen / browser tool: a worker
    # that legitimately fetched data from the outside DID work.
})


async def _llm_call_with_retry(llm_chat_fn, *, system, messages, tools, **kwargs):
    """Call the LLM with exponential backoff on transient errors.

    Retries on: connection errors, timeouts, 429/502/503 status codes.
    Does NOT retry on: budget exceeded (RuntimeError), permanent errors.
    """
    from src.agent.llm import LLMRetryableError

    last_exc: Exception = RuntimeError("LLM call failed after all retries")
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await llm_chat_fn(system=system, messages=messages, tools=tools, **kwargs)
        except LLMRetryableError as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                wait = _BACKOFF_BASE * (2 ** attempt) * 5  # longer backoff for rate limits: 5, 10, 20
                logger.warning(
                    f"LLM call rate-limited, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                )
                await asyncio.sleep(wait)
                continue
            raise
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

    # Class-level default so ``_tool_filter_kw`` is safe even when ``__init__``
    # is bypassed (e.g. ``AgentLoop.__new__`` in tests). None = grouped tool
    # search inactive (the default); the instance value is set in ``__init__``.
    _grouped_plan: "GroupedPlan | None" = None

    def __init__(
        self,
        agent_id: str,
        role: str,
        memory: MemoryStore,
        tools: ToolRegistry,
        llm: LLMClient,
        mesh_client: MeshClient,
        workspace: WorkspaceManager | None = None,
        context_manager: ContextManager | None = None,
        allowed_tools: frozenset[str] | None = None,
    ):
        # Round caps resolve through the central limits table (env -> default,
        # both clamped to the spec range). Defaults are HIGH and operator-
        # adjustable; the host injects per-agent / settings.json overrides into
        # the container env at creation. See src/shared/limits.py.
        self.MAX_ITERATIONS = limits.resolve("max_iterations")
        self.CHAT_MAX_TOOL_ROUNDS = limits.resolve("chat_max_tool_rounds")
        self.CHAT_MAX_TOTAL_ROUNDS = limits.resolve("chat_max_total_rounds")
        # Per-task convergence budget (RC-1). A bound on tool rounds that
        # applies ONLY when a durable ``task_id`` is driving the chat turn
        # (handoff / lane dispatch). It bounds the task path below the
        # interactive ceiling (CHAT_MAX_TOOL_ROUNDS × session auto-continues)
        # AND below the mesh-side lane wall-clock cap. This is an ADVISORY
        # convergence/UX bound, NOT a security control: it does NOT replace or
        # weaken any mesh-side control (per-agent daily budget preflight, lane
        # wall-clock cap). It only ADDS a soft bound below them. Interactive
        # chat (no task_id) is completely unaffected.
        self.TASK_MAX_TOOL_ROUNDS = limits.resolve("task_max_tool_rounds")
        # Item 3 (Codex r4): TASK_MAX and CHAT_MAX are independently
        # env-clamped, so a misconfig (TASK_MAX > CHAT_MAX) could let a task
        # exhaust the interactive CHAT_MAX_TOOL_ROUNDS bound — falling through
        # to the ``tool_limit_reached`` exit — BEFORE its per-task cap ever
        # fired. The whole design assumes TASK_MAX <= CHAT_MAX (the top-of-round
        # cap break always wins). Enforce that invariant here so the effective
        # per-task budget can never exceed the interactive ceiling.
        if self.TASK_MAX_TOOL_ROUNDS > self.CHAT_MAX_TOOL_ROUNDS:
            logger.info(
                "TASK_MAX_TOOL_ROUNDS=%d exceeds CHAT_MAX_TOOL_ROUNDS=%d — "
                "clamping the effective per-task budget to %d",
                self.TASK_MAX_TOOL_ROUNDS, self.CHAT_MAX_TOOL_ROUNDS,
                self.CHAT_MAX_TOOL_ROUNDS,
            )
            self.TASK_MAX_TOOL_ROUNDS = self.CHAT_MAX_TOOL_ROUNDS
        self.agent_id = agent_id
        self.role = role
        self.memory = memory
        self.tools = tools
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
        # RC-3: per-task convergence round counts, keyed by ``task_id``.
        # PERSISTS across wakes for the same task so repeated lane followups
        # can't each get a fresh per-task budget. Advisory convergence state
        # only — cleared on the task's terminal close. Bounded by the number
        # of distinct in-flight tasks (small); the entry for a task is wiped
        # in ``chat()`` when it reaches a terminal status.
        self._task_round_counts: dict[str, int] = {}
        self._steer_queue: asyncio.Queue[str] = asyncio.Queue()
        self._fleet_roster: list[dict] | None = None  # cached fleet info
        self._fleet_roster_ts: float = 0  # timestamp of last fetch
        self._introspect_cache: dict | None = None
        self._introspect_cache_ts: float = 0
        self._goals_cache: dict | None | object = AgentLoop._GOALS_NOT_FETCHED
        self._goals_cache_ts: float = 0
        self._loop_detector = ToolLoopDetector(
            exempt_tools=tools.get_loop_exempt_tools(),
        )
        # When an explicit allowlist is provided (e.g. operator agent),
        # only those tools are exposed and exclude is ignored.
        if allowed_tools:
            self._allowed_tools: frozenset[str] | None = allowed_tools
            self._excluded_tools: frozenset[str] | None = None
        else:
            self._allowed_tools = None
            # Standalone agents have no project blackboard — hide those tools.
            excluded: set[str] = (
                set(_BLACKBOARD_TOOLS) if mesh_client.is_standalone else set()
            )
            # Demote agent-authored Python: hide create_tool/reload_tools unless
            # the advanced authoring opt-in is set. Skills are the default path.
            from src.agent.builtins.tool_authoring import tool_authoring_enabled
            if not tool_authoring_enabled():
                excluded |= _TOOL_AUTHORING_TOOLS
            # Operator-only orchestration tools (edit_agent, create_team,
            # manage_*, apply_template, install_skill, …) self-reject for
            # worker callers at call time. Drop their schemas from the worker
            # surface entirely so they never bloat a worker's LLM context or
            # tempt it to call a tool it can never use. The operator gets them
            # via its explicit allowlist, so this exclude never touches it.
            # ``.update()`` (not ``|=``) so a non-set return can only no-op,
            # never silently rebind ``excluded`` via the RHS's ``__ror__``.
            excluded.update(tools.operator_only_tools())
            self._excluded_tools: frozenset[str] | None = (
                frozenset(excluded) if excluded else None
            )
        # Runtime-disabled tools — flips on top of the static allowlist
        # without requiring a restart. The mesh pushes these via the
        # agent's ``/config`` endpoint when a permission changes (e.g.
        # the Operator Settings → Internet access toggle removes
        # ``http_request`` / ``web_search`` from the operator's
        # effective surface). Empty by default; populated only when
        # mesh explicitly tells the agent to hide tools.
        #
        # Boot-time seeding: ``OL_INTERNET_ACCESS_ENABLED=false`` /
        # ``OL_BROWSER_ACCESS_ENABLED=false`` env vars set the runtime
        # filter immediately so a restart while a toggle is OFF doesn't
        # briefly re-expose the tools. Mesh passes these env vars when
        # launching the operator container based on
        # ``operator.can_use_internet`` / ``can_use_browser`` in
        # permissions.json. Both gates are tracked independently so one
        # toggle never clobbers the other.
        self._disabled_gates: set[str] = set()
        if os.environ.get("OL_INTERNET_ACCESS_ENABLED", "true").lower() == "false":
            self._disabled_gates.add("internet")
        if os.environ.get("OL_BROWSER_ACCESS_ENABLED", "true").lower() == "false":
            self._disabled_gates.add("browser")
        self._runtime_disabled_tools: frozenset[str] = frozenset()
        self._recompute_runtime_disabled()
        # ── Grouped Tool Search (B2, default-OFF via OPENLEGION_GROUPED_TOOLS) ──
        # ``_loaded_tool_groups`` are groups whose full schemas are present in
        # context. ``_pending_tool_groups`` are groups requested via
        # ``load_tools`` this turn — promoted into ``_loaded_tool_groups`` at the
        # NEXT system-prompt build (turn boundary) so the toolset never mutates
        # mid-conversation (which would bust the prompt cache). The currently
        # planned defer set is cached so ``_tool_filter_kw`` and the system
        # prompt stay consistent within a single turn.
        self._loaded_tool_groups: set[str] = set()
        self._pending_tool_groups: set[str] = set()
        self._grouped_plan: GroupedPlan | None = None
        self._tools_reloaded: bool = False
        self._is_operator: bool = allowed_tools is not None
        self._operator_playbook_state: dict[str, int] = {}  # playbook -> turns since trigger
        self._last_active_playbooks: list[str] = []
        self._operator_playbook_scan_idx: int = 0
        # Reference to the active messages list — set during tool execution
        # so tools.execute() can inject it into provenance-gated tools.
        self._current_messages: list[dict] = []
        # Liveness signal (Bug 1): single-task asyncio model means no lock
        # needed — the loop only ticks one iteration at a time. The
        # mesh health monitor reads these via /status to detect a dead
        # inner loop with a live FastAPI thread.
        self._last_iteration_ts: float | None = None
        self._iterations_since_boot: int = 0

    def _bump_liveness(self) -> None:
        """Stamp the last-iteration timestamp + bump the boot-relative counter.

        Called at the head of every iteration in execute_task and per-round
        in the chat loops. Cheap on purpose — no I/O, no lock.
        """
        self._last_iteration_ts = time.time()
        self._iterations_since_boot += 1

    @property
    def _tool_filter_kw(self) -> dict:
        """Build kwargs dict for ToolRegistry filter methods.

        Returns ``{"exclude": ..., "allowed": ...}`` only including keys whose
        values are not None, so callers that don't yet accept ``allowed`` (e.g.
        mocks in older tests) keep working.

        ``_runtime_disabled_tools`` is folded into both branches:
          * If an ``_allowed_tools`` allowlist exists (operator path),
            the runtime-disabled set is subtracted from it.
          * If an ``_excluded_tools`` exclude-set exists (worker path),
            the runtime-disabled set is unioned into it.
          * Otherwise the runtime-disabled set is passed as ``exclude``
            on its own so the filter still hides the tools.
        """
        kw: dict = {}
        runtime_disabled = self._runtime_disabled_tools
        if self._excluded_tools is not None:
            kw["exclude"] = (
                self._excluded_tools | runtime_disabled
                if runtime_disabled
                else self._excluded_tools
            )
        elif runtime_disabled:
            kw["exclude"] = runtime_disabled
        if self._allowed_tools is not None:
            kw["allowed"] = (
                self._allowed_tools - runtime_disabled
                if runtime_disabled
                else self._allowed_tools
            )
        # Grouped Tool Search (B2): omit deferred tool schemas for this turn.
        # The plan is recomputed at each system-prompt build (turn boundary) so
        # the defer set here stays consistent with the capability index that was
        # injected into the same turn's system prompt. ``defer`` folds into the
        # ``get_tool_definitions`` memo cache key, so a different loaded-groups
        # set yields different definitions (and a fresh cache entry).
        plan = self._grouped_plan
        if plan is not None and plan.active and plan.defer:
            kw["defer"] = plan.defer
        return kw

    # ── Grouped Tool Search (B2) ───────────────────────────────────────────
    def _refresh_grouped_plan(self) -> str:
        """Promote pending loads, recompute the grouped-tools plan, return index.

        Called at each system-prompt build — the TURN BOUNDARY. This is where a
        ``load_tools`` request made during the previous turn actually takes
        effect (pending → loaded), so the toolset never changes mid-turn (which
        would bust the prompt cache; mirrors hermes' "don't change toolsets
        mid-conversation" invariant).

        Returns the capability-index text to append to the system prompt (""
        when the feature is off or the budget gate didn't trip).
        """
        if not grouped_tools_enabled():
            self._grouped_plan = None
            return ""
        # Apply deferred loads requested last turn.
        if self._pending_tool_groups:
            self._loaded_tool_groups |= self._pending_tool_groups
            self._pending_tool_groups.clear()
        # Available = what this agent can actually call this turn, BEFORE the
        # grouped defer (so the index reflects every callable capability).
        base_kw = {
            k: v for k, v in self._tool_filter_kw.items() if k != "defer"
        }
        available = set(self.tools.list_tools(**base_kw))
        context_window = (
            self.context_manager.max_tokens if self.context_manager else 0
        )
        self._grouped_plan = plan_grouped_tools(
            available=available,
            loaded_groups=self._loaded_tool_groups,
            operator=(self._allowed_tools is not None),
            context_window=context_window,
        )
        return self._grouped_plan.index_text if self._grouped_plan.active else ""

    def request_load_tools(self, *, group: str | None, tool: str | None) -> dict:
        """Bridge for the ``load_tools`` builtin — queue a group for next turn.

        Does NOT mutate the loaded set immediately: the actual schema change is
        deferred to the next system-prompt build (``_refresh_grouped_plan``) so
        the toolset stays stable for the remainder of the current turn.
        """
        if not grouped_tools_enabled():
            return {
                "loaded": [],
                "note": "Grouped tool search is disabled; all tools already loaded.",
            }
        base_kw = {
            k: v for k, v in self._tool_filter_kw.items() if k != "defer"
        }
        available = set(self.tools.list_tools(**base_kw))
        keys, error = resolve_load_request(
            group=group, tool=tool, available=available,
        )
        if error:
            return {"loaded": [], "error": error}
        self._pending_tool_groups |= keys
        return {
            "loaded": sorted(keys),
            "note": (
                "Full schemas for these group(s) will be available on your "
                "NEXT turn. Call the tool then."
            ),
        }

    def _recompute_runtime_disabled(self) -> None:
        """Rebuild ``_runtime_disabled_tools`` from the active gate set."""
        tools: set[str] = set()
        for gate in self._disabled_gates:
            tools |= _RUNTIME_GATE_TOOLS.get(gate, frozenset())
        self._runtime_disabled_tools = frozenset(tools)

    def set_runtime_gate(self, gate: str, enabled: bool) -> None:
        """Flip a single capability gate (``internet`` / ``browser``).

        Called by the agent's ``/config`` endpoint when the mesh pushes a
        permission change. ``enabled=True`` clears the gate (tools become
        visible again); ``enabled=False`` hides that gate's tools. Other
        gates are untouched, so the internet and browser toggles compose.
        """
        if enabled:
            self._disabled_gates.discard(gate)
        else:
            self._disabled_gates.add(gate)
        self._recompute_runtime_disabled()

    def set_runtime_disabled_tools(self, tools: list[str] | set[str]) -> None:
        """Replace the runtime-disabled-tool set directly.

        Retained for callers/tests that set the raw tool set. Prefer
        ``set_runtime_gate`` for the internet/browser capability toggles.
        """
        self._runtime_disabled_tools = frozenset(tools or ())

    def _update_operator_playbooks(self) -> list[str]:
        """Update operator playbook state based on recent tool calls.

        Returns list of currently active playbook keys.
        """
        if not self._is_operator:
            return []

        from src.shared.operator_playbooks import PLAYBOOK_STICKY_TURNS, extract_triggered_playbooks

        # Scan only messages added since last check.
        # After context compaction, the message list may shrink — clamp the
        # start index so we re-scan from the new beginning rather than
        # skipping everything.
        scan_start = min(self._operator_playbook_scan_idx, len(self._chat_messages))
        new_messages = self._chat_messages[scan_start:]
        self._operator_playbook_scan_idx = len(self._chat_messages)

        # Reset counter only for playbooks triggered in NEW messages
        triggered = extract_triggered_playbooks(new_messages)
        for pb in triggered:
            self._operator_playbook_state[pb] = 0

        # Return active playbooks (within sticky window)
        active = [
            pb for pb, turns in sorted(self._operator_playbook_state.items(), key=lambda x: (x[1], x[0]))
            if turns <= PLAYBOOK_STICKY_TURNS
        ]

        if active != self._last_active_playbooks:
            logger.debug("Operator playbooks: %s", active)

        return active

    def _age_operator_playbooks(self) -> None:
        """Increment turn counter for all active playbooks. Call once per user turn."""
        if not self._is_operator:
            return
        from src.shared.operator_playbooks import PLAYBOOK_STICKY_TURNS

        expired = [pb for pb, turns in self._operator_playbook_state.items() if turns > PLAYBOOK_STICKY_TURNS]
        for pb in expired:
            del self._operator_playbook_state[pb]
        for pb in self._operator_playbook_state:
            self._operator_playbook_state[pb] += 1

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
            # Sync project assignment from mesh host (supports runtime add/remove)
            project = data.get("project")
            if project != self.mesh_client.project_name:
                logger.info("Project assignment updated: %s → %s", self.mesh_client.project_name, project)
                self.mesh_client.project_name = project
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
        from src.shared.trace import current_task_id, current_trace_id
        current_trace_id.set(trace_id)
        # ``execute_task`` runs as a fresh coroutine per task — contextvar
        # scope dies with the coroutine, so no explicit reset is needed
        # (mirrors the ``current_trace_id`` pattern on the line above).
        current_task_id.set(assignment.task_id)
        self._loop_detector.reset()
        # State is already set to "working" by receive_task() before spawning
        # this coroutine. Setting current_task is a no-op but documents intent.
        self.current_task = assignment.task_id
        start = time.time()

        # ── Check for existing checkpoint matching this task ──
        checkpoint = None
        if self.memory:
            try:
                checkpoint = await self.memory._run_db(self.memory.load_task_checkpoint)
            except Exception as e:
                logger.warning("Failed to load task checkpoint: %s", e)

        if checkpoint and checkpoint["task_id"] == assignment.task_id:
            # Resume from checkpoint
            messages = checkpoint["messages"]
            start_iteration = checkpoint["iteration"] + 1
            total_tokens = checkpoint["tokens_used"]
            assignment_json = checkpoint["assignment_json"]

            # Bug 5 (stale checkpoint rejection): if the persisted iteration
            # is already at or past MAX_ITERATIONS, resuming would skip the
            # for-loop entirely and immediately fall through to the
            # "Max iterations reached" branch — looking like a 0-token,
            # 0-iteration completion from outside. Detect that here and
            # start fresh, clearing the corrupt checkpoint.
            if start_iteration >= self.MAX_ITERATIONS:
                logger.warning(
                    "Discarding stale checkpoint for task=%s (iteration %d "
                    ">= MAX %d) — starting fresh",
                    assignment.task_id, start_iteration, self.MAX_ITERATIONS,
                )
                if self.memory:
                    try:
                        await self.memory._run_db(self.memory.clear_task_checkpoint)
                    except Exception as clr_err:
                        logger.warning(
                            "Failed to clear stale checkpoint: %s", clr_err,
                        )
                checkpoint = None
                total_tokens = 0
                start_iteration = 0
                assignment_json = assignment.model_dump_json()
                messages = await self._build_initial_context(assignment)
                if self.memory:
                    await self.memory.decay_all()
            else:
                # Reconcile TokenBudget mutable state
                if assignment.token_budget:
                    assignment.token_budget.used_tokens = checkpoint["budget_used_tokens"]
                    assignment.token_budget.estimated_cost_usd = checkpoint["budget_estimated_cost"]

                # Restore context manager flush state
                if self.context_manager:
                    self.context_manager._flush_triggered = checkpoint["flush_triggered"]

                # Inject continuation prompt — but only if last message isn't already
                # a user message (avoids violating role alternation invariant).
                if messages and messages[-1].get("role") != "user":
                    messages.append({
                        "role": "user",
                        "content": (
                            "You were working on this task and your session was interrupted. "
                            "Your conversation history above reflects your progress. "
                            "Continue from where you left off. Do not repeat completed work."
                        ),
                    })

                logger.info(
                    "Resumed task %s from iteration %d (%d tokens used)",
                    assignment.task_id, start_iteration, total_tokens,
                )
        elif checkpoint:
            # Stale checkpoint from a different task — clear it
            logger.warning(
                "Clearing stale task checkpoint (expected %s, found %s)",
                assignment.task_id, checkpoint["task_id"],
            )
            if self.memory:
                await self.memory._run_db(self.memory.clear_task_checkpoint)
            checkpoint = None

        if not checkpoint:
            # Fresh start
            total_tokens = 0
            start_iteration = 0
            assignment_json = assignment.model_dump_json()
            messages = await self._build_initial_context(assignment)
            # Decay salience scores only on fresh start (not resume, to avoid double-decay)
            if self.memory:
                await self.memory.decay_all()

        introspect_data = await self._fetch_introspect_cached()
        system_prompt = self._build_system_prompt(assignment, introspect_data=introspect_data)

        # Bug F (codex r4): seed the tool-call counter from messages so a
        # checkpoint-resumed task picks up where it left off; thereafter
        # we increment locally at the dispatch site so the count survives
        # ``maybe_compact`` dropping the assistant-with-tool_calls entry
        # from the live ``messages`` list. The guard at the terminal
        # branch reads this counter, not the message history.
        tool_calls_count = sum(
            len(m.get("tool_calls") or [])
            for m in messages
            if m.get("role") == "assistant"
        )
        # Round-5 strengthening: count only OUTBOUND-effect tool calls
        # so a task that ran ``check_inbox`` + ``read_blackboard`` and
        # then claimed done falls back to the lazy-completion guard.
        # Seeded from message history at task resume in case compaction
        # dropped explicit tool_call records.
        outbound_tool_calls_count = sum(
            1
            for m in messages
            if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
            if (tc.get("function", {}).get("name") or "")
                not in _READ_ONLY_TOOLS
        )
        # Codex r5: if we resumed past iteration 0 from a checkpoint and
        # the seed read 0, the prior tool-call history was almost
        # certainly compacted away — ``context.maybe_compact`` keeps a
        # summary tail and is not tool-call-group-aware. Treat that as
        # "at least one prior call" so the lazy-completion guard doesn't
        # trip on a task that did real work pre-compaction. False
        # negative (a genuinely tool-less long-running task that resumed
        # from a checkpoint) is acceptable; false positive (failing a
        # real task because compaction hid its tool calls) is not.
        if start_iteration > 0 and tool_calls_count == 0:
            tool_calls_count = 1
            outbound_tool_calls_count = 1

        try:
            for iteration in range(start_iteration, self.MAX_ITERATIONS):
                self._bump_liveness()
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
                    logger.info(
                        "execute_task exit branch=cancelled_iter_head status=%s "
                        "iterations=%d tokens=%d",
                        result.status, iteration, total_tokens,
                    )
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
                    logger.info(
                        "execute_task exit branch=token_budget_exhausted "
                        "status=%s iterations=%d tokens=%d",
                        result.status, iteration, total_tokens,
                    )
                    return result

                # === DECIDE (LLM call) ===
                # Refresh system prompt with context warning if applicable
                effective_system = system_prompt
                if self.context_manager:
                    warning = self.context_manager.context_warning(messages)
                    if warning:
                        effective_system = system_prompt + f"\n\n## {warning}"

                available_tools = self.tools.get_tool_definitions(**self._tool_filter_kw) or None
                llm_response = await _llm_call_with_retry(
                    self.llm.chat_collect,
                    system=effective_system,
                    messages=messages,
                    tools=available_tools,
                )
                # Bug 1 (codex P2 r2): tick after the LLM call returns —
                # a single deep-research call can run >5 min, and bumping
                # only at iteration head would let the staleness check fire
                # during a perfectly healthy turn.
                self._bump_liveness()
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
                    logger.info(
                        "execute_task exit branch=cancelled_post_llm "
                        "status=%s iterations=%d tokens=%d",
                        result.status, iteration, total_tokens,
                    )
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
                        logger.info(
                            "execute_task exit branch=tool_loop_terminate "
                            "status=%s iterations=%d tokens=%d",
                            result.status, iteration, total_tokens,
                        )
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
                    # Bug F (codex r4): track tool dispatches independently
                    # of the message list so summarizing compaction can't
                    # drop the signal mid-task.
                    tool_calls_count += len(llm_response.tool_calls)
                    # Round-5: parallel counter for outbound-effect calls
                    # only — a turn that ran nothing but ``check_inbox``
                    # still trips the lazy-completion guard.
                    outbound_tool_calls_count += sum(
                        1 for tc in llm_response.tool_calls
                        if tc.name not in _READ_ONLY_TOOLS
                    )

                    # Execute tools — parallel-safe tools run concurrently
                    self._current_messages = messages
                    tool_results = await self._run_tools_parallel(
                        llm_response.tool_calls,
                    )
                    # Bug 1 (codex P2 r2): tick after tool execution — a
                    # long-running shell or browser action can sit at the
                    # 300s _TOOL_TIMEOUT cap. Without this bump the next
                    # iteration head might already be past the staleness
                    # threshold even though the loop is healthy.
                    self._bump_liveness()
                    for i, (result_str, _result) in enumerate(tool_results):
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call_entries[i]["id"],
                            "content": result_str,
                        }
                        # Tag tool results from coordination tools with agent provenance
                        _tc_name = llm_response.tool_calls[i].name
                        if _tc_name in ("check_inbox",):
                            _from = _result.get("from_agent", "unknown") if isinstance(_result, dict) else "unknown"
                            tool_msg["_origin"] = f"agent:{_from}"
                        messages.append(tool_msg)

                    # Rebuild system prompt after tool hot-reload
                    if self._tools_reloaded:
                        self._tools_reloaded = False
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

                    # Checkpoint after compaction — restored messages will already be
                    # context-managed, preventing oversized context on resume.
                    await self._checkpoint_task(assignment, assignment_json, messages, iteration, total_tokens)

                else:
                    # LLM returned text with no tool calls.
                    # Codex r8 (post-regression-fix audit): detect a
                    # structured ``{"result": {...}}`` payload up front so
                    # the iter-0 nudge can skip when the LLM already
                    # delivered a legitimate final answer. Without this
                    # hoist, an iter-0 structured noop with tools available
                    # gets nudged unnecessarily — wasting a turn and risking
                    # a false-positive lazy-completion failure if the LLM
                    # responds with "I already told you" on iter 1.
                    # Codex r9: extracted to ``_is_structured_final`` so the
                    # chat-path handoff auto-close can apply the same check
                    # without code duplication.
                    is_structured_final = self._is_structured_final(
                        llm_response.content,
                    )

                    # If this is iteration 0, the agent hasn't used any tools,
                    # AND tools are actually available, nudge it to take action
                    # — UNLESS the response is already a structured final
                    # answer (no point nudging a task that's correctly done).
                    if (
                        iteration == 0
                        and available_tools
                        and not is_structured_final
                    ):
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
                    duration_s = round(time.time() - start, 1)
                    iterations_executed = iteration + 1

                    # Bug 5 (pathological-success rejection): catch the ghost
                    # completion signature — first-iteration completion with
                    # zero LLM tokens AND empty content. That combination
                    # means the LLM never really did work (mocked no-op,
                    # checkpoint resurrection of a finished state, double-
                    # completion race). Token count alone is unreliable —
                    # some providers (Ollama, certain proxies) omit usage
                    # metadata so a legitimate first-turn answer can read
                    # as ``tokens_used=0``. Requiring empty content too
                    # avoids false-positive failures for those providers.
                    # Evaluated BEFORE success counters / logs / workspace
                    # activity so the failure path doesn't leave inconsistent
                    # "Task complete" entries behind (codex P3).
                    _raw_content = (llm_response.content or "").strip()
                    if (
                        iterations_executed == 1
                        and total_tokens == 0
                        and not _raw_content
                    ):
                        self.state = "idle"
                        self.current_task = None
                        self.tasks_failed += 1
                        logger.error(
                            "execute_task pathological success guard tripped "
                            "for task=%s: iterations=%d tokens=%d — downgrading "
                            "to failed (possible checkpoint corruption)",
                            assignment.task_id, iterations_executed, total_tokens,
                        )
                        if self.workspace:
                            self.workspace.append_activity(
                                trigger="task",
                                summary=(
                                    f"Task failed (pathological guard): "
                                    f"{assignment.task_type} | "
                                    f"{iterations_executed} iterations, "
                                    f"{total_tokens} tokens, {duration_s}s"
                                ),
                                tools_used=[],
                                duration_ms=int(duration_s * 1000),
                                tokens_used=total_tokens,
                                outcome="failed",
                            )
                        result = TaskResult(
                            task_id=assignment.task_id,
                            status="failed",
                            error=(
                                "execute_task returned success without doing "
                                "work — possible checkpoint corruption"
                            ),
                            tokens_used=total_tokens,
                            duration_ms=int(duration_s * 1000),
                        )
                        self._last_result = result
                        logger.info(
                            "execute_task exit branch=pathological_success_guard "
                            "status=%s iterations=%d tokens=%d",
                            result.status, iterations_executed, total_tokens,
                        )
                        return result

                    # Bug F (lazy completion) guard: when the LLM reaches
                    # the final-answer branch having used zero tools AND
                    # its reply is plain chatter (not a structured
                    # ``{"result": {...}}`` JSON response), that's a
                    # "I'll do it now" / "Done!" acknowledgment with no
                    # real work. ``is_structured_final`` was computed once
                    # at the top of this branch (line ~893) and is reused
                    # here. PR #918's pathological-success guard misses
                    # this case because the LLM was actually called
                    # (tokens > 0) and content is non-empty chatter.
                    # The structured-final contract:
                    #  - r4: ``{"result": ...}`` payload escapes the guard
                    #    (legitimate "impossible" / "the answer is X").
                    #  - r5: ``result`` must be a DICT (no chatter wrapped
                    #    as ``{"result": "I'll do it now"}``).
                    #  - r6: dict must be NON-EMPTY (no ``{"result": {}}``
                    #    compact bypass).
                    # Regression follow-up to #932: the prior guard required
                    # ``iterations_executed > 1`` on the assumption that the
                    # iter-0 nudge ALWAYS fires for handoff-originated tasks
                    # and bumps the counter to 2 before this branch runs.
                    # That contract is fragile — the nudge is gated on
                    # ``available_tools`` (line ~890) and silently skips
                    # whenever tools are empty/None (tool-filter quirks,
                    # runtime-disabled tools, hot-reload races). Operator
                    # hit a trend-scout task that fell straight through
                    # with iterations_executed=1 and got marked done despite
                    # zero side effects. Drop the iteration count from the
                    # condition: ``tool_calls_count == 0 AND not
                    # structured_final`` is already the complete lazy-
                    # completion signature. The iter-0 nudge is a SOFT path
                    # (give the model one chance to course-correct); this
                    # guard is the HARD fail that doesn't depend on the
                    # nudge having run. Legitimate one-iteration noop tasks
                    # must return the documented contract
                    # ``{"result": {"status": "noop", "reason": "..."}}``,
                    # which escapes via ``is_structured_final``.
                    if outbound_tool_calls_count == 0 and not is_structured_final:
                        read_only_count = tool_calls_count - outbound_tool_calls_count
                        final_text = (llm_response.content or "").strip()
                        # Bug 3 (explained-deferral carve-out): an agent that
                        # called at least one READ-ONLY tool to verify a
                        # prerequisite (backpressure / dedup / state check)
                        # AND then explained its decision in plain prose —
                        # rather than the structured ``{"result": {...}}``
                        # envelope — is correctly DEFERRING, not ghosting. It
                        # examined state and made a reasoned call. Close as
                        # ``done`` with a ``deferred`` result payload (mirrors
                        # the structured-noop success shape; ``deferred`` lives
                        # only inside the payload — there is no ``deferred``
                        # task STATUS, see VALID_STATUSES). The genuine ghost
                        # case (zero tools, OR no explanation) still hard-fails
                        # below.
                        if read_only_count > 0 and final_text:
                            self.state = "idle"
                            self.current_task = None
                            self.tasks_completed += 1
                            logger.info(
                                "execute_task explained-deferral carve-out for "
                                "task=%s: iterations=%d tokens=%d "
                                "outbound_tool_calls=0 read_only_tool_calls=%d "
                                "non_empty_final=True — closing done with "
                                "deferred result (agent examined state via "
                                "read-only tools and explained its decision)",
                                assignment.task_id, iterations_executed,
                                total_tokens, read_only_count,
                            )
                            if self.workspace:
                                self.workspace.append_activity(
                                    trigger="task",
                                    summary=(
                                        f"Task deferred: {assignment.task_type} | "
                                        f"{iterations_executed} iterations, "
                                        f"{total_tokens} tokens, {duration_s}s | "
                                        f"{read_only_count} read-only tool call(s)"
                                    ),
                                    tools_used=[],
                                    duration_ms=int(duration_s * 1000),
                                    tokens_used=total_tokens,
                                    outcome="deferred",
                                )
                            result = TaskResult(
                                task_id=assignment.task_id,
                                status="complete",
                                result={
                                    "status": "deferred",
                                    "reason": final_text[:500],
                                    # ``summary`` mirrors the normal done-close
                                    # convention so the done back-edge (server
                                    # reads ``result.summary``) surfaces the
                                    # deferral context to the originator's
                                    # check_inbox instead of an empty event.
                                    "summary": final_text[:500],
                                },
                                tokens_used=total_tokens,
                                duration_ms=int(duration_s * 1000),
                            )
                            self._last_result = result
                            logger.info(
                                "execute_task exit branch=explained_deferral "
                                "status=%s iterations=%d tokens=%d",
                                result.status, iterations_executed, total_tokens,
                            )
                            return result
                        self.state = "idle"
                        self.current_task = None
                        self.tasks_failed += 1
                        logger.error(
                            "execute_task lazy-completion guard tripped for "
                            "task=%s: iterations=%d tokens=%d "
                            "outbound_tool_calls=0 read_only_tool_calls=%d "
                            "structured_final=False — downgrading to failed "
                            "(no outbound effect produced; LLM either made "
                            "no tool calls or called only read-only / "
                            "diagnostic tools)",
                            assignment.task_id, iterations_executed,
                            total_tokens, read_only_count,
                        )
                        if self.workspace:
                            self.workspace.append_activity(
                                trigger="task",
                                summary=(
                                    f"Task failed (no_outbound_effects): "
                                    f"{assignment.task_type} | "
                                    f"{iterations_executed} iterations, "
                                    f"{total_tokens} tokens, {duration_s}s | "
                                    f"{read_only_count} read-only tool call(s)"
                                ),
                                tools_used=[],
                                duration_ms=int(duration_s * 1000),
                                tokens_used=total_tokens,
                                outcome="failed",
                            )
                        result = TaskResult(
                            task_id=assignment.task_id,
                            status="failed",
                            error=(
                                "no_outbound_effects: the task completed "
                                f"{iterations_executed} iteration(s) without "
                                "producing any outbound effect (no hand_off, "
                                "write_blackboard, notify_user, publish_event, "
                                "etc.) and without emitting a structured "
                                "{\"result\": {...}} payload"
                                + (
                                    f" — {read_only_count} read-only tool "
                                    "call(s) were made"
                                    if read_only_count else ""
                                )
                                + ". Tasks must either perform real work via "
                                "an outbound tool OR return a structured "
                                "result envelope (e.g. {\"result\": "
                                "{\"status\": \"noop\", \"reason\": \"...\"}})."
                            ),
                            tokens_used=total_tokens,
                            duration_ms=int(duration_s * 1000),
                        )
                        self._last_result = result
                        logger.info(
                            "execute_task exit branch=lazy_completion_guard "
                            "status=%s iterations=%d tokens=%d",
                            result.status, iterations_executed, total_tokens,
                        )
                        return result

                    self.state = "idle"
                    self.current_task = None
                    self.tasks_completed += 1

                    logger.info(
                        f"Task {assignment.task_id} complete",
                        extra={"extra_data": {"iterations": iteration + 1, "tokens": total_tokens}},
                    )

                    # Log task completion to daily log + activity
                    if self.workspace:
                        task_tools = self._collect_tool_names(messages)
                        input_summary = truncate(str(assignment.input_data).replace("\n", " "), 120)
                        tools_str = ", ".join(task_tools) if task_tools else "none"
                        summary = (
                            f"Task complete: {assignment.task_type} | "
                            f"{iteration + 1} iterations, {total_tokens} tokens, {duration_s}s | "
                            f"Tools: {tools_str} | Input: {input_summary}"
                        )
                        self.workspace.append_daily_log(summary)
                        self.workspace.append_activity(
                            trigger="task",
                            summary=summary,
                            tools_used=task_tools,
                            duration_ms=int(duration_s * 1000),
                            tokens_used=total_tokens,
                            outcome="complete",
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
                    logger.info(
                        "execute_task exit branch=final_response "
                        "status=%s iterations=%d tokens=%d",
                        result.status, iterations_executed, total_tokens,
                    )
                    return result

            # Max iterations reached
            self.state = "idle"
            self.current_task = None
            self.tasks_failed += 1
            if self.workspace:
                input_summary = truncate(str(assignment.input_data).replace("\n", " "), 120)
                summary = (
                    f"Task FAILED (max iterations): {assignment.task_type} | "
                    f"{total_tokens} tokens | Input: {input_summary}"
                )
                self.workspace.append_daily_log(summary)
                self.workspace.append_activity(
                    trigger="task",
                    summary=summary,
                    duration_ms=int((time.time() - start) * 1000),
                    tokens_used=total_tokens,
                    outcome="failed",
                )
            result = TaskResult(
                task_id=assignment.task_id,
                status="failed",
                error=f"Max iterations ({self.MAX_ITERATIONS}) reached",
                tokens_used=total_tokens,
                duration_ms=int((time.time() - start) * 1000),
            )
            self._last_result = result
            logger.info(
                "execute_task exit branch=max_iterations_reached status=%s "
                "iterations=%d tokens=%d",
                result.status, self.MAX_ITERATIONS, total_tokens,
            )
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
            logger.info(
                "execute_task exit branch=cancelled_exception status=%s tokens=%d",
                result.status, total_tokens,
            )
            return result
        except LLMAuthError as e:
            # Distinguished credential failure. Recording is mesh-side
            # only (CredentialVault.execute_api_call already called the
            # auth-failure recorder before tagging the response) — the
            # agent does NOT self-report here, otherwise each failure
            # would double-increment and trip quarantine at 2 instead
            # of 3. The mesh endpoint is the failsafe for paths that
            # don't go through the proxy (e.g. legacy clients) but
            # within the loop we trust the proxy path.
            self.state = "idle"
            self.current_task = None
            self.tasks_failed += 1
            result = TaskResult(
                task_id=assignment.task_id,
                status="failed",
                error=f"auth_failure: {e}",
                tokens_used=total_tokens,
                duration_ms=int((time.time() - start) * 1000),
            )
            self._last_result = result
            logger.info(
                "execute_task exit branch=auth_failure status=%s tokens=%d provider=%s",
                result.status, total_tokens, e.provider,
            )
            return result
        except LLMConfigError as e:
            # Distinguished config failure — agent NOT quarantined; this
            # is operator misconfig that's actionable via edit_agent.
            self.state = "idle"
            self.current_task = None
            self.tasks_failed += 1
            result = TaskResult(
                task_id=assignment.task_id,
                status="failed",
                error=f"config_error: {e}",
                tokens_used=total_tokens,
                duration_ms=int((time.time() - start) * 1000),
            )
            self._last_result = result
            logger.info(
                "execute_task exit branch=config_error status=%s tokens=%d",
                result.status, total_tokens,
            )
            return result
        except Exception as e:
            self.state = "idle"
            self.current_task = None
            self.tasks_failed += 1
            logger.error(f"Task {assignment.task_id} failed: {e}", exc_info=True)
            if self.workspace:
                error_summary = truncate(str(e).replace("\n", " "), 200)
                summary = f"Task FAILED (error): {assignment.task_type} | {error_summary}"
                self.workspace.append_daily_log(summary)
                self.workspace.append_activity(
                    trigger="task",
                    summary=summary,
                    duration_ms=int((time.time() - start) * 1000),
                    tokens_used=total_tokens,
                    outcome="error",
                )
            result = TaskResult(
                task_id=assignment.task_id,
                status="failed",
                error=str(e),
                tokens_used=total_tokens,
                duration_ms=int((time.time() - start) * 1000),
            )
            self._last_result = result
            logger.info(
                "execute_task exit branch=exception status=%s tokens=%d error=%r",
                result.status, total_tokens, str(e)[:120],
            )
            return result
        finally:
            # Clear task checkpoint on ANY exit (success, failure, cancel, exception).
            if self.memory:
                try:
                    await self.memory._run_db(self.memory.clear_task_checkpoint)
                except BaseException:
                    logger.debug("Failed to clear task checkpoint", exc_info=True)

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

        # Enrich any 📎 /data/uploads/ references in the task so a worker handed
        # an image/PDF can actually see it (mirrors the chat path). Returns a
        # plain string unchanged when there is nothing to enrich.
        content = enrich_message_with_attachments("\n\n".join(parts))
        return [{"role": "user", "content": content}]

    def _trim_context(self, messages: list[dict], max_tokens: int = 100_000) -> list[dict]:
        """Trim old tool exchanges to manage context window.

        Groups messages into tool-call groups (assistant+tool responses)
        via the shared :func:`group_messages_by_tool_call` helper so we
        never split a tool-call from its results.
        """
        from src.agent.context import _content_chars, group_messages_by_tool_call
        estimated_tokens = sum(
            _content_chars(m.get("content", "")) // 4 + len(json.dumps({
                k: v for k, v in m.items() if k != "content"
            })) // 4
            for m in messages
        )
        if estimated_tokens <= max_tokens:
            return messages

        groups = group_messages_by_tool_call(messages)

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
        elif first_group[0].get("role") == "user":
            first_msg = first_group[0]
            content0 = first_msg.get("content")
            if isinstance(content0, list):
                # Multimodal first message (e.g. an uploaded image): append the
                # summary as a trailing text block so the turn stays a single
                # user message and the LLM role-alternation invariant holds.
                merged = content0 + [{"type": "text", "text": summary_text}]
            else:
                merged = (content0 or "") + summary_text
            result = [{**first_msg, "content": merged}] + first_group[1:]
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

    async def _maybe_reload_tools(self, result: Any) -> None:
        """If a tool returned reload_requested, hot-reload the tool registry.

        Sets ``_tools_reloaded`` so callers can rebuild system prompts
        with updated tool descriptions.  Re-registers with the mesh so the
        dashboard receives an ``agent_state: registered`` event and can
        refresh the capabilities view in real time.
        """
        if isinstance(result, dict) and result.get("reload_requested"):
            count = self.tools.reload()
            self._tools_reloaded = True
            logger.info(f"Hot-reloaded tools: {count} available")
            try:
                await self.mesh_client.register(
                    capabilities=self.tools.list_tools(),
                )
            except Exception as e:
                logger.warning("Failed to re-register after tool reload: %s", e)

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
            f"- For a non-trivial task, call skills_list first — if a skill "
            f"(saved step-by-step procedure) fits, skill_view it and follow it.\n"
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

        index_text = self._refresh_grouped_plan()
        if index_text:
            parts.append(index_text)

        return "\n\n".join(parts)

    # Round-4 structural fix: hand_off failures are now enforced from
    # tool-output evidence, not the LLM's discretion. Any of these
    # flags in a ``hand_off`` tool result means the downstream chain
    # did not land — the task must NOT auto-close to ``done`` even if
    # the LLM produced text claiming success. The previous design
    # depended on the LLM faithfully reading the failure envelope's
    # ``MUST NOT report success`` directive, which models ignored
    # across three repro cycles. Enforcing system-side closes that
    # trust gap completely.
    _HANDOFF_FAILURE_FLAGS = frozenset({
        "create_failed", "wake_failed", "output_write_failed",
    })

    @staticmethod
    def _has_outbound_effect(tool_outputs: list[dict] | None) -> bool:
        """True if ``tool_outputs`` carries at least one non-read-only call.

        Workers that only invoke diagnostic reads (``check_inbox`` +
        ``read_blackboard`` + …) and then claim done are committing
        ghost-completion — they produced no outbound effect for
        downstream consumers. The lazy-completion guards treat such
        turns as if no tools were called at all. Unknown tools are
        treated as outbound by default — soft-fail open for new tools
        is better than blocking real work.
        """
        if not tool_outputs:
            return False
        for t in tool_outputs:
            name = t.get("tool") or t.get("name") or ""
            if name and name not in _READ_ONLY_TOOLS:
                return True
        return False

    @staticmethod
    def _last_round_terminal_completion(tool_outputs: list[dict] | None) -> bool:
        """True iff the most recent tool round closed the task for real.

        A "genuine completion" is a SUCCESSFUL terminal coordination tool:

          * ``complete_task`` whose output carries ``completed is True``
          * ``hand_off`` whose output carries ``handed_off is True``

        This is deliberately NARROWER than ``_has_outbound_effect``: a
        ``notify_user`` / ``write_blackboard`` / ``publish_event`` is an
        outbound side-effect but NOT a task completion, so it must NOT pre-empt
        the convergence cap. A FAILED terminal tool (``handed_off=False`` /
        ``complete_task_failed`` flag) is likewise NOT a completion — it is
        handled by ``_chat_result_failure_reason`` (→ ``failed``) and must not
        short-circuit the cap either.

        Item 1 fix (Codex r4): the budgeted chat loop calls this AFTER tool
        execution. When it returns True, the loop returns the NORMAL
        (non-capped) result immediately so chat()'s done/handoff close runs,
        instead of continuing to the next round's top-of-loop cap check that
        would otherwise override the just-landed completion with ``blocked``.

        Only the LAST round's outputs matter (the round that just executed),
        but scanning the full accumulated list is equivalent and simpler: a
        successful terminal tool anywhere in the turn means the task converged.
        """
        if not tool_outputs:
            return False
        for t in tool_outputs:
            name = t.get("tool") or t.get("name") or ""
            output = t.get("output")
            if not isinstance(output, dict):
                continue
            if name == "complete_task" and output.get("completed") is True:
                return True
            if name == "hand_off" and output.get("handed_off") is True:
                return True
        return False

    @staticmethod
    def _chat_result_failure_reason(result: dict) -> str | None:
        """Return a failure-reason string if ``result`` carries a known
        ``_chat_inner`` failure marker, else None.

        Codex r10: ``_chat_inner`` catches ``LLMAuthError``,
        ``LLMConfigError``, and bare ``Exception`` and returns ordinary
        result dicts tagged with ``auth_failure``, ``config_error``, or
        ``exception_caught`` respectively. ``tool_limit_reached`` is
        already in the result envelope for bounded-iteration exits.
        The chat() auto-close site checks these BEFORE the
        lazy-completion guard so a failure after ≥1 tool call doesn't
        slip past the empty-tool_outputs check and auto-close as
        ``done``.

        Round-4 extension: also scans ``tool_outputs`` for ``hand_off``
        results that indicate the downstream chain failed. If any
        hand_off in this turn returned ``handed_off=False`` or any of
        ``create_failed`` / ``wake_failed`` / ``output_write_failed``,
        the task fails — the LLM no longer has the option to mark
        ``done`` after a broken handoff just by producing text.
        """
        if not isinstance(result, dict):
            return None
        if result.get("tool_limit_reached"):
            return "max_iterations_reached"
        if result.get("auth_failure"):
            return f"auth_failure: {(result.get('response') or '')[:400]}"
        if result.get("config_error"):
            return f"config_error: {(result.get('response') or '')[:400]}"
        if result.get("exception_caught"):
            return f"exception: {(result.get('response') or '')[:400]}"
        # System-side handoff enforcement (Round-5 key-name fix).
        # The ``tool_outputs`` schema is ``{"tool": tool_name, "input":
        # ..., "output": ...}`` per ``_chat_inner``. PR #953 mistakenly
        # scanned ``tool_out["name"]`` which never matches any real
        # output — enforcement was dead code in production. Tests
        # passed because they were written against the wrong key too.
        for tool_out in result.get("tool_outputs") or []:
            if not isinstance(tool_out, dict):
                continue
            if tool_out.get("tool") != "hand_off":
                continue
            payload = tool_out.get("output") or tool_out.get("result") or {}
            if not isinstance(payload, dict):
                continue
            failing = [
                k for k in AgentLoop._HANDOFF_FAILURE_FLAGS
                if payload.get(k)
            ]
            if failing or payload.get("handed_off") is False:
                target = payload.get("to") or "unknown"
                detail = (payload.get("error") or "")[:300]
                return (
                    f"handoff_failed: hand_off to {target!r} reported "
                    f"{failing or ['handed_off=False']} — {detail}"
                )
        return None

    def _is_structured_final(self, content: str | None) -> bool:
        """True iff ``content`` is the documented final-answer contract.

        The contract is whole-content JSON: ``{"result": {...}}`` where
        ``result`` is a non-empty dict. Rejects scalar/empty/list/null
        ``result`` values and any non-JSON wrapper (fenced markdown,
        prose-wrapped JSON, etc.). Single source of truth for the
        lazy-completion guards in ``execute_task`` and the handoff
        auto-close in ``chat``.
        """
        try:
            parsed = json.loads(content or "")
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(parsed, dict):
            return False
        result_field = parsed.get("result")
        return isinstance(result_field, dict) and bool(result_field)

    def _parse_final_output(self, content: str) -> tuple[dict, dict]:
        """Parse the LLM's final response into result data and blackboard promotions."""
        try:
            parsed = json.loads(content)
            return parsed.get("result", {"raw": content}), parsed.get("promote", {})
        except (json.JSONDecodeError, AttributeError):
            return {"raw": content}, {}

    # ── Heartbeat mode ────────────────────────────────────────

    async def execute_heartbeat(self, message: str, *, force_llm: bool = False) -> dict:
        """Execute an autonomous heartbeat — stateless, separate from chat.

        Returns a structured dict with response, tools used, duration, etc.
        Does NOT touch _chat_messages.  Uses its own message list and the
        _heartbeat_mode ContextVar so tools can detect heartbeat context.
        Notifications are still persisted to the chat transcript so users
        can find them in chat history.

        ``force_llm`` (Bug 6 — codex P2 r2): pipeline-kicker agents have
        no probes and ship with an empty HEARTBEAT.md, which makes the
        ``no_heartbeat_rules`` skip below fire on every tick. When the
        cron job sets ``force_llm: true`` the dispatcher forwards the
        flag (via ``x-force-llm`` header) so the LLM is invoked anyway.
        The ``agent_busy`` skip is NOT bypassed — busy is busy.
        """
        # Restrict operator to heartbeat-only tools during unsupervised execution.
        # Non-operator agents have _allowed_tools=None, so the swap is skipped.
        saved_allowed = getattr(self, '_allowed_tools', None)
        # All heartbeats — operator and non-operator — share the same
        # iteration budget. The operator's heartbeat procedure now lists
        # 7 numbered steps plus per-cycle goal seeding and up to 10
        # rate_delivery calls — 5 iterations couldn't cover the prompted
        # work, so the operator was silently truncating mid-procedure.
        # ``HEARTBEAT_MAX_ITERATIONS=12`` matches the prompt's stated
        # budget of "10 tool calls per cycle, 12 leaves headroom for
        # the final assistant turn." (Codex pre-merge review of PR 972
        # flagged the 5-cap as inconsistent with the prompted budget.)
        max_iters = HEARTBEAT_MAX_ITERATIONS
        if saved_allowed is not None:
            self._allowed_tools = _HEARTBEAT_TOOLS
        try:
            # Don't run if the agent is busy with a task, chat, or queued chat
            if self.state != "idle" or self._chat_lock.locked():
                return {"skipped": True, "reason": "agent_busy"}

            # Skip the LLM call entirely when HEARTBEAT.md has no actionable
            # content and no goals are set — saves tokens on empty heartbeats.
            # Bug 6 fix: ``force_llm`` lets the operator opt out of this
            # optimization for pipeline-kicker agents.
            if (
                not force_llm
                and self.workspace
                and _is_heartbeat_empty(self.workspace.load_heartbeat_rules())
            ):
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
                    f"- You have max {max_iters} iterations.\n"
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

                index_text = self._refresh_grouped_plan()
                if index_text:
                    parts.append(index_text)

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
                messages: list[dict] = [{"role": "user", "content": message, "_origin": "system:heartbeat"}]

                for _iteration in range(max_iters):
                    self._bump_liveness()
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
                    _remaining = max_iters - _iteration
                    if _remaining == 2:
                        messages.append({
                            "role": "user",
                            "content": (
                                "[SYSTEM] You have 2 iterations remaining. "
                                "Start wrapping up — use notify_user to report "
                                "your results, then give your final answer."
                            ),
                            "_origin": "system:heartbeat",
                        })
                    elif _remaining == 1:
                        messages.append({
                            "role": "user",
                            "content": (
                                "[SYSTEM] LAST iteration. Give your final answer "
                                "now. Do NOT call any more tools."
                            ),
                            "_origin": "system:heartbeat",
                        })

                    # On the very last iteration, withhold tools so the LLM is
                    # forced to produce a text-only response.
                    iter_tools = (
                        None if _remaining == 1
                        else self.tools.get_tool_definitions(
                            **self._tool_filter_kw,
                        ) or None
                    )

                    llm_response = await _llm_call_with_retry(
                        self.llm.chat_collect,
                        system=system_prompt,
                        messages=messages,
                        tools=iter_tools,
                    )
                    # Bug 1 (codex P2 r2): tick after the LLM call returns —
                    # a single deep-research call can run >5 min, and bumping
                    # only at iteration head would let the staleness check fire
                    # during a perfectly healthy turn.
                    self._bump_liveness()
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
                    self._current_messages = messages
                    tool_results = await self._run_tools_parallel(
                        llm_response.tool_calls,
                    )
                    # Bug 1 (codex P2 r2): tick after tool execution — a
                    # long-running shell or browser action can sit at the
                    # 300s _TOOL_TIMEOUT cap. Without this bump the next
                    # iteration head might already be past the staleness
                    # threshold even though the loop is healthy.
                    self._bump_liveness()
                    for i, (result_str, _result) in enumerate(tool_results):
                        hb_tool_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call_entries[i]["id"],
                            "content": result_str,
                        }
                        # Tag tool results from coordination tools with agent provenance
                        _hb_tc_name = llm_response.tool_calls[i].name
                        if _hb_tc_name in ("check_inbox",):
                            _hb_from = _result.get("from_agent", "unknown") if isinstance(_result, dict) else "unknown"
                            hb_tool_msg["_origin"] = f"agent:{_hb_from}"
                        messages.append(hb_tool_msg)

                    # Clear reload flag if set (heartbeat rarely creates tools,
                    # but the flag must be consumed to avoid stale state).
                    self._tools_reloaded = False

                    # Trim if context grows large
                    messages = self._trim_context(messages, max_tokens=_FALLBACK_MAX_TOKENS)

                # Safety net — should not normally be reached because the last
                # iteration withholds tools and forces a text response.
                self.state = "idle"
                duration_ms = int((time.time() - start) * 1000)
                if self.workspace:
                    self.workspace.append_activity(
                        trigger="heartbeat",
                        summary=f"Max iterations ({max_iters}) reached",
                        tools_used=tools_used,
                        duration_ms=duration_ms,
                        tokens_used=total_tokens,
                        outcome="max_iterations",
                    )
                return {
                    "response": f"Max iterations ({max_iters}) reached",
                    "summary": f"Max iterations ({max_iters}) reached",
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
            except LLMAuthError as e:
                # Heartbeat hit a credential failure. Same pattern as
                # execute_task — mesh-side recording is authoritative,
                # the agent does NOT self-report to avoid double-counting.
                self.state = "idle"
                duration_ms = int((time.time() - start) * 1000)
                logger.warning("Heartbeat auth failure: %s", e)
                if self.workspace:
                    self.workspace.append_activity(
                        trigger="heartbeat",
                        summary=f"Auth failure: {e}",
                        tools_used=tools_used,
                        duration_ms=duration_ms,
                        tokens_used=total_tokens,
                        outcome="error",
                    )
                return {
                    "response": f"Auth failure: {e}",
                    "summary": f"Auth failure: {e}",
                    "tools_used": tools_used,
                    "duration_ms": duration_ms,
                    "tokens_used": total_tokens,
                    "outcome": "auth_failure",
                    "skipped": False,
                }
            except LLMConfigError as e:
                self.state = "idle"
                duration_ms = int((time.time() - start) * 1000)
                logger.warning("Heartbeat config error: %s", e)
                if self.workspace:
                    self.workspace.append_activity(
                        trigger="heartbeat",
                        summary=f"Config error: {e}",
                        tools_used=tools_used,
                        duration_ms=duration_ms,
                        tokens_used=total_tokens,
                        outcome="error",
                    )
                return {
                    "response": f"Config error: {e}",
                    "summary": f"Config error: {e}",
                    "tools_used": tools_used,
                    "duration_ms": duration_ms,
                    "tokens_used": total_tokens,
                    "outcome": "config_error",
                    "skipped": False,
                }
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
        finally:
            if saved_allowed is not None:
                self._allowed_tools = saved_allowed

    # ── Chat mode ──────────────────────────────────────────────

    CHAT_MAX_TOOL_ROUNDS = 30
    CHAT_MAX_TOTAL_ROUNDS = 200
    _CHAT_ROUND_WARNING = 160
    _MAX_SESSION_CONTINUES = 5
    # Class-default fallback for the per-task convergence budget. The live
    # value is the per-instance ``self.TASK_MAX_TOOL_ROUNDS`` set in
    # ``__init__`` (env-clamped). This attr only exists so the constant is
    # discoverable next to the other CHAT_* bounds and gives a sane default
    # if an AgentLoop is ever constructed without running ``__init__``.
    TASK_MAX_TOOL_ROUNDS = 20
    # Codex #3 size bound: hard cap on the number of distinct task entries
    # held in ``self._task_round_counts``. The dict is freed per-task on
    # terminal close, so it normally tracks only in-flight tasks (small). This
    # bound guards the degenerate case where sustained mesh-write failures keep
    # terminal closes from popping entries — once exceeded, adding a new task
    # evicts an arbitrary existing entry instead of growing unbounded.
    _TASK_ROUND_COUNTS_MAX = 256
    # Append the convergence wrap-up directive to the SYSTEM PROMPT (NOT a
    # chat message — that would break LLM role-alternation, see Codex
    # finding #1) once the per-task budget has this many rounds or fewer
    # remaining. A second, harder escalation fires at ``<= 1`` remaining.
    _TASK_CONVERGENCE_NUDGE_REMAINING = 4
    # Static blocker_note used when the per-task budget is exhausted without
    # a terminal coordination tool. Short + static-worded on purpose — never
    # interpolate untrusted task content into this note.
    _TASK_CONVERGENCE_BLOCKER_NOTE = (
        "convergence_cap: task hit its per-task round budget without "
        "completing — produce the deliverable and call complete_task or "
        "hand_off"
    )
    # System-prompt suffixes for the convergence nudge (Codex finding #1).
    # Appended to THIS round's ``system`` string — never injected as a chat
    # message — so the user/assistant/tool role-alternation the LLM call
    # requires is preserved. Static-worded; no untrusted content.
    _TASK_CONVERGENCE_SOFT_SUFFIX = (
        "\n\n## Task wrap-up\n"
        "You are nearing this task's per-task round budget. Wrap up this "
        "task soon — produce your deliverable and call complete_task or "
        "hand_off. Do not start new work."
    )
    _TASK_CONVERGENCE_FINAL_SUFFIX = (
        "\n\n## Task wrap-up — FINAL ROUND\n"
        "This is the final round for this task. Call complete_task or "
        "hand_off now, or give your final answer. Do NOT start new work."
    )

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

    async def _checkpoint_task(
        self,
        assignment: TaskAssignment,
        assignment_json: str,
        messages: list[dict],
        iteration: int,
        tokens_used: int,
    ) -> None:
        """Persist task state for crash recovery. Called after each iteration."""
        if not self.memory:
            return
        try:
            budget_used = assignment.token_budget.used_tokens if assignment.token_budget else 0
            budget_cost = assignment.token_budget.estimated_cost_usd if assignment.token_budget else 0.0
            flush_triggered = self.context_manager._flush_triggered if self.context_manager else False
            await self.memory._run_db(
                self.memory.save_task_checkpoint,
                assignment.task_id,
                assignment_json,
                messages,
                iteration,
                tokens_used,
                budget_used,
                budget_cost,
                flush_triggered,
            )
        except Exception as e:
            logger.warning("Failed to save task checkpoint: %s", e)

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

    async def chat(
        self, user_message: str, *, trace_id: str | None = None,
        origin: "MessageOrigin | None" = None,
        task_id: str | None = None,
    ) -> dict:
        """Handle a single chat turn with persistent conversation history.

        On first message of a session, loads workspace context (INSTRUCTIONS.md,
        SOUL.md, USER.md, MEMORY.md, daily logs) into the system prompt
        and auto-searches memory for relevant context.

        Uses an asyncio.Lock so concurrent callers queue instead of being
        rejected.  The lock serialises chat turns; the /status endpoint
        remains available while the lock is held.

        ``task_id`` (Bug 2 fix) — when set, the chat turn is treated as
        the execution of that specific task. On successful return we
        auto-call ``set_task_status(task_id, "done")`` with a brief
        summary; on exception we call it with ``"failed"`` and the error
        string. Without ``task_id`` (legacy callers, heartbeats, manual
        chat) the auto-close is skipped entirely.

        Returns {"response": str, "tool_outputs": list[dict], "tokens_used": int}.
        """
        # Redirect to steer queue if a task is running — prevents concurrent
        # state corruption (shared loop_detector, state, flush_triggered).
        # Guard is BEFORE _chat_lock to avoid blocking on a held lock.
        if self.current_task is not None:
            # Two distinct paths depending on whether this is a real
            # handoff (task_id set) or a free-form chat:
            #
            # - Handoff path: reject cleanly. Closing the task as
            #   ``failed`` AND queueing it on the steer would let the
            #   originator retry/reroute (because they see task_failed)
            #   while THIS agent still processes the queued message —
            #   duplicate / conflicting work (codex P2). Skip the
            #   queue, surface the rejection in the chat reply.
            # - Free chat path: keep legacy behavior — queue the message
            #   into the active conversation. No durable task exists,
            #   so there's no double-execution risk.
            if task_id:
                await self._auto_close_task(
                    task_id, "failed",
                    error="agent_busy_handoff_rejected",
                )
                return {
                    "response": (
                        "Agent is working on another task — handoff "
                        "rejected. Originator notified via back-edge."
                    ),
                    "tool_outputs": [],
                    "tokens_used": 0,
                }
            await self._steer_queue.put(user_message)
            return {
                "response": (
                    "Agent is working on a task. Your message has been queued "
                    "and will be included in the next conversation turn."
                ),
                "tool_outputs": [],
                "tokens_used": 0,
            }

        from src.shared.trace import (
            current_origin,
            current_task_id,
            current_trace_id,
        )
        current_trace_id.set(trace_id)
        origin_token = current_origin.set(origin)
        # ``current_task_id`` is set only when a real durable task drove
        # this chat turn (handoff/lane dispatch). Free chats and heartbeats
        # leave it ``None`` so downstream tools (``hand_off``) don't
        # falsely parent their new tasks off a non-existent ancestor.
        task_id_token = current_task_id.set(task_id)
        try:
            async with self._chat_lock:
                await self._maybe_restore_session()
                # State machine requires pending → working before → done,
                # so open the task as ``working`` here. Best-effort: a
                # same-state transition or a task already cancelled is
                # logged, not raised. ``_auto_close_task`` is misnamed —
                # it handles any auto-transition, not just terminal ones.
                if task_id:
                    await self._auto_close_task(task_id, "working")
                try:
                    try:
                        result = await self._chat_inner(user_message)
                    except asyncio.CancelledError:
                        # Codex r10 (MEDIUM): _chat_inner re-raises
                        # CancelledError, which is BaseException and slips
                        # past the ``except Exception`` arm below. Without
                        # this branch the durable task stays at
                        # ``working`` forever after a cancellation.
                        if task_id:
                            await self._auto_close_task(
                                task_id, "cancelled",
                                error="chat_cancelled",
                            )
                        raise
                    except Exception as e:
                        # Auto-fail the originating task before propagating.
                        if task_id:
                            await self._auto_close_task(
                                task_id, "failed",
                                error=str(e)[:500],
                            )
                        raise
                    # Auto-close on success. Three terminal branches:
                    #   1. Failure markers in ``result`` → ``failed``. Codex
                    #      r10: ``_chat_inner`` catches LLMAuthError /
                    #      LLMConfigError / generic Exception and returns
                    #      a normal-shaped dict tagged with the relevant
                    #      flag (``auth_failure`` / ``config_error`` /
                    #      ``exception_caught``) plus ``tool_limit_reached``
                    #      for the bounded-iteration exit. Without this
                    #      check, an exception fired AFTER one or more
                    #      tool calls would surface as a result with
                    #      non-empty ``tool_outputs`` and slip past the
                    #      lazy-completion guard → task auto-closes as
                    #      ``done`` despite the failure.
                    #   2. Lazy-completion guard (codex r9): empty
                    #      ``tool_outputs`` AND non-structured response.
                    #      Handoff tasks must perform work via tools or
                    #      return ``{"result": {...}}``.
                    #   3. Success → ``done`` with response prefix as
                    #      summary.
                    if task_id:
                        # RC-1 (Codex finding #3): per-task convergence cap
                        # reached ONLY if the agent did NOT converge on its
                        # own. ``task_convergence_capped`` is set exclusively
                        # by ``_chat_inner``'s budget-exhausted fall-through —
                        # the path where the model kept emitting tool_calls
                        # past its per-task round budget. A genuine completion
                        # (the model returned a final text reply, or closed the
                        # task via complete_task / hand_off) DOES NOT set the
                        # flag, so it falls through to the normal success path
                        # below and closes ``done`` / handoff — the cap never
                        # pre-empts a real completion. When the flag IS set,
                        # drive the task to a TERMINAL ``blocked`` state with a
                        # short, static blocker_note (never dangling
                        # ``working``, never leaking untrusted task content).
                        # Checked BEFORE the generic failure-reason scan so the
                        # convergence note wins over the
                        # ``max_iterations_reached`` label the
                        # ``tool_limit_reached`` envelope would otherwise emit.
                        if result.get("task_convergence_capped"):
                            logger.warning(
                                "chat task=%s hit per-task convergence cap "
                                "(TASK_MAX_TOOL_ROUNDS=%d) — closing blocked",
                                task_id, self.TASK_MAX_TOOL_ROUNDS,
                            )
                            await self._auto_close_task(
                                task_id, "blocked",
                                blocker_note=self._TASK_CONVERGENCE_BLOCKER_NOTE,
                            )
                            return result
                        failure_reason = self._chat_result_failure_reason(result)
                        if failure_reason is not None:
                            logger.error(
                                "chat handoff task=%s closing as failed: %s",
                                task_id, failure_reason,
                            )
                            await self._auto_close_task(
                                task_id, "failed", error=failure_reason,
                            )
                        else:
                            response_text = result.get("response") or ""
                            tool_outputs = result.get("tool_outputs") or []
                            # Round-5 strengthening: ghost-completion
                            # was slipping past the previous guard
                            # whenever the LLM called ANY tool — even
                            # diagnostic reads (``check_inbox``,
                            # ``read_blackboard``) that produced no
                            # downstream work. Tighten to require at
                            # least one OUTBOUND-effect tool call OR a
                            # structured final payload.
                            outbound_used = AgentLoop._has_outbound_effect(tool_outputs)
                            handoff_is_lazy = (
                                not outbound_used
                                and not self._is_structured_final(response_text)
                            )
                            if handoff_is_lazy:
                                tool_names = [
                                    (t.get("tool") or t.get("name") or "?")
                                    for t in tool_outputs
                                ]
                                # Bug 3 (explained-deferral carve-out): if the
                                # turn called at least one READ-ONLY tool AND
                                # produced a non-empty prose explanation, the
                                # recipient correctly DEFERRED (examined state
                                # and decided) rather than ghosting. Close as
                                # ``done`` with a ``deferred`` result payload
                                # (mirrors the structured-noop success shape;
                                # ``deferred`` lives only inside the payload,
                                # the task STATUS stays ``done``). The genuine
                                # ghost case — zero tools OR empty response —
                                # still hard-fails below.
                                #
                                # ``silent_reply`` guard: a synthetic empty-turn
                                # marker (Bug 3) lands in ``response_text`` as
                                # non-empty prose that literally says no text was
                                # generated. ``_chat_inner`` is the only place that
                                # sets ``silent_reply=True`` alongside a non-empty
                                # response, so a marker turn must NOT be treated as
                                # a genuine deferral explanation — otherwise a ghost
                                # (read-only tools, originally empty) would slip
                                # into the deferral carve-out instead of failing.
                                if (
                                    tool_outputs
                                    and response_text.strip()
                                    and not result.get("silent_reply")
                                ):
                                    logger.info(
                                        "chat explained-deferral carve-out for "
                                        "handoff task=%s: outbound_effect=False "
                                        "non_empty_response=True tools_called=%s "
                                        "— auto-closing done with deferred "
                                        "result (recipient examined state via "
                                        "read-only tools and explained its "
                                        "decision)",
                                        task_id, tool_names,
                                    )
                                    await self._auto_close_task(
                                        task_id, "done",
                                        result_payload={
                                            "status": "deferred",
                                            "reason": response_text[:500],
                                            # ``summary`` feeds the done
                                            # back-edge (server reads
                                            # ``result.summary``) so the
                                            # originator's check_inbox shows the
                                            # deferral reason, not an empty event.
                                            "summary": response_text[:500],
                                        },
                                    )
                                else:
                                    logger.error(
                                        "chat lazy-completion guard tripped for "
                                        "handoff task=%s: outbound_effect=False "
                                        "structured_final=False tools_called=%s "
                                        "— auto-closing as failed (LLM either "
                                        "made no tool calls or called only "
                                        "read-only / diagnostic tools)",
                                        task_id, tool_names,
                                    )
                                    read_only_summary = (
                                        f" (read-only tools called: {tool_names})"
                                        if tool_names else ""
                                    )
                                    await self._auto_close_task(
                                        task_id, "failed",
                                        error=(
                                            "no_outbound_effects: the recipient "
                                            "completed the turn without producing "
                                            "any outbound effect (no hand_off, "
                                            "write_blackboard, notify_user, "
                                            "publish_event, etc.) and without "
                                            "returning a structured "
                                            "{\"result\": {...}} payload."
                                            f"{read_only_summary} Handoff tasks "
                                            "must either perform real work via an "
                                            "outbound tool OR return a structured "
                                            "result envelope (e.g. {\"result\": "
                                            "{\"status\": \"noop\", \"reason\": "
                                            "\"...\"}})."
                                        ),
                                        result_payload=(
                                            # Capture the worker's answer as the
                                            # deliverable — but NOT a synthetic
                                            # ``silent_reply`` marker (mirrors the
                                            # carve-out above), which is an
                                            # internal empty-turn placeholder, not
                                            # a real result.
                                            {"summary": response_text[:500]}
                                            if response_text.strip()
                                            and not result.get("silent_reply")
                                            else None
                                        ),
                                    )
                            else:
                                # Non-lazy success. When the worker returned a
                                # structured ``{"result": {...}}`` final (the
                                # answer-delivery shape this prompt now steers
                                # them to), pass the PARSED envelope through —
                                # exactly as execute_task does — so the
                                # originator's await/check_inbox surfaces the
                                # clean ``result.summary`` answer rather than the
                                # raw JSON wrapper. The mesh reads ``.summary``
                                # tolerantly (``.get`` → "" when absent, e.g. a
                                # noop), so no summary is synthesized here.
                                if self._is_structured_final(response_text):
                                    result_payload, _ = self._parse_final_output(
                                        response_text,
                                    )
                                else:
                                    result_payload = {
                                        "summary": response_text[:500],
                                    }
                                await self._auto_close_task(
                                    task_id, "done",
                                    result_payload=result_payload,
                                )
                    # Bug 3 final net: a chat turn must never surface an
                    # empty reply unless the model deliberately chose
                    # ``__SILENT__``. ``_chat_inner`` already retries the
                    # compose once (tools withheld) and substitutes a
                    # marker on every exit — normal, tool-limit, and the
                    # zero-tools case — REGARDLESS of ``task_id`` /
                    # ``tool_limit_reached``, setting ``silent_reply`` on
                    # the marker path so we can distinguish it here. This
                    # block is the belt-and-suspenders backstop: if a
                    # response still came back empty without the silence
                    # flag (e.g. a future code path that bypasses the
                    # inner substitution), stamp the marker so the user
                    # never sees a blank bubble. It now fires regardless
                    # of ``task_id`` and even with zero tool_outputs (the
                    # marker is generic in that case — see
                    # ``_synthesize_empty_chat_fallback``).
                    if (
                        not (result.get("response") or "").strip()
                        and not result.get("silent_reply")
                    ):
                        tool_outputs = result.get("tool_outputs") or []
                        logger.warning(
                            "chat empty-response final net tripped: %d tool "
                            "call(s) executed but response still empty after "
                            "inner retry/marker — surfacing synthetic notice",
                            len(tool_outputs),
                        )
                        # Single source of truth for the wording —
                        # ``_log_chat_turn`` writes the same string to the
                        # transcript so the dashboard chat view and
                        # ``/chat/history`` agree after a page refresh.
                        result["response"] = (
                            self._synthesize_empty_chat_fallback(tool_outputs)
                        )
                        result["silent_reply"] = True
                    return result
                finally:
                    await self._checkpoint_chat_session()
        finally:
            current_origin.reset(origin_token)
            current_task_id.reset(task_id_token)

    async def _auto_close_task(
        self, task_id: str, status: str, *,
        result_payload: dict | None = None,
        error: str | None = None,
        blocker_note: str | None = None,
    ) -> None:
        """Best-effort terminal transition. Never raises — failures log only.

        Log severity differentiates state-conflict (HTTP 400, the
        transition was already done or invalid for current state — benign)
        from infrastructure failures (network / 5xx — operator action
        required because the task will dangle in ``pending`` forever).

        ``blocker_note`` is forwarded directly to the mesh status endpoint.
        The mesh promotes ``error`` → ``blocker_note`` only for ``failed``
        transitions, so a ``blocked`` close that wants a reason must pass
        ``blocker_note`` explicitly (e.g. the convergence-cap close).
        """
        try:
            await self.mesh_client.set_task_status(
                task_id, status,
                result=result_payload,
                error=error,
                blocker_note=blocker_note,
            )
            # RC-3 (Codex finding #4): drop the advisory per-task convergence
            # count ONLY AFTER a successful NON-``working`` status write. The
            # budget is per-task, not forever, so a terminal close frees it.
            # ``working`` keeps the count alive (it's the in-progress marker
            # the chat() open transition uses). Popping BEFORE the write
            # raced: a 5xx / network failure left the task ``working`` on the
            # mesh but wiped the local budget, so a re-wake would get a fresh
            # full window. By popping only after success, a failed write keeps
            # the accumulated budget intact.
            if status != "working":
                self._task_round_counts.pop(task_id, None)
        except Exception as e:
            # Discriminate HTTP errors: 4xx is usually "state machine said
            # no" (benign, e.g. concurrent transition already landed);
            # 5xx / network / anything else means the mesh is unhealthy
            # and the task will silently dangle without an alert.
            resp = getattr(e, "response", None)
            http_status = getattr(resp, "status_code", None)
            if isinstance(http_status, int) and 400 <= http_status < 500:
                logger.warning(
                    "Auto-close %s → %s rejected by state machine (%s): %s",
                    task_id, status, http_status, e,
                )
                # RC-3 (Codex finding #4): a 4xx means the task is ALREADY in
                # a terminal/incompatible state (a concurrent transition
                # landed) — this agent's involvement is over, so it's safe to
                # drop the advisory per-task convergence count.
                if status != "working":
                    self._task_round_counts.pop(task_id, None)
            else:
                # No HTTP response or 5xx → the originating agent will
                # never see this task close. Surface loudly so operators
                # notice via standard error-log monitoring.
                logger.error(
                    "Auto-close %s → %s FAILED (%s) — task may dangle "
                    "in pending until a heartbeat or manual close: %s",
                    task_id, status, http_status or "no response", e,
                )

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
        self._chat_messages.append({"role": "user", "content": llm_content, "_origin": "user"})
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

        self._age_operator_playbooks()

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
                self.tools.execute(
                    tool_call.name,
                    tool_call.arguments,
                    mesh_client=self.mesh_client,
                    workspace_manager=self.workspace,
                    memory_store=self.memory,
                    _messages=self._current_messages,
                    agent_loop=self,
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
                result_str = dumps_safe(result) if isinstance(result, dict) else str(result)
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
                await self._maybe_reload_tools(result)
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
            safe = self.tools.is_parallel_safe(tool_calls[i].name)
            j = i + 1
            if safe:
                while j < n and self.tools.is_parallel_safe(tool_calls[j].name):
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
        self._current_messages = self._chat_messages
        tool_results = await self._run_tools_parallel(tool_calls)
        for i, (result_str, result) in enumerate(tool_results):
            msg = {
                "role": "tool",
                "tool_call_id": entries[i]["id"],
                "content": result_str,
            }
            # Tag tool results from coordination tools with agent provenance
            tool_name = tool_calls[i].name
            if tool_name in ("check_inbox",):
                from_agent = result.get("from_agent", "unknown") if isinstance(result, dict) else "unknown"
                msg["_origin"] = f"agent:{from_agent}"
            self._chat_messages.append(msg)
            tool_outputs.append({
                "tool": tool_name,
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
            # Mark the response object so downstream chat() can
            # distinguish a deliberately-silent reply from a model that
            # produced no text accidentally. Surfaces via the
            # ``llm_response`` attribute (a Pydantic model) — we attach
            # the flag without mutating the schema by reading it back
            # from a sentinel attribute name.
            try:
                llm_response.__dict__["__silent_reply__"] = True
            except (AttributeError, TypeError):
                pass
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

    async def _retry_empty_compose(self, system: str) -> tuple[str, int, bool]:
        """Bug 3 — single idempotent retry of the final compose call.

        When a chat turn's final text comes back empty (but it was NOT a
        deliberate ``__SILENT__`` reply and NOT an auth/config/exception
        error turn), re-ask the model ONCE with ``tools=None`` so it
        cannot emit more tool calls and must produce prose. This is a
        side-effect-free recovery: the messages are unchanged, only a
        fresh completion is requested. ``_llm_call_with_retry`` still
        wraps transient transport errors; this adds exactly ONE extra
        compose attempt on top, never a loop.

        Returns the tri-state ``(content, tokens_used, deliberate_silence)``:

        - ``content`` is the resolved (non-silent) text, or ``""`` when the
          retry came back empty / blank, or when it was a deliberate
          ``__SILENT__`` reply.
        - ``tokens_used`` is the retry's token cost (folded into the turn
          total by the caller).
        - ``deliberate_silence`` is ``True`` ONLY when the retry itself
          emitted ``__SILENT__`` — in that case the tuple is
          ``("", tokens, True)`` and the caller MUST keep the turn silent
          (empty reply, NO marker). This is the bit that distinguishes a
          deliberately-silent retry from an empty/blip retry: without it
          the caller can't tell them apart and would paper a deliberate
          ``__SILENT__`` over with a synthetic marker.

        ``_resolve_content`` is what SETS the ``__silent_reply__`` flag, so
        it must run BEFORE the flag is read — hence the resolve-then-check
        ordering below.

        Typed credential errors (``LLMAuthError`` / ``LLMConfigError``) are
        RE-RAISED so the ``auth_failure`` / ``config_error`` taxonomy is
        preserved — both call sites run inside a ``try`` whose typed arms
        surface those flags (non-stream: ``auth_failure`` / ``config_error``
        result dicts; streaming: the matching ``done`` events). Only
        transient / transport errors degrade to ``("", 0, False)``. Callers
        add the recovered text to ``self._chat_messages`` themselves so the
        retry stays a pure read here, and fold ``tokens_used`` into the turn
        total.
        """
        try:
            llm_response = await _llm_call_with_retry(
                self.llm.chat_collect,
                system=system,
                messages=self._chat_messages,
                tools=None,
            )
        except (LLMAuthError, LLMConfigError):
            raise  # preserve the auth_failure/config_error taxonomy
        except Exception as e:
            logger.warning("Bug 3 empty-compose retry failed: %s", e)
            return "", 0, False
        tokens = llm_response.tokens_used
        # Resolve FIRST — ``_resolve_content`` is what sets the
        # ``__silent_reply__`` flag when the model emits ``__SILENT__``.
        content = self._resolve_content(llm_response)
        # Deliberate silence on the retry is honoured — do NOT paper over
        # a model that explicitly chose ``__SILENT__`` the second time.
        if llm_response.__dict__.get("__silent_reply__"):
            return "", tokens, True
        return content, tokens, False

    # ── Non-streaming chat ────────────────────────────────────

    async def _chat_inner(self, user_message: str) -> dict:
        self.state = "working"
        total_tokens = 0
        tool_outputs: list[dict] = []
        # Fix B — turn_id is generated once per user turn so the partial
        # assistant entry (written before tool dispatch on a tool-calling
        # round) and the final assistant entry (written by _log_chat_turn
        # at turn close) share an identity. ``load_chat_transcript``
        # dedupes by turn_id, keeping the latest, so a dashboard refresh
        # during a long-running tool call sees the in-flight partial and
        # the eventual completion overwrites it cleanly.
        turn_id = str(uuid.uuid4())
        # PE review follow-up — accumulate tool names and assistant text
        # across multi-round turns so the in-flight partial bubble shows
        # ALL tools used so far (not just the current round) and the
        # full assistant prose emitted to date. ``llm_response.content``
        # is per-round, not cumulative — joining with newlines yields
        # the full turn text.
        turn_tool_names: list[str] = []
        turn_content_parts: list[str] = []

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

            # RC-1 / RC-3: per-task convergence budget. Read the durable
            # task_id from the contextvar set by chat() — ``None`` for
            # interactive operator chat / heartbeats, which must be
            # COMPLETELY unaffected (no cap, no nudge).
            from src.shared.trace import current_task_id
            _task_id = current_task_id.get()
            # Pre-round-boundary redesign (Codex r3): the loop runs over the
            # NORMAL interactive bound (CHAT_MAX_TOOL_ROUNDS) exactly as it did
            # before the convergence feature. The per-task cap is enforced as a
            # PRE-ROUND boundary at the TOP of each iteration (below), BEFORE
            # the LLM call — it is the ONLY place ``task_convergence_capped`` is
            # set for tasks. This is structurally immune to the two Codex
            # blockers:
            #   #1 (terminal tool on last budgeted round overridden): a
            #      terminal complete_task / hand_off / final-text in a round
            #      returns via the normal no-tool-calls path BELOW *before* the
            #      next round's top-of-loop cap check can run, so the cap can
            #      never pre-empt a real completion.
            #   #2 (exhausted re-wake gets another tool round): an exhausted
            #      re-wake (count already >= cap) breaks at the FIRST round's
            #      top, before any LLM/tool call — no extra tool round is ever
            #      granted.
            _task_convergence_capped = False

            steer_interrupts = 0
            for _round_idx in range(self.CHAT_MAX_TOOL_ROUNDS):
                self._bump_liveness()
                # ── Per-task convergence cap: PRE-ROUND boundary ──
                # The ONLY place the cap flag is set for tasks. Checked at the
                # very top of the round, BEFORE the LLM call. If this task has
                # already consumed its per-task budget across all wakes, stop
                # here and let chat() close it ``blocked``. Because a terminal
                # tool / final reply in the PRIOR round already returned via the
                # normal path below, this check can never override a real
                # completion; and an exhausted re-wake breaks immediately with
                # NO additional tool round.
                if _task_id and (
                    self._task_round_counts.get(_task_id, 0)
                    >= self.TASK_MAX_TOOL_ROUNDS
                ):
                    _task_convergence_capped = True
                    break
                # RC-1 convergence forcing function (task path only).
                #
                # Codex finding #1: the nudge is appended to THIS round's
                # SYSTEM PROMPT — never injected as a ``user`` chat message.
                # Appending a synthetic user message after a ``tool`` result
                # (or after another user message) breaks the
                # user→assistant→tool→assistant role-alternation the LLM call
                # requires (Constraint #7). The system prompt is rebuildable
                # per round, so a transient suffix is safe and leaves
                # ``self._chat_messages`` untouched.
                #
                # Codex finding #2: TOOLS REMAIN AVAILABLE on every round. The
                # agent converges by CALLING complete_task / hand_off, so
                # withholding tools would PREVENT convergence and wrongly force
                # a ``blocked`` close on the deliverable. There is no
                # task-path tool-withhold.
                #
                # ``_task_left`` is the rounds remaining in the WHOLE per-task
                # budget, including prior wakes — so a re-woken near-exhausted
                # task is nudged immediately, not given a fresh window.
                _round_system = system
                if _task_id:
                    _task_left = self.TASK_MAX_TOOL_ROUNDS - self._task_round_counts.get(
                        _task_id, 0,
                    )
                    if _task_left <= 1:
                        _round_system = system + self._TASK_CONVERGENCE_FINAL_SUFFIX
                    elif _task_left <= self._TASK_CONVERGENCE_NUDGE_REMAINING:
                        _round_system = system + self._TASK_CONVERGENCE_SOFT_SUFFIX
                _iter_tools = (
                    self.tools.get_tool_definitions(**self._tool_filter_kw) or None
                )
                llm_response = await _llm_call_with_retry(
                    self.llm.chat_collect,
                    system=_round_system,
                    messages=self._chat_messages,
                    tools=_iter_tools,
                )
                # Bug 1 (codex P2 r2): tick after the LLM call returns —
                # a single deep-research call can run >5 min, and bumping
                # only at iteration head would let the staleness check fire
                # during a perfectly healthy turn.
                self._bump_liveness()
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
                    # Bug 3 — the turn produced no final text. Distinguish
                    # a deliberate ``__SILENT__`` reply (honoured: empty,
                    # no retry, no marker) from an accidental empty turn
                    # (retry compose once, then fall back to a marker).
                    deliberate_silence = bool(
                        llm_response.__dict__.get("__silent_reply__")
                    )
                    marker_substituted = False
                    if not content.strip() and not deliberate_silence:
                        retried, retry_tokens, retry_silent = await self._retry_empty_compose(system)
                        total_tokens += retry_tokens
                        if retried.strip():
                            content = retried
                        elif retry_silent:
                            # Model deliberately chose ``__SILENT__`` on the
                            # retry — honour it (empty reply, no marker).
                            deliberate_silence = True
                        else:
                            content = self._synthesize_empty_chat_fallback(tool_outputs)
                            marker_substituted = True
                    self._chat_messages.append({"role": "assistant", "content": content})
                    self._log_chat_turn(user_message, content, tool_outputs, turn_id=turn_id)
                    if tool_outputs and self.workspace:
                        tool_names = list({t.get("tool") or t.get("name", "?") for t in tool_outputs})
                        self.workspace.append_activity(
                            trigger="chat",
                            summary=truncate(content.replace("\n", " "), 200),
                            tools_used=tool_names,
                            tokens_used=total_tokens,
                            outcome="complete",
                        )
                    self.state = "idle"
                    result_dict: dict = {
                        "response": content,
                        "tool_outputs": tool_outputs,
                        "tokens_used": total_tokens,
                    }
                    # Codex finding #3: reaching this branch means the model
                    # produced a final TEXT reply (no tool calls) of its own
                    # accord — a genuine convergence. It must NOT be tagged
                    # ``task_convergence_capped``; it flows to chat()'s normal
                    # success path and closes ``done`` (or the lazy-completion
                    # guard fires for a ghost reply). The convergence cap is
                    # raised ONLY at the budget-exhausted fall-through below,
                    # where the model kept emitting tool_calls past its budget.
                    if deliberate_silence:
                        # Mark a deliberate ``__SILENT__`` reply so the
                        # chat() empty-response fallback doesn't
                        # paper over it with a synthetic notice.
                        result_dict["silent_reply"] = True
                    elif marker_substituted:
                        # Empty-turn marker substituted above. Flag so the
                        # dashboard can badge "recovered from empty-text
                        # turn"; reuses ``silent_reply`` (the dashboard's
                        # existing "no live bubble to render" signal).
                        result_dict["silent_reply"] = True
                    return result_dict

                # Pre-scan for terminate before appending assistant message
                terminate_msg = self._check_tool_loop_terminate(llm_response.tool_calls)
                if terminate_msg:
                    self.state = "idle"
                    msg = f"Stopped: {terminate_msg}"
                    self._finalize_chat_turn(
                        turn_id=turn_id,
                        accumulated_content="\n".join(turn_content_parts),
                        tool_names=turn_tool_names,
                        closing_message=msg,
                    )
                    return {
                        "response": msg,
                        "tool_outputs": tool_outputs,
                        "tokens_used": total_tokens,
                    }

                entries = self._build_tool_call_entries(llm_response)
                # Fix B — persist a ``partial`` assistant entry BEFORE
                # tool dispatch. When a tool call runs long (e.g.
                # ``await_task_event`` with a 240s timeout) and the user
                # refreshes the dashboard mid-flight, the legacy
                # transcript only carried the user message — the
                # assistant bubble vanished until the turn closed.
                # ``load_chat_transcript`` dedupes by ``turn_id``, so
                # the final entry written by ``_log_chat_turn`` at turn
                # close supersedes this partial cleanly.
                #
                # PE review follow-up — accumulate tool names + content
                # across rounds so a mid-flight refresh shows ALL tools
                # used so far and ALL prose emitted to date, not just
                # the current round. ``llm_response.content`` is the
                # CURRENT round's content only; joining with newlines
                # yields the full turn text.
                if llm_response.content:
                    turn_content_parts.append(llm_response.content)
                for tc in llm_response.tool_calls:
                    if tc.name not in turn_tool_names:
                        turn_tool_names.append(tc.name)
                if self.workspace:
                    self.workspace.append_chat_message(
                        "assistant",
                        "\n".join(turn_content_parts),
                        tool_names=list(turn_tool_names),
                        turn_id=turn_id,
                        partial=True,
                    )
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
                # Bug 1 (codex P2 r2): tick after tool execution — a
                # long-running shell or browser action can sit at the
                # 300s _TOOL_TIMEOUT cap. Without this bump the next
                # iteration head might already be past the staleness
                # threshold even though the loop is healthy.
                self._bump_liveness()
                self._chat_total_rounds += 1
                # RC-3: a tool round just completed for this task. Persist
                # the running per-task count so it survives across wakes —
                # repeated lane followups for the same task_id share one
                # budget instead of each getting a fresh window.
                #
                # Codex #3 (size bound): the dict is normally freed on each
                # task's terminal close (``_auto_close_task`` pops it), so it
                # tracks only in-flight tasks. But sustained mesh-write
                # failures (5xx) keep counts alive indefinitely, so cap the
                # dict size.
                #
                # Item 4 fix (Codex r4): the previous design evicted an
                # ARBITRARY existing entry to make room for a new task. That
                # was unsafe — the evicted entry could belong to a still-working
                # task, which would then regain a FRESH full per-task budget on
                # its next round (the exact failure the cap exists to prevent).
                # New design: never evict a tracked entry. Once at the bound,
                # simply DO NOT begin tracking a brand-new task (log a warning).
                # An untracked task runs without the advisory per-task cap — but
                # that only happens with 256 simultaneously in-flight tasks
                # under sustained mesh-write failure, a far more benign outcome
                # than silently resetting a live task's budget. A task already
                # in the dict always keeps incrementing its own count.
                if _task_id:
                    if (
                        _task_id not in self._task_round_counts
                        and len(self._task_round_counts)
                        >= self._TASK_ROUND_COUNTS_MAX
                    ):
                        logger.warning(
                            "task convergence count table at bound (%d) — not "
                            "tracking new task=%s; it runs without the advisory "
                            "per-task cap until an in-flight task closes",
                            self._TASK_ROUND_COUNTS_MAX, _task_id,
                        )
                    else:
                        self._task_round_counts[_task_id] = (
                            self._task_round_counts.get(_task_id, 0) + 1
                        )

                # Item 1 fix (Codex r4) — completion wins over the convergence
                # cap. If THIS round landed a SUCCESSFUL terminal coordination
                # tool (complete_task → completed / hand_off → handed_off), the
                # task genuinely converged. Return the NORMAL (non-capped)
                # result NOW so chat()'s done/handoff close path runs — instead
                # of looping to the next round's top-of-loop cap check, which
                # (when this round pushed the count to TASK_MAX) would break and
                # mislabel a real completion as ``blocked``. Gated on
                # ``_task_id`` so interactive chat is untouched. A non-terminal
                # outbound (notify_user, write_blackboard, …) does NOT trip this
                # — those still fall through to the cap, so a task that hit the
                # cap doing non-terminal work is still closed ``blocked``.
                if _task_id and AgentLoop._last_round_terminal_completion(
                    tool_outputs
                ):
                    logger.info(
                        "chat task=%s landed a terminal coordination tool this "
                        "round — returning to let chat() close done/handoff "
                        "(convergence cap does not pre-empt a real completion)",
                        _task_id,
                    )
                    self.state = "idle"
                    self._chat_messages.append(
                        {"role": "assistant", "content": ""}
                    )
                    self._log_chat_turn(
                        user_message, "", tool_outputs, turn_id=turn_id,
                    )
                    return {
                        "response": "",
                        "tool_outputs": tool_outputs,
                        "tokens_used": total_tokens,
                        # Intentionally NOT ``task_convergence_capped``: this is
                        # a genuine completion, so chat() runs its normal close.
                        # ``silent_reply`` suppresses the empty-response marker
                        # (the task closes done/handoff, no chat bubble wanted).
                        "silent_reply": True,
                    }

                # Rebuild system prompt if operator playbook state changed
                if self._is_operator:
                    new_active = self._update_operator_playbooks()
                    old_active = self._last_active_playbooks
                    if set(new_active) != set(old_active):
                        self._last_active_playbooks = new_active
                        system = self._build_chat_system_prompt(
                            goals=self._goals_cache if self._goals_cache is not self._GOALS_NOT_FETCHED else None,
                            fleet_roster=self._fleet_roster,
                            introspect_data=self._introspect_cache,
                        )

                # If tools were hot-reloaded during tool execution,
                # rebuild the system prompt so tool descriptions stay in sync.
                if self._tools_reloaded:
                    self._tools_reloaded = False
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

            # Per-task convergence cap hit (top-of-round break). Return
            # immediately WITHOUT a force-compose LLM call: the task never
            # converged, so chat() will close it ``blocked`` with the static
            # convergence note. Returning here is what makes an exhausted
            # re-wake (Codex blocker #2) consume ZERO LLM calls — it breaks at
            # the first round top and returns straight away. A genuine
            # completion (terminal tool / final text) already returned via the
            # no-tool-calls path inside the loop, so it never reaches here.
            if _task_convergence_capped:
                self.state = "idle"
                if tool_outputs and self.workspace:
                    tool_names = list(
                        {t.get("tool") or t.get("name", "?") for t in tool_outputs}
                    )
                    self.workspace.append_activity(
                        trigger="chat",
                        summary="convergence cap reached",
                        tools_used=tool_names,
                        tokens_used=total_tokens,
                        outcome="convergence_capped",
                    )
                return {
                    "response": "",
                    "tool_outputs": tool_outputs,
                    "tokens_used": total_tokens,
                    "task_convergence_capped": True,
                    # ``silent_reply`` suppresses chat()'s empty-response
                    # final-net marker: the response is intentionally empty
                    # (the task is being closed ``blocked``, not surfaced as a
                    # chat reply), so no synthetic "no text" notice is wanted.
                    "silent_reply": True,
                }

            # Max tool rounds exhausted — force final text response.
            # Omit tools so the LLM cannot return more tool calls.
            llm_response = await _llm_call_with_retry(
                self.llm.chat_collect,
                system=system,
                messages=self._chat_messages,
                tools=None,
            )
            total_tokens += llm_response.tokens_used
            content = self._resolve_content(llm_response)
            # Bug 3 — even the force-compose exit can come back empty. The
            # call above already withheld tools, so it IS the compose
            # retry: do NOT stack another ``_retry_empty_compose`` on top
            # (that would double-retry the identical request). Honour a
            # deliberate ``__SILENT__`` reply; otherwise substitute the
            # marker so the user never sees a blank turn.
            deliberate_silence = bool(llm_response.__dict__.get("__silent_reply__"))
            marker_substituted = False
            if not content.strip() and not deliberate_silence:
                content = self._synthesize_empty_chat_fallback(tool_outputs)
                marker_substituted = True
            self._chat_messages.append({"role": "assistant", "content": content})
            self._log_chat_turn(user_message, content, tool_outputs, turn_id=turn_id)
            if tool_outputs and self.workspace:
                tool_names = list({t.get("tool") or t.get("name", "?") for t in tool_outputs})
                self.workspace.append_activity(
                    trigger="chat",
                    summary=truncate(content.replace("\n", " "), 200),
                    tools_used=tool_names,
                    tokens_used=total_tokens,
                    outcome="tool_limit_reached",
                )
            self.state = "idle"
            tool_limit_result: dict = {
                "response": content,
                "tool_outputs": tool_outputs,
                "tokens_used": total_tokens,
                "tool_limit_reached": True,
            }
            # Pre-round-boundary redesign (Codex r3): this fall-through keeps
            # ONLY its original pre-feature ``tool_limit_reached`` behaviour —
            # reached when the interactive CHAT_MAX_TOOL_ROUNDS bound is hit
            # while still emitting tool_calls. It is deliberately NOT tagged
            # ``task_convergence_capped``: the convergence cap is set
            # EXCLUSIVELY by the top-of-round break above (which returns early,
            # before this block), and that break always fires FIRST for a task
            # since TASK_MAX_TOOL_ROUNDS <= CHAT_MAX_TOOL_ROUNDS. A task can
            # therefore only leave the loop via the cap break (→ early return)
            # or a terminal/final return inside the loop — never here.
            if deliberate_silence or marker_substituted:
                tool_limit_result["silent_reply"] = True
            return tool_limit_result

        except asyncio.CancelledError:
            self.state = "idle"
            raise
        except LLMAuthError as e:
            # Chat path hit a credential failure. Mesh-side recording is
            # authoritative (CredentialVault records before tagging the
            # response) — agent does NOT self-report to avoid double-
            # counting against the quarantine threshold.
            self.state = "idle"
            logger.warning(f"Chat auth failure: {e}")
            msg = f"Auth failure: {e}"
            self._finalize_chat_turn(
                turn_id=turn_id,
                accumulated_content="\n".join(turn_content_parts),
                tool_names=turn_tool_names,
                closing_message=msg,
            )
            return {
                "response": msg, "tool_outputs": tool_outputs,
                "tokens_used": total_tokens, "auth_failure": True,
            }
        except LLMConfigError as e:
            self.state = "idle"
            logger.warning(f"Chat config error: {e}")
            msg = f"Config error: {e}"
            self._finalize_chat_turn(
                turn_id=turn_id,
                accumulated_content="\n".join(turn_content_parts),
                tool_names=turn_tool_names,
                closing_message=msg,
            )
            return {
                "response": msg, "tool_outputs": tool_outputs,
                "tokens_used": total_tokens, "config_error": True,
            }
        except Exception as e:
            self.state = "idle"
            logger.error(f"Chat failed: {e}", exc_info=True)
            # Some exceptions stringify to empty, which produced the useless
            # "exception: Error:" blocker notes seen in production. Fall back
            # to the exception class name so the reason is always meaningful.
            msg = f"Error: {e}" if str(e).strip() else f"Error: {type(e).__name__}"
            self._finalize_chat_turn(
                turn_id=turn_id,
                accumulated_content="\n".join(turn_content_parts),
                tool_names=turn_tool_names,
                closing_message=msg,
            )
            # Codex r10: tag the bare-exception return with
            # ``exception_caught`` so the chat() auto-close site can
            # detect failures-after-tool-calls and route them to
            # ``failed`` instead of ``done``. Without this marker, any
            # exception that fires AFTER at least one tool dispatch
            # would surface as a normal result dict with non-empty
            # tool_outputs, slip past the lazy-completion guard, and
            # auto-close the handoff task as successful.
            return {
                "response": msg, "tool_outputs": tool_outputs,
                "tokens_used": total_tokens, "exception_caught": True,
            }

    @staticmethod
    def _synthesize_empty_chat_fallback(
        tool_outputs: list[dict] | None,
    ) -> str:
        """Build the synthetic notice for a chat turn that emitted no
        final text. Single source of truth so the dashboard chat panel
        (live `done` event) and the persisted transcript (`/chat/history`
        after refresh) carry identical text — without this helper the two
        diverged and the operator's reply appeared to vanish on page
        refresh.

        Bug 3 — NEVER returns ``""``. With tools, the notice points the
        user at the dashboard for the tool results. With ZERO tools, an
        empty final text is almost always a transient model hiccup (the
        retry-compose path already exhausted its one retry by the time we
        get here), so we surface a generic re-send prompt rather than a
        blank bubble. Deliberate ``__SILENT__`` replies are filtered out
        by the callers BEFORE reaching this helper, so a generic marker
        here can never paper over an intentional silence.
        """
        if not tool_outputs:
            return (
                "(No response was generated for this turn — likely a "
                "transient model issue. Please re-send your message.)"
            )
        n = len(tool_outputs)
        word = "tool call" if n == 1 else "tool calls"
        return (
            f"(Completed {n} {word}; no text response was generated. "
            "Check the dashboard for tool outputs and any notifications "
            "I sent.)"
        )

    def _finalize_chat_turn(
        self,
        *,
        turn_id: str,
        accumulated_content: str,
        tool_names: list[str],
        closing_message: str,
    ) -> None:
        """Write a final assistant transcript entry for an abnormal turn close.

        Used by exception handlers and tool-loop terminate paths. Sharing
        the in-flight partial's ``turn_id`` ensures
        :meth:`WorkspaceManager.load_chat_transcript`'s dedupe replaces
        the partial cleanly — the user sees ONE bubble carrying every
        token streamed so far + the closing message (e.g. ``Error: ...``
        or ``Stopped: ...``), instead of an orphaned ``partial=True``
        entry next to a separate error bubble.

        Empty ``accumulated_content`` is fine — the closing_message
        stands alone (no leading blank lines). No-op when the agent has
        no workspace mounted.

        ``turn_id`` is required and must be non-empty — without it the
        transcript dedupe cannot match the partial and the helper
        degrades to writing an orphaned final entry, defeating its
        purpose. ``_chat_inner`` / ``_chat_stream_inner`` both generate
        ``turn_id = str(uuid.uuid4())`` at the top of the function
        before any work that can raise, so the contract is structurally
        safe.
        """
        if not turn_id:
            raise ValueError(
                "_finalize_chat_turn requires a non-empty turn_id — "
                "callers must mint one before the try block so the "
                "partial dedupe is well-defined"
            )
        if not self.workspace:
            return
        prefix = (accumulated_content or "").strip()
        final_content = (
            f"{prefix}\n\n{closing_message}" if prefix else closing_message
        )
        self.workspace.append_chat_message(
            "assistant",
            final_content,
            tool_names=list(tool_names) if tool_names else None,
            turn_id=turn_id,
        )

    def _log_chat_turn(
        self, user_msg: str, assistant_msg: str,
        tool_outputs: list[dict] | None = None,
        *,
        turn_id: str | None = None,
    ) -> None:
        """Append a rich summary of the chat turn to the daily log.

        ``turn_id`` (Fix B) — when set, propagates to the persistent
        transcript ``append_chat_message`` call so this FINAL assistant
        entry supersedes any earlier ``partial`` entry written before
        tool dispatch. Legacy callers without a turn_id are unchanged.
        """
        if not self.workspace:
            return
        # Empty final text + tool calls → substitute the synthetic
        # fallback so the transcript reflects what the dashboard saw
        # for this turn. Without this the assistant entry was skipped
        # entirely and a page refresh lost the turn's persistence.
        #
        # Bug 3 — only substitute when there ARE tool_outputs. The
        # ``_chat_inner`` / ``_chat_stream_inner`` retry-then-marker paths
        # already pass the resolved text (recovered prose OR the marker)
        # as ``assistant_msg``, so a genuinely empty ``assistant_msg``
        # reaching here means a deliberate ``__SILENT__`` reply (no tools)
        # — which must stay unlogged. Substituting the (now never-empty)
        # generic no-tools marker here would wrongly persist silent turns.
        if not assistant_msg.strip() and tool_outputs:
            assistant_msg = self._synthesize_empty_chat_fallback(tool_outputs)
        # Skip logging only for genuinely silent responses (SILENT
        # token, no tools called — there is nothing to persist).
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

        # Build rich tool data for transcript persistence (truncated previews)
        tools: list[dict] | None = None
        if tool_outputs:
            tools = []
            for t in tool_outputs:
                name = t.get("tool") or t.get("name", "?")
                inp = t.get("input")
                out = t.get("output")
                tools.append({
                    "name": name,
                    "status": "done",
                    "inputPreview": truncate(
                        dumps_safe(inp), 200,
                    ) if inp else "",
                    "outputPreview": truncate(
                        dumps_safe(out), 200,
                    ) if out else "",
                })

        # Persist assistant response to transcript. ``turn_id`` (when
        # threaded through from chat()/chat_stream()) lets this FINAL
        # entry supersede a ``partial`` entry written before tool
        # dispatch — see ``WorkspaceManager.load_chat_transcript`` for
        # the dedup semantics.
        self.workspace.append_chat_message(
            "assistant", assistant_msg,
            tools=tools,
            tool_names=tool_names or None,
            turn_id=turn_id,
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
            # Clear operator playbook state for fresh session
            self._operator_playbook_state = {}
            self._last_active_playbooks = []
            self._operator_playbook_scan_idx = 0
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
            "browser_navigate" in self.tools.tools
            and (
                (self._allowed_tools is not None and "browser_navigate" in self._allowed_tools)
                or (
                    self._allowed_tools is None
                    and (not self._excluded_tools or "browser_navigate" not in self._excluded_tools)
                )
            )
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
            "- For HANDOFF tasks (dispatched via lane → /chat with an "
            "x-task-id header): either call tools to do the work, OR "
            "return your result as a structured final answer — including "
            "when your deliverable is simply an answer or decision: "
            "{\"result\": {\"status\": \"done\", \"summary\": \"<your "
            "answer here>\"}} (use status \"noop\"/\"impossible\" when "
            "there is genuinely nothing to do). A plain-text reply with no "
            "tool call and no {\"result\": {...}} envelope auto-closes the "
            "task as failed (no_outbound_effects).\n"
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

        # Inject operator playbooks based on tool-call patterns
        if self._is_operator:
            active_playbooks = self._update_operator_playbooks()
            if active_playbooks:
                from src.shared.operator_playbooks import get_playbook_content

                playbook_text = get_playbook_content(active_playbooks)
                if playbook_text:
                    parts.append(playbook_text)

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

        index_text = self._refresh_grouped_plan()
        if index_text:
            parts.append(index_text)

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
            capabilities=self.tools.list_tools(**self._tool_filter_kw),
            uptime_seconds=time.time() - self._start_time,
            tasks_completed=self.tasks_completed,
            tasks_failed=self.tasks_failed,
            context_tokens=ctx_tokens,
            context_max=ctx_max,
            context_pct=ctx_pct,
            last_iteration_ts=self._last_iteration_ts,
            iterations_since_boot=self._iterations_since_boot,
        )

    # ── Streaming chat ────────────────────────────────────────

    async def chat_stream(
        self, user_message: str, *, trace_id: str | None = None,
        origin: "MessageOrigin | None" = None,
    ):
        """Streaming chat that yields SSE events as they happen.

        Events yielded (as dicts, caller serialises to SSE):
          {"type": "tool_start", "name": str, "input": dict}
          {"type": "tool_result", "name": str, "output": any}
          {"type": "text_delta", "content": str}
          {"type": "done", "response": str, "tool_outputs": list, "tokens_used": int}
        """
        # Redirect to steer queue if a task is running.
        if self.current_task is not None:
            await self._steer_queue.put(user_message)
            yield {
                "type": "text_delta",
                "content": (
                    "Agent is working on a task. Your message has been queued "
                    "and will be included in the next conversation turn."
                ),
            }
            yield {
                "type": "done",
                "response": "",
                "tool_outputs": [],
                "tokens_used": 0,
            }
            return

        from src.shared.trace import current_origin, current_trace_id
        current_trace_id.set(trace_id)
        origin_token = current_origin.set(origin)
        try:
            async with self._chat_lock:
                await self._maybe_restore_session()
                try:
                    async for event in self._chat_stream_inner(user_message):
                        yield event
                finally:
                    await self._checkpoint_chat_session()
        finally:
            current_origin.reset(origin_token)

    async def _chat_stream_inner(self, user_message: str):
        self.state = "working"
        total_tokens = 0
        tool_outputs: list[dict] = []
        accumulated_text: list[str] = []  # all text_delta content across rounds
        # Fix B — see ``_chat_inner`` for the partial-persistence rationale.
        turn_id = str(uuid.uuid4())
        # PE review follow-up — accumulate tool names across rounds so a
        # mid-flight refresh shows every tool used so far (not just the
        # current round's). Streaming content reuses ``accumulated_text``.
        turn_tool_names: list[str] = []

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
                self._bump_liveness()
                # Try token-level streaming, fall back to non-streaming on error
                llm_response = None
                used_streaming = False
                any_text_streamed = False
                tools = self.tools.get_tool_definitions(**self._tool_filter_kw) or None
                try:
                    async for event in self.llm.chat_stream(
                        system=system, messages=self._chat_messages, tools=tools,
                    ):
                        etype = event.get("type", "")
                        if etype == "text_delta":
                            any_text_streamed = True
                            accumulated_text.append(event.get("content", ""))
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

                # Bug 1 (codex P2 r2): tick after the LLM call returns —
                # a single deep-research call can run >5 min, and bumping
                # only at iteration head would let the staleness check fire
                # during a perfectly healthy turn.
                self._bump_liveness()
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
                    # Use accumulated text from all rounds for the transcript
                    # so intermediate messages are preserved after refresh.
                    full_content = "".join(accumulated_text) if accumulated_text else content
                    # Bug 3 — empty final text and NOT a deliberate
                    # ``__SILENT__`` reply: retry the compose once (tools
                    # withheld), then fall back to a marker. The recovered
                    # text or the marker MUST be emitted as a text_delta
                    # so the user actually sees it (not merely placed in
                    # the ``done`` envelope). Applies regardless of
                    # whether tool_outputs is empty — the marker is
                    # generic in the no-tools case.
                    deliberate_silence = bool(
                        llm_response.__dict__.get("__silent_reply__")
                    )
                    if not full_content.strip() and not deliberate_silence:
                        retried, retry_tokens, retry_silent = await self._retry_empty_compose(system)
                        total_tokens += retry_tokens
                        if retried.strip():
                            full_content = retried
                        elif retry_silent:
                            # Model deliberately chose ``__SILENT__`` on the
                            # retry — honour it: keep the turn silent, emit
                            # no text_delta, leave full_content empty.
                            full_content = ""
                        else:
                            full_content = self._synthesize_empty_chat_fallback(tool_outputs)
                        if full_content:
                            # The retried text / marker was never streamed (the
                            # compose ran non-streaming) — sync it into the
                            # in-memory assistant message (matches the
                            # non-stream path) and emit it now.
                            self._chat_messages[-1]["content"] = full_content
                            yield {"type": "text_delta", "content": full_content}
                    self._log_chat_turn(
                        user_message, full_content, tool_outputs,
                        turn_id=turn_id,
                    )
                    if tool_outputs and self.workspace:
                        tool_names = list({t.get("tool") or t.get("name", "?") for t in tool_outputs})
                        self.workspace.append_activity(
                            trigger="chat",
                            summary=truncate(content.replace("\n", " "), 200),
                            tools_used=tool_names,
                            tokens_used=total_tokens,
                            outcome="complete",
                        )
                    yield {
                        "type": "done",
                        "response": full_content,
                        "tool_outputs": tool_outputs,
                        "tokens_used": total_tokens,
                    }
                    return

                # Pre-scan for terminate before appending assistant message
                terminate_msg = self._check_tool_loop_terminate(llm_response.tool_calls)
                if terminate_msg:
                    msg = f"Stopped: {terminate_msg}"
                    self._finalize_chat_turn(
                        turn_id=turn_id,
                        accumulated_content="".join(accumulated_text),
                        tool_names=turn_tool_names,
                        closing_message=msg,
                    )
                    yield {
                        "type": "done",
                        "response": msg,
                        "tool_outputs": tool_outputs,
                        "tokens_used": total_tokens,
                    }
                    return

                entries = self._build_tool_call_entries(llm_response)
                # Fix B — persist a ``partial`` assistant entry BEFORE
                # tool dispatch so a dashboard refresh mid-flight sees
                # the in-flight bubble. ``_log_chat_turn`` writes the
                # final at turn close with the same ``turn_id`` and
                # supersedes this entry on the next ``load_chat_transcript``.
                #
                # PE review follow-up — accumulate tool names across
                # rounds, and reuse ``accumulated_text`` for the body so
                # all prose streamed so far is visible after a refresh.
                for tc in llm_response.tool_calls:
                    if tc.name not in turn_tool_names:
                        turn_tool_names.append(tc.name)
                if self.workspace:
                    # Fall back to ``llm_response.content`` when no
                    # text_delta events streamed — some providers return
                    # content as a single block instead of streaming
                    # deltas, leaving ``accumulated_text`` empty even
                    # though the assistant produced prose. Without this
                    # fallback the partial entry is empty, which renders
                    # as an empty bubble on mid-flight refresh.
                    partial_content = (
                        "".join(accumulated_text) if accumulated_text
                        else (llm_response.content or "")
                    )
                    self.workspace.append_chat_message(
                        "assistant",
                        partial_content,
                        tool_names=list(turn_tool_names),
                        turn_id=turn_id,
                        partial=True,
                    )
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
                # Bug 1 (codex P2 r2): tick after tool execution — a
                # long-running shell or browser action can sit at the
                # 300s _TOOL_TIMEOUT cap. Without this bump the next
                # iteration head might already be past the staleness
                # threshold even though the loop is healthy.
                self._bump_liveness()
                self._chat_total_rounds += 1

                # Rebuild system prompt if operator playbook state changed
                if self._is_operator:
                    new_active = self._update_operator_playbooks()
                    old_active = self._last_active_playbooks
                    if set(new_active) != set(old_active):
                        self._last_active_playbooks = new_active
                        system = self._build_chat_system_prompt(
                            goals=self._goals_cache if self._goals_cache is not self._GOALS_NOT_FETCHED else None,
                            fleet_roster=self._fleet_roster,
                            introspect_data=self._introspect_cache,
                        )

                # Rebuild system prompt after tool hot-reload
                if self._tools_reloaded:
                    self._tools_reloaded = False
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
                accumulated_text.append(content)
                yield {"type": "text_delta", "content": content}
            self._chat_messages.append({"role": "assistant", "content": content})
            full_content = "".join(accumulated_text) if accumulated_text else content
            # Bug 3 — the force-compose call above already withheld tools,
            # so it IS the compose retry; do NOT stack another
            # ``_retry_empty_compose`` (that would double-retry the same
            # request). Honour a deliberate ``__SILENT__`` reply;
            # otherwise substitute the marker and emit it as a text_delta
            # so the user never sees a blank turn. Applies regardless of
            # whether tool_outputs is empty (generic marker in that case).
            deliberate_silence = bool(llm_response.__dict__.get("__silent_reply__"))
            if not full_content.strip() and not deliberate_silence:
                full_content = self._synthesize_empty_chat_fallback(tool_outputs)
                self._chat_messages[-1]["content"] = full_content
                yield {"type": "text_delta", "content": full_content}
            self._log_chat_turn(
                user_message, full_content, tool_outputs, turn_id=turn_id,
            )
            if tool_outputs and self.workspace:
                tool_names = list({t.get("tool") or t.get("name", "?") for t in tool_outputs})
                self.workspace.append_activity(
                    trigger="chat",
                    summary=truncate(content.replace("\n", " "), 200),
                    tools_used=tool_names,
                    tokens_used=total_tokens,
                    outcome="tool_limit_reached",
                )
            yield {
                "type": "done",
                "response": full_content,
                "tool_outputs": tool_outputs,
                "tokens_used": total_tokens,
                "tool_limit_reached": True,
            }

        except asyncio.CancelledError:
            raise
        except LLMAuthError as e:
            # Streaming chat path hit a credential failure. Mesh-side
            # recording is authoritative — agent does NOT self-report
            # to avoid double-counting against the quarantine threshold.
            logger.warning("Streaming chat auth failure: %s", e)
            msg = f"Auth failure: {e}"
            self._finalize_chat_turn(
                turn_id=turn_id,
                accumulated_content="".join(accumulated_text),
                tool_names=turn_tool_names,
                closing_message=msg,
            )
            yield {
                "type": "done",
                "response": msg,
                "tool_outputs": tool_outputs,
                "tokens_used": total_tokens,
                "auth_failure": True,
            }
        except LLMConfigError as e:
            logger.warning("Streaming chat config error: %s", e)
            msg = f"Config error: {e}"
            self._finalize_chat_turn(
                turn_id=turn_id,
                accumulated_content="".join(accumulated_text),
                tool_names=turn_tool_names,
                closing_message=msg,
            )
            yield {
                "type": "done",
                "response": msg,
                "tool_outputs": tool_outputs,
                "tokens_used": total_tokens,
                "config_error": True,
            }
        except Exception as e:
            logger.error("Streaming chat failed: %s", e, exc_info=True)
            msg = f"Error: {e}"
            self._finalize_chat_turn(
                turn_id=turn_id,
                accumulated_content="".join(accumulated_text),
                tool_names=turn_tool_names,
                closing_message=msg,
            )
            yield {
                "type": "done",
                "response": msg,
                "tool_outputs": tool_outputs,
                "tokens_used": total_tokens,
            }
        finally:
            self.state = "idle"
