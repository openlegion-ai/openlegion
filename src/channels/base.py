"""Abstract base for messaging channel adapters.

Each channel bridges an external messaging platform (Telegram, Discord, etc.)
to the OpenLegion mesh. Provides a unified multi-agent chat interface that
mirrors the CLI REPL: per-user active agent, @mentions, /commands, and
agent labels on responses.
"""

from __future__ import annotations

import abc
import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.channels import AT_MENTION_RE
from src.host.credentials import is_system_credential
from src.shared.utils import sanitize_for_prompt, setup_logging, usable_agent_reply

if TYPE_CHECKING:
    from src.shared.types import MessageOrigin

logger = setup_logging("channels.base")


class PairingManager:
    """Shared pairing-code security for channel adapters.

    Manages owner/allowed-user state persisted as a JSON file.
    All channels use the same pairing flow: the owner enters a code
    shown during ``openlegion start``, then can allow/revoke others.
    """

    def __init__(self, config_path: str | Path):
        self._path = Path(config_path)
        self._data = self._load()

    # ── persistence ───────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Pairing config load failed: %s", e)
        return {"owner": None, "allowed": []}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2) + "\n")

    # ── access checks ─────────────────────────────────────────

    @property
    def owner(self):
        return self._data.get("owner")

    @property
    def pairing_code(self) -> str:
        return self._data.get("pairing_code", "")

    def is_allowed(self, user_id) -> bool:
        owner = self.owner
        if owner is None:
            return False
        if user_id == owner:
            return True
        return user_id in self._data.get("allowed", [])

    def is_owner(self, user_id) -> bool:
        return user_id == self.owner

    # ── mutations ─────────────────────────────────────────────

    def claim_owner(self, user_id) -> None:
        self._data["owner"] = user_id
        self._data.setdefault("allowed", [])
        self._data.pop("pairing_code", None)
        self.save()

    def allow(self, user_id) -> bool:
        """Add user to allowed list. Returns True if newly added."""
        allowed = self._data.setdefault("allowed", [])
        if user_id not in allowed:
            allowed.append(user_id)
            self.save()
            return True
        return False

    def revoke(self, user_id) -> bool:
        """Remove user from allowed list. Returns True if was present."""
        allowed = self._data.setdefault("allowed", [])
        if user_id in allowed:
            allowed.remove(user_id)
            self.save()
            return True
        return False

    def allowed_list(self) -> list:
        return list(self._data.get("allowed", []))

DispatchFn = Callable[..., Coroutine[Any, Any, str]]
StreamDispatchFn = Callable[[str, str], AsyncIterator[dict]]
ListAgentsFn = Callable[[], dict]
StatusFn = Callable[[str], dict | None]
CostsFn = Callable[[], list[dict]]
ResetFn = Callable[[str], bool]
AddKeyFn = Callable[[str, str], None]
SteerFn = Callable[[str, str], None]
DebugFn = Callable[[str | None], list[dict]]


class Channel(abc.ABC):
    """Base class for messaging platform adapters.

    Provides unified multi-agent chat identical to the CLI REPL:
    - Per-user active agent tracking
    - @agent mentions for routing
    - /use, /agents, /status, /broadcast, /costs, /reset, /help commands
    - Agent name labels on every response
    - Async notification push for cron/heartbeat results

    Subclasses should set ``CHANNEL_TYPE`` to the platform identifier string
    (e.g. ``"whatsapp"``) used for origin-based response routing.
    """

    CHANNEL_TYPE: str = ""

    def __init__(
        self,
        dispatch_fn: DispatchFn,
        default_agent: str = "",
        list_agents_fn: ListAgentsFn | None = None,
        status_fn: StatusFn | None = None,
        costs_fn: CostsFn | None = None,
        reset_fn: ResetFn | None = None,
        stream_dispatch_fn: StreamDispatchFn | None = None,
        addkey_fn: AddKeyFn | None = None,
        steer_fn: SteerFn | None = None,
        debug_fn: DebugFn | None = None,
    ):
        self.dispatch_fn = dispatch_fn
        self.default_agent = default_agent
        self.list_agents_fn = list_agents_fn
        self.status_fn = status_fn
        self.costs_fn = costs_fn
        self.reset_fn = reset_fn
        self.stream_dispatch_fn = stream_dispatch_fn
        self.addkey_fn = addkey_fn
        self.steer_fn = steer_fn
        self.debug_fn = debug_fn
        self._active_agent: dict[str, str] = {}  # user_id -> agent_name
        self._notify_targets: list[Any] = []

    @abc.abstractmethod
    async def start(self) -> None:
        """Start receiving messages from the platform."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the channel."""

    @abc.abstractmethod
    async def send_notification(self, text: str) -> None:
        """Push a notification (e.g. cron result) to all registered users."""

    async def send_to_user(self, user_id: str, text: str) -> None:
        """Send a message to a specific user on this channel.

        Subclasses should override to deliver via the platform's API.
        Base implementation logs and drops — prevents crashes when a
        channel that hasn't implemented it receives an auto-notify.
        """
        logger.warning(
            "%s.send_to_user not implemented; dropping message to %s",
            type(self).__name__, user_id,
        )

    def _resolve_owner(self, user_id: str) -> bool:
        """Whether ``user_id`` is the channel owner.

        Base default is ``False`` (fail-closed): a channel with no owner
        concept grants no privileged access.  Subclasses with a
        ``PairingManager`` override this to consult ``_is_owner`` and to
        normalize their platform-specific user-key format (e.g. Slack's
        ``user_id:thread_ts`` composite) before the lookup.
        """
        return False

    def _get_active_agent(self, user_id: str) -> str:
        return self._active_agent.get(user_id, self.default_agent)

    def _set_active_agent(self, user_id: str, agent: str) -> None:
        self._active_agent[user_id] = agent

    def _get_agent_names(self) -> list[str]:
        if self.list_agents_fn:
            return list(self.list_agents_fn().keys())
        return []

    async def dispatch(
        self, agent: str, message: str,
        origin: "MessageOrigin | None" = None,
    ) -> str:
        """Route a message to an agent and return the response.

        ``dispatch_fn`` must accept ``origin`` as a keyword argument (defaulting
        to None). Production dispatch_fns (``RuntimeContext.async_dispatch``)
        already do; test stubs should use ``**kwargs`` or an explicit
        ``origin=None`` kwarg.
        """
        from src.shared.trace import current_trace_id, new_trace_id

        target = agent or self.default_agent
        if not target:
            return "No agent specified and no default agent configured."
        current_trace_id.set(new_trace_id())
        try:
            return await self.dispatch_fn(target, message, origin=origin)
        except Exception as e:
            logger.error(f"Dispatch to '{target}' failed: {e}")
            return f"Error: {e}"

    async def handle_message(self, user_id: str, text: str) -> str:
        """Process a user message with full REPL-like command support.

        Returns the formatted response to send back to the user.
        """
        text = text.strip()
        if not text:
            return ""

        text = sanitize_for_prompt(text)

        current = self._get_active_agent(user_id)
        agents = self._get_agent_names()

        # @agent mention routing
        target = current
        message = text
        match = AT_MENTION_RE.match(text)
        if match:
            mentioned = match.group(1)
            if mentioned in agents:
                target = mentioned
                message = match.group(2)
            else:
                return f"Unknown agent: '{mentioned}'. Use /agents to list."

        # Slash commands
        if message.startswith("/"):
            is_owner = self._resolve_owner(user_id)
            return await self._handle_command(
                user_id, message, current, agents, is_owner,
            )

        # Normal message: build origin and dispatch to agent.
        # Task 2b: stamp typed ``MessageOrigin`` (kind="human") so the
        # downstream lane / agent sees an authorization-bearing origin
        # rather than a raw dict.
        from src.shared.types import MessageOrigin

        origin: MessageOrigin | None = None
        if self.CHANNEL_TYPE and user_id:
            origin = MessageOrigin(
                kind="human",
                channel=self.CHANNEL_TYPE,
                user=str(user_id),
            )
        response = await self.dispatch(target, message, origin=origin)
        # Plan §8 #24 recon minor item: a bare ``response.strip()`` truthy
        # check let the ``__SILENT__`` sentinel (a stopped/unresponsive
        # agent) and the "(no response)"/``dispatch_error:`` lane-dispatch
        # shapes through as if they were real agent text — a channel user
        # would see the literal token. ``usable_agent_reply`` is the shared
        # gate every other consumer of these non-success shapes uses.
        if not usable_agent_reply(response):
            return ""  # Suppress silent/unusable responses
        return f"[{target}] {response}"

    #: Commands gated to the channel owner only. Non-owner allowed users
    #: keep chat + read-only commands (/use, /agents, /status, /costs, /help)
    #: but are refused these privileged, state-mutating / sensitive ones.
    _OWNER_ONLY_COMMANDS = frozenset({
        "/addkey", "/steer", "/broadcast", "/reset", "/debug",
    })

    #: Deterministic command → agent routing. When a command matches a key
    #: in this map, it is dispatched directly to the target agent without
    #: consuming an LLM round-trip. Value is ``(target_agent, task_template)``
    #: where ``task_template`` may contain ``{args}`` for the command tail.
    #:
    #: Empty by default — deployments populate it with their own routes.
    #: Routes are skipped silently if the target agent is not running.
    _BOT_COMMAND_ROUTES: dict[str, tuple[str, str]] = {}

    async def _handle_command(
        self, user_id: str, message: str, current: str, agents: list[str],
        is_owner: bool = False,
    ) -> str:
        parts = message.split(None, 1)
        cmd = parts[0].lower()

        # H2: owner-gate privileged commands. Allowed (non-owner) users are
        # already authenticated for chat + read-only commands, but these
        # commands can mint credentials, inject context, broadcast, reset
        # state, or expose traces — restrict them to the owner.
        if cmd in self._OWNER_ONLY_COMMANDS and not is_owner:
            return f"'{cmd}' is owner only."

        # Deterministic bot-command routing — bypasses LLM for reliability.
        if cmd in self._BOT_COMMAND_ROUTES:
            target_agent, task_tmpl = self._BOT_COMMAND_ROUTES[cmd]
            if target_agent in agents:
                args = parts[1].strip() if len(parts) > 1 else ""
                task = task_tmpl.format(args=args) if "{args}" in task_tmpl else task_tmpl
                from src.shared.types import MessageOrigin

                origin: MessageOrigin | None = None
                if self.CHANNEL_TYPE and user_id:
                    origin = MessageOrigin(
                        kind="human",
                        channel=self.CHANNEL_TYPE,
                        user=str(user_id),
                    )
                response = await self.dispatch(target_agent, task, origin=origin)
                if not response or not response.strip():
                    return ""
                return f"[{target_agent}] {response}"

        if cmd == "/use":
            if len(parts) < 2:
                return f"Usage: /use <agent>  (current: {current})"
            new_agent = parts[1].strip()
            if new_agent not in agents:
                return f"Unknown agent: '{new_agent}'. Use /agents to list."
            self._set_active_agent(user_id, new_agent)
            return f"Now chatting with '{new_agent}'."

        if cmd == "/agents":
            if not agents:
                return "No agents available."
            lines = [f"  {name}" + (" (active)" if name == current else "") for name in agents]
            return "Agents:\n" + "\n".join(lines)

        if cmd == "/status":
            if not self.status_fn:
                return "Status not available."
            lines = []
            for name in agents:
                info = self.status_fn(name)
                if info:
                    state = info.get("state", "unknown")
                    tasks = info.get("tasks_completed", 0)
                    ctx_max = info.get("context_max", 0)
                    if ctx_max:
                        ctx_pct = int(info.get("context_pct", 0) * 100)
                        lines.append(f"  {name}: {state} ({tasks} tasks, ctx {ctx_pct}%)")
                    else:
                        lines.append(f"  {name}: {state} ({tasks} tasks)")
                else:
                    lines.append(f"  {name}: unreachable")
            return "Agent status:\n" + "\n".join(lines)

        if cmd == "/broadcast":
            if len(parts) < 2:
                return "Usage: /broadcast <message>"
            bc_msg = parts[1]
            responses = await asyncio.gather(
                *(self.dispatch(name, bc_msg) for name in agents),
                return_exceptions=True,
            )
            results = [
                f"[{name}] {resp}" if not isinstance(resp, BaseException)
                else f"[{name}] Error: {resp}"
                for name, resp in zip(agents, responses, strict=True)
            ]
            return f"Broadcast to {len(agents)} agent(s):\n\n" + "\n\n".join(results)

        if cmd == "/costs":
            if not self.costs_fn:
                return "Cost tracking not available."
            try:
                spend = self.costs_fn()
                if not spend:
                    return "No usage recorded today."
                # Filter to active agents only (ignore stale DB entries)
                active = set(agents) if agents else None
                if active:
                    spend = [a for a in spend if a["agent"] in active]
                if not spend:
                    return "No usage recorded today."
                total = sum(a["cost"] for a in spend)
                lines = [f"Today's spend: ${total:.4f}\n"]
                for a in spend:
                    lines.append(f"  {a['agent']:<16} {a['tokens']:>8,} tokens  ${a['cost']:.4f}")
                return "\n".join(lines)
            except Exception as e:
                return f"Error: {e}"

        if cmd == "/reset":
            if not self.reset_fn:
                return "Reset not available."
            self.reset_fn(current)
            return f"Conversation with '{current}' reset."

        if cmd == "/addkey":
            if not self.addkey_fn:
                return "Credential management not available."
            args = message.split(None, 2)
            if len(args) < 2:
                return "Usage: /addkey <service> <key>"
            service = args[1]
            # Normalize bare provider names to include _api_key suffix.
            # Local import: importing SYSTEM_CREDENTIAL_PROVIDERS at module
            # scope would trigger src.host.credentials' lazy __getattr__
            # at channel-import time, which transitively pulls litellm
            # (~2.25s).  Defer until the user actually runs /addkey.
            from src.host.credentials import SYSTEM_CREDENTIAL_PROVIDERS
            if service.lower() in SYSTEM_CREDENTIAL_PROVIDERS and not service.lower().endswith("_api_key"):
                service = f"{service}_api_key"
            key = args[2] if len(args) > 2 else ""
            if not key:
                return "Usage: /addkey <service> <key>"
            try:
                self.addkey_fn(service, key)
                tier = "system" if is_system_credential(service) else "agent"
                return f"Credential '{service}' stored ({tier} tier)."
            except Exception as e:
                return f"Error storing credential: {e}"

        if cmd == "/steer":
            if not self.steer_fn:
                return "Steer not available."
            if len(parts) < 2:
                return f"Usage: /steer <message>  (injects into {current}'s context)"
            try:
                self.steer_fn(current, parts[1])
                return f"Steered '{current}' with message."
            except Exception as e:
                return f"Error steering: {e}"

        if cmd == "/debug":
            if not self.debug_fn:
                return "Debug not available."
            trace_id = parts[1].strip() if len(parts) > 1 else None
            try:
                traces = self.debug_fn(trace_id)
                if not traces:
                    return "No traces found."
                header = f"Trace {trace_id}:" if trace_id else "Recent traces:"
                lines = [header]
                for t in traces[:10]:
                    tid = t.get("trace_id", "?")[:12]
                    agent_name = t.get("agent", "-")
                    etype = t.get("event_type", "?")
                    lines.append(f"  {tid}  {agent_name:<16} {etype}")
                return "\n".join(lines)
            except Exception as e:
                return f"Error: {e}"

        if cmd == "/help":
            lines = [
                "Commands:",
                "  @agent <msg>      Send message to a specific agent",
                "  /use <agent>      Switch active agent",
                "  /agents           List all agents",
                "  /status           Show agent health",
            ]
            # Owner-only commands surface in help only for the owner so
            # non-owner allowed users aren't told about commands they can't run.
            if is_owner:
                lines.append("  /broadcast <msg>  Send to all agents")
                if self.steer_fn:
                    lines.append("  /steer <msg>      Inject message into busy agent's context")
                if self.debug_fn:
                    lines.append("  /debug [trace_id] Show recent traces or trace detail")
            lines.append("  /costs            Show today's LLM spend")
            if is_owner:
                lines.extend([
                    "  /addkey <svc> <key>  Add an API credential",
                    "  /reset            Clear conversation with active agent",
                ])
            lines.append("  /help             Show this help")
            return "\n".join(lines)

        return f"Unknown command: {cmd}. Type /help for commands."


def chunk_text(text: str, max_len: int) -> list[str]:
    """Split text into chunks respecting a platform's message limit."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks
