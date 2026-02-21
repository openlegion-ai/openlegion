"""Abstract base for messaging channel adapters.

Each channel bridges an external messaging platform (Telegram, Discord, etc.)
to the OpenLegion mesh. Provides a unified multi-agent chat interface that
mirrors the CLI REPL: per-user active agent, @mentions, /commands, and
agent labels on responses.
"""

from __future__ import annotations

import abc
import json
import re
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("channels.base")


class PairingManager:
    """Shared pairing-code security for channel adapters.

    Manages owner/allowed-user state persisted as a JSON file.
    All channels use the same pairing flow: the owner enters a code
    shown during ``openlegion setup``, then can allow/revoke others.
    """

    def __init__(self, config_path: str | Path):
        self._path = Path(config_path)
        self._data = self._load()

    # ── persistence ───────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
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

DispatchFn = Callable[[str, str], Coroutine[Any, Any, str]]
StreamDispatchFn = Callable[[str, str], AsyncIterator[dict]]
ListAgentsFn = Callable[[], dict]
StatusFn = Callable[[str], dict | None]
CostsFn = Callable[[], list[dict]]
ResetFn = Callable[[str], bool]
AddKeyFn = Callable[[str, str], None]


class Channel(abc.ABC):
    """Base class for messaging platform adapters.

    Provides unified multi-agent chat identical to the CLI REPL:
    - Per-user active agent tracking
    - @agent mentions for routing
    - /use, /agents, /status, /broadcast, /costs, /reset, /help commands
    - Agent name labels on every response
    - Async notification push for cron/heartbeat results
    """

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
    ):
        self.dispatch_fn = dispatch_fn
        self.default_agent = default_agent
        self.list_agents_fn = list_agents_fn
        self.status_fn = status_fn
        self.costs_fn = costs_fn
        self.reset_fn = reset_fn
        self.stream_dispatch_fn = stream_dispatch_fn
        self.addkey_fn = addkey_fn
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

    def _get_active_agent(self, user_id: str) -> str:
        return self._active_agent.get(user_id, self.default_agent)

    def _set_active_agent(self, user_id: str, agent: str) -> None:
        self._active_agent[user_id] = agent

    def _get_agent_names(self) -> list[str]:
        if self.list_agents_fn:
            return list(self.list_agents_fn().keys())
        return []

    async def dispatch(self, agent: str, message: str) -> str:
        """Route a message to an agent and return the response."""
        from src.shared.trace import current_trace_id, new_trace_id

        target = agent or self.default_agent
        if not target:
            return "No agent specified and no default agent configured."
        current_trace_id.set(new_trace_id())
        try:
            return await self.dispatch_fn(target, message)
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

        current = self._get_active_agent(user_id)
        agents = self._get_agent_names()

        # @agent mention routing
        target = current
        message = text
        match = re.match(r"^@(\w+)\s+(.+)$", text, re.DOTALL)
        if match:
            mentioned = match.group(1)
            if mentioned in agents:
                target = mentioned
                message = match.group(2)
            else:
                return f"Unknown agent: '{mentioned}'. Use /agents to list."

        # Slash commands
        if message.startswith("/"):
            return await self._handle_command(user_id, message, current, agents)

        # Normal message: dispatch to agent
        response = await self.dispatch(target, message)
        if not response or not response.strip():
            return ""  # Suppress silent/empty responses
        return f"[{target}] {response}"

    async def _handle_command(
        self, user_id: str, message: str, current: str, agents: list[str],
    ) -> str:
        parts = message.split(None, 1)
        cmd = parts[0].lower()

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
            results = []
            for name in agents:
                resp = await self.dispatch(name, bc_msg)
                results.append(f"[{name}] {resp}")
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
            if self.reset_fn:
                self.reset_fn(current)
            return f"Conversation with '{current}' reset."

        if cmd == "/addkey":
            if not self.addkey_fn:
                return "Credential management not available."
            args = message.split(None, 2)
            if len(args) < 2:
                return "Usage: /addkey <service> <key>"
            service = args[1]
            key = args[2] if len(args) > 2 else ""
            if not key:
                return "Usage: /addkey <service> <key>"
            try:
                self.addkey_fn(service, key)
                return f"Credential '{service}' stored."
            except Exception as e:
                return f"Error storing credential: {e}"

        if cmd == "/help":
            return (
                "Commands:\n"
                "  @agent <msg>      Send message to a specific agent\n"
                "  /use <agent>      Switch active agent\n"
                "  /agents           List all agents\n"
                "  /status           Show agent health\n"
                "  /broadcast <msg>  Send to all agents\n"
                "  /costs            Show today's LLM spend\n"
                "  /addkey <svc> <key>  Add an API credential\n"
                "  /reset            Clear conversation with active agent\n"
                "  /help             Show this help"
            )

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
