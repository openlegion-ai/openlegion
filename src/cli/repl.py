"""REPLSession: interactive command loop for multi-agent chat."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from typing import TYPE_CHECKING

import click

from src.cli.config import _create_agent, _get_default_model, _load_config
from src.cli.formatting import (
    agent_prompt,
    display_response,
    display_stream_text_delta,
    display_stream_tool_result,
    display_stream_tool_start,
    user_prompt,
)

if TYPE_CHECKING:
    from src.cli.runtime import RuntimeContext


class REPLSession:
    """Interactive REPL supporting multiple agents, @mentions, and slash commands."""

    def __init__(self, ctx: RuntimeContext):
        self.ctx = ctx
        self.current = list(ctx.agents.keys())[0]
        self._commands = {
            "/quit":      (self._cmd_quit,      "Exit and stop runtime"),
            "/exit":      (self._cmd_quit,      "Exit and stop runtime"),
            "/agents":    (self._cmd_agents,    "List all agents"),
            "/use":       (self._cmd_use,       "Switch active agent"),
            "/add":       (self._cmd_add,       "Add a new agent"),
            "/status":    (self._cmd_status,    "Show agent health"),
            "/broadcast": (self._cmd_broadcast, "Send to all agents"),
            "/steer":     (self._cmd_steer,     "Inject message into busy agent"),
            "/costs":     (self._cmd_costs,     "Show today's LLM spend"),
            "/addkey":    (self._cmd_addkey,     "Add an API credential"),
            "/reset":     (self._cmd_reset,     "Clear conversation"),
            "/help":      (self._cmd_help,      "Show this help"),
        }

    def run(self) -> None:
        """Main REPL loop."""
        while True:
            try:
                user_input = input(user_prompt(self.current)).strip()
            except EOFError:
                break

            if not user_input:
                continue

            target, message = self._parse_input(user_input)
            if target is None:
                continue

            if message.startswith("/"):
                if self._dispatch_command(message) == "quit":
                    break
                continue

            self._send_message(target, message)

    def _parse_input(self, text: str) -> tuple[str | None, str]:
        """Parse @mentions, return (target_agent, message)."""
        if text.startswith("@"):
            parts = text.split(None, 1)
            mentioned = parts[0][1:]
            if mentioned in self.ctx.agents:
                message = parts[1] if len(parts) > 1 else ""
                if not message:
                    click.echo(f"Usage: @{mentioned} <message>")
                    return None, ""
                return mentioned, message
            else:
                click.echo(f"Unknown agent: '{mentioned}'. Type /agents to list.")
                return None, ""
        return self.current, text

    def _dispatch_command(self, message: str) -> str | None:
        """Look up and execute slash command. Returns 'quit' to exit."""
        cmd_parts = message.split(None, 1)
        cmd = cmd_parts[0].lower()
        handler_entry = self._commands.get(cmd)
        if handler_entry is None:
            click.echo(f"Unknown command: {cmd}. Type /help for commands.")
            return None
        handler, _ = handler_entry
        arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
        return handler(arg)

    # ── Command handlers ────────────────────────────────────

    def _cmd_quit(self, arg: str) -> str:
        return "quit"

    def _cmd_agents(self, arg: str) -> None:
        for name in self.ctx.agents:
            marker = " (active)" if name == self.current else ""
            click.echo(f"  {name}{marker}")

    def _cmd_use(self, arg: str) -> None:
        if not arg.strip():
            click.echo(f"Usage: /use <agent>  (current: {self.current})")
            return
        new_agent = arg.strip()
        if new_agent not in self.ctx.agents:
            click.echo(f"Unknown agent: '{new_agent}'. Type /agents to list.")
            return
        self.current = new_agent
        click.echo(f"Now chatting with '{self.current}'.")

    def _cmd_add(self, arg: str) -> None:
        from src.cli.config import _PROVIDER_MODELS
        from src.host.transport import HttpTransport

        new_name = click.prompt("Agent name")
        if new_name in self.ctx.agents:
            click.echo(f"Agent '{new_name}' already exists.")
            return
        new_desc = click.prompt(
            "What should this agent do?",
            default=f"General-purpose {new_name} assistant",
        )
        default_model = _get_default_model()
        provider = default_model.split("/")[0] if "/" in default_model else "anthropic"
        models = _PROVIDER_MODELS.get(provider, [default_model])
        default_idx = 1
        for i, m in enumerate(models, 1):
            marker = " (default)" if m == default_model else ""
            click.echo(f"  {i}. {m}{marker}")
            if m == default_model:
                default_idx = i
        model_choice = click.prompt(
            "Model",
            type=click.IntRange(1, len(models)),
            default=default_idx,
        )
        model = models[model_choice - 1]
        _create_agent(new_name, new_desc, model)
        # Reload permissions so the mesh grants the new agent API access
        self.ctx.permissions.reload()
        agent_cfg_data = _load_config().get("agents", {}).get(new_name, {})
        skills_dir = os.path.abspath(agent_cfg_data.get("skills_dir", ""))
        add_mcp_servers = agent_cfg_data.get("mcp_servers") or None
        add_browser_backend = agent_cfg_data.get("browser_backend", "")
        url = self.ctx.runtime.start_agent(
            agent_id=new_name,
            role=new_desc,
            skills_dir=skills_dir,
            system_prompt=agent_cfg_data.get("system_prompt", ""),
            model=agent_cfg_data.get("model", model),
            mcp_servers=add_mcp_servers,
            browser_backend=add_browser_backend,
        )
        self.ctx.router.register_agent(new_name, url)
        self.ctx.agent_urls[new_name] = url
        if isinstance(self.ctx.transport, HttpTransport):
            self.ctx.transport.register(new_name, url)
        click.echo(f"Starting '{new_name}'...")
        ready = asyncio.run(self.ctx.runtime.wait_for_agent(new_name, timeout=60))
        if ready:
            click.echo(f"Agent '{new_name}' ready.")
        else:
            click.echo(f"Agent '{new_name}' failed to start.", err=True)

    def _cmd_status(self, arg: str) -> None:
        agents_cfg = self.ctx.cfg.get("agents", {})
        default_model = self.ctx.cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        for name in self.ctx.agents:
            agent_cfg = agents_cfg.get(name, {})
            model = agent_cfg.get("model", default_model)
            browser = agent_cfg.get("browser_backend", "basic") or "basic"
            try:
                data = self.ctx.transport.request_sync(name, "GET", "/status", timeout=3)
                state = data.get("state", "unknown")
                tasks = data.get("tasks_completed", 0)
                click.echo(f"  {name:<20} {state:<12} {tasks} tasks    model: {model:<20} browser: {browser}")
            except Exception:
                click.echo(f"  {name:<20} unreachable")

    def _cmd_broadcast(self, arg: str) -> None:
        if not arg.strip():
            click.echo("Usage: /broadcast <message>")
            return
        bc_msg = arg.strip()
        click.echo(f"Broadcasting to {len(self.ctx.agents)} agent(s)...\n")

        def _send(aid: str) -> tuple[str, str]:
            try:
                data = self.ctx.transport.request_sync(
                    aid, "POST", "/chat",
                    json={"message": bc_msg}, timeout=120,
                )
                return aid, data.get("response", "(no response)")
            except Exception as e:
                return aid, f"(error: {e})"

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.ctx.agents)) as pool:
            futures = {pool.submit(_send, aid): aid for aid in self.ctx.agents}
            for future in concurrent.futures.as_completed(futures):
                aid, response = future.result()
                click.echo(f"[{aid}] {response}\n")

    def _cmd_steer(self, arg: str) -> None:
        if not arg.strip():
            click.echo("Usage: /steer <message>")
            return
        if self.ctx.lane_manager is None or self.ctx.dispatch_loop is None:
            click.echo("Steer not available in this mode.")
            return
        steer_msg = arg.strip()
        future = asyncio.run_coroutine_threadsafe(
            self.ctx.lane_manager.enqueue(self.current, steer_msg, mode="steer"),
            self.ctx.dispatch_loop,
        )
        click.echo(future.result())

    def _cmd_costs(self, arg: str) -> None:
        try:
            agents_spend = self.ctx.cost_tracker.get_all_agents_spend("today")
            agents_spend = [a for a in agents_spend if a["agent"] in self.ctx.agents]
            if not agents_spend:
                click.echo("  No usage recorded today.")
            else:
                total = sum(a["cost"] for a in agents_spend)
                click.echo(f"\n  Today's spend: ${total:.4f}\n")
                for a in agents_spend:
                    click.echo(f"  {a['agent']:<20} {a['tokens']:>8,} tokens    ${a['cost']:.4f}")

            # Context window usage
            click.echo("\n  Context usage")
            for name in self.ctx.agents:
                try:
                    data = self.ctx.transport.request_sync(name, "GET", "/status", timeout=3)
                    ctx_tokens = data.get("context_tokens", 0)
                    ctx_max = data.get("context_max", 0)
                    ctx_pct = data.get("context_pct", 0.0)
                    if ctx_max:
                        pct_str = f"{int(ctx_pct * 100)}%"
                        click.echo(f"  {name:<20} {ctx_tokens:,}/{ctx_max:,} tokens ({pct_str})")
                    else:
                        click.echo(f"  {name:<20} n/a")
                except Exception:
                    click.echo(f"  {name:<20} unreachable")

            # Model health
            model_health = self.ctx.credential_vault.get_model_health() if self.ctx.credential_vault else []
            if model_health:
                click.echo("\n  Model health")
                for mh in model_health:
                    status = "ok" if mh["available"] else f"cooldown {mh['cooldown_remaining']:.0f}s"
                    click.echo(
                        f"  {mh['model']:<40} {status:<20} "
                        f"{mh['success_count']} ok / {mh['failure_count']} fail"
                    )
        except Exception as e:
            click.echo(f"Error: {e}")

    def _cmd_addkey(self, arg: str) -> None:
        if not arg.strip():
            service = click.prompt("Service name (e.g. anthropic, openai, brave_search)")
        else:
            service = arg.split()[0]
        # Normalize: bare provider names get _api_key suffix
        known_providers = {"anthropic", "openai", "gemini", "deepseek", "moonshot", "xai", "groq"}
        if service.lower() in known_providers and not service.lower().endswith("_api_key"):
            service = f"{service}_api_key"
        inline_parts = arg.split(None, 1) if arg.strip() else []
        if len(inline_parts) > 1:
            key_value = inline_parts[1]
        else:
            key_value = click.prompt(f"  {service} key", hide_input=True)
        if not key_value:
            click.echo("No key provided.")
            return
        self.ctx.credential_vault.add_credential(service, key_value)
        click.echo(f"Credential '{service}' stored.")

    def _cmd_reset(self, arg: str) -> None:
        try:
            self.ctx.transport.request_sync(self.current, "POST", "/chat/reset", timeout=5)
            click.echo(f"Conversation with '{self.current}' reset.")
        except Exception as e:
            click.echo(f"Error: {e}")

    def _cmd_help(self, arg: str) -> None:
        click.echo("Commands:")
        click.echo(f"  {'@agent <msg>':<18} Send message to a specific agent")
        for cmd, (_, desc) in self._commands.items():
            if cmd != "/exit":  # skip alias
                click.echo(f"  {cmd:<18} {desc}")

    # ── Message sending ─────────────────────────────────────

    def _send_message(self, target: str, message: str) -> None:
        """Send via streaming transport with formatted output."""
        if target not in self.ctx.agents:
            click.echo(f"Agent '{target}' not found.")
            return

        def _steer_poll(timeout: float) -> None:
            """Poll stdin during streaming; dispatch /steer commands inline."""
            import select
            import sys

            readable, _, _ = select.select([sys.stdin], [], [], timeout)
            if readable:
                line = sys.stdin.readline().strip()
                if line.startswith("/steer ") and self.ctx.lane_manager:
                    steer_msg = line[7:].strip()
                    if steer_msg:
                        sf = asyncio.run_coroutine_threadsafe(
                            self.ctx.lane_manager.enqueue(
                                target, steer_msg, mode="steer",
                            ),
                            self.ctx.dispatch_loop,
                        )
                        try:
                            click.echo(sf.result(timeout=10))
                        except Exception as e:
                            click.echo(f"Steer error: {e}")
                elif line:
                    click.echo("(type /steer <msg> to redirect the agent)")

        try:
            _send_message_streaming(
                self.ctx.transport, target, message, self.ctx.dispatch_loop,
                steer_fn=_steer_poll,
            )
        except Exception as e:
            click.echo(f"Error: {e}")


def _send_message_streaming(
    transport, target: str, message: str,
    dispatch_loop=None,
    steer_fn=None,
) -> None:
    """Send a message via streaming endpoint, falling back to non-streaming.

    When *dispatch_loop* is provided, the async work runs on that loop
    (via ``run_coroutine_threadsafe``) so the httpx client is shared with
    the lane-manager dispatch path.  Otherwise falls back to ``asyncio.run()``.

    When *steer_fn* is provided, it is called with a timeout (in seconds)
    while waiting for the stream to complete, allowing the caller to poll
    stdin for ``/steer`` commands.
    """
    from src.host.transport import HttpTransport

    if isinstance(transport, HttpTransport):
        async def _stream():
            response_text = ""
            tool_count = 0
            try:
                async for event in transport.stream_request(
                    target, "POST", "/chat/stream",
                    json={"message": message}, timeout=120,
                ):
                    if isinstance(event, dict):
                        etype = event.get("type", "")
                        if etype == "tool_start":
                            tool_count += 1
                            display_stream_tool_start(
                                event.get("name", "?"),
                                event.get("input", {}),
                                tool_count,
                            )
                        elif etype == "tool_result":
                            display_stream_tool_result(
                                event.get("name", "?"),
                                event.get("output", {}),
                            )
                        elif etype == "text_delta":
                            content = event.get("content", "")
                            display_stream_text_delta(target, content, not response_text)
                            response_text += content
                        elif etype == "done":
                            if not response_text:
                                resp = event.get("response", "(no response)")
                                click.echo(f"\n{agent_prompt(target)}{resp}")
                            else:
                                click.echo("")  # newline after streamed text
                            return
                        elif "error" in event:
                            click.echo(f"Error: {event['error']}")
                            return
            except Exception:
                # Fallback to non-streaming
                data = await transport.request(
                    target, "POST", "/chat",
                    json={"message": message}, timeout=120,
                )
                if "error" in data and "response" not in data:
                    click.echo(f"Error: {data['error']}")
                    return
                display_response(target, data)
                return

            click.echo("")

        if dispatch_loop is not None:
            future = asyncio.run_coroutine_threadsafe(_stream(), dispatch_loop)
            if steer_fn is not None:
                while not future.done():
                    steer_fn(0.2)
                future.result()  # re-raise exceptions
            else:
                future.result()
        else:
            asyncio.run(_stream())
    else:
        # Non-streaming fallback for SandboxTransport
        data = transport.request_sync(
            target, "POST", "/chat",
            json={"message": message}, timeout=120,
        )
        if "error" in data and "response" not in data:
            click.echo(f"Error: {data['error']}")
            return
        display_response(target, data)
