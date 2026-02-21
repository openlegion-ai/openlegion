"""REPLSession: interactive command loop for multi-agent chat."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from typing import TYPE_CHECKING

import click

from src.cli.config import (
    _create_agent,
    _default_description,
    _edit_agent_interactive,
    _get_default_model,
    _load_config,
    _load_permissions,
    _pick_browser_interactive,
    _pick_model_interactive,
    _prompt_brightdata_key,
    _save_permissions,
)
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

    _COMMAND_GROUPS = [
        ("Chat", [
            ("/use <name>",       "Switch active agent"),
            ("/broadcast <msg>",  "Send to all agents"),
            ("/steer <msg>",      "Redirect busy agent"),
            ("/reset",            "Clear conversation history"),
        ]),
        ("Agents", [
            ("/status",           "Show agents, models, and state"),
            ("/add",              "Add a new agent"),
            ("/edit [name]",      "Change agent settings"),
            ("/remove [name]",    "Remove an agent"),
        ]),
        ("System", [
            ("/costs",            "Show spend, context, and model health"),
            ("/debug [trace]",    "Show recent request traces"),
            ("/cron [del id]",    "List or delete cron jobs"),
            ("/addkey [service]", "Store a credential"),
            ("/help",             "Show this help"),
            ("/quit",             "Exit and stop runtime"),
        ]),
    ]

    def __init__(self, ctx: RuntimeContext):
        self.ctx = ctx
        self.current = list(ctx.agents.keys())[0]
        self._commands = {
            "/quit":      (self._cmd_quit,      "Exit and stop runtime"),
            "/exit":      (self._cmd_quit,      "Exit and stop runtime"),
            "/agents":    (self._cmd_status,    "Show agents, models, and state"),
            "/use":       (self._cmd_use,       "Switch active agent"),
            "/add":       (self._cmd_add,       "Add a new agent"),
            "/edit":      (self._cmd_edit,      "Change agent settings"),
            "/remove":    (self._cmd_remove,    "Remove an agent"),
            "/status":    (self._cmd_status,    "Show agents, models, and state"),
            "/broadcast": (self._cmd_broadcast, "Send to all agents"),
            "/steer":     (self._cmd_steer,     "Redirect busy agent"),
            "/costs":     (self._cmd_costs,     "Show spend, context, and model health"),
            "/debug":     (self._cmd_debug,     "Show recent request traces"),
            "/cron":      (self._cmd_cron,      "List or delete cron jobs"),
            "/addkey":    (self._cmd_addkey,     "Store a credential"),
            "/reset":     (self._cmd_reset,     "Clear conversation history"),
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

            from src.shared.trace import new_trace_id
            self._send_message(target, message, trace_id=new_trace_id())

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
                click.echo(f"Unknown agent: '{mentioned}'. Type /status to list.")
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

    def _cmd_use(self, arg: str) -> None:
        available = ", ".join(self.ctx.agents)
        if not arg.strip():
            click.echo(f"Usage: /use <agent>  (current: {self.current})")
            click.echo(f"  Available: {available}")
            return
        new_agent = arg.strip()
        if new_agent not in self.ctx.agents:
            click.echo(f"Unknown agent: '{new_agent}'. Available: {available}")
            return
        self.current = new_agent
        click.echo(f"Now chatting with '{self.current}'.")

    def _cmd_add(self, arg: str) -> None:
        from src.host.transport import HttpTransport

        new_name = click.prompt("Agent name")
        if new_name in self.ctx.agents:
            click.echo(f"Agent '{new_name}' already exists.")
            return
        new_desc = click.prompt(
            "What should this agent do?",
            default=_default_description(new_name),
        )
        default_model = _get_default_model()
        model = _pick_model_interactive(default_model, label="default")
        browser = _pick_browser_interactive()

        if browser == "advanced":
            _prompt_brightdata_key()

        _create_agent(new_name, new_desc, model, browser_backend=browser)
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
            if self.ctx.event_bus:
                self.ctx.event_bus.emit("agent_state", agent=new_name,
                    data={"state": "added", "role": new_desc, "ready": True})
        else:
            click.echo(f"Agent '{new_name}' failed to start.", err=True)
            if self.ctx.event_bus:
                self.ctx.event_bus.emit("agent_state", agent=new_name,
                    data={"state": "added", "ready": False})

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
        from src.shared.trace import TRACE_HEADER, new_trace_id

        bc_msg = arg.strip()
        click.echo(f"Broadcasting to {len(self.ctx.agents)} agent(s)...\n")

        def _send(aid: str) -> tuple[str, str]:
            try:
                hdrs = {TRACE_HEADER: new_trace_id()}
                data = self.ctx.transport.request_sync(
                    aid, "POST", "/chat",
                    json={"message": bc_msg}, timeout=120,
                    headers=hdrs,
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
        from src.shared.trace import new_trace_id
        steer_msg = arg.strip()
        future = asyncio.run_coroutine_threadsafe(
            self.ctx.lane_manager.enqueue(
                self.current, steer_msg, mode="steer", trace_id=new_trace_id(),
            ),
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

    def _cmd_debug(self, arg: str) -> None:
        if not self.ctx.trace_store:
            click.echo("Trace store not available.")
            return

        from datetime import datetime, timezone

        trace_id = arg.strip()
        if trace_id:
            events = self.ctx.trace_store.get_trace(trace_id)
            if not events:
                click.echo(f"No events for trace {trace_id}")
                return
            click.echo(f"\n  Trace {trace_id} ({len(events)} events)\n")
            for ev in events:
                ts = datetime.fromtimestamp(ev["timestamp"], tz=timezone.utc).strftime("%H:%M:%S")
                dur = f" ({ev['duration_ms']}ms)" if ev["duration_ms"] else ""
                agent = f" [{ev['agent']}]" if ev["agent"] else ""
                click.echo(f"  {ts}{agent} {ev['event_type']}: {ev['detail']}{dur}")
        else:
            events = self.ctx.trace_store.list_recent(limit=20)
            if not events:
                click.echo("  No traces recorded yet.")
                return
            click.echo()
            for ev in events:
                ts = datetime.fromtimestamp(ev["timestamp"], tz=timezone.utc).strftime("%H:%M:%S")
                agent = ev["agent"] or "-"
                click.echo(f"  {ev['trace_id']}  {ts}  {agent:<16} {ev['event_type']}: {ev['detail'][:60]}")
        click.echo()

    def _cmd_cron(self, arg: str) -> None:
        if not self.ctx.cron_scheduler:
            click.echo("Cron scheduler not available.")
            return
        parts = arg.strip().split(None, 1)
        if parts and parts[0] in ("delete", "del", "rm"):
            if len(parts) < 2:
                click.echo("Usage: /cron delete <job_id>")
                return
            job_id = parts[1].strip()
            if self.ctx.cron_scheduler.remove_job(job_id):
                click.echo(f"  Deleted cron job: {job_id}")
            else:
                click.echo(f"  Job not found: {job_id}")
            return
        jobs = self.ctx.cron_scheduler.list_jobs()
        if not jobs:
            click.echo("  No cron jobs configured.")
            return
        click.echo()
        click.echo(f"  {'ID':<24} {'Agent':<16} {'Schedule':<16} Status")
        click.echo(f"  {'-'*24} {'-'*16} {'-'*16} {'-'*12}")
        for j in jobs:
            status = "active" if j["enabled"] else "paused"
            hb = " [heartbeat]" if j.get("heartbeat") else ""
            click.echo(f"  {j['id']:<24} {j['agent']:<16} {j['schedule']:<16} {status}{hb}")
            if j.get("message"):
                click.echo(f"  {'':24} {j['message'][:60]}")
        click.echo()

    def _cmd_addkey(self, arg: str) -> None:
        if not arg.strip():
            click.echo("  Known services: anthropic, openai, gemini, deepseek, moonshot, xai, groq")
            click.echo("  Other keys: brave_search, brightdata_cdp_url, or any custom name")
            service = click.prompt("Service name")
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

    def _cmd_edit(self, arg: str) -> None:
        """Interactive property editor for an agent. Auto-applies changes."""
        name = arg.strip() if arg.strip() else self.current
        if name not in self.ctx.agents:
            click.echo(f"Agent '{name}' not found.")
            return

        changed_field = _edit_agent_interactive(name)
        if not changed_field:
            return

        if changed_field == "budget":
            # Budget is enforced by the mesh host — no container restart needed.
            fresh_cfg = _load_config()
            budget = fresh_cfg.get("agents", {}).get(name, {}).get("budget", {})
            if budget and self.ctx.cost_tracker:
                self.ctx.cost_tracker.set_budget(
                    name,
                    daily_usd=budget.get("daily_usd", 10.0),
                    monthly_usd=budget.get("monthly_usd", 200.0),
                )
            click.echo("Applied.")
        else:
            # Model, browser, description, system prompt — restart the container.
            self._restart_agent(name)

    def _restart_agent(self, name: str) -> None:
        """Stop and restart an agent container with fresh config."""
        from src.host.transport import HttpTransport

        click.echo(f"Restarting '{name}'...", nl=False)

        # Stop old container
        try:
            self.ctx.runtime.stop_agent(name)
        except Exception:
            pass

        # Read fresh config
        fresh_cfg = _load_config()
        agent_cfg = fresh_cfg.get("agents", {}).get(name, {})
        default_model = fresh_cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
        skills_dir = os.path.abspath(agent_cfg.get("skills_dir", ""))
        agent_model = agent_cfg.get("model", default_model)
        agent_mcp_servers = agent_cfg.get("mcp_servers") or None
        agent_browser_backend = agent_cfg.get("browser_backend", "")

        # Start new container
        url = self.ctx.runtime.start_agent(
            agent_id=name,
            role=agent_cfg.get("role", ""),
            skills_dir=skills_dir,
            system_prompt=agent_cfg.get("system_prompt", ""),
            model=agent_model,
            mcp_servers=agent_mcp_servers,
            browser_backend=agent_browser_backend,
        )

        # Update router and transport
        self.ctx.router.register_agent(name, url)
        self.ctx.agent_urls[name] = url
        if isinstance(self.ctx.transport, HttpTransport):
            self.ctx.transport.register(name, url)

        # Wait for readiness
        ready = asyncio.run(self.ctx.runtime.wait_for_agent(name, timeout=60))
        if ready:
            click.echo(" ready.")
        else:
            click.echo(" failed to start.", err=True)

    def _cmd_remove(self, arg: str) -> None:
        """Remove an agent from config and stop its container."""
        from src.host.transport import HttpTransport

        if not self.ctx.agents:
            click.echo("No agents to remove.")
            return

        name = arg.strip() if arg.strip() else None
        if name is None:
            # Interactive picker
            names = sorted(self.ctx.agents.keys())
            if len(names) == 1:
                name = names[0]
            else:
                click.echo("Agents:")
                for i, n in enumerate(names, 1):
                    click.echo(f"  {i}. {n}")
                choice = click.prompt("Select agent to remove", type=click.IntRange(1, len(names)))
                name = names[choice - 1]

        if name not in self.ctx.agents:
            click.echo(f"Agent '{name}' not found.")
            return

        if not click.confirm(f"Remove agent '{name}'?"):
            return

        # Stop the container
        try:
            self.ctx.runtime.stop_agent(name)
        except Exception:
            pass

        # Remove from router and transport
        if self.ctx.router:
            self.ctx.router.unregister_agent(name)
        self.ctx.agent_urls.pop(name, None)
        if isinstance(self.ctx.transport, HttpTransport):
            self.ctx.transport._urls.pop(name, None)
        if self.ctx.health_monitor:
            self.ctx.health_monitor.unregister(name)

        # Clean up PubSub subscriptions, cron jobs, and lane state
        if self.ctx.pubsub:
            self.ctx.pubsub.unsubscribe_agent(name)
        if self.ctx.cron_scheduler:
            removed = self.ctx.cron_scheduler.remove_agent_jobs(name)
            if removed:
                click.echo(f"  Removed {removed} cron job(s).")
        if self.ctx.lane_manager:
            self.ctx.lane_manager.remove_lane(name)

        # Remove from config and permissions
        import yaml
        from src.cli.config import AGENTS_FILE

        if AGENTS_FILE.exists():
            with open(AGENTS_FILE) as f:
                agents_cfg = yaml.safe_load(f) or {}
            agents_cfg.get("agents", {}).pop(name, None)
            with open(AGENTS_FILE, "w") as f:
                yaml.dump(agents_cfg, f, default_flow_style=False, sort_keys=False)

        perms = _load_permissions()
        perms.get("permissions", {}).pop(name, None)
        _save_permissions(perms)

        click.echo(f"Removed agent '{name}'.")
        if self.ctx.event_bus:
            self.ctx.event_bus.emit("agent_state", agent=name,
                data={"state": "removed"})

        # Switch to another agent if we removed the active one
        if name == self.current and self.ctx.agents:
            self.current = list(self.ctx.agents.keys())[0]
            click.echo(f"Switched to '{self.current}'.")
        elif not self.ctx.agents:
            click.echo("No agents remaining. Add one with /add.")

    def _cmd_help(self, arg: str) -> None:
        click.echo(f"\n  {'@agent <msg>':<22} Send to a specific agent\n")
        for group_name, commands in self._COMMAND_GROUPS:
            click.echo(f"  {group_name}")
            for cmd, desc in commands:
                click.echo(f"    {cmd:<20} {desc}")
            click.echo()

    # ── Message sending ─────────────────────────────────────

    def _send_message(self, target: str, message: str, trace_id: str | None = None) -> None:
        """Send via streaming transport with formatted output."""
        if target not in self.ctx.agents:
            click.echo(f"Agent '{target}' not found.")
            return

        def _steer_poll(timeout: float) -> None:
            """Poll stdin during streaming; dispatch /steer commands inline."""
            import select
            import sys

            from src.shared.trace import new_trace_id

            readable, _, _ = select.select([sys.stdin], [], [], timeout)
            if readable:
                line = sys.stdin.readline().strip()
                if line.startswith("/steer ") and self.ctx.lane_manager:
                    steer_msg = line[7:].strip()
                    if steer_msg:
                        sf = asyncio.run_coroutine_threadsafe(
                            self.ctx.lane_manager.enqueue(
                                target, steer_msg, mode="steer",
                                trace_id=new_trace_id(),
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
                steer_fn=_steer_poll, trace_id=trace_id,
                event_bus=self.ctx.event_bus,
            )
        except Exception as e:
            click.echo(f"Error: {e}")


def _send_message_streaming(
    transport, target: str, message: str,
    dispatch_loop=None,
    steer_fn=None,
    trace_id: str | None = None,
    event_bus=None,
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
    from src.shared.trace import current_trace_id, trace_headers

    if isinstance(transport, HttpTransport):
        async def _stream():
            current_trace_id.set(trace_id)
            hdrs = trace_headers()
            response_text = ""
            tool_count = 0
            if event_bus:
                event_bus.emit("message_received", agent=target,
                    data={"message": message[:200], "source": "repl"})
            try:
                async for event in transport.stream_request(
                    target, "POST", "/chat/stream",
                    json={"message": message}, timeout=120, headers=hdrs,
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
                            if event_bus:
                                event_bus.emit("tool_start", agent=target,
                                    data={k: v for k, v in event.items() if k != "type"})
                        elif etype == "tool_result":
                            display_stream_tool_result(
                                event.get("name", "?"),
                                event.get("output", {}),
                            )
                            if event_bus:
                                event_bus.emit("tool_result", agent=target,
                                    data={k: v for k, v in event.items() if k != "type"})
                        elif etype == "text_delta":
                            content = event.get("content", "")
                            display_stream_text_delta(target, content, not response_text)
                            response_text += content
                        elif etype == "done":
                            if not response_text:
                                resp = event.get("response", "(no response)")
                                click.echo(f"\n{agent_prompt(target)}{resp}")
                            else:
                                resp = response_text
                                click.echo("")  # newline after streamed text
                            if event_bus:
                                event_bus.emit("message_sent", agent=target,
                                    data={"message": message[:200],
                                          "response_length": len(resp),
                                          "tool_count": tool_count,
                                          "source": "repl"})
                            return
                        elif "error" in event:
                            click.echo(f"Error: {event['error']}")
                            return
            except Exception:
                # Fallback to non-streaming
                data = await transport.request(
                    target, "POST", "/chat",
                    json={"message": message}, timeout=120, headers=hdrs,
                )
                if "error" in data and "response" not in data:
                    click.echo(f"Error: {data['error']}")
                    return
                display_response(target, data)
                if event_bus:
                    resp = data.get("response", "")
                    event_bus.emit("message_sent", agent=target,
                        data={"message": message[:200],
                              "response_length": len(resp),
                              "source": "repl"})
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
        from src.shared.trace import TRACE_HEADER
        sync_hdrs = {TRACE_HEADER: trace_id} if trace_id else {}
        data = transport.request_sync(
            target, "POST", "/chat",
            json={"message": message}, timeout=120,
            headers=sync_hdrs or None,
        )
        if "error" in data and "response" not in data:
            click.echo(f"Error: {data['error']}")
            return
        display_response(target, data)
