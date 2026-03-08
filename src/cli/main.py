"""CLI entry point for OpenLegion.

Commands:
  start                    Start runtime + interactive REPL
  start -d                 Start in background (detached)
  stop                     Stop all containers
  status                   Show agent status
  chat <name>              Connect to a running agent
  version [-v]             Show version info
"""

from __future__ import annotations

import json as _json
import logging
import os
import subprocess
import sys
import threading

import click

from src.cli import config as cli_config
from src.cli.config import (
    _load_config,
    _suppress_host_logs,
)

logger = logging.getLogger("cli")

_json_mode = False


def _set_json(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    """Eager callback to enable global JSON output mode."""
    global _json_mode
    _json_mode = value


def _fail(msg: str) -> None:
    """Print error message to stderr and exit with code 1."""
    click.echo(msg, err=True)
    raise SystemExit(1)


# ── Shell completion helpers ─────────────────────────────────


def _complete_agent_names(ctx, param, incomplete):
    """Return matching agent names for shell completion."""
    try:
        cfg = _load_config()
        names = sorted(cfg.get("agents", {}).keys())
        return [n for n in names if n.startswith(incomplete)]
    except Exception as e:
        logger.debug("Shell completion failed: %s", e)
        return []


# ── Main group ───────────────────────────────────────────────

@click.group()
@click.version_option(package_name="openlegion")
@click.option(
    "--json", "json_flag", is_flag=True, is_eager=True, expose_value=False,
    callback=_set_json, help="Output in JSON format (where supported)",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output (DEBUG logging)")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Quiet output (ERROR logging only)")
def cli(verbose: bool, quiet: bool):
    """OpenLegion -- Autonomous AI agent fleet."""
    from dotenv import load_dotenv

    load_dotenv(cli_config.ENV_FILE, interpolate=False)

    if verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)
    elif quiet:
        logging.basicConfig(level=logging.ERROR, force=True)
    else:
        _suppress_host_logs()


# ── start ────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "config_path", default=str(cli_config.CONFIG_FILE), help="Path to mesh config")
@click.option("--detach", "-d", is_flag=True, help="Run in background (no interactive REPL)")
@click.option("--sandbox", is_flag=True, help="Use Docker Sandbox microVMs (requires Docker Desktop 4.58+)")
@click.option("--serve", is_flag=True, hidden=True, help="Headless mode (used by -d internally)")
def start(config_path: str, detach: bool, sandbox: bool, serve: bool):
    """Start the runtime and chat with your agents.

    By default, starts the mesh and all agents then drops into an interactive
    REPL. Use -d to run in the background instead.

    \b
    Examples:
      openlegion start              # interactive mode
      openlegion start -d           # background mode
      openlegion start --sandbox    # use microVM isolation
    """
    if detach:
        _start_detached(config_path)
        return

    from src.cli.runtime import RuntimeContext

    ctx = RuntimeContext(config_path, use_sandbox=sandbox)
    try:
        ctx.start()
        if serve or not sys.stdin.isatty():
            # Headless mode: keep alive via signal wait (used by -d and systemd)
            import signal
            shutdown_event = threading.Event()
            signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())
            signal.signal(signal.SIGINT, lambda *_: shutdown_event.set())
            shutdown_event.wait()
        else:
            _redirect_host_logs_to_file()
            from src.cli.repl import REPLSession
            repl = REPLSession(ctx)
            repl.run()
    except KeyboardInterrupt:
        click.echo("")
    finally:
        ctx.shutdown()


def _redirect_host_logs_to_file() -> None:
    """Redirect host-side loggers to .openlegion.log during interactive REPL.

    Prevents structured JSON log lines from interleaving with chat output.
    """
    log_path = cli_config.PROJECT_ROOT / ".openlegion.log"
    file_handler = logging.FileHandler(log_path, mode="a")
    log_format = os.environ.get("OPENLEGION_LOG_FORMAT", "json").lower()
    if log_format == "text":
        from src.shared.utils import TextFormatter
        file_handler.setFormatter(TextFormatter())
    else:
        from src.shared.utils import StructuredFormatter
        file_handler.setFormatter(StructuredFormatter())

    for name in ["host.health", "host.runtime", "host.mesh", "host.transport",
                 "host.server", "host.costs", "host.cron", "host.credentials",
                 "host.permissions", "host.lanes",
                 "channels.slack", "channels.telegram", "channels.discord",
                 "channels.whatsapp", "channels.base"]:
        host_logger = logging.getLogger(name)
        # Replace stderr handlers with file handler
        host_logger.handlers = [file_handler]


def _start_detached(config_path: str) -> None:
    """Start the runtime in a background subprocess.

    Uses a log file instead of a pipe for child stdout.  This prevents
    the pipe buffer from filling up after the parent exits, which would
    cause SIGPIPE to kill the background process and terminal lag while
    the buffer is filling.
    """
    import time

    log_path = cli_config.PROJECT_ROOT / ".openlegion.log"
    log_fd = open(log_path, "a")

    try:
        cmd = [sys.executable, "-m", "src.cli", "start", "--config", config_path, "--serve"]

        popen_kwargs: dict = dict(
            cwd=str(cli_config.PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
        )
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(cmd, **popen_kwargs)

        # Poll log file for startup output
        deadline = time.time() + 90
        printed_lines = 0
        ready = False

        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                lines = log_path.read_text().splitlines()
            except OSError:
                time.sleep(0.5)
                continue

            for line in lines[printed_lines:]:
                click.echo(line)
            printed_lines = len(lines)

            if any("Chatting with" in line for line in lines):
                ready = True
                break
            time.sleep(0.5)
    finally:
        log_fd.close()

    if proc.poll() is not None:
        # Process died during startup
        try:
            lines = log_path.read_text().splitlines()
            for line in lines[printed_lines:]:
                click.echo(f"  {line}", err=True)
        except OSError:
            pass
        _fail(f"Runtime failed to start. Check logs.\n  Log: {log_path}")

    if not ready:
        _fail(f"Startup timed out. Check logs.\n  Log: {log_path}")

    # Write PID file so `openlegion stop` can kill the host process
    pid_path = cli_config.PROJECT_ROOT / ".openlegion.pid"
    pid_path.write_text(str(proc.pid))

    click.echo(f"\nOpenLegion running in background (PID {proc.pid}).")
    click.echo(f"  Log file:            {log_path}")
    click.echo("  Chat with an agent:  openlegion chat <name>")
    click.echo("  Stop the runtime:    openlegion stop")


# ── chat (for detached mode) ─────────────────────────────────

@cli.command("chat")
@click.argument("name", required=False, default=None, shell_complete=_complete_agent_names)
@click.option("--port", default=8420, type=int, help="Mesh host port")
def chat(name: str | None, port: int):
    """Connect to a running agent and start chatting.

    \b
    Examples:
      openlegion chat              # pick from running agents
      openlegion chat assistant    # connect directly
    """
    import httpx

    try:
        resp = httpx.get(f"http://localhost:{port}/mesh/agents", timeout=5)
        agents = resp.json()
    except httpx.ConnectError:
        _fail("Mesh is not running. Start it first: openlegion start")
    except Exception as e:
        _fail(f"Error contacting mesh: {e}")

    if not agents:
        _fail("No agents running.")

    # Interactive agent selection when no name given
    if name is None:
        agent_names = sorted(agents.keys())
        if len(agent_names) == 1:
            name = agent_names[0]
        else:
            click.echo("Running agents:")
            for i, n in enumerate(agent_names, 1):
                click.echo(f"  {i}. {n}")
            choice = click.prompt(
                "Select agent",
                type=click.IntRange(1, len(agent_names)),
                default=1,
            )
            name = agent_names[choice - 1]

    agent_info = agents.get(name)
    if not agent_info:
        available = ", ".join(agents.keys()) if agents else "(none)"
        _fail(f"Agent '{name}' is not running. Running agents: {available}")

    agent_url = agent_info.get("url", agent_info) if isinstance(agent_info, dict) else agent_info
    click.echo(f"Connected to '{name}' at {agent_url}")
    click.echo("Type a message to chat. /help for commands.\n")

    try:
        _single_agent_repl(agent_name=name, agent_url=agent_url, mesh_port=port)
    except KeyboardInterrupt:
        click.echo("\nDisconnected.")


def _single_agent_repl(agent_name: str, agent_url: str, mesh_port: int = 8420) -> None:
    """Interactive chat REPL with a single agent (for detached mode)."""
    import httpx

    from src.cli.formatting import user_prompt

    while True:
        try:
            user_input = input(user_prompt(agent_name)).strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/reset":
                try:
                    httpx.post(f"{agent_url}/chat/reset", timeout=5)
                    click.echo("Conversation reset.")
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
                continue
            elif cmd == "/status":
                try:
                    resp = httpx.get(f"{agent_url}/status", timeout=5)
                    data = resp.json()
                    click.echo(f"  State: {data['state']}")
                    click.echo(f"  Role: {data['role']}")
                    click.echo(f"  Tools: {', '.join(data.get('capabilities', []))}")
                    click.echo(f"  Tasks completed: {data.get('tasks_completed', 0)}")
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
                continue
            elif cmd == "/costs":
                try:
                    data = httpx.get(f"http://localhost:{mesh_port}/dashboard/api/costs?period=today", timeout=5).json()
                    agents = data if isinstance(data, list) else data.get("agents", [])
                    if not agents:
                        click.echo("  No costs recorded today.")
                    else:
                        for a in agents:
                            cost = a.get('cost', 0)
                            tokens = a.get('tokens', 0)
                            click.echo(f"  {a.get('agent', '?')}: ${cost:.4f} ({tokens} tokens)")
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
                continue
            elif cmd == "/agents":
                try:
                    resp = httpx.get(f"http://localhost:{mesh_port}/mesh/agents", timeout=5)
                    agents = resp.json()
                    for aid, info in agents.items():
                        role = info.get("role", "") if isinstance(info, dict) else ""
                        click.echo(f"  {aid}: {role}")
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
                continue
            elif cmd == "/debug":
                try:
                    data = httpx.get(f"http://localhost:{mesh_port}/dashboard/api/traces", timeout=5).json()
                    traces = data if isinstance(data, list) else data.get("traces", [])
                    for t in traces[:10]:
                        click.echo(f"  {t.get('trace_id', '?')} {t.get('agent', '?')} {t.get('event_type', '?')}")
                    if not traces:
                        click.echo("  No recent traces.")
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
                continue
            elif cmd.startswith("/blackboard"):
                try:
                    data = httpx.get(f"http://localhost:{mesh_port}/dashboard/api/blackboard", timeout=5).json()
                    entries = data if isinstance(data, list) else data.get("entries", [])
                    if not entries:
                        click.echo("  Blackboard is empty.")
                    else:
                        for e in entries:
                            click.echo(f"  {e.get('key', '?')}: {_json.dumps(e.get('value', {}), default=str)[:60]}")
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
                continue
            elif cmd == "/help":
                click.echo("Commands:")
                click.echo(f"  {'/reset':<18} Clear conversation history")
                click.echo(f"  {'/status':<18} Show agent status")
                click.echo(f"  {'/costs':<18} Show cost breakdown (today)")
                click.echo(f"  {'/agents':<18} List running agents")
                click.echo(f"  {'/debug':<18} Show recent traces")
                click.echo(f"  {'/blackboard':<18} Show blackboard entries")
                click.echo(f"  {'/quit':<18} Exit chat")
                click.echo(f"  {'/help':<18} Show this help")
                continue
            else:
                click.echo(f"Unknown command: {cmd}. Type /help for commands.")
                continue

        # Try streaming first, fall back to sync
        try:
            _stream_detached_chat(agent_name, agent_url, user_input)
        except (httpx.HTTPError, httpx.StreamError, OSError):
            _sync_detached_chat(agent_name, agent_url, user_input)


def _stream_detached_chat(agent_name: str, agent_url: str, message: str) -> None:
    """Stream a chat response via SSE."""
    import httpx

    from src.cli.formatting import (
        agent_prompt,
        display_stream_text_delta,
        display_stream_tool_result,
        display_stream_tool_start,
    )

    response_text = ""
    tool_count = 0
    with httpx.stream(
        "POST", f"{agent_url}/chat/stream",
        json={"message": message}, timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                event = _json.loads(line[6:])
            except _json.JSONDecodeError:
                continue
            etype = event.get("type", "")
            if etype == "tool_start":
                tool_count += 1
                display_stream_tool_start(
                    event.get("name", "?"), event.get("input", {}), tool_count,
                )
            elif etype == "tool_result":
                display_stream_tool_result(
                    event.get("name", "?"), event.get("output", {}),
                )
            elif etype == "text_delta":
                content = event.get("content", "")
                display_stream_text_delta(
                    agent_name, content, not response_text,
                )
                response_text += content
            elif etype == "done":
                if not response_text:
                    resp_text = event.get("response", "(no response)")
                    click.echo(f"\n{agent_prompt(agent_name)}{resp_text}")
                click.echo("")
                return
    if response_text:
        click.echo("")


def _sync_detached_chat(agent_name: str, agent_url: str, message: str) -> None:
    """Synchronous chat fallback."""
    import httpx

    from src.cli.formatting import display_response

    resp = httpx.post(
        f"{agent_url}/chat", json={"message": message}, timeout=120,
    )
    if resp.status_code != 200:
        click.echo(f"Error: HTTP {resp.status_code}: {resp.text}", err=True)
        return
    display_response(agent_name, resp.json())


# ── status ───────────────────────────────────────────────────

@cli.command("status")
@click.option("--port", default=8420, type=int, help="Mesh host port")
@click.option("--wide", "-w", is_flag=True, help="Show additional columns")
@click.option(
    "--watch", "watch_interval", default=None, type=int,
    help="Auto-refresh every N seconds",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def status(port: int, wide: bool, watch_interval: int | None, as_json: bool):
    """Show status of all agents."""
    as_json = as_json or _json_mode
    import httpx

    def _collect_status():
        cfg = _load_config()
        configured = cfg.get("agents", {})
        mesh_agents: dict = {}
        mesh_online = False
        try:
            resp = httpx.get(f"http://localhost:{port}/mesh/agents", timeout=5)
            mesh_agents = resp.json()
            mesh_online = True
        except httpx.ConnectError:
            pass
        except Exception as e:
            logger.debug("Error checking mesh: %s", e)

        all_names = sorted(set(list(configured.keys()) + list(mesh_agents.keys())))
        agents_data = []
        for name in all_names:
            role = configured.get(name, {}).get("role", "n/a")
            if len(role) > 20:
                role = role[:17] + "..."
            state = "stopped"
            tasks = 0
            cost = 0.0
            if name in mesh_agents:
                agent_url = mesh_agents[name]
                if isinstance(agent_url, dict):
                    agent_url = agent_url.get("url", "")
                try:
                    sr = httpx.get(f"{agent_url}/status", timeout=3)
                    sdata = sr.json()
                    state = sdata.get("state", "running")
                    tasks = sdata.get("tasks_completed", 0)
                except Exception:
                    state = "unreachable"
                # Fetch cost data only when wide output is requested
                if wide:
                    try:
                        cr = httpx.get(
                            f"http://localhost:{port}/dashboard/api/agents/{name}",
                            timeout=3,
                        )
                        cdata = cr.json()
                        cost = cdata.get("spend_today", {}).get("cost", 0)
                    except Exception:
                        pass
            agents_data.append({
                "name": name, "role": role, "status": state,
                "tasks": tasks, "cost": cost,
            })
        return agents_data, mesh_online, configured

    def _print_status(agents_data, mesh_online, _configured):
        if not agents_data:
            click.echo("No agents configured. Run: openlegion start")
            return
        if wide:
            click.echo(
                f"{'Agent':<16} {'Role':<20} {'Status':<12} {'Tasks':<8} {'Cost'}"
            )
            click.echo("-" * 65)
            for a in agents_data:
                cost_str = f"${a['cost']:.4f}" if a["cost"] else "-"
                click.echo(
                    f"{a['name']:<16} {a['role']:<20} {a['status']:<12} "
                    f"{a['tasks']:<8} {cost_str}"
                )
        else:
            click.echo(f"{'Agent':<16} {'Role':<20} {'Status':<12}")
            click.echo("-" * 48)
            for a in agents_data:
                click.echo(f"{a['name']:<16} {a['role']:<20} {a['status']:<12}")
        if not mesh_online:
            click.echo("\nMesh is not running. Start with: openlegion start")

    if as_json:
        agents_data, mesh_online, _ = _collect_status()
        click.echo(_json.dumps({"agents": agents_data, "mesh_online": mesh_online}))
        return

    if watch_interval:
        import time

        try:
            while True:
                click.clear()
                agents_data, mesh_online, configured = _collect_status()
                _print_status(agents_data, mesh_online, configured)
                click.echo(
                    f"\nRefreshing every {watch_interval}s... (Ctrl+C to stop)"
                )
                time.sleep(watch_interval)
        except KeyboardInterrupt:
            pass
        return

    agents_data, mesh_online, configured = _collect_status()
    _print_status(agents_data, mesh_online, configured)


# ── stop ─────────────────────────────────────────────────────

@cli.command()
def stop():
    """Stop the runtime and all agent containers."""
    import signal
    import time

    host_stopped = False

    # Stop background host process if running — it will shut down containers
    pid_path = cli_config.PROJECT_ROOT / ".openlegion.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            click.echo(f"Stopping host process (PID {pid})...", nl=False)
            # Wait for the host to finish its shutdown (which stops containers)
            for _ in range(20):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)  # check if still alive
                except ProcessLookupError:
                    break
            click.echo(" done.")
            host_stopped = True
        except (ProcessLookupError, ValueError):
            pass  # already dead or invalid
        finally:
            pid_path.unlink(missing_ok=True)

    # Clean up any containers the host didn't get to
    try:
        import docker

        client = docker.from_env()
        containers = client.containers.list(filters={"name": "openlegion_"})
        if not containers:
            if not host_stopped:
                click.echo("No OpenLegion containers running.")
            return
        for container in containers:
            try:
                click.echo(f"Stopping {container.name}...")
                container.stop(timeout=10)
                container.remove()
            except docker.errors.NotFound:
                pass  # already removed
        click.echo(f"Stopped {len(containers)} remaining container(s).")
    except Exception as e:
        if not host_stopped:
            click.echo(f"Could not connect to Docker: {e}")
            click.echo("If agents are running in Docker, ensure Docker is available.")
        else:
            logger.debug("Docker cleanup skipped: %s", e)


# ── version ──────────────────────────────────────────────────

@cli.command("version")
@click.option("--verbose", "-v", is_flag=True)
def version_cmd(verbose: bool):
    """Show version and environment information."""
    import platform
    from importlib.metadata import version as pkg_version

    try:
        ver = pkg_version("openlegion")
    except Exception:
        ver = "dev"
    click.echo(f"OpenLegion v{ver}")
    if verbose:
        click.echo(f"Python {sys.version.split()[0]}")
        try:
            import docker

            docker_ver = docker.from_env().version().get("Version", "unknown")
            click.echo(f"Docker {docker_ver}")
        except Exception:
            click.echo("Docker: not available")
        click.echo(
            f"OS: {platform.system()} {platform.release()} ({platform.machine()})"
        )
        click.echo(f"Config: {cli_config.CONFIG_FILE}")
        cfg = _load_config()
        n_agents = len(cfg.get("agents", {}))
        click.echo(f"Agents: {n_agents} configured")


if __name__ == "__main__":
    cli()
