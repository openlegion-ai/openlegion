"""CLI entry point for OpenLegion.

Core:
  setup                    First-time setup (API key, model, agents)
  start                    Start runtime + interactive REPL
  start -d                 Start in background (detached)
  stop                     Stop all containers
  status                   Show agent status
  chat <name>              Connect to a running agent

Agent management:
  agent add [name]         Add a new agent
  agent list               List configured agents
  agent model <name>       Change an agent's model
  agent browser <name>     Change an agent's browser backend
  agent remove <name>      Remove an agent
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading

import click
import yaml

from src.cli import config as cli_config
from src.cli.config import (
    CHANNEL_TYPES,
    _create_agent,
    _get_default_model,
    _load_config,
    _load_permissions,
    _save_permissions,
    _set_env_key,
    _suppress_host_logs,
)

logger = logging.getLogger("cli")


# ── Main group ───────────────────────────────────────────────

@click.group()
def cli():
    """OpenLegion -- Autonomous AI agent fleet."""
    from dotenv import load_dotenv

    load_dotenv(cli_config.ENV_FILE)
    _suppress_host_logs()


# ── setup ────────────────────────────────────────────────────

@cli.command()
def setup():
    """Interactive setup: API key, project, agents, collaboration."""
    from src.setup_wizard import SetupWizard
    wizard = SetupWizard(cli_config.PROJECT_ROOT)
    wizard.run_full()


# ── agent subgroup ───────────────────────────────────────────

@cli.group(invoke_without_command=True)
@click.pass_context
def agent(ctx):
    """Manage agents (add, list, model, browser, remove)."""
    if ctx.invoked_subcommand is None:
        commands = [
            ("list", "List configured agents"),
            ("add", "Add a new agent"),
            ("model", "Change an agent's model"),
            ("browser", "Change an agent's browser"),
            ("remove", "Remove an agent"),
        ]
        click.echo("Agent management:\n")
        for i, (name, desc) in enumerate(commands, 1):
            click.echo(f"  {i}. {name:<10} {desc}")
        choice = click.prompt(
            "\nSelect action",
            type=click.IntRange(1, len(commands)),
            default=1,
        )
        ctx.invoke(agent.commands[commands[choice - 1][0]])


@agent.command("add")
@click.argument("name", required=False, default=None)
@click.option("--model", "model_override", default=None, help="LLM model")
@click.option("--browser", "browser_override", default=None,
              type=click.Choice(["basic", "stealth", "advanced"]),
              help="Browser backend")
def agent_add(name: str | None, model_override: str | None, browser_override: str | None):
    """Add a new agent with model and browser selection.

    \b
    Examples:
      openlegion agent add researcher
      openlegion agent add coder --model anthropic/claude-sonnet-4-6
      openlegion agent add scraper --browser stealth
      openlegion agent add              # interactive
    """
    from src.cli.config import _pick_browser_interactive, _pick_model_interactive

    cfg = _load_config()

    if name is None:
        name = click.prompt("Agent name")

    if name in cfg.get("agents", {}):
        click.echo(f"Agent '{name}' already exists.")
        return

    description = click.prompt(
        "What should this agent do?",
        default=f"General-purpose {name} agent",
    )

    default_model = _get_default_model()
    model = model_override or _pick_model_interactive(default_model, label="default")

    browser = browser_override or _pick_browser_interactive()

    # Prompt for Bright Data CDP URL if advanced browser selected
    if browser == "advanced":
        _prompt_brightdata_key()

    _create_agent(name, description, model, browser_backend=browser)

    click.echo(f"\nAgent '{name}' created.")
    click.echo(f"  Model:   {model}")
    click.echo(f"  Browser: {browser}")
    click.echo("\nStart chatting: openlegion start")


@agent.command("model")
@click.argument("name", required=False, default=None)
@click.argument("model", required=False, default=None)
def agent_model(name: str | None, model: str | None):
    """Change an agent's LLM model.

    \b
    Examples:
      openlegion agent model assistant anthropic/claude-sonnet-4-6
      openlegion agent model assistant    # interactive picker
      openlegion agent model              # pick agent then model
    """
    from src.cli.config import _pick_model_interactive

    cfg = _load_config()
    name = _resolve_agent_name(cfg, name)
    if name is None:
        return

    current_model = cfg["agents"][name].get("model", _get_default_model())

    if model is None:
        click.echo(f"  Current: {current_model}\n")
        model = _pick_model_interactive(current_model, label="current")

    if model == current_model:
        click.echo(f"Agent '{name}' already uses {model}.")
        return

    _update_agent_field(name, "model", model)
    click.echo(f"Agent '{name}' model: {current_model} -> {model}")
    click.echo("Restart to apply: openlegion start")


@agent.command("browser")
@click.argument("name", required=False, default=None)
@click.argument("backend", required=False, default=None,
                type=click.Choice(["basic", "stealth", "advanced"]))
def agent_browser(name: str | None, backend: str | None):
    """Change an agent's browser backend.

    \b
    Backends:
      basic      Built-in Chromium (default)
      stealth    Camoufox anti-fingerprint browser
      advanced   Bright Data cloud proxy (requires CDP URL)

    \b
    Examples:
      openlegion agent browser scraper stealth
      openlegion agent browser scraper    # interactive picker
      openlegion agent browser            # pick agent then browser
    """
    from src.cli.config import _pick_browser_interactive

    cfg = _load_config()
    name = _resolve_agent_name(cfg, name)
    if name is None:
        return

    current = cfg["agents"][name].get("browser_backend", "basic") or "basic"

    if backend is None:
        click.echo(f"  Current: {current}\n")
        backend = _pick_browser_interactive(current)

    if backend == current:
        click.echo(f"Agent '{name}' already uses {backend} browser.")
        return

    if backend == "advanced":
        _prompt_brightdata_key()

    _update_agent_field(name, "browser_backend", backend)
    click.echo(f"Agent '{name}' browser: {current} -> {backend}")
    click.echo("Restart to apply: openlegion start")


def _update_agent_field(name: str, field: str, value: str) -> None:
    """Update a single field in agents.yaml for an agent."""
    if cli_config.AGENTS_FILE.exists():
        with open(cli_config.AGENTS_FILE) as f:
            agents_cfg = yaml.safe_load(f) or {"agents": {}}
    else:
        agents_cfg = {"agents": {}}
    if name in agents_cfg.get("agents", {}):
        agents_cfg["agents"][name][field] = value
        with open(cli_config.AGENTS_FILE, "w") as f:
            yaml.dump(agents_cfg, f, default_flow_style=False, sort_keys=False)


def _prompt_brightdata_key() -> None:
    """Prompt for Bright Data CDP URL if not already set."""
    existing = os.environ.get("OPENLEGION_CRED_BRIGHTDATA_CDP_URL", "")
    if existing:
        click.echo("  Bright Data CDP URL already configured.")
        return
    click.echo(
        "\n  Advanced browser requires a Bright Data Scraping Browser CDP URL.\n"
        "  Get one at: https://brightdata.com/products/scraping-browser\n"
    )
    cdp_url = click.prompt("  Bright Data CDP URL (wss://...)", default="", show_default=False)
    if cdp_url.strip():
        _set_env_key("brightdata_cdp_url", cdp_url.strip())
        click.echo("  Saved.\n")
    else:
        click.echo("  Skipped. Set OPENLEGION_CRED_BRIGHTDATA_CDP_URL in .env later.\n")


def _resolve_agent_name(cfg: dict, name: str | None) -> str | None:
    """Resolve agent name: return as-is if valid, prompt interactively if None."""
    agents = cfg.get("agents", {})
    if not agents:
        click.echo("No agents configured. Run: openlegion agent add")
        return None
    if name is not None:
        if name not in agents:
            click.echo(f"Agent '{name}' not found.")
            return None
        return name
    # Interactive picker
    names = sorted(agents.keys())
    if len(names) == 1:
        return names[0]
    click.echo("Agents:")
    for i, n in enumerate(names, 1):
        click.echo(f"  {i}. {n}")
    choice = click.prompt("Select agent", type=click.IntRange(1, len(names)), default=1)
    return names[choice - 1]


@agent.command("list")
def agent_list():
    """List all configured agents and their status."""
    cfg = _load_config()
    agents = cfg.get("agents", {})
    if not agents:
        click.echo("No agents configured. Add one: openlegion agent add")
        return

    running = set()
    try:
        import docker
        client = docker.from_env()
        containers = client.containers.list(filters={"name": "openlegion_"})
        for c in containers:
            agent_id = c.name.replace("openlegion_", "")
            running.add(agent_id)
    except Exception:
        pass

    default_model = _get_default_model()
    click.echo(f"{'Name':<16} {'Model':<40} {'Status':<10}")
    click.echo("-" * 66)
    for name, info in agents.items():
        status = "running" if name in running else "stopped"
        model = info.get("model", default_model)
        click.echo(f"{name:<16} {model:<40} {status:<10}")


@agent.command("remove")
@click.argument("name", required=False, default=None)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def agent_remove(name: str | None, yes: bool):
    """Remove an agent from configuration."""
    cfg = _load_config()
    name = _resolve_agent_name(cfg, name)
    if name is None:
        return
    if not yes:
        click.confirm(f"Remove agent '{name}'? This deletes its config and permissions.", abort=True)

    if cli_config.AGENTS_FILE.exists():
        with open(cli_config.AGENTS_FILE) as f:
            agents_cfg = yaml.safe_load(f) or {}
        agents_cfg.get("agents", {}).pop(name, None)
        with open(cli_config.AGENTS_FILE, "w") as f:
            yaml.dump(agents_cfg, f, default_flow_style=False, sort_keys=False)

    perms = _load_permissions()
    perms.get("permissions", {}).pop(name, None)
    _save_permissions(perms)

    click.echo(f"Removed agent '{name}'.")


# ── channels subgroup ────────────────────────────────────────

@cli.group()
def channels():
    """Connect Telegram, Discord, or other messaging channels."""
    pass


@channels.command("add")
@click.argument("channel_type", required=False, default=None)
def channels_add(channel_type: str | None):
    """Connect a messaging channel.

    \b
    Examples:
      openlegion channels add telegram
      openlegion channels add discord
      openlegion channels add           # interactive
    """
    if channel_type is None:
        click.echo("Available channels:\n")
        for i, (key, info) in enumerate(CHANNEL_TYPES.items(), 1):
            click.echo(f"  {i}. {info['label']}")
        click.echo("")
        choice = click.prompt("Select channel", type=click.IntRange(1, len(CHANNEL_TYPES)), default=1)
        channel_type = list(CHANNEL_TYPES.keys())[choice - 1]

    channel_type = channel_type.lower()
    if channel_type not in CHANNEL_TYPES:
        click.echo(f"Unknown channel '{channel_type}'. Available: {', '.join(CHANNEL_TYPES)}")
        return

    ch = CHANNEL_TYPES[channel_type]
    click.echo(f"\n  {ch['label']} Setup")
    click.echo(f"  {ch['token_help']}\n")

    token = click.prompt(f"  {ch['label']} bot token", hide_input=True)
    if not token.strip():
        click.echo("  No token provided. Skipped.")
        return

    _set_env_key(ch["env_key"], token.strip())

    # Slack needs a second token
    if channel_type == "slack":
        click.echo("\n  Socket Mode requires an app-level token (xapp-...).")
        app_token = click.prompt("  Slack app-level token", hide_input=True)
        if app_token.strip():
            _set_env_key("slack_app_token", app_token.strip())
        else:
            click.echo("  No app token provided. Socket Mode will not work.")
            return

    # WhatsApp needs a phone number ID
    if channel_type == "whatsapp":
        phone_id = click.prompt("  WhatsApp phone number ID")
        if phone_id.strip():
            _set_env_key("whatsapp_phone_number_id", phone_id.strip())
        else:
            click.echo("  No phone number ID provided.")
            return

    # Enable in mesh config
    mesh_cfg = {}
    if cli_config.CONFIG_FILE.exists():
        with open(cli_config.CONFIG_FILE) as f:
            mesh_cfg = yaml.safe_load(f) or {}
    mesh_cfg.setdefault("channels", {}).setdefault(ch["config_section"], {})
    mesh_cfg["channels"][ch["config_section"]]["enabled"] = True
    with open(cli_config.CONFIG_FILE, "w") as f:
        yaml.dump(mesh_cfg, f, default_flow_style=False, sort_keys=False)

    click.echo(f"\n  {ch['label']} connected. A pairing code will appear on next `openlegion start`.")


@channels.command("list")
def channels_list():
    """Show configured channels and their status."""
    mesh_cfg = {}
    if cli_config.CONFIG_FILE.exists():
        with open(cli_config.CONFIG_FILE) as f:
            mesh_cfg = yaml.safe_load(f) or {}

    channel_cfg = mesh_cfg.get("channels", {})
    if not channel_cfg:
        click.echo("No channels configured. Add one: openlegion channels add")
        return

    click.echo(f"{'Channel':<16} {'Status':<12}")
    click.echo("-" * 28)
    for key, info in CHANNEL_TYPES.items():
        section = channel_cfg.get(info["config_section"], {})
        env_key = f"OPENLEGION_CRED_{info['env_key'].upper()}"
        has_token = bool(os.environ.get(env_key))
        if section.get("enabled"):
            status = "ready" if has_token else "no token"
            click.echo(f"{info['label']:<16} {status:<12}")


@channels.command("remove")
@click.argument("channel_type")
def channels_remove(channel_type: str):
    """Disconnect a messaging channel.

    \b
    Examples:
      openlegion channels remove telegram
      openlegion channels remove discord
    """
    channel_type = channel_type.lower()
    if channel_type not in CHANNEL_TYPES:
        click.echo(f"Unknown channel '{channel_type}'. Available: {', '.join(CHANNEL_TYPES)}")
        return

    ch = CHANNEL_TYPES[channel_type]

    mesh_cfg = {}
    if cli_config.CONFIG_FILE.exists():
        with open(cli_config.CONFIG_FILE) as f:
            mesh_cfg = yaml.safe_load(f) or {}
    channels_section = mesh_cfg.get("channels", {})
    if ch["config_section"] in channels_section:
        del channels_section[ch["config_section"]]
        with open(cli_config.CONFIG_FILE, "w") as f:
            yaml.dump(mesh_cfg, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Removed {ch['label']} channel.")
    click.echo(f"  Token remains in .env — delete the OPENLEGION_CRED_{ch['env_key'].upper()} line to fully remove.")


# ── skill marketplace ─────────────────────────────────────────

@cli.group()
def skill():
    """Install, list, or remove marketplace skills."""
    pass


@skill.command("install")
@click.argument("repo_url")
@click.option("--ref", default="", help="Git ref to pin (tag, branch, or commit)")
def skill_install(repo_url: str, ref: str):
    """Install a skill from a git repository.

    \b
    Examples:
      openlegion skill install https://github.com/user/my-skill
      openlegion skill install https://github.com/user/my-skill --ref v1.0.0
    """
    from src.marketplace import install_skill

    click.echo(f"Installing skill from {repo_url}...")
    result = install_skill(repo_url, cli_config.MARKETPLACE_DIR, ref=ref)
    if "error" in result:
        click.echo(f"Error: {result['error']}", err=True)
        return
    click.echo(f"Installed '{result['name']}' v{result.get('version', '?')}")
    click.echo(f"  {result.get('description', '')}")
    click.echo("\nRestart agents to load the new skill.")


@skill.command("list")
def skill_list():
    """List installed marketplace skills."""
    from src.marketplace import list_skills

    skills = list_skills(cli_config.MARKETPLACE_DIR)
    if not skills:
        click.echo("No marketplace skills installed.")
        click.echo("Install one: openlegion skill install <repo_url>")
        return

    click.echo(f"{'Name':<20} {'Version':<12} {'Description'}")
    click.echo("-" * 60)
    for s in skills:
        click.echo(f"{s.get('name', '?'):<20} {s.get('version', '?'):<12} {s.get('description', '')[:40]}")


@skill.command("remove")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def skill_remove(name: str, yes: bool):
    """Remove an installed marketplace skill.

    \b
    Examples:
      openlegion skill remove my-skill
      openlegion skill remove my-skill -y
    """
    from src.marketplace import remove_skill

    if not yes:
        click.confirm(f"Remove skill '{name}'?", abort=True)

    result = remove_skill(name, cli_config.MARKETPLACE_DIR)
    if "error" in result:
        click.echo(f"Error: {result['error']}", err=True)
        return
    click.echo(f"Removed skill '{name}'.")
    click.echo("Restart agents for changes to take effect.")


# ── start ────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "config_path", default="config/mesh.yaml", help="Path to mesh config")
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
        if serve:
            # Headless mode: keep alive via signal wait (used by -d)
            import signal
            shutdown_event = threading.Event()
            signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())
            signal.signal(signal.SIGINT, lambda *_: shutdown_event.set())
            shutdown_event.wait()
        else:
            from src.cli.repl import REPLSession
            repl = REPLSession(ctx)
            repl.run()
    except KeyboardInterrupt:
        click.echo("")
    finally:
        ctx.shutdown()


def _start_detached(config_path: str) -> None:
    """Start the runtime in a background subprocess.

    Uses a log file instead of a pipe for child stdout.  This prevents
    the pipe buffer from filling up after the parent exits, which would
    cause SIGPIPE to kill the background process and terminal lag while
    the buffer is filling.
    """
    import time

    log_path = cli_config.PROJECT_ROOT / ".openlegion.log"
    log_fd = open(log_path, "w")

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

    log_fd.close()

    if proc.poll() is not None:
        # Process died during startup
        try:
            lines = log_path.read_text().splitlines()
            for line in lines[printed_lines:]:
                click.echo(f"  {line}", err=True)
        except OSError:
            pass
        click.echo("Runtime failed to start. Check logs.", err=True)
        click.echo(f"  Log: {log_path}", err=True)
        return

    if not ready:
        click.echo("Startup timed out. Check logs.", err=True)
        click.echo(f"  Log: {log_path}", err=True)
        return

    # Write PID file so `openlegion stop` can kill the host process
    pid_path = cli_config.PROJECT_ROOT / ".openlegion.pid"
    pid_path.write_text(str(proc.pid))

    click.echo(f"\nOpenLegion running in background (PID {proc.pid}).")
    click.echo(f"  Log file:            {log_path}")
    click.echo("  Chat with an agent:  openlegion chat <name>")
    click.echo("  Stop the runtime:    openlegion stop")


# ── chat (for detached mode) ─────────────────────────────────

@cli.command("chat")
@click.argument("name", required=False, default=None)
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
        click.echo("Mesh is not running. Start it first: openlegion start", err=True)
        return
    except Exception as e:
        click.echo(f"Error contacting mesh: {e}", err=True)
        return

    if not agents:
        click.echo("No agents running.", err=True)
        return

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
        click.echo(f"Agent '{name}' is not running. Running agents: {available}", err=True)
        return

    agent_url = agent_info.get("url", agent_info) if isinstance(agent_info, dict) else agent_info
    click.echo(f"Connected to '{name}' at {agent_url}")
    click.echo("Type a message to chat. /help for commands.\n")

    try:
        _single_agent_repl(name, agent_url)
    except KeyboardInterrupt:
        click.echo("\nDisconnected.")


def _single_agent_repl(agent_name: str, agent_url: str) -> None:
    """Interactive chat REPL with a single agent (for detached mode)."""
    import httpx

    from src.cli.formatting import display_response, user_prompt

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
                    click.echo(f"Error: {e}")
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
                    click.echo(f"Error: {e}")
                continue
            elif cmd == "/help":
                click.echo("Commands:")
                click.echo(f"  {'/reset':<18} Clear conversation history")
                click.echo(f"  {'/status':<18} Show agent status")
                click.echo(f"  {'/quit':<18} Exit chat")
                click.echo(f"  {'/help':<18} Show this help")
                continue
            else:
                click.echo(f"Unknown command: {cmd}. Type /help for commands.")
                continue

        try:
            resp = httpx.post(
                f"{agent_url}/chat",
                json={"message": user_input},
                timeout=120,
            )
            if resp.status_code != 200:
                click.echo(f"Error: HTTP {resp.status_code}: {resp.text}")
                continue

            data = resp.json()
            display_response(agent_name, data)

        except httpx.TimeoutException:
            click.echo("Request timed out. The agent may still be processing.")
        except Exception as e:
            click.echo(f"Error: {e}")


# ── status ───────────────────────────────────────────────────

@cli.command("status")
@click.option("--port", default=8420, type=int, help="Mesh host port")
def status(port: int):
    """Show status of all agents."""
    import httpx

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

    if not configured and not mesh_agents:
        click.echo("No agents configured. Run: openlegion setup")
        return

    all_names = sorted(set(list(configured.keys()) + list(mesh_agents.keys())))

    click.echo(f"{'Agent':<16} {'Role':<20} {'Status':<12}")
    click.echo("-" * 48)
    for name in all_names:
        role = configured.get(name, {}).get("role", "n/a")
        # Truncate long role names to keep columns aligned
        if len(role) > 20:
            role = role[:17] + "..."

        if name in mesh_agents:
            agent_url = mesh_agents[name]
            if isinstance(agent_url, dict):
                agent_url = agent_url.get("url", "")
            try:
                sr = httpx.get(f"{agent_url}/status", timeout=3)
                state = sr.json().get("state", "running")
            except Exception:
                state = "unreachable"
        else:
            state = "stopped"

        click.echo(f"{name:<16} {role:<20} {state:<12}")

    if not mesh_online:
        click.echo("\nMesh is not running. Start with: openlegion start")


# ── stop ─────────────────────────────────────────────────────

@cli.command()
def stop():
    """Stop the runtime and all agent containers."""
    import signal
    import time

    import docker

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


if __name__ == "__main__":
    cli()
