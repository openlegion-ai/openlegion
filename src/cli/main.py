"""CLI entry point for OpenLegion.

Core:
  start                    Start runtime + interactive REPL
  start -d                 Start in background (detached)
  stop                     Stop all containers
  status                   Show agent status
  chat <name>              Connect to a running agent
  logs [-f] [--level]      View runtime logs
  health                   Pre-flight health checks
  version [-v]             Show version info

Configuration:
  config show [--json]     Show effective configuration
  config validate          Check for config errors
  config path              Show config file locations

Agent management:
  agent add [name]         Add a new agent
  agent list               List configured agents
  agent edit [name]        Change an agent's settings
  agent remove [name]      Remove an agent
  agent restart [names]    Restart one or more agents

Webhooks:
  webhook list             List configured webhooks
  webhook test <name>      Send a test payload to a webhook
"""

from __future__ import annotations

import json as _json
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
    _default_description,
    _get_default_model,
    _load_config,
    _remove_agent,
    _set_env_key,
    _suppress_host_logs,
    _update_agent_field,
)

logger = logging.getLogger("cli")


def _fail(msg: str) -> None:
    """Print error message to stderr and exit with code 1."""
    click.echo(msg, err=True)
    raise SystemExit(1)


# ── Main group ───────────────────────────────────────────────

@click.group()
@click.version_option(package_name="openlegion")
def cli():
    """OpenLegion -- Autonomous AI agent fleet."""
    from dotenv import load_dotenv

    load_dotenv(cli_config.ENV_FILE)
    _suppress_host_logs()


# ── agent subgroup ───────────────────────────────────────────

@cli.group(invoke_without_command=True)
@click.pass_context
def agent(ctx):
    """Manage agents (add, list, edit, remove)."""
    if ctx.invoked_subcommand is None:
        commands = [
            ("list", "List configured agents"),
            ("add", "Add a new agent"),
            ("edit", "Change an agent's settings"),
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
def agent_add(name: str | None, model_override: str | None):
    """Add a new agent.

    \b
    Examples:
      openlegion agent add researcher
      openlegion agent add coder --model anthropic/claude-sonnet-4-6
      openlegion agent add              # interactive
    """
    from src.cli.config import _pick_model_interactive

    cfg = _load_config()

    if name is None:
        name = click.prompt("Agent name")

    if name in cfg.get("agents", {}):
        _fail(f"Agent '{name}' already exists.")

    description = click.prompt(
        "What should this agent do?",
        default=_default_description(name),
    )

    default_model = _get_default_model()
    model = model_override or _pick_model_interactive(default_model, label="default")

    _create_agent(name, description, model)

    click.echo(f"\nAgent '{name}' created.")
    click.echo(f"  Model: {model}")
    click.echo("\nStart chatting: openlegion start")


@agent.command("edit")
@click.argument("name", required=False, default=None)
@click.option("--model", "model_override", default=None, help="Set LLM model")
@click.option("--description", "desc_override", default=None, help="Set role/description")
@click.option("--budget", "budget_override", default=None, type=click.FloatRange(min=0), help="Set daily budget (USD)")
def agent_edit(
    name: str | None,
    model_override: str | None,
    desc_override: str | None,
    budget_override: float | None,
):
    """Change an agent's settings.

    \b
    Examples:
      openlegion agent edit mybot --model anthropic/claude-sonnet-4-6
      openlegion agent edit mybot --description "Web research specialist"
      openlegion agent edit mybot --budget 10.0
      openlegion agent edit mybot           # interactive property picker
      openlegion agent edit                 # pick agent then property
    """
    from src.cli.config import _edit_agent_interactive

    cfg = _load_config()
    name = _resolve_agent_name(cfg, name)
    if name is None:
        raise SystemExit(1)

    # Direct flag mode: apply each provided flag
    has_flags = any(v is not None for v in [
        model_override, desc_override,
        budget_override,
    ])
    if has_flags:
        agent_cfg = cfg["agents"][name]
        default_model = _get_default_model()
        changed = False

        if model_override is not None:
            old = agent_cfg.get("model", default_model)
            if model_override == old:
                click.echo(f"Agent '{name}' already uses {model_override}.")
            else:
                _update_agent_field(name, "model", model_override)
                click.echo(f"Agent '{name}' model: {old} -> {model_override}")
                changed = True

        if desc_override is not None:
            _update_agent_field(name, "role", desc_override)
            click.echo(f"Agent '{name}' description updated.")
            changed = True

        if budget_override is not None:
            _update_agent_field(name, "budget", {"daily_usd": budget_override})
            click.echo(f"Agent '{name}' budget: ${budget_override:.2f}/day")
            changed = True

        if changed:
            click.echo("Restart to apply: openlegion start")
        return

    # Interactive mode
    changed_field = _edit_agent_interactive(name)
    if changed_field:
        click.echo("Restart to apply: openlegion start")


def _resolve_agent_name(cfg: dict, name: str | None) -> str | None:
    """Resolve agent name: return as-is if valid, prompt interactively if None."""
    agents = cfg.get("agents", {})
    if not agents:
        click.echo("No agents configured. Run: openlegion agent add", err=True)
        return None
    if name is not None:
        if name not in agents:
            click.echo(f"Agent '{name}' not found. Available: {', '.join(sorted(agents))}", err=True)
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
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def agent_list(as_json: bool):
    """List all configured agents and their status."""
    cfg = _load_config()
    agents = cfg.get("agents", {})
    if not agents:
        if as_json:
            click.echo(_json.dumps({"agents": []}))
            return
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
    except Exception as e:
        logger.debug("Docker container list failed: %s", e)

    default_model = _get_default_model()

    if as_json:
        agents_data = []
        for name, info in agents.items():
            status = "running" if name in running else "stopped"
            model = info.get("model", default_model)
            agents_data.append({"name": name, "model": model, "status": status})
        click.echo(_json.dumps({"agents": agents_data}))
        return

    click.echo(f"{'Name':<16} {'Model':<40} {'Status':<10}")
    click.echo("-" * 69)
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
        raise SystemExit(1)
    if not yes:
        click.confirm(f"Remove agent '{name}'? This deletes its config and permissions.", abort=True)

    _remove_agent(name)
    click.echo(f"Removed agent '{name}'.")


@agent.command("restart")
@click.argument("names", nargs=-1)
@click.option("--all", "restart_all", is_flag=True, help="Restart all agents")
@click.option("--port", default=8420, type=int)
def agent_restart(names: tuple, restart_all: bool, port: int):
    """Restart one or more agents.

    \b
    Examples:
      openlegion agent restart researcher coder
      openlegion agent restart --all
    """
    import httpx

    try:
        resp = httpx.get(f"http://localhost:{port}/mesh/agents", timeout=5)
        running_agents = resp.json()
    except httpx.ConnectError:
        click.echo("Mesh is not running. Start it first: openlegion start", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error contacting mesh: {e}", err=True)
        raise SystemExit(1)

    if restart_all:
        targets = list(running_agents.keys())
    else:
        targets = list(names)

    if not targets:
        click.echo("No agents specified. Use agent names or --all.")
        raise SystemExit(1)

    errors = 0
    for name in targets:
        if name not in running_agents:
            click.echo(f"Agent '{name}' is not running. Skipping.")
            continue
        try:
            resp = httpx.post(
                f"http://localhost:{port}/dashboard/api/agents/{name}/restart",
                timeout=120,
            )
            if resp.status_code == 200:
                click.echo(f"Restarted '{name}'.")
            else:
                click.echo(f"Failed to restart '{name}': {resp.text}", err=True)
                errors += 1
        except Exception as e:
            click.echo(f"Error restarting '{name}': {e}", err=True)
            errors += 1
    if errors:
        raise SystemExit(1)


# ── channels subgroup ────────────────────────────────────────

@cli.group(invoke_without_command=True)
@click.pass_context
def channels(ctx):
    """Connect Telegram, Discord, or other messaging channels."""
    if ctx.invoked_subcommand is None:
        commands = [
            ("add", "Connect a messaging channel"),
            ("list", "Show configured channels"),
            ("remove", "Disconnect a channel"),
        ]
        click.echo("Channel management:\n")
        for i, (name, desc) in enumerate(commands, 1):
            click.echo(f"  {i}. {name:<10} {desc}")
        choice = click.prompt(
            "\nSelect action",
            type=click.IntRange(1, len(commands)),
            default=1,
        )
        ctx.invoke(channels.commands[commands[choice - 1][0]])


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
        _fail(f"Unknown channel '{channel_type}'. Available: {', '.join(CHANNEL_TYPES)}")

    ch = CHANNEL_TYPES[channel_type]
    click.echo(f"\n  {ch['label']} Setup")
    click.echo(f"  {ch['token_help']}\n")

    token = click.prompt(f"  {ch['label']} bot token", hide_input=True)
    if not token.strip():
        _fail("No token provided.")

    _set_env_key(ch["env_key"], token.strip(), system=True)

    # Slack needs a second token
    if channel_type == "slack":
        click.echo("\n  Socket Mode requires an app-level token (xapp-...).")
        app_token = click.prompt("  Slack app-level token", hide_input=True)
        if app_token.strip():
            _set_env_key("slack_app_token", app_token.strip(), system=True)
        else:
            click.echo("  No app token provided. Socket Mode will not work.")
            return

    # WhatsApp needs a phone number ID
    if channel_type == "whatsapp":
        phone_id = click.prompt("  WhatsApp phone number ID")
        if phone_id.strip():
            _set_env_key("whatsapp_phone_number_id", phone_id.strip(), system=True)
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
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def channels_list(as_json: bool):
    """Show configured channels and their status."""
    mesh_cfg = {}
    if cli_config.CONFIG_FILE.exists():
        with open(cli_config.CONFIG_FILE) as f:
            mesh_cfg = yaml.safe_load(f) or {}

    channel_cfg = mesh_cfg.get("channels", {})
    if not channel_cfg:
        if as_json:
            click.echo(_json.dumps({"channels": []}))
            return
        click.echo("No channels configured. Add one: openlegion channels add")
        return

    if as_json:
        channels_data = []
        for key, info in CHANNEL_TYPES.items():
            section = channel_cfg.get(info["config_section"], {})
            sys_key = f"OPENLEGION_SYSTEM_{info['env_key'].upper()}"
            cred_key = f"OPENLEGION_CRED_{info['env_key'].upper()}"
            has_token = bool(os.environ.get(sys_key) or os.environ.get(cred_key))
            if section.get("enabled"):
                ch_status = "ready" if has_token else "no token"
                channels_data.append(
                    {"type": key, "label": info["label"], "status": ch_status}
                )
        click.echo(_json.dumps({"channels": channels_data}))
        return

    click.echo(f"{'Channel':<16} {'Status':<12}")
    click.echo("-" * 28)
    for key, info in CHANNEL_TYPES.items():
        section = channel_cfg.get(info["config_section"], {})
        sys_key = f"OPENLEGION_SYSTEM_{info['env_key'].upper()}"
        cred_key = f"OPENLEGION_CRED_{info['env_key'].upper()}"
        has_token = bool(os.environ.get(sys_key) or os.environ.get(cred_key))
        if section.get("enabled"):
            ch_status = "ready" if has_token else "no token"
            click.echo(f"{info['label']:<16} {ch_status:<12}")


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
        _fail(f"Unknown channel '{channel_type}'. Available: {', '.join(CHANNEL_TYPES)}")

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
    env_name = ch['env_key'].upper()
    click.echo(
        f"  Token remains in .env — delete the OPENLEGION_SYSTEM_{env_name}"
        f" (or OPENLEGION_CRED_{env_name}) line to fully remove."
    )


# ── skill marketplace ─────────────────────────────────────────

@cli.group(invoke_without_command=True)
@click.pass_context
def skill(ctx):
    """Install, list, or remove marketplace skills."""
    if ctx.invoked_subcommand is None:
        commands = [
            ("list", "List installed skills"),
            ("install", "Install a skill from git"),
            ("remove", "Remove an installed skill"),
        ]
        click.echo("Skill management:\n")
        for i, (name, desc) in enumerate(commands, 1):
            click.echo(f"  {i}. {name:<10} {desc}")
        choice = click.prompt(
            "\nSelect action",
            type=click.IntRange(1, len(commands)),
            default=1,
        )
        ctx.invoke(skill.commands[commands[choice - 1][0]])


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
        _fail(f"Error: {result['error']}")
    click.echo(f"Installed '{result['name']}' v{result.get('version', '?')}")
    click.echo(f"  {result.get('description', '')}")
    click.echo("\nRestart agents to load the new skill.")


@skill.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def skill_list(as_json: bool):
    """List installed marketplace skills."""
    from src.marketplace import list_skills

    skills = list_skills(cli_config.MARKETPLACE_DIR)
    if as_json:
        click.echo(_json.dumps({"skills": skills}))
        return

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
        _fail(f"Error: {result['error']}")
    click.echo(f"Removed skill '{name}'.")
    click.echo("Restart agents for changes to take effect.")


# ── credential management ──────────────────────────────────

@cli.group(invoke_without_command=True)
@click.pass_context
def credential(ctx):
    """Manage API credentials (add, list, remove)."""
    if ctx.invoked_subcommand is None:
        commands = [
            ("add", "Add an API credential"),
            ("list", "List stored credentials"),
            ("remove", "Remove a credential"),
        ]
        click.echo("Credential management:\n")
        for i, (name, desc) in enumerate(commands, 1):
            click.echo(f"  {i}. {name:<10} {desc}")
        choice = click.prompt(
            "\nSelect action",
            type=click.IntRange(1, len(commands)),
            default=1,
        )
        ctx.invoke(credential.commands[commands[choice - 1][0]])


@credential.command("add")
@click.argument("service", required=False, default=None)
@click.option("--key", "api_key", default=None, help="API key value")
@click.option("--tier", type=click.Choice(["agent", "system"]), default="system", help="Credential tier")
@click.option("--base-url", default=None, help="Custom API base URL")
def credential_add(service: str | None, api_key: str | None, tier: str, base_url: str | None):
    """Add an API credential.

    \b
    Examples:
      openlegion credential add anthropic_api_key
      openlegion credential add openai_api_key --key sk-...
      openlegion credential add                 # interactive
    """
    from src.host.credentials import SYSTEM_CREDENTIAL_PROVIDERS

    if service is None:
        click.echo("Available providers:\n")
        providers = list(SYSTEM_CREDENTIAL_PROVIDERS)
        for i, p in enumerate(providers, 1):
            click.echo(f"  {i}. {p}")
        click.echo(f"  {len(providers) + 1}. (custom service name)")
        choice = click.prompt(
            "\nSelect provider",
            type=click.IntRange(1, len(providers) + 1),
            default=1,
        )
        if choice <= len(providers):
            service = providers[choice - 1]
        else:
            service = click.prompt("Service name")

    if api_key is None:
        api_key = click.prompt(f"API key for {service}", hide_input=True)

    if not api_key or not api_key.strip():
        click.echo("No key provided.", err=True)
        raise SystemExit(1)

    is_system = tier == "system"
    _set_env_key(service, api_key.strip(), system=is_system)

    if base_url:
        _set_env_key(f"{service}_base_url", base_url.strip(), system=is_system)

    click.echo(f"Credential '{service}' saved ({tier} tier).")
    click.echo("  Restart to apply: openlegion start")


@credential.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def credential_list(as_json: bool):
    """List stored credentials (keys are masked).

    \b
    Examples:
      openlegion credential list
      openlegion credential list --json
    """
    from src.host.credentials import AGENT_PREFIX, SYSTEM_PREFIX

    # Read from .env file
    creds = []
    env_file = cli_config.ENV_FILE
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key.startswith(SYSTEM_PREFIX) or key.startswith(AGENT_PREFIX):
                name = key
                if key.startswith(SYSTEM_PREFIX):
                    name = key[len(SYSTEM_PREFIX):]
                    cred_tier = "system"
                else:
                    name = key[len(AGENT_PREFIX):]
                    cred_tier = "agent"
                masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
                creds.append({"name": name.lower(), "tier": cred_tier, "masked_key": masked})

    if as_json:
        import json as _json
        click.echo(_json.dumps({"credentials": creds}))
        return

    if not creds:
        click.echo("No credentials stored. Add one: openlegion credential add")
        return

    click.echo(f"{'Name':<30} {'Tier':<10} {'Key'}")
    click.echo("-" * 55)
    for c in creds:
        click.echo(f"{c['name']:<30} {c['tier']:<10} {c['masked_key']}")


@credential.command("remove")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def credential_remove(name: str, yes: bool):
    """Remove a stored credential.

    \b
    Examples:
      openlegion credential remove anthropic_api_key
      openlegion credential remove openai_api_key -y
    """
    from src.host.credentials import AGENT_PREFIX, SYSTEM_PREFIX

    env_file = cli_config.ENV_FILE
    if not env_file.exists():
        click.echo(f"Credential '{name}' not found.", err=True)
        raise SystemExit(1)

    name_upper = name.upper()
    lines = env_file.read_text().splitlines()

    # First pass: check if credential exists
    target_keys = {f"{SYSTEM_PREFIX}{name_upper}", f"{AGENT_PREFIX}{name_upper}"}
    found = any(
        (stripped := line.strip())
        and "=" in stripped
        and stripped.split("=", 1)[0].strip() in target_keys
        for line in lines
    )

    if not found:
        click.echo(f"Credential '{name}' not found.", err=True)
        raise SystemExit(1)

    if not yes:
        click.confirm(f"Remove credential '{name}'?", abort=True)

    # Second pass: filter out matching lines
    new_lines = []
    for line in lines:
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        if key in target_keys:
            continue
        new_lines.append(line)

    env_file.write_text("\n".join(new_lines) + "\n" if new_lines else "")
    click.echo(f"Removed credential '{name}'.")
    click.echo("  Restart to apply: openlegion start")


# ── project subgroup ─────────────────────────────────────────

@cli.group(invoke_without_command=True)
@click.pass_context
def project(ctx):
    """Manage projects (create, list, delete, add-agent, remove-agent)."""
    if ctx.invoked_subcommand is None:
        commands = [
            ("list", "List all projects"),
            ("create", "Create a new project"),
            ("delete", "Delete a project"),
            ("add-agent", "Add an agent to a project"),
            ("remove-agent", "Remove an agent from a project"),
            ("edit", "Edit project description or context"),
        ]
        click.echo("Project management:\n")
        for i, (name, desc) in enumerate(commands, 1):
            click.echo(f"  {i}. {name:<16} {desc}")
        choice = click.prompt(
            "\nSelect action",
            type=click.IntRange(1, len(commands)),
            default=1,
        )
        ctx.invoke(project.commands[commands[choice - 1][0]])


@project.command("create")
@click.argument("name", required=False, default=None)
@click.option("--description", "-D", "desc", default="", help="Project description")
@click.option("--agents", "-a", "agents_str", default="", help="Comma-separated agent names")
def project_create(name: str | None, desc: str, agents_str: str):
    """Create a new project.

    \b
    Examples:
      openlegion project create my-project
      openlegion project create my-project -D "Marketing automation" -a agent1,agent2
      openlegion project create           # interactive
    """
    from src.cli.config import _create_project, _load_projects, _validate_project_name

    if name is None:
        name = click.prompt("Project name")

    try:
        _validate_project_name(name)
    except ValueError as e:
        _fail(str(e))

    existing = _load_projects()
    if name in existing:
        _fail(f"Project '{name}' already exists.")

    if not desc:
        desc = click.prompt("Description", default="")

    cfg = _load_config()
    members: list[str] = []
    if agents_str:
        members = [a.strip() for a in agents_str.split(",") if a.strip()]
    else:
        available = sorted(cfg.get("agents", {}).keys())
        if available:
            click.echo(f"\nAvailable agents: {', '.join(available)}")
            agents_input = click.prompt(
                "Add agents (comma-separated, or empty to skip)", default="",
            )
            if agents_input.strip():
                members = [a.strip() for a in agents_input.split(",") if a.strip()]

    # Validate agent names exist
    all_agents = set(cfg.get("agents", {}))
    invalid = [a for a in members if a not in all_agents]
    if invalid:
        _fail(f"Unknown agent(s): {', '.join(invalid)}")

    try:
        _create_project(name, description=desc, members=members)
    except ValueError as e:
        _fail(str(e))

    click.echo(f"\nProject '{name}' created.")
    if members:
        click.echo(f"  Members: {', '.join(members)}")
    click.echo("  Restart to apply: openlegion start")


@project.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def project_list(as_json: bool):
    """List all projects and their members."""
    from src.cli.config import _load_projects

    cfg = _load_config()
    projects = _load_projects()
    all_agents = set(cfg.get("agents", {}))
    assigned = set()
    for pdata in projects.values():
        assigned.update(pdata.get("members", []))
    standalone = sorted(all_agents - assigned)

    if as_json:
        click.echo(
            _json.dumps({"projects": projects, "standalone": standalone})
        )
        return

    if not projects and not standalone:
        click.echo("No projects or agents configured.")
        return

    if projects:
        click.echo(f"{'Project':<20} {'Members':<40} {'Description'}")
        click.echo("-" * 80)
        for pname, pdata in sorted(projects.items()):
            members = ", ".join(pdata.get("members", [])) or "(none)"
            desc = pdata.get("description", "")
            if len(desc) > 30:
                desc = desc[:27] + "..."
            click.echo(f"{pname:<20} {members:<40} {desc}")

    if standalone:
        click.echo(f"\nStandalone agents: {', '.join(standalone)}")


@project.command("delete")
@click.argument("name", required=False, default=None)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def project_delete(name: str | None, yes: bool):
    """Delete a project. Agents become standalone.

    \b
    Examples:
      openlegion project delete my-project
      openlegion project delete my-project -y
    """
    from src.cli.config import _delete_project, _load_projects

    projects = _load_projects()
    if not projects:
        click.echo("No projects configured.")
        return

    if name is None:
        names = sorted(projects.keys())
        for i, n in enumerate(names, 1):
            click.echo(f"  {i}. {n}")
        choice = click.prompt("Select project", type=click.IntRange(1, len(names)), default=1)
        name = names[choice - 1]

    if name not in projects:
        _fail(f"Project '{name}' not found.")

    if not yes:
        members = projects[name].get("members", [])
        msg = f"Delete project '{name}'?"
        if members:
            msg += f" Agents ({', '.join(members)}) will become standalone."
        click.confirm(msg, abort=True)

    try:
        _delete_project(name)
    except ValueError as e:
        _fail(str(e))

    click.echo(f"Deleted project '{name}'.")


@project.command("add-agent")
@click.argument("project_name", required=False, default=None)
@click.argument("agent_name", required=False, default=None)
def project_add_agent(project_name: str | None, agent_name: str | None):
    """Add an agent to a project.

    \b
    Examples:
      openlegion project add-agent my-project my-agent
      openlegion project add-agent   # interactive
    """
    from src.cli.config import _add_agent_to_project, _load_projects

    projects = _load_projects()
    if not projects:
        click.echo("No projects configured. Create one first: openlegion project create")
        return

    if project_name is None:
        names = sorted(projects.keys())
        if len(names) == 1:
            project_name = names[0]
        else:
            for i, n in enumerate(names, 1):
                click.echo(f"  {i}. {n}")
            choice = click.prompt("Select project", type=click.IntRange(1, len(names)), default=1)
            project_name = names[choice - 1]

    if project_name not in projects:
        _fail(f"Project '{project_name}' not found.")

    cfg = _load_config()
    if agent_name is None:
        agent_name = _resolve_agent_name(cfg, None)
    if agent_name is None:
        raise SystemExit(1)
    if agent_name not in cfg.get("agents", {}):
        _fail(f"Agent '{agent_name}' not found.")

    try:
        _add_agent_to_project(project_name, agent_name)
    except ValueError as e:
        _fail(str(e))

    click.echo(f"Added '{agent_name}' to project '{project_name}'.")
    click.echo("  Restart to apply: openlegion start")


@project.command("remove-agent")
@click.argument("project_name", required=False, default=None)
@click.argument("agent_name", required=False, default=None)
def project_remove_agent(project_name: str | None, agent_name: str | None):
    """Remove an agent from a project (becomes standalone).

    \b
    Examples:
      openlegion project remove-agent my-project my-agent
    """
    from src.cli.config import _load_projects, _remove_agent_from_project

    projects = _load_projects()
    if not projects:
        click.echo("No projects configured.")
        return

    if project_name is None:
        names = sorted(projects.keys())
        if len(names) == 1:
            project_name = names[0]
        else:
            for i, n in enumerate(names, 1):
                click.echo(f"  {i}. {n}")
            choice = click.prompt("Select project", type=click.IntRange(1, len(names)), default=1)
            project_name = names[choice - 1]

    if project_name not in projects:
        _fail(f"Project '{project_name}' not found.")

    members = projects[project_name].get("members", [])
    if not members:
        _fail(
            f"Project '{project_name}' has no members."
            f" Add one: openlegion project add-agent {project_name}"
        )

    if agent_name is None:
        if len(members) == 1:
            agent_name = members[0]
        else:
            for i, m in enumerate(members, 1):
                click.echo(f"  {i}. {m}")
            choice = click.prompt("Select agent", type=click.IntRange(1, len(members)), default=1)
            agent_name = members[choice - 1]

    if agent_name not in members:
        _fail(f"Agent '{agent_name}' is not in project '{project_name}'.")

    try:
        _remove_agent_from_project(project_name, agent_name)
    except ValueError as e:
        _fail(str(e))

    click.echo(f"Removed '{agent_name}' from project '{project_name}' (now standalone).")
    click.echo("  Restart to apply: openlegion start")


@project.command("edit")
@click.argument("name", required=False, default=None)
def project_edit(name: str | None):
    """Edit a project's description or open project.md in $EDITOR.

    \b
    Examples:
      openlegion project edit my-project
    """
    from src.cli.config import PROJECTS_DIR, _load_projects

    projects = _load_projects()
    if not projects:
        click.echo("No projects configured.")
        return

    if name is None:
        names = sorted(projects.keys())
        if len(names) == 1:
            name = names[0]
        else:
            for i, n in enumerate(names, 1):
                click.echo(f"  {i}. {n}")
            choice = click.prompt("Select project", type=click.IntRange(1, len(names)), default=1)
            name = names[choice - 1]

    if name not in projects:
        _fail(f"Project '{name}' not found.")

    click.echo(f"\nProject: {name}")
    click.echo(f"  Description: {projects[name].get('description', '(none)')}")
    click.echo(f"  Members: {', '.join(projects[name].get('members', [])) or '(none)'}\n")

    options = [
        ("Edit description", "desc"),
        ("Open project.md in editor", "editor"),
    ]
    for i, (label, _) in enumerate(options, 1):
        click.echo(f"  {i}. {label}")
    choice = click.prompt("\nSelect", type=click.IntRange(1, len(options)), default=1)

    if options[choice - 1][1] == "desc":
        new_desc = click.prompt("Description", default=projects[name].get("description", ""))
        meta_file = PROJECTS_DIR / name / "metadata.yaml"
        with open(meta_file) as f:
            data = yaml.safe_load(f) or {}
        data["description"] = new_desc
        with open(meta_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        click.echo("Description updated.")
    else:
        project_md = PROJECTS_DIR / name / "project.md"
        click.edit(filename=str(project_md))


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
        if serve:
            # Headless mode: keep alive via signal wait (used by -d)
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
    import logging

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
                 "host.permissions", "host.lanes"]:
        logger = logging.getLogger(name)
        # Replace stderr handlers with file handler
        logger.handlers = [file_handler]


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
        _single_agent_repl(name, agent_url)
    except KeyboardInterrupt:
        click.echo("\nDisconnected.")


def _single_agent_repl(agent_name: str, agent_url: str) -> None:
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

        # Try streaming first, fall back to sync
        try:
            _stream_detached_chat(agent_name, agent_url, user_input)
        except (httpx.HTTPError, httpx.StreamError, OSError):
            _sync_detached_chat(agent_name, agent_url, user_input)


def _stream_detached_chat(agent_name: str, agent_url: str, message: str) -> None:
    """Stream a chat response via SSE."""
    import json as _json

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
        import json as _json

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


# ── webhook ─────────────────────────────────────────────────


@cli.group(invoke_without_command=True)
@click.pass_context
def webhook(ctx):
    """Manage webhooks (list, test)."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(webhook_list)


@webhook.command("list")
@click.option("--port", default=8420, type=int)
def webhook_list(port: int):
    """List configured webhooks."""
    import httpx

    try:
        resp = httpx.get(
            f"http://localhost:{port}/dashboard/api/webhooks", timeout=5
        )
        if resp.status_code == 404:
            click.echo("No webhook endpoints available.")
            return
        webhooks = resp.json()
        if not webhooks:
            click.echo("No webhooks configured.")
            return
        click.echo(f"{'Name':<20} {'URL'}")
        click.echo("-" * 50)
        for wh in webhooks if isinstance(webhooks, list) else []:
            click.echo(f"{wh.get('name', '?'):<20} {wh.get('url', '')}")
    except httpx.ConnectError:
        click.echo("Mesh is not running.", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@webhook.command("test")
@click.argument("name")
@click.option("--port", default=8420, type=int)
def webhook_test(name: str, port: int):
    """Send a test payload to a webhook."""
    from urllib.parse import quote

    import httpx

    try:
        resp = httpx.post(
            f"http://localhost:{port}/dashboard/api/webhooks/{quote(name, safe='')}/test",
            json={"test": True, "source": "cli"},
            timeout=30,
        )
        if resp.status_code == 200:
            click.echo(f"Webhook '{name}' test: OK")
        elif resp.status_code == 404:
            click.echo(f"Webhook '{name}' not found.", err=True)
            raise SystemExit(1)
        else:
            click.echo(f"Webhook '{name}' test failed: {resp.text}", err=True)
    except httpx.ConnectError:
        click.echo("Mesh is not running.", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


# ── stop ─────────────────────────────────────────────────────

@cli.command()
def stop():
    """Stop the runtime and all agent containers."""
    import signal
    import time

    try:
        import docker
    except ImportError:
        click.echo("Docker SDK not installed. Install with: pip install docker", err=True)
        sys.exit(1)

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


@cli.command("completion", hidden=True)
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell: str):
    """Generate shell completion script.

    \b
    Usage:
      eval "$(openlegion completion bash)"
      eval "$(openlegion completion zsh)"
      openlegion completion fish | source
    """
    # Click uses an env var to generate completion scripts.
    # The var name is derived from the CLI group name.
    env_var = "_OPENLEGION_COMPLETE"
    source_type = f"{shell}_source"
    if shell == "fish":
        click.echo(f"set -x {env_var} {source_type}; openlegion; set -e {env_var}")
    else:
        click.echo(f'eval "$({env_var}={source_type} openlegion)"')


# ── logs ─────────────────────────────────────────────────────

@cli.command("logs")
@click.option("--lines", "-n", default=50, type=int, help="Number of lines to show")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option(
    "--level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default=None,
)
def logs(lines: int, follow: bool, level: str | None):
    """View runtime and agent logs.

    \b
    Examples:
      openlegion logs                 # last 50 lines of mesh log
      openlegion logs -f              # follow mesh log
      openlegion logs --level error   # filter by level
    """
    log_path = cli_config.PROJECT_ROOT / ".openlegion.log"
    if not log_path.exists():
        click.echo("No log file found. Start the runtime first: openlegion start")
        raise SystemExit(1)

    content = log_path.read_text()
    all_lines = content.splitlines()

    # Filter by level if specified
    level_pat = None
    if level:
        import re

        level_upper = level.upper()
        level_pat = re.compile(r"\b" + re.escape(level_upper) + r"\b")
        all_lines = [line for line in all_lines if level_pat.search(line.upper())]

    # Show last N lines
    for line in all_lines[-lines:]:
        click.echo(line)

    if follow:
        import time

        seen = len(content)
        try:
            while True:
                time.sleep(0.5)
                new_content = log_path.read_text()
                if len(new_content) > seen:
                    new_lines = new_content[seen:].splitlines()
                    for line in new_lines:
                        if level_pat and not level_pat.search(line.upper()):
                            continue
                        click.echo(line)
                    seen = len(new_content)
        except KeyboardInterrupt:
            pass


# ── config ───────────────────────────────────────────────────

@cli.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """View and validate configuration."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(config_show)


@config.command("show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def config_show(as_json: bool):
    """Show effective configuration."""
    cfg = _load_config()
    # Remove internal keys
    cfg.pop("_agent_projects", None)
    if as_json:
        import json as _json

        click.echo(_json.dumps(cfg, default=str))
    else:
        click.echo(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


@config.command("validate")
def config_validate():
    """Check configuration for errors."""
    from src.cli.formatting import echo_fail, echo_ok

    errors = 0

    # Check config files exist
    if cli_config.CONFIG_FILE.exists():
        echo_ok(f"mesh.yaml: {cli_config.CONFIG_FILE}")
        try:
            with open(cli_config.CONFIG_FILE) as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            echo_fail(f"  Parse error: {e}")
            errors += 1
    else:
        echo_fail(f"mesh.yaml not found: {cli_config.CONFIG_FILE}")
        errors += 1

    if cli_config.AGENTS_FILE.exists():
        echo_ok(f"agents.yaml: {cli_config.AGENTS_FILE}")
        try:
            with open(cli_config.AGENTS_FILE) as f:
                data = yaml.safe_load(f)
            agents = data.get("agents", {}) if data else {}
            echo_ok(f"  {len(agents)} agent(s) configured")
        except yaml.YAMLError as e:
            echo_fail(f"  Parse error: {e}")
            errors += 1
    else:
        from src.cli.formatting import echo_warn

        echo_warn(f"agents.yaml not found: {cli_config.AGENTS_FILE}")

    if cli_config.PERMISSIONS_FILE.exists():
        import json as _json

        echo_ok(f"permissions.json: {cli_config.PERMISSIONS_FILE}")
        try:
            with open(cli_config.PERMISSIONS_FILE) as f:
                _json.load(f)
        except _json.JSONDecodeError as e:
            echo_fail(f"  Parse error: {e}")
            errors += 1
    else:
        from src.cli.formatting import echo_warn

        echo_warn(f"permissions.json not found: {cli_config.PERMISSIONS_FILE}")

    # Check .env
    if cli_config.ENV_FILE.exists():
        echo_ok(f".env: {cli_config.ENV_FILE}")
    else:
        from src.cli.formatting import echo_warn

        echo_warn(f".env not found: {cli_config.ENV_FILE}")

    if errors:
        raise SystemExit(1)
    else:
        echo_ok("Configuration valid")


@config.command("path")
def config_path():
    """Show configuration file locations."""
    click.echo(f"Project root:    {cli_config.PROJECT_ROOT}")
    click.echo(f"Mesh config:     {cli_config.CONFIG_FILE}")
    click.echo(f"Agent config:    {cli_config.AGENTS_FILE}")
    click.echo(f"Permissions:     {cli_config.PERMISSIONS_FILE}")
    click.echo(f"Environment:     {cli_config.ENV_FILE}")
    click.echo(f"Projects:        {cli_config.PROJECTS_DIR}")
    click.echo(f"Skills market:   {cli_config.MARKETPLACE_DIR}")


@config.command("export")
@click.argument("output", default="-", type=click.Path())
def config_export(output: str):
    """Export configuration as a tarball.

    \b
    Examples:
      openlegion config export backup.tar.gz
      openlegion config export -  # to stdout
    """
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in [cli_config.CONFIG_FILE, cli_config.AGENTS_FILE, cli_config.PERMISSIONS_FILE]:
            if path.exists():
                tar.add(str(path), arcname=path.name)
        # Add project metadata
        if cli_config.PROJECTS_DIR.exists():
            for meta_file in cli_config.PROJECTS_DIR.glob("*/metadata.yaml"):
                arcname = f"projects/{meta_file.parent.name}/{meta_file.name}"
                tar.add(str(meta_file), arcname=arcname)

    if output == "-":
        sys.stdout.buffer.write(buf.getvalue())
    else:
        with open(output, "wb") as f:
            f.write(buf.getvalue())
        click.echo(f"Configuration exported to {output}")


@config.command("import")
@click.argument("archive", type=click.Path(exists=True))
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def config_import(archive: str, yes: bool):
    """Import configuration from a tarball.

    \b
    Examples:
      openlegion config import backup.tar.gz
    """
    import tarfile

    if not yes:
        click.confirm("Import will overwrite existing configuration. Continue?", abort=True)

    with tarfile.open(archive, "r:gz") as tar:
        # Security: validate member names and reject symlinks/hardlinks
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                click.echo(f"Unsafe path in archive: {member.name}", err=True)
                raise SystemExit(1)
            if member.issym() or member.islnk():
                click.echo(f"Unsafe link in archive: {member.name}", err=True)
                raise SystemExit(1)

        config_dir = cli_config.CONFIG_FILE.parent
        config_dir.mkdir(parents=True, exist_ok=True)

        for member in tar.getmembers():
            if member.name.startswith("projects/"):
                # Extract to projects dir
                dest = cli_config.PROJECTS_DIR / member.name[len("projects/"):]
                dest.parent.mkdir(parents=True, exist_ok=True)
                f = tar.extractfile(member)
                if f:
                    dest.write_bytes(f.read())
                    click.echo(f"  Imported {member.name}")
            else:
                # Extract to config dir
                dest = config_dir / member.name
                f = tar.extractfile(member)
                if f:
                    dest.write_bytes(f.read())
                    click.echo(f"  Imported {member.name}")

    click.echo("Configuration imported. Restart to apply: openlegion start")


# ── health ───────────────────────────────────────────────────

@cli.command("health")
@click.option("--port", default=8420, type=int)
def health(port: int):
    """Run pre-flight health checks.

    \b
    Examples:
      openlegion health
    """
    from src.cli.formatting import echo_fail, echo_ok, echo_warn

    errors = 0

    # 1. Docker
    from src.cli.config import _check_docker_running

    if _check_docker_running():
        echo_ok("Docker daemon running")
    else:
        echo_fail("Docker daemon not running")
        errors += 1

    # 2. Port check
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(("localhost", port))
            if result == 0:
                echo_ok(f"Port {port}: in use (mesh may be running)")
            else:
                echo_ok(f"Port {port}: available")
    except Exception:
        echo_warn(f"Port {port}: could not check")

    # 3. Config files
    if cli_config.CONFIG_FILE.exists():
        echo_ok("Config files present")
    else:
        echo_warn("No mesh.yaml found")

    # 4. Credentials
    if cli_config.ENV_FILE.exists():
        env_content = cli_config.ENV_FILE.read_text()
        has_creds = (
            "OPENLEGION_SYSTEM_" in env_content
            or "OPENLEGION_CRED_" in env_content
        )
        if has_creds:
            echo_ok("Credentials configured")
        else:
            echo_warn("No credentials in .env")
    else:
        echo_warn("No .env file")

    # 5. Mesh status
    try:
        import httpx

        resp = httpx.get(f"http://localhost:{port}/mesh/agents", timeout=3)
        agents = resp.json()
        echo_ok(f"Mesh online: {len(agents)} agent(s)")
    except Exception:
        echo_warn("Mesh not running")

    if errors:
        raise SystemExit(1)


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
