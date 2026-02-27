"""CLI entry point for OpenLegion.

Core:
  start                    Start runtime + interactive REPL
  start -d                 Start in background (detached)
  stop                     Stop all containers
  status                   Show agent status
  chat <name>              Connect to a running agent

Agent management:
  agent add [name]         Add a new agent
  agent list               List configured agents
  agent edit [name]        Change an agent's settings
  agent remove [name]      Remove an agent
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
    _default_description,
    _get_default_model,
    _load_config,
    _remove_agent,
    _set_env_key,
    _suppress_host_logs,
    _update_agent_field,
)

logger = logging.getLogger("cli")


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
        click.echo(f"Agent '{name}' already exists.")
        return

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
@click.option("--budget", "budget_override", default=None, type=float, help="Set daily budget (USD)")
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
        return

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
    except Exception as e:
        logger.debug("Docker container list failed: %s", e)

    default_model = _get_default_model()
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
        return
    if not yes:
        click.confirm(f"Remove agent '{name}'? This deletes its config and permissions.", abort=True)

    _remove_agent(name)
    click.echo(f"Removed agent '{name}'.")


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
        click.echo(f"Unknown channel '{channel_type}'. Available: {', '.join(CHANNEL_TYPES)}")
        return

    ch = CHANNEL_TYPES[channel_type]
    click.echo(f"\n  {ch['label']} Setup")
    click.echo(f"  {ch['token_help']}\n")

    token = click.prompt(f"  {ch['label']} bot token", hide_input=True)
    if not token.strip():
        click.echo("  No token provided. Skipped.")
        return

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
        sys_key = f"OPENLEGION_SYSTEM_{info['env_key'].upper()}"
        cred_key = f"OPENLEGION_CRED_{info['env_key'].upper()}"
        has_token = bool(os.environ.get(sys_key) or os.environ.get(cred_key))
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
@click.option("--description", "-d", "desc", default="", help="Project description")
@click.option("--agents", "-a", "agents_str", default="", help="Comma-separated agent names")
def project_create(name: str | None, desc: str, agents_str: str):
    """Create a new project.

    \b
    Examples:
      openlegion project create my-project
      openlegion project create my-project -d "Marketing automation" -a agent1,agent2
      openlegion project create           # interactive
    """
    from src.cli.config import _create_project, _load_projects, _validate_project_name

    if name is None:
        name = click.prompt("Project name")

    try:
        _validate_project_name(name)
    except ValueError as e:
        click.echo(str(e), err=True)
        return

    existing = _load_projects()
    if name in existing:
        click.echo(f"Project '{name}' already exists.")
        return

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
        click.echo(f"Unknown agent(s): {', '.join(invalid)}", err=True)
        return

    try:
        _create_project(name, description=desc, members=members)
    except ValueError as e:
        click.echo(str(e), err=True)
        return

    click.echo(f"\nProject '{name}' created.")
    if members:
        click.echo(f"  Members: {', '.join(members)}")
    click.echo("  Restart to apply: openlegion start")


@project.command("list")
def project_list():
    """List all projects and their members."""
    from src.cli.config import _load_projects

    cfg = _load_config()
    projects = _load_projects()
    all_agents = set(cfg.get("agents", {}))
    assigned = set()
    for pdata in projects.values():
        assigned.update(pdata.get("members", []))
    standalone = sorted(all_agents - assigned)

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
        click.echo(f"Project '{name}' not found.")
        return

    if not yes:
        members = projects[name].get("members", [])
        msg = f"Delete project '{name}'?"
        if members:
            msg += f" Agents ({', '.join(members)}) will become standalone."
        click.confirm(msg, abort=True)

    try:
        _delete_project(name)
    except ValueError as e:
        click.echo(str(e), err=True)
        return

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
        click.echo(f"Project '{project_name}' not found.")
        return

    cfg = _load_config()
    if agent_name is None:
        agent_name = _resolve_agent_name(cfg, None)
    if agent_name is None:
        return
    if agent_name not in cfg.get("agents", {}):
        click.echo(f"Agent '{agent_name}' not found.")
        return

    try:
        _add_agent_to_project(project_name, agent_name)
    except ValueError as e:
        click.echo(str(e), err=True)
        return

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
        click.echo(f"Project '{project_name}' not found.")
        return

    members = projects[project_name].get("members", [])
    if not members:
        click.echo(f"Project '{project_name}' has no members.")
        return

    if agent_name is None:
        if len(members) == 1:
            agent_name = members[0]
        else:
            for i, m in enumerate(members, 1):
                click.echo(f"  {i}. {m}")
            choice = click.prompt("Select agent", type=click.IntRange(1, len(members)), default=1)
            agent_name = members[choice - 1]

    if agent_name not in members:
        click.echo(f"Agent '{agent_name}' is not in project '{project_name}'.")
        return

    try:
        _remove_agent_from_project(project_name, agent_name)
    except ValueError as e:
        click.echo(str(e), err=True)
        return

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
        click.echo(f"Project '{name}' not found.")
        return

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
        click.echo("No agents configured. Run: openlegion start")
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
            except Exception as e:
                logger.debug("Agent status check failed for %s: %s", name, e)
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
