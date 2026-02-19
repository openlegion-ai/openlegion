"""CLI entry point for OpenLegion.

Core:
  setup             One-time setup: API key, project, first agent, Docker image
  start             Start runtime + interactive chat REPL
  start -d          Start runtime in background (detached)
  stop              Stop all agent containers
  status            Show agent status
  chat <name>       Connect to a running agent (detached mode)

Agent management:
  agent add [name]        Add a new agent
  agent list              List configured agents
  agent remove <name>     Remove an agent
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_FILE = PROJECT_ROOT / "config" / "mesh.yaml"
AGENTS_FILE = PROJECT_ROOT / "config" / "agents.yaml"
PERMISSIONS_FILE = PROJECT_ROOT / "config" / "permissions.json"
PROJECT_FILE = PROJECT_ROOT / "PROJECT.md"
TEMPLATES_DIR = Path(__file__).parent / "templates"

_PROVIDERS = [
    {"name": "anthropic", "label": "Anthropic (recommended)"},
    {"name": "moonshot", "label": "Moonshot / Kimi (recommended)"},
    {"name": "deepseek", "label": "DeepSeek"},
    {"name": "openai", "label": "OpenAI"},
    {"name": "gemini", "label": "Google Gemini"},
    {"name": "xai", "label": "xAI (Grok)"},
    {"name": "groq", "label": "Groq"},
]

_PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": [
        "openai/gpt-5.2",
        "openai/gpt-5.2-pro",
        "openai/gpt-5.1-codex",
        "openai/gpt-5",
        "openai/gpt-5-mini",
        "openai/o3",
        "openai/o4-mini",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
    ],
    "anthropic": [
        "anthropic/claude-opus-4-6",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-sonnet-4-5-20250929",
        "anthropic/claude-haiku-4-5-20251001",
    ],
    "gemini": [
        "gemini/gemini-3-pro-preview",
        "gemini/gemini-3-flash-preview",
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
    ],
    "xai": [
        "xai/grok-4-1-fast-reasoning",
        "xai/grok-4",
        "xai/grok-3",
        "xai/grok-3-mini",
    ],
    "deepseek": [
        "deepseek/deepseek-chat",
        "deepseek/deepseek-reasoner",
    ],
    "moonshot": [
        "moonshot/kimi-k2.5",
        "moonshot/kimi-k2",
        "moonshot/moonshot-v1-128k",
    ],
    "groq": [
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "groq/llama-3-groq-70b-tool-use",
    ],
}


# ── Helpers ──────────────────────────────────────────────────

def _load_config(mesh_path: Path | None = None) -> dict:
    """Load mesh config and merge agent definitions from agents.yaml."""
    path = mesh_path or CONFIG_FILE
    cfg: dict = {
        "mesh": {"host": "0.0.0.0", "port": 8420},
        "llm": {"default_model": "openai/gpt-4o-mini"},
        "agents": {},
    }
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
            cfg.update(data)
    if AGENTS_FILE.exists():
        with open(AGENTS_FILE) as f:
            agents_data = yaml.safe_load(f) or {}
            cfg.setdefault("agents", {}).update(agents_data.get("agents", {}))
    return cfg


def _load_permissions() -> dict:
    if not PERMISSIONS_FILE.exists():
        return {"permissions": {}}
    with open(PERMISSIONS_FILE) as f:
        return json.load(f)


def _save_permissions(perms: dict) -> None:
    PERMISSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PERMISSIONS_FILE, "w") as f:
        json.dump(perms, f, indent=2)
        f.write("\n")


def _set_env_key(name: str, value: str) -> None:
    """Set or update a key in the .env file."""
    env_key = f"OPENLEGION_CRED_{name.upper()}"
    lines: list[str] = []
    found = False

    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith(f"{env_key}=") or line.startswith(f"# {env_key}="):
                lines.append(f"{env_key}={value}")
                found = True
            else:
                lines.append(line)

    if not found:
        lines.append(f"{env_key}={value}")

    ENV_FILE.write_text("\n".join(lines) + "\n")
    os.environ[env_key] = value


def _check_docker_running() -> bool:
    """Verify Docker daemon is running and accessible."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def _check_docker_image() -> bool:
    """Check if the agent Docker image exists."""
    try:
        import docker
        client = docker.from_env()
        client.images.get("openlegion-agent:latest")
        return True
    except Exception:
        return False


def _docker_image_is_stale() -> bool:
    """Check if source files are newer than the Docker image."""
    try:
        from datetime import datetime, timezone

        import docker
        client = docker.from_env()
        image = client.images.get("openlegion-agent:latest")
        created_str = image.attrs.get("Created", "")
        if not created_str:
            return True
        image_time = datetime.fromisoformat(created_str.replace("Z", "+00:00"))

        src_dirs = [PROJECT_ROOT / "src" / "agent", PROJECT_ROOT / "src" / "shared"]
        for src_dir in src_dirs:
            if not src_dir.exists():
                continue
            for py_file in src_dir.rglob("*.py"):
                file_mtime = datetime.fromtimestamp(py_file.stat().st_mtime, tz=timezone.utc)
                if file_mtime > image_time:
                    return True

        dockerfile = PROJECT_ROOT / "Dockerfile.agent"
        if dockerfile.exists():
            df_mtime = datetime.fromtimestamp(dockerfile.stat().st_mtime, tz=timezone.utc)
            if df_mtime > image_time:
                return True
        return False
    except Exception:
        return False


def _ensure_docker_image() -> None:
    """Build the Docker image if missing or stale."""
    if not _check_docker_image():
        _build_docker_image()
    elif _docker_image_is_stale():
        click.echo("Source code changed since last Docker build.")
        _build_docker_image()


def _build_docker_image() -> None:
    """Build the agent Docker image."""
    import subprocess
    click.echo("Building Docker image (this may take a few minutes)...")
    result = subprocess.run(
        ["docker", "build", "-t", "openlegion-agent:latest", "-f", "Dockerfile.agent", "."],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Docker build failed:\n{result.stderr}", err=True)
        sys.exit(1)
    click.echo("Docker image built successfully.")


def _add_agent_to_config(name: str, role: str, model: str, system_prompt: str) -> None:
    """Add an agent entry to agents.yaml."""
    agents_cfg: dict = {"agents": {}}
    if AGENTS_FILE.exists():
        with open(AGENTS_FILE) as f:
            agents_cfg = yaml.safe_load(f) or {"agents": {}}
    if "agents" not in agents_cfg:
        agents_cfg["agents"] = {}

    agents_cfg["agents"][name] = {
        "role": role,
        "model": model,
        "skills_dir": f"./skills/{name}",
        "system_prompt": system_prompt,
        "resources": {"memory_limit": "512m", "cpu_limit": 0.5},
    }
    AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AGENTS_FILE, "w") as f:
        yaml.dump(agents_cfg, f, default_flow_style=False, sort_keys=False)


def _add_agent_permissions(name: str) -> None:
    """Add default permissions for a new agent.

    If collaboration mode is enabled in mesh.yaml, agents can message
    all other agents. Otherwise they can only message the orchestrator.
    """
    cfg = _load_config()
    collab = cfg.get("collaboration", True)

    perms = _load_permissions()
    if collab:
        other_agents = [a for a in cfg.get("agents", {}) if a != name]
        can_message = ["orchestrator", "*"] if other_agents else ["orchestrator"]
    else:
        can_message = ["orchestrator"]

    perms["permissions"][name] = {
        "can_message": can_message,
        "can_publish": [f"{name}_complete"],
        "can_subscribe": ["*"] if collab else [],
        "blackboard_read": ["context/*", "tasks/*", "goals/*", "signals/*", "artifacts/*"],
        "blackboard_write": ["context/*", "goals/*", "signals/*", "artifacts/*"],
        "allowed_apis": ["llm"],
    }
    _save_permissions(perms)


def _set_collaborative_permissions() -> None:
    """Update all agent permissions to allow inter-agent messaging."""
    perms = _load_permissions()
    for name, p in perms.get("permissions", {}).items():
        if name == "default":
            continue
        if "*" not in p.get("can_message", []):
            p["can_message"] = list({*p.get("can_message", []), "*"})
        if "*" not in p.get("can_subscribe", []):
            p["can_subscribe"] = list({*p.get("can_subscribe", []), "*"})
    _save_permissions(perms)


def _set_isolated_permissions() -> None:
    """Restrict agent permissions to orchestrator-only messaging."""
    perms = _load_permissions()
    for name, p in perms.get("permissions", {}).items():
        if name == "default":
            continue
        p["can_message"] = ["orchestrator"]
        p["can_subscribe"] = []
    _save_permissions(perms)


def _create_agent(name: str, description: str, model: str) -> None:
    """Create an agent: config, permissions, skills directory."""
    system_prompt = (
        f"You are the '{name}' agent. {description} "
        "Use your tools and knowledge to accomplish tasks. "
        "Check PROJECT.md for the current priorities and constraints."
    )
    _add_agent_to_config(name, description, model, system_prompt)
    _add_agent_permissions(name)
    skills_dir = PROJECT_ROOT / "skills" / name
    skills_dir.mkdir(parents=True, exist_ok=True)


def _suppress_host_logs() -> None:
    """Set host-side loggers to WARNING for clean CLI output."""
    import logging
    for name in [
        "host", "host.containers", "host.credentials", "host.orchestrator",
        "host.mesh", "host.costs", "host.permissions", "host.cron", "host.webhooks",
        "host.health", "host.lanes", "host.runtime", "host.watchers",
        "channels", "channels.base", "channels.telegram", "channels.discord",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)


def _ensure_pairing_code(pairing_path: Path) -> str | None:
    """Ensure a pairing file exists with a code. Returns code if unpaired, None if already paired."""
    data: dict = {}
    if pairing_path.exists():
        try:
            data = json.loads(pairing_path.read_text())
        except Exception:
            pass
    if data.get("owner"):
        return None
    code = data.get("pairing_code")
    if not code:
        code = secrets.token_hex(8)
        data = {"owner": None, "allowed": [], "pairing_code": code}
        pairing_path.parent.mkdir(parents=True, exist_ok=True)
        pairing_path.write_text(json.dumps(data, indent=2) + "\n")
    return code


def _start_channels(
    cfg: dict,
    dispatch_fn,
    agent_registry: dict,
    active_channels: list,
    status_fn=None,
    costs_fn=None,
    reset_fn=None,
    stream_dispatch_fn=None,
) -> list[str]:
    """Start configured messaging channels (Telegram, Discord) in background threads.

    Channels receive the same callbacks as the CLI REPL so they present
    a unified multi-agent chat interface.

    Returns a list of pairing instruction strings to display to the user.
    """
    import threading

    pairing_instructions: list[str] = []
    channels_cfg = cfg.get("channels", {})
    all_agents = cfg.get("agents", {})
    first_agent = next(iter(all_agents), "")

    def list_agents_fn():
        return dict(agent_registry)

    common = {
        "dispatch_fn": dispatch_fn,
        "list_agents_fn": list_agents_fn,
        "status_fn": status_fn,
        "costs_fn": costs_fn,
        "reset_fn": reset_fn,
        "stream_dispatch_fn": stream_dispatch_fn,
    }

    # Telegram
    tg_cfg = channels_cfg.get("telegram", {})
    tg_token = (
        tg_cfg.get("token")
        or os.environ.get("OPENLEGION_CRED_TELEGRAM_BOT_TOKEN", "")
        or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    )
    if tg_token:
        tg_code = _ensure_pairing_code(PROJECT_ROOT / "config" / "telegram_paired.json")
        from src.channels.telegram import TelegramChannel
        tg = TelegramChannel(
            token=tg_token,
            default_agent=tg_cfg.get("default_agent", first_agent),
            allowed_users=tg_cfg.get("allowed_users"),
            **common,
        )
        def run_tg():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(tg.start())
            loop.run_forever()
        t = threading.Thread(target=run_tg, daemon=True)
        t.start()
        active_channels.append(tg)

        if tg_code:
            pairing_instructions.append(
                f"  Telegram: send to your bot →  /start {tg_code}"
            )
        else:
            click.echo("  Telegram channel active (paired)")

    # Discord
    dc_cfg = channels_cfg.get("discord", {})
    dc_token = (
        dc_cfg.get("token")
        or os.environ.get("OPENLEGION_CRED_DISCORD_BOT_TOKEN", "")
        or os.environ.get("DISCORD_BOT_TOKEN", "")
    )
    if dc_token:
        dc_code = _ensure_pairing_code(PROJECT_ROOT / "config" / "discord_paired.json")
        from src.channels.discord import DiscordChannel
        dc = DiscordChannel(
            token=dc_token,
            default_agent=dc_cfg.get("default_agent", first_agent),
            allowed_guilds=dc_cfg.get("allowed_guilds"),
            **common,
        )
        def run_dc():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(dc.start())
            loop.run_forever()
        t = threading.Thread(target=run_dc, daemon=True)
        t.start()
        active_channels.append(dc)

        if dc_code:
            pairing_instructions.append(
                f"  Discord: DM your bot →  !start {dc_code}"
            )
        else:
            click.echo("  Discord channel active (paired)")

    return pairing_instructions


def _stop_channels(active_channels: list) -> None:
    """Stop all active messaging channels."""
    import asyncio
    for ch in active_channels:
        try:
            asyncio.run(ch.stop())
        except Exception:
            pass


def _get_default_model() -> str:
    cfg = _load_config()
    return cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")


def _load_templates() -> dict[str, dict]:
    """Load available team templates from src/templates/."""
    available: dict[str, dict] = {}
    if not TEMPLATES_DIR.exists():
        return available
    for tpl_file in sorted(TEMPLATES_DIR.glob("*.yaml")):
        with open(tpl_file) as f:
            tpl = yaml.safe_load(f) or {}
        name = tpl.get("name", tpl_file.stem)
        available[name] = tpl
    return available


def _apply_template(template_name: str, tpl: dict) -> list[str]:
    """Apply a team template, creating all agents. Returns list of agent names."""
    cfg = _load_config()
    default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
    tpl_agents = tpl.get("agents", {})
    created: list[str] = []

    for agent_name, agent_def in tpl_agents.items():
        model = agent_def.get("model", default_model).replace("{default_model}", default_model)
        _add_agent_to_config(
            name=agent_name,
            role=agent_def.get("role", agent_name),
            model=model,
            system_prompt=agent_def.get("system_prompt", ""),
        )
        if "resources" in agent_def:
            agents_cfg: dict = {"agents": {}}
            if AGENTS_FILE.exists():
                with open(AGENTS_FILE) as f:
                    agents_cfg = yaml.safe_load(f) or {"agents": {}}
            if agent_name in agents_cfg.get("agents", {}):
                agents_cfg["agents"][agent_name]["resources"] = agent_def["resources"]
                with open(AGENTS_FILE, "w") as f:
                    yaml.dump(agents_cfg, f, default_flow_style=False, sort_keys=False)
        _add_agent_permissions(agent_name)
        skills_dir = PROJECT_ROOT / "skills" / agent_name
        skills_dir.mkdir(parents=True, exist_ok=True)
        created.append(agent_name)

    return created


def _discover_workflows() -> list[dict]:
    """Discover workflow definitions from config/workflows/."""
    wf_dir = PROJECT_ROOT / "config" / "workflows"
    if not wf_dir.exists():
        return []
    workflows = []
    for wf_file in sorted(wf_dir.glob("*.yaml")):
        with open(wf_file) as f:
            wf = yaml.safe_load(f) or {}
        if "name" not in wf or "steps" not in wf:
            continue
        agents_used = sorted({s["agent"] for s in wf["steps"] if "agent" in s})
        workflows.append({
            "name": wf["name"],
            "file": wf_file.name,
            "agents": agents_used,
            "label": wf["name"].replace("_", " ").title(),
        })
    return workflows


# ── Main group ───────────────────────────────────────────────

@click.group()
def cli():
    """OpenLegion -- Autonomous AI agent fleet."""
    _suppress_host_logs()


# ── setup ────────────────────────────────────────────────────

@cli.command()
def setup():
    """One-time setup: API key, project definition, first agent, Docker image."""

    click.echo("=== OpenLegion Setup ===\n")

    # 0. Docker pre-flight
    if not _check_docker_running():
        click.echo(
            "Docker is not running or not accessible.\n"
            "Please start Docker and ensure your user has permission to use it.\n"
            "  - Linux: sudo systemctl start docker && sudo usermod -aG docker $USER\n"
            "  - macOS/Windows: Start Docker Desktop",
            err=True,
        )
        sys.exit(1)

    # Step 1: LLM provider + model + API key
    click.echo("Step 1: LLM Provider\n")

    for i, p in enumerate(_PROVIDERS, 1):
        click.echo(f"  {i}. {p['label']}")
    click.echo(
        "\n  Tip: Anthropic Claude and Moonshot Kimi are recommended for agentic\n"
        "  tasks (browser automation, web interaction, tool use). They have\n"
        "  built-in computer use training and strong tool-calling support.\n"
    )
    choice = click.prompt("  Select provider", type=click.IntRange(1, len(_PROVIDERS)), default=1)
    provider = _PROVIDERS[choice - 1]["name"]
    click.echo(f"  Selected: {_PROVIDERS[choice - 1]['label']}\n")

    # Model selection
    models = _PROVIDER_MODELS[provider]
    click.echo("  Available models:")
    for i, m in enumerate(models, 1):
        click.echo(f"  {i}. {m}")
    model_choice = click.prompt("\n  Select model", type=click.IntRange(1, len(models)), default=1)
    selected_model = models[model_choice - 1]
    click.echo(f"  Selected: {selected_model}\n")

    # API key
    key_name = f"{provider}_api_key"
    existing_key = os.environ.get(f"OPENLEGION_CRED_{key_name.upper()}", "")
    if existing_key:
        click.echo(f"  API key already set for {provider}.")
        if click.confirm("  Replace it?", default=False):
            api_key = click.prompt("  API key", hide_input=True)
            _set_env_key(key_name, api_key)
    else:
        api_key = click.prompt(f"  {_PROVIDERS[choice - 1]['label']} API key", hide_input=True)
        _set_env_key(key_name, api_key)

    # Update default model in mesh config
    mesh_cfg = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            mesh_cfg = yaml.safe_load(f) or {}
    mesh_cfg.setdefault("llm", {})["default_model"] = selected_model
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(mesh_cfg, f, default_flow_style=False, sort_keys=False)

    # Step 2: Project definition (optional north star)
    click.echo("\nStep 2: Your Project (optional)\n")

    project_desc = click.prompt(
        "  What are you building? (press Enter to skip)",
        default="",
        show_default=False,
    )
    if project_desc:
        PROJECT_FILE.write_text(
            f"# PROJECT.md\n\n"
            f"## What We're Building\n{project_desc}\n\n"
            f"## Current Priority\n[Define your current focus]\n\n"
            f"## Hard Constraints\n[Budget limits, deadlines, compliance rules]\n"
        )
        click.echo("  Saved to PROJECT.md. Every agent will see this as their north star.")
    elif not PROJECT_FILE.exists():
        click.echo("  Skipped. You can define it later by editing PROJECT.md.")

    # Step 3: First agent (or team template)
    click.echo("\nStep 3: Your Agents\n")

    cfg = _load_config()
    existing_agents = list(cfg.get("agents", {}).keys())

    if existing_agents:
        click.echo(f"  Existing agents: {', '.join(existing_agents)}")
        if not click.confirm("  Add another agent?", default=False):
            click.echo("  Keeping existing agents.")
        else:
            _setup_agent_wizard(selected_model)
    else:
        templates = _load_templates()
        if templates:
            tpl_names = list(templates.keys())
            tpl_display = ", ".join(tpl_names)
            use_template = click.prompt(
                f"  Start from a template? ({tpl_display}) or 'none' for custom",
                default="none",
            )
            if use_template != "none" and use_template in templates:
                created = _apply_template(use_template, templates[use_template])
                click.echo(f"  Created agents: {', '.join(created)}")
            else:
                _setup_agent_wizard(selected_model)
        else:
            _setup_agent_wizard(selected_model)

    # Step 4: Messaging channels (optional)
    click.echo("\nStep 4: Messaging Channels (optional)\n")

    if click.confirm("  Connect a Telegram bot?", default=False):
        tg_token = click.prompt("  Telegram bot token", hide_input=True)
        _set_env_key("telegram_bot_token", tg_token)
        mesh_cfg = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                mesh_cfg = yaml.safe_load(f) or {}
        mesh_cfg.setdefault("channels", {}).setdefault("telegram", {})
        mesh_cfg["channels"]["telegram"]["enabled"] = True
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(mesh_cfg, f, default_flow_style=False, sort_keys=False)
        click.echo("  Telegram configured. Pairing code will appear when you run `openlegion start`.")

    if click.confirm("  Connect a Discord bot?", default=False):
        dc_token = click.prompt("  Discord bot token", hide_input=True)
        _set_env_key("discord_bot_token", dc_token)
        mesh_cfg = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                mesh_cfg = yaml.safe_load(f) or {}
        mesh_cfg.setdefault("channels", {}).setdefault("discord", {})
        mesh_cfg["channels"]["discord"]["enabled"] = True
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(mesh_cfg, f, default_flow_style=False, sort_keys=False)
        click.echo("  Discord configured. Pairing code will appear when you run `openlegion start`.")

    has_tg = os.environ.get("OPENLEGION_CRED_TELEGRAM_BOT_TOKEN")
    has_dc = os.environ.get("OPENLEGION_CRED_DISCORD_BOT_TOKEN")
    if not has_tg and not has_dc:
        click.echo("  Skipped. Add channels later during setup or by setting tokens in .env.")

    # Step 5: Agent collaboration mode
    click.echo("\nStep 5: Agent Collaboration\n")
    click.echo("  Isolated:      Agents work independently, no shared context or messaging.")
    click.echo("  Collaborative: Agents can message each other, share blackboard data.\n")

    collab = click.confirm("  Enable agent collaboration?", default=True)
    mesh_cfg = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            mesh_cfg = yaml.safe_load(f) or {}
    mesh_cfg["collaboration"] = collab
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(mesh_cfg, f, default_flow_style=False, sort_keys=False)

    if collab:
        # Update permissions so agents can message each other
        _set_collaborative_permissions()
        click.echo("  Collaboration enabled. Agents can communicate via the mesh.")
    else:
        _set_isolated_permissions()
        click.echo("  Isolation mode. Agents operate independently.")

    # Step 6: Docker image
    _ensure_docker_image()

    # Done
    click.echo("\nSetup complete.")
    click.echo("  Start the runtime:  openlegion start")
    click.echo("  Add more agents:    openlegion agent add")


def _setup_agent_wizard(model: str) -> str:
    """Interactive agent creation for setup. Returns agent name."""
    agent_name = click.prompt("  Agent name", default="assistant")
    description = click.prompt(
        "  What should this agent do?",
        default="General-purpose assistant",
    )
    _create_agent(agent_name, description, model)
    click.echo(f"  Created agent '{agent_name}'.")
    return agent_name


# ── agent subgroup ───────────────────────────────────────────

@cli.group()
def agent():
    """Add, list, or remove agents."""
    pass


@agent.command("add")
@click.argument("name", required=False, default=None)
def agent_add(name: str | None):
    """Add a new agent.

    Examples:
      openlegion agent add researcher
      openlegion agent add              # interactive mode
    """
    cfg = _load_config()

    if name is None:
        name = click.prompt("Agent name")

    if name in cfg.get("agents", {}):
        click.echo(f"Agent '{name}' already exists.")
        return

    description = click.prompt(
        "What should this agent do?",
        default=f"General-purpose {name} assistant",
    )
    model = _get_default_model()
    _create_agent(name, description, model)

    click.echo(f"\nAgent '{name}' created.")
    click.echo(f"  Role:  {description}")
    click.echo(f"  Model: {model}")
    click.echo("\nStart chatting: openlegion start")


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

    click.echo(f"{'Name':<16} {'Role':<24} {'Status':<10}")
    click.echo("-" * 50)
    for name, info in agents.items():
        status = "running" if name in running else "stopped"
        click.echo(f"{name:<16} {info.get('role', 'n/a'):<24} {status:<10}")


@agent.command("remove")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def agent_remove(name: str, yes: bool):
    """Remove an agent from configuration."""
    cfg = _load_config()
    if name not in cfg.get("agents", {}):
        click.echo(f"Agent '{name}' not found.")
        return
    if not yes:
        click.confirm(f"Remove agent '{name}'? This deletes its config and permissions.", abort=True)

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


# ── start ────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "config_path", default="config/mesh.yaml", help="Path to mesh config")
@click.option("--detach", "-d", is_flag=True, help="Run in background (no interactive REPL)")
def start(config_path: str, detach: bool):
    """Start the runtime and chat with your agents.

    By default, starts the mesh and all agents then drops into an interactive
    REPL. Use -d to run in the background instead.

    \b
    Examples:
      openlegion start          # interactive mode
      openlegion start -d       # background mode
    """
    if detach:
        _start_detached(config_path)
    else:
        _start_interactive(config_path)


def _start_interactive(config_path: str) -> None:
    """Start mesh + agents in background threads, then drop into REPL."""
    import asyncio
    import threading

    import httpx
    import uvicorn

    from src.channels.webhook import create_webhook_router
    from src.host.costs import CostTracker
    from src.host.credentials import CredentialVault
    from src.host.cron import CronScheduler
    from src.host.health import HealthMonitor
    from src.host.lanes import LaneManager
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.orchestrator import Orchestrator
    from src.host.permissions import PermissionMatrix
    from src.host.runtime import DockerBackend, SandboxBackend, select_backend
    from src.host.server import create_mesh_app
    from src.host.transport import HttpTransport, SandboxTransport
    from src.host.webhooks import WebhookManager

    cfg = _load_config(Path(config_path))
    mesh_port = cfg["mesh"]["port"]
    agents_cfg = cfg.get("agents", {})

    if not agents_cfg:
        click.echo("No agents configured. Run: openlegion setup", err=True)
        return

    if not _check_docker_running():
        click.echo("Docker is not running. Please start Docker first.", err=True)
        sys.exit(1)

    # Select runtime backend (sandbox microVM if available, else containers)
    runtime = select_backend(
        mesh_host_port=mesh_port, project_root=str(PROJECT_ROOT),
    )
    backend_label = runtime.backend_name()
    is_sandbox = isinstance(runtime, SandboxBackend)

    if is_sandbox:
        transport = SandboxTransport()
        click.echo("Starting OpenLegion (microVM isolation)...\n")
    else:
        transport = HttpTransport()
        _ensure_docker_image()
        click.echo("Starting OpenLegion (container isolation)...\n")
        click.echo(
            "  WARNING: Docker Sandbox not detected. Agents are running in "
            "standard containers\n"
            "  (shared host kernel). For hypervisor-level isolation, install "
            "Docker Desktop 4.58+\n"
            "  and enable Docker Sandbox. See: "
            "https://docs.docker.com/sandbox/\n",
            err=True,
        )

    blackboard = Blackboard()
    pubsub = PubSub(db_path="pubsub.db")
    permissions = PermissionMatrix()
    cost_tracker = CostTracker()
    failover_config = cfg.get("llm", {}).get("failover", {})
    credential_vault = CredentialVault(
        cost_tracker=cost_tracker,
        failover_config=failover_config or None,
    )
    router = MessageRouter(permissions, {})

    orchestrator = Orchestrator(
        mesh_url=f"http://localhost:{mesh_port}",
        blackboard=blackboard,
        pubsub=pubsub,
        container_manager=runtime,
    )

    default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
    agent_urls: dict[str, str] = {}

    for agent_id, agent_cfg in agents_cfg.items():
        budget = agent_cfg.get("budget", {})
        if budget:
            cost_tracker.set_budget(
                agent_id,
                daily_usd=budget.get("daily_usd", 10.0),
                monthly_usd=budget.get("monthly_usd", 200.0),
            )
        skills_dir = os.path.abspath(agent_cfg.get("skills_dir", ""))
        agent_model = agent_cfg.get("model", default_model)
        try:
            url = runtime.start_agent(
                agent_id=agent_id,
                role=agent_cfg["role"],
                skills_dir=skills_dir,
                system_prompt=agent_cfg.get("system_prompt", ""),
                model=agent_model,
            )
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            if isinstance(runtime, SandboxBackend):
                click.echo(
                    f"\n  Sandbox failed for '{agent_id}': {exc}\n"
                    "  Falling back to Docker container isolation...\n",
                    err=True,
                )
                runtime = DockerBackend(
                    mesh_host_port=mesh_port,
                    use_host_network=True,
                    project_root=str(PROJECT_ROOT),
                )
                transport = HttpTransport()
                _ensure_docker_image()
                url = runtime.start_agent(
                    agent_id=agent_id,
                    role=agent_cfg["role"],
                    skills_dir=skills_dir,
                    system_prompt=agent_cfg.get("system_prompt", ""),
                    model=agent_model,
                )
            else:
                raise
        router.register_agent(agent_id, url)
        agent_urls[agent_id] = url
        if isinstance(transport, HttpTransport):
            transport.register(agent_id, url)

    health_monitor = HealthMonitor(
        runtime=runtime, transport=transport, router=router,
    )
    for agent_id in agents_cfg:
        health_monitor.register(agent_id)

    async def _direct_dispatch(agent_name: str, message: str) -> str:
        try:
            result = await transport.request(
                agent_name, "POST", "/chat", json={"message": message},
            )
            return result.get("response", "(no response)")
        except Exception as e:
            return f"Error: {e}"

    lane_manager = LaneManager(dispatch_fn=_direct_dispatch)

    # Dedicated event loop for dispatching — all asyncio primitives
    # (lane queues, futures, tasks) live here. Thread-safe from any caller.
    _dispatch_loop = asyncio.new_event_loop()

    def _run_dispatch_loop():
        asyncio.set_event_loop(_dispatch_loop)
        _dispatch_loop.run_forever()

    _dispatch_thread = threading.Thread(target=_run_dispatch_loop, daemon=True)
    _dispatch_thread.start()

    async def dispatch_to_agent(agent_name: str, message: str) -> str:
        """Thread-safe dispatch: schedules onto the dedicated dispatch loop."""
        future = asyncio.run_coroutine_threadsafe(
            lane_manager.enqueue(agent_name, message), _dispatch_loop,
        )
        # If we're already in an event loop (Telegram, cron), await via a
        # thread-pool so we don't block the caller's loop.
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None:
            return await running_loop.run_in_executor(None, future.result)
        return future.result()

    active_channels: list = []

    async def cron_dispatch(agent_name: str, message: str) -> str:
        """Dispatch for cron -- prints the response to the REPL and channels."""
        result = await dispatch_to_agent(agent_name, message)
        if result and result.strip():
            notification = f"[cron -> {agent_name}] {result}"
            sys.stdout.write(f"\n{notification}\nYou> ")
            sys.stdout.flush()
            for ch in active_channels:
                try:
                    await ch.send_notification(notification)
                except Exception:
                    pass
        return result

    async def trigger_workflow(workflow_name: str, payload: dict) -> str:
        exec_id = await orchestrator.trigger_workflow(workflow_name, payload)
        return f"workflow:{exec_id}"

    cron_scheduler = CronScheduler(
        dispatch_fn=cron_dispatch,
        workflow_trigger_fn=trigger_workflow,
        blackboard=blackboard,
    )
    if cron_scheduler.jobs:
        click.echo(f"  Cron scheduler: {len(cron_scheduler.jobs)} jobs loaded")

    webhook_manager = WebhookManager(dispatch_fn=dispatch_to_agent)

    app = create_mesh_app(
        blackboard, pubsub, router, permissions, credential_vault,
        cron_scheduler, runtime, transport, orchestrator,
    )
    app.include_router(create_webhook_router(orchestrator))
    app.include_router(webhook_manager.create_router())

    # Start mesh server in background
    server_config = uvicorn.Config(app, host="0.0.0.0", port=mesh_port, log_level="warning")
    server = uvicorn.Server(server_config)
    mesh_thread = threading.Thread(target=server.run, daemon=True)
    mesh_thread.start()

    # Wait for mesh to be ready
    mesh_ready = False
    for _ in range(30):
        try:
            httpx.get(f"http://localhost:{mesh_port}/mesh/agents", timeout=1)
            mesh_ready = True
            break
        except Exception:
            time.sleep(0.5)

    if not mesh_ready:
        click.echo(
            f"Mesh server failed to start on port {mesh_port}. "
            f"Port may be in use. Try: openlegion stop",
            err=True,
        )
        runtime.stop_all()
        return

    click.echo(f"  Mesh host ready on port {mesh_port}")
    click.echo(f"  Isolation: {backend_label}")

    # Wait for agents to be ready (concurrently)
    async def _wait_all_agents():
        async def _wait_one(aid):
            ready = await runtime.wait_for_agent(aid, timeout=60)
            return aid, ready
        return await asyncio.gather(*[_wait_one(aid) for aid in agents_cfg])

    agent_results = asyncio.run(_wait_all_agents())
    for agent_id, ready in agent_results:
        if ready:
            click.echo(f"  Agent '{agent_id}' ready")
        else:
            logs = runtime.get_logs(agent_id, tail=15)
            click.echo(f"  Agent '{agent_id}' failed to start", err=True)
            if logs:
                click.echo(logs, err=True)

    # Start cron in background
    def run_cron():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(cron_scheduler.start())

    cron_thread = threading.Thread(target=run_cron, daemon=True)
    cron_thread.start()

    # Start health monitor in background
    def run_health():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(health_monitor.start())

    health_thread = threading.Thread(target=run_health, daemon=True)
    health_thread.start()

    # Channel callback helpers (mirror REPL capabilities)
    def _channel_status(agent_name: str) -> dict | None:
        try:
            return transport.request_sync(agent_name, "GET", "/status", timeout=3)
        except Exception:
            return None

    def _channel_costs() -> list[dict]:
        return cost_tracker.get_all_agents_spend("today")

    def _channel_reset(agent_name: str) -> bool:
        try:
            transport.request_sync(agent_name, "POST", "/chat/reset", timeout=5)
            return True
        except Exception:
            return False

    async def stream_dispatch_to_agent(agent_name: str, message: str):
        """Streaming dispatch for channels — yields SSE events."""
        async for event in transport.stream_request(
            agent_name, "POST", "/chat/stream",
            json={"message": message}, timeout=120,
        ):
            yield event

    # Start messaging channels (Telegram, Discord) if configured
    pairing_instructions = _start_channels(
        cfg, dispatch_to_agent, router.agent_registry, active_channels,
        status_fn=_channel_status,
        costs_fn=_channel_costs,
        reset_fn=_channel_reset,
        stream_dispatch_fn=stream_dispatch_to_agent,
    )

    # Pick the first agent as the default active one
    active_agents = list(agents_cfg.keys())
    active_agent = active_agents[0]

    click.echo(f"\nChatting with '{active_agent}'.", nl=False)
    if len(active_agents) > 1:
        click.echo(" Use @agent to direct messages. /help for commands.")
    else:
        click.echo(" /help for commands.")

    # Pairing instructions last — right above the prompt where the user looks
    if pairing_instructions:
        click.echo("")
        for instruction in pairing_instructions:
            click.echo(instruction)
    click.echo("")

    # Interactive REPL
    try:
        _multi_agent_repl(
            active_agent, agent_urls, router, cost_tracker, runtime, cfg,
            transport=transport, dispatch_loop=_dispatch_loop,
            credential_vault=credential_vault,
        )
    except KeyboardInterrupt:
        click.echo("")
    finally:
        click.echo("Stopping OpenLegion...")
        _stop_channels(active_channels)
        health_monitor.stop()
        cron_scheduler.stop()
        runtime.stop_all()
        cost_tracker.close()
        pubsub.close()
        blackboard.close()
        # Close shared httpx clients on the dispatch loop (where they were created)
        import asyncio as _asyncio
        async def _close_clients():
            for closeable in [transport, router, orchestrator, credential_vault]:
                if hasattr(closeable, 'close') and callable(closeable.close):
                    try:
                        result = closeable.close()
                        if hasattr(result, '__await__'):
                            await result
                    except Exception:
                        pass
        try:
            future = _asyncio.run_coroutine_threadsafe(_close_clients(), _dispatch_loop)
            future.result(timeout=5)
        except Exception:
            pass
        _dispatch_loop.call_soon_threadsafe(_dispatch_loop.stop)
        server.should_exit = True
        click.echo("Stopped.")


def _start_detached(config_path: str) -> None:
    """Start the runtime in a background subprocess."""
    import subprocess

    cmd = [sys.executable, "-m", "src.cli", "start", "--config", config_path]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    # Read output until agents are ready or timeout
    started_lines: list[str] = []
    import select
    deadline = time.time() + 90
    while time.time() < deadline:
        ready_fds, _, _ = select.select([proc.stdout], [], [], 1.0)
        if ready_fds:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.rstrip()
            started_lines.append(line)
            click.echo(line)
            if "Chatting with" in line:
                break
        if proc.poll() is not None:
            break

    if proc.poll() is not None:
        click.echo("Runtime failed to start. Check logs.", err=True)
        for line in started_lines:
            click.echo(f"  {line}", err=True)
        return

    click.echo(f"\nOpenLegion running in background (PID {proc.pid}).")
    click.echo("  Chat with an agent:  openlegion chat <name>")
    click.echo("  Stop the runtime:    openlegion stop")


# ── chat (for detached mode) ─────────────────────────────────

@cli.command("chat")
@click.argument("name")
@click.option("--port", default=8420, type=int, help="Mesh host port")
def chat(name: str, port: int):
    """Connect to a running agent and start chatting.

    The runtime must already be running (openlegion start -d).
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

    agent_info = agents.get(name)
    if not agent_info:
        available = ", ".join(agents.keys()) if agents else "(none)"
        click.echo(f"Agent '{name}' is not running. Running agents: {available}", err=True)
        return

    agent_url = agent_info.get("url", agent_info) if isinstance(agent_info, dict) else agent_info
    click.echo(f"Connected to '{name}' at {agent_url}")
    click.echo("Type a message to chat. /help for commands.\n")

    try:
        _single_agent_repl(agent_url)
    except KeyboardInterrupt:
        click.echo("\nDisconnected.")


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
    except Exception:
        pass

    if not configured and not mesh_agents:
        click.echo("No agents configured. Run: openlegion setup")
        return

    all_names = sorted(set(list(configured.keys()) + list(mesh_agents.keys())))

    click.echo(f"{'Agent':<16} {'Role':<20} {'Status':<12}")
    click.echo("-" * 48)
    for name in all_names:
        role = configured.get(name, {}).get("role", "n/a")

        if name in mesh_agents:
            agent_url = mesh_agents[name]
            if isinstance(agent_url, dict):
                agent_url = agent_url.get("url", "")
            try:
                sr = httpx.get(f"{agent_url}/status", timeout=3)
                state = sr.json().get("state", "running")
            except Exception:
                state = "unreachable"
        elif mesh_online:
            state = "stopped"
        else:
            state = "unknown"

        click.echo(f"{name:<16} {role:<20} {state:<12}")

    if not mesh_online:
        click.echo("\nMesh is not running. Start with: openlegion start")


# ── stop ─────────────────────────────────────────────────────

@cli.command()
def stop():
    """Stop all agent containers."""
    import docker

    client = docker.from_env()
    containers = client.containers.list(filters={"name": "openlegion_"})
    if not containers:
        click.echo("No OpenLegion containers running.")
        return
    for container in containers:
        click.echo(f"Stopping {container.name}...")
        container.stop(timeout=10)
        container.remove()
    click.echo(f"Stopped {len(containers)} container(s).")


# ── Streaming message helper ──────────────────────────────────

def _send_message_streaming(
    transport, target: str, message: str,
    dispatch_loop=None,
) -> None:
    """Send a message via streaming endpoint, falling back to non-streaming.

    When *dispatch_loop* is provided, the async work runs on that loop
    (via ``run_coroutine_threadsafe``) so the httpx client is shared with
    the lane-manager dispatch path.  Otherwise falls back to ``asyncio.run()``.
    """
    import asyncio

    from src.host.transport import HttpTransport

    if isinstance(transport, HttpTransport):
        # Use streaming path
        async def _stream():
            response_text = ""
            tool_count = 0
            last_tool_input = {}
            try:
                async for event in transport.stream_request(
                    target, "POST", "/chat/stream",
                    json={"message": message}, timeout=120,
                ):
                    if isinstance(event, dict):
                        etype = event.get("type", "")
                        if etype == "tool_start":
                            tool_count += 1
                            name = event.get("name", "?")
                            last_tool_input = event.get("input", {})
                            summary = _format_tool_summary(name, last_tool_input, {})
                            click.echo(f"  [{tool_count}] {name}: {summary}", nl=False)
                            sys.stdout.flush()
                        elif etype == "tool_result":
                            name = event.get("name", "?")
                            output = event.get("output", {})
                            out = output if isinstance(output, dict) else {}
                            # Show concise result after the tool line
                            result_hint = _format_tool_result_hint(name, out)
                            if result_hint:
                                click.echo(f" → {result_hint}")
                            else:
                                click.echo(" ✓")
                        elif etype == "text_delta":
                            # Progressive text rendering
                            content = event.get("content", "")
                            if not response_text:
                                sys.stdout.write(f"\n{target}> ")
                            sys.stdout.write(content)
                            sys.stdout.flush()
                            response_text += content
                        elif etype == "done":
                            if not response_text:
                                resp = event.get("response", "(no response)")
                                click.echo(f"\n{target}> {resp}")
                            else:
                                click.echo("")  # newline after streamed text
                            return
                        elif "error" in event:
                            click.echo(f"Error: {event['error']}")
                            return
            except Exception:
                # Fallback to non-streaming on any stream error
                data = await transport.request(
                    target, "POST", "/chat",
                    json={"message": message}, timeout=120,
                )
                if "error" in data and "response" not in data:
                    click.echo(f"Error: {data['error']}")
                    return
                for tool_out in data.get("tool_outputs", []):
                    tool_name = tool_out.get("tool", "unknown")
                    tool_input = tool_out.get("input", {})
                    tool_result = tool_out.get("output", {})
                    summary = _format_tool_summary(tool_name, tool_input, {})
                    result_hint = _format_tool_result_hint(tool_name, tool_result if isinstance(tool_result, dict) else {})
                    line = f"  {tool_name}: {summary}"
                    if result_hint:
                        line += f" → {result_hint}"
                    click.echo(line)
                click.echo(f"\n{target}> {data.get('response', '(no response)')}")

            click.echo("")

        if dispatch_loop is not None:
            future = asyncio.run_coroutine_threadsafe(_stream(), dispatch_loop)
            future.result()  # blocks until done (main thread has no loop)
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
        for tool_out in data.get("tool_outputs", []):
            tool_name = tool_out.get("tool", "unknown")
            tool_input = tool_out.get("input", {})
            tool_result = tool_out.get("output", {})
            summary = _format_tool_summary(tool_name, tool_input, {})
            result_hint = _format_tool_result_hint(tool_name, tool_result if isinstance(tool_result, dict) else {})
            line = f"  {tool_name}: {summary}"
            if result_hint:
                line += f" → {result_hint}"
            click.echo(line)
        click.echo(f"\n{target}> {data.get('response', '(no response)')}\n")


# ── Multi-agent REPL (used by `start`) ───────────────────────

def _multi_agent_repl(
    active_agent: str,
    agent_urls: dict[str, str],
    router: object,
    cost_tracker: object,
    container_manager: object,
    cfg: dict,
    transport: object | None = None,
    dispatch_loop=None,
    credential_vault: object | None = None,
) -> None:
    """Interactive REPL supporting multiple agents, @mentions, and slash commands."""
    import concurrent.futures

    from src.host.transport import HttpTransport, resolve_url

    def _resolve_url(agent_id: str) -> str | None:
        return resolve_url(router.agent_registry, agent_id)

    current = active_agent

    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        # @agent prefix: send to a specific agent
        target = current
        message = user_input
        if user_input.startswith("@"):
            parts = user_input.split(None, 1)
            mentioned = parts[0][1:]
            if mentioned in agent_urls:
                target = mentioned
                message = parts[1] if len(parts) > 1 else ""
                if not message:
                    click.echo(f"Usage: @{mentioned} <message>")
                    continue
            else:
                click.echo(f"Unknown agent: '{mentioned}'. Type /agents to list.")
                continue

        # Slash commands
        if message.startswith("/"):
            cmd_parts = message.split(None, 1)
            cmd = cmd_parts[0].lower()

            if cmd in ("/quit", "/exit"):
                break

            elif cmd == "/agents":
                for name in agent_urls:
                    marker = " (active)" if name == current else ""
                    click.echo(f"  {name}{marker}")
                continue

            elif cmd == "/use":
                if len(cmd_parts) < 2:
                    click.echo(f"Usage: /use <agent>  (current: {current})")
                    continue
                new_agent = cmd_parts[1].strip()
                if new_agent not in agent_urls:
                    click.echo(f"Unknown agent: '{new_agent}'. Type /agents to list.")
                    continue
                current = new_agent
                click.echo(f"Now chatting with '{current}'.")
                continue

            elif cmd == "/add":
                new_name = click.prompt("Agent name")
                if new_name in agent_urls:
                    click.echo(f"Agent '{new_name}' already exists.")
                    continue
                new_desc = click.prompt(
                    "What should this agent do?",
                    default=f"General-purpose {new_name} assistant",
                )
                model = _get_default_model()
                _create_agent(new_name, new_desc, model)
                agent_cfg_data = _load_config().get("agents", {}).get(new_name, {})
                skills_dir = os.path.abspath(agent_cfg_data.get("skills_dir", ""))
                import asyncio
                url = container_manager.start_agent(
                    agent_id=new_name,
                    role=new_desc,
                    skills_dir=skills_dir,
                    system_prompt=agent_cfg_data.get("system_prompt", ""),
                    model=agent_cfg_data.get("model", model),
                )
                router.register_agent(new_name, url)
                agent_urls[new_name] = url
                if isinstance(transport, HttpTransport):
                    transport.register(new_name, url)
                click.echo(f"Starting '{new_name}'...")
                ready = asyncio.run(container_manager.wait_for_agent(new_name, timeout=60))
                if ready:
                    click.echo(f"Agent '{new_name}' ready.")
                else:
                    click.echo(f"Agent '{new_name}' failed to start.", err=True)
                continue

            elif cmd == "/status":
                for name in agent_urls:
                    try:
                        data = transport.request_sync(name, "GET", "/status", timeout=3)
                        state = data.get("state", "unknown")
                        tasks = data.get("tasks_completed", 0)
                        click.echo(f"  {name}: {state} ({tasks} tasks completed)")
                    except Exception:
                        click.echo(f"  {name}: unreachable")
                continue

            elif cmd == "/broadcast":
                if len(cmd_parts) < 2:
                    click.echo("Usage: /broadcast <message>")
                    continue
                bc_msg = cmd_parts[1]
                click.echo(f"Broadcasting to {len(agent_urls)} agent(s)...\n")

                def _send(aid: str) -> tuple[str, str]:
                    try:
                        data = transport.request_sync(
                            aid, "POST", "/chat",
                            json={"message": bc_msg}, timeout=120,
                        )
                        return aid, data.get("response", "(no response)")
                    except Exception as e:
                        return aid, f"(error: {e})"

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(agent_urls)) as pool:
                    futures = {pool.submit(_send, aid): aid for aid in agent_urls}
                    for future in concurrent.futures.as_completed(futures):
                        aid, response = future.result()
                        click.echo(f"[{aid}] {response}\n")
                continue

            elif cmd == "/costs":
                try:
                    agents_spend = cost_tracker.get_all_agents_spend("today")
                    # Filter to active agents only
                    agents_spend = [a for a in agents_spend if a["agent"] in agent_urls]
                    if not agents_spend:
                        click.echo("No usage recorded today.")
                    else:
                        total = sum(a["cost"] for a in agents_spend)
                        click.echo(f"Today's spend: ${total:.4f}\n")
                        for a in agents_spend:
                            click.echo(f"  {a['agent']:<16} {a['tokens']:>8,} tokens  ${a['cost']:.4f}")
                    # Model health summary
                    model_health = credential_vault.get_model_health() if credential_vault else []
                    if model_health:
                        click.echo("\nModel health:")
                        for mh in model_health:
                            status = "ok" if mh["available"] else f"cooldown {mh['cooldown_remaining']:.0f}s"
                            click.echo(
                                f"  {mh['model']:<40} {status:<20} "
                                f"{mh['success_count']} ok / {mh['failure_count']} fail"
                            )
                except Exception as e:
                    click.echo(f"Error: {e}")
                continue

            elif cmd == "/reset":
                try:
                    transport.request_sync(current, "POST", "/chat/reset", timeout=5)
                    click.echo(f"Conversation with '{current}' reset.")
                except Exception as e:
                    click.echo(f"Error: {e}")
                continue

            elif cmd == "/help":
                click.echo("Commands:")
                click.echo("  @agent <msg>      Send message to a specific agent")
                click.echo("  /use <agent>      Switch active agent")
                click.echo("  /agents           List all agents")
                click.echo("  /add              Add a new agent")
                click.echo("  /status           Show agent health")
                click.echo("  /broadcast <msg>  Send to all agents")
                click.echo("  /costs            Show today's LLM spend")
                click.echo("  /reset            Clear conversation with active agent")
                click.echo("  /quit             Exit and stop runtime")
                continue

            else:
                click.echo(f"Unknown command: {cmd}. Type /help for commands.")
                continue

        # Send message to the target agent
        if target not in agent_urls:
            click.echo(f"Agent '{target}' not found.")
            continue

        try:
            _send_message_streaming(transport, target, message, dispatch_loop)
        except Exception as e:
            click.echo(f"Error: {e}")


# ── Single-agent REPL (used by `chat`) ───────────────────────

def _single_agent_repl(agent_url: str) -> None:
    """Interactive chat REPL with a single agent (for detached mode)."""
    import httpx

    while True:
        try:
            user_input = input("You> ").strip()
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
                click.echo("  /reset   - Clear conversation history")
                click.echo("  /status  - Show agent status")
                click.echo("  /quit    - Exit chat")
                click.echo("  /help    - Show this help")
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

            for tool_out in data.get("tool_outputs", []):
                tool_name = tool_out.get("tool", "unknown")
                tool_input = tool_out.get("input", {})
                tool_result = tool_out.get("output", {})
                summary = _format_tool_summary(tool_name, tool_input, {})
                result_hint = _format_tool_result_hint(tool_name, tool_result if isinstance(tool_result, dict) else {})
                line = f"  {tool_name}: {summary}"
                if result_hint:
                    line += f" → {result_hint}"
                click.echo(line)

            click.echo(f"\nAgent> {data.get('response', '(no response)')}\n")

        except httpx.TimeoutException:
            click.echo("Request timed out. The agent may still be processing.")
        except Exception as e:
            click.echo(f"Error: {e}")


def _format_tool_summary(name: str, inp: dict, out: dict) -> str:
    """One-line summary of tool INPUT for the REPL."""
    if name == "exec":
        cmd = inp.get("command", "")
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."
        return f"`{cmd}`"
    elif name in ("read_file", "write_file", "list_files"):
        return inp.get("path", inp.get("directory", ""))
    elif name == "http_request":
        return f"{inp.get('method', 'GET')} {inp.get('url', '')}"
    elif name == "browser_navigate":
        return inp.get("url", "")
    elif name == "browser_click":
        return inp.get("selector", "")
    elif name == "browser_type":
        sel = inp.get("selector", "")
        text = inp.get("text", "")
        if len(text) > 30:
            text = text[:27] + "..."
        return f"{sel} ← \"{text}\""
    elif name == "browser_evaluate":
        script = inp.get("script", "")
        # Show first meaningful line of JS
        for line in script.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                if len(line) > 60:
                    line = line[:57] + "..."
                return line
        return script[:60]
    elif name == "browser_screenshot":
        return inp.get("filename", "screenshot.png")
    elif name == "web_search":
        return inp.get("query", "")
    elif name == "memory_save":
        return inp.get("key", inp.get("content", ""))[:60]
    elif name == "memory_search":
        return inp.get("query", "")
    else:
        text = json.dumps(inp, default=str)
        if len(text) > 80:
            text = text[:77] + "..."
        return text


def _format_tool_result_hint(name: str, out: dict) -> str:
    """Concise result hint shown after tool completion."""
    if "error" in out:
        err = str(out["error"])
        if len(err) > 60:
            err = err[:57] + "..."
        return f"error: {err}"
    if name == "exec":
        code = out.get("exit_code", 0)
        return f"exit {code}" if code != 0 else ""
    elif name == "browser_navigate":
        title = out.get("title", "")
        status = out.get("status", "")
        return f"{status} {title}".strip()[:60] if title else ""
    elif name == "browser_click":
        return out.get("url", "")[:60] if out.get("url") else ""
    elif name == "browser_type":
        return ""  # success is implicit
    elif name == "browser_screenshot":
        return out.get("path", "")
    elif name == "web_search":
        results = out.get("results", [])
        return f"{len(results)} results" if results else ""
    return ""


if __name__ == "__main__":
    cli()
