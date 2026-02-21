"""Configuration loading, Docker helpers, and agent management."""

from __future__ import annotations

import json
import logging
import secrets
import subprocess
import sys
from pathlib import Path

import click
import yaml

logger = logging.getLogger("cli")

# ── Path constants ──────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_FILE = PROJECT_ROOT / "config" / "mesh.yaml"
AGENTS_FILE = PROJECT_ROOT / "config" / "agents.yaml"
PERMISSIONS_FILE = PROJECT_ROOT / "config" / "permissions.json"
PROJECT_FILE = PROJECT_ROOT / "PROJECT.md"
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
DOCKER_IMAGE = "openlegion-agent:latest"

MARKETPLACE_DIR = PROJECT_ROOT / "skills" / "_marketplace"

# ── Provider data ───────────────────────────────────────────

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

# ── Channel metadata ────────────────────────────────────────

CHANNEL_TYPES = {
    "telegram": {
        "label": "Telegram",
        "env_key": "telegram_bot_token",
        "config_section": "telegram",
        "token_help": "Get one from @BotFather on Telegram: https://t.me/BotFather",
    },
    "discord": {
        "label": "Discord",
        "env_key": "discord_bot_token",
        "config_section": "discord",
        "token_help": "Create one at: https://discord.com/developers/applications",
    },
    "slack": {
        "label": "Slack",
        "env_key": "slack_bot_token",
        "config_section": "slack",
        "token_help": "Create a Slack app at https://api.slack.com/apps -- enable Socket Mode",
    },
    "whatsapp": {
        "label": "WhatsApp",
        "env_key": "whatsapp_access_token",
        "config_section": "whatsapp",
        "token_help": "Set up at https://developers.facebook.com/apps -- WhatsApp Business API",
    },
}

# ── Config loading ──────────────────────────────────────────


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
    from src.host.credentials import _persist_to_env

    env_key = f"OPENLEGION_CRED_{name.upper()}"
    _persist_to_env(env_key, value, env_file=str(ENV_FILE))


# ── Docker helpers ──────────────────────────────────────────


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
        client.images.get(DOCKER_IMAGE)
        return True
    except Exception:
        return False


def _docker_image_is_stale() -> bool:
    """Check if source files are newer than the Docker image."""
    try:
        from datetime import datetime, timezone

        import docker
        client = docker.from_env()
        image = client.images.get(DOCKER_IMAGE)
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
    """Build the agent Docker image with visible progress."""
    click.echo("Building Docker image...")
    click.echo("  First build downloads base image + Chromium (~2 min). Rebuilds are fast.\n")
    proc = subprocess.Popen(
        [
            "docker", "build",
            "--platform", "linux/amd64",
            "-t", DOCKER_IMAGE,
            "-f", "Dockerfile.agent", ".",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("Step ") or line.startswith("#") or "Downloading" in line or "Installing" in line:
            click.echo(f"  {line}")
    proc.wait()
    if proc.returncode != 0:
        click.echo("Docker build failed. Run manually for full output:", err=True)
        click.echo(f"  docker build --platform linux/amd64 -t {DOCKER_IMAGE} -f Dockerfile.agent .", err=True)
        sys.exit(1)
    click.echo("\n  Docker image built successfully.")


# ── Agent management ────────────────────────────────────────


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
        "can_publish": ["*"] if collab else [f"{name}_complete"],
        "can_subscribe": ["*"] if collab else [],
        "blackboard_read": ["context/*", "tasks/*", "goals/*", "signals/*", "artifacts/*"],
        "blackboard_write": ["context/*", "goals/*", "signals/*", "artifacts/*"],
        "allowed_apis": ["llm"],
    }
    _save_permissions(perms)


def _set_collaborative_permissions() -> None:
    """Update all agent permissions to allow inter-agent messaging and pub/sub."""
    perms = _load_permissions()
    for name, p in perms.get("permissions", {}).items():
        if name == "default":
            continue
        if "*" not in p.get("can_message", []):
            p["can_message"] = list({*p.get("can_message", []), "*"})
        if "*" not in p.get("can_publish", []):
            p["can_publish"] = list({*p.get("can_publish", []), "*"})
        if "*" not in p.get("can_subscribe", []):
            p["can_subscribe"] = list({*p.get("can_subscribe", []), "*"})
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
    for name in [
        "host", "host.containers", "host.credentials", "host.orchestrator",
        "host.mesh", "host.costs", "host.permissions", "host.cron", "host.webhooks",
        "host.health", "host.lanes", "host.runtime", "host.watchers",
        "channels", "channels.base", "channels.telegram", "channels.discord",
        "channels.slack", "channels.whatsapp",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)


def _ensure_pairing_code(pairing_path: Path) -> str | None:
    """Ensure a pairing file exists with a code. Returns code if unpaired, None if already paired."""
    data: dict = {}
    if pairing_path.exists():
        try:
            data = json.loads(pairing_path.read_text())
        except (json.JSONDecodeError, OSError):
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
