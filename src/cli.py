"""CLI entry point for OpenLegion.

Commands:
  quickstart         - One-command setup: set API key, create agent, start chatting
  start              - Start the mesh host process and all configured agents
  stop               - Stop all agent containers
  trigger            - Trigger a workflow via the running mesh
  status             - Show status of all registered agents
  agent create       - Create a new agent interactively
  agent chat         - Interactive REPL with an agent
  agent list         - List configured agents
  config set-key     - Save an API key to .env
  cron add/list/run/pause/resume/remove - Manage scheduled jobs
  webhook add/list/test/remove          - Manage webhook endpoints
  costs              - View per-agent LLM spend
"""

from __future__ import annotations

import json
import os
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
PERMISSIONS_FILE = PROJECT_ROOT / "config" / "permissions.json"


# ── Helpers ──────────────────────────────────────────────────

def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {"mesh": {"host": "0.0.0.0", "port": 8420}, "llm": {"default_model": "openai/gpt-4o-mini"}, "agents": {}}
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


def _save_config(cfg: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


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


def _check_docker_image() -> bool:
    """Check if the agent Docker image exists."""
    try:
        import docker
        client = docker.from_env()
        client.images.get("openlegion-agent:latest")
        return True
    except Exception:
        return False


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
    """Add an agent entry to mesh.yaml."""
    cfg = _load_config()
    if "agents" not in cfg:
        cfg["agents"] = {}

    skills_dir = f"./skills/{name}"
    cfg["agents"][name] = {
        "role": role,
        "model": model,
        "skills_dir": skills_dir,
        "system_prompt": system_prompt,
        "resources": {"memory_limit": "512m", "cpu_limit": 0.5},
    }
    _save_config(cfg)


def _add_agent_permissions(name: str) -> None:
    """Add default permissions for a new agent."""
    perms = _load_permissions()
    perms["permissions"][name] = {
        "can_message": ["orchestrator"],
        "can_publish": [f"{name}_complete"],
        "can_subscribe": [],
        "blackboard_read": ["context/*", "tasks/*"],
        "blackboard_write": [f"context/{name}_*"],
        "allowed_apis": ["llm"],
    }
    _save_permissions(perms)


# ── Main group ───────────────────────────────────────────────

@click.group()
def cli():
    """OpenLegion -- Secure multi-agent runtime."""
    pass


# ── quickstart ───────────────────────────────────────────────

@cli.command()
def quickstart():
    """One-command setup: set API key, create agent, start chatting."""

    click.echo("=== OpenLegion Quickstart ===\n")

    # 1. API key
    provider = click.prompt(
        "LLM provider",
        type=click.Choice(["openai", "anthropic", "groq"], case_sensitive=False),
        default="openai",
    )
    key_name = f"{provider}_api_key"
    existing = os.environ.get(f"OPENLEGION_CRED_{key_name.upper()}", "")
    if existing:
        click.echo(f"API key already set for {provider}.")
        if not click.confirm("Replace it?", default=False):
            click.echo("Keeping existing key.")
        else:
            api_key = click.prompt("API key", hide_input=True)
            _set_env_key(key_name, api_key)
    else:
        api_key = click.prompt(f"{provider.upper()} API key", hide_input=True)
        _set_env_key(key_name, api_key)

    # 2. Create default agent if none exist
    cfg = _load_config()
    agent_name = "assistant"
    if not cfg.get("agents"):
        model_default = f"{provider}/gpt-4o-mini" if provider == "openai" else f"{provider}/claude-sonnet-4-5-20250929"
        model = click.prompt("Default model", default=model_default)
        _add_agent_to_config(
            name=agent_name,
            role="assistant",
            model=model,
            system_prompt=(
                "You are a general-purpose assistant. You can run commands, "
                "read and write files, browse the web, and make HTTP requests. "
                "Help the user accomplish their goals."
            ),
        )
        skills_dir = PROJECT_ROOT / "skills" / agent_name
        skills_dir.mkdir(parents=True, exist_ok=True)
        _add_agent_permissions(agent_name)
        click.echo(f"Created agent '{agent_name}'.")
    else:
        agent_name = next(iter(cfg["agents"]))
        click.echo(f"Using existing agent '{agent_name}'.")

    # 3. Build Docker image if needed
    if not _check_docker_image():
        _build_docker_image()

    # 4. Start mesh + agent + drop into chat
    click.echo("\nStarting OpenLegion...")
    _start_runtime_and_chat(agent_name)


# ── agent subgroup ───────────────────────────────────────────

@cli.group()
def agent():
    """Manage agents."""
    pass


@agent.command("create")
@click.argument("name")
def agent_create(name: str):
    """Create a new agent interactively."""
    cfg = _load_config()
    if name in cfg.get("agents", {}):
        click.echo(f"Agent '{name}' already exists.")
        return

    role = click.prompt("Role description (becomes agent identity)", default=name)
    model = click.prompt("Model", default=cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini"))
    system_prompt = click.prompt(
        "System prompt",
        default=f"You are the '{role}' agent. Help the user by using your tools and knowledge.",
    )

    skills_dir = PROJECT_ROOT / "skills" / name
    skills_dir.mkdir(parents=True, exist_ok=True)

    _add_agent_to_config(name, role, model, system_prompt)
    _add_agent_permissions(name)
    click.echo(f"Agent '{name}' created. Skills directory: skills/{name}/")
    click.echo(f"Start chatting: openlegion agent chat {name}")


@agent.command("chat")
@click.argument("name")
@click.option("--port", default=None, type=int, help="Mesh port override")
def agent_chat(name: str, port: int | None):
    """Interactive REPL with an agent."""
    cfg = _load_config()
    if name not in cfg.get("agents", {}):
        click.echo(f"Agent '{name}' not found. Create it first: openlegion agent create {name}")
        return

    _start_runtime_and_chat(name, mesh_port_override=port)


@agent.command("list")
def agent_list():
    """List all configured agents and their status."""
    cfg = _load_config()
    agents = cfg.get("agents", {})
    if not agents:
        click.echo("No agents configured. Create one: openlegion agent create <name>")
        return

    # Check if any containers are running
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

    click.echo(f"{'Name':<16} {'Role':<16} {'Model':<28} {'Status':<10}")
    click.echo("-" * 70)
    for name, info in agents.items():
        status = "running" if name in running else "stopped"
        click.echo(f"{name:<16} {info.get('role', 'n/a'):<16} {info.get('model', 'default'):<28} {status:<10}")


# ── config subgroup ──────────────────────────────────────────

@cli.group()
def config():
    """Configuration commands."""
    pass


@config.command("set-key")
@click.argument("provider")
@click.argument("key")
def config_set_key(provider: str, key: str):
    """Save an API key. Usage: openlegion config set-key openai sk-..."""
    _set_env_key(f"{provider}_api_key", key)
    click.echo(f"Saved {provider} API key to .env")


# ── start / stop / trigger / status (preserved) ─────────────

@cli.command()
@click.option("--config", "config_path", default="config/mesh.yaml", help="Path to mesh config")
def start(config_path: str):
    """Start the OpenLegion runtime (mesh + all configured agents + cron + webhooks)."""
    import asyncio
    import threading

    import httpx
    import uvicorn

    from src.channels.webhook import create_webhook_router
    from src.host.containers import ContainerManager
    from src.host.costs import CostTracker
    from src.host.credentials import CredentialVault
    from src.host.cron import CronScheduler
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.orchestrator import Orchestrator
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app
    from src.host.webhooks import WebhookManager

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mesh_port = cfg["mesh"]["port"]

    blackboard = Blackboard()
    pubsub = PubSub()
    permissions = PermissionMatrix()
    cost_tracker = CostTracker()
    credential_vault = CredentialVault(cost_tracker=cost_tracker)
    container_manager = ContainerManager(mesh_host_port=mesh_port)

    router = MessageRouter(permissions, {})

    orchestrator = Orchestrator(
        mesh_url=f"http://localhost:{mesh_port}",
        blackboard=blackboard,
        pubsub=pubsub,
        container_manager=container_manager,
    )

    # Load per-agent budgets from config
    default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
    for agent_id, agent_cfg in cfg.get("agents", {}).items():
        budget = agent_cfg.get("budget", {})
        if budget:
            cost_tracker.set_budget(
                agent_id,
                daily_usd=budget.get("daily_usd", 10.0),
                monthly_usd=budget.get("monthly_usd", 200.0),
            )

        skills_dir = os.path.abspath(agent_cfg.get("skills_dir", ""))
        agent_model = agent_cfg.get("model", default_model)
        url = container_manager.start_agent(
            agent_id=agent_id,
            role=agent_cfg["role"],
            skills_dir=skills_dir,
            system_prompt=agent_cfg.get("system_prompt", ""),
            model=agent_model,
        )
        router.register_agent(agent_id, url)
        click.echo(f"Started agent '{agent_id}' at {url}")

    # Dispatch function: sends a chat message to an agent
    async def dispatch_to_agent(agent_name: str, message: str) -> str:
        agent_info = router.agent_registry.get(agent_name)
        if not agent_info:
            return f"Agent '{agent_name}' not found"
        agent_url = agent_info.get("url", agent_info) if isinstance(agent_info, dict) else agent_info
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{agent_url}/chat", json={"message": message})
            return r.json().get("response", "(no response)")

    # Wire cron scheduler
    cron_scheduler = CronScheduler(dispatch_fn=dispatch_to_agent)
    if cron_scheduler.jobs:
        click.echo(f"Cron scheduler: {len(cron_scheduler.jobs)} jobs loaded")

    # Wire webhook manager
    webhook_manager = WebhookManager(dispatch_fn=dispatch_to_agent)

    app = create_mesh_app(blackboard, pubsub, router, permissions, credential_vault)
    app.include_router(create_webhook_router(orchestrator))
    app.include_router(webhook_manager.create_router())

    # Start cron in background
    async def start_cron():
        await cron_scheduler.start()

    def run_cron():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start_cron())

    cron_thread = threading.Thread(target=run_cron, daemon=True)
    cron_thread.start()

    click.echo(f"OpenLegion running on port {mesh_port}")
    uvicorn.run(app, host=cfg["mesh"]["host"], port=mesh_port)


@cli.command()
@click.argument("workflow_name")
@click.argument("payload", type=str)
@click.option("--port", default=8420, help="Mesh port")
def trigger(workflow_name: str, payload: str, port: int):
    """Trigger a workflow via CLI."""
    import httpx

    response = httpx.post(
        f"http://localhost:{port}/webhook/trigger/{workflow_name}",
        json=json.loads(payload),
    )
    click.echo(json.dumps(response.json(), indent=2))


@cli.command("status")
@click.option("--port", default=8420, help="Mesh port")
def status(port: int):
    """Show status of all agents."""
    import httpx

    response = httpx.get(f"http://localhost:{port}/mesh/agents")
    agents = response.json()
    if not agents:
        click.echo("No agents registered.")
        return
    for agent_id, info in agents.items():
        click.echo(f"  {agent_id}: {info}")


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


# ── cron subgroup ────────────────────────────────────────────

@cli.group()
def cron():
    """Manage scheduled jobs."""
    pass


@cron.command("add")
@click.argument("agent_name")
@click.option("--schedule", "-s", required=True, help='Cron expression or "every 30m"')
@click.option("--message", "-m", required=True, help="Message to send to agent")
@click.option("--tz", default="UTC", help="Timezone (default UTC)")
def cron_add(agent_name: str, schedule: str, message: str, tz: str):
    """Add a scheduled job. Example: openlegion cron add researcher -s '0 9 * * 1-5' -m 'Morning check'"""
    from src.host.cron import CronScheduler

    scheduler = CronScheduler()
    job = scheduler.add_job(agent=agent_name, schedule=schedule, message=message, timezone=tz)
    click.echo(f"Created job {job.id}: agent={agent_name} schedule='{schedule}'")


@cron.command("list")
def cron_list():
    """List all scheduled jobs."""
    from src.host.cron import CronScheduler

    scheduler = CronScheduler()
    jobs = scheduler.list_jobs()
    if not jobs:
        click.echo("No cron jobs configured.")
        return
    click.echo(f"{'ID':<22} {'Agent':<14} {'Schedule':<18} {'Enabled':<8} {'Runs':<6} {'Last Run'}")
    click.echo("-" * 90)
    for j in jobs:
        last = j.get("last_run", "never")
        if last and last != "never":
            last = last[:19]
        click.echo(
            f"{j['id']:<22} {j['agent']:<14} {j['schedule']:<18} "
            f"{'yes' if j['enabled'] else 'no':<8} {j['run_count']:<6} {last}"
        )


@cron.command("run")
@click.argument("job_id")
@click.option("--port", default=8420, type=int, help="Mesh port")
def cron_run(job_id: str, port: int):
    """Manually trigger a cron job."""
    import asyncio

    import httpx

    from src.host.cron import CronScheduler

    async def dispatch(agent: str, message: str) -> str:
        r = httpx.post(
            f"http://localhost:{port}/mesh/agents",
            timeout=5,
        )
        agents = r.json()
        agent_url = agents.get(agent, {}).get("url")
        if not agent_url:
            return f"Agent '{agent}' not found in mesh"
        r = httpx.post(f"{agent_url}/chat", json={"message": message}, timeout=120)
        return r.json().get("response", "(no response)")

    scheduler = CronScheduler(dispatch_fn=dispatch)
    result = asyncio.run(scheduler.run_job(job_id))
    if result is None:
        click.echo(f"Job '{job_id}' not found.")
    else:
        click.echo(f"Response: {result}")


@cron.command("pause")
@click.argument("job_id")
def cron_pause(job_id: str):
    """Pause a cron job."""
    from src.host.cron import CronScheduler

    scheduler = CronScheduler()
    if scheduler.pause_job(job_id):
        click.echo(f"Paused job {job_id}")
    else:
        click.echo(f"Job '{job_id}' not found.")


@cron.command("resume")
@click.argument("job_id")
def cron_resume(job_id: str):
    """Resume a paused cron job."""
    from src.host.cron import CronScheduler

    scheduler = CronScheduler()
    if scheduler.resume_job(job_id):
        click.echo(f"Resumed job {job_id}")
    else:
        click.echo(f"Job '{job_id}' not found.")


@cron.command("remove")
@click.argument("job_id")
def cron_remove(job_id: str):
    """Remove a cron job."""
    from src.host.cron import CronScheduler

    scheduler = CronScheduler()
    if scheduler.remove_job(job_id):
        click.echo(f"Removed job {job_id}")
    else:
        click.echo(f"Job '{job_id}' not found.")


# ── webhook subgroup ─────────────────────────────────────────

@cli.group()
def webhook():
    """Manage webhook endpoints."""
    pass


@webhook.command("add")
@click.option("--agent", "-a", required=True, help="Target agent")
@click.option("--name", "-n", required=True, help="Webhook name")
@click.option("--port", default=8420, type=int, help="Mesh port")
def webhook_add(agent: str, name: str, port: int):
    """Create a new webhook endpoint."""
    from src.host.webhooks import WebhookManager

    mgr = WebhookManager()
    hook = mgr.add_hook(agent=agent, name=name)
    click.echo(f"Created webhook: {hook['id']}")
    click.echo(f"URL: http://localhost:{port}/webhook/hook/{hook['id']}")


@webhook.command("list")
def webhook_list():
    """List all webhook endpoints."""
    from src.host.webhooks import WebhookManager

    mgr = WebhookManager()
    hooks = mgr.list_hooks()
    if not hooks:
        click.echo("No webhooks configured.")
        return
    click.echo(f"{'ID':<18} {'Agent':<14} {'Name':<20} {'Calls':<6}")
    click.echo("-" * 60)
    for h in hooks:
        click.echo(f"{h['id']:<18} {h['agent']:<14} {h['name']:<20} {h.get('call_count', 0):<6}")


@webhook.command("test")
@click.argument("hook_id")
@click.option("--payload", "-p", default='{"event": "test"}', help="JSON payload")
@click.option("--port", default=8420, type=int, help="Mesh port")
def webhook_test(hook_id: str, payload: str, port: int):
    """Send a test payload to a webhook."""
    import httpx

    r = httpx.post(
        f"http://localhost:{port}/webhook/hook/{hook_id}",
        json=json.loads(payload),
        timeout=120,
    )
    click.echo(json.dumps(r.json(), indent=2))


@webhook.command("remove")
@click.argument("hook_id")
def webhook_remove(hook_id: str):
    """Remove a webhook endpoint."""
    from src.host.webhooks import WebhookManager

    mgr = WebhookManager()
    if mgr.remove_hook(hook_id):
        click.echo(f"Removed webhook {hook_id}")
    else:
        click.echo(f"Webhook '{hook_id}' not found.")


# ── costs command ────────────────────────────────────────────

@cli.command("costs")
@click.option("--agent", "-a", default=None, help="Filter by agent")
@click.option("--period", "-p", default="today", type=click.Choice(["today", "week", "month"]))
def costs(agent: str | None, period: str):
    """Show LLM spend per agent."""
    from src.host.costs import CostTracker

    tracker = CostTracker()
    if agent:
        spend = tracker.get_spend(agent, period)
        click.echo(f"Agent: {agent}  Period: {period}")
        click.echo(f"Total: ${spend['total_cost']:.4f}  Tokens: {spend['total_tokens']:,}")
        if spend["by_model"]:
            click.echo(f"\n{'Model':<32} {'Tokens':>10} {'Cost':>10}")
            click.echo("-" * 55)
            for model, info in spend["by_model"].items():
                click.echo(f"{model:<32} {info['total']:>10,} ${info['cost']:>9.4f}")
    else:
        agents = tracker.get_all_agents_spend(period)
        if not agents:
            click.echo(f"No usage recorded for period: {period}")
            return
        total = sum(a["cost"] for a in agents)
        click.echo(f"Period: {period}  Total: ${total:.4f}\n")
        click.echo(f"{'Agent':<16} {'Tokens':>12} {'Cost':>10}")
        click.echo("-" * 40)
        for a in agents:
            click.echo(f"{a['agent']:<16} {a['tokens']:>12,} ${a['cost']:>9.4f}")
    tracker.close()


# ── Runtime + Chat REPL ──────────────────────────────────────

def _start_runtime_and_chat(agent_name: str, mesh_port_override: int | None = None) -> None:
    """Start mesh + agent container, then enter interactive chat REPL."""
    import threading

    import httpx
    import uvicorn

    from src.host.containers import ContainerManager
    from src.host.credentials import CredentialVault
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    cfg = _load_config()
    mesh_port = mesh_port_override or cfg.get("mesh", {}).get("port", 8420)
    default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
    agent_cfg = cfg["agents"][agent_name]

    blackboard = Blackboard()
    pubsub = PubSub()
    permissions = PermissionMatrix()
    credential_vault = CredentialVault()
    container_manager = ContainerManager(mesh_host_port=mesh_port)

    router = MessageRouter(permissions, {})

    app = create_mesh_app(blackboard, pubsub, router, permissions, credential_vault)

    # Start mesh in background thread
    server_config = uvicorn.Config(app, host="0.0.0.0", port=mesh_port, log_level="warning")
    server = uvicorn.Server(server_config)
    mesh_thread = threading.Thread(target=server.run, daemon=True)
    mesh_thread.start()

    # Wait for mesh to be ready
    for _ in range(30):
        try:
            httpx.get(f"http://localhost:{mesh_port}/mesh/agents", timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    # Start agent container
    skills_dir = os.path.abspath(agent_cfg.get("skills_dir", ""))
    agent_model = agent_cfg.get("model", default_model)
    agent_url = container_manager.start_agent(
        agent_id=agent_name,
        role=agent_cfg["role"],
        skills_dir=skills_dir,
        system_prompt=agent_cfg.get("system_prompt", ""),
        model=agent_model,
    )
    router.register_agent(agent_name, agent_url)
    click.echo(f"Agent '{agent_name}' starting at {agent_url}...")

    # Wait for agent to be ready
    import asyncio
    ready = asyncio.run(container_manager.wait_for_agent(agent_name, timeout=60))
    if not ready:
        click.echo("Agent failed to start. Check Docker logs.", err=True)
        container_manager.stop_all()
        return

    click.echo(f"Agent '{agent_name}' is ready.\n")
    click.echo("Type a message to chat. Commands: /reset /status /quit /help\n")

    # REPL
    try:
        _chat_repl(agent_url)
    except KeyboardInterrupt:
        click.echo("\nExiting...")
    finally:
        click.echo("Stopping agent container...")
        container_manager.stop_all()
        server.should_exit = True


def _chat_repl(agent_url: str) -> None:
    """Interactive chat REPL with an agent."""
    import httpx

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit" or cmd == "/exit":
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

            # Show tool usage
            for tool_out in data.get("tool_outputs", []):
                tool_name = tool_out.get("tool", "unknown")
                tool_input = tool_out.get("input", {})
                tool_result = tool_out.get("output", {})
                click.echo(f"  [{tool_name}] {_format_tool_summary(tool_name, tool_input, tool_result)}")

            click.echo(f"\nAgent: {data.get('response', '(no response)')}\n")

        except httpx.TimeoutException:
            click.echo("Request timed out. The agent may still be processing.")
        except Exception as e:
            click.echo(f"Error: {e}")


def _format_tool_summary(name: str, inp: dict, out: dict) -> str:
    """One-line summary of a tool invocation for the REPL."""
    if name == "exec":
        cmd = inp.get("command", "")
        code = out.get("exit_code", "?")
        return f"`{cmd}` -> exit {code}"
    elif name in ("read_file", "write_file"):
        return inp.get("path", "")
    elif name == "http_request":
        return f"{inp.get('method', 'GET')} {inp.get('url', '')}"
    elif name == "browser_navigate":
        return inp.get("url", "")
    else:
        return json.dumps(inp, default=str)[:80]


if __name__ == "__main__":
    cli()
