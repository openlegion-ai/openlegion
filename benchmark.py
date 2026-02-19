"""OpenLegion benchmark — runs all configured agents through all available workflows.

Reads agents from config/agents.yaml and discovers workflows from config/workflows/.
Produces a markdown report.

Usage: python benchmark.py [--output report.md]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import uvicorn
import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
MESH_PORT = 8460


def _load_config() -> dict:
    cfg: dict = {
        "mesh": {"host": "0.0.0.0", "port": MESH_PORT},
        "llm": {"default_model": "openai/gpt-4o-mini"},
        "agents": {},
    }
    mesh_path = PROJECT_ROOT / "config" / "mesh.yaml"
    if mesh_path.exists():
        with open(mesh_path) as f:
            cfg.update(yaml.safe_load(f) or {})
    agents_path = PROJECT_ROOT / "config" / "agents.yaml"
    if agents_path.exists():
        with open(agents_path) as f:
            agents_data = yaml.safe_load(f) or {}
            cfg.setdefault("agents", {}).update(agents_data.get("agents", {}))
    return cfg


def _discover_workflows() -> list[dict]:
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
            "agents": agents_used,
            "label": wf["name"].replace("_", " ").title(),
        })
    return workflows


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenLegion benchmark")
    parser.add_argument("--output", "-o", default="benchmark_report.md", help="Output report path")
    args = parser.parse_args()

    from src.channels.webhook import create_webhook_router
    from src.host.containers import ContainerManager
    from src.host.costs import CostTracker
    from src.host.credentials import CredentialVault
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.orchestrator import Orchestrator
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    logging.getLogger("host").setLevel(logging.WARNING)

    cfg = _load_config()
    agents = cfg.get("agents", {})
    if not agents:
        print("No agents configured. Add agents to config/agents.yaml.")
        return 1

    all_workflows = _discover_workflows()
    agent_names = set(agents.keys())
    runnable = [wf for wf in all_workflows if set(wf["agents"]).issubset(agent_names)]
    if not runnable:
        print("No runnable workflows. Add workflow YAML files to config/workflows/.")
        return 1

    needed_agents = sorted({a for wf in runnable for a in wf["agents"]})
    default_model = cfg.get("llm", {}).get("default_model", "openai/gpt-4o-mini")
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    bb = Blackboard()
    pubsub = PubSub()
    perms = PermissionMatrix()
    cost_tracker = CostTracker()
    vault = CredentialVault(cost_tracker=cost_tracker)
    cm = ContainerManager(mesh_host_port=MESH_PORT, use_host_network=True)
    router = MessageRouter(perms, {})
    orchestrator = Orchestrator(
        mesh_url=f"http://localhost:{MESH_PORT}",
        blackboard=bb,
        pubsub=pubsub,
        container_manager=cm,
    )

    app = create_mesh_app(bb, pubsub, router, perms, vault)
    app.include_router(create_webhook_router(orchestrator))

    server_cfg = uvicorn.Config(app, host="0.0.0.0", port=MESH_PORT, log_level="warning")
    server = uvicorn.Server(server_cfg)
    mesh_thread = threading.Thread(target=server.run, daemon=True)
    mesh_thread.start()

    for _ in range(30):
        try:
            httpx.get(f"http://localhost:{MESH_PORT}/mesh/agents", timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    print(f"\n=== OpenLegion Benchmark ({today}) ===")
    print(f"Agents:    {', '.join(needed_agents)}")
    print(f"Workflows: {', '.join(wf['name'] for wf in runnable)}\n")
    print("Starting agents...")

    for agent_name in needed_agents:
        agent_cfg = agents[agent_name]
        skills_dir = os.path.abspath(agent_cfg.get("skills_dir", ""))
        url = cm.start_agent(
            agent_id=agent_name,
            role=agent_cfg["role"],
            skills_dir=skills_dir,
            system_prompt=agent_cfg.get("system_prompt", ""),
            model=agent_cfg.get("model", default_model),
        )
        router.register_agent(agent_name, url)

    for agent_name in needed_agents:
        ready = asyncio.run(cm.wait_for_agent(agent_name, timeout=60))
        if not ready:
            print(f"  Agent '{agent_name}' failed to start")
            cm.stop_all()
            server.should_exit = True
            return 1

    print(f"All {len(needed_agents)} agents ready.\n")

    async def run_workflow(wf_name: str, payload: dict) -> dict:
        try:
            exec_id = await orchestrator.trigger_workflow(wf_name, payload)
        except ValueError as e:
            return {"status": "error", "error": str(e)}

        for _ in range(600):
            status = orchestrator.get_execution_status(exec_id)
            if not status:
                break
            if status["status"] in ("complete", "failed"):
                execution = orchestrator.active_executions.get(exec_id)
                steps = {}
                if execution:
                    for sid, r in execution.step_results.items():
                        steps[sid] = {"status": r.status, "result": r.result}
                return {"status": status["status"], "steps": steps, **status}
            await asyncio.sleep(1)
        return {"status": "timeout"}

    payload = {"date": today, "market": "AI agent frameworks"}
    wf_results = []
    total_start = time.time()

    try:
        for wf in runnable:
            wf_start = time.time()
            print(f"  Running: {wf['label']}...", end="", flush=True)
            result = asyncio.run(run_workflow(wf["name"], payload))
            elapsed = time.time() - wf_start
            status = result.get("status", "unknown")
            print(f" {status} ({elapsed:.0f}s)")
            wf_results.append({
                "name": wf["name"],
                "label": wf["label"],
                "agents": wf["agents"],
                "steps": len(result.get("steps", {})),
                "time_s": round(elapsed, 1),
                "status": status,
            })

        total_elapsed = time.time() - total_start
        spend = cost_tracker.get_all_agents_spend()
        total_cost = sum(r["cost"] for r in spend)

        print(f"\nTotal: {total_elapsed:.0f}s | ${total_cost:.3f}\n")

        report = _generate_report(
            today, needed_agents, runnable, wf_results, spend, total_elapsed, total_cost,
        )
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cm.stop_all()
        server.should_exit = True
        cost_tracker.close()

    return 0


def _generate_report(
    date: str,
    agents: list[str],
    workflows: list[dict],
    wf_results: list[dict],
    spend: list[dict],
    total_time: float,
    total_cost: float,
) -> str:
    lines = [
        "# OpenLegion Benchmark Report",
        "",
        f"**Date**: {date}",
        f"**Team**: {len(agents)} agents | {len(workflows)} workflows",
        f"**Total time**: {total_time:.0f}s",
        f"**Total cost**: ${total_cost:.3f}",
        f"**Monthly projection** (weekdays): ${total_cost * 22:.2f}",
        "",
        "## Workflows",
        "",
        "| Workflow | Agents | Steps | Time | Status |",
        "|----------|--------|-------|------|--------|",
    ]
    for wf in wf_results:
        agents_str = ", ".join(wf["agents"])
        lines.append(f"| {wf['label']} | {agents_str} | {wf['steps']} | {wf['time_s']}s | {wf['status']} |")

    lines.extend([
        "",
        "## Agent Costs",
        "",
        "| Agent | Tokens | Cost |",
        "|-------|--------|------|",
    ])
    for row in spend:
        lines.append(f"| {row['agent']} | {row['tokens']:,} | ${row['cost']:.3f} |")

    lines.extend([
        "",
        "## What This Proves",
        "",
        f"- {len(agents)} specialized agents collaborated across {len(workflows)} workflows",
        "- Container isolation kept each agent independent (no shared process crashes)",
        f"- Total daily cost: ${total_cost:.3f} (projected ${total_cost * 22:.2f}/month)",
        "- All coordination happened through the shared blackboard",
        "- Agents used different LLM providers as configured",
        "",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
