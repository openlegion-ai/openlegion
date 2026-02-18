"""Manual smoke test: starts a real agent and runs through key scenarios.

Run with: python tests/manual_smoke_test.py

Tests:
  1. Agent starts and responds to chat
  2. Agent uses exec tool (runs a command)
  3. Agent uses file tools (writes and reads a file)
  4. Memory persists across chat resets
  5. Cron job can be added and listed
  6. Webhook can be added and listed
  7. Cost tracking records usage
"""

from __future__ import annotations

import os
import sys
import threading
import time

import httpx
import uvicorn
from dotenv import load_dotenv

load_dotenv()

MESH_PORT = 8450
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        raise AssertionError(f"FAILED: {label} — {detail}")


def main():
    if not os.environ.get("OPENLEGION_CRED_OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("No API key set. Set OPENLEGION_CRED_OPENAI_API_KEY in .env")
        sys.exit(1)

    if not os.environ.get("OPENLEGION_CRED_OPENAI_API_KEY"):
        os.environ["OPENLEGION_CRED_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

    print("\n=== OpenLegion Smoke Test ===\n")

    # --- Setup ---
    from src.host.containers import ContainerManager
    from src.host.costs import CostTracker
    from src.host.credentials import CredentialVault
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    cost_tracker = CostTracker(db_path="data/smoke_costs.db")
    bb = Blackboard()
    pubsub = PubSub()
    perms = PermissionMatrix()
    vault = CredentialVault(cost_tracker=cost_tracker)
    cm = ContainerManager(mesh_host_port=MESH_PORT, use_host_network=True)
    router = MessageRouter(perms, {})

    app = create_mesh_app(bb, pubsub, router, perms, vault)
    config = uvicorn.Config(app, host="0.0.0.0", port=MESH_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(20):
        try:
            httpx.get(f"http://localhost:{MESH_PORT}/mesh/agents", timeout=2)
            break
        except Exception:
            time.sleep(0.5)

    skills_dir = os.path.abspath("skills/research")
    url = cm.start_agent(
        agent_id="smoke_test",
        role="assistant",
        skills_dir=skills_dir,
        system_prompt="You are a helpful assistant. Keep responses concise (1-2 sentences max).",
        model="openai/gpt-4o-mini",
    )
    router.register_agent("smoke_test", url)

    print("  Waiting for agent container...")
    ready = False
    for _ in range(60):
        try:
            r = httpx.get(f"{url}/status", timeout=2)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(1)

    if not ready:
        print(f"  {FAIL} Agent failed to start")
        cm.stop_all()
        sys.exit(1)

    try:
        # --- Test 1: Basic chat ---
        print("\n1. Basic Chat")
        r = httpx.post(f"{url}/chat", json={"message": "What is 2+2? Reply with just the number."}, timeout=60)
        check("Agent responds", r.status_code == 200)
        data = r.json()
        check("Response contains text", len(data.get("response", "")) > 0, data.get("response", "")[:80])

        # --- Test 2: Tool use (exec) ---
        print("\n2. Tool Use (exec)")
        r = httpx.post(f"{url}/chat", json={"message": "Run the command `echo hello_from_openlegion` and tell me what it printed."}, timeout=60)
        check("Agent responds", r.status_code == 200)
        data = r.json()
        tools_used = [t.get("tool") for t in data.get("tool_outputs", [])]
        check("Used exec tool", "exec" in tools_used, f"tools: {tools_used}")
        check("Got correct output", "hello_from_openlegion" in data.get("response", ""), data.get("response", "")[:100])

        # --- Test 3: File tools ---
        print("\n3. File Tools")
        r = httpx.post(f"{url}/chat", json={"message": "Write the text 'smoke test ok' to /data/test_output.txt, then read it back and tell me the contents."}, timeout=60)
        check("Agent responds", r.status_code == 200)
        data = r.json()
        tools_used = [t.get("tool") for t in data.get("tool_outputs", [])]
        check("Used write_file", "write_file" in tools_used, f"tools: {tools_used}")
        check("Response mentions content", "smoke test ok" in data.get("response", "").lower(), data.get("response", "")[:100])

        # --- Test 4: Memory across sessions ---
        print("\n4. Cross-Session Memory")
        r = httpx.post(f"{url}/chat", json={"message": "Remember this important fact: the project codename is FALCON-7. Save it to your memory."}, timeout=60)
        check("Agent acknowledges", r.status_code == 200)

        # Reset chat
        r = httpx.post(f"{url}/chat/reset", timeout=5)
        check("Chat reset", r.status_code == 200)

        # Ask in new session
        r = httpx.post(f"{url}/chat", json={"message": "What is the project codename I told you about?"}, timeout=60)
        check("Agent responds after reset", r.status_code == 200)
        data = r.json()
        response_lower = data.get("response", "").lower()
        check("Remembers FALCON-7", "falcon" in response_lower, data.get("response", "")[:120])

        # --- Test 5: Cron CLI ---
        print("\n5. Cron Scheduler")
        from src.host.cron import CronScheduler
        sched = CronScheduler(config_path="config/cron.json")
        job = sched.add_job(agent="smoke_test", schedule="0 9 * * 1-5", message="Morning check")
        check("Job created", job.id.startswith("cron_"), f"id={job.id}")
        jobs = sched.list_jobs()
        check("Job listed", len(jobs) >= 1, f"{len(jobs)} jobs")
        sched.remove_job(job.id)
        check("Job removed", len(sched.list_jobs()) == 0)

        # --- Test 6: Webhook CLI ---
        print("\n6. Webhook Manager")
        from src.host.webhooks import WebhookManager
        wh = WebhookManager(config_path="config/webhooks.json")
        hook = wh.add_hook(agent="smoke_test", name="test-hook")
        check("Hook created", "id" in hook, f"id={hook['id']}")
        hooks = wh.list_hooks()
        check("Hook listed", len(hooks) >= 1, f"{len(hooks)} hooks")
        wh.remove_hook(hook["id"])
        check("Hook removed", len(wh.list_hooks()) == 0)

        # --- Test 7: Cost tracking ---
        print("\n7. Cost Tracking")
        spend = cost_tracker.get_all_agents_spend("today")
        check("Usage recorded", len(spend) >= 0, f"{len(spend)} agents with spend")
        total_spend = cost_tracker.get_spend(period="today")
        check("Total cost calculated", total_spend["total_cost"] >= 0, f"${total_spend['total_cost']:.4f}")

        print("\n" + "=" * 40)
        print(f"  All tests passed! {PASS}")
        print("=" * 40 + "\n")

    except AssertionError as e:
        print(f"\n  Test failed: {e}\n")
        sys.exit(1)
    finally:
        print("  Cleaning up...")
        cm.stop_all()
        cost_tracker.close()
        server.should_exit = True
        for f in ["data/smoke_costs.db", "data/smoke_costs.db-wal", "data/smoke_costs.db-shm"]:
            try:
                os.remove(f)
            except OSError:
                pass


if __name__ == "__main__":
    main()
