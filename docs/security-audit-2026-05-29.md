# OpenLegion Engine — System-Wide Security Audit

**Date:** 2026-05-29
**Reviewer:** Principal engineering / security review (Claude Code, adversarial multi-agent audit)
**Scope:** Full engine — agent isolation, operator trust boundary, mesh, containers, credentials, SSRF, browser, channels, cron, wallet/payments, data layer, deserialization, input validation/DoS, agent execution core, CLI/secrets/SSO.
**Method:** Adversarial code review across 15 trust-boundary investigations. Every finding was traced to confirm reachability against the actual code; theoretical-only concerns are labelled as such. The two highest-impact findings (the unauthenticated agent server and the wallet cap bypass) were additionally hand-verified by the lead reviewer.

> Status legend: **CONFIRMED** = exploit path traced end-to-end in code · **CONDITIONAL** = real but gated behind a precondition (config, token leak) · **BY-DESIGN** = accepted trade-off, flagged for visibility.

---

## Threat model

Three trust zones (per `CLAUDE.md`): **User** (full trust), **Mesh** (trusted coordinator), **Agents** (untrusted, sandboxed). The two questions this audit set out to answer:

1. **Can one agent infect another?** (lateral movement, data theft, identity tampering, impersonation between untrusted agent containers)
2. **Can an untrusted agent infect the trusted operator?** (privilege escalation to the delegated-user tier; the operator has a user-toggleable internet-access switch)

The audit then broadened to the full external and internal attack surface.

---

## Executive summary

The **mesh-mediated** trust model is genuinely well-built: identity is HMAC-bound to bearer tokens, `X-Agent-ID` is never trusted for authorization in production, boot fails closed, and **no IDOR/confused-deputy was found across the 109 mesh endpoints**. SSRF protection (DNS-pin + redirect re-validation), container hardening, YAML/deserialization safety, the dashboard (CSRF/XSS/WebSocket auth/SSO), and the SQL layer (fully parameterized) are all **sound**.

The problems cluster into five themes:

| # | Theme | Representative findings |
|---|---|---|
| **T1** | **The agent-to-agent boundary is not enforced** — agents share a flat bridge and run an unauthenticated HTTP server | C1 (critical) |
| **T2** | **Security policy enforced client-side instead of server-side** | H1 operator ceiling · H3 LLM model authz · H6 wallet token valuation |
| **T3** | **Missing resource bounds → DoS** | H4 body size · H5 task-create · H7 lane queues · H8 cron crash · M-series |
| **T4** | **Fail-open / under-gated external & privileged surfaces** | H2 channel commands · H9 WhatsApp webhook · M notification phishing |
| **T5** | **Sanitization & binding gaps** | M compaction-summary laundering · M `x-task-id` binding · M MCP tool-poisoning · M memory-write injection |
| **T6** | **Lifecycle gaps — identity & data outlive the agent** | H11 token not revoked on delete · H12 /data volume remnance · M16 orphan token on failed create |
| **T7** | **Missing defense-in-depth at the HTTP edge** | M19 /openapi.json exposed · M18 clickjacking · H14 VNC proxy traversal · L11 absent security headers |

**Headline:** the answer to question 1 is currently **"yes, an agent can infect another agent"** (C1). The answer to question 2 is **"the identity boundary holds, but the most important permission wall behind it is advisory, not enforced"** (H1).

### Finding count (rounds 1–3)

- **Critical: 1** — C1 *(independently re-confirmed in round 3 from the container, lifecycle, and info-disclosure angles)*
- **High: 16** — H1–H9 (rounds 1–2), **H10–H16** (round 3)
- **Medium: 26** — M1–M11 (rounds 1–2), **M12–M26** (round 3)
- **Low / Info: 19** — L1–L9 (rounds 1–2), **L10–L19** (round 3)

Round 3 ("catch what the first two missed") examined nine surfaces the earlier rounds did not cover in depth — MCP integration, permission-engine matching logic, agent/team lifecycle, concurrency & resource leaks, the reverse-proxy implementations, HTTP security headers / CORS / SSO-bypass, the browser as an SSRF vector, supply chain (marketplace + deps + traces), and information disclosure. Its findings are in the **Round 3 Addendum** below; the rounds-1–2 findings above are unchanged.

---

## CRITICAL

### C1 — Cross-agent infection: flat bridge network + unauthenticated agent server  ·  CONFIRMED

This is the direct answer to "can agents infect one another." The mesh permission matrix is bypassed entirely because the mesh is not the only path between agents.

**The chain:**
1. All agent containers join one shared bridge network with inter-container communication **enabled** — `runtime.py:316-318` creates it with `driver="bridge"` and **no** `enable_icc:false` option. Container names are deterministic (`openlegion_<safe_name>`, `runtime.py:437`) and resolvable via Docker embedded DNS.
2. The agent FastAPI server binds `0.0.0.0:8400` (`agent/__main__.py:272`) and has **no inbound authentication on any route** (`agent/server.py:76-104` — no middleware, no dependency).
3. The http_request SSRF guard *would* block this (it rejects RFC1918), but agents also have `run_command` (`exec_tool.py:43-59`, "the container IS the sandbox") with **no egress filter** (only the browser container has iptables egress rules). `run_command` → `curl`/`python3` reaches a peer's `172.x:8400` with no SSRF check at all.

**What a compromised worker A can do to peer B:**

| Endpoint | Auth | Impact |
|---|---|---|
| `GET /files/{path}` (`server.py:689`) | **none** | Exfiltrate B's entire private memory DB (`/data/{B}.db`), workspace, artifacts |
| `POST /invoke` (`server.py:174`) | **none** | Execute any tool **as B** — runs with B's `mesh_client` (B's `MESH_AUTH_TOKEN`), B's permissions, B's workspace/memory |
| `PUT /workspace/{file}` (`server.py:453`) | `x-mesh-internal` **presence only — forgeable** | Rewrite B's SOUL/INSTRUCTIONS/MEMORY — reprogram B's identity |
| `POST /config` (`server.py:510`) | forgeable | Hot-swap B's model/thinking config |
| `PUT /team`, `POST /artifacts/ingest` | forgeable | Tamper shared state, plant files |

**"Act as B" (the `/invoke` escalation), explicitly:** the tool runs with `mesh_client=loop.mesh_client` (server.py:200-205) — B's token. To the mesh, the caller *is* B: `write_blackboard`/`send_message`/`create_task`/`hand_off`/`vault_resolve`/wallet/browser calls arrive with B's bearer token, get **B's permissions**, and are stamped `written_by=B` / `creator=B`. So A (a) inherits permissions A lacks (a credential B can `vault_resolve` comes back in the `/invoke` HTTP response → A now holds B's secret; B's wallet, B's blackboard namespaces), and (b) leaves audit tracks that point at B, not A. The allowlist at `server.py:192-197` only checks "is this tool in B's repertoire" — never "is A allowed to drive B."

**Why `x-mesh-internal` is no defense here:** the host transport sets `x-mesh-internal: 1` on every host→agent hop (`transport.py:38`), so it is a *static, forgeable* header, not a secret. The agent server checks only its presence (`server.py:453`). Contrast the mesh's own `_is_internal_caller` (`host/server.py:257-280`), which correctly requires the header **and** a loopback peer IP, with a comment explaining the header alone is insufficient. The agent server never got that lesson.

**Per-agent "private memory" isolation therefore does not hold** once any single worker is compromised — and compromise is the design assumption for the untrusted zone.

**Fix (do both):**
- **Network:** create `openlegion_agents` with `options={"com.docker.network.bridge.enable_icc": "false"}` so peers can't address each other directly. The mesh reaches agents via published host ports, which keeps working. Also publish to `127.0.0.1` not `0.0.0.0` (`runtime.py:453`; see H-ref M9).
- **Auth:** add a FastAPI dependency on every agent-server route that constant-time-compares a mesh-issued inbound secret (the mesh already holds `auth_tokens[agent_id]` and can present it), and convert the `x-mesh-internal` presence checks to that real check.

---

## HIGH

### H1 — Operator permission ceiling (`_OPERATOR_PERMISSION_CEILING`) enforced only client-side  ·  CONFIRMED

This is the key wall behind the operator boundary, and it's advisory. The ceiling that blocks granting `can_spawn`/`can_use_wallet` lives only in the operator container's skill code, `operator_tools.py:_validate_edit` (122-146). The mesh endpoint `PUT /mesh/agents/{id}/edit-soft` applies a `field=="permissions"` edit via a raw `perms[...].update(new_value)` (`host/server.py:6381-6383`) with **no ceiling re-check**, gated only on caller-is-operator-or-internal.

**Impact:** an operator LLM steered by a prompt injection (see context below) — or any future/alternate caller holding the operator token, or the dashboard internal path — that emits a raw permissions edit can grant a worker `can_use_wallet`/`can_spawn`. The docstring at `operator_tools.py:521-523` claims the ceiling "still blocks irreversible grants"; server-side it does not.

**Context — there is no semantic anti-injection defense.** `sanitize_for_prompt()` (`shared/utils.py:62`) is Unicode hygiene only (strips zero-width/control chars; defeats invisible-text smuggling) but passes plaintext injection through untouched. Every worker→operator text path *is* wrapped in it at the tool boundary (`loop.py:2729`), so hygiene is uniform — but the system's real safety against a fooled operator rests entirely on the permission walls. That design is sound *only if the walls are server-enforced.* This one isn't.

**Fix:** re-enforce the ceiling inside `_apply_pending_change`/`edit_agent_soft` when `field=="permissions"`, with `credentials.py`/`permissions.py` as the single source of truth that both the skill and the endpoint call. Consider re-adding a user-provenance gate (`_last_message_is_user_origin`, already used by `apply_template`) to permission edits specifically.

### H2 — Privileged channel commands available to any allowed (non-owner) user  ·  CONFIRMED

`Channel._handle_command` (`channels/base.py:263-421`) gates only on the function pointer being wired, never on owner-vs-allowed. The owner/allowed split is enforced **only** for `/allow`, `/revoke`, `/paired`. Every other command — including **`/addkey <service> <key>`**, `/steer`, `/broadcast`, `/reset` — runs for any user who passed `_is_allowed`. `/addkey` writes the credential vault and explicitly handles the SYSTEM tier (`base.py:357-366`).

**Exploit:** owner runs `/allow 12345` to let a teammate *chat*. The teammate sends `/addkey openai sk-attacker-key` → all LLM traffic is billed to / routed through the attacker's key (or DoS'd with junk). Or `/steer` to inject into a busy agent's context; `/broadcast` to fan to every agent.

**Fix:** add an owner check to `/addkey`, `/steer`, `/broadcast`, `/reset` in `_handle_command`; "allowed" users get chat + read-only commands only.

### H3 — LLM proxy has no call-time model authorization  ·  CONFIRMED

`proxy_api_call` / `proxy_api_stream` (`host/server.py:1506-1652`) gate only on `can_use_api("llm")`. The agent fully controls `params["model"]`, and `_handle_llm` (`credentials.py:2664-2761`) picks the provider key from the model string with **no per-agent allowlist and no `is_model_compatible` re-check** (that gate runs only at config-*write* time). `_ALLOWED_LLM_PARAMS` blocks `api_key`/`api_base` injection but not `model` selection.

**Exploit:** an agent provisioned for a cheap model submits `model: "openai/gpt-5-pro"` → the proxy authenticates with `OPENAI_API_KEY` and bills the operator's most expensive key; can also steer prompts through whichever provider has weaker logging. Not key exfiltration (keys never leave the mesh) but real cross-provider abuse + cost drain.

**Fix:** re-run `is_model_compatible` (and ideally a per-agent permitted-model check) inside the proxy before dispatch.

### H4 — No HTTP request-body size limit on the mesh and agent servers  ·  CONFIRMED

Neither the mesh app (`host/server.py:598`) nor the agent app (`agent/server.py:78`) registers a body-size middleware; uvicorn launches with defaults (`cli/runtime.py:833`, `agent/__main__.py:272`). 28+ endpoints buffer the full body via `await request.json()`; per-field caps run *after* parsing. The **browser** server already has the correct pattern (`browser/server.py:91-106`, `_MAX_BODY_BYTES=8 MiB`) — it just wasn't applied to mesh/agent.

**Impact:** any authenticated agent (or SSO'd dashboard user) POSTs a multi-hundred-MB/GB JSON body to e.g. `POST /mesh/tasks`, `POST /mesh/publish`, `PUT /mesh/blackboard/{key}` → OOM/GC-thrash of the single coordination process → fleet-wide DoS. Also mitigates M-payload findings.

**Fix:** add a `content-length`-checking `@app.middleware("http")` to mesh and agent apps mirroring `browser/server.py:91-106` (also guard `request.stream()` since `Content-Length` can be omitted).

### H5 — `POST /mesh/tasks` has no rate limit, no per-agent cap, no cycle guard  ·  CONFIRMED

`create_task` (`host/server.py:4609-4718`) has **no `_check_rate_limit` call**, and `Tasks.create` (`orchestration.py:468-556`) inserts unconditionally: no per-creator/assignee ceiling, no `parent_task_id` cycle detection, no self-assign restriction. The legacy `can_route_tasks` gate was retired; the only guard is `can_message`, which collab-mode templates default to `["*"]`.

**Impact:** a worker loops `POST /mesh/tasks` thousands of times → SQLite WAL bloat, unbounded lane-queue growth (H7), dashboard event-storm. Task-layer fork-bomb at the coordination layer.

**Fix:** add `_check_rate_limit("task_create", caller)`; add a per-assignee pending-task ceiling; reject `parent_task_id` chains beyond a depth.

### H6 — ERC-20 / SPL token transfers bypass per-tx and daily USD spend caps entirely  ·  CONFIRMED (hand-verified)

The spend policy gates only on `value_usd`, derived solely from the transaction's **native** value. The ERC-20 branch sets `"native_value": 0, "value_wei": 0` (`wallet.py:493-494`), and `_estimate_value_usd` returns `0.0` when the native value is `0` (`wallet.py:888-889`). So any fungible-token transfer (USDC/USDT/WETH — first-class agent skills, `wallet_tool.py:170-184`) is valued at **$0.00** and sails past `per_tx` and `daily` in `_check_policy` (`wallet.py:928-941`).

**Exploit:** `wallet_transfer("evm:base", attacker_addr, "50000", token=<USDC contract>)` → `value_usd=0.0` → both cap checks false → signed and broadcast. An agent with `can_use_wallet` and a funded wallet drains the entire stablecoin balance in one call regardless of the `$10/tx` / `$100/day` defaults.

**Fix:** value tokens in USD before the policy check (fetch token price keyed on contract, or maintain a stablecoin=1:1 allowlist; `decimals` is already fetched at `wallet.py:485`). Fail closed when USD value is undeterminable while a cap is configured. Count token notional in the daily sum.

### H7 — Lane queues are unbounded `asyncio.Queue()`  ·  CONFIRMED

`lanes.py:175` — `self._queues[agent] = asyncio.Queue()` with no `maxsize`; `_handle_followup` appends to a parallel `_pending` list with no depth check. Reachable via `/mesh/wake` (`server.py:1119,1205`), throttled only by the reused `blackboard_write` bucket at **10,000/60s**. A fast producer (or H5) outpaces the serial LLM-bound consumer → unbounded heap growth keyed per target agent → OOM, and the target's lane is starved. Steer mode is capped (`_STEER_WAKEUP_MAX=10`); **followup mode is not.**

**Fix:** bound the queue and return `429`/backpressure on overflow; add a dedicated tight `wake`/`task` rate-limit category.

### H8 — Pathological cron schedule crashes the entire scheduler loop  ·  CONFIRMED

`_validate_schedule` (`cron.py:745-760`) validates only the **field count**, never field contents. A schedule like `"*/0 * * * *"` passes. On update, `update_job` (`cron.py:327-331`) does `setattr(job, "schedule", ...)` **before** `_compute_next_run`, so the poison value persists in-memory even though the endpoint 500s. The next `_tick` (`cron.py:407-413`) calls `_is_due → _match_cron_field("*/0", …)` → `int("0")` → **`ZeroDivisionError` propagates out of the `start()` loop and terminates the scheduler task** — all heartbeats and scheduled jobs across the fleet stop until restart.

**Exploit:** an agent with `can_manage_cron` creates a benign job, then `PUT /mesh/cron/{id}` with `{"schedule": "*/0 * * * *"}`. The 500 looks transient; the next tick (≤5s) kills the scheduler fleet-wide.

**Fix:** (1) make `_validate_schedule` parse each field (reject step `0`, malformed ranges) and run it on update; (2) validate the candidate schedule before mutating `job.schedule`; (3) wrap the per-job body of `_tick` in try/except so one poison job can't kill the loop; (4) guard `_match_cron_field` against `step <= 0`.

### H9 — WhatsApp webhook is fully open when `WHATSAPP_APP_SECRET` is unset; enables sender spoofing  ·  CONFIRMED (CONDITIONAL on config)

Signature verification is conditional on `app_secret` being set (`whatsapp.py:225`); when empty, the receiver processes the body anyway (`:219-224`). The only thing forcing the secret to exist is the `start()` guard (`whatsapp.py:103`), which fires **only if `MESH_AUTH_TOKEN` is set** — coupling fail-closed behavior to an unrelated env var. Any deployment running channels without `MESH_AUTH_TOKEN` exposes a fully unauthenticated POST endpoint. CLAUDE.md's "warns when disabled" = **fully open**, not degraded.

**Exploit (when open):** attacker POSTs Cloud-API-shaped JSON to `/channels/whatsapp/webhook` with `from` set to the owner's phone number. `_process_message` (`whatsapp.py:257`) trusts `message["from"]` as the sender identity → **spoofs the paired owner and drives any agent**, bypassing the pairing gate. The mesh `_validated_origin` defense does not apply (the channel calls `dispatch_fn` in-process, never crossing that gate).

**Fix:** make the signature mandatory unconditionally when the WhatsApp channel is enabled (decouple from `MESH_AUTH_TOKEN`); fail-closed (503) in the receiver when no secret.

---

## MEDIUM

### M1 — AST skill validator is bypassable (but the container is the real boundary)  ·  CONFIRMED, framing issue
A subagent proved end-to-end that `string.Formatter().get_field("0.__globals__[__builtins__][__import__]"...)` passes `_validate_skill_code` (`skill_tool.py:44-112`, a deny-list over AST node types — format-string attribute access is invisible to it) and runs `os.popen`. Other bypasses: `().__getattribute__(...)` (`__getattribute__` not in `_FORBIDDEN_ATTRS`), format-string subclass-walk gadgets. **However**, agents already have `run_command` for in-container code execution, so this widens nothing — the Docker container is the boundary. Also: `write_file` + `reload_skills` loads custom skills with **no validation at all** (`skills.py` `_discover`). **Action: documentation honesty** — CLAUDE.md and the docstring present AST validation as a security boundary; it is hardening at best. If it must mean something, rewrite as an allow-list (block `__`-prefixed attrs, allow-list imports, ban `string`/`codecs`/format primitives) and run it on load.

### M2 — Compaction summary re-injected into context without sanitization (injection laundering)  ·  CONFIRMED
`context.py:469-472` re-injects the auto-compaction summary as a `role:user` message **without** `sanitize_for_prompt` — unlike the per-task memory and bootstrap paths, which all sanitize. The summary is LLM output over older messages that include attacker-influenced tool results (`check_inbox` events, fetched web pages). A payload like `IGNORE PRIOR INSTRUCTIONS…` that the summarizer quotes survives compaction stripped of its tool-result framing — now reading as if the *user* said it — and via `_flush_to_memory` (`context.py:350-354`) persists into MEMORY.md for future tasks. **Fix:** wrap the summary (and per-fact lines) in `sanitize_for_prompt()` before re-injection. One-line change matching the existing convention.

### M3 — `/mesh/wake` does not bind `x-task-id` to the target → peer-induced false-fail of a victim's in-flight task  ·  CONFIRMED
`/mesh/wake` reads `x-task-id` unchecked (`server.py:1200`) and threads it into the lane; the recipient auto-closes whatever task_id was supplied, as long as the recipient is its assignee/creator. A worker that `can_message` victim V can wake V with `x-task-id = <a different real task V is running>`. If V is busy, the busy-guard marks that task `failed` (`loop.py:2347-2351`) → silent sabotage of V's legitimate work + a `task_failed` back-edge to its originator. **Fix:** in `/mesh/wake`, when `x-task-id` is present, verify the task exists and `assignee == target` (ideally `creator == caller`) before threading it; drop otherwise.

### M4 — Wallet daily/per-tx check is a TOCTOU race  ·  CONFIRMED
`_check_policy` reads the daily total (`wallet.py:932-936`) and only *after* the on-chain broadcast does `_audit` insert the `broadcast` row (`wallet.py:590-593`) — **no lock** across check→broadcast→record, multiple `await`s in between. Endpoints are `async` on a shared service with a 600/60s limit. N concurrent transfers all read the same `daily_used` before any commits → daily cap overshot to ~N×per_tx. Compounds H6. **Fix:** hold a per-agent `asyncio.Lock` across the full span, or write a reserved ledger row inside the check transaction. The captcha layer's `check_and_charge` (`captcha_cost_counter.py:355-384`) is the correct atomic pattern to port.

### M5 — Browser service has no per-agent identity (confused-deputy if the token leaks)  ·  CONDITIONAL
`browser/server.py:177-182` checks only one process-wide `BROWSER_AUTH_TOKEN`; every endpoint takes `agent_id` from the URL path with no binding to the caller. **Safe today** because agents never receive the browser token (only their own `MESH_AUTH_TOKEN`) and reach the browser only via `POST /mesh/browser/command` where identity is token-derived. **But** if the shared token ever leaks to an agent (future SSRF, log leak, misconfig), that agent can drive/read-cookies-into/wipe **any** peer's browser — blast radius "all agents at once." **Fix:** per-agent browser tokens, or require an `X-Agent-ID` the mesh signs; at minimum document the token as a fleet-wide superuser credential and audit every leak path.

### M6 — KasmVNC binds `0.0.0.0` with no authentication; isolation rests solely on network segmentation  ·  CONDITIONAL
Each agent's Xvnc runs with `-SecurityTypes None -disableBasicAuth -interface 0.0.0.0` on ports 6100..6163 (`service.py:3635-3648`); the browser container's INPUT chain is `ACCEPT`. **Safe today** because those ports are unpublished and the browser container sits on Docker's *default* bridge while agents sit on the isolated `openlegion_agents` bridge (the two don't route to each other). **But** this is a single point of failure: `use_host_network` mode makes the ports host-reachable (and disables egress filtering), and any future topology change placing the browser on `openlegion_agents` yields instant cross-agent screen/input takeover with no test pinning the separation. **Fix:** bind KasmVNC to `127.0.0.1` (the mesh proxies via loopback inside the container anyway); add a regression test asserting the browser container is not on `openlegion_agents` and ports are unpublished.

### M7 — No replay protection on webhook receivers  ·  CONFIRMED
Neither the WhatsApp (`whatsapp.py:214-250`) nor the generic webhook (`host/webhooks.py:158-199`) records processed message IDs or enforces a timestamp window. A captured valid signed body replays indefinitely (WhatsApp even retries on >5s). **Fix:** dedup on `message["id"]` (bounded TTL set) + reject stale timestamps; add a nonce/timestamp to the generic webhook envelope.

### M8 — LLM proxy has no input-size cap; OAuth path skips budget enforcement entirely  ·  CONFIRMED
`APIProxyRequest.params` is unbounded `dict[str, Any]` (`types.py:633`); input prompt size is never capped (only `max_tokens` output default 4096). Worse, `_needs_budget = (is_llm and not _is_oauth)` (`credentials.py:581`) — when an **OAuth subscription token is configured, budget checks and cost tracking are skipped completely**, behind a 6000/min limit. An agent can send arbitrarily large prompts with no dollar gate → token/quota burn. **Fix:** cap serialized `params`/message size before dispatch; apply a coarse per-call input ceiling on the OAuth path.

### M9 — Agent & browser ports published on all host interfaces (`0.0.0.0`)  ·  CONFIRMED
`runtime.py:453` (`{"8400/tcp": port}`) and `:696` (`:8500`) publish to `0.0.0.0` — no `HostIp`. Combined with C1's missing auth, any host with network reach to the VPS's published port range can POST tasks; in production this is mitigated only by external host firewalling (a deployment concern, not enforced in engine code). **Fix:** publish to `127.0.0.1` (the mesh always reaches agents via `127.0.0.1:{port}`). Cheap defense-in-depth that shrinks C1's blast radius.

### M10 — `http_request` buffers full response before truncating; unclamped timeout  ·  CONFIRMED
`http_tool.py:246` sends with `stream=False`, then `:478` slices `response.text[:50_000]` **after** the entire body is read into memory. An attacker-controlled server (the agent picks the URL) returns a multi-GB body → agent-container OOM (384m cap). `timeout` is unclamped → pin a pooled connection (pool max 20). Contained to the calling agent's container, but a self-DoS / pool-wedge. **Fix:** stream and abort at `_MAX_BODY`; clamp `timeout` to e.g. 1–120s.

### M11 — No per-agent cron job cap  ·  CONFIRMED
No limit anywhere on jobs per agent (`cron.py`/`server.py`); `cron_create` bucket is 1000/min. An agent with `can_manage_cron` can mint thousands of persisted jobs (full `cron.json` rewrite each `_save`) scanned every 5s, each tick `create_task`-ing due jobs with no concurrency ceiling. **Fix:** per-agent job cap in `add_job`; lower the create bucket; semaphore the per-tick fan-out.

---

## LOW / INFO

- **L1 — OAuth access-token prefix+suffix logged at INFO/ERROR.** `credentials.py:1814-1818, 1915, 1934, 2042-2043` log `token[:15]...token[-4:]` + exact length, not via redaction. The `[-4:]` + length leak real entropy. **Fix:** drop the preview or log a `sha256[:8]` hash. *(CONFIRMED, multiple sites.)*
- **L2 — Notification phishing via `credential_request`.** `POST /mesh/credential-request` (`server.py:2044`) has **no permission gate** — explicit `TODO(Task 4)` at `:2054` that `can_request_user_credentials` is wired but unenforced. Any agent can mint a legit-looking "needs your Stripe key" bell card (`dashboard/server.py:947-985`) to socially-engineer the operator. Value never transits the event (no silent exfil); rendered via `x-text` (no XSS). **Fix:** enforce the already-wired gate; surface the originating agent id in the card. *(CONFIRMED, LOW.)*
- **L3 — CRED-tier secrets are agent-readable by design.** `/mesh/vault/resolve` (`server.py:1778`) returns plaintext CRED values to the agent zone for `$CRED{}` substitution. SYSTEM_* keys are hard-blocked (verified), but any agent whose `allowed_credentials` glob matches a CRED can resolve it and POST it to an attacker URL. **Action:** document explicitly; long-term consider mesh-side `$CRED{}` substitution. *(BY-DESIGN.)*
- **L4 — Fleet template `permissions` block bypasses the operator ceiling.** `cli/config.py:553-560` sets `can_spawn`/`can_manage_fleet`/etc. from a template with no ceiling clamp. **Not runtime-reachable** — templates are bundled read-only image code with no upload endpoint, and none of the 13 shipped templates escalate. **Fix:** clamp template booleans against `_OPERATOR_PERMISSION_CEILING` to future-proof a custom-template feature. *(CONDITIONAL / hardening.)*
- **L5 — Discord guild allow-list does not cover DMs.** `discord.py:335-337` skips the guild check when `message.guild` is falsy (DMs). Pairing still blocks unpaired DMs, so it's a defense-in-depth gap (false sense of network confinement), not direct compromise. **Fix:** reject DMs or require prior pairing when `allowed_guilds` is set.
- **L6 — Slack logs full inbound message text + sender at WARNING.** `slack.py:184-187`, before auth/pairing, not redacted. Leftover debug. **Fix:** drop to DEBUG, remove payload, route through `deep_redact`.
- **L7 — Per-tenant captcha cost cap is observability-only.** `captcha_cost_counter.py:747-813` only emits threshold alerts; the only enforced gate is per-agent (`service.py:8472`). K agents each just under cap collectively spend K×cap. May be intended (docstring says "alerts"). **Fix:** enforce a hard tenant ceiling if desired, else document as advisory.
- **L8 — `_docker_safe_name` collision can merge two agents' data volumes.** `runtime.py:40-42` — distinct valid IDs (`a-b`, `a_b`) map to the same `openlegion_data_*` volume/container. Requires create privileges, so it's a blast-radius escalation. **Fix:** make the mapping injective (reject colliding safe names, or hex/base32-encode the raw ID).
- **L9 — Back-edge `origin_user` not bound to creator.** `server.py:4567` wakes `origin_user` derived from the stored task origin; for `kind="agent"` origins `_validated_origin` returns `user` unchanged. Today the contextvar prevents agent choice, but any non-`mesh_client` path that sets `X-Origin.user` while keeping `kind=agent` could redirect completion wakes. **Fix:** validate `origin.user` against `record["creator"]` before using it as a wake target. *(LOW–MEDIUM, theoretical.)*

---

## Defenses verified SOUND (do not regress these)

- **Mesh identity binding & fail-closed boot.** `_extract_verified_agent_id` (`server.py:912-952`) derives identity from the bearer token via `hmac.compare_digest`, ignoring `X-Agent-ID` on the token path. `enforce` mode + empty `auth_tokens` → `SystemExit` (`server.py:749-757`). `_caller_is_operator` and the 14-gate carve-out are keyed on verified identity — **not forgeable by a worker.** Synchronous worker→operator wakes are blocked outright (`server.py:1176`).
- **No IDOR across mesh blackboard/task endpoints.** Every handler re-resolves identity before both the permission check and the data op; `written_by`/`creator`/`deleted_by` use the verified id. Task status/inbox/reroute authorize against the stored record.
- **SSRF protection is correct.** DNS-pin resolves once and rewrites the URL to the IP literal (no re-resolution → no rebinding/TOCTOU), validates *all* resolved IPs, re-validates every redirect hop, strips cross-origin `Authorization`, blocks metadata + IPv4-mapped/6to4/Teredo/CGNAT (`http_tool.py:133-293`). *(Minor: also strip non-`Authorization` custom auth headers on cross-origin redirect; add NAT64 unwrap.)*
- **SYSTEM_* keys never resolvable by agents** (4-layer defense, `credentials.py` + `permissions.py:349-350`). LLM param injection blocked by `_ALLOWED_LLM_PARAMS`.
- **Container hardening is real and applied** (`runtime.py:441-447`): `cap_drop=[ALL]`, `no-new-privileges`, `read_only`, `tmpfs /tmp` noexec/nosuid, mem/cpu/pids limits. No docker.sock mount, no host write-mounts, no secrets baked into images, browser privilege-drop correct.
- **SQL layer fully parameterized** — every agent-controlled value bound via `?` across mesh/lanes/summaries/notifications/telemetry/change_history; LIKE wildcards escaped; LIMIT/OFFSET int-coerced. **No SQLi.**
- **Audit log append-only** from agents; archive is operator/internal-only and soft. **Undo/pending_actions** correctly scoped, TTL-enforced server-side, race-free (`BEGIN IMMEDIATE` + consumed flag), replay-proof. **Summary ratings** scope-checked, lock TOCTOU-free.
- **Deserialization is clean.** All YAML `safe_load`; **zero** pickle/marshal/shelve/dill; no `eval`/`exec` on untrusted input outside the (separately-noted) skill validator; no tarfile/zipfile extraction (no Zip Slip). `.env` injection defended (`interpolate=False`, CR/LF rejection, key regex, atomic 0o600 write).
- **Dashboard.** Jinja autoescape on; CSRF via `X-Requested-With` as a router dependency (multipart form can't forge it); all `x-html` sinks run through DOMPurify or escape-first; VNC URL attribute-bound with a regex-validated agent_id; `/ws/events` verifies the session cookie before upgrade and rejects agent tokens. *(CSP allows `unsafe-inline`/`unsafe-eval` — no XSS backstop, so the sanitizers are load-bearing; flagged as defense-in-depth debt.)*
- **Token entropy & comparison.** All tokens `secrets.token_urlsafe(32)` (256-bit); all comparisons `hmac.compare_digest`. **CLI** has no shell-injection surface; prints no secrets. **SSO** correctly delegated to the Caddy auth-gate; engine-side `verify_session_cookie` independently enforces expiry/lifetime-cap/constant-time compare.
- **Memory store isolation, handoff origin/kind non-escalation, auto-close authorization, role-alternation merge, context-growth bounds, MCP config strict model** — all verified sound.

---

## Remediation roadmap (priority order)

**P0 — close the infection vector (answers the original question):**
1. **C1** — disable bridge ICC + authenticate the agent server (+ M9 loopback bind). The only fix that actually stops agent→agent infection.

**P1 — make server-side the real enforcement point (theme T2):**
2. **H1** — server-side `_OPERATOR_PERMISSION_CEILING` re-check on permission edits.
3. **H3** — call-time model authorization in the LLM proxy.
4. **H6** — value token transfers in USD before the wallet cap check (+ **M4** wallet TOCTOU lock).

**P2 — DoS / resource bounds (theme T3):**
5. **H4** — body-size middleware on mesh + agent (reuse the browser pattern).
6. **H5 / H7** — task-create rate limit + bounded lane queues.
7. **H8 / M11** — cron schedule validation + tick try/except + per-agent job cap.
8. **M8 / M10** — LLM proxy input cap (+ OAuth-path guard) + http_tool streaming/timeout clamp.

**P3 — external & privileged surfaces (theme T4):**
9. **H2** — owner-gate `/addkey`/`/steer`/`/broadcast`/`/reset`.
10. **H9 / M7** — mandatory WhatsApp signature + webhook replay protection.
11. **L2** — enforce `can_request_user_credentials`.

**P4 — hardening & hygiene (themes T5 + docs):**
12. **M2** — sanitize the compaction summary. **M3 / L9** — bind `x-task-id` / `origin_user`.
13. **M5 / M6** — browser per-agent identity + KasmVNC loopback bind + regression test.
14. **M1** — reframe AST validator in docs (and/or allow-list + validate-on-load). **L1/L6** — log redaction. **L4/L8** — template ceiling clamp + injective volume naming.

---

## Appendix — key file references

| Area | Files |
|---|---|
| Agent server (C1, M9) | `src/agent/server.py:76,174,453,485,510,689`, `src/agent/__main__.py:272`, `src/agent/builtins/exec_tool.py:43` |
| Network (C1, M9) | `src/host/runtime.py:316-318,437,453,696` |
| Operator ceiling (H1) | `src/agent/builtins/operator_tools.py:122-146,521`, `src/host/server.py:6381-6383,6651` |
| Channels (H2,H9,M7,L5,L6) | `src/channels/base.py:263-421`, `whatsapp.py:103,214-257`, `discord.py:335`, `slack.py:184`, `src/host/webhooks.py:158` |
| LLM proxy (H3,M8) | `src/host/server.py:1506-1652`, `src/host/credentials.py:581,2664-2761` |
| DoS bounds (H4,H5,H7,M10) | `src/host/server.py:598,4609`, `src/agent/server.py:78`, `src/host/lanes.py:175`, `src/host/orchestration.py:468`, `src/agent/builtins/http_tool.py:246,478`, `src/browser/server.py:91-106` |
| Wallet (H6,M4) | `src/host/wallet.py:477-515,565-593,887-954` |
| Cron (H8,M11) | `src/host/cron.py:313-331,407-413,745-760,792-811` |
| Browser isolation (M5,M6) | `src/browser/server.py:177`, `src/browser/service.py:3635-3648`, `docker/browser-entrypoint.sh:124` |
| Agent core (M2,M3,L9) | `src/agent/context.py:350-354,469-472`, `src/agent/loop.py:2347-2351,2729`, `src/host/server.py:1200,4567` |
| Secrets/logging (L1,L3) | `src/host/credentials.py:1778,1814-1818,1915,1934,2042` |

---
---

# Round 3 Addendum (2026-05-29) — coverage of surfaces missed by rounds 1–2

Round 3 was a deliberate "find what we missed" pass over nine previously-unexamined surfaces. Every finding below was traced to confirm reachability in real code; `STATUS` distinguishes **CONFIRMED** (exploit path traced end-to-end), **CONDITIONAL** (real but gated by config/precondition), and **THEORETICAL**. Findings are numbered continuing from rounds 1–2.

> **C1 re-confirmation.** Three independent round-3 investigations (container, lifecycle, info-disclosure) re-derived the unauthenticated agent server. Two specifics worth recording under C1: the agent server's **read** endpoints (`/files`, `/files/{path}`, `/workspace/{filename}`, `/history`, `/chat/history`, `/artifacts`, `/heartbeat-context`) have **zero** auth (`agent/server.py:407-714`), and the **write** endpoints' `x-mesh-internal` check validates *header presence only* with no loopback/source check (`agent/server.py:453,485,510,741`) — so a peer on the bridge forges it. Both are facets of C1; the fix is the same (authenticate the agent server + disable ICC).

## HIGH (round 3)

### H10 — Cross-team blackboard READ/WRITE is unenforced for `["*"]`-ACL agents; the default/starter fleet ships flat  ·  CONFIRMED
`can_read_blackboard`/`can_write_blackboard` (`permissions.py:160-189`) match the raw key against the agent's glob patterns. Team isolation depends **entirely** on those patterns being narrowed to `projects/{team}/*` — which only happens in `_add_project_blackboard_permissions` (`cli/config.py:990-1021`), called only when an agent is explicitly added to a team. But agents are **not** auto-placed in teams: `starter.yaml:91-92` ships `blackboard_read/write: ["*"]`; `_apply_template` (`config.py:1205-1288`) never calls `_add_agent_to_project`; operator-created agents get `["*"]` (`server.py:3305`). For any such agent, `fnmatch(any_key, "*")` → True and the pub/sub project-prefix gate is skipped (`_agent_projects` is empty). The blackboard read/write endpoints only **count** cross-project access ("pure observability — NOT a denial", `server.py:1256-1262`); `OPENLEGION_TEAM_SCOPE_MODE=enforce` filters only `/mesh/agents` and `/mesh/tasks`, never the blackboard. **Exploit:** any starter/template/operator-created agent reads/writes every other team's blackboard via `GET/PUT /mesh/blackboard/projects/{otherteam}/...`. The advertised "team scope isolation" does not cover the blackboard data plane. **Fix:** derive effective scope from `_agent_projects[agent]` at the blackboard gate and require the `projects/{team}/` prefix under enforce mode (mirror the pub/sub gate); stop minting `["*"]` on the create/apply paths. *(This is the enforced-reality behind the round-1 M2 observation — promoted to HIGH now that the default fleet shape is confirmed flat.)*

### H11 — Deleted/archived agent auth tokens are not revoked  ·  CONFIRMED
The mesh verifies identity against `_auth_tokens`, which is the **same dict object** as `runtime.auth_tokens` (aliased, `cli/runtime.py:736` → `server.py:725`). Tokens are popped **only** inside `RuntimeBackend.stop_agent` (`runtime.py:806,1115`), gated on `agent_id in self.agents`. The **delete** path (`server.py:5984-6011` → `cli/config.py:1052-1087`) stops the container via a *separate* raw `docker.from_env()` call — never `stop_agent` — so the token is never removed; `app.cleanup_agent` (`server.py:854-892`) doesn't pop it either, and `permissions.reload()` is not called (the in-memory matrix keeps the deleted agent's ACLs). The **archive** path calls `stop_agent` but only best-effort (try/except) — if the container stop fails, the token + perms survive. Since the bearer token is the agent's own `MESH_AUTH_TOKEN` env (readable by agent code), a compromised agent that exfiltrates its token keeps a **fully working mesh identity after the operator "deletes" it**, until the next mesh restart (tokens are in-memory). **Fix:** `_auth_tokens.pop(agent_id, None)` + `permissions.reload()` in `_cleanup_agent`, independent of container-stop success.

### H12 — Agent deletion orphans the `/data` volume → private-memory remnance across same-name recreate  ·  CONFIRMED
The delete path removes only the container, never the named volume `openlegion_data_{safe_name}` (`runtime.py:386-387`); even `stop_agent` defaults `remove_data=False`. Create→delete→recreate-with-same-name re-mounts the orphaned volume, so a **new agent inherits the deleted agent's private memory DB and workspace** (secrets, MEMORY.md, scraped data). This is the lifecycle-reachable counterpart to the round-1 L8 volume-collision finding. **Fix:** on confirmed delete, `stop_agent(target_id, remove_data=True)` or explicitly remove the volume. Policy: delete should wipe; only archive retains.

### H13 — Browser display/port slot leak on post-launch failure → fleet-wide browser DoS  ·  CONFIRMED (path) / CONDITIONAL (trigger)
In `service.py:_start_browser`, `allocate()` reserves a `(display, vnc_port)` slot (line ~3937) and the X stack + Camoufox launch; but the awaits *after* launch — `context.new_page()` (~4057), `_maybe_restore_session` (~4107), WID discovery (~4122) — are **not** guarded. The caller `get_or_start` (~3562) doesn't wrap `_start_browser` in try/except, so on a post-launch exception `self._instances[agent_id]` is never set, the slot stays in `_display_allocator._allocated`, and the launched Camoufox/Xvnc process group is never torn down. `_cleanup_idle`/`stop_all` iterate only `_instances`, so the orphaned slot is invisible — only a service restart reclaims it. With **64 slots total**, repeated failures (a flaky page or malformed restore-state, both agent-influenceable) monotonically drain the pool → `PoolExhausted` → every agent denied a browser. **Fix:** wrap the tail of `_start_browser` to tear down + release on any exception; add a reconciliation sweep that releases allocated displays with no live instance.

### H14 — Path traversal in the VNC reverse proxy escapes `/agent-vnc/` to arbitrary authenticated GETs on the browser service  ·  CONFIRMED
`/agent-vnc/{agent_id}/{path:path}` builds the upstream URL by raw interpolation: `target = f"{svc_url}/agent-vnc/{agent_id}/{path}"` (`server.py:8567`). `agent_id` is regex-validated; `{path}` is **not**. Starlette URL-decodes `{path:path}` and httpx normalizes dot-segments, so `/agent-vnc/x/..%2f..%2fbrowser%2fvictim%2fsession` → upstream `GET /browser/victim/session`, **carrying the mesh's valid `Bearer {browser_token}`** that `_verify_auth` accepts (verified with starlette TestClient + httpx). This bypasses the browser service's own agent_id guard (it lands on a *different* route) and the entire `/mesh/browser/command` permission model. Any VNC-authorized dashboard user reaches any GET endpoint for any agent_id with mesh-internal trust; today GET-reachable endpoints are mostly low-sensitivity, but it's a standing authorization break and any new/sensitive GET is immediately exposed. The WS handler (`8632`) has the identical raw interpolation. **Fix:** reject `..` in `path` (post-decode) before building `target`, on BOTH the HTTP (8567) and WS (8632) handlers; assert the constructed URL path starts with `/agent-vnc/{agent_id}/`.

### H15 — Marketplace skills are `exec_module`'d with NO load-time AST validation  ·  CONFIRMED (defect) / CONDITIONAL (needs a file in the marketplace dir)
Self-authored skills pass `_validate_skill_code`, but marketplace skills do **not** at load time: `SkillRegistry._discover` → `_load_modules_from` (`skills.py:153-164`) blindly `exec_module`s every `.py` in the marketplace dir at boot and on every `reload_skills`, with no AST gate, manifest check, signature, or commit-SHA pin. The only validation lives in `marketplace.py:install_skill` (line 97), which has **zero production callers** (referenced only by tests). So whatever Python sits in `skills/_marketplace/<name>/*.py` runs in the agent container with full (un-sandboxed-Python) privileges — `os`/`subprocess`/`socket` all importable, defeating the documented skill sandbox. Today the dir is operator/host-populated (mounted read-only, not agent-writable — that isolation HOLDS), so it's CONDITIONAL; but if `install_skill` is ever wired to an API/agent-reachable path without load-time re-validation it's remote-to-RCE. **Fix:** call `_validate_skill_code` inside `_load_modules_from` (fail-closed) for the marketplace/custom dirs; pin installs to a verified commit SHA.

### H16 — `/mesh/traces*` gated by `_require_any_auth` → cross-agent prompt/response disclosure + unredacted secrets at rest  ·  CONFIRMED
`/mesh/traces` and `/mesh/traces/{id}` (`server.py:3693-3707`) call `_require_any_auth`, which accepts **any** agent's bearer token (`server.py:1014-1050`), and `query()`/`get_trace()` return rows for **all** agents with no per-caller filter. Traces store a 500-char `prompt_preview` + `response_preview` per LLM call (`server.py:1523-1567`), **unredacted** (`traces.py:68-89` applies no `deep_redact`/`redact_url`). So Agent A's token reads Agent B's prompts and model responses — including any secret a user pasted into chat or any credential the model echoed. Direct agent exploitation is gated by http_tool's SSRF block (agents can't trivially curl the mesh), making it CONDITIONAL via that transport — but the **authorization tier is simply wrong** for a cross-zone data surface, and a malicious MCP tool / marketplace skill / second integration holding a token reads everything. **Fix:** gate with `_require_operator_or_internal`; redact previews at capture and run `deep_redact`/`redact_url` before `TraceStore.record`.

## MEDIUM (round 3)

- **M12 — MCP tool descriptions/names/schemas reach the LLM unsanitized (tool poisoning).** CONFIRMED. `mcp_client.py:109-115` stores live server-supplied `tool.description`/`name`/`inputSchema` verbatim; `skills.py:170-173,449-457` emits them into the LLM tool payload with **no** `sanitize_for_prompt` (contrast tool *results*, sanitized at `loop.py:2729`). A malicious/rug-pulled MCP server (config approved once, content arrives later and can change) embeds injection text in a description → enters the agent's context as instructions. Also no length cap on descriptions. **Fix:** sanitize + length-cap MCP tool metadata at registration.
- **M13 — `*` crosses `/` in permission fnmatch; `..` not normalized in blackboard keys.** CONFIRMED. `fnmatch("a/b/c","a/*")` is True, so `projects/teamA/*` (and patterns like `leads/*/research`) match at arbitrary depth — wider than authors expect; and raw `{key:path}` with `..` is never normalized before the ACL check (`server.py:1242,1266`). Safe **today** only because the blackboard is a flat exact-key SQLite store; becomes a traversal bypass if any consumer ever resolves keys hierarchically. **Fix:** separator-aware matcher (`*`→`[^/]*`) where scoping matters; reject `..` segments at the gate.
- **M14 — `permissions.json` write is unlocked + non-atomic.** CONFIRMED. `_save_permissions` (`config.py:263-267`) truncate-then-`json.dump`, no temp+rename, no lock; multiple endpoints do full-document read-modify-write (`server.py:6362-6384`, team join/leave, create/delete). Concurrent operator edits silently lose changes; a crash mid-write corrupts the file, and `PermissionMatrix._load` (`permissions.py:87-99`) has no try/except → **corrupt file crashes mesh boot**. **Fix:** atomic write (`os.replace`) + lock; wrap `_load` to fall back to deny-all on parse error.
- **M15 — `MAX_AGENTS`/`MAX_TEAMS` checked-then-created with no lock; `apply_template` has no spawn rate limit.** CONFIRMED. `create_custom_agent` (`server.py:3242`), `apply_fleet_template` (`:2935`), and `mesh_create_team` (`:4207`) all read-count-then-create non-atomically, from inconsistent count sources, with no mutex; `apply_fleet_template` lacks the `_check_rate_limit("spawn",…)` the other paths have. Concurrent creates overshoot the plan/billing limit. Bounded to `can_manage_fleet`/operator callers. **Fix:** serialize creation behind one lock spanning count→insert; one authoritative count source; add the rate limit.
- **M16 — Failed mid-creation orphans a valid auth token.** CONFIRMED. `start_agent` writes `auth_tokens[agent_id]` (`runtime.py:358`) *before* `containers.run` (`:467`); if the run raises, the token stays but `agent_id` was never added to `self.agents`, so neither rollback nor `stop_agent` (gated on `in self.agents`) removes it. Repeated failed creates accumulate valid orphan tokens (compounds H11). **Fix:** insert the token only after a successful run; pop unconditionally on failure.
- **M17 — VNC proxy emits upstream `content-type` with no `nosniff`/CSP → XSS chain.** CONFIRMED mechanics / CONDITIONAL exploit. The proxy copies upstream `content-type` verbatim into a `StreamingResponse` with no other headers (`server.py:8583-8591`); the dashboard CSP is per-document and not applied here. Chained with H14 to reach `GET /uploads/{x}.html` (content-type guessed from filename), an attacker-staged HTML file renders as `text/html` **in the dashboard origin with no CSP** → JS execution in the operator's session. **Fix:** allowlist the proxied content-type (or force `nosniff` + `default-src 'none'; sandbox` CSP); serve the VNC client from a sandboxed origin.
- **M18 — Clickjacking: no `X-Frame-Options` / CSP `frame-ancestors` on the dashboard.** CONFIRMED. CSP (`dashboard/server.py:1077-1087,7385-7395`) sets `frame-src 'self'` (what it may embed) but omits `frame-ancestors` (who may embed it), and no `X-Frame-Options` exists anywhere. A malicious site iframes the dashboard and clickjacks the operator into privileged actions (spawn/edit/wallet-prompt/credential-approval); `X-Requested-With` CSRF does not mitigate framing. **Fix:** add `frame-ancestors 'none'` (+ `X-Frame-Options: DENY`).
- **M19 — FastAPI `/docs`, `/redoc`, `/openapi.json` exposed unauthenticated.** CONFIRMED (CONDITIONAL on reaching :8420 directly). None of the three apps pass `docs_url=None`/`openapi_url=None` (`server.py:598`, `agent/server.py:78`, `browser/server.py:55`); these routes are registered at app construction with no auth dependency (the dashboard auth lives on `api_router`, not the app-level docs routes). Anyone reaching :8420 directly (Caddy bypass / internal pivot / the 0.0.0.0 bind) gets the complete schema of all 109 mesh + 141 dashboard endpoints — a full attack-surface map (wallet transfer, vault store/resolve, spawn, credential endpoints, parameter shapes) with zero auth. **Fix:** construct the apps with `docs_url=None, redoc_url=None, openapi_url=None` in production (at minimum on the user-facing port).
- **M20 — Browser DNS-rebinding TOCTOU: mesh validates the URL, then the browser re-resolves it unpinned.** CONFIRMED path. `/mesh/browser/command` calls `_resolve_and_pin(nav_url)` purely for its `ValueError` side-effect and **discards the pinned URL** (`server.py:7665-7679`), forwarding the original hostname; Camoufox then re-resolves at `page.goto()` with no pinning and no per-redirect revalidation (`service.py:5010,10743`). An attacker DNS that returns a public IP to the mesh and `169.254.169.254`/`10.x` to Firefox milliseconds later bypasses the check — rebinding defense rests entirely on the iptables backstop (which the mesh comment acknowledges). **Fix:** forward the *pinned* URL (with original Host header), as http_tool does.
- **M21 — Browser SSRF defense collapses to a single layer; the service has no host validation; egress-disable/host-network modes open it fully.** CONFIRMED. Service-side `navigate`/`open_tab` validate **scheme only**, never host (`service.py:4930-4941,10693-10698`) — `http://10.0.0.5/` and `http://169.254.169.254/` pass to `page.goto()`. SSRF IP-checking exists only at the mesh (bypassable per M20) and in iptables; `BROWSER_EGRESS_DISABLE=1` (forced by host-network mode, `runtime.py:570,681-694`) removes iptables, and the IPv6 backstop is conditional on `ip6tables` being present. In those modes the browser is a full SSRF primitive to the host's private nets + metadata. **Fix:** mirror `_is_blocked_ip` into the service-side `navigate`/`open_tab` so SSRF defense is not single-layer.
- **M22 — `traces.db` retention GC disabled by default → unbounded sensitive data at rest.** CONFIRMED. `TraceStore()` is instantiated with no `max_age_hours` (`cli/runtime.py:269`), so `_maybe_gc_old` early-returns and the `DELETE WHERE timestamp < cutoff` never runs (`traces.py:27,91-102`); no row cap either. Combined with H16's unredacted previews, sensitive plaintext accumulates indefinitely. **Fix:** set a default `max_age_hours` (e.g. 168) and/or a row cap.
- **M23 — Structured memory is unsanitized-on-write and auto-reinjected every task (injection persistence).** CONFIRMED (reinforces M2). `memory_save` stores `content` verbatim at `confidence=0.9` (`memory_tool.py:144-169`); `loop.py:1381-1392` auto-injects `get_high_salience_facts`/`search_hierarchical` into **every** task prompt. A single poisoned tool result that drives one `memory_save` becomes a durable, self-reactivating implant (read-back is `sanitize_for_prompt`'d — Unicode hygiene only, does not stop plaintext injection). Per-agent (does not cross agents). **Fix:** sanitize on write; reduce salience / exclude non-`agent`-sourced facts from passive auto-injection.
- **M24 — `can_message` permission bypass on `/mesh/agents/{id}/profile` via omitted `requesting_agent`.** CONFIRMED. The `can_message` gate runs only inside `if requesting_agent:` (`server.py:3440-3449`); omitting the param falls through to `_require_any_auth`, with no team-scope filter. Any authenticated agent reads any target's subscriptions, blackboard watch/write keys, role/capabilities, and full INTERFACE.md cross-team. **Fix:** resolve the caller via `_extract_verified_agent_id` unconditionally and apply the check regardless of the optional hint.
- **M25 — `Pillow>=10.0` floor + no lockfile, on an attacker-influenced image path.** CONDITIONAL/posture. The floor admits Pillow 10.0–10.2 (CVE-2023-50447, CVE-2024-28219); `service.py:2271-2315` runs `Image.open/resize/save` on browser-screenshot bytes from arbitrary pages. Decompression-bomb is mitigated and `ImageMath.eval` isn't used, so it's posture not a live exploit, but `>=` + no lockfile means a vulnerable Pillow can ship silently. **Fix:** `Pillow>=10.3.0`; adopt a lockfile (most deps are open-floored; `litellm` is correctly pinned).
- **M26 — Fire-and-forget WebSocket broadcasts race on shared sockets.** CONDITIONAL. `events.py:110` schedules untracked `_broadcast` tasks; `_clients` is mutated without a lock (unlike `_listeners`) and two broadcasts can `send_text` on the same WS concurrently (`:180-184`), risking frame errors that evict live clients + GC-vulnerable untracked tasks. **Fix:** per-client send serialization; snapshot `_clients`; hold task references.

## LOW / INFO (round 3)

- **L10 — `is_system_credential` name-shape heuristic too narrow.** CONDITIONAL. Blocks only `<provider>_api_key`/`_api_base` (`credentials.py:272-284`); an OAuth token or custom-named system secret landing in the agent tier would match a `["*"]` `allowed_credentials`. Base the agent-tier block on the loaded vault tier, not the name shape.
- **L11 — Security headers absent on all API/static responses.** CONFIRMED. No `X-Content-Type-Options: nosniff`, `Referrer-Policy`, `HSTS`, or `Permissions-Policy` anywhere (zero matches in `src/`); CSP is on the two HTML responses only. Add one response middleware on the mesh app stamping these (also folds in M18 globally).
- **L12 — KasmVNC framebuffer served `X-Frame-Options=ALLOWALL` + `Access-Control-Allow-Origin: *`.** CONFIRMED (proxy-gated). `service.py:3645-3647`. Broader than needed since the client is always same-origin via the proxy; drop `ACAO:*`, scope framing to the deployment origin.
- **L13 — Deleted agent's browser X-stack/display not proactively released.** CONFIRMED (self-healing). `_cleanup_agent` doesn't tear down the browser session; the idle reaper reclaims it within ~30 min (`service.py:3499-3510`). Best-effort release on delete frees the slot immediately.
- **L14 — MCP resolved secrets live in the agent container env (`MCP_SERVERS`).** CONFIRMED (by-design). `runtime.py:374` writes CRED-resolved plaintext into the agent env; the agent can read its own `/proc/self/environ`. Only the agent's own allowed creds (no escalation), but it breaks the "agents never hold keys" guarantee for MCP-referenced creds (unlike http_tool's never-plaintext handles). Document the asymmetry in `docs/mcp.md`.
- **L15 — Raw upstream LLM/provider exception text returned to agents.** CONFIRMED. The generic `except → APIProxyResponse(error=str(e))` (`credentials.py:723-728`) and third-party handlers (`:3152,3174,3196,3299`) forward litellm/provider error strings (may embed base URLs/endpoints/topology; no credential values) to the agent. Map generic upstream errors to a fixed message; keep detail in logs only.
- **L16 — Found-vs-forbidden status-code oracle on task/summary reads.** CONFIRMED. `get_task`/`list_task_events`/`get_work_summary` return 404 for missing but 403 for exists-unauthorized (`server.py:4856-4892,5298-5313`), letting an agent distinguish valid cross-team ids (bounded — ids are random UUIDs). `team_status`/`list_team_tasks` already avoid this (check membership first). Return a uniform code.
- **L17 — MCP server startup has no timeout → agent boot-hang DoS.** CONFIRMED. `mcp_client.py:86-95` awaits `stdio_client`/`initialize`/`list_tools` with no `asyncio.wait_for` (unlike `call_tool`'s 60s); a server that connects but never responds hangs agent boot indefinitely, and servers start sequentially. Wrap each in `wait_for` and mark failed-and-continue on timeout.
- **L18 — Minor concurrency edges.** Cron `_tick` `locked()`-then-acquire is check-then-act → a duplicate run rather than a skip (`cron.py:417,435`); `_check_rate_limit` lock can be popped from under an in-flight check (`server.py:854-865`) — both saved today by single-loop no-await atomicity. Harden the cron gate with a per-job running flag; don't pop held locks.
- **L19 — Permission-engine minor notes.** `global/` is not a protected namespace except two hardcoded sub-prefixes (a `["*"]` agent reads/writes arbitrary `global/...`); `can_message`/`can_publish`/`can_subscribe` use exact membership not glob (fails *closed* — a trap, not a bypass); `wallet_allowed_contracts == []` → allow-all for the contract dimension (gated by `can_use_wallet`). Document/tighten as appropriate.

## Round-3 defenses verified SOUND (do not regress)

- **MCP write path is dashboard-only (User zone).** `mcp_servers` is absent from `_EDITABLE_FIELDS` (`types.py:56,76`), so `edit_agent`/edit-soft cannot set it — the prompt-injection → arbitrary-subprocess path is closed. `$CRED{}` in `command` is rejected; CRED resolution is permission-gated and SYSTEM-tier-blind; MCP is stdio-only today (no SSRF transport). MCP tool *results* are sanitized.
- **Permission default-deny holds for the unknown-agent and missing-file cases**; capability *tightening* of mesh-checked permissions is immediate (every gate calls `get_permissions` live; create/apply/edit call `reload()`); tokens are freshly minted per `start_agent` (no cross-restart reuse); agent IDs are regex-constrained so prefix special-cases (`teamA` vs `teamA-evil`) don't collide; blackboard CAS/claim is atomic (`mesh.py:196` `UPDATE … WHERE version=? RETURNING`, single winner → 409 loser).
- **Lifecycle:** reserved-id and duplicate-id blocked at create; **archive** correctly revokes token + reaps cron (delete is the regression — H11/H12); lane/queue drained on delete; ephemeral spawns TTL-expire.
- **Reverse proxies:** host/port prefix is fixed (no proxy-SSRF to a peer's KasmVNC port or arbitrary host); inbound headers are NOT forwarded (the dangerous "client sets `x-mesh-internal`, it crosses to an agent server" path does **not** exist — outbound headers are built fresh); agent tokens rejected on VNC routes; no redirect-following; `/mesh/browser/command` substitutes only the verified `req_agent_id` + allowlisted action.
- **Browser is not the wide-open SSRF/file hole feared:** scheme allow-list (http/https only) blocks `file://`/`chrome://`/`data:`/`about:` (no local-file read, no peer-profile read via navigation); the `evaluate` arbitrary-JS endpoint is removed; `download` takes a click-ref not a URL; the browser container is network-isolated from agents (default bridge, not `openlegion_agents`); iptables OUTPUT governs Camoufox (UID 1000) and blocks RFC1918 + metadata; `/uploads` has traversal protection and is read-only.
- **No CORS misconfiguration** (no `CORSMiddleware`, no Origin reflection on any app); dashboard auth is router-level across all verbs; dashboard-session and mesh-token auth are distinct (no cross-token escalation); WS routes authenticate before `accept()`; no `debug=True` / no traceback leakage; `/mesh/system/metrics` operator-gated; `/mesh/agents` + `/mesh/teams` scope-filtered/operator-only; `/ws/events` operator-cookie-gated (LLM-preview payloads stay within the single-operator trust boundary — re-audit if multi-viewer dashboards land).
- **Skill load-path is per-agent isolated** (`/data/custom_skills` on a per-agent volume; marketplace dir read-only) — H15 is a *validation* gap, not an isolation gap. Trace SQL is parameterized; marketplace `install_skill` hardening (git flag/symlink/hook lockdown) is sound *when wired*.
- **Concurrency:** `_skill_staging` lock, `spawn_subagent`/display-allocator/container-lifecycle, and the lazy-lock pattern are all correct **under the production single-event-loop assumption** — flagged latent risk: moving any blackboard call into an executor would create a real data race (lockless reads on a non-autocommit shared connection).

## Updated remediation roadmap (rounds 1–3)

**P0 — the infection vector (unchanged):** C1 (disable ICC + authenticate the agent server, incl. the read endpoints and the loopback check on `x-mesh-internal`; loopback-bind ports per M9).

**P1 — server-side enforcement & cross-zone authorization:** H1 (operator ceiling) · H3 (LLM model authz) · H6+M4 (wallet) · **H10** (blackboard team-scope enforcement) · **H16** (trace endpoint auth tier) · **M24** (profile endpoint auth).

**P2 — identity/data lifecycle (new theme T6):** **H11** (token revocation on delete) · **H12** (volume wipe on delete) · **M16** (orphan token on failed create).

**P3 — DoS / resource bounds:** H4/H5/H7 (body/task/lane) · H8/M11 (cron) · **H13** (browser slot leak) · M8/M10 (LLM/http) · **M14** (perms.json atomicity) · **M15** (create-limit race).

**P4 — supply chain & sandbox integrity:** **H15** (validate marketplace skills on load) · M1 (reframe AST validator) · **M12** (sanitize MCP tool metadata) · **M23** (memory write injection) · **M25** (Pillow + lockfile).

**P5 — HTTP edge defense-in-depth (new theme T7):** **H14**+**M17** (VNC proxy traversal + content-type) · **M19** (disable /openapi.json) · **M18**+**L11** (clickjacking + security-header middleware) · **M20/M21** (browser SSRF layering).

**P6 — external surfaces & hygiene:** H2/H9/M7 (channels) · L2 (credential-request gate) · M2/M3/L9 (sanitization/binding) · M5/M6 (browser identity/VNC bind) · M13/L10/L19 (permission matcher) · M22/L15/L16/L17 (retention, error disclosure, MCP timeout) · L1/L6/L12/L13/L14 (logging/headers/cleanup).

## Round-3 file reference additions

| Area | Files |
|---|---|
| Blackboard team scope (H10, M13) | `src/host/permissions.py:160-189`, `src/cli/config.py:990-1021,1205-1288`, `src/host/server.py:1256-1262`, `src/templates/starter.yaml:91-92` |
| Token/volume lifecycle (H11, H12, M16) | `src/host/server.py:725,854-892,5727-6011`, `src/cli/config.py:1052-1087`, `src/host/runtime.py:357-358,386-387,806,1115` |
| Browser slot leak (H13) | `src/browser/service.py:3499-3510,3937,4007-4140`, `src/browser/display_allocator.py` |
| VNC proxy traversal (H14, M17) | `src/host/server.py:8552-8637` |
| Marketplace/skills (H15) | `src/agent/skills.py:134,153-187`, `src/marketplace.py:82-177`, `src/host/runtime.py:418-423` |
| Traces (H16, M22) | `src/host/server.py:1523-1567,3693-3707`, `src/host/traces.py:27,68-102`, `src/cli/runtime.py:269` |
| MCP (M12, L14, L17) | `src/agent/mcp_client.py:86-115,197`, `src/agent/skills.py:166-173,449-457`, `src/host/runtime.py:373-374`, `src/shared/types.py:56,76` |
| Permissions engine (M14, L10, L19) | `src/host/permissions.py:87-99,160-189,272-284,349`, `src/cli/config.py:263-267` |
| Create limits (M15) | `src/host/server.py:2935-2946,3242-3249,4207-4220` |
| HTTP edge (M18, M19, L11, L12) | `src/dashboard/server.py:1077-1087,7385-7395`, `src/host/server.py:598`, `src/agent/server.py:78`, `src/browser/server.py:55`, `src/browser/service.py:3645-3647` |
| Browser SSRF (M20, M21) | `src/host/server.py:7665-7679,7735-7739`, `src/browser/service.py:4930-4941,5010,10693-10743`, `docker/browser-entrypoint.sh:62-199` |
| Memory injection (M23) | `src/agent/builtins/memory_tool.py:144-169`, `src/agent/memory.py:230-260`, `src/agent/loop.py:1381-1392` |
| Info disclosure (M24, L15, L16) | `src/host/server.py:3432-3522,4856-4892,5298-5313`, `src/host/credentials.py:723-728` |
| Concurrency (M26, L18) | `src/dashboard/events.py:110,180-184`, `src/host/cron.py:417,435`, `src/host/server.py:854-865` |
| Dependencies (M25) | `pyproject.toml:26` |
