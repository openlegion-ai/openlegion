# Security Remediation Review — CPO + Principal-Engineer pass

**Date:** 2026-05-29
**Companion to:** `docs/security-audit-2026-05-29.md` (the findings record — 1 Critical, 16 High, 26 Medium, 19 Low across three audit rounds).
**Purpose:** This document does NOT re-list findings. It judges them. Each was re-verified against code with the question "is fixing this the right thing to build *now*, for *this* product?" — then pressure-tested as an engineer: is the obvious fix the right one, what does it touch, what breaks, what's the simplest version that still solves the real problem.

---

## 1. The frame that decides everything

**OpenLegion is single-tenant, single-operator.** One customer per engine instance, behind their own SSO subdomain. The operator is a **full-trust delegate** (like the user). So:

- **The threat that matters is a *compromised/prompt-injected agent*** harming *other agents, the operator, or the wallet* — which is exactly the "can agents infect each other" question that started this. And a **fooled operator LLM** acting on injected instructions.
- **Cross-*tenant* is not a thing.** "Cross-team" is within one customer's own fleet — it matters as blast-radius-containment if one agent is compromised, not as a tenancy boundary.
- **The product's whole pitch is simplicity:** *"tell the operator what you need, it builds your workforce."* A fix that adds operator friction, or breaks the agent-coordination that makes the product work, can be worse than the bug.

**The good news from re-verification:** the vast majority of the right fixes are **invisible to the operator** — server-side enforcement, network flags, resource caps, cleanup-on-delete. Those *fit* the simplicity pitch (they make the product trustworthy without asking the operator to do anything). Only a handful create real tension, and those are called out explicitly below.

---

## 2. Six corrections to the audit (where it was wrong or oversold)

The pressure-test changed the call on several findings. These corrections drive the rankings:

1. **C1's fix is one line, not a project.** The mesh reaches agents over **loopback-published ports** (`runtime.py:468` → `127.0.0.1:{port}`), *not* container-name DNS. So disabling inter-container communication (`enable_icc:false`) breaks **nothing** legitimate and kills the entire peer-to-peer attack path. Agent-server auth is real defense-in-depth (for host-network mode), but ICC-off alone solves the stated problem.

2. **The operator permission ceiling (H1) is *already* meant to be operator-overridable.** There are two surfaces: the LLM-tool/mesh `edit-soft` path (should be walled) and the dashboard `PUT /api/agents/{id}/permissions` path (`dashboard/server.py:3142`, deliberately **no** ceiling — it's the human's intentional "advanced permissions" escalation). The bug is only that the LLM path doesn't re-enforce server-side. **Do not** add a config toggle for the ceiling (that's just a new injection target) and **do not** patch the dashboard path.

3. **The audit's H3 fix does not fix H3.** Re-running `is_model_compatible` in the proxy catches OAuth-allowlist evasion but **not** the cost-drain it describes, because `is_model_compatible` permits the entire API-key catalog. The real fix needs a per-agent model pin — which is safe because models are config-fixed today (no agent switches model at runtime), but it's a bigger change than the audit implied.

4. **`sanitize_for_prompt()` is Unicode hygiene, not an injection defense.** It strips invisible/control characters; plaintext `"IGNORE PRIOR INSTRUCTIONS…"` passes through byte-for-byte. So the "wrap X in sanitize_for_prompt" fixes for **M2, M23, and the M12 description** are **largely theater** — they close invisible-Unicode smuggling only. Worse, M23's read-back is *already* sanitized, so sanitizing on write is pure redundancy. Don't ship these as "injection fixed." (The genuinely useful bits in that cluster are M12's *length cap*, M3, L17, L1, L2 — see below.)

5. **H15/M1 are documentation-honesty problems, not blast-radius problems.** Agents already have `run_command` ("the container IS the sandbox"). Marketplace/skill code reaching `exec_module` is the *same* privilege as `run_command` — no escalation. The marketplace dir is read-only/operator-populated. Enforcing the AST denylist on load would *break legitimate skills* (it forbids `os`/`subprocess`/`io`) and buy nothing. Fix the **docs** (CLAUDE.md overstates the validator as a security boundary) and add SHA-pinning *if* a remote-install path is ever wired.

6. **Two "High" findings are overstated for single-tenant.** **H14** (VNC proxy traversal) requires the *operator's own authenticated SSO session* — not reachable by an agent or external party — and lands on low-sensitivity GETs; it's worth a one-line fix but is not High here. **M19** (`/openapi.json` exposed) requires a Caddy-bypass precondition in any real deployment. Both are cheap to fix; neither is urgent.

---

## 3. Master ranking — criticality × UX-cost-to-patch

Severity below is **re-rated for the single-tenant/single-operator threat model** (not the raw audit severity). "Patch UX impact" is what the operator/agents experience if fixed: **Invisible** (no one notices) · **Setup** (one-time config, no ongoing friction) · **Friction** (ongoing operator cost) · **Breaks-collab** (regresses how the product works).

| ID | Finding (short) | Real severity | Fix effort | Patch UX impact | Verdict |
|---|---|---|---|---|---|
| **C1** | Agents reach peers (unauth server + ICC) | **Critical** | XS (ICC) + M (auth) | Invisible | **DO NOW** |
| **H11** | Token not revoked on delete | High | S | Invisible | **DO NOW** |
| **H12** | /data volume remnance on recreate | High | S | Invisible | **DO NOW** |
| **H6** | Token transfers bypass wallet caps | High | S | Invisible | **DO NOW** |
| **M4** | Wallet cap TOCTOU race | High* | S | Invisible | **DO NOW** (with H6) |
| **H1** | Operator ceiling client-side only | High | S | Invisible | **DO NOW** |
| **H16** | /mesh/traces readable cross-agent | High | XS | Invisible | **DO NOW** |
| **H8** | Cron schedule crashes scheduler | High | S | Invisible | **DO NOW** |
| **H13** | Browser slot leak → fleet DoS | High | S | Invisible | **DO NOW** |
| **H4** | No body-size limit (mesh/agent) | High | S | Invisible (8 MiB) | **DO NOW** |
| **M3** | wake `x-task-id` not bound → task sabotage | Med-High | S | Invisible | **DO** |
| **H5** | task-create unbounded (rate/cap/cycle) | High | S | Invisible (generous defaults) | **DO** |
| **H7** | Lane queue unbounded | High | S | Invisible | **DO** (with H5) |
| **H3** | LLM proxy no model authz | High | M | Invisible (models config-fixed) | **DO** |
| **M24** | profile endpoint can_message bypass | Medium | XS | Invisible | **DO** |
| **H2** | Channel admin cmds for allowed users | Medium | S | Restores intended behavior | **DO** |
| **H9** | WhatsApp webhook fail-open | Medium | XS | None (Meta requires secret anyway) | **DO** |
| **L17** | MCP startup no timeout → boot hang | Medium | XS | Invisible | **DO** |
| **L1** | OAuth token slices logged | Low | XS | Invisible | **DO** |
| **M14** | permissions.json non-atomic write | Medium | S | Invisible | **DO** |
| **M11** | No per-agent cron cap | Medium | XS | Invisible (50) | **DO** (with H8) |
| **M22** | Trace retention GC off by default | Medium | XS | Invisible (168h) | **DO** |
| **M10** | http_request buffers body / no timeout clamp | Medium | XS | Invisible | **DO** |
| **L2** | credential-request ungated (phishing) | Low | S | Invisible (+ template field) | **DO** |
| **H10** | Cross-team blackboard unenforced | Medium | M | **Breaks-collab if naive** | **CAREFUL — do (b)+(c), not (a)** |
| **M18** | Clickjacking (no frame-ancestors) | Medium | XS | Invisible (use `'self'`) | **DO** |
| **M19** | /docs, /openapi.json exposed | Low (overstated) | XS | Dev only (env-gate) | **DO** |
| **H14** | VNC proxy path traversal | Low (overstated) | XS | Invisible | **DO** |
| **M17** | VNC proxy content-type no nosniff | Medium | XS–S | Invisible (CSP needs test) | **DO** (with H14) |
| **M6** | KasmVNC binds 0.0.0.0 | Low (conditional) | XS | Invisible (loopback bind safe) | **DO** + regression test |
| **M8** | LLM input cap / OAuth budget skip | Medium | S | Invisible | **DO** (H4 covers half) |
| **M15** | create-limit race (MAX_AGENTS) | Medium | S | Invisible | **DO** |
| **M26** | WS broadcast socket race | Low-Med | S | Invisible | **DO** |
| **M7** | Webhook replay | Low | S | Invisible (idempotency win) | **DO** (with H9) |
| **M13** | fnmatch `*` crosses `/` | Medium | S | Invisible | **DO with H10(b)** |
| **M12** | MCP tool-desc poisoning | Medium | S (length cap) | Invisible | **DO length-cap; DOC residual** |
| **M21** | Browser SSRF single-layer (service has no host check) | Low (iptables covers default) | S | Invisible | **DO** (service-side `_is_blocked_ip`) |
| **L9** | back-edge origin_user not bound | Low | XS | Invisible | **DO** (cheap) |
| **L10** | system-cred name heuristic narrow | Low | S | Invisible | **DO** (load-tier classify) |
| **L13** | deleted agent's display not freed | Low (self-heals) | XS | Invisible | **DO** (folds into delete fix) |
| **L12** | KasmVNC ALLOWALL/ACAO:* | Low | XS | Invisible (test viewer) | **DO** |
| **L11** | Security headers absent | Low-Med | S | Invisible | **DO** (one middleware) |
| **L15** | Raw upstream error text to agents | Low | XS | Invisible | **DO** |
| **L16** | 404-vs-403 ID oracle | Low | XS | Invisible | **DO** (cheap) |
| **M5** | Browser service no per-agent identity | Low (token never leaves mesh) | L (structural) | Invisible | **DEFER + DOC** |
| **M20** | Browser DNS-rebinding TOCTOU | Low (iptables covers) | M (SNI risk) | Risk of breakage | **DEFER** (M21 covers most) |
| **M16** | Failed-create orphans token | Medium | XS | Invisible | **DO** (folds into lifecycle fix) |
| **M1** | AST validator bypassable | — (no real impact) | Doc | None | **DOC-ONLY** |
| **H15** | Marketplace skills no load validation | — (blast-radius-neutral) | Doc + SHA-pin | None | **DOC + pin-if-remote** |
| **M2** | Compaction summary not sanitized | Low | XS | Invisible | **DO 1-line (sell as hygiene, not anti-injection)** |
| **M23** | Memory write injection persistence | Low | S | Could hurt recall | **Lower confidence on tool-derived facts; else ACCEPT** |
| **M25** | Pillow floor + no lockfile | Low (posture) | XS | Invisible | **DO** (bump floor) |
| **L3** | CRED-tier agent-readable | — (by design) | Doc | — | **DOC** |
| **L14** | MCP secrets in agent env | — (by design) | Doc | — | **DOC** |
| **L19** | global/ namespace, contract `[]`=all | Low | S | Invisible | **DOC / tighten opportunistically** |

\* M4 is "High" only in combination with H6 (the two compound — token valuation is still racily bypassable without the lock).

**The shape of the table is the headline:** ~40 of these are **Invisible-to-patch** — they make the product safer with zero operator-facing cost, which is the ideal alignment with the simplicity pitch. Exactly **one** finding (H10) genuinely threatens how the product works, and **two** (M5, M20) are best deferred. The "highly-secure vs great-UX" tension the brief worried about is, for this codebase, mostly absent — *if* the fixes are scoped correctly.

---

## 4. The consequential calls (prose)

### C1 — the one that answers the original question. **Do now, ship the one-liner first.**
- **CPO:** This is the product promise ("agents are isolated"). It's invisible to fix and on-thesis. No toggle — there is no legitimate consumer of agent-to-agent reachability to preserve.
- **PE:** Right solution is **both** `enable_icc:false` (XS, zero-impact, kills the DNS-addressable peer route) **and** agent-server auth (defense-in-depth for `use_host_network` mode where there's no bridge to harden). They're not redundant — they close different doors. **Simplest version that solves the real problem = ICC-off alone**; a compromised agent can no longer reach a peer's `:8400`. Ship that immediately (one line at `runtime.py:316-318`), then add the bearer via the single injection point `transport._resolve_headers` (covers mesh/cron/health automatically — they all route through it) as a fast-follow. **Touches:** nothing legitimate (mesh→agent is loopback). **Risk:** the one load-bearing assumption is "nothing addresses a peer container by name" — verified true; ICC-off would only break an undiscovered peer-to-peer path, of which there are none. **Sequencing:** ICC-off ships with zero dependencies; agent-auth after.

### Lifecycle cluster (H11 + H12 + M16 + L13) — one root cause, one fix. **Do now.**
- **PE:** These four share a root cause: **the delete path diverged from `stop_agent`** and uses a raw `docker.from_env()` call (`cli/config.py:1059-1068`) that misses the token-pop and volume-removal `stop_agent(remove_data=True)` already implements. The right solution is **re-converge delete onto `stop_agent(remove_data=True)`** + add `_auth_tokens.pop()` and `permissions.reload()` to `_cleanup_agent` + pop the token on `start_agent` failure. That single consolidation fixes all four. **Critical constraint:** archive *deliberately* retains the volume (archive→unarchive), and archive already uses `stop_agent(remove_data=False)` — **do not touch it.** The delete-vs-archive verb *is* the retain/wipe toggle; that's the right abstraction, don't add a flag.
- **CPO:** "Delete should actually revoke and wipe" is table stakes for a product that handles credentials and private memory — and it's invisible. The privacy angle (a recreated same-name agent inheriting the deleted one's memory) is a genuine trust bug. Do it.

### H1 — operator ceiling. **Do now, but only the LLM-facing path.**
- **CPO:** The two-surface design is *correct and worth preserving*: the operator *LLM* is walled (it's the injection target), the *human* operator can grant wallet/spawn via the dashboard (intentional power-user escalation). The threat is a fooled operator LLM, not the human. Don't flatten this into a config toggle — that just hands an injected agent a "turn off the wall" lever.
- **PE:** Extract the ceiling check from `operator_tools.py:_validate_edit` into `permissions.py` as one function; call it from **both** the operator tool and `edit_agent_soft` when `field=="permissions"`. **Do not** call it from `dashboard/server.py:3142` (the human path). The audit's "add a user-provenance gate" is over-engineering — the undo receipt already covers the human; only the LLM path needs the server-side wall.

### H3 — LLM model authorization. **Do, but build the right fix, not the audit's.**
- **PE:** The audit's "re-run `is_model_compatible`" does **not** stop the cost-drain (that check passes the whole API-key catalog). The real fix is a **per-agent model pin**: store the agent's configured model (already in `LLM_MODEL`/agent config) and reject proxy requests whose `model != configured`. This is **safe because models are config-fixed today** — no agent switches model at runtime (grep confirms zero `model=` overrides in the loop), so pinning breaks nothing. **Gotcha:** apply the check to the *requested* model, not the failover substitute, or legit failover 403s. **Touches:** a new per-agent field (small plumb through `types.py`/config/proxy). Cheap immediate partial win: also call `is_model_compatible` in the proxy (closes OAuth-allowlist evasion in one line) while the pin is built.
- **CPO:** Invisible (agents only ever use their assigned model). Real money/abuse protection. Do it.

### H6 + M4 — wallet caps. **Do now, together, with a stablecoin allowlist.**
- **PE:** Value tokens in USD before the cap check — but **a stablecoin 1:1 allowlist `{USDC,USDT,DAI: 1.0}` (× the `decimals` already fetched at `wallet.py:485`) is simpler and safer than a live price feed**: it works offline, so the cap check can be **fail-closed** without depending on coingecko uptime. The drain exploit uses stablecoins, so this covers the real case. Non-stable tokens → coingecko-by-contract; unknown + cap-configured → reject. **M4 must land with H6** (a per-agent `asyncio.Lock` across check→broadcast→record) — without it, token transfers are still *racily* bypassable, and the lock is independently correct for EVM nonce safety. **Risk:** fail-closed-on-unknown-price would block legit exotic-token transfers when a price API is down — the stablecoin allowlist avoids that for the 99% case, which is why it's the right shape, not a live feed.
- **CPO:** Real money. Invisible to fix for the common path (stablecoin payments just work). Do it.

### H10 — cross-team blackboard. **The one real judgment call. Do (b)+(c). NOT (a).**
- **CPO:** Cross-*team* isolation is **aspirational/half-built, not a delivered guarantee**. The default fleet is a single agent; multi-agent templates coordinate over *shared* semantic namespaces (`tasks/*`, `output/*`) **by design** — that shared blackboard *is* the collaboration substrate, and it's what makes "it builds your workforce" actually work. For a single-tenant product, the threat isn't "another tenant reads my data," it's "a compromised agent tampers with a peer's keys." **Naively enforcing a `projects/{team}/` prefix (option a) would break the default agent *and every multi-agent template's coordination*** — a fleet-wide functional regression dressed as a security fix. That is the textbook "quietly complicate the thing whose whole pitch is simplicity" failure.
- **PE:** Right solution = **(b)** make `apply_template` create+join a team so the *existing* narrowing machinery (`_add_project_blackboard_permissions` + the `mesh_client` auto-prefix) runs — collaboration is preserved *within the team*, cross-team narrowing happens for free, and `enforce` mode finally has a scope to enforce against; **plus (c)** stop minting bare `["*"]` on the create/operator paths and **fix the misleading "team isolation covers the blackboard" framing**. Bundle **M13** (separator-aware matcher) here, since `projects/{team}/*` scoping only starts mattering once teams are real. **This is a feature-completion, not a lockdown** — it makes the half-built thing coherent without regressing collaboration. **Risk:** must verify every template's agents land in a team and their natural keys auto-prefix correctly before flipping any enforcement; ship the team-wiring first, enforcement later, behind the existing `OPENLEGION_TEAM_SCOPE_MODE` flag (which already exists — the gap was that it was never wired to the blackboard gate).

### The sanitization reframe (M2, M23, M12, H15, M1) — **stop selling hygiene as security.**
- **PE + CPO:** Be honest in the docs and the fixes. `sanitize_for_prompt` does not stop plaintext prompt injection. The defensible actions in this cluster: **M12 length-cap** (real context-bloat/DoS control) + treat MCP servers as semi-trusted config (they're operator-approved, agents can't add them); **M2 one-line sanitize** for *consistency/Unicode-hygiene* (not anti-injection); **M23 lower the confidence/salience of tool-derived facts** (the real lever against persistent implants — sanitize-on-write is redundant since read-back is already sanitized); **H15/M1 documentation** (reframe the AST validator as authoring-hygiene, keep the marketplace dir read-only, SHA-pin only if a remote-install path lands). Don't do the denylist→allowlist rewrite — it false-positives on legit skills and `run_command` beats it anyway. **The one genuine integrity bug hiding in this cluster is M3** (wake `x-task-id` lets a peer mark a victim's in-flight task `failed`) — that's real cross-agent sabotage, fix it with an `assignee == target` binding (safe for legit handoff auto-close, which always reassigns to the recipient first).

### Edge overstatements (H14, M19) and the `frame-ancestors` correction (M18).
- **PE:** H14 and M19 are cheap one-liners worth doing, but neither is High in single-tenant (H14 needs the operator's own SSO session; M19 needs a Caddy bypass). **M18 correction:** use `frame-ancestors 'self'` + `X-Frame-Options: SAMEORIGIN`, **not** `'none'`/`DENY` — the dashboard embeds the VNC viewer same-origin, so `'none'` would break the viewer. **M6 correction:** loopback-binding KasmVNC is safe because the proxy already reaches it over `127.0.0.1` inside the container; add a regression test pinning the browser-container network separation. **M19:** `docs_url=None` gated behind `OPENLEGION_ENABLE_DOCS` so dev keeps the explorer.

---

## 5. Sequenced remediation waves

Ordered so each wave is independently shippable and testable, and nothing blocks on something later. Every item carries a regression test that pins the new boundary.

**Wave 0 — the infection vector (hours, zero UX risk):**
- C1: `enable_icc:false` on the agent network (one line). Test: a peer agent gets connection-refused to another agent's `:8400`.

**Wave 1 — invisible high-value enforcement (small, server-side, no UX change):**
- Lifecycle (H11/H12/M16/L13): re-converge delete onto `stop_agent(remove_data=True)`; pop token + `permissions.reload()` in `_cleanup_agent`; pop on create-failure. Test: deleted agent's token 401s; recreated same-name agent gets a fresh empty volume; archive still retains.
- H1: single-source ceiling, enforced in `edit_agent_soft`. Test: edit-soft granting `can_use_wallet` → 400; dashboard path still works.
- H6 + M4: stablecoin allowlist valuation + per-agent wallet lock. Test: USDC transfer over cap → rejected; N concurrent transfers don't overshoot daily.
- H16, M24: operator-gate `/mesh/traces`; unconditional caller-resolve on the profile endpoint. Test: agent token → 403 on traces; profile honors can_message regardless of param.
- H8 + M11: `_tick` try/except (crash containment, do first) + content-aware schedule validation + per-agent cron cap. Test: `*/0 * * * *` rejected; scheduler survives a poison job.

**Wave 2 — resource bounds (small, additive, generous defaults):**
- H4 (body middleware, reuse `browser/server.py:91-106`, 8 MiB) — also covers M8's size half.
- H5 + H7 (task-create rate 300/min, pending cap 200, parent-depth 25; lane queue maxsize 100 with 429 backpressure; stop reusing the `blackboard_write` bucket).
- H13 (browser slot try/except + reconciliation sweep).
- M10 (stream-abort http body, clamp timeout 1–120s), M14 (atomic+locked perms.json write, fail-closed load), M15 (create lock), M22 (trace retention 168h), M26 (WS send serialization), M8 (OAuth input cap).
- Toggles to expose with safe defaults: `OPENLEGION_MAX_BODY_MB`, `OPENLEGION_MAX_PENDING_TASKS_PER_AGENT`, `OPENLEGION_MAX_CRON_JOBS_PER_AGENT`, `OPENLEGION_TRACE_RETENTION_HOURS`. Pure-correctness fixes stay non-toggleable.

**Wave 3 — model/identity & external surfaces (small–medium):**
- H3 (per-agent model pin; `is_model_compatible` in proxy as the one-line interim).
- H2 (owner-gate `/addkey`/`/steer`/`/broadcast`/`/reset`), H9 (decouple mandatory WhatsApp secret), M7 (webhook dedup), L2 (flip `can_request_user_credentials` + populate on templates).
- M3 (`x-task-id` assignee binding), L9 (`origin_user` creator binding), L17 (MCP startup timeout), M12 (length-cap MCP tool metadata).
- Edge: H14 (`..` reject) + M17 (`nosniff`), M18 (`frame-ancestors 'self'`), M19 (`docs_url=None` env-gated), M6 (loopback bind + separation test), L11 (security-header middleware), L12, M21 (service-side `_is_blocked_ip`), L15, L16.
- Hygiene: L1 (stop logging token slices), L10 (load-tier credential classification), M25 (Pillow floor + adopt a lockfile).

**Wave 4 — the careful feature-completion (medium, needs validation):**
- H10 (b)+(c): `apply_template` creates+joins a team; stop minting `["*"]`; bundle M13 (separator-aware matcher); then wire `OPENLEGION_TEAM_SCOPE_MODE=enforce` to the blackboard gate. Test: every template's agents land in a team and coordinate via auto-prefixed keys *before* any enforcement flips. Fix the docs framing.

**Documentation-only (no code):** M1 (AST validator is hygiene, not a boundary), H15 (marketplace trust posture + SHA-pin-if-remote), L3 (CRED-tier is agent-readable), L14 (MCP secret-in-env asymmetry), M23 residual, M5 (`BROWSER_AUTH_TOKEN` is a fleet-wide superuser credential), the host-network/egress-disable double-opt-in footgun, L19.

---

## 6. What to explicitly NOT do (and why)

- **Don't enforce a `projects/{team}/` blackboard prefix naively (H10 option a).** It breaks the default agent and every multi-agent template's coordination. Complete the team-wiring instead.
- **Don't make the operator ceiling a config toggle (H1).** It becomes an injection target; the dashboard already *is* the human override.
- **Don't rewrite the AST validator as an allow-list (M1).** False-positives on legit skills; `run_command` beats it regardless. The container is the boundary — say so in the docs.
- **Don't ship the "sanitize_for_prompt" fixes as anti-injection (M2/M23/M12-desc).** They're Unicode hygiene. Sell them as such, and put the real effort into M3, length-caps, and trust posture.
- **Don't build per-agent browser identity now (M5).** `BROWSER_AUTH_TOKEN` never leaves the mesh; document it as a superuser credential and audit leak paths instead.
- **Don't fail-closed wallet on unknown token price (H6).** Use the stablecoin allowlist so the cap works offline; only reject unknown *non-stable* tokens.
- **Don't pin the browser to an IP-literal URL for SSRF (M20).** SNI/cert breakage risk; the service-side `_is_blocked_ip` check (M21) plus the existing iptables layer is the lower-risk equivalent.

---

## Bottom line for the CPO

The original question — *can agents infect each other, can the operator be infected* — has a clear answer and a clean fix path: **C1 (one line) + the Wave-1 server-side enforcement** close the real exposure, and they're **invisible to the operator**, so they strengthen the trust story without touching the simplicity pitch. The wallet caps (H6/M4) protect real money, also invisibly. The single place where security and the product's nature genuinely trade off — cross-team blackboard (H10) — resolves *for* the product: the right move is to finish the team feature, not to bolt on a lockdown that breaks collaboration. Almost everything else is a cheap, invisible hardening you can ship in waves behind regression tests, plus a short list of "be honest in the docs" items where the current code oversells its own guarantees.
