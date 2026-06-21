# Operator Memory & Context Overhaul — Implementation Plan

**Created:** 2026-06-09. **Author:** Claude (principal-eng review, pre-implementation).
**Status:** SHIPPED — archived. Operator memory & context overhaul shipped (all phases merged + de-flagged, all agents).

> **How to use this doc (post-compaction entry point):** This is self-contained. Read §1–§4 for context + diagnosis, resolve §5 Decision Gates with the user, then implement Phase 1 (§6) in the Sequencing order (§8). Every integration point is pinned to `file:line` against `origin/main` as of 2026-06-09 (HEAD `9dc1210`/`8cbfb55d` region) — re-grep to confirm line numbers haven't drifted before editing. Verify on cake using §9 (no-secrets discipline is mandatory).

---

## 1. The problem

User report: the cake operator "is slow, doesn't act, doesn't understand." Diagnosis (live + code) showed the operator is *capable* (it correctly followed a 3-stage chain and diagnosed a dedup-skip) but is **drowning in stale, bloated context**:

- **MEMORY.md = 29 KB** injected every turn — 17 near-duplicate "Proactive Flush"/"Extracted" sections describing a **dead fleet** (`openlegion-marketing`: strategist/seo-analyst/copywriter) that **contradicts the live fleet** (`openlegion-content-seo`: trend-scout/seo-strategist/page-writer). opus reads two conflicting fleet models every turn.
- **~53 non-browser tool schemas** presented (76 allowlist − 23 browser_* gated off); operator uses ~10–20.
- Real turns ~**56 K tokens** on **claude-opus-4-8** (slowest model), non-streaming for handoffs.

The good news: **our memory architecture is already correct** — gbrain ("Compiled Truth + Timeline") and hermes-agent ("bounded files + Curator background consolidation") both independently arrive at the design we already have (compiled head + append-only log + an LLM consolidator). **The maintenance loop is what's broken**, not the architecture.

### Guiding principle (the UX north star)
"Users chat with the operator → it knows exactly what to do." Decompose that into three properties every change must preserve or improve:
1. **Reasoning clarity** — the operator correctly maps user intent → action. Served by the **memory track** (clean, current, non-contradictory context). This is the bulk of "doesn't understand / doesn't act."
2. **Capability awareness** — the operator always knows *what it can do*. A tool being expensive to load must NOT make the operator forget it exists. Served by an always-present **capability index** (Phase 2), never by removing tools.
3. **Capability access** — the operator can actually use the right tool when needed, ideally without a visible search detour (intent-prefetch).

Removing tools (the earlier "trim" idea) is **rejected** — it trades capability awareness for leanness. All tools are important across operator + worker scope; the lever is *grouping + lazy schema-loading*, not deletion.

## 2. Environment / deploy mechanics (load-bearing — don't skip)

- **cake** = `cake.engine.openlegion.ai`. SSH: `ssh -i ~/.ssh/provisioner_key -o IdentitiesOnly=yes -o BatchMode=yes root@cake.engine.openlegion.ai`. Repo at `/opt/openlegion/repo`, `.venv/bin/python`. Mesh = host systemd service `openlegion`; agents = Docker containers `openlegion_{agent}`; operator workspace = volume `openlegion_data_operator` → `/data/workspace/`; per-agent memory DB = `/data/{agent}.db`.
- **cake is on the Anthropic OAuth subscription** (proven via `data/costs.db`: native `anthropic/*` = $0.00; gateway `openlegion/*`/`minimax`/`gpt` carry cost). Don't reason about per-token $ for Claude on cake — the lever is rate-limit headroom + latency.
- **`PYTHONUNBUFFERED=1` drop-in** was added at `/etc/systemd/system/openlegion.service.d/unbuffered.conf` so mesh INFO logs actually flush. **cake INFO logs were below the effective level / buffered before** — do not trust INFO log greps for verification; use the **cost ledger** and `docker exec` file inspection instead.
- **Deploy by code location:**
  - **Host/mesh code** (`src/host/**`, `src/cli/config.py` launcher, `src/dashboard/**`): `cd /opt/openlegion/repo && git checkout . && git pull --ff-only && systemctl restart openlegion`. (`config/` is gitignored, so pull is safe.)
  - **Agent code** (`src/agent/**` — memory.py, context.py, workspace.py, loop.py): **baked into `openlegion-agent:latest`** → must `docker build -t openlegion-agent:latest -f Dockerfile.agent .` **then** `systemctl restart openlegion`. A git-pull alone does NOT update agent code. First restart triggers a ~120 s browser-image rebuild.
- **Conventions:** worktrees for ALL code changes (`isolation: "worktree"` for subagents). One PR per track. `gh pr create` → CI green (`gh pr checks N --watch`) → Codex review → squash-merge `gh pr merge N --squash --admin --delete-branch`. Never merge to main directly. No Co-Authored-By trailers. Run tests in subagents (worktrees) to keep output out of context.
- **No-secrets discipline (hard rule):** never read `.env` values, keys, tokens, `docker inspect` env, auth_tokens. OK: service state, `docker ps`, config (agents.yaml / permissions / ALLOWED_TOOLS names), workspace markdown, transcripts (user's own data), cost aggregates, structural checks that print only booleans/lengths/fingerprints.

## 3. Diagnosis — five compounding root causes (all pinned)

1. **Consolidation never fires.** `_maybe_consolidate_memory` (context.py:411) runs *only* inside compaction `_do_compact` (context.py:268-318), which triggers at 60%/70% of the context window (context.py:31-33). Operator at ~28% of opus's 200 K never reaches it → head never re-derived.
2. **When it does fire, it reads the OLDEST log.** `f"## Recent activity log (newest last)\n{log[:_SUMMARIZATION_INPUT_LIMIT]}"` (context.py:445) — `log[:20000]` is the *first* 20 K chars; the log is oldest-first, so the newest entries are truncated off. **Verified bug.**
3. **Salience never decays in the chat path.** `decay_all` (memory.py:523) is called only on fresh *task* starts (loop.py:803, 848), never in the chat/operator path → `get_high_salience_facts` (memory.py:531, `ORDER BY decay_score DESC`) drifts permanently toward old, frequently-touched facts. These feed both consolidation (context.py:429) and task initial-context (loop.py:1542).
4. **Fact dedup is exact-key only** (memory.py:266-303, `WHERE key = ?`). Free-form LLM keys (`preferred_language` vs `user_language_preference`) create separate rows; no embedding-similarity merge on write (the `facts_vec` index at memory.py:177 exists but is used only for category assignment + search).
5. **The append-log never dedups.** `append_memory` (workspace.py:644) blindly appends every flush block; the injected `## Recent` 5 K slice (workspace.py:487, `_MEMORY_RECENT_LOG_CHARS=5000` at :106) re-surfaces near-duplicates every turn until consolidation folds them.

**Bonus finding — caching connection:** #1073 marks `cache_control` mesh-side (credentials.py), but the agent's system prompt embeds per-turn-volatile content — round/context warnings (loop.py:4259-4272), operator playbooks (loop.py:4243-4251), the shifting `## Recent` memory slice, and 5-min runtime context. Any of these mutating busts the cached prefix. The bloated, churning system prompt both slows the model *and* defeats the caching we just shipped.

## 4. What the prior art tells us (mapped, not copied)

- **gbrain (garrytan/gbrain, master):** Compiled-Truth (rewritten on update) + Timeline (append-only) = our head + log. Injects **only the head**; ranks compiled-truth > timeline at read time. "Dream cycle" cron rewrites the head from the timeline with a **hard spend cap**, auto-applying safe merges and **escalating** risky contradictions (never blind-overwrite). Facts are **dated + sourced** (makes prefer-recent possible). Contradiction probe = date pre-filter → small-model judge (confidence floor 0.7) → cache by content-hash → severity tiers.
- **hermes-agent (NousResearch, main):** Hard-capped memory files (MEMORY.md 2,200 chars) where the **write tool refuses overflow** ("consolidate now") = budget-forced dedup. **Frozen snapshot** at session start (mid-session edits defer to next session) protects the prompt-cache prefix. **Retrieve-cold/inject-hot** (FTS over cold history). **The Curator** = a background `AIAgent` fork on a 7-day interval that consolidates near-dup skills and archives stale (move, don't delete). **Tool Search** = core tools always loaded; long tail behind `tool_search`/`tool_describe`/`tool_call`, auto-activated only when deferrable schemas exceed ~10% of the window. **`execute_code`** = model writes one Python block; intermediate tool results never enter context, only `print()` — collapses N tool rounds into one turn.

**Architectural note (load-bearing) — DG-1 resolved: background cron pass. [Codex-corrected]** Consolidation needs the agent's **in-container** workspace + memory DB + LLM, which the mesh-side cron scheduler can't reach directly. The user chose a **background cron pass** (off the live conversation, like gbrain's "dream cycle" / hermes' "Curator") over the heartbeat. **CORRECTION:** `CronScheduler` does NOT support calling an arbitrary endpoint — it dispatches a job only via `/invoke` (a tool), heartbeat, or chat (cron.py:308-336, 461-518). So model it **exactly** on the `ensure_summary_job` → `compose_work_summary` precedent: register a cron job with `tool_name="run_maintenance"` (or similar) that the scheduler dispatches over the **existing cron→`/invoke` path** to a new **maintenance builtin tool** which calls `ContextManager.run_maintenance()`. `/invoke` runs the tool directly (not a chat turn) → still off the live latency path, fleet-wide, observable via the cron audit log, and **reuses existing machinery (no new endpoint, no new scheduler mode).** *Alternative if the tool/invoke path proves awkward:* an agent-internal asyncio timer (hermes' Curator model) — self-contained, no cron job.

---

## 5. Decision Gates — RESOLVED (user, 2026-06-09)

- **DG-1 — A2 trigger:** ✅ **Background cron pass** (mesh `CronScheduler` job → agent `/maintenance/consolidate` endpoint, off the live loop). See §4 architectural note.
- **DG-2 — Operator model (C4):** ✅ **Keep opus-4-8 for now.** The bloat is the real latency driver; "doesn't understand" is context-pollution, not model weakness. Re-assess after Phase 1. (Reversible config change.)
- **DG-3 — Phase 2 scope:** ✅ **Bigger bets approved for the NEXT roadmap phase** (not this effort). Phase 1 = memory + model. Phase 2 (its own authorized project, sequenced after Phase 1): Grouped Tool Search (the new centerpiece — see §7), `execute_code`, semantic dedup, dated-facts memory v2.
- **DG-4 — A1 keep-list:** the curated current-fleet facts to retain — **show user before saving** (operational, at implementation time).
- **DG-5 — C3 inclusion:** cache-prefix stabilization stays in Phase 1 as the **last/lowest-priority** item (protects #1073's value); defer to Phase 2 if its behavior risk feels high at implementation time.
- **Tools (was B1):** ✅ **No tool removal.** Phase 1 makes **no tool changes**; the tool problem is solved properly in Phase 2 via Grouped Tool Search (capability index + lazy schemas + intent-prefetch). All tools remain available in the interim (their schemas are cached by #1073, so the interim cost is bounded; the memory track is the bigger latency win).

---

## 6. Phase 1 — authorized core

### A1 — Immediate MEMORY.md hotfix (operational, cake-only; OPTIONAL relief)

**Goal:** give the operator an instant clean baseline before A2 ships. *May be skipped* if we go straight to A2 and let its first run self-clean — but A1 also gives A2 a clean baseline to validate "maintain" against.

**Approach (simplest):** with the operator **process stopped** (see risk), rewrite all three surfaces:
- (a) Compiled head: replace with a tight, deduped, current-fleet fact set, written **in-format** so `_split_memory` (workspace.py:429) parses it (use `write_compiled_memory`, workspace.py:661, or write the exact `# Long-Term Memory\n\n<!-- compiled:begin -->\n{head}\n<!-- compiled:end -->\n\n{log}` shape).
- (b) Log: truncate the 17 stale flush sections.
- (c) DB: prune stale high-salience facts in `/data/operator.db` `facts` table (else they re-inject at the first consolidation via `get_high_salience_facts`, memory.py:531).

**Codex must-fixes:** Stop the operator *process/container* (not just wait for idle — `execute_heartbeat` sets `state="working"` after its gate, loop.py:1956-1980, and there's no file lock on `append_memory`/`write_compiled_memory`). Clean **all three** surfaces, not just the markdown.

**Risk/gaps:** losing genuinely-useful facts → mitigate with DG-4 (show before/after). Hand-editing prod memory is operational; do it with the container stopped, keep a backup copy of the originals first.

**Deploy:** operational only (no PR). `docker stop openlegion_operator` → edit `/data/workspace/MEMORY.md` + prune DB → `docker start` (or `systemctl restart openlegion`).

**Acceptance:** head < ~6 KB, describes only the live fleet, no `openlegion-marketing` references; operator's next turn is visibly leaner.

### A2 — Fix + relocate consolidation (THE core memory fix) — PR, agent-side

**Goal:** make consolidation actually run for a large-window, frequently-restarted operator, reading the *right* data, so the head stays current and deduped automatically. Reuse the existing `_maybe_consolidate_memory` LLM merge (it already says "merge duplicates, prefer recent, drop stale", context.py:438-445) — **fix the plumbing, don't rebuild the algorithm.**

**Integration points:**
- `context.py:445` — **fix the slice** `log[:_SUMMARIZATION_INPUT_LIMIT]` → `log[-_SUMMARIZATION_INPUT_LIMIT:]` (read NEWEST), re-aligned to a `\n## ` boundary so a section isn't split.
- `context.py:411-468` — extract the body into a NEW public callable `ContextManager.run_maintenance()` (does not exist yet — create it) that runs **outside compaction**, keeping the `consolidation_due` (≥6 h, workspace.py:254/673) + `_CONSOLIDATION_MIN_LOG_CHARS` (1,500, context.py:37) gates.
- **Trigger = background cron pass via a maintenance TOOL [Codex-corrected]:** create a maintenance **builtin tool** (e.g. `run_maintenance`) that calls `run_maintenance()`, and register a cron job with `tool_name="run_maintenance"` modeled on `ensure_summary_job`/`compose_work_summary`. The scheduler dispatches it over the **existing cron→`/invoke` path** (cron.py:308-336 + the callback wiring in **cli/runtime.py:1107-1187, 1329-1340**). No new endpoint, no new scheduler mode. Default every 6 h; fleet-wide; observable via cron audit log.
- **Locking [Codex must-fix]:** chat holds `_chat_lock` (loop.py:383, 2639) but workspace memory writes are plain file read-modify-write (workspace.py:644-669) and `MemoryStore` uses a shared SQLite connection via executor with NO maintenance lock (memory.py:68-70, 254-258). `run_maintenance()` must **acquire `_chat_lock` (or skip if busy/mid-turn) AND hold a dedicated consolidation lock**, or it races `_flush_to_memory()` + salience updates. Cron has per-job locks but **nothing per-agent** to inherit (cron.py:441-453) — implement the busy/skip policy in the tool.
- **Failure backoff [Codex must-fix]:** consolidation currently stamps the sentinel only on non-empty LLM output + write success and silently returns on failure (context.py:447-460). A cron firing every tick would **retry on every tick** after a failure. Add a failure backoff / error sentinel so repeated LLM/write failures don't hammer the model.
- **Chat-path decay fix:** add a **time-gated** `decay_all` (memory.py:523) in the same maintenance pass, tracked by its own sentinel (mirror `consolidation_due`), so the operator's salience decays ~once per cycle. **Guard against double-decay** with the task-path decay (loop.py:803/848) — gate by the sentinel, don't decay if the task path decayed within the window.
- **Log trim decision:** after writing the new head, trim the consolidated log sections (the head now subsumes them). `write_compiled_memory` preserves the log (workspace.py:661), so the log regrows otherwise. Decide explicitly: trim sections older than the newest-N, or archive them.

**Simplest version that still solves it:** slice-fix + heartbeat trigger + lock + decay-fix + log-trim. **NOT** in A2: semantic dedup (A4), contradiction probe, dated facts — those are Phase 2. The existing LLM merge is sufficient for v1.

**Load-bearing assumption (must validate first):** the existing consolidation prompt produces a *good* deduped head. It has **never run** for this operator. **Validate by running it once on a test/throwaway agent (or cake operator after A1) and inspecting the output** before trusting it in the heartbeat loop.

**Risks/gaps:** extra LLM call per cycle (gated ≥6 h → ≤4×/day, acceptable; cap it). Failed consolidation doesn't stamp the sentinel (context.py:447-455) → rapid-failure could retry — add a short backoff. Consolidation reads `head + newest-log + high-salience-DB` — if A1 didn't clean the DB, stale-salient facts can still leak in (sequencing: A1 before A2, or accept the first run cleans imperfectly then converges).

**Tests:** unit — slice now returns newest entries; `run_maintenance()` runs when gates met and skips otherwise; lock prevents re-entry; busy-skip works; decay is gated (no double-decay with the task-path decay); log trimmed after head rewrite; the `/maintenance/consolidate` endpoint rejects non-internal callers. Mock the LLM (return a known deduped head). Run in a worktree subagent.

**Deploy [Codex-corrected split]:** spans both zones — the `run_maintenance` tool + `ContextManager.run_maintenance()` are **agent code → image rebuild + restart**; the cron job registration + its callback wiring (`src/host/cron.py` AND **`src/cli/runtime.py:1107-1187, 1329-1340`**) are **host code → git pull + `systemctl restart openlegion`**. Ship/deploy together.

**Acceptance:** on cake, after deploy + one cron cycle (or a manual `/invoke` of `run_maintenance` via the internal path), `docker exec openlegion_operator cat /data/workspace/MEMORY.md` head is current + deduped and stays bounded across cycles; `.memory_consolidated` sentinel advances; cron audit log shows the job firing.

### B1 — Tools — NO CHANGE in Phase 1 (deliberate)

Per the user's direction (2026-06-09): do **not** remove or trim operator/worker tools. Removal trades capability-awareness for leanness (violates UX north star #2). All tools stay available in Phase 1; their schemas are part of the #1073-cached prefix, so the interim per-turn cost is bounded, and the memory track is the larger latency win. The tool-context problem is solved properly in **Phase 2 → Grouped Tool Search** (§7). *Nothing to build here in Phase 1* — this entry exists to record the decision and prevent a future "just trim it" regression.

### C4 — Operator model — KEEP OPUS (DG-2 resolved)

Keep `claude-opus-4-8`. Rationale: "doesn't understand" is context-pollution (fixed by the memory track), not model weakness; opus maximizes reasoning clarity (north star #1). Re-assess speed after Phase 1 lands. No code; revisit only if post-Phase-1 latency is still unacceptable (then it's a one-line `model:` config change + restart, validated by `is_model_compatible`).

### C3 — Cache-prefix stabilization → MOVED to Phase 2 [Codex PT3a]

Codex confirmed C3 reorders behavior-sensitive system-prompt content (loop.py:4168-4271, workspace.py:487-513) and is risky to ship alongside A1/A2. **Decision: move C3 to Phase 2** so Phase 1 stays a clean, low-risk memory-only effort. (It pairs naturally with A5's "inject head-only" anyway.) See §7.

---

## 7. Phase 2 — approved for the NEXT roadmap phase (DG-3); sequenced after Phase 1

### B2 — Grouped Tool Search (the centerpiece; the *correct* fix for tool bloat)

**Problem statement (principal-eng + CPO):** keep per-turn schema load lean WITHOUT losing the operator's "I know exactly what to do" property. The naive Tool Search trap: a model won't search for a tool it doesn't know exists → deferred capabilities become invisible → the operator fails to do what the user asked. So the design must separate **capability awareness** (always present, cheap) from **schema loading** (lazy).

**Design — three layers:**
1. **Always-loaded core (full schemas).** The ~15-20 daily-driver tools (orchestrate: hand_off/await_task_event/manage_task/list_agent_queue/workflow_snapshot; diagnose: inspect_agents/read_agent_config/get_system_status; memory: memory_save/search; comms: notify_user; http_request/web_search). The common case needs zero search.
2. **Always-loaded capability INDEX (names + one-line purpose, grouped by job-to-be-done — NO full schemas, ~300-500 tokens).** This is the key innovation that answers the user's "out of context until searched" worry: the capability is **never invisible**, only its verbose schema is lazy. The index doubles as a self-documenting menu of what the operator can do. Groups (intent-named, not module-named):
   - *Fleet setup* — create_agent, create_team, apply_template, list_templates, add_agents_to_team, manage_team … ("build or restructure a team")
   - *Scheduling* — set_cron, list_cron, remove_cron ("recurring/timed work")
   - *Credentials & access* — vault_list, request_credential, request_browser_login ("an agent needs to reach a service")
   - *Goals & review* — set_team_goal, manage_goals, rate_delivery
   - *Audit & undo* — list_pending, cancel_pending_action, archive_audit_before, undo_change
   - *Web & browse* — browser_* (the 23) ("research / interact with sites")
3. **On-demand schema loading.** Bridge tool `load_tools(group | tool_name)` (or hermes' `tool_search`/`tool_describe`/`tool_call`) pulls full schemas into context for subsequent turns. Two activation modes: **explicit** (operator calls it when it recognizes the intent) and **intent-prefetch** (the CPO touch — on a new user message, map intent → likely group(s) and pre-load, so the operator acts without a visible search round-trip: "set up an SEO team" → prefetch *Fleet setup*).

**Budget-gated (hermes):** only activate index+defer when an agent's schemas exceed ~10% of the window; small-toolset agents present everything unchanged. **Applies to operator AND worker agents** ("operator scope and agent scope") — general per-agent capability in `get_tool_definitions` (tools.py:451), with a per-role core/deferred grouping.

**Integration points:** `tools.py:451` `get_tool_definitions` (new grouped/index mode), `loop.py:472-501` `_tool_filter_kw` + `loop.py:940` request build (schema injection), the bridge-tool builtins, and the loop state that tracks "currently loaded" groups across turns. Define groups as data (a registry mapping group → tool names + intent string + role-eligibility).

**Required design details [Codex PT2b/PT2d — specify before coding]:** (a) Grouped Tool Search must **extend the existing filter model** (`get_tool_definitions(exclude, allowed)` memoized on `(exclude, allowed)`, tools.py:451-463; `_tool_filter_kw`, loop.py:470-502) or introduce a separate schema-selection path — it can't bolt on without touching the filter logic. (b) "Loaded group persists for subsequent turns" is **not automatic** — tools are requested fresh at each LLM call (loop.py:940, 2120, 3484), so a new **loaded-groups set must live on the `AgentLoop` instance and be folded into the `get_tool_definitions` cache key / builder**; otherwise `load_tools` mutating state immediately would also change schemas *mid-turn* and bust the #1073 cache. Defer the load to the **next turn boundary** (mirrors hermes' "don't change toolsets mid-conversation" cache invariant). **Risk/gaps:** intent-prefetch misclassification → `load_tools` remains the always-available fallback (capability never lost); prefetch is practical on the chat path (user message available pre-build, loop.py:2974-3017) but task/heartbeat paths need separate design. This is the schema/exec decoupling B1 couldn't give (fewer schemas, everything still callable).

### A4 — Semantic fact dedup + contradiction probe
Vector-similarity merge on the consolidation/write path (`facts_vec` infra exists, memory.py:177). gbrain contradiction probe: date pre-filter → small-model judge (confidence floor 0.7) → content-hash cache → severity tiers (auto-merge safe, escalate identity-level). Hooks: `store_fact` (memory.py:305) / the maintenance pass. Strengthens A2's dedup beyond the LLM-merge.

### A5 — Memory v2: dated + sourced facts, inject-head-only, read-time precedence
Add `(source_type, date)` to facts (schema change + migration, memory.py:165) to enable real prefer-recent; drop/shrink the `## Recent` injection (gbrain injects only the head); rank head > log at retrieval. Pairs with C3.

### C3 — Cache-prefix stabilization (moved here from Phase 1 per Codex PT3a)
Make #1073's caching actually hit: move per-turn-volatile content OUT of the cached system block — context/round warnings (loop.py:4250-4271), operator playbooks (loop.py:4236-4248), the `## Recent` memory slice (loop.py:4168-4175 / workspace.py:487-513; subsumed by A5's inject-head-only), 5-min runtime context — appending after the cache breakpoint or into the first user message. **Behavior-sensitive (reordering instructions changes weighting)** → behind a flag, with before/after behavior checks. Sequence right after A5 (they share the head-only change).

### C2 — `execute_code` (code-as-action)
Highest-leverage *speed* lever for the orchestrator: collapse multi-tool-round workflows into one Python block; intermediate tool results never hit context, only `print()`. We already have Docker isolation. Significant build + security surface (env scrub for KEY/TOKEN/SECRET, tool whitelist, no recursion into execute_code/delegate). Directly attacks the operator's 10+ silent tool rounds per task.

---

## 8. Sequencing (each step testable; nothing blocked on later work)

**Phase 1 (this effort) = MEMORY ONLY.** C4 = keep opus (no work). B1 = no tool changes (no work). C3 = moved to Phase 2.
1. **A1 hotfix** — instant relief on cake; gives A2 a clean baseline (stop process → clean head+log+DB → restart). Optional but recommended.
2. **A2 validate-once** — run the slice-fixed consolidation on a throwaway/cake operator and inspect the output BEFORE wiring the cron (load-bearing: the prompt has never run).
3. **A2 PR** — create `ContextManager.run_maintenance()` (slice-fix + decay-fix + log-trim + chat-lock/busy-skip + consolidation-lock + failure-backoff) + a `run_maintenance` builtin tool + a cron job (`tool_name="run_maintenance"`, modeled on `ensure_summary_job`) wired in cron.py + cli/runtime.py. The core fix; makes A1 durable. Spans agent (image rebuild) + host (restart).

**Phase 2 (next roadmap phase, separate authorized project):** B2 Grouped Tool Search (centerpiece) → A4 semantic dedup → A5 memory v2 (inject-head-only) → **C3 cache-prefix stabilization** (pairs with A5) → C2 execute_code. Each its own PR-set.

Dependency notes: A1 should precede A2's first prod run (clean DB) OR accept that A2 converges over a few cycles. C3 pairs with A5's inject-head-only. Phase 2's A4/A5 build on A2's maintenance pass (extend it, don't replace).

## 9. Verification playbook (cake, no-secrets)

- **Memory:** `docker exec openlegion_operator sh -c 'wc -c /data/workspace/MEMORY.md; grep -c "^## " /data/workspace/MEMORY.md'` (size + section count); read the head to confirm current-fleet-only. Sentinel: `ls -la /data/workspace/.memory_consolidated`.
- **OAuth still healthy / not silently billed:** `data/costs.db` — native `anthropic/*` rows stay $0.00 (subscription); gateway rows are the only billed ones.
- **Tools:** `echo $ALLOWED_TOOLS | tr , '\n' | wc -l` in the operator container.
- **Health after any deploy:** `systemctl is-active openlegion`, `docker ps | grep -c openlegion_` (expect 17), error scan `journalctl -u openlegion --since "-5 min" | grep -aiE "ERROR|Traceback"`. INFO logs now flush (PYTHONUNBUFFERED added) but remain unreliable — prefer file/ledger checks.
- **Consolidation actually ran:** inspect the head changed + sentinel advanced after a heartbeat cycle (don't rely on the INFO log line).

## 10. Decisions — status (mirror of §5, all resolved 2026-06-09)
DG-1 ✅ background cron pass · DG-2 ✅ keep opus · DG-3 ✅ bigger bets → Phase 2 roadmap · DG-4 ⏳ A1 keep-list shown-before-save at implementation time · DG-5 ✅ C3 in Phase 1, lowest priority · Tools ✅ no removal, grouped Tool Search in Phase 2. **Only remaining at-implementation gate:** DG-4 (show the A1 keep-list before writing it).

## 11. Reference: today's shipped context (2026-06-09)
Merged to main: #1068 (await polls durable status), #1069 (result_summary on task row), #1072 (handoff prompt + envelope extraction), #1073 (Anthropic prompt caching, OAuth + litellm paths). All live-verified or deployed on cake. The await/handoff flow works at the primitive level; multi-hop chain-following is per-hop (operator awaits each task; workers can't await — operator-only). Do not regress these.
