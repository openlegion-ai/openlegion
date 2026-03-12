# Dashboard UX Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve dashboard UX with chat timestamps, clearer agent cards (health badge rename + heartbeat countdown), cron button fixes, and heartbeat deletion protection.

**Architecture:** All changes are in the dashboard layer (Alpine.js SPA + FastAPI endpoints). No engine/agent loop changes. The `/api/agents` endpoint gets heartbeat data from `cron_scheduler.find_heartbeat_job()`. A 1s client-side interval computes countdown from `next_run` timestamps. Cron buttons get visual and functional fixes.

**Tech Stack:** Alpine.js, Tailwind CSS, FastAPI, Python dataclasses

---

### Task 1: Backend — Heartbeat info in `/api/agents` and heartbeat delete guard

**Files:**
- Modify: `src/dashboard/server.py:191-207` (api_agents endpoint — agent entry construction)
- Modify: `src/dashboard/server.py:1550-1556` (api_cron_delete endpoint)
- Test: `tests/test_dashboard.py`

**Step 1: Add heartbeat data to agent response**

In `src/dashboard/server.py`, inside `api_agents()` (after line 206, before the `vnc_url` check), add heartbeat lookup for each agent:

```python
            # Inside the for loop, after building `entry` dict:
            if cron_scheduler is not None:
                hb = cron_scheduler.find_heartbeat_job(agent_id)
                if hb:
                    entry["heartbeat_schedule"] = hb.schedule
                    entry["heartbeat_enabled"] = hb.enabled
                    entry["heartbeat_next_run"] = hb.next_run
```

Also add the same block for over-limit agents (after line 231), but with `None` values:

```python
                # Over-limit agents have no heartbeat data (not running)
```

No heartbeat fields for over-limit agents — the frontend handles missing fields by hiding the row.

**Step 2: Add heartbeat delete guard**

In `src/dashboard/server.py`, modify `api_cron_delete` (line 1550-1556):

```python
    @api_router.delete("/api/cron/{job_id}")
    async def api_cron_delete(job_id: str) -> dict:
        if cron_scheduler is None:
            raise HTTPException(status_code=503, detail="Cron scheduler not available")
        job = cron_scheduler.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.heartbeat:
            raise HTTPException(status_code=403, detail="Heartbeat jobs cannot be deleted")
        if not cron_scheduler.remove_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return {"deleted": True, "job_id": job_id}
```

**Step 3: Write tests**

Add to `tests/test_dashboard.py`:

```python
class TestHeartbeatAPI:
    def test_agents_include_heartbeat_data(self):
        """GET /api/agents includes heartbeat schedule/enabled/next_run."""
        # Setup: create mock cron_scheduler with find_heartbeat_job returning a CronJob
        # Assert: agent entry has heartbeat_schedule, heartbeat_enabled, heartbeat_next_run

    def test_agents_no_heartbeat_when_none(self):
        """Agents without heartbeat jobs don't have heartbeat fields."""
        # Setup: find_heartbeat_job returns None
        # Assert: agent entry has no heartbeat_schedule key

    def test_delete_heartbeat_blocked(self):
        """DELETE /api/cron/{id} returns 403 for heartbeat jobs."""
        # Setup: mock cron_scheduler.jobs with a heartbeat=True job
        # Assert: DELETE returns 403 with "Heartbeat jobs cannot be deleted"

    def test_delete_regular_cron_allowed(self):
        """DELETE /api/cron/{id} works for non-heartbeat jobs."""
        # Setup: mock cron_scheduler.jobs with a heartbeat=False job
        # Assert: DELETE returns 200
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_dashboard.py -x -v -k "Heartbeat"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/dashboard/server.py tests/test_dashboard.py
git commit -m "feat(dashboard): add heartbeat info to agents API, block heartbeat deletion"
```

---

### Task 2: JS — Cron button error handling fixes

**Files:**
- Modify: `src/dashboard/static/js/app.js:1955-1975` (pauseCronJob, resumeCronJob)

**Step 1: Fix `pauseCronJob` to check response status**

At `app.js:1955`, change:

```javascript
    async pauseCronJob(jobId) {
      if (this.cronRunLoading[jobId]) return;
      this.cronRunLoading = { ...this.cronRunLoading, [jobId]: true };
      try {
        const resp = await fetch(`${window.__config.apiBase}/cron/${jobId}/pause`, { method: 'POST' });
        if (resp.ok) {
          this.showToast(`Job ${jobId} paused`);
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Pause failed'}`);
        }
        this.fetchCronJobs();
      } catch (e) { console.warn('pauseCronJob failed:', e); }
      finally { this.cronRunLoading = { ...this.cronRunLoading, [jobId]: false }; }
    },
```

**Step 2: Fix `resumeCronJob` the same way**

At `app.js:1966`, change:

```javascript
    async resumeCronJob(jobId) {
      if (this.cronRunLoading[jobId]) return;
      this.cronRunLoading = { ...this.cronRunLoading, [jobId]: true };
      try {
        const resp = await fetch(`${window.__config.apiBase}/cron/${jobId}/resume`, { method: 'POST' });
        if (resp.ok) {
          this.showToast(`Job ${jobId} resumed`);
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Resume failed'}`);
        }
        this.fetchCronJobs();
      } catch (e) { console.warn('resumeCronJob failed:', e); }
      finally { this.cronRunLoading = { ...this.cronRunLoading, [jobId]: false }; }
    },
```

**Step 3: Also add error toast to `runCronJob` for consistency**

At `app.js:1944`, the run handler checks `resp.ok` for the toast but doesn't show an error. Add:

```javascript
        const resp = await fetch(`${window.__config.apiBase}/cron/${jobId}/run`, { method: 'POST' });
        if (resp.ok) {
          this.showToast(`Job ${jobId} triggered`);
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Run failed'}`);
        }
```

**Step 4: Commit**

```bash
git add src/dashboard/static/js/app.js
git commit -m "fix(dashboard): check response status in cron pause/resume/run handlers"
```

---

### Task 3: JS — Chat message timestamps

**Files:**
- Modify: `src/dashboard/static/js/app.js` (6 message push sites + formatter + session save/restore)
- Modify: `src/dashboard/templates/index.html:1013-1044` (message rendering)

**Step 1: Add `formatRelativeTime` helper**

Add near the top of the Alpine data object (after `renderMarkdown` at ~line 845):

```javascript
    formatRelativeTime(ts) {
      if (!ts) return '';
      const diff = Date.now() - ts;
      if (diff < 60000) return '';  // under 1 min — too fresh to label
      if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
      const d = new Date(ts);
      const now = new Date();
      const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
      if (d.toDateString() === now.toDateString()) return time;
      return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ', ' + time;
    },
```

**Step 2: Add `ts` to all message push sites**

There are 6 places where messages enter `chatHistories`. Add `ts: Date.now()` to each:

1. **Notification push** (`app.js:910`):
```javascript
          this.chatHistories[evt.agent].push({
            role: 'notification',
            content: evt.data?.message || '',
            streaming: false,
            tools: [],
            ts: Date.now(),
          });
```

2. **User message** (`app.js:2516`):
```javascript
      this.chatHistories[agentId].push({ role: 'user', content: msg, ts: Date.now() });
```

3. **Agent response placeholder** (`app.js:2520`):
```javascript
      this.chatHistories[agentId].push({
        role: 'agent',
        content: '',
        streaming: true,
        phase: 'thinking',
        tools: [],
        timeline: [],
        _sawTextDelta: false,
        ts: Date.now(),
      });
```

4. **Steer message** (`app.js:2677`):
```javascript
      this.chatHistories[agentId].push({ role: 'user', content: `[steer] ${msg}`, ts: Date.now() });
```

5. **History restore from API** (`app.js:2292`): No timestamp available from the backend — leave `ts` undefined. The formatter returns `''` for missing timestamps.

6. **Session storage restore** (`app.js:2102`): Preserve `ts` from saved data:
```javascript
          const capped = msgs.slice(-50).map(m => ({
            role: m.role,
            content: m.content,
            streaming: false,
            phase: (m.phase === 'error' || m.phase === 'done') ? m.phase : 'done',
            ts: m.ts || 0,
            // ... rest unchanged
```

**Step 3: Render timestamp in chat template**

In `index.html`, after line 1042 (after the streaming cursor span, before the closing `</div>` of the bubble):

```html
                <div x-show="formatRelativeTime(msg.ts)"
                  class="text-[9px] mt-0.5 opacity-40"
                  :class="msg.role === 'user' ? 'text-indigo-200' : 'text-gray-500'"
                  x-text="formatRelativeTime(msg.ts)"></div>
```

**Step 4: Commit**

```bash
git add src/dashboard/static/js/app.js src/dashboard/templates/index.html
git commit -m "feat(dashboard): add timestamps to chat messages"
```

---

### Task 4: HTML/JS — Agent card health badge rename

**Files:**
- Modify: `src/dashboard/templates/index.html:778-786` (health badge text)
- Modify: `src/dashboard/static/js/app.js` (add `healthLabel` helper)

**Step 1: Add `healthLabel` helper to JS**

Add near other formatters in `app.js`:

```javascript
    healthLabel(status) {
      const map = { healthy: 'Online', unhealthy: 'Degraded', restarting: 'Degraded', failed: 'Offline', unknown: 'Starting' };
      return map[status] || 'Starting';
    },
```

**Step 2: Update badge text in card template**

In `index.html:785`, change:

```html
                      x-text="healthLabel(agent.health_status)"
```

From the current:
```html
                      x-text="agent.health_status || 'unknown'"
```

**Step 3: Update badge text in agent detail view**

Find the detail health badge (around line 1336-1344) and apply the same `healthLabel()` call.

**Step 4: Commit**

```bash
git add src/dashboard/templates/index.html src/dashboard/static/js/app.js
git commit -m "feat(dashboard): rename health badges to Online/Degraded/Offline/Starting"
```

---

### Task 5: HTML/JS — Heartbeat row with countdown on agent cards

**Files:**
- Modify: `src/dashboard/templates/index.html:791-822` (stats grid)
- Modify: `src/dashboard/static/js/app.js` (countdown timer logic)

**Step 1: Add countdown state and timer to JS**

Add to the Alpine data object (near `agentStates`):

```javascript
    _heartbeatCountdowns: {},  // { agentId: "12m 34s" }
    _heartbeatTimer: null,
```

Add `startHeartbeatTimer` and `stopHeartbeatTimer` methods:

```javascript
    startHeartbeatTimer() {
      if (this._heartbeatTimer) return;
      this._heartbeatTimer = setInterval(() => {
        const updated = {};
        for (const agent of this.agents) {
          if (!agent.heartbeat_next_run || !agent.heartbeat_enabled) continue;
          const nextRun = new Date(agent.heartbeat_next_run).getTime();
          const diff = nextRun - Date.now();
          if (diff <= 0) {
            updated[agent.id] = 'running...';
          } else {
            const mins = Math.floor(diff / 60000);
            const secs = Math.floor((diff % 60000) / 1000);
            updated[agent.id] = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
          }
        }
        this._heartbeatCountdowns = updated;
      }, 1000);
    },

    stopHeartbeatTimer() {
      if (this._heartbeatTimer) { clearInterval(this._heartbeatTimer); this._heartbeatTimer = null; }
    },
```

Call `startHeartbeatTimer()` inside `init()` after `fetchAgents()`, and `stopHeartbeatTimer()` in `destroy()`.

**Step 2: Add heartbeat row to agent card stats grid**

In `index.html`, after the Activity stat-row (after line 806) and before the Cost stat-row (line 808), add:

```html
                <div class="stat-row" x-show="agent.heartbeat_schedule">
                  <span class="stat-label">
                    <svg fill="currentColor" viewBox="0 0 24 24" stroke="none" class="w-3 h-3"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>
                    Heartbeat
                  </span>
                  <span class="text-[11px] font-medium"
                    :class="agent.heartbeat_enabled ? 'text-pink-400' : 'text-gray-600'">
                    <template x-if="agent.heartbeat_enabled">
                      <span>
                        <span x-text="_heartbeatCountdowns[agent.id] || '...'"></span>
                        <span class="text-gray-600 font-normal">&middot;</span>
                        <span class="text-gray-500 font-normal text-[10px]" x-text="agent.heartbeat_schedule"></span>
                      </span>
                    </template>
                    <template x-if="!agent.heartbeat_enabled">
                      <span>paused</span>
                    </template>
                  </span>
                </div>
```

**Step 3: Commit**

```bash
git add src/dashboard/templates/index.html src/dashboard/static/js/app.js
git commit -m "feat(dashboard): add heartbeat countdown row to agent cards"
```

---

### Task 6: HTML — Cron button visual fixes + heartbeat delete protection

**Files:**
- Modify: `src/dashboard/templates/index.html:1891-1902` (agent detail cron buttons)
- Modify: `src/dashboard/templates/index.html:2107-2113` (global cron table buttons)

**Step 1: Fix button styling in agent detail automations**

For all Run/Pause/Resume buttons at lines 1891-1899, change:
- `text-[10px]` → `text-[11px]`
- `bg-gray-800` → `bg-gray-700`
- `text-gray-400` → `text-gray-300`

For Run button specifically, add an indigo tint: `bg-indigo-900/30 hover:bg-indigo-800/40 text-indigo-300 hover:text-indigo-200`

**Step 2: Hide Delete for heartbeat jobs**

At line 1900 (agent detail Delete button), add `x-show="!job.heartbeat"`:

```html
                        <button x-show="!job.heartbeat" @click="deleteCronJob(job.id)"
                          class="text-[11px] px-1.5 py-0.5 rounded bg-gray-700 hover:bg-red-900/30 text-gray-300 hover:text-red-400 transition-colors">Del</button>
```

**Step 3: Apply same changes to global cron table**

At lines 2107-2113 (System tab cron table buttons), apply identical styling changes and add `x-show="!job.heartbeat"` to the Delete button.

**Step 4: Commit**

```bash
git add src/dashboard/templates/index.html
git commit -m "feat(dashboard): improve cron button contrast, protect heartbeat from deletion"
```

---

### Task 7: Final review and integration test

**Step 1: Run full test suite**

```bash
python -m pytest tests/test_dashboard.py -x -v
```

Expected: All pass (including new heartbeat tests, with pre-existing agent count failures still known).

**Step 2: Visual verification (if dashboard is running)**

- Open dashboard, verify agent cards show "Online" instead of "healthy"
- Verify heartbeat countdown ticks in agent card stats
- Verify chat messages show relative timestamps
- Verify cron buttons are visually more prominent
- Verify heartbeat jobs have no Delete button
- Verify regular cron jobs still have Delete button

**Step 3: Final commit (if any tweaks needed)**

```bash
git add -A && git commit -m "fix(dashboard): integration polish"
```

**Step 4: Create PR**

```bash
gh pr create --title "Dashboard UX: timestamps, heartbeat countdown, health badges, cron fixes" --body "..."
```
