/**
 * OpenLegion Dashboard — Alpine.js application.
 *
 * Six panels: Agents, Activity, Blackboard, Costs, Automation, System.
 * Real-time updates via WebSocket + periodic REST polling.
 */
function dashboard() {
  return {
    // Navigation
    activeTab: 'fleet',
    tabs: [
      { id: 'fleet', label: 'Agents' },
      { id: 'activity', label: 'Activity' },
      { id: 'blackboard', label: 'Blackboard' },
      { id: 'costs', label: 'Costs' },
      { id: 'automation', label: 'Automation' },
      { id: 'system', label: 'System' },
    ],
    connected: false,
    loading: true,
    lastRefresh: 0,
    unreadEvents: 0,
    _lastSeenEventCount: 0,
    toastMessage: '',
    _toastTimer: null,

    // Fleet
    agents: [],
    agentStates: {},
    _stateTimers: {},

    // Events
    events: [],
    eventFilters: new Set(),
    eventTypes: [
      'agent_state', 'message_sent', 'message_received',
      'tool_start', 'tool_result', 'llm_call',
      'blackboard_write', 'health_change',
    ],

    // Agent detail
    selectedAgent: null,
    agentDetail: null,
    agentEvents: [],

    // Agent config (V2)
    agentConfigs: {},
    editingAgent: null,
    editForm: {},
    availableModels: [],
    availableBrowsers: [],

    // Add agent
    addAgentMode: false,
    addAgentForm: { name: '', role: '', model: '', browser_backend: '' },
    addAgentLoading: false,

    // Blackboard
    bbEntries: [],
    bbPrefix: '',
    bbHighlights: new Set(),
    bbLoading: false,
    bbWriteMode: false,
    bbNewKey: '',
    bbNewValue: '{}',
    bbWriterFilter: '',

    // Costs
    costData: {},
    costPeriod: 'today',
    costChart: null,
    _costDebounce: null,

    // Traces
    traces: [],
    selectedTraceId: null,
    traceDetail: null,
    tracesLoading: false,

    // Queues (V2)
    queueStatus: {},

    // Cron / Automation
    cronJobs: [],
    editingCronJob: null,
    cronEditSchedule: '',

    // Settings (V2)
    settingsData: null,

    // Chat (persistent per agent)
    chatAgent: null,
    chatMessage: '',
    chatHistories: {},
    chatLoading: false,

    // Broadcast
    broadcastMessage: '',
    broadcastLoading: false,
    broadcastResults: null,

    // Credentials
    credService: '',
    credKey: '',

    // WebSocket
    _ws: null,
    _refreshInterval: null,
    _fleetDebounce: null,
    _queueInterval: null,
    _cronInterval: null,

    // ── Computed ───────────────────────────────────────────

    get filteredEvents() {
      if (this.eventFilters.size === 0) {
        return this.events.filter(e =>
          !(e.type === 'agent_state' && e.data?.state === 'registered'));
      }
      return this.events.filter(e => this.eventFilters.has(e.type));
    },

    get fleetTotalCost() {
      return this.agents.reduce((sum, a) => sum + (a.daily_cost || 0), 0);
    },

    get fleetTotalTokens() {
      return this.agents.reduce((sum, a) => sum + (a.daily_tokens || 0), 0);
    },

    get costTotal() {
      return (this.costData.agents || []).reduce((sum, a) => sum + (a.cost || 0), 0);
    },

    get filteredBbEntries() {
      if (!this.bbWriterFilter) return this.bbEntries;
      return this.bbEntries.filter(e => e.written_by === this.bbWriterFilter);
    },

    get bbWriters() {
      return [...new Set(this.bbEntries.map(e => e.written_by))].sort();
    },

    // ── Lifecycle ─────────────────────────────────────────

    init() {
      const cfg = window.__config || {};
      this._ws = new DashboardWebSocket(cfg.wsUrl, {
        onEvent: (evt) => this.onWsEvent(evt),
        onConnect: () => { this.connected = true; },
        onDisconnect: () => { this.connected = false; },
      });
      this._ws.connect();

      this.fetchAgents();
      this._refreshInterval = setInterval(() => this.fetchAgents(), 15000);
    },

    destroy() {
      if (this._ws) this._ws.disconnect();
      if (this._refreshInterval) clearInterval(this._refreshInterval);
      if (this._queueInterval) clearInterval(this._queueInterval);
      if (this._cronInterval) clearInterval(this._cronInterval);
      if (this._costDebounce) clearTimeout(this._costDebounce);
      if (this._fleetDebounce) clearTimeout(this._fleetDebounce);
      if (this._toastTimer) clearTimeout(this._toastTimer);
      if (this.costChart) this.costChart.destroy();
      Object.values(this._stateTimers).forEach(clearTimeout);
    },

    // ── Tab switching ─────────────────────────────────────

    switchTab(tab) {
      this.activeTab = tab;
      // Clear tab-specific auto-refresh intervals
      if (this._queueInterval) { clearInterval(this._queueInterval); this._queueInterval = null; }
      if (this._cronInterval) { clearInterval(this._cronInterval); this._cronInterval = null; }
      if (tab === 'activity') {
        this.unreadEvents = 0;
        this._lastSeenEventCount = this.events.length;
      }
      if (tab === 'blackboard') this.fetchBlackboard();
      if (tab === 'costs') this.fetchCosts();
      if (tab === 'fleet') {
        this.fetchAgents();
        this.fetchQueues();
        this.fetchSettings();
        this.agents.forEach(a => this.fetchAgentConfig(a.id));
        this._queueInterval = setInterval(() => this.fetchQueues(), 5000);
      }
      if (tab === 'automation') {
        this.fetchCronJobs();
        this._cronInterval = setInterval(() => this.fetchCronJobs(), 10000);
      }
      if (tab === 'system') {
        this.fetchTraces();
        this.fetchSettings();
      }
    },

    // ── Toast helper ──────────────────────────────────────

    showToast(msg) {
      this.toastMessage = msg;
      if (this._toastTimer) clearTimeout(this._toastTimer);
      this._toastTimer = setTimeout(() => { this.toastMessage = ''; }, 3000);
    },

    // ── WebSocket event handler ───────────────────────────

    onWsEvent(evt) {
      // Append to event feed (newest first, cap at 500)
      this.events.unshift(evt);
      if (this.events.length > 500) this.events.splice(500);

      // Track unread events when not on Activity tab
      if (this.activeTab !== 'activity') {
        this.unreadEvents++;
      }

      // Update agent activity state
      const agent = evt.agent;
      if (agent) {
        this._updateAgentState(agent, evt.type);

        // Feed agent-detail events
        if (this.selectedAgent === agent) {
          this.agentEvents.unshift(evt);
          if (this.agentEvents.length > 100) this.agentEvents.splice(100);
        }
      }

      // Live-update fleet on llm_call/health changes (debounced)
      if (evt.type === 'llm_call' || evt.type === 'health_change') {
        this._debouncedFleetRefresh();
      }

      // Highlight blackboard writes
      if (evt.type === 'blackboard_write' && evt.data && evt.data.key) {
        this.bbHighlights.add(evt.data.key);
        setTimeout(() => this.bbHighlights.delete(evt.data.key), 5000);
        if (this.activeTab === 'blackboard') this.fetchBlackboard();
      }

      // Debounced cost panel refresh on llm_call events
      if (evt.type === 'llm_call' && this.activeTab === 'costs') {
        if (this._costDebounce) clearTimeout(this._costDebounce);
        this._costDebounce = setTimeout(() => this.fetchCosts(), 2000);
      }
    },

    _updateAgentState(agent, eventType) {
      const stateMap = {
        llm_call: 'thinking',
        tool_start: 'tool',
        tool_result: 'thinking',
        message_sent: 'thinking',
        message_received: 'thinking',
      };

      const newState = stateMap[eventType];
      if (newState) {
        this.agentStates[agent] = newState;

        // Reset to idle after 30s of no events
        if (this._stateTimers[agent]) clearTimeout(this._stateTimers[agent]);
        this._stateTimers[agent] = setTimeout(() => {
          this.agentStates[agent] = 'idle';
        }, 30000);
      }
    },

    _debouncedFleetRefresh() {
      if (this._fleetDebounce) clearTimeout(this._fleetDebounce);
      this._fleetDebounce = setTimeout(() => this.fetchAgents(), 3000);
    },

    toggleEventFilter(type) {
      if (this.eventFilters.has(type)) {
        this.eventFilters.delete(type);
      } else {
        this.eventFilters.add(type);
      }
    },

    // ── REST fetchers ─────────────────────────────────────

    async fetchAgents() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents`);
        if (resp.ok) {
          this.agents = (await resp.json()).agents;
          this.lastRefresh = Date.now() / 1000;
        }
      } catch (e) { console.warn('fetchAgents failed:', e); }
      this.loading = false;
    },

    async fetchAgentDetail(agentId) {
      this.agentDetail = null;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}`);
        if (resp.ok) this.agentDetail = await resp.json();
      } catch (e) { console.warn('fetchAgentDetail failed:', e); }
    },

    async fetchBlackboard() {
      this.bbLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/blackboard?prefix=${encodeURIComponent(this.bbPrefix)}`);
        if (resp.ok) this.bbEntries = (await resp.json()).entries;
      } catch (e) { console.warn('fetchBlackboard failed:', e); }
      this.bbLoading = false;
    },

    async fetchCosts() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/costs?period=${this.costPeriod}`);
        if (resp.ok) {
          this.costData = await resp.json();
          this.$nextTick(() => this.renderCostChart());
        }
      } catch (e) { console.warn('fetchCosts failed:', e); }
    },

    async fetchTraces() {
      this.tracesLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/traces?limit=50`);
        if (resp.ok) this.traces = (await resp.json()).traces;
      } catch (e) { console.warn('fetchTraces failed:', e); }
      this.tracesLoading = false;
    },

    async fetchTraceDetail(traceId) {
      this.selectedTraceId = traceId;
      this.traceDetail = null;
      try {
        const resp = await fetch(`${window.__config.apiBase}/traces/${traceId}`);
        if (resp.ok) this.traceDetail = await resp.json();
      } catch (e) { console.warn('fetchTraceDetail failed:', e); }
    },

    // ── V2: Agent config fetchers ─────────────────────────

    async fetchAgentConfig(agentId) {
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/config`);
        if (resp.ok) {
          const cfg = await resp.json();
          this.agentConfigs[agentId] = cfg;
          return cfg;
        }
      } catch (e) { console.warn('fetchAgentConfig failed:', e); }
      return null;
    },

    startEditAgent(agentId) {
      const cfg = this.agentConfigs[agentId];
      if (!cfg) return;
      this.editingAgent = agentId;
      this.editForm = {
        model: cfg.model || '',
        browser_backend: cfg.browser_backend || 'basic',
        role: cfg.role || '',
        system_prompt: cfg.system_prompt || '',
        budget_daily: cfg.budget?.daily_usd || '',
      };
    },

    cancelEdit() {
      this.editingAgent = null;
      this.editForm = {};
    },

    async saveAgentConfig(agentId) {
      const body = {};
      const cfg = this.agentConfigs[agentId] || {};
      if (this.editForm.model && this.editForm.model !== cfg.model) body.model = this.editForm.model;
      if (this.editForm.browser_backend && this.editForm.browser_backend !== cfg.browser_backend) body.browser_backend = this.editForm.browser_backend;
      if (this.editForm.role !== undefined && this.editForm.role !== cfg.role) body.role = this.editForm.role;
      if (this.editForm.system_prompt !== undefined && this.editForm.system_prompt !== cfg.system_prompt) body.system_prompt = this.editForm.system_prompt;
      if (this.editForm.budget_daily && parseFloat(this.editForm.budget_daily) > 0) {
        body.budget = { daily_usd: parseFloat(this.editForm.budget_daily) };
      }
      if (Object.keys(body).length === 0) {
        this.cancelEdit();
        return;
      }
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/config`, {
          method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
        });
        if (resp.ok) {
          const result = await resp.json();
          this.showToast(`Updated: ${result.updated.join(', ')}${result.restart_required ? ' (restart required)' : ''}`);
          await this.fetchAgentConfig(agentId);
        } else {
          const err = await resp.json();
          this.showToast(`Error: ${err.detail || 'Update failed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      this.cancelEdit();
    },

    async restartAgent(agentId) {
      if (!confirm(`Restart agent "${agentId}"? This will interrupt any active work.`)) return;
      this.showToast(`Restarting ${agentId}...`);
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/restart`, { method: 'POST' });
        if (resp.ok) {
          const data = await resp.json();
          this.showToast(data.ready ? `${agentId} restarted and ready` : `${agentId} restarted (not ready)`);
          this.fetchAgents();
        } else {
          this.showToast(`Restart failed`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
    },

    async updateBudget(agentId, dailyUsd) {
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/budget`, {
          method: 'PUT', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ daily_usd: parseFloat(dailyUsd) }),
        });
        if (resp.ok) this.showToast(`Budget updated for ${agentId}`);
      } catch (e) { console.warn('updateBudget failed:', e); }
    },

    // ── Add / Remove agents ──────────────────────────────

    async addAgent() {
      const f = this.addAgentForm;
      if (!f.name.trim()) { this.showToast('Name is required'); return; }
      this.addAgentLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            name: f.name.trim(),
            role: f.role.trim(),
            model: f.model,
            browser_backend: f.browser_backend,
          }),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.showToast(data.ready ? `${data.agent} added and ready` : `${data.agent} added (starting)`);
          this.addAgentMode = false;
          this.addAgentForm = { name: '', role: '', model: '', browser_backend: '' };
          this.fetchAgents();
        } else {
          const err = await resp.json();
          this.showToast(`Error: ${err.detail || 'Failed to add agent'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      this.addAgentLoading = false;
    },

    async removeAgent(agentId) {
      if (!confirm(`Remove agent "${agentId}"? This will stop the container and remove its config.`)) return;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}`, { method: 'DELETE' });
        if (resp.ok) {
          this.showToast(`${agentId} removed`);
          this.fetchAgents();
        } else {
          const err = await resp.json();
          this.showToast(`Error: ${err.detail || 'Failed to remove agent'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
    },

    // ── V2: Blackboard write/delete ───────────────────────

    async writeBlackboard() {
      if (!this.bbNewKey.trim()) return;
      let value;
      try { value = JSON.parse(this.bbNewValue); } catch (_) { this.showToast('Invalid JSON'); return; }
      try {
        const resp = await fetch(`${window.__config.apiBase}/blackboard/${this.bbNewKey}`, {
          method: 'PUT', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ value }),
        });
        if (resp.ok) {
          this.showToast(`Written: ${this.bbNewKey}`);
          this.bbNewKey = ''; this.bbNewValue = '{}'; this.bbWriteMode = false;
          this.fetchBlackboard();
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
    },

    async deleteBlackboard(key) {
      if (!confirm(`Delete blackboard key "${key}"?`)) return;
      try {
        const resp = await fetch(`${window.__config.apiBase}/blackboard/${key}`, { method: 'DELETE' });
        if (resp.ok) { this.showToast(`Deleted: ${key}`); this.fetchBlackboard(); }
        else { const err = await resp.json(); this.showToast(`Error: ${err.detail}`); }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
    },

    // ── V2: Queue fetcher ─────────────────────────────────

    async fetchQueues() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/queues`);
        if (resp.ok) this.queueStatus = (await resp.json()).queues;
      } catch (e) { console.warn('fetchQueues failed:', e); }
    },

    // ── V2: Cron fetchers ─────────────────────────────────

    async fetchCronJobs() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/cron`);
        if (resp.ok) this.cronJobs = (await resp.json()).jobs;
      } catch (e) { console.warn('fetchCronJobs failed:', e); }
    },

    async runCronJob(jobId) {
      try {
        const resp = await fetch(`${window.__config.apiBase}/cron/${jobId}/run`, { method: 'POST' });
        if (resp.ok) this.showToast(`Job ${jobId} triggered`);
        this.fetchCronJobs();
      } catch (e) { console.warn('runCronJob failed:', e); }
    },

    async pauseCronJob(jobId) {
      try {
        await fetch(`${window.__config.apiBase}/cron/${jobId}/pause`, { method: 'POST' });
        this.showToast(`Job ${jobId} paused`);
        this.fetchCronJobs();
      } catch (e) { console.warn('pauseCronJob failed:', e); }
    },

    async resumeCronJob(jobId) {
      try {
        await fetch(`${window.__config.apiBase}/cron/${jobId}/resume`, { method: 'POST' });
        this.showToast(`Job ${jobId} resumed`);
        this.fetchCronJobs();
      } catch (e) { console.warn('resumeCronJob failed:', e); }
    },

    // ── Cron inline editing ─────────────────────────────

    editCronJob(job) {
      this.editingCronJob = job.id;
      this.cronEditSchedule = job.schedule;
    },

    cancelCronEdit() {
      this.editingCronJob = null;
      this.cronEditSchedule = '';
    },

    async saveCronEdit(jobId) {
      if (!this.cronEditSchedule.trim()) { this.cancelCronEdit(); return; }
      try {
        const resp = await fetch(`${window.__config.apiBase}/cron/${jobId}`, {
          method: 'PUT', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ schedule: this.cronEditSchedule.trim() }),
        });
        if (resp.ok) {
          this.showToast(`Schedule updated for ${jobId}`);
          this.fetchCronJobs();
        } else {
          const err = await resp.json();
          this.showToast(`Error: ${err.detail || 'Update failed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      this.cancelCronEdit();
    },

    // ── Settings fetcher ─────────────────────────────────

    async fetchSettings() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/settings`);
        if (resp.ok) {
          this.settingsData = await resp.json();
          // Extract models and browsers for agent edit forms
          if (this.settingsData.provider_models) {
            this.availableModels = Object.values(this.settingsData.provider_models).flat();
          }
          if (this.settingsData.browser_backends) {
            this.availableBrowsers = this.settingsData.browser_backends.map(b => b.name);
          }
        }
      } catch (e) { console.warn('fetchSettings failed:', e); }
    },

    // ── Chat with agent (persistent per agent) ───────────

    openChat(agentId) {
      this.chatAgent = agentId;
      this.chatMessage = '';
    },

    closeChat() {
      this.chatAgent = null;
      this.chatMessage = '';
    },

    async sendChat() {
      if (!this.chatMessage.trim() || !this.chatAgent) return;
      const msg = this.chatMessage.trim();
      if (!this.chatHistories[this.chatAgent]) this.chatHistories[this.chatAgent] = [];
      this.chatHistories[this.chatAgent].push({ role: 'user', content: msg });
      this.chatMessage = '';
      this.chatLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${this.chatAgent}/chat`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ message: msg }),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.chatHistories[this.chatAgent].push({ role: 'agent', content: data.response });
        } else {
          const err = await resp.json();
          this.chatHistories[this.chatAgent].push({ role: 'error', content: err.detail || 'Failed' });
        }
      } catch (e) {
        this.chatHistories[this.chatAgent].push({ role: 'error', content: e.message });
      }
      this.chatLoading = false;
    },

    // ── Broadcast ────────────────────────────────────────

    async sendBroadcast() {
      if (!this.broadcastMessage.trim()) return;
      this.broadcastLoading = true;
      this.broadcastResults = null;
      try {
        const resp = await fetch(`${window.__config.apiBase}/broadcast`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ message: this.broadcastMessage.trim() }),
        });
        if (resp.ok) {
          this.broadcastResults = (await resp.json()).responses;
          this.broadcastMessage = '';
          this.showToast('Broadcast sent');
        } else {
          this.showToast('Broadcast failed');
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      this.broadcastLoading = false;
    },

    // ── Reset ────────────────────────────────────────────

    async resetAgent(agentId) {
      if (!confirm(`Reset conversation for "${agentId}"? This clears their chat history.`)) return;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/reset`, { method: 'POST' });
        if (resp.ok) this.showToast(`${agentId} conversation reset`);
        else this.showToast('Reset failed');
      } catch (e) { this.showToast(`Error: ${e.message}`); }
    },

    // ── Credentials ──────────────────────────────────────

    async addCredential() {
      if (!this.credService.trim() || !this.credKey.trim()) return;
      try {
        const resp = await fetch(`${window.__config.apiBase}/credentials`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ service: this.credService.trim(), key: this.credKey.trim() }),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.showToast(`Credential stored: ${data.service}`);
          this.credService = '';
          this.credKey = '';
          this.fetchSettings();
        } else {
          const err = await resp.json();
          this.showToast(`Error: ${err.detail}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
    },

    // ── Agent drill-down ──────────────────────────────────

    drillDown(agentId) {
      this.selectedAgent = agentId;
      this.agentEvents = this.events.filter(e => e.agent === agentId).slice(0, 100);
      this.fetchAgentDetail(agentId);
      this.activeTab = 'agent-detail';
    },

    // ── V2: Agents panel drill-down ───────────────────────

    async openAgentConfig(agentId) {
      await this.fetchAgentConfig(agentId);
      this.selectedAgent = agentId;
      this.startEditAgent(agentId);
    },

    // ── Chart.js rendering ────────────────────────────────

    renderCostChart() {
      const canvas = document.getElementById('costChart');
      if (!canvas) return;
      if (this.costChart) this.costChart.destroy();

      const agents = (this.costData.agents || []);
      if (agents.length === 0) {
        this.costChart = null;
        return;
      }

      const labels = agents.map(a => a.agent);
      const costs = agents.map(a => a.cost);
      const tokens = agents.map(a => a.tokens);

      this.costChart = new Chart(canvas, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            {
              label: 'Cost (USD)',
              data: costs,
              backgroundColor: 'rgba(99, 102, 241, 0.4)',
              borderColor: 'rgb(99, 102, 241)',
              borderWidth: 1,
              borderRadius: 4,
              yAxisID: 'y',
            },
            {
              label: 'Tokens',
              data: tokens,
              backgroundColor: 'rgba(168, 85, 247, 0.25)',
              borderColor: 'rgb(168, 85, 247)',
              borderWidth: 1,
              borderRadius: 4,
              yAxisID: 'y1',
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: {
              labels: { color: '#6b7280', usePointStyle: true, pointStyleWidth: 8, font: { size: 11 } },
            },
          },
          scales: {
            x: {
              ticks: { color: '#4b5563', font: { size: 11 } },
              grid: { color: 'rgba(31, 41, 55, 0.5)' },
            },
            y: {
              position: 'left',
              title: { display: true, text: 'Cost (USD)', color: '#6b7280', font: { size: 11 } },
              ticks: { color: '#4b5563', font: { size: 10 } },
              grid: { color: 'rgba(31, 41, 55, 0.5)' },
            },
            y1: {
              position: 'right',
              title: { display: true, text: 'Tokens', color: '#6b7280', font: { size: 11 } },
              ticks: { color: '#4b5563', font: { size: 10 } },
              grid: { drawOnChartArea: false },
            },
          },
        },
      });
    },

    // ── Formatting helpers ────────────────────────────────

    eventColor(type) {
      const map = {
        agent_state: 'text-blue-400',
        message_sent: 'text-green-400',
        message_received: 'text-green-400',
        tool_start: 'text-amber-400',
        tool_result: 'text-amber-400',
        llm_call: 'text-purple-400',
        blackboard_write: 'text-cyan-400',
        health_change: 'text-red-400',
        // V2 trace event types
        chat: 'text-green-400',
        chat_response: 'text-green-300',
        message_route: 'text-teal-400',
        pubsub_publish: 'text-sky-400',
        agent_spawn: 'text-indigo-400',
        workflow_step_start: 'text-orange-400',
        workflow_step_end: 'text-orange-300',
        lane_start: 'text-yellow-400',
        lane_complete: 'text-yellow-300',
        cron_trigger: 'text-pink-400',
        llm_stream: 'text-purple-300',
      };
      return map[type] || 'text-gray-400';
    },

    eventBgColor(type) {
      const map = {
        agent_state: 'bg-blue-400',
        message_sent: 'bg-green-400',
        message_received: 'bg-green-400',
        tool_start: 'bg-amber-400',
        tool_result: 'bg-amber-400',
        llm_call: 'bg-purple-400',
        blackboard_write: 'bg-cyan-400',
        health_change: 'bg-red-400',
        chat: 'bg-green-400',
        chat_response: 'bg-green-300',
        message_route: 'bg-teal-400',
        pubsub_publish: 'bg-sky-400',
        agent_spawn: 'bg-indigo-400',
        workflow_step_start: 'bg-orange-400',
        workflow_step_end: 'bg-orange-300',
        lane_start: 'bg-yellow-400',
        lane_complete: 'bg-yellow-300',
        cron_trigger: 'bg-pink-400',
        llm_stream: 'bg-purple-300',
      };
      return map[type] || 'bg-gray-400';
    },

    eventSummary(evt) {
      const d = evt.data || {};
      switch (evt.type) {
        case 'llm_call': {
          const model = (d.model || '?').split('/').pop();
          const tokens = (d.total_tokens || d.tokens_used || 0).toLocaleString();
          const cost = d.cost_usd != null ? ` \u00b7 $${d.cost_usd.toFixed(4)}` : '';
          const dur = d.duration_ms ? ` \u00b7 ${d.duration_ms}ms` : '';
          const stream = d.streaming ? ' (stream)' : '';
          return `${model} \u00b7 ${tokens} tok${cost}${dur}${stream}`;
        }
        case 'tool_start':
          return `${d.tool || d.name || '?'}(${(d.preview || '').substring(0, 60)})`;
        case 'tool_result':
          return `${d.tool || d.name || '?'} \u2192 ${(d.preview || d.result || d.output || '').substring(0, 60) || 'done'}`;
        case 'message_sent':
          return `\u2192 ${(d.message || '').substring(0, 70)}`;
        case 'message_received':
          return `\u2190 ${(d.message || '').substring(0, 70)}`;
        case 'health_change':
          return `${d.previous || '?'} \u2192 ${d.current || '?'}${d.failures ? ` (${d.failures} failures)` : ''}`;
        case 'blackboard_write':
          return [d.key, d.version && `v${d.version}`, d.written_by && `by ${d.written_by}`].filter(Boolean).join(' \u00b7 ');
        case 'agent_state': {
          const s = d.state || '?';
          if (s === 'registered' && d.capabilities) return `registered (${Array.isArray(d.capabilities) ? d.capabilities.length : '?'} tools)`;
          if (s === 'added') return d.ready ? 'added (ready)' : 'added (starting)';
          if (s === 'removed') return `removed${d.reason ? ` (${d.reason})` : ''}`;
          if (s === 'spawned') return d.ready ? 'spawned (ready)' : 'spawned (starting)';
          return s;
        }
        default:
          return JSON.stringify(d).substring(0, 80);
      }
    },

    timeAgo(ts) {
      if (!ts) return '';
      const now = Date.now() / 1000;
      const diff = now - ts;
      if (diff < 0) return 'just now';
      if (diff < 5) return 'just now';
      if (diff < 60) return Math.floor(diff) + 's ago';
      if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
      if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
      return Math.floor(diff / 86400) + 'd ago';
    },

    formatCost(usd) {
      if (usd === 0 || usd == null) return '$0.00';
      if (usd < 0.01) return '$' + usd.toFixed(4);
      return '$' + usd.toFixed(2);
    },

    truncateJson(value) {
      const s = JSON.stringify(value);
      if (s.length <= 120) return s;
      return s.substring(0, 117) + '\u2026';
    },

    fullJson(value) {
      return JSON.stringify(value, null, 2);
    },

    waterfall(evt, allEvents) {
      if (!allEvents || allEvents.length < 2) return '';
      const minTs = allEvents[0].timestamp;
      const maxTs = allEvents[allEvents.length - 1].timestamp;
      const span = maxTs - minTs || 1;
      const left = ((evt.timestamp - minTs) / span) * 100;
      const width = Math.max(2, (evt.duration_ms / 1000 / span) * 100);
      return `left:${left.toFixed(1)}%;width:${Math.min(width, 100 - left).toFixed(1)}%`;
    },
  };
}
