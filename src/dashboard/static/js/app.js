/**
 * OpenLegion Dashboard — Alpine.js application.
 *
 * Six panels: Fleet, Events, Agent Detail, Blackboard, Costs, Traces.
 * Real-time updates via WebSocket + periodic REST polling.
 */
function dashboard() {
  return {
    // Navigation
    activeTab: 'fleet',
    tabs: [
      { id: 'fleet', label: 'Fleet' },
      { id: 'events', label: 'Events' },
      { id: 'blackboard', label: 'Blackboard' },
      { id: 'costs', label: 'Costs' },
      { id: 'traces', label: 'Traces' },
    ],
    connected: false,
    loading: true,
    lastRefresh: 0,
    unreadEvents: 0,
    _lastSeenEventCount: 0,

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
      'blackboard_write', 'cost_update', 'health_change',
    ],

    // Agent detail
    selectedAgent: null,
    agentDetail: null,
    agentEvents: [],

    // Blackboard
    bbEntries: [],
    bbPrefix: '',
    bbHighlights: new Set(),
    bbLoading: false,

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

    // WebSocket
    _ws: null,
    _refreshInterval: null,
    _fleetDebounce: null,

    // ── Computed ───────────────────────────────────────────

    get filteredEvents() {
      if (this.eventFilters.size === 0) return this.events;
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
      if (this._costDebounce) clearTimeout(this._costDebounce);
      if (this._fleetDebounce) clearTimeout(this._fleetDebounce);
      if (this.costChart) this.costChart.destroy();
      Object.values(this._stateTimers).forEach(clearTimeout);
    },

    // ── Tab switching ─────────────────────────────────────

    switchTab(tab) {
      this.activeTab = tab;
      if (tab === 'events') {
        this.unreadEvents = 0;
        this._lastSeenEventCount = this.events.length;
      }
      if (tab === 'blackboard') this.fetchBlackboard();
      if (tab === 'costs') this.fetchCosts();
      if (tab === 'traces') this.fetchTraces();
    },

    // ── WebSocket event handler ───────────────────────────

    onWsEvent(evt) {
      // Append to event feed (newest first, cap at 500)
      this.events.unshift(evt);
      if (this.events.length > 500) this.events.splice(500);

      // Track unread events when not on Events tab
      if (this.activeTab !== 'events') {
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

      // Live-update fleet on cost/health changes (debounced)
      if (evt.type === 'cost_update' || evt.type === 'health_change') {
        this._debouncedFleetRefresh();
      }

      // Highlight blackboard writes
      if (evt.type === 'blackboard_write' && evt.data && evt.data.key) {
        this.bbHighlights.add(evt.data.key);
        setTimeout(() => this.bbHighlights.delete(evt.data.key), 5000);
        if (this.activeTab === 'blackboard') this.fetchBlackboard();
      }

      // Debounced cost panel refresh on cost updates
      if (evt.type === 'cost_update' && this.activeTab === 'costs') {
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
      } catch (_) {}
      this.loading = false;
    },

    async fetchAgentDetail(agentId) {
      this.agentDetail = null;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}`);
        if (resp.ok) this.agentDetail = await resp.json();
      } catch (_) {}
    },

    async fetchBlackboard() {
      this.bbLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/blackboard?prefix=${encodeURIComponent(this.bbPrefix)}`);
        if (resp.ok) this.bbEntries = (await resp.json()).entries;
      } catch (_) {}
      this.bbLoading = false;
    },

    async fetchCosts() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/costs?period=${this.costPeriod}`);
        if (resp.ok) {
          this.costData = await resp.json();
          this.$nextTick(() => this.renderCostChart());
        }
      } catch (_) {}
    },

    async fetchTraces() {
      this.tracesLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/traces?limit=50`);
        if (resp.ok) this.traces = (await resp.json()).traces;
      } catch (_) {}
      this.tracesLoading = false;
    },

    async fetchTraceDetail(traceId) {
      this.selectedTraceId = traceId;
      this.traceDetail = null;
      try {
        const resp = await fetch(`${window.__config.apiBase}/traces/${traceId}`);
        if (resp.ok) this.traceDetail = await resp.json();
      } catch (_) {}
    },

    // ── Agent drill-down ──────────────────────────────────

    drillDown(agentId) {
      this.selectedAgent = agentId;
      this.agentEvents = this.events.filter(e => e.agent === agentId).slice(0, 100);
      this.fetchAgentDetail(agentId);
      this.activeTab = 'agent-detail';
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
        cost_update: 'text-rose-400',
        health_change: 'text-red-400',
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
        cost_update: 'bg-rose-400',
        health_change: 'bg-red-400',
        chat: 'bg-green-400',
      };
      return map[type] || 'bg-gray-400';
    },

    eventSummary(evt) {
      const d = evt.data || {};
      switch (evt.type) {
        case 'llm_call':
          return `model=${d.model || '?'} tokens=${d.total_tokens || '?'}`;
        case 'tool_start':
          return `${d.tool || d.name || '?'}(${(d.preview || '').substring(0, 60)})`;
        case 'tool_result':
          return `${d.tool || d.name || '?'} \u2192 ${(d.preview || d.result || '').substring(0, 60)}`;
        case 'message_sent':
        case 'message_received':
          return (d.message || '').substring(0, 80);
        case 'cost_update':
          return `$${(d.cost_usd || 0).toFixed(4)} (${d.model || '?'})`;
        case 'health_change':
          return `${d.previous || '?'} \u2192 ${d.current || '?'}`;
        case 'blackboard_write':
          return d.key || '';
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
