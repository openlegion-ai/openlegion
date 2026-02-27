/**
 * OpenLegion Dashboard — Alpine.js application.
 *
 * Three panels: Agents, Activity (Traces / Events), System (Costs / Blackboard / Automation / Credentials).
 * Real-time updates via WebSocket + periodic REST polling.
 */
const _IDENTITY_TABS = [
  { id: 'config', label: 'Config', file: null, access: 'user', desc: 'Model, role, and daily budget.' },
  { id: 'identity', label: 'Identity', file: null, access: 'user', desc: 'Agent personality and instructions.' },
  { id: 'memory', label: 'Memory', file: null, access: 'agent', desc: 'Long-term memory and autonomous heartbeat rules.' },
  { id: 'logs', label: 'Logs', file: null, access: 'auto', desc: 'Activity logs and learned corrections.' },
];

const _IDENTITY_FILE_MAP = {
  identity: [
    { file: 'SOUL.md', label: 'Soul', cap: 4000, access: 'user', desc: 'Core personality and behavioral guidelines.' },
    { file: 'AGENTS.md', label: 'Instructions', cap: 8000, access: 'user', desc: 'Operating instructions for the agent.' },
  ],
  memory: [
    { file: 'MEMORY.md', label: 'Memory', cap: 16000, access: 'agent', desc: 'Long-term facts from conversations.' },
    { file: 'USER.md', label: 'Preferences', cap: 4000, access: 'agent', desc: 'Your preferences and working style.' },
    { file: 'HEARTBEAT.md', label: 'Heartbeat', cap: null, access: 'agent', desc: 'Rules for autonomous operation.' },
  ],
};

function dashboard() {
  return {
    // Navigation
    activeTab: 'fleet',
    tabs: [
      { id: 'fleet', label: 'Agents' },
      { id: 'activity', label: 'Activity' },
      { id: 'system', label: 'System' },
    ],
    connected: false,
    loading: true,
    lastRefresh: 0,
    unreadEvents: 0,
    _lastSeenEventCount: 0,
    toastQueue: [],

    // Fleet
    agents: [],
    agentStates: {},
    _stateTimers: {},

    // Events
    events: [],
    eventFilters: [],
    eventTypes: [
      'agent_state', 'message_sent', 'message_received',
      'tool_start', 'tool_result', 'text_delta', 'llm_call',
      'blackboard_write', 'health_change', 'notification',
    ],

    // Agent detail
    detailAgent: null,
    selectedAgent: null,
    agentDetail: null,
    showBrowserViewer: false,

    // Agent config
    agentConfigs: {},
    editForm: {},
    availableModels: [],

    // Add agent
    addAgentMode: false,
    addAgentForm: { name: '', role: '', model: '' },
    addAgentLoading: false,

    // Blackboard
    bbEntries: [],
    bbPrefix: '',
    bbHighlights: [],
    bbLoading: false,
    bbWriteMode: false,
    bbNewKey: '',
    bbNewValue: '{}',
    bbWriterFilter: '',
    bbExpanded: {},

    // PROJECT.md (per-project only)
    projectContent: '',
    projectExists: false,
    projectLoading: false,
    projectEditing: false,
    projectEditBuffer: '',
    projectSaving: false,

    // Multi-project support
    projects: [],
    activeProject: null,
    projectsLoaded: false,

    // Project management
    showProjectForm: false,
    newProjectName: '',
    newProjectDesc: '',
    projectFormLoading: false,

    // Costs
    costData: {},
    costPeriod: 'today',
    costChart: null,
    _costDebounce: null,

    // Traces
    traces: [],
    tracesLoading: false,

    // Activity view
    activityView: 'traces',
    expandedTraces: {},
    _activityRefresh: null,
    _tracesDebounce: null,

    // Queues
    queueStatus: {},

    // Cron / Automation
    cronJobs: [],
    editingCronJob: null,
    cronEditSchedule: '',
    cronEditAmount: 15,
    cronEditUnit: 'm',
    cronEditCron: '',

    // Settings
    settingsData: null,

    // Slide-over chat panel
    openChats: [],             // Array of agent IDs with open chat panels
    chatHistories: {},         // Preserved — keyed by agent ID
    chatLoadingAgents: {},     // { agentId: true/false }
    chatStreamingAgents: {},   // { agentId: true/false }
    _chatAborts: {},           // { agentId: AbortController }
    activeChatId: '',          // Currently active chat tab
    chatPanelMinimized: false, // Whether the slide-over is minimized to pill
    chatUnread: {},            // { agentId: count } — unread notifications while minimized

    // Identity panel
    identityTabs: _IDENTITY_TABS,
    identityFileMap: _IDENTITY_FILE_MAP,
    identityTab: 'identity',
    identityFiles: [],
    identityLoading: false,
    identityContent: {},
    identityContentLoading: false,
    identityEditing: false,
    identityEditBuffer: '',
    identitySaving: false,
    identityEditingFile: null,  // Which specific file is being edited
    configEditing: false,
    configSaving: false,
    identityLogs: null,
    identityLogsLoading: false,
    identityLearnings: null,
    identityLearningsLoading: false,

    // Broadcast
    broadcastMessage: '',

    // Command palette (Cmd+K)
    cmdPaletteOpen: false,
    cmdPaletteQuery: '',
    cmdPaletteResults: [],
    cmdPaletteIdx: 0,

    // System tab — collapsible infrastructure
    systemInfraExpanded: false,

    // PROJECT.md banner on Agents tab
    projectBannerExpanded: false,

    // Credentials
    showCredForm: false,
    credService: '',
    credCustomService: '',
    credKey: '',
    credBaseUrl: '',
    credTier: 'agent',

    // Onboarding
    onboardProvider: '',
    onboardKey: '',
    onboardBaseUrl: '',

    // Confirm modal
    confirmModal: { open: false, title: '', message: '', action: null, destructive: false },

    // Logs viewer
    systemLogs: [],
    systemLogsLoading: false,
    systemLogsLevel: '',
    systemLogsMaxLines: 200,

    // Activity search/filter
    activitySearch: '',
    activityAgentFilter: '',
    activityTimeRange: 'all',

    // WebSocket
    _ws: null,
    _refreshInterval: null,
    _fleetDebounce: null,
    _queueInterval: null,
    _cronInterval: null,
    _seenEventIds: new Set(),

    // URL routing
    _skipPush: false,
    _popstateHandler: null,

    // ── URL Routing ──────────────────────────────────────────

    _buildPath() {
      if (this.detailAgent) {
        const tab = this.identityTab || 'identity';
        return tab === 'identity'
          ? `/agents/${this.detailAgent}`
          : `/agents/${this.detailAgent}/${tab}`;
      }
      if (this.activeTab === 'activity') {
        return this.activityView === 'events'
          ? '/activity/events'
          : '/activity';
      }
      if (this.activeTab === 'system') return '/system';
      return '/';
    },

    _buildTitle() {
      if (this.detailAgent) {
        const tabLabel = (_IDENTITY_TABS.find(t => t.id === this.identityTab) || _IDENTITY_TABS[0]).label;
        return `${this.detailAgent} \u00b7 ${tabLabel} \u2014 OpenLegion`;
      }
      if (this.activeTab === 'activity') {
        return this.activityView === 'events'
          ? 'Events \u2014 OpenLegion'
          : 'Traces \u2014 OpenLegion';
      }
      if (this.activeTab === 'system') return 'System \u2014 OpenLegion';
      return 'Agents \u2014 OpenLegion';
    },

    _parsePath(path) {
      const clean = path.replace(/^\/+/, '').replace(/\/+$/, '');
      const route = { tab: 'fleet', activityView: 'traces', agentId: null, identityTab: 'identity' };
      if (!clean) return route;

      const agentMatch = clean.match(/^agents\/([^/]+)(?:\/([^/]+))?$/);
      if (agentMatch) {
        route.agentId = agentMatch[1];
        const tab = agentMatch[2];
        if (tab && _IDENTITY_TABS.some(t => t.id === tab)) route.identityTab = tab;
        return route;
      }

      if (clean === 'activity/events') { route.tab = 'activity'; route.activityView = 'events'; }
      else if (clean === 'activity') { route.tab = 'activity'; }
      else if (clean === 'system') { route.tab = 'system'; }
      return route;
    },

    _pushUrl(replace) {
      if (this._skipPush) return;
      const path = this._buildPath();
      const title = this._buildTitle();
      document.title = title;
      if (window.location.pathname === path) return;
      if (replace) {
        history.replaceState(null, '', path);
      } else {
        history.pushState(null, '', path);
      }
    },

    _applyRoute(route) {
      this._skipPush = true;
      try {
        if (route.agentId) {
          if (this.detailAgent !== route.agentId) {
            this.drillDown(route.agentId);
          }
          if (this.identityTab !== route.identityTab) {
            this.identityTab = route.identityTab;
            if (this.selectedAgent) this.loadIdentityTabContent(this.selectedAgent);
          }
        } else {
          if (this.detailAgent) {
            this.detailAgent = null;
            this.selectedAgent = null;
          }
          if (this.activeTab !== route.tab) {
            this.switchTab(route.tab);
          }
          if (route.tab === 'activity' && this.activityView !== route.activityView) {
            this.setActivityView(route.activityView);
          }
        }
      } finally {
        this._skipPush = false;
      }
    },

    closeDetail() {
      this.detailAgent = null;
      this.selectedAgent = null;
      this._pushUrl(false);
    },

    // ── Confirm modal ─────────────────────────────────────

    showConfirm(title, message, action, destructive = true) {
      this.confirmModal = { open: true, title, message, action, destructive };
    },

    cancelConfirm() {
      this.confirmModal = { open: false, title: '', message: '', action: null, destructive: false };
    },

    async executeConfirm() {
      if (this.confirmModal.action) await this.confirmModal.action();
      this.cancelConfirm();
    },

    // ── Computed ───────────────────────────────────────────

    get showOnboarding() {
      if (this.loading || !this.settingsData) return false;
      return !this.settingsData.has_llm_credentials || this.agents.length === 0;
    },

    get filteredEvents() {
      if (this.eventFilters.length === 0) {
        return this.events.filter(e =>
          !(e.type === 'agent_state' && e.data?.state === 'registered'));
      }
      return this.events.filter(e => this.eventFilters.includes(e.type));
    },

    get fleetTotalCost() {
      return this.agents.reduce((sum, a) => sum + (a.daily_cost || 0), 0);
    },

    get fleetTotalTokens() {
      return this.agents.reduce((sum, a) => sum + (a.daily_tokens || 0), 0);
    },

    get filteredAgents() {
      if (this.activeProject) {
        return this.agents.filter(a => a.project === this.activeProject);
      }
      // When projects exist, show only standalone (unassigned) agents
      return this.projects.length > 0 ? this.unassignedAgents : this.agents;
    },

    get filteredFleetCost() {
      return this.filteredAgents.reduce((sum, a) => sum + (a.daily_cost || 0), 0);
    },

    get filteredFleetTokens() {
      return this.filteredAgents.reduce((sum, a) => sum + (a.daily_tokens || 0), 0);
    },

    get fleetHealthCounts() {
      const counts = { healthy: 0, unhealthy: 0, failed: 0, unknown: 0 };
      for (const a of this.filteredAgents) {
        const s = a.health_status || 'unknown';
        if (s === 'healthy') counts.healthy++;
        else if (s === 'unhealthy' || s === 'restarting') counts.unhealthy++;
        else if (s === 'failed') counts.failed++;
        else counts.unknown++;
      }
      return counts;
    },

    get costTotal() {
      return (this.costData.agents || []).reduce((sum, a) => sum + (a.cost || 0), 0);
    },

    get identityCurrentTab() {
      return _IDENTITY_TABS.find(t => t.id === this.identityTab) || _IDENTITY_TABS[0];
    },

    fileBudgetPct(file, cap) {
      if (!cap) return 0;
      const text = (this.identityEditing && this.identityEditingFile === file) ? this.identityEditBuffer : (this.identityContent[file] || '');
      return Math.min(100, (text.length / cap) * 100);
    },

    fileBudgetColor(file, cap) {
      const pct = this.fileBudgetPct(file, cap);
      if (pct >= 95) return 'bg-red-500';
      if (pct >= 80) return 'bg-amber-500';
      return 'bg-indigo-500';
    },

    fileCharCount(file) {
      const text = (this.identityEditing && this.identityEditingFile === file) ? this.identityEditBuffer : (this.identityContent[file] || '');
      return text.length;
    },

    get filteredTraces() {
      let items = this.traces;
      if (this.activitySearch) {
        const q = this.activitySearch.toLowerCase();
        items = items.filter(t => {
          const summary = (t.trigger_preview || t.trigger_detail || '').toLowerCase();
          const agent = (t.agents || []).join(' ').toLowerCase();
          const detail = (t.trace_id || '').toLowerCase();
          return summary.includes(q) || agent.includes(q) || detail.includes(q);
        });
      }
      if (this.activityAgentFilter) {
        items = items.filter(t => (t.agents || []).includes(this.activityAgentFilter));
      }
      return items;
    },

    get activityAgents() {
      const allAgents = new Set();
      for (const t of this.traces) {
        for (const a of (t.agents || [])) {
          if (a) allAgents.add(a);
        }
      }
      return [...allAgents].sort();
    },

    get filteredBbEntries() {
      if (!this.bbWriterFilter) return this.bbEntries;
      return this.bbEntries.filter(e => e.written_by === this.bbWriterFilter);
    },

    get bbWriters() {
      return [...new Set(this.bbEntries.map(e => e.written_by))].sort();
    },

    get bbNamespaceCounts() {
      const counts = {};
      for (const entry of this.bbEntries) {
        const ns = this.bbNamespaceOf(entry.key);
        counts[ns] = (counts[ns] || 0) + 1;
      }
      return counts;
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
      this.fetchSettings();
      this.fetchProject();
      this.fetchProjects();
      this._refreshInterval = setInterval(() => this.fetchAgents(), 15000);

      // Command palette: Cmd+K / Ctrl+K + tab shortcuts 1/2/3
      this._cmdPaletteHandler = (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
          e.preventDefault();
          this.cmdPaletteOpen = !this.cmdPaletteOpen;
          if (this.cmdPaletteOpen) {
            this.cmdPaletteQuery = '';
            this.cmdPaletteResults = [];
            this.cmdPaletteIdx = 0;
            this.$nextTick(() => {
              const el = document.getElementById('cmd-palette-input');
              if (el) el.focus();
            });
          }
        }
        if (e.key === 'Escape' && this.cmdPaletteOpen) {
          e.stopPropagation();
          e.preventDefault();
          this.cmdPaletteOpen = false;
        }
        // Tab switching: 1/2/3 when no input focused and no modal open
        if (!e.metaKey && !e.ctrlKey && !e.altKey && ['1', '2', '3'].includes(e.key)) {
          const active = document.activeElement;
          const tag = active ? active.tagName.toLowerCase() : '';
          if (tag === 'input' || tag === 'textarea' || tag === 'select' || active?.isContentEditable) return;
          if (this.cmdPaletteOpen || this.detailAgent || this.addAgentMode) return;
          e.preventDefault();
          const tabMap = { '1': 'fleet', '2': 'activity', '3': 'system' };
          this.switchTab(tabMap[e.key]);
        }
      };
      document.addEventListener('keydown', this._cmdPaletteHandler);

      // Normalize legacy /dashboard/ URL to root
      if (/^\/dashboard\/?$/.test(window.location.pathname)) {
        history.replaceState(null, '', '/');
      }

      // Deep link restoration: parse initial URL and apply route
      const initRoute = this._parsePath(window.location.pathname);
      const isDeepLink = initRoute.agentId || initRoute.tab !== 'fleet' || initRoute.activityView !== 'traces';
      if (isDeepLink) {
        this.$nextTick(() => {
          this._applyRoute(initRoute);
          document.title = this._buildTitle();
          history.replaceState(null, '', this._buildPath());
        });
      } else {
        document.title = this._buildTitle();
      }

      // Popstate listener for browser back/forward
      this._popstateHandler = () => {
        // Guard unsaved edits
        if (this.identityEditing || this.configEditing || this.projectEditing) {
          if (!confirm('You have unsaved changes. Discard and navigate away?')) {
            // Re-push current URL to cancel the back navigation
            this._pushUrl(true);
            return;
          }
          this.identityEditing = false;
          this.identityEditBuffer = '';
          this.configEditing = false;
          this.editForm = {};
          this.projectEditing = false;
          this.projectEditBuffer = '';
        }
        const route = this._parsePath(window.location.pathname);
        this._applyRoute(route);
        document.title = this._buildTitle();
      };
      window.addEventListener('popstate', this._popstateHandler);

      // Pause polling when tab is hidden to save resources
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
          if (this._refreshInterval) { clearInterval(this._refreshInterval); this._refreshInterval = null; }
          if (this._queueInterval) { clearInterval(this._queueInterval); this._queueInterval = null; }
          if (this._cronInterval) { clearInterval(this._cronInterval); this._cronInterval = null; }
          this._stopActivityRefresh();
        } else {
          // Resume polling — restore intervals without navigating (switchTab clears detailAgent)
          this._refreshInterval = setInterval(() => this.fetchAgents(), 15000);
          this.fetchAgents();
          if (this.activeTab === 'fleet') {
            this.fetchQueues();
            this._queueInterval = setInterval(() => this.fetchQueues(), 5000);
          }
          if (this.activeTab === 'activity' && this.activityView === 'traces') {
            this.fetchTraces();
            this._startActivityRefresh();
          }
          if (this.activeTab === 'system') {
            this.fetchCronJobs();
            this._cronInterval = setInterval(() => this.fetchCronJobs(), 10000);
          }
          // Refresh agent detail if we're viewing one
          if (this.detailAgent) {
            this.fetchAgentDetail(this.detailAgent);
          }
        }
      });
    },

    destroy() {
      if (this._ws) this._ws.disconnect();
      if (this._refreshInterval) clearInterval(this._refreshInterval);
      if (this._queueInterval) clearInterval(this._queueInterval);
      if (this._cronInterval) clearInterval(this._cronInterval);
      if (this._costDebounce) clearTimeout(this._costDebounce);
      if (this._fleetDebounce) clearTimeout(this._fleetDebounce);
      if (this._activityRefresh) clearInterval(this._activityRefresh);
      if (this._tracesDebounce) clearTimeout(this._tracesDebounce);
      if (this.costChart) this.costChart.destroy();
      Object.values(this._stateTimers).forEach(clearTimeout);
      Object.values(this._scrollTimers).forEach(clearTimeout);
      Object.values(this._chatAborts).forEach(c => c?.abort());
      if (this._cmdPaletteHandler) document.removeEventListener('keydown', this._cmdPaletteHandler);
      if (this._popstateHandler) window.removeEventListener('popstate', this._popstateHandler);
    },

    // ── Tab switching ─────────────────────────────────────

    switchTab(tab) {
      this.activeTab = tab;
      this.detailAgent = null;
      // Clear tab-specific auto-refresh intervals
      if (this._queueInterval) { clearInterval(this._queueInterval); this._queueInterval = null; }
      if (this._cronInterval) { clearInterval(this._cronInterval); this._cronInterval = null; }
      this._stopActivityRefresh();
      if (tab === 'activity') {
        this.unreadEvents = 0;
        this._lastSeenEventCount = this.events.length;
        if (this.activityView === 'traces') {
          this.fetchTraces();
          this._startActivityRefresh();
        }
      }
      if (tab === 'fleet') {
        this.fetchAgents();
        this.fetchQueues();
        this.fetchSettings();
        this.fetchProject();
        this.fetchProjects();
        this._queueInterval = setInterval(() => this.fetchQueues(), 5000);
      }
      if (tab === 'system') {
        this.fetchSettings();
        this.fetchCosts();
        this.fetchBlackboard();
        this.fetchCronJobs();
        this._cronInterval = setInterval(() => this.fetchCronJobs(), 10000);
      }
      if (!this._skipPush) this._pushUrl(false);
    },

    // ── Markdown rendering for chat messages ─────────────

    renderMarkdown(text) {
      if (!text) return '';
      // Strip <think>...</think> blocks and unclosed <think> (still streaming)
      let cleaned = text.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').trim();
      if (!cleaned) return '';
      const html = marked.parse(cleaned, { breaks: true, gfm: true });
      return DOMPurify.sanitize(html);
    },

    // ── Toast helper ──────────────────────────────────────

    _toastId: 0,

    showToast(msg) {
      const id = ++this._toastId;
      this.toastQueue.push({ id, msg });
      setTimeout(() => {
        this.toastQueue = this.toastQueue.filter(t => t.id !== id);
      }, 4000);
    },

    // ── WebSocket event handler ───────────────────────────

    onWsEvent(evt) {
      // Deduplicate by event ID (handles reconnect replays)
      if (evt.id && this._seenEventIds.has(evt.id)) return;
      if (evt.id) {
        this._seenEventIds.add(evt.id);
        if (this._seenEventIds.size > 1000) {
          const iter = this._seenEventIds.values();
          for (let i = 0; i < 500; i++) this._seenEventIds.delete(iter.next().value);
        }
      }

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
      }

      // Live-update fleet on llm_call/health changes (debounced)
      if (evt.type === 'llm_call' || evt.type === 'health_change') {
        this._debouncedFleetRefresh();
      }

      // Show toast for agent notifications + inject into chat panel
      if (evt.type === 'notification' && evt.agent) {
        this.showToast(`[${evt.agent}] ${(evt.data?.message || '').substring(0, 120)}`);
        if (!this.chatHistories[evt.agent]) this.chatHistories[evt.agent] = [];
        this.chatHistories[evt.agent].push({
          role: 'notification',
          content: evt.data?.message || '',
          streaming: false,
          tools: [],
        });
        if (this.openChats.includes(evt.agent)) {
          if (this.chatPanelMinimized || this.activeChatId !== evt.agent) {
            this.chatUnread = { ...this.chatUnread, [evt.agent]: (this.chatUnread[evt.agent] || 0) + 1 };
          } else {
            this.$nextTick(() => this._scrollChat(evt.agent));
          }
        } else {
          // Open a new chat tab without stealing focus
          this.openChats.push(evt.agent);
          if (!this.activeChatId) this.activeChatId = evt.agent;
          this.$nextTick(() => this._scrollChat(evt.agent));
        }
      }

      // Highlight blackboard writes
      if (evt.type === 'blackboard_write' && evt.data && evt.data.key) {
        if (!this.bbHighlights.includes(evt.data.key)) this.bbHighlights.push(evt.data.key);
        setTimeout(() => { const i = this.bbHighlights.indexOf(evt.data.key); if (i !== -1) this.bbHighlights.splice(i, 1); }, 5000);
        if (this.activeTab === 'system') this.fetchBlackboard();
      }

      // Debounced cost panel refresh on llm_call events
      if (evt.type === 'llm_call' && this.activeTab === 'system') {
        if (this._costDebounce) clearTimeout(this._costDebounce);
        this._costDebounce = setTimeout(() => this.fetchCosts(), 2000);
      }

      // Debounced trace refresh when on traces view
      if (this.activeTab === 'activity' && this.activityView === 'traces') {
        if (['llm_call', 'message_sent', 'tool_result'].includes(evt.type)) {
          if (this._tracesDebounce) clearTimeout(this._tracesDebounce);
          this._tracesDebounce = setTimeout(() => this.fetchTraces(), 3000);
        }
      }
    },

    _updateAgentState(agent, eventType) {
      const stateMap = {
        llm_call: 'thinking',
        tool_start: 'tool',
        tool_result: 'thinking',
        text_delta: 'streaming',
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

    setActivityView(view) {
      this.activityView = view;
      this._stopActivityRefresh();
      if (view === 'traces') {
        this.fetchTraces();
        this._startActivityRefresh();
      }
      if (!this._skipPush) this._pushUrl(false);
    },

    async toggleTraceExpand(traceId) {
      if (this.expandedTraces[traceId]) {
        // Reassign to trigger Alpine reactivity (delete doesn't reliably trigger proxy updates)
        const copy = { ...this.expandedTraces };
        delete copy[traceId];
        this.expandedTraces = copy;
        return;
      }
      this.expandedTraces = { ...this.expandedTraces, [traceId]: { events: [], loading: true } };
      try {
        const resp = await fetch(`${window.__config.apiBase}/traces/${traceId}`);
        if (resp.ok) {
          const data = await resp.json();
          this.expandedTraces = { ...this.expandedTraces, [traceId]: { events: data.events || [], loading: false } };
        } else {
          this.expandedTraces = { ...this.expandedTraces, [traceId]: { events: [], loading: false } };
        }
      } catch (e) {
        console.warn('fetchTraceDetail failed:', e);
        this.expandedTraces = { ...this.expandedTraces, [traceId]: { events: [], loading: false } };
      }
    },

    _startActivityRefresh() {
      this._stopActivityRefresh();
      this._activityRefresh = setInterval(() => this.fetchTraces(), 10000);
    },

    _stopActivityRefresh() {
      if (this._activityRefresh) { clearInterval(this._activityRefresh); this._activityRefresh = null; }
      if (this._tracesDebounce) { clearTimeout(this._tracesDebounce); this._tracesDebounce = null; }
    },

    toggleEventFilter(type) {
      const idx = this.eventFilters.indexOf(type);
      if (idx !== -1) {
        this.eventFilters.splice(idx, 1);
      } else {
        this.eventFilters.push(type);
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

    // ── Identity panel methods ─────────────────────────────

    async fetchIdentityFiles(agentId) {
      this.identityLoading = true;
      this.identityFiles = [];
      this.identityContent = {};
      this.identityEditing = false;
      this.identityEditBuffer = '';
      this.identityLogs = null;
      this.identityLearnings = null;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/workspace`);
        if (resp.ok) this.identityFiles = (await resp.json()).files || [];
      } catch (e) { console.warn('fetchIdentityFiles failed:', e); }
      this.identityLoading = false;
      await this.loadIdentityTabContent(agentId);
    },

    async loadIdentityTabContent(agentId) {
      const tab = this.identityCurrentTab;
      if (tab.id === 'config') {
        if (!this.agentConfigs[agentId]) await this.fetchAgentConfig(agentId);
        return;
      }
      // Composite tabs — load all mapped files
      const fileMap = _IDENTITY_FILE_MAP[tab.id];
      if (fileMap) {
        this.identityContentLoading = true;
        for (const entry of fileMap) {
          if (this.identityContent[entry.file] !== undefined) continue;
          try {
            const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/workspace/${entry.file}`);
            if (resp.ok) {
              const data = await resp.json();
              this.identityContent = { ...this.identityContent, [entry.file]: data.content || '' };
            }
          } catch (e) { console.warn('loadIdentityTabContent failed:', e); }
        }
        this.identityContentLoading = false;
        return;
      }
      if (tab.id === 'logs') {
        if (this.identityLogs === null) await this.fetchIdentityLogs(agentId);
        if (this.identityLearnings === null) await this.fetchIdentityLearnings(agentId);
      }
    },

    async switchIdentityTab(agentId, tabId) {
      if (this.identityEditing || this.configEditing) {
        this.showConfirm('Discard changes?', 'You have unsaved changes that will be lost.', async () => {
          this.identityEditing = false;
          this.identityEditBuffer = '';
          this.configEditing = false;
          this.editForm = {};
          this.identityTab = tabId;
          this.identityContentLoading = false;
          await this.loadIdentityTabContent(agentId);
          if (!this._skipPush) this._pushUrl(false);
        }, true);
        return;
      }
      this.identityTab = tabId;
      // Reset file-loading flag to prevent stale spinner on non-file tabs
      this.identityContentLoading = false;
      await this.loadIdentityTabContent(agentId);
      if (!this._skipPush) this._pushUrl(false);
    },

    startIdentityEdit(file) {
      if (!file) return;
      this.identityEditingFile = file;
      this.identityEditBuffer = this.identityContent[file] || '';
      this.identityEditing = true;
    },

    cancelIdentityEdit() {
      this.identityEditing = false;
      this.identityEditingFile = null;
      this.identityEditBuffer = '';
    },

    async saveIdentityFile(agentId, file) {
      if (!file) file = this.identityEditingFile;
      if (!file) return;
      this.identitySaving = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/workspace/${file}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: this.identityEditBuffer }),
        });
        if (resp.ok) {
          this.identityContent = { ...this.identityContent, [file]: this.identityEditBuffer };
          this.identityEditing = false;
          this.identityEditingFile = null;
          this.identityEditBuffer = '';
          this.showToast(`Saved ${file}`);
          try {
            const listResp = await fetch(`${window.__config.apiBase}/agents/${agentId}/workspace`);
            if (listResp.ok) this.identityFiles = (await listResp.json()).files || [];
          } catch (_) {}
        } else {
          try {
            const err = await resp.json();
            this.showToast(`Save failed: ${err.detail || 'Unknown error'}`);
          } catch (_) { this.showToast('Save failed'); }
        }
      } catch (e) { this.showToast(`Save failed: ${e.message}`); }
      this.identitySaving = false;
    },

    async fetchIdentityLogs(agentId) {
      this.identityLogsLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/workspace-logs?days=3`);
        if (resp.ok) this.identityLogs = (await resp.json()).logs || '';
      } catch (e) { console.warn('fetchIdentityLogs failed:', e); }
      this.identityLogsLoading = false;
    },

    async fetchIdentityLearnings(agentId) {
      this.identityLearningsLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/workspace-learnings`);
        if (resp.ok) this.identityLearnings = await resp.json();
      } catch (e) { console.warn('fetchIdentityLearnings failed:', e); }
      this.identityLearningsLoading = false;
    },

    async fetchProject() {
      if (!this.activeProject) {
        this.projectContent = '';
        this.projectExists = false;
        this.projectLoading = false;
        return;
      }
      this.projectLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/project?project=${encodeURIComponent(this.activeProject)}`);
        if (resp.ok) {
          const data = await resp.json();
          this.projectContent = data.content || '';
          this.projectExists = data.exists;
        }
      } catch (e) { console.warn('fetchProject failed:', e); }
      this.projectLoading = false;
    },

    async saveProject() {
      if (!this.activeProject) return;
      this.projectSaving = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/project?project=${encodeURIComponent(this.activeProject)}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: this.projectEditBuffer }),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.projectContent = this.projectEditBuffer;
          this.projectExists = true;
          this.projectEditing = false;
          this.projectEditBuffer = '';
          const pushed = Object.values(data.pushed || {}).filter(Boolean).length;
          const total = Object.keys(data.pushed || {}).length;
          this.showToast(`${this.activeProject} PROJECT.md saved${total > 0 ? ` (pushed to ${pushed}/${total} agents)` : ''}`);
        } else {
          try {
            const err = await resp.json();
            this.showToast(`Save failed: ${err.detail || 'Unknown error'}`);
          } catch (_) { this.showToast('Save failed'); }
        }
      } catch (e) { this.showToast(`Save failed: ${e.message}`); }
      this.projectSaving = false;
    },

    startProjectEdit() {
      this.projectEditBuffer = this.projectContent;
      this.projectEditing = true;
    },

    cancelProjectEdit() {
      this.projectEditing = false;
      this.projectEditBuffer = '';
    },

    async fetchProjects() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/projects`);
        if (resp.ok) {
          const data = await resp.json();
          this.projects = data.projects || [];
          this.projectsLoaded = true;
        }
      } catch (e) { console.warn('fetchProjects failed:', e); }
    },

    switchProject(name) {
      if (this.activeProject === name) return;
      this.activeProject = name;
      this.projectEditing = false;
      this.projectEditBuffer = '';
      this.projectBannerExpanded = false;
      this.showProjectForm = false;
      this.fetchProject();
    },

    async createProject() {
      const name = this.newProjectName.trim();
      if (!name) return;
      if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(name)) {
        this.showToast('Project name must start with a letter or number and contain only letters, numbers, hyphens, underscores');
        return;
      }
      this.projectFormLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/projects`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, description: this.newProjectDesc.trim(), members: [] }),
        });
        if (resp.ok) {
          this.newProjectName = '';
          this.newProjectDesc = '';
          this.showProjectForm = false;
          await this.fetchProjects();
          this.switchProject(name);
          this.showToast(`Project "${name}" created`);
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Create failed: ${err.detail || 'Unknown error'}`);
        }
      } catch (e) { this.showToast(`Create failed: ${e.message}`); }
      this.projectFormLoading = false;
    },

    async deleteProject(name) {
      this.showConfirm('Delete Project', `Delete project "${name}"? Members will become standalone.`, async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/projects/${encodeURIComponent(name)}`, {
            method: 'DELETE',
          });
          if (resp.ok) {
            this.projectEditing = false;
            this.projectEditBuffer = '';
            await this.fetchProjects();
            if (this.activeProject === name) this.switchProject(null);
            this.showToast(`Project "${name}" deleted`);
          } else {
            const err = await resp.json().catch(() => ({}));
            this.showToast(`Delete failed: ${err.detail || 'Unknown error'}`);
          }
        } catch (e) { this.showToast(`Delete failed: ${e.message}`); }
      }, true);
    },

    async addMember(project, agent) {
      try {
        const resp = await fetch(`${window.__config.apiBase}/projects/${encodeURIComponent(project)}/members`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ agent }),
        });
        if (resp.ok) {
          await this.fetchProjects();
          this.fetchAgents();
          this.showToast(`${agent} added to ${project}`);
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Add failed: ${err.detail || 'Unknown error'}`);
        }
      } catch (e) { this.showToast(`Add failed: ${e.message}`); }
    },

    async removeMember(project, agent) {
      try {
        const resp = await fetch(`${window.__config.apiBase}/projects/${encodeURIComponent(project)}/members/${encodeURIComponent(agent)}`, {
          method: 'DELETE',
        });
        if (resp.ok) {
          await this.fetchProjects();
          this.fetchAgents();
          this.showToast(`${agent} removed from ${project}`);
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Remove failed: ${err.detail || 'Unknown error'}`);
        }
      } catch (e) { this.showToast(`Remove failed: ${e.message}`); }
    },

    /** Agents in the registry that are not in any project. */
    get unassignedAgents() {
      const assigned = new Set();
      for (const p of this.projects) {
        for (const m of (p.members || [])) assigned.add(m);
      }
      return this.agents.filter(a => !assigned.has(a.id));
    },

    /** Get the active project object. */
    get activeProjectData() {
      return this.projects.find(p => p.name === this.activeProject) || null;
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

    async fetchSystemLogs() {
      this.systemLogsLoading = true;
      try {
        const params = new URLSearchParams({ lines: this.systemLogsMaxLines });
        if (this.systemLogsLevel) params.set('level', this.systemLogsLevel);
        const r = await fetch(`${window.__config.apiBase}/logs?${params}`);
        const data = await r.json();
        this.systemLogs = data.lines || [];
      } catch (e) {
        console.error('Failed to fetch logs:', e);
      } finally {
        this.systemLogsLoading = false;
      }
    },

    // ── Agent config fetchers ─────────────────────────────

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

    startConfigEdit() {
      const cfg = this.agentConfigs[this.selectedAgent];
      if (!cfg) return;
      const creds = cfg.allowed_credentials || [];
      const credsStr = creds.join(', ');
      let credMode = 'none';
      if (creds.length === 1 && creds[0] === '*') credMode = 'all';
      else if (creds.length > 0) credMode = 'custom';
      this.editForm = {
        model: cfg.model || '',
        role: cfg.role || '',
        budget_daily: cfg.budget?.daily_usd || '',
        allowed_credentials: credsStr,
        _credMode: credMode,
      };
      this.configEditing = true;
    },

    cancelConfigEdit() {
      this.configEditing = false;
      this.editForm = {};
    },

    async saveConfigFromDetail(agentId) {
      this.configSaving = true;
      await this.saveAgentConfig(agentId);
      // saveAgentConfig already calls cancelConfigEdit + fetchAgentConfig
      this.configSaving = false;
      await this.fetchAgentDetail(agentId);
    },

    async saveAgentConfig(agentId) {
      const body = {};
      const cfg = this.agentConfigs[agentId] || {};
      if (this.editForm.model && this.editForm.model !== cfg.model) body.model = this.editForm.model;
      if (this.editForm.role !== undefined && this.editForm.role !== cfg.role) body.role = this.editForm.role;
      if (this.editForm.budget_daily && parseFloat(this.editForm.budget_daily) > 0) {
        body.budget = { daily_usd: parseFloat(this.editForm.budget_daily) };
      }
      // Handle allowed_credentials via the permissions endpoint
      const newCreds = (this.editForm.allowed_credentials || '').split(',').map(s => s.trim()).filter(Boolean);
      const oldCreds = cfg.allowed_credentials || [];
      const credsChanged = JSON.stringify(newCreds) !== JSON.stringify(oldCreds);
      if (Object.keys(body).length === 0 && !credsChanged) {
        this.cancelConfigEdit();
        return;
      }
      try {
        const allUpdated = [];
        let configResult = null;
        if (Object.keys(body).length > 0) {
          const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/config`, {
            method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
          });
          if (resp.ok) {
            configResult = await resp.json();
            allUpdated.push(...configResult.updated);
          } else {
            const err = await resp.json();
            this.showToast(`Error: ${err.detail || 'Update failed'}`);
            this.cancelConfigEdit();
            return;
          }
        }
        if (credsChanged) {
          const permResp = await fetch(`${window.__config.apiBase}/agents/${agentId}/permissions`, {
            method: 'PUT', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ allowed_credentials: newCreds }),
          });
          if (permResp.ok) {
            allUpdated.push('allowed_credentials');
          } else {
            const err = await permResp.json();
            this.showToast(`Error updating permissions: ${err.detail || 'Update failed'}`);
          }
        }
        if (configResult && configResult.restart_required) {
          this.showToast(`Updated: ${allUpdated.join(', ')} — restarting ${agentId}...`);
          const restartResp = await fetch(`${window.__config.apiBase}/agents/${agentId}/restart`, { method: 'POST' });
          if (restartResp.ok) {
            const data = await restartResp.json();
            this.showToast(data.ready ? `${agentId} restarted and ready` : `${agentId} restarted (warming up)`);
            this.fetchAgents();
          } else {
            this.showToast(`Config updated but restart failed — restart manually`);
          }
        } else if (allUpdated.length > 0) {
          this.showToast(`Updated: ${allUpdated.join(', ')}`);
        }
        await this.fetchAgentConfig(agentId);
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      this.cancelConfigEdit();
    },

    async restartAgent(agentId) {
      this.showConfirm('Restart Agent', `Restart agent "${agentId}"? This will interrupt any active work.`, async () => {
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
      }, true);
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
          }),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.showToast(data.ready ? `${data.agent} added and ready` : `${data.agent} added (starting)`);
          this.addAgentMode = false;
          this.addAgentForm = { name: '', role: '', model: '' };
          this.fetchAgents();
        } else {
          const err = await resp.json();
          this.showToast(`Error: ${err.detail || 'Failed to add agent'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      this.addAgentLoading = false;
    },

    openAddAgentModal(defaultName = '') {
      this.addAgentMode = true;
      if (defaultName) this.addAgentForm.name = defaultName;
      this.fetchSettings();
      this.$nextTick(() => {
        const el = document.getElementById('add-agent-name-input');
        if (el) el.focus();
      });
    },

    closeAddAgentModal() {
      if (this.addAgentLoading) return;
      this.addAgentMode = false;
    },

    async removeAgent(agentId) {
      this.showConfirm('Remove Agent', `Remove agent "${agentId}"? This will stop the container and remove its config.`, async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}`, { method: 'DELETE' });
          if (resp.ok) {
            this.showToast(`${agentId} removed`);
            this.closeChat(agentId);
            delete this.chatHistories[agentId];
            if (this.detailAgent === agentId) this.closeDetail();
            this.fetchAgents();
          } else {
            const err = await resp.json();
            this.showToast(`Error: ${err.detail || 'Failed to remove agent'}`);
          }
        } catch (e) { this.showToast(`Error: ${e.message}`); }
      }, true);
    },

    // ── Blackboard write/delete ────────────────────────────

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
      this.showConfirm('Delete Entry', `Delete blackboard key "${key}"?`, async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/blackboard/${key}`, { method: 'DELETE' });
          if (resp.ok) { this.showToast(`Deleted: ${key}`); this.fetchBlackboard(); }
          else { const err = await resp.json(); this.showToast(`Error: ${err.detail}`); }
        } catch (e) { this.showToast(`Error: ${e.message}`); }
      }, true);
    },

    toggleBbExpand(key) {
      this.bbExpanded = { ...this.bbExpanded, [key]: !this.bbExpanded[key] };
    },

    // ── Queue fetcher ────────────────────────────────────

    async fetchQueues() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/queues`);
        if (resp.ok) this.queueStatus = (await resp.json()).queues;
      } catch (e) { console.warn('fetchQueues failed:', e); }
    },

    // ── Cron fetchers ───────────────────────────────────

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

    async deleteCronJob(jobId) {
      this.showConfirm('Delete Cron Job', 'Delete this cron job?', async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/cron/${jobId}`, { method: 'DELETE' });
          if (resp.ok) {
            this.showToast(`Job ${jobId} deleted`);
            this.fetchCronJobs();
          } else {
            try {
              const err = await resp.json();
              this.showToast(`Error: ${err.detail || 'Delete failed'}`);
            } catch (_) { this.showToast('Delete failed'); }
          }
        } catch (e) { console.warn('deleteCronJob failed:', e); }
      }, true);
    },

    // ── Cron inline editing ─────────────────────────────

    editCronJob(job) {
      this.editingCronJob = job.id;
      this.cronEditSchedule = job.schedule;
      const intervalMatch = job.schedule.match(/^every\s+(\d+)([smhd])$/i);
      if (intervalMatch) {
        this.cronEditAmount = parseInt(intervalMatch[1]);
        this.cronEditUnit = intervalMatch[2].toLowerCase();
        this.cronEditCron = '';
      } else {
        this.cronEditAmount = 15;
        this.cronEditUnit = 'm';
        this.cronEditCron = job.schedule;
      }
    },

    cancelCronEdit() {
      this.editingCronJob = null;
      this.cronEditSchedule = '';
      this.cronEditCron = '';
    },

    async saveCronPreset(jobId, preset) {
      this.cronEditSchedule = 'every ' + preset;
      await this.saveCronEdit(jobId);
    },

    async saveCronInterval(jobId) {
      const n = Math.round(this.cronEditAmount);
      if (!n || n < 1) {
        this.showToast('Enter a positive number'); return;
      }
      this.cronEditAmount = n;
      this.cronEditSchedule = `every ${n}${this.cronEditUnit}`;
      await this.saveCronEdit(jobId);
    },

    async saveCronExpression(jobId) {
      if (!this.cronEditCron.trim()) {
        this.showToast('Enter a cron expression'); return;
      }
      this.cronEditSchedule = this.cronEditCron.trim();
      await this.saveCronEdit(jobId);
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
          this.cancelCronEdit();
          this.fetchCronJobs();
        } else {
          const err = await resp.json();
          this.showToast(`Error: ${err.detail || 'Update failed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
    },

    // ── Settings fetcher ─────────────────────────────────

    async fetchSettings() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/settings`);
        if (resp.ok) {
          this.settingsData = await resp.json();
          // Extract models for agent edit forms
          if (this.settingsData.provider_models) {
            this.availableModels = Object.values(this.settingsData.provider_models).flat();
          }
        }
      } catch (e) { console.warn('fetchSettings failed:', e); }
    },

    // ── Chat slide-over panel ──────────────────────────

    openChat(agentId) {
      this.chatPanelMinimized = false;
      if (this.openChats.includes(agentId)) {
        this.activeChatId = agentId;
        if (this.chatUnread[agentId]) this.chatUnread = { ...this.chatUnread, [agentId]: 0 };
        this.$nextTick(() => {
          this._scrollChat(agentId);
          const input = document.getElementById('chat-slide-input');
          if (input) input.focus();
        });
        return;
      }
      this.openChats.push(agentId);
      this.activeChatId = agentId;
      this.$nextTick(() => this._scrollChat(agentId));
    },

    closeChat(agentId) {
      if (this._chatAborts[agentId]) {
        this._chatAborts[agentId].abort();
        delete this._chatAborts[agentId];
      }
      this.openChats = this.openChats.filter(id => id !== agentId);
      this.chatLoadingAgents[agentId] = false;
      this.chatStreamingAgents[agentId] = false;
      // Switch to next open chat or clear
      if (this.activeChatId === agentId) {
        this.activeChatId = this.openChats.length > 0 ? this.openChats[this.openChats.length - 1] : '';
      }
    },

    clearChat(agentId) {
      if (this._chatAborts[agentId]) {
        this._chatAborts[agentId].abort();
        delete this._chatAborts[agentId];
      }
      this.chatHistories[agentId] = [];
      this.chatLoadingAgents[agentId] = false;
      this.chatStreamingAgents[agentId] = false;
    },

    _scrollTimers: {},

    _scrollChat(agentId) {
      if (this._scrollTimers[agentId]) return;
      this._scrollTimers[agentId] = setTimeout(() => {
        delete this._scrollTimers[agentId];
        const el = document.getElementById('chat-messages-' + agentId);
        if (el) el.scrollTop = el.scrollHeight;
      }, 50);
    },

    _escapeHtml(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    },

    _truncateText(value, max = 220) {
      const text = String(value ?? '');
      if (text.length <= max) return text;
      return text.slice(0, max) + '…';
    },

    _chatToolValueToText(value) {
      if (value == null) return '';
      if (typeof value === 'string') return value;
      try {
        return JSON.stringify(value, null, 2);
      } catch (_) {
        return String(value);
      }
    },

    chatToolValueBlock(value) {
      return this._chatToolValueToText(value);
    },

    chatToolPreview(value) {
      const text = this._chatToolValueToText(value).replace(/\s+/g, ' ').trim();
      return this._truncateText(text, 140);
    },

    chatToolCount(msg) {
      if (!msg) return 0;
      if (Array.isArray(msg.tools)) return msg.tools.length;
      if (!Array.isArray(msg.timeline)) return 0;
      return msg.timeline.filter(step => step && step.kind === 'tool').length;
    },

    chatPhaseLabel(msg) {
      if (!msg) return 'Idle';
      if (msg.role === 'error') return 'Error';
      if (msg.streaming) {
        if (msg.phase === 'tool') return 'Using tools';
        if (msg.phase === 'responding') return 'Responding';
        return 'Thinking';
      }
      if (msg.phase === 'done') return 'Completed';
      if (msg.phase === 'responding') return 'Responded';
      if (msg.phase === 'tool') return 'Tool run';
      return 'Ready';
    },

    chatPhaseClass(msg) {
      if (!msg) return 'chat-phase-neutral';
      if (msg.role === 'error') return 'chat-phase-error';
      if (msg.streaming) {
        if (msg.phase === 'tool') return 'chat-phase-tool';
        if (msg.phase === 'responding') return 'chat-phase-responding';
        return 'chat-phase-thinking';
      }
      if (msg.phase === 'done' || msg.phase === 'responding') return 'chat-phase-done';
      if (msg.phase === 'tool') return 'chat-phase-tool';
      return 'chat-phase-neutral';
    },

    chatPhaseMarkerClass(phase) {
      return ({
        thinking: 'chat-phase-thinking',
        tool: 'chat-phase-tool',
        responding: 'chat-phase-responding',
        done: 'chat-phase-done',
        error: 'chat-phase-error',
      })[phase] || 'chat-phase-neutral';
    },

    renderChatMarkdown(text) {
      const source = String(text ?? '');
      if (!source) return '';
      const fallback = this._escapeHtml(source).replace(/\n/g, '<br>');
      const markedLib = window.marked;
      const purify = window.DOMPurify;
      if (!markedLib || !purify) return fallback;
      try {
        const raw = markedLib.parse(source, {
          gfm: true,
          breaks: true,
          headerIds: false,
          mangle: false,
        });
        const sanitized = purify.sanitize(raw, { USE_PROFILES: { html: true } });
        const tpl = document.createElement('template');
        tpl.innerHTML = sanitized;
        tpl.content.querySelectorAll('a[href]').forEach((a) => {
          a.setAttribute('target', '_blank');
          a.setAttribute('rel', 'noopener noreferrer');
        });
        return tpl.innerHTML;
      } catch (_) {
        return fallback;
      }
    },

    _findRunningToolIndex(tools, name) {
      for (let i = tools.length - 1; i >= 0; i -= 1) {
        if (tools[i].name === name && tools[i].status === 'running') return i;
      }
      return -1;
    },

    _chatPhaseStepLabel(phase) {
      const labels = {
        thinking: 'Thinking',
        tool: 'Tool call',
        responding: 'Drafting response',
        done: 'Done',
        error: 'Error',
      };
      return labels[phase] || 'Update';
    },

    _pushChatTimelinePhase(entry, phase) {
      if (!entry) return;
      if (!Array.isArray(entry.timeline)) entry.timeline = [];
      const last = entry.timeline[entry.timeline.length - 1];
      if (last && last.kind === 'phase' && last.phase === phase) return;
      entry.timeline.push({
        kind: 'phase',
        phase,
        label: this._chatPhaseStepLabel(phase),
      });
    },

    _appendChatTimelineText(entry, chunk) {
      if (!entry || !chunk) return;
      if (!Array.isArray(entry.timeline)) entry.timeline = [];
      const last = entry.timeline[entry.timeline.length - 1];
      if (last && last.kind === 'text') {
        last.content += chunk;
        return;
      }
      entry.timeline.push({ kind: 'text', content: chunk });
    },

    _chatTimelineHasText(entry) {
      return !!(entry && Array.isArray(entry.timeline) &&
        entry.timeline.some(step => step && step.kind === 'text' && step.content));
    },

    async sendChatTo(agentId, inputValue) {
      const msg = (inputValue || '').trim();
      if (!msg) return;
      if (!this.chatHistories[agentId]) this.chatHistories[agentId] = [];
      this.chatHistories[agentId].push({ role: 'user', content: msg });
      this.chatLoadingAgents[agentId] = true;
      this.chatStreamingAgents[agentId] = true;

      this.chatHistories[agentId].push({
        role: 'agent',
        content: '',
        streaming: true,
        phase: 'thinking',
        tools: [],
        timeline: [],
        _sawTextDelta: false,
      });
      const idx = this.chatHistories[agentId].length - 1;
      this._pushChatTimelinePhase(this.chatHistories[agentId][idx], 'thinking');
      this.$nextTick(() => this._scrollChat(agentId));

      const controller = new AbortController();
      this._chatAborts[agentId] = controller;

      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/chat/stream`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ message: msg }),
          signal: controller.signal,
        });
        if (!resp.ok) {
          let errMsg = `HTTP ${resp.status}`;
          try { const err = await resp.json(); errMsg = err.detail || errMsg; } catch (_) {}
          this.chatHistories[agentId][idx].content = errMsg;
          this.chatHistories[agentId][idx].role = 'error';
          this.chatHistories[agentId][idx].streaming = false;
          this.chatLoadingAgents[agentId] = false;
          this.chatStreamingAgents[agentId] = false;
          return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let firstToken = true;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            let data;
            try { data = JSON.parse(line.slice(6)); } catch (_) { continue; }

            const entry = this.chatHistories[agentId][idx];

            if (data.type === 'text_delta') {
              if (firstToken) { this.chatLoadingAgents[agentId] = false; firstToken = false; }
              entry.phase = 'responding';
              entry._sawTextDelta = true;
              this._pushChatTimelinePhase(entry, 'responding');
              this._appendChatTimelineText(entry, data.content || '');
              entry.content += data.content || '';
              this._scrollChat(agentId);
            } else if (data.type === 'tool_start') {
              if (firstToken) { this.chatLoadingAgents[agentId] = false; firstToken = false; }
              entry.phase = 'tool';
              this._pushChatTimelinePhase(entry, 'tool');
              const toolEntry = {
                id: `${Date.now()}-${entry.tools.length}`,
                name: data.name,
                status: 'running',
                input: data.input,
                inputPreview: this.chatToolPreview(data.input),
                output: null,
                outputPreview: '',
              };
              entry.tools.push(toolEntry);
              entry.timeline.push({ kind: 'tool', ...toolEntry });
              this._scrollChat(agentId);
            } else if (data.type === 'tool_result') {
              const ti = this._findRunningToolIndex(entry.tools, data.name);
              if (ti >= 0) {
                const tool = entry.tools[ti];
                tool.status = 'done';
                tool.output = data.output;
                tool.outputPreview = this.chatToolPreview(data.output);
                const timelineToolIndex = this._findRunningToolIndex(entry.timeline, data.name);
                if (timelineToolIndex >= 0) {
                  const timelineTool = entry.timeline[timelineToolIndex];
                  timelineTool.status = 'done';
                  timelineTool.output = data.output;
                  timelineTool.outputPreview = tool.outputPreview;
                }
              }
              entry.phase = 'thinking';
              this._pushChatTimelinePhase(entry, 'thinking');
            } else if (data.type === 'done') {
              const finalResponse = data.response || '';
              if (finalResponse && !entry._sawTextDelta && !this._chatTimelineHasText(entry)) {
                this._pushChatTimelinePhase(entry, 'responding');
                this._appendChatTimelineText(entry, finalResponse);
              }
              entry.content = finalResponse || entry.content;
              entry.streaming = false;
              entry.phase = 'done';
            } else if (data.type === 'error') {
              entry.content = data.message || 'Stream error';
              entry.role = 'error';
              entry.streaming = false;
              entry.phase = 'error';
              this._pushChatTimelinePhase(entry, 'error');
            }
          }
        }
        this.chatHistories[agentId][idx].streaming = false;
        if (this.chatHistories[agentId][idx].role !== 'error' && this.chatHistories[agentId][idx].phase !== 'done') {
          this.chatHistories[agentId][idx].phase = 'done';
        }
      } catch (e) {
        if (e.name === 'AbortError') return;
        this.chatHistories[agentId][idx].content = e.message;
        this.chatHistories[agentId][idx].role = 'error';
        this.chatHistories[agentId][idx].streaming = false;
        this.chatHistories[agentId][idx].phase = 'error';
        this._pushChatTimelinePhase(this.chatHistories[agentId][idx], 'error');
      }
      delete this._chatAborts[agentId];
      this.chatLoadingAgents[agentId] = false;
      this.chatStreamingAgents[agentId] = false;
      this.$nextTick(() => this._scrollChat(agentId));
    },

    // ── Broadcast ────────────────────────────────────────

    get broadcastTargets() {
      // Project selected → project members; no project → all agents
      return this.activeProject ? this.filteredAgents : this.agents;
    },

    sendBroadcast() {
      const msg = (this.broadcastMessage || '').trim();
      if (!msg) return;
      const targets = this.broadcastTargets.map(a => a.id);
      if (targets.length === 0) return;
      this.broadcastMessage = '';
      for (const agentId of targets) this.openChat(agentId);
      this.activeChatId = targets[0];
      let sent = 0;
      for (const agentId of targets) {
        if (this.chatStreamingAgents[agentId]) continue;
        this.sendChatTo(agentId, msg).catch(e => {
          console.warn('Broadcast sendChatTo failed for', agentId, e);
        });
        sent++;
      }
      if (sent === 0) {
        this.showToast('All targeted agents are busy');
      } else {
        this.showToast(`Broadcast sent to ${sent} agent${sent !== 1 ? 's' : ''}`);
      }
    },

    // ── Command palette (Cmd+K) ────────────────────────────

    updateCmdPaletteResults() {
      const q = this.cmdPaletteQuery.toLowerCase().trim();
      if (q.length < 2) { this.cmdPaletteResults = []; this.cmdPaletteIdx = 0; return; }
      const results = [];
      // Match agents
      for (const agent of this.agents) {
        const name = (agent.id || '').toLowerCase();
        const role = (agent.role || '').toLowerCase();
        if (name.includes(q) || role.includes(q)) {
          results.push({ type: 'agent', label: agent.id, desc: agent.role || 'Agent', action: () => this.drillDown(agent.id) });
        }
      }
      // Match tabs with keywords
      const tabKeywords = {
        fleet: ['agents', 'fleet', 'cards', 'project'],
        activity: ['activity', 'traces', 'events', 'logs'],
        system: ['system', 'costs', 'cron', 'automation', 'credentials', 'infrastructure', 'pricing', 'browsers', 'pubsub', 'blackboard'],
      };
      for (const [tabId, keywords] of Object.entries(tabKeywords)) {
        const tab = this.tabs.find(t => t.id === tabId);
        if (!tab) continue;
        if (keywords.some(kw => kw.includes(q)) || tab.label.toLowerCase().includes(q)) {
          results.push({ type: 'tab', label: tab.label, desc: `Switch to ${tab.label} tab`, action: () => { this.switchTab(tabId); } });
        }
      }
      // Quick actions
      const actions = [
        { label: 'Add Agent', desc: 'Open add agent form', keywords: ['add', 'agent', 'new', 'create'], action: () => { this.switchTab('fleet'); this.openAddAgentModal(); } },
        { label: 'Broadcast', desc: this.activeProject ? `Broadcast to ${this.activeProject} agents` : (this.projects.length > 0 ? 'Broadcast to standalone agents' : 'Send message to all agents'), keywords: ['broadcast', 'send', 'all', 'message'], action: () => { this.switchTab('fleet'); this.$nextTick(() => document.getElementById('broadcast-input')?.focus()); } },
        ...(this.activeProject ? [{ label: 'Edit PROJECT.md', desc: `Edit ${this.activeProject} project context`, keywords: ['project', 'edit', 'context'], action: () => { this.switchTab('fleet'); this.projectBannerExpanded = true; this.startProjectEdit(); } }] : []),
      ];
      for (const act of actions) {
        if (act.keywords.some(kw => kw.includes(q)) || act.label.toLowerCase().includes(q)) {
          results.push({ type: 'action', label: act.label, desc: act.desc, action: act.action });
        }
      }
      // Match projects
      if (this.projects.length > 0) {
        if ('standalone'.startsWith(q) || 'unassigned'.startsWith(q)) {
          results.push({ type: 'action', label: 'Standalone agents', desc: 'Show agents not in any project', action: () => { this.switchTab('fleet'); this.switchProject(null); } });
        }
        for (const proj of this.projects) {
          const pname = (proj.name || '').toLowerCase();
          if (pname.includes(q) || 'project'.startsWith(q)) {
            results.push({ type: 'action', label: proj.name, desc: `Switch to project (${(proj.members || []).length} members)`, action: () => { this.switchTab('fleet'); this.switchProject(proj.name); } });
          }
        }
      }
      // Match cron jobs
      for (const job of this.cronJobs || []) {
        const id = (job.id || '').toLowerCase();
        const agent = (job.agent || '').toLowerCase();
        const message = (job.message || '').toLowerCase();
        if (id.includes(q) || agent.includes(q) || message.includes(q)) {
          results.push({ type: 'cron', label: job.id, desc: `${job.agent} · ${job.schedule}`, action: () => { this.switchTab('system'); } });
        }
      }
      // Match blackboard keys
      for (const entry of this.bbEntries || []) {
        const key = (entry.key || '').toLowerCase();
        if (key.includes(q)) {
          results.push({ type: 'blackboard', label: entry.key, desc: `by ${entry.written_by || 'unknown'}`, action: () => { this.switchTab('system'); } });
        }
      }
      this.cmdPaletteResults = results.slice(0, 10);
      this.cmdPaletteIdx = 0;
    },

    cmdPaletteNavigate(dir) {
      const len = this.cmdPaletteResults.length;
      if (!len) return;
      this.cmdPaletteIdx = (this.cmdPaletteIdx + dir + len) % len;
    },

    executeCmdPaletteResult(idx) {
      const result = this.cmdPaletteResults[idx ?? this.cmdPaletteIdx];
      if (!result) return;
      this.cmdPaletteOpen = false;
      result.action();
    },

    // ── Reset ────────────────────────────────────────────

    async resetAgent(agentId) {
      this.showConfirm('Reset Conversation', `Reset conversation for "${agentId}"? This clears their chat history.`, async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/reset`, { method: 'POST' });
          if (resp.ok) this.showToast(`${agentId} conversation reset`);
          else this.showToast('Reset failed');
        } catch (e) { this.showToast(`Error: ${e.message}`); }
      }, true);
    },

    // ── Credentials ──────────────────────────────────────

    async addCredential() {
      const service = this.credService === '__custom__' ? this.credCustomService.trim() : this.credService.trim();
      if (!service || !this.credKey.trim()) return;
      try {
        const body = { service, key: this.credKey.trim() };
        if (this.credService === '__custom__' && this.credTier === 'system') body.tier = 'system';
        if (this.credBaseUrl.trim()) body.base_url = this.credBaseUrl.trim();
        const resp = await fetch(`${window.__config.apiBase}/credentials`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.showToast(`Credential stored: ${data.service} (${data.tier} tier)`);
          this.credService = '';
          this.credCustomService = '';
          this.credKey = '';
          this.credBaseUrl = '';
          this.credTier = 'agent';
          this.showCredForm = false;
          this.fetchSettings();
        } else {
          const err = await resp.json();
          this.showToast(`Error: ${err.detail}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
    },

    async deleteCredential(name) {
      this.showConfirm('Remove Credential', `Remove credential "${name}"?`, async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/credentials/${encodeURIComponent(name)}`, {
            method: 'DELETE',
          });
          if (resp.ok) {
            this.showToast(`Credential removed: ${name}`);
            this.fetchSettings();
          } else {
            const err = await resp.json();
            this.showToast(`Error: ${err.detail}`);
          }
        } catch (e) { this.showToast(`Error: ${e.message}`); }
      }, true);
    },

    async submitOnboardCredential() {
      if (!this.onboardProvider || !this.onboardKey.trim()) return;
      try {
        const body = { service: this.onboardProvider, key: this.onboardKey.trim() };
        if (this.onboardBaseUrl.trim()) body.base_url = this.onboardBaseUrl.trim();
        const resp = await fetch(`${window.__config.apiBase}/credentials`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body),
        });
        if (resp.ok) {
          this.showToast('API key saved');
          this.onboardProvider = '';
          this.onboardKey = '';
          this.onboardBaseUrl = '';
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
      this.detailAgent = agentId;
      this.showBrowserViewer = false;
      this.identityTab = 'identity';
      this.identityFiles = [];
      this.identityContent = {};
      this.identityEditing = false;
      this.identityEditBuffer = '';
      this.identityEditingFile = null;
      this.configEditing = false;
      this.identityLogs = null;
      this.identityLearnings = null;
      this.fetchAgentDetail(agentId);
      this.fetchIdentityFiles(agentId);
      this.fetchAgentConfig(agentId);
      this.activeTab = 'fleet';
      if (!this._skipPush) this._pushUrl(false);
    },

    // ── Chart.js rendering ────────────────────────────────

    renderCostChart() {
      const canvas = document.getElementById('costChart');
      if (!canvas) return;

      const agents = (this.costData.agents || []);
      if (agents.length === 0) {
        if (this.costChart) { this.costChart.destroy(); this.costChart = null; }
        return;
      }

      const labels = agents.map(a => a.agent);
      const costs = agents.map(a => a.cost);
      const tokens = agents.map(a => a.tokens);

      // Update existing chart in-place if possible
      if (this.costChart) {
        this.costChart.data.labels = labels;
        this.costChart.data.datasets[0].data = costs;
        this.costChart.data.datasets[1].data = tokens;
        this.costChart.update();
        return;
      }

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
        text_delta: 'text-emerald-400',
        llm_call: 'text-purple-400',
        blackboard_write: 'text-cyan-400',
        health_change: 'text-red-400',
        notification: 'text-amber-300',
        // Trace event types
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
        text_delta: 'bg-emerald-400',
        llm_call: 'bg-purple-400',
        blackboard_write: 'bg-cyan-400',
        health_change: 'bg-red-400',
        notification: 'bg-amber-300',
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

    eventTypeLabel(type) {
      const map = {
        agent_state: 'state',
        message_sent: 'msg out',
        message_received: 'msg in',
        tool_start: 'tool start',
        tool_result: 'tool result',
        text_delta: 'stream',
        llm_call: 'llm',
        blackboard_write: 'blackboard',
        health_change: 'health',
      };
      if (!type) return 'event';
      return map[type] || String(type).replace(/_/g, ' ');
    },

    eventClock(ts) {
      if (!ts) return '';
      const ms = typeof ts === 'number' ? ts * 1000 : Date.parse(ts);
      if (!Number.isFinite(ms)) return '';
      return new Date(ms).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    },

    eventDetail(evt) {
      const d = evt.data || {};
      const fields = [];
      const fmt = (v) => {
        if (v == null) return null;
        if (typeof v === 'object') return JSON.stringify(v, null, 2);
        return String(v);
      };
      const add = (label, value) => {
        const s = fmt(value);
        if (s != null && s !== '') fields.push({ label, value: s });
      };

      switch (evt.type) {
        case 'llm_call':
          add('Model', d.model);
          add('Input tokens', d.input_tokens ?? d.prompt_tokens);
          add('Output tokens', d.output_tokens ?? d.completion_tokens);
          add('Total tokens', d.total_tokens ?? d.tokens_used);
          if (d.cost_usd != null) add('Cost', '$' + d.cost_usd.toFixed(6));
          if (d.duration_ms) add('Duration', d.duration_ms + 'ms');
          add('Service', d.service);
          add('Action', d.action);
          if (d.streaming) add('Streaming', 'yes');
          if (d.prompt_preview) add('Prompt', d.prompt_preview);
          if (d.response_preview) add('Response', d.response_preview);
          break;
        case 'tool_start':
          add('Tool', d.tool || d.name);
          add('Arguments', d.arguments || d.args || d.preview || d.input);
          break;
        case 'tool_result':
          add('Tool', d.tool || d.name);
          add('Result', d.result || d.output || d.preview);
          break;
        case 'text_delta':
          add('Content', d.content);
          break;
        case 'message_sent':
        case 'message_received':
          add('Message', d.message);
          add('Source', d.source || d.from);
          add('Target', d.target || d.to);
          if (d.response_length) add('Response length', d.response_length);
          break;
        case 'health_change':
          add('Previous', d.previous);
          add('Current', d.current);
          add('Failures', d.failures);
          add('Restarts', d.restarts);
          break;
        case 'blackboard_write':
          add('Key', d.key);
          add('Version', d.version);
          add('Written by', d.written_by);
          add('Value', d.value_preview || d.value);
          break;
        case 'notification':
          add('Message', d.message);
          break;
        case 'agent_state':
          add('State', d.state);
          add('Role', d.role);
          if (Array.isArray(d.capabilities)) add('Capabilities', d.capabilities.length + ' tools');
          add('Reason', d.reason);
          if (d.ready !== undefined) add('Ready', String(d.ready));
          break;
        default:
          for (const [k, v] of Object.entries(d)) {
            add(k, v);
          }
      }

      if (evt.agent && fields.every(f => f.label !== 'Agent')) add('Agent', evt.agent);
      if (evt.timestamp) {
        const ts = typeof evt.timestamp === 'number' ? evt.timestamp * 1000 : evt.timestamp;
        add('Timestamp', new Date(ts).toLocaleString());
      }

      return fields;
    },

    eventSummary(evt) {
      const d = evt.data || {};
      switch (evt.type) {
        case 'llm_call': {
          const model = (d.model || '').split('/').pop();
          if (!model) return `${d.service || 'api'}/${d.action || '?'}${d.streaming ? ' (streaming)' : ''}`;
          const tokens = (d.total_tokens || d.tokens_used || 0).toLocaleString();
          const cost = d.cost_usd != null ? ` \u00b7 $${d.cost_usd.toFixed(4)}` : '';
          const dur = d.duration_ms ? ` \u00b7 ${d.duration_ms}ms` : '';
          const prompt = d.prompt_preview ? ` \u00b7 "${d.prompt_preview.substring(0, 40)}${d.prompt_preview.length > 40 ? '\u2026' : ''}"` : '';
          return `${model} \u00b7 ${tokens} tok${cost}${dur}${prompt}`;
        }
        case 'tool_start':
          return `${d.tool || d.name || '?'}(${(d.preview || '').substring(0, 60)})`;
        case 'tool_result':
          return `${d.tool || d.name || '?'} \u2192 ${(d.preview || d.result || d.output || '').substring(0, 60) || 'done'}`;
        case 'text_delta':
          return `${(d.content || '').substring(0, 80)}`;
        case 'message_sent':
          return `\u2192 ${(d.message || '').substring(0, 70)}`;
        case 'message_received':
          return `\u2190 ${(d.message || '').substring(0, 70)}`;
        case 'health_change':
          return `${d.previous || '?'} \u2192 ${d.current || '?'}${d.failures ? ` (${d.failures} failures)` : ''}`;
        case 'blackboard_write':
          return [d.key, d.version && `v${d.version}`, d.written_by && `by ${d.written_by}`].filter(Boolean).join(' \u00b7 ');
        case 'notification':
          return (d.message || '').substring(0, 80);
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
      const epoch = typeof ts === 'string' ? new Date(ts).getTime() / 1000 : ts;
      if (isNaN(epoch)) return '';
      const diff = Date.now() / 1000 - epoch;
      if (diff < 0) return 'just now';
      if (diff < 5) return 'just now';
      if (diff < 60) return Math.floor(diff) + 's ago';
      if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
      if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
      if (diff < 172800) return 'yesterday';
      const d = new Date(epoch * 1000);
      const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      return months[d.getMonth()] + ' ' + d.getDate();
    },

    formatCost(usd) {
      if (usd === 0 || usd == null) return '$0.00';
      if (usd < 0.01) return '$' + usd.toFixed(4);
      return '$' + usd.toFixed(2);
    },

    agentColorIndex(agentId) {
      if (!agentId) return 0;
      let hash = 0;
      for (let i = 0; i < agentId.length; i++) {
        hash = ((hash << 5) - hash) + agentId.charCodeAt(i);
        hash |= 0;
      }
      return Math.abs(hash) % 8;
    },

    agentInitials(agentId) {
      if (!agentId) return '??';
      const parts = agentId.replace(/[_-]/g, ' ').trim().split(/\s+/);
      if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
      return agentId.substring(0, 2).toUpperCase();
    },

    valueSummary(value) {
      if (value == null) return '';
      if (typeof value === 'string') return value.length > 120 ? value.substring(0, 117) + '\u2026' : value;
      if (typeof value !== 'object') return String(value);
      const fields = ['text', 'summary', 'status', 'message', 'description', 'name', 'result', 'error'];
      for (const f of fields) {
        if (value[f] != null && typeof value[f] === 'string') {
          const s = value[f];
          return s.length > 120 ? s.substring(0, 117) + '\u2026' : s;
        }
      }
      return this.truncateJson(value);
    },

    bbNamespaceOf(key) {
      const idx = key.indexOf('/');
      return idx === -1 ? '' : key.substring(0, idx + 1);
    },

    _bbNsMap: {
      'tasks/': 'tasks',
      'context/': 'context',
      'signals/': 'signals',
      'goals/': 'goals',
      'artifacts/': 'artifacts',
      'history/': 'history',
    },

    bbNsName(key) {
      return this._bbNsMap[this.bbNamespaceOf(key)] || 'default';
    },

    bbNsBadgeClass(key) {
      return 'ns-badge ns-' + this.bbNsName(key);
    },

    bbNsAccentClass(key) {
      return 'ns-accent-' + this.bbNsName(key);
    },

    bbKeyAfterNs(key) {
      const idx = key.indexOf('/');
      return idx === -1 ? key : key.substring(idx + 1);
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
      const toEpoch = (ts) => typeof ts === 'string' ? new Date(ts).getTime() / 1000 : ts;
      const minTs = toEpoch(allEvents[0].timestamp);
      const maxTs = toEpoch(allEvents[allEvents.length - 1].timestamp);
      const span = maxTs - minTs || 1;
      const left = ((toEpoch(evt.timestamp) - minTs) / span) * 100;
      const width = Math.max(2, (evt.duration_ms / 1000 / span) * 100);
      return `left:${left.toFixed(1)}%;width:${Math.min(width, 100 - left).toFixed(1)}%`;
    },
  };
}
