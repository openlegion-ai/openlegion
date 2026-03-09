/**
 * OpenLegion Dashboard — Alpine.js application.
 *
 * Three panels: Agents, Activity (Traces / Events / Logs), System (Costs / Comms / Automation / Integrations).
 * Real-time updates via WebSocket + periodic REST polling.
 */
const _IDENTITY_TABS = [
  { id: 'config', label: 'Config', file: null, access: 'user' },
  { id: 'identity', label: 'Identity', file: null, access: 'user' },
  { id: 'memory', label: 'Memory', file: null, access: 'agent' },
  { id: 'logs', label: 'Logs', file: null, access: 'auto' },
  { id: 'capabilities', label: 'Tools', file: null, access: 'auto' },
];

const _IDENTITY_FILE_MAP = {
  identity: [
    { file: 'SOUL.md', label: 'Soul', cap: 4000, access: 'both', desc: 'Personality, tone, and communication style. Shapes how the agent speaks and approaches problems.' },
    { file: 'INSTRUCTIONS.md', label: 'Instructions', cap: 8000, access: 'both', desc: 'Step-by-step procedures, workflow rules, tool patterns, and domain knowledge. The agent\'s operating manual.' },
  ],
  memory: [
    { file: 'MEMORY.md', label: 'Memory', cap: 16000, access: 'auto', desc: 'Facts and context the agent remembers across sessions. Auto-updated during conversations — you can also edit directly.' },
    { file: 'USER.md', label: 'Preferences', cap: 4000, access: 'both', desc: 'Your preferences, communication style, and project context. Helps the agent serve you better over time.' },
    { file: 'HEARTBEAT.md', label: 'Heartbeat', cap: null, access: 'both', desc: 'Rules for what the agent does during periodic autonomous wakeups — what to check, what to work on, when to notify you.' },
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
    _heartbeatCountdowns: {},
    _heartbeatTimer: null,

    // Events
    events: [],
    eventFilters: [],
    eventTypes: [
      'agent_state', 'message_sent', 'message_received',
      'tool_start', 'tool_result', 'text_delta', 'llm_call',
      'blackboard_write', 'health_change', 'notification', 'workspace_updated',
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
    addAgentForm: { name: '', role: '', model: '', avatar: 1, template: '', _showPicker: false },
    addAgentLoading: false,
    agentTemplates: [],

    // Blackboard / Comms
    bbEntries: [],
    bbPrefix: '',
    bbHighlights: [],
    bbLoading: false,
    bbWriteMode: false,
    bbNewKey: '',
    bbNewValue: '{}',
    bbWriterFilter: '',
    bbExpanded: {},
    commsView: 'activity',  // 'activity', 'state', or 'artifacts'
    commsExpanded: false,
    commsActivity: [],
    commsActivityLoading: false,
    commsSubs: {},

    // Artifacts
    artifactsList: [],
    artifactsLoading: false,
    artifactPreview: null,
    artifactPreviewContent: null,
    artifactPreviewLoading: false,

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

    // Workflows
    workflows: [],
    workflowsActive: [],
    workflowsLoading: false,

    // Settings
    settingsData: null,

    // Storage
    storageData: null,

    // Slide-over chat panel
    openChats: [],             // Array of agent IDs with open chat panels
    chatHistories: {},         // Preserved — keyed by agent ID
    chatLoadingAgents: {},     // { agentId: true/false }
    chatStreamingAgents: {},   // { agentId: true/false }
    _chatAborts: {},           // { agentId: AbortController }
    _chatFetchedAt: {},        // { agentId: timestamp } — debounce refetches
    activeChatId: '',          // Currently active chat tab
    chatPanelMinimized: false, // Whether the slide-over is minimized to pill
    chatUnread: {},            // { agentId: count } — unread notifications while minimized

    // Identity panel
    identityTabs: _IDENTITY_TABS,
    identityFileMap: _IDENTITY_FILE_MAP,
    identityTab: 'config',
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
    agentCapabilities: null,

    // Connection state
    connectionError: false,
    identityLogsLoading: false,
    identityLearnings: null,
    identityLearningsLoading: false,

    // Broadcast
    broadcastMessage: '',
    _broadcastPending: null,

    // Breadcrumb project context
    _detailReturnProject: null,

    // Identity file save flash
    identitySavedFile: null,

    // Command palette (Cmd+K)
    cmdPaletteOpen: false,
    cmdPaletteQuery: '',
    cmdPaletteResults: [],
    cmdPaletteIdx: 0,

    // System tab — sub-navigation
    systemTab: 'costs',
    systemTabs: [
      { id: 'costs', label: 'Costs & Budgets' },
      { id: 'automation', label: 'Automation' },
      { id: 'integrations', label: 'Integrations' },
    ],

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
    credAuthType: 'api_key',

    // Channels
    channels: [],
    channelConnectType: '',
    channelTokens: {},


    // Webhooks
    webhooks: [],
    showWebhookForm: false,
    webhookFormName: '',
    webhookFormAgent: '',
    webhookFormSecret: '',

    // Model health
    modelHealth: [],

    // Cron creation
    showCronForm: false,
    cronFormAgent: '',
    cronFormSchedule: 'every 15m',
    cronFormMessage: '',
    cronCreating: false,

    // Credential update
    editingCredential: null,
    editCredKey: '',
    credentialSaving: false,

    // Workflow cancel tracking
    _cancellingWorkflows: {},

    // Restart loading
    _restartingAgents: {},

    // Loading states for double-submit prevention
    bbWriteLoading: false,
    cronRunLoading: {},
    cronPauseLoading: {},
    cronEditSaving: false,
    workflowRunLoading: {},
    credSaving: false,
    onboardSaving: false,
    channelConnecting: false,
    webhookCreating: false,
    webhookTesting: {},
    broadcastSending: false,
    confirmLoading: false,

    // Onboarding
    onboardProvider: '',
    onboardKey: '',
    onboardBaseUrl: '',
    onboardAuthType: 'api_key',

    // WebSocket reconnect countdown (Alpine-reactive mirror)
    wsReconnectIn: 0,

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
    _queuePollInterval: null,
    _cronInterval: null,
    _seenEventIds: new Set(),
    _initTime: new Date().toISOString(),  // page-load timestamp for replay filtering

    // URL routing
    _skipPush: false,
    _popstateHandler: null,

    // ── URL Routing ──────────────────────────────────────────

    _buildPath() {
      if (this.detailAgent) {
        const tab = this.identityTab || 'config';
        return tab === 'config'
          ? `/agents/${this.detailAgent}`
          : `/agents/${this.detailAgent}/${tab}`;
      }
      if (this.activeTab === 'activity') {
        if (this.activityView === 'events') return '/activity/events';
        if (this.activityView === 'logs') return '/activity/logs';
        return '/activity';
      }
      if (this.activeTab === 'system') return '/system/' + (this.systemTab || 'costs');
      return '/';
    },

    _buildTitle() {
      if (this.detailAgent) {
        const tabLabel = (_IDENTITY_TABS.find(t => t.id === this.identityTab) || _IDENTITY_TABS[0]).label;
        return `${this.detailAgent} \u00b7 ${tabLabel} \u2014 OpenLegion`;
      }
      if (this.activeTab === 'activity') {
        if (this.activityView === 'events') return 'Events \u2014 OpenLegion';
        if (this.activityView === 'logs') return 'Logs \u2014 OpenLegion';
        return 'Traces \u2014 OpenLegion';
      }
      if (this.activeTab === 'system') {
        const st = this.systemTabs.find(t => t.id === this.systemTab);
        return (st ? st.label : 'System') + ' \u2014 OpenLegion';
      }
      return 'Agents \u2014 OpenLegion';
    },

    _parsePath(path) {
      const clean = path.replace(/^\/+/, '').replace(/\/+$/, '');
      const route = { tab: 'fleet', activityView: 'traces', systemTab: 'costs', agentId: null, identityTab: 'config' };
      if (!clean) return route;

      const agentMatch = clean.match(/^agents\/([^/]+)(?:\/([^/]+))?$/);
      if (agentMatch) {
        route.agentId = agentMatch[1];
        const tab = agentMatch[2];
        if (tab && _IDENTITY_TABS.some(t => t.id === tab)) route.identityTab = tab;
        return route;
      }

      if (clean === 'activity/events') { route.tab = 'activity'; route.activityView = 'events'; }
      else if (clean === 'activity/logs') { route.tab = 'activity'; route.activityView = 'logs'; }
      else if (clean === 'activity') { route.tab = 'activity'; }
      else if (clean.startsWith('system')) {
        route.tab = 'system';
        const sub = clean.split('/')[1];
        if (sub && ['costs', 'automation', 'integrations'].includes(sub)) route.systemTab = sub;
      }
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
          if (route.tab === 'system' && this.systemTab !== route.systemTab) {
            this.systemTab = route.systemTab;
            if (route.systemTab === 'integrations') {
              this.fetchChannels(); this.fetchWebhooks(); this.fetchSettings();
            }
          }
        }
      } finally {
        this._skipPush = false;
      }
    },

    closeDetail() {
      this.detailAgent = null;
      this.selectedAgent = null;
      if (this._detailReturnProject !== null && this._detailReturnProject !== undefined) {
        this.activeProject = this._detailReturnProject;
      }
      this._detailReturnProject = null;
      this._pushUrl(false);
    },

    // ── Confirm modal ─────────────────────────────────────

    showConfirm(title, message, action, destructive = true) {
      this.confirmModal = { open: true, title, message, action, destructive };
    },

    cancelConfirm() {
      if (this.confirmLoading) return;
      this.confirmModal = { open: false, title: '', message: '', action: null, destructive: false };
    },

    async executeConfirm() {
      if (this.confirmLoading) return;
      this.confirmLoading = true;
      try {
        if (this.confirmModal.action) await this.confirmModal.action();
      } catch (e) {
        this.showToast(`Error: ${e.message || String(e)}`);
      } finally {
        this.confirmLoading = false;
        this.confirmModal = { open: false, title: '', message: '', action: null, destructive: false };
      }
    },

    // ── Computed ───────────────────────────────────────────

    get showOnboarding() {
      if (this.loading) return false;
      // Show onboarding when settings haven't loaded yet (fallback) or when
      // credentials are missing / no agents exist — prevents blank page.
      if (!this.settingsData) return this.agents.length === 0;
      return !this.settingsData.has_llm_credentials || this.agents.length === 0;
    },

    get addAgentNameValid() {
      const name = this.addAgentForm.name;
      if (!name) return true; // empty is valid (not yet typed)
      return /^[a-z][a-z0-9_]{0,29}$/.test(name);
    },

    get maxAgents() {
      return this.settingsData?.plan_limits?.max_agents ?? 0;
    },
    get maxProjects() {
      return this.settingsData?.plan_limits?.max_projects ?? 0;
    },
    get runningAgents() {
      return this.agents.filter(a => a.running !== false);
    },
    get atAgentLimit() {
      if (this.maxAgents === 0) return false;
      return this.runningAgents.length >= this.maxAgents;
    },
    get projectsEnabled() {
      const limits = this.settingsData?.plan_limits;
      if (!limits) return true; // no limits loaded yet, allow everything
      return limits.projects_enabled !== false;
    },
    get atProjectLimit() {
      if (!this.projectsEnabled) return true;
      if (this.maxProjects === 0) return false; // unlimited
      const projectCount = this.projects?.length ?? 0;
      return projectCount >= this.maxProjects;
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
      const counts = { healthy: 0, unhealthy: 0, failed: 0, unknown: 0, stopped: 0 };
      for (const a of this.filteredAgents) {
        const s = a.health_status || 'unknown';
        if (s === 'stopped') counts.stopped++;
        else if (s === 'healthy') counts.healthy++;
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

    get isMac() {
      return /mac/i.test(navigator.userAgentData?.platform || navigator.platform || '');
    },

    isFileDefault(file) {
      const info = this.identityFiles.find(f => f.name === file);
      return info ? info.is_default : true;
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
      if (this.activityTimeRange && this.activityTimeRange !== 'all') {
        const hoursMap = { '1h': 1, '6h': 6, '24h': 24 };
        const hours = hoursMap[this.activityTimeRange];
        if (hours) {
          const cutoff = Date.now() / 1000 - hours * 3600;
          items = items.filter(t => {
            const ts = typeof t.started === 'string' ? new Date(t.started).getTime() / 1000 : t.started;
            return ts >= cutoff;
          });
        }
      }
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

    get bbJsonValid() {
      try {
        JSON.parse(this.bbNewValue);
        return { valid: true, error: '' };
      } catch (e) {
        return { valid: false, error: e.message };
      }
    },

    get modelsDown() {
      return this.modelHealth.filter(m => !m.available).length;
    },

    // ── Lifecycle ─────────────────────────────────────────

    init() {
      const cfg = window.__config || {};
      this._ws = new DashboardWebSocket(cfg.wsUrl, {
        onEvent: (evt) => this.onWsEvent(evt),
        onConnect: () => { this.connected = true; },
        onDisconnect: () => { this.connected = false; },
        onReconnectTick: (secs) => { this.wsReconnectIn = secs; },
      });
      this._ws.connect();

      this.fetchAgents();
      this.startHeartbeatTimer();
      this.fetchSettings();
      this.fetchProject();
      this.fetchProjects();
      this.fetchModelHealth();
      this.fetchQueues();
      this._refreshInterval = setInterval(() => this.fetchAgents(), 15000);
      this._modelHealthInterval = setInterval(() => this.fetchModelHealth(), 60000);
      this._queuePollInterval = setInterval(() => this.fetchQueues(), 2000);

      // Renew session cookie every 6 hours (sliding window).
      // The auth gate issues 24h cookies; renew when <12h remaining.
      // Also renew on tab focus (browsers throttle timers in background tabs).
      this._cookieRenewalInterval = setInterval(() => {
        fetch('/__auth/renew', { credentials: 'same-origin' }).catch(() => {});
      }, 6 * 3600 * 1000);
      this._visibilityHandler = () => {
        if (document.visibilityState === 'visible') {
          fetch('/__auth/renew', { credentials: 'same-origin' }).catch(() => {});
        }
      };
      document.addEventListener('visibilitychange', this._visibilityHandler);

      // Restore chat history from localStorage (persists across tabs & browser restarts)
      try {
        const saved = localStorage.getItem('ol_chats');
        if (saved) {
          const parsed = JSON.parse(saved);
          if (parsed.histories) this.chatHistories = parsed.histories;
          if (parsed.openChats) this.openChats = parsed.openChats;
          if (parsed.activeChatId) this.activeChatId = parsed.activeChatId;
        }
      } catch (e) {
        console.debug('chat history fetch skipped:', e.message || e);
      }

      // Sync restored open chats from server so cross-device history is fresh
      for (const agentId of this.openChats) {
        this._loadChatHistory(agentId);
      }

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
      };
      document.addEventListener('keydown', this._cmdPaletteHandler);

      // Normalize legacy /dashboard/ URL to root
      if (/^\/dashboard\/?$/.test(window.location.pathname)) {
        history.replaceState(null, '', '/');
      }

      // Deep link restoration: parse initial URL and apply route
      const initRoute = this._parsePath(window.location.pathname);
      const isDeepLink = initRoute.agentId || initRoute.tab !== 'fleet' || initRoute.activityView !== 'traces' || initRoute.systemTab !== 'costs';
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
          if (this._queuePollInterval) { clearInterval(this._queuePollInterval); this._queuePollInterval = null; }
          if (this._cronInterval) { clearInterval(this._cronInterval); this._cronInterval = null; }
          if (this._modelHealthInterval) { clearInterval(this._modelHealthInterval); this._modelHealthInterval = null; }
          this._stopActivityRefresh();
        } else {
          // Clear favicon badge
          document.title = document.title.replace(/^\(\d+\)\s*/, '');
          // Resume polling — restore intervals without navigating (switchTab clears detailAgent)
          this._refreshInterval = setInterval(() => this.fetchAgents(), 15000);
          this.fetchAgents();
          if (this.activeTab === 'activity' && this.activityView === 'traces') {
            this.fetchTraces();
            this._startActivityRefresh();
          }
          if (this.activeTab === 'activity' && this.activityView === 'logs') {
            this.fetchSystemLogs();
          }
          if (this.activeTab === 'system') {
            this.fetchCronJobs();
            this.fetchStorage();
            this._cronInterval = setInterval(() => this.fetchCronJobs(), 10000);
          }
          // Resume model health + queue polling
          this.fetchModelHealth();
          this._modelHealthInterval = setInterval(() => this.fetchModelHealth(), 60000);
          this.fetchQueues();
          this._queuePollInterval = setInterval(() => this.fetchQueues(), 2000);
          // Refresh agent detail if we're viewing one
          if (this.detailAgent) {
            this.fetchAgentDetail(this.detailAgent);
          }
          // Sync open chat histories from server (cross-device consistency)
          for (const agentId of this.openChats) {
            delete this._chatFetchedAt[agentId];
            this._loadChatHistory(agentId);
          }
        }
      });
    },

    destroy() {
      if (this._ws) this._ws.disconnect();
      if (this._refreshInterval) clearInterval(this._refreshInterval);
      if (this._queuePollInterval) clearInterval(this._queuePollInterval);
      if (this._cronInterval) clearInterval(this._cronInterval);
      if (this._modelHealthInterval) clearInterval(this._modelHealthInterval);
      if (this._cookieRenewalInterval) clearInterval(this._cookieRenewalInterval);
      if (this._visibilityHandler) document.removeEventListener('visibilitychange', this._visibilityHandler);
      if (this._costDebounce) clearTimeout(this._costDebounce);
      if (this._fleetDebounce) clearTimeout(this._fleetDebounce);
      if (this._activityRefresh) clearInterval(this._activityRefresh);
      if (this._tracesDebounce) clearTimeout(this._tracesDebounce);
      if (this.costChart) this.costChart.destroy();
      Object.values(this._stateTimers).forEach(clearTimeout);
      Object.values(this._scrollTimers).forEach(clearTimeout);
      Object.values(this._chatAborts).forEach(c => c?.abort());
      this.stopHeartbeatTimer();
      if (this._cmdPaletteHandler) document.removeEventListener('keydown', this._cmdPaletteHandler);
      if (this._popstateHandler) window.removeEventListener('popstate', this._popstateHandler);
    },

    // ── Tab switching ─────────────────────────────────────

    switchTab(tab) {
      this.activeTab = tab;
      this.detailAgent = null;
      // Clear tab-specific auto-refresh intervals
      if (this._cronInterval) { clearInterval(this._cronInterval); this._cronInterval = null; }
      this._stopActivityRefresh();
      if (tab === 'activity') {
        this.unreadEvents = 0;
        this._lastSeenEventCount = this.events.length;
        if (this.activityView === 'traces') {
          this.fetchTraces();
          this._startActivityRefresh();
        } else if (this.activityView === 'logs') {
          this.fetchSystemLogs();
        }
      }
      if (tab === 'fleet') {
        this.fetchAgents();
        this.fetchQueues();
        this.fetchSettings();
        this.fetchProject();
        this.fetchProjects();
      }
      if (tab === 'system') {
        this.fetchSettings();
        this.fetchCosts();
        this.fetchStorage();
        this.fetchCronJobs();
        this.fetchWorkflows();
        if (this.systemTab === 'integrations') {
          this.fetchWebhooks();
          this.fetchChannels();
        }
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

    dismissToast(id) {
      this.toastQueue = this.toastQueue.filter(t => t.id !== id);
    },

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

    healthLabel(status) {
      const map = { healthy: 'Online', unhealthy: 'Degraded', restarting: 'Degraded', failed: 'Offline', unknown: 'Starting' };
      return map[status] || 'Starting';
    },

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

      // Favicon badge: prefix document title with unread count when tab not visible
      if (document.hidden && this.unreadEvents > 0) {
        const base = document.title.replace(/^\(\d+\)\s*/, '');
        document.title = `(${this.unreadEvents}) ${base}`;
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

      // Refresh model health on health_change events
      if (evt.type === 'health_change') {
        this.fetchModelHealth();
      }

      // Show toast for agent notifications + inject into chat panel.
      // Skip replayed events (timestamp before page load) to avoid
      // re-showing old toasts and duplicating chat history on reconnect.
      if (evt.type === 'notification' && evt.agent) {
        const isReplay = evt.timestamp && evt.timestamp < this._initTime;
        if (!isReplay) {
          this.showToast(`[${evt.agent}] ${(evt.data?.message || '').substring(0, 120)}`);
          if (!this.chatHistories[evt.agent]) this.chatHistories[evt.agent] = [];
          this.chatHistories[evt.agent].push({
            role: 'notification',
            content: evt.data?.message || '',
            streaming: false,
            tools: [],
            ts: Date.now(),
          });
          this._saveChatToSession();
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
      }

      // Highlight blackboard writes + update comms badge
      if (evt.type === 'blackboard_write' && evt.data && evt.data.key) {
        if (!this.bbHighlights.includes(evt.data.key)) this.bbHighlights.push(evt.data.key);
        setTimeout(() => { const i = this.bbHighlights.indexOf(evt.data.key); if (i !== -1) this.bbHighlights.splice(i, 1); }, 5000);
        if (this.activeProject && this.activeTab === 'fleet' && !this.detailAgent) {
          if (this._commsDebounce) clearTimeout(this._commsDebounce);
          this._commsDebounce = setTimeout(() => {
            this.fetchBlackboard();
            this.fetchCommsActivity();
            if (evt.data.key.includes('artifacts/')) this.fetchArtifacts();
          }, 1000);
        }
      }

      // Re-fetch identity/memory content when workspace files change (debounced, skip if editing)
      if (evt.type === 'workspace_updated' && evt.agent) {
        const viewing = this.selectedAgent || this.detailAgent;
        if (viewing === evt.agent && !this.identityEditing) {
          if (this._wsUpdateDebounce) clearTimeout(this._wsUpdateDebounce);
          this._wsUpdateDebounce = setTimeout(() => this.loadIdentityTabContent(viewing), 1000);
        }
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
      if (view === 'logs') {
        this.fetchSystemLogs();
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
          this.connectionError = false;
          // Prune restored chat tabs for agents that no longer exist
          const agentIds = new Set(this.agents.map(a => a.id));
          const stale = this.openChats.filter(id => !agentIds.has(id));
          if (stale.length) {
            this.openChats = this.openChats.filter(id => agentIds.has(id));
            if (this.activeChatId && !agentIds.has(this.activeChatId)) {
              this.activeChatId = this.openChats[0] || null;
            }
            this._saveChatToSession();
          }
        }
      } catch (e) {
        console.warn('fetchAgents failed:', e);
        this.connectionError = true;
      }
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
      // Composite tabs — load all mapped files (always refetch for freshness)
      const fileMap = _IDENTITY_FILE_MAP[tab.id];
      if (fileMap) {
        this.identityContentLoading = true;
        for (const entry of fileMap) {
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
        await this.fetchIdentityLogs(agentId);
        await this.fetchIdentityLearnings(agentId);
      }
      if (tab.id === 'capabilities') {
        await this.fetchAgentCapabilities(agentId);
      }
    },

    async fetchAgentCapabilities(agentId) {
      this.agentCapabilities = null;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/capabilities`);
        if (resp.ok) {
          const data = await resp.json();
          // Agent returns tool_definitions (OpenAI format: {type, function: {name, description}})
          const defs = data.tool_definitions || [];
          this.agentCapabilities = defs.map(t => ({
            name: t.function?.name || t.name || '?',
            description: t.function?.description || t.description || '',
          }));
        }
      } catch (e) { console.warn('fetchAgentCapabilities failed:', e); }
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
      if (this.identitySaving) return;
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
          this.identitySavedFile = file;
          setTimeout(() => { this.identitySavedFile = null; }, 2000);
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
      if (this.projectSaving) return;
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
      this.commsView = 'activity';
      this.commsExpanded = false;
      this.bbPrefix = '';
      this.bbWriteMode = false;
      this.bbExpanded = {};
      this.artifactPreview = null;
      this.commsActivity = [];
      this.bbEntries = [];
      this.commsSubs = {};
      this.artifactsList = [];
      this.fetchProject();
      if (name) {
        this.fetchCommsActivity();
        this.fetchBlackboard();
        this.fetchArtifacts();
      }
    },

    openProjectModal() {
      if (this.atProjectLimit) return;
      this.showProjectForm = true;
      this.$nextTick(() => {
        const el = document.getElementById('project-name-input');
        if (el) el.focus();
      });
    },

    closeProjectModal() {
      if (this.projectFormLoading) return;
      this.showProjectForm = false;
      this.newProjectName = '';
      this.newProjectDesc = '';
    },

    async createProject() {
      if (!this.projectsEnabled) {
        this.showToast('Projects are not available on your current plan.');
        return;
      }
      if (this.atProjectLimit) {
        this.showToast('Project limit reached. Upgrade your plan for more projects.');
        return;
      }
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
          this.projectFormLoading = false;
          this.closeProjectModal();
          await this.fetchProjects();
          this.switchProject(name);
          this.showToast(`Project "${name}" created`);
        } else {
          const err = await resp.json().catch(() => ({}));
          if (resp.status === 403) {
            this.showToast(err.detail || 'Upgrade your plan.');
            this.fetchSettings(); // refresh limits so UI updates
          } else {
            this.showToast(`Create failed: ${err.detail || 'Unknown error'}`);
          }
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


    _bbProjectPrefix() {
      // Scope blackboard keys to the active project
      return this.activeProject ? `projects/${this.activeProject}/` : '';
    },

    _bbStripProjectPrefix(key) {
      const pfx = this._bbProjectPrefix();
      if (pfx && key.startsWith(pfx)) return key.slice(pfx.length);
      // If no active project, still strip any projects/*/  prefix for display
      const m = key.match(/^projects\/[^/]+\/(.*)/);
      return m ? m[1] : key;
    },

    async fetchBlackboard() {
      this.bbLoading = true;
      try {
        // Prepend project prefix so namespace filters match project-scoped keys
        const searchPrefix = this._bbProjectPrefix() + this.bbPrefix;
        const resp = await fetch(`${window.__config.apiBase}/blackboard?prefix=${encodeURIComponent(searchPrefix)}`);
        if (resp.ok) {
          const entries = (await resp.json()).entries;
          // Strip project prefix so display/namespace logic works on the real key
          for (const e of entries) e.key = this._bbStripProjectPrefix(e.key);
          this.bbEntries = entries;
        }
      } catch (e) { console.warn('fetchBlackboard failed:', e); }
      this.bbLoading = false;
    },

    async fetchCommsActivity() {
      this.commsActivityLoading = true;
      try {
        const proj = this.activeProject;
        const params = new URLSearchParams({ limit: '100' });
        if (proj) params.set('project', proj);
        const resp = await fetch(`${window.__config.apiBase}/comms/activity?${params}`);
        if (resp.ok) {
          const data = await resp.json();
          const activity = data.activity || [];
          // Strip project prefix from blackboard keys and pubsub topics
          for (const item of activity) {
            if (item.key) item.key = this._bbStripProjectPrefix(item.key);
            if (item.topic) item.topic = this._bbStripProjectPrefix(item.topic);
          }
          this.commsActivity = activity;
          // Strip project prefix from subscription topic names
          const rawSubs = data.subscriptions || {};
          const subs = {};
          const pfx = this._bbProjectPrefix();
          for (const [topic, agents] of Object.entries(rawSubs)) {
            const clean = pfx && topic.startsWith(pfx) ? topic.slice(pfx.length) : topic;
            subs[clean] = agents;
          }
          this.commsSubs = subs;
        }
      } catch (e) { console.warn('fetchCommsActivity failed:', e); }
      this.commsActivityLoading = false;
    },

    async fetchArtifacts() {
      this.artifactsLoading = true;
      try {
        // Gather artifacts from all agents in the current project
        const proj = this.activeProject;
        if (!proj) { this.artifactsList = []; this.artifactsLoading = false; return; }
        const projectAgents = this.agents.filter(a => a.project === proj);
        const results = await Promise.allSettled(
          projectAgents.map(async (a) => {
            const resp = await fetch(`${window.__config.apiBase}/agents/${a.id}/artifacts`, { credentials: 'same-origin' });
            if (!resp.ok) return [];
            const data = await resp.json();
            return (data.artifacts || []).map(art => ({ ...art, agent: a.id }));
          })
        );
        const all = [];
        for (const r of results) {
          if (r.status === 'fulfilled') all.push(...r.value);
        }
        all.sort((a, b) => (b.modified || 0) - (a.modified || 0));
        this.artifactsList = all;
      } catch (e) { console.warn('fetchArtifacts failed:', e); }
      this.artifactsLoading = false;
    },

    async previewArtifact(art) {
      this.artifactPreview = art;
      this.artifactPreviewContent = null;
      this.artifactPreviewLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${art.agent}/artifacts/${encodeURIComponent(art.name)}`, { credentials: 'same-origin' });
        if (!resp.ok) {
          this.artifactPreviewContent = `Error loading artifact: ${resp.status}`;
          return;
        }
        const data = await resp.json();
        this.artifactPreviewContent = data.encoding === 'base64'
          ? atob(data.content)
          : data.content;
      } catch (e) {
        this.artifactPreviewContent = `Failed to load: ${e.message || e}`;
      }
      this.artifactPreviewLoading = false;
    },

    async downloadArtifact(art) {
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${art.agent}/artifacts/${encodeURIComponent(art.name)}`, { credentials: 'same-origin' });
        if (!resp.ok) { this.showToast('Download failed: ' + resp.status); return; }
        const data = await resp.json();
        let blob;
        if (data.encoding === 'base64') {
          const bytes = Uint8Array.from(atob(data.content), c => c.charCodeAt(0));
          blob = new Blob([bytes], { type: data.mime_type || 'application/octet-stream' });
        } else {
          blob = new Blob([data.content], { type: data.mime_type || 'text/plain' });
        }
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = art.name.split('/').pop();
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } catch (e) {
        this.showToast('Download failed: ' + (e.message || e));
      }
    },

    async deleteArtifact(art) {
      if (!confirm(`Delete artifact "${art.name}" from ${art.agent}?`)) return;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${art.agent}/artifacts/${encodeURIComponent(art.name)}`, {
          method: 'DELETE', credentials: 'same-origin',
        });
        if (resp.ok) {
          this.showToast(`Deleted ${art.name}`);
          this.artifactsList = this.artifactsList.filter(a => !(a.agent === art.agent && a.name === art.name));
          if (this.artifactPreview?.agent === art.agent && this.artifactPreview?.name === art.name) {
            this.artifactPreview = null;
          }
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Delete failed: ${err.detail || resp.status}`);
        }
      } catch (e) {
        this.showToast('Delete failed: ' + (e.message || e));
      }
    },

    artifactIsText(name) {
      const textExts = ['.md', '.txt', '.json', '.csv', '.yaml', '.yml', '.xml', '.html', '.css', '.js', '.ts', '.py', '.sh', '.sql', '.log', '.env', '.toml', '.ini', '.cfg'];
      return textExts.some(ext => name.toLowerCase().endsWith(ext));
    },

    formatFileSize(bytes) {
      if (bytes === 0) return '0 B';
      const units = ['B', 'KB', 'MB'];
      const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
      const val = bytes / Math.pow(1024, i);
      return (i === 0 ? val : val.toFixed(1)) + ' ' + units[i];
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
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
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
        avatar: cfg.avatar || 1,
        _showAvatarPicker: false,
        budget_daily: cfg.budget?.daily_usd || '',
        budget_monthly: cfg.budget?.monthly_usd || '',
        thinking: cfg.thinking || 'off',
        allowed_credentials: credsStr,
        _credMode: credMode,
      };
      this.configEditing = true;
    },

    cancelConfigEdit() {
      this.configEditing = false;
      this.editForm = {};
    },

    thinkingOptionsForModel(model) {
      if (!model) return [];
      if (model.startsWith('anthropic/')) {
        return [
          { value: 'off', label: 'Off' },
          { value: 'low', label: 'Low (5K budget tokens)' },
          { value: 'medium', label: 'Medium (10K budget tokens)' },
          { value: 'high', label: 'High (25K budget tokens)' },
        ];
      }
      if (model.startsWith('openai/o') || model.startsWith('o1') || model.startsWith('o3') || model.startsWith('o4')) {
        return [
          { value: 'off', label: 'Off' },
          { value: 'low', label: 'Low' },
          { value: 'medium', label: 'Medium' },
          { value: 'high', label: 'High' },
        ];
      }
      return [];
    },

    async saveConfigFromDetail(agentId) {
      if (this.configSaving) return;
      this.configSaving = true;
      try {
        await this.saveAgentConfig(agentId);
        await this.fetchAgentDetail(agentId);
      } catch (e) {
        this.showToast(`Error: ${e.message || String(e)}`);
      } finally {
        this.configSaving = false;
      }
    },

    async saveAgentConfig(agentId) {
      const body = {};
      const cfg = this.agentConfigs[agentId] || {};
      if (this.editForm.model && this.editForm.model !== cfg.model) body.model = this.editForm.model;
      if (this.editForm.role !== undefined && this.editForm.role !== cfg.role) body.role = this.editForm.role;
      if (this.editForm.avatar && this.editForm.avatar !== (cfg.avatar || 1)) body.avatar = this.editForm.avatar;
      if ((this.editForm.budget_daily && parseFloat(this.editForm.budget_daily) > 0) ||
          (this.editForm.budget_monthly && parseFloat(this.editForm.budget_monthly) > 0)) {
        const budget = {};
        if (this.editForm.budget_daily && parseFloat(this.editForm.budget_daily) > 0)
          budget.daily_usd = parseFloat(this.editForm.budget_daily);
        if (this.editForm.budget_monthly && parseFloat(this.editForm.budget_monthly) > 0)
          budget.monthly_usd = parseFloat(this.editForm.budget_monthly);
        body.budget = budget;
      }
      if (this.editForm.thinking !== undefined && this.editForm.thinking !== (cfg.thinking || 'off')) {
        body.thinking = this.editForm.thinking;
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
        if (credsChanged && !configResult?.restart_required) {
          this.showToast(`Credentials updated for ${agentId} — restart may be needed`);
        }
        if (configResult && configResult.restart_required) {
          this.showToast(`Updated: ${allUpdated.join(', ')} — restarting ${agentId}...`);
          const restartResp = await fetch(`${window.__config.apiBase}/agents/${agentId}/restart`, { method: 'POST' });
          if (restartResp.ok) {
            const data = await restartResp.json();
            this.showToast(data.ready ? `${agentId} restarted and ready` : `${agentId} restarted (warming up)`);
            this.fetchAgents();
          } else {
            this.showToast(`Config updated but restart failed — restart ${agentId} manually`);
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
        this._restartingAgents = { ...this._restartingAgents, [agentId]: true };
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
        const copy = { ...this._restartingAgents };
        delete copy[agentId];
        this._restartingAgents = copy;
      }, true);
    },

    async toggleHeartbeat(agent) {
      const jobId = agent.heartbeat_job_id;
      if (!jobId || this.cronPauseLoading[jobId]) return;
      this.cronPauseLoading = { ...this.cronPauseLoading, [jobId]: true };
      const action = agent.heartbeat_enabled ? 'pause' : 'resume';
      try {
        const resp = await fetch(`${window.__config.apiBase}/cron/${jobId}/${action}`, { method: 'POST' });
        if (resp.ok) {
          this.showToast(`${agent.id} heartbeat ${action}d`);
          this.fetchAgents();
          this.fetchCronJobs();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || action + ' failed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.cronPauseLoading = { ...this.cronPauseLoading, [jobId]: false }; }
    },

    async updateBudget(agentId, dailyUsd, monthlyUsd) {
      try {
        const body = {};
        if (dailyUsd != null) body.daily_usd = parseFloat(dailyUsd);
        if (monthlyUsd != null) body.monthly_usd = parseFloat(monthlyUsd);
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/budget`, {
          method: 'PUT', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body),
        });
        if (resp.ok) this.showToast(`Budget updated for ${agentId}`);
      } catch (e) { console.warn('updateBudget failed:', e); }
    },

    // ── Add / Remove agents ──────────────────────────────

    async addAgent() {
      if (this.atAgentLimit) {
        this.showToast('Agent limit reached. Upgrade your plan for more agents.');
        return;
      }
      const f = this.addAgentForm;
      if (!f.name.trim()) { this.showToast('Name is required'); return; }
      if (!this.addAgentNameValid) { this.showToast('Invalid name: lowercase letters, numbers, underscores only. Max 30 chars.'); return; }
      this.addAgentLoading = true;
      try {
        const payload = {
          name: f.name.trim(),
          role: f.role.trim(),
          model: f.model,
          avatar: f.avatar || 1,
        };
        if (f.template) payload.template = f.template;
        const resp = await fetch(`${window.__config.apiBase}/agents`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.showToast(data.ready ? `${data.agent} added and ready` : `${data.agent} added (starting)`);
          this.addAgentMode = false;
          this.addAgentForm = { name: '', role: '', model: '', avatar: 1, template: '', _showPicker: false };
          this.fetchAgents();
        } else {
          const err = await resp.json();
          if (resp.status === 403) {
            this.showToast(err.detail || 'Upgrade your plan.');
            this.fetchSettings(); // refresh limits so UI updates
          } else {
            this.showToast(`Error: ${err.detail || 'Failed to add agent'}`);
          }
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      this.addAgentLoading = false;
    },

    openAddAgentModal(defaultName = '') {
      if (this.atAgentLimit) return;
      this.addAgentMode = true;
      if (defaultName) this.addAgentForm.name = defaultName;
      this.fetchSettings();
      this.fetchAgentTemplates();
      this.$nextTick(() => {
        const el = document.getElementById('add-agent-name-input');
        if (el) el.focus();
      });
    },

    closeAddAgentModal() {
      if (this.addAgentLoading) return;
      this.addAgentMode = false;
      this.addAgentForm = { name: '', role: '', model: '', avatar: 1, template: '', _showPicker: false };
    },

    async fetchAgentTemplates() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/agent-templates`);
        if (resp.ok) this.agentTemplates = await resp.json();
      } catch (e) { /* ignore */ }
    },

    applyAgentTemplate() {
      const tpl = this.agentTemplates.find(t => t.id === this.addAgentForm.template);
      if (tpl) {
        this.addAgentForm.role = tpl.role;
      } else {
        // Switched back to "Blank agent" — clear auto-filled role
        this.addAgentForm.role = '';
      }
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
      if (this.bbWriteLoading) return;
      if (!this.bbNewKey.trim()) return;
      let value;
      try { value = JSON.parse(this.bbNewValue); } catch (_) { this.showToast('Invalid JSON'); return; }
      this.bbWriteLoading = true;
      try {
        const fullKey = this._bbProjectPrefix() + this.bbNewKey;
        const resp = await fetch(`${window.__config.apiBase}/blackboard/${fullKey}`, {
          method: 'PUT', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ value }),
        });
        if (resp.ok) {
          this.showToast(`Written: ${this.bbNewKey}`);
          this.bbNewKey = ''; this.bbNewValue = '{}'; this.bbWriteMode = false;
          this.fetchBlackboard();
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.bbWriteLoading = false; }
    },

    async deleteBlackboard(key) {
      this.showConfirm('Delete Entry', `Delete blackboard key "${key}"?`, async () => {
        try {
          const fullKey = this._bbProjectPrefix() + key;
          const resp = await fetch(`${window.__config.apiBase}/blackboard/${fullKey}`, { method: 'DELETE' });
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
      if (this.cronRunLoading[jobId]) return;
      this.cronRunLoading = { ...this.cronRunLoading, [jobId]: true };
      try {
        const resp = await fetch(`${window.__config.apiBase}/cron/${jobId}/run`, { method: 'POST' });
        if (resp.ok) {
          this.showToast(`Job ${jobId} triggered`);
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Run failed'}`);
        }
        this.fetchCronJobs();
      } catch (e) { console.warn('runCronJob failed:', e); }
      finally { this.cronRunLoading = { ...this.cronRunLoading, [jobId]: false }; }
    },

    async pauseCronJob(jobId) {
      if (this.cronPauseLoading[jobId]) return;
      this.cronPauseLoading = { ...this.cronPauseLoading, [jobId]: true };
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
      finally { this.cronPauseLoading = { ...this.cronPauseLoading, [jobId]: false }; }
    },

    async resumeCronJob(jobId) {
      if (this.cronPauseLoading[jobId]) return;
      this.cronPauseLoading = { ...this.cronPauseLoading, [jobId]: true };
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
      finally { this.cronPauseLoading = { ...this.cronPauseLoading, [jobId]: false }; }
    },

    async fetchWorkflows() {
      this.workflowsLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/workflows`);
        if (resp.ok) {
          const data = await resp.json();
          this.workflows = data.workflows || [];
          this.workflowsActive = data.active || [];
        }
      } catch (e) { console.warn('fetchWorkflows failed:', e); }
      this.workflowsLoading = false;
    },

    async fetchModelHealth() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/model-health`);
        if (resp.ok) this.modelHealth = (await resp.json()).models || [];
      } catch (e) { console.warn('fetchModelHealth failed:', e); }
    },

    async cancelWorkflow(executionId) {
      if (this._cancellingWorkflows[executionId]) return;
      this._cancellingWorkflows = { ...this._cancellingWorkflows, [executionId]: true };
      try {
        const resp = await fetch(`${window.__config.apiBase}/workflows/${encodeURIComponent(executionId)}/cancel`, { method: 'POST' });
        if (resp.ok) {
          this.showToast(`Workflow execution cancelled`);
          await this.fetchWorkflows();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Cancel failed: ${err.detail || 'Unknown error'}`);
        }
      } catch (e) { this.showToast(`Cancel failed: ${e.message}`); }
      finally { const copy = { ...this._cancellingWorkflows }; delete copy[executionId]; this._cancellingWorkflows = copy; }
    },

    async createCronJob() {
      if (this.cronCreating) return;
      const agent = this.cronFormAgent.trim();
      const schedule = this.cronFormSchedule.trim();
      const message = this.cronFormMessage.trim();
      if (!agent || !schedule || !message) {
        this.showToast('Agent, schedule, and message are required');
        return;
      }
      this.cronCreating = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/cron`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ agent, schedule, message }),
        });
        if (resp.ok) {
          this.showToast('Cron job created');
          this.cronFormAgent = '';
          this.cronFormSchedule = 'every 15m';
          this.cronFormMessage = '';
          this.showCronForm = false;
          await this.fetchCronJobs();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Failed to create job'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.cronCreating = false; }
    },

    async updateCredential(name) {
      if (this.credentialSaving) return;
      if (!this.editCredKey.trim()) {
        this.showToast('Key is required');
        return;
      }
      this.credentialSaving = true;
      this.showToast('Validating API key...');
      if (!await this._validateCredential(name, this.editCredKey.trim())) {
        this.credentialSaving = false;
        return;
      }
      try {
        const resp = await fetch(`${window.__config.apiBase}/credentials`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ service: name, key: this.editCredKey.trim() }),
        });
        if (resp.ok) {
          this.showToast(`Credential updated: ${name}`);
          this.editingCredential = null;
          this.editCredKey = '';
          await this.fetchSettings();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Update failed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.credentialSaving = false; }
    },

    copyToClipboard(text) {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
          this.showToast('URL copied');
        }).catch(() => {
          this.showToast('Failed to copy');
        });
      } else {
        // Fallback for insecure contexts (non-HTTPS, non-localhost)
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); this.showToast('URL copied'); }
        catch (_) { this.showToast('Failed to copy'); }
        document.body.removeChild(ta);
      }
    },

    _saveChatToSession() {
      try {
        // Cap each agent's history to 50 messages to avoid storage bloat
        const histories = {};
        for (const [agentId, msgs] of Object.entries(this.chatHistories)) {
          if (!Array.isArray(msgs) || msgs.length === 0) continue;
          const capped = msgs.slice(-50).map(m => ({
            role: m.role,
            content: m.content,
            streaming: false,
            phase: (m.phase === 'error' || m.phase === 'done') ? m.phase : 'done',
            ts: m.ts || 0,
            tools: Array.isArray(m.tools) ? m.tools.map(t => ({
              name: t.name,
              status: t.status === 'running' ? 'done' : (t.status || 'done'),
              inputPreview: t.inputPreview || '',
              outputPreview: t.outputPreview || '',
            })) : [],
            timeline: Array.isArray(m.timeline) ? m.timeline.map(step => {
              if (step.kind === 'tool') {
                return {
                  kind: 'tool', name: step.name,
                  status: step.status === 'running' ? 'done' : (step.status || 'done'),
                  inputPreview: step.inputPreview || '',
                  outputPreview: step.outputPreview || '',
                };
              }
              return {
                kind: step.kind, name: step.name,
                content: step.kind === 'text' ? step.content : undefined,
              };
            }) : [],
          }));
          histories[agentId] = capped;
        }
        const payload = JSON.stringify({
          histories,
          openChats: this.openChats,
          activeChatId: this.activeChatId,
        });
        localStorage.setItem('ol_chats', payload);
      } catch (e) {
        // On quota exceeded, evict oldest agent history and retry once
        if (e instanceof DOMException && e.name === 'QuotaExceededError') {
          try {
            const keys = Object.keys(this.chatHistories);
            if (keys.length > 1) {
              const oldest = keys[0];
              delete this.chatHistories[oldest];
              this._saveChatToSession();
            }
          } catch (_) {}
        }
      }
    },

    agentHealthColor(result) {
      if (!result || result.type !== 'agent') return '';
      const agent = this.agents.find(a => a.id === result.label);
      if (!agent) return 'bg-gray-500';
      const status = agent.health_status || 'unknown';
      if (status === 'healthy') return 'bg-green-500';
      if (status === 'unhealthy' || status === 'restarting') return 'bg-amber-500';
      if (status === 'failed') return 'bg-red-500';
      return 'bg-gray-500';
    },

    async runWorkflow(name) {
      if (this.workflowRunLoading[name]) return;
      this.workflowRunLoading = { ...this.workflowRunLoading, [name]: true };
      try {
        const resp = await fetch(`${window.__config.apiBase}/workflows/${encodeURIComponent(name)}/run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
        if (resp.ok) {
          const data = await resp.json();
          this.showToast(`Workflow '${name}' started (${data.execution_id})`);
          this.fetchWorkflows();
        } else {
          this.showToast(`Failed to start workflow '${name}'`);
        }
      } catch (e) { console.warn('runWorkflow failed:', e); }
      finally { this.workflowRunLoading = { ...this.workflowRunLoading, [name]: false }; }
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
      if (this.cronEditSaving) return;
      if (!this.cronEditSchedule.trim()) { this.cancelCronEdit(); return; }
      this.cronEditSaving = true;
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
      finally { this.cronEditSaving = false; }
    },

    // ── Settings fetcher ─────────────────────────────────

    async fetchSettings() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/settings`);
        if (resp.ok) {
          this.settingsData = await resp.json();
          // Extract models for agent edit forms
          const apm = this.settingsData.available_provider_models;
          const pm = (apm && Object.keys(apm).length > 0) ? apm : this.settingsData.provider_models;
          if (pm) {
            this.availableModels = Object.values(pm).flat();
          }
        }
      } catch (e) { console.warn('fetchSettings failed:', e); }
    },

    async fetchStorage() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/storage`);
        if (resp.ok) this.storageData = await resp.json();
      } catch (e) { console.warn('fetchStorage failed:', e); }
    },

    formatBytes(bytes) {
      if (!bytes || bytes <= 0) return '0 B';
      const units = ['B', 'KB', 'MB', 'GB', 'TB'];
      const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
      return (bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0) + ' ' + units[i];
    },

    // ── Chat slide-over panel ──────────────────────────

    async _loadChatHistory(agentId) {
      // Always fetch from server — the persistent transcript is the
      // source of truth, ensuring history is consistent across devices.
      // Skip if streaming (avoid clobber) or fetched recently (debounce tab switches).
      if (this.chatStreamingAgents[agentId]) return;
      const now = Date.now();
      if (this._chatFetchedAt[agentId] && (now - this._chatFetchedAt[agentId]) < 5000) return;
      this._chatFetchedAt[agentId] = now;
      try {
        const resp = await fetch(`/dashboard/api/agents/${agentId}/chat/history`, { credentials: 'same-origin' });
        if (!resp.ok) return;
        const data = await resp.json();
        if (!data.messages || data.messages.length === 0) {
          if (this.chatHistories[agentId]?.length > 0) {
            this.chatHistories[agentId] = [];
            this._saveChatToSession();
          }
          return;
        }
        this.chatHistories[agentId] = data.messages.map(m => ({
          role: m.role,
          content: m.content,
          streaming: false,
          phase: 'done',
          ts: m.ts || 0,
          tools: Array.isArray(m.tools) ? m.tools.map(t =>
            typeof t === 'string' ? { name: t, status: 'done', inputPreview: '', outputPreview: '' } : t
          ) : [],
        }));
        this._saveChatToSession();
        this.$nextTick(() => this._scrollChat(agentId));
      } catch (e) {
        console.debug('_loadChatHistory failed for', agentId, e.message || e);
      }
    },

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
        // Load from server if local history is empty
        this._loadChatHistory(agentId);
        return;
      }
      this.openChats.push(agentId);
      this.activeChatId = agentId;
      // Load from server if local history is empty
      this._loadChatHistory(agentId);
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
      delete this._chatFetchedAt[agentId];
      this._saveChatToSession();
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
      this.chatHistories[agentId].push({ role: 'user', content: msg, ts: Date.now() });
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
        ts: Date.now(),
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
        const entry = this.chatHistories[agentId][idx];
        const hadContent = !!(entry.content || (entry.tools && entry.tools.length > 0));
        if (hadContent) {
          // Stream interrupted (e.g. tab backgrounded on mobile) but we have partial data.
          // Mark any running tools as done and finalize the message gracefully.
          entry.streaming = false;
          entry.phase = 'done';
          if (Array.isArray(entry.tools)) {
            entry.tools.forEach(t => { if (t.status === 'running') t.status = 'done'; });
          }
          if (Array.isArray(entry.timeline)) {
            entry.timeline.forEach(t => { if (t.status === 'running') t.status = 'done'; });
          }
        } else {
          // No content at all — show error but with a friendlier message
          entry.content = 'Connection interrupted — please try again.';
          entry.role = 'error';
          entry.streaming = false;
          entry.phase = 'error';
          this._pushChatTimelinePhase(entry, 'error');
        }
      }
      delete this._chatAborts[agentId];
      this.chatLoadingAgents[agentId] = false;
      this.chatStreamingAgents[agentId] = false;
      this.$nextTick(() => this._scrollChat(agentId));
      this._saveChatToSession();
      this.fetchQueues();
    },

    isAgentBusy(agentId) {
      if (this.chatStreamingAgents[agentId]) return true;
      return this.queueStatus?.[agentId]?.busy === true;
    },

    async steerAgent(agentId, message) {
      const msg = (message || '').trim();
      if (!msg) return;
      if (!this.chatHistories[agentId]) this.chatHistories[agentId] = [];
      this.chatHistories[agentId].push({ role: 'user', content: `[steer] ${msg}`, ts: Date.now() });
      try {
        await fetch(`${window.__config.apiBase}/agents/${agentId}/steer`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ message: msg }),
        });
        this.showToast(`Steered ${agentId}`);
      } catch (e) {
        this.showToast(`Steer failed: ${e.message}`);
      }
      this._saveChatToSession();
      this.fetchQueues();
    },

    // ── Broadcast ────────────────────────────────────────

    get detailAgentCronJobs() {
      if (!this.detailAgent) return [];
      return this.cronJobs.filter(j => j.agent === this.detailAgent);
    },

    addCronForAgent() {
      // Navigate from agent detail to System > Automation with agent pre-selected
      const agent = this.detailAgent;
      this.selectedAgent = null;
      if (this._detailReturnProject !== null && this._detailReturnProject !== undefined) {
        this.activeProject = this._detailReturnProject;
      }
      this._detailReturnProject = null;
      this.detailAgent = null;
      this.cronFormAgent = agent;
      this.showCronForm = true;
      this.systemTab = 'automation';
      this.switchTab('system');
    },

    get broadcastTargets() {
      // Project selected → project members; no project → standalone agents only
      // Exclude over-limit (locked) agents — they aren't running
      if (this.activeProject) {
        return this.filteredAgents.filter(a => !a.over_limit);
      }
      return this.agents.filter(a => !a.over_limit && !a.project);
    },

    sendBroadcast() {
      if (this.broadcastSending) return;
      const msg = (this.broadcastMessage || '').trim();
      if (!msg) return;
      const targets = this.broadcastTargets.map(a => a.id);
      if (targets.length === 0) return;
      this.broadcastMessage = '';
      for (const agentId of targets) this.openChat(agentId);
      this.activeChatId = targets[0];
      let sent = 0;
      const sentTargets = [];
      for (const agentId of targets) {
        if (this.chatStreamingAgents[agentId]) continue;
        sentTargets.push(agentId);
        this.sendChatTo(agentId, msg).then(() => {
          this._checkBroadcastComplete(agentId);
        }).catch(e => {
          console.warn('Broadcast sendChatTo failed for', agentId, e);
          this._checkBroadcastComplete(agentId);
        });
        sent++;
      }
      if (sent === 0) {
        this.showToast('All targeted agents are busy');
      } else {
        this.broadcastSending = true;
        this._broadcastPending = { targets: sentTargets, total: sent, completed: 0 };
        this.showToast(`Broadcast sent to ${sent} agent${sent !== 1 ? 's' : ''}`);
      }
    },

    _checkBroadcastComplete(agentId) {
      if (!this._broadcastPending) return;
      if (!this._broadcastPending.targets.includes(agentId)) return;
      this._broadcastPending.completed++;
      if (this._broadcastPending.completed >= this._broadcastPending.total) {
        this.showToast(`Broadcast complete — all ${this._broadcastPending.total} agents responded`);
        this._broadcastPending = null;
        this.broadcastSending = false;
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
        activity: ['activity', 'traces', 'events', 'logs', 'runtime'],
        system: ['system', 'costs', 'cron', 'automation', 'credentials', 'integrations', 'infrastructure', 'pricing', 'browsers', 'pubsub', 'blackboard', 'comms', 'communication', 'workflows'],
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
          results.push({ type: 'cron', label: job.id, desc: `${job.agent} · ${job.schedule}`, action: () => { this.systemTab = 'automation'; this.switchTab('system'); } });
        }
      }
      // Match workflows
      for (const wf of this.workflows || []) {
        if ((wf.name || '').toLowerCase().includes(q)) {
          results.push({ type: 'action', label: `Run ${wf.name}`, desc: `Workflow · ${wf.steps} steps`, action: () => { this.systemTab = 'automation'; this.switchTab('system'); this.runWorkflow(wf.name); } });
        }
      }
      // Match credentials
      for (const name of this.settingsData?.credentials?.names || []) {
        if (name.toLowerCase().includes(q)) {
          results.push({ type: 'action', label: name, desc: 'Credential', action: () => { this.systemTab = 'integrations'; this.switchTab('system'); } });
        }
      }
      // System quick actions
      const sysActions = [
        { label: 'View Logs', desc: 'Open runtime logs', keywords: ['logs', 'runtime', 'debug'], action: () => { this.switchTab('activity'); this.setActivityView('logs'); } },
        { label: 'Add Credential', desc: 'Add new API key', keywords: ['key', 'api', 'credential', 'token'], action: () => { this.systemTab = 'integrations'; this.switchTab('system'); this.showCredForm = true; } },
        { label: 'Manage Webhooks', desc: 'View and create webhooks', keywords: ['webhook', 'hook', 'endpoint'], action: () => { this.systemTab = 'integrations'; this.switchTab('system'); this.fetchWebhooks(); } },
        { label: 'Manage Channels', desc: 'Connect Telegram, Discord, Slack, WhatsApp', keywords: ['channel', 'telegram', 'discord', 'slack', 'whatsapp'], action: () => { this.systemTab = 'integrations'; this.switchTab('system'); this.fetchChannels(); } },
      ];
      for (const act of sysActions) {
        if (act.keywords.some(kw => kw.includes(q)) || act.label.toLowerCase().includes(q)) {
          results.push({ type: 'action', label: act.label, desc: act.desc, action: act.action });
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

    // ── Browser ────────────────────────────────────────

    async focusBrowser(agentId) {
      try {
        await fetch(`${window.__config.apiBase}/browser/${agentId}/focus`, { method: 'POST' });
      } catch (e) { console.warn('focusBrowser failed:', e); }
    },

    toggleBrowser() {
      this.showBrowserViewer = !this.showBrowserViewer;
      if (this.showBrowserViewer) {
        // Reset loading overlay for fresh connection
        this.$nextTick(() => {
          if (this.$refs.vncLoading) {
            this.$refs.vncLoading.style.opacity = '1';
          }
        });
        if (this.selectedAgent) {
          this.focusBrowser(this.selectedAgent);
        }
      }
    },

    // ── Credentials ──────────────────────────────────────

    async _validateCredential(service, key, baseUrl) {
      try {
        const body = { service, key };
        if (baseUrl) body.base_url = baseUrl;
        const resp = await fetch(`${window.__config.apiBase}/credentials/validate`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body),
        });
        if (resp.ok) {
          const data = await resp.json();
          if (!data.valid) {
            this.showToast(`Invalid API key: ${data.reason || 'authentication failed'}`);
            return false;
          }
        }
      } catch (e) {
        // Validation endpoint unavailable — allow save
      }
      return true;
    },

    async addCredential() {
      if (this.credSaving) return;
      const service = this.credService === '__custom__' ? this.credCustomService.trim() : this.credService.trim();
      if (!service || !this.credKey.trim()) return;
      this.credSaving = true;
      const isOAuth = this.credAuthType === 'oauth' && this.credService === 'anthropic';
      this.showToast(isOAuth ? 'Validating token...' : 'Validating API key...');
      try {
        const baseUrl = isOAuth ? '' : this.credBaseUrl.trim();
        if (!await this._validateCredential(service, this.credKey.trim(), baseUrl)) return;
        const body = { service, key: this.credKey.trim() };
        if (this.credService === '__custom__' && this.credTier === 'system') body.tier = 'system';
        if (baseUrl) body.base_url = baseUrl;
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
          this.credAuthType = 'api_key';
          this.showCredForm = false;
          this.fetchSettings();
        } else {
          const err = await resp.json();
          this.showToast(`Error: ${err.detail}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.credSaving = false; }
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
      if (this.onboardSaving) return;
      if (!this.onboardProvider || !this.onboardKey.trim()) return;
      this.onboardSaving = true;
      const isOAuth = this.onboardAuthType === 'oauth' && this.onboardProvider === 'anthropic';
      this.showToast(isOAuth ? 'Validating token...' : 'Validating API key...');
      try {
        const baseUrl = isOAuth ? '' : this.onboardBaseUrl.trim();
        if (!await this._validateCredential(this.onboardProvider, this.onboardKey.trim(), baseUrl)) return;
        const body = { service: this.onboardProvider, key: this.onboardKey.trim() };
        if (baseUrl) body.base_url = baseUrl;
        const resp = await fetch(`${window.__config.apiBase}/credentials`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.showToast(isOAuth ? 'Subscription token saved' : 'API key saved');
          this.onboardProvider = '';
          this.onboardKey = '';
          this.onboardBaseUrl = '';
          this.onboardAuthType = 'api_key';
          this.fetchSettings();
        } else {
          const err = await resp.json();
          this.showToast(`Error: ${err.detail}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.onboardSaving = false; }
    },

    // ── Channels ──────────────────────────────────────────

    async fetchChannels() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/channels`);
        if (resp.ok) this.channels = (await resp.json()).channels || [];
      } catch (e) { console.warn('fetchChannels failed:', e); }
    },

    _channelFields(type) {
      const fields = {
        telegram: [{ key: 'token', label: 'Bot Token', placeholder: '123456:ABC-DEF...' }],
        discord: [{ key: 'token', label: 'Bot Token', placeholder: 'MTIz...' }],
        slack: [
          { key: 'bot_token', label: 'Bot Token', placeholder: 'xoxb-...' },
          { key: 'app_token', label: 'App Token', placeholder: 'xapp-...' },
        ],
        whatsapp: [
          { key: 'access_token', label: 'Access Token', placeholder: 'EAAx...' },
          { key: 'phone_number_id', label: 'Phone Number ID', placeholder: '1234567890' },
        ],
      };
      return fields[type] || [];
    },

    startChannelConnect(type) {
      this.channelConnectType = type;
      this.channelTokens = {};
    },

    cancelChannelConnect() {
      this.channelConnectType = '';
      this.channelTokens = {};
    },

    async connectChannel() {
      if (this.channelConnecting) return;
      const type = this.channelConnectType;
      const fields = this._channelFields(type);
      const missing = fields.filter(f => !this.channelTokens[f.key]?.trim());
      if (missing.length) {
        this.showToast(`Missing: ${missing.map(f => f.label).join(', ')}`);
        return;
      }
      this.channelConnecting = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/channels/${type}/connect`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tokens: this.channelTokens }),
        });
        if (resp.ok) {
          this.showToast(`${type.charAt(0).toUpperCase() + type.slice(1)} connected`);
          this.cancelChannelConnect();
          this.fetchChannels();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Failed to connect'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.channelConnecting = false; }
    },

    disconnectChannel(type) {
      const label = type.charAt(0).toUpperCase() + type.slice(1);
      this.showConfirm(`Disconnect ${label}`, `Stop the ${label} channel?`, async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/channels/${type}/disconnect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}',
          });
          if (resp.ok) {
            this.showToast(`${label} disconnected`);
            this.fetchChannels();
          } else {
            const err = await resp.json().catch(() => ({}));
            this.showToast(`Error: ${err.detail || 'Disconnect failed'}`);
          }
        } catch (e) { this.showToast(`Error: ${e.message}`); }
      }, true);
    },

    // ── Webhooks ──────────────────────────────────────────

    async fetchWebhooks() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/webhooks`);
        if (resp.ok) this.webhooks = (await resp.json()).webhooks || [];
      } catch (e) { console.warn('fetchWebhooks failed:', e); }
    },

    async createWebhook() {
      if (this.webhookCreating) return;
      const name = this.webhookFormName.trim();
      const agent = this.webhookFormAgent;
      if (!name || !agent) { this.showToast('Name and agent are required'); return; }
      this.webhookCreating = true;
      try {
        const body = { name, agent };
        if (this.webhookFormSecret.trim()) body.secret = this.webhookFormSecret.trim();
        const resp = await fetch(`${window.__config.apiBase}/webhooks`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.showToast(`Webhook "${name}" created`);
          this.webhookFormName = '';
          this.webhookFormAgent = '';
          this.webhookFormSecret = '';
          this.showWebhookForm = false;
          this.fetchWebhooks();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Failed to create webhook'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.webhookCreating = false; }
    },

    async deleteWebhook(name) {
      this.showConfirm('Delete Webhook', `Delete webhook "${name}"?`, async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/webhooks/${encodeURIComponent(name)}`, { method: 'DELETE' });
          if (resp.ok) {
            this.showToast(`Webhook "${name}" deleted`);
            this.fetchWebhooks();
          } else {
            const err = await resp.json().catch(() => ({}));
            this.showToast(`Error: ${err.detail || 'Delete failed'}`);
          }
        } catch (e) { this.showToast(`Error: ${e.message}`); }
      }, true);
    },

    async testWebhook(name) {
      if (this.webhookTesting[name]) return;
      this.webhookTesting = { ...this.webhookTesting, [name]: true };
      try {
        const resp = await fetch(`${window.__config.apiBase}/webhooks/${encodeURIComponent(name)}/test`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
        if (resp.ok) {
          this.showToast(`Webhook "${name}" test sent`);
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Test failed: ${err.detail || 'Unknown error'}`);
        }
      } catch (e) { this.showToast(`Test failed: ${e.message}`); }
      finally { this.webhookTesting = { ...this.webhookTesting, [name]: false }; }
    },

    // ── Agent drill-down ──────────────────────────────────

    drillDown(agentId) {
      this._detailReturnProject = this.activeProject;
      this.selectedAgent = agentId;
      this.detailAgent = agentId;
      this.showBrowserViewer = false;
      this.identityTab = 'config';
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
      this.fetchCronJobs();
      this.activeTab = 'fleet';
      if (!this._skipPush) this._pushUrl(false);
    },

    // ── Chart.js rendering ────────────────────────────────

    _AGENT_CHART_COLORS: [
      '#6366f1', '#8b5cf6', '#06b6d4', '#10b981',
      '#f59e0b', '#ef4444', '#ec4899', '#3b82f6',
    ],

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

      const costColors = agents.map(a => this._AGENT_CHART_COLORS[this.agentColorIndex(a.agent)]);
      const costColorsBg = costColors.map(c => c + '66');
      const tokenColorsBg = costColors.map(c => c + '40');

      // Update existing chart in-place if possible
      if (this.costChart) {
        this.costChart.data.labels = labels;
        this.costChart.data.datasets[0].data = costs;
        this.costChart.data.datasets[0].backgroundColor = costColorsBg;
        this.costChart.data.datasets[0].borderColor = costColors;
        this.costChart.data.datasets[1].data = tokens;
        this.costChart.data.datasets[1].backgroundColor = tokenColorsBg;
        this.costChart.data.datasets[1].borderColor = costColors;
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
              backgroundColor: costColorsBg,
              borderColor: costColors,
              borderWidth: 1,
              borderRadius: 4,
              yAxisID: 'y',
            },
            {
              label: 'Tokens',
              data: tokens,
              backgroundColor: tokenColorsBg,
              borderColor: costColors,
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
        workspace_updated: 'text-teal-300',
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
        workspace_updated: 'bg-teal-300',
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

    formatModelName(model) {
      if (!model) return '';
      // Strip provider prefix (e.g. "anthropic/claude-sonnet-4-6" → "claude-sonnet-4-6")
      const parts = model.split('/');
      return parts.length > 1 ? parts.slice(1).join('/') : model;
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

    agentAvatarNum(agentId) {
      if (!agentId) return 1;
      const cfg = this.agentConfigs[agentId];
      if (cfg && cfg.avatar != null) return cfg.avatar;
      const agent = this.agents.find(a => a.id === agentId);
      if (agent && agent.avatar != null) return agent.avatar;
      return 1;
    },

    agentAvatarUrl(agentId) {
      const num = this.agentAvatarNum(agentId);
      const v = window.__config.assetVersion || '';
      return `/dashboard/static/avatars/${num}.svg` + (v ? `?v=${v}` : '');
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
      'status/': 'status',
      'tasks/': 'tasks',
      'research/': 'research',
      'drafts/': 'drafts',
      'artifacts/': 'artifacts',
      'alerts/': 'alerts',
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

    commsActionVerb(item) {
      if (item.source === 'pubsub') return 'published';
      if (item.action === 'delete') return 'deleted';
      if (item.action === 'cas_write') return 'claimed';
      return 'wrote';
    },

    commsActionTarget(item) {
      if (item.source === 'pubsub') return item.topic || '?';
      return item.key || '?';
    },

    commsActionClass(item) {
      if (item.source === 'pubsub') return 'text-purple-400';
      if (item.action === 'delete') return 'text-red-400';
      if (item.action === 'cas_write') return 'text-amber-400';
      return 'text-cyan-400';
    },

    commsIconClass(item) {
      if (item.source === 'pubsub') return 'bg-purple-500/20 text-purple-400';
      if (item.action === 'delete') return 'bg-red-500/20 text-red-400';
      return 'bg-cyan-500/20 text-cyan-400';
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
