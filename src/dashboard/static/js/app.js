/**
 * OpenLegion Dashboard — Alpine.js application.
 *
 * Two panels: Agents, System (Activity / Costs / Automation / Integrations / API Keys / Wallet / Storage / Settings).
 * Real-time updates via WebSocket + periodic REST polling.
 */

// CSRF protection: inject X-Requested-With on all non-GET fetch calls
// so the server can reject cross-origin state-changing requests.
const _origFetch = window.fetch;
window.fetch = function(input, init) {
  init = init || {};
  const method = (init.method || 'GET').toUpperCase();
  if (method !== 'GET' && method !== 'HEAD') {
    init.headers = new Headers(init.headers || {});
    if (!init.headers.has('X-Requested-With')) {
      init.headers.set('X-Requested-With', 'XMLHttpRequest');
    }
  }
  return _origFetch.call(this, input, init);
};
const _IDENTITY_TABS = [
  { id: 'config', label: 'Config', file: null, access: 'user' },
  { id: 'identity', label: 'Identity', file: null, access: 'user' },
  { id: 'memory', label: 'Memory', file: null, access: 'agent' },
  { id: 'activity', label: 'Activity', file: null, access: 'auto' },
  { id: 'logs', label: 'Logs', file: null, access: 'auto' },
  { id: 'capabilities', label: 'Tools', file: null, access: 'auto' },
  { id: 'files', label: 'Files', file: null, access: 'auto' },
];

const _IDENTITY_FILE_MAP = {
  identity: [
    { file: 'SOUL.md', label: 'Soul', cap: 4000, access: 'both', desc: 'Personality, tone, and communication style. Shapes how the agent speaks and approaches problems.' },
    { file: 'INSTRUCTIONS.md', label: 'Instructions', cap: 12000, access: 'both', desc: 'Step-by-step procedures, workflow rules, tool patterns, and domain knowledge. The agent\'s operating manual.' },
    { file: 'INTERFACE.md', label: 'Interface', cap: 4000, access: 'both', desc: 'Public collaboration contract — what this agent accepts, produces, and how other agents should interact with it. Teammates read this via get_agent_profile.' },
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
    activeTab: 'chat',
    tabs: [
      { id: 'chat', label: 'Chat' },
      { id: 'fleet', label: 'Agents' },
      { id: 'system', label: 'System' },
    ],
    connected: false,
    loading: true,
    lastRefresh: 0,
    toastQueue: [],

    // Operator readiness
    operatorReady: false,

    // Fleet Digest (parsed from operator's OBSERVATIONS.md)
    fleetDigest: null,
    _fleetDigestTimer: null,

    // Fleet
    agents: [],
    agentStates: {},
    agentCoordStatus: {},
    agentInboxCounts: {},
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
      'heartbeat_complete', 'cron_change', 'credit_exhausted', 'credential_request',
      'browser_login_request', 'browser_login_completed', 'browser_login_cancelled',
    ],

    // Agent detail
    detailAgent: null,
    selectedAgent: null,
    agentDetail: null,
    showBrowserViewer: false,
    _browserFocusDone: false,
    _browserViewOnly: true,
    _browserPendingAgent: null,

    // Agent config
    agentConfigs: {},
    editForm: {},
    availableModels: [],

    // Add agent
    addAgentMode: false,
    addAgentForm: { name: '', role: '', model: '', avatar: 1, color: null, project: '', template: '', _showPicker: false, _showColorPicker: false, _templateSearch: '', _templateDropdownOpen: false, _modelSearch: '', _modelDropdownOpen: false },
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
    modelChart: null,
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

    // Browser settings
    browserSpeed: 1.0,
    browserDelay: 0,
    browserSettingsLoading: false,
    _browserSettingsDebounce: null,
    _browserDelayDebounce: null,

    // Live per-agent browser metrics (Phase 2 §5.1/§5.2) — keyed by agent_id.
    // Populated from `browser_metrics` WS events; each entry is the payload
    // emitted by BrowserManager._emit_metrics plus a receivedAt wall-clock
    // stamp the dashboard uses to flag stale rows.
    browserMetrics: {},
    // Per-agent rolling history (Phase 7 §10.1). Keyed by agent_id; each
    // value is an array of recent per-minute payloads (oldest → newest)
    // capped by _BROWSER_METRICS_HISTORY_MAX. Seeded via
    // GET /api/agents/{id}/browser/metrics on detail-panel open, then
    // appended by the existing browser_metrics WS handler.
    browserMetricsHistory: {},
    // High-water seq per agent so subsequent fetches with ?since=N skip
    // payloads we've already absorbed (panel re-open / hot reload).
    _browserMetricsSeq: {},
    // Boot id last seen per agent — when the browser service restarts the
    // server-side seq counter resets, so we have to flush the local
    // watermark to avoid missing the new payloads.
    _browserMetricsBootId: {},
    _browserMetricsLoading: {},
    _browserMetricsError: {},
    _BROWSER_METRICS_HISTORY_MAX: 60,  // ~1 hour at 60s emit cadence

    captchaSolverProvider: '',
    captchaSolverKeyMasked: '',
    captchaSolverSaving: false,

    // System settings
    systemSettings: null,
    systemSettingsLoading: false,
    _systemSettingsDebounce: null,
    _restartingAll: false,
    _defaultModelSearch: '',
    _defaultModelDropdownOpen: false,

    // Storage
    storageData: null,
    dbDetails: null,
    dbDetailsLoading: false,
    dbPurging: {},  // { dbId: true } while purging
    purgeOpenId: null,   // db.id of the open purge dropdown, or null

    // Messenger-style chat panel
    openChats: [],             // Array of agent IDs with open chat panels
    chatHistories: {},         // Preserved — keyed by agent ID
    chatLoadingAgents: {},     // { agentId: true/false }
    chatStreamingAgents: {},   // { agentId: true/false }
    _chatAborts: {},           // { agentId: AbortController }
    _chatStreamTarget: {},     // { agentId: idx } — mutable target for SSE loop (steer redirect)
    _chatFetchedAt: {},        // { agentId: timestamp } — debounce refetches
    _chatWasStreaming: {},     // { agentId: true } — tracks streams active when tab was hidden
    _chatRecoveryPolls: {},   // { agentId: intervalId } — polls for stream recovery after tab return
    activeChatId: '',          // Currently active chat tab
    chatPanelMinimized: false, // Whether the chat panel is minimized to pill
    chatFullScreen: false,     // Whether the chat panel is expanded to full screen
    chatUnread: {},            // { agentId: count } — unread notifications while minimized
    _chatSessionId: (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : Math.random().toString(36).slice(2),

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
    agentActivity: [],
    agentActivityLoading: false,

    // Credit exhausted state
    creditExhausted: false,
    creditExhaustedDismissed: false,

    // Connection state
    connectionError: false,
    identityLogsLoading: false,
    identityLearnings: null,
    identityLearningsLoading: false,

    // Files tab
    agentFiles: [],
    agentFilesLoading: false,
    agentFilesPath: '.',
    agentFilePreview: null,   // { path, content, mime_type, encoding, size }
    agentFilePreviewLoading: false,

    // Uploads panel (System tab)
    uploadsList: [],
    uploadsLoading: false,
    uploadsUploading: false,
    uploadsError: null,

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
    systemTab: 'activity',
    systemTabs: [
      { id: 'activity', label: 'Activity' },
      { id: 'costs', label: 'Costs' },
      { id: 'automation', label: 'Automation' },
      { id: 'integrations', label: 'Integrations' },
      { id: 'apikeys', label: 'API Keys' },
      { id: 'wallet', label: 'Wallet' },
      { id: 'network', label: 'Network' },
      { id: 'storage', label: 'Storage' },
      { id: 'operator', label: 'Operator' },
      { id: 'browser', label: 'Browser' },
      { id: 'settings', label: 'Settings' },
    ],

    // Audit sub-tab
    auditLog: [],
    auditTotal: 0,
    auditPage: 1,
    auditLoading: false,

    // Unified Project Hub (replaces separate PROJECT.md + Comms + Broadcast panels)
    projectHubExpanded: false,
    projectHubTab: 'docs',  // 'docs' | 'activity' | 'state' | 'artifacts' | 'broadcast' | 'members'

    // PROJECT.md banner on Agents tab (kept for backward compat, not used by template)
    projectBannerExpanded: false,

    // Credentials
    showCredForm: false,
    credService: '',
    credCustomService: '',
    credKey: '',
    credBaseUrl: '',
    credTier: 'agent',
    credAuthType: 'api_key',
    credIsLlmProvider: false,
    credCustomModels: '',
    credCustomLabel: '',
    _credServiceDropdownOpen: false,
    _credServiceSearch: '',
    credServiceOptions: [],

    // Channels
    channels: [],
    channelConnectType: '',
    channelTokens: {},


    // Webhooks
    webhooks: [],
    showWebhookForm: false,
    webhookFormName: '',
    webhookFormAgent: '',
    webhookFormInstructions: '',
    webhookFormRequireSig: false,
    editingWebhookId: null,
    webhookEditName: '',
    webhookEditAgent: '',
    webhookEditInstructions: '',
    webhookEditRequireSig: false,

    // Webhook inline edit
    editingWebhookId: null,
    webhookEditName: '',
    webhookEditAgent: '',
    webhookEditInstructions: '',

    // Model health
    modelHealth: [],

    // Cron creation
    showCronForm: false,
    cronFormAgent: '',
    cronFormSchedule: 'every 15m',
    cronFormMessage: '',
    cronCreating: false,

    // Credential update / reveal
    editingCredential: null,
    editCredKey: '',
    credentialSaving: false,
    revealedCredentials: {},  // { name: value } for temporarily revealed credentials
    revealingCredential: null,  // name currently being fetched

    // Wallet
    walletData: null,  // { configured, agents, seed? }
    walletLoading: false,
    walletSeedVisible: false,
    walletSeed: '',
    walletInitializing: false,
    walletRpcChains: [],  // [{chain_id, label, rpc_current, rpc_default, is_custom}]
    walletRpcEditing: null,  // chain_id being edited
    walletRpcValue: '',  // input value for editing
    walletRpcSaving: false,

    // Network / Proxy
    networkProxy: {
      system_proxy: { configured: false, managed: false, managed_url: '', overridden: false, url: '' },
      no_proxy: '',
      agents: [],
      form: { url: '', username: '', password: '' },
      loading: false,
      saving: false,
      saved: false,
      removingSystemProxy: false,
      editingAgentProxy: null,
      agentProxyForm: { mode: 'inherit', url: '', username: '', password: '' },
      agentProxySaving: false,
    },

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
    webhookSaving: false,
    webhookRevealedSecret: null,
    webhookRevealedSecretHookId: null,
    apiKeys: [],
    apiKeysLegacy: false,
    apiKeyNewValue: null,
    apiKeyNewName: '',
    apiKeyCreating: false,
    showApiKeyForm: false,
    broadcastSending: false,
    confirmLoading: false,

    // Onboarding
    onboardProvider: '',
    onboardKey: '',
    onboardBaseUrl: '',
    onboardAuthType: 'api_key',
    onboardCustomService: '',
    onboardIsLlmProvider: false,
    onboardCustomModels: '',
    onboardCustomLabel: '',

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
    _cronDebounce: null,
    _seenEventIds: new Set(),


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
      if (this.activeTab === 'chat') return '/chat';
      if (this.activeTab === 'system') {
        if (this.systemTab === 'activity') {
          if (this.activityView === 'events') return '/system/activity/events';
          if (this.activityView === 'logs') return '/system/activity/logs';
          return '/system/activity';
        }
        return '/system/' + (this.systemTab || 'activity');
      }
      if (this.activeTab === 'fleet') return '/agents';
      return '/';
    },

    _buildTitle() {
      if (this.detailAgent) {
        const tabLabel = (_IDENTITY_TABS.find(t => t.id === this.identityTab) || _IDENTITY_TABS[0]).label;
        return `${this.detailAgent} \u00b7 ${tabLabel} \u2014 OpenLegion`;
      }
      if (this.activeTab === 'chat') return 'Chat \u2014 OpenLegion';
      if (this.activeTab === 'system') {
        if (this.systemTab === 'activity') {
          if (this.activityView === 'events') return 'Live Feed \u2014 OpenLegion';
          if (this.activityView === 'logs') return 'Logs \u2014 OpenLegion';
          return 'Traces \u2014 OpenLegion';
        }
        const st = this.systemTabs.find(t => t.id === this.systemTab);
        return (st ? st.label : 'System') + ' \u2014 OpenLegion';
      }
      return 'Agents \u2014 OpenLegion';
    },

    _parsePath(path) {
      const clean = path.replace(/^\/+/, '').replace(/\/+$/, '');
      const route = { tab: 'chat', activityView: 'traces', systemTab: 'activity', agentId: null, identityTab: 'config' };
      if (!clean) return route;

      if (clean === 'chat') { route.tab = 'chat'; return route; }
      if (clean === 'agents' || clean.startsWith('agents/')) { route.tab = 'fleet'; }

      const agentMatch = clean.match(/^agents\/([^/]+)(?:\/([^/]+))?$/);
      if (agentMatch) {
        route.agentId = agentMatch[1];
        const tab = agentMatch[2];
        if (tab && _IDENTITY_TABS.some(t => t.id === tab)) route.identityTab = tab;
        return route;
      }

      if (clean === 'activity/events') { route.tab = 'system'; route.systemTab = 'activity'; route.activityView = 'events'; }
      else if (clean === 'activity/logs') { route.tab = 'system'; route.systemTab = 'activity'; route.activityView = 'logs'; }
      else if (clean === 'activity') { route.tab = 'system'; route.systemTab = 'activity'; }
      else if (clean.startsWith('system')) {
        route.tab = 'system';
        const sub = clean.split('/')[1];
        // Backward compat for old URLs
        const _tabAliases = { schedules: 'automation', connections: 'integrations', uploads: 'storage' };
        const resolved = _tabAliases[sub] || sub;
        if (resolved && ['activity', 'costs', 'automation', 'integrations', 'apikeys', 'wallet', 'network', 'storage', 'operator', 'browser', 'settings'].includes(resolved)) {
          route.systemTab = resolved;
          if (resolved === 'activity') {
            const view = clean.split('/')[2];
            if (view === 'events') route.activityView = 'events';
            else if (view === 'logs') route.activityView = 'logs';
          }
        }
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
            // Sync sub-state before switchTab so it uses the correct values
            if (route.tab === 'system') {
              this.systemTab = route.systemTab;
              if (route.systemTab === 'activity') this.activityView = route.activityView;
            }
            this.switchTab(route.tab);
          } else if (route.tab === 'system') {
            if (this.systemTab !== route.systemTab) {
              if (this.systemTab === 'activity') this._stopActivityRefresh();
              this.systemTab = route.systemTab;
              if (route.systemTab === 'integrations') {
                this.fetchChannels(); this.fetchWebhooks(); this.fetchApiKeys();
              }
              if (route.systemTab === 'apikeys') {
                this.fetchSettings();
              }
              if (route.systemTab === 'settings') {
                this.fetchBrowserSettings();
                this.fetchSystemSettings();
              }
              if (route.systemTab === 'network') {
                this.loadNetworkProxy();
              }
              if (route.systemTab === 'storage') {
                this.fetchUploads(); this.fetchStorage(); this.fetchDatabaseDetails();
              }
              if (route.systemTab === 'operator') {
                this.fetchAuditLog();
              }
              if (route.systemTab === 'activity') {
                this.activityView = route.activityView;
                if (route.activityView === 'traces') { this.fetchTraces(); this._startActivityRefresh(); }
                else if (route.activityView === 'logs') { this.fetchSystemLogs(); }
              }
            } else if (route.systemTab === 'activity' && this.activityView !== route.activityView) {
              this.setActivityView(route.activityView);
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
      return this.agents.filter(a => a.running !== false && a.id !== 'operator');
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
      // Hide the operator system agent from the fleet view — it's managed via the Chat tab
      if (this.activeProject) {
        return this.agents.filter(a => a.id !== 'operator' && a.project === this.activeProject);
      }
      // When projects exist, show only standalone (unassigned) agents
      const base = this.projects.length > 0 ? this.unassignedAgents : this.agents;
      return base.filter(a => a.id !== 'operator');
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

    get costBudgetSummary() {
      const budgets = this.costData.budgets || {};
      let total = 0, overBudget = 0;
      for (const [, b] of Object.entries(budgets)) {
        total++;
        if (!b.allowed) overBudget++;
      }
      return { total, overBudget };
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
      this._initTs = Date.now();  // track page load time to skip replayed events
      const cfg = window.__config || {};

      // Build credential service options from server-injected providers
      const allProviders = cfg.allProviders || [];
      this.credServiceOptions = [
        ...allProviders.map(p => ({ value: p.name, label: p.label, group: 'LLM Providers (system tier)' })),
        { value: 'brave_search_api_key', label: 'Brave Search', group: 'Agent tools' },
        { value: 'apollo_api_key', label: 'Apollo', group: 'Agent tools' },
        { value: 'hunter_api_key', label: 'Hunter', group: 'Agent tools' },
      ];

      this._ws = new DashboardWebSocket(cfg.wsUrl, {
        onEvent: (evt) => this.onWsEvent(evt),
        onConnect: () => {
          const isReconnect = this._wsConnectedOnce;
          this._wsConnectedOnce = true;
          this.connected = true;
          // On reconnect, refresh chat histories to catch messages from other sessions.
          // Skip on initial connect — init() already fetches them.
          if (isReconnect) {
            for (const agentId of this.openChats) {
              delete this._chatFetchedAt[agentId];
              this._loadChatHistory(agentId);
            }
          }
        },
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

      // Restore persisted state from localStorage
      try {
        const saved = localStorage.getItem('ol_chats');
        if (saved) {
          const parsed = JSON.parse(saved);
          if (parsed.histories) this.chatHistories = parsed.histories;
          if (parsed.openChats) this.openChats = parsed.openChats;
          if (parsed.activeChatId) this.activeChatId = parsed.activeChatId;
          if (parsed.chatPanelMinimized) this.chatPanelMinimized = true;
        }
      } catch (e) {
        console.debug('localStorage restore skipped:', e.message || e);
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
      const isDeepLink = initRoute.agentId || initRoute.tab !== 'chat' || initRoute.activityView !== 'traces' || initRoute.systemTab !== 'costs';
      if (isDeepLink) {
        this.$nextTick(() => {
          this._applyRoute(initRoute);
          document.title = this._buildTitle();
          history.replaceState(null, '', this._buildPath());
        });
      } else {
        document.title = this._buildTitle();
      }

      // Ensure operator chat is initialized when landing on the chat tab
      if (this.activeTab === 'chat' || initRoute.tab === 'chat') {
        if (!this.openChats.includes('operator')) {
          this.openChats.push('operator');
        }
        this.activeChatId = 'operator';
        this._loadChatHistory('operator');
        this._startFleetDigestRefresh();
        this.$nextTick(() => {
          this._scrollChat('operator', true);
          const el = document.getElementById('operator-chat-input');
          if (el) el.focus();
        });
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
          if (this._cronDebounce) { clearTimeout(this._cronDebounce); this._cronDebounce = null; }
          if (this._modelHealthInterval) { clearInterval(this._modelHealthInterval); this._modelHealthInterval = null; }
          this._stopActivityRefresh();
          // Snapshot which chats had active SSE streams before tab hides
          this._chatWasStreaming = {};
          for (const agentId of this.openChats) {
            if (this.chatStreamingAgents[agentId]) this._chatWasStreaming[agentId] = true;
          }
        } else {
          // Clear favicon badge
          document.title = document.title.replace(/^\(\d+\)\s*/, '');
          // Resume polling — restore intervals without navigating (switchTab clears detailAgent)
          this._refreshInterval = setInterval(() => this.fetchAgents(), 15000);
          this.fetchAgents();
          if (this.activeTab === 'system') {
            this.fetchCronJobs();
            this.fetchStorage();
            if (this.systemTab === 'activity' && this.activityView === 'traces') {
              this.fetchTraces();
              this._startActivityRefresh();
            }
            if (this.systemTab === 'activity' && this.activityView === 'logs') {
              this.fetchSystemLogs();
            }
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
          // Recover chats whose SSE streams died while tab was hidden,
          // then sync histories for non-recovering agents.
          this._recoverDeadStreams().then(() => {
            for (const agentId of this.openChats) {
              if (this._chatRecoveryPolls[agentId]) continue; // being recovered
              delete this._chatFetchedAt[agentId];
              this._loadChatHistory(agentId);
            }
          });
        }
      });

      // ── Tooltip positioning for .setting-hint elements ──
      // Uses position:fixed to escape overflow:hidden ancestors.
      const _positionHint = (hint) => {
        const text = hint.querySelector('.setting-hint-text');
        if (!text) return;
        const r = hint.getBoundingClientRect();
        const gap = 6;
        // Measure tooltip (show off-screen briefly)
        text.style.visibility = 'hidden';
        text.style.display = 'block';
        text.style.top = '-9999px';
        text.style.left = '-9999px';
        const tw = text.offsetWidth;
        const th = text.offsetHeight;
        // Default: above, left-aligned with icon
        let top = r.top - gap - th;
        let left = r.left;
        let below = false;
        // Flip below if clipped at top
        if (top < 8) {
          top = r.bottom + gap;
          below = true;
        }
        // Clamp to right edge
        if (left + tw > window.innerWidth - 8) {
          left = window.innerWidth - tw - 8;
        }
        // Clamp to left edge
        if (left < 8) left = 8;
        // Arrow points at icon center
        const arrowX = Math.min(Math.max(r.left + r.width / 2 - left, 8), tw - 8);
        text.style.setProperty('--arrow-x', arrowX + 'px');
        text.style.top = top + 'px';
        text.style.left = left + 'px';
        text.style.visibility = '';
        text.classList.toggle('hint-below', below);
      };
      const _hideHint = (hint) => {
        const text = hint.querySelector('.setting-hint-text');
        if (!text) return;
        text.style.display = '';
        text.style.top = '';
        text.style.left = '';
        text.style.visibility = '';
        text.classList.remove('hint-below');
        text.style.removeProperty('--arrow-x');
      };
      document.addEventListener('mouseenter', (e) => {
        const hint = e.target.closest('.setting-hint');
        if (hint) _positionHint(hint);
      }, true);
      document.addEventListener('mouseleave', (e) => {
        const hint = e.target.closest('.setting-hint');
        if (hint) _hideHint(hint);
      }, true);
      document.addEventListener('focusin', (e) => {
        const hint = e.target.closest('.setting-hint');
        if (hint) _positionHint(hint);
      }, true);
      document.addEventListener('focusout', (e) => {
        const hint = e.target.closest('.setting-hint');
        if (hint) _hideHint(hint);
      }, true);
    },

    destroy() {
      if (this._ws) this._ws.disconnect();
      if (this._refreshInterval) clearInterval(this._refreshInterval);
      if (this._queuePollInterval) clearInterval(this._queuePollInterval);
      if (this._cronDebounce) clearTimeout(this._cronDebounce);
      if (this._modelHealthInterval) clearInterval(this._modelHealthInterval);
      if (this._cookieRenewalInterval) clearInterval(this._cookieRenewalInterval);
      if (this._visibilityHandler) document.removeEventListener('visibilitychange', this._visibilityHandler);
      if (this._costDebounce) clearTimeout(this._costDebounce);
      if (this.modelChart) { this.modelChart.destroy(); this.modelChart = null; }
      if (this._fleetDebounce) clearTimeout(this._fleetDebounce);
      if (this._activityRefresh) clearInterval(this._activityRefresh);
      if (this._tracesDebounce) clearTimeout(this._tracesDebounce);
      if (this.costChart) { this.costChart.destroy(); this.costChart = null; }
      Object.values(this._stateTimers).forEach(clearTimeout);
      Object.values(this._scrollTimers).forEach(clearTimeout);
      Object.values(this._chatAborts).forEach(c => c?.abort());
      Object.values(this._chatRecoveryPolls).forEach(clearInterval);
      this._chatRecoveryPolls = {};
      this.stopHeartbeatTimer();
      this._stopFleetDigestRefresh();
      if (this._cmdPaletteHandler) document.removeEventListener('keydown', this._cmdPaletteHandler);
      if (this._popstateHandler) window.removeEventListener('popstate', this._popstateHandler);
    },

    // ── Fleet Digest ─────────────────────────────────────

    async fetchFleetDigest() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/operator/workspace/OBSERVATIONS.md`);
        if (!resp.ok) { this.fleetDigest = null; return; }
        const text = await resp.text();
        // Parse JSON block from markdown format
        const match = text.match(/```json\n([\s\S]*?)\n```/);
        if (match) {
          this.fleetDigest = JSON.parse(match[1]);
        } else {
          this.fleetDigest = null;
        }
      } catch (e) {
        this.fleetDigest = null;
      }
    },

    _startFleetDigestRefresh() {
      this.fetchFleetDigest();
      if (!this._fleetDigestTimer) {
        this._fleetDigestTimer = setInterval(() => {
          if (this.activeTab === 'chat') this.fetchFleetDigest();
        }, 300000); // 5 minutes
      }
    },

    _stopFleetDigestRefresh() {
      if (this._fleetDigestTimer) {
        clearInterval(this._fleetDigestTimer);
        this._fleetDigestTimer = null;
      }
    },

    // ── Operator readiness ───────────────────────────────

    checkOperatorReady() {
      const op = this.agents.find(a => a.id === 'operator');
      this.operatorReady = op && op.health_status === 'healthy';
    },

    // ── Tab switching ─────────────────────────────────────

    switchTab(tab) {
      this.activeTab = tab;
      this.detailAgent = null;
      // Clear tab-specific auto-refresh intervals
      this._stopActivityRefresh();
      if (tab === 'chat') {
        if (!this.openChats.includes('operator')) {
          this.openChats.push('operator');
        }
        this.activeChatId = 'operator';
        this._loadChatHistory('operator');
        this._startFleetDigestRefresh();
        this.$nextTick(() => {
          this._scrollChat('operator', true);
          const el = document.getElementById('operator-chat-input');
          if (el) el.focus();
        });
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
          this.fetchApiKeys();
        }
        if (this.systemTab === 'network') {
          this.loadNetworkProxy();
        }
        if (this.systemTab === 'settings') {
          this.fetchBrowserSettings();
          this.fetchSystemSettings();
        }
        if (this.systemTab === 'browser') {
          this.fetchBrowserSettings();
          this.fetchSystemSettings();
          this.fetchCaptchaSolver();
        }
        if (this.systemTab === 'activity') {
          if (this.activityView === 'traces') {
            this.fetchTraces();
            this._startActivityRefresh();
          } else if (this.activityView === 'logs') {
            this.fetchSystemLogs();
          }
        }
      }
      if (!this._skipPush) this._pushUrl(false);
    },

    switchSystemTab(tabId) {
      if (this.systemTab === 'activity' && tabId !== 'activity') this._stopActivityRefresh();
      this.systemTab = tabId;
      this._pushUrl(false);
      if (tabId === 'integrations') { this.fetchChannels(); this.fetchWebhooks(); this.fetchApiKeys(); }
      if (tabId === 'apikeys') { this.fetchSettings(); }
      if (tabId === 'storage') { this.fetchUploads(); this.fetchStorage(); this.fetchDatabaseDetails(); }
      if (tabId === 'network') { this.loadNetworkProxy(); }
      if (tabId === 'settings') { this.fetchBrowserSettings(); this.fetchSystemSettings(); }
      if (tabId === 'browser') { this.fetchBrowserSettings(); this.fetchSystemSettings(); this.fetchCaptchaSolver(); }
      if (tabId === 'operator') {
        this.fetchAuditLog();
      }
      if (tabId === 'activity') {
        if (this.activityView === 'traces') { this.fetchTraces(); this._startActivityRefresh(); }
        else if (this.activityView === 'logs') { this.fetchSystemLogs(); }
      }
    },

    // ── Audit log ────────────────────────────────────────

    async fetchAuditLog() {
      this.auditLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/operator-audit?per_page=20&page=${this.auditPage}`);
        if (resp.ok) {
          const data = await resp.json();
          this.auditLog = data.entries || [];
          this.auditTotal = data.total || 0;
        }
      } catch (e) {
        console.error('Failed to fetch audit log:', e);
      } finally {
        this.auditLoading = false;
      }
    },

    // ── Markdown rendering for chat messages ─────────────

    renderMarkdown(text) {
      if (!text) return '';
      // Strip <think>...</think> blocks and unclosed <think> (still streaming)
      let cleaned = text.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').trim();
      if (!cleaned) return '';
      try {
        const html = marked.parse(cleaned, { breaks: true, gfm: true });
        const sanitized = DOMPurify.sanitize(html);
        // Open all links in new tab so chat/artifact links don't navigate away
        const tpl = document.createElement('template');
        tpl.innerHTML = sanitized;
        tpl.content.querySelectorAll('a[href]').forEach(a => {
          a.setAttribute('target', '_blank');
          a.setAttribute('rel', 'noopener noreferrer');
        });
        return tpl.innerHTML;
      } catch (_) {
        return this._escapeHtml(cleaned).replace(/\n/g, '<br>');
      }
    },

    // ── Toast helper ──────────────────────────────────────

    _toastId: 0,

    showToast(msg, duration) {
      const id = ++this._toastId;
      this.toastQueue.push({ id, msg });
      setTimeout(() => {
        this.toastQueue = this.toastQueue.filter(t => t.id !== id);
      }, duration || 4000);
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

      // Update agent activity state
      const agent = evt.agent;
      if (agent) {
        this._updateAgentState(agent, evt.type);
      }

      // Live-update fleet on llm_call/health/agent_state changes (debounced)
      if (evt.type === 'llm_call' || evt.type === 'health_change' || evt.type === 'agent_state') {
        this._debouncedFleetRefresh();
        // Refresh capabilities when an agent re-registers (e.g. new skill authored)
        if (evt.type === 'agent_state' && agent && evt.data?.state === 'registered') {
          const viewing = this.selectedAgent || this.detailAgent;
          if (viewing === agent) this.fetchAgentCapabilities(agent);
        }
      }

      // Refresh cron panel on cron changes (replaces polling)
      if (evt.type === 'cron_change') {
        if (this._cronDebounce) clearTimeout(this._cronDebounce);
        this._cronDebounce = setTimeout(() => this.fetchCronJobs(), 500);
      }

      // Refresh model health on health_change events
      if (evt.type === 'health_change') {
        this.fetchModelHealth();
      }

      // Show credit exhausted banner when LLM call fails with 402
      if (evt.type === 'credit_exhausted') {
        this.creditExhausted = true;
        this.creditExhaustedDismissed = false;
        // Inject credit exhaustion message into the agent's chat
        const agent = evt.agent;
        if (agent && this.chatHistories[agent]) {
          const recent = this.chatHistories[agent].slice(-3);
          const alreadyShown = recent.some(m => m._creditExhausted);
          if (!alreadyShown) {
            this.chatHistories[agent].push({
              role: 'credit_exhausted',
              content: evt.data?.error || 'Credits depleted — LLM calls are currently failing.',
              _creditExhausted: true,
              ts: Date.now() / 1000,
            });
          }
        }
      }

      // Clear credit exhausted state on successful LLM call
      if (evt.type === 'llm_call' && this.creditExhausted) {
        this.creditExhausted = false;
      }

      // Per-agent browser metrics (Phase 2 §5.1/§5.2). Payload shape comes
      // from BrowserManager._emit_metrics — we just index it by agent_id
      // and stamp receipt time so stale rows can fade.
      if (evt.type === 'browser_metrics' && evt.agent && evt.data) {
        // Phase 7 third-pass: if the mesh stamped a boot_id and it differs
        // from the last one we saw, the browser service restarted — its
        // seq counter reset to 0, so any history we accumulated is from
        // a previous boot. Flush before appending so the dedup-by-seq
        // path doesn't get fooled into mixing two counter generations.
        const newBootId = evt.data.boot_id || '';
        const prevBootId = this._browserMetricsBootId[evt.agent] || '';
        if (newBootId && prevBootId && newBootId !== prevBootId) {
          this.browserMetricsHistory = {
            ...this.browserMetricsHistory,
            [evt.agent]: [],
          };
          this._browserMetricsSeq = {
            ...this._browserMetricsSeq,
            [evt.agent]: 0,
          };
        }
        if (newBootId) {
          this._browserMetricsBootId = {
            ...this._browserMetricsBootId,
            [evt.agent]: newBootId,
          };
        }
        this.browserMetrics = {
          ...this.browserMetrics,
          [evt.agent]: { ...evt.data, receivedAt: Date.now() },
        };
        // Phase 7 §10.1 — also append to the per-agent rolling history so
        // the detail panel can render trend sparklines without an extra
        // fetch round-trip per tick.
        this._appendBrowserMetricsHistory(evt.agent, evt.data);
      }

      // §6.3 navigator self-test result (one-shot per browser start).
      // We attach the probe summary to the per-agent metrics entry so it
      // renders alongside click rate / snapshot bytes, and toast on
      // mismatch so operators see fingerprint regressions immediately.
      if (evt.type === 'browser_nav_probe' && evt.agent && evt.data) {
        const existing = this.browserMetrics[evt.agent] || {};
        this.browserMetrics = {
          ...this.browserMetrics,
          [evt.agent]: {
            ...existing,
            probe_ok: evt.data.ok,
            probe_mismatches: evt.data.mismatches || [],
            probe_signals: evt.data.signals || {},
            probe_at: Date.now(),
            // Preserve the most recent drain timestamp if any so the
            // staleness check still works. For probe-only updates we
            // intentionally do NOT bump receivedAt — otherwise an
            // agent that only ever probes (no clicks/navs) would never
            // hit the 30-min eviction even after going truly idle.
            receivedAt: existing.receivedAt || Date.now(),
          },
        };
        if (!evt.data.ok && (evt.data.mismatches || []).length) {
          // Toast dedup: a fleet-wide regression (e.g. Camoufox version
          // bump that breaks navigator.connection injection) would
          // otherwise stack one 8s toast per agent on a mass restart.
          // Fingerprint signature = the sorted mismatch list. Suppress
          // identical signatures fired within the same 30s window;
          // surface a "+N more" toast instead.
          const sig = (evt.data.mismatches || []).slice().sort().join('|');
          this._probeToastSeen = this._probeToastSeen || new Map();
          const now = Date.now();
          // Cap the dedup map at ~5 minutes of history so a long-lived
          // dashboard session doesn't accumulate signatures forever.
          // Past 5 minutes, the entry is irrelevant (way past the 30s
          // dedup window) and just leaks memory.
          const STALE_AT = now - 5 * 60 * 1000;
          if (this._probeToastSeen.size > 50) {
            for (const [k, v] of this._probeToastSeen) {
              if (v.firstAt < STALE_AT) this._probeToastSeen.delete(k);
            }
          }
          const last = this._probeToastSeen.get(sig);
          if (!last || now - last.firstAt > 30000) {
            this._probeToastSeen.set(sig, { firstAt: now, count: 1 });
            this.showToast(
              'Browser fingerprint drift on ' + evt.agent + ': ' +
              evt.data.mismatches.slice(0, 2).join('; '),
              8000,
            );
          } else {
            last.count += 1;
            // Coalesce: only emit the rollup toast on the second hit
            // (rollup-of-rollup would itself spam).
            if (last.count === 2) {
              this.showToast(
                'Browser fingerprint drift hit ' + last.count +
                ' agents in the last 30s — check Browser tab',
                8000,
              );
            }
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
        // Refresh coordination data when status/ or tasks/ keys change
        if (evt.data.key.startsWith('status/') || evt.data.key.startsWith('tasks/')) {
          if (this._coordDebounce) clearTimeout(this._coordDebounce);
          this._coordDebounce = setTimeout(() => this._fetchCoordination(), 1000);
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

      // Auto-refresh activity tab when a heartbeat completes for the agent being viewed
      if (evt.type === 'heartbeat_complete' && evt.agent) {
        const viewing = this.selectedAgent || this.detailAgent;
        if (viewing === evt.agent && this.identityTab === 'activity') {
          if (this._activityDebounce) clearTimeout(this._activityDebounce);
          this._activityDebounce = setTimeout(() => this.fetchAgentActivity(viewing), 1000);
        }
      }

      // Surface agent notifications as toasts + inject into chat history.
      // Skip toast/unread/insertion for replayed events (sent before this
      // page loaded) — _loadChatHistory handles restoring them from the
      // server transcript and localStorage.
      if (evt.type === 'notification' && agent && evt.data && evt.data.message) {
        const msg = evt.data.message;
        const evtTs = this._normalizeEventTs(evt);
        const isReplay = evtTs < this._initTs - 5000;  // 5s grace for clock skew

        if (!isReplay) {
          this.showToast(`${agent}: ${msg}`);
        }

        // Push to chat history only if not already present (prevents duplicates
        // when _loadChatHistory already fetched it from the server transcript).
        if (!this.chatHistories[agent]) this.chatHistories[agent] = [];
        const isDup = this.chatHistories[agent].some(m =>
          m.role === 'notification' && m.content === msg && Math.abs((m.ts || 0) - evtTs) < 2000
        );
        if (!isDup && !isReplay) {
          this.chatHistories[agent].push({
            role: 'notification',
            content: msg,
            ts: evtTs,
          });
          if (this.activeChatId === agent) {
            this.$nextTick(() => this._scrollChat(agent));
          } else {
            this.chatUnread = { ...this.chatUnread, [agent]: (this.chatUnread[agent] || 0) + 1 };
          }
        }
      }

      // Surface credential requests as secure input cards in chat.
      if (evt.type === 'credential_request' && agent && evt.data && evt.data.name) {
        const evtTs = this._normalizeEventTs(evt);
        const credCard = {
          role: 'credential_request',
          content: evt.data.description || '',
          name: evt.data.name,
          service: evt.data.service || '',
          saved: false,
          ts: evtTs,
        };
        // Show in the requesting agent's chat
        if (!this.chatHistories[agent]) this.chatHistories[agent] = [];
        const isDup = this.chatHistories[agent].some(m =>
          m.role === 'credential_request' && m.name === evt.data.name && Math.abs((m.ts || 0) - evtTs) < 5000
        );
        if (!isDup) {
          this.chatHistories[agent].push(credCard);
          if (this.activeChatId === agent) {
            this.$nextTick(() => this._scrollChat(agent));
          } else {
            this.chatUnread = { ...this.chatUnread, [agent]: (this.chatUnread[agent] || 0) + 1 };
          }
        }
        // Also surface in operator chat so users see it from the main Chat tab
        if (agent !== 'operator') {
          if (!this.chatHistories['operator']) this.chatHistories['operator'] = [];
          const opDup = this.chatHistories['operator'].some(m =>
            m.role === 'credential_request' && m.name === evt.data.name && Math.abs((m.ts || 0) - evtTs) < 5000
          );
          if (!opDup) {
            this.chatHistories['operator'].push({ ...credCard, _from_agent: agent });
            if (this.activeTab === 'chat') {
              this.$nextTick(() => this._scrollChat('operator'));
            }
          }
        }
      }

      // Surface browser login requests as interactive VNC cards in chat.
      if (evt.type === 'browser_login_request' && agent && evt.data && evt.data.service) {
        const evtTs = this._normalizeEventTs(evt);
        const loginCard = {
          role: 'browser_login_request',
          content: evt.data.description || '',
          service: evt.data.service || '',
          url: evt.data.url || '',
          completed: false,
          cancelled: false,
          ts: evtTs,
        };
        // Show in the requesting agent's chat
        if (!this.chatHistories[agent]) this.chatHistories[agent] = [];
        const isDup = this.chatHistories[agent].some(m =>
          m.role === 'browser_login_request' && m.service === evt.data.service && Math.abs((m.ts || 0) - evtTs) < 5000
        );
        if (!isDup) {
          this.chatHistories[agent].push(loginCard);
          if (this.activeChatId === agent) {
            this.$nextTick(() => this._scrollChat(agent));
          } else {
            this.chatUnread = { ...this.chatUnread, [agent]: (this.chatUnread[agent] || 0) + 1 };
          }
        }
        // Also surface in operator chat
        if (agent !== 'operator') {
          if (!this.chatHistories['operator']) this.chatHistories['operator'] = [];
          const opDup = this.chatHistories['operator'].some(m =>
            m.role === 'browser_login_request' && m._from_agent === agent && m.service === evt.data.service && Math.abs((m.ts || 0) - evtTs) < 5000
          );
          if (!opDup) {
            this.chatHistories['operator'].push({ ...loginCard, _from_agent: agent });
            if (this.activeTab === 'chat') {
              this.$nextTick(() => this._scrollChat('operator'));
            }
          }
        }
      }

      // Sync browser login card state across all copies (agent chat + operator chat).
      // When one card is completed/cancelled, the event marks all copies so the
      // other card can't send a duplicate steer message.
      if (evt.type === 'browser_login_completed' && agent && evt.data?.service) {
        for (const chatId of [agent, 'operator']) {
          const hist = this.chatHistories[chatId];
          if (!hist) continue;
          for (const m of hist) {
            if (m.role === 'browser_login_request' && m.service === evt.data.service
                && (chatId === agent || m._from_agent === agent)) {
              m.completed = true;
            }
          }
        }
      }
      if (evt.type === 'browser_login_cancelled' && agent && evt.data?.service) {
        for (const chatId of [agent, 'operator']) {
          const hist = this.chatHistories[chatId];
          if (!hist) continue;
          for (const m of hist) {
            if (m.role === 'browser_login_request' && m.service === evt.data.service
                && (chatId === agent || m._from_agent === agent)) {
              m.cancelled = true;
            }
          }
        }
      }

      // ── Cross-session chat synchronization ──
      // These events let other tabs/devices see chat activity in real-time.
      // Skip events from our own session (already handled by SSE stream).
      if (evt.data?.session === this._chatSessionId) {
        // Own session — already handled via SSE, skip
      } else if (evt.type === 'chat_user_message' && agent && evt.data?.message) {
        // Another session sent a message — add to our history.
        // Skip replayed events from before page load (stale buffer entries
        // would create phantom thinking bubbles with stuck streaming flags).
        if (!this.chatHistories[agent]) this.chatHistories[agent] = [];
        const msg = evt.data.message;
        const evtTs = this._normalizeEventTs(evt);
        const isReplay = evtTs < this._initTs - 5000;
        const isDup = isReplay || this.chatHistories[agent].some(m =>
          m.role === 'user' && m.content === msg && Math.abs((m.ts || 0) - evtTs) < 5000
        );
        if (!isDup) {
          this.chatHistories[agent].push({ role: 'user', content: msg, ts: evtTs });
          // Show thinking indicator
          this.chatStreamingAgents[agent] = true;
          this.chatLoadingAgents[agent] = true;
          // Add a thinking bubble
          this.chatHistories[agent].push({
            role: 'agent', content: '', streaming: true,
            phase: 'thinking', tools: [], timeline: [],
            _sawTextDelta: false, _remoteStream: true, ts: Date.now(),
          });
          if (this.activeChatId === agent) {
            this.$nextTick(() => this._scrollChat(agent));
          } else {
            this.chatUnread = { ...this.chatUnread, [agent]: (this.chatUnread[agent] || 0) + 1 };
          }
          this._saveChatToSession();
        }
      } else if (evt.type === 'text_delta' && agent && evt.data?.session && evt.data?.content) {
        // Another session's agent is streaming text
        const hist = this.chatHistories[agent] || [];
        const last = hist[hist.length - 1];
        if (last && last._remoteStream && last.streaming) {
          last.content += evt.data.content;
          last.phase = 'responding';
          last._sawTextDelta = true;
          this.chatLoadingAgents[agent] = false;
          if (this.activeChatId === agent) this._scrollChat(agent);
        }
      } else if (evt.type === 'tool_start' && agent && evt.data?.session && evt.data.session !== this._chatSessionId) {
        // Another session's agent is using a tool — update remote stream bubble
        const hist = this.chatHistories[agent] || [];
        const last = hist[hist.length - 1];
        if (last && last._remoteStream && last.streaming) {
          last.phase = 'tool';
          this.chatLoadingAgents[agent] = false;
          const toolEntry = {
            id: `${Date.now()}-${(last.tools || []).length}`,
            name: evt.data.name || 'tool',
            status: 'running',
            inputPreview: '', outputPreview: '',
          };
          if (!last.tools) last.tools = [];
          last.tools.push(toolEntry);
          if (this.activeChatId === agent) this._scrollChat(agent);
        }
      } else if (evt.type === 'tool_result' && agent && evt.data?.session && evt.data.session !== this._chatSessionId) {
        const hist = this.chatHistories[agent] || [];
        const last = hist[hist.length - 1];
        if (last && last._remoteStream && last.streaming && Array.isArray(last.tools)) {
          const ti = this._findRunningToolIndex(last.tools, evt.data.name);
          if (ti >= 0) {
            last.tools[ti].status = 'done';
            last.tools[ti].outputPreview = this.chatToolPreview(evt.data.output);
          }
          last.phase = 'thinking';
        }
      } else if (evt.type === 'chat_done' && agent) {
        // Another session's chat completed — finalize bubble
        const hist = this.chatHistories[agent] || [];
        const last = hist[hist.length - 1];
        if (last && last._remoteStream && last.streaming) {
          const rResp = evt.data?.response || '';
          if (rResp) {
            if (last._sawTextDelta) {
              // text_delta events already built up last.content — keep as-is.
            } else {
              last.content = last.content
                ? last.content + '\n\n' + rResp
                : rResp;
            }
          }
          last.streaming = false;
          last.phase = 'done';
          delete last._remoteStream;
          if (Array.isArray(last.tools)) {
            last.tools.forEach(t => { if (t.status === 'running') t.status = 'done'; });
          }
        }
        // If the local session has an active SSE stream for this agent,
        // don't clear streaming state or reload history — the SSE finally
        // block will handle cleanup when the stream naturally completes.
        if (!this._chatAborts[agent]) {
          this.chatStreamingAgents[agent] = false;
          this.chatLoadingAgents[agent] = false;
          // Refresh from server to ensure consistency
          delete this._chatFetchedAt[agent];
          this._loadChatHistory(agent);
        }
        this._saveChatToSession();
      } else if (evt.type === 'chat_reset' && agent) {
        // Another session reset this agent's conversation — clear local history
        this.chatHistories[agent] = [];
        this.chatStreamingAgents[agent] = false;
        this.chatLoadingAgents[agent] = false;
        this._stopChatRecovery(agent);
        if (this._chatAborts[agent]) {
          this._chatAborts[agent].abort();
          delete this._chatAborts[agent];
        }
        delete this._chatStreamTarget[agent];
        delete this._chatFetchedAt[agent];
        this.chatUnread = { ...this.chatUnread, [agent]: 0 };
        this._saveChatToSession();
      }

      // Debounced cost panel refresh on llm_call events
      if (evt.type === 'llm_call' && this.activeTab === 'system') {
        if (this._costDebounce) clearTimeout(this._costDebounce);
        this._costDebounce = setTimeout(() => this.fetchCosts(), 2000);
      }

      // Debounced trace refresh when on traces view
      if (this.activeTab === 'system' && this.systemTab === 'activity' && this.activityView === 'traces') {
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
          // Fetch coordination status from blackboard
          this._fetchCoordination();
          // Update operator readiness for the Chat tab
          this.checkOperatorReady();
        }
      } catch (e) {
        console.warn('fetchAgents failed:', e);
        this.connectionError = true;
      }
      this.loading = false;
    },

    async _fetchCoordination() {
      // Only fetch when we have a project with agents
      const proj = this.activeProject;
      if (!proj || !this.agents.length) return;
      try {
        // Fetch status/ and tasks/ entries in parallel
        const [statusResp, tasksResp] = await Promise.all([
          fetch(`${window.__config.apiBase}/blackboard?prefix=status/`),
          fetch(`${window.__config.apiBase}/blackboard?prefix=tasks/`),
        ]);
        if (statusResp.ok) {
          const entries = await statusResp.json();
          const coordStatus = {};
          for (const entry of entries) {
            try {
              const parts = entry.key.split('/');
              if (parts.length >= 2) {
                const agentId = parts[1];
                const val = typeof entry.value === 'string' ? JSON.parse(entry.value) : entry.value;
                if (val && typeof val === 'object') {
                  coordStatus[agentId] = {
                    state: val.state || 'unknown',
                    summary: val.summary || '',
                    ts: val.ts || 0,
                  };
                }
              }
            } catch (_) { /* skip malformed entry */ }
          }
          this.agentCoordStatus = coordStatus;
        }
        if (tasksResp.ok) {
          const entries = await tasksResp.json();
          const counts = {};
          for (const entry of entries) {
            try {
              const parts = entry.key.split('/');
              if (parts.length >= 2) {
                const agentId = parts[1];
                const val = typeof entry.value === 'string' ? JSON.parse(entry.value) : entry.value;
                if (!val || typeof val !== 'object' || val.status !== 'done') {
                  counts[agentId] = (counts[agentId] || 0) + 1;
                }
              }
            } catch (_) { /* skip malformed entry */ }
          }
          this.agentInboxCounts = counts;
        }
      } catch (e) { /* coordination data is supplementary — don't break the dashboard */ }
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
      this.agentActivity = [];
      this.agentFiles = [];
      this.agentFilesPath = '.';
      this.agentFilePreview = null;
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
      if (tab.id === 'activity') {
        await this.fetchAgentActivity(agentId);
      }
      if (tab.id === 'logs') {
        await this.fetchIdentityLogs(agentId);
        await this.fetchIdentityLearnings(agentId);
      }
      if (tab.id === 'capabilities') {
        await this.fetchAgentCapabilities(agentId);
      }
      if (tab.id === 'files') {
        await this.fetchAgentFiles(agentId, '.');
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
          const sources = data.tool_sources || {};
          this.agentCapabilities = defs.map(t => ({
            name: t.function?.name || t.name || '?',
            description: t.function?.description || t.description || '',
            source: sources[t.function?.name || t.name] || 'custom',
          }));
        }
      } catch (e) { console.warn('fetchAgentCapabilities failed:', e); }
    },

    async fetchAgentFiles(agentId, path) {
      this.agentFilesPath = path || '.';
      this.agentFiles = [];
      this.agentFilesLoading = true;
      try {
        const resp = await fetch(
          `${window.__config.apiBase}/agents/${agentId}/files?path=${encodeURIComponent(this.agentFilesPath)}`
        );
        if (resp.ok) {
          const data = await resp.json();
          this.agentFiles = data.entries || [];
        }
      } catch (e) { console.warn('fetchAgentFiles failed:', e); }
      this.agentFilesLoading = false;
    },

    _encodeFilePath(path) {
      // Encode each segment individually to preserve slashes as path separators.
      // encodeURIComponent('/') = '%2F' which breaks {path:path} routing.
      return path.split('/').map(encodeURIComponent).join('/');
    },

    async previewAgentFile(agentId, path) {
      this.agentFilePreview = null;
      this.agentFilePreviewLoading = true;
      try {
        const resp = await fetch(
          `${window.__config.apiBase}/agents/${agentId}/files/${this._encodeFilePath(path)}`
        );
        if (resp.ok) {
          this.agentFilePreview = await resp.json();
        }
      } catch (e) { console.warn('previewAgentFile failed:', e); }
      this.agentFilePreviewLoading = false;
    },

    downloadAgentFile(agentId, path) {
      const url = `${window.__config.apiBase}/agents/${agentId}/files/${this._encodeFilePath(path)}`;
      fetch(url).then(r => r.json()).then(data => {
        let blob;
        if (data.encoding === 'base64') {
          const bin = atob(data.content);
          const arr = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
          blob = new Blob([arr], { type: data.mime_type || 'application/octet-stream' });
        } else {
          blob = new Blob([data.content], { type: data.mime_type || 'text/plain' });
        }
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = path.split('/').pop();
        a.click();
        URL.revokeObjectURL(a.href);
      }).catch(e => this.showToast(`Download failed: ${e.message}`));
    },

    agentFilesParentPath(path) {
      if (!path || path === '.') return null;
      const parts = path.split('/');
      parts.pop();
      return parts.length ? parts.join('/') : '.';
    },

    formatFileSize(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    },

    async fetchUploads() {
      this.uploadsLoading = true;
      this.uploadsError = null;
      try {
        const resp = await fetch(`${window.__config.apiBase}/uploads`);
        if (resp.ok) {
          const data = await resp.json();
          this.uploadsList = data.uploads || [];
        } else {
          this.uploadsError = `Failed to list uploads (${resp.status})`;
        }
      } catch (e) {
        this.uploadsError = e.message;
      }
      this.uploadsLoading = false;
    },

    async handleUploadFiles(files) {
      if (!files || files.length === 0) return;
      this.uploadsUploading = true;
      this.uploadsError = null;
      for (const file of files) {
        try {
          const resp = await fetch(
            `${window.__config.apiBase}/uploads/${encodeURIComponent(file.name)}`,
            { method: 'POST', body: file, headers: { 'Content-Type': file.type || 'application/octet-stream' } }
          );
          if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            this.uploadsError = `Upload failed for "${file.name}": ${err.detail || resp.status}`;
          }
        } catch (e) {
          this.uploadsError = `Upload failed for "${file.name}": ${e.message}`;
        }
      }
      this.uploadsUploading = false;
      await this.fetchUploads();
    },

    downloadUpload(name) {
      const url = `${window.__config.apiBase}/uploads/${encodeURIComponent(name)}/download`;
      const a = document.createElement('a');
      a.href = url;
      a.download = name.split('/').pop();
      a.click();
    },

    async deleteUpload(name) {
      try {
        const resp = await fetch(
          `${window.__config.apiBase}/uploads/${encodeURIComponent(name)}`,
          { method: 'DELETE' }
        );
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          this.uploadsError = `Delete failed: ${err.detail || resp.status}`;
          return;
        }
        this.uploadsList = this.uploadsList.filter(u => u.name !== name);
      } catch (e) {
        this.uploadsError = e.message;
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

    async fetchAgentActivity(agentId) {
      this.agentActivityLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/activity?limit=100`);
        if (resp.ok) {
          const data = await resp.json();
          this.agentActivity = (data.activity || []).reverse();
        }
      } catch (e) { console.warn('fetchAgentActivity failed:', e); }
      this.agentActivityLoading = false;
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
      this.projectHubExpanded = false;
      this.projectHubTab = 'docs';
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
          this.$nextTick(() => { this.renderCostChart(); this.renderModelChart(); });
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
        color: cfg.color ?? null,
        _showColorPicker: false,
        _modelSearch: '',
        _modelDropdownOpen: false,
        budget_daily: cfg.budget?.daily_usd || '',
        budget_monthly: cfg.budget?.monthly_usd || '',
        thinking: cfg.thinking || 'off',
        can_use_browser: cfg.can_use_browser ?? false,
        can_spawn: cfg.can_spawn ?? false,
        can_manage_cron: cfg.can_manage_cron ?? false,
        can_use_wallet: cfg.can_use_wallet ?? false,
        proxy_mode: cfg.proxy?.mode || 'inherit',
        proxy_url: '',
        proxy_username: '',
        proxy_password: '',
        _current_proxy_host: cfg.proxy?.host || '',
        _current_proxy_scheme: cfg.proxy?.scheme || '',
        _current_proxy_has_credential: cfg.proxy?.has_credential || false,
        _showProxyChange: false,
        _walletChains: Object.fromEntries(
          (cfg.wallet_available_chains || []).map(ch => {
            const allowed = cfg.wallet_allowed_chains || [];
            // Default all chains to selected when none were previously configured
            const checked = allowed.length === 0 ? true : allowed.includes(ch.id) || allowed.includes('*');
            return [ch.id, checked];
          })
        ),
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
      const m = model.startsWith('openrouter/') ? model.slice('openrouter/'.length) : model;
      if (m.startsWith('anthropic/')) {
        return [
          { value: 'off', label: 'Off' },
          { value: 'low', label: 'Low (5K budget tokens)' },
          { value: 'medium', label: 'Medium (10K budget tokens)' },
          { value: 'high', label: 'High (25K budget tokens)' },
        ];
      }
      if (m.startsWith('openai/o') || m.startsWith('o1') || m.startsWith('o3') || m.startsWith('o4')) {
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
      if (this.editForm.color !== (cfg.color ?? null)) body.color = this.editForm.color;
      const newDaily = this.editForm.budget_daily ? parseFloat(this.editForm.budget_daily) : null;
      const newMonthly = this.editForm.budget_monthly ? parseFloat(this.editForm.budget_monthly) : null;
      const oldDaily = cfg.budget?.daily_usd ?? null;
      const oldMonthly = cfg.budget?.monthly_usd ?? null;
      if (newDaily !== oldDaily || newMonthly !== oldMonthly) {
        const budget = {};
        if (newDaily !== null && newDaily > 0) budget.daily_usd = newDaily;
        if (newMonthly !== null && newMonthly > 0) budget.monthly_usd = newMonthly;
        body.budget = budget;
      }
      if (this.editForm.thinking !== undefined && this.editForm.thinking !== (cfg.thinking || 'off')) {
        body.thinking = this.editForm.thinking;
      }
      // Handle allowed_credentials + capability flags via the permissions endpoint
      const newCreds = (this.editForm.allowed_credentials || '').split(',').map(s => s.trim()).filter(Boolean);
      const oldCreds = cfg.allowed_credentials || [];
      const credsChanged = JSON.stringify(newCreds) !== JSON.stringify(oldCreds);
      const permBody = {};
      if (credsChanged) permBody.allowed_credentials = newCreds;
      for (const flag of ['can_use_browser', 'can_spawn', 'can_manage_cron', 'can_use_wallet']) {
        if (this.editForm[flag] !== (cfg[flag] ?? false)) permBody[flag] = this.editForm[flag];
      }
      // Wallet allowed chains (from checkbox state)
      const wc = this.editForm._walletChains || {};
      const newChains = Object.keys(wc).filter(k => wc[k]).sort();
      const oldChains = [...(cfg.wallet_allowed_chains || [])].sort();
      if (JSON.stringify(newChains) !== JSON.stringify(oldChains)) {
        permBody.wallet_allowed_chains = newChains;
      }
      const permsChanged = Object.keys(permBody).length > 0;
      const oldProxyMode = cfg.proxy?.mode || 'inherit';
      const proxyChanged = this.editForm.proxy_mode !== oldProxyMode ||
        (this.editForm.proxy_mode === 'custom' && this.editForm.proxy_url);
      if (Object.keys(body).length === 0 && !permsChanged && !proxyChanged) {
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
        if (permsChanged) {
          const permResp = await fetch(`${window.__config.apiBase}/agents/${agentId}/permissions`, {
            method: 'PUT', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(permBody),
          });
          if (permResp.ok) {
            const permResult = await permResp.json();
            allUpdated.push(...permResult.updated);
          } else {
            const err = await permResp.json();
            this.showToast(`Error updating permissions: ${err.detail || 'Update failed'}`);
          }
        }
        if (proxyChanged) {
          const proxyBody = { mode: this.editForm.proxy_mode };
          if (this.editForm.proxy_mode === 'custom') {
            proxyBody.url = this.editForm.proxy_url;
            if (this.editForm.proxy_username) proxyBody.username = this.editForm.proxy_username;
            if (this.editForm.proxy_password) proxyBody.password = this.editForm.proxy_password;
          }
          const proxyResp = await fetch(`${window.__config.apiBase}/agents/${agentId}/proxy`, {
            method: 'PUT', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(proxyBody),
          });
          if (proxyResp.ok) {
            allUpdated.push('proxy');
            if (!configResult) configResult = {};
            configResult.restart_required = true;
          } else {
            const err = await proxyResp.json().catch(() => ({}));
            this.showToast(`Error updating proxy: ${err.detail || 'Update failed'}`);
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

    async resetBrowser(agentId) {
      this.showConfirm('Reset Browser', `Reset the browser session for "${agentId}"? The browser will relaunch with current proxy and settings on next use.`, async () => {
        // Close VNC viewer if open — the session is about to die
        if (this.showBrowserViewer && this.selectedAgent === agentId) {
          this.showBrowserViewer = false;
          this._browserFocusDone = false;
          this._browserToggling = false;
          this._browserViewOnly = true;
        }
        this.showToast(`Resetting browser for ${agentId}...`);
        try {
          const resp = await fetch(`${window.__config.apiBase}/browser/${agentId}/reset`, { method: 'POST' });
          if (resp.ok) {
            this.showToast(`Browser reset for ${agentId}`);
          } else {
            const err = await resp.json().catch(() => ({}));
            this.showToast(`Reset failed: ${err.detail || 'Unknown error'}`);
          }
        } catch (e) { this.showToast(`Error: ${e.message}`); }
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

    async saveOpHeartbeat(schedule) {
      const agent = this.agents.find(a => a.id === 'operator');
      const jobId = agent?.heartbeat_job_id;
      if (!jobId) { this.showToast('No heartbeat job found for operator'); return false; }
      try {
        const resp = await fetch(`${window.__config.apiBase}/cron/${jobId}`, {
          method: 'PUT',
          headers: {'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest'},
          body: JSON.stringify({ schedule }),
        });
        if (resp.ok) {
          this.showToast('Operator heartbeat updated');
          this.fetchAgents();
          this.fetchCronJobs();
          return true;
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Update failed'}`);
          return false;
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); return false; }
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
        if (f.color !== null) payload.color = f.color;
        if (f.project) payload.project = f.project;
        if (f.template) payload.template = f.template;
        const resp = await fetch(`${window.__config.apiBase}/agents`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        if (resp.ok) {
          const data = await resp.json();
          const projectNote = data.project ? ` in ${data.project}` : '';
          this.showToast(data.ready ? `${data.agent} added and ready${projectNote}` : `${data.agent} added (starting)${projectNote}`);
          this.addAgentMode = false;
          this.addAgentForm = { name: '', role: '', model: '', avatar: 1, color: null, project: '', template: '', _showPicker: false, _showColorPicker: false, _templateSearch: '', _templateDropdownOpen: false, _modelSearch: '', _modelDropdownOpen: false };
          this.fetchAgents();
          if (data.project) this.fetchProjects();
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
      // Pre-select the active project if one is open
      if (this.activeProject) this.addAgentForm.project = this.activeProject;
      this.fetchSettings();
      this.fetchAgentTemplates();
      this.fetchProjects();
      this.$nextTick(() => {
        const el = document.getElementById('add-agent-name-input');
        if (el) el.focus();
      });
    },

    closeAddAgentModal() {
      if (this.addAgentLoading) return;
      this.addAgentMode = false;
      this.addAgentForm = { name: '', role: '', model: '', avatar: 1, color: null, project: '', template: '', _showPicker: false, _showColorPicker: false, _templateSearch: '', _templateDropdownOpen: false, _modelSearch: '', _modelDropdownOpen: false };
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
      this.showConfirm('Delete Agent', `Delete agent "${agentId}"? This will stop the container and permanently remove its config.`, async () => {
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
          chatPanelMinimized: this.chatPanelMinimized,
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
          this.fetchAgents();  // Refresh heartbeat_schedule in agent profiles
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

    // ── Browser settings ─────────────────────────────────

    async fetchBrowserSettings() {
      this.browserSettingsLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/browser-settings`);
        if (resp.ok) {
          const data = await resp.json();
          this.browserSpeed = data.speed ?? 1.0;
          this.browserDelay = data.delay ?? 0;
        }
      } catch (e) { console.warn('fetchBrowserSettings failed:', e); }
      this.browserSettingsLoading = false;
    },

    saveBrowserSpeed(value) {
      this.browserSpeed = parseFloat(value);
      // Debounce save — user may be dragging the slider
      if (this._browserSettingsDebounce) clearTimeout(this._browserSettingsDebounce);
      this._browserSettingsDebounce = setTimeout(async () => {
        try {
          await fetch(`${window.__config.apiBase}/browser-settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ speed: this.browserSpeed }),
          });
        } catch (e) { console.warn('saveBrowserSpeed failed:', e); }
      }, 300);
    },

    get browserSpeedLabel() {
      const f = this.browserSpeed;
      if (f >= 3.0) return 'Lightning';
      if (f >= 1.8) return 'Fast';
      if (f >= 1.2) return 'Quick';
      if (f >= 0.8) return 'Normal';
      if (f >= 0.5) return 'Careful';
      return 'Stealth';
    },

    saveBrowserDelay(value) {
      this.browserDelay = parseFloat(value);
      if (this._browserDelayDebounce) clearTimeout(this._browserDelayDebounce);
      this._browserDelayDebounce = setTimeout(async () => {
        try {
          await fetch(`${window.__config.apiBase}/browser-settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ delay: this.browserDelay }),
          });
        } catch (e) { console.warn('saveBrowserDelay failed:', e); }
      }, 300);
    },

    get browserDelayLabel() {
      const d = this.browserDelay;
      if (d <= 0) return 'Off';
      if (d <= 2.0) return 'Light';
      if (d <= 4.0) return 'Moderate';
      if (d <= 7.0) return 'Heavy';
      return 'Maximum';
    },

    async fetchCaptchaSolver() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/captcha-solver`);
        if (resp.ok) {
          const data = await resp.json();
          this.captchaSolverProvider = data.provider || '';
          this.captchaSolverKeyMasked = data.key_masked || '';
        }
      } catch (e) { console.warn('fetchCaptchaSolver failed:', e); }
    },

    async saveCaptchaSolver() {
      this.captchaSolverSaving = true;
      try {
        const body = { provider: this.captchaSolverProvider };
        const keyInput = document.getElementById('captcha-solver-key');
        if (keyInput && keyInput.value) {
          body.key = keyInput.value;
        }
        const resp = await fetch(`${window.__config.apiBase}/captcha-solver`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
          body: JSON.stringify(body),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.captchaSolverProvider = data.provider || '';
          this.captchaSolverKeyMasked = data.key_masked || '';
          if (keyInput) keyInput.value = '';
          this._showToast('CAPTCHA solver settings saved');
        } else {
          const err = await resp.json().catch(() => ({}));
          this._showToast(err.detail || 'Failed to save', 'error');
        }
      } catch (e) {
        console.warn('saveCaptchaSolver failed:', e);
        this._showToast('Failed to save CAPTCHA solver settings', 'error');
      }
      this.captchaSolverSaving = false;
    },

    async removeCaptchaSolver() {
      this.captchaSolverSaving = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/captcha-solver`, {
          method: 'DELETE',
          headers: { 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (resp.ok) {
          this.captchaSolverProvider = '';
          this.captchaSolverKeyMasked = '';
          this._showToast('CAPTCHA solver removed');
        }
      } catch (e) { console.warn('removeCaptchaSolver failed:', e); }
      this.captchaSolverSaving = false;
    },

    // ── Network / Proxy ────────────────────────────────

    async loadNetworkProxy() {
      this.networkProxy.loading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/network/proxy`);
        if (resp.ok) {
          const data = await resp.json();
          this.networkProxy.system_proxy = data.system_proxy || { configured: false, managed: false, managed_url: '', overridden: false, url: '' };
          this.networkProxy.no_proxy = data.no_proxy || '';
          this.networkProxy.agents = data.agents || [];
          // Never pre-fill form with masked URLs — display masked value as
          // read-only info, keep form empty for new input only.
          this.networkProxy.form = { url: '', username: '', password: '' };
        }
      } catch (e) { console.warn('loadNetworkProxy failed:', e); }
      this.networkProxy.loading = false;
    },

    async saveSystemProxy() {
      this.networkProxy.saving = true;
      try {
        const body = { no_proxy: this.networkProxy.no_proxy };
        body.system_proxy = this.networkProxy.form.url ? {
          url: this.networkProxy.form.url,
          username: this.networkProxy.form.username || undefined,
          password: this.networkProxy.form.password || undefined,
        } : null;
        const resp = await fetch(`${window.__config.apiBase}/network/proxy`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'dashboard' },
          body: JSON.stringify(body),
        });
        if (resp.ok) {
          this.networkProxy.saved = true;
          setTimeout(() => this.networkProxy.saved = false, 2000);
          this.showToast('Proxy settings saved — restart agents to apply');
          await this.loadNetworkProxy();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Save failed'}`);
        }
      } catch (e) {
        this.showToast(`Error: ${e.message || String(e)}`);
      }
      this.networkProxy.saving = false;
    },

    async revertSystemProxy() {
      this.networkProxy.removingSystemProxy = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/network/proxy`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'dashboard' },
          body: JSON.stringify({ system_proxy: null, no_proxy: this.networkProxy.no_proxy }),
        });
        if (resp.ok) {
          this.networkProxy.form = { url: '', username: '', password: '' };
          this.showToast(this.networkProxy.system_proxy.managed ? 'Reverted to managed proxy — restart agents to apply' : 'System proxy removed — restart agents to apply');
          await this.loadNetworkProxy();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Remove failed'}`);
        }
      } catch (e) {
        this.showToast(`Error: ${e.message || String(e)}`);
      }
      this.networkProxy.removingSystemProxy = false;
    },

    async saveNoProxy() {
      this.networkProxy.saving = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/network/proxy`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'dashboard' },
          body: JSON.stringify({ no_proxy: this.networkProxy.no_proxy }),
        });
        if (resp.ok) {
          this.showToast('NO_PROXY updated');
          await this.loadNetworkProxy();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Save failed'}`);
        }
      } catch (e) {
        this.showToast(`Error: ${e.message || String(e)}`);
      }
      this.networkProxy.saving = false;
    },

    proxyModeLabel(mode) {
      if (mode === 'inherit') return 'System proxy';
      if (mode === 'custom') return 'Custom proxy';
      if (mode === 'direct') return 'No proxy';
      return mode || '-';
    },

    startAgentProxyEdit(agent) {
      this.networkProxy.editingAgentProxy = agent.agent_id;
      this.networkProxy.agentProxyForm = {
        mode: agent.mode || 'inherit',
        url: '',
        username: '',
        password: '',
      };
    },

    cancelAgentProxyEdit() {
      this.networkProxy.editingAgentProxy = null;
      this.networkProxy.agentProxyForm = { mode: 'inherit', url: '', username: '', password: '' };
    },

    async saveAgentProxy(agentId) {
      if (this.networkProxy.agentProxySaving) return;
      this.networkProxy.agentProxySaving = true;
      try {
        const form = this.networkProxy.agentProxyForm;
        const body = { mode: form.mode };
        if (form.mode === 'custom') {
          body.url = form.url;
          if (form.username) body.username = form.username;
          if (form.password) body.password = form.password;
        }
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/proxy`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'dashboard' },
          body: JSON.stringify(body),
        });
        if (resp.ok) {
          this.cancelAgentProxyEdit();
          try {
            await fetch(`${window.__config.apiBase}/agents/${agentId}/restart`, {
              method: 'POST', headers: { 'X-Requested-With': 'dashboard' },
            });
            this.showToast(`Proxy updated for ${agentId} — restarting`);
          } catch (_) {
            this.showToast(`Proxy updated for ${agentId} — restart failed, do it manually`);
          }
          await this.loadNetworkProxy();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Update failed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.networkProxy.agentProxySaving = false; }
    },

    // ── System settings ─────────────────────────────────

    async fetchSystemSettings() {
      this.systemSettingsLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/system-settings`);
        if (resp.ok) {
          this.systemSettings = await resp.json();
        }
      } catch (e) { console.warn('fetchSystemSettings failed:', e); }
      this.systemSettingsLoading = false;
    },

    saveSystemSetting(key, value) {
      if (!this.systemSettings) return;
      const strKeys = ['image_gen_provider'];
      if (strKeys.includes(key)) {
        this.systemSettings[key] = value;
      } else {
        const typ = ['default_daily_budget', 'default_monthly_budget'].includes(key) ? parseFloat : parseInt;
        this.systemSettings[key] = typ(value);
      }
      if (this._systemSettingsDebounce) clearTimeout(this._systemSettingsDebounce);
      this._systemSettingsDebounce = setTimeout(async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/system-settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ [key]: this.systemSettings[key] }),
          });
          if (resp.ok) {
            const data = await resp.json();
            if (data.updated?.length) this.showToast(`Updated ${data.updated.join(', ')}`);
          } else {
            const err = await resp.json().catch(() => ({}));
            this.showToast(`Error: ${err.detail || 'Update failed'}`);
          }
        } catch (e) { console.warn('saveSystemSetting failed:', e); }
      }, 500);
    },

    async saveDefaultModel(model) {
      if (!model) return;
      try {
        const resp = await fetch(`${window.__config.apiBase}/default-model`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model }),
        });
        if (resp.ok) {
          if (this.systemSettings) this.systemSettings.default_model = model;
          this.showToast(`Default model set to ${model}`);
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Update failed'}`);
        }
      } catch (e) { console.warn('saveDefaultModel failed:', e); }
    },

    restartAllAgents() {
      this.showConfirm('Restart All Agents', 'Restart all agents and the browser service to apply settings changes? This will interrupt any active work.', async () => {
        this._restartingAll = true;
        this.showToast('Restarting all agents...');
        try {
          const resp = await fetch(`${window.__config.apiBase}/restart-agents`, { method: 'POST' });
          if (resp.ok) {
            const data = await resp.json();
            const agents = Object.entries(data.restarted || {});
            const ok = agents.filter(([, s]) => s === 'ready').length;
            this.showToast(`Restarted ${ok}/${agents.length} agents`);
            this.fetchAgents();
          } else {
            const err = await resp.json().catch(() => ({}));
            this.showToast(`Error: ${err.detail || 'Restart failed'}`);
          }
        } catch (e) { this.showToast(`Error: ${e.message}`); }
        this._restartingAll = false;
      }, true);
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

    formatNumber(n) {
      if (n == null) return '0';
      return Number(n).toLocaleString();
    },

    async fetchDatabaseDetails() {
      this.dbDetailsLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/storage/databases`);
        if (resp.ok) {
          const data = await resp.json();
          this.dbDetails = data.databases || [];
        }
      } catch (e) { console.warn('fetchDatabaseDetails failed:', e); }
      this.dbDetailsLoading = false;
    },

    async purgeDatabase(dbId, olderThanDays) {
      const db = (this.dbDetails || []).find(d => d.id === dbId);
      const label = db ? db.label : dbId;
      const desc = olderThanDays
        ? `Delete records older than ${olderThanDays} days from ${label}. This cannot be undone.`
        : `Delete ALL records from ${label}. This cannot be undone.`;

      this.showConfirm(`Purge ${label}?`, desc, async () => {
        this.dbPurging[dbId] = true;
        try {
          const body = olderThanDays ? { older_than_days: olderThanDays } : {};
          const resp = await fetch(`${window.__config.apiBase}/storage/databases/${dbId}/purge`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'dashboard' },
            body: JSON.stringify(body),
          });
          if (resp.ok) {
            const data = await resp.json();
            this.showToast(`Purged ${data.deleted_records.toLocaleString()} records from ${label}`);
            await Promise.all([this.fetchDatabaseDetails(), this.fetchStorage()]);
          } else {
            const err = await resp.json().catch(() => ({}));
            this.showToast(`Purge failed: ${err.detail || resp.status}`);
          }
        } catch (e) { this.showToast(`Purge failed: ${e.message}`); }
        delete this.dbPurging[dbId];
      }, true);
    },

    // ── Chat slide-over panel ──────────────────────────

    async _loadChatHistory(agentId) {
      // Always fetch from server — the persistent transcript is the
      // source of truth, ensuring history is consistent across devices.
      // Skip if streaming (avoid clobber), recently fetched (debounce), or
      // a stream just finished (server may not have persisted the message yet).
      if (this.chatStreamingAgents[agentId] || this._chatAborts[agentId]) return;
      const now = Date.now();
      if (this._chatFetchedAt[agentId] && (now - this._chatFetchedAt[agentId]) < 5000) return;
      const lastEnd = this._chatStreamEndAt?.[agentId] || 0;
      if (now - lastEnd < 3000) return;
      this._chatFetchedAt[agentId] = now;
      try {
        const resp = await fetch(`/dashboard/api/agents/${agentId}/chat/history`, { credentials: 'same-origin' });
        if (!resp.ok) return;
        const data = await resp.json();
        const localMsgs = this.chatHistories[agentId] || [];
        if (!data.messages || data.messages.length === 0) {
          // Server returned empty — agent may be restarting or transcript not yet
          // initialized. Preserve local messages to avoid losing visible history.
          return;
        }
        // Re-check after await — streaming may have started during the fetch.
        if (this.chatStreamingAgents[agentId] || this._chatAborts[agentId]) return;
        const serverMsgs = data.messages.map(m => ({
          role: m.role === 'assistant' ? 'agent' : m.role,
          content: m.content,
          streaming: false,
          phase: 'done',
          ts: (m.ts || 0) < 1e12 ? (m.ts || 0) * 1000 : (m.ts || 0),
          tools: Array.isArray(m.tools) ? m.tools.map(t =>
            typeof t === 'string' ? { name: t, status: 'done', inputPreview: '', outputPreview: '' } : t
          ) : [],
        }));
        // Preserve local user messages not yet on the server (e.g., sent
        // right before tab-out, before the server could persist them).
        const lastServerTs = Math.max(...data.messages.map(m => {
          const t = m.ts || 0;
          return t < 1e12 ? t * 1000 : t;
        }));
        // Preserve local messages not yet on the server:
        // 1. User messages sent after the last server timestamp
        // 2. Agent messages from just-completed streams not yet persisted
        // 3. Notifications injected via WebSocket not yet in the server transcript
        const trailing = localMsgs.filter(m => {
          if (m.role === 'notification') {
            if ((m.ts || 0) <= lastServerTs) return false;
            return !serverMsgs.some(s =>
              s.role === 'notification' && s.content === m.content && Math.abs((s.ts || 0) - (m.ts || 0)) < 2000
            );
          }
          if ((m.role !== 'user' && m.role !== 'agent') || (m.ts || 0) <= lastServerTs) return false;
          if (m.role === 'agent' && !m.content) return false;  // Drop empty agent placeholders
          // Skip if server already has a message with matching content
          return !serverMsgs.some(s => s.role === m.role && s.content === m.content && Math.abs((s.ts || 0) - (m.ts || 0)) < 10000);
        });
        this.chatHistories[agentId] = trailing.length > 0
          ? [...serverMsgs, ...trailing]
          : serverMsgs;
        this._saveChatToSession();
        this.$nextTick(() => this._scrollChat(agentId));
      } catch (e) {
        console.debug('_loadChatHistory failed for', agentId, e.message || e);
      }
    },

    async _recoverDeadStreams() {
      // After tab return, check if any formerly-streaming chats had their
      // SSE connection killed by the browser.  If the agent is still busy,
      // show a thinking indicator and poll until the response is complete.
      const wasStreaming = this._chatWasStreaming || {};
      this._chatWasStreaming = {};
      const agentIds = Object.keys(wasStreaming).filter(id => !this.chatStreamingAgents[id]);
      if (agentIds.length === 0) return;
      // Fetch queue status once for all agents
      try {
        const resp = await fetch(`${window.__config.apiBase}/queues`);
        if (resp.ok) this.queueStatus = (await resp.json()).queues;
      } catch (_) { /* ignore */ }
      for (const agentId of agentIds) {
        if (!this.queueStatus?.[agentId]?.busy) {
          // Agent finished — _loadChatHistory will pick up the completed response
          continue;
        }
        // Agent is still working — show thinking state and poll for completion
        this.chatLoadingAgents[agentId] = true;
        this.chatStreamingAgents[agentId] = true;
        // Add a fresh thinking bubble if the last message isn't already one
        if (!this.chatHistories[agentId]) this.chatHistories[agentId] = [];
        const hist = this.chatHistories[agentId];
        const last = hist[hist.length - 1];
        if (!last || last.role !== 'agent' || last.phase === 'done' || last.phase === 'error') {
          hist.push({
            role: 'agent', content: '', streaming: true,
            phase: 'thinking', tools: [], timeline: [], _sawTextDelta: false,
            ts: Date.now(),
          });
          this._pushChatTimelinePhase(hist[hist.length - 1], 'thinking');
        } else {
          last.streaming = true;
          last.phase = 'thinking';
        }
        this.$nextTick(() => this._scrollChat(agentId));
        // Poll every 3s — refresh history and check if agent is done
        this._chatRecoveryPolls[agentId] = setInterval(async () => {
          try {
            const qr = await fetch(`${window.__config.apiBase}/queues`);
            if (qr.ok) this.queueStatus = (await qr.json()).queues;
          } catch (_) { /* ignore */ }
          const stillBusy = this.queueStatus?.[agentId]?.busy;
          // Always refresh history to pick up incremental progress
          delete this._chatFetchedAt[agentId];
          // Temporarily allow _loadChatHistory to run by clearing streaming flag
          if (!stillBusy) {
            this._stopChatRecovery(agentId);
            this._loadChatHistory(agentId);
          }
        }, 3000);
      }
    },

    _stopChatRecovery(agentId) {
      if (this._chatRecoveryPolls[agentId]) {
        clearInterval(this._chatRecoveryPolls[agentId]);
        delete this._chatRecoveryPolls[agentId];
      }
      this.chatLoadingAgents[agentId] = false;
      this.chatStreamingAgents[agentId] = false;
      // Finalize any streaming bubble
      const hist = this.chatHistories[agentId] || [];
      for (const msg of hist) {
        if (msg.streaming) {
          msg.streaming = false;
          if (msg.phase !== 'error') msg.phase = 'done';
        }
      }
    },

    openChat(agentId) {
      this.chatPanelMinimized = false;
      if (this.openChats.includes(agentId)) {
        this.activeChatId = agentId;
        this._loadChatHistory(agentId);
        this.$nextTick(() => {
          this._scrollChat(agentId, true);
          if (this.chatUnread[agentId]) this.chatUnread = { ...this.chatUnread, [agentId]: 0 };
          const input = document.getElementById('chat-slide-input');
          if (input) input.focus();
        });
        return;
      }
      this.openChats.push(agentId);
      this.activeChatId = agentId;
      this._loadChatHistory(agentId);
      this.$nextTick(() => {
        this._scrollChat(agentId, true);
        if (this.chatUnread[agentId]) this.chatUnread = { ...this.chatUnread, [agentId]: 0 };
      });
    },

    closeChat(agentId) {
      if (this._chatAborts[agentId]) {
        this._chatAborts[agentId].abort();
        delete this._chatAborts[agentId];
      }
      delete this._chatStreamTarget[agentId];
      this._stopChatRecovery(agentId);
      this.openChats = this.openChats.filter(id => id !== agentId);
      // Switch to next open chat or clear
      if (this.activeChatId === agentId) {
        this.activeChatId = this.openChats.length > 0 ? this.openChats[this.openChats.length - 1] : '';
      }
      this._saveChatToSession();
    },

    clearChat(agentId) {
      if (this._chatAborts[agentId]) {
        this._chatAborts[agentId].abort();
        delete this._chatAborts[agentId];
      }
      delete this._chatStreamTarget[agentId];
      this._stopChatRecovery(agentId);
      this.chatHistories[agentId] = [];
      delete this._chatFetchedAt[agentId];
      this._saveChatToSession();
    },

    _scrollTimers: {},

    _scrollChat(agentId, force) {
      if (this._scrollTimers[agentId]) return;
      this._scrollTimers[agentId] = setTimeout(() => {
        delete this._scrollTimers[agentId];
        const el = document.getElementById('chat-messages-' + agentId);
        if (!el) return;
        // Only auto-scroll if user is near the bottom (within 150px) or forced
        const nearBottom = force || (el.scrollHeight - el.scrollTop - el.clientHeight < 150);
        if (nearBottom) el.scrollTop = el.scrollHeight;
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

    chatToolHasDetail(tool) {
      return !!(tool.input || tool.inputPreview || tool.output || tool.outputPreview || tool.status === 'running');
    },

    chatToolDetailText(value, preview) {
      const full = this._chatToolValueToText(value);
      return this._truncateText(full || preview || '', 4000);
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

    _normalizeEventTs(evt) {
      if (!evt.timestamp) return Date.now();
      if (typeof evt.timestamp === 'number') return evt.timestamp < 1e12 ? evt.timestamp * 1000 : evt.timestamp;
      const parsed = new Date(evt.timestamp).getTime();
      return isNaN(parsed) ? Date.now() : parsed;
    },

    _findRunningToolIndex(tools, name) {
      for (let i = tools.length - 1; i >= 0; i -= 1) {
        if (tools[i].name === name && tools[i].status === 'running') return i;
      }
      return -1;
    },

    chatHeaderStatus(agentId) {
      if (!this.chatStreamingAgents[agentId] && !this.chatLoadingAgents[agentId]) {
        return { label: 'Online', color: 'text-gray-600', dot: 'bg-gray-600' };
      }
      if (this.chatLoadingAgents[agentId]) {
        return { label: 'Thinking...', color: 'text-purple-400', dot: 'bg-purple-400' };
      }
      const hist = this.chatHistories[agentId] || [];
      for (let i = hist.length - 1; i >= 0; i--) {
        const msg = hist[i];
        if (msg.role === 'agent' && msg.streaming) {
          if (msg.phase === 'tool') return { label: 'Using tools...', color: 'text-blue-400', dot: 'bg-blue-400' };
          if (msg.phase === 'thinking') return { label: 'Thinking...', color: 'text-purple-400', dot: 'bg-purple-400' };
          if (msg.phase === 'responding') return { label: 'Responding...', color: 'text-green-400', dot: 'bg-green-400' };
        }
      }
      return { label: 'Responding...', color: 'text-green-400', dot: 'bg-green-400' };
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
      this._chatStreamTarget[agentId] = idx;
      this._pushChatTimelinePhase(this.chatHistories[agentId][idx], 'thinking');
      this.$nextTick(() => this._scrollChat(agentId, true));

      const controller = new AbortController();
      this._chatAborts[agentId] = controller;
      // Idle timeout: abort if no SSE data received for 120s.  Resets on
      // each chunk so long-running tool chains stay alive indefinitely.
      let streamTimeout = setTimeout(() => {
        if (this.chatStreamingAgents[agentId]) controller.abort();
      }, 120000);
      const _resetStreamTimeout = () => {
        clearTimeout(streamTimeout);
        streamTimeout = setTimeout(() => {
          if (this.chatStreamingAgents[agentId]) controller.abort();
        }, 120000);
      };

      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/chat/stream`, {
          method: 'POST', headers: {'Content-Type': 'application/json', 'X-Chat-Session': this._chatSessionId},
          body: JSON.stringify({ message: msg }),
          signal: controller.signal,
        });
        if (!resp.ok) {
          let errMsg = `HTTP ${resp.status}`;
          try { const err = await resp.json(); errMsg = err.detail || errMsg; } catch (_) {}
          const isCreditError = resp.status === 402 || /insufficient.*(fund|credit)|credit.*deplet|payment.*required/i.test(errMsg);
          this.chatHistories[agentId][idx].content = errMsg;
          this.chatHistories[agentId][idx].role = isCreditError ? 'credit_exhausted' : 'error';
          if (isCreditError) this.chatHistories[agentId][idx]._creditExhausted = true;
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
          _resetStreamTimeout();
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            let data;
            try { data = JSON.parse(line.slice(6)); } catch (_) { continue; }

            const entry = this.chatHistories[agentId][this._chatStreamTarget[agentId]];

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
              if (finalResponse) {
                if (entry._sawTextDelta) {
                  // text_delta events already streamed content into
                  // entry.content (including the final response) — keep as-is
                  // so intermediate messages aren't overwritten.
                } else {
                  entry.content = entry.content
                    ? entry.content + '\n\n' + finalResponse
                    : finalResponse;
                }
              }
              entry.streaming = false;
              entry.phase = 'done';
              entry.tool_limit_reached = data.tool_limit_reached || false;
            } else if (data.type === 'error') {
              const errContent = data.message || 'Stream error';
              const isCreditErr = /insufficient.*(fund|credit)|credit.*deplet|budget.*exceed/i.test(errContent);
              entry.content = errContent;
              entry.role = isCreditErr ? 'credit_exhausted' : 'error';
              if (isCreditErr) entry._creditExhausted = true;
              else entry._recoverable = true;
              entry.streaming = false;
              entry.phase = 'error';
              this._pushChatTimelinePhase(entry, 'error');
            }
          }
        }
        const finalIdx = this._chatStreamTarget[agentId];
        this.chatHistories[agentId][finalIdx].streaming = false;
        if (this.chatHistories[agentId][finalIdx].role !== 'error' && this.chatHistories[agentId][finalIdx].phase !== 'done') {
          this.chatHistories[agentId][finalIdx].phase = 'done';
        }
      } catch (e) {
        const entry = this.chatHistories[agentId]?.[this._chatStreamTarget[agentId]];
        if (entry) {
          const hadContent = !!(entry.content || (entry.tools && entry.tools.length > 0));
          if (hadContent) {
            // Stream interrupted (timeout, tab backgrounded, network) but we have partial data.
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
            entry.content = e.name === 'AbortError'
              ? 'Response timed out — please try again.'
              : 'Connection interrupted — please try again.';
            entry.role = 'error';
            entry._recoverable = true;
            entry.streaming = false;
            entry.phase = 'error';
            this._pushChatTimelinePhase(entry, 'error');
          }
        }
      } finally {
        clearTimeout(streamTimeout);
        delete this._chatAborts[agentId];
        delete this._chatStreamTarget[agentId];
        this.chatLoadingAgents[agentId] = false;
        this.chatStreamingAgents[agentId] = false;
        if (!this._chatStreamEndAt) this._chatStreamEndAt = {};
        this._chatStreamEndAt[agentId] = Date.now();
        this.$nextTick(() => this._scrollChat(agentId));
        this._saveChatToSession();
        this.fetchQueues();
        // Auto-recover from stream interruptions: the agent may still be
        // processing via non-streaming fallback or may have already finished.
        const _lastMsg = (this.chatHistories[agentId] || []).slice(-1)[0];
        if (_lastMsg && _lastMsg._recoverable) {
          setTimeout(async () => {
            if (!this.openChats.includes(agentId)) return;
            try {
              const qr = await fetch(`${window.__config.apiBase}/queues`);
              if (qr.ok) this.queueStatus = (await qr.json()).queues;
            } catch (_) { /* ignore */ }
            if (this.queueStatus?.[agentId]?.busy) {
              this._chatWasStreaming = this._chatWasStreaming || {};
              this._chatWasStreaming[agentId] = true;
              this._recoverDeadStreams();
            } else {
              delete this._chatFetchedAt[agentId];
              this._loadChatHistory(agentId);
            }
          }, 4000);
        }
      }
    },

    isAgentBusy(agentId) {
      if (this.chatStreamingAgents[agentId]) return true;
      return this.queueStatus?.[agentId]?.busy === true;
    },

    async steerAgent(agentId, message) {
      const msg = (message || '').trim();
      if (!msg) return;
      if (!this.chatHistories[agentId]) this.chatHistories[agentId] = [];

      // If an SSE stream is active, finalize the current bubble so the
      // agent's continued response appears below the steer message.
      if (this.chatStreamingAgents[agentId] && this._chatStreamTarget[agentId] !== undefined) {
        const oldIdx = this._chatStreamTarget[agentId];
        const oldEntry = this.chatHistories[agentId][oldIdx];
        if (oldEntry && oldEntry.streaming) {
          if (Array.isArray(oldEntry.tools)) {
            oldEntry.tools.forEach(t => { if (t.status === 'running') t.status = 'done'; });
          }
          if (Array.isArray(oldEntry.timeline)) {
            oldEntry.timeline.forEach(t => { if (t.status === 'running') t.status = 'done'; });
          }
          oldEntry.streaming = false;
          oldEntry.phase = 'done';
        }
      }

      this.chatHistories[agentId].push({ role: 'user', content: `[steer] ${msg}`, ts: Date.now() });

      // Create a new agent response bubble after the steer message
      if (this.chatStreamingAgents[agentId]) {
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
        const newIdx = this.chatHistories[agentId].length - 1;
        this._chatStreamTarget[agentId] = newIdx;
        this._pushChatTimelinePhase(this.chatHistories[agentId][newIdx], 'thinking');
        this.$nextTick(() => this._scrollChat(agentId, true));
      }

      try {
        await fetch(`${window.__config.apiBase}/agents/${agentId}/steer`, {
          method: 'POST', headers: {'Content-Type': 'application/json', 'X-Chat-Session': this._chatSessionId},
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
        system: ['system', 'costs', 'cron', 'schedules', 'automation', 'credentials', 'api keys', 'connections', 'integrations', 'infrastructure', 'pricing', 'browsers', 'pubsub', 'blackboard', 'comms', 'communication', 'workflows', 'storage', 'uploads', 'disk', 'network', 'proxy', 'socks'],
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
        { label: 'Broadcast', desc: this.activeProject ? `Broadcast to ${this.activeProject} agents` : (this.projects.length > 0 ? 'Broadcast to standalone agents' : 'Send message to all agents'), keywords: ['broadcast', 'send', 'all', 'message'], action: () => { this.switchTab('fleet'); if (this.activeProject) { this.projectHubExpanded = true; this.projectHubTab = 'broadcast'; this.$nextTick(() => document.getElementById('broadcast-input')?.focus()); } else { this.$nextTick(() => document.getElementById('broadcast-standalone-input')?.focus()); } } },
        ...(this.activeProject ? [{ label: 'Edit PROJECT.md', desc: `Edit ${this.activeProject} project context`, keywords: ['project', 'edit', 'context'], action: () => { this.switchTab('fleet'); this.projectHubExpanded = true; this.projectHubTab = 'docs'; this.$nextTick(() => this.startProjectEdit()); } }] : []),
        ...(this.activeProject ? [{ label: 'Project Members', desc: `Manage ${this.activeProject} members`, keywords: ['members', 'team', 'assign', 'agents'], action: () => { this.switchTab('fleet'); this.projectHubExpanded = true; this.projectHubTab = 'members'; } }] : []),
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
          results.push({ type: 'action', label: name, desc: 'API Key', action: () => { this.systemTab = 'apikeys'; this.switchTab('system'); } });
        }
      }
      // System quick actions
      const sysActions = [
        { label: 'Activity', desc: 'Traces, live feed, and logs', keywords: ['activity', 'traces', 'events', 'live'], action: () => { this.systemTab = 'activity'; this.switchTab('system'); } },
        { label: 'View Logs', desc: 'Open runtime logs', keywords: ['logs', 'runtime', 'debug'], action: () => { this.systemTab = 'activity'; this.switchTab('system'); this.setActivityView('logs'); } },
        { label: 'Add API Key', desc: 'Add new API key or credential', keywords: ['key', 'api', 'credential', 'token'], action: () => { this.systemTab = 'apikeys'; this.switchTab('system'); this.showCredForm = true; } },
        { label: 'Manage Webhooks', desc: 'View and create webhooks', keywords: ['webhook', 'hook', 'endpoint'], action: () => { this.systemTab = 'integrations'; this.switchTab('system'); this.fetchWebhooks(); } },
        { label: 'Manage Channels', desc: 'Connect Telegram, Discord, Slack, WhatsApp', keywords: ['channel', 'telegram', 'discord', 'slack', 'whatsapp'], action: () => { this.systemTab = 'integrations'; this.switchTab('system'); this.fetchChannels(); } },
        { label: 'Model Pricing', desc: 'Token costs by model', keywords: ['model', 'pricing', 'tokens'], action: () => { this.systemTab = 'costs'; this.switchTab('system'); } },
        { label: 'Browser Settings', desc: 'Browser speed, delay, and timing', keywords: ['browser', 'speed', 'delay', 'settings', 'timing', 'stealth'], action: () => { this.systemTab = 'settings'; this.switchTab('system'); this.fetchBrowserSettings(); } },
        { label: 'Default Model', desc: 'Change the default LLM model', keywords: ['model', 'llm', 'default', 'openai', 'anthropic', 'ollama'], action: () => { this.systemTab = 'settings'; this.switchTab('system'); this.fetchSystemSettings(); } },
        { label: 'Budget Settings', desc: 'Default daily and monthly budgets', keywords: ['budget', 'cost', 'daily', 'monthly', 'limit', 'spend'], action: () => { this.systemTab = 'settings'; this.switchTab('system'); this.fetchSystemSettings(); } },
        { label: 'Agent Limits', desc: 'Max iterations, tool rounds, timeouts', keywords: ['iterations', 'rounds', 'timeout', 'limit', 'agent', 'execution'], action: () => { this.systemTab = 'settings'; this.switchTab('system'); this.fetchSystemSettings(); } },
        { label: 'Health Settings', desc: 'Poll interval, failure thresholds, restart limits', keywords: ['health', 'poll', 'restart', 'failure', 'recovery', 'monitor'], action: () => { this.systemTab = 'settings'; this.switchTab('system'); this.fetchSystemSettings(); } },
        { label: 'Operator Settings', desc: 'Operator model, status, and audit log', keywords: ['audit', 'history', 'changes', 'operator', 'log', 'settings', 'model'], action: () => { this.switchSystemTab('operator'); this.switchTab('system'); } },
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
      this.showConfirm('Reset Conversation', `Start a fresh conversation with "${agentId}"? The conversation thread is wiped, but memories and skills are preserved.`, async () => {
        try {
          // Abort active stream and recovery before clearing history
          if (this._chatAborts[agentId]) {
            this._chatAborts[agentId].abort();
            delete this._chatAborts[agentId];
          }
          delete this._chatStreamTarget[agentId];
          this._stopChatRecovery(agentId);
          const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/reset`, { method: 'POST' });
          if (resp.ok) {
            this.showToast(`${agentId} conversation reset`);
            this.chatHistories[agentId] = [];
            delete this._chatFetchedAt[agentId];
            this._saveChatToSession();
          } else {
            this.showToast('Reset failed');
          }
        } catch (e) { this.showToast(`Error: ${e.message}`); }
      }, true);
    },

    // ── Browser ────────────────────────────────────────

    async focusBrowser(agentId) {
      try {
        const resp = await fetch(`${window.__config.apiBase}/browser/${agentId}/focus`, { method: 'POST' });
        const data = await resp.json().catch(() => ({ success: false }));
        if (!data.success) {
          this.showToast('Browser focus failed — agent may not have a browser running', 5000);
          return false;
        }
        return true;
      } catch (e) {
        console.warn('focusBrowser failed:', e);
        this.showToast('Could not connect to browser service', 5000);
        return false;
      }
    },

    _getVncUrl() {
      const match = this.agents.find(ag => ag.vnc_url);
      return match ? match.vnc_url : '';
    },

    async _completeBrowserLogin(msg, agentId) {
      const prev = { completed: msg.completed, cancelled: msg.cancelled };
      msg.completed = true;
      try {
        const resp = await fetch(window.__config.apiBase + '/browser-login/complete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
          body: JSON.stringify({ agent_id: agentId || '', service: msg.service }),
        });
        if (!resp.ok) {
          msg.completed = prev.completed;
          this.showToast('Failed to notify agent — please try again');
        }
      } catch (_) {
        msg.completed = prev.completed;
        this.showToast('Network error — please try again');
      }
    },

    async _cancelBrowserLogin(msg, agentId) {
      const prev = { completed: msg.completed, cancelled: msg.cancelled };
      msg.cancelled = true;
      try {
        const resp = await fetch(window.__config.apiBase + '/browser-login/cancel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
          body: JSON.stringify({ agent_id: agentId || '', service: msg.service }),
        });
        if (!resp.ok) {
          msg.cancelled = prev.cancelled;
          this.showToast('Failed to notify agent — please try again');
        }
      } catch (_) {
        msg.cancelled = prev.cancelled;
        this.showToast('Network error — please try again');
      }
    },

    async toggleBrowser() {
      if (this.showBrowserViewer) {
        await this.closeBrowserViewer();
        return;
      }
      if (this._browserToggling) return;
      this._browserToggling = true;
      try {
        const agentId = this.selectedAgent;
        // Show loading overlay before the focus call so the user gets
        // immediate feedback, but do NOT set the iframe src yet —
        // otherwise KasmVNC connects and renders whatever window
        // happens to be on top *before* focus completes.
        this._browserPendingAgent = agentId;
        this.showBrowserViewer = true;
        this._browserFocusDone = false;
        this.$nextTick(() => {
          if (this.$refs.vncLoading) {
            this.$refs.vncLoading.style.opacity = '1';
          }
        });
        // Await focus so the correct agent's window is raised before
        // the iframe connects to KasmVNC.
        if (agentId) {
          const ok = await this.focusBrowser(agentId);
          if (!ok) {
            this.showBrowserViewer = false;
            this._browserFocusDone = false;
            return;
          }
        }
        // Staleness guard: if the user switched agents while we were
        // awaiting, abandon — the new agent's toggle will handle it.
        if (this.selectedAgent !== agentId) return;
        this._browserFocusDone = true;
      } finally {
        this._browserToggling = false;
      }
    },

    async toggleBrowserControl() {
      const taking = this._browserViewOnly;  // currently view-only → taking control
      this._browserViewOnly = !taking;

      const agentId = this.selectedAgent;
      if (!agentId) return;
      try {
        await fetch(`${window.__config.apiBase}/browser/${agentId}/control`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
          body: JSON.stringify({ user_control: taking }),
        });
      } catch (e) {
        console.warn('toggleBrowserControl failed:', e);
      }
    },

    async closeBrowserViewer() {
      const wasControlling = !this._browserViewOnly;
      this.showBrowserViewer = false;
      this._browserFocusDone = false;
      this._browserToggling = false;
      this._browserViewOnly = true;

      // Release agent X11 pause if user had control
      if (wasControlling && this.selectedAgent) {
        try {
          await fetch(`${window.__config.apiBase}/browser/${this.selectedAgent}/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify({ user_control: false }),
          });
        } catch (e) {
          console.warn('closeBrowserViewer release failed:', e);
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
      const isKeyless = this.credService === 'ollama';
      if (!service || (!isKeyless && !this.credKey.trim())) return;
      if (isKeyless && !this.credBaseUrl.trim()) return;
      this.credSaving = true;
      const isOAuth = this.credAuthType === 'oauth' && (this.credService === 'anthropic' || this.credService === 'openai');
      if (isKeyless) {
        this.showToast('Saving Ollama configuration...');
      } else {
        this.showToast(isOAuth ? 'Validating token...' : 'Validating API key...');
      }
      try {
        const baseUrl = isOAuth ? '' : this.credBaseUrl.trim();
        const key = isKeyless ? 'ollama' : this.credKey.trim();
        if (!isKeyless && !isOAuth && !await this._validateCredential(service, key, baseUrl)) return;
        const body = { service, key };
        if (this.credService === '__custom__' && this.credTier === 'system') body.tier = 'system';
        if (this.credService === '__custom__' && this.credTier === 'system' && this.credIsLlmProvider && this.credCustomModels.trim()) {
          body.custom_llm_models = this.credCustomModels;
          if (this.credCustomLabel.trim()) body.custom_llm_label = this.credCustomLabel;
        }
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
          this.credIsLlmProvider = false;
          this.credCustomModels = '';
          this.credCustomLabel = '';
          this._credServiceDropdownOpen = false;
          this._credServiceSearch = '';
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

    // ── Credential reveal ──

    async revealCredential(name) {
      if (this.revealedCredentials[name]) {
        // Toggle off
        delete this.revealedCredentials[name];
        this.revealedCredentials = { ...this.revealedCredentials };
        return;
      }
      this.revealingCredential = name;
      try {
        const resp = await fetch(`${window.__config.apiBase}/credentials/${encodeURIComponent(name)}/value`);
        if (resp.ok) {
          const data = await resp.json();
          this.revealedCredentials = { ...this.revealedCredentials, [name]: data.value };
          // Auto-hide after 30 seconds
          setTimeout(() => {
            delete this.revealedCredentials[name];
            this.revealedCredentials = { ...this.revealedCredentials };
          }, 30000);
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Could not reveal credential'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.revealingCredential = null; }
    },

    // ── Wallet management ──

    async fetchWallet() {
      this.walletLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/wallet/addresses`);
        if (resp.ok) {
          this.walletData = await resp.json();
        }
      } catch (e) { console.warn('fetchWallet failed:', e); }
      finally { this.walletLoading = false; }
    },

    async initWallet() {
      if (this.walletInitializing) return;
      this.showConfirm(
        'Initialize Wallet',
        'This will generate a master seed for all agent wallets. The seed will be displayed once — make sure to save it securely.',
        async () => {
          this.walletInitializing = true;
          try {
            const resp = await fetch(`${window.__config.apiBase}/wallet/init`, { method: 'POST' });
            if (resp.ok) {
              const data = await resp.json();
              this.walletSeed = data.seed;
              this.walletSeedVisible = true;
              this.showToast('Wallet initialized. Save your seed phrase!');
              await this.fetchWallet();
            } else {
              const err = await resp.json().catch(() => ({}));
              this.showToast(`Error: ${err.detail || 'Initialization failed'}`);
            }
          } catch (e) { this.showToast(`Error: ${e.message}`); }
          finally { this.walletInitializing = false; }
        }
      );
    },

    async revealWalletSeed() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/wallet/seed`);
        if (resp.ok) {
          const data = await resp.json();
          this.walletSeed = data.seed;
          this.walletSeedVisible = true;
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Could not retrieve seed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
    },

    hideWalletSeed() {
      this.walletSeed = '';
      this.walletSeedVisible = false;
    },

    async enableWalletForAgent(agentId) {
      try {
        const resp = await fetch(`${window.__config.apiBase}/wallet/enable/${agentId}`, { method: 'POST' });
        if (resp.ok) {
          this.showToast(`Wallet enabled for ${agentId}`);
          await this.fetchWallet();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Failed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
    },

    async fetchWalletRpc() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/wallet/rpc`);
        if (resp.ok) {
          const data = await resp.json();
          this.walletRpcChains = data.chains || [];
        }
      } catch (e) { console.warn('fetchWalletRpc failed:', e); }
    },

    startRpcEdit(chain) {
      this.walletRpcEditing = chain.chain_id;
      this.walletRpcValue = chain.is_custom ? chain.rpc_current : '';
    },

    cancelRpcEdit() {
      this.walletRpcEditing = null;
      this.walletRpcValue = '';
    },

    async saveRpcUrl(chainId) {
      if (this.walletRpcSaving) return;
      this.walletRpcSaving = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/wallet/rpc`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ chain_id: chainId, rpc_url: this.walletRpcValue.trim() }),
        });
        if (resp.ok) {
          this.showToast(this.walletRpcValue.trim() ? 'RPC updated' : 'Reset to default');
          this.walletRpcEditing = null;
          this.walletRpcValue = '';
          await this.fetchWalletRpc();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Update failed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.walletRpcSaving = false; }
    },

    copyToClipboard(text) {
      navigator.clipboard.writeText(text).then(
        () => this.showToast('Copied to clipboard'),
        () => this.showToast('Copy failed — select and copy manually'),
      );
    },

    async submitOnboardCredential() {
      if (this.onboardSaving) return;
      if (!this.onboardProvider || !this.onboardKey.trim()) return;
      this.onboardSaving = true;
      const isOAuth = this.onboardAuthType === 'oauth' && (this.onboardProvider === 'anthropic' || this.onboardProvider === 'openai');
      this.showToast(isOAuth ? 'Validating token...' : 'Validating API key...');
      try {
        const baseUrl = isOAuth ? '' : this.onboardBaseUrl.trim();
        const isOnboardOAuth = this.onboardAuthType === 'oauth' && (this.onboardProvider === 'anthropic' || this.onboardProvider === 'openai');
        const service = this.onboardProvider === '__custom__' ? this.onboardCustomService.trim() : this.onboardProvider;
        if (!isOnboardOAuth && this.onboardProvider !== '__custom__' && !await this._validateCredential(service, this.onboardKey.trim(), baseUrl)) return;
        const body = { service, key: this.onboardKey.trim() };
        if (this.onboardProvider === '__custom__') body.tier = 'system';
        if (baseUrl) body.base_url = baseUrl;
        if (this.onboardProvider === '__custom__' && this.onboardIsLlmProvider && this.onboardCustomModels.trim()) {
          body.custom_llm_models = this.onboardCustomModels;
          if (this.onboardCustomLabel.trim()) body.custom_llm_label = this.onboardCustomLabel;
        }
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
          this.onboardCustomService = '';
          this.onboardIsLlmProvider = false;
          this.onboardCustomModels = '';
          this.onboardCustomLabel = '';
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
        if (this.webhookFormInstructions.trim()) body.instructions = this.webhookFormInstructions.trim();
        if (this.webhookFormRequireSig) body.secret = true;
        const resp = await fetch(`${window.__config.apiBase}/webhooks`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (resp.ok) {
          const data = await resp.json();
          const hook = data.hook || {};
          if (hook.secret) {
            this.webhookRevealedSecret = hook.secret;
            this.webhookRevealedSecretHookId = hook.id;
            if (navigator.clipboard) navigator.clipboard.writeText(hook.secret).catch(() => {});
            this.showToast(`Webhook "${name}" created — copy the secret below`, 8000);
          } else if (hook.url && navigator.clipboard) {
            navigator.clipboard.writeText(hook.url).catch(() => {});
            this.showToast(`Webhook "${name}" created — URL copied`);
          } else {
            this.showToast(`Webhook "${name}" created`);
          }
          this.webhookFormName = '';
          this.webhookFormAgent = '';
          this.webhookFormInstructions = '';
          this.webhookFormRequireSig = false;
          this.showWebhookForm = false;
          this.fetchWebhooks();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Failed to create webhook'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.webhookCreating = false; }
    },

    async deleteWebhook(hookId, name) {
      this.showConfirm('Delete Webhook', `Delete webhook "${name}"?`, async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/webhooks/${encodeURIComponent(hookId)}`, { method: 'DELETE' });
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

    async testWebhook(hookId) {
      if (this.webhookTesting[hookId]) return;
      this.webhookTesting = { ...this.webhookTesting, [hookId]: true };
      try {
        const resp = await fetch(`${window.__config.apiBase}/webhooks/${encodeURIComponent(hookId)}/test`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
        if (resp.ok) {
          this.showToast('Webhook test sent');
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Test failed: ${err.detail || 'Unknown error'}`);
        }
      } catch (e) { this.showToast(`Test failed: ${e.message}`); }
      finally { this.webhookTesting = { ...this.webhookTesting, [hookId]: false }; }
    },

    editWebhook(wh) {
      this.showWebhookForm = false;
      this.editingWebhookId = wh.id;
      this.webhookEditName = wh.name;
      this.webhookEditAgent = wh.agent;
      this.webhookEditInstructions = wh.instructions || '';
      this.webhookEditRequireSig = !!wh.has_secret;
    },

    cancelWebhookEdit() {
      this.editingWebhookId = null;
      this.webhookEditName = '';
      this.webhookEditAgent = '';
      this.webhookEditInstructions = '';
      this.webhookEditRequireSig = false;
    },

    async saveWebhookEdit() {
      if (this.webhookSaving) return;
      const id = this.editingWebhookId;
      if (!id) return;
      const wh = this.webhooks.find(w => w.id === id);
      if (!wh) return;
      const name = this.webhookEditName.trim();
      const agent = this.webhookEditAgent;
      if (!name || !agent) { this.showToast('Name and agent are required'); return; }

      const body = {};
      if (name !== wh.name) body.name = name;
      if (agent !== wh.agent) body.agent = agent;
      const newInstr = this.webhookEditInstructions.trim();
      const oldInstr = (wh.instructions || '').trim();
      if (newInstr !== oldInstr) body.instructions = newInstr;
      const hadSecret = !!wh.has_secret;
      if (this.webhookEditRequireSig !== hadSecret) body.require_signature = this.webhookEditRequireSig;

      if (Object.keys(body).length === 0) { this.showToast('No changes to save'); return; }

      this.webhookSaving = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/webhooks/${encodeURIComponent(id)}`, {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (resp.ok) {
          const data = await resp.json();
          const hook = data.hook || {};
          if (hook.secret) {
            this.webhookRevealedSecret = hook.secret;
            this.webhookRevealedSecretHookId = hook.id || this.editingWebhookId;
            if (navigator.clipboard) navigator.clipboard.writeText(hook.secret).catch(() => {});
            this.showToast(`Webhook updated — copy the secret below`, 8000);
          } else {
            this.showToast(`Webhook "${name}" updated`);
          }
          this.cancelWebhookEdit();
          this.fetchWebhooks();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Update failed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.webhookSaving = false; }
    },

    regenerateWebhookSecret(hookId, name) {
      this.showConfirm('Regenerate Secret', `Regenerate HMAC secret for "${name}"? The old secret will stop working immediately.`, async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/webhooks/${encodeURIComponent(hookId)}`, {
            method: 'PATCH', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ regenerate_secret: true }),
          });
          if (resp.ok) {
            const data = await resp.json();
            const hook = data.hook || {};
            if (hook.secret) {
              this.webhookRevealedSecret = hook.secret;
              this.webhookRevealedSecretHookId = hook.id || hookId;
              if (navigator.clipboard) navigator.clipboard.writeText(hook.secret).catch(() => {});
              this.showToast('New secret shown below — copy it now', 8000);
            } else {
              this.showToast('Secret regenerated');
            }
            this.cancelWebhookEdit();
            this.fetchWebhooks();
          } else {
            const err = await resp.json().catch(() => ({}));
            this.showToast(`Error: ${err.detail || 'Regeneration failed'}`);
          }
        } catch (e) { this.showToast(`Error: ${e.message}`); }
      }, true);
    },

    // ── External API key ──────────────────────────────────

    async fetchApiKeys() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/external-api-keys`);
        if (resp.ok) {
          const data = await resp.json();
          this.apiKeys = data.keys || [];
          this.apiKeysLegacy = data.legacy || false;
        }
      } catch (e) { console.warn('fetchApiKeys failed:', e); }
    },

    async createApiKey() {
      const name = this.apiKeyNewName.trim();
      if (!name || this.apiKeyCreating) return;
      this.apiKeyCreating = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/external-api-keys`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.apiKeyNewValue = data.key;
          if (navigator.clipboard) navigator.clipboard.writeText(data.key).catch(() => {});
          this.showToast(`API key "${name}" created — copy it from below`, 8000);
          this.apiKeyNewName = '';
          this.showApiKeyForm = false;
          this.fetchApiKeys();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Creation failed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message}`); }
      finally { this.apiKeyCreating = false; }
    },

    revokeApiKey(keyId, name) {
      this.showConfirm('Revoke API Key', `Revoke "${name}"? Any integration using this key will lose access immediately.`, async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/external-api-keys/${encodeURIComponent(keyId)}`, { method: 'DELETE' });
          if (resp.ok) {
            this.showToast(`API key "${name}" revoked`);
            this.fetchApiKeys();
          } else {
            const err = await resp.json().catch(() => ({}));
            this.showToast(`Error: ${err.detail || 'Revoke failed'}`);
          }
        } catch (e) { this.showToast(`Error: ${e.message}`); }
      }, true);
    },

    // ── Agent drill-down ──────────────────────────────────

    drillDown(agentId) {
      this._detailReturnProject = this.activeProject;
      this.activeTab = 'fleet';
      this.selectedAgent = agentId;
      this.detailAgent = agentId;
      this.showBrowserViewer = false;
      this._browserFocusDone = false;
      this._browserPendingAgent = null;
      this._browserToggling = false;
      this.identityTab = 'config';
      this.identityFiles = [];
      this.identityContent = {};
      this.identityEditing = false;
      this.identityEditBuffer = '';
      this.identityEditingFile = null;
      this.configEditing = false;
      this.identityLogs = null;
      this.identityLearnings = null;
      this.agentActivity = [];
      this.agentFiles = [];
      this.agentFilesPath = '.';
      this.agentFilePreview = null;
      this.fetchAgentDetail(agentId);
      this.fetchIdentityFiles(agentId);
      this.fetchAgentConfig(agentId);
      this.fetchCronJobs();
      // Phase 7 §10.1 — seed per-agent browser metrics history so the
      // detail panel renders sparklines immediately on open. Subsequent
      // updates flow through the existing browser_metrics WS handler.
      this.fetchBrowserMetricsHistory(agentId);
      this.activeTab = 'fleet';
      if (!this._skipPush) this._pushUrl(false);
    },

    // ── Chart.js rendering ────────────────────────────────

    _AGENT_CHART_COLORS: [
      '#6366f1', '#06b6d4', '#10b981', '#f59e0b',
      '#ef4444', '#ec4899', '#8b5cf6', '#14b8a6',
      '#84cc16', '#f97316', '#0ea5e9', '#f43f5e',
      '#d946ef', '#eab308', '#22c55e', '#64748b',
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

    _MODEL_CHART_COLORS: [
      '#8b5cf6', '#10b981', '#3b82f6', '#f59e0b',
      '#ef4444', '#06b6d4', '#ec4899', '#14b8a6',
      '#f97316', '#84cc16', '#d946ef', '#64748b',
    ],

    renderModelChart() {
      const canvas = document.getElementById('modelChart');
      if (!canvas) return;

      const models = (this.costData.by_model || []);
      if (models.length === 0) {
        if (this.modelChart) { this.modelChart.destroy(); this.modelChart = null; }
        return;
      }

      const labels = models.map(m => this.formatModelName(m.model));
      const costs = models.map(m => m.cost);
      const colors = models.map((_, i) => this._MODEL_CHART_COLORS[i % this._MODEL_CHART_COLORS.length]);

      // Always recreate — closures in legend/tooltip/centerText must reflect current totals
      if (this.modelChart) { this.modelChart.destroy(); this.modelChart = null; }

      this.modelChart = new Chart(canvas, {
        type: 'doughnut',
        data: {
          labels,
          datasets: [{
            data: costs,
            backgroundColor: colors,
            borderColor: colors.map(c => c + '40'),
            borderWidth: 2,
            hoverOffset: 4,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '65%',
          plugins: {
            legend: {
              position: 'right',
              labels: {
                color: '#9ca3af',
                usePointStyle: true,
                pointStyleWidth: 8,
                font: { size: 11 },
                padding: 12,
                generateLabels(chart) {
                  const ds = chart.data.datasets[0].data;
                  const total = ds.reduce((s, v) => s + v, 0);
                  return chart.data.labels.map((label, i) => {
                    const pct = total > 0 ? Math.round((ds[i] / total) * 100) : 0;
                    return {
                      text: `${label}  ${pct}%`,
                      fillStyle: chart.data.datasets[0].backgroundColor[i],
                      strokeStyle: 'transparent',
                      pointStyle: 'rectRounded',
                      index: i,
                      hidden: false,
                    };
                  });
                },
              },
            },
            tooltip: {
              callbacks: {
                label(ctx) {
                  const cost = ctx.parsed;
                  const total = ctx.chart.data.datasets[0].data.reduce((s, v) => s + v, 0);
                  const pct = total > 0 ? ((cost / total) * 100).toFixed(1) : '0';
                  return ` $${cost < 0.01 ? cost.toFixed(4) : cost.toFixed(2)}  (${pct}%)`;
                },
              },
            },
          },
        },
        plugins: [{
          id: 'centerText',
          afterDraw(chart) {
            const { ctx, chartArea } = chart;
            const total = chart.data.datasets[0].data.reduce((s, v) => s + v, 0);
            const cx = (chartArea.left + chartArea.right) / 2;
            const cy = (chartArea.top + chartArea.bottom) / 2;
            ctx.save();
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#9ca3af';
            ctx.font = '500 11px ui-monospace, monospace';
            ctx.fillText('total', cx, cy - 10);
            ctx.fillStyle = '#f3f4f6';
            ctx.font = '600 16px ui-monospace, monospace';
            ctx.fillText('$' + (total < 0.01 ? total.toFixed(4) : total.toFixed(2)), cx, cy + 8);
            ctx.restore();
          },
        }],
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
        heartbeat_complete: 'text-pink-300',
        cron_change: 'text-pink-400',
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
        browser_metrics: 'text-sky-400',
        browser_nav_probe: 'text-amber-400',
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
        heartbeat_complete: 'bg-pink-300',
        cron_change: 'bg-pink-400',
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
        browser_metrics: 'bg-sky-400',
        browser_nav_probe: 'bg-amber-400',
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

    // ── Browser metrics card helpers (Phase 2 §5.1/§5.2) ───────────────
    _BROWSER_METRICS_EVICT_MS: 30 * 60 * 1000,  // 30 minutes

    browserMetricsList() {
      // Evict agents that haven't reported in 30+ minutes so stopped /
      // renamed / deleted agents eventually fall off the card. Without
      // this, ghost rows accumulate forever.
      const cutoff = Date.now() - this._BROWSER_METRICS_EVICT_MS;
      const fresh = {};
      for (const [agent, m] of Object.entries(this.browserMetrics)) {
        if ((m.receivedAt || 0) >= cutoff) fresh[agent] = m;
      }
      if (Object.keys(fresh).length !== Object.keys(this.browserMetrics).length) {
        this.browserMetrics = fresh;
      }
      return Object.entries(fresh)
        .map(([agent, m]) => ({ agent, ...m }))
        .sort((a, b) => a.agent.localeCompare(b.agent));
    },
    fmtClickRate(rate) {
      if (rate == null) return '—';
      return (rate * 100).toFixed(0) + '%';
    },
    clickRateColor(rate) {
      if (rate == null) return 'text-gray-500';
      if (rate >= 0.9) return 'text-green-400';
      if (rate >= 0.7) return 'text-yellow-400';
      return 'text-red-400';
    },
    fmtBytes(n) {
      if (n == null || !Number.isFinite(n)) return '—';
      if (n < 1024) return n + 'B';
      if (n < 1024 * 1024) return (n / 1024).toFixed(1) + 'KB';
      return (n / (1024 * 1024)).toFixed(1) + 'MB';
    },
    browserMetricsAge(receivedAt) {
      if (!receivedAt) return '';
      const secs = Math.max(0, Math.round((Date.now() - receivedAt) / 1000));
      if (secs < 60) return secs + 's ago';
      const mins = Math.round(secs / 60);
      return mins + 'm ago';
    },
    browserMetricsStale(receivedAt) {
      // Emit cadence is 60s, so a payload older than ~3 minutes means the
      // browser stopped or the poller broke.
      return receivedAt && Date.now() - receivedAt > 3 * 60 * 1000;
    },

    // ── Per-agent browser metrics panel (Phase 7 §10.1) ─────────────────
    _appendBrowserMetricsHistory(agentId, payload) {
      // §6.3 navigator-probe payloads also flow through `browser_metrics`
      // when emitted as drain payloads, but the dashboard receives them as
      // a separate `browser_nav_probe` event handled below — skip any
      // probe-shaped payload here defensively so the trend bars only show
      // per-minute drains.
      if (!payload || payload.kind === 'nav_probe') return;
      const existing = (this.browserMetricsHistory[agentId] || []).slice();
      // Dedup by seq when present — WS replay (reconnect) plus the seed
      // fetch both deliver the same payload; we want a single entry.
      const seq = payload.seq;
      if (seq != null && existing.some(p => p.seq === seq)) return;
      existing.push({ ...payload, _receivedAt: Date.now() });
      // Stable sort by seq (or ts as a fallback) to keep the rendered
      // sparkline left-to-right ordered even if WS and HTTP fetch race.
      existing.sort((a, b) => {
        const sa = a.seq != null ? a.seq : (a.ts || 0);
        const sb = b.seq != null ? b.seq : (b.ts || 0);
        return sa - sb;
      });
      while (existing.length > this._BROWSER_METRICS_HISTORY_MAX) {
        existing.shift();
      }
      this.browserMetricsHistory = {
        ...this.browserMetricsHistory,
        [agentId]: existing,
      };
      if (seq != null) {
        this._browserMetricsSeq = {
          ...this._browserMetricsSeq,
          [agentId]: Math.max(this._browserMetricsSeq[agentId] || 0, seq),
        };
      }
    },

    async fetchBrowserMetricsHistory(agentId) {
      // Seed the per-agent rolling history from the dashboard endpoint when
      // the detail panel opens — WS events only deliver payloads received
      // since this dashboard session started, so a freshly-loaded panel
      // would otherwise show empty until the next 60s tick.
      if (!agentId) return;
      this._browserMetricsLoading = {
        ...this._browserMetricsLoading,
        [agentId]: true,
      };
      this._browserMetricsError = {
        ...this._browserMetricsError,
        [agentId]: '',
      };
      // Phase 7 third-pass: always seed from seq=0 rather than the
      // WS-derived high-water mark. The race we have to handle is:
      // (1) WS subscribes at app boot and starts populating history
      //     and `_browserMetricsSeq[agentId]` from live emits; then
      // (2) operator opens the agent panel and calls this seeder.
      // If we paginate from `_browserMetricsSeq[agentId]`, we'd ask
      // upstream "give me payloads with seq > N" — and since N is
      // the *latest* seq we already received via WS, we'd get an
      // empty list, leaving the rolling-hour history that lived in
      // the buffer BEFORE the panel opened invisible. The browser
      // service ring buffer is bounded (1024 entries service-wide),
      // so requesting from 0 is cheap; `_appendBrowserMetricsHistory`
      // dedupes by seq so any overlap with WS-derived entries is a
      // no-op.
      const since = 0;
      try {
        const resp = await fetch(
          `${window.__config.apiBase}/agents/${agentId}/browser/metrics?since=${since}`,
        );
        if (!resp.ok) {
          this._browserMetricsError = {
            ...this._browserMetricsError,
            [agentId]: 'HTTP ' + resp.status,
          };
          return;
        }
        const data = await resp.json();
        if (data.success === false) {
          this._browserMetricsError = {
            ...this._browserMetricsError,
            [agentId]: (data.error && data.error.code) || 'unknown',
          };
          return;
        }
        // Boot-id change → server seq counter reset; flush local history
        // so we don't render impossible gaps where seqs jumped backwards.
        const prevBootId = this._browserMetricsBootId[agentId] || '';
        if (data.boot_id && prevBootId && prevBootId !== data.boot_id) {
          this.browserMetricsHistory = {
            ...this.browserMetricsHistory,
            [agentId]: [],
          };
          this._browserMetricsSeq = {
            ...this._browserMetricsSeq,
            [agentId]: 0,
          };
        }
        this._browserMetricsBootId = {
          ...this._browserMetricsBootId,
          [agentId]: data.boot_id || '',
        };
        for (const p of data.metrics || []) {
          this._appendBrowserMetricsHistory(agentId, p);
        }
        // Bump the high-water mark even when no payloads belonged to this
        // agent so subsequent calls page forward.
        const cs = data.current_seq || 0;
        if (cs > (this._browserMetricsSeq[agentId] || 0)) {
          this._browserMetricsSeq = {
            ...this._browserMetricsSeq,
            [agentId]: cs,
          };
        }
      } catch (e) {
        this._browserMetricsError = {
          ...this._browserMetricsError,
          [agentId]: 'network',
        };
      } finally {
        this._browserMetricsLoading = {
          ...this._browserMetricsLoading,
          [agentId]: false,
        };
      }
    },

    browserHistoryFor(agentId) {
      return this.browserMetricsHistory[agentId] || [];
    },

    browserHistoryClickRate(agentId) {
      // Aggregate success rate over the rolling window — total successes /
      // total clicks. Returns null when there's been no click traffic so
      // the panel can render "—" instead of a misleading 0%.
      const hist = this.browserHistoryFor(agentId);
      let succ = 0;
      let fail = 0;
      for (const p of hist) {
        succ += p.click_success || 0;
        fail += p.click_fail || 0;
      }
      if (succ + fail === 0) return null;
      return succ / (succ + fail);
    },

    browserHistorySnapshotCount(agentId) {
      const hist = this.browserHistoryFor(agentId);
      let total = 0;
      for (const p of hist) total += p.snapshot_count || 0;
      return total;
    },

    browserHistoryNavTimeouts(agentId) {
      const hist = this.browserHistoryFor(agentId);
      let total = 0;
      for (const p of hist) total += p.nav_timeout || 0;
      return total;
    },

    browserHistoryTrendArrow(agentId) {
      // Compare the last data point's success rate to the second-to-last.
      // Returns 'up' / 'down' / 'flat' / null. Fewer than two non-null
      // points → null (no signal yet).
      const hist = this.browserHistoryFor(agentId);
      const ratios = [];
      for (const p of hist) {
        const denom = (p.click_success || 0) + (p.click_fail || 0);
        if (denom > 0) {
          ratios.push((p.click_success || 0) / denom);
        }
      }
      if (ratios.length < 2) return null;
      const last = ratios[ratios.length - 1];
      const prev = ratios[ratios.length - 2];
      const delta = last - prev;
      if (Math.abs(delta) < 0.05) return 'flat';
      return delta > 0 ? 'up' : 'down';
    },

    browserHistoryBars(agentId, field, max) {
      // Build a list of {height, label, value} entries for the CSS sparkline.
      // ``field`` is one of: 'snapshot_bytes_p50', 'snapshot_bytes_p95',
      // 'snapshot_count', 'click_total' (computed). ``max`` clamps the
      // upper bound for height normalization; null = autoscale.
      const hist = this.browserHistoryFor(agentId);
      if (hist.length === 0) return [];
      const values = hist.map(p => {
        if (field === 'click_total') {
          return (p.click_success || 0) + (p.click_fail || 0);
        }
        return p[field] || 0;
      });
      const peak = max != null
        ? max
        : Math.max(1, ...values);
      return values.map((v, i) => {
        const heightPct = peak > 0 ? Math.min(100, (v / peak) * 100) : 0;
        const p = hist[i];
        return {
          height: heightPct,
          value: v,
          ts: p.ts,
          seq: p.seq,
        };
      });
    },

    browserHistoryActiveCount(agentId) {
      // Active means we received a payload in the last 3 minutes (matches
      // browserMetricsStale threshold) — instances that went idle drop to
      // zero on the dashboard even though the BrowserManager may still
      // hold a stopped CamoufoxInstance.
      const last = this.browserMetrics[agentId];
      if (!last) return 0;
      return this.browserMetricsStale(last.receivedAt) ? 0 : 1;
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
        case 'heartbeat_complete':
          add('Summary', d.summary);
          if (d.duration_ms) add('Duration', d.duration_ms + 'ms');
          add('Tokens', d.tokens_used);
          add('Tools', d.tools_used);
          add('Outcome', d.outcome);
          break;
        case 'cron_change':
          add('Action', d.action);
          add('Job', d.job_id);
          if (d.schedule) add('Schedule', d.schedule);
          if (d.count) add('Count', d.count);
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
        case 'heartbeat_complete':
          return [d.outcome || 'done', (d.summary || '').substring(0, 60)].filter(Boolean).join(' \u00b7 ');
        case 'cron_change':
          return [d.action, d.job_id, d.schedule].filter(Boolean).join(' \u00b7 ');
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
      const palette = this._AGENT_CHART_COLORS.length; // 16
      // Explicitly configured color takes highest priority.
      const cfg = this.agentConfigs[agentId];
      if (cfg && cfg.color != null) return cfg.color % palette;
      const agent = this.agents.find(a => a.id === agentId);
      if (agent && agent.color != null) return agent.color % palette;
      // Position-based assignment over the active fleet so no two
      // running agents share a color (sorted for determinism).
      if (this.agents && this.agents.length > 0) {
        const sorted = [...new Set(this.agents.map(a => a.id))].sort();
        const idx = sorted.indexOf(agentId);
        if (idx !== -1) return idx % palette;
      }
      // Fallback for agent IDs referenced only in logs / events.
      let hash = 0;
      for (let i = 0; i < agentId.length; i++) {
        hash = ((hash << 5) - hash) + agentId.charCodeAt(i);
        hash |= 0;
      }
      return Math.abs(hash) % palette;
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
      if (agentId === 'operator') {
        const v = window.__config.assetVersion || '';
        return `/dashboard/static/avatars/operator.png` + (v ? `?v=${v}` : '');
      }
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
      'output/': 'output',
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
      // Coordination-aware verbs
      if (item.key) {
        if (item.key.startsWith('tasks/')) return 'handed off to';
        if (item.key.startsWith('status/')) return 'updated status';
        if (item.key.startsWith('output/')) return 'shared output';
      }
      return 'wrote';
    },

    commsActionTarget(item) {
      if (item.source === 'pubsub') return item.topic || '?';
      // Coordination-aware targets: show agent name instead of raw key
      if (item.key) {
        if (item.key.startsWith('tasks/')) {
          const parts = item.key.split('/');
          return parts.length >= 2 ? parts[1] : item.key;
        }
        if (item.key.startsWith('status/')) {
          const parts = item.key.split('/');
          return parts.length >= 2 ? parts[1] : item.key;
        }
      }
      return item.key || '?';
    },

    commsActionClass(item) {
      if (item.source === 'pubsub') return 'text-purple-400';
      if (item.action === 'delete') return 'text-red-400';
      if (item.action === 'cas_write') return 'text-amber-400';
      // Coordination-aware colors
      if (item.key) {
        if (item.key.startsWith('tasks/')) return 'text-blue-400';
        if (item.key.startsWith('status/')) return 'text-green-400';
        if (item.key.startsWith('output/')) return 'text-indigo-400';
      }
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
