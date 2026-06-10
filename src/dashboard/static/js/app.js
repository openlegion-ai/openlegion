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
  { id: 'skills', label: 'Skills', file: null, access: 'auto' },
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
    { file: 'HEARTBEAT.md', label: 'Auto-checkup', cap: null, access: 'both', desc: 'Rules for what the agent does during periodic autonomous wakeups — what to check, what to work on, when to notify you.' },
  ],
};

function dashboard() {
  return {
    // Navigation
    activeTab: 'chat',
    tabs: [
      { id: 'chat', label: 'Chat' },
      // Phase 2 Board UX overhaul — engineer-speak → user-speak
      // (Agents → Team, Board → Work, System → Settings).
      // Work sits 2nd: it's the second-most-visited tab after Chat.
      { id: 'workplace', label: 'Work' },
      { id: 'fleet', label: 'Teams' },
      { id: 'system', label: 'Settings' },
    ],
    // Side panel toggle for non-Chat tabs (Phase 1 Decision 5). Persists
    // across navigation via Alpine root scope; localStorage carries it
    // through reloads so users keep their messenger open as they wander.
    messengerSidePanelOpen: false,
    // Per-agent visible message count for "Load older →" pagination
    // (Phase 1 Decision 12). Default 50; click appends 50 more.
    _chatVisibleLimit: {},
    // Notifications dropdown — top-right bell. Subtle gray indicator.
    notificationsOpen: false,
    connected: false,
    loading: true,
    lastRefresh: 0,
    toastQueue: [],

    // Operator readiness
    operatorReady: false,

    // Phase 3 — Operator default "Quick actions" chips. Rendered before
    // the user has typed anything in this conversation OR after a long
    // pause (>5 min since last user message). Hidden once the user types
    // or sends. ``_operatorLastUserMessageTs`` tracks the last user-sent
    // timestamp so the menu can re-appear after pauses without storing
    // anything server-side.
    operatorDefaultChips: [
      "What's happening?",
      'Show me what we delivered',
      'Add someone to my team',
      'Pause everything',
    ],
    _operatorLastUserMessageTs: 0,

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
      'credential_request_cancelled',
      'browser_login_request', 'browser_login_completed', 'browser_login_cancelled',
      'browser_captcha_help_request', 'browser_captcha_help_completed', 'browser_captcha_help_cancelled',
      // Task 9 — Workplace tab + pending action review
      'task_created', 'task_status_changed', 'task_outcome',
      'pending_action_created', 'pending_action_resolved', 'pending_action_expired',
      // PR 1 — soft-edit receipts + undo
      'operator_action_receipt', 'operator_action_receipt_undone',
      'operator_action_receipt_superseded',
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

    // Cookie / session import (operator-only, modal-triggered from
    // the agent ⋯ actions menu). Field state lives on the modal's own
    // `x-data` (template x-if) so cookie text never persists in DOM
    // after close.
    cookieImportOpen: false,

    // Add agent
    addAgentMode: false,
    addAgentForm: { name: '', role: '', model: '', avatar: 1, color: null, team: '', template: '', _showPicker: false, _showColorPicker: false, _templateSearch: '', _templateDropdownOpen: false, _modelSearch: '', _modelDropdownOpen: false },
    addAgentLoading: false,
    agentTemplates: [],

    // Teams tab — chip strip expand/collapse when many teams. Collapsed
    // state caps chips at 2 rows via CSS so the panel doesn't dominate
    // the page; "Show all" toggle reveals the rest. Persisted so the
    // user's preference survives reloads.
    teamFilterExpanded: false,

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

    // TEAM.md (per-team only)
    teamContent: '',
    teamExists: false,
    teamLoading: false,
    teamEditing: false,
    teamEditBuffer: '',
    teamSaving: false,

    // Multi-team support
    teams: [],
    activeTeam: null,
    teamsLoaded: false,

    // Team management
    showTeamForm: false,
    newTeamName: '',
    newTeamDesc: '',
    teamFormLoading: false,

    // Costs
    costData: {},
    costPeriod: 'today',
    costChart: null,
    modelChart: null,
    _costDebounce: null,

    // Traces
    traces: [],
    tracesLoading: false,
    costsLoading: false,
    cronLoading: false,
    storageLoading: false,

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

    // Per-agent privacy-safe session sidecar summary (Phase 10 §20).
    // Keyed by agent_id; value is the `data` object returned by
    // /api/agents/{id}/session: {has_persisted_session, saved_at,
    // origin_count, cookie_count}. Drives the "Saved sessions" line
    // in the agent settings Browser detail row. Counts only — no
    // origin domains.
    agentBrowserSession: {},

    captchaSolverProvider: '',
    captchaSolverKeyMasked: '',
    captchaSolverSaving: false,

    // Per-platform success rollup (24h rolling window; aggregated by
    // the dashboard from browser_metrics EventBus payloads).  Refreshed
    // on a sensible cadence — not the noisy 1s timer; 30s aligns with
    // the operator's expected debugging cadence and also keeps the
    // load on the dashboard process modest when the panel is left open.
    platformSuccessData: { platforms: [], since: null },
    platformSuccessLoading: false,
    _platformSuccessDebounce: null,

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
    // Phase 0 baseline telemetry guards. ``_firstActionTracked`` prevents
    // ``time_to_first_action`` from firing more than once per page load.
    // ``_dockOpen`` shadows the side-panel state so dockOpen/dockClose
    // helpers are idempotent — Phase 1 wires the actual UI; for now the
    // helpers are no-ops at the UI level but still emit telemetry so the
    // event is wireable from any future caller.
    _firstActionTracked: false,
    _dockOpen: false,
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
    // Re-entrance flag for the operator SOUL/INSTRUCTIONS confirm gate.
    // Set true inside the confirm modal's action so the recursive
    // saveIdentityFile() call skips the gate.
    _operatorIdentityConfirmed: false,
    configEditing: false,
    configSaving: false,
    identityLogs: null,
    agentCapabilities: null,
    // Per-agent MCP side-channels surfaced by the agent /capabilities
    // endpoint (PR 1 of #959). agentMcpStatus[] is the per-server
    // startup/discovery registry — drives status dots and click-to-see-
    // error in the MCP Servers panel. agentMcpToolMap is tool_name →
    // server_name; reserved for tool-filtering UX in a follow-up.
    agentMcpStatus: [],
    agentMcpToolMap: {},
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
    _detailReturnTeam: null,

    // Identity file save flash
    identitySavedFile: null,

    // Command palette (Cmd+K)
    cmdPaletteOpen: false,
    cmdPaletteQuery: '',
    cmdPaletteResults: [],
    cmdPaletteIdx: 0,

    // Workplace tab (peer of Chat / Agents / System). The Work tab
    // lands directly on summary cards — operator-composed daily
    // narratives the user can accept / acknowledge / rework. The
    // Goals chip strip renders above when the operator has tracked
    // any business goals. Needs You panel sits on top; Stuck Tasks
    // panel below summaries; Cancel modal + Task drill-in are
    // reachable from Needs You and the notification bell.
    // ``workplaceEnabled`` reflects whether the tasks store
    // responded — kept for graceful empty-state rendering on
    // transient errors.
    workplaceEnabled: true,
    workplaceTeams: [],
    workplaceTasks: [],
    workplaceBlockers: [],
    // Workplace-wide business goals (PR 2). Operator manages via
    // ``manage_goals``; dashboard renders a read-only chip strip
    // between Needs You (top sticky) and the Summary cards. Hidden
    // entirely when empty.
    workplaceGoals: [],
    // Work summaries surface (PR-B). One row per team per period. The
    // user rates via three SVG icon buttons (accept / acknowledge /
    // rework) with optional feedback on rework; rating + feedback
    // flow back into operator's next composition. List is ordered
    // newest-first by ``generated_at``.
    workplaceSummaries: [],
    // In-flight operator-rooted pipelines (live card). Each entry:
    // {root_task_id, title, assignee, created_at, stages[], summary{}, stalled}.
    // Loaded from /api/workplace/pipelines, WS-debounce-refreshed on task
    // status changes so the card tracks chains as they move.
    workplacePipelines: [],
    // Per-summary inline feedback box state. Keyed by summary id;
    // value is the current draft feedback string. The user opens the
    // box by clicking the rework icon → enters reason → submits →
    // entry is cleared.
    summaryFeedbackDrafts: {},
    // Monotonic per-summary rating sequence. ``rateSummary`` bumps the
    // value on each request, captures it locally, and the
    // ``work_summary_rated`` WS handler ignores any event whose
    // implicit "before" state predates the latest local in-flight
    // request. Prevents a delayed/stale WS event from clobbering the
    // newer POST result (codex r1 P2 — rapid rating-edit race).
    _summaryRateSeq: {},
    // Set of summary ids whose rating POST is currently in flight.
    // The template uses this to disable the rating buttons + show a
    // subtle loading affordance, preventing double-fires.
    _summaryRateInFlight: {},
    workplacePending: [],
    // Open credential / browser-login / captcha asks — the authoritative
    // feed (GET /api/help-requests) that drives those "Needs you" rows,
    // replacing the old volatile operator-chat scrape.
    needsYouRequests: [],
    workplaceLoading: false,
    // Per-section loading + error state. Failures used to be swallowed
    // silently (console.error) which left the user staring at an empty
    // panel with no way to retry. Each loadWorkplace* function now flips
    // its own bucket here so the template can render skeletons during
    // the in-flight window and a "Couldn't load — Retry" banner on
    // failure. Banner click handlers clear the error and re-run the
    // load, keeping the recovery path one click away.
    workplaceSectionLoading: {
      teams: false,
      tasks: false,
      blockers: false,
      pending: false,
      help: false,
      summaries: false,
      goals: false,
      pipelines: false,
    },
    workplaceErrors: {
      teams: '',
      tasks: '',
      blockers: '',
      pending: '',
      help: '',
      summaries: '',
      goals: '',
      pipelines: '',
    },
    // In-flight audit-log undo so we can disable the button per-row.
    auditReverting: {},

    // Pending-action collapse state. When two-or-more unresolved
    // ``pending_action_card`` messages live in the operator chat, the
    // template renders a single "N actions awaiting you" bar instead
    // of three full amber cards. Click / Enter / Space toggles
    // ``pendingExpanded``; ``pendingPulse`` flashes the count badge
    // for ~2s when a new pending arrives while collapsed.
    pendingExpanded: false,
    pendingPulse: false,
    _pendingPulseTimer: null,

    // Workplace task drill-in modal (PR 4) — populated lazily on
    // card click. ``drillInData`` carries ``{task, events, artifacts}``
    // from /api/workplace/tasks/{id}; the comment box is reset on
    // every open so prior text doesn't bleed across tasks.
    drillInTaskId: null,
    drillInData: null,
    drillInComment: '',
    drillInLoading: false,
    drillInSubmitting: false,
    drillInError: '',

    // System tab — sub-navigation
    systemTab: 'settings',
    // Order (and default landing): General first (its id is 'settings',
    // relabelled "General" via systemTabLabelFor), Operator second — these two
    // are the operator's home + primary control surface. The rest descend by
    // steady-state likelihood of use: money + observability (Costs, Activity),
    // then connectivity (Integrations, API Keys), then scheduling (Automation),
    // then niche/infrastructure plumbing (Browser, Wallet, Network, Storage).
    systemTabs: [
      { id: 'settings', label: 'Settings' },
      { id: 'operator', label: 'Operator' },
      { id: 'costs', label: 'Costs' },
      { id: 'activity', label: 'Activity' },
      { id: 'integrations', label: 'Connectors' },
      { id: 'skills', label: 'Skills' },
      { id: 'apikeys', label: 'API Keys' },
      { id: 'automation', label: 'Automation' },
      { id: 'browser', label: 'Browser' },
      { id: 'wallet', label: 'Wallet' },
      { id: 'network', label: 'Network' },
      { id: 'storage', label: 'Storage' },
    ],

    // Skills — per-agent assignment tab + fleet catalog (System → Skills).
    // agentSkills: rows for the selected agent (carry agent_assigned +
    // fleet_assigned); fleetSkillsCatalog: the admin/catalog view (no
    // agent_id) whose rows carry fleet_assigned. null = not yet loaded
    // (spinner), [] = loaded-but-empty (empty state).
    agentSkills: null,
    agentSkillsLoading: false,
    agentSkillsSaving: false,
    fleetSkillsCatalog: null,
    fleetSkillsCatalogLoading: false,
    fleetSkillsSaving: false,
    // MCP connectors — fleet catalog (System → Connectors). One record =
    // an MCP server definition + its agent assignment (['*'] = all).
    // connectorsData: null while loading; {connectors, pending_restart,
    // available_credentials, agents} once fetched. connectorDraft is the
    // open add/edit form (null = closed); _-prefixed draft fields are
    // UI-only and stripped from the wire payload.
    connectorsData: null,
    connectorsLoading: false,
    connectorSaving: false,
    connectorDraft: null,
    connectorErrors: {},
    connectorGlobalError: null,
    // Post-save restart prompt: {name, affected: [...]} — D7: catalog
    // edits never auto-restart; the operator confirms.
    connectorRestartPrompt: null,
    connectorRestarting: false,
    skillInstallRepo: '',
    skillInstallRef: '',
    skillInstalling: false,
    skillRemoving: '',

    // Audit sub-tab
    auditLog: [],
    auditTotal: 0,
    auditPage: 1,
    auditLoading: false,
    auditIncludeArchived: false,
    // "Archive entries older than" control: 7 / 30 / 90 days; default 30.
    auditArchiveDays: 30,
    auditArchiving: false,

    // Unified Team Hub (replaces separate TEAM.md + Comms + Broadcast panels).
    // Leads with the live Work view; raw State + Activity are demoted under
    // the Advanced sub-tab (teamHubAdvTab).
    teamHubExpanded: true,
    teamHubTab: 'work',  // 'work' | 'artifacts' | 'docs' | 'members' | 'broadcast' | 'advanced'
    teamHubAdvTab: 'state',  // within Advanced: 'state' | 'activity'
    _teamWorkLoaded: false,  // load the workplace ledger once, then WS keeps it live

    // TEAM.md banner on Agents tab (kept for backward compat, not used by template)
    teamBannerExpanded: false,

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

    // OAuth integrations (Google Drive/Gmail/Calendar via bring-your-own-app)
    integrations: [],                 // [{ key, label, configured, redirect_uri, scope_bundles, connections }]
    integrationSetup: {},             // { [providerKey]: { client_id, client_secret } } — setup form state
    integrationSetupSaving: '',       // providerKey currently saving its OAuth client
    integrationConnectName: {},       // { [providerKey]: connectionName } — defaults to provider key
    integrationSelectedScopes: {},    // { [providerKey]: { [bundleKey]: bool } } — checkbox state
    integrationDisconnecting: '',     // connection name currently being disconnected


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
    // Initial setup modal — model selections + completion flag.
    onboardOperatorModel: '',
    onboardDefaultModel: '',
    onboardFinishing: false,
    _setupDone: false,

    // Phase -1 — Empty-fleet onboarding wizard (hypothesis test).
    // States: 'idle' (off) | 'ask' (chip card) | 'confirming' (operator
    // proposed a team) | 'building' (apply_template running) |
    // 'first-output' (success card). Persisted to localStorage so a
    // page reload mid-wizard doesn't lose progress. Only renders inside
    // the Chat tab; non-empty fleets force step='idle'.
    wizard: { step: 'idle', plan: null, startedAt: 0, lastChip: '' },
    _wizardLastTrack: '',  // dedupe key for step_advanced events
    _wizardBuildPoll: null,
    _wizardCompleteTimer: null,

    // Track the element focused before the side panel opened so ESC
    // restores focus to the original trigger on close.
    _messengerSidePanelPrevFocus: null,

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

    // Phase 2 Board UX overhaul — activity translation toggle.
    // Persisted to localStorage so the user's choice survives reloads.
    // When false (default), implementation events (blackboard_write,
    // llm_call, message_received) are hidden from the Activity feed
    // and engineer event types are run through formatActivityForUser
    // for plain-English summaries.
    showTechDetail: false,

    // Phase 2 Board UX overhaul — notifications bell.
    notifications: [],
    notificationsUnreadCount: 0,
    notificationsOpen: false,
    notificationsLoading: false,
    _notificationsRefreshTimer: null,

    // Per-agent "restarting" pulse — populated by ``agent_restarting``
    // events and cleared by ``agent_restarted`` / ``agent_state``
    // ``restart_failed``. Bound by templates as
    // ``agentRestartingMap[agent_id]`` for the spinner indicator.
    agentRestartingMap: {},

    // Browser Notification API integration. ``browserNotifyEnabled`` is
    // the user's local opt-in (persisted to ``olBrowserNotifyEnabled``);
    // ``browserNotifyPermission`` mirrors ``Notification.permission``
    // for template gating. We only fire notifications when:
    //   1. Permission has been granted by the OS,
    //   2. The user explicitly toggled the in-app opt-in on,
    //   3. The dashboard tab is not visible (foreground tabs already
    //      have full UI signal).
    // Triple-redundant gate keeps us from spamming the user.
    browserNotifyEnabled: false,
    browserNotifyPermission: 'default',
    // Track the highest notification id we've seen so the 60s
    // fetchNotifications poll only fires browser notifications for
    // genuinely new arrivals.
    _lastNotifiedId: 0,
    _browserNotifyKinds: ['approval', 'credential', 'alert', 'blocker'],

    // WebSocket
    _ws: null,
    _refreshInterval: null,
    _fleetDebounce: null,
    _queueRefreshDebounce: null,
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
          ? `/teams/${this.detailAgent}`
          : `/teams/${this.detailAgent}/${tab}`;
      }
      if (this.activeTab === 'chat') return '/chat';
      if (this.activeTab === 'system') {
        if (this.systemTab === 'activity') {
          if (this.activityView === 'events') return '/system/activity/events';
          if (this.activityView === 'logs') return '/system/activity/logs';
          return '/system/activity';
        }
        return '/system/' + (this.systemTab || 'settings');
      }
      if (this.activeTab === 'fleet') return '/teams';
      // Single Work-tab URL — ``/home``. Legacy ``/home/kanban``,
      // ``/home/activity``, ``/home/summaries``, ``/home/tasks`` from
      // old bookmarks all normalize to ``/home`` in ``_parsePath`` so
      // they 200 instead of 404.
      if (this.activeTab === 'workplace') {
        return '/home';
      }
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
      if (this.activeTab === 'workplace') {
        return 'Work \u2014 OpenLegion';
      }
      return 'Teams \u2014 OpenLegion';
    },

    _parsePath(path) {
      const clean = path.replace(/^\/+/, '').replace(/\/+$/, '');
      const route = { tab: 'chat', activityView: 'traces', systemTab: 'settings', agentId: null, identityTab: 'config' };
      if (!clean) return route;

      if (clean === 'chat') { route.tab = 'chat'; return route; }
      // /teams is the canonical fleet URL; /agents is kept as a back-compat
      // alias so old bookmarks resolve. The first _pushUrl after load rewrites
      // the URL to /teams via replaceState.
      if (clean === 'teams' || clean.startsWith('teams/') || clean === 'agents' || clean.startsWith('agents/')) { route.tab = 'fleet'; }
      // Single Work-tab URL. Any /home or legacy
      // /home/{kanban,activity,summaries,tasks} sub-route normalizes
      // silently to ``/home`` so old bookmarks survive without a 404
      // or visible redirect.
      if (clean === 'home' || clean.startsWith('home/')) {
        route.tab = 'workplace';
        return route;
      }

      const agentMatch = clean.match(/^(?:teams|agents)\/([^/]+)(?:\/([^/]+))?$/);
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
        if (resolved && ['activity', 'costs', 'automation', 'skills', 'integrations', 'apikeys', 'wallet', 'network', 'storage', 'operator', 'browser', 'settings'].includes(resolved)) {
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
                this.fetchChannels(); this.fetchWebhooks(); this.fetchApiKeys(); this.loadIntegrations();
                this.loadConnectors();
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
              if (route.systemTab === 'skills') {
                this.loadSkillsCatalog();
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
      if (this._detailReturnTeam !== null && this._detailReturnTeam !== undefined) {
        this.activeTeam = this._detailReturnTeam;
      }
      this._detailReturnTeam = null;
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

    // Non-skippable initial-setup modal. Shows for genuinely-new
    // installs until the user adds LLM access (a key OR OpenLegion
    // credits) AND picks their Operator + Default agent models. Hidden
    // for established installs (already have credentials + a real
    // teammate) and once setup has been finished.
    get showSetup() {
      if (this.loading || !this.settingsData) return false;
      if (this._setupDone) return false;
      try { if (localStorage.getItem('ol_setup_done') === '1') return false; } catch (_) { /* ignore */ }
      // Established install — don't interrupt returning users.
      if (this.settingsData.has_llm_credentials && this.agents.some(a => a.id !== 'operator')) return false;
      return true;
    },

    // Finish button is enabled only once LLM access is configured and
    // both model choices are made.
    get setupCanFinish() {
      return !!(this.settingsData && this.settingsData.has_llm_credentials
        && this.onboardOperatorModel && this.onboardDefaultModel);
    },

    get addAgentNameValid() {
      const name = this.addAgentForm.name;
      if (!name) return true; // empty is valid (not yet typed)
      return /^[a-z][a-z0-9_]{0,29}$/.test(name);
    },

    get maxAgents() {
      return this.settingsData?.plan_limits?.max_agents ?? 0;
    },
    get maxTeams() {
      return this.settingsData?.plan_limits?.max_projects ?? 0;
    },
    get runningAgents() {
      return this.agents.filter(a => a.running !== false && a.id !== 'operator');
    },
    get atAgentLimit() {
      if (this.maxAgents === 0) return false;
      return this.runningAgents.length >= this.maxAgents;
    },
    get teamsEnabled() {
      const limits = this.settingsData?.plan_limits;
      if (!limits) return true; // no limits loaded yet, allow everything
      return limits.projects_enabled !== false;
    },
    get atTeamLimit() {
      if (!this.teamsEnabled) return true;
      if (this.maxTeams === 0) return false; // unlimited
      const teamCount = this.teams?.length ?? 0;
      return teamCount >= this.maxTeams;
    },

    get filteredEvents() {
      // Phase 2 Board UX overhaul — when "Show technical detail" is
      // off (default), hide implementation-noise events
      // (blackboard_write, llm_call, message_received, message_sent,
      // text_delta, agent_state) so the activity feed reads as
      // plain-English status updates. The eventFilters array still
      // takes precedence when the user has explicitly chosen filters
      // — power users opting in to specific types should always see
      // them.
      let base;
      if (this.eventFilters.length === 0) {
        base = this.events.filter(e =>
          !(e.type === 'agent_state' && e.data?.state === 'registered'));
      } else {
        base = this.events.filter(e => this.eventFilters.includes(e.type));
      }
      if (this.showTechDetail || this.eventFilters.length > 0) return base;
      return base.filter(e => this.isActivityEventVisible(e));
    },

    get fleetTotalCost() {
      return this.agents.reduce((sum, a) => sum + (a.daily_cost || 0), 0);
    },

    get fleetTotalTokens() {
      return this.agents.reduce((sum, a) => sum + (a.daily_tokens || 0), 0);
    },

    get filteredAgents() {
      // User-fleet view: excludes the operator system agent. Drives all stats (cost, tokens, health),
      // count-against-quota displays, and broadcast targeting. Operator is rendered separately via displayAgents.
      if (this.activeTeam) {
        return this.agents.filter(a => a.id !== 'operator' && a.project === this.activeTeam);
      }
      const base = this.teams.length > 0 ? this.soloAgents : this.agents;
      return base.filter(a => a.id !== 'operator');
    },

    /** Agent grid render list. Standalone view prepends the operator card (system agent, links to operator settings). */
    get displayAgents() {
      if (this.activeTeam) return this.filteredAgents;
      const operator = this.agents.find(a => a.id === 'operator');
      return operator ? [operator, ...this.filteredAgents] : this.filteredAgents;
    },

    get filteredFleetCost() {
      return this.filteredAgents.reduce((sum, a) => sum + (a.daily_cost || 0), 0);
    },

    get filteredFleetTokens() {
      return this.filteredAgents.reduce((sum, a) => sum + (a.daily_tokens || 0), 0);
    },

    get fleetHealthCounts() {
      const counts = { healthy: 0, unhealthy: 0, quarantined: 0, failed: 0, unknown: 0, stopped: 0 };
      for (const a of this.filteredAgents) {
        const s = a.health_status || 'unknown';
        if (s === 'stopped') counts.stopped++;
        else if (s === 'healthy') counts.healthy++;
        else if (s === 'quarantined') counts.quarantined++;
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

      // Surface the OAuth-integration callback result (?integration_connected /
      // ?integration_error) as a toast, then scrub the query param.
      this._handleIntegrationRedirect();

      // NOTE: the side panel is no longer restored from the legacy
      // ``ol_messenger_side_panel_open`` flag. Its only writer was the
      // messenger toggle button, now removed from both navbars, so a stale
      // key from an older session would otherwise force the panel open with
      // no obvious way to dismiss it. The panel now restores purely from the
      // persisted open-chats set.

      // Restore Teams chip-strip expanded preference.
      try {
        if (localStorage.getItem('ol_team_filter_expanded') === '1') {
          this.teamFilterExpanded = true;
        }
      } catch (e) { /* ignore */ }
      // Persist whenever the user toggles the chip strip.
      this.$watch('teamFilterExpanded', (val) => {
        try {
          if (val) localStorage.setItem('ol_team_filter_expanded', '1');
          else localStorage.removeItem('ol_team_filter_expanded');
        } catch (e) { /* ignore */ }
      });

      // Restore the currently-active team. Reads the new ``activeTeam`` key,
      // falling back to the legacy ``activeProject`` key one time so users
      // upgrading from PR 2 don't lose their selection. The legacy key is
      // cleared after migration so subsequent reads stay clean.
      try {
        let active = localStorage.getItem('activeTeam');
        if (!active) {
          const legacy = localStorage.getItem('activeProject');
          if (legacy) {
            active = legacy;
            localStorage.setItem('activeTeam', legacy);
          }
        }
        if (localStorage.getItem('activeProject') !== null) {
          localStorage.removeItem('activeProject');
        }
        if (active) this.activeTeam = active;
      } catch (_) { /* ignore */ }

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
            // Disconnect invalidates ephemeral chat-stream state — a
            // ``text_delta`` flips ``chatStreamingAgents[id]=true`` and only
            // ``chat_done`` clears it. If the WS dropped between those two,
            // the flag is stuck and ``_loadChatHistory`` early-returns,
            // silently swallowing the refetch.
            //
            // BUT: ``_chatAborts[id]`` is an AbortController for a LOCAL
            // SSE ``/chat/stream`` fetch — wiping it kills our handle on
            // an active local stream and the next text_delta could land
            // in a stale render target. Skip agents with an active local
            // stream (their state is correct as-is) and reset state for
            // everyone else so a stuck flag elsewhere doesn't block a
            // later reopen.
            const agentsToReset = Object.keys({
              ...this.chatStreamingAgents,
              ...this._chatStreamEndAt,
              ...this._chatFetchedAt,
            });
            for (const id of agentsToReset) {
              if (this._chatAborts && this._chatAborts[id]) continue;
              delete this.chatStreamingAgents[id];
              delete this._chatStreamEndAt[id];
              delete this._chatFetchedAt[id];
            }
            for (const agentId of this.openChats) {
              // Skip the refetch for agents with an active local SSE
              // stream — the persistent transcript won't include the
              // in-flight bubble yet, and replacing chatHistories[agent]
              // mid-stream invalidates the streaming render target.
              if (this._chatAborts && this._chatAborts[agentId]) continue;
              this._loadChatHistory(agentId);
            }
            // Queue state is WS-driven (no poll) — re-seed it on reconnect so
            // any transitions missed during the outage are reflected.
            this.fetchQueues();
            // The "Needs you" help-request feed is likewise refreshed so a
            // request that arrived or resolved during the outage is reflected
            // — otherwise the panel could under- or over-report after a blip.
            if (typeof this.loadWorkplaceHelpRequests === 'function') {
              this.loadWorkplaceHelpRequests();
            }
            // Platform success is browser_metrics-driven (no poll); if the
            // operator is on the Browser panel, re-seed it too so a blip
            // doesn't leave the rollup frozen until the next metrics event.
            if (this.activeTab === 'system' && this.systemTab === 'browser') {
              this.fetchPlatformSuccess();
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
      this.fetchTeamContent();
      this.fetchTeams();
      this.fetchModelHealth();
      this.fetchQueues();
      // Seed the "Needs you" help-request feed at startup so the navbar badge
      // is correct on EVERY tab (not just after the Work tab mounts). Live
      // WS events keep it fresh; this slow poll is a completeness backstop in
      // case an event is dropped (the panel must never silently under-report).
      this.loadWorkplaceHelpRequests();
      this._helpRequestsInterval = setInterval(() => this.loadWorkplaceHelpRequests(), 60000);
      this._refreshInterval = setInterval(() => this.fetchAgents(), 15000);
      this._modelHealthInterval = setInterval(() => this.fetchModelHealth(), 60000);
      // Queue depth/busy is now pushed live via the ``queue_changed`` WS event
      // (see onWsEvent); the one-shot fetchQueues() above seeds initial state.

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

      // Phase -1 wizard — restore step state across reloads. If the user
      // was mid-build when they hit refresh, we want the building card
      // back, not the chip selector. After restore we still let
      // ``_maybeStartWizard`` evaluate first-visit detection in case
      // ``ol_wizard`` is missing entirely.
      this._wizardLoad();
      // Resume polling if we restored into ``building`` — the build
      // poller is interval-based and was destroyed on unload.
      if (this.wizard.step === 'building') {
        this._wizardStartBuildPolling();
      }
      // Emit abandon telemetry when the page unloads mid-wizard. This
      // is best-effort — sendBeacon survives navigation; setInterval
      // ticks may not. We only fire if we're not already in 'idle'.
      window.addEventListener('beforeunload', () => {
        if (this.wizard && this.wizard.step !== 'idle' && this.wizard.step !== 'first-output') {
          try {
            const payload = JSON.stringify({
              event_name: 'wizard_abandoned',
              props: {
                lastStep: this.wizard.step,
                totalSeconds: this._wizardSecondsSinceStart(),
              },
              ts: Date.now() / 1000,
            });
            const blob = new Blob([payload], { type: 'application/json' });
            navigator.sendBeacon(`${window.__config.apiBase}/telemetry`, blob);
          } catch (_) { /* best-effort */ }
        }
      });

      // Sync restored open chats from server so cross-device history is fresh
      for (const agentId of this.openChats) {
        this._loadChatHistory(agentId);
      }

      // Phase 2 Board UX — restore "Show technical detail" preference.
      try {
        const saved = localStorage.getItem('olShowTechDetail');
        if (saved === '1') this.showTechDetail = true;
      } catch (_) {
        // ignore — localStorage unavailable.
      }

      // Browser Notification API — restore the user's local opt-in and
      // sync ``Notification.permission`` so the wizard button renders
      // the right state on reload. The opt-in toggle is independent of
      // the OS-level permission; we require both to fire.
      try {
        const saved = localStorage.getItem('olBrowserNotifyEnabled');
        if (saved === 'true') this.browserNotifyEnabled = true;
      } catch (_) {
        // ignore — localStorage unavailable.
      }
      if (typeof Notification !== 'undefined' && Notification && Notification.permission) {
        this.browserNotifyPermission = Notification.permission;
      }

      // Phase 2 Board UX — initial notifications fetch + 60s poll.
      // Fetched lazily on bell open as well; the poll keeps the
      // unread badge fresh without forcing the dropdown to open.
      this.fetchNotifications();
      this._notificationsRefreshTimer = setInterval(
        () => this.fetchNotifications(),
        60_000,
      );

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
        // ESC also dismisses the messenger side panel (a11y). The panel is
        // visible whenever a chat is open (clicking an agent drives
        // ``openChats``) OR the legacy ``messengerSidePanelOpen`` flag is
        // set — gate on both so ESC closes the agent-opened panel too, not
        // just the (now removed) toggle-opened one. Dismiss by minimizing
        // (keeps the chats; reopen by clicking an agent); ``closeSidePanel``
        // also clears any legacy flag + restores focus.
        // Order matters: the cmd palette ESC handler runs first above and
        // returns; we only get here if the palette wasn't open.
        if (e.key === 'Escape' && this.activeTab !== 'chat' && !this.chatPanelMinimized
            && (this.openChats.length > 0 || this.messengerSidePanelOpen)) {
          this.chatPanelMinimized = true;
          this.closeSidePanel();
        }
      };
      document.addEventListener('keydown', this._cmdPaletteHandler);

      // Normalize legacy /dashboard/ URL to root
      if (/^\/dashboard\/?$/.test(window.location.pathname)) {
        history.replaceState(null, '', '/');
      }

      // Deep link restoration: parse initial URL and apply route
      const initRoute = this._parsePath(window.location.pathname);
      const isDeepLink = initRoute.agentId || initRoute.tab !== 'chat' || initRoute.activityView !== 'traces' || initRoute.systemTab !== 'settings';
      if (isDeepLink) {
        this.$nextTick(() => {
          this._applyRoute(initRoute);
          document.title = this._buildTitle();
          history.replaceState(null, '', this._buildPath());
        });
      } else {
        document.title = this._buildTitle();
      }

      // Operator chat container is x-show="operatorReady" — until fetchAgents
      // resolves it's display:none and any scroll fired against it no-ops.
      // Re-fire once when the gate flips so users land on the latest message
      // rather than the top of the restored history.
      this.$watch('operatorReady', (val, oldVal) => {
        if (!val || oldVal) return;
        this.$nextTick(() => {
          if (this.activeTab === 'chat' || this.messengerSidePanelOpen) {
            this._scrollChat(this.activeChatId || 'operator', true);
          }
        });
      });

      // Side-panel open transition — the chat-messages container mounts
      // at scrollTop=0; force a scroll on the open edge so click-toggles
      // land on the latest message. localStorage-restored opens are
      // handled below since the assignment already happened during state
      // restore (the watcher only fires on subsequent changes).
      this.$watch('messengerSidePanelOpen', (val) => {
        if (!val) return;
        this.$nextTick(() => {
          if (this.activeChatId) this._scrollChat(this.activeChatId, true);
        });
      });

      // Ensure operator chat is initialized when landing on the chat tab
      if (this.activeTab === 'chat' || initRoute.tab === 'chat') {
        if (!this.openChats.includes('operator')) {
          this.openChats.push('operator');
        }
        this.activeChatId = 'operator';
        this._loadChatHistory('operator');
        this.$nextTick(() => {
          this._scrollChat('operator', true);
          const el = document.getElementById('operator-chat-input');
          if (el) el.focus();
        });
      }

      // Side panel restored from localStorage above (line ~1233) — the
      // $watch above doesn't fire on init-time assignment, so scroll
      // explicitly here.
      if (this.messengerSidePanelOpen && this.activeChatId) {
        this.$nextTick(() => {
          this._scrollChat(this.activeChatId, true);
        });
      }

      // Popstate listener for browser back/forward
      this._popstateHandler = () => {
        // Guard unsaved edits
        if (this.identityEditing || this.configEditing || this.teamEditing) {
          if (!confirm('You have unsaved changes. Discard and navigate away?')) {
            // Re-push current URL to cancel the back navigation
            this._pushUrl(true);
            return;
          }
          this.identityEditing = false;
          this.identityEditBuffer = '';
          this.configEditing = false;
          this.editForm = {};
          this.teamEditing = false;
          this.teamEditBuffer = '';
        }
        const route = this._parsePath(window.location.pathname);
        this._applyRoute(route);
        document.title = this._buildTitle();
      };
      window.addEventListener('popstate', this._popstateHandler);

      // ── iOS Safari keyboard-scroll recovery ──────────────────
      // The whole app scrolls through inner containers (`main.overflow-auto`,
      // the chat logs) while the document itself is deliberately locked:
      // `<body class="h-[100dvh] overflow-hidden">`. That's the right model on
      // desktop, but it has a nasty failure on iOS Safari.
      //
      // When the on-screen keyboard opens for a text field — in practice the
      // chat composer, the only persistently-focused input — Safari tries to
      // scroll the focused caret above the keyboard. The composer lives in a
      // `position: fixed` panel (or the fixed-height chat tab), so Safari can't
      // satisfy the caret-reveal by scrolling an inner container and instead
      // scrolls the DOCUMENT (scrollingElement). On blur it frequently fails to
      // restore that offset, leaving the entire UI shifted up with a dead strip
      // at the bottom — which reads exactly as "I can't scroll the page anymore"
      // until a reload or rotate. Intermittent, mobile-only, always right after
      // using the chat — matches the reported symptom.
      //
      // Because the body is intentionally non-scrollable, ANY document-level
      // scroll offset is spurious: snap it back to 0. We only do this once the
      // field is blurred (keyboard dismissed) — while a field is focused we
      // leave the offset alone so we don't fight Safari positioning the caret.
      this._resetDocScroll = () => {
        const ae = document.activeElement;
        // Still typing → keyboard is up → let Safari keep the caret in view.
        if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA' || ae.isContentEditable)) {
          return;
        }
        const se = document.scrollingElement || document.documentElement;
        if (se && se.scrollTop !== 0) se.scrollTop = 0;
        if (window.pageYOffset !== 0) window.scrollTo(0, 0);
      };
      // focusout bubbles (blur doesn't), so one listener covers every input in
      // the app. Defer past the keyboard-dismiss reflow so the reset sticks.
      this._focusOutResetHandler = () => { setTimeout(this._resetDocScroll, 50); };
      document.addEventListener('focusout', this._focusOutResetHandler);
      // The keyboard close often only settles on the visualViewport resize that
      // follows it; catch that too (guarded by the activeElement check above so
      // it no-ops mid-typing while the keyboard is animating in).
      //
      // iOS-only: this whole recovery exists for the iOS Safari/WebKit
      // caret-reveal-scrolls-the-document bug. On Chrome for Android the
      // document is never scrolled (body is `overflow-hidden`), so the reset
      // is a no-op there — but `visualViewport.resize` ALSO fires on Android's
      // URL-bar show/hide during ordinary scrolling, so attaching it there is
      // pure churn on the scroll path. Scope the listener to iOS to keep it.
      // (iPadOS 13+ reports as MacIntel, hence the touch-points fallback.)
      const isIOS = /iP(hone|ad|od)/.test(navigator.userAgent) ||
        (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
      if (isIOS && window.visualViewport) {
        this._vvResetHandler = () => this._resetDocScroll();
        window.visualViewport.addEventListener('resize', this._vvResetHandler);
      }

      // Pause polling when tab is hidden to save resources
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
          if (this._refreshInterval) { clearInterval(this._refreshInterval); this._refreshInterval = null; }
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
          // Resume model-health polling; queue state is WS-driven now, so
          // just re-seed it once on tab-return.
          this.fetchModelHealth();
          this._modelHealthInterval = setInterval(() => this.fetchModelHealth(), 60000);
          this.fetchQueues();
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

      // ── Tooltip portal for .setting-hint elements ──
      // .agent-card has `backdrop-filter` and a hover `transform`; .nav-bar
      // has `backdrop-filter`. All three properties create a new containing
      // block for position:fixed descendants AND a new stacking context, so
      // a tooltip rendered inside one of these ancestors gets trapped:
      // it's clipped by the card's `overflow: hidden`, and its z-index can't
      // escape the ancestor's stacking layer. We sidestep this with a single
      // shared portal node parked on <body>, populated on hover.
      const _hintPortal = (() => {
        let el = null;
        return () => {
          if (!el || !document.body.contains(el)) {
            el = document.createElement('span');
            el.className = 'setting-hint-text';
            el.setAttribute('role', 'tooltip');
            el.style.display = 'none';
            document.body.appendChild(el);
          }
          return el;
        };
      })();
      const _positionHint = (hint) => {
        const source = hint.querySelector('.setting-hint-text');
        if (!source) return;
        const portal = _hintPortal();
        // Current tooltips are plain text. If HTML is ever needed,
        // swap textContent for cloned childNodes.
        portal.textContent = source.textContent;
        portal.classList.remove('hint-below');
        const r = hint.getBoundingClientRect();
        const gap = 6;
        // Measure off-screen
        portal.style.visibility = 'hidden';
        portal.style.display = 'block';
        portal.style.top = '-9999px';
        portal.style.left = '-9999px';
        const tw = portal.offsetWidth;
        const th = portal.offsetHeight;
        // Default: above the icon, left-aligned
        let top = r.top - gap - th;
        let left = r.left;
        let below = false;
        if (top < 8) {
          top = r.bottom + gap;
          below = true;
        }
        if (left + tw > window.innerWidth - 8) {
          left = window.innerWidth - tw - 8;
        }
        if (left < 8) left = 8;
        const arrowX = Math.min(Math.max(r.left + r.width / 2 - left, 8), tw - 8);
        portal.style.setProperty('--arrow-x', arrowX + 'px');
        portal.style.top = top + 'px';
        portal.style.left = left + 'px';
        portal.style.visibility = '';
        portal.classList.toggle('hint-below', below);
      };
      const _hideHint = () => {
        const portal = _hintPortal();
        portal.style.display = 'none';
        portal.classList.remove('hint-below');
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
      if (this._queueRefreshDebounce) clearTimeout(this._queueRefreshDebounce);
      if (this._platformSuccessDebounce) clearTimeout(this._platformSuccessDebounce);
      if (this._cronDebounce) clearTimeout(this._cronDebounce);
      if (this._modelHealthInterval) clearInterval(this._modelHealthInterval);
      if (this._helpRequestsInterval) clearInterval(this._helpRequestsInterval);
      if (this._helpRefreshTimer) clearTimeout(this._helpRefreshTimer);
      if (this._cookieRenewalInterval) clearInterval(this._cookieRenewalInterval);
      if (this._visibilityHandler) document.removeEventListener('visibilitychange', this._visibilityHandler);
      if (this._costDebounce) clearTimeout(this._costDebounce);
      if (this._pendingPulseTimer) {
        clearTimeout(this._pendingPulseTimer);
        this._pendingPulseTimer = null;
      }
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
      if (this._cmdPaletteHandler) document.removeEventListener('keydown', this._cmdPaletteHandler);
      if (this._popstateHandler) window.removeEventListener('popstate', this._popstateHandler);
      if (this._focusOutResetHandler) document.removeEventListener('focusout', this._focusOutResetHandler);
      if (this._vvResetHandler && window.visualViewport) {
        window.visualViewport.removeEventListener('resize', this._vvResetHandler);
      }
    },

    // ── Phase 1 — Board UX overhaul helpers ──────────────────

    /**
     * Display label for a primary nav tab. Renames Agents/Board/System
     * to Teams/Work/Settings without touching the tab IDs (URLs, routes,
     * and JS state vars all stay on the legacy IDs). The Teams sidebar
     * + mobile aria-label render an explicit ``Teams (N)`` with the team
     * count alongside; this map is the fallback when callers don't
     * specialize the fleet case.
     */
    tabLabelFor(tab) {
      const map = { fleet: 'Teams', workplace: 'Work', system: 'Settings' };
      return map[tab.id] || tab.label;
    },

    /**
     * Return the System sub-tab label. ``settings`` is renamed to
     * ``General`` so the parent "Settings" doesn't collide with a
     * sub-tab of the same name.
     */
    systemTabLabelFor(tab) {
      if (tab.id === 'settings') return 'General';
      return tab.label;
    },

    /** Toggle the side-panel messenger (visible on non-Chat tabs). */
    toggleSidePanel() {
      this.messengerSidePanelOpen = !this.messengerSidePanelOpen;
      try {
        if (this.messengerSidePanelOpen) {
          // Capture the previously focused element so ESC can restore
          // focus on close.
          try { this._messengerSidePanelPrevFocus = document.activeElement; } catch (_) {}
          localStorage.setItem('ol_messenger_side_panel_open', '1');
          // Open Operator by default if no chat is active.
          if (!this.openChats.includes('operator')) this.openChats.push('operator');
          if (!this.activeChatId) this.activeChatId = 'operator';
          this.chatPanelMinimized = false;
          this._loadChatHistory(this.activeChatId || 'operator');
          this.$nextTick(() => {
            // Without this, the side-panel mounts at scrollTop=0 even
            // though restored history is already in the container.
            this._scrollChat(this.activeChatId || 'operator', true);
            const el = document.getElementById('operator-chat-input');
            if (el) el.focus();
          });
        } else {
          localStorage.removeItem('ol_messenger_side_panel_open');
          // Restore focus when toggled closed via the same button.
          try {
            if (this._messengerSidePanelPrevFocus && this._messengerSidePanelPrevFocus.focus) {
              this._messengerSidePanelPrevFocus.focus();
            }
          } catch (_) {}
          this._messengerSidePanelPrevFocus = null;
        }
      } catch (e) { /* ignore */ }
    },

    /** Close the side panel (ESC handler). */
    closeSidePanel() {
      if (!this.messengerSidePanelOpen) return;
      this.messengerSidePanelOpen = false;
      try { localStorage.removeItem('ol_messenger_side_panel_open'); } catch (e) { /* ignore */ }
      // Restore focus to the trigger element captured on open.
      try {
        if (this._messengerSidePanelPrevFocus && this._messengerSidePanelPrevFocus.focus) {
          this._messengerSidePanelPrevFocus.focus();
        }
      } catch (_) {}
      this._messengerSidePanelPrevFocus = null;
    },

    /**
     * Phase 1 conversation pagination — render last N messages by default
     * with "Load older →" appending more. Returns the visible slice.
     */
    visibleChatHistory(agentId) {
      const all = this.chatHistories[agentId] || [];
      const limit = this._chatVisibleLimit[agentId] || 50;
      if (all.length <= limit) return all;
      return all.slice(all.length - limit);
    },

    /** Whether "Load older →" should render for the active chat. */
    hasOlderMessages(agentId) {
      const all = this.chatHistories[agentId] || [];
      const limit = this._chatVisibleLimit[agentId] || 50;
      return all.length > limit;
    },

    /** Append 50 more messages to the visible window for the given chat. */
    loadOlderMessages(agentId) {
      const cur = this._chatVisibleLimit[agentId] || 50;
      this._chatVisibleLimit = { ...this._chatVisibleLimit, [agentId]: cur + 50 };
    },

    /**
     * Caption for the "Load older" button — shows the visible-vs-total
     * count when computable so users know how deep the history is.
     * Format: "Load 50 older (340 of 500)" when total > limit; falls
     * back to "Load 50 older" when totals aren't yet known.
     */
    loadOlderCaption(agentId) {
      const all = this.chatHistories[agentId] || [];
      const limit = this._chatVisibleLimit[agentId] || 50;
      const total = all.length;
      const visible = Math.min(limit, total);
      if (total > limit) return `Load 50 older (${visible} of ${total})`;
      return 'Load 50 older';
    },

    /** Mark a worker conversation opened on the server (Phase 1 contract). */
    async openConversation(agentId) {
      if (agentId === 'operator') return;
      try {
        await fetch(`${window.__config.apiBase}/conversations/${encodeURIComponent(agentId)}/open`, {
          method: 'POST',
          headers: { 'X-Requested-With': 'XMLHttpRequest', 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: '{}',
        });
      } catch (e) { /* best-effort; client-side ``openChats`` is the source of truth */ }
    },

    /** Mark a worker conversation closed on the server. */
    async closeConversation(agentId) {
      if (agentId === 'operator') return;
      try {
        await fetch(`${window.__config.apiBase}/conversations/${encodeURIComponent(agentId)}/close`, {
          method: 'POST',
          headers: { 'X-Requested-With': 'XMLHttpRequest', 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: '{}',
        });
      } catch (e) { /* best-effort */ }
    },

    // ── Operator readiness ───────────────────────────────

    checkOperatorReady() {
      const op = this.agents.find(a => a.id === 'operator');
      const wasReady = this.operatorReady;
      // The operator is a fleet-global agent that the health monitor does NOT
      // track (it's excluded from the fleet status), so /api/agents reports its
      // health_status as the default 'unknown' — it never becomes 'healthy'.
      // Requiring === 'healthy' left the main chat stuck on "starting up..."
      // forever even though the operator was up and serving (the side panel,
      // which doesn't gate, worked). Ready = present and not genuinely down;
      // only the real not-ready states keep the "starting up" message.
      this.operatorReady = !!op
        && !['stopped', 'failed', 'restarting'].includes(op.health_status);
      // Fix #4 — when operator transitions unhealthy → healthy and the
      // user is still in a first-visit state, retry starting the
      // wizard. Without this, an operator that boots slowly (e.g.
      // first start) would have skipped wizard kickoff at fetchAgents.
      if (!wasReady && this.operatorReady) {
        try { this._maybeStartWizard(); } catch (_) { /* ignore */ }
      }
    },

    // ── Phase -1 onboarding wizard ───────────────────────
    //
    // Card-styled empty-fleet flow inside the Chat tab. Renders only when
    // ``wizard.step !== 'idle'``; the existing operator empty-state is
    // suppressed while active. State persists to localStorage so a page
    // reload mid-wizard restores progress. All transitions emit
    // telemetry events (``wizard_started`` / ``wizard_chip_clicked`` /
    // ``wizard_step_advanced`` / ``wizard_completed`` /
    // ``wizard_abandoned``) so we can answer the activation hypothesis.

    _wizardLoad() {
      // Backwards-compatible — unknown step values from older versions
      // (or hand-edited localStorage) reset to ``idle`` rather than
      // wedging the UI in a state with no rendering branch.
      const KNOWN_STEPS = new Set([
        'idle', 'ask', 'confirming', 'building', 'first-output', 'build_failed',
      ]);
      try {
        const raw = localStorage.getItem('ol_wizard');
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object' && typeof parsed.step === 'string') {
          const step = KNOWN_STEPS.has(parsed.step) ? parsed.step : 'idle';
          this.wizard = {
            step: step,
            plan: parsed.plan || null,
            startedAt: Number(parsed.startedAt) || 0,
            lastChip: parsed.lastChip || '',
          };
        }
      } catch (_) { /* ignore */ }
    },

    _wizardSave() {
      try {
        localStorage.setItem('ol_wizard', JSON.stringify(this.wizard));
      } catch (_) { /* quota / private mode — ignore */ }
    },

    _wizardClear() {
      try { localStorage.removeItem('ol_wizard'); } catch (_) { /* ignore */ }
    },

    _isFirstVisit() {
      // Empty-fleet detection: no user-fleet agents AND the operator has
      // no real user messages yet. The bootstrap_greeting we seed in the
      // operator's transcript is an assistant role — it doesn't count.
      const hasFleetAgents = this.agents.some(a => a && a.id !== 'operator');
      if (hasFleetAgents) return false;
      const hist = this.chatHistories['operator'] || [];
      const hasUserMsg = hist.some(m => m && m.role === 'user');
      return !hasUserMsg;
    },

    _operatorHasReplyAfter(ts) {
      const hist = this.chatHistories['operator'] || [];
      for (let i = hist.length - 1; i >= 0; i--) {
        const m = hist[i];
        if (!m) continue;
        if ((m.role === 'agent' || m.role === 'assistant') && (m.ts || 0) >= ts) {
          // We treat any non-streaming assistant reply after the chip
          // send as the trigger to advance to ``confirming``. We don't
          // attempt to parse the reply for a structured team — that's
          // for a future phase. Today the operator's natural reply IS
          // the proposed team, and we trust the user to read it.
          if (!m.streaming) return true;
        }
      }
      return false;
    },

    track(eventName, props) {
      try {
        fetch(`${window.__config.apiBase}/telemetry`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
          },
          body: JSON.stringify({
            event_name: eventName,
            props: props || {},
            ts: Date.now() / 1000,
          }),
          credentials: 'same-origin',
        }).catch(() => { /* fire-and-forget */ });
      } catch (_) { /* ignore */ }
    },

    // ── Phase 0 baseline telemetry helpers ──────────────────
    //
    // These instrument the existing dashboard before Phase 1's
    // structural rewrite ships, so we have a "before" benchmark to
    // compare against. All helpers funnel through ``track()`` and are
    // safe to call repeatedly — internal flags guard against duplicate
    // emissions for the once-per-load events.

    _trackFirstAction(actionType) {
      // Fire ``time_to_first_action`` exactly once per page load. Any
      // user interaction (tab switch, sub-tab click, Needs-You item,
      // empty-state CTA, dock toggle) counts; navigation that happens
      // automatically during init() does not — those paths bypass this
      // helper and only run after the user has done something.
      if (this._firstActionTracked) return;
      this._firstActionTracked = true;
      const since = this._initTs ? Math.max(0, (Date.now() - this._initTs) / 1000) : 0;
      this.track('time_to_first_action', {
        action_type: actionType || 'unknown',
        seconds_since_load: Math.round(since * 100) / 100,
      });
    },

    _handleNeedsYouAction(item, action) {
      // Wraps Needs-You item-button clicks so we can count engagement.
      // The handler may be a no-op for items without one (defensive),
      // and we still record the click so empty-handler bugs surface in
      // telemetry rather than silently dropping signal.
      try {
        const itemCount = (this.needsYouItems || []).length;
        this.track('needs_you_click', {
          item_count: itemCount,
          item_kind: item && item.kind ? item.kind : 'unknown',
          action_label: action && action.label ? action.label : '',
        });
        this._trackFirstAction('needs_you_click');
      } catch (_) { /* ignore — telemetry must not block the action */ }
      if (action && typeof action.handler === 'function') {
        action.handler();
      }
    },

    trackEmptyStateCta(sectionId) {
      // Phase-0 baseline telemetry hook. Currently unwired — the operator
      // empty-state intent chips that called it were removed when
      // onboarding collapsed into the single setup modal — but retained
      // (like trackSubtabUsage) so a future empty-state CTA can re-wire it
      // without re-introducing the helper. See TestPhase0FrontendWiring.
      this.track('empty_state_cta_click', { section_id: sectionId || 'unknown' });
      this._trackFirstAction('empty_state_cta_click');
    },

    trackSubtabUsage(fromSubtab, toSubtab) {
      // Phase 0 tracks Board sub-tab churn so Phase 3 (single-scroll
      // Home) can be evaluated against actual sub-tab usage instead of
      // assumed habits.
      if (fromSubtab === toSubtab) return;
      this.track('subtab_usage', {
        from_subtab: fromSubtab || '',
        to_subtab: toSubtab || '',
      });
      this._trackFirstAction('subtab_usage');
    },

    dockOpen(opts) {
      // Phase 1 will mount a side-panel messenger that calls this.
      // Today the helper is wireable but no UI invokes it — we keep
      // the call-site contract (``{ conversation_id? }``) so Phase 1
      // doesn't have to refactor it. Idempotent via ``_dockOpen``.
      if (this._dockOpen) return;
      this._dockOpen = true;
      const props = { from_tab_id: this.activeTab || '' };
      if (opts && opts.conversation_id) props.conversation_id = opts.conversation_id;
      this.track('dock_open', props);
      this._trackFirstAction('dock_open');
    },

    dockClose(opts) {
      // Counterpart to ``dockOpen``. Idempotent — calling close on an
      // already-closed dock is a no-op so accidental double-fires
      // (e.g. ESC + click-outside both firing) don't double-count.
      if (!this._dockOpen) return;
      this._dockOpen = false;
      const props = { from_tab_id: this.activeTab || '' };
      if (opts && opts.conversation_id) props.conversation_id = opts.conversation_id;
      this.track('dock_close', props);
    },

    _wizardSecondsSinceStart() {
      if (!this.wizard.startedAt) return 0;
      return Math.max(0, Math.round((Date.now() - this.wizard.startedAt) / 1000));
    },

    _wizardAdvance(toStep, extraProps) {
      const fromStep = this.wizard.step;
      if (fromStep === toStep) return;
      this.wizard.step = toStep;
      this._wizardSave();
      const props = Object.assign({
        from: fromStep,
        to: toStep,
        secondsSinceStart: this._wizardSecondsSinceStart(),
      }, extraProps || {});
      const dedupe = `${fromStep}->${toStep}`;
      if (this._wizardLastTrack !== dedupe) {
        this._wizardLastTrack = dedupe;
        this.track('wizard_step_advanced', props);
      }
    },

    startWizard() {
      // Idempotent — only kick off once per visit. If the user already
      // dismissed (idle) but the fleet is still empty, we DO restart on
      // a hard reload (the user came back). Use _wizardLoad guard.
      if (this.wizard.step !== 'idle') return;
      this.wizard = {
        step: 'ask',
        plan: null,
        startedAt: Date.now(),
        lastChip: '',
      };
      this._wizardLastTrack = '';
      this._wizardSave();
      this.track('wizard_started', { startedAt: this.wizard.startedAt });
    },

    wizardStartAsk() {
      // External trigger — same as startWizard but never collapses an
      // active wizard. Useful when a "show wizard again" button is
      // wired up later. For now ``loadAgents`` calls ``_maybeStartWizard``.
      this.startWizard();
    },

    wizardChipClicked(label) {
      if (!label) return;
      this.wizard.lastChip = label;
      this._wizardSave();
      this.track('wizard_chip_clicked', {
        label: label,
        step: this.wizard.step,
      });
      // "Something else…" seeds the input with a stem instead of
      // sending a chip — the user types freely. Wizard goes idle so
      // normal Operator chat resumes.
      const ELSE_LABEL = 'Something else…';
      if (label === ELSE_LABEL) {
        const el = document.getElementById('operator-chat-input');
        if (el) {
          // Seed the textarea via the embedded x-data ``opMsg`` model.
          try {
            const s = window.Alpine && Alpine.$data(el.closest('[x-data]'));
            if (s) s.opMsg = (s.opMsg ? s.opMsg + ' ' : '') + 'I want to ';
          } catch (_) { /* ignore */ }
          this.$nextTick(() => {
            el.focus();
            try { el.setSelectionRange(el.value.length, el.value.length); } catch (_) {}
          });
        }
        this._wizardComplete({ reason: 'something_else' });
        return;
      }
      // Send the chip label as a user message to the Operator. Use the
      // existing chat-send pipeline so transcript persistence + SSE
      // streaming work the same as a typed message.
      const sentAt = Date.now();
      this._wizardChipSentAt = sentAt;
      this.sendChatTo('operator', label);
      // Watch for an Operator reply to advance to confirming. Polls the
      // local chatHistories; the SSE handler updates that array as
      // tokens stream in.
      this._wizardWatchForReply(sentAt);
    },

    _wizardWatchForReply(sentAt) {
      if (this._wizardReplyPoll) clearInterval(this._wizardReplyPoll);
      const start = Date.now();
      this._wizardReplyPoll = setInterval(() => {
        // Bail out if the user dismissed mid-wait.
        if (this.wizard.step === 'idle') {
          clearInterval(this._wizardReplyPoll);
          this._wizardReplyPoll = null;
          return;
        }
        if (this._operatorHasReplyAfter(sentAt)) {
          clearInterval(this._wizardReplyPoll);
          this._wizardReplyPoll = null;
          this._wizardAdvance('confirming', { trigger: 'operator_reply' });
        } else if (Date.now() - start > 120000) {
          // Operator never replied within 2 minutes — give up the
          // automatic transition. The user can still advance manually
          // or abandon. Silent timeout — no telemetry to keep the
          // schema lean.
          clearInterval(this._wizardReplyPoll);
          this._wizardReplyPoll = null;
        }
      }, 750);
    },

    wizardConfirm() {
      // "Let's go" — operator already proposed a team in the previous
      // message. We move into the building card and start polling the
      // fleet for the first non-operator agent. The Operator's own
      // tool calls (apply_template / create_agent) are what actually
      // create agents — we don't call mesh APIs from here. We rely on
      // the operator having staged the build during the chat exchange.
      this._wizardAdvance('building', { trigger: 'confirm_button' });
      // Send a confirm message to the Operator so it actually executes
      // the proposed plan (the previous turn was a proposal).
      this.sendChatTo('operator', "Let's go — please build the team you proposed.");
      this._wizardStartBuildPolling();
    },

    wizardCustomize() {
      // "Customize…" exits the wizard without abandoning telemetry —
      // the user is engaged, just wants to type freely.
      this.track('wizard_chip_clicked', {
        label: 'customize',
        step: this.wizard.step,
      });
      this._wizardComplete({ reason: 'customize' });
    },

    _wizardStartBuildPolling() {
      // Stability-based detection: a multi-agent template is created
      // slot-by-slot, so a naive ``length >= 1`` check fires the success
      // card prematurely while the rest of the team is still being
      // provisioned. We require the non-operator fleet to be NON-EMPTY
      // and COUNT-STABLE for at least 10 seconds before declaring the
      // build done. Each new agent resets the stability window.
      if (this._wizardBuildPoll) clearInterval(this._wizardBuildPoll);
      const start = Date.now();
      let stableSince = 0;
      let lastCount = 0;
      this._wizardBuildPoll = setInterval(() => {
        const fleetAgents = this.agents.filter(a => a && a.id !== 'operator');
        const count = fleetAgents.length;
        const now = Date.now();
        if (count === 0) {
          // No fleet yet — keep waiting and reset the stability window.
          stableSince = 0;
          lastCount = 0;
        } else if (count === lastCount) {
          // Count unchanged — start (or continue) the stability window.
          if (!stableSince) stableSince = now;
          if (now - stableSince >= 10_000) {
            clearInterval(this._wizardBuildPoll);
            this._wizardBuildPoll = null;
            this._wizardAdvance('first-output', {
              trigger: 'fleet_stable',
              agents: fleetAgents.map(a => a.id).slice(0, 8),
              stableSeconds: 10,
            });
            return;
          }
        } else {
          // Count changed — reset stability window to current count.
          stableSince = now;
          lastCount = count;
        }
        if (now - start > 5 * 60 * 1000) {
          // 5 minute hard cap — if the build never stabilised we
          // surface a terminal ``build_failed`` card instead of
          // silently leaving the spinner on screen.
          clearInterval(this._wizardBuildPoll);
          this._wizardBuildPoll = null;
          this._wizardAdvance('build_failed', {
            trigger: 'timeout',
            elapsedMs: now - start,
          });
          this.track('wizard_build_timeout', {
            elapsedMs: now - start,
            fleetCount: count,
          });
          return;
        }
        // Each poll triggers a fetchAgents to keep the local cache fresh.
        if (typeof this.fetchAgents === 'function') this.fetchAgents();
      }, 2000);
    },

    wizardComplete() {
      // Public — bound to the "Continue" button on the first-output card.
      this._wizardComplete({ reason: 'continue_button' });
    },

    _wizardComplete(extra) {
      const totalSeconds = this._wizardSecondsSinceStart();
      const plan = this.wizard.plan;
      this.track('wizard_completed', Object.assign({
        totalSeconds: totalSeconds,
        plan: plan,
      }, extra || {}));
      this._wizardTeardown();
      this.wizard = { step: 'idle', plan: null, startedAt: 0, lastChip: '' };
      this._wizardSave();
      this._wizardLastTrack = '';
    },

    wizardAbandon() {
      const lastStep = this.wizard.step;
      this.track('wizard_abandoned', {
        lastStep: lastStep,
        totalSeconds: this._wizardSecondsSinceStart(),
      });
      this._wizardTeardown();
      this.wizard = { step: 'idle', plan: null, startedAt: 0, lastChip: '' };
      this._wizardSave();
      this._wizardLastTrack = '';
    },

    // Phase 4 — the wizard "x" / "skip wizard" handler. Same effect
    // as wizardAbandon (sets step=idle, persists, tears down pollers)
    // but with a different telemetry reason so we can tell deliberate
    // skips apart from page-unload abandons.
    wizardSkip() {
      const lastStep = this.wizard.step;
      this.track('wizard_abandoned', {
        lastStep: lastStep,
        totalSeconds: this._wizardSecondsSinceStart(),
        reason: 'skip_link',
      });
      this._wizardTeardown();
      this.wizard = { step: 'idle', plan: null, startedAt: 0, lastChip: '' };
      this._wizardSave();
      this._wizardLastTrack = '';
    },

    // Phase 4 — return progress-dot index for a given step. Used by
    // the wizard card progress indicator (4 dots: ask, confirming,
    // building, first-output). ``build_failed`` is a terminal sad-state
    // alternative to ``first-output`` and renders the same dot index
    // (3) so the user can see the progress reached the build phase
    // before failing.
    wizardStepIndex() {
      const order = ['ask', 'confirming', 'building', 'first-output'];
      if (this.wizard.step === 'build_failed') return 3;
      const idx = order.indexOf(this.wizard.step);
      return idx < 0 ? 0 : idx;
    },

    // Fix #2 — terminal ``build_failed`` retry button. Resets the
    // wizard back to ``ask`` so the user can try again with a new
    // chip selection.
    wizardRetryBuild() {
      this.track('wizard_chip_clicked', {
        label: 'retry_build',
        step: this.wizard.step,
      });
      this._wizardTeardown();
      this.wizard = {
        step: 'ask',
        plan: null,
        startedAt: Date.now(),
        lastChip: '',
      };
      this._wizardLastTrack = '';
      this._wizardSave();
    },

    // Fix #2 — terminal ``build_failed`` "Type instead" exit button.
    // Closes the wizard so the user can chat freely with the operator
    // without the failed-build card lingering.
    wizardExitToTyping() {
      this.track('wizard_chip_clicked', {
        label: 'type_instead',
        step: this.wizard.step,
      });
      this._wizardComplete({ reason: 'build_failed_exit' });
    },

    _wizardTeardown() {
      if (this._wizardBuildPoll) {
        clearInterval(this._wizardBuildPoll);
        this._wizardBuildPoll = null;
      }
      if (this._wizardReplyPoll) {
        clearInterval(this._wizardReplyPoll);
        this._wizardReplyPoll = null;
      }
      if (this._wizardCompleteTimer) {
        clearTimeout(this._wizardCompleteTimer);
        this._wizardCompleteTimer = null;
      }
    },

    _maybeStartWizard() {
      // Called after fetchAgents resolves AND chat history is loaded.
      // Idempotent — re-entrant calls during the same visit are no-ops
      // because startWizard guards on step !== 'idle'. We also bail out
      // if the user already advanced past 'ask' (restored from
      // localStorage), so a hot reload mid-build keeps the building UI.
      //
      // Fix #4 — operatorReady gate: the wizard card's parent element
      // has ``x-show="operatorReady"``, so starting the wizard while
      // the operator is unhealthy would persist ``step='ask'`` to
      // localStorage and emit telemetry while the user sees nothing.
      // ``checkOperatorReady`` retries this on health transitions.
      if (this.wizard.step !== 'idle') return;
      // Don't surface the in-chat starting-point card under the
      // non-skippable setup modal — wait until setup is finished.
      if (this.showSetup) return;
      if (!this.operatorReady) return;
      if (!this._isFirstVisit()) return;
      this.startWizard();
    },

    // ── Tab switching ─────────────────────────────────────

    switchTab(tab) {
      const fromTab = this.activeTab;
      this.activeTab = tab;
      this.detailAgent = null;
      // Clear tab-specific auto-refresh intervals
      this._stopActivityRefresh();
      // Phase 0 telemetry — record tab views (skip self-switches and
      // initial route restoration where ``fromTab === tab``). Also
      // counts as a user action for ``time_to_first_action``.
      if (fromTab !== tab) {
        this.track('tab_view', { tab_id: tab, from_tab_id: fromTab || '' });
        this._trackFirstAction('tab_view');
      }
      if (tab === 'chat') {
        if (!this.openChats.includes('operator')) {
          this.openChats.push('operator');
        }
        this.activeChatId = 'operator';
        this._loadChatHistory('operator');
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
        this.fetchTeamContent();
        this.fetchTeams();
        // Lead-with-Work: ensure the task ledger is loaded when entering the
        // Teams tab with a team active (init restores activeTeam directly, not
        // via switchTeam, so the Work view would otherwise render empty).
        if (this.activeTeam) this._ensureWorkplaceTasks();
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
          this.loadIntegrations();
          this.loadConnectors();
        }
        // apikeys needs only fetchSettings(), already called unconditionally above.
        if (this.systemTab === 'network') {
          this.loadNetworkProxy();
        }
        if (this.systemTab === 'skills') {
          this.loadSkillsCatalog();
        }
        if (this.systemTab === 'settings') {
          this.fetchBrowserSettings();
          this.fetchSystemSettings();
        }
        if (this.systemTab === 'browser') {
          this.fetchBrowserSettings();
          this.fetchSystemSettings();
          this.fetchCaptchaSolver();
          this.fetchPlatformSuccess();
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
      if (tabId === 'integrations') { this.fetchChannels(); this.fetchWebhooks(); this.fetchApiKeys(); this.loadIntegrations(); this.loadConnectors(); }
      if (tabId === 'apikeys') { this.fetchSettings(); }
      if (tabId === 'storage') { this.fetchUploads(); this.fetchStorage(); this.fetchDatabaseDetails(); }
      if (tabId === 'skills') { this.loadSkillsCatalog(); }
      if (tabId === 'network') { this.loadNetworkProxy(); }
      if (tabId === 'settings') { this.fetchBrowserSettings(); this.fetchSystemSettings(); }
      if (tabId === 'browser') { this.fetchBrowserSettings(); this.fetchSystemSettings(); this.fetchCaptchaSolver(); this.fetchPlatformSuccess(); }
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
        const params = new URLSearchParams({
          per_page: '20',
          page: String(this.auditPage),
        });
        if (this.auditIncludeArchived) params.set('include_archived', 'true');
        const resp = await fetch(`${window.__config.apiBase}/operator-audit?${params.toString()}`);
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

    // Compute an ISO date string for "now - N days" so the archive
    // request matches the SQLite TEXT timestamps the audit_log table
    // stores. We use the date-only form (YYYY-MM-DD) so users get a
    // predictable cutoff rather than a moving wall-clock window.
    _archiveCutoffIso(days) {
      const d = new Date();
      d.setUTCDate(d.getUTCDate() - Math.max(1, Math.floor(days || 1)));
      return d.toISOString().slice(0, 10);  // YYYY-MM-DD
    },

    async archiveOldAuditEntries() {
      if (this.auditArchiving) return;
      const days = Number(this.auditArchiveDays) || 30;
      const cutoff = this._archiveCutoffIso(days);
      this.auditArchiving = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/operator-audit/archive`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
          },
          body: JSON.stringify({ before_date: cutoff }),
        });
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          this.showToast(`Archive failed: ${data.detail || resp.status}`);
          return;
        }
        const data = await resp.json();
        const n = data.archived_count || 0;
        this.showToast(
          n === 0
            ? `No audit entries older than ${cutoff}.`
            : `Archived ${n} ${n === 1 ? 'entry' : 'entries'} older than ${cutoff}.`,
        );
        // Reload audit list so the UI reflects the change immediately.
        this.auditPage = 1;
        await this.fetchAuditLog();
      } catch (e) {
        this.showToast(`Archive failed: ${e.message || e}`);
      } finally {
        this.auditArchiving = false;
      }
    },

    toggleIncludeArchived() {
      this.auditIncludeArchived = !this.auditIncludeArchived;
      this.auditPage = 1;
      this.fetchAuditLog();
    },

    // ── Task 9 — Workplace tab loaders / event handlers ────────

    async loadWorkplace() {
      this.workplaceLoading = true;
      try {
        await Promise.all([
          this.loadWorkplaceTeams(),
          this.loadWorkplaceTasks(),
          this.loadWorkplaceBlockers(),
          this.loadWorkplacePending(),
          this.loadWorkplaceHelpRequests(),
          this.loadWorkplaceSummaries(),
          this.loadWorkplaceGoals(),
          this.loadWorkplacePipelines(),
        ]);
      } finally {
        this.workplaceLoading = false;
      }
    },

    // Per-section retry helper. Banners call ``retryWorkplaceSection``
    // with the section key — we clear the error first so the banner
    // hides immediately, then re-run the section's loader.
    retryWorkplaceSection(section) {
      this.workplaceErrors[section] = '';
      const fn = {
        teams: this.loadWorkplaceTeams,
        tasks: this.loadWorkplaceTasks,
        blockers: this.loadWorkplaceBlockers,
        pending: this.loadWorkplacePending,
        help: this.loadWorkplaceHelpRequests,
        summaries: this.loadWorkplaceSummaries,
        goals: this.loadWorkplaceGoals,
        pipelines: this.loadWorkplacePipelines,
      }[section];
      if (fn) fn.call(this);
    },

    // Live pipeline card — in-flight operator-rooted chains. Serial-guarded
    // (mirrors loadWorkplaceSummaries) because it's WS-debounce-refreshed and
    // can race the initial load / manual retry.
    async loadWorkplacePipelines() {
      const serial = (this._pipelinesFetchSerial || 0) + 1;
      this._pipelinesFetchSerial = serial;
      this.workplaceSectionLoading.pipelines = true;
      this.workplaceErrors.pipelines = '';
      try {
        const resp = await fetch(`${window.__config.apiBase}/workplace/pipelines`);
        if (serial !== this._pipelinesFetchSerial) return;  // superseded
        if (!resp.ok) {
          this.workplaceErrors.pipelines = `Couldn't load pipelines (HTTP ${resp.status})`;
          return;
        }
        const data = await resp.json();
        if (serial !== this._pipelinesFetchSerial) return;  // superseded
        this.workplacePipelines = data.pipelines || [];
      } catch (e) {
        if (serial !== this._pipelinesFetchSerial) return;  // superseded
        this.workplaceErrors.pipelines = (e && e.message)
          ? `Couldn't load pipelines: ${e.message}` : "Couldn't load pipelines";
      } finally {
        if (serial === this._pipelinesFetchSerial) {
          this.workplaceSectionLoading.pipelines = false;
        }
      }
    },

    // Tailwind classes for a pipeline stage chip, by task status.
    pipelineStageClass(status) {
      switch (status) {
        case 'done': return 'bg-emerald-900/30 border-emerald-700/40 text-emerald-300';
        case 'working': return 'bg-blue-900/30 border-blue-700/40 text-blue-300';
        case 'blocked': return 'bg-amber-900/30 border-amber-700/40 text-amber-300';
        case 'failed': return 'bg-red-900/30 border-red-700/40 text-red-300';
        case 'cancelled': return 'bg-gray-800/40 border-gray-700/40 text-gray-400';
        default: return 'bg-gray-800/40 border-gray-700/40 text-gray-300'; // pending / accepted
      }
    },

    // Solid dot color for the pipeline card's condensed (collapsed) stage
    // flow — same status palette as ``pipelineStageClass`` but as a filled
    // swatch rather than a bordered chip.
    pipelineStageDot(status) {
      switch (status) {
        case 'done': return 'bg-emerald-400/80';
        case 'working': return 'bg-blue-400/80';
        case 'blocked': return 'bg-amber-400/80';
        case 'failed': return 'bg-red-400/80';
        case 'cancelled': return 'bg-gray-600';
        default: return 'bg-gray-500/70'; // pending / accepted
      }
    },

    // Compact "age in current state" for a pipeline stage row: <1m / 12m /
    // 3h / 2d. Takes ``age_in_state_seconds`` from workflow_snapshot.
    humanizeAge(seconds) {
      const s = Math.max(0, Number(seconds) || 0);
      if (s < 60) return '<1m';
      if (s < 3600) return `${Math.round(s / 60)}m`;
      if (s < 86400) return `${Math.round(s / 3600)}h`;
      return `${Math.round(s / 86400)}d`;
    },

    // One-line "what's happening now" for the collapsed pipeline card:
    // the first non-terminal stage and what it's doing. Plain text.
    pipelineCurrentLabel(p) {
      const terminal = { done: 1, failed: 1, cancelled: 1 };
      const stages = (p && p.stages) || [];
      const cur = stages.find((s) => !terminal[s.status]);
      if (!cur) return 'wrapping up';
      const who = this.agentDisplayName(cur.assignee) || 'someone';
      if (cur.status === 'blocked') return `${who} blocked`;
      if (cur.status === 'working') return `${who} working`;
      return `${who} · ${cur.status}`;
    },

    async loadWorkplaceSummaries() {
      // Monotonic serial. ``loadWorkplaceSummaries`` can be called
      // concurrently (initial load + WS-debounce + manual retry); the
      // older fetch's resolve must not clobber a newer fetch's data.
      // We capture the serial at start and bail out at every write
      // point if a newer fetch has been kicked off since.
      const serial = (this._summariesFetchSerial || 0) + 1;
      this._summariesFetchSerial = serial;
      this.workplaceSectionLoading.summaries = true;
      this.workplaceErrors.summaries = '';
      try {
        const resp = await fetch(`${window.__config.apiBase}/workplace/summaries`);
        if (serial !== this._summariesFetchSerial) return;  // superseded
        if (!resp.ok) {
          this.workplaceErrors.summaries =
            `Couldn't load summaries (HTTP ${resp.status})`;
          return;
        }
        const data = await resp.json();
        if (serial !== this._summariesFetchSerial) return;  // superseded
        // Before replacing the list, cancel any pending deferred-apply
        // timers + pins on the old row objects. Without this cleanup
        // those setTimeout callbacks fire on rows no longer in the
        // displayed list (orphan mutations, wasted work). Each row's
        // `_pendingExternalTimer` is cleared by id-keyed dedupe.
        for (const row of (this.workplaceSummaries || [])) {
          if (row && row._pendingExternalTimer) {
            clearTimeout(row._pendingExternalTimer);
          }
        }
        // ``enabled: false`` means the dashboard's mesh app didn't wire
        // the summaries store (e.g. legacy deploy). Hide the section
        // cleanly — the rest of the Work tab still renders.
        if (data.enabled === false) {
          this.workplaceSummaries = [];
          return;
        }
        this.workplaceSummaries = data.summaries || [];
      } catch (e) {
        if (serial !== this._summariesFetchSerial) return;  // superseded
        this.workplaceErrors.summaries =
          (e && e.message) ? `Couldn't load summaries: ${e.message}`
                           : "Couldn't load summaries";
      } finally {
        // Only clear the loading flag for the LATEST serial — earlier
        // fetches' finally blocks must not flip the spinner off while
        // a newer fetch is still in-flight.
        if (serial === this._summariesFetchSerial) {
          this.workplaceSectionLoading.summaries = false;
        }
      }
    },

    // Submit a rating for a summary (accept / acknowledge / rework).
    // Optimistically updates the local row + closes any open feedback
    // box. Bumps the per-summary rate-sequence so a delayed
    // ``work_summary_rated`` WS event can't roll back to a stale state
    // (codex r1 P2). Returns early if a request for the same summary
    // is already in flight to prevent double-fires.
    async rateSummary(summaryId, rating, feedback = '') {
      const trimmed = (feedback || '').trim();
      if (rating === 'rework' && !trimmed) {
        // The template enforces this too, but defend at the handler
        // so a stray keyboard shortcut can't post a bare rework.
        return;
      }
      if (this._summaryRateInFlight[summaryId]) return;
      this._summaryRateInFlight[summaryId] = true;
      const seq = (this._summaryRateSeq[summaryId] || 0) + 1;
      this._summaryRateSeq[summaryId] = seq;
      const body = trimmed ? { rating, feedback: trimmed } : { rating };
      try {
        const resp = await fetch(
          `${window.__config.apiBase}/workplace/summaries/${encodeURIComponent(summaryId)}/rating`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Requested-With': 'XMLHttpRequest',
            },
            body: JSON.stringify(body),
          },
        );
        if (!resp.ok) {
          let detail = '';
          try { detail = (await resp.json()).detail || ''; } catch (_) {}
          this.workplaceErrors.summaries =
            `Rating failed (HTTP ${resp.status}${detail ? ': ' + detail : ''})`;
          return;
        }
        const updated = await resp.json();
        // Optimistic state sync. The WebSocket event will arrive
        // shortly and ratify (matching pin → clears pin) or be a
        // stale echo of an older POST (non-matching, within pin TTL
        // → stashed and applied after TTL).
        const list = this.workplaceSummaries || [];
        const row = list.find(r => r.id === summaryId);
        if (row) {
          // Clear any leftover stale-event timer + stash from a
          // PREVIOUS pin cycle on the same row — without this, a
          // deferred apply scheduled by an earlier accept → rework
          // sequence could fire AFTER a newer rating and overwrite
          // it with its stale stashed event (codex r5 P2 —
          // overlapping pin race).
          if (row._pendingExternalTimer) {
            clearTimeout(row._pendingExternalTimer);
            delete row._pendingExternalTimer;
          }
          delete row._pendingExternal;
          row.rating = updated.rating;
          row.feedback = updated.feedback;
          row.rated_at = updated.rated_at;
          row.rated_by = updated.rated_by;
          row._lastLocalRateSeq = seq;
          // Local-state pin: for the next ``_RATE_PIN_TTL`` seconds,
          // the locally-confirmed rating is authoritative. The WS
          // handler treats any incoming event that doesn't match
          // the pin as a stale delayed echo (no server-side rev
          // counter to compare, so client state is the anchor).
          // Pin clears on the matching ratification event OR on
          // TTL expiry.
          row._localPin = {
            rating: updated.rating,
            feedback: updated.feedback || null,
            ts: Date.now() / 1000,
          };
        }
        delete this.summaryFeedbackDrafts[summaryId];
      } catch (e) {
        this.workplaceErrors.summaries =
          (e && e.message) ? `Rating failed: ${e.message}` : 'Rating failed';
      } finally {
        delete this._summaryRateInFlight[summaryId];
      }
    },

    openSummaryFeedback(summaryId) {
      // Initialise the draft so the textarea binding has a key. Idempotent.
      if (!(summaryId in this.summaryFeedbackDrafts)) {
        this.summaryFeedbackDrafts[summaryId] = '';
      }
    },

    cancelSummaryFeedback(summaryId) {
      delete this.summaryFeedbackDrafts[summaryId];
    },

    async loadWorkplaceTeams() {
      this.workplaceSectionLoading.teams = true;
      this.workplaceErrors.teams = '';
      try {
        const resp = await fetch(`${window.__config.apiBase}/workplace/teams`);
        if (!resp.ok) {
          this.workplaceErrors.teams = `Couldn't load teams (HTTP ${resp.status})`;
          return;
        }
        const data = await resp.json();
        this.workplaceEnabled = data.enabled !== false;
        this.workplaceTeams = data.teams || [];
      } catch (e) {
        this.workplaceErrors.teams = (e && e.message) ? `Couldn't load teams: ${e.message}` : "Couldn't load teams";
      } finally {
        this.workplaceSectionLoading.teams = false;
      }
    },

    async loadWorkplaceTasks() {
      // Feeds the Stuck Tasks panel and the drill-in modal's
      // sibling-row updates. Pulls the unfiltered ledger — the
      // team-filter query param is no longer needed.
      this.workplaceSectionLoading.tasks = true;
      this.workplaceErrors.tasks = '';
      try {
        const resp = await fetch(`${window.__config.apiBase}/workplace/tasks`);
        if (!resp.ok) {
          this.workplaceErrors.tasks = `Couldn't load tasks (HTTP ${resp.status})`;
          return;
        }
        const data = await resp.json();
        this.workplaceEnabled = data.enabled !== false;
        this.workplaceTasks = data.tasks || [];
      } catch (e) {
        this.workplaceErrors.tasks = (e && e.message) ? `Couldn't load tasks: ${e.message}` : "Couldn't load tasks";
      } finally {
        this.workplaceSectionLoading.tasks = false;
      }
    },

    async loadWorkplaceBlockers() {
      this.workplaceSectionLoading.blockers = true;
      this.workplaceErrors.blockers = '';
      try {
        const resp = await fetch(`${window.__config.apiBase}/workplace/blockers`);
        if (!resp.ok) {
          this.workplaceErrors.blockers = `Couldn't load blockers (HTTP ${resp.status})`;
          return;
        }
        const data = await resp.json();
        this.workplaceEnabled = data.enabled !== false;
        this.workplaceBlockers = data.blockers || [];
      } catch (e) {
        this.workplaceErrors.blockers = (e && e.message) ? `Couldn't load blockers: ${e.message}` : "Couldn't load blockers";
      } finally {
        this.workplaceSectionLoading.blockers = false;
      }
    },

    async loadWorkplacePending() {
      this.workplaceSectionLoading.pending = true;
      this.workplaceErrors.pending = '';
      try {
        const resp = await fetch(`${window.__config.apiBase}/workplace/pending`);
        if (!resp.ok) {
          this.workplaceErrors.pending = `Couldn't load approvals (HTTP ${resp.status})`;
          return;
        }
        const data = await resp.json();
        this.workplacePending = data.pending || [];
        // Backfill the inline chat-card surface so a page reload still
        // shows any open pending actions in the operator chat. The
        // injection helper is idempotent on event_id so this is safe
        // to call alongside the live WS stream.
        for (const p of this.workplacePending) {
          this._injectPendingActionCard({
            nonce: p.nonce,
            actor: p.actor,
            target_kind: p.target_kind,
            target_id: p.target_id,
            action_kind: p.action_kind,
            summary: p.summary,
            preview_diff: p.preview_diff,
            expires_at: p.expires_at,
          });
        }
      } catch (e) {
        this.workplaceErrors.pending = (e && e.message) ? `Couldn't load approvals: ${e.message}` : "Couldn't load approvals";
      } finally {
        this.workplaceSectionLoading.pending = false;
      }
    },

    async loadWorkplaceHelpRequests() {
      // Authoritative source for the credential / browser-login / captcha
      // rows in "Needs you". Replaces scraping volatile operator-chat state
      // (which silently vanished on reload / transcript refresh). An empty
      // list reliably means nothing of these kinds needs the user; an HTTP
      // error sets ``help`` so the panel shows an explicit error state rather
      // than a misleading empty.
      this.workplaceSectionLoading.help = true;
      this.workplaceErrors.help = '';
      try {
        const resp = await fetch(`${window.__config.apiBase}/help-requests`);
        if (!resp.ok) {
          this.workplaceErrors.help = `Couldn't load open requests (HTTP ${resp.status})`;
          return;
        }
        const data = await resp.json();
        this.needsYouRequests = data.help_requests || [];
      } catch (e) {
        this.workplaceErrors.help = (e && e.message) ? `Couldn't load open requests: ${e.message}` : "Couldn't load open requests";
      } finally {
        this.workplaceSectionLoading.help = false;
      }
    },

    // Debounced refresh used by WS event handlers — credential/login/captcha
    // request + resolve + cancel events all just re-pull the authoritative
    // feed rather than surgically mutating client state (avoids drift).
    _refreshHelpRequestsSoon() {
      if (this._helpRefreshTimer) clearTimeout(this._helpRefreshTimer);
      this._helpRefreshTimer = setTimeout(() => {
        this._helpRefreshTimer = null;
        this.loadWorkplaceHelpRequests();
      }, 250);
    },

    async loadWorkplaceGoals() {
      // PR 2 — workplace-wide business goals managed by the operator
      // via ``manage_goals`` (PR 1). Render is a horizontal chip strip
      // above the Needs You panel; hidden entirely when empty.
      this.workplaceSectionLoading.goals = true;
      this.workplaceErrors.goals = '';
      try {
        const resp = await fetch(`${window.__config.apiBase}/workplace/goals`);
        if (!resp.ok) {
          this.workplaceErrors.goals = `Couldn't load goals (HTTP ${resp.status})`;
          return;
        }
        const data = await resp.json();
        this.workplaceGoals = data.goals || [];
      } catch (e) {
        this.workplaceErrors.goals = (e && e.message)
          ? `Couldn't load goals: ${e.message}`
          : "Couldn't load goals";
      } finally {
        this.workplaceSectionLoading.goals = false;
      }
    },

    // ── Task drill-in modal (PR 4) ────────────────────────
    //
    // Exposed as ``loadTaskDrillIn`` so the activity-feed PR's
    // ``openTaskDrillIn`` shim (which feature-detects this method by
    // name) delegates here cleanly when both PRs are merged. The
    // ``openTaskDrillIn`` alias keeps PR 4's own templates working in
    // isolation and survives merges in either order.

    async loadTaskDrillIn(taskId) {
      if (!taskId) return;
      this.drillInTaskId = taskId;
      this.drillInData = null;
      this.drillInComment = '';
      this.drillInError = '';
      this.drillInLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/workplace/tasks/${encodeURIComponent(taskId)}`);
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          this.drillInError = data.detail || `Failed to load task (HTTP ${resp.status})`;
          return;
        }
        this.drillInData = await resp.json();
      } catch (e) {
        this.drillInError = e.message || String(e);
      } finally {
        this.drillInLoading = false;
      }
    },

    async openTaskDrillIn(taskId) {
      // Alias kept so PR 4's templates (and any caller that races the
      // activity-feed PR's shim) reach the same loader.
      return this.loadTaskDrillIn(taskId);
    },

    closeTaskDrillIn() {
      this.drillInTaskId = null;
      this.drillInData = null;
      this.drillInComment = '';
      this.drillInError = '';
      this.drillInLoading = false;
      this.drillInSubmitting = false;
    },

    drillInIsTerminal() {
      const t = this.drillInData?.task;
      if (!t) return false;
      return t.status === 'done' || t.status === 'failed' || t.status === 'cancelled';
    },

    drillInOutcomeLabel(outcome) {
      if (outcome === 'accepted') return 'Accepted';
      if (outcome === 'acknowledged') return 'Acknowledged';
      if (outcome === 'rework') return 'Marked for rework';
      if (outcome === 'rejected') return 'Rejected';
      return '';
    },

    drillInTimeToComplete() {
      const t = this.drillInData?.task;
      if (!t || !t.created_at || !t.completed_at) return '';
      const secs = Math.max(0, Math.round(t.completed_at - t.created_at));
      if (secs < 60) return `${secs}s`;
      const mins = Math.floor(secs / 60);
      if (mins < 60) return `${mins}m ${secs % 60}s`;
      const hrs = Math.floor(mins / 60);
      return `${hrs}h ${mins % 60}m`;
    },

    drillInFormatTimestamp(ts) {
      if (!ts) return '';
      try {
        return new Date(ts * 1000).toLocaleString();
      } catch (_) {
        return '';
      }
    },

    drillInCanSubmit(outcome) {
      if (this.drillInSubmitting) return false;
      if (!this.drillInData?.task) return false;
      // Outcomes are write-many — an existing rating can be overwritten
      // (e.g. operator hit "Reject" by accident). The submit button for
      // the existing rating is disabled to prevent a no-op double-click.
      if (this.drillInData.task.outcome === outcome) return false;
      // ``accepted`` and ``acknowledged`` are silent-allowed; ``rework``
      // and ``rejected`` require feedback text so the agent has
      // something to learn from.
      if (outcome === 'accepted' || outcome === 'acknowledged') return true;
      return Boolean((this.drillInComment || '').trim());
    },

    async submitOutcome(outcome) {
      if (!this.drillInTaskId) return;
      if (!this.drillInCanSubmit(outcome)) return;
      this.drillInSubmitting = true;
      try {
        const resp = await fetch(
          `${window.__config.apiBase}/workplace/tasks/${encodeURIComponent(this.drillInTaskId)}/outcome`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Requested-With': 'XMLHttpRequest',
            },
            body: JSON.stringify({ outcome, feedback: this.drillInComment || '' }),
          }
        );
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          this.drillInError = data.detail || `Submit failed (HTTP ${resp.status})`;
          return;
        }
        // Optimistically reflect outcome in workplaceTasks so the
        // Stuck Tasks panel reflects the rating without waiting for
        // the next reload. (PR 2 dropped ``workplaceOutputs``.)
        const updated = data.task || {};
        const row = (this.workplaceTasks || []).find(r => r.id === this.drillInTaskId);
        if (row) {
          row.outcome = updated.outcome || outcome;
          row.feedback_text = updated.feedback_text ?? (this.drillInComment || null);
        }
        if (outcome === 'rework' && data.rework_task_id) {
          this.showToast(`Rework task created${data.rework_assignee ? ' for ' + data.rework_assignee : ''}.`);
        } else if (outcome === 'rework' && data.rework_error) {
          // Surface the rework spawn failure prominently — the outcome
          // saved but no follow-up task was created, so the operator
          // needs to know to retry from the rework task tools.
          this.showToast(
            `Outcome saved as needs-rework, but rework task could not be `
            + `spawned: ${data.rework_error}. Please retry from the rework `
            + `task tools.`
          );
        } else {
          this.showToast(`Outcome recorded: ${this.drillInOutcomeLabel(outcome) || outcome}.`);
        }
        this.closeTaskDrillIn();
      } catch (e) {
        this.drillInError = e.message || String(e);
      } finally {
        this.drillInSubmitting = false;
      }
    },

    handleDrillInKey(evt) {
      // Modal-scoped shortcuts. Skip when typing in the comment
      // textarea so 'a'/'r'/'x' keys don't fire while writing
      // feedback. Also skip when no task is loaded.
      if (!this.drillInTaskId || !this.drillInData?.task) return;
      const tag = (evt.target?.tagName || '').toLowerCase();
      if (tag === 'textarea' || tag === 'input' || tag === 'select') return;
      const key = evt.key?.toLowerCase();
      if (key === 'a' && this.drillInCanSubmit('accepted')) {
        evt.preventDefault();
        this.submitOutcome('accepted');
      } else if (key === 'r') {
        evt.preventDefault();
        // Focus the comment box so the operator can type the rework brief.
        const ta = document.getElementById('drill-in-comment');
        if (ta) ta.focus();
      } else if (key === 'x') {
        evt.preventDefault();
        const ta = document.getElementById('drill-in-comment');
        if (ta) ta.focus();
      }
    },

    workplaceFormatExpiry(expiresAt) {
      if (!expiresAt) return '';
      const remain = Math.max(0, Math.floor(expiresAt - (Date.now() / 1000)));
      if (remain < 60) return `${remain}s`;
      if (remain < 3600) return `${Math.floor(remain / 60)}m`;
      return `${Math.floor(remain / 3600)}h`;
    },

    // Aggregate every BLOCKING, user-resolvable surface into one ordered
    // list for the sticky panel. Only two sources, both SERVER-AUTHORITATIVE
    // (so an empty panel reliably means "nothing needs you" and items never
    // silently vanish on reload/tab-switch the way the old chat-scrape did):
    //   - workplacePending     (← /api/workplace/pending) durable delete-confirms
    //   - needsYouRequests      (← /api/help-requests) open credential /
    //     browser-login / captcha asks from the mesh registry
    // Deliberately NOT here: worker DMs (a notification, not a blocker — the
    // bell/unread dot owns those) and free-form blocked tasks (operator-
    // handled, not user-resolvable — showing them would be a dead end).
    get needsYouItems() {
      const items = [];

      for (const p of (this.workplacePending || [])) {
        // Build a "Review →" / "Confirm" two-step flow when the row
        // carries a preview_diff so non-technical users can see the
        // change before approving. Without a diff we keep the
        // immediate Confirm to preserve the old behaviour for rows
        // the server didn't enrich.
        const hasDiff = !!(p.preview_diff && String(p.preview_diff).trim());
        const itemId = 'pending-' + p.nonce;
        const expanded = !!this._needsYouPreviewExpanded[itemId];
        const previewAriaId = 'needs-you-preview-' + p.nonce;
        const primary = hasDiff && !expanded
          ? {
              label: 'Review',
              style: 'indigo',
              handler: () => { this._needsYouPreviewExpanded = { ...this._needsYouPreviewExpanded, [itemId]: true }; },
              ariaExpanded: 'false',
              ariaControls: previewAriaId,
            }
          : { label: 'Confirm', style: 'emerald', handler: () => this.confirmPendingAction(p.nonce) };
        items.push({
          id: itemId,
          kind: 'pending',
          title: this._humanizeAction(p.action_kind, p.target_kind, p.target_id, p.summary),
          subtitle: this._needsYouSubtitle({
            actor: p.actor,
            expiresAt: p.expires_at,
          }),
          previewDiff: hasDiff ? p.preview_diff : null,
          previewExpanded: expanded,
          previewToggleAriaId: previewAriaId,
          actions: [
            primary,
            { label: 'Cancel', style: 'gray', handler: () => this._confirmCancelPendingAction(p.nonce, this._humanizeAction(p.action_kind, p.target_kind, p.target_id, p.summary)) },
          ],
        });
      }

      // Credential / browser-login / captcha asks, straight from the
      // authoritative open-requests feed (NOT scraped from chat — that's why
      // they no longer vanish on reload). Each record carries request_id,
      // agent_id, service/name, and description.
      for (const req of (this.needsYouRequests || [])) {
        const who = this.agentDisplayName ? this.agentDisplayName(req.agent_id || '') : (req.agent_id || 'an agent');
        if (req.kind === 'credential_request') {
          // Resolve IN PLACE: the panel renders the same paste-and-save form
          // the chat card uses (POST /credentials/agent), passing request_id
          // so the mesh atomically pops the record + steers the agent. The
          // vault KEY is ``name``; ``service`` is the human label.
          items.push({
            id: 'help-' + req.request_id,
            kind: 'credential',
            title: `Add the ${req.service || req.name} key`,
            subtitle: this._needsYouSubtitle({
              actor: req.agent_id,
              text: req.description
                ? req.description
                : `Paste it below — saved to your vault, ${who} continues automatically.`,
            }),
            inlineCredential: {
              service: req.name || req.service,
              agentId: req.agent_id || '',
              display: req.service || req.name,
              requestId: req.request_id,
            },
            actions: [
              { label: 'Not now', style: 'gray', handler: () => this._cancelHelpRequest(req, 'credential') },
            ],
          });
        } else if (req.kind === 'browser_login_request') {
          // Login needs the live VNC viewer (which lives in the chat card),
          // so the action ensures that card exists — reconstructing it from
          // this authoritative record if the transcript was refreshed away —
          // then flashes it. No more dead-end "open chat at the bottom".
          items.push({
            id: 'help-' + req.request_id,
            kind: 'browser_login',
            title: `Sign in to ${req.service || 'a site'} for ${who}`,
            subtitle: this._needsYouSubtitle({
              actor: req.agent_id,
              text: req.description || 'Opens a live browser here — sign in, then press Complete login.',
            }),
            actions: [
              { label: 'Sign in', style: 'indigo', handler: () => this._openHelpRequestCard(req) },
              { label: 'Not now', style: 'gray', handler: () => this._cancelHelpRequest(req, 'browser-login') },
            ],
          });
        } else if (req.kind === 'browser_captcha_help_request') {
          items.push({
            id: 'help-' + req.request_id,
            kind: 'captcha',
            title: `Solve a CAPTCHA on ${req.service || 'a site'} for ${who}`,
            subtitle: this._needsYouSubtitle({
              actor: req.agent_id,
              text: req.description || 'Opens a live browser here — clear the check, then press Done.',
            }),
            actions: [
              { label: 'Solve it', style: 'indigo', handler: () => this._openHelpRequestCard(req) },
              { label: 'Not now', style: 'gray', handler: () => this._cancelHelpRequest(req, 'browser-captcha-help') },
            ],
          });
        }
      }

      return items;
    },

    // Cancel an open help request from the panel (feed-driven, no chat msg).
    // Hits the request_id-scoped cancel proxy (which steers the agent that
    // the ask won't be answered) then re-pulls the authoritative feed.
    async _cancelHelpRequest(req, slug) {
      try {
        const resp = await fetch(
          `${window.__config.apiBase}/${slug}-request/${encodeURIComponent(req.request_id)}/cancel`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify({ reason: 'user_cancelled' }),
          },
        );
        // A 404 means it was already resolved/cancelled elsewhere — benign,
        // the feed refresh below will drop the row. Surface only real errors.
        if (!resp.ok && resp.status !== 404 && typeof this.showToast === 'function') {
          this.showToast('Cancel failed — please try again');
        }
      } catch (_) {
        if (typeof this.showToast === 'function') this.showToast('Cancel failed — please try again');
      }
      this._refreshHelpRequestsSoon();
    },

    // Open (and if necessary RECONSTRUCT) the operator-chat card for a
    // login/captcha request, then flash it. The live VNC viewer + Complete
    // button live in that card; since the feed — not the transcript — is now
    // the source of truth, the card may have been dropped on a refresh, so we
    // re-synthesize it from the authoritative record. Carries request_id so
    // its Complete/Cancel resolve the registry record.
    _openHelpRequestCard(req) {
      if (!this.chatHistories['operator']) this.chatHistories['operator'] = [];
      let card = this.chatHistories['operator'].find(
        m => m.role === req.kind && m.request_id === req.request_id,
      );
      if (!card) {
        card = {
          role: req.kind,
          service: req.service || '',
          url: req.url || '',
          content: req.description || '',
          request_id: req.request_id,
          _from_agent: req.agent_id || '',
          ts: req.created_at ? req.created_at * 1000 : Date.now(),
        };
        this.chatHistories['operator'].push(card);
      }
      this._jumpToNeedsYouCard(card);
    },

    // Plain-English label for a pending action card. The server
    // already supplies ``summary`` for rich propose-edit rows; for
    // those we just use it. Otherwise we map known machine kinds
    // (``hard_edit``, ``delete_agent``, ...) into a sentence the
    // operator's user can act on without learning the codebase.
    // Returns a non-empty string (falls back to "{kind} on {target_kind}
    // {target_id}" with underscores replaced by spaces).
    _humanizeAction(kind, targetKind, targetId, summary) {
      if (summary && String(summary).trim()) return String(summary);
      const k = String(kind || '').toLowerCase();
      const tid = String(targetId || '').trim();
      const tkind = String(targetKind || '').trim();
      switch (k) {
        case 'hard_edit':
          return tid ? `Change settings on ${tid}` : 'Change settings';
        case 'soft_edit':
          return tid ? `Tune ${tid}` : 'Tune agent';
        case 'delete_agent':
          return tid ? `Remove agent ${tid}` : 'Remove agent';
        case 'delete_project':
          return tid ? `Remove project ${tid}` : 'Remove project';
        case 'archive_agent':
          return tid ? `Archive agent ${tid}` : 'Archive agent';
        case 'archive_project':
          return tid ? `Archive project ${tid}` : 'Archive project';
        default: {
          const verb = (k || 'action').replace(/_/g, ' ').trim();
          const tail = [tkind, tid].filter(Boolean).join(' ').trim();
          return tail ? `${verb} on ${tail}` : verb;
        }
      }
    },

    // Plain-English mapping for blocker_note codes. Returns
    // ``{ kind, label, service }``: ``label`` is the human sentence (rendered
    // in the drill-in banner and the "Needs you" subtitle); ``kind`` drives the
    // CTA (credential / browser-login get a "Fix" → operator chat; the rest are
    // operator-handled and not surfaced as user actions). ``service`` is
    // best-effort — from a ``cred:<name>`` shorthand or a ``:`` hint.
    _humanizeBlocker(rawNote) {
      const note = String(rawNote || '').trim();
      if (!note) return { kind: 'unknown', label: '', service: '' };
      const lower = note.toLowerCase();

      // ── User-actionable (credential / browser-login get a working Fix) ──
      if (lower === 'no_credentials' || lower === 'missing_credentials') {
        return { kind: 'credential', label: 'Needs a credential', service: '' };
      }
      if (lower.startsWith('cred:')) {
        const service = note.slice(5).trim();
        return {
          kind: 'credential',
          label: service ? `Needs a credential: ${service}` : 'Needs a credential',
          service,
        };
      }
      // Browser-login codes. URL/service hint may follow with a colon
      // (``needs_browser_login:example.com``) — surface it if present.
      if (lower === 'needs_browser_login' || lower === 'browser_login_required'
          || lower.startsWith('needs_browser_login:') || lower.startsWith('browser_login_required:')) {
        const idx = note.indexOf(':');
        const hint = idx >= 0 ? note.slice(idx + 1).trim() : '';
        return {
          kind: 'browser_login',
          label: hint ? `Needs a browser login: ${hint}` : 'Needs a browser login',
          service: hint,
        };
      }
      if (lower === 'awaiting_feedback' || lower === 'awaiting_user') {
        return { kind: 'feedback', label: 'Waiting for your feedback', service: '' };
      }
      if (lower === 'quota_exceeded' || lower === 'budget_exceeded') {
        return { kind: 'budget', label: 'Hit cost limit', service: '' };
      }
      if (lower.startsWith('lane_timeout')) {
        return { kind: 'too_big', label: 'This task was too large to finish in one run — try breaking it into smaller pieces.', service: '' };
      }

      // ── Engine-internal / transient (operator-handled; friendly label only
      //    surfaces if such a note reaches the drill-in banner) ──
      if (lower.startsWith('output_too_large')) {
        return { kind: 'internal', label: 'The agent tried to save more than fits in one step.', service: '' };
      }
      if (lower.startsWith('convergence_cap') || lower.startsWith('max_iterations')) {
        return { kind: 'internal', label: 'The task ran out of steps before finishing.', service: '' };
      }
      if (lower.startsWith('no_outbound_effects')) {
        return { kind: 'internal', label: 'The agent finished without producing any output.', service: '' };
      }
      if (lower.startsWith('agent_quarantined')) {
        return { kind: 'internal', label: 'The agent was paused after repeated errors.', service: '' };
      }
      // Transient AI-provider hiccups. Match only unambiguous error phrases.
      if (lower.includes('llm call failed') || lower.includes('returned empty response')
          || lower.includes('connection to the ai provider')) {
        return { kind: 'provider', label: 'The AI provider was briefly unavailable.', service: '' };
      }
      if (lower === 'internal_error' || lower.startsWith('exception')
          || lower.startsWith('dispatch_error') || lower.startsWith('auth_failure')
          || lower.startsWith('config_error')) {
        return { kind: 'internal', label: 'Something went wrong inside the system.', service: '' };
      }

      // A free-form note an agent wrote deliberately — render verbatim.
      return { kind: 'other', label: note, service: '' };
    },

    // Per-card preview disclosure state for the Needs-you panel.
    // Keyed by item id (``pending-<nonce>``). Rebuilt on every
    // needsYouItems read; we keep it on the component instance so
    // toggling stays sticky as the list re-renders.
    _needsYouPreviewExpanded: {},

    // Jump-to-card: slide the operator messenger in over the current tab
    // (it renders whenever activeTab !== 'chat') and flag THIS message so
    // its card scrolls into view and flashes — instead of openChat's
    // default scroll-to-bottom, which is what stranded the old "Open chat"
    // buttons at the wrong spot. If already on the Chat tab, openChat just
    // refocuses operator and the same flash fires there.
    _jumpToNeedsYouCard(msg) {
      if (this.openChat) this.openChat('operator');
      this.$nextTick(() => {
        if (!msg) return;
        // Increment (not just set true) so a repeat click on an already-
        // flagged card retriggers the x-effect and re-scrolls/re-flashes.
        msg._flash = (msg._flash || 0) + 1;
        // Clear after the highlight has been seen so it doesn't linger.
        setTimeout(() => { try { msg._flash = 0; } catch (_) { /* gone */ } }, 2600);
      });
    },

    // Build the small grey sub-line shared by every Needs-you card.
    // Joins the populated bits with " · " so blank fields don't leave
    // dangling separators.
    _needsYouSubtitle({ actor, expiresAt, project, text }) {
      const parts = [];
      if (expiresAt) parts.push('expires in ' + this.workplaceFormatExpiry(expiresAt));
      if (actor) parts.push('from ' + actor);
      if (project) parts.push('project ' + project);
      if (text) {
        const trimmed = text.length > 80 ? text.slice(0, 77) + '...' : text;
        parts.push(trimmed);
      }
      return parts.join(' · ');
    },

    // Tailwind colour map for Needs-you action buttons. Keeps the
    // template free of conditionals while letting each item declare
    // its own intent (emerald confirm, gray cancel, etc.).
    needsYouButtonClass(style) {
      switch (style) {
        case 'emerald':
          return 'bg-emerald-700 hover:bg-emerald-600 text-emerald-50';
        case 'amber':
          return 'bg-amber-700 hover:bg-amber-600 text-amber-50';
        case 'indigo':
          return 'bg-indigo-700 hover:bg-indigo-600 text-indigo-50';
        case 'gray':
        default:
          return 'bg-gray-700 hover:bg-gray-600 text-gray-100';
      }
    },

    // Coerce a wire-shape ``completed_at`` to a numeric epoch in seconds.
    // The orchestration store has historically returned epoch-seconds as a
    // float, but legacy / external feeders sometimes hand us strings or
    // millisecond integers — and a wall-clock skew can yield future
    // timestamps that should never count as "recent" for the user. We
    // accept all three shapes and clamp to <= now so the filter stays
    // monotonic even if the source clock drifts.
    _coerceCompletedAtSeconds(value) {
      if (value === null || value === undefined || value === '') return 0;
      let n;
      if (typeof value === 'number') {
        n = value;
      } else if (typeof value === 'string') {
        const parsed = Date.parse(value);
        if (Number.isFinite(parsed)) {
          n = parsed / 1000;
        } else {
          // Numeric string ("1730000000" or "1730000000.5") that
          // Date.parse refused — fall back to Number().
          const asNum = Number(value);
          n = Number.isFinite(asNum) ? asNum : 0;
        }
      } else {
        return 0;
      }
      // Heuristic: anything >= 10^12 is almost certainly milliseconds
      // (10^12 seconds = year ~33658). Drop to seconds.
      if (n >= 1e12) n = n / 1000;
      const nowSec = Date.now() / 1000;
      if (n > nowSec) n = nowSec;
      if (n < 0) n = 0;
      return n;
    },

    // Title truncation for the Stuck tasks panel and drill-in panel.
    // Long titles (some agents accidentally hand off with their full
    // instruction text — ~250 chars seen in production) wreck panel
    // layout; truncate to 80 chars + ellipsis on render. The full
    // title still surfaces in the drill-in modal and via the
    // ``title=`` tooltip on hover.
    truncateTitle(title, max) {
      const limit = Number.isFinite(max) ? max : 80;
      const s = String(title || '').trim();
      if (s.length <= limit) return s;
      return s.slice(0, limit - 1).trimEnd() + '…';
    },

    // Phase 4 — Stuck tasks: tasks that have been pending or working
    // for >24h with no status change. Computed from ``workplaceTasks``
    // by checking ``updated_at`` (or ``created_at`` for never-started
    // tasks). Returns an array of ``{id, title, assignee, project_id,
    // status, stuck_seconds, stuck_label}`` so the template can
    // render the count badge + ``Stuck for N days`` caption without
    // recomputing on every reactive read. Capped at 25 to keep the
    // panel from blowing past the fold; if a fleet has 25+ stuck
    // tasks the operator has bigger problems than UI density.
    get stuckTasks() {
      const tasks = this.workplaceTasks || [];
      const now = Date.now() / 1000;
      const cutoff = 24 * 3600;  // 24 hours in seconds
      const out = [];
      for (const t of tasks) {
        if (t.status !== 'pending' && t.status !== 'working') continue;
        // Defensive: operator is the orchestrator, not a worker — its
        // own ledger should never produce user-facing "stuck" rows even
        // if a future code path accidentally assigns work to it.
        if (t.assignee === 'operator') continue;
        const updated = typeof t.updated_at === 'number'
          ? t.updated_at
          : (t.updated_at ? new Date(t.updated_at).getTime() / 1000 : 0);
        const created = typeof t.created_at === 'number'
          ? t.created_at
          : (t.created_at ? new Date(t.created_at).getTime() / 1000 : 0);
        const ts = updated || created || 0;
        if (!ts) continue;
        const age = now - ts;
        if (age < cutoff) continue;
        const days = Math.floor(age / 86400);
        const label = days >= 1
          ? `Stuck for ${days} day${days === 1 ? '' : 's'}`
          : `Stuck for ${Math.floor(age / 3600)}h`;
        out.push({
          id: t.id,
          title: t.title || '(untitled)',
          assignee: t.assignee || '',
          project_id: t.project_id || '',
          status: t.status,
          stuck_seconds: age,
          stuck_label: label,
        });
        if (out.length >= 25) break;
      }
      // Oldest first so the most painful entries surface at the top.
      out.sort((a, b) => b.stuck_seconds - a.stuck_seconds);
      return out;
    },

    // Phase 4 — task cancel modal state. ``cancelTaskCandidate`` is
    // the task object the user is about to cancel; null means the
    // modal is closed. ``cancelTaskInFlight`` disables the confirm
    // button while the POST is in flight so a double-click doesn't
    // fire two cancels.
    cancelTaskCandidate: null,
    cancelTaskInFlight: false,

    // Open the cancel-task confirmation modal. Accepts a task object
    // (from the stuck-tasks panel or drill-in modal) — we copy just
    // the fields the modal needs so a downstream re-fetch doesn't
    // blow away the candidate while the user is staring at the dialog.
    confirmCancelTask(task) {
      if (!task || !task.id) return;
      this.cancelTaskCandidate = {
        id: task.id,
        title: task.title || '(untitled)',
        assignee: task.assignee || '',
        project_id: task.project_id || '',
      };
    },

    closeCancelTaskModal() {
      if (this.cancelTaskInFlight) return;
      this.cancelTaskCandidate = null;
    },

    // Fire the cancel — POSTs to the dashboard proxy which forwards
    // to the mesh's ``/mesh/tasks/{id}/cancel``. On success we close
    // the modal and reload tasks so the stuck-tasks panel reflects
    // the change immediately. The mesh also emits a
    // ``task_status_changed`` event so the WS path catches up too.
    async cancelTaskNow() {
      const cand = this.cancelTaskCandidate;
      if (!cand || this.cancelTaskInFlight) return;
      this.cancelTaskInFlight = true;
      try {
        const resp = await fetch(
          `${window.__config.apiBase}/workplace/tasks/${encodeURIComponent(cand.id)}/cancel`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Requested-With': 'XMLHttpRequest',
            },
            body: JSON.stringify({ reason: 'Cancelled from work tab' }),
          },
        );
        if (!resp.ok) {
          let detail = '';
          try { detail = (await resp.json()).detail || ''; } catch (e) { /* ignore */ }
          if (typeof this.showToast === 'function') {
            this.showToast(`Cancel failed: ${detail || resp.status}`);
          }
          return;
        }
        if (typeof this.showToast === 'function') {
          this.showToast(`Cancelled "${this.truncateTitle(cand.title, 40)}"`);
        }
        this.cancelTaskCandidate = null;
        // Refresh tasks so the stuck-tasks panel updates immediately.
        if (typeof this.loadWorkplaceTasks === 'function') {
          await this.loadWorkplaceTasks();
        }
      } catch (e) {
        if (typeof this.showToast === 'function') {
          this.showToast(`Cancel failed: ${e.message || e}`);
        }
      } finally {
        this.cancelTaskInFlight = false;
      }
    },

    // Restart agent — used by the [Restart agent] button on the
    // Stuck tasks panel. Hits the existing
    // ``/api/agents/{id}/restart`` dashboard endpoint; on success we
    // reload tasks so the panel reflects whatever the agent emits
    // when it comes back online.
    //
    // Routed through ``showConfirm`` (same modal pattern as the regular
    // Restart Agent button on the agent card — see ``restartAgent``)
    // so a stray click in the Stuck tasks panel doesn't kill an agent's
    // active work without an explicit confirmation. The fetch logic
    // lives in the confirm callback so the modal's spinner state ties
    // to the actual restart, not just the click.
    async restartAgentForStuck(agentId) {
      if (!agentId) return;
      const display = typeof this.agentDisplayName === 'function'
        ? this.agentDisplayName(agentId)
        : agentId;
      this.showConfirm(
        'Restart agent?',
        `This will interrupt any active work for "${display}". Their current task will be cancelled. Continue?`,
        async () => {
          try {
            const resp = await fetch(
              `${window.__config.apiBase}/agents/${encodeURIComponent(agentId)}/restart`,
              {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                  'X-Requested-With': 'XMLHttpRequest',
                },
                body: '{}',
              },
            );
            if (!resp.ok) {
              let detail = '';
              try { detail = (await resp.json()).detail || ''; } catch (e) { /* ignore */ }
              if (typeof this.showToast === 'function') {
                this.showToast(`Restart failed: ${detail || resp.status}`);
              }
              return;
            }
            if (typeof this.showToast === 'function') {
              this.showToast(`Restarting ${agentId}…`);
            }
          } catch (e) {
            if (typeof this.showToast === 'function') {
              this.showToast(`Restart failed: ${e.message || e}`);
            }
          }
        },
        true,
      );
    },

    // Audit-log Revert: re-uses the soft-edit undo endpoint via the
    // change_id stored on each audit row (which IS the undo_token).
    // The template already gates the button on ``entry.undoable``, but
    // we re-check here so a stale row (or a programmatic call) doesn't
    // hit the endpoint just to receive a 404.
    async revertAuditEntry(entry) {
      if (!entry || !entry.change_id || !entry.undoable) {
        this.showToast('This entry can no longer be reverted.');
        return;
      }
      this.auditReverting = { ...this.auditReverting, [entry.id]: true };
      try {
        const resp = await fetch(`${window.__config.apiBase}/changes/undo/${encodeURIComponent(entry.change_id)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '{}',
        });
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          this.showToast(`Revert failed: ${data.detail || resp.status}`);
          return;
        }
        this.showToast('Reverted.');
        await this.fetchAuditLog();
      } catch (e) {
        this.showToast(`Revert failed: ${e.message || e}`);
      } finally {
        const next = { ...this.auditReverting };
        delete next[entry.id];
        this.auditReverting = next;
      }
    },

    async confirmPendingAction(nonce) {
      try {
        // Route via the dashboard's loopback proxy so the mesh sees
        // x-mesh-internal + a human X-Origin. A direct browser-side
        // fetch to /mesh/pending/.../confirm fails the auth +
        // human-origin gates (see api_workplace_pending_confirm).
        const resp = await fetch(
          `${window.__config.apiBase}/workplace/pending/${encodeURIComponent(nonce)}/confirm`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Requested-With': 'XMLHttpRequest',
            },
            body: '{}',
          },
        );
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          this.showToast(`Confirm failed: ${data.detail || resp.status}`);
          return;
        }
        this.workplacePending = this.workplacePending.filter(p => p.nonce !== nonce);
        this.showToast('Approval confirmed.');
      } catch (e) {
        this.showToast(`Confirm failed: ${e.message || e}`);
      }
    },

    async cancelPendingAction(nonce) {
      try {
        const resp = await fetch(`${window.__config.apiBase}/workplace/pending/${encodeURIComponent(nonce)}/cancel`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '{}',
        });
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          this.showToast(`Cancel failed: ${data.detail || resp.status}`);
          return;
        }
        this.workplacePending = this.workplacePending.filter(p => p.nonce !== nonce);
        this.showToast('Approval cancelled.');
      } catch (e) {
        this.showToast(`Cancel failed: ${e.message || e}`);
      }
    },

    /**
     * Confirm-then-cancel wrapper for the Needs-You "Cancel" pending
     * action button. Pops the existing confirm modal so a stray click
     * doesn't dismiss the team's request — they can't retry until they
     * ask again. The label is the action title shown on the chip.
     */
    _confirmCancelPendingAction(nonce, label) {
      const safeLabel = (label && String(label).trim()) || 'this approval';
      this.showConfirm(
        'Cancel approval?',
        `Cancelling will dismiss "${safeLabel}". The team won't be able to retry until they ask again.`,
        async () => { await this.cancelPendingAction(nonce); },
        true,
      );
    },

    // Confirm-button handler for the inline pending_action_card.
    // Routes through the legacy ``/mesh/pending/{nonce}/confirm`` thin
    // wrapper which dispatches to the right backend (config edit vs.
    // destructive delete) by inspecting the stored row. The card's
    // ``resolved_status`` flips to ``confirmed`` when the
    // ``pending_action_resolved`` WS event lands; we don't mutate the
    // message here so the round-trip stays the source of truth.
    async confirmPendingActionCard(msg) {
      try {
        // Loopback proxy — see confirmPendingAction note. payload_digest
        // threads through so the mesh's drift check still fires.
        const resp = await fetch(
          `${window.__config.apiBase}/workplace/pending/${encodeURIComponent(msg.event_id)}/confirm`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Requested-With': 'XMLHttpRequest',
            },
            body: JSON.stringify({ payload_digest: msg.payload_digest || undefined }),
          },
        );
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          this.showToast(`Confirm failed: ${data.detail || resp.status}`);
          return;
        }
      } catch (e) {
        this.showToast(`Confirm failed: ${e.message || e}`);
      }
    },

    async cancelPendingActionCard(msg) {
      try {
        // Loopback proxy (the cancel endpoint doesn't require human
        // origin, but a direct browser call still fails in prod
        // because the mesh requires a bearer token or x-mesh-internal).
        const resp = await fetch(
          `${window.__config.apiBase}/workplace/pending/${encodeURIComponent(msg.event_id)}/cancel`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Requested-With': 'XMLHttpRequest',
            },
            body: '{}',
          },
        );
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          this.showToast(`Cancel failed: ${data.detail || resp.status}`);
          return;
        }
      } catch (e) {
        this.showToast(`Cancel failed: ${e.message || e}`);
      }
    },

    // Inject a pending_action_card into the operator's chat history (and,
    // if the user is currently viewing the operator in the main chat,
    // into ``activeChatId === 'operator'`` — which is the same array
    // because chatHistories is keyed by agent ID). Idempotent on
    // event_id so a reload that re-fires the WS event doesn't duplicate
    // the card.
    _injectPendingActionCard(data) {
      const target = 'operator';
      if (!this.chatHistories[target]) this.chatHistories[target] = [];
      const existing = this.chatHistories[target].find(
        m => m.role === 'pending_action_card' && m.event_id === data.nonce,
      );
      if (existing) return;
      const isDestructive = (
        data.action_kind === 'delete'
        && (data.target_kind === 'project' || data.target_kind === 'agent')
      );
      this.chatHistories[target].push({
        role: 'pending_action_card',
        event_id: data.nonce,
        actor: data.actor || 'operator',
        target_kind: data.target_kind || '',
        target_id: data.target_id || '',
        action_kind: data.action_kind || '',
        summary: data.summary || '',
        preview_diff: data.preview_diff || '',
        expires_at: data.expires_at || 0,
        _isDestructive: isDestructive,
        resolved_status: null,
        ts: Date.now() / 1000,
      });
      // If the collapsed bar is currently up, flash the count badge
      // for ~2s so the user notices a new pending arrived without
      // the bar yelling at them. We don't auto-expand: the operator's
      // view stays calm and the user opens the bar on their schedule.
      if (!this.pendingExpanded && this._countPendingActionCards() >= 2) {
        this._flashPendingPulse();
      }
    },

    // Count the number of unresolved pending_action_card messages in
    // the operator chat. Drives the "N actions awaiting you" headline
    // and the collapse-vs-expand decision.
    _countPendingActionCards() {
      const arr = this.chatHistories?.['operator'] || [];
      let n = 0;
      for (const m of arr) {
        if (m.role === 'pending_action_card' && !m.resolved_status) n++;
      }
      return n;
    },

    // Toggle the collapsed bar. Used by the click handler and the
    // Enter/Space keyboard handler. Reset the pulse when the user
    // expands so the badge stops drawing attention. When expanding,
    // move keyboard focus to the first revealed card so screen-reader
    // and keyboard users land on the new content (matches the typical
    // disclosure-widget contract).
    togglePendingExpanded() {
      this.pendingExpanded = !this.pendingExpanded;
      if (this.pendingExpanded) {
        this.pendingPulse = false;
        // Defer until Alpine has re-rendered the now-visible cards.
        this.$nextTick(() => {
          // Both render sites tag their cards container with
          // ``data-pending-cards``; we focus the first card under the
          // first matching container in DOM order. If neither
          // container is mounted (e.g. operator chat collapsed),
          // there's nothing to focus and we silently no-op.
          const container = document.querySelector('[data-pending-cards]');
          if (!container) return;
          const card = container.querySelector('[data-pending-card]');
          if (card && typeof card.focus === 'function') card.focus();
        });
      }
    },

    // Flash the count badge for ~2s so a fresh pending while
    // collapsed catches the eye. Cancellable so back-to-back arrivals
    // don't pile up.
    _flashPendingPulse() {
      this.pendingPulse = true;
      if (this._pendingPulseTimer) {
        clearTimeout(this._pendingPulseTimer);
      }
      this._pendingPulseTimer = setTimeout(() => {
        this.pendingPulse = false;
        this._pendingPulseTimer = null;
      }, 2000);
    },

    // Find the matching card by event_id and stamp a terminal state.
    // Skips silently if the card isn't in history (e.g. WS event arrived
    // for a nonce we never injected because the operator chat wasn't
    // loaded yet).
    _resolvePendingActionCard(eventId, status) {
      for (const key of Object.keys(this.chatHistories || {})) {
        const arr = this.chatHistories[key];
        if (!Array.isArray(arr)) continue;
        const m = arr.find(x => x.role === 'pending_action_card' && x.event_id === eventId);
        if (m) m.resolved_status = status;
      }
    },

    // Apply a live event from the WebSocket to the in-memory workplace
    // state so the user sees changes without reloading. Each handler is
    // a small upsert into one of the lists; misses are tolerated (the
    // periodic full-load is the source of truth on reconnect).
    handleWorkplaceEvent(evt) {
      if (!evt || !evt.type) return;
      const data = evt.data || {};
      // Live pipeline card — any task lifecycle change can move a chain
      // through its stages, so debounce-refresh the in-flight pipelines.
      if (evt.type === 'task_created' || evt.type === 'task_status_changed'
          || evt.type === 'task_outcome') {
        if (this._pipelinesDebounce) clearTimeout(this._pipelinesDebounce);
        this._pipelinesDebounce = setTimeout(() => {
          if (typeof this.loadWorkplacePipelines === 'function') {
            this.loadWorkplacePipelines();
          }
        }, 300);
      }
      if (evt.type === 'task_created') {
        // Add stub row so the Stuck Tasks panel + drill-in see it
        // immediately; background reload fills in the rest.
        const stub = {
          id: data.task_id,
          project_id: data.project_id,
          creator: data.creator,
          assignee: data.assignee,
          title: data.title || '(untitled)',
          status: data.status || 'pending',
          created_at: data.created_at,
        };
        if (!this.workplaceTasks.find(t => t.id === stub.id)) {
          this.workplaceTasks.unshift(stub);
        }
      } else if (evt.type === 'task_status_changed') {
        const t = this.workplaceTasks.find(x => x.id === data.task_id);
        const prevStatus = t ? t.status : (data.from_status || data.old_status || '');
        if (t) {
          t.status = data.new_status;
          if (data.assignee) t.assignee = data.assignee;
          if (data.blocker_note !== undefined) t.blocker_note = data.blocker_note;
          // The status-change payload omits completed_at, but the team-hub
          // Work view buckets "Delivered today" by it — without this a task
          // that completes while the hub is open drops out of every bucket
          // until a full reload. Prefer a server value if one is ever added;
          // otherwise approximate with the client clock (corrected on reload).
          if (data.new_status === 'done' && !t.completed_at) {
            t.completed_at = (data.completed_at != null) ? data.completed_at : (Date.now() / 1000);
          }
        }
        // Pinned blockers list lives on a separate endpoint; refresh it
        // whenever a task transitions in to or out of ``blocked``.
        const newStatus = data.new_status || '';
        if (
          prevStatus === 'blocked'
          || newStatus === 'blocked'
        ) {
          this.loadWorkplaceBlockers();
        }
      } else if (evt.type === 'task_outcome') {
        // Reflect the rating in ``workplaceTasks`` (Stuck panel) and
        // the open drill-in modal. Background reload is source of
        // truth on reconnect.
        const row = (this.workplaceTasks || []).find(r => r.id === data.task_id);
        if (row) {
          row.outcome = data.outcome;
          row.feedback_text = data.feedback || row.feedback_text || null;
        }
        if (this.drillInData?.task && this.drillInData.task.id === data.task_id) {
          this.drillInData.task.outcome = data.outcome;
          this.drillInData.task.feedback_text = data.feedback || null;
        }
      } else if (evt.type === 'work_summary_created') {
        // New summary card landed. Re-fetch the list (the full record
        // is one fetch away; avoids racing the store-write completion
        // when the WS event arrives first). Dedupe by id.
        // Debounced to 250ms so the daily cron firing across N teams
        // in the same second triggers ONE re-fetch, not N. Without
        // this guard a 30-team fleet would N-storm the dashboard's
        // /api/workplace/summaries route at every cron tick.
        if (data.summary_id && !(this.workplaceSummaries || []).find(
              s => s.id === data.summary_id)) {
          if (this._summariesRefetchDebounce) {
            clearTimeout(this._summariesRefetchDebounce);
          }
          this._summariesRefetchDebounce = setTimeout(() => {
            this._summariesRefetchDebounce = null;
            this.loadWorkplaceSummaries();
          }, 250);
        }
      } else if (evt.type === 'work_summary_rated') {
        // Reflect the rating live on the summary card UNLESS the
        // local user just posted a fresher rating that hasn't been
        // ratified yet. ``_lastLocalRateSeq`` is the monotonic
        // counter stamped by ``rateSummary`` after a successful
        // POST; while a newer request is in flight, ignore the WS
        // event so a delayed echo of the prior rating can't roll
        // back our state (codex r1 P2 — rapid rating-edit race).
        const row = (this.workplaceSummaries || []).find(
          s => s.id === data.summary_id);
        if (row) {
          // If a POST is currently in flight, skip — the response
          // handler will write the canonical state.
          if (this._summaryRateInFlight[data.summary_id]) return;
          // Local-state pin: a recent local POST anchors the row's
          // rating for ``_RATE_PIN_TTL`` seconds. Within that window,
          // WS events that DON'T match the pin are EITHER stale
          // delayed echoes of an older POST (should drop) OR
          // legitimate external mutations from somewhere else
          // (operator re-rating via chat, etc.). Without a server-
          // side revision counter we can't tell them apart, so we
          // stash the latest non-matching event and apply it after
          // the pin's TTL elapses. The matching ratification clears
          // both the pin and any stashed pending event.
          const _RATE_PIN_TTL_S = 10;
          const pin = row._localPin;
          if (pin) {
            const pinAge = Date.now() / 1000 - pin.ts;
            if (pinAge < _RATE_PIN_TTL_S) {
              const eventMatchesPin = (
                data.rating === pin.rating
                && (data.feedback || null) === pin.feedback
              );
              if (eventMatchesPin) {
                // Ratification of our local POST — clear both the
                // pin and any stashed external event (the canonical
                // state is the pin's anchor).
                delete row._localPin;
                if (row._pendingExternalTimer) {
                  clearTimeout(row._pendingExternalTimer);
                  delete row._pendingExternalTimer;
                }
                delete row._pendingExternal;
                return;
              }
              // Non-matching event while pin is fresh. Stash the
              // latest non-matching event; schedule a deferred
              // apply for when the pin's TTL elapses. If a newer
              // non-matching event arrives, it replaces the stash
              // (the timer keeps the original deadline so we don't
              // extend the pin indefinitely).
              row._pendingExternal = data;
              if (!row._pendingExternalTimer) {
                const ttlRemainingMs = Math.max(
                  100, (pin.ts + _RATE_PIN_TTL_S - Date.now() / 1000) * 1000 + 50,
                );
                row._pendingExternalTimer = setTimeout(() => {
                  const ev = row._pendingExternal;
                  if (ev) {
                    row.rating = ev.rating;
                    row.feedback = ev.feedback || row.feedback || null;
                    row.rated_at = ev.ts || row.rated_at;
                    row.rated_by = ev.actor || row.rated_by;
                  }
                  delete row._pendingExternal;
                  delete row._pendingExternalTimer;
                  delete row._localPin;
                }, ttlRemainingMs);
              }
              return;
            }
            // TTL expired — clear pin and fall through to apply.
            delete row._localPin;
            if (row._pendingExternalTimer) {
              clearTimeout(row._pendingExternalTimer);
              delete row._pendingExternalTimer;
            }
            delete row._pendingExternal;
          }
          row.rating = data.rating;
          row.feedback = data.feedback || row.feedback || null;
          row.rated_at = data.ts || row.rated_at;
          row.rated_by = data.actor || row.rated_by;
        }
      } else if (evt.type === 'pending_action_created') {
        if (!this.workplacePending.find(p => p.nonce === data.nonce)) {
          this.workplacePending.push({
            nonce: data.nonce,
            actor: data.actor,
            target_kind: data.target_kind,
            target_id: data.target_id,
            action_kind: data.action_kind,
            summary: data.summary,
            preview_diff: data.preview_diff,
            expires_at: data.expires_at,
            created_at: Date.now() / 1000,
          });
        }
        // Also surface as an inline chat card in the operator chat —
        // the new single visual language for "operator wants the human
        // to act." Idempotent on event_id.
        this._injectPendingActionCard(data);
      } else if (evt.type === 'pending_action_resolved') {
        this.workplacePending = this.workplacePending.filter(p => p.nonce !== data.nonce);
        // ``status`` is "confirmed" (success) or "cancelled".
        this._resolvePendingActionCard(data.nonce, data.status || 'confirmed');
      } else if (evt.type === 'pending_action_expired') {
        this.workplacePending = this.workplacePending.filter(p => p.nonce !== data.nonce);
        this._resolvePendingActionCard(data.nonce, 'expired');
      }
    },

    // ── Operator action chips (Phase 3) ──────────────────
    //
    // Server-side, the operator's prompt instructs every response to end
    // with 2-4 ``ACTION: <label>`` lines. ``_parseOperatorActions`` strips
    // those lines from the message body and returns ``{body, actions}``
    // so the chat renderer can show the prose untouched and render the
    // labels as clickable chips below the bubble. Click → sends the
    // label as the user's next message.
    //
    // The format is intentionally tolerant — the LLM occasionally emits
    // bullet variants ("- ACTION:" / "* ACTION:") or wraps the block in
    // a fenced code fence. We strip those wrappers, accept dash-prefixed
    // lines, and bail out cleanly if no ACTION lines are present.
    //
    // Returns ``{body, actions}`` where ``actions`` is an array of label
    // strings (≤40 chars, deduped, max 6). When no chips parsed,
    // ``actions`` is empty and ``body`` is the original text untouched —
    // the renderer falls back to free-text only (Decision #16).
    _parseOperatorActions(text) {
      if (!text || typeof text !== 'string') return { body: text || '', actions: [] };
      const lines = text.split(/\r?\n/);
      const actions = [];
      let trailing = lines.length;
      // Walk from the end, peeling off trailing blank / fence / ACTION
      // lines until we hit a real content line. We stop at the first
      // non-matching line so an ACTION block that ends in the middle of
      // a longer message stays embedded as plain text (the format
      // contract says they go at the very end).
      for (let i = lines.length - 1; i >= 0; i--) {
        const raw = lines[i];
        const stripped = raw.trim();
        if (!stripped) { trailing = i; continue; }
        if (stripped === '```' || stripped.startsWith('```')) { trailing = i; continue; }
        const m = stripped.match(/^(?:[-*]\s+)?ACTION\s*:\s*(.+)$/i);
        if (m) {
          const label = m[1].trim().replace(/^["'`]+|["'`]+$/g, '').trim();
          if (label && label.length <= 80) actions.unshift(label.slice(0, 60));
          trailing = i;
          continue;
        }
        break;
      }
      if (actions.length === 0) return { body: text, actions: [] };
      // Dedupe (case-insensitive) and cap at 6 chips so a runaway model
      // can't paint a wall of buttons.
      const seen = new Set();
      const deduped = [];
      for (const label of actions) {
        const key = label.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        deduped.push(label);
        if (deduped.length >= 6) break;
      }
      const body = lines.slice(0, trailing).join('\n').replace(/\s+$/g, '');
      return { body, actions: deduped };
    },

    // Apply ACTION-line parsing to a message entry in place. Idempotent:
    // safe to call multiple times on the same entry (e.g. mid-stream and
    // again on stream done) since ``content`` is overwritten with the
    // body and ``suggested_actions`` with the latest parsed list.
    _applyOperatorActions(entry) {
      if (!entry || entry.role !== 'agent') return;
      const { body, actions } = this._parseOperatorActions(entry.content || '');
      entry.content = body;
      entry.suggested_actions = actions;
    },

    // Click handler for operator action chips. Sends the label as the
    // user's next message via the existing send infrastructure (steer
    // if operator busy, sendChatTo otherwise).
    sendOperatorChip(label) {
      const msg = (label || '').trim();
      if (!msg) return;
      this._operatorLastUserMessageTs = Date.now();
      if (this.isAgentBusy('operator')) {
        this.steerAgent('operator', msg);
      } else {
        this.sendChatTo('operator', msg);
      }
    },

    // Whether the default "Quick actions" menu should render in the
    // operator chat. Shown when there are no user messages in the
    // history OR the last user message was >5 min ago. Hidden while a
    // stream is in flight so chips don't appear above an in-progress
    // response.
    showOperatorDefaultChips() {
      if (this.chatStreamingAgents && this.chatStreamingAgents['operator']) return false;
      if (this.chatLoadingAgents && this.chatLoadingAgents['operator']) return false;
      const hist = (this.chatHistories && this.chatHistories['operator']) || [];
      // No chat history yet — show defaults so first-visit user has a
      // clear starting point.
      if (!hist.some(m => m && m.role === 'user')) return true;
      // Find the most recent user message timestamp and gate on 5 min.
      let lastTs = this._operatorLastUserMessageTs || 0;
      for (let i = hist.length - 1; i >= 0; i--) {
        if (hist[i] && hist[i].role === 'user') { lastTs = Math.max(lastTs, hist[i].ts || 0); break; }
      }
      if (!lastTs) return false;
      return (Date.now() - lastTs) > 5 * 60 * 1000;
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

    // PR-L' — render "Last seen Nm ago · Last task Mm ago" on the
    // agent card. ``last_healthy`` is epoch SECONDS (HealthMonitor),
    // ``last_task_event_ts`` is epoch SECONDS (Tasks.task_events).
    // Fresh-enough timestamps (<1 min) yield empty strings from
    // formatRelativeTime — fall back to "just now" for legibility.
    agentActivityLabel(agent) {
      // "Last activity" = the last time the agent actually WORKED, i.e. its
      // most recent LLM call (last_worked_ts, from the usage ledger). This
      // deliberately does NOT use last_healthy — that's a container health
      // probe, not work. Falls back to the last task event when the agent
      // has no recorded LLM call yet (e.g. freshly created).
      const fmt = (sec) => {
        if (!sec) return '';
        const rel = this.formatRelativeTime(sec * 1000);
        return rel || 'just now';
      };
      const worked = fmt(agent.last_worked_ts);
      if (worked) return 'Worked ' + worked;
      const task = fmt(agent.last_task_event_ts);
      if (task) return 'Task ' + task;
      return '';
    },

    healthLabel(status) {
      const map = { healthy: 'Online', unhealthy: 'Degraded', restarting: 'Degraded', quarantined: 'Quarantined', failed: 'Offline', unknown: 'Starting' };
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

      // Append to event feed (newest first, cap at 500). queue_changed is a
      // high-frequency, contentless refetch trigger (≈3 per task) with no
      // human-readable rendering — keep it out of the bounded activity buffer
      // so it neither crowds real events out of the 500-slot history nor
      // inflates the "+N coordination events" counter. Its handler still runs
      // below; dedup above already recorded its id.
      if (evt.type !== 'queue_changed') {
        this.events.unshift(evt);
        if (this.events.length > 500) this.events.splice(500);
      }

      // Update agent activity state
      const agent = evt.agent;
      if (agent) {
        this._updateAgentState(agent, evt.type);
      }

      // Live-update fleet on llm_call/health/agent_state changes (debounced)
      if (evt.type === 'llm_call' || evt.type === 'health_change' || evt.type === 'agent_state') {
        this._debouncedFleetRefresh();
        // Refresh capabilities when an agent re-registers (e.g. new tool authored)
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

      // Task 9 — Workplace tab + pending action review live updates.
      // The handler upserts into the Alpine.js state lists so the SPA
      // reflects task lifecycle and pending-action arrivals without a
      // full reload.
      if (evt.type === 'task_created' || evt.type === 'task_status_changed' ||
          evt.type === 'task_outcome' ||
          evt.type === 'pending_action_created' || evt.type === 'pending_action_resolved' ||
          evt.type === 'pending_action_expired' ||
          // PR-B — work summary lifecycle on the Work tab. Without
          // these entries the handler arms below stay dead and the
          // summaries view doesn't update live.
          evt.type === 'work_summary_created' || evt.type === 'work_summary_rated') {
        this.handleWorkplaceEvent(evt);
      }

      // Keep the "Needs you" help-request feed live. Any
      // request/resolve/cancel for the three user-actionable kinds re-pulls
      // the authoritative feed (debounced) — the chat-card handlers below
      // still render the in-chat copy; this just keeps the panel in sync
      // without scraping that volatile chat state.
      if (evt.type === 'credential_request' || evt.type === 'credential_stored' ||
          evt.type === 'credential_request_cancelled' ||
          evt.type === 'browser_login_request' || evt.type === 'browser_login_completed' ||
          evt.type === 'browser_login_cancelled' ||
          evt.type === 'browser_captcha_help_request' || evt.type === 'browser_captcha_help_completed' ||
          evt.type === 'browser_captcha_help_cancelled') {
        this._refreshHelpRequestsSoon();
      }

      // Live task-artifact attach. The orchestration layer emits
      // ``task_artifact_added`` whenever a tool result is logged onto a
      // task. Without an SPA handler the user has to refresh the task
      // drawer to see the new artifact. Best-effort: refresh the drawer
      // when it's open on the task in question, and nudge the workplace
      // tasks list (debounced) so any aggregate counts stay current.
      if (evt.type === 'task_artifact_added') {
        const taskId = evt.data?.task_id;
        if (taskId) {
          if (this.drillInTaskId === taskId && typeof this.loadTaskDrillIn === 'function') {
            this.loadTaskDrillIn(taskId);
          }
          if (typeof this.loadWorkplaceTasks === 'function') {
            if (this._workplaceTasksDebounce) clearTimeout(this._workplaceTasksDebounce);
            this._workplaceTasksDebounce = setTimeout(() => this.loadWorkplaceTasks(), 250);
          }
        }
      }

      // PR 1 — operator_action_receipt: append a receipt card to the
      // operator chat history so the user sees what was changed and gets
      // a one-click [Undo]. Also append into the affected agent's chat
      // (if open) so they see the change without switching tabs.
      if (evt.type === 'operator_action_receipt') {
        const data = evt.data || {};
        const card = {
          role: 'operator_action_receipt',
          agent_id: data.agent_id,
          field: data.field,
          summary: data.summary,
          old_value: data.old_value,
          new_value: data.new_value,
          undo_token: data.undo_token,
          expires_at: data.expires_at,
          reason: data.reason,
          ts: Date.now() / 1000,
        };
        if (!this.chatHistories['operator']) this.chatHistories['operator'] = [];
        this.chatHistories['operator'].push(card);
        if (data.agent_id && data.agent_id !== 'operator') {
          if (!this.chatHistories[data.agent_id]) this.chatHistories[data.agent_id] = [];
          this.chatHistories[data.agent_id].push({ ...card });
        }
        // Refresh the open agent detail panel so the config card flips
        // to the new value live. Soft edits no longer fire
        // ``agent_config_updated`` (gated to hard fields per the audit
        // follow-up); this is the receipt-side equivalent of the
        // agent_config_updated handler below.
        const viewing = this.selectedAgent || this.detailAgent;
        if (
          data.agent_id && viewing === data.agent_id
          && typeof this.fetchAgentDetail === 'function'
        ) {
          if (this._agentDetailsDebounce) clearTimeout(this._agentDetailsDebounce);
          this._agentDetailsDebounce = setTimeout(
            () => this.fetchAgentDetail(viewing), 250,
          );
        }
      }
      if (evt.type === 'operator_action_receipt_undone') {
        const data = evt.data || {};
        const token = data.undo_token;
        if (token) {
          for (const aid of Object.keys(this.chatHistories || {})) {
            for (const m of this.chatHistories[aid] || []) {
              if (m.role === 'operator_action_receipt' && m.undo_token === token) {
                m._undone = true;
              }
            }
          }
        }
        // Refresh the agent detail panel so the reverted value lands
        // live (the undo apply path goes through `_apply_pending_change`
        // which no longer fires `agent_config_updated` for soft fields).
        const viewing = this.selectedAgent || this.detailAgent;
        if (
          data.agent_id && viewing === data.agent_id
          && typeof this.fetchAgentDetail === 'function'
        ) {
          if (this._agentDetailsDebounce) clearTimeout(this._agentDetailsDebounce);
          this._agentDetailsDebounce = setTimeout(
            () => this.fetchAgentDetail(viewing), 250,
          );
        }
      }
      if (evt.type === 'operator_action_receipt_superseded') {
        // Mark prior receipt(s) on the same agent_id+field that pre-date
        // a newer edit. The older receipt's [Undo] still works, but
        // doing so would erase the intervening edit(s) — the card
        // shows a "superseded by newer edits" warning so the operator
        // is aware before clicking.
        const data = evt.data || {};
        const token = data.undo_token;
        if (token) {
          for (const aid of Object.keys(this.chatHistories || {})) {
            for (const m of this.chatHistories[aid] || []) {
              if (m.role === 'operator_action_receipt' && m.undo_token === token) {
                m._superseded = true;
                m._supersededByCount = (m._supersededByCount || 0)
                  + (data.superseded_by_count || 1);
              }
            }
          }
        }
      }

      // Live notification bell — emitted right after the
      // ``_notifications_producer`` writes a row to the persistent
      // notifications store. We optimistically prepend the row to
      // the in-memory ``notifications`` list so the bell badge ticks
      // up live; the existing 60s poll still acts as a safety net
      // for any in-flight events that landed before the WS
      // subscription resumed after a reconnect.
      if (evt.type === 'notification_added') {
        const data = evt.data || {};
        const nid = typeof data.id === 'number' ? data.id : 0;
        // Skip dupes (the 60s poll may race with the live event).
        const exists = nid > 0 && (this.notifications || []).some(n => n.id === nid);
        if (!exists) {
          const row = {
            id: nid,
            kind: data.kind || 'info',
            title: data.title || '',
            body: data.body || '',
            agent_id: data.agent_id || null,
            read_at: null,
            ts: evt.timestamp || (Date.now() / 1000),
            payload: data.payload || {},
          };
          this.notifications = [row, ...(this.notifications || [])];
          this.notificationsUnreadCount = (this.notificationsUnreadCount || 0) + 1;
          // Keep ``_lastNotifiedId`` in sync so the next poll skips
          // browser-notification replay for this row.
          if (nid > (this._lastNotifiedId || 0)) {
            this._lastNotifiedId = nid;
          }
          // Best-effort browser notification hook (gated by user opt-in).
          try { this._maybeFireBrowserNotification && this._maybeFireBrowserNotification(row); } catch (_) {}
        }
      }

      // Live agent archive / unarchive — refresh the relevant list
      // so the SPA reflects the new state without a full reload.
      if (evt.type === 'agent_archived' || evt.type === 'agent_unarchived') {
        if (this._fleetRefreshDebounce) clearTimeout(this._fleetRefreshDebounce);
        this._fleetRefreshDebounce = setTimeout(() => {
          if (typeof this.fetchAgents === 'function') this.fetchAgents();
        }, 250);
      }

      // Live agent restart — pulse while the container is bouncing
      // and clear once the new container reports ready. Failures
      // surface via the ``agent_state`` ``restart_failed`` payload
      // which clears the pulse and exposes ``error`` for the toast.
      if (evt.type === 'agent_restarting' && evt.agent) {
        if (!this.agentRestartingMap) this.agentRestartingMap = {};
        this.agentRestartingMap = { ...this.agentRestartingMap, [evt.agent]: true };
      }
      if (evt.type === 'agent_restarted' && evt.agent) {
        if (!this.agentRestartingMap) this.agentRestartingMap = {};
        const next = { ...this.agentRestartingMap };
        delete next[evt.agent];
        this.agentRestartingMap = next;
        // Re-fetch the agent's runtime details when the user is
        // currently viewing it.
        const viewing = this.selectedAgent || this.detailAgent;
        if (viewing === evt.agent && typeof this.fetchAgentDetail === 'function') {
          this.fetchAgentDetail(viewing);
        }
      }
      if (evt.type === 'agent_state' && evt.data?.state === 'restart_failed' && evt.agent) {
        if (this.agentRestartingMap) {
          const next = { ...this.agentRestartingMap };
          delete next[evt.agent];
          this.agentRestartingMap = next;
        }
        // Surface the reason. Comment above promises the SPA exposes
        // ``error`` for the toast — wire that through here so the user
        // sees WHY the restart failed instead of just the silent clear.
        if (typeof this.showToast === 'function') {
          const err = evt.data?.error || 'Unknown error';
          const who = (typeof this.agentDisplayName === 'function')
            ? this.agentDisplayName(evt.agent) : evt.agent;
          this.showToast(`Restart failed for ${who}: ${err}`);
        }
      }

      // Live agent config update — flip the agent config card to the
      // new value without a manual refresh. Soft fields layer their
      // own ``operator_action_receipt`` cards on top; hard fields
      // depend solely on this event because they don't get a
      // revertible receipt.
      if (evt.type === 'agent_config_updated' && evt.agent) {
        const viewing = this.selectedAgent || this.detailAgent;
        if (viewing === evt.agent && typeof this.fetchAgentDetail === 'function') {
          if (this._agentDetailsDebounce) clearTimeout(this._agentDetailsDebounce);
          this._agentDetailsDebounce = setTimeout(
            () => this.fetchAgentDetail(viewing), 250,
          );
        }
      }

      // Live team CRUD — refresh the teams list. We use the existing
      // ``fetchTeams`` rather than mutating in place so the teams view
      // picks up the latest server-side ordering / member counts in
      // one round-trip.
      if (evt.type === 'team_created' || evt.type === 'team_deleted' ||
          evt.type === 'team_updated' || evt.type === 'team_archived' ||
          evt.type === 'team_unarchived') {
        if (typeof this.fetchTeams === 'function') {
          if (this._teamsRefreshDebounce) clearTimeout(this._teamsRefreshDebounce);
          this._teamsRefreshDebounce = setTimeout(() => this.fetchTeams(), 250);
        }
        // Brief (TEAM.md) edited elsewhere — live-reload the content for the
        // active team unless the user is mid-edit (don't clobber their buffer).
        // Two producers: the dashboard's own TEAM.md save emits
        // field 'project_md' (dashboard/server.py), while the operator
        // agent's update_team_context mesh endpoint emits field 'context'
        // (host/server.py).
        const d = evt.data || {};
        const changedTeam = d.team_name || d.name || d.team_id || d.project_id;
        if ((d.field === 'project_md' || d.field === 'context') &&
            changedTeam === this.activeTeam &&
            !this.teamEditing && typeof this.fetchTeamContent === 'function') {
          this.fetchTeamContent();
        }
      }

      // Lane queue depth/busy changed — debounced refetch of /api/queues
      // (replaces the old 2s poll). The endpoint merges with the agent
      // registry so idle agents keep their zero rows; chat-recovery reads
      // the same ``queueStatus`` this keeps fresh.
      if (evt.type === 'queue_changed') {
        if (this._queueRefreshDebounce) clearTimeout(this._queueRefreshDebounce);
        this._queueRefreshDebounce = setTimeout(() => this.fetchQueues(), 300);
      }

      // System-tab config changed elsewhere — re-fetch just the affected
      // panel, and only while the System tab is on screen (off-screen panels
      // re-sync on their next open). ``data.scope`` maps to the panel's
      // read-only list loader; unknown scopes are ignored.
      //
      // NOTE: only the display-list panels are auto-refetched. The edit-FORM
      // panels (browser_settings / system_settings / captcha_solver /
      // network_proxy / wallet) are deliberately excluded — their loaders
      // overwrite the same state the form inputs bind to (e.g.
      // loadNetworkProxy resets networkProxy.form), so a background reload
      // would clobber an operator's in-progress edit. Those panels already
      // refresh on tab-open, which is sufficient for rarely-changed,
      // single-operator settings.
      if (evt.type === 'config_changed' && this.activeTab === 'system') {
        const _configLoaders = {
          channels: ['fetchChannels'],
          webhooks: ['fetchWebhooks'],
          api_keys: ['fetchApiKeys'],
          integrations: ['loadIntegrations'],
          // Safe to background-refresh: the connector add/edit form
          // binds to connectorDraft, not connectorsData.
          connectors: ['loadConnectors'],
          storage: ['fetchStorage', 'fetchDatabaseDetails'],
          uploads: ['fetchUploads'],
          skills: ['loadSkillsCatalog'],
        };
        for (const fn of (_configLoaders[evt.data?.scope] || [])) {
          if (typeof this[fn] === 'function') this[fn]();
        }
      }

      // Credential stored — flip the matching credential_request
      // card (in either the agent's chat or the operator chat) to
      // ``saved=true`` so the user can dismiss it. Match by
      // ``request_id`` when present (new flow), fall back to name.
      if (evt.type === 'credential_stored') {
        const data = evt.data || {};
        const reqId = data.request_id || '';
        const name = data.name || data.service || '';
        const fromAgent = data.agent_id || evt.agent || '';
        for (const chatId of Object.keys(this.chatHistories || {})) {
          const hist = this.chatHistories[chatId] || [];
          for (const m of hist) {
            if (m.role !== 'credential_request') continue;
            if (chatId !== fromAgent && fromAgent && m._from_agent !== fromAgent) continue;
            const matchById = reqId && m.request_id === reqId;
            const matchByName = !reqId && name && m.name === name;
            if (matchById || matchByName) {
              m.saved = true;
              m.cancelled = false;
            }
          }
        }
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
        // Platform-success rollup is driven off this same event (replaces the
        // old 30s poll). The server-side aggregate is updated synchronously by
        // this emit, so a refetch here is fresh. Only the captcha/fingerprint/
        // pre-nav sub-types change the rollup, and only refetch when the panel
        // is on screen.
        const _psKinds = ['captcha_gate', 'fingerprint_event', 'fingerprint_burn', 'platform_pre_nav_delay'];
        if (_psKinds.includes(evt.data.type) &&
            this.activeTab === 'system' && this.systemTab === 'browser') {
          if (this._platformSuccessDebounce) clearTimeout(this._platformSuccessDebounce);
          this._platformSuccessDebounce = setTimeout(() => this.fetchPlatformSuccess(), 500);
        }
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
          // Toast dedup: a fleet-wide fingerprint regression (e.g. a
          // Camoufox version bump that changes navigator/platform shape)
          // would otherwise stack one 8s toast per agent on a mass restart.
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
        if (this.activeTeam && this.activeTab === 'fleet' && !this.detailAgent) {
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

      // Mirror of blackboard_write for the delete endpoint so the
      // SPA can drop the entry from the viewer / refresh comms.
      if (evt.type === 'blackboard_delete' && evt.data && evt.data.key) {
        if (this.activeTeam && this.activeTab === 'fleet' && !this.detailAgent) {
          if (this._commsDebounce) clearTimeout(this._commsDebounce);
          this._commsDebounce = setTimeout(() => {
            this.fetchBlackboard();
            this.fetchCommsActivity();
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
          cancelled: false,
          request_id: evt.data.request_id || '',
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

      // Sync credential_request card across agent chat + operator chat
      // when the user cancels (PR 3). Match by request_id when present
      // (new flow); fall back to name (legacy messages without an id).
      if (evt.type === 'credential_request_cancelled' && agent && evt.data) {
        const reqId = evt.data.request_id || '';
        const name = evt.data.name || '';
        for (const chatId of [agent, 'operator']) {
          const hist = this.chatHistories[chatId];
          if (!hist) continue;
          for (const m of hist) {
            if (m.role !== 'credential_request') continue;
            if (chatId !== agent && m._from_agent !== agent) continue;
            const matchById = reqId && m.request_id === reqId;
            const matchByName = !reqId && name && m.name === name;
            if (matchById || matchByName) {
              m.cancelled = true;
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
          request_id: evt.data.request_id || '',
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
            if (m.role === 'browser_login_request'
                && (evt.data.request_id
                    ? m.request_id === evt.data.request_id
                    : (m.service === evt.data.service && (chatId === agent || m._from_agent === agent)))) {
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
            if (m.role === 'browser_login_request'
                && (evt.data.request_id
                    ? m.request_id === evt.data.request_id
                    : (m.service === evt.data.service && (chatId === agent || m._from_agent === agent)))) {
              m.cancelled = true;
            }
          }
        }
      }

      // Surface browser CAPTCHA help requests as interactive VNC cards in chat.
      // Mirrors the browser_login_request handler — Phase 8 §11.14.
      if (evt.type === 'browser_captcha_help_request' && agent && evt.data && evt.data.service) {
        const evtTs = this._normalizeEventTs(evt);
        const captchaCard = {
          role: 'browser_captcha_help_request',
          content: evt.data.description || '',
          service: evt.data.service || '',
          url: evt.data.url || '',
          completed: false,
          cancelled: false,
          request_id: evt.data.request_id || '',
          ts: evtTs,
        };
        // Show in the requesting agent's chat
        if (!this.chatHistories[agent]) this.chatHistories[agent] = [];
        const isDup = this.chatHistories[agent].some(m =>
          m.role === 'browser_captcha_help_request' && m.service === evt.data.service && Math.abs((m.ts || 0) - evtTs) < 5000
        );
        if (!isDup) {
          this.chatHistories[agent].push(captchaCard);
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
            m.role === 'browser_captcha_help_request' && m._from_agent === agent && m.service === evt.data.service && Math.abs((m.ts || 0) - evtTs) < 5000
          );
          if (!opDup) {
            this.chatHistories['operator'].push({ ...captchaCard, _from_agent: agent });
            if (this.activeTab === 'chat') {
              this.$nextTick(() => this._scrollChat('operator'));
            }
          }
        }
      }

      // Sync browser CAPTCHA help card state across all copies (agent chat + operator chat).
      if (evt.type === 'browser_captcha_help_completed' && agent && evt.data?.service) {
        for (const chatId of [agent, 'operator']) {
          const hist = this.chatHistories[chatId];
          if (!hist) continue;
          for (const m of hist) {
            if (m.role === 'browser_captcha_help_request'
                && (evt.data.request_id
                    ? m.request_id === evt.data.request_id
                    : (m.service === evt.data.service && (chatId === agent || m._from_agent === agent)))) {
              m.completed = true;
            }
          }
        }
      }
      if (evt.type === 'browser_captcha_help_cancelled' && agent && evt.data?.service) {
        for (const chatId of [agent, 'operator']) {
          const hist = this.chatHistories[chatId];
          if (!hist) continue;
          for (const m of hist) {
            if (m.role === 'browser_captcha_help_request'
                && (evt.data.request_id
                    ? m.request_id === evt.data.request_id
                    : (m.service === evt.data.service && (chatId === agent || m._from_agent === agent)))) {
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
          // Prune restored chat tabs for agents that no longer exist.
          // Operator is a synthetic system chat — it is never present in
          // ``/api/agents`` (special-cased server-side, outside
          // ``agent_registry``), so it is pinned and excluded from
          // staleness here. A background fleet refresh must not evict
          // Operator from the messenger.
          const agentIds = new Set(this.agents.map(a => a.id));
          const stale = this.openChats.filter(id => id !== 'operator' && !agentIds.has(id));
          if (stale.length) {
            this.openChats = this.openChats.filter(id => id === 'operator' || agentIds.has(id));
            if (this.activeChatId && this.activeChatId !== 'operator' && !agentIds.has(this.activeChatId)) {
              this.activeChatId = this.openChats[0] || null;
            }
            this._saveChatToSession();
          }
          // Fetch coordination status from blackboard
          this._fetchCoordination();
          // Update operator readiness for the Chat tab
          this.checkOperatorReady();
          // Phase -1 wizard — first-visit detection runs after every
          // ``fetchAgents`` resolve so a freshly-spawned agent (still
          // loading at init time) doesn't pop the wizard. Re-entrant
          // calls during the same visit are no-ops.
          this._maybeStartWizard();
        }
      } catch (e) {
        console.warn('fetchAgents failed:', e);
        this.connectionError = true;
      }
      this.loading = false;
    },

    async _fetchCoordination() {
      // Only fetch when we have a project with agents
      const proj = this.activeTeam;
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
        // Fire-and-forget capabilities fetch so the MCP Servers panel's
        // status dots populate without blocking the Config tab render.
        // (The Capabilities tab path already runs this awaited; here we
        // just need it to land so ``agentMcpStatus`` reflects per-server
        // state.) Errors are logged inside fetchAgentCapabilities.
        this.fetchAgentCapabilities(agentId);
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
      if (tab.id === 'skills') {
        await this.loadAgentSkills(agentId);
      }
      if (tab.id === 'files') {
        await this.fetchAgentFiles(agentId, '.');
      }
    },

    async fetchAgentCapabilities(agentId) {
      this.agentCapabilities = null;
      this.agentMcpStatus = [];
      this.agentMcpToolMap = {};
      // The agent Config tab's read-only Connectors panel needs the
      // fleet catalog; lazy-load it alongside capabilities.
      if (this.connectorsData === null) this.loadConnectors();
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/capabilities`);
        if (resp.ok) {
          const data = await resp.json();
          // Guard against an in-flight fetch landing after the user
          // switched to a different agent (or closed the detail panel
          // entirely). Without this, agent A's response would overwrite
          // the now-visible agent B's tools/status — or worse, write
          // stale data into agentMcpStatus while no agent is selected,
          // which then bleeds into the next agent the user opens.
          if (this.selectedAgent !== agentId) return;
          // Agent returns tool_definitions (OpenAI format: {type, function: {name, description}})
          const defs = data.tool_definitions || [];
          const sources = data.tool_sources || {};
          this.agentCapabilities = defs.map(t => ({
            name: t.function?.name || t.name || '?',
            description: t.function?.description || t.description || '',
            source: sources[t.function?.name || t.name] || 'custom',
          }));
          // PR 1 side-channels: per-server startup status registry +
          // tool→server mapping. Both are omitted by agents that
          // haven't been restarted into PR 1 code; the empty defaults
          // collapse the MCP panel to the "pending" state in that case.
          this.agentMcpStatus = Array.isArray(data.mcp_servers) ? data.mcp_servers : [];
          this.agentMcpToolMap = (data.mcp_tool_to_server && typeof data.mcp_tool_to_server === 'object')
            ? data.mcp_tool_to_server : {};
        }
      } catch (e) { console.warn('fetchAgentCapabilities failed:', e); }
    },

    // ── Skills: per-agent assignment (identity Skills tab) ──────────────
    async loadAgentSkills(agentId) {
      this.agentSkills = null;
      this.agentSkillsLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/skills?agent_id=${encodeURIComponent(agentId)}`);
        if (resp.ok) {
          const data = await resp.json();
          // Guard against a stale fetch landing after the user switched
          // agents (mirrors fetchAgentCapabilities).
          if (this.selectedAgent !== agentId) return;
          this.agentSkills = Array.isArray(data.skills) ? data.skills : [];
        } else {
          this.agentSkills = [];
        }
      } catch (e) {
        console.warn('loadAgentSkills failed:', e);
        this.agentSkills = [];
      } finally {
        this.agentSkillsLoading = false;
      }
    },

    async toggleAgentSkill(agentId, skill) {
      // Fleet-assigned skills are always-on; the toggle is locked.
      if (skill.fleet_assigned || this.agentSkillsSaving) return;
      const current = (this.agentSkills || [])
        .filter(s => s.agent_assigned)
        .map(s => s.name);
      const set = new Set(current);
      if (skill.agent_assigned) set.delete(skill.name);
      else set.add(skill.name);
      this.agentSkillsSaving = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${encodeURIComponent(agentId)}/permissions`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ allowed_skills: Array.from(set) }),
        });
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          this.showToast(`Skill update failed: ${data.detail || data.error || resp.status}`);
        }
      } catch (e) {
        this.showToast(`Skill update failed: ${e.message || e}`);
      } finally {
        this.agentSkillsSaving = false;
        // Re-fetch to reflect server truth (fleet ∪ per-agent).
        await this.loadAgentSkills(agentId);
      }
    },

    // ── Skills: fleet catalog (System → Skills) ─────────────────────────
    async loadSkillsCatalog() {
      this.fleetSkillsCatalog = null;
      this.fleetSkillsCatalogLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/skills`);
        if (resp.ok) {
          const data = await resp.json();
          this.fleetSkillsCatalog = Array.isArray(data.skills) ? data.skills : [];
        } else {
          this.fleetSkillsCatalog = [];
        }
      } catch (e) {
        console.warn('loadSkillsCatalog failed:', e);
        this.fleetSkillsCatalog = [];
      } finally {
        this.fleetSkillsCatalogLoading = false;
      }
    },

    async toggleFleetSkill(skill) {
      if (this.fleetSkillsSaving) return;
      const current = (this.fleetSkillsCatalog || [])
        .filter(s => s.fleet_assigned)
        .map(s => s.name);
      const set = new Set(current);
      if (skill.fleet_assigned) set.delete(skill.name);
      else set.add(skill.name);
      this.fleetSkillsSaving = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/fleet/skills`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ skills: Array.from(set) }),
        });
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          this.showToast(`Fleet skill update failed: ${data.detail || data.error || resp.status}`);
        }
      } catch (e) {
        this.showToast(`Fleet skill update failed: ${e.message || e}`);
      } finally {
        this.fleetSkillsSaving = false;
        await this.loadSkillsCatalog();
      }
    },

    async installSkill() {
      const repo = (this.skillInstallRepo || '').trim();
      if (!repo || this.skillInstalling) return;
      const ref = (this.skillInstallRef || '').trim();
      this.skillInstalling = true;
      try {
        const body = { repo_url: repo };
        if (ref) body.ref = ref;
        const resp = await fetch(`${window.__config.apiBase}/skills/install`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await resp.json().catch(() => ({}));
        if (resp.ok && !data.error) {
          this.showToast(`Installed skill: ${data.name || repo}`);
          this.skillInstallRepo = '';
          this.skillInstallRef = '';
          await this.loadSkillsCatalog();
        } else {
          this.showToast(`Install failed: ${data.error || data.detail || resp.status}`);
        }
      } catch (e) {
        this.showToast(`Install failed: ${e.message || e}`);
      } finally {
        this.skillInstalling = false;
      }
    },

    async removeSkill(name) {
      if (!name || this.skillRemoving) return;
      this.showConfirm('Remove skill?', `This removes the installed pack "${name}" from the fleet catalog.`, async () => {
        this.skillRemoving = name;
        try {
          const resp = await fetch(`${window.__config.apiBase}/skills/remove`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
          });
          if (resp.ok) {
            this.showToast(`Removed skill: ${name}`);
            await this.loadSkillsCatalog();
          } else {
            const data = await resp.json().catch(() => ({}));
            this.showToast(`Remove failed: ${data.error || data.detail || resp.status}`);
          }
        } catch (e) {
          this.showToast(`Remove failed: ${e.message || e}`);
        } finally {
          this.skillRemoving = '';
        }
      }, true);
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
      // Navigate to the streaming attachment endpoint so the browser saves the
      // FULL file to disk (the JSON /files route is capped at 500 KB and would
      // silently truncate). Same-origin navigation carries the session cookie;
      // Content-Disposition makes it a download, not a page load.
      const url = `${window.__config.apiBase}/agents/${agentId}/file-download/${this._encodeFilePath(path)}`;
      const a = document.createElement('a');
      a.href = url;
      a.download = path.split('/').pop();
      document.body.appendChild(a);
      a.click();
      a.remove();
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
      // For the operator agent, SOUL.md / INSTRUCTIONS.md edits literally
      // rewrite how the operator behaves on its next turn. Route through
      // the shared confirm modal so the user sees a clear "are you sure"
      // gate; everything else (other agents, MEMORY.md/USER.md/HEARTBEAT.md
      // on the operator) saves immediately as before.
      const isOperatorIdentity = agentId === 'operator'
        && (file === 'SOUL.md' || file === 'INSTRUCTIONS.md');
      if (isOperatorIdentity && !this._operatorIdentityConfirmed) {
        this.showConfirm(
          'This changes how your Operator behaves',
          `Saving ${file} on the operator agent will reshape its next response. `
            + 'Are you sure you want to apply this change?',
          async () => {
            this._operatorIdentityConfirmed = true;
            try {
              await this.saveIdentityFile(agentId, file);
            } finally {
              this._operatorIdentityConfirmed = false;
            }
          },
          false,  // not destructive — indigo button, "Confirm"
        );
        return;
      }
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

    async fetchTeamContent() {
      if (!this.activeTeam) {
        this.teamContent = '';
        this.teamExists = false;
        this.teamLoading = false;
        return;
      }
      this.teamLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/team?team=${encodeURIComponent(this.activeTeam)}`);
        if (resp.ok) {
          const data = await resp.json();
          this.teamContent = data.content || '';
          this.teamExists = data.exists;
        }
      } catch (e) { console.warn('fetchTeamContent failed:', e); }
      this.teamLoading = false;
    },

    async saveTeamContent() {
      if (this.teamSaving) return;
      if (!this.activeTeam) return;
      this.teamSaving = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/team?team=${encodeURIComponent(this.activeTeam)}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: this.teamEditBuffer }),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.teamContent = this.teamEditBuffer;
          this.teamExists = true;
          this.teamEditing = false;
          this.teamEditBuffer = '';
          const pushed = Object.values(data.pushed || {}).filter(Boolean).length;
          const total = Object.keys(data.pushed || {}).length;
          this.showToast(`${this.activeTeam} TEAM.md saved${total > 0 ? ` (pushed to ${pushed}/${total} agents)` : ''}`);
        } else {
          try {
            const err = await resp.json();
            this.showToast(`Save failed: ${err.detail || 'Unknown error'}`);
          } catch (_) { this.showToast('Save failed'); }
        }
      } catch (e) { this.showToast(`Save failed: ${e.message}`); }
      this.teamSaving = false;
    },

    startTeamEdit() {
      this.teamEditBuffer = this.teamContent;
      this.teamEditing = true;
    },

    cancelTeamEdit() {
      this.teamEditing = false;
      this.teamEditBuffer = '';
    },

    async fetchTeams() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/teams`);
        if (resp.ok) {
          const data = await resp.json();
          this.teams = data.teams || [];
          this.teamsLoaded = true;
        }
      } catch (e) { console.warn('fetchTeams failed:', e); }
    },

    switchTeam(name) {
      if (this.activeTeam === name) return;
      this.activeTeam = name;
      try {
        if (name) localStorage.setItem('activeTeam', name);
        else localStorage.removeItem('activeTeam');
      } catch (_) { /* ignore */ }
      this.teamEditing = false;
      this.teamEditBuffer = '';
      this.teamBannerExpanded = false;
      this.teamHubExpanded = true;
      this.teamHubTab = 'work';
      this.showTeamForm = false;
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
      this.fetchTeamContent();
      if (name) {
        // Work leads — ensure the task ledger is loaded (WS keeps it live).
        // State/Activity (Advanced) and Files lazy-load on their own tabs, so
        // switching teams no longer eagerly fans out blackboard + artifact reads.
        this._ensureWorkplaceTasks();
      }
    },

    // Load the workplace task ledger once per session; the WS reducer
    // (handleWorkplaceEvent) + the debounced reload keep it current after that,
    // so the team-hub Work view derives from already-live shared state rather
    // than its own fetch loop.
    _ensureWorkplaceTasks() {
      if (this.workplaceEnabled === false) return;
      if (this.workplaceSectionLoading && this.workplaceSectionLoading.tasks) return;  // already in flight
      // Skip only if we loaded successfully once. If the last load errored,
      // workplaceErrors.tasks is set — retry on the next entry rather than
      // wedging the Work view empty with no recovery path.
      if (this._teamWorkLoaded && !(this.workplaceErrors && this.workplaceErrors.tasks)) return;
      if (typeof this.loadWorkplaceTasks !== 'function') return;
      this._teamWorkLoaded = true;
      this.loadWorkplaceTasks();
    },

    // Read-only PROGRESS for the active team's hub Work view: in-flight +
    // delivered-today only. Derives from workplaceTasks (shared with the Work
    // tab) — no extra fetch. Blocked / failed individual tasks are deliberately
    // NOT surfaced here: the user's unit is the team, not the task. The operator
    // handles blockers/failures and narrates them in the team's Work summary;
    // the only user-actionable items (credential / browser-login approvals)
    // surface in the Work tab's "Needs you" panel.
    get teamWork() {
      const team = this.activeTeam;
      const active = { pending: 1, accepted: 1, working: 1 };
      const midnight = new Date(); midnight.setHours(0, 0, 0, 0);
      const todayMs = midnight.getTime();
      const inflight = [], doneToday = [];
      for (const t of (this.workplaceTasks || [])) {
        if (t.project_id !== team) continue;
        if (active[t.status]) inflight.push(t);
        else if (t.status === 'done' && ((t.completed_at || 0) * 1000) >= todayMs) doneToday.push(t);
      }
      return { inflight, doneToday, total: inflight.length + doneToday.length };
    },

    // One-hop handoff label for a task row: "creator → assignee" (or just the
    // assignee when it wasn't handed off). Plain text — bind with x-text.
    teamHandoffLabel(t) {
      const who = this.agentDisplayName(t.assignee);
      if (t && t.creator && t.creator !== t.assignee) {
        return `${this.agentDisplayName(t.creator)} → ${who}`;
      }
      return who;
    },

    openTeamModal() {
      if (this.atTeamLimit) return;
      this.showTeamForm = true;
      this.$nextTick(() => {
        const el = document.getElementById('team-name-input');
        if (el) el.focus();
      });
    },

    closeTeamModal() {
      if (this.teamFormLoading) return;
      this.showTeamForm = false;
      this.newTeamName = '';
      this.newTeamDesc = '';
    },

    async createTeam() {
      if (!this.teamsEnabled) {
        this.showToast('Teams are not available on your current plan.');
        return;
      }
      if (this.atTeamLimit) {
        this.showToast('Team limit reached. Upgrade your plan for more teams.');
        return;
      }
      const name = this.newTeamName.trim();
      if (!name) return;
      if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(name)) {
        this.showToast('Team name must start with a letter or number and contain only letters, numbers, hyphens, underscores');
        return;
      }
      this.teamFormLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/teams`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, description: this.newTeamDesc.trim(), members: [] }),
        });
        if (resp.ok) {
          this.teamFormLoading = false;
          this.closeTeamModal();
          await this.fetchTeams();
          this.switchTeam(name);
          this.showToast(`Team "${name}" created`);
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
      this.teamFormLoading = false;
    },

    async deleteTeam(name) {
      this.showConfirm('Delete Team', `Delete team "${name}"? Members will become solo.`, async () => {
        try {
          const resp = await fetch(`${window.__config.apiBase}/teams/${encodeURIComponent(name)}`, {
            method: 'DELETE',
          });
          if (resp.ok) {
            this.teamEditing = false;
            this.teamEditBuffer = '';
            await this.fetchTeams();
            if (this.activeTeam === name) this.switchTeam(null);
            this.showToast(`Team "${name}" deleted`);
          } else {
            const err = await resp.json().catch(() => ({}));
            this.showToast(`Delete failed: ${err.detail || 'Unknown error'}`);
          }
        } catch (e) { this.showToast(`Delete failed: ${e.message}`); }
      }, true);
    },

    async addMember(team, agent) {
      try {
        const resp = await fetch(`${window.__config.apiBase}/teams/${encodeURIComponent(team)}/members`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ agent }),
        });
        if (resp.ok) {
          await this.fetchTeams();
          this.fetchAgents();
          this.showToast(`${agent} added to ${team}`);
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Add failed: ${err.detail || 'Unknown error'}`);
        }
      } catch (e) { this.showToast(`Add failed: ${e.message}`); }
    },

    async removeMember(team, agent) {
      try {
        const resp = await fetch(`${window.__config.apiBase}/teams/${encodeURIComponent(team)}/members/${encodeURIComponent(agent)}`, {
          method: 'DELETE',
        });
        if (resp.ok) {
          await this.fetchTeams();
          this.fetchAgents();
          this.showToast(`${agent} removed from ${team}`);
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Remove failed: ${err.detail || 'Unknown error'}`);
        }
      } catch (e) { this.showToast(`Remove failed: ${e.message}`); }
    },

    /** Agents in the registry that are not in any team. Excludes operator (system agent, not team-assignable). */
    get soloAgents() {
      const assigned = new Set();
      for (const t of this.teams) {
        for (const m of (t.members || [])) assigned.add(m);
      }
      return this.agents.filter(a => a.id !== 'operator' && !assigned.has(a.id));
    },

    /** Get the active team object. */
    get activeTeamData() {
      return this.teams.find(t => t.name === this.activeTeam) || null;
    },


    _bbTeamPrefix() {
      // Scope blackboard keys to the active team. The on-disk key
      // namespace is still ``projects/`` — that's a backend storage
      // prefix, not a domain term, and renaming it is a separate
      // migration outside the project→team rename's scope.
      return this.activeTeam ? `projects/${this.activeTeam}/` : '';
    },

    _bbStripTeamPrefix(key) {
      const pfx = this._bbTeamPrefix();
      if (pfx && key.startsWith(pfx)) return key.slice(pfx.length);
      // If no active team, still strip any projects/*/  prefix for display.
      const m = key.match(/^projects\/[^/]+\/(.*)/);
      return m ? m[1] : key;
    },

    async fetchBlackboard() {
      this.bbLoading = true;
      try {
        // Prepend team prefix so namespace filters match team-scoped keys
        const searchPrefix = this._bbTeamPrefix() + this.bbPrefix;
        const resp = await fetch(`${window.__config.apiBase}/blackboard?prefix=${encodeURIComponent(searchPrefix)}`);
        if (resp.ok) {
          const entries = (await resp.json()).entries;
          // Strip team prefix so display/namespace logic works on the real key
          for (const e of entries) e.key = this._bbStripTeamPrefix(e.key);
          this.bbEntries = entries;
        }
      } catch (e) { console.warn('fetchBlackboard failed:', e); }
      this.bbLoading = false;
    },

    async fetchCommsActivity() {
      this.commsActivityLoading = true;
      try {
        const team = this.activeTeam;
        const params = new URLSearchParams({ limit: '100' });
        if (team) params.set('project', team);
        const resp = await fetch(`${window.__config.apiBase}/comms/activity?${params}`);
        if (resp.ok) {
          const data = await resp.json();
          const activity = data.activity || [];
          // Strip team prefix from blackboard keys and pubsub topics
          for (const item of activity) {
            if (item.key) item.key = this._bbStripTeamPrefix(item.key);
            if (item.topic) item.topic = this._bbStripTeamPrefix(item.topic);
          }
          this.commsActivity = activity;
          // Strip team prefix from subscription topic names
          const rawSubs = data.subscriptions || {};
          const subs = {};
          const pfx = this._bbTeamPrefix();
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
        // Gather artifacts from all agents in the current team
        const team = this.activeTeam;
        if (!team) { this.artifactsList = []; this.artifactsLoading = false; return; }
        const teamAgents = this.agents.filter(a => a.project === team);
        const results = await Promise.allSettled(
          teamAgents.map(async (a) => {
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
      this.costsLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/costs?period=${this.costPeriod}`);
        if (resp.ok) {
          this.costData = await resp.json();
          this.$nextTick(() => { this.renderCostChart(); this.renderModelChart(); });
        }
      } catch (e) { console.warn('fetchCosts failed:', e); }
      this.costsLoading = false;
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
        max_output_tokens: cfg.max_output_tokens || '',
        max_tool_rounds: cfg.max_tool_rounds || '',
        llm_timeout_seconds: cfg.llm_timeout_seconds || '',
        can_use_browser: cfg.can_use_browser ?? false,
        can_use_internet: cfg.can_use_internet ?? false,
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

    // ── MCP Connectors (System → Connectors fleet catalog) ──────────────
    // One record = an MCP server definition + its agent assignment.
    // The catalog applies on agent restart; saves return the affected
    // agents and the UI prompts restart-now/later (never auto-restarts).

    async loadConnectors() {
      if (this.connectorsLoading) return;
      this.connectorsLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/connectors`);
        if (resp.ok) this.connectorsData = await resp.json();
      } catch (e) {
        console.error('Failed to load connectors:', e);
      } finally {
        this.connectorsLoading = false;
      }
    },

    // Connectors assigned to one agent — drives the read-only panel on
    // the agent's Config tab (status dots come from agentMcpStatus).
    // Uses the server-expanded assigned_agents so '*'-expansion
    // semantics live in exactly one place (the mesh).
    connectorsForAgent(agentId) {
      if (!agentId || !this.connectorsData) return [];
      return (this.connectorsData.connectors || []).filter(
        c => (c.assigned_agents || []).includes(agentId),
      );
    },

    connectorAssignmentLabel(c) {
      const a = c.agents || [];
      if (a.includes('*')) return 'All agents';
      if (!a.length) return 'No agents';
      return a.length === 1 ? a[0] : `${a.length} agents`;
    },

    connectorStartAdd() {
      // No form until the catalog GET has landed: the duplicate-name
      // guard in saveConnector diffs against connectorsData, and an
      // unknowing "create" over an existing name is a silent
      // overwrite that wipes its env (new drafts replace env).
      if (!this.connectorsData) return;
      this.connectorErrors = {};
      this.connectorGlobalError = null;
      const agentIds = [...(this.connectorsData?.agents || [])];
      this.connectorDraft = {
        _isNew: true,
        name: '',
        command: '',
        _argRows: [],            // [{value: string}]
        _envRows: [],            // [{key, type: 'cred'|'plain', value}]
        _replaceEnv: true,       // new connector: env is whatever you type
        _envKeys: [],
        _assignMode: 'all',      // 'all' | 'some'
        _agentIds: agentIds,     // checkbox rows (frozen at form open)
        _agentSel: Object.fromEntries(agentIds.map(a => [a, false])),
      };
    },

    connectorStartEdit(c) {
      this.connectorErrors = {};
      this.connectorGlobalError = null;
      const assignedAll = (c.agents || []).includes('*');
      const explicit = assignedAll ? [] : (c.agents || []);
      // Checkbox rows = running agents ∪ the connector's explicit
      // assignment. Without the union, an assigned-but-not-running
      // agent would have no checkbox and a Save would silently strip
      // its assignment.
      const agentIds = [...new Set([
        ...(this.connectorsData?.agents || []), ...explicit,
      ])].sort();
      this.connectorDraft = {
        _isNew: false,
        name: c.name,
        command: c.command || '',
        _argRows: (c.args || []).map(v => ({ value: v })),
        _envRows: [],
        // Editing an existing record defaults to PRESERVE — env values
        // are never returned by GET, so the user only replaces env when
        // they explicitly toggle (and must then re-supply ALL vars).
        _replaceEnv: false,
        _envKeys: Array.isArray(c.env_keys) ? [...c.env_keys] : [],
        _assignMode: assignedAll ? 'all' : 'some',
        _agentIds: agentIds,
        _agentSel: Object.fromEntries(
          agentIds.map(a => [a, !assignedAll && explicit.includes(a)]),
        ),
      };
    },

    connectorCancelDraft() {
      this.connectorDraft = null;
      this.connectorErrors = {};
      this.connectorGlobalError = null;
    },

    connectorAddArgRow() {
      if (this.connectorDraft) this.connectorDraft._argRows.push({ value: '' });
    },
    connectorRemoveArgRow(idx) {
      if (this.connectorDraft) this.connectorDraft._argRows.splice(idx, 1);
    },
    connectorAddEnvRow() {
      if (this.connectorDraft) this.connectorDraft._envRows.push({ key: '', type: 'cred', value: '' });
    },
    connectorRemoveEnvRow(idx) {
      if (this.connectorDraft) this.connectorDraft._envRows.splice(idx, 1);
    },
    connectorToggleReplaceEnv() {
      if (!this.connectorDraft) return;
      this.connectorDraft._replaceEnv = !this.connectorDraft._replaceEnv;
      // Rows are kept across an off→on cycle — the wire payload simply
      // omits env while _replaceEnv is false (backend preserves it).
    },

    async saveConnector() {
      const d = this.connectorDraft;
      if (!d || this.connectorSaving) return;
      const name = (d.name || '').trim();
      const command = (d.command || '').trim();
      if (!name || !command) {
        this.showToast('Name and command are required.');
        return;
      }
      // The PUT is an upsert — guard the CREATE path against silently
      // overwriting an existing connector of the same name.
      if (d._isNew) {
        const lowered = name.toLowerCase();
        const exists = (this.connectorsData?.connectors || []).some(
          c => (c.name || '').toLowerCase() === lowered,
        );
        if (exists) {
          this.connectorErrors = {
            name: `A connector named "${name}" already exists — edit it instead.`,
          };
          return;
        }
      }
      if (d._replaceEnv) {
        for (const row of (d._envRows || [])) {
          const k = (row.key || '').trim();
          if (!k) continue;
          if (row.type === 'cred' && !(row.value || '').trim()) {
            this.showToast(`Env "${k}" is set to Credential mode but no credential is selected.`);
            return;
          }
        }
      }
      const body = {
        command,
        args: d._argRows.map(r => (r.value || '')).filter(v => v !== ''),
        agents: d._assignMode === 'all'
          ? ['*']
          : Object.keys(d._agentSel || {}).filter(a => d._agentSel[a]).sort(),
      };
      if (d._replaceEnv) {
        const env = {};
        for (const row of (d._envRows || [])) {
          const k = (row.key || '').trim();
          if (!k) continue;
          env[k] = row.type === 'cred' ? `$CRED{${(row.value || '').trim()}}` : (row.value || '');
        }
        body.env = env;
      }
      this.connectorSaving = true;
      this.connectorErrors = {};
      this.connectorGlobalError = null;
      try {
        const resp = await fetch(`${window.__config.apiBase}/connectors/${encodeURIComponent(name)}`, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          const detail = data.detail;
          // Structured Pydantic errors → per-field inline display.
          if (detail && typeof detail === 'object' && Array.isArray(detail.errors)) {
            const perField = {};
            const topLevel = [];
            for (const e of detail.errors) {
              const loc = e.loc || [];
              if (loc.length) perField[String(loc[0])] = e.msg || 'Invalid';
              else topLevel.push(e.msg || 'Invalid');
            }
            this.connectorErrors = perField;
            this.connectorGlobalError = topLevel.join('; ') || null;
          } else {
            this.connectorGlobalError = typeof detail === 'string' ? detail : 'Save failed';
          }
          return; // keep the form open with inline errors
        }
        this.connectorCancelDraft();
        await this.loadConnectors();
        if ((data.affected_agents || []).length) {
          this.connectorRestartPrompt = { name, affected: data.affected_agents };
        } else {
          this.showToast(`Connector "${name}" saved.`);
        }
      } catch (e) {
        this.connectorGlobalError = e.message || String(e);
      } finally {
        this.connectorSaving = false;
      }
    },

    deleteConnector(name) {
      this.showConfirm(
        'Remove connector',
        `Remove "${name}"? Assigned agents keep its tools until their next restart.`,
        async () => {
          try {
            const resp = await fetch(`${window.__config.apiBase}/connectors/${encodeURIComponent(name)}`, {
              method: 'DELETE',
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) {
              this.showToast(`Error: ${typeof data.detail === 'string' ? data.detail : 'Remove failed'}`);
              return;
            }
            await this.loadConnectors();
            if ((data.affected_agents || []).length) {
              this.connectorRestartPrompt = { name, affected: data.affected_agents };
            } else {
              this.showToast(`Connector "${name}" removed.`);
            }
          } catch (e) {
            this.showToast(`Error: ${e.message || String(e)}`);
          }
        },
        true,
      );
    },

    async restartConnectorAgents(agents) {
      if (this.connectorRestarting || !(agents || []).length) return;
      this.connectorRestarting = true;
      try {
        // The batch runs server-side in the background (a fleet-wide
        // restart can take minutes per agent — far past fetch/proxy
        // timeouts). Progress arrives through the per-agent restart
        // events the SPA already renders; the server emits a final
        // config_changed(connectors) that refreshes pending state.
        const resp = await fetch(`${window.__config.apiBase}/agents/restart-batch`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ agents }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          this.showToast(`Error: ${typeof data.detail === 'string' ? data.detail : 'Restart failed'}`);
          return;
        }
        const started = data.started || [];
        const skipped = Object.keys(data.skipped || {});
        this.showToast(started.length
          ? `Restarting ${started.length} agent${started.length === 1 ? '' : 's'}…`
            + (skipped.length ? ` (skipped: ${skipped.join(', ')})` : '')
          : `Nothing to restart${skipped.length ? ` (skipped: ${skipped.join(', ')})` : ''}.`);
        this.connectorRestartPrompt = null;
        this.fetchAgents();
      } catch (e) {
        this.showToast(`Error: ${e.message || String(e)}`);
      } finally {
        this.connectorRestarting = false;
      }
    },

    mcpDetectJsRuntime(command) {
      if (!command) return false;
      return /^(npx|bunx|pnpm\s+dlx|yarn\s+dlx|node|npm|pnpm|yarn|bun)(\s|$)/.test(command);
    },

    mcpDetectSecretLike(value) {
      if (!value) return false;
      // Catches the most common prefixes that almost always indicate a secret.
      if (/^(sk-|sk_|ghp_|gho_|ghu_|ghs_|github_pat_|pat-|xoxb-|xoxp-|Bearer\s+)/.test(value)) {
        return true;
      }
      // High-entropy heuristic: long enough, mixed case, contains digits.
      if (value.length >= 32 && /[A-Z]/.test(value) && /[a-z]/.test(value) && /[0-9]/.test(value)) {
        return true;
      }
      return false;
    },

    mcpStatusFor(serverName) {
      const match = (this.agentMcpStatus || []).find(s => s.name === serverName);
      if (!match) return { state: 'pending', toolsCount: 0, error: null };
      return {
        state: match.state || 'pending',
        toolsCount: match.tools_count || 0,
        error: match.error || null,
      };
    },

    mcpStatusDotClass(state) {
      if (state === 'running') return 'bg-green-400/70';
      if (state === 'failed') return 'bg-red-400/70';
      return 'bg-gray-500/70';  // pending
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
      // Execution caps — only send when non-empty AND changed vs the
      // effective value the GET seeded into the form.
      const _caps = {
        max_output_tokens: cfg.max_output_tokens,
        max_tool_rounds: cfg.max_tool_rounds,
        llm_timeout_seconds: cfg.llm_timeout_seconds,
      };
      for (const key of Object.keys(_caps)) {
        const raw = this.editForm[key];
        if (raw === '' || raw === null || raw === undefined) continue;
        const val = parseInt(raw, 10);
        if (Number.isNaN(val)) continue;
        if (val !== _caps[key]) body[key] = val;
      }
      // Handle allowed_credentials + capability flags via the permissions endpoint
      const newCreds = (this.editForm.allowed_credentials || '').split(',').map(s => s.trim()).filter(Boolean);
      const oldCreds = cfg.allowed_credentials || [];
      const credsChanged = JSON.stringify(newCreds) !== JSON.stringify(oldCreds);
      const permBody = {};
      if (credsChanged) permBody.allowed_credentials = newCreds;
      for (const flag of ['can_use_browser', 'can_use_internet', 'can_spawn', 'can_manage_cron', 'can_use_wallet']) {
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
            const err = await resp.json().catch(() => ({}));
            const detail = err.detail;
            this.showToast(`Error: ${typeof detail === 'string' ? detail : 'Update failed'}`);
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
            // Capability changes apply live — the permissions endpoint
            // reloads the mesh matrix, so no restart is needed here.
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
            await this.fetchAgentCapabilities(agentId);
          } else {
            const err = await restartResp.json().catch(() => ({}));
            const msg = (typeof err.detail === 'string' && err.detail) ? err.detail
                      : `restart endpoint returned ${restartResp.status}`;
            this.showToast(`${agentId}: config saved, restart failed: ${msg}`);
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
        if (f.team) payload.team = f.team;
        if (f.template) payload.template = f.template;
        const resp = await fetch(`${window.__config.apiBase}/agents`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        if (resp.ok) {
          const data = await resp.json();
          const teamName = data.team || data.project;
          const teamNote = teamName ? ` in ${teamName}` : '';
          this.showToast(data.ready ? `${data.agent} added and ready${teamNote}` : `${data.agent} added (starting)${teamNote}`);
          this.addAgentMode = false;
          this.addAgentForm = { name: '', role: '', model: '', avatar: 1, color: null, team: '', template: '', _showPicker: false, _showColorPicker: false, _templateSearch: '', _templateDropdownOpen: false, _modelSearch: '', _modelDropdownOpen: false };
          this.fetchAgents();
          if (teamName) this.fetchTeams();
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
      // Pre-select the active team if one is open
      if (this.activeTeam) this.addAgentForm.team = this.activeTeam;
      this.fetchSettings();
      this.fetchAgentTemplates();
      this.fetchTeams();
      this.$nextTick(() => {
        const el = document.getElementById('add-agent-name-input');
        if (el) el.focus();
      });
    },

    closeAddAgentModal() {
      if (this.addAgentLoading) return;
      this.addAgentMode = false;
      this.addAgentForm = { name: '', role: '', model: '', avatar: 1, color: null, team: '', template: '', _showPicker: false, _showColorPicker: false, _templateSearch: '', _templateDropdownOpen: false, _modelSearch: '', _modelDropdownOpen: false };
    },

    openCookieImport() {
      if (!this.selectedAgent || this.selectedAgent === 'operator') return;
      if (!this.agentDetail || !this.agentDetail.vnc_url) {
        this.showToast('Browser is not available for this agent');
        return;
      }
      this.cookieImportOpen = true;
    },

    closeCookieImport() {
      // The modal body is wrapped in <template x-if> so this unmounts
      // the inner x-data scope — cookie text in DOM is dropped along
      // with it. Don't gate on a "submitting" flag; the in-flight
      // fetch will resolve harmlessly into a torn-down scope.
      this.cookieImportOpen = false;
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
        const fullKey = this._bbTeamPrefix() + this.bbNewKey;
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
          const fullKey = this._bbTeamPrefix() + key;
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
      this.cronLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/cron`);
        if (resp.ok) this.cronJobs = (await resp.json()).jobs;
      } catch (e) { console.warn('fetchCronJobs failed:', e); }
      this.cronLoading = false;
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
      if (status === 'quarantined') return 'bg-orange-500';
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

    async fetchPlatformSuccess() {
      // Pull the rollup; the dashboard aggregates events in-process,
      // so this is a cheap dict-walk on the server side.  On error we
      // keep the previous payload visible (operators see a stale row
      // rather than an empty panel).
      this.platformSuccessLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/dashboard/platform-success`);
        if (resp.ok) {
          this.platformSuccessData = await resp.json();
        }
      } catch (e) { console.warn('fetchPlatformSuccess failed:', e); }
      this.platformSuccessLoading = false;
    },

    platformSuccessBarClass(rate) {
      if (rate === null || rate === undefined) return 'bg-gray-700';
      if (rate >= 0.8) return 'bg-emerald-500';
      if (rate >= 0.5) return 'bg-yellow-500';
      return 'bg-red-500';
    },

    platformSuccessTextClass(rate) {
      if (rate === null || rate === undefined) return 'text-gray-400';
      if (rate >= 0.8) return 'text-emerald-400';
      if (rate >= 0.5) return 'text-yellow-400';
      return 'text-red-400';
    },

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
            // Execution-limit settings are env-seeded at launch — apply
            // them automatically so the change isn't silently stale.
            if (data.restart_required) {
              await this._restartAllAgentsAuto('Applying setting — restarting agents…');
            }
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
          // Agents resolve their model at launch, so the new default
          // only takes effect for default-model agents after a restart.
          // Auto-apply so the change isn't silently stale.
          await this._restartAllAgentsAuto('Applying default model — restarting agents…');
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Update failed'}`);
        }
      } catch (e) { console.warn('saveDefaultModel failed:', e); }
    },

    async saveEmbeddingModel(value) {
      if (!value) return;
      try {
        const resp = await fetch(`${window.__config.apiBase}/embedding-model`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ value }),
        });
        if (resp.ok) {
          // Refresh the resolved status (on/off + effective model) from
          // the server, which recomputes against the available keys.
          await this.fetchSystemSettings();
          this.showToast(`Embedding provider set to ${value}`);
          // EMBEDDING_MODEL is only read at agent launch, so a restart is
          // required to apply. Auto-apply so the change isn't silently stale.
          await this._restartAllAgentsAuto('Applying embedding setting — restarting agents…');
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Update failed'}`);
        }
      } catch (e) { console.warn('saveEmbeddingModel failed:', e); }
    },

    // Restart the whole fleet WITHOUT a confirmation prompt. Used to
    // auto-apply dashboard changes that are only picked up at container
    // launch (execution limits, default model). The user-facing
    // ``restartAllAgents`` keeps its confirm dialog for manual use.
    async _restartAllAgentsAuto(reason) {
      this._restartingAll = true;
      this.showToast(reason || 'Applying changes — restarting agents…');
      try {
        const resp = await fetch(`${window.__config.apiBase}/restart-agents`, { method: 'POST' });
        if (resp.ok) {
          const data = await resp.json();
          const agents = Object.entries(data.restarted || {});
          const ok = agents.filter(([, s]) => s === 'ready').length;
          this.showToast(`Applied — restarted ${ok}/${agents.length} agents`);
          this.fetchAgents();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Saved, but restart failed: ${err.detail || resp.status}`);
        }
      } catch (e) { this.showToast(`Saved, but restart failed: ${e.message}`); }
      this._restartingAll = false;
    },

    // Operator Settings → model change. PUTs the operator config (which
    // hot-reloads the model live when possible) and auto-restarts the
    // operator if the server reports the change needs a bounce.
    async saveOperatorModel(model) {
      if (!model) return;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/operator/config`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
          body: JSON.stringify({ model }),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Update failed'}`);
          return;
        }
        const data = await resp.json().catch(() => ({}));
        if (data.restart_required) {
          this.showToast('Operator model set — restarting operator…');
          const r = await fetch(`${window.__config.apiBase}/agents/operator/restart`, { method: 'POST' });
          if (r.ok) {
            const d = await r.json().catch(() => ({}));
            this.showToast(d.ready ? 'Operator restarted and ready' : 'Operator restarting…');
          } else {
            this.showToast('Operator model saved, restart failed');
          }
        } else {
          this.showToast(`Operator model set to ${model}`);
        }
        this.fetchAgents();
      } catch (e) { this.showToast(`Error: ${e.message}`); }
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
      this.storageLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/storage`);
        if (resp.ok) this.storageData = await resp.json();
      } catch (e) { console.warn('fetchStorage failed:', e); }
      this.storageLoading = false;
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
        const wasEmpty = localMsgs.length === 0;
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
        // Phase 3 — strip ACTION: lines off the trailing edge of every
        // operator agent message so the historical transcript renders
        // chips without re-running the LLM. Idempotent.
        if (agentId === 'operator') {
          for (const sm of serverMsgs) this._applyOperatorActions(sm);
        }
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
        // Force scroll when the local cache was empty (first fetch on
        // mount) — the sticky nearBottom heuristic would otherwise treat
        // scrollTop=0 as "user scrolled to top" and skip. Live-update
        // refreshes keep the conservative behavior.
        this.$nextTick(() => this._scrollChat(agentId, wasEmpty));
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
      // Phase 1 — mirror to the server-side opened-conversations set.
      // Best-effort: a network failure shouldn't block the UI.
      if (agentId !== 'operator') {
        this.openConversation(agentId);
      }
    },

    closeChat(agentId) {
      // Operator is permanent — it can never be removed from the messenger.
      // The chat-header X, while Operator is active, either closes the whole
      // panel (when Operator is the only open chat) or simply shifts focus to
      // another open chat, leaving Operator pinned in ``openChats``.
      if (agentId === 'operator') {
        const others = this.openChats.filter(id => id !== 'operator');
        if (others.length === 0) {
          // Operator is the only chat left — dismiss the whole messenger
          // rather than removing Operator. Mirrors the ESC "dismiss panel"
          // path: minimize + clear the legacy side-panel flag + restore
          // focus (``closeSidePanel`` no-ops when not toggle-opened).
          this.chatPanelMinimized = true;
          this.closeSidePanel();
          this._saveChatToSession();
          return;
        }
        // Keep Operator open; move focus to the most recently opened worker.
        this.activeChatId = others[others.length - 1];
        this._saveChatToSession();
        return;
      }
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
      // Phase 1 — mirror to the server-side opened-conversations set so
      // ``/api/conversations`` reflects what's currently in the user's
      // messenger. Best-effort: we don't block on the network call.
      if (agentId !== 'operator') {
        this.closeConversation(agentId);
      }
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
      if (!agentId) return;
      // Newest caller wins — the original dedup silently dropped subsequent
      // triggers, so if the first attempt fired before the chat container
      // had mounted (Alpine x-for / x-show / async _loadChatHistory), no
      // later attempt would catch up.
      if (this._scrollTimers[agentId]) clearTimeout(this._scrollTimers[agentId]);
      let lastSet = null;
      const findVisible = () => {
        // The chat for a given agent can render in two surfaces: the main
        // /chat tab (static id="chat-messages-operator", line ~481 in
        // index.html) and the slide-out side panel (dynamic
        // id="side-chat-messages-{agentId}"). When activeChatId='operator'
        // both surfaces want to scroll. The main-tab element is always
        // mounted via x-show (never removed from DOM), so a plain
        // getElementById('chat-messages-operator') would always return the
        // main-tab element — which is display:none whenever activeTab !==
        // 'chat'. Pick whichever element is actually visible (offsetParent
        // is null for display:none) so the sidebar surface gets scrolled
        // when it's the one the user sees.
        const candidates = [
          document.getElementById('chat-messages-' + agentId),
          document.getElementById('side-chat-messages-' + agentId),
        ].filter(Boolean);
        return candidates.find(el => el.offsetParent !== null) || candidates[0] || null;
      };
      const tryScroll = (allowForce) => {
        const el = findVisible();
        if (!el) return;
        const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
        // If we set scrollTop earlier and it has since drifted, the user
        // scrolled manually — respect them and stop chasing.
        const userMoved = lastSet !== null && Math.abs(el.scrollTop - lastSet) > 4;
        if ((allowForce && force) || (distance < 150 && !userMoved)) {
          el.scrollTop = el.scrollHeight;
          lastSet = el.scrollTop;
        }
      };
      this._scrollTimers[agentId] = setTimeout(() => {
        delete this._scrollTimers[agentId];
        tryScroll(true);
        // Late-render safety net — covers (a) Alpine x-for renders that
        // land after the 50ms timer, (b) async _loadChatHistory replacing
        // the message array, (c) avatar image-load reflows that grow
        // scrollHeight after the initial paint. Retries respect manual
        // user scrolling (force is one-shot via allowForce).
        setTimeout(() => tryScroll(false), 250);
        setTimeout(() => tryScroll(false), 800);
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
        return { label: 'Online', color: 'text-gray-500', dot: 'bg-gray-600' };
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
      if (agentId === 'operator') this._operatorLastUserMessageTs = Date.now();
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
              // Phase 3 — operator response chips. Strip ACTION: lines
              // off the trailing edge of the message and store them on
              // ``entry.suggested_actions`` so the renderer can paint
              // chips. Operator-only: worker chats render free-text.
              if (agentId === 'operator') this._applyOperatorActions(entry);
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
      if (agentId === 'operator') this._operatorLastUserMessageTs = Date.now();

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
      if (this._detailReturnTeam !== null && this._detailReturnTeam !== undefined) {
        this.activeTeam = this._detailReturnTeam;
      }
      this._detailReturnTeam = null;
      this.detailAgent = null;
      this.cronFormAgent = agent;
      this.showCronForm = true;
      this.systemTab = 'automation';
      this.switchTab('system');
    },

    get broadcastTargets() {
      // Team selected → team members; no team → solo agents only.
      // Exclude over-limit (locked) agents — they aren't running. Always exclude operator (system agent, has its own chat).
      if (this.activeTeam) {
        return this.filteredAgents.filter(a => !a.over_limit);
      }
      return this.agents.filter(a => a.id !== 'operator' && !a.over_limit && !a.project);
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
        chat: ['chat', 'operator', 'message', 'talk', 'ask'],
        workplace: ['work', 'home', 'board', 'kanban', 'tasks', 'activity', 'delivered', 'in progress', 'stuck'],
        fleet: ['team', 'agents', 'fleet', 'cards', 'project'],
        system: ['settings', 'system', 'costs', 'cron', 'schedules', 'automation', 'credentials', 'api keys', 'connections', 'integrations', 'infrastructure', 'pricing', 'browsers', 'pubsub', 'blackboard', 'comms', 'communication', 'workflows', 'storage', 'uploads', 'disk', 'network', 'proxy', 'socks'],
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
        { label: 'Broadcast', desc: this.activeTeam ? `Broadcast to ${this.activeTeam} agents` : (this.teams.length > 0 ? 'Broadcast to solo agents' : 'Send message to all agents'), keywords: ['broadcast', 'send', 'all', 'message'], action: () => { this.switchTab('fleet'); if (this.activeTeam) { this.teamHubExpanded = true; this.teamHubTab = 'broadcast'; this.$nextTick(() => document.getElementById('broadcast-input')?.focus()); } else { this.$nextTick(() => document.getElementById('broadcast-solo-input')?.focus()); } } },
        ...(this.activeTeam ? [{ label: 'Edit TEAM.md', desc: `Edit ${this.activeTeam} team context`, keywords: ['team', 'edit', 'context'], action: () => { this.switchTab('fleet'); this.teamHubExpanded = true; this.teamHubTab = 'docs'; this.$nextTick(() => this.startTeamEdit()); } }] : []),
        ...(this.activeTeam ? [{ label: 'Team Members', desc: `Manage ${this.activeTeam} members`, keywords: ['members', 'team', 'assign', 'agents'], action: () => { this.switchTab('fleet'); this.teamHubExpanded = true; this.teamHubTab = 'members'; } }] : []),
      ];
      for (const act of actions) {
        if (act.keywords.some(kw => kw.includes(q)) || act.label.toLowerCase().includes(q)) {
          results.push({ type: 'action', label: act.label, desc: act.desc, action: act.action });
        }
      }
      // Match teams (legacy 'standalone' / 'unassigned' / 'project' keywords kept
      // for back-compat — users with muscle memory still find the action).
      if (this.teams.length > 0) {
        if ('standalone'.startsWith(q) || 'unassigned'.startsWith(q) || 'solo'.startsWith(q)) {
          results.push({ type: 'action', label: 'Solo agents', desc: 'Show agents not in any team', action: () => { this.switchTab('fleet'); this.switchTeam(null); } });
        }
        for (const t of this.teams) {
          const tname = (t.name || '').toLowerCase();
          if (tname.includes(q) || 'team'.startsWith(q) || 'project'.startsWith(q)) {
            results.push({ type: 'action', label: t.name, desc: `Switch to team (${(t.members || []).length} members)`, action: () => { this.switchTab('fleet'); this.switchTeam(t.name); } });
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
      this.showConfirm('Reset Conversation', `Start a fresh conversation with "${agentId}"? The conversation thread is wiped, but memories and tools are preserved.`, async () => {
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

    _getVncUrl(agentId) {
      // Each agent's vnc_url bakes its OWN agent_id into the reverse-proxy
      // path, so a card MUST pass its target agent — otherwise, with 2+
      // agents holding browsers, the operator gets shown a DIFFERENT agent's
      // framebuffer than the one they're focusing / completing the login on.
      const withVnc = this.agents.filter(ag => ag.vnc_url);
      if (agentId) {
        const exact = withVnc.find(ag => ag.id === agentId);
        if (exact) return exact.vnc_url;
        // No URL for this agent yet (e.g. /api/agents not repolled since the
        // browser came up). Only fall back to the sole browser when it's
        // unambiguous — never show a different agent when several exist.
        return withVnc.length === 1 ? withVnc[0].vnc_url : '';
      }
      // Bare call (back-compat probe): first available browser.
      return withVnc.length ? withVnc[0].vnc_url : '';
    },

    async _completeBrowserLogin(msg, agentId) {
      const prev = { completed: msg.completed, cancelled: msg.cancelled };
      msg.completed = true;
      try {
        const resp = await fetch(window.__config.apiBase + '/browser-login/complete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
          // request_id lets the mesh atomically resolve the registry record
          // (pop + steer once), which clears the matching "Needs you" row.
          body: JSON.stringify({ agent_id: agentId || '', service: msg.service, request_id: msg.request_id || '' }),
        });
        if (!resp.ok) {
          msg.completed = prev.completed;
          this.showToast('Failed to notify agent — please try again');
        } else {
          this._refreshHelpRequestsSoon();
        }
      } catch (_) {
        msg.completed = prev.completed;
        this.showToast('Network error — please try again');
      }
    },

    async _cancelBrowserLogin(msg, agentId) {
      const prev = { completed: msg.completed, cancelled: msg.cancelled };
      msg.cancelled = true;
      // PR 3: prefer the request_id-scoped endpoint when the message
      // carries one — that path also pushes a cancellation steer to the
      // awaiting agent so it can react instead of waiting on a TTL.
      // Old messages (pre-PR 3) fall back to the legacy endpoint.
      try {
        let resp;
        if (msg.request_id) {
          resp = await fetch(
            window.__config.apiBase
              + '/browser-login-request/' + encodeURIComponent(msg.request_id) + '/cancel',
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
              body: JSON.stringify({ reason: 'user_cancelled' }),
            },
          );
        } else {
          resp = await fetch(window.__config.apiBase + '/browser-login/cancel', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify({ agent_id: agentId || '', service: msg.service }),
          });
        }
        if (!resp.ok) {
          msg.cancelled = prev.cancelled;
          this.showToast('Failed to notify agent — please try again');
        }
      } catch (_) {
        msg.cancelled = prev.cancelled;
        this.showToast('Network error — please try again');
      }
    },

    async _cancelCredentialRequest(msg) {
      // PR 3: cancel an open credential request via the new
      // request_id-scoped endpoint. The mesh emits
      // ``credential_request_cancelled`` (other card copies sync
      // from the same handler that watches for the event) and pushes
      // a steer to the awaiting agent.
      if (!msg || !msg.request_id) return;
      const prev = { saved: msg.saved, cancelled: msg.cancelled };
      msg.cancelled = true;
      try {
        const resp = await fetch(
          window.__config.apiBase
            + '/credential-request/' + encodeURIComponent(msg.request_id) + '/cancel',
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify({ reason: 'user_cancelled' }),
          },
        );
        if (!resp.ok) {
          msg.saved = prev.saved;
          msg.cancelled = prev.cancelled;
          this.showToast('Failed to cancel — please try again');
        }
      } catch (_) {
        msg.saved = prev.saved;
        msg.cancelled = prev.cancelled;
        this.showToast('Network error — please try again');
      }
    },

    async _completeBrowserCaptchaHelp(msg, agentId) {
      const prev = { completed: msg.completed, cancelled: msg.cancelled };
      msg.completed = true;
      try {
        const resp = await fetch(window.__config.apiBase + '/browser-captcha-help/complete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
          body: JSON.stringify({ agent_id: agentId || '', service: msg.service, request_id: msg.request_id || '' }),
        });
        if (!resp.ok) {
          msg.completed = prev.completed;
          this.showToast('Failed to notify agent — please try again');
        } else {
          this._refreshHelpRequestsSoon();
        }
      } catch (_) {
        msg.completed = prev.completed;
        this.showToast('Network error — please try again');
      }
    },

    async _cancelBrowserCaptchaHelp(msg, agentId) {
      const prev = { completed: msg.completed, cancelled: msg.cancelled };
      msg.cancelled = true;
      // PR 3: prefer request_id-scoped endpoint when available.
      try {
        let resp;
        if (msg.request_id) {
          resp = await fetch(
            window.__config.apiBase
              + '/browser-captcha-help-request/' + encodeURIComponent(msg.request_id) + '/cancel',
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
              body: JSON.stringify({ reason: 'user_cancelled' }),
            },
          );
        } else {
          resp = await fetch(window.__config.apiBase + '/browser-captcha-help/cancel', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify({ agent_id: agentId || '', service: msg.service }),
          });
        }
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
        // Optimistic: focusBrowser succeeded → the agent has a browser
        // by definition. Update agentDetail locally so the iframe
        // gating (which checks ``browser_running``) doesn't have to
        // wait for the next /api/agents poll. Next poll will confirm.
        if (this.agentDetail && this.agentDetail.id === agentId) {
          this.agentDetail.browser_running = true;
        }
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

    // Complete the initial-setup modal: persist the Default agent model
    // (mesh.yaml) and the Operator model, then mark setup done so the
    // modal never reappears. The operator model hot-reloads live; the
    // default model only matters for future agents (none exist yet at
    // first setup), so no fleet restart is needed here.
    async finishSetup() {
      if (this.onboardFinishing) return;
      if (!this.settingsData?.has_llm_credentials) {
        this.showToast('Add an API key or use OpenLegion credits to continue.');
        return;
      }
      if (!this.onboardOperatorModel || !this.onboardDefaultModel) {
        this.showToast('Pick both an Operator model and a Default agent model.');
        return;
      }
      this.onboardFinishing = true;
      try {
        const dmResp = await fetch(`${window.__config.apiBase}/default-model`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
          body: JSON.stringify({ model: this.onboardDefaultModel }),
        });
        if (!dmResp.ok) {
          const err = await dmResp.json().catch(() => ({}));
          this.showToast(`Error setting default model: ${err.detail || dmResp.status}`);
          return;
        }
        if (this.systemSettings) this.systemSettings.default_model = this.onboardDefaultModel;

        const opResp = await fetch(`${window.__config.apiBase}/agents/operator/config`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
          body: JSON.stringify({ model: this.onboardOperatorModel }),
        });
        if (!opResp.ok) {
          const err = await opResp.json().catch(() => ({}));
          this.showToast(`Error setting operator model: ${err.detail || opResp.status}`);
          return;
        }
        // The model normally hot-reloads live, but if the container
        // couldn't be reached the server asks for a restart — honour it
        // so the operator isn't left on its boot default. Mirrors
        // saveOperatorModel().
        const opData = await opResp.json().catch(() => ({}));
        if (opData.restart_required) {
          await fetch(`${window.__config.apiBase}/agents/operator/restart`, { method: 'POST' }).catch(() => {});
        }

        try { localStorage.setItem('ol_setup_done', '1'); } catch (_) { /* ignore */ }
        this._setupDone = true;
        this.showToast('Setup complete — say hello to your Operator.');
        this.fetchAgents();
        this.fetchSettings();
      } catch (e) {
        this.showToast(`Setup failed: ${e.message}`);
      } finally {
        this.onboardFinishing = false;
      }
    },

    // ── Channels ──────────────────────────────────────────

    async fetchChannels() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/channels`);
        if (resp.ok) this.channels = (await resp.json()).channels || [];
      } catch (e) { console.warn('fetchChannels failed:', e); }
    },

    // ── OAuth integrations (Google Drive/Gmail/Calendar) ─────────────
    // The connect/disconnect endpoints live under ``apiBase`` (/dashboard/api);
    // the full-page consent redirect lives one level up at /dashboard/integrations.
    _integrationsBase() {
      // apiBase is "/dashboard/api" → strip the trailing "/api" for the
      // browser-redirect connect route which is NOT under /api.
      return (window.__config.apiBase || '').replace(/\/api$/, '');
    },

    async loadIntegrations() {
      try {
        const resp = await fetch(`${window.__config.apiBase}/integrations`);
        if (!resp.ok) return;
        const data = await resp.json();
        this.integrations = data.providers || [];
        // Seed per-provider form defaults without clobbering in-flight edits.
        for (const p of this.integrations) {
          if (!this.integrationSetup[p.key]) {
            this.integrationSetup[p.key] = { client_id: '', client_secret: '' };
          }
          if (this.integrationConnectName[p.key] === undefined) {
            this.integrationConnectName[p.key] = p.key;
          }
          if (!this.integrationSelectedScopes[p.key]) {
            this.integrationSelectedScopes[p.key] = {};
          }
        }
      } catch (e) { console.warn('loadIntegrations failed:', e); }
    },

    async setupIntegration(providerKey) {
      if (this.integrationSetupSaving) return;
      const form = this.integrationSetup[providerKey] || {};
      const clientId = (form.client_id || '').trim();
      const clientSecret = (form.client_secret || '').trim();
      if (!clientId || !clientSecret) {
        this.showToast('Client ID and Client Secret are required');
        return;
      }
      this.integrationSetupSaving = providerKey;
      try {
        const resp = await fetch(`${window.__config.apiBase}/integrations/${encodeURIComponent(providerKey)}/setup`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
          credentials: 'same-origin',
          body: JSON.stringify({ client_id: clientId, client_secret: clientSecret }),
        });
        if (resp.ok) {
          this.showToast('OAuth app saved. You can now connect.');
          this.integrationSetup[providerKey] = { client_id: '', client_secret: '' };
          await this.loadIntegrations();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Setup failed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message || e}`); }
      finally { this.integrationSetupSaving = ''; }
    },

    connectIntegration(providerKey) {
      const name = (this.integrationConnectName[providerKey] || providerKey).trim() || providerKey;
      const selected = this.integrationSelectedScopes[providerKey] || {};
      const bundles = Object.keys(selected).filter(k => selected[k]);
      if (!bundles.length) {
        this.showToast('Select at least one access bundle to connect');
        return;
      }
      const params = new URLSearchParams({ name, scopes: bundles.join(',') });
      // Full-page navigation — the auth cookie rides Google's redirect back.
      window.location.href = `${this._integrationsBase()}/integrations/${encodeURIComponent(providerKey)}/connect?${params.toString()}`;
    },

    async disconnectIntegration(name) {
      if (this.integrationDisconnecting) return;
      this.integrationDisconnecting = name;
      try {
        const resp = await fetch(`${window.__config.apiBase}/integrations/${encodeURIComponent(name)}/disconnect`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
          credentials: 'same-origin',
          body: '{}',
        });
        if (resp.ok) {
          this.showToast(`Disconnected: ${name}`);
          await this.loadIntegrations();
        } else {
          const err = await resp.json().catch(() => ({}));
          this.showToast(`Error: ${err.detail || 'Disconnect failed'}`);
        }
      } catch (e) { this.showToast(`Error: ${e.message || e}`); }
      finally { this.integrationDisconnecting = ''; }
    },

    // Read ?integration_connected / ?integration_error from the OAuth callback
    // redirect, surface a toast, then scrub the query param so a refresh
    // doesn't re-toast.
    _handleIntegrationRedirect() {
      try {
        const params = new URLSearchParams(window.location.search);
        const connected = params.get('integration_connected');
        const error = params.get('integration_error');
        if (!connected && !error) return;
        if (connected) {
          this.showToast(`Connected: ${connected}`);
        } else if (error) {
          const pretty = String(error).replace(/_/g, ' ');
          this.showToast(`Integration failed: ${pretty}`);
        }
        params.delete('integration_connected');
        params.delete('integration_error');
        const qs = params.toString();
        const newUrl = window.location.pathname + (qs ? `?${qs}` : '') + window.location.hash;
        window.history.replaceState({}, '', newUrl);
      } catch (e) { /* ignore */ }
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
      // Operator is a system agent — route to its dedicated settings page instead of the generic agent detail panel.
      // Centralizes operator routing for all callers (card clicks, URL routes, search results).
      if (agentId === 'operator') {
        // Suppress switchTab's URL push so the back button returns to the source page, not the intermediate /system/<previous-tab>.
        // Save/restore _skipPush so we don't clobber the outer invariant (e.g. _applyRoute holds _skipPush=true).
        const prevSkip = this._skipPush;
        this._skipPush = true;
        try { this.switchTab('system'); } finally { this._skipPush = prevSkip; }
        this.switchSystemTab('operator');
        return;
      }
      this._detailReturnTeam = this.activeTeam;
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
      // detail panel renders the click-rate signal immediately on open.
      // Subsequent updates flow through the existing browser_metrics
      // WS handler.
      this.fetchBrowserMetricsHistory(agentId);
      // Phase 10 §20 — privacy-safe session sidecar summary, used by
      // the Browser detail row to surface "N origins saved" so the
      // operator knows whether Reset Browser will blow away auth state.
      this.fetchAgentBrowserSession(agentId);
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
      if (rate == null) return 'text-gray-400';
      if (rate >= 0.9) return 'text-green-400';
      if (rate >= 0.7) return 'text-yellow-400';
      return 'text-red-400';
    },
    clickRateBarColor(rate) {
      // Bar-chart variant of clickRateColor — returns a Tailwind bg
      // utility for the 9-segment mini bar in the Browser Health table.
      if (rate == null) return 'bg-gray-700';
      if (rate >= 0.9) return 'bg-green-500/70';
      if (rate >= 0.7) return 'bg-yellow-500/70';
      return 'bg-red-500/70';
    },
    browserHealthRows() {
      // Sort failing agents to the top so a regressing site is the
      // first thing the operator sees. Stale agents (no recent
      // metrics) drop to the bottom — they're informational, not
      // actionable. Idle (no clicks recorded yet) sort by name.
      return this.browserMetricsList().slice().sort((a, b) => {
        const aStale = this.browserMetricsStale(a.receivedAt) ? 1 : 0;
        const bStale = this.browserMetricsStale(b.receivedAt) ? 1 : 0;
        if (aStale !== bStale) return aStale - bStale;
        const ar = a.click_success_rate_100;
        const br = b.click_success_rate_100;
        // Both null → name; one null → null last; otherwise ascending rate
        if (ar == null && br == null) return a.agent.localeCompare(b.agent);
        if (ar == null) return 1;
        if (br == null) return -1;
        return ar - br;
      });
    },
    browserHealthSummary() {
      // One-line summary shown in the card header — agent count plus
      // the worst current click rate, so the page tells the operator
      // "everything's fine" or "something's red" without expanding.
      const list = this.browserMetricsList();
      const live = list.filter(m => !this.browserMetricsStale(m.receivedAt));
      if (live.length === 0) return list.length ? 'all idle' : '';
      const rated = live.filter(m => m.click_success_rate_100 != null);
      if (rated.length === 0) return live.length + ' active · no clicks yet';
      const worst = Math.min(...rated.map(m => m.click_success_rate_100));
      return live.length + ' active · low ' + this.fmtClickRate(worst);
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

    async fetchAgentBrowserSession(agentId) {
      // Privacy-safe sidecar summary — counts only, no origin domains.
      // The dashboard endpoint returns the §2.3 envelope so we can
      // distinguish "service down" from "no session"; the caller only
      // cares about the data shape, so we skip surface-level errors
      // (the row segment hides itself when has_persisted_session is
      // falsy).
      if (!agentId || agentId === 'operator') return;
      try {
        const resp = await fetch(`${window.__config.apiBase}/agents/${agentId}/session`);
        if (!resp.ok) return;
        const body = await resp.json().catch(() => ({}));
        if (body && body.success && body.data) {
          this.agentBrowserSession = {
            ...this.agentBrowserSession,
            [agentId]: body.data,
          };
        }
      } catch (e) {
        // Service unavailable / offline — leave the entry absent so the
        // row segment stays hidden rather than rendering a stale count.
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
          if (d.oauth) add('Cost', 'subscription (no per-call cost)');
          else if (d.cost_usd != null) add('Cost', '$' + d.cost_usd.toFixed(6));
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
          const cost = d.oauth ? ' \u00b7 sub' : (d.cost_usd != null ? ` \u00b7 $${d.cost_usd.toFixed(4)}` : '');
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

    // ── Phase 2 Board UX — Activity translation ─────────────
    //
    // Maps engineer-style event types to plain-English summaries
    // for the Activity feed. Returns null when the event is an
    // implementation detail that should be hidden from non-power
    // users (toggleable via showTechDetail). Returns a string
    // when the event has a user-friendly summary; falls back to
    // the existing eventSummary() output otherwise.
    formatActivityForUser(event) {
      if (!event || !event.type) return null;
      const d = event.data || {};
      const agent = event.agent ? this.agentDisplayName(event.agent) : 'An agent';
      switch (event.type) {
        case 'tool_start': {
          const toolName = d.tool || d.name || event.tool_name || '';
          return `${agent} is ${this.verbForTool(toolName)}`;
        }
        case 'tool_result': {
          const toolName = d.tool || d.name || event.tool_name || '';
          return `${agent} finished ${this.verbForTool(toolName)}`;
        }
        case 'task_status_changed': {
          const newStatus = d.new_status || d.status || event.new_status || '';
          const reason = (newStatus === 'cancelled' && d.reason) ? ` (${d.reason})` : '';
          return `${agent} ${this.verbForStatus(newStatus)}${reason}`;
        }
        case 'task_outcome': {
          const outcome = d.outcome || event.outcome || 'reviewed';
          return `${agent}'s work was ${outcome}`;
        }
        case 'credential_request': {
          const label = d.credential_label || d.label || event.credential_label || 'credential';
          return `${agent} needs a ${label} — your call`;
        }
        case 'pending_action_created': {
          const actionLabel = d.action_label || event.action_label || d.action || 'make a change';
          return `${agent} wants to ${actionLabel}`;
        }
        case 'task_created': {
          const title = (d.title || '').substring(0, 60);
          return title ? `${agent} picked up "${title}"` : `${agent} picked up a new task`;
        }
        case 'health_change':
          return `${agent} is now ${d.current || 'unknown'}`;
        case 'heartbeat_complete': {
          const out = d.outcome ? ` (${d.outcome})` : '';
          return `${agent} finished a checkup${out}`;
        }
        case 'notification':
          return (d.message || '').substring(0, 100) || null;
        case 'browser_login_request':
          return `${agent} needs a sign-in — your call`;
        case 'browser_captcha_help_request':
          return `${agent} hit a CAPTCHA — your call`;
        case 'credit_exhausted':
          return `${agent} is out of credit for now`;
        // Hidden by default — implementation noise. Power users see
        // them via the "Show technical detail" toggle (which falls
        // back to eventSummary()).
        case 'blackboard_write':
        case 'llm_call':
        case 'message_received':
        case 'message_sent':
        case 'text_delta':
        case 'agent_state':
          return null;
        // Phase 4 — heartbeat-driven re-affirmations of the same
        // status (the orchestration layer emits ``status_unchanged``
        // when a task transition is requested but already at the
        // target). Filtered for the same reason as the noise events
        // above: hidden by default, available via "Show technical
        // detail" for debugging. The engine emits the single name
        // ``status_unchanged`` (see orchestration.py) — there is no
        // alternate spelling on the wire.
        case 'status_unchanged':
          return null;
        default:
          return event.summary || null;
      }
    },

    // Human verb for a tool name. Default returns the tool name
    // unchanged so unknown tools still surface (better than dropping
    // the event entirely).
    verbForTool(toolName) {
      if (!toolName) return 'working';
      const map = {
        web_search: 'searching the web',
        web_fetch: 'reading a web page',
        browser_navigate: 'opening a website',
        browser_click: 'clicking a link',
        browser_type: 'typing in a form',
        browser_screenshot: 'taking a screenshot',
        browser_get_elements: 'reading the page',
        browser_find_text: 'searching the page',
        browser_fill_form: 'filling out a form',
        browser_solve_captcha: 'solving a CAPTCHA',
        http_request: 'making an HTTP request',
        exec: 'running a command',
        execute_code: 'running code',
        file_read: 'reading a file',
        file_write: 'writing a file',
        file_list: 'listing files',
        commit_file: 'committing a file to GitHub',
        memory_search: 'checking memory',
        memory_think: 'reasoning over memory',
        memory_save: 'saving to memory',
        hand_off: 'handing off to a teammate',
        check_inbox: 'checking the inbox',
        update_status: 'updating status',
        complete_task: 'wrapping up a task',
        notify_user: 'pinging you',
        spawn: 'spinning up a helper',
        write_blackboard: 'writing to the shared workspace',
        read_blackboard: 'reading the shared workspace',
        image_gen: 'generating an image',
        wallet_get_address: 'looking up a wallet address',
        wallet_get_balance: 'checking a wallet balance',
        wallet_transfer: 'sending a wallet transfer',
        edit_agent: 'updating a teammate',
        read_agent_config: 'reviewing a teammate',
        read_user_notifications: 'reviewing recent notifications',
        read_peer_artifact: 'reading a teammate\'s file',
        list_peer_artifacts: 'browsing a teammate\'s files',
        read_peer_file: 'reading a teammate\'s file',
        list_peer_files: 'browsing a teammate\'s files',
        create_agent: 'adding a teammate',
        apply_template: 'building a team from a template',
        inspect_agents: 'reviewing the team',
        list_available_models: 'checking available models',
        compose_work_summary: 'composing a work summary',
        workflow_snapshot: 'mapping the workflow',
        await_task_event: 'waiting on a task',
        inspect_task_run: 'diagnosing a task run',
        update_team_brief: 'updating the team brief',
        skills_list: 'browsing its skills',
        skill_view: 'reading a skill',
        install_skill: 'installing a skill',
        remove_skill: 'removing a skill',
        list_skill_assignments: 'reviewing skill assignments',
        assign_skill: 'assigning a skill',
      };
      if (map[toolName]) return map[toolName];
      // Fallback: humanise the snake_case tool name.
      return `using ${toolName.replace(/_/g, ' ')}`;
    },

    verbForStatus(status) {
      if (!status) return 'updated a task';
      const map = {
        new: 'picked up a new task',
        working: 'started working',
        in_progress: 'started working',
        blocked: 'is blocked',
        done: 'finished a task',
        completed: 'finished a task',
        failed: 'hit an error',
        cancelled: 'cancelled a task',
        delivered: 'delivered work',
        accepted: 'got the green light',
        rework: 'is reworking a task',
        rejected: 'had work marked rejected',
      };
      return map[status] || `moved a task to ${status.replace(/_/g, ' ')}`;
    },

    // Tiny markdown renderer for summary narratives. Handles the
    // shapes ``compose_work_summary`` emits: ``## H2``, ``**bold**``,
    // ``- bullet``, paragraph breaks. HTML-escapes input first so a
    // hostile narrative (or one that quoted user text) can't inject.
    // Bigger surfaces (chat history, etc.) use a real renderer; this
    // is the right grain for the summary cards.
    renderSummaryMarkdown(md) {
      if (!md) return '';
      const esc = (s) => s
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
      const lines = esc(String(md)).split(/\r?\n/);
      const out = [];
      let inList = false;
      const closeList = () => { if (inList) { out.push('</ul>'); inList = false; } };
      for (const raw of lines) {
        const line = raw.trimEnd();
        if (!line.trim()) { closeList(); continue; }
        if (line.startsWith('## ')) {
          closeList();
          out.push(`<h3 class="font-semibold text-gray-100 text-sm">${line.slice(3)}</h3>`);
          continue;
        }
        if (line.startsWith('- ')) {
          if (!inList) { out.push('<ul class="list-disc ml-4 space-y-0.5">'); inList = true; }
          const item = line.slice(2).replace(/\*\*(.+?)\*\*/g,
            '<strong class="text-gray-100">$1</strong>');
          out.push(`<li class="text-xs text-gray-300">${item}</li>`);
          continue;
        }
        closeList();
        const para = line.replace(/\*\*(.+?)\*\*/g,
          '<strong class="text-gray-100">$1</strong>');
        out.push(`<p class="text-xs text-gray-300">${para}</p>`);
      }
      closeList();
      return out.join('');
    },

    // Human-readable relative timestamp for summary cards. ``ts`` is
    // a unix seconds float (matches ``generated_at`` / ``rated_at``).
    relativeTime(ts) {
      if (!ts) return '';
      const seconds = Math.max(0, Date.now() / 1000 - Number(ts));
      if (seconds < 60) return 'just now';
      if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
      if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
      return `${Math.round(seconds / 86400)}d ago`;
    },

    // Pretty-print an agent id for the activity feed. Operator gets
    // its brand label; all others get the agent.role label when
    // available, falling back to the id with underscores spaced.
    agentDisplayName(agentId) {
      if (!agentId) return 'Someone';
      if (agentId === 'operator') return 'Operator';
      const agent = this.agents.find(a => a.id === agentId);
      if (agent && agent.role) return agent.role;
      return agentId.replace(/[_-]/g, ' ');
    },

    // ── Undo countdown helpers (operator_action_receipt) ───────
    // The receipt card embeds a per-message ``_tick`` counter that
    // increments every second; both helpers accept it as a reactive
    // dependency so Alpine re-renders the countdown each tick. Without
    // ``_tick`` the helper would be pure Date.now() and Alpine would
    // never observe a state change. The receipt template is duplicated
    // in two messenger surfaces (operator chat tab + side panel) — both
    // use these helpers so the countdown logic lives in one place.
    undoSecondsLeft(msg, _tick) {
      // Touch ``_tick`` so Alpine treats it as a reactive dep. A void
      // expression keeps lint quiet without changing semantics.
      void _tick;
      if (!msg || !msg.expires_at) return 0;
      const left = Math.floor((Number(msg.expires_at) * 1000 - Date.now()) / 1000);
      return left > 0 ? left : 0;
    },

    formatUndoCountdown(seconds) {
      const s = Math.max(0, Math.floor(Number(seconds) || 0));
      const m = Math.floor(s / 60);
      const r = s % 60;
      return `(${m}:${String(r).padStart(2, '0')})`;
    },

    // Whether an activity event should be visible on the user-facing
    // feed. Implementation noise (blackboard_write, llm_call,
    // message_received, text_delta, agent_state) is hidden unless
    // the user has flipped the "Show technical detail" toggle.
    isActivityEventVisible(event) {
      if (this.showTechDetail) return true;
      const summary = this.formatActivityForUser(event);
      return summary !== null;
    },

    /**
     * Count of events hidden by the showTechDetail filter — power
     * users see them inline, others get a low-noise rollup at the
     * bottom of the feed ("+ N coordination events"). Returns 0 when
     * there's nothing to surface (filter active, tech detail on, or
     * no hidden events).
     */
    get hiddenCoordinationEventsCount() {
      if (this.showTechDetail) return 0;
      if (this.eventFilters && this.eventFilters.length > 0) return 0;
      const events = this.events || [];
      let n = 0;
      for (const e of events) {
        if (e && e.type === 'agent_state' && (e.data?.state === 'registered')) continue;
        if (!this.isActivityEventVisible(e)) n += 1;
      }
      return n;
    },

    // Toggle the persistent "Show technical detail" preference.
    setShowTechDetail(value) {
      this.showTechDetail = !!value;
      try {
        localStorage.setItem('olShowTechDetail', this.showTechDetail ? '1' : '0');
      } catch (_) {
        // localStorage may be unavailable (private mode); ignore.
      }
    },

    // ── Phase 2 Board UX — Notifications bell ───────────────
    async fetchNotifications() {
      this.notificationsLoading = true;
      try {
        const resp = await fetch(`${window.__config.apiBase}/notifications`);
        if (!resp.ok) {
          this.notifications = [];
          this.notificationsUnreadCount = 0;
          return;
        }
        const data = await resp.json();
        const fresh = data.notifications || [];
        const previousLastId = this._lastNotifiedId;
        // Track the high-water mark across the merged set so the next
        // poll can identify genuinely-new entries.
        let highestId = previousLastId;
        for (const n of fresh) {
          if (typeof n.id === 'number' && n.id > highestId) highestId = n.id;
        }
        this.notifications = fresh;
        this.notificationsUnreadCount = data.unread_count || 0;
        // Fire browser notifications only for entries newer than the
        // last poll. Skip on the very first fetch (previousLastId === 0)
        // so reloading the page doesn't replay the inbox.
        if (previousLastId > 0) {
          for (const n of fresh) {
            if (n.id && n.id > previousLastId && !n.read_at) {
              this._maybeFireBrowserNotification(n);
            }
          }
        }
        this._lastNotifiedId = highestId;
      } catch (e) {
        // Silent failure; the bell is a polish surface.
        this.notifications = [];
        this.notificationsUnreadCount = 0;
      } finally {
        this.notificationsLoading = false;
      }
    },

    // ── Browser Notification API ─────────────────────────────
    //
    // Off-tab signal for users without a messaging channel configured.
    // Triple-gated: requires browserNotifyEnabled === true AND
    // Notification.permission === 'granted' AND the dashboard tab is
    // not currently visible. We never auto-prompt; the user must click
    // "Enable browser notifications" in the wizard first-output card
    // (or the equivalent settings affordance).
    //
    // This consumes ``notifications`` produced by the existing notification
    // bell fetch (PR-B). We don't manufacture our own queue.
    async requestBrowserNotificationPermission() {
      if (typeof Notification === 'undefined' || !Notification) {
        return 'unsupported';
      }
      let permission = Notification.permission;
      if (permission === 'default') {
        try {
          permission = await Notification.requestPermission();
        } catch (_) {
          // Some browsers throw on call-without-gesture; treat as denied.
          permission = Notification.permission || 'denied';
        }
      }
      this.browserNotifyPermission = permission;
      if (permission === 'granted') {
        this.browserNotifyEnabled = true;
        try {
          localStorage.setItem('olBrowserNotifyEnabled', 'true');
        } catch (_) { /* ignore */ }
      }
      return permission;
    },

    // Toggle off without revoking the OS-level permission (browsers
    // don't let JS revoke that — user must visit site settings). The
    // localStorage flag is the in-app off-switch.
    disableBrowserNotifications() {
      this.browserNotifyEnabled = false;
      try {
        localStorage.setItem('olBrowserNotifyEnabled', 'false');
      } catch (_) { /* ignore */ }
    },

    // Fire a Notification for a specific notification entry, but only
    // if all three gates pass. Click handler focuses the window and
    // routes to the relevant in-app context.
    _maybeFireBrowserNotification(notification) {
      if (!notification || !notification.kind) return;
      if (!this.browserNotifyEnabled) return;
      if (typeof Notification === 'undefined' || !Notification) return;
      if (Notification.permission !== 'granted') return;
      // Only fire when the tab isn't visible — the in-app UI already
      // signals foreground users plenty.
      if (typeof document !== 'undefined' && document.visibilityState === 'visible') return;
      if (this._browserNotifyKinds.indexOf(notification.kind) === -1) return;
      let title = notification.title || 'OpenLegion';
      let body = notification.body || '';
      try {
        const n = new Notification(title, {
          body: body || ' ',
          icon: '/dashboard/static/favicon.png',
          tag: 'ol-' + notification.id,
        });
        n.onclick = (ev) => {
          try {
            ev.preventDefault();
            if (typeof window !== 'undefined' && window.focus) window.focus();
            this.onNotificationClick(notification);
          } catch (_) { /* ignore */ }
          try { n.close(); } catch (_) { /* ignore */ }
        };
      } catch (_) {
        // Notification constructors can throw on unsupported platforms
        // (e.g. iOS Safari pre-PWA). Fail closed.
      }
    },

    // ── Long-task progress rollup ─────────────────────────────
    //
    // Presentation-layer aggregation for the activity feed. When the
    // same agent fires the same tool repeatedly (e.g. a Researcher
    // running web_search 50 times across a 30-minute task), we collapse
    // the run into a single row: "Researcher is searching the web (4
    // queries · 12m elapsed)".
    //
    // Reset triggers (per spec):
    //   - tool changes (different tool name)
    //   - status changes (task_status_changed in the stream)
    //   - different agent
    //
    // Underlying ``events`` array is unchanged — the "Show technical
    // detail" toggle still surfaces the raw stream. Returns a list of
    // entries either ``{kind: 'single', event}`` or
    // ``{kind: 'rollup', event, count, startTs, endTs}``.
    _rollupActivityEvents(events) {
      if (!Array.isArray(events) || events.length === 0) return [];
      const out = [];
      let group = null; // {agent, tool, events: [], startTs, endTs}
      const flush = () => {
        if (!group) return;
        if (group.events.length === 1) {
          out.push({ kind: 'single', event: group.events[0] });
        } else {
          out.push({
            kind: 'rollup',
            event: group.events[group.events.length - 1],
            count: group.events.length,
            startTs: group.startTs,
            endTs: group.endTs,
            tool: group.tool,
          });
        }
        group = null;
      };
      for (const ev of events) {
        if (!ev || !ev.type) {
          flush();
          continue;
        }
        // Status changes always reset grouping (per spec).
        if (ev.type === 'task_status_changed') {
          flush();
          out.push({ kind: 'single', event: ev });
          continue;
        }
        if (ev.type !== 'tool_start') {
          flush();
          out.push({ kind: 'single', event: ev });
          continue;
        }
        const toolName = (ev.data && (ev.data.tool || ev.data.name)) || ev.tool_name || '';
        const agentId = ev.agent || '';
        const ts = ev.timestamp || ev.ts || 0;
        if (group && group.agent === agentId && group.tool === toolName) {
          group.events.push(ev);
          group.endTs = ts || group.endTs;
        } else {
          flush();
          group = { agent: agentId, tool: toolName, events: [ev], startTs: ts, endTs: ts };
        }
      }
      flush();
      return out;
    },

    // Format a rollup entry as a single feed line. Counts use plural
    // "queries"/"requests"/"actions" depending on the verb category.
    formatRolledActivityLine(entry) {
      if (!entry) return null;
      if (entry.kind !== 'rollup') {
        return this.formatActivityForUser(entry.event);
      }
      const ev = entry.event;
      const baseLine = this.formatActivityForUser(ev);
      if (!baseLine) return null;
      const count = entry.count || 0;
      const noun = this._rollupNounForTool(entry.tool);
      const elapsed = this._formatRolledElapsed(entry.startTs, entry.endTs);
      const elapsedSuffix = elapsed ? ` · ${elapsed} elapsed` : '';
      return `${baseLine} (${count} ${noun}${elapsedSuffix})`;
    },

    _rollupNounForTool(toolName) {
      const map = {
        web_search: 'queries',
        web_fetch: 'pages',
        browser_navigate: 'visits',
        browser_click: 'clicks',
        browser_type: 'inputs',
        browser_screenshot: 'screenshots',
        browser_get_elements: 'reads',
        browser_find_text: 'searches',
        browser_fill_form: 'forms',
        http_request: 'requests',
        exec: 'commands',
        file_read: 'reads',
        file_write: 'writes',
        file_list: 'listings',
        memory_search: 'lookups',
        memory_save: 'saves',
        read_blackboard: 'reads',
        write_blackboard: 'writes',
      };
      return map[toolName] || 'actions';
    },

    _formatRolledElapsed(startTs, endTs) {
      if (!startTs || !endTs) return '';
      const startEpoch = typeof startTs === 'string' ? new Date(startTs).getTime() / 1000 : startTs;
      const endEpoch = typeof endTs === 'string' ? new Date(endTs).getTime() / 1000 : endTs;
      if (isNaN(startEpoch) || isNaN(endEpoch)) return '';
      const diff = Math.max(0, endEpoch - startEpoch);
      if (diff < 1) return '';
      if (diff < 60) return `${Math.round(diff)}s`;
      if (diff < 3600) return `${Math.round(diff / 60)}m`;
      return `${Math.round(diff / 3600)}h`;
    },

    // Convenience getter — feed-renderer-friendly rollup over the
    // (already filtered) activity feed.
    get rolledFilteredEvents() {
      return this._rollupActivityEvents(this.filteredEvents);
    },

    toggleNotifications() {
      this.notificationsOpen = !this.notificationsOpen;
      if (this.notificationsOpen) this.fetchNotifications();
    },

    async markNotificationRead(notification) {
      if (!notification || !notification.id) return;
      if (notification.read_at) return;
      // Optimistic update — flip the row in-place so the dropdown
      // doesn't flicker. Rollback only if the request errors.
      const prevReadAt = notification.read_at;
      notification.read_at = Date.now() / 1000;
      this.notificationsUnreadCount = Math.max(0, this.notificationsUnreadCount - 1);
      try {
        const resp = await fetch(`${window.__config.apiBase}/notifications/${notification.id}/read`, {
          method: 'POST',
        });
        if (!resp.ok) {
          notification.read_at = prevReadAt;
          this.notificationsUnreadCount += 1;
        }
      } catch (e) {
        notification.read_at = prevReadAt;
        this.notificationsUnreadCount += 1;
      }
    },

    async markAllNotificationsRead() {
      // Optimistic update.
      const previous = this.notifications.map(n => n.read_at);
      const now = Date.now() / 1000;
      for (const n of this.notifications) {
        if (!n.read_at) n.read_at = now;
      }
      const previousUnread = this.notificationsUnreadCount;
      this.notificationsUnreadCount = 0;
      try {
        const resp = await fetch(`${window.__config.apiBase}/notifications/read-all`, {
          method: 'POST',
        });
        if (!resp.ok) {
          this.notifications.forEach((n, i) => { n.read_at = previous[i]; });
          this.notificationsUnreadCount = previousUnread;
        }
      } catch (e) {
        this.notifications.forEach((n, i) => { n.read_at = previous[i]; });
        this.notificationsUnreadCount = previousUnread;
      }
    },

    // Click handler for a notification row. Marks read; if the
    // payload includes a click-through target (agent / task id),
    // navigates accordingly.
    onNotificationClick(notification) {
      this.markNotificationRead(notification);
      const payload = notification && notification.payload;
      if (!payload) return;
      try {
        if (payload.agent_id) {
          this.notificationsOpen = false;
          this.drillDown(payload.agent_id);
        } else if (payload.task_id && typeof this.openTaskDrillIn === 'function') {
          this.notificationsOpen = false;
          this.openTaskDrillIn(payload.task_id);
        }
      } catch (_) {
        // Best-effort navigation — failures shouldn't block the read.
      }
    },

    // Icon glyph for the notification kind. Plain text glyphs keep
    // the markup emoji-free.
    notificationKindIcon(kind) {
      switch (kind) {
        case 'delivered': return '★';
        case 'approval': return '?';
        case 'alert': return '⚠';
        case 'blocker': return '⚠';
        case 'credential': return 'K';
        case 'info':
        default: return '•';
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
