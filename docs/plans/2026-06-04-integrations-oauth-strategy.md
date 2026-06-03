# Integrations / OAuth — Product Strategy & Decision Record

**Date:** 2026-06-04
**Author:** Product (CPO review)
**Status:** Accepted direction; implementation phased
**Related:** `docs/plans/2026-06-04-oauth-integrations-connect-flow.md` (the Option-B implementation)

---

## 1. The job-to-be-done

Users want their **agents to act on their behalf in the SaaS tools they already use** — read my
Drive, send Gmail, post to my Facebook Page, read my Slack — **without becoming an OAuth-app
developer.** The original complaint was concrete: *"setting up Google Drive / Meta is a real pain —
many steps just to get a token to paste into the vault."*

## 2. Honest assessment of what we shipped (Option B, BYO app)

Option B (bring-your-own OAuth app: the user registers their own app, supplies client_id/secret
once, then connects) is the **right foundation but not the end-state.**

- ✅ It fixes the **repeat** pain — no more hand-minting tokens, no more expiry surprises
  (transparent refresh). It validates the entire vault → `$CRED{}` → agent-tool plumbing.
- ❌ It does **not** remove the pain the user actually described. The "many steps" *are* the
  app registration (create a cloud project, configure the consent screen, add scopes + redirect
  URI, pass verification). Option B leaves all of that with the user.

**Conclusion:** ship B as the bridge (it unblocks self-hosters and power users today), but treat
**Option A as the destination.**

## 3. The destination — Option A (first-party "Connect")

We own **verified OAuth apps**; the user clicks *Connect Google*, consents, done. **Zero developer
steps.** This is the Zapier / Notion / Linear pattern and the version that genuinely kills the pain.
Deferred for real reasons — provider app verification (Google/Meta), central secret custody, and a
broker. Because we built B's `store_connection` to be callable from both the local callback *and* an
internal push endpoint, A is mostly "swap where the client secret comes from + a broker in `app/`."

## 4. The third axis — build vs. adopt (decide before provider #5)

Hand-maintaining OAuth dances for N providers is a forever-tax (API deprecations, scope changes,
re-verification). Two ways to avoid owning the long tail:

- **MCP connectors** — the ecosystem is standardizing here; mature Google/Slack/GitHub MCP servers
  already exist with brokered OAuth. The engine can be an MCP *client* and inherit them.
- **Aggregator** (Nango / Composio / Paragon / Merge) — vendors whose whole product is
  "OAuth + refresh + normalized APIs for 200+ SaaS." Rent the treadmill.

**Recommended posture:** hand-build the top 3–5 highest-value providers (where we want deep, typed
tools); adopt MCP/aggregator for the long tail.

## 5. Sequencing (north star → now)

| Phase | What | Status |
|---|---|---|
| **Now** | Option B generic BYO-app OAuth; Google first; on the credentials page | ✅ Built |
| **Next** | Option A first-party connectors for the **top 3** (broker in `app/`) | Proposed |
| **Then** | Decide build-vs-buy; adopt MCP/aggregator for the long tail | Proposed |
| **Layer** | Typed skills (`drive_list_files()`, `gmail_send()`) over top connections so agents stop doing raw REST | Proposed |

**Lighthouse first-party three (vote):** **Google + Microsoft + Slack** for broad business coverage —
swap Slack → **GitHub** if the ICP skews dev-fleet.

## 6. Placement decision (UI/IA)

OAuth connections are **outbound credentials** (agents reach *out*), not **inbound channels**
(platforms reaching *in*, e.g. Telegram/Slack/Webhooks). "Integrations" in this product already means
the latter. **Decision:** the connect UI lives on the **credentials page** (the `apikeys` sub-tab,
relabeled "API Keys & Connections"), presented as "Connect a service" beside manual key entry —
*not* under Channels/Webhooks.

## 7. Provider roadmap (the registry is provider-agnostic; friction is the provider, not our code)

| Tier | Providers | Effort | Notes |
|---|---|---|---|
| **Drop-in** (standard OAuth2 + refresh) | Microsoft 365 / Graph (Outlook, OneDrive, Teams), Slack, GitHub/GitLab, Notion, Linear/Jira, Dropbox/Box | Low | ~A registry row each. Notion tokens don't expire — our "no refresh → return current" path already handles it. **Microsoft is the highest-value easy win for business users.** |
| **Wrinkles** | HubSpot, Salesforce, Shopify, Stripe Connect | Medium | Salesforce returns a per-org `instance_url`; Shopify is per-store subdomain. The flexible connection dict stores the extras, but each needs a small per-provider hook. |
| **Hard** | **Meta** (Facebook/Instagram/Threads), X/Twitter | High | See §8. |

## 8. Meta — the hard case (named by the user)

Meta is very doable but is a **tier-2 build**, and the strongest argument for first-party + aggregator:

1. **Different token lifecycle.** No standard `refresh_token` grant. Short-lived user token →
   exchange for a **long-lived (~60-day) token** (`fb_exchange_token`), re-exchanged before expiry.
   Needs a **provider-specific refresh hook**, not the generic one (our architecture anticipated the
   "long-lived / no-refresh" branch, but Meta's refresh is custom logic).
2. **Heavy review.** Production permissions require **App Review + Business Verification**,
   per-capability (Pages, IG publishing, Ads, WhatsApp). Slower/stricter than Google.
3. **Asset selection.** After connecting, the agent must pick *which* Page / IG Business account /
   Ad account to act on — an extra UX step beyond the token.
4. **WhatsApp overlap.** We already have a WhatsApp *inbound channel* (webhook). OAuth into a
   WhatsApp Business account is a different thing — keep them distinct in the UI to avoid confusion.

## 9. Differentiators worth marketing

- **Connect once → the whole agent fleet can use it** (governed per-agent via `allowed_credentials`).
  Single-assistant competitors can't say that.
- **Agents only ever see a short-lived access token** — never the refresh token, never the client
  secret. A real enterprise-trust story, not just plumbing.

## 10. Open decisions / next actions

- [ ] Confirm the lighthouse-three for Option A (Google + Microsoft + Slack vs. GitHub swap).
- [ ] Build-vs-buy spike: evaluate MCP-client path and one aggregator (Nango or Composio) before #5.
- [ ] First typed skill on top of a connection (likely `gmail_send` / `drive_list_files`) to remove
      the raw-REST caveat.
- [ ] Option A broker design in `app/` (central verified apps + internal push into the engine vault).

## 11. Engineering follow-ups (from the Option-B PR review)

Tracked, non-blocking — surfaced during the end-to-end review of the shipped Option-B code:

- [ ] **MCP `$CRED{}` env injection doesn't refresh.** `resolve_cred_handles` (sync, called at
      agent spawn in `runtime.py`) returns a connection's stored token without on-demand refresh, so
      an agent that uses a connection *only* via an MCP server (never via `http_request`) could get a
      stale token. The primary `http_request` path refreshes correctly. Fix: async-resolve in the MCP
      spawn path, or pre-refresh connections before spawn.
- [ ] **System-cred denylist not re-synced at runtime.** `set_system_credential_names` is populated
      at startup; runtime `add_credential(system=True)` for `*_client_id/secret` doesn't update it, so
      `can_access_credential`/`/vault/status` can confirm a client-cred *name's existence* to an agent
      (the value stays unresolvable — no leak). Fix: register integration client-cred names into the
      permission denylist on setup. (Note: the `is_system_credential` suffix approach won't work —
      `google` isn't a known provider name.)
- [ ] **Post-connect redirect lands on `/dashboard/`** (default tab), not back on the API Keys page.
      Toast fires either way; restore the `apikeys` sub-tab via the return URL.
- [ ] **Typed connection skills** (`drive_list_files()`, `gmail_send()`) so agents stop hand-writing
      Google REST calls through `http_request`.
