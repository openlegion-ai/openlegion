# One-Click Integrations — OAuth "Connect" Flow

**Date:** 2026-06-04
**Status:** ACTIVE — proposed.
**Scope:** Replace the manual "go fetch a token and paste it" flow for third-party data
integrations with an in-engine OAuth authorization-code flow. Ship **Option B** (bring-your-own
OAuth app, engine handles the dance + refresh) first, with **Google** (Drive / Gmail / Calendar)
as the first provider. Keep the design provider-agnostic and broker-ready so **Option A**
(first-party verified connectors, zero user setup) can layer on later.

---

## 1. Problem

Today, connecting a third-party service (Google Drive, Meta, Slack) means the user acts as their
own OAuth-app developer:

1. Create a project in the provider's dev console, enable APIs, mint a token by hand.
2. Paste it into the dashboard → `POST /api/credentials` → stored as `OPENLEGION_CRED_<name>`.
3. Agents later reference it as `$CRED{name}`, resolved by the mesh (`http_tool.py:_resolve_creds`
   → `mesh_client.vault_resolve` → `CredentialVault.resolve_credential`); the raw secret is never
   exposed to the agent and is redacted from logs/output.

There is **no OAuth flow** for these providers and **no refresh** — a hand-pasted token that expires
forces the user to repeat the whole dance.

The irony: `src/host/credentials.py` already implements a **full authorization-code flow with
refresh + expiry handling** — but only for the OpenAI LLM endpoint
(`_ensure_openai_oauth_token` → `POST https://auth.openai.com/oauth/token`,
`grant_type=refresh_token`, 5-min expiry buffer, atomic re-persist). Token storage, refresh,
expiry, and redaction are solved. What's missing:

- A **callback endpoint** on the mesh host (the code even references the OpenAI redirect URI
  `http://localhost:1455/auth/callback`, which lives outside this codebase).
- A **per-provider OAuth client registry** for data providers (not LLM providers).
- A vault concept for **structured, auto-refreshing connections** that still resolve as
  `$CRED{...}` bearer tokens.

---

## 2. Design overview

```
Dashboard  ──Connect──▶  GET /integrations/google/connect
                              │  build state (CSRF, single-use, TTL, session-bound)
                              │  build authorize URL (access_type=offline, prompt=consent)
                              ▼
                         302 → accounts.google.com consent screen
                              │  user approves
                              ▼
Google  ──redirect──▶  GET /integrations/google/callback?code&state
                              │  validate state
                              │  exchange code → {access, refresh, expiry, scopes, email}
                              │  store as structured connection in vault
                              ▼
                         302 → dashboard /#integrations?connected=google

Agent later:  Authorization: Bearer $CRED{google_drive}
                 └─ vault resolves → ensures fresh token (refresh if near expiry) → bearer
```

The key elegance: agents keep using `$CRED{google_drive}` exactly as today. The vault transparently
refreshes the access token on resolve, so connections never go stale and the agent never sees the
refresh token.

---

## 3. Components

### 3.1 Provider registry — `src/host/oauth_providers.py` (new)

Provider-agnostic, declarative. No user-supplied URLs (SSRF-safe: endpoints are fixed per provider).

```python
@dataclass(frozen=True)
class OAuthProvider:
    key: str                       # "google"
    label: str                     # "Google"
    authorize_url: str             # https://accounts.google.com/o/oauth2/v2/auth
    token_url: str                 # https://oauth2.googleapis.com/token
    scopes: dict[str, list[str]]   # named scope bundles: drive_ro, gmail_send, calendar, ...
    extra_authorize_params: dict   # {"access_type": "offline", "prompt": "consent"}
    userinfo_url: str | None       # for display: which account is connected
    uses_pkce: bool = True
```

`GOOGLE` entry ships first. `scopes` lets the operator pick least-privilege bundles
(e.g. `drive.readonly` vs `drive`). `access_type=offline` + `prompt=consent` are required to
guarantee Google returns a refresh token.

### 3.2 BYO client credentials (Option B)

The user registers their **own** Google OAuth app once and supplies:

- `OPENLEGION_SYSTEM_GOOGLE_CLIENT_ID`
- `OPENLEGION_SYSTEM_GOOGLE_CLIENT_SECRET`

Stored **system-tier** (mesh-only, never agent-resolvable, redacted via `deep_redact`). Entered
through a small "Set up provider" form in the dashboard (writes via the existing system-tier
`add_credential` path).

The user adds this exact redirect URI to their Google app:
`https://{their-subdomain}.engine.openlegion.ai/integrations/google/callback`

Requires a new config value `OPENLEGION_PUBLIC_BASE_URL` (the engine's externally reachable origin)
so the callback URL is built correctly. Document the one-time setup in the dashboard UI.

### 3.3 Structured connections in the vault — `src/host/credentials.py`

Add a `connections: dict[str, dict]` store, mirroring `_openai_oauth`'s shape:

```json
{
  "provider": "google",
  "access_token": "...",
  "refresh_token": "...",
  "expires_at": 1733300000,
  "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
  "account": "user@example.com"
}
```

- `store_connection(name, data)` — persist atomically to `.env` (reuse `_persist_to_env`), keyed
  e.g. `OPENLEGION_CONN_<NAME>` as a JSON blob (new tier, distinct from CRED/SYSTEM).
- `ensure_connection_token(name)` — generalize `_ensure_openai_oauth_token`: if within 5-min of
  `expires_at`, `POST token_url` with `grant_type=refresh_token` + client_id/secret, update + persist.
- **Resolve path:** when `resolve_credential(name)` hits a connection, return
  `ensure_connection_token(name)` (a fresh bearer) instead of a static secret. This is what makes
  `$CRED{google_drive}` "just work" for agents, with auto-refresh, no agent exposure.

Permissioning: connections respect the same `allowed_credentials` glob patterns agents already use,
so an agent must be granted `google_*` (or similar) to resolve them.

### 3.4 Mesh endpoints — `src/host/server.py`

- `GET /integrations` — list providers + connection status (dashboard-auth gated).
- `GET /integrations/{provider}/connect?name=&scopes=` — build `state`, redirect (302) to the
  provider authorize URL. Operator/dashboard-auth gated.
- `GET /integrations/{provider}/callback?code=&state=` — validate `state`, exchange code, store
  connection, 302 back to the dashboard. (Reaches the user's browser carrying the `ol_session`
  cookie, so it passes Caddy `forward_auth`; `state` is the real CSRF guard.)
- `POST /integrations/{provider}/disconnect` — revoke (best-effort) + delete connection.

`state` store: single-use, TTL-bound (~10 min), bound to the dashboard session and to the requested
`{name, scopes}`. Persist in SQLite (WAL) or in-memory with TTL — must be validated before any code
exchange. Reuse the CSRF/`X-Requested-With` conventions already in the dashboard layer.

### 3.5 Dashboard UI — `src/dashboard/server.py` + templates

Turn the credentials panel into an **Integrations** list:

- Per provider: status chip (Connected as `user@…`, scopes, expiry) or **Connect** button.
- Connect → full-page navigation to `/integrations/google/connect` (returns to dashboard on success).
- First-run per provider: "Set up your Google app" form (client_id/secret + a copyable redirect URI
  and a link to the provider console steps).
- Disconnect button → `POST /integrations/google/disconnect`.

Keep the existing raw paste-a-token path available as an "Advanced / manual" fallback.

---

## 4. Security checklist

- `state`: unguessable, single-use, TTL-bound, session-bound; validated before code exchange.
- `client_secret` + all tokens: system/connection tier, never agent-resolvable, run through
  `deep_redact` / `redact_url` in every log + response path.
- Fixed per-provider endpoints (no user-supplied token/authorize URLs) → no SSRF surface added.
- PKCE on by default where the provider supports it.
- Least-privilege scopes: operator selects scope bundles; default to read-only where it exists.
- Refresh-on-resolve returns only a short-lived access token to the agent; refresh token stays in
  the vault.
- Disconnect attempts provider-side revocation, then purges local tokens + `.env` entry.

## 5. Option A readiness (later)

The registry is already provider-agnostic. First-party connectors differ only in:

1. client_id/secret come from **central** config instead of BYO.
2. The consent dance runs through a **broker in `app/`** (central verified redirect URI), which then
   pushes the resulting connection into the engine vault via an internal,
   `x-mesh-internal`-authenticated endpoint.

So: make `store_connection` callable from **both** the local callback (Option B) and an internal
push endpoint (Option A). No engine rework needed when Option A lands — only Google/Meta app
verification + the broker.

---

## 6. Work breakdown (Option B, Google)

1. `oauth_providers.py` registry + `GOOGLE` entry + scope bundles.
2. `credentials.py`: connection store, `store_connection`, `ensure_connection_token`
   (generalized from `_ensure_openai_oauth_token`), resolve-path hook, `.env` persistence + redaction.
3. `OPENLEGION_PUBLIC_BASE_URL` config + redirect-URI builder.
4. `server.py`: `/integrations` list, `/connect`, `/callback`, `/disconnect` + `state` store.
5. Dashboard: Integrations panel, provider-setup form, connect/disconnect, status display.
6. Tests: mocked token exchange, state validation (reuse/expiry/forgery), refresh-on-resolve,
   redaction, permission gating via `allowed_credentials`.
7. Docs: per-provider setup steps (redirect URI, scopes) surfaced in-UI.

---

## 7. Open questions

- **`state` store backing:** new tiny SQLite table vs in-memory dict with TTL? (Leaning SQLite WAL
  to match the rest of the host and survive restart mid-flow.)
- **Connection naming:** auto-name (`google_drive`) vs operator-chosen handle? Auto with override.
- **Multi-account:** support multiple Google connections (work + personal) under distinct names?
- **`OPENLEGION_PUBLIC_BASE_URL`:** derive from the SSO subdomain config if one already exists,
  else require explicit config.
