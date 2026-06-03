"""HTTP-level tests for the OAuth integrations dashboard endpoints.

Exercises the connect/callback/setup/disconnect wiring through the real
dashboard router + TestClient: state minting + single-use/session-bound
consumption, the 302 redirects (success and every error branch), and that
setup persists client creds system-tier. The token exchange is mocked — no
network. (Vault-level refresh/store logic is unit-tested in
``tests/test_oauth_integrations.py``.)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from unittest import mock
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.host.credentials as cred_mod
from src.dashboard.events import EventBus
from src.host.costs import CostTracker
from src.host.credentials import CredentialVault
from src.host.health import HealthMonitor
from src.host.mesh import Blackboard
from src.host.traces import TraceStore


class _CSRFTestClient(TestClient):
    def request(self, method, url, **kwargs):
        if method.upper() not in ("GET", "HEAD", "OPTIONS"):
            headers = kwargs.get("headers") or {}
            headers.setdefault("X-Requested-With", "XMLHttpRequest")
            kwargs["headers"] = headers
        return super().request(method, url, **kwargs)


def _make_components(tmp_path: str) -> dict:
    bb = Blackboard(db_path=os.path.join(tmp_path, "bb.db"))
    cost_tracker = CostTracker(db_path=os.path.join(tmp_path, "costs.db"))
    trace_store = TraceStore(db_path=os.path.join(tmp_path, "traces.db"))
    runtime_mock = MagicMock()
    runtime_mock.browser_service_url = None
    health_monitor = HealthMonitor(
        runtime=runtime_mock, transport=MagicMock(), router=MagicMock(),
    )
    return {
        "blackboard": bb,
        "health_monitor": health_monitor,
        "cost_tracker": cost_tracker,
        "trace_store": trace_store,
        "event_bus": EventBus(),
        "agent_registry": {},
        "runtime": runtime_mock,
    }


class TestIntegrationEndpoints:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        # No real .env writes from setup/store/remove.
        self._p = mock.patch.object(cred_mod, "_persist_to_env", lambda *a, **k: None)
        self._r = mock.patch.object(cred_mod, "_remove_from_env", lambda *a, **k: None)
        self._p.start()
        self._r.start()
        self.vault = CredentialVault()
        # Start clean so ambient OPENLEGION_* env doesn't leak into assertions.
        self.vault.system_credentials.pop("google_client_id", None)
        self.vault.system_credentials.pop("google_client_secret", None)
        self.vault.connections.clear()
        self.components = _make_components(self._tmpdir)
        self.components["credential_vault"] = self.vault
        router = __import__(
            "src.dashboard.server", fromlist=["create_dashboard_router"],
        ).create_dashboard_router(**self.components, mesh_port=8420)
        app = FastAPI()
        app.include_router(router)
        self.client = _CSRFTestClient(app)

    def teardown_method(self):
        self._p.stop()
        self._r.stop()
        self.components["cost_tracker"].close()
        self.components["trace_store"].close()
        self.components["blackboard"].close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _setup_google(self):
        return self.client.post(
            "/dashboard/api/integrations/google/setup",
            json={"client_id": "cid.apps.googleusercontent.com", "client_secret": "GOCSPX-x"},
        )

    # ── list + setup ────────────────────────────────────────────────────

    def test_list_unconfigured(self):
        resp = self.client.get("/dashboard/api/integrations")
        assert resp.status_code == 200
        providers = resp.json()["providers"]
        google = next(p for p in providers if p["key"] == "google")
        assert google["configured"] is False
        assert google["connections"] == []
        assert any(b["key"] == "drive_readonly" for b in google["scope_bundles"])
        assert google["redirect_uri"].endswith("/dashboard/integrations/google/callback")

    def test_setup_then_configured(self):
        assert self._setup_google().status_code == 200
        assert self.vault.system_credentials["google_client_id"] == "cid.apps.googleusercontent.com"
        google = next(
            p for p in self.client.get("/dashboard/api/integrations").json()["providers"]
            if p["key"] == "google"
        )
        assert google["configured"] is True

    def test_setup_missing_fields_400(self):
        resp = self.client.post(
            "/dashboard/api/integrations/google/setup",
            json={"client_id": "x", "client_secret": ""},
        )
        assert resp.status_code == 400

    def test_setup_unknown_provider_404(self):
        resp = self.client.post(
            "/dashboard/api/integrations/dropbox/setup",
            json={"client_id": "x", "client_secret": "y"},
        )
        assert resp.status_code == 404

    # ── connect ─────────────────────────────────────────────────────────

    def test_connect_requires_client(self):
        resp = self.client.get(
            "/dashboard/integrations/google/connect", follow_redirects=False,
        )
        assert resp.status_code == 400  # client not configured

    def test_connect_redirects_to_provider(self):
        self._setup_google()
        resp = self.client.get(
            "/dashboard/integrations/google/connect?name=mydrive&scopes=drive_readonly",
            follow_redirects=False,
        )
        assert resp.status_code == 302
        loc = resp.headers["location"]
        assert loc.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
        q = parse_qs(urlparse(loc).query)
        assert q["state"][0]
        assert "code_challenge" in q
        assert q["code_challenge_method"][0] == "S256"
        assert "drive.readonly" in q["scope"][0]
        assert q["access_type"][0] == "offline"

    def test_connect_unknown_provider_404(self):
        resp = self.client.get(
            "/dashboard/integrations/dropbox/connect", follow_redirects=False,
        )
        assert resp.status_code == 404

    # ── callback ────────────────────────────────────────────────────────

    def _mint_state(self) -> str:
        self._setup_google()
        resp = self.client.get(
            "/dashboard/integrations/google/connect?name=mydrive&scopes=drive_readonly",
            follow_redirects=False,
        )
        return parse_qs(urlparse(resp.headers["location"]).query)["state"][0]

    def test_callback_invalid_state(self):
        resp = self.client.get(
            "/dashboard/integrations/google/callback?code=abc&state=bogus",
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "integration_error=invalid_state" in resp.headers["location"]

    def test_callback_provider_error(self):
        resp = self.client.get(
            "/dashboard/integrations/google/callback?error=access_denied",
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "integration_error=access_denied" in resp.headers["location"]

    def test_callback_happy_path_stores_connection(self):
        state = self._mint_state()
        self.vault.exchange_oauth_code = AsyncMock(return_value={
            "provider": "google", "access_token": "at", "refresh_token": "rt",
            "expires_at": int(time.time()) + 3600, "scopes": ["s"], "account": "u@e.com",
        })
        resp = self.client.get(
            f"/dashboard/integrations/google/callback?code=authcode&state={state}",
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "integration_connected=mydrive" in resp.headers["location"]
        assert "mydrive" in self.vault.connections
        # state is single-use — replaying the same state now fails
        replay = self.client.get(
            f"/dashboard/integrations/google/callback?code=authcode&state={state}",
            follow_redirects=False,
        )
        assert "integration_error=invalid_state" in replay.headers["location"]

    def test_callback_no_refresh_token_rejected(self):
        # Google needs a refresh token; a connection without one would die at
        # first expiry, so the callback must reject it rather than store it.
        state = self._mint_state()
        self.vault.exchange_oauth_code = AsyncMock(return_value={
            "provider": "google", "access_token": "at", "refresh_token": "",
            "expires_at": int(time.time()) + 3600, "scopes": ["s"], "account": "",
        })
        resp = self.client.get(
            f"/dashboard/integrations/google/callback?code=c&state={state}",
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "integration_error=no_refresh_token" in resp.headers["location"]
        assert "mydrive" not in self.vault.connections

    def test_connect_unknown_scope_bundle_400(self):
        self._setup_google()
        resp = self.client.get(
            "/dashboard/integrations/google/connect?scopes=bogus_bundle",
            follow_redirects=False,
        )
        assert resp.status_code == 400

    def test_callback_exchange_failure_redirects(self):
        state = self._mint_state()
        self.vault.exchange_oauth_code = AsyncMock(side_effect=RuntimeError("bad_grant"))
        resp = self.client.get(
            f"/dashboard/integrations/google/callback?code=authcode&state={state}",
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "integration_error=exchange_failed" in resp.headers["location"]
        assert "mydrive" not in self.vault.connections

    # ── disconnect ──────────────────────────────────────────────────────

    def test_disconnect(self):
        self.vault.connections["mydrive"] = {
            "provider": "google", "access_token": "at", "refresh_token": "rt",
        }
        resp = self.client.post("/dashboard/api/integrations/mydrive/disconnect")
        assert resp.status_code == 200
        assert "mydrive" not in self.vault.connections

    def test_disconnect_missing_404(self):
        resp = self.client.post("/dashboard/api/integrations/nope/disconnect")
        assert resp.status_code == 404
