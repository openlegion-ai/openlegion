"""Named API key management for external integrations.

Stores salted SHA-256 hashes of API keys in ``config/api_keys.json``.
Raw keys are returned once at creation time and never stored.

Falls back to the ``OPENLEGION_API_KEY`` env var for backward compatibility
with single-key deployments.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import os
import secrets
import time
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("host.api_keys")

_FLUSH_INTERVAL = 60  # seconds between last_used_at disk flushes


def _hash_key(key_id: str, raw_key: str) -> str:
    """Salted SHA-256: ``sha256(key_id + raw_key)``."""
    return hashlib.sha256((key_id + raw_key).encode()).hexdigest()


class ApiKeyManager:
    """Manages named API keys with salted hashes and buffered writes."""

    def __init__(self, config_path: str = "config/api_keys.json") -> None:
        self.config_path = Path(config_path)
        self.keys: dict[str, dict] = {}
        self._dirty = False
        self._last_flush = time.monotonic()
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            return
        try:
            data = json.loads(self.config_path.read_text())
            self.keys = data.get("keys", {})
        except Exception as e:
            logger.warning("Failed to load API keys config: %s", e)

    def _save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps({"keys": self.keys}, indent=2) + "\n",
        )
        self._dirty = False
        self._last_flush = time.monotonic()

    def _maybe_flush(self) -> None:
        """Flush to disk if dirty and enough time has passed."""
        if self._dirty and time.monotonic() - self._last_flush >= _FLUSH_INTERVAL:
            self._save()

    def create_key(self, name: str) -> tuple[str, str]:
        """Create a named API key.

        Returns ``(key_id, raw_key)``.  The raw key is never stored —
        only its salted hash.
        """
        name = name.strip()
        if not name:
            raise ValueError("Key name must not be empty")
        if len(name) > 128:
            raise ValueError("Key name must be 128 characters or fewer")
        key_id = "ak_" + secrets.token_hex(6)
        raw_key = secrets.token_urlsafe(32)
        self.keys[key_id] = {
            "name": name,
            "key_hash": _hash_key(key_id, raw_key),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "last_used_at": None,
        }
        self._save()
        logger.info("API key created: %s (%s)", key_id, name)
        return key_id, raw_key

    def revoke_key(self, key_id: str) -> bool:
        """Revoke a key by ID.  Returns True if it existed."""
        if key_id not in self.keys:
            return False
        name = self.keys[key_id].get("name", "")
        del self.keys[key_id]
        self._save()
        logger.info("API key revoked: %s (%s)", key_id, name)
        return True

    def list_keys(self) -> list[dict]:
        """Return metadata for all keys (never hashes)."""
        return [
            {
                "id": kid,
                "name": meta["name"],
                "created_at": meta["created_at"],
                "last_used_at": meta.get("last_used_at"),
            }
            for kid, meta in self.keys.items()
        ]

    def authenticate(self, raw_key: str) -> dict | None:
        """Authenticate a raw key.  Returns key metadata or None.

        Checks named keys first (salted hash comparison), then falls
        back to the ``OPENLEGION_API_KEY`` env var for legacy compat.
        Updates ``last_used_at`` in memory (flushed periodically).
        """
        if not raw_key:
            return None

        for kid, meta in self.keys.items():
            expected = meta.get("key_hash", "")
            candidate = _hash_key(kid, raw_key)
            if _hmac.compare_digest(candidate, expected):
                meta["last_used_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
                )
                self._dirty = True
                self._maybe_flush()
                return {"id": kid, "name": meta["name"]}

        legacy = os.environ.get("OPENLEGION_API_KEY", "")
        if legacy and _hmac.compare_digest(raw_key, legacy):
            return {"id": "_legacy", "name": "legacy (env var)"}

        return None

    def has_keys(self) -> bool:
        """True if any named keys exist or the legacy env var is set."""
        if self.keys:
            return True
        return bool(os.environ.get("OPENLEGION_API_KEY", ""))

    def flush(self) -> None:
        """Force-flush buffered writes to disk."""
        if self._dirty:
            self._save()
