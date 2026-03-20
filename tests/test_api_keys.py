"""Unit tests for the ApiKeyManager."""

import json
import os
import time

import pytest

from src.host.api_keys import ApiKeyManager, _hash_key


@pytest.fixture
def manager(tmp_path):
    return ApiKeyManager(config_path=str(tmp_path / "api_keys.json"))


def test_create_key(manager):
    key_id, raw_key = manager.create_key("test-key")
    assert key_id.startswith("ak_")
    assert len(raw_key) > 20
    assert len(manager.keys) == 1
    assert manager.keys[key_id]["name"] == "test-key"


def test_create_key_persists(tmp_path):
    path = str(tmp_path / "api_keys.json")
    m = ApiKeyManager(config_path=path)
    m.create_key("persisted")
    data = json.loads((tmp_path / "api_keys.json").read_text())
    assert len(data["keys"]) == 1


def test_create_key_empty_name_raises(manager):
    with pytest.raises(ValueError, match="must not be empty"):
        manager.create_key("")


def test_create_key_long_name_raises(manager):
    with pytest.raises(ValueError, match="128 characters"):
        manager.create_key("x" * 129)


def test_authenticate_valid_key(manager):
    key_id, raw_key = manager.create_key("auth-test")
    result = manager.authenticate(raw_key)
    assert result is not None
    assert result["id"] == key_id
    assert result["name"] == "auth-test"


def test_authenticate_invalid_key(manager):
    manager.create_key("auth-test")
    assert manager.authenticate("wrong-key") is None


def test_authenticate_empty_key(manager):
    assert manager.authenticate("") is None


def test_salted_hash(manager):
    """Same raw key with different key_ids produces different hashes."""
    raw = "same-raw-key"
    h1 = _hash_key("ak_aaa", raw)
    h2 = _hash_key("ak_bbb", raw)
    assert h1 != h2


def test_authenticate_updates_last_used(manager):
    key_id, raw_key = manager.create_key("usage-test")
    assert manager.keys[key_id]["last_used_at"] is None
    manager.authenticate(raw_key)
    assert manager.keys[key_id]["last_used_at"] is not None


def test_revoke_key(manager):
    key_id, raw_key = manager.create_key("revoke-test")
    assert manager.revoke_key(key_id) is True
    assert key_id not in manager.keys
    assert manager.authenticate(raw_key) is None


def test_revoke_nonexistent(manager):
    assert manager.revoke_key("ak_nonexistent") is False


def test_list_keys(manager):
    manager.create_key("key-a")
    manager.create_key("key-b")
    keys = manager.list_keys()
    assert len(keys) == 2
    names = {k["name"] for k in keys}
    assert names == {"key-a", "key-b"}
    for k in keys:
        assert "key_hash" not in k
        assert "id" in k
        assert "created_at" in k


def test_legacy_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLEGION_API_KEY", "legacy-secret-123")
    m = ApiKeyManager(config_path=str(tmp_path / "api_keys.json"))
    result = m.authenticate("legacy-secret-123")
    assert result is not None
    assert result["id"] == "_legacy"
    assert result["name"] == "legacy (env var)"


def test_legacy_env_var_wrong_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLEGION_API_KEY", "legacy-secret-123")
    m = ApiKeyManager(config_path=str(tmp_path / "api_keys.json"))
    assert m.authenticate("wrong") is None


def test_named_key_takes_priority_over_legacy(tmp_path, monkeypatch):
    """Named keys are checked before the legacy env var."""
    monkeypatch.setenv("OPENLEGION_API_KEY", "legacy-key")
    m = ApiKeyManager(config_path=str(tmp_path / "api_keys.json"))
    _key_id, raw_key = m.create_key("named")
    result = m.authenticate(raw_key)
    assert result["id"] != "_legacy"


def test_has_keys_with_named(manager, monkeypatch):
    monkeypatch.delenv("OPENLEGION_API_KEY", raising=False)
    assert manager.has_keys() is False
    manager.create_key("test")
    assert manager.has_keys() is True


def test_has_keys_with_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLEGION_API_KEY", "something")
    m = ApiKeyManager(config_path=str(tmp_path / "api_keys.json"))
    assert m.has_keys() is True


def test_has_keys_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENLEGION_API_KEY", raising=False)
    m = ApiKeyManager(config_path=str(tmp_path / "api_keys.json"))
    assert m.has_keys() is False


def test_buffered_writes(manager):
    """last_used_at is buffered in memory, not flushed on every auth."""
    key_id, raw_key = manager.create_key("buffer-test")
    initial_mtime = os.path.getmtime(manager.config_path)
    time.sleep(0.05)
    manager.authenticate(raw_key)
    assert manager._dirty is True
    # File should not have been rewritten (flush interval not reached)
    assert os.path.getmtime(manager.config_path) == initial_mtime


def test_flush_writes_to_disk(manager):
    key_id, raw_key = manager.create_key("flush-test")
    manager.authenticate(raw_key)
    assert manager._dirty is True
    manager.flush()
    assert manager._dirty is False
    data = json.loads(manager.config_path.read_text())
    assert data["keys"][key_id]["last_used_at"] is not None


def test_reload_from_disk(tmp_path):
    path = str(tmp_path / "api_keys.json")
    m1 = ApiKeyManager(config_path=path)
    _key_id, raw_key = m1.create_key("reload-test")
    m2 = ApiKeyManager(config_path=path)
    result = m2.authenticate(raw_key)
    assert result is not None
    assert result["name"] == "reload-test"


def test_multiple_keys_auth(manager):
    """Multiple keys can coexist and each authenticates independently."""
    _id1, key1 = manager.create_key("key-1")
    _id2, key2 = manager.create_key("key-2")
    r1 = manager.authenticate(key1)
    r2 = manager.authenticate(key2)
    assert r1["name"] == "key-1"
    assert r2["name"] == "key-2"
    assert r1["id"] != r2["id"]
