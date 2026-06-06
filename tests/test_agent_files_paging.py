"""Agent ``/files/{path}`` offset/max_bytes paging.

The read endpoint gained ``offset``/``max_bytes`` so a caller (the operator
peer-file bridge, the dashboard download) can page a file larger than the
500 KB default read cap. These tests pin: (1) the default call is unchanged
(back-compat), (2) offset seeks, (3) max_bytes caps a page and reports the
right next_offset/truncated, and (4) full-file reassembly across pages is
byte-exact.
"""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def agent_client(tmp_path, monkeypatch):
    """Agent app whose /data root is redirected at the tmp dir.

    ``_safe_path`` reads ``file_tool._ALLOWED_ROOT`` on every call, so
    monkeypatching it points /files at the tmp dir without a reload.
    """
    data_root = tmp_path / "data"
    (data_root / "workspace").mkdir(parents=True)

    import src.agent.builtins.file_tool as file_tool
    monkeypatch.setattr(file_tool, "_ALLOWED_ROOT", str(data_root))

    loop = MagicMock()
    loop.workspace = MagicMock()
    loop.workspace.root = str(data_root / "workspace")
    loop.agent_id = "test-agent"
    loop.result = None
    loop.last_task_id = None
    loop.mesh_url = ""
    loop.memory = None

    import src.agent.server as agent_server_module
    importlib.reload(agent_server_module)
    app = agent_server_module.create_agent_app(loop)
    return TestClient(app), data_root


def test_default_read_is_backcompat(agent_client):
    client, data_root = agent_client
    (data_root / "workspace" / "data.md").write_text("col_a,col_b\n1,2\n")
    resp = client.get("/files/workspace/data.md")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["content"] == "col_a,col_b\n1,2\n"
    assert body["encoding"] == "utf-8"
    assert body["truncated"] is False
    assert body["size"] == len("col_a,col_b\n1,2\n")
    # New fields present and coherent.
    assert body["offset"] == 0
    assert body["next_offset"] == body["size"]


def test_offset_and_max_bytes_page(agent_client):
    client, data_root = agent_client
    payload = "0123456789abcdef"  # 16 bytes, pure ascii
    (data_root / "blob.txt").write_text(payload)

    # First page: 10 bytes.
    p1 = client.get("/files/blob.txt", params={"offset": 0, "max_bytes": 10}).json()
    assert p1["content"] == "0123456789"
    assert p1["truncated"] is True
    assert p1["next_offset"] == 10
    assert p1["size"] == 16

    # Second page from next_offset: the remainder.
    p2 = client.get(
        "/files/blob.txt", params={"offset": p1["next_offset"], "max_bytes": 10},
    ).json()
    assert p2["content"] == "abcdef"
    assert p2["truncated"] is False
    assert p2["next_offset"] == 16

    # Reassembly is byte-exact.
    assert p1["content"] + p2["content"] == payload


def test_max_bytes_clamped_to_ceiling(agent_client):
    """An absurd max_bytes is clamped to the 5 MB hard ceiling, not honored
    blindly — a small file still returns fully without error."""
    client, data_root = agent_client
    (data_root / "small.txt").write_text("hello")
    body = client.get(
        "/files/small.txt", params={"max_bytes": 999_999_999},
    ).json()
    assert body["content"] == "hello"
    assert body["truncated"] is False


def test_missing_file_404(agent_client):
    client, _ = agent_client
    resp = client.get("/files/workspace/nope.md")
    assert resp.status_code == 404
