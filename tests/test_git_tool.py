"""Unit tests for commit_file (reliable code-side base64 commit + verify)."""

import base64
import json

import pytest

from src.agent.builtins import git_tool


class FakeGitHub:
    """In-memory stand-in for the GitHub contents API via http_request."""

    def __init__(self, existing: bytes | None = None, *, verify_returns: bytes | None = "match",
                 put_status_override: int | None = None):
        self.stored = existing
        self.existing_at_start = existing
        self.put_body: dict | None = None
        self.verify_returns = verify_returns  # "match" -> echo stored; else override bytes/None
        self.put_status_override = put_status_override
        self._put_done = False

    async def __call__(self, url, method="GET", headers=None, body="", timeout=30, *, mesh_client=None):
        if method == "GET":
            # After a PUT, the read-back may be overridden to simulate a bad commit.
            data = self.stored
            if self._put_done and self.verify_returns != "match":
                data = self.verify_returns
            if data is None:
                return {"status_code": 404, "body": json.dumps({"message": "Not Found"})}
            return {"status_code": 200, "body": json.dumps({
                "sha": "oldsha",
                "encoding": "base64",
                "content": base64.b64encode(data).decode(),
            })}
        if method == "PUT":
            self.put_body = json.loads(body)
            self.stored = base64.b64decode(self.put_body["content"])
            self._put_done = True
            status = self.put_status_override or (201 if self.existing_at_start is None else 200)
            if status not in (200, 201):
                return {"status_code": status, "body": json.dumps({"message": "Unprocessable"})}
            return {"status_code": status, "body": json.dumps({
                "commit": {"html_url": "https://github.com/o/r/commit/abc"},
                "content": {"sha": "newsha"},
            })}
        return {"status_code": 400, "body": ""}


@pytest.mark.asyncio
async def test_commit_new_file_encodes_in_code_and_verifies(monkeypatch):
    fake = FakeGitHub(existing=None)
    monkeypatch.setattr(git_tool, "http_request", fake)
    csv = 'name,hook\n"Doe, John","a ""quoted"" hook"\n'  # commas + quotes
    out = await git_tool.commit_file(repo="o/r", path="data/leads.csv", message="add", content=csv)
    assert out["committed"] is True
    assert out["verified"] is True
    assert out["bytes"] == len(csv.encode())
    # The whole point: the tool base64-encoded the exact bytes (no model hand-encoding).
    assert base64.b64decode(fake.put_body["content"]) == csv.encode()
    assert "sha" not in fake.put_body  # new file -> no sha


@pytest.mark.asyncio
async def test_commit_update_includes_existing_sha(monkeypatch):
    fake = FakeGitHub(existing=b"old content")
    monkeypatch.setattr(git_tool, "http_request", fake)
    out = await git_tool.commit_file(repo="o/r", path="f.txt", message="upd", content="new")
    assert out["committed"] is True and out["verified"] is True
    assert fake.put_body["sha"] == "oldsha"


@pytest.mark.asyncio
async def test_verification_fails_when_readback_mismatches(monkeypatch):
    # Commit "succeeds" (API 201) but the read-back is truncated -> not verified.
    fake = FakeGitHub(existing=None, verify_returns=b"x")
    monkeypatch.setattr(git_tool, "http_request", fake)
    out = await git_tool.commit_file(repo="o/r", path="f.txt", message="m", content="full content")
    assert out["committed"] is True
    assert out["verified"] is False
    assert "warning" in out


@pytest.mark.asyncio
async def test_put_failure_returns_error(monkeypatch):
    fake = FakeGitHub(existing=None, put_status_override=422)
    monkeypatch.setattr(git_tool, "http_request", fake)
    out = await git_tool.commit_file(repo="o/r", path="f.txt", message="m", content="x")
    assert "error" in out
    assert "committed" not in out


@pytest.mark.asyncio
async def test_source_path_reads_file_in_code(monkeypatch, tmp_path):
    f = tmp_path / "leads.csv"
    payload = "a,b\n1,2\n" * 100
    f.write_text(payload)
    monkeypatch.setattr(git_tool, "_safe_path", lambda p: f)
    fake = FakeGitHub(existing=None)
    monkeypatch.setattr(git_tool, "http_request", fake)
    out = await git_tool.commit_file(repo="o/r", path="data/leads.csv", message="m", source_path="leads.csv")
    assert out["verified"] is True
    assert base64.b64decode(fake.put_body["content"]) == payload.encode()


@pytest.mark.asyncio
async def test_validation_errors(monkeypatch):
    monkeypatch.setattr(git_tool, "http_request", FakeGitHub())
    assert "error" in await git_tool.commit_file(repo="bad", path="f", message="m", content="x")
    assert "error" in await git_tool.commit_file(repo="o/r", path="f", message="m")  # no content/source
    assert "error" in await git_tool.commit_file(
        repo="o/r", path="f", message="m", content="x", source_path="y",
    )  # both


@pytest.mark.asyncio
async def test_oversize_rejected(monkeypatch):
    monkeypatch.setattr(git_tool, "http_request", FakeGitHub())
    big = "x" * (git_tool._MAX_COMMIT_BYTES + 1)
    out = await git_tool.commit_file(repo="o/r", path="f", message="m", content=big)
    assert "error" in out and "committed" not in out
