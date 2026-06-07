"""Unit tests for commit_file (reliable code-side base64 commit + verify)."""

import base64
import json

import pytest

from src.agent.builtins import git_tool


class FakeGitHub:
    """In-memory stand-in for the GitHub contents API via http_request.

    Models the two behaviours that matter: the contents GET returns the file's
    blob sha (and, for files, the base64 content — which http_request truncates
    at ~50KB), and the PUT returns ``content.sha`` = the git blob sha of what
    was stored (which commit_file verifies against a locally-computed sha).
    """

    def __init__(self, existing: bytes | None = None, *, put_status_override: int | None = None,
                 corrupt_sha: bool = False, get_truncated: bool = False):
        self.stored = existing
        self.existing_at_start = existing
        self.put_body: dict | None = None
        self.put_status_override = put_status_override
        self.corrupt_sha = corrupt_sha
        self.get_truncated = get_truncated

    async def __call__(self, url, method="GET", headers=None, body="", timeout=30, *, mesh_client=None):
        if method == "GET":
            if self.stored is None:
                return {"status_code": 404, "body": json.dumps({"message": "Not Found"})}
            full = json.dumps({
                "sha": git_tool._git_blob_sha(self.stored),
                "size": len(self.stored),
                "encoding": "base64",
                "content": base64.b64encode(self.stored).decode(),
            })
            if self.get_truncated:
                full = full[:120]  # simulate http_request's ~50KB body cap mid-content
            return {"status_code": 200, "body": full}
        if method == "PUT":
            self.put_body = json.loads(body)
            self.stored = base64.b64decode(self.put_body["content"])
            status = self.put_status_override or (201 if self.existing_at_start is None else 200)
            if status not in (200, 201):
                return {"status_code": status, "body": json.dumps({"message": "Unprocessable"})}
            sha = "0" * 40 if self.corrupt_sha else git_tool._git_blob_sha(self.stored)
            return {"status_code": status, "body": json.dumps({
                "commit": {"html_url": "https://github.com/o/r/commit/abc"},
                "content": {"sha": sha, "size": len(self.stored)},
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
    assert fake.put_body["sha"] == git_tool._git_blob_sha(b"old content")


@pytest.mark.asyncio
async def test_update_sha_survives_truncated_get(monkeypatch):
    # The motivating case: a large EXISTING file. http_request truncates the
    # GET body, so full JSON won't parse — but the sha must still be recovered
    # or the PUT omits it and GitHub 422s. Regression guard for that must-fix.
    existing = b"x" * 100_000
    fake = FakeGitHub(existing=existing, get_truncated=True)
    monkeypatch.setattr(git_tool, "http_request", fake)
    out = await git_tool.commit_file(repo="o/r", path="data/big.csv", message="upd", content="new rows")
    assert out["committed"] is True and out["verified"] is True
    assert fake.put_body["sha"] == git_tool._git_blob_sha(existing)


@pytest.mark.asyncio
async def test_verification_fails_when_github_reports_wrong_sha(monkeypatch):
    fake = FakeGitHub(existing=None, corrupt_sha=True)
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
async def test_blob_sha_matches_git_convention():
    # git hash-object of an empty blob is the well-known constant.
    assert git_tool._git_blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


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
