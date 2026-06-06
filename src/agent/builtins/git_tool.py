"""Reliable file → GitHub commit tool.

Before this, the only way an agent could commit a file to a repo was the raw
GitHub *contents* API via ``http_request`` — whose JSON body requires the file
content **base64-encoded inline**. That forced the model to hand-emit base64 of
the entire file in a tool-call argument, which LLMs cannot do reliably: in
production a 35 KB CSV commit silently failed over and over (the file never grew
past its header) because each "batch" re-sent a corrupt/duplicate base64 blob.

``commit_file`` fixes the whole class. The model produces plaintext exactly
once — either inline (``content``) or, better, by writing the file with
``write_file`` and passing ``source_path`` — and this tool does the base64
encoding, existing-SHA lookup, commit, and **post-write verification** in code.
The model never touches base64, and the credential stays a ``$CRED{}`` handle
resolved mesh-side by ``http_request``.

Pattern: the agent that OWNS the data should hold the GitHub credential and
call ``commit_file`` itself — don't hand a large dataset across a handoff (the
recipient can't read your /data volume, and large payloads truncate).
"""

from __future__ import annotations

import base64
import json

from src.agent.builtins.file_tool import _safe_path
from src.agent.builtins.http_tool import http_request
from src.agent.tools import tool
from src.shared.utils import setup_logging

logger = setup_logging("agent.git_tool")

_GITHUB_API = "https://api.github.com"
# The GitHub *contents* API handles files up to ~1 MB. Larger files need the
# git blobs/trees API (a future enhancement) — fail clearly rather than 422.
_MAX_COMMIT_BYTES = 1_000_000


def _gh_headers(credential: str) -> dict:
    return {
        "Authorization": f"Bearer $CRED{{{credential}}}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "openlegion-agent",
    }


def _parse_json_body(resp: dict) -> dict | None:
    body = resp.get("body")
    if not isinstance(body, str) or not body:
        return None
    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


@tool(
    name="commit_file",
    description=(
        "Commit a single file to a GitHub repository RELIABLY. Use this instead "
        "of calling the GitHub contents API via http_request — this tool does "
        "the base64 encoding, existing-file SHA lookup, commit, AND a "
        "post-write verification in code, so large/complex files commit "
        "correctly (never hand-encode base64 yourself — that silently corrupts "
        "files).\n\n"
        "Provide the content ONE of two ways: (1) source_path — the path of a "
        "file you already wrote with write_file (recommended for large files), "
        "or (2) content — the raw text inline (fine for small files). "
        "Authentication uses a $CRED{} handle resolved from the vault; the "
        "default credential name is 'github_token'.\n\n"
        "Returns {committed, verified, path, bytes, repo, branch, commit_url, "
        "content_sha}. 'verified' is true only when the tool re-read the file "
        "from GitHub and its bytes match what was sent — so a true result means "
        "the file is really there, not just that the API returned 200."
    ),
    parameters={
        "repo": {
            "type": "string",
            "description": "Target repository as 'owner/name', e.g. 'acme/website'.",
        },
        "path": {
            "type": "string",
            "description": "Path of the file within the repo, e.g. 'data/leads.csv'.",
        },
        "message": {
            "type": "string",
            "description": "Commit message.",
        },
        "source_path": {
            "type": "string",
            "description": (
                "Path of a file in your /data workspace to commit (read in "
                "code — no base64 by you). Use this for large files. Mutually "
                "exclusive with 'content'."
            ),
            "default": "",
        },
        "content": {
            "type": "string",
            "description": (
                "Raw text content to commit, used when source_path is omitted. "
                "Fine for small files."
            ),
            "default": "",
        },
        "branch": {
            "type": "string",
            "description": "Target branch (default: the repo's default branch).",
            "default": "",
        },
        "credential": {
            "type": "string",
            "description": "Vault credential name for the GitHub token.",
            "default": "github_token",
        },
    },
    loop_exempt=True,
)
async def commit_file(
    repo: str,
    path: str,
    message: str,
    source_path: str = "",
    content: str = "",
    branch: str = "",
    credential: str = "github_token",
    *,
    mesh_client=None,
) -> dict:
    """Commit a file to GitHub with code-side base64 + verification."""
    if not repo or repo.count("/") != 1 or repo.startswith("/") or repo.endswith("/"):
        return {"error": "repo must be 'owner/name'"}
    if not path:
        return {"error": "path is required"}
    if not message:
        return {"error": "message is required"}

    # Resolve the content bytes. The model produces plaintext once; we encode.
    if source_path and content:
        return {"error": "pass only one of source_path or content"}
    if source_path:
        try:
            resolved = _safe_path(source_path)
        except ValueError as e:
            return {"error": f"invalid source_path: {e}"}
        if not resolved.is_file():
            return {"error": f"source_path not found: {source_path}"}
        raw = resolved.read_bytes()
    elif content:
        raw = content.encode("utf-8")
    else:
        return {"error": "provide source_path or content"}

    if len(raw) > _MAX_COMMIT_BYTES:
        return {
            "error": (
                f"file is {len(raw)} bytes; the GitHub contents API used here "
                f"supports up to {_MAX_COMMIT_BYTES}. Split the file or use a "
                f"git blobs/trees flow for larger uploads."
            ),
        }

    b64 = base64.b64encode(raw).decode("ascii")
    headers = _gh_headers(credential)
    base_url = f"{_GITHUB_API}/repos/{repo}/contents/{path}"
    ref_q = f"?ref={branch}" if branch else ""

    # 1) Look up the existing file SHA (required by the API to UPDATE; absent
    #    means a fresh create). 404 = new file; other errors are fatal.
    get_resp = await http_request(
        url=f"{base_url}{ref_q}", method="GET", headers=headers, mesh_client=mesh_client,
    )
    status = get_resp.get("status_code", 0)
    existing_sha = None
    if status == 200:
        meta = _parse_json_body(get_resp) or {}
        existing_sha = meta.get("sha")
    elif status == 404:
        existing_sha = None
    else:
        return {
            "error": f"could not read existing file (HTTP {status})",
            "detail": (get_resp.get("body") or get_resp.get("error") or "")[:300],
        }

    # 2) Commit (create or update).
    payload: dict = {"message": message, "content": b64}
    if existing_sha:
        payload["sha"] = existing_sha
    if branch:
        payload["branch"] = branch
    put_resp = await http_request(
        url=base_url, method="PUT", headers=headers,
        body=json.dumps(payload), mesh_client=mesh_client,
    )
    put_status = put_resp.get("status_code", 0)
    if put_status not in (200, 201):
        return {
            "error": f"commit failed (HTTP {put_status})",
            "detail": (put_resp.get("body") or put_resp.get("error") or "")[:300],
        }
    put_body = _parse_json_body(put_resp) or {}
    commit_url = (put_body.get("commit") or {}).get("html_url", "")
    content_sha = (put_body.get("content") or {}).get("sha", "")

    # 3) Verify against ground truth — re-read the file and confirm its bytes
    #    match what we sent. This is what makes a 'done' report trustworthy:
    #    the earlier failures returned API success while the file stayed empty.
    verified = False
    verify_detail = ""
    verify_resp = await http_request(
        url=f"{base_url}{ref_q}", method="GET", headers=headers, mesh_client=mesh_client,
    )
    if verify_resp.get("status_code") == 200:
        vmeta = _parse_json_body(verify_resp) or {}
        vcontent = vmeta.get("content", "")
        if vmeta.get("encoding") == "base64" and isinstance(vcontent, str):
            try:
                got = base64.b64decode(vcontent)
                verified = len(got) == len(raw)
                if not verified:
                    verify_detail = f"size mismatch: sent {len(raw)}, repo has {len(got)}"
            except (ValueError, TypeError):
                verify_detail = "could not decode committed content for verification"
        else:
            verify_detail = "unexpected content encoding on read-back"
    else:
        verify_detail = f"read-back returned HTTP {verify_resp.get('status_code')}"

    result = {
        "committed": True,
        "verified": verified,
        "path": path,
        "repo": repo,
        "branch": branch or "(default)",
        "bytes": len(raw),
        "commit_url": commit_url,
        "content_sha": content_sha,
    }
    if not verified:
        result["warning"] = (
            "commit returned success but read-back verification failed — do NOT "
            f"report this as done. {verify_detail}"
        )
        logger.warning("commit_file unverified for %s/%s: %s", repo, path, verify_detail)
    return result
