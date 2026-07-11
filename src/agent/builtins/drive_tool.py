"""Team Drive tool — the team's shared git repository, hosted on the mesh.

Kills the 256KB-blackboard artifact-shuttling tax (plan §6 Phase 2):
teammates share real files through a mesh-mediated bare git repo at
``{MESH_URL}/mesh/teams/{TEAM_NAME}/drive``. The local checkout lives
in this container's private ``/data/drive``; every push/pull crosses
the mesh (auth, rate limits, quota) — there is no shared volume.

``refs/heads/main`` is integrate-only: pushes to it are rejected
server-side. The flow is: ``branch`` → work → ``sync`` (commit+push
the branch) → ``submit_review`` → the operator merges or rejects.

Auth rides per-invocation ``-c http.extraHeader=...`` git config so the
bearer token is NEVER written to ``.git/config``; tool output is
scrubbed of the token before it reaches the LLM context.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from src.agent.tools import tool
from src.shared.redaction import redact_text_with_urls
from src.shared.utils import sanitize_for_prompt, setup_logging

logger = setup_logging("agent.builtins.drive_tool")

# Local checkout location on the agent's private /data volume.
# Module CONSTANT (tests monkeypatch it) — not mutable state.
_DRIVE_PATH = "/data/drive"

_GIT_TIMEOUT = 120
_MAX_OUTPUT = 20_000
_BRANCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,200}")

_ACTIONS = (
    "clone", "pull", "sync", "branch", "submit_review", "list_reviews", "record_verdict", "log", "status",
)


def _drive_env() -> tuple[str, str, str, str]:
    """(mesh_url, team, agent_id, token) from the container env."""
    return (
        os.environ.get("MESH_URL", "").rstrip("/"),
        os.environ.get("TEAM_NAME", ""),
        os.environ.get("AGENT_ID", ""),
        os.environ.get("MESH_AUTH_TOKEN", ""),
    )


def _fail(error: str, recovery_hint: str, **extras) -> dict:
    """Directive failure envelope (Constraint #10 shape)."""
    out = dict(extras)
    out.update({"ok": False, "error": error, "recovery_hint": recovery_hint})
    return out


def _scrub(text: str, token: str) -> str:
    """Strip the bearer token + URL credentials from subprocess output
    before it enters the LLM context."""
    if token:
        text = text.replace(token, "[REDACTED]")
    return redact_text_with_urls(text)[:_MAX_OUTPUT]


def _git_base_args(agent_id: str, token: str) -> list[str]:
    """Per-invocation config: auth headers + commit identity. The token
    never lands in .git/config — it exists only for this process."""
    args = [
        "-c", f"user.name={agent_id}",
        "-c", f"user.email={agent_id}@agents.local",
    ]
    if token:
        args += ["-c", f"http.extraHeader=Authorization: Bearer {token}"]
    args += ["-c", f"http.extraHeader=X-Agent-ID: {agent_id}"]
    return args


async def _git(args: list[str], *, cwd: str | None, agent_id: str, token: str) -> tuple[int, str]:
    """Run git with auth/identity config; returns (rc, combined output)."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        *_git_base_args(agent_id, token),
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_GIT_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, f"git {args[0]} timed out after {_GIT_TIMEOUT}s"
    return proc.returncode or 0, _scrub(stdout.decode("utf-8", errors="replace"), token)


async def _current_branch(agent_id: str, token: str) -> str:
    rc, out = await _git(["symbolic-ref", "--short", "HEAD"], cwd=_DRIVE_PATH, agent_id=agent_id, token=token)
    return out.strip() if rc == 0 else ""


def _cloned() -> bool:
    return (Path(_DRIVE_PATH) / ".git").exists()


@tool(
    name="team_drive",
    description=(
        "Your team's shared git repository — the durable home for files "
        "you collaborate on with teammates (documents, code, data). It "
        "lives on the mesh; your working copy is /data/drive.\n\n"
        "Workflow: team_drive('clone') once, then "
        "team_drive('branch', branch='my-feature') to start work, edit "
        "files under /data/drive with your normal file tools, "
        "team_drive('sync', message='what changed') to commit+push, and "
        "team_drive('submit_review', title='...', summary='...') when "
        "the branch is ready to integrate. The 'main' branch is "
        "integrate-only — you CANNOT push to it; a human merges "
        "approved reviews. 'pull' refreshes from the drive, "
        "'list_reviews'/'log'/'status' inspect state.\n\n"
        "'record_verdict' records YOUR advisory approve/reject opinion "
        "on an open review (review_id, verdict, optional note) — LEAD-"
        "ONLY, enforced server-side (403 if you aren't your team's "
        "lead). It has no effect on merge/reject; a human still "
        "decides those.\n\n"
        "Solo agents (not on a team) have no Team Drive."
    ),
    parameters={
        "action": {
            "type": "string",
            "enum": list(_ACTIONS),
            "description": "Drive operation to perform",
        },
        "message": {
            "type": "string",
            "description": "Commit message (sync). Required for sync.",
            "default": "",
        },
        "branch": {
            "type": "string",
            "description": (
                "Branch name: target for 'branch' (create/switch), "
                "optional override for 'sync'/'submit_review' (defaults "
                "to the current branch)."
            ),
            "default": "",
        },
        "title": {
            "type": "string",
            "description": "Review title (submit_review). Required for submit_review.",
            "default": "",
        },
        "summary": {
            "type": "string",
            "description": "What the branch changes and why (submit_review).",
            "default": "",
        },
        "status_filter": {
            "type": "string",
            "description": "list_reviews filter: open|merged|rejected|superseded (default all)",
            "default": "",
        },
        "review_id": {
            "type": "string",
            "description": "Review id (record_verdict). Required for record_verdict.",
            "default": "",
        },
        "verdict": {
            "type": "string",
            "enum": ["approve", "reject"],
            "description": "Your advisory verdict (record_verdict). Required for record_verdict.",
            "default": "",
        },
        "note": {
            "type": "string",
            "description": "Optional note explaining your verdict (record_verdict), max 2000 chars.",
            "default": "",
        },
    },
)
async def team_drive(
    action: str,
    message: str = "",
    branch: str = "",
    title: str = "",
    summary: str = "",
    status_filter: str = "",
    review_id: str = "",
    verdict: str = "",
    note: str = "",
    *,
    mesh_client=None,
) -> dict:
    mesh_url, team, agent_id, token = _drive_env()
    if action not in _ACTIONS:
        return _fail(
            f"Unknown action '{action}'.",
            f"Use one of: {', '.join(_ACTIONS)}.",
        )
    # Solo team-of-one: TEAM_NAME falls back to the agent's own id, and
    # teams/agents share one namespace — so team == agent_id means "no
    # real team". The mesh would 403 anyway; fail fast and directive.
    if not team or team == agent_id:
        return _fail(
            "You are not on a team, so there is no Team Drive.",
            "Team Drives exist per team. Ask the operator to add you to "
            "a team, or use your private /data workspace and hand_off "
            "for sharing results.",
        )
    if not mesh_url:
        return _fail(
            "MESH_URL is not configured in this container.",
            "Report this to the operator — it is an infrastructure fault.",
        )
    if branch and (not _BRANCH_RE.fullmatch(branch) or ".." in branch):
        return _fail(
            f"Invalid branch name '{branch}'.",
            "Use letters, digits, dots, dashes, underscores, slashes; no leading dash.",
        )
    remote_url = f"{mesh_url}/mesh/teams/{team}/drive"

    if action == "clone":
        if _cloned():
            rc, out = await _git(["fetch", "origin"], cwd=_DRIVE_PATH, agent_id=agent_id, token=token)
            if rc != 0:
                return _fail(
                    f"Drive already cloned but fetch failed: {out.strip()}",
                    "Check team_drive('status'); if the drive is corrupted, ask the operator for help.",
                )
            return {"ok": True, "cloned": True, "path": _DRIVE_PATH, "note": "already cloned; fetched latest"}
        Path(_DRIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        rc, out = await _git(["clone", remote_url, _DRIVE_PATH], cwd=None, agent_id=agent_id, token=token)
        if rc != 0:
            return _fail(
                f"Clone failed: {out.strip()}",
                "If the error mentions 403/404, you may have been moved off "
                "the team — verify with list_agents or ask the operator. "
                "Do NOT retry in a loop.",
            )
        return {"ok": True, "cloned": True, "path": _DRIVE_PATH}

    if not _cloned():
        return _fail(
            "The Team Drive is not cloned yet.",
            "Run team_drive('clone') first.",
        )

    if action == "pull":
        rc, out = await _git(["fetch", "origin"], cwd=_DRIVE_PATH, agent_id=agent_id, token=token)
        if rc != 0:
            return _fail(
                f"Fetch failed: {out.strip()}",
                "Verify connectivity with team_drive('status'); surface persistent failures to the operator.",
            )
        current = await _current_branch(agent_id, token)
        merged = ""
        if current == "main":
            rc2, out2 = await _git(
                ["merge", "--ff-only", "origin/main"],
                cwd=_DRIVE_PATH, agent_id=agent_id, token=token,
            )
            merged = out2.strip()
            if rc2 != 0:
                return _fail(
                    f"Fast-forward of main failed: {merged}",
                    "Your local main diverged — main is integrate-only, so "
                    "move your work to a branch (team_drive('branch', ...)) "
                    "and reset main to origin/main.",
                )
        return {"ok": True, "fetched": True, "branch": current, "detail": merged or out.strip()}

    if action == "branch":
        if not branch:
            return _fail("branch is required for action='branch'.", "Pass branch='descriptive-name'.")
        rc, out = await _git(["switch", branch], cwd=_DRIVE_PATH, agent_id=agent_id, token=token)
        if rc != 0:
            rc, out = await _git(["switch", "-c", branch], cwd=_DRIVE_PATH, agent_id=agent_id, token=token)
            if rc != 0:
                return _fail(
                    f"Could not create/switch branch: {out.strip()}",
                    "Check team_drive('status') for uncommitted conflicts.",
                )
        return {"ok": True, "branch": branch, "detail": out.strip()}

    if action == "sync":
        current = await _current_branch(agent_id, token)
        if branch and branch != current:
            switch = await team_drive("branch", branch=branch, mesh_client=mesh_client)
            if not switch.get("ok"):
                return switch
            current = branch
        if current == "main" or not current:
            return _fail(
                "You are on 'main', which is integrate-only — the mesh "
                "will reject the push.",
                "Create a feature branch first: team_drive('branch', "
                "branch='my-change'), then sync again.",
            )
        if not message.strip():
            return _fail("message is required for action='sync'.", "Pass message='what changed and why'.")
        await _git(["add", "-A"], cwd=_DRIVE_PATH, agent_id=agent_id, token=token)
        rc, out = await _git(
            ["commit", "-m", sanitize_for_prompt(message)[:500]],
            cwd=_DRIVE_PATH, agent_id=agent_id, token=token,
        )
        committed = rc == 0
        # "nothing to commit" is fine — the branch may already be ahead.
        if rc != 0 and "nothing to commit" not in out:
            return _fail(f"Commit failed: {out.strip()}", "Inspect team_drive('status') and fix the reported problem.")
        rc, out = await _git(
            ["push", "origin", f"HEAD:refs/heads/{current}"],
            cwd=_DRIVE_PATH, agent_id=agent_id, token=token,
        )
        if rc != 0:
            return _fail(
                f"Push failed: {out.strip()}",
                "If it mentions the quota, prune large files. If it "
                "mentions 'protected', you targeted main — use a feature "
                "branch. Otherwise surface the error to the operator. "
                "Your commit is safe locally either way.",
            )
        return {"ok": True, "synced": True, "branch": current, "committed": committed, "detail": out.strip()}

    if action == "submit_review":
        if mesh_client is None:
            return _fail("No mesh_client available.", "This is an infrastructure fault — report it to the operator.")
        target = branch or await _current_branch(agent_id, token)
        if not target or target == "main":
            return _fail(
                "submit_review needs a feature branch (not main).",
                "Sync your branch first, then submit_review with branch='...'.",
            )
        if not title.strip():
            return _fail("title is required for submit_review.", "Pass title='what this branch delivers'.")
        try:
            resp = await mesh_client.submit_drive_review(
                target, title.strip()[:200], summary.strip()[:4000],
            )
        except Exception as e:
            return _fail(
                f"Review submission failed: {_scrub(str(e), token)[:300]}",
                "If the branch was never pushed, run team_drive('sync') "
                "first. You MUST NOT report the review as submitted.",
            )
        review = resp.get("review") or {}
        return {
            "ok": True,
            "submitted": True,
            "review_id": review.get("id", ""),
            "branch": target,
            "note": "A human will merge or reject this review — check list_reviews later.",
        }

    if action == "list_reviews":
        if mesh_client is None:
            return _fail("No mesh_client available.", "This is an infrastructure fault — report it to the operator.")
        try:
            resp = await mesh_client.list_drive_reviews(status_filter.strip())
        except Exception as e:
            return _fail(
                f"Listing reviews failed: {_scrub(str(e), token)[:300]}",
                "Retry once; surface persistent failures to the operator.",
            )
        reviews = []
        for r in resp.get("reviews", []):
            # Teammate-authored text re-enters THIS agent's LLM context —
            # sanitize at the input boundary.
            reviews.append({
                "id": str(r.get("id", "")),
                "branch": sanitize_for_prompt(str(r.get("branch", ""))),
                "author": sanitize_for_prompt(str(r.get("author", ""))),
                "title": sanitize_for_prompt(str(r.get("title", ""))),
                "summary": sanitize_for_prompt(str(r.get("summary", "")))[:1000],
                "status": str(r.get("status", "")),
                # Short reviewed-tip sha so a reader sees what gets integrated.
                "head_sha": sanitize_for_prompt(str(r.get("head_sha_short") or r.get("head_sha") or "")),
                "created_at": r.get("created_at"),
                "resolved_at": r.get("resolved_at"),
                # Advisory lead verdict (plan §8 #13) — informational only,
                # zero effect on merge/reject. Sanitized like every other
                # teammate-authored field re-entering this agent's context.
                "lead_verdict": r.get("lead_verdict"),
                "lead_verdict_note": sanitize_for_prompt(str(r.get("lead_verdict_note") or ""))[:1000] or None,
                "lead_verdict_at": r.get("lead_verdict_at"),
                # Verified reviewer identity that recorded the verdict (plan
                # §8 #20) — an agent id, not free text, so no sanitize.
                "lead_verdict_by": r.get("lead_verdict_by"),
            })
        return {"ok": True, "reviews": reviews, "count": len(reviews)}

    if action == "record_verdict":
        if mesh_client is None:
            return _fail("No mesh_client available.", "This is an infrastructure fault — report it to the operator.")
        if not review_id.strip():
            return _fail("review_id is required for record_verdict.", "Pass review_id='...' from list_reviews.")
        if verdict not in ("approve", "reject"):
            return _fail(
                f"Invalid verdict '{verdict}'.",
                "Pass verdict='approve' or verdict='reject'.",
            )
        clean_note = sanitize_for_prompt(note)[:2000] if note else ""
        try:
            resp = await mesh_client.record_drive_verdict(review_id.strip(), verdict, clean_note)
        except Exception as e:
            return _fail(
                f"Recording verdict failed: {_scrub(str(e), token)[:300]}",
                "Only your team's lead can record a verdict — if you aren't "
                "the lead, this will always 403. If the review is no "
                "longer open, it can't take a verdict.",
            )
        review = resp.get("review") or {}
        return {
            "ok": True,
            "recorded": True,
            "review_id": str(review.get("id", review_id)),
            "verdict": verdict,
            "note": "This is advisory only — a human still merges or rejects the review.",
        }

    if action == "log":
        rc, out = await _git(
            ["log", "--oneline", "--graph", "--decorate", "-20"],
            cwd=_DRIVE_PATH, agent_id=agent_id, token=token,
        )
        return {"ok": rc == 0, "log": sanitize_for_prompt(out)}

    # action == "status"
    rc, out = await _git(["status", "--short", "--branch"], cwd=_DRIVE_PATH, agent_id=agent_id, token=token)
    return {"ok": rc == 0, "status": sanitize_for_prompt(out)}
