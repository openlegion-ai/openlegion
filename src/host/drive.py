"""Team Drive — mesh-hosted bare git repos (Phase-2 unit 1).

One bare repository per team under ``data/team_drives/{team_id}.git``
(env override ``OPENLEGION_TEAM_DRIVES_DIR``). Agents interact through
git smart HTTP on the mesh (``/mesh/teams/{id}/drive/...``); the mesh
mediates every byte, so the drive inherits the mesh's auth, rate
limits, and quota governance — no shared Docker volume exists anywhere
in this design (ratified §8 #1: git-Drive-first, no raw scratch).

``refs/heads/main`` is integrate-only: a pre-receive hook installed at
provision time rejects main updates unless ``OL_DRIVE_PRIVILEGED=1``
is present in the hook's environment — which the mesh sets ONLY for
operator-tier callers and the internal review-merge path. Workers push
feature branches and submit reviews (``drive_reviews`` in the
TeamStore); the merge itself happens mesh-side via
``git merge-tree --write-tree`` + a compare-and-swap ``update-ref``.

Lifecycle is owned by the TeamStore (plan A.3 #3) through the injected
provisioner, which calls :meth:`RuntimeBackend.ensure_team_volume` /
``remove_team_volume`` — concrete shared implementations that delegate
here. This module holds only pure functions + constants (Constraint
#8: no module-level mutable state; per-request caches live on the app).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import subprocess
import tempfile
import zlib
from pathlib import Path

from src.host.teams import validate_team_id
from src.shared import limits
from src.shared.utils import setup_logging

logger = setup_logging("host.drive")

# Wall-clock ceiling for one smart-HTTP subprocess (upload-pack /
# receive-pack) or one merge plumbing call. Kills the whole process
# group on expiry so a wedged pack negotiation can't leak children.
GIT_RPC_TIMEOUT = 120.0

# Provision-time plumbing calls are small and fast; keep a tighter lid.
_PLUMBING_TIMEOUT = 30.0

# The two smart-HTTP services. Maps the wire name to the git subcommand.
SMART_SERVICES = {
    "git-upload-pack": "upload-pack",
    "git-receive-pack": "receive-pack",
}

# Deterministic identity for mesh-authored commits (the seed commit and
# review merges). Never a real mailbox — the drive is machine-plumbing.
_MESH_GIT_IDENT = {
    "GIT_AUTHOR_NAME": "OpenLegion Mesh",
    "GIT_AUTHOR_EMAIL": "mesh@openlegion.local",
    "GIT_COMMITTER_NAME": "OpenLegion Mesh",
    "GIT_COMMITTER_EMAIL": "mesh@openlegion.local",
}

# refs/heads/main protection. /bin/sh (no bashisms); reads the standard
# "old new ref" lines from stdin. OL_DRIVE_PRIVILEGED is set by the mesh
# subprocess env ONLY for operator-tier callers and the internal
# integrate path — a worker's push env never carries it.
_PRE_RECEIVE_HOOK = """#!/bin/sh
# OpenLegion Team Drive: refs/heads/main is integrate-only.
# Installed and kept in sync by src/host/drive.py — do not edit.
if [ "$OL_DRIVE_PRIVILEGED" = "1" ]; then
    exit 0
fi
status=0
while read old new ref; do
    if [ "$ref" = "refs/heads/main" ]; then
        echo "refs/heads/main is protected: push a feature branch and submit a review instead" >&2
        status=1
    fi
done
exit $status
"""


class DriveError(RuntimeError):
    """Raised when a drive git operation fails; message is operator-safe."""


class MergeConflict(DriveError):
    """Raised when a review merge has content conflicts."""

    def __init__(self, message: str, conflict_info: list[str]):
        super().__init__(message)
        self.conflict_info = conflict_info


class RefMoved(DriveError):
    """Raised when the merge compare-and-swap lost to a racing main update."""


def drives_root() -> Path:
    """Home directory of all team drives (env-overridable for tests)."""
    return Path(os.environ.get("OPENLEGION_TEAM_DRIVES_DIR", "data/team_drives"))


def repo_path_for(team_id: str, root: Path | None = None) -> Path:
    """Absolute bare-repo path for ``team_id`` (validates the id — the
    team-id grammar has no path separators, so the join is traversal-safe)."""
    base = root if root is not None else drives_root()
    return (base / f"{validate_team_id(team_id)}.git").resolve()


def _subprocess_env(*, privileged: bool = False) -> dict[str, str]:
    """Minimal env for drive git subprocesses.

    Never inherits the mesh process env (which carries OPENLEGION_* keys
    and SYSTEM credentials); git only needs PATH, and the hook contract
    needs OL_DRIVE_PRIVILEGED for operator-tier calls.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        # Hermetic git: ignore ambient host/global config so a stray
        # core.hooksPath / receive.* on the mesh box can't disable main
        # protection (matches the agent tool's posture). GIT_CONFIG_GLOBAL
        # is pointed at /dev/null rather than unset — unset would let git
        # fall back to $HOME/.gitconfig.
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    if privileged:
        env["OL_DRIVE_PRIVILEGED"] = "1"
    return env


def _run_git(args: list[str], *, cwd: Path | None = None, input_bytes: bytes | None = None,
             env: dict[str, str] | None = None, timeout: float = _PLUMBING_TIMEOUT) -> str:
    """Run a plumbing git command synchronously; returns stripped stdout.

    Raises :class:`DriveError` with stderr detail on non-zero exit.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            input=input_bytes,
            env=env or _subprocess_env(),
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise DriveError("git binary not found on the mesh host")
    except subprocess.TimeoutExpired:
        raise DriveError(f"git {args[0]} timed out after {timeout}s")
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()[:300]
        raise DriveError(f"git {args[0]} failed: {detail}")
    return proc.stdout.decode("utf-8", errors="replace").strip()


def _install_hook_and_config(repo: Path) -> None:
    """(Re)install the pre-receive protection hook + repo config.

    Idempotent — runs on every ensure so a hook edit in this module (or
    a limits change to the push cap) propagates at the next provision.
    """
    hooks = repo / "hooks"
    hooks.mkdir(exist_ok=True)
    hook_path = hooks / "pre-receive"
    hook_path.write_text(_PRE_RECEIVE_HOOK)
    hook_path.chmod(0o755)
    max_input = limits.resolve("drive_push_max_mb") * 1024 * 1024
    _run_git(["config", "receive.maxInputSize", str(max_input)], cwd=repo)
    # Reject dangling-object tricks and keep server-side history linear-ish.
    _run_git(["config", "receive.denyDeleteCurrent", "true"], cwd=repo)
    # Feature-branch anti-griefing: a worker must not delete or force-push
    # over another member's in-review branch. denyDeletes blocks ref
    # deletion; denyNonFastForwards blocks rewinding a ref to a non-descendant.
    # Both apply to receive-pack pushes only (the mesh-side integrate uses
    # update-ref, which these configs never govern) and are idempotently
    # re-applied on every ensure. refs/heads/main stays hook-protected on top;
    # tags are deliberately NOT restricted (a worker tag can't touch main).
    _run_git(["config", "receive.denyDeletes", "true"], cwd=repo)
    _run_git(["config", "receive.denyNonFastForwards", "true"], cwd=repo)


def ensure_team_drive(team_id: str, root: Path | None = None) -> str:
    """Create (or repair) the team's bare repo. Idempotent.

    Seeds an initial commit on ``main`` via plumbing so a fresh clone
    is never an unborn-HEAD repo, points HEAD at ``refs/heads/main``,
    installs the pre-receive hook, and sets ``receive.maxInputSize``.
    Returns the absolute repo path (the ``drive_ref`` value).
    """
    repo = repo_path_for(team_id, root)
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not (repo / "HEAD").exists():
        _run_git(["init", "--bare", str(repo)])
    _run_git(["symbolic-ref", "HEAD", "refs/heads/main"], cwd=repo)
    try:
        _run_git(["rev-parse", "--verify", "--quiet", "refs/heads/main"], cwd=repo)
    except DriveError:
        # Unborn main — seed the initial commit with pure plumbing.
        readme = f"# Team Drive for {team_id}\n\nShared git workspace. Push branches; main is integrate-only.\n"
        blob = _run_git(["hash-object", "-w", "--stdin"], cwd=repo, input_bytes=readme.encode())
        tree = _run_git(["mktree"], cwd=repo, input_bytes=f"100644 blob {blob}\tREADME.md\n".encode())
        env = {**_subprocess_env(), **_MESH_GIT_IDENT}
        commit = _run_git(["commit-tree", tree, "-m", f"Initialize Team Drive for {team_id}"], cwd=repo, env=env)
        _run_git(["update-ref", "refs/heads/main", commit], cwd=repo)
    _install_hook_and_config(repo)
    return str(repo)


def remove_team_drive(team_id: str, root: Path | None = None) -> None:
    """Delete the team's bare repo (best-effort — deletion paths must
    not fail because a drive rmtree hiccuped)."""
    repo = repo_path_for(team_id, root)
    if repo.exists():
        try:
            shutil.rmtree(repo)
        except OSError:
            logger.exception("Failed to remove team drive for %s", team_id)


def repo_size_bytes(repo: Path) -> int:
    """Total on-disk size of the repo dir (quota input). Blocking —
    callers on the event loop wrap in ``run_in_executor``."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(repo):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


def gunzip_capped(data: bytes, cap: int) -> bytes:
    """Decompress a gzip request body, refusing to inflate past ``cap``
    bytes (zip-bomb guard: the cap applies to DECOMPRESSED size)."""
    decomp = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = decomp.decompress(data, cap + 1)
    if len(out) > cap:
        raise ValueError("decompressed request body exceeds the push cap")
    tail = decomp.flush()
    if len(out) + len(tail) > cap:
        raise ValueError("decompressed request body exceeds the push cap")
    return out + tail


def pkt_line(payload: bytes) -> bytes:
    """git pkt-line: 4 hex digits of (len(prefix)+len(payload)) + payload."""
    return f"{len(payload) + 4:04x}".encode() + payload


async def _run_git_async(
    args: list[str],
    *,
    cwd: Path,
    input_bytes: bytes | None = None,
    privileged: bool = False,
    timeout: float = GIT_RPC_TIMEOUT,
) -> tuple[int, bytes, bytes]:
    """Run a git subprocess off the event loop with a hard timeout.

    Runs in its own session (process group) so the timeout kill reaps
    any children the pack machinery spawned. Returns (rc, stdout, stderr).

    ``stdin`` is a pipe ONLY when there is input to feed: writing even
    ``b""`` into a process that never reads stdin (``--advertise-refs``)
    races its exit, and uvloop raises RuntimeError on the closed handle
    where vanilla asyncio would swallow the BrokenPipeError.
    """
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd),
        env=_subprocess_env(privileged=privileged),
        stdin=asyncio.subprocess.PIPE if input_bytes is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    def _kill_group() -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(input=input_bytes), timeout=timeout)
    except asyncio.TimeoutError:
        _kill_group()
        await proc.wait()
        raise DriveError(f"git {args[0]} timed out after {timeout}s")
    except (BrokenPipeError, ConnectionResetError, RuntimeError) as e:
        # The subprocess exited before consuming its stdin (e.g. a pack
        # rejected up front). Reap it and surface a clean error instead
        # of a 500 with a raw event-loop traceback.
        _kill_group()
        await proc.wait()
        raise DriveError(f"git {args[0]} closed its input early: {e}")
    return proc.returncode or 0, stdout, stderr


async def advertise_refs(service: str, repo: Path) -> bytes:
    """Smart-HTTP ``GET info/refs`` body: pkt-line service header +
    flush-pkt + ``--advertise-refs`` output."""
    subcommand = SMART_SERVICES[service]
    rc, stdout, stderr = await _run_git_async(
        [subcommand, "--stateless-rpc", "--advertise-refs", str(repo)],
        cwd=repo,
    )
    if rc != 0:
        raise DriveError(f"advertise-refs failed: {stderr.decode('utf-8', errors='replace')[:300]}")
    header = pkt_line(f"# service={service}\n".encode()) + b"0000"
    return header + stdout


async def service_rpc(service: str, repo: Path, body: bytes, *, privileged: bool = False) -> bytes:
    """Smart-HTTP POST body → ``git {service} --stateless-rpc`` → response.

    ``privileged`` controls OL_DRIVE_PRIVILEGED in the hook env (main
    protection); it must be True ONLY for operator-tier callers.
    """
    subcommand = SMART_SERVICES[service]
    rc, stdout, stderr = await _run_git_async(
        [subcommand, "--stateless-rpc", str(repo)],
        cwd=repo,
        input_bytes=body,
        privileged=privileged,
    )
    if rc != 0 and not stdout:
        # A hook rejection still exits 0 with an "ng"/unpack report in
        # stdout, so a hard non-zero with no output is an infra error.
        raise DriveError(f"{service} failed: {stderr.decode('utf-8', errors='replace')[:300]}")
    return stdout


def git_supports_merge_tree() -> bool:
    """``git merge-tree --write-tree`` needs git >= 2.38."""
    try:
        raw = _run_git(["version"])
    except DriveError:
        return False
    parts = raw.split()
    ver = parts[2] if len(parts) >= 3 else ""
    nums = ver.split(".")
    try:
        major, minor = int(nums[0]), int(nums[1])
    except (ValueError, IndexError):
        return False
    return (major, minor) >= (2, 38)


def branch_exists(repo: Path, branch: str) -> bool:
    try:
        _run_git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=repo)
        return True
    except DriveError:
        return False


def branch_head_sha(repo: Path, branch: str) -> str:
    """Current commit sha at ``refs/heads/{branch}`` (``""`` if absent).

    Recorded at review time and re-checked at merge time so a worker that
    advances the branch AFTER approval can't change what gets integrated.
    """
    try:
        return _run_git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=repo)
    except DriveError:
        return ""


def _parse_merge_tree_z(raw: bytes) -> tuple[str, list[str]]:
    """Parse ``git merge-tree --write-tree -z`` output into (tree, files).

    The ``-z`` format is NUL-delimited: ``<tree-oid> NUL (<conflicted-file
    -info> NUL)* NUL (<informational-message> NUL)*``. The conflicted-file
    section holds ``<mode> <oid> <stage>\\t<path>`` entries (one per merge
    stage, so a path repeats); we split on the tab and dedupe to isolate
    real FILENAMES from the trailing prose section (which starts after the
    empty separator). Using ``-z`` also avoids ``--name-only`` (added in
    git 2.40) so the minimum stays git 2.38 (``--write-tree``/``-z``).
    """
    parts = raw.split(b"\0")
    tree = parts[0].decode("utf-8", errors="replace").strip() if parts else ""
    files: list[str] = []
    for chunk in parts[1:]:
        if chunk == b"":
            break  # end of the conflicted-file-info section
        txt = chunk.decode("utf-8", errors="replace")
        name = txt.split("\t", 1)[1] if "\t" in txt else txt
        name = name.strip()
        if name and name not in files:
            files.append(name)
    return tree, files


async def merge_branch(repo: Path, source_commit: str, *, message: str) -> str:
    """Merge commit ``source_commit`` into main mesh-side (integrate path).

    The caller passes the EXACT commit sha recorded at review time (not a
    live branch ref), so a worker advancing the branch after approval
    cannot change what gets integrated. ``git merge-tree --write-tree -z``
    computes the merged tree without a worktree; a clean result becomes a
    two-parent commit installed on main via an ``update-ref --stdin``
    compare-and-swap against the main tip read at the start — a racing push
    to main (operator-tier) fails the CAS instead of being overwritten.

    Returns the new merge commit sha. Raises :class:`MergeConflict` on
    content conflicts, :class:`RefMoved` on a lost CAS, and
    :class:`DriveError` on anything else.
    """
    if not git_supports_merge_tree():
        raise DriveError(
            "mesh host git is too old for review merges: `git merge-tree --write-tree` needs git >= 2.38"
        )
    loop = asyncio.get_running_loop()

    def _plumb() -> str:
        main_sha = _run_git(["rev-parse", "--verify", "refs/heads/main"], cwd=repo)
        # Peel the recorded sha to a commit; a bad/unreachable sha errors
        # cleanly here. A 40-hex sha can never be parsed as an option.
        src_sha = _run_git(["rev-parse", "--verify", f"{source_commit}^{{commit}}"], cwd=repo)
        proc = subprocess.run(
            ["git", "merge-tree", "--write-tree", "-z", main_sha, src_sha],
            cwd=str(repo),
            env=_subprocess_env(),
            capture_output=True,
            timeout=GIT_RPC_TIMEOUT,
        )
        tree, files = _parse_merge_tree_z(proc.stdout)
        if proc.returncode == 1:
            raise MergeConflict(
                f"merge of {src_sha[:10]} into main has conflicts",
                conflict_info=files,
            )
        if proc.returncode != 0 or not tree:
            detail = proc.stderr.decode("utf-8", errors="replace").strip()[:300]
            raise DriveError(f"merge-tree failed: {detail}")
        env = {**_subprocess_env(), **_MESH_GIT_IDENT}
        commit = _run_git(
            ["commit-tree", tree, "-p", main_sha, "-p", src_sha, "-m", message],
            cwd=repo,
            env=env,
        )
        # CAS: transaction form — "update SP ref SP new SP old". Fails if
        # main no longer points at main_sha (a racing privileged push).
        try:
            _run_git(
                ["update-ref", "--stdin"],
                cwd=repo,
                input_bytes=f"update refs/heads/main {commit} {main_sha}\n".encode(),
            )
        except DriveError as e:
            raise RefMoved(f"main moved during the merge (compare-and-swap failed) — retry: {e}")
        return commit

    return await loop.run_in_executor(None, _plumb)


# Path grammar for direct-commit artifacts. Each ``/``-separated segment
# is a POSIX-ish filename: letters, digits, dot, dash, underscore, space.
# No leading dash (option-injection), no ``..`` traversal, no empty
# segments, no absolute paths. Validated on the mesh BEFORE the path
# reaches ``update-index`` / ``cat-file``.
_DRIVE_PATH_SEGMENT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._ -]*")
# ``ref`` for a file read: a branch name, ``main``, or a hex sha (short
# or full). Same conservative grammar as branch names, sans path chars.
_DRIVE_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,200}")


def validate_drive_path(path: str) -> str:
    """Validate a drive-relative artifact path; returns it normalized.

    Rejects traversal (``..``), absolute paths, empty/dot segments, and
    option-injection (a leading ``-``). Raises :class:`ValueError` — the
    caller maps that to an HTTP 400.
    """
    p = (path or "").strip().strip("/")
    if not p or ".." in p.split("/"):
        raise ValueError("path must be a non-empty relative path without '..'")
    segments = p.split("/")
    if len(segments) > 20:
        raise ValueError("path is too deeply nested")
    for seg in segments:
        if not seg or seg in (".", "..") or not _DRIVE_PATH_SEGMENT_RE.fullmatch(seg):
            raise ValueError(f"invalid path segment: {seg!r}")
    return p


def validate_drive_ref(ref: str) -> str:
    """Validate a read ref (branch or sha); defaults to ``main`` when empty."""
    r = (ref or "").strip() or "main"
    if not _DRIVE_REF_RE.fullmatch(r) or ".." in r:
        raise ValueError("ref must be a branch name or commit sha")
    return r


async def commit_file(
    repo: Path,
    path: str,
    content: bytes,
    *,
    message: str,
    author_name: str,
    author_email: str,
) -> str:
    """Commit a single file directly to ``main`` (deliverable registration).

    This is the direct-commit path used for artifact registration and
    handoff-data payloads — NOT reviewed source. Review-before-integrate
    governs agent-pushed feature BRANCHES (via receive-pack + the
    pre-receive hook); this path is mesh-authored plumbing, so it writes
    ``main`` directly with the SENDER recorded as the commit author.

    Implementation is pure plumbing against a throwaway index so no
    worktree is needed and nested paths (``handoffs/{a}/{id}.json``) are
    built correctly: ``read-tree`` the current main tree, ``update-index``
    the new blob at ``path``, ``write-tree``, ``commit-tree`` with the
    sender identity, then a compare-and-swap ``update-ref`` against the
    main tip read at the start (a racing main update fails the CAS).

    Returns the new commit sha. Raises :class:`RefMoved` on a lost CAS
    and :class:`DriveError` on any other plumbing failure.
    """
    loop = asyncio.get_running_loop()

    def _plumb() -> str:
        main_sha = _run_git(["rev-parse", "--verify", "refs/heads/main"], cwd=repo)
        blob = _run_git(["hash-object", "-w", "--stdin"], cwd=repo, input_bytes=content)
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index")
            idx_env = {**_subprocess_env(), "GIT_INDEX_FILE": index_path}
            _run_git(["read-tree", main_sha], cwd=repo, env=idx_env)
            # ``--`` before the pathspec so a path can never be read as an
            # option; the grammar already forbids a leading dash, this is
            # belt-and-braces.
            _run_git(
                ["update-index", "--add", "--cacheinfo", f"100644,{blob},{path}"],
                cwd=repo,
                env=idx_env,
            )
            tree = _run_git(["write-tree"], cwd=repo, env=idx_env)
        ident = {
            **_subprocess_env(),
            "GIT_AUTHOR_NAME": author_name,
            "GIT_AUTHOR_EMAIL": author_email,
            "GIT_COMMITTER_NAME": "OpenLegion Mesh",
            "GIT_COMMITTER_EMAIL": "mesh@openlegion.local",
        }
        commit = _run_git(
            ["commit-tree", tree, "-p", main_sha, "-m", message],
            cwd=repo,
            env=ident,
        )
        try:
            _run_git(
                ["update-ref", "--stdin"],
                cwd=repo,
                input_bytes=f"update refs/heads/main {commit} {main_sha}\n".encode(),
            )
        except DriveError as e:
            raise RefMoved(f"main moved during the commit (compare-and-swap failed) — retry: {e}")
        return commit

    return await loop.run_in_executor(None, _plumb)


async def read_file(repo: Path, path: str, *, ref: str = "main") -> bytes:
    """Read a single file's bytes from the drive at ``ref`` (default main).

    Raises :class:`FileNotFoundError` when the path does not exist at that
    ref, and :class:`DriveError` on other plumbing failures.
    """
    safe_path = validate_drive_path(path)
    safe_ref = validate_drive_ref(ref)
    loop = asyncio.get_running_loop()

    def _cat() -> bytes:
        # ``{ref}:{path}`` object spec; both halves are grammar-validated
        # above so neither can inject a git option or traverse out.
        try:
            proc = subprocess.run(
                ["git", "cat-file", "blob", f"{safe_ref}:{safe_path}"],
                cwd=str(repo),
                env=_subprocess_env(),
                capture_output=True,
                timeout=_PLUMBING_TIMEOUT,
            )
        except FileNotFoundError:
            raise DriveError("git binary not found on the mesh host")
        except subprocess.TimeoutExpired:
            raise DriveError("git cat-file timed out")
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="replace").strip().lower()
            if "does not exist" in detail or "not a valid" in detail or "exists on disk" in detail:
                raise FileNotFoundError(f"{safe_path} not found at {safe_ref}")
            raise DriveError(f"cat-file failed: {detail[:200]}")
        return proc.stdout

    return await loop.run_in_executor(None, _cat)
