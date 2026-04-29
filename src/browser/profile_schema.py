"""Profile schema versioning + migration framework (Phase 1.4).

Establishes a single source of truth for the state of a browser profile on
disk so later phases (fonts in §6.2, uBlock Origin in §7.1, resolution
randomization in §6.1) can evolve the profile format without leaving stale
caches, mismatched prefs, or half-installed extensions behind.

Design principles:

* **Single marker file.** Each profile carries ``.ol_schema`` at its root
  containing the integer version number. Missing marker = version 0
  (pre-upgrade or fresh profile).
* **Pre-launch migration.** Migrations run in
  ``BrowserManager.get_or_start()`` **before** Camoufox / Playwright opens
  the profile directory. Camoufox writes into the profile during launch,
  so post-launch migrations would race its writes and cause corruption.
* **Atomic per-profile lock.** ``fcntl.flock`` on a sidecar ``.ol_schema.lock``
  file serializes concurrent migrations of the same profile. A second
  process started during migration skips with an INFO log; it can retry
  next startup.
* **Crash-safe writes.** Marker writes are ``write-to-tmp`` + ``fsync`` +
  ``os.replace``. A crash mid-migration leaves either the pre-swap ``.bak``
  snapshot or the post-swap final state, never a half-written version.
* **Orphan cleanup on boot.** Any ``*.tmp`` from a prior crash is deleted
  at :func:`migrate_profile` entry so we never inherit partial writes.
* **Cookies / localStorage / IndexedDB preserved.** Migrations only touch
  build-time artifacts (font cache, startupCache, ephemeral prefs). Never
  touch ``cookies.sqlite``, ``webappsstore.sqlite``, ``storage/default``
  (IndexedDB), or ``bookmarks.sqlite``. Users' sessions and saved data
  survive migrations.

Usage:

    from src.browser.profile_schema import migrate_profile

    async def get_or_start(self, agent_id: str) -> CamoufoxInstance:
        profile_dir = self.profiles_dir / agent_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        migrate_profile(profile_dir)            # must run before Camoufox
        browser = await AsyncNewBrowser(pw, user_data_dir=str(profile_dir), ...)

Phase 1.4 ships the framework with version ``1`` as a no-op baseline. Later
phases register migrations that advance the version and run their transforms.
"""

from __future__ import annotations

import fcntl
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("browser.profile_schema")


# ── Current schema version & migration registry ────────────────────────────


# Bump this monotonically when adding a new migration. Never decrement.
# Never reuse a number — migrations are applied by version key in order.
PROFILE_SCHEMA_VERSION: int = 3


# uBlock Origin's Firefox addon ID. Used as the filename when copying
# the bundled XPI into a profile's ``extensions/`` directory — Firefox
# requires the file to be named ``{addon-id}.xpi`` for auto-discovery.
UBLOCK_ADDON_ID: str = "uBlock0@raymondhill.net"


def _ublock_source_path() -> Path:
    """Source XPI inside the browser container (placed by Dockerfile.browser).

    Override via ``OPENLEGION_UBLOCK_XPI`` for unit tests + dev environments
    where the container artifact isn't available. When the source is
    missing, the migration logs and stamps the schema anyway — the launch
    still succeeds without the extension, which matters for fresh dev
    installs whose Dockerfile.browser may not yet have shipped uBO.
    """
    return Path(
        os.environ.get(
            "OPENLEGION_UBLOCK_XPI",
            "/opt/openlegion/extensions/uBlock0.xpi",
        )
    )


def _adblock_enabled() -> bool:
    """Operator escape hatch (§7.1 / §2.1 ``BROWSER_ENABLE_ADBLOCK``).

    Defaults to ``true``. Setting ``BROWSER_ENABLE_ADBLOCK=false`` skips
    the install entirely — already-installed profiles keep their
    extension (cleanly uninstalling Firefox extensions in-place is
    fragile; operators wanting a hard reset should ``openlegion reset``).
    """
    from src.browser.flags import get_bool

    return get_bool("BROWSER_ENABLE_ADBLOCK", True)


def _v2_clear_font_caches(profile: Path) -> None:
    """Migration v2 (Phase 3 §6.2): clear Firefox font caches.

    The container image just gained the Carlito/Caladea/Liberation/DejaVu
    font stack plus fontconfig aliases that resolve Segoe UI / Calibri /
    Cambria to those substitutes. Firefox caches the font list it sees
    at first launch in ``startupCache/`` and ``fontlist.json`` under the
    profile root. If we don't clear them, existing profiles keep using
    the stale "no Segoe UI available" font table and the fingerprint
    alignment we just installed never takes effect.

    **Critical:** we deliberately DO NOT touch ``compatibility.ini``.
    Firefox uses that file to track "is this the same Firefox build that
    last opened this profile?" — if we delete it, the next launch
    triggers the full first-run path: about:welcome tab, default-browser
    nag, profile-reset prompts, etc. All of which block automation. The
    font cache rebuilds correctly without compatibility.ini being
    touched.

    Idempotent. Missing files on a fresh profile are fine. Touches only
    cache artifacts — never cookies / localStorage / IndexedDB /
    bookmarks / the user's session state.

    Raises ``OSError`` on any unlink failure so the migration framework
    triggers its backup-restore path. Letting unlinks fail silently
    would leave the profile in a half-state (some cache cleared, some
    not) with the marker stamped at v2 — unrecoverable on next launch.
    """
    # startupCache/ holds compiled-XUL + fontlist blobs Firefox rebuilds
    # on any chrome/resource change. Whole directory is safe to wipe —
    # it's rebuilt automatically on next launch.
    #
    # Use ``shutil.rmtree`` directly (NOT ``_remove_tree``) so any
    # OSError propagates. ``_remove_tree`` is the best-effort helper
    # used during framework backup teardown where partial-fail is
    # tolerable; here partial-fail must trigger backup-restore.
    startup_cache = profile / "startupCache"
    if startup_cache.exists() and startup_cache.is_dir():
        shutil.rmtree(startup_cache)

    # Top-level cache blobs Firefox uses for font metadata. Names are
    # stable across versions; removing them is safe. ``compatibility.ini``
    # is INTENTIONALLY OMITTED — see docstring.
    for cache_file in (
        "fontlist.json",
        "font.properties",
    ):
        target = profile / cache_file
        if target.is_file():
            # Re-raise on failure so the migration framework's restore
            # path triggers. A locked/permissioned cache file is the
            # operator's signal that this profile needs investigation.
            target.unlink()


# Callables registered here run in `migrate_profile()` when the on-disk
# version is LESS than the key. Each callable takes the profile directory
# Path and mutates it in place.
#
# Contract for a migration function:
#   - Idempotent: running it twice on the same profile must be safe.
#   - Never raise on expected-missing files (e.g. font cache may not exist
#     on a freshly-created profile).
#   - Never touch cookies.sqlite, webappsstore.sqlite, storage/default/,
#     or bookmarks.sqlite. Preserve user sessions.
#   - Raise on unrecoverable failure. The caller will restore from .bak.
def _v3_install_ublock(profile: Path) -> None:
    """Migration v3 (Phase 4 §7.1): install uBlock Origin into the profile.

    Firefox auto-discovers extensions placed at
    ``{profile}/extensions/{addon-id}.xpi`` on launch (combined with
    ``extensions.autoDisableScopes=0`` and ``extensions.startupScanScopes``
    set in stealth prefs). The XPI itself is bundled at build time by
    ``Dockerfile.browser`` to ``/opt/openlegion/extensions/uBlock0.xpi``.

    Behavior:

    - **Source missing.** Log INFO and return. Happens on dev installs
      whose browser image hasn't been rebuilt yet, or test environments
      that don't ship the XPI. Migration framework still stamps the
      profile to v3 — the next launch on a properly-built container
      will fix it because :func:`sync_adblock_extension` runs every
      launch (see ``src/browser/service.py``).
    - **Adblock disabled.** Operator set ``BROWSER_ENABLE_ADBLOCK=false``;
      skip without raising so existing profiles still upgrade their
      schema marker (no-op-ifying the migration prevents a stuck
      "stamped at v2 forever" state).
    - **Default install.** Copy XPI to ``{profile}/extensions/``,
      ``chmod 0644``. Firefox installs and enables on next launch.

    Idempotent — copies are guarded by mtime/size comparison so
    re-running the migration on an already-installed profile is cheap.

    Raises ``OSError`` only on unrecoverable I/O failures (disk full,
    permission denied on a directory we own). The migration framework's
    backup-restore path triggers; the operator gets a stable profile
    rolled back to v2.
    """
    source = _ublock_source_path()
    if not source.is_file():
        logger.info(
            "uBlock Origin XPI not found at %s — skipping install for %s "
            "(profile will be stamped to v3; launch-time sync will install "
            "later when the source becomes available)",
            source, profile,
        )
        return
    if not _adblock_enabled():
        logger.info(
            "BROWSER_ENABLE_ADBLOCK=false — skipping uBlock Origin install "
            "for %s (existing extension files left untouched)", profile,
        )
        return

    extensions_dir = profile / "extensions"
    extensions_dir.mkdir(parents=True, exist_ok=True)
    target = extensions_dir / f"{UBLOCK_ADDON_ID}.xpi"

    # Idempotent copy: skip when the destination already matches the
    # source (same size + mtime within 1s). Re-applies the chmod regardless
    # so a profile restored from a backup with stale perms gets fixed.
    if target.is_file():
        try:
            src_stat = source.stat()
            tgt_stat = target.stat()
            if (
                src_stat.st_size == tgt_stat.st_size
                and abs(src_stat.st_mtime - tgt_stat.st_mtime) < 1.0
            ):
                target.chmod(0o644)
                return
        except OSError:
            # Stat failure is non-fatal — fall through to a fresh copy.
            pass

    # ``copy2`` preserves mtime so the size+mtime guard above will hit on
    # the next migration call (idempotency).
    shutil.copy2(source, target)
    target.chmod(0o644)


_MIGRATIONS: dict[int, Callable[[Path], None]] = {
    2: _v2_clear_font_caches,
    3: _v3_install_ublock,
}


# ── On-disk marker & lock file naming ──────────────────────────────────────


_MARKER_FILENAME = ".ol_schema"
_LOCK_FILENAME = ".ol_schema.lock"
_BACKUP_SUFFIX = ".bak"
_TMP_SUFFIX = ".tmp"


class ProfileMigrationBusy(Exception):
    """Raised when a peer process holds the per-profile migration lock AND
    there are pending migrations the target version needs.

    Distinct from a generic failure so the caller can treat it as
    transient (worth a short retry) rather than a hard error.
    """


# ── Public API ──────────────────────────────────────────────────────────────


def sync_adblock_extension(profile_dir: Path | str) -> bool:
    """Refresh the uBlock Origin install for ``profile_dir`` at launch.

    Distinct from :func:`migrate_profile`: that runs once per schema bump.
    This runs every launch so flipping ``BROWSER_ENABLE_ADBLOCK`` on a
    long-lived profile, or rebuilding the browser image with a newer XPI
    version, takes effect on the next browser start without needing a
    schema bump.

    Returns ``True`` if the profile now has the bundled uBlock XPI in
    place, ``False`` otherwise (source missing, adblock disabled, or
    copy failed). Never raises — extension install is best-effort and
    must not block browser launch.

    Note on uninstall: when the operator flips
    ``BROWSER_ENABLE_ADBLOCK=false`` AND the profile previously had
    uBlock, this function does *not* attempt removal. Cleanly removing
    a Firefox extension after first install requires editing
    ``extensions.json`` + the profile DB, which is fragile across
    Firefox versions. Operators wanting a hard reset should
    ``openlegion reset`` (full profile wipe).
    """
    profile = Path(profile_dir)
    if not profile.is_dir():
        return False
    # Acquire the same per-profile flock that ``migrate_profile`` uses so
    # two browser-service processes (e.g. agent restart racing the
    # provisioner update path) can't both write to ``extensions/`` at
    # once. Non-blocking — if the peer holds it, skip; the peer is
    # doing the same work. Idempotency means the eventual XPI state is
    # correct regardless of who wins.
    lock_path = profile / _LOCK_FILENAME
    try:
        with _try_lock(lock_path) as acquired:
            if not acquired:
                logger.debug(
                    "sync_adblock_extension: peer holds lock on %s; "
                    "skipping (peer will install)", profile,
                )
                return (
                    profile / "extensions" / f"{UBLOCK_ADDON_ID}.xpi"
                ).is_file()
            try:
                # Re-use the migration helper — already idempotent and
                # respects the source-missing / flag-disabled guards.
                _v3_install_ublock(profile)
            except Exception as e:
                logger.warning(
                    "Launch-time uBlock sync failed for %s: %s "
                    "(browser will start without ad-blocking)",
                    profile, e,
                )
                return False
    except OSError as e:
        # Lock-file creation failure (FS read-only, perms) — treat as
        # best-effort and continue; the browser still launches.
        logger.warning(
            "sync_adblock_extension: lock acquire failed for %s: %s",
            profile, e,
        )
        return False
    target = profile / "extensions" / f"{UBLOCK_ADDON_ID}.xpi"
    return target.is_file()


def migrate_profile(profile_dir: Path | str) -> int:
    """Bring ``profile_dir`` to :data:`PROFILE_SCHEMA_VERSION`.

    Returns the version the profile is at after the call. Idempotent —
    calling on an already-current profile is a fast no-op.

    Steps:

    1. Clean up orphan ``*.tmp`` files (crash recovery).
    2. Acquire non-blocking per-profile lock. If another process holds it,
       log + return current-on-disk version (caller launches anyway; the
       other process will finish).
    3. Read the on-disk version. If ``>= PROFILE_SCHEMA_VERSION``, release
       lock and return.
    4. Snapshot the profile to ``profile_dir.bak`` (directory-level copy,
       preserves cookies/storage/prefs).
    5. Run each registered migration for versions
       ``(current_version + 1)..PROFILE_SCHEMA_VERSION`` in order.
    6. On success: write the new marker, remove the backup, release lock.
    7. On any exception: restore from backup, re-raise. The lock is
       released in the ``finally`` so a retry can proceed next startup.

    Raises ``FileNotFoundError`` if ``profile_dir`` doesn't exist.
    Raises any exception from a migration (after restoring backup).
    """
    profile = Path(profile_dir)
    if not profile.exists():
        raise FileNotFoundError(f"Profile dir does not exist: {profile}")

    _clean_orphan_tmp(profile)

    lock_path = profile / _LOCK_FILENAME
    with _try_lock(lock_path) as acquired:
        if not acquired:
            # Another process holds the migration lock. If the on-disk
            # version is ALREADY current, they're doing a pointless
            # re-check — safe to proceed with the launch. If it's BELOW
            # target, they're mid-migration and launching Camoufox now
            # would race their writes into user_data_dir. Refuse loudly
            # so the caller (``BrowserManager.get_or_start``) can either
            # retry after a short wait or bubble up to the user.
            current = _read_marker(profile)
            if current >= PROFILE_SCHEMA_VERSION:
                logger.info(
                    "Profile %s already current (v%d); peer holds lock but "
                    "no migration pending", profile, current,
                )
                return current
            raise ProfileMigrationBusy(
                f"Another process is migrating {profile} "
                f"(on-disk v{current}, target v{PROFILE_SCHEMA_VERSION}) — "
                f"cannot launch until they finish",
            )

        current = _read_marker(profile)
        if current >= PROFILE_SCHEMA_VERSION:
            return current

        pending = sorted(
            v for v in _MIGRATIONS if current < v <= PROFILE_SCHEMA_VERSION
        )

        # Always advance the marker even when there are no transform
        # functions between current and target (Phase 1.4 ships empty
        # _MIGRATIONS but still needs to stamp profiles as version 1).
        if not pending and current < PROFILE_SCHEMA_VERSION:
            _write_marker_atomic(profile, PROFILE_SCHEMA_VERSION)
            logger.info(
                "Profile %s stamped from version %d to %d (no transforms needed)",
                profile, current, PROFILE_SCHEMA_VERSION,
            )
            return PROFILE_SCHEMA_VERSION

        backup = _snapshot_backup(profile)
        try:
            for version in pending:
                logger.info(
                    "Applying profile migration v%d → %s", version, profile,
                )
                _MIGRATIONS[version](profile)
            _write_marker_atomic(profile, PROFILE_SCHEMA_VERSION)
            logger.info(
                "Profile %s migrated from v%d to v%d",
                profile, current, PROFILE_SCHEMA_VERSION,
            )
        except Exception:
            logger.exception(
                "Migration failed for %s — restoring backup", profile,
            )
            _restore_backup(profile, backup)
            raise
        else:
            # Remove the backup only after a successful write of the new
            # marker. If we crashed between `_write_marker_atomic` and
            # this line the backup would still be on disk; that's
            # harmless — the next startup sees the current version and
            # skips the migration path, and the orphan .bak is cleaned
            # below.
            _remove_tree(backup)

        return PROFILE_SCHEMA_VERSION


# ── Helpers ────────────────────────────────────────────────────────────────


class _LockHandle:
    """Context-manager wrapper around an fcntl.flock'd file.

    ``__enter__`` returns ``True`` on exclusive acquisition, ``False`` if the
    lock was already held by another process. Always releases on exit.
    """

    def __init__(self, path: Path):
        self._path = path
        self._fd: int | None = None
        self._acquired: bool = False

    def __enter__(self) -> bool:
        self._fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._acquired = True
        except BlockingIOError:
            self._acquired = False
        return self._acquired

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None:
            try:
                if self._acquired:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


def _try_lock(lock_path: Path) -> _LockHandle:
    """Return a context manager that tries to acquire a non-blocking lock."""
    return _LockHandle(lock_path)


def _read_marker(profile: Path) -> int:
    """Return the on-disk schema version, or 0 if missing / unreadable."""
    marker = profile / _MARKER_FILENAME
    if not marker.exists():
        return 0
    try:
        raw = marker.read_text(encoding="utf-8").strip()
        return int(raw)
    except (OSError, ValueError):
        logger.warning(
            "Malformed %s at %s; treating as version 0",
            _MARKER_FILENAME, marker,
        )
        return 0


def _write_marker_atomic(profile: Path, version: int) -> None:
    """Durably write the version marker.

    Write-to-tmp + fsync + os.replace is atomic on POSIX: a reader either
    sees the old marker or the new marker, never a half-written file.
    """
    marker = profile / _MARKER_FILENAME
    tmp = profile / f"{_MARKER_FILENAME}{_TMP_SUFFIX}"

    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, f"{version}\n".encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, marker)


def _snapshot_backup(profile: Path) -> Path:
    """Copy the profile to a sibling ``.bak`` dir. Returns backup path.

    Removes any pre-existing backup first — a stale .bak from an
    earlier-crashed migration is replaced by the current pre-migration
    state, which is what we'd want to restore to.
    """
    backup = profile.with_name(profile.name + _BACKUP_SUFFIX)
    if backup.exists():
        _remove_tree(backup)
    shutil.copytree(profile, backup, symlinks=True)
    return backup


def _restore_backup(profile: Path, backup: Path) -> None:
    """Replace ``profile`` contents with ``backup``. Destroys the backup."""
    if not backup.exists():
        logger.error(
            "Cannot restore: backup missing at %s (profile left as-is)", backup,
        )
        return
    # Remove the (partially-migrated) profile contents, then move the
    # backup into place. ``shutil.move`` across same-fs does a rename.
    _remove_tree(profile)
    shutil.move(str(backup), str(profile))


def _clean_orphan_tmp(profile: Path) -> None:
    """Delete any ``*.tmp`` files left by a prior crashed write.

    Scoped to the profile root. We do NOT recurse — Firefox creates its
    own ``.tmp`` files inside ``storage/`` etc. during normal operation,
    and removing those would corrupt live state.
    """
    try:
        for entry in profile.iterdir():
            if entry.name.endswith(_TMP_SUFFIX) and entry.is_file():
                try:
                    entry.unlink()
                    logger.info(
                        "Removed orphan tmp file from prior migration: %s",
                        entry,
                    )
                except OSError:
                    logger.warning("Could not remove orphan tmp: %s", entry)
    except OSError:
        # Permissions or missing — don't block startup.
        logger.debug("Could not scan %s for orphan tmp files", profile)


def _remove_tree(path: Path) -> None:
    """Best-effort recursive remove; logs on failure, doesn't raise."""
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as e:
        logger.warning("Could not remove %s: %s", path, e)
