"""One-shot blackboard → orchestration tasks migration.

Walks every legacy ``tasks/{agent}/{handoff_id}`` and
``global/tasks/operator/{handoff_id}`` (and their project-prefixed
forms ``projects/{name}/tasks/{agent}/{handoff_id}``) blackboard entry,
inserts a corresponding row in the durable ``tasks`` table, and deletes
the legacy keys on success.

Auto-runs at mesh startup so existing fleets transition seamlessly.
Idempotent — tasks are keyed on the legacy ``handoff_id`` so a partial
run + restart skips already-migrated handoffs. Returns
``{migrated, skipped, deleted, errors}`` summary.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.shared.utils import setup_logging

logger = setup_logging("host.orchestration_migration")


# Recognized legacy task key shapes:
#   tasks/{agent}/{handoff_id}                              -- standalone path
#   projects/{project}/tasks/{agent}/{handoff_id}           -- project-scoped
#   global/tasks/operator/{handoff_id}                      -- operator inbox
_LEGACY_PROJECT_RE = re.compile(
    r"^projects/(?P<project>[^/]+)/tasks/(?P<assignee>[^/]+)/(?P<handoff_id>[^/]+)$"
)
_LEGACY_BARE_RE = re.compile(
    r"^tasks/(?P<assignee>[^/]+)/(?P<handoff_id>[^/]+)$"
)
_LEGACY_OPERATOR_RE = re.compile(
    r"^global/tasks/operator/(?P<handoff_id>[^/]+)$"
)


def _coerce_value(value: Any) -> dict:
    """Coerce a blackboard ``entry.value`` into a dict.

    Accepts the typical handoff dict, falls back to ``{"text": str(...)}``
    for legacy / corrupt rows so the migration does not blow up partway
    through.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"text": value}
        except (json.JSONDecodeError, TypeError):
            return {"text": value}
    return {"text": str(value)}


def _classify_key(key: str) -> tuple[str, dict[str, str]] | None:
    """Match ``key`` against the three legacy shapes. Returns (kind, parts)."""
    m = _LEGACY_OPERATOR_RE.match(key)
    if m:
        return ("operator", m.groupdict())
    m = _LEGACY_PROJECT_RE.match(key)
    if m:
        return ("project", m.groupdict())
    m = _LEGACY_BARE_RE.match(key)
    if m:
        return ("bare", m.groupdict())
    return None


def migrate_blackboard_to_tasks(blackboard, tasks) -> dict:
    """Migrate every legacy task blackboard entry into the ``tasks`` table.

    Idempotent. Tasks are keyed on the legacy ``handoff_id`` so partial
    migrations resume cleanly. After successful insert, the source
    blackboard key is deleted; if the insert errors, the legacy key is
    preserved so the migration can be re-run.

    Args:
        blackboard: a ``Blackboard`` instance from ``src.host.mesh``.
        tasks: a ``Tasks`` instance from ``src.host.orchestration``.

    Returns:
        ``{migrated, skipped, deleted, errors}`` — counts of newly
        migrated handoffs, already-existing-and-skipped handoffs,
        legacy keys deleted, and per-key errors collected.
    """
    summary = {
        "migrated": 0,
        "skipped": 0,
        "deleted": 0,
        "errors": [],
    }

    # Walk both top-level prefixes. ``list_by_prefix`` returns
    # ``BlackboardEntry`` objects with the project scope already in the
    # key; we re-classify here so the migration handles bare /
    # project-scoped / global all in one pass.
    candidates = []
    for prefix in ("tasks/", "projects/", "global/tasks/operator/"):
        try:
            candidates.extend(blackboard.list_by_prefix(prefix))
        except Exception as e:
            summary["errors"].append({"prefix": prefix, "error": str(e)})

    # Dedupe on key — ``projects/`` and ``tasks/`` may overlap when both
    # forms exist.
    seen_keys: set[str] = set()

    for entry in candidates:
        key = entry.key
        if key in seen_keys:
            continue
        seen_keys.add(key)

        classified = _classify_key(key)
        if classified is None:
            continue
        kind, parts = classified

        handoff_id = parts["handoff_id"]
        # Idempotent: skip if a task row with this id already exists.
        if tasks.get(handoff_id) is not None:
            summary["skipped"] += 1
            try:
                blackboard.delete(key, deleted_by="orchestration-migration")
                summary["deleted"] += 1
            except Exception as e:
                summary["errors"].append({"key": key, "error": str(e)})
            continue

        value = _coerce_value(entry.value)

        if kind == "operator":
            assignee = "operator"
            project_id = None
        elif kind == "project":
            assignee = parts["assignee"]
            project_id = parts["project"]
        else:  # bare
            assignee = parts["assignee"]
            project_id = None

        creator = str(value.get("from") or entry.written_by or "unknown")
        summary_text = str(value.get("summary") or "(migrated handoff)")
        # Preserve the original output_key in the artifact_refs list so
        # operators can still find the legacy output blob if/when it is
        # also being migrated. After migration runs to completion, the
        # output keys remain in the blackboard until separate cleanup
        # — we do not auto-delete them here because they may live under
        # a different prefix that another tool is responsible for.
        artifact_refs = []
        output_key = value.get("output_key")
        if isinstance(output_key, str) and output_key:
            artifact_refs.append(output_key)

        origin = value.get("origin")
        if not isinstance(origin, dict):
            origin = None

        legacy_status = value.get("status") or "pending"
        try:
            tasks.create(
                creator=creator,
                assignee=assignee,
                title=summary_text[:200] or "(migrated handoff)",
                description=summary_text,
                project_id=project_id,
                priority=0,
                artifact_refs=artifact_refs or None,
                origin=origin,
                task_id=handoff_id,
            )
            # If the legacy task carried a status other than "pending",
            # apply the transition so the migrated row reflects the
            # current state. Skip silently when the transition is not
            # a single hop — the safer default is "leave as pending"
            # rather than corrupt the audit trail.
            if legacy_status in ("working", "blocked", "done"):
                try:
                    tasks.update_status(
                        handoff_id, legacy_status,
                        actor="orchestration-migration",
                    )
                except Exception:
                    # Multi-hop transition — keep at "pending" so the
                    # invariant holds. The audit event still records the
                    # creation.
                    pass
        except Exception as e:
            summary["errors"].append({"key": key, "error": str(e)})
            continue

        try:
            blackboard.delete(key, deleted_by="orchestration-migration")
            summary["deleted"] += 1
        except Exception as e:
            summary["errors"].append({"key": key, "error": str(e)})
        summary["migrated"] += 1

    logger.info(
        "orchestration migration complete: migrated=%d skipped=%d deleted=%d errors=%d",
        summary["migrated"], summary["skipped"], summary["deleted"],
        len(summary["errors"]),
    )
    return summary
