"""One operator process for authorized discovery, research and optional notification delivery."""

from __future__ import annotations

import os
import time

from .store import ConflictError, canonical_json


def runtime_cycle(service, config, *, worker_id, notifications=False):
    from .approval import deliver_notifications
    from .runs import advance_run
    from .scan import scan_next
    from .worker import worker_cycle

    if notifications and config.approval is None:
        raise ValueError("Notification delivery requires production.approval")
    result = {"runs": [], "scans": [], "research": [], "notifications": None, "errors": []}
    run_scans = set()
    with service.store.transaction(read_only=True) as db:
        runs = [
            r[0]
            for r in db.execute(
                "SELECT id FROM records WHERE kind='workspace_run' AND json_extract(payload,'$.status')='running' ORDER BY rowid LIMIT 32"
            )
        ]
    for run_id in runs:
        try:
            report = advance_run(service, run_id, worker_config=config.worker)
            run_scans.update(scan["scan_id"] for scan in report["scans"].values())
            result["runs"].append(
                {
                    "run_id": run_id,
                    "status": report["run"]["data"]["status"],
                    "actions": report["actions"],
                }
            )
        except (ValueError, OSError, TimeoutError) as error:
            result["errors"].append({"run_id": run_id, "error": str(error)[:1000]})
    with service.store.transaction(read_only=True) as db:
        scans = [
            service.store.get(db, "scan", r[0])
            for r in db.execute(
                "SELECT id FROM records WHERE kind='scan' AND json_extract(payload,'$.status')='running' "
                "AND json_extract(payload,'$.lease_until')<=? "
                "AND id NOT IN (SELECT value FROM json_each(?)) ORDER BY rowid LIMIT 16",
                (time.time(), canonical_json(sorted(run_scans))),
            ).fetchall()
        ]
    for scan in scans:
        try:
            updated = scan_next(service, scan["id"], expected_version=scan["version"])
            result["scans"].append(
                {
                    "scan_id": scan["id"],
                    "status": updated["data"]["status"],
                    "pages": updated["data"]["pages"],
                }
            )
        except (ValueError, OSError, TimeoutError) as error:
            result["errors"].append({"scan_id": scan["id"], "error": str(error)[:1000]})
    if config.worker:
        try:
            result["research"] = worker_cycle(service, config.worker, worker_id=worker_id)
        except (ValueError, OSError, TimeoutError) as error:
            result["errors"].append({"component": "research", "error": str(error)[:1000]})
    if notifications:
        result["notifications"] = deliver_notifications(
            service, config.approval, actor=worker_id, limit=20
        )
    with service.store.transaction() as db:
        exists = db.execute(
            "SELECT 1 FROM records WHERE kind='runtime' AND id=?", (worker_id,)
        ).fetchone()
        version = service.store.get(db, "runtime", worker_id)["version"] if exists else None
        service.store.put(
            db,
            "runtime",
            worker_id,
            {
                "pid": os.getpid(),
                "last_cycle_at": time.time(),
                "notifications_enabled": notifications,
                "status": "attention" if result["errors"] else "running",
                "last_cycle": result,
            },
            version,
        )
    return result


def require_workspace_scope(service, workspace):
    from .workspace import identities

    if identities(workspace) != service.repo_identities:
        raise ConflictError("Runtime identity mapping differs from its configured workspace")
