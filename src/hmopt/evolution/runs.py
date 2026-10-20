"""Multi-repository discovery orchestration; explicit scope, frozen commits, no approval rights."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .mining import Hotspot, mine_git_history
from .scan import control_scan, scan_next, start_scan
from .store import ConflictError, digest
from .workspace import freeze_workspace


def start_run(
    service, workspace, *, actor, request_id, projects=None, max_commits=10000, full_history=False
):
    actor = service._actor(actor)
    if type(full_history) is not bool:
        raise ValueError("full_history must be a boolean")
    if type(max_commits) is not int or not 1 <= max_commits <= 1_000_000:
        raise ValueError("max_commits must be 1..1000000 per project")
    request = {
        "workspace": workspace,
        "projects": projects,
        "max_commits": max_commits,
        "actor": actor,
        "full_history": full_history,
    }
    with service.store.transaction() as db:
        replay = service.store.replay(db, request_id, request)
        if replay is not None:
            return replay
    manifest = freeze_workspace(workspace, git_bin=service.git_bin, selected=projects)
    key = "workspace_run_" + digest([request_id, request])
    with service.store.transaction() as db:
        replay = service.store.replay(db, request_id, request)
        if replay is not None:
            return replay
        sha = service.store.evidence(db, manifest)
        states = {}
        for p in manifest["projects"]:
            existing = db.execute(
                "SELECT payload FROM records WHERE kind='workspace_claim' AND id=?", (p["repo_id"],)
            ).fetchone()
            if existing:
                raise ConflictError(
                    "Project already belongs to an unfinished workspace run: " + p["project_id"]
                )
            service.store.put(db, "workspace_claim", p["repo_id"], {"run_id": key})
            cursor_key = digest([p["repo_id"], p["revision_selector"]])
            cursor_row = db.execute(
                "SELECT payload FROM records WHERE kind='workspace_cursor' AND id=?", (cursor_key,)
            ).fetchone()
            cursor = (
                json.loads(cursor_row[0])["revision"] if cursor_row and not full_history else None
            )
            # Carry unresolved work from the prior incremental run; ingestion is
            # not a semantic-analysis completion claim.
            if cursor_row and not full_history:
                previous_run = json.loads(cursor_row[0])["run_id"]
                for source in db.execute(
                    "SELECT a.id FROM records s JOIN records a ON a.kind='history_analysis' "
                    "AND a.id=json_extract(s.payload,'$.source_id') WHERE s.kind='workspace_source' "
                    "AND json_extract(s.payload,'$.run_id')=? AND json_extract(s.payload,'$.project_id')=? "
                    "AND json_extract(a.payload,'$.status') IN ('pending','needs_context')",
                    (previous_run, p["project_id"]),
                ).fetchall():
                    service.store.put(
                        db,
                        "workspace_source",
                        digest([key, source[0]]),
                        {"run_id": key, "project_id": p["project_id"], "source_id": source[0]},
                    )
            states[p["project_id"]] = {
                "stage": "history",
                "cursor": cursor,
                "history_after": cursor,
                "cursor_key": cursor_key,
                "commits": 0,
                "campaign_id": None,
                "scan_id": None,
                "lease_until": 0,
            }
        row = service.store.put(
            db,
            "workspace_run",
            key,
            {
                "manifest_sha256": sha,
                "config_sha256": digest(workspace),
                "actor": actor,
                "max_commits": max_commits,
                "projects": states,
                "status": "running",
                "scope": "history, model research and active-pattern scanning; no approvals or source edits",
            },
        )
        service.store.remember(db, request_id, request, row)
        service.store.event(db, key, "workspace_run_created", actor, {"manifest_sha256": sha})
        return row


def run_status(service, run_id):
    # One consistent read snapshot; aggregate membership once for all selected projects.
    with service.store.transaction(read_only=True) as db:
        row = service.store.get(db, "workspace_run", run_id)
        manifest = service.store.read_evidence(row["data"]["manifest_sha256"], _db=db)
        by_project = {}
        for name, status, count in db.execute(
            "SELECT json_extract(s.payload,'$.project_id'),json_extract(a.payload,'$.status'),count(*) "
            "FROM records s JOIN records a ON a.kind='history_analysis' "
            "AND a.id=json_extract(s.payload,'$.source_id') WHERE s.kind='workspace_source' "
            "AND json_extract(s.payload,'$.run_id')=? "
            "GROUP BY json_extract(s.payload,'$.project_id'),json_extract(a.payload,'$.status')",
            (run_id,),
        ):
            by_project.setdefault(name, {})[status] = count
        scan_rows = {
            name: service.store.get(db, "scan", project["scan_id"])
            for name, project in row["data"]["projects"].items()
            if project.get("scan_id")
        }
    actions = []
    research = {}
    scans = {}
    for name, p in row["data"]["projects"].items():
        counts = by_project.get(name, {})
        research[name] = {
            "counts": counts,
            "unresolved": counts.get("pending", 0) + counts.get("needs_context", 0),
            "investigation_entry": f"/evolve-discover {name}",
        }
        if p.get("scan_id"):
            scan = scan_rows[name]
            scans[name] = {
                "scan_id": scan["id"],
                "version": scan["version"],
                **{
                    k: scan["data"][k]
                    for k in (
                        "status",
                        "pages",
                        "files",
                        "candidate_count",
                        "skipped",
                        "coverage_complete",
                    )
                },
            }
        if p["stage"] == "review":
            actions.append(
                {
                    "project_id": name,
                    "action": "Review draft patterns, activate curated versions, assess candidates and request owner decisions.",
                    "queue_entry": f"/evolve-queue {name} pending",
                    "scan_id": p.get("scan_id"),
                    "unresolved_sources": research[name]["unresolved"],
                }
            )
        elif p["stage"] in {"attention", "budget_exhausted"}:
            actions.append(
                {
                    "project_id": name,
                    "action": p.get("error", p["stage"]),
                    "resume": "Inspect the project checkpoint; explicit retry preserves its frozen revision.",
                }
            )
    return {
        "run": row,
        "manifest": manifest,
        "actions": actions,
        "research": research,
        "scans": scans,
        "execution_entry": "/evolve-candidate <approved-candidate-id> full",
    }


def _save_project(service, run_id, name, expected, state):
    with service.store.transaction() as db:
        row = service.store.get(db, "workspace_run", run_id)
        if row["data"]["projects"][name] != expected or row["data"]["status"] == "cancelled":
            raise ConflictError("Workspace project checkpoint was superseded")
        if expected["stage"] == "history" and state["stage"] == "analysis":
            cursor_row = db.execute(
                "SELECT 1 FROM records WHERE kind='workspace_cursor' AND id=?",
                (state["cursor_key"],),
            ).fetchone()
            version = (
                service.store.get(db, "workspace_cursor", state["cursor_key"])["version"]
                if cursor_row
                else None
            )
            service.store.put(
                db,
                "workspace_cursor",
                state["cursor_key"],
                {"revision": state["cursor"], "run_id": run_id},
                version,
            )
        row["data"]["projects"][name] = state
        stages = {p["stage"] for p in row["data"]["projects"].values()}
        row["data"]["status"] = (
            "awaiting_review"
            if stages <= {"review"}
            else "attention"
            if stages <= {"review", "attention", "budget_exhausted"}
            else "running"
        )
        if state["stage"] == "review":
            manifest = service.store.read_evidence(row["data"]["manifest_sha256"], _db=db)
            repo_id = next(p["repo_id"] for p in manifest["projects"] if p["project_id"] == name)
            db.execute(
                "DELETE FROM records WHERE kind='workspace_claim' AND id=? AND json_extract(payload,'$.run_id')=?",
                (repo_id, run_id),
            )
        return service.store.put(db, "workspace_run", run_id, row["data"], row["version"])


def advance_run(service, run_id, *, worker_config=None, page_size=20):
    """One step per project; faults are isolated and all external jobs remain queryable."""
    if type(page_size) is not int or not 1 <= page_size <= 100:
        raise ValueError("page_size must be 1..100")
    row = service.store.read("workspace_run", run_id)
    if row["data"]["status"] == "cancelled":
        return run_status(service, run_id)
    manifest = service.store.read_evidence(row["data"]["manifest_sha256"])
    for project in manifest["projects"]:
        name = project["project_id"]
        with service.store.transaction() as db:
            row = service.store.get(db, "workspace_run", run_id)
            state = row["data"]["projects"][name]
            if (
                row["data"]["status"] == "cancelled"
                or state["stage"] in {"review", "attention", "budget_exhausted"}
                or state["lease_until"] > time.time()
            ):
                continue
            state["lease_until"] = time.time() + 300
            row = service.store.put(db, "workspace_run", run_id, row["data"], row["version"])
            expected = json.loads(json.dumps(state))
        try:
            repo = project["repo_path"]
            if service.repo_identity(repo) != project["repo_id"]:
                raise ConflictError("Configured project identity changed")
            if state["stage"] == "history":
                remaining = row["data"]["max_commits"] - state["commits"]
                if remaining <= 0:
                    state.update(
                        stage="budget_exhausted",
                        error="History budget exhausted; explicitly increase the run budget to continue.",
                    )
                else:
                    changes = mine_git_history(
                        Path(repo),
                        revision=project["revision"],
                        after_revision=state["cursor"],
                        max_commits=min(page_size, remaining),
                        git_bin=service.git_bin,
                        repo_id=project["repo_id"],
                    )
                    service.ingest_history(changes)
                    # Record run membership independently of a global ingestion cursor.
                    with service.store.transaction() as db:
                        for change in changes:
                            service.store.put(
                                db,
                                "workspace_source",
                                digest([run_id, change.source_id]),
                                {
                                    "run_id": run_id,
                                    "project_id": name,
                                    "source_id": change.source_id,
                                },
                            )
                    if changes:
                        state["cursor"] = changes[-1].revision
                        state["commits"] += len(changes)
                    if state["cursor"] == project["revision"]:
                        state["stage"] = "analysis"
            elif state["stage"] == "analysis":
                from .worker import campaign_status, create_campaign

                if state["campaign_id"]:
                    campaign = campaign_status(service, state["campaign_id"])
                    if not campaign["finished"]:
                        state["lease_until"] = 0
                        _save_project(service, run_id, name, expected, state)
                        continue
                    state["campaign_id"] = None
                with service.store.transaction(read_only=True) as db:
                    records = db.execute(
                        "SELECT a.id FROM records s JOIN records a ON a.kind='history_analysis' "
                        "AND a.id=json_extract(s.payload,'$.source_id') WHERE s.kind='workspace_source' "
                        "AND json_extract(s.payload,'$.run_id')=? AND json_extract(s.payload,'$.project_id')=? "
                        "AND json_extract(a.payload,'$.status')='pending' AND NOT EXISTS "
                        "(SELECT 1 FROM records j WHERE j.kind='mining_job' "
                        "AND json_extract(j.payload,'$.source_id')=a.id "
                        "AND json_extract(j.payload,'$.status') IN ('attention','cancelled')) "
                        "ORDER BY a.rowid LIMIT 100",
                        (run_id, name),
                    ).fetchall()
                pending = [r["id"] for r in records]
                if pending and worker_config:
                    campaign = create_campaign(
                        service,
                        repo,
                        pending,
                        worker_config,
                        actor=row["data"]["actor"],
                        request_id="run-mining:" + digest([run_id, name, pending]),
                    )
                    state["campaign_id"] = campaign["id"]
                elif records:
                    state.update(
                        stage="attention",
                        error="Pending research requires production.worker or workbench analysis.",
                    )
                else:
                    state["stage"] = "scan"
                    state["research_note"] = (
                        "Unresolved/failed sources remain explicit in run status. Only independently activated patterns are scanned; they do not establish full history coverage."
                    )
            elif state["stage"] == "scan":
                if state["scan_id"] is None:
                    scan = start_scan(
                        service,
                        repo,
                        revision=project["revision"],
                        owners=project["owners"],
                        hotspots=[Hotspot.model_validate(h) for h in project.get("hotspots", [])],
                        request_id=f"run-scan:{run_id}:{name}",
                        actor=row["data"]["actor"],
                        workspace_manifest_sha256=row["data"]["manifest_sha256"],
                        _run_checkpoint=(run_id, name, expected),
                    )
                    state["scan_id"] = scan["id"]
                    expected["scan_id"] = scan["id"]
                scan = service.store.read("scan", state["scan_id"])
                if scan["data"]["status"] == "running":
                    scan = scan_next(service, scan["id"], expected_version=scan["version"])
                if scan["data"]["status"] in {"complete", "no_active_patterns"}:
                    state.update(stage="review", scan_status=scan["data"]["status"])
                elif scan["data"]["status"] in {"attention", "cancelled"}:
                    state.update(
                        stage="attention",
                        resume_stage="scan",
                        error="Scan requires inspection: "
                        + scan["data"].get("error", scan["data"]["status"]),
                    )
            state["lease_until"] = 0
            _save_project(service, run_id, name, expected, state)
        except (ValueError, OSError, TimeoutError) as error:
            failed = {
                **expected,
                "stage": "attention",
                "resume_stage": expected["stage"],
                "lease_until": 0,
                "error": str(error)[:1000],
            }
            _save_project(service, run_id, name, expected, failed)
    return run_status(service, run_id)


def control_run(
    service, run_id, *, action, actor, expected_version, project_id=None, max_commits=None
):
    actor = service._actor(actor)
    with service.store.transaction() as db:
        row = service.store.get(db, "workspace_run", run_id)
        if row["version"] != expected_version:
            raise ConflictError("Workspace run changed; refresh its version")
        data = row["data"]
        if any(p["lease_until"] > time.time() for p in data["projects"].values()):
            raise ConflictError("A project step is running; wait before controlling the run")
        if action == "cancel":
            data["status"] = "cancelled"
            for state in data["projects"].values():
                if state.get("scan_id"):
                    scan = service.store.get(db, "scan", state["scan_id"])
                    if scan["data"]["status"] in {"running", "attention"}:
                        control_scan(
                            service,
                            scan["id"],
                            action="cancel",
                            actor=actor,
                            expected_version=scan["version"],
                            _db=db,
                        )
                if state.get("campaign_id"):
                    campaign = service.store.get(db, "mining_campaign", state["campaign_id"])
                    for job_id in campaign["data"]["job_ids"]:
                        job = service.store.get(db, "mining_job", job_id)
                        if job["data"]["status"] not in {
                            "complete",
                            "needs_context",
                            "attention",
                            "cancelled",
                        }:
                            job["data"]["status"] = "cancelled"
                            service.store.put(db, "mining_job", job_id, job["data"], job["version"])
            db.execute(
                "DELETE FROM records WHERE kind='workspace_claim' AND json_extract(payload,'$.run_id')=?",
                (run_id,),
            )
        elif action == "retry" and project_id in data["projects"]:
            state = data["projects"][project_id]
            if data["status"] == "cancelled" or state["stage"] not in {
                "attention",
                "budget_exhausted",
            }:
                raise ValueError(
                    "Retry selects an attention/budget-exhausted project in an active run"
                )
            if max_commits is not None:
                if (
                    type(max_commits) is not int
                    or not data["max_commits"] < max_commits <= 1_000_000
                ):
                    raise ValueError(
                        "Explicit history budget extension must increase the existing limit"
                    )
                data["max_commits"] = max_commits
            stage = state.get(
                "resume_stage", "history" if state["stage"] == "budget_exhausted" else "analysis"
            )
            if stage == "scan" and state.get("scan_id"):
                scan = service.store.get(db, "scan", state["scan_id"])
                if scan["data"]["status"] == "cancelled":
                    raise ValueError(
                        "Attached scan was cancelled; cancel this run and start a new run"
                    )
                if scan["data"]["status"] == "attention":
                    control_scan(
                        service,
                        scan["id"],
                        action="retry",
                        actor=actor,
                        expected_version=scan["version"],
                        _db=db,
                    )
            state.update(stage=stage, lease_until=0)
            state.pop("resume_stage", None)
            state.pop("error", None)
            data["status"] = "running"
        else:
            raise ValueError("Choose cancel or retry with an explicit blocked project_id")
        service.store.event(
            db,
            run_id,
            "workspace_run_" + action,
            actor,
            {"project_id": project_id, "max_commits": max_commits},
        )
        return service.store.put(db, "workspace_run", run_id, data, row["version"])
