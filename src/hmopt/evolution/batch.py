"""Explicit, serial execution batches with durable claims and isolated Git worktrees.

The workbench runs Agents. This module stages work and derives completion from
authoritative gates; it never runs models, source scripts, builds or devices.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .catalog import STAGES, approvals
from .methods import skill_snapshot
from .mining import _Git
from .store import ConflictError, digest

SCOPES = {"full", "stage", "plan", "implement", "review", "validate"}
NEXT = {
    "confirmed": "plan_approved",
    "plan_approved": "implemented",
    "implemented": "code_approved",
    "code_approved": "validated",
}
START = {
    "plan": "confirmed",
    "implement": "plan_approved",
    "review": "implemented",
    "validate": "code_approved",
}
STAGE_ORDER = [*NEXT, "validated"]


def create_batch(service, repo, candidate_ids, *, scope, actor, request_id):
    actor = service._actor(actor)
    if (
        scope not in SCOPES
        or not 1 <= len(candidate_ids) <= 10
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise ValueError("Choose one to ten distinct candidate IDs and an explicit execution scope")
    request = {
        "action": "create_execution_batch",
        "repo": str(Path(repo).resolve()),
        "candidate_ids": candidate_ids,
        "scope": scope,
        "actor": actor,
    }
    with service.store.transaction() as db:
        replay = service.store.replay(db, request_id, request)
    if replay is not None:
        return replay
    method = skill_snapshot(service, ["evolution-batch", "evolution-execution"])
    receipts = {identity: approvals(service, identity) for identity in candidate_ids}
    identity = "execution_" + digest([request, request_id])
    with service.store.transaction() as db:
        replay = service.store.replay(db, request_id, request)
        if replay is not None:
            return replay
        items = []
        for candidate_id in candidate_ids:
            row = service.store.get(db, "candidate", candidate_id)
            data = row["data"]
            if (
                Path(data["candidate"]["repo_path"]).resolve() != Path(repo).resolve()
                or data["stage"] not in STAGES
                or data.get("validation")
            ):
                raise ValueError("Select approved, executable candidates from this profile")
            if scope in START and data["stage"] != START[scope]:
                raise ValueError("Named batch scope requires its exact prerequisite stage")
            if not any(r["data"]["decision"] == "confirm" for r in receipts[candidate_id]):
                raise ValueError("Candidate has no archived owner confirmation")
            prior = db.execute(
                "SELECT 1 FROM records WHERE kind='execution_claim' AND id=?", (candidate_id,)
            ).fetchone()
            if prior:
                raise ConflictError(
                    "Candidate is reserved by another batch; finish or explicitly block it first"
                )
            if db.execute(
                "SELECT 1 FROM records WHERE kind='dispatch' AND json_extract(payload,'$.candidate_id')=? "
                "AND json_extract(payload,'$.candidate_version')=? LIMIT 1",
                (candidate_id, row["version"]),
            ).fetchone():
                raise ConflictError(
                    "Candidate already has a current dispatch; finish that stage before starting a batch"
                )
            service.store.put(db, "execution_claim", candidate_id, {"batch_id": identity})
            items.append(
                {
                    "candidate_id": candidate_id,
                    "selected_version": row["version"],
                    "selected_digest": digest(data),
                    "initial_stage": data["stage"],
                    "target_stage": "validated" if scope == "full" else NEXT[data["stage"]],
                    "approval_ids": [r["id"] for r in receipts[candidate_id]],
                    "status": "pending",
                }
            )
        result = service.store.put(
            db,
            "execution_batch",
            identity,
            {
                "schema_version": 1,
                "repo_path": request["repo"],
                "scope": scope,
                "actor": actor,
                "status": "pending",
                "items": items,
                "method_sha256": method["sha256"],
            },
        )
        service.store.event(db, identity, "create_execution_batch", actor, request)
        service.store.remember(db, request_id, request, result)
    return result


def _isolate(service, row, method_sha256):
    data = row["data"]
    candidate = data["candidate"]
    if data.get("execution_context"):
        service._execution_repo(candidate, data["execution_context"])
        return row
    path = service.store.root / "worktrees" / digest(row["id"])
    if path.resolve() != path or path.parent.resolve() != service.store.root / "worktrees":
        raise ValueError("Execution worktree directory is redirected")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use argv, bounded output/time, and explicitly disable checkout hooks/filters.
    # Creation is recoverable if interrupted before its store update.
    if not path.exists():
        git = _Git(Path(candidate["repo_path"]), service.git_bin)
        raw, cut = git.read(["config", "--null", "--list"], 1024 * 1024)
        if cut:
            raise ValueError("Git configuration exceeds the worktree preparation budget")
        options = [
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.hooksPath=" + str(service.store.root / "disabled-hooks"),
        ]
        for setting in raw.decode("utf-8").split("\x00"):
            key = setting.split("\n", 1)[0]
            if key.lower().startswith("filter.") and key.rsplit(".", 1)[-1].lower() in {
                "process",
                "smudge",
                "required",
            }:
                options += ["-c", key + ("=false" if key.lower().endswith(".required") else "=")]
        _, cut = git.read(
            [
                *options,
                "worktree",
                "add",
                "--detach",
                str(path),
                data.get("implementation", {}).get("revision", candidate["repo_revision"]),
            ],
            65536,
        )
        if cut:
            raise ValueError(
                "Worktree preparation output exceeded its budget; inspect before retry"
            )
    context = {
        "candidate_id": row["id"],
        "repo_path": str(path),
        "candidate_digest": digest(candidate),
        "baseline_revision": candidate["repo_revision"],
        "method_sha256": method_sha256,
    }
    service._execution_repo(candidate, context)
    expected = data.get("implementation", {}).get("revision", candidate["repo_revision"])
    if (
        service._revision(path, "HEAD") != expected
        or service._git(path, "status", "--porcelain").strip()
    ):
        raise ConflictError(
            "Existing worktree differs from its expected clean revision; inspect before recovery"
        )
    with service.store.transaction() as db:
        current = service.store.get(db, "candidate", row["id"])
        if current != row:
            raise ConflictError("Candidate changed while isolating execution")
        service.store.put(db, "execution_context", row["id"], context)
        return service.store.put(
            db, "candidate", row["id"], {**data, "execution_context": context}, row["version"]
        )


def _save(service, db, row, data):
    statuses = {item["status"] for item in data["items"]}
    data["status"] = (
        "running"
        if statuses & {"pending", "running"}
        else "completed_with_blocks"
        if "blocked" in statuses
        else "completed"
    )
    if data == row["data"]:
        return row
    return service.store.put(db, "execution_batch", row["id"], data, row["version"])


def next_candidate(service, batch_id, *, worker_id):
    """Resume the same claim; another worker cannot take a running claim on timeout."""
    worker_id = service._actor(worker_id)
    with service.store.transaction() as db:
        row = service.store.get(db, "execution_batch", batch_id)
        data = deepcopy(row["data"])
        for item in data["items"]:
            if item["status"] == "running":
                if item["worker_id"] != worker_id:
                    raise ConflictError(
                        "Batch already has a running worker; inspect it before explicit recovery"
                    )
                current = service.store.get(db, "candidate", item["candidate_id"])
                state = current["data"]
                if state["stage"] == item["target_stage"]:
                    item.update(status="completed", result_version=current["version"])
                elif state["stage"] in STAGE_ORDER and STAGE_ORDER.index(
                    state["stage"]
                ) > STAGE_ORDER.index(item["target_stage"]):
                    item.update(
                        status="blocked",
                        reason="Candidate advanced beyond the selected scope; inspect external execution",
                        result_version=current["version"],
                    )
                elif state["stage"] == "rejected" or state.get("validation"):
                    item.update(
                        status="blocked",
                        reason="Candidate rejected or validation did not pass",
                        result_version=current["version"],
                    )
                if item["status"] != "running":
                    db.execute(
                        "DELETE FROM records WHERE kind='execution_claim' AND id=?",
                        (item["candidate_id"],),
                    )
        selected = next(
            (item for item in data["items"] if item["status"] in {"running", "pending"}), None
        )
        if selected and selected["status"] == "pending":
            current = service.store.get(db, "candidate", selected["candidate_id"])
            if (
                current["version"] != selected["selected_version"]
                or digest(current["data"]) != selected["selected_digest"]
            ):
                selected.update(
                    status="blocked",
                    reason="Candidate changed after batch selection; select its current version again",
                )
                db.execute(
                    "DELETE FROM records WHERE kind='execution_claim' AND id=?",
                    (selected["candidate_id"],),
                )
                selected = None
            else:
                selected.update(status="running", worker_id=worker_id)
        row = _save(service, db, row, data)
    if selected is None:
        return {
            "batch": row,
            "item": None,
            "instruction": "Inspect status; call next again if pending items remain.",
        }
    current = service.store.read("candidate", selected["candidate_id"])
    current = _isolate(service, current, row["data"]["method_sha256"])
    packet = service.handoff(current["id"])
    return {
        "batch": service.store.read("execution_batch", batch_id),
        "item": selected,
        "handoff": packet,
        "entry": f"/evolve-candidate {current['id']} {data['scope']}",
        "instruction": "Resume the claimed candidate using evolution-execution. Call next only after its child roles stop.",
    }


def block_candidate(service, batch_id, *, worker_id, reason, expected_version):
    """Explicit stop acknowledgement, never an automatic timeout takeover."""
    worker_id = service._actor(worker_id)
    if type(expected_version) is not int or expected_version < 1:
        raise ValueError("expected_version must be a positive integer")
    service._note({"note": reason})
    with service.store.transaction() as db:
        row = service.store.get(db, "execution_batch", batch_id)
        if row["version"] != expected_version:
            raise ConflictError("Batch changed; reread before acknowledging stopped execution")
        data = deepcopy(row["data"])
        item = next((i for i in data["items"] if i["status"] == "running"), None)
        if item is None or item["worker_id"] != worker_id:
            raise ConflictError(
                "Only the current worker can acknowledge that its child execution stopped"
            )
        item.update(status="blocked", reason=reason)
        candidate = service.store.get(db, "candidate", item["candidate_id"])
        stopped = {"batch_id": batch_id, "worker_id": worker_id, "reason": reason}
        service.store.put(
            db,
            "candidate",
            candidate["id"],
            {**candidate["data"], "execution_stop": stopped},
            candidate["version"],
        )
        service.store.event(db, candidate["id"], "execution_stopped", worker_id, stopped)
        db.execute(
            "DELETE FROM records WHERE kind='execution_claim' AND id=?", (item["candidate_id"],)
        )
        service.store.event(
            db,
            batch_id,
            "execution_stopped",
            worker_id,
            {"candidate_id": item["candidate_id"], "reason": reason},
        )
        return _save(service, db, row, data)
