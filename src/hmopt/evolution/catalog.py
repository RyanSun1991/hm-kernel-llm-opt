"""Candidate dossiers, immutable approval receipts, and paged execution readiness."""

from __future__ import annotations

import json
from pathlib import Path

from .store import digest

STAGES = {"confirmed", "plan_approved", "implemented", "code_approved"}


def archive_approval(service, db, candidate, request, evidence, approved_context):
    identity = "approval_" + digest(request)
    record = {
        "candidate_id": request["candidate_id"],
        "decision": request["action"],
        "actor": request["actor"],
        "note": service._note(request["payload"]),
        "before_version": request["expected_version"],
        "after_version": request["expected_version"] + 1,
        "candidate_digest": digest(candidate),
        "baseline_revision": candidate["repo_revision"],
        "pattern_key": f"{candidate['pattern_id']}@{candidate['pattern_version']}",
        "request_evidence": evidence,
        "context_sha256": service.store.evidence(db, approved_context),
        "created_at": db.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')").fetchone()[0],
        "identity_assurance": "trusted_local_operator_label",
    }
    if request.get("identity"):
        record["identity_assurance"] = "authenticated_approval_gateway"
        record["authenticated_identity"] = request["identity"]
    service.store.put(db, "approval", identity, record)
    return identity


def approvals(service, candidate_id):
    """Recover legacy receipts from original request evidence without changing candidate versions."""
    with service.store.transaction() as db:
        events = db.execute(
            "SELECT * FROM events WHERE entity_id=? AND action IN ('confirm','reject') ORDER BY sequence",
            (candidate_id,),
        ).fetchall()
        for event in events:
            details = json.loads(event["details"])
            sha = details.get("evidence")
            raw = db.execute("SELECT content FROM evidence WHERE sha256=?", (sha,)).fetchone()
            if raw is None:
                continue
            request = json.loads(raw[0])
            if digest(request) != sha or request.get("candidate_id") != candidate_id:
                raise ValueError("Approval request evidence integrity check failed")
            identity = "approval_" + digest(request)
            if not db.execute(
                "SELECT 1 FROM records WHERE kind='approval' AND id=?", (identity,)
            ).fetchone():
                service.store.put(
                    db,
                    "approval",
                    identity,
                    {
                        "candidate_id": candidate_id,
                        "decision": event["action"],
                        "actor": event["actor"],
                        "note": request.get("payload", {}).get("note"),
                        "before_version": request["expected_version"],
                        "after_version": details["version"],
                        "request_evidence": sha,
                        "created_at": event["created_at"],
                        "identity_assurance": "legacy_trusted_local_operator_label",
                        "legacy_event_sequence": event["sequence"],
                    },
                )
        rows = db.execute(
            "SELECT id FROM records WHERE kind='approval' AND json_extract(payload,'$.candidate_id')=? ORDER BY rowid",
            (candidate_id,),
        ).fetchall()
        return [service.store.get(db, "approval", row["id"]) for row in rows]


def readiness(service, row):
    data = row["data"]
    with service.store.transaction(read_only=True) as db:
        claim = db.execute(
            "SELECT payload FROM records WHERE kind='execution_claim' AND id=?", (row["id"],)
        ).fetchone()
    if claim:
        return {
            "ready": False,
            "reason": "Candidate is reserved; resume its batch instead of a standalone dispatch.",
            "batch_id": json.loads(claim[0])["batch_id"],
        }
    if data["stage"] not in STAGES or data.get("validation"):
        return {
            "ready": False,
            "reason": "No executable stage; inspect current decision/validation.",
        }
    try:
        packet = service.handoff(row["id"])
    except (ValueError, OSError, TimeoutError) as error:
        return {"ready": False, "reason": str(error)}
    return {
        "ready": True,
        "next_role": packet["role"],
        "source_changes_allowed": packet["source_changes_allowed"],
    }


def candidates(service, repo, *, owner=None, state="approved", limit=20, offset=0):
    if state not in {"all", "pending", "approved", "ready", "completed", "rejected"}:
        raise ValueError("Unknown candidate queue state")
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
        raise ValueError("limit must be 1..100 and offset nonnegative")
    root = Path(repo).resolve()

    def selected(raw):
        value = json.loads(raw)
        candidate = value["candidate"]
        if Path(candidate["repo_path"]).resolve() != root or (
            owner is not None and candidate["owner"] != owner
        ):
            return False
        stage = value["stage"]
        return (
            state == "all"
            or (state == "pending" and stage == "discovered")
            or (state in {"approved", "ready"} and stage in STAGES)
            or (state == "completed" and stage == "validated")
            or (state == "rejected" and stage == "rejected")
        )

    with service.store.transaction(read_only=True) as db:
        db.create_function("selected_candidate", 1, selected)
        rows = db.execute(
            "SELECT id FROM records WHERE kind='candidate' AND selected_candidate(payload)=1 ORDER BY rowid LIMIT ? OFFSET ?",
            (limit + 1, offset),
        ).fetchall()
        page = [service.store.get(db, "candidate", row["id"]) for row in rows[:limit]]
    items = []
    for row in page:
        data = row["data"]
        ready = readiness(service, row)
        if state == "ready" and not ready["ready"]:
            continue
        receipts = approvals(service, row["id"])
        items.append(
            {
                "candidate_id": row["id"],
                "version": row["version"],
                "stage": data["stage"],
                "owner": data["candidate"]["owner"],
                "path": data["candidate"]["path"],
                "repo_id": data["candidate"].get("repo_id"),
                "workspace_manifest_sha256": data["candidate"].get("workspace_manifest_sha256"),
                "baseline_revision": data["candidate"]["repo_revision"],
                "approval_ids": [r["id"] for r in receipts],
                **ready,
            }
        )
    return {
        "items": items,
        "inspected": len(page),
        "has_more": len(rows) > limit,
        "next_offset": offset + len(page) if len(rows) > limit else None,
        "pagination": "Offset advances inspected records, including blocked records in ready view.",
    }


def dossier(service, candidate_id):
    row = service.store.read("candidate", candidate_id)
    candidate = row["data"]["candidate"]
    key = f"{candidate['pattern_id']}@{candidate['pattern_version']}"
    with service.store.transaction(read_only=True) as db:
        pattern = service.store.get(db, "pattern", key)
        linked = {}
        for kind in (
            "dispatch",
            "research",
            "execution_context",
            "approval_request",
            "continuation",
            "experiment",
        ):
            rows = db.execute(
                "SELECT id FROM records WHERE kind=? AND json_extract(payload,'$.candidate_id')=? ORDER BY rowid LIMIT 101",
                (kind, candidate_id),
            ).fetchall()
            linked[kind] = {"ids": [r["id"] for r in rows[:100]], "has_more": len(rows) > 100}
        batches = db.execute(
            "SELECT id FROM records WHERE kind='execution_batch' AND EXISTS "
            "(SELECT 1 FROM json_each(payload,'$.items') WHERE json_extract(value,'$.candidate_id')=?) "
            "ORDER BY rowid LIMIT 101",
            (candidate_id,),
        ).fetchall()
        linked["execution_batch"] = {
            "ids": [r["id"] for r in batches[:100]],
            "has_more": len(batches) > 100,
        }
    return {
        "candidate": row,
        "pattern": pattern,
        "approvals": approvals(service, candidate_id),
        "linked_records": linked,
        "source_ids": pattern["data"]["pattern"]["source_ids"],
        "audit": service.store.audit(candidate_id),
        "execution": readiness(service, row),
        "entry": f"/evolve-candidate {candidate_id} full",
    }
