"""Restartable immutable-tree traversal and pattern partitions; no whole-tree truncation."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

from .mining import _Git, _path, active_pattern_records
from .store import ConflictError, digest


def _attach_run_scan(service, db, scan, checkpoint):
    if checkpoint is None:
        return
    run_id, project_id, expected = checkpoint
    run = service.store.get(db, "workspace_run", run_id)
    if run["data"]["status"] == "cancelled" or run["data"]["projects"][project_id] != expected:
        raise ConflictError("Workspace project changed before scan attachment")
    run["data"]["projects"][project_id]["scan_id"] = scan["id"]
    service.store.put(db, "workspace_run", run_id, run["data"], run["version"])


def start_scan(
    service,
    repo,
    *,
    revision,
    owners,
    hotspots=None,
    request_id,
    actor,
    workspace_manifest_sha256=None,
    _run_checkpoint=None,
):
    actor = service._actor(actor)
    request = {
        "repo": str(Path(repo).resolve()),
        "revision": revision,
        "owners": owners,
        "hotspots": [h.model_dump(mode="json") for h in (hotspots or [])],
        "actor": actor,
        "workspace_manifest_sha256": workspace_manifest_sha256,
    }
    with service.store.transaction() as db:
        replay = service.store.replay(db, request_id, request)
        if replay is not None:
            _attach_run_scan(service, db, replay, _run_checkpoint)
            return replay
    target = service._revision(repo, revision)
    if service.source_workspace and workspace_manifest_sha256 is None:
        from .workspace import freeze_workspace

        manifest = freeze_workspace(service.source_workspace, git_bin=service.git_bin)
        matching = [p for p in manifest["projects"] if p["repo_id"] == service.repo_identity(repo)]
        if len(matching) != 1:
            raise ValueError("Scan repository is outside the selected workspace")
        matching[0]["revision"] = target
        with service.store.transaction() as db:
            workspace_manifest_sha256 = service.store.evidence(db, manifest)
    if any(h.revision != target for h in (hotspots or [])):
        raise ValueError("Hotspots must bind the frozen target revision")
    with service.store.transaction() as db:
        replay = service.store.replay(db, request_id, request)
        if replay is not None:
            _attach_run_scan(service, db, replay, _run_checkpoint)
            return replay
        rows = db.execute(
            "SELECT id,payload FROM records WHERE kind='pattern' ORDER BY id"
        ).fetchall()
        patterns = active_pattern_records(
            [{"id": row["id"], "data": json.loads(row["payload"])} for row in rows]
        )
        overlays = {
            r["id"]: json.loads(r["payload"])
            for r in db.execute("SELECT id,payload FROM records WHERE kind='overlay'")
        }
        snapshot = {
            "patterns": patterns,
            "overlays": overlays,
            "revision": target,
            "repo_path": request["repo"],
            "repo_id": service.repo_identity(repo),
            "owners": owners,
            "hotspots": request["hotspots"],
            "workspace_manifest_sha256": workspace_manifest_sha256,
        }
        sha = service.store.evidence(db, snapshot)
        key = "scan_" + digest([request, request_id])
        result = service.store.put(
            db,
            "scan",
            key,
            {
                "snapshot_sha256": sha,
                "status": "running" if patterns else "no_active_patterns",
                "stack": [{"tree": target, "prefix": "", "offset": 0}],
                "paths": [],
                "pattern_offset": 0,
                "pages": 0,
                "entries": 0,
                "files": 0,
                "skipped": {},
                "candidate_count": 0,
                "coverage_complete": False,
                "lease_until": 0,
            },
        )
        # A crash must not leave an independently schedulable scan detached from its run.
        _attach_run_scan(service, db, result, _run_checkpoint)
        service.store.remember(db, request_id, request, result)
        service.store.event(db, key, "scan_started", actor, {"snapshot_sha256": sha})
        return result


def control_scan(service, scan_id, *, action, actor, expected_version, _db=None):
    """Retry the frozen checkpoint or cancel it, fencing any outstanding page writer."""
    actor = service._actor(actor)
    with nullcontext(_db) if _db is not None else service.store.transaction() as db:
        row = service.store.get(db, "scan", scan_id)
        if row["version"] != expected_version:
            raise ConflictError("Scan version changed; read its current checkpoint")
        data = row["data"]
        if action == "retry" and data["status"] == "attention":
            data.update(status="running", lease_until=0)
            data.pop("error", None)
        elif action == "cancel" and data["status"] in {"running", "attention"}:
            data.update(status="cancelled", lease_until=0)
        else:
            raise ValueError("Retry requires an attention scan; cancel requires running/attention")
        service.store.event(db, scan_id, "scan_" + action, actor, {})
        return service.store.put(db, "scan", scan_id, data, expected_version)


def scan_next(service, scan_id, *, expected_version, page_size=8):
    if type(page_size) is not int or not 1 <= page_size <= 32:
        raise ValueError("Scan page_size must be 1..32")
    with service.store.transaction() as db:
        row = service.store.get(db, "scan", scan_id)
        if row["version"] != expected_version:
            raise ConflictError("Scan version changed; read its current checkpoint")
        data = row["data"]
        if data["status"] != "running":
            return row
        if data["lease_until"] > time.time():
            raise ConflictError("Scan page is already being processed")
        data["lease_until"] = time.time() + 300
        row = service.store.put(db, "scan", scan_id, data, row["version"])
    try:
        snapshot = service.store.read_evidence(data["snapshot_sha256"])
        if service.repo_identity(snapshot["repo_path"]) != snapshot["repo_id"]:
            raise ConflictError("Scan repository identity changed")
        git = _Git(Path(snapshot["repo_path"]), service.git_bin)
        # Each tree is immutable. Offsets address entries of that exact tree, not HEAD.
        visited = 0
        while not data["paths"] and data["stack"] and visited < page_size:
            frame = data["stack"][-1]
            raw, cut = git.read(["ls-tree", "-z", "-l", frame["tree"], "--"], 16 * 1024 * 1024)
            if cut:
                raise ValueError(
                    "A single directory exceeds 16 MiB of tree entries; explicit investigation required"
                )
            rows = raw.split(b"\x00")[:-1]
            while frame["offset"] < len(rows) and visited < page_size:
                raw_row = rows[frame["offset"]]
                frame["offset"] += 1
                visited += 1
                data["entries"] += 1
                info, name = raw_row.split(b"\t", 1)
                mode, kind, oid, size = info.split()
                path = _path(frame["prefix"] + name.decode("utf-8"))
                if kind == b"tree":
                    data["stack"].append(
                        {"tree": oid.decode("ascii"), "prefix": path + "/", "offset": 0}
                    )
                    break
                if kind != b"blob" or mode not in {b"100644", b"100755"}:
                    reason = "non_regular"
                elif int(size) > 1_000_000:
                    reason = "over_1mb"
                else:
                    data["paths"].append(path)
                    data["files"] += 1
                    continue
                data["skipped"][reason] = data["skipped"].get(reason, 0) + 1
            if frame["offset"] >= len(rows) and data["stack"][-1] is frame:
                data["stack"].pop()
        result = {"candidates": [], "coverage": {}}
        if data["paths"]:
            from .mining import Hotspot

            offset = data["pattern_offset"]
            result = service.scan(
                snapshot["repo_path"],
                revision=snapshot["revision"],
                owners=snapshot["owners"],
                hotspots=[Hotspot.model_validate(h) for h in snapshot["hotspots"]],
                top_k=1000,
                max_files=32,
                _paths=data["paths"],
                _patterns=snapshot["patterns"][offset : offset + 16],
                _overlays=snapshot["overlays"],
                _workspace_manifest_sha256=snapshot.get("workspace_manifest_sha256"),
                _scan_checkpoint=(scan_id, row["version"]),
            )
            if result["coverage"].get("traversal_incomplete") or result["coverage"].get(
                "results_capped"
            ):
                raise ValueError(
                    "Partition exceeded a scan budget; checkpoint retained for inspection"
                )
            for name in ("skipped_binary_or_encoding", "skipped_large"):
                # Count skip observations in every pattern partition. A binary
                # may become eligible only in a later partition; never claim
                # complete coverage merely because the first partition ignored it.
                if result["coverage"].get(name):
                    data["skipped"][name] = data["skipped"].get(name, 0) + result["coverage"][name]
            data["pattern_offset"] += 16
            if data["pattern_offset"] >= len(snapshot["patterns"]):
                data.update(paths=[], pattern_offset=0)
        data["pages"] += 1
        data["candidate_count"] += len(result["candidates"])
        if not data["stack"] and not data["paths"]:
            data["status"] = "complete"
            data["coverage_complete"] = not any(data["skipped"].values())
        data["lease_until"] = 0
        with service.store.transaction() as db:
            page = {
                "scan_id": scan_id,
                "page": data["pages"],
                "candidate_ids": [r["id"] for r in result["candidates"]],
                "coverage": result["coverage"],
            }
            service.store.put(db, "scan_page", f"{scan_id}:{data['pages']}", page)
            return service.store.put(db, "scan", scan_id, data, row["version"])
    except (ValueError, OSError, TimeoutError) as error:
        with service.store.transaction() as db:
            current = service.store.get(db, "scan", scan_id)
            if current["version"] != row["version"]:
                raise ConflictError("Scan checkpoint was superseded") from error
            # Preserve pre-page traversal coordinates, including on partial filesystem failures.
            previous = current["data"]
            previous.update(status="attention", lease_until=0, error=str(error)[:1000])
            service.store.put(db, "scan", scan_id, previous, current["version"])
        raise
