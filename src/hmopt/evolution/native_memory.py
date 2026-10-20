"""Native Team Memory adapters: explicit evidence import and reviewed L1 staging.

Execution evidence remains in Evolution. Personal experience uses the existing
journal format; reusable knowledge still belongs to the existing Hub curator,
schemas, evaluations and release process. No function publishes or promotes it.
Root paths are trusted operator configuration, never agent-controlled MCP inputs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml

from hmopt.sediment import journal as jm
from hmopt.sediment.validate import validate_candidate
from hmopt.skillhub.records import infer_kind, parse_frontmatter

from .learning import _proposal, _read_evidence
from .sources import (
    EvidenceRecord,
    _check_path,
    _is_link,
    _limit,
    _read_document,
    ingest_records,
)
from .store import ConflictError, canonical_json, digest

_MAX_FILES = 10000
_MAX_ENTRIES = 100000
_MAX_BYTES = 64 * 1024 * 1024


def _root(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_dir() or _is_link(path):
        raise ValueError("root must be an existing directory Path without a symlink")
    absolute = path.absolute()
    _check_path(Path(absolute.anchor), absolute)
    resolved = path.resolve()
    _check_path(Path(resolved.anchor), resolved)
    return resolved


def _namespace(contributor: str | None, project: str | None) -> None:
    for error in (
        jm.validate_namespace(contributor or "", field_name="contributor"),
        jm.validate_project(project or ""),
    ):
        if error:
            raise ValueError(error)


def _redact(text: str, hub_root: Path | None = None) -> None:
    hits = jm.redact_scan(text, hub_root=hub_root)
    if hits:
        names = ", ".join(sorted({name for _line, name, _snippet in hits}))
        raise ValueError(f"Native memory rejected secret pattern(s): {names}")


def evidence_reference(sha256: str) -> str:
    """Encode a known evidence digest without weakening the Hub's secret scan."""
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Expected an evidence SHA-256")
    encoded = base64.b32encode(bytes.fromhex(sha256)).decode("ascii").rstrip("=")
    return "evolution:evidence:sha256-b32:" + "-".join(
        encoded[index : index + 13] for index in range(0, len(encoded), 13)
    )


def resolve_native_evidence(service: Any, reference: str) -> Any:
    """Resolve a native staging source reference in the originating Evolution store."""
    prefix = "evolution:evidence:sha256-b32:"
    if not isinstance(reference, str) or not re.fullmatch(
        prefix + r"[A-Z2-7]{13}(?:-[A-Z2-7]{13}){3}", reference
    ):
        raise ValueError("Expected a native Evolution evidence reference")
    sha = base64.b32decode(reference[len(prefix) :].replace("-", "") + "====").hex()
    if evidence_reference(sha) != reference:
        raise ValueError("Noncanonical native evidence reference")
    return service.store.read_evidence(sha)


def _snapshot(root: Path, binding: dict) -> dict:
    selected = (
        root / binding["contributor"] / binding["project"] / "journal"
        if binding["collection"] == "journal"
        else root / "knowledge"
    )
    pending = [selected]
    files, errors = [], []
    consumed = entries = 0
    exhausted = False
    while pending and not exhausted:
        directory = pending.pop()
        if not directory.exists() and not directory.is_symlink():
            continue
        try:
            _check_path(root, directory)
            children = []
            with os.scandir(directory) as iterator:
                for child in iterator:
                    entries += 1
                    if entries > _MAX_ENTRIES:
                        exhausted = True
                        break
                    children.append(child)
            for child in sorted(children, key=lambda item: item.name):
                if len(errors) >= 1000:
                    exhausted = True
                    break
                path = directory / child.name
                uri = path.relative_to(root).as_posix()
                if _is_link(path):
                    errors.append({"source_uri": uri, "code": "linked_source"})
                    continue
                if child.name.startswith(".") or child.name in {"index", "README.md"}:
                    continue
                if child.is_dir(follow_symlinks=False):
                    pending.append(path)
                    continue
                if path.suffix != ".md":
                    continue
                if binding["collection"] == "journal" and not jm.is_journal_id(path.stem):
                    continue
                if len(files) >= _MAX_FILES or len(errors) >= 1000:
                    exhausted = True
                    break
                if path.stat().st_size > _MAX_BYTES - consumed:
                    exhausted = True
                    break
                try:
                    raw = _read_document(root, path, _MAX_BYTES - consumed)
                    consumed += len(raw)
                    files.append(
                        {"path": uri, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                    )
                except (OSError, ValueError):
                    errors.append({"source_uri": uri, "code": "invalid_source"})
        except (OSError, ValueError):
            errors.append(
                {"source_uri": directory.relative_to(root).as_posix(), "code": "invalid_directory"}
            )
    return {
        "schema_version": 1,
        "binding": binding,
        "files": sorted(files, key=lambda item: item["path"]),
        "errors": errors,
        "snapshot_bytes": consumed,
        "directory_entries": entries,
        "requires_partition": exhausted,
    }


def _native_record(root: Path, path: Path, raw: bytes, binding: dict) -> EvidenceRecord:
    content = raw.decode("utf-8-sig", errors="strict")
    _redact(content)  # Import never executes code from the imported Hub's tools directory.
    fields, body = parse_frontmatter(content)
    fields = jm._stringify_dates(fields)
    if binding["collection"] == "journal":
        errors = jm.validate_entry_input(
            type=fields.get("type", "fact"),
            title=fields.get("title", ""),
            body=body,
            outcome=fields.get("outcome", "unknown"),
            confidence=fields.get("confidence", ""),
            contributor=fields.get("contributor", ""),
            project=fields.get("project", ""),
            target_slug=fields.get("target_slug", ""),
        )
        if (
            errors
            or fields.get("id") != path.stem
            or fields.get("contributor") != binding["contributor"]
            or fields.get("project") != binding["project"]
        ):
            raise ValueError("Journal schema or contributor/project/filename mismatch")
    else:
        family = infer_kind(fields, path)
        _check_path(root, root / "schemas" / f"{family}.schema.json")
        record = dict(fields)
        if family == "memory_item":
            record["body"] = body
        if validate_candidate({"schema": family, "record": record}, root):
            raise ValueError("Hub knowledge record does not satisfy its native schema")
    return EvidenceRecord(
        repo_id=binding["repo_id"],
        source_kind="memory",
        source_uri=f"native-{binding['collection']}/" + path.relative_to(root).as_posix(),
        content=content,
        decision_reason="unknown",
    )


def import_native_memory(
    service: Any,
    root: Path,
    repo_id: str,
    *,
    collection: Literal["journal", "hub"],
    actor: str,
    contributor: str | None = None,
    project: str | None = None,
    max_files: int = 200,
    max_bytes: int = 8 * 1024 * 1024,
    cursor: str | None = None,
) -> dict:
    """Import an immutable, bounded page from one explicit native memory scope.

    Journal import requires both contributor and project; no all-member harvest.
    Hub import reads only knowledge, including superseded evidence as historical
    context. Native status/outcome are metadata, never automatic pattern approval.
    """
    actor = service._actor(actor)
    _limit(max_files, "max_files", 1000)
    _limit(max_bytes, "max_bytes", 16 * 1024 * 1024)
    root = _root(root)
    if collection not in {"journal", "hub"}:
        raise ValueError("collection must be journal or hub")
    if collection == "journal":
        _namespace(contributor, project)
    elif contributor is not None or project is not None:
        raise ValueError("Hub collection uses knowledge scope, not contributor/project")
    if collection == "hub" and not (root / "schemas").is_dir():
        raise ValueError("Hub collection requires native schemas")
    # Validate repo identity before snapshot writes, even for an empty collection.
    EvidenceRecord(repo_id=repo_id, source_kind="memory", source_uri="probe.md", content="probe")
    binding = {
        "root": str(root),
        "collection": collection,
        "repo_id": repo_id,
        "contributor": contributor,
        "project": project,
    }
    with service.store.transaction() as db:
        if cursor is None:
            manifest = _snapshot(root, binding)
            manifest_id = "native_manifest_" + digest(manifest)
            service.store.put(db, "native_manifest", manifest_id, manifest)
            service.store.evidence(db, manifest)
            offset = 0
        else:
            if not isinstance(cursor, str) or not re.fullmatch(
                r"native_cursor_[0-9a-f]{64}", cursor
            ):
                raise ValueError("cursor must be a persisted native memory cursor")
            token = service.store.get(db, "native_cursor", cursor)["data"]
            if cursor != "native_cursor_" + digest(token):
                raise ConflictError("Native cursor integrity check failed")
            manifest_id, offset = token["manifest_id"], token["offset"]
            manifest = service.store.get(db, "native_manifest", manifest_id)["data"]
            if manifest_id != "native_manifest_" + digest(manifest):
                raise ConflictError("Native manifest integrity check failed")
            evidence = db.execute(
                "SELECT content FROM evidence WHERE sha256=?", (digest(manifest),)
            ).fetchone()
            if evidence is None or json.loads(evidence["content"]) != manifest:
                raise ConflictError("Native manifest evidence is missing or changed")
            if manifest["binding"] != binding:
                raise ValueError("Native cursor belongs to a different memory scope")
            if type(offset) is not int or not 0 <= offset <= len(manifest["files"]):
                raise ValueError("Invalid native cursor offset")
    errors = list(manifest["errors"]) if offset == 0 else []
    position, consumed, considered = offset, 0, 0
    records = []
    while position < len(manifest["files"]) and considered < max_files:
        item = manifest["files"][position]
        if item["bytes"] > max_bytes - consumed:
            if considered == 0:
                errors.append(
                    {
                        "source_uri": item["path"],
                        "code": "page_bytes",
                        "message": "Increase max_bytes; no progress was made",
                    }
                )
            break
        considered += 1
        try:
            path = root / item["path"]
            raw = _read_document(root, path, max_bytes - consumed)
            consumed += len(raw)
            if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Source changed after snapshot creation")
            records.append(_native_record(root, path, raw, binding))
        except (
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            RecursionError,
            yaml.YAMLError,
        ) as exc:
            errors.append(
                {
                    "source_uri": item["path"],
                    "code": "invalid_native_source",
                    "message": type(exc).__name__ + ": native source rejected",
                }
            )
        position += 1
    result = ingest_records(service, records, actor=actor)
    errors.extend(result["errors"])
    more = position < len(manifest["files"])
    next_cursor = None
    if more and position > offset and not manifest["requires_partition"]:
        token = {"manifest_id": manifest_id, "offset": position}
        next_cursor = "native_cursor_" + digest(token)
        with service.store.transaction() as db:
            service.store.put(db, "native_cursor", next_cursor, token)
    result.update(
        collection=collection,
        manifest_id=manifest_id,
        manifest_files=len(manifest["files"]),
        offset=offset,
        next_offset=position,
        files_considered=considered,
        bytes_read=consumed,
        errors=errors,
        next_cursor=next_cursor,
        has_more=more,
        requires_partition=manifest["requires_partition"],
        requires_attention=bool(errors),
        partial=bool(errors) or more or manifest["requires_partition"],
        snapshot_bytes=manifest["snapshot_bytes"],
    )
    return result


def _existing_file(root: Path, relative: str, expected: str) -> bool:
    path = root / relative
    if not path.exists() and not path.is_symlink():
        return False
    raw = _read_document(root, path, 256 * 1024)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ConflictError("Native export file changed; refusing to overwrite it")
    return True


def _write_export_file(root: Path, relative: str, text: str, expected: str) -> None:
    from .workflow import _exclusive_artifact

    if _existing_file(root, relative, expected):
        return
    path = root / relative
    current = root
    for component in path.parent.relative_to(root).parts:
        current /= component
        if current.exists() or current.is_symlink():
            _check_path(root, current)
        jm._ensure_private_dir(current)
    _check_path(root, path.parent)
    # Publish the complete fsynced temporary inode with no replacement. A crash
    # during the write cannot leave a partial final journal or staging file.
    _exclusive_artifact(path, text)
    path.chmod(0o600)


def _verify_intent(db: Any, prepared: dict) -> None:
    original = {
        key: value
        for key, value in prepared.items()
        if key not in {"intent_sha256", "result", "status"}
    }
    original["status"] = "prepared"
    if _read_evidence(db, prepared.get("intent_sha256")) != original:
        raise ConflictError("Native export intent failed its evidence integrity check")


def export_native_memory(
    service: Any,
    skill_id: str,
    *,
    memory_root: Path,
    hub_root: Path,
    contributor: str,
    project: str,
    actor: str,
    request_id: str,
    title: str,
    body: str,
    applies_when: list[str],
    invalidated_by: list[str],
    target_slug: str = "",
) -> dict:
    """Write one reviewed fact to personal journal and native L1 Hub staging.

    Explicit shareable text is required: no automatic copying of private source,
    plans or device metadata. A durable intent binds exact file bytes before I/O;
    retries recover an interrupted write without duplicating journal entries.
    Completed retries verify destinations and never recreate a forgotten entry.
    """
    actor = service._actor(actor)
    _namespace(contributor, project)
    memory_root, hub_root = _root(memory_root), _root(hub_root)
    if memory_root.is_relative_to(hub_root) or hub_root.is_relative_to(memory_root):
        raise ValueError("Personal memory root and Hub root must not overlap")
    if not (hub_root / "schemas").is_dir():
        raise ValueError("Native export requires Hub schemas")
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 200:
        raise ValueError("request_id must be nonempty text of at most 200 characters")
    for name, values in (("applies_when", applies_when), ("invalidated_by", invalidated_by)):
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 10
            or any(
                not isinstance(value, str) or not value.strip() or len(value) > 300
                for value in values
            )
        ):
            raise ValueError(f"{name} requires 1..10 nonempty strings of at most 300 characters")
    errors = jm.validate_entry_input(
        type="fact",
        title=title,
        body=body,
        outcome="validated",
        contributor=contributor,
        project=project,
        target_slug=target_slug,
    )
    if errors:
        raise ValueError("; ".join(errors))
    request = {
        "operation": "export_native_memory",
        "skill_id": skill_id,
        "memory_root": str(memory_root),
        "hub_root": str(hub_root),
        "contributor": contributor,
        "project": project,
        "actor": actor,
        "title": title,
        "body": body,
        "applies_when": applies_when,
        "invalidated_by": invalidated_by,
        "target_slug": target_slug,
    }
    export_id = (
        "native-export-" + digest({"store": str(service.store.root), "request_id": request_id})[:32]
    )
    with service.store.transaction() as db:
        replay = service.store.replay(db, request_id, request)
        if replay is not None:
            for root, prefix in ((memory_root, "journal"), (hub_root, "staging")):
                if not _existing_file(
                    root, replay[prefix + "_relative"], replay[prefix + "_sha256"]
                ):
                    raise ConflictError("Completed native export destination is missing")
            return replay
        skill = service.store.get(db, "skill", skill_id)
        proposal = _proposal(service, db, skill)
        if actor != skill["data"]["curator"]:
            raise ValueError("Native export must be requested by the independent evidence curator")
        existing = db.execute(
            "SELECT id FROM records WHERE kind='native_export' AND id=?", (export_id,)
        ).fetchone()
        if existing:
            intent = service.store.get(db, "native_export", export_id)
            prepared = intent["data"]
            _verify_intent(db, prepared)
            if prepared["request_sha256"] != digest(request):
                raise ConflictError(
                    "Native export request_id already used with different arguments"
                )
            if prepared["proposal_sha256"] != digest(proposal):
                raise ConflictError("Evidence changed since native export was prepared")
        else:
            entry = jm.JournalEntry(
                id="J-" + jm.new_ulid(),
                type="fact",
                title=title.strip(),
                body=body.strip(),
                contributor=contributor,
                project=project,
                target_slug=target_slug,
                tags=["evolution", proposal["validation_kind"]],
                outcome="validated",
                evidence=[evidence_reference(sha) for sha in proposal["evidence_sha256"]],
                applies_when=applies_when,
                invalidated_by=invalidated_by,
                ts=jm._now(),
            )
            text = jm.render_entry(entry)
            _redact(text, hub_root)
            candidates, gated, errors = jm.journal_to_candidates([entry], contributor=contributor)
            if errors or gated or len(candidates) != 1:
                raise ValueError("Native journal mapping did not produce one eligible candidate")
            candidate = candidates[0]
            candidate["record"]["id"] = "F" + str(int(digest(export_id)[:24], 16))
            errors = validate_candidate(candidate, hub_root)
            if errors:
                raise ValueError("; ".join(errors))
            staged = canonical_json(candidate) + "\n"
            _redact(staged, hub_root)
            prepared = {
                "status": "prepared",
                "request_sha256": digest(request),
                "proposal_sha256": digest(proposal),
                "skill_id": skill_id,
                "journal_id": entry.id,
                "native_schema": "memory_item",
                "journal_relative": f"{contributor}/{project}/journal/{entry.ts[:7]}/{entry.id}.md",
                "staging_relative": f"staging/{contributor}/{export_id}.jsonl",
                "journal_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "staging_sha256": hashlib.sha256(staged.encode()).hexdigest(),
                "journal_text": text,
                "staging_text": staged,
            }
            intent_sha256 = service.store.evidence(db, prepared)
            prepared["intent_sha256"] = intent_sha256
            intent = service.store.put(db, "native_export", export_id, prepared)
            service.store.evidence(db, proposal)
    # Keep a prepared intent if either filesystem is unavailable. A retry resumes
    # exact bytes; existing different content is never overwritten or deleted.
    with service.store.transaction() as db:
        latest = service.store.get(db, "native_export", export_id)
        prepared = latest["data"]
        _verify_intent(db, prepared)
        replay = service.store.replay(db, request_id, request)
        if replay is not None:
            return replay
        current_skill = service.store.get(db, "skill", skill_id)
        if digest(_proposal(service, db, current_skill)) != prepared["proposal_sha256"]:
            raise ConflictError("Evidence changed before native export materialization")
        for root, prefix in ((memory_root, "journal"), (hub_root, "staging")):
            _write_export_file(
                root,
                prepared[prefix + "_relative"],
                prepared[prefix + "_text"],
                prepared[prefix + "_sha256"],
            )
        result = {
            key: value
            for key, value in prepared.items()
            if key not in {"journal_text", "staging_text", "status"}
        }
        result.update(
            export_id=export_id,
            status="staged",
            maturity="L1",
            publication_status="not_published",
            merged=False,
            journal_path=str(memory_root / result["journal_relative"]),
            staging_path=str(hub_root / result["staging_relative"]),
            curator_review_required=True,
        )
        service.store.put(
            db,
            "native_export",
            export_id,
            {**prepared, "status": "staged", "result": result},
            latest["version"],
        )
        service.store.event(
            db,
            export_id,
            "stage_native_memory",
            actor,
            {
                "skill_id": skill_id,
                "journal_id": result["journal_id"],
                "staging_sha256": result["staging_sha256"],
            },
        )
        service.store.remember(db, request_id, request, result)
        return result


__all__ = ["export_native_memory", "import_native_memory", "resolve_native_evidence"]
