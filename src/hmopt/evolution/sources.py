"""Bounded local evidence adapters and explicitly heuristic draft distillation.

Source documents are evidence, never instructions to execute. Workspace import
only reads an explicit list of operational-knowledge locations; it does not
scan configuration files, run commands, or synthesize Git history. A changed
document creates a new immutable source ID while preserving its location ID.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .mining import Matcher, Pattern
from .store import ConflictError, digest

_MAX_CONTENT_BYTES = 256 * 1024
_MAX_BATCH_BYTES = 16 * 1024 * 1024
_MAX_BATCH_RECORDS = 1000
_MAX_MANIFEST_FILES = 10000
_MAX_MANIFEST_ENTRIES = 100000
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_DISTILLER = "literal-source-v1"
_SOURCE_DIRS = (
    (".opencode/memory", "memory"),
    (".opencode/plans", "decision"),
    (".opencode/reviews", "review"),
    (".opencode/experiments", "experiment"),
    (".opencode/bench/results", "experiment"),
)
_SUFFIXES = {".md", ".txt", ".json"}
_CODE_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".rs", ".py", ".go"}
_TECHNICAL = re.compile(
    r"optimi[sz]|bottleneck|regression|deadlock|race|redundan|repeated|"
    r"allocation|latency|correctness|性能|优化|瓶颈|死锁|竞争|重复|分配|回归|正确性",
    re.IGNORECASE,
)


def _text(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("Expected nonblank text without control characters")
    return value


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or value == "."
        or ".." in path.parts
        or len(path.parts) > 64
        or any(char in value for char in "\\:*?[]")
    ):
        raise ValueError("source_uri must be a normalized relative POSIX path")
    return _text(value)


class EvidenceRecord(BaseModel):
    """A document snapshot with caller-declared, separately auditable provenance."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", validate_default=True)

    repo_id: Annotated[str, Field(min_length=1, max_length=160)]
    source_kind: Literal["memory", "review", "experiment", "decision"]
    source_uri: Annotated[str, Field(min_length=1, max_length=2048)]
    content: Annotated[str, Field(min_length=1, max_length=_MAX_CONTENT_BYTES)]
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    decision_reason: (
        Literal[
            "technical_rejection", "resource_limit", "measurement_failure", "accepted", "unknown"
        ]
        | None
    ) = None
    related_revision: str | None = None

    _repo_id = field_validator("repo_id")(_text)
    _source_uri = field_validator("source_uri")(_relative)

    @field_validator("related_revision")
    @classmethod
    def valid_revision(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value):
            raise ValueError("related_revision must be a full lowercase Git object ID")
        return value

    @field_validator("content")
    @classmethod
    def bounded_content(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("content must be nonblank text without NUL")
        if len(value.encode("utf-8")) > _MAX_CONTENT_BYTES:
            raise ValueError("content exceeds the 256 KiB UTF-8 limit")
        return value

    @model_validator(mode="after")
    def verify_content_digest(self) -> EvidenceRecord:
        actual = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_sha256 is not None and self.content_sha256 != actual:
            raise ValueError("content_sha256 does not match the supplied content")
        object.__setattr__(self, "content_sha256", actual)
        return self

    @property
    def location_id(self) -> str:
        return "source_location_" + digest(
            {
                "repo_id": self.repo_id,
                "source_kind": self.source_kind,
                "source_uri": self.source_uri,
            }
        )

    @property
    def source_id(self) -> str:
        return "source_" + digest(self.model_dump(mode="json"))


def _limit(value: int, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def _actor(service: Any, actor: str) -> str:
    result = service._actor(actor)
    return _text(result)


def _error(index: int | None, source_uri: str | None, code: str, message: str) -> dict:
    return {"index": index, "source_uri": source_uri, "code": code, "message": message}


def ingest_records(service: Any, records: list[EvidenceRecord | dict], *, actor: str) -> dict:
    """Import independent rows, reporting every rejection without echoing document text."""
    actor = _actor(service, actor)
    if not isinstance(records, list) or len(records) > _MAX_BATCH_RECORDS:
        raise ValueError("records must be a list of at most 1000 items")
    result: dict[str, Any] = {"imported": [], "existing": [], "errors": [], "partial": False}
    consumed = 0
    for index, incoming in enumerate(records):
        uri = incoming.source_uri if isinstance(incoming, EvidenceRecord) else None
        try:
            raw = (
                incoming.model_dump(mode="python")
                if isinstance(incoming, EvidenceRecord)
                else incoming
            )
            record = EvidenceRecord.model_validate(raw)
            uri = record.source_uri
            consumed += len(record.content.encode("utf-8"))
            if consumed > _MAX_BATCH_BYTES:
                raise ValueError("Batch content exceeds the 16 MiB budget")
            with service.store.transaction() as db:
                row = db.execute(
                    "SELECT id FROM records WHERE kind='source' AND id=?", (record.source_id,)
                ).fetchone()
                if row:
                    stored = service.store.get(db, "source", record.source_id)
                    if stored["data"].get("record") != record.model_dump(mode="json"):
                        raise ConflictError("Stored source payload differs from its content ID")
                    result["existing"].append(record.source_id)
                    continue
                # Location lineage is metadata. Each content version stays immutable.
                versions = db.execute(
                    "SELECT version FROM records WHERE kind='source_location' AND id=?",
                    (record.location_id,),
                ).fetchone()
                version = versions["version"] + 1 if versions else 1
                evidence_sha = service.store.evidence(db, record.model_dump(mode="json"))
                data = {
                    "record": record.model_dump(mode="json"),
                    "location_id": record.location_id,
                    "source_version": version,
                    "evidence_sha256": evidence_sha,
                    "imported_by": actor,
                }
                service.store.put(db, "source", record.source_id, data)
                service.store.put(
                    db,
                    "source_location",
                    record.location_id,
                    {"latest_source_id": record.source_id},
                    versions["version"] if versions else None,
                )
                service.store.event(
                    db,
                    record.source_id,
                    "ingest_source",
                    actor,
                    {
                        "location_id": record.location_id,
                        "source_version": version,
                        "evidence_sha256": evidence_sha,
                    },
                )
                result["imported"].append(record.source_id)
        except (ValidationError, ValueError) as exc:
            # Pydantic's default string includes input values; never echo source content.
            message = (
                "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                    for e in exc.errors(include_input=False)
                )
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            result["errors"].append(_error(index, uri, "invalid_record", message))
    result["partial"] = bool(result["errors"])
    result["content_bytes"] = min(consumed, _MAX_BATCH_BYTES)
    return result


def _is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _check_path(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if _is_link(current):
            raise ValueError("Symlinks and reparse points are not imported")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Source resolves outside the selected workspace")


def _read_document(root: Path, path: Path, budget: int) -> bytes:
    _check_path(root, path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Source must be a regular file")
    cap = min(_MAX_CONTENT_BYTES, budget)
    if before.st_size > cap:
        raise ValueError("Source exceeds the per-file or remaining batch byte budget")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(os.open(path, flags), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("Opened source must be a regular file")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("Source changed while being opened")
        content = stream.read(cap + 1)
        after = os.fstat(stream.fileno())
        if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source changed while being read")
    _check_path(root, path)
    if len(content) > cap:
        raise ValueError("Source exceeds the per-file or remaining batch byte budget")
    return content


def _metadata(content: str, suffix: str) -> dict:
    """Only explicit metadata is classified; the filename 'bad' proves no cause."""
    if suffix == ".json":
        value = json.loads(content)
        value = value if isinstance(value, dict) else {}
        return {key: value[key] for key in ("decision_reason", "related_revision") if key in value}
    match = re.search(r"(?mi)^\s*decision_reason\s*:\s*([a-z_]+)\s*$", content)
    revision = re.search(r"(?mi)^\s*related_revision\s*:\s*(\S+)\s*$", content)
    result = {"decision_reason": match.group(1)} if match else {}
    if revision:
        result["related_revision"] = revision.group(1)
    return result


def _workspace_manifest(root: Path, repo_id: str) -> dict:
    """Freeze a bounded file list and raw hashes before publishing any page cursor."""
    errors: list[dict] = []
    skipped: list[dict] = []
    skipped_count = 0
    files: list[dict] = []
    pending = [(root / relative, kind, False) for relative, kind in reversed(_SOURCE_DIRS)]
    pending.append((root / ".opencode/state", "decision", True))
    consumed = 0
    entries = 0
    exhausted = False
    while pending:
        directory, kind, state_only = pending.pop()
        if not directory.exists() and not directory.is_symlink():
            continue
        relative_dir = directory.relative_to(root).as_posix()
        try:
            _check_path(root, directory)
            children = []
            with os.scandir(directory) as iterator:
                for child in iterator:
                    entries += 1
                    if entries > _MAX_MANIFEST_ENTRIES:
                        exhausted = True
                        break
                    children.append(child)
            if exhausted:
                errors.append(
                    _error(
                        None,
                        relative_dir,
                        "entry_limit",
                        "Snapshot directory-entry budget exhausted; partition the workspace",
                    )
                )
                break
            for child in sorted(children, key=lambda item: item.name):
                path = directory / child.name
                uri = path.relative_to(root).as_posix()
                if len(errors) >= 1000:
                    exhausted = True
                    errors.append(
                        _error(
                            None,
                            relative_dir,
                            "manifest_limit",
                            "Snapshot error limit reached; partition or repair the workspace",
                        )
                    )
                    break
                if child.name.startswith("."):
                    continue
                if _is_link(path):
                    errors.append(
                        _error(
                            None,
                            uri,
                            "invalid_source",
                            "Symlinks and reparse points are not imported",
                        )
                    )
                    continue
                if child.is_dir(follow_symlinks=False):
                    if not state_only:
                        pending.append((path, kind, False))
                    continue
                if path.suffix.lower() not in _SUFFIXES:
                    continue
                if state_only and not (
                    child.name == "bad_plans.md" or child.name.endswith("-bad_plans.md")
                ):
                    continue
                if "template" in child.name.lower() or child.name.lower().startswith("readme"):
                    skipped_count += 1
                    if len(skipped) < 1000:
                        skipped.append({"source_uri": uri, "reason": "template_or_readme"})
                    continue
                if len(files) >= _MAX_MANIFEST_FILES or len(errors) >= 1000:
                    exhausted = True
                    errors.append(
                        _error(
                            None,
                            uri,
                            "manifest_limit",
                            "Snapshot file/error limit reached; partition the workspace",
                        )
                    )
                    break
                try:
                    if path.stat().st_size > _MAX_MANIFEST_BYTES - consumed:
                        exhausted = True
                        errors.append(
                            _error(
                                None,
                                uri,
                                "manifest_bytes",
                                "Snapshot exceeds 64 MiB; partition the workspace",
                            )
                        )
                        break
                    content_bytes = _read_document(root, path, _MAX_MANIFEST_BYTES - consumed)
                    consumed += len(content_bytes)
                    files.append(
                        {
                            "source_uri": uri,
                            "source_kind": kind,
                            "raw_sha256": hashlib.sha256(content_bytes).hexdigest(),
                            "bytes": len(content_bytes),
                        }
                    )
                except (OSError, ValueError, RecursionError) as exc:
                    code = "invalid_source"
                    message = (
                        str(exc)
                        if not isinstance(exc, ValidationError)
                        else "Invalid source content or metadata"
                    )
                    errors.append(_error(None, uri, code, message))
            if exhausted:
                break
        except (OSError, ValueError) as exc:
            errors.append(_error(None, relative_dir, "invalid_directory", str(exc)))
    return {
        "schema_version": 1,
        "workspace": str(root),
        "repo_id": repo_id,
        "files": sorted(files, key=lambda value: value["source_uri"]),
        "errors": errors,
        "skipped": skipped,
        "skipped_count": skipped_count,
        "requires_partition": exhausted,
        "snapshot_bytes": consumed,
        "directory_entries": entries,
    }


def import_workspace(
    service: Any,
    workspace: Path,
    repo_id: str,
    *,
    actor: str,
    max_files: int = 200,
    max_bytes: int = 8 * 1024 * 1024,
    cursor: str | None = None,
) -> dict:
    """Import a page from a persisted file/hash snapshot, or create a new snapshot.

    ``next_cursor`` resumes the same immutable manifest. Files added later belong
    to a new import. Every selected file must still match its original raw hash;
    edits or deletions fail explicitly instead of mixing snapshots. Manifest
    creation reads at most 64 MiB/10,000 files/100,000 directory entries; each
    import page reads at most ``max_bytes`` and processes ``max_files`` entries.
    """
    actor = _actor(service, actor)
    _limit(max_files, "max_files", 1000)
    _limit(max_bytes, "max_bytes", _MAX_BATCH_BYTES)
    if not isinstance(workspace, Path) or not workspace.is_dir() or _is_link(workspace):
        raise ValueError("workspace must be an existing directory Path without a symlink")
    _text(repo_id)
    root = workspace.resolve()
    if cursor is None:
        manifest = _workspace_manifest(root, repo_id)
        manifest_id = "source_manifest_" + digest(manifest)
        offset = 0
        with service.store.transaction() as db:
            sha = service.store.evidence(db, manifest)
            service.store.put(
                db, "source_manifest", manifest_id, {"manifest": manifest, "evidence_sha256": sha}
            )
    else:
        if not isinstance(cursor, str) or not re.fullmatch(r"source_cursor_[0-9a-f]{64}", cursor):
            raise ValueError("cursor must be a persisted source page cursor")
        with service.store.transaction() as db:
            token = service.store.get(db, "source_cursor", cursor)["data"]
            if cursor != "source_cursor_" + digest(token):
                raise ConflictError("Source cursor integrity check failed")
            manifest_id, offset = token["manifest_id"], token["offset"]
            stored = service.store.get(db, "source_manifest", manifest_id)["data"]
            manifest = stored["manifest"]
            if (
                manifest_id != "source_manifest_" + digest(manifest)
                or digest(manifest) != stored["evidence_sha256"]
            ):
                raise ConflictError("Source manifest integrity check failed")
            evidence = db.execute(
                "SELECT content FROM evidence WHERE sha256=?", (stored["evidence_sha256"],)
            ).fetchone()
            if evidence is None or json.loads(evidence["content"]) != manifest:
                raise ConflictError("Source manifest evidence is missing or changed")
        if manifest["workspace"] != str(root) or manifest["repo_id"] != repo_id:
            raise ValueError("Source cursor belongs to a different workspace or repo_id")
        if type(offset) is not int or not 0 <= offset <= len(manifest["files"]):
            raise ValueError("Invalid source cursor offset")
    errors = list(manifest["errors"]) if offset == 0 else []
    records: list[EvidenceRecord] = []
    consumed = 0
    considered = 0
    position = offset
    requires_attention = False
    while position < len(manifest["files"]) and considered < max_files:
        item = manifest["files"][position]
        if item["bytes"] > max_bytes - consumed:
            if considered == 0:
                errors.append(
                    _error(
                        None,
                        item["source_uri"],
                        "page_bytes",
                        "A file cannot fit the page byte budget; increase max_bytes or partition the source",
                    )
                )
                requires_attention = True
            break
        considered += 1
        path = root / item["source_uri"]
        try:
            _relative(item["source_uri"])
            content_bytes = _read_document(root, path, max_bytes - consumed)
            consumed += len(content_bytes)
            if (
                len(content_bytes) != item["bytes"]
                or hashlib.sha256(content_bytes).hexdigest() != item["raw_sha256"]
            ):
                raise ValueError("Source changed after snapshot creation; start a new snapshot")
            content = content_bytes.decode("utf-8-sig", errors="strict")
            records.append(
                EvidenceRecord(
                    repo_id=repo_id,
                    source_kind=item["source_kind"],
                    source_uri=item["source_uri"],
                    content=content,
                    **_metadata(content, path.suffix.lower()),
                )
            )
        except (OSError, ValueError, RecursionError) as exc:
            message = (
                str(exc)
                if not isinstance(exc, ValidationError)
                else "Invalid source content or metadata"
            )
            errors.append(_error(None, item["source_uri"], "invalid_source", message))
            requires_attention = True
        position += 1
    result = ingest_records(service, records, actor=actor)
    errors.extend(result["errors"])
    has_more = position < len(manifest["files"])
    next_cursor = None
    if has_more and not manifest["requires_partition"] and position > offset:
        token = {"manifest_id": manifest_id, "offset": position}
        next_cursor = "source_cursor_" + digest(token)
        with service.store.transaction() as db:
            service.store.put(db, "source_cursor", next_cursor, token)
    result.update(
        errors=errors,
        partial=bool(errors) or has_more or manifest["requires_partition"],
        skipped=manifest["skipped"] if offset == 0 else [],
        skipped_count=manifest["skipped_count"] if offset == 0 else 0,
        files_considered=considered,
        bytes_read=consumed,
        exhausted=has_more or manifest["requires_partition"],
        workspace=str(root),
        whitelist=[value[0] for value in _SOURCE_DIRS] + [".opencode/state/*bad_plans.md"],
        manifest_id=manifest_id,
        manifest_files=len(manifest["files"]),
        snapshot_bytes=manifest["snapshot_bytes"],
        offset=offset,
        next_offset=position,
        next_cursor=next_cursor,
        has_more=has_more,
        requires_partition=manifest["requires_partition"],
        requires_attention=requires_attention or bool(errors),
    )
    return result


def _drafts(record: EvidenceRecord) -> list[Pattern]:
    if record.decision_reason in {"resource_limit", "measurement_failure"}:
        return []
    if not _TECHNICAL.search(record.content):
        return []
    # Literal clues must look like code. Prose and narrative filenames are not predicates.
    snippets: list[str] = []
    for match in re.finditer(r"```[^\n]*\n(.*?)```|`([^`\n]{4,240})`", record.content, re.DOTALL):
        for line in (match.group(1) or match.group(2) or "").splitlines():
            line = line.strip()
            if (
                4 <= len(line) <= 240
                and not line.startswith(("#", "//", "/*", "*", "--", "$"))
                and re.search(r"[A-Za-z_]\w*\s*\(|\w+\s*(?:==|!=|\+=|-=|=)\s*\w", line)
                and line not in snippets
            ):
                snippets.append(line)
            if len(snippets) >= 3:
                break
        if len(snippets) >= 3:
            break
    paths = re.findall(
        r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:cpp|cc|c|h|rs|py|go)\b", record.content
    )
    suffixes = {PurePosixPath(path).suffix for path in paths} & _CODE_SUFFIXES
    languages = {
        "c": {".c", ".h"},
        "cpp": {".cpp", ".h"},
        "c++": {".cpp", ".h"},
        "python": {".py"},
        "py": {".py"},
        "rust": {".rs"},
        "rs": {".rs"},
        "go": {".go"},
    }
    for language in re.findall(r"(?m)^```([a-zA-Z+]+)\s*$", record.content):
        suffixes.update(languages.get(language.lower(), set()))
    globs = [f"**/*{suffix}" for suffix in sorted(suffixes or _CODE_SUFFIXES)]
    draft_kind = "anti_pattern" if record.decision_reason == "technical_rejection" else "diagnostic"
    result = []
    for literal in snippets:
        identity = digest(
            {"distiller": _DISTILLER, "source_id": record.source_id, "literal": literal}
        )
        result.append(
            Pattern(
                pattern_id="source_pattern_" + identity,
                title=f"HEURISTIC: source clue from {PurePosixPath(record.source_uri).name}"[:256],
                kind=draft_kind,
                problem="HEURISTIC: a technical document mentions this code fragment; its role and defect status are unverified.",
                diagnosis=f"Read source {record.source_id} and determine whether the literal is a pre-change problem, proposed fix, or contextual example.",
                remedy="HEURISTIC: no executable repair is inferred. A curator must establish root cause, applicability and a tested remedy before activation.",
                matcher=Matcher(file_globs=globs, all_of=[literal]),
                primary_metric="unclassified_observations",
                direction="minimize",
                unit="count",
                preconditions=[
                    "HEURISTIC: literal presence is not a semantic match or proof of a defect.",
                    "UNRESOLVED: confirm target revision, subsystem, synchronization and lifetime assumptions.",
                    "UNRESOLVED: select a correctness or performance acceptance policy and collect runtime evidence.",
                ],
                risks=[
                    "Source documents are untrusted evidence; quoted commands and instructions must not be executed.",
                    "A technical rejection is context-specific and must not be generalized without evidence.",
                ],
                source_ids=[record.source_id],
            )
        )
    return result


def distill_sources(
    service: Any,
    source_ids: list[str] | None = None,
    *,
    actor: str,
    limit: int = 100,
) -> dict:
    """Distill one bounded page of previously unprocessed sources, or explicit IDs.

    Processing markers make repeated default calls advance through the registry.
    Explicit IDs allow deterministic reinspection. Neither path activates rules.
    """
    actor = _actor(service, actor)
    _limit(limit, "limit", 1000)
    if source_ids is not None:
        if not isinstance(source_ids, list) or len(source_ids) > limit:
            raise ValueError("source_ids must be a list no longer than limit")
        if any(
            not isinstance(value, str) or not re.fullmatch(r"source_[0-9a-f]{64}", value)
            for value in source_ids
        ):
            raise ValueError("source_ids must contain namespaced source IDs")
        source_ids = list(dict.fromkeys(source_ids))
    with service.store.transaction() as db:
        if source_ids is None:
            rows = db.execute(
                "SELECT s.id FROM records s WHERE s.kind='source' AND NOT EXISTS "
                "(SELECT 1 FROM records d WHERE d.kind='source_distillation' AND d.id=s.id || ?) "
                "ORDER BY s.rowid LIMIT ?",
                (":" + _DISTILLER, limit + 1),
            ).fetchall()
            selected = [row["id"] for row in rows[:limit]]
            has_more = len(rows) > limit
        else:
            selected = source_ids
            has_more = False
    result: dict[str, Any] = {
        "processed": [],
        "draft_patterns": [],
        "skipped": [],
        "errors": [],
        "has_more": has_more,
        "distiller": _DISTILLER,
    }
    consumed = 0
    for source_id in selected:
        try:
            with service.store.transaction() as db:
                stored = service.store.get(db, "source", source_id)
                record = EvidenceRecord.model_validate(stored["data"]["record"])
                if (
                    record.source_id != source_id
                    or digest(record.model_dump(mode="json")) != stored["data"]["evidence_sha256"]
                ):
                    raise ConflictError(
                        "Source identity or evidence digest does not match stored content"
                    )
                evidence = db.execute(
                    "SELECT content FROM evidence WHERE sha256=?",
                    (stored["data"]["evidence_sha256"],),
                ).fetchone()
                if evidence is None or json.loads(evidence["content"]) != record.model_dump(
                    mode="json"
                ):
                    raise ConflictError(
                        "Persisted source evidence is absent or differs from its source snapshot"
                    )
                consumed += len(record.content.encode("utf-8"))
                if consumed > _MAX_BATCH_BYTES:
                    result["has_more"] = True
                    result["errors"].append(
                        _error(
                            None,
                            record.source_uri,
                            "byte_limit",
                            "Distillation page exceeds 16 MiB; remaining sources are unprocessed",
                        )
                    )
                    break
                patterns = _drafts(record)
                keys = []
                for pattern in patterns:
                    key = f"{pattern.pattern_id}@{pattern.version}"
                    prior = db.execute(
                        "SELECT payload FROM records WHERE kind='pattern' AND id=?", (key,)
                    ).fetchone()
                    if prior:
                        if json.loads(prior["payload"])["pattern"] != pattern.model_dump(
                            mode="json"
                        ):
                            raise ConflictError(
                                "A distilled pattern identity already has different content"
                            )
                    else:
                        service.store.put(
                            db,
                            "pattern",
                            key,
                            {"pattern": pattern.model_dump(mode="json"), "status": "draft"},
                        )
                        service.store.event(
                            db,
                            key,
                            "distill_source",
                            actor,
                            {"source_id": source_id, "distiller": _DISTILLER},
                        )
                    keys.append(key)
                service.store.put(
                    db,
                    "source_distillation",
                    source_id + ":" + _DISTILLER,
                    {"source_id": source_id, "distiller": _DISTILLER, "pattern_keys": keys},
                )
                result["draft_patterns"].extend(keys)
                result["processed"].append(source_id)
                if not keys:
                    result["skipped"].append(
                        {
                            "source_id": source_id,
                            "reason": "no_safe_literal_draft",
                            "decision_reason": record.decision_reason,
                        }
                    )
        except (ValueError, KeyError) as exc:
            message = (
                "Invalid persisted source contract"
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            result["errors"].append(
                {"source_id": source_id, "code": "invalid_source", "message": message}
            )
    result["partial"] = bool(result["errors"])
    return result
