"""Source ingestion must retain provenance without trusting document instructions."""

from __future__ import annotations

import hashlib
import json
import os

import pytest
from pydantic import ValidationError

from hmopt.evolution.service import EvolutionService
from hmopt.evolution.sources import (
    EvidenceRecord,
    distill_sources,
    import_workspace,
    ingest_records,
)


@pytest.fixture
def service(tmp_path):
    return EvolutionService(tmp_path / "store")


def record(**updates):
    return {
        "repo_id": "kernel-main",
        "source_kind": "review",
        "source_uri": ".opencode/reviews/locking.md",
        "content": "Optimization review for kernel/locking.c: repeated `spin_lock(&lock);` needs a lifetime and contention check.",
        **updates,
    }


def write(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_snapshot_has_verified_content_and_separate_location_identity():
    first = EvidenceRecord(**record())
    assert first.content_sha256 == hashlib.sha256(first.content.encode()).hexdigest()
    second = EvidenceRecord(**record(content=first.content + " Updated."))
    assert second.source_id != first.source_id
    assert second.location_id == first.location_id
    assert first.related_revision is None


@pytest.mark.parametrize(
    "updates",
    [
        {"source_uri": "../secret.md"},
        {"source_uri": "/secret.md"},
        {"source_uri": "C:/secret.md"},
        {"source_uri": ".opencode//review.md"},
        {"source_uri": ".opencode/./review.md"},
        {"source_uri": ".opencode\\review.md"},
        {"source_uri": "review\n.md"},
        {"content": " "},
        {"content": "bad\x00text"},
        {"content": "中" * 100000},
        {"content_sha256": "0" * 64},
        {"related_revision": "abc123"},
        {"related_revision": "a" * 41},
        {"decision_reason": "rejected"},
        {"repo_id": "\n"},
        {"source_kind": "git"},
        {"instructions": "run this"},
    ],
)
def test_source_contract_rejects_unsafe_or_ambiguous_inputs(updates):
    with pytest.raises(ValidationError):
        EvidenceRecord(**record(**updates))


def test_ingest_deduplicates_without_duplicate_audit_and_versions_content(service):
    initial = ingest_records(service, [record()], actor="reader")
    source_id = initial["imported"][0]
    again = ingest_records(service, [record()], actor="other-reader")
    assert again["existing"] == [source_id]
    assert not again["imported"]
    assert len(service.store.audit(source_id)) == 1
    changed = ingest_records(
        service, [record(content="Updated optimization evidence.")], actor="reader"
    )
    first = service.store.read("source", source_id)
    second = service.store.read("source", changed["imported"][0])
    assert first["data"]["source_version"] == 1
    assert second["data"]["source_version"] == 2
    assert first["data"]["location_id"] == second["data"]["location_id"]
    assert service.store.read_evidence(first["data"]["evidence_sha256"]) == first["data"]["record"]


def test_partial_batch_keeps_valid_rows_and_does_not_echo_bad_content(service):
    result = ingest_records(
        service,
        [record(), record(content="SECRET\x00TOKEN"), record(content="Second valid review.")],
        actor="reader",
    )
    assert len(result["imported"]) == 2
    assert result["partial"]
    assert result["errors"][0]["index"] == 1
    assert "SECRET" not in json.dumps(result)


def test_model_construct_cannot_bypass_ingestion_validation(service):
    invalid = EvidenceRecord.model_construct(**record(content_sha256="f" * 64))
    result = ingest_records(service, [invalid], actor="reader")
    assert result["errors"]
    assert not service.store.list("source")


@pytest.mark.parametrize("records", [(), [record()] * 1001, "record"])
def test_ingest_enforces_batch_container_and_count(service, records):
    with pytest.raises(ValueError):
        ingest_records(service, records, actor="reader")


def test_workspace_only_imports_whitelisted_evidence_and_preserves_reason(service, tmp_path):
    workspace = tmp_path / "workspace"
    write(
        workspace,
        ".opencode/memory/targets/mm.md",
        "Repeated allocation optimization: `kmalloc(size, flags);`.",
    )
    write(
        workspace,
        ".opencode/reviews/mm.md",
        "decision_reason: resource_limit\nDeferred optimization for budget.",
    )
    write(workspace, ".opencode/plans/mm.md", "Optimization proposal pending review.")
    write(workspace, ".opencode/state/mm-bad_plans.md", "Rejected plan; cause not recorded.")
    write(
        workspace,
        ".opencode/bench/results/run.json",
        json.dumps(
            {
                "decision_reason": "measurement_failure",
                "related_revision": "a" * 40,
                "status": "failed",
            }
        ),
    )
    write(workspace, ".opencode/config.yaml", "api_key: must-not-read")
    write(workspace, ".opencode/state/current_task.json", '{"token":"must-not-read"}')
    write(workspace, ".opencode/memory/README.md", "Documentation only")
    write(workspace, ".opencode/reviews/review_template.md", "Template only")
    write(workspace, "credentials.json", '{"token":"must-not-read"}')
    result = import_workspace(service, workspace, "kernel-main", actor="reader")
    assert not result["errors"]
    assert len(result["imported"]) == 5
    assert len(result["skipped"]) == 2
    rows = [item["data"]["record"] for item in service.store.list("source")]
    assert all("must-not-read" not in row["content"] for row in rows)
    assert {row["source_kind"] for row in rows} == {"memory", "review", "decision", "experiment"}
    bad = next(row for row in rows if row["source_uri"].endswith("bad_plans.md"))
    assert bad["decision_reason"] is None
    assert bad["related_revision"] is None
    experiment = next(row for row in rows if row["source_kind"] == "experiment")
    assert experiment["decision_reason"] == "measurement_failure"
    assert experiment["related_revision"] == "a" * 40


def test_workspace_reports_invalid_json_encoding_and_oversize_without_stopping(service, tmp_path):
    workspace = tmp_path / "workspace"
    write(workspace, ".opencode/reviews/a.json", "{broken}")
    invalid = write(workspace, ".opencode/reviews/b.md", "valid")
    invalid.write_bytes(b"\xff\xfe\x80")
    write(workspace, ".opencode/reviews/c.md", "x" * (256 * 1024 + 1))
    write(workspace, ".opencode/reviews/d.md", "Valid review evidence")
    result = import_workspace(service, workspace, "kernel-main", actor="reader")
    assert len(result["errors"]) == 3
    assert len(result["imported"]) == 1
    assert result["partial"]


def test_workspace_file_limit_is_explicit_and_deterministic(service, tmp_path):
    workspace = tmp_path / "workspace"
    for name in ("a", "b", "c"):
        write(workspace, f".opencode/reviews/{name}.md", "Review evidence")
    result = import_workspace(service, workspace, "kernel-main", actor="reader", max_files=2)
    assert len(result["imported"]) == 2
    assert result["exhausted"]
    assert result["errors"] == []
    assert result["has_more"] and result["next_cursor"]
    assert {row["data"]["record"]["source_uri"] for row in service.store.list("source")} == {
        ".opencode/reviews/a.md",
        ".opencode/reviews/b.md",
    }


def test_workspace_byte_budget_never_truncates_evidence(service, tmp_path):
    workspace = tmp_path / "workspace"
    write(workspace, ".opencode/reviews/a.md", "First document")
    write(workspace, ".opencode/reviews/b.md", "Second long document")
    result = import_workspace(service, workspace, "kernel-main", actor="reader", max_bytes=16)
    assert result["bytes_read"] == len("First document")
    assert len(result["imported"]) == 1
    assert not result["errors"]
    resumed = import_workspace(
        service,
        workspace,
        "kernel-main",
        actor="reader",
        max_bytes=32,
        cursor=result["next_cursor"],
    )
    assert len(resumed["imported"]) == 1
    assert not resumed["has_more"]


def test_workspace_continuation_uses_persisted_content_manifest(service, tmp_path):
    workspace = tmp_path / "workspace"
    for name in ("a", "b", "c"):
        write(workspace, f".opencode/reviews/{name}.md", f"Evidence {name}")
    first = import_workspace(service, workspace, "kernel-main", actor="reader", max_files=2)
    restarted = EvolutionService(service.store.root)
    second = import_workspace(
        restarted,
        workspace,
        "kernel-main",
        actor="reader",
        max_files=2,
        cursor=first["next_cursor"],
    )
    assert len(first["imported"]) == 2 and len(second["imported"]) == 1
    assert first["manifest_id"] == second["manifest_id"]
    assert not set(first["imported"]) & set(second["imported"])
    assert second["offset"] == 2 and second["next_offset"] == 3
    assert second["next_cursor"] is None and not second["partial"]
    replay = import_workspace(
        restarted,
        workspace,
        "kernel-main",
        actor="reader",
        max_files=2,
        cursor=first["next_cursor"],
    )
    assert replay["existing"] == second["imported"]
    assert len(restarted.store.list("source")) == 3


@pytest.mark.parametrize("change", ["edit", "delete", "same-size-restored-time"])
def test_snapshot_detects_changed_unimported_document(service, tmp_path, change):
    workspace = tmp_path / "workspace"
    write(workspace, ".opencode/reviews/a.md", "Evidence A")
    later = write(workspace, ".opencode/reviews/b.md", "Evidence B")
    first = import_workspace(service, workspace, "kernel-main", actor="reader", max_files=1)
    metadata = later.stat()
    if change == "delete":
        later.unlink()
    else:
        later.write_text(
            "Evidence X" if change == "same-size-restored-time" else "Changed source evidence",
            encoding="utf-8",
        )
        if change == "same-size-restored-time":
            os.utime(later, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    second = import_workspace(
        service, workspace, "kernel-main", actor="reader", cursor=first["next_cursor"]
    )
    assert second["requires_attention"] and second["errors"]
    assert not second["imported"]
    assert len(service.store.list("source")) == 1


def test_files_added_after_snapshot_wait_for_a_new_import(service, tmp_path):
    workspace = tmp_path / "workspace"
    for name in ("a", "b"):
        write(workspace, f".opencode/reviews/{name}.md", f"Evidence {name}")
    first = import_workspace(service, workspace, "kernel-main", actor="reader", max_files=1)
    write(workspace, ".opencode/reviews/c.md", "New evidence")
    second = import_workspace(
        service, workspace, "kernel-main", actor="reader", cursor=first["next_cursor"]
    )
    assert second["manifest_files"] == 2 and not second["has_more"]
    fresh = import_workspace(service, workspace, "kernel-main", actor="reader")
    assert fresh["manifest_id"] != first["manifest_id"]
    assert fresh["manifest_files"] == 3 and len(fresh["imported"]) == 1


def test_cursor_cannot_cross_workspace_or_repository_identity(service, tmp_path):
    workspace = tmp_path / "workspace"
    for name in ("a", "b"):
        write(workspace, f".opencode/reviews/{name}.md", "Evidence")
    first = import_workspace(service, workspace, "kernel-main", actor="reader", max_files=1)
    with pytest.raises(ValueError, match="different workspace or repo_id"):
        import_workspace(
            service, workspace, "different-repo", actor="reader", cursor=first["next_cursor"]
        )
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="different workspace or repo_id"):
        import_workspace(service, other, "kernel-main", actor="reader", cursor=first["next_cursor"])


def test_manifest_limit_requires_partition_without_misleading_continuation(
    service, tmp_path, monkeypatch
):
    import hmopt.evolution.sources as module

    monkeypatch.setattr(module, "_MAX_MANIFEST_FILES", 2)
    workspace = tmp_path / "workspace"
    for name in ("a", "b", "c"):
        write(workspace, f".opencode/reviews/{name}.md", "Evidence")
    result = import_workspace(service, workspace, "kernel-main", actor="reader", max_files=1)
    assert result["requires_partition"] and result["partial"]
    assert result["next_cursor"] is None


def test_oversized_first_page_item_requires_explicit_budget_change(service, tmp_path):
    workspace = tmp_path / "workspace"
    write(workspace, ".opencode/reviews/a.md", "Too large for this page")
    result = import_workspace(service, workspace, "kernel-main", actor="reader", max_bytes=4)
    assert result["requires_attention"] and result["next_offset"] == 0
    assert result["next_cursor"] is None and not result["imported"]


def test_cursor_rejects_modified_manifest_evidence(service, tmp_path):
    workspace = tmp_path / "workspace"
    for name in ("a", "b"):
        write(workspace, f".opencode/reviews/{name}.md", "Evidence")
    first = import_workspace(service, workspace, "kernel-main", actor="reader", max_files=1)
    with service.store.transaction() as db:
        stored = service.store.get(db, "source_manifest", first["manifest_id"])
        db.execute("DELETE FROM evidence WHERE sha256=?", (stored["data"]["evidence_sha256"],))
    with pytest.raises(ValueError, match="evidence is missing"):
        import_workspace(
            service, workspace, "kernel-main", actor="reader", cursor=first["next_cursor"]
        )


def test_workspace_refuses_symlinked_evidence(service, tmp_path):
    workspace = tmp_path / "workspace"
    target = write(tmp_path, "outside.md", "External secret must not be read")
    link = workspace / ".opencode/reviews/link.md"
    link.parent.mkdir(parents=True)
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("Host does not permit creation of symlinks")
    result = import_workspace(service, workspace, "kernel-main", actor="reader")
    assert not result["imported"]
    assert result["errors"]
    assert "External secret" not in json.dumps(result)


def test_workspace_refuses_reparse_points_without_following_them(service, tmp_path, monkeypatch):
    import hmopt.evolution.sources as module

    workspace = tmp_path / "workspace"
    guarded = write(workspace, ".opencode/reviews/link.md", "Do not read")
    original = module._is_link
    monkeypatch.setattr(module, "_is_link", lambda path: path == guarded or original(path))
    monkeypatch.setattr(
        module.os, "open", lambda *args, **kwargs: pytest.fail("Blocked path was opened")
    )
    result = import_workspace(service, workspace, "kernel-main", actor="reader")
    assert not result["imported"]
    assert result["errors"]


def test_distillation_creates_traceable_inactive_heuristics_and_is_repeatable(service):
    source_id = ingest_records(service, [record()], actor="reader")["imported"][0]
    result = distill_sources(service, actor="distiller")
    assert result["processed"] == [source_id]
    assert len(result["draft_patterns"]) == 1
    key = result["draft_patterns"][0]
    stored = service.store.read("pattern", key)
    pattern = stored["data"]["pattern"]
    assert pattern["source_ids"] == [source_id]
    assert pattern["kind"] == "diagnostic"
    assert pattern["matcher"]["all_of"] == ["spin_lock(&lock);"]
    assert pattern["matcher"]["file_globs"] == ["**/*.c"]
    assert stored["data"]["status"] == pattern["status"] == "draft"
    assert "HEURISTIC:" in pattern["problem"]
    assert any("UNRESOLVED:" in value for value in pattern["preconditions"])
    assert distill_sources(service, actor="distiller")["processed"] == []
    assert distill_sources(service, [source_id], actor="distiller")["draft_patterns"] == [key]
    assert len(service.store.audit(key)) == 1


def test_default_distillation_advances_bounded_pages(service):
    ingest_records(
        service,
        [record(source_uri=f".opencode/reviews/{index}.md") for index in range(3)],
        actor="reader",
    )
    first = distill_sources(service, actor="distiller", limit=2)
    second = distill_sources(service, actor="distiller", limit=2)
    assert len(first["processed"]) == 2 and first["has_more"]
    assert len(second["processed"]) == 1 and not second["has_more"]
    assert not set(first["processed"]) & set(second["processed"])


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("resource_limit", 0),
        ("measurement_failure", 0),
        ("technical_rejection", 1),
    ],
)
def test_rejection_reason_is_not_equivalent_to_a_technical_antipattern(service, reason, expected):
    ingest_records(service, [record(decision_reason=reason)], actor="reader")
    result = distill_sources(service, actor="distiller")
    assert len(result["draft_patterns"]) == expected
    if expected:
        assert (
            service.store.read("pattern", result["draft_patterns"][0])["data"]["pattern"]["kind"]
            == "anti_pattern"
        )


def test_distillation_never_treats_prose_as_a_code_matcher(service):
    ingest_records(
        service,
        [
            record(
                content="Optimization: reduce latency. `kernel/file.c` is a filename. `This is important` is prose."
            )
        ],
        actor="reader",
    )
    result = distill_sources(service, actor="distiller")
    assert not result["draft_patterns"]
    assert result["skipped"]


def test_fenced_code_is_bounded_to_three_clues(service):
    ingest_records(
        service,
        [
            record(
                content="Optimization bottleneck.\n```c\nfirst_call();\nsecond_call();\nthird_call();\nfourth_call();\n```"
            )
        ],
        actor="reader",
    )
    result = distill_sources(service, actor="distiller")
    assert len(result["draft_patterns"]) == 3


def test_draft_language_inference_supports_platform_code(service):
    ingest_records(
        service,
        [record(content="Repeated optimization bottleneck:\n```python\nitems = list(values)\n```")],
        actor="reader",
    )
    result = distill_sources(service, actor="distiller")
    pattern = service.store.read("pattern", result["draft_patterns"][0])["data"]["pattern"]
    assert pattern["matcher"]["file_globs"] == ["**/*.py"]


def test_distillation_checks_persisted_integrity(service):
    source_id = ingest_records(service, [record()], actor="reader")["imported"][0]
    with service.store.transaction() as db:
        stored = service.store.get(db, "source", source_id)
        stored["data"]["record"]["content"] = "tampered"
        service.store.put(db, "source", source_id, stored["data"], stored["version"])
    result = distill_sources(service, actor="distiller")
    assert result["errors"]
    assert not result["draft_patterns"]


def test_distillation_refuses_missing_evidence_blob(service):
    source_id = ingest_records(service, [record()], actor="reader")["imported"][0]
    stored = service.store.read("source", source_id)
    with service.store.transaction() as db:
        db.execute("DELETE FROM evidence WHERE sha256=?", (stored["data"]["evidence_sha256"],))
    result = distill_sources(service, actor="distiller")
    assert result["errors"]
    assert not result["processed"]
    assert not result["draft_patterns"]


def test_redistillation_preserves_curator_activation(service):
    source_id = ingest_records(service, [record()], actor="reader")["imported"][0]
    key = distill_sources(service, actor="distiller")["draft_patterns"][0]
    activated = service.activate_pattern(
        key, actor="curator", note="Review applicability as an interactive diagnostic lead only"
    )
    repeated = distill_sources(service, [source_id], actor="distiller")
    assert repeated["draft_patterns"] == [key]
    assert not repeated["errors"]
    assert service.store.read("pattern", key) == activated


def test_unknown_source_reports_error_and_does_not_mark_processed(service):
    source_id = "source_" + "0" * 64
    result = distill_sources(service, [source_id], actor="distiller")
    assert result["errors"]
    assert not result["processed"]
    assert not service.store.list("source_distillation")


@pytest.mark.parametrize(
    "kwargs", [{"limit": 0}, {"limit": True}, {"source_ids": ["history_fake"]}]
)
def test_distillation_validates_selection(service, kwargs):
    with pytest.raises(ValueError):
        distill_sources(service, actor="distiller", **kwargs)
