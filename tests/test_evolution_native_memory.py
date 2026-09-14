"""Native adapters preserve journal privacy, evidence gates and Hub ownership."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_evolution_service import Scenario
from test_evolution_service import git_bin as shared_git_bin

from hmopt.evolution import native_memory as native
from hmopt.evolution.service import EvolutionService
from hmopt.evolution.sources import import_workspace
from hmopt.evolution.store import ConflictError
from hmopt.sediment import journal as jm
from hmopt.sediment.validate import validate_candidate

git_bin = shared_git_bin
HUB = Path(__file__).resolve().parents[1] / "hm-skill-hub"


@pytest.fixture
def hub(tmp_path):
    target = tmp_path / "hub"
    shutil.copytree(HUB / "schemas", target / "schemas")
    return target


def journal_entry(root, *, contributor="owner", project="kernel", **kwargs):
    entry, errors = jm.write_entry(
        root,
        contributor=contributor,
        project=project,
        type="fact",
        title="Allocation lifetime constraint",
        body="Optimization must preserve the caller's ownership of the allocation.",
        **kwargs,
    )
    assert not errors
    return entry


def export_options(tmp_path, hub):
    root = tmp_path / "team-memory"
    root.mkdir(exist_ok=True)
    return {
        "memory_root": root,
        "hub_root": hub,
        "contributor": "owner",
        "project": "kernel",
        "actor": "curator",
        "request_id": "native-export-1",
        "title": "Reuse a lookup under the same ownership contract",
        "body": "Cache the lookup only while the caller holds the original lifetime reference.",
        "applies_when": ["The reference remains stable throughout the operation."],
        "invalidated_by": ["The object can be reclaimed between the lookup and its use."],
    }


def curated(tmp_path, git_bin):
    scenario = Scenario(tmp_path, git_bin).ready()
    scenario.validate()
    skill = scenario.skill()
    scenario.service.promote(
        skill["id"],
        tier="staging",
        actor="curator",
        expected_version=skill["version"],
        request_id="curate-native",
        note="Independent review of validated reusable evidence.",
    )
    return scenario, skill["id"]


def test_native_journal_import_is_scoped_paged_and_preserves_outcomes(tmp_path):
    root = tmp_path / "memory"
    entries = [
        journal_entry(root, outcome=outcome) for outcome in ("unknown", "validated", "failed")
    ]
    journal_entry(root, contributor="other")
    journal_entry(root, project="another")
    service = EvolutionService(tmp_path / "state")
    args = {
        "collection": "journal",
        "actor": "owner",
        "contributor": "owner",
        "project": "kernel",
        "max_files": 1,
    }
    first = native.import_native_memory(service, root, "kernel", **args)
    assert first["manifest_files"] == 3
    assert first["has_more"]
    ids = list(first["imported"])
    cursor = first["next_cursor"]
    while cursor:
        page = native.import_native_memory(service, root, "kernel", cursor=cursor, **args)
        ids.extend(page["imported"])
        cursor = page["next_cursor"]
    assert len(ids) == 3
    assert service.store.list("pattern") == []
    assert {
        service.store.read("source", sid)["data"]["record"]["source_uri"].split("/")[-1]
        for sid in ids
    } == {entry.id + ".md" for entry in entries}
    repeated = native.import_native_memory(service, root, "kernel", **{**args, "max_files": 10})
    assert len(repeated["existing"]) == 3
    with pytest.raises(ValueError, match="different memory scope"):
        native.import_native_memory(
            service, root, "kernel", cursor=first["next_cursor"], **{**args, "project": "another"}
        )


def test_native_snapshot_rejects_changed_file_and_ignores_later_additions(tmp_path):
    root = tmp_path / "memory"
    journal_entry(root)
    second = journal_entry(root)
    service = EvolutionService(tmp_path / "state")
    args = {
        "collection": "journal",
        "actor": "owner",
        "contributor": "owner",
        "project": "kernel",
        "max_files": 1,
    }
    first = native.import_native_memory(service, root, "kernel", **args)
    journal_entry(root)
    with Path(second.path).open("a", encoding="utf-8") as stream:
        stream.write("Changed since snapshot.\n")
    next_page = native.import_native_memory(
        service, root, "kernel", cursor=first["next_cursor"], **args
    )
    assert next_page["manifest_files"] == 2
    assert next_page["requires_attention"]
    assert not next_page["imported"]
    assert not next_page["has_more"]


def test_native_import_rejects_cross_contributor_secret_and_invalid_namespace(tmp_path):
    root = tmp_path / "memory"
    entry = journal_entry(root)
    path = Path(entry.path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("contributor: owner", "contributor: other"),
        encoding="utf-8",
    )
    secret = journal_entry(root)
    with Path(secret.path).open("a", encoding="utf-8") as stream:
        stream.write("ghp_" + "A" * 36)
    service = EvolutionService(tmp_path / "state")
    args = {"collection": "journal", "actor": "owner", "contributor": "owner", "project": "kernel"}
    page = native.import_native_memory(service, root, "kernel", **args)
    assert len(page["errors"]) == 2
    assert not page["imported"]
    assert "A" * 36 not in json.dumps(page)
    for change in ({"contributor": None}, {"project": "../kernel"}, {"project": None}):
        with pytest.raises(ValueError):
            native.import_native_memory(service, root, "kernel", **{**args, **change})


def test_native_hub_import_preserves_active_and_retired_context(tmp_path, hub):
    entry = journal_entry(tmp_path / "journal", outcome="validated")
    candidates, _, _ = jm.journal_to_candidates([entry], contributor="owner")
    record = candidates[0]["record"]
    knowledge = hub / "knowledge" / "targets" / "kernel" / "facts"
    knowledge.mkdir(parents=True)
    import yaml

    for index, status in enumerate(("active", "deprecated")):
        rec = {**record, "id": f"F00{index + 1}", "status": status}
        body = rec.pop("body")
        (knowledge / f"F00{index + 1}.md").write_text(
            "---\n" + yaml.safe_dump(rec, allow_unicode=True) + "---\n" + body, encoding="utf-8"
        )
    (hub / "staging").mkdir()
    (hub / "staging" / "must-not-read.md").write_text("unrelated", encoding="utf-8")
    service = EvolutionService(tmp_path / "state")
    page = native.import_native_memory(service, hub, "kernel", collection="hub", actor="owner")
    assert len(page["imported"]) == 2
    assert not page["errors"]
    assert not service.store.list("pattern")


def test_native_import_reads_the_real_hub_with_unquoted_yaml_dates(tmp_path):
    service = EvolutionService(tmp_path / "state")
    page = native.import_native_memory(service, HUB, "kernel", collection="hub", actor="owner")
    assert len(page["imported"]) == 6
    assert not page["errors"]


def test_workbench_collector_matches_actual_workflow_output_paths(tmp_path):
    from hmopt.evolution.workflow import _steps

    task = tmp_path / ".opencode/local/workspaces/optimize"
    expected = {}
    for role in ("architect", "implementer", "reviewer", "validator"):
        for correctness in (False, True):
            for step in _steps(role, correctness):
                for output in step["outputs"]:
                    path = task / "artifacts" / step["id"] / output
                    if path.suffix not in {".md", ".json"}:
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        "{}" if path.suffix == ".json" else "Reviewed evidence.", encoding="utf-8"
                    )
                    expected[path.relative_to(tmp_path).as_posix()] = (
                        "review"
                        if step["role"] == "reviewer"
                        else "experiment"
                        if step["role"] == "validator"
                        else "memory"
                        if step["role"] == "researcher"
                        else "decision"
                    )
    service = EvolutionService(tmp_path / "state")
    page = import_workspace(service, tmp_path, "kernel", actor="owner")
    records = [service.store.read("source", sid)["data"]["record"] for sid in page["imported"]]
    assert {record["source_uri"]: record["source_kind"] for record in records} == expected
    assert not page["errors"]


def test_native_no_progress_page_reports_attention(tmp_path):
    root = tmp_path / "memory"
    journal_entry(root)
    service = EvolutionService(tmp_path / "state")
    page = native.import_native_memory(
        service,
        root,
        "kernel",
        collection="journal",
        actor="owner",
        contributor="owner",
        project="kernel",
        max_bytes=1,
    )
    assert page["has_more"] and page["requires_attention"]
    assert page["next_cursor"] is None


def test_workbench_collector_imports_only_named_durable_artifacts(tmp_path):
    task = tmp_path / ".opencode/local/workspaces/optimize"
    artifacts = task / "artifacts"
    artifacts.mkdir(parents=True)
    for filename in ("task.md", "capsule.md", "decisions.md"):
        (task / filename).write_text("Optimization decision evidence.", encoding="utf-8")
    for filename in ("research-note.md", "plan.md", "code-review.md", "validation.md"):
        (artifacts / filename).write_text("Reviewed technical observation.", encoding="utf-8")
    for filename in ("session.json", "device-logs.md", "credentials.json", "research-logs.md"):
        (artifacts / filename).write_text("Not an allowed durable artifact.", encoding="utf-8")
    service = EvolutionService(tmp_path / "state")
    page = import_workspace(service, tmp_path, "kernel", actor="owner")
    assert len(page["imported"]) == 7
    assert not page["errors"]


def test_native_export_uses_existing_schema_and_only_stages(tmp_path, hub, git_bin):
    scenario, skill_id = curated(tmp_path, git_bin)
    args = export_options(tmp_path, hub)
    result = native.export_native_memory(scenario.service, skill_id, **args)
    assert result["status"] == "staged" and result["maturity"] == "L1"
    assert result["publication_status"] == "not_published" and result["merged"] is False
    entry = jm.parse_entry_file(Path(result["journal_path"]))
    assert entry.outcome == "validated"
    candidate = json.loads(Path(result["staging_path"]).read_text(encoding="utf-8"))
    assert not validate_candidate(candidate, hub)
    assert candidate["record"]["maturity"] == "L1"
    assert any(
        source["ref"].startswith("evolution:evidence:") for source in candidate["record"]["source"]
    )
    for source in candidate["record"]["source"]:
        if source["ref"].startswith("evolution:"):
            assert native.resolve_native_evidence(scenario.service, source["ref"])
    assert not (hub / "knowledge").exists()
    assert not (hub / "skills").exists()
    assert native.export_native_memory(scenario.service, skill_id, **args) == result
    assert len(list(args["memory_root"].rglob("J-*.md"))) == 1
    report = subprocess.run(
        [
            sys.executable,
            str(HUB / "tools/central_curate.py"),
            result["staging_path"],
            "--plan",
            "--knowledge-dir=" + str(hub / "knowledge"),
        ],
        cwd=HUB,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    assert "Curation report" in report.stdout and "add" in report.stdout
    assert not (hub / "knowledge").exists()
    with pytest.raises(ConflictError):
        native.export_native_memory(
            scenario.service, skill_id, **{**args, "body": "Changed recipe."}
        )


def test_native_export_requires_reviewed_real_evidence(tmp_path, hub, git_bin):
    scenario = Scenario(tmp_path, git_bin).ready()
    scenario.validate(allow_synthetic=True)
    args = export_options(tmp_path, hub)
    with pytest.raises(ValueError, match="curated real passing"):
        native.export_native_memory(scenario.service, scenario.skill()["id"], **args)
    assert not list(args["memory_root"].rglob("*.md"))
    assert not (hub / "staging").exists()


def test_native_export_requires_matching_independent_curator_and_redaction(tmp_path, hub, git_bin):
    scenario, skill_id = curated(tmp_path, git_bin)
    args = export_options(tmp_path, hub)
    with pytest.raises(ValueError, match="independent evidence curator"):
        native.export_native_memory(scenario.service, skill_id, **{**args, "actor": "implementer"})
    with pytest.raises(ValueError, match="secret pattern"):
        native.export_native_memory(
            scenario.service, skill_id, **{**args, "body": "Token ghp_" + "B" * 36}
        )
    assert not list(args["memory_root"].rglob("*.md"))


def test_native_export_recovers_prepared_intent_without_duplicate_journal(
    tmp_path,
    hub,
    git_bin,
    monkeypatch,
):
    scenario, skill_id = curated(tmp_path, git_bin)
    args = export_options(tmp_path, hub)
    original = native._write_export_file

    def interrupted(root, relative, text, expected):
        if relative.startswith("staging/"):
            raise OSError("Injected staging filesystem failure")
        return original(root, relative, text, expected)

    monkeypatch.setattr(native, "_write_export_file", interrupted)
    with pytest.raises(OSError, match="Injected"):
        native.export_native_memory(scenario.service, skill_id, **args)
    assert len(list(args["memory_root"].rglob("J-*.md"))) == 1
    assert scenario.service.store.list("native_export")[0]["data"]["status"] == "prepared"
    monkeypatch.setattr(native, "_write_export_file", original)
    result = native.export_native_memory(scenario.service, skill_id, **args)
    assert result["status"] == "staged"
    assert len(list(args["memory_root"].rglob("J-*.md"))) == 1
    assert len(list((hub / "staging").rglob("*.jsonl"))) == 1
    assert len(scenario.service.store.audit(result["export_id"])) == 1


def test_native_export_partial_temporary_write_never_publishes_a_final_file(
    tmp_path,
    hub,
    git_bin,
    monkeypatch,
):
    from hmopt.evolution import workflow

    scenario, skill_id = curated(tmp_path, git_bin)
    args = export_options(tmp_path, hub)
    factory = workflow.tempfile.NamedTemporaryFile

    def broken_writer(*positional, **keywords):
        stream = factory(*positional, **keywords)
        write = stream.write

        def partial_write(raw):
            write(raw[:5])
            raise OSError("Interrupted during temporary write")

        stream.write = partial_write
        return stream

    monkeypatch.setattr(workflow.tempfile, "NamedTemporaryFile", broken_writer)
    with pytest.raises(OSError, match="Interrupted"):
        native.export_native_memory(scenario.service, skill_id, **args)
    assert not list(args["memory_root"].rglob("J-*.md"))
    assert not list((hub / "staging").rglob("*.jsonl"))
    monkeypatch.setattr(workflow.tempfile, "NamedTemporaryFile", factory)
    assert native.export_native_memory(scenario.service, skill_id, **args)["status"] == "staged"


def test_native_export_does_not_recreate_forgotten_or_overwrite_changed_data(
    tmp_path, hub, git_bin
):
    scenario, skill_id = curated(tmp_path, git_bin)
    args = export_options(tmp_path, hub)
    result = native.export_native_memory(scenario.service, skill_id, **args)
    Path(result["staging_path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ConflictError, match="changed"):
        native.export_native_memory(scenario.service, skill_id, **args)
    assert jm.forget_entry(args["memory_root"], "owner", result["journal_id"])
    with pytest.raises(ConflictError, match="missing"):
        native.export_native_memory(scenario.service, skill_id, **args)
    assert not Path(result["journal_path"]).exists()


def test_native_export_rejects_overlapping_roots(tmp_path, hub):
    service = EvolutionService(tmp_path / "state")
    args = export_options(tmp_path, hub)
    with pytest.raises(ValueError, match="must not overlap"):
        native.export_native_memory(service, "not-read", **{**args, "memory_root": hub})


def test_native_evidence_encoding_preserves_digest_without_secret_scan_exemption():
    for sha in ("0" * 64, "f" * 64, "1a" * 32):
        assert not jm.redact_scan(native.evidence_reference(sha))


def test_native_snapshot_budget_and_malformed_yaml_remain_explicit(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    journal_entry(root)
    malformed = journal_entry(root)
    Path(malformed.path).write_text("---\ntitle: [broken\n---\nbody", encoding="utf-8")
    service = EvolutionService(tmp_path / "state")
    args = {"collection": "journal", "actor": "owner", "contributor": "owner", "project": "kernel"}
    result = native.import_native_memory(service, root, "kernel", **args)
    assert len(result["imported"]) == 1 and len(result["errors"]) == 1
    monkeypatch.setattr(native, "_MAX_FILES", 1)
    capped = native.import_native_memory(service, root, "kernel", **args)
    assert capped["requires_partition"] and capped["partial"]
    assert capped["next_cursor"] is None
