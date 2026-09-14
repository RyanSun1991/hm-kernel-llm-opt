"""Real Git evidence and protocol tests with explicitly supplied model-response fixtures."""

from copy import deepcopy

import pytest
from evolution_analysis_helpers import no_pattern_report
from test_evolution_mining import GIT, commit, git, history
from test_evolution_mining import repo as shared_repo

from hmopt.evolution.change_analysis import (
    HistoryAnalysis,
    analysis_backlog,
    prepare_analysis,
    submit_analysis,
)
from hmopt.evolution.discovery import DiscoveryConfig, run_discovery
from hmopt.evolution.service import EvolutionService
from hmopt.evolution.store import ConflictError

repo = shared_repo


@pytest.fixture
def change(repo, tmp_path):
    old = "def count(raw):\n    return int(raw) if raw else 0\n"
    new = "def count(raw):\n    return int(raw) if raw.strip() else 0\n"
    (repo / "count.py").write_text(old, encoding="utf-8", newline="\n")
    base = commit(repo, "initial")
    (repo / "count.py").write_text(new, encoding="utf-8", newline="\n")
    revision = commit(repo, "update")  # Deliberately no fix/perf keywords.
    service = EvolutionService(tmp_path.parent / (tmp_path.name + "-state"), git_bin=GIT)
    records = history(repo, after_revision=base)
    result = service.ingest_history(records)
    assert result["analysis_required"] and result["draft_patterns"] == []
    packet = prepare_analysis(service, repo, records[0].source_id)
    return service, repo, packet, old, new, revision


def report(prepared):
    value = no_pattern_report(prepared)
    value.update(outcome="patterns", summary="Normalize blank input before numeric conversion.")
    value["findings"][0].update(
        what_changed="The nonempty guard now strips whitespace before testing emptiness.",
        how_behavior_changes="Whitespace-only strings take the zero branch instead of int().",
        why="The code suggests avoiding conversion errors for blank input; author intent is inferred.",
        why_status="inferred",
    )
    value["proposals"] = [
        {
            "title": "Whitespace normalization before numeric conversion",
            "kind": "anti_pattern",
            "mechanism": "A truthiness guard on raw text does not reject whitespace-only strings.",
            "problem": "Whitespace is nonempty but cannot be converted directly to an integer.",
            "diagnosis": "Check whether the input contract treats blank text as the default value.",
            "remedy": "Normalize only the guard when whitespace-only input is defined as empty.",
            "matcher": {"file_globs": ["**/*.py"], "all_of": ["int(raw) if raw else 0"]},
            "exemplar_path": "count.py",
            "finding_indexes": [0],
            "preconditions": ["The input must be a string and blank input must mean zero."],
            "risks": ["An API that intentionally rejects whitespace must retain that behavior."],
            "negative_examples": [
                "Strict parsers intentionally reject blank strings as invalid input."
            ],
            "validation_plan": ["Compare empty, blank, signed, padded and invalid numeric inputs."],
            "primary_metric": "correctness_failure_count",
            "metric_rationale": "This changes input handling; instruction count is not the success metric.",
        }
    ]
    return value


def submit(service, prepared, value, **kwargs):
    return submit_analysis(
        service,
        HistoryAnalysis.model_validate(value),
        actor="researcher",
        expected_version=prepared["job"]["version"],
        request_id=kwargs.get("request_id", "analysis-1"),
    )


def test_large_file_small_diff_keeps_exact_hunk_line_numbers(repo, tmp_path):
    lines = [f"int filler_{n};" for n in range(9000)]
    lines[4500] = "int target(void) { return expensive_call(); }"
    (repo / "large.c").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    base = commit(repo, "large source")
    lines[4500] = "int target(void) { return cached_value; }"
    (repo / "large.c").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    commit(repo, "update")
    service = EvolutionService(tmp_path / "state", git_bin=GIT)
    source = history(repo, after_revision=base)[0]
    service.ingest_history([source])
    prepared = prepare_analysis(service, repo, source.source_id)
    assert prepared["packet"]["coverage_complete"]
    before = prepared["packet"]["files"][0]["before"]
    assert before["scope"] == "all_changed_hunks_only"
    assert 4501 in before["line_numbers"] and len(before["content"]) < 1000
    value = no_pattern_report(prepared)
    assert value["findings"][0]["citations"][0]["line_start"] == 4501
    result = submit(service, prepared, value)
    assert result["data"]["status"] == "no_pattern"


def test_explicit_packet_refresh_preserves_old_evidence_and_fences_old_submission(change):
    service, repo, prepared, *_ = change
    refreshed = prepare_analysis(
        service,
        repo,
        prepared["job"]["id"],
        refresh=True,
        expected_version=prepared["job"]["version"],
    )
    assert refreshed["job"]["version"] > prepared["job"]["version"]
    assert service.store.read_evidence(prepared["job"]["data"]["packet_sha256"])
    with pytest.raises(ConflictError, match="Stale analysis"):
        submit(service, prepared, report(prepared))


def test_bland_title_code_analysis_produces_checked_draft_then_search(change):
    service, repo, prepared, old, new, _revision = change
    assert prepared["packet"]["auxiliary"]["subject"] == "update"
    file = prepared["packet"]["files"][0]
    assert file["before"]["content"] == old and file["after"]["content"] == new
    (repo / "count.py").write_text("dirty worktree", encoding="utf-8", newline="\n")
    assert prepare_analysis(service, repo, prepared["packet"]["source_id"]) == prepared
    row = submit(service, prepared, report(prepared))
    key = row["data"]["draft_patterns"][0]
    draft = service.store.read("pattern", key)
    assert draft["data"]["status"] == "draft"
    assert draft["data"]["matcher_check"]["after_matches"] is False
    assert draft["data"]["pattern"]["primary_metric"] == "correctness_failure_count"
    assert service.scan(repo, owners={"**": "owner"})["candidates"] == []
    service.activate_pattern(
        key, actor="curator", note="Reviewed applicability and counterexamples."
    )
    assert service.scan(repo, owners={"**": "owner"})["candidates"] == []
    (repo / "other.py").write_text(old, encoding="utf-8", newline="\n")
    commit(repo, "another usage")
    candidates = service.scan(repo, owners={"**": "owner"})["candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]["data"]["candidate"]
    assert candidate["path"] == "other.py" and candidate["lane"] == "workbench"
    assert candidate["source_ids"] == [prepared["packet"]["source_id"]]


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("quote", "exact source lines"),
        ("line", "exact source lines"),
        ("no_before", "both available"),
        ("unknown_reason", "Resolve the mechanism"),
        ("no_changed_line", "actual changed line"),
        ("after_match", "after code"),
        ("no_before_match", "before code"),
        ("wrong_glob", "globs exclude"),
        ("no_coverage", "cover every"),
        ("wrong_digest", "exact prepared packet"),
    ],
)
def test_untrusted_model_results_fail_closed_without_partial_drafts(change, mutation, error):
    service, _, prepared, *_ = change
    value = report(prepared)
    finding = value["findings"][0]
    if mutation == "quote":
        finding["citations"][0]["quote"] = "invented code"
    elif mutation == "line":
        finding["citations"][0]["line_start"] = 999
    elif mutation == "no_before":
        finding["citations"] = [finding["citations"][1]]
    elif mutation == "unknown_reason":
        finding["why_status"] = "unknown"
    elif mutation == "no_changed_line":
        for citation in finding["citations"]:
            citation.update(line_start=1, quote="def count(raw):")
    elif mutation == "after_match":
        value["proposals"][0]["matcher"]["all_of"] = ["int(raw)"]
    elif mutation == "no_before_match":
        value["proposals"][0]["matcher"]["all_of"] = ["int(raw) if raw.strip() else 0"]
    elif mutation == "wrong_glob":
        value["proposals"][0]["matcher"]["file_globs"] = ["*.c"]
    elif mutation == "no_coverage":
        value.update(outcome="no_pattern", findings=[], proposals=[])
    elif mutation == "wrong_digest":
        value["packet_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=error):
        submit(service, prepared, value)
    assert service.store.list("pattern") == []
    assert service.store.read("history_analysis", value["source_id"])["data"]["status"] == "pending"


def test_atomic_multi_proposal_replay_stale_and_conflicting_requests(change):
    service, _, prepared, *_ = change
    value = report(prepared)
    invalid = deepcopy(value)
    invalid["proposals"].append(deepcopy(value["proposals"][0]))
    invalid["proposals"][1]["matcher"]["all_of"] = ["not present"]
    with pytest.raises(ValueError):
        submit(service, prepared, invalid)
    assert not service.store.list("pattern")
    accepted = submit(service, prepared, value)
    assert submit(service, prepared, value) == accepted
    with pytest.raises(ConflictError, match="Stale"):
        submit(service, prepared, value, request_id="other")
    value["summary"] = "An altered explanation under the same request ID."
    with pytest.raises(ConflictError, match="different arguments"):
        submit(service, prepared, value)
    assert len(service.store.audit(value["source_id"])) == 1


def test_keyword_title_without_code_is_no_pattern(repo, tmp_path):
    (repo / "x.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    base = commit(repo, "base")
    git(repo, "commit", "--allow-empty", "-m", "fix perf optimize fast")
    service = EvolutionService(tmp_path.parent / (tmp_path.name + "-state"), git_bin=GIT)
    [record] = history(repo, after_revision=base)
    service.ingest_history([record])
    prepared = prepare_analysis(service, repo, record.source_id)
    assert prepared["packet"]["files"] == []
    result = submit(service, prepared, no_pattern_report(prepared))
    assert result["data"]["status"] == "no_pattern"
    assert service.store.list("pattern") == []


def test_truncated_and_binary_context_cannot_yield_patterns(change, monkeypatch):
    from hmopt.evolution import change_analysis

    service, repo, old_packet, *_ = change
    (repo / "x.bin").write_bytes(b"\x00binary")
    commit(repo, "files")
    changes = history(repo, after_revision=old_packet["packet"]["revision"])
    service.ingest_history(changes)
    monkeypatch.setattr(change_analysis, "MAX_BLOB_BYTES", 4)
    prepared = prepare_analysis(service, repo, changes[0].source_id)
    assert prepared["packet"]["coverage_complete"] is False
    value = no_pattern_report(prepared)
    with pytest.raises(ValueError, match="Incomplete"):
        submit(service, prepared, value)
    value.update(
        outcome="needs_context", unknowns=["Binary content needs a suitable domain decoder."]
    )
    result = submit(service, prepared, value)
    assert result["data"]["status"] == "needs_context"
    assert changes[0].source_id in [j["id"] for j in analysis_backlog(service, repo)["jobs"]]


def test_root_rename_multifile_and_deletion_are_explicit(repo, tmp_path):
    service = EvolutionService(tmp_path.parent / (tmp_path.name + "-state"), git_bin=GIT)
    (repo / "中文.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    initial = commit(repo, "initial")
    git(repo, "mv", "中文.py", "renamed.py")
    (repo / "test.py").write_text("assert True\n", encoding="utf-8", newline="\n")
    final = commit(repo, "update")
    result = service.mine(repo)
    root, renamed = [
        prepare_analysis(service, repo, source_id) for source_id in result["source_ids"]
    ]
    assert root["packet"]["parent_revision"] is None
    assert root["packet"]["files"][0]["before"]["status"] == "absent"
    assert renamed["packet"]["revision"] == final
    assert renamed["packet"]["parent_revision"] == initial
    assert {f["path"] for f in renamed["packet"]["files"]} == {"中文.py", "renamed.py", "test.py"}
    assert submit(service, root, no_pattern_report(root))["data"]["status"] == "no_pattern"
    assert (
        submit(service, renamed, no_pattern_report(renamed), request_id="rename")["data"]["status"]
        == "no_pattern"
    )


def test_discovery_waits_for_real_reports_resumes_and_migrates_old_history(change):
    from evolution_analysis_helpers import complete_fixture_history

    service, repo, _prepared, *_ = change
    # Model an older database with mined sources but no code-analysis jobs.
    with service.store.transaction() as db:
        db.execute("DELETE FROM records WHERE kind='history_analysis'")
    cfg = DiscoveryConfig(repo_path=str(repo), owners={"**": "owner"})
    row = run_discovery(service, cfg, actor="researcher")
    assert row["data"]["status"] == "awaiting_analysis"
    assert "scan" not in row["data"]["results"]
    assert row["data"]["results"]["history_analysis"]["unresolved"] == 2
    complete_fixture_history(service, repo)
    final = run_discovery(service, cfg, actor="researcher", batch_id=row["id"])
    assert final["data"]["status"] == "awaiting_review"
    assert final["data"]["completed_stages"][-2:] == ["history_analysis", "scan"]
