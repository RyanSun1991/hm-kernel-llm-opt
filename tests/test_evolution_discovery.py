import pytest
from test_evolution_service import Scenario
from test_evolution_service import git_bin as shared_git_bin

from hmopt.evolution.discovery import DiscoveryConfig, run_discovery
from hmopt.evolution.mining import Pattern, scan_candidates
from hmopt.evolution.service import EvolutionService
from hmopt.evolution.store import ConflictError

git_bin = shared_git_bin


def config(s, **overrides):
    return DiscoveryConfig(repo_path=str(s.repo), owners={"**/*.c": "owner"}, **overrides)


def test_discovery_checkpoints_pages_and_stops_before_approval(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin)
    for i in range(2):
        (s.repo / "unrelated.txt").write_text(f"version {i}", encoding="utf-8")
        s.commit(f"fixture {i}")
    cfg = config(s, page_size=1, max_pages=1)
    row = run_discovery(s.service, cfg, actor="analyst")
    assert row["data"]["status"] == "partial"
    assert len(row["data"]["history_pages"]) == 1
    restarted = EvolutionService(s.service.store.root, git_bin=git_bin)
    for _ in range(3):
        row = run_discovery(restarted, cfg, actor="analyst", batch_id=row["id"])
    assert row["data"]["status"] == "awaiting_review"
    assert row["data"]["completed_stages"] == ["mine", "sources", "distill", "scan"]
    assert row["data"]["source_changes_allowed"] is False
    assert row["data"]["results"]["scan"]["coverage"]["coverage_complete"]
    assert all(item["data"]["stage"] == "discovered" for item in restarted.store.list("candidate"))
    assert run_discovery(restarted, cfg, actor="analyst", batch_id=row["id"]) == row


def test_batch_rejects_changed_configuration(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin)
    row = run_discovery(s.service, config(s), actor="analyst")
    with pytest.raises(ConflictError):
        run_discovery(s.service, config(s, top_k=1), actor="analyst", batch_id=row["id"])


def test_import_and_distill_workspace_remains_draft(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin)
    directory = tmp_path / "workspace" / ".opencode" / "reviews"
    directory.mkdir(parents=True)
    (directory / "finding.md").write_text(
        "Inspect `redundant_lookup(x)` and preserve lifetime before optimization.", encoding="utf-8"
    )
    cfg = config(s, workspace=str(directory.parents[1]), repo_id="fixture-kernel")
    row = run_discovery(s.service, cfg, actor="analyst")
    assert row["data"]["status"] == "awaiting_review"
    sources = row["data"]["results"]["sources"]["imported"]
    assert len(sources) == 1
    drafts = row["data"]["results"]["distill"]["draft_patterns"]
    assert drafts
    assert all(s.service.store.read("pattern", key)["data"]["status"] == "draft" for key in drafts)


def test_failed_batch_keeps_recoverable_error_without_approval(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin)
    cfg = config(s, revision="missing-ref")
    row = run_discovery(s.service, cfg, actor="analyst")
    assert row["data"]["status"] == "failed"
    assert row["data"]["errors"][0]["stage"] == "resolve_revision"
    assert s.service.store.read("candidate", s.candidate_id)["data"]["stage"] == "discovered"


def test_crashed_running_batch_requires_explicit_recovery(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin)
    cfg = config(s)
    row = run_discovery(s.service, cfg, actor="analyst")
    with s.service.store.transaction() as db:
        data = {**row["data"], "status": "running"}
        row = s.service.store.put(db, "batch", row["id"], data, row["version"])
    with pytest.raises(ConflictError, match="running"):
        run_discovery(s.service, cfg, actor="analyst", batch_id=row["id"])
    recovered = run_discovery(
        s.service, cfg, actor="analyst", batch_id=row["id"], recover_running=True
    )
    assert recovered["data"]["generation"] == 2
    assert recovered["data"]["status"] == "awaiting_review"


def test_incomplete_scan_reports_coverage_even_with_no_matches(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin)
    pattern = Pattern.model_validate(
        {**s.service.store.read("pattern", s.pattern_key)["data"]["pattern"], "status": "active"}
    )
    pattern = pattern.model_copy(
        update={"matcher": pattern.matcher.model_copy(update={"all_of": ["NO_MATCH"]})}
    )
    telemetry = {}
    assert (
        scan_candidates(s.repo, [pattern], git_bin=git_bin, max_files=1, telemetry=telemetry) == []
    )
    assert telemetry["coverage_complete"] is False
    assert telemetry["matches"] == 0


def test_overlay_affects_ranking_absolutely_and_routes_probation(tmp_path, git_bin):
    from hmopt.evolution.learning import set_pattern_overlay
    from hmopt.evolution.mining import Hotspot

    s = Scenario(tmp_path, git_bin)
    original = s.row["data"]["candidate"]["score"]
    set_pattern_overlay(
        s.service,
        s.pattern_key,
        0.6,
        "probation",
        "curator",
        "Review quality in the explicitly bounded probation window.",
        "overlay",
    )
    for _ in range(2):
        row = s.service.scan(
            s.repo,
            owners={"**/*.c": "owner"},
            hotspots=[Hotspot(path=s.path, weight=0.9, revision=s.base)],
        )["candidates"][0]
        assert row["data"]["candidate"]["score"] == pytest.approx(original * 0.6)
        assert row["data"]["candidate"]["lane"] == "workbench"


def source_fixture(tmp_path, count, *, page_size=200):
    """Use real source/store adapters and isolate unrelated Git work."""
    workspace = tmp_path / "workspace"
    reviews = workspace / ".opencode/reviews"
    reviews.mkdir(parents=True)
    for index in range(count):
        (reviews / f"{index:04d}.md").write_text(
            "Valid review evidence without a code matcher.", encoding="utf-8"
        )
    service = EvolutionService(tmp_path / "state")
    service._revision = lambda *args: "1" * 40
    service.mine = lambda *args, **kwargs: {"caught_up": True, "cursor": "1" * 40}
    service.scan = lambda *args, **kwargs: {"candidates": [], "partial": False}
    cfg = DiscoveryConfig(
        repo_path=str(workspace),
        workspace=str(workspace),
        repo_id="fixture",
        owners={},
        max_pages=1,
        source_page_size=page_size,
    )
    return service, cfg, reviews


def test_201_source_documents_resume_without_reimporting_first_page(tmp_path):
    service, cfg, _ = source_fixture(tmp_path, 201)
    first = run_discovery(service, cfg, actor="reader")
    assert first["data"]["status"] == "partial"
    assert len(first["data"]["results"]["sources"]["source_ids"]) == 200
    assert len(service.store.list("source", limit=1000)) == 200
    second = run_discovery(service, cfg, actor="reader", batch_id=first["id"])
    assert len(second["data"]["results"]["sources"]["source_ids"]) == 201
    assert len(second["data"]["source_pages"]) == 2
    assert len(second["data"]["source_pages"][1]["imported"]) == 1
    assert not second["data"]["source_pages"][1]["existing"]
    assert second["data"]["distill_page"] == 1
    final = run_discovery(service, cfg, actor="reader", batch_id=first["id"])
    assert final["data"]["status"] == "awaiting_review"
    assert len(final["data"]["results"]["distill"]["processed"]) == 201
    assert final["data"]["completed_stages"] == ["mine", "sources", "distill", "scan"]


def test_more_than_1000_source_ids_distill_using_original_page_boundaries(tmp_path):
    service, cfg, _ = source_fixture(tmp_path, 1001, page_size=500)
    row = run_discovery(service, cfg, actor="reader")
    for _ in range(6):
        if row["data"]["status"] == "awaiting_review":
            break
        row = run_discovery(service, cfg, actor="reader", batch_id=row["id"])
    assert row["data"]["status"] == "awaiting_review"
    assert len(row["data"]["results"]["sources"]["source_ids"]) == 1001
    assert len(row["data"]["results"]["distill"]["processed"]) == 1001
    assert row["data"]["distill_page"] == 3


def test_changed_source_page_stops_with_actionable_terminal_state(tmp_path):
    service, cfg, reviews = source_fixture(tmp_path, 3, page_size=2)
    first = run_discovery(service, cfg, actor="reader")
    (reviews / "0002.md").write_text("Changed source", encoding="utf-8")
    second = run_discovery(service, cfg, actor="reader", batch_id=first["id"])
    assert second["data"]["status"] == "requires_attention"
    assert second["data"]["errors"]
    assert "scan" not in second["data"]["completed_stages"]
    assert run_discovery(service, cfg, actor="reader", batch_id=first["id"]) == second
    assert len(service.store.list("source")) == 2


def test_source_snapshot_bound_requires_partition_without_no_progress_loop(tmp_path, monkeypatch):
    from hmopt.evolution import sources

    service, cfg, _ = source_fixture(tmp_path, 3, page_size=1)
    monkeypatch.setattr(sources, "_MAX_MANIFEST_FILES", 2)
    row = run_discovery(service, cfg, actor="reader")
    assert row["data"]["status"] == "requires_partition"
    assert "partition" in row["data"]["next_action"].lower()
    assert run_discovery(service, cfg, actor="reader", batch_id=row["id"]) == row


def test_partial_scan_is_not_retried_forever_without_a_cursor(tmp_path):
    service, cfg, _ = source_fixture(tmp_path, 0)
    service.scan = lambda *args, **kwargs: {"candidates": [], "partial": True}
    row = run_discovery(service, cfg, actor="reader")
    assert row["data"]["status"] == "requires_partition"
    assert run_discovery(service, cfg, actor="reader", batch_id=row["id"]) == row
