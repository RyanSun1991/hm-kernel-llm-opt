"""Exercise source, owner, dispatch, report and learning adapters together."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hmopt.evolution.demo_v2 import run_demo_v2
from hmopt.evolution.service import EvolutionService
from hmopt.evolution.store import digest


@pytest.fixture(scope="module")
def git_binary():
    binary = shutil.which("git")
    bundled = (
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe"
    )
    if binary:
        return binary
    if bundled.is_file():
        return str(bundled)
    pytest.skip("Local Git is required for the disposable integration demo")


@pytest.fixture(scope="module")
def demo(tmp_path_factory, git_binary):
    output = tmp_path_factory.mktemp("evolution-v2") / "demo"
    result = run_demo_v2(output, git_bin=git_binary)
    return result, EvolutionService(output / "state", git_bin=git_binary)


def test_v2_source_to_journal_workflow_remains_explicitly_synthetic(demo):
    summary, service = demo
    assert summary["simulation"] is True
    assert summary["hardware_executed"] is False
    assert summary["agent_started"] is False
    assert summary["published"] is summary["merged"] is False
    assert summary["functional_smoke_passed"] is True
    assert summary["owner_review"]["applied"] == 1
    assert summary["owner_review"]["errors"] == 0
    assert summary["source_import"]["imported"] == [summary["source_id"]]
    assert summary["source_distillation"]["draft_patterns"]
    data = service.store.read("candidate", summary["candidate_id"])["data"]
    assert data["stage"] == "code_approved"
    assert data["validation"]["simulation"] is True
    assert data["validation"]["result"]["verdict"] == "pass"
    assert data["validation"]["eligible_for_promotion"] is False
    assert summary["knowledge"]["data"]["tier"] == "journal"
    assert summary["knowledge"]["data"]["eligible_for_promotion"] is False
    assert {
        "dispatch_before_owner_confirmation",
        "synthetic_skill_promotion",
        "synthetic_bundle_export",
    } <= set(summary["blocked_gates"])
    assert not service.store.list("bundle")
    assert not (Path(summary["output"]) / "prohibited-bundle").exists()


def test_v2_dispatches_are_staged_for_all_roles_and_bind_approved_evidence(demo):
    summary, _ = demo
    assert [dispatch["role"] for dispatch in summary["dispatches"]] == [
        "architect",
        "implementer",
        "reviewer",
        "validator",
    ]
    for dispatch in summary["dispatches"]:
        assert dispatch["status"] == "staged"
        assert dispatch["automatic_execution"] is False
        assert Path(dispatch["state_path"]).is_file()
        assert Path(dispatch["prompt_path"]).is_file()
        packet = json.loads(Path(dispatch["handoff_path"]).read_text(encoding="utf-8"))
        assert packet["candidate_id"] == summary["candidate_id"]
        assert packet["role"] == dispatch["role"]
        assert packet["state_version"] == dispatch["candidate_version"]
    assert not (Path(summary["repo"]) / ".opencode/state/current_task.json").exists()


def test_v2_report_manifest_round_trip_retains_raw_pairs_and_frozen_provenance(demo):
    summary, service = demo
    output = Path(summary["output"])
    compare = json.loads((output / "synthetic_ic_compare.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "synthetic_ic_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((output / "synthetic_ab_report.json").read_text(encoding="utf-8"))
    assert manifest["compare_sha256"] == digest(compare)
    evidence = service.store.read_evidence(summary["source_evidence"])
    assert evidence == {"kind": "ic_compare", "compare": compare, "manifest": manifest}
    assert report["baseline"]["hardware"] is report["candidate"]["hardware"] is False
    assert len(report["baseline"]["measurements"]) == 3
    assert [row["metrics"]["instructions"] for row in report["baseline"]["measurements"]] == [
        1000.0
    ] * 3
    assert [row["metrics"]["instructions"] for row in report["candidate"]["measurements"]] == [
        900.0
    ] * 3
    assert report["baseline"]["repo_revision"] != report["candidate"]["repo_revision"]
    assert (
        report["candidate"]["repo_revision"]
        == summary["final_state"]["data"]["implementation"]["revision"]
    )


def test_v2_quality_separates_acceptance_from_precision_and_simulation(demo):
    summary, _ = demo
    quality = summary["quality"]["patterns"][summary["pattern_key"]]
    assert quality["candidates"] == quality["owner_confirmed"] == 1
    assert quality["acceptance_rate"] == 1.0
    assert quality["true_precision"] is None
    assert quality["validated_candidates"] == 0
    assert quality["latest_reports"]["simulation"] == 1
    assert quality["attempts"]["pass"] == 0
    assert quality["attempts"]["simulation"] == 1


def test_v2_implementation_preserves_source_review_and_changes_only_allowed_fixture_file(demo):
    summary, service = demo
    source = service.store.read("source", summary["source_id"])["data"]["record"]
    assert source["related_revision"] is None
    assert source["content"] == (Path(summary["repo"]) / source["source_uri"]).read_bytes().decode(
        "utf-8-sig"
    )
    implementation = summary["final_state"]["data"]["implementation"]
    assert implementation["paths"] == ["target.py"]
    assert summary["final_state"]["data"]["plan"]["allowed_paths"] == ["target.py"]
    assert not service.store.list("history")


def test_v2_demo_never_overwrites_existing_output(demo, git_binary):
    summary, _ = demo
    output = Path(summary["output"])
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_demo_v2(output, git_bin=git_binary)
    assert (output / "summary.json").read_bytes() == before


def test_v2_git_overrides_cannot_redirect_fixture_writes(tmp_path, git_binary, monkeypatch):
    outside = tmp_path / "existing-workspace"
    outside.mkdir()
    sentinel = outside / "untouched.txt"
    sentinel.write_text("existing project", encoding="utf-8")
    monkeypatch.setenv("GIT_DIR", str(outside / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(outside))
    monkeypatch.setenv("GIT_INDEX_FILE", str(outside / "index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(outside / "objects"))
    summary = run_demo_v2(tmp_path / "isolated-demo", git_bin=git_binary)
    assert summary["simulation"] is True
    assert sentinel.read_text(encoding="utf-8") == "existing project"
    assert sorted(path.name for path in outside.iterdir()) == ["untouched.txt"]
