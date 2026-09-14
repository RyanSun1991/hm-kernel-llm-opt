"""Immutable Git history and literal-funnel tests; no models or services needed."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from hmopt.evolution.mining import (
    Hotspot,
    Matcher,
    Pattern,
    mine_git_history,
    read_git,
    scan_candidates,
)

_BUNDLED_GIT = (
    Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe"
)
GIT = (
    os.environ.get("HMOPT_TEST_GIT")
    or shutil.which("git")
    or (str(_BUNDLED_GIT) if _BUNDLED_GIT.is_file() else None)
)
pytestmark = pytest.mark.skipif(GIT is None, reason="Git executable unavailable")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run([GIT, "-C", str(repo), *args], capture_output=True, check=True)
    return result.stdout.decode("utf-8").strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.email", "mining-test@example.invalid")
    git(tmp_path, "config", "user.name", "Mining test")
    git(tmp_path, "config", "core.autocrlf", "false")
    return tmp_path


def commit(repo: Path, message: str) -> str:
    git(repo, "add", "--all")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def active_pattern(**updates) -> Pattern:
    payload = {
        "pattern_id": "curated-repeated-read",
        "title": "Repeated reads",
        "kind": "optimization",
        "problem": "A repeated read may do redundant work.",
        "diagnosis": "Inspect the loop and read lifetime.",
        "remedy": "Cache only where the value is invariant and lifetime rules permit.",
        "matcher": Matcher(
            file_globs=["**/*.c"],
            all_of=["read_value()"],
            any_of=["for (", "while ("],
            none_of=["volatile"],
        ),
        "primary_metric": "instruction_count",
        "preconditions": ["The read is invariant during the loop."],
        "risks": ["A cached value could become stale."],
        "source_ids": ["history-1"],
        "status": "active",
    }
    payload.update(updates)
    return Pattern(**payload)


def scan(repo: Path, patterns: list[Pattern], **kwargs):
    return scan_candidates(repo, patterns, git_bin=GIT, **kwargs)


def history(repo: Path, **kwargs):
    return mine_git_history(repo, git_bin=GIT, **kwargs)


def test_history_pages_oldest_first_without_skipping_overflow(repo: Path):
    revisions = []
    for n in range(5):
        (repo / "loop.c").write_text(f"int loop(void) {{ return {n}; }}\n")
        revisions.append(commit(repo, f"perf: reduce repeated work {n}"))
    first = history(repo, max_commits=2)
    assert [item.revision for item in first] == revisions[:2]
    assert first[0].parent_revision is None
    assert first[1].parent_revision == revisions[0]
    assert not any(item.truncated for item in first)
    second = history(repo, after_revision=first[-1].revision, max_commits=2)
    third = history(repo, after_revision=second[-1].revision, max_commits=2)
    assert [item.revision for item in second + third] == revisions[2:]
    assert history(repo, after_revision=revisions[-1]) == []
    assert history(repo, max_commits=2)[0].source_id == first[0].source_id
    assert history(repo, max_commits=5)[:2] == first


def test_patch_bounds_and_fixed_revision_ignore_worktree(repo: Path):
    (repo / "loop.c").write_text("read_value();\n" * 100)
    before = commit(repo, "Initial code")
    (repo / "loop.c").write_text("cached_value();\n" * 100)
    after = commit(repo, "perf: eliminate redundant reads")
    (repo / "loop.c").write_text("uncommitted contents")
    record = history(repo, after_revision=before, max_patch_bytes=48)[0]
    assert record.revision == after
    assert record.truncated is True
    assert len(record.patch.encode("utf-8")) <= 48
    complete = history(repo, revision=after, after_revision=before)[0]
    assert complete.source_id == record.source_id
    assert complete.paths == ["loop.c"]
    assert "cached_value" in complete.patch and "uncommitted" not in complete.patch


def test_rejects_nonancestor_cursor_and_revision_option_injection(repo: Path):
    (repo / "a.c").write_text("initial\n")
    base = commit(repo, "base")
    git(repo, "checkout", "-b", "other")
    (repo / "a.c").write_text("other\n")
    other = commit(repo, "other")
    git(repo, "checkout", "main")
    (repo / "a.c").write_text("main\n")
    commit(repo, "main")
    with pytest.raises(ValueError):
        history(repo, after_revision=other)
    for bad in ["--help", "HEAD\n--all", "", "-n1"]:
        with pytest.raises(ValueError):
            history(repo, revision=bad)
    assert history(repo, revision=base)[0].revision == base


@pytest.mark.parametrize("message", ["perf: reduce instruction work", "style"])
def test_history_ingestion_requires_analysis_regardless_of_commit_title(repo: Path, message):
    from hmopt.evolution.service import EvolutionService

    (repo / "loop.c").write_text("read_value();\n")
    base = commit(repo, "initial")
    (repo / "loop.c").write_text("cached_value();\n")
    commit(repo, message)
    changes = history(repo, after_revision=base)
    service = EvolutionService(repo / "evolution-state", git_bin=GIT)
    result = service.ingest_history(changes + changes)
    assert result["analysis_required"] is True
    assert result["draft_patterns"] == []
    assert service.store.list("pattern") == []
    [analysis] = service.store.list("history_analysis")
    assert analysis["data"]["status"] == "pending"


def test_correctness_drafts_and_truncated_provenance_stay_workbench(repo: Path):
    pattern = active_pattern(
        kind="anti_pattern",
        matcher=Matcher(file_globs=["**/*.c"], all_of=["read_value()"]),
        preconditions=["TRUNCATED: historical evidence requires additional context."],
    )
    (repo / "x.c").write_text("read_value();\n")
    revision = commit(repo, "target")
    activated = Pattern.model_validate({**pattern.model_dump(), "status": "active"})
    [candidate] = scan(
        repo,
        [activated],
        hotspots=[Hotspot(path="x.c", weight=1.0, revision=revision)],
        owners={"**/*.c": "kernel"},
    )
    assert candidate.lane == "workbench"
    assert any("Unresolved" in reason for reason in candidate.reasons)


def test_scan_matches_committed_bytes_ignores_untracked_dirty_and_disabled(repo: Path):
    original = "void f(void) {\nfor (;;) { read_value(); }\n}\n"
    (repo / "x.c").write_bytes(original.encode("utf-8"))
    revision = commit(repo, "source")
    (repo / "x.c").write_text("volatile read_value();\n")
    (repo / "untracked.c").write_text(original)
    pattern = active_pattern()
    [candidate] = scan(repo, [pattern, pattern])
    assert candidate.path == "x.c"
    assert candidate.repo_revision == revision
    assert candidate.source_sha256 == hashlib.sha256(original.encode()).hexdigest()
    assert candidate.line_start == 2
    assert candidate.candidate_id == scan(repo, [pattern])[0].candidate_id
    assert (
        scan(
            repo,
            [active_pattern(status="draft"), active_pattern(status="retired", pattern_id="other")],
        )
        == []
    )
    with pytest.raises(ValidationError):
        candidate.candidate_id = "changed"


def test_predicates_owner_precedence_hotspot_score_and_top_k(repo: Path):
    (repo / "kernel").mkdir()
    text = "void hot(void) {\nfor (;;) { read_value(); }\n}\n"
    (repo / "kernel/x.c").write_text(text)
    (repo / "a.c").write_text(text)
    (repo / "b.c").write_text(text + "volatile int counter;\n")
    (repo / "c.c").write_text("read_value();\n")
    revision = commit(repo, "source")
    hotspots = [Hotspot(path="kernel/x.c", symbol="read_value", weight=0.9, revision=revision)]
    owners = {"**/*.c": "general", "kernel/*.c": "kernel", "kernel/x.c": "specific"}
    candidates = scan(repo, [active_pattern()], hotspots=hotspots, owners=owners)
    assert [item.path for item in candidates] == ["kernel/x.c", "a.c"]
    first = candidates[0]
    assert first.owner == "specific" and first.lane == "pipeline"
    assert first.score == pytest.approx(sum(first.score_breakdown.values()))
    assert candidates[1].lane == "workbench"
    assert scan(repo, [active_pattern()], hotspots=hotspots, owners=owners, top_k=1) == [first]
    reordered = scan(
        repo, [active_pattern()], hotspots=hotspots, owners=dict(reversed(list(owners.items())))
    )
    assert reordered == candidates


def test_stale_hotspot_evidence_fails_before_scan(repo: Path):
    (repo / "x.c").write_text("for (;;) read_value();\n")
    old = commit(repo, "old")
    (repo / "x.c").write_text("for (;;) read_value(); /* new */\n")
    commit(repo, "new")
    with pytest.raises(ValueError, match="Stale hotspot"):
        scan(repo, [active_pattern()], hotspots=[Hotspot(path="x.c", weight=0.5, revision=old)])
    assert scan(
        repo,
        [active_pattern()],
        revision=old,
        hotspots=[Hotspot(path="x.c", weight=0.5, revision=old)],
    )


def test_scan_file_and_byte_limits_binary_and_glob_anchoring(repo: Path):
    (repo / "a.c").write_text("for (;;) read_value();\n")
    (repo / "b.c").write_text("for (;;) read_value();\n" * 30)
    (repo / "binary.c").write_bytes(b"for (;;) read_value();\x00\xff")
    (repo / "nested").mkdir()
    (repo / "nested/x.c").write_text("for (;;) read_value();\n")
    commit(repo, "source")
    result = scan(repo, [active_pattern()], max_files=1)
    assert len(result) == 1 and result[0].path == "a.c"
    assert any("coverage is incomplete" in reason for reason in result[0].reasons)
    assert {item.path for item in scan(repo, [active_pattern()], max_file_bytes=32)} == {
        "a.c",
        "nested/x.c",
    }
    root_only = active_pattern(matcher=Matcher(file_globs=["*.c"], all_of=["read_value()"]))
    assert all("/" not in item.path for item in scan(repo, [root_only]))


@pytest.mark.parametrize(
    "glob", ["../*.c", "/tmp/*.c", "C:/x.c", "a\\b.c", "a/**b.c", "[ab].c", "a//b.c"]
)
def test_rejects_unsafe_globs(glob: str):
    with pytest.raises(ValidationError):
        Matcher(file_globs=[glob], all_of=["literal"])


def test_strict_models_and_limits(repo: Path):
    (repo / "x.c").write_text("for (;;) read_value();\n")
    revision = commit(repo, "source")
    with pytest.raises(ValidationError):
        active_pattern(version="1")
    for value in [-1.0, 1.1, float("nan")]:
        with pytest.raises(ValidationError):
            Hotspot(path="x.c", weight=value, revision=revision)
    for kwargs in [{"top_k": 0}, {"max_files": True}, {"max_file_bytes": -1}]:
        with pytest.raises(ValueError):
            scan(repo, [active_pattern()], **kwargs)
    with pytest.raises(ValueError):
        history(repo, max_commits=0)
    with pytest.raises(ValueError):
        scan(repo, [active_pattern()], owners={"../*.c": "owner"})
    changed = active_pattern(remedy="A different remedy under the same version.")
    with pytest.raises(ValueError, match="Conflicting"):
        scan(repo, [active_pattern(), changed])


def test_bounded_read_helper_rejects_partial_output_and_mutations(repo: Path):
    (repo / "x.c").write_text("for (;;) read_value();\n" * 40)
    revision = commit(repo, "source")
    assert read_git(repo, ["rev-parse", "HEAD"], git_bin=GIT).decode().strip() == revision
    with pytest.raises(ValueError, match="exceeded max_bytes"):
        read_git(repo, ["show", f"{revision}:x.c"], git_bin=GIT, max_bytes=10)
    for arguments in [
        ["reset", "--hard"],
        ["diff", "--output=escape"],
        ["show", "--textconv", "HEAD:x.c"],
    ]:
        with pytest.raises(ValueError):
            read_git(repo, arguments, git_bin=GIT)
