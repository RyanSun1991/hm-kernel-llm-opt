"""Real local Git/SQLite workflows; supplied research responses are fixtures, not model evaluation."""

import shutil
from pathlib import Path

import pytest
from test_evolution_service import Scenario
from test_evolution_service import git_bin as shared_git_bin

from hmopt.evolution.batch import block_candidate, create_batch, next_candidate
from hmopt.evolution.catalog import candidates, dossier
from hmopt.evolution.methods import code_context, skill_snapshot
from hmopt.evolution.research import prepare_research, submit_research
from hmopt.evolution.store import ConflictError

git_bin = shared_git_bin
PROJECT = Path(__file__).resolve().parents[1]


def configure_methods(service, tmp_path):
    skills = tmp_path / "workbench/.opencode/skills"
    shutil.copytree(PROJECT / ".opencode/skills", skills)
    service.workspace_root = skills.parent / "local/workspaces"


@pytest.fixture
def scenario(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin)
    configure_methods(s.service, tmp_path)
    return s


def require_assessment(s):
    with s.service.store.transaction() as db:
        row = s.service.store.get(db, "pattern", s.pattern_key)
        s.service.store.put(
            db,
            "pattern",
            s.pattern_key,
            {**row["data"], "requires_assessment": True},
            row["version"],
        )


def assess(s, result="met"):
    prepared = prepare_research(s.service, s.repo, "assess", [s.candidate_id], actor="researcher")
    context = code_context(s.service, s.repo, s.base, s.path)
    citation = {"context_sha256": context["sha256"], "line_start": 1, "quote": context["lines"][0]}
    report = {
        "result": "applicable" if result == "met" else "needs_context",
        "summary": "Fixture assessment of lookup applicability with explicit source evidence.",
        "checks": [
            {
                "criterion_id": criterion["id"],
                "result": result,
                "explanation": "Fixture: source supports the stated criterion for this path.",
                "citations": [citation] if result == "met" else [],
            }
            for criterion in prepared["research"]["data"]["inputs"]["criteria"]
        ],
    }
    return prepared, report


def submit(s, prepared, report, request_id="research-response"):
    return submit_research(
        s.service,
        prepared["research"]["id"],
        report,
        actor="researcher",
        expected_version=prepared["research"]["version"],
        request_id=request_id,
    )


def test_assessment_gate_receipt_dossier_and_replay(scenario):
    s = scenario
    require_assessment(s)
    with pytest.raises(ValueError, match="applicability"):
        s.confirm()
    prepared, report = assess(s)
    first = submit(s, prepared, report)
    assert first == submit(s, prepared, report)
    s.row = s.service.store.read("candidate", s.candidate_id)
    s.confirm()
    archive = dossier(s.service, s.candidate_id)
    receipt = archive["approvals"][0]
    assert receipt["id"] == s.row["data"]["approval_id"]
    assert receipt["data"]["decision"] == "confirm"
    assert receipt["data"]["before_version"] == 2
    s.approve_plan()
    assert dossier(s.service, s.candidate_id)["approvals"] == [receipt]
    queue = candidates(s.service, s.repo, owner="owner")
    assert queue["items"][0]["next_role"] == "implementer"
    assert candidates(s.service, s.repo, owner="someone-else")["items"] == []


@pytest.mark.parametrize("fault", ["missing", "quote", "location", "unknown", "stale"])
def test_assessment_cannot_promote_invalid_or_uncertain_evidence(scenario, fault):
    s = scenario
    require_assessment(s)
    prepared, report = assess(s, "unknown" if fault == "unknown" else "met")
    if fault == "missing":
        report["checks"].pop()
    elif fault == "quote":
        report["checks"][0]["citations"][0]["quote"] = "invented source code"
    elif fault == "location":
        report["checks"][0]["citations"][0]["line_start"] = 2
    elif fault == "stale":
        with s.service.store.transaction() as db:
            s.service.store.put(db, "candidate", s.candidate_id, s.row["data"], s.row["version"])
    if fault == "unknown":
        submit(s, prepared, report)
        s.row = s.service.store.read("candidate", s.candidate_id)
        with pytest.raises(ValueError, match="applicability"):
            s.confirm()
    else:
        with pytest.raises(ValueError):
            submit(s, prepared, report)
    assert not s.service.store.list("approval")


def test_method_snapshot_retains_original_content_and_context_ignores_dirty_files(scenario):
    s = scenario
    first = skill_snapshot(s.service, ["candidate-assessment"])
    path = Path(first["skills"][0]["path"])
    path.write_text(
        path.read_text(encoding="utf-8") + "\nNew method version.\n", encoding="utf-8", newline="\n"
    )
    second = skill_snapshot(s.service, ["candidate-assessment"])
    assert first["sha256"] != second["sha256"]
    assert s.service.store.read_evidence(first["sha256"])["skills"] == first["skills"]
    (s.repo / s.path).write_text("dirty replacement\n", encoding="utf-8")
    context = code_context(s.service, s.repo, s.base, s.path)
    assert "redundant_lookup" in context["lines"][0]
    for path in ("../escape", ".git/config"):
        with pytest.raises(ValueError):
            code_context(s.service, s.repo, s.base, path)


def test_legacy_approval_rebuilt_without_changing_candidate(scenario):
    s = scenario
    s.confirm()
    with s.service.store.transaction() as db:
        db.execute("DELETE FROM records WHERE kind='approval'")
    archive = dossier(s.service, s.candidate_id)
    assert archive["candidate"] == s.row
    assert archive["approvals"][0]["data"]["identity_assurance"].startswith("legacy_")


def make_batch(s, ids=None, scope="full", request_id="batch-one"):
    return create_batch(
        s.service,
        s.repo,
        ids or [s.candidate_id],
        scope=scope,
        actor="coordinator",
        request_id=request_id,
    )


def test_batch_claim_isolation_resume_and_explicit_stop(scenario):
    s = scenario
    s.confirm()
    batch = make_batch(s, scope="stage")
    assert make_batch(s, scope="stage") == batch
    with pytest.raises(ConflictError, match="reserved"):
        make_batch(s, request_id="different-batch")
    first = next_candidate(s.service, batch["id"], worker_id="session-one")
    worktree = Path(first["handoff"]["repo_path"])
    assert worktree != s.repo and (worktree / s.path).is_file()
    assert (
        next_candidate(s.service, batch["id"], worker_id="session-one")["handoff"]
        == first["handoff"]
    )
    with pytest.raises(ConflictError, match="running worker"):
        next_candidate(s.service, batch["id"], worker_id="another-session")
    current = s.service.store.read("execution_batch", batch["id"])
    result = block_candidate(
        s.service,
        batch["id"],
        worker_id="session-one",
        expected_version=current["version"],
        reason="Child session has stopped; missing build environment.",
    )
    assert result["data"]["status"] == "completed_with_blocks"
    assert (worktree / s.path).read_bytes() == (s.repo / s.path).read_bytes()
    # Explicit new selection retains the same worktree and never drops local work.
    retry = make_batch(s, scope="stage", request_id="retry-batch")
    assert next_candidate(s.service, retry["id"], worker_id="session-two")["handoff"][
        "repo_path"
    ] == str(worktree)


def test_two_candidates_from_same_baseline_implement_in_distinct_worktrees(scenario):
    s = scenario
    (s.repo / "module_second.c").write_text(
        "int second(int x) { return redundant_lookup(x); }\n", encoding="utf-8"
    )
    s.base = s.commit("add second optimization target")
    rows = s.service.scan(s.repo, owners={"**/*.c": "owner"})["candidates"]
    assert len(rows) == 2
    ids = []
    for row in rows:
        s.row, s.candidate_id = row, row["id"]
        s.confirm()
        ids.append(s.candidate_id)
    original_repo = s.repo
    batch = make_batch(s, ids, scope="full")
    worktrees = []
    for candidate_id in ids:
        task = next_candidate(s.service, batch["id"], worker_id="coordinator-session")
        assert task["item"]["candidate_id"] == candidate_id
        s.row = s.service.store.read("candidate", candidate_id)
        s.candidate_id, s.path = candidate_id, s.row["data"]["candidate"]["path"]
        s.approve_plan()
        worktree = Path(task["handoff"]["repo_path"])
        worktrees.append(worktree)
        s.repo = worktree
        s.implement()
        s.approve_code()
        # Fixture evidence verifies gate integration only, not actual hardware performance.
        from hmopt.evolution.validation import ABReport

        s.row = s.service.validate(
            candidate_id,
            ABReport.model_validate(s.report_data()),
            actor="validator",
            expected_version=s.row["version"],
            request_id="validation-" + candidate_id,
        )
        assert s.row["data"]["stage"] == "validated"
        s.repo = original_repo
        assert s.git("rev-parse", "HEAD").strip() == s.base
        assert s.git("status", "--porcelain").strip() == ""
    finished = next_candidate(s.service, batch["id"], worker_id="coordinator-session")
    assert finished["batch"]["data"]["status"] == "completed"
    assert len(set(worktrees)) == 2
    for tree in worktrees:
        assert sum("cached_lookup" in p.read_text() for p in tree.glob("*.c")) == 1


def test_batch_stale_selection_and_named_scope_cannot_skip_gates(scenario):
    s = scenario
    with pytest.raises(ValueError, match="approved"):
        make_batch(s)
    s.confirm()
    with pytest.raises(ValueError, match="prerequisite"):
        make_batch(s, scope="validate")
    batch = make_batch(s)
    s.transition(
        "reject", "owner", {"note": "Owner withdrew this selected candidate before execution."}
    )
    result = next_candidate(s.service, batch["id"], worker_id="session")
    assert result["item"] is None
    assert result["batch"]["data"]["items"][0]["status"] == "blocked"
    assert not (s.service.store.root / "worktrees").exists()


def test_selected_stage_is_a_service_gate_and_completion_is_stable(scenario):
    from hmopt.evolution.workflow import dispatch_candidate

    s = scenario
    s.confirm()
    batch = make_batch(s, scope="plan")
    with pytest.raises(ValueError, match="not been claimed"):
        s.approve_plan()
    next_candidate(s.service, batch["id"], worker_id="worker")
    s.row = s.service.store.read("candidate", s.candidate_id)
    s.approve_plan()
    with pytest.raises(ValueError, match="batch scope"):
        s.transition("record_implementation", "implementer", {"revision": s.base})
    with pytest.raises(ValueError, match="batch scope"):
        dispatch_candidate(
            s.service,
            s.candidate_id,
            s.service.store.root / "dispatches",
            actor="coordinator",
            request_id="overscope",
            batch_id=batch["id"],
            worker_id="worker",
        )
    finished = next_candidate(s.service, batch["id"], worker_id="worker")
    assert finished["batch"]["data"]["status"] == "completed"
    assert next_candidate(s.service, batch["id"], worker_id="worker") == finished


def test_concurrent_batch_creation_reserves_candidate_once(scenario):
    from concurrent.futures import ThreadPoolExecutor

    s = scenario
    s.confirm()

    def create(request_id):
        try:
            return make_batch(s, request_id=request_id)["id"]
        except ConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, ["concurrent-1", "concurrent-2"]))
    assert results.count("conflict") == 1
    assert len(s.service.store.list("execution_batch")) == 1


def test_existing_standalone_dispatch_cannot_be_taken_over(scenario):
    from hmopt.evolution.workflow import dispatch_candidate

    s = scenario
    s.confirm()
    dispatch_candidate(
        s.service,
        s.candidate_id,
        s.service.store.root / "dispatches",
        actor="coordinator",
        request_id="standalone",
    )
    with pytest.raises(ConflictError, match="current dispatch"):
        make_batch(s)


async def exercise_skill_mcp(session, s):
    """Shared real-transport exercise; owner decision stays outside the model MCP."""
    from test_evolution_interfaces import tool_value

    async def call(name, **arguments):
        return tool_value(await session.call_tool(name, arguments))

    require_assessment(s)
    prepared = await call(
        "evolution_prepare_research",
        profile="kernel",
        method="assess",
        subject_ids=[s.candidate_id],
        actor="researcher",
    )
    context = await call("evolution_code_context", profile="kernel", revision=s.base, path=s.path)
    report = {
        "result": "applicable",
        "summary": "Explicit fixture assessment through the actual MCP transport.",
        "checks": [
            {
                "criterion_id": c["id"],
                "result": "met",
                "explanation": "Fixture evidence for this applicability criterion.",
                "citations": [
                    {
                        "context_sha256": context["sha256"],
                        "line_start": 1,
                        "quote": context["lines"][0],
                    }
                ],
            }
            for c in prepared["research"]["data"]["inputs"]["criteria"]
        ],
    }
    await call(
        "evolution_submit_research",
        research_id=prepared["research"]["id"],
        payload=report,
        actor="researcher",
        expected_version=prepared["research"]["version"],
        request_id="mcp-assessment",
    )
    s.row = s.service.store.read("candidate", s.candidate_id)
    s.confirm()
    queue = await call("evolution_candidates", profile="kernel")
    assert queue["items"][0]["approval_ids"] == [s.row["data"]["approval_id"]]
    archive = await call("evolution_dossier", candidate_id=s.candidate_id)
    assert archive["approvals"][0]["data"]["context_sha256"]
    batch = await call(
        "evolution_create_batch",
        profile="kernel",
        candidate_ids=[s.candidate_id],
        scope="plan",
        actor="coordinator",
        request_id="mcp-batch",
    )
    item = await call("evolution_batch_next", batch_id=batch["id"], worker_id="mcp-worker")
    assert item["handoff"]["repo_path"] != str(s.repo)
    assert item["handoff"]["method_sha256"] == batch["data"]["method_sha256"]
    from test_evolution_interfaces import tool_error

    tool_error(
        await session.call_tool(
            "evolution_dispatch",
            {
                "candidate_id": s.candidate_id,
                "actor": "coordinator",
                "request_id": "duplicate-standalone",
            },
        ),
        "claimed by a batch",
    )
    staged = await call(
        "evolution_dispatch",
        candidate_id=s.candidate_id,
        actor="coordinator",
        request_id="batch-dispatch",
        batch_id=batch["id"],
        worker_id="mcp-worker",
    )
    assert staged["dispatch_id"]
    stopped = await call(
        "evolution_batch_block",
        batch_id=batch["id"],
        worker_id="mcp-worker",
        reason="Fixture child has stopped; no source execution was requested.",
        expected_version=item["batch"]["version"],
    )
    assert stopped["data"]["status"] == "completed_with_blocks"
    tool_error(
        await session.call_tool(
            "evolution_materialize_workspace", {"dispatch_id": staged["dispatch_id"]}
        ),
        "stale",
    )
