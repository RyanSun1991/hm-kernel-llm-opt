"""Multi-Git fixtures exercise actual repository boundaries, checkpoints and identity."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path

import pytest
from evolution_analysis_helpers import no_pattern_report
from test_evolution_mining import GIT, commit, git
from test_evolution_service import Scenario

from hmopt.api.evolution_mcp_service import build_evolution_fastmcp_server, load_evolution_config
from hmopt.evolution.change_analysis import (
    HistoryAnalysis,
    analysis_backlog,
    prepare_analysis,
    submit_analysis,
)
from hmopt.evolution.runs import advance_run, control_run, start_run
from hmopt.evolution.scan import scan_next, start_scan
from hmopt.evolution.service import EvolutionService
from hmopt.evolution.store import ConflictError
from hmopt.evolution.workspace import (
    freeze_workspace,
    identities,
    workspace_profiles,
)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "business"
    root.mkdir()
    projects = {}
    for name in ("kernel", "memory", "unselected"):
        repo = root / name
        repo.mkdir()
        git(repo, "init", "-b", "main")
        git(repo, "config", "user.email", "fixture@example.invalid")
        git(repo, "config", "user.name", "Fixture")
        git(repo, "config", "core.autocrlf", "false")
        for n in range(3):
            (repo / "code.c").write_text(
                f"int answer(void) {{ return {n}; }}\n", encoding="utf-8", newline="\n"
            )
            commit(repo, "update")
        projects[name] = {"path": name, "owners": {"**": name + "-owner"}}
    config = {
        "workspace_id": "business",
        "root": str(root),
        "projects": projects,
        "selected": ["kernel", "memory"],
    }
    service = EvolutionService(tmp_path / "state", git_bin=GIT, repo_identities=identities(config))
    return service, config


def complete_pending(service, config):
    for profile in workspace_profiles(config).values():
        for job in analysis_backlog(service, profile["repo_path"], limit=100)["jobs"]:
            prepared = prepare_analysis(service, profile["repo_path"], job["id"])
            submit_analysis(
                service,
                HistoryAnalysis.model_validate(no_pattern_report(prepared)),
                actor="fixture-researcher",
                expected_version=prepared["job"]["version"],
                request_id="fixture:" + job["id"],
            )


def test_workspace_config_selects_exact_roots_and_never_requires_parent_git(workspace, tmp_path):
    service, config = workspace
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root": str(service.store.root),
                "git_bin": GIT,
                "source_workspace": config,
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    options = load_evolution_config(path)
    assert set(options["discovery_profiles"]) == {"kernel", "memory"}
    assert not (Path(config["root"]) / ".git").exists()
    manifest = freeze_workspace(config, git_bin=GIT)
    assert [p["project_id"] for p in manifest["projects"]] == config["selected"]
    assert all(
        p["revision"] == git(Path(p["repo_path"]), "rev-parse", "HEAD")
        for p in manifest["projects"]
    )


@pytest.mark.parametrize("mutation", ["traversal", "absolute", "duplicate", "unknown", "not_root"])
def test_workspace_rejects_invalid_or_unselected_project_scope(workspace, mutation):
    _, original = workspace
    value = deepcopy(original)
    if mutation == "traversal":
        value["projects"]["kernel"]["path"] = "../outside"
    elif mutation == "absolute":
        value["projects"]["kernel"]["path"] = str(Path(value["root"]) / "kernel")
    elif mutation == "duplicate":
        value["selected"] = ["kernel", "kernel"]
    elif mutation == "unknown":
        value["selected"] = ["missing"]
    else:
        (Path(value["root"]) / "kernel/subdirectory").mkdir()
        value["projects"]["kernel"]["path"] = "kernel/subdirectory"
    with pytest.raises(ValueError):
        freeze_workspace(value, git_bin=GIT)


def test_multi_git_run_is_idempotent_frozen_and_resumes_after_manual_analysis(workspace):
    service, config = workspace
    row = start_run(service, config, actor="coordinator", request_id="run")
    assert start_run(service, config, actor="coordinator", request_id="run") == row
    kernel = Path(config["root"]) / "kernel"
    (kernel / "later.c").write_text("int later;\n", encoding="utf-8", newline="\n")
    newer = commit(kernel, "after run starts")
    for _ in range(3):
        advance_run(service, row["id"], page_size=1)
    current = service.store.read("workspace_run", row["id"])
    assert all(p["stage"] == "analysis" for p in current["data"]["projects"].values())
    assert len(service.store.list("history")) == 6
    assert newer not in {r["data"]["revision"] for r in service.store.list("history")}
    assert {r["data"]["repo_id"] for r in service.store.list("history")} == set(
        service.repo_identities.values()
    )
    complete_pending(service, config)
    advance_run(service, row["id"])
    result = advance_run(service, row["id"])
    assert result["run"]["data"]["status"] == "awaiting_review"
    assert len(result["actions"]) == 2
    assert not service.store.list("workspace_claim")
    assert all(p["revision"] != newer for p in result["manifest"]["projects"])


def test_budget_and_one_project_failure_do_not_advance_or_block_other_project(
    workspace, monkeypatch
):
    service, config = workspace
    row = start_run(service, config, actor="operator", request_id="budget", max_commits=1)
    advance_run(service, row["id"], page_size=1)
    result = advance_run(service, row["id"], page_size=1)
    current = result["run"]
    assert all(s["stage"] == "budget_exhausted" for s in current["data"]["projects"].values())
    control_run(
        service,
        row["id"],
        action="retry",
        actor="operator",
        expected_version=current["version"],
        project_id="kernel",
        max_commits=4,
    )
    advance_run(service, row["id"], page_size=1)
    current = service.store.read("workspace_run", row["id"])
    assert current["data"]["projects"]["kernel"]["commits"] == 2
    assert current["data"]["projects"]["memory"]["commits"] == 1
    with pytest.raises(ConflictError):
        start_run(service, config, actor="operator", request_id="competing")


def test_logical_history_identity_is_stable_across_checkouts(workspace, tmp_path):
    service, config = workspace
    source = Path(config["root"]) / "kernel"
    clone_root = tmp_path / "clone-root"
    clone_root.mkdir()
    git(clone_root, "clone", "--quiet", str(source), "kernel")
    clone_config = {**config, "root": str(clone_root), "selected": ["kernel"]}
    clone = EvolutionService(
        tmp_path / "clone-state", git_bin=GIT, repo_identities=identities(clone_config)
    )
    first = service.mine(source)
    second = clone.mine(clone_root / "kernel")
    assert first["source_ids"] == second["source_ids"]
    assert service.repo_identity(source) == clone.repo_identity(clone_root / "kernel")


def test_scan_tree_and_pattern_pages_resume_without_skipping_matches(tmp_path):
    scenario = Scenario(tmp_path, GIT)
    for n in range(19):
        path = scenario.repo / f"dir-{n % 3}" / f"file-{n}.c"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "int call(void) { return redundant_lookup(x); }\n", encoding="utf-8", newline="\n"
        )
    scenario.commit("many scan partitions")
    row = start_scan(
        scenario.service,
        scenario.repo,
        revision="HEAD",
        owners={"**": "owner"},
        request_id="scan",
        actor="researcher",
    )
    (scenario.repo / "late.c").write_text(
        "int late = redundant_lookup(x);\n", encoding="utf-8", newline="\n"
    )
    scenario.commit("excluded after frozen scan")
    for _ in range(40):
        if row["data"]["status"] == "complete":
            break
        row = scan_next(scenario.service, row["id"], expected_version=row["version"], page_size=2)
    assert row["data"]["status"] == "complete"
    assert row["data"]["coverage_complete"]
    ids = [
        c
        for page in scenario.service.store.list("scan_page")
        for c in page["data"]["candidate_ids"]
    ]
    assert len(ids) == len(set(ids)) == 20
    assert all(
        scenario.service.store.read("candidate", c)["data"]["candidate"]["path"] != "late.c"
        for c in ids
    )


def test_unified_workspace_tools_expose_only_selected_projects(workspace):
    service, config = workspace
    server = build_evolution_fastmcp_server(
        service.store.root, git_bin=GIT, source_workspace=config
    )
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"evolution_workspace", "evolution_scan"} <= names
    assert all(t.inputSchema.get("additionalProperties") is False for t in tools)


def test_real_stdio_workspace_run_uses_same_config_and_store(workspace, tmp_path):
    from types import SimpleNamespace

    from test_evolution_interfaces import local_session, tool_value

    service, config = workspace
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root": str(service.store.root),
                "git_bin": GIT,
                "source_workspace": config,
            }
        ),
        encoding="utf-8",
        newline="\n",
    )

    async def exercise():
        async with local_session(
            SimpleNamespace(service=service, git_bin=GIT),
            extra_env={"HMOPT_EVOLUTION_CONFIG": str(path)},
        ) as session:
            started = tool_value(
                await session.call_tool(
                    "evolution_workspace",
                    {
                        "action": "start",
                        "projects": ["kernel"],
                        "request_id": "real-stdio",
                        "actor": "coordinator",
                    },
                )
            )
            run_id = started["id"]
            await session.call_tool("evolution_workspace", {"action": "advance", "run_id": run_id})
            complete_pending(service, config)
            for _ in range(2):
                result = tool_value(
                    await session.call_tool(
                        "evolution_workspace", {"action": "advance", "run_id": run_id}
                    )
                )
            assert result["run"]["data"]["status"] == "awaiting_review"
            assert list(result["run"]["data"]["projects"]) == ["kernel"]
            assert len(service.store.list("history")) == 3

    asyncio.run(exercise())


def test_build_dependencies_are_frozen_but_not_mined(workspace):
    service, config = workspace
    config["dependencies"] = ["unselected"]
    row = start_run(service, config, actor="operator", request_id="deps")
    manifest = service.store.read_evidence(row["data"]["manifest_sha256"])
    assert [p["project_id"] for p in manifest["dependencies"]] == ["unselected"]
    advance_run(service, row["id"])
    assert len(service.store.list("history")) == 6


def test_subset_research_keeps_other_selected_projects_in_build_context(workspace):
    _, config = workspace
    manifest = freeze_workspace(config, git_bin=GIT, selected=["kernel"])
    assert [p["project_id"] for p in manifest["projects"]] == ["kernel"]
    assert [p["project_id"] for p in manifest["dependencies"]] == ["memory"]


def test_workspace_profiles_and_frozen_manifest_preserve_versioned_hotspots(workspace):
    _, config = workspace
    revision = git(Path(config["root"]) / "kernel", "rev-parse", "HEAD")
    hotspot = {"path": "code.c", "symbol": "answer", "revision": revision, "weight": 0.8}
    config["projects"]["kernel"]["hotspots"] = [hotspot]
    profiles = workspace_profiles(config)
    manifest = freeze_workspace(config, git_bin=GIT)
    assert profiles["kernel"]["hotspots"] == [hotspot]
    assert manifest["projects"][0]["hotspots"] == [hotspot]


def test_incremental_run_carries_unresolved_sources_without_reingesting_history(workspace):
    service, config = workspace
    first = start_run(service, config, actor="operator", request_id="first")
    advance_run(service, first["id"])
    complete_pending(service, config)
    advance_run(service, first["id"])
    advance_run(service, first["id"])
    kernel = Path(config["root"]) / "kernel"
    (kernel / "new.c").write_text("int new_value;\n", encoding="utf-8", newline="\n")
    added = commit(kernel, "new change")
    second = start_run(service, config, actor="operator", request_id="second")
    second = advance_run(service, second["id"])["run"]
    assert second["data"]["projects"]["kernel"]["commits"] == 1
    assert second["data"]["projects"]["memory"]["commits"] == 0
    assert second["data"]["projects"]["kernel"]["cursor"] == added
    assert len(service.store.list("history")) == 7
    control_run(
        service, second["id"], action="cancel", actor="operator", expected_version=second["version"]
    )
    third = start_run(service, config, actor="operator", request_id="third")
    result = advance_run(service, third["id"])
    assert all(p["commits"] == 0 for p in result["run"]["data"]["projects"].values())
    assert result["research"]["kernel"]["unresolved"] == 1
    assert result["research"]["memory"]["unresolved"] == 0
    assert (
        advance_run(service, third["id"])["run"]["data"]["projects"]["kernel"]["stage"]
        == "attention"
    )


def test_explicit_full_history_replays_collection_after_completed_run(workspace):
    service, config = workspace
    first = start_run(service, config, actor="operator", request_id="initial")
    current = advance_run(service, first["id"])["run"]
    control_run(
        service,
        current["id"],
        action="cancel",
        actor="operator",
        expected_version=current["version"],
    )
    replay = start_run(service, config, actor="operator", request_id="full", full_history=True)
    result = advance_run(service, replay["id"])
    assert all(p["commits"] == 3 for p in result["run"]["data"]["projects"].values())
    assert len(service.store.list("history")) == 6


def test_partition_scan_records_binary_skips_in_later_pattern_partition(tmp_path):
    from hmopt.evolution.mining import Pattern

    scenario = Scenario(tmp_path, GIT)
    original = scenario.service.store.read("pattern", scenario.pattern_key)["data"]["pattern"]
    for n in range(16):
        pattern = {**original, "pattern_id": f"aaa-pattern-{n:02}"}
        scenario.service.import_pattern(Pattern.model_validate(pattern))
        scenario.service.activate_pattern(
            pattern["pattern_id"] + "@1",
            actor="curator",
            note="Explicit fixture duplicate for partition coverage",
        )
    binary = {
        **original,
        "pattern_id": "zzz-binary",
        "matcher": {"file_globs": ["*.bin"], "all_of": ["needle"]},
    }
    scenario.service.import_pattern(Pattern.model_validate(binary))
    scenario.service.activate_pattern(
        "zzz-binary@1", actor="curator", note="Explicit binary coverage fixture"
    )
    (scenario.repo / "data.bin").write_bytes(b"needle\x00binary")
    scenario.commit("binary in later partition")
    scan = start_scan(
        scenario.service,
        scenario.repo,
        revision="HEAD",
        owners={"**": "owner"},
        request_id="binary",
        actor="researcher",
    )
    for _ in range(20):
        if scan["data"]["status"] != "running":
            break
        scan = scan_next(scenario.service, scan["id"], expected_version=scan["version"])
    assert scan["data"]["status"] == "complete"
    assert not scan["data"]["coverage_complete"]
    assert scan["data"]["skipped"]["skipped_binary_or_encoding"] == 1


def test_actual_cli_setup_start_and_runtime_cycle(workspace, tmp_path):
    import os
    import shutil
    import subprocess
    import sys

    from hmopt.evolution.setup import _ASSETS

    _, config = workspace
    project_root = Path(__file__).resolve().parents[1]
    workbench = tmp_path / "workbench"
    for asset in _ASSETS:
        target = workbench / ".opencode" / asset
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / ".opencode" / asset, target)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("HMOPT_EVOLUTION_")}
    environment["PYTHONPATH"] = str(project_root / "src")

    def cli(*args):
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "hmopt", "evolve", *args],
            env=environment,
            capture_output=True,
            timeout=60,
            check=True,
        )
        return json.loads(completed.stdout.decode("utf-8"))

    result = cli(
        "--git-bin",
        GIT,
        "setup-workspace",
        "--workbench",
        str(workbench),
        "--source-root",
        config["root"],
        "--workspace-id",
        "business",
        "--project",
        "kernel=kernel",
        "--project",
        "memory=memory",
        "--owner",
        "pilot-owner",
    )
    path = Path(result["config_path"])
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["source_workspace"]["selected"] == ["kernel", "memory"]
    assert not Path(loaded["root"]).exists()
    started = cli("--config", str(path), "workspace-start", "--request-id", "cli")
    cycle = cli("--config", str(path), "serve", "--once")
    assert not cycle["errors"]
    assert cycle["runs"][0]["run_id"] == started["id"]
    assert cycle["notifications"] is None
    status = cli("--config", str(path), "workspace-status", started["id"])
    assert all(p["stage"] == "analysis" for p in status["run"]["data"]["projects"].values())
    assert sum(p["commits"] for p in status["run"]["data"]["projects"].values()) == 6
