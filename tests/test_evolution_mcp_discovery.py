"""Exercise configured discovery tools over real local MCP stdio, without external calls."""

from __future__ import annotations

import asyncio
import json

import pytest
from evolution_analysis_helpers import no_pattern_report
from test_evolution_interfaces import local_session, tool_error, tool_value
from test_evolution_service import Scenario
from test_evolution_service import git_bin as shared_git_bin

from hmopt.evolution.mcp import build_server
from hmopt.evolution.sources import ingest_records
from hmopt.evolution.store import digest

git_bin = shared_git_bin


@pytest.fixture
def configured(tmp_path, git_bin):
    scenario = Scenario(tmp_path, git_bin)
    reviews = tmp_path / "workspace" / ".opencode" / "reviews"
    reviews.mkdir(parents=True)
    for index in range(2):
        (reviews / f"{index}.md").write_text(
            f"Optimization review {index}: repeated `redundant_lookup(x)` requires lifetime checks.",
            encoding="utf-8",
        )
    profile = {
        "repo_path": str(scenario.repo),
        "repo_id": "fixture-kernel",
        "workspace": str(reviews.parents[1]),
        "owners": {"**/*.c": "owner"},
        "page_size": 1,
        "max_pages": 1,
        "source_page_size": 1,
    }
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"kernel": profile}), encoding="utf-8")
    return scenario, profile, profiles


def test_stdio_discovery_resumes_pages_reads_batches_and_preserves_owner_gates(configured):
    scenario, _, profiles = configured
    for index in range(2):
        (scenario.repo / "unrelated.txt").write_text(f"revision {index}", encoding="utf-8")
        scenario.commit(f"discovery fixture {index}")

    async def exercise():
        async with local_session(scenario, ["--discovery-profiles", str(profiles)]) as session:
            available = tool_value(await session.call_tool("evolution_discovery_profiles", {}))
            assert [item["name"] for item in available["profiles"]] == ["kernel"]
            registered = available["profiles"][0]
            assert registered["config_sha256"] == digest(registered["config"])
            assert available["operator_actions"]

            async def run(batch_id=None):
                arguments = {"profile": "kernel", "actor": "researcher"}
                if batch_id is not None:
                    arguments["batch_id"] = batch_id
                return tool_value(await session.call_tool("evolution_run_discovery", arguments))

            first = await run()
            assert first["data"]["status"] == "partial"
            assert len(first["data"]["history_pages"]) == 1
            assert first["data"]["config_digest"] == registered["config_sha256"]
            # The running server must retain its operator-approved startup snapshot.
            profiles.write_text("{}", encoding="utf-8")
            assert (
                tool_value(await session.call_tool("evolution_discovery_profiles", {})) == available
            )
            row = first
            for _ in range(8):
                if row["data"]["status"] == "awaiting_review":
                    break
                if row["data"]["status"] == "awaiting_analysis":
                    queue = tool_value(
                        await session.call_tool("evolution_history_analysis", {"profile": "kernel"})
                    )
                    for job in queue["jobs"]:
                        prepared = tool_value(
                            await session.call_tool(
                                "evolution_history_analysis",
                                {"profile": "kernel", "source_id": job["id"]},
                            )
                        )
                        result = tool_value(
                            await session.call_tool(
                                "evolution_submit_history_analysis",
                                {
                                    "profile": "kernel",
                                    "analysis": no_pattern_report(prepared),
                                    "actor": "fixture-researcher",
                                    "expected_version": prepared["job"]["version"],
                                    "request_id": "fixture-analysis:" + job["id"],
                                },
                            )
                        )
                        assert result["data"]["status"] == "no_pattern"
                row = await run(row["id"])
            assert row["data"]["status"] == "awaiting_review"
            assert row["data"]["completed_stages"] == [
                "mine",
                "sources",
                "distill",
                "history_analysis",
                "scan",
            ]
            assert row["data"]["source_changes_allowed"] is False
            assert len(row["data"]["history_pages"]) == 3
            assert len(row["data"]["source_pages"]) == 2
            assert len(row["data"]["results"]["distill"]["processed"]) == 2
            assert not row["data"]["errors"]
            assert await run(row["id"]) == row
            assert (
                tool_value(
                    await session.call_tool(
                        "evolution_read", {"kind": "batch", "record_id": row["id"]}
                    )
                )
                == row
            )
            candidates = tool_value(
                await session.call_tool("evolution_list", {"kind": "candidate"})
            )
            assert candidates and all(item["data"]["stage"] == "discovered" for item in candidates)
            for key in row["data"]["results"]["distill"]["draft_patterns"]:
                pattern = tool_value(
                    await session.call_tool("evolution_read", {"kind": "pattern", "record_id": key})
                )
                assert pattern["data"]["status"] == "draft"
            tool_error(
                await session.call_tool(
                    "evolution_dispatch",
                    {
                        "candidate_id": scenario.candidate_id,
                        "actor": "agent",
                        "request_id": "unconfirmed",
                    },
                ),
                "No executable handoff",
            )

    asyncio.run(exercise())


def test_stdio_discovery_steps_are_independent_and_distill_only_selected_sources(configured):
    scenario, _, profiles = configured

    async def exercise():
        async with local_session(scenario, ["--discovery-profiles", str(profiles)]) as session:

            async def step(name, **arguments):
                return tool_value(
                    await session.call_tool(
                        "evolution_discovery_step",
                        {"profile": "kernel", "step": name, "actor": "researcher", **arguments},
                    )
                )

            page = await step("sources")
            assert page["has_more"] is True
            assert len(page["imported"]) == 1
            following = await step("sources", cursor=page["next_cursor"])
            assert following["has_more"] is False
            assert len(following["imported"]) == 1
            assert set(page["imported"]).isdisjoint(following["imported"])
            distilled = await step("distill", source_ids=page["imported"])
            assert distilled["processed"] == page["imported"]
            assert distilled["draft_patterns"]
            markers = scenario.service.store.list("source_distillation")
            assert [row["data"]["source_id"] for row in markers] == page["imported"]
            mined = await step("mine")
            assert mined["caught_up"] is True
            scanned = await step("scan")
            assert scanned["candidates"]
            assert not scenario.service.store.list("batch")
            assert all(row["data"]["stage"] == "discovered" for row in scanned["candidates"])
            # Default read retains the same candidate contract as evolution_show.
            assert tool_value(
                await session.call_tool("evolution_read", {"record_id": scenario.candidate_id})
            ) == scenario.service.store.read("candidate", scenario.candidate_id)

    asyncio.run(exercise())


def test_stdio_discovery_rejects_missing_profiles_and_unconfigured_sources(configured):
    scenario, profile, profiles = configured
    no_workspace = {key: value for key, value in profile.items() if key != "workspace"}
    profiles.write_text(json.dumps({"kernel": no_workspace}), encoding="utf-8")

    async def exercise():
        async with local_session(scenario) as session:
            assert (
                tool_value(await session.call_tool("evolution_discovery_profiles", {}))["profiles"]
                == []
            )
            tool_error(
                await session.call_tool(
                    "evolution_run_discovery", {"profile": "kernel", "actor": "reader"}
                ),
                "profile",
            )
        async with local_session(scenario, ["--discovery-profiles", str(profiles)]) as session:
            for name in ("unknown", "../kernel"):
                tool_error(
                    await session.call_tool(
                        "evolution_run_discovery", {"profile": name, "actor": "reader"}
                    ),
                    "profile",
                )
            tool_error(
                await session.call_tool(
                    "evolution_discovery_step",
                    {"profile": "kernel", "step": "sources", "actor": "reader"},
                ),
                "workspace",
            )
        assert not scenario.service.store.list("batch")
        assert not scenario.service.store.list("source")

    asyncio.run(exercise())


def test_stdio_discovery_rejects_cross_repo_sources_and_wrong_step_arguments(configured):
    scenario, _, profiles = configured
    initial_patterns = scenario.service.store.list("pattern")
    ingested = ingest_records(
        scenario.service,
        [
            {
                "repo_id": "different-kernel",
                "source_kind": "review",
                "source_uri": ".opencode/reviews/other.md",
                "content": "Repeated `redundant_lookup(x)` optimization evidence.",
            }
        ],
        actor="operator",
    )

    async def exercise():
        async with local_session(scenario, ["--discovery-profiles", str(profiles)]) as session:
            cases = [
                ({"step": "distill"}, "source_ids"),
                ({"step": "distill", "source_ids": []}, "source_ids"),
                ({"step": "distill", "source_ids": ingested["imported"] * 2}, "source"),
                ({"step": "distill", "source_ids": ingested["imported"]}, "repo"),
                ({"step": "scan", "cursor": "cursor-not-allowed"}, "cursor"),
                ({"step": "mine", "source_ids": ingested["imported"]}, "source_ids"),
                ({"step": "implement"}, "Input"),
            ]
            for arguments, message in cases:
                tool_error(
                    await session.call_tool(
                        "evolution_discovery_step",
                        {"profile": "kernel", "actor": "reader", **arguments},
                    ),
                    message,
                )
            assert scenario.service.store.list("pattern") == initial_patterns
            assert not scenario.service.store.list("batch")

    asyncio.run(exercise())


def test_stdio_discovery_cannot_override_paths_recover_workers_or_change_batch(configured):
    scenario, profile, profiles = configured
    profiles.write_text(
        json.dumps({"kernel": profile, "changed": {**profile, "top_k": 1}}), encoding="utf-8"
    )

    async def exercise():
        async with local_session(scenario, ["--discovery-profiles", str(profiles)]) as session:
            schemas = {tool.name: tool.inputSchema for tool in (await session.list_tools()).tools}
            for name in ("evolution_run_discovery", "evolution_discovery_step"):
                assert not {"repo_path", "workspace", "recover_running", "config"} & set(
                    schemas[name]["properties"]
                )
                for field, value in (("repo_path", str(scenario.repo)), ("recover_running", True)):
                    arguments = {"profile": "kernel", "actor": "researcher", field: value}
                    if name.endswith("step"):
                        arguments["step"] = "mine"
                    result = await session.call_tool(name, arguments)
                    assert result.isError, f"{name} accepted the operator-only field {field}"
            assert not scenario.service.store.list("batch")
            row = tool_value(
                await session.call_tool(
                    "evolution_run_discovery", {"profile": "kernel", "actor": "researcher"}
                )
            )
            tool_error(
                await session.call_tool(
                    "evolution_run_discovery",
                    {"profile": "changed", "actor": "researcher", "batch_id": row["id"]},
                ),
                "configuration changed",
            )
            with scenario.service.store.transaction() as db:
                scenario.service.store.put(
                    db, "batch", row["id"], {**row["data"], "status": "running"}, row["version"]
                )
            tool_error(
                await session.call_tool(
                    "evolution_run_discovery",
                    {"profile": "kernel", "actor": "researcher", "batch_id": row["id"]},
                ),
                "running",
            )
            assert (
                scenario.service.store.read("batch", row["id"])["data"]["generation"]
                == row["data"]["generation"]
            )

    asyncio.run(exercise())


def test_stdio_discovery_env_configuration_rejects_non_repository_roots(configured):
    scenario, profile, profiles = configured
    nested = scenario.repo / "nested"
    nested.mkdir()
    profiles.write_text(
        json.dumps({"kernel": {**profile, "repo_path": str(nested)}}), encoding="utf-8"
    )

    async def exercise():
        async with local_session(
            scenario, extra_env={"HMOPT_EVOLUTION_DISCOVERY_PROFILES": str(profiles)}
        ) as session:
            registered = tool_value(await session.call_tool("evolution_discovery_profiles", {}))
            assert registered["profiles"][0]["config"]["repo_path"] == str(nested.resolve())
            tool_error(
                await session.call_tool(
                    "evolution_run_discovery", {"profile": "kernel", "actor": "researcher"}
                ),
                "repository root",
            )
            assert not scenario.service.store.list("batch")

    asyncio.run(exercise())


@pytest.mark.parametrize("alias", ["", "../kernel", "has space", "a" * 65, "_kernel"])
def test_server_rejects_invalid_discovery_profile_aliases(configured, alias):
    scenario, profile, _ = configured
    with pytest.raises(ValueError, match="[Pp]rofile|alias"):
        build_server(scenario.service.store.root, discovery_profiles={alias: profile})


@pytest.mark.parametrize("field", ["repo_path", "workspace"])
@pytest.mark.parametrize("value", ["relative-directory", "absent", "file"])
def test_server_rejects_unbounded_discovery_profile_paths(configured, tmp_path, field, value):
    scenario, profile, profiles = configured
    path = {
        "relative-directory": value,
        "absent": str(tmp_path / "missing"),
        "file": str(profiles),
    }[value]
    with pytest.raises(ValueError):
        build_server(
            scenario.service.store.root, discovery_profiles={"kernel": {**profile, field: path}}
        )


def test_server_limits_registered_discovery_profiles(configured):
    scenario, profile, _ = configured
    with pytest.raises(ValueError, match="32"):
        build_server(
            scenario.service.store.root,
            discovery_profiles={f"kernel-{index}": profile for index in range(33)},
        )
