"""Unified API MCP integration over real stdio, without indexes, models or hardware."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from test_evolution_interfaces import tool_error, tool_value
from test_evolution_service import Scenario
from test_evolution_service import git_bin as shared_git_bin
from test_evolution_setup import locations as setup_locations

from hmopt.evolution.setup import setup
from hmopt.evolution.store import digest

git_bin = shared_git_bin
locations = setup_locations
ROOT = Path(__file__).resolve().parents[1]
INDEX_TOOLS = {
    "kernel_index_code",
    "kernel_symbol_graph",
    "kernel_hotspot_context",
    "kernel_call_chain",
    "kernel_get_snippets",
}


def test_skill_research_approvals_and_batch_through_real_stdio(tmp_path, git_bin):
    from test_evolution_skill_workflow import configure_methods, exercise_skill_mcp

    scenario = Scenario(tmp_path, git_bin)
    configure_methods(scenario.service, tmp_path)
    profiles = tmp_path / "skill-profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "kernel": {
                    "repo_path": str(scenario.repo),
                    "repo_id": "fixture-kernel",
                    "owners": {"**/*.c": "owner"},
                }
            }
        ),
        encoding="utf-8",
    )

    async def exercise():
        async with session_for(
            scenario.service.store.root,
            git_bin,
            profiles=profiles,
            workspace=scenario.service.workspace_root,
        ) as session:
            await exercise_skill_mcp(session, scenario)

    asyncio.run(exercise())


EVOLUTION_TOOLS = {
    "evolution_workspace",
    "evolution_scan",
    "evolution_experiment",
    "evolution_list",
    "evolution_show",
    "evolution_handoff",
    "evolution_submit",
    "evolution_validate",
    "evolution_recall",
    "evolution_audit",
    "evolution_evidence",
    "evolution_capture",
    "evolution_quality",
    "evolution_convert_ic",
    "evolution_dispatch",
    "evolution_digest",
    "evolution_materialize_workspace",
    "evolution_convert_lmbench",
    "evolution_discovery_profiles",
    "evolution_run_discovery",
    "evolution_discovery_step",
    "evolution_code_context",
    "evolution_prepare_research",
    "evolution_submit_research",
    "evolution_candidates",
    "evolution_dossier",
    "evolution_create_batch",
    "evolution_batch_next",
    "evolution_batch_block",
    "evolution_production_status",
    "evolution_start_mining",
    "evolution_mining_control",
    "evolution_suggest_groups",
    "evolution_request_approval",
    "evolution_evaluate",
    "evolution_history_analysis",
    "evolution_submit_history_analysis",
    "evolution_read",
}

# Run the real entrypoint in a fresh interpreter. The importer proves that an
# installation without the optional indexing extra can still discover and run
# Evolution tools; the audit hook catches accidentally initialized backends.
_BOOTSTRAP = """
import importlib.abc
import runpy
import socket
import sys

class NoIndexing(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'llama_index' or fullname.startswith('llama_index.'):
            raise ModuleNotFoundError('Optional indexing dependency unavailable in fixture')
        if fullname == 'hmopt.indexing.llamaindex_pipeline':
            raise AssertionError('MCP startup/Evolution must not import the indexing pipeline')

def no_connections(event, args):
    if event == 'socket.connect':
        caller = sys._getframe(1)
        if (caller.f_code.co_name == '_fallback_socketpair'
                and caller.f_code.co_filename == socket.__file__):
            return
        raise AssertionError('Unified MCP attempted an external connection')

sys.meta_path.insert(0, NoIndexing())
sys.addaudithook(no_connections)
module = sys.argv.pop(1)
runpy.run_module(module, run_name='__main__')
"""


@asynccontextmanager
async def session_for(
    root,
    git_bin,
    *,
    legacy=False,
    profiles=None,
    workspace=None,
    artifacts=None,
    config=None,
    platform=None,
    platform_env=None,
):
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(("HMOPT_", "OPENAI_"))
    }
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        NO_COLOR="1",
        TERM="dumb",
        HMOPT_EVOLUTION_ROOT=str(root),
        HMOPT_EVOLUTION_GIT_BIN=git_bin,
        HMOPT_MCP_CONFIG=str(root.parent / "intentionally-missing-app.yaml"),
    )
    for name, path in (
        ("HMOPT_EVOLUTION_DISCOVERY_PROFILES", profiles),
        ("HMOPT_EVOLUTION_WORKSPACE_ROOT", workspace),
        ("HMOPT_EVOLUTION_ARTIFACTS_ROOT", artifacts),
    ):
        if path is not None:
            env[name] = str(path)
    if config is not None:
        env = {key: value for key, value in env.items() if not key.startswith("HMOPT_EVOLUTION_")}
        env["HMOPT_EVOLUTION_CONFIG"] = str(config)
    if platform is not None:
        env = {key: value for key, value in env.items() if not key.startswith("HMOPT_EVOLUTION_")}
        env["HMOPT_MCP_CONFIG"] = str(platform)
        env.update(platform_env or {})
    module = "hmopt.evolution.cli" if legacy else "hmopt.api.mcp_stdio"
    arguments = ["-X", "utf8", "-c", _BOOTSTRAP, module]
    if legacy:
        arguments += (
            ["--config", str(config), "mcp-stdio"]
            if config is not None
            else ["--root", str(root), "--git-bin", git_bin, "mcp-stdio"]
        )
        if profiles is not None:
            arguments += ["--discovery-profiles", str(profiles)]
    params = StdioServerParameters(
        command=sys.executable,
        args=arguments,
        cwd=str(root.parent),
        env=env,
    )
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errlog:
        async with (
            stdio_client(params, errlog=errlog) as (read, write),
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=20)) as session,
        ):
            await session.initialize()
            yield session


def test_unified_registry_and_profiles_are_lazy_and_match_legacy_contract(tmp_path, git_bin):
    root = tmp_path / "must-remain-absent"
    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "artifacts"
    workspace.mkdir()
    artifacts.mkdir()

    async def exercise():
        async with session_for(root, git_bin, workspace=workspace, artifacts=artifacts) as session:
            assert not root.exists(), "Initializing the protocol must not initialize the store"
            unified = {tool.name: tool for tool in (await session.list_tools()).tools}
            assert set(unified) == INDEX_TOOLS | EVOLUTION_TOOLS
            profiles = tool_value(await session.call_tool("evolution_discovery_profiles", {}))
            assert profiles["store_root"] == str(root.resolve())
            assert profiles["workspace_root"] == str(workspace.resolve())
            assert profiles["artifacts_root"] == str(artifacts.resolve())
            assert profiles["profiles"] == []
            assert not root.exists(), "Listing tools and frozen configuration must be read-only"

        # Both transports must expose the exact same names, docs and contracts;
        # the legacy server remains Evolution-only for existing deployments.
        async with session_for(root, git_bin, legacy=True) as session:
            legacy = {tool.name: tool for tool in (await session.list_tools()).tools}
            assert set(legacy) == EVOLUTION_TOOLS
            for name in EVOLUTION_TOOLS:
                assert unified[name].inputSchema == legacy[name].inputSchema, name
                assert unified[name].description == legacy[name].description, name
                assert unified[name].outputSchema == legacy[name].outputSchema, name
                assert unified[name].inputSchema["additionalProperties"] is False
            assert not root.exists()

        # The first actual state read may initialize the empty store. Restarting
        # through the alternate entrypoint must see the same persistent records.
        async with session_for(root, git_bin) as session:
            assert tool_value(await session.call_tool("evolution_list", {})) == []
            assert (root / "evolution.sqlite3").is_file()
        async with session_for(root, git_bin, legacy=True) as session:
            assert tool_value(await session.call_tool("evolution_list", {})) == []

    asyncio.run(exercise())


def test_setup_single_config_starts_unified_mcp_and_runs_a_bounded_batch(locations, git_bin):
    workbench, repository = locations
    generated = setup(workbench, repository, "pilot-owner", git_bin=git_bin)
    config = Path(generated["config_path"])
    settings = json.loads(config.read_text(encoding="utf-8"))
    root = Path(settings["root"])
    fragment = json.loads(Path(generated["fragment_path"]).read_text(encoding="utf-8"))
    assert fragment["mcp"]["hmopt_kernel_index"]["environment"] == {
        "HMOPT_EVOLUTION_CONFIG": str(config)
    }

    async def exercise():
        async with session_for(root, git_bin, config=config) as session:
            assert {tool.name for tool in (await session.list_tools()).tools} == (
                INDEX_TOOLS | EVOLUTION_TOOLS
            )
            configured = tool_value(await session.call_tool("evolution_discovery_profiles", {}))
            assert configured["store_root"] == str(root)
            assert configured["profiles"][0]["config"]["owners"] == {"**": "pilot-owner"}
            assert not root.exists()
            row = tool_value(
                await session.call_tool(
                    "evolution_run_discovery", {"profile": "pilot", "actor": "researcher"}
                )
            )
            assert row["data"]["status"] == "awaiting_analysis"
            assert row["data"]["target_revision"] == generated["target_revision"]
            assert row["data"]["source_changes_allowed"] is False
            assert row["data"]["completed_stages"] == ["mine", "sources", "distill"]
            assert (root / "evolution.sqlite3").is_file()
        async with session_for(root, git_bin, legacy=True, config=config) as session:
            assert tool_value(await session.call_tool("evolution_list", {"kind": "batch"})) == [row]

    asyncio.run(exercise())


def test_unified_discovery_freezes_operator_config_and_preserves_human_gates(tmp_path, git_bin):
    scenario = Scenario(tmp_path, git_bin)
    root = scenario.service.store.root
    profile = {
        "repo_path": str(scenario.repo),
        "owners": {"**/*.c": "owner"},
        "page_size": 10,
        "max_pages": 1,
    }
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"kernel": profile}), encoding="utf-8")

    async def exercise():
        async with session_for(root, git_bin, profiles=profiles) as session:
            startup = tool_value(await session.call_tool("evolution_discovery_profiles", {}))
            registered = startup["profiles"][0]
            assert registered["name"] == "kernel"
            assert registered["config_sha256"] == digest(registered["config"])
            profiles.write_text("{}", encoding="utf-8")
            assert (
                tool_value(await session.call_tool("evolution_discovery_profiles", {})) == startup
            )
            step = tool_value(
                await session.call_tool(
                    "evolution_discovery_step",
                    {"profile": "kernel", "step": "mine", "actor": "researcher"},
                )
            )
            assert step["target_revision"] == scenario.base
            assert step["caught_up"] is True
            assert tool_value(await session.call_tool("evolution_list", {"kind": "batch"})) == []

            for override in ({"repo_path": str(tmp_path)}, {"recover_running": True}):
                tool_error(
                    await session.call_tool(
                        "evolution_run_discovery",
                        {"profile": "kernel", "actor": "researcher", **override},
                    ),
                    "Unexpected tool arguments",
                )
            tool_error(
                await session.call_tool(
                    "evolution_run_discovery", {"profile": "unknown", "actor": "researcher"}
                ),
                "profile",
            )
            row = tool_value(
                await session.call_tool(
                    "evolution_run_discovery", {"profile": "kernel", "actor": "researcher"}
                )
            )
            assert row["data"]["status"] == "awaiting_analysis"
            assert row["data"]["config_digest"] == registered["config_sha256"]
            assert row["data"]["source_changes_allowed"] is False
            assert row["data"]["completed_stages"] == ["mine", "sources", "distill"]
            assert (
                tool_value(
                    await session.call_tool(
                        "evolution_read", {"kind": "batch", "record_id": row["id"]}
                    )
                )
                == row
            )
            candidates = tool_value(await session.call_tool("evolution_list", {}))
            assert candidates and all(item["data"]["stage"] == "discovered" for item in candidates)
            current = next(item for item in candidates if item["id"] == scenario.candidate_id)
            tool_error(
                await session.call_tool(
                    "evolution_submit",
                    {
                        "candidate_id": scenario.candidate_id,
                        "action": "confirm",
                        "actor": "owner",
                        "expected_version": current["version"],
                        "request_id": "not-an-owner-approval-channel",
                        "payload": {"note": "Agent text cannot grant approval."},
                    },
                ),
                "operator CLI",
            )
            tool_error(
                await session.call_tool(
                    "evolution_dispatch",
                    {
                        "candidate_id": scenario.candidate_id,
                        "actor": "agent",
                        "request_id": "unconfirmed-unified-dispatch",
                    },
                ),
                "No executable handoff",
            )
            assert scenario.service.store.read("candidate", scenario.candidate_id) == current

        # A legacy caller sees the actual batch produced by the unified API.
        async with session_for(root, git_bin, legacy=True) as session:
            assert tool_value(await session.call_tool("evolution_list", {"kind": "batch"})) == [row]

    asyncio.run(exercise())
