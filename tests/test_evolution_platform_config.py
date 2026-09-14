"""Shared YAML, existing environment and real setup/CLI contracts without model calls."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from test_evolution_mining import GIT, commit, git
from typer.testing import CliRunner

from hmopt.api.evolution_mcp_service import load_evolution_config, read_environment_options
from hmopt.core.config import AppConfig
from hmopt.evolution.cli import app
from hmopt.evolution.setup import _ASSETS, setup_workspace


@pytest.fixture
def platform(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("HMOPT_EVOLUTION_", "OPENCODE_")) or name in {
            "HMOPT_MCP_CONFIG",
            "PROJECT_REPO_PATH",
            "KERNEL_REPO_PATH",
            "KERNEL_WORKSPACE_PATH",
        }:
            monkeypatch.delenv(name)
    workbench = tmp_path / "platform"
    path = workbench / "configs/app.yaml"
    path.parent.mkdir(parents=True)
    business = tmp_path / "business"
    for name in ("kernel/main", "foundation/memory", "unselected"):
        (business / name).mkdir(parents=True)
    monkeypatch.setenv("PROJECT_REPO_PATH", str(business))
    monkeypatch.setenv("KERNEL_REPO_PATH", str(business / "kernel/main"))
    monkeypatch.setenv("HMOPT_MCP_CONFIG", str(path))
    raw = {
        "project": {"name": "kernel", "repo_path": "unused-fallback"},
        "llm": {"api_key_env": "HMOPT_LLM_API_KEY"},
        "storage": {"artifacts": {"root_dir": "shared/evidence"}},
        "evolution": {
            "source_workspace": {
                "workspace_id": "business",
                "projects": {
                    "kernel": {"owners": {"**": "owner"}},
                    "memory": {"path": "foundation/memory", "owners": {"**": "owner"}},
                },
                "selected": ["kernel", "memory"],
            }
        },
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8", newline="\n")
    return workbench, business, path, raw


def write_config(path, raw):
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8", newline="\n")


def test_shared_config_inherits_existing_paths_without_database_or_llm(platform):
    workbench, business, path, raw = platform
    options = read_environment_options()
    assert options == load_evolution_config(path)
    assert options["root"] == workbench / "shared/evolution"
    assert options["artifacts_root"] == workbench / "shared/evidence"
    assert options["workspace_root"] == workbench / ".opencode/local/workspaces"
    assert options["git_bin"] == "git"
    assert options["source_workspace"]["root"] == str(business)
    assert options["discovery_profiles"]["kernel"]["repo_path"] == str(business / "kernel/main")
    assert set(options["discovery_profiles"]) == {"kernel", "memory"}
    assert AppConfig.from_yaml(path).evolution == raw["evolution"]
    assert not options["root"].exists()


def test_explicit_section_paths_win_and_are_config_relative(platform, monkeypatch):
    workbench, business, path, raw = platform
    monkeypatch.setenv("PROJECT_REPO_PATH", str(business / "wrong"))
    raw["evolution"].update(root="custom/state", artifacts_root="custom/results")
    raw["evolution"]["source_workspace"]["root"] = str(business)
    write_config(path, raw)
    options = load_evolution_config(path)
    assert options["root"] == workbench / "custom/state"
    assert options["artifacts_root"] == workbench / "custom/results"
    assert options["source_workspace"]["root"] == str(business)
    monkeypatch.setenv("HMOPT_EVOLUTION_ROOT", str(workbench / "legacy-state"))
    assert read_environment_options()["root"] == str(workbench / "legacy-state")


def test_worker_reuses_opencode_researcher_model_and_local_server_without_copying_secrets(platform):
    workbench, _, path, raw = platform
    (workbench / "opencode.jsonc").write_text(
        """{
      // The URL and key deliberately contain comment-like tokens.
      "model": "provider/default", "agent": {"researcher": {"model": "provider/research"}},
      "server": {"port": 49174, "hostname": "0.0.0.0"},
      "provider": {"provider": {"options": {"baseURL":"https://model.invalid/v1", "apiKey":"secret//do-not-copy"}}},
    }""",
        encoding="utf-8",
        newline="\n",
    )
    raw["evolution"]["production"] = {"worker": {}}
    write_config(path, raw)
    options = load_evolution_config(path)
    worker = options["production"]["worker"]
    assert (worker["provider_id"], worker["model_id"]) == ("provider", "research")
    assert worker["directory"] == str(workbench)
    assert worker["url"] == "http://127.0.0.1:49174"
    assert "secret//do-not-copy" not in json.dumps(options, default=str)
    assert worker["concurrency"] == 2


def test_explicit_worker_connection_is_not_overwritten_by_opencode_config(platform):
    workbench, _, path, raw = platform
    (workbench / "opencode.jsonc").write_text('{"model": "another/model"}', encoding="utf-8")
    raw["evolution"]["production"] = {
        "worker": {"url": "http://127.0.0.1:4000", "provider_id": "selected", "model_id": "exact"}
    }
    write_config(path, raw)
    worker = load_evolution_config(path)["production"]["worker"]
    assert (worker["provider_id"], worker["model_id"], worker["url"]) == (
        "selected",
        "exact",
        "http://127.0.0.1:4000",
    )


def test_unresolvable_model_does_not_fall_back_to_a_fake_or_different_backend(platform):
    _, _, path, raw = platform
    raw["evolution"]["production"] = {"worker": {"url": "http://127.0.0.1:4096"}}
    write_config(path, raw)
    with pytest.raises(ValueError, match="provider_id"):
        load_evolution_config(path)


def test_validation_reuses_business_cwd_and_conservative_default_resource(platform):
    _, business, path, raw = platform
    raw["evolution"]["production"] = {
        "validation": {"command": [sys.executable, "existing_adapter.py"]}
    }
    write_config(path, raw)
    validation = load_evolution_config(path)["production"]["validation"]
    assert validation["cwd"] == str(business)
    assert validation["resource_id"] == "business-validation"
    assert validation["timeout_seconds"] == 3600


def test_cli_uses_the_same_platform_config_without_evolution_environment(platform):
    workbench, _, _, _ = platform
    result = CliRunner().invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["root"] == str(workbench / "shared/evolution")


def test_legacy_explicit_file_still_wins_over_platform_config(platform, monkeypatch):
    workbench, _, _, _ = platform
    path = workbench / "legacy.json"
    path.write_text(
        json.dumps({"schema_version": 1, "root": str(workbench / "previous-state")}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HMOPT_EVOLUTION_CONFIG", str(path))
    assert read_environment_options()["root"] == workbench / "previous-state"


@pytest.mark.parametrize("via_cli", [False, True])
def test_setup_reuses_platform_bytes_and_existing_repo_env_idempotently(platform, via_cli):
    workbench, business, path, raw = platform
    for relative in ("kernel/main", "foundation/memory"):
        repo = business / relative
        git(repo, "init", "-b", "main")
        git(repo, "config", "user.name", "Fixture")
        git(repo, "config", "user.email", "fixture@example.invalid")
        (repo / "code.c").write_text("int value;\n", encoding="utf-8", newline="\n")
        commit(repo, "baseline")
    for asset in _ASSETS:
        target = workbench / ".opencode" / asset
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Existing workbench asset\n", encoding="utf-8", newline="\n")
    del raw["evolution"]
    before = "# Operator comment must survive.\n" + yaml.safe_dump(raw, sort_keys=False)
    path.write_text(before, encoding="utf-8", newline="\n")
    args = (workbench, None, "business", {"kernel": None, "memory": "foundation/memory"}, "owner")
    if via_cli:
        command = CliRunner().invoke(
            app,
            [
                "--git-bin",
                GIT,
                "setup-workspace",
                "--workbench",
                str(workbench),
                "--project",
                "kernel",
                "--project",
                "memory=foundation/memory",
                "--owner",
                "owner",
            ],
        )
        assert command.exit_code == 0, command.output
        result = json.loads(command.stdout)
    else:
        result = setup_workspace(*args, git_bin=GIT)
    assert result["config_path"] == str(path)
    assert path.read_text(encoding="utf-8").startswith(before)
    added = yaml.safe_load(path.read_text(encoding="utf-8"))["evolution"]
    assert set(added) == {"source_workspace", "git_bin"}
    assert "root" not in added["source_workspace"]
    assert "path" not in added["source_workspace"]["projects"]["kernel"]
    fragment = json.loads(Path(result["fragment_path"]).read_text(encoding="utf-8"))
    assert fragment["mcp"]["hmopt_kernel_index"]["environment"] == {"HMOPT_MCP_CONFIG": str(path)}
    snapshot = path.read_bytes()
    setup_workspace(*args, git_bin=GIT)
    assert path.read_bytes() == snapshot
    assert not (workbench / "shared/evolution/evolution.sqlite3").exists()
    assert not (Path(result["fragment_path"]).parent / "config.json").exists()
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ready_for_discovery"]


def test_documented_workspace_config_generates_profiles(platform):
    _, _, path, raw = platform
    guide = (Path(__file__).resolve().parents[1] / "docs/EVOLUTION_USAGE_CN.md").read_text(
        encoding="utf-8"
    )
    example = yaml.safe_load(guide.split("```yaml\n", 1)[1].split("\n```", 1)[0])
    raw.update(example)
    write_config(path, raw)
    options = load_evolution_config(path)
    assert set(options["discovery_profiles"]) == {"kernel", "memory"}
    assert set(example["evolution"]) == {"source_workspace"}


def test_shared_config_through_real_unified_stdio_without_evolution_env(platform):
    from test_evolution_unified_stdio import EVOLUTION_TOOLS, INDEX_TOOLS, session_for, tool_value

    workbench, business, path, raw = platform
    raw["evolution"]["git_bin"] = GIT
    write_config(path, raw)
    root = workbench / "shared/evolution"
    root.parent.mkdir()

    async def exercise():
        async with session_for(
            root,
            GIT,
            platform=path,
            platform_env={
                "PROJECT_REPO_PATH": str(business),
                "KERNEL_REPO_PATH": str(business / "kernel/main"),
            },
        ) as session:
            assert {
                tool.name for tool in (await session.list_tools()).tools
            } == INDEX_TOOLS | EVOLUTION_TOOLS
            settings = tool_value(await session.call_tool("evolution_discovery_profiles", {}))
            assert settings["store_root"] == str(root)
            assert {item["name"] for item in settings["profiles"]} == {"kernel", "memory"}
            assert not root.exists(), "Tool discovery must stay read-only"
            assert tool_value(await session.call_tool("evolution_list", {})) == []
            assert root.exists(), "Operations must use the inherited state directory"

    asyncio.run(exercise())


def test_shared_config_through_http_without_evolution_env(platform, monkeypatch):
    from fastapi.testclient import TestClient
    from test_evolution_unified_http import TOKEN, rpc, rpc_value

    from hmopt.api import mcp_service
    from hmopt.api.mcp_server import create_app

    workbench, _, _, _ = platform
    monkeypatch.setenv("HMOPT_MCP_ALLOWED_HOSTS", "testserver,localhost")
    mcp_service.build_fastmcp_server.cache_clear()
    try:
        application = create_app(
            mcp_service.build_fastmcp_server(), api_key=TOKEN, mount_path="/platform/mcp/"
        )
        with TestClient(application) as client:
            settings = rpc_value(
                rpc(client, "tools/call", {"name": "evolution_discovery_profiles", "arguments": {}})
            )
            assert settings["store_root"] == str(workbench / "shared/evolution")
            assert {item["name"] for item in settings["profiles"]} == {"kernel", "memory"}
            assert not (workbench / "shared/evolution").exists()
    finally:
        mcp_service.build_fastmcp_server.cache_clear()


def test_missing_or_outside_kernel_never_expands_scope(platform, monkeypatch):
    _, business, path, _ = platform
    monkeypatch.setenv("KERNEL_REPO_PATH", str(business.parent / "unselected-kernel"))
    with pytest.raises(ValueError, match="outside the business workspace"):
        load_evolution_config(path)


def test_partial_model_override_cannot_inherit_a_different_pair(platform):
    workbench, _, path, raw = platform
    (workbench / "opencode.json").write_text(
        '{"model":"provider/default","server":{"port":4096}}', encoding="utf-8"
    )
    raw["evolution"]["production"] = {"worker": {"model_id": "custom"}}
    write_config(path, raw)
    with pytest.raises(ValueError, match="provider_id"):
        load_evolution_config(path)


def test_explicit_missing_opencode_config_and_yaml_errors_are_actionable(platform, monkeypatch):
    workbench, _, path, raw = platform
    raw["evolution"]["production"] = {"worker": {}}
    write_config(path, raw)
    monkeypatch.setenv("OPENCODE_CONFIG", str(workbench / "missing.json"))
    with pytest.raises(ValueError, match="Explicit OPENCODE_CONFIG is missing"):
        load_evolution_config(path)
    path.write_text("evolution: [\nsecret-key: PRIVATE-VALUE", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid platform YAML") as error:
        load_evolution_config(path)
    assert "PRIVATE-VALUE" not in str(error.value)


def test_setup_never_publishes_invalid_yaml_after_document_end(tmp_path):
    from hmopt.evolution.platform_setup import _append_section

    path = tmp_path / "app.yaml"
    before = b"project: {}\n...\n"
    path.write_bytes(before)
    with pytest.raises(ValueError, match="document boundaries"):
        _append_section(path, before, {"source_workspace": {}})
    assert path.read_bytes() == before
    assert not path.with_name("app.yaml.evolution-setup.lock").exists()


@pytest.mark.parametrize("section", ["storage", "projects"])
def test_invalid_platform_sections_report_config_errors(platform, section):
    _, _, path, raw = platform
    if section == "storage":
        raw["storage"] = []
    else:
        raw["evolution"]["source_workspace"]["projects"] = []
    write_config(path, raw)
    with pytest.raises(ValueError, match=section):
        load_evolution_config(path)
