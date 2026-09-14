"""Single-file operator configuration and compatibility with explicit CLI settings."""

from __future__ import annotations

import json
import os

import pytest
from test_evolution_setup import git_bin as shared_git_bin
from test_evolution_setup import locations as shared_locations
from typer.testing import CliRunner

from hmopt.api.evolution_mcp_service import load_evolution_config, read_environment_options
from hmopt.evolution.cli import app

git_bin = shared_git_bin
locations = shared_locations


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    for name in os.environ:
        if name.startswith("HMOPT_EVOLUTION_"):
            monkeypatch.delenv(name)
    repo = tmp_path / "repo"
    repo.mkdir()
    payload = {
        "schema_version": 1,
        "root": str(tmp_path / "state"),
        "git_bin": "configured-git",
        "workspace_root": str(tmp_path / ".opencode/local/workspaces"),
        "artifacts_root": str(tmp_path / "artifacts"),
        "discovery_profiles": {"pilot": {"repo_path": str(repo), "owners": {"**": "owner"}}},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_file_load_and_empty_environment_preserve_one_config(configuration, monkeypatch):
    path, payload = configuration
    monkeypatch.setenv("HMOPT_EVOLUTION_CONFIG", str(path))
    monkeypatch.setenv("HMOPT_EVOLUTION_ROOT", "")
    monkeypatch.setenv("HMOPT_EVOLUTION_GIT_BIN", "")
    direct = load_evolution_config(path)
    effective = read_environment_options()
    assert direct == effective
    assert str(effective["root"]) == payload["root"]
    assert effective["git_bin"] == "configured-git"
    assert not direct["root"].exists()


def test_explicit_advanced_overrides_remain_supported(configuration, monkeypatch, tmp_path):
    path, _ = configuration
    override = tmp_path / "override"
    monkeypatch.setenv("HMOPT_EVOLUTION_ROOT", str(override))
    monkeypatch.setenv("HMOPT_EVOLUTION_GIT_BIN", "operator-git")
    options = read_environment_options(path)
    assert str(options["root"]) == str(override)
    assert options["git_bin"] == "operator-git"
    assert not override.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"root": "relative/state"},
        {"workspace_root": "relative/workspace"},
        {"approval_bypass": True},
        {"discovery_profiles": {"../outside": {}}},
    ],
)
def test_single_config_rejects_invalid_contract_before_any_state_write(configuration, change):
    path, payload = configuration
    path.write_text(json.dumps({**payload, **change}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_evolution_config(path)
    assert not (path.parent / "state").exists()


def test_cli_uses_same_file_and_explicit_root_takes_precedence(configuration, tmp_path):
    path, payload = configuration
    runner = CliRunner()
    result = runner.invoke(app, ["--config", str(path), "init"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["root"] == payload["root"]
    override = tmp_path / "explicit-cli-state"
    result = runner.invoke(app, ["--config", str(path), "--root", str(override), "init"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["root"] == str(override)


def test_cli_invalid_config_is_actionable_without_traceback(configuration):
    path, payload = configuration
    path.write_text(json.dumps({**payload, "unknown": True}), encoding="utf-8")
    result = CliRunner().invoke(app, ["--config", str(path), "init"])
    assert result.exit_code != 0
    assert "--config" in result.output
    assert "Traceback" not in result.output
    assert not (path.parent / "state").exists()


@pytest.mark.parametrize("command", ["init", "doctor"])
def test_cli_invalid_explicit_config_is_redacted_and_doctor_still_reports_json(
    configuration, command
):
    path, payload = configuration
    path.write_text(
        json.dumps({**payload, "credential": "fixture-secret-must-not-be-echoed"}),
        encoding="utf-8",
    )
    arguments = ["--config", str(path), command]
    if command == "doctor":
        arguments.append("--json")
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 2
    assert "fixture-secret-must-not-be-echoed" not in result.output
    assert "Traceback" not in result.output
    if command == "doctor":
        report = json.loads(result.stdout)
        assert not report["ready_for_discovery"]
        assert report["checks"][0]["status"] == "blocked"
        assert "credential" in report["checks"][0]["message"]
    assert not (path.parent / "state").exists()


def test_cli_doctor_reports_missing_explicit_config_as_json_without_creating_it(tmp_path):
    path = tmp_path / "missing" / "config.json"
    result = CliRunner().invoke(app, ["--config", str(path), "doctor", "--json"])
    assert result.exit_code == 2
    report = json.loads(result.stdout)
    assert report["checks"][0]["status"] == "blocked"
    assert not path.parent.exists()


def test_cli_setup_and_doctor_offer_simple_text_and_machine_output(locations, git_bin, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("HMOPT_EVOLUTION_"):
            monkeypatch.delenv(name)
    workbench, repo = locations
    runner = CliRunner()
    arguments = [
        "--git-bin",
        git_bin,
        "setup",
        "--workbench",
        str(workbench),
        "--repo",
        str(repo),
        "--owner",
        "pilot-owner",
    ]
    generated = runner.invoke(app, [*arguments, "--json"])
    assert generated.exit_code == 0, generated.output
    config = json.loads(generated.stdout)["config_path"]
    repeated = runner.invoke(app, arguments)
    assert repeated.exit_code == 0, repeated.output
    assert "配置已生成" in repeated.output
    assert "HMOPT_EVOLUTION_CONFIG" in repeated.output
    assert "合并到现有连接" in repeated.output
    assert "尚无主 MCP" in repeated.output
    assert "remote" in repeated.output
    assert "不新增并行本地连接" in repeated.output
    assert "--config" in repeated.output and config in repeated.output
    monkeypatch.chdir(workbench)
    before = {path: path.read_bytes() for path in workbench.rglob("*") if path.is_file()}
    report = runner.invoke(app, ["--config", config, "doctor", "--json"])
    assert report.exit_code == 0, report.output
    value = json.loads(report.stdout)
    assert value["ready_for_discovery"]
    assert not value["production_validated"]
    assert not value["opencode_connected"]
    human = runner.invoke(app, ["doctor"])
    assert human.exit_code == 0, human.output
    assert "发现准备：检查通过" in human.output
    assert "未验证 OpenCode 连接" in human.output
    assert "/evolve-discover pilot" in human.output
    after = {path: path.read_bytes() for path in workbench.rglob("*") if path.is_file()}
    assert before == after
