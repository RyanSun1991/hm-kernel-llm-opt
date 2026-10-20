"""Doctor validates local setup without creating state or claiming live execution."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from test_evolution_service import git_bin as shared_git_bin

from hmopt.evolution.doctor import doctor
from hmopt.evolution.setup import _ASSETS

git_bin = shared_git_bin


@pytest.fixture
def configured(tmp_path, monkeypatch, git_bin):
    for name in tuple(os.environ):
        if name.startswith("HMOPT_EVOLUTION_"):
            monkeypatch.delenv(name)
    repo = tmp_path / "target"
    repo.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0")

    def git(*arguments):
        return subprocess.run(
            [git_bin, "-C", str(repo), *arguments],
            capture_output=True,
            check=True,
            env=env,
        )

    git("init", "-q")
    (repo / "kernel.c").write_text("int target(void) { return 1; }\n", encoding="utf-8")
    git("add", "kernel.c")
    git(
        "-c",
        "user.name=Doctor Test",
        "-c",
        "user.email=doctor@example.invalid",
        "commit",
        "-qm",
        "baseline",
    )
    workbench = tmp_path / "workbench"
    for asset in _ASSETS:
        path = workbench / ".opencode" / asset
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Fixture {path.name}\n", encoding="utf-8")
    workspace = workbench / ".opencode/local/workspaces"
    workspace.mkdir(parents=True)
    config = workbench / ".opencode/local/evolution/config.json"
    config.parent.mkdir(parents=True)
    values = {
        "schema_version": 1,
        "root": str(config.parent / "state"),
        "git_bin": git_bin,
        "workspace_root": str(workspace),
        "artifacts_root": str(config.parent / "artifacts"),
        "discovery_profiles": {
            "pilot": {
                "repo_path": str(repo),
                "owners": {"**": "module-owner"},
                "workspace": str(workbench),
                "repo_id": "fixture-target",
            }
        },
    }
    config.write_text(json.dumps(values), encoding="utf-8")
    return config, values, repo, workbench


def checks(result):
    return {item["name"]: item for item in result["checks"]}


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_doctor_fresh_setup_is_readonly_offline_and_does_not_claim_production(
    configured, tmp_path, monkeypatch
):
    config, values, _, _ = configured
    from hmopt.evolution.store import EvolutionStore

    def no_store(*args, **kwargs):
        raise AssertionError("Doctor must not initialize the workflow store")

    real_connect = socket.socket.connect

    def no_network(self, address):
        caller = sys._getframe(1)
        if (
            caller.f_code.co_name == "_fallback_socketpair"
            and caller.f_code.co_filename == socket.__file__
        ):
            return real_connect(self, address)
        raise AssertionError("Doctor must not connect to a model, index or device")

    monkeypatch.setattr(EvolutionStore, "__init__", no_store)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    before = snapshot(tmp_path)
    result = doctor(config)
    assert result["ready_for_discovery"] is True, result
    assert result["execution_prerequisites"] is True
    assert checks(result)["mcp_registry"]["evolution_tool_count"] == 38
    assert checks(result)["workflow_store"]["status"] == "not_configured"
    assert checks(result)["lmbench_raw_evidence"]["status"] == "not_configured"
    assert checks(result)["production_execution"]["status"] == "not_verified"
    assert all(
        result[name] is False
        for name in (
            "production_validated",
            "model_executed",
            "device_executed",
            "opencode_connected",
        )
    )
    assert not Path(values["root"]).exists()
    assert snapshot(tmp_path) == before
    json.dumps(result)


def test_doctor_dirty_tracked_target_blocks_execution_but_allows_history_discovery(configured):
    config, _, repo, _ = configured
    (repo / "kernel.c").write_text("int target(void) { return 2; }\n", encoding="utf-8")
    result = doctor(config)
    assert result["ready_for_discovery"] is True
    assert result["execution_prerequisites"] is False
    assert checks(result)["tracked_worktree:pilot"]["dirty"] is True
    assert checks(result)["tracked_worktree:pilot"]["status"] == "warning"


def test_doctor_missing_role_and_non_top_level_repository_are_actionable(configured):
    config, values, repo, workbench = configured
    (workbench / ".opencode/agents/reviewer.md").write_text(" \n", encoding="utf-8")
    subdir = repo / "module"
    subdir.mkdir()
    values["discovery_profiles"]["pilot"]["repo_path"] = str(subdir)
    config.write_text(json.dumps(values), encoding="utf-8")
    result = doctor(config)
    assert result["ready_for_discovery"] is False
    assert result["execution_prerequisites"] is False
    assert checks(result)["workbench_assets"]["status"] == "blocked"
    assert "reviewer.md" in checks(result)["workbench_assets"]["missing"][0]
    assert "top-level" in checks(result)["profile:pilot"]["message"]


def test_doctor_reports_effective_environment_overrides_without_showing_values(
    configured, monkeypatch, tmp_path
):
    config, _, _, _ = configured
    redirected = tmp_path / "state-from-explicit-environment"
    monkeypatch.setenv("HMOPT_EVOLUTION_ROOT", str(redirected))
    monkeypatch.setenv("HMOPT_LLM_API_KEY", "fixture-secret-do-not-print")
    result = doctor(config)
    assert result["ready_for_discovery"] is True
    assert result["environment_overrides"] == ["HMOPT_EVOLUTION_ROOT"]
    assert str(redirected) not in json.dumps(result)
    assert "fixture-secret-do-not-print" not in json.dumps(result)
    assert not redirected.exists()


def test_doctor_invalid_configuration_is_reported_without_echoing_inputs(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root": str(tmp_path / "state"),
                "credential": "do-not-echo-this-value",
            }
        ),
        encoding="utf-8",
    )
    result = doctor(path)
    assert result["ready_for_discovery"] is False
    assert checks(result)["configuration"]["status"] == "blocked"
    assert "credential" in checks(result)["configuration"]["message"]
    assert "do-not-echo-this-value" not in json.dumps(result)
    assert not (tmp_path / "state").exists()


def test_doctor_counts_existing_store_readonly_and_refuses_unstable_wal(configured, tmp_path):
    config, values, _, _ = configured
    root = Path(values["root"])
    root.mkdir()
    database = root / "evolution.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE records (kind TEXT, payload TEXT)")
        db.execute("INSERT INTO records VALUES (?,?)", ("pattern", '{"status":"active"}'))
        db.execute("INSERT INTO records VALUES (?,?)", ("pattern", '{"status":"draft"}'))
    before = snapshot(tmp_path)
    result = doctor(config)
    assert result["ready_for_discovery"] is True
    assert checks(result)["workflow_store"]["records"] == {"pattern": 2}
    assert checks(result)["workflow_store"]["active_pattern_versions"] == 1
    assert snapshot(tmp_path) == before
    database.with_name(database.name + "-wal").write_bytes(b"uncheckpointed fixture")
    before = snapshot(tmp_path)
    result = doctor(config)
    assert checks(result)["workflow_store"]["status"] == "not_verified"
    assert "active_pattern_versions" not in checks(result)["workflow_store"]
    assert result["execution_prerequisites"] is False
    assert snapshot(tmp_path) == before


def test_doctor_missing_git_executable_blocks_discovery(configured):
    config, values, _, _ = configured
    values["git_bin"] = "hmopt-doctor-missing-git-executable"
    config.write_text(json.dumps(values), encoding="utf-8")
    result = doctor(config)
    assert result["ready_for_discovery"] is False
    assert checks(result)["profile:pilot"]["status"] == "blocked"
    assert "Git executable" in checks(result)["profile:pilot"]["message"]


@pytest.mark.parametrize("owners", [{"../outside": "owner"}, {"**": "owner\nforged"}])
def test_doctor_owner_mapping_obeys_the_actual_discovery_contract(configured, owners):
    config, values, _, _ = configured
    values["discovery_profiles"]["pilot"]["owners"] = owners
    config.write_text(json.dumps(values), encoding="utf-8")
    result = doctor(config)
    assert result["ready_for_discovery"] is False
    assert checks(result)["profile:pilot"]["status"] == "blocked"


def test_doctor_historical_revision_does_not_claim_execution_checkout_ready(configured):
    config, values, repo, _ = configured
    original = checks(doctor(config))["profile:pilot"]["head"]
    (repo / "kernel.c").write_text("int target(void) { return 2; }\n", encoding="utf-8")
    subprocess.run(
        [
            values["git_bin"],
            "-C",
            str(repo),
            "-c",
            "user.name=Doctor Test",
            "-c",
            "user.email=doctor@example.invalid",
            "commit",
            "-qam",
            "new baseline",
        ],
        capture_output=True,
        check=True,
    )
    values["discovery_profiles"]["pilot"]["revision"] = original
    config.write_text(json.dumps(values), encoding="utf-8")
    result = doctor(config)
    assert result["ready_for_discovery"] is True
    assert result["execution_prerequisites"] is False
    assert checks(result)["execution_revision:pilot"]["status"] == "warning"


def test_doctor_rejects_store_below_a_file_without_writing_anything(configured, tmp_path):
    config, values, _, _ = configured
    blocked_parent = tmp_path / "file-instead-of-directory"
    blocked_parent.write_text("preserve this file", encoding="utf-8")
    values["root"] = str(blocked_parent / "state")
    config.write_text(json.dumps(values), encoding="utf-8")
    before = snapshot(tmp_path)
    result = doctor(config)
    assert result["ready_for_discovery"] is False
    assert checks(result)["workflow_store"]["status"] == "blocked"
    assert snapshot(tmp_path) == before
