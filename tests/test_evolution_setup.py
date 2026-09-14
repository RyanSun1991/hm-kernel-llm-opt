"""Setup creates reviewable local configuration, without starting or approving work."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_evolution_service import git_bin as shared_git_bin

from hmopt.evolution.discovery import DiscoveryConfig
from hmopt.evolution.setup import setup

git_bin = shared_git_bin
PROJECT = Path(__file__).resolve().parents[1]


def git(repo, executable, *arguments):
    environment = dict(os.environ)
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
    ):
        environment.pop(key, None)
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return subprocess.run(
        [
            executable,
            "-C",
            str(repo),
            "-c",
            "user.name=Setup Fixture",
            "-c",
            "user.email=setup@example.invalid",
            "-c",
            f"core.hooksPath={repo / 'no-hooks'}",
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    ).stdout.strip()


@pytest.fixture
def locations(tmp_path, git_bin):
    workbench = tmp_path / "existing workbench"
    assets = workbench / ".opencode"
    for name in ("agents", "commands", "skills"):
        shutil.copytree(PROJECT / ".opencode" / name, assets / name)
    shutil.copy2(PROJECT / ".opencode" / "config.yaml", assets / "config.yaml")
    repository = tmp_path / "kernel checkout"
    repository.mkdir()
    git(repository, git_bin, "init", "-q")
    git(repository, git_bin, "commit", "--allow-empty", "-qm", "Setup fixture baseline")
    return workbench, repository


def test_setup_prepares_one_local_config_without_touching_existing_connections(locations, git_bin):
    workbench, repository = locations
    connection = workbench / "opencode.jsonc"
    previous = '// operator config\n{"mcp":{"main":{"type":"remote","url":"https://mcp.invalid"}}}'
    connection.write_text(previous, encoding="utf-8")
    result = setup(workbench, repository, "kernel-owner", git_bin=git_bin)
    assert result["status"] == "setup_ready"
    assert result["workflow_started"] is False
    assert result["target_revision"] == git(repository, git_bin, "rev-parse", "HEAD")
    assert result["config_created"] and result["fragment_created"]
    assert len(result["next_steps"]) == 3
    assert "remote" in result["next_steps"][0]
    assert "不新增" in result["next_steps"][0]
    assert "/evolve-discover pilot" in result["next_steps"][2]
    config_path = Path(result["config_path"])
    assert config_path == workbench / ".opencode/local/evolution/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert set(config) == {
        "schema_version",
        "root",
        "git_bin",
        "workspace_root",
        "artifacts_root",
        "discovery_profiles",
    }
    assert config["schema_version"] == 1
    from hmopt.api.evolution_mcp_service import load_evolution_config

    loaded = load_evolution_config(config_path)
    assert loaded["root"] == Path(config["root"])
    assert loaded["workspace_root"] == Path(config["workspace_root"])
    pilot = DiscoveryConfig.model_validate(config["discovery_profiles"]["pilot"])
    assert pilot.repo_path == str(repository.resolve())
    assert pilot.workspace == str(workbench.resolve())
    assert pilot.owners == {"**": "kernel-owner"}
    assert (
        pilot.max_pages,
        pilot.page_size,
        pilot.source_page_size,
        pilot.scheduling_budget_s,
    ) == (1, 50, 50, 30)
    assert pilot.revision == "HEAD"
    assert Path(config["workspace_root"]).is_dir()
    assert Path(config["artifacts_root"]).is_dir()
    assert not Path(config["root"]).exists()
    assert not list(workbench.rglob("*.sqlite3"))
    fragment = json.loads(Path(result["fragment_path"]).read_text(encoding="utf-8"))
    assert set(fragment) == {"mcp"}
    entry = fragment["mcp"]["hmopt_kernel_index"]
    assert entry["timeout"] == 180000
    assert entry["timeout"] > pilot.scheduling_budget_s * 1000
    assert entry["command"] == [str(Path(sys.executable).absolute()), "-m", "hmopt.api.mcp_stdio"]
    assert entry["environment"] == {"HMOPT_EVOLUTION_CONFIG": str(config_path)}
    assert connection.read_text(encoding="utf-8") == previous


def test_setup_is_semantically_idempotent_across_json_formatting_and_head_changes(
    locations, git_bin
):
    workbench, repository = locations
    first = setup(workbench, repository, "owner", git_bin=git_bin)
    config_path = Path(first["config_path"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    before = config_path.read_bytes(), config_path.stat().st_mtime_ns
    git(repository, git_bin, "commit", "--allow-empty", "-qm", "New pilot target")
    second = setup(workbench, repository, "owner", git_bin=git_bin)
    assert not second["config_created"] and not second["fragment_created"]
    assert second["target_revision"] != first["target_revision"]
    assert (config_path.read_bytes(), config_path.stat().st_mtime_ns) == before


def test_setup_uses_git_top_level_and_stable_repository_identity(locations, git_bin):
    workbench, repository = locations
    nested = repository / "submodule-name"
    nested.mkdir()
    first = setup(workbench, nested, "owner", git_bin=git_bin)
    second = setup(workbench, repository, "owner", git_bin=git_bin)
    assert first["repo_path"] == second["repo_path"] == str(repository.resolve())
    assert not second["config_created"]


@pytest.mark.parametrize("owner", ["", "   ", "line\nbreak", "x" * 201, None])
def test_setup_rejects_invalid_owner_before_writing(locations, git_bin, owner):
    workbench, repository = locations
    with pytest.raises(ValueError, match="Owner"):
        setup(workbench, repository, owner, git_bin=git_bin)
    assert not (workbench / ".opencode/local").exists()


@pytest.mark.parametrize(
    "asset",
    [
        "agents/reviewer.md",
        "commands/evolve-discover.md",
        "skills/infra/pipeline/evolution-execution/SKILL.md",
        "skills/infra/pipeline/evolution-mining/SKILL.md",
    ],
)
def test_setup_rejects_incomplete_workbench_instead_of_installing_fake_roles(
    locations, git_bin, asset
):
    workbench, repository = locations
    missing = workbench / ".opencode" / asset
    missing.unlink()
    with pytest.raises(ValueError, match="missing installed Evolution") as exc:
        setup(workbench, repository, "owner", git_bin=git_bin)
    assert asset in str(exc.value)
    assert not missing.exists()
    assert not (workbench / ".opencode/local").exists()


def test_setup_rejects_nonrepo_and_unborn_head_before_writing(locations, git_bin, tmp_path):
    workbench, _ = locations
    empty = tmp_path / "empty"
    empty.mkdir()
    for initialize in (False, True):
        if initialize:
            git(empty, git_bin, "init", "-q")
        with pytest.raises(ValueError, match="HEAD commit"):
            setup(workbench, empty, "owner", git_bin=git_bin)
        assert not (workbench / ".opencode/local").exists()


def test_setup_refuses_changed_owner_without_rewriting_either_file(locations, git_bin):
    workbench, repository = locations
    result = setup(workbench, repository, "original-owner", git_bin=git_bin)
    paths = [Path(result[key]) for key in ("config_path", "fragment_path")]
    before = [path.read_bytes() for path in paths]
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        setup(workbench, repository, "different-owner", git_bin=git_bin)
    assert [path.read_bytes() for path in paths] == before


def test_setup_does_not_accept_boolean_as_the_same_integer_schema_version(locations, git_bin):
    workbench, repository = locations
    result = setup(workbench, repository, "owner", git_bin=git_bin)
    path = Path(result["config_path"])
    invalid = json.loads(path.read_text(encoding="utf-8"))
    invalid["schema_version"] = True
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        setup(workbench, repository, "owner", git_bin=git_bin)
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] is True


def test_setup_checks_existing_fragment_before_creating_config(locations, git_bin):
    workbench, repository = locations
    directory = workbench / ".opencode/local/evolution"
    directory.mkdir(parents=True)
    fragment = directory / "opencode.fragment.json"
    fragment.write_text('{"mcp": {"custom": {"type": "remote"}}}', encoding="utf-8")
    before = fragment.read_bytes()
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        setup(workbench, repository, "owner", git_bin=git_bin)
    assert fragment.read_bytes() == before
    assert not (directory / "config.json").exists()
    assert not (directory / "state").exists()


def test_setup_rejects_a_file_at_the_future_store_location(locations, git_bin):
    workbench, repository = locations
    directory = workbench / ".opencode/local/evolution"
    directory.mkdir(parents=True)
    state = directory / "state"
    state.write_text("An unrelated local file", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory"):
        setup(workbench, repository, "owner", git_bin=git_bin)
    assert state.read_text(encoding="utf-8") == "An unrelated local file"
    assert not (directory / "config.json").exists()


def test_setup_atomic_publication_failure_leaves_no_partial_json(locations, git_bin, monkeypatch):
    workbench, repository = locations

    def unsupported_link(*args, **kwargs):
        raise OSError("Fixture filesystem does not support atomic publication")

    monkeypatch.setattr(os, "link", unsupported_link)
    with pytest.raises(OSError, match="atomic publication"):
        setup(workbench, repository, "owner", git_bin=git_bin)
    directory = workbench / ".opencode/local/evolution"
    assert not (directory / "config.json").exists()
    assert not list(directory.glob(".evolution-setup-*"))


def test_setup_can_resume_after_config_was_published_but_fragment_was_not(
    locations, git_bin, monkeypatch
):
    workbench, repository = locations
    original_link = os.link

    def fail_fragment(source, destination):
        if Path(destination).name == "opencode.fragment.json":
            raise OSError("Fixture interrupted after configuration publication")
        return original_link(source, destination)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(os, "link", fail_fragment)
        with pytest.raises(OSError, match="interrupted"):
            setup(workbench, repository, "owner", git_bin=git_bin)
    result = setup(workbench, repository, "owner", git_bin=git_bin)
    assert not result["config_created"] and result["fragment_created"]
    assert not (workbench / ".opencode/local/evolution/state").exists()


def test_setup_rejects_redirected_local_directory(locations, git_bin, tmp_path):
    workbench, repository = locations
    redirected = tmp_path / "outside-workbench"
    redirected.mkdir()
    try:
        (workbench / ".opencode/local").symlink_to(redirected, target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory symlinks requires platform privileges")
    with pytest.raises(ValueError, match="redirected"):
        setup(workbench, repository, "owner", git_bin=git_bin)
    assert not list(redirected.iterdir())


def test_setup_preserves_python_virtual_environment_launcher_path(locations, git_bin, tmp_path):
    workbench, repository = locations
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    try:
        launcher.symlink_to(sys.executable)
    except OSError:
        pytest.skip("Creating executable symlinks requires platform privileges")
    result = setup(workbench, repository, "owner", git_bin=git_bin, python_executable=str(launcher))
    fragment = json.loads(Path(result["fragment_path"]).read_text(encoding="utf-8"))
    command = fragment["mcp"]["hmopt_kernel_index"]["command"]
    assert command[0] == str(launcher.absolute())
    assert command[0] != str(launcher.resolve())
