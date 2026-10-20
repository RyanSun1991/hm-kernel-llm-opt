"""Add only Evolution scope to the existing platform YAML, preserving its other bytes."""

from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

from .configuration import kernel_path, platform_options, platform_root, read_platform
from .workspace import freeze_workspace


def find_platform(workbench: Path, explicit: Path | None) -> Path | None:
    if explicit is None and os.getenv("HMOPT_MCP_CONFIG", "").strip():
        explicit = Path(os.environ["HMOPT_MCP_CONFIG"])
    path = Path(explicit).absolute() if explicit else workbench / "configs/app.yaml"
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return None
    if explicit is not None or path.is_file():
        return path
    return None


def _append_section(path: Path, previous: bytes, section: dict) -> None:
    # No reserialization of existing platform fields, comments or model credentials.
    from .setup import _local_path

    _local_path(path.parent)
    if path.is_symlink():
        raise ValueError("Platform setup config is redirected")
    lock = path.with_name(path.name + ".evolution-setup.lock")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError(
            "Another setup owns this config; inspect its setup lock before retrying"
        ) from None
    os.close(descriptor)
    temporary = None
    try:
        if path.read_bytes() != previous:
            raise ValueError("Platform config changed during setup; reload before retrying")
        addition = yaml.safe_dump({"evolution": section}, sort_keys=False, allow_unicode=True)
        raw = previous.rstrip(b"\r\n") + b"\n\n" + addition.encode("utf-8")
        try:
            yaml.safe_load(raw)
        except yaml.YAMLError:
            raise ValueError(
                "Cannot append evolution to this YAML document; add the section inside its document boundaries"
            ) from None
        with tempfile.NamedTemporaryFile(
            prefix=".evolution-config-", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, path.stat().st_mode)
        if path.read_bytes() != previous:
            raise ValueError("Platform config changed during setup; reload before retrying")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        lock.unlink()


def setup_platform(
    workbench, path, source_root, workspace_id, projects, owner, *, git_bin, python_executable
):
    from .configuration import validate_options
    from .setup import _executable, _local_path, _matches_existing, _mcp_fragment, _publish_json

    before = path.read_bytes()
    raw = read_platform(path)
    base = platform_root(path)
    existing = raw.get("evolution")
    if "evolution" in raw and not isinstance(existing, dict):
        raise ValueError("Existing evolution section must be an object; inspect it before setup")
    section = deepcopy(existing or {})
    workspace = deepcopy(section.get("source_workspace") or {})
    scope = {}
    for name, relative in projects.items():
        project = deepcopy(workspace.get("projects", {}).get(name, {}))
        if relative is not None:
            project["path"] = relative
        project["owners"] = {"**": owner}
        scope[name] = project
    # Keep explicitly configured dependency definitions when repeating setup.
    for name in workspace.get("dependencies", []):
        if name not in scope:
            scope[name] = workspace["projects"][name]
    workspace.update(workspace_id=workspace_id, projects=scope, selected=list(projects))
    if source_root is not None:
        workspace["root"] = str(Path(source_root).absolute())
    section["source_workspace"] = workspace
    executable = _executable(git_bin, "Git")
    if git_bin != "git":
        section["git_bin"] = executable
    python_executable = _executable(python_executable, "Python")
    if base != workbench:
        section.setdefault("workspace_root", str(workbench / ".opencode/local/workspaces"))
    options = validate_options(platform_options(path, {**raw, "evolution": section}))
    manifest = freeze_workspace(options["source_workspace"], git_bin=options["git_bin"])
    if existing is not None and section != existing:
        raise ValueError(
            "Existing platform evolution section differs; update that section explicitly instead of overwriting it with setup defaults"
        )
    directory = workbench / ".opencode/local/evolution" / workspace_id
    for location in (
        directory,
        options["workspace_root"],
        options["artifacts_root"],
        options["root"],
    ):
        if location is not None:
            _local_path(Path(location))
    fragment_path = directory / "opencode.fragment.json"
    fragment = _mcp_fragment(python_executable, {"HMOPT_MCP_CONFIG": str(path)})
    _matches_existing(fragment_path, fragment)
    directory.mkdir(parents=True, exist_ok=True)
    for location in (options["workspace_root"], options["artifacts_root"]):
        if location is not None:
            Path(location).mkdir(parents=True, exist_ok=True)
    if existing is None:
        _append_section(path, before, section)
    _publish_json(fragment_path, fragment)
    first = manifest["projects"][0]
    return {
        "status": "setup_ready",
        "config_path": str(path),
        "fragment_path": str(fragment_path),
        "workbench": str(workbench),
        "repo_path": first["repo_path"],
        "profile": first["project_id"],
        "target_revision": first["revision"],
        "owner": owner,
        "manifest": manifest,
        "config_created": existing is None,
        "workflow_started": False,
        "next_steps": [
            "已复用平台 YAML；将片段中的 HMOPT_MCP_CONFIG 合并到现有主 MCP。remote 服务使用服务端可见路径。",
            f'python -m hmopt evolve --config "{path}" doctor',
            "启动 evolve serve，然后在 OpenCode 使用 /evolve-workspace start；已有角色、审批和设备门禁继续生效。",
        ],
    }


def single_project_inputs(path, repo):
    """Reuse a configured kernel checkout without guessing any additional repositories."""
    from .configuration import source_root

    raw = read_platform(path)
    base = platform_root(path)
    target = Path(repo).absolute() if repo is not None else Path(kernel_path(raw, base))
    inherited_root = bool(os.getenv("PROJECT_REPO_PATH") or os.getenv("KERNEL_WORKSPACE_PATH"))
    root = Path(source_root(base)) if inherited_root else target.parent
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError:
        raise ValueError("Selected kernel is outside PROJECT_REPO_PATH") from None
    return (
        (None, {"kernel": None})
        if repo is None and inherited_root
        else (root, {"kernel": relative})
    )
