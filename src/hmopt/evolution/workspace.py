"""Explicit multi-Git workspaces; a workspace root is not assumed to be a Git root."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .discovery import DiscoveryConfig
from .mining import Hotspot, _digest, _owners, _path, read_git


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    owners: dict[str, str] = Field(min_length=1, max_length=1024)
    revision: str = "HEAD"
    hotspots: list[Hotspot] = Field(default_factory=list, max_length=10000)

    @field_validator("path")
    @classmethod
    def relative_path(cls, value):
        return _path(value)

    @field_validator("owners")
    @classmethod
    def owner_rules(cls, value):
        _owners(value)
        return value


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    workspace_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    root: str
    projects: dict[str, Project] = Field(min_length=1, max_length=256)
    selected: list[str] = Field(min_length=1, max_length=32)
    dependencies: list[str] = Field(default_factory=list, max_length=256)

    @model_validator(mode="after")
    def scope(self):
        if not Path(self.root).is_absolute():
            raise ValueError("Workspace root must be absolute")
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", p) for p in self.projects):
            raise ValueError("Project IDs must be 1..64 letters, digits, '-' or '_'")
        if (
            len(self.selected) != len(set(self.selected))
            or not set(self.selected) <= self.projects.keys()
        ):
            raise ValueError("Select distinct registered project IDs")
        if (
            len(self.dependencies) != len(set(self.dependencies))
            or not set(self.dependencies) <= self.projects.keys()
            or set(self.dependencies) & set(self.selected)
        ):
            raise ValueError(
                "Dependencies must be distinct registered projects outside the mining selection"
            )
        paths = [p.path.casefold() for p in self.projects.values()]
        if len(paths) != len(set(paths)):
            raise ValueError("A checkout path cannot be registered as multiple projects")
        return self


def project_path(config: WorkspaceConfig, name: str) -> Path:
    if name not in config.selected and name not in config.dependencies:
        raise ValueError("Project is not in the operator-selected workspace scope")
    root = Path(config.root)
    if not root.is_dir() or root.resolve() != root:
        raise ValueError("Workspace root is missing or redirected")
    path = root / config.projects[name].path
    if not path.is_dir() or path.resolve() != path or not path.is_relative_to(root):
        raise ValueError("Selected checkout is missing, redirected or outside workspace root")
    return path


def workspace_profiles(value: dict | None) -> dict:
    """Compile selected projects into legacy-compatible profiles without touching Git/state."""
    if value is None:
        return {}
    config = WorkspaceConfig.model_validate(value)
    return {
        name: DiscoveryConfig(
            repo_path=str(project_path(config, name)),
            repo_id=f"{config.workspace_id}/{name}",
            owners=config.projects[name].owners,
            revision=config.projects[name].revision,
            hotspots=config.projects[name].hotspots,
        ).model_dump(mode="json")
        for name in config.selected
    }


def identities(value: dict | None) -> dict[str, str]:
    if value is None:
        return {}
    config = WorkspaceConfig.model_validate(value)
    return {
        str(project_path(config, name)): _digest([config.workspace_id, name], "repo_")
        for name in config.selected
    }


def merge_profiles(profiles: dict | None, workspace: dict | None) -> dict:
    generated = workspace_profiles(workspace)
    supplied = dict(profiles or {})
    for name, value in generated.items():
        if (
            name in supplied
            and DiscoveryConfig.model_validate(supplied[name]).model_dump(mode="json") != value
        ):
            raise ValueError("Workspace project conflicts with a discovery profile: " + name)
        supplied[name] = value
    # Workspace mode has one explicit allowlist, including its legacy tool surface.
    if workspace and set(supplied) != set(generated):
        raise ValueError("Workspace mode exposes only selected project profiles")
    return supplied


def freeze_workspace(value: dict, *, git_bin="git", selected=None) -> dict:
    config = WorkspaceConfig.model_validate(value)
    names = config.selected if selected is None else selected
    if not names or len(names) != len(set(names)) or not set(names) <= set(config.selected):
        raise ValueError("Run projects must be a nonempty subset of the selected projects")
    projects = []
    dependencies = []
    # A subset research request still freezes the other configured projects as
    # build context. Only names receive research steps/candidates in this run.
    for name in [*config.selected, *config.dependencies]:
        path = project_path(config, name)
        top = Path(
            read_git(path, ["rev-parse", "--show-toplevel"], git_bin=git_bin).decode().strip()
        ).resolve()
        if top != path:
            raise ValueError("Selected project must be an exact Git checkout root: " + name)
        revision = (
            read_git(
                path,
                [
                    "rev-parse",
                    "--verify",
                    "--end-of-options",
                    config.projects[name].revision + "^{commit}",
                ],
                git_bin=git_bin,
                max_bytes=128,
            )
            .decode("ascii")
            .strip()
        )
        item = {
            "project_id": name,
            "repo_id": _digest([config.workspace_id, name], "repo_"),
            "relative_path": config.projects[name].path,
            "repo_path": str(path),
            "revision": revision,
            "revision_selector": config.projects[name].revision,
            "owners": config.projects[name].owners,
            "hotspots": [h.model_dump(mode="json") for h in config.projects[name].hotspots],
        }
        (projects if name in names else dependencies).append(item)
    return {
        "schema_version": 1,
        "workspace_id": config.workspace_id,
        "workspace_root": config.root,
        "projects": projects,
        "dependencies": dependencies,
        "scope": "Only listed Git projects; unselected projects are not inspected or modified.",
    }
