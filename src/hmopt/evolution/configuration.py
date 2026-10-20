"""Reuse platform paths and OpenCode model selection without copying credentials."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


def platform_root(path: Path) -> Path:
    path = path.resolve()
    return path.parent.parent if path.parent.name == "configs" else path.parent


def read_platform(path: Path) -> dict:
    from hmopt.core.config import load_yaml

    if not path.is_file() or path.stat().st_size > 20_000_000:
        raise ValueError("Platform config must be an existing file smaller than 20 MB")
    try:
        raw = load_yaml(path)
    except yaml.YAMLError:
        raise ValueError("Invalid platform YAML; inspect the config locally") from None
    except AttributeError:
        raise ValueError("Platform config must be an object") from None
    if not isinstance(raw, dict):
        raise ValueError("Platform config must be an object")  # noqa: TRY004
    return raw


def absolute(value, base: Path) -> str:
    if not isinstance(value, str) or not value.strip() or "${" in value or "{env:" in value:
        raise ValueError(
            "A configured path is empty or contains an unresolved environment reference"
        )
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).absolute())


def source_root(base: Path, explicit=None) -> str:
    value = explicit or os.getenv("PROJECT_REPO_PATH") or os.getenv("KERNEL_WORKSPACE_PATH")
    if not value:
        raise ValueError(
            "Configure existing PROJECT_REPO_PATH or pass --source-root for the business workspace"
        )
    return absolute(str(value), base)


def kernel_path(raw: dict, base: Path) -> str:
    project = raw.get("project", {})
    if not isinstance(project, dict):
        raise ValueError("Platform project section must be an object")  # noqa: TRY004
    value = os.getenv("KERNEL_REPO_PATH") or project.get("repo_path")
    if not value:
        raise ValueError(
            "Configure existing KERNEL_REPO_PATH or project.repo_path, or pass an explicit repository"
        )
    return absolute(value, base)


def _jsonc(text: str) -> dict:
    # Keep strings verbatim: URLs and credential strings may contain comment tokens.
    strings = r'("(?:\\.|[^"\\])*")'
    text = re.sub(
        strings + r"|/\*.*?\*/|//[^\r\n]*", lambda m: m.group(1) or "", text, flags=re.DOTALL
    )
    text = re.sub(strings + r"|,\s*(?=[}\]])", lambda m: m.group(1) or "", text)
    try:
        value = json.loads(text)
    except ValueError:
        raise ValueError(
            "Cannot read OpenCode JSON/JSONC; inspect the configuration locally"
        ) from None
    if not isinstance(value, dict):
        raise ValueError("OpenCode config must be an object")  # noqa: TRY004
    return value


def opencode_defaults(workbench: Path) -> dict:
    """Read project and explicitly supplied config; never return provider options/secrets."""
    from hmopt.core.config import deep_merge

    raw = {}
    paths = [workbench / "opencode.json", workbench / "opencode.jsonc"]
    explicit = None
    if os.getenv("OPENCODE_CONFIG", "").strip():
        explicit = Path(os.environ["OPENCODE_CONFIG"].strip()).expanduser()
        paths.append(explicit)
    for path in dict.fromkeys(paths):
        if not path.exists():
            if path == explicit:
                raise ValueError("Explicit OPENCODE_CONFIG is missing")
            continue
        if not path.is_file() or path.stat().st_size > 2_000_000:
            raise ValueError("OpenCode config must be a bounded regular file")
        raw = deep_merge(raw, _jsonc(path.read_text(encoding="utf-8-sig")))
    if os.getenv("OPENCODE_CONFIG_CONTENT", "").strip():
        text = os.environ["OPENCODE_CONFIG_CONTENT"]
        if len(text) > 2_000_000:
            raise ValueError("OPENCODE_CONFIG_CONTENT exceeds the local read budget")
        raw = deep_merge(raw, _jsonc(text))
    result = {}
    agents = raw.get("agent") or {}
    researcher = agents.get("researcher", {}) if isinstance(agents, dict) else {}
    model = (researcher.get("model") if isinstance(researcher, dict) else None) or raw.get("model")
    if isinstance(model, str):
        model = re.sub(
            r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.getenv(m.group(1), ""), model
        )
        provider, sep, name = model.partition("/")
        if sep and provider and name:
            result.update(provider_id=provider, model_id=name)
    server = raw.get("server", {})
    if isinstance(server, dict) and type(server.get("port")) is int and 0 < server["port"] <= 65535:
        host = server.get("hostname", "127.0.0.1")
        if host in {"0.0.0.0", "127.0.0.1", "localhost", "::", "::1"}:
            host = "[::1]" if host in {"::", "::1"} else "127.0.0.1"
            result["url"] = f"http://{host}:{server['port']}"
    return result


def platform_options(path: Path, raw: dict | None = None) -> dict | None:
    """Expand only an explicit evolution section, leaving ordinary index startup lazy."""
    raw = read_platform(path) if raw is None else raw
    if raw.get("evolution") is None:
        return None
    if not isinstance(raw["evolution"], dict):
        raise ValueError("Platform evolution section must be an object")  # noqa: TRY004
    value = deepcopy(raw["evolution"])
    base = platform_root(path)
    storage = raw.get("storage", {})
    if not isinstance(storage, dict) or not isinstance(storage.get("artifacts", {}), dict):
        raise ValueError("Platform storage/artifacts sections must be objects")  # noqa: TRY004
    artifacts = absolute(
        storage.get("artifacts_root")
        or storage.get("artifacts", {}).get("root_dir", "data/artifacts"),
        base,
    )
    value.setdefault("schema_version", 1)
    value.setdefault("artifacts_root", artifacts)
    value.setdefault("root", str(Path(artifacts).parent / "evolution"))
    value.setdefault("workspace_root", str(base / ".opencode/local/workspaces"))
    for name in ("root", "workspace_root", "artifacts_root"):
        if value[name] is not None:
            value[name] = absolute(value[name], base)
    workspace = value.get("source_workspace")
    if isinstance(workspace, dict):
        workspace["root"] = source_root(base, workspace.get("root"))
        projects = workspace.get("projects", {})
        if not isinstance(projects, dict):
            raise ValueError("Workspace projects must be an object keyed by project ID")  # noqa: TRY004
        for name, project in projects.items():
            if isinstance(project, dict) and "path" not in project and name == "kernel":
                checkout = Path(kernel_path(raw, base))
                try:
                    project["path"] = checkout.relative_to(workspace["root"]).as_posix()
                except ValueError:
                    raise ValueError(
                        "Configured kernel checkout is outside the business workspace"
                    ) from None
    production = value.get("production", {})
    if isinstance(production, dict) and isinstance(production.get("worker"), dict):
        worker = production["worker"]
        workbench = base
        if value["workspace_root"]:
            root = Path(value["workspace_root"])
            if root.parts[-3:] == (".opencode", "local", "workspaces"):
                workbench = root.parents[2]
        worker.setdefault("directory", str(workbench))
        worker["directory"] = absolute(worker["directory"], base)
        if any(k not in worker for k in ("provider_id", "model_id", "url")):
            defaults = opencode_defaults(Path(worker["directory"]))
            # A partially explicit model must never be paired with a different provider.
            if "provider_id" not in worker and "model_id" not in worker:
                for key in ("provider_id", "model_id"):
                    if key in defaults:
                        worker[key] = defaults[key]
            if "url" not in worker and "url" in defaults:
                worker["url"] = defaults["url"]
    if isinstance(production, dict) and isinstance(production.get("validation"), dict):
        validation = production["validation"]
        validation.setdefault(
            "cwd", workspace["root"] if isinstance(workspace, dict) else str(base)
        )
        validation["cwd"] = absolute(validation["cwd"], base)
    return value


def selected_config(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    for name in ("HMOPT_EVOLUTION_CONFIG", "HMOPT_MCP_CONFIG"):
        if os.getenv(name, "").strip():
            return Path(os.environ[name].strip())
    # Preserve a previous local setup when no deployment selected a configuration.
    for path in (Path(".opencode/local/evolution/config.json"), Path("configs/app.yaml")):
        if path.is_file():
            return path
    return None


class EvolutionMCPConfig(BaseModel):
    """Single operator-owned connection profile; no credentials or approval bypasses."""

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    root: str = Field(min_length=1)
    git_bin: str = Field(default="git", min_length=1)
    workspace_root: str | None = None
    artifacts_root: str | None = None
    discovery_profiles: dict = Field(default_factory=dict)
    production: dict = Field(default_factory=dict)
    source_workspace: dict | None = None


def load_evolution_config(path: Path) -> dict:
    """Read and validate one configuration without creating its workflow store."""
    if path.suffix.lower() in {".yaml", ".yml"}:
        values = platform_options(path)
        if values is None:
            return _default_options()
    elif not path.is_file() or path.stat().st_size > 20_000_000:
        raise ValueError("Evolution config must be an existing JSON file smaller than 20 MB")
    else:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(values, dict) and "evolution" in values:
            values = platform_options(path, values)
            if values is None:
                return _default_options()
    return validate_options(values)


def validate_options(values: dict) -> dict:
    from hmopt.evolution.mcp_discovery import prepare_profiles

    if not isinstance(values, dict) or type(values.get("schema_version")) is not int:
        raise ValueError("Evolution config must be an object with integer schema_version=1")
    config = EvolutionMCPConfig.model_validate(values)
    options = config.model_dump(exclude={"schema_version"})
    from hmopt.evolution.production import production_config

    options["production"] = production_config(options["production"]).model_dump(
        mode="json", exclude_none=True
    )
    for name in ("root", "workspace_root", "artifacts_root"):
        if options[name] is not None:
            location = Path(options[name])
            if not location.is_absolute():
                raise ValueError(f"Evolution config {name} must be an absolute path")
            options[name] = location.resolve()
    from hmopt.evolution.workspace import merge_profiles

    profiles = prepare_profiles(
        merge_profiles(options["discovery_profiles"], options["source_workspace"])
    )
    options["discovery_profiles"] = {
        name: profile.model_dump(mode="json") for name, profile in profiles.items()
    }
    return options


def _default_options() -> dict:
    return {
        "root": "data/evolution",
        "git_bin": "git",
        "workspace_root": None,
        "artifacts_root": None,
        "discovery_profiles": None,
    }


def read_environment_options(config_path: Path | None = None) -> dict:
    """Load operator configuration once while building the unified MCP registry."""

    def optional_path(name):
        value = os.getenv(name, "").strip()
        return Path(value) if value else None

    selected = selected_config(config_path)
    # A missing index-only config must not prevent legacy Evolution-only tool listing.
    explicit = config_path is not None or optional_path("HMOPT_EVOLUTION_CONFIG") is not None
    options = (
        load_evolution_config(selected)
        if selected and (explicit or selected.is_file())
        else _default_options()
    )
    # Explicit advanced environment overrides preserve existing deployments.
    for name, variable in (
        ("root", "HMOPT_EVOLUTION_ROOT"),
        ("git_bin", "HMOPT_EVOLUTION_GIT_BIN"),
        ("workspace_root", "HMOPT_EVOLUTION_WORKSPACE_ROOT"),
        ("artifacts_root", "HMOPT_EVOLUTION_ARTIFACTS_ROOT"),
    ):
        value = os.getenv(variable, "").strip()
        if value:
            options[name] = Path(value) if name.endswith("_root") else value
    profiles_path = optional_path("HMOPT_EVOLUTION_DISCOVERY_PROFILES")
    if profiles_path is not None:
        if not profiles_path.is_file() or profiles_path.stat().st_size > 20_000_000:
            raise ValueError("Discovery profiles must be an existing JSON file smaller than 20 MB")
        options["discovery_profiles"] = json.loads(profiles_path.read_text(encoding="utf-8-sig"))
    return options
