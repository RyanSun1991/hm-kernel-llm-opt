"""Path and version helpers for indexing pipeline."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from hmopt.core.config import AppConfig
from hmopt.tools.git_tools import get_repo_state

from .types import IndexPaths

_VERSION_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(text: str) -> str:
    text = text.strip().lower().replace(" ", "-")
    text = _VERSION_SAFE.sub("-", text)
    return text.strip("-") or "project"


def sanitize_version(version: str) -> str:
    version = version.strip()
    return _VERSION_SAFE.sub("-", version) or "unknown"


def index_roots(config: AppConfig) -> tuple[Path, Path, Path]:
    base = Path(config.indexing.persist_dir)
    code_root = Path(config.indexing.code_index_root) if config.indexing.code_index_root else base / "code"
    runtime_root = (
        Path(config.indexing.runtime_index_root)
        if config.indexing.runtime_index_root
        else base / "runtime"
    )
    return base, code_root, runtime_root


def latest_subdir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_code_version(
    config: AppConfig, repo_path: Path, override: Optional[str] = None
) -> Optional[str]:
    if override:
        return sanitize_version(override)
    if config.indexing.code_index_version:
        return sanitize_version(config.indexing.code_index_version)
    scheme = (config.indexing.versioning_scheme or "git_commit").lower()
    if scheme in {"git_commit", "git", "commit"}:
        repo_state = get_repo_state(repo_path)
        commit = repo_state.get("commit")
        if commit:
            version = commit[:12]
            if repo_state.get("dirty") and config.indexing.include_dirty_suffix:
                version = f"{version}-dirty"
            return sanitize_version(version)
    if scheme in {"timestamp", "time"}:
        return datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return None


def resolve_runtime_version(
    config: AppConfig, run_id: str, override: Optional[str] = None
) -> Optional[str]:
    if override:
        return sanitize_version(override)
    if config.indexing.runtime_index_version:
        return sanitize_version(config.indexing.runtime_index_version)
    return sanitize_version(run_id)


def resolve_repo_config(
    config: AppConfig,
    *,
    repo_name: Optional[str] = None,
    repo_path: Optional[str] = None,
) -> tuple[str, Path, Optional[Path]]:
    if repo_path:
        path = Path(repo_path)
        name = repo_name or path.name
        return name, path, None

    repos = getattr(config.project, "repos", []) or []
    if repo_name and repos:
        for entry in repos:
            if entry.name == repo_name:
                return entry.name, Path(entry.repo_path), entry.compile_commands_dir
    if repo_name and not repos:
        return repo_name, Path(config.project.repo_path), None
    if repo_name and repos:
        return repo_name, Path(config.project.repo_path), None
    if repos:
        entry = repos[0]
        return entry.name, Path(entry.repo_path), entry.compile_commands_dir
    return config.project.name, Path(config.project.repo_path), None


def index_paths(
    config: AppConfig,
    *,
    repo_path: Optional[Path] = None,
    repo_name: Optional[str] = None,
    run_id: Optional[str] = None,
    code_version: Optional[str] = None,
    runtime_version: Optional[str] = None,
) -> IndexPaths:
    base, code_root, runtime_root = index_roots(config)
    repo_path = repo_path or Path(config.project.repo_path)
    repo_slug = slugify(repo_name or repo_path.name or config.project.name)

    resolved_code_version = resolve_code_version(config, repo_path, code_version)
    if resolved_code_version:
        code_dir = code_root / repo_slug / resolved_code_version
    else:
        code_dir = code_root / repo_slug

    resolved_runtime_version = None
    if run_id:
        resolved_runtime_version = resolve_runtime_version(config, run_id, runtime_version)
    elif runtime_version:
        resolved_runtime_version = sanitize_version(runtime_version)
    if resolved_runtime_version:
        runtime_dir = runtime_root / repo_slug / resolved_runtime_version
    else:
        runtime_dir = runtime_root / repo_slug

    # Legacy fallback if no versioned dir exists yet.
    if config.indexing.allow_legacy_paths:
        legacy_code = base / "code"
        if not code_dir.exists() and legacy_code.exists():
            code_dir = legacy_code
        legacy_runtime = base / "runtime"
        if not runtime_dir.exists() and legacy_runtime.exists():
            runtime_dir = legacy_runtime

    return IndexPaths(
        base_dir=base,
        code_root=code_root,
        runtime_root=runtime_root,
        code_dir=code_dir,
        runtime_dir=runtime_dir,
        code_version=resolved_code_version,
        runtime_version=resolved_runtime_version,
    )


def infer_version_from_dir(dir_path: Path, root: Path) -> Optional[str]:
    try:
        rel = dir_path.relative_to(root)
    except ValueError:
        return None
    if len(rel.parts) >= 2:
        return rel.parts[-1]
    return None


def select_existing_dir(primary: Path, root: Path) -> Path:
    if primary.exists():
        return primary
    latest = latest_subdir(root)
    if latest:
        return latest
    return primary
