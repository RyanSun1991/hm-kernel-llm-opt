"""Operator-registered discovery profiles for the interactive MCP surface."""

from __future__ import annotations

import re
from pathlib import Path

from .discovery import DiscoveryConfig


def prepare_profiles(values: dict | None) -> dict[str, DiscoveryConfig]:
    """Copy and freeze configured locations; tool inputs never supply filesystem roots."""
    if values is None:
        return {}
    if not isinstance(values, dict) or len(values) > 32:
        raise ValueError("Discovery profiles must be an object with at most 32 named profiles")
    profiles = {}
    for name, value in values.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("Discovery profile names must be 1..64 letters, digits, '-' or '_'")
        config = DiscoveryConfig.model_validate(value)
        parameters = config.model_dump(mode="json")
        for key in ("repo_path", "workspace"):
            if parameters[key] is None:
                continue
            path = Path(parameters[key])
            if not path.is_absolute() or not path.is_dir():
                raise ValueError(f"Profile {name}: {key} must be an existing absolute directory")
            parameters[key] = str(path.resolve())
        profiles[name] = DiscoveryConfig.model_validate(parameters)
    return profiles


def select_profile(service, profiles: dict[str, DiscoveryConfig], name: str) -> DiscoveryConfig:
    if name not in profiles:
        raise ValueError(
            "Unknown discovery profile; run setup or configure selected projects in the platform evolution section"
        )
    config = profiles[name].model_copy(deep=True)
    for location in (config.repo_path, config.workspace):
        if location is None:
            continue
        path = Path(location)
        if not path.is_dir() or path.resolve() != path:
            raise ValueError("Configured discovery directory is missing or has been redirected")
    top = Path(service._git(config.repo_path, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != Path(config.repo_path):
        raise ValueError(
            "Discovery repo_path must name the Git repository root, not a subdirectory"
        )
    return config


def discovery_step(
    service,
    config: DiscoveryConfig,
    step: str,
    *,
    actor: str,
    cursor: str | None = None,
    source_ids: list[str] | None = None,
) -> dict:
    """Run one bounded operation without advancing a discovery batch or owner gate."""
    from .sources import distill_sources, import_workspace

    actor = service._actor(actor)
    if step not in {"mine", "sources", "distill", "scan"}:
        raise ValueError("Discovery step must be mine, sources, distill or scan")
    if cursor is not None and step != "sources":
        raise ValueError("cursor is only accepted for the sources step")
    if source_ids is not None and step != "distill":
        raise ValueError("source_ids is only accepted for the distill step")
    if step == "sources":
        if not config.workspace or not config.repo_id:
            raise ValueError("The discovery profile must configure workspace and repo_id")
        return import_workspace(
            service,
            Path(config.workspace),
            config.repo_id,
            actor=actor,
            max_files=config.source_page_size,
            max_bytes=config.source_page_bytes,
            cursor=cursor,
        )
    if step == "distill":
        if (
            not config.repo_id
            or not isinstance(source_ids, list)
            or not 1 <= len(source_ids) <= config.source_page_size
        ):
            raise ValueError(
                "Distill requires a configured repo_id and nonempty source_ids within the profile page limit"
            )
        # Check the complete input before distillation can write any drafts.
        for source_id in source_ids:
            if not isinstance(source_id, str) or not re.fullmatch(
                r"source_[0-9a-f]{64}", source_id
            ):
                raise ValueError("source_ids must contain namespaced source IDs")
            record = service.store.read("source", source_id)["data"]["record"]
            if record["repo_id"] != config.repo_id:
                raise ValueError("Source belongs to a different repo_id than the discovery profile")
        return distill_sources(service, source_ids, actor=actor, limit=config.source_page_size)
    target = service._revision(config.repo_path, config.revision)
    if step == "mine":
        result = service.mine(config.repo_path, revision=target, max_commits=config.page_size)
    else:
        result = service.scan(
            config.repo_path,
            owners=config.owners,
            hotspots=config.hotspots,
            revision=target,
            top_k=config.top_k,
            max_files=config.max_files,
        )
    return {**result, "target_revision": target}
