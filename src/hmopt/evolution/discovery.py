"""Checkpointed local discovery batches which stop at human review.

The budget governs scheduling between bounded stages, not preemption of a running
Git command. A batch never confirms, dispatches, implements or validates a target.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .mining import Hotspot
from .store import ConflictError, digest


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    repo_path: str = Field(min_length=1, max_length=4096)
    owners: dict[str, str] = Field(max_length=1024)
    revision: str = "HEAD"
    workspace: str | None = None
    repo_id: str | None = None
    hotspots: list[Hotspot] = Field(default_factory=list, max_length=10000)
    page_size: int = Field(default=200, ge=1, le=1000)
    max_pages: int = Field(default=5, ge=1, le=100)
    source_page_size: int = Field(default=200, ge=1, le=1000)
    source_page_bytes: int = Field(default=8 * 1024 * 1024, ge=1, le=16 * 1024 * 1024)
    max_files: int = Field(default=5000, ge=1, le=50000)
    top_k: int = Field(default=20, ge=1, le=1000)
    scheduling_budget_s: int = Field(default=120, ge=1, le=3600)

    @model_validator(mode="after")
    def source_identity(self) -> DiscoveryConfig:
        if self.workspace and not self.repo_id:
            raise ValueError("Workspace ingestion requires an explicit stable repo_id")
        return self


def run_discovery(
    service,
    config: DiscoveryConfig,
    *,
    actor: str,
    batch_id: str | None = None,
    recover_running: bool = False,
) -> dict:
    from .sources import distill_sources, import_workspace

    actor = service._actor(actor)
    config = DiscoveryConfig.model_validate(config.model_dump(mode="python"))
    parameters = config.model_dump(mode="json")
    supplied_id = batch_id is not None
    batch_id = batch_id or "discovery-" + uuid.uuid4().hex
    if not batch_id or len(batch_id) > 160:
        raise ValueError("Batch ID must be 1..160 characters")
    with service.store.transaction() as db:
        existing = db.execute(
            "SELECT 1 FROM records WHERE kind='batch' AND id=?", (batch_id,)
        ).fetchone()
        if existing:
            row = service.store.get(db, "batch", batch_id)
            data = row["data"]
            if data["config_digest"] != digest(parameters):
                raise ConflictError("Batch configuration changed; start a new batch")
            if data["status"] in {"awaiting_review", "requires_attention", "requires_partition"}:
                return row
            if data["status"] == "running" and not recover_running:
                raise ConflictError(
                    "Batch is running; explicit recovery is required after checking the old worker"
                )
            data.update(status="running", generation=data["generation"] + 1, actor=actor)
            row = service.store.put(db, "batch", batch_id, data, row["version"])
        else:
            if supplied_id:
                raise ValueError("Unknown discovery batch; omit batch_id to start one")
            data = {
                "schema_version": 1,
                "status": "running",
                "generation": 1,
                "config": parameters,
                "config_digest": digest(parameters),
                "actor": actor,
                "target_revision": None,
                "completed_stages": [],
                "history_pages": [],
                "source_pages": [],
                "distill_page": 0,
                "errors": [],
                "results": {},
                "source_changes_allowed": False,
            }
            row = service.store.put(db, "batch", batch_id, data)
        generation = data["generation"]
        service.store.event(db, batch_id, "discovery_claim", actor, {"generation": generation})
    started = time.monotonic()

    def save(status="running"):
        nonlocal row
        with service.store.transaction() as db:
            current = service.store.get(db, "batch", batch_id)
            if current["data"]["generation"] != generation or current["version"] != row["version"]:
                raise ConflictError("Discovery worker was superseded; discard its completion")
            data["status"] = status
            row = service.store.put(db, "batch", batch_id, data, row["version"])
        return row

    def budget_exhausted():
        return time.monotonic() - started >= config.scheduling_budget_s

    stage = "resolve_revision"
    try:
        if data["target_revision"] is None:
            data["target_revision"] = service._revision(config.repo_path, config.revision)
            save()
        stage = "mine"
        if stage not in data["completed_stages"]:
            for _ in range(config.max_pages):
                if budget_exhausted():
                    return save("partial")
                page = service.mine(
                    config.repo_path, revision=data["target_revision"], max_commits=config.page_size
                )
                data["history_pages"].append(page)
                if page["caught_up"]:
                    data["completed_stages"].append(stage)
                save()
                if page["caught_up"]:
                    break
            if stage not in data["completed_stages"]:
                return save("partial")
        stage = "sources"
        if stage not in data["completed_stages"]:
            aggregate = data["results"].setdefault(
                stage, {"imported": [], "existing": [], "source_ids": [], "next_cursor": None}
            )
            for _ in range(config.max_pages):
                if budget_exhausted():
                    return save("partial")
                result = (
                    import_workspace(
                        service,
                        Path(config.workspace),
                        config.repo_id,
                        actor=actor,
                        max_files=config.source_page_size,
                        max_bytes=config.source_page_bytes,
                        cursor=aggregate.get("next_cursor"),
                    )
                    if config.workspace
                    else {"imported": [], "existing": [], "partial": False, "has_more": False}
                )
                if aggregate.get("manifest_id") and aggregate["manifest_id"] != result.get(
                    "manifest_id"
                ):
                    raise ValueError("Source snapshot changed across pages; start a new batch")
                data.setdefault("source_pages", []).append(result)
                for key in ("imported", "existing"):
                    aggregate[key] = list(dict.fromkeys(aggregate[key] + result.get(key, [])))
                aggregate["source_ids"] = list(
                    dict.fromkeys(aggregate["imported"] + aggregate["existing"])
                )
                aggregate.update(
                    {
                        key: result.get(key)
                        for key in (
                            "manifest_id",
                            "manifest_files",
                            "next_cursor",
                            "next_offset",
                            "partial",
                            "has_more",
                        )
                    }
                )
                if result.get("requires_partition"):
                    data["next_action"] = (
                        "The source snapshot exceeded a hard bound. Partition the workspace and start a new batch; retrying this batch cannot advance."
                    )
                    return save("requires_partition")
                if result.get("errors") or result.get("requires_attention"):
                    data["errors"].append(
                        {
                            "stage": stage,
                            "errors": result.get("errors", []),
                            "generation": generation,
                        }
                    )
                    data["next_action"] = (
                        "Repair invalid or changed source documents (or increase the source page byte budget), then start a new batch with a fresh snapshot."
                    )
                    return save("requires_attention")
                if not result.get("has_more"):
                    data["completed_stages"].append(stage)
                    save()
                    break
                if not result.get("next_cursor"):
                    data["next_action"] = (
                        "The source page has no continuation cursor. Partition its input and start a new batch."
                    )
                    return save("requires_partition")
                save()
            if stage not in data["completed_stages"]:
                return save("partial")

        stage = "distill"
        if stage not in data["completed_stages"]:
            aggregate = data["results"].setdefault(
                stage,
                {
                    "processed": [],
                    "draft_patterns": [],
                    "skipped": [],
                    "errors": [],
                    "partial": False,
                },
            )
            # Reuse import page boundaries: each page already obeys the 16 MiB
            # distillation budget, even when individual documents vary in size.
            pages = data.get("source_pages", [])
            if not pages and data["results"]["sources"].get("source_ids"):
                data["next_action"] = (
                    "Legacy source checkpoint has no page boundaries; start a new batch."
                )
                return save("requires_attention")
            for _ in range(config.max_pages):
                index = data.setdefault("distill_page", 0)
                if index >= len(pages):
                    data["completed_stages"].append(stage)
                    save()
                    break
                if budget_exhausted():
                    return save("partial")
                source_page = pages[index]
                ids = list(
                    dict.fromkeys(source_page.get("imported", []) + source_page.get("existing", []))
                )
                result = (
                    distill_sources(service, ids, actor=actor, limit=1000)
                    if ids
                    else {
                        "processed": [],
                        "draft_patterns": [],
                        "skipped": [],
                        "errors": [],
                        "partial": False,
                    }
                )
                for key in ("processed", "draft_patterns"):
                    aggregate[key] = list(dict.fromkeys(aggregate[key] + result.get(key, [])))
                aggregate["skipped"].extend(result.get("skipped", []))
                aggregate["errors"].extend(result.get("errors", []))
                if (
                    result.get("partial")
                    or result.get("has_more")
                    or set(result.get("processed", [])) != set(ids)
                ):
                    aggregate["partial"] = True
                    data["next_action"] = (
                        "A source page could not be distilled completely; inspect its errors and repair evidence before starting a new batch."
                    )
                    return save("requires_attention")
                data["distill_page"] = index + 1
                save()
            if data["distill_page"] >= len(pages) and stage not in data["completed_stages"]:
                data["completed_stages"].append(stage)
                save()
            if stage not in data["completed_stages"]:
                return save("partial")

        stage = "history_analysis"
        if stage not in data["completed_stages"]:
            from .change_analysis import analysis_backlog

            result = analysis_backlog(service, config.repo_path)
            data["results"][stage] = result
            if result["unresolved"]:
                data["next_action"] = (
                    "Use evolution_history_analysis and evolution_submit_history_analysis in "
                    "OpenCode to analyze actual before/after code. Then resume this batch. "
                    "Missing context stays unresolved; commit titles never select patterns."
                )
                return save("awaiting_analysis")
            data["completed_stages"].append(stage)
            save()

        stage = "scan"
        if stage not in data["completed_stages"]:
            if budget_exhausted():
                return save("partial")
            result = service.scan(
                config.repo_path,
                owners=config.owners,
                hotspots=config.hotspots,
                revision=data["target_revision"],
                top_k=config.top_k,
                max_files=config.max_files,
            )
            data["results"][stage] = result
            if result.get("partial") or result.get("has_more"):
                data["next_action"] = (
                    "The bounded scan did not cover the whole target. Review available candidates, then use a larger bound or partitioned input in a new batch."
                )
                return save("requires_partition")
            data["completed_stages"].append(stage)
            save()
        data["next_action"] = (
            "Curate draft patterns and review discovered candidates; this batch grants no execution authority."
        )
        return save("awaiting_review")
    except ConflictError:
        raise
    except (ValueError, OSError, TimeoutError) as exc:
        data["errors"].append({"stage": stage, "error": str(exc)[:4000], "generation": generation})
        return save("failed")
