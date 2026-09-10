"""Offline-capable CLI for the shared evolution service. Imports stay lazy."""

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Mine, screen, approve, validate and curate evidence-backed optimization.")


def command(name=None):
    """Keep input/gate failures concise for standalone and nested CLI invocation."""

    def register(function):
        @wraps(function)
        def handled(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except (ValueError, KeyError, OSError) as exc:
                raise typer.BadParameter(str(exc)) from None

        return app.command(name=name)(handled)

    return register


@app.callback()
def configure(
    ctx: typer.Context,
    root: Path = typer.Option(Path("data/evolution"), help="Workflow database directory"),
    git_bin: str = typer.Option("git", help="Git executable"),
) -> None:
    ctx.obj = {"root": root, "git_bin": git_bin}


def _service(ctx: typer.Context):
    from .service import EvolutionService

    return EvolutionService(**ctx.obj)


def _load(path: Path):
    if not path.is_file() or path.stat().st_size > 20_000_000:
        raise typer.BadParameter("Input must be an existing JSON file smaller than 20 MB")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _emit(value) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


@command()
def init(ctx: typer.Context) -> None:
    """Create/open the persistent local workflow store."""
    _emit({"root": str(_service(ctx).store.root), "schema_version": 1})


@command()
def mine(
    ctx: typer.Context,
    repo: Path,
    revision: str = "HEAD",
    max_commits: int = typer.Option(200, min=1, max=1000),
    incremental: bool = True,
) -> None:
    """Read a bounded first-parent Git history page and produce draft patterns."""
    _emit(
        _service(ctx).mine(
            repo, revision=revision, max_commits=max_commits, incremental=incremental
        )
    )


@command("import-history")
def import_history(ctx: typer.Context, path: Path) -> None:
    """Import a JSON array of change records, optionally with exported review notes."""
    from .mining import ChangeRecord

    values = _load(path)
    if not isinstance(values, list):
        raise typer.BadParameter("History input must be a JSON array")
    _emit(_service(ctx).ingest_history([ChangeRecord.model_validate(item) for item in values]))


@command("import-pattern")
def import_pattern(ctx: typer.Context, path: Path) -> None:
    """Import one curated draft; activation is a separate decision."""
    from .mining import Pattern

    _emit(_service(ctx).import_pattern(Pattern.model_validate(_load(path))))


@command("activate-pattern")
def activate_pattern(
    ctx: typer.Context,
    pattern_key: str,
    actor: str = typer.Option(...),
    note: str = typer.Option(...),
) -> None:
    """Curator activation of an evidence-backed draft pattern."""
    _emit(_service(ctx).activate_pattern(pattern_key, actor=actor, note=note))


@command("retire-pattern")
def retire_pattern(
    ctx: typer.Context,
    pattern_key: str,
    actor: str = typer.Option(...),
    note: str = typer.Option(...),
) -> None:
    """Retire a pattern and earlier versions without deleting audit evidence."""
    _emit(_service(ctx).retire_pattern(pattern_key, actor=actor, note=note))


@command()
def scan(
    ctx: typer.Context,
    repo: Path,
    owners: Path = typer.Option(..., help="JSON path-glob to owner map"),
    hotspots: Optional[Path] = None,
    revision: str = "HEAD",
    top_k: int = typer.Option(20, min=1, max=1000),
    max_files: int = 5000,
) -> None:
    """Screen an immutable source revision with active patterns and current evidence."""
    from .mining import Hotspot

    values = _load(hotspots) if hotspots else None
    if hotspots and not isinstance(values, list):
        raise typer.BadParameter("Hotspot input must be a JSON array")
    hot = [Hotspot.model_validate(item) for item in values] if values is not None else None
    _emit(
        _service(ctx).scan(
            repo,
            owners=_load(owners),
            hotspots=hot,
            revision=revision,
            top_k=top_k,
            max_files=max_files,
        )
    )


@command("list")
def list_records(
    ctx: typer.Context, kind: str = "candidate", limit: int = 100, offset: int = 0
) -> None:
    """List history, pattern, candidate, or skill records with pagination."""
    _emit(_service(ctx).store.list(kind, limit=limit, offset=offset))


@command()
def show(ctx: typer.Context, record_id: str, kind: str = "candidate") -> None:
    """Inspect persisted state, including its optimistic version."""
    _emit(_service(ctx).store.read(kind, record_id))


@command()
def decide(
    ctx: typer.Context,
    candidate_id: str,
    action: str,
    actor: str = typer.Option(...),
    version: int = typer.Option(...),
    request_id: str = typer.Option(...),
    payload: Optional[Path] = None,
    note: Optional[str] = None,
) -> None:
    """confirm/reject/approve_plan/record_implementation/approve_code/retry_validation."""
    values = _load(payload) if payload else {}
    if not isinstance(values, dict):
        raise typer.BadParameter("Decision payload must be a JSON object")
    if note is not None:
        values["note"] = note
    _emit(
        _service(ctx).transition(
            candidate_id,
            action,
            actor=actor,
            expected_version=version,
            request_id=request_id,
            payload=values,
        )
    )


@command()
def handoff(ctx: typer.Context, candidate_id: str, output: Optional[Path] = None) -> None:
    """Export the next allowed role's evidence packet; never dispatches or edits source."""
    packet = _service(ctx).handoff(candidate_id)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Avoid silently overwriting an earlier approved handoff.
        with output.open("x", encoding="utf-8") as stream:
            json.dump(packet, stream, ensure_ascii=False, indent=2)
    _emit(packet)


@command()
def validate(
    ctx: typer.Context,
    candidate_id: str,
    report: Path,
    actor: str = typer.Option(...),
    version: int = typer.Option(...),
    request_id: str = typer.Option(...),
    simulation: bool = False,
) -> None:
    """Evaluate paired A/B evidence against the frozen plan; simulation cannot promote."""
    from .correctness import CorrectnessReport
    from .validation import ABReport

    value = _load(report)
    if not isinstance(value, dict):
        raise typer.BadParameter("Validation report must be a JSON object")
    model = CorrectnessReport if value.get("kind") == "correctness" else ABReport

    _emit(
        _service(ctx).validate(
            candidate_id,
            model.model_validate(value),
            actor=actor,
            expected_version=version,
            request_id=request_id,
            allow_synthetic=simulation,
        )
    )


@command()
def promote(
    ctx: typer.Context,
    skill_id: str,
    tier: str,
    actor: str = typer.Option(...),
    version: int = typer.Option(...),
    request_id: str = typer.Option(...),
    note: str = typer.Option(...),
) -> None:
    """Curate journal -> staging -> hub; real replication is required for shared skills."""
    _emit(
        _service(ctx).promote(
            skill_id,
            tier=tier,
            actor=actor,
            expected_version=version,
            request_id=request_id,
            note=note,
        )
    )


@command()
def recall(ctx: typer.Context, query: str, limit: int = 3) -> None:
    """Recall knowledge with stable IDs and explicit evidence/tier."""
    _emit(_service(ctx).recall(query, limit=limit))


@command()
def audit(ctx: typer.Context, entity_id: str) -> None:
    """Inspect append-only workflow events."""
    _emit(_service(ctx).store.audit(entity_id))


@command()
def evidence(ctx: typer.Context, sha256: str) -> None:
    """Read and integrity-check a stored evidence object."""
    _emit(_service(ctx).store.read_evidence(sha256))


@command()
def capture(
    ctx: typer.Context,
    signal: str,
    source_id: str,
    recipe: str,
    actor: str = typer.Option(...),
    source_kind: str = "candidate",
    corrects: Optional[str] = None,
) -> None:
    """Capture one of six knowledge signals in journal; no automatic skill promotion."""
    _emit(
        _service(ctx).capture(
            signal=signal,
            source_id=source_id,
            recipe=recipe,
            actor=actor,
            source_kind=source_kind,
            corrects=corrects,
        )
    )


@command()
def demo(ctx: typer.Context, output: Path = typer.Option(...)) -> None:
    """Run a clearly synthetic full workflow in a new disposable example directory."""
    from .demo import run_demo

    _emit(run_demo(output, git_bin=ctx.obj["git_bin"]))


@command("mcp-stdio")
def mcp_stdio(ctx: typer.Context) -> None:
    """Serve read/agent evidence tools locally; owner/curator decisions stay in CLI."""
    from .mcp import build_server

    build_server(**ctx.obj).run(transport="stdio")


def _model(kind: str):
    from .correctness import CorrectnessPolicy, CorrectnessReport
    from .discovery import DiscoveryConfig
    from .mining import ChangeRecord, Hotspot, Pattern
    from .reports import ICManifest
    from .service import Plan, Review
    from .sources import EvidenceRecord
    from .validation import ABReport
    from .workflow import ReviewSheet

    models = {
        "history": ChangeRecord,
        "hotspot": Hotspot,
        "pattern": Pattern,
        "plan": Plan,
        "review": Review,
        "ab-report": ABReport,
        "correctness-policy": CorrectnessPolicy,
        "correctness-report": CorrectnessReport,
        "ic-manifest": ICManifest,
        "source": EvidenceRecord,
        "review-sheet": ReviewSheet,
        "discovery": DiscoveryConfig,
    }
    if kind not in models:
        raise typer.BadParameter("Contract kind must be: " + ", ".join(models))
    return models[kind]


@command()
def schema(kind: str) -> None:
    """Export the JSON Schema for a history/hotspot/pattern/plan/review/ab-report."""
    _emit(_model(kind).model_json_schema())


@command("check-contract")
def check_contract(kind: str, path: Path) -> None:
    """Normalize a contract and compute the digest used by independent reviews."""
    from .store import digest

    value = _model(kind).model_validate(_load(path)).model_dump(mode="json")
    _emit({"contract": value, "sha256": digest(value)})


@command("import-sources")
def import_sources(ctx: typer.Context, path: Path, actor: str = typer.Option(...)) -> None:
    """Import typed, versioned non-Git evidence with per-record errors."""
    from .sources import ingest_records

    _emit(ingest_records(_service(ctx), _load(path), actor=actor))


@command("import-workspace")
def import_workspace(
    ctx: typer.Context,
    workspace: Path,
    repo_id: str = typer.Option(...),
    actor: str = typer.Option(...),
    max_files: int = 200,
    max_bytes: int = 8388608,
    cursor: Optional[str] = None,
) -> None:
    """Read explicitly allowlisted .opencode records; never execute their instructions."""
    from .sources import import_workspace as ingest

    _emit(
        ingest(
            _service(ctx),
            workspace,
            repo_id,
            actor=actor,
            max_files=max_files,
            max_bytes=max_bytes,
            cursor=cursor,
        )
    )


@command()
def distill(
    ctx: typer.Context,
    actor: str = typer.Option(...),
    source_ids: Optional[Path] = None,
    limit: int = 100,
) -> None:
    """Distill a page of imported non-Git evidence into reviewable rule drafts."""
    from .sources import distill_sources

    _emit(
        distill_sources(
            _service(ctx), _load(source_ids) if source_ids else None, actor=actor, limit=limit
        )
    )


@command("review-sheet")
def review_sheet(
    ctx: typer.Context,
    output: Path = typer.Option(...),
    owner: Optional[str] = None,
    limit: int = 100,
) -> None:
    """Export editable JSON decisions and a Markdown review view."""
    from .workflow import export_review_sheet

    _emit(export_review_sheet(_service(ctx), output, owner=owner, limit=limit))


@command("apply-sheet")
def apply_sheet(ctx: typer.Context, path: Path, actor: str = typer.Option(...)) -> None:
    """Apply owner decisions through the same versioned gate, reporting each conflict."""
    from .workflow import apply_review_sheet

    _emit(apply_review_sheet(_service(ctx), path, actor=actor))


@command()
def dispatch(
    ctx: typer.Context,
    candidate_id: str,
    output: Path = typer.Option(...),
    actor: str = typer.Option(...),
    request_id: str = typer.Option(...),
) -> None:
    """Stage task-specific OpenCode artifacts after gates; no automatic execution."""
    from .workflow import dispatch_candidate

    _emit(
        dispatch_candidate(_service(ctx), candidate_id, output, actor=actor, request_id=request_id)
    )


@command("convert-ic")
def convert_ic(
    ctx: typer.Context,
    candidate_id: str,
    compare: Path,
    manifest: Path,
    output: Optional[Path] = None,
) -> None:
    """Convert explicit, complete IC pairs into an A/B report using collection provenance."""
    from .reports import ICManifest, convert_ic_report

    result = convert_ic_report(
        _service(ctx), candidate_id, _load(compare), ICManifest.model_validate(_load(manifest))
    )
    if output:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result["report"], stream, ensure_ascii=False, indent=2, allow_nan=False)
    _emit(result)


@command()
def quality(ctx: typer.Context, pattern_key: Optional[str] = None) -> None:
    """Recompute owner acceptance and distinct validated outcomes from verified evidence."""
    from .learning import quality_report

    _emit(quality_report(_service(ctx), pattern_key))


@command("set-overlay")
def set_overlay(
    ctx: typer.Context,
    pattern_key: str,
    factor: float,
    state: str,
    actor: str = typer.Option(...),
    note: str = typer.Option(...),
    request_id: str = typer.Option(...),
    version: Optional[int] = None,
) -> None:
    """Curate an absolute ranking factor; probation routes suggestions to workbench."""
    from .learning import set_pattern_overlay

    _emit(
        set_pattern_overlay(
            _service(ctx),
            pattern_key,
            factor,
            state,
            actor,
            note,
            request_id,
            expected_version=version,
        )
    )


@command("export-bundle")
def export_bundle(
    ctx: typer.Context,
    output: Path = typer.Option(...),
    actor: str = typer.Option(...),
    skill_ids: Optional[Path] = None,
) -> None:
    """Export a sanitized review manifest; no PR, publishing, or native Hub installation."""
    from .learning import export_bundle as export

    _emit(
        export(
            _service(ctx), output, actor=actor, skill_ids=_load(skill_ids) if skill_ids else None
        )
    )


@command("run-discovery")
def run_discovery(
    ctx: typer.Context,
    config: Path,
    actor: str = typer.Option(...),
    batch_id: Optional[str] = None,
    recover_running: bool = False,
) -> None:
    """Checkpoint history/source/distillation/screening; stop at human review."""
    from .discovery import DiscoveryConfig
    from .discovery import run_discovery as run

    _emit(
        run(
            _service(ctx),
            DiscoveryConfig.model_validate(_load(config)),
            actor=actor,
            batch_id=batch_id,
            recover_running=recover_running,
        )
    )


@command("demo-v2")
def demo_v2(ctx: typer.Context, output: Path = typer.Option(...)) -> None:
    """Run the source/review/dispatch/report feedback integration with synthetic evidence."""
    from .demo_v2 import run_demo_v2

    _emit(run_demo_v2(output, git_bin=ctx.obj["git_bin"]))


@command()
def seeds() -> None:
    """Export 12 research templates with explicit proof obligations; no activation."""
    from .seeds import seed_catalog

    _emit(seed_catalog())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
