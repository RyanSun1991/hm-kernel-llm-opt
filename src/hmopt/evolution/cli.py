"""Offline-capable CLI for the shared evolution service. Imports stay lazy."""

# Typer consumes these descriptors; they are not mutable runtime default values.
# ruff: noqa: B008

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path

import typer

app = typer.Typer(
    help="Start with setup and doctor, then use /evolve-discover in OpenCode. Advanced workflow commands follow."
)


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
    config: Path | None = typer.Option(
        None,
        envvar="HMOPT_EVOLUTION_CONFIG",
        help="Platform YAML (HMOPT_MCP_CONFIG) or legacy Evolution JSON",
    ),
) -> None:
    from .configuration import selected_config

    options = {}
    # Doctor must report even a malformed explicit config as a structured,
    # read-only diagnosis instead of failing early in the shared callback.
    if ctx.invoked_subcommand not in {"doctor", "setup", "setup-workspace"}:
        from .configuration import read_environment_options

        try:
            options = read_environment_options(config)
        except (ValueError, OSError) as exc:
            from .doctor import _message

            raise typer.BadParameter(_message(exc), param_hint="--config") from None
        # Some Typer releases vendor Click's enum; compare its stable member
        # name instead of mixing two distinct ParameterSource classes.
        if getattr(ctx.get_parameter_source("root"), "name", None) == "DEFAULT":
            root = options["root"]
        if getattr(ctx.get_parameter_source("git_bin"), "name", None) == "DEFAULT":
            git_bin = options["git_bin"]
    ctx.meta["evolution_config"] = selected_config(config)
    import os

    ctx.meta["evolution_setup_config"] = config or (
        Path(os.environ["HMOPT_MCP_CONFIG"]) if os.getenv("HMOPT_MCP_CONFIG", "").strip() else None
    )
    ctx.meta["evolution_options"] = options
    ctx.obj = {"root": root, "git_bin": git_bin}


def _service(ctx: typer.Context):
    from .service import EvolutionService
    from .workspace import identities

    return EvolutionService(
        **ctx.obj,
        workspace_root=ctx.meta.get("evolution_options", {}).get("workspace_root"),
        repo_identities=identities(ctx.meta.get("evolution_options", {}).get("source_workspace")),
        source_workspace=ctx.meta.get("evolution_options", {}).get("source_workspace"),
    )


def _load(path: Path):
    if not path.is_file() or path.stat().st_size > 20_000_000:
        raise typer.BadParameter("Input must be an existing JSON file smaller than 20 MB")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _emit(value) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


@command()
def setup(
    ctx: typer.Context,
    repo: Path | None = typer.Option(
        None, help="Defaults to existing KERNEL_REPO_PATH or project.repo_path"
    ),
    owner: str = typer.Option(..., help="Explicit owner of this pilot repository scope"),
    workbench: Path = typer.Option(Path("."), help="OpenCode workbench project; defaults to cwd"),
    json_output: bool = typer.Option(
        False, "--json", help="Return the complete machine-readable result"
    ),
) -> None:
    """Create one local configuration and connection fragment; never start optimization."""
    from .setup import setup as prepare

    result = prepare(
        workbench,
        repo,
        owner,
        git_bin=ctx.obj["git_bin"],
        platform_config=ctx.meta.get("evolution_setup_config"),
    )
    if json_output:
        _emit(result)
        return
    typer.echo("配置已生成，尚未启动工作流。")
    typer.echo(f"试点仓库：{result['repo_path']}\n负责人：{result['owner']}")
    typer.echo(f"配置文件：{result['config_path']}")
    typer.echo(f"连接片段：{result['fragment_path']}")
    for number, step in enumerate(result["next_steps"], start=1):
        typer.echo(f"{number}. {step}")
    typer.echo(f'本地配置检查：python -m hmopt evolve --config "{result["config_path"]}" doctor')


@command()
def doctor(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Return all checks as JSON"),
) -> None:
    """Check local readiness without models, devices, source changes or a new database."""
    from .doctor import doctor as diagnose

    path = ctx.meta.get("evolution_config") or (
        Path.cwd() / ".opencode/local/evolution/config.json"
    )
    report = diagnose(path)
    if json_output:
        _emit(report)
    else:
        typer.echo(
            "发现准备：" + ("检查通过" if report.get("ready_for_discovery") else "需要处理缺项")
        )
        typer.echo(f"配置文件：{report['config_path']}")
        statuses = {
            "blocked": "阻塞",
            "warning": "需留意",
            "not_configured": "未配置",
            "not_verified": "未验证",
        }
        for check in report["checks"]:
            if check["status"] != "passed":
                typer.echo(
                    f"[{statuses.get(check['status'], check['status'])}] {check['name']}: {check['message']}"
                )
        if report.get("environment_overrides"):
            typer.echo("这些旧环境变量覆盖配置文件：" + ", ".join(report["environment_overrides"]))
        typer.echo("本次只检查本地准备条件；未验证 OpenCode 连接、模型执行或真机收益。")
        if report.get("ready_for_discovery"):
            typer.echo("下一步：在 OpenCode 确认已连接主 MCP，再选择一个已配置试点：")
            for profile in report.get("discovery_profiles", []):
                typer.echo(f"  /evolve-discover {profile}")
    if not report.get("ready_for_discovery"):
        raise typer.Exit(code=2)


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
    """Read first-parent Git history into the code-analysis queue; no keyword distillation."""
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
    hotspots: Path | None = None,
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
    payload: Path | None = None,
    note: str | None = None,
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
def handoff(ctx: typer.Context, candidate_id: str, output: Path | None = None) -> None:
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


@command("resolve-native-evidence")
def resolve_native_evidence(ctx: typer.Context, reference: str) -> None:
    """Resolve the encoded source reference in a native journal/Hub staging record."""
    from .native_memory import resolve_native_evidence as resolve

    _emit(resolve(_service(ctx), reference))


@command()
def capture(
    ctx: typer.Context,
    signal: str,
    source_id: str,
    recipe: str,
    actor: str = typer.Option(...),
    source_kind: str = "candidate",
    corrects: str | None = None,
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
def mcp_stdio(
    ctx: typer.Context,
    workspace_root: Path | None = typer.Option(None, envvar="HMOPT_EVOLUTION_WORKSPACE_ROOT"),
    artifacts_root: Path | None = typer.Option(None, envvar="HMOPT_EVOLUTION_ARTIFACTS_ROOT"),
    discovery_profiles: Path | None = typer.Option(
        None,
        envvar="HMOPT_EVOLUTION_DISCOVERY_PROFILES",
        help="Operator-owned JSON object mapping profile names to discovery configurations",
    ),
) -> None:
    """Serve read/agent evidence tools locally; owner/curator decisions stay in CLI."""
    from .mcp import build_server

    options = ctx.meta.get("evolution_options", {})
    build_server(
        **ctx.obj,
        workspace_root=workspace_root or options.get("workspace_root"),
        artifacts_root=artifacts_root or options.get("artifacts_root"),
        discovery_profiles=(
            _load(discovery_profiles) if discovery_profiles else options.get("discovery_profiles")
        ),
        production=options.get("production"),
        source_workspace=options.get("source_workspace"),
    ).run(transport="stdio")


def _model(kind: str):
    from .approval import Decision
    from .change_analysis import HistoryAnalysis
    from .correctness import CorrectnessPolicy, CorrectnessReport
    from .discovery import DiscoveryConfig
    from .lmbench import LmbenchManifest, LmbenchProfile
    from .mining import ChangeRecord, Hotspot, Pattern
    from .production import ProductionConfig
    from .quality_eval import EvaluationSet
    from .reports import ICManifest
    from .service import Plan, Review
    from .sources import EvidenceRecord
    from .validation import ABReport
    from .workflow import ReviewSheet

    models = {
        "production-config": ProductionConfig,
        "quality-evaluation": EvaluationSet,
        "expert-decision": Decision,
        "history": ChangeRecord,
        "history-analysis": HistoryAnalysis,
        "hotspot": Hotspot,
        "pattern": Pattern,
        "plan": Plan,
        "review": Review,
        "ab-report": ABReport,
        "correctness-policy": CorrectnessPolicy,
        "correctness-report": CorrectnessReport,
        "ic-manifest": ICManifest,
        "lmbench-manifest": LmbenchManifest,
        "lmbench-profile": LmbenchProfile,
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
    cursor: str | None = None,
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
    source_ids: Path | None = None,
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
    owner: str | None = None,
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
    workspace_root: Path | None = None,
) -> None:
    """Stage task-specific OpenCode artifacts after gates; no automatic execution."""
    from .workflow import dispatch_candidate

    _emit(
        dispatch_candidate(
            _service(ctx),
            candidate_id,
            output,
            actor=actor,
            request_id=request_id,
            workspace_root=workspace_root,
        )
    )


@command("materialize-workspace")
def materialize_workspace(
    ctx: typer.Context,
    dispatch_id: str,
    workspace_root: Path = typer.Option(...),
) -> None:
    """Bind a staged dispatch to an isolated native OpenCode task workspace."""
    from .workflow import materialize_dispatch_workspace

    _emit(materialize_dispatch_workspace(_service(ctx), dispatch_id, workspace_root))


@command("convert-lmbench")
def convert_lmbench(
    ctx: typer.Context,
    candidate_id: str,
    manifest: Path,
    artifacts_root: Path | None = None,
    output: Path | None = None,
) -> None:
    """Convert explicit raw suite pairs; the frozen validation gate runs separately."""
    from .lmbench import LmbenchManifest, convert_lmbench_report

    result = convert_lmbench_report(
        _service(ctx),
        candidate_id,
        LmbenchManifest.model_validate(_load(manifest)),
        artifacts_root=artifacts_root,
    )
    if output:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result["report"], stream, ensure_ascii=False, indent=2, allow_nan=False)
    _emit(result)


@command("import-native-memory")
def import_native_memory(
    ctx: typer.Context,
    source_root: Path,
    collection: str = typer.Option(
        ..., help="journal or hub; journal requires contributor/project"
    ),
    repo_id: str = typer.Option(...),
    actor: str = typer.Option(...),
    contributor: str | None = None,
    project: str | None = None,
    max_files: int = 200,
    max_bytes: int = 8388608,
    cursor: str | None = None,
) -> None:
    """Import a scoped page from native Team Memory/Hub as untrusted source evidence."""
    from .native_memory import import_native_memory as ingest

    _emit(
        ingest(
            _service(ctx),
            source_root,
            repo_id,
            collection=collection,
            actor=actor,
            contributor=contributor,
            project=project,
            max_files=max_files,
            max_bytes=max_bytes,
            cursor=cursor,
        )
    )


@command("export-native-memory")
def export_native_memory(
    ctx: typer.Context,
    skill_id: str,
    content: Path = typer.Option(..., help="Reviewed title/body/applicability JSON"),
    memory_root: Path = typer.Option(...),
    hub_root: Path = typer.Option(...),
    contributor: str = typer.Option(...),
    project: str = typer.Option(...),
    actor: str = typer.Option(...),
    request_id: str = typer.Option(...),
) -> None:
    """Write a curated evidence projection to native journal and Hub staging."""
    from .native_memory import export_native_memory as export

    value = _load(content)
    required = {"title", "body", "applies_when", "invalidated_by"}
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or (value.keys() - required - {"target_slug"})
    ):
        raise typer.BadParameter(
            "Content needs title, body, applies_when, invalidated_by; optional target_slug"
        )
    if any(not isinstance(value[key], str) for key in ("title", "body")) or (
        "target_slug" in value and not isinstance(value["target_slug"], str)
    ):
        raise typer.BadParameter("Content title, body and target_slug must be strings")
    _emit(
        export(
            _service(ctx),
            skill_id,
            memory_root=memory_root,
            hub_root=hub_root,
            contributor=contributor,
            project=project,
            actor=actor,
            request_id=request_id,
            **value,
        )
    )


@command("convert-ic")
def convert_ic(
    ctx: typer.Context,
    candidate_id: str,
    compare: Path,
    manifest: Path,
    output: Path | None = None,
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
def quality(ctx: typer.Context, pattern_key: str | None = None) -> None:
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
    version: int | None = None,
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
    skill_ids: Path | None = None,
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
    batch_id: str | None = None,
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


@command("queue")
def queue(
    ctx: typer.Context,
    repo: Path,
    state: str = "approved",
    owner: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> None:
    """List archived approvals and current execution readiness for a repository."""
    from .catalog import candidates

    _emit(candidates(_service(ctx), repo, state=state, owner=owner, limit=limit, offset=offset))


@command("dossier")
def candidate_dossier(ctx: typer.Context, candidate_id: str) -> None:
    """Read the complete evidence and approval index for one candidate ID."""
    from .catalog import dossier

    _emit(dossier(_service(ctx), candidate_id))


def _production(ctx):
    from .production import production_config

    return production_config(ctx.meta.get("evolution_options", {}).get("production"))


@command("production-status")
def production_status(ctx: typer.Context):
    """Check configured worker/directory/secret references locally, without external calls."""
    from .production import check_production

    _emit(check_production(ctx.meta.get("evolution_options", {}).get("production")))


@command("mine-campaign")
def mine_campaign(
    ctx: typer.Context,
    profile: str,
    source_ids: list[str],
    actor: str = "coordinator",
    request_id: str = typer.Option(...),
):
    """Queue explicit source IDs for the operator-configured OpenCode research worker."""
    from .mcp_discovery import prepare_profiles, select_profile
    from .worker import create_campaign

    config = _production(ctx).worker
    if config is None:
        raise ValueError("Configure production.worker")
    service = _service(ctx)
    profiles = prepare_profiles(ctx.meta["evolution_options"].get("discovery_profiles"))
    selected = select_profile(service, profiles, profile)
    _emit(
        create_campaign(
            service, selected.repo_path, source_ids, config, actor=actor, request_id=request_id
        )
    )


@command("mining-status")
def mining_status(ctx: typer.Context, campaign_id: str):
    from .worker import campaign_status

    _emit(campaign_status(_service(ctx), campaign_id))


@command("mining-cancel")
def mining_cancel(ctx: typer.Context, campaign_id: str, actor: str = "operator"):
    from .worker import cancel_campaign

    _emit(cancel_campaign(_service(ctx), campaign_id, actor=actor))


@command("worker")
def production_worker(ctx: typer.Context, once: bool = False, worker_id: str = "research-worker"):
    """Run bounded concurrent research polling; --once performs one cycle. Ctrl+C stops polling."""
    import time

    from .worker import worker_cycle

    config = _production(ctx).worker
    if config is None:
        raise ValueError("Configure production.worker")
    service = _service(ctx)
    while True:
        _emit(worker_cycle(service, config, worker_id=worker_id))
        if once:
            break
        time.sleep(3)


@command("mining-retire")
def mining_retire(
    ctx: typer.Context, job_id: str, actor: str = "operator", worker_stopped: bool = False
):
    """Abort an uncertain/terminal remote session and release its reserved worker capacity."""
    from .worker import retire_job

    config = _production(ctx).worker
    if config is None:
        raise ValueError("Configure production.worker")
    _emit(retire_job(_service(ctx), job_id, config, actor=actor, worker_stopped=worker_stopped))


@command("mining-retry")
def mining_retry(
    ctx: typer.Context, job_id: str, version: int = typer.Option(...), actor: str = "operator"
):
    """Explicitly retry a retired attention job, preserving failed output and attempt budget."""
    from .worker import retry_job

    _emit(retry_job(_service(ctx), job_id, actor=actor, expected_version=version))


@command("request-approval")
def expert_request(
    ctx: typer.Context,
    candidate_id: str,
    actor: str = "operator",
    request_id: str = typer.Option(...),
):
    """Archive expert context and queue one notification to the configured owner directory."""
    from .approval import request_approval

    config = _production(ctx).approval
    if config is None:
        raise ValueError("Configure production.approval")
    _emit(request_approval(_service(ctx), candidate_id, config, actor=actor, request_id=request_id))


@command("notify")
def notify_experts(
    ctx: typer.Context, limit: int = 20, actor: str = "operator", watch: bool = False
):
    """Deliver pending notifications via configured SMTP/webhook. This sends external messages."""
    import time

    from .approval import deliver_notifications

    config = _production(ctx).approval
    if config is None:
        raise ValueError("Configure production.approval")
    service = _service(ctx)
    while True:
        result = deliver_notifications(service, config, actor=actor, limit=limit)
        if result["notifications"] or not watch:
            _emit(result)
        if not watch:
            break
        time.sleep(3)


@command("retry-notification")
def notification_retry(
    ctx: typer.Context,
    notification_id: str,
    version: int = typer.Option(...),
    actor: str = "operator",
    delivery_stopped: bool = False,
):
    """Explicitly requeue an uncertain send; duplicate delivery is possible."""
    from .approval import retry_notification

    _emit(
        retry_notification(
            _service(ctx),
            notification_id,
            actor=actor,
            expected_version=version,
            delivery_stopped=delivery_stopped,
        )
    )


@command("evaluate")
def evaluate_quality(ctx: typer.Context, dataset: Path, actor: str = "evaluator"):
    """Evaluate frozen expert labels against archived model reports; no live model calls."""
    from .quality_eval import EvaluationSet, evaluate

    _emit(evaluate(_service(ctx), EvaluationSet.model_validate(_load(dataset)), actor=actor))


@command("suggest-groups")
def grouping(ctx: typer.Context, repo: Path, limit: int = 200, offset: int = 0):
    from .quality_eval import suggest_groups

    _emit(suggest_groups(_service(ctx), repo, limit=limit, offset=offset))


@command("workspace-start")
def workspace_start(
    ctx: typer.Context,
    projects: list[str] = typer.Option(None),
    request_id: str = typer.Option(...),
    max_commits: int = 10000,
    actor: str = "operator",
    full_history: bool = False,
):
    """Freeze selected Git projects and queue research; start serve to advance them."""
    from .runs import start_run

    workspace = ctx.meta.get("evolution_options", {}).get("source_workspace")
    if workspace is None:
        raise ValueError("Configure source_workspace or run setup-workspace")
    _emit(
        start_run(
            _service(ctx),
            workspace,
            actor=actor,
            request_id=request_id,
            projects=projects or None,
            max_commits=max_commits,
            full_history=full_history,
        )
    )


@command("setup-workspace")
def setup_workspace_command(
    ctx: typer.Context,
    source_root: Path | None = typer.Option(None, help="Defaults to existing PROJECT_REPO_PATH"),
    workspace_id: str = typer.Option("business"),
    project: list[str] = typer.Option(
        ..., help="id=relative/path; kernel reuses existing kernel checkout path"
    ),
    owner: str = typer.Option(...),
    workbench: Path = Path("."),
):
    """Configure only explicit --project id=relative/path Git checkouts under a repo workspace."""
    from .setup import setup_workspace

    projects = {}
    for item in project:
        name, sep, path = item.partition("=")
        if (not sep and name != "kernel") or name in projects:
            raise ValueError("Use distinct --project id=relative/path entries")
        projects[name] = path if sep else None
    _emit(
        setup_workspace(
            workbench,
            source_root,
            workspace_id,
            projects,
            owner,
            git_bin=ctx.obj["git_bin"],
            platform_config=ctx.meta.get("evolution_setup_config"),
        )
    )


@command("workspace-status")
def workspace_status(ctx: typer.Context, run_id: str):
    from .runs import run_status

    _emit(run_status(_service(ctx), run_id))


@command("workspace-control")
def workspace_control(
    ctx: typer.Context,
    run_id: str,
    action: str,
    version: int = typer.Option(...),
    project_id: str | None = None,
    max_commits: int | None = None,
    actor: str = "operator",
):
    from .runs import control_run

    _emit(
        control_run(
            _service(ctx),
            run_id,
            action=action,
            expected_version=version,
            actor=actor,
            project_id=project_id,
            max_commits=max_commits,
        )
    )


@command("serve")
def serve(
    ctx: typer.Context,
    once: bool = False,
    notifications: bool = False,
    worker_id: str = "evolution-runtime",
    interval: int = 3,
):
    """Advance authorized multi-Git runs and research. --notifications explicitly enables delivery."""
    import time

    from .runtime import require_workspace_scope, runtime_cycle

    if not 1 <= interval <= 60:
        raise ValueError("interval must be 1..60 seconds")
    service, config = _service(ctx), _production(ctx)
    require_workspace_scope(service, ctx.meta.get("evolution_options", {}).get("source_workspace"))
    try:
        while True:
            _emit(runtime_cycle(service, config, worker_id=worker_id, notifications=notifications))
            if once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo(
            "Polling stopped. Remote jobs remain archived; inspect status before retiring them."
        )


@command("experiment-prepare")
def experiment_prepare(
    ctx: typer.Context,
    candidate_id: str,
    request_id: str = typer.Option(...),
    actor: str = "validator",
):
    from .experiments import prepare_experiment

    config = _production(ctx).validation
    if config is None:
        raise ValueError("Configure production.validation")
    _emit(
        prepare_experiment(_service(ctx), candidate_id, config, actor=actor, request_id=request_id)
    )


@command("experiment-run")
def experiment_run(ctx: typer.Context, experiment_id: str):
    """Execute the configured build/test/device adapter for this exact prepared experiment."""
    from .experiments import run_experiment

    config = _production(ctx).validation
    if config is None:
        raise ValueError("Configure production.validation")
    _emit(run_experiment(_service(ctx), experiment_id, config))


@command("experiment-retire")
def experiment_retire(
    ctx: typer.Context,
    experiment_id: str,
    version: int = typer.Option(...),
    execution_stopped: bool = False,
    actor: str = "operator",
):
    from .experiments import retire_experiment

    _emit(
        retire_experiment(
            _service(ctx),
            experiment_id,
            actor=actor,
            expected_version=version,
            execution_stopped=execution_stopped,
        )
    )


@command("scan-start")
def scan_start(
    ctx: typer.Context, profile: str, request_id: str = typer.Option(...), actor: str = "operator"
):
    from .mcp_discovery import prepare_profiles, select_profile
    from .scan import start_scan

    service = _service(ctx)
    config = select_profile(
        service,
        prepare_profiles(ctx.meta.get("evolution_options", {}).get("discovery_profiles")),
        profile,
    )
    _emit(
        start_scan(
            service,
            config.repo_path,
            revision=config.revision,
            owners=config.owners,
            hotspots=config.hotspots,
            request_id=request_id,
            actor=actor,
        )
    )


@command("scan-next")
def scan_page(ctx: typer.Context, scan_id: str, version: int = typer.Option(...)):
    from .scan import scan_next

    _emit(scan_next(_service(ctx), scan_id, expected_version=version))


@command("scan-control")
def scan_control(
    ctx: typer.Context,
    scan_id: str,
    action: str,
    version: int = typer.Option(...),
    actor: str = "operator",
):
    from .scan import control_scan

    _emit(
        control_scan(_service(ctx), scan_id, action=action, actor=actor, expected_version=version)
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
