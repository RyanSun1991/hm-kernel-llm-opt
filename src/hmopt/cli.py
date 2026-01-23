"""CLI entrypoint using Typer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import typer

from hmopt.core.config import AppConfig
from hmopt.orchestration import run_artifact_analysis, run_pipeline
from hmopt.indexing import (
    build_kernel_index,
    build_kernel_indexes,
    build_runtime_aggregate_index,
    build_runtime_index,
    route_query,
)
from hmopt.storage.artifact_store import ArtifactStore
from hmopt.storage.db.engine import init_engine
from hmopt.storage.db import models
from hmopt.storage.db.engine import session_scope
from hmopt.core.config import load_yaml, normalize_raw_config
from hmopt.models.hiperf_report import HiperfReport, Frame

app = typer.Typer(help="HM-VERIF kernel optimization platform")


def _load_config(path: str) -> AppConfig:
    return AppConfig.from_yaml(path)


@app.command()
def run(config: str = typer.Option("configs/app.yaml", help="Path to config YAML")) -> None:
    # Demo: python -m hmopt.cli run --config configs/app.yaml
    # Purpose: run full optimization pipeline (baseline profile + iterative loop).
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    run_id = run_pipeline(cfg)
    typer.echo(f"Run completed: {run_id}")


@app.command()
def optimize(
    config: str = typer.Option("configs/app.yaml", help="Path to config YAML"),
    iterations: int = typer.Option(2, help="Max iterations"),
) -> None:
    # Demo: python -m hmopt.cli optimize --config configs/app.yaml --iterations 3
    # Purpose: same as run, but override iteration budget.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    cfg.iterations = iterations
    run_id = run_pipeline(cfg)
    typer.echo(f"Optimization finished. run_id={run_id}")


@app.command()
def ingest_artifact(
    path: str = typer.Argument(..., help="File to store"),
    kind: str = typer.Option("generic", help="Artifact kind"),
    run_id: Optional[str] = typer.Option(None, help="Run ID to attach"),
    config: str = typer.Option("configs/app.yaml", help="Config for storage settings"),
) -> None:
    # Demo: python -m hmopt.cli ingest-artifact outputs/flamegraph.json --kind flamegraph
    # Purpose: manually stash an artifact into the DB/artifact store.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    engine = init_engine(cfg.storage.db_url, schema_path=Path("src/hmopt/storage/db/schema.sql"))
    store = ArtifactStore(cfg.storage.artifacts_root)
    with session_scope(engine) as session:
        art = store.store_file(Path(path), kind=kind, run_id=run_id, session=session)
        typer.echo(f"Stored artifact {art.artifact_id} at {art.path}")


@app.command()
def analyze(config: str = typer.Option("configs/app.yaml", help="Config YAML")) -> None:
    # Demo: python -m hmopt.cli analyze --config configs/app.yaml
    # Purpose: run a single-iteration baseline analysis (no extra iterations).
    cfg = _load_config(config)
    cfg.iterations = 1
    run_id = run_pipeline(cfg)
    typer.echo(f"Analysis baseline run_id={run_id}")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID to summarize"),
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
) -> None:
    # Demo: python -m hmopt.cli report <run_id> --config configs/app.yaml
    # Purpose: fetch status/metrics/hotspots for a finished run.
    cfg = _load_config(config)
    engine = init_engine(cfg.storage.db_url, schema_path=Path("src/hmopt/storage/db/schema.sql"))
    with session_scope(engine) as session:
        run = session.query(models.Run).filter(models.Run.run_id == run_id).one_or_none()
        if not run:
            typer.echo("run not found")
            raise typer.Exit(code=1)
        metrics = session.query(models.Metric).filter(models.Metric.run_id == run_id).all()
        hotspots = session.query(models.Hotspot).filter(models.Hotspot.run_id == run_id).all()
        # graph = session.query(models.Graph).filter(models.Graph.run_id == run_id).all()
        # patch = session.query(models.Patch).filter(models.Patch.run_id == run_id).all()
        # agentMessage = session.query(models.AgentMessage).filter(models.AgentMessage.run_id == run_id).all()
        # vectorEmbedding = session.query(models.VectorEmbedding).filter(models.VectorEmbedding.run_id == run_id).all()
        summary = {
            "run_id": run_id,
            "status": run.status,
            "metrics": {m.metric_name: m.value for m in metrics},
            "hotspots": [h.symbol for h in hotspots],
            # "graph": [h.metadata_json for h in graph],
            # "patch": [h.files_changed_json for h in patch],
            # "agentMessage": [h.output_artifact_id for h in agentMessage],
            # "vectorEmbedding": [h.embedding_json for h in vectorEmbedding],
        }
        typer.echo(json.dumps(summary, indent=2))

@app.command()
def analyze_artifacts(
    artifact: list[str] = typer.Option(
        [],
        "--artifact",
        help="Artifact spec kind:path (e.g., flamegraph:outputs/flamegraph.json). Repeatable.",
    ),
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
    repo_path: Optional[str] = typer.Option(None, help="Override repo path for this run"),
    with_patch: bool = typer.Option(True, help="Run Conductor+Coder to suggest patches"),
    with_verify: bool = typer.Option(False, help="Run build/test verification after patch"),
    with_profile: bool = typer.Option(False, help="Re-profile candidate after patch"),
) -> None:
    # Demo: python -m hmopt.cli analyze-artifacts \
    #          --artifact flamegraph:outputs/flamegraph.json \
    #          --artifact hitrace:outputs/hitrace.json \
    #          --artifact hiperf:outputs/hiperf.json \
    #          --repo-path /path/to/hm-verif-kernel
    # Purpose: ingest existing traces (no live profiling), run analysis -> hotspots/report.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    if repo_path:
        cfg.project.repo_path = repo_path
    artifacts = []
    for spec in artifact:
        if ":" not in spec:
            typer.echo(f"Invalid artifact spec: {spec}")
            raise typer.Exit(code=1)
        kind, path = spec.split(":", 1)
        artifacts.append({"kind": kind, "path": path})
    run_id = run_artifact_analysis(
        cfg,
        artifacts,
        run_conductor=with_patch,
        run_coder=with_patch,
        run_verify=with_verify,
        run_profile=with_profile,
    )
    typer.echo(f"Artifact analysis complete. run_id={run_id}")






@app.command()
def index_kernel(
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
    repo_path: Optional[str] = typer.Option(None, help="Override repo path for this run"),
    compile_commands_dir: Optional[str] = typer.Option(
        None, help="Directory containing compile_commands.json"
    ),
    repo_name: Optional[str] = typer.Option(None, help="Repo name (for multi-repo configs)"),
    all_repos: bool = typer.Option(False, help="Index all repos in config.project.repos"),
    index_version: Optional[str] = typer.Option(None, help="Override code index version tag"),
    incremental: bool = typer.Option(False, help="Use git diff incremental indexing"),
    base_ref: Optional[str] = typer.Option(None, help="Base git ref for incremental diff"),
    incremental_mode: Optional[str] = typer.Option(
        None, help="Incremental mode: rebuild|merge"
    ),
) -> None:
    # Demo: python -m hmopt.cli index-kernel --repo-path /path/to/hm-verif-kernel \
    #          --compile-commands-dir /path/to/hm-verif-kernel
    # Demo: python -m hmopt.cli index-kernel --incremental --base-ref HEAD~1
    # Purpose: build kernel code ingestion + LlamaIndex index (clangd preferred).
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    repo_path_override = repo_path or None
    if repo_path_override:
        cfg.project.repo_path = repo_path_override
    if compile_commands_dir:
        cfg.indexing.clangd.compile_commands_dir = Path(compile_commands_dir)
    if all_repos:
        build_kernel_indexes(
            cfg,
            repo_names=None,
            index_version=index_version,
            incremental=incremental,
            base_ref=base_ref,
            incremental_mode=incremental_mode,
        )
        typer.echo("Kernel code indexes built for all repos")
        return
    build_kernel_index(
        cfg,
        repo_path=repo_path_override,
        repo_name=repo_name,
        compile_commands_dir=compile_commands_dir,
        index_version=index_version,
        incremental=incremental,
        base_ref=base_ref,
        incremental_mode=incremental_mode,
    )
    typer.echo("Kernel code index built")


@app.command()
def index_runtime(
    run_id: str = typer.Argument(..., help="Run ID to index runtime data"),
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
    index_version: Optional[str] = typer.Option(None, help="Override runtime index version tag"),
    repo_name: Optional[str] = typer.Option(None, help="Repo name for runtime index path"),
) -> None:
    # Demo: python -m hmopt.cli index-runtime <run_id>
    # Purpose: build runtime metrics/hotspots index for a run.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    build_runtime_index(cfg, run_id, index_version=index_version, repo_name=repo_name)
    typer.echo(f"Runtime index built for run_id={run_id}")


@app.command()
def index_runtime_aggregate(
    run_id: List[str] = typer.Option(..., "--run-id", "-r", help="Run IDs to include"),
    group_name: str = typer.Option("aggregate", help="Aggregate group name"),
    index_version: Optional[str] = typer.Option(None, help="Override aggregate index version tag"),
    repo_name: Optional[str] = typer.Option(None, help="Repo name for aggregate index path"),
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
) -> None:
    # Demo: python -m hmopt.cli index-runtime-aggregate -r run1 -r run2 --group-name perf-batch
    # Purpose: build a cross-run runtime aggregate index.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    build_runtime_aggregate_index(
        cfg,
        run_id,
        group_name=group_name,
        index_version=index_version,
        repo_name=repo_name,
    )
    typer.echo(f"Runtime aggregate index built: group={group_name} runs={len(run_id)}")


@app.command()
def query(
    query_str: str = typer.Argument(..., help="Query to run against indexes"),
    mode: str = typer.Option("auto", help="auto|code|runtime|graph|runtime_code"),
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
    code_version: Optional[str] = typer.Option(None, help="Code index version tag"),
    runtime_version: Optional[str] = typer.Option(None, help="Runtime index version tag"),
    run_id: Optional[str] = typer.Option(None, help="Runtime run_id for versioned lookup"),
    repo_name: Optional[str] = typer.Option(None, help="Repo name for multi-repo indexes"),
) -> None:
    # Demo: python -m hmopt.cli query "Which function is hot?" --mode runtime
    # Demo: python -m hmopt.cli query "Optimize hotspots" --mode runtime_code --run-id <run_id>
    # Purpose: query routing across code/runtime indexes.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    response = route_query(
        cfg,
        query_str,
        mode=mode,
        code_version=code_version,
        runtime_version=runtime_version,
        run_id=run_id,
        repo_name=repo_name,
    )
    typer.echo(response)



if __name__ == "__main__":
    app()
