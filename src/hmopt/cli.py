"""CLI entrypoint using Typer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import subprocess
from typing import Optional

import typer

from hmopt.core.config import AppConfig
from hmopt.opencode import (
    initialize_pipeline_session,
    load_pipeline_profiles,
    resume_pipeline_session,
)
from hmopt.storage.artifact_store import ArtifactStore
from hmopt.storage.db.engine import init_engine
from hmopt.storage.db import models
from hmopt.storage.db.engine import session_scope

app = typer.Typer(help="HM-VERIF kernel optimization platform")


def _load_config(path: str) -> AppConfig:
    return AppConfig.from_yaml(path)


@app.command()
def run(config: str = typer.Option("configs/app.yaml", help="Path to config YAML")) -> None:
    # Demo: python -m hmopt.cli run --config configs/app.yaml
    # Purpose: run full optimization pipeline (baseline profile + iterative loop).
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    from hmopt.orchestration import run_pipeline

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
    from hmopt.orchestration import run_pipeline

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
    from hmopt.orchestration import run_pipeline

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
    repo_path: Optional[str] = typer.Option(
        None, help="Override repo path (legacy pipeline only)"
    ),
    with_patch: bool = typer.Option(False, help="Run Conductor+Coder to suggest patches (legacy)"),
    with_verify: bool = typer.Option(False, help="Run build/test verification after patch"),
    with_profile: bool = typer.Option(False, help="Re-profile candidate after patch"),
    legacy_pipeline: bool = typer.Option(
        False,
        help="Use legacy analyze_artifacts pipeline (PSG + LLM evidence/patch).",
    ),
) -> None:
    # Demo: python -m hmopt.cli analyze-artifacts \
    #          --artifact flamegraph:outputs/flamegraph.json \
    #          --artifact hitrace:outputs/hitrace.json \
    #          --artifact hiperf:outputs/hiperf.json \
    #          --repo-path /path/to/hm-verif-kernel
    # Purpose: ingest existing traces (no live profiling), run analysis -> hotspots/report.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    if repo_path and (legacy_pipeline or with_patch):
        cfg.project.repo_path = repo_path
    elif repo_path:
        logging.getLogger(__name__).info(
            "Ignoring repo_path override for lightweight runtime ingest."
        )
    artifacts = []
    for spec in artifact:
        if ":" not in spec:
            typer.echo(f"Invalid artifact spec: {spec}")
            raise typer.Exit(code=1)
        kind, path = spec.split(":", 1)
        artifacts.append({"kind": kind, "path": path})
    if legacy_pipeline or with_patch:
        from hmopt.orchestration import run_artifact_analysis

        run_id = run_artifact_analysis(
            cfg,
            artifacts,
            run_conductor=with_patch,
            run_coder=with_patch,
            run_verify=with_verify,
            run_profile=with_profile,
        )
    else:
        from hmopt.orchestration import run_runtime_ingest

        run_id = run_runtime_ingest(cfg, artifacts)
    typer.echo(f"Artifact analysis complete. run_id={run_id}")






@app.command()
def index_kernel(
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
    repo_path: Optional[str] = typer.Option(None, help="Override repo path for this run"),
    compile_commands_dir: Optional[str] = typer.Option(
        None, help="Directory containing compile_commands.json"
    ),
    backend: Optional[str] = typer.Option(
        None,
        help=(
            "Code-index backend: clangd | scip-clang. Overrides config.indexing.backend. "
            "scip-clang requires the scip-clang binary on PATH and protobuf>=4.25."
        ),
    ),
) -> None:
    # Demo: python -m hmopt.cli index-kernel --repo-path /path/to/hm-verif-kernel \
    #          --compile-commands-dir /path/to/hm-verif-kernel --backend scip-clang
    # Purpose: build kernel code ingestion + LlamaIndex index. Either backend
    # produces the same CodeIndex shape; scip-clang additionally populates
    # call_site_* fields and other scip-only enrichments.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    if repo_path:
        cfg.project.repo_path = repo_path
    if compile_commands_dir:
        cfg.indexing.clangd.compile_commands_dir = Path(compile_commands_dir)
    if backend:
        normalized = backend.strip().lower()
        if normalized not in ("clangd", "scip-clang"):
            typer.echo(
                f"Unknown backend {backend!r}; expected clangd | scip-clang", err=True
            )
            raise typer.Exit(code=2)
        cfg.indexing.backend = normalized
    from hmopt.indexing import build_kernel_index

    build_kernel_index(cfg, repo_path=cfg.project.repo_path)
    typer.echo(f"Kernel code index built (backend={cfg.indexing.backend})")


@app.command()
def index_runtime(
    run_id: str = typer.Argument(..., help="Run ID to index runtime data"),
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
) -> None:
    # Demo: python -m hmopt.cli index-runtime <run_id>
    # Purpose: build runtime metrics/hotspots index for a run.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    from hmopt.indexing import build_runtime_index

    build_runtime_index(cfg, run_id)
    typer.echo(f"Runtime index built for run_id={run_id}")


@app.command()
def query(
    query_str: str = typer.Argument(
        ..., help="Query text or @/path to a prompt file"
    ),
    mode: str = typer.Option("auto", help="auto|code|runtime|runtime_code|graph"),
    symbols: Optional[str] = typer.Option(
        None, "--symbols", help="Comma-separated symbol list to analyze"
    ),
    hotspot_top_k: Optional[int] = typer.Option(
        None, "--hotspot-top-k", help="Top-K hotspots to analyze"
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Run ID for selecting runtime index"
    ),
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
    prompt_file: Optional[str] = typer.Option(
        None,
        "--prompt-file",
        "-p",
        help="Path to prompt template file (txt/md/json)",
    ),
) -> None:
    # Demo: python -m hmopt.cli query "Which function is hot?" --mode runtime
    # Demo: python -m hmopt.cli query @queries/runtime_prompt.md --mode runtime_code
    # Demo: python -m hmopt.cli query "analyze hotspots" --prompt-file configs/prompts/analysis.md
    # Purpose: query routing across code/runtime indexes with optional custom prompt.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    prompt_path = Path(prompt_file) if prompt_file else None
    symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None
    from hmopt.indexing import route_query

    response = route_query(
        cfg,
        query_str,
        mode=mode,
        prompt_file=prompt_path,
        run_id=run_id,
        focus_symbols=symbol_list,
        hotspot_top_k=hotspot_top_k,
    )
    typer.echo(response)


@app.command("call-chain")
def call_chain_cmd(
    symbols: str = typer.Option(
        ..., "--symbols", help="Comma-separated root symbols"
    ),
    direction: str = typer.Option(
        "both", "--direction", help="callers|callees|both"
    ),
    depth: int = typer.Option(3, "--depth", help="BFS depth (1..6)"),
    per_hop_limit: int = typer.Option(
        100, "--per-hop-limit", help="Per-hop edge limit (Cypher LIMIT)"
    ),
    frontier_cap: int = typer.Option(
        50, "--frontier-cap", help="Max frontier symbols expanded per layer"
    ),
    edge_kinds: Optional[str] = typer.Option(
        None,
        "--edge-kinds",
        help="Comma-separated edge kinds: calls,uses_type,uses_macro",
    ),
    output_format: str = typer.Option(
        "text", "--output-format", help="text|json"
    ),
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
) -> None:
    # Demo: hmopt call-chain --symbols do_syscall --depth 6 --direction callees
    # Purpose: pure structural call-chain retrieval (no code bodies, depth up to 6).
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    kinds_list = (
        [k.strip() for k in edge_kinds.split(",") if k.strip()] if edge_kinds else None
    )
    from hmopt.indexing import retrieve_call_chain

    out = retrieve_call_chain(
        cfg,
        sym_list,
        direction=direction,
        depth=depth,
        per_hop_limit=per_hop_limit,
        frontier_cap=frontier_cap,
        edge_kinds=kinds_list,
        output_format="json" if output_format == "json" else "text",
    )
    if output_format == "json":
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        typer.echo(out)


@app.command("get-snippets")
def get_snippets_cmd(
    symbols: str = typer.Option(
        ..., "--symbols", help="Comma-separated symbol list to fetch bodies for"
    ),
    per_symbol_max_chars: int = typer.Option(
        4000, "--per-symbol-max-chars", help="Per-snippet char cap"
    ),
    total_max_chars: Optional[int] = typer.Option(
        None, "--total-max-chars", help="Optional total char budget across all snippets"
    ),
    output_format: str = typer.Option(
        "text", "--output-format", help="text|json"
    ),
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
) -> None:
    # Demo: hmopt get-snippets --symbols do_syscall,handle_mm_fault
    # Purpose: batch fetch function bodies for an explicit symbol list.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    from hmopt.indexing import fetch_code_snippets

    out = fetch_code_snippets(
        cfg,
        sym_list,
        per_symbol_max_chars=per_symbol_max_chars,
        total_max_chars=total_max_chars,
        output_format="json" if output_format == "json" else "text",
    )
    if output_format == "json":
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        typer.echo(out)


@app.command("sediment")
def sediment(
    run_id: str | None = typer.Option(
        None, "--run-id", help="Run id under .opencode/local/runs/"
    ),
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Explicit run directory to scan"
    ),
    staging_dir: Path = typer.Option(
        Path(".opencode/local/sediment_staging"), help="Tier-1 JSONL staging dir"
    ),
    bundle: bool = typer.Option(
        False, "--bundle", help="Bundle staged JSONL files instead of extracting one run"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output JSONL path"
    ),
    open_pr: bool = typer.Option(False, "--open-pr", help="Prepare a PR bundle summary"),
) -> None:
    from hmopt.cli_sediment import sediment_command

    sediment_command(
        run_id=run_id,
        run_dir=run_dir,
        staging_dir=staging_dir,
        bundle=bundle,
        output=output,
        open_pr=open_pr,
    )


@app.command()
def serve_mcp(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(7331, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    # Demo: python -m hmopt.cli serve-mcp --host 0.0.0.0 --port 7331
    # Purpose: run MCP streamable-http server (+ legacy /tools/call).
    import uvicorn

    uvicorn.run("hmopt.api.mcp_server:app", host=host, port=port, reload=reload)


@app.command()
def list_pipeline_profiles(
    profiles_path: str = typer.Option(
        "configs/pipeline_profiles.yaml",
        help="Path to pipeline profiles YAML",
    ),
    as_json: bool = typer.Option(False, help="Print JSON instead of plain text"),
) -> None:
    profiles = load_pipeline_profiles(profiles_path)
    if as_json:
        payload = {
            name: {
                "title": profile.title,
                "description": profile.description,
                "specialist_hint": profile.specialist_hint,
                "pipeline_card": profile.pipeline_card,
                "primary_goal": profile.primary_goal,
                "plan_reviewer_agent": profile.plan_reviewer_agent,
                "code_reviewer_agent": profile.code_reviewer_agent,
                "tester_agent": profile.tester_agent,
                "validation_mode": profile.validation_mode,
            }
            for name, profile in profiles.items()
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    for name, profile in profiles.items():
        typer.echo(f"{name}: {profile.title}")
        typer.echo(f"  description: {profile.description}")
        typer.echo(f"  specialist_hint: {profile.specialist_hint or 'auto'}")
        typer.echo(f"  pipeline_card: {profile.pipeline_card or '-'}")
        typer.echo(f"  primary_goal: {profile.primary_goal}")
        typer.echo(f"  plan_reviewer_agent: {profile.plan_reviewer_agent}")
        typer.echo(f"  code_reviewer_agent: {profile.code_reviewer_agent}")
        typer.echo(f"  tester_agent: {profile.tester_agent}")
        typer.echo(f"  validation_mode: {profile.validation_mode}")


@app.command()
def start_pipeline(
    profile: str = typer.Option(..., help="Pipeline profile name"),
    target: str = typer.Option(..., help="Target file, symbol, or subsystem"),
    objective: Optional[str] = typer.Option(None, help="Override the profile objective"),
    artifact: list[str] = typer.Option(
        [],
        "--artifact",
        help="Artifact spec kind:path. Repeatable.",
    ),
    config: str = typer.Option("configs/app.yaml", help="Path to config YAML"),
    profiles_path: str = typer.Option(
        "configs/pipeline_profiles.yaml",
        help="Path to pipeline profiles YAML",
    ),
    launch_opencode: bool = typer.Option(
        False,
        help="Launch opencode in the repo root after staging the prompt if the binary exists.",
    ),
    open_cmd: Optional[str] = typer.Option(
        None,
        help="Custom command to launch OpenCode or a wrapper in the repo root.",
    ),
    as_json: bool = typer.Option(False, help="Print JSON instead of prose"),
) -> None:
    cfg = _load_config(config)
    repo_root = Path.cwd()
    profiles = load_pipeline_profiles(profiles_path)
    chosen = profiles.get(profile)
    if chosen is None:
        typer.echo(f"Unknown pipeline profile: {profile}")
        raise typer.Exit(code=1)

    session = initialize_pipeline_session(
        repo_root=repo_root,
        profile=chosen,
        target=target,
        objective=objective,
        artifact_specs=artifact,
    )
    task = session["task"]

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "config_repo_path": cfg.project.repo_path,
                    "state_path": session["state_path"],
                    "prompt_path": session["prompt_path"],
                    "task": task,
                    "prompt_text": session["prompt_text"],
                },
                indent=2,
            )
        )
    else:
        typer.echo(f"Pipeline staged: {task['task_id']}")
        typer.echo(f"Profile: {task['profile']}")
        typer.echo(f"Target: {task['target']}")
        typer.echo(f"State file: {session['state_path']}")
        typer.echo(f"Prompt file: {session['prompt_path']}")
        typer.echo("")
        typer.echo("Paste this into OpenCode if you are not auto-launching:")
        typer.echo("")
        typer.echo(session["prompt_text"])

    if open_cmd:
        subprocess.run(open_cmd, shell=True, cwd=repo_root, check=False)
    elif launch_opencode:
        if shutil.which("opencode") is None:
            typer.echo("OpenCode binary not found in PATH; prompt has been staged on disk.")
            raise typer.Exit(code=1)
        subprocess.run(["opencode"], cwd=repo_root, check=False)


@app.command()
def resume_pipeline(
    state_path: str = typer.Option(
        ".opencode/state/current_task.json",
        help="Path to staged current task state",
    ),
    as_json: bool = typer.Option(False, help="Print JSON instead of prose"),
) -> None:
    session = resume_pipeline_session(repo_root=Path.cwd(), state_path=state_path)
    if as_json:
        typer.echo(json.dumps(session, indent=2))
        return

    task = session["task"]
    typer.echo(f"Current task: {task.get('task_id', '')}")
    typer.echo(f"Profile: {task.get('profile', '')}")
    typer.echo(f"Target: {task.get('target', '')}")
    typer.echo(f"Status: {task.get('status', '')}")
    typer.echo(f"State file: {session['state_path']}")
    typer.echo(f"Prompt file: {session['prompt_path']}")
    if session["prompt_text"]:
        typer.echo("")
        typer.echo(session["prompt_text"])


@app.command("mcp-stdio")
def mcp_stdio() -> None:
    # Demo: python -m hmopt.cli mcp-stdio
    # Purpose: run MCP server over stdio for OpenCode local MCP mode.
    from hmopt.api.mcp_stdio import main as run_mcp_stdio

    run_mcp_stdio()


if __name__ == "__main__":
    app()
