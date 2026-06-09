"""Typer command implementation for Team Skill Hub sediment."""

from __future__ import annotations

from pathlib import Path

import typer

from hmopt.sediment import bundle_staging, extract_run, write_jsonl


def sediment_command(
    run_id: str | None = typer.Option(None, "--run-id", help="Run id under .opencode/local/runs/"),
    run_dir: Path | None = typer.Option(None, "--run-dir", help="Explicit run directory to scan"),
    staging_dir: Path = typer.Option(Path(".opencode/local/sediment_staging"), help="Tier-1 JSONL staging dir"),
    bundle: bool = typer.Option(False, "--bundle", help="Bundle staged JSONL files instead of extracting one run"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output JSONL path"),
    open_pr: bool = typer.Option(False, "--open-pr", help="Prepare a PR bundle summary; network submission is intentionally manual"),
) -> None:
    """Distill local OpenCode artifacts into Team Skill Hub Tier-1 candidates."""

    if bundle:
        out = output or staging_dir / "bundle.jsonl"
        n = bundle_staging(staging_dir, out)
        typer.echo(f"Bundled {n} sediment candidate(s) into {out}")
        if open_pr:
            summary = out.with_suffix(".pr.md")
            summary.write_text(
                "# Team Skill Hub sediment PR\n\n"
                f"- Candidate bundle: `{out}`\n"
                "- Source: local staging\n"
                "- Required checks: lint, redact, dedup, curator review\n",
                encoding="utf-8",
            )
            typer.echo(f"Prepared PR summary at {summary}; submit it to the hub repo after review.")
        return

    root = run_dir or (Path(".opencode/local/runs") / run_id if run_id else None)
    if root is None:
        typer.echo("Provide --run-id/--run-dir, or use --bundle.", err=True)
        raise typer.Exit(code=2)
    if not root.exists():
        typer.echo(f"Run directory not found: {root}", err=True)
        raise typer.Exit(code=2)
    records = extract_run(root)
    out = output or staging_dir / f"{root.name}.jsonl"
    n = write_jsonl(records, out)
    typer.echo(f"Wrote {n} sediment candidate(s) to {out}")
