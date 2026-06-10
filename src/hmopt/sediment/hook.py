"""Pipeline close-out hook (plan P1-3): sediment a finished run, non-blocking.

This is the wiring that turns sediment from a CLI-only tool into something a real
optimization run actually fires. It is called from the LangGraph close-out
(`orchestration.graph._report`) and is re-usable from the OpenCode harness.

Contract: it **never raises** and **never gates the run** (design §8 cadence) —
any failure is logged and swallowed so the pipeline's report step is unaffected.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_sediment_input(
    run_dir: str | Path,
    *,
    reviews: list[dict[str, Any]] | None = None,
    bench: list[dict[str, Any]] | None = None,
    ledger: list[dict[str, Any]] | None = None,
    free_text: list[str] | None = None,
) -> Path | None:
    """Write the run's distillable signals into `<run_dir>/sediment_input.json`.

    This is the missing producer: without it the close-out hook always read an
    empty run and yielded 0 candidates. Returns the path, or None when there is
    nothing to sediment (so an empty run stays honestly empty).
    """
    payload: dict[str, Any] = {}
    if reviews:
        payload["reviews"] = reviews
    if bench:
        payload["bench"] = bench
    if ledger:
        payload["ledger"] = ledger
    if free_text:
        payload["free_text"] = free_text
    if not payload:
        return None
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "sediment_input.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def sediment_at_closeout(
    run_dir: str | Path,
    repo_root: str | Path | None = None,
    run_id: str | None = None,
    *,
    contributor: str = "pipeline",
    no_llm: bool = True,
) -> Any | None:
    """Distill a finished run's artifacts into Tier-1 staging candidates.

    Returns the `SedimentResult` (or None on any failure). Output lands in
    `<repo_root>/.opencode/local/sediment_staging/<run_id>.jsonl`. Safe to call
    even when the run produced no sediment-able artifacts (yields 0 candidates).
    """
    try:
        from ..skillhub.resolver import find_hub_root
        from .pipeline import sediment_run

        run_dir = Path(run_dir)
        root = Path(repo_root) if repo_root else _infer_repo_root(run_dir)
        hub_root = find_hub_root(root) or find_hub_root(Path.cwd())
        out_dir = root / ".opencode" / "local" / "sediment_staging"
        res = sediment_run(run_dir, out_dir=out_dir, run_id=run_id,
                           contributor=contributor, hub_root=hub_root, no_llm=no_llm)
        logger.info("sediment close-out: run=%s -> %d candidate(s), %d invalid -> %s",
                    res.run_id, res.n_valid, len(res.invalid), res.out_path)
        return res
    except Exception as e:  # noqa: BLE001 - close-out must never break the run
        logger.warning("sediment close-out hook skipped (non-fatal): %s", e)
        return None


def _infer_repo_root(run_dir: Path) -> Path:
    """Walk up from the run dir to the repo root (the ancestor with .opencode or
    a hub); fall back to cwd."""
    for p in [run_dir, *run_dir.parents]:
        if (p / ".opencode").exists() or (p / "hm-skill-hub").exists() or (p / ".git").exists():
            return p
    return Path.cwd()
