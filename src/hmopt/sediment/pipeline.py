"""Sediment orchestrator: run artifacts -> validated Tier-1 candidate jsonl.

`sediment_run` is what `hmopt sediment` and the pipeline close-out hooks call.
It is non-blocking by contract: extraction failures are collected, not raised,
so the main pipeline is never gated on sedimentation (design §8 cadence).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import extractors as ex
from .readers import RunArtifacts, read_run
from .salience import llm_salience_pass
from .validate import validate_candidate


@dataclass
class SedimentResult:
    run_id: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    invalid: list[tuple[dict[str, Any], list[str]]] = field(default_factory=list)
    out_path: str | None = None

    @property
    def n_valid(self) -> int:
        return len(self.candidates)


def extract_candidates(arts: RunArtifacts, *, contributor: str, llm: Any | None = None,
                       no_llm: bool = False) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []
    seq = {"fact": 0, "anti": 0, "bad": 0, "idea": 0}

    for b in arts.bench:
        if b.verdict in ("landed", "approved"):
            seq["fact"] += 1
            cands.append(ex.bench_to_fact(b, run_id=arts.run_id, contributor=contributor,
                                          seq=seq["fact"]))
    for v in arts.reviews:
        if v.decision == "reject":
            seq["anti"] += 1
            cands.append(ex.review_to_anti_pattern(v, run_id=arts.run_id, contributor=contributor,
                                                   seq=seq["anti"]))
            seq["bad"] += 1
            cands.append(ex.review_to_bad_plan(v, run_id=arts.run_id, seq=seq["bad"]))
    for c in arts.ledger:
        seq["idea"] += 1
        cands.append(ex.ledger_to_idea(c, run_id=arts.run_id, seq=seq["idea"]))

    if not no_llm:
        cands.extend(llm_salience_pass(arts.free_text, llm, run_id=arts.run_id,
                                       contributor=contributor))
    return cands


def sediment_run(
    run_dir: str | Path,
    *,
    out_dir: str | Path,
    run_id: str | None = None,
    contributor: str = "pipeline",
    hub_root: str | Path | None = None,
    llm: Any | None = None,
    no_llm: bool = False,
) -> SedimentResult:
    arts = read_run(run_dir, run_id=run_id)
    raw = extract_candidates(arts, contributor=contributor, llm=llm, no_llm=no_llm)

    result = SedimentResult(run_id=arts.run_id)
    hub = hub_root if hub_root is not None else run_dir
    for cand in raw:
        errs = validate_candidate(cand, hub)
        if errs:
            result.invalid.append((cand, errs))
        else:
            result.candidates.append(cand)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{arts.run_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for cand in result.candidates:
            f.write(json.dumps(cand, ensure_ascii=False) + "\n")
    result.out_path = str(out_path)
    return result


def bundle_staging(staging_dir: str | Path, out_path: str | Path) -> tuple[Path, int]:
    """Collate all per-run staging jsonl into one bundle (for a hub PR)."""
    staging_dir = Path(staging_dir)
    out_path = Path(out_path)
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for jf in sorted(staging_dir.glob("*.jsonl")):
            if jf.resolve() == out_path.resolve():
                continue
            for line in jf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line + "\n")
                    n += 1
    return out_path, n
