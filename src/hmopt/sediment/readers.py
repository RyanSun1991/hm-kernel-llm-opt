"""Read a run directory into structured sediment inputs.

Primary contract: the pipeline writes a structured `sediment_input.json` at
close-out (deterministic, lossless). We also collect free-form text
(`*_design.md`, `reviews/*.md`, `plans/*.md`) to feed the optional LLM salience
pass (§8 stage 2). A best-effort markdown bench scanner is provided as a
fallback for hand-authored runs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .extractors import BenchResult, LedgerChange, ReviewVerdict

_DELTA_RE = re.compile(r"delta[_\s-]*pct\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_COMPARE_RE = re.compile(r"compare[_\s-]*level\s*[:=]\s*(total|process|thread|lib|function)", re.I)
_MECH_RE = re.compile(r"mechanism\s*[:=]\s*([a-z0-9-]+)", re.IGNORECASE)


@dataclass
class RunArtifacts:
    run_id: str
    bench: list[BenchResult] = field(default_factory=list)
    reviews: list[ReviewVerdict] = field(default_factory=list)
    ledger: list[LedgerChange] = field(default_factory=list)
    free_text: list[str] = field(default_factory=list)


def _from_manifest(data: dict, run_id: str) -> RunArtifacts:
    arts = RunArtifacts(run_id=run_id)
    for b in data.get("bench", []) or []:
        arts.bench.append(BenchResult(**b))
    for r in data.get("reviews", []) or []:
        arts.reviews.append(ReviewVerdict(**r))
    for c in data.get("ledger", []) or []:
        arts.ledger.append(LedgerChange(**c))
    arts.free_text.extend(data.get("free_text", []) or [])
    return arts


def _scan_free_text(run_dir: Path) -> list[str]:
    texts: list[str] = []
    for sub in ("", "reviews", "plans"):
        d = run_dir / sub if sub else run_dir
        if not d.exists():
            continue
        for p in sorted(d.glob("*.md")):
            try:
                texts.append(p.read_text(encoding="utf-8"))
            except OSError:
                continue
    return texts


def _scan_bench_markdown(run_dir: Path) -> list[BenchResult]:
    out: list[BenchResult] = []
    bench_dir = run_dir / "bench"
    if not bench_dir.exists():
        return out
    for p in sorted(bench_dir.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        dm, cm, mm = _DELTA_RE.search(text), _COMPARE_RE.search(text), _MECH_RE.search(text)
        if dm and cm and mm:
            out.append(
                BenchResult(
                    mechanism=mm.group(1),
                    target=p.stem,
                    delta_pct=float(dm.group(1)),
                    compare_level=cm.group(1).lower(),  # type: ignore[arg-type]
                    validation_path=str(p),
                )
            )
    return out


def read_run(run_dir: str | Path, run_id: str | None = None) -> RunArtifacts:
    run_dir = Path(run_dir)
    rid = run_id or run_dir.name
    manifest = run_dir / "sediment_input.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        arts = _from_manifest(data, rid)
    else:
        arts = RunArtifacts(run_id=rid)
        arts.bench = _scan_bench_markdown(run_dir)
    if not arts.free_text:
        arts.free_text = _scan_free_text(run_dir)
    return arts
