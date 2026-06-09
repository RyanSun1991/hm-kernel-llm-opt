"""Read a run directory into structured sediment inputs.

Primary contract: the pipeline writes a structured `sediment_input.json` at
close-out (deterministic, lossless). We also collect free-form text
(`*_design.md`, `reviews/*.md`, `plans/*.md`) to feed the optional LLM salience
pass (§8 stage 2). A best-effort markdown bench scanner is provided as a
fallback for hand-authored runs.
"""
from __future__ import annotations

import dataclasses
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
    parse_errors: list[str] = field(default_factory=list)


def _safe_build(cls, d: object, errors: list[str], label: str):
    """Build a dataclass from a dict, dropping unexpected keys and collecting
    (not raising) on malformed entries — sediment must never gate the run (§8)."""
    if not isinstance(d, dict):
        errors.append(f"{label}: expected a mapping, got {type(d).__name__}")
        return None
    valid = {f.name for f in dataclasses.fields(cls)}
    kw = {k: v for k, v in d.items() if k in valid}
    try:
        return cls(**kw)
    except (TypeError, ValueError) as e:
        errors.append(f"{label}: {e}")
        return None


def _from_manifest(data: dict, run_id: str) -> RunArtifacts:
    arts = RunArtifacts(run_id=run_id)
    for i, b in enumerate(data.get("bench", []) or []):
        if (obj := _safe_build(BenchResult, b, arts.parse_errors, f"bench[{i}]")) is not None:
            arts.bench.append(obj)
    for i, r in enumerate(data.get("reviews", []) or []):
        if (obj := _safe_build(ReviewVerdict, r, arts.parse_errors, f"reviews[{i}]")) is not None:
            arts.reviews.append(obj)
    for i, c in enumerate(data.get("ledger", []) or []):
        if (obj := _safe_build(LedgerChange, c, arts.parse_errors, f"ledger[{i}]")) is not None:
            arts.ledger.append(obj)
    ft = data.get("free_text", []) or []
    arts.free_text.extend(str(t) for t in ft if t)
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
    """Read a run dir into structured inputs. Never raises — malformed manifest
    entries are collected into `RunArtifacts.parse_errors` (§8 non-blocking)."""
    run_dir = Path(run_dir)
    rid = run_id or run_dir.name
    manifest = run_dir / "sediment_input.json"
    arts: RunArtifacts | None = None
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            arts = RunArtifacts(run_id=rid, parse_errors=[f"sediment_input.json: {e}"])
        else:
            if isinstance(data, dict):
                arts = _from_manifest(data, rid)
            else:
                arts = RunArtifacts(run_id=rid,
                                    parse_errors=["sediment_input.json: top-level must be a mapping"])
    if arts is None:  # no manifest -> best-effort markdown scan
        arts = RunArtifacts(run_id=rid)
        arts.bench = _scan_bench_markdown(run_dir)
    if not arts.free_text:
        arts.free_text = _scan_free_text(run_dir)
    return arts
