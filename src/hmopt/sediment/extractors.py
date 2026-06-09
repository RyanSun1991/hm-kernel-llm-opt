"""Extract Tier-1 Team Skill Hub sediment candidates from local OpenCode runs.

The extractors are intentionally conservative and file-based: they never call an
LLM, never mutate hub records directly, and only create schema-shaped candidate
objects for later curator review.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BENCH_KEYS = ("delta_pct", "instruction_delta_pct", "instr_count_delta", "pass_rate")
IDEA_HDR = re.compile(r"^###\s+(?P<id>L[0-9]{3,})(?:\s+[—\-:]\s+(?P<title>.+?))?\s*$", re.MULTILINE)
STATUS_RE = re.compile(r"^-\s+\*\*status\*\*\s*:\s*(?P<status>\S+)", re.MULTILINE)


@dataclass(frozen=True)
class SedimentRecord:
    """JSONL-friendly candidate emitted by ``hmopt sediment``."""

    id: str
    type: str
    title: str
    body: str
    scope: dict[str, Any]
    source: list[dict[str, str]]
    created_at: str
    maturity: str = "L1"
    status: str = "active"
    evidence: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    contributor: str = "hmopt-sediment"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:8].upper()
    return f"{prefix}{int(digest, 16) % 900000 + 100000}"


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _iter_files(root: Path, suffixes: tuple[str, ...]) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def extract_bench_records(run_dir: Path) -> list[SedimentRecord]:
    """Convert benchmark / validation JSON into fact candidates."""

    out: list[SedimentRecord] = []
    for path in _iter_files(run_dir, (".json",)):
        if not any(token in path.name.lower() for token in ("bench", "score", "validation", "report")):
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        evidence = {k: data[k] for k in BENCH_KEYS if isinstance(data.get(k), (int, float))}
        if not evidence:
            continue
        target = str(data.get("target") or data.get("workload") or run_dir.name)
        title = f"Observed benchmark delta for {target}"
        out.append(
            SedimentRecord(
                id=_stable_id("F", str(path), json.dumps(evidence, sort_keys=True)),
                type="fact",
                title=title,
                body=f"Benchmark artifact {path.name} reports {evidence}.",
                scope={"level": "global", "target_slug": target},
                source=[{"kind": "bench", "ref": str(path)}],
                evidence=evidence,
                tags=["bench", "sediment"],
                created_at=_now(),
            )
        )
    return out


def extract_review_records(run_dir: Path) -> list[SedimentRecord]:
    """Extract anti-pattern candidates from reviews mentioning rejection/regression."""

    out: list[SedimentRecord] = []
    for path in _iter_files(run_dir, (".md", ".txt")):
        name = path.name.lower()
        if "review" not in name and "decision" not in name:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        if not any(word in lowered for word in ("reject", "rejected", "regression", "bad plan", "反模式", "拒绝", "回退")):
            continue
        first = next((ln.strip("# -*") for ln in text.splitlines() if ln.strip()), path.stem)
        out.append(
            SedimentRecord(
                id=_stable_id("A", str(path), first),
                type="anti_pattern",
                title=first[:160],
                body="Review/decision artifact contains rejection or regression evidence; curator must generalize before L2.",
                scope={"level": "global"},
                source=[{"kind": "review", "ref": str(path)}],
                evidence={"confirmations": 1},
                tags=["review", "sediment"],
                created_at=_now(),
            )
        )
    return out


def extract_idea_records(run_dir: Path) -> list[SedimentRecord]:
    """Convert landed/approved idea-ledger rows into playbook-step candidates."""

    out: list[SedimentRecord] = []
    for path in _iter_files(run_dir, (".md",)):
        if "idea" not in path.name.lower() and "ledger" not in path.name.lower():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = list(IDEA_HDR.finditer(text))
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            block = text[start:end].strip()
            status_match = STATUS_RE.search(block)
            status = status_match.group("status") if status_match else "unknown"
            if status not in {"approved", "landed"}:
                continue
            title = match.group("title") or match.group("id")
            out.append(
                SedimentRecord(
                    id=_stable_id("R", match.group("id"), title, str(path)),
                    type="playbook_step",
                    title=title[:160],
                    body=block[:2000] or title,
                    scope={"level": "global"},
                    source=[{"kind": "run_id", "ref": f"{run_dir.name}:{match.group('id')}"}],
                    evidence={"confirmations": 1},
                    tags=["idea-ledger", "sediment"],
                    created_at=_now(),
                )
            )
    return out


def extract_run(run_dir: Path) -> list[SedimentRecord]:
    """Run every extractor over a local run directory."""

    run_dir = run_dir.resolve()
    records: list[SedimentRecord] = []
    records.extend(extract_bench_records(run_dir))
    records.extend(extract_review_records(run_dir))
    records.extend(extract_idea_records(run_dir))
    seen: set[str] = set()
    deduped: list[SedimentRecord] = []
    for rec in records:
        if rec.id not in seen:
            seen.add(rec.id)
            deduped.append(rec)
    return deduped


def write_jsonl(records: Iterable[SedimentRecord], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def bundle_staging(staging_dir: Path, output: Path) -> int:
    """Concatenate staged JSONL files into one review bundle."""

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as dst:
        for src in sorted(staging_dir.glob("*.jsonl")):
            for line in src.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    dst.write(line.rstrip() + "\n")
                    count += 1
    return count
