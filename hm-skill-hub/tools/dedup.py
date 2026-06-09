"""Lightweight duplicate detector for Team Skill Hub knowledge.

It uses deterministic token Jaccard similarity plus alias overlap. This is the
Phase-2 local fallback for the design's embedding/faiss dedup gate; it has no
network or model dependency and emits merge/new/conflict decisions as JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_memory import parse  # type: ignore[import-not-found]

HUB = Path(__file__).resolve().parent.parent
TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")


@dataclass
class Decision:
    candidate_id: str
    existing_id: str | None
    decision: str
    similarity: float
    reason: str


def _tokens(value: Any) -> set[str]:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text) if len(m.group(0)) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_records(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for md in sorted((root / "knowledge").rglob("*.md")) if (root / "knowledge").exists() else []:
        if md.name == "README.md":
            continue
        for rec in parse(md):
            out.append({"id": rec.id, "title": rec.title, "source_path": str(md), **rec.fields})
    return out


def load_candidates(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    return [{"id": rec.id, "title": rec.title, "source_path": str(path), **rec.fields} for rec in parse(path)]


def decide(candidates: list[dict[str, Any]], existing: list[dict[str, Any]], threshold: float) -> list[Decision]:
    decisions: list[Decision] = []
    indexed = [(rec, _tokens(rec)) for rec in existing]
    for cand in candidates:
        cand_tokens = _tokens(cand)
        best_rec: dict[str, Any] | None = None
        best = 0.0
        for rec, toks in indexed:
            sim = _jaccard(cand_tokens, toks)
            aliases = set(cand.get("aliases") or []) & set(rec.get("aliases") or [])
            if aliases:
                sim = max(sim, 0.99)
            if sim > best:
                best = sim
                best_rec = rec
        if best_rec and best >= threshold:
            same_status = cand.get("status", "active") == best_rec.get("status", "active")
            decision = "merge" if same_status else "conflict"
            reason = "similarity/alias overlap" if same_status else "similar content but status differs"
            decisions.append(Decision(str(cand.get("id", "candidate")), str(best_rec.get("id")), decision, round(best, 4), reason))
        else:
            decisions.append(Decision(str(cand.get("id", "candidate")), None, "new", round(best, 4), "below threshold"))
    return decisions


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", nargs="*", type=Path, help="Candidate JSONL/JSON/MD files; defaults to staging/*.jsonl")
    parser.add_argument("--hub", type=Path, default=HUB)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--fail-on-conflict", action="store_true")
    args = parser.parse_args(argv)
    paths = args.candidates or sorted((args.hub / "staging").glob("*.jsonl"))
    existing = load_records(args.hub)
    decisions: list[Decision] = []
    for path in paths:
        if path.exists():
            decisions.extend(decide(load_candidates(path), existing, args.threshold))
    print(json.dumps([asdict(d) for d in decisions], ensure_ascii=False, indent=2))
    if args.fail_on_conflict and any(d.decision == "conflict" for d in decisions):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
