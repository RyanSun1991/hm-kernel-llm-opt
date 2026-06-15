"""Central dedup detector (engine A, design §10.1.b / plan P2-2).

Three-state classification of each incoming candidate against the existing hub
knowledge:

    merge    -- near-duplicate (same scope + mechanism + same conclusion):
                the Curator should merge provenance + confirmations++, NOT add a
                second record with a new id.
    conflict -- same (scope, mechanism) but opposite conclusion: needs
                conflict_resolve.py (double-time) or human escalation.
    new      -- everything else (novel / additive: temporal / conditional /
                evidence / selector are all handled additively elsewhere).

Similarity is offline (token-hashing + Jaccard); a real embedder can replace it
without changing the verdict logic. CI gate (`--check`) fails on any unresolved
`conflict`, so a PR can't merge contradictory knowledge silently.

CLI:
    python tools/dedup.py <incoming.jsonl> [--check] [--threshold 0.82]
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hub_records import HubRecord, load_hub_knowledge, record_from_candidate  # type: ignore
from similarity import alias_hit, text_similarity  # type: ignore

DEFAULT_THRESHOLD = 0.82


@dataclass
class DedupVerdict:
    incoming_id: str
    verdict: str  # merge | conflict | new
    match_id: str | None
    score: float
    rationale: str


def _same_scope(a: HubRecord, b: HubRecord) -> bool:
    if a.target_slug and b.target_slug:
        return a.target_slug == b.target_slug
    if a.subsystem and b.subsystem:
        return a.subsystem == b.subsystem
    return a.scope_level == "global" and b.scope_level == "global"


def _related(a: HubRecord, b: HubRecord) -> bool:
    mech = alias_hit(a.aliases, b.aliases)
    return (_same_scope(a, b) and mech) or (mech and text_similarity(a.text(), b.text()) >= 0.3)


def classify_one(inc: HubRecord, existing: list[HubRecord], threshold: float = DEFAULT_THRESHOLD
                 ) -> DedupVerdict:
    best: tuple[float, HubRecord] | None = None
    for ex in existing:
        if ex.status != "active":
            continue
        if not _related(inc, ex):
            continue
        sim = text_similarity(inc.text(), ex.text())
        if best is None or sim > best[0]:
            best = (sim, ex)

    if best is None:
        return DedupVerdict(inc.id, "new", None, 0.0, "no related active record")

    sim, ex = best
    pol_i, pol_e = inc.polarity(), ex.polarity()
    opposite = pol_i != 0 and pol_e != 0 and pol_i != pol_e
    same_cond = (inc.applies_when.strip().lower() == ex.applies_when.strip().lower())

    if opposite and same_cond:
        return DedupVerdict(inc.id, "conflict", ex.id, sim,
                            "same (scope, mechanism, condition), opposite conclusion")
    # `merge` (= duplicate) requires the SAME scope — two near-identical facts at
    # different target_slug are distinct per-target instances, not duplicates
    # (collapsing them loses provenance + starves the >=2-instance promotion rule).
    if sim >= threshold and not opposite and _same_scope(inc, ex):
        return DedupVerdict(inc.id, "merge", ex.id, sim,
                            "near-duplicate: same scope/mechanism/conclusion")
    return DedupVerdict(inc.id, "new", ex.id, sim, "related but distinct (additive)")


def classify_candidates(candidates: list[dict], existing: list[HubRecord],
                        threshold: float = DEFAULT_THRESHOLD) -> list[DedupVerdict]:
    return [classify_one(record_from_candidate(c), existing, threshold) for c in candidates]


def _read_candidates(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main(argv: list[str]) -> int:
    check = "--check" in argv
    threshold = DEFAULT_THRESHOLD
    positional: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--threshold" and i + 1 < len(argv):  # space form
            threshold = float(argv[i + 1])
            i += 2
            continue
        if a.startswith("--threshold="):              # equals form
            threshold = float(a.split("=", 1)[1])
        elif not a.startswith("--"):
            positional.append(a)
        i += 1
    args = positional
    if not args:
        sys.stderr.write("usage: dedup.py <incoming.jsonl> [--check] [--threshold 0.82]\n")
        return 2
    candidates = _read_candidates(Path(args[0]))
    existing = load_hub_knowledge()
    verdicts = classify_candidates(candidates, existing, threshold)
    n_conflict = 0
    for v in verdicts:
        print(f"{v.incoming_id:>8}  {v.verdict:<8} match={v.match_id or '-':<8} "
              f"sim={v.score:.2f}  {v.rationale}")
        if v.verdict == "conflict":
            n_conflict += 1
    if check and n_conflict:
        sys.stderr.write(f"\n{n_conflict} unresolved conflict(s) — resolve before merge.\n")
        return 1
    print(f"\nOK — {len(verdicts)} candidate(s): "
          f"{sum(v.verdict=='merge' for v in verdicts)} merge, "
          f"{sum(v.verdict=='new' for v in verdicts)} new, {n_conflict} conflict.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
