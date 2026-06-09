"""Subsumption detector (engine A central-only, design §10.1.b / §11.5, P2-9).

The third merge judgment beyond dedup/conflict: does one record *generalize*
another (B concept-subsumes A)? Example the design calls out:

    A = "in shrink_node, hoist sc->priority out of the reclaim loop"  (specific,
        target-level)
    B = "in mm reclaim hot loops, hoist loop-invariant state out"     (general,
        pattern-level)
    => B subsumes A.  This is NOT duplicate (different scope) and NOT
       contradiction (same polarity).

Discipline:
- Build links only: A.subsumed_by += [B], B.subsumes += [A], and A becomes a
  *source/evidence* of B. **A is never deduped/absorbed/deleted.**
- Emit a promotion signal for B only once it subsumes >= 2 *distinct* instances
  (different target / contributor) — guards against single-instance over-
  generalization (§11.5).

Entailment is offline-heuristic by default (same mechanism + one general/one
specific + text overlap); an `llm(prompt)->str` judge may be injected to decide
ambiguous pairs. CLI prints detected links over the current hub knowledge.

CLI:
    python tools/subsumption.py [--emit-only]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hub_records import HubRecord, load_hub_knowledge  # type: ignore
from similarity import alias_hit, text_similarity  # type: ignore

MIN_OVERLAP = 0.12
MIN_INSTANCES_TO_EMIT = 2


@dataclass
class SubsumptionLink:
    general_id: str
    specific_id: str
    score: float
    general: HubRecord
    specific: HubRecord


def heuristic_judge(a: HubRecord, b: HubRecord) -> str:
    """Return 'a_subsumes_b' | 'b_subsumes_a' | 'none' (offline).

    Same optimization concept (shared mechanism OR strong text overlap) + exactly
    one general and one specific record. Generalizations (e.g. an `H` heuristic)
    carry no `mechanism` field, so text overlap is the primary signal there.
    """
    if not (a.is_claim and b.is_claim):
        return "none"  # ledger entries are verdicts, not generalizable claims
    # opposite conclusions are a *contradiction*, not a generalization — a
    # "don't inline" pattern must not subsume an "inlining helped" fact. Defer to
    # the dedup/conflict route (subsumption runs first in curate_batch, so this
    # guard is what keeps a real contradiction from being relabeled subsumption).
    pa, pb = a.polarity(), b.polarity()
    if pa != 0 and pb != 0 and pa != pb:
        return "none"
    sim = text_similarity(a.text(), b.text())
    # same optimization concept: a shared mechanism is the strong signal; when one
    # side is a mechanism-less generalization (an `H` heuristic), require strong
    # text overlap instead.
    same_concept = alias_hit(a.aliases, b.aliases) or sim >= 0.30
    if not same_concept or sim < MIN_OVERLAP:
        return "none"
    ga, gb = a.granularity, b.granularity
    if ga == gb:
        return "none"  # subsumption needs exactly one general + one specific
    return "a_subsumes_b" if ga == "general" else "b_subsumes_a"


def judge_subsumption(a: HubRecord, b: HubRecord, llm=None) -> str:
    if llm is None:
        return heuristic_judge(a, b)
    try:  # pragma: no cover - exercised only with a real judge injected
        verdict = str(llm(_entailment_prompt(a, b))).strip().lower()
        if "a_subsumes_b" in verdict:
            return "a_subsumes_b"
        if "b_subsumes_a" in verdict:
            return "b_subsumes_a"
        if "none" in verdict:
            return "none"
    except Exception:  # noqa: BLE001 - never let a flaky judge break curation
        pass
    return heuristic_judge(a, b)


def _entailment_prompt(a: HubRecord, b: HubRecord) -> str:
    return (
        "Decide the generalization relation between two kernel-optimization "
        "knowledge records sharing a mechanism. Answer exactly one of "
        "a_subsumes_b / b_subsumes_a / none.\n"
        f"A ({a.id}, granularity={a.granularity}): {a.text()[:600]}\n"
        f"B ({b.id}, granularity={b.granularity}): {b.text()[:600]}\n"
    )


def _orient(a: HubRecord, b: HubRecord, verdict: str) -> SubsumptionLink | None:
    if verdict == "a_subsumes_b":
        return SubsumptionLink(a.id, b.id, text_similarity(a.text(), b.text()), a, b)
    if verdict == "b_subsumes_a":
        return SubsumptionLink(b.id, a.id, text_similarity(a.text(), b.text()), b, a)
    return None


def detect_for_incoming(incoming: HubRecord, hub: list[HubRecord], k: int = 8, llm=None
                        ) -> list[SubsumptionLink]:
    ranked = sorted(hub, key=lambda h: text_similarity(incoming.text(), h.text()), reverse=True)[:k]
    out: list[SubsumptionLink] = []
    for ex in ranked:
        if ex.id == incoming.id:
            continue
        link = _orient(incoming, ex, judge_subsumption(incoming, ex, llm=llm))
        if link is not None:
            out.append(link)
    return out


def detect_in_set(records: list[HubRecord], llm=None) -> list[SubsumptionLink]:
    out: list[SubsumptionLink] = []
    for i, a in enumerate(records):
        for b in records[i + 1 :]:
            link = _orient(a, b, judge_subsumption(a, b, llm=llm))
            if link is not None:
                out.append(link)
    return out


def _as_source(specific: HubRecord) -> dict:
    return {"kind": "doc", "ref": specific.id}


def build_links(general_fields: dict, specific_fields: dict, general: HubRecord,
                specific: HubRecord) -> tuple[dict, dict]:
    """Return (general_fields, specific_fields) with the subsumption edges added.

    A (specific) is added as a source/evidence of B (general) — never removed.
    """
    g = dict(general_fields)
    s = dict(specific_fields)
    g["subsumes"] = sorted(set(_as_list(g.get("subsumes")) + [specific.id]))
    s["subsumed_by"] = sorted(set(_as_list(s.get("subsumed_by")) + [general.id]))
    # specific becomes evidence/source of the general record (engine for graduation)
    src_key = "evidence" if general.schema in {"global_lesson", "bad_plan"} else "source"
    srcs = _as_list(g.get(src_key))
    if _as_source(specific) not in srcs:
        srcs.append(_as_source(specific))
    g[src_key] = srcs
    return g, s


def _as_list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def distinct_instances(general: HubRecord, links: list[SubsumptionLink]) -> int:
    """Count distinct subsumed instances by (target_slug, contributor) — the
    §11.5 distinctness criterion ("different target / different contributor").
    Records with neither signal collapse to a single key (conservative: cannot
    fabricate a promotion from anonymous instances)."""
    keys = set()
    for link in links:
        if link.general_id == general.id:
            sp = link.specific
            keys.add((sp.target_slug or "", sp.contributor or ""))
    return len(keys)


def should_emit_promotion(general: HubRecord, links: list[SubsumptionLink]) -> bool:
    return distinct_instances(general, links) >= MIN_INSTANCES_TO_EMIT


def main(argv: list[str]) -> int:
    emit_only = "--emit-only" in argv
    hub = load_hub_knowledge()
    links = detect_in_set(hub)
    by_general: dict[str, list[SubsumptionLink]] = {}
    for link in links:
        by_general.setdefault(link.general_id, []).append(link)
    if not links:
        print("OK — no subsumption links in current hub knowledge.")
        return 0
    for gid, ls in sorted(by_general.items()):
        g = ls[0].general
        emit = should_emit_promotion(g, links)
        if emit_only and not emit:
            continue
        flag = "EMIT-PROMOTION" if emit else "link-only (1 instance)"
        print(f"{gid} subsumes {sorted({x.specific_id for x in ls})}  [{flag}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
