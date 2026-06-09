"""Promotion-candidate auto-detector (design §11.5, plan P2-8).

Two input paths feed candidate detection; the detector only **suggests** — it
opens a `promote-candidate` PR for human review and NEVER merges (§11.5
discipline, no exemption):

    (a) clustering   -- records sharing (mechanism, scope-level) whose summed
                        `confirmations` >= 3 across >= 2 distinct contributors.
    (b) subsumption  -- a general record that subsumes >= 2 distinct instances
                        (fed by subsumption.py / the `subsumes[]` field). The
                        subsumed instances are carried as evidence, NOT deleted.

Output is a list of candidates + a ready-to-file PR body. Graduating knowledge
into a `technique/<mechanism>` skill is the engine for §6.2 "knowledge -> skill".

CLI:
    python tools/promotion_detector.py [--pr-body]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hub_records import HubRecord, load_hub_knowledge  # type: ignore
from subsumption import detect_in_set, distinct_instances  # type: ignore

MIN_CONFIRMATIONS = 3
MIN_CONTRIBUTORS = 2
MIN_SUBSUMED = 2


@dataclass
class PromotionCandidate:
    kind: str  # cluster | subsumption
    mechanism: str
    proposed_skill: str
    member_ids: list[str]
    evidence_ids: list[str]
    contributors: list[str]
    confirmations: int
    rationale: str = ""


def _cluster_candidates(records: list[HubRecord]) -> list[PromotionCandidate]:
    groups: dict[tuple[str, str], list[HubRecord]] = defaultdict(list)
    for r in records:
        if r.status != "active" or not r.mechanism:
            continue
        groups[(r.mechanism, r.scope_level or "?")].append(r)
    out: list[PromotionCandidate] = []
    for (mech, _level), members in sorted(groups.items()):
        contributors = {m.contributor for m in members if m.contributor}
        confirmations = sum(m.confirmations for m in members)
        if confirmations >= MIN_CONFIRMATIONS and len(contributors) >= MIN_CONTRIBUTORS:
            out.append(PromotionCandidate(
                kind="cluster", mechanism=mech,
                proposed_skill=f"technique/{mech}",
                member_ids=sorted(m.id for m in members),
                evidence_ids=sorted(m.id for m in members),
                contributors=sorted(contributors),
                confirmations=confirmations,
                rationale=f"{len(members)} records, {confirmations} confirmations "
                          f"across {len(contributors)} contributors",
            ))
    return out


def _subsumption_candidates(records: list[HubRecord]) -> list[PromotionCandidate]:
    by_id = {r.id: r for r in records}
    links = detect_in_set(records)
    out: list[PromotionCandidate] = []
    # path 1: explicit `subsumes[]` fields already present (fed by P2-9)
    seen: set[str] = set()
    for r in records:
        subsumed = [s for s in r.subsumes if s in by_id]
        instances = {(by_id[s].target_slug or by_id[s].contributor or s) for s in subsumed}
        if len(instances) >= MIN_SUBSUMED:
            seen.add(r.id)
            out.append(_mk_subsumption(r, subsumed))
    # path 2: links discovered now with >= 2 distinct instances
    for r in records:
        if r.id in seen:
            continue
        if distinct_instances(r, links) >= MIN_SUBSUMED:
            subsumed = sorted({x.specific_id for x in links if x.general_id == r.id})
            out.append(_mk_subsumption(r, subsumed))
    return out


def _mk_subsumption(general: HubRecord, subsumed_ids: list[str]) -> PromotionCandidate:
    mech = general.mechanism or "general"
    return PromotionCandidate(
        kind="subsumption", mechanism=mech,
        proposed_skill=f"technique/{mech}" if general.mechanism else f"technique/{general.id.lower()}",
        member_ids=[general.id],
        evidence_ids=sorted(subsumed_ids),  # subsumed instances PRESERVED as evidence
        contributors=[general.contributor] if general.contributor else [],
        confirmations=general.confirmations,
        rationale=f"{general.id} generalizes {len(subsumed_ids)} distinct instances "
                  f"(kept as evidence, not absorbed)",
    )


def detect_promotions(records: list[HubRecord]) -> list[PromotionCandidate]:
    return _cluster_candidates(records) + _subsumption_candidates(records)


def pr_body(c: PromotionCandidate) -> str:
    return (
        f"## promote-candidate: `{c.proposed_skill}`\n\n"
        f"- **path**: {c.kind}\n- **mechanism**: `{c.mechanism}`\n"
        f"- **members**: {', '.join(c.member_ids)}\n"
        f"- **evidence (preserved instances)**: {', '.join(c.evidence_ids)}\n"
        f"- **contributors**: {', '.join(c.contributors) or '—'}\n"
        f"- **confirmations**: {c.confirmations}\n\n"
        f"_{c.rationale}_\n\n"
        f"> Detector only suggests — requires §9 three gates + double review. "
        f"Subsumed instances are kept as evidence, never deleted.\n"
    )


def main(argv: list[str]) -> int:
    want_pr = "--pr-body" in argv
    records = load_hub_knowledge()
    cands = detect_promotions(records)
    if not cands:
        print("OK — no promotion candidates (no cluster/subsumption threshold met).")
        return 0
    for c in cands:
        if want_pr:
            print(pr_body(c))
        else:
            print(f"[{c.kind:<11}] {c.proposed_skill:<28} members={c.member_ids} "
                  f"evidence={c.evidence_ids}  ({c.rationale})")
    print(f"\n{len(cands)} promote-candidate(s) — suggestions only, never auto-merged.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
