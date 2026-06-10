"""Central batch Curator orchestrator (engine A §10.1.b, plan P2-1 engine).

Realizes the §10.1.b decision over a batch of incoming sediment candidates vs
the existing hub knowledge, composing the focused tools:

    subsumption.py  -> generalization links (checked first: a general/specific
                       pair must NOT be deduped away)
    dedup.py        -> merge / conflict / new
    conflict_resolve.py -> double-time supersede (never delete)
    promotion_detector.py -> promote-candidate signals after the batch

Per-candidate decision is one of: subsumption / merge / supersede / drop /
escalate / add. The matching `merge_curator.md` agent prompt produces the same
shape for a human-in-the-loop review; this engine is the deterministic, testable
backbone (and the CI dry-run).

CLI:
    python tools/central_curate.py <incoming.jsonl> [--report out.md]
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conflict_resolve import resolve  # type: ignore
from dedup import classify_one  # type: ignore
from hub_records import HubRecord, load_hub_knowledge, record_from_candidate  # type: ignore
from promotion_detector import detect_promotions  # type: ignore
from subsumption import detect_for_incoming, detect_in_set, should_emit_promotion  # type: ignore


@dataclass
class Decision:
    incoming_id: str
    kind: str  # subsumption | merge | supersede | drop | escalate | add
    target_id: str | None = None
    detail: str = ""


@dataclass
class CurationReport:
    decisions: list[Decision] = field(default_factory=list)
    promotion_signals: list[str] = field(default_factory=list)
    promotions: list = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.decisions:
            out[d.kind] = out.get(d.kind, 0) + 1
        return out


def curate_batch(candidates: list[dict], hub: list[HubRecord], *, llm=None) -> CurationReport:
    report = CurationReport()
    working = list(hub)  # hub + records added during this batch

    for cand in candidates:
        inc = record_from_candidate(cand)

        # 1) subsumption first — a general/specific pair must not be deduped away
        links = detect_for_incoming(inc, working, llm=llm)
        if links:
            link = links[0]
            other = link.specific_id if link.general_id == inc.id else link.general_id
            report.decisions.append(Decision(
                inc.id, "subsumption", other,
                f"{link.general_id} subsumes {link.specific_id} (kept as evidence)"))
            working.append(inc)
            continue

        # 2) dedup
        v = classify_one(inc, working)
        if v.verdict == "merge":
            report.decisions.append(Decision(inc.id, "merge", v.match_id,
                                              "merge provenance + confirmations++"))
            continue
        if v.verdict == "conflict":
            existing = next((w for w in working if w.id == v.match_id), None)
            if existing is not None:
                res = resolve(inc, existing)
                report.decisions.append(Decision(inc.id, res.decision, v.match_id,
                                                  "; ".join(res.actions[:2])))
                if res.decision == "supersede":
                    working.append(inc)
            continue
        # 3) novel / additive
        report.decisions.append(Decision(inc.id, "add", v.match_id, v.rationale))
        working.append(inc)

    # promotion signals: recompute over the FULL working set (hub + batch) so a
    # general record reaching >= 2 distinct subsumed instances across the merge
    # boundary is surfaced (§11.5). report.promotions carries the full candidates.
    claims = [w for w in working if w.is_claim]
    final_links = detect_in_set(claims, llm=llm)
    generals = {link.general_id: link.general for link in final_links}
    report.promotion_signals = sorted(
        gid for gid, g in generals.items() if should_emit_promotion(g, final_links))
    report.promotions = detect_promotions(working)
    return report


def render_report(report: CurationReport) -> str:
    lines = ["# Curation report", "", "## Per-candidate decisions", ""]
    for d in report.decisions:
        lines.append(f"- `{d.incoming_id}` → **{d.kind}**"
                     + (f" (vs `{d.target_id}`)" if d.target_id else "")
                     + (f" — {d.detail}" if d.detail else ""))
    lines += ["", f"**Counts**: {report.counts()}", ""]
    lines += ["## Promotion candidates", ""]
    if report.promotions:
        for c in report.promotions:
            lines.append(f"- [{c.kind}] `{c.proposed_skill}` ← evidence {c.evidence_ids} "
                         f"({c.rationale})")
    else:
        lines.append("- (none)")
    lines += ["", "> Curator suggests; §9 three gates + double review still required. "
              "No physical deletes — superseded records stay auditable.", ""]
    return "\n".join(lines)


def _read_candidates(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main(argv: list[str]) -> int:
    # Accept both --report=out.md and --report out.md (the agent prompt and the
    # usage string both show the space form, which was previously dropped).
    report_path: str | None = None
    positional: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--report="):
            report_path = a.split("=", 1)[1]
        elif a == "--report" and i + 1 < len(argv):
            report_path = argv[i + 1]
            i += 2
            continue
        elif not a.startswith("--"):
            positional.append(a)
        i += 1
    if not positional:
        sys.stderr.write("usage: central_curate.py <incoming.jsonl> [--report out.md]\n")
        return 2
    candidates = _read_candidates(Path(positional[0]))
    report = curate_batch(candidates, load_hub_knowledge())
    md = render_report(report)
    print(md)
    if report_path:
        Path(report_path).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
