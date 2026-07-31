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
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml

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


# --------------------------------------------------------------------------- #
# --apply: materialize the safe `add` decisions into knowledge/**/*.md
# (auto-assign stable ids + scope-derived paths). merge/conflict/subsumption are
# left for a human — they mutate existing records. Mirrors tools/path_scope.py.
# --------------------------------------------------------------------------- #
_GLOBAL_LESSON_DIR = {
    "heuristic": "heuristics",
    "anti_pattern": "anti_patterns",
    "validation_pitfall": "validation_pitfalls",
}


def _fname_slug(rec: dict) -> str:
    text = str(rec.get("title") or rec.get("lesson") or rec.get("mechanism") or rec.get("id", ""))
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:40].rstrip("-") or "record"


def _safe_path_component(value: object) -> str | None:
    """Reject candidate-controlled path traversal before materialization."""
    text = str(value or "")
    if (
        not text
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", text)
    ):
        return None
    return text


def _dest_path(schema: str, rec: dict, new_id: str) -> PurePosixPath | None:
    """Reverse of path_scope.derive_path_scope: scope -> knowledge-relative path."""
    fn = f"{new_id}-{_fname_slug(rec)}.md"
    if schema == "memory_item":
        scope = rec.get("scope") or {}
        if scope.get("level") == "subsystem" and scope.get("subsystem"):
            subsystem = _safe_path_component(scope["subsystem"])
            return PurePosixPath("subsystems") / subsystem / fn if subsystem else None
        if scope.get("target_slug"):
            slug = _safe_path_component(scope["target_slug"])
            return PurePosixPath("targets") / slug / "facts" / fn if slug else None
        return None
    if schema == "global_lesson":
        cat = _GLOBAL_LESSON_DIR.get(rec.get("kind"))
        return PurePosixPath("global") / cat / fn if cat else None
    if schema == "bad_plan":
        subs = (rec.get("applies_to") or {}).get("subsystems")
        if not subs or subs == ["*"]:
            return PurePosixPath("global") / "bad_plans" / fn
        subsystem = _safe_path_component(subs[0])
        return PurePosixPath("subsystems") / subsystem / "bad_plans" / fn if subsystem else None
    if schema == "idea":
        slug = _safe_path_component(rec.get("target_slug"))
        return PurePosixPath("targets") / slug / "idea_ledger" / fn if slug else None
    return None


def _next_id(prefix: str, used: set[str]) -> str:
    nums = [int(i[1:]) for i in used if i[:1] == prefix and i[1:].isdigit()]
    return f"{prefix}{(max(nums) + 1) if nums else 1:03d}"


def _to_markdown(rec: dict) -> str:
    rec = dict(rec)
    body = (rec.pop("body", "") or "").strip()
    if not body:
        headline = rec.get("title") or rec.get("lesson") or rec.get("mechanism") or rec["id"]
        body = f"# {rec['id']} — {headline}"
    fm = yaml.safe_dump(rec, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm}\n---\n\n{body}\n"


def apply_additions(report: CurationReport, candidates: list[dict], hub: list,
                    knowledge_dir: Path, *, write: bool) -> list[tuple]:
    """Materialize (or plan) the `add` decisions. Returns (old_id, new_id, rel_path, status)."""
    cand_by_id = {c["record"]["id"]: c for c in candidates}
    used = {r.id for r in hub}
    out: list[tuple] = []
    for d in report.decisions:
        if d.kind != "add":
            out.append((d.incoming_id, None, None, f"SKIP ({d.kind} — needs human)"))
            continue
        c = cand_by_id.get(d.incoming_id)
        if c is None:
            out.append((d.incoming_id, None, None, "SKIP (candidate not found)"))
            continue
        rec = dict(c["record"])
        new_id = _next_id(rec["id"][:1], used)
        rel = _dest_path(c["schema"], rec, new_id)
        if rel is None:
            out.append((d.incoming_id, None, None, "SKIP (cannot derive path — fix scope)"))
            continue
        used.add(new_id)
        rec["id"] = new_id
        if write:
            root = knowledge_dir.resolve()
            dest = (root / Path(str(rel))).resolve()
            if not dest.is_relative_to(root):
                out.append((d.incoming_id, None, None, "SKIP (unsafe path escaped knowledge root)"))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(_to_markdown(rec), encoding="utf-8")
        out.append((d.incoming_id, new_id, str(rel), "WROTE" if write else "PLAN"))
    return out


def render_apply(actions: list[tuple], *, write: bool) -> str:
    head = "## Materialized into knowledge/" if write else "## Apply plan (dry-run — pass --apply to write)"
    lines = ["", head, ""]
    for old, new, rel, status in actions:
        lines.append(f"- `{old}` → `{new or '-'}`  {rel or ''}  [{status}]")
    lines.append("")
    lines.append("> Stable ids auto-assigned (next-free per prefix). Review `git diff`, "
                 "run `python tools/lint.py`, then commit. Content quality (titles, real "
                 "mechanism/target) is NOT auto-fixed — edit the candidate first.")
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
    knowledge_dir = "knowledge"
    apply = False
    plan = False
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
        elif a.startswith("--knowledge-dir="):
            knowledge_dir = a.split("=", 1)[1]
        elif a == "--apply":
            apply = True
        elif a == "--plan":
            plan = True
        elif not a.startswith("--"):
            positional.append(a)
        i += 1
    if not positional:
        sys.stderr.write(
            "usage: central_curate.py <incoming.jsonl> [--report out.md] "
            "[--plan | --apply] [--knowledge-dir DIR]\n")
        return 2
    candidates = _read_candidates(Path(positional[0]))
    hub = load_hub_knowledge()
    report = curate_batch(candidates, hub)
    md = render_report(report)
    print(md)
    if report_path:
        Path(report_path).write_text(md, encoding="utf-8")
    if apply or plan:
        actions = apply_additions(report, candidates, hub, Path(knowledge_dir), write=apply)
        print(render_apply(actions, write=apply))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
