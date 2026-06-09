"""Bounded-edit skill optimizer under the eval gate (SkillOpt, design §10.2,
plan P3-5/P3-7).

    def merge_skill_edit(skill, edit):
        if edit in bad_edits: return REJECT
        edit = clip_to_budget(edit, textual_lr)     # bounded add/del/replace
        cand = apply(skill, edit); s = run_evals(cand, suite)
        if s.strictly_better_than(skill.score): accept; write_scorecard
        else: bad_edits.append(edit)
        pareto = update_pareto(...)                  # keep complementary candidates

Discipline (the three SkillOpt safety pieces):
  - **bounded edits** — `max_edits_per_round` (textual learning rate);
  - **bad_edits buffer** — rejected edits are recorded and skipped next time;
  - **slow update** — only a *strictly better* (monotone, no-regression) candidate
    is accepted; everything else is buffered, never force-merged.

The optimizer only *proposes* and (with --apply) writes `best_skill.md`; the
design keeps merge half-automatic (auto-PR, human merge) early on.

HONESTY: `propose_edits` synthesizes its guidance bullet from the SAME
`expected_terms`/`expected_mechanism` the `ProxyScorer` reads back, so a
strictly-better candidate is guaranteed on any failing group. The 0.67→1.00
demo therefore validates the SkillOpt *control flow* (bounded edit → eval gate →
accept/reject → Pareto → bad_edits), NOT edit *discovery* or that keyword
coverage tracks instruction-count reduction. A real scorer (static instr-count
estimator / real-machine A/B) plugs in via the `scorer=` parameter unchanged.

CLI:
    python tools/skill_optimizer.py <skill_dir> [--apply] [--suite=<dir>]
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pareto import ParetoFrontier  # type: ignore
from run_evals import (  # type: ignore
    ProxyScorer, Scorecard, Suite, load_skill, load_suite, run_evals,
)

HUB = Path(__file__).resolve().parent.parent
_PLAYBOOK_HEADER = "## Mechanism playbook"


@dataclass
class Edit:
    op: str  # add | del | replace
    payload: str
    mechanism: str = ""
    rationale: str = ""

    def signature(self) -> str:
        h = hashlib.sha256(f"{self.op}|{self.mechanism}|{self.payload}".encode()).hexdigest()
        return h[:16]


@dataclass
class OptimizeResult:
    baseline: Scorecard
    final: Scorecard
    accepted: list[Edit] = field(default_factory=list)
    rejected: list[Edit] = field(default_factory=list)
    pareto: ParetoFrontier = field(default_factory=ParetoFrontier)


# --- bad_edits buffer -------------------------------------------------------

def _bad_edits_path(skill_dir: Path) -> Path:
    return skill_dir / "bad_edits.jsonl"


def load_bad_edits(skill_dir: Path) -> set[str]:
    p = _bad_edits_path(skill_dir)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.add(json.loads(line).get("signature", ""))
    return out


def append_bad_edit(skill_dir: Path, edit: Edit) -> None:
    with open(_bad_edits_path(skill_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps({"signature": edit.signature(), "op": edit.op,
                            "mechanism": edit.mechanism, "payload": edit.payload}) + "\n")


# --- propose + apply --------------------------------------------------------

def propose_edits(skill_text: str, card: Scorecard, suite: Suite, bad_edits: set[str],
                  max_edits: int = 1) -> list[Edit]:
    """Synthesize bounded add-edits that cover the highest-impact failing group.

    Group failing cases by expected_mechanism; for the heaviest group, build one
    guidance bullet containing the mechanism + its (currently-missing) terms.
    """
    text_l = skill_text.lower()
    groups: dict[str, list] = {}
    for c in suite.cases:
        if not card.passed.get(c.id, False):
            groups.setdefault(c.expected_mechanism, []).append(c)

    ranked = sorted(groups.items(), key=lambda kv: sum(c.weight for c in kv[1]), reverse=True)
    edits: list[Edit] = []
    for mech, cases in ranked:
        terms = sorted({t for c in cases for t in c.expected_terms})
        missing = [t for t in terms if t.lower() not in text_l]
        if not missing and mech.lower() in text_l:
            continue
        pattern = cases[0].subsystem or cases[0].target
        bullet = (f"- **{pattern} hot pattern**: apply `{mech}` — "
                  f"{', '.join(terms)}. Re-measure at the right compare_level.")
        edit = Edit(op="add", payload=bullet, mechanism=mech,
                    rationale=f"covers {len(cases)} failing case(s) missing {missing or [mech]}")
        if edit.signature() in bad_edits:
            continue
        edits.append(edit)
        if len(edits) >= max_edits:
            break
    return edits


def apply_edit(skill_text: str, edit: Edit) -> str:
    if edit.op != "add":
        raise NotImplementedError(f"op {edit.op!r} not supported in this PoC")
    if _PLAYBOOK_HEADER in skill_text:
        # insert the bullet at the end of the playbook section
        lines = skill_text.splitlines()
        idx = next(i for i, ln in enumerate(lines) if ln.strip() == _PLAYBOOK_HEADER)
        j = idx + 1
        while j < len(lines) and not lines[j].startswith("## "):
            j += 1
        lines.insert(j, edit.payload)
        return "\n".join(lines) + ("\n" if skill_text.endswith("\n") else "")
    return skill_text.rstrip() + f"\n\n{_PLAYBOOK_HEADER}\n\n{edit.payload}\n"


# --- optimize loop ----------------------------------------------------------

def optimize(skill_dir: str | Path, suite_dir: str | Path | None = None, *, scorer=None,
             max_rounds: int = 4, max_edits_per_round: int = 1, apply: bool = False
             ) -> OptimizeResult:
    skill_dir = Path(skill_dir)
    scorer = scorer or ProxyScorer()
    if suite_dir is None:
        suite_dir = HUB / "eval" / "task_suites" / "core_optimization_suite"
    suite = load_suite(suite_dir)
    name, version, text = load_skill(skill_dir)

    baseline = run_evals(name, version, text, suite, scorer)
    current = baseline
    bad = load_bad_edits(skill_dir)
    result = OptimizeResult(baseline=baseline, final=baseline)
    result.pareto.add("baseline", baseline.per_instance, {"version": version})

    for _ in range(max_rounds):
        edits = propose_edits(text, current, suite, bad, max_edits=max_edits_per_round)
        if not edits:
            break
        accepted_this_round = False
        for edit in edits:
            cand_text = apply_edit(text, edit)
            cand = run_evals(name, version, cand_text, suite, scorer)
            result.pareto.add(edit.signature(), cand.per_instance,
                              {"mechanism": edit.mechanism, "op": edit.op})
            if cand.strictly_better_than(current):
                text, current = cand_text, cand
                result.accepted.append(edit)
                accepted_this_round = True
                break
            result.rejected.append(edit)
            bad.add(edit.signature())
            if apply:
                append_bad_edit(skill_dir, edit)
        if not accepted_this_round:
            break

    result.final = current
    if apply and result.accepted:
        (skill_dir / "best_skill.md").write_text(text, encoding="utf-8")
        out_dir = skill_dir / "scorecards"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}__{version}.json").write_text(
            json.dumps(current.to_json(), indent=2), encoding="utf-8")
        (skill_dir / "candidates").mkdir(exist_ok=True)
        (skill_dir / "candidates" / "pareto.json").write_text(
            json.dumps(result.pareto.to_json(), indent=2), encoding="utf-8")
    return result


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        sys.stderr.write("usage: skill_optimizer.py <skill_dir> [--apply] [--suite=<dir>]\n")
        return 2
    apply = "--apply" in argv
    suite_dir = next((a.split("=", 1)[1] for a in argv if a.startswith("--suite=")), None)
    res = optimize(args[0], suite_dir, apply=apply)
    print(f"baseline: pass_rate={res.baseline.pass_rate:.2f} mean={res.baseline.mean_score:.3f}")
    print(f"final:    pass_rate={res.final.pass_rate:.2f} mean={res.final.mean_score:.3f}")
    print(f"accepted edits: {len(res.accepted)}  rejected->bad_edits: {len(res.rejected)}  "
          f"pareto: {res.pareto.ids()}")
    for e in res.accepted:
        print(f"  + [{e.mechanism}] {e.payload[:80]}  ({e.rationale})")
    if not apply:
        print("\n(dry-run; pass --apply to write best_skill.md + scorecard + bad_edits)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
