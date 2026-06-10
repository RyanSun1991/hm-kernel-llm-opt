"""Skill eval executor (engine B / SkillOpt, design §9 / §10.2, plan P3-2/P3-3).

Scores a skill version (its `best_skill.md` artifact) against a task suite and
writes a scorecard. The default `ProxyScorer` is a **static keyword/mechanism-
coverage proxy** standing in for real-machine A/B instruction-count delta —
design §15 names real-machine ground truth as the long pole, so Phase 3 starts
on a proxy and the scorer is **pluggable** (swap in a static instr-count
estimator or a real-machine A/B harness without changing the gate/optimizer).

Honesty: the proxy measures "does the skill text surface the guidance that would
steer toward the expected mechanism for each case". It is NOT a claim that
keyword coverage == kernel speedup; it exercises the SkillOpt control flow
(bounded edit → eval gate → accept/reject → bad_edits → Pareto) end-to-end.

CLI:
    python tools/run_evals.py <skill_dir> [--suite <dir>] [--baseline <card.json>]
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("PyYAML required\n")
    sys.exit(2)

HUB = Path(__file__).resolve().parent.parent
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


@dataclass
class Case:
    id: str
    target: str
    expected_mechanism: str
    expected_terms: list[str] = field(default_factory=list)
    avoid_terms: list[str] = field(default_factory=list)
    subsystem: str = ""
    weight: float = 1.0


@dataclass
class Suite:
    name: str
    cases: list[Case]
    pass_threshold: float = 0.6


@dataclass
class Scorecard:
    skill: str
    version: str
    suite: str
    per_instance: dict[str, float] = field(default_factory=dict)
    passed: dict[str, bool] = field(default_factory=dict)
    pass_rate: float = 0.0
    mean_score: float = 0.0
    n: int = 0
    run_at: str = ""

    def regression_rate_vs(self, baseline: "Scorecard | None") -> float:
        if baseline is None or not self.per_instance:
            return 0.0
        # tolerance >= the 4-decimal rounding step used in to_json, so comparing a
        # freshly-computed score against a stored (rounded) baseline isn't a false
        # regression; still far below any meaningful score change.
        worse = sum(
            1 for cid, s in self.per_instance.items()
            if cid in baseline.per_instance and s < baseline.per_instance[cid] - 1e-3
        )
        return worse / len(self.per_instance)

    def strictly_better_than(self, other: "Scorecard | None") -> bool:
        """SkillOpt accept rule: pass_rate not down, mean up, and NO instance
        regressed (monotone improvement — the safety property)."""
        if other is None:
            return self.pass_rate > 0 or self.mean_score > 0
        return (
            self.pass_rate >= other.pass_rate
            and self.mean_score > other.mean_score + 1e-9
            and self.regression_rate_vs(other) == 0.0
        )

    def to_json(self) -> dict:
        """Schema-conformant (schemas/scorecard.schema.json): metrics{} + per_case[]."""
        return {
            "skill": self.skill,
            "version": self.version,
            "suite": self.suite,
            "run_at": self.run_at or _now_iso(),
            "runner": "run_evals",
            "metrics": {
                "pass_rate": round(self.pass_rate, 4),
                "n_cases": self.n,
                "mean_score": round(self.mean_score, 4),
            },
            "per_case": [
                {"case_id": cid, "pass": bool(self.passed.get(cid, False)), "score": round(s, 4)}
                for cid, s in self.per_instance.items()
            ],
        }

    @classmethod
    def from_json(cls, d: dict) -> "Scorecard":
        """Parse a scorecard (schema-conformant; tolerates the legacy flat shape)."""
        if "metrics" in d:
            m = d.get("metrics", {}) or {}
            per = {c["case_id"]: c.get("score", 0.0) for c in d.get("per_case", []) or []}
            passed = {c["case_id"]: bool(c.get("pass", False)) for c in d.get("per_case", []) or []}
            return cls(skill=d.get("skill", ""), version=d.get("version", ""),
                       suite=d.get("suite", ""), per_instance=per, passed=passed,
                       pass_rate=m.get("pass_rate", 0.0), mean_score=m.get("mean_score", 0.0),
                       n=m.get("n_cases", len(per)), run_at=d.get("run_at", ""))
        return cls(skill=d.get("skill", ""), version=d.get("version", ""),
                   suite=d.get("suite", ""), per_instance=d.get("per_instance", {}),
                   passed=d.get("passed", {}), pass_rate=d.get("pass_rate", 0.0),
                   mean_score=d.get("mean_score", 0.0), n=d.get("n", 0), run_at=d.get("run_at", ""))


# --- scorers ----------------------------------------------------------------

class Scorer(Protocol):
    def score(self, skill_text: str, case: Case) -> float:  # pragma: no cover - protocol
        ...


def _load_mechanism_aliases() -> dict[str, set[str]]:
    path = HUB / "_registry" / "mechanisms.yaml"
    out: dict[str, set[str]] = {}
    if not path.exists():
        return out
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for m in raw.get("mechanisms", []) or []:
        name = m.get("name")
        if name:
            out[name] = {name, *(m.get("aliases") or [])}
    return out


def _present(term: str, text_l: str) -> bool:
    """Word-boundary match (hyphen-aware) so an avoid_term that is a *substring*
    of an expected_term (e.g. `coalesce` inside `batch-coalesce`) doesn't mis-fire."""
    return re.search(r"(?<![\w-])" + re.escape(term.lower()) + r"(?![\w-])", text_l) is not None


class ProxyScorer:
    """Static keyword/mechanism-coverage proxy (offline, deterministic).

    NOTE (honesty): the optimizer's `propose_edits` synthesizes guidance from the
    SAME `expected_terms`/`expected_mechanism` this scorer reads back. So the
    0.67→1.00 demo validates the SkillOpt *control flow* (propose → gate → accept
    / reject / Pareto / bad_edits), NOT that keyword coverage tracks real
    instruction-count reduction. Swap in a real-machine scorer for capability."""

    def __init__(self) -> None:
        self.aliases = _load_mechanism_aliases()

    def score(self, skill_text: str, case: Case) -> float:
        text = skill_text.lower()
        terms = case.expected_terms or []
        # a rubric-less case must NOT auto-pass: empty expected_terms => recall 0.
        recall = (sum(_present(t, text) for t in terms) / len(terms)) if terms else 0.0
        names = self.aliases.get(case.expected_mechanism, {case.expected_mechanism})
        mech_bonus = 1.0 if any(_present(n, text) for n in names) else 0.0
        penalty = (
            sum(_present(a, text) for a in case.avoid_terms) / len(case.avoid_terms)
            if case.avoid_terms else 0.0
        )
        return max(0.0, min(1.0, 0.6 * recall + 0.4 * mech_bonus - 0.5 * penalty))


# --- loading ----------------------------------------------------------------

def load_suite(suite_dir: str | Path) -> Suite:
    d = Path(suite_dir)
    meta = {}
    if (d / "suite.yaml").exists():
        meta = yaml.safe_load((d / "suite.yaml").read_text(encoding="utf-8")) or {}
    cases: list[Case] = []
    for p in sorted((d / "cases").glob("*.yaml")):
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cases.append(Case(
            id=str(raw["id"]), target=str(raw.get("target", "")),
            expected_mechanism=str(raw.get("expected_mechanism", "")),
            expected_terms=list(raw.get("expected_terms") or []),
            avoid_terms=list(raw.get("avoid_terms") or []),
            subsystem=str(raw.get("subsystem", "")),
            weight=float(raw.get("weight", 1.0)),
        ))
    return Suite(name=str(meta.get("name", d.name)), cases=cases,
                 pass_threshold=float(meta.get("pass_threshold", 0.6)))


def load_skill(skill_dir: str | Path) -> tuple[str, str, str]:
    """Return (name, version, optimizable_text). Text = best_skill.md if present."""
    d = Path(skill_dir)
    fm = {}
    if (d / "SKILL.md").exists():
        m = _FRONTMATTER_RE.match((d / "SKILL.md").read_text(encoding="utf-8"))
        if m:
            fm = yaml.safe_load(m.group(1)) or {}
    artifact = d / "best_skill.md"
    text = artifact.read_text(encoding="utf-8") if artifact.exists() else \
        (d / "SKILL.md").read_text(encoding="utf-8")
    return str(fm.get("name", d.name)), str(fm.get("version", "0.0.0")), text


# --- run --------------------------------------------------------------------

def run_evals(skill_name: str, version: str, skill_text: str, suite: Suite,
              scorer: Scorer | None = None) -> Scorecard:
    scorer = scorer or ProxyScorer()
    card = Scorecard(skill=skill_name, version=version, suite=suite.name, run_at=_now_iso())
    total_w = sum(c.weight for c in suite.cases) or 1.0
    wsum = 0.0
    npass = 0
    for c in suite.cases:
        s = scorer.score(skill_text, c)
        card.per_instance[c.id] = s
        ok = s >= suite.pass_threshold
        card.passed[c.id] = ok
        npass += int(ok)
        wsum += s * c.weight
    card.n = len(suite.cases)
    card.pass_rate = npass / card.n if card.n else 0.0
    card.mean_score = wsum / total_w
    return card


def evaluate_skill_dir(skill_dir: str | Path, suite_dir: str | Path | None = None,
                       scorer: Scorer | None = None) -> Scorecard:
    name, version, text = load_skill(skill_dir)
    if suite_dir is None:
        suite_dir = HUB / "eval" / "task_suites" / "core_optimization_suite"
    return run_evals(name, version, text, load_suite(suite_dir), scorer)


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        sys.stderr.write("usage: run_evals.py <skill_dir> [--suite=<dir>] [--baseline=<card.json>]\n")
        return 2
    suite_dir = None
    baseline = None
    for a in argv:
        if a.startswith("--suite="):
            suite_dir = a.split("=", 1)[1]
        if a.startswith("--baseline="):
            baseline = json.loads(Path(a.split("=", 1)[1]).read_text(encoding="utf-8"))
    card = evaluate_skill_dir(args[0], suite_dir)
    out_dir = Path(args[0]) / "scorecards"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{card.skill}__{card.version}.json"
    out_path.write_text(json.dumps(card.to_json(), indent=2), encoding="utf-8")
    print(f"{card.skill}@{card.version} on {card.suite}: "
          f"pass_rate={card.pass_rate:.2f} mean={card.mean_score:.3f} (n={card.n})")
    for cid, s in card.per_instance.items():
        print(f"  {'PASS' if card.passed[cid] else 'fail'}  {cid:<16} {s:.2f}")
    print(f"scorecard -> {out_path}")
    if baseline is not None:
        # Use from_json so a baseline in the schema shape (metrics{}/per_case[])
        # — which is exactly what to_json writes and every committed scorecard
        # uses — is parsed, not just the legacy flat-key shape. The old direct
        # constructor raised KeyError('per_instance') on the tool's own output.
        bcard = Scorecard.from_json(baseline)
        print(f"regression_rate vs baseline: {card.regression_rate_vs(bcard):.2f}; "
              f"strictly_better={card.strictly_better_than(bcard)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
