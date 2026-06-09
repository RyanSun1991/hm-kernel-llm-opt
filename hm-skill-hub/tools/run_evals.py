"""Static-proxy evaluator for Team Skill Hub skills.

Case YAML format:
  id: mm_001
  required_terms: ["evidence", "A/B"]
  forbidden_terms: ["single image"]
  weight: 1.0

This is the Phase-3 proxy gate: it is deterministic and cheap, while later
true-device A/B runners can add metrics into the same scorecard schema.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HUB = Path(__file__).resolve().parent.parent
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _frontmatter(skill: Path) -> dict[str, Any]:
    text = skill.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    return yaml.safe_load(match.group(1)) if match else {}


def _load_cases(suite: Path) -> list[dict[str, Any]]:
    case_dir = suite / "cases" if suite.is_dir() else suite
    return [yaml.safe_load(path.read_text(encoding="utf-8")) for path in sorted(case_dir.glob("*.yaml"))]


def _case_score(skill_text: str, case: dict[str, Any]) -> tuple[bool, float, str]:
    haystack = skill_text.lower()
    required = [str(t).lower() for t in (case.get("required_terms") or [])]
    forbidden = [str(t).lower() for t in (case.get("forbidden_terms") or [])]
    hits = [t for t in required if t in haystack]
    misses = [t for t in required if t not in haystack]
    bad = [t for t in forbidden if t in haystack]
    req_score = len(hits) / len(required) if required else 1.0
    penalty = 1.0 if bad else 0.0
    score = max(0.0, req_score - penalty)
    passed = not misses and not bad
    notes = f"hits={hits}; misses={misses}; forbidden_hits={bad}"
    return passed, score, notes


def run(skill: Path, suite: Path, output_dir: Path, baseline: Path | None = None) -> Path:
    skill_text = skill.read_text(encoding="utf-8")
    fm = _frontmatter(skill)
    cases = _load_cases(suite)
    per_case = []
    weighted_total = 0.0
    weighted_score = 0.0
    n_pass = 0
    for case in cases:
        passed, score, notes = _case_score(skill_text, case)
        weight = float(case.get("weight", 1.0))
        weighted_total += weight
        weighted_score += score * weight
        n_pass += int(passed)
        per_case.append({"case_id": str(case.get("id")), "pass": passed, "score": round(score, 4), "notes": notes})
    pass_rate = (n_pass / len(cases)) if cases else 1.0
    scorecard = {
        "skill": fm.get("name", skill.parent.name),
        "version": str(fm.get("version", "0.0.0")),
        "suite": suite.name,
        "run_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runner": "static-proxy-v1",
        "metrics": {
            "pass_rate": round(pass_rate, 4),
            "n_cases": len(cases),
            "instr_count_delta": round((weighted_score / weighted_total) if weighted_total else 0.0, 4),
            "regression_rate": round(1.0 - pass_rate, 4),
        },
        "per_case": per_case,
    }
    if baseline and baseline.exists():
        base = json.loads(baseline.read_text(encoding="utf-8"))
        scorecard["baseline_version"] = base.get("version", "0.0.0")
        scorecard["improvement_over_baseline"] = pass_rate > float(base.get("metrics", {}).get("pass_rate", -1))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{scorecard['skill']}__{scorecard['version']}.json"
    out.write_text(json.dumps(scorecard, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", type=Path, default=HUB / "skills/core/example/SKILL.md")
    parser.add_argument("--suite", type=Path, default=HUB / "eval/task_suites/core_kernel_optimization")
    parser.add_argument("--output-dir", type=Path, default=HUB / "eval/scorecards")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    print(run(args.skill, args.suite, args.output_dir, args.baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
