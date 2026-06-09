"""eval-gate: reject a skill change that does not strictly hold the line
(design §9 gate 3 / §13 "抗回归", plan P3-4).

For every skill with a committed scorecard, re-evaluate its current
`best_skill.md` against the skill's task suite and compare to the committed
scorecard (the last *accepted* baseline). A change is **rejected** if
`pass_rate` drops or any instance regresses — "metrics.pass_rate 不增即拒".

Convention: the committed `scorecards/<name>__<version>.json` is updated only by
the gated optimizer (or a reviewed PR), so it represents the accepted baseline; a
hand edit to `best_skill.md` that regresses is caught here because it re-evaluates
worse than the committed card.

THREAT MODEL / LIMITATION (be honest): the baseline is the *in-tree, mutable*
scorecard. A PR that regresses `best_skill.md` **and** co-rewrites the scorecard
to the worse numbers will pass this gate (baseline == current). So this gate is
NOT self-sufficient — it assumes the scorecard is not hand-edited and relies on a
reviewer inspecting the scorecard diff. A robust implementation pins the baseline
to a non-co-editable source (the card at the PR merge-base / previous git
revision, or a card stored under `evidence/`); that is a Phase-4 follow-up.

CLI:
    python tools/eval_gate.py [skills_dir]        # default: skills/
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evals import Scorecard, evaluate_skill_dir, load_skill  # type: ignore

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("PyYAML required\n")
    sys.exit(2)

HUB = Path(__file__).resolve().parent.parent
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _suite_dir_for(skill_dir: Path) -> Path | None:
    sk = skill_dir / "SKILL.md"
    if not sk.exists():
        return None
    m = _FRONTMATTER_RE.match(sk.read_text(encoding="utf-8"))
    if not m:
        return None
    fm = yaml.safe_load(m.group(1)) or {}
    eval_id = fm.get("eval_id")
    if not eval_id:
        return None
    d = HUB / eval_id
    return d if (d / "cases").exists() else None


def _semver_key(path: Path) -> tuple[int, int, int]:
    m = re.search(r"__(\d+)\.(\d+)\.(\d+)\.json$", path.name)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)  # type: ignore[return-value]


def _committed_card(skill_dir: Path) -> Scorecard | None:
    sc_dir = skill_dir / "scorecards"
    cards = list(sc_dir.glob("*.json")) if sc_dir.exists() else []
    if not cards:
        return None
    # prefer the card matching the current best_skill version; else the highest
    # semver (parsed, not lexical — 0.10.0 > 0.9.0).
    try:
        _, version, _ = load_skill(skill_dir)
    except Exception:  # noqa: BLE001
        version = ""
    chosen = next((c for c in cards if c.name.endswith(f"__{version}.json")), None)
    if chosen is None:
        chosen = max(cards, key=_semver_key)
    raw = json.loads(chosen.read_text(encoding="utf-8"))
    return Scorecard(skill=raw["skill"], version=raw["version"], suite=raw["suite"],
                     per_instance=raw.get("per_instance", {}),
                     pass_rate=raw.get("pass_rate", 0.0), mean_score=raw.get("mean_score", 0.0))


def check_skill(skill_dir: Path) -> tuple[bool, str]:
    """Return (ok, message). ok=False means a regression was detected."""
    suite_dir = _suite_dir_for(skill_dir)
    baseline = _committed_card(skill_dir)
    if suite_dir is None or baseline is None:
        return True, f"{skill_dir.name}: no suite/scorecard — skipped"
    current = evaluate_skill_dir(skill_dir, suite_dir)
    reg = current.regression_rate_vs(baseline)
    if current.pass_rate < baseline.pass_rate - 1e-3 or reg > 0.0:
        return False, (f"{skill_dir.name}: REGRESSION — pass_rate "
                       f"{baseline.pass_rate:.2f}→{current.pass_rate:.2f}, "
                       f"instance regression_rate={reg:.2f}")
    return True, (f"{skill_dir.name}: ok — pass_rate {current.pass_rate:.2f} "
                  f"(baseline {baseline.pass_rate:.2f}), no regression")


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else HUB / "skills"
    failures = 0
    checked = 0
    for sk in sorted(root.rglob("SKILL.md")):
        skill_dir = sk.parent
        if not (skill_dir / "best_skill.md").exists():
            continue
        checked += 1
        ok, msg = check_skill(skill_dir)
        print(("OK   " if ok else "FAIL ") + msg)
        if not ok:
            failures += 1
    if failures:
        sys.stderr.write(f"\neval-gate: {failures} skill(s) regressed — change rejected.\n")
        return 1
    print(f"\neval-gate: {checked} skill(s) checked, no regression.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
