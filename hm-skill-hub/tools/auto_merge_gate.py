"""Auto-merge trust gate (plan P4-5, design §11 early-safety).

The closed loop starts **half-automatic**: the optimizer auto-opens a PR but a
human merges. A skill earns auto-merge only after it has proven stable —
`>= N` eval improvements with **zero rollbacks** in its scorecard history.
Until then `decide()` returns `human` (the no-exemption default).

CLI:
    python tools/auto_merge_gate.py [skills_dir] [--min=3]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parent.parent


def _semver(v: str) -> tuple[int, int, int]:
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", str(v))
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)  # type: ignore[return-value]


def is_trusted(pass_rates: list[float], min_improvements: int = 3) -> tuple[bool, str]:
    """`pass_rates` ordered oldest→newest. Trusted iff 0 rollbacks and at least
    `min_improvements` strict improvements (so the skill has a proven upward,
    never-regressing track record)."""
    if len(pass_rates) < min_improvements + 1:
        return False, f"only {len(pass_rates)} scorecard(s); need >= {min_improvements + 1}"
    deltas = [pass_rates[i] - pass_rates[i - 1] for i in range(1, len(pass_rates))]
    rollbacks = sum(1 for d in deltas if d < -1e-9)
    improvements = sum(1 for d in deltas if d > 1e-9)
    if rollbacks:
        return False, f"{rollbacks} rollback(s) in history"
    if improvements < min_improvements:
        return False, f"{improvements} improvement(s) < {min_improvements} required"
    return True, f"{improvements} improvements, 0 rollbacks"


def decide(scorecards: list[dict], min_improvements: int = 3) -> tuple[str, str]:
    """Return ('auto'|'human', reason) for a skill's scorecard history."""
    cards = sorted(scorecards, key=lambda c: _semver(c.get("version", "0.0.0")))
    prs = [float(c.get("pass_rate", 0.0)) for c in cards]
    trusted, reason = is_trusted(prs, min_improvements)
    return ("auto" if trusted else "human"), reason


def _collect(hub_root: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    root = hub_root / "skills"
    if not root.exists():
        return out
    for p in sorted(root.rglob("scorecards/*.json")):
        try:
            card = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.setdefault(str(card.get("skill", p.parent.parent.name)), []).append(card)
    return out


def main(argv: list[str]) -> int:
    root = Path(next((a for a in argv if not a.startswith("--")), HUB / "skills")).resolve()
    hub_root = root.parent if root.name == "skills" else HUB
    min_improvements = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--min=")), "3"))
    by_skill = _collect(hub_root)
    if not by_skill:
        print("no scorecards found.")
        return 0
    for skill in sorted(by_skill):
        mode, reason = decide(by_skill[skill], min_improvements)
        print(f"[{mode:<5}] {skill}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
