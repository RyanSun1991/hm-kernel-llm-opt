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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evals import Scorecard  # type: ignore

HUB = Path(__file__).resolve().parent.parent


def _semver(v: str) -> tuple[int, int, int]:
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", str(v))
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)  # type: ignore[return-value]


def is_trusted(pass_rates: list[float], min_improvements: int = 3) -> tuple[bool, str]:
    """`pass_rates` ordered oldest→newest. Trusted iff there have been at least
    `min_improvements` strict improvements **since the last rollback** (a rollback
    resets the trust counter; the skill can rebuild it). Matches
    `policies/auto_merge_policy.md`."""
    deltas = [pass_rates[i] - pass_rates[i - 1] for i in range(1, len(pass_rates))]
    last_rollback = max((i for i, d in enumerate(deltas) if d < -1e-9), default=-1)
    window = deltas[last_rollback + 1:]
    improvements = sum(1 for d in window if d > 1e-9)
    since = " since last rollback" if last_rollback >= 0 else ""
    if improvements < min_improvements:
        return False, (f"{improvements} consecutive improvement(s){since} "
                       f"< {min_improvements} required")
    return True, f"{improvements} consecutive improvements{since}, 0 rollbacks in window"


def decide(scorecards: list[dict], min_improvements: int = 3) -> tuple[str, str]:
    """Return ('auto'|'human', reason) for a skill's scorecard history. Accepts raw
    scorecard dicts (schema-conformant or legacy) — normalized via from_json."""
    parsed = [Scorecard.from_json(c) for c in scorecards]
    parsed.sort(key=lambda c: _semver(c.version))
    trusted, reason = is_trusted([c.pass_rate for c in parsed], min_improvements)
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
    explicit = next((a for a in argv if not a.startswith("--")), None)
    if explicit:
        p = Path(explicit).resolve()
        if p.name == "skills":
            hub_root = p.parent
        elif (p / "skills").exists():
            hub_root = p
        else:
            sys.stderr.write(f"{p} is neither a hub root nor a skills/ dir\n")
            return 2
    else:
        hub_root = HUB
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
