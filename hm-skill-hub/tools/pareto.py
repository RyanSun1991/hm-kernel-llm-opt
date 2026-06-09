"""GEPA-style Pareto frontier over per-instance scores (design §10.2, plan P3-6).

When N members each propose edits, a single global aggregate collapses
"complementary but mutually-exclusive" edits to one local optimum. The Pareto
frontier keeps every candidate that is best on *some* eval instance (none
dominates it), so complementary lessons survive to be merged — this is the
"everyone sediments, merge without overwriting" property.

A candidate A dominates B iff A >= B on every instance and A > B on at least one.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    keys = set(a) | set(b)
    ge_all = all(a.get(k, 0.0) >= b.get(k, 0.0) - 1e-9 for k in keys)
    gt_any = any(a.get(k, 0.0) > b.get(k, 0.0) + 1e-9 for k in keys)
    return ge_all and gt_any


@dataclass
class Candidate:
    id: str
    scores: dict[str, float]
    meta: dict = field(default_factory=dict)


@dataclass
class ParetoFrontier:
    members: list[Candidate] = field(default_factory=list)

    def add(self, cand_id: str, scores: dict[str, float], meta: dict | None = None) -> bool:
        """Add a candidate; drop any it dominates. Returns True if it was kept."""
        for m in self.members:
            if dominates(m.scores, scores) and m.id != cand_id:
                return False  # dominated by an existing member -> not on the frontier
        # keep only members not dominated by the newcomer
        self.members = [m for m in self.members
                        if not (dominates(scores, m.scores) and m.id != cand_id)]
        if not any(m.id == cand_id for m in self.members):
            self.members.append(Candidate(cand_id, dict(scores), meta or {}))
        return True

    def ids(self) -> list[str]:
        return [m.id for m in self.members]

    def to_json(self) -> list[dict]:
        return [{"id": m.id, "scores": {k: round(v, 4) for k, v in m.scores.items()},
                 "meta": m.meta} for m in self.members]
