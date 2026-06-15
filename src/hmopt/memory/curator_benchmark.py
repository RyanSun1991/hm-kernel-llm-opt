"""Classification benchmark for the local curator (plan P1-8).

>=6 cases per route across the seven §10.1.0 relations (42 local + subsumption
in central mode), each carrying the discriminating structured signals. The
runner reports seven-way accuracy, per-route accuracy, the temporal+conditional
**false-delete rate** (the PoC veto metric), and latency.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .local_curator import (
    CuratorItem,
    RelationKind,
    apply_relation,
    classify_relation,
)


@dataclass
class Case:
    incoming: CuratorItem
    neighbors: list[CuratorItem]
    expected: RelationKind
    mode: str = "local"


def _it(idx: str, **kw) -> CuratorItem:
    return CuratorItem(id=idx, **kw)


# (slug, mechanism, base_text) seeds for variety across routes
_SEEDS = [
    ("mm-vmscan-c-shrink-node", "hoist-invariant", "hoist sc->priority out of the reclaim loop"),
    ("kernel-workqueue-c-process-one-work", "batch-coalesce", "coalesce small work writes"),
    ("kernel-sched-c-pick-next", "branch-elimination", "predicate the pick-next hot branch"),
    ("drivers-hyperhold-io-c", "redundant-load-elimination", "drop repeated header reload"),
    ("kernel-locking-mutex-c", "lock-split", "split the coarse mutex per shard"),
    ("mm-page-alloc-c", "prefetch", "prefetch the next free-list entry"),
]


def build_cases() -> list[Case]:
    cases: list[Case] = []

    for i, (slug, mech, text) in enumerate(_SEEDS):
        sym = slug.rsplit("-", 1)[-1]

        # duplicate: same scope/mechanism/conclusion
        cases.append(Case(
            _it("I", target_slug=slug, mechanism=mech, text=text, delta_pct=-0.5,
                applies_when="hot loop", compare_level="function"),
            [_it("N", target_slug=slug, mechanism=mech, text=text + " removes loads",
                 delta_pct=-0.6, applies_when="hot loop", compare_level="function")],
            "duplicate"))

        # contradiction: same condition, opposite verdict, no version bump
        cases.append(Case(
            _it("I", target_slug=slug, mechanism=mech, text=text, delta_pct=+0.7,
                applies_when="hot loop", maturity="L2", confirmations=3),
            [_it("N", target_slug=slug, mechanism=mech, text=text, delta_pct=-0.6,
                 applies_when="hot loop", maturity="L1", confirmations=1)],
            "contradiction"))

        # temporal: opposite verdict explained by a newer kernel version
        cases.append(Case(
            _it("I", target_slug=slug, mechanism=mech, text=text, delta_pct=+0.3,
                applies_when="hot loop", kernel_version=6.6),
            [_it("N", target_slug=slug, mechanism=mech, text=text, delta_pct=-0.5,
                 applies_when="hot loop", kernel_version=6.1)],
            "temporal"))

        # conditional: both hold, different applies_when (no version bump)
        cases.append(Case(
            _it("I", target_slug=slug, mechanism=mech, text=text, delta_pct=-0.4,
                applies_when="loop trip count > 100"),
            [_it("N", target_slug=slug, mechanism=mech, text=text, delta_pct=+0.1,
                 applies_when="loop trip count < 10")],
            "conditional"))

        # selector drift: symbol stable, path/offset moved by a rebase
        cases.append(Case(
            _it("I", target_slug=slug, mechanism=mech, text=text, symbol=sym,
                invalidation="rebase moved the offset; re-resolve selector"),
            [_it("N", target_slug=slug, mechanism=mech, text=text, symbol=sym)],
            "selector"))

        # evidence divergence: same delta, different compare_level
        cases.append(Case(
            _it("I", target_slug=slug, mechanism=mech, text=text, delta_pct=-0.8,
                compare_level="function"),
            [_it("N", target_slug=slug, mechanism=mech, text=text, delta_pct=-0.8,
                 compare_level="process")],
            "evidence"))

        # novel: different scope AND different mechanism
        other = _SEEDS[(i + 3) % len(_SEEDS)]
        cases.append(Case(
            _it("I", target_slug=slug, mechanism=mech, text=text),
            [_it("N", target_slug=other[0], mechanism=other[1], text=other[2])],
            "novel"))

        # subsumption (central only): specific instance vs general pattern
        cases.append(Case(
            _it("I", target_slug=slug, mechanism=mech, text=text, granularity="specific"),
            [_it("N", subsystem=slug.split("-")[0], mechanism=mech,
                 text="generalize: " + text, granularity="general")],
            "subsumption", mode="central"))

    return cases


@dataclass
class BenchMetrics:
    total: int
    correct: int
    accuracy: float
    per_route: dict[str, tuple[int, int]] = field(default_factory=dict)
    false_delete_total: int = 0
    false_delete_count: int = 0
    false_delete_rate: float = 0.0
    avg_latency_s: float = 0.0
    local_subsumption_emitted: int = 0


def _false_delete(case: Case, rel) -> bool:
    """A temporal/conditional case is a false-delete if the record that must stay
    active is superseded, or anything is physically deleted (never should be)."""
    nb = case.neighbors[0] if case.neighbors else None
    ar = apply_relation(rel, case.incoming, nb)
    if ar.deleted:
        return True
    if case.expected == "conditional":
        return bool(ar.superseded)   # a still-valid conditional fact must not be superseded
    if case.expected == "temporal":
        return not ar.added          # the current fact must be retained
    return False


def run_benchmark(cases: list[Case] | None = None, *, llm=None) -> BenchMetrics:
    cases = cases or build_cases()
    correct = 0
    per: dict[str, list[int]] = {}
    fd_total = fd_count = 0
    local_subsum = 0
    t0 = time.perf_counter()

    for c in cases:
        rel = classify_relation(c.incoming, c.neighbors, mode=c.mode, llm=llm)
        ok = rel.kind == c.expected
        correct += int(ok)
        per.setdefault(c.expected, [0, 0])
        per[c.expected][0] += int(ok)
        per[c.expected][1] += 1
        if c.mode == "local" and rel.kind == "subsumption":
            local_subsum += 1
        if c.expected in ("temporal", "conditional"):
            fd_total += 1
            fd_count += int(_false_delete(c, rel))

    elapsed = time.perf_counter() - t0
    return BenchMetrics(
        total=len(cases),
        correct=correct,
        accuracy=correct / len(cases) if cases else 0.0,
        per_route={k: (v[0], v[1]) for k, v in per.items()},
        false_delete_total=fd_total,
        false_delete_count=fd_count,
        false_delete_rate=(fd_count / fd_total) if fd_total else 0.0,
        avg_latency_s=elapsed / len(cases) if cases else 0.0,
        local_subsumption_emitted=local_subsum,
    )
