"""Tests for skillhub.resolver (Phase 1 runtime composition, design §12.2)."""
from __future__ import annotations

import json
from pathlib import Path

from hmopt.skillhub.resolver import (
    Resolver,
    _slugify,
    find_hub_root,
    load_selectors,
    load_skills,
    match_subsystems,
    split_target,
)

REPO = Path(__file__).resolve().parent.parent
TARGET = "mm/vmscan.c::shrink_node"


# ---- selectors / skills ----------------------------------------------------

def test_find_hub_root():
    hub = find_hub_root(REPO)
    assert hub is not None and (hub / "knowledge").exists()


def test_split_target():
    assert split_target("mm/vmscan.c::shrink_node") == ("mm/vmscan.c", "shrink_node")
    assert split_target("kernel/workqueue.c") == ("kernel/workqueue.c", None)


def test_slugify_matches_pipeline_convention():
    # must equal hmopt.opencode.pipeline._slugify so target_slug aligns everywhere
    from hmopt.opencode.pipeline import _slugify as pipeline_slugify

    assert _slugify(TARGET) == pipeline_slugify(TARGET) == "mm-vmscan-c-shrink-node"


def test_match_subsystems_by_path_and_symbol():
    sel = load_selectors(find_hub_root(REPO))
    assert "mm-reclaim" in match_subsystems(TARGET, sel)
    # symbol-only match
    assert "mm-reclaim" in match_subsystems("somewhere/else.c::shrink_node", sel)
    # path-only match
    assert "workqueue-threadpool" in match_subsystems("kernel/workqueue.c::foo", sel)


def test_load_skills_has_scaffolds():
    skills = load_skills(find_hub_root(REPO))
    assert "domain/mm-reclaim" in skills
    assert "technique/hoist-loop-invariant" in skills
    assert "core/example" in skills


# ---- resolve end-to-end ----------------------------------------------------

def _resolver() -> Resolver:
    return Resolver(REPO)


def test_resolve_mounts_target_knowledge_and_skill_closure(tmp_path: Path):
    r = _resolver()
    ctx = r.resolve(TARGET, stage="research", run_dir=tmp_path)
    assert ctx.subsystems == ["mm-reclaim"]
    # skill closure: domain pulls its core + technique requires
    refs = {s.ref for s in ctx.skills}
    assert {"domain/mm-reclaim", "technique/hoist-loop-invariant", "core/example"} <= refs
    # target-anchored knowledge mounted
    assert "F001" in [h.record.id for h in ctx.knowledge]
    # retrieval.jsonl written (observability, §12.4)
    log = (tmp_path / "retrieval.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(log[-1])
    assert rec["stage"] == "research" and "F001" in rec["returned_ids"]
    assert rec["token_used"] <= rec["token_budget"]


def test_stage_budget_limits_topk():
    r = _resolver()
    research = r.resolve(TARGET, stage="research")
    test_stage = r.resolve(TARGET, stage="test")
    assert research.token_budget == 3000
    assert test_stage.token_budget == 1000
    assert len(test_stage.knowledge) <= 3


def test_mechanism_anchored_query_added():
    r = _resolver()
    ctx = r.resolve(TARGET, stage="plan-review", mechanism="hoist-invariant")
    assert "mechanism_anchored" in ctx.queries
    assert ctx.queries["mechanism_anchored"] == "hoist-invariant"


def test_local_overlay_adds_non_promoted_and_hub_wins_on_same_id(tmp_path: Path):
    # local in-flight memory: one brand-new local id + one shadow of hub F001
    local = tmp_path / "memory"
    slug = _slugify(TARGET)
    (local / "targets" / slug / "facts").mkdir(parents=True, exist_ok=True)
    (local / "targets" / slug / "facts" / "F500-local.md").write_text(
        "---\nid: F500\ntype: fact\ntitle: local in-flight shrink_node idea\n"
        f"scope: {{level: function, target_slug: {slug}}}\n"
        "source: [{kind: run_id, ref: r_local}]\nmaturity: L1\nstatus: active\n"
        "created_at: 2026-06-01T00:00:00Z\n---\n\nshrink_node local idea\n",
        encoding="utf-8",
    )
    (local / "targets" / slug / "facts" / "F001-shadow.md").write_text(
        "---\nid: F001\ntype: fact\ntitle: LOCAL SHADOW should not win\n"
        f"scope: {{level: function, target_slug: {slug}}}\n"
        "source: [{kind: run_id, ref: r_local}]\nmaturity: L1\nstatus: active\n"
        "created_at: 2026-06-01T00:00:00Z\n---\n\nshrink_node shadow\n",
        encoding="utf-8",
    )
    r = Resolver(REPO, local_memory_root=local)
    ctx = r.resolve(TARGET, stage="research")
    by_id = {h.record.id: h for h in ctx.knowledge}
    assert "F500" in by_id  # local non-promoted idea surfaced
    assert by_id["F001"].record.origin == "hub"  # hub wins on the shared stable id


def _write_skill(hub: Path, kind: str, name: str, *, subsystems=None, requires=None) -> None:
    d = hub / "skills" / kind / name
    d.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f"kind: {kind}", "version: 0.1.0", "maturity: L0",
             'owners: ["@m"]', "status: experimental"]
    if subsystems is not None:
        lines += ["applies_to:", f"  subsystems: [{', '.join(subsystems)}]"]
    if requires is not None:
        lines.append(f"requires: [{', '.join(requires)}]")
    lines += ["---", "", f"# {name}", "", "## When to use", "x", "## How to use", "y", ""]
    (d / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")


def test_requires_cycle_terminates(tmp_path: Path):
    # a requires-cycle (core/a <-> core/b) must not hang the closure resolver
    hub = tmp_path / "hm-skill-hub"
    (hub / "knowledge").mkdir(parents=True)
    (hub / "_registry").mkdir(parents=True)
    (hub / "_registry" / "subsystem_selectors.yaml").write_text(
        "subsystems:\n  - name: s\n    path_globs: ['foo.c']\n    symbol_selectors: []\n",
        encoding="utf-8")
    _write_skill(hub, "domain", "d", subsystems=["s"], requires=["core/a"])
    _write_skill(hub, "core", "a", requires=["core/b"])
    _write_skill(hub, "core", "b", requires=["core/a"])  # cycle
    r = Resolver(tmp_path, hub_root=hub)
    ctx = r.resolve("foo.c", stage="research")  # must return, not hang
    refs = {s.ref for s in ctx.skills}
    assert {"domain/d", "core/a", "core/b"} <= refs


def test_trim_keeps_strongest_when_single_record_oversized():
    from hmopt.skillhub.records import Record
    from hmopt.skillhub.retrieval import RetrievalHit

    r = Resolver(REPO)
    big = Record(id="BIG", kind="memory_item", path="x",
                 fields={"title": "big", "maturity": "L2", "score": 1.0}, body="word " * 5000)
    kept, used = r._trim_to_budget([RetrievalHit(record=big, score=1.0)], top_k=3, token_cap=10)
    assert len(kept) == 1 and kept[0].record.id == "BIG"  # never returns empty


def test_find_hub_root_none_and_resolver_errors_clearly(tmp_path: Path):
    import pytest

    assert find_hub_root(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        Resolver(tmp_path)


def test_token_trim_drops_weakest(tmp_path: Path):
    # build a local memory with many large records; ensure trim keeps under cap
    local = tmp_path / "memory"
    slug = _slugify(TARGET)
    d = local / "targets" / slug / "facts"
    d.mkdir(parents=True, exist_ok=True)
    big = "shrink_node reclaim priority hoist " * 200
    for i in range(8):
        (d / f"F60{i}-big.md").write_text(
            f"---\nid: F60{i}\ntype: fact\ntitle: big shrink_node note {i}\n"
            f"scope: {{level: function, target_slug: {slug}}}\n"
            "source: [{kind: doc, ref: x}]\nmaturity: L2\nstatus: active\n"
            f"created_at: 2026-06-01T00:00:00Z\n---\n\n{big}\n",
            encoding="utf-8",
        )
    r = Resolver(REPO, local_memory_root=local)
    ctx = r.resolve(TARGET, stage="implement")  # cap 1500
    assert ctx.token_used <= ctx.token_budget
