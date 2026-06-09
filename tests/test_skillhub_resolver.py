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
