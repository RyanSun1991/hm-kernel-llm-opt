"""Tests for the .opencode/memory -> hub sediment bridge + skill promotion.

Members work through the .opencode/ harness, not `hmopt run`, so their durable
findings accumulate as markdown under .opencode/memory/. This bridge parses those
stores into schema-valid candidates that flow through the existing sediment
downstream (validate -> staging -> hub).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from hmopt.sediment.opencode_reader import read_opencode_memory
from hmopt.sediment.pipeline import sediment_opencode

REPO = Path(__file__).resolve().parent.parent


def _write_memory(root: Path) -> Path:
    mem = root / ".opencode" / "memory"
    (mem / "idea_ledger").mkdir(parents=True)
    (mem / "targets").mkdir()
    (mem / "subsystems").mkdir()
    (mem / "idea_ledger" / "mm-vmscan-c-shrink-node.md").write_text(
        "# Idea Ledger — shrink_node\n"
        "Target slug: `mm-vmscan-c-shrink-node`\n\n"
        "## Landed (pipeline run later confirmed a win)\n"
        "### L001 hoist invariant sc->priority read out of the per-page loop\n"
        "- **scope**: mm/vmscan.c :: shrink_node\n"
        "- **status**: landed\n"
        "- **verdicted_by**: hm-opt-manager\n"
        "- **landed_on**: 2026-04-24 19:02 UTC\n"
        "- **delta_pct**: -0.8\n"
        "- **compare_level**: function\n"
        "- **rationale**: clean A/B, no regression\n\n"
        "## Rejected\n"
        "### L002 inline hot kworker thread entry\n"
        "- **scope**: kernel/workqueue.c :: process_one_work\n"
        "- **status**: rejected\n"
        "- **rationale**: blanket inline regresses i-cache\n\n"
        "<!-- ### L999 TEMPLATE EXAMPLE must be ignored\n- **status**: landed -->\n",
        encoding="utf-8",
    )
    (mem / "global_lessons.md").write_text(
        "# Global Lessons\n\n"
        "### Hoist loop-invariant reads before inlining\n"
        "- **kind**: heuristic\n"
        "- **applies_when**: hot loop re-reads an invariant value\n"
        "- **do_or_dont**: do: hoist then re-measure\n\n"
        "### Single-image A/B is not a landing signal\n"
        "- **kind**: validation_pitfall\n"
        "- **do_or_dont**: don't: land on a single-image delta\n",
        encoding="utf-8",
    )
    (mem / "targets" / "mm-vmscan-c-shrink-node.md").write_text(
        "# Target Memory\n## Target\nmm/vmscan.c :: shrink_node\n"
        "## Stable Structural Facts\n"
        "- sc->priority is invariant for the duration of a single shrink_node scan\n"
        "## Known Bad Plans\n"
        "- blanket-inlining shrink_page_list regresses i-cache on A78\n",
        encoding="utf-8",
    )
    return root / ".opencode"


def test_reader_parses_all_families(tmp_path: Path):
    oc = _write_memory(tmp_path)
    mem = read_opencode_memory(oc, contributor="alice")
    kinds = Counter(c["schema"] for c in mem.candidates)
    assert kinds["idea"] == 2          # landed + rejected
    assert kinds["global_lesson"] == 2  # heuristic + validation_pitfall
    assert kinds["memory_item"] == 1    # the structural fact
    assert kinds["bad_plan"] == 1       # the known bad plan


def test_template_comment_rows_are_ignored(tmp_path: Path):
    oc = _write_memory(tmp_path)
    mem = read_opencode_memory(oc, contributor="alice")
    ids = {c["record"]["id"] for c in mem.candidates}
    # the L999 example lives inside an HTML comment -> must not sediment
    related = {r for c in mem.candidates for r in c["record"].get("related_ids", [])}
    assert "L999" not in related and "L999" not in ids


def test_candidates_validate_against_hub_schemas(tmp_path: Path):
    oc = _write_memory(tmp_path)
    res = sediment_opencode(oc, out_dir=tmp_path / "staging", hub_root=REPO, contributor="alice")
    # every parsed candidate is schema-valid -> none rejected
    assert res.invalid == [], res.invalid
    assert res.n_valid == 6
    # ledger ids are renumbered to the provisional band but keep provenance
    ideas = [c for c in res.candidates if c["schema"] == "idea"]
    assert all(c["record"]["id"].startswith("L9") for c in ideas)
    assert {"L001", "L002"} == {r for c in ideas for r in c["record"]["related_ids"]}


def test_global_lesson_kind_matches_id_prefix(tmp_path: Path):
    oc = _write_memory(tmp_path)
    res = sediment_opencode(oc, out_dir=tmp_path / "staging", hub_root=REPO, contributor="alice")
    lessons = {c["record"]["id"][:1]: c["record"]["kind"]
               for c in res.candidates if c["schema"] == "global_lesson"}
    assert lessons.get("H") == "heuristic"
    assert lessons.get("V") == "validation_pitfall"


def test_missing_memory_is_empty_not_error(tmp_path: Path):
    (tmp_path / ".opencode").mkdir()
    res = sediment_opencode(tmp_path / ".opencode", out_dir=tmp_path / "s", hub_root=REPO)
    assert res.n_valid == 0 and not res.invalid


def test_promote_skill_scaffolds_valid_hub_skill(tmp_path: Path):
    from hmopt.sediment.skill_promote import promote_opencode_skill

    # minimal member skill + minimal hub layout
    sk = tmp_path / ".opencode" / "skills" / "optimization-funnel"
    sk.mkdir(parents=True)
    (sk / "SKILL.md").write_text(
        "---\nname: optimization-funnel\ndescription: rank ideas by ROI\n---\n\n"
        "# Optimization Funnel\n\nDo the funnel.\n", encoding="utf-8")
    hub = tmp_path / "hub"
    (hub / "schemas").mkdir(parents=True)
    (hub / "skills").mkdir()

    res = promote_opencode_skill(sk, kind="core", hub_root=hub)
    dest = Path(res["skill_md"])
    assert dest == hub / "skills" / "core" / "optimization-funnel" / "SKILL.md"
    text = dest.read_text(encoding="utf-8")
    assert "kind: core" in text and "maturity: L0" in text
    assert "description:" not in text.split("---")[1]  # dropped (schema forbids)
    assert res["next_steps"]


def _write_run_artifacts(oc: Path) -> None:
    """Tier-0 artifacts in the harness's documented formats (.opencode/CLAUDE.md)."""
    (oc / "reviews").mkdir(parents=True, exist_ok=True)
    (oc / "bench").mkdir(exist_ok=True)
    (oc / "state").mkdir(exist_ok=True)
    (oc / "plans").mkdir(exist_ok=True)
    (oc / "reviews" / "wq-inline_plan_review.md").write_text(
        "# Review\n\n## Target Artifact\nwq-inline\n\n## Review Type\nplan\n\n"
        "## Decision\n\nreject\n\n## Findings\n\n- blanket inline regresses i-cache on A78\n\n"
        "## Risk Summary\n\nhigh icache pressure\n", encoding="utf-8")
    (oc / "reviews" / "ok-thing_code_review.md").write_text(
        "# Review\n\n## Decision\n\napprove\n\n## Findings\n\n- fine\n", encoding="utf-8")
    (oc / "reviews" / "review_template.md").write_text(
        "# Review Template\n\n## Decision\n\nreject\n", encoding="utf-8")  # must be skipped
    (oc / "bench" / "hoist-priority_validation.md").write_text(
        "# Validation\n\n## Step 5\naggregate delta_pct: -0.8\ncompare_level: function\n\n"
        "## Step 6 (Decision)\nverdict: pass\n", encoding="utf-8")
    (oc / "bench" / "failed-idea_validation.md").write_text(
        "# Validation\n\n## Step 6\nverdict: fail\ndelta_pct: 0.5\n", encoding="utf-8")
    (oc / "state" / "bad_plans.md").write_text(
        "# Global Bad Plans\n\n- moving wq lockdep checks out of the loop breaks debug builds\n",
        encoding="utf-8")
    (oc / "plans" / "wq_plan.md").write_text("# Plan\nFree-form design text.\n", encoding="utf-8")
    (oc / "plans" / "optimization_plan_template.md").write_text("# T\n", encoding="utf-8")


def test_run_artifacts_sediment_reviews_bench_state(tmp_path: Path):
    oc = _write_memory(tmp_path)  # memory/ present -> run-artifact scan active
    _write_run_artifacts(oc)
    res = sediment_opencode(oc, out_dir=tmp_path / "staging", hub_root=REPO, contributor="alice")
    assert res.invalid == [], res.invalid
    by = {}
    for c in res.candidates:
        by.setdefault(c["schema"], []).append(c["record"])
    # review reject -> anti_pattern + bad_plan; template + approve skipped
    rejects = [r for r in by["global_lesson"] if r.get("kind") == "anti_pattern"]
    assert any("wq-inline" in r["lesson"] for r in rejects)
    assert not any("ok-thing" in r["lesson"] for r in rejects)
    # bench pass -> fact; fail bench yields nothing
    facts = [r for r in by["memory_item"] if "hoist-priority" in r["title"]]
    assert len(facts) == 1 and facts[0]["evidence"]["delta_pct"] == -0.8
    assert not any("failed-idea" in r["title"] for r in by["memory_item"])
    # state bad_plans bullet -> global bad_plan with target_pattern "*"
    state_bps = [r for r in by["bad_plan"] if "lockdep" in r["reason"]]
    assert len(state_bps) == 1 and state_bps[0]["target_pattern"] == "*"
    # ids are unique across the whole batch (per-file 901s renumbered)
    ids = [c["record"]["id"] for c in res.candidates]
    assert len(ids) == len(set(ids)), ids


def test_free_text_collected_and_salience_pass_runs(tmp_path: Path):
    oc = _write_memory(tmp_path)
    _write_run_artifacts(oc)
    from hmopt.sediment.opencode_reader import read_opencode_memory

    mem = read_opencode_memory(oc, contributor="alice")
    assert len(mem.free_text) == 1  # wq_plan.md only; template skipped

    class _LLM:  # offline fake: one reusable heuristic
        def chat(self, messages):
            return '[{"lesson":"batch counters outside hot loops",' \
                   '"applies_when":"per-item shared writes","do_or_dont":"do: batch"}]'

    res = sediment_opencode(oc, out_dir=tmp_path / "s2", hub_root=REPO,
                            contributor="alice", llm=_LLM())
    lessons = [c["record"]["lesson"] for c in res.candidates
               if c["schema"] == "global_lesson"]
    assert any("batch counters" in les for les in lessons)
