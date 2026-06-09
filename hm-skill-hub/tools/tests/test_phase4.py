"""Tests for the Phase 4 auto-optimization loop (design §11, plan P4-1..P4-5).

Runnable via pytest or standalone:
    cd hm-skill-hub && python -m pytest tools/tests/test_phase4.py -q
    cd hm-skill-hub && python tools/tests/test_phase4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
HUB = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import auto_merge_gate as amg  # type: ignore
import broadcast  # type: ignore
import dashboard  # type: ignore
import nightly  # type: ignore
import release  # type: ignore
from release import Inventory  # type: ignore


# ---- release: semver bump --------------------------------------------------

def test_bump_version_arithmetic():
    assert release.bump_version("1.2.3", "patch") == "1.2.4"
    assert release.bump_version("1.2.3", "minor") == "1.3.0"
    assert release.bump_version("1.2.3", "major") == "2.0.0"


def test_infer_bump_major_on_removal_and_schema_change():
    old = Inventory(skills={"core/a": {"version": "1.0.0"}}, knowledge_count=1, schema_hash="h1")
    assert release.infer_bump(old, Inventory(skills={}, schema_hash="h1"))[0] == "major"  # removed
    assert release.infer_bump(old, Inventory(skills=old.skills, schema_hash="h2"))[0] == "major"  # schema


def test_infer_bump_minor_on_add_or_version_raise():
    old = Inventory(skills={"core/a": {"version": "1.0.0", "maturity": "L1"}}, schema_hash="h")
    add = Inventory(skills={**old.skills, "core/b": {"version": "0.1.0"}}, schema_hash="h")
    assert release.infer_bump(old, add)[0] == "minor"
    raised = Inventory(skills={"core/a": {"version": "1.1.0", "maturity": "L1"}}, schema_hash="h")
    assert release.infer_bump(old, raised)[0] == "minor"
    matured = Inventory(skills={"core/a": {"version": "1.0.0", "maturity": "L2"}}, schema_hash="h")
    assert release.infer_bump(old, matured)[0] == "minor"


def test_infer_bump_patch_on_knowledge_only():
    old = Inventory(skills={"core/a": {"version": "1.0.0"}}, knowledge_count=5, schema_hash="h")
    new = Inventory(skills={"core/a": {"version": "1.0.0"}}, knowledge_count=7, schema_hash="h")
    assert release.infer_bump(old, new)[0] == "patch"


def test_infer_bump_minor_on_version_change_even_downgrade():
    # review #7: a version change (incl. a downgrade/revert) is a real change, not
    # "no asset change".
    old = Inventory(skills={"core/a": {"version": "1.1.0", "maturity": "L2"}}, schema_hash="h")
    new = Inventory(skills={"core/a": {"version": "1.0.0", "maturity": "L2"}}, schema_hash="h")
    bump, reasons = release.infer_bump(old, new)
    assert bump == "minor" and any("changed" in r for r in reasons)


def test_scan_inventory_finds_real_skills():
    inv = release.scan_inventory(HUB)
    assert "core/instruction-count-first" in inv.skills
    assert inv.knowledge_count >= 6 and inv.schema_hash


def test_compute_release_bumps_from_registry():
    r = release.compute_release(HUB, bump="auto")
    assert release.bump_version(r["old_version"], r["bump"]) == r["new_version"]


# ---- broadcast -------------------------------------------------------------

def test_broadcast_lock_content_in_repo_and_submodule():
    inrepo = broadcast.lock_content("0.2.0", sha="abc", mode="in-repo")
    assert "hub_version: 0.2.0" in inrepo and "pin: abc" in inrepo and "mode: in-repo" in inrepo
    sub = broadcast.lock_content("0.2.0", sha="deadbeef", mode="submodule", url="git@x")
    assert "semver: 0.2.0" in sub and "sha: deadbeef" in sub and "mode: submodule" in sub


def test_broadcast_write_lock(tmp_path: Path):
    p = tmp_path / "skill-memory.lock"
    broadcast.write_lock(p, "1.0.0", sha="z")
    assert "hub_version: 1.0.0" in p.read_text(encoding="utf-8")


# ---- dashboard -------------------------------------------------------------

def test_dashboard_render_sorts_unsorted_input():
    # feed an UNSORTED list: render() must order rows by semver itself (review #3)
    by_skill = {"s": [
        {"skill": "s", "version": "0.10.0", "pass_rate": 1.0, "mean_score": 1.0, "n": 9},
        {"skill": "s", "version": "0.1.0", "pass_rate": 0.6, "mean_score": 0.7, "n": 9},
        {"skill": "s", "version": "0.2.0", "pass_rate": 0.8, "mean_score": 0.85, "n": 9},
    ]}
    md = dashboard.render(by_skill)
    assert "▲ +0.20" in md          # 0.6 -> 0.8 improvement arrow
    # rows in semver order (not input order, not lexical: 0.2.0 before 0.10.0)
    assert md.index("| 0.1.0 |") < md.index("| 0.2.0 |") < md.index("| 0.10.0 |")
    assert "latest 0.10.0" in md     # latest is the semver-max, not the last input


def test_dashboard_collect_real():
    by_skill = dashboard.collect_scorecards(HUB)
    assert "instruction-count-first" in by_skill


# ---- auto-merge trust gate -------------------------------------------------

def test_is_trusted_requires_history_and_no_rollback():
    assert amg.is_trusted([0.6, 0.7], min_improvements=3)[0] is False        # too short
    assert amg.is_trusted([0.6, 0.7, 0.65, 0.8, 0.9], 3)[0] is False         # a rollback
    assert amg.is_trusted([0.5, 0.6, 0.7, 0.8], 3)[0] is True                # 3 ups, 0 rollback


def test_decide_half_automatic_until_trusted():
    young = [{"version": "0.1.0", "pass_rate": 0.6}]
    assert amg.decide(young)[0] == "human"
    proven = [{"version": f"0.{i}.0", "pass_rate": 0.5 + 0.1 * i} for i in range(4)]
    assert amg.decide(proven)[0] == "auto"


def test_trust_recovers_after_rollback_window():
    # review #6: a rollback resets the counter; rebuilding N improvements regains trust
    assert amg.is_trusted([0.5, 0.6, 0.4, 0.5, 0.6, 0.7], 3)[0] is True   # 3 ups since the dip
    assert amg.is_trusted([0.5, 0.6, 0.7, 0.4, 0.5], 3)[0] is False        # only 1 up since the dip


def test_auto_merge_gate_main_errors_on_non_hub_dir(tmp_path: Path):
    # review #5: a path that is neither a hub root nor a skills/ dir must error,
    # not silently scan the real hub
    assert amg.main([str(tmp_path)]) == 2


# ---- nightly orchestrator --------------------------------------------------

def test_nightly_dry_run_is_safe_and_promotes():
    before = (HUB / "registry.yaml").read_bytes()
    rep = nightly.run_nightly(HUB, apply=False)
    assert rep.normalize_ok and rep.validate_ok
    assert sum(len(o["accepted"]) for o in rep.optimize) >= 1     # optimizer found a win
    # promote computes a strictly higher version
    assert release.bump_version(rep.release["old_version"], rep.release["bump"]) == \
        rep.release["new_version"]
    assert "hub_version:" in rep.lock
    assert rep.auto_merge.get("instruction-count-first") == "human"   # not yet trusted
    assert (HUB / "registry.yaml").read_bytes() == before            # dry-run mutated nothing


def test_nightly_apply_on_temp_writes_artifacts(tmp_path: Path):
    import shutil
    hub = tmp_path / "hub"
    shutil.copytree(HUB, hub, ignore=shutil.ignore_patterns("__pycache__", ".git"))
    rep = nightly.run_nightly(hub, apply=True)
    assert rep.applied and rep.validate_ok
    # registry bumped + release notes + dashboard + improved skill written
    assert (hub / "registry.yaml").read_text(encoding="utf-8").count(rep.release["new_version"]) >= 1
    assert (hub / "eval" / "scorecards" / "_dashboard.md").exists()
    bs = (hub / "skills" / "core" / "instruction-count-first" / "best_skill.md").read_text()
    assert "hoist" in bs.lower()   # optimizer applied the edit on the temp copy


# ---- standalone runner -----------------------------------------------------

def _run_standalone() -> int:
    import inspect
    import tempfile

    g = dict(globals())
    tests = sorted((n, f) for n, f in g.items() if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in tests:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
