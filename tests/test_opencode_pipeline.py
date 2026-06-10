import json
from pathlib import Path

from hmopt.opencode.pipeline import (
    build_pipeline_prompt,
    initialize_pipeline_session,
    load_pipeline_profiles,
    validate_profile_assets,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_all_profiles_reference_existing_assets():
    # Every entry agent / skill / card / doc a profile names must exist on disk.
    # This pins the os-opt-manager -> hm-opt-manager and flat-skill-path fixes so
    # the staged prompt can never silently point at nothing again.
    profiles = load_pipeline_profiles("configs/pipeline_profiles.yaml")
    problems = {
        name: validate_profile_assets(prof, REPO_ROOT)
        for name, prof in profiles.items()
    }
    problems = {name: missing for name, missing in problems.items() if missing}
    assert not problems, f"profiles reference missing assets: {problems}"


def test_load_pipeline_profiles_reads_expected_profile():
    profiles = load_pipeline_profiles("configs/pipeline_profiles.yaml")
    assert "generic_full" in profiles
    assert "hyperhold_full" in profiles
    assert profiles["hyperhold_full"].entry_agent == "hm-opt-manager"
    assert profiles["generic_full"].primary_goal == "instruction_count"
    assert profiles["generic_full"].plan_reviewer_agent == "kernel-plan-reviewer"
    assert profiles["generic_full"].code_reviewer_agent == "kernel-code-reviewer"
    assert profiles["generic_full"].tester_agent == "kernel-tester-agent"


def test_initialize_pipeline_session_writes_state_and_prompt(tmp_path: Path):
    repo_root = tmp_path
    profiles = load_pipeline_profiles("configs/pipeline_profiles.yaml")
    session = initialize_pipeline_session(
        repo_root=repo_root,
        profile=profiles["sync_review"],
        target="hp_iotab.c",
    )
    state = json.loads(Path(session["state_path"]).read_text(encoding="utf-8"))
    prompt = Path(session["prompt_path"]).read_text(encoding="utf-8")
    assert state["profile"] == "sync_review"
    assert state["target"] == "hp_iotab.c"
    assert state["plan_reviewer_agent"] == "kernel-plan-reviewer"
    assert state["code_reviewer_agent"] == "kernel-code-reviewer"
    assert state["tester_agent"] == "kernel-tester-agent"
    assert prompt.startswith("@hm-opt-manager")
    assert state["primary_metric"] == "instruction_count"
    assert state["plan_review_status"] == ""
    assert state["code_review_status"] == ""
    assert state["test_status"] == ""
    assert "Primary metric: instruction_count" in prompt


def test_initialize_pipeline_session_adds_memory_files_for_generic_profile(tmp_path: Path):
    repo_root = tmp_path
    profiles = load_pipeline_profiles("configs/pipeline_profiles.yaml")
    session = initialize_pipeline_session(
        repo_root=repo_root,
        profile=profiles["generic_full"],
        target="sysmgr/memmgr/mem/swap/hyperhold/hp_iotab.c",
    )
    state = json.loads(Path(session["state_path"]).read_text(encoding="utf-8"))
    assert ".opencode/memory/global_lessons.md" in state["memory_files"]
    assert any(item.endswith("hp-iotab-c.md") for item in state["memory_files"])


def test_build_pipeline_prompt_includes_artifacts():
    profiles = load_pipeline_profiles("configs/pipeline_profiles.yaml")
    prompt = build_pipeline_prompt(
        profiles["hyperhold_full"],
        target="sysmgr/memmgr/mem/swap/hyperhold/hp_iotab.c",
        objective="Optimize hp_iotab",
        artifact_specs=["flamegraph:outputs/flamegraph.json"],
    )
    assert "Artifacts:" in prompt
    assert "flamegraph: outputs/flamegraph.json" in prompt
    assert "Plan reviewer: kernel-plan-reviewer" in prompt
    assert "Code reviewer: kernel-code-reviewer" in prompt
    assert "Tester: kernel-tester-agent" in prompt
