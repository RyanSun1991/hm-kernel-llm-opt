import json
from pathlib import Path

from hmopt.opencode.pipeline import (
    build_pipeline_prompt,
    initialize_pipeline_session,
    load_pipeline_profiles,
)


def test_load_pipeline_profiles_reads_expected_profile():
    profiles = load_pipeline_profiles("configs/pipeline_profiles.yaml")
    assert "generic_full" in profiles
    assert "hyperhold_full" in profiles
    assert profiles["hyperhold_full"].entry_agent == "kernel-pipeline-starter"


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
    assert prompt.startswith("@kernel-pipeline-starter")


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
