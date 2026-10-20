"""Pin Evolution's integration to the existing OpenCode workbench contracts."""

import json
from pathlib import Path

import yaml

from hmopt.evolution.workflow import _dispatch_files

ROOT = Path(__file__).resolve().parents[1]


def frontmatter(path):
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])


def test_dispatch_roles_exist_and_preserve_write_authority(tmp_path):
    packet = {
        "role": "architect",
        "candidate_id": "candidate",
        "state_version": 2,
        "baseline_revision": "a" * 40,
        "allowed_paths": [],
        "source_changes_allowed": False,
        "evidence": {"candidate": {"path": "src/target.c"}},
    }
    for role in ("architect", "implementer", "reviewer", "validator"):
        packet["role"] = role
        packet["source_changes_allowed"] = role == "implementer"
        task = json.loads(_dispatch_files(packet, "dispatch-id", tmp_path)["task.json"])
        for step in task["stage_steps"]:
            definition = frontmatter(ROOT / ".opencode/agents" / f"{step['role']}.md")
            assert definition["name"] == step["role"]
            assert definition["mode"] in {"all", "subagent"}, (
                "task() cannot invoke primary-only roles"
            )
            assert definition["permission"]["skill"]["delegate"] == "deny"
            edit = definition["permission"]["edit"]
            if step["source_changes_allowed"]:
                assert step["role"] == "implementer" and edit == "ask"
            else:
                assert edit["*"] == "deny"
                assert edit[".opencode/local/**"] == "allow"
    coordinator = frontmatter(ROOT / ".opencode/agents/coordinator.md")
    assert coordinator["mode"] in {"all", "primary"}
    assert coordinator["permission"]["task"] == "allow"
    assert coordinator["permission"]["edit"]["*"] == "deny"


def test_evolution_recipe_has_separate_state_contract_and_no_legacy_cast():
    from test_opencode_golden_commands import parse_command

    command = parse_command(ROOT / ".opencode/commands/evolve-candidate.md")
    runtime = frontmatter(ROOT / ".opencode/commands/evolve-candidate.md")
    assert runtime["agent"] == "coordinator"
    assert runtime["subtask"] is False
    names = {Path(path).parent.name for path in command["skill_paths"]}
    assert command["entry_agent"] == "coordinator"
    assert "evolution-execution" in names
    assert (
        not {"recipe-execution", "handoff-contract", "delegate", "instruction-count-first"} & names
    )
    for path in command["skill_paths"]:
        assert (ROOT / path).is_file()
    # The command does not turn an ordinary prompt into a coordinator session.
    config = (ROOT / "opencode.jsonc").read_text(encoding="utf-8")
    assert '"default_agent": "assistant"' in config


def test_evolution_discovery_uses_researcher_without_delegation_or_execution():
    from test_opencode_golden_commands import parse_command

    path = ROOT / ".opencode/commands/evolve-discover.md"
    command = parse_command(path)
    runtime = frontmatter(path)
    assert runtime["agent"] == command["entry_agent"] == "researcher"
    assert runtime["subtask"] is False
    assert command["profile"] is None
    assert {Path(p).parent.name for p in command["skill_paths"]} == {
        "agent-core",
        "language-config",
        "evolution-mining",
    }
    assistant = frontmatter(ROOT / command["agent_file"])
    assert assistant["permission"]["task"] != "allow"
    assert assistant["permission"]["skill"]["delegate"] == "deny"


def test_named_execution_scopes_match_actual_dispatch_stage_actions(tmp_path):
    recipe = (ROOT / ".opencode/skills/infra/pipeline/evolution-execution/SKILL.md").read_text(
        encoding="utf-8"
    )
    rows = {}
    for line in recipe.splitlines():
        if line.startswith("| `"):
            scope, stage, behavior = (part.strip() for part in line.strip("|").split("|"))
            rows[scope.strip("`")] = (stage, behavior)
    assert set(rows) == {
        "status",
        "full",
        "stage",
        "research",
        "plan",
        "implement",
        "review",
        "validate",
    }
    packet = {
        "candidate_id": "candidate",
        "state_version": 2,
        "baseline_revision": "a" * 40,
        "allowed_paths": [],
        "evidence": {"candidate": {"path": "src/target.c"}},
    }
    for scope, stage, role in (
        ("plan", "confirmed", "architect"),
        ("implement", "plan_approved", "implementer"),
        ("review", "implemented", "reviewer"),
        ("validate", "code_approved", "validator"),
    ):
        packet.update(role=role, source_changes_allowed=role == "implementer")
        task = json.loads(_dispatch_files(packet, "dispatch-id", tmp_path)["task.json"])
        assert rows[scope][0] == stage
        if scope != "validate":
            assert task["stage_steps"][-1]["submit_action"] in rows[scope][1]
        else:
            assert task["stage_steps"][-1]["submit_action"] == "validate"
            assert "validation verdict" in rows[scope][1]
    assert rows["research"][0] == "confirmed"
    assert "without a gate submission" in rows["research"][1]
    assert rows["status"][0] == "any"
    assert "without creating a dispatch" in rows["status"][1]


def test_recipe_briefs_bind_fresh_roles_to_target_repository_and_actual_payload_schema():
    recipe = (ROOT / ".opencode/skills/infra/pipeline/evolution-execution/SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "target_repo_path" in recipe and "workbench_root" in recipe
    assert "git -C <target_repo_path>" in recipe
    assert "submission_contract" in recipe
    assert "fresh child context" in recipe
    assert "omit `task_id`" in recipe
    assert "payload={plan, review}" in recipe
    assert "without another `review` wrapper" in recipe
    assert "code\nreview findings requiring source changes stop this recipe" in recipe
    assert "already writes a `validation_result` journal record" in recipe
    assert "publication_status=not_published" in recipe
