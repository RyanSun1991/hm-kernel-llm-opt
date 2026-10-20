"""Golden regression for the OpenCode pipeline lane (design: Agent_Workbench §13-14).

The golden file was first generated from the pre-M1 tree and then deliberately
re-frozen at each migration phase that intended a contract change (see the golden
file's `_meta.frozen_from`); at every re-freeze the drift was verified to be exactly
the intended commands before regenerating. Going forward it pins the pipeline
contract at the level that must NOT drift:

- WHICH skills each command / pipeline profile loads (by skill name — the
  directory leaf — deliberately independent of where the skill lives, so the
  3-tier reorg passes while a dropped or added skill fails);
- WHICH agents each command / profile names;
- gates and objectives (validation_mode, primary_goal, research_first);
- that every referenced path (agent file, skill pack, memory pack, bootstrap
  doc, pipeline card) exists on disk — a staged prompt must never silently
  point at nothing.

When a later phase intends a contract change (e.g. M4 rewires /optimize_* to
the coordinator), update tests/golden/opencode_commands_golden.yaml in the same
change, deliberately.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from hmopt.opencode.pipeline import load_pipeline_profiles, validate_profile_assets

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "opencode_commands_golden.yaml"

SECTION_RE = re.compile(r"^(Skill packs|Memory packs|Bootstrap docs|Config):\s*$")
FIRST_LINE_RE = re.compile(r"^@([\w-]+)\s+@(\S+)")
PROFILE_RE = re.compile(r"^Profile:\s*(\S+)(?:\s+@(\S+))?")
AUTO_ITER_RE = re.compile(r"^Auto-Iterate:")
# A concrete .opencode path (no placeholders/wildcards), possibly @-prefixed.
PATH_TOKEN_RE = re.compile(r"@?(\.opencode/[\w\-./]+\.(?:md|yaml|json))")


def _skill_name(path: str) -> str:
    return Path(path).parent.name


def parse_command(path: Path) -> dict:
    entry_agent = agent_file = profile = pipeline_card = None
    auto_iterate = False
    sections: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if (m := FIRST_LINE_RE.match(line)) and entry_agent is None:
            entry_agent, agent_file = m.group(1), m.group(2)
            continue
        if m := PROFILE_RE.match(line):
            profile, pipeline_card = m.group(1), m.group(2)
            continue
        if AUTO_ITER_RE.match(line):
            auto_iterate = True
            continue
        if m := SECTION_RE.match(line):
            current = m.group(1)
            sections[current] = []
            continue
        if current and line.strip().startswith("- @"):
            sections[current].append(line.strip()[3:])
        elif current and line.strip() and not line.strip().startswith("-"):
            current = None
    return {
        "entry_agent": entry_agent,
        "agent_file": agent_file,
        "auto_iterate": auto_iterate,
        "profile": profile,
        "pipeline_card": pipeline_card,
        "skill_paths": sections.get("Skill packs", []),
        "memory_packs": sections.get("Memory packs", []),
        "bootstrap_docs": sections.get("Bootstrap docs", []),
        "config": sections.get("Config", []),
    }


def _golden() -> dict:
    return yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_commands_match_golden_contract():
    golden = _golden()["commands"]
    on_disk = {
        p.stem for p in (REPO_ROOT / ".opencode" / "commands").glob("*.md") if p.stem != "README"
    }
    assert on_disk == set(golden), (
        f"command set drifted: extra={on_disk - set(golden)}, missing={set(golden) - on_disk}"
    )
    problems: dict[str, list[str]] = {}
    for name, want in golden.items():
        got = parse_command(REPO_ROOT / ".opencode" / "commands" / f"{name}.md")
        errs: list[str] = []
        if got["entry_agent"] != want["entry_agent"]:
            errs.append(f"entry_agent {got['entry_agent']!r} != {want['entry_agent']!r}")
        if got["profile"] != want.get("profile"):
            errs.append(f"profile {got['profile']!r} != {want.get('profile')!r}")
        if got["auto_iterate"] != want["auto_iterate"]:
            errs.append(f"auto_iterate {got['auto_iterate']} != {want['auto_iterate']}")
        got_skills = sorted(_skill_name(p) for p in got["skill_paths"])
        if got_skills != want["skills"]:
            errs.append(f"skill set {got_skills} != {want['skills']}")
        if sorted(got["memory_packs"]) != want["memory_packs"]:
            errs.append(f"memory packs {got['memory_packs']} != {want['memory_packs']}")
        if sorted(got["bootstrap_docs"]) != want["bootstrap_docs"]:
            errs.append(f"bootstrap docs {got['bootstrap_docs']} != {want['bootstrap_docs']}")
        if errs:
            problems[name] = errs
    assert not problems, f"commands drifted from golden contract: {problems}"


def test_command_references_exist_on_disk():
    missing: dict[str, list[str]] = {}
    for cmd in (REPO_ROOT / ".opencode" / "commands").glob("*.md"):
        if cmd.stem == "README":
            continue
        got = parse_command(cmd)
        refs = [got["agent_file"], got["pipeline_card"]]
        refs += got["skill_paths"] + got["memory_packs"] + got["bootstrap_docs"] + got["config"]
        gone = [r for r in refs if r and not (REPO_ROOT / r).exists()]
        if gone:
            missing[cmd.stem] = gone
    assert not missing, f"commands reference nonexistent paths: {missing}"


def test_profiles_match_golden_contract():
    golden = _golden()["profiles"]
    profiles = load_pipeline_profiles(REPO_ROOT / "configs" / "pipeline_profiles.yaml")
    assert set(profiles) == set(golden), (
        f"profile set drifted: extra={set(profiles) - set(golden)}, "
        f"missing={set(golden) - set(profiles)}"
    )
    problems: dict[str, list[str]] = {}
    for name, want in golden.items():
        prof = profiles[name]
        errs: list[str] = []
        for f in (
            "entry_agent",
            "manager_agent",
            "plan_reviewer_agent",
            "code_reviewer_agent",
            "tester_agent",
            "specialist_hint",
            "validation_mode",
            "primary_goal",
            "research_first",
            "pipeline_card",
        ):
            if getattr(prof, f) != want[f]:
                errs.append(f"{f} {getattr(prof, f)!r} != {want[f]!r}")
        got_skills = sorted(_skill_name(p) for p in prof.skills)
        if got_skills != want["skills"]:
            errs.append(f"skill set {got_skills} != {want['skills']}")
        if sorted(prof.bootstrap_docs) != want["bootstrap_docs"]:
            errs.append(f"bootstrap docs {prof.bootstrap_docs} != {want['bootstrap_docs']}")
        if _skill_name(prof.handoff_contract) != want["handoff_contract_skill"]:
            errs.append(f"handoff contract skill != {want['handoff_contract_skill']}")
        if errs:
            problems[name] = errs
        gone = validate_profile_assets(prof, REPO_ROOT)
        if gone:
            problems.setdefault(name, []).append(f"missing assets: {gone}")
    assert not problems, f"profiles drifted from golden contract: {problems}"


def test_artifact_contract_pinned_in_recipe_execution():
    # The cross-stage artifact contract frozen in the golden file must stay stated
    # verbatim in the coordinator's operational manual — the file every recipe run
    # inlines. If either side changes, this fails and the change must be deliberate.
    contract = _golden()["artifact_contract"]
    manual = (
        REPO_ROOT
        / ".opencode"
        / "skills"
        / "infra"
        / "pipeline"
        / "recipe-execution"
        / "SKILL.md"
    ).read_text(encoding="utf-8")
    missing = {kind: path for kind, path in contract.items() if path not in manual}
    assert not missing, (
        f"artifact contract entries absent from recipe-execution/SKILL.md: {missing}"
    )


def test_pipeline_cards_reference_existing_paths():
    missing: dict[str, list[str]] = {}
    for card in (REPO_ROOT / ".opencode" / "pipelines").glob("*.md"):
        text = card.read_text(encoding="utf-8")
        gone = sorted(
            {
                token
                for token in PATH_TOKEN_RE.findall(text)
                if not (REPO_ROOT / token).exists()
            }
        )
        if gone:
            missing[card.stem] = gone
    assert not missing, f"pipeline cards reference nonexistent paths: {missing}"
