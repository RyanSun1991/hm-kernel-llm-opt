"""OpenCode pipeline preset loading and session bootstrapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_PATH = Path("configs/pipeline_profiles.yaml")
DEFAULT_STATE_PATH = Path(".opencode/state/current_task.json")
DEFAULT_PROMPT_PATH = Path(".opencode/state/current_prompt.md")


@dataclass
class PipelineProfile:
    name: str
    title: str
    description: str
    entry_agent: str = "coordinator"
    manager_agent: str = "coordinator"
    plan_reviewer_agent: str = "reviewer"
    code_reviewer_agent: str = "reviewer"
    tester_agent: str = "validator"
    specialist_hint: str = ""
    pipeline_card: str = ""
    objective_template: str = ""
    bootstrap_docs: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    validation_mode: str = "plan_review"
    research_first: bool = True
    artifacts_expected: list[str] = field(default_factory=list)
    memory_mode: str = "auto"
    primary_goal: str = "instruction_count"
    handoff_contract: str = ".opencode/skills/infra/pipeline/handoff-contract/SKILL.md"
    flash_relay_url: str = ""
    stock_image_dir: str = ""


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "task"


def _ensure_workspace(repo_root: Path) -> None:
    for rel in (
        ".opencode",
        ".opencode/agents",
        ".opencode/docs",
        ".opencode/state",
        ".opencode/plans",
        ".opencode/reviews",
        ".opencode/bench",
        ".opencode/patches",
        ".opencode/pipelines",
        ".opencode/skills",
        ".opencode/memory",
        ".opencode/memory/targets",
        ".opencode/memory/subsystems",
    ):
        (repo_root / rel).mkdir(parents=True, exist_ok=True)


def _target_memory_slug(target: str) -> str:
    return _slugify(target)


def _infer_memory_paths(repo_root: Path, target: str) -> list[str]:
    slug = _target_memory_slug(target)
    candidates = [
        repo_root / ".opencode" / "memory" / "targets" / f"{slug}.md",
        repo_root / ".opencode" / "memory" / "global_lessons.md",
    ]
    # as_posix keeps the emitted paths forward-slashed on Windows too — they are
    # written into prompts and state files that the rest of the harness matches
    # against `.opencode/...` literals.
    return [path.relative_to(repo_root).as_posix() for path in candidates]


def load_pipeline_profiles(path: str | Path = DEFAULT_PROFILES_PATH) -> dict[str, PipelineProfile]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    profiles = raw.get("profiles", {}) or {}
    result: dict[str, PipelineProfile] = {}
    for name, data in profiles.items():
        if not isinstance(data, dict):
            raise ValueError(f"pipeline profile {name!r} must be a mapping")
        result[name] = PipelineProfile(
            name=name,
            title=str(data.get("title") or name),
            description=str(data.get("description") or ""),
            entry_agent=str(data.get("entry_agent") or "coordinator"),
            manager_agent=str(data.get("manager_agent") or "coordinator"),
            plan_reviewer_agent=str(data.get("plan_reviewer_agent") or "reviewer"),
            code_reviewer_agent=str(data.get("code_reviewer_agent") or "reviewer"),
            tester_agent=str(data.get("tester_agent") or "validator"),
            specialist_hint=str(data.get("specialist_hint") or ""),
            pipeline_card=str(data.get("pipeline_card") or ""),
            objective_template=str(data.get("objective_template") or ""),
            bootstrap_docs=[str(item) for item in (data.get("bootstrap_docs") or [])],
            skills=[str(item) for item in (data.get("skills") or [])],
            validation_mode=str(data.get("validation_mode") or "plan_review"),
            research_first=bool(data.get("research_first", True)),
            artifacts_expected=[str(item) for item in (data.get("artifacts_expected") or [])],
            memory_mode=str(data.get("memory_mode") or "auto"),
            primary_goal=str(data.get("primary_goal") or "instruction_count"),
            handoff_contract=str(
                data.get("handoff_contract")
                or ".opencode/skills/infra/pipeline/handoff-contract/SKILL.md"
            ),
            flash_relay_url=str(data.get("flash_relay_url") or ""),
            stock_image_dir=str(data.get("stock_image_dir") or ""),
        )
    return result


def validate_profile_assets(profile: PipelineProfile, repo_root: str | Path) -> list[str]:
    """Return a list of referenced asset paths that do not exist on disk.

    The staged prompt names an entry agent and a set of skill/card/doc files.
    If any are missing (e.g. a stale `os-opt-manager` reference or a flat
    `skills/<name>.md` path), the prompt silently points the harness at nothing.
    Surface those so the mismatch is visible instead of failing at runtime.
    """
    root = Path(repo_root)
    missing: list[str] = []

    # Agents may live at agents/ top level (workbench roles), agents/legacy/
    # (pre-workbench pipeline cast), or agents/profiles/ (compositions).
    agents_root = root / ".opencode" / "agents"
    agent_candidates = [
        agents_root / f"{profile.entry_agent}.md",
        agents_root / "legacy" / f"{profile.entry_agent}.md",
        agents_root / "profiles" / f"{profile.entry_agent}.md",
    ]
    if not any(candidate.exists() for candidate in agent_candidates):
        missing.append(agent_candidates[0].relative_to(root).as_posix())

    candidates: list[str] = list(profile.skills)
    candidates += list(profile.bootstrap_docs)
    if profile.pipeline_card:
        candidates.append(profile.pipeline_card)
    if profile.handoff_contract:
        candidates.append(profile.handoff_contract)
    for rel in candidates:
        if (root / rel).exists():
            continue
        missing.append(rel)
    return missing


def _parse_artifact_specs(artifact_specs: list[str]) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for spec in artifact_specs:
        if ":" not in spec:
            raise ValueError(f"invalid artifact spec: {spec!r}; expected kind:path")
        kind, path = spec.split(":", 1)
        artifacts.append({"kind": kind.strip(), "path": path.strip()})
    return artifacts


def build_pipeline_prompt(
    profile: PipelineProfile,
    *,
    target: str,
    objective: str,
    artifact_specs: list[str] | None = None,
) -> str:
    artifacts = _parse_artifact_specs(artifact_specs or [])
    lines = [
        f"@{profile.entry_agent}",
        f"Profile: {profile.name}",
        f"Title: {profile.title}",
        f"Target: {target}",
        f"Objective: {objective}",
        f"Manager: {profile.manager_agent}",
        f"Plan reviewer: {profile.plan_reviewer_agent}",
        f"Code reviewer: {profile.code_reviewer_agent}",
        f"Tester: {profile.tester_agent}",
        f"Specialist hint: {profile.specialist_hint or 'auto'}",
        f"Primary metric: {profile.primary_goal}",
        f"Validation mode: {profile.validation_mode}",
        f"Memory mode: {profile.memory_mode}",
        f"Research first: {'yes' if profile.research_first else 'no'}",
    ]
    if profile.pipeline_card:
        lines.append(f"Pipeline preset: {profile.pipeline_card}")
    if profile.bootstrap_docs:
        lines.append("Bootstrap docs:")
        lines.extend(f"- {item}" for item in profile.bootstrap_docs)
    if profile.skills:
        lines.append("Skill packs:")
        lines.extend(f"- {item}" for item in profile.skills)
    if profile.handoff_contract:
        lines.append(f"Handoff contract: {profile.handoff_contract}")
    if profile.flash_relay_url:
        lines.append(f"Flash relay URL: {profile.flash_relay_url}")
    if profile.stock_image_dir:
        lines.append(f"Stock image dir: {profile.stock_image_dir}")
    if artifacts:
        lines.append("Artifacts:")
        lines.extend(f"- {item['kind']}: {item['path']}" for item in artifacts)
    lines.extend(
        [
            "Requirements:",
            "- Treat instruction-count reduction as the primary optimization target unless the staged task explicitly overrides it.",
            "- Use the full staged multi-agent workflow.",
            "- Route research outputs to the plan reviewer before implementation.",
            "- Route implementation outputs to the code reviewer before test execution.",
            "- Route reviewed patches to the tester agent for Build MCP and Auto-Test MCP validation.",
            "- Every handoff must include target files, hot path, instruction-count hypothesis, risks, and required next actions.",
            "- Save all durable artifacts under .opencode/.",
            "- Research first unless the approved preset explicitly skips it.",
            "- Use Sequential Thinking MCP first and Kernel Index MCP early.",
            "- Produce design doc, plan, plan-review, implementation handoff, code-review, and validation outputs as applicable.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def initialize_pipeline_session(
    *,
    repo_root: str | Path,
    profile: PipelineProfile,
    target: str,
    objective: str | None = None,
    artifact_specs: list[str] | None = None,
    state_path: str | Path = DEFAULT_STATE_PATH,
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    _ensure_workspace(root)

    missing_assets = validate_profile_assets(profile, root)
    if missing_assets:
        logger.warning(
            "Pipeline profile %r references %d missing asset(s): %s",
            profile.name,
            len(missing_assets),
            ", ".join(missing_assets),
        )

    task_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{profile.name}-{_slugify(target)}"
    resolved_objective = objective or profile.objective_template.format(target=target).strip()
    if not resolved_objective:
        resolved_objective = f"Run the full {profile.name} pipeline for {target}"

    prompt_text = build_pipeline_prompt(
        profile,
        target=target,
        objective=resolved_objective,
        artifact_specs=artifact_specs or [],
    )
    memory_paths = _infer_memory_paths(root, target) if profile.memory_mode == "auto" else []
    if memory_paths:
        memory_lines = ["Memory files:"]
        memory_lines.extend(f"- {item}" for item in memory_paths)
        prompt_text = prompt_text.rstrip() + "\n" + "\n".join(memory_lines) + "\n"

    resolved_state_path = root / Path(state_path)
    resolved_prompt_path = root / Path(prompt_path)
    resolved_state_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_prompt_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "task_id": task_id,
        "title": profile.title,
        "status": "staged",
        "manager": profile.manager_agent,
        "active_agent": profile.entry_agent,
        "plan_reviewer_agent": profile.plan_reviewer_agent,
        "code_reviewer_agent": profile.code_reviewer_agent,
        "tester_agent": profile.tester_agent,
        "primary_metric": profile.primary_goal,
        "approved_plan": "",
        "review_status": "",
        "plan_review_status": "",
        "code_review_status": "",
        "test_status": "",
        "current_handoff": "",
        "handoff_log": [],
        "profile": profile.name,
        "target": target,
        "objective": resolved_objective,
        "primary_goal": profile.primary_goal,
        "handoff_contract": profile.handoff_contract,
        "pipeline_card": profile.pipeline_card,
        "bootstrap_docs": profile.bootstrap_docs,
        "skills": profile.skills,
        "flash_relay_url": profile.flash_relay_url,
        "stock_image_dir": profile.stock_image_dir,
        "artifacts": _parse_artifact_specs(artifact_specs or []),
        "memory_files": memory_paths,
        "missing_assets": missing_assets,
        "prompt_file": str(resolved_prompt_path.relative_to(root)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": [],
    }
    resolved_state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    resolved_prompt_path.write_text(prompt_text, encoding="utf-8")
    logger.info(
        "Initialized OpenCode pipeline session task_id=%s profile=%s target=%s",
        task_id,
        profile.name,
        target,
    )
    return {
        "task": payload,
        "state_path": str(resolved_state_path),
        "prompt_path": str(resolved_prompt_path),
        "prompt_text": prompt_text,
    }


def resume_pipeline_session(
    *,
    repo_root: str | Path,
    state_path: str | Path = DEFAULT_STATE_PATH,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    resolved_state_path = root / Path(state_path)
    payload = json.loads(resolved_state_path.read_text(encoding="utf-8"))
    prompt_file = payload.get("prompt_file")
    prompt_text = ""
    prompt_path = ""
    if prompt_file:
        resolved_prompt_path = root / str(prompt_file)
        prompt_path = str(resolved_prompt_path)
        if resolved_prompt_path.exists():
            prompt_text = resolved_prompt_path.read_text(encoding="utf-8")
    return {
        "task": payload,
        "state_path": str(resolved_state_path),
        "prompt_path": prompt_path,
        "prompt_text": prompt_text,
    }
