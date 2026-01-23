"""Prompt registry for LLM agents and query pipelines."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


@dataclass(frozen=True)
class PromptSpec:
    name: str
    system: str
    template: str
    description: str | None = None


DEFAULT_PROMPTS: dict[str, dict[str, str]] = {
    "conductor": {
        "system": "Conductor focusing on perf + correctness.",
        "template": (
            "You are the Conductor agent coordinating an optimization loop.\n"
            "Iteration: {iteration}/{max_iterations}\n"
            "Best run summary: {best_summary}\n"
            "Current evidence:\n{evidence_summary}\n\n"
            "Decide whether to continue optimizing or stop. "
            "If continuing, propose a concise next action for the coder."
        ),
    },
    "coder": {
        "system": "Return only a unified diff.",
        "template": (
            "You are the Coder agent. Produce a minimal unified diff. "
            "Prefer editing or creating documentation/config files unless a specific "
            "code path is given.\n"
            "Iteration: {iteration}\n"
            "Instruction: {instructions}"
        ),
    },
    "trace_analyst": {
        "system": "Trace analyst focusing on HM kernel perf.",
        "template": (
            "You are the Trace Analyst. Summarize performance symptoms and hotspot classes.\n"
            "Metrics: {metrics}\n"
            "Hotspots:\n"
            "{hotspots}\n"
            "{code_context_block}"
            "{insight_block}"
        ),
    },
    "query_runtime_code": {
        "system": "You are a kernel performance expert.",
        "template": (
            "Step 1: interpret runtime signals.\n"
            "Step 2: use the candidate code snippets to ground the analysis.\n"
            "Step 3: expand with graph relations (callers/callees/types).\n\n"
            "Runtime summary:\n{runtime_summary}\n\n"
            "Candidate code snippets:\n{code_context}\n\n"
            "Question: {query}"
        ),
    },
    "query_code": {
        "system": "You are a kernel performance expert.",
        "template": "Answer the question using code context.\n\nQuestion: {query}",
    },
    "query_runtime": {
        "system": "You are a kernel performance analyst.",
        "template": "Answer the question using runtime evidence.\n\nQuestion: {query}",
    },
    "query_graph": {
        "system": "You are a kernel performance expert.",
        "template": "Answer the question using graph relations.\n\nQuestion: {query}",
    },
    "code_review": {
        "system": "You are a senior kernel reviewer focused on correctness and performance risk.",
        "template": (
            "Review the proposed patch for correctness and performance impact.\n"
            "Evidence summary:\n{evidence_summary}\n\n"
            "Patch diff:\n{patch_diff}\n\n"
            "Return a short decision (approve/reject) and rationale."
        ),
    },
}


class PromptRegistry:
    def __init__(self, root: Path, overrides: Mapping[str, Mapping[str, str]] | None = None):
        self.root = Path(root)
        self.overrides = overrides or {}
        self._cache: dict[str, PromptSpec] = {}

    def load(self, name: str) -> PromptSpec:
        if name in self._cache:
            return self._cache[name]

        data: dict[str, str] | None = None
        path = self._find_prompt_file(name)
        if path:
            raw = os.path.expandvars(path.read_text(encoding="utf-8"))
            if path.suffix in {".yaml", ".yml"}:
                payload = yaml.safe_load(raw) or {}
            else:
                payload = json.loads(raw)
            if isinstance(payload, dict):
                data = {k: v for k, v in payload.items() if isinstance(v, str)}

        if not data:
            data = dict(DEFAULT_PROMPTS.get(name, {}))

        override = self.overrides.get(name) or {}
        if override:
            merged = dict(data)
            merged.update({k: v for k, v in override.items() if isinstance(v, str)})
            data = merged

        if not data:
            raise KeyError(f"prompt '{name}' not found")

        spec = PromptSpec(
            name=name,
            system=data.get("system") or data.get("system_prompt", ""),
            template=data.get("template") or data.get("prompt") or data.get("user", ""),
            description=data.get("description"),
        )
        self._cache[name] = spec
        return spec

    def render(self, name: str, **kwargs: object) -> tuple[str, str]:
        spec = self.load(name)
        try:
            rendered = spec.template.format(**kwargs)
        except KeyError as exc:
            raise ValueError(f"Missing template variable {exc} for prompt '{name}'") from exc
        return spec.system, rendered

    def _find_prompt_file(self, name: str) -> Path | None:
        if not self.root:
            return None
        for ext in (".yaml", ".yml", ".json"):
            candidate = self.root / f"{name}{ext}"
            if candidate.exists():
                return candidate
        return None


def build_prompt_registry(config) -> PromptRegistry:
    root = getattr(getattr(config, "prompts", None), "dir", None) or Path("configs/prompts")
    overrides = getattr(getattr(config, "prompts", None), "overrides", None) or {}
    return PromptRegistry(Path(root), overrides=overrides)


def resolve_prompt_name(config, purpose: str, default_name: str) -> str:
    prompts_cfg = getattr(config, "prompts", None)
    if not prompts_cfg:
        return default_name
    stage_profiles = getattr(prompts_cfg, "stage_profiles", None) or {}
    profiles = getattr(prompts_cfg, "profiles", None) or {}
    profile = stage_profiles.get(purpose) or getattr(prompts_cfg, "profile", None)
    if profile and profile in profiles:
        mapped = profiles.get(profile) or {}
        name = mapped.get(purpose)
        if name:
            return name
    return default_name
