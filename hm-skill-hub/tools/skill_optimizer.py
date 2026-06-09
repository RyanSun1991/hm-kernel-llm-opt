"""Bounded-edit skill optimizer scaffold.

Consumes JSONL rollout traces with optional fields `missing_terms`, `lesson`, and
`anchor`, then emits a schema-shaped skill_patch manifest. It intentionally does
not edit SKILL.md in-place; reviewers apply accepted bounded edits after eval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--baseline-version", default="0.0.0")
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()
    traces = _load_jsonl(args.traces)
    edit_ops = []
    seen: set[str] = set()
    for trace in traces:
        lesson = trace.get("lesson") or "; ".join(trace.get("missing_terms", []))
        if not lesson or lesson in seen:
            continue
        seen.add(lesson)
        edit_ops.append({
            "op": "add",
            "anchor": trace.get("anchor", "## How to use"),
            "rationale": trace.get("rationale", "rollout trace gap"),
            "after": f"- {lesson}",
        })
        if len(edit_ops) >= 3:
            break
    manifest = {
        "skill": args.skill,
        "edit_ops": edit_ops or [{"op": "add", "anchor": "## Notes", "rationale": "no-op candidate", "after": "- Preserve current behavior."}],
        "task_suite": args.task_suite,
        "metrics": {"pass_rate": 0.0, "instr_count_delta": 0.0, "regression_rate": 1.0},
        "baseline_version": args.baseline_version,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
