"""Evaluate effective Workbench permissions, rather than just checking YAML keys.

Pinned upstream semantics: OpenCode v1.18.30, not a repository-defined ordering:
https://github.com/anomalyco/opencode/blob/v1.18.30/packages/opencode/src/permission/index.ts#L25
https://github.com/anomalyco/opencode/blob/v1.18.30/packages/opencode/src/permission/index.ts#L169
https://github.com/anomalyco/opencode/blob/v1.18.30/packages/opencode/src/permission/index.ts#L185
https://github.com/anomalyco/opencode/blob/v1.18.30/packages/core/src/util/wildcard.ts#L2

The small Python adapter below covers the exercised evaluate/fromConfig/disabled
semantics for offline CI. It was cross-checked against the actual tagged
JavaScript function bodies and OpenCode GET /agent rules. In particular, this is
last-match precedence; a more-specific-first or YAML-key-presence test would miss
both hidden edit tools and disabled destructive-command denials.

These tests evaluate command strings, not a shell parser or a security sandbox.
Dangerous examples below are never executed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROLES = (
    "assistant", "researcher", "architect", "coordinator", "implementer", "reviewer", "validator",
)
PROFILES = (
    "bug-fix", "hyperhold-io", "kernel-understand", "reclaim-investigator", "sync-mechanism", "workqueue",
)
DEFINITIONS = (*ROLES, *("profiles/" + name for name in PROFILES))
SCOPED_WRITERS = tuple(name for name in DEFINITIONS if name not in {"assistant", "implementer"})
APPROVAL_GATED_SHELLS = tuple(name for name in DEFINITIONS if name != "implementer")


def _match(value, pattern):
    expression = re.escape(pattern.replace("\\", "/")).replace(r"\*", ".*").replace(r"\?", ".")
    expression = expression.replace(r"\ ", " ")
    if expression.endswith(" .*"):
        expression = expression[:-3] + "( .*)?"
    # Exercise Windows normalization explicitly even when CI runs on Linux.
    return re.fullmatch(
        expression, value.replace("\\", "/"), flags=re.DOTALL | re.IGNORECASE
    ) is not None


def _rules(config):
    return [
        (permission, pattern, action)
        for permission, value in config.items()
        for pattern, action in (value.items() if isinstance(value, dict) else [("*", value)])
    ]


def _evaluate(rules, permission, resource):
    return next(
        (
            action for tool, pattern, action in reversed(rules)
            if _match(permission, tool) and _match(resource, pattern)
        ),
        "ask",
    )


def _edit_tool_hidden(rules):
    # Upstream disabled() checks the last matching tool rule, even before it
    # knows the actual file path. All three edit/write/apply_patch tools share it.
    last = next((rule for rule in reversed(rules) if _match("edit", rule[0])), None)
    return last is not None and last[1:] == ("*", "deny")


def _role(name):
    path = ROOT / ".opencode/agents" / f"{name}.md"
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])


def test_adapter_keeps_upstream_last_match_wildcard_and_tool_visibility_semantics():
    wrong = _rules({"edit": {".opencode/local/**": "allow", "*": "deny"}})
    right = _rules({"edit": {"*": "deny", ".opencode/local/**": "allow"}})
    target = ".opencode/local/workspaces/probe/capsule.md"
    assert _evaluate(wrong, "edit", target) == "deny"
    assert _edit_tool_hidden(wrong)
    assert _evaluate(right, "edit", target) == "allow"
    assert not _edit_tool_hidden(right)
    assert _evaluate(right, "edit", "src/kernel.c") == "deny"
    assert _match(".OPENCODE\\local\\workspaces\\capsule.md", ".opencode/local/**")
    assert _match("rm -rf", "rm -rf *")
    assert not _match("prefix/rm -rf", "rm -rf *")


@pytest.mark.parametrize("name", SCOPED_WRITERS)
def test_role_can_write_workspace_artifacts_while_source_remains_denied(name):
    rules = _rules(_role(name)["permission"])
    for target in (
        ".opencode/local/workspaces/probe/capsule.md",
        ".opencode/local/workspaces/probe/artifacts/research/research-note.md",
        ".OPENCODE\\local\\workspaces\\probe\\execution.json",
    ):
        assert _evaluate(rules, "edit", target) == "allow", (name, target)
    assert not _edit_tool_hidden(rules), name
    for target in ("src/kernel.c", "README.md", "opencode.jsonc"):
        assert _evaluate(rules, "edit", target) == "deny", (name, target)


@pytest.mark.parametrize("name", DEFINITIONS)
def test_fixed_inspection_commands_retain_effective_permission(name):
    rules = _rules(_role(name)["permission"])
    for command in (
        "pwd",
        "git status --short",
        "git rev-parse HEAD",
        "git rev-parse --show-toplevel",
        "git log --no-patch --oneline -10",
        "git diff --no-ext-diff --no-textconv --stat",
        "git show --no-ext-diff --no-textconv --stat HEAD",
    ):
        assert _evaluate(rules, "bash", command) == "allow", (name, command)
    if name != "implementer":
        assert _evaluate(rules, "bash", "python mutate.py") == "ask"
        assert _evaluate(rules, "bash", "git -C target status --short") == "ask"


@pytest.mark.parametrize("name", APPROVAL_GATED_SHELLS)
def test_shell_side_effects_and_unlisted_arguments_require_approval(name):
    rules = _rules(_role(name)["permission"])
    for command in (
        "git diff --output=counter.py",
        "git diff --no-ext-diff --no-textconv --stat --output=counter.py",
        "git log --output=counter.py",
        "git log --no-patch --oneline -10 --output=counter.py",
        "git show --output=counter.py HEAD",
        "git diff --ext-diff",
        "git diff --textconv",
        "git show --no-ext-diff --no-textconv --stat HEAD --ext-diff",
        "git status-helper",
        "git rev-parse-helper",
        "ls-helper",
        "find . -delete",
        "find . -exec custom-command {} +",
        "rg --pre custom-command pattern src",
        "rg pattern src",
        "git diff --stat",
        "git log -1",
        "cat counter.py",
        "head -n 10 counter.py",
        "tail -n 10 counter.py",
        "grep pattern counter.py",
        "wc -l counter.py",
        "git -C target diff --no-ext-diff --no-textconv",
    ):
        assert _evaluate(rules, "bash", command) == "ask", (name, command)


@pytest.mark.parametrize("name", APPROVAL_GATED_SHELLS)
def test_inspection_allow_rules_cannot_absorb_additional_arguments(name):
    rules = _rules(_role(name)["permission"])
    for permission, command, action in rules:
        if permission != "bash" or action != "allow":
            continue
        # No wildcard can silently accept new options with different semantics.
        assert "*" not in command and "?" not in command, (name, command)
        assert _evaluate(rules, "bash", command + " --output=counter.py") == "ask"


@pytest.mark.parametrize("command", ["git push origin opencode", "git reset --hard HEAD", "git clean -fd", "rm -rf output"])
def test_implementer_specific_destructive_denials_override_default_allow(command):
    rules = _rules(_role("implementer")["permission"])
    assert _evaluate(rules, "bash", command) == "deny"
    assert _evaluate(rules, "bash", "python -m pytest tests") == "allow"
    assert _evaluate(rules, "edit", "src/kernel.c") == "ask"


@pytest.mark.parametrize("name", DEFINITIONS)
def test_delegation_and_skill_ceilings_are_unchanged(name):
    rules = _rules(_role(name)["permission"])
    assert _evaluate(rules, "task", "reviewer") == ("allow" if name == "coordinator" else "ask")
    assert _evaluate(rules, "skill", "delegate") == ("allow" if name == "coordinator" else "deny")
