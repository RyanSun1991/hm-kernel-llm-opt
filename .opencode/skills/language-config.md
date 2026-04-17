# Language Configuration

## Purpose

This skill ensures all agents in the pipeline respond in the language configured in `.opencode/config.yaml`.

## Startup Rule (Non-Negotiable)

At the start of every agent session, before any other work:

1. Read `.opencode/config.yaml`.
2. Find the `language` field.
3. Apply the language setting to **all** subsequent output for the current session.

## Language Scope

The configured language applies to:

- all user-facing dialogue and status updates
- research notes and analysis prose
- optimization plans and design docs (prose sections)
- handoff packet prose fields (target description, risk notes, open questions, next action)
- plan review and code review verdicts and commentary
- tester reports and validation summaries
- memory updates (prose sections)
- error messages and clarification questions

The configured language does **NOT** apply to:

- code (source code, patches, diffs)
- code comments in source files (keep English for upstream compatibility)
- variable names, function names, struct names
- git commit messages (keep English)
- file paths and artifact references
- technical terms that have no widely accepted translation (e.g., hot path, instruction count, MCP) — keep the English term inline, optionally with a parenthetical translation on first use

## Supported Values

| Value   | Language              |
|---------|-----------------------|
| `zh-CN` | Chinese (Simplified)  |
| `en`    | English               |

## Default

If `.opencode/config.yaml` is missing or the `language` field is absent, default to `en` (English).

## Agent Behavior

- When `language: zh-CN`: respond in Simplified Chinese. Use Chinese for all prose. Keep English technical terms inline where no standard Chinese equivalent exists.
- When `language: en`: respond in English. This is the legacy default behavior.

## Example

With `language: zh-CN`, a plan reviewer would write:

> **审核结论**: 批准。该计划的 instruction-count 假设是可信的。hot path 上的冗余分支移除预计减少约 15% 的指令数。主要风险是并发场景下的 lifetime 保证，实现时必须保留 refcount 检查。

With `language: en`, the same output would be:

> **Review Verdict**: Approved. The instruction-count hypothesis is credible. Removing the redundant branch on the hot path is expected to reduce instruction count by ~15%. The main risk is lifetime guarantees under concurrency — the implementation must preserve the refcount check.
