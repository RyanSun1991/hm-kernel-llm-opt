# OpenCode Slash Commands

This directory contains command files for the OpenCode `/commands` feature. Each `.md` file defines a pre-configured task that can be triggered inside OpenCode by typing `/` followed by the command name.

## How It Works

1. Place `.md` files in this directory (`.opencode/commands/`).
2. Launch OpenCode from the kernel directory root.
3. Type `/` in the OpenCode session to see available commands.
4. Select a command to inject the full prompt with all `@`-referenced context files.

OpenCode expands `@<path>` references inline, so the agent receives the full content of all referenced agents, pipelines, skills, docs, and memory files in a single prompt.

## Available Commands

| Command | Pipeline / Agent | Description |
|---------|------------------|-------------|
| `optimize_generic` | `generic_full` pipeline | Full pipeline for any kernel target with auto-routing |
| `optimize_memmgr_reclaim` | `memmgr_reclaim_full` pipeline | Memory reclaim and allocator analysis |
| `optimize_hyperhold` | `hyperhold_full` pipeline | Swap I/O, compression, hpio, iotab |
| `optimize_workqueue` | `workqueue_full` pipeline | Workqueue and thread-pool optimization |
| `review_sync` | `sync_review` pipeline | Synchronization and lock safety review (no implementation) |
| `research_only` | `generic_full` pipeline | Research and analysis only (stops before implementation) |
| `function_detail` | `@kernel-function-research` primary agent | One-shot deep dive on ONE kernel function — design + callee-graph report, no optimization |
| `research` | `@kernel-research` primary agent | Iterative subsystem / file / function research with a human in the loop — builds a living design doc across many turns, no optimization |
| `plan` | `@kernel-plan` primary agent | Iterative ideation + planning with a human in the loop — reads an existing design doc + memory + idea ledger, triages per idea, produces a plan |

## Customizing a Command

Before triggering a command, you typically need to edit it to set your specific target:

1. Open the `.md` file in this directory.
2. Change the `Target:` line to your actual file or subsystem path.
3. Optionally adjust the `Objective:` to narrow or broaden the scope.
4. Optionally set `Auto-Iterate: N` — on clean pass verdicts the manager will auto-start another full pipeline pass on the same target, up to N passes total. Prior passes' plans/patches are treated as LANDED context and the researcher must find **orthogonal** new wins each pass. Default is 1 (single pass, legacy behavior). See `.opencode/skills/iterative-optimization/SKILL.md`.
5. Save and trigger via `/commands` in OpenCode.

## Creating Your Own Command

Copy any existing command file as a starting point:

```bash
cp .opencode/commands/optimize_generic.md .opencode/commands/my_custom_task.md
```

Then edit:

- **Profile / Pipeline**: pick from `pipelines/` or use `generic_full` for auto-routing
- **Target**: the kernel path or subsystem to analyze
- **Objective**: what the pipeline should achieve
- **Skill packs**: add or remove skills as needed
- **Bootstrap docs**: add subsystem-specific docs if available

## Language Control

All commands reference `@.opencode/config.yaml` and `@.opencode/skills/language-config/SKILL.md`. The session language is determined by the `language` field in `config.yaml`:

- `language: zh-CN` → all agent dialogue in Chinese
- `language: en` → all agent dialogue in English
