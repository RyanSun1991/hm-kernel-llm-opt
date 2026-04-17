# OpenCode One-Click Pipeline Guide

This guide describes the fastest repo-native way to stage and run the OpenCode multi-agent workflow.

## What Was Added

The repo now has:

- a one-shot starter agent at `.opencode/agents/kernel-pipeline-starter.md`
- pipeline presets under `.opencode/pipelines/`
- reusable skill packs under `.opencode/skills/`
- machine-readable profiles in `configs/pipeline_profiles.yaml`
- CLI staging commands:
  - `python3 -m hmopt.cli list-pipeline-profiles`
  - `python3 -m hmopt.cli start-pipeline`
  - `python3 -m hmopt.cli resume-pipeline`
- a wrapper script:
  - `bash scripts/run_opencode_pipeline.sh`

## Fastest Interactive Flow

1. Start MCP services.

2. Stage a pipeline session:

```bash
python3 -m hmopt.cli start-pipeline \
  --profile hyperhold_full \
  --target sysmgr/memmgr/mem/swap/hyperhold/hp_iotab.c
```

3. Copy the staged prompt from `.opencode/state/current_prompt.md` into OpenCode.

4. OpenCode starts with `kernel-pipeline-starter`, then hands off to `os-opt-manager`, then to the appropriate specialists.

## Fastest Wrapper Flow

```bash
bash scripts/run_opencode_pipeline.sh \
  --profile hyperhold_full \
  --target sysmgr/memmgr/mem/swap/hyperhold/hp_iotab.c \
  --start-mcp
```

Optional flags:

- `--index-kernel`
- `--repo-path ...`
- `--compile-commands-dir ...`
- `--launch-opencode`
- `--open-cmd "opencode"`

## Preset Profiles

Current presets:

- `hyperhold_full`
- `memmgr_reclaim_full`
- `sync_review`
- `workqueue_full`

Use `python3 -m hmopt.cli list-pipeline-profiles` to inspect them.

## Current Automation Boundary

The repo can now stage the entire OpenCode control-plane session with one command.

If the local `opencode` binary exists, the launcher can open it.

The remaining human action is usually one of:

- paste the staged prompt into OpenCode
- approve the top-ranked optimization idea
- approve final code landing

That is intentional, because the workflow keeps explicit human gates around optimization choice and final review.
