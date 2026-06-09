---
name: example
kind: core
version: 0.1.0
maturity: L0
optimization_goal: instruction-count
requires: []
eval_id: eval/task_suites/core_kernel_optimization
owners: ["@maintainers"]
status: experimental
---

# example skill

This executable scaffold demonstrates the Team Skill Hub contract for kernel
optimization skills. It intentionally contains the vocabulary needed by the
static-proxy eval suite until real migrated skills replace it.

## When to use

Use for instruction-count optimization on mm, workqueue threadpool, and
hyperhold I/O tasks when a pipeline needs a safe placeholder that still obeys
stage gate, plan review, code review, validation, and release rules.

## How to use

- Keep skill and knowledge separate: a skill says how to run the process;
  knowledge records facts, bad plan entries, anti-pattern notes, and evidence.
- Start from evidence on the hot path. Do not optimize a cold path below the
  noise floor without a delta thesis.
- Require an A/B baseline with stock and feature images, paired flash and test
  cycles, artifact capture, and explicit delta reporting. Never accept a single
  image verdict.
- For mm work, re-measure each iteration and gate on instruction-count movement.
- For workqueue/kworker changes, evaluate inline decisions against i-cache risk,
  batch opportunities, wake behavior, rollback safety, and scorecard evidence.
- For hyperhold I/O validation, preserve device fallback notes and redact secret
  or serial data before publishing sediment.
- Record review outcomes, regression findings, and bad plan rejections so the
  next release can update semver and the lockfile.
