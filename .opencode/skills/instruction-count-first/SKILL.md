---
name: instruction-count-first
description: Default optimization objective — lower instruction count on the hot path. Defines what counts as IC work, required framing for every artifact, and tie-breakers for ranking ideas.
---

# Instruction-Count First

## Scope: the `compute-bound` default (not universal)

This skill is the **`compute-bound`** playbook in `perf-bottleneck-playbooks` and
the default when the bottleneck class is undetermined. It is the right primary
metric when the hot path is straight-line, cache-hot CPU work.

It is NOT the primary metric when the funnel's Stage 0 classifies the target as:

- `memory-tlb-bound` (mprotect/madvise/munmap…) — TLB flush + page-walk dominate,
  which IC cannot see; use `memory-tlb-optimization`.
- `ipc-bound` (fcntl `F_GETFL`, getrandom, dup…) — a cross-component round-trip
  dominates; IC is meaningful only counted **whole-path** (client+transfer+server).
- `io-bound` (fio…) — faults/writeback/storage dominate; IC is near-irrelevant.

For those classes IC is at most a **secondary, whole-path** signal — never declare
a win on the in-kernel-leg IC alone. See `perf-bottleneck-playbooks/SKILL.md`.

## Default Objective (compute-bound)

When the class is `compute-bound` or undetermined — and unless the staged task
explicitly says otherwise — optimize for lower instruction count on the hot path first.

## What Counts As Instruction-Count Work

- redundant branches
- repeated loads and stores
- repeated pointer chasing
- duplicated bookkeeping
- avoidable synchronization overhead
- repeated allocation, copy, and serialization work
- control-flow churn that does not change externally visible behavior

## Required Framing

Every research note, plan, code review, and tester report should answer:

- where instructions are currently being spent
- what exact mechanism is expected to reduce them
- what correctness or lifecycle constraints limit the change
- what artifact will later confirm or falsify the expected win

## Tie-Breakers

When two ideas look similar, prefer the one that:

1. touches the measured hot path more directly
2. removes instructions with less concurrency or lifetime risk
3. is easier to validate with Build MCP, Auto-Test MCP, and trace artifacts
