---
name: mm-reclaim
kind: domain
version: 0.1.0
maturity: L0
applies_to:
  subsystems: [mm-reclaim]
  path_globs: ["mm/vmscan.c", "mm/*reclaim*", "mm/page_alloc.c"]
  symbol_selectors: ["shrink_*", "*_reclaim", "kswapd*"]
requires: [core/example, technique/hoist-loop-invariant]
eval_id: eval/task_suites/mm_reclaim_suite
owners: ["@maintainers"]
status: experimental
---

# mm-reclaim (domain skill scaffold)

Example domain skill demonstrating the §6.2 selector binding: the `applies_to`
selectors are resolved by `resolver.py` against the current code index, so a
kernel rebase only updates `_registry/subsystem_selectors.yaml`, not this file.

This is a Phase-1 scaffold so the resolver's `selector → domain → requires`
chain is exercisable end-to-end. Replace with migrated `memmgr-reclaim` content
in the Phase 0.5 content-migration session.

## When to use

Target is in the memory-reclaim subsystem (`mm/vmscan.c`, `shrink_*`, `*_reclaim`,
`kswapd*`).

## How to use

Pull the required `core/` and `technique/` skills, then consult the
target-anchored knowledge the resolver mounts for the specific function.
