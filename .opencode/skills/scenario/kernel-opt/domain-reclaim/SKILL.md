---
name: domain-reclaim
description: >-
  Domain pack for sysmgr/memmgr reclaim — allocator slow paths, sync/async reclaim,
  vmpressure, PSI, watermarks, and reclaim-control interactions. Pure domain knowledge
  extracted from the legacy memmgr-reclaim-research specialist: any role loads it (a
  researcher to investigate, a reviewer to know the failure modes). No process rules.
---

# Domain Pack — memmgr / reclaim

## Scope (path anchors)

- `sysmgr/memmgr/mem/reclaim/**` — reclaim instances, scan/shrink control
- `sysmgr/memmgr/page/**` — page allocator coupling, slow-path entry into reclaim
- `sysmgr/memmgr/psi/**` — pressure stall information
- `sysmgr/memmgr/mem/vmpressure.c` — vmpressure signaling
- `sysmgr/memmgr/mem/stat/**` — bookkeeping the hot paths keep updating

## Read these before exploring from scratch

1. `.opencode/docs/memmgr-reclaim_bootstrap.md` — accumulated subsystem context
   (fold stable new context back into it)
2. `.opencode/docs/memmgr-reclaim_design.md` and `_trace.md` if present
3. `.opencode/memory/subsystems/memmgr-reclaim.md` if present, plus
   `.opencode/memory/targets/<target>.md` when the task names a target
4. Reject ledgers before proposing mechanisms: `.opencode/state/bad_plans.md` and
   `.opencode/state/memmgr-reclaim-bad_plans.md` if present (`ls .opencode/state/`)

## The system model a claim must rest on

Establish these before any conclusion about reclaim behavior:

- **Entry points**: how allocation slow paths enter reclaim; who else triggers it
  (watermark breach, pressure signal, explicit control)
- **Sync vs async reclaim**: which path the target actually takes under the workload
  in question — their costs and interleavings differ fundamentally
- **Reclaim instance ordering and callbacks**: registration order, per-instance
  scan/shrink callbacks, how much work each pass does and what cuts it short
- **Watermarks and pressure signals**: thresholds, hysteresis, who reads them and how
  stale they may be
- **PSI / vmpressure interaction**: what feeds the signals, who consumes them, and
  the feedback loops they close
- **Page allocator coupling**: what reclaim returns to the allocator and how retry
  loops react

## Optimization-sensitive spots (where cost usually hides)

- repeated per-page bookkeeping inside scan loops (stat updates, flag checks that
  cannot change mid-loop)
- re-computed watermark / pressure checks on paths where inputs are loop-invariant
- callback indirection in reclaim instances whose flexibility no in-tree instance
  uses
- lock acquisitions inside scan loops that could hoist, batch, or narrow
- sync-reclaim work performed where async already covers it (or vice versa)

## Domain-specific review cautions

- Reclaim correctness is lifetime correctness: a page's state can change between
  check and use unless the owning lock is held — verify the lock actually covers the
  callback window a change relies on.
- Pressure signals are deliberately damped; "optimizing away" a read or delaying an
  update can destabilize the control loop far from the edited function.
- Watermark semantics encode product tuning — a threshold change is a behavior
  change, never a pure optimization.
