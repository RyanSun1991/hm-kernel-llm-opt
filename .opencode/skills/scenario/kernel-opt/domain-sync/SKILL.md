---
name: domain-sync
description: >-
  Domain pack for synchronization mechanisms — lock ownership and scope, waiter queues
  and wakeup ordering, refcount lifetime, state machines, race windows, contention.
  Pure domain knowledge extracted from the legacy basic-mechanism-sync-opt specialist:
  any role loads it (reviewers especially — these are the failure modes). No process rules.
---

# Domain Pack — synchronization / lifecycle mechanisms

## Scope (anchors)

- lock primitives in use (mutex / rwlock / spinlock / seqlock variants) and the
  fields each instance protects
- waiter queues and wakeup ordering guarantees
- refcount transitions and object lifetime boundaries
- state machines with concurrent observers, and their race windows
- contention amplification (lock convoys, cacheline bouncing, reader/writer
  starvation)

## Read these before exploring from scratch

1. Existing sync design docs matching the target (`ls .opencode/docs/` — pattern
   `*_sync_design.md`)
2. `.opencode/memory/subsystems/sync.md` if present, plus target memory as named
3. Reject ledgers: `.opencode/state/bad_plans.md` and
   `.opencode/state/basic-mechanism-sync-opt-bad_plans.md` if present

## The system model a claim must rest on

- **What each primitive protects** — the field set, stated precisely. "The lock
  around this function" is not a protection model.
- **Ownership**: which context may hold what, in which order; the implied lock
  hierarchy and every place it is documented or merely assumed
- **Lifetime**: what keeps the object alive at each use site — refcount held,
  RCU-like grace, ownership transfer — and where the last-put can land
- **Waiter behavior**: wait conditions, wakeup sources, whether wakeups can be
  spurious/lost/reordered, and who re-checks conditions after waking
- **The race windows**: for each check-then-act, what can interleave between the
  check and the act, and what makes it safe today

## Optimization-sensitive spots (where cost usually hides)

- lock scope wider than the protected access (work under the lock that touches
  nothing the lock protects)
- locks protecting fields touched by disjoint call paths — splittable per-field or
  per-path when the sharing is accidental
- refcount round-trips (get+put within one call chain that already holds a
  reference)
- reader-heavy data under writer-grade locking
- wait/wakeup machinery invoked on paths that can never actually block

## Domain-specific review cautions (strict by default)

- Every proposal here must state the **replacement protection model**, not just the
  removal: what now guarantees the invariant the old lock/count guaranteed.
- Lock-order changes are system-wide: verify against every existing acquisition
  order, not just the edited path.
- A narrowed critical section that unlocks between check and use converts an
  invariant into a race — the burden of proof sits with the change, and "no crash in
  testing" is not proof.
- Refcount elisions must show the reference that outlives every dereference on every
  path, including error paths.
