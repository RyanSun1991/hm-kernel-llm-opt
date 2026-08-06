---
name: domain-workqueue
description: >-
  Domain pack for workqueue / thread-pool paths — enqueue/dequeue, worker loop,
  queueing structures, wakeup and dispatch behavior. Pure domain knowledge extracted
  from the legacy wq-threadpool-opt specialist: any role loads it. No process rules.
---

# Domain Pack — workqueue / thread pool

## Scope (anchors)

- workqueue implementation and its API boundaries (queue/cancel/flush entry points)
- worker loop: fetch-next, execute, idle transition
- queueing data structures (per-cpu vs global, priority handling if any)
- worker lifecycle: spawn, park/idle, wakeup, reap
- dispatch/wakeup interplay with the scheduler

## Read these before exploring from scratch

1. Existing design docs matching the target (`ls .opencode/docs/`)
2. `.opencode/memory/subsystems/` + `.opencode/memory/targets/<target>.md` as named
3. Reject ledgers: `.opencode/state/bad_plans.md` and
   `.opencode/state/wq-threadpool-opt-bad_plans.md` (this subsystem has one — prior
   rejected mechanisms are recorded there; do not re-propose them)

## The system model a claim must rest on

- **API boundary**: which operations callers actually use (queue, delayed queue,
  cancel, flush) and their frequency mix under the target workload
- **Enqueue path**: locking, list/queue insertion, wakeup decision — what runs on
  every single enqueue vs only on state transitions
- **Worker loop hot path**: the steady-state cycle (dequeue → run → check-more) and
  every instruction of bookkeeping it repeats per item
- **Queueing structures**: contention topology — what is per-cpu, what is shared,
  where cache lines bounce
- **Wakeup behavior**: when an enqueue wakes a worker, when it must not, and how
  spurious wakeups are absorbed

## Optimization-sensitive spots (where cost usually hides)

- per-item work in the worker loop that is invariant across a batch (re-checking
  flags, re-taking locks, re-reading state that cannot change)
- wakeup storms: one enqueue waking more workers than there is work
- enqueue-side lock scope wider than the insertion itself
- state distinctions (idle vs parked vs running) with no observable behavioral
  consequence in current callers
- flush/cancel machinery cost imposed on the common path that never flushes

## Domain-specific review cautions

- Wakeup ordering is correctness, not just performance: a "saved" wakeup that can
  race with a worker's sleep transition is a lost wakeup — verify the check-then-sleep
  protocol the change touches.
- Flush semantics promise completion of everything queued before the flush; batching
  or reordering enqueue-side bookkeeping can silently break that promise.
- Per-cpu structures assume affinity discipline — an optimization that lets an item
  migrate between cpus mid-lifecycle needs an explicit ownership story.
