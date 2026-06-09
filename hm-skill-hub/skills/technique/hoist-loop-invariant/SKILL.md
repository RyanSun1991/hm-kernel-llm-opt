---
name: hoist-loop-invariant
kind: technique
version: 0.1.0
maturity: L0
optimization_goal: instruction-count
requires: []
owners: ["@maintainers"]
status: experimental
---

# hoist-loop-invariant (technique skill scaffold)

Mechanism-named, topology-independent optimization "move": lift a computation
that is invariant across a loop's iterations out to the outermost stable scope,
then re-measure. Canonical mechanism `hoist-invariant` (see
`_registry/mechanisms.yaml`).

Phase-1 scaffold so the resolver can resolve `domain/mm-reclaim → requires →
technique/hoist-loop-invariant`. The concrete heuristic and target evidence live
in `knowledge/` (e.g. H001 generalizes the per-target fact F001).

## When to use

A hot loop re-computes or re-reads a value that does not change across
iterations (loop-invariant), and the read/compute is on the measured hot path.

## How to use

1. Confirm invariance across the loop body (no writes to the source inside).
2. Hoist to the outermost stable scope; re-measure at function level.
3. Record the verdict in the target idea_ledger; promote to a fact on a
   confirmed delta.
