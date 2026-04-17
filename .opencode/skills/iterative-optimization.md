# Iterative Close-Loop Optimization

This skill governs **continuous** pipeline runs — when one pass completes with a clean pass verdict, the manager automatically starts the next pass on the same target to find further instruction-count wins **on top of what the previous pass landed**.

Loading this skill into a command enables the auto-iterate behavior; the command's `Auto-Iterate:` field controls how many passes to run.

## Concept

A single pipeline pass is:

```
research → plan review → implement → code review → tester → decision
```

Without this skill, pass 1 ends at "decision". The manager stops, writes memory, and waits for the user.

With this skill, when pass N ends with verdict **pass** (or the code-review-only variant when tester is skipped) AND `auto_iterate.current_iteration < auto_iterate.max_iterations`, the manager immediately:

1. Commits the current pass's findings to memory (as usual).
2. Increments `auto_iterate.current_iteration`.
3. Treats every artifact produced so far (plans, patches, validation reports) as **landed prior context**.
4. Re-delegates to the research specialist with iteration-aware instructions.
5. Repeats until one of: max iterations hit, pipeline returns fail/inconclusive that cannot be auto-recovered, or the researcher declares no credible new idea remains.

## Critical Rule — Prior Plans Are Landed Context, Not "Already-Done" Gates

Under the default single-pass flow, a researcher who sees `.opencode/plans/<target>_plan.md` already exists might conclude "someone already did this target, nothing to do". Under the iterative flow, the same file means "iteration N-1 landed this plan and the patch is already in the tree — find a **different** win on top of it."

The researcher, plan reviewer, coder, and tester MUST treat prior-iteration artifacts as:

- **In-force code state.** The tree being benchmarked in pass N already contains every pass-<N patch. The Step 2 stock flash + Step 3 stock test of pass N naturally include those landed optimizations in the baseline.
- **De-dup context.** Mechanisms from prior iterations must NOT be re-proposed. The optimization-funnel dedup step already checks `bad_plans.md`; under iterative mode it ALSO checks every `.opencode/plans/<target>__iter*_plan.md` and the current `.opencode/plans/<target>_plan.md`.
- **Success evidence, not failure.** A landed mechanism is a positive signal — the new iteration should aim at adjacent code the previous pass did not touch.

## Artifact Slugs Across Iterations

To avoid overwriting between passes:

- iteration 1 uses `<base_slug>` (e.g., `sysmgr_pwrmgr`) — identical to the non-iterative flow for backwards compatibility.
- iteration N ≥ 2 uses `<base_slug>__iter<N>` (e.g., `sysmgr_pwrmgr__iter2`, `sysmgr_pwrmgr__iter3`).

Every artifact path derives from the current iteration's slug:

```
.opencode/docs/<slug>_design.md
.opencode/plans/<slug>_plan.md
.opencode/reviews/<slug>_plan_review.md
.opencode/reviews/<slug>_code_review.md
.opencode/patches/<slug>.patch
.opencode/bench/<slug>_validation.md
.opencode/bench/<slug>_after_patch.md
```

Memory files stay at `.opencode/memory/targets/<target>.md` etc. — they are shared across iterations.

## State — `current_task.json`

The manager maintains iteration state in `.opencode/state/current_task.json`:

```json
{
  "target": "sysmgr/pwrmgr",
  "base_slug": "sysmgr_pwrmgr",
  "artifact_slug": "sysmgr_pwrmgr__iter3",
  "auto_iterate": {
    "enabled": true,
    "max_iterations": 5,
    "current_iteration": 3,
    "iteration_history": [
      {"iteration": 1, "slug": "sysmgr_pwrmgr",           "verdict": "pass", "delta_pct": -3.1, "mechanism": "hoist-check"},
      {"iteration": 2, "slug": "sysmgr_pwrmgr__iter2",    "verdict": "pass", "delta_pct": -1.8, "mechanism": "drop-redundant-refcount"}
    ]
  }
}
```

The manager writes `current_iteration` and `artifact_slug` **before** delegating to the research specialist each pass, so downstream agents read the correct slug.

## Manager — Entering an Iteration

On every pass N:

1. Read `current_task.json`.
2. If `auto_iterate.current_iteration == 1`: use `base_slug` as `artifact_slug`. Normal flow.
3. If `auto_iterate.current_iteration >= 2`:
   a. Compute `artifact_slug = f"{base_slug}__iter{N}"` and write it back.
   b. Build an "iteration context" listing all prior slugs (from `iteration_history`).
   c. Pass that list to the research specialist in the delegation packet under a `prior_iterations` field.

## Manager — Exiting a Pass

After the decision stage produces a final verdict for pass N:

| Verdict | `auto_iterate.enabled` | Action |
|---|---|---|
| pass | true AND N < max_iterations | increment counter, start pass N+1 on same target |
| pass | true AND N == max_iterations | stop; write a summary `.opencode/bench/<base_slug>_iteration_summary.md` |
| pass | false | stop; normal single-pass end |
| fail / inconclusive (handled by back-edge routing) | — | back-edge takes over; do NOT count this as an iteration — counter is incremented ONLY on a clean pass |
| fail (stall cap hit per Q1 feedback-routing rules) | — | stop; write the stall artifact and surface to the user |

The key invariant: **`current_iteration` increments only on a successful pass**, so a failed-then-recovered cycle does not burn the iteration budget.

## Researcher — Iteration-Aware Ideation

When `current_iteration >= 2`, the research specialist (kernel-source-research or the active domain specialist) MUST:

1. Read every prior `.opencode/plans/<prior_slug>_plan.md` listed in `iteration_history`.
2. Read every prior `.opencode/bench/<prior_slug>_validation.md` to know what actually landed and how much it moved the metric.
3. Extract the prior mechanism for each — add them to the local dedup set in addition to the normal `bad_plans.md` sources.
4. Produce a **new** instruction-count hypothesis that:
   - targets code not touched by any prior iteration, OR
   - targets an orthogonal inefficiency in the same hot path (e.g., prior iteration removed a branch; this iteration removes a redundant load).
5. If after the funnel no new credible idea exists, return `no_more_ideas` to the manager. The manager then stops iteration and reports.

The researcher MUST state, in the handoff packet, which prior-iteration mechanism each of the 5 candidate ideas is orthogonal to, so the plan reviewer can audit dedup.

## Plan Reviewer — Iteration-Aware Bad-Plan Gate

The existing bad-plan gate (`kernel-plan-reviewer.md`) already cross-references `bad_plans.md` + `memory/targets/<target>.md`. Under iterative mode it ALSO cross-references every `.opencode/plans/<prior_slug>_plan.md` listed in `iteration_history`. Matching a prior-iteration landed mechanism → reject as a duplicate; the researcher must produce something genuinely new.

## Tester — Iteration-Aware Baseline

Nothing changes structurally — the tester still flashes stock + feature and compares. The "stock" image for pass N is the tree-with-iterations-1-through-N-1-landed, not a pristine tree. That is correct by construction: each iteration's gain is measured relative to the tree that carries the prior wins.

## Stop Conditions

Iteration stops (cleanly) when ANY holds:

- `current_iteration == max_iterations` after a pass
- researcher returns `no_more_ideas`
- user interrupts the session
- two consecutive passes land `inconclusive` within noise (< 1 %) — the target is saturated

Iteration stops (with a stall record) when:

- the Q1 feedback-routing stall cap is hit within a single pass
- an infrastructure SKIPPED verdict appears twice in a row

On any stop, write `.opencode/bench/<base_slug>_iteration_summary.md` with: per-iteration slug, verdict, delta_pct, mechanism, and a one-paragraph close.
