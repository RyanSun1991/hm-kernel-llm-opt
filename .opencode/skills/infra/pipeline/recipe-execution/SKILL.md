---
name: recipe-execution
description: >-
  The coordinator's operational manual for running /optimize_* pipeline recipes on the
  workbench role chain — per-turn state rebuild, routing to domain packs, delegation
  targets, feedback routing on failures, iteration budget, auto-iterate close-loop,
  hub-bridge touchpoints, and the end-of-iteration anchor. Extracted in M4 from the
  legacy hm-opt-manager agent; coordinator-only.
depends_on:
  - stage-gate-enforcement
  - handoff-contract
  - delegate
---

# Recipe Execution — coordinator operational manual

This skill turns the generic coordinator role into the pipeline hub. It is loaded
only by `/optimize_*`-style recipe commands (inlined) and applies only inside them.
Gates come from `stage-gate-enforcement`; packet format from `handoff-contract`;
delegation mechanics from `delegate`. This file adds the operational protocol the
legacy `hm-opt-manager` carried in its prompt.

## The two chains (transition mapping)

The stage cast on the **new chain (default)** — and the legacy cast the older skills
and docs may still name:

| Stage | New chain (default) | Legacy chain (fallback until deletion) |
|---|---|---|
| hub / intake / decision | `coordinator` (this role) | `hm-opt-manager` |
| research | `researcher` + routed domain pack | `kernel-source-research`, `memmgr-reclaim-research`, `hyperhold-io-opt`, `basic-mechanism-sync-opt`, `wq-threadpool-opt` |
| plan review (GATE) | `reviewer` | `kernel-plan-reviewer` |
| implementation | `implementer` | `kernel-code-agent` |
| code review (GATE) | `reviewer` | `kernel-code-reviewer` |
| tester A/B validation | `validator` | `kernel-tester-agent` |

The legacy chain remains runnable by invoking `@hm-opt-manager`
(`agents/legacy/hm-opt-manager.md`) with the same command body — use it if the new
chain misbehaves on a target, and report the divergence.

## Per-turn state rebuild — mandatory at the START of every turn

OpenCode does not signal compaction. Rebuild state from disk before any decision —
every turn, not conditionally:

1. Read `.opencode/state/current_task.json`. It is authoritative over conversation
   memory for: `current_stage`, `iteration`, `auto_iterate.current_iteration`,
   `artifact_slug`, `pending_action`, `gates_passed[]`, `last_verdict`,
   `last_handoff_path`. The file wins every disagreement.
2. If `auto_iterate.current_iteration >= 2` AND `last_handoff_path` is set, Read that
   handoff file — it summarizes prior iterations and exhausted mechanisms.
3. At gate stages (`intake`, `plan_review`, `code_review`, `decision`) re-apply the
   stage-gate and handoff-contract rules from your inlined context.
4. If `target` is non-empty, Read `.opencode/memory/targets/<target>.md` (skip
   silently if absent).
5. Resume from `current_stage` per the stage table below — derived from the file,
   never from chat memory.

| current_stage | This turn's action |
|---|---|
| `intake` | Run Session Startup below; route per Routing Rules; run HUB_READ (`--stage research`) and inject `## Hub context` into the research brief; set `current_stage = research`; delegate. |
| `research` | If `pending_action.expected_artifact` exists on disk → set `current_stage = plan_review`, `pending_action.next_agent = reviewer`, run HUB_READ (`--stage plan-review`), write back, delegate the plan review. If not on disk → re-delegate the same research brief. |
| `plan_review` | Read `.opencode/reviews/<artifact_slug>_plan_review.md`; on `approve` → advance to `implementation`, append `plan_review:iter<N>` to `gates_passed`; on `needs revision` / `reject` → Feedback Routing Table. |
| `implementation` | Refuse unless `gates_passed` contains `plan_review:iter<N>`. Delegate to `implementer`. |
| `code_review` | Like `plan_review`, for `<artifact_slug>_code_review.md`; append `code_review:iter<N>` on approve. |
| `tester` | Delegate to `validator` only if the code review set tester to required/recommended. |
| `decision` | Run End-of-Iteration Anchor; update `auto_iterate.iteration_history`; after memory-accumulation writes local memory (clean pass only) run HUB_WRITE and surface the PR command; then evaluate Post-Decision Auto-Iterate. |
| `iteration_boundary` | Compute next slug, bump `current_iteration`, set `current_stage = intake` for iteration N+1, write back, delegate research for the next pass. |

**Before EVERY delegate call**: write back `current_task.json` with updated
`current_stage`, `pending_action.next_agent`, `pending_action.expected_artifact`,
and any new `gates_passed` entry.

## Session Startup (intake)

1. Acknowledge the task in one sentence.
2. Apply the session language (config.yaml + language-config, already inlined).
3. Your command has inlined: the recipe card, this skill, the pipeline pack, and the
   role/scenario packs the run needs. Do NOT re-Read `.opencode/skills/` files.
4. Read the bootstrap docs and memory files the command lists, at their exact paths.
5. **Parse `Auto-Iterate: N`** from the prompt. If N ≥ 2: set
   `auto_iterate.enabled=true`, `max_iterations=N`; compute `base_slug` from the
   target (`/`→`_`, strip separators); pass 1 → `artifact_slug = base_slug`, pass K≥2
   → `<base_slug>__iter<K>`; `ls .opencode/plans/ reviews/ bench/` once — artifacts
   matching `<base_slug>*` are prior-iteration landed context to pass to the
   researcher, never an "already done" short-circuit. No `Auto-Iterate:` → single
   pass.
6. Initialize/update `.opencode/state/current_task.json` (profile, target,
   artifact_slug, iteration state) before the first delegation.

## Delegation targets (new chain)

Delegate with `task(subagent_type=..., prompt=<full handoff packet>)` — the packet
inside the tool call, never narrated to the user (see `delegate`):

- **research** → `subagent_type="researcher"`. The brief NAMES the domain pack to
  apply (per Routing Rules below) — the pack's full text is already in the
  sub-agent's context because the command inlined it (domain commands inline their
  one pack; `optimize_generic` inlines all four, and your brief tells the researcher
  which one applies — the others are dead weight it must ignore). The brief also
  requires: bottleneck classification (Stage-0), structural audit, dedup against
  reject ledgers, design doc + plan at the slug paths, full handoff packet back.
- **plan review** → `subagent_type="reviewer"`, brief framed as PLAN review: clean
  context (plan + design doc + dedup sources + prior-iteration plans), verdict to
  `.opencode/reviews/<slug>_plan_review.md`.
- **implementation** → `subagent_type="implementer"`: approved plan + review
  conditions; minimal diff; `.opencode/bench/after_patch.md` with the
  `## Modified functions` list.
- **code review** → `subagent_type="reviewer"`, brief framed as CODE review: plan +
  patch + implementation note; verdict + tester decision (required / recommended /
  skipped-with-reason) to `.opencode/reviews/<slug>_code_review.md`; forward the
  modified-functions list verbatim.
- **tester** → `subagent_type="validator"`: stock/feature image paths, device
  target, test method + parameters, relay reference; A/B protocol per the inlined
  validation skills; report to `.opencode/bench/<slug>_validation.md`.

Sub-agents return to you; they never chain onward. If a `task()` call rejects a role
name, stop and report — do not impersonate the stage yourself.

## Routing Rules (target → domain pack for the research brief)

| Task emphasizes | Domain pack to name in the brief |
|---|---|
| memmgr · reclaim · page alloc · vmpressure · psi · palloc | `domain-reclaim` |
| hyperhold · zswap · swap io · hpio · iotab · eid · zsmalloc · compression | `domain-hyperhold-io` |
| mutex · rwlock · futex · semaphore · refcount · wait · race · contention | `domain-sync` |
| workqueue · thread pool · worker · task dispatch | `domain-workqueue` |
| anything else / ambiguous | no pack — generic researcher, research-first |

## Feedback Routing Table — mandatory for failing stages

Every failing sub-agent result triggers exactly one route, carrying prior artifacts +
failure reason + a loop-counter increment.

From **validator**:

| Failed phase | Verdict | Route back to | Why |
|---|---|---|---|
| build/sign | fail | `implementer` | patch does not compile/sign — diagnose stderr_tail, re-patch |
| feature flash failed (stock OK) | fail | `implementer` | patch breaks the boot image |
| aggregate delta regressed | fail | `researcher` | the optimization thesis is disproven — new mechanism needed |
| targeted symbol disappeared on feature side | fail | `researcher` | the plan's target assumption was wrong — re-scope |
| within noise / pairs missing | inconclusive | `researcher` if the hypothesis looks exhausted; `implementer` if the patch shape was too small | judge from the per-pair table |
| stock flash / stock test / relay infra failed | skipped | no bounce — report to the user; infra, not patch |
| 180-min ceiling | inconclusive | no bounce by default — ask the user before re-running |

From **reviewer (code review)**: `needs revision`/`reject` → `implementer` with the
full review; `reject` citing a plan-level flaw → `researcher` (then plan review
again before re-coding).

From **reviewer (plan review)**: `needs revision` → same researcher brief + review
notes, then re-review; `reject` (bad-plan gate or non-credible thesis) → researcher
for a fresh mechanism — and record the rejected mechanism in
`.opencode/state/bad_plans.md` (or the subsystem ledger) before re-delegating.

**Iteration budget** (stored in `current_task.json` → `iteration`, incremented on
every back-edge): plan-review↔research cap 3 · code-review↔code cap 3 ·
tester↔upstream cap 2. Past a cap: stop, write
`.opencode/bench/<artifact>_stall.md` summarizing every bounce + residual
hypothesis, surface to the user. Never silently loop.

## Hub Bridge touchpoints

Per the inlined `hub-bridge` skill: `skillhub_status` at session start (record
`hub.version`; unreachable → `hub: "unavailable"`, continue). `skillhub_resolve`
before delegating research and before plan review — inject the returned
`## Hub context` inside the handoff packet (dedup + bad-plan ids). At decision
(clean pass, after memory-accumulation): `skillhub_sediment(bundle=true)`, record
`hub.bundle_path`, surface the human PR command. The hub never gates a run.

## End-of-Iteration Anchor — compaction recency shield

When decision completes (before starting iteration N+1's research), emit exactly:

```
=== ITERATION N ANCHOR ===
target: <target>
profile: <profile>
base_slug: <base_slug>
current_iteration: <N>
max_iterations: <max>
landed_iterations:
  - iter1: <mechanism> Δ=<delta_pct>%
  - iter<N>: <mechanism> Δ=<delta_pct>%   [JUST LANDED]
exhausted_mechanisms:
  - <one per line, from iteration_history + bad_plans.md>
next_iteration: <N+1>
next_artifact_slug: <base_slug>__iter<N+1>
hard_gates_active: research → plan_review → code → code_review → tester → decision
state_file: .opencode/state/current_task.json
last_handoff: .opencode/state/iteration_<N>_handoff.md
=== END ANCHOR ===
```

And write `.opencode/state/iteration_<N>_handoff.md` = the anchor content plus:
`## Open hypotheses for iteration <N+1>` (2-3 directions with file:line),
`## Stop check` (consecutive_inconclusive · researcher_no_more_ideas ·
back_edge_caps_remaining), and `## Required first 3 actions` (read state file →
read this file → write back the new slug). Then set `last_handoff_path`,
`current_stage = "iteration_boundary"`.

## Post-Decision Auto-Iterate

| Condition | Action |
|---|---|
| `auto_iterate.enabled == false` | Stop; single-pass report. |
| pass verdict, `current_iteration < max` | Start the next pass automatically. |
| pass verdict, `current_iteration == max` | Stop; write `.opencode/bench/<base_slug>_iteration_summary.md`. |
| fail/inconclusive with a valid back-edge | Run the back-edge; do NOT burn iteration budget. |
| back-edge stall cap hit, or infra-SKIPPED | Stop; stall artifact; surface to user. |
| researcher returned `no_more_ideas` | Stop; iteration summary with saturation reason. |
| two consecutive passes within noise | Stop; target saturated; iteration summary. |

Starting pass K+1: append pass K's outcome
(`{"iteration": K, "slug": ..., "verdict": ..., "delta_pct": ..., "mechanism": ...}`)
to `iteration_history`; bump `current_iteration`; `artifact_slug =
<base_slug>__iter<K+1>`; write back BEFORE delegating; the research brief carries
`iteration: K+1`, the full `prior_iterations` history, and the instruction to
propose an orthogonal mechanism or return `no_more_ideas`.

Invariants: `current_iteration` increments only on clean passes · memory and
bad-plans are shared across iterations, artifacts use the per-iteration slug ·
never auto-start K+1 while a pass-K failure is under back-edge handling.

## Required outputs (unchanged artifact contract)

design docs → `.opencode/docs/<slug>_design.md` · plans →
`.opencode/plans/<slug>_plan.md` · plan reviews →
`.opencode/reviews/<slug>_plan_review.md` · code reviews →
`.opencode/reviews/<slug>_code_review.md` · validation →
`.opencode/bench/<slug>_validation.md` · patches →
`.opencode/patches/<slug>.patch`. The slug matches across stages; sub-agents read it
from `current_task.json` or the packet — they never derive their own.
