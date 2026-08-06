---
name: bug-fix
description: >-
  Scenario pack for correctness work — something is wrong (crash, hang, corruption,
  wrong result, regression) and must be made right. Repro-first discipline, root-cause
  before fix, minimal change with a regression guard. The objective is correctness
  only; optimization framing is out of scope.
---

# Scenario Pack — bug fix (correctness only)

A bug fix is an argument in three parts: this is the mechanism of the failure, this
change removes that mechanism, and this check keeps it removed. Skipping part one
produces symptom patches; skipping part three produces regressions.

## Method

1. **Reproduce or pin the trigger.** Best: a repro command/test. When live repro is
   impossible (device-only, timing-dependent), pin the exact trigger conditions from
   evidence — trace, log, crash dump — and say which conditions remain unconfirmed.
   No repro and no pinned trigger → you are not fixing yet, you are still
   investigating.
2. **Root-cause to the mechanism.** Walk from symptom to cause until the answer is a
   code-level mechanism ("the completion path reads a field the submit path
   publishes after the doorbell"), not a restatement ("it crashes because the
   pointer is NULL"). Ask "why" until the next why would leave the code.
3. **Enumerate what shares the mechanism.** The same wrong pattern usually exists at
   sibling sites — fix the class where cheap, or record the survey result: the other
   sites and why they are (not) affected.
4. **Design the minimal fix at the mechanism.** The fix targets the root cause, not
   the crash site. If the honest fix is structural and large, say so and offer the
   choice: interim guard now (documented as such) + structural fix planned, or
   structural fix directly.
5. **Guard against regression.** A test that fails before and passes after is the
   gold standard. Where untestable, state why, and what manual verification was done
   instead — silence is not an option.
6. **Verify beyond the repro**: the repro passes, adjacent behavior is unchanged, and
   kernel-specific regression axes are re-checked — locking order, error paths,
   lifetime/refcounts, concurrent access windows around the changed code.

## Hard rules

- **Correctness only.** No performance work rides along with a bug fix — not "while
  we're here", not in the same diff. If the investigation reveals an optimization
  opportunity, record it separately for the kernel-opt lane.
- No refactors piggybacking on the fix; the diff is exactly the fix + its guard.
- A fix whose failure mechanism you cannot state is a guess — label it as one and do
  not present a guess as a fix.
- Fixes follow the normal review chain (the fix claims `ready-to-land` only after
  independent review + passing build — agent-core §6); an urgent bug does not skip
  review, it prioritizes it.

## Artifacts

- `artifacts/diagnosis.md` — symptom, trigger, mechanism (with `file:line`), shared
  sites survey; written BEFORE the fix.
- The fix diff + `artifacts/fix-note.md` — mechanism removed, guard added,
  verification performed, residual risk.
