---
name: plan-funnel
description: >-
  Generic option-funnel method for the architect role — turn evidence into genuinely
  different options, score the trade-offs, record the decision, and write a plan with
  acceptance criteria and a validation path. Domain-free; for kernel performance
  optimization use scenario/kernel-opt/optimization-funnel instead.
---

# Plan Funnel — evidence to options to a defensible plan

The architect's method skill. Input: a research note or an established system model.
Output: a plan whose choices someone else can audit — options that were considered,
reasons the losers lost, and criteria that make the winner falsifiable.

## Preconditions

- A trustworthy model of the affected system exists (research note, design doc, or the
  user's confirmed description). If it does not, say so and suggest `handoff researcher`
  — planning on top of guesses produces confident nonsense.
- The objective and constraints are stated in the capsule. If they are not, extract
  them from the user first.

## The funnel

### 1. Frame

One paragraph: what must become true, what must not change, and how success will be
observed. If success cannot be observed, the task is not plannable yet — say what
measurement or evidence is missing.

### 2. Generate 3–5 genuinely different options

Different means different **mechanism or scope**, not parameter variations of one idea.
Deliberately span the range: at least one minimal/local option, at least one structural
option, and — when relevant — the "do nothing / defer" option with its cost.

### 3. Dedup against what is already known

Before scoring, check each option against:

- the workspace `decisions.md` (already rejected here, and why)
- prior decision records and team memory (`memory_recall`) for this component
- any domain bad-plan ledger a loaded scenario pack points at

Drop matches and cite what they matched. Re-proposing a documented reject without new
evidence wastes a review cycle.

### 4. Score the survivors

Per option, one row each — no essays:

| Option | Mechanism (1 line) | Expected effect + how measured | Risk | Effort | Evidence |
|---|---|---|---|---|---|

"Expected effect" must name the observable it moves. "Evidence" cites file:line or an
artifact — an option with no evidence column is a hypothesis and must be labeled one.

### 5. Recommend and record

- Pick one option (or a staged combination) and say **why it beats the runner-up** —
  the comparison is the content, not the confidence.
- Append the decision + every rejected option with its reason to `decisions.md`.
  Rejections are as valuable as the pick: they stop the next person from re-walking
  dead ends.

### 6. Write the plan artifact

`artifacts/plan.md` (status: draft) with:

- objective and constraints (from the capsule, restated in one line each)
- chosen option + trade-off table from step 4
- concrete change list: files/components touched, in what order, each step reversible
  or gated
- acceptance criteria: the checks that must pass, the observable that must move, the
  threshold that counts as success
- validation path: who validates, with what method, against what baseline
- risks and rollback: what could go wrong, how it would be noticed, how to back out
- open questions the implementer must not resolve silently

## Discipline

- You design; you do not implement. The plan names changes, it does not contain the
  patch.
- A plan the reviewer cannot falsify is not ready — every claim needs a check.
- If the user pre-selected an approach, still record one alternative and why it lost;
  "the user chose it" is a decision record, not a trade-off analysis.
- Plans go to `consult reviewer` before implementation when the change is non-trivial;
  suggest it in Next options rather than treating your own draft as approved (status
  gating, agent-core §6).
