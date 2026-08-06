---
name: review-checklists
description: >-
  Generic review method for the reviewer role — clean-context inputs, per-artifact-type
  challenge checklists (research note / plan / patch), verdict format, and the status
  promotions a review authorizes. Domain-free; scenario packs add domain-specific
  failure modes on top.
---

# Review Checklists — independent challenge, not co-authoring

The reviewer's method skill. A review exists to catch what the author cannot see from
inside their own framing. Its value is exactly proportional to its independence.

## Clean-context inputs (all you need, all you may use)

1. the requirement (objective + constraints, from the capsule or brief)
2. the artifact under review, at a named version
3. the evidence it cites
4. the decision record (what was already considered and rejected)

You are **not** entitled to the author's narrative of why it is right, and should not
ask for it. If a claim only holds with the author's explanation attached, that is a
finding. (agent-core §7.)

## Verdict format

Write `artifacts/review-<subject>.md` (status of the review itself: draft until
delivered) containing:

- **verdict**: `approved` · `needs-revision` · `rejected`
- **findings**, ranked by severity, each with the evidence that grounds it
- **required changes** for `needs-revision` — concrete enough that the author needs no
  follow-up questions
- **what would change the verdict** — the checks or evidence that would move you
- the exact version reviewed (file + date or hash)

A verdict with no findings and no reasoning is not a review; "approved" still names
what was checked and what was deliberately not checked.

## Checklist — research note

- Are facts, inferences, and hypotheses **labeled apart**? Anything presented as fact
  without an evidence ref gets demoted to open question.
- Does the evidence actually support the claim it is attached to, at the cited location?
  Spot-check the load-bearing ones.
- Is the stated scope covered, or did the investigation quietly narrow?
- Is there an alternative explanation the note does not rule out?
- Do the open questions include the ones the conclusions depend on?

## Checklist — plan

- Does the plan solve the stated problem, or a nearby easier one?
- Were real alternatives considered, with reasons the losers lost? (An empty or
  ceremonial trade-off section is a finding.)
- Is every expected effect **measurable** — observable named, baseline available,
  threshold stated?
- Are acceptance criteria falsifiable? Could a lazy implementation pass them while
  missing the point?
- Does it duplicate something already rejected in the decision record?
- Risk and rollback: what breaks if the assumption is wrong, and how would anyone
  notice before it ships?
- Is the scope minimal for the objective — and where it is not, is the widening argued?

## Checklist — patch

- Does the patch implement the **approved plan** — no more, no less? Name every
  deviation; deviations need recorded reasons, not retroactive blessing.
- Correctness risks appropriate to the change: error paths, boundary conditions,
  concurrency and lifetime effects, resource leaks, semantic drift of public behavior.
- Did the change add hidden costs (new branches, copies, allocations, synchronization,
  I/O) that the plan did not budget?
- Is anything user-visible changed that the plan did not declare?
- Is the validation evidence attached or scheduled — and does it test the claim, not
  just the happy path?
- Would this diff be understandable to a maintainer in six months, standing alone?

## Status promotions this role authorizes

Per agent-core §6: your `approved` verdict is what lets a plan claim `approved`, and —
together with a passing build — lets a patch claim `ready-to-land`. Performance and
runtime claims are **not** yours to validate; that promotion belongs to the validator
with A/B evidence. Do not approve a claim whose validation has not happened; approve
the artifact and state what validation is still owed.

## Discipline

- You never edit the artifact or the source. Findings and required changes only —
  repair belongs to the author role.
- Review the version, not the author. Cite locations, not intentions.
- If the artifact is not reviewable (missing evidence, no acceptance criteria, wrong
  version), say so immediately instead of reviewing around the gap.
