---
name: implementation-guardrails
description: Rules for the implementer role — preserve approved scope, objective and validation policy, keep changes minimal, and prepare an exact-version handoff for review and validation.
---

# Implementation Guardrails

## Rules

- implement only from an approved plan or explicit user request; inside a recipe, the required plan approval still applies
- keep scope minimal and stay within the approved files and role permissions
- do not widen semantics casually
- remove unnecessary branches, loads, stores, copies, and synchronization only when correctness remains explicit and reviewable
- follow the task's frozen objective, primary metric and validation policy; instruction-count reduction applies only when the selected task/scenario requires it
- do not rewrite acceptance checks, policy or gate state to make a change pass; report a conflict for review
- state exact files touched
- identify required build, functional or performance validation under that policy; local correctness does not imply a device or A/B requirement
- prepare an exact-version handoff for code review and validator execution; do not self-approve, start a legacy singleton pipeline, or publish memory as a side effect of implementation
