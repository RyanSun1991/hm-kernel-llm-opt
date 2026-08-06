---
name: implementation-guardrails
description: Rules for kernel-code-agent — implement only from an approved plan, keep scope minimal, target instruction-count reduction, and prepare a clean handoff for code review and tester validation.
---

# Implementation Guardrails

## Rules

- implement only from an approved plan or explicit user request
- keep scope minimal
- do not widen semantics casually
- remove unnecessary branches, loads, stores, copies, and synchronization only when correctness remains explicit and reviewable
- treat instruction-count reduction as the primary optimization target unless the task says otherwise
- state exact files touched
- identify required build and runtime validation
- prepare a clean handoff for code review and tester validation
