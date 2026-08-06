---
name: kernel-understand
description: >-
  Scenario pack for explanation-only kernel work — "how does X work", "what calls Y",
  "walk me through this path", onboarding to unfamiliar code. Method for building and
  presenting an explanation. Explicitly forbids optimization framing: no performance
  vocabulary, no improvement suggestions, the deliverable is understanding.
---

# Scenario Pack — kernel understanding (explanation only)

The task is done when the asker can predict the code's behavior without you. Nothing
here changes code, and nothing here evaluates code — description, not judgment.

## Method

1. **Pin the question.** "Explain X" hides a real question — behavior on a specific
   path? interaction between two parts? why-is-it-designed-this-way? Restate what
   will be answered and at what depth before reading further.
2. **Read in dependency order**, not file order:
   - entry points (syscall / callback / init) → who invokes this and when
   - data structures → what state exists, who owns it, what invariants hold
   - control flow → the main path first, then branches by trigger condition
   - concurrency → what runs in parallel, what synchronizes access
3. **Walk one concrete scenario end-to-end.** Pick a representative invocation and
   trace it with `file:line` at every hop. A trace of one real path teaches more
   than a summary of all paths.
4. **Then generalize**: the variations (error paths, config branches, edge triggers)
   as deltas from the walked path.

## Presenting the explanation

- Layered: one-paragraph summary → the walked path → the variations → open corners.
  The reader stops when satisfied.
- Every mechanism claim carries `file:line`. If you infer intent ("this exists so
  that…"), label it as inference — design intent is rarely provable from code alone.
- Use a diagram when structure carries the answer (call graph, state machine,
  ownership map); skip it when a sentence does.
- End with what was NOT covered — the honest boundary of the explanation.

## Hard prohibitions (this pack's whole point)

- No optimization vocabulary: no "hot path", "overhead", "cost", "could be faster",
  no instruction/cycle/latency framing.
- No improvement suggestions — not even parenthetical ones. If the user asks "should
  it be changed?", that is a different task: point them at the bug-fix pack
  (correctness) or the kernel-opt packs (performance) and stop.
- No judgment of code quality. Describe what is, not what ought to be.

## Artifacts

- Conversational answers for one-shot questions.
- `artifacts/explanation.md` (or a living `.opencode/docs/<slug>_design.md` when the
  user wants standing documentation) for multi-turn understanding tasks — status and
  receipt headers per agent-core §6.
