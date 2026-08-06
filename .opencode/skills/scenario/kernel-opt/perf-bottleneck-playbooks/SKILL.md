---
name: perf-bottleneck-playbooks
description: Bottleneck classification gate + registry. Before ideation, classify the target's dominant cost (compute / ipc / memory-tlb / io) and adopt that class's primary metric + optimization playbook, instead of assuming instruction count. The extension point for adding new performance scenarios.
---

# Performance Bottleneck Playbooks (classification gate + registry)

## Why this exists

`instruction-count-first` is the right discipline for **compute-bound** hot paths,
but it silently misleads on others. We proved it on lmbench: optimizing the
instruction count of a syscall hook while the real cost is a cross-component RPC
round-trip (fcntl `F_GETFL`, getrandom) shows **IC down, lmbench flat** — a
proxy-target mismatch. So the funnel must **classify the bottleneck first**, then
pick the metric and the playbook that match it.

This skill is the **gate** (how to classify) and the **registry** (class → metric →
playbook). It is deliberately thin: the per-class *how* lives in each playbook
skill; the shared pipeline / gates / handoff / A-B validation are NOT duplicated
here.

## The classification gate

Classify the target by what dominates its measured/suspected hot path. Use the
research note's hot-path evidence (call chain, where time/instructions go, whether
it crosses an address-space boundary, whether it touches page tables / TLB / I/O):

| Signal in the hot path | Class |
|---|---|
| Straight-line CPU work, cache-hot, no boundary crossing | `compute-bound` |
| Cross-component round-trip (syscall → server, capability/IPC, activation transfer) | `ipc-bound` |
| VMA lookup/split-merge, page-table walk, **TLB maintenance / shootdown** | `memory-tlb-bound` |
| Page faults, page-cache/writeback, filesystem + storage device | `io-bound` |

A target can be mixed (e.g. `mprotect` is memory-tlb with an `ipc-bound` leg to
`memmgr`). Pick the **dominant** class for the primary metric, and name the
secondary in the handoff so its playbook is consulted too.

## Registry

| Class | Dominant cost | Primary metric (rank + verdict by this) | IC's role | Playbook skill | Status |
|---|---|---|---|---|---|
| `compute-bound` | In-core instruction execution | **instruction count** on the hot path | primary | `instruction-count-first` | active (status quo) |
| `memory-tlb-bound` | VMA + page-table + TLB maintenance | **TLB / page-walk perf counters + measured latency** | partial — covers VMA/cap/IPC legs only; blind to TLB flush & SMP shootdown | `memory-tlb-optimization` | active |
| `ipc-bound` | Cross-component round-trips | round-trip count + **whole-path** retired instructions + measured latency | only when counted whole-path (client+transfer+server) | `ipc-roundtrip-optimization` | reserved (planned) |
| `io-bound` | Faults / writeback / storage | fault & writeback counts + measured latency/bandwidth | minimal | `io-storage-optimization` | reserved (planned) |

"Whole-path IC" = retired instructions across **every** component the operation
touches (client + transfer + server). A microkernel round-trip is thousands of
instructions; counting it whole-path re-aligns IC with latency, so eliminating a
round-trip shows up as a big IC drop AND a big latency drop. Counting only the
in-kernel hook's IC is the trap.

## How the funnel uses this (Stage 0 contract)

1. **Classify** the target into one class above (dominant) + note any secondary.
2. **Adopt that class's primary metric** for funnel ranking (step 3) and for the
   tester verdict — not a blanket "instruction count".
3. **Apply the matched playbook.** Playbooks are `@`-inlined by the command's
   `Skill packs:` (see `.opencode/CLAUDE.md` — sub-agents MUST NOT Read skill
   files at runtime). If the matched playbook is **not** in context, say so in the
   handoff (`playbook_missing: <class>`) and fall back to **whole-path IC** as a
   conservative proxy — do not silently optimize the in-kernel leg alone.
4. **Record the classification** in the research handoff and in
   `<target>_validation.md` so reviewer + tester judge against the right metric.

## Adding a new playbook (the extension point)

Reserved rows above (`ipc-bound`, `io-bound`) are wired the same way memory was.
To activate one — or add a new class:

1. Write `.opencode/skills/<class>-optimization/SKILL.md`, **thin**: scope (which
   syscalls/paths), hot-path taxonomy, **primary metric + IC's partial role**,
   ranked optimization moves, HongMeng source anchors, validation tie-in. Do NOT
   re-implement pipeline/gates/handoff/A-B — those are shared.
2. Add/flip its row in the Registry table here (set Status `active`).
3. Add it (and this skill) to the `Skill packs:` of every command whose targets
   hit that class, and to the matching pipeline's `Load First`.

Keeping every class behind this one registry is what lets the funnel stay stable
while scenarios accrue.

## Cross-references

- `optimization-funnel/SKILL.md` — Stage 0 calls this gate before ideation.
- `instruction-count-first/SKILL.md` — the `compute-bound` playbook + the default.
- `memory-tlb-optimization/SKILL.md` — the `memory-tlb-bound` playbook.
- `ab-test-comparison-lmbench/SKILL.md` — runs the A/B; this skill says which
  metric/counter the verdict reads, the lmbench skill says how to run it.
