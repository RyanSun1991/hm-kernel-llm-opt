---
name: research-discipline
description: Non-negotiable research order — Sequential Thinking, Kernel Index, local files, design doc, structural audit across five dimensions, then IC hypothesis. No optimization before the model is stable.
---

# Research Discipline

## Non-Negotiable Order

1. Sequential Thinking MCP
2. Kernel Index MCP
3. local file reading
4. design doc update — structural model first (entry points, data flow, ownership, locks, hot-path layering)
5. **Structural Audit — mandatory before any instruction-count hypothesis.** Survey all five dimensions; each yields either a candidate mechanism or an explicit `none observed — <reason>`. Both go into the design doc's Structural Audit section.
   a. **Cross-call-site patterns** — enumerate all callers of the hot function(s) via `kernel_call_chain`. Do ≥2 callers share pre/post work that could be hoisted into the callee or shared via a helper? Do they all repeat the same lookup, the same lock dance, the same allocation?
   b. **Indirection cost** — any layer (wrapper, vtable, conditional dispatch, function pointer, generic helper) in the hot path whose flexibility is unused in the current product configuration? Is the cost of the indirection (extra branch, load, stack frame) recovered by any caller that exercises the flexibility?
   c. **Data round-trip / coalescing** — does data cross a subsystem boundary more than once per request? Are repeated lookups, repeated lock acquisitions, repeated allocations, or repeated serialize/deserialize cycles coalescable into one?
   d. **Dead / vestigial policy** — any knob, sysctl, config branch, or compatibility shim present only for a use case retired, deprecated, or not configured in the current product? Confirm via grep for callers/setters and product config files.
   e. **State / lock granularity** — any state distinction with no observable behavioral consequence in current callers? Any lock that protects fields touched by disjoint call paths and could be split into per-field or per-path locks?
6. **Hub consult — before the hypothesis.** Consult the team Skill Hub for prior work on this target (the `## Hub context` block the manager injected into your handoff, or your own `skillhub_resolve` MCP call per `.opencode/skills/infra/hub-bridge/SKILL.md`). Record which prior facts/heuristics you are building on and which `bad_plan` ids you must avoid. If the hub is unavailable, note it and proceed.
7. instruction-count hypothesis update — informed by the structural audit, the hub context, and hot-path micro analysis
8. optimization only after the model is stable

## Minimum Questions

- what are the entry points
- what data is protected
- what ownership or lifecycle boundaries exist
- what cross-file dependencies matter
- what is likely hot versus incidental
- where instruction count is likely being spent
- which repeated work, branches, loads/stores, synchronization, or copying dominate the hot path
- what proof artifact can later validate an instruction-count win
