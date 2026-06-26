---
name: handoff-contract
description: Non-negotiable handoff packet requirements for every stage transition — defines mandatory artifacts, delegation message structure, naming conventions, and receiving-agent verification rules.
---

# Handoff Contract

## Rule

No agent should hand work to the next stage without a compact handoff packet.

## Manager-Visible Return Cap — Hard Limit ≤500 tokens

When a sub-agent (researcher / plan-reviewer / coder / code-reviewer / tester) returns to `hm-opt-manager`, the chat-visible return that the manager sees in its conversation context MUST stay under ~500 tokens. This is a compaction-survival measure: the manager runs many iterations in one session, and full sub-agent outputs in chat would balloon the manager's context until OpenCode auto-compacts and key state is summarized away.

The chat-visible return MUST contain ONLY:

1. `verdict:` line — one of `pass | fail | inconclusive | approve | reject | needs_revision | skipped`
2. `artifact:` line(s) — exact path under `.opencode/{docs,plans,patches,reviews,bench}/` for every file the sub-agent produced
3. ≤3 key-fact bullets, each ≤30 tokens, each citing `file:line` from the artifact (no inline quoting of patch hunks, design prose, or review prose)
4. `next:` line — the next agent name, or `manager_decide` if the manager must route based on the artifact contents
5. `blocking:` bullets (only if non-empty, ≤2)

Forbidden in the chat-visible return — these MUST be on disk only:

- pasted patch hunks (write to `.opencode/patches/<slug>.patch`)
- full design / plan / review / validation prose (write to the corresponding `.opencode/` subdir)
- re-stated facts that already live in an artifact you just wrote
- any quoted file content the manager can re-Read from disk

The manager Reads the artifact only when its decision needs the details — not to "review" it inline. A sub-agent that returns more than ~500 tokens of chat-visible content is violating this contract; the manager should record the violation and remind the sub-agent on the next bounce.

## Minimum Handoff Packet

- target and subsystem in scope
- primary metric, normally instruction count
- hot path and evidence source
- files, functions, and structs in scope
- optimization or review hypothesis
- constraints: correctness, lock, lifetime, memory, API, logic
- unresolved questions
- required next action from the receiving agent
- required artifacts to read before acting

## Stage Contracts

### Research -> Plan Reviewer

- design doc path
- plan path
- instruction-count hypothesis
- baseline evidence and hotspot mapping
- top risks and rejected alternatives

### Plan Reviewer -> Coder

- decision: approve, revise, or reject
- must-keep semantics
- must-not-cross boundaries
- expected instruction-count mechanism
- required validation steps

### Coder -> Code Reviewer

- exact files changed
- exact hot path changed
- **modified functions**: the authoritative list of function names whose body the patch edited, to be forwarded to the tester for per-function instruction-count comparison. Mirrors the `## Modified functions` section in `.opencode/bench/after_patch.md`. Use `none` if only macros / headers / Kconfig were touched.
- why this should reduce instruction count
- known tradeoffs and open risks
- required validation commands or MCP actions

### Code Reviewer -> Tester

- review decision
- findings to validate explicitly
- build/test/profiling requirements
- regression watch list
- stock image path (baseline kernel without patches)
- feature image path (kernel with optimization patch, from Build MCP output)
- device target for flash (serial or identifier)
- modelCase test workspace override if not `D:\modelCase_OH_single`
- **comparison granularity**: `compare_level` in {`total`, `process`, `thread`, `lib`, `function`} plus the target names at and above that level — `compare_process`, `compare_thread`, `compare_lib`, `compare_function`. `total` requires no names; `function` requires all four. If the plan doesn't specify, default to `total` and note it. (Applies to `instruction-count` only.)
- **test_method** (default `lmbench-suite`): which validation the tester runs — `lmbench-suite` (full lmbench A/B; **default**) or `instruction-count` (modelCase per-function IC A/B). For `lmbench-suite` the `compare_level` above is ignored; results are per-benchmark-group + HM-vs-Linux and the verdict uses the benchmark delta with a ~2% noise floor. See `ab-test-comparison-lmbench/SKILL.md`.

### Tester -> Manager Or User

- stock flash result
- feature flash result
- stock async task_id, wait time, terminal status, report_path
- feature async task_id, wait time, terminal status, report_path
- compare result: level, target names, aggregate baseline / candidate / delta / delta_pct, pairs_compared, any missing pairs
- per-pair breakdown for the cases that moved most
- **per-modified-function compare rows** (one per function touched by the patch) with baseline / candidate / delta / delta_pct when the tester was given the patch diff — see `ab-test-comparison/SKILL.md` "Per-Modified-Function Comparison"
- notable stderr_tail findings (crashes, exceptions) from either phase
- remaining validation gaps
- whether the instruction-count thesis still looks plausible
- verdict: pass, fail, inconclusive, or skipped
- confidence: high, medium, or low
- recommended next route: `accept` | `kernel-code-agent` | `kernel-source-research` | `iterate` | `reject` — see `kernel-tester-agent.md` → "Recommended Next Route" for which failure maps to which agent, and `hm-opt-manager.md` → "Feedback Routing Table" for the manager-side rules
