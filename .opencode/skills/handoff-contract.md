# Handoff Contract

## Rule

No agent should hand work to the next stage without a compact handoff packet.

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
- **comparison granularity**: `compare_level` in {`total`, `process`, `thread`, `lib`, `function`} plus the target names at and above that level — `compare_process`, `compare_thread`, `compare_lib`, `compare_function`. `total` requires no names; `function` requires all four. If the plan doesn't specify, default to `total` and note it.

### Tester -> Manager Or User

- stock flash result
- feature flash result
- stock async task_id, wait time, terminal status, report_path
- feature async task_id, wait time, terminal status, report_path
- compare result: level, target names, aggregate baseline / candidate / delta / delta_pct, pairs_compared, any missing pairs
- per-pair breakdown for the cases that moved most
- **per-modified-function compare rows** (one per function touched by the patch) with baseline / candidate / delta / delta_pct when the tester was given the patch diff — see `ab-test-comparison.md` "Per-Modified-Function Comparison"
- notable stderr_tail findings (crashes, exceptions) from either phase
- remaining validation gaps
- whether the instruction-count thesis still looks plausible
- verdict: pass, fail, inconclusive, or skipped
- confidence: high, medium, or low
- recommended next route: `accept` | `kernel-code-agent` | `kernel-source-research` | `iterate` | `reject` — see `kernel-tester-agent.md` → "Recommended Next Route" for which failure maps to which agent, and `os-opt-manager.md` → "Feedback Routing Table" for the manager-side rules
