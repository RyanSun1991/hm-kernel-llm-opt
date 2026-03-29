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
- why this should reduce instruction count
- known tradeoffs and open risks
- required validation commands or MCP actions

### Code Reviewer -> Tester

- review decision
- findings to validate explicitly
- build/test/profiling requirements
- regression watch list

### Tester -> Manager Or User

- build result
- auto-test result
- artifact locations
- remaining validation gaps
- whether the instruction-count thesis still looks plausible
- pass/fail or inconclusive decision
- recommended next route
