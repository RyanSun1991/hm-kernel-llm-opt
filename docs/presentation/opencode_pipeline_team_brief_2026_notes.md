# Speaker Notes - `.opencode` Pipeline Team Brief

These notes are also embedded in `opencode_pipeline_team_brief_2026.pptx`.

## Slide 1 - Title

Open by framing this as both a usage walkthrough and an implementation review. The key message is that `.opencode` is not a single prompt; it is a staged harness with ownership, gates, validation, state, and memory.

## Slide 2 - Why It Exists

Explain why the harness exists before describing the mechanics. A single free-running agent is attractive but unreliable for kernel work. The harness turns common failure modes into structural rules: stage ownership, gates, disk-backed state, and memory.

## Slide 3 - System Context

Use this slide to separate the platform from the harness. `hmopt` gives us indexing, storage, APIs, and profiling. MCP servers are the tool bridge. `.opencode` is the agent-control layer that decides what to do and enforces the gates.

## Slide 4 - Pipeline Spine

Walk left to right. The important part is not just the order but the gates. Plan review blocks coding until the instruction-count thesis is credible. Code review blocks acceptance and decides whether tester validation is required.

## Slide 5 - Entry Routes

This is the usage model. The full pipeline is for end-to-end land-it runs. The human-in-loop route is for building durable research and planning with expert feedback. The function route is a scoped explainer that can unblock the other two.

## Slide 6 - Topology

The manager is the only hub. Specialists, reviewers, coder, and tester do not spawn other agents. This prevents hidden work, makes stage transitions inspectable, and means the manager can route failures to the right upstream owner.

## Slide 7 - State Model

This is one of the strongest implementation points. OpenCode may compact the conversation, so the manager does not rely on what it remembers. It reads `current_task.json` at the start of every turn, reloads gate rules when needed, and writes state before delegation.

## Slide 8 - Gates And Back-Edges

This slide shows why the loop is self-correcting rather than just repetitive. Every failure has one back-edge target and a cap. A build failure goes to the coder. A measured regression goes back to research because the thesis is wrong.

## Slide 9 - Validation

Make the A/B rule concrete. The tester builds and signs the feature image, flashes stock, tests stock, flashes feature, tests feature with the baseline report, then compares aggregate and per-function evidence. One side alone is not proof.

## Slide 10 - Policy And Tools

Explain the separation. Skills are reusable instructions and policy. MCP servers are the tool layer for thinking, indexing, building, flashing, testing, and Git inspection. The command file wires the right skill packs into a run.

## Slide 11 - Run Memory

The run is not hidden in chat. Each stage writes to a predictable folder. Memory is separate from artifacts: docs and bench hold run-specific detail, while target memory, subsystem memory, global lessons, human decisions, and idea ledgers carry reusable information forward.

## Slide 12 - How To Run

Show the command as the user's mental model. The profile chooses specialist hints and skill packs. The target names the subsystem or file. `Auto-Iterate` controls how many clean passes the manager should run on the same target.

## Slide 13 - Implementation Map

End the implementation section by showing the folder map. The important point for the team is inspectability. We can review the manager prompt, gates, skills, current state, plans, reviews, validation reports, memory, and patches as normal files.

## Slide 14 - Takeaways

Close by reinforcing the core takeaway: the system is designed to make AI optimization reviewable, measurable, and recoverable. A good next step for the team is a live demo that launches one command and then inspects the state file and artifacts.
