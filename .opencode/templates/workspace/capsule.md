# Capsule: <task name>

objective: <one line>
scope: <files · commit · symbols>
constraints: [<hard limits, approvals required>]
active: <role> + [<active skills>] · mode: guided
confirmed_facts:
  - <fact> (evidence: <file:line | artifact ref>)
open_questions:
  - <what is genuinely unresolved>
decisions:
  - <what was settled, and by whom — details in decisions.md>
artifacts:
  - <artifacts/... path> (status: draft)

<!--
The capsule is the task's current projection — the ONLY thing passed on handoff /
consult, and the first thing read on resume or after compaction. The active role
updates it at the end of every turn that changed anything (agent-core §5).
Keep it bounded: facts carry evidence refs; superseded detail is pruned — the
workspace files are the durable authority.
-->
