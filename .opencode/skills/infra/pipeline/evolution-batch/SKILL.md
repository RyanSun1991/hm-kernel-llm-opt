---
name: evolution-batch
description: Execute explicitly selected approved candidate IDs serially through isolated worktrees and existing gates.
---

# Durable Evolution batches

Only coordinator delegates. This explicit recipe preserves every role permission
and the existing evolution-execution gates. Default assistant routing is unchanged.

1. Require a named profile, one to ten explicit candidate IDs and scope
   full|stage|plan|implement|review|validate. Default full only when the command
   explicitly requests batch execution. Never select all queue entries implicitly.
   Read dossiers, approval receipts and current stages before create.
2. Call `evolution_create_batch` with a stable request ID. Save batch ID and the
   archived method digest in the conversation resume brief. Read its archived
   Skill contents via `evolution_evidence` and use them for this batch.
3. Use a stable worker ID (current coordinator session ID). Call
   `evolution_batch_next`. It reserves one item and prepares a detached worktree;
   no source implementation occurs until the existing plan review gate passes.
   Every target code/Git/build operation must use returned handoff.repo_path.
   Workbench role/Skill/workspace paths remain in the configured Workbench root.
4. Delegate through existing evolution-execution roles and task-local capsules,
   bounded by the selected scope. Pass batch ID, candidate ID, source/workbench
   roots, approval IDs and frozen method digest to child briefs. No parallel child
   execution for the same candidate. These batches intentionally execute serially.
5. A repeated next returns the same running claim. Inspect/resume its current
   child sessions before continuing; never launch a second copy on timeout.
   On a new session, inspect the saved worker and verify old children have stopped
   before resuming its stable worker ID. IDs are trusted labels, not authentication.
6. After children stop, call next again. Completion is derived from accepted
   service stages; full requires validated. Rejection or inconclusive/failing
   validation is blocked, never success. Pending stale inputs are blocked.
7. On tool/model/build failure, retain task evidence and call evolution_batch_block
   with fresh batch version, current worker ID and concrete reason ONLY after all
   child execution has stopped. It acknowledges a stop; it cannot stop a process.
   Then next may continue another independent item. No automatic timeout release.
8. Report each candidate ID, approval ID, achieved stage, worktree, evidence and
   block reason. Do not automatically merge, push, promote Skills or notify experts.
   Worktrees remain for review; cleanup is a separate explicit operator action.

For a blocked retry, resolve its cause, inspect for dirty/unsubmitted work, then
explicitly create a new batch selecting that current candidate. Preserve its
existing worktree and approvals; never discard edits to make a retry pass.
