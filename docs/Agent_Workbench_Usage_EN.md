# HMOPT Agent Workbench — Usage Guide

Status: Published with the M4 wrap-up · 2026-08-05
Design: `Agent_Workbench_Design_EN.md` · Chinese counterpart: `Agent_Workbench_Usage_CN.md`

The `.opencode/` harness is now a **composable, user-controlled workbench**: you talk
to one role at a time, skills carry the domain knowledge, a lightweight workspace
carries the task state, and **you decide every transfer**. The old automated pipeline
still exists — as an explicit recipe you start on purpose.

---

## 1. Getting started — three entry paths

1. **Just ask.** Open OpenCode and type. The default agent is `assistant`: simple
   questions get a direct answer with `file:line` evidence; bigger tasks get a
   proposal — "open a workspace + bring in researcher with these skills?" — and
   nothing happens until you confirm.
2. **Pick a role.** Tab-switch or mention one: `@researcher`, `@architect`,
   `@implementer`, `@reviewer`, `@validator`. Roles are domain-free; each suggests
   matching skills from the registry (with the trigger that matched) and waits for
   your OK.
3. **Use a profile.** `@reclaim-investigator`, `@hyperhold-io`, `@workqueue`,
   `@sync-mechanism`, `@kernel-understand`, `@bug-fix` — preloaded role+skill
   compositions that skip the suggestion round and start working immediately.

## 2. The cast

| Role | What it does | Hard ceiling (runtime-enforced) |
|---|---|---|
| `assistant` | default entry; answers, small approved changes, triage to roles | edits/bash/consults all ask |
| `researcher` | builds the system model: facts vs inferences vs hypotheses, evidence-cited notes | **cannot edit source** |
| `architect` | options + trade-offs + decision records + plan with acceptance criteria | **cannot edit source** |
| `implementer` | minimal diff from an accepted plan, records deviations, never self-approves | every edit asks; destructive ops denied |
| `reviewer` | independent verdict in a clean context; findings + required changes | **cannot edit source or the artifact under review** — writes only reviews + workspaces |
| `validator` | proves/falsifies claims by build/test/benchmark/device A/B | mutating cmds ask; device ops per-action approval |
| `coordinator` | pipeline recipes + genuinely parallel work only | cannot edit source; only role that delegates |

Each ceiling is a pattern-scoped frontmatter `edit` map: a role's own artifact
directories (workspaces, and e.g. `docs/`+`memory/` for researcher, `plans/` for
architect, `reviews/` for reviewer, `bench/` for validator, `state/` for
coordinator) are writable; **everything else — source above all — is denied**, and
read-only shell commands run freely while mutating ones ask. Profiles inherit their
base role's ceiling. A denial (e.g. researcher touching a source file) is **by
design, not a bug** — the role will name who can, with a forwardable brief.

One honest caveat: device operations (flash, on-device tests) go through MCP tools,
which frontmatter permissions do not gate — their per-action approval is a contract
obligation the validator announces and follows (each device action asks, every
time), not a runtime guarantee. Verify it in the runbook below.

## 3. Skills — how loading works

- `.opencode/skills/_registry.yaml` is the only index (role/ · scenario/ · infra/).
- A role reads the registry once (metadata only), matches your brief against
  `applies_when` / `not_for`, suggests **≤3 skills with reasons**, and reads a
  skill's full text only after you confirm. At most 4 non-core skills stay active.
- Ask "why this skill?" and you get the registry trigger that matched. Say "load
  memory-tlb" / "drop the IC skill" to steer manually.
- Skills never widen permissions: an R3 skill on a read-only role degrades to
  advisory content, per-action asks, or a handoff suggestion.

## 4. Task workspaces

For anything that outlives one turn, the role opens (or you ask for) a workspace:

```
.opencode/local/workspaces/<task-slug>/     (git-ignored; template in .opencode/templates/workspace/)
  task.md        objective · scope · constraints · state
  capsule.md     the current projection — THE handoff/resume carrier
  artifacts/     research-note.md · plan.md · review-*.md · validation.md …
  decisions.md   append-only decisions + rejected alternatives
```

- `bash scripts/new_workspace.sh <slug>` creates one; `--fork <src-slug>` copies a
  branch to compare alternatives (forks never overwrite).
- The active role updates `capsule.md` every turn — that is why you can close the
  session and later say **"continue <task-slug>"**: the role reloads the capsule and
  resumes with objective, state, open questions, next step.
- Handoff and consult pass **the capsule + artifact refs, never chat history**.

## 5. Moving work around — the six verbs

| Verb | What happens |
|---|---|
| `continue` | same role keeps working |
| `add/remove skill` | method changes, role stays |
| `consult` | one-shot question to another role in a **clean context**; the conclusion returns; you keep the floor |
| `handoff` | responsibility transfers; you forward (and may edit) the brief the role drafted |
| `fork` | copy the workspace, explore an alternative in parallel |
| `recipe` | you explicitly start the automated pipeline (`/optimize_*`) |

Every role turn ends with **Next options** — 1–3 suggestions, each with verb, target
role, reason, and a directly forwardable brief. Nothing executes until you pick.

## 6. Artifact statuses — who may claim what

Artifacts carry `status: draft → reviewed → approved → validated` (+ `superseded`)
and a `produced_by:` receipt (role + skills + date). Promotions are earned:

- plan → `approved`: reviewer verdict
- patch → `ready-to-land`: approved code review **and** a passing build
- performance claim → `validated`: validator A/B evidence (baseline + candidate +
  same metric + noise floor)

Don't cite a `draft` as a conclusion; ask for the consult/validation that promotes it.

## 7. Typical flows

**Deep investigation**
`@researcher investigate the suspected race in shrink_node` → workspace opens →
skills suggested (domain-reclaim: path match; domain-sync: keyword "race") → confirm
→ research note with evidence → Next options: consult reviewer / handoff architect /
continue.

**Full change, human-routed**
researcher note → you forward the brief to `@architect` → options + trade-off table
+ plan (draft) → consult `@reviewer` (clean context) → approved → handoff
`@implementer` (edits ask) → implementation note → consult reviewer (code) →
handoff `@validator` → A/B evidence → claim `validated`.

**Explanation only** — `@kernel-understand how does hp_iotab slot reuse work?`
(zero optimization vocabulary, walkthrough with file:line).

**Bug diagnosis** — `@bug-fix intermittent hang in reclaim under pressure` →
repro/trigger pinned → mechanism diagnosis → handoff implementer for the minimal fix.

**Automated pipeline (unchanged entry)**
`/optimize_workqueue`, `/optimize_generic`, `/optimize_hyperhold`,
`/optimize_memmgr_reclaim` — since M4 these run `@coordinator` driving the same
roles (researcher/reviewer/implementer/validator) under the pipeline pack's stage
gates: plan-review GATE → code-review GATE → tester A/B. `Auto-Iterate: N` still
works. The legacy chain (`@hm-opt-manager`, `agents/legacy/`) remains the fallback
until the live comparison run is archived — invoke it with the same command body if
the new chain misbehaves, and report the divergence.

## 8. Quick reference

| Want to… | Do this |
|---|---|
| Switch roles | Tab or `@role`, or accept a Next-options handoff |
| Add/remove a skill | "load memory-tlb" / "drop the IC skill" |
| Why was a skill suggested? | "why this skill?" |
| Independent review without losing the floor | consult reviewer |
| Save and resume | automatic — reopen and say "continue <task-slug>" |
| Compare two approaches | fork |
| Run the automated optimization | `/optimize_*` |
| Legacy pipeline fallback | same command body, first line → `@hm-opt-manager @.opencode/agents/legacy/hm-opt-manager.md` |

**Do**: run reviews via consult (clean context) · let decisions land in
decisions.md · treat permission denials as the system working · promote good
compositions into profiles.
**Don't**: paste chat history between roles (capsule + artifacts only) · bypass the
implementer's edit approval · cite drafts as conclusions · ask the researcher to
edit code.

## 9. Runtime verification runbook (the remaining DoD items)

The migration's static gates are green (registry lint · pytest · golden command
contract). These design-mandated checks need a **live OpenCode session** — run them
once and archive the results:

1. **End-to-end human-routed task (M2 DoD)**: assistant → researcher → consult
   reviewer → architect → implementer → reviewer → validator on a real target, with
   capsule-only handoffs.
2. **Permission ceilings (M2/§14)**: ask `@researcher` to edit a source file —
   expect a runtime denial; same for `@reviewer`. Confirm implementer edits prompt
   for approval and device flashes ask per action. Also probe the bash allowlist
   edges: a mutating command dressed as a read-only one (`find … -delete`,
   `ls > file` redirection) should hit the `ask` gate, not slip through — if it
   slips, tighten the role's bash patterns.
3. **Restart-resume (M2 DoD)**: close the session mid-task, reopen, "continue
   <task-slug>" — the capsule must fully restore working state.
4. **Old-vs-new researcher depth (M2/§14)**: same target through `@researcher` and
   legacy `@kernel-research`; domain depth must not drop.
5. **Golden pipeline run + comparison (M4 DoD)**: one real task through
   `/optimize_generic` on the new chain AND through the legacy chain; compare
   quality/tokens/turns; archive the comparison; **only then delete
   `agents/legacy/`** (and drop the legacy references in the two-chain mapping
   notes).
6. **Trigger quality (§14)**: 10 positive + 10 negative task descriptions against
   the registry's applies_when/not_for; refine wording from the misses.
7. **Token measurement (§14/§19)**: measure base context tokens for a role session
   (registry metadata only) vs a pre-migration agent session on the same task, and
   record the delta — the design's progressive-disclosure claim must be quantified,
   not asserted.
