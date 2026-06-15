---
name: hub-bridge
description: Connects the local .opencode pipeline to the team Skill Hub via MCP tools. Read path — `skillhub_resolve` mounts team knowledge + skills into research and plan-review. Write path — `skillhub_sediment` distills a finished run into hub candidates at the decision stage. Exposed by the platform's skill-hub MCP server (Docker), so agents in a kernel repo use it without the `hmopt` CLI.
---

# Hub Bridge — local pipeline ⇄ team Skill Hub

The team Skill Hub (`hm-skill-hub`) is the cross-member knowledge base: validated
facts, reusable heuristics, and rejected bad-plans — versioned and quality-gated.
This skill is the single source of truth for **how the `.opencode` pipeline reads
from the hub and writes back to it**. Everything else (agents, commands, other
skills) only references this file.

## How agents reach the hub — MCP tools (not the `hmopt` CLI)

OpenCode agents run inside a **kernel repo** and reach this platform only through
MCP — exactly like the kernel-index / build / auto-test servers. `hmopt` is **not**
on their PATH. The hub bridge is therefore exposed as **MCP tools** by the
platform's skill-hub MCP server (`src/hmopt/api/skillhub_mcp_server.py`, port 7338,
started in Docker as `hmopt-skillhub-mcp`):

| MCP tool | Purpose |
|---|---|
| `skillhub_resolve(target, stage, [opencode_dir], [mechanism])` | **READ** — returns a `## Hub context` block (skills + curated knowledge + bad-plans) |
| `skillhub_sediment([opencode_dir], [contributor], [bundle])`   | **WRITE** — distill `.opencode/memory` → hub candidates + `_bundle.jsonl` |
| `skillhub_status([opencode_dir])`                              | pinned hub version + reachability |

The **MCP server** (not the agent) does all `.opencode/` file I/O — it reads
`<opencode>/memory`, writes `<opencode>/state/retrieval.jsonl`, and writes the
sediment bundle, via the volume-mounted kernel repo. So the filesystem sandbox on
sub-agents (`glob: "**/.opencode/**": deny`) does **not** apply to the hub: **any
mcp-enabled agent may call these tools.** `opencode_dir` defaults to the server's
`HMOPT_SKILLHUB_OPENCODE_DIR`, so the usual call is just
`skillhub_resolve("<target>", "<stage>")`.

**Who calls what** (the manager orchestrates *when*; the tool is the same):
- **Read** — the researcher and plan-reviewer call `skillhub_resolve` at their
  stage; equivalently the manager calls it and injects the returned `## Hub context`
  block into the handoff. Primary `kernel-research` calls it and folds the block into
  its design doc.
- **Write** — the manager calls `skillhub_sediment` at the decision stage.

**MCP wiring (one-time):** register the skill-hub server in the kernel repo's
OpenCode MCP config (e.g. `http://<host>:7338/mcp`) next to the kernel-index / build
/ test servers. Pass the raw target (e.g. `mm/vmscan.c::shrink_node`) — the tool
slugifies internally; never pass the pipeline's `_`-slug.

**CLI fallback (co-located only):** when running *inside the platform repo* with
`hmopt` installed (local dev / CI), the equivalent commands are
`hmopt resolve "<target>" --stage <s> --local-memory .opencode/memory --run-dir .opencode/state`
and `hmopt sediment-opencode --opencode-dir .opencode --contributor <m> --bundle`.

## Hub reachability (one-time, per repo)

The MCP server discovers the hub via `find_hub_root`: it checks the kernel repo's
`.opencode/hub`, then `<repo>/hm-skill-hub`, then the platform install's
`hm-skill-hub`, then `HMOPT_SKILLHUB_HUB_ROOT`. In a kernel repo, make the hub
reachable as `.opencode/hub` (`git submodule add <hub-url> .opencode/hub`, pinned by
`.opencode/skill-memory.lock`). If no hub is found, every tool returns a
`hub: unavailable` note and the pipeline continues — the hub is an enhancement,
never a gate.

## READ path — `skillhub_resolve` (Phase A)

**When:** at the start of `research` and `plan_review` (optionally `implement` /
`code-review` for ABI/constraint context). **Call:**

```
skillhub_resolve(target="<raw target>", stage="research|plan-review|implement|code-review")
```

It overlays `<opencode>/memory` (local in-flight notes; hub wins on a shared id,
curated L2 outranks local L1), appends an audit line to
`<opencode>/state/retrieval.jsonl`, and returns this block — **paste it into the
living design doc (primary agents) or the handoff packet's `## Hub context` section
(manager → sub-agent):**

```
## Hub context (resolved <UTC> @ hub <version>, stage=<stage>)
Skills (read-only guidance): <skill refs, e.g. core/…, technique/…, domain/…>
Team knowledge (cite by id; do NOT re-derive; dedup against these):
- [<id>·<maturity>·<kind>] <title>            # fact / heuristic to reuse
- [<id>·bad_plan] <title> — DO NOT propose    # rejected mechanism
Audit: <opencode>/state/retrieval.jsonl (returned_ids=[…])
```

Downstream agents MUST treat listed ids as known: cite them, dedup against them, and
never re-propose a `bad_plan` id. In the funnel this is **dedup source (8)** — see
`optimization-funnel/SKILL.md`.

## WRITE path — `skillhub_sediment` (Phase B)

**When:** at the `decision` stage, immediately after `memory-accumulation` has
written local memory, and ONLY for a completed clean pass (never on a back-edge
bounce). **Call:**

```
skillhub_sediment(contributor="<member>", bundle=true)
```

It distills `<opencode>/memory` (+ tier-0 reviews/bench/state) into schema-valid
candidates under `<opencode>/local/sediment_staging/` and a `_bundle.jsonl`, and
returns a summary. Non-blocking: 0 candidates is honest and never gates the run.

**Human PR gate** — the member decides what to share; do NOT auto-push. The tool
returns this instruction; surface it to the user:

```
cp <opencode>/local/sediment_staging/_bundle.jsonl \
   <hub>/staging/<member>/<date>.jsonl      # then: git commit + open a PR
```

For the harvest to find candidates, `memory-accumulation` MUST write the formats it
reads: idea-ledger `### L00x` rows with `- **status**: landed` + `- **delta_pct**`;
`## Known Bad Plans` / `## Stable Structural Facts` sections in target memory; review
`## Decision reject`; bench `verdict: pass` + `delta_pct`.

## LOOP — closing it (Phase C)

After a member's PR merges into the hub, a maintainer runs the hub's Phase-4 tools
(`nightly` → `release` → `broadcast`): bump semver, then update each consumer repo's
`.opencode/hub` (submodule bump) and `.opencode/skill-memory.lock` (`hub_version` +
`pin`). The next run's `skillhub_resolve` then surfaces the newly-merged knowledge —
your finding appears in a teammate's `## Hub context`. That is the closed loop.

## State (audit) — `current_task.json`

The manager records hub I/O under a `hub` key for audit/resume:

```json
"hub": {
  "version": "<from skillhub_status / skill-memory.lock, or 'unavailable'>",
  "read": {"research": ["F001", "H001"], "plan_review": ["B001"]},
  "bundle_path": "<opencode>/local/sediment_staging/_bundle.jsonl"
}
```

## Failure / degradation rules (non-negotiable)

- skill-hub MCP unreachable, hub not found, or any tool error → the tool returns a
  `hub: unavailable …` string; log one line, set `hub: "unavailable"`, and continue.
  **The pipeline NEVER blocks on a hub call.**
- `skillhub_resolve` returns empty knowledge → fine; the target has no hub coverage
  yet.
- Always preserve the audit trail (`retrieval.jsonl`, `_bundle.jsonl`) so a human can
  verify exactly what was read and written.
