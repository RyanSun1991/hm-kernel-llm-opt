# HMOPT Platform Overview & Quickstart (Multi-Agent + Memory + Index)

Audience: every team member — start here if you are new to the platform.
Goal: understand the architecture → deploy in ~30 minutes → start using in 5 minutes.
Deep dives: each section ends with pointers to the detailed docs.

---

## 1. What this system is (three sentences)

1. A **multi-role AI workbench** built on **OpenCode**: you talk directly to 7 generic
   roles (research / design / implement / review / validate…) to solve kernel problems,
   and **you decide who works next** — or run the fully automated optimization pipeline
   with one command.
2. A **team memory system**: lessons from everyday conversations (pitfalls,
   conclusions, recipes) get captured on the spot and — with your confirmation —
   sedimented into the team knowledge hub, so the next person's question recalls them
   automatically.
3. A set of **index & tool services** (MCP): kernel code index, build, on-device
   testing, flashing — the "hands and eyes" agents use to do real work.

## 2. The architecture in one picture

```
              You (team member, chatting in OpenCode)
                          │
        ┌─────────────────┴──────────────────┐
        │      Workbench (.opencode/)        │
        │  7 generic roles: assistant (entry)│
        │  researcher·architect·implementer  │
        │  reviewer·validator·coordinator    │
        │  + skill library (loaded on demand)│
        │  + task workspaces (task/capsule)  │
        └───┬──────────────┬──────────────┬──┘
            │ tools         │ memory       │ pipeline mode
            ▼              ▼              ▼
   ┌────────────────┐ ┌──────────────┐ ┌─────────────────┐
   │ MCP services   │ │ Memory tiers │ │ /optimize_* cmds │
   │ code index 7332│ │ journal      │ │ coordinator runs │
   │ git 7334       │ │  (personal)  │ │ the same roles   │
   │ build 7335     │ │   ↓ sediment │ │ through mandatory│
   │ device test    │ │ hub knowledge│ │ stage gates      │
   │   7336         │ │  (team)      │ └─────────────────┘
   │ flash 7337     │ │   ↓ curation │
   │ skill-hub 7338 │ │ hub skills   │
   └───────┬────────┘ └──────────────┘
           ▼
   Kernel repo (hm-verif-kernel) + Windows relay + real devices
```

One-line flow: **you ask → the role recalls team experience + queries the code index →
does the work and produces artifacts → valuable lessons go into your personal journal →
with your confirmation they sediment into the team hub → which feeds everyone's next
question.**

## 3. The three building blocks

### 3.1 The multi-agent workbench (the default way to work)

**Problem it solves**: the old fully-automated pipeline left users no control, and
agents were welded to optimization — useless for anything else. Now roles are generic
and you drive.

Key points:

- **Talk to `assistant` by default.** Simple questions get answered directly; for
  complex tasks it proposes "open a task with researcher" and **waits for your OK**;
- **7 roles, one responsibility each**: researcher establishes facts (source edits
  hard-denied) → architect develops options → implementer edits code (every edit asks
  you) → reviewer challenges independently → validator verifies on real hardware. Each
  turn ends with "suggested next steps + a forwardable brief" — **forwarding is your
  call**;
- **Skills load on demand**: domain knowledge (reclaim / hyperhold / optimization
  methods) lives in skill packs; roles match your task, suggest ≤3 with reasons, and
  load only after you confirm — context stays lean;
- **Profiles are one-step entry**: `@reclaim-investigator` etc. — 6 preloaded
  role+domain compositions;
- **Task workspaces**: one directory per task (goal / progress capsule / artifacts /
  decisions) — switching roles or reconnecting never loses state;
- **The old pipeline still works**: `/optimize_workqueue` etc. run the full automated
  flow under coordinator.

> More: `docs/Agent_Workbench_Usage_EN.md` (usage) · `docs/Agent_Workbench_Design_EN.md` (design)

### 3.2 The memory system (how experience flows)

**Problem it solves**: one person's pitfall gets re-hit by the next; good recipes get
lost in chat history.

Three tiers, promoted step by step:

```
journal (personal, unreviewed) ──sediment (you confirm)──▶ hub knowledge (team, curated) ──evals──▶ hub skills (methods)
```

Key points:

- **Auto-capture**: when one of 6 salience signals appears (verified conclusion / your
  verdict / structural fact / pitfall root cause / working recipe / correction to old
  knowledge), the LLM calls `memory_log` into your personal journal automatically;
  unsure, it asks "log this?"; if you say **"记一下 / log it"**, logging is mandatory;
- **Auto-recall**: before proposing, roles call `memory_recall`; results carry layered
  attribution (`journal·unreviewed` / `hub·curated`) and are cited by ID;
- **Entering the team hub is gated**: at session close you confirm, then
  `skillhub_sediment` → hub staging → PR → 5 CI gates → human curation → official
  knowledge (F/H/A/B IDs). **Nothing reaches the team hub without your confirmation**;
- **Members who never run the pipeline can join too**: any repo, two steps — one MCP
  URL + one CLAUDE.md snippet (see §5.3).

> More: `docs/Team_Memory_Onboarding_CN.md` (onboarding) · `docs/Team_Memory_Design_CN.md` (design) · `docs/Skill_Hub_Runbook_EN.md` (hub ops)

### 3.3 Index & tool services (the agents' hands and eyes)

**Problem it solves**: agents analyzing a kernel from thin air are unreliable — they
need real code queries, real builds, real devices.

| Service | Port | What it does |
|---|---|---|
| Code index MCP | 7332 (7331 in-container) | clangd + vector index: symbols, call chains, hotspot context |
| Sequential Thinking | 7333 | step-by-step reasoning aid |
| Git MCP | 7334 | repository operations |
| Build MCP | 7335 | trigger kernel builds / signing |
| Device test MCP | 7336 | instruction-count tests, **lmbench full-suite A/B** (via Windows relay) |
| Flash MCP | 7337 | flash stock/feature images (via Windows relay) |
| Skill-Hub MCP | 7338 | memory verbs (log/recall/sediment) + hub read/write |
| REST API | 8001 | /runs, /metrics, /report queries |

> More: `docs/Kernel_Index_MCP_Onboarding_zh.md` · `docs/OpenCode_MCP_Integration_Guide.md`

## 4. Quick deployment (~30 minutes, one Linux server)

Prerequisites: Docker + docker-compose; a reachable LLM gateway; a kernel repo checkout.

```bash
# 1) Get the code
git clone <this repo> && cd hm-kernel-llm-opt && git checkout opencode

# 2) Configure (only 3 required values)
cp .env.example .env
#   HMOPT_LLM_BASE_URL=http://<your LLM gateway>:<port>/v1   ← required
#   HMOPT_LLM_API_KEY=<key>                                  ← required
#   KERNEL_REPO_PATH=/path/to/hm-verif-kernel                ← required
#   (device lane only: HMOPT_FLASH_RELAY_URL=http://<windows-relay>:9100 + image dirs)

# 3) Start services
docker compose up -d
docker compose ps          # expect hmopt / git-mcp / build-mcp / skillhub-mcp all Up

# 4) Build the code index (first run is slow; incremental afterwards)
docker exec -it hmopt hmopt index-kernel \
  --repo-path /workspace/kernel --compile-commands-dir /workspace/kernel

# 5) Verify
curl -s localhost:8001/health          # REST alive
curl -s localhost:7338/mcp -o /dev/null -w "%{http_code}\n"   # skill-hub MCP reachable
docker exec -it hmopt hmopt query "who calls shrink_node" --mode code   # index answers
```

Common issues: LLM gateway unreachable → check BASE_URL is reachable *from inside the
container* (`host.docker.internal` or an internal IP); "Docstore empty" on queries →
step 4 hasn't finished.

## 5. Quick start (5 minutes)

### 5.1 Your first session (workbench mode, recommended)

Open OpenCode in the kernel repo and just talk (assistant is the default):

```
You: what does sc->priority do in shrink_node?
assistant: (direct explanation with file:line)      ← simple question, done

You: investigate the suspected race in shrink_node thoroughly
assistant: worth opening a task — I'd hand it to researcher, brief drafted. Send?
You: send
researcher: workspace opened. Suggest loading domain-reclaim + method-lifetime (reasons…). OK?
You: ok
researcher: (queries index, reads code…) Conclusion: … (evidence: vmscan.c:137-155)
            Next: 1. consult reviewer to challenge this  2. handoff architect for options
You: 1                                              ← you are always the router
```

Whenever a conclusion is worth keeping, just say **"log it"** → it goes to your
journal. At session close the LLM inventories candidates ("2 worth sedimenting —
confirm?") → confirmed entries enter the team-hub flow.

### 5.2 Run the automated optimization pipeline (unchanged)

```
/optimize_workqueue    # coordinator drives: research → plan review → implement → code review → device A/B
```

### 5.3 Onboard a new member to team memory (no repo clone, two steps)

In any repo of your own:

```jsonc
// ① add one MCP entry to .mcp.json / opencode.json
{ "mcpServers": { "skill-hub": { "type": "http", "url": "http://<server>:7338/mcp" } } }
```

```markdown
② paste the snippet into your CLAUDE.md / AGENTS.md (full version in Team_Memory_Onboarding_CN.md):
   essence = recall before proposing; memory_log on the 6 signals; sediment at close
   after confirmation; secrets never enter the journal
```

From then on your everyday conversations recall team experience and capture your own.

### 5.4 Cheat sheet

| Want to… | Do this |
|---|---|
| Switch roles | Tab or `@researcher`, or accept the last turn's suggestion |
| One-step domain entry | `@reclaim-investigator` `@hyperhold-io` `@workqueue` `@sync-mechanism` `@kernel-understand` `@bug-fix` |
| Add/remove a skill | "load memory-tlb" / "drop the IC skill" |
| Log a lesson | say "log it / 记一下" |
| Query team experience | roles recall automatically; or ask "check the hub about X" |
| Sediment to the team hub | confirm at session close (or say "sediment") |
| Resume a task | "continue <task-slug>" (workspaces persist) |
| Automated optimization | `/optimize_generic|workqueue|hyperhold|memmgr_reclaim` |

## 6. Going deeper (read as needed)

| Topic | Doc |
|---|---|
| Full workbench usage (scenarios, fork, reviews) | `Agent_Workbench_Usage_EN.md` |
| Workbench design (roles/skills/permissions/status) | `Agent_Workbench_Design_EN.md` |
| Memory onboarding & entry format | `Team_Memory_Onboarding_CN.md` |
| Skill Hub curation & release ops | `Skill_Hub_Runbook_EN.md` |
| Pipeline-lane spec (gates/handoffs) | `.opencode/docs/harness_engineer_system.md` |
| lmbench device A/B protocol | `.opencode/skills/scenario/kernel-opt/ab-test-comparison-lmbench/SKILL.md` |
