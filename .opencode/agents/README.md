# Agents — roles, profiles, and the legacy pipeline cast

Two lanes share this directory (Agent Workbench design §3.5):

```
agents/
  assistant.md      ← default entry (ordinary prompts land here; never starts a pipeline)
  researcher.md     ── the 7 generic workbench roles (mode: all, directly conversable
  architect.md         and consultable; permission ceilings in frontmatter are
  implementer.md       runtime-enforced)
  reviewer.md
  validator.md
  coordinator.md    ← mode: primary; pipeline recipes + eligible parallel work only
  profiles/         ← named role+skill compositions: thin agent files
  legacy/           ← the pre-workbench pipeline cast (hm-opt-manager + specialists).
                       Since M4, /optimize_* runs on coordinator + the roles above;
                       this cast is the fallback chain, deleted only after a real task
                       has run through both chains and the comparison is archived
```

## Picking an entry point

| You want to… | Use |
|---|---|
| Ask a question, make a small change | just talk (assistant is the default) |
| Investigate something properly | `@researcher` |
| Turn findings into a plan | `@architect` |
| Implement an accepted plan | `@implementer` |
| Get an independent verdict | consult `@reviewer` (clean context) |
| Prove or falsify a claim by execution | `@validator` |
| Run the automated optimization pipeline | `/optimize_*` (explicit recipe) |
| A preloaded domain composition | `@<profile>` from `profiles/` |

Roles contain **no domain knowledge** — domain comes from `skills/scenario/` packs,
suggested from `skills/_registry.yaml` and confirmed by you. Roles suggest; **you
route** (handoff / consult / fork are yours to trigger; see
`skills/infra/agent-core/SKILL.md`).

## Task state

Workbench tasks live in `.opencode/local/workspaces/<task-slug>/` (git-ignored;
created from `.opencode/templates/workspace/` via `scripts/new_workspace.sh`). The
capsule file is the handoff/resume carrier. The pipeline lane keeps its own state in
`.opencode/state/current_task.json` — the two do not mix; during M2–M3 the state file
remains the pipeline lane's authority and workspaces are the workbench lane's.

## Adding "a custom agent"

You almost never need a new role. Compose one: profile = base role + preloaded skills
(+ optional model/permission preferences) as a thin agent file in `profiles/`. Create
a new role only when responsibility or authority genuinely changes (the golden rule,
design §2).
