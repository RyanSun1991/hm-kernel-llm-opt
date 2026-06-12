# Team Skill Hub Runbook (English)

> For every stage: **who · when · exact commands · inputs · artifacts produced · where they flow next**.
> All commands verified. Two environment-variable conventions (set to your real paths):
>
> ```bash
> export HUB=~/work/hm-kernel-llm-opt-main/hm-skill-hub      # central hub repo
> export KOPEN=~/.../hm-verif-kernel/.opencode               # your kernel repo's .opencode
> ```

## Chain overview (stages numbered; sections below match)

```
┌─ Knowledge track (engine A · append-only) ───────────────────────────┐
│ ① daily work       ② distill          ③ contribute     ④ CI gates    │
│ .opencode/memory → _bundle.jsonl   →  hub staging/  →  lint·redact·  │
│ (+reviews/bench)   (local staging)    <you>/<date>     dedup·eval    │
│                                                          ↓           │
│ ⑥ promotion  ←───  hub knowledge/  ←───  ⑤ curation (human)          │
│ (mech ≥2 targets)  (team knowledge)      report → place → assign ids │
└──────────┬───────────────────────────────────────────────────────────┘
           ↓ suggests graduating into a technique skill
┌─ Skill track (engine B · eval gate) ────────────────────────────────┐
│ ⑦ scaffold          ⑧ write exam + register    ⑨ exam + lint + PR    │
│ promote-skill   →   suite/cases + eval_id   →  run_evals→scorecard   │
│ (L0 intern)         + best_skill.md            →lint → merge         │
└──────────┬──────────────────────────────────────────────────────────┘
           ↓
┌─ Operations & consumption ──────────────────────────────────────────┐
│ ⑩ nightly loop (maintainer)     ⑪ consume (any member, before        │
│ nightly dry-run → --apply           researching a target)            │
│ release + lock + dashboard      hmopt resolve "<path::symbol>"       │
│                                  → team facts+traps+techniques → ①   │
└─────────────────────────────────────────────────────────────────────┘
```

## Stage 0 · One-time setup

| | |
|---|---|
| Who/When | every member, once |
| Commands | `cd ~/work/hm-kernel-llm-opt-main && pip install -e ".[dev]"` |
| Check | `hmopt --help` lists `sediment-opencode / resolve / retrieval-eval / promote-skill` |
| Notes | fully offline; only `--llm-extract` needs `HMOPT_LLM_API_KEY`/`HMOPT_LLM_BASE_URL` |

---

# Knowledge track (your experience → team knowledge base)

## Stage ① · Daily work accumulates memory (no new command)

| | |
|---|---|
| Who/When | member + harness agents, working as usual |
| Action | run optimizations via the `.opencode/` manager/research entry; let verdicts land per the memory-accumulation convention |
| Artifacts | `$KOPEN/memory/idea_ledger/<target>.md` (`### L00x` verdict rows) · `memory/global_lessons.md` (`### title` + bullets) · `memory/targets|subsystems/*.md` (`## Known Bad Plans` etc.); safety-net sources: `reviews/*_review.md`, `bench/*_validation.md`, `state/*bad_plans*.md` |
| Flows to | stage ② |
| ⚠️ Format | ledger rows must be `### L001 one-liner` + `- **status**: landed` field bullets; template examples inside HTML comments are never collected |

## Stage ② · Distill (one command at close-out)

| | |
|---|---|
| Who/When | member, at task/session close-out |
| Commands | `hmopt sediment-opencode --opencode-dir "$KOPEN" --hub "$HUB" --contributor <you> --bundle` (optional `--llm-extract --config <platform-repo>/configs/app.yaml` to let an LLM distill docs/plans free text) |
| Inputs | all stage-① files |
| Artifacts | `<opencode-dir>/local/sediment_staging/opencode-<repo>.jsonl` (this batch) + **`_bundle.jsonl` (the contribution payload, one schema-valid candidate per line)** |
| Check | terminal prints `N valid candidate(s)`; on 0 it lists the scan summary and expected formats |
| Flows to | `_bundle.jsonl` → stage ③ |
| ⚠️ Gotchas | point at the whole `.opencode` dir (not the memory subdir); a non-existent path fails loudly with a hint |

## Stage ③ · Contribute (git PR)

| | |
|---|---|
| Who/When | member, after eyeballing the bundle (especially LLM-distilled entries) |
| Commands | `mkdir -p "$HUB/staging/<you>" && cp <bundle> "$HUB/staging/<you>/$(date +%F).jsonl"` → git commit, push, open PR |
| Artifact | `hm-skill-hub/staging/<you>/<date>.jsonl` (Tier-1 inbox — **not yet in the knowledge base**) |
| Flows to | the PR triggers stage ④ |

## Stage ④ · CI gates (automatic)

| | |
|---|---|
| Who/When | CI on the PR (can be pre-run locally) |
| Local pre-run | `cd "$HUB" && python tools/lint.py && python tools/redact.py --check && python tools/dedup.py staging/<you>/<date>.jsonl --check` |
| Output | per-candidate three-state verdicts: `merge` = fold provenance / `conflict` = same condition, opposite conclusion · **CI red, must resolve first** / `new` = add |
| Check | GitHub Checks all green |
| Flows to | green → stage ⑤ |

## Stage ⑤ · Curation finalize (maintainer · human)

| | |
|---|---|
| Who/When | maintainer, on a contribution PR |
| Commands | `python tools/central_curate.py staging/<you>/<date>.jsonl --report report.md` → per the report, **hand-write each accepted candidate as a md file** in the right knowledge/ directory with a final stable id → `python tools/lint.py` |
| Artifacts | `report.md` (per-candidate add/merge/conflict advice) + **the official records in knowledge/** |
| Placement rules (path = scope, CI-enforced) | idea → `knowledge/targets/<slug>/idea_ledger/L###.md`; function-level fact → `targets/<slug>/facts/F###.md`; reusable lesson → `global/heuristics|anti_patterns|validation_pitfalls/` (H/A/V); global trap → `global/bad_plans/B###.md`; subsystem-level → `subsystems/<sub>/` |
| Flows to | knowledge base updated → stage ⑥ (auto detection) and ⑪ (retrieval) |

## Stage ⑥ · Promotion detection (knowledge → skill bridge)

| | |
|---|---|
| Who/When | maintainer, periodically (or as part of nightly) |
| Commands | `python tools/promotion_detector.py --pr-body` |
| Artifact | a promote-candidate PR body: same mechanism verified on ≥2 distinct targets → suggests `skills/technique/<mechanism>/` (**suggest-only, human merges**; the knowledge instances stay in place as evidence) |
| Flows to | if adopted → skill track stages ⑧⑨ for its exam and graduation |

---

# Skill track (your .opencode process skills → team skill library)

## Stage ⑦ · Scaffold promotion

| | |
|---|---|
| Who/When | member, when a process skill is worth sharing |
| Commands | `hmopt promote-skill .opencode/skills/<name> --kind core --hub "$HUB"` (process skills → core; domain needs applies_to + the selector table, skip initially; technique usually comes from stage ⑥) |
| Artifacts | `$HUB/skills/core/<name>/SKILL.md` (L0/experimental placeholder) + the printed graduation checklist (= ⑧⑨) |
| Check | `python tools/lint.py` skill count +1 |

## Stage ⑧ · Write the exam + register + working draft

| | |
|---|---|
| Who/When | member/maintainer, when the skill should reach L1 / be gate-protected (**not required for L0 entry**) |
| Actions | ① create `eval/task_suites/<suite>/suite.yaml` (name/description/pass_threshold) + `cases/*.yaml` (per case: `expected_terms` the points good guidance must mention + `avoid_terms` red flags + weight); ② add `eval_id: eval/task_suites/<suite>` to the SKILL.md frontmatter; ③ write `best_skill.md` (action checklist — engine B's optimization target) |
| Question sourcing | **derive cases from verified knowledge/ records** (best) or LLM-draft + human approval; **never generate from the skill text itself** (circular) |
| Suite sharing | skills with the same "what good guidance looks like" may share one suite (`eval_id` points to it); different focus → separate suite |
| Artifacts | the suite dir + updated SKILL.md + best_skill.md |

## Stage ⑨ · Exam + lint + PR

| | |
|---|---|
| Who/When | member, after ⑧ |
| Commands | `python tools/run_evals.py skills/core/<name> --suite=eval/task_suites/<suite>` → `python tools/lint.py` → open PR |
| Artifact | `skills/core/<name>/scorecards/<name>__<version>.json` (re-running the same version overwrites) |
| Check | `pass_rate ≥ pass_threshold` and lint green |
| Flows to | after merge, engine B (stage ⑩) takes over continuous improvement |
| ⚠️ Gotchas | **manual runs must pass `--suite=`** (otherwise it falls back to the default suite and scores a meaningless 0); the frontmatter `eval_id` is what the automated gates (eval_gate/nightly) use — you need both |

---

# Operations & consumption

## Stage ⑩ · Nightly loop (maintainer)

| | |
|---|---|
| Who/When | maintainer, nightly/weekly |
| Commands | `python tools/nightly.py` (dry-run, prints the 7-step report) → after human review and **only when content actually changed**, `--apply` |
| Artifacts (--apply) | engine-B accepted skill edits + new scorecards; `registry.yaml` version bump; `releases/<ver>.md`; **`.opencode/skill-memory.lock` updated (consumers pin this)**; `eval/scorecards/_dashboard.md` (▲ = improving) |
| Check | all 7 report lines ok; a failed `normalize`/`validate` auto-aborts all writes |

## Stage ⑪ · Consume (the loop returns to ①)

| | |
|---|---|
| Who/When | any member, **before researching a new target** |
| Commands | `hmopt resolve "<path::symbol>" --stage research --run-dir .opencode/state` (auto-discovers the hub when run from the platform repo; add `--hub` elsewhere) |
| Artifacts | terminal: mounted skills + knowledge (with scores/maturity); `retrieval.jsonl` appended under `--run-dir` (the per-call audit of "what the AI saw") |
| Effect | the record you landed in stage ⑤ shows up in **someone else's** list = the loop is closed |

## Appendix · Observability quick reference

| What to check | Command / location |
|---|---|
| Skill health trend | `python tools/dashboard.py` → `eval/scorecards/_dashboard.md` |
| Retrieval self-test (only when retrieval code changes; not a member routine) | `hmopt retrieval-eval --eval-dir "$HUB/eval/retrieval"`, exits 1 below baseline |
| One retrieval call's audit | `<run-dir>/retrieval.jsonl` |
| Format health (knowledge + skills) | `python tools/lint.py` (format only; quality belongs to run_evals) |

## Appendix · Artifact flow table

| Artifact | Produced by | Consumed by | Final home |
|---|---|---|---|
| `memory/idea_ledger` etc. (md) | ① harness | ② sediment-opencode | stays on the member machine (source of truth) |
| `_bundle.jsonl` | ② | ③ you (cp) | disposable (gitignored) |
| `staging/<you>/<date>.jsonl` | ③ | ④ CI · ⑤ curation | cleanable after finalize |
| `knowledge/**/*.md` | ⑤ | ⑥ promotion · ⑪ resolve · ④ dedup baseline | **permanent (append-only; tombstones, never deleted)** |
| `skills/<kind>/<name>/SKILL.md` | ⑦/⑥ | ⑪ resolve (via requires/selectors) · ⑩ engine B | permanent (edited in place, eval-gated) |
| `scorecards/*.json` | ⑨/⑩ | eval_gate regression check · dashboard | accumulates per version |
| `skill-memory.lock` | ⑩ broadcast | consumers pin the hub version | updated each release |
| `retrieval.jsonl` | ⑪ | human audit | local log |
