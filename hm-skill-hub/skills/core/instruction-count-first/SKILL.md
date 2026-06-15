---
name: instruction-count-first
kind: core
version: 0.1.0
maturity: L1
optimization_goal: instruction-count
requires: [technique/hoist-loop-invariant]
eval_id: eval/task_suites/core_optimization_suite
owners: ["@core-skill-owners"]
status: experimental
---

# instruction-count-first (engine B / SkillOpt artifact)

Process skill: pick the mechanism that lowers the **primary metric**
(instruction count on the measured hot path) for the pattern in front of you, and
re-measure before accepting.

This `SKILL.md` is the stable loader stub. The optimizable text lives in
`best_skill.md` (the SkillOpt artifact) — that is what `tools/run_evals.py`
scores and `tools/skill_optimizer.py` edits under the eval gate. Seeded
deliberately incomplete so the optimizer→gate→accept loop is demonstrable.

## When to use

Every optimization task that targets instruction count.

## How to use

Load `best_skill.md` and match the target's hot pattern to a mechanism; consult
the resolver-mounted `technique/` skill + target knowledge; re-measure at the
right `compare_level`.
