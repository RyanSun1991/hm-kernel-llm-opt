# Auto Merge Policy

Automation may merge a Team Skill Hub PR only after all conditions hold:

1. The touched asset has three consecutive releases with improved or equal eval
   scorecards and zero rollback incidents.
2. Knowledge PRs have `dedup.py` output with no `conflict` decisions.
3. Skill PRs include a bounded `skill_patch` manifest and scorecard whose
   `metrics.pass_rate` is strictly greater than the baseline unless the change
   is documentation-only.
4. CODEOWNERS approvals are still required for `skills/core`, `schemas`, and
   `policies`; automation can only perform the final merge click.

Rollback resets the trust counter to zero.
