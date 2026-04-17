@os-opt-manager
Profile:
Target:
Objective:
Auto-Iterate: 1     # Set to N to auto-run N close-loop passes on clean verdicts; 1 = single pass (default). See iterative-optimization skill.

Requirements:
- Use the full staged multi-agent workflow.
- Save all durable artifacts under .opencode/.
- Auto-Iterate > 1 treats prior .opencode/plans/<target>*_plan.md files as LANDED context (not "already-done"), and the researcher must find orthogonal new instruction-count wins each iteration.
