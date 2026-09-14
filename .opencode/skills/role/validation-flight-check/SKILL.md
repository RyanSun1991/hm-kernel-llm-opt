---
name: validation-flight-check
description: Validate a task's frozen claim with the required build, correctness or performance evidence; bind results to the reviewed revision and report limits without substituting metrics or execution environments.
---

# Validation Flight Check

## Validation Ladder

1. Read the task capsule, reviewed revision, required review gates and frozen validation policy. Bind the target repository, allowed execution environment, workload/checks and any baseline or primary metric before executing.
2. Check required plan/code review and perform relevant static sanity, dependency and build checks. Missing required approval or evidence blocks the corresponding claim; the validator does not grant that approval.
3. For **correctness**, execute the frozen functional checks, including baseline reproduction when the policy requires it. Preserve raw outputs, exit codes, revision and environment/workload identity. Do not require instruction counts, device access or A/B measurements for a local correctness policy.
4. For **performance**, follow the frozen primary metric, comparable baseline/candidate protocol, thresholds and noise controls. Preserve actual raw measurements; functional success does not prove a performance win.
5. For a selected **kernel/device recipe**, retain its stock/feature build, relay/device readiness, authorized flash, auto-test and A/B gates. Use the selected scenario's protocol (for example lmbench or instruction count); generic correctness guidance does not waive those requirements or authorize hardware actions.
6. Report pass/fail/inconclusive/blocked or the verdict vocabulary required by the task, with missing evidence and unexecuted checks explicit. Do not substitute a local result for a required device run or a different metric for the frozen one.
7. Return an independent summary and evidence references to the task's review/execution flow. Memory capture or publication follows its separate authorized workflow; validation does not start a legacy singleton or automatically promote a lesson to the Hub.

## Required Outputs

- validation plan bound to the reviewed revision and policy
- references to required plan/code reviews
- executed commands/checks, raw result artifacts and environment identity
- validation summary: claim, expected result, actual result, verdict and limitations; baseline/candidate comparison and delta analysis when required by the selected policy
